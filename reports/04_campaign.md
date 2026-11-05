# 04 — Campaign to 20 tok/s: system model, roofline, idea list (Phase A + B)

Written 2026-11-01, before any Phase C code change. Companion to `03_final.md` (R-log) and
`results.csv`. Rules: `RULES.md` plus the owner's amendments of 2026-11-01 (measured tolerance behind an
env var, ≤ +0.5 % total PPL and ≤ 1e-3 per chunk, wikitext-2 ≤ 9.90; in-memory re-layout of weights
allowed; derived files on disk only as regenerable experiments; no device access from here).

Labels used throughout: **measured on device** (owner's Z Flip 3, app speed test), **measured on host**
(M4 Pro, relative only), **estimated** (and how).

## 0. Vollie's benchmark, quoted from the app (`vollie/lib/assistant.dart`)

What the owner's numbers are. It is not changed by this campaign.

- **Prompt** (`_benchPrompt`, ~120 tokens; 146 tokens of text before the chat template on host):

  > Read these notes, then write one paragraph about why the ocean is salty.
  > Note 1: rivers carry dissolved minerals from rocks into the sea, and evaporation leaves the salt behind.
  > … (Note 2 to Note 6 identical)

  Sent as one user turn through the model's chat template, thinking off, **greedy** (`temp: 0`).
- **Reply length**: `_benchTokens = 32` tokens per timed run ("enough for a steady rate, short enough to
  keep the phone from heating up across ~7 runs"). The profiled replies use 64 tokens, the A/B reply 72.
- **Decode tok/s** (`decodeTokensPerSec`): `(completionTokens − 1) × 1000 / (duration − timeToFirstToken)`
  in ms; null under 8 tokens. **Prefill tok/s**: new prompt tokens / time to first token.
- **Heat**: before each timed run `_coolDown()` waits up to 90 s until Android's thermal headroom
  forecast is **below 0.65**, and records the headroom the run started at. Runs are back to back.
- **Context**: the model loads with `contextSize` = 4096 on this phone (8 GB with < 3.5 GB available →
  4096), q8_0 K and V cache, flash attention on, mmap, dense weights repacked (`useExtraBuffers` is
  true: 8 GB − 0.7 GB copy ≥ 5 GB). Threads are pinned to the big cores (`pinInferenceThreads`).
- **Selection** (`optimize()`): sweeps `GGML_TINY_OP` ∈ {0, 65536, 262144} then `GGML_EXPERT_IO` ∈
  {0, 1, 5, 13} at the last pick's thread count, then thread counts ≤ big cores, GPU last; a sweep run
  scores by `secondsPerTurn = 500/prefill + 200/decode`; the final pick keeps each thread count's best
  run. Then two profiled 64-token replies ("Write a short story about a lighthouse.") and one Q3_K A/B.
- **Effective tok/s for this campaign** = exactly that decode number. A technique that emits several
  tokens per model evaluation counts through `completionTokens` and `duration` like any other.

## 1. One decoded token on the phone today (Phase A)

Model (frozen D6-e6, `tools/cost.py`, machine-independent): 40 layers = 36 Mamba-2 + 4 attention; 64
experts of (gate 512×1536, up 512×1536, down 1536×512) in Q3_K, 6 used per token; shared expert 1024
wide in Q4_0; `ssm_in` 1536×6448 and `ssm_out` 3072×1536 in Q4_0; tied Q6_K embedding used as lm_head.

| bytes touched per token | MB | how read |
|---|---|---|
| dense Q4_0 (ssm_in 200.6, ssm_out 95.6, shared experts 106.7, attention 14.2, conv/norms ~3) | 420 | repacked copy in app memory (anon, 0.7 GB) |
| router F32 | 16 | mmap |
| lm_head Q6_K (tied `token_embd`) | 126 | mmap |
| experts Q3_K: 6 of 64 × 3 matrices × 40 layers (= 566 M weights) | 243 | mmap, prefetched by `GGML_EXPERT_IO` |
| **weights total** | **805** | |
| Mamba-2 state (36 × 3072×128 f32 + conv) read and written | 58 + 58 | anon |
| KV cache (4 attention layers, q8_0) at ~150 tokens, activations | < 1 | anon |
| **traffic total** | **~921** | |

Where the time goes, from the last device profile (R14: app 2.2.4, 3 threads, `GGML_TINY_OP=65536`,
`GGML_EXPERT_IO=5`, headroom 71-79 %, **measured on device**). The profiled reply ran at 98.2 ms/token;
the best timed run at the same settings was 82 ms (12.2 tok/s), so the profile is ~20 % slow (heat,
noise); shares are what matter.

| phase | compute ms | wait ms | MB | GB/s-equivalent | note |
|---|---|---|---|---|---|
| experts (`MUL_MAT_ID`, 120 ops) | 34.2 | 16.9 | 243 | 7.1 | compute-bound in the Q3_K dot (scales 34 → 24 ms from 3 to 4 threads, R13/R14); the wait is imbalance + in-flight reads |
| dense + lm_head (`MUL_MAT`, 265 ops) | 28.5 | 3.4 | 562 | 19.7 (22.7 at 4 threads) | at this SoC's achievable CPU bandwidth |
| SSM scan + conv, state copies, norms, adds, glu, argsort, get_rows… (~1,580 small ops) | ~12 | ~2.7 | ~120 | — | mostly on thread 0 alone (tiny-op), 818 barriers/token remain (+120 inside `MUL_MAT_ID`) |
| **total** | **74.8** | **23.0** | **~921** | | **98.2 ms/token** profiled, 82 ms best run |

Flash: with expert prefetch on, 5-28 major faults/token (R12-R14) but ~870 minor faults/token. If the
kernel maps 16 pages per minor fault (fault-around), that is up to ~55 MB/token still arriving from UFS
each token behind the prefetch — consistent with Vollie's 2.07 GB of resident file pages against 2.6 GB
of experts + 0.13 GB lm_head. **Unknown; first phone measurement** (see §3).

Host picture of the same token for scale (**measured on host**, M4 Pro, 3 threads, same build and knobs):
18.9 ms/token = experts 9.6 + dense 5.0 + SSM/other 4.3, wait 0.7. The host's experts run at 25 GB/s-
equivalent and its dense at 112 GB/s: the M4 is compute-bound on experts and nowhere near memory-bound,
which is why speed effects of I/O and syncs cannot be seen there (Rules §3).

## 2. Roofline and the binding constraint

Phone constants used (SD888, 1× X1 2.84 GHz + 3× A78 2.42 GHz + 4× A55; LPDDR5 51.2 GB/s theoretical).
The CPU-achievable bandwidth is taken from the dense mat-muls themselves: 19.7 GB/s at 3 threads,
22.7 GB/s at 4 (measured on device, R13/R14). Peak int8 dot-product throughput of the 3 compute cores
≈ 0.6 TFLOPS (X1 four 128-bit pipes, A78 two, 16 MACs each per cycle) — never reachable from a
quantized GEMV, quoted only to show compute is not the limit in principle.

| resource | bytes or work per token | rate | time if it were the only limit | tok/s ceiling |
|---|---|---|---|---|
| DRAM, everything at the dense rate (3 threads) | 921 MB | 19.7 GB/s | 46.8 ms | **21.4** |
| DRAM, 4 threads' rate | 921 MB | 22.7 GB/s | 40.6 ms | **24.6** |
| compute, 2.93 GFLOP | 2.93 GFLOP | 0.6 TFLOPS peak / ~25 % realistic | 4.8 / 19 ms | 200 / 52 |
| flash, if all experts were re-read (R9-R10 state) | 243 MB | ~1.5-2 GB/s UFS | 120-160 ms | 6-8 |
| flash, today's estimate (fault-around reads behind prefetch) | ~55 MB | ~2 GB/s, overlapped | ~27 ms hidden if fully overlapped | — |
| syncs, 938 barriers at 3 threads | 938 | 5-20 µs each (unknown on SD888) | 5-19 ms | — |

All of the above are **estimated** from device-measured rates and model bytes.

**Binding constraint at 12.2 tok/s (82 ms):** none of the four alone; it is a sum. Taking the 98.2 ms
profile's shares: 35 % expert *compute* (a 3-bit kernel at 7 GB/s-equivalent), 23 % waiting at
barriers (imbalance and in-flight reads), 29 % dense bytes (already at bandwidth), 13 % small ops.

**Ceiling if each constraint were removed alone** (from the 98.2 ms profile; the best run would be
~0.84× of each time):

| remove | ms/token (profile scale) | tok/s | at best-run scale |
|---|---|---|---|
| all waits → 0 | 75.2 | 13.3 | 15.9 |
| experts at the dense bandwidth rate (12.3 ms) | 59.4 | 16.8 | 20.2 |
| dense bytes free | 69.7 | 14.3 | 17.2 |
| waits → 0 **and** experts at bandwidth | 53.3 | 18.8 | 22.5 |

**Verdict on raw decode:** 20 tok/s = 50 ms/token sits *at* the DRAM roofline for this file
(46.8 ms at the measured 3-thread rate). It would need the Q3_K expert path to run at the dense weights'
bandwidth rate (today 2.8× slower per byte), every barrier wait gone, and the small ops free, all at
once. Each is partly attackable (§4), none fully. **Realistic raw-decode ceiling: 15-17 tok/s
(estimated).** Raw 20 is not reachable within the rules; the owner's amended reading of the goal
(effective tok/s) is the only route, and §4 D shows the one candidate with the numbers to make it.

What would make raw 20 possible: a file with ~25 % fewer bytes per token (e.g. a 2.5-bit expert format or
a smaller lm_head) — forbidden (file frozen) and anyway failing the gate in D4/D5/D7; or a chip with
~30 GB/s of CPU bandwidth and i8mm — not this phone.

## 3. Top 3 unknowns → first phone-round measurements

1. **How many bytes per token still come from flash, and does I/O wait hide inside "compute"?**
   Decides whether kernel work (ideas C) pays or just exposes I/O. Measure `/proc/self/io` `read_bytes`
   per token next to major/minor faults, with expert I/O 0 / 1 / 5, and the resident file pages
   (`RssFile`) right after load vs after 64 tokens (tells whether the dense tensors' never-again-read
   source pages still occupy cache, idea A2).
2. **What does one barrier cost on this SoC, and how much of the 23 ms wait is imbalance?**
   Micro-benchmark in the kit: 10,000 empty barriers at 2/3/4 pinned threads; and the profile split
   per thread (busy µs per thread in `MUL_MAT_ID`). Decides between fewer barriers (B1-B3), finer
   chunks (B4), and core-weighted splits (B6).
3. **Does a 4th compute thread on the big cores win once the helper moves to an A55?**
   Experts were 24 ms at 4 threads vs 34 at 3 (R13/R14), but 4-thread writing has lost on every warm
   run since R6, before the helper existed. Measure 3t vs 4t with the helper pinned to cpu0-3, best of 3,
   same heat, with core clocks (`cpuMhz`) per run to separate throttling from contention.

Also in round 1 (needed only by idea D1): **batched-verify cost on the device**: time of one model
evaluation of 1, 2, 3, 4 tokens (ubatch) with the real routing. Machine-specific; the host cannot
substitute (its experts are compute-bound in a different ratio).

## 4. Candidate ideas (Phase B)

Gains are ms/token at the 98.2 ms profile scale (3 threads) unless noted; today's best run is 82 ms.
"Rules" lists what could be violated. "Phone" = needs device data to judge. Ideas marked ★ are from
outside the owner's idea map; ✦ is deliberately strange. The ones marked **ask** need the owner first.

### A. Flash I/O and page cache

| # | idea | expected gain | cheapest falsifier | rules | phone |
|---|---|---|---|---|---|
| A1 | **Measure flash bytes/token** (`/proc/self/io`) — not a speedup, the unknown that ranks A2-A5 | — | kit round 1 | — | yes |
| A2 ★ | **Give the dense tensors' source pages back**: after repack the 420 MB of Q4_0 source pages are never read again but stay mapped (the loader only unmaps before/after the used range, `llama-model-loader.cpp:1846`); `madvise(MADV_DONTNEED)` (or `unmap_fragment`) on them at load → up to 420 MB more cache for experts | 0 (kernel already evicted them) to −10 ms (fewer expert refetches) | `RssFile` after load on device; host: `vmmap`/`RssFile` before/after | none; load-time only | yes |
| A3 | **App-managed expert cache**: `pread` the routed slices into an anon ring sized to what is free, LRU per expert (PowerInfer-2 style), instead of mmap faults | −0 to −17 ms (the whole remaining expert wait) | A1 first; host emulation: run with a tiny cgroup-like limit is not available on macOS → the owner's phone only | memory: anon buffer risks the low-memory killer (SwapFree = 0) | yes |
| A4 | **Prefetch the previous token's experts too** for the same layer (45-50 % of a token's experts were used by the previous token, measured on host today) as a complement to the next-layer guess | −0 to −4 ms | host coverage already known; device wait split | none | yes |
| A5 | **Two-layer lead**: apply layer L+2's router to layer L's input (accuracy will drop from 85 %) to give the flash more time | −0 to −5 ms | host hit rate of the L+2 guess (one afternoon) | none | yes |
| A6 | Hot-expert pinning: copy each layer's 8 most-used experts (≈ 325 MB) into anon memory; the 16 most-used cover 70-80 % of uses (host) | −0 to −8 ms | A1; host: coverage per text | memory as A3; derived "hot list" file = experiment only, regenerable | yes |

### B. Synchronisation and threading

| # | idea | expected gain | cheapest falsifier | rules | phone |
|---|---|---|---|---|---|
| B1 | **Barrier micro-cost + per-thread busy split** (measurement) | — | kit round 1 | — | yes |
| B2 | **Drop the barrier inside `MUL_MAT_ID`** (120/token): every thread builds the expert→row map itself (6 ids, trivial) instead of thread 0 then a barrier | −1 to −3 ms | host: barriers/token 938 → 818, bit-identical | none; bit-exact | yes for the gain |
| B3 | **Fuse gate and up expert mat-muls** into one pass per expert (same ids, same input): −40 barriers, one activation quantisation, better cache locality | −1 to −3 ms | host A/B (`GGML_CPU_AB`) | bit-exact if sums are per row | partly |
| B4 | **Finer dynamic chunks in `MUL_MAT_ID`**: today 512 rows × 1 token are split statically into 3 chunks (one per thread), so one slow core stalls the other two; 8-16 dynamic chunks let fast cores take more | −3 to −8 ms of the 16.9 ms expert wait | B1's imbalance number; device A/B | none; bit-exact | yes |
| B5 | **Helper thread on an A55** (affinity) → frees the 4th big core → 4 compute threads | −10 ms (experts 34 → 24) if 4t does not collapse; else 0 | round 1 unknown #3 | none | yes |
| B6 | **Capacity-weighted static splits** (X1 gets ~1.5× an A78's rows) where chunks are static | −1 to −4 ms | B1 | none; bit-exact | yes |
| B7 | **Cheaper barrier** for this SoC: `wfe`-based wait or backoff instead of a tight `yield` spin across the X1/A78 clusters | −2 to −8 ms if a barrier costs ≥ 10 µs here | B1 | none | yes |
| B8 ★ | **Idle threads prefetch**: while thread 0 runs a tiny op, the others `prfm` the next big op's weight rows into L2 instead of spinning | −0 to −3 ms (dense at 19.7 → ~21 GB/s) | host shows nothing (bandwidth is not the host's limit) → device A/B | none | yes |
| B9 | More graph fusion for the CPU backend (residual scale + add; `MUL(silu, up)` into the down mat-mul's quantisation) → fewer nodes/barriers | −2 to −5 ms | host barrier count; device | bit-exact if the same float ops in the same order | yes |

### C. Kernels (the Q3_K expert path is 35 % of the token)

| # | idea | expected gain | cheapest falsifier | rules | phone |
|---|---|---|---|---|---|
| C1 | **Q3_K multi-row GEMV**: 4 rows per call from 4 row pointers (no re-layout), sharing the 8 q8 loads per 128 weights and giving the pipes independent chains | −10 to −20 % of 34 ms = −3 to −7 ms | host in-generation A/B on `MUL_MAT_ID` µs | bit-exact (per-row integer sums unchanged) | A/B on device to confirm |
| C2 | **Q3_K unpack trick**: `vsli` to insert the high bit, or fold the −4 offset into Q8_K `bsums` → ~1 fewer op per 16 weights | −1 to −3 ms | host A/B | bit-exact (integer) | A/B |
| C3 | **Multi-column Q3_K** (unpack once, dot with k activation columns) | 0 for raw decode; **−40 % of expert compute per token when verifying batches** (D1) | host: time `-ub 4` vs 4 × `-ub 1` | bit-exact | with D1 |
| C4 | lm_head (126 MB Q6_K, ~6 ms): confirm it is bandwidth-bound on the A78; if not, same treatment as C1 | −0 to −2 ms | device profile per op (it is inside `MUL_MAT`) | — | yes |
| C5 | **Transparent huge pages** (`MADV_HUGEPAGE`) on the 0.7 GB repacked copy → fewer TLB misses on the dense stream | −0 to −1.5 ms; 0 if the kernel has THP off | device: `AnonHugePages` in `/proc/self/smaps_rollup` | none | yes |
| C6 ★ | **ADPF performance hint** (`PerformanceHintManager`, Android 12+) from the app around each token: tells the governor the work deadline so clocks stay up | −0 to −8 ms on a warm phone; 0 if Samsung ignores it | device: `cpuMhz` per run with/without | app-runtime tool (allowed by the owner's note) | yes |

### D. Algorithmic — effective tok/s

| # | idea | expected gain | cheapest falsifier | rules | phone |
|---|---|---|---|---|---|
| D1 **ask** | **Draft-model speculation with batched verify**: Granite-4.0-350m (Q4_0 228 MB, same tokenizer, `lab/models/`) drafts 3-4 tokens; the frozen model verifies them in one evaluation. Output is identical under greedy. Host acceptance measured earlier (R-log §3, `lab/logs/g-d350*.log`): 0.26-0.73, mean accepted length 2.0-3.9 per draft. Cost model for the phone (**estimated**): dense bytes read once per batch (~28 ms for k ≤ 4), experts = union of the tokens' experts (host today: k=3 → 11.7 distinct per layer, k=4 → 13.8, i.e. 0.58-0.66 of k × 6) at the Q3_K kernel rate, so ~55-78 ms with C3, draft 4 × ~12 ms, state rollback ~5 ms → ~110-140 ms per ~3 accepted tokens → **17-21 tok/s effective**, 15-18 without C3 | the only candidate that reaches 20 on paper | round 1: batched-verify time for 1-4 tokens on the device; host: acceptance on the official prompt and on story/code/list text | **another GGUF** (owner must approve), +228 MB on a phone with SwapFree 0; speculation changes nothing in output under greedy | yes |
| D2 | **Prompt-lookup / n-gram drafting** | **falsified today (host, machine-independent):** 1.20-1.27 tokens per evaluation on the official prompt, 1.01-1.17 on story/code/list → at most +20 % before verify costs | — | — | no |
| D3 | **Layer-skip self-draft** (exit after ~20 of 40 layers): Granite has no trained early exit | likely < 30 % acceptance → ≤ +10 %; low | host acceptance with a layer-count override (half a day) | none | no |
| D4 | Draft = same model with top-1 expert: still reads the 546 MB dense + lm_head → a draft costs ≥ 60 % of a token | ≤ +15 %, likely negative | arithmetic above | none | no |
| D5 ✦ | **Pipelined speculation on the A55 cluster**: the 350m draft runs on the four A55s while the big cores verify the previous batch, so drafting is free wall-clock; the A55s' bandwidth share and the Mamba state copy decide it | turns D1's 110-140 ms into ~95-125 ms → 19-24 tok/s **estimated** | D1 first; then A55-pinned draft timing in the kit | as D1 | yes |

### E. Numerics within the owner's tolerance (**ask** before building; listed to be complete)

| # | idea | expected gain | falsifier | rules |
|---|---|---|---|---|
| E1 ✦ | **Skip the lowest-weighted expert when its softmax weight is below a threshold** (effective k = 5 on some tokens): −1/6 of expert compute on those tokens | −2 to −6 ms | host PPL vs threshold; at k=5 for all tokens the old file lost 2 PPL points, so the threshold must be tight | changes results → tolerance rule; likely beyond ±0.5 % → **ask with numbers first** |
| E2 ✦ | **Candidate-set lm_head**: compute logits only for the ~2k tokens a cheap predictor (previous layer's state, or last-token bigram table) proposes; greedy output identical whenever the true argmax is in the set | −3 to −5 ms | host: argmax-in-set rate on four texts | per-chunk PPL cannot be computed the same way → needs a different quality check → **ask** |

### F. Rejected with numbers

- **Adreno GPU as co-processor for the dense stream**: shares the same DRAM the CPU already saturates;
  OpenCL numerics are not bit-exact; whole-model GPU runs measured 0.7-2.9 tok/s (device). No.
- **Hexagon**: v68 on the SD888, no supported path in this llama.cpp (Phase 2). No.
- **More threads than big cores**: measured minutes-long runs (R4). No; the A55s only as asynchronous
  helpers (B5, D5).
- **In-memory requantisation of lm_head to Q4** (D2/D4 showed +10.3 % / +9.5 % PPL): over the 9.90
  margin. No.

### Assumptions in the R-log that are now questioned

- "Speculation loses on this MoE" (R-log §3, §6) was **measured on host**, where the big model is
  compute-bound and a draft buys nothing. On the phone the dense 546 MB are bandwidth-bound and shared
  across a verify batch; the economics are different and must be measured on the device (D1).
- "Experts are compute-bound" (R14) rests on 24 vs 34 ms at 4 vs 3 threads from two different runs
  on a warm phone (±20 % noise). A1 and B1 test it directly.
- "4 threads collapse on a warm phone" has no cause attached; unknown #3 separates throttling (clocks)
  from contention (the helper, Android threads).

## 5. Host facts measured today (R16)

- **Expert overlap across consecutive tokens** (`GGML_EXPERT_LOG=1`, lab-only knob; D6-e6, greedy, 4
  texts: the official prompt (84 tokens), lighthouse story, Python code, Mars list (199 each);
  `lab/expert_overlap.py`): distinct experts per layer over k consecutive tokens = **9.0-9.3 (k=2),
  11.5-11.9 (k=3), 13.5-14.1 (k=4), 16.9-18.0 (k=6)** vs 6k if independent; 45-50 % of a token's experts
  were also used by the previous token; each layer's 16 most-used experts cover 70-80 % of uses.
- **Prompt-lookup acceptance** (`lab/ngram_sim.py`, deterministic on the greedy token sequence):
  tokens per model evaluation with 2-gram/4-token drafts **1.27 official prompt, 1.02 story, 1.17 code,
  1.03 list**; 3-gram 1.20 / 1.01 / 1.12 / 1.01. D2 is dropped.

## 6. Proposed order for Phase C (for the owner's go-ahead)

Round 1 is a **measurement round** with two cheap, bit-exact changes riding along, so the owner's first
trip answers the unknowns and tests something:

1. Kit: flash bytes/token, barrier micro-cost, per-thread busy split, RssFile after load, batched-verify
   time for 1-4 tokens, 3t vs 4t with the helper on an A55, THP/ADPF availability — one result file.
2. Riding along (off by default, env-gated, bit-identical at `-ub 1/32/512`): B2 (no barrier inside
   `MUL_MAT_ID`), B4 (dynamic expert chunks), C1 (multi-row Q3_K), B5 (helper affinity), A2 (release
   dense source pages).
3. Pre-registered predictions per variant go into `03_final.md` before the build is frozen.

Round 2 then depends on the owner's answer on D1 (second GGUF) and on round 1's numbers.

## 7. Round 1 answers (R18, 2026-11-02; details in `03_final.md` R18)

The three unknowns of §3, answered or not:

1. **Flash bytes per token:** *not answered* — `/proc/self/io` is not readable from the app on this phone. Major faults ran 4-10 per token (one 270/token eviction storm). Round 2 tries `getrusage(RUSAGE_SELF).ru_inblock`.
2. **Barrier cost vs imbalance:** **answered** — 0.17 / 0.30 / 2.0 µs per barrier at 2 / 3 / 4 threads, i.e. 0.25 ms of the ~20 ms wait. The wait is imbalance. Dynamic 32-row chunks in `MUL_MAT_ID` (B4) cut its wait 16.5 → 6.0 ms on the device.
3. **4th thread with the helper on an A55:** **answered, negative** — 4t loses by 8-15 % with the helper gone and at equal clocks (2112 / 2035 MHz): Android's own threads take the 4th big core. And the helper on an A55 costs 13 % by itself (the guess arrives too late).

Batched-verify cost (D1's input): *not measured* — every draft run failed on the first rejected draft because the phone's prebuilt `libllama.so` clamps `n_rs_seq` to 0 for Granite-H (the one-line arch-list fix exists only in the host build).

Idea status after round 1:

| idea | round-1 result | status |
|---|---|---|
| B2 no barrier in `MUL_MAT_ID` | −2.8 ms, but +2.9 ms of duplicated quantization | superseded by B4 |
| **B4 dynamic expert chunks** | **−7.5 ms** (`MUL_MAT_ID` wait 16.5 → 6.0) | **default in the next build** |
| **C1 multi-row Q3_K** | **−4.9 ms** end-to-end, compute itself only −2.4 % | **default**; the expert path is load/stall-bound on the phone, not arithmetic-bound |
| B4 + C1 | **−8.4 ms (−8.3 %)**; ≈ 13.3 tok/s at R13 scale (estimated) | ship |
| tiny-op 262144 | −4.3 ms in one A/B, against two separate-run sweeps saying the opposite | repeat ×3 |
| E1 expert skip | −0.7 ms where PPL is safe; −9.1 ms only at +0.82 % wiki / +2.1 % en | dropped as a default |
| B5 helper on an A55 | −13 % | dead |
| B7 cheaper barrier | barrier is 0.3 µs | dead |
| 4 compute threads | loses at equal clocks without the helper | dead |
| A2 release repack source | within noise, no comparable `file=` row | inconclusive, low priority |
| D1 draft-model speculation | build bug; draft costs 15 ms/token on the big cores, +265 MB anon costs the target nothing | **round 2, with a rebuilt `libllama.so`** |
| A1 flash bytes/token | `/proc/self/io` denied | round 2 via `getrusage` |

Remaining raw-decode gap with B4 + C1 on: 93 ms/token = `MUL_MAT_ID` 42.5 + 8.2 wait, `MUL_MAT` 26.0 + 2.2, `SSM_SCAN` 3.8, `CPY` 3.6, other ≈ 6, against a 46.8 ms roofline. The experts still move bytes ~4× slower than the dense stream (≈ 5.7 vs 21.5 GB/s-equivalent); that ~30 ms is the gap, and it is I/O-shaped, not kernel-shaped. Raw 20 stays out of reach on the CPU path; effective tok/s (D1) is the route, and round 1 showed the phone can afford the draft's memory.

**Round 2, proposed order (for the owner's go-ahead):** (1) defaults `GGML_MMID=3`, `GGML_Q3K_X4=1`; (2) patched `libllama.so` (arch list) built in the `tool/build_cpu.sh` tree and handed to llamadart as a local bundle (`LLAMADART_ALLOW_LEGACY_LOCAL_BUNDLES=1`, `third_party/bin/android-arm64/`), kit r2 spec phase with a forced-rejection warm-up, batched-verify time for 1-4 tokens, effective tok/s at n = 2 / 4 on the official and story prompts; (3) A/B repeats: tiny-op 262144 ×3, 16-row chunks (B4 finer), B3 fused gate/up; (4) `getrusage` for flash bytes.

**Round 2 build (2026-11-02, app 2.2.7+30, kit r2; `03_final.md` R19):** items (1) and (2) of the proposed order are built — new defaults, rebuilt `libllama.so` as a llamadart local bundle, fail-fast in llamadart, kit r2 with a forced-rejection warm-up and `GGML_CPU_PROF=2` for the verify-batch cost; tiny-op ×3 rides along. Items (3) finer chunks / fused gate-up and (4) `getrusage` wait for round 3. Predictions are pre-registered in `03_final.md` R19.

## 8. Round 2 answers (R20, 2026-11-02 pm; details in `03_final.md` R20)

- **B4 + C1 as defaults:** confirmed, −10.0 % per token by A/B (R18: −8.3 %).
- **Tiny-op 262,144:** −8 to −12.5 % in three A/Bs (the expert mat-mul's own compute drops 7-8 ms, unexplained) → default; round 3 tries larger thresholds.
- **D1 draft-model speculation: falsified on this CPU.** The rollback fix works (warm-up 36 drafted / 14 accepted, no error) and acceptance meets the host numbers (2.5-2.9 tokens per evaluation at n = 2-4 on the official prompt, 1.8-2.1 on the story), but one verification of 2 / 3 / 4 / 5 tokens costs **2.3 / 2.9 / 3.1 / 3.4×** a single token (official) and 2.5 / 3.9× (story), the draft 14-24 ms per token, and the slots + draft (+530 MB) alone cut the plain rate 8.0 → 6.1. Effective rates **0.48-0.76× the no-draft rate at every setting**. The §2 cost model assumed a bandwidth-bound decoder; the phone's dense mat-mul is at the bandwidth/compute ridge at one token and goes linear with rows (25.6 → 47-55 ms per graph at 5 tokens), and the experts are Q3_K compute plus un-prefetched page faults. **D5** (A55 drafting) could hide ≤ 61 of 557 ms per step; **C3** saves unpacking, not multiply-adds. Both dropped.
- **Goal:** with the output-identical effective route closed, nothing in the idea list reaches 20 tok/s on the SD888 CPU within the rules. Raw: 13-15 estimated with the three defaults at R13's memory state; the remaining levers (≈ 7 ms of `MUL_MAT_ID` wait, larger tiny-op, A3 expert cache on pressured days) aim at the 15-17 ceiling of §2. The GPU was 0.7-2.9 tok/s; llama.cpp's Hexagon backend targets newer DSPs than the SD888's. **The decision on the goal is the owner's**: keep 20 and relax a rule (the frozen file — fewer experts per token is the lever with the arithmetic — or the device), or restate the target as the raw ceiling.
- **Round 3 (if raw):** tiny-op 524,288 / 1,048,576 A/B; finer `MUL_MAT_ID` chunks and fused gate/up for the last ~7 ms of wait; `getrusage` flash bytes; kit fixes (pin untested knobs; `PROF=2` to count tokens from the embedding node and ignore graphs without `MUL_MAT_ID`).

## 9. Goal restated by the owner (2026-11-02, after R20) and the research reset

The owner's words, recorded verbatim in `RULES.md` §1 and `CLAUDE.md` §1: build a system that challenges
the hardware's limit; research a new way to **print** ≥ 20 tok/s under the rules (under the rules,
everything is allowed); the point is **not** to compute 20 tokens per second — not every printed token has
to be fully calculated; the method must generalize to other models and devices.

What R16-R20 fixed as fact: on this CPU, output per second is bounded by multiply-adds per printed token
(compute-bound at any batch > 1), the barrier and I/O stories are largely told, and bit-identical runtime
work has ≈ 13-15 tok/s raw in it (estimated) with a 15-17 ceiling. So the research question becomes:
**which tokens, layers, experts or operations can be left uncomputed — or computed once and reused — with a
quality cost inside the gate (+0.5 % working tolerance, +10 % hard), by a rule that does not depend on
this model or this phone?** Candidate families to rank with host-measurable, machine-independent numbers
first (Bonnie, memo `reports/05_compute_less.md`): per-token adaptive depth (early exit / layer skipping
by a confidence rule), per-token adaptive expert count (the router's own weights as the signal), reuse
across tokens (states or expert outputs of repeated contexts), cheap-token paths (tokens a tiny predictor
gets right at high confidence, verified lazily or not at all when the rule allows), and anything that
changes the arithmetic-per-token rather than the arithmetic's speed. Closed and not to be re-proposed
without new arithmetic: batched speculation (D1/D5/C3), n-gram drafting, 4 threads, A55 helper, expert
skip at a safe threshold, barrier latency.

Team standard from here on (`CLAUDE.md` §2): Clyde leads, Bonnie researches with Clyde, Tom codes,
Annie keeps git and the books.

## 10. Round 3 answers (R22-R23, 2026-11-02 evening; details `03_final.md` R22-R23, memo `05_compute_less.md` §5-6)

Budget set by the owner the same day: hard limit +20 % vs F16, speed rule (router-weight expert threshold) +2 %
over stock 9.6051 full set (≤ 9.797).

- **Which expert-budget rule (R22, host):** the absolute weight threshold beats cumulative-mass and gap rules
  2-3× in PPL per expert saved. One number per model. **Placed on the full set at 115 permille: PPL 9.7771
  (+1.79 %), k̄ 4.11** — inside the budget.
- **The expert kernel (R23, device):** the Q3_K x4 dot with independent accumulator chains (`GGML_Q3K_ILP`,
  bit-identical) cuts `MUL_MAT_ID` compute −13 to −18 % on the A78 (−9 to −14 ms/token in fault-heavy runs,
  −4 to −6 ms expected clean) while the M4 host showed +1.3 %: the A78 stalls on dependency-chain latency
  at 2 NEON pipes, the M4's wider core already hides it. **Default from round 4.** Bonnie's reading was right,
  Clyde's memory reading wrong: an anon-memory cache of the 16 hottest experts per layer reached 75-78 % hit
  as predicted and saved ≈ 1 ms, then its 618 MB evicted the file cache and cost 20 %. Prefetch: nothing.
  **Memory and prefetch terms of the expert path: closed.**
- **The skip rule on the device:** always-on at 115, `MUL_MAT_ID` −27 % (proportional to the 27 % of slots
  skipped), ≈ −7.5 to −8 ms/token. (A profiling bug — the per-run window carried over graphs from the previous
  run — made the A/B look weak; found, fixed, host-reproduced.)
- **Where it leaves 20 (Bonnie §6.6):** exact levers + the +2 % skip sum to −12 to −18 ms of the −15 needed →
  **17-18.5 tok/s story-scale, 19.5 in the top corner**; the official 32-token bench reads the page cache more
  than the runtime on a swapping day (5.4 / 11.2 in r3). **20 at the centre of the band needs SKIP 130
  (≈ +4 % PPL est.) → 18.5-19.6 or SKIP 140 (≈ +6 % est.) → ≈ 20.0** — inside the hard limit, beyond the +2 %
  budget: the owner's decision, with round-4 device A/Bs for 130/140 and full-set PPL for both.
- **Round 4 (kit r4, pre-registered in `03_final.md` R24):** ILP default; A/Bs ILP, skip 115 ×2, 130, 140;
  always-on ILP + skip 115 story 200 (predicted 54-58 ms → 17-18.5 tok/s) and official 32; SSM state
  round-trip probe (−1.5 to −2.5 ms, bit-exact) if Tom gets it identical in time.

## 11. Round 4 answers (R24, 2026-11-03; details `03_final.md` "R24 — round-4 device results", JSON `reports/device/r4_2026-11-03.json`)

Swapping day again (swapfree 0-351 MB, 18-1,390 majflt per token); official baselines 7.1 / 5.3 tok/s.

- **The skip rule, three thresholds on the device:** `MUL_MAT_ID` falls by 0.87 × the skipped share at every
  threshold — 115: −25 to −32 % at share 27 % (k̄ 4.33); 130: −34 % at 39 % (k̄ 3.61); 140: −40 % at 46 %
  (k̄ 3.20). Always-on 115 over a 200-token story: **14.2 tok/s printed vs 10.2 control (×1.40; withdrawn in R25, see §12)**, steady
  profile 60.8 ms (16.4 tok/s) vs 79-86. Official 32 with skip: 11.3 vs 7.1 / 5.3. The 54-58 ms band was
  missed narrowly (60-62) on a day that fails its clean precondition.
- **SSM state round trip (`GGML_SSM_INPLACE`, bit-exact): −4.5 ms** in its own ops (`CPY` −3.3, `SSM_SCAN`
  −1.2, 72 fewer barriers), twice the predicted band. **Default on from round 5.**
- **ILP corrected:** the r4 A/B ran ILP first and showed "off" 2.6 ms faster. Solving r3 (`0:1`) and r4
  (`1:0`) together separates the knob from the order: **ILP ≈ −6 % of `MUL_MAT_ID` (≈ −2 ms), window-2
  advantage ≈ 10 %**. R23's −18 % was two-thirds order bias. ILP stays default (exact) but is no longer in
  the 20 tok/s sum.
- **Method finding (generalizes):** side A of the kit's A/B is tokens 1-32, side B 33-64; window 1 of every
  run is 25-60 % slower (page-cache fill + DVFS from 710 MHz). Any one-order on-device A/B is biased by
  about that; round 5 alternates sides per 8-token window after a warm-up and scores steady windows only.
- **Where 20 stands (steady profile, estimated from the measured pieces):** 115 + SSM ≈ 56 ms (17.8); 130 +
  SSM ≈ 53 (19); 140 + SSM ≈ 50.5 (19.8). Printed over 200 tokens is ≈ 2 tok/s under steady profile on a
  swapping day. **Owner's threshold decision still open: 115 (+1.79 %, approved) / 130 (+3.67 %) / 140
  (+5.25 %).** (Decided the same evening: the +2 % budget is final; edge found at 118 = +1.98 %, R25.) Exact levers left: waits −0.8 to −1.5 ms.

## 12. Round 5 answers (R25, 2026-11-04; details `03_final.md` "R25 — round-5 device results", JSON `reports/device/r5_2026-11-04.json`, pre-registration memo `05_compute_less.md` §7.7)

Swapping day again (swapfree 0-386 MB, 6.5-119 majflt per token on the stories, 309-573 minflt); clocks full
(2,112 / 2,035) at every sample. 8 of 12 pre-registered items hit, ILP at the edge, `CONV_RING` unresolvable at
token level, 2 misses (the printed skip-118 story, the warm-up size).

- **Vanilla vs Vollie, same app, same file, same run (the README table):** official 32-token bench **9.4 →
  12.0 (bit-identical) → 13.9 (skip 118)** tok/s; 200-token story printed **10.2 → 14.5 → 12.5**, steady profile
  **82-97 → 62-66 → 55.4 ms/token**. The exact stack (tiny-op, `MMID 3`, `Q3K_X4`, ILP, SSM in place) is worth
  **−27 ms/token steady (−30 %)**: `MUL_MAT_ID` −21 (−13 of it waits), `CPY` −2.7, `SSM_SCAN` −1.5; the dense
  `MUL_MAT` is untouched (25.9 → 23.1). Ratio 1.42, below the predicted clean band (1.5-1.65), above the
  falsifier (1.35): the stack was over-priced by 5-8 ms, from R20's bad-day tiny-op number. Vanilla landed in
  the clean band on a swapping day: `EXPERT_IO=0` triples the faults (109 vs 34/token) but not the time.
- **The skip rule's own value, measured in-run (ABBA):** 115 **−8.3 ± 2.9 ms**, 118 **−7.6 ± 6.0 ms** per token
  (share 28.5 / 29.8 %, k̄ 4.29 / 4.21, `MUL_MAT_ID` −25 % / −30 % steady = coefficient 0.9-1.0 × share). That is
  **≈ ×1.12-1.15 over a 62 ms control**. R24's "×1.40 printed" was the control's bad page-cache day: today the
  control printed 14.5 and the skip-118 run 12.5 in consecutive runs (34 vs 77 majflt/token). The README drops
  the ×1.40.
- **ABBA settles the A/B method:** ILP, which read −18 % and +3.7 % in the two one-order rounds, reads
  **+1.2 ± 5.2 ms pooled** (≈ −1 to −2 ms real; stays default, out of the sum); `SSM_INPLACE` shows its own
  −4.6 ms to ±0.2 (second confirmation); `CONV_RING` shows **−0.7 ms exact** in its own ops (`CONCAT` → 0,
  `SSM_CONV` −0.15) → round-6 default. One-order A/Bs from r3/r4 are superseded.
- **What "printed" measures on this phone now:** steady at 118 is **55-56 ms = 17.7-18.1 tok/s**, but the
  200-token printed number is set by the page-cache term (window-1 penalty 22-62 ms, 34-109 majflt/token):
  10.2 / 14.5 / 12.5 printed for 91 / 63 / 55 ms steady. The runtime's remaining lever on the printed number is
  bytes read per token, not MACs.
- **Where 20 stands:** the k̄ lever is spent under the +2 % budget (118 = −8); exact crumbs `CONV_RING` −0.7 +
  waits −0.8 to −1.5 + madvise −0.4 to −1.0 → ≈ 53 ms = **18.5-19 steady at best**; 20 printed needs 48.6
  steady and a quiet warm-up. The dense `MUL_MAT` (23-25 ms, 42-45 % of the token at 118, bandwidth ridge) is
  the only block big enough for the last 4-6 ms: bytes, not MACs, must fall. **Round 6:** (1) `CONV_RING`
  default on; (2) Tom checks whether `EXPERT_IO`'s `MADV_WILLNEED` is issued before the skip decision (if so,
  the runtime reads ≈ 30 % of expert rows it never multiplies — an exact cut of bytes/token and of the
  swapping-day printed number); (3) Bonnie's memo on the dense `MUL_MAT` byte term inside the rules and the
  frozen file (round-6 pre-registration); (4) waits and madvise-off-compute-threads after (2). [Item (4) madvise closed by R27, §13.]

## 13. Round 6 answers (R27, 2026-11-05; details `03_final.md` "R27 — round-6 device results", JSON `reports/device/r6_2026-11-05.json`, pre-registration memo `05_compute_less.md` §7.8)

The worst day yet: the four fast cores sat at **1,555 MHz** for every story run (r5: 2,112 / 2,035; heat 0.81-0.93
against the 0.933 threshold) and swap was exhausted after every run. The ABBA items are immune to that (both
sides share the clock) and hit; every absolute band missed by the clock. 8 hits, 3 clock misses, the 1,555 cap
itself, **1 falsified**.

- **Three knobs, three device confirmations each (ABBA, in-run):** `CONV_RING` own ops **+1.25 ms** when off
  (above the +0.6-0.9 band — the conv `CPY` is not free), default on stands; `EXPERT_SKIP` 118 **−10.5 ± 2.5
  ms** pooled (share 29.8 %, k̄ 4.21, coefficient 0.96, `adv` 720 → 505 = the skip-aware read-ahead works as
  built); `SSM_INPLACE` **+3.96** own ops when off. The A/B method is settled; the kit stays ABBA.
- **The madvise lever is dead.** `eio 0` costs **+0.85 ms steady**, −0.27 tok/s printed, and **×9.6 faults**
  (365 vs 38/token); `mmid_thr` spread is < 1 % on *both* sides, so thread 0 was never paying for the advice.
  `WILLNEED` stays on; nothing to move off the compute threads.
- **The one falsified item:** with the advice on, the skip run still faulted **1.55 ×** the control (pre-registered
  0.65-0.8, falsifier ≥ 1.3; start states 820 vs 1,174 MB, inside the ±400 MB rule); with the advice off it
  faulted **0.42 ×**. Per-run majflt with a 354 MB start-state gap cannot separate "the prefetch fix did not
  reduce bytes touched" from "the start state did it"; round 7 puts the fault counters inside the ABBA blocks.
- **A free clock experiment, device-grounded:** 2,112 → 1,555 MHz (×0.736) moved `MUL_MAT_ID` 27-29 → 34.6
  (×1.24) and the skip Δ −8 → −10.5 (×1.3) but the dense `MUL_MAT` only 23.1 → 25.0-25.4 (×1.08-1.10) and
  `SSM_SCAN` not at all. The expert products are compute-bound, the dense products bandwidth-bound — the R24
  ridge is now shown, not inferred. On a hot phone the skip rule is worth more and the dense term weighs less.
- **Vanilla, same run:** official 32 **6.1**, story **8.9** printed / 100.9 steady (waits 20-22 ms, 1,926 barriers)
  vs Vollie control 12.05 / 74.6 → **ratio 1.35** (R25 1.42, band 1.35-1.5). Exact stack −26 ms/token today.
- **Where 20 stands:** steady at 118 **63.7 ms = 15.7 tok/s** at 1,555 MHz, ≈ **54-56 ms = 18** clock-corrected
  (*estimated*); printed 12.7. Round 6 delivered its pre-registered −1 (CONV_RING) and answered its two questions
  (madvise: no; prefetch bytes: unresolved by this instrument). Under the +2 % budget only waits (−0.8 to −1.5)
  are left as exact crumbs; the dense `MUL_MAT` byte term is the only path to 20 inside the rules. **Round 7
  (not built):** (1) per-ABBA-block `getrusage` faults in `cpu-prof` (Tom, exact instrument) + kit r7 with
  `EXPERT_SKIP:0:118` ABBA at eio 1 and eio 0; (2) Bonnie's dense `MUL_MAT` byte memo (`05_compute_less.md` §8 (owed by Bonnie, round 7; not yet written)), including the
  repaired two-stage lm_head screen (126 MB Q6_K ≈ 6 of the 25 ms); (3) waits only if (2) finds nothing bigger.
  Closed: madvise off the compute threads, `EXPERT_IO=0`, steady numbers quoted without their clock.
