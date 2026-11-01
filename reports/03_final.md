# 03 — Final report and research log (living document)

Last updated: **2026-11-02, app 2.2.6+29 (research kit r1, R17 pre-registered)**; campaign plan in `04_campaign.md`. Each new result is appended to §5 and `results.csv`; the
summary in §1 is rewritten to match.

Every speed number is labelled: **measured on device** (the user's Galaxy Z Flip 3, run by the user
through Vollie's Settings speed test), **measured on host** (Apple M4 Pro, CPU only, relative only), or
**estimated**. No speed was measured on a device by the assistant; there is no adb here.

## 1. Status in one table

| | quality (wikitext-2 PPL, gate ≤ 9.948 = +10% vs F16 9.0438) | decode on the phone | how measured |
|---|---|---|---|
| Starting point: shipped file (i1-Q4_0, 4 experts), repack off | 10.1619 (**+12.4%, fails**) | not measured | — |
| Shipped file D1-e5, stock CPU module, 4 threads | 9.6230 (+6.4%, passes) | **7.4 tok/s** writing, 14 reading | measured on device, cool phone |
| Same, stock settings at ~70% thermal headroom | 9.6230 | **3.4** writing, 16 reading | measured on device |
| Current file D6-e6, stock runtime settings (R12 sweep, expert I/O off) | 9.6051 (+6.2%) | **6.1** writing, 17 reading, at 74% | measured on device |
| **Current best: D6-e6 + tiny-op + expert prefetch + next-layer guess, 3 threads** | 9.6051 (bit-identical to stock kernels) | **12.2** writing (3t), 25 reading (4t), at 73-76% | measured on device |
| Goal | ≤ 9.948 | **≥ 20** | — |

**Verdict so far:** quality was brought back inside the gate (it was failing at the start), and the
phone's best writing speed went from 3.4 tok/s (stock settings, warm phone) to **12.2 tok/s** with
runtime changes only. What turned out to limit an 8 GB SD888 was not bandwidth but (1) thread syncs
around ~2,000 small ops per token, then (2) MoE experts that don't fit in the page cache and were
re-read from flash by synchronous page faults. Both are now handled in Vollie's CPU module (small-op
sync skipping, expert prefetch, next-layer expert guessing). From the last profile: dense weights
~25 ms/token (near this SoC's bandwidth), experts ~24 ms compute + remaining read waits, ~10 ms other.
**20 tok/s (50 ms/token) is now the plausible ceiling, not yet reached**: it needs the remaining
expert read waits and sync waits close to zero (estimated).

## 2. Setup that stayed fixed

- **Model (locked in Phase 0, `00_model_choice.md`):** Huihui Granite-4.0-H-Tiny abliterated (IBM
  Granite 4.0 H Tiny: 7B MoE, 64 experts, 36 Mamba2 + 4 attention layers). Rejected: Qwen3.5-4B (B,
  0.909 vs 0.750 bits/byte, 2.4x the bytes/token), Ling-3.0-tiny (C, 0.906 bpb).
- **Quality gate:** `llama-perplexity -c 512 -b 512`, wikitext-2-raw test (all chunks) against the same
  model's F16 with its own 6 experts (9.0438). `tools/ppl.sh`. Sanity set `lab/text/en.txt` (F16 10.3768) is
  reported with every result. *Until 2026-11-01 en.txt was also held to ≤ 10% (my addition); the owner
  then set the gate to wikitext-2 only. Under that rule D4 (+9.5%) and D5 (+7.8%) would have passed; no
  decision is reopened, since the served file is frozen at D6.* Rules: `RULES.md` in the research repo.
- **Host:** M4 Pro, llama.cpp v0.8.0 @ 8642541b and Vollie's 620d2225c, built CPU-only for
  armv8.2-a+dotprod+fp16 (the SD888's ISA: dotprod, fp16, no i8mm/SVE).
- **Phone:** Z Flip 3, SD888 (1x X1 + 3x A78 + 4x A55), 8 GB, Android; Vollie's CPU backend.

## 3. Phases 0-2 in brief

- **Phase 0:** Granite chosen; the shipped 4-expert file found to fail the gate (+12.4%, half of it from
  cutting experts 6 → 4). Details: `00_model_choice.md`.
- **Phase 1:** Vollie's llama.cpp is within 2.6% of v0.8.0 and gives identical PPL; the big difference was
  repack disabled on 8 GB phones (after an OOM). Details: `01_vollie_vs_llamacpp.md`.
- **Phase 2:** 17 methods ranked (`02_methods.md`). Speculative decoding and MTP lose on this MoE
  (measured on host), Hexagon is out (v68), OpenCL is slower than the CPU here (measured on device).

## 4. Phase 3 — model file (quantization) iterations, all PPL measured on host

| variant | wiki PPL | Δ | en.txt Δ | host tg 4 thr | kept? | why |
|---|---|---|---|---|---|---|
| shipped i1-Q4_0, 4 experts | 10.1619 | +12.4% | +17.6% | 80.1 (repack off) | no | fails gate |
| same file, 5 experts | 9.6176 | +6.3% | +7.6% | 75.5 | → basis | gate needs ≥ 5 experts |
| IQ4_NL all, 4 experts | 9.9999 | +10.6% | — | 73.1 | no | fails (just) |
| **D1: Q4_0 dense + IQ4_XS experts + Q6_K head, 5 experts** | **9.6230** | **+6.4%** | **+8.1%** | **92.1** (dense repack) | **shipped** | passes; only the Q4_0 dense part repacks (0.5 GB copy instead of 3.6 GB) → repack fits 8 GB |
| D2: D1 + lm_head Q4_0 | 9.9743 | +10.3% | +11.7% | 97.5 | no | fails |
| D4: D1 + lm_head Q4_K | 9.9030 | +9.5% | +10.5% | 95.5 | no | en.txt fails |
| D5: Q3_K experts + Q6_K head | 9.7518 | +7.8% | +10.2% | 85.0 | no | en.txt fails, and slower |
| **D6: D5 with 6 experts/token** (after R9: the file must shrink) | **9.6051** | **+6.2%** | **+7.0%** | 78.3 (−15% vs D1 on host, no memory pressure there) | **shipped in 2.2.0+23** | better than D1 on both sets and **0.61 GB smaller** (3.16 GB; experts 2.47 GB); bytes/token ≈ D1 |
| D7: IQ3_XXS experts, 6 experts/token | 9.7836 | +8.2% | +10.05% | — | no | en.txt fails (just); deleted |

Shipped 2026-11-01: `HoeioUser/vollie-qwen3.5-4b` / `vollie-granite-4.0-h-tiny-d1-e5.gguf`, commit
`481aad16`, 3,774,686,848 B, sha256 `403ed810…`. Built from mradermacher's F16 with mradermacher's
imatrix (`tools/quant_variant.sh`). The app pins that commit (`ModelDownloadService.kt`); phones on older
files keep them until the new one verifies.

## 5. Phase 3 — runtime iterations, chronological

All device rows are user-run Settings speed tests ("Optimize for this phone"): ~120-token prompt, 32-64
written tokens per run, CPU unless noted. "heat" = Android thermal headroom forecast (100% = severe
throttling), recorded from app 2.1.7 on.

| # | app | change | device result (measured on device) | verdict |
|---|---|---|---|---|
| R1 | 2.1.1+13 | D1-e5 file; repack on for 8 GB (`ModelStore.copyBytes` → `canCopyWeights`) | 4 threads **7.4** w / 14 r (cool phone) | kept; first Granite number on the phone; below the 10-14 estimate |
| R2 | 2.1.2+14 | heat throttle changes threads in place (`LlamaEngine.setThreads`, our llamadart patch → `llama_set_n_threads`) instead of a 3.8 GB reload | not verified on device (host: same greedy output before/after a 4 → 2 switch) | kept |
| R3 | 2.1.3+15 | own CPU module (`tool/build_cpu.sh`, KleidiAI on like stock) with **tiny-op v1** (`lab/tiny-op.patch`): ops ≤ N elements run on thread 0, syncs between consecutive ones skipped; speed test sweeps N | (superseded by R4 before a result) | — |
| R4 | 2.1.4+16 / 2.1.5+17 | speed test: one load for all thread counts; no thread counts above the big cores (5 threads = 4 big + 1 A55 took minutes) | 2 threads, N = 0 / 4096 / 16384 / 65536: **3.2 / 3.7 / 3.5 / 7.0** w | **sync cost confirmed as the main per-token cost (2.2x at fixed threads)** |
| R5 | 2.1.6+18 | **bug fix** (below); sweep moved before the thread sweep, larger N | all ~2x lower incl. GPU 2.9 → 0.7 | discarded: phone hot from back-to-back tests |
| R6 | 2.1.7+19 | speed test waits for headroom < 65% (≤ 90 s) and records it; GPU last; 32 tokens/run | 4t N 0 / 65536 / 262144: 3.4 / **5.6** / 5.0 w (69-73%); with 65536: 4t 4.3, **3t 7.5** w / 18 r, 2t 5.8 | tiny-op +65% at 4 threads, matched heat; 3 threads best |
| R7 | 2.1.8+20 | **tiny-op v2**: GET_ROWS sized by output; fused RMS_NORM+MUL may run tiny. Syncs per decode token (simulated on the graph): stock 1,909, v1 1,184, **v2 874** | **3t 8.2** w / 15 r at 78% (best so far, on a hotter phone); 4t 2.9, 2t 5.7, GPU 1.6 | kept |
| R8 | 2.1.9+21 | `GGML_CPU_PROF` profiler in the CPU module (per op: µs computing + µs waiting at the sync after it); speed test adds 2 profiled replies; "Copy speed test details" | 3t N 0 / 65536 / 262144: 5.0 / 5.2 / 5.4 w (79-83%); 4t 3.9, 3t 6.3, 2t 6.5 (75-78%). **Profile, 2 threads, N 262144: 120.7 ms/token = experts (MUL_MAT_ID) 71.4 + dense/head (MUL_MAT) 29.7 + SSM_SCAN 3.9 + state CPY 3.3 + rest ~5; waiting at syncs only 7.0.** N 0: 135.2 ms (experts 86.2) | syncs are no longer the bottleneck; **the expert weights are**: ~250 MB in 71 ms ≈ 3.5 GB/s vs ~19 GB/s for the repacked dense weights |
| R9 | 2.1.10+22 | profiled replies also record major/minor page faults per token and MemAvailable/Cached | **354 and 1,181 major faults per token**; MemAvailable ~2.0-2.1 GB, Cached ~2.1-2.2 GB system-wide; 163 / 137 ms per token (experts 104 / 82 ms); best 2t 6.1 w | **hypothesis confirmed: the experts don't fit in the page cache and are re-read from flash every token.** The model needs ~3.8 GB resident (3.1 GB experts + 0.66 GB repack copy); the phone keeps ~2.1 GB cached in total |
| R10 | 2.2.0+23 | **file D6-e6** (Q3_K experts, 6 experts/token; HF commit `8ab9bf2d`, 3.16 GB, sha256 `5dc1daa0…`); D1 kept on the phone until D6 verifies | **632 major faults/token**; MemAvailable 1.74 GB, Cached 1.85 GB (≈0.3 GB less than R9); 149 ms/token (experts 80 + 23 wait; syncs 31 ms total, threads stalled on each other's faults); best 3t 5.5 w / 18 r | **not fixed**: the cache shrank as much as the file did; no fair D6-vs-D1 comparison under different free memory |
| R11 | 2.2.1+24 | profile line adds Vollie's own memory (RssAnon = app memory incl. repacked copy, RssFile = resident model pages, VmSwap) and SwapFree| superseded before a run | the clean-phone run was withdrawn: the user's direction is to fix this in the runtime, with the phone as it is |
| R12 | 2.2.2+25 | **expert I/O in the CPU module** (`GGML_EXPERT_IO`): once a layer's routing is known, `madvise(MADV_WILLNEED)` the chosen experts' slices of gate, up and down together (bit 1), and `MADV_RANDOM` on expert tensors so faults don't read neighbouring experts (bit 2). Speed test sweeps 0/1/3 after small ops, with major faults per token for each run; second profile = everything off | **expert I/O 0 / 1 / 3: 6.1 / 11.8 / 9.3 tok/s, f = 708 / 253 / 1,229 major faults per token** (3 threads, 74-76%); best setting 3t **10.7 w / 19 r**. Profile (I/O 1): 80 ms/token = experts 32 + 9 wait, dense 25.5, SSM 3.2; 5 major faults/token; Vollie anon 674 MB, model resident 2.07 GB | **kept: 2x writing** (R10 5.5 → 10.7). Bit 2 (no read-around) hurts: more, smaller faults. Experts still ~7.6 GB/s vs dense ~22 GB/s: the prefetch has no lead time |
| R13 | 2.2.3+26 | **next-layer expert prediction** (`GGML_EXPERT_IO` bit 4): at layer L's first expert mat-mul, a helper thread on the 4th big core applies layer L+1's router and norm weight to layer L's FFN input and prefetches the top 6 a layer early; real routing still prefetches. Sweep 0/1/5 (3 dropped) | expert I/O 0 / 1 / 5 (3t): **9.2 / 10.6 / 12.0** w, f = 2,038 / 235 / 249; small ops 65,536 3t: **12.2 w** (best so far). **Guess right 85% on the device.** Profile at 4t: 90.5 ms/token (experts 24 + 22 wait; syncs 31 ms in total) vs 126.7 ms with everything off. Test picked 4t / 9.2 w / 25 r from a low 3t outlier (8.4) | kept. At 4 threads the helper competes for a big core; writing wants 3, reading 4 |
| R14 | 2.2.4+27 | guess 2 more experts (bit 8; host: top-8 covers 91.6% vs 83.6%), sweep 0/1/5/13; the sweep's pick counts as a timing of its thread count, each count scored by its best run (fixes the R13 pick) | picked **writes 3 / reads 4: 11.2 w / 24 r**. Expert I/O (4t) 0/1/5/13 = 6.6/11.4/9.8/9.1 w with f = 543/1,958/380/28: faults no longer track speed. Profile 3t: 98 ms, experts 34 + 17 wait (24 ms at 4t → scales with threads) | selection fix works. On a warm phone the experts are now **compute-bound in the Q3_K kernel**, and single runs vary ±20%, too much to judge the remaining options |
| R15 | 2.2.5+28 | **faster NEON Q3_K dot** (sub-block sums kept in vectors, 4 scales per multiply-add, one horizontal add per block; bit-identical) and an **in-generation A/B** (`GGML_CPU_AB`): the module alternates a compute setting every token and splits the profile, so heat and clocks hit both sides alike. Speed test adds a Q3_K A/B reply; profile lines record core clocks | *pending* | host A/B: expert mat-muls −4% and −8% with the new kernel (end-to-end runs can't see it) |
| R17 | 2.2.6+29 (research kit r1) | **Round 1 build, pre-registered.** CPU module: `GGML_MMID` (1 = no barrier inside `MUL_MAT_ID`, every thread quantizes the activation and builds the expert map itself; 2 = 32-row chunks claimed dynamically, counters reset by thread 0 before the preceding barrier), `GGML_Q3K_X4` (four Q3_K rows per call sharing the activation loads), `GGML_EXPERT_SKIP=<permille>` (+`_RENORM`; **not bit-exact**), `GGML_HELPER_CPUS` (expert-guess helper on the little cores), `GGML_REPACK_RELEASE` (page out the repacked tensors' mmap source after load), `GGML_BARRIER_BENCH`, generic `GGML_CPU_AB=knob:…`, per-thread `mmid_thr` in the profile. llamadart: `speculativeRollbackTokenMax` accepted on hybrids (n_rs_seq snapshots) → `draftSimple` speculation with Granite-4.0-350m Q4_0 (owner-approved, ibm-granite commit b8208a86). Kit: Settings → "Run research kit", one JSON (`tools/research_kit.md`, parser `tools/research_parse.py`) | *pending — predictions below* | bit-identical at ub 1/32/512 for MMID 1/3, X4, combined, RELEASE, HELPER (3 chunks); host A/B relative numbers in the text below |
| R16 | — (host only) | **Campaign to 20 tok/s, Phase A/B** (`04_campaign.md`): per-token model and roofline (DRAM roofline for this file = 46.8 ms at the measured 19.7 GB/s → raw 20 tok/s sits at the roofline; realistic raw ceiling 15-17 tok/s, estimated), 30+ ideas with numbers. Two machine-independent facts measured on host with a lab-only `GGML_EXPERT_LOG` dump (D6-e6, greedy, 4 texts): **expert overlap** — distinct experts per layer over k consecutive tokens 9.0-9.3 (k=2), 11.5-11.9 (k=3), 13.5-14.1 (k=4) vs 6k independent; 45-50% of a token's experts were used by the previous token. **Prompt-lookup drafting** — 1.20-1.27 tokens per model evaluation on the official prompt, 1.01-1.17 on story/code/list (`lab/ngram_sim.py`) | no device run | n-gram speculation **falsified** (≤ +27% before verify costs); batched verification of 3-4 tokens reads 0.58-0.66 of k × 6 experts → draft-model speculation (another GGUF, owner's call) is the one route with the numbers for 20; raw-decode ideas B2/B4/C1/B5/A2 queued for round 1 |

**R17 pre-registered predictions (written 2026-11-02 before the phone run; all "estimated", device scale = R14 profile, 3 threads, 98 ms/token).**

| kit phase / variant | prediction | falsified if |
|---|---|---|
| barrier bench, 3 threads | 5-20 µs per empty barrier (4 threads 10-30); 818 + 120 barriers → 5-19 ms of the 23 ms wait is sync latency | < 2 µs (then the wait is all imbalance → B4/B6 matter, B2/B7 don't) or > 50 µs |
| A/B `MMID:0:1` | −1 to −3 ms/token (120 fewer barriers); `MUL_MAT_ID` wait down, compute unchanged | B ≥ A |
| A/B `MMID:0:3` | −3 to −8 ms/token; `MUL_MAT_ID` wait 16.9 → ≤ 12 ms; `mmid_thr` spread narrows | wait unchanged, or compute up by more than the wait saved |
| A/B `Q3K_X4:0:1` | `MUL_MAT_ID` compute −10 to −20% (34 → 27-31 ms) | < −5% |
| A/B `MMID:0:3,Q3K_X4:0:1` | −6 to −14 ms/token together (98 → 84-92 ms profile scale → best-run ≈ 13.5-14.5 tok/s) | < −4 ms |
| A/B `TINY_OP:65536:262144` | within ±2 ms (R7-R14 saw small differences) | — |
| A/B `EXPERT_SKIP:0:60` (+renorm) | ~1% of routed experts skipped (host) → within ±1 ms; **E1 is nearly worthless on this router** | — |
| A/B `EXPERT_SKIP:0:100` (+renorm) | ~3-6% skipped → −1 to −2 ms; PPL cost +0.46% on 60 chunks (host) — at the tolerance edge, so not for shipping unless the full set says otherwise | — |
| threads: 4t + helper on 0-3 vs 3t inherit | if the 4-thread collapse was helper contention: 4t+helper-on-A55 beats 3t by 5-15% (experts 34 → ~25 ms); if it was throttling/Android contention: 4t still loses and clocks (`mhz_after`) show it | neither (equal within noise) |
| threads: 3t + helper on 0-3 vs 3t inherit | ±3% (the helper rarely collides with 3 compute threads) | — |
| spec: reload with 4 rollback slots, no draft | within −5% of baseline (232 MB more anon → fewer cached experts → a few more faults) | > 10% slower (then the slots cost too much memory on this phone) |
| spec: official 64, draft n=2 | tokens/eval ≈ 2.0 (host) → **effective 15-19 tok/s** if a 3-token verify costs ≤ 1.3× a single token | effective ≤ no-draft (verify too expensive) |
| spec: official 64, draft n=4 | tokens/eval ≈ 2.5 → **14-20 tok/s**; verify of 5 tokens 1.6-1.9× | as above |
| spec: story 64, draft n=4 / n=2 | tokens/eval 2.0 / 1.9 → 12-16 tok/s (adversarial text: smaller gain) | — |
| release: RssFile after load | ≥ 300 MB lower `file=` than the normal load; official runs +0 to +8% vs baseline | `file=` unchanged (the kernel had already evicted the source pages → 0 gain) |

**Why are the experts slow on the phone? (open, R8 → R9).** First guess: IQ4_XS is costly to decode
on the A78. Tested on host (1 thread, warm, repack off for both): IQ4_XS experts 14.0 ms for ~251 MB,
Q4_0/Q4_1 experts 11.7 ms for ~213 MB, i.e. the same cost per byte (17.9 vs 18.3 GB/s-equivalent). So a
format change (e.g. Q4_1 experts) would not help; dropped. Current hypothesis: the experts are the only
large weights still read through mmap (3.2 GB, all of them touched every few tokens), and on an 8 GB
phone with Android, the 0.7 GB repacked copy and the app, the page cache cannot keep them all, so each
token re-reads part of them from flash. Fits: ~3.5 GB/s is in UFS-read range, identical settings vary
5.4-8.2 tok/s between runs, and the user's earlier "the model doesn't stay in memory". A cold page cache
on the host reproduced the effect (1 thread: 140 ms/token cold vs 40.5 ms warm). R9's fault counts decide it.

**Bug found and fixed (R5).** Tiny-op v1's look-ahead treated a following RMS_NORM as tiny, but the
compute loop then ran it fused with MUL on all threads, so other threads read the previous op's output
before thread 0 had written it. Decode was safe by accident (one row, done by thread 0), prompt batches
under ~42 tokens at N = 65536 were not: on the host, `-ub 32` gave PPL **100352** (= vocab size, i.e.
garbage). App 2.1.5+17 shipped with it for one test run. Fix: loop and look-ahead share one rule
(`ggml_cpu_runs_tiny`). Since then every module is checked bit-identical against stock at N = 0, 65536,
262144, 10⁹ with `-ub 1`, `-ub 32` and `-b 512` before it goes into an APK.

**Host measurements of the same patches (measured on host, relative only).** M4 Pro, 3 threads, D1-e5:
tiny-op off 13.4 ms/token, N = 262144 14.1 ms/token (−5%). The profiler shows why the host can't see the
effect: 1,870 syncs cost ~0.7 ms/token in total (~0.4 µs each), vs ~60-70 ms of overhead per token on the
phone.

## 6. What worked, what didn't

Worked:
- Requantizing (D1) to pass the gate while making repack fit in 8 GB: +6.4% PPL, host +15% decode vs
  the shipped config.
- Skipping thread syncs around small ops (tiny-op): 2.2x at 2 threads, 1.65x at 4 threads, on the device.
- Fewer threads than big cores (3 of 4) on a warm phone.
- Thermal-fair measurement (cool-down wait + recorded headroom); without it, runs varied 2x.

Didn't (measured):
- Speculative decoding (draft model, n-gram, EAGLE-3) and MTP: slower on this MoE (host).
- OpenCL on Adreno 660: 0.7-2.9 tok/s, always below the CPU (device).
- Cheaper heads/experts (D2, D4, D5): fail the gate on one of the two sets.
- 5 and 8 threads on the SD888 (A55 in every sync).

Open, ranked by expected value (re-ranked after R9):
1. **Experts don't fit in memory (R9).** Options considered:
   - *Kernel/thread changes:* cannot help, the CPU waits on flash.
   - *Repack off:* same total demand (the file's dense pages would need caching instead of the copy).
   - *No mmap (read the model into app memory):* no eviction, but ~3.8 GB of app memory; this phone was
     killed at ~3.6 GB before → too risky as a default.
   - *mlock:* Android limits apps' locked memory (RLIMIT_MEMLOCK, typically 64 KB) → not available.
   - **Smaller file inside the gate (in progress):** 3-bit experts (−0.6 to −0.85 GB) with 6 experts per
     token (the model's own setting; +2 PPL points vs 5 on the old file) so quality holds; bytes read per
     token stay about the same (6 × 3.4 vs 5 × 4.25 bits). Upper bound if all experts stayed resident and
     ran at the dense weights' rate: ~65 ms/token → ~15 tok/s (estimated).
   - **Runtime expert I/O (R12, in progress):** the phone's memory is a given (user direction after
     R10), so the runtime has to read the experts back well: the faults are synchronous, one page at a
     time on 2-3 compute threads, while the router has already named the 6 experts per layer. R12
     prefetches exactly those slices in one batch and stops read-around into unchosen experts.
   - Next if R12 is not enough: an app-managed expert cache (pread into anonymous memory sized to what
     the phone has free, LRU per expert, as in PowerInfer-2 / "LLM in a flash").
2. Dense mat-muls: 29.7 ms for ~560 MB (~19 GB/s) is near this SoC's bandwidth; little left there.
3. Syncs: 7 ms/token left; a further cut is worth ≤ 6%.
4. Fold residual_scale (file-only): SCALE ops no longer show up in the profile → dropped.
5. Heat: idle threads spin at syncs and burn power, which may cost sustained speed on this SoC.

## 7. Claims that still need on-device verification

- Any effect of tiny-op v2 beyond R7's single run, and its sustained (10 min) behaviour.
- That the in-place thread change (R2) keeps the model loaded when the phone gets hot.
- Why 4 threads collapses on a warm phone (contention with Android's own threads vs throttled cores).
- That the 0.7 GB repack copy never causes low-memory kills (no kill reported since R1).
- Reading speed (14-18 tok/s) is still the SD888's compute ceiling; no change targeted it.

## 8. How to reproduce

- Quality gate: `tools/ppl.sh <tag> <model.gguf>` (wiki + en.txt), host build in `lab/llama.cpp-v080`.
- Host speed: `tools/host_bench.sh <tag> <model.gguf>` → `lab/logs/p/bench.csv`.
- CPU module: `git -C lab/llama.cpp worktree add ../llama.cpp-tiny 620d2225c && git -C lab/llama.cpp-tiny
  apply ../tiny-op.patch`, then `vollie/tool/build_cpu.sh lab/llama.cpp-tiny` (needs the NDK's ninja on PATH).
- Tiny-op exactness: `GGML_TINY_OP=<n> llama-perplexity ... -ub 32 --chunks 3` vs `GGML_TINY_OP=0`.
- Device: Settings → Optimize for this phone → Copy speed test details; or `tools/bench_device.sh`
  (Termux/adb, CSV format in `tools/BENCH_DEVICE.md`).
- Every row: `reports/results.csv`.
