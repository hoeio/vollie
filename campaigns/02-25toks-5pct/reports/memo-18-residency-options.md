# Memo 18: residency under 1.6 to 2.4 GB of page cache, priced on the host for OLMoE and Granite

Agent model: claude-opus-5-5

Date 2026-11-14. Author: Tom (code). **Nothing here ran on a phone and no phone speed is measured here.** Labels:
*device* = read from a kit report (named each time), *host* = Apple M4 Pro, campaign 1's runtime built from
`patches/llama.cpp/0001-0003` (the r16/r20 CPU module's source, `GGML_NATIVE=OFF`, `armv8.2-a+dotprod+fp16`, BLAS and
KleidiAI off), *estimated* = arithmetic or simulation. Device reports used: `device/r20_2026-11-14_summary.json` (R56),
`device/r16-control_2026-11-14_summary.json` (R57) and campaign 1's `reports/device/r16_2026-11-08.json` (R49). The
Granite file in this memo is **campaign 1's `d6-q3k-e6.gguf`** (the control's file, not a campaign-2 file).

## 0. Summary

1. **Memo-13's working set was wrong, and the Granite discrepancy is resolved.** Memo-13's tracer read
   `ffn_moe_topk` (a strided view of the argsort) as if it were contiguous, so every prompt token after the first got
   wrong expert ids. A new tracer that records the pages the runtime actually touches (§1) gives, with 4 KB
   accounting: **Granite union 1,966 to 1,986 MB** (memo-13: 2,569), **OLMoE union 2,673 MB** (memo-13: 2,725).
   Granite's r16 runs (2026-11-08, *device*) settled at `RssFile` 1,990 to 2,015 MB with 4 to 40 major faults. That
   fits a 1.97 to 1.99 GB working set, so the device and the trace agree. OLMoE's r20 mincore never went above
   2,394 MB, below the 2,673 MB it needs, so it was never resident.
2. **The decode alone needs more than the cache for OLMoE.** Its 32 decode tokens touch 790 distinct slices
   (2,155 MB). Granite's touch 1,031 slices (1,111 MB). That one number explains most of what follows. Granite's
   decode set fits in 1.6 to 1.8 GB if the prefill does not push it out. OLMoE's does not fit below about 2.2 GB, even
   with perfect foresight.
3. **The simulator** (LRU over 4 KB-page units, cache size = the run's measured residency, waits = oldest pages
   reclaimed down to the next run's measured residency) reproduces how faults spread over the tokens of r20's
   profiled runs: per-token time against simulated re-read MB, r = 0.92 and 0.82. It also reproduces the medians of
   both same-day sessions at the claim positions (*estimated* against *device*): OLMoE predicted 11.6, printed 11.25;
   Granite predicted 10.7, printed 11.46. It does **not** reproduce the spread from run to run. Runs that follow a
   long wait are slower than predicted. Major-fault counts per run are not reproduced either (r = 0.14 to 0.36),
   because with `EXPERT_IO=1` a major fault marks a race, not a unit of I/O. These fits are in-sample: the ms/MB
   values come from the same reports.
4. **What the options give** (*estimated*, central, back-to-back runs, tok/s):

   | option | OLMoE C=1.8 | OLMoE C=2.2 | Granite C=1.8 | Granite C=2.2 |
   |---|---|---|---|---|
   | (a) ship, `EXPERT_IO=1` | 7.9 | 11.5 | 14.1 | 19.8 |
   | (b) `EXPERT_IO=5` | 6.9 (sim, pessimistic; R56 measured +6.6 %) | 10.8 | 10.0 | 13.9 |
   | (c) re-read before token 1, oracle / prompt frequency | 11.8 / 8.4 | 19.7 / 13.0 | 18.6 / 15.7 | 19.8 / 19.8 |
   | (d) prefill `MADV_COLD` of slices routed < 8 times | 8.2 | 12.9 | 17.3 | 19.8 |
   | (c)+(d) frequency set | 8.4 | 13.1 | 16.4 | 19.8 |
   | (e) no repack, 50 % of the freed anon becomes cache | 6.7 | 10.1 | 10.8 | 17.0 |
   | (f) OLMoE skip `MASS=900` decode / + (d) | 9.1 / 9.6 | 12.6 / 14.7 | (prefill `SKIP=200`: 19.8, rules question) | 19.8 |
   | (g) R23-style anon copy of the 1 GB most-routed slices | 7.3 | 12.3 | 15.2 | 19.8 |

5. **Recommendation: build (d) first, prefill `MADV_COLD` by routing count** (`GGML_PREFILL_COLD=<τ>`). It is advice
   only, so bit-identical, and generic. It cuts the time to first token by 0.6 to 2.0 s and does not add to it. It
   is the best implementable option for Granite: 12.1 → 15.9 at 1.6 GB, 14.1 → 17.3 at 1.8 GB. Replayed on
   2026-11-14's measured residency it gives Granite 10.7 → 13.4 and OLMoE 11.6 → 13.2. **For OLMoE no
   residency policy reaches 20 tok/s below about 2.2 GB of cache.** The oracle bound is 11.8 at 1.8 GB and 15.5
   at 2.0 GB. That is a limit of the file and the phone, not of the runtime.

## 1. Working sets (task 1)

### 1.1 What was wrong in memo-13

`ffn_moe_topk-<il>` is `ggml_top_k`, a view of the argsort with rows `nb[1]` = 64 × 4 B apart. Memo-13's tracer copied
`k × n_tok` contiguous ints. That is right for one-token graphs (decode) but garbage for the 156-token prefill: rows 2
to 156 were read from the middle of token 1's full argsort. The decode sets in memo-13 were right. Its prefill unions
were not: Granite 2,502 slices in memo-13, 2,069 routed in fact (1,827 kept after skip 117); OLMoE 968 in memo-13, 926
in fact. `tools/c2_ws_trace_ids.cpp` is the fixed id tracer. Its prefill sets equal the page-trace sets slice for
slice, for both models.

### 1.2 The page-touch tracer

`tools/c2_ws_trace.cpp` links the runtime under test. After load and the app's one-token "Hi" warm-up, it sets the
whole model mapping to `PROT_NONE`. Each first touch per phase faults, is logged and the page is re-opened, and every
phase starts closed again. That gives the exact set of host pages (16 KB) the runtime reads in the prefill and in each
decode graph, with the real kernels, repack, `EXPERT_IO`, skip and head2. The ranges llama.cpp unmapped after load
(`unmap_fragment`) are reported. Run with the device's env:

- OLMoE `olmoe-Mi.gguf`: `GGML_TINY_OP=262144 GGML_EXPERT_IO=1 VOLLIE_SAMPLER_BYPASS=1`, 3 threads, the r17/r20 render
  (`<s>` fallback, 156 tokens), and the device reply forced (r17 to r20 printed one text).
- Granite `d6-q3k-e6.gguf`: r16's served env plus `GGML_EXPERT_SKIP=117 GGML_HEAD2=2048 GGML_HEAD2_Q=q4_0`, 155 tokens,
  r16's device text forced.

KV q8_0, flash attention, `n_ubatch` 512 (llamadart's default; the prompt is one batch). Two runs back to back gave
identical page sets.

`tools/c2_ws_trace.py` turns the log into units on 4 KB pages, the phone's page size:

- One unit per (layer, expert) slice: gate + up + down, rounded out to 4 KB. A slice counts as touched when more than
  half of its interior host pages were touched.
- One unit per other touched tensor.
- `token_embd` row gathers sized by expectation. This is *estimated*: 16 KB pages cannot show 4 KB pages.

### 1.3 Numbers (*host* page trace; MB in 4 KB accounting, *estimated* from the file layout)

| | OLMoE, repack on | OLMoE, repack off | Granite d6-e6, repack on | Granite, repack off |
|---|---|---|---|---|
| prefill: slices / MB | 926 / 2,524 | 927 / 2,763 | 1,827 (of 2,069 routed; skip 117 also acts in the prefill) / 1,908 | 1,823 / 2,317 |
| decode per token: experts / dense / `token_embd` MB | 347.6 / 9.3 / 0.0 | 347.6 / 245.0 / 0.0 | 167.9 / 25.4 / 7.5 (head2 rescoring rows) | 167.7 / 438.2 / 7.5 |
| decode, 32 tokens: distinct slices / MB | 790 / 2,155 | 786 / 2,380 | 1,031 / 1,111 | 1,032 / 1,525 |
| decode slices never seen in the prefill | 59 | 59 | 73 | 75 |
| **union** (distinct 4 KB pages) | **2,673** | 2,923 | **1,957 + 10 to 29** (`token_embd`) | 2,415 |
| model pages left unmapped by llama.cpp after load | 86 MB (the repacked Q6_K head at the file's tail) | 2 MB | 6 MB | 4 MB |

- **Repack on: which mmapped source pages are still touched after load? None.** On the host only the 16 KB pages
  shared with a neighbouring F32 tensor showed a touch. The repacked attention (151 MB OLMoE) and dense Q4_0 (412 MB
  Granite) are read once at load and then age out. OLMoE's head (84.5 MB) is even unmapped.
- What is read through the mmap every token with repack on:
  - OLMoE: router F32 (8.4 MB) and norms;
  - Granite: router F32 15.7, conv1d 2.4, the Q4_1 shared-expert down 4.9, and about 7.5 MB of `token_embd` rows for
    head2's exact rescoring.
- **Repack off** puts OLMoE's 235 MB and Granite's 412 MB of dense weights through the mmap every token (+0.24 / +0.41
  GB of working set). They are always the most recent pages, so they come straight out of the cache.
- Correction to the task text: Granite's repacked copy is 412 MB of anon. The app's 0.62 to 0.7 GB of anon also holds
  head2's Q4_0 copy, the KV cache and compute buffers, so "0.7 GB freed" overstates it.

### 1.4 Reconciliation with the phone

**Granite, r16 2026-11-08 (*device*):** runs 10, 11, 15, 16 and 19 to 21 had `RssFile` 1,990 to 2,015 MB before and
after, 4 to 40 major faults, about 3.6 to 3.9 k minor faults and a prefill of 3.5 to 3.6 s. `RssFile` counts every
mapped file page of the process, the model's and some library pages. A run that faults nothing must find everything it
touches already mapped, so the touched set was at most about 2.0 GB. The trace gives 1.97 to 1.99 GB. **The trace is
right, and memo-13's 2,569 MB was its bug.** The other candidates are not needed:

- page-aligned accounting: at most +18 MB from rounding;
- skipped slices: skip 117 removes 242 routed prefill slices, and the trace already contains that;
- `REPACK_RELEASE`: never set in r16;
- `RssFile` missing pages: not needed;
- read-ahead: it adds pages, it does not remove them.

**OLMoE, r20 (*device*):** after a run 2,030 to 2,394 MB of the file were resident; before a run 197 to 2,362 MB. The
benchmark needs 2,673 MB, plus whatever source pages are still cached after load. No run could be resident, which
matches every run's major faults (≥ 468). The minimum run-start residency for a fault-free run would be about 2.67 GB.
The maximum seen was 2.39 GB.

## 2. The simulator (task 2)

`tools/c2_pagecache_sim.py` (with `--selftest`) runs LRU over the units with partial residency.

- **Why LRU at unit level.** The pages of a slice are read together, so they age together. The page-level LRU order
  is the unit order.
- **Why LRU stands in for the kernel.** Linux 5.4 (the SD888's kernel line) keeps file pages on an active/inactive
  pair. Mapped pages referenced twice are promoted. For one process sweeping its own file that behaves like
  second-chance LRU. Not modelled: activation subtleties, and readahead pages that are never used.
- Per run: the prefill, layer by layer, then the 32 decode graphs.
- `EXPERT_IO=1` is modelled as exact-slice reads. They are bytes, not faults.
- `EXPERT_IO=5` adds wrong next-layer guesses: 15 % of each layer's slices, drawn at random, read for nothing.
- Cold, pageout, re-read and skip options act on the unit lists.

**Validation runs.**

- Each r20 run: capacity = that run's `res_after_mb`. Before the run, the oldest units are reclaimed down to
  `res_before_mb` (the wait). Stories are treated as official runs.
- The control: the same with `RssFile` after/before (no mincore in kit r16).
- r16 2026-11-08 as a third set.

**Fits (*estimated* against *device*).**

| data | quantity | result |
|---|---|---|
| r20 pos 13, 14 (OLMoE, per token) | ms against simulated MB re-read in that token | r = 0.92 / 0.82; slope 1.86 / 1.24 ms/MB, intercept 29.7 / 34.0 ms. The first four tokens, the dip at token 5 and the spike at token 18 all appear in the simulation |
| the same | major faults against simulated MB | r = 0.86 / 0.76, 1.3 to 1.5 faults per MB |
| r20, 27 full-clock official runs | decode ms, floor fixed at 43.4 to 50.8 ms/token | **1.24 to 1.51 ms per re-read MB** (gated 1.55, warm block 1.18); rmse 386 to 404 ms per run |
| r20, same runs | TTFT against simulated prefill MB | 2.27 ms/MB + 3.28 s, r = 0.82 |
| r20 claim positions | median tok/s | predicted 11.6 (floor 47 ms, 1.4 ms/MB; 9.7 to 14.1 over the parameter range); printed **11.25** |
| r20 `c1_off` (no `EXPERT_IO`, 1,555 MHz) | MB per major fault | 0.17 to 0.23 MB, which **fits a 256 KB read-around window** again (r17's fit) |
| control 2026-11-14 (Granite) | decode ms, floor 46 ms | 2.4 ms/MB, r = 0.86; TTFT 1.98 ms/MB, r = 0.70 |
| control | claim-position median | predicted 10.7 (floor 49 ms, 2.5 ms/MB; 9.4 to 12.2); printed **11.46** |
| r16 2026-11-08 (Granite) | decode ms, floor 46 ms | 2.7 ms/MB, r = 0.87. At `RssFile` ≈ 1,990 the simulation still predicts 0.9 GB of prefill re-read where the phone read nothing: the simulated union is 1 to 2 % too large, or `RssFile` is a little below residency |

**Where it does not fit.**

1. **Variation between runs.** Simulated decode MB per run varies only from 587 to 1,120. The device's decode time
   varies from 1.85 to 4.1 s. Runs after a long wait, with low `res_before`, are 0.5 to 1.5 s slower than predicted.
   In the simulation the prefill sweep restores the same order every run, so a wait only changes the prefill. On the
   phone something more persists. The most likely cause is reclaim continuing during the run, so the effective
   capacity is below `res_after`.
2. **Per-run major faults** (1,001 to 33,288) are not predicted. With `EXPERT_IO=1`, thread 0 advises while the other
   threads already read. Under pressure the readahead allocation (`__GFP_NORETRY`) can also fail and fall back to
   single-page faults.
3. **Every ms/MB value is fitted on the reports it is then checked against.**

The raw r20 report also has the mincore split `experts_mb`/`rest_mb` and `/proc/self/io` `read_bytes` per run. They are
not in the summary. Transcribing them would allow a direct check of simulated bytes against read bytes.

**Rates used below:**

| quantity | rate | source |
|---|---|---|
| OLMoE decode | 1.4 ms per re-read MB (1.0 to 1.9) | the r20 fits above |
| Granite decode | 2.5 ms/MB (2.0 to 3.0) | the control and r16 fits; slices are 1 MB against OLMoE's 2.7, so each WILLNEED is smaller |
| prefill, both | 2.2 ms/MB | r20 and control fits |
| a bulk re-read before token 1 | 0.7 to 2.2 ms/MB | 0.7 = about 1.4 GB/s sequential UFS 3.1, an assumption; 2.2 = the prefill rate |

Memo-15's 3.5 ms/MB was fitted on r17, with `EXPERT_IO` off and blocking faults. It is the rate for that state, not for
the shipping one.

**Fault-free tokens:** OLMoE 43 / 47 / 51 ms, Granite 46 / 49 / 53 ms (*device*, R56/R57, 2,112 MHz).

**Formula:** tok/s = 31,000 / (32 × floor + k × decode MB). That is the benchmark's definition: `decode_ms` is the sum
of the 32 decode graphs, and 31 tokens are counted.

## 3. Options × cache (task 3)

C is the model file's page cache during a run, in mincore terms; for Granite it is in `RssFile` terms. The table shows
back-to-back runs, run 4, *estimated*. "Wait" runs (0.5 GB of the oldest pages reclaimed before each run, r20's median
loss) change only the prefill column in the simulation. See §2 for why the phone is worse than that. Each cell gives
tok/s central [range over floor and ms/MB], decode re-read MB over 32 tokens, prefill re-read MB (+ the re-read before
token 1), and ΔTTFT in s.

#### OLMoE

| option | C = 1.6 GB | C = 1.8 GB | C = 2.0 GB | C = 2.2 GB | C = 2.4 GB |
|---|---|---|---|---|---|
| a ship (EXPERT_IO=1) | 6.3 [4.9-8.1] · 2439 · 1935 · +0.0 | 7.9 [6.3-9.9] · 1744 · 1746 · +0.0 | 9.5 [7.7-11.8] · 1249 · 1510 · +0.0 | 11.5 [9.5-13.9] · 853 · 1202 · +0.0 | 13.4 [11.4-15.9] · 577 · 878 · +0.0 |
| b EXPERT_IO=5 (sim, pessimistic) | 5.4 [4.2-7.0] · 3023 · 1949 · +0.0 | 6.9 [5.5-8.9] · 2121 · 1752 · +0.0 | 8.6 [6.9-10.8] · 1507 · 1496 · -0.0 | 10.8 [8.9-13.2] · 973 · 1223 · +0.0 | 12.9 [10.8-15.3] · 648 · 851 · -0.1 |
| c re-read, oracle | 8.5 [6.8-10.7] · 1532 · 1935 (+810) · +1.8 | 11.8 [9.8-14.2] · 800 · 1746 (+874) · +1.9 | 15.5 [13.4-17.9] · 356 · 1510 (+853) · +1.9 | 19.7 [18.0-21.8] · 48 · 1202 (+782) · +1.7 | 20.5 [18.9-22.4] · 5 · 878 (+543) · +1.2 |
| c re-read, prompt frequency | 6.6 [5.2-8.5] · 2274 · 1935 (+857) · +1.9 | 8.4 [6.8-10.6] · 1556 · 1746 (+853) · +1.9 | 10.3 [8.4-12.6] · 1078 · 1510 (+798) · +1.8 | 13.0 [11.0-15.5] · 627 · 1202 (+591) · +1.3 | 15.5 [13.5-17.9] · 352 · 796 (+330) · +0.5 |
| c re-read, last 32 prompt tokens | 6.2 [4.9-8.0] · 2492 · 1935 (+845) · +1.9 | 7.6 [6.1-9.6] · 1838 · 1746 (+782) · +1.7 | 9.7 [7.8-11.9] · 1220 · 1510 (+720) · +1.6 | 11.9 [9.9-14.3] · 793 · 1195 (+549) · +1.2 | 14.7 [12.7-17.2] · 430 · 807 (+225) · +0.3 |
| d prefill COLD, count < 8 | 6.4 [5.1-8.3] · 2362 · 1673 · -0.6 | 8.2 [6.6-10.4] · 1613 · 1326 · -0.9 | 10.2 [8.4-12.6] · 1088 · 820 · -1.5 | 12.9 [10.8-15.3] · 646 · 506 · -1.5 | 16.6 [14.6-18.9] · 260 · 306 · -1.3 |
| d prefill PAGEOUT, frequency set | 6.5 [5.1-8.4] · 2332 · 1327 · -1.3 | 8.4 [6.7-10.5] · 1568 · 1076 · -1.5 | 10.2 [8.4-12.5] · 1094 · 910 · -1.3 | 12.3 [10.3-14.8] · 720 · 648 · -1.2 | 14.2 [12.2-16.7] · 482 · 594 · -0.6 |
| c+d COLD + re-read, frequency set | 6.6 [5.2-8.5] · 2264 · 1327 · -1.3 | 8.4 [6.8-10.6] · 1549 · 1076 · -1.5 | 10.3 [8.4-12.7] · 1072 · 910 · -1.3 | 13.1 [11.1-15.6] · 615 · 645 · -1.2 | 15.7 [13.6-18.1] · 340 · 442 · -1.0 |
| c+d oracle (bound) | 8.5 [6.9-10.7] · 1517 · 1171 (+23) · -1.6 | 11.8 [9.8-14.2] · 800 · 952 (+27) · -1.7 | 15.5 [13.5-18.0] · 350 · 723 (+27) · -1.7 | 19.9 [18.2-21.9] · 38 · 524 · -1.5 | 20.6 [19.0-22.5] · 0 · 284 · -1.3 |
| e no repack, 0 % to cache | 4.9 [3.8-6.4] · 3402 · 2307 · +0.8 | 5.8 [4.6-7.5] · 2667 · 2121 · +0.8 | 7.3 [5.8-9.2] · 1891 · 1910 · +0.9 | 8.8 [7.2-10.9] · 1374 · 1655 · +1.0 | 10.7 [8.8-12.9] · 940 · 1396 · +1.1 |
| e no repack, 50 % to cache | 5.4 [4.2-7.0] · 2991 · 2203 · +0.6 | 6.7 [5.3-8.6] · 2153 · 2008 · +0.6 | 8.1 [6.5-10.1] · 1595 · 1786 · +0.6 | 10.1 [8.3-12.3] · 1057 · 1519 · +0.7 | 11.7 [9.8-14.0] · 752 · 1180 · +0.7 |
| e no repack, 100 % to cache | 6.2 [4.9-7.9] · 2439 · 2082 · +0.3 | 7.7 [6.1-9.7] · 1750 · 1881 · +0.3 | 9.3 [7.6-11.4] · 1256 · 1630 · +0.3 | 11.0 [9.2-13.3] · 871 · 1325 · +0.3 | 12.8 [10.9-15.1] · 593 · 994 · +0.3 |
| f skip MASS=900, decode only | 7.5 [5.9-9.6] · 1978 · 1938 · +0.0 | 9.1 [7.3-11.5] · 1446 · 1737 · -0.0 | 11.0 [9.0-13.6] · 1021 · 1512 · +0.0 | 12.6 [10.5-15.4] · 766 · 1202 · +0.0 | 14.8 [12.5-17.6] · 509 · 918 · +0.1 |
| f skip 900 decode + prefill COLD <8 | 7.6 [6.0-9.8] · 1920 · 1667 · -0.6 | 9.6 [7.7-12.0] · 1325 · 1270 · -1.0 | 12.2 [10.1-14.9] · 823 · 793 · -1.6 | 14.7 [12.4-17.5] · 523 · 509 · -1.5 | 18.5 [16.3-21.2] · 209 · 278 · -1.3 |
| f skip 900 decode + MASS=700 prefill (rules q.) | 7.5 [6.0-9.7] · 1948 · 1713 · -0.5 | 9.2 [7.4-11.6] · 1415 · 1500 · -0.5 | 11.4 [9.3-14.1] · 955 · 1227 · -0.6 | 13.5 [11.2-16.2] · 659 · 913 · -0.6 | 16.3 [14.0-19.1] · 374 · 621 · -0.6 |
| g anon copy of 0.5 GB most-routed (R23) | 6.0 [4.7-7.8] · 2611 · 1748 · -0.4 | 7.9 [6.3-10.0] · 1736 · 1586 · -0.4 | 9.5 [7.7-11.8] · 1252 · 1367 · -0.3 | 11.8 [9.8-14.2] · 801 · 1115 · -0.2 | 13.7 [11.7-16.2] · 538 · 819 · -0.1 |
| g anon copy of 1.0 GB most-routed (R23) | 6.2 [4.9-8.0] · 2480 · 1397 · -1.2 | 7.3 [5.8-9.3] · 1962 · 1341 · -0.9 | 9.9 [8.1-12.3] · 1152 · 1180 · -0.7 | 12.3 [10.3-14.8] · 721 · 975 · -0.5 | 14.2 [12.2-16.7] · 482 · 721 · -0.3 |

Major faults in the timed decode, OLMoE, *estimated* from the r20 per-token fit (about 1.4 per re-read MB): base 2,400
at 1.8 GB and 1,200 at 2.2 GB; (d) 2,260 and 900.

#### Granite d6-e6

| option | C = 1.6 GB | C = 1.8 GB | C = 2.0 GB | C = 2.2 GB | C = 2.4 GB |
|---|---|---|---|---|---|
| a ship (EXPERT_IO=1) | 12.1 [10.7-13.6] · 401 · 1175 · +0.0 | 14.1 [12.7-15.7] · 250 · 989 · +0.0 | 18.5 [16.9-19.9] · 45 · 893 · +0.0 | 19.8 [18.3-21.1] · 0 · 0 · +0.0 | 19.8 [18.3-21.1] · 0 · 0 · +0.0 |
| b EXPERT_IO=5 (sim, pessimistic) | 8.5 [7.4-9.9] · 835 · 1122 · -0.1 | 10.0 [8.8-11.5] · 612 · 966 · -0.1 | 12.4 [11.0-14.0] · 374 · 892 · -0.0 | 13.9 [12.4-15.4] · 267 · 893 · +2.0 | 19.8 [18.3-21.1] · 0 · 0 · +0.0 |
| c re-read, oracle | 18.5 [17.0-19.9] · 43 · 1175 (+385) · +0.8 | 18.6 [17.1-20.0] · 38 · 989 (+198) · +0.4 | 19.5 [18.0-20.8] · 9 · 893 (+4) · +0.0 | 19.8 [18.3-21.1] · 0 · 0 · +0.0 | 19.8 [18.3-21.1] · 0 · 0 · +0.0 |
| c re-read, prompt frequency | 14.2 [12.8-15.8] · 243 · 1033 (+495) · +0.8 | 15.7 [14.2-17.2] · 165 · 642 (+198) · -0.3 | 18.5 [16.9-19.9] · 45 · 176 · -1.6 | 19.8 [18.3-21.1] · 0 · 0 · +0.0 | 19.8 [18.3-21.1] · 0 · 0 · +0.0 |
| c re-read, last 32 prompt tokens | 13.4 [11.9-15.0] · 301 · 1008 (+407) · +0.5 | 15.7 [14.2-17.2] · 163 · 609 (+146) · -0.5 | 18.5 [16.9-19.9] · 45 · 273 · -1.4 | 19.8 [18.3-21.1] · 0 · 0 · +0.0 | 19.8 [18.3-21.1] · 0 · 0 · +0.0 |
| d prefill COLD, count < 8 | 15.9 [14.4-17.4] · 153 · 404 · -1.7 | 17.3 [15.8-18.8] · 89 · 204 · -1.7 | 19.7 [18.2-21.0] · 1 · 4 · -2.0 | 19.8 [18.3-21.1] · 0 · 0 · +0.0 | 19.8 [18.3-21.1] · 0 · 0 · +0.0 |
| d prefill PAGEOUT, frequency set | 14.5 [13.0-16.1] · 227 · 896 · -0.6 | 16.1 [14.6-17.6] · 144 · 895 · -0.2 | 18.5 [16.9-19.9] · 45 · 893 · +0.0 | 19.8 [18.3-21.1] · 0 · 0 · +0.0 | 19.8 [18.3-21.1] · 0 · 0 · +0.0 |
| c+d COLD + re-read, frequency set | 15.0 [13.5-16.5] · 201 · 547 · -1.4 | 16.4 [14.9-17.9] · 129 · 284 · -1.6 | 18.5 [16.9-19.9] · 45 · 176 · -1.6 | 19.8 [18.3-21.1] · 0 · 0 · +0.0 | 19.8 [18.3-21.1] · 0 · 0 · +0.0 |
| c+d oracle (bound) | 19.8 [18.3-21.1] · 0 · 404 · -1.7 | 19.8 [18.3-21.1] · 0 · 204 · -1.7 | 19.8 [18.3-21.1] · 0 · 4 · -2.0 | 19.8 [18.3-21.1] · 0 · 0 · +0.0 | 19.8 [18.3-21.1] · 0 · 0 · +0.0 |
| e no repack, 0 % to cache | 8.0 [7.0-9.3] · 877 · 1635 · +1.0 | 9.4 [8.3-10.8] · 646 · 1468 · +1.1 | 10.8 [9.6-12.2] · 476 · 1249 · +0.8 | 12.9 [11.6-14.3] · 287 · 1026 · +2.3 | 17.0 [15.6-18.2] · 57 · 891 · +2.0 |
| e no repack, 50 % to cache | 9.4 [8.3-10.8] · 643 · 1466 · +0.6 | 10.8 [9.6-12.2] · 475 · 1245 · +0.6 | 12.9 [11.6-14.4] · 283 · 1022 · +0.3 | 17.0 [15.7-18.3] · 53 · 890 · +2.0 | 18.4 [17.1-19.5] · 0 · 0 · +0.0 |
| e no repack, 100 % to cache | 10.8 [9.6-12.3] · 469 · 1242 · +0.1 | 13.0 [11.7-14.4] · 280 · 1019 · +0.1 | 17.3 [15.9-18.5] · 43 · 890 · -0.0 | 18.4 [17.1-19.5] · 0 · 0 · +0.0 | 18.4 [17.1-19.5] · 0 · 0 · +0.0 |
| f prefill SKIP=200 (rules q.) | 19.8 [18.3-21.1] · 0 · 0 · -2.6 | 19.8 [18.3-21.1] · 0 · 0 · -2.2 | 19.8 [18.3-21.1] · 0 · 0 · -2.0 | 19.8 [18.3-21.1] · 0 · 0 · +0.0 | 19.8 [18.3-21.1] · 0 · 0 · +0.0 |
| g anon copy of 0.5 GB most-routed (R23) | 12.9 [11.5-14.5] · 334 · 1005 · -0.4 | 14.6 [13.2-16.2] · 219 · 849 · -0.3 | 18.4 [16.9-19.8] · 46 · 777 · -0.3 | 19.8 [18.3-21.1] · 0 · 0 · +0.0 | 19.8 [18.3-21.1] · 0 · 0 · +0.0 |
| g anon copy of 1.0 GB most-routed (R23) | 14.1 [12.6-15.7] · 253 · 745 · -0.9 | 15.2 [13.7-16.8] · 188 · 641 · -0.8 | 18.4 [16.9-19.8] · 46 · 573 · -0.7 | 19.8 [18.3-21.1] · 0 · 0 · +0.0 | 19.8 [18.3-21.1] · 0 · 0 · +0.0 |

### 3.1 Notes per option

- **(a)** Below the union the run is a cyclic sweep larger than the cache, which is LRU's worst case. The prefill
  evicts the slices the first decode tokens need. That is r20's "first ten tokens fault".
- **(b) `EXPERT_IO=5`.** The simulation charges every wrong guess as a random slice read for nothing, so it is
  pessimistic. Real wrong guesses probably fall on popular, already-resident slices. R56 measured +6.6 % against
  `c1_ship` in the warm block at a 1,555 MHz cap (*device*). The simulator gives no reason to expect more than a
  small gain. The guess's real value is overlap with compute, which the byte model does not price.
- **(c) re-read before token 1.**
  - Hit rates of the predictors against the real decode set, in slices / weighted by accesses:

    | predictor | OLMoE: MB, slices / accesses | Granite: MB, slices / accesses |
    |---|---|---|
    | last N prompt tokens, N = 8 | 1,073 MB, 0.37 / 0.30 | 729 MB, 0.39 / 0.52 |
    | N = 16 | 1,817 MB, 0.69 / 0.71 | 1,150 MB, 0.64 / 0.76 |
    | N = 32 | 2,099 MB, 0.79 / 0.81 | 1,429 MB, 0.76 / 0.83 |
    | N = 64 | 2,121 MB, 0.80 / 0.81 | 1,492 MB, 0.79 / 0.86 |
    | prompt frequency, whole prompt | 2,515 MB, 0.93 / 0.94 | 1,874 MB, 0.93 / 0.96 |
    | prompt frequency, inside a 1.7 GB budget | 0.68 / 0.74 | 0.88 / 0.94 |
    | combined (frequency plus the last 16 tokens at the recent end) | no better than frequency at equal MB | no better than frequency at equal MB |

    Weighting by routing mass or by position in the prompt changed nothing (*host*, ±0.02).
  - The re-read costs 0.2 to 0.9 GB before the first token: +0.3 to +1.9 s at 0.7 to 2.2 ms/MB. The OLMoE oracle is
    an upper bound only. Even the oracle stays at 11.8 tok/s at 1.8 GB, because the decode set itself (2.15 GB) does
    not fit.
- **(d) prefill advice.**
  - The rule: after each prefill layer, the slices the prompt routed fewer than τ times are marked `MADV_COLD`
    (deactivated; they remain resident if nothing needs the room) or `MADV_PAGEOUT` (dropped).
  - COLD beats PAGEOUT whenever the cache has room. PAGEOUT forces a re-read of the prompt-only slices on every run;
    COLD only puts them first in line for reclaim. In a wait, the system then takes the cold slices first, which is
    what protects the decode set.
  - It cuts the prefill re-read by 0.3 to 0.8 GB, so TTFT falls.
  - Sensitivity to τ, as tok/s at C = 1.6, 1.8, 2.0, 2.2, 2.4 GB:

    | model | τ = 2 | τ = 4 | τ = 8 | τ = 12 | τ = 24 |
    |---|---|---|---|---|---|
    | Granite | 13.2, 17.1, 19.7, 19.8, 19.8 | 15.4, 17.8, 19.7, 19.8, 19.8 | 15.9, 17.3, 19.7, 19.8, 19.8 | 15.6, 17.2, 19.7, 19.8, 19.8 | 14.4, 16.8, 19.8, 19.8, 19.8 |
    | OLMoE | 6.3, 7.9, 9.6, 11.6, 13.9 | 6.4, 8.0, 9.8, 12.2, 15.2 | 6.4, 8.2, 10.2, 12.9, 16.6 | 6.5, 8.4, 10.4, 13.3, 16.8 | 6.6, 8.4, 10.4, 13.3, 15.8 |

    The cut is flat from 4 to 16.
  - **Replay of 2026-11-14 with the measured residency, run by run (claim-position median, *estimated*):**

    | model | today | τ = 4 | τ = 8 | τ = 12 |
    |---|---|---|---|---|
    | Granite (control) | 10.7 | 11.9 | 13.4 | 13.9 |
    | OLMoE (r20 part A) | 11.6 | 12.5 | 13.2 | 13.2 |

- **(e) no repack** is a net loss in every cell, even when 100 % of the freed anon becomes cache. The dense pages then
  go through the mmap every token and take that room themselves (+245 / +440 MB of working set). The dense kernel
  also costs more: OLMoE +2.7 ms (+0.4 to +6.6, memo-08 §2, plain attention + head); Granite +3.7 ms (*estimated* from
  memo-08's plain Q4_0 rate 0.041 to 0.065 ms/MB against 0.0411 repacked, on 412 MB), both in the floors above. It is
  also **not bit-identical** to the shipping path: the repacked and plain kernels give different greedy text (memo-13
  §7: md5 `bde0011c…` against `88bae85d…`).
- **(f) skip.**
  - OLMoE `MASS=900` in the decode cuts decode bytes 13 % and the floor by about 3.9 ms (R51/memo-03). With (d) it
    gives 14.7 at 2.2 GB and 18.5 at 2.4 GB, the best implementable OLMoE cells. Its cost (+3.32 %, *host*, R51) is
    inside the budget. It lives in c2's series (`patches/llama.cpp-c2/0001`), not in campaign 1's runtime, so it
    would have to be ported to the `c1-olmoe` series.
  - Stronger rules in the prefill only:
    - OLMoE: `MASS=700` gives a 2,300 MB prefill and a 2,585 MB union. That is little gain, because 156 tokens × 8
      cover most slices anyway.
    - Granite: `SKIP=200` gives a 1,063 MB prefill and a 1,424 MB union, enough to make it resident at 1.6 GB in the
      simulation. That cell holds the decode sets fixed. A different prefill changes the KV and so the decode routing
      and text, which the simulation ignores.
  - **Rules question for the lead:** `llama-perplexity -b 512` runs the batched path, the prefill. A numerics rule
    that depends on batch size is therefore priced by the gate only on its prefill half, and its decode half is
    priced only by a `-ub 1` run. A decode-only skip would be free by construction, and a prefill-only skip fully
    charged. I suggest: any batch-size-dependent rule is gated at both `-b 512` and `-ub 1`, and the budget applies
    to the worse of the two. It should not be assumed free either way.
- **(g) and campaign 1's other residency work:**
  - **R23 `EXPERT_ANON`** (hot set in anon memory): the simulator shows no gain at 0.5 GB and small gains at 1 GB.
    The copy takes its room out of the same memory, so the file cache shrinks by the same amount. Campaign 1 closed
    it at Δ ≈ −0.7 ms (noise), hit 66 to 78 %. One effect is not modelled: with swap full, anon pages survive the
    waits that empty the page cache. That would be a real advantage on 2026-11-14's phone, at the price of a higher
    low-memory-killer risk. It is not recommended without a measurement.
  - **R32/R33 static Q4_0 copies (`EXPERT_Q4_LAYERS`):** failed the numerics and are dead. They are also anon, as
    above.
  - **`MMID_TOUCH`:** touching is what the kernel already does, so the simulation is identical to (a). It was parked
    in R39 to R41 because the latency model did not hold.
  - **`REPACK_RELEASE`:** the source pages are never touched after load (§1.3), so it helps only the first run after
    load. That is R17/R18's "0 gain".
  - **`DENSE_WILLNEED`:** moot. The dense pages are touched every token and are always the most recent.
  - **`EXPERT_IO` bit 2 (`MADV_RANDOM`):** lost in R12. With bit 1 most reads are advised, so little read-around
    remains to trim.
  - **Keep-warm:** closed in R56. The phone's reclaim in a wait does not follow the process's own LRU order.

## 4. Each option: identity, TTFT, code, predicted speed (task 4)

| option | bit-identical? | TTFT | code on campaign 1's runtime (new series `patches/llama.cpp-c1-olmoe/`) | predicted tok/s (OLMoE / Granite), C = 1.6 · 1.8 · 2.0 · 2.2 · 2.4 GB |
|---|---|---|---|---|
| (a) ship | yes | — | none | 6.3 · 7.9 · 9.5 · 11.5 · 13.4 / 12.1 · 14.1 · 18.5 · 19.8 · 19.8 |
| (b) `EXPERT_IO=5` | yes (advice) | ≈ 0 | none (in 0001) | measured +6.6 % (R56) |
| (c) re-read, prompt frequency | yes (advice + one byte read per page) | +0.3 to +1.9 s | about 60 lines: counts per (tensor, expert) from the prefill's ids, then `WILLNEED` + touch at the first one-token graph | 6.6 · 8.4 · 10.3 · 13.0 · 15.5 / 14.2 · 15.7 · 18.5 · 19.8 · 19.8 |
| **(d) prefill `COLD`, τ = 8** | **yes (advice only)** | **−0.6 to −2.0 s** | **about 80 lines: in `ggml_cpu_expert_prefetch`'s place, thread 0 at a batch `MUL_MAT_ID` with `ids->ne[1] > 1` counts the routed rows per expert and, after the layer (at the next layer's first `MUL_MAT_ID`, past its barrier, and at graph end), advises `MADV_COLD` on the slices with count < τ for every `MUL_MAT_ID` sharing those ids. `GGML_PREFILL_COLD=<τ>`, off when unset. No architecture names: it keys on `MUL_MAT_ID` and `ids` only, so it covers `olmoe` and `granitehybrid` alike. The `PAGEOUT`/`COLD` fallback helper of `vollie-release.h` (0001) is reused** | **6.4 · 8.2 · 10.2 · 12.9 · 16.6 / 15.9 · 17.3 · 19.7 · 19.8 · 19.8** |
| (c)+(d) | yes | −1.0 to −1.6 s | both | 6.6 · 8.4 · 10.3 · 13.1 · 15.7 / 15.0 · 16.4 · 18.5 · 19.8 · 19.8 |
| (e) no repack | **no** (other kernel; another text) | +0.1 to +2.3 s | none (app flag) | all cells below (a) |
| (f) OLMoE `MASS=900` decode + (d) | no: a declared numerics change (+3.32 %, R51) | −0.6 to −1.6 s | port c2's 0001 rule into the c1-olmoe series (about 150 lines) | 7.6 · 9.6 · 12.2 · 14.7 · 18.5 / — |
| (f) prefill-only stronger skip | no | −0.5 to −2.6 s | a batch-size condition on the skip rule | rules question (§3.1) |
| (g) anon hot set | yes | +load-time copy | `EXPERT_ANON` exists in 0001 | ≤ (d) in every cell |

ms per re-read MB as in §2: OLMoE 1.4 (1.0 to 1.9), Granite 2.5 (2.0 to 3.0), fitted on r20, the control and r16 with
`EXPERT_IO=1`.

## 5. Recommendation (task 5)

**Build (d) first: `GGML_PREFILL_COLD=<τ>` in `patches/llama.cpp-c1-olmoe/0002-prefill-cold.patch`.** It is advice
only, bit-identical, off unless set, and has no architecture name in it. Reasons:

- It is the best implementable option for Granite at 1.6 to 2.0 GB.
- It is as good as anything implementable for OLMoE.
- It lowers TTFT instead of raising it.
- It is the base that a later declared skip (f) would sit on.

**Host checks before the kit:**

- identity at `-ub 1`, `-ub 32` and `-b 512` with the knob set (it acts in those batches);
- the same greedy 32-token text with the knob on and off;
- one `c2-pcold` line per prefill graph: slices advised, MB advised, the `madvise` return value. On macOS the call is
  absent, so the line shows that it was skipped.

**Predictions to pre-register** (*estimated*, central, replay of the session's measured residency):

| model | ship | `PREFILL_COLD=8` |
|---|---|---|
| Granite | about 10.7 | about 13.4 (+25 %) |
| OLMoE | about 11.6 | about 13.2 (+13 %) |

Also: decode re-read bytes per run −40 to −45 % for Granite and −15 to −30 % for OLMoE; TTFT −0.6 to −2.0 s.

**What the next kit must measure:**

- per run `/proc/self/io` `read_bytes` at start, at the first streamed token and at the end, i.e. prefill bytes and
  decode bytes apart (the field is already read in r20; add the first-token read);
- mincore `experts_mb` and `rest_mb` before and after;
- the `c2-pcold` line;
- the per-token arrays of two profiled runs per row.

**Kill rule:** decode-phase `read_bytes` with the knob on, against off in the same ABBA block, not lower by at least
15 % for Granite, or decode tok/s not higher for either model (ABBA pairs; the median of the paired differences
≤ 0). Then the lever is closed and the residency line stops at "the phone's cache decides".

**Say it with numbers:** even with (d) + (f), OLMoE needs about 2.4 GB of cache for 18.5 tok/s, and the oracle bound is
15.5 at 2.0 GB. Below 2.2 GB no runtime residency policy brings OLMoE to 20, let alone 25.

**A one-session, two-model kit:**

- **Storage.** Both files need 6.2 GB (OLMoE 3.07 GB, Granite 3.16 GB from `HoeioUser/vollie-qwen3.5-4b`). Note
  that d6-e6 is campaign 1's abliterated file. It is the control, not a campaign-2 served file. Campaign 2's stock
  Granite file needs its own trace, about two minutes with the tools here.
- **Order.** Model blocks G O O G (ABBA by model), one model loaded per block:
  - each block begins with a load and **two unscored warm-up official runs**. They absorb the cold start and push
    the other file out; the mincore of both files at block start shows the eviction;
  - then 8 scored official runs, knob in ABBA within the block (off, on, on, off, ×2). Two of the eight are
    profiled, one per arm;
  - r16's gate (< 0.65 or 90 s, polled every 5 s) before every scored run, the same for both models;
  - no extra idle and no story runs (their pages muddle the comparison).
- **Time.** About 4 × (load 0.5 min + 10 runs × 0.2 to 1.6 min) ≈ 25 to 65 minutes. If time is short, drop the
  second profiled pair, not the second G block.
- **Fairness.** Both models see the same gate, the same number of warm-ups and the same position mix. The OLMoE and
  Granite blocks interleave in time, so a drift of the day falls on both. The comparison is then the within-block
  on/off delta per model, not the absolute tok/s across models.
- The kit should also log `Cached`, `SwapFree`, the clock caps and the battery, as r20 did.

## 6. Not verified

- The 4 KB accounting is derived from 16 KB host traces and the file layout. The `token_embd` share is an
  expectation (10 to 29 MB of Granite's union).
- Routing on the phone may differ from the host's in a few slices. The 32-token texts are equal.
- LRU stands in for 5.4's two lists. `MADV_COLD` on 5.4 deactivates pages (inactive list, referenced bit cleared);
  the simulation treats it as "first to reclaim". That `MADV_COLD`/`MADV_PAGEOUT` exist on the phone's kernel and are
  allowed for the app's own file is not checked; the patch must log the return value.
- Capacity during a run is taken as the residency after it. The phone does worse after long waits (§2).
- Every ms/MB value is in-sample. The `EXPERT_IO=5` pollution model is a guess.
- The Granite rows are for campaign 1's d6-e6 file with its shipping skip 117 and head2.

## 7. Files

- `tools/c2_ws_trace.cpp`: page-touch tracer, macOS, `mprotect`-based.
- `tools/c2_ws_trace_ids.cpp`: the fixed id tracer, prefill routing.
- `tools/c2_ws_trace.py`: log → access spec on 4 KB pages, with a summary.
- `tools/c2_pagecache_sim.py`: the simulator; `--selftest` passes.

The traces (about 2 to 40 MB each), the specs, the validation and option drivers are scratch files, not kept. They are
regenerated in a few minutes with the commands in the tools' headers.
