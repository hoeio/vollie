# Bonnie's draft for the joint round: the gap to 25 tok/s, open levers re-priced, TEST and TRY lists

Agent model: claude-fable-5-2. Date 2026-11-18. Repository HEAD `023c529`. Draft, not committed, not a memo yet.

**Nothing here ran on a phone or as a host benchmark.** It is my own short scripts over the four stored raw reports
(`campaigns/02-25toks-5pct/reports/device/r22_2026-11-15.json`, `r23_2026-11-16.json`, `r25_2026-11-16.json`,
`r26_2026-11-18.json`), the repository parser on r26, reading of memos 21, 24, 25, 27 and R59 to R66, and arithmetic.
Labels: *device* = read or computed from a stored report (named); *host* = quoted from a host entry; *estimated* =
arithmetic or an in-sample fit. Every fit below is in-sample, mixes runs of different phone states and was not
pre-registered; the R-log verdicts stand as written. Scripts: `scratchpad/joint/decomp.py`, `tokfit.py`, `runfit.py`.

Conversion used throughout (*device*, r26: `decode_tps × decode_ms / 1000` = 30.87 to 30.91 in every run):
**tok/s = 965.6 / (decode wall ms per token, `decode_ms / 32`)**. 25 tok/s is therefore **38.6 ms** per token, wall,
outside-graph time included (memo-21 used 968.75 and 38.75; same thing to 0.4 %).

## 0. Summary

1. **Clean-run floors exist now as whole runs, not as selected tokens** (*device*, runs at cap 2,035 with < 5 MB read in the decode and, but for
   R64 pos 33 with 11,833 cheap ones, under 300 decode major faults): Granite num 20.5 to 21.2 tok/s (7 runs), OLMoE off 20.9 to 22.0
   (8 runs), OLMoE num 21.4 to 23.5 (7 runs, R64 only). Memo-24's estimates (20.8 and 23.2) were about right for
   Granite and about 1 tok/s optimistic for OLMoE num as a run median.
2. **Gap to 38.6 ms from the clean floor:** Granite num **−8.2 ms**, OLMoE off **−6.3 ms**, OLMoE num **−4.7 ms**
   (best single run −2.4). On top of that the median gated run carries a state penalty of **+2.2 to +9.5 ms
   (Granite)** and **+0.8 to +7.4 ms (OLMoE)** depending on the day (R64 evening against R66 with apps restarting).
3. **The state penalty is priced by one number: MB of the wanted file pages missing at run start.** R66 (*device*,
   fit): decode wall = floor + **0.046 ± 0.010 ms/token per missing MB (Granite, r = 0.79, n = 17)** and **0.034 ±
   0.008 (OLMoE, r = 0.71, n = 19)**, i.e. 1.1 to 1.5 ms per MB per 32-token run, memo-18's rate. The app's `VmSwap`
   before the run does **not** predict decode time (slope −0.003 ± 0.014 and +0.007 ± 0.019 ms/token per MB): swap
   costs TTFT, missing file pages cost the metric. This re-prices every memory lever below.
4. **OLMoE `MASS=900` is visible in the op times** although its tok/s effect was "not shown" three times: expert op
   26.2 to 26.6 ms (num, R64 pos 23 and 36, tokens with ≤ 5 majors) against 28.5 ms (off, pos 35), **−1.9 to
   −2.3 ms** (*device*, n = 2 runs against 1, same session, a reading). The parking was a noise problem (SE 1.0
   tok/s ≈ 2.4 ms), not an absent effect. Whether that brings it back is the lead's call.
5. **Position with numbers (*estimated*, clean state, centres of the bands in §2):** Granite 46.8 → about 42.6 ms
   (22.7 tok/s) with pinning, helpers, crumbs, outside-graph and a higher skip threshold; 24.6 with every band
   at its top; 25 needs the expert op near its byte floor as well (LUT or helpers at f ≈ 1). OLMoE num 43.3 → about
   39.8 ms (24.3 tok/s) at the centres; it crosses 38.6 if helpers reach f ≥ 0.6 or an exact LUT lands. **Neither is
   at 25 at the centre, and both need the state penalty near zero in at least 6 of 10 gated runs.**

## 1. The remaining gap per model, decomposed

### 1.1 Clean-run reference tokens (*device*)

Whole profiled runs, tokens 2 to 32 (mean, additive; `compute + wait` as the profiler prints them; "other dense" =
`MUL_MAT` compute + wait minus `head=` and `router=`, which include their waits).

| ms per token | Granite num (R66 pos 47; 13 decode majors, 0 MB read) | OLMoE off (R64 pos 35; 15 majors, 0 MB) | OLMoE num (R64 pos 23 / 36; 110 / 97 majors, 4 to 5 MB) |
|---|---|---|---|
| experts `MUL_MAT_ID` compute | 13.18 | 28.49 | 26.49 / 27.13 |
| experts wait | 0.79 | 0.46 | 0.57 / 0.62 |
| other dense (attention; Granite `ssm_in`, `ssm_out`, shexp), with its waits | 18.81 | 6.86 | 7.34 / 7.38 |
| head | 3.76 | 3.39 | 3.41 / 3.12 |
| router | 0.85 | 0.41 | 0.46 / 0.47 |
| `SSM_SCAN` compute + wait | 2.61 + 0.52 | – | – |
| `GLU` | 0.21 | – | – |
| rest (small ops + per-node dispatch; residual) | 4.91 | 4.23 | 4.25 / 4.25 |
| **graph, tokens 2 to 32** | **45.69** | **43.85** | **42.52 / 42.97** |
| first decode token, excess over the mean, per token | +0.16 (50.8 ms) | +0.45 (58.1 ms) | +0.63 / −0.02 |
| outside the graph (`decode_ms / 32` − mean graph) | 0.95 | 0.80 | 0.85 / 0.65 |
| **wall per token** | **46.8 (20.61 tok/s)** | **45.1 (21.40)** | **44.0 / 43.6 (21.93 / 22.13)** |
| waits inside the four ops (already counted above) | 3.24 | 1.19 | 1.19 / 1.28 |

Scored (unprofiled) clean runs agree, so the profiler does not cost a visible amount: Granite num 21.17, 21.21
(R64 pos 9, 45), 20.79, 20.54, 20.64, 20.09 (R66 pos 42, 43, 45, 46), 20.61 profiled (pos 47); OLMoE off 21.52, 21.81, 21.53, 21.99 (R66
pos 22, 30, 32, 33), 21.44, 20.94, 20.88 (R64); OLMoE num 22.63, 23.54, 22.94, 22.32, 21.43 (R64 pos 20, 21, 32, 33,
17). Medians: **Granite num 20.6 (46.8 ms; the seven runs listed plus pos 47), OLMoE off 21.5 (44.9 ms), OLMoE num 22.3
(43.3 ms).**

### 1.2 The gap by pool

Floors: expert byte floor from memo-21 §0.2 (*estimated*: Granite at skip 130 6.0 to 7.7 ms, OLMoE off 15.2 to
18.2, OLMoE `MASS=900` 12.4 to 15.8); pure sync 666 × 0.30 µs = 0.2 ms (Granite), 305 × 0.30 = 0.1 ms (OLMoE)
(memo-24 B2); Granite's rest split 3.3 ms small ops + 1.7 to 2.5 ms dispatch from campaign 1's r15 (memo-24 B2).
OLMoE's rest has no device split.

| pool | Granite num, ms now → pool above its floor | OLMoE off | OLMoE num | label |
|---|---|---|---|---|
| **compute: experts** (Q3_K kernel time) | 13.2 → **5.5 to 7.2** | 28.5 → **10.3 to 13.3** | 26.8 → **11.0 to 14.4** | now *device*, floor *estimated* |
| **compute: dense at the byte rate** | 18.8 → 0 to 1.3 (the num-arm excess, below); more only if the dense rate is not the DRAM ceiling | 6.9 → 0 to 0.5 | 7.4 → 0 to 0.5 | *device* / *estimated* |
| **head** | 3.76 → 0 (at its screen floor, `last_3ms.md`) | 3.39 → 0 exact; the untied two-stage head measured ≈ 0 (R62) | 3.1 to 3.4 → 0 | *device* |
| **dispatch / sync** (waits in the four ops + per-node dispatch) | 3.24 + 1.7 to 2.5 → **3.0 + ≤ 2.3** | 1.19 + ≈ 0.5 → **1.1 + ≤ 0.5** | same as off | waits *device*; dispatch *estimated* |
| **small ops** (rest minus dispatch, `SSM_SCAN`, `GLU`, router) | 2.4 to 3.2 + 3.1 + 0.2 + 0.85 → **≤ 1.5** (expert-sum `ADD`s, the slow `MUL`) | 3.7 + 0.4 → **≤ 0.6** | same | *estimated* split |
| **outside the graph + first token** | 0.95 + 0.16 → ≤ 0.5 | 0.80 + 0.45 → ≤ 0.7 | 0.75 + 0.3 → ≤ 0.6 | *device* / *estimated* |
| **faults / residency** | 0 in a clean run; **+2.2 ms (R64, r16 runs) to +9.5 ms (R66)** at the session median | **+1.3 (R64, plateau runs) to +7.4 (R64 r16 runs and R66)** | +0.8 to +2.5 (R64 only) | *device* medians against §1.1 |
| **needed** | **−8.2** clean, −10 to −18 at the median | **−6.3** clean, −10 to −14 | **−4.7** clean, −5.5 to −7 | |

Reading: no single pool covers Granite's 8.2 ms; it needs most of the expert pool (5.5 to 7.2)
plus 1 to 2.5 ms from dispatch, small ops and outside. OLMoE num needs about 40 % of its expert pool, or a third of
it plus the crumbs.

Dense excess on Granite, checked again with clean runs: other dense 18.81 (num, R66 pos 47) and 18.99 / 18.70
(fast halves R64 pos 12 / 47) against 17.95 (off, R64 pos 11, clean) and 17.72 (its fast half): **num is 0.8 to
1.3 ms slower on the same 417 MB** (*device*). Memo-24's 1.1 to 1.5 stands, slightly smaller. Cause unknown.

### 1.3 What the state penalty is made of (*device* data, *estimated* fits)

- **Missing file pages at run start** (`want_mb − res_before.total_mb`, uncapped gated runs): R66 Granite median
  61 MB (0 to 631), OLMoE 332 MB (27 to 1,018). Slopes in §0.3. At the medians: Granite +2.8 ms, OLMoE +11 ms
  (the OLMoE fit's intercept, 44.7, equals the clean floor; Granite's, 51.0 ± 6.6, sits 4.4 above its floor, so
  Granite has a part this variable does not explain). R64: OLMoE 0.015 ± 0.007 (median missing 48 MB); Granite no
  relation (median 7 MB).
- **The hold is memory-limited, not time-limited** (R66 parser table): 84.5 % back at the median, 95 % or more in
  10 of 40 runs; stops are `stall` and `rounds` with the advised MB summed at 2 to 3 times the distinct MB (pages
  evicted again while being read back).
- **Per-token view** (profiled runs at 3 threads and 2,112 MHz, tokens 2 to 32, run fixed effects): one decode
  major fault coincides with **0.26 to 0.44 ms** (Granite num 0.27 ± 0.04, n = 279 tokens; OLMoE off 0.44 ± 0.07,
  n = 310), and one minor fault with 3 to 5 µs (Granite) or 16 to 20 µs (OLMoE). Across scored runs the best
  single predictor of a slow OLMoE run is its decode minor-fault count (3,300 to 4,800 in clean runs, 14,000 to
  29,000 in slow ones; 62 ± 9 µs each in that fit). These are 50 times memo-24's 4 to 6 µs per swap-in. Both can
  be true: memo-24's fit is dominated by the cheap bulk swap-ins of the first token (for example R66 pos 24: 4,335
  majors in token 1 for about 40 ms); the 14 to 25 majors that every later token of a pressured run takes are a
  different event. Granite's slow runs read only 2 to 21 MB of blocks for 400 to 1,000 decode majors, so they are
  not bulk file reads. **I cannot say from these reports what those faults are** (reclaim in the fault path, waits
  on pages under read-back, or other apps taking the big cores at the same moments). TEST 1 is there to split it.
- **Slow without faults exists:** Granite num R64 pos 44: 59.0 ms per token with 71 decode majors and 0.5 MB read,
  cap 2,035, next to pos 45 at 45.5 ms. R66 pos 7: 55.8 ms with 114 majors. Not explained by any recorded counter.
- **Capped clock** (1,555): Granite num R66 pos 40, fault-free, 55.4 ms: +8.8 ms (× 1.19).

## 2. Open levers from memo-21 and memo-24, re-priced

All ms are per token, *estimated*, negative = faster. "Clean" = at the §1.1 floor; "pressure" = an R66-like day.
B = bit-identical (to be shown by the three-setting identity check), N = declared numerics change inside the budget.

| # | lever (origin) | Granite num | OLMoE (off / num) | label | generic across models? | what changed since the memo |
|---|---|---|---|---|---|---|
| A1 | barrier-free work-stealing helpers in `MUL_MAT_ID` (memo-21 N1) | 0 to −3, centre −1.2 | 0 to −6, centre −2.5 | B | yes, a property of the phone; both models, any MoE | bands cut (were 0 to −4 / 0 to −8): plain four threads lost 3.9 to 5.1 tok/s (R62) and that is far more than 666 barriers × 2 µs = 1.3 ms, so the fourth big core is not reliably free; f = 0.3 to 0.7 is my working range (−1.2 to −2.5 Granite, −2.6 to −5.4 OLMoE by 13.2 or 28.5 × f / (3 + f)). Heat is a risk: R62's four-thread runs started hotter. |
| A2 | LUT Q3_K kernel, exact int16 (memo-21 N4) | 0 to −3.2 | 0 to −6.4 | B | yes, both store experts as Q3_K | unchanged, unmeasured. Needs a runtime-derived expert file (+2.6 / +2.8 GB storage, owner). Stacks with A1 only down to the byte floor. |
| A3 | LUT Q3_K kernel, lossy int8 (memo-21 N4) | −3 to −6 | −5 to −11 | N | yes | unchanged. Budget left: Granite 0.176 at `-ub 1` (9.6879 − 9.5120, KleidiAI build, R61); OLMoE 0.203 with `MASS=900`, the full 0.586 without it. |
| A4 | dense-rate pool / L3 warm by a helper (memo-24 B4.1, memo-21 N3) | −0.8 to −1.3 if the num-arm excess is removable; 0 to −3 if the rate is not the ceiling | 0 to −1 | B | yes; larger on dense-heavy hybrids | lower end confirmed again on clean runs (§1.2); upper end still a guess. One microbench decides both (TEST 4). |
| A5 | per-thread pinning (memo-24 B4.2, memo-21 N5) | −0.3 to −1.0, centre −0.5 | −0.1 to −0.5, centre −0.3 | B | yes | pool confirmed: waits 3.24 (Granite) and 1.19 (OLMoE) in clean runs against 0.2 / 0.1 of pure sync. Work is smaller than memo-24 said: the tree already has `cpumask` + `strict_cpu` in `ggml_threadpool_params` (`lab/llama.cpp-c1r26/ggml/src/ggml-cpu/ggml-cpu.c:4408`, one core per worker when strict); only the app's parameter is missing. R62's thread samples put about a fifth of OLMoE's samples on cpus 0 to 3. |
| A6 | graph crumbs: expert-sum fused into the `MUL_MAT_ID` epilogue, the slow `MUL` (memo-24 B4.3) | −0.5 to −1.5, centre −0.8 | −0.2 to −0.6, centre −0.4 | B if the order is kept | yes | unchanged; rest is 4.9 / 4.2 in clean runs. OLMoE's split still unknown (TEST 3). |
| A7 | outside-graph pipelining (memo-21 N5) | 0 to −0.5 | 0 to −0.5 | B | yes | pool measured in clean runs: 0.95 / 0.65 to 0.85. |
| A8 | OLMoE `MASS=900` (+ head 1024), built, parked by R63 §4 | – | num against off: **−1.9 to −2.3 in the expert op (*device*)**; 13.4 % × 28.5 = −3.8 by bytes | N (12.0927 / 12.0976, inside) | rule generic, setting per model | new reading, §0.4. No build work. |
| A9 | Granite skip threshold above 130 (R59: 131 to 144 never screened; 145 over by 0.0083 at `-ub 1`) | −1.2 to −1.6 if about 141 fits (≈ +6 to +8 points of expert bytes × 20.7 ms) | – | N; spends the rest of Granite's budget | rule generic, setting per model | new item, not in memo-21/24. `en.txt` rises faster than wikitext near the cap (STATUS caution). Host screen first (TEST 6). |
| A10 | gate-guided up-row skip (memo-21 (a)) | −0.7 for ≈ +1.6 % | −1.4 for ≈ +1.6 % | N | yes | unchanged, parked; worse per PPL point than A8 / A9. |
| M1 | Granite num: tied head kept out of KleidiAI (memo-24 A3a) | clean 0; pressure **0 to −3, about −1 over days** (155 MB × 0.046 = −7 at most, limited by the MB actually missing: median 61 → −2.8); TTFT −0.1 to −0.5 s | n/a | B in num (to check); N in off | no: tied-head models with head2 only | raised from "0 to −1": every MB of anon given back is priced at the missing-MB slope. Still state-dependent and unmeasured. |
| M2 | routing prior, built as 0007 (memo-21 N2, memo-25) | 0 | clean 0; pressure **0 to −3, about −1 over days** (simulated re-read −33 % at 1.8 GB × a 7.5 to 9 ms penalty) | N (num + prior 12.2454 / 12.2435; 9.5487 / 9.5478: inside) | rule and setting generic; value is not (OLMoE only) | raised from memo-25's 0.2 to 0.7 (R62 state) because R66's state is worse; on an R64-like day still ≈ 0. Leaves OLMoE 0.055 of budget. |
| M3 | anon hold, `MADV_WILLNEED` on hot anon after the gate (memo-24 A3c) | decode ≈ 0; TTFT −0.1 to −0.5 s | decode ≈ 0; TTFT −0.05 to −0.15 s | B | yes | decode value now has evidence for ≈ 0: `VmSwap` before the run does not predict decode time (§0.3). A TTFT lever only. R66 named it next; by the metric it should not be. |
| M4 | `mlock` ≤ 64 MiB (memo-21 N5, memo-24 A3c) | 0 to −0.5 | 0 to −0.5 | B | yes | unchanged; no target known until a census says which pages go (TEST 1). |
| M5 | read the decode set back at the end of the prefill, most-routed slices first, and touch it (R56's "order of page use" candidate, never built) | clean 0; pressure 0 to −1.5 | clean 0; pressure 0 to −4, centre −2; TTFT +0.1 to +0.5 s on such days | B | yes | new pricing of an old candidate: it moves the 1.1 to 1.5 ms/MB from the timed decode into TTFT. Works only as far as the cache can hold the decode set (Granite 1.11 GB yes; OLMoE 2.16 GB often no). Simulator first (TEST 5). |
| M6 | cooler prefill (memo-21 N6) | 0 clean; capped days up to −9 | same | B | yes | owner decision (TTFT 2 to 4 × longer); not recommended while the plateau gate result is "speed not shown" (R64). |
| – | KV / recurrent clear by `MADV_DONTNEED` (memo-24 #1) | done, on (R66) | done, on | B | yes | closed as a lever: TTFT −704 ± 149 ms OLMoE, decode not shown. |
| – | plain four threads, untied head on OLMoE, plateau gate, prefill-cold, keep-warm | – | – | | | closed or not shown (R58, R62, R64). Not re-proposed. |

Not in the table because it is not a speed-up of inference: the benchmark times 32 decodes for about 31 counted
tokens (knowledge.md §6: the last decode's logits are unused, about 3 %, left in by campaign 1). Not running that
decode would print +3.2 % (+0.7 tok/s) with identical text. Owner decision 3 says the metric is unchanged, so I do
not propose it; it is listed so that nobody rediscovers it as a lever.

Sum at the centres, clean state (*estimated*; A1 and A2 share a floor, A4's upper end excluded):
- Granite num: 46.8 − 0.5 (A5) − 1.2 (A1) − 0.8 (A6) − 0.3 (A7) − 1.4 (A9) = **42.6 ms, 22.7 tok/s**. All tops:
  46.8 − 1.0 − 3.0 − 1.5 − 0.5 − 1.6 = 39.2 ms, 24.6. With A2 exact at its top the expert op would sit at its byte
  floor and the total near 37 to 38 ms (25.4 to 26.1).
- OLMoE num (A8 on): 43.3 − 0.3 − 2.5 − 0.4 − 0.3 = **39.8 ms, 24.3 tok/s**. All tops: 35.7 ms, 27.0.
- OLMoE off: 44.9 − 3.5 = 41.4 ms, 23.3 tok/s at the centres.

## 3. Draft TEST list (checks that decide a question)

Cost: S = under half a day of Tom, M = 1 to 2 days, L = a week or more. "Rides" = no extra phone time, goes into
the next kit.

| # | test | where | cost | decides | closing result |
|---|---|---|---|---|---|
| T1 | **State attribution counters.** Per run and per phase (prompt, decode): `getrusage` user and system CPU, voluntary and involuntary context switches; per worker `/proc/self/task/<tid>/schedstat` (run time, run-queue wait); a `mincore` count of the anon ranges (KleidiAI buffer, head copy, screen, recurrent state) at first chunk and at the end. The report's `io_wait_ticks` is 0 in every run and `/proc/vmstat` is unreadable, so these are the readable substitutes (availability on this phone not verified). | device, rides | S | what the +2.2 to +9.5 ms is: CPU taken by other apps (→ A5 with priority), kernel time in faults (→ M1, M2, M5), or anon going out during the run (→ M4 target) | run-queue wait + involuntary switches explain at least half of (wall − floor) in slow runs → pinning / priority first; system time explains it → memory levers first; neither → the unexplained slow runs (R64 pos 44) stay open and are said so |
| T2 | **Per-core bench and fourth-core idleness.** `GGML_CPU_BENCH` Q3_K row pinned once to cpu 7 (X1) and once to cpu 0 (A55), clocks logged; per-cpu busy share during a decode from the thread samples or `/proc/stat` if readable. Memo-21 N1's "zero-code" test whose second half (the pinned bench) was never run. | device, minutes | S | f for A1 | kill A1's A55 part if A55 < 0.25 × the A78's rate; kill A1 if the fourth big core is busy in more than half of the samples and the A55 part is dead |
| T3 | **Block-mode profile**, one run per model with `GGML_CPU_AB_TOKEN` unset (the 14-op table; env only). | device, rides | 0 code | OLMoE's rest (4.2 ms, no split anywhere), the slow `MUL`, dispatch per node on both | build A6 only where one fusable op class is ≥ 0.5 ms; if the rest is spread below 0.3 ms per class, close A6 on that model |
| T4 | **Read-bandwidth microbench** in the bench harness: 64 MB stream with the Q4_0x4 kernel, 1 to 4 threads, A78 only against A78 + X1, with and without a second core streaming; a 3 MB slice cold against warmed from another core. | device, minutes | S to M | A4 (both ends) and the Granite num-arm dense excess | kill the upper end of A4 if 3 A78 threads cannot exceed 27 GB/s; kill the L3 warm if the warmed slice is < 15 % faster |
| T5 | **Simulator run for M5 against M2**: memo-18's `tools/c2_pagecache_sim.py` on the stored traces with (a) today, (b) decode set read back most-routed-first after the prefill, (c) prior λ 0.5 N 16, at 1.6 to 2.4 GB; output MB re-read inside the 32-token decode. | host | S | whether M5 is worth building and whether it makes M2 redundant | build M5 only if the decode re-read falls by ≥ 30 % at 1.8 to 2.2 GB on OLMoE; if (b) ≥ (c), M2 is not needed for residency and its budget stays free |
| T6 | **Granite skip 135 / 138 / 141 / 143 + head2** at `-b 512` and `-ub 1` on R61's KleidiAI build, rule written first ("largest screened threshold inside 9.6879 at both"), with the decode skip counter. | host, one night (*estimated* from R61's five rows in 4.5 h) | 0 code | A9 | close A9 if the largest inside is ≤ 133 or adds < 4 points of skipped expert bytes |
| T7 | **LUT microbench**: one 16-row interleaved Q3_K slice, exact int16 and lossy int8, against the in-register kernel, A78-pinned (memo-21 A1's test, unchanged). | device, minutes | M | A2, A3, and whether the storage question goes to the owner at all | kill exact < 1.10 ×, lossy < 1.40 × |
| T8 | **`MASS=900` from the stored profiles** (done in §0.4; a second look by Tom with the parser's own token filter). | none | 0 | whether R63 §4's "host-side reason" clause is met | lead's decision; the datum is −1.9 to −2.3 ms in the expert op, n = 2 against 1 |

## 4. Draft TRY list (levers to build), ranked by expected ms per unit of work

Expected ms = centre of the band, weighted "clean / pressure" where they differ. Rank is centre ms divided by work.

| rank | lever | work | Granite num | OLMoE | label | generic? | needs first |
|---|---|---|---|---|---|---|---|
| 1 | A8 `MASS=900` back in OLMoE's arm | none (built, gated) | – | −2 (op time, *device* reading) | N, inside | setting per model | lead's call on T8 |
| 2 | A5 per-thread pinning (three workers on fixed big cores; arms {4,5,6}, {5,6,7}, floating) | S | −0.5 | −0.3; plus fewer A55 excursions | B | yes | nothing; ABBA on the wait columns |
| 3 | A9 Granite skip threshold | S (setting) | −1.4 | – | N | setting per model | T6 |
| 4 | M1 Granite tied head out of KleidiAI in num | S (load knob like 0006) | 0 clean, about −1 over days | n/a | B in num | no (tied head only) | host load log 541 → ≈ 394 MiB + identity |
| 5 | A1 barrier-free helpers in the expert op | M | −1.2 | −2.5 | B | yes | T2 |
| 6 | M2 routing prior as a kit arm on OLMoE | none (built) | 0 | about −1 over days, 0 clean | N, inside | value OLMoE only | T5 (it may lose to M5) |
| 7 | A6 expert-sum fusion and the slow `MUL` | M | −0.8 | −0.4 | B | yes | T3 |
| 8 | M5 decode-set read-back at the prefill's end | M | 0 clean, −0.5 pressure | 0 clean, −2 pressure | B | yes | T5 |
| 9 | A7 outside-graph pipelining | S to M (app and llamadart side) | −0.3 | −0.3 | B | yes | nothing |
| 10 | A2 exact LUT kernel + derived expert file | L + owner storage | −1.5 (0 to −3.2) | −3 (0 to −6.4) | B | yes | T7, then the owner |
| 11 | A4 dense rate / L3 warm | M | −1 (0 to −3) | −0.5 | B | yes | T4 |
| 12 | A3 lossy LUT | L + host PPL | −4.5 | −8 | N | yes | T7, budget after A8 / A9 / M2 |
| – | M3 anon hold, M4 `mlock` | S | 0 in the metric | 0 in the metric | B | yes | only if TTFT becomes a goal, or T1 names a target |

Conflicts to settle before a kit: A9, M2 and A3 draw on the same budget per model (Granite 0.176 left; OLMoE 0.203
with A8). A1, A2 and A3 share the expert byte floor. M1 and M5 pay only on pressure days, so their A/B needs the
day's missing MB recorded next to it (it is: `want_mb`, `res_before`).

## 5. Where this draft contradicts earlier memos or the brief

1. **Memo-24's "a decode swap-in costs 4 to 6 µs":** true of the bulk swap-ins around the first token; the majors
   of tokens 2 to 32 in pressured runs coincide with 0.26 to 0.44 ms each (§1.3). The cause is not established.
2. **Memo-24 / R66 "next memory lever: item 3 (anon hold)":** for the 32-token metric the evidence says ≈ 0
   (`VmSwap` before the run does not predict decode time). Missing file MB does. M1, M2 and M5 act on that; M3
   does not.
3. **"OLMoE fault-free floor about 23.2" (brief, memo-24):** that was the fastest half of fault-free tokens. As
   whole clean runs OLMoE num is 21.4 to 23.5, median 22.3; OLMoE off 21.5. Granite's 20.8 holds (20.6).
4. **"OLMoE `MASS=900` gain not shown":** not shown in tok/s; shown in the expert-op time of the same reports.
5. **Memo-25's 0.2 to 0.7 ms for the prior:** priced at R62's state; at R66's state it is up to about −3.
6. **Memo-21's helper band (0 to −8 / 0 to −4):** cut after R62's four-thread result.

## 6. What I could not verify

- Whether `schedstat`, `/proc/stat` and per-thread files are readable from the app on this phone (T1, T2).
- What the per-token majors and the expensive minor faults of pressured runs are (§1.3).
- The cause of the unexplained slow fault-free runs (R64 pos 44, R66 pos 7) and of Granite's num-arm dense excess.
- f for the helpers; the A55's Q3_K rate (one unpinned campaign-1 bench).
- A9's fit: thresholds 131 to 144 were never screened; my "about 141" is a linear reading between 130 and 145.
- The current state of the T-MAC llama.cpp PR #10181: two searches today returned only mirrors without a merge
  badge; unchanged from memo-21 ("could not be confirmed"). It does not bear on our tree (base `620d2225c`).
- That llamadart exposes the threadpool's `cpumask` / `strict_cpu`; I checked only that the fields and the
  per-worker code exist in the tree (upstream since the threadpool PR #8672, per the search result).

## 7. Sources

- Stored device reports r22, r23, r25, r26 and the R-log entries R59 to R66 (`campaigns/02-25toks-5pct/reports/`).
- memo-18, memo-21, memo-24, memo-25, memo-27; knowledge.md §6; STATUS.md.
- llama.cpp threadpool affinity fields: https://cdn04132025.gitlink.org.cn/replica/llama.cpp/commit/42c76d1358021ccdbe8ba89109c143dd7ae166df
  (mirror of "Threadpool: take 2 (#8672)", search result, 2026-11-18); Arm learning path showing `-C` pinning:
  https://learn.arm.com/learning-paths/servers-and-cloud-computing/llama_cpp_streamline/6_multithread_analyze/
- T-MAC: https://arxiv.org/html/2407.00088v2 ; PR page mirror https://github.1485827954.workers.dev/ggml-org/llama.cpp/pull/10181
  (status not shown).
