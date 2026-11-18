# Memo 29: joint research round, one TEST list and one TRY list for Tom

Agent models: claude-fable-5-2 (Bonnie), claude-fable-5-2 (Ebony, run as a stand-in for the `ebony` agent)

Date 2026-11-18. Authors: Bonnie (pricing against this repository's measurements) and Ebony (frontier: papers,
kernel sources, other runtimes). Merged by Bonnie from the two drafts and the two cross-reviews. Repository HEAD
when written: `023c529`.

**Nothing here ran on a phone, and no host benchmark ran.** It is short scripts over the four stored raw reports
(`device/r22_2026-11-15.json`, `r23_2026-11-16.json`, `r25_2026-11-16.json`, `r26_2026-11-18.json`), the repository
parser on r26, reading of memos 15, 18, 21, 24, 25, 27 and R59 to R66, web sources, and arithmetic. Labels:
*device* = read or computed from a stored report (named); *host* = quoted from a host entry; *estimated* =
arithmetic or an in-sample fit; "source" = a published claim not reproduced here. **Every fit is in-sample, post
hoc, not pre-registered, and selects runs on their own counters; the R-log verdicts stand as written.** This memo
proposes; verdicts and pre-registrations are the lead's, and items marked "owner" are the owner's.

Where the two authors still disagree, both positions are in §6.

## 0. Summary

1. **Clean whole-run floors (*device*, cap 2,035):** Granite num 20.1 to 21.2 tok/s, median 20.6 (46.8 ms per
   token); OLMoE off 20.9 to 22.0, median 21.5 (44.9 ms); OLMoE num 21.4 to 23.5, median 22.3 (43.3 ms; one
   session, R64). 25 tok/s is 38.6 ms. **Gap from the floor: Granite −8.2 ms, OLMoE off −6.3, OLMoE num −4.7.**
   Memo-24's 23.2 for OLMoE was the fastest half of fault-free tokens, about 1 tok/s above the run median.
2. **On top of the floor, the median gated run carries +2.2 to +9.5 ms (Granite) and +0.8 to +7.4 ms (OLMoE)**,
   by the day (R64 evening against R66 with apps restarting).
3. **What predicts that penalty is the MB of wanted file pages missing at run start**, robustly on OLMoE (R66:
   0.034 ± 0.008 ms per token per MB, r 0.71, n 19; stable under leave-one-out). On Granite the same fit (0.046,
   r 0.79, n 17) rests on two runs and is not a per-MB price. **The app's swap before the run does not predict
   decode time** on either model. So swap levers move TTFT; file-page levers move the metric.
4. **The first lever to try is new: give the model's mapping the kernel's protected page class (`PROT_EXEC`) and
   read pages back through the mapping**, bit-identical and generic. *Estimated* OLMoE 0 to −6 ms on pressure
   days (about −2 over days), Granite 0 to −1.5; 0 on a clean run.
5. **OLMoE's `MASS=900` shows in the op times** (expert op −1.9 to −2.3 ms in R64, −3.9 in R60; about −1.1 to −2.0
   net, because the dense ops get slower in the skip arm on both models). It was parked by a rule on tok/s; we
   recommend a re-test with the expert-op time as the endpoint, under a new pre-registration.
6. **Position with numbers (*estimated*, clean state, centres):** Granite about 42.8 ms (22.6 tok/s), OLMoE num
   about 39.8 ms (24.3 tok/s), with Bonnie's helper centre (§3; 43.4 and 41.0 ms with Ebony's). Neither is at 25 at the centre; both also need the state penalty near zero in most
   gated runs. Granite reaches 38.6 only with the expert op near its byte floor (exact LUT, or helpers at their
   top) **or** if the 2,035 MHz ceiling turns out not to be thermal (§1.4, a free desk check).

## 1. Where the milliseconds are

Conversion (*device*, r26: `decode_tps × decode_ms / 1000` = 30.87 to 30.91 in every run):
tok/s = 965.6 / (`decode_ms` / 32).

### 1.1 Clean-run tokens (*device*, tokens 2 to 32 of whole profiled runs, additive means)

| ms per token | Granite num (R66 pos 47) | OLMoE off (R64 pos 35) | OLMoE num (R64 pos 23 / 36) |
|---|---|---|---|
| experts `MUL_MAT_ID` compute + wait | 13.18 + 0.79 | 28.49 + 0.46 | 26.49 + 0.57 / 27.13 + 0.62 |
| other dense, with its waits | 18.81 | 6.86 | 7.34 / 7.38 |
| head / router | 3.76 / 0.85 | 3.39 / 0.41 | 3.41 / 0.46 and 3.12 / 0.47 |
| `SSM_SCAN` + wait, `GLU` | 2.61 + 0.52, 0.21 | – | – |
| rest (small ops + dispatch, residual) | 4.91 | 4.23 | 4.25 |
| graph | 45.69 | 43.85 | 42.52 / 42.97 |
| outside the graph; first-token excess per token | 0.95; +0.16 | 0.80; +0.45 | 0.85 / 0.65; +0.63 / 0 |
| wall (tok/s) | 46.8 (20.61) | 45.1 (21.40) | 44.0 / 43.6 (21.93 / 22.13) |
| waits inside the four ops (counted above) | 3.24 | 1.19 | 1.19 / 1.28 |

Scored clean runs agree with the profiled ones (Granite num 20.09 to 21.21 in six scored runs of R64 and R66;
OLMoE off 20.88 to 21.99 in seven; OLMoE num 21.43 to 23.54 in five). Caveats: the runs are selected (under 300
decode majors but for one run, under 5 MB read in the decode); the floor is conditional on the 2,035 cap; OLMoE
num has one session only.

### 1.2 The gap by pool (*estimated* floors: memo-21 §0.2, memo-24 §B2)

| pool | Granite num | OLMoE off | OLMoE num |
|---|---|---|---|
| compute, experts: now → above the byte floor | 13.2 → 5.5 to 7.2 | 28.5 → 10.3 to 13.3 | 26.8 → 11.0 to 14.4 |
| compute, dense (byte-bound) | 18.8 → 0.8 to 1.3 (skip-arm excess) | 6.9 → 0 | 7.4 → 0.5 (skip-arm excess) |
| head | 3.76 → 0 (screen floor) | 3.39 → 0 | 3.1 to 3.4 → 0 |
| dispatch / sync (waits; per-node dispatch) | 3.24; 1.7 to 2.5 → ≤ 3.0; ≤ 2.3 | 1.19; ≈ 0.5 → ≤ 1.1; ≤ 0.5 | as off |
| small ops | ≤ 1.5 | ≤ 0.6 | as off |
| outside the graph, first token | ≤ 0.5 | ≤ 0.7 | ≤ 0.6 |
| faults / residency (session median) | +2.2 (R64) to +9.5 (R66) | +1.3 to +7.4 | +0.8 to +2.6 (R64) |
| **needed from a clean run** | **−8.2** | **−6.3** | **−4.7** |

**The dense ops are slower in the skip arm on both models** (*device*): Granite other dense 18.8 (num) against
17.95 (off), +0.8 to +1.3 ms on the same 417 MB; OLMoE 7.34 / 7.38 against 6.86, +0.5 ms. Same sign on two
models: a property of "less expert work before the dense op", cause unknown. Until it is found, every skip lever
gives back about 15 to 25 % of its expert-op gain.

### 1.3 The state penalty (*device* data, *estimated* fits; filter: gated runs, cap 2,035, 3 threads)

- Decode wall per token against `want_mb − res_before.total_mb`, R66: OLMoE off 0.034 ± 0.008 (r 0.71, Spearman
  0.78, leave-one-out 0.030 to 0.044, median missing 332 MB); Granite num 0.046 ± 0.010 (r 0.79), but **0.020 with
  r 0.41 without the two runs missing most**, median missing 61 MB. R64: OLMoE off 0.015 ± 0.007 (median 48 MB),
  nothing on Granite. Supported statements: OLMoE's decode cost tracks missing file pages at about 1.1 ms per MB
  per run (memo-18's rate); for Granite, "runs missing several hundred MB are slow".
- `VmSwap` before the run: slopes −0.003 ± 0.014 (Granite) and +0.007 ± 0.019 (OLMoE) ms per token per MB. A null
  for swap held before the run, fitted on one pressured session; it does not test swap-outs during the run.
- The hold got back 84.5 % at the median in R66 and advised 2 to 3 times the distinct MB. Two readings: the phone
  cannot hold the set (memory), or pages read back by `posix_fadvise` are unmapped and cold and go first (list
  position; a 2026 write-up reports 1.76 GB restored by `MADV_WILLNEED` reclaimed within 8 s, source below). T1
  and Y1's test tell which.
- Slow OLMoE runs take 14,000 to 29,000 minor faults in the decode against 3,300 to 4,800 in clean ones. Linux
  counts a fault on a cached but unmapped page, or on a page still locked by a read, as minor. The per-fault
  coefficients of our token fits (0.26 to 0.44 ms per major, tens of µs per minor) absorb other things and are
  **not** costs; memo-24's 4 to 6 µs per swap-in stands for the bulk swap-ins around the first token.
- Slow without faults exists (Granite num, R64 pos 44: 59.0 ms per token, 71 decode majors, 0.5 MB read) and
  Granite's R66 fit has an intercept 4 ms above its floor: a part no stored counter explains.

### 1.4 The clock ceiling (Ebony; not in either earlier memo)

In r22, r23, r25 and r26 the kit's `cap=` field reads 2,035 in 154 of 191 runs across heat 0.67 to 1.08; 1,785,
1,670 and 1,555 appear when hot and 2,150 three times; cpus 4 to 6 read 2,112 at the same time (*device*). The
hardware maxima are 2,419 (A78) and 2,841 (X1). If that ceiling were a policy an app can lift, the compute-bound
expert op alone scales by 2,112 / 2,419: Granite −1.7 ms, OLMoE −3.4 ms (*estimated*), against more heat. What
the field measures has to be read from the kit code first (T4).

## 2. TEST list (checks that decide a question), in the order we would run them

Cost: S = under half a day of Tom, M = 1 to 2 days. "Rides" = goes into the next kit, no extra phone time.
Who: Tom builds or reads code, the owner runs the phone, the lead runs host jobs and decides.

| # | test | who | cost | decides | closing result |
|---|---|---|---|---|---|
| T1 | **State attribution in the kit.** Per run and phase: `getrusage` user / system CPU and voluntary / involuntary context switches; per worker `/proc/self/task/<tid>/schedstat`; at run start, first chunk and end: model pages resident (`mincore`) against mapped (`Rss` of the mapping in `smaps`, or `pagemap`), and `mincore` of the anon ranges (KleidiAI buffer, head copy, screen, recurrent state). Readability on this phone not verified (`io_wait_ticks` is 0 and `/proc/vmstat` unreadable in every stored run). | Tom, owner; rides | S | what the +2 to +9 ms is: CPU taken by others, kernel time in faults, pages cached but unmapped, or anon leaving during the run | run-queue wait + involuntary switches explain half of (wall − floor) in slow runs → Y4 and thread priority first; resident ≫ mapped at run start → the hold must go through the mapping (Y1); neither → say the remainder is unexplained |
| T2 | **Does the phone accept an executable file mapping?** `mmap` 1 MB of the model with and without `PROT_EXEC` from the external and the internal files folder, print errno and the two `mountinfo` rows. | Tom, owner; rides | S (about 30 lines of app code) | Y1 | the internal folder refuses too → Y1's page-class half is dead on this phone; its read-back half remains |
| T3 | **Block-mode profile**, one run per model with `GGML_CPU_AB_TOKEN` unset (the 14-op table). | owner; rides | env only | OLMoE's rest (4.2 ms, no split anywhere), the slow `MUL`, dispatch per node | build Y5 only where one fusable op class is ≥ 0.5 ms; spread below 0.3 ms per class → Y5 closed on that model |
| T4 | **Clock-ceiling desk test.** What `cap=` reads (kit code); then every stored report of both campaigns: cap against heat, phase and time since load; when was a cap above 2,150 recorded; is `scaling_cur_freq` ever below the cap in a decode. | Tom (code), researchers (table) | desk, hours | whether §1.4 is a pool at all | cap steps with heat everywhere → thermal, priced 0, closed; a fixed ceiling independent of heat → one kit line trying a hint session, then price |
| T5 | **Per-core bench and fourth-core idleness.** `GGML_CPU_BENCH` Q3_K row pinned to cpu 7 and to cpu 0, clocks and `cap_after` logged; busy share of the fourth big core during a decode. | Tom, owner; minutes | S | f for the helpers (Y8) | A55 < 0.25 × the A78's rate → no A55 helpers; fourth big core busy in more than half of the samples as well → Y8 closed |
| T6 | **Host numerics frontier, one design, rule written first** ("largest setting inside the budget by at least 0.02 at `-b 512` and `-ub 1`", KleidiAI build of R61, `en.txt` reported per setting). Stage 1, no code: Granite flat threshold 135 / 138 / 141 / 143 + head2. Stage 2: per-layer sensitivity on wikitext-2 **train** (60 chunks) for both models, then a per-layer table at matched skipped bytes. Optional if nearly free: scalar emulation of an `hmask`-only expert tier. | Tom (scripts), lead (host: a night, then about a day) | 0 code, then S to M | Y7; whether a per-layer table beats the flat rule; whether B2 is worth a kernel | stage 1: largest inside ≤ 133 or < 4 points of bytes → flat raise closed. Stage 2: table buys < 1.3 × the flat rule's bytes at equal PPL → per-layer closed. Tier: no better than skip-only by 1 ms at equal PPL → closed |
| T7 | **Dense excess in the skip arm.** In-run A/B per token, skip on against off, reading the other-dense time (if `GGML_CPU_AB` can drive that knob; Tom to confirm); and a read-bandwidth microbench (64 MB stream, Q4_0x4 kernel, 1 to 4 threads, A78 only against A78 + X1; a 3 MB slice cold against warmed from another core). | Tom, owner; rides + minutes | S to M | the 0.5 to 1.3 ms that skip levers give back; the dense-rate pool and the L3 warm (Y11) | no excess in the in-run A/B → an artefact of run selection, drop it; 3 A78 threads cannot exceed 27 GB/s → dense rate is the ceiling, Y11's upper end closed; warmed slice < 15 % faster → L3 warm closed |
| T8 | **`MASS=900` desk re-read** with the parser's own token filter over r22, r23, r25 (expert op and wall, num against off). | Tom | desk, an hour | the endpoint and effect size for Y3's pre-registration | lead's decision |
| T9 | **Read-back in memo-18's simulator** (decode set read back most-routed-first after the prefill, against today and against the prior), 1.6 to 2.4 GB. Plain LRU: an upper bound only. | Tom or researchers; host | S | whether Y1's in-run arm is worth its code; whether Y9 is redundant | even the upper bound is < 30 % less decode re-read at 1.8 to 2.2 GB on OLMoE → in-run arm dropped |
| T10 | **LUT microbench**: one 16-row interleaved Q3_K slice, exact int16 and lossy int8 tables, against the in-register kernel, A78-pinned (memo-21 §A1, unchanged). | Tom, owner; minutes | M | Y10, the lossy LUT, and whether the storage question goes to the owner | exact < 1.10 × and lossy < 1.40 × → both closed |
| T11 | **Head row-norm check**: k-means the dequantised head rows (C = 2,000), print median residual norm / row norm. | Tom or researchers; host | minutes | the certified head (Ebony's C1) | ratio above about 0.5 → closed without a model run (see §6.5) |

Answered while reviewing and dropped: double caching of the model through the external path (r26's first run:
`cached=2614MB` with 2,007 MB of the model resident, no); dma-buf feasibility and zram compressibility (their
lever is withdrawn, §4).

## 3. TRY list (levers to build), ranked by expected ms per unit of work

All ms per token, *estimated*, negative = faster; "clean" = at the §1.1 floor, "pressure" = an R66-like day.
B = bit-identical (three-setting identity check required), N = declared numerics change inside the budget at
`-b 512` and `-ub 1`.

| rank | lever | who, work | Granite num | OLMoE | label | generic? | needs first |
|---|---|---|---|---|---|---|---|
| Y1 | **Residency pair.** (a) The model mapping gets `PROT_EXEC`, switched per run with `mprotect` so ABBA needs no reload; (b) pages are read back **through the mapping** (advice, then one touch per page) at the end of the gate wait and, as a second arm, the decode set at the end of the prefill. Model file moved to internal storage (same bytes and hash). Endpoints: MB missing at run start, advised MB per distinct MB, then decode ms. | Tom M; owner: one file move | clean 0; pressure 0 to −1.5, about −0.5 over days | clean 0; pressure 0 to −6, about −2 over days; TTFT +0.05 to +0.5 s on such days | B | yes: a property of the kernel, any model, any mmapped runtime | T2; lead's (Bonnie: and owner's) word on §6.2 |
| Y2 | **Granite num: tied head kept out of KleidiAI** (load knob like 0006); −155 MB of anon the num arm never reads. | Tom S | clean 0; about −0.5 over days, up to −3 on a day missing hundreds of MB; TTFT −0.1 to −0.5 s | n/a | B in num (to check); N in off | **no**: tied head + head2 only; nothing for the trust purpose | host load log 541 → ≈ 394 MiB, identity |
| Y3 | **OLMoE `MASS=900` re-test** under a new pre-registration, primary endpoint the expert-op time in ABBA, tok/s secondary. | lead; no build | – | op −1.9 to −2.3 (R64) and −3.9 (R60), *device*; net about −1.1 to −2.0 | N (12.0927 / 12.0976, inside; costs 0.38 of 0.586) | rule yes, setting per model | T8; lead's reading of R63 §4 |
| Y4 | **Per-thread placement**: arms floating on the mask (today), strict {5, 6, 7}, strict {4, 5, 6}; an unequal static share for the X1 as a later arm (MNN's idea). The tree already has `cpumask` + `strict_cpu`; only the app's parameter is missing. | Tom S | clean −0.3 to −1.0; pressure +1 to −1 | clean −0.1 to −0.5; pressure +1 to −1 | B | yes | T1 in the same kit (§6.3) |
| Y5 | **Graph crumbs**: weighted expert sum fused into the `MUL_MAT_ID` epilogue in the same order; the slow `MUL`. | Tom M | −0.5 to −1.5 | −0.2 to −0.6 | B if the order is kept | yes | T3 |
| Y6 | **Outside-graph pipelining** (detokenise, callback, UI after the argmax on another thread). | Tom S to M (app, llamadart) | 0 to −0.5 | 0 to −0.5 | B | yes | nothing |
| Y7 | **The winner of T6**: a higher flat Granite threshold, or a per-layer table on either model. | Tom S (setting) or M (table) | gross −1.2 to −1.6 at about 141, net about −1.0 to −1.3; spends all of 0.176 | per-layer table instead of `MASS=900` at the same PPL: 0 to −2.4 | N | rule yes; setting or table per model | T6 |
| Y8 | **Barrier-free work-stealing helpers** in the expert op (memo-21 N1). | Tom M | 0 to −3 | 0 to −6 | B | yes: a property of the phone | T5; centre in §6.4 |
| Y9 | **Routing prior** (built, 0007) as a kit arm. | lead; no build | 0 | clean 0; pressure 0 to −3; costs 0.148 of the 0.203 left | N (12.2454 / 12.2435; 9.5487 / 9.5478) | rule yes; value OLMoE only | only if Y1 fails or T9 shows it adds to Y1 |
| Y10 | **Exact LUT Q3_K kernel** + runtime-derived expert file (+2.6 / +2.8 GB). | Tom L; owner: storage | 0 to −3.2 | 0 to −6.4 | B | yes | T10, then the owner |
| Y11 | **Dense rate / L3 warm by a helper.** | Tom M | 0 to −3 | 0 to −1 | B | yes | T7 |
| Y12 | **Clock ceiling**, only if T4 finds a ceiling that is not thermal. | unknown | −1.7 | −3.4 | B | yes | T4 |
| – | Behind their tests, not ranked: `hmask`-only expert tier (N; Granite 0 to −0.7, OLMoE 0 to −1.5; new kernel), lossy LUT (N; −3 to −6 / −5 to −11; PPL unknown), certified head (token-identical, no logits; 0 to −2.5 if it passed). | | | | | | T6, T10, T11 |

Sum at the centres, clean state (*estimated*; Y8 and Y10 share the expert byte floor):
- Granite num: 46.8 − 0.5 (Y4) − 0.8 (Y5) − 0.3 (Y6) − 1.2 (Y7 net) − 1.2 (Y8, Bonnie's centre) = **42.8 ms,
  22.6 tok/s**; with Ebony's helper centre 43.4 ms, 22.2. All tops about 39.5 ms, 24.4.
- OLMoE num (Y3 on): 43.3 − 0.3 − 0.4 − 0.3 − 2.5 (Y8, Bonnie) = **39.8 ms, 24.3 tok/s**; with Ebony's centre
  41.0 ms, 23.6. All tops about 36 ms.
- Y1, Y2 and Y9 do not lower these; they bring the median run toward them.

**One budget per model for the numerics claimants** (absolute PPL, KleidiAI build): Granite has 0.176 left after
skip 130 + head2; OLMoE 0.586, of which `MASS=900` takes 0.38 and the prior 0.148 more. Y7, Y9, the `hmask` tier
and the lossy LUT all draw on it; on Granite only one of them fits.

## 4. Withdrawn, parked or closed in this round

| item | status | reason |
|---|---|---|
| anon hold, `MADV_WILLNEED` on hot anon (memo-24 item 3; R66 "next memory lever") | parked: TTFT only | swap before the run does not predict decode time (§1.3) |
| dense anon pinned in a dma-buf (Ebony A2); derived file for repacked dense weights (A3) | withdrawn by Ebony | same evidence; pinning can only push reclaim onto the model's file pages. Would also have pinned Granite's 155 MB of unread head copy, which Y2 frees |
| `mlock` 64 MiB, pipe pin 64 MiB | parked | no target until T1's census |
| read-back alone at the prefill's end (Bonnie M5) | merged into Y1 | on plain pages a touch protects for one scan only |
| cooler prefill (memo-21 N6) | owner decision, not recommended now | plateau gate "speed not shown" (R64) |
| speculative decoding, five 2025-2026 MoE papers | stays closed (R20) | every reported gain is from avoided expert transfers or idle batch capacity; this phone is compute-bound above batch one |
| Hexagon NPU backend; `MADV_POPULATE_READ`; userfaultfd; MGLRU; THP; memcg `memory.min` | dead on this phone | Hexagon 780 older than the supported versions (not checked); kernel 5.4.274 |
| the benchmark's final unused decode (about 3 %, knowledge.md §6) | not a lever | owner decision 3: the metric is unchanged. Listed so nobody rediscovers it |

## 5. Corrections to earlier memos and to our own drafts

1. Memo-24 / brief: "OLMoE fault-free floor 23.2" → 22.3 as a run median (num), 21.5 (off). Granite 20.8 → 20.6.
2. Memo-24 item 3 and R66's "next memory lever": for the 32-token metric the evidence is ≈ 0.
3. Memo-25's 0.2 to 0.7 ms for the prior was priced at R62's state; at R66's it is up to about −3.
4. Memo-21's helper band (0 to −8 / 0 to −4) is cut after R62's four-thread loss.
5. "OLMoE `MASS=900` not shown": not shown in tok/s (SE 1.0 tok/s ≈ 2.4 ms); shown in the expert-op time in two
   sessions, with sizes differing by two.
6. Bonnie's draft: a per-MB price for Granite (0.046) is not supported at the median; "the hold is
   memory-limited" is one of two readings; the per-fault coefficients are not costs; `MASS=900` was ranked by
   the op gain, the net is smaller.
7. Ebony's draft: baselines were memo-21 / memo-24's (floors 46.5 / 41.7, fault excess 10 to 11 / 4 to 5);
   "the clock is thermal, a hint is worth 0" is withdrawn pending T4; A2 and A3 withdrawn.
8. Campaign 1 evidence that bears on a per-layer table and was not in Ebony's draft: static per-layer expert
   counts on Granite were falsified (uniform beat both splits, `reports/05_compute_less.md`), and on OLMoE the
   `SUM` and `MASS` rule forms lie on one curve (R51).

## 6. Where the authors still disagree

1. **Why keep-warm failed in R56, and so how much `PROT_EXEC` buys.** Ebony: on 5.4 a plain file page's
   reference bit is ignored on the active list, so touching plain pages was structurally weak; the exec class
   changes the mechanism. Bonnie: on the inactive list a referenced plain mapped page is kept and then activated,
   so R56 lost on timing (scans faster than its 10 s passes); the exec class buys one or two extra list rotations
   per reference, not immunity, and it protects mapped pages only, never what `posix_fadvise` read back. Both
   agree on the lever, its price band and that only its device test settles it.
2. **Whether Y1 needs the owner.** Ebony: this is what the kernel does for every shared library; no rule-5
   question, the lead's call. Bonnie: 2.7 to 3.1 GB marked as code shifts reclaim onto other apps and, with swap
   slots gone, onto lmkd kills; that is the question memo-15 left to the owner, so the owner should hear it
   before a claim round. Neither thinks it breaks rule 5 as written.
3. **Per-thread pinning (Y4).** Bonnie: clean-state pool is measured (waits 3.2 and 1.2 ms against 0.2 and 0.1
   of pure sync), work is S, run it as an ABBA arm in the same kit as T1. Ebony: on a pressure day a strictly
   pinned worker preempted by another process stalls all three at the next of 666 barriers, and {4, 5, 6} gives
   up the X1 (about +0.9 ms Granite, +1.7 ms OLMoE of expert compute); centre 0 on such days, and decide after T1.
4. **Helper centre (Y8).** Bonnie: f = 0.3 to 0.7 (Granite −1.2, OLMoE −2.5); the profiles show the X1 idle or
   slow on some tokens (`mhz7` 844 to 1,785 while `mhz4` is 2,112), so a big core is free at least sometimes.
   Ebony: no datum for f, R18 and R62 are both against it, and a fourth active core is the likeliest way to step
   the cap down; centre f = 0.15 (−0.6, −1.3) until T5.
5. **The certified head.** Ebony: P(pass) about 0.2, one host hour with real hidden states. Bonnie: with 25 rows
   per cluster in 1,536 to 2,048 dimensions the cluster radius stays near the row norm and exceeds any winning
   logit's margin; P below 0.05; T11's row-norm check closes it in minutes.
6. **Per-layer table prior (Y7, OLMoE).** Ebony: 0 to −4 ms (Ban&Pick reports uneven layer sensitivity, by task
   accuracy, on other models). Bonnie: 0 to −2.4 with P about 0.25, by §5.8 and by `MASS=900` realising 55 to
   60 % of its byte arithmetic in the op time.
7. **The clock ceiling.** Ebony: a value that does not move from heat 0.67 to 1.08 does not look like a thermal
   step; worth a desk test before anything else in the TRY list is sized. Bonnie: 2,150, 2,035, 1,785 and 1,555
   all appear and campaign 1 saw the same two clocks at the end of its claim runs, so a vendor thermal table is
   the likely reading; P(an app can lift it) about 0.1. Both want T4 because it is free.
8. **`MASS=900` (Y3).** Agreed on the re-test and its endpoint. Ebony would not call it rank 1 except in the
   sense of "no build"; Bonnie ranks it by ms per work. The lead decides whether R63 §4 allows it.

## 7. What neither of us could verify

- The vendor kernel's reclaim code (`5.4.274-qgki`); only upstream v5.4 `mm/vmscan.c` was read (Ebony fetched the
  file; Bonnie's fetch returned a summary with short excerpts).
- Mount options and SELinux behaviour for an executable data mapping on this phone (T2).
- Readability of `schedstat`, `smaps`, `pagemap`, `/proc/stat` from the app (T1, T5).
- What the kit's `cap=` field measures and whether an app can change the ceiling (T4).
- Whether llamadart exposes the threadpool's `cpumask` / `strict_cpu`; whether `GGML_CPU_AB` can drive the skip knob.
- f for the helpers; the A55's Q3_K rate; the cause of the dense excess in the skip arm and of slow runs without
  faults.
- That Granite thresholds 131 to 144 behave linearly between 130 and 145 ("about 141" is an interpolation).
- llama.cpp pull-request states cited by Ebony (search summaries); T-MAC PR #10181's merge state (mirrors only).

## 8. Sources

- Linux v5.4 `mm/vmscan.c` (`page_check_references`, `shrink_active_list`):
  https://raw.githubusercontent.com/torvalds/linux/v5.4/mm/vmscan.c (fetched 2026-11-18)
- Android 15 sepolicy, `untrusted_app_all.te` (`execute` on `app_data_file`, with `auditallow`):
  https://android.googlesource.com/platform/system/sepolicy/+/refs/tags/android-15.0.0_r1/private/untrusted_app_all.te
- BigMoeOnEdge, Android memory notes (2026-07-21; Android 16, kernel 6.12, one informal source):
  https://raw.githubusercontent.com/Helldez/BigMoeOnEdge/master/docs/android-memory.md
- Ban&Pick: https://arxiv.org/html/2509.06346v2 ; EAC-MoE (abstract): https://aclanthology.org/2025.acl-long.633
- SliceMoE: https://arxiv.org/html/2512.12990v3 ; CSV-Decode: https://arxiv.org/html/2511.21702v1
- MoE speculative decoding (abstracts): https://arxiv.org/abs/2505.19645 ; https://www.alphaxiv.org/abs/2607.24434v1
- MNN-LLM (search summary): https://arxiv.org/pdf/2506.10443 ; phone core-count study: https://arxiv.org/html/2410.03613v3
- llama.cpp threadpool affinity (mirror of #8672, search result):
  https://cdn04132025.gitlink.org.cn/replica/llama.cpp/commit/42c76d1358021ccdbe8ba89109c143dd7ae166df
- T-MAC: https://arxiv.org/html/2407.00088v2
- Repository: `device/r22`, `r23`, `r25`, `r26` raw reports; R51, R56, R58 to R66; memos 15, 18, 21, 24, 25, 27;
  `reports/05_compute_less.md`; knowledge.md §6; `lab/llama.cpp-c1r26/ggml/src/ggml-cpu/ggml-cpu.c:4408`.
- Working files of this round (session scratchpad, not tracked, will not survive): the two drafts, the two
  reviews and the scripts `decomp.py`, `tokfit.py`, `runfit.py`, `missfit.py`, `ebony_check.py`, `ebony_cap.py`.
  Fit definitions are in §1.3 so they can be redone from the stored reports.
