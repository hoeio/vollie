# Memo 15: the cost model refitted with an I/O term and a clock term (after R52), predictions for kit r19, residency

Agent model: claude-opus-5-5

Date 2026-11-14. Author: Bonnie (research). **No measurement was made for this memo.** I read the device JSON of
campaign 1 (`reports/device/r1…r16`, `r13b`, `r14b` at the repository root) and of R52
(`reports/device/r17_2026-11-14.json` in this campaign) with my own short read-only code (scratch, outside the
repository). I ran no repository script. Labels: *device* (read from a named report and run; "run i" is the index in
the report's `runs` list), *device fit* (a least-squares fit to device numbers; the scatter is given), *host* (from
memo-13's traces), *source* (read from code at a named path), *estimated* (arithmetic). Field meanings follow
`tools/research_parse.py`: `mem_before`/`mem_after` give system `MemAvailable`, `Cached`, `SwapFree` and the app's
`RssAnon` (`anon`), `RssFile` (`file`) and `VmSwap` (`swap`); `mhz_after` is one sample at run end (A78 = cpu4-6,
X1 = cpu7); `heat` is the gate's headroom before the run (campaign 1), `heat_after` the headroom after it (r17).
Inputs: memo-08, memo-13, the R52 entry, R9-R49 in `reports/03_final.md`, knowledge.md.

## 0. Short version

1. **Memo-08 had no I/O term and no clock term.** It priced every byte at DRAM speed and every op at 2,112 MHz and
   called the band "fault-free by definition" (§8 item 5), and then gave that band as the R52 prediction anyway. It
   also treated "experts 2.77 GB against a 2.1-2.4 GB cache" (§6, §9.3) as a risk without seeing that the
   official prefill touches 95 % of the expert file, and its §9.3 claim that OLMoE needs about 0.3 GB less memory
   than R49's Granite left out OLMoE's 285 MB KV cache (§4.3). R52 printed 3.34 tok/s against a band of 12.1 to 15.0.
2. **Refitted model** (one decode token): T = C(2,112)·[(1 − s) + s·2,112/f] + κ·M(c) + outside, where f is the
   A78 clock, s the clock-bound share, M(c) the MB re-read from flash per decode token at an effective cache of c GB
   for this file (memo-13's LRU table), and κ the cost per re-read MB.
   - **Clock, *device fit*** (Granite, campaign-1 shipping stack, 16 fault-free per-run medians, r15/r16,
     1,555-2,112 MHz): **T = 19.1 + 27.1 · (2,112/f) ms**, residual sd **0.9 ms**, so s = 0.59 ± 0.04.
   - **I/O with expert read-ahead, *device fit*** (Granite, r15/r16, 614 faulting decode tokens): **+0.065 ms per
     fault event** (major + minor), sd 16 ms per token; 0.057 ms (sd 6.6) on r16's official tokens at 2,112 MHz.
     Run level, 21 official runs at 2,112: **decode ms/token = 71.6 − 11.8 · RssFile_before (GB)**, sd 2.8, flat
     at 46-48 above 1.95 GB.
   - **I/O without read-ahead**: campaign 1 R29 (r7, eio 0) about **0.22 ms per major fault** (*device*-derived);
     R52 implies **κ_sync ≈ 3.5 ms per MB re-read (2.7-5.0), i.e. about 0.29 GB/s, about 0.55 ms per major fault
     (0.25-1.1)** (*estimated* from *device* totals and memo-13's LRU bytes).
3. **R52 is reproduced** (not predicted; κ was fitted on it, and two independent checks pass): compute at 1,075-1,440
   MHz is 92-120 ms (the fastest *device* token, 116 ms at 1,075 MHz, bounds it), I/O 140-200 ms, sum 250-310 against
   the *device* 251-358. About **60-70 % of R52's token is I/O, 30-40 % compute, and a third of that compute is the
   clock cap.** The check that does not depend on the fit: early tokens (1-8) carry 2.3-2.95 times the I/O of late
   ones, and memo-13's LRU gives 2.6 to 3.3 times at 1.8-2.1 GB of cache.
4. **r19 table** in §3. Resident and ≥ 2,000 MHz: (a) 66 ms, 14.5 tok/s; (b) 48 ms, 19.9; (c) 48 ms, 19.8. On an
   R52-like day: (a) about 3.7-4.3, (b) 7.4-9.2, (c) 7.8-9.8 tok/s. **I expect the R52-like memory state (not
   resident) with probability about 0.8-0.9, at a mixed clock.**
5. **Residency.** Granite was fault-free only when `Cached` before the run was at least about 2.70 GB. Scaled by the
   working-set difference, OLMoE needs about **2.86 GB `Cached` (2.75-3.0)** and a cool phone (headroom ≤ 0.70 for ≥
   2,000 MHz). In the 19 recorded sessions (18 of campaign 1 plus R52), only one session reached that before a run
   (r16, 8 of 21 runs); r15 reached 2.91 once, after its last run. **P(OLMoE fault-free in a given session) ≈ 0.10
   (0.05-0.35 depending on the open RssFile question).** No runtime lever inside the rules closes a 0.4-1.1 GB gap.
   Lazy KV pages are worth up to 0.27 GB. A full anonymous or pinned copy would close the gap only by forcing the
   system to kill other apps, which needs the owner's ruling.
6. **The 25 tok/s question is now conditional on the phone's state.** Memo-08's kernel numbers describe a resident,
   ≥ 2,000 MHz session and nothing else. On a non-resident day even a perfect 38 ms compute token prints about 10
   tok/s. "Median of 10 plain runs" measures the day as much as the runtime.

## 1. What memo-08 got wrong, and the rebuilt model

### 1.1 Plainly

- **No I/O term.** Memo-08 priced bytes at 0.040-0.150 ms/MB, which are DRAM and kernel rates. It noted the risk
  that the file would not be resident (§6, §8 item 5, §9.3) but kept it outside the band. A band that holds "if
  the model is resident", given without saying when the phone is in that state, is not a prediction of a phone run.
  R52's own §6 hygiene rule did catch the fault-bound runs (the round is void), but the number was missed by four
  times.
- **No clock term.** Every band was written for 2,112 MHz. Campaign 1 had already shown that the clocks cap at
  1,555 MHz on warm days (R27). R52 ran at 1,075-1,440 MHz.
- **Wrong picture of the working set.** Memo-08 §6 reasoned "each token reads 1/8 of the experts". The official
  benchmark's 156-token prefill touches 968 of the 1,024 expert slices (memo-13, *host*), so the benchmark's working
  set is the whole expert file, 2,725 MB.
- **Wrong memory comparison.** Memo-08 §9.3 put OLMoE's total demand (2.77 GB experts + 0.24 GB repack copy) about
  0.3 GB below R49's Granite (2.6 + 0.55-0.7). It left out the KV cache. At the app's context of 4,096, OLMoE's MHA
  KV cache (16 layers × 2 × 2,048 values × 1.0625 B q8_0) is **285 MB** (*estimated*) against a few tens of MB for
  Granite's 4 attention layers. The app's RssAnon agrees: 0.53-0.61 GB in r17 with repack on, the same as
  Granite's 0.60-0.63 in r16 (*device*). **Both models need about the same total memory.**

### 1.2 The model

> T_token = C(2,112) · [(1 − s) + s · 2,112 / f] + κ · M(c) + t_out

- C(2,112): fault-free in-graph ms at 2,112 MHz (memo-08's bands, recentred for row (a) in §3).
- s: share of C that scales with 1/f.
- M(c): MB re-read from flash per decode token, as a function of c, the cache available to this file. I take it from
  memo-13's LRU table (repack on), averaged over the 32 decode tokens (8 early + 24 late, *estimated*): c = 1.2 GB:
  123 MB; 1.5: 89; 1.8: 54; 2.1: 33; 2.4: 19.5; ≥ 2.73: 0.
- κ: ms per re-read MB. It depends on whether the runtime reads ahead (§1.4).
- t_out: about 0.4-0.8 ms outside the graph (memo-08 §4).
- Printed tok/s ≈ 0.995 × 31,000 / (32 · T + 12) (memo-08 §4). It checks on r17 run 1: decode_ms 8,556 → 3.62
  against 3.61 printed.

### 1.3 Clock term (campaign 1, *device fit*)

The per-token `cpu-prof` lines of r15 and r16 carry their own `mhz=` field, so the clock is known per token. For each
run and clock I took the median of fault-free tokens (majflt 0, minflt < 100, token ≥ 4):

| clock (MHz) | per-run medians, ms/token (*device*) |
|---|---|
| 2,112 | r15 runs 3/4/5: 47.3, 47.1, 46.9; r16 runs 5, 8, 9, 12, 13, 16, 17: 46.0, 45.3, 45.3, 45.2, 46.6, 45.5, 46.8 |
| 1,881 | r15 runs 5/6: 49.2, 47.9 |
| 1,670 | r15 runs 6/7: 53.1, 55.0 |
| 1,555 | r16 run 2: 55.5; r15 runs 7/8: 57.2/57.4, half the tokens with `Q3K_KERN` off, corrected by −1.75 (R47 −3.53/2) to 55.6 |

**Fit: T = 19.1 + 27.1 · (2,112 / f) ms, residual sd 0.86 ms, slope standard error 1.7.** So 59 % of the Granite
shipping token scales with the clock, and the rest (repacked dense products at the memory-bandwidth limit) does not.
That is what R27 said in words. Predicted values: 46.1 at 2,112; 55.8 at 1,555; 58.7 at 1,440; 72.2 at 1,075.

**Clock against thermal headroom** (all runs of the 19 reports, A78 MHz at run end, *device*):

| headroom | runs | median MHz | share ≥ 2,000 |
|---|---|---|---|
| < 0.65 | 53 | 2,112 | 0.94 |
| 0.65-0.70 | 40 | 2,112 | 0.88 |
| 0.70-0.75 | 45 | 2,112 | 0.73 |
| 0.75-0.80 | 36 | 1,613 | 0.42 |
| 0.80-0.95 | 41 | 1,555 | 0.02 |
| ≥ 0.95 (r17 only) | 10 | 1,075-1,440 | 0 |

The r17 values are headroom after the run; campaign 1's are the gate's value before the run. The step is clear:
**at ≥ 2,000 MHz you need a gate at ≤ 0.70; above 0.80 the cap is 1,555 MHz; at ≥ 0.95 it is 1,075-1,440.**

### 1.4 I/O term (campaign 1, *device fit*)

**With campaign 1's expert read-ahead (`EXPERT_IO=1`, r15/r16).** A decode token's excess over the same run's
fault-free median at the same clock was regressed on that token's fault counts. One token is excluded, r16 run 5
token 1 (6,332 majflt costing 80 ms, 13 µs each: swap-ins of the SSM state, not flash):

| set | n tokens | ms per major | ms per minor | residual sd | one coefficient on (major + minor) |
|---|---|---|---|---|---|
| all | 614 | 0.004 | 0.066 | 16.3 | **0.062** (sd 16.4) |
| 2,112 MHz only | 257 | 0.044 | 0.074 | 12.4 | **0.072** (sd 12.5) |
| r16 official, 2,112 | 65 | 0.047 | 0.063 | 6.6 | **0.057** (sd 6.6) |

The cost sits in the **minor** faults, at 60-75 µs each. With `MADV_WILLNEED` the reads are already in flight, so the
compute thread's fault finds the page in the cache but locked for I/O, waits, and is counted as minor. **With
read-ahead on, majflt undercounts I/O. Minor faults above the clean level (about 3.6-3.9 k per official run, r16)
are the I/O instrument.** This also explains r9-r14: 9-15 tok/s official runs with only 2-9 k majflt but 70-140 k
minflt.

**Prefill (TTFT), r15/r16 official runs at 2,112:** TTFT = 3.33 s + **26.5 µs per fault event**, sd 0.65 s (n 21).
That is 2.5 times cheaper per event than in decode, because the prefill reads slices in order and the read-ahead
streams. This is why R49's prefill fault count did not decide a run (R49 correction note).

**Without read-ahead (campaign 1 R29, r7, `EXPERT_IO=0`, 1,555 MHz).** Side A: 178.8 ms/token, 467 majflt and 1,542
minflt per token. The clean reference at 1,555 is 74.6 (r6 control, R27). The excess is 104 ms, i.e. **0.22 ms per
major fault**. The same day with read-ahead on (side A of (i)): 111.6 ms, 26 majflt, 904 minflt. That is 37 ms, or
0.040 ms per event. **Read-ahead cut the I/O cost of the same routing by 2.8 times.** R12's overall 6.1 → 11.8 tok/s
is the same effect.

**Run level (Granite, r15 + r16 official 32, 21 runs at 2,112, *device fit*):**

- **Against RssFile before the run:** decode ms/token = 71.6 − 11.8 · RssFile (GB), sd 2.8.
  - RssFile ≥ 1.95 GB: 46.0-47.9 ms (20.1-20.95 tok/s; 9 of 10 runs). The exception is r16 run 17, 51.7 ms, with
    55.8 k majflt and SwapFree 113 MB.
  - 1.5-1.9 GB: 46.8-60.2 ms.
  - < 1.1 GB: 57-68.5 ms.
- **Against `Cached`:** ms = 122.6 − 25.8 · Cached (GB), sd 3.9. It is weaker, because `Cached` also holds other
  files.
- **Threshold: fast only from about 2.70 GB `Cached`.**
  - The fast runs started at 2.71-2.97 GB. r16 run 8 at 2.67 had 587 majflt and printed 18.5.
  - Below 2.70 no run was fully fast: 2.24-2.68 GB gave 14.1-18.9 tok/s. Above it, runs were fast unless the
    previous step had evicted pages: r16 run 12 after an idle pause, 17.1; run 17, 18.7. That is R49's correction
    note.

## 2. R52 (r17) through the model

**Inputs (*device*, r17):**

- decode_ms/32 = 251, 267, 289, 358, 285, 313, 257, 293 (repack on) and 330, 414 (repack off);
- A78 at run end 1,440 (runs 0-2), 1,209 (run 3), 1,075 (runs 4-7);
- majflt 16.8-35.4 k and minflt 155-200 k per run;
- `Cached` 1.85-2.35 GB (2.59 in the repack-off runs), SwapFree 0 from run 1 on;
- app RssFile 1.45-1.71 GB;
- per-token gaps of runs 6 and 7: median 247/258 ms, fastest 116/124 ms (token 17 in both, at 1,075 MHz).

**Compute at the session's clocks.** For the stock arithmetic, s is higher than Granite's stack: the stock Q3_K
`vec_dot`, its waits and the small nodes are about 85 % of memo-08's terms (*estimated*, s = 0.85, band 0.75-0.90).
At memo-08's centre of 69.4 ms (repack on), C would be 97 at 1,440 and 126 at 1,075. **The fastest token (116 ms at
1,075, *device*) bounds C(1,075) ≤ 116.** That means C(2,112) ≤ 64-67, the lower half of memo-08's band. I recentre
row (a) at **C(2,112) = 66 (60-74)**. That gives **92 ms at 1,440 and 120 at 1,075** (*estimated*).

**I/O.** Decode − compute = **160-198 ms per token at 1,440** (runs 0-2) and **137-193 at 1,075** (runs 4-7), about
60-70 % of the token. Of the compute, 26-54 ms is the clock cap (92-120 against 66).

**Is it consistent with the faults and the memory lines?** Three checks:

1. **Bytes.** The app's RssFile of 1.45-1.71 GB and memo-13's effective 1.6-2.0 GB give M(c) of 40-75 MB per token.
   At c = 1.8-2.0 that is about 45-54 MB, so **κ_sync ≈ 3.5 ms/MB (2.7-5.0), 0.29 GB/s** (*estimated*). Memo-13
   assumed 0.3-0.9 GB/s for this access pattern; this is its slow end.
2. **Shape, independent of κ.** Mean gap of tokens 1-8 in run 6 is 398 ms and of tokens 9-31 is 214 ms. Minus 120 ms
   of compute, the early/late I/O ratio is 278/94 = **2.95**; run 7 (423 / 254 ms) gives **2.3**. Memo-13's LRU gives
   early/late 2.6 at 1.8 GB and 3.3 at 2.1 GB (*estimated*). This is a hit, so the I/O term is the cache shortfall the
   simulation describes and not something else. The fastest token at 116 ms means one token's experts happened to be
   all resident, and since routing is deterministic it is the same token in both runs.
3. **Per fault.** r17 has no per-token fault counts. Memo-13's LRU puts 33-57 % of a run's re-read bytes into the
   decode, which gives **175-630 decode majflt per token** (*estimated*). Against 140-200 ms of I/O that is **0.25-1.1
   ms per major fault, central about 0.55**, at **0.11-0.16 MB per fault** (memo-13). Campaign 1 without read-ahead
   (R29 eio 0) gave **0.22 ms** per major fault, which at 128-256 KB per fault is 0.9-1.8 ms/MB.
   - **So r17's flash reads cost about 2-4 times campaign 1's per MB.** That is the same order of magnitude but dearer,
     and outside the uncertainty of the decode split only at its ends.
   - Candidate causes, none of them measured:
     - **direct reclaim:** SwapFree 0, so every page-cache allocation must first evict a page;
     - **the fault path running at 1,075-1,440 MHz;**
     - **no `MADV_WILLNEED`,** so the queue depth is at most 3;
     - **spinning threads:** ggml's barrier threads spin while one thread waits on a fault, which adds heat and does
       no work.
   - r19's `/proc/self/io` and `read_ahead_kb`, if readable, would separate these.

**Breakdown of a typical r17 token (about 290 ms, *estimated*):** compute at 2,112 about 66 ms (23 %); clock penalty
26-54 ms (9-19 %); flash I/O 160-195 ms (55-67 %); outside under 1 ms. **The model reproduces 250-360 ms only with
I/O as the largest term. Compute at reduced clock alone gives 92-120 ms; faults at campaign-1 costs alone give
40-140 ms. Both terms are needed.**

The repack-off runs (330, 414 ms) fit the same model with M at repack off (+0.24 GB of file set, memo-13: 68-122
MB/token at 1.8-2.1 GB) and the clock falling to 1,075.

## 3. Predictions for kit r19 (OLMoE `olmoe-Mi`, campaign 1's runtime, repack on, 3 threads)

**Rows:**

- **(a)** all knobs off (R25's vanilla set, `EXPERT_IO=0`; stock arithmetic). C = 66 (60-74), s = 0.85 (0.75-0.90),
  κ_sync 3.5 (2.7-5.0).
- **(b)** r16 shipping minus Granite numerics (`TINY_OP=262144 EXPERT_IO=1`, compiled defaults). C = memo-08 §9.1's
  47.5 + 0.5 outside = 48 (44-52); s = 0.75 (0.60-0.85), between Granite's measured 0.59 and OLMoE's larger expert
  share. Read-ahead κ_ra = κ_sync / 2.8 = 1.25 (0.9-2.5).
- **(c)** (b) + `EXPERT_IO=5 GGML_REPACK_RELEASE=1`. Resident: (b) + 0.3 (−0.5 to +1.5; the helper thread's guess
  and madvise cost). Not resident: next-layer guess overlaps 0-30 % of the I/O, κ = 1.1 (0.7-2.5). The release
  matters only for the first run after a load (memo-13: 0 in steady state).

**States**, as the report will show them:

- *resident* = mincore of the file ≥ 2.6 GB before the run;
- *≥ 2,000* = A78 ≥ 2,000 MHz at run end;
- *throttled* = 1,075-1,450 MHz (central computed at 1,250, band 1,440 to 1,075);
- *not resident*: the central and the main band use an R52-like cache (`Cached` 2.2-2.35 GB, M = 45 MB, band 39-54).
  "wide" extends the band to `Cached` 1.6 GB (M up to 123 MB).

Every number is *estimated*. Steady = decode_ms / 32, in ms; printed tok/s = 0.995 · 31,000 / (32 · T + 12).

| row | resident, ≥ 2,000 | resident, throttled | not resident, ≥ 2,000 | not resident, throttled |
|---|---|---|---|---|
| (a) ms | **66** (60-74) | **105** (81-138) | **224** (165-344; wide 165-689) | **262** (186-408; wide 186-753) |
| (a) tok/s | **14.5** (13.0-16.0) | **9.2** (7.0-11.8) | **4.3** (2.8-5.8; wide 1.4-5.8) | **3.7** (2.4-5.2; wide 1.3-5.2) |
| (b) ms | **48** (44-52) | **73** (56-95) | **104** (79-187; wide 79-360) | **129** (91-230; wide 91-402) |
| (b) tok/s | **19.9** (18.4-21.7) | **13.2** (10.1-17.0) | **9.2** (5.1-12.1; wide 2.7-12.1) | **7.4** (4.2-10.5; wide 2.4-10.5) |
| (c) ms | **48** (44-54) | **73** (56-97) | **98** (71-188; wide 71-361) | **123** (83-232; wide 83-405) |
| (c) tok/s | **19.8** (17.9-21.8) | **13.1** (9.9-17.1) | **9.8** (5.1-13.5; wide 2.7-13.5) | **7.8** (4.1-11.5; wide 2.4-11.5) |

**Relations that hold more tightly than the absolutes** (they share the day):

- In the same state, (b)/(a) is 0.68-0.80 if resident and 0.45-0.55 if not. R52's (a)-equivalent at the r17 state is
  reproduced by construction.
- (c) − (b) is −0.5 to +1.5 ms if resident and 0 to −30 % of the I/O term if not.

**For runs in between** (1,450-2,000 MHz, or mincore 2.0-2.6 GB), use the §1.2 formula per run with the measured
clock and mincore. The intended score is per-run residuals against the formula, not one band.

**Which state I expect.**

- **Memory:** not resident, with probability about **0.8-0.9**.
  - The kit cannot change memory, and §4's record says a session with `Cached` ≥ 2.86 GB before a run occurred once in
    19 sessions.
  - The day's state is set by the owner's other apps. Swap was full in every session except the evening of r15/r16.
  - Expect mincore of about 1.5-2.1 GB and RssFile 1.45-1.9 GB.
- **Clock:** likely **≥ 2,000 MHz for the first run or two, then 1,440-1,800** (*estimated*).
  - A 10-minute gate removes R52's starting heat, which followed a 3 GB hash. In r16 the gated phone went from 0.87 to
    0.65 in about 10 minutes of light work. Starting from 1.0 or higher, the 0.65 target may not be reached within 10
    minutes; I put that at about 0.3-0.5.
  - A non-resident run lasts about 18-25 s, against about 6 s when resident. Its three compute threads spin at
    barriers while one of them waits on a fault, so each run heats the phone about as much as several resident runs
    do. r17's headroom climbed 0.97 → 1.10 over 10 such runs.
  - Unless every run is gated to < 0.65 (90 s will not be enough after a thrash run), later runs fall to the throttled
    column.
- **The most likely cells:** (a) 3.5-4.5 tok/s, (b) 7-10, (c) 7.5-10.5. Expect R52 again for (a), with read-ahead
  worth about ×2 in (b) and (c).
- **Advice for the pre-registration** (the lead decides):
  - interleave the rows (a b c c b a …) so that heat and cache reach all three equally;
  - score every run against its own state's cell or the per-run formula, never a pooled band;
  - log `/proc/self/limits` (memlock), `MemAvailable`, `SwapFree` and mincore before and after each run.

## 4. Can OLMoE be resident on this phone at all?

### 4.1 Memory record of every session (*device*, `Cached` in GB)

| report | `Cached` at start | max / min in session | max SwapFree (MB) | official runs, tok/s (Granite, that day's build) |
|---|---|---|---|---|
| r1-r2 (11-02) | 1.82, 1.52 | 2.22 / 1.37; 2.29 / 1.06 | 411; 267 | 4.1-8.4 |
| r3 (11-02) | 1.77 | **2.77** / 1.77 | 613 | 5.4-11.2 |
| r4 (11-03) | 1.94 | 2.39 / 1.51 | 351 | 5.3-11.3 |
| r5 (11-04) | 2.55 | 2.63 / 1.88 | 386 | 8.2-13.9 |
| r6 (11-05) | 2.12 | 2.62 / 1.84 | 513 | 6.1-14.0 |
| r7 (11-06) | 1.50 | 1.92 / 1.36 | 35 | 3.9-6.1 |
| r8 (11-06) | 2.65 | **2.67** / 2.06 | 482 | 8.5-16.4 |
| r9 (11-06) | 2.25 | 2.45 / 1.63 | 87 | 5.5-12.9 |
| r10-r14b (11-07, 11-08 day) | 1.03-1.97 | 1.59-2.36 / 0.99-1.67 | 20-231 | 3.8-14.9 |
| **r15 (11-08 eve)** | **2.86** | **2.91** / 1.84 | 992 | 12.0-18.2 |
| **r16 (11-08 eve)** | **2.75** | **2.98** / 2.24 | **1,268** | **12.3-20.95 (R49 median 20.51)** |
| r17 (11-14, OLMoE) | 1.85 | 2.59 / 1.85 | 238 | 2.3-3.8 |

### 4.2 The threshold and the probability

- **Granite:** fault-free decode needed `Cached` ≳ 2.70 GB before the run (§1.4). Its *host* working set is 2,569 MB +
  24 MB of dense pages per token, so about 0.1-0.15 GB of other hot file pages shared the cache.
- **OLMoE:** 2,725 + 9 MB (*host*), so the threshold is **`Cached` ≳ 2.86 GB (2.75-3.0)** (*estimated*). The app's
  anonymous memory is the same for both models (§1.1), so the threshold transfers.
- **The record:**
  - `Cached` ≥ 2.86 before a run: **r16 only**, 8 of its 21 runs (runs 7, 11, 15-20). r15 reached 2.91 once, after
    its last run.
  - ≥ 2.70: r16 and r15, plus a moment each in r3 (2.77, after a run).
  - ≥ 2.55 at any time: 7 of 19 sessions.
  - Both high-cache sessions are one evening (11-08), which began with 0.99-1.27 GB SwapFree; no other session had
    more than 0.61 GB.
  - So it is **one good evening out of eight calendar days.**
- **The open discrepancy (memo-13 §2) moves the answer.** In r16's fault-free runs RssFile stayed at 2.0-2.02 GB with
  about 3.7 k minflt, so the device touched at most about 2.25 GB of the Granite file, not 2.57 (*device*-derived
  bound).
  - **If the device touches less than the host trace** by the same 0.3 GB, OLMoE needs about 2.55 GB `Cached`. That
    happened in about 7 of 19 sessions, but mostly briefly. r17 itself peaked at 2.59 in the repack-off runs, whose
    file set is 0.24 GB larger.
  - mincore in r19 settles this.
- **P(a given session offers OLMoE a fault-free official run) ≈ 0.10, band 0.05-0.35** (*estimated*; 19 sessions on
  8 days, not independent draws). The low end holds if the host trace is right, the high end if the RssFile bound
  is.
- **Also cool (≥ 2,000 MHz, gate ≤ 0.70):** roughly ×0.6-0.9 of that.
- **Measurable conditions to pre-register:**
  - resident: mincore of the file ≥ 2.6 GB before the run, or `Cached` ≥ 2.86 GB with SwapFree > 0 as a proxy;
  - cool: headroom ≤ 0.70 at the gate;
  - decode not faulting: run majflt ≤ about 1,000 with read-ahead off, minflt ≤ about 5 k with it on.

### 4.3 What the runtime can do about residency, priced (rule: fix it in the runtime, never in the user's phone)

Gap to close on an R52-like day: 2.86 − 2.35 = **0.5 GB** at the best R52 moment; up to 1.1 GB on a 1.8 GB day.

| lever | MB it frees or protects | ms effect | closes 0.4-1.1 GB? |
|---|---|---|---|
| `mlock` / `mlockall` / `MAP_LOCKED` | **0.06 MB**: AOSP's `init.rc` sets `setrlimit memlock 65536 65536`; apps inherit it, and raising it needs a capability apps lack. A 2026 write-up measured 65,536 soft and hard on an Android 16 phone. *Not checked on this phone*; r19 can read `/proc/self/limits` | none | **no** |
| keep-warm / touch (re-advise or re-touch the set) | **0 MB of capacity**; it only changes which pages age out | on a day with capacity it would have saved R49's two evicted runs (−4 to −13 ms/token on those runs); on a short day it evicts the decode set for the prefill, which is harmful | **no**. Worth it only behind a runtime check (`MemAvailable` ≥ set + margin) |
| whole-region `MADV_WILLNEED` at the start of the prefill | 0 MB | moves the prefill re-read to 1-1.5 GB/s sequential, overlapped with compute (*estimated*); decode is unchanged if capacity is short | no |
| expert read-ahead (`EXPERT_IO=1`, rows (b)/(c)) | 0 MB | I/O ÷ 2.8 (R29) | no; it is the measured ×2 on thrash days |
| next-layer guess (bit 4) | 0 MB | 0 to −30 % of the I/O (*estimated*) | no |
| expert skip `MASS=900` | prefill union 968 → 967 slices: **3 MB**; decode distinct −116 MB, per-token bytes −13 % | late decode re-read −10 to −30 % | no |
| repack off | −235 MB anon, +235 MB file set: **0 net** (r17's repack-off runs had 0.2 GB more `Cached` and were slower) | +0 to +5 ms compute | no |
| `GGML_REPACK_RELEASE` | 0 in steady state, ≤ 235 MB for the first run | — | no |
| decode-hot set protected from the prefill sweep (memo-13 §3) | 0 MB | 40-125 → 25-40 MB/token, i.e. −20 to −100 ms at κ_ra | no; it makes the miss cheaper |
| **lazy KV cache** (do not memset the 4,096-cell q8_0 buffer; let untouched cells stay unmapped zero pages) | **up to 267 MB** of anon (285 MB, of which about 18 MB used up to position 188, *estimated*). Part of it may already sit in zram as zero pages: r17's app `VmSwap` reached 213 MB. So the real gain is **0-0.27 GB** | 0 ms; must be identity-checked (zero pages equal the clear) | **no**, a quarter to half of the gap at best. *Source:* `ggml_backend_buffer_clear(buf, 0)` at `src/llama-kv-cache.cpp:293` (base `620d2225c`), and again in `clear(true)` at line 399 |
| smaller `n_ctx` (512) | same ~0.27 GB | must be checked for identity; it changes the app's product setting | needs a lead or owner call |
| **full anonymous (or dma-buf pinned) copy of the experts** | moves 2.73 GB from evictable page cache to memory the kernel cannot reclaim cheaply. dma-buf through `AHardwareBuffer` is an unprivileged pinning path, capped at 2,047 MiB per buffer (a 2026 write-up). Capacity is not created: on R52's day the app's anon 0.6 + `MemAvailable` 2.0-2.47 = 2.6-3.1 GB against a need of about 3.4 GB | fault-free decode if granted; a copy of 2-4 s at load | **only by making the system kill background apps** (lmkd) or by being killed. R23 showed that partial anon copies shrink the cache for the rest. A full copy changes the victim, not the size. **Whether "the runtime takes memory so the OS kills other apps" is inside rule 5 is an owner ruling; I price it, I do not propose it** |

**Reading.** Rule 5 forbids asking the user to free memory. No legitimate lever enlarges the memory the phone gives
this app on a given day, except taking memory by force, which needs a ruling, and the lazy KV pages (≤ 0.27 GB). **On
an R52-like day OLMoE cannot be resident. On an r16-like evening it fits with about 0-0.15 GB of margin, and with
0.1-0.4 GB if lazy KV works and the RssFile bound holds.**

## 5. The honest restatement of the 25 tok/s question

- **Memo-08's kernel-level numbers apply only when all three hold:**
  1. the benchmark's working set is resident at run start (mincore ≥ about 2.6-2.7 GB of `olmoe-Mi`);
  2. the A78 runs at ≥ 2,000 MHz through the decode (gate ≤ 0.70);
  3. decode faults are about 0 (majflt ≲ 1 per token).

  Then (b) ≈ 48 ms (19.9 printed), and memo-08's full package about 42.5 ms (22.5 printed). 25 needs R53 at about 43
  ms or less, a coin flip at best (memo-08 §9.5).
- **Outside that state the kernel terms are noise.**
  - On an R52-like day the I/O term is about 56 ms per token even with read-ahead (κ_ra × 45 MB). A perfect 38 ms
    compute token would print about 10 tok/s.
  - 25 tok/s needs total I/O under 0-2 ms per token, which means residency.
- **OLMoE, plainly:** 25 is reachable only on a resident, cool session, and only if the device compute comes in at
  the fast end. My rough joint estimate is P(a claim run at 25) ≲ 0.1 (residency 0.05-0.35 × package ≤ 38 ms about
  0.2-0.3; *estimated*, not a pre-registrable band).
- **Stock Granite (`granite-Mi`):** it has the same bytes and working set as campaign 1's file (2.57 GB *host*), so it
  has campaign 1's residency profile: one good evening in eight days.
  - Even resident, memo-08 §7 puts it at about 23.8 printed, short by 1.9-2.6 ms.
  - Residency can only lower that. The proposed "no" stands.
  - If the per-file copy rule turns its repack on, it carries 0.42-0.7 GB of anon, the same as R49.
- **What the claim rule implies.** "Median of 10 plain runs ≥ 25" is a property of the runtime **and the session**.
  - Within a resident session R49 showed 8 of 10 fast runs. The cold first run and evictions after idle are the
    other two, and the median survives them.
  - Across sessions the median is set by the day: about 3-10 tok/s on non-resident days and up to 20-22 on resident
    ones, whatever the runtime does.
  - Campaign 1's 20.51 was in fact a resident-evening result. It was the only session in which that state existed.
- **Options for the lead (not my decision):**
  1. Write the state into the claim, fixed before any run, with all attempted sessions reported. For example: "on a
     session whose pre-run mincore ≥ X GB and headroom ≤ 0.70, the median of 10 plain runs ≥ 25".
  2. Keep the rule unconditional and say that it is a lottery on the phone's memory with odds of about 0.1 per
     session.
  3. Ask the owner for a ruling on forced residency (§4.3, last row), which would turn the lottery into a runtime
     property at a cost to the user's other apps.

  Choosing the day after seeing a result is not allowed under rules 3 and 11. Choosing it on a pre-registered,
  measured state is allowed if the frequency of that state is reported with the claim.

## 6. Not verified

- RLIMIT_MEMLOCK on this phone (AOSP default and an Android 16 measurement only); dma-buf limits on Android 11-14.
- κ_sync rests on memo-13's LRU byte estimates and on assumed effective caches of 1.8-2.0 GB. r17 has no per-token
  faults, so the per-fault cost (0.25-1.1 ms) is a range, not a measurement.
- The clock share s for OLMoE rows (a) and (b) is not measured. Granite's 0.59 is the only device fit. The r17
  fastest-token bound assumes the clock was 1,075 MHz during that token (both the before and after samples read
  1,075).
- The RssFile discrepancy (2.0-2.25 GB touched against 2.57 *host*) is unresolved, and it moves the residency
  probability by a factor of three.
- Lazy KV: whether bionic and scudo hand back zeroed pages for the ggml CPU buffer, and how much of the KV is already
  in zram.
- Whether forced residency (anon or dma-buf) is even granted on an R52-like day, or whether lmkd kills the app.

## Sources

- AOSP commit "Set memlock rlimit to 64KB" (`init.rc`: `setrlimit memlock 65536 65536`; the CTS test applies to vendor
  API level U and later):
  <https://gerrit.omnirom.org/plugins/gitiles/android_system_core/+/1447120f78bd653ed7d3501f4c1588a1aec15d99%5E%21>
- "Increase RLIMIT_MEMLOCK for keystore2" (the default 65,536 is too low for a system service):
  <https://gerrit.omnirom.org/plugins/gitiles/android_system_security/+/2202fcc19f4587f4c9e159b2568e760b37be232b>
- BigMoeOnEdge, `docs/android-memory.md` (Android 16 phone, unrooted: RLIMIT_MEMLOCK 65,536 soft and hard; mlock
  ineffective; dma-buf through `AHardwareBuffer` as an unprivileged pin, 2,047 MiB per buffer; MADV_COLD/PAGEOUT
  steer reclaim but do not stop it):
  <https://raw.githubusercontent.com/Helldez/BigMoeOnEdge/master/docs/android-memory.md>
- mlock(2) (the RLIMIT_MEMLOCK rule without CAP_IPC_LOCK):
  <https://git.tldp.org/LDP/man-pages/src/tag/man-pages-5.04/man2/mlock.2>
