# Bonnie, memo-12 work, round 35: heat with the GPU on, the flush, and bands for an x4s re-cell

2026-11-25. Labels: *device* (phone), *host*, *estimated* (arithmetic), *source*, *external* (a published page).
Inputs: the campaign-3 probe JSONs in `reports/device/` (R72 kit r33, R74 kit r34, R75 kits r35 and r36), my r33
and r34, Ebony's r18, `patches/llama.cpp-c3/0007-x4-join-probe.patch`. Four web searches and one page fetch
(§3, §6; sources at the end). Ebony's r19 was not read; it is being written in parallel. Not committed.

Script (same folder, new): `python3 -I -B bonnie_gpuheat_r35.py <repo root> [-v]` (seconds). It rebuilds a clock
for every probe block from the arms' own lengths, lists caps, headroom and gate waits, tabulates the flush against
the GPU's idle gap, and prices a claim run. Three self-checks (the x4s events r34 read by position; the knee; the
pricing's two limits).

**What the data cannot do.** Probe lines carry no time stamp. Each event below has a band: the clock from the arms'
lengths, plus the block's unplaced time (setup, buffer fills). No block is a clean "passed gate, then lane power
until the first cap". Everything in §1.3 and §2 is *estimated* from two to three blocks per kind of load.

## 0. Answer first

1. **Seconds to the first cap from a passed gate** (*estimated* from *device* lines, §1):

   | load | first cap | blocks |
   |---|---|---|
   | four (or three) CPU threads, bus-bound, GPU off | **about 30 s (28 to 41)**; second step (1,785) about 18 s later | R72 l23, l4, l7 (n = 3); l6: none in 26 s |
   | GPU alone (74 to 93 % busy) with one feeder thread | **none in 18 s and in 32 s** | R75 x4c, R74 x3g |
   | lane-like: CPU set with the GPU | **about 17 s (13 to 24)**; second CPU step 6 to 10 s later; GPU cap (676) about 10 s after the first CPU step | R74 x3c (13 to 21 s, warm body); R75 x4s (14 to 19 s, but from 0.735, gate timed out) |
   | GPU-heavy mix on a cool body | GPU cap first at 23 to 39 s; CPU cap (2,380) at 43 to 59 s | R75 x4m (n = 1) |

   **The GPU roughly halves the time to the first cap: 0.57 (0.4 to 0.8) of the CPU-only figure.** The first cap of
   either kind came at a headroom reading of 0.79 to 0.90 in all three GPU blocks.
2. **My r33 figure for CPU-only was pessimistic.** r33 derived "22 % by 12 s, 52 % by 16 s" for four threads from
   campaign 2's three-thread chains. R72's three direct blocks from a passed gate read 28 to 41 s. r33's value is
   now my hot end, not my centre. Its P(cap before the decode ends) of 0.3 / 0.45 / 0.35 becomes
   **0.06 / 0.14 / 0.08 at the centre** (hot end 0.23 / 0.35 / 0.26) for a stock run without the lane.
3. **The price of heat is specific to the lane** (*estimated*, §2). Without the lane a cap costs the decode
   0 to 3 % (r32). With the lane, the cap's onset is r34's state B, where the lane cell ran at 1.03 to 1.14 of the
   cool CPU-alone time (*device*): the lane's gain is gone for the rest of the run. Per run, Yi / Llama-2 / DeepSeek:

   | case | P(cap by the end of the decode), centre (cool end to hot end) | a run capped at mid-decode |
   |---|---|---|
   | no lane, stock | 0.06 / 0.14 / 0.08 (0.00 to 0.23 / 0.01 to 0.35 / 0.00 to 0.26) | 6.79 / 6.12 / 6.13 |
   | no lane, a method at 12.0 | 0.03 / 0.08 / 0.04 (to 0.17 / 0.28 / 0.19) | 11.82 on all three |
   | G, width 1 (8.41 / 7.85 / 7.72) | 0.11 / 0.20 / 0.13 (to 0.33 / 0.45 / 0.37) | 7.37 / 6.74 / 6.69 |
   | G in a method at 12.0 | 0.06 / 0.13 / 0.07 (to 0.26 / 0.37 / 0.28) | 10.51 / 10.30 / 10.41 |

4. **For the claim's median of ten, heat is a second-order price at the centre and a first-order one at the hot
   end, on Llama-2 only.** Centre: the median does not move (P(5 or more of 10 runs capped) 0.005 or less with G
   in a method at 12.0). Hot end: Llama-2's median of ten with G in a 12.0 method has a mean of 11.61 and a 10th
   percentile of 10.42 (Yi 11.93 / 12.00, DeepSeek 11.90 / 11.82). **A 12.0 claim on a lane needs about 0.4 tok/s
   of margin on Llama-2 to be safe at the hot end; without the lane it needs 0.05.** This does not change which
   rungs reach 12 (none as built); it is one more cost the lane rungs carry and the CPU rungs do not. P(12 on all
   three) stays about 0.002 (0.0005 to 0.005, judgement); r34's term "the claim run stays in state A: 0.8" is
   confirmed at 0.8 to 0.9 per run on Llama-2 at the centre.
5. **Gate wait after a run-sized load is not visibly longer with the GPU on** (*device*, §1.4): 45, 45 and 85 s
   after 10 to 18 s GPU blocks; 40 to 120 s after 10 to 19 s CPU-only blocks. After 61 and 63 s of lane-like load
   the 180 s gate timed out twice (0.735, 0.705).
6. **Recordable queue (R75 x4d, 0.356)**: the 0.356 is seven joins made into one, not a cheaper flush. One replay
   with its join cost 310 µs, one plain launch with its join 124.5 µs (*device*). For the split's one kernel per
   join it can remove only the enqueue's share of the 214 µs: **0 to 50 µs, centre 20 µs: +0.05 tok/s (0 to +0.15)
   as built, +0.02 under G; with two or three kernels per join (width 2, G's flag kernel) about +0.1 (0.03 to
   0.2)** (*estimated*). Bytes 0. Under every build line I know of.
7. **`cl_qcom_perf_hint`**: HIGH is the default for a context (*external*, Khronos registry, extension version
   1.0.5), and neither upstream's backend nor the probe sets the hint (*source*). The phone agrees: the GPU read
   840 MHz in all 30 samples of an arm at 23 % duty (*device*). **Setting HIGH is worth 0 µs** (P that it changes
   the flush: 0.05, judgement). The hint can only lower the clock; NORMAL is a possible heat lever, not a speed
   one (§3.3, try item).
8. **New reading, the largest of the round: the flush has a step at a GPU idle gap of about 0.3 ms** (§6).
   In x4s, launch plus flush was 124 to 147 µs in the three arms whose GPU had been idle under 0.33 ms and
   200 to 245 µs in the fifteen arms above it, with no overlap, across all four phone states (*device*; the gap is
   *estimated* as `ms_per_mat` − `gpu_ms`). It answers r34's open item "why 124 to 147 in state C", withdraws my
   counter-example to Ebony's reading (pos 9 had a gap of 0.39 ms, not zero), and gives her T10 a prediction. A
   candidate mechanism with the right size exists (*external*, not the phone's tree): the GPU's inter-frame
   power collapse with a hysteresis of 0x1680 cycles, 300 µs at 19.2 MHz by my arithmetic. Worth
   **+0.21 to +0.26 tok/s as built, about +0.1 under G** (*estimated*), and under G it is free: a share on the
   GPU-critical side keeps the gap short.
9. **Bands for the re-cell** are in §4, merged with Ebony's T3 and T4, as a proposal for the lead.

## 1. Heat with the GPU on (*device* lines; clocks and classes *estimated*)

### 1.1 Every block with caps printed

From the script. "t" is seconds of arm time since the block began; the band adds the block's unplaced time. Gate
0.65, limit 180 s. Power saving off, unplugged, in every block.

| round, block | battery | gate before (start → end, wait) | load | first CPU cap | later |
|---|---|---|---|---|---|
| R72 l23 | 70 % | 0.960 → 0.646, 160 s | CPU only, 3 and 4 threads, 56 s | 2,150 at **28 to 31 s** | none further in 59 s |
| R72 l4 | 73 % | 0.733 → 0.638, 30 s | CPU only, 3 and 4 threads, 249 s | 2,150 at **36 s** (arm before it ended at 30 s; unplaced 33 s over the block, about 5 s by share) | 1,785 at 54 s; 1,555 at 64 s |
| R72 l7 | 65 % | 0.697 → 0.648, 30 s | CPU only, 3 and 4 threads, 113 s with 63 s unplaced | 2,150 at 33 to 96 s; back to full 9 s later | weak cell |
| R72 l6 | 66 % | 0.727 → 0.649, 40 s | CPU only, light, 27 s | none | – |
| R72 l1, l5 | 76 %, 69 % | passed | mostly one thread, 523 s and 579 s | none | – |
| R74 x3g | 68 % | 0.709 → 0.650, 85 s | GPU alone, 31 to 76 % busy, 32 s | none | headroom 0.673 → 0.783 |
| R75 x4c | 23 % | 0.688 → 0.643, 45 s | GPU alone, 59 to 93 % busy, 18 s | none | 0.687 → 0.750 |
| R74 x3c | 67 % | 0.897 → 0.647, 60 s | 12 s of together arms (4 s CPU, 4 s GPU, 4 s both at 86 to 89 % busy), then split arms | 2,150 at **13 to 21 s**, reading 0.843 | 1,785 at 19 to 27 s; GPU stays at 840 |
| R75 x4m | 22 % | 0.760 → 0.647, 45 s | 14 s GPU alone, then 31 s of together arms | 2,380 at **43 to 59 s**, reading 0.977 | **GPU 676 first, at 23 to 39 s**, reading 0.823; after the block `thermal_pwrlevel` 3 |
| R75 x4s | 15 % | 1.148 → 0.735, **timed out** | lane power from the start: four threads with the GPU at 23 to 77 % | 2,150 at **14 to 19 s**, reading 0.793 → 0.903 | 1,785 and GPU 676 at 25 to 29 s; 1,420 at 33 to 38 s; GPU 491 at 43 to 48 s; after the block `thermal_pwrlevel` 6 |

Readings:
- **CPU only from a passed gate: 28, 30 to 41, 33 or later.** Centre 30 s. These arms are gemv loads of the
  decode's kind. No such block read a cap before 28 s.
- **GPU alone does not reach a cap in the times probed.** Headroom rises 0.003 to 0.005 per second.
- **CPU and GPU together reach one in about half the CPU-only time.** x3c is the only block that passed its gate
  and then drew both; its cap was printed after 12 s of mixed arms. Its weakness: R74's cool start was spoiled by
  campaign 2's cap probe 18 s before the round, so the body was warm behind a passed gate. x4s is the only block at
  true lane power from its first second; its weakness is the start at 0.735.
- **x4m is the cool-body block and it is slower: GPU cap at 23 to 39 s, CPU cap at 43 to 59 s.** Its first 14 s
  were GPU alone, which §1.1's GPU rows say costs little. Counting from the first together arm, the GPU cap came
  after 9 to 25 s and the CPU cap after 29 to 45 s. It is the reason my band's top is 24 s and not 19.
- **Which cap comes first follows the load**: GPU-heavy (x4m, 87 to 95 % busy) → the GPU's; lane duty (x4s, 23 to
  54 %) → the CPU's, with the GPU's 10 s later.
- **The steps follow each other faster with the GPU on**: 6 and 10 s from 2,150 to 1,785 (x3c, x4s) against 18 s
  (l4).
- Battery level does not sort the blocks (x3c at 67 %, x4s at 15 %, both fast).

### 1.2 Headroom

- The probe's `hr` changes about every 10 s, so only block-long slopes can be read: lane-like 0.005 to 0.008 per
  second over a block (x3c, x4m, x4s); 0.009 to 0.011 per second over the first 13 to 17 s of x3c and x4s.
- First reading inside a block after a passed gate: 0.673 to 0.697 (five blocks), not 0.65.
- **First cap of either kind at a reading of 0.79 to 0.90**: x3c 0.843 (the reading before it 0.737), x4m 0.823
  (GPU), x4s between 0.793 and 0.903. x4m's CPU cap waited until 0.977, so the reading does not say which cap.
- From 0.69 to 0.82 at 0.009 to 0.011 per second is 12 to 14 s: the same answer as §1.1 by another route, from
  the same blocks. It is not independent evidence.

### 1.3 The figure I carry

**From a passed gate at lane power: first cap at 17 s (13 to 24 s); CPU only: 30 s (28 to 41 s).** Ratio 0.57
(0.4 to 0.8). The centre is x4s's 14 to 19 s plus 2 to 4 s for the start at 0.65 instead of 0.735 (r33 §2.2 gave
3 to 5 s for that step with the CPU alone). n = 2 lane-like blocks and one cool-body block that disagrees upward.
A linear "dose" model fits these blocks badly (the together arms are several times hotter than the sum of their
parts), so I do not offer a formula for other duties.

### 1.4 The gate after a block (*device*)

| block before | load | next gate |
|---|---|---|
| x4d (GPU, 7 s) | – | passes at once (0.645) |
| x4e (child, 10 s), x4f (launcher, 10 s) | – | 45 s (peaks 0.835, 0.761) |
| x4c (GPU alone, 18 s) | – | 85 s (peak 0.947) |
| x3g (GPU alone, 32 s) | – | 60 s (peak 0.912) |
| R69 x1 (10 s), x2 (14 s); R72 x2 (14 s) | CPU with GPU cells | 60 to 70 s; 50 to 85 s; 40 s |
| R72 a (CPU only, 19 s) | – | 120 s (from 0.86) |
| x3c (35 s), x4m (61 s), x4s (63 s) | lane-like | **timed out at 180 s** (0.660, 0.735, 0.705) |

- The reading keeps rising for 5 to 20 s after the load stops (peaks at 5 to 20 s into the wait).
- A claim run is 10 to 15 s of load (§2). The table puts its gate wait at **45 to 120 s, with or without the
  GPU**. The kit's 90 s limit (r22 row 15) is inside that range; the probes' 180 s is not.

## 2. What it does to a 12 tok/s claim run (*estimated*)

### 2.1 Inputs

| input | value | from |
|---|---|---|
| prefill, four threads | 7.6 / 9.8 / 8.0 s | r33 §1.4 centres (*estimated*) |
| decode, 32 tokens | 32 ÷ the case's speed | – |
| stock | 6.90 / 6.21 / 6.22 tok/s | R75 rung 0 |
| lane as built, width 1 | 7.76 / 7.04 / 6.98 | r34 §4.1 |
| G, width 1; G, rung-7 form | 8.41 / 7.85 / 7.72; 10.59 / 10.46 / 10.14 | Ebony r18 §3.2 |
| time to the first cap: prefill (compute-bound, CPU) | 22 s (16 to 30) | between r33's 16 and R72's 30 |
| the same: CPU decode | 30 s (28 to 36) | §1.3 |
| the same: lane decode | 17 s (13 to 24) | §1.3 |
| spread of the cap's arrival (log scale) | 0.45 (0.30 to 0.60) | r32's chains: 22 % by 11 s, 52 % by 16 s gives 0.46 |
| after the cap, no lane | 0.97 × the speed | r32: the body loses under 3 % at 2,150 |
| after the cap, lane | (speed ÷ the lane's gain) × 0.95 (0.88 to 1.00) | state B: lane arms at 1.03 to 1.14 of the cool CPU arm, pos 5 to 9 over pos 0 (*device*) |

Rule: a run uses t ÷ T of each load's time to the first cap; the cap comes when the used fraction passes a
threshold with median 1. The gate is taken as passed before every run. The prefill is on the CPU in every case
(a GPU prefill is not built).

### 2.2 Result (script; Yi / Llama-2 / DeepSeek)

Centre:

| case | load, s | used at the end | P(cap in the prefill) | P(cap by the end) | a run capped at mid-decode | P(5 or more of 10 capped) |
|---|---|---|---|---|---|---|
| no lane, stock | 12.2 / 15.0 / 13.1 | 0.50 / 0.62 / 0.54 | 0.01 / 0.04 / 0.01 | 0.06 / 0.14 / 0.08 | 6.79 / 6.12 / 6.13 | 0.000 / 0.008 / 0.001 |
| no lane, a method at 12.0 | 10.3 / 12.5 / 10.7 | 0.43 / 0.53 / 0.45 | 0.01 / 0.04 / 0.01 | 0.03 / 0.08 / 0.04 | 11.82 / 11.82 / 11.82 | 0.000 / 0.001 / 0.000 |
| lane as built, width 1 | 11.7 / 14.3 / 12.6 | 0.59 / 0.71 / 0.63 | 0.01 / 0.04 / 0.01 | 0.12 / 0.23 / 0.16 | 7.11 / 6.42 / 6.40 | 0.004 / 0.053 / 0.011 |
| G, width 1 | 11.4 / 13.9 / 12.1 | 0.57 / 0.69 / 0.61 | 0.01 / 0.04 / 0.01 | 0.11 / 0.20 / 0.13 | 7.37 / 6.74 / 6.69 | 0.002 / 0.033 / 0.006 |
| G, rung-7 form | 10.6 / 12.9 / 11.2 | 0.52 / 0.63 / 0.55 | 0.01 / 0.04 / 0.01 | 0.08 / 0.15 / 0.09 | 9.28 / 8.98 / 8.79 | 0.000 / 0.009 / 0.001 |
| G in a method at 12.0 | 10.3 / 12.5 / 10.7 | 0.50 / 0.60 / 0.52 | 0.01 / 0.04 / 0.01 | 0.06 / 0.13 / 0.07 | 10.51 / 10.30 / 10.41 | 0.000 / 0.005 / 0.000 |

The median of ten equals the uncapped speed in every centre row (largest shift 0.01 tok/s).

Hot end (first cap at 16 / 28 / 13 s, spread 0.60, lane after the cap 0.88):

| case | P(cap by the end) | median of ten: mean | its 10th percentile | P(5 or more of 10 capped) |
|---|---|---|---|---|
| no lane, stock | 0.23 / 0.35 / 0.26 | 6.90 / 6.19 / 6.22 | 6.90 / 6.14 / 6.22 | 0.056 / 0.254 / 0.094 |
| no lane, a method at 12.0 | 0.17 / 0.28 / 0.19 | 12.00 / 11.98 / 12.00 | 12.00 / 11.96 / 12.00 | 0.019 / 0.122 / 0.029 |
| lane as built, width 1 | 0.35 / 0.47 / 0.40 | 7.66 / 6.71 / 6.84 | 7.38 / 6.03 / 6.46 | 0.246 / 0.559 / 0.355 |
| G, width 1 | 0.33 / 0.45 / 0.37 | 8.30 / 7.41 / 7.54 | 7.97 / 6.37 / 7.04 | 0.206 / 0.493 / 0.293 |
| G, rung-7 form | 0.28 / 0.39 / 0.31 | 10.51 / 10.05 / 10.01 | 10.44 / 8.90 / 9.69 | 0.121 / 0.346 / 0.168 |
| G in a method at 12.0 | 0.26 / 0.37 / 0.28 | 11.93 / 11.61 / 11.90 | 12.00 / 10.42 / 11.82 | 0.091 / 0.291 / 0.118 |

Cool end (30 / 36 / 24 s, spread 0.30): every probability is 0.01 or less; heat has no price.

### 2.3 Readings

- **Without the lane the first cap is nearly free for the claim**: at most 0.02 tok/s on the mean of the median,
  hot end, Llama-2.
- **With the lane it is not, at the hot end**: G at width 1 loses 0.11 / 0.44 / 0.18 tok/s on the mean of the
  median, which is 7 / 27 / 12 % of G's gain over stock (1.51 / 1.64 / 1.50).
- **Llama-2 is the exposed subject in every row**: longest prompt, largest body (r33 said so for the CPU alone).
- **Most of a run's heat is the prefill** (0.35 to 0.45 of the 0.50 to 0.71 used at the centre). A lane decode adds
  0.16 to 0.26; a CPU decode at stock 0.15 to 0.17. The lane's decode is shorter and hotter, and the two nearly
  cancel: G raises "used at the end" by 0.07 over stock.
- **So the two levers on this price are not lane levers**: a prefill that leaves less heat (three threads, or the
  GPU: r33 §4 item 3, Ebony r5), and the gate before each run actually passing (§1.4).
- **Ebony's L6 (self-timed split) is the lane's own guard**: if it refuses the split in state B, the lane rows of
  §2.2 fall back to the no-lane loss after the cap. It needs state B to be a state of the CPU beside the GPU and
  not of the CPU alone; no CPU-alone arm ran in state B, so this is open (§4, H6).
- What the model leaves out: heat carried from run to run when a gate times out (it would push all ten runs
  toward the hot end together, which is worse for a median than independent caps), and the second cap step
  (6 to 10 s after the first with the GPU on, so a run capped in its prefill can meet it).

## 3. The in-situ flush: recordable queue and perf hint

### 3.1 What x4d measured (*device*, *source*)

- The cell records one Llama-2 layer (seven kernels) per recording and replays it with one `clFinish`, against
  seven plain launches each followed by `clFinish` (patch lines 526 to 571).
- Per layer: plain 5.031 ms, replay 4.470 ms, kernels 4.160 ms. Overhead 871 µs for seven joins (124.5 each)
  against **310 µs for one replay with one join**. The printed 44.3 µs is 310 ÷ 7.
- So a replay with its join costs 2.5 times one plain launch with its join. Each kernel after the first adds
  about (310 − 124.5) ÷ 6 = **31 µs** (*estimated*), which is the price of a plain enqueue (30 to 37 µs in x4a's
  batch of eight, 12 µs in x4f's tight loop, *device*).
- **The recording saves joins when several kernels may share one. It does not make a join cheaper.** The split
  joins once per matrix by construction (the CPU rows and the GPU rows of one matrix meet before the next op).

### 3.2 Price for the split (*estimated*)

- x4s's `enq_us` is `launch` plus `clFlush` in one stamp (patch line 1106); the two parts were not printed apart.
  Isolated, an enqueue after idle costs 64 to 88 µs and a warm one 12 to 37 µs (*device*, x4a, x4f).
- A replay of a one-kernel recording still needs its own call and the flush. It can remove at most the cold
  enqueue's excess: **0 to 50 µs of the 214, centre 20 µs.**
- At r34 §4.1's sensitivity (155 µs of serial launch = +0.41 / +0.51 / +0.46 tok/s, so 0.0026 to 0.0033 tok/s per
  µs at width 1): **+0.05 to +0.07 tok/s, band 0 to +0.16, as built.** Under G the launch runs beside the rows on
  one thread and counts 0.41 as much: **+0.02**.
- Where a join carries two or three kernels (width 2 by two launches, r34's "+40 µs for the second enqueue"; G's
  trailing flag kernel), the recording removes the extra enqueues, 30 to 75 µs per join on thread 0: **about
  +0.1 tok/s (0.03 to 0.2) at rung 6 or 7 under G.**
- Bytes 0. Risks not priced: arguments changed per replay (the extension takes them at enqueue; the probe passed
  none), the 512-command limit (*device*, x4i), a second queue.
- R75's Y14 price (+0.50 / +0.43 / +0.40) put 44.3 + 13.3 µs in place of the whole K. That holds for a design
  that shares joins across a layer, not for the split. For the split my figure replaces it.

### 3.3 `cl_qcom_perf_hint`

- *External*: the extension (version 1.0.5, 2025-04-01) defines HIGH, NORMAL and LOW; "CL_PERF_HINT_HIGH_QCOM:
  requests the highest performance level from device. This is the default setting for devices in an OpenCL
  context"; NORMAL is "set dynamically by the GPU frequency and power management"; `clSetPerfHintQCOM(context,
  hint)` changes it on a live context. The page says nothing on idle, power collapse or latency.
- *Source*: no `perf_hint` in `lab/llama.cpp-c3/ggml/src/ggml-opencl/` or in the probe patches. The phone lists
  the extension (*device*, x4i).
- *Device*: `gpu_mhz=840..840,n=30` in lead arms at 23 to 44 % duty, 315 in the CPU arm. The clock is already
  pinned high while the context has work, as HIGH describes.
- **Price of setting HIGH: 0 µs, 0 bytes.** The step of §6 exists with HIGH in force, so HIGH does not prevent it.
- **NORMAL as a heat lever (try item, unpriced)**: at the as-built best share the GPU ends 0.4 ms or more before
  the CPU rows, so a lower GPU clock costs nothing until the GPU becomes critical (rates *device*: 21.3 GB/s at
  676 MHz against 24 at 840; 17.6 at 491). Whether NORMAL lowers the clock at 25 to 45 % duty, and whether that
  delays the first cap, only the phone can say. One arm pair in the heat pass of §4 would. P(it delays the first
  cap by 3 s or more): 0.25 (judgement).

## 4. Proposal: bands for the x4s re-cell, merged with Ebony's T3 and T4

**A proposal for the lead, not a pre-registration.** It needs the owner's go as a device exception. Sources: my
r34 §6, Ebony r18 §4 (T3, T4, and her bands under the table), this memo's §1 and §6. Where our bands differ I give
both and a merged one.

### 4.1 Cells (kit r36's x4s code; no new kernel; no model; 0 bytes)

| cell | what | load |
|---|---|---|
| **C, cool pass** | triplets cpu (0.5 s) \| lead split (1.0 s) \| cpu (0.5 s) at shares 0.40, 0.45, 0.50, 0.55, **ascending then descending** (T4); one feeder triplet at 0.25; Ebony's stall arm; 5 to 10 s of rest between triplets; rings allocated once before the block | about 22 s |
| **G, gap sweep** (Ebony's T10) | launch plus flush against the idle gap since the last kernel: 0, 0.1, 0.2, 0.3, 0.4, 0.5, 1, 5 ms | 5 s |
| **H, heat pass** (new, last) | from a passed gate: lead split at share 0.45 without rests until 10 s after the second CPU step, or 45 s; a cpu-alone second every 8 s | 45 s at most |

Printed in every cell:
- **a time stamp per arm** (ms since the block began; its absence is why §1 has bands);
- per cycle, p10 / p50 / p90 over the measured window only, of: launch, flush (apart), CPU rows, END to seen (T3),
  wait;
- my r34 §6 items: the 0.1 s compute canary on cpu 5 after every arm, `/proc/vmstat` and per-cpu `/proc/stat`
  deltas, `cpuinfo_cur_freq` beside `scaling_cur_freq`;
- in H: caps, GPU max, `thermal_pwrlevel` and headroom once a second from a thread on a little core (the cap
  probe's sampler).

Each cell after its own gate. About 75 s of load, 8 to 12 min with gates.

### 4.2 Bands (all *estimated*; P = my probability that the result is inside)

Cool pass. Ratios are per triplet (split ÷ mean of its two cpu arms), median of the two orders.

| # | quantity | Bonnie | Ebony r18 | merged band | P |
|---|---|---|---|---|---|
| C1 | best lead ratio, shares 0.40 to 0.55 | 0.83 (0.78 to 0.86) | 0.82 (0.76 to 0.88) | **0.83, 0.76 to 0.88** | 0.75 |
| C2 | share of the best ratio | 0.40 to 0.50 | 0.50 to 0.55 | 0.40 to 0.55 (all four) | – |
| C2b | the best is at 0.50 or 0.55 | 0.4 | (hers: yes) | – | 0.45 |
| C3 | CPU slowdown beside the GPU, measured window, share 0.50 | 1.38 (1.28 to 1.54) | 1.40 (1.30 to 1.50) | **1.38, 1.28 to 1.54** | 0.75 |
| C4 | launch plus flush, p50, shares 0.40 and 0.45 | 207 (190 to 235) µs | – | 190 to 235 µs | 0.7 |
| C5 | END to seen, p50, arms where the GPU ends within 0.2 ms of the CPU rows | 80 to 270 µs | 200 (120 to 350) µs | **80 to 350 µs** | 0.75 |
| C6 | END to seen, p50, arms where the GPU ends 0.4 ms or more early | under 30 µs | (not split by her) | under 30 µs | 0.75 |
| C7 | \|ascending − descending\| of the ratio at each share | 0.04 or less | – | 0.04 or less at 3 of 4 shares | 0.6 |
| C8 | feeder triplet at 0.25 | 0.85 (0.81 to 0.89) | – | 0.81 to 0.89 | 0.7 |
| C9 | stall arm excess | – | 80 (30 to 160) µs | hers | hers |
| C10 | canary rate, spread over the pass | 2 % or less | – | 2 % or less | 0.85 |
| C11 | no cap and `thermal_pwrlevel` 0 through the pass | – | – | yes | 0.8 |

Where we differ, and what decides: C2b (my slack fit puts the best share at 0.42; her clean arm is 0.55) and
C5 against C6 (I expect the tail only near balance; she reads 150 to 350 µs as the join's own). T3's percentiles
decide both.

Gap sweep.

| # | statement | P |
|---|---|---|
| G1 | launch plus flush is 60 µs or more higher at a 0.5 ms gap than at a 0.2 ms gap | 0.7 |
| G2 | the step lies between 0.25 and 0.40 ms | 0.55 |
| G3 | at gaps of 0.2 ms and under: 110 to 160 µs; at 0.5 ms and over: 195 to 250 µs | 0.55 |
| G4 | no further step between 1 and 5 ms (30 µs or less) | 0.7 |

Heat pass.

| # | quantity or statement | band | P |
|---|---|---|---|
| H1 | seconds from the block's start to the first cap of either kind | **17 (12 to 26)** | 0.65 (under 12: 0.15; over 26 or none: 0.20) |
| H2 | the CPU's cap comes before the GPU's | – | 0.7 |
| H3 | headroom reading at the first cap | 0.76 to 0.92 | 0.75 |
| H4 | the second CPU step follows within 12 s of the first | – | 0.7 |
| H5 | a slow stretch (cycle 1.10 × the block's first 8 s or more) begins within 3 s of the first cap | – | 0.6 |
| H6 | the cpu-alone second inside the slow stretch loses 8 % or less against the first one | – | 0.55 |
| H7 | the canary falls 10 % or more in the slow stretch | – | 0.35 |
| H8 | launch plus flush at share 0.45 stays within 15 % from start to end | – | 0.6 |

### 4.3 What each outcome changes (*estimated*)

| outcome | effect on the prices |
|---|---|
| C1 at 0.86 or under | r34's cool column stands: lane as built +0.8 to +1.0 tok/s, G +1.5 to +1.6 over stock |
| C1 at 0.90 or over with no cap | the CPU term is wrong at high share even cool; the lane rungs fall to about +0.5 |
| C5 high and C6 low | the tail is a balance effect: G's flag removes it only near balance; the best share moves up under G |
| G1 yes | the step is real: +0.21 to +0.26 as built, about +0.1 under G (§6); T10 closes; L5 gets a price |
| H1 under 12 s | §2's hot end is the phone: Llama-2 with G loses up to 0.44 tok/s on the median's mean |
| H1 over 26 s | heat is not a price of the lane at claim length |
| H6 yes | state B is a state of the CPU beside the GPU: L6 (refuse the split) is a complete guard |
| H7 yes | a clock limit the nodes do not show (r34 §3.4, first candidate) |

Voids I would write: any cap or `thermal_pwrlevel` above 0 in C or G; a triplet whose two cpu arms differ by more
than 3 %; H's gate timing out. Kill lines are the lead's; C1 at 0.90 or over is the one I would offer for the
lane's as-built rungs.

## 5. Lists (additions only)

| # | test | where | cost | closes when |
|---|---|---|---|---|
| B-T35a | time stamp per arm and a 1 Hz cap sampler in every probe block | kit, a few lines | 0 s | §1's bands become points |
| B-T35b | the heat pass H of §4 | device | 45 s | H1 outside 12 to 26 s replaces §1.3 |
| B-T35c | launch and flush stamped apart in the split loop | kit | 0 s | sets §3.2's 0 to 50 µs |
| B-T33b | unchanged: `heat_after` on gated runs | – | – | – |

| # | try | numerics | price (*estimated*) | needs |
|---|---|---|---|---|
| B-L35a | share chosen on the GPU-critical side under G, so the GPU's idle gap stays under 0.3 ms | bit-identical | about +0.1 tok/s; 0 bytes | G1, G's flag |
| B-L35b | perf hint NORMAL while the GPU is not critical | bit-identical | speed 0 or negative; heat unknown; 0 bytes | one arm pair in H |
| B-L35c | recording for joins with two or three kernels | bit-identical | about +0.1 (0.03 to 0.2); 0 bytes | G built first |

All three are model-independent: they act on the join, which is the same code for the three subjects.

## 6. Next question 1, first pass: the step in the flush at a 0.3 ms idle gap

### 6.1 The data (*device* fields; the gap is *estimated* as `ms_per_mat` − `gpu_ms`)

x4s, the eighteen lead arms, sorted by the GPU's idle time before the next launch:

| gap, ms | launch + flush, µs | pos | CPU cap | GPU max |
|---|---|---|---|---|
| 0.26 | 124 | 13 | 1,785 | 676 |
| 0.27 | 126 | 12 | 1,785 | 676 |
| 0.31 | 147 | 14 | 1,785 | 676 |
| 0.39 | 200 | 9 | 2,150 | 840 |
| 0.43 | 233 | 15 | 1,420 | 676 |
| 0.48 | 210 | 8 | 2,150 | 840 |
| 0.64 to 1.27 (twelve arms) | 203 to 245 | 1 to 7, 16 to 20 | none to 1,420 | 840 to 491 |

- **Under 0.33 ms: 124 to 147 µs (n = 3). At 0.39 ms and over: 200 to 245 µs (n = 15). No overlap.**
- Above the step the cost does not depend on the gap (0.39 to 1.27 ms), on the CPU cap (full to 1,420) or on the
  GPU's clock (840 to 491).
- The three short-gap arms are all state C. So state and gap are confounded in this block, and Ebony's T10 (the
  gap swept in one state) is the cell that separates them. That is why §4 carries it.
- r34 §7 item 4 corrected: I called pos 9 a counter-example to her reading ("a launch into a GPU that has just
  finished is cheaper"). Its gap was 0.39 ms. **Her reading fits all eighteen arms; I withdraw the objection.**

Other cells of R75 point the same way (*device*): SUBMIT to START is 6 to 20 µs where launches follow each other
in a spin loop (x4f `rtargs` 6.2, `la plain` 6.9, x4b spin on cpus 0 to 3 18.6) and 170 to 200 µs where the host
slept between them (x4b `finish` and `wfe` 181 to 201, x4f `la gate` 175, the first launch of x4a's batches 187 to
200). One cell does not fit: x4a `batch8` on cpu 4, first launch 10.2 µs.

### 6.2 A candidate mechanism (*external*; hypothesis, not the phone's tree)

- Adreno 6xx GPUs with a GMU power the core down between frames ("inter-frame power collapse") after a
  hysteresis counted in GMU clock cycles. A downstream kgsl commit sets the main hysteresis to 0x5DC cycles and
  gives it as 78.1 µs; the upstream driver's long value is 0x1680.
- My arithmetic: 0x5DC = 1,500 cycles in 78.1 µs is 19.2 MHz; **0x1680 = 5,760 cycles is 300 µs.** The step in
  §6.1 lies between 0.31 and 0.39 ms.
- Not shown: which value the Flip 3's kernel (`5.4.274-qgki`) uses, and that the extra 75 to 90 µs is the wake-up.
  kgsl settings are Ebony's r19 ground; this table is for her.
- If it is this, an app cannot switch it off (the kgsl nodes need root, and rule 5 forbids asking the user). The
  runtime can only keep the gap short.

### 6.3 Price (*estimated*)

| lever | serial µs saved per join | width 1, Yi / Llama-2 / DeepSeek |
|---|---|---|
| as built, gap kept under 0.3 ms (a keep-awake kernel, Ebony's L5) | 75 to 90 | **+0.21 / +0.26 / +0.24** (at 80 µs) |
| under G, share on the GPU-critical side (B-L35a) | 75 to 90 beside the rows, × 0.41 | about +0.09 to +0.11 |

- As built, the short gap comes with the end slack (pos 13: launch 124 µs but wait 152 µs), so the two cancel;
  that is why state C's cell is 0.816 and not better. **The gain needs G's flag join**, which removes the slack
  and lets the share sit where the GPU is critical.
- A keep-awake kernel holds the GPU out of its low-power state between matrices. That is GPU power at lane duty,
  and §1 says GPU-heavy load brings the GPU's cap first (x4m). I would not build L5 before H1 is known.
- Ebony's F (a kernel that never leaves the GPU) gets this for free and pays the same heat.
- This does not move any rung to 12: +0.1 on G is inside r34 §4.1's bands.

## 7. Next questions, ranked

1. **Prefill heat**: three threads (A78 only) or a GPU prefill against four threads, priced with §2's rule: how
   much of the 0.35 to 0.45 "used" by the prefill each one gives back, and what it does to P(cap by the end) on
   Llama-2 at the hot end. Arithmetic on r33 §1.2 and §2 here; no device. Mine; **I continue with this.**
2. **The 0.3 ms step against Ebony's r19** (kgsl hysteresis on SM8350, whether `gpu_busy` or any app-readable node
   shows the collapse count): merge once her memo is in.
3. **Run-to-run heat in a block of ten**: with §1.4's gate waits (45 to 120 s) and the kit's 90 s limit, how many
   of ten dense runs start above 0.65, and what that does to §2's median (the correlated case). Needs campaign 2's
   gate series and r33's script; second in line.
4. **G's best share with the step and without the slack**: re-run r34's cell model with a launch of 125 µs below a
   0.3 ms gap and 214 above; the share and the ratio under G. Small; after 1.
5. **States B and D**: wait for §4's canary.
6. **α with pos 0 and pos 21** (r34 item 6), unchanged.

## Sources

- `cl_qcom_perf_hint`, version 1.0.5 (values, default HIGH, `clSetPerfHintQCOM`):
  https://registry.khronos.org/OpenCL/extensions/qcom/cl_qcom_perf_hint.html
- Recordable queues named with the OpenCL ML SDK (the extension exists and is used to record and dispatch a
  model; no timing given; search summary only, page not read in full):
  https://www.qualcomm.com/developer/blog/2021/05/accelerate-your-models-our-opencl-ml-sdk and Qualcomm's Adreno
  SDK FAQ https://developer.qualcomm.com/software/adreno-gpu-sdk/faq
- Inter-frame power collapse, upstream series (long hysteresis 0x1680; search summary):
  https://patchew.org/linux/20250908-ifpc-support-v2-0-631b1080bf91@oss.qualcomm.com and
  https://lists.freedesktop.org/archives/freedreno/2025-July/039810.html
- Downstream kgsl hysteresis (0x5DC cycles given as 78.1 µs; search summary of the commit, not the Flip 3's tree):
  https://gitlab.e.foundation/e/devices/android_kernel_fairphone_FP3/-/commit/4edfb6b4b99533f3fb92fd162f91bbf4be587d54
- Repository: the six probe JSONs named above; `patches/llama.cpp-c3/0007-x4-join-probe.patch` lines 526 to 571
  and 1106 to 1126; `lab/llama.cpp-c3/ggml/src/ggml-opencl/`; R75-result.md rows V5, D1, Y14; my r32, r33, r34;
  Ebony r18.
