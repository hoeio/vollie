# c3 memo-12 work (Ebony, round 2): full review of bonnie-joins.md, R75 bands, and what could pass the lane's ceiling

Date: 2026-11-23. Author: Ebony (`claude-fable-5-2`). Not committed; the lead commits. No verdict here.

Labels: *device* = printed by the phone (R74 unless said); *device-derived* = my arithmetic on *device* fields of the
same block (no model in between); *estimated* = a model; *external* = another group's number on other hardware;
*host* = the laptop. Triples are Yi / Llama-2 / DeepSeek. 12 tok/s = 79.2 ms. Nothing here is a new phone number.

Parts: **A** review of `bonnie-joins.md` (item 1). **B** R75 bands (item 2). **C** scouting past the ceiling
(item 3). **D** ranked next questions, and what I did with them.

---

## A. Review of bonnie-joins.md (the parts I had not seen: §1 to §3, §4.1, §5 to §8)

I ran `python3 -I bonnie_joins.py` (self-check passes, tables reproduce) and re-read the R74 raw lines for every
*derived* cell of her §2. Her tables in §2.1, §2.2 and §2.3 are arithmetically right (checked: 143.1, 56.7, 57.1,
727 / 705 / 510, 65.6 / 56.4 / 54.8).

### A.1 Where we agree (independently reached)

- Four dependency stages per layer, so a fully split lane has 128 / 128 / 120 forced joins per token (her §3).
  I looked again for a schedule that overlaps two stages (her §8 item 2) and found none that pays:
  - a second position on the GPU (pipeline across positions) hides every join but makes both processors read the
    whole body: closed by price in my round-1 §4.1 (5.7 to 6.1 tok/s, *estimated*);
  - the FFN column partition (round-1 §4.4) makes the FFN one join instead of two, but its GPU chain is three
    launches, and R74 says dead time is per launch. It is the only schedule with fewer joins than hers, and it is
    not cheaper unless R75's timeline shows launches behind each other are free.
- The join is paid as lost GPU rows while the CPU keeps working, not as CPU waiting (her §2.3, my F9).
- Polling is worth little (her B, −30 µs). Agreed.
- The 199 µs between submit and start is not a slumber exit (her C). Agreed.
- Her §3 stage times are weight shares of the R72 body. One *device* cross-check exists and it holds: her `gate`+`up`
  stage is 1.964 ms for two matrices, 0.982 each; the split cell's CPU-only arm reads 0.948 ms for one 11008 × 4096
  (*device*). Within 4 %.

### A.2 Finding 1: both of our models count the launch gap twice, and it moves her headline

Her §6 says it in one line ("charging K on top double counts a little: at most about 5 ms per token"). I priced
it, and it is not little for the conclusion.

- The lane factor 0.621 and κ 0.521 use the GPU's **wall** rate beside the CPU, 21.53 GB/s (`x3 togsum`). That
  rate comes from the batch loop (8 launches, one `clFinish`), so it already holds the launch gaps.
- The GPU's **device-time** rate beside the CPU is higher. `gpu_busy` reads 89 / 86 % in the two `both` arms
  (*device*), and `gpu_busy` equals dev ÷ wall within 2 points in all nine `x3 alone` batch cells (*device*, her
  own §2.2 observation; I re-checked each row). So r_g(dev, both) = 21.80 ÷ 0.89 and 21.26 ÷ 0.86 = **24.5 and
  24.7 GB/s** (*device-derived*).
  - Cross-check on the GPU-only arm: 24.88 ÷ 0.875 = 28.4 GB/s; the `x3 alone` cell prints `dev_gbs` 28.3 to
    29.4 for the same shape (*device*). The two agree, so my round-1 F5 assumption (same `dev_us` in `tog`) holds.
  - Caveat: in the sync7 cells `gpu_busy` reads above dev ÷ wall (66 against 46 % on Llama-2). If it also
    overstates in the `both` arms, the device-time rate is higher than 24.6, not lower (bounded by 28.3).
- A stage model that charges K per launch must use the device-time rate; otherwise the gap is in both terms.
- What the CPU reads when the GPU has no gaps is not measured. Three readings: 18.9 (CPU loss proportional to GPU
  traffic), 19.3 (centre), 20.7 (R72's cell with the GPU reading 25.8 GB/s, *device*).

Her own script with `G_EFF`, `RC` and `SUM` set from these rates, nothing else changed (*estimated*; P_G / centre
stack):

| K µs | as committed (21.53 / 19.77) | device-time rate, centre (24.6 / 19.3) | low (24.6 / 18.9) | high (24.6 / 20.7) |
|---|---|---|---|---|
| 0 | 10.34 / 12.23, 9.53 / 11.87, 9.22 / 11.44 | 10.86 / **12.74**, 10.05 / **12.39**, 9.67 / **11.90** | 12.66, 12.31, 11.83 | 13.02, 12.66, 12.15 |
| 30 | 12.02, 11.69, 11.27 | 12.50, 12.18, 11.71 | 12.42, 12.10, 11.64 | 12.78, 12.46, 11.96 |
| 100 | 11.57, 11.30, 10.91 | 11.97, 11.72, 11.29 | 11.89, 11.64, 11.22 | 12.25, 12.00, 11.55 |
| 300 | 10.51, 10.37, 10.06 | 10.75, 10.64, 10.31 | 10.68, 10.57, 10.24 | 10.99, 10.88, 10.53 |
| 452 | 10.13, 9.87, 9.60 | 10.30, 10.07, 9.79 | 10.25, 10.00, 9.72 | 10.50, 10.32, 10.01 |

(lane 0.584, κ 0.560 at the centre; stack columns only where one number is given.)

Stack = 12.00 is reached at:

| | Yi | Llama-2 | DeepSeek |
|---|---|---|---|
| as committed | K ≤ 33 µs | never | never |
| device-time rate, centre | K ≤ 96 µs | K ≤ 56 µs | never (11.90 at K 0) |
| low | K ≤ 85 | K ≤ 44 | never |
| high | K ≤ 135 | K ≤ 99 | K ≤ 23 |

- **Her §1 item 1 ("no design reaches 12 on Llama-2 or DeepSeek at any join cost") rests on the wall rate.** With
  the device-time rate, Llama-2 has a line (about 55 µs) and DeepSeek sits 0.1 below 12 at the centre.
- This does not move the campaign verdict: every row still needs the trained head, C7, C5 and O 5.2 at their
  centres, and a join under 100 µs, none of which exists.
- It does move where the lane's ceiling is, and so what R75 should print: **the GPU's device time inside the
  `both` and `split` cells** (events on the profiling queue). With that, the device-time rate is read, not derived.
- The correction is small at the measured K (+0.17 to +0.20 tok/s at 452 µs). It matters only for the low-K
  designs (C, F).

### A.3 Finding 2: the κ check in §2.3 is not a check at the measured share, and the split cell holds a jitter term

- Her model is t(f) = max((1 − f)·A·RC, f·A·RG + K). Her check "0.621 × 0.948 + 0.521 × 0.530 = 0.865" uses the
  **balanced** formula, which corresponds to share 0.297 at K 530. The cell was measured at 0.256 (2,816 rows).
- At the measured shares her t(f) gives, with K 530:

  | share | her t(f) (CPU side, GPU side) | measured `ms_per_mat` | measured `cpu_ms` | measured `gpu_ms` |
  |---|---|---|---|---|
  | 0.256 | max(0.915, 0.819) = 0.915 | 0.870 / 0.860 | 0.820 / 0.823 | 0.719 / 0.796 |
  | 0.349 | max(0.801, 0.924) = 0.924 | 0.904 / 0.900 | 0.748 / 0.783 | 0.839 / 0.832 |
  | 0.500 | max(0.615, 1.095) = 1.095 | 1.069 / 0.866 | 0.623 / 0.664 | 1.010 / 0.857 |

  The CPU side is 10 % faster than RC says at 0.256 (0.82 against 0.915), and about as slow as RC says at 0.50.
- Reading: the CPU is slowed only while the GPU kernel executes, not during the GPU's dead time. At share 0.256
  the kernel runs for about 0.28 of the 0.82 ms.
  - "Slowed only during device time" predicts `cpu_ms` 0.768 / 0.706 / 0.603; measured are +0.02 to +0.08 above.
  - "Slowed throughout" (hers) predicts 0.915 / 0.801 / 0.615; measured are −0.09 to +0.05 around it.
  - Neither fits all three shares. At 0.256 the CPU's slow-down during the dead time comes out near 1.10
    (*device-derived*, one cell), which is **κ about 0.44**, between her centre 0.52 and her best 0.38.
- **A term neither model has: jitter.** At share 0.256 the GPU side ends 25 to 100 µs before the CPU side on
  average, yet `wait_ms` is 35 and 68 µs. `wait_ms` is the mean of max(0, GPU end − CPU end) per matrix, so the
  join's spread is being paid. `x3 rt` at the same rows reads p50 710, p90 865 µs (*device*): a spread of 100 to
  200 µs.
  - At a balanced static share the expected stage time is E[max(CPU, GPU)], about 0.4 × the spread above the
    mean: **40 to 80 µs per split stage** (*estimated* from the two fields).
  - This is why her check needs K 530 while the GPU-side K that her own model implies is lower: `gpu_ms` −
    f·A·RG = 437 / 514 (0.256), 444 / 437 (0.349), 446 / 293 (0.50), centre **about 445 µs** (*device-derived*).
    Her 500 (480 to 557) subtracts the GPU's time alone, not its time beside the CPU.
- Consequence for the ladder:
  - E's price is about right (K 500 at κ 0.52 ≈ K 445 at κ 0.44 plus 60 µs of jitter per stage).
  - The lower rungs are not. Jitter does not shrink with the mean. If the spread stays at 100 to 200 µs, a design
    with mean K 120 still pays 40 to 80 µs per stage on top: C's centre stack falls from 11.45 / 11.19 / 10.82 to
    about 11.0 / 10.8 / 10.4 (*estimated*: 128 × 0.06 ms = 7.7 ms at width 1, scaled by the stack's τ).
  - Only a design without a driver call per stage (F, my Y15) or with dynamic balance (rows claimed at run time)
    removes it.
- R75's split re-run should print the per-matrix spread of `gpu_ms` (p10, p50, p90), not only means.

### A.4 Finding 3: "launches are never back to back" is too strong (her §1 item 5, §2.2 reading)

- Her `tog` row shows a gap of 112 to 135 µs per launch at `gpu_busy` 86 to 89 %; her bullets then say "the GPU
  is idle during the gap: the kernels of a batch are not run back to back" and "145 to 265 µs per launch is spent
  with no wait at all".
- Both hold for the `x3 alone` cells. They do not hold for `tog`: there the gap (112 to 135 µs) is nearly all
  feeder CPU time (94 to 117 µs, her column), and the rest is under 20 µs once the one finish per 8 is spread.
- So R74 already has one state in which the queued launch cost is **about 110 µs, almost all of it host CPU on an
  A55**, and the submit→start latency is hidden behind the previous kernel. The low end of "launch cost with no
  wait" is 110, not 145; and the low end of B+ should be 150, as I wrote in round 1 (§6.1).
- What differs between the two states is not known (the feeder's clock range 691–1,094 against 806–1,804 MHz, and
  the unreadable GPU clock). R75's launcher-core cell decides it.

### A.5 Smaller points

1. **Batch 2 on the GPU is assumed, not measured; but the kernel exists.** Every stack number uses lane(2) with
   the GPU slowed by 1.45 (memo-10's rule for the CPU kernel). R74 ran the one-column kernel only.
   - *Source* (read in `lab/llama.cpp-base` at `620d2225c`, and in `patches/llama.cpp-c3/0005-x3-gpu-probe.patch`
     lines 347 to 366, which embeds the file verbatim): the same file `gemv_noshuffle_q4_0_f32.cl` holds a second
     kernel, **`kernel_gemv_noshuffle_q4_0_f32_mc3`**, "Multi-column (N in [2..4]) variant of the q4_0 decode
     GEMV, for the speculative / MTP verify batch". It loads each block's weights once and reuses them across the
     columns; upstream's comment says each column is byte-identical to a standalone gemv. Same weight image, same
     scale buffer, same work sizes; 8 arguments. Upstream switches it on with `GGML_OPENCL_Q40_MC3`.
   - So the program R74 built on the phone (`x3 build rc=0`) already contains it. It was never created or run.
     A probe cell needs `CreateKernel` by that name, an N-column activation image and one identity line.
   - Its cost cg (device time at N = 2 ÷ one column) is the unknown. If the phone ran the one-column kernel twice
     instead, the memo-10 composition gives lane(2) 0.766 and a zero-join stack of 11.36 / 11.00 / 10.65 against
     12.23 / 11.87 / 11.44 (*estimated*); A.6 re-prices both cases with a shared bus.
2. **GPU buffers "replace a CPU copy one for one" only for a static share.** True for a file-backed body whose
   lane rows the CPU never touches. A design that balances shares at run time needs the overlap rows in both
   places. So dynamic balance (the cure for jitter in A.3) and her memory column pull against each other. F and
   Y15 would need the overlap; E to C do not.
3. **B+ fast-core cost.** She charges 10 to 25 µs of one worker. In a balanced stage the launching worker's rows
   are late by the launch time; the other three wait at the barrier. The cost is the launch's host time (80 to
   196 µs) ÷ 4 threads of lost work = 20 to 49 µs of stage time, about twice her figure. Small, same sign.
4. **F's watchdog.** Her 2,000 ms command timeout (older tree, *external*) is well above one token (80 to 150 ms).
   The live risk is the screen: a kernel that holds the GPU for a whole token blocks composition. Per layer (about
   3 to 4 ms) is the safer grain, at one launch per layer (30 to 32 launches per token).
5. **P(F) 0.08, P(C) 0.15.** Mine: F 0.10, C 0.10 (C needs the 199 µs to be removable; the `tog` state says it
   can be hidden behind a running kernel, which helps queued launches and not a lane's isolated ones).
6. Her §5 device cell 8 (mailbox visibility) and my X4-e are the same cell. Her kill line (100 µs) is looser than
   mine (50 µs); with A.2's table Yi's line is about 96 µs, so **100 µs is the right kill line**. I withdraw 50.

### A.6 Finding 4 (the largest): the stack multiplies two slow-downs that cannot both apply. Width is cheaper inside the lane

This is about `bonnie_tree.best`, which her file and R74-result §4 both use for the "centre stack" column. I
reviewed that model in memo-10 and did not see this then.

- The stack charges a width-2 pass as `body × lane(2) × kB(2)` = body × 0.704 × 1.124 = body × 0.791.
  - kB(2) = 1.124 is the **CPU-alone** slow-down: with the whole bus (29.9 GB/s) the CPU is compute-bound at two
    positions (1.45 ÷ 40 against 1 ÷ 29.9 per byte).
  - lane(2) keeps the CPU at its **contended** W = 1 rate and slows the GPU by 1.45.
  - Together: the CPU is charged the bus loss from the GPU (25.6 → 19.8 GB/s) **and** the compute loss of a CPU
    that has the bus to itself; the GPU is slowed by 1.45 and then by 1.124 again.
- A reader that is bus-bound has compute slack, and a reader that is compute-bound does not load the bus. Beside
  the GPU the CPU's bus time per byte is 1 ÷ 21.5, above its width-2 compute time 1.45 ÷ 40. So the second
  position rides on slack the lane created. And when the GPU slows at width 2, it frees bus for the CPU.
- A model with one shared bus ([ebony_pair.py](ebony_pair.py), self-checked: it reproduces `x3 tog` at W 1 by
  construction and her 12.23 / 11.87 / 11.44 when switched to her composition). CPU rate from memo-09 §10.3's fit
  with its bus share as R; GPU rate min(24.6, GC ÷ c(W)), GC = its compute ceiling (28.3 = `dev_gbs` alone,
  *device*; band 24.6 to 40); bus sum 46.1 GB/s, calibrated on the W 1 cell (R72's read sum is 46.5, *device*):

  | | W 1 sum | W 2: CPU / GPU / sum | pair kB(2) | her kB(2) × lane(2) ÷ lane(1) |
  |---|---|---|---|---|
  | C 40, GC 28.3 (centre) | 43.5 | 21.4 / 19.5 / 40.9 | **1.06** | 1.27 |
  | C 40, GC 24.6 (GPU compute-bound already) | 43.5 | 22.3 / 17.0 / 39.3 | 1.11 | 1.27 |
  | C 40, GC 40 | 43.5 | 18.0 / 24.6 / 42.6 | 1.02 | 1.27 |
  | C 35 (the clocks of the measured bodies, my memo-10 position) | 43.5 | 19.7 / 19.5 / 39.2 | 1.11 | 1.41 |
  | C 24 (L4 caps, X1 thread on a small core) | 43.5 | 14.5 / 19.5 / 34.0 | 1.28 | 1.57 |
  | C 40, GPU runs the one-column kernel twice (cg 2) | 43.5 | 22.4 / 14.2 / 36.6 | 1.19 | 1.39 |

  GB/s of weight bytes per pass, *estimated*. In her composition the W 2 sum is 32.4.

- Centre stack with the pair model (trained head as measured on the host, C7, C5, O 5.2 as in her rows; κ 0.45;
  stage left on the CPU where a split does not pay; *estimated*):

  | K µs per join | memo-10 composition | pair, C 40, GC 28.3 | pair, C 35 | pair, C 24 | pair, C 40, + 60 µs jitter per split stage | pair, C 40, GPU two launches |
  |---|---|---|---|---|---|---|
  | 0 | 12.23 / 11.87 / 11.44 | 14.89 / 14.57 / 13.84 | 14.38 / 14.05 / 13.38 | 12.73 / 12.38 / 11.89 | 13.73 / 13.54 / 12.93 | 13.55 / 13.20 / 12.63 |
  | 120 | 11.40 / 11.15 / 10.78 | 13.84 / 13.64 / 13.01 | 13.40 / 13.18 / 12.61 | 11.95 / 11.70 / 11.28 | 12.83 / 12.74 / 12.20 | 12.67 / 12.44 / 11.94 |
  | 220 | 10.80 / 10.62 / 10.29 | 13.07 / 12.95 / 12.40 | 12.67 / 12.54 / 12.03 | 11.37 / 11.19 / 10.81 | 12.17 / 12.14 / 11.66 | 12.02 / 11.89 / 11.42 |
  | 300 | 10.46 / 10.31 / 10.01 | 12.51 / 12.45 / 11.95 | 12.15 / 12.07 / 11.60 | 10.95 / 10.81 / 10.47 | 11.73 / 11.74 / 11.30 | 11.55 / 11.52 / 11.04 |
  | 452 (measured) | 10.06 / 9.79 / 9.53 | **11.65 / 11.66 / 11.23** | 11.26 / 11.26 / 10.87 | 10.23 / 10.16 / 9.87 | 11.20 / 11.15 / 10.77 | 10.91 / 10.88 / 10.49 |
  | **12.00 on the slowest subject at** | never | **K ≤ 290** | K ≤ 224 | never | K ≤ 156 | K ≤ 108 |

  (The "two launches" column does not charge the second launch's own gap, so it is optimistic.)

- What this says, if the pair model holds:
  - the lane as measured (K 452) with every other lever at its centre prices about 11.2 to 11.7, not 9.5 to 10.1;
  - **12 on all three has a join line of about 150 to 300 µs, not "none"**, at C 35 to 40;
  - it is gone again at C 24. The clock caps of a sustained decode decide it, as in every earlier memo.
- What it does not say: nothing here is built. The row needs the trained head on the device (top-1 0.379 / 0.497 /
  0.438 are *host*, DeepSeek's is the mean of the other two), the W-position CPU kernel (exact on random data on the
  host only), the multi-column GPU kernel (in the built program, never run; A.5 item 1), C7, C5 and O 5.2 (unbuilt), and a balanced lane with four
  joins per layer. P(the pair model's kB(2) ≤ 1.15 on the phone) 0.55, my judgement; P(all three at 12 | that, and
  K ≤ 250) perhaps 0.1. The campaign verdict is the lead's; I only say the stack column is model-dependent by
  ± 2 tok/s and one device cell decides which model is right.
- **The deciding cell** (model-free, about 3 phone-minutes; **not in R75's scope**, so it needs the owner's word):
  `x3 tog` at width 2. CPU arm: the W-position gemv at widths 1 and 2 on cpus 4-7. GPU arm: the same matrix
  launched twice per batch slot, and upstream's multi-column kernel at N = 2 and 3 (already in the built program,
  A.5 item 1) with its identity line and device time. Arms: CPU alone, GPU
  alone, both; 20 s each, ABBA, caps logged.

  | reading (weight GB/s per pass, both arms summed) | pair model | memo-10 composition |
  |---|---|---|
  | W 2, GPU two launches | 36.6 (33 to 39) | 29.8 |
  | W 2, upstream multi-column kernel | 40.9 (38 to 43) | 32.4 |
  | GPU alone, device time at N = 2 ÷ N = 1 (cg) | 1.2 to 1.6 (no model; her §4 of round 2 sweeps it) | same |
  | CPU arm alone, W 2 ÷ W 1 | 0.89 (both models; this is T2 of memo-10) | 0.89 |

  Kill line for the pair model: the two-launch sum under 33, or the two-column sum under 35.

### A.7 Positions, side by side (for the memo)

| item | Bonnie | Ebony |
|---|---|---|
| κ | 0.52 (0.38 to 0.58) | 0.44 at the measured share, plus a jitter term of 40 to 80 µs per split stage |
| in-situ K | 500 (452 to 560) | 445 (290 to 515) on her own model's definition |
| GPU rate in the stage model | wall, 21.53 | device time, 24.6 (*device-derived*); wall double counts the gap |
| ceiling at K 0 (centre stack) | 12.23 / 11.87 / 11.44 | 12.74 / 12.39 / 11.90 (12.66–13.02 / 12.31–12.66 / 11.83–12.15) |
| "12 on Llama-2 at some K" | no | yes, K ≤ about 55 µs (44 to 99), every other lever at its centre |
| ladder B+ → A → C centre K | 350 → 290 → 120 | 350 → 320 → 220, and jitter on top below about 300 |
| queued launch, best state | 145 to 265 µs | 110 µs in `tog`, nearly all host CPU |
| width inside the lane (the stack column) | body × lane(2) × kB(2) = × 0.791 | one shared bus: × 0.589 × 1.06 = × 0.63 at C 40 (0.65 at C 35, 0.75 at C 24) |
| centre stack at the measured K 452 | 10.06 / 9.79 / 9.53 (her E: 10.01 / 9.72 / 9.47) | 11.65 / 11.66 / 11.23 at C 40; 11.2 / 11.15 / 10.8 with jitter; 10.2 / 10.2 / 9.9 at C 24 |
| join line for 12 on all three | none | 150 to 300 µs at C 35 to 40; none at C 24 |
| campaign verdict | unchanged | the lead's; the stack column moves by ± 2 tok/s with the model, one device cell decides (A.6) |

---
