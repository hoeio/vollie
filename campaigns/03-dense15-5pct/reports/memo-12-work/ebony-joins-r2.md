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

## B. Review of bonnie-joins-r2.md, and R75 bands (item 2)

Read in full, script run (`python3 -I -B bonnie_joins_r2.py`, self-checks pass). I also read Tom's probe source
`patches/llama.cpp-c3/0007-x4-join-probe.patch` for what each field measures.

### B.1 Her §0, §1 and §3: where we now stand

- **Converged, reached separately:** the multi-column kernel `_mc3` is in the built program (her §0 item 5, my A.5);
  the committed stack penalises the GPU twice at width 2 (her §3.1, my A.6); the gap-free lane factor is below
  0.621 (her §3.3: 0.57 to 0.63; my A.2: 0.589); F5 is confirmed by `gpu_busy` (her N1, my A.2).
- **Her N3 is right and I had it wrong in round 1:** the R74 split feeder sleeps in `clFinish`, so that cell says
  nothing about a hot launcher.
- **Her §1.3 item 1 (a launching worker pays the launch in series, K_eff = K + 0.92 × L_h): accepted.** It
  replaces my A.5 item 3.
- **Her §1.3 item 2 (the kernel side may run on the little cluster): accepted as a third hypothesis.** kgsl's
  dispatcher threads and the GPU interrupt are not pinned by us. My split: little cluster 0.45, launcher's core
  0.35, GPU or fixed 0.20 (hers 0.5 / 0.3 / 0.2). Not a real difference.
- **Still different: how much the width-2 correction is worth.** Same direction, different size.

  | term at width 2 | Bonnie r2 (§3.1, §3.2) | Ebony (A.6, `ebony_pair.py`) |
  |---|---|---|
  | GPU rate beside the CPU, W 1 | 21.53 (wall, with the `tog` gaps) | 24.6 (device time); gaps are charged as K |
  | GPU at W 2 | 21.53 ÷ cg, cg swept 1.07 to 2.0, centre 1.2 | min(24.6, GC ÷ 1.45), GC 28.3: 19.5. In her terms cg 1.26 on the device-time rate |
  | CPU at W 2 beside the GPU | 18.73: its bus share stays what it was at W 1 | 21.4: the bus the slower GPU no longer uses goes back to the CPU |
  | body factor at W 2 | 0.764 (cg 1.45), 0.70 (cg 1.2) | 0.627 (0.60 to 0.66) |
  | stack at K 0 | 12.60 / 12.24 / 11.77 (cg 1.45); 13.58 / 13.24 / 12.66 (cg 1.2) | 14.89 / 14.57 / 13.84 (C 40); 14.38 / 14.05 / 13.38 (C 35) |
  | join line for 12 on all three | 104 µs (cg 1.2), 184 (cg 1.08), none (cg 1.45) | 290 µs (C 40), 224 (C 35), 156 with 60 µs of jitter per stage, none at C 24 |

  - Her attack request (§3.1): the correction is right, and it is too small. It removes the second GPU penalty
    and lowers the CPU's kB, but keeps the CPU's bus share fixed. The CPU lost 5.9 GB/s **because** the GPU read
    21.5; when the GPU reads less, that loss shrinks. P(the CPU regains most of it) 0.7, my judgement.
  - Both of us now say the same qualitative thing: **with `_mc3` at cg near 1.2 to 1.3, 12 on all three has a
    finite join line, somewhere between 100 and 300 µs**, every other unbuilt lever at its centre. R75's likely
    in-situ K (hers 300, mine 290) sits at the top of that range.
  - Joint probability: hers about 0.01, mine about 0.03 (the difference is the model row above). Neither is a
    verdict.
- **Her §3.3 "head on the lane" and "Yi at width 1":** agreed as priced; not checked further.

### B.2 Three things the kit as built will not show (for the lead and Tom; not conditions of the bands)

1. **x4e's kernels use the OpenCL 1.x atomics (`atomic_add`, `atomic_xchg`) on a `volatile __global int *`.**
   *External, read first-hand* (Qualcomm 80-NB295-11 Rev. C, §7.5): "Starting from Adreno A5x GPUs, the coarse-
   grain SVM and the more advanced fine-grain buffer SVM with atomics are all supported"; for data "modified by
   the host and the device needs to use the new data, then atomic or fence is required", the buffer "must have
   the flag CL_MEM_SVM_ATOMICS", and "Inside the kernel, memory_scope_all_svm_devices must be used" with the
   C11-style atomic functions.
   - So the vendor documents the mailbox, and documents it with C11 atomics at all-SVM-devices scope. The kit's
     `svm` mode sets the atomics flag when offered but reads with a 1.x atomic, whose scope is the device.
   - Reading rule I ask for in the pre-registration: **"flag not seen" in x4e closes F only if a C11
     `atomic_load_explicit(..., memory_order_seq_cst, memory_scope_all_svm_devices)` variant was also run.** If
     Tom can still add that variant (a few lines; the program is already built with `-cl-std=CL2.0`), it should
     go in. Otherwise "not seen" is "not decided".
   - This answers her r1 question 4 (any published case of a kernel polling host-written memory on a mobile
     driver): the vendor guide, as a supported feature, with a warning that its "hidden cost" is high.
2. **No cell hands the stage's input to the GPU or reads its output back.** x4s launches on a fixed `x4` image
   and never reads `dst` (source read). A lane writes 16 to 44 KB of input before every launch and reads its rows
   back after. Qualcomm's guide §7.7 (read first-hand): "there could be a CPU cache flush on an input buffer
   right before a kernel is launched. CPU cache operations have a very measurable cost, which can be observed as a
   delta between CL_PROFILING_COMMAND_QUEUED and CL_PROFILING_COMMAND_SUBMIT"; and the cache "must be invalidated
   when mapping the output buffer of a kernel for reading by the host". So R75's in-situ K is a lower bound for a
   real lane's K by the hand-off, size unknown (my guess 20 to 80 µs per stage with map / unmap; near 0 with
   `cl_qcom_ext_host_ptr_iocoherent` or fine-grain SVM buffers, both reported or documented for this GPU). A
   later kit needs this cell; if r35 is not frozen, a lead-mode arm that rewrites `x` and reads `dst` per matrix
   would give it.
3. **No width-2 cell** (her §2.6 item 2, my A.6): `_mc3` at n_cols 2 against n_cols 1. I support her request and
   add the `tog`-style sum (CPU W-position kernel beside it), which separates her row of B.1 from mine. It is
   outside R75's written scope (join cost), so it is the owner's and the lead's call.

Also from the same guide, for the readings: §4.5.1 says the QUEUED→SUBMIT delta is "the software overhead and the
overhead of CPU cache operations", that the driver "may queue the kernel first and submit it along with several
following kernels in the queue later", and that `clFlush` "speed[s] up the submission"; §4.5.4 says "Without the
performance mode settings, the first OpenCL kernel in a sequence typically shows more significant launch latency
and slower execution time" (performance mode needs root); §9.1.1 says perf hint HIGH is the default and forces
the GPU's maximum frequency, "just a hint" that thermal control can override; §9.1.3 says a recording "can save
CPU power and improve dispatch latency", only `clEnqueueNDRangeKernel` can be recorded, and it is replayed on a
separate in-order queue with per-kernel argument updates.

### B.3 Bands, cell by cell (her §2.1 and §2.2)

"Agree" = I would write her band. Where I give my own, the merged band should span both and the two centres be
printed. All *estimated*.

Gating events (her §2.1):

| event | hers | mine | note |
|---|---|---|---|
| recordable queue in the extension string | 0.99 | agree | R72 printed it |
| `x4 recq present=1`, `max_size` > 0 | 0.80 | agree | |
| `svm_coarse` / `svm_fine_buffer` / `svm_fine_system` | 0.85 / 0.30 / 0.05 | **0.90 / 0.70 / 0.05**; `svm_atomics` 0.65 | the vendor guide says A5x and later support fine-grain buffer SVM with atomics. **Disagree** on the middle value |
| x4d runs | 0.55 | agree | |
| x4d passes (replay ≤ 0.5 × sync1) | 0.25 | 0.20 | the guide promises less CPU work, not a shorter submit→start |
| x4c identity ok on both stacked shapes | 0.90 | agree | |
| x4e flag seen within 1 ms, any mode | 0.35 | **0.40** as built (1.x atomics); 0.6 with the C11 variant | B.2 item 1 |
| x4e all three lines pass | 0.10 | 0.12 | |
| x4e a child hangs or times out | 0.35 | agree | |
| x4a the driver serialises (`submit_us(i)` ≥ `end_us(i−1)`) | 0.60 | 0.50 | the guide says it may batch submissions; the `tog` state shows gaps of 110 µs, which a strict serialiser with a 199 µs submit→start would not give |
| x4b best `ovh_us` ≤ 250 / ≤ 150 / ≤ 100 | 0.80 / 0.12 / 0.02 | **0.75 / 0.20 / 0.03** | I give the launcher hypothesis more weight at the low end. Small disagreement |
| x4b 0-3 spin submit→start ≥ 30 µs below cpu 4 spin | 0.50 | 0.45 | |
| x4f the driver prepares at enqueue | 0.40 | 0.30 | cache operations and submission are tied to the submit by the guide's wording |
| x4r held | 0.35 | **0.25** | R74's 3.5 GB arm ended at 0.59 GB available with slowing passes; this one is 4.1 GB |
| x4r process killed | 0.10 | 0.15 | |

Cells (her §2.2):

| cell, field | hers (centre; 80 %) | mine |
|---|---|---|
| x4a `gap_p50_us`, launcher 0-3 | 200; 145–270 | centre agree; **low end 110** (the `tog` state) |
| x4a `gap_p50_us`, launcher cpu 4 | 210; 140–285 | **185; 100–285** |
| x4a flush after each enqueue: change of gap | −10; −60 to +30 | agree |
| x4a `s2b_first_us` | 190; 130–220 | agree |
| x4a `enq_us` per launch, 0-3 / cpu 4 | 125 / 62; 85–185 / 35–115 | agree |
| x4a Llama-2 layer of 7, per launch, 0-3 / cpu 4 | 235 / 215; 150–300 / 130–290 | agree; cpu 4 centre 200 |
| x4b `ovh_us` 0-3: finish / spin / wait | 400 / 265 / 395 | agree |
| x4b `ovh_us` cpu 4 and cpu 7 | 310 / 255 / 305; 295 / 250 / 290 | spin cells **235 (130–345)**; the others agree |
| x4b best of nine | 215; 140–280 | **205; 120–290** |
| x4b `queued_to_start_us`, `submit_to_start_us` | as her table | agree (bimodal, as she says) |
| x4b p90 − p50: finish / spin | 150 / 90 | agree. **This is the jitter term of A.3; please keep it in the pre-registration as a scored row** |
| x4b launcher CPU, `dev_us` | 300; 240 | agree |
| x4f `set_to_start_p50_us` 0-3 / cpu 4 | 190 / 210; 110–280 | agree; centre 210 / 215 at my lower P(prepare at enqueue) |
| x4f `rtargs` 3 against 15 args | −8; −30 to +5 | agree |
| x4c gate‖up cold GB/s; ratio to 11008×4096 | 24.3; 1.14 (1.04–1.20) | **23.5; 1.10 (1.00–1.20)**: R74's launch gap grew with the buffer (199 → 320 µs from 2.4 to 25 MB), so I add about 60 µs at 50.7 MB |
| x4c q‖k‖v | 21.6; 1.42; 1.02 | agree |
| x4d replay overhead per launch; ratio to sync1 | 138; 0.87 (26–230; 0.74–0.97) | **170; 0.80 (40–250; 0.55–0.97)**: I expect the launcher's CPU share (about 55 % of the gap) to shrink, not the rest |
| x4e flag `rtt_p50_us` (given seen) | 30; 5–300 | agree |
| x4e counter `iv_p50_us` | 10; 1–100 | agree |
| x4e `q4p48_vs_q4n` | 0.85; 0.50–1.02 | agree |
| x4s `t_cpu_alone_ms` | 1.90; 1.80–2.05 | agree |
| x4s feeder anchor, share 0.25 | 0.86; `gpu_ms` 1.00; wait 0.02 | agree |
| x4s lead mode, best ratio | 0.735; 0.69–0.80 | **0.72; 0.67–0.80**; best share 0.50 (0.55), 0.45 (0.35) |
| x4s lead mode, share 0.20 / 0.35 | 0.92 / 0.82 | agree |
| x4s share 0.50: `gpu_ms`; `gpu_dev_ms`; `wait_ms`; `enq_us` | 1.29; 0.985; 0.04; 65 | `gpu_dev_ms` **1.03 (0.92–1.10)** (28.3 → 24.6 GB/s is × 1.15, not × 1.10); `wait_ms` **0.08 (0.02–0.25)** (jitter, A.3); the rest agree |
| in-situ K = `gpu_ms` − `gpu_dev_ms` | 300; 200–380 | 290; 170–400 |
| x4r (if held) | 18.5; 0 | agree |

Rows I ask the lead to add (new, mine; *estimated*):

| row | band | why |
|---|---|---|
| x4s lead mode: GPU device-time rate beside the CPU = GPU bytes ÷ `gpu_dev_ms`, best share | 24.6 GB/s; 22.5–27.5 | A.2: decides the gap-free lane factor (0.57 to 0.63) |
| x4s lead mode: `cpu_ms` ÷ (CPU rows' share × `t_cpu_alone_ms`) at share 0.25 / 0.50 | 1.14 / 1.30; 1.08–1.22 / 1.20–1.40 | A.3: is the CPU slowed only while the kernel runs (κ 0.38 to 0.45) or throughout (0.52) |
| x4a: `gap_us` against buffer size (2816 rows against the layer's 11008 × 4096 launches) | larger buffer + 40 µs; 0 to + 120 | the guide's cache operations; decides whether stacking pays per launch |

### B.4 Her §2.3 (kill and build lines): agree, with two changes

- **F / Y15:** the mailbox kill line is 100 µs for flag and counter together (A.5 item 6), and "not seen" kills
  only with the C11 variant run (B.2 item 1).
- **B+ "worth a build" at best `ovh_us` ≤ 250:** keep. In my pair model 250 to 300 µs in situ is where 12 on all
  three becomes a centre price, so the same line is also the line for "the width-2 cell must be run next".
- Her mapping §2.4 (K = `ovh_us` + 40): keep, and add the hand-off of B.2 item 2 as an unknown above it.

---

## C. Scouting: what moves 7B dense decode past the lane's ceiling (item 3)

Checked against `docs/negative-results.md`, memo-03, memo-09 §5.3 and §10.7, memo-10, memo-11 before writing; a
reopened item says why. Prices are ms per token against the centre stack unless said; all *estimated*.

### C.1 Summary table

| # | idea | price | P | kill test | status |
|---|---|---|---|---|---|
| S1 | **Width 2 inside the lane on upstream's `_mc3` kernel** (A.5, A.6; Bonnie r2 §3) | −2.5 to −15 ms against the committed stack at K 0 (Bonnie r2: −2.5 at cg 1.45, −8 at cg 1.2; mine −15) | 0.55 that pair kB(2) ≤ 1.15 | the width-2 `tog` cell of A.6: sum under 33 (two launches) or under 35 (`_mc3`) GB/s | **new; the largest open term.** Needs a device cell outside R75's scope |
| S2 | Gap-free lane factor (device-time rate) | −2.6 to −4.5 ms at low K; under 1 ms at K 450 | 0.7 | x4s `gpu_dev_ms` (row added in B.3) | new, small; rides in R75 |
| S3 | Fine-grain SVM with C11 atomics as the mailbox of the persistent kernel (F / Y15) | makes K 30 reachable: −13 to −17 ms against K 290 | 0.20 for the whole of F (was 0.10) | x4e with the C11 variant (B.2 item 1): flag + counter over 100 µs, a hang, or looped rate under 0.8 × | **source found** (vendor guide §7.5); the kit's variant may not be the documented one |
| S4 | Zero-copy input and output (`cl_qcom_ext_host_ptr_iocoherent`, AHardwareBuffer or SVM) for the stage hand-off | avoids an unmeasured + 20 to 80 µs of K per join (2 to 7 ms per token at 128 joins, κ 0.45, τ 1.4) | 0.6 that the hand-off is under 20 µs with it | a lead-mode split arm that rewrites `x` and reads `dst` per matrix, with and without the io-coherent buffer | new cost, not a gain; no cell in r35 (B.2 item 2) |
| S5 | Dynamic balance against jitter: the GPU's rows are split in two launches, the second sized from the first's finish time; or rows claimed at run time in the persistent kernel | −2 to −5 ms (40 to 80 µs per split stage, A.3) at the price of one more launch gap in the first form: a loss unless the queued gap is under 60 µs | 0.15 (launch form), 0.5 given F works | x4b p90 − p50 under 40 µs in the spin cells → nothing to gain | new; decided by R75's spread rows |
| S6 | Width 3 inside the lane (`_mc3` takes up to 4 columns) | +1 to +5 ms (a loss) at GC 28.3; −0 to −3 ms on Llama-2 only if the GPU's second and third columns are nearly free (GC 40) | 0.1 | the `_mc3` cell at n_cols 3: device time over 1.5 × one column closes it | width 2 stays the centre; memo-10's conclusion holds inside the lane too |
| S7 | Third reader: Hexagon 780 (HVX) beside CPU + GPU | −2.4 to −5.3 ms at zero dispatch (bus headroom 1.5 to 3 GB/s), before FastRPC | 0.03 | none worth a cell | **closed by the bus**, see C.2. Not reopened |
| S8 | GPU-side attention, norms, KV in GPU memory | under 1 ms; each op is tens of µs, below any K above 30 µs | – | – | closed by size (memo-08: under 3 % of bytes); only inside F |
| S9 | GPU drafts with a small model, CPU verifies (or the reverse) | needs a second model file: rule 1, owner question E5 of memo-10 | – | – | not proposed; "GPU verifies a second position" **is** S1 |
| S10 | Integer (dp4a) GPU kernel so that the lane spends no PPL budget | 0 ms; frees budget for C5 / C7 | 0.3 | host emulation of upstream's dp4a arithmetic against the CPU kernel, per-chunk PPL | note only: upstream has a dense Q4_0 dp4a kernel, but for prefill tiles of 32 columns (`gemm_noshuffle_q4_0_q8_1_dp4a.cl`, *source*), not a gemv |
| S11 | Head on the lane (Bonnie r2 §3.3) | −0.8 to −2.7 ms | 0.3 | Q6_K GPU kernel identity and numerics | hers; agreed |

### C.2 Third reader (Hexagon / HTP): numbers found, and why it stays closed

- *External, read through the fetch tool with quotes* (arXiv 2607.05475, July 2026, Snapdragon 8 Elite class
  phones, not ours): "the NPU's DDR bandwidth reaches only 39.7 GB/s, 1.4–1.5× lower than the 57–60 GB/s achieved
  by CPUs and GPUs"; "a single HVX worker reaches only about 11.6 GB/s for large regions"; "the NPU has only 1 MB
  L2 cache"; "each host–NPU round trip takes 20–30 µs"; Qwen2.5-7B decode on those phones: CPU llama.cpp 8.7 to
  14.3, NPU llama.cpp 6.4 to 9.2 tok/s. No Snapdragon 888 or v68 number exists in it.
- *External* (HeteroInfer, read): on an 8 Gen 3 two processors together reach 59.5 of an achievable 61.9 GB/s
  (96 %). Ours: CPU + gap-free GPU 43.5 of a measured 46.5 read sum (*device-derived* / *device*): 94 %.
- So a third reader has 1.5 to 3 GB/s to add: lane 0.589 → 0.569 to 0.551, **2.4 to 5.3 ms per token at zero
  dispatch** (*estimated*). FastRPC on an older stack is put at 200 to 300 µs per call by one report (*external*,
  search summary only, a Berkeley technical report), which at 64 to 128 calls is 13 to 38 ms. Net loss.
- The interesting number is the 20 to 30 µs round trip on a 2025 SoC: there the DSP's join is 15 × cheaper than
  our GPU's. That is not our kernel (5.4) or DSP (v68, HVX only, about 5 GB/s by memo-09's estimate), and
  reachability from a retail Samsung app is still not established (memo-09: P 0.5 to 0.6; I found nothing new
  today). **Not reopened.** It would be reopened only as the *second* reader if the GPU lane failed for a reason
  the DSP does not share.

### C.3 Other runtimes and papers read or located this round

| source | what it gives us | label |
|---|---|---|
| Qualcomm, *Snapdragon Mobile Platform OpenCL General Programming and Optimization*, 80-NB295-11 Rev. C (the PDF opened this time; text extracted locally) | §4.5.1 profiling stamps and what QUEUED→SUBMIT holds; §4.5.4 performance mode; §5.7.2 "Avoid blocking API calls", event lists, callbacks; §7.4 zero copy; §7.5 fine-grain SVM with atomics from A5x; §7.7 CPU cache operations at launch and at map; §9.1.1 perf hint; §9.1.3 recordable queues (API and limits). No latency figure in µs anywhere | *external*, **read first-hand** |
| llama.cpp base `620d2225c`, `gemv_noshuffle_q4_0_f32.cl` | `_mc3`, 2 to 4 columns, per-column identical to the one-column kernel by its comment; opt-in `GGML_OPENCL_Q40_MC3`; 8 subgroups for M under 4,096 | *source*, read locally |
| arXiv 2607.05475 "Is Your NPU Ready for LLMs?" | C.2's numbers; RPC polling default 9,999 µs in GENIE and MLLM, 5 to 20 µs polling cuts energy 31 to 38 % | *external*, read through the fetch tool (first 100,000 characters) |
| HeteroInfer (SOSP 2025) | CPU is "only a control plane" there; decode 14.01 tok/s on Llama-8B on an 8 Gen 3 with GPU + NPU. Nobody in the papers found splits a matrix between CPU and GPU: our lane has no published twin | *external*, read |
| Lever (arXiv 2605.16786), sd.npu (2510.15312), AHASD (2604.25326, DAC 2026), HELIX (2611.06513), HeteroMosaic (2607.12839, MICRO 2026) | speculative or heterogeneous decode on CPU-NPU, NPU-PIM, server GPUs, Ryzen AI. None on an Adreno 6xx, none gives a per-join figure, none shares one weight read between two processors | *external*, **abstracts only** |
| Sereno (OSDI 2026), EdgePace (arXiv 2610.37307) | memory-bandwidth contention on unified memory is treated as the first-order resource | *external*, **search summary only** |
| Dalek (arXiv 2508.10481) | OpenCL launch latency about 35 to 40 µs on Intel integrated GPUs, about 5 µs on a desktop GPU | *external*, **search summary only**; not Adreno |

Not done: MLC, ExecuTorch, mllm and PowerInfer-2 were not opened this round either; MNN's OpenCL LLM path was not
re-read for a multi-column 4-bit kernel; the last 24,000 characters of 2607.05475 were not read.

### C.4 Where this leaves the stack (my reading; the verdict is the lead's)

Pair model, C 35 (the clocks of the measured bodies), 60 µs of jitter per split stage, head top-1 as measured on
the host, C7, C5, O 5.2 (*estimated*):

| in-situ K | best width | Yi / Llama-2 / DeepSeek |
|---|---|---|
| 290 (R75's likely value) | 2 | 11.4 / 11.4 / 11.0 |
| 150 | 2 | 12.25 / 12.2 / 11.7 |
| 0 | 2 | 14.4 / 14.05 / 13.4 (no jitter) |
| 0, no guess head (width 1) | 1 | 12.5 / 11.3 / 11.3 |

Head top-1 needed for 12 at K 290 with jitter: 0.46 / 0.58 / 0.60; measured 0.379 / 0.497 / (0.438 assumed).
In the committed model the same row is about 10.4 / 10.2 / 9.9 (Bonnie r2 §0 item 3). The two models differ by
about 1 tok/s at R75's likely K, and the width-2 `tog` cell separates them.

## D. Next questions, ranked, and what I did with them

| # | question | state |
|---|---|---|
| 1 | Is width 2 nearly free inside the lane (pair kB(2) 1.06 against 1.27)? | **Priced** (A.6, `ebony_pair.py`); Bonnie reached a smaller version separately (B.1). Open: the device cell (A.6). Not in r35 |
| 2 | Does `_mc3` exist and fit the probe's buffers? | **Done:** yes, in the program R74 built (A.5 item 1). Open: its device time (cg) |
| 3 | Can a running kernel legally see a host write on Adreno? | **Done:** vendor guide §7.5, fine-grain SVM with C11 atomics (B.2 item 1). Open: x4e as built uses 1.x atomics |
| 4 | Does width 3 pay inside the lane? | **Done:** no at the centre (C.1 S6); only if extra GPU columns are nearly free |
| 5 | Is there room for a third reader? | **Done:** 1.5 to 3 GB/s, 2.4 to 5.3 ms; closed (C.2) |
| 6 | What does the per-stage input / output hand-off cost? | Source found (guide §7.7); **no number**; cell proposed (B.2 item 2, C.1 S4) |
| 7 | How large is the join's spread, and can a static share live with it? | Estimated 40 to 80 µs per stage (A.3); R75's p90 − p50 rows and x4s `wait_ms` decide |
| 8 | Do the clock caps hold at C 35 with CPU and GPU both loaded for a whole decode? | **Not done.** No source and no cell; R75's blocks are seconds long. The pair model's whole gain goes at C 24 (A.6). This is the question for the first real decode (R73, held) |
| 9 | Does the freed bus really return to the CPU when the GPU slows (B.1, the term Bonnie and I still differ on)? | Not decidable on the host. The width-2 `tog` cell answers it; a cheaper proxy already in R74: `tog` both arms at two GPU shapes would do, R74 ran one |
| 10 | MNN's and MLC's 4-bit OpenCL kernels at 2 columns as a second source for cg | not done |

## E. Lists for Tom (through the lead; additions to my round-1 §7 and Bonnie's r2 §4)

**Test list**

| # | test | where | cost | closes when |
|---|---|---|---|---|
| E-J7 | Width-2 `tog` cell: CPU W-position gemv at widths 1 and 2; GPU one-column kernel twice and `_mc3` at n_cols 2 and 3; alone and together, with `_mc3`'s identity line and device time | device, model-free; **needs the owner's word (outside R75's scope)** | about 3 phone-minutes; `_mc3` is already in the built program, the CPU kernel is `memo-10-work/ebony_gw.c` put into the probe | sum under 33 / 35 GB/s (two launches / `_mc3`) → the pair model is wrong and the committed composition stands; cg over 1.5 → Bonnie's r2 line "none on DeepSeek" stands |
| E-J8 | x4e variant with C11 atomics at `memory_scope_all_svm_devices` on a fine-grain SVM buffer with the atomics flag | device, r35 if not frozen | a few lines | not seen within 1 ms with this variant → F closed; without this variant "not seen" is not a kill |
| E-J9 | Hand-off arm: lead-mode split that rewrites `x` and reads `dst` per matrix, plain buffers against io-coherent or SVM buffers | device, a later kit (or r35 if open) | one arm pair | hand-off over 80 µs with every buffer type → add it to K in every price |
| E-J10 | Host: `_mc3` arithmetic in Tom's half-precision emulation gives per-column values equal to the one-column path (Bonnie r2 test 4; same test) | host | small | any difference → width 2 on the lane is its own numerics change |
| E-J11 | Host: re-run the stack with both width-2 forms (Bonnie §3.1, `ebony_pair.py`) and put the two columns side by side in R74-result §4 as a dated note | host | minutes | – |

**Try list** (all env-gated, off by default, model-generic, two subjects; labels relative to the lane, which is a
declared numerics change)

| # | lever | after | label | price | P |
|---|---|---|---|---|---|
| Y17 | Width-2 lane: CPU W-position kernel beside `_mc3` on the same row split | E-J7, E-J10; the trained head on the device | same arithmetic per column as the lane at width 1 (to be shown by E-J10) | −2.5 to −15 ms against the committed stack (C.1 S1) | 0.5 given E-J7 passes |
| Y18 | Zero-copy stage input and output (io-coherent host pointer or fine-grain SVM) | E-J9 | bit-identical to the lane | avoids + 2 to 7 ms | 0.6 |
| Y15 (changed) | Persistent kernel on fine-grain SVM with C11 atomics, one launch per layer, rows claimed at run time | E-J8, x4e parts 2 and 3 | same kernel arithmetic | K 30: −13 to −17 ms against K 290, and no jitter term | 0.20 |

Sources added this round (2026-11-23): the Qualcomm guide
<https://docs.qualcomm.com/bundle/publicresource/80-NB295-11_REV_C_Qualcomm_Snapdragon_Mobile_Platform_Opencl_General_Programming_and_Optimization.pdf>
(read first-hand); <https://arxiv.org/html/2607.05475> (read, first 100,000 characters);
<https://arxiv.org/html/2501.14794v2> (read); abstracts of <https://arxiv.org/abs/2605.16786>,
<https://arxiv.org/abs/2604.25326>, <https://arxiv.org/abs/2611.06513>, <https://arxiv.org/abs/2607.12839>; the
index <https://github.com/qhy991/Awesome-On-Device-Inference> (a curated list, not checked against its sources);
search summaries only for Sereno, EdgePace, Dalek, sd.npu and the FastRPC figure
(<https://digitalassets.lib.berkeley.edu/techreports/ucb/incoming/EECS-2022-144.pdf>).
