# Bonnie, memo-12 work, round 30: the reply count stored; an on-core X1 hot reading; the bus-against-compute rule bracketed

2026-11-24. Labels: *device* (phone), *host*, *estimated* (arithmetic), *source*. No web search (nothing is stated
about upstream's current state). R75's report and `tom-kai-stock.md` are still not in the repository.

Scripts (same folder): `bonnie_reply_count.py` (new), `bonnie_reader_r30.py` (new; loads `bonnie_ratio_r29.py`
unchanged). The X1 search of §2 is a read of committed probe JSON lines; the fields are named.

## 0. Answer first

1. **r29 §2 reproduces from the repository:** `bonnie_reply_count.py <repo root>` prints the 12 cells, 560 runs,
   one reply text per cell, and stops with an error if any cell ever holds two. **One correction to r29's
   table:** its "report files" column was counted on file names cut to 12 characters. The right counts are 9
   (not 8) for OLMoE off and Granite num at 3 threads, and 7 (not 6) for OLMoE num. Runs and texts are unchanged.
2. **There is no on-core X1 hot reading of the Q4_0 kernel in any committed report.** The thread meant for cpu 7
   ran elsewhere in every hot Q4_0 cell of R69, R69b and R72; R74 has no such cell. What exists (*device*, R72
   L5): two single-thread cells on other weight types where the thread was verified on cpu 7 at 2,841 MHz. They
   put the X1 at **about 1.4 to 1.5 times an A78** on plain (non-KleidiAI) kernels, with the A78 side inferred,
   not verified (§2.2).
3. **What that does to C** (hot compute rate of the four threads; *estimated* from *device* parts):

   | state | C |
   |---|---|
   | as R72 ran it: three A78 on-core, the fourth thread off cpu 7 | **36.9 GB/s** (*device*, L3 four-thread cell) |
   | X1 on-core, equal to an A78 | 46.1 |
   | X1 on-core at 1.4 to 1.5 × an A78 | 50.7 to 51.9 |

   The stacks' C 40 is inside this range and nearer the bottom. **All of these are at A78 2,227 MHz.** C scales
   with the clock, and the body was taken at caps of 1,555 to 1,785 MHz (r14 factor 1); that is next question 1.
4. **A second form of the bus-against-compute rule moves r29's ratios by +0.00 to +0.05 and changes one
   reading.** (*estimated*; Llama-2, K 300, C 40, `pair` / `linear`)

   | what exists | form A (committed) | form B (p-norm), range over p |
   |---|---|---|
   | both kernels as priced | 1.11 / 1.15 | **1.14 to 1.16 / 1.16 to 1.18** |
   | W-position kernel, two GPU launches | 1.21 / 1.25 | 1.22 to 1.25 / 1.28 to 1.30 |
   | one-row kernel twice, batched GPU kernel | 1.24 / 1.25 | 1.23 to 1.26 / 1.25 to 1.27 |
   | neither | 1.43 / 1.45 | 1.39 to 1.43 / 1.44 to 1.46 |
   | stock batch 2 (tile) | 1.60 / 1.62 | 1.56 to 1.58 / 1.58 to 1.61 |

   The four lower rows keep their class. **The priced row no longer sits clear of the gate's line (1.20):** with
   form B and C from 36.9 to 46 it is 1.12 to 1.18. So "rule T as it is" holds only if both kernels deliver and
   the rule is form A; otherwise the one-number gate is in play. I raise P(the first device ratio is 1.19 or
   more) from 0.5 to 0.6.
5. Form B cannot pass both phone points with one p (it misses one by 5 to 6 %); form A passes both. Form A stays
   the centre. Form B is the bracket, not a replacement.

## 1. The stored count

`python3 -I -B bonnie_reply_count.py <repo root>`, under a second. Cell = (model, arm, threads, repack, loaded
file, prompt tokens) over every committed device JSON whose `runs[]` have `kind` = `official32`, `tokens` = 32 and
a `text`. Output today: 12 cells, 560 runs, 0 cells with more than one text (*device*). Its last line is an
assertion, so a later report that breaks r29 §2 makes the script fail.

| cell | runs | files (corrected) | r29 said |
|---|---|---|---|
| OLMoE off, 3 threads | 192 | 9 | 8 |
| OLMoE num, 3 threads | 68 | 7 | 6 |
| Granite num, 3 threads | 180 | 9 | 8 |
| the other nine cells | – | as r29 | – |

## 2. An on-core X1 hot reading (r29 next question 1)

### 2.1 Q4_0: none

Every hot line of the Q4_0 kernel that names cpu 7, in `r69`, `r69b`, `r72` and `r74` under
`campaigns/03-dense15-5pct/reports/device/` (*device*):

| record | cell | cpu 7's thread | what it read |
|---|---|---|---|
| R69, R69b `p2hot core=x1` | hot loop | `cpu_ok=0/1` | 0.0467, 0.0477 ns per weight, not on the X1 |
| R72 `l1`, every kernel, cpu 7 | hot, 4,096 and 11,008 inputs | `cpu_ok=0/1`, the X1 at 844 MHz | 0.255 ns per weight: a small core |
| R72 `l3 q4x4_hot` set 7; 7,4,5; 4,5,6,7 | 1, 3, 4 threads | off in all three (`cpu_ok` 0/1, 2/3, 3/4) | 3.95 G weights/s for that thread against 20.5 for each A78 |
| R72 `l5 q4_0`, set 7, hot | KleidiAI, 1 thread | `cpu0_ok=0`, all calls off-core | equal to the A78 line (0.0805 against 0.0806): it ran on an A78 |
| R74 | – | no CPU hot cell | – |

- R72 `l3`, four threads: 65.54 G weights/s × 0.5625 bytes = **36.9 GB/s** (*device*). That is the four-thread hot
  rate as the phone actually ran it, with three A78 (20.52 each, 34.6 GB/s together) and one thread on a small
  core. A78 clock 2,227 MHz in those lines.
- The three-thread `l5 q4_0` cell on 7,4,5 did have the X1 thread on-core (`cpu0_ok=1`, X1 at 2,841 MHz) and read
  18.20 GB/s. It does not isolate the X1: it is a ggml graph call of 0.77 MB with 23,500 calls per second, so the
  per-call overhead and the wait for the slowest thread dominate (a single A78 reads 6.98 GB/s in the same cell
  type against 11.5 in `l3`). I do not use it.

### 2.2 Other types: two verified cells

R72 `l5`, one thread, set 7, `cpu0_ok=1`, no off-core call, `ramp_ok=1`, X1 at 2,841 MHz (*device*):

| type (plain ggml kernel) | cell | X1, ns per weight | A78 counterpart | ratio |
|---|---|---|---|---|
| IQ4_XS | hot, 4,096 inputs, 112 rows | 0.0830 | 0.1228: the set-4 line of the same cell, flagged off cpu 4 while the A78 cluster was at 2,419 MHz | 1.48 |
| IQ4_XS | cold, 4,096 × 4,096 | 0.0666 | 0.0989, the same kind of line; the verified A78 line of the 11,008 shape reads 0.0992 | 1.49 |
| Q3_K | hot, 11,008 inputs, 48 rows | 0.1198 | 0.1657: verified on cpu 4, but the 4,096 shape | 1.38 |

- The A78 side of the first two rows is an inference: the thread was off cpu 4, and I take it to have run on
  another A78 because its cold value equals a verified A78 value to 0.3 %. The third row compares two shapes.
  **So "the X1 is 1.4 to 1.5 times an A78" is a device-derived estimate with one unverified leg, on kernels that
  are not the subjects'.** 2,841 ÷ 2,419 MHz is 1.17 of it; the rest is the wider core.
- memo-09's prediction for the X1 on the Q4_0 kernel was 0.078 against the A78's 0.118 cycles per weight, a ratio
  of 1.51 per cycle and about 1.8 with the clock. The plain-kernel cells say less than that; they do not test it.
- For C it means: if the fourth thread sits on the X1, C is 46 to 52 GB/s at these clocks (*estimated*); if it
  does not, 36.9 (*device*). Which of the two a decode gets is a placement question, and R72's own record is
  that the probe's cpu-7 thread was off-core every time. **B-T29b (a hot Q4_0 cell with the thread verified on
  cpu 7) stays worth its 5 seconds, and it should print the placement of each of the four threads.**

## 3. The rule bracketed (r29 next question 3)

### 3.1 The two phone points and the two forms

From memo-09 §10.3: stock reads 26.0 GB/s at full clocks and 24.7 GB/s at the L4 caps, against a read rate of
29.9 GB/s (*device*), with compute taken as 50 and 35 GB/s (assumed; they lean on the X1).

| form | time per byte | first point (26.0) | second point (24.7) |
|---|---|---|---|
| A, as committed | max(1/R, x/C) + 0.25 × min(1/R, x/C) | 26.0 | 24.6 |
| B, p-norm, p 2.10 | ((1/R)^p + (x/C)^p)^(1/p) | 26.0 | 23.1 |
| B, p 2.35 | the same | 26.8 | 23.9 |
| B, p 2.65 | the same | 27.4 | 24.7 |

- A is flat in compute while the bus binds and has a kink where they cross. B is smooth: it starts charging
  for compute earlier. A width-2 pass at κ 1.45 and C 40 sits exactly at the crossing (compute time 1.45 ÷ 40
  against a bus share near 1 ÷ 30.5), which is why the form matters here and hardly anywhere else.
- Both forms are anchored on the same one-column device rates: CPU alone 25.64 GB/s, and the CPU beside the GPU
  as form A gives it at width 1.
- Both points use assumed compute rates. §2 says the second (35) is near the measured 34.6 to 36.9; the first
  (50) is the X1-on-core case that no record verifies.

### 3.2 r29's table under each form (*estimated*; Llama-2; Yi about 0.02 lower; `pair` / `linear`)

| row | C | A | B p 2.10 | B p 2.35 | B p 2.65 |
|---|---|---|---|---|---|
| as priced | 40 | 1.11 / 1.15 | 1.16 / 1.18 | 1.15 / 1.17 | 1.14 / 1.16 |
| | 36.9 | 1.13 / 1.16 | 1.17 / 1.18 | 1.16 / 1.18 | 1.15 / 1.17 |
| | 46 | 1.09 / 1.13 | 1.14 / 1.17 | 1.13 / 1.16 | 1.12 / 1.15 |
| W kernel, two GPU launches | 40 | 1.21 / 1.25 | 1.25 / 1.30 | 1.24 / 1.29 | 1.22 / 1.28 |
| | 36.9 | 1.25 / 1.28 | 1.27 / 1.31 | 1.26 / 1.30 | 1.24 / 1.29 |
| | 46 | 1.15 / 1.23 | 1.22 / 1.29 | 1.20 / 1.28 | 1.18 / 1.27 |
| one-row kernel twice, batched GPU kernel | 40 | 1.24 / 1.25 | 1.26 / 1.27 | 1.25 / 1.26 | 1.23 / 1.25 |
| | 36.9 | 1.27 / 1.28 | 1.27 / 1.28 | 1.26 / 1.28 | 1.25 / 1.27 |
| | 46 | 1.19 / 1.20 | 1.23 / 1.25 | 1.21 / 1.24 | 1.20 / 1.23 |
| neither kernel | 40 | 1.43 / 1.45 | 1.43 / 1.46 | 1.41 / 1.45 | 1.39 / 1.44 |
| | 36.9 | 1.47 / 1.49 | 1.46 / 1.48 | 1.44 / 1.47 | 1.43 / 1.46 |
| | 46 | 1.35 / 1.37 | 1.37 / 1.43 | 1.36 / 1.41 | 1.34 / 1.40 |
| stock batch 2 (tile) | 40 | 1.60 / 1.62 | 1.58 / 1.61 | 1.57 / 1.60 | 1.56 / 1.58 |

Without a lane (CPU alone):

| row | C | A | B p 2.10 | B p 2.35 | B p 2.65 |
|---|---|---|---|---|---|
| W kernel | 40 | 1.17 | 1.23 | 1.21 | 1.20 |
| | 36.9 | 1.24 | 1.26 | 1.25 | 1.23 |
| | 46 | 1.10 | 1.18 | 1.17 | 1.15 |
| one-row kernel twice | 40 | 1.52 | 1.51 | 1.49 | 1.47 |
| stock batch 2 (tile) | 40 | 1.89 | 1.84 | 1.82 | 1.79 |

Reading:
- **The form is worth as much as C.** On the priced row the spread over forms is 0.05 and over C (36.9 to 46)
  0.04. Together: **1.09 to 1.18.** r29's "1.09 to 1.15" is the form-A half of that.
- **Far from the crossing the forms agree** (the last two rows move by 0.02 to 0.05 and downwards): where the
  second position is plainly compute-bound, any reasonable rule gives the same time. So the two closing lines of
  r29 are robust: neither kernel → width 2 does not pay (1.34 to 1.49 in every cell); stock batch 2 loses.
- **Near the crossing form B always reads higher.** If the phone behaves like B, a width-2 pass with both kernels
  costs 3 to 5 points more than priced: about −0.3 to −0.5 tok/s on the width-2 rungs of r14 (*estimated*: a pass
  of 120 ms, 5 points of it, at 10.9 tok/s).
- Without a lane the W-position kernel gives 1.10 to 1.26. Against Yi's break-even of 1.34 that leaves 6 to 18 %
  for a guess source to earn; at the top of the range rule T's +2.7 % over the head is most of what is left.

### 3.3 What would tell the forms apart on the phone

One model-free cell, and it is the W-position kernel's own timing cell once the kernel exists: the kernel on a
cold body-sized arena at one and at two positions, four threads, placement printed. The kernel-only ratio (no
attention, no extras; *estimated*, CPU alone):

| C | form A | form B, p 2.10 / 2.35 / 2.65 |
|---|---|---|
| 36.9 | 1.21 | 1.23 / 1.22 / 1.21 |
| 40 | 1.14 | 1.20 / 1.19 / 1.17 |
| 46 | 1.06 | 1.15 / 1.14 / 1.12 |

The cell should run each arm three times and print the spread: the forms differ by 0.03 to 0.09 at C 40 to 46 and by nothing at C 36.9 (there the second position is
compute-bound under either), and
I have no record of how well a cold cell repeats inside one block, so I do not claim the gap is resolvable.
C has to be known from the same kit (B-T29b), or the two unknowns cannot be separated. **Until then every
width-2 rung carries about ±0.03 on its pass ratio from the form alone.**

## 4. Lists (additions only)

| # | test | where | cost | closes when |
|---|---|---|---|---|
| B-T29b, extended | a hot Q4_0 cell with the thread verified on cpu 7, printing each of the four threads' placement and rate | device, rider, about 5 s | 0 | fourth thread off cpu 7 → C is 36.9 and the X1 question is one of placement, not of speed |
| B-T30a | when the W-position kernel has a device timing cell: cold, body-sized arena, one and two positions, four threads | device, with that kit | about 10 s | read against §3.3's table at the C the same kit measured: on form A's value → form A stands; 0.03 or more above it → form B, and the width-2 rungs lose 0.3 to 0.5 tok/s; at C near 37 the cell cannot tell them apart |

No try item.

## 5. Next questions, ranked

| # | question | state |
|---|---|---|
| 1 | **C at the clocks of a claim block.** Every C above is at A78 2,227 MHz; the body (R72 L4) was taken at caps of 1,555 to 1,785 MHz, where three A78 would give 24 to 28 GB/s, at or below the bus rate. Which cap state do R74's 25.64 / 19.77 belong to, and is a width-1 pass already compute-bound there (the model asserts it is not)? | a read of the `mhz` and cap fields of R72 and R74; **the top item I can do now** |
| 2 | R75 into r18's draft, r25's script, r27 §2.1 and r29 §1.2 (B-T29a) | waits for the report |
| 3 | DeepSeek's lookup arms and break-even line | waits for the owner on r28 row 40, then Tom |
| 4 | Where the four decode threads actually sit in campaign 2's device runs (`threads_loaded`, `thread_samples`, `pinned_cpus` in the kit JSONs): is cpu 7 held by a worker in a real decode, unlike in the probes? | a read of committed JSONs; second in line |
| 5 | r15's O and the second position's attention against Tom's host decode and KV screen | waits for Tom |
| 6 | Softmax time on the A78 | only if the gate is built on the top-1 probability |

## 6. Sources (2026-11-24)

Repository: device JSONs `r69-…`, `r69b-…`, `r72-c3probe-r33-…`, `r74-c3probe-r34-…` under
`campaigns/03-dense15-5pct/reports/device/` (lines `p2hot`, `l1`, `l3`, `l5`; fields `set`, `threads`, `cpu_ok`,
`cpu0_ok`, `cpu0_off_calls`, `calls`, `ns_per_w`, `gw_per_s`, `per_thread_gw_per_s`, `mhz`, `mhz_range`, `ramp_ok`);
the campaign-2 and campaign-1 device JSONs of r29 §2; `memo-09-gap.md` §10.3; `memo-10-work/bonnie_tree.py`;
`bonnie_tree_r2.py`, `bonnie_joins_r5.py`, `bonnie_stack_r13.py`, `bonnie_ratio_r29.py`; my r14 and r29. No external
source this round.
