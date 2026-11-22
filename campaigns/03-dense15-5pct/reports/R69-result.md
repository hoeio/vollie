# R69 result: dense read rate, compute ceiling, resident capacity and GPU headroom (c3 probe kit r29)

### R69 Probe: read rate, compute ceiling, residency, GPU headroom (kit r29, no model; amendment 1)

- **Date:** 2026-11-22, 16:46:52 to 16:56:06 local. **Label:** *device* for the measured values. Ratios and
  prices derived from them are marked.
- **Sources:**
  - [device/r69-c3probe-r29-2026-11-22T1646.json](device/r69-c3probe-r29-2026-11-22T1646.json), sha256
    `db4985af4ca73354bf045bea7d17950ff7e84f5210fe83c816b2e2a24894cad2`, committed `2a38091`.
  - Parser `tools/c3_probe_parse.py`; scorer [memo-08-work/clyde_r69_score.py](memo-08-work/clyde_r69_score.py) and
    sheet [memo-08-work/clyde-r69-scoring.md](memo-08-work/clyde-r69-scoring.md).
  - Pre-registration [R69](../prereg/R69-probe-read-rate-capacity-gpu-device.md) (`458102c`) and
    [amendment 1](../prereg/R69-amendment-1.md).
- **Subjects** (memo-06 §6, `e582606`): Yi-1.5-6B-Chat, Llama-2-7b-chat, DeepSeek-LLM-7B-chat. The goal is 12 tok/s on
  each.

## 1. Void checks (§4): none met

| void | state | source |
|---|---|---|
| APK hash differs | **not triggered** | The owner installed `lab/dist-c3/vollie-c3-probe-r29.apk` from the laptop. Its sha256 `02736b568503624fa036f86c994c5407516605522f1db901ed2f0d58b866eaf9` equals the pre-registration's (owner's statement, 2026-11-22; checked by the lead with `shasum`; re-checked by clyde 2026-11-22). The JSON reads `app=1.29.0+29`, `kit=r29` |
| power saving on | not triggered | `powerSave: false` at the start, at the end, and before each of the 4 blocks |
| charger plugged | not triggered | `plugged_start` false; `plugged_before` false in all 4 blocks |
| process killed | not triggered | every block has its end line: p2 71.4 s, x1 9.8 s, x2 14.1 s, p1 158.0 s; `stopped_early` false |

Recorded, not voids:
- the deep-sleeping list is on;
- `model_unloaded=olmoe-Mi.gguf`, so no model was loaded during the probe;
- free storage at the start was 42.6 GB.

**Conditions beside every number.** These are not void rules in R69 §4, so no void is added for them.
- **Gates:** the r16 cool-down gate timed out before p2 (headroom 0.731 after 90 s) and before x1 (0.675). It passed
  before x2 and p1.
- **Clocks:** 1 of 26 cells read 2,419 MHz on cpu 4 and 2,841 MHz on cpu 7 at both readings. The A78 cluster ran
  mostly at 1,555 to 1,881 MHz (710 in places), and the X1 at 844 to 1,555 MHz.
- **Workers:** in 6 cells a worker did not stay on its cpu (cpu_ok < n; the parser marks them `*`). They are used as
  the rules are written, with cpu_ok stated.

*Correction 2026-11-22 (lead):* the scoring sheet's reading, written before the data, excluded `*` cells from
every rule input. That would have added a void that R69 §4 does not contain. It is withdrawn, and the scorer was
changed to use the cells as written.

## 2. Rule inputs, with the cell each comes from

| quantity | value (*device*, or arithmetic on *device*) | cell, cpu_ok, clocks cpu0/cpu4/cpu7 before → after |
|---|---|---|
| r (q4x4, X1 + 2 A78) | **28.56 GB/s** | p2 q4x4 `7,4-5`, 2/3, 1094/710/1555 → 1094/1555/1555 |
| read, 3 A78 | 28.79 GB/s | p2 read `4-6`, 3/3, 1804/2419/844 → 1094/2419/844 |
| q4x4, 3 A78 | 30.09 GB/s | p2 q4x4 `4-6`, 2/3, 1094/1881/844 → 1094/1555/1555 |
| q4x4, X1 + 3 A78 | 28.56 GB/s | p2 q4x4 `4-7`, 4/4, 1094/1555/1555 both readings |
| q4x4, 3 A78 + 4 A55 | 24.75 GB/s | p2 q4x4 `0-6`, 7/7, 1094/1555/1555 → 1804/1555/844 |
| P2hot A78 | 0.0700 ns/w = 0.109 cycles/w | cpu 4, 1/1, cpu4 1555 → 1555 |
| P2hot X1 | 0.0467 ns/w = 0.056 cycles/w | cpu 7, **0/1**, cpu7 844 → 1555 |
| P2hot A55 | 0.2589 ns/w = 0.375 cycles/w | cpu 0, 1/1, cpu0 1094 → 1804 |
| c_b = 0.87 × 0.5625 × (1/ns_X1 + 2/ns_A78) | **24.46 GB/s** | from the two P2hot cells above |
| R = c_b ÷ r | 0.856 | |
| k, column kernel (H6), pinned A78, keep 0.60, cold | **3.06** (0.2200 ÷ 0.0718 ns per touched weight) | p2col `4`, 1/1 and 1/1, cpu4 1555. Identity col_exact = 1 |
| k, block-major skip list (H6b) | 1.80 | p2col `4`, 1/1. Identity skip_exact = 1 |
| X1 gain, 2 / 4 streams | 0.918 / 0.863 (28.84, 27.12 ÷ 31.42) | x1 `4-6`. Streams 1 at 2/3, 2419/2841 → 2419/2841; streams 2 at 3/3, ending 2419/844; streams 4 at 3/3, 2419/844 |
| X2 GPU read / q4 | 30.86 / 1.88 GB/s | x2 (Adreno; kernels `read`, `q4`), no retry made (not in the build) |
| S = CPU + GPU sum ÷ CPU-alone q4x4 (3 A78) | **1.575** (47.40 ÷ 30.09) | x1gpu gpu_kern read, 2/3 (CPU 21.20 + GPU 26.19); the divisor is p2 q4x4 `4-6` |
| bus ratio (3 A78 + 4 A55) ÷ 3 A78, q4x4 | 0.823 | |

**c_b is below r.** The cold q4x4 gemv on X1 + 2 A78 delivered 28.56 GB/s. The compute ceiling as R69 defines it
(single-thread hot cells × 0.87) came out lower, at 24.46.
- The hot cells ran with the A78 at 1,555 MHz, and with the X1 thread off its cpu at 844 to 1,555 MHz.
- So c_b as written is depressed by the clock state. The cold cell's own throughput is a lower bound on the
  three-core compute rate under these clocks (*device*).
- The rules are scored with c_b as written. Both values are carried in §5.

## 3. Decision rules (§4 with amendment 1)

| rule | outcome | subjects |
|---|---|---|
| **A, column kernel (Y3/Y4, the H6 kernel)** at reference d 0.60 | **KILL**: k 3.06 > k\*(r) = max(R, 1) ÷ 0.60 = 1.67; token gain 0.544 | reference density |
| A, Yi (amendment item 2; provisional until its freeze) | **KILL** at d 0.58 (k\* 1.72, gain 0.563) and at d 0.60 (k\* 1.67, gain 0.544) | Yi |
| A, Llama-2, DeepSeek (k,v-dense rule, body d 0.666, nearest row 0.65) | **not scored from R69**: kit r29 has no 0.65 row (build gap, scoring sheet §3). Descriptive only (*estimated*): priced on the 0.60 row the gain is 0.588 | Llama-2, DeepSeek |
| A, reference d 0.65 | not scored from R69: no row | – |
| **B, R20 draft speculation** | **KILL**: r ÷ c_b = 1.168 ≥ 0.71. R20 stays closed | all |
| **C, residency** | **in between**: 3.4 held (r arm), 4.2 held (x arm), 4.6 not held in either arm. **C = 4.2 GB, file-backed** | see below |
| **D, CPU + GPU split** | **rank 1 stays open**: S 1.575 ≥ 1.30 and ≥ 1.15 × best X1 gain (1.056); X2 read 30.86 ≥ 10. Bus-limit clause not met (ratio 0.823 ≤ 1.10, but S ≥ 1.15). Rank 2 (multi-stream) is not supported: X1 gains 0.92 and 0.86 | all |

**Rule A scope.** The killed lever is the exact column kernel of memo-02 H6 that R69 measured. K1, the strip-register
kernel at TR 128 of memo-05, is a different kernel. It is not in kit r29, so R69 does not score it. It stays open, gated
on its own device row (0002's D2 rows, R72 L1). The generality clause cannot be applied: one subject is killed, and
two are not scored.

**Rule C details** (held = passes 6 to 20 median ≥ 0.80 × the arm's 2.7 GB median, and median major faults per
pass ≤ 50; the better arm counts):

| step | `map=r` ratio, majflt | `map=x` ratio, majflt | held |
|---|---|---|---|
| 2.7 | 1.000, 0 | 1.000, 0 | yes |
| 3.0 | 0.995, 0 | 1.030, 0 | yes |
| 3.4 | 0.860, 0 | 0.789, 0 | yes (r) |
| 4.2 | 0.093, 263 | 0.940, 3 | yes (x) |
| 4.6 | 0.042, 388 | 0.070, 446 | no |

- **As written,** the in-between branch states "the lead candidates do not fit at Q4_0". Amendment item 4 makes this
  "a Q4_0 body of 6 B or more", and the branch also requires the served files to be priced against C.
  - That sentence was written expecting C under 4.2. At C 4.2 it holds only for bodies above C − 0.15 = 4.05 GB, not
    for the three subjects.
  - Pricing against C, as the same branch requires: the Q4_0 files are Yi 3.48, Llama-2 3.83 and DeepSeek 4.00 GB
    (*host*). All three are at or under 4.05 GB; DeepSeek's margin is 0.05 GB.
  - With Bonnie's resident estimate (file + head + KV at n_ctx 512, *estimated*): Yi 3.36, Llama-2 4.02,
    DeepSeek 4.01 GB, against 4.2.
- **Known limit of P1** (Bonnie, memo-08-work/bonnie-path.md §C; not a change to R69). P1 measures file-backed mapped
  memory. With repack on, the body, and K1's layout copy too (tom-split.md §4), is anonymous memory, which P1 did
  not measure.
  - C 4.2 is therefore a residency result only for a body mapped from the file: Y1, or the `--no-repack` path of D1.
  - An anonymous-arena arm is a candidate rider for R72.
- **Rule C as amended:** Y1 (kernel layout in a mapped file) is not made the first build, because 4.6 was not held.
  The subjects stay as chosen, since their Q4_0 files fit under C file-backed. Reserves are those of memo-06 §6.

**§6 (what a result means).** §6's "not reachable" sentence applies when Rule A and Rule D are both killed. Rule D
is not killed, so the sentence is not stated. The descriptive prices are in §5.

## 4. Predictions (§3)

| prediction | band | result |
|---|---|---|
| r, q4x4 3 threads | 24 (19 to 30) | 28.56, inside |
| plain read, 3 A78 | 27 (20 to 36) | 28.79, inside |
| q4x4 ÷ read, 3 A78 | 0.90 (0.75 to 1.00) | 1.045, **MISS** high |
| (X1 + 3 A78) ÷ 3 A78, q4x4 | 1.13 (1.00 to 1.30) | 0.949, **MISS** low |
| P2hot cycles/w, A78 | 0.105 (0.095 to 0.118) | 0.109, inside |
| P2hot cycles/w, X1 | 0.078 (0.070 to 0.090) | 0.056, **MISS** low (cpu_ok 0/1) |
| P2hot cycles/w, A55 | 0.33 (0.22 to 0.48) | 0.375, inside |
| c_b | 40.4 (35 to 45) | 24.46, **MISS** low |
| P2col k, column kernel | 2.6 (2.0 to 3.3) | 3.06, inside (0.60 row only) |
| P2col k, block-major skip list | 0.95 (0.85 to 1.10) | 1.80, **MISS** high |
| X1 gain, 2 streams | 1.05 (0.95 to 1.25) | 0.918, **MISS** low |
| X1 gain, 4 streams | 1.05 (0.88 to 1.35) | 0.863, **MISS** low |
| X2 prints all lines | P 0.80 | yes |
| X2 GPU read | 30 (15 to 42) | 30.86, inside |
| X2 GPU q4 kernel | 10 (3 to 22) | 1.88, **MISS** low |
| S | 1.25 (0.95 to 1.60) | 1.575, inside |
| C | 3.1 (2.6 to 3.8) | 4.2, **MISS** high |
| P(3.4 held) 0.35, P(4.2 held) 0.12, P(4.6 held) 0.06 | | 3.4 held, 4.2 held, 4.6 not held |
| clocks at 2,419 / 2,841 in ≥ 80 % of cells | P 0.8 | 1 of 26, **MISS** |

7 of 16 banded quantities are inside their band and 9 miss. The clock prediction misses.

## 5. Amendment item 6 lines, and the gap ledger at the measured values (*estimated* from *device* inputs)

**Byte and weight lines** (B12 = r × (83.3 − O), W12 = c_b ÷ 0.5625 × (83.3 − O)):

| O, ms | B12, GB | W12, G weights |
|---|---|---|
| 6 | 2.21 | 3.36 |
| 8 | 2.15 | 3.28 |
| 12 | 2.04 | 3.10 |

**Bonnie's ledger** (memo-08-work/bonnie_path.py, `cost()` and `resident()`, unchanged; r 28.56, O 8, printed = 0.95 ×
1000 ÷ steady; three cores):
- The K1 estimate is Bonnie's k 1.645, which is not measured.
- "H6" uses the device k of 3.06.
- c is shown three ways: as written (c_b 24.46); the cold-gemv floor (28.56 GB/s of Q4_0); and the pre-registered
  40.4, which is not a measurement.

| subject | resident vs C 4.2 | stock | K1 at frontier (est. k) | H6 at frontier (device k) |
|---|---|---|---|---|
| Yi-1.5-6B | 3.36 GB (margin 0.84) | 6.5 / 7.5 / 7.6 | 6.8 / 7.8 / 10.8 | 3.9 / 4.5 / 6.3 |
| Llama-2-7B | 4.02 GB (margin 0.18) | 5.8 / 6.7 / 6.7 | 5.9 / 6.8 / 9.1 | 3.6 / 4.1 / 5.8 |
| DeepSeek-7B | 4.01 GB (margin 0.19) | 5.8 / 6.7 / 6.7 | 5.9 / 6.8 / 8.9 | 3.6 / 4.2 / 5.9 |

(printed tok/s at c = 24.46 / 28.56 / 40.4 GB/s)

- **Bytes line at r 28.56:** 71.2 ms holds 2.03 GB. At its frontier Yi reads 2.18 GB (needs r 30.6), Llama-2 2.76
  (r 38.8) and DeepSeek 2.83 (r 39.8). Yi is 7 % over the line; the MHA subjects are 36 to 39 % over.
- **Compute is the binding line at the measured c.** At c_b 24.46 (43.5 G weights/s), 71.2 ms holds 3.10 G weight-steps
  against 5.7 to 6.7 G. Even at the pre-registered 40.4, Yi's K1 row is compute-bound at 10.8.
- **No subject reaches 12 with the CPU levers priced so far,** at any of the three c values (*estimated*).
- **Open:**
  - the 4th core (its read rate did not rise: 0.949);
  - K1 and K1m on the phone (R72);
  - the GPU lane: S 1.575 on reads, but the GPU's Q4_0 kernel ran at 1.88 GB/s, so the lane is open on bytes, not yet
    on compute.

## 6. Consequences (lead)

- **Closed:**
  - the H6 exact column kernel (Rule A), on Yi and at the reference density;
  - R20 draft speculation (Rule B);
  - multi-stream GEMV as a lever (X1 gains below 1).
- **Open:**
  - the CPU + GPU split (Rule D rank 1), on reads;
  - K1 / K1m (not measured by R69);
  - the 4th core on compute (R72 L3; R69 measured only its read rate).
- **Residency:** C 4.2 GB file-backed. All three subjects' Q4_0 files fit under it. Anonymous capacity is not
  measured (P1 limit).
- **Clock state:** the probe ran at reduced clocks after gate timeouts. c_b and the P2hot X1 cell are the numbers
  most affected. R72's L3 (all threads at once, clocks logged) is the measurement that replaces c_b. It needs a cool
  start, or a longer gate.

## Addendum 2026-11-22: owner's repeat run (descriptive, does not replace the scored run)

The owner ran kit r29 a second time at 17:05 after letting the phone cool
(`reports/device/r69b-c3probe-r29-2026-11-22T1705.json`, same APK, power saving off, charger out, deep-sleeping
list on; p2 gate passed this time, x1 gate timed out). R69 stays scored on the first run as pre-registered; a
repeat is not a best-of choice. Verdicts from the same scorer are unchanged: A KILL (k 3.09), B KILL
(r/c_b 1.15), C in between (4.2 GB held, 4.6 not), D rank 1 open (S 1.51).

| quantity (*device*) | run 1 (16:46, scored) | run 2 (17:05, cooled) |
|---|---|---|
| r, q4x4 X1 + 2 A78 | 28.56 GB/s | 31.30 GB/s |
| plain read 3 A78 | 28.79 | 31.43 |
| c_b | 24.46 | 27.22 |
| q4x4 X1 alone | 11.67 | 15.79 |
| skip-list k | 1.80 | 1.66 |
| S (CPU + GPU) | 1.575 | 1.507 |
| C | 4.2 GB | 4.2 GB |

Reading: cooling raised r and c_b by about 10 %, but c_b stays below r in both runs and the X1 hot cell was off
its core in both, so the compute ceiling is still not measured at full clocks. R72 must measure it.
