# memo-08 work: Tom, K1 cost per touched weight (K1m and other variants), and Bonnie's T2 and T3

Date: 2026-11-22. Author: Tom (`claude-opus-5-5`). Host: the owner's laptop, Apple M4 Pro (8 performance and 4
efficiency cores).

Labels used in this file:
- *host*: an M4 timing or PPL. It is relative only, and the M4 is a proxy for the phone (§4).
- *estimated*: counts and floors derived from the assembly with the A78's published issue rules. They are not
  measurements.
- No number here is from the phone.

## 0. What was built

- **`tools/c3_k1v_bench.c`** (new; bench only, no runtime knob, nothing in `patches/` changed). It times nine
  column-kernel variants against ggml's NEON Q4_0 4x4 gemv (`ggml_gemv_q4_0_4x4_q8_0`, the phone's non-i8mm path).
  - Shapes: the three subjects share the same matrices. Yi-1.5-6B, Llama-2-7B and DeepSeek-7B all have
    **11008 x 4096** (`ffn_gate`, `ffn_up`), **4096 x 11008** (`ffn_down`) and **4096 x 4096** (`attn_q`,
    `attn_output`; Llama-2 and DeepSeek also `attn_k`, `attn_v`). The shapes were read from the GGUF headers.
  - Build and run:
    ```sh
    clang -O3 -march=armv8.2-a+dotprod+fp16 -o lab/c3/bench/k1v tools/c3_k1v_bench.c -lpthread
    lab/c3/bench/k1v <rows> <inputs> <keep> <threads> [reps]      # K1V_SELFCHECK=1: exhaustive checks first (§1)
    <NDK r28>/aarch64-linux-android28-clang -O3 -mcpu=cortex-a78 -static -o lab/c3/bench/k1v-android tools/c3_k1v_bench.c
    ```
  - Log: `lab/c3/logs/k1v.log`.
- **`tools/c3_a78_pipes.py`** (new). It counts vector-pipe use in a function's aarch64 assembly under the A78 issue
  rules of bonnie-path.md §2:
  - SMLAL(2) on V0 only;
  - SCVTF and FCVTL (Q form) on V0 only, at half rate;
  - immediate shifts and shift-long on V1 only;
  - every other vector op on either pipe.

  It is a count, not a simulation. `llvm-mca` is not installed on this laptop, so the `-mcpu=neoverse-n1/v1` runs of
  Bonnie's T1 were not done.
- **Variants.** Each one reads the same K1 layout as 0002's `p2d` (TR 128: per tile and block, 128 fp16 scales and
  then 32 columns of 64 B).

  | name | what changes against K1 | lead's list item |
  |---|---|---|
  | `k1` | 0002's kernel verbatim (int16 SMLAL accumulation, float step SSHLL, SCVTF, FCVTL, FMUL, FMLA) | baseline |
  | `k1m` | Bonnie's K1m: SCVTF → `ADD 0x4B400000` + `FSUB 12582912.0f`. clang fuses the widen and the add into `SADDW`, so `k1m` and `k1m2` compile to the same code | Bonnie T1 |
  | `k1m2` | written as `SADDW` + `FSUB` (widen and magic add in one op) | – |
  | `k1m3` | `k1m2`, plus the fp16 scale without FCVTL: magnitude bits `(h & 0x7fff) << 13` = \|dw\| × 2⁻¹¹², times dx × 2¹¹². The sign goes into the integer sum | interleaved scale path |
  | **`k1m4`** | `k1m2`, plus the signed scale in two ops: `SSHLL #13` (sign-extended) and `BIC 0x70000000` give `sign \| (h & 0x7fff) << 13`, times dx × 2¹¹². No FCVTL and no SCVTF remain, so the only V0-only op left is SMLAL | interleaved scale path |
  | `k1x2` | two kept columns per iteration, two accumulator sets | register blocking |
  | `k1t` | four kept columns at a time, transposed with 4-register TBL into row quads, then SDOT | TBL/LUT |
  | `k1tp` | `k1t` with software prefetch of block b+2's scales and kept columns | prefetch distance |
  | `k1z` | as `k1t`, with the transposition done by two ZIP levels | transpose to SDOT |

## 1. Identity

**All nine variants are bit-identical** (`memcmp` of every output) to the stock 4x4 gemv fed the same q8 input with the
dropped inputs set to 0.
- **Coverage:** the three subject shapes, keep 0.58, 0.60 and 1.0, 1 and 3 threads (every row of §2), plus
  256 x 4096 at keep 0.05 and 1.0 and 256 x 11008 at 0.58.
- **Adversarial weight scales:** about 50 % negative (Q4_0 stores d = max / −8, so real files have negative scales),
  1 % −0 (the scale of an all-zero block), 0.5 % +0, 1 % fp16 subnormals and 0.5 % large values.
- **Adversarial input scales:** dx = 0 on every 97th block and 60000 on every 89th.
- **Negative controls** (deliberately broken copies, scratch only). Each fails identity where expected:
  - k1m3 without its sign step: `k1m3=0`;
  - magic 0x4B400001: k1m, k1m2, k1m3 = 0;
  - dx × 2¹¹¹: k1m3 = 0;
  - BIC mask 0x30000000: `k1m4=0`.
- **Exhaustive check** (`K1V_SELFCHECK=1`, 0 mismatches):
  - every int in [−2²², 2²²) through the magic convert, against SCVTF;
  - every finite fp16 scale (63,488 patterns) against `dx × (float) h`, for 4,096 dx values including 65504, through
    k1m3's path (±0 products excepted, see below) and k1m4's path (no exception).

  Injected errors are caught: the magic +1 gives 2,097,152 mismatches, and dx × (1 + 2⁻²³) gives 260,038,656.
- **Why it is exact:**
  - K1's block sums are \|i\| ≤ 8 × 127 × 32 = 32,512, which is under 2²², so the magic convert gives exactly i.
  - Multiplying by 2¹¹² or 2⁻¹¹² is exact; dx × 2¹¹² < 2¹²⁸ because dx ≤ 65504.
  - So round(dx₂ × y) = round(dx × dw) bit for bit.
  - k1m4 carries the sign, −0 included, in the float.
  - k1m3 negates the integer sum instead. It differs from stock only if acc is −0 when a −0 scale arrives, and acc
    starts at +0 and cannot become −0 under round-to-nearest.
- **`llama-perplexity` per-chunk at `-ub 1`, `-ub 32`, `-b 512`: not applicable yet.**
  - No runtime code was changed: K1 and K1m exist only in the bench and in 0002's off-by-default probe, and nothing in
    the graph calls them.
  - The per-chunk check is due when a K1/K1m mat-mul path enters a runtime patch. It must then be run as the rule says.

## 2. M4 k (*host*)

k = ns per touched weight ÷ the stock gemv's ns per weight, at the same shape, regime and thread count.
- Best of 5 trials of 30 calls.
- **warm:** the whole matrix, repeated (8 to 24 MB).
- **cold:** round-robin through about 1 GB of copies.
- Masks are random per input at the given keep. The list build (bitmask → kept list) is timed inside each column
  variant.
- Threads have QoS user-interactive. macOS cannot pin threads, so performance-core placement is likely but not
  enforced.
- Each cell reads warm / cold.

| shape | keep | thr | k1 | k1m | k1m4 | k1z | k1x2 | k1t | k1tp | stock GB/s |
|---|---|---|---|---|---|---|---|---|---|---|
| 11008x4096 | 0.58 | 1 | 1.60 / 1.56 | 1.61 / 1.55 | 1.68 / 1.58 | 1.62 / 1.63 | 1.82 / 1.75 | 1.88 / 1.81 | 1.91 / 3.05 | 54.0 / 51.9 |
| 4096x11008 | 0.58 | 1 | 1.62 / 1.61 | 1.62 / 1.59 | 1.70 / 1.67 | 1.65 / 1.64 | 1.83 / 1.79 | 1.89 / 1.82 | 1.91 / 3.01 | 53.3 / 52.3 |
| 4096x4096 | 0.58 | 1 | 1.59 / 1.59 | 1.59 / 1.58 | 1.66 / 1.65 | 1.63 / 1.58 | 1.79 / 1.77 | 1.88 / 1.85 | 1.82 / 3.01 | 52.6 / 51.5 |
| 11008x4096 | 0.60 | 1 | 1.60 / 1.55 | 1.61 / 1.52 | 1.69 / 1.62 | 1.66 / 1.60 | 1.82 / 1.70 | 1.89 / 1.73 | 1.92 / 3.12 | 54.5 / 51.8 |
| 4096x11008 | 0.60 | 1 | 1.58 / 1.58 | 1.58 / 1.57 | 1.69 / 1.65 | 1.68 / 1.63 | 1.82 / 1.78 | 1.88 / 1.79 | 1.91 / 3.07 | 53.6 / 52.3 |
| 4096x4096 | 0.60 | 1 | 1.53 / 1.57 | 1.51 / 1.54 | 1.63 / 1.63 | 1.66 / 1.56 | 1.75 / 1.75 | 1.85 / 1.85 | 1.85 / 3.07 | 51.9 / 51.5 |
| 11008x4096 | 1.00 | 1 | 1.41 / 1.38 | 1.39 / 1.38 | 1.45 / 1.42 | 1.34 / 1.34 | 1.53 / 1.53 | 1.56 / 1.54 | 1.79 / 3.60 | 53.2 / 52.1 |
| 4096x11008 | 1.00 | 1 | 1.40 / 1.40 | 1.39 / 1.40 | 1.45 / 1.45 | 1.39 / 1.36 | 1.55 / 1.55 | 1.54 / 1.57 | 1.83 / 3.64 | 53.1 / 52.2 |
| 4096x4096 | 1.00 | 1 | 1.43 / 1.38 | 1.39 / 1.38 | 1.47 / 1.39 | 1.39 / 1.35 | 1.57 / 1.49 | 1.54 / 1.53 | 1.58 / 3.56 | 53.3 / 51.4 |
| 11008x4096 | 0.58 | 3 | 1.58 / 1.58 | 1.59 / 1.73 | 1.64 / 1.74 | 1.67 / 1.81 | 1.77 / 1.83 | 1.84 / 1.88 | 1.89 / 3.16 | 147.3 / 148.0 |
| 4096x11008 | 0.58 | 3 | 1.64 / 1.79 | 1.64 / 1.77 | 1.71 / 1.77 | 1.71 / 1.73 | 1.86 / 1.85 | 1.90 / 1.88 | 2.00 / 3.08 | 147.9 / 142.4 |
| 4096x4096 | 0.58 | 3 | 1.66 / 1.77 | 1.66 / 1.71 | 1.73 / 1.73 | 1.81 / 1.71 | 1.88 / 1.81 | 2.05 / 1.94 | 2.02 / 2.97 | 150.5 / 140.3 |
| 11008x4096 | 0.60 | 3 | 1.56 / 1.62 | 1.56 / 1.61 | 1.62 / 1.61 | 1.60 / 1.67 | 1.75 / 1.71 | 1.82 / 1.77 | 1.87 / 3.08 | 147.3 / 141.2 |
| 4096x11008 | 0.60 | 3 | 1.62 / 1.73 | 1.62 / 1.71 | 1.69 / 1.72 | 1.68 / 1.77 | 1.80 / 1.82 | 1.87 / 1.86 | 1.97 / 3.18 | 147.9 / 143.2 |
| 4096x4096 | 0.60 | 3 | 1.65 / 1.75 | 1.65 / 1.70 | 1.73 / 1.70 | 1.71 / 1.63 | 1.84 / 1.78 | 1.91 / 1.85 | 1.95 / 3.05 | 150.5 / 139.8 |
| 11008x4096 | 1.00 | 3 | 1.38 / 1.33 | 1.38 / 1.33 | 1.42 / 1.37 | 1.37 / 1.32 | 1.53 / 1.49 | 1.55 / 1.49 | 1.77 / 3.39 | 147.1 / 141.5 |
| 4096x11008 | 1.00 | 3 | 1.43 / 1.40 | 1.44 / 1.40 | 1.48 / 1.43 | 1.43 / 1.40 | 1.60 / 1.55 | 1.62 / 1.57 | 1.84 / 3.54 | 147.6 / 143.0 |
| 4096x4096 | 1.00 | 3 | 1.46 / 1.38 | 1.45 / 1.37 | 1.49 / 1.41 | 1.46 / 1.39 | 1.63 / 1.53 | 1.65 / 1.56 | 1.70 / 3.44 | 150.0 / 140.4 |

Reading (*host*):
- **On the M4, no variant beats K1.** K1m equals K1 within 0.03 (1.51 to 1.62 single-thread, 1.55 to 1.79 on 3
  threads, at keep 0.58 / 0.60). k1m4 is 0.05 to 0.10 worse, and k1m3 is worse again (log). The M4 runs these
  kernels at their total vector-op count, and K1m3 and K1m4 have more ops in total (§3). This matches memo-05: "on
  the M4 the kernel runs at its µop count".
- **The 3-thread cold cells are 0.1 to 0.2 higher than single-thread** for the column kernels, while stock keeps its
  rate (141 to 148 GB/s on 3 threads). The column kernels' scattered 64 B reads with holes cost more when three
  cores share the fabric. This is an M4 observation; the phone's fabric is different.
- **Closed on this evidence (host and count agree):**
  - **Register blocking (`k1x2`):** +0.12 to +0.22. The second chain spills (26 loads/stores per iteration in the A78
    build) and adds nothing when the core already issues at its op count.
  - **TBL (`k1t`):** +0.17 to +0.39. In the A78 count it is also worse than K1m (§3).
  - **Prefetch distance (`k1tp`, PF 2 blocks):** cold 2.97 to 3.64. Explicit prefetch hurts on this core, as memo-05
    found at distance 1. I did not sweep further distances; the direction is already clear.
  - **More rows per tile:** not repeated. memo-05 measured TR 512 cold at 2.20 to 2.54 against TR 128's 1.53 to 1.64.
- **ZIP + SDOT (`k1z`):** equal to K1 at keep 1.0 and +0.03 to +0.12 at 0.58 to 0.60 (the cost of padding the last
  kept quad). On the A78 it moves work from SMLAL (V0 only) to SDOT (both pipes), but its float step brings SCVTF and
  FCVTL back. It only becomes interesting combined with k1m4's float step, which is not built.
- **Skip-list encoding:** the list build from a 32-bit mask per block (ctz loop; kept index list, packed x and
  Σx) costs 1.2 to 5.6 µs per call, **0.25 to 3.3 % of a stock call**. The most is on 4096 x 11008 on 3 threads,
  where the call is shortest.
  - It runs once per matrix and token, and is shared by all rows.
  - A bitmask walked inside the kernel would run that ctz loop once per 128-row tile instead of once per call, so it
    was not built. The index list is the right encoding.
- **Fusing the threshold test into the kernel:** the test is the list build's input: one compare per input, against
  one table entry. Its whole cost is inside the 0.25 to 3.3 % above, so fusing can save at most that. It is not worth a
  kernel change; it belongs in the quantise step of the activation (one pass over x per matrix and token).

## 3. A78 issue counts and floors (*estimated*)

**Build:** NDK r28 clang 19.0.1, `-O3 -mcpu=cortex-a78`, the same source.

**Counts per loop:**

| kernel | column loop (one kept column, 128 rows): V0 / V1 / either | float step (8 rows of one block): V0 / V1 / either | per block: either |
|---|---|---|---|
| stock 4x4 gemv | (per 4 rows x 32 inputs) 6 / 4 / 15 | – | – |
| k1 | 16 / 4 / 4 | 8 / 2 / 5 | 12 |
| k1m (= k1m2) | 16 / 4 / 4 | 4 / 0 / 9 | 13 |
| k1m3 | 16 / 4 / 4 | 0 / 2 / 13 | 13 |
| **k1m4** | 16 / 4 / 4 | **0** / 2 / 11 | 13 |
| k1z | per quad and 32 rows: 0 / 4 / 20 | per 32 rows: 32 / 0 / 32 | – |
| k1t | per quad and 32 rows: 0 / 4 / 28 (TBL4 counted as 4) | per 32 rows: 32 / 0 / 32 | – |

Half-rate ops count 2 V0 cycles. The constants of k1m3 and k1m4 are pinned in registers; without that, clang 19
rematerialised them with two DUP per 8 rows.

**Floor** = max(V0, V1, total ÷ 2) per 128-row strip and 32-input block, with nk = 32 × keep kept columns; cycles per
touched weight:

| kernel | keep 0.58 | keep 0.60 | keep 1.0 | binds at 0.58 | k vs stock, my count (0.098) | k vs Bonnie's stock (0.086) |
|---|---|---|---|---|---|---|
| k1 | 0.179 | 0.177 | 0.156 | V0 | 1.83 | 2.08 |
| k1m | 0.152 | 0.151 | 0.141 | V0 | 1.56 | 1.77 |
| k1m3 | 0.147 | 0.145 | 0.125 | total | 1.51 | 1.71 |
| **k1m4** | **0.140** | **0.139** | **0.125** | total (V0 297 of 343 cycles) | **1.44** | **1.63** |
| k1z | 0.155 | 0.153 | 0.125 | – | 1.59 | 1.80 |
| k1t | 0.189 | 0.187 | 0.156 | – | 1.94 | 2.20 |

- **My K1 and K1m values reproduce Bonnie's** (0.179 and 0.152), from the compiler's real output rather than a
  hand mix.
- **The two stock floors differ** only because my count includes the loop's `MOVI`/`MOV` (2 per 128 weights).
  Bonnie's 0.086 omits them. The k ratios are given against both.
- **k1m4 takes SCVTF and FCVTL off V0.** It is the lowest floor of the kernels built: −0.039 cycles per weight
  against K1 and −0.012 against K1m (*estimated*).
  - Its binding limit is now the total op count, not V0. The column loop's SMLAL floor (0.125) is reached only at
    keep 1.0.
  - Getting closer at keep 0.58 needs about 70 fewer non-SMLAL ops per strip and block. The float step is the place,
    and I see no further exact cut in it.
- **Assumptions to check.** `SADDW`, `BIC` (vector immediate) and `CMLT` are taken as "either pipe", and 4-register
  TBL as 4 µops. I did not find them in Bonnie's table. Bonnie (who has the A78 guide open) should confirm, because
  k1m4's floor depends on `SADDW` and `BIC` not being V0-only.

## 4. M4 against X1 and A78: why the M4 k is only a proxy

- **The M4 runs every one of these kernels at its total vector-op count.**
  - The evidence: K1, K1m and K1m4 rank on the M4 by total ops, and the A78's V0-only classes make no difference.
  - Apple publishes no pipe map; this is my reading of the timings, not a documented fact.
  - So **the M4 cannot show K1m's gain**. On the M4 it is 0; on the A78 it is the V0 saving (*estimated* −15 % for
    K1m, −22 % for K1m4 against K1).
- **The instruction mix that differs:**
  - **SMLAL:** V0 only on the A78, two pipes on the X1 (by analogy with the X2 guide).
  - **SCVTF and FCVTL:** half-rate on the A78's V0.
  - **SSHLL, USHR, SSHR:** V1 only on the A78.
  - **SDOT:** both pipes on the A78, four on the X1.
  - On the X1 (4 x 128-bit, by analogy) the kernels are total-bound, as on the M4. There k1m4's extra SSHLL and BIC
    roughly cancel the SCVTF/FCVTL it removes (*estimated*). Its gain is an A78 gain.
- **The M4 has no fp16-scale or int16-widening penalty** that I could detect.
- **Its single-core read rate is about 52 GB/s, against the phone's 19 to 30.** So the host is compute-bound where the
  phone is byte-bound. K1's byte saving shows only on the phone.

**Proposal for R72 D2s** (the lead and Clyde decide):
- Add **k1m** and **k1m4** beside k1 at the subject shapes and keeps 0.58 / 0.60 / 1.0, on cpu 4 (A78) and cpu 7
  (X1), hot and cold.
- Each with its `memcmp` identity line.
- `lab/c3/bench/k1v-android` (static, `-mcpu=cortex-a78`) runs the same bench under `adb shell` as a cross-check of
  the probe. It is a bench, not a kit. On Android, threads are not pinned in this bench.

## 5. Bonnie's T3: per-group levels (40 chunks; `-b 512`, best cells also at `-ub 1`)

**Setup.**
- One level for the q/o inputs (the `qkv` and `o` groups), one for gate/up, one for `down`.
- k and v are dense on Llama-2 (`C3_DENSE_W=attn_k,attn_v`). Yi has no dense weights, so its q/o level also
  sparsifies its small k and v.
- Tables: thr6. Its 0.40 thresholds equal thr2's, and the base cells reproduce the earlier 40 % cells exactly: 7.7010
  and 7.8158.
- **Design:** one factor at a time around 0.40 (levels 0.30 / 0.50 / 0.60), then combinations guided by those cells,
  then `-ub 1` on the best cells. The full 4 x 4 x 4 grid (64 cells per model and setting) was not run.
- **Skipped share** (*estimated*) = Σ level × the group's share of the block's mat-mul weights:
  - Llama-2: q+o 0.166, gate/up 0.446, down 0.223 (k, v 0.166 dense);
  - Yi: q+k+v+o 0.218, gate/up 0.521, down 0.261.

**Llama-2-7B, k and v dense** (stock 7.4071 at `-b 512`, 7.4075 at `-ub 1`; line +4.5 %):

| q/o | gate/up | down | `-b 512` | Δ | `-ub 1` | Δ | skipped share |
|---|---|---|---|---|---|---|---|
| 0.40 | 0.40 | 0.40 | 7.7010 | +3.97 % | 7.7000 | +3.95 % | 0.334 (the frontier) |
| 0.30 | 0.40 | 0.40 | 7.7328 | +4.40 % | | | 0.317 |
| 0.50 | 0.40 | 0.40 | 7.7146 | +4.15 % | 7.7458 | **+4.57 %** | 0.350 |
| 0.60 | 0.40 | 0.40 | 7.8018 | +5.33 % | | | 0.367 |
| 0.40 | 0.30 | 0.40 | 7.5200 | +1.52 % | | | 0.289 |
| 0.40 | 0.50 | 0.40 | 8.0153 | +8.21 % | | | 0.378 |
| 0.40 | 0.60 | 0.40 | 9.2413 | +24.76 % | | | 0.423 |
| 0.40 | 0.40 | 0.30 | 7.6660 | +3.50 % | | | 0.311 |
| 0.40 | 0.40 | 0.50 | 7.7982 | +5.28 % | | | 0.356 |
| 0.40 | 0.40 | 0.60 | 7.8883 | +6.50 % | | | 0.378 |
| 0.50 | 0.30 | 0.60 | 7.6725 | +3.58 % | 7.7174 | +4.18 % | **0.350** |
| 0.60 | 0.30 | 0.50 | 7.7127 | +4.13 % | | | 0.345 |
| 0.50 | 0.35 | 0.50 | 7.7094 | +4.08 % | | | 0.350 |

The 40 % `-ub 1` value is from memo-07-work/tom-split.md §5.

**Yi-1.5-6B** (stock 7.6966 at `-b 512`, 7.6817 at `-ub 1`):

| q/o (with k, v) | gate/up | down | `-b 512` | Δ | `-ub 1` | Δ | skipped share |
|---|---|---|---|---|---|---|---|
| 0.40 | 0.40 | 0.40 | 7.8158 | +1.55 % | 7.8962 | +2.79 % | 0.400 |
| (uniform 0.42, tom-split.md §2) | | | 7.8992 | +2.63 % | 7.9851 | +3.95 % | 0.420 (the frontier) |
| 0.30 | 0.40 | 0.40 | 7.8602 | +2.13 % | | | 0.378 |
| 0.50 | 0.40 | 0.40 | 7.9792 | +3.67 % | | | 0.422 |
| 0.60 | 0.40 | 0.40 | 8.5815 | +11.50 % | | | 0.444 |
| 0.40 | 0.30 | 0.40 | 7.6982 | +0.02 % | | | 0.348 |
| 0.40 | 0.50 | 0.40 | 8.3791 | +8.87 % | | | 0.452 |
| 0.40 | 0.60 | 0.40 | 10.5023 | +36.45 % | | | 0.504 |
| 0.40 | 0.40 | 0.30 | 7.8657 | +2.20 % | | | 0.374 |
| 0.40 | 0.40 | 0.50 | 7.8721 | +2.28 % | 7.9544 | +3.55 % | 0.426 |
| 0.40 | 0.40 | 0.60 | 7.9685 | +3.53 % | 7.9476 | +3.46 % | **0.452** |
| 0.40 | 0.35 | 0.60 | 7.8323 | +1.76 % | | | 0.426 |
| 0.30 | 0.40 | 0.60 | 7.9125 | +2.81 % | | | 0.430 |

All values are *host*, 40 chunks.

**Reading:**
- **The same ranking holds on both models: gate/up is the intolerant input, `down` and q/o are tolerant.** At 0.60:
  - gate/up costs +24.8 % (Llama-2) and +36.5 % (Yi);
  - `down` costs +6.5 % and +3.5 %;
  - q/o costs +5.3 % and +11.5 %. Yi's q/o level also sparsifies its k and v, which may explain its higher cost.
  - At 0.30, gate/up gives back the most: Llama-2 +1.52 % and Yi +0.02 %.
- **Single-group changes do not add exactly.** On Llama-2, q/o 0.30 is worse than 0.40 (+4.40 % against +3.97 %).
  That is within the noise of a 40-chunk cell (SE about 0.5 point).
- **Best cell inside the line at both settings:**
  - **Yi: q/o 0.40, gate/up 0.40, `down` 0.60.** +3.53 % / +3.46 %, skipped share 0.452 against the uniform
    frontier's 0.420: **+0.032**.
  - **Llama-2: q/o 0.50, gate/up 0.30, `down` 0.60.** +3.58 % / +4.18 %, share 0.350 against 0.334: **+0.017**.
    The cheaper-looking q/o 0.50 alone fails at `-ub 1` (+4.57 %).
- **Against Bonnie's pass rule as written:**
  - The rule asks for a skipped share at the best cell ≥ frontier + 0.03 on both models.
  - Yi passes, by 0.002. Llama-2 fails: +0.017.
  - **So T3 closes as written: per-group levels do not pass on both models.**
  - On Yi alone the gain is small: 3.2 points of skipped weight, about 7.6 % fewer bytes in the sparsified matrices
    (*estimated*). It sits next to a cliff in gate/up.
- **Not run:** the full grid; finer gate/up levels (0.25, 0.32) combined with `down` 0.6 / 0.7 on Llama-2; DeepSeek.
  If the lead wants Llama-2 pushed further, gate/up 0.25 with `down` 0.70 and q/o 0.50 is the next cell. It needs a
  0.7 level, which thr6 has.

## 6. Bonnie's T2: full-set checks (`-b 512`)

Setup:
- The whole of wikitext-2 test at ctx 512, `-b 512`: 655 chunks on Llama-2 and 596 on DeepSeek (its tokenizer is
  coarser).
- Stock is `llama-perplexity` (lab build, `-t 8`). The cell is `tools/c3_colskip.py` with `C3_DENSE_W=attn_k,attn_v`
  at 40 %, with the same tables as the 40-chunk cells:
  - Llama-2: thr2;
  - DeepSeek: thr7, calibrated with BOS on the BOS copy (memo-07-work/tom-split.md §6).
- The paired SE comes from per-chunk values. For stock they are recovered from `llama-perplexity`'s running values
  (`[k]` to 4 decimals); the recovered values reproduce the final estimate exactly.
- `-ub 1` was not run on the full set; the lead asked for `-b 512`. At about 8 to 15 min per 40 chunks it would take
  2.5 to 4 h per model.

| model | stock (full) | k, v dense 40 % (full) | Δ (paired SE) | budget ×1.05 | 40-chunk Δ for comparison | verdict |
|---|---|---|---|---|---|---|
| Llama-2-7B | 7.7435 | 8.0065 | **+3.40 %** (0.15) | 8.1307 | +3.97 % | inside, 1.1 points under the +4.5 % line |
| DeepSeek-7B, BOS on | 8.8342 | 9.1480 | **+3.55 %** (0.13) | 9.2759 | +3.81 % | inside, 0.95 point under the line |

All values are *host*.

- **Llama-2's 40 % holds on the full set**, with more margin than on 40 chunks: 1.1 points against 0.5. Bonnie gave
  this P 0.8 (bonnie-path.md §4).
- **DeepSeek with BOS holds too.** Its full-set stock is 8.8342 with BOS. A served file with BOS has this as its stock
  reference; the 40-chunk value was 8.3095.
- **Realised zeroed fractions on the full set:** `down` 0.410 / 0.411, gate/up 0.392 / 0.394, o 0.400 / 0.396, q
  0.391 / 0.393 (DeepSeek / Llama-2). Each is within 0.01 of the 40-chunk cells.
- Logs (untracked) in `lab/c3/logs/`: `ppl-llama2-7b-q4_0-b512-full.log`, `ppl-deepseek7b-bos-q4_0-b512-full.log`,
  `full-llama2-7b-b512.log`, `full-deepseek7b-bos-b512.log`.
