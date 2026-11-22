# memo-11 U2 (Tom): LUT/TBL gemv for uniform 2-bit and 3-bit weights against the Q4_0 4x4 gemv

Date: 2026-11-22. Author: Tom (`claude-opus-5-5`). Not committed. Host: the owner's laptop, Apple M4 Pro (8P + 4E),
**contended**: tools/c3_freeze.sh's `llama-perplexity` ran next to every timing (about 750 % CPU when sampled).

Labels: *host (contended)* = M4 timings, relative only; *estimated* = A78 issue counts from compiled assembly. Nothing
here is from the phone.

## 0. What was built

- [tom_lut.c](tom_lut.c): scratch bench, no runtime code, nothing in `patches/`.
  - **stock**: `ggml_gemv_q4_0_4x4_q8_0` (NEON + dotprod, base `620d2225c`, copied from `tools/c3_k1v_bench.c`,
    the path R72 measured), with ggml's `quantize_row_q8_0` for the input.
  - **lut2 / lut3** (T-MAC style): 16-row tiles; per G-input block 16 fp16 row scales and B bit-planes of 4-bit
    indices (two groups per byte). Symmetric uniform grid `w = s (q - (2^B-1)/2)`, no zero point. Per group of 4
    inputs a 16-entry table of signed sums, quantised to int8 with one scale per G-block. Inner loop: `AND`/`USHR`
    unpack, `TBL`, widening add to int16 per plane, Horner over planes, then float.
  - Format knobs: **G** = 32 (2.5 / 3.5 bpw) or 128 (2.125 / 3.125 bpw); **ACC** = 1 (T-MAC's int8 tables, +-127),
    2 or 4 (tables at +-63 or +-31, so 2 or 4 lookups are added in int8 before one widen; lossy, my addition).
- Raw output: [u2-runs2.txt](u2-runs2.txt) (plane loop rolled) and [u2-runs3.txt](u2-runs3.txt) (as committed here), 72 lines each.
- Build lines are in the file header. Host: `clang -O3 -march=armv8.2-a+dotprod+fp16 -DG=.. -DACC=..`.
  A78: NDK r28 `aarch64-linux-android28-clang -O3 -mcpu=cortex-a78 -DASM_ONLY -S`.
- **Correctness:** each LUT kernel matches a double evaluation of its own int8 tables and packed file
  (max relative difference 2.6e-4 over sampled rows, float rounding on cancelling sums). Stock's NMSE confirms the
  repack.
- **Regimes:** hot means 448 x 4096 (Q4_0 0.98 MB) at 1 thread, and 1792 x 4096 (1 MB per thread) at 4 threads,
  the same matrix repeated. Cold means 11008 x 4096 through a ring of 22 copies (stock 559 MB, lut2 and lut3
  proportionally smaller).
- **k** = LUT ns per weight / stock ns per weight. Each run takes 9 trials with the kernels interleaved, and k is the
  median of the 9 per-trial ratios. Three runs per cell; the median of the three is reported.
- Table and q8 build are excluded from k. The build is scalar: 6.4 to 8 us per 4096 inputs against q8's 2 us
  (*host*). Spread over a real layer (about 23k inputs and about 200M weights), it is about 1.5 % of the layer's
  gemv time (*estimated*).

## 1. Host k (*host, contended*; 1-thread rows are stable to +-0.01, 4-thread rows scatter up to +-0.3)

Run 3 is the file as committed here, with the plane loop unrolled. Run 2 had the plane loop rolled. Where they differ
the better one is the kernel's k; both are shown.

| G | ACC | NMSE lut2 / lut3 (gauss; 0.5 % outliers) | k2 hot t1 | k2 cold t1 | k2 hot t4 | k2 cold t4 | k3 hot t1 | k3 cold t1 | k3 hot t4 | k3 cold t4 |
|---|---|---|---|---|---|---|---|---|---|---|
| 32 | 1 | 3.4e-5 / 3.4e-5; 3.7e-5 | **0.96** (r2 1.02) | **0.96** | 0.89 to 0.94 | 0.92 to 0.95 | **1.34** | **1.34** | 1.12 to 1.25 | 1.19 to 1.29 |
| 32 | 2 | 1.4e-4; 1.6e-4 | 0.87 | 0.87 | 0.73 to 0.84 | 0.67 to 0.86 | 1.21 | 1.20 | 0.98 to 1.07 | 1.03 to 1.10 |
| 32 | 4 | 5.4e-4; 6.3e-4 | 0.82 | 0.82 | 0.72 to 0.76 | 0.66 to 0.68 | 1.14 | 1.14 | 1.06 to 1.08 | 0.82 to 0.94 |
| 128 | 1 | 5.2e-5 / 5.3e-5; 1.0e-4 | **0.85** (r3 1.02) | **0.84** | 0.87 | 0.81 | **1.21** (r3 1.76) | **1.21** | 1.22 | 1.11 |
| 128 | 2 | 2.1e-4; 4.4e-4 | 0.71 | 0.70 to 0.71 | 0.74 | 0.62 | 1.02 to 1.06 | 1.02 to 1.06 | 0.99 to 1.09 | 0.86 to 0.99 |
| 128 | 4 | 8.6e-4; 1.8e-3 | **0.64 to 0.66** | **0.65 to 0.66** | 0.62 to 0.69 | 0.50 to 0.51 | **0.95 to 0.96** | **0.95** | 0.98 to 1.01 | 0.83 to 0.97 |

- Stock reference (*host, contended*): 0.0125 ns/w at 1 thread, both hot and cold. One M4 core is not bus-bound at
  about 47 GB/s, so hot and cold read alike at 1 thread. At 4 threads cold, stock runs 0.0032 ns/w (about 175 GB/s).
  The 4-thread hot stock (0.0054 to 0.0071) is slower than 1 thread / 4; I read that as contention with the
  perplexity job, so the 4-thread columns are weak.
- NMSE is against a float64 dot of the same dequantised weights with the float input. For comparison, stock with q8
  inputs gives 2.75e-5 (gauss) and 1.0e-4 (with outliers). The T-MAC-faithful tables (ACC 1) sit at that level, as the
  paper says. ACC 2 and ACC 4 are 5 to 30 x worse and would be a further declared numerics change.

## 2. A78 issue count (*estimated*, `tools/c3_a78_pipes.py` rules on the NDK r28 `-mcpu=cortex-a78` assembly; innermost loop = one block)

| kernel | weights per loop | V0 / V1 / any | TBL | loads | floor cycles | cycles/w | k by count |
|---|---|---|---|---|---|---|---|
| stock 4x4 | 128 | 6 / 4 / 15 | – | 6 | 12.5 | 0.0977 | 1.00 |
| lut2 G32 ACC1 | 512 | 16 / 12 / 64 | 16 | 10 | 46.0 | 0.0898 | **0.92** |
| lut3 G32 ACC1 | 512 | 16 / 18 / 94 | 24 | 21 | 64.0 | 0.1250 | **1.28** |
| lut2 G32 ACC2 / ACC4 | 512 | 16 / 12 / 56, 52 | 16 | 10 | 42.0, 40.0 | 0.082, 0.078 | 0.84, 0.80 |
| lut3 G32 ACC2 / ACC4 | 512 | 16 / 18 / 86, 80 | 24 | 18, 22 | 60.0, 57.0 | 0.117, 0.111 | 1.20, 1.14 |
| lut2 G128 ACC1 / 2 / 4 | 2048 | 16 / 36 / 233, 207, 184 | 64 | 153, 125, 101 | 142.5, 129.5, 118.0 | 0.070, 0.063, 0.058 | **0.71**, 0.65, 0.59 |
| lut3 G128 ACC1 / 2 / 4 | 2048 | 16 / 54 / 346, 301, 272 | 96 | 272, 269, 190 | 208.0, 185.5, 171.0 | 0.102, 0.091, 0.084 | **1.04**, 0.93, 0.85 |

- The stock floor reproduces tom-k1.md's 0.098.
- `SADDL(2)` is not in the tool's table, so it is counted on either pipe, like `SADDW`. A 1-register `TBL` counts as 1
  µop.
- **Where the time goes:** each lookup (16 rows x 4 inputs x 1 bit) costs about 4 vector µops: `TBL`, two widening
  adds, and half of an `AND` + `USHR` pair. Stock spends about 0.17 µops per weight, and the LUT kernel spends
  4 µops per 64 weight-bits. That is a tie at 2 bits before the float step, and 1.5 x at 3 bits.
- The G=128 builds spill the tables: 101 to 272 loads per 2048 weights. On the A78's two load pipes that is under
  the vector floor, but not by much for lut3.

## 3. Verdict against memo-11's kill line

The kill line: host k(2-bit) > 0.8 or k(3-bit) > 1.0 closes LUT kernels.

- **The T-MAC-faithful kernel (int8 tables, ACC 1) meets the kill line in every regime.** At 1 thread: k2 = 0.96
  (G32) and 0.84 (G128); k3 = 1.34 (G32) and 1.21 (G128) (*host*). The A78 counts agree: 0.92 / 1.28 at G32 and
  0.71 / 1.04 at G128 (*estimated*). **At 3 bits the LUT kernel is slower than `sdot` everywhere**, so lever A (Z1,
  re-encode to about 3.3 to 3.9 bpw with a LUT kernel) has no kernel. With ggml's Q3_K kernel it is slower than stock
  (memo-11 §4), so it is closed.
- **Only one variant passes both lines on the host: G=128 with 6-bit tables (ACC 4).** It gives k2 0.64 to 0.66 and
  k3 0.95 at 1 thread (*host*), and 0.59 / 0.85 by A78 count (*estimated*). It takes two extra numerics steps beyond
  T-MAC: one weight scale per 128 inputs, which is a format question for U3, and kernel NMSE about 30 x stock's q8
  input error (1.8e-3 with outliers). Even so, it does not reach **k ≤ 0.5**, which memo-11 §1 needs for 12 tok/s
  at 2.3 to 2.7 bpw. The lowest 2-bit numbers are 0.64 (*host*, 1 thread) and 0.59 (A78 count), and 0.50 appears
  only in a contended 4-thread cold row.
- **My reading:** close LUT kernels, and with them Z1 and the kernel half of §7/Z2. At 3 bits nothing beats `sdot`.
  At 2 bits the best measured kernel is about 0.6 to 0.65, not 0.5, and it needs coarser tables whose PPL cost is
  unknown. memo-11's centre k (0.50 at 2 bits, 0.75 at 3 bits) was optimistic by about 0.15 at 2 bits and by about
  0.5 at 3 bits for this kernel shape.
- **Why T-MAC's published speed-ups do not carry over:** T-MAC's 2-bit and 3-bit baselines were llama.cpp's Q2_K and
  Q3_K kernels of May 2024, not today's repacked Q4_0 4x4 `sdot` gemv. ggml's Q2_K already runs at about Q4_0's cost
  (R72), so a 5.8 x against Q2_K does not imply about 2 x against this stock.
- **Not covered:**
  - An asymmetric grid (Q2_K/Q3_K mins): one extra fused multiply-add per row per block, about +4 µops per 512
    weights at G32 (*estimated*).
  - A NEON table build.
  - T-MAC's own tiling and weight permutation, beyond the 16-row tile.
  - Mirror-consolidated 8-entry tables, which halve table memory but not the µops per lookup.
  - TRN-plus-`SDOT` and 2-register-`TBL` accumulation. I priced both on paper at the same 4 µops per lookup and did
    not build them.
- No device test is asked for.
