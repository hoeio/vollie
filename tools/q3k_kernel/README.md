# Round 10 audit: the Q3_K expert gemv on the A78 (Tom, 2026-11-06)

Run: `lab/logs/r33/q3k_kernel/bench.sh` (host M4, 1 thread, cache-hot, build-888 kernels + `kernels.c`;
exits 1 unless every prototype is bit-identical to `ggml_vec_dot_q3_K_q8_K`). It also prints the A78
instruction mix of each kernel compiled by the NDK clang (`-mcpu=cortex-a78`, the device compiler) and of the
shipped `libggml-cpu-android_armv8.2_2.so`. Output of one run: `bench.out`. Weights: expert 7 of layer 20 from
`lab/models/var/d6-q3k-e6.gguf` (read-only); activation N(0,1) quantized to Q8_K.

Labels: *host* = M4 measurement; *device-asm* = static count of the device compiler's code (exact, per block);
*estimated* = arithmetic on those.

## 0. Stop condition hit (both of them)

1. **The device path already is a 4-row `sdot` kernel.** `GGML_MMID=3` → `ggml_cpu_mmid_decode` →
   `ggml_compute_forward_mul_mat_id_one_chunk` → `ggml_vec_dot_q3_K_q8_K_x4` (`GGML_Q3K_X4=1`) → `..._x4_ilp`
   (`GGML_Q3K_ILP=1`): four Q3_K rows in the stock layout, the Q8_K activation loaded once per block and shared,
   per-row `sdot` + `addp` tree + per-16 scale `mla` (`GGML_Q3K_VEC=1` is the same vectorized scale step for
   the 1-row tail). It is not lane-interleaved (lanes = weights, not rows), but that is not what costs.
2. **The 3-bit unpack dominates every variant on dotprod.** The kernel is NEON-issue bound, not byte bound.

## 1. Where the 116 µs goes

Shapes (GGUF): n_embd 1,536, n_ff_exp 512. gate, up: 512 rows × 1,536 (6 blocks, 660 B/row); down: 1,536 rows ×
512 (2 blocks, 220 B/row). Slot = 2,560 rows, 3 × 786,432 = 2.36 M MACs, 1,013,760 B.

Device-asm, shipped `ggml_vec_dot_q3_K_q8_K_x4`, per Q3_K block × 4 rows (1,024 MACs):

| | sdot | other SIMD | loads/stores | scalar | SIMD per 16 weights |
|---|---|---|---|---|---|
| shipped x4 ILP | 64 | 498 (and 104, ushr 72, **movi 69**, sub 64, **addp 48**, shl/add/mvn/bic 72, scale widen/mla 40, …) | 84 | 98 | **8.8** |
| Q4_0x4 gemv (same 1,024 MACs) | 64 | 112 | 48 | 40 | 2.75 |

0.55 SIMD op/MAC (0.73 instr/MAC) vs 0.17 (0.26) for Q4_0x4. The A78 issues SIMD on 2 pipes: 562 / 2 = 281
cycles per 4-row block → 2,304 blocks per slot / 3 threads / 2,112 MHz = **102 µs of pure SIMD issue against the
measured 116** (88 %). Byte floor of the slot 1,013,760 B / 22.8 GB/s = 44.5 µs. The 2.5 × is instruction
issue: at the byte rate a core gets ≈ 3.8 SIMD slots per 16 weights; `sdot` takes 1, the 3-bit unpack alone
needs ≥ 3.5-4.25 (2-bit plane: and / ushr+and; high bit: shift+and; merge: orr), the per-16 scales ≈ 1.

Host ns/row, 1 thread, cache-hot (4 runs, spread ±5 %; load average 2-4 from other work):

| kernel | 1,536 cols | 512 cols |
|---|---|---|
| Q3_K vec_dot (stock 1-row) | 52-56 | 18-19 |
| **Q3_K x4 ILP (device path)** | 43-51 | 15.5-16.5 |
| Q4_0 vec_dot | 30-34 | 10.6-11.7 |
| Q4_0x4 gemv (repacked sdot) | 15-17 | 4.9-5.6 |
| Q6_Kx8 gemv (repacked 8x4) | 30-34 | 10.4-11.2 |

Slot (gate+up+down): device path 72-75 µs, Q4_0x4 24-26 → **Q3_K/Q4_0x4 = 2.9-3.0 on the host** (device-asm
SIMD ratio 3.2).

## 2. Prototypes (all bit-identical: integer block sums exact, stock float step; max abs/rel error 0, every row)

| variant | host slot µs (× device path) | host ns/row 1,536 / 512 | SIMD/4-row block (device-asm) | instr/MAC | bytes/row | A78 µs/slot *est.* | no copy? | Tom-days |
|---|---|---|---|---|---|---|---|---|
| device path (x4 ILP) | 72-75 (1.00) | 43-51 / 15.5-16.5 | 562 | 0.73 | 660 / 220 | 116 (*device*) | yes | — |
| (c) x4 + bsl unpack, bias from `bsums` | 68-74 (0.91-1.00) | 42-48 / 14.2-15.1 | 573 | 0.67 | same | 118 | yes | 0.5 (no gain) |
| (a) 4-row lane-interleave in registers | 69-71 (0.92-0.96) | 42-47 / 13.7-15.1 | 486 | 0.59 | same | 100 | **yes** | 1 |
| (a2) = (a), high bit as its own `sdot` | 73-74 (0.98-0.99) | 44-49 / 15.0-15.5 | 518 | 0.63 | same | 107 | yes | — |
| **(a3) = (a) tuned** (4 rows' scales together, planes 1-2 in place, hmask shifted once) | **66-69 (0.89-0.93)** | 40-45 / 14.8-14.9 | **471** | 0.58 | same | **97** | **yes** | **1** |
| (a-rp) (a) on a `block_q3_Kx4` copy | 61-63 (0.82-0.86) | 37-42 / 12.4-14.1 | 442 | 0.54 | same | 91 | no | — |
| ↳ + repack at read time (per 4 rows) | +22.5-23 (0.30) | +13-15 / +4.5-5.1 | +48 SIMD +60 LS | | | ≈ 103+ | scratch | — |
| (b) Q3_K8 int8 pre-expanded (floor ref.) | 29-32 (0.41-0.43) | 18-21 / 6.5-8.0 | 167 | 0.26 | 2.53 × | 34 compute, **113 bytes** | no | n/a |
| Q4_0x4 gemv (reference) | 24-26 (0.33-0.35) | 15-17 / 4.9-5.6 | 176 | 0.26 | 18/32 w | 58 (bytes) | — | — |

A78 µs/slot *estimated* = 116 × SIMD ops / 562 (issue-bound model calibrated on the measured slot; the model
puts the shipped kernel at 88 % SIMD-issue), floored by bytes / 22.8 GB/s.

- (a) vs (b): **2.2 ×**, not ≤ 1.3 × → unpacking is not amortised; it is ≈ 2/3 of the SIMD ops (486 − 167).
- No-repack vs repacked: (a3) is within 7 % of (a-rp)'s kernel, and the repack costs 0.30 × the device path
  per slot — **repack at read time is a net loss**. It cannot amortise at batch 1: every weight is used once per
  token, so a repack (or an int8 expansion) into scratch only moves the same per-weight work into another pass.
  The register interleave is the feasible form: 4 row pointers, `trn1/trn2` 4×4 word transposes (0.75 SIMD /
  16 weights), `sdot` by lane, no `addp` tree, no per-group `movi` (the accumulator starts at −4·bsum).
- Plugs into `MMID=3` as a drop-in for `ggml_vec_dot_q3_K_q8_K_x4` (same signature and 4-row granularity, the
  chunk loop and the 1-row tail unchanged), no copy, no memory.

## 3. Recommendation (host / estimated)

1. **A byte-rate Q3_K gemv does not exist on this ISA.** At 22.8 GB/s a core has ≈ 3.8 SIMD slots per 16 Q3_K
   weights; `sdot` plus the cheapest 3-bit unpack I found is already ≥ 5. Best demonstrated:
   **(a3), −7 to −11 % of the expert op on the host, −16 % by the A78 SIMD count → −1.3 to −3.1 ms/token
   *estimated*** (164 slots × 8-19 µs). Bit-identical, no memory, ≈ 1 Tom-day (knob `GGML_Q3K_T4`, identity
   check, phone build). Worth one round as an exact lever; it does not reach 20 alone.
2. Ceiling of the same idea in hand-written asm (no compiler-spilled `movi`/`mov`, ≈ 3.5 SIMD per 16 weights):
   ≈ 400 SIMD per 4-row block → ≈ 83 µs/slot, **−5 ms/token *estimated*, not demonstrated**, 2-3 Tom-days,
   uncertain.
3. Not prototyped, flagged: the A78 has 4 integer ALUs idle during this kernel. Doing part of the unpack with
   64-bit scalar SWAR (8 weights per op) and handing bytes to NEON through L1 could move work off the 2 SIMD pipes
   (e.g. 40 % of vectors → ≈ 360 SIMD per block). It is the only route I see below the SIMD-issue floor; it cannot
   be judged on the M4 (8 ALUs, 4 SIMD pipes) — a device microbench would be needed first.
4. Dead: per-slice repack (net loss, above), the int8 copy (bytes 2.53 ×), (a2) and (c).

Not done: 3-thread op-level numbers and any device measurement; the A78 numbers above are a static
instruction-count model, not a run. Q6_Kx8 was timed on random bytes (timing only).
