# The expert kernel: what the 116 µs slot is made of, and what a byte-rate Q3_K gemv on the A78 can and cannot be (Bonnie, round 10; 2026-11-06)

Labels: *device* = `reports/device/r8_2026-11-06.json` (clean day, 2,112 / 2,035 MHz, 117 + head2 story w5-6 and the (c) ABBA) and R27's clock
experiment; *host* = disassembly of the compiled kernels (`lab/logs/r33/q3k_kernel/bonnie_disasm/`: `build-888` = `armv8.2-a+dotprod+fp16`, Apple
clang, **and the r9 APK's `libggml-cpu-android_armv8.2_2.so` = NDK clang — same loops to the instruction**); *estimated* = arithmetic. Pipeline
model: Cortex-A78 = 2 × 128-bit ASIMD pipes (V0/V1), `sdot`/logical/shift/`addp` dual-issue, 4-wide decode — from the A78 SOG as I remember it
(the fetched PDF would not render); **Tom's microbench is the check** (slot `[T1]`). Slots `[T…]` are for Tom's numbers when they land.

## 1. Cost model of the current kernel — the 2.5 × is instruction issue, not memory

Shapes (GGUF: `n_embd` 1536, `n_ff_exp` 512, 64 experts, 40 MoE layers, Q3_K experts): one slot = gate + up + down of one expert =
**3 × 786,432 = 2,359,296 MACs = 9,216 Q3_K row-blocks (256 weights, 110 B) = 1,013,760 B**; 3 threads on 32-row chunks → **3,072 row-blocks
per thread per slot**. Measured: `MUL_MAT_ID` compute 19.3 ms / (40 × k̄ 4.01 = 160 slots) = **120 µs**; skip ABBA 8.28 ms / 71.5 slots = **116 µs**
(*device*) → **80-82 cycles per row-block at 2,112 MHz**.

Instruction mix of `ggml_vec_dot_q3_K_q8_K_x4` (ILP path = the served default), per row-block, *host* disassembly, device build in brackets:
**183 [185] instructions: 140-143 NEON-ALU, 15 loads, 3 stores, ≈ 22 scalar/branch.** The NEON ops by purpose:

| purpose | ops / row-block | share |
|---|---|---|
| 3-bit → int8 unpack: `and` 24, `ushr` 18, `sub` 16, `mvn`/`bic`/`shl` 12, `add.16b` 4 | **74** | 53 % |
| `sdot` | 16 | 11 % |
| per-16-weight scale apply: `addp` tree 12 + `mul`/`mla` 6 + `sshll` widen 6 | 24 | 17 % |
| compiler zeroing (`movi` for every `vdotq(vzero, …)`) | **16** | 11 % |
| scale unpack (`ushl`/`and`/`shl`/`orr`, per 4 rows ÷ 4) | 7 | 5 % |
| float tail (`addv`, `fcvt`, `fmul`, `fmla`) | 3 | 2 % |

Floors per row-block: **NEON issue 140 / 2 pipes = 70-71.5 cycles**; front end 185 / 4 = 46; loads 15 / 2 = 7.5. Measured 80-82 → the kernel runs at
**87-89 % of its NEON-issue floor**. Per slot: issue floor 70 × 3,072 / 2,112 = **102-104 µs**, stalls 12-16 µs. Memory is not it: the Q8_K activation
is 1.75 KB (gate/up) / 0.6 KB (down), in L1, loaded once per block for 4 rows (4 of the 15 loads); the `hmask` is 2 loads inside the same 110-B
block; the weight stream runs at 1.01 MB / 116 µs = **8.7 GB/s = 38 % of the byte rate**. R27's clock experiment (×0.736 clock → `MUL_MAT_ID`
×1.24-1.3) says 65-80 % of the line scales with the clock; the rest is DRAM stream-head latency per 32-row chunk plus chunk atomics and thread
arrival, overlapped by the OoO core. **Verdict: the 2.5 × over the byte floor (116 vs 44.5 µs) is 2.3 × of NEON instruction count, 0.3 × of stalls.**
§6.2's 120-op / 94 % estimate was a hand count; the compiled code carries 16 `movi` and 6 in-loop widens it did not see.

Why the repacked Q4_0 head ran at the byte rate (R31): `ggml_gemv_q4_0_4x4_q8_0` is **33 instructions / 128 MACs, 22 NEON** (8 `sdot` by lane,
4 `shl`, 4 `and`, 6 float) = 0.17 NEON ops/MAC against Q3_K's 0.55 → compute at Q4_0 bytes would be 32 µs/slot under a 58 µs byte floor. The
difference is not "sdot vs vec_dot": both are sdot. It is (1) nibbles unpack with one op per 16 weights, 3-bit fields need 4.6; (2) one scale per
32 weights applied in float per 4 rows, Q3_K has one per 16 applied in integer per row (the `addp` tree).

## 2. Floors and the target (per slot, 3 threads, 2,112 MHz; all *estimated* from the counts above)

| floor | arithmetic | µs/slot |
|---|---|---|
| bytes (served Q3_K, unchanged) | 1,013,760 B / 22.8-23.9 GB/s | **42-45** |
| `sdot` throughput | 147,456 sdot / 3 threads / 2 per cycle / 2,112 | 11.6 |
| unpack, dotprod-only, Q3_K layout in place | per 16 w: `and` + `bic` + `sub` + 1.5 shifts = 4.5, + `sdot` 1, + scale apply 0.5 (`mla` by lane, tree gone), + scale unpack/float 0.9 → **≈ 108-110 NEON / row-block = 54-55 cycles** | **78-80** ideal; **87-90 at the measured 88 % issue efficiency** |

The in-place unpack floor is 1.8-2 × the byte floor: **a Q3_K kernel that reads the served bytes cannot reach the byte rate on a 2-pipe dotprod core**;
the 3-bit field extraction alone (4.5 ops / 16 w) costs more issue slots than the bytes cost bandwidth. Every alternative I costed loses: the h-bit as
a second `sdot` (+1 sdot per 16 w, saves the `sub`: 0 net, §6.2 agreed); masked-in-place values with `ssra` normalisation (2.75 + 3.25 = 6.0 per
16 w); the 8 × int8 pre-multiplied `q·scale` (overflows at raw scale 0: −4 × −32 = 128). **Target (bytes unchanged): 85-95 µs/slot, central 90**
= −21..−31 µs × 164 slots = **−3.4 to −5.1 ms/token, central −4.2.** Steady 53.0 → **47.9-49.6**; printed at 1.13 ms outside, fault-free:
1000 / (steady + 1.13) = **19.7-20.4, central 20.1**. 20.0 printed needs steady ≤ 48.9 → slot ≤ **91 µs** (−25 µs, −4.1 ms/token). So the
in-place kernel crosses 20 printed **only in the better half of its band, on a clean 2,112 day, fault-free** — margin ±0.3 tok/s, not a safe
cross. What crosses it with margin is §3 (e) on a phone that can hold the copy, or (c) + (f) together (−3.8..−5.8 → 19.9-20.7).

## 3. Design options, ranked (ns/row relative to the current 38 ns/row-block *device*; "exact" = integer block sums identical, float ops identical)

| rank | option | per-row time | bytes | exact | plugs into | risk |
|---|---|---|---|---|---|---|
| **1** | **(c′) in-register rewrite of the x4 kernel**: explicit accumulators (no `movi` per `sdot`: chain `sdot` into one acc per group or keep one zero register), scales widened once per block outside the r-loop, `bic`+`shl` for the h-term (not `mvn`/`shl`/`and`), **`addp` tree → `mla` by lane** (4 per half instead of 8), no acc spills; inner loop pinned with intrinsics that both clangs keep (or inline asm) | 140 → 108-115 NEON ops → **−20..−25 %** (slot 88-95) | unchanged | **bit-identical** (integer adds reordered = exact; float tail unchanged) | `arch/arm/quants.c` behind `GGML_Q3K_KERN=1`, called from `ggml-cpu.c:1802` | the NDK may re-introduce `movi`; the `mla`-by-lane may be V0-only (then the tree is cheaper by 1-2 cycles — Tom's microbench decides `[T1]`) |
| 2 | **(f) fused gate + up (+ GLU) per expert**: one pass over the shared Q8_K activation computing 4 gate rows and 4 up rows, `silu(g)·u` in the epilogue; one `MUL_MAT_ID` node instead of two per layer | kernel ±0; saves 40 ops' dispatch + barrier + the GLU op (0.39 ms) | unchanged | exact if the same `silu` path is called | graph build + `mmid_decode` | R20's "open item"; −0.4..−0.7 ms/token, orthogonal to 1 |
| 3 | **(e) exact 4-bit container, 4x4-interleaved** (Q3_K values re-laid as nibbles with the Q4_0 top-nibble sign trick, per-16 int8 scales pre-expanded, `sshr #4` once per block): the Q4_0x4 code shape with K-quant scales | **≈ 40 NEON / row-block → 28 µs compute, 59 µs bytes → slot 62-68 (−45 %)** = the byte rate | **+33 % (1.35 MB/slot)** | exact (same ints, same float tail) | a per-expert anon copy like `expq4` (`ggml_cpu_expq4_chunk`), new type | **memory**: all experts 3.45 GB anon = R9's OOM; a 16-expert/layer LRU (862 MB, R23 hit 0.64-0.78) nets 85-95 µs = no better than rank 1 on this phone, thrashing-day negative; **the generalisable byte-rate kernel for ≥ 12 GB phones** |
| 4 | (a) Tom's 4-row interleave via per-slice repack, uncached | repack = the unpack + a store pass ≈ 60 % of a slot, then a 62-68 µs gemv → **≥ 125 µs, slower** | +33 % transient | exact | scratch per slot | pays only as (e) with a cache |
| 4 | (a′) in-register interleave (`zip`/`tbl` after unpack to put rows in lanes) | ≥ 4 ops added per 4 vectors to delete 8 of the tree → **0 ± 5 %** | unchanged | exact | quants.c | not worth a day |
| 5 | (b) int8 pre-expanded reference | host: the no-unpack bound, **≈ 30 NEON / row-block → 3-4 × faster per row cache-hot** `[T2]`; device: 2.7 MB/slot = 118 µs byte floor → no gain | ×2.67 | exact | diagnostic only | tells how much of a prototype's gain is unpack (should match the 53 % share) |
| 6 | (c) ILP on the current kernel as is (more rows in flight, prefetch) | the kernel is at 88 % of issue; **−0..−10 %** | unchanged | exact | quants.c | R24 already took this step; the stall is 12-16 µs, not the chain |

Not an option: i8mm (`smmla`) is Armv8.6, the SD888's A78/X1 do not have it, and it halves the `sdot` share (11 %), not the unpack.

## 4. Generalisation and the device measurement

Rank 1 helps **every Q3_K tensor on every dotprod arm64 core with 2 ASIMD pipes** (A76/A77/A78/A710 and the X1/X2 at 4 pipes to a lesser degree):
the compiled kernel is 11 % `movi`, 17 % scale tree, and the unpack dominates on all of them; it also helps dense Q3_K `MUL_MAT` on any model.
Rank 3 is the portable byte-rate expert kernel for phones with ≥ 12 GB (4.3 GB for a 3.4 GB Q3_K expert set). The measurement: ABBA
`Q3K_KERN:0:1` on 117 + head2, `MUL_MAT_ID` **compute** line, converted to µs/slot with each side's `kavg × 40`; pass/falsify in §5. At 1,555 MHz
the Δ scales ×1.24-1.36 (compute-bound, R27); quote both clocks, correct with the §7.9 rule.

## 5. Round-10 pre-registration draft (for Clyde to freeze; all *estimated*; ABBA 88 tokens/side on 117 + head2, every number with its `mhz_blocks`)

- **(i) `Q3K_KERN:0:1`** (rank 1, bit-identical). `MUL_MAT_ID` compute B − A at 2,112: **−3.4 to −5.1 ms** (−21..−31 µs/slot), central −4.2;
  **pass ≤ −3.0 ms (≤ −18 µs/slot); falsified > −2.0 ms (> −12 µs/slot)** → the kernel is not issue-bound as modelled, in-place Q3_K is closed and
  only (e) remains. At 1,555: band −4.2..−6.9, pass ≤ −3.7, falsified > −2.5. Wait 0 ± 0.3; `MUL_MAT` 0 ± 0.3; k̄ / share / `adv` unchanged; fault-matched
  (B majflt ≤ A + 10) or unscorable. **Byte rate reached** = slot ≤ 1.15 × 44.5 = **≤ 52 µs** (`MUL_MAT_ID` ≤ 8.5 ms at 164 slots): not expected for (i).
- **(ii) `MMID_FUSE:0:1`** (rank 2, if built): `MUL_MAT_ID` wait −0.2..−0.4, `GLU` −0.3..−0.4, compute 0 ± 0.3, token −0.4..−0.7; falsified token > 0.
- **(iii) always-on** default + (i) [+ (ii)]: story 200 steady **47.9-49.6** [47.5-49.2] at 2,112 on a ≤ 10 majflt/token run; printed fault-free
  19.7-20.4 [19.9-20.7]; falsified > 50.5 on such a run. Official 32 ≥ 17.0 on an r8-like day.
- **Identity, before the build** (host, `lab/logs/r34/id.sh`): (1) unit check, 10⁵ random Q3_K row-quads × Q8_K rows, `s[r]` **bitwise equal** old vs
  new, all 4 rows, K = 512 and 1,536; (2) PPL chunks 1-3 at −ub 1 / −ub 32 / −b 512 equal to the r9 default **to all printed digits** (5.4733 /
  7.3025 / 7.0285-class values; "4 digits" is the floor, bit-identical is the claim); (3) greedy 32 on the kit prompts identical text; (4) the device
  build's disassembly of the new inner loop shows **≤ 2 `movi` and no `str q` per half-loop** (otherwise the NDK undid the rewrite — fix before the kit).
- **Clyde's wrong-prediction guard**: if (i) lands at −2.0..−3.0 the gain is real but 20 is not crossed; the next step is then (e) on a 12 GB device,
  not more Q3_K work on this one.

`[T1]` Tom's microbench: ns/row-block of x4-ILP vs the rewrite, cache-hot, 1 thread, host; expected ratio 0.75-0.80 on the M4 (4 pipes, so smaller
than the device's). `[T2]` (b) ratio. `[T3]` (a) per-slice repack cost per slot in µs.
