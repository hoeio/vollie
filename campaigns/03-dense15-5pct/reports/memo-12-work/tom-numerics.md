# c3 memo-12 work T-N1: host numerics check of the GPU lane (patch 0006, Adreno gemv emulation)

- **Date:** 2026-11-23 to 2026-11-24. **Author:** tom, `claude-opus-5-5`. Not committed.
- **Asked by:** R74 §7 item 1, the host numerics check, run on the owner's laptop with the owner's permission.
  The Q6_K head arm was added on the lead's request (Bonnie r4 §2).
- **Labels.** Every perplexity value and timing here is *host* (owner's Apple M4 Pro laptop). Percentages are
  arithmetic on *host* values. The identity-line replay of §2 compares a *host* computation with *device* values (R74).
  Nothing ran on the phone.

## 0. Result in short

| subject | setting | chunks | emulated (all Q4_0, mode 2) | stock served | Δ | budget | verdict |
|---|---|---|---|---|---|---|---|
| Yi-1.5-6B | `-b 512` | **656 (full)** | **8.3804** | 8.3875 | −0.085 % | 8.8069 | **inside** |
| Llama-2-7B | `-b 512` | **655 (full)** | **7.7448** | 7.7435 | +0.017 % | 8.1307 | **inside** |
| Yi-1.5-6B | `-ub 1` | 40 (subset) | 7.6803 (running value at 40) | 7.6817 | −0.018 % | (8.8048 full set) | not a full-set number |
| Llama-2-7B | `-ub 1` | 40 (subset) | 7.4052 (running value at 40) | 7.4075 | −0.031 % | (8.1314 full set) | not a full-set number |

- **Reading.** At share 1.0, the lane's arithmetic moves perplexity by less than 0.1 % on the full set. The budget is
  +5 %, so this is at least 50 times inside it, on both subjects. The sign is not consistent: Yi improves slightly,
  Llama-2 gets slightly worse. Both shifts are about 0.1 to 0.3 of the run's own standard error (±0.057 / ±0.053).
- Over the whole set, the running value never moves more than 0.48 % from stock on either subject (Yi −0.48 % to
  +0.02 %; Llama-2 −0.39 % to +0.04 %). The largest gaps are in the first 5 chunks, where one chunk carries the
  running mean.
- **Full-set `-ub 1` was not run.** At about 73 to 107 s per chunk on this host, which another agent was sharing, one
  run would take 13 to 19 h (§5). The case for reading `-b 512` as the decode figure:
  - The emulated output of a row is the same function at every batch size. Each column runs through the same
    per-row code, so the matmul arithmetic is identical at `-b 512` and at `-ub 1`.
  - At chunk 40, emulated `-b 512` and emulated `-ub 1` agree within 0.02 % (Yi 7.6816 / 7.6803; Llama-2
    7.4057 / 7.4052).
  - That is a reading, not a measurement: the `-ub 1` budget is formally checked only on the 40-chunk subset.
- **Q6_K head arm** (40-chunk subsets, §4): head alone −0.26 % (Yi) / +0.03 % (Llama-2) at `-b 512`. Body + head:
  −0.38 % / −0.02 % at `-b 512`, and −0.21 % / +0.02 % at `-ub 1` (10 chunks). All inside the budget, which is
  shared.
- **Budget verdict:** the GPU lane's numerics pass on both subjects at `-b 512` (full set). They also pass on the
  stated `-ub 1` subset. This holds at the worst case of lane share 1.0 on every Q4_0 matmul, and with the head too.

## 1. What 0006 does

- **Patch:** `patches/llama.cpp-c3/0006-x3-emu.patch`, 320 lines, sha256
  `b7a47981187bf307e103d320b6276c08ea505edcf706886de0a62a9d63b85fba`.
  - It sits on 0005 and applies with or without 0004, and together with 0007. Checked: the full series applies to a
    clean `git archive 620d2225c`, and without 0004 and 0007 it equals the tested tree `lab/llama.cpp-c3x6`.
  - Files: `ggml-cpu.c` (driver and hook), `repack.cpp` and `kleidiai/kleidiai.cpp` (decline the emulated type), and
    a new `vollie-x3emu.h` (kernels).
  - Self-check: `tools/c3_x3emu_check.c` (new; build line in its header).
- **Knobs** (off by default; unset, `0` or any other value means off):
  - `VOLLIE_X3_EMU=1|2`: every `MUL_MAT` with Q4_0 weights and f32 activations, at any batch size.
  - `VOLLIE_X3_EMU_Q6K=1|2`: the same for Q6_K, which here is the output head.
  - `1` = f32 multiply-add fused; `2` = multiply and add rounded separately. Every run below uses **2** (§2 says
    why).
- **Arithmetic, read from upstream's kernels** (`gemv_noshuffle_q4_0_f32.cl` and `_spec.cl` at `620d2225c`; the two
  have the same order):
  - **Weight:** `w = half((q & 0xF) − 8) × d`. OpenCL C forms the int × half product in half. The emulation forms the
    exact product in f32 and rounds it once to half, round to nearest even. That is the same value.
  - **Activation:** f32, with no Q8_0 step.
  - **Partial sums:** each work item owns two rows. The 4 subgroups of the work-group take blocks k ≡ g (mod 4), and
    inside a block the 32 weights are taken in order 0 to 31. So the emulation keeps 4 partials per row:
    `acc[k mod 4] += f32(w) × x`.
  - **Final sum:** subgroup 0 adds subgroups 1, 2 and 3 from local memory, so `out = ((acc0 + acc1) + acc2) + acc3`.
  - The phone has `vsgb=0` (R74 `x3 info`), so it runs the `_1_` macros. Their per-row order is the same as `_8_`'s.
- **Q6_K arm, read from `gemv_noshuffle_q6_k_f32.cl`:**
  - The kernel writes `((float) q − 32.f) × scale_s × scale_d × y`, where `scale_s` is a char and `scale_d` a half.
    The left operand is already float, so every step is f32. **Nothing is rounded to half.**
  - Bonnie r4 §2's concern (that (q − 32) × sub-scale, up to 4,064, is formed in half) does not apply to this kernel.
    The product is exact in f32, then rounded once in f32 by the × d.
  - Emulated weight: `f32(((q − 32) × sc) × d)`. Blocks of 32 weights, with the same 4-partial split and the same
    reduction as Q4_0.
  - The difference from stock CPU is mainly that the activation stays f32, where stock quantises it to Q8_K.
  - The noshuffle conversion (`kernel_convert_block_q6_k_trans4_ns`) keeps ggml's natural weight order. The check
    tool verifies the unpack against `dequantize_row_q6_K`'s loop.
- **Paths:**
  - With a knob set, the repack buffer type declines the emulated type, so those weights stay plain. Without that,
    the repacked Q4_0 would bypass the hook.
  - On this host, the Q6_K head stays repacked (`CPU_REPACK` 205 MiB on Yi) unless `VOLLIE_X3_EMU_Q6K` is set. The
    head's numerics in the body runs are therefore stock's.
  - Work is taken in 16-row chunks through ggml's chunk counter. A NEON tile of 16 rows per chunk uses fp16 vector
    multiplies, then f32 lanes, one row per lane, so each row keeps its own order. Rows left over (under 16) go
    through the scalar reference; none of these models has any.
- **At `-b 512` the emulation also runs.** Every batched Q4_0 matmul goes through it column by column. So `-b 512`
  here means "every token through the GPU kernel's arithmetic, prefill included".
  - A real lane would run only decode (ne1 = 1), with prefill on the CPU's path.
  - In a perplexity run at `-b 512`, a decode-only lane would never be taken, so the result would equal stock.
  - The emulation is therefore the worst case at both settings. No separate decode-only mode was built.

## 2. Identity checks

### 2.1 Knob unset: bit-identical (*host*, `llama-perplexity -c 512 -t 8 -ngl 0`, running values)

Reference: the stock build `lab/llama.cpp-c3/build`, the build that produced the S3 freeze values. Its CMake cache
equals `llama.cpp-c3x6/build`'s in every `GGML_*` and `LLAMA_*` entry.

| subject | setting | chunks | stock | 0006, knob unset |
|---|---|---|---|---|
| Llama-2-7B | `-b 512` | 4 | 5.8104, 6.3386, 7.0099, 7.9916 | identical |
| Llama-2-7B | `-b 512 -ub 32` | 4 | 5.7851, 6.3350, 7.0107, 7.9935 | identical |
| Llama-2-7B | `-b 512 -ub 1` | 2 | 5.7791, 6.3466 | identical |
| Yi-1.5-6B | `-b 512` | 4 | 5.0145, 6.9742, 7.4150, 8.4733 | identical |
| Yi-1.5-6B | `-b 512 -ub 32` | 4 | 5.0084, 6.9175, 7.3651, 8.4080 | identical |
| Yi-1.5-6B | `-b 512 -ub 1` | 2 | 5.0373, 6.9391 | identical |

- The stock `-b 512` and `-ub 1` values also equal the first chunks of the S3 freeze logs.
- The Llama-2 rows were rerun with each binary that was used: `build` (before the Q6_K arm, used for the body runs)
  and `build-q6` (the final source). All were identical. `build-q6` with only `VOLLIE_X3_EMU=2` reproduced `build`'s
  emulated values (Yi `-b 512`, 2 chunks: 5.0157, 6.9514).
- With the knob unset, no `x3emu:` line is printed.
- The last source edit, the KleidiAI decline, is not compiled in either macOS build: `GGML_CPU_KLEIDIAI` is off. It was
  syntax-checked against KleidiAI v1.31.0's sources only.

### 2.2 Emulation against the phone's own kernel (*host* replay, *device* reference)

`tools/c3_x3emu_check.c` replays x3b's identity cells:
- the same generator and seed as patch 0001;
- the same call order (x4, x11, then six matrices) and the same shapes;
- the same float64 reference as patch 0005.

It prints the emulation's max_rel_err and max_abs beside R74's device lines.

| cell | device (R74) max_rel_err / max_abs | mode 2 (unfused) | mode 1 (fused) |
|---|---|---|---|
| general 512×4096 | 6.336e-04 / 2.367e-03 | **6.336e-04 / 2.367e-03** | 6.334e-04 / 2.367e-03 |
| general 2816×4096 | 6.687e-04 / 2.593e-03 | **6.687e-04 / 2.593e-03** | 6.683e-04 / 2.592e-03 |
| general 11008×4096 | 7.634e-04 / 3.005e-03 | **7.634e-04 / 3.005e-03** | 7.639e-04 / 3.008e-03 |
| s4096_1_4096 | 6.910e-04 / 2.718e-03 | **6.910e-04 / 2.718e-03** | 6.908e-04 / 2.717e-03 |
| s4096_1_11008 | 6.233e-04 / 4.010e-03 | **6.233e-04 / 4.010e-03** | 6.235e-04 / 4.011e-03 |
| s11008_1_4096 | 6.882e-04 / 2.713e-03 | **6.882e-04 / 2.713e-03** | 6.880e-04 / 2.713e-03 |

- **Mode 2 reproduces all 12 printed device values to every printed digit. Mode 1 misses 10 of them** in the 4th
  digit.
- So the Adreno compile (`-cl-mad-enable`, E031.38) rounds the multiply and the add separately, and the emulation's
  order and half rounding match the device. Mode 2 is used for every run.
- **Limit:** these are maxima over rows, so the match is strong evidence, not a bit-level proof for every row.
- Also checked by the tool, bit for bit:
  - the NEON tile equals the scalar reference, for Q4_0 and Q6_K, modes 1 and 2, at 1 and 5 columns, including a
    K tail and subnormal scales;
  - the Q6_K unpack equals ggml's natural order.
- No Q6_K device identity exists. R74 did not build the Q6_K kernel, so the head arm rests on reading the source.

## 3. Body arm, per-chunk running values (*host*, `VOLLIE_X3_EMU=2`)

| subject, setting | [1] | [5] | [10] | [40] | [100] | [400] | final |
|---|---|---|---|---|---|---|---|
| Yi `-b 512` emulated | 5.0157 | 8.4580 | 9.0042 | 7.6816 | 8.4872 | 8.0948 | **8.3804** (±0.05697) |
| Yi `-b 512` stock | 5.0145 | 8.4984 | 9.0373 | 7.6966 | 8.5003 | 8.1032 | 8.3875 |
| Yi `-ub 1` emulated | 5.0118 | 8.4544 | 9.0037 | 7.6803 | | | (40 chunks) |
| Yi `-ub 1` stock | 5.0373 | 8.4567 | 9.0047 | 7.6817 | | | 8.3855 |
| Llama-2 `-b 512` emulated | 5.7876 | 8.1304 | 8.9735 | 7.4057 | 7.9393 | 7.6894 | **7.7448** (±0.05257) |
| Llama-2 `-b 512` stock | 5.8104 | 8.1321 | 8.9739 | 7.4071 | 7.9371 | 7.6876 | 7.7435 |
| Llama-2 `-ub 1` emulated | 5.7867 | 8.1282 | 8.9737 | 7.4052 | | | (40 chunks) |
| Llama-2 `-ub 1` stock | 5.7791 | 8.1337 | 8.9745 | 7.4075 | | | 7.7442 |

Logs are in `lab/c3/logs/`: `x6emu-<subject>-b512.log` and `x6emu-<subject>-ub1-40.log`. The stock logs are the S3
freeze logs, `lab/logs/p/ppl-c3f-<subject>-q4_0-{b512,ub1}-wiki.log`.

## 4. Q6_K head arm (*host*, `VOLLIE_X3_EMU_Q6K=2`, subsets)

| subject | arms | setting | chunks | running value | stock at the same chunk | Δ |
|---|---|---|---|---|---|---|
| Yi | head | `-b 512` | 40 | 7.6768 | 7.6966 | −0.257 % |
| Yi | body + head | `-b 512` | 40 | 7.6672 | 7.6966 | −0.382 % |
| Yi | body + head | `-ub 1` | 10 | 8.9859 | 9.0047 | −0.209 % |
| Llama-2 | head | `-b 512` | 40 | 7.4091 | 7.4071 | +0.027 % |
| Llama-2 | body + head | `-b 512` | 40 | 7.4059 | 7.4071 | −0.016 % |
| Llama-2 | body + head | `-ub 1` | 10 | 8.9763 | 8.9745 | +0.020 % |

- The head arm moves Yi more than the body arm does (−0.26 % against about −0.2 % at 40), and always downwards.
- One reading, not tested: Yi's head is 64,000 × 4,096, and stock quantises the head's input to Q8_K. The f32 input of
  the GPU kernel removes that error.
- Every value is far inside +5 %. These are subsets, not full-set numbers.
- Logs: `lab/c3/logs/x6q-{head,both}-<subject>-b512-40.log` and `x6q-both-<subject>-ub1-10.log`.

## 5. Run times (*host*, wall, shared machine)

- The laptop was shared for most of these runs: the owner's other project, a VM, and the other Tom's KleidiAI stock
  chain. Load average was 9 to 50 on 12 cores, so these times are not speeds.
- For scale: the S3 freeze took 2.5 to 2.8 s per pass at `-b 512` and 9.1 to 11.0 s at `-ub 1`.

| run | wall | s per pass printed (first pass) |
|---|---|---|
| Yi `-ub 1`, 40 chunks | 57 min 43 s | 422 (first pass under heavy load; later about 75 to 120) |
| Llama-2 `-ub 1`, 40 chunks | 49 min 36 s | 73.4 |
| Yi `-b 512`, full | 4 h 02 min | 15.7 |
| Llama-2 `-b 512`, full | 3 h 44 min | 35.3 |
| Q6_K: Yi head / body + head `-b 512`, 40 | 6 min 57 s / 22 min 10 s | 15.9 / 33.4 |
| Q6_K: Llama-2 head / body + head `-b 512`, 40 | 7 min 53 s / 14 min 03 s | 12.7 / 35.8 |
| Q6_K: Yi / Llama-2 body + head `-ub 1`, 10 | 11 min 54 s / 13 min 08 s | 70.6 / 78.6 |
| 0006 identity set (12 runs, two builds) | about 1 h 10 min under load | |

- A full-set `-ub 1` emulated run needs about 13 to 19 h on this host, one at a time (*estimated* from 73 to 107 s
  per chunk × 655 chunks).
- The emulation is compute-bound at about 0.15 ns per weight per thread (*host*, single thread, mode 2). Moving the
  transpose into the byte domain, 8 rows per fp16 vector, might roughly halve that (*estimated*). That work was
  not done.
- Each run had `-t 8`. Some ran in parallel with my own second chain and starved the other Tom's KleidiAI stock runs.
  Coordinator's note: future runs go one at a time.

## 6. Which build these figures are on, and the KleidiAI question (Bonnie r10 §1)

- **Every figure here is on the macOS host build without KleidiAI:**
  - `GGML_CPU_KLEIDIAI` off; `GGML_CPU_ARM_ARCH=armv8.2-a+dotprod+fp16`; repack on; BLAS off; Metal off.
  - The CMake cache is equal to `lab/llama.cpp-c3/build`, the S3 freeze's stock.
  - The stock comparator is therefore the freeze's non-KleidiAI stock.
- **The phone's Q4_0 path is KleidiAI.** Before this round, 0006 patched only the generic matmul and the repack type,
  so on a KleidiAI build the Q4_0 weights would go to KleidiAI first and the emulation would never run.
  - The final 0006 adds the same decline to KleidiAI's `supports_op`: with a knob set, Q4_0 and Q6_K stay plain.
  - That edit is not compiled or run here.
- **What a KleidiAI-build reading needs:**
  1. Build 0006 with `-DGGML_CPU_KLEIDIAI=ON` (the same recipe as `tom-kai-stock.md` §1, separate build directory).
     Check that with the knob set, no Q4_0 or Q6_K tensor is in the KleidiAI buffer: `-lv 4` shows no `CPU_KLEIDIAI`
     buffer for them.
  2. With the knob unset: per-chunk identity against that build's own stock at `-b 512`, `-ub 32` and `-ub 1`.
  3. Use the KleidiAI-build stock (the other Tom's runs) as the comparator.
- **Expectation** (*estimated*, to be checked):
  - With the knob set, every Q4_0 and Q6_K matmul goes through the same emulation. The other ops (attention, norms,
    the F32 paths) are the same code on both builds, unless KleidiAI also takes F32 or F16 matmuls; `supports_op`
    includes `GGML_TYPE_F32` weights. The emulated PPL would then equal the values above, and only the reference
    changes: Δ = emulated ÷ KleidiAI stock.
  - A few chunks at `-b 512` would confirm whether the emulated running values equal those in §3.
- Not run, at the coordinator's request: the laptop goes to the KleidiAI stock runs first.

## 7. Why every `x3 loadlog` reads `Error: Program not built!` (no device run; inference)

- **What the probe does** (0005 `get_prog`), in order:
  1. A child process (`libvollie_clc.so`) builds the source and saves `CL_PROGRAM_BINARIES`. Its build log is empty
     (`x3 build ... log_bytes=0`).
  2. The parent calls `clCreateProgramWithBinary` and gets `bin_status=0` and `create_err=0`.
  3. It calls `clBuildProgram`, which returns `build_err=0`.
  4. It queries `CL_PROGRAM_BUILD_LOG` (0x1183; the constant is correct) and prints any log longer than one byte.
- **Reading.** The binary is a fully built executable, so the Qualcomm driver's `clBuildProgram` on it compiles
  nothing in this process. The driver's front end never runs, so the program has no build log of its own. Asked for
  one, this driver returns its placeholder text "Error: Program not built!", not an empty string.
- **Evidence that nothing failed:**
  - every load cell shows `build_err=0`, and `clCreateKernel` returns `kernel_err=0`, with sane work-group values
    (`wg_max=512`);
  - every identity cell passes;
  - the emulation in §2.2 reproduces the device's identity numbers exactly, so the loaded kernels computed the
    intended arithmetic;
  - the same text appears for `how=cached` loads.
- **Not checked** (no device run, no driver source). Two queries in a later kit would confirm the reading, at no
  cost:
  - `CL_PROGRAM_BUILD_STATUS` (0x1181): expected `CL_BUILD_SUCCESS` = 0;
  - `CL_PROGRAM_BINARY_TYPE` (0x1184): expected `CL_PROGRAM_BINARY_TYPE_EXECUTABLE` = 4.
- **Suggestion for the probe:** print `loadlog` only when `build_err != 0`, or print it with the build status beside
  it, so the line is not read as a failure.

## 8. Limits

- `-ub 1` is a 40-chunk subset for the body arm and a 10-chunk subset for body + head. The full-set `-ub 1` budget is not
  measured.
- Mode 2 matches the device's identity maxima; per-row bit equality with the Adreno kernel is not proven. FTZ
  behaviour for half subnormals on the Adreno is unknown. The emulation keeps subnormals, as the CPU does, and
  subnormal Q4_0 scales are rare.
- The Q6_K arm rests on reading the source. The 660 never ran that kernel.
- Lane share 1.0 is the worst case for the body. A real lane at share 0.25 to 0.5 mixes CPU rows and GPU rows, and
  should sit between stock and these values (*estimated*).
- Every number is *host*. Speeds here say nothing about the phone.
