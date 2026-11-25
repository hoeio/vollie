# c3 memo-12 work: `_mc3` as built, kernel or probe?

- **Date:** 2026-11-25. **Author:** tom, `claude-opus-5-5`. Not committed. No shipped patch is changed.
- **Question** (R75-result.md, next step 2): in R75 x4m, two columns cost 3.316 times one column (2,951 µs against
  889.9 µs, *device*). Is that the kernel or the probe's launch?
- **Labels.** The x4m values are *device* (kit r36 JSON `reports/device/r75a1-c3probe-r36-2026-11-24T1345.json`). The PoCL
  timings are *host*, from a CPU OpenCL in Docker, and are relative only. They say nothing about the Adreno. The
  per-column arithmetic is *estimated*.

## 1. Answer

**The cost is in the kernel, not in the probe.** The probe launches `kernel_gemv_noshuffle_q4_0_f32_mc3` exactly as
upstream's `ggml_cl_mul_mat_q4_0_f32_adreno` does at 620d2225c, and builds it with the same options. The *device* lines
show a large fixed step at the first extra column and a small linear cost after it. A two-column kernel is therefore
**an unbuilt idea, not a dead one**, but this kernel will not give it. The R75 rule stands as scored: cg 3.316 and
`_mc3` dropped.

## 2. Dispatch, probe against upstream (620d2225c, `ggml-opencl.cpp` lines 19991 to 20089; probe `0007` `launch_mc`)

| item | upstream | probe | same? |
|---|---|---|---|
| kernel | `kernel_gemv_noshuffle_q4_0_f32_mc3`, from the `gemv_noshuffle_general` program (also holds the 1-column `kernel_gemv_noshuffle_q4_0_f32`) | the same program, built by x4i | yes |
| build options | `-cl-std=CL2.0 -cl-mad-enable -DSIMDGROUP_WIDTH=64` (`adreno_wave_size` 64), `-DVECTOR_SUB_GROUP_BROADCAST` if supported, `-DGGML_CL_A6X_CONSTFOLD_FIX` on A6x | *device* log: `opts=-cl-std=CL2.0 -cl-mad-enable -DSIMDGROUP_WIDTH=64 -DGGML_CL_A6X_CONSTFOLD_FIX`, try 1 (not the `-cl-opt-disable` fallback); `a6x=1 vsgb=0` | yes |
| weight image | `CL_R` / `CL_UNSIGNED_INT32`, `IMAGE1D_BUFFER`, width M·K/8 | the same (identity passes on all 9 columns) | yes |
| activations | sub-buffer at `offset1`, `CL_RGBA` / `CL_FLOAT` `IMAGE1D_BUFFER`, width K·N/4, column-major | the buffer itself (offset 0), the same format and width | yes (offset 0 is upstream's usual case) |
| arguments 0 to 7 | q image, d buffer, b image, dst, `offsetd`, K, M, `n_cols` | the same order and types | yes |
| local size | {64, nsg, 1}, nsg = 8 if M < 4096, else 4 | {64, 4, 1} at M 11008 | yes |
| global size | {CEIL_DIV(M/2, 64)·64, nsg, 1} | {((M/2 + 63)/64)·64, 4, 1} | yes |
| enable | opt-in `GGML_OPENCL_Q40_MC3`, N 2 to 4, M < 32768, not under the Q4_0 binary kernels | always (it is a probe) | n/a |

The 1-column reference matches as well. The probe uses the general kernel, while ggml at 11008 × 4096 would pick the
specialised `kernel_gemv_noshuffle_q4_0_f32_11008_1_4096`. R74's *device* cold rates for the two are equal
(895.6 µs and 894.4 µs per gemv), so the choice does not move cg.

Upstream's own comment on the dispatch says "Small-M mc3 verify is occupancy/latency-bound". That points at the same
kind of cause as §3.

## 3. What the *device* lines say (x4m, r36, GPU at 840 MHz throughout, ABBA means)

| n_cols | kernel | dev µs | step from previous |
|---|---|---|---|
| 1 | general | 889.9 | – |
| 2 | mc3 | 2,951.0 | **+2,061** |
| 3 | mc3 | 3,481.9 | +531 |
| 4 | mc3 | 4,029.7 | +548 |

- The third and fourth columns each add about **540 µs, 0.61 of a whole 1-column gemv**. The weights are loaded once,
  but the kernel redoes the dequantisation for every column. The `MC_COL_Q40` macro expands both 32-step
  `dequantizeBlockAccum_ns_sgbroadcast_1_*` macros per column: mask, shift, −8, × scale, and on this phone
  (`vsgb=0`) one scalar `sub_group_broadcast` per element. So most of a column's ALU work is paid again for each
  column.
- Extrapolating the line back to one column gives an mc3 cost of about **2,410 µs, 2.7 times the general kernel for
  the same single column** (*estimated*). That fixed step is not arithmetic. The likely causes are resources, and
  none of them is checked on the phone:
  - **Local memory:** `__local float8 reduceLM[SIMDGROUP_WIDTH * 8]` is 16 KB per work-group, sized for nsg 8 even at
    nsg 4. The general kernel uses `float2 [64 × 3]`, 1.5 KB. Fewer resident work-groups means less latency hiding
    on a memory-streaming kernel.
  - **Registers:** four float2 accumulators, `regA_hi` + `regA_lo` (8 uints, all loaded before any use) and a float8
    `acc`. All four columns stay live because `n_cols` is a runtime value. The general kernel holds one float2 and 4
    uints.
  - **K stride:** `nsg = get_local_size(1)`, passed through `MATERIALIZE_WG` on A6x, so the K loop has a runtime
    stride. The general kernel's stride is the constant 4.

## 4. PoCL host micro-timing (*host*, relative only)

`tom_mc3t.cpp`: a standalone OpenCL program with upstream's kernel file verbatim and upstream's dispatch (sizes and
arguments as in §2). It uses random weights in the noshuffle layout and checks every column against float64 and,
bit for bit, against the 1-column kernel. Docker image `vollie-ocl` (PoCL 5.0, arm64), `--cpus=4`. It uses x3's host
shim (PoCL has no `cl_khr_fp16`: scales read with `vload_half2`) and `-cl-std=CL1.2`. Median of 21 profiled launches,
11008 × 4096, two runs. Variant `var_mc_fixed2` (`tom_mc3_var.cl`, probe-only, not upstream) has compile-time 2
columns, a constant stride, and 4 KB of local memory.

| kernel | n | max rel err vs float64 | bits vs 1-col kernel | µs (run 1 / run 2) | ratio to n 1 |
|---|---|---|---|---|---|
| general | 1 | 3.6e-6 | – | 6,412 / 6,732 | 1 |
| mc3 | 2 | 3.6e-6 | 0 differ | 10,709 / 10,877 | 1.67 / 1.62 |
| mc3 | 3 | 3.6e-6 | 0 differ | 13,885 / 14,471 | 2.17 / 2.15 |
| mc3 | 4 | 3.6e-6 | 0 differ | 20,645 / 17,766 | 3.2 / 2.6 (noisy) |
| var_mc_fixed2 | 2 | 3.6e-6 | 0 differ | 11,537 / 10,671 | 1.80 / 1.59 |

- **Correctness:** upstream's claim holds on PoCL. Every mc3 column is bit-identical to the 1-column kernel on the same
  column, at n 2, 3 and 4.
- On a CPU device, where there is no occupancy to lose, the extra column costs 0.6 to 0.7 of a column. That is the
  same linear term as on the phone. **The phone's +1,500 µs step at n 2 is absent on the host.** That fits a GPU
  resource effect (§3) and not the arithmetic or the dispatch. The host cannot say which resource it is, and the
  variant cannot be tested here (PoCL reports 0 bytes of private and local memory for every kernel).

## 5. What would settle it (a list, not a device request)

- **Free on any later kit:** print `CL_KERNEL_LOCAL_MEM_SIZE`, `CL_KERNEL_PRIVATE_MEM_SIZE` and
  `CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE` for the general kernel and `_mc3`. Upstream itself prints the same
  fields for its q4_K / q6_K mc3 in a `[MC3-PROBE]` line.
- **One cell:** `var_mc_fixed2` beside `_mc3` and the general kernel at n 2, in x4m's ABBA form. If it lands near
  890 + 540 ≈ 1,430 µs, cg ≈ 1.6 (*estimated*), the step was resources.
- Even then the linear term remains. A two-column kernel that dequantises each weight once into registers and only
  multiplies per column would need measuring before any cg below about 1.6 is priced. On this phone the scalar
  broadcast (`vsgb=0`) is per column and per element as well. The centre of 1.25 used before R75 has nothing measured
  under it.
