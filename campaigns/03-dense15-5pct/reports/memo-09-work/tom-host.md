# memo-09 work: Tom, host work after R72 (GPU lane build proof and x3 probe design, placement guard 0004, four-way thread label)

Date: 2026-11-22. Author: Tom (`claude-opus-5-5`). Host: the owner's laptop, Apple M4 Pro, with Docker's arm64 Linux VM
for the Linux path. Hand-off from the lead after [R72](../R72-result.md) (§2 conditions, §7 GPU lane, §11 joint reading).

Labels: every number here is *host* (laptop, relative only) or *estimated* (arithmetic). No number in this file was
measured on the phone, and nothing here ran on the phone. Nothing is committed; no device data or pre-registration
file was touched.

## 0. Summary

| task | built | not built / open |
|---|---|---|
| 1. GPU lane | Upstream `ggml-opencl` at our base `620d2225c` cross-compiles for arm64 Android (NDK r28, API 28) as a `GGML_BACKEND_DL` module. It runs against OpenCL 2.0 (35 imports, none newer than 1.2) and links against the kit r33 `libggml-base.so` (27 of 27 ggml imports present). This works stock, and also with campaign 1's compile-retry patch. The Q4_0 decode kernels pass a clang OpenCL C 2.0 compile. A host emulation confirms the weight layout. No newer upstream is needed: the Q4_0 gemv and its conversion kernel are byte-identical at upstream master `e300919cc` (2026-11-22). | Not run anywhere. The Adreno E031.38 compiler is the open risk, and only the phone can test it. The x3 probe is a design only (§1.5), with no APK. |
| 2. Placement guard | `patches/llama.cpp-c3/0004-pin-guard.patch` (139 lines; `VOLLIE_PIN_GUARD`, `VOLLIE_PIN_WARM`). Off by default. Bit-identical on the macOS host at `-b 512`, `-ub 32` and `-ub 1`, with the knob unset and set. Compiles clean for Android (`-Wall -Wextra`, 0 warnings in `ggml-cpu.c`). Also bit-identical on arm64 Linux (Docker), with the guard running: §2.4. The recovery path (thread off its mask → back on it) cannot be provoked on the host; only the failure path was exercised. |
| 3. Four-way label | `tools/c3_probe_parse.py`: labels M / U / P / C for every thread of the l1 to l4 cells, a summary table, and an L3 per-thread cross-check. `--selftest` passes, including checks against the R72 device file. | No affinity read-back exists in r33's lines, so M uses the sched_getcpu checks only. |

## 1. GPU lane

### 1.1 What the upstream source at `620d2225c` contains

- **The Adreno Q4_0 decode gemv is in our base.** Under `GGML_OPENCL_USE_ADRENO_KERNELS` (CMake default ON), a Q4_0
  weight with `ne0, ne1 ≥ 512` gets three things:
  - it is converted by `kernel_convert_block_q4_0_noshuffle` (cvt.cl);
  - it is transposed as 16-bit words (`transpose_2d_as_16b`);
  - at one token it runs `kernel_gemv_noshuffle_q4_0_f32`.
  - The thresholds are `ne0, ne1 ≥ 128` on compilers older than E031.38.11. Ours is E031.38.01.08 (*device*, R72
    x2info), so the lower thresholds apply.
- **Kernel files:**
  - `gemv_noshuffle_q4_0_f32.cl`, sha256 `636126fd…cdf65`;
  - `gemv_noshuffle_q4_0_f32_spec.cl`, sha256 `3f0b490a…00711`. This is the same source specialised by `-D` for
    4096×4096, 4096×11008, 11008×4096 and 32000×4096.
  - These are exactly the three subjects' shapes. Yi's k and v (512×4096) take the general kernel.
- **A6x fixes are already in the base:**
  - `64df9183f` (#30176): A6X detection includes the 660, plus `-DGGML_CL_A6X_CONSTFOLD_FIX`;
  - `6503355df` (#26476): the q6_K flat fix for E031.38-class compilers;
  - `9166ceaf1` (#30421): an OOB-tail fix for the noshuffle gemvs.
- **Build options upstream would use on our phone:**
  - `-cl-std=CL<c_version> -cl-mad-enable -DSIMDGROUP_WIDTH=64 -DGGML_CL_A6X_CONSTFOLD_FIX`.
  - No `-DVECTOR_SUB_GROUP_BROADCAST`: that needs E031 ≥ 47.
  - The kernel asks for `qcom_reqd_sub_group_size("half")` (64 lanes) and a work-group of 64 × 4.
- **Newer upstream?** Not needed for this kernel.
  - The only commit on `gemv_noshuffle_q4_0_f32.cl` since the base date is `9166ceaf1`, which is in the base.
  - The file and `transpose.cl` are byte-identical to master `e300919cc704` (2026-11-22T08:08Z).
  - The Q4_0 noshuffle convert function in `cvt.cl` is identical too (the file differs elsewhere).
  - The later OpenCL commits (#30947 to #32686) add Adreno A7/A8/X2 binary kernels, coopmat for the 850, FA, and new
    types. None targets A6x.
  - Source: GitHub REST API commit list for `ggml/src/ggml-opencl`, read 2026-11-22.

### 1.2 Cross-compile proof (*host*)

The script is the scratchpad's `build_cl.sh`. It is the same recipe as campaign 1's `vollie/tool/build_opencl.sh`
(sibling app, `lab/app-c3/tool/build_opencl.sh` here):

- Khronos OpenCL-Headers `82f60316` and OpenCL-ICD-Loader `d0be526a`, the loader built only as a link stub (the phone's
  own `libOpenCL.so` is used at run time);
- `-DGGML_OPENCL=ON -DGGML_OPENCL_TARGET_VERSION=200 -DGGML_BACKEND_DL=ON -DBUILD_SHARED_LIBS=ON`;
- NDK 28.2.13676358, `android-28`, flexible page sizes.

| build | result | `libggml-opencl.so` (unstripped) |
|---|---|---|
| stock `620d2225c` | builds; 1 upstream `#pragma message` warning | 14,291,584 B, sha256 `2d6b0193…54b84` |
| + campaign 1 `opencl-compile-retry.patch` (sha256 `d9b8fbda…a5d00`) | applies and builds | 14,085,072 B, sha256 `88459845…89225` |

- **Loadable next to the CPU module:**
  - `NEEDED`: libggml-base.so, libOpenCL.so, libm, libdl, libc.
  - The 35 OpenCL imports are all OPENCL_1.0 to 1.2 symbols. `clCreateBufferWithProperties`, the 3.0 import that broke
    llamadart's prebuilt module on the 660 (campaign 1), is absent.
  - All 27 ggml imports are defined by the kit r33 `libggml-base.so`, taken from the r33 APK.
  - Campaign 1's app loaded a module built by the same recipe (same `NEEDED libOpenCL.so`, resolved to the vendor
    library) on this phone, and that is how its OpenCL tok/s were measured. So the link-time choice is known to load;
    the kernels are the unknown.
  - llamadart already knows the module: `libggml-opencl.so` is in its backend list, `_tryLoadBackendModuleIfBundled('opencl')`.
  - The research APK excludes it since kit r27 (`build.gradle`: `excludes += ['**/libggml-opencl.so', '**/libvollie_clc.so']`).
    So an R74 kit that wants the backend must drop that exclude. The probe below needs only `libvollie_clc.so` back.
- **Kernel compile (*host*, clang 19 from the NDK, `-x cl -cl-std=CL2.0 --target=spir64 -emit-llvm`, cl_khr_subgroups
  and cl_khr_fp16 on):**
  - `gemv_noshuffle_q4_0_f32.cl` compiles under each of `{}`, `-DVECTOR_SUB_GROUP_BROADCAST` and `-DGGML_CL_A6X_CONSTFOLD_FIX`;
  - `_spec.cl` compiles with `-DLINE_STRIDE_A=2048 -DBLOCK_STRIDE_A=16384` and the same three.
  - **Limit:** clang does not know `cl_qcom_reqd_sub_group_size`, so the `ADRENO_GPU` branch (the subgroup-size
    attribute) is not compiled. This proves the C, not the Adreno compiler. Campaign 1 saw E031.38 segfault on some
    newer kernels (`tool/clc.c`), and R72's `x2r` retry got err −6 (*device*). Whether this kernel builds on
    E031.38.01.08 is the phone's question.
- **Layout check (*host*, `layout_check.py` in the scratchpad, numpy float64):** random Q4_0 rows at M 256, K 512 go
  through upstream's noshuffle convert and 16-bit transpose. The kernel's index math is then emulated (texel
  `gid + k·4M + j·M/2`; low and high ushort = rows 2gid and 2gid+1; nibble n of ushort j ↔ element 4j+n; scales `half2`
  at `gid + k·M/2`). Max relative error against the plain Q4_0 dot is 8.6e-17. The CPU-side layout in §1.5 is the
  one the kernel reads.

### 1.3 What campaign 1 already found (repository record; *device*)

- OpenCL on the Adreno 660 gave 0.7 to 2.9 tok/s, always below the CPU (`reports/03_final.md`).
- That run used `620d2225c` built for CL 2.0 with the retry patch. The patch skips the mrow kernels and other kernels
  on E031 < 39 (memo-01-feasibility.md, GPU paragraph; memo-03-work/ebony-draft.md §1.1).
- No model is attached to those numbers except Qwen3.5-4B.
- Whether the Q4_0 noshuffle gemv itself built and ran then is not recorded. The x3 probe answers that directly,
  without the rest of the backend.

### 1.4 Numerics

- The kernel computes `((q & 0xF) − 8) * scale * y`. In OpenCL C the int × half product is formed in **half**, then
  multiplied by the float activation. The activations stay f32 (not Q8_0 as on the CPU).
- So a GPU lane gives different sums from the CPU's Q4_0 × Q8_0 path. It is a **declared numerics change** and needs
  the +5 % PPL budget at `-b 512` and `-ub 1`.
- The host cannot run the Adreno kernels. The perplexity check would need a C emulation of the kernel's arithmetic
  (half rounding of the (q−8)·d product, f32 accumulation in the kernel's order) inside the CPU backend, behind a knob.
  That is later work, after x3 says the lane is worth it.

### 1.5 x3 probe design (for an R74 kit; not built)

A new probe block `x3` in `src/llama-vollie-probe.cpp` (a later probe patch, e.g. `0005-x3-gpu-gemv.patch`). It is off
unless `VOLLIE_PROBE` is set, and nothing outside the file calls it.

1. **Loader.** Reuse X2's `dlopen` struct. Add `CreateImage`, `CreateProgramWithBinary`, `GetEventProfilingInfo`,
   `GetKernelWorkGroupInfo`, `WaitForEvents` and `ReleaseEvent`, and create the queue with
   `CL_QUEUE_PROFILING_ENABLE`. All are ≤ 1.2.
2. **Compile in a child process.** Write the verbatim upstream kernel source (the two files above, with their sha256
   printed) to the probe dir. Run `libvollie_clc.so <src> <bin> <opts>`, campaign 1's `tool/clc.c` (exit 0 ok, 3 only
   with `-cl-opt-disable`, 1 failed, signal = compiler crash), with upstream's options (§1.1). Then load the binary
   with `clCreateProgramWithBinary`.
   - A compiler crash costs the child, not the app.
   - Lines: `x3 build kern=general|s4096x4096|s4096x11008|s11008x4096 rc=<exit or -signal> how=opt|noopt|fail s=<secs>`.
   - The APK must ship `libvollie_clc.so` again (one gradle exclude removed), with `useLegacyPackaging` so it can be
     executed (campaign 1).
3. **Weights.** Random Q4_0 rows (finite fp16 scales, as X2's fill) made on the CPU. They are converted on the CPU with
   the layout checked in §1.2 (noshuffle bytes; q as `[K/4][M]` ushorts, d as `[K/32][M]` halves). They are uploaded
   once with `CL_MEM_COPY_HOST_PTR`, and q is wrapped as an `IMAGE1D_BUFFER` (`CL_R`, `CL_UNSIGNED_INT32`, width M·K/8).
   The activations x are f32 as an `IMAGE1D_BUFFER` `CL_RGBA`/`CL_FLOAT` (width K/4).
   Dispatch: global `{CEIL(M/2, 64)·64, 4}`, local `{64, 4}`.
   - Check first: `CL_DEVICE_MAX_MEM_ALLOC_SIZE` and `CL_DEVICE_IMAGE_MAX_BUFFER_SIZE` against the largest image
     (11008·4096/8 = 5.6 M texels).
   - Check also that the kernel's max work-group size is ≥ 256. Print it.
4. **Identity line.** One gemv per shape against a CPU float64 reference:
   `x3 identity kern=… max_rel_err=… rows_bad=…`. Not exact, because of the fp16 products (§1.4). The pass for
   "kernel works" is max_rel_err < 1e-2. It is a functional check, not a numerics gate.
5. **Cells, GPU alone.** For each kern and each regime (hot = one matrix; cold = a ring of copies totalling about
   512 MB, e.g. 20 × 25.4 MB for 11008×4096):
   - 1 s warm, then 2 s measured, `clFinish` per batch of 8 enqueues;
   - print `gbs` = (q + d bytes) × gemvs ÷ wall, and `dev_gbs` from the profiling events;
   - print the GPU clock (`/sys/class/kgsl/kgsl-3d0/gpuclk` and `max_gpuclk`, if readable, else `e<errno>`) and the CPU
     policies' clocks, as the other cells do.
   - Plus a **layer cell** per subject: the 7 matrices of one layer of Yi / Llama-2 / DeepSeek (Yi: q, o 4096², k, v
     512×4096, gate, up 11008×4096, down 4096×11008), in the graph's order, over a 4-layer ring. Print `ms_per_layer`
     and `body_ms` = ms × n_layer.
     - Layer bytes (*estimated*, Q4_0 18 B per 32 weights): Llama-2 and DeepSeek 113.8 MB; Yi 97.3 MB.
6. **Cells, together.** ABBA within the block: CPU alone, GPU alone, both, both, GPU alone, CPU alone.
   - CPU side: the ggml Q4_0 gemv cell of `l23` on 4-7 (the adopted set), over its own cold arena.
   - GPU side: the cold 11008×4096 cell.
   - The thread that feeds the GPU queue is pinned to cpus 0-3, so the big cores stay the CPU arm's.
   - Print `cpu_gbs gpu_gbs sum_gbs`, the CPU thread labels (thr field as l23, so the parser's M/U/P/C applies), and
     `cpu_alone_gbs gpu_alone_gbs` from the alone arms.
   - The probe's first pin of the CPU workers has the R72 placement issue. It needs its own guard, or the cell is read
     with the labels. Placement guard 0004 is for the runtime pool, not the probe's `cell2`.
7. **Split cell** (the lever's own shape). One 11008×4096 matrix: rows [0, (1−s)M) on the CPU (4-7), rows [(1−s)M, M)
   on the GPU, s ∈ {0.25, 0.35, 0.50}, both started together. Time until both are done; print `t_split ÷ t_cpu_alone`.
   This is the number the lane lives on. Fence and launch costs are inside it.
8. **Size.** About 600 lines (*estimated*), and about 90 s on the phone (*estimated*). Memory is about 0.6 GB GPU and
   0.5 GB CPU arena. This is under R72's anon34 headroom, but the block must run alone, not beside a loaded model.
9. **Kill lines** (R72 §11, to be pre-registered by the lead): `x3 build` fails for the general kernel; or GPU alone,
   cold, 11008×4096 under 7 GB/s; or `sum_gbs` under 1.15 × `cpu_alone_gbs`.

## 2. Placement guard: `patches/llama.cpp-c3/0004-pin-guard.patch`

### 2.1 Design

- **Where.** In the c2 pool's per-thread graph start, `ggml_c2_thr_begin` (c1-olmoe 0010). This is the only place the
  app's runtime pins workers (`VOLLIE_POOL=1` with `VOLLIE_CPUMASK`; the app passes no ggml threadpool).
- **Already there.** The existing code already compares `sched_getaffinity` with the wanted mask and re-sets it when it
  differs. It also counts `sched_getcpu` at graph starts. R72's case is different: the affinity reads back as asked
  (CPU ÷ wall 0.96 to 1.00), but the thread runs on cpus 0-3.
- **New, with `VOLLIE_PIN_GUARD=<us>` (> 0):**
  1. After the mask check, `cpu = sched_getcpu()`. If that cpu is in the mask, nothing more happens.
  2. If it is not: re-pin in two steps. First set mask | current cpu, then the mask. The first step is a different
     mask, so the second is a real change and the kernel migrates the thread at once; Linux skips a set equal to the
     current mask.
  3. Then spin and re-check every 50 µs, re-pinning, until the thread is on a wanted cpu or `<us>` has passed.
  4. `VOLLIE_PIN_WARM=<us>` (default 0): after a recovery, spin that long on the new cpu. This gives the governor and
     core control load to see.
  5. At graph end, `sched_getcpu()` is read again. A thread off its mask then is counted (moved during the graph), not
     fixed.
- **Log.** With `GGML_C2_IO=1`, per prompt, a line
  `c2-pin run=… guard=… warm=… pool=… w<i>=<checked>:<off at start>:<recovered>:<failed>:<us in guard>:<off at end>`.
  The `c2-pool` histogram keeps the cpu seen before the guard, so it means the same with the guard on or off.
- **Off.** Unset or 0: nothing runs, no extra syscall. Placement only: same nth, same chunking, same kernels. So it is
  bit-identical by construction. Linux only; elsewhere the knob is parsed and ignored.
- **Known ceiling.** If core control keeps cpu 7 isolated, no affinity call can put the thread there. The guard then
  burns `<us>` per graph per thread and records `failed`. The R74 decode kit should run it at a small budget, e.g.
  500 µs, and read the counters before raising it. The hypothesis that a short warm-up makes core control unisolate the
  X1 is the open device question (R72 §2 hypothesis, §11 X1 placement).

### 2.2 Applies

The full series (`patches/llama.cpp/0*`, `llama.cpp-c1-olmoe/0*`, `llama.cpp-c3/0001` to `0004`) applies in order to a
clean `git archive 620d2225c`. The result equals the tested tree. The tree without 0004 equals `lab/llama.cpp-c3r32`.

### 2.3 Identity, macOS (*host*)

Llama-2-7B Q4_0, `llama-perplexity -c 512 -t 8`, running values. The reference is `lab/llama.cpp-c3r32/build`, the
series without 0004, same CMake options.

| config | reference | 0004, knobs unset | 0004, `VOLLIE_POOL=1 VOLLIE_CPUMASK=0xff VOLLIE_PIN_GUARD=2000 VOLLIE_PIN_WARM=200 GGML_C2_IO=1` |
|---|---|---|---|
| `-b 512`, 4 chunks | 5.8104, 6.3386, 7.0099, 7.9916 | identical | identical |
| `-b 512 -ub 32`, 4 chunks | 5.7851, 6.3350, 7.0107, 7.9935 | identical | identical |
| `-b 512 -ub 1`, 2 chunks | 5.7791, 6.3466 | identical | identical |

On macOS the guard is compiled out, because it needs `sched_getcpu`. This run also had no `VOLLIE_NO_ATTACH=1`, so the
c2 pool was not used. It shows only that the patch leaves the build's numerics alone. The Linux run below exercises
the code.

### 2.4 Identity and function, Linux (*host*, Docker arm64 VM, `--cpuset-cpus=0-3`, `-t 4`)

Llama-2-7B Q4_0, `-c 512 -t 4`, running values. Builds were made inside the container from the same trees:
`ref` is the series without 0004, `0004` the series with it. The guard arms set `VOLLIE_NO_ATTACH=1`. That is
c1-olmoe 0010's host knob: without it `common` attaches its own threadpool and the c2 pool, and so the guard, is never
reached. The first attempt showed this (`pool_graphs=0/0`, `c2-pin w*=0:…`) and was discarded.

| config | ref | 0004 unset | 0004 guard, masks inside the cpuset (`0x1,0x2,0x4,0x8`) | 0004 guard, masks outside the cpuset (`0x10,…,0x80`) |
|---|---|---|---|---|
| `-b 512`, 4 chunks | 5.8018, 6.3506, 7.0196, 7.9981 | identical | identical | identical |
| `-b 512 -ub 32`, 4 chunks | 5.7751, 6.3435, 7.0126, 7.9950 | identical | identical | identical |
| `-b 512 -ub 1`, 2 chunks | 5.7761, 6.3327 | identical | identical | identical |

The guard arms used `VOLLIE_PIN_GUARD=2000 VOLLIE_PIN_WARM=100 GGML_C2_IO=1`. The Linux values differ from macOS
because the builds differ; each arm is compared with its own platform's reference.

Function, from the `c2-pool` and `c2-pin` lines (*host*):
- **Masks inside the cpuset.** All graphs ran on the pool (`pool_graphs=65/0` at `-ub 32`). Each worker sat on its own
  cpu (`c0=65` … `c3=65`). The guard checked every graph start and found nothing to fix:
  `w0=65:0:0:0:0:0` for every thread.
- **Masks outside the cpuset** (the failure path: `sched_setaffinity` fails, the thread stays on cpus 0-3).
  - At `-ub 32`: every graph start was off-mask (`65:65`), with 0 recovered, 65 failed, and about 130.3 ms in the guard
    per thread over 65 graphs, i.e. the 2,000 µs budget each time. Every graph end was also off-mask (65).
  - At `-b 512`: `5:5:0:5:≈10.1 ms:5`.
  - The PPL values are unchanged.
- **`-ub 1`:** no c2 lines. GGML_C2_IO opens a prompt only at a multi-token graph, so the counters stay off, but the
  guard itself runs on every pool graph. Its identity is in the table.
- **Not shown on the host:** the recovery path (`recovered` > 0). The VM always places a thread on an allowed cpu when
  the mask is legal, so it cannot reproduce "affinity as asked, thread elsewhere". That is the phone's case.

### 2.5 Android compile

The CPU module variant `ggml-cpu-android_armv8.2_2` was built with the series plus 0004 (NDK r28, KleidiAI on as in
`android/build_cpu.sh`). `ggml-cpu.c` compiles with `-Wall -Wextra -Wshadow -Wpedantic …` and gives 0 warnings. The
strings `VOLLIE_PIN_GUARD`, `VOLLIE_PIN_WARM` and the `c2-pin` format are in the .so. No APK was built.

## 3. Four-way per-thread label in `tools/c3_probe_parse.py`

- **Labels** (Ebony's, R72 §11). One letter per condition met; the first is the label.
  - **M** misplaced: on its pinned cpu in < 0.9 of the `sched_getcpu` checks. The probe prints no affinity read-back,
    so that half of Ebony's rule is not available in r33 lines.
  - **U** under-clocked: the pinned cpu's policy sampled below 0.9 × hardware maximum. It uses the lowest `mhz_range`
    sample and `cpuinfo_max_freq` from `max_mhz` (fallback 1804 / 2419 / 2841). A lowered `scaling_max_freq` therefore
    counts.
  - **P** pre-empted: involuntary switches > 50 in the cell, or CPU ÷ wall < 0.9.
  - **C** compute: none of these.
  - The thresholds are parser defaults (`ON_MIN`, `CLK_MIN`, `CW_MIN`, `NIV_HI`), not pre-registered. In R72 the
    involuntary switches run p50 1, p99 32, max 128 per thread-cell (*device*, 380 thread-cells).
- **Output.**
  - A `labels` column (e.g. `7:MU,4:C,5:C`) in the l1, l2, l3 and L4-arm tables.
  - A count table by tag and pinned cpu.
  - An L3 table: each thread's G weights/s against its cluster's reference, which is the first single-thread l3 cell
    labelled C on that cluster.
- **Selftest.**
  - Synthetic cases, including the old 4-field `thr`.
  - The R72 file's facts as written in R72-result.md §2:
    - L1 cpu 7: 84 thread-cells, 67 M and 17 C;
    - L4 cpu 7: 21 of 36 M, with all 18 `7,4,5` arms M on cpu 7;
    - L1 cpu 4: 1 M;
    - every l3 cpu-7 thread M.
  - `python3 -I tools/c3_probe_parse.py --selftest` prints `selftest r72 ok`, then `selftest ok`.
- **R72 read with the labels** (*device* counts; thread-cells):

  | tag | pinned cpu | threads | M | U | P | C |
  |---|---|---|---|---|---|---|
  | l1 | 4 | 84 | 1 | 0 | 0 | 83 |
  | l1 | 7 | 84 | 67 | 0 | 0 | 17 |
  | l2 | 4 / 5 / 6 / 7 | 6 / 6 / 4 / 4 | 0 / 1 / 0 / 4 | 0 | 0 / 1 / 0 / 0 | 6 / 4 / 4 / 0 |
  | l3 | 4 / 5 / 6 / 7 | 4 / 3 / 2 / 3 | 0 / 0 / 1 / 3 | 0 | 0 | 4 / 3 / 1 / 0 |
  | l4 | 4 / 5 / 6 / 7 | 54 / 54 / 36 / 36 | 1 / 1 / 1 / 21 | 42 / 41 / 29 / 15 | 0 | 11 / 12 / 6 / 0 |

- **Two things R72-result.md §2 does not say** (for the lead; R72 is not edited here):
  1. The pos 0 stock arm on `4,5,6` had an A78 thread off-core for **all three** subjects, not only Llama-2 and
     DeepSeek. Yi's had cpu 4 seen on cpu 3 in 324/324 checks (`4:0.990:0:0:324/324@3`).
  2. The l3 cell on `4,5,6` had its cpu-6 thread off-core in 1,420/1,420 checks, at 3.968 G weights/s (0.194 of the
     A78 reference 20.421). The l2 `read` cell on `4,5,6` had cpu 5 off-core in 391/391 checks. These cells are not in
     R72's scored rows, but "A78 threads: 20.5" holds only for the C-labelled ones.
- **U dominates L4:** 127 of 180 L4 thread-cells are under-clocked, i.e. the A78 cap was 1,555 to 2,150 MHz (§2 of
  R72). Only 29 L4 A78 thread-cells are C.

## 4. Files

- `patches/llama.cpp-c3/0004-pin-guard.patch` (new, 139 lines, sha256 `d6c66599…5008b`).
- `patches/llama.cpp-c3/README.md` (0004 section added).
- `tools/c3_probe_parse.py` (labels, selftest).
- Scratch only, not in the repository:
  - `build_cl.sh`, `layout_check.py`, `ident.sh`, `idl/run.sh`;
  - the built modules (`b200/`, `b200r/`) and the identity logs (`id/`, `idl/`).
