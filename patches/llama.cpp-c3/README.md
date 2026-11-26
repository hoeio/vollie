# Campaign 3 patch series for llama.cpp

Base: the campaign 2 phone runtime of kit r28, i.e. `620d2225c` + `../llama.cpp/0001` to `0003` + `../llama.cpp-c1-olmoe/0001`
to `0012`, applied in that order (campaign 3 starts from campaign 2). Every lever is off unless its environment variable
is set.

```sh
for p in patches/llama.cpp/0*.patch patches/llama.cpp-c1-olmoe/0*.patch patches/llama.cpp-c3/0*.patch; do git -C <llama.cpp at 620d2225c> apply "$PWD/$p"; done
```

## 0001-probe.patch: device probes P1, P2, X1, X2 (no numerics change)

New file `src/llama-vollie-probe.cpp` in libllama, built with `-march=armv8.2-a+dotprod+fp16` on arm64 (a stub that returns
-1 elsewhere); `src/CMakeLists.txt` lists it. Design: `campaigns/03-dense15-5pct/reports/memo-02-host-screens.md` §6.

| knob / call | meaning |
|---|---|
| `VOLLIE_PROBE` | unset, empty or `0` = off: `llama_vollie_probe` returns -1 and does nothing |
| `int llama_vollie_probe(const char * block, const char * dir, char * out, size_t cap)` | runs one block (`p2`, `x1`, `x2`, `p1`; `dir` = folder for P1's scratch file) and writes one `c3-probe:` line per cell into `out`; returns the bytes written, -1 off, -2 unknown block |

Nothing in the graph, the kernels or the model path calls the probe, so with the knob unset or set the per-chunk PPL is
the r28 runtime's. Checked (*host*, Mistral-7B-v0.3 Q4_0, `llama-perplexity` running values): identical to the r28
runtime and to stock `620d2225c` at `-b 512` and `-ub 32` (4 chunks) and `-ub 1` (2 chunks), and with `VOLLIE_PROBE=1`
at `-b 512`. The probe itself was run on the host (functional test only; host numbers mean nothing for the phone).
Parser: `tools/c3_probe_parse.py` (`--selftest`).

## 0002-d-rows.patch: D2 rows and D3 header of memo-05 §4 (no numerics change)

Changes only `src/llama-vollie-probe.cpp` (on 0001). Still off unless `VOLLIE_PROBE` is set, and nothing outside the file
calls it. Design: `campaigns/03-dense15-5pct/reports/memo-05-y1-column-kernels.md` §3, §4; kernels as
`tools/c3_colkern_bench.c` (memo-05 work file `tom-host.md` T1, T2, E6).

| call / block | what it adds |
|---|---|
| every block (and the new block `hdr`, header only) | D3 `hdr` lines first: `props` (the readable `ro.lmk.*`, `persist.device_config.lmkd_native.*`, `ro.config.low_ram`, via `__system_property_foreach`; `n=` count), `zram` (`/sys/block/zram0/disksize`, `mm_stat`, legacy `mem_used_total`, `/proc/swaps` line 2), `psi` (`/proc/pressure/memory`, `/proc/pressure/cpu`, one line per `some`/`full`), `mem` (`/proc/meminfo`), `vm` (swappiness, min_free_kbytes, watermark_scale_factor, extra_free_kbytes, the old in-kernel LMK minfree). A source that cannot be read prints `e<errno>` (PSI: `errno=<n>`) |
| `p2` (at its end) and the new block `p2d` (D2 alone) | an identity line on the phone (K1 at keep 1.0 and 0.6, Y1b, Y1b on a row list, each `memcmp` against the stock 4x4 gemv), then one `p2d` line per cell on cpu 4 (one pinned A78), single thread, 1 s warm-up + 2 s measured: inputs 4096 (gate/up) and 14336 (down); hot (one matrix of about 260 to 290 KB, inside the A78's 512 KB L2) and cold (1024-row matrices walked through a 512 MB arena); kernels stock, K1 at TR 128 at keep 1.0 / 0.65 / 0.60 (not hot on 14336: one 128-row tile is 1 MB), Y1b (dense), and Y1b on a sorted half-kept row list (4096 inputs, 2304 B rows; Qwen2.5's are 2016 B) |

Each `p2d` line prints `k` = ns per touched weight ÷ the stock ns per weight of the same shape and regime (the mean of a
stock cell run before and one run after the group, `drift` = after ÷ before), and `t_vs_stock` = k × d, the time against
dense stock. D2 adds about 70 s to `p2` (*host* functional run: 69.5 s).

Checked (*host*, Mistral-7B-v0.3 Q4_0, `llama-perplexity -c 512 -t 8`, knob unset): running values identical to stock
`620d2225c` at `-b 512` and `-ub 32` (4 chunks) and `-ub 1` (2 chunks). The new code compiles with the NDK (API 28,
`-march=armv8.2-a+dotprod+fp16`). Blocks `hdr` and `p2d` ran on the host as a functional test (identity lines all 1;
host timings mean nothing for the phone). Parser: `tools/c3_probe_parse.py` (`--selftest`) reads `hdr` and `p2d`.

## 0003-r72-rows.patch: kit r32 rows for R72 (no numerics change)

Changes only `src/llama-vollie-probe.cpp` (on 0002). Still off unless `VOLLIE_PROBE` is set, and nothing outside the file
calls it. Plan: `campaigns/03-dense15-5pct/reports/memo-08-work/clyde-r72-plan.md`; build notes and limits:
`memo-08-work/tom-r32-build.md`. Kernels K1, K1m, k1m4 as `tools/c3_k1v_bench.c`.

| block | what it runs |
|---|---|
| `a` | R69's p2 read and q4x4 anchors on 4-6, 7+4-5, 4-7 (R69 timing) |
| `l1` | K1 / K1m / k1m4 at keep 1.0, 0.666, 0.65, 0.60, 0.58, 0.50, widths 4096 and 11008, hot and cold, on cpu 4 and cpu 7; stock before and after each group; identity lines first |
| `l23` | 3 and 4 big cores: read, q4x4 and ggml Q4_0 gemv, cold and per-thread hot slices |
| `l4` | one layer of real shapes per subject (Yi-1.5-6B, Llama-2-7B, DeepSeek-7B), stock vs rule (k, v stock; others K1 / K1m), 3 core sets, ABBA, plus a ggml stock layer |
| `l5` | ggml gemv cost of 10 weight types (Q4_0 twice) at 4096 and 11008, hot and cold |
| `l6` | graph barrier cost: chains of 617, 579, 225 nodes on 4 core sets, plus a spin barrier |
| `l7` | Q6_K and Q8_0 output head at vocab 64000, 32000, 102400 |
| `x2` | r29's X2 rows, then a retry built with `-cl-opt-disable` (`x2r`), and an `x2info` line (driver, versions, extensions) |
| `anon34`, `anon42` | 3.4 GB and 4.2 GB of anonymous memory, filled and read back, with faults, swap and RSS |

Every measured cell prints its clock range sampled during the measurement, a ramp check (`ramp_s`, `ramp_ok`, the
policy's max), and per thread CPU time over wall time, involuntary and voluntary context switches and the
`sched_getcpu` off-core count. Checked (*host*, Llama-2-7B Q4_0, `llama-perplexity -c 512 -t 8`, knob unset): running
values identical to stock `620d2225c` at `-b 512` and `-ub 32` (4 chunks) and `-ub 1` (2 chunks). Compiles with the NDK
(`-Wall -Wextra` clean). All blocks ran on the host as a functional test (identity lines all 1; host timings mean nothing
for the phone). Parser: `tools/c3_probe_parse.py` (`--selftest`).

## 0004-pin-guard.patch: placement guard for the c2 pool's workers (no numerics change)

Changes only `ggml/src/ggml-cpu/ggml-cpu.c` (on 0003; it extends c1-olmoe 0010's `ggml_c2_thr_begin`). Off unless set;
placement only (same nth, chunking and kernels). Design and checks: `campaigns/03-dense15-5pct/reports/memo-09-work/tom-host.md` §2.

| knob | meaning |
|---|---|
| `VOLLIE_PIN_GUARD=<us>` | (> 0; Linux; pool graphs, `VOLLIE_POOL=1`, with a `VOLLIE_CPUMASK` entry for the thread) at each graph start, after the mask check: if `sched_getcpu()` is not in the thread's mask (R72: affinity read back as asked, thread on cpus 0-3), re-pin in two steps (mask + current cpu, then the mask, so the kernel migrates it) and spin, re-pinning every 50 us, until on a wanted cpu or `<us>` have passed. At graph end the cpu is read again and an off-mask thread is counted. Unset or 0: nothing runs |
| `VOLLIE_PIN_WARM=<us>` | (default 0) after a recovery, spin that long on the new cpu |
| `GGML_C2_IO=1` | per prompt with the guard set, one `c2-pin` line: `w<i>=<graphs checked>:<off at start>:<recovered>:<failed>:<us in guard>:<off at end>` |

Checked (*host*, Llama-2-7B Q4_0, `llama-perplexity -c 512`, running values): identical to the series without 0004 at
`-b 512` and `-ub 32` (4 chunks) and `-ub 1` (2 chunks), knob unset and set, on macOS (guard compiled out) and on arm64
Linux (Docker, guard running). Compiles for Android with 0 warnings in `ggml-cpu.c`. Not run on the phone.

## 0005-x3-gpu-probe.patch: kit r34 x3 blocks for R74, the GPU lane probe (no numerics change)

Changes only `src/llama-vollie-probe.cpp` (on 0003; it does not touch 0004's file and applies with or without 0004). Still
off unless `VOLLIE_PROBE` is set, and nothing outside the file calls it. Design: `campaigns/03-dense15-5pct/reports/memo-09-work/tom-host.md`
§1.5; build notes, checks and limits: `memo-09-work/tom-r34-build.md`. Kit r34 is built from 0001, 0002, 0003 and 0005
(0004 is not in it).

| block | what it runs |
|---|---|
| every x3 block | `x3 cond` lines first (THP mode, every `/sys/class/devfreq` node, the kgsl clock), then `x3 info` (device, driver, `-cl-std`, A6x, E031) |
| `x3b` | upstream's `gemv_noshuffle_q4_0_f32.cl` and `_spec.cl` (embedded verbatim, sha256 printed) compiled by `libvollie_clc.so` in a child process with upstream's option strings (general; specialised 4096x4096, 4096x11008, 11008x4096), a `-cl-opt-disable` retry when the child crashed or timed out, the compiler log; the binary loaded in-process (create / build / kernel error codes, work-group limits); identity against a float64 reference on six kernel-shape pairs |
| `x3g` | GPU alone, hot and cold (512 MB ring), 4 shapes plus the general kernel on 11008x4096: GB/s, G weights/s, device GB/s; launch + finish round trip at 512, 1024, 2816 and 4096 rows; one layer per subject (Yi, Llama-2, DeepSeek) over a 4-layer ring, finish per layer and per launch |
| `x3c` | ggml Q4_0 gemv on 4-7 (cold, 604 MB) and the GPU (11008x4096 cold) alone and together, ABBA, the GPU fed from a thread on 0-3; row split of 11008x4096 at GPU share 0.25 / 0.35 / 0.50 against the CPU alone, ABBA |
| `x3r` | 0.9 GB of GPU buffers held and read by the GPU while 3 A78 read a 2.6 GB anonymous arena for 15 s, with faults, MemAvailable, swap and RSS per pass |

Timed arms print the CPU policies' clock range against scaling and hardware max, the kgsl GPU clock range, thermal
headroom (AThermal) and per-thread `thr` fields of every thread pinned to one cpu of 4-7 (from `/proc/self/task`, so ggml's
workers get the parser's M / U / P / C labels). Checked (*host*): with the knob unset the probe returns -1; the Android
build of libllama recompiles only `llama-vollie-probe.cpp.o` (559 of 560 objects byte-identical to the 0003 build), and the
CPU module ships byte-identical from kit r33. All four blocks ran on arm64 Linux (Docker) against PoCL as a functional
test; identity passed on all six pairs there with a host-only scale shim (`VOLLIE_X3_HOST_TEST`, never set in a kit).
Parser: `tools/c3_probe_parse.py` (`--selftest`).

## 0006-x3-emu.patch: host emulation of the Adreno Q4_0 gemv's arithmetic (declared numerics change, host check only)

Changes `ggml/src/ggml-cpu/ggml-cpu.c`, `repack.cpp` and `kleidiai/kleidiai.cpp`, adds `ggml/src/ggml-cpu/vollie-x3emu.h` (on 0005; applies with or
without 0004). For memo-12 T-N1, the perplexity check of the GPU lane; not meant for a kit. Results and design:
`campaigns/03-dense15-5pct/reports/memo-12-work/tom-numerics.md`.

| knob | meaning |
|---|---|
| `VOLLIE_X3_EMU` | unset, `0` or any other value = off. `1` or `2`: every `MUL_MAT` with Q4_0 weights and f32 activations, at any batch size, computes each output as upstream's `gemv_noshuffle_q4_0_f32.cl` / `_spec.cl` do: `((q & 0xF) - 8) * d` rounded to half, times the f32 activation (no Q8_0), summed in f32 into four partials (block index mod 4, weights in order inside a block), then `((p0 + p1) + p2) + p3`. `2` = multiply and add rounded separately (reproduces R74's six device identity lines to every printed digit); `1` = fused multiply-add. With the knob set, the repack buffer type declines Q4_0 so the weights stay plain (Q6_K and the other types repack as before) |
| `VOLLIE_X3_EMU_Q6K` | same values, for Q6_K weights (the output head): upstream's `gemv_noshuffle_q6_k_f32.cl`, same work split and reduction; its weight is `((q - 32) * sc) * d` in f32 (the kernel promotes to float before the scales, so nothing is rounded to half). The repack buffer type then declines Q6_K. Independent of `VOLLIE_X3_EMU` |
| (KleidiAI builds) | with either knob set, KleidiAI's buffer type also declines the emulated type, so the emulation runs on a KleidiAI build too (syntax-checked only; no KleidiAI build was run) |

Self-check: `tools/c3_x3emu_check.c` (the NEON tile equals the scalar reference bit for bit, Q4_0 and Q6_K; the Q6_K unpack
follows ggml's `dequantize_row_q6_K` order; the Q4_0 reference replays R74's x3 identity lines). Knob unset: per-chunk PPL identical to the stock build at `-b 512`, `-ub 32` (4 chunks) and `-ub 1`
(2 chunks), Llama-2-7B and Yi-1.5-6B (*host*).

## 0007-x4-join-probe.patch: kit r35 x4 blocks for R75, the GPU join cost (no numerics change)

Changes only `src/llama-vollie-probe.cpp` (on 0005; independent of 0004 and 0006). Still off unless `VOLLIE_PROBE` is set,
and nothing outside the file calls it. Design: campaign 3 memo-12 work (`bonnie-joins.md` §5, `ebony-joins.md` §7.1); line
formats, checks and limits: `campaigns/03-dense15-5pct/reports/memo-12-work/tom-r35-build.md`. Kit r35 is built from 0001,
0002, 0003, 0005 and 0007.

| block | what it runs |
|---|---|
| `x4i` | `CL_DEVICE_EXTENSIONS` in full, SVM capabilities, `CL_DEVICE_RECORDABLE_QUEUE_MAX_SIZE` (qcom), optional entry points; compiles the four upstream kernels (x3b's clc child) |
| `x4a` | launch timeline on the profiling queue: 8 x 2816x4096 then one `clFinish`, one Llama-2 layer (7 launches), the 8 with `clFlush` after each; every event's QUEUED / SUBMIT / START / END; launcher on 0-3 and cpu 4 |
| `x4b` | the x3 round trip at 2816 rows from 0-3, cpu 4, cpu 7 (0.5 s spin to full clock) x wait by `clFinish`, `clFlush` + event status spin, `clFlush` + `clWaitForEvents` |
| `x4f` | launch-ahead behind a user event set 1 ms after the enqueue; round trip with 2 `SetKernelArg` calls instead of 15 |
| `x4c` | stacked gate‖up (22016x4096) and q‖k‖v (12288x4096) on the general kernel: identity and cold rate against 11008x4096 and 4096x4096 |
| `x4d` | recordable queue (`cl_qcom_recordable_queues`): a Llama-2 layer recorded and replayed against sync1, in a child process |
| `x4e` | persistent kernels in child processes: CPU-to-kernel flag round trip, kernel-to-CPU counter (each with 1.x atomics on a mapped `ALLOC_HOST_PTR` buffer and on fine-grain SVM, and with C11 acquire / release atomics on fine-grain SVM), a looped row-claiming gemv |
| `x4m` | upstream's multi-column gemv `kernel_gemv_noshuffle_q4_0_f32_mc3` on 11008x4096 cold at 2, 3, 4 columns against the one-column kernel (identity per column, ABBA); then CPU (ggml Q4_0, 4-7) and GPU together at two columns, at one column, and mixed (CPU one column, GPU mc3 four columns, three if four fails), x3c's ABBA on the profiling queue |
| `x4s` | row split of the fused [gate;up] at batch 1, GPU share 0.20 to 0.60: launch from cpu 4, `clFlush`, CPU rows, event spin; R74's feeder mode at 0.25; then the hand-off of one join (input in, output out) by write/read, map/unmap, `ALLOC_HOST_PTR` map/unmap and kept-mapped `ALLOC_HOST_PTR` |
| `x4r` | residency with 1.5 GB of GPU buffers beside a 2.6 GB anonymous arena (as x3r) |

Small changes to 0005's code, none visible to the x3 blocks' numbers: `Out` can tee every line to a `FILE *` (child
processes only, `VOLLIE_PROBE_TEE=1`); `run_clc`'s spawn/timeout part is `run_proc` (same behaviour); `launch()` takes an
optional wait list; `make_wset()` takes the number of x columns (default 1); `alone()` returns its GB/s; `res()` takes the GPU GB and the line tag; the x3 info line is `info()`; `GClk`
falls back to `devfreq/cur_freq` and `clock_mhz` when `gpuclk` is unreadable and prints `src=`. Riders on every x4 block: `x4 therm` (Android thermal status, kgsl `thermal_pwrlevel`) before and after; `gpu_max_mhz=` in the launcher tail and in `x4 res pass` (an appended field, also in x3r's line); x4i runs a child `x4k_c` that prints `x4 kai` (the KleidiAI primary Q4_0 kernel from ggml's log, the CPU's HWCAP features). `Therm` also resolves `AThermal_getCurrentThermalStatus`. The child process is
`libvollie_x4c.so`, built from `tools/c3_x4child.c` (a loader: it opens libllama.so and calls the probe entry).
Checked (*host*): `clang -fsyntax-only -Wall -Wextra` clean on macOS; Android build without new warnings, libllama only
recompiled; every x4 block ran on arm64 Linux (Docker, PoCL) as a functional test, including the child processes.

## 0008-x4f-fix.patch: kit r36 for R75 amendment 1, the x4f hang fix (no numerics change)

Changes only `src/llama-vollie-probe.cpp` (on 0007). Still off unless `VOLLIE_PROBE` is set. Cause: R75's x4f spun 2.5 h in its
`gate` arm on the phone. The gated command was flushed before its user event was released and not flushed after, and the
event spin had no deadline. Under a host emulation of "a released command is not submitted until the next flush" (an OpenCL
shim on PoCL), r35's x4f hangs and r36's completes. Details: `campaigns/03-dense15-5pct/reports/memo-12-work/tom-r36-build.md`.

| change | where |
|---|---|
| `clFlush` right after `clSetUserEventStatus` | x4f gate arm |
| every event spin (`spin_ev`) gives up after 2 s with err -2001 | x4b / x4f `rt` waits, x4f gate and plain, x4s lead joins, x4s hand-off |
| the same deadline on x4s feeder mode's wait for the feeder | x4s (x3's split is unchanged) |
| rtcell's 100 profiled calls stop at the first deadline | x4b, x4f `rtargs` |
| a deadline stops its x4f mode (gate: one more `clFlush`, `flush_recovered=`); `x4 la` gains `dl= stopped_at= flush_recovered=` | x4f |
| `x4 la nf`: three gated launches released without a flush (2 s), then flushed (2 s): tests the cause on the device | x4f, after the gate/plain arm |
| `x4 dl blk= n=`: the block's deadline count | every parent x4 block |
| parent x4 blocks also write every line to `<dir>/x4-live-<blk>.txt` as it is printed (the app keeps it if the block hangs) | every parent x4 block |
| `x4 pin cpu4= .. cpu7=`: can the thread pin to each big cpu at idle (R75's x4b cpu 7 cells ran on cpu 4) | x4i |

The x3 blocks and every block's cell order and timing windows are unchanged. Kit r36 is built from 0001, 0002, 0003, 0005,
0007, 0008.

## 0009-r37-cells.patch: kit r37 for R76, x4s re-celled (no numerics change)

Changes only `src/llama-vollie-probe.cpp` (on 0008). Still off unless `VOLLIE_PROBE` is set. No new GPU kernel; the arms reuse
x4s's lead and feeder code, `run_arm`, the rings and the general kernel. Details:
`campaigns/03-dense15-5pct/reports/memo-12-work/tom-r37-build.md`.

| block | what |
|---|---|
| `x4sc` | cool pass: triplets cpu 0.5 s \| lead split 1.0 s \| cpu 0.5 s at shares 0.40, 0.45, 0.50, 0.55, then a feeder triplet at 0.25 and the stall triplet (200 us busy stall on the lead before each CPU-alone graph), then 0.55 to 0.40; 6 s rest between triplets; all rings made once |
| `x4sg` | launch + flush of 9856 GPU rows against a host-timed idle gap of 0, 0.1, 0.2, 0.3, 0.4, 0.5, 1, 5 ms, round robin, 5 s measured |
| `x4sh` | heat pass: lead split at 0.45 in 1 s arms without rests, every 8th arm CPU alone, until 10 s after the second CPU cap step or 45 s; 1 Hz sampler on 0-3 (caps, GPU max, thermal_pwrlevel, headroom) |

Per arm: `t_ms`, p10/p50/p90 over the measured window of launch, flush, CPU rows, END to seen, wait; then a canary line (0.1 s hot
sdot on cpu 5, 20 ms register spins on cpu 7 and 4, `dcvsh_freq_limit` of cpu 0/4/7, `isolated`, `thermal_pwrlevel`). Kit r37
is built from 0001, 0002, 0003, 0005, 0007, 0008, 0009.
