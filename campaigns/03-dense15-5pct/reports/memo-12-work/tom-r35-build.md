# Kit r35: the x4 GPU join probe (R75), build notes

Tom, 2026-11-23. Owner, 2026-11-23: a device round for the GPU join cost is allowed ("측정 시도 허락합니다"). Round R75 (R73
stays reserved for the first real decode), kit r35. Design: [bonnie-joins.md](bonnie-joins.md) §2, §5, §7 and
[ebony-joins.md](ebony-joins.md) §2, §7 (cells X4-a to X4-f), with the lead's block list. Added on request after the first
build: block x4m and the x4s shares up to 0.60 ([bonnie-joins-r2.md](bonnie-joins-r2.md) §0 item 6, §2.6); the x4e C11 mode
and the x4s hand-off cell ([ebony-joins-r2.md](ebony-joins-r2.md) E-J8, E-J9); the x4m together cell (Bonnie r3 §4 item 3,
Ebony B.1). This file describes the final build, which has all of them. Not committed, not uploaded. The pre-registration is
the lead's; the APK goes up only after it is pushed.

## Contents

| item | file | sha256 |
|---|---|---|
| runtime patch | `patches/llama.cpp-c3/0007-x4-join-probe.patch` (1,165 lines, only `src/llama-vollie-probe.cpp`, on 0005) | `e4e061ffa413ce029d4f10f3a54c15103438e03c4ccaf4e054224a52b0987fae` |
| app patch | `patches/app-c3/0005-kit-r35-probe.patch` (100 lines, on 0004) | `14c96c38faad7e01959beb793b6087d64b8580920ae157e975943b17c654fad0` |
| child process source | `tools/c3_x4child.c` (new) | `d75cf8ca58252f85fdc55667395e297fd738d8fbb4f92656d339eb2edae38469` |
| parser | `tools/c3_probe_parse.py` (`x4report`, `x4selftest`) | (tracked file) |
| APK | `lab/dist-c3/r35/vollie-c3-probe-r35.apk`, 16,619,478 B, versionCode 35, versionName 1.35.0, package `com.insaneutils.vollie.c2`, debug certificate SHA-256 `c49dde85…1716` (same as r34) | `089ab530c9bc17dca382260f848660fe251a572abf0c8d0363ca3a732e324c62` |
| build record | `lab/dist-c3/r35/BUILD-r35.txt` | `e91a99c2f18ac6a710449057f4ce3fedf553241b33a4049ef9239d1a70615dff` |
| sums | `lab/dist-c3/r35/SHA256SUMS` | `4efafdc1942858eccd6518ca0d3efb2c8d615d1d864f1058e97bbaa48dc22e7b` |

Libraries in the APK, against r34:
- `libggml-cpu-android_armv8.2_2.so` byte-identical to r33 and r34: `46e0735d2824ba6f3a156e219e058a701f6024fdbecbb2ef937b3127115a9694`.
- `libllama.so` rebuilt from c3 0001, 0002, 0003, 0005, 0007: `0db48e404aa2ad3ecdfb6f42bbf55a9db318e019a7cf5e69ca961a4abe7bffd1`.
- `libvollie_x4c.so` is new, 5,184 B, `fb61545acea9990c3bff418682f46970eba24fd92b56af2b256368302779315f` after the gradle strip. The jniLibs source is
  `de6ae452…facfe`, built with NDK 28.2 `aarch64-linux-android28-clang -O2 -s -Wl,-z,max-page-size=16384 tools/c3_x4child.c -ldl`.
- `libapp.so` (Dart) changed.
- Every other library is identical to r34, `libvollie_clc.so` included.

0004-pin-guard and 0006 (the other Tom's host emulation) are not in r35.

## Knob and child process

- `VOLLIE_PROBE` (set by the app's probe button only). Unset: the probe entry returns -1. Nothing else in libllama calls
  it. No numerics change: the probe runs no model and touches no inference path.
- `VOLLIE_PROBE_TEE=1` (set only by the parent for its child): every line is also written to stdout as it is printed.
- x4d and x4e run in a child process. Parent block x4d spawns `libvollie_x4c.so x4d_c <dir>`; x4e spawns `x4e1_c`,
  `x4e2_c`, `x4e3_c` in turn.
  - The child gets `VOLLIE_PROBE=1`, `VOLLIE_PROBE_TEE=1` and `LD_LIBRARY_PATH=<app lib dir>:/vendor/lib64:/system/lib64`.
  - The child loads libllama.so and calls the same entry. Its stdout and stderr go to `<dir>/x4-<blk>.out`.
  - The parent kills the child with SIGKILL at the timeout: x4d_c 30 s, x4e1_c and x4e2_c 9 s (three modes each, up to
    2.3 s per mode), x4e3_c 12 s.
  - The parent copies the child's lines that begin `c3-probe: x4 ` or `c3-probe: x4d `. Any other non-probe output is
    printed as one `x4 childlog` line.
  - Because the child tees, a killed child keeps the lines it printed before the kill.
  - Inside the child, every x4e host wait gives up after 2 s (`PK_DEADLINE`). Every kernel loop has an iteration cap, so a
    flag that is never seen ends the kernel instead of hanging the queue.
- No GPU context is open in the parent while a child runs.

## Run order and blocks (button "Run c3 probe r35 (GPU joins). Cool start: do not run any other probe before this")

Order: x4i, x4a, x4b, x4f, x4c, x4d, x4e, x4m, x4s, x4r. Each block runs after the r16 gate (up to 180 s). The report is saved
before and after every block. Every parent block (not the children) starts with x3's `x3 cond` lines and then `x4 info`,
whose fields are x3 info's. Each block except x4d and x4e then loads the four kernels from x4i's binaries. Those loads
print x3's `x3 load` lines, plus `x3 loadlog` where the driver gives a log. x4i compiles the kernels fresh and prints
`x3 src` and `x3 build` / `x3 buildlog` as x3b did. The parser reports x3-tagged lines from x4 blocks under R75, not R74.

### Common tail of every timed launcher cell (`tl`, `rt`, `la`, `rtargs` lines): written `<tail>` below

`lcpu=<0-3|4|7> lcpu_us= lcpu_over_wall= lplace=<off>/<checks>@<last other cpu> thr= mhz_range= max_mhz= gpu_mhz= hr= ramp_s= ramp_ok=`

- `lcpu_us` is the launcher thread's CPU time (`CLOCK_THREAD_CPUTIME_ID`) per call over the cell's kept calls. `lcpu_over_wall`
  is that CPU time divided by the kept window.
- `lplace` comes from `sched_getcpu()`, checked once per call against the launcher's set.
- `thr` is x3's format (threads pinned to one cpu of 4-7; the parser turns it into M / U / P / C labels).
- `mhz_range` is the CPU policies 0 / 4 / 7, lowest..highest sample, `n=`. A sampler thread on 0-3 samples every 50 ms.
- `gpu_mhz` is `lo..hi,n=<samples>,src=<node>`. The node is the first readable of kgsl `gpuclk`, `devfreq/cur_freq`,
  `clock_mhz`; `e<errno>` if none. R74 read `gpuclk` as EACCES and `devfreq/cur_freq` as readable.
- `ramp_s` and `ramp_ok` describe the launcher's warm-up. On cpu 4 and cpu 7, and on 0-3 in x4b, the launcher spins 0.5 s
  on its own cpu, then until the policy reads `scaling_max_freq` (at most 2 s more). In x4a, the 0-3 launcher does not spin
  (as R74): `ramp_s` about 0, and `ramp_ok` is the at-max check at the start.
- `max_mhz` is `p<policy>:<scaling_max>/<cpuinfo_max>` for 0, 4 and 7.

Cells that run through x3's `run_arm` (x4c `x3 alone`, x4d `x4 rec`, x4m `x4 mc` and `x4 tog`, x4s `x4 split` and
`x4 handoffarm`) end with x3's `astr` tail instead:
`thr= feeder=<cpu/wall>:<cpu mask> mhz= mhz_range= max_mhz= gpu_mhz= gpu_max_mhz= gpu_busy= hr= ramp_s= ramp_ok=`. Here
`gpu_mhz` uses the same fallback and `src=` as above.

### x4i

| line | fields |
|---|---|
| `x4 ext` | `n=<extension count> text=<CL_DEVICE_EXTENSIONS, spaces -> ','>` (in full; cap 12,000 chars) |
| `x4 dev` | `svm_err= svm_caps=0x.. svm_coarse= svm_fine_buffer= svm_fine_system= svm_atomics= queue_props=0x.. host_queue_props=0x.. timer_res_ns= compute_units=` |
| `x4 recq` | present: `present=1 max_size=<CL_DEVICE_RECORDABLE_QUEUE_MAX_SIZE, param 0x41DE> ret_bytes= err= entry=<dlsym\|extaddr\|none>`; absent: `present=0 entry=` |
| `x4 entry` | `flush= wait_for_events= event_info= user_event= map_buffer= svm_alloc= svm_arg= ext_addr= recording=<dlsym\|extaddr\|none>` (1 = the entry point resolved) |
| `x3 src`, `x3 build`, `x3 buildlog`, `x3 load`, `x3 loadlog` | as x3b (fresh compile in the clc child) |
| `x3 identity` | as x3b, general kernel at 2816 x 4096 (a sanity check that the fresh binaries run) |

### x4a launch timeline (profiling queue)

- **Matrices:** an 8-copy ring of 2816 x 4096 (general kernel), and one Llama-2 layer: q, k, v, o 4096x4096; gate, up
  11008x4096; down 4096x11008. Kernels are picked as in x3.
- **Cases:**
  - `batch8`: 8 launches, then one `clFinish`.
  - `layer7`: the 7 layer launches, then one `clFinish`.
  - `batch8flush`: the 8 launches with `clFlush` after each.
- **Launchers:** 0-3 first (no spin), then cpu 4 (spin).
- **Repetitions:** one warm-up (not printed), then 5 printed.

| line | fields |
|---|---|
| `x4 tl setup` | `batch=8x2816x4096 kern=general layer=llama2-7b kerns=<name:kernel,...> queue=profiling` |
| `x4 tl` | `case= lcpu= rep=<0..4> i=<launch index> m= k= enq_us= queued_us= submit_us= start_us= end_us= dev_us= gap_us=` |
| `x4 tl` (failure) | `case= lcpu= rep= ok=0 err=` |
| `x4 tlsum` | `case= lcpu= reps= launches= wall_us= enq_us= q2s_first_us= s2b_first_us= q2s_rest_p50_us= s2b_rest_p50_us= gap_p50_us= gap_mean_us= gap_max_us= dev_sum_us= span_us= busy= err= <tail>` |

Definitions:
- `queued_us` / `submit_us` / `start_us` / `end_us` are the event's stamps minus the first launch's QUEUED, in µs (device
  timer).
- `gap_us` = START(i) − END(i−1); -1 for i = 0.
- `enq_us` is host time per `launch()`: 15 `SetKernelArg` calls and the enqueue (plus `clFlush` in `batch8flush`).
- `wall_us` is host time per repetition, from the first enqueue to the return of `clFinish`.
- `q2s` = SUBMIT − QUEUED. `s2b` = START − SUBMIT.
- `_first` is the median over the repetitions of launch 0. `_rest_p50` is the median over launches 1 and later.
- `gap_max_us` is the largest gap of any repetition.
- `dev_sum_us` is the mean, per repetition, of the sum of START→END.
- `span_us` is the mean END(last) − QUEUED(first).
- `busy` = `dev_sum_us` ÷ `span_us`.
- The tail's `lcpu_us` divides by 6 × launches: the warm-up repetition is included.

X4-a's kill line reads `gap_p50_us` with `lcpu=4`. Bonnie's §5 item 2 reads START(j) − END(j−1) for j ≥ 2; that is the
`gap_us` per event.

### x4b launcher core × wait mode (2816 x 4096, general kernel)

- **Cells:** for each launcher (0-3, 4, 7), each wait mode (`finish`, `spin`, `wfe`), then 0-3 `finish` again as a drift
  anchor (`rep=1`): 10 cells.
- **Spin before each cell:** each cell is preceded by the 0.5 s spin on the launcher's cpu.
- **Timed loop:** 0.8 s of launch + wait calls on the plain queue; calls in the first 0.2 s are not kept.
- **Profiled calls:** then 100 calls on the profiling queue with the same wait mode.
- **Wait modes:**
  - `finish`: `clFinish`, no event (as R74).
  - `spin`: `clFlush`, then a spin on `clGetEventInfo(CL_EVENT_COMMAND_EXECUTION_STATUS)` until `CL_COMPLETE`.
  - `wfe`: `clFlush`, then `clWaitForEvents`. The memo's "callback" variant is `wfe`; no callback is used.

| line | fields |
|---|---|
| `x4 rt` | `rep=<0\|1> wait=<finish\|spin\|wfe> kern=general m=2816 k=4096 calls= wall_us= p50_us= p90_us= enq_us= dev_us= ovh_us= queued_to_start_us= submit_to_start_us= queued_to_submit_us= err= <tail>` |
| `x4 rt` (failure) | `ok=0 reason=<no_kernel\|no_entry>` or `ok=0 err=` |

Definitions:
- `wall_us`, `p50_us` and `p90_us` are over the kept calls.
- `enq_us` is the host time of `launch()`, without the wait.
- `dev_us`, `queued_to_start_us` (START − QUEUED), `submit_to_start_us` (START − SUBMIT) and `queued_to_submit_us`
  (SUBMIT − QUEUED) are means over the 100 profiled calls.
- `ovh_us` = `wall_us` − `dev_us`, as R74.

### x4f launch-ahead and SetKernelArg cut (launcher cpu 4, spun)

| line | fields |
|---|---|
| `x4 la` | `mode=<gate\|plain> n= ahead_us= set_to_done_p50_us= set_to_done_p90_us= dev_us= set_to_start_p50_us= set_to_start_p90_us= queued_to_submit_p50_us= submit_to_start_p50_us= enq_us= err= <tail>` (gate line first) |
| `x4 la` (failure) | `ok=0 reason=<no_kernel\|no_entry>` or `ok=0 err=` |
| `x4 rtargs` | `pos=<0..3> args=<15\|2> wait=spin kern=general m=2816 k=4096` + the `x4 rt` fields from `calls=` to `err=` + `<tail>` |

`la` definitions:
- **Order:** 10 warm-up iterations, then 100 measured, gate and plain alternating, so `n` is about 50 each.
- **`gate` iteration:**
  1. `clCreateUserEvent`.
  2. Enqueue on the profiling queue with that event as the wait list, then `clFlush`.
  3. Spin until 1 ms after the enqueue began, then `clSetUserEventStatus(CL_COMPLETE)` at host time *s*.
  4. Spin on the kernel event until it completes (host time *d*).
- **`plain` iteration:** the same without the user event; *s* is the moment the enqueue and flush return.
- **Fields:**
  - `ahead_us` = enqueue start → *s* (gate only).
  - `set_to_done` = *d* − *s* (host clock).
  - `set_to_start` = `set_to_done` − START→END, per call. It is derived, because device and host clocks are not compared
    directly. This is the X4-f number.
  - `queued_to_submit` and `submit_to_start` are from the profiling stamps. For a gated command, `queued_to_submit`
    includes the 1 ms wait if the driver submits on release.
  - `enq_us` is enqueue + flush host time.
- **Safety:** the user event is always completed once created (status -1 if the enqueue failed), so no command waits forever.

`rtargs`:
- **Order:** ABBA (15, 2, 2, 15), wait = spin.
- **`args=2`:** after one full `launch()`, each call sets only args 0 (weight image) and 1 (scale buffer) and enqueues;
  two matrices of the same shape alternate.
- **`args=15`:** each call is `launch()` on one matrix.

### x4c stacked buffers (general kernel)

| line | fields |
|---|---|
| `x3 identity` | x3's format, `m=22016 k=4096` (gate‖up) and `m=12288 k=4096` (q‖k‖v), general kernel, against float64 |
| `x3 alone` | x3's format, `kern=general regime=cold` (512 MB ring), in the order 11008, 22016, 4096, 12288, 12288, 4096, 22016, 11008 rows |
| `x4 stacksum` | `gate_up_gbs= m11008_gbs= gate_up_vs_11008= qkv_gbs= m4096_gbs= qkv_vs_4096=` (each rate the mean of its two arms; ratios are sums over the two arms; -1 if an arm failed) |

The output buffer of every x4 block holds 22,016 floats.

### x4d recordable queue (child `x4d_c`)

- **Ring:** one Llama-2 layer per entry, 4 entries (28 buffers).
- **Recording:** a queue created with `CL_QUEUE_RECORDABLE_QCOM` (1 << 30). One recording per layer, made with
  `clNewRecordingQCOM`, the 7 `launch()` calls on that queue, then `clEndRecordingQCOM`.
- **Replay:**
  - Each replay is `clEnqueueRecordingQCOM` with empty argument, offset and size arrays (replay as recorded).
  - It is tried on the plain queue first; on error it is retried on the recordable queue.
  - The queue that worked is used.
- **Comparison:**
  - ABBA: sync1 (7 launches + `clFinish`, as R74's x3 layer), replay (+ `clFinish`), replay, sync1.
  - Each arm runs through `run_arm` with the feeder on 0-3, as R74's x3 layer (0.5 s + 1.0 s).
  - Device time per layer comes from one profiled sweep of the 4 layers.

| line | fields |
|---|---|
| `x4d skip` | `reason=<no_extension\|no_entry_points\|setup err=\|queue err=>` |
| `x4 rec setup` | `subj=llama2-7b layers_in_ring=4 layer_mb= new_err= end_err= launch_err= replay_plain_err= replay_recq_err= replay_queue=<plain\|recq> entry= ok=` |
| `x4 rec` | `pos=<0..3> mode=<sync1\|replay> layers= ms_per_layer= gbs= err= <astr tail>` |
| `x4 recsum` | `ms_sync1= ms_replay= dev_ms= ovh_us_per_launch_sync1= ovh_us_per_launch_replay= replay_vs_sync1=` (ovh = (ms − dev_ms) × 1000 ÷ 7; ratio replay ÷ sync1) |
| `x4 child` | `blk=x4d_c rc=<0 ok, -1001 killed at timeout, -n signal, -1000 no executable> s= timeout=<0\|1> lines=<copied>` (+ `reason=no_exe errno=` when the executable is missing) |
| `x4 childlog` | `blk= text=<other output, sanitised, cap 2000>` |

The `cl_qcom_recordable_queues` entry-point signatures come from Qualcomm's extension as I know it, with no header
available. They were not checked on any device. Only empty arrays are passed, so a different array layout cannot matter. A
wrong parameter count would at worst crash the child; the parent then prints `rc=-11` or similar.

### x4e persistent kernels (children `x4e1_c`, `x4e2_c`, `x4e3_c`)

**Setup:**
- One small OpenCL program, compiled in the child with `-cl-std=CL2.0` when the device reports C 2.0.
- Flag memory has three slots: `f[0]` CPU → GPU, `f[16]` GPU → CPU, `f[32]` the kernel's exit code (separate 64 B lines).
- Two modes:
  - `hostptr`: a `CL_MEM_ALLOC_HOST_PTR` buffer, mapped once and kept mapped while the kernel runs (zero copy).
  - `svm`: `clSVMAlloc` with fine-grain buffer (+ SVM atomics if offered). It needs `svm_fine_buffer`, else
    `ok=0 reason=no_fine_grain_buffer_caps_0x..`.
  - `svm_c11` (E-J8): the same fine-grain SVM allocation, with a second program whose flag and counter kernels take
    `__global atomic_int *` and use `atomic_load_explicit(p, memory_order_acquire, memory_scope_all_svm_devices)` and
    `atomic_store_explicit(p, v, memory_order_release, memory_scope_all_svm_devices)`. It is built on its own with
    `-cl-std=CL2.0`, then `-cl-std=CL3.0` if that fails, so that a compiler without C11 atomics costs only this mode.
    Caps, iteration counts and deadlines are those of the other two modes.
  - Ebony's E-J8 text also mentions seq_cst. The kit uses acquire / release, as the request specified; seq_cst would be a
    change of the two macros.
- The CPU side uses acquire / release atomics (`__atomic_load_n` / `__atomic_store_n`) in every mode.
- The GPU side of `hostptr` and `svm` uses `atomic_add(p, 0)` reads and `atomic_xchg` writes (OpenCL 1.x).
- Modes run in the order hostptr, svm, svm_c11 in each of `x4e1_c` (flag) and `x4e2_c` (counter).

| line | fields |
|---|---|
| `x4 pk c11` | `test=<flag\|counter> opt=<-cl-std=CL2.0\|-cl-std=CL3.0, the last option tried> err=<build error, 0 ok> svm_caps=0x.. svm_atomics=<0\|1>` (once per child, before the modes) |
| `x4 pk test=flag` | `mode=<hostptr\|svm\|svm_c11> n=200 seen= timeout_at=<ping that timed out, 0 none> first_us= rtt_p50_us= rtt_p90_us= rtt_min_us= kernel_exit= dev_ms= queued_to_start_us= err= ok=<1 all 200 seen>` |
| `x4 pk test=counter` | `mode= n=50000 spin=16 first_change_us= changes= iv_p50_us= iv_p90_us= step_p50= last_seen= kernel_exit= final= stop_seen_us= dev_ms= visible_during_run=<1 if ≥ 2 changes seen> err=` |
| `x4 pk test=gemv` | `kern=<q4n\|q4p> groups= m=11008 k=4096 copies=6 mb= launches= gbs= err=` |
| `x4 pksum` | `test=gemv q4n_gbs= q4p48_gbs= q4p192_gbs= q4p48_vs_q4n= q4p192_vs_q4n= same_rows=<1 bit-identical rows q4p vs q4n>` |
| failures | `x4 pk test=<flag\|counter> mode= ok=0 reason=<why> err=`; `x4 pk test=<flag\|counter> mode=svm_c11 ok=0 reason=build_c11 opt= err= log=<build log, sanitised, cap 1500>`; `x4 pk test=gemv ok=0 reason=alloc err=`; `x4 pk test=<1\|2\|3> ok=0 reason=build err= log=` |
| `x4 child`, `x4 childlog` | as in x4d, `blk=x4e1_c` / `x4e2_c` / `x4e3_c` |

`flag` (X4-e part 1):
- **Kernel:** a single work-item. For s = 1..200 it waits for `f[0] ≥ s`, then writes `f[16] = s`. The cap is 20 M
  polls in total; at the cap it writes `f[32] = −s` and exits. On normal exit `kernel_exit` = polls + 1.
- **Host:** writes s, spins until `f[16] = s` (2 s deadline), then writes `1 << 30` so a late kernel finishes.
- **Fields:**
  - `first_us` = flush → first answer (includes the kernel's start).
  - `rtt` is for pings 2..200: CPU write → CPU sees the answer.
  - The one-way time is not separable; rtt ÷ 2 is an estimate.

`counter` (X4-e part 2):
- **Kernel:** writes `f[16] = i` for i = 1..50000, with 16 polls of `f[0]` between writes. A negative `f[0]` stops it
  (`kernel_exit = −i`).
- **Host:** samples `f[16]` for up to 0.3 s or until exit, then writes −1. `stop_seen_us` is the time until the kernel's
  exit code appears; -1 if it had already finished.
- **Fields:**
  - `iv` is the host time between observed changes.
  - `step` is the value change per observation.

`gemv` (X4-e part 3):
- **Kernel:** the probe's own simple Q4_0 gemv, not upstream's. Rows are 18-byte blocks, x is float, one work-group of 64
  per row.
- **`q4n`:** one group per row (11,008 groups).
- **`q4p`:** a fixed grid of 48 or 192 groups. Each group claims the next row with `atomic_inc` on a counter, until the
  rows run out.
- **Ring:** 6 copies (145 MB).
- **Timing:** 0.3 s warm, then 1.0 s measured per kernel; `clFinish` after each ring pass.
- **Comparison:** the ratio q4p ÷ q4n isolates the looping and claiming cost. Compare it with the upstream kernel's
  rate through x4c's `x3 alone` 11008 line, not directly.

### x4m multi-column gemv and the together cell

**cg cell** (Bonnie r2 §2.6):
- **Kernel:** upstream's `kernel_gemv_noshuffle_q4_0_f32_mc3`, from the general program x4i built (it is in
  `gemv_noshuffle_q4_0_f32.cl` at 620d2225c), against the one-column general kernel.
- **Arguments:** as ggml's mc3 path: weight image, scales, activations as an `IMAGE1D_BUFFER` RGBA float of K × n ÷ 4 pixels
  (column-major), dst column-major [M × n], `offsetd` 0, K, M, n_cols. Local size {64, nsg, 1}, nsg 4 (8 if M < 4096);
  global {CEIL(M ÷ 2, 64) × 64, nsg, 1}.
- **Matrix:** 11008 x 4096 on x3's 512 MB cold ring (22 copies).
- **Identity:** at n 2, 3, 4, every column against a float64 dot of that column (pass: max relative error < 1e-2 of the
  reference rms and no bad rows).
- **Rates:** ABBA over n = 1, 2, 3, 4, 4, 3, 2, 1. Each arm is `run_arm` (feeder on 0-3, 8 launches + `clFinish` per call,
  0.5 s + 1.0 s), then 0.3 s of the same walk on the profiling queue for device time. An n whose identity failed is skipped.
- **Skip:** if the kernel cannot be made or its work-group limit is below 256, the cell prints why and the together cell
  runs one column only.

**Together cell** (after the cg cell):
- **Arms:** x3c's ABBA (cpu, gpu, both, both, gpu, cpu), first at two columns, then at one column.
- **CPU:** ggml Q4_0 mul_mat with x [4096, n], 64 copies of 4096 x 4096 (576 MB, repacked), `VOLLIE_POOL` on 4-7, 4 threads.
- **GPU:** 11008 x 4096 cold ring (532 MB). mc3 with n_cols 2 at two columns, the general kernel at one. The feeder on 0-3
  runs 8 launches then `clFinish` per call, on the profiling queue (one event per launch, for `gpu_dev_us`). x3c used the
  plain queue.
- **Rates:** `cpu_gbs` and `gpu_gbs` are weight bytes per second. A gemv of n columns reads its weights once, so the rates
  are not per column.
- **If mc3 failed** (no kernel, work-group limit, allocation, or the n = 2 identity), the two-column part prints
  `x4 tog2 ok=0` and the one-column part runs.

| line | fields |
|---|---|
| `x4 mc setup` | `m=11008 k=4096 kern1=general kernmc=kernel_gemv_noshuffle_q4_0_f32_mc3 wg_max= copies= mb= nsg=4` |
| `x4 mcid` | `n_cols=<2..4> col=<0..n-1> err= max_rel_err= rms_ref= rows_bad= nonfinite= ok=` |
| `x4 mc` | `pos=<0..7> n_cols=<1..4> kern=<general\|mc3> m= k= gemvs= gbs= us_per_gemv= dev_us= dev_gbs= err= <astr tail>` |
| `x4 mcsum` | `n_cols=<2..4> dev_us_1= dev_us_n= dev_ratio= wall_us_1= wall_us_n= wall_ratio=` (each the mean of its two arms; ratio n ÷ 1), or `n_cols= ok=0 reason=<identity\|no_rate>` |
| `x4 mc` (skip) | `ok=0 reason=<no_general_program\|create_kernel err=\|wg_max wg_max=\|alloc err=>` |
| `x4 tog setup` | `n_cols=<2\|1> cpu=ggml_q4_0 cpu_buft= cpu_mb= gpu_kern=<mc3\|general> gpu_m=11008 gpu_k=4096 gpu_copies= gpu_mb= set=4,5,6,7 feeder=0-3 queue=profiling backend=` |
| `x4 tog` | `pos=<0..5> n_cols= arm=<cpu\|gpu\|both> set=4,5,6,7 cpu_gbs= gpu_gbs= sum_gbs= cpu_ms_per_graph= gpu_dev_us= gpu_dev_gbs= cpu_calls= gpu_batches= err= <astr tail>` |
| `x4 tog2sum`, `x4 tog1sum` | `cpu_alone_gbs= gpu_alone_gbs= both_cpu_gbs= both_gpu_gbs= sum_gbs= sum_vs_cpu_alone= cpu_loss_gbs= gpu_loss_gbs= cpu_alone_ms= both_cpu_ms= gpu_alone_dev_us= both_gpu_dev_us= err=` |
| `x4 tog2` (skip) | `ok=0 reason=<identity\|create_kernel\|wg_max\|alloc\|no_general_program\|mc_alloc> then=n_cols_1_only` |
| `x4 tog` (failure) | `ok=0 reason=<no_kernel\|no_cpu_backend>`, `ok=0 reason=gpu_alloc err=`, `n_cols= ok=0 reason=cpu_alloc` |

Definitions (together):
- `cpu_ms_per_graph` is the mean wall time of one graph (64 mat-vecs of 4096 x 4096) over the measured window; -1 in gpu
  arms.
- `gpu_dev_us` is the mean END − START per gemv over every launch of the arm, warm-up included; -1 in cpu arms.
  `gpu_dev_gbs` = 11008 x 4096 weight bytes ÷ `gpu_dev_us`.
- Summary fields are means of the two arms of each kind: `cpu_alone` from the cpu arms, `gpu_alone` from the gpu arms,
  `both_*` from the both arms. `sum_vs_cpu_alone` = (both_cpu + both_gpu) ÷ cpu_alone, as x3c.

### x4s split of the fused [gate;up] (22016 x 4096, batch 1)

- **Placement:** CPU rows through ggml (repacked Q4_0, 4-7, `VOLLIE_POOL`). GPU rows run on the general kernel. Rings of 4
  copies, built per arm and freed after it.
- **Arm order (22 arms):** cpu, lead 0.20..0.60 (step 0.05), feeder 0.25, feeder 0.25, lead 0.60..0.20, cpu.
- **Rows:** `gpu_rows` = round(share × 22016 ÷ 128) × 128.
- **`lead` mode:** the arm's main thread is pinned to cpu 4, which is ggml's worker 0.
  1. Launch the GPU rows on the profiling queue, then `clFlush`.
  2. Compute the CPU rows.
  3. Spin on the event.
- **`feeder` mode:** R74's x3 split. A thread on 0-3 spins on a counter, launches on the plain queue and runs `clFinish`.

| line | fields |
|---|---|
| `x4 split setup` | `m=22016 k=4096 batch=1 copies=4 cpu_buft= gpu_kern=general set=4,5,6,7 lead=cpu4_flush_spin_profiling feeder=0-3_finish_plain` |
| `x4 split` | `pos=<0..21> share=<0.00 for cpu> mode=<cpu\|lead\|feeder> gpu_rows= cpu_rows= mats= ms_per_mat= cpu_ms= wait_ms= gpu_ms= gpu_dev_ms= enq_us= gbs= err= <astr tail>` |
| `x4 splitsum` | `share= mode=<lead\|feeder> gpu_rows= arms=<2> t_split_ms= t_cpu_alone_ms= ratio= wait_ms= gpu_ms=` |
| failures | `x4 split error=<no_cpu_backend\|cpu_alloc>` (+ `share=`), `x4 split ok=0 reason=<no_kernel\|no_entry>`, `x4 split ok=0 err= reason=gpu_alloc share=` |

Definitions:
- `ms_per_mat` is the measured window (arm wall ÷ matrices). `cpu_ms`, `wait_ms`, `gpu_ms` and `enq_us` are means over
  every matrix of the arm, warm-up included (as x3 split).
- `lead` mode:
  - `enq_us` = launch + flush host time, on cpu 4, before the CPU rows start.
  - `cpu_ms` = the ggml compute.
  - `wait_ms` = the event spin after it.
  - `gpu_ms` = END − QUEUED of the GPU event (device timer).
  - `gpu_dev_ms` = END − START.
- `feeder` mode:
  - `gpu_ms` is the feeder's launch + `clFinish` wall (as R74).
  - `gpu_dev_ms` and `enq_us` are -1.
- `t_cpu_alone_ms` is the mean of the two cpu arms. `ratio` = `t_split_ms` ÷ `t_cpu_alone_ms`.
- The thread labels apply to the CPU workers (`thr`). In lead mode, cpu 4's thread also carries the launch and the spin.

**Hand-off cell** (E-J9; after the split arms, in the same block):
- **Shape:** one GPU join at lead share 0.30: 6656 x 4096 on the general kernel, a ring of 4 copies.
- **Thread:** the lead thread alone on cpu 4 (r16 ready check on cpu 4). No CPU rows: the split arms above price those.
- **Per join:** put the f32 input (4096 floats, 16 KB) into the GPU input buffer; launch on the profiling queue; `clFlush`;
  spin on the event; take the 6656 output floats (26 KB).
- **Modes:**
  - `none`: fixed input, written once before the arm; no read. The baseline.
  - `write`: blocking `clEnqueueWriteBuffer` in, blocking `clEnqueueReadBuffer` out (plain `CL_MEM_READ_WRITE` buffers).
  - `map`: plain buffers. In: map with `CL_MAP_WRITE_INVALIDATE_REGION`, `memcpy`, unmap. Out: map with `CL_MAP_READ`,
    `memcpy`, unmap.
  - `ahp_map`: the same on `CL_MEM_ALLOC_HOST_PTR` buffers.
  - `ahp_persist`: the `ALLOC_HOST_PTR` buffers are mapped once for the arm, read and write. In and out are plain
    `memcpy` with a full fence. Using a mapped buffer in a kernel is outside OpenCL 1.2's rules; `match=` shows whether
    it held on this driver.
- **Order:** ABBA over the five modes, 10 arms of `run_arm` (0.3 s + 1.0 s).
- **Check:** two inputs alternate. Each read output is compared bitwise with a write-mode reference of that copy and input,
  made before the arms.

| line | fields |
|---|---|
| `x4 handoff setup` | `gpu_rows=6656 k=4096 copies=4 in_bytes=16384 out_bytes=26624 kern=general lead=cpu4_flush_spin_profiling cpu_rows=0` |
| `x4 handoffarm` | `pos=<0..9> mode= joins= joins_per_s= write_us= read_us= wall_us= checked= matched= err= <astr tail>` |
| `x4 handoff` (summary) | `mode=<none\|write\|map\|ahp_map\|ahp_persist> write_us= read_us= wall_us= extra_us= checked= matched= match=<1 all matched, 0 not, -1 none> arms=` |
| failures | `x4 handoff ok=0 reason=<no_entry\|alloc err=\|reference err=>`, `x4 handoff mode= ok=0 reason=error` |

Definitions (hand-off):
- `write_us` is the host time from the start of the join to the launch (the input move). `read_us` is the time from the
  event's completion to the end of the output move. `wall_us` is the whole join. All are means over every join of the arm,
  warm-up included. `joins_per_s` is the measured window.
- `extra_us` = the mode's `wall_us` − `none`'s `wall_us` (means of two arms each). It can be negative by noise; read
  `write_us` + `read_us` beside it.

### x4r residency

x3r's arm with 1.5 GB of GPU buffers: round(1.5e9 ÷ 25.4 MB) = 59 copies of 11008 x 4096. The lines are x3's with the tag
`x4`: `x4 res stage=...`, `x4 res pass=...`, `x4 res sum ...`, `x4 res done err=`; every field is as in x3r. The
per-pass `gpu_mhz` there still reads `gpuclk` only.

### Other lines

- `x4 error=<no_opencl|no_gpu|context_..|queue_..>` and `x4 error=io_alloc err=`: the parser lists them as errors.
- `x4i end s=` … `x4r end s=`: block seconds.

## Checks (all *host*; host timings mean nothing for the phone and are not reported)

- The tree `lab/llama.cpp-c3r35` is a fresh clone of the base at `620d2225c` with patches/llama.cpp, patches/llama.cpp-c1-olmoe
  and c3 0001, 0002, 0003, 0005 applied, then 0007. It is not the other Tom's tree. `tools/c2_build_kit.sh` checked it file
  by file against the series.
- 0001, 0002, 0003, 0005, 0007 applied to an empty tree reproduce the r35 tree's probe file exactly (`cmp`).
- macOS `clang++ -std=c++17 -fsyntax-only -Wall -Wextra -Wcast-qual`: no warning in new code. The two at lines 101 and
  102 are 0001's.
- Android build (NDK 28.2, `android/build_cpu.sh`): no new warnings. Linux gcc (Docker): no new warnings after three
  fixes in my code (misleading indentation, a 40-byte snprintf buffer, a cast).
- **CPU module:**
  - The kit ships r34's `libggml-cpu-android_armv8.2_2.so`, sha256 `46e0735d…9694`, byte-identical (checked in the APK).
  - The ggml sources of the r35 tree equal those of `lab/llama.cpp-c3r32` (r33's source tree): `diff -rq` is empty.
  - A fresh Android build of the CPU module from them still differs from r33's file in 29,096 bytes, in `.text` and
    `.rodata`. Its strings are equal except the tree path. The cause is not traced.
  - It does not affect R75 (the shipped bytes are r33's), but a fresh rebuild is not byte-reproducible.
- **Functional test:** every block (x4i, x4a, x4b, x4f, x4c, x4d, x4e, x4m, x4s, x4r) ran on arm64 Linux in Docker (`--cpus=4`)
  against PoCL.
  - The host build used `VOLLIE_X3_HOST_TEST` (x3's scale shim), and a host copy of `clc.c` with `CL_DEVICE_TYPE_ALL`.
    Neither is in the kit.
  - Every cell printed with err=0.
  - Identity at 2816, 22016 and 12288 rows: ok=1, max relative error about 3e-6 with the shim.
  - x4d printed `x4d skip reason=no_extension` (PoCL has no qcom extension), through the child process.
  - x4e: all three children returned rc=0. Flag and counter ran in all three modes (hostptr, svm, svm_c11) with ok=1 and
    `visible_during_run=1`; the gemv gave `same_rows=1`. PoCL reports OpenCL C 1.2: the C11 program failed with
    `-cl-std=CL2.0` and built with `-cl-std=CL3.0` (`opt=-cl-std=CL3.0 err=0`), so the fallback path is exercised.
  - x4m: identity ok=1 on all 9 columns (n 2, 3, 4); all 8 rate arms and both together parts (12 arms, `x4 tog2sum` and
    `x4 tog1sum`) printed with err=0.
  - x4s: all 22 split arms and the 10 hand-off arms printed with err=0; every hand-off mode with reads matched its
    reference bitwise (`match=1`), `ahp_persist` included (PoCL's memory is the host's, so that says nothing for Adreno).
  - x4r ran 200 passes with err=0.
  - With the probe knob unset, the child got no lines (the entry returned -1).
- `python3 -I tools/c3_probe_parse.py --selftest`: `selftest r72 ok`, `selftest x3 ok`, `selftest x4 ok`, `selftest ok`.
  The x4 selftest has 53 synthetic lines, including svm_c11 (a build failure and the c11 line), x4m mc / mcid / mcsum,
  tog / tog2sum / tog1sum, and handoff / handoffarm. `x4report` also ran on the host outputs of all ten blocks.
- The kill-line inputs printed by `x4report` gained two lines: `together sum_vs_cpu_alone n_cols 2 .., n_cols 1 ..` and
  `hand-off extra_us <mode>:<value>/...`; the X4-e line now lists svm_c11 beside hostptr and svm.
- The APK is versionCode 35, versionName 1.35.0, package `com.insaneutils.vollie.c2`, and has the same signing
  certificate digest as r34 (`apksigner`).

## Phone time (*estimated*)

| block | estimate |
|---|---|
| x4i | 2 to 5 s |
| x4a | 5 to 10 s |
| x4b | 15 to 35 s (10 cells, each a 0.5 to 2.5 s spin + 0.9 s) |
| x4f | 6 to 10 s |
| x4c | 25 to 40 s (8 cold cells with 512 MB rings) |
| x4d | 8 to 16 s (timeout 30) |
| x4e | 5 to 12 s (worst 30 s by the timeouts 9 + 9 + 12) |
| x4m | 45 to 85 s (8 cg arms of 1.8 to 3.8 s; 12 together arms of 2 to 4 s; two 576 MB CPU sets and a 532 MB ring to fill) |
| x4s | 70 to 135 s (22 split arms, a ring built per arm; 10 hand-off arms of 1.3 to 3.3 s) |
| x4r | 25 to 35 s |

That is about 3.5 to 6.5 min of blocks, plus ten r16 gates of up to 180 s each. Total about 6 to 37 min, depending on the
gates. For scale (*host*, PoCL, not a phone figure): x4m took 50 s and x4s 61 s in Docker.

Memory peak (*estimated*):
- x4c: about 0.8 GB.
- x4m: about 1.2 GB in the together cell (576 MB CPU weights + 532 MB GPU ring + the app); the cg cell about 0.6 GB.
- x4s: about 0.6 GB.
- x4r: about 4.1 GB (1.5 GB GPU + 2.6 GB arena + the app).
  - R74's x3r held 3.5 GB, with MemAvailable falling to 0.59 GB and passes 91 to 99 slowing.
  - At 4.1 GB a low-memory kill or swap is likely, not just possible. x4r is last and the report is saved before it, so a
    kill loses only x4r.
  - The lead may want a pre-registered reading for "killed in x4r".

## Limits and open points

- **Cool start** is the owner's. The button text says "Cool start: do not run any other probe before this". The kit
  cannot check it; the gate series and the first block's clocks show the state.
- **Comparison with R74:** x4a and x4b's 0-3 cells reproduce R74's launcher placement, but on a spun or unspun little
  cluster, as labelled. Compare them with R74's `x3 rt` only with the clocks beside them.
- **x4f `rtargs`:** `args=2` alternates two matrices, while `args=15` reuses one (6.5 MB, larger than the GPU caches). Read
  `enq_us` and `lcpu_us` for the cut, not `ovh_us`.
- **x4m together** uses the profiling queue for the GPU (one event per launch), where x3c used the plain queue. Within the
  cell the two widths are measured the same way; against R74's `x3 togsum` read the difference as including that.
- **x4m cg** uses the upstream mc3 kernel as it is at 620d2225c. It is the kernel ggml would run for 2 to 4 columns on Adreno;
  it is not tuned here.
- **x4s hand-off** runs without CPU rows, on one thread. It prices the data moves of a join, not their overlap with CPU
  work. The f32 output read is 26 KB at share 0.30; at other shares it scales with the rows.
- **x4e svm_c11** prints the build log only when both options fail. When CL2.0 fails and CL3.0 builds, the line shows
  `opt=-cl-std=CL3.0` and the CL2.0 log is not kept.
- **x4e gemv** is the probe's own kernel, not a looped upstream kernel. It prices claiming and looping, not Y15's real
  rate.
- **x4d** uses unverified qcom signatures (see above); it is in a child process for that reason.
- **"Callback"** in the memo's X4-b is implemented as `clWaitForEvents` (`wfe`).
- **Not in scope:**
  - batch 2 inside the split (x4m measures two columns, but only on the whole 11008 x 4096 matrix and the together cell);
  - the "striped" flag variant (Ebony X4-e);
  - levels stacked inside the split (x4s splits only the fused [gate;up]);
  - any model;
  - the R74 `x3 loadlog` text ("Program not built!"), which is with the numerics-check Tom (R74 §7 item 1).
- **Shared files:** 0007's README section is appended to `patches/llama.cpp-c3/README.md`, after the other Tom's
  uncommitted 0006 section, which was left as it was. `.gitignore`, `.ignore` and `tools/__pycache__/` in the working tree
  are not this kit's. `tools/c3_x3emu_check.c` is the other Tom's.
