# Kit r34: the x3 GPU lane probe (R74), build notes

Tom, 2026-11-22. Owner decision 2026-11-22: the GPU measurement is allowed as an exception to the device hold
(CAMPAIGN.md, commit `cf059df`). Design: [tom-host.md](tom-host.md) §1.5, plus memo-09-gap.md §10 (dispatch round trip,
residency arm) and memo-10-tree.md T3. Not committed, not uploaded.

## Contents

| item | file | sha256 |
|---|---|---|
| runtime patch | `patches/llama.cpp-c3/0005-x3-gpu-probe.patch` (1,438 lines, only `src/llama-vollie-probe.cpp`) | `6fbb4ec83ea53a6fd39cde94d24b6df3102784cd3c4a7820dbb63bfc6f0e753c` |
| app patch | `patches/app-c3/0004-kit-r34-probe.patch` (113 lines) | `632a589de7c7a3084bfdaa656d2ac9edb8c290f2a40ada27b490257e2ba7b1ec` |
| parser | `tools/c3_probe_parse.py` (x3 report, `--selftest`) | (tracked file) |
| APK | `lab/dist-c3/r34/vollie-c3-probe-r34.apk`, 16,564,106 B, versionCode 34, versionName 1.34.0, package `com.insaneutils.vollie.c2`, same debug certificate as r33 | `2b0a292978913c3d546d798aac895b414f1845b655133edc5d0eb2a5fcbe81de` |
| build record | `lab/dist-c3/r34/BUILD-r34.txt` | `caf3ab6614d78f13e9452a5d6b8bc97afb9d01e3d5149b311db6fcd08cbe2c5f` |
| sums | `lab/dist-c3/r34/SHA256SUMS` | `d4897c69739bb1879ce309c2e26248816418b783e0dec552d38619e2f0c035f8` |

Libraries in the APK: `libggml-cpu-android_armv8.2_2.so` byte-identical to r33 (`46e0735d…`); `libllama.so` rebuilt from
c3 0001, 0002, 0003, 0005 (0004-pin-guard is not in r34); `libvollie_clc.so` (campaign 1's `tool/clc.c`, unchanged,
7,048 B, `e2c867f7…` after the gradle strip) packaged again; `libapp.so` (Dart) changed; every other library identical to
r33. Upstream kernels, embedded verbatim, at base `620d2225c`: `gemv_noshuffle_q4_0_f32.cl` (22,301 B, `636126fd…cdf65`)
and `gemv_noshuffle_q4_0_f32_spec.cl` (16,486 B, `3f0b490a…00711`); both shas are printed
by `x3 src` on the phone.

## The knob

`VOLLIE_PROBE` (set by the app's probe button only). Unset: the probe entry returns -1 and nothing else in libllama calls
it. `VOLLIE_X3_HOST_TEST` is a compile-time define for the host functional test only (CPU OpenCL without fp16: scales
read with `vload_half2` into float2, device type ALL); it is never defined in a kit build. No numerics change: the
probe runs no model and touches no inference path.

## Blocks and cells (button "Run c3 probe r34 (GPU)", each block after the r16 gate, up to 180 s)

Every block starts with `x3 cond` (THP mode, every `/sys/class/devfreq` node with cur/min/max/governor, kgsl GPU clock),
then `x3 info` (device, driver, OpenCL C version, A6x, E031). Every timed arm ends with: CPU policy clock ranges against
scaling and hardware max, kgsl GPU MHz range, AThermal headroom, and `thr` fields per thread pinned to 4-7 (the parser's
M / U / P / C labels).

| block | line | prints |
|---|---|---|
| x3b | `x3 src` | kernel file sha256 and bytes |
| x3b | `x3 build` | per kernel (general, 4096x4096, 4096x11008, 11008x4096) and try: child exit code (0 ok, 3 opt-disable only, 1 failed, 2 setup, negative = signal, -1000 no clc, -1001 timeout 120 s), seconds, exact option string, binary and log bytes; then the final how (`opt`, `optdisable`, `fail`) |
| x3b | `x3 buildlog` | the compiler log (sanitised, capped) |
| x3b | `x3 load` | binary load: create / build-status / build / kernel error codes, wg max and multiple, local and private memory, ok |
| x3b | `x3 identity` | six kernel-shape pairs vs a float64 reference: max relative error, max abs, rms, rows bad, non-finite, ok |
| x3g | `x3 alone` | per shape and regime (hot, cold = 512 MB ring): copies, MB, gemvs, s, GB/s, G weights/s, us/gemv, device-profiled GB/s and us |
| x3g | `x3 rt` | launch + finish round trip at 512 / 1024 / 2816 / 4096 rows: wall, p50, p90, device us, overhead us, queued-to-start, submit-to-start, GPU MHz |
| x3g | `x3 layer`, `x3 layersum` | one full layer per subject (Yi 512x4096 via general, Llama-2, DeepSeek) on a 4-layer ring, sync after each launch (sync7) and once per layer (sync1): ms/layer, body ms, GB/s; sum line with ms sync1 / sync7 / device and round trip and overhead per dispatch |
| x3c | `x3 tog`, `x3 togsum` | setup; ABBA cpu, gpu, both, both, gpu, cpu: ggml Q4_0 gemv on 4-7 (604 MB cold set) and GPU 11008x4096 cold, feeder on 0-3: CPU GB/s, GPU GB/s, sum; sum line with alone, together, sum vs CPU alone, losses |
| x3c | `x3 split`, `x3 splitsum` | setup; 11008x4096 row split at GPU share 0.25 / 0.35 / 0.50 (2,816 / 3,840 / 5,504 rows), order cpu, .25, .35, .5, .5, .35, .25, cpu: ms per matrix, CPU ms, wait ms, GPU ms, GB/s; sum line per share with ratio to CPU alone and feeder CPU time over wall |
| x3r | `x3 res` | 0.9 GB of GPU buffers (about 35 copies of 11008x4096) allocated and read by the GPU while 3 A78 read a 2.6 GB anonymous arena for 15 s (or 200 passes): stages (start, gpu_alloc, anon_fill), per pass GB/s, GPU GB/s, faults, MemAvailable, cached, swap, RSS, MHz; `x3 res sum` (median and min CPU GB/s, GPU GB/s, major faults) |

## Checks (all *host*)

- 0005 applied to the r32 probe file reproduces the r34 tree's file exactly (`cmp`).
- Android build of libllama, same flags as `android/build_cpu.sh`: from the 0003 state to 0005, ninja recompiled only
  `llama-vollie-probe.cpp.o`; 559 of 560 objects byte-identical. The CPU module ships byte-identical from r33, so 0003's
  knob-unset PPL identity (`-b 512`, `-ub 32`, `-ub 1`) carries over; no new PPL run (the laptop runs the freeze job).
- All four blocks ran on arm64 Linux in Docker (`--cpus=4`) against PoCL as a functional test: every cell printed with
  err=0, identity max relative error about 3e-6 on all six pairs (with the host scale shim), thread labels filled for
  ggml's workers. Host speeds mean nothing and are not reported.
- macOS `clang -fsyntax-only -Wall -Wextra` on the probe file: clean. Android build: no new warnings.
- `python3 -I tools/c3_probe_parse.py --selftest`: `selftest r72 ok`, `selftest x3 ok`, `selftest ok`; the x3 report ran
  on the host JSON.

## Phone time (*estimated*)

x3b 15-60 s (Adreno compile time unknown), x3g 35-45 s, x3c about 40 s, x3r 20-25 s: about 2 to 2.5 min of blocks plus
four r16 gates of up to 180 s each. Memory peak *estimated* about 1.1 GB in x3c and about 3.5 GB in x3r (last block, so a
low-memory kill ends only the residency arm; the report is saved before and after each block).

## Limits

- Whether the phone's driver (Adreno 660, E031.38-class) compiles upstream's kernels is unknown until x3b runs; the
  `-cl-opt-disable` retry and the general-kernel fallback (`Kset::pick`) cover a failed spec kernel, not a failed
  general one. A failed general kernel leaves x3g to x3r printing `ok=0 reason=no_kernel`.
- Host identity used float scales (PoCL has no fp16); the phone path uses upstream's half2 scales and is checked by
  `x3 identity` on the device.
- The split arm's idle workers are expected to label P (parked) while the GPU part runs; that is the design, not a fault.
- Not in scope: batch-2 calls, T2/T4, decode, any model. 0004-pin-guard is not in r34.
- `.gitignore` and `.ignore` changes in the working tree are graft's, not this kit's.
