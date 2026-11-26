# Kit r37 for R76: x4s re-celled (cool pass, gap sweep, heat pass), build notes

Tom, 2026-11-26. The owner approved option (a) on 2026-11-26. Scope: `clyde-r76-plan.md` §3, `bonnie-joins-r35.md` §4.1 and §5
(B-T35a, B-T35c), `ebony-joins-r19.md` §5 (T10, T11). Kit r36's x4s code, re-celled. No new GPU kernel, no model, 0 bytes of
weights. Not committed, not uploaded.

## What is in it

**Runtime patch:** `patches/llama.cpp-c3/0009-r37-cells.patch`. It sits on 0008 and changes only `src/llama-vollie-probe.cpp`.
It adds three blocks, `x4sc`, `x4sg` and `x4sh`, dispatched from r35's `x4body`. That means each block also gets r36's
`x4 therm` before and after, `x3 cond`, `x4 dl` and the live file `<dir>/x4-live-<blk>.txt`. The probe is off unless
`VOLLIE_PROBE` is set.

**App patch:** `patches/app-c3/0007-kit-r37-probe.patch`.
- Version 1.37.0+37, kit r37.
- Block list `x4i, x4sc, x4sg, x4sh`.
- Button "Run c3 probe r37 (R76: cool pass, gap sweep, heat pass)".
- Design string names this file.
- r36's r16 gate (up to 180 s) runs before every block, as does r36's 600 s per-block limit.
- x4i stays first because it compiles the kernel binaries that the later blocks load (R75a1 *device*: about 1 s).

**Dropped from r36's list:** x4f, x4c, x4d, x4e, x4m, x4s (with its hand-off cell) and x4r. Their code is still in
libllama.so but nothing in this kit calls it. The BUILD file says so.

### Timing and labels shared by every block

- `t_ms` is milliseconds since the block's load began, which is after its rings are made.
- `setup_s` in each setup line is the time spent before that point.
- All of Bonnie's H times are on the same clock as `t_ms`.

### x4sc, cool pass

**Rings:** made once before the first arm, and none are made between arms:
- CPU-alone 22016 rows;
- lead lanes at shares 0.40, 0.45, 0.50, 0.55;
- feeder lane at 0.25.

Each lane has 4 copies, as in x4s. The pool runs on 4-7 and the lead on cpu 4, as in x4s.

**Ten triplets**, each cpu (0.5 s) | split arm (1.0 s) | cpu (0.5 s), in this order:

| order | arms |
|---|---|
| asc | lead 0.40, 0.45, 0.50, 0.55 |
| mid | feeder 0.25 (R74's mode) |
| mid | stall triplet |
| desc | lead 0.55, 0.50, 0.45, 0.40 |

- **Stall triplet:** the split arm is the CPU-alone graph with a 200 µs busy stall on the lead before each `graph_compute`
  (Ebony's T1 stall arm). The stall is not counted in `cpu_ms`. `excess_us` = (stall arm ms per matrix − mean cpu arm) × 1000 − 200.
- **Warm-up:** each arm has 0.2 s of warm-up plus run_arm's wait for full clock (up to 2 s).
- **Rest:** 6 s between triplets.

**Lines:**
- `x4 sc trip`: `cpu1_ms split_ms cpu2_ms cpu_diff ratio excess_us`. Ratio = split ÷ mean of the two cpu arms; `cpu_diff` is
  the cpu arms' relative difference, for Bonnie's 3 % void.
- `x4 sc sum` per share: `ratio_asc ratio_desc ratio_med d_ratio`. The median of two orders is their mean.
- `x4 sc best`: the share with the lowest `ratio_med`. This is the C1 input.

### x4sg, gap sweep (Ebony T10)

- Only the 0.45 lane's GPU ring (9856 rows, 4 copies, 87 MB).
- The lead thread on cpu 4 busy-waits until the host-timed gap since the previous kernel was seen done reaches 0, 0.1, 0.2,
  0.3, 0.4, 0.5, 1 or 5 ms.
- Then it launches, flushes and spins.
- The gaps run round robin, with ascending and descending rounds in turn. Warm-up is 0.5 s, then 5 s measured (run_arm, set {4}).

**Lines:**
- `x4 sg gap` per gap: `launch_us flush_us lf_us wait_us e2s_us e2q_us dgap_us dev_us`. `dgap_us` = START − previous END on
  the device clock, which is the GPU's own idle gap.
- `x4 sg arm`: rounds, `g1_step_us` (lf p50 at 0.5 ms − at 0.2 ms, the G1 input), `g4_step_us` (5 ms − 1 ms, G4), and run_arm's tail.

### x4sh, heat pass (last block)

- Only the 0.45 lane and the CPU-alone ring.
- Arms of 1 s without rests: lead split at 0.45, with every 8th arm CPU alone. Warm-up is 0 s plus run_arm's clock wait.
- **Sampler:** a thread on 0-3, once a second (`x4 sh samp`), reads:
  - caps (`scaling_max_freq` of policies 0 / 4 / 7);
  - `gpu_max_mhz`;
  - `thermal_pwrlevel`;
  - headroom, from its own `AThermal` manager.
- **Steps:** a CPU step is any policy's cap falling from one sample to the next. A GPU step is GPU max falling or
  `thermal_pwrlevel` rising.
- **Stop:** 10 s after the second CPU step, or at 45 s.
- **Summary line:** `x4 sh sum t_s arms first_cap_s first_cpu_step_s second_cpu_step_s first_gpu_step_s cpu_steps gpu_steps stop`.
- **Not done in the runtime:** "from a passed gate". The app records the gate before x4sh (`gate.timeout`). A timed-out gate
  is the prereg's void, not a skip.

### Per arm, in all three blocks

**`x4 <sc|sg|sh> arm` fields:**
- `pos`, `t_ms`, `mode` (cpu, lead, feeder, stall), `share`, rows, `mats`, `ms_per_mat` (the measured window).
- p10/p50/p90 over the measured window (the last `cpu_n` cycles of run_arm) of:
  - `launch_us` (EnqueueNDRange) and `flush_us`, timed apart (B-T35c);
  - `lf_us`;
  - `cpu_ms` (CPU rows);
  - `wait_us` (spin to seen);
  - `e2s_us` (END to seen);
  - `slack_us` (CPU rows' end − END; positive means the GPU finished first, for Bonnie's C5 / C6 split);
  - `e2q_us` (END to seen with QUEUED placed at the launch call);
  - `qoff_us` (QUEUED − launch call);
  - `dev_us`.
- `clk=mono` when |p50 qoff| < 2 ms, meaning the profiling clock is the monotonic clock and `e2s` / `slack` are read on it
  directly. Otherwise `clk=queued`, and they fall back to the QUEUED-anchored values.
- Then r36's run_arm tail (`thr`, `mhz_range`, `max_mhz`, `gpu_mhz`, `gpu_max_mhz`, `hr`, `ramp_s`).

**`x4 <sc|sg|sh> can` (after every arm; about 0.14 s):**
- `c5_ns_per_w`: 0.1 s of p2hot's sdot loop on a 256 KB set, pinned to cpu 5.
- `spin7_mhz` and `spin4_mhz`: 20 ms register spins, 8 dependent adds per step, MHz against the monotonic clock (T11).
- For each: `_pin`, `_at` (cpu seen), and `_cur` = `scaling_cur_freq/cpuinfo_cur_freq` (kHz) read mid-spin.
- `dcvsh=cpu0:..,cpu4:..,cpu7:..` (`cpu{0,4,7}/dcvsh_freq_limit`), `isolated`, `thermal_pwrlevel`.
- An unreadable node prints `e<errno>`. An empty file prints `empty`.
- **A refused pin gives `-1` for that number.** R75a1 (*device*) printed `x4 pin cpu5=e22 cpu7=e22` at idle. The canary runs
  right after an arm, but if the pin is refused there too it says so rather than timing the wrong core.

**Left out on purpose (lead's §3 list, smallest form):**
- Bonnie r34 §6's `/proc/vmstat` and per-cpu `/proc/stat` deltas. The canary line and r36's per-thread `thr` cover the
  clock/bus split the lead asked for. Add them if the prereg wants them.
- T5, T5b, T9, T12, as the plan drops them.

## Hashes for the pre-registration

| item | file | sha256 |
|---|---|---|
| runtime patch | `patches/llama.cpp-c3/0009-r37-cells.patch` (348 lines, on 0008) | `039cafd7b1563e10d1cc1e0535338049af30e9e4ed922ea4bf3c1e09b52d9d93` |
| app patch | `patches/app-c3/0007-kit-r37-probe.patch` (95 lines, on 0006) | `b382e6b4f0de441f33725e609b05407eadfa513b1d2d72f7f5c30bae866929b7` |
| parser | `tools/c3_probe_parse.py` (r37report, `x4r37selftest`) | `2cce500135fca1fc49639821fc46df25cf8e24d72e131bc4ea1a74fe3fe364bb` (working copy) |
| APK | `lab/dist-c3/r37/vollie-c3-probe-r37.apk`, 16,646,106 B, versionCode 37, versionName 1.37.0, package `com.insaneutils.vollie.c2`, certificate SHA-256 `c49dde85…1716` (same as r36) | `72541fcc4db0dde9d8d85446be8c33c92d175aec4fc9bf5deed5f29e2683f96c` |
| build record | `lab/dist-c3/r37/BUILD-r37.txt` | `8e27a4bed234b0e7755922f4e2c27e600d1110472d2f50ab32faac1aa21a0210` |
| sums | `lab/dist-c3/r37/SHA256SUMS` | `c91225f784bca125368421b464d4ae469601aa903eae921763712738cc67ad3b` |

Libraries in the APK, against r36:

| library | sha256 | against r36 |
|---|---|---|
| `libllama.so` (c3 0001, 0002, 0003, 0005, 0007, 0008, 0009) | `4202674776dd6f6b8571a3a42e6a2a342c571b8484c8f043add6e3d746497c2d` | changed |
| `libapp.so` (Dart) | `5d06700a2f78f554a6767ba38eff8f586b398ffaf5f9247651ff3c30ae257b88` | changed |
| `libggml-cpu-android_armv8.2_2.so` | `46e0735d2824ba6f3a156e219e058a701f6024fdbecbb2ef937b3127115a9694` | byte-identical (= r33 to r36) |
| `libvollie_x4c.so` | `fb61545acea9990c3bff418682f46970eba24fd92b56af2b256368302779315f` | byte-identical |
| `libvollie_clc.so` | `e2c867f769cd854c4f96b475447a2791b55d1d9dbb693ab4e59b50c910e144d6` | byte-identical |

Every other library in the APK is also byte-identical to r36: libggml-base, libggml, the other CPU variants, libllama-common,
libllamadart, libmtmd, libflutter and libdatastore_shared_counter.

**Numerics: none.**
- The CPU module is r36's file, copied byte for byte.
- 0009 touches only the probe file, which is outside every inference path and off unless `VOLLIE_PROBE` is set.
- A perplexity check therefore has nothing to compare. Per-chunk identity at `-ub 1`, `-ub 32` and `-b 512` follows from the
  identical module and the unchanged graph code.

**Build:**
- Tree `lab/llama.cpp-c3r37`: a clone of 620d2225c with patches/llama.cpp 0001-0003, patches/llama.cpp-c1-olmoe 0001-0012 and
  c3 0001, 0002, 0003, 0005, 0007, 0008 applied (equal to the r36 tree file for file, `cmp`), then 0009.
- New files are marked with `git add -N`.
- Commands:

```
sh android/build_cpu.sh lab/llama.cpp-c3r37 <libs>        # then the r36 APK's CPU module copied over <libs>
LIBS_FROM=<libs> APP_SERIES="<app-c2 0001-0012> patches/app-c3" sh tools/c2_build_kit.sh lab/llama.cpp-c3r37 lab/app-c3 patches/llama.cpp patches/llama.cpp-c1-olmoe <dir with c3 0001 0002 0003 0005 0007 0008 0009>
```

- The APK was then copied to a plain name.
- A fresh build of the CPU module from the same ggml sources (`diff -rq` against r36's tree is empty) differs from the shipped
  file, as r35's notes recorded. That is why the r36 bytes are shipped.

## Checks (all *host*; host timings say nothing about the phone)

**Source and warnings:**
- The series 0001, 0002, 0003, 0005, 0007, 0008, 0009 applied to a clean clone reproduces the r37 probe file (`cmp`).
- The app patch applied to r36's three files reproduces the r37 files (`cmp`).
- No warnings in new code from any of: macOS `clang++ -std=c++17 -fsyntax-only -Wall -Wextra -Wcast-qual` (only 0001's two),
  the Android build with NDK 28.2 (the same two), Linux gcc in Docker (old-code pragma, cast and indentation warnings only).

**Functional run: arm64 Linux in Docker, PoCL, `--cpus=8`, x4i then x4sc, x4sg, x4sh (2 min 22 s).**
- Every arm err=0 and every `x4 dl` n=0. All live files were written.
- x4sc: 10 triplets, 30 arms, 30 canary lines, four `sum` lines and `best`. The stall triplet printed `excess_us`.
- x4sg: 8 gap lines, `n=72` rounds each.
- x4sh: 39 arms, 46 samples, `stop=45s`.
- `clk=mono` held on PoCL (`qoff` about 22 µs). PoCL's END-to-seen came out slightly negative on that basis, which is PoCL's
  stamp order. The phone decides its own; `e2q` is printed beside it.
- The register spin read about 4.46 GHz on the M4 Pro's P-cores, consistent with 8 cycles per step at its known clock. This
  is a check of the canary's arithmetic, not a phone number.
- The PoCL "GPU" is the host CPU, so every ratio of this run is meaningless and none is reported.

**Heat-pass logic under fake caps:**
- Setup: `--privileged`, a tmpfs over `/sys/devices/system/cpu` with copied topology, and fake `cpufreq/policy{0,4,7}`.
- A script lowered policy 4's `scaling_max_freq` at about 12 s and again at about 17 s.
- Result: `x4 sh samp` marked `cpu_step=1` at t_ms 12003 and 17000, and the summary read
  `first_cap_s=12.0 second_cpu_step_s=17.0 cpu_steps=2 stop=second_cpu_step+10s t_s=28.2`.
- The capped `max_mhz` showed in the arm tail.
- The GPU-step path was not exercised, because kgsl cannot be faked in the container.

**Knob and block names:** with the knob unset, `x4sc` and `x4sg` return -1. An unknown block returns -2.

**Parser:**
- `python3 -I tools/c3_probe_parse.py --selftest` prints `r72 ok`, `x3 ok`, `x4 ok`, `r36 ok`, `r37 ok`, `selftest ok`.
- The new `r37report` section has tables for triplets, per share, arms, canaries, gaps and cap steps, and the R76 inputs:
  - C1 best `ratio_med` and its share;
  - triplets with cpu arms more than 3 % apart;
  - `thermal_pwrlevel` above 0 in C or G;
  - G1 and G4 steps.
- The host run's JSON parses.
- All seven committed c3 device JSONs (R69 to R75a1) give output byte-identical to the committed parser's.

**Not checked:**
- on any phone;
- the GPU-cap step detection (no kgsl on the host);
- whether Adreno's profiling clock is the monotonic clock. The `clk=` field says so on the phone.

Host files (scratch, not kept): the Docker runner and the fake-sysfs script were in the session scratchpad.

## Phone time and memory (*estimated*)

| block | estimate | basis |
|---|---|---|
| x4i | about 1 s | R75a1 *device* |
| x4sc | 90 to 150 s | 54 s of rests; 30 arms × (0.2 s warm-up + 0.5 / 1.0 s measured + 0.14 s canary) ≈ 30 s; run_arm clock waits 0 to 2 s per arm; ring setup about 1.2 GB, 5 to 25 s |
| x4sg | 7 to 10 s | 5.5 s of arm, canary, ring upload, clock wait |
| x4sh | 20 to 50 s | stops 10 s after the second CPU step, else at 45 s from load start; setup about 0.4 GB, a few s |

- **Normal case:** about 2 to 3.5 min of blocks, plus four r16 gates of up to 180 s each. Total about 2 to 15 min. The lead's
  8 to 12 min with a cool start fits.
- **Load (lead's figure):** about 22 s in C, 5 s in G and 45 s at most in H. The rest is rests, warm-ups and canaries.
- **Ceiling:** 4 × (180 s gate + 600 s block limit) = 52 min, reached only if every block hung. The 2 s event deadline of
  r36 holds in every new wait.

**Memory peak:**
- x4sc: about 1.2 GB (host: CPU rings 745 MB, GPU rings 416 MB; it holds all five lanes at once, against x4s's one at a time).
- x4sg: 87 MB.
- x4sh: about 0.4 GB.
- x4r (4.1 GB) is gone, so no low-memory kill is expected.

## Summary

1. Kit r37 is built: `lab/dist-c3/r37/vollie-c3-probe-r37.apk` (1.37.0+37, sha256 `72541fcc…f96c`) with `BUILD-r37.txt`
   and `SHA256SUMS`. It is not uploaded and not committed.
2. New runtime patch `0009-r37-cells.patch` (blocks x4sc, x4sg, x4sh, off unless `VOLLIE_PROBE`) and app patch
   `0007-kit-r37-probe.patch` (blocks x4i, x4sc, x4sg, x4sh). r36's other blocks are dropped.
3. No numerics change: the CPU module is byte-identical to r36's (`46e0735d…9694`). Only libllama.so and libapp.so changed.
4. Host: the build is clean, all four blocks ran on PoCL with err=0, the heat-pass stop logic worked under fake CPU caps, and
   the parser selftests pass with old reports unchanged. Nothing was run on the phone.
5. Phone time is estimated at 2 to 3.5 min of blocks plus gates, with a memory peak of about 1.2 GB in x4sc. The kit prints a
   canary line after every arm, and a refused pin there prints -1.
