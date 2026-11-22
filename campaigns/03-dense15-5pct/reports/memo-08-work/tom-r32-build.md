# Kit r32 build (for R72): Tom, 2026-11-22

Plan: [clyde-r72-plan.md](clyde-r72-plan.md). Bands: [bonnie-path.md](bonnie-path.md) §10.4, [ebony-paths.md](ebony-paths.md) §12.
The kit is built. It is not uploaded: the upload waits until the R72 pre-registration is pushed. Nothing is committed.

## 1. What is in it

| part | file | sha256 |
|---|---|---|
| runtime probe rows | `patches/llama.cpp-c3/0003-r72-rows.patch` (on 0001, 0002; changes only `src/llama-vollie-probe.cpp`) | `a8799a36ba581e35fb5a1f0663f3a0abc645c64d27617317235be6d1c4ffb934` |
| app | `patches/app-c3/0002-kit-r32-probe.patch` (on app-c2 0001 to 0012 and app-c3 0001) | `a54d2cc8cac009377651f88c11e8ac9483c7941fc51b1a19da9766c5a36f0343` |
| APK | `vollie-c3-probe-r32.apk`, 16,506,426 B, `com.insaneutils.vollie.c2`, versionCode 32, versionName 1.32.0 | `08a37bd5896816fef88b6b0bd21fc8d753208be0af1313462e94c25c59f3a50c` |
| BUILD file | `BUILD-r32.txt` (the APK's `kit_build.json`: every series and app patch sha256) | `c8e8b9ea2460d0f14971315a7abe08b5af1d24111f1563f777811d680778188b` |
| libllama.so | in the APK | `350b57c3fd3cfd2b4018b5a5bc56dbb8695415ba2bac599740ad64f571134e63` |
| libggml-cpu-android_armv8.2_2.so | in the APK | `46e0735d2824ba6f3a156e219e058a701f6024fdbecbb2ef937b3127115a9694` |

The upload layout is `r32/vollie-c3-probe-r32.apk`, `r32/BUILD-r32.txt` and `r32/SHA256SUMS` (in the lab's `dist-c3/r32/`). Build
command (`tools/c2_build_kit.sh` now also accepts single patch files in `APP_SERIES`):

```sh
APP_SERIES="$(cd patches/app-c2 && ls 00[01][0-9]-kit-r[12][0-9].patch | sed 's#^#patches/app-c2/#' | tr '\n' ' ')patches/app-c3" \
  tools/c2_build_kit.sh lab/llama.cpp-c3r32 lab/app-c3 patches/llama.cpp patches/llama.cpp-c1-olmoe patches/llama.cpp-c3
```

The script's tree check passed: the tree is HEAD plus the series, file by file. The APK's libraries and `kit_build.json` are
the ones just built.

## 2. Blocks (button "Run c3 probe r32", no model needed)

The run order is a, l1, l4, l23, l5, l6, l7, x2, anon34, anon42. Each block runs after the cool-down gate.

| block | rows | host functional run (s, *host*) |
|---|---|---|
| `a` | R69 anchors: p2 read and q4x4 on 4-6, 7+4-5, 4-7 | 18.7 |
| `l1` | L1, L1m, L1m4: K1, K1m and k1m4 at keep 1.0, 0.666, 0.65, 0.60, 0.58 and 0.50; widths 4096 and 11008; hot and cold; on cpu 4 (A78) and cpu 7 (X1). Each group has a stock cell before and after it (`drift`). There is also a lists-build cell. 168 cells. | 257.7 |
| `l4` | L4: one layer of each subject's real shapes (Yi-1.5-6B kv 512 at keep 0.58; Llama-2-7B and DeepSeek-7B kv 4096 at keep 0.60). It compares stock with the rule (k and v stock; q, o, gate, up and down on K1 or K1m), on 3 A78, X1+2 A78 and X1+3 A78, in ABBA order. Then an `l4sum` line, then a ggml stock layer cell. | 134 |
| `l23` | L2 and L3: 3 and 4 big cores. Read and q4x4 cold; q4x4 hot with a private slice per thread; ggml Q4_0 gemv cold. | 29 |
| `l5` | L5: ggml gemv cost of Q4_0, IQ4_NL, Q4_K, IQ4_XS, Q3_K, IQ3_S, IQ3_XXS, Q2_K, Q6_K and Q8_0, with Q4_0 run again as pass 2. Widths 4096 and 11008, hot and cold, on {4}, {7}, 7+4-5 and 4-7. | 207 |
| `l6` | L6: graph barriers. Chains of 617, 579 and 225 nodes on {7}, 4-6, 7+4-5 and 4-7, plus a spin-barrier arm. | 4.6 |
| `l7` | L7: Q6_K head (with Q8_0 for reference) at vocab 64000, 32000 and 102400, cold, on 3 core sets | 38 |
| `x2` | r29's X2 rows, then a retry of the same kernels built with `-cl-opt-disable` (`x2r` lines), then an `x2info` line | 19 |
| `anon34`, `anon42` | anonymous-memory capacity (Bonnie's finding): 3.4 GB and 4.2 GB anonymous mmap, filled by 3 pinned threads and read for 20 passes or 45 s. Records majflt, minflt, Cached, SwapFree, VmSwap and VmRSS, then `hdr` lines. | about 1 each |

Phone time is an *estimate*. The cells are time-bound: each has warm-up, then a ramp of at most 4 s, then the measurement. On
top of that come up to ten gates of at most 180 s. That gives about 20 to 40 min, and about 65 min in the worst case (every
ramp and every gate at its cap). The report is saved before each block (`running_block`) and after each one. If the phone
kills the app in an anon arm, the blocks before it are kept.

The L6 node counts come from the *host* decode graph at `-ub 1`, counted with a `cb_eval` callback on the lab's Q4_0 files.
Yi-1.5-6B and Llama-2-7B each have 1001 nodes, 617 of them non-empty and 225 MUL_MAT. DeepSeek-7B has 939 nodes, 579
non-empty and 211 MUL_MAT.

## 3. The lead's R69 changes

1. **Clocks.** Every r32 cell samples the policy clocks every 0.1 s during the measurement and prints the min-max per
   policy (`mhz=`). After its warm-up, a cell waits until the clocks of its policies reach the policy's max. The cap is 4 s.
   It prints `ramp_s`, `ramp_ok` and `max_mhz=pN:scaling/cpuinfo`, so that a cell measured below max is visible and can be
   dropped. The cool-down gate now waits at most 180 s (R69: 90 s). The gate record has `max_s` and `timeout`.
2. **Off-core workers.** Every worker calls `sched_getcpu` every 16th work unit. Its `thr` field is
   `cpu:cpu_over_wall:nivcsw:nvcsw:off/checks@other_cpu`. The per-thread CPU time comes from `RUSAGE_THREAD`. The ggml
   cells (l5, l7, l23's ggml row, l4's ggml stock row) count the calls in which thread 0 was off its core
   (`cpu0_off_calls`). They report process-level CPU over wall and nivcsw.
3. **0.65 and 0.666 K1 rows.** These are in l1 at both widths, both cores, hot and cold. The H6 column kernel is not in r32.
4. **GPU lane.** This covers the `-cl-opt-disable` retry and `x2info` only: driver, device and C version, compute units,
   max MHz, global memory and the extension list. There is no new GPU kernel.

## 4. Checks

- **Identity of the runtime with the knob unset** (*host*; Llama-2-7B Q4_0, `llama-perplexity -c 512 -t 8`,
  `VOLLIE_PROBE` unset). The per-chunk running values are byte-identical to stock `620d2225c`:
  - `-b 512`: 5.8104, 6.3386, 7.0099, 7.9916
  - `-b 512 -ub 32`: 5.7851, 6.3350, 7.0107, 7.9935
  - `-b 512 -ub 1`: 5.7791, 6.3466
- **Bit identity inside the probe** (the phone checks it again in its own run, as `identity` lines). Before timing, l1 and
  l4 compare K1, K1m and k1m4 with stock over 256 rows using `memcmp`. The inputs are adversarial, with block scales of 0
  and 60000, and the dropped inputs are zeroed. l1 checks widths 4096 and 11008 at keep 1.0, 0.60 and 0.58; l4 checks
  each subject's keep. On the *host* every line is 1.
- **Build and functional run.** The NDK compile of the probe is clean with `-Wall -Wextra`. Every block ran on the host;
  the timings mean nothing for the phone. On the host the off-core counts read as "off" and `ramp_ok=1`, because macOS has
  no `sched_getcpu` or cpufreq. The parser `tools/c3_probe_parse.py` has an r32 report (`r32report`) and its `--selftest`
  passes.
- **Patch checks.** 0003 applies on 0001 and 0002, and reverses cleanly on the built tree. App 0002 applies with
  `git apply --check` on the r29 app files.

## 5. Known limits

- `a` keeps R69's timing, to stay comparable with R69, so it has no per-thread, clock-range or ramp fields.
- The ggml cells have process-level CPU over wall only. The thread-0 off-core count is the only per-thread field there.
- The L6 node counts are from the host graph (flash-attn on). The phone's graph may differ by a few nodes.
- Not included: the THP, devfreq and full clinfo riders. Only `x2info` is there.
- The ramp waits for `scaling_max_freq` (the ceiling the governor allows, which can sit below `cpuinfo_max_freq` under
  thermal limits; both are printed in `max_mhz`). If a policy's files cannot be read, the check counts it as at max
  (`ramp_ok=1`, `max_mhz` shows a non-positive value), so read `ramp_ok` together with `max_mhz` and `mhz`.
