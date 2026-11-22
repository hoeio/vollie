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
