# Memo 30: research kit r27, thread pin (0010) and executable model mapping (0011) in one 2x2, models on internal storage

Agent model: claude-opus-5-5

Date 2026-11-18. Author: Tom (code). **Nothing here ran on a phone, and no phone speed is stated.** The APK was built
and inspected on the laptop. Labels: *host* = Apple M4 Pro (macOS, or a Linux aarch64 container on it), relative only;
*estimated* = arithmetic; *device* = quoted from an existing report. Kit r27 is kit r26 ([memo-27](memo-27-kit-r26.md))
with the lead's decisions of [memo-28](memo-28-code-audit.md) and memo-29's T1, T3, Y1 and the pin question (Y4 in
its simple form). Nothing is committed by this memo's author; the lead commits.

## 1. The runtime changes

Tree `lab/llama.cpp-c1r27` = 620d2225c + `patches/llama.cpp/0001-0003` + `patches/llama.cpp-c1-olmoe/0001-0011`;
every patch applies in order on a clean export of the base and the result equals the tree. Both are declared
bit-identical and act only with their knobs. Full description: `patches/llama.cpp-c1-olmoe/README.md`.

- **0010-c2-pool** (CPU module, plus a host-only knob in `common/common.cpp`). `VOLLIE_POOL=1`: graphs that bring no
  threadpool (the app's case) run on one persistent ggml pool with poll 0 instead of creating and joining the workers
  per graph. `VOLLIE_CPUMASK=<hex>[,<hex>...]`: each thread (0 = the caller) checks its affinity at every graph start
  and sets the mask when it differs, so a change of the calling Dart thread (memo-28 §2.1) no longer moves the work off
  cpus 4-7. `VOLLIE_SCHED=1`: per-thread schedstat and context switches (T1). Lines `c2-pool` (per prompt) and
  `c2-sched` (per prompt and phase). `VOLLIE_NO_ATTACH=1` exists only so that `llama-perplexity` takes the app's path.
- **0011-c2-exec-map** (`libllama.so`). `VOLLIE_EXEC_MAP=1`: the model mapping becomes `PROT_READ|PROT_EXEC` by
  `mprotect` (memo-29 Y1), applied by the exported `llama_vollie_exec_sync()`, which the kit calls right after it sets
  the run's env and before the gate wait; `llama_decode` repeats the check (a no-op when the state matches). Line
  `c2-exec` once per run id: want, mappings, bytes, `mprotect` calls and failures, errno, time, `where=api|decode`, and
  the protection read back from `/proc/self/maps`.

## 2. Host checks (*host*, 2026-11-18)

Host builds as memo-27 §2 (macOS AppleClang without OpenMP; Ubuntu 24.04 aarch64 container, gcc 13.3,
`GGML_OPENMP=OFF`; `armv8.2-a+dotprod+fp16`, no KleidiAI, shared libraries). Kit off env `GGML_TINY_OP=262144
GGML_EXPERT_IO=1 VOLLIE_SAMPLER_BYPASS=1 GGML_C2_IO=1`, `-c 512`, wikitext-2-raw test. Configurations: `attach` = the
tool as shipped (its own threadpool attached); `off` = `VOLLIE_NO_ATTACH=1` (the app's path: a disposable pool per
graph); `pool` = off + `VOLLIE_POOL=1 VOLLIE_CPUMASK=<mask> VOLLIE_SCHED=1`; `exec` = off + `VOLLIE_EXEC_MAP=1`;
`both` = pool + exec. macOS `-t 8`, mask `0xf0` (ignored on macOS); Linux `-t 6`, mask `0x1,0x2,0x4,0x8,0x10,0x20`
(one cpu per thread, the strictest case).

**Per-chunk PPL** (20 chunks at `-b 512` and `-b 512 -ub 32`, 8 at `-b 512 -ub 1`; `off`'s first four values and its
last; every other configuration compared value by value with `off`):

| platform | model | `-b 512` | `-ub 32` | `-ub 1` |
|---|---|---|---|---|
| macOS | olmoe-Mi | attach, pool, exec, both equal, 20 of 20 (8.9646 10.4244 9.2074 9.4783 … 12.5367) | pool, exec, both equal, 20 (8.9131 10.3435 9.0932 9.3987 … 12.5029) | equal, 8 (8.9467 10.3912 9.1413 9.4300 … 10.2291) |
| macOS | granite-Mi | equal, 20 (4.9584 6.6520 6.1119 6.6024 … 9.7318) | equal, 20 (4.9398 6.6449 6.0940 6.5732 … 9.7486) | equal, 8 (4.9839 6.6927 6.0907 6.5628 … 8.5192) |
| Linux | olmoe-Mi | equal, 20 (8.9607 10.4468 9.2244 9.5063 … 12.5459) | equal, 20 (8.8685 10.3989 9.1726 9.4239 … 12.4817) | equal, 8 (8.9975 10.4310 9.1597 9.4500 … 10.2878) |
| Linux | granite-Mi | equal, 20 (4.9811 6.6764 6.1185 6.6045 … 9.7685) | equal, 20 (4.9737 6.6967 6.1349 6.6100 … 9.7453) | equal, 8 (4.9805 6.6797 6.1233 6.6200 … 8.5720) |

- The `-b 512` values and the last values of all three settings equal memo-27's (0001-0009 build): 0010 and 0011
  changed no value at the chunks both runs have.
- **The knobs acted** (Linux): every `pool` / `both` run at `-b 512` and `-ub 32` printed `pool_graphs` = `graphs`
  (21/0 and 321/0), each worker on its own cpu (`w<i>=<tid>:0x<1 << i>:s1:e0:c<i>=<graphs>`: one mask set, no failure,
  every graph start on its cpu); `c2-sched` readable (prefill `n=21`). Every `exec` / `both` run printed `errno=0
  prot=r-xs:1` (OLMoE) or `r-xs:2` (Granite, two fragments); the `mprotect` of the resident mapping took 236,108 to
  387,430 µs over the 12 runs (*host*). On macOS the pool ran (`pool_graphs` = `graphs`) and `mprotect` succeeded
  (`errno=0`, 439 µs), with no affinity API and no `/proc` (`prot=n/a`, masks `0x0`).
- At `-ub 1` no `c2-pool` line is printed: with no graph of more than one token the c2-io prompt never opens (0002's
  rule, unchanged), so the line has no prompt to close. The kit's prompts always have a prefill graph. That the pool
  runs single-token graphs is shown by the greedy test below.

**Greedy text on the app's settings** (memo-26's test program: `n_ctx` 4096, q8_0 K and V, flash attention, three
prompts in one process, 32 greedy tokens, `VOLLIE_KV_DONTNEED=1 VOLLIE_SCHED=1`, `llama_vollie_exec_sync()` called before
each prompt; 3 threads; configurations off, pool, exec, both; Granite also with its num arm):

| platform | OLMoE | Granite | Granite num |
|---|---|---|---|
| macOS (mask `0xf0`) | four configurations equal (md5 `6716c00a…`, memo-27's) | equal (`041ac181…`, memo-27's) | equal (`041ac181…`; `c2-skip-run c1_skip=130`, `c2-head-run c1_m=2048` printed) |
| Linux (mask `0x10,0x20,0x40`) | equal (`e128edc3…`, memo-27's) | equal (`041ac181…`) | equal (`041ac181…`) |

Linux `both`, OLMoE: `c2-pool … graphs=1/32 pool_graphs=1/32 w0=…:0x10:s1:e0:c4=33 w1=…:0x20:s1:e0:c5=33
w2=…:0x40:s1:e0:c6=33` (decode graphs on the pool, each thread on its cpu), `c2-sched phase=decode n=32`, `c2-exec …
errno=0 us=327643 where=api prot=r-xs:1` (the exported call applied it, outside `llama_decode`), `c2-kvclear … why=ok`.
Off: `pool=0`, workers with affinity `0xfff` spread over cpus 0-11 (the case pin is meant to remove).

macOS and Linux values differ from each other (compiler); every comparison is within one build.

## 3. Storage step (owner, 2026-11-18)

Runs first, before the idle baseline and every warm-up (`kitStorageStep`, report field `storage`).

| | path on the phone |
|---|---|
| internal (used when the check passes) | `<filesDir>/kit-models` = `/data/user/0/com.insaneutils.vollie.c2/files/kit-models` (Android's `filesDir`; the report records the real path in `storage.internal_dir`) |
| external (today's files, never deleted or changed) | the folder of the app's loaded model, `getExternalFilesDir(null)` = `/storage/emulated/0/Android/data/com.insaneutils.vollie.c2/files` (`storage.external_dir`) |
| probe file | `vollie-exec-probe.bin`, 1 MiB of zeros, written, probed and deleted in each folder |

1. **Exec check, internal, first.** The folder's `/proc/self/mountinfo` row (`internal_check.mount`), then the probe
   file: `open` O_RDONLY, `mmap` `PROT_READ` `MAP_SHARED`, `mprotect` to `PROT_READ|PROT_EXEC`, and a fresh
   `mmap` with `PROT_READ|PROT_EXEC`; the errno of each (`internal_check.probe` = `open`, `mmap_r`, `mprotect_rx`,
   `mmap_rx`, 0 = success). Pass = `mprotect_rx == 0` and `mmap_rx == 0`.
2. **T2, external.** The same on the external folder (`external_check`), and the same probe on the first 1 MiB of each
   external model file (`external_check.models`).
3. **Only if 1 passed:** each internal copy of the right size is hashed (full sha256, every session; `hash_ms`). Missing
   or wrong copies are listed; one free-space check for all of them (bytes to write + 200 MB; both models from empty:
   6,233,668,896 + 200,000,000 B, about 6.4 GB) and the kit stops with a message if short (`stopped: space`, nothing
   copied). Each copy comes from the external file when it is on the phone and checked (copy to `.part`, fsync,
   sha256, rename; `copy_ms`, `hash_ms`, `sha256`), else from the kit's resumable download (Wi-Fi only, sha256;
   `download_ms`). After the copies the probe is repeated on the real internal files (`models.<m>.probe`). Free space
   at the start and the end is recorded (`free_start`, `free_end`).
4. **If 1 failed:** the session runs from the external files, `used: external`, `exec_factor: "void: ..."`; the
   parser marks the exec factor VOID and still reports pin.

The internal copies stay after the session (about 6.2 GB in the app's private storage, removed with the app's data).
The app's chat keeps using its own external model file; only the kit loads the internal copies.

## 4. Protocol

r26's protocol (memo-27 §3: blocks G O O G, one arm per model, r16 gate, hold, 3 threads, charger guard, idle
baseline, every counter), with:

- `VOLLIE_KV_DONTNEED=1` and `VOLLIE_SCHED=1` in every run (kv: R66).
- **Two factors in a balanced 2x2 per quad.** Pin: P = `VOLLIE_POOL=1 VOLLIE_CPUMASK=0xf0`, p = today's behaviour (no
  pool; the app's DartWorker pin stays in both). Exec: X = `VOLLIE_EXEC_MAP=1`, x = unset. Cells 1 = Px, 2 = pX,
  3 = PX, 4 = px. Quad orders are the rows of a Williams square, `[1,2,4,3] [2,3,1,4] [3,4,2,1] [4,1,3,2]`; the
  model's k-th quad (k = 0..3 over its two blocks) takes row (k + 2 x model index) % 4. Every cell sits once in every
  position per model, every cell follows every other once, and pin runs P p p P or p P P p in every quad. Per model and
  factor 8 on and 8 off scored runs.
- Warm-ups (2 reps) and the two profiled runs: pin on, exec off.
- The second profiled run per block is T3's block mode (`GGML_CPU_AB_WARM=32`, one 14-op table over the 32 decode
  graphs).

Scored run order (cells by name):

| block | model | runs 1-4 | runs 5-8 |
|---|---|---|---|
| b1 | granite (num) | pin, exec, plain, pin_exec | exec, pin_exec, pin, plain |
| b2 | olmoe (off) | pin_exec, plain, exec, pin | plain, pin, pin_exec, exec |
| b3 | olmoe (off) | pin, exec, plain, pin_exec | exec, pin_exec, pin, plain |
| b4 | granite (num) | pin_exec, plain, exec, pin | plain, pin, pin_exec, exec |

48 runs per session (44 specs), 40 gated, as r26. The choice of P: `0xf0` floating on cpus 4-7 is campaign 1's intended
pin plus the persistent pool. A strict one-cpu-per-thread pin (`VOLLIE_CPUMASK=0x10,0x20,0x40`) is supported by the
runtime but is not in the design; it would be a third factor or a later round.

**Names for the pre-registration (from the built app code):**

| | value |
|---|---|
| kit / app / proto | `r27` / `1.27.0+27` / `r27` |
| always set | `VOLLIE_LINES=stderr`, `VOLLIE_RUN=<id>.<rep>` |
| Granite env, every run | `GGML_TINY_OP=262144 GGML_EXPERT_IO=1 VOLLIE_SAMPLER_BYPASS=1 GGML_C2_IO=1 VOLLIE_KV_DONTNEED=1 VOLLIE_SCHED=1 GGML_EXPERT_SKIP=130 GGML_HEAD2=2048 GGML_HEAD2_Q=q4_0 VOLLIE_SKIP_STATS=1` |
| OLMoE env, every run | `GGML_TINY_OP=262144 GGML_EXPERT_IO=1 VOLLIE_SAMPLER_BYPASS=1 GGML_C2_IO=1 VOLLIE_KV_DONTNEED=1 VOLLIE_SCHED=1` |
| OLMoE load env | `VOLLIE_HEAD2_REPACK=1` (loads only); Granite none |
| pin on adds | `VOLLIE_POOL=1 VOLLIE_CPUMASK=0xf0` |
| exec on adds | `VOLLIE_EXEC_MAP=1` |
| profiled run `_prof` adds | `GGML_CPU_PROF=1 GGML_CPU_AB_TOKEN=1 GGML_CPU_AB=knob:MMID_FUSE+CHUNK2:1:1` (and pin) |
| profiled run `_profb` adds | `GGML_CPU_PROF=1 GGML_CPU_AB=knob:MMID_FUSE+CHUNK2:1:1 GGML_CPU_AB_WARM=32` (and pin) |
| run ids | `<m>_<arm>_wu`, `<m>_<arm>_r16`, `_r16_pin`, `_r16_exec`, `_r16_pin_exec`, `<m>_<arm>_prof`, `<m>_<arm>_profb`, with `<m>_<arm>` = `granite_num`, `olmoe_off` |
| report fields | `storage`, `context_rule`, `pin_rule`, `exec_rule`, `design`, `instrument_changes`; spec `pin`, `exec`; row `exec_sync`, `smaps_before`, `smaps_after`, `ru` |
| kit lines | `c2-pool`, `c2-sched`, `c2-exec` (formats in the c1-olmoe README), plus r26's |

## 5. Declared instrument changes (memo-28 decisions)

All listed in the report's `instrument_changes`:

- n_ctx fixed at 4096 at every kit load (`Research.nCtx`; every stored session had 4096 by the app's rule).
- Static dark kit screen while the kit runs: black, the status text and a Stop button, no spinner or animation.
- No OpenCL backend or OpenCL helper in the APK (`libggml-opencl.so`, `libvollie_clc.so` excluded; the APK has none).
- **Field meanings changed:** `res_after` now comes from the run-end page vector (one `mincore` less; its `ms` is the
  count from the vector, marked `from: end vector`). `vmstat_first` and `io_read_bytes_to_first_chunk` are gone (both
  were read inside the timed window, and the latter was unreadable). `rusage` gains user and system µs and context
  switches (`ru` = start, first chunk, end).
- Native library hashes once per session (per-load `libs` unchanged in meaning); native env strings freed; the log read
  from the last offset.
- Riders: T1 (`smaps_before` / `smaps_after`: model mapping size, Rss, Pss, Swap in kB, anon totals and top 12; the
  `c2-sched` lines; `ru`), T3 (`_profb`).
- Not done: memo-28 §3.4, §3.7 (zones), §3.9 (prefs), §3.10; §3.3 kept as it was. T1's mapped-against-resident at the
  first chunk is not taken (a full `smaps` read inside the window).

## 6. Parser

`tools/c2_research_parse.py` reads proto r27: checks (r24's; arm per model; id suffix against env; kv in every run;
warm-ups and profiled runs pin on and exec off; every quad holds the four cells; greedy texts per cell, a difference
is the kill rule; the storage record: mount rows, probes, `exec_factor`, copies, free space); main effects per model
and factor (paired within quad as mean(on) - mean(off) ± SE over the 4 quads, and Welch) for tok/s, decode graph
ms/token, TTFT, pre-prompt and decode majors, residency before, lost in wait, mapped before (smaps), run-queue wait
and involuntary switches per token; 2x2 cell means; worker placement per pin level (runs with a worker on cpus 0-3 or
a mask outside 4-7, pool graphs, mask sets and failures); protection per exec level (`c2-exec` want, errno, where,
prot; X runs where the mapping was executable, and x runs where it was, which voids the factor); T1 and T3 tables.
Output on the stored r20, r22, r23, r25 and r26 reports is
byte-identical to before this change (r21's summary path crashes before and after, pre-existing); the selftest gains a
synthetic r27 report (pin +1.0 tok/s, exec +0.5) with assertions, and passes.

## 7. Build

`tools/c2_build_kit.sh lab/llama.cpp-c1r27 lab/app-c2 patches/llama.cpp patches/llama.cpp-c1-olmoe`, app patches
`0001` to `0011`, `android/build_cpu.sh` flags unchanged. `flutter test` (39 tests) and `flutter analyze lib test`
pass.

| | bytes | sha256 |
|---|---|---|
| `vollie-c2-1.27.0+27-research-r27.apk` | 16,352,258 | `c05ebfd1a1454f726ab3632bdd16638d41547874dc0ce03057c4e540da3986d4` |
| `libggml-cpu-android_armv8.2_2.so` | 1,772,800 | `edc68be1e4f3840e185955ca0c5a59c48b7a6c46852a4f8fe64917f4b9077398` |
| `libllama.so` | 4,293,928 | `1394a5fe1b01022df47eaf6cc49e76405ecd121cd93b6e7dbd847fcc6efa12f2` |
| `0010-c2-pool.patch` | | `ee4a42d296a824359a7998c840df4283b974b6bc26e769fee182d3ae81539b12` |
| `0011-c2-exec-map.patch` | | `d0b7f8a2db092142deb297d0e19d20313220eb1fb4d80e483569dbb64b24be43` |
| app `0011-kit-r27.patch` | | `1c4904267e4c60206975ac68af12eb58650773383da5e2008fadab4199fdd424` |

- Same debug key as r26 (certificate SHA-256 `c49dde85…69fbb1716`); package `com.insaneutils.vollie.c2`, versionCode
  27, versionName 1.27.0. The APK is smaller than r26's (17,228,499 B) because the OpenCL libraries are out.
- `libllama.so` exports `llama_vollie_exec_sync` and imports `mprotect`; the CPU module imports `sched_getaffinity`,
  `sched_setaffinity`, `sched_getcpu`, `getrusage` and holds the `c2-pool`, `c2-sched` and
  `/proc/thread-self/schedstat` strings.
- Build record: `lab/dist-c2/BUILD-r27.txt` (and `assets/kit_build.json` in the APK).

## 8. Owner steps

1. Before the session: close background apps, unplug the charger, and have at least 6.4 GB free on the phone for the
   first r27 session (the copies; later sessions need none if the copies are intact). Wi-Fi only matters if a model
   file is missing on the phone.
2. Install `vollie-c2-1.27.0+27-research-r27.apk` (HF repo `r27/`), an update over r26 that keeps both models.
3. In Settings, tap **"Run research kit r27 (~40-80 min)"**. The storage step runs first (status line "Kit r27
   storage · ..."); the first session copies both models, then the idle baseline and the runs follow.
4. When it ends, tap **"Save research results to Downloads"** and send `Download/vollie-research-r27-<time>.json`.

## 9. Time (*estimated*)

r26's run list and waits (memo-27 §7: about 60 to 65 minutes when every r16 wait runs to 91 s, 32 to 40 when the gates
are reached), plus the storage step: the first session copies 6.2 GB and hashes it twice (copy and check) and every
session hashes 6.2 GB once; at an assumed 0.2 to 0.5 GB/s per pass that is 0.5 to 1.5 minutes per pass, so about 2 to
5 minutes the first time and 0.5 to 1.5 minutes later.

## 10. Never run on a phone, not verified

- Whether the phone's internal app storage accepts `PROT_EXEC` on a file mapping (the check answers it) and whether
  the external FUSE path refuses it (T2 answers it).
- `mprotect` time on the phone (container: about 0.3 s for a resident 3 GB mapping, *host*); it is outside the timed
  window only if `llama_vollie_exec_sync()` reaches the same `libllama.so` instance the app uses. `DynamicLibrary.open
  ('libllama.so')` should return the loaded library; if it does not, the call acts on an empty registry and the
  mprotect runs inside the first `llama_decode`, that is in TTFT. The `c2-exec` line's `where=api|decode` shows which
  happened in every run.
- `sched_setaffinity` on cpus 4-7 from the app's process (allowed in the container; Android's cpuset for a foreground
  app normally includes 4-7, unverified). `e<n>` in `c2-pool` counts failures.
- `/proc/thread-self/schedstat` readability under the app's SELinux domain (`n=0` in `c2-sched` would show it is not).
- The persistent pool's sleep and wake cost on the phone's kernel, and whether a Dart-side change of thread between
  prompts ever makes two graphs meet the pool at once (falls back to a disposable pool; `pool_graphs` counts it).
- No read-back through the executable mapping is done (memo-29 Y1 (b)); pages are not touched by 0011.
- The copy and hash speed of the phone's storage (§9 is an assumption).
