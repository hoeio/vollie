# Memo 28: code audit of the research kit (app r26 and runtime c1r26): what costs CPU, memory, I/O or wake-ups during a run

Agent model: claude-opus-5-5

Date 2026-11-18. Author: Tom (code). Owner's request (2026-11-18): the app is a research kit, so anything that costs
CPU, memory, I/O or wake-ups on the phone during a run and is not needed for the measurement is a candidate to cut.
**No code was changed and nothing ran on a phone.** Sources: the app copy `lab/app-c2` (2.3.0+45 + `patches/app-c2`
0001-0010), the runtime tree `lab/llama.cpp-c1r26` (620d2225c + `patches/llama.cpp/0001-0003` +
`patches/llama.cpp-c1-olmoe/0001-0009`), llamadart under `lab/app-c2/third_party/llamadart`, and the stored device
reports `reports/device/r23_2026-11-16.json`, `r25_2026-11-16.json`, `r26_2026-11-18.json`.

Labels: *device* = read or computed from those three reports (short scripts over the JSON); *host* = a microbenchmark
in the Ubuntu 24.04 aarch64 container on the M4 Pro (kernel 6.10.14-linuxkit, gcc 13.3), relative only; *estimated* =
arithmetic or judgement; *inferred* = a reading of code plus device data that no experiment has confirmed. File:line
refers to the files as they are now.

## 0. Summary

Ranked by what could matter for tok/s or memory on the phone (§2), then cleanliness (§3), then things that stay (§4).

1. **The thread pin is lost in gated runs** (§2.1, *device* + *inferred*). The app pins the "DartWorker" threads that
   exist at the moment a run starts; after a 90 s gate wait the Dart VM has retired its idle pool threads, so the
   generation runs on a fresh, unpinned thread and ggml's workers inherit that. Worker threads were seen on the A55
   cores 0-3 in 39 of 60 gated OLMoE runs (r23, r25, r26) and in **0 of 12** OLMoE warm-ups. Effect on tok/s not shown
   (r26 +13.7 ms/token, r25 +1.4, r23 0). Fix in the runtime: an affinity knob applied at graph start. Bit-identical.
2. **The settings screen animates a spinner for the whole session** (§2.2, *device* lower bound + *estimated*). While
   the kit runs, `optimizing` is set, so a `CircularProgressIndicator` draws every frame (60-120 Hz). That is UI,
   raster, GPU and display work during every decode and every cooling wait. Lower bound from the reports: the process
   used **0.55 to 0.69 s of CPU per Granite run outside the three llama threads** (median, all three sessions). Fix:
   a static, dark kit screen. App only, no numerics, but it changes the bench, so it needs a pre-registration.
3. **The expert read-ahead (`GGML_EXPERT_IO=1`) issues 384 (OLMoE) to about 720 (Granite) `madvise` calls per decode
   token on thread 0**, also when the pages are resident (§2.3). Host cost 1.4 µs per call (0.53 ms per 384 calls).
   Phone cost unknown, *estimated* 0.5 to 3 ms of thread-0 time per token. A "skip slices advised in the last N tokens"
   variant is bit-identical and worth a device ABBA.
4. **ggml creates and joins two worker threads for every graph** (§2.4): llamadart never attaches a threadpool. Host
   cost 45 µs per graph; phone *estimated* 0.1 to 0.3 ms per token. A persistent pool (knob, bit-identical) also allows
   per-thread pinning, which fixes item 1 at its root.
5. Smaller memory items (§2.5-2.8), all *estimated*:
   - the OpenCL backend and the Adreno driver are loaded into a CPU-only kit;
   - `n_ctx` is not set by the kit (4096 so far, but it would be 8192 on a morning with 3.5 GB free at app start);
   - Granite's unread 148 MiB tied-head copy (memo-24, unchanged);
   - the sampler's full-vocabulary fill (0.12 to 0.20 ms per token, *device*).
6. **Instrument-only cuts** (§3):
   - the Settings build gzips and base64-encodes the whole report (> 300 k chars) on every rebuild, about 20 times per
     gate wait;
   - three whole-file `mincore` scans run back to back around every run (about 100 ms each, *device*);
   - every hold pass scans all 750 k pages again;
   - dead `/proc` reads sit inside the timed window;
   - native libraries are hashed at every load;
   - 82 thermal zones are read before every run;
   - native strings leak at every env call.
   None of these changes tok/s; together they spend about 0.4 to 1.0 s of wall time between runs (*device*).
7. **Things that stay** (§4): the hold's kernel read-ahead (the bulk of its 1.8 to 2.1 s CPU per wait is the read-back
   itself); the per-graph `getenv` block (host 0.8 µs per graph); `GGML_C2_IO`'s per-graph sample (host 5 µs, the
   `/proc/self/io` open fails on the phone anyway); the metric's 32nd decode graph; the end-of-kit model hashes;
   `EXPERT_IO` itself.
8. **The app and bindings add about 0.04 ms per token to the decode window** (*device*:
   duration − TTFT − decode_ms − sample_ms, over 31 tokens, r25 and r26, both models). The per-token Dart path
   (llamadart's loop, streaming, chunk callbacks) is not where the time goes.

## 1. Evidence used

| quantity | Granite | OLMoE | source |
|---|---|---|---|
| process CPU minus 3 × wall per scored or warm-up 3-thread run, median (lower bound of CPU outside the llama threads) | r23 +671 ms, r25 +692, r26 +553 (78 to 112 ms per s of run) | r23 +44, r25 +641, r26 +118 | *device*, `run_cpu_ms` (getrusage RUSAGE_SELF over the run) − 3 × `wall_ms` |
| decode window not in graph or sampler, per token | 0.04 ms | 0.04 ms | *device*, (`duration_ms` − `ttft_ms` − `decode_ms` − `sample_ms`) / 31, r25 and r26 |
| sampler + token-to-piece per token | 0.18 to 0.20 ms | 0.10 to 0.12 ms | *device*, `sample_ms` / 32 |
| decode graphs per run | 32 | 32 | *device*, `c2-io decode_graphs` |
| worker samples on cores 0-3 (runs with any) | gated 2 of 60, warm-ups 0 of 12 | gated 39 of 60, warm-ups 0 of 12 | *device*, `thread_samples`, r23 + r25 + r26 |
| hold CPU per gated run, median | r26 2.13 s, r25 0.39 s | r26 1.81 s, r25 0.58 s | *device*, `hold_stats.cpu_us` |
| hold: mincore wall per pass, median | 134 ms (r26) | 137 ms (r26) | *device*, `per_pass.mincore_ms` |
| hold: CPU per MB advised | 1.0 ms (r26), 1.2 (r25) | 0.7 (r26), 1.8 (r25) | *device*, passes with > 50 MB advised |
| `_residency` (whole-file mincore) wall per call | 96 to 114 ms | 108 to 127 ms | *device*, `res_before.ms` medians |
| wall time between runs not spent in the gate, hold or pause | r23 382 ms, r25 1,012, r26 412 (gated, median, both models) | | *device*, `idle_ms` − `cool_ms` − `hold_stats.final_ms` |
| `madvise(WILLNEED)` on a resident 900 KB file slice | 1.4 µs per call | | *host* (overlayfs, 512 MB file in cache) |
| create + join 2 threads | 45 µs per graph | | *host* |
| 60 `getenv` with 40 extra variables | 0.8 µs | | *host* |
| `getrusage(RUSAGE_SELF)` | 0.13 µs | | *host* |
| fopen + parse `/proc/self/io` | 4 to 6 µs | | *host* (unreadable in the app on the phone) |

The microbenchmark is a 60-line C file in the session's scratch folder, not in the repository. The host kernel's page
cache uses large folios and its `mincore` took 7 ms per 3 GB, against 110 to 135 ms on the phone. Host syscall costs
therefore do not scale to the phone by a fixed factor.

## 2. Cuts that could change tok/s or memory on the phone (b)

### 2.1 Gated runs lose the thread pin (rank 1)

- **Where.**
  - `lib/assistant.dart:552` `_pin()`, called from `lib/research.dart:331` (`_kitComplete`) just before
    `_engine.create`.
  - `lib/device.dart:814-828` `pinInferenceThreads`: `sched_setaffinity` on every thread named `DartWorker`, listed
    from `/proc/self/task` at that moment.
  - llamadart runs llama.cpp on its worker isolate (`third_party/llamadart/lib/src/backends/llama_cpp/llama_cpp_backend.dart:340`).
    ggml starts its workers from that thread for every graph (§2.4), so they inherit its mask.
- **Mechanism (*inferred*).**
  - A Dart isolate runs on whichever pool thread picks up its next message.
  - The Dart VM retires pool threads that are idle longer than its worker timeout (`worker_timeout_millis`, 5 s by
    default; Dart VM source, not checked in this Flutter version).
  - After a 90 s gate wait, the pinned threads are gone. The request message sent right after `_pin()` starts a new,
    unpinned `DartWorker`, and the prefill and the whole decode run on it.
  - In warm-ups (runs about 0.5 s apart) the pool threads survive, and all of them were pinned.
- **Evidence (*device*).**
  - `pinned_cpus` is `[4,5,6,7]` in every row.
  - Yet `thread_samples` shows running DartWorker threads on cores 0, 1, 2 or 3 in OLMoE gated runs: r23 13 of 16
    scored and 3 of 4 profiled, r25 5 of 16 and 3 of 4, r26 11 of 16 and 4 of 4. It shows none in any warm-up (0 of 4
    per session).
  - Granite shows it rarely (2 of 60 gated), because its samples mostly catch one thread (the sample is taken in the
    chunk callback, between graphs). That does not show that Granite is pinned.
- **Cost.** Not shown. Gated OLMoE runs with a full 3-thread sample, decode ms per token, with against without a
  little-core sample: r26 58.6 (n 11) against 44.9 (n 5); r25 47.7 (5) against 46.3 (11); r23 58.2 (6) against 58.4
  (2). This is confounded with residency, and one sample on a little core does not say how long the thread stayed
  there.
- **Reading for past rounds.** R60 to R66 gated runs ran partly without the pin that campaign 1's protocol meant to
  set. Warm-ups did not. Any "gated runs are slower" reading carries this confound as well as page loss.
- **Fix, runtime (preferred, model-generic).** A knob, for example `VOLLIE_CPUMASK=0xf0`, read in
  `ggml_graph_compute` (`ggml/src/ggml-cpu/ggml-cpu.c:6960`, next to the other per-graph `getenv`s).
  - When set, `sched_setaffinity(0, mask)` on the calling thread once per graph. The cost is one syscall of about 1 µs,
    which can be skipped if `sched_getaffinity` already matches.
  - The disposable workers then inherit the mask.
  - It needs a new patch in `patches/llama.cpp-c1-olmoe/`. Off when unset.
  - Alternative, app side: pin `gettid()` inside llamadart's worker isolate at the start of each generation (a
    llamadart patch). This covers the prefill as well, but it ties the fix to the binding.
- **Numerics.** None. Thread placement does not change the work split (same `nth`), so the change is bit-identical by
  construction. Check it with the usual per-chunk PPL at `-ub 1`, `-ub 32` and `-b 512`, knob unset and set.
- **Protocol.** It changes the run condition of every gated run. Pre-register it as a factor (ABBA per quad), or as a
  declared condition change, with the existing `thread_samples` as the check: no sample below core 4 with the knob on.
- **Effort.** Small: a patch of about 20 lines, host identity, one kit field.

### 2.2 A spinner animates during the whole kit (rank 2)

- **Where.**
  - `lib/settings_screen.dart:292-297`: the "Optimize" button shows a `CircularProgressIndicator` whenever
    `assistant.optimizing != null`.
  - `lib/assistant.dart:501-504` `_researchStatus` sets `optimizing` on every `show()` (`lib/research.dart:631-634`).
    It is reset only at the kit's end (`research.dart:1128`).
  - The owner starts the kit from Settings, so this screen is visible, kept on by `FLAG_KEEP_SCREEN_ON`, for 35 to 75
    minutes: the idle baseline, every gate wait and every decode.
  - Campaign 1's app has the same code (`vollie/vollie/lib/settings_screen.dart:295`, read only), so kit r16's 20.51
    tok/s was measured with it too.
- **Cost.**
  - Per frame: an indeterminate spinner rebuilds and repaints its card on the UI thread, rasterises on the raster
    thread, and keeps the GPU and the panel composing. That happens 60 or 120 times a second (the inner panel supports
    120 Hz).
  - These threads are not pinned. They can land on the big cores that the llama threads use.
  - *Device* lower bound: at least 0.55 to 0.69 s of process CPU per Granite run (78 to 112 ms per second of run) was
    spent outside the three llama threads (§1). That bound holds because the llama threads cannot use more than 3 ×
    wall. OLMoE's bound is smaller or negative where threads blocked on I/O.
  - The spinner's share of this is not known. Other candidates are the Dart GC and the chunk callbacks (§3.5).
  - *Estimated* power: 0.1 to 0.4 W of GPU, display and CPU over a static screen, during the cooling waits as well.
  - Headroom is a skin-temperature forecast, and the gate has plateaued near 0.72 since R60, so this is a candidate
    contributor to the gate timeouts (not shown).
- **Fix (app only).**
  - While `researching`, show a static status, with no progress indicator and no animation, on a dark background (OLED
    pixels off).
  - Update the text only between runs, which `show()` already does.
  - Optional, **owner decision**: dim the kit window's own brightness (`WindowManager.LayoutParams.screenBrightness`).
    This is app behaviour, not a phone setting, but it is a bench condition that users do not have.
- **Numerics.** None.
- **Protocol.** Changes heat and CPU state for every run. Pre-register it; it also moves away from kit r16's
  conditions.
  - Cleanest test: per block, alternating static and animated screens (G O O G, so one of each per model), with
    `idle_baseline`, the gate series and `run_cpu_ms − 3 × wall_ms` as the read-outs.
  - Prediction band (*estimated*): non-llama CPU per run −0.3 to −0.6 s. Headroom at the end of the wait −0.00 to
    −0.03. tok/s 0 to +0.5.
- **Effort.** Small: about 10 lines of Dart, plus one report field for the screen mode.

### 2.3 Expert read-ahead calls `madvise` for every routed slice of every token (rank 3)

- **Where.** `ggml/src/ggml-cpu/ggml-cpu.c:5832-5890` `ggml_cpu_expert_prefetch`. Thread 0 runs it before each layer's
  first `MUL_MAT_ID`; line 5886 issues one `madvise(MADV_WILLNEED)` per routed (and not skipped) expert slice of every
  expert weight (gate, up, down).
- **Count per token.**
  - OLMoE: 8 experts × 3 weights × 16 layers = 384.
  - Granite: 6 × 3 × 40 = 720 (off arm). The num arm skips about 40 % of pairs, which leaves about 430 calls.
- **Cost.**
  - *Host*: 1.4 µs per call on a resident 900 KB slice, so 0.53 ms per 384 calls.
  - Phone: unknown. Kernel 5.4's read-ahead walks the page cache page by page; the phone's whole-file `mincore` costs
    14 to 20 times the host's. So *estimated* 2 to 5 µs per call, that is 0.8 to 2 ms (OLMoE) and 0.9 to 3.6 ms
    (Granite) of thread-0 time per token.
  - Part of this is absorbed, because the other threads take thread 0's chunks of the op (dynamic chunking). *Estimated*
    wall cost: one third to all of it.
  - For scale: OLMoE's fault-free gap to 25 tok/s is about 2.7 to 3.2 ms (memo-24).
- **Why it exists.** At low residency it is essential: campaign 1 measured 6.1 → 11.8 tok/s for `EXPERT_IO` 0 → 1
  (`docs/runtime-knobs.md:11`). So it is not cut. It is made cheaper where it is redundant.
- **Fix (runtime, new knob, for example `GGML_EXPERT_IO` bit 16).** Skip a slice that this thread advised within the
  last N decode graphs (a 512-entry last-advised array per weight, N = 4 to 16).
  - Advice only, so bit-identical by construction.
  - A cheaper variant: one `madvise` per layer over the union of consecutive routed slices, merging adjacent ones. The
    read-ahead is the same; there are fewer syscalls.
- **Protocol.** It changes I/O behaviour, so it needs a device ABBA: per quad, bit 16 on against off. Read-outs: decode
  ms per token, decode majors and in-run read bytes, on both models. A loss at low residency is possible, so the kill
  rule must include "decode majors up".
- **Effort.** Small to medium: a patch, host identity, a kit factor.

### 2.4 A new worker pair for every graph (disposable threadpool) (rank 4)

- **Where.**
  - `ggml/src/ggml-cpu/ggml-cpu.c:7234-7237`: with `cplan->threadpool == NULL`, every `ggml_graph_compute` builds a
    threadpool (`pthread_create` × 2 at 3 threads) and frees it at the end.
  - llama.cpp passes its context threadpool, which is NULL unless `llama_attach_threadpool` was called
    (`src/llama-context.cpp:2735-2741`). llamadart never calls it (no use outside the generated bindings).
  - The app's own comment knows the effect ("ggml workers live for one graph", `lib/device.dart:282`).
- **Cost.**
  - *Host*: 45 µs per graph for create plus join of 2 threads.
  - Phone *estimated* 0.1 to 0.3 ms per token (thread creation, stack mapping, first scheduling).
  - Each worker also starts with no scheduler load history, which affects WALT's placement and clock choice. That is a
    possible extra cost, not shown: the profiled lines show the clocks at their caps, 2,112 / 2,035 MHz in every r26
    profiled token (*device*).
  - Minor faults per token are 124 to 610 in decode (*device*, `c2-io`); the host measured 0 minor faults per thread
    pair, so these come from model pages, not from stacks.
- **Fix (runtime, knob, for example `VOLLIE_THREADPOOL=1`).**
  - On the first graph, the CPU backend creates one persistent `ggml_threadpool` for `n_threads` and keeps it,
    recreating it when the count changes.
  - Set `poll` low (sleep between tokens: the sampler and llamadart take about 0.25 ms) and optionally
    `cpumask` / `strict_cpu` per worker. That is memo-24's per-thread pinning (Granite −0.3 to −1.0 ms, OLMoE −0.1 to
    −0.5, *estimated* there) and removes §2.1's root cause.
  - Same `nth`, same chunking, so bit-identical. Check it at `-ub 1`, `-ub 32` and `-b 512`.
- **Protocol.** A device factor (ABBA).
- **Effort.** Medium: lifecycle on unload and reload, prompt (`n_threads_batch`) against decode counts, a host identity
  check, and a test that the pool is freed with the context.

### 2.5 The OpenCL backend and the Adreno driver are loaded in a CPU-only kit

- **Where.**
  - `android/app/src/main/jniLibs/arm64-v8a/libggml-opencl.so` and `libvollie_clc.so` are in the APK.
  - `libggml-opencl.so` is mapped at every kit load (r26 `libs`, all four loads, *device*).
  - The settings screen's `FutureBuilder(future: assistant.gpuName())` (`lib/settings_screen.dart:278-279`) and the
    report header (`lib/assistant.dart:1110`) call `listGpuDevices(probeBackends: [opencl])`.
  - That reaches `ggml_opencl_probe_devices`, which calls `clGetPlatformIDs` and `clGetDeviceIDs` and so loads the
    vendor OpenCL stack into the process (*inferred*; the report's library list keeps only ggml/llama names).
- **Cost.** *Estimated* tens of MB of mapped driver code (clean file pages) and a few MB of anonymous driver state,
  possibly driver threads. Not measured: nothing in the reports splits the app's 131 to 147 MB anon remainder
  (memo-24 §A1).
- **Fix.** A kit build without the OpenCL library and without the GPU probe: `jniLibs` without the two files, the
  header without `gpuName()`, and the FutureBuilder skipped while `researching`.
- **Numerics.** None.
- **Protocol.** A condition change (app memory). Pre-register it as part of the next kit's state, together with §3.7's
  smaps split so that the saving is seen.
- **Effort.** Small.

### 2.6 `n_ctx` is not fixed by the kit

- **Where.**
  - `lib/assistant.dart:459` passes `contextSize`, which comes from `DeviceProfile.contextSizeFor(availableRam)`
    (`lib/device.dart:329`).
  - The result is 8192 when the phone has 7 GB or more and at least 3.5 GB available at the app's startup load,
    otherwise 4096. The kit's own loads reuse that value.
- **Device so far.** 4096 in r22, r23, r25 and r26 (`saved.context`), with MemAvailable 1.9 to 2.5 GB at the kit's
  start. 8192 would double OLMoE's KV to 570 MB, which matters whenever the kv knob is off.
- **What the run needs.** Prompt 156 (OLMoE) and 177 (Granite) tokens plus 32.
- **Fix (two steps).**
  - (i) Set the kit's context explicitly, as 4096 and recorded. This is instrument hygiene; it makes no change on the
    days measured so far.
  - (ii) Optionally 512 or 1024.
    - Memory: with `VOLLIE_KV_DONTNEED=1` on (R66) the untouched KV pages are not resident, so the RSS gain is small
      (*estimated* under 10 MB).
    - Without the knob it is about 250 MB of OLMoE memset per prompt.
    - The worst-case compute buffer shrinks too, but its untouched pages are not resident either (memo-24 §A1).
- **Numerics.** (ii) is expected to be identical: flash attention pads `n_kv` to 256, so a 210-token run sees 256 cells
  either way. This is *inferred* and must be checked (greedy texts and per-chunk PPL).
- **Protocol.** A pre-registered condition change.
- **Effort.** Small.

### 2.7 Granite's tied head in KleidiAI is unread in the num arm (memo-24 §A1, unchanged)

148 MiB of anon, of which Granite's VmSwap holds +107 to +122 MB in num (*device*, memo-24). This is already on
memo-24's shortlist (item 2). It is listed here only so that the memory items are in one place.

### 2.8 Sampler: the full vocabulary array is filled every token

- **Where.** llamadart's loop calls `llama_sampler_sample` once per token
  (`third_party/llamadart/lib/src/backends/llama_cpp/llama_cpp_service.dart:6910`). It fills a `llama_token_data`
  array of `n_vocab` entries (Granite 100,352 × 12 B), applies penalties (`patches/llama.cpp/0003` already avoids the
  per-candidate map lookups) and takes the greedy argmax.
- **Cost.** *Device*: `sample_ms` / 32 = 0.18 to 0.20 ms (Granite), 0.10 to 0.12 ms (OLMoE), token-to-piece included.
- **Fix.** At temperature 0, a direct argmax over the logits with the same penalty arithmetic on the at most `last_n`
  penalised tokens. Same comparisons, same tie rule (lowest id), so bit-identical if written carefully (the greedy
  texts check it). Off by default.
- **Gain.** *Estimated* 0.1 to 0.15 ms per token (0.2 to 0.3 %).
- **Effort.** Small to medium (the sampler chain composition must be matched exactly).

## 3. Cuts that only make the instrument cleaner (a)

Most of these run between runs, on the UI isolate, not on the llama cores. They do not move tok/s. They shorten the
0.4 to 1.0 s of bookkeeping between runs (*device*, §1) and remove noise. None changes numerics. A kit that adopts them
should list them in its pre-registration's "changes to the instrument", without a factor.

| # | where | what | cost (label) | fix | effort |
|---|---|---|---|---|---|
| 3.1 | `lib/settings_screen.dart:390`, getter `lib/assistant.dart:494-498` | `assistant.researchReport != null && !assistant.researching` evaluates the getter first. Once the in-progress report passes 300,000 characters (r26's is 445 k), every Settings rebuild runs `gzip` + `base64` over it. A rebuild follows every `show()`: each 5 s gate poll (about 19 per wait) and about 4 per run | *estimated* 5 to 20 ms of UI-thread CPU per rebuild, about 0.1 to 0.4 s per gate wait (in the cooling time) | swap the two operands, or cache the encoded string per save | trivial |
| 3.2 | `lib/research.dart:1022-1030` | after each run `pageVector(path)` (full `mincore`) and then `_residency(path)` (a second full `mincore`) | about 100 ms each (*device* `res_after.ms` 96 to 127) | compute `res_after` from `endVec` (the expert ranges are known) | small |
| 3.3 | `lib/research.dart:974` after the hold loop `:915-948` | `res_before` does a third full `mincore` right after the last hold pass, just before the run | about 100 ms (*device*), delays the run start | keep it (it is the record of "resident before the run") but count it from the same vector type; or drop the last hold pass's own scan | small |
| 3.4 | `lib/research.dart:782-831`, `lib/device.dart:157-195` | every hold pass maps the whole file and `mincore`s all 750 k pages, copies the vector, then loops over it 2 to 3 times in Dart | *device* 134 to 137 ms wall per pass, 4 to 5 passes per gated run on r26's day: about 0.6 s of the 1.8 to 2.1 s hold CPU. The rest is the kernel's read-ahead (0.7 to 1.0 ms CPU per MB advised) | `mincore` only the runs of wanted pages (Granite wants 2.1 to 2.3 GB of 3.16, OLMoE 2.6 to 2.7 of 3.07); stop the stats loops from re-walking `union` | small |
| 3.5 | `lib/research.dart:1000-1014` (chunk callback, inside the timed window) | at the first chunk: `vmstat()` (unreadable on this phone: `vmstat_*` null in r25/r26), `ioReadBytes()` (`/proc/self/io` unreadable), `memInfo()`, `rusage()`; then up to 3 `/proc/self/task` scans (2 file reads per thread) | *estimated* 1 to 5 ms per scan on a fourth core, about 10 to 20 ms per run | drop the two dead reads; keep `memInfo` and the thread sampler (it found §2.1) | trivial |
| 3.6 | `lib/research.dart:868` | sha256 of the 7 mapped native libraries at every block load (4 per session) | *estimated* 0.1 to 0.3 s and about 15 MB of reads per load | hash once per session (the files cannot change) | trivial |
| 3.7 | `lib/research.dart:980`, `lib/device.dart:612-623` | `zones_before`: 82 thermal-zone reads before every run; some zones sample ADCs | not measured | read a fixed short list (cpu, skin, battery zones); and add the missing anon split: `/proc/self/smaps` summed by mapping name once per block, between runs (*estimated* 20 to 50 ms per read), so that the 131 to 147 MB remainder and §2.5 can be attributed | small |
| 3.8 | `lib/device.dart:761, 766-774, 799-803` | `setNativeEnv`, `unsetNativeEnv`, `getNativeEnv` `malloc` C strings and never free them (about 75 unset + 10 set + 150 get-backs per run) | *estimated* under 1 MB per session | free after `unsetenv` / `getenv` (`setenv` copies its arguments) | trivial |
| 3.9 | `lib/research.dart:630, 641-671` | after every run: `jsonEncode` of the whole report into SharedPreferences (Android rewrites the whole preferences file); `harvest` re-reads the whole native log twice per run | *estimated* 10 to 50 ms and about 0.5 to 1 MB written per run | append rows to a file and keep only the final report in prefs; read the log from the saved offset | small |
| 3.10 | `ggml/src/ggml-cpu/ggml-cpu.c:6426, 6466-6467`, `4951-4990` | `VOLLIE_SKIP_STATS=1` (every Granite run) counts per `MUL_MAT_ID` on thread 0 before the op's barrier: two 512-byte arrays zeroed and a loop over `n_as` | *estimated* under 0.04 ms per token | set it only in the profiled runs (the per-prompt `c2-skip-run` line in scored runs is the only use) | trivial, kit env only |

## 4. Looks wasteful but stays (c)

- **The hold's read-back CPU.**
  - This is 1.8 to 2.1 s per gated run on r26's day (*device*), 0.4 to 0.6 s on r25's.
  - About two thirds is kernel read-ahead of pages that the wait took (0.7 to 1.8 ms per MB advised, *device*). That is
    the lever itself: +4.7 / +5.3 tok/s in R60. §3.4 cuts only the scanning share.
  - The minor faults that follow in the run (pages back in the cache but not mapped: `RssFile` 90 MB before against
    1.9 GB after the prefill in r26 pos 6, *device*) are the known price of advice without mapping. Kernel 5.4 has no
    `MADV_POPULATE_READ`.
- **The per-graph `getenv` block** (`ggml-cpu.c:6974-7119`, about 60 calls).
  - Host cost: 0.8 µs per graph.
  - Reading per graph is what lets the kit change knobs between runs without a reload. The app changes the env only
    between runs, so there is no concurrent `setenv`.
- **`GGML_C2_IO`'s sample after every graph** (`ggml-cpu.c:6165-6169`): one failing `fopen` of `/proc/self/io` on the
  phone plus two `getrusage`. Host: about 5 µs and 0.26 µs. The `c2-io` line is the decode-time source of every score.
- **The 32nd decode graph** (`llama_cpp_service.dart:6899-6950`).
  - llamadart decodes every sampled token, including the last, whose logits only feed the end-of-generation check
    after the loop.
  - The metric (`lib/assistant.dart:1162-1167`) divides 31 tokens by a window of 32 graphs (*device*: 32 decode graphs
    per run). Removing it would raise every number by about 3 % (*estimated*), and the metric is fixed by owner
    decision 3. Report it, do not cut it.
- **`EXPERT_IO` itself** (essential at low residency, §2.3). Only redundant calls are candidates.
- **The end-of-kit sha256 of both model files** (`research.dart:1107-1115`). It is after every timed run, and it is the
  evidence that the frozen file was served.
- **r16's gate polling** (every 5 s, thermal reads spaced at least 1.1 s) and **`FLAG_KEEP_SCREEN_ON`** (keeps the app
  in the foreground and at foreground OOM priority). Both are protocol.
- **The thread sampler** (§3.5) and the profiled runs' per-token lines (8 of 48 runs, unscored). Diagnostic value is
  shown: §2.1 came from them.
- **The KV clear line and the maps check** (`src/llama-impl.cpp`). The `/proc/self/maps` read is cached per base
  address. It costs microseconds to milliseconds in TTFT, never in the decode.
- **The other CPU variants in the APK** (1.6 MB each).
  - They are loaded, scored and unloaded once at engine start. Only `armv8.2_2` stays mapped (*device*, `libs`).
  - Dropping them gains nothing during a run. It would only matter if the variant scoring ever picked another module,
    which it does not on this SoC.

## 5. Not checked

- On the phone: any number in §2 except the counts and bounds read from the reports. In particular:
  - the cost per `madvise` call;
  - the cost of thread creation;
  - the spinner's CPU and power;
  - the OpenCL driver's memory;
  - whether Dart's worker timeout in this Flutter build is 5 s.
- Whether a worker on a little core stays there for a graph or only for a sample.
- Bit-identity of §2.1, §2.4, §2.6 (ii) and §2.8. Each is expected by construction and needs the usual check before a
  kit.

## 6. Suggested order (for the lead to decide)

1. §2.1 and §2.4 together as one runtime patch: a persistent pool with a cpumask, off unless set. It needs one ABBA
   factor and has a clear mechanism check (`thread_samples` on cores 4 to 7 only).
2. §2.2 static screen: one app change, tested per block. It also bears on the gate problem.
3. §2.3 read-ahead de-duplication as a second runtime factor, on both models.
4. All of §3 and §2.5 / §2.6 (i) as instrument changes in the same kit, declared in its pre-registration, with the
   smaps split (§3.7) so that §2.5's memory effect is visible.
