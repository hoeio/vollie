# Campaign 2 app patches (research kit)

The app source is not in this repository. These patches apply to a private copy of the Vollie app at version
**2.3.0+45** (the production build of 2026-11-08; the source folder is not a git repository, so the version is
identified by file hashes):

| file | sha256 before the patch |
|---|---|
| `pubspec.yaml` | `0b876797a6c7efcf246ad42e7f8e9f35cfabab0b9323d5f752648fa01d02f258` |
| `lib/assistant.dart` | `fa51cc6fa2b6702f39000ffbab43890eec3bec85e50d944a3de83caadbc17dd6` |
| `lib/research.dart` | `b445efbc5dd45d9e91529103c031cf1e4a96aab6242de1eca407dbd34bb8eb07` |

## 0001-kit-r17.patch

The kit r17 app (`com.insaneutils.vollie.c2`, version 1.17.0+17). Design and report schema:
`campaigns/02-25toks-5pct/reports/memo-10-kit-r17.md`.

- Package `com.insaneutils.vollie.c2`, label "Vollie C2": the app installs **beside** the owner's Vollie and never
  over it.
- Release build signed with the Android debug key.
- Model: `olmoe-Mi.gguf` from `HoeioUser/vollie2` at commit `e893e8a6`. The download is resumable and checks the
  sha256 before use; the storage check asks for about 3.2 GB.
- No Granite defaults. No skip 117, no two-stage head, no sampler bypass. No saved or fallback `GGML_TINY_OP` /
  `GGML_EXPERT_IO` is set at start. Every known knob is unset at every load and before every kit run.
- Weight repack is set per kit run (`repack` in the run spec); a change reloads the model.
- `lib/research.dart` is rewritten as a table-driven kit. Edit `Research.runs` to change what runs.

## 0002-kit-r18.patch (on top of 0001)

Kit r18, the lever kit (version 1.18.0+18). It keeps the package, the debug key and the model constants, so it installs as
an update over r17 and keeps the downloaded model. Design, run list and report schema:
`campaigns/02-25toks-5pct/reports/memo-12-kit-r18.md`.

- The run list of memo-12 (stock, stack, profiled, one in-run A/B per lever, skip, head2, package, repack off).
- The runtime's lines go to stderr (`VOLLIE_LINES=stderr`, always set). They are found anywhere in a log line, not only
  at its start, and are filed to the right run by their `run=` id (`VOLLIE_RUN` is set per run). A prompt's per-prompt
  lines arrive late (memo-11 §8); the last run's lines are flushed by freeing the context at the end of the kit.
- Per-token lines are kept as compact tuples (`tok_fields`). `profiled32` runs also keep the full lines.
- The env actually set is read back with `getenv`.
- The build record `assets/kit_build.json` (llama.cpp commit, sha256 of every series patch, app patch and built
  library) is written by `tools/c2_build_kit.sh` and goes into the report. The file is not in the patch.
- Reports over 300,000 characters are copied to the clipboard as `c2gz:` + base64(gzip(JSON)); the file is plain JSON.
- `test/kit_lines_test.dart`: line parsing (`flutter test test/kit_lines_test.dart`).

## 0003-kit-r19.patch (on top of 0002)

Kit r19 (version 1.19.0+19), campaign 1's runtime on OLMoE. Same package and debug key: it installs as an update over
r17/r18 and keeps the model. Native side: llama.cpp 620d2225c + `patches/llama.cpp/0001-0003` (campaign 1's series as
r16 shipped it), built with `tools/c2_build_kit.sh <tree> lab/app-c2 patches/llama.cpp`. Design, rows, schema:
`campaigns/02-25toks-5pct/reports/memo-14-kit-r19.md`.

- Run list: `cold`, then rows `c1_off` / `c1_ship` / `c1_ship_io` (official 32) in two passes (3 reps, then 2 reps
  plus one profiled run per row), all on the same CPU module, repack on.
- Cool-down gate rewritten: thermal reads are spaced at least 1.1 s apart (`DeviceProfile.thermal`), a null reading
  is read again and never counts as cool, the wait is 10 min before the kit's first timed run and 90 s otherwise, and
  every row keeps the headroom series.
- `GGML_REPACK_RELEASE=1` is set for the kit's own model load (forced once at the start), then unset.
- Residency per run: page-cache bytes of the model file (mincore over the app's own read-only mapping), split into the
  routed-expert tensors and the rest, before and after; clock caps (`scaling_max_freq` of policies 4 and 7).
- Campaign 1's per-token line (`cpu-prof ab= tok=`, r16 format) is kept as a compact tuple (`tok_format` `c1`,
  `tok_fields_c1`); profiled runs keep the full lines.
- `test/kit_lines_test.dart`: campaign 1's line, including a host line with `mhz=-1/-1`.

## 0004-kit-r20.patch (on top of 0003)

Kit r20 (version 1.20.0+20): campaign 1's kit r16 protocol on OLMoE. Same package and debug key: installs as an update
over r17 to r19 and keeps the model. Native side: llama.cpp 620d2225c + `patches/llama.cpp/0001-0003` +
`patches/llama.cpp-c1-olmoe/0001-keep-warm.patch`, built with
`tools/c2_build_kit.sh <tree> lab/app-c2 patches/llama.cpp patches/llama.cpp-c1-olmoe`. Design, protocol, run list,
schema: `campaigns/02-25toks-5pct/reports/memo-17-kit-r20.md`.

- Run list (48 runs): part 1 = kit r16's 21 runs in r16's order (baseline, two 200-token stories, groups 1-4 of plain /
  profiled / profiled / plain with r16's 20 s pause before groups 2 and 4, two plain runs), baseline configuration
  `c1_ship`; part 2 = r16's group 2 again with `GGML_KEEP_WARM=10`; part 3 = the warm block (no gate): warm-up + 10
  `c1_ship`, warm-up + 5 `EXPERT_IO=5`, warm-up + 5 `c1_off`.
- Gate as r16: below 0.65, read every 5 s, at most 90 s, also before the first run (r19's 10-minute first wait is gone).
  Kept from r19: thermal reads spaced 1.1 s, a null reading is read again. No end-of-run headroom read in gated runs
  (r16 had none); the warm block reads it once after each run and uses it as the next run's "before".
- No reload at the kit's start and no `GGML_REPACK_RELEASE` (r16 had neither); the model stays loaded as the app loaded it.
- r16's streaming for every run (llamadart's default batching); stories record `mhz_blocks` as r16 did.
- Rows carry `pos` (place in the list; part 1's pos k = r16's run k), `proto` (`r16`, `kw`, `warm`) and `phase`.
- Untagged runtime lines printed during a run's wait (the keep-warm passes) are filed into that run's `c2` list.
- `test/kit_lines_test.dart`: + the keep-warm line.

## Rebuild

```sh
# a private copy of the app (2.3.0+45) without build outputs or signing secrets, e.g. lab/app-c2/
for p in patches/app-c2/0*.patch; do git -C lab/app-c2 apply "$PWD/$p"; done
# llama.cpp 620d2225c + every patch of the series dirs in order, in a scratch tree, then:
tools/c2_build_kit.sh <llama.cpp tree> lab/app-c2 [series dir ...]   # default patches/llama.cpp-c2; checks the tree file by file
```

`tools/c2_build_kit.sh` builds the CPU module and `libllama.so` with `android/build_cpu.sh`. It copies them into the
app copy (`jniLibs` and llamadart's local bundle), wipes the hook caches, and runs
`flutter clean && flutter build apk --release`. It then checks that both libraries in the APK are byte-identical to
the ones just built, and writes the APK and `BUILD-<kit>.txt` to `lab/dist-c2/`.

Do not put a keystore, `key.properties` or `local.properties` into the app copy or into this repository.

## 0005-kit-r21.patch (on top of 0004)

Kit r21 (version 1.21.0+21), campaign 1's runtime + `patches/llama.cpp-c1-olmoe/0001-0002`. Same package and debug key:
installs as an update over r20 and keeps `olmoe-Mi`. Design, run list, schema and owner steps:
`campaigns/02-25toks-5pct/reports/memo-19-kit-r21.md`.

- **Model table** (`Research.kitModels`): id, file, URL, bytes, sha256, architecture, template source, repack rule and
  baseline settings per model. Downloads, the storage check, the blocks, the residency split (expert byte ranges per
  file) and the report are generated from it. Entries: `granite` (`granite-Mi.gguf`, fetched by the kit) and `olmoe`
  (the app's own download). A third model is one entry plus its host checks.
- **Second file:** "Download kit r21 models" fetches every missing kit file: Wi-Fi only, one storage check first,
  resumable (HTTP Range into a `.part`), sha256 checked before the rename; a `.verified` marker as the app's own
  download writes. In the app with the screen on (no background service).
- **Run list:** blocks in the table's order forward then backward (G O O G); each block loads its model (repack on,
  set explicitly), runs two unscored warm-ups, 8 scored official runs with `GGML_PREFILL_COLD=8` in ABBA, one profiled
  run per arm; r16's gate before every run but the warm-ups. Every run sets `GGML_C2_IO=1`. No skip, no head2.
- **Per row:** model, arm, loaded file, `repack_loaded`, gate series, caps, mincore experts/rest before and after (and of
  every file before a block's load), meminfo (MemAvailable, Cached, SwapFree, RssAnon, RssFile, VmSwap), faults,
  `/proc/self/io` read bytes over the run and to the first streamed chunk, the `c2-io` and `c2-pcold` lines, reply text,
  library hashes at each load. Report: every model's file, marker and end-of-kit sha256, each model's template render.
- **"Save research results to Downloads"** next to the copy button (MediaStore, Android 10+).

## 0006-kit-r22.patch (on top of 0005)

Kit r22 (version 1.22.0+22), campaign 1's runtime + `patches/llama.cpp-c1-olmoe/0001-0005`. Same package and debug key:
installs as an update over r21 and keeps both models. Design, run list, schema and owner steps:
`campaigns/02-25toks-5pct/reports/memo-20-kit-r22.md`.

- **Arms:** `off` = r21's knob-off arm; `num` = off + the model's `num` env from the model table (OLMoE
  `VOLLIE_SKIP_MASS=900 VOLLIE_HEAD2=1024`, Granite `GGML_EXPERT_SKIP=130 GGML_HEAD2=2048 GGML_HEAD2_Q=q4_0`, both
  `VOLLIE_SKIP_STATS=1`). `GGML_PREFILL_COLD` and `GGML_KEEP_WARM` are not used.
- **Hold** (app side, no env): at every gate poll and once after the gate, the model file's pages that were resident at
  the end of the previous run of that model and are not now are read back with `posix_fadvise(WILLNEED)`; after the
  gate it waits until 98 % are back or 5 s passed. Its passes, MB, CPU time, blocks read and ms after the gate are in
  `hold_stats`.
- **Run list:** blocks G O O G; per block 2 warm-ups, two ABBA quads (off num num off), the first with hold in odd
  blocks and without in even ones, then one profiled run per arm (with hold).
- **Per row:** + `arm`, `hold`, `hold_stats`, `want_mb`, `wait_lost_mb`, `inblock_read_bytes` (getrusage, also to the
  first chunk), `io_wait_ticks` (`/proc/self/stat` field 42), `run_cpu_ms`, and the runtime's `c2-io-blk`,
  `c2-skip-run`, `c2-head-run` lines. Report: `memlock_rlimit`, `hold_rule`, each model's `num`.

## 0007-kit-r23.patch (on top of 0006)

Kit r23 (version 1.23.0+23), campaign 1's runtime + `patches/llama.cpp-c1-olmoe/0001-0006`. Same package and debug key:
installs as an update over r22 and keeps both models. Design, run list, schema and owner steps:
`campaigns/02-25toks-5pct/reports/memo-22-kit-r23.md`.

- **Load env per model** (`KitModel.load`): OLMoE `VOLLIE_HEAD2_REPACK=1` during every load, in both arms (0006: the
  untied Q6_K head stays out of KleidiAI); Granite none. What native code saw is in the load row's `load_env_set`.
- **Hold in every gated run:** one pass in the wait (at least 45 s in), then a pass every 1 s after the gate until
  95 % of the wanted pages are back, a pass gains under 1 %, 4 passes or 8 s; no other `mincore`. `hold_stats` adds
  distinct and came-back MB, the stop reason and `per_pass`.
- **Factor threads:** per block one ABBA quad at 4 threads and one at 3 (4 first in odd blocks); warm-ups and profiled
  runs at 3. Per row `pinned_cpus`, `threads_loaded`, `thread_samples` (CPU of the running DartWorker threads).
- **Report:** `kernel_release`, `owner_note`, `thread_rule`, the new `hold_rule`, `load_env` per model. The start
  screen shows "close background apps (R60: a banking app was running)".

## 0008-kit-r24.patch (on top of 0007)

Kit r24 (version 1.24.0+24), runtime unchanged from r23 (`patches/llama.cpp-c1-olmoe/0001-0006`). Same package and
debug key: installs as an update over r23. Design, replay, schema and owner steps:
`campaigns/02-25toks-5pct/reports/memo-23-kit-r24.md`.

- **Gate rule per quad** (`RunSpec.gateRule`): `r16` (below 0.65 or 90 s) or `plateau` (also end at >= 30 s when the
  lowest of the last 2 readings is not more than 0.01 below the 2 before and at least 0.05 below the wait's highest);
  plateau first in odd blocks. Warm-ups `none`, profiled runs `r16`. Per row `gate_rule`, `wait_end`, `wait_ms`.
- **3 threads** in every run (R62 closed plain four-thread decode).
- **Charger guard:** a gate poll that finds a charger pauses the kit with a message until it is unplugged, then the
  wait restarts (`plug_pauses`); `plugged_after`, `plugged_run` per row.
- **Records:** idle baseline (30 s) and `/sys/class/thermal` zones at the start (errors kept), `zones_before`,
  `/proc/vmstat` swap and fault counters at start / first chunk / end, `mem_first`.

## 0009-kit-r25.patch (on top of 0008)

Kit r25 (version 1.25.0+25): r24's protocol unchanged plus one factor, kv. Runtime `patches/llama.cpp-c1-olmoe/0001-0008`
(0007 present and off: no `VOLLIE_PRIOR`; 0008 is the factor). Same package and debug key: installs as an update over r24.
Design, run list and owner steps: `campaigns/02-25toks-5pct/reports/memo-26-kit-r25.md`.

- **kv factor:** `VOLLIE_KV_DONTNEED=1` in half of the scored runs, alternated per run (odd blocks on off on off / off on
  off on over the two quads, even blocks the reverse); one kv run and one plain run per arm × gate rule cell. Never in
  warm-ups or profiled runs. Run id suffix `_kv`, spec field `kv`, report `kv_rule`.
- `VOLLIE_KV_DONTNEED` and 0007's `VOLLIE_PRIOR`, `_N`, `_J` added to the knobs cleared before every run.
- The runtime's `c2-kvclear` lines are tagged with `run=` and filed into the row's runtime lines like `c2-io-blk`.

## 0010-kit-r26.patch (on top of 0009)

Kit r26 (version 1.26.0+26). Runtime `patches/llama.cpp-c1-olmoe/0001-0009` (0009: the kv clear on the untagged
address, R64). Same package and debug key: installs as an update over r25 / r25b. Design, run list and owner steps:
`campaigns/02-25toks-5pct/reports/memo-27-kit-r26.md`.

- **One arm per model** (`Research.armOf`): Granite `num`, OLMoE `off`, in every run of their blocks (warm-ups, scored,
  profiled). OLMoE keeps its load env `VOLLIE_HEAD2_REPACK=1`, so its off arm is r23-r25's. Report `arm_rule`, `models.<m>.arm`.
- **r16's gate only** in every gated run (no plateau factor; the plateau code and its report fields stay, unused).
- **kv factor in two ABBA quads per block:** on off off on, off on on off in odd blocks, the reverse in even ones; 8 kv
  and 8 plain scored runs per model; never in warm-ups and profiled runs (two per block, the model's arm). `proto: r26`.
- Every recorded field of r25 is kept.

## 0011-kit-r27.patch (on top of 0010)

Kit r27 (version 1.27.0+27). Runtime `patches/llama.cpp-c1-olmoe/0001-0011` (0010: persistent pool + per-thread
mask; 0011: PROT_EXEC model mapping). Same package and debug key: installs as an update over r26. Design, run list,
storage step and owner steps: `campaigns/02-25toks-5pct/reports/memo-30-kit-r27.md`.

- **Storage step first** (`Research.kitStorageStep`, before the idle baseline): exec check on `filesDir/kit-models`
  (mountinfo row, then a 1 MiB test file mapped `PROT_READ`, `mprotect` to `PROT_READ|PROT_EXEC`, and mapped
  `PROT_READ|PROT_EXEC`; errno of each), the same probes on the external models folder and its model files (T2); only
  if the internal check passes: verify internal copies (sha256), one free-space check, copy from the external file
  (fsync, sha256) or download (Wi-Fi, resumable, sha256). The external files are never changed. Failed check: the
  session runs from the external files and the exec factor is void. Report `storage`.
- **Two factors per model as a 2x2 in every quad** (`Research.cells`, `quadOrders`: a Williams square): pin
  (`VOLLIE_POOL=1 VOLLIE_CPUMASK=0xf0`) and exec (`VOLLIE_EXEC_MAP=1`, applied through `llama_vollie_exec_sync` right
  after the run's env is set, before its gate wait). 8 vs 8 runs per factor per model. Warm-ups and profiled runs: pin
  on, exec off. Ids `<m>_<arm>_r16[_pin][_exec]`, `_wu`, `_prof`, `_profb` (T3: block-mode profile). `proto: r27`.
- **In every run:** `VOLLIE_KV_DONTNEED=1`, `VOLLIE_SCHED=1` (T1: per-thread schedstat and context switches in the
  runtime's `c2-sched` lines).
- **Instrument changes:** n_ctx fixed at 4096; static dark screen while the kit runs; no OpenCL libraries in the APK
  (gradle excludes); `res_after` from the run-end vector; no vmstat / `/proc/self/io` read inside the timed window;
  native library hashes once per session; native env strings freed; log read from the last byte offset. New records:
  `ru` (getrusage start / first chunk / end), `smaps_before` / `smaps_after`, `exec_sync`, `context_rule`, `pin_rule`,
  `exec_rule`, `design`, `instrument_changes`.
- New MethodChannel call `kitStorage` (MainActivity.kt): internal and external folders and their free bytes.

## 0012-kit-r28.patch (on top of 0011)

Kit r28 (version 1.28.0+28), memo-31's tests and tries plus the lead's additions of 2026-11-18. Design, run lists and
fields: `campaigns/02-25toks-5pct/reports/memo-32-kit-r28.md`.

- R68 run list on the c2 runtime (c1-olmoe 0001-0012): per model a claim-format block of 10 runs (pin + exec), then a
  block of four touch x nice quads (Williams rows), then OLMoE's num arm (MASS=900 + head 1024) as a claim-format block
  with touch and nice. Blocks O G O G O; two warm-ups per block; no profiled runs.
- Reference APKs from the same code (`RUNTIME=c1|stock tools/c2_build_kit.sh ...`): `assets/kit_build.json` carries
  `runtime`, and the run list follows it (10 claim-format runs per model, Granite then OLMoE).
- Touch calls (`llama_vollie_touch_begin` with the hold's wanted set before the gate wait, `llama_vollie_touch_end` after
  the hold), T-B per-run fields (`power`, `workers_after`, `proc_before` / `proc_after`), T-D smaps match by inode,
  session label (deep-sleeping list, required before the kit or the probe can start), power-save state at start and end
  (read only, warning when on), and T-A's cap probe (`capProbe`, six minutes, no model run, its own report and "Save cap
  probe to Downloads"). Kotlin: a read-only `powerState` channel method.

## 0013-kit-r30.patch (on top of 0012)

Kit r30 (version 1.30.0+30, owner 2026-11-20): one APK that re-runs all of R68, unattended. Same package
(`com.insaneutils.vollie.c2`) and debug key: installs as an update over r28 / r28-c1 / r29 and keeps the internal model
copies. Built with `tools/c2_build_kit.sh` and `LIBS_FROM` (r28's c2 libraries) plus `EXTRA_RUNTIMES` (r28-c1's and
r28-stock's `libllama.so` and CPU module, byte-identical, packed as `libllama_rt_<rt>.so` and `libggml_cpu_rt_<rt>.so`).

- **Runtime per process** (`MainActivity.preloadRuntime`, before Flutter starts): the stored choice (default c2). For c1 /
  stock, `System.loadLibrary("llama_rt_<rt>")` first (SONAME `libllama.so`, so libllamadart's DT_NEEDED and Dart's opens by
  name resolve to it) and `VOLLIE_CPU_MODULE=libggml_cpu_rt_<rt>.so`. c2 sets nothing: r28's loading path as it was.
- **llamadart** (`llama_cpp_service.dart`): with `VOLLIE_CPU_MODULE` set, the CPU backend candidates are exactly that file
  and `GpuBackend.auto` skips `ggml_backend_load_all_from_path` (it would also register the standard-named CPU module).
  Unset: unchanged.
- **Proof per load:** `Research.runtimeCheck` compares the mapped libllama / CPU module (`/proc/self/maps`, sha256) with
  `kit_build.json`'s `runtimes.<rt>` (R68 §0 / amendment 5); a mismatch stops the kit (row `runtime_check`, report
  `runtime_info`, `runtime_expected`). The run list follows the active runtime (R68's plans unchanged; rows keep proto `r28`).
- **Switch:** runtime chips in settings (store, then restart through `RestartActivity`, its own process; no permission).
- **Unattended chain** (Auto button): c2 plan, 30 min idle cool-down, restart into c1 and its reference plan, restart into
  stock and the same; state in `<files>/r30-chain.json` (resume at the first unfinished segment, a cut-off segment run
  again at most twice, partial report kept), log `<files>/r30-chain-log.txt` (restarts, mapped libraries per segment,
  battery % and charger per run), one report file per segment. Stops (recorded): battery below 15 %, mapped-library
  check failed, power saving on at a segment start after three re-checks 10 min apart, owner's Stop. End summary over
  every screen, "Save chain files to Downloads". Cap probes are not in the chain.
- Static kit screen moved into `KitScreen`, shown over every screen (main.dart) so a resumed chain has it.
- Session label preset "on" (the chain's default). `test/runtime_check_test.dart`.
- Test builds only: `--dart-define=VOLLIE_CHAIN_TEST=true` shortens the chain (a small model, 20 s cool-down) for the
  emulator; false and compiled out in the kit APK.
