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
