# Kit r33 build (for R72 attempt 2): Tom, 2026-11-22

Kit r33 replaces kit r32 after R72 attempt 1 failed
(`../device/r72-attempt1-c3probe-r32-2026-11-22T1749.json`: finished in 5 s, `blocks: []`, `stopped_early: false`).
The kit is built. It is not uploaded and nothing is committed.

## 1. Cause

In `lib/research.dart`, `c3Probe()`, r32 added two local closures, `report()` and `save()`. They use `this`
(`_stopTest`, `_prefs`). The Dart VM keeps the captured variables of one scope in one shared context. So the closure
handed to the isolate, `Isolate.run(() => DeviceProfile.c3Probe(blk, dir))`, also reached `this` through its context
chain. `this` is the Assistant, which holds a ReceivePort. Sending it fails with "Illegal argument in isolate message:
object is unsendable".

The exception was thrown on the first block. It left the loop before `blocks.add`, and the `finally` saved an empty
report marked done. r29 worked because its `c3Probe()` scope had no closure that used `this`. The lead found this and
confirmed it with a host Dart repro. §3 repeats that repro.

## 2. Fix: `patches/app-c3/0003-kit-r33-probe.patch` (applied after 0002)

- **Static isolate entry points.** Both isolate calls in the app now go through static members of the `Research`
  extension. A static member has no `this`, so the closure captures only its parameters:
  - `_c3ProbeIso(String blk, String dir)` for the c3 probe;
  - `_spinIso(int mask, int untilMs)` for the cap probe's three spinners (`capProbe()`, the call that used to be
    research.dart:834).
- **The cap probe was not broken.** In `capProbe()` the only other closure is `mhz`, and it does not use `this`. The
  call is moved to a static helper anyway, so that a later edit cannot break it the same way. There are no other
  `Isolate.run`, `Isolate.spawn` or `compute` calls in `lib/`.
- **Errors are caught per block.**
  - The isolate call has its own try/catch. If it fails, the block's entry gets `error: "isolate: <message>"`, keeps
    its gate and power fields, and `VOLLIE_PROBE` is unset in a `finally`.
  - The whole block body (gate wait, power reads, probe) is wrapped too. Any other exception gives
    `{block, error: "block: <message>", gate}`.
  - In both cases the report is saved and the loop goes on to the next block, so a block can no longer disappear
    silently.
  - `tools/c3_probe_parse.py` already lists a block-level `error` under errors (its line 31), so the parser needs no
    change.
- **Version and names.** Kit `r33`, app `1.33.0+33` (versionCode 33, versionName 1.33.0). The button texts say r33.
  The file names follow `Research.kit`: `vollie-c3probe-r33-<time>.json` and `c3probe-r33-<time>.json`.
- **Unchanged.** The block list and the probe design are r32's.

## 3. Checks (all *host*, the owner's laptop)

- **Isolate repro.** The script is the scratchpad file `iso_test.dart` (path in the hand-off). It is a stand-alone
  Dart VM program with the same shape as the app: an `Assistant` class with a ReceivePort, and a `Research` extension.
  `dart run` gives:
  - **r32 pattern:** a local closure that uses `this`, with `Isolate.run` in the same scope. It throws
    `Illegal argument in isolate message: object is unsendable ... <- port in Instance of 'Assistant' <- Context ...
    <- Closure`.
  - **r33 pattern:** the static helper plus the per-block catch. Both blocks return their results.
  - **r32 pattern with only the per-block catch:** the block is kept, with `error: "isolate: ... unsendable ..."`.
  - The script prints `PASS`.
- **`flutter analyze`** on `lib/research.dart`, `lib/settings_screen.dart` and `lib/assistant.dart`: no issues found.
- **The patch applies.** `git apply` on a copy of the r32 sources gives files identical to the built app copy, and the
  reverse check passes on the app copy.
- **Build.** The build line is in `patches/app-c3/README.md` (0003). The run used `LIBS_FROM` = the r32 APK's
  `lib/arm64-v8a`. `tools/c2_build_kit.sh` checked that `lab/llama.cpp-c3r32` is exactly 620d2225c plus the series,
  and that the APK carries the libraries it was given.
- **Runtime identity with r32.** All 13 native libraries were unpacked from both APKs and compared with `cmp`. Twelve
  are byte-identical: libllama.so, every libggml-cpu-*.so, libggml-base, libggml, libllama-common, libllamadart,
  libmtmd, libflutter and libdatastore_shared_counter. Only `libapp.so` (the Dart AOT code) differs. So there is no
  runtime change, and the knob-unset identity checked for r32 (tom-r32-build.md) carries over unchanged.
- **aapt badging:** `package: name='com.insaneutils.vollie.c2' versionCode='33' versionName='1.33.0'`.
- **Not done.** Nothing ran on the phone. That r33 runs every block on the device is still to be shown by the owner's
  run.

## 4. Artefacts

| part | file | size | sha256 |
|---|---|---|---|
| app patch | `patches/app-c3/0003-kit-r33-probe.patch` (159 lines; changes research.dart, settings_screen.dart, pubspec.yaml) | | `0ee03f5aa15e5a4934f09b52c1c20a4aefb4bb905e7ae601e5e063aa6c93aa1c` |
| APK | `lab/dist-c3/r33/vollie-c3-probe-r33.apk` (build-time name `vollie-c2-1.33.0+33-research-r33.apk`) | 16,506,302 B | `1917aa4cc2b28918f21b36f74a80fb2a927f711051c501ce5550bbd38a97db7c` |
| BUILD file | `lab/dist-c3/r33/BUILD-r33.txt` (the APK's `kit_build.json` plus the library list) | | `d227277778a28e5b36a2f765403bae2f374e0ab8faabd1536b1db044189084df` |
| libllama.so | in the APK, same as r32 | 4,492,920 B | `350b57c3fd3cfd2b4018b5a5bc56dbb8695415ba2bac599740ad64f571134e63` |
| libggml-cpu-android_armv8.2_2.so | in the APK, same as r32 | 1,773,736 B | `46e0735d2824ba6f3a156e219e058a701f6024fdbecbb2ef937b3127115a9694` |

`lab/dist-c3/r33/SHA256SUMS` lists the APK and the BUILD file. The upload layout is `r33/vollie-c3-probe-r33.apk`,
`r33/BUILD-r33.txt` and `r33/SHA256SUMS`. It waits for the lead: the R72 pre-registration names kit r32, so the lead
decides whether attempt 2 on r33 needs a dated note in the pre-registration first.
