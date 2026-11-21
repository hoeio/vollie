# Production app patches (Play Store)

The app source is not in this repository. These patches apply to a private copy of the Vollie app at version
**2.3.0+45** (the production build of 2026-11-08, campaign 1's shipped app; the source folder is not a git repository,
so the version is identified by file hashes). They do not build a research kit: the package stays
`com.insaneutils.vollie`, signed with the app's own release key, so the result installs as an update on Play.

| file | sha256 before the patch |
|---|---|
| `pubspec.yaml` | `0b876797a6c7efcf246ad42e7f8e9f35cfabab0b9323d5f752648fa01d02f258` |
| `android/app/src/main/AndroidManifest.xml` | `a078cd8a58fba01c85c136feeb36b431558769dd41ae02fc45483e6e9a0f6ff3` |
| `android/app/src/main/kotlin/com/example/vollie/MainActivity.kt` | `e36cd546ab24441636f2e375ba83e3003f6f6f258b86432e757d84b616807641` |
| `android/app/src/main/kotlin/com/example/vollie/ModelDownloadService.kt` | `0e3fe606b00fcedc96bfad5fcecac6e983490a543b4d6b3f2dc1b302169d94a5` |
| `lib/assistant.dart` | `fa51cc6fa2b6702f39000ffbab43890eec3bec85e50d944a3de83caadbc17dd6` |
| `lib/device.dart` | `3b9e797728b6c832d97a04c4bb2f580c14fa6f6a50ddf82d73a7c49b91725462` |
| `lib/main.dart` | `5b8b4302c473c084b07f22d39ac667c4732cb277ca908f86bde8b02d54c570d6` |
| `lib/model_store.dart` | `9b515f3c59f8853003d735f2c39a37e5603c26a9e8c3753913748df9a5ceb617` |
| `lib/research.dart` (deleted) | `b445efbc5dd45d9e91529103c031cf1e4a96aab6242de1eca407dbd34bb8eb07` |
| `lib/settings_screen.dart` | `aebe1d86ab4567a3c7431d82ad3858fc63c1975279f80382566dd9cac7777e8a` |
| `lib/setup_screen.dart` | `682e23fe542699beace38349c03ea731cd75e91953657cd360cc49f511cc72f3` |
| `test/widget_test.dart` | `bbed9aa0744ed5566902a57957882fca114e558cef03c61387cec71ce4fdff12` |
| `third_party/llamadart/lib/src/backends/llama_cpp/llama_cpp_service.dart` | `49a61c85c73dc89f12e0d73a6807e3c54bcae8ca24a52117a288baf1b8001e7e` |

New files: `lib/engine.dart`, `RestartActivity.kt`, `android/app/src/test/.../EngineTest.kt`, `test/engine_test.dart`.

## 0001-vollie-3.0.0.patch

Vollie **3.0.0+46**: campaign 2's result for users, with campaign 1 kept as a choice. Apply with
`git apply -p1` in the app copy's root (checked: it reproduces the built copy byte for byte).

### Two engines, one per process

- Settings > Engine: **Vollie 2** (default) or **Vollie 1 classic**. A switch asks for confirmation, pauses a running
  download, stores the choice and restarts the app (`RestartActivity` in its own `:restart` process ends the old
  process and starts `MainActivity`; the same mechanism as kit r30/r31). The kill of the old process is not reported
  as a crash (a SIGKILL within 15 s of the stored switch time is ignored).
- The engine is fixed before Dart starts: `MainActivity.preloadEngine` runs before `super.onCreate`.
  - **Classic** loads `libllama_rt_c1.so` (2.3.0+45's own `libllama.so`, renamed; its SONAME stays `libllama.so`, so
    llamadart's later `dlopen("libllama.so")` finds it already loaded) and, only where ggml would pick the
    `android_armv8.2_2` CPU variant (cpuinfo has `asimddp` and `asimdhp`, not `i8mm`), sets
    `VOLLIE_CPU_MODULE=libggml_cpu_rt_c1.so` (2.3.0+45's own 8.2_2 module, renamed). On other phones the stock
    variants load as in 2.3.0+45 (they are byte-identical to 2.3.0+45's).
  - **Vollie 2** loads nothing early: the standard `libllama.so` and `libggml-cpu-android_armv8.2_2.so` are kit r31's
    c2 libraries (= r30 = r28 c2), byte-identical.
  - llamadart: `VOLLIE_CPU_MODULE` restricts the CPU candidates to that one file and skips load-all (kit r30's change).
- Classic is 2.3.0+45's path: its libraries, its model file in its old place (`getExternalFilesDir`, same URL and
  sha256 as before), its saved performance settings and its served env at every load (`GGML_EXPERT_SKIP=117`,
  `GGML_HEAD2=2048`, `GGML_HEAD2_Q=q4_0`, `VOLLIE_SAMPLER_BYPASS=1`; tiny-op and expert I/O from the saved settings,
  as 2.3.0+45's `Research.applyServed` did).

### Vollie 2: models and settings

- First run: the user picks **OLMoE 1B-7B** (`olmoe-Mi.gguf`) or **Granite 4.0 H Tiny** (`granite-Mi.gguf`), both from
  `HoeioUser/vollie2` at commit `e893e8a65cff2f09f3de45e4840537c625806e0b`, sha256 checked after the download
  (`5abf710b...444abb` and `8df94de7...0f45`). Settings > Model switches later (download if needed, then reload);
  other downloaded models are listed with a delete button.
- Vollie 2 models go to internal storage (`filesDir`): the executable model mapping is refused on external storage
  (R67). A user of 2.3.0+45 keeps the classic file; "Keep using Vollie 1 classic" on the first-run screen avoids a
  second download.
- Fixed for every phone: CPU only, 3 threads (or fewer on phones with fewer fast cores), context 4,096 tokens
  (OLMoE's training length; kit r31's `n_ctx`), weight repack as the app's rule had it. The performance card shows
  this and hides the optimise/threads/compute controls.
- Native settings (`Engines.v2Env`, `lib/engine.dart`); every Vollie 2 name is unset before each load:

| | OLMoE (R70 arm b5) | Granite (R71 S1 arm b2) |
|---|---|---|
| both | `GGML_TINY_OP=262144` `GGML_EXPERT_IO=1` `VOLLIE_SAMPLER_BYPASS=1` `VOLLIE_KV_DONTNEED=1` `VOLLIE_POOL=1` `VOLLIE_CPUMASK=<fast cores>` `VOLLIE_EXEC_MAP=1` | same |
| numerics | `VOLLIE_SKIP_MASS=900` `VOLLIE_HEAD2=1024` `VOLLIE_HEAD2_REPACK=1` | `GGML_EXPERT_SKIP=130` `GGML_HEAD2=2048` `GGML_HEAD2_Q=q4_0` |
| other | `VOLLIE_TOUCH=1` `VOLLIE_TOUCH_S=12` `VOLLIE_NICE=-10` | |

  `VOLLIE_CPUMASK` is the fast cores as a hex mask (`0xf0` on an SD888); unset when none are known. The CPU-module
  knobs (skip, head2, pool, nice, tiny-op, expert I/O) act only where the c2 8.2_2 module is chosen; the libllama knobs
  (KV clear, exec mapping, touch) act everywhere.
- OLMoE turn markup (Tulu): user turn `<|user|>` ... `\n`, reply end `<|endoftext|>`. Granite's is 2.3.0+45's.

### Touch and hold outside a gate

The kit ran touch and hold around a cool-down gate. In the app:

- **Touch** (OLMoE only): `llama_vollie_touch_begin` when a load or a reply ends, `llama_vollie_touch_end` before the
  next reply, a reload, an unload, or when the app leaves the screen (`AppLifecycleListener`); calls are serialized on
  helper isolates. It never runs in the background. The end's last pass costs about 0.2 s before a reply (*device*,
  R71 S1 c2-touch lines: n 26, median 219.9 ms).
- **Hold** is dropped: in chat there is no gate wait to bridge, and the touch's last pass re-reads the same pages.

### Removed from 2.3.0+45

The phone test, the research buttons, `lib/research.dart` and its report code. Kit instrumentation is never set:
`GGML_C2_IO`, `VOLLIE_SKIP_STATS`, `VOLLIE_SCHED`, `VOLLIE_LINES`, `VOLLIE_RUN`. The kit's gate, cool-down and
reporting are not carried over.

### Smoke hook (not in a release build)

`--dart-define=VOLLIE_SMOKE_MODEL=FILE` makes both engines load `FILE` from the classic folder and print one
`vollie-smoke` line (engine, threads, context, touch, 8 tokens, CPU module, mapped libraries). Without the define,
the code compiles out; the build script checks that the release `libapp.so` has none of these strings.

## Build

```
tools/prod_build_aab.sh <app copy> <2.3.0+45 source folder> <2.3.0+45 AAB> <kit r31 APK>
```

The script checks the four runtime libraries against their sha256 records, places them, links (never copies) the
release key files from the 2.3.0+45 folder for the build and removes the links afterwards, builds the bundle, checks that
the libraries in it are the ones placed, that every `.so` is 16 KB aligned and that the signing certificate equals
2.3.0+45's, and writes `lab/dist-prod/vollie-3.0.0+46.aab` and `lab/dist-prod/BUILD-3.0.0+46.txt`.
`SMOKE=FILE` builds an emulator smoke APK (same key) into `lab/dist-prod/smoke/` instead.

Checks: `flutter analyze lib test`, `flutter test`, `./gradlew :app:testDebugUnitTest` (EngineTest: the classic CPU
module rule), bundletool `build-apks` + `install-apks` on an arm64 emulator, and the smoke APK for both engines and
both switch directions.
