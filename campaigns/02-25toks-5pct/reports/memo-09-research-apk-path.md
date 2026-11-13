# Memo 09: how a campaign-2 research APK gets built on the laptop

Agent model: claude-opus-5-5

Date 2026-11-14. Author: Tom (code). Read from the app source in the sibling folder `../vollie/vollie` (read-only;
not a git repository), its private lab `../vollie/lab`, and this repository. **Nothing was installed or run on a
phone, and nothing was sent anywhere. No speed is stated.** Times marked *estimated* are arithmetic on campaign
1's device reports.

**Short answer: yes, an APK can be built on the laptop today.** The toolchain built a debug APK of the unmodified
app end to end (§4). For r17 the app needs a code change: model, kit and side-by-side install. The CPU module needs
a new campaign-2 patch, because the knobs of `c2/0001` are read once per process and the app cannot switch them
between runs (§3).

## 1. Toolchain and signing

| item | state |
|---|---|
| Flutter | 3.41.2 stable, Dart 3.11.0 (Homebrew cask; the folder is still named 3.27.4 after an in-place upgrade) |
| Java used by Gradle | Android Studio's JBR 21.0.4 (`java` on the PATH is OpenJDK 11, not used) |
| Android SDK | platforms 28 to 36, build-tools 30.0.3 to 35.0.1, cmake 3.22.1 (with ninja); `flutter doctor`: some SDK licences not accepted (did not block the build) |
| NDK | 25.1.8937393 and **28.2.13676358** (campaign 1's) |
| adb | 36.0.2 (no use now) |
| llamadart native bundle | `v0.7.0-2` (= llama.cpp `620d2225c`) cached in the app's `third_party/llamadart/.dart_tool`, plus the local bundle in `third_party/llamadart/third_party/bin/android-arm64/` |

**Signing in campaign 1.** `android/key.properties` and `android/app/upload-keystore.jks` exist in the app
folder (contents not read). `build.gradle` signs release *and* debug builds with that key when `key.properties`
exists. Every campaign-1 APK in `../vollie/lab/dist/` checked here (r1, r16, production 2.3.0+45) has the same
signer, `CN=insaneutils`, certificate SHA-256 `171d1a93…3ce5a7`: the release (upload) key, not a debug key.

**applicationId** is `com.insaneutils.vollie` for every build. A research APK therefore **replaced** the owner's
installed Vollie as an update. That kept the 3 GB model and the app data, and needed a versionCode at least as high
as the installed one (r16 = 44, production 2.3.0+45 = 45). One risk is not settled by anything on the laptop: if the
phone now runs the Google Play build, and Play App Signing re-signs it with Google's key, an upload-key APK will
not install over it without an uninstall (which deletes the model). The owner can tell which build is installed.

**Recommendation for campaign 2: install beside, not over.** Add `applicationIdSuffix ".c2"` and its own label.
The manifest has no content providers, and the model sits in the app's own external files directory, so a second
package does not collide with Vollie. The c2 app downloads olmoe-Mi into its own directory, and the owner's Vollie and
its Granite file stay untouched. Then any key works. I suggest the debug key or a new campaign-2 key; neither the
upload keystore nor `key.properties` is copied into this repository or into `lab/app-c2/`. Installing still needs
the owner to allow installs from the browser or file app, as with the 16 campaign-1 APKs.

## 2. The build chain in campaign 1

1. **llama.cpp tree:** `620d2225c` + `lab/tiny-op.patch` (the 0001 series) + granite-rollback + sampler-bypass.
2. **Native libraries** (`../vollie/vollie/tool/build_cpu.sh`; the copy in this repository is
   `android/build_cpu.sh`): an NDK 28 build with `GGML_CPU_ALL_VARIANTS=ON GGML_BACKEND_DL=ON BUILD_SHARED_LIBS=ON`,
   KleidiAI ON, 16 KB pages. It builds two libraries:
   - `libggml-cpu-android_armv8.2_2.so` goes to `android/app/src/main/jniLibs/arm64-v8a/`. `pubspec.yaml`'s
     `cpu_variants` leaves `android_armv8.2_2` out of llamadart's list, so only this copy ships, and the SD888
     loads it (dotprod+fp16).
   - `libllama.so` goes into llamadart's "local bundle" `third_party/llamadart/third_party/bin/android-arm64/`
     (the cached upstream bundle with this one file swapped in; `patches/llamadart/0001` makes the directory take
     precedence; built with `LLAMADART_ALLOW_LEGACY_LOCAL_BUNDLES=1`).
3. **APK** (`../vollie/lab/logs/r48/freeze.sh`): set the kit constants, `dart analyze`, build the libraries,
   delete the three hook caches and run `flutter clean` (R39: otherwise a stale `libllama.so` ships), then
   `flutter build apk --release`. The script checks that the shipped `libllama.so` and module are byte-identical
   to the local build (`cmp` after `unzip`), and copies the APK to `lab/dist/` with its size and sha256.
4. **The kit is all Dart**, in `lib/research.dart` (an extension on `Assistant`): `kit`/`app` constants, the
   `served` env map, the `_knobs` list (reset between runs), `measure()` (cool-down gate, `mhz_before/after`,
   `mem_before/after`, `majflt/minflt` from `/proc/self/stat`, `io_read_bytes`, the `cpu-prof` lines read back
   from the stderr log file, `text`, llamadart's `sample_ms`/`decode_ms`), and the phase list. Env vars are set
   per run with `setNativeEnv` (`setenv`), without a reload. In `lib/assistant.dart`: `_benchPrompt`
   (line 1130; the same text as `tools/c2_official_prompt.txt`), `_benchTokens = 32`, `_complete()` (one user turn,
   `temp: 0`, llamadart's default repeat penalty 1.1), `decodeTokensPerSec` = (tokens − 1) × 1000 / (duration −
   ttft), `_coolDown()` (Android thermal headroom < 0.65, at most 90 s). The module's stderr goes to
   `vollie_load.log` (`DeviceProfile.stderrTo`, a `dup2`). The report is JSON on the clipboard plus a file
   `research-<kit>-<time>.json` beside the model; the schema is in `tools/research_kit.md`.
5. **Model on the phone:** `android/app/src/main/kotlin/.../ModelDownloadService.kt` hard-codes `URL` (a Hugging Face
   `resolve/<commit>/` URL), `FILE_NAME` and `SHA256`, and verifies the hash before the file is renamed into
   place. `lib/model_store.dart` holds `displayName`, `expectedBytes` (3,161,269,888) and `copyBytes`.

## 3. What r17 needs

**Granite assumptions in the app** (all read, none tested with OLMoE):

| place | assumption | r17 |
|---|---|---|
| `ModelDownloadService.kt` | URL, file name, SHA-256 of the Granite file | `https://huggingface.co/HoeioUser/vollie2/resolve/<commit>/olmoe-Mi.gguf`, sha256 `5abf710b…`, 3,072,399,552 B (pin the commit, as campaign 1 did) |
| `model_store.dart` | `displayName`, `expectedBytes`, `copyBytes` (700 MB of Granite repack) | OLMoE values. `useExtraBuffers` = `canCopyWeights(...)`: 7.3 GB − 3.07 GB < 5 GB, so **no repack**, the phone's path as in memo-05 (*estimated* from the rule) |
| `research.dart` `served` | `GGML_EXPERT_SKIP=117`, `GGML_HEAD2=2048`, `GGML_HEAD2_Q=q4_0`, `VOLLIE_SAMPLER_BYPASS=1` | empty map. These names do nothing in a c2 module, but they must not appear in a stock run's env |
| `research.dart` `_knobs`, `setTinyOps`/`setExpertIo` | campaign-1 knob names | the c2 knob list; the TINY_OP/EXPERT_IO setters are inert in a c2 module (leave them, or drop them for clarity) |
| `assistant.dart` `_userStart`/`_turnEnd` | Granite turn markup for system-prompt priming and prefill-while-typing | the OLMoE markup is `<|user|>` / `<|endoftext|>`. `_primeSystem` returns early when the marker is not found, so the kit is not affected; chat features degrade silently |
| `assistant.dart` `optimize()` | speed test sweeps TINY_OP and EXPERT_IO | not used by the kit; hide the button or leave it |
| chat template | `_template = null` = the GGUF's own | OLMoE's GGUF Jinja template (BOS + `<|user|>\n…\n<|assistant|>\n`). Whether llamadart renders it: **not verified**. The kit should log `prompt_tokens` (151 with llama.cpp's renderer on the host) |
| context | 4096 or 8192 by free memory | OLMoE has 16 attention layers: q8_0 KV ≈ 69 KB per token → 283 MB at 4096, 566 MB at 8192 (*estimated*); log `context` as the kit already does |

**Runtime: base + `patches/llama.cpp-c2/` only.** The c2 patch touches `src/llama-graph.cpp`, so **both**
`libllama.so` and the CPU module must come from the c2 tree. That drops campaign 1's sampler bypass and Granite
rollback from `libllama.so` (both unneeded: the bypass is a speed lever, off by default, and the rollback is 0).
Three things `c2/0001` cannot do inside an app:

1. **The knobs are read once per process** (a `static const` lambda in `build_moe_ffn`; first call in the CPU
   module). The app is one process, so whichever value the first graph sees holds for the whole kit, and a stock run
   and a skip run cannot share one APK session. Needed: wire `src[3]` whenever a load-time switch is set (say
   `VOLLIE_SKIP_WIRE=1`, so a stock load keeps the stock graph). The CPU module would then re-read SUM/MAX/MASS/MINK
   per graph, and threshold 0 is the stock arithmetic; that must be checked bit-identical at `-ub 1`, `-ub 32`,
   `-b 512` with the wire on and thresholds 0. About 40 lines.
2. **The `c2-skip:` stats line prints at process exit**, which never happens in the app. Needed: print and reset it
   per run (for example at each prompt graph while `VOLLIE_SKIP_STATS=1`), about 20 lines.
3. **No profiler and no in-run A/B.** `GGML_CPU_PROF`, `GGML_CPU_AB=knob:…`, `GGML_CPU_AB_TOKEN`, the per-token
   `cpu-prof ab= tok=` line (µs/token, majflt, minflt, cpu4/cpu7 MHz, op table) live in campaign 1's 0001
   (`ggml-cpu.c`, mixed into about 3,000 added lines with Granite-only kernels). A port needs: thread-0 per-op
   timing of one-token graphs, the A/B side counter (warm-up, ABBA blocks or per-token flips, the reset at each
   prompt), getrusage faults and the sysfs clock per window, the line printer, and a knob table with one entry so
   far (skip MASS, using item 1's per-graph re-read). About 300 to 400 lines as `c2/0002-profiler-ab.patch`, off
   unless set and checked bit-identical with the knobs off.

These patches are another Tom's area (`patches/llama.cpp-c2/`); nothing was changed there.

**Kit r17 phases (proposal; the lead writes the pre-registration):**

| phase | runs | env |
|---|---|---|
| (a) official 32, stock | 1 cold baseline + 10 plain | none (all knobs unset; load without `VOLLIE_SKIP_WIRE` = stock graph) |
| (b) profiled | 1 dropped warm-up story 200 + 1 reference story 200 + 2 to 4 profiled official 32 | `GGML_CPU_PROF=1 GGML_CPU_AB_TOKEN=1 GGML_CPU_AB=knob:SKIP_MASS:0:0` (A/A label) |
| (c) ABBA per lever | one block per bit-identical lever once ported; **none exists in c2 yet**, so empty | |
| (d) skip MASS=900 | reload with `VOLLIE_SKIP_WIRE=1`, then per-token A/B story runs `knob:SKIP_MASS:0:900` until ≥ 60 fault-free pairs (r15's rule), plus 4 plain official 32 at MASS=900 | `VOLLIE_SKIP_STATS=1` per run |
| (e) identity | `text` of every official run kept whole. Stock runs must agree with each other (and with the host's greedy 32 of the same file at `-nr`, if the lead wants that check). Skip runs are expected to differ | |

One reload sits between (a)/(b) and (d), because the wire is a load-time switch. The kit already reloads at its end.

**Kit run time on the phone (*estimated*).** r16 (Granite, 21 runs) took 22.6 min, mostly cool-down waits (60 to
90 s before most runs; one official run 5 to 12 s). r17 as above is about 30 to 35 runs plus one reload, so about
**25 to 50 min**, mostly cool-down. This is before the one-time 3.07 GB model download in the c2 app.

**Work (*estimated*):**

| item | files | lines |
|---|---|---|
| c2 patches: knob re-read + per-run stats; profiler/A/B port | `patches/llama.cpp-c2/0002…` (and an edit to 0001's knob reading, or a 0003) | 60 + 300 to 400, plus host identity checks |
| app copy `lab/app-c2/`: model constants | `ModelDownloadService.kt`, `model_store.dart` | about 10 |
| side-by-side package + signing | `android/app/build.gradle` (suffix, release `signingConfig` fallback to debug), `AndroidManifest.xml` label, `pubspec.yaml` version | about 6 |
| kit r17 | `research.dart` (served, knobs, phases, reload for the wire, per-run `c2-skip:` capture) | about 120 changed |
| build script for c2 (campaign 1's freeze.sh adapted: c2 tree, both libraries, cache wipe, `cmp` check, size and sha256) | new `lab/logs/r52/freeze.sh` or `tools/c2_freeze.sh` | about 40 |
| parser for the r17 JSON | new `tools/c2_kit_parse.py` (campaign 1's `research_parse.py` is not edited) | about 150 to 250 |

## 4. What was done here (*host*)

- **App copy:** `../vollie/vollie` is 1.6 GB. It was copied to `lab/app-c2/` (gitignored) without `build/`, the top
  `.dart_tool/`, the iOS/macOS/Windows/Linux/web folders, `android/key.properties`, `android/local.properties` or
  any `*.jks`. The copy is 560 MB. The original was not modified and no git command ran in it.
- **CPU module + libllama from base + c2/0001**, with campaign 1's `android/build_cpu.sh` (unchanged, KleidiAI ON as
  the app ships) on a private scratch worktree `lab/llama.cpp-c2-android` (`620d2225c` + `0001-olmoe-expert-skip.patch`,
  sha256 `62b02dc0…e9e42a`; the diff stat matches the patch, 138 insertions). Output `lab/device-bin/c2/app-module/`:

  | file | bytes | sha256 |
  |---|---|---|
  | `libggml-cpu-android_armv8.2_2.so` | 1,644,688 | `97b85b392269d633612398edde57f84503dd2111625c57861052912c4f394e80` |
  | `libllama.so` | 4,286,400 | `e2949cd98b346a2c5e1d6996b5dd399f4d5c1910fa77264ecbbf8c803513e1ed` |

  Both carry the `VOLLIE_SKIP_*` strings. They are not in any APK yet.
- **Debug APK of the unmodified app copy:** `LLAMADART_ALLOW_LEGACY_LOCAL_BUNDLES=1 flutter build apk --debug` in
  `lab/app-c2/` built in about 60 s of Gradle. `app-debug.apk` is 58,898,853 B, sha256
  `95f54a8758f6e93a90843975f1c4960be0f54fe15beebdb109ddc35335139cbd`, `com.insaneutils.vollie` versionCode 45,
  signed with the **Android Debug** key (so it would not install over the owner's app). It carries campaign 1's
  module and `libllama.so` (the copy's jniLibs and local bundle), not the c2 ones. This proves only that the
  toolchain works. A kit build must be `--release` (a debug build runs Dart in JIT mode and ships the Vulkan
  validation layer, which distorts the time outside the graph) with a signing config set, because without
  `key.properties` the copy's release build is left unsigned and cannot be installed.

**Not verified:** that the app loads OLMoE through llamadart, how llamadart renders OLMoE's template, any phone
behaviour, and the install path on the owner's phone (Play build or sideloaded).

Earlier adb work: [memo-06](memo-06-device-cli-kit.md), stopped.
