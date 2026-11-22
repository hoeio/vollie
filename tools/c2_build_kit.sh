#!/bin/sh
# Campaign 2: builds the research APK: release mode (AOT), Android debug key, package com.insaneutils.vollie.c2
# (installs beside the owner's Vollie; later kits install as updates over earlier ones and keep the model).
# Native side: llama.cpp's CPU module and libllama.so from <llama.cpp tree> via campaign 1's android/build_cpu.sh,
# swapped into the app copy.
# usage: tools/c2_build_kit.sh <llama.cpp tree> <app copy> [series dir ... (default patches/llama.cpp-c2)]
#   <llama.cpp tree>: 620d2225c with every patch of the series dirs applied in order (dir by dir); checked file by file here.
#   kit r20: tools/c2_build_kit.sh <tree> lab/app-c2 patches/llama.cpp patches/llama.cpp-c1-olmoe
#   kit r27: tools/c2_build_kit.sh lab/llama.cpp-c1r27 lab/app-c2 patches/llama.cpp patches/llama.cpp-c1-olmoe
#            (c1-olmoe 0001-0011; 0010 also touches common/common.cpp, a host-check knob not in the APK's libraries)
#   kit r28: tools/c2_build_kit.sh lab/llama.cpp-c1r28 lab/app-c2 patches/llama.cpp patches/llama.cpp-c1-olmoe   (c1-olmoe 0001-0012)
#   kit r28 reference APKs (memo-32): RUNTIME=c1 tools/c2_build_kit.sh lab/llama.cpp-r28c1 lab/app-c2 patches/llama.cpp
#            RUNTIME=stock tools/c2_build_kit.sh lab/llama.cpp-r28stock lab/app-c2 none   ("none": no series, 620d2225c as is)
#   RUNTIME (default c2) goes into kit_build.json ("runtime": the app's run list follows it) and, unless c2, into the APK
#   and BUILD file names (-research-<kit>-<runtime>).
#   <app copy>: app 2.3.0+45 with patches/app-c2/*.patch applied in order (APP_SERIES: the app series dirs recorded in
#   kit_build.json, default patches/app-c2; c3 kit r29: APP_SERIES="patches/app-c2 patches/app-c3").
#   kit r29 (c3 probe): APP_SERIES="patches/app-c2 patches/app-c3" tools/c2_build_kit.sh lab/llama.cpp-c3k lab/app-c3 patches/llama.cpp patches/llama.cpp-c1-olmoe patches/llama.cpp-c3
#   An APP_SERIES entry may also be one patch file (kit r32: app-c2 0001 to 0012 by name, since app-c2 0013 and 0014 are
#   not in the c3 app copy): APP_SERIES="$(cd patches/app-c2 && ls 00[01][0-9]-kit-r[12][0-9].patch | sed 's#^#patches/app-c2/#' | tr '\n' ' ')patches/app-c3"
#   tools/c2_build_kit.sh lab/llama.cpp-c3r32 lab/app-c3 patches/llama.cpp patches/llama.cpp-c1-olmoe patches/llama.cpp-c3
#   APP_ID_SUFFIX / APP_LABEL (default unset: the .c2 app, unchanged): build a separate app beside it, e.g. the r28 stock
#   reference after the phone refused it as an update (R68 session R): APP_ID_SUFFIX=.c2stock APP_LABEL="Vollie stock r28"
#   RUNTIME=stock ... ; build.gradle's suffix and the manifest label are changed for the build only and restored, the
#   kit_build.json gets "app_id", and the APK and BUILD names get -app. r28 stock app (with LIBS_FROM, step 2):
#   LIBS_FROM=<lib/arm64-v8a of the r28-stock APK> APP_ID_SUFFIX=.c2stock APP_LABEL="Vollie stock r28" RUNTIME=stock \
#     tools/c2_build_kit.sh lab/llama.cpp-r28stock lab/app-c2 none
#   EXTRA_RUNTIMES="<rt>=<apk> ..." (default unset: one runtime): kit r30's multi-runtime APK. Each <apk>'s libllama.so and
#   libggml-cpu-android_armv8.2_2.so are copied byte-identical into the app as libllama_rt_<rt>.so and
#   libggml_cpu_rt_<rt>.so (names llamadart's CPU pattern and ggml's load-all do not match; the app's Android side loads
#   them for runtime <rt>, app patch 0013) and removed again after the build; kit_build.json gets "runtimes" (per runtime
#   the file names and sha256 of both, and the source APK's own kit_build.json). kit r30 (c2 libraries from the r28 APK):
#   LIBS_FROM=<lib/arm64-v8a of the r28 APK> EXTRA_RUNTIMES="c1=lab/dist-c2/vollie-c2-1.28.0+28-research-r28-c1.apk
#     stock=lab/dist-c2/vollie-c2-1.28.0+28-research-r28-stock.apk" tools/c2_build_kit.sh lab/llama.cpp-c1r28 lab/app-c2 \
#     patches/llama.cpp patches/llama.cpp-c1-olmoe
# out: lab/dist-c2/vollie-c2-<version>-research-<kit>.apk and BUILD-<kit>.txt; the app's assets/kit_build.json
#      (llama.cpp commit, sha256 of every series patch, app patch and built library) goes into the kit's report.
set -eu
R=$(cd "$(dirname "$0")/.." && pwd); SRC=$(cd "$1" && pwd); APP=$(cd "$2" && pwd)
shift 2; [ $# -gt 0 ] || set -- "$R/patches/llama.cpp-c2"
RT=${RUNTIME:-c2}
PATCHES=; for d in "$@"; do [ "$d" = none ] && continue; d=$(cd "$d" && pwd); PATCHES="$PATCHES $(ls "$d"/0*.patch | tr '\n' ' ')"; done
M=$R/lab/device-bin/c2/app-module; O=$R/lab/dist-c2
V=$(sed -n 's/^version: //p' "$APP/pubspec.yaml"); KIT=$(sed -n "s/.*static const kit = '\(.*\)';/\1/p" "$APP/lib/research.dart")
grep -q "applicationIdSuffix \".c2\"" "$APP/android/app/build.gradle" || { echo "app copy lacks the .c2 suffix: apply patches/app-c2 first"; exit 2; }
grep -q "static const app = '$V';" "$APP/lib/research.dart" || { echo "pubspec version $V != Research.app"; exit 2; }
command -v ninja >/dev/null || PATH=$HOME/Library/Android/sdk/cmake/3.22.1/bin:$PATH

# 1. The tree must be exactly HEAD + the series: apply the series to a clean export of HEAD and compare every touched file.
X=$(mktemp -d); git -C "$SRC" archive HEAD | tar -x -C "$X"
for p in $PATCHES; do (cd "$X" && git apply "$p") || { echo "series patch does not apply: $p"; exit 1; }; done
files=$([ -z "$PATCHES" ] || cat $PATCHES | sed -n 's#^+++ b/##p' | sort -u)
[ "$files" = "$(git -C "$SRC" diff --name-only | sort -u)" ] || { echo "the tree's changed files differ from the series'"; exit 1; }
for f in $files; do cmp -s "$X/$f" "$SRC/$f" || { echo "tree differs from the series: $f"; exit 1; }; done
rm -rf "$X"

# 2. Libraries. LIBS_FROM=<dir> (default unset: build them): take both libraries from <dir> instead, e.g. unpacked from an
#    earlier APK of the same tree, so a rebuild is byte-identical (a fresh build differs in the 20-byte ELF build-id only).
if [ -n "${LIBS_FROM:-}" ]; then M=$(cd "$LIBS_FROM" && pwd); else sh "$R/android/build_cpu.sh" "$SRC" "$M"; fi
J=$APP/android/app/src/main/jniLibs/arm64-v8a; B=$APP/third_party/llamadart/third_party/bin/android-arm64
cp "$M/libggml-cpu-android_armv8.2_2.so" "$J/"; cp "$M/libllama.so" "$B/"
EXTRA=; XR=$(mktemp -d); G=$APP/android/app/build.gradle; MF=$APP/android/app/src/main/AndroidManifest.xml
cleanup() { [ -z "$EXTRA" ] || rm -f $EXTRA; rm -rf "$XR"; [ ! -f "$G.orig" ] || mv "$G.orig" "$G"; [ ! -f "$MF.orig" ] || mv "$MF.orig" "$MF"; }
trap cleanup EXIT
for e in ${EXTRA_RUNTIMES:-}; do
  rt=${e%%=*}; apk=${e#*=}; mkdir -p "$XR/$rt"
  unzip -o -q "$apk" lib/arm64-v8a/libllama.so lib/arm64-v8a/libggml-cpu-android_armv8.2_2.so assets/flutter_assets/assets/kit_build.json -d "$XR/$rt"
  cp "$XR/$rt/lib/arm64-v8a/libllama.so" "$J/libllama_rt_$rt.so"; cp "$XR/$rt/lib/arm64-v8a/libggml-cpu-android_armv8.2_2.so" "$J/libggml_cpu_rt_$rt.so"
  EXTRA="$EXTRA $J/libllama_rt_$rt.so $J/libggml_cpu_rt_$rt.so"
done

# 3. Build record for the report.
sha() { shasum -a 256 "$1" | cut -d' ' -f1; }
{ printf '{"kit": "%s", "app": "%s", "runtime": "%s", "llama_cpp": "%s",\n "series": [' "$KIT" "$V" "$RT" "$(git -C "$SRC" rev-parse HEAD)"
  sep=; for p in $PATCHES; do printf '%s\n  {"file": "%s", "sha256": "%s"}' "$sep" "$(basename "$p")" "$(sha "$p")"; sep=,; done
  printf '],\n "app_patches": ['; sep=
  for p in $(for d in ${APP_SERIES:-patches/app-c2}; do if [ -f "$R/$d" ]; then echo "$R/$d"; else ls "$R/$d"/0*.patch; fi; done); do printf '%s\n  {"file": "%s", "sha256": "%s"}' "$sep" "$(basename "$p")" "$(sha "$p")"; sep=,; done
  printf '],\n "libs": {"libggml-cpu-android_armv8.2_2.so": "%s", "libllama.so": "%s"}}\n' \
    "$(sha "$M/libggml-cpu-android_armv8.2_2.so")" "$(sha "$M/libllama.so")"; } > "$APP/assets/kit_build.json"
python3 -I -c 'import json,sys; json.load(open(sys.argv[1]))' "$APP/assets/kit_build.json"
if [ -n "${EXTRA_RUNTIMES:-}" ]; then
  python3 -I - "$APP/assets/kit_build.json" "$XR" "$J" $EXTRA_RUNTIMES <<'PY'
import hashlib, json, os, sys
kb, xr, j, extra = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4:]
sha = lambda f: hashlib.sha256(open(f, 'rb').read()).hexdigest()
b = json.load(open(kb))
rts = {b['runtime']: {'llama': {'file': 'libllama.so', 'sha256': b['libs']['libllama.so']},
                      'cpu': {'file': 'libggml-cpu-android_armv8.2_2.so', 'sha256': b['libs']['libggml-cpu-android_armv8.2_2.so']}}}
for e in extra:
    rt, apk = e.split('=', 1)
    rts[rt] = {'llama': {'file': f'libllama_rt_{rt}.so', 'sha256': sha(f'{j}/libllama_rt_{rt}.so')},
               'cpu': {'file': f'libggml_cpu_rt_{rt}.so', 'sha256': sha(f'{j}/libggml_cpu_rt_{rt}.so')},
               'apk': os.path.basename(apk), 'apk_sha256': sha(apk),
               'build': json.load(open(f'{xr}/{rt}/assets/flutter_assets/assets/kit_build.json'))}
b['runtimes'] = rts
json.dump(b, open(kb, 'w'), indent=1)
open(kb, 'a').write('\n')
PY
fi

# 4. APK. With APP_ID_SUFFIX the app copy's gradle and manifest are swapped for this build only (restored on exit).
if [ -n "${APP_ID_SUFFIX:-}" ]; then
  cp "$G" "$G.orig"; cp "$MF" "$MF.orig"
  sed -i '' "s/applicationIdSuffix \".c2\"/applicationIdSuffix \"$APP_ID_SUFFIX\"/" "$G"
  sed -i '' "s/android:label=\"Vollie C2\"/android:label=\"${APP_LABEL:-Vollie $APP_ID_SUFFIX}\"/" "$MF"
  grep -q "applicationIdSuffix \"$APP_ID_SUFFIX\"" "$G" && ! cmp -s "$MF" "$MF.orig" || { echo "app id or label swap failed"; exit 2; }
  sed -i '' "s/^{\"kit\"/{\"app_id\": \"com.insaneutils.vollie$APP_ID_SUFFIX\", \"kit\"/" "$APP/assets/kit_build.json"
  python3 -I -c 'import json,sys; json.load(open(sys.argv[1]))' "$APP/assets/kit_build.json"
fi
# Stale hook caches shipped an old libllama.so in campaign 1 (R39): wipe them and flutter clean.
rm -rf "$APP/.dart_tool/hooks_runner/shared/llamadart" "$APP/.dart_tool/hooks_runner/llamadart" "$APP/.dart_tool/flutter_build"
(cd "$APP" && flutter clean >/dev/null && LLAMADART_ALLOW_LEGACY_LOCAL_BUNDLES=1 flutter build apk --release)
A=$APP/build/app/outputs/flutter-apk/app-release.apk; X=$(mktemp -d)
for f in "$M/libllama.so" "$M/libggml-cpu-android_armv8.2_2.so" $EXTRA; do
  b=$(basename "$f"); unzip -o -q "$A" "lib/arm64-v8a/$b" -d "$X"
  cmp -s "$X/lib/arm64-v8a/$b" "$f" || { echo "MISMATCH: the APK's $b is not the one just built"; exit 1; }; done
unzip -o -q "$A" assets/flutter_assets/assets/kit_build.json -d "$X"
cmp -s "$X/assets/flutter_assets/assets/kit_build.json" "$APP/assets/kit_build.json" || { echo "MISMATCH: kit_build.json"; exit 1; }
rm -rf "$X"; mkdir -p "$O"; N=$KIT; [ "$RT" = c2 ] || N=$KIT-$RT; [ -z "${APP_ID_SUFFIX:-}" ] || N=$N-app; D=$O/vollie-c2-$V-research-$N.apk; cp "$A" "$D"
{ echo "apk $(basename "$D") $(wc -c < "$D" | tr -d ' ') B sha256 $(sha "$D")"
  cat "$APP/assets/kit_build.json"
  unzip -l "$D" | awk '$4 ~ /^lib\/arm64-v8a\// {print $1, $4}'; } > "$O/BUILD-$N.txt"
cat "$O/BUILD-$N.txt"
