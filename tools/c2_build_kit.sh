#!/bin/sh
# Campaign 2: builds the research APK: release mode (AOT), Android debug key, package com.insaneutils.vollie.c2
# (installs beside the owner's Vollie; later kits install as updates over earlier ones and keep the model).
# Native side: llama.cpp's CPU module and libllama.so from <llama.cpp tree> via campaign 1's android/build_cpu.sh,
# swapped into the app copy.
# usage: tools/c2_build_kit.sh <llama.cpp tree> <app copy> [series dir (default patches/llama.cpp-c2)]
#   <llama.cpp tree>: 620d2225c with every patch of the series applied in order; checked file by file here.
#   <app copy>: app 2.3.0+45 with patches/app-c2/*.patch applied in order.
# out: lab/dist-c2/vollie-c2-<version>-research-<kit>.apk and BUILD-<kit>.txt; the app's assets/kit_build.json
#      (llama.cpp commit, sha256 of every series patch, app patch and built library) goes into the kit's report.
set -eu
R=$(cd "$(dirname "$0")/.." && pwd); SRC=$(cd "$1" && pwd); APP=$(cd "$2" && pwd)
SER=$(cd "${3:-$R/patches/llama.cpp-c2}" && pwd)
M=$R/lab/device-bin/c2/app-module; O=$R/lab/dist-c2
V=$(sed -n 's/^version: //p' "$APP/pubspec.yaml"); KIT=$(sed -n "s/.*static const kit = '\(.*\)';/\1/p" "$APP/lib/research.dart")
grep -q "applicationIdSuffix \".c2\"" "$APP/android/app/build.gradle" || { echo "app copy lacks the .c2 suffix: apply patches/app-c2 first"; exit 2; }
grep -q "static const app = '$V';" "$APP/lib/research.dart" || { echo "pubspec version $V != Research.app"; exit 2; }
command -v ninja >/dev/null || PATH=$HOME/Library/Android/sdk/cmake/3.22.1/bin:$PATH

# 1. The tree must be exactly HEAD + the series: apply the series to a clean export of HEAD and compare every touched file.
X=$(mktemp -d); git -C "$SRC" archive HEAD | tar -x -C "$X"
for p in "$SER"/0*.patch; do (cd "$X" && git apply "$p") || { echo "series patch does not apply: $p"; exit 1; }; done
files=$(cat "$SER"/0*.patch | sed -n 's#^+++ b/##p' | sort -u)
[ "$files" = "$(git -C "$SRC" diff --name-only | sort -u)" ] || { echo "the tree's changed files differ from the series'"; exit 1; }
for f in $files; do cmp -s "$X/$f" "$SRC/$f" || { echo "tree differs from the series: $f"; exit 1; }; done
rm -rf "$X"

# 2. Libraries.
sh "$R/android/build_cpu.sh" "$SRC" "$M"
J=$APP/android/app/src/main/jniLibs/arm64-v8a; B=$APP/third_party/llamadart/third_party/bin/android-arm64
cp "$M/libggml-cpu-android_armv8.2_2.so" "$J/"; cp "$M/libllama.so" "$B/"

# 3. Build record for the report.
sha() { shasum -a 256 "$1" | cut -d' ' -f1; }
{ printf '{"kit": "%s", "app": "%s", "llama_cpp": "%s",\n "series": [' "$KIT" "$V" "$(git -C "$SRC" rev-parse HEAD)"
  sep=; for p in "$SER"/0*.patch; do printf '%s\n  {"file": "%s", "sha256": "%s"}' "$sep" "$(basename "$p")" "$(sha "$p")"; sep=,; done
  printf '],\n "app_patches": ['; sep=
  for p in "$R"/patches/app-c2/0*.patch; do printf '%s\n  {"file": "%s", "sha256": "%s"}' "$sep" "$(basename "$p")" "$(sha "$p")"; sep=,; done
  printf '],\n "libs": {"libggml-cpu-android_armv8.2_2.so": "%s", "libllama.so": "%s"}}\n' \
    "$(sha "$M/libggml-cpu-android_armv8.2_2.so")" "$(sha "$M/libllama.so")"; } > "$APP/assets/kit_build.json"
python3 -I -c 'import json,sys; json.load(open(sys.argv[1]))' "$APP/assets/kit_build.json"

# 4. APK. Stale hook caches shipped an old libllama.so in campaign 1 (R39): wipe them and flutter clean.
rm -rf "$APP/.dart_tool/hooks_runner/shared/llamadart" "$APP/.dart_tool/hooks_runner/llamadart" "$APP/.dart_tool/flutter_build"
(cd "$APP" && flutter clean >/dev/null && LLAMADART_ALLOW_LEGACY_LOCAL_BUNDLES=1 flutter build apk --release)
A=$APP/build/app/outputs/flutter-apk/app-release.apk; X=$(mktemp -d)
for f in "$M/libllama.so" "$M/libggml-cpu-android_armv8.2_2.so"; do
  b=$(basename "$f"); unzip -o -q "$A" "lib/arm64-v8a/$b" -d "$X"
  cmp -s "$X/lib/arm64-v8a/$b" "$f" || { echo "MISMATCH: the APK's $b is not the one just built"; exit 1; }; done
unzip -o -q "$A" assets/flutter_assets/assets/kit_build.json -d "$X"
cmp -s "$X/assets/flutter_assets/assets/kit_build.json" "$APP/assets/kit_build.json" || { echo "MISMATCH: kit_build.json"; exit 1; }
rm -rf "$X"; mkdir -p "$O"; D=$O/vollie-c2-$V-research-$KIT.apk; cp "$A" "$D"
{ echo "apk $(basename "$D") $(wc -c < "$D" | tr -d ' ') B sha256 $(sha "$D")"
  cat "$APP/assets/kit_build.json"
  unzip -l "$D" | awk '$4 ~ /^lib\/arm64-v8a\// {print $1, $4}'; } > "$O/BUILD-$KIT.txt"
cat "$O/BUILD-$KIT.txt"
