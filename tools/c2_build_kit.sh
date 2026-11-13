#!/bin/sh
# Campaign 2: builds the research APK (kit r17): release mode (AOT), Android debug key, package com.insaneutils.vollie.c2
# (installs beside the owner's Vollie). Native side: llama.cpp's CPU module and libllama.so from <llama.cpp tree>
# (620d2225c + patches/llama.cpp-c2/* only) via campaign 1's android/build_cpu.sh, swapped into the app copy.
# usage: tools/c2_build_kit.sh <llama.cpp tree> <app copy with patches/app-c2/0001-kit-r17.patch applied>
# out:   lab/dist-c2/vollie-c2-<version>-research-<kit>.apk and BUILD-<kit>.txt (sizes, sha256, tree state)
set -eu
R=$(cd "$(dirname "$0")/.." && pwd); SRC=$(cd "$1" && pwd); APP=$(cd "$2" && pwd)
M=$R/lab/device-bin/c2/app-module; O=$R/lab/dist-c2
V=$(sed -n 's/^version: //p' "$APP/pubspec.yaml"); KIT=$(sed -n "s/.*static const kit = '\(.*\)';/\1/p" "$APP/lib/research.dart")
grep -q "applicationIdSuffix \".c2\"" "$APP/android/app/build.gradle" || { echo "app copy lacks the .c2 suffix: apply patches/app-c2 first"; exit 2; }
grep -q "static const app = '$V';" "$APP/lib/research.dart" || { echo "pubspec version $V != Research.app"; exit 2; }
command -v ninja >/dev/null || PATH=$HOME/Library/Android/sdk/cmake/3.22.1/bin:$PATH
sh "$R/android/build_cpu.sh" "$SRC" "$M"
J=$APP/android/app/src/main/jniLibs/arm64-v8a; B=$APP/third_party/llamadart/third_party/bin/android-arm64
cp "$M/libggml-cpu-android_armv8.2_2.so" "$J/"; cp "$M/libllama.so" "$B/"
# stale hook caches shipped an old libllama.so in campaign 1 (R39): wipe them and flutter clean
rm -rf "$APP/.dart_tool/hooks_runner/shared/llamadart" "$APP/.dart_tool/hooks_runner/llamadart" "$APP/.dart_tool/flutter_build"
(cd "$APP" && flutter clean >/dev/null && LLAMADART_ALLOW_LEGACY_LOCAL_BUNDLES=1 flutter build apk --release)
A=$APP/build/app/outputs/flutter-apk/app-release.apk; X=$(mktemp -d)
for f in "$M/libllama.so" "$M/libggml-cpu-android_armv8.2_2.so"; do
  b=$(basename "$f"); unzip -o -q "$A" "lib/arm64-v8a/$b" -d "$X"
  cmp -s "$X/lib/arm64-v8a/$b" "$f" || { echo "MISMATCH: the APK's $b is not the one just built"; exit 1; }; done
rm -rf "$X"; mkdir -p "$O"; D=$O/vollie-c2-$V-research-$KIT.apk; cp "$A" "$D"
{ echo "apk $(basename "$D") $(wc -c < "$D" | tr -d ' ') B sha256 $(shasum -a 256 "$D" | cut -d' ' -f1)"
  echo "llama.cpp $(git -C "$SRC" rev-parse HEAD)"; git -C "$SRC" diff --stat
  (cd "$M" && shasum -a 256 libggml-cpu-android_armv8.2_2.so libllama.so)
  unzip -l "$D" | awk '$4 ~ /^lib\/arm64-v8a\// {print $1, $4}'; } > "$O/BUILD-$KIT.txt"
cat "$O/BUILD-$KIT.txt"
