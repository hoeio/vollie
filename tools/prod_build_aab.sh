#!/bin/sh
# Vollie 3.0.0+47 for the Play Store: builds the signed Android App Bundle from an app copy that is 2.3.0+45 with
# patches/app-prod/0001-vollie-3.0.0.patch applied (see patches/app-prod/README.md).
# Native libraries are placed byte-identical, never rebuilt:
#   Vollie 2 (standard names): libllama.so and libggml-cpu-android_armv8.2_2.so from kit r31's APK (= r30 = r28 c2,
#     lab/dist-c2/BUILD-r31.txt);
#   Vollie 1 classic: 2.3.0+45's libllama.so and libggml-cpu-android_armv8.2_2.so from its AAB, renamed
#     libllama_rt_c1.so and libggml_cpu_rt_c1.so (MainActivity.preloadEngine loads them).
# Signing: the app's own release key. android/key.properties and android/app/upload-keystore.jks are symlinked from the
# 2.3.0+45 source folder for the build and removed afterwards; nothing of them is copied or printed.
# usage: tools/prod_build_aab.sh <app copy> <2.3.0+45 source folder> <2.3.0+45 AAB> <kit r31 APK>
#   e.g. tools/prod_build_aab.sh lab/app-prod3 ../vollie/vollie ../vollie/vollie/build/app/outputs/bundle/release/app-release.aab \
#        lab/dist-c2/vollie-c2-1.31.0+31-research-r31.apk
#   SMOKE=<file> (default unset): an emulator smoke APK instead (--dart-define=VOLLIE_SMOKE_MODEL=<file>, same key), into
#   lab/dist-prod/smoke/; never uploaded.
# out: lab/dist-prod/vollie-3.0.0+47.aab and BUILD-3.0.0+47.txt (sha256 of the AAB, the patch, every native library per
#      runtime, 16 KB alignment of every .so, the signing certificate's SHA-256).
set -eu
R=$(cd "$(dirname "$0")/.." && pwd); APP=$(cd "$1" && pwd); SRC=$(cd "$2" && pwd); AAB45=$3; APK31=$4
O=$R/lab/dist-prod; V=$(sed -n 's/^version: //p' "$APP/pubspec.yaml")
[ "$V" = "3.0.0+47" ] && [ -f "$APP/lib/engine.dart" ] || { echo "app copy is not 3.0.0+47: apply patches/app-prod first"; exit 2; }
NDK=$(ls -d "$HOME"/Library/Android/sdk/ndk/* | tail -1); RE=$(ls "$NDK"/toolchains/llvm/prebuilt/*/bin/llvm-readelf)
JAVA_HOME=${JAVA_HOME:-"/Applications/Android Studio.app/Contents/jbr/Contents/Home"}; export JAVA_HOME
sha() { shasum -a 256 "$1" | cut -d' ' -f1; }

# 1. Libraries, checked against their records.
J=$APP/android/app/src/main/jniLibs/arm64-v8a; B=$APP/third_party/llamadart/third_party/bin/android-arm64
X=$(mktemp -d); K=$APP/android/key.properties; KS=$APP/android/app/upload-keystore.jks
cleanup() { rm -rf "$X"; [ ! -L "$K" ] || rm "$K"; [ ! -L "$KS" ] || rm "$KS"; }
trap cleanup EXIT
unzip -q -o "$AAB45" base/lib/arm64-v8a/libllama.so base/lib/arm64-v8a/libggml-cpu-android_armv8.2_2.so -d "$X/c1"
unzip -q -o "$APK31" lib/arm64-v8a/libllama.so lib/arm64-v8a/libggml-cpu-android_armv8.2_2.so -d "$X/c2"
want() { [ "$(sha "$1")" = "$2" ] || { echo "MISMATCH: $1 is not $2"; exit 1; }; }
want "$X/c2/lib/arm64-v8a/libllama.so" 126bfb4f04ec7fff1135cf6e7c1c65e191556aad695c52f718e0e3a645d9ede8
want "$X/c2/lib/arm64-v8a/libggml-cpu-android_armv8.2_2.so" 6f1d859a962f473c7f7b5d1547a2530157a3ca98bbe046524bce3d90b4e3d28e
want "$X/c1/base/lib/arm64-v8a/libllama.so" d2dfa639b8be98b02b8644eab2639c5ebea5c270300f70d71acafa87f5cd2508
want "$X/c1/base/lib/arm64-v8a/libggml-cpu-android_armv8.2_2.so" 195bed88cd12d530de61696b79befcd824e9fcfe3bcd3bfe6021d5b4d71a7f54
cp "$X/c2/lib/arm64-v8a/libllama.so" "$B/libllama.so"
cp "$X/c2/lib/arm64-v8a/libggml-cpu-android_armv8.2_2.so" "$J/libggml-cpu-android_armv8.2_2.so"
cp "$X/c1/base/lib/arm64-v8a/libllama.so" "$J/libllama_rt_c1.so"
cp "$X/c1/base/lib/arm64-v8a/libggml-cpu-android_armv8.2_2.so" "$J/libggml_cpu_rt_c1.so"

# 2. Release key, by reference only.
[ -e "$K" ] || ln -s "$SRC/android/key.properties" "$K"
[ -e "$KS" ] || ln -s "$SRC/android/app/upload-keystore.jks" "$KS"

# 3. Build. Stale hook caches shipped an old libllama.so in campaign 1 (R39): wipe them and flutter clean.
rm -rf "$APP/.dart_tool/hooks_runner/shared/llamadart" "$APP/.dart_tool/hooks_runner/llamadart" "$APP/.dart_tool/flutter_build"
(cd "$APP" && flutter clean >/dev/null && flutter pub get >/dev/null)
if [ -n "${SMOKE:-}" ]; then
  (cd "$APP" && LLAMADART_ALLOW_LEGACY_LOCAL_BUNDLES=1 flutter build apk --release --dart-define=VOLLIE_SMOKE_MODEL="$SMOKE")
  mkdir -p "$O/smoke"; cp "$APP/build/app/outputs/flutter-apk/app-release.apk" "$O/smoke/vollie-$V-smoke.apk"
  echo "smoke apk $O/smoke/vollie-$V-smoke.apk $(sha "$O/smoke/vollie-$V-smoke.apk")"; exit 0
fi
(cd "$APP" && LLAMADART_ALLOW_LEGACY_LOCAL_BUNDLES=1 flutter build appbundle --release)
A=$APP/build/app/outputs/bundle/release/app-release.aab

# 4. Checks on the bundle: the four placed libraries unchanged, every .so 16 KB aligned, signed with 2.3.0+45's key.
unzip -q -o "$A" 'base/lib/*' 'BUNDLE-METADATA/*' -d "$X/out" 2>/dev/null || unzip -q -o "$A" 'base/lib/*' -d "$X/out"
L=$X/out/base/lib/arm64-v8a
cmp -s "$L/libllama.so" "$B/libllama.so" && cmp -s "$L/libggml-cpu-android_armv8.2_2.so" "$J/libggml-cpu-android_armv8.2_2.so" \
  && cmp -s "$L/libllama_rt_c1.so" "$J/libllama_rt_c1.so" && cmp -s "$L/libggml_cpu_rt_c1.so" "$J/libggml_cpu_rt_c1.so" \
  || { echo "MISMATCH: a runtime library in the bundle is not the one placed"; exit 1; }
align=
for f in "$L"/*.so; do
  a=$("$RE" -lW "$f" | awk '$1 == "LOAD" {print $NF}' | sort -u | tr '\n' ' ')
  for x in $a; do [ $((x)) -ge 16384 ] || { echo "NOT 16 KB ALIGNED: $(basename "$f") LOAD align $x"; exit 1; }; done
  align="$align$(basename "$f"): LOAD align $a
"
done
cert() { "$JAVA_HOME/bin/keytool" -printcert -jarfile "$1" | sed -n 's/^[[:space:]]*SHA256: //p' | head -1; }
C46=$(cert "$A"); C45=$(cert "$AAB45")
[ -n "$C46" ] && [ "$C46" = "$C45" ] || { echo "SIGNING: the bundle is not signed with 2.3.0+45's certificate"; exit 1; }

# 5. Output and build record.
mkdir -p "$O"; D=$O/vollie-$V.aab; cp "$A" "$D"
P=$R/patches/app-prod/0001-vollie-3.0.0.patch
{ echo "aab $(basename "$D") $(wc -c < "$D" | tr -d ' ') B sha256 $(sha "$D")"
  echo "app com.insaneutils.vollie $V, base 2.3.0+45 (AAB sha256 $(sha "$AAB45"))"
  echo "patch $(basename "$P") sha256 $([ -f "$P" ] && sha "$P" || echo missing)"
  echo "signing certificate SHA-256 $C46 (equal to 2.3.0+45's)"
  echo "runtime v2 (campaign 2, kit r31 c2 = r30 = r28 c2; source $(basename "$APK31") sha256 $(sha "$APK31")):"
  echo "  libllama.so $(sha "$L/libllama.so")"
  echo "  libggml-cpu-android_armv8.2_2.so $(sha "$L/libggml-cpu-android_armv8.2_2.so")"
  echo "runtime classic (2.3.0+45's own, from its AAB):"
  echo "  libllama_rt_c1.so $(sha "$L/libllama_rt_c1.so")"
  echo "  libggml_cpu_rt_c1.so $(sha "$L/libggml_cpu_rt_c1.so")"
  echo "shared by both (llamadart 0.15.6 bundle, Flutter, app):"
  for f in "$L"/*.so; do b=$(basename "$f"); case $b in libllama.so|libggml-cpu-android_armv8.2_2.so|libllama_rt_c1.so|libggml_cpu_rt_c1.so) ;; *) echo "  $b $(wc -c < "$f" | tr -d ' ') B $(sha "$f")";; esac; done
  echo "16 KB page alignment (llvm-readelf, every LOAD segment):"; printf '%s' "$align" | sed 's/^/  /'
} > "$O/BUILD-$V.txt"
cat "$O/BUILD-$V.txt"
