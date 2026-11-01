#!/bin/sh
# Builds llama.cpp's CPU module for armv8.2-a+dotprod+fp16 phones (Snapdragon 888 class:
# Cortex-X1/A78, no i8mm/SVE) as the runtime-selectable variant android_armv8.2_2, with
# KleidiAI like the stock Android bundles. Use a checkout with patches/llama.cpp applied.
# usage: android/build_cpu.sh <llama.cpp checkout> [outdir]    env: ANDROID_NDK_HOME, KLEIDI=ON|OFF
set -eu
SRC=$(cd "$1" && pwd); OUT=${2:-out}; mkdir -p "$OUT"; OUT=$(cd "$OUT" && pwd)
NDK=${ANDROID_NDK_HOME:-$HOME/Library/Android/sdk/ndk/28.2.13676358}
TC=$NDK/build/cmake/android.toolchain.cmake
command -v ninja >/dev/null || PATH=$(ls -d "$HOME"/Library/Android/sdk/cmake/*/bin | tail -1):$PATH
W=$(mktemp -d)

cmake -S "$SRC" -B "$W/b" -G Ninja -DCMAKE_TOOLCHAIN_FILE="$TC" -DANDROID_ABI=arm64-v8a \
  -DANDROID_PLATFORM=android-28 -DANDROID_SUPPORT_FLEXIBLE_PAGE_SIZES=ON -DCMAKE_BUILD_TYPE=Release \
  -DGGML_NATIVE=OFF -DGGML_CPU_ALL_VARIANTS=ON -DGGML_CPU_KLEIDIAI=${KLEIDI:-ON} \
  -DBUILD_SHARED_LIBS=ON -DGGML_BACKEND_DL=ON -DGGML_OPENMP=OFF -DGGML_LLAMAFILE=OFF \
  -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_TOOLS=OFF \
  -DLLAMA_BUILD_SERVER=OFF -DLLAMA_CURL=OFF
# The variant build (not a plain armv8.2 build) so the module carries ggml_backend_score:
# ggml then loads it only on chips with dotprod+fp16 instead of dying with SIGILL elsewhere.
cmake --build "$W/b" --target ggml-cpu-android_armv8.2_2
"$(dirname "$TC")/../../toolchains/llvm/prebuilt/"*/bin/llvm-strip \
  -o "$OUT/libggml-cpu-android_armv8.2_2.so" "$W/b/bin/libggml-cpu-android_armv8.2_2.so"
rm -rf "$W"
echo "wrote $OUT/libggml-cpu-android_armv8.2_2.so"
