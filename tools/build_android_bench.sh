#!/bin/sh
# Cross-builds llama-bench + llama-perplexity for arm64 Android (Snapdragon 888 class:
# armv8.2-a+dotprod+fp16, no i8mm/SVE), statically linked, no OpenMP (like Vollie's bundle).
# usage: tools/build_android_bench.sh <llama.cpp checkout> [outdir]   env: KLEIDI=ON|OFF (default OFF)
set -eu
SRC=$(cd "$1" && pwd); OUT=${2:-$(cd "$(dirname "$0")" && pwd)/device-bin}
NDK=${ANDROID_NDK_HOME:-$HOME/Library/Android/sdk/ndk/28.2.13676358}
command -v ninja >/dev/null || PATH=$HOME/Library/Android/sdk/cmake/3.22.1/bin:$PATH
W=$(mktemp -d); mkdir -p "$OUT"
cmake -S "$SRC" -B "$W" -G Ninja -DCMAKE_TOOLCHAIN_FILE="$NDK/build/cmake/android.toolchain.cmake" \
  -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-28 -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_SHARED_LIBS=OFF -DGGML_NATIVE=OFF -DGGML_CPU_ARM_ARCH=armv8.2-a+dotprod+fp16 \
  -DGGML_CPU_KLEIDIAI=${KLEIDI:-OFF} -DGGML_OPENMP=OFF -DGGML_LLAMAFILE=OFF -DLLAMA_CURL=OFF -DLLAMA_OPENSSL=OFF \
  -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_SERVER=OFF >/dev/null
cmake --build "$W" --target llama-bench llama-perplexity
for b in llama-bench llama-perplexity; do
  "$NDK"/toolchains/llvm/prebuilt/*/bin/llvm-strip -o "$OUT/$b" "$W/bin/$b"; done
git -C "$SRC" rev-parse HEAD > "$OUT/COMMIT"; git -C "$SRC" diff --stat >> "$OUT/COMMIT"
rm -rf "$W"; ls -la "$OUT"
