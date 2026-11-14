#!/bin/sh
# Campaign 2: cross-builds llama-bench, llama-perplexity and llama-completion for arm64 Android
# (Snapdragon 888: armv8.2-a+dotprod+fp16, no i8mm/SVE), llama libraries static, no OpenMP, KleidiAI off.
# Same flags as campaign 1's tools/build_android_bench.sh, plus llama-completion and a sha256 list.
# usage: tools/c2_build_android.sh <llama.cpp checkout> <outdir> [patch file | series dir]   env: JOBS (default 4)
# (the patch argument is only recorded in COMMIT, with each file's sha256; apply the patches to the checkout first)
set -eu
SRC=$(cd "$1" && pwd); OUT=${2:?outdir}; PATCH=${3:-}
NDK=${ANDROID_NDK_HOME:-$HOME/Library/Android/sdk/ndk/28.2.13676358}
command -v ninja >/dev/null || PATH=$HOME/Library/Android/sdk/cmake/3.22.1/bin:$PATH
W=$(mktemp -d); mkdir -p "$OUT"
cmake -S "$SRC" -B "$W" -G Ninja -DCMAKE_TOOLCHAIN_FILE="$NDK/build/cmake/android.toolchain.cmake" \
  -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-28 -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_SHARED_LIBS=OFF -DGGML_NATIVE=OFF -DGGML_CPU_ARM_ARCH=armv8.2-a+dotprod+fp16 \
  -DGGML_CPU_KLEIDIAI=OFF -DGGML_OPENMP=OFF -DGGML_LLAMAFILE=OFF -DLLAMA_CURL=OFF -DLLAMA_OPENSSL=OFF \
  -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_SERVER=OFF >/dev/null
cmake --build "$W" -j "${JOBS:-4}" --target llama-bench llama-perplexity llama-completion
B="llama-bench llama-perplexity llama-completion"
for b in $B; do "$NDK"/toolchains/llvm/prebuilt/*/bin/llvm-strip -o "$OUT/$b" "$W/bin/$b"; done
{ echo "commit $(git -C "$SRC" rev-parse HEAD)"
  echo "ndk $(basename "$NDK")"
  echo "flags armv8.2-a+dotprod+fp16 static no-openmp kleidiai-off llamafile-off"
  if [ -d "$PATCH" ]; then for f in "$PATCH"/0*.patch; do echo "patch $(basename "$f") sha256 $(shasum -a 256 "$f" | cut -d' ' -f1)"; done
  elif [ -n "$PATCH" ]; then echo "patch $(basename "$PATCH") sha256 $(shasum -a 256 "$PATCH" | cut -d' ' -f1)"
  else echo "patch none"; fi
  echo "tree diff:"; git -C "$SRC" diff --stat; } > "$OUT/COMMIT"
(cd "$OUT" && shasum -a 256 $B > SHA256SUMS)
rm -rf "$W"; cat "$OUT/COMMIT" "$OUT/SHA256SUMS"
