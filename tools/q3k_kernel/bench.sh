#!/bin/sh
# Host microbench (M4, 1 thread) + bit-identity check of the Q3_K expert gemv prototypes; also writes the
# NDK (device compiler) asm of kernels.c for instruction counts. Exit 1 if any kernel is not bit-identical.
set -e
D=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$D/../../../.." && pwd)
TINY=$ROOT/lab/llama.cpp-tiny
BIN=$TINY/build-888/bin
OUT=${TMPDIR:-/tmp}/q3k_kernel; mkdir -p "$OUT"
L=${L:-20}; E=${E:-7}
[ -f "$OUT/slot-$L-$E.bin" ] || python3 "$D/slot.py" "$TINY" "$ROOT/lab/models/var/d6-q3k-e6.gguf" "$OUT/slot-$L-$E.bin" $L $E
CF="-O3 -march=armv8.2-a+dotprod+fp16"
clang $CF -o "$OUT/bench" "$D/bench.c" "$D/kernels.c" -L"$BIN" -lggml-cpu -lggml-base -Wl,-rpath,"$BIN"
NDK=$(ls -d "$HOME"/Library/Android/sdk/ndk/* | tail -1)
CC=$NDK/toolchains/llvm/prebuilt/darwin-x86_64/bin/clang
if [ -x "$CC" ]; then
    # A78 instruction mix per Q3_K block x 4 rows (1,024 MACs), from the device compiler's code: the hot
    # loop is the last backward b.ne of each function (block loop, body fully unrolled).
    $CC --target=aarch64-linux-android28 $CF -mcpu=cortex-a78 -S -o "$D/kernels.ndk.s" "$D/kernels.c" 2>/dev/null
    $CC --target=aarch64-linux-android28 $CF -mcpu=cortex-a78 -c -o "$OUT/k.o" "$D/kernels.c" 2>/dev/null
    OD=$(dirname "$CC")/llvm-objdump
    hot() { # file -> "start-end" line range of the last backward b.ne loop
        T=$(grep -o "b.ne.*0x[0-9a-f]* <" "$1" | tail -1 | grep -o "0x[0-9a-f]*")
        echo "$(grep -n "^ *${T#0x}:" "$1" | cut -d: -f1)-$(grep -n "b\.ne" "$1" | tail -1 | cut -d: -f1)"; }
    for f in q3k_x4c q3k_t4 q3k_t4h q3k_t4b q3k_t4_rp q3k8_t4 q3k_repack4; do
        $OD -d --no-show-raw-insn --disassemble-symbols=$f "$OUT/k.o" > "$OUT/$f.s"
        printf "%-12s " $f; python3 "$D/icount.py" "$OUT/$f.s" "$(hot "$OUT/$f.s")" | head -1
    done
    SO=$ROOT/vollie/android/app/src/main/jniLibs/arm64-v8a/libggml-cpu-android_armv8.2_2.so
    if [ -f "$SO" ]; then # the shipped kernel (x4 ILP: two inner 4-row loops per block, line ranges found by hand
        # for this build: see README; re-derive them if the .so changes)
        $OD -d --no-show-raw-insn --disassemble-symbols=ggml_vec_dot_q3_K_q8_K_x4 "$SO" > "$OUT/dev_x4.s"
        $OD -d --no-show-raw-insn --disassemble-symbols=ggml_gemv_q4_0_4x4_q8_0 "$SO" > "$OUT/dev_q40x4.s"
        printf "%-12s " device_x4; python3 "$D/icount.py" "$OUT/dev_x4.s" 62-63 81-159 '160-235*4' 236-242 '243-322*4' 323-354 | head -1
        printf "%-12s " "q4_0x4 x8"; python3 "$D/icount.py" "$OUT/dev_q40x4.s" "$(hot "$OUT/dev_q40x4.s")*8" | head -1
    fi
fi
"$OUT/bench" "$OUT/slot-$L-$E.bin"
