#!/bin/sh
# Builds lab/checks/q3k_x4_check.c against the lab host build's libggml-cpu and runs it (exit 1 = not bit-identical).
cd "$(dirname "$0")/../.."
L=$PWD/lab/llama.cpp-tiny/build-888/bin
cc -O2 -o lab/checks/q3k_x4_check lab/checks/q3k_x4_check.c -L$L -lggml-cpu -lggml-base -Wl,-rpath,$L && lab/checks/q3k_x4_check
