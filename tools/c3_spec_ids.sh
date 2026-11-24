#!/bin/sh
# Builds tools/c3_spec_ids.cpp against a llama.cpp build of the base tree (c3 S-T1). Host only.
#   sh tools/c3_spec_ids.sh <llama.cpp tree> <build dir> <out binary>
set -eu
T=$(cd "$1" && pwd) B=$(cd "$2" && pwd) O=$3
c++ -std=c++17 -O2 -I"$T/common" -I"$T/include" -I"$T/ggml/include" -I"$T/vendor" tools/c3_spec_ids.cpp \
  -L"$B/bin" -lllama-common -lllama -lggml -lggml-base -Wl,-rpath,"$B/bin" -o "$O"
