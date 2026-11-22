#!/bin/sh
# Build tools/c3_head_dump.cpp against a llama.cpp tree's libllama. The libraries are copied first, so that a
# rebuild of the tree during a long dump does not swap them underneath it.
#   tools/c3_head_build.sh <llama.cpp tree> <its build bin dir> <out dir> [tool name, default c3_head_dump]
set -e
T=$1; B=$2; O=$3; N=${4:-c3_head_dump}
mkdir -p "$O/lib"
cp "$B"/*.dylib "$O/lib/"
c++ -std=c++17 -O2 -I"$T/include" -I"$T/ggml/include" "$(dirname "$0")/$N.cpp" \
    -L"$O/lib" -lllama -lggml-base -Wl,-rpath,"$O/lib" -o "$O/$N"
(cd "$T" && git log --oneline -1 && git status --short | head -5) > "$O/TREE.txt"
echo "built $O/$N"
