#!/bin/sh
# host: campaign 3 H3 grid (memo-01 §5) for one model with tools/c3_colskip.py; one RESULT and one PERCHUNK line per cell.
# usage: tools/c3_h3_grid.sh <libdir> <model.gguf> <table.npz> <test text> <n_chunks> <ub> <log> [cells...]
#   a cell is <sparsity>:<groups>:<dense_first_N>; default = the short grid (model-wide 0.2 to 0.5 at N 0; 0.4 and 0.5 at N 1 and 8 and per group)
set -e
lib=$1 model=$2 tab=$3 text=$4 n=$5 ub=$6 log=$7; shift 7
cells=${*:-"0:all:0 0.2:all:0 0.3:all:0 0.4:all:0 0.5:all:0 0.4:all:1 0.5:all:1 0.4:all:8 0.5:all:8 0.4:qkv,o:0 0.5:qkv,o:0 0.4:gu:0 0.5:gu:0 0.4:down:0 0.5:down:0"}
for c in $cells; do
  sp=${c%%:*} r=${c#*:} g=${r%%:*} nd=${r#*:}
  t=$tab; [ "$sp" = 0 ] && t=-
  g2=$g; [ "$g" = all ] && g2=qkv,o,gu,down
  grep -q "RESULT ub $ub sp $sp groups $(echo $g2 | tr , '\n' | sort | paste -sd, -) dense_first $nd:" "$log" 2>/dev/null && continue   # resumable
  python3 -I "$(dirname "$0")/c3_colskip.py" eval "$lib" "$model" "$text" "$n" "$ub" "$t" "$sp" "$g" "$nd" 8 2>/dev/null | grep -E '^(RESULT|PERCHUNK)' >> "$log"
done
