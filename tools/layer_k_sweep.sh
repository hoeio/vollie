#!/bin/sh
# Per-layer expert-count sensitivity on wikitext-2 VALID (tuning set; test stays untouched).
# For each layer L: experts=6 at L, 4 elsewhere. Output: lab/logs/p/layer_k_sweep.txt "layer ppl"
# usage: tools/layer_k_sweep.sh <model.gguf> [chunks=16]   (needs lab/llama.cpp-opt build with the per-layer patch)
ROOT=$(cd "$(dirname "$0")/.." && pwd); BIN=$ROOT/lab/llama.cpp-opt/build-888/bin; M=$1; N=${2:-16}
OUT=$ROOT/lab/logs/p/layer_k_sweep.txt
run() { LLAMA_EXPERTS_PER_LAYER=$1 "$BIN/llama-perplexity" -m "$M" -f "$ROOT/lab/text/wikitext-2-raw/wiki.valid.raw" \
  -c 512 -b 512 -t "${THREADS:-4}" --chunks "$N" 2>&1 | sed -n 's/.*Final estimate: PPL = \([0-9.]*\).*/\1/p'; }
grep -q '^base4 ' "$OUT" 2>/dev/null || echo "base4 $(run 4)" >> "$OUT"
grep -q '^base6 ' "$OUT" 2>/dev/null || echo "base6 $(run 6)" >> "$OUT"
for L in $(seq 0 39); do
  grep -q "^$L " "$OUT" 2>/dev/null && continue
  k=""; for i in $(seq 0 39); do [ "$i" = "$L" ] && k="$k,6" || k="$k,4"; done
  echo "$L $(run "${k#,}")" >> "$OUT"
done
