#!/bin/sh
# Host CPU-only speed (RELATIVE numbers, not phone speed). Vollie's config: FA on, q8_0 KV, mmap.
# usage: tools/host_bench.sh <tag> <model.gguf> [extra llama-bench args]   env: LLAMA_BIN, THREADS="1 4", REPACK="0 1"
# appends to lab/logs/p/bench.csv: tag,threads,repack,test,tok_s,stddev
set -e
ROOT=$(cd "$(dirname "$0")/.." && pwd); BIN=${LLAMA_BIN:-$ROOT/lab/llama.cpp-v080/build-888/bin}
tag=$1 model=$2; shift 2; out=$ROOT/lab/logs/p/bench.csv
for t in ${THREADS:-1 4}; do for rp in ${REPACK:-0 1}; do
  "$BIN/llama-bench" -m "$model" -t "$t" --repack "$rp" -fa on -ctk q8_0 -ctv q8_0 -ngl 0 -p 64 -n 64 -r "${REPS:-5}" \
    --delay 2 -o csv "$@" 2>/dev/null | awk -F, -v pre="$tag,$t,$rp" '
    NR==1 { for (i=1;i<=NF;i++) { gsub(/"/,"",$i); c[$i]=i }; next }
    { gsub(/"/,""); printf "%s,%s,%.2f,%.2f\n", pre, ($c["n_gen"]>0?"tg64":"pp64"), $c["avg_ts"], $c["stddev_ts"] }' | tee -a "$out"
done; done
