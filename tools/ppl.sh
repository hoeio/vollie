#!/bin/sh
# Fixed eval for the whole project: wikitext-2-raw test (primary) + en.txt (sanity), ctx 512, llama-perplexity.
# usage: tools/ppl.sh <tag> <model.gguf> [extra llama-perplexity args...]   (resumable: skips finished logs)
# env: LLAMA_BIN (default upstream v0.8.0 dotprod build), THREADS (default 8)
set -e
ROOT=$(cd "$(dirname "$0")/.." && pwd)
BIN=${LLAMA_BIN:-$ROOT/lab/llama.cpp-v080/build-888/bin}
OUT=$ROOT/lab/logs/p; mkdir -p "$OUT"
tag=$1 model=$2; shift 2
for set in wiki:$ROOT/lab/text/wikitext-2-raw/wiki.test.raw en:$ROOT/lab/text/en.txt; do
  name=${set%%:*} file=${set#*:} log=$OUT/ppl-$tag-$name.log
  grep -q 'Final estimate' "$log" 2>/dev/null && { echo "skip $log"; continue; }
  "$BIN/llama-perplexity" -m "$model" -f "$file" -c 512 -b 512 -t "${THREADS:-8}" -ngl 0 "$@" > "$log" 2>&1
  grep 'Final estimate' "$log"
done
