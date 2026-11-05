#!/bin/sh
# Portable entry point for outside readers: tools/reproduce.sh (this script defaults to the private lab/ layout).
# Fixed eval for the whole project: wikitext-2-raw test (primary) + en.txt (sanity), ctx 512, llama-perplexity.
# usage: tools/ppl.sh [--llama-dir <dir with llama-perplexity>] [--text <wiki.test.raw>] <tag> <model.gguf> [extra llama-perplexity args...]
#   (resumable: skips finished logs; the en.txt sanity run is skipped if lab/text/en.txt is absent)
# env: LLAMA_BIN (default upstream v0.8.0 dotprod build; --llama-dir wins), THREADS (default 8)
set -e
ROOT=$(cd "$(dirname "$0")/.." && pwd)
BIN=${LLAMA_BIN:-$ROOT/lab/llama.cpp-v080/build-888/bin}
TEXT=$ROOT/lab/text/wikitext-2-raw/wiki.test.raw
while :; do case $1 in
  --llama-dir) BIN=$2; shift 2;;
  --text) TEXT=$2; shift 2;;
  *) break;;
esac; done
[ $# -ge 2 ] || { sed -n 4p "$0" | cut -c3- >&2; exit 2; }
OUT=$ROOT/lab/logs/p; mkdir -p "$OUT"
tag=$1 model=$2; shift 2
for set in wiki:$TEXT en:$ROOT/lab/text/en.txt; do
  name=${set%%:*} file=${set#*:} log=$OUT/ppl-$tag-$name.log
  [ "$name" = en ] && [ ! -f "$file" ] && { echo "skip en (no $file)"; continue; }
  grep -q 'Final estimate' "$log" 2>/dev/null && { echo "skip $log"; continue; }
  "$BIN/llama-perplexity" -m "$model" -f "$file" -c 512 -b 512 -t "${THREADS:-8}" -ngl 0 "$@" > "$log" 2>&1
  grep 'Final estimate' "$log"
done
