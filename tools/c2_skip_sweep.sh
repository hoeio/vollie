#!/bin/sh
# Campaign 2 expert-skip frontier (patches/llama.cpp-c2/0001-olmoe-expert-skip.patch): full wikitext-2 perplexity per
# skip setting, -c 512 -b 512, all chunks, -ngl 0. Resumable: a setting whose log has "Final estimate" is not rerun.
# usage: tools/c2_skip_sweep.sh <llama dir> <model.gguf> <wiki.test.raw> <out dir> [setting ...]
#   llama dir: a source tree with the c2 series applied and built (build/bin/llama-perplexity), or its bin directory
#   setting:   "stock" (no knob) or comma-separated KEY=VALUE for VOLLIE_SKIP_<KEY>, e.g. SUM=65  MASS=900  SUM=80,MINK=2
#   default settings: stock SUM=55 SUM=65 SUM=80 SUM=100 MASS=925 MASS=900 MASS=850 MASS=800
# env: THREADS (default 8), STOCK_PPL (reference for "% over stock" if no stock row is in the table; default 11.7146,
#      the OLMoE served file's stock PPL, R50 host)
set -e
[ $# -ge 4 ] || { sed -n 4,7p "$0" | cut -c3- >&2; exit 2; }
L=$1 MODEL=$2 TEXT=$3 OUT=$4; shift 4
BIN=$L/build/bin/llama-perplexity; [ -x "$BIN" ] || BIN=$L/llama-perplexity
[ -x "$BIN" ] || { echo "no llama-perplexity under $L" >&2; exit 1; }
[ $# -gt 0 ] || set -- stock SUM=55 SUM=65 SUM=80 SUM=100 MASS=925 MASS=900 MASS=850 MASS=800
mkdir -p "$OUT"
SETTINGS="$*"

for s in $SETTINGS; do
  log=$OUT/c2skip-$(echo "$s" | tr ',=' '_-').log
  if grep -q 'Final estimate' "$log" 2>/dev/null; then echo "done   $s"; continue; fi
  envs="VOLLIE_SKIP_STATS=1"
  [ "$s" = stock ] || for kv in $(echo "$s" | tr ',' ' '); do envs="$envs VOLLIE_SKIP_$kv"; done
  echo "run    $s ($envs)"
  # shellcheck disable=SC2086
  env $envs "$BIN" -m "$MODEL" -f "$TEXT" -c 512 -b 512 -t "${THREADS:-8}" -ngl 0 > "$log" 2>&1 \
    || { echo "failed $s: see $log" >&2; exit 1; }
done

# table; "% over stock" uses this table's stock row if present, else STOCK_PPL
stock=$(grep -o 'Final estimate: PPL = [0-9.]*' "$OUT/c2skip-stock.log" 2>/dev/null | awk '{print $NF}')
stock=${stock:-${STOCK_PPL:-11.7146}}
echo "stock reference PPL $stock (host)"
printf '%-16s %18s %9s %13s %13s %13s %14s\n' setting PPL "% stock" "pairs skip %" "bytes skip %" "MB/tok sel" "MB/tok skip"
for s in $SETTINGS; do
  log=$OUT/c2skip-$(echo "$s" | tr ',=' '_-').log
  awk -v s="$s" -v ref="$stock" '
    /Final estimate/ { for (i = 1; i <= NF; i++) { if ($i == "=") p = $(i + 1); if ($i == "+/-") e = $(i + 1) } }
    /^c2-skip:/ { for (i = 2; i <= NF; i++) { split($i, kv, "="); v[kv[1]] = kv[2] } }
    END {
      if (p == "") { printf "%-16s %18s\n", s, "(no result)"; exit }
      if (!("pairs_skipped" in v)) { v["pairs_skipped"] = "0%"; v["expert_bytes_skipped"] = "0%"; v["MB_per_token_selected"] = "-"; v["MB_per_token_skipped"] = "0" }
      printf "%-16s %9.4f +/- %.4f %+8.2f%% %13s %13s %13s %14s\n", s, p, e, 100 * (p / ref - 1),
             v["pairs_skipped"], v["expert_bytes_skipped"], v["MB_per_token_selected"], v["MB_per_token_skipped"]
    }' "$log"
done
echo "MB/tok: expert bytes per evaluated token in this perplexity run (the last layer runs only the scored half of each"
echo "chunk, so 'sel' is below the 346 MB of a decode token); bytes skip % is the per-token fraction, as at -ub 1."
