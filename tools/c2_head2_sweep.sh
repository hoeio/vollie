#!/bin/sh
# Campaign 2 untied two-stage head (patches/llama.cpp-c2/0009-head2-untied.patch, a declared numerics change), alone and
# with expert skip: full wikitext-2 perplexity per setting, -c 512 -b 512, all chunks, -ngl 0, VOLLIE_HEAD2_BATCH=1 (the
# test mode that runs the head2 path on every column of a batch). Resumable: a setting whose log has "Final estimate"
# is not rerun.
# usage: tools/c2_head2_sweep.sh <llama dir> <model.gguf> <wiki.test.raw> <out dir> [setting ...]
#   llama dir: a source tree with the c2 series 0001..0009 applied and built (build/bin/llama-perplexity), or its bin dir
#   setting:   "stock" (no knob) or comma-separated KEY=VALUE: K=<rows> sets VOLLIE_HEAD2, any other KEY sets
#              VOLLIE_SKIP_<KEY>, e.g. K=1024  MASS=900  K=2048,MASS=900
#   default:   stock K=1024 K=2048 K=4096 MASS=900 K=1024,MASS=900 K=2048,MASS=900 K=4096,MASS=900
# env: PPL_ARGS (extra llama-perplexity arguments; none = repack on, the kit's OLMoE path; -nr = no repack),
#      THREADS (default 8). "% stock" is against this table's stock row (run with the same PPL_ARGS).
set -e
[ $# -ge 4 ] || { sed -n 5,9p "$0" | cut -c3- >&2; exit 2; }
L=$1 MODEL=$2 TEXT=$3 OUT=$4; shift 4
BIN=$L/build/bin/llama-perplexity; [ -x "$BIN" ] || BIN=$L/llama-perplexity
[ -x "$BIN" ] || { echo "no llama-perplexity under $L" >&2; exit 1; }
[ $# -gt 0 ] || set -- stock K=1024 K=2048 K=4096 MASS=900 K=1024,MASS=900 K=2048,MASS=900 K=4096,MASS=900
mkdir -p "$OUT"
SETTINGS="$*"

for s in $SETTINGS; do
  log=$OUT/c2head2-$(echo "$s" | tr ',=' '_-').log
  if grep -q 'Final estimate' "$log" 2>/dev/null; then echo "done   $s"; continue; fi
  envs="VOLLIE_SKIP_STATS=1"
  [ "$s" = stock ] || for kv in $(echo "$s" | tr ',' ' '); do
    case $kv in K=*) envs="$envs VOLLIE_HEAD2=${kv#K=} VOLLIE_HEAD2_BATCH=1" ;; *) envs="$envs VOLLIE_SKIP_$kv" ;; esac
  done
  echo "run    $s ($envs)"
  # shellcheck disable=SC2086
  env $envs "$BIN" -m "$MODEL" -f "$TEXT" -c 512 -b 512 -t "${THREADS:-8}" -ngl 0 ${PPL_ARGS:-} > "$log" 2>&1 \
    || { echo "failed $s: see $log" >&2; exit 1; }
done

stock=$(grep -o 'Final estimate: PPL = [0-9.]*' "$OUT/c2head2-stock.log" 2>/dev/null | awk '{print $NF}')
[ -n "$stock" ] || { echo "no stock row in $OUT: add the setting 'stock'" >&2; exit 1; }
echo "stock PPL $stock (host, PPL_ARGS='${PPL_ARGS:-}')"
printf '%-18s %18s %9s %13s %18s %8s\n' setting PPL "% stock" "pairs skip %" "exact rows/column" copies
for s in $SETTINGS; do
  log=$OUT/c2head2-$(echo "$s" | tr ',=' '_-').log
  awk -v s="$s" -v ref="$stock" '
    /Final estimate/ { for (i = 1; i <= NF; i++) { if ($i == "=") p = $(i + 1); if ($i == "+/-") e = $(i + 1) } }
    /^c2-skip:/ || /^c2-head:/ { for (i = 2; i <= NF; i++) { split($i, kv, "="); v[kv[1]] = kv[2] } }
    END {
      if (p == "") { printf "%-18s %18s\n", s, "(no result)"; exit }
      if (!("pairs_skipped" in v)) v["pairs_skipped"] = "0%"
      if (!("exact_rows_per_col" in v)) { v["exact_rows_per_col"] = "all"; v["copies"] = "-" }
      printf "%-18s %9.4f +/- %.4f %+8.2f%% %13s %18s %8s\n", s, p, e, 100 * (p / ref - 1),
             v["pairs_skipped"], v["exact_rows_per_col"], v["copies"]
    }' "$log"
done
echo "exact rows/column: rows recomputed exactly per token (repack on: all 8 rows of every touched 8-row group);"
echo "the rest keep the Q4_0 screen logit. Batch mode: the exact rows use the decode gemv, not the stock -b 512 gemm."
