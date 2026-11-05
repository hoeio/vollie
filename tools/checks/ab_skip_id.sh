#!/bin/sh
# Portable entry point for outside readers: tools/reproduce.sh (this check assumes the private lab/ layout).
# Vollie: identity of the A/B path with a no-op skip: GGML_CPU_AB=knob:EXPERT_SKIP:0:0 (+ GGML_CPU_PROF=1)
# must equal off per chunk, chunks 1-3, -t 6, -ub 1 / -ub 32 / -b 512. Exit 1 if any differs.
# Usage: sh lab/checks/ab_skip_id.sh
cd "$(dirname "$0")/../.."
B=lab/llama.cpp-tiny/build-888/bin/llama-perplexity; M=lab/models/var/d6-q3k-e6.gguf; T=lab/text/wikitext-2-raw/wiki.test.raw
O=${TMPDIR:-/tmp}/ab_skip_id; mkdir -p $O; rc=0
for ub in "-ub 1" "-ub 32" "-b 512"; do
  for cfg in off ab; do
    case $cfg in off) E="";; ab) E="GGML_CPU_PROF=1 GGML_CPU_AB=knob:EXPERT_SKIP:0:0";; esac
    env $E $B -m $M -f $T -c 512 --chunks 3 -t 6 $ub > $O/$cfg.log 2>&1
    eval "$cfg=\"\$(grep -o '\[1\].*' $O/$cfg.log)\""
  done
  [ "$off" = "$ab" ] && s=same || { s=DIFFERENT; rc=1; }
  echo "$ub off: $off | ab: $ab | $s"
done
exit $rc
