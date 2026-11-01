#!/bin/sh
# Requantize Granite from the F16 reference with mradermacher's imatrix (same calibration as the shipped
# i1-Q4_0), experts/token = $EXPERTS (default 4, like the shipped file). usage: tools/quant_variant.sh <tag> <base type> [llama-quantize args]
# e.g. tools/quant_variant.sh exp-q3k Q4_0 --tensor-type ffn_up_exps=q3_k --tensor-type ffn_gate_exps=q3_k
set -e
ROOT=$(cd "$(dirname "$0")/.." && pwd); BIN=${LLAMA_BIN:-$ROOT/lab/llama.cpp-v080/build-888/bin}
tag=$1 type=$2; shift 2; R=$ROOT/lab/models/ref; out=$ROOT/lab/models/var/$tag.gguf; mkdir -p "$(dirname "$out")"
[ -f "$out" ] && { echo "exists $out"; exit 0; }
"$BIN/llama-quantize" --imatrix "$R/Huihui-granite-4.0-h-tiny-abliterated.imatrix.gguf" \
  --override-kv granitehybrid.expert_used_count=int:${EXPERTS:-4} "$@" \
  "$R/Huihui-granite-4.0-h-tiny-abliterated.f16.gguf" "$out" "$type" > "$ROOT/lab/logs/p/quant-$tag.log" 2>&1
ls -la "$out"
