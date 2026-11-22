#!/bin/sh
# c3 S3 freeze of one subject (owner's laptop; reports/memo-10-work/clyde-freeze.md). Publisher weights at the pinned
# commit -> F16 (sha256 must equal the screen's, memo-06-work/tom-screens.md §5.1) -> Q4_0 re-made from it (must equal
# the kept file byte for byte) -> full wikitext-2 test PPL, ctx 512: F16 at -b 512, served file at -b 512 and -ub 1.
# usage: tools/c3_freeze.sh yi15-6b|llama2-7b|deepseek7b    env: F16_UB1=1 also runs F16 at -ub 1 (long, optional)
# Resumable: finished PPL logs are skipped (tools/ppl.sh); the F16 is deleted after its runs.
set -eu
ROOT=$(cd "$(dirname "$0")/.." && pwd); cd "$ROOT/lab"
m=${1:?usage: c3_freeze.sh yi15-6b|llama2-7b|deepseek7b}
B=$ROOT/lab/llama.cpp-c3/build/bin; G=c3/models; D=c3/dl/$m; OUT=$ROOT/lab/logs/p; log=c3/logs/freeze-$m.log
case $m in
yi15-6b)    repo=01-ai/Yi-1.5-6B-Chat; rev=771924d1c83d67527d665913415d7086f11ea9c0; wg='*.safetensors'
            f16=7ecff8aaf72278b573d53eb62b7102e28a9f72d3d61960a4d532a3298823b114
            q4=702cc2ebaa3bd4933598e25b556961b95ac260ed15df74fa09df2b2b7b48a299; S=$G/yi15-6b-q4_0.gguf; s=$q4;;
llama2-7b)  repo=meta-llama/Llama-2-7b-chat-hf; rev=f5db02db724555f92da89c216ac04704f23d4590; wg='*.safetensors'
            f16=175c701f53317f0f209479a3ec3706de06714ea1115d01143ace93fe26bfefdb
            q4=8eb4952f2592fdfc1288695f6b85aaf48d5628ec5d2c75f9387d3144652ad4bf; S=$G/llama2-7b-q4_0.gguf; s=$q4;;
deepseek7b) repo=deepseek-ai/deepseek-llm-7b-chat; rev=afbda8b347ec881666061fa67447046fc5164ec8; wg='*.bin'
            f16=29b971949796eec79583562e740d861b66680d8f8782aeeaa88c5e04b45bd5b2
            q4=f403ddc298020af0fe8dfbbfec9a74cd3f8951938d6c21540054d85f09315118; S=$G/deepseek7b-q4_0.gguf; s=$q4;;
            # owner 2026-11-22: the BOS metadata change is not allowed; the served file is the unmodified Q4_0
*) echo "unknown subject $m" >&2; exit 2;;
esac
F=$G/$m-f16.gguf
sha() { shasum -a 256 "$1" | cut -d' ' -f1; }
say() { echo "$(date '+%F %T') $*" | tee -a $log; }
fin() { grep -q 'Final estimate' "$OUT/ppl-$1-wiki.log" 2>/dev/null; }
ppl() { sh "$ROOT/tools/ppl.sh" --llama-dir "$B" --text "$ROOT/lab/c3/text/wiki.test.raw" "$@"; }

say "freeze $m: $repo @ $rev; llama.cpp-c3 $(git -C llama.cpp-c3 rev-parse --short HEAD); perplexity $(sha $B/llama-perplexity | cut -c1-12)"
[ "$(sha $S)" = $s ] || { say "served file $S sha256 is not $s: stop"; exit 1; }
say "served $S sha256 $s ok ($(wc -c < $S | tr -d ' ') B)"

need_f16=1; fin c3f-$m-f16-b512 && { [ "${F16_UB1:-0}" != 1 ] || fin c3f-$m-f16-ub1; } && need_f16=0
if [ $need_f16 = 1 ] && [ ! -f $F ]; then
  free=$(df -g . | awk 'NR==2 {print $4}'); [ "$free" -ge 30 ] || { say "only $free GB free, 30 needed: stop"; exit 1; }
  rm -rf $D; mkdir -p $D   # the directory name sets general.name in the GGUF; it must be c3/dl/<m> to reproduce the sha
  venv-c3/bin/python -I - "$repo" "$rev" "$D" "$wg" >> $log 2>&1 <<'PY'
import sys, hashlib, os
from huggingface_hub import snapshot_download, HfApi
repo, rev, d, wg = sys.argv[1:]
snapshot_download(repo, revision=rev, local_dir=d, allow_patterns=[wg, '*.json', '*.model', 'tokenizer*', '*.tiktoken'])
for s in HfApi().model_info(repo, revision=rev, files_metadata=True).siblings:
    p = os.path.join(d, s.rfilename)
    if s.lfs and os.path.exists(p):
        h = hashlib.sha256()
        with open(p, 'rb') as f:
            for b in iter(lambda: f.read(1 << 24), b''): h.update(b)
        ok = h.hexdigest() == s.lfs.sha256; print('lfs', s.rfilename, h.hexdigest(), 'OK' if ok else 'MISMATCH'); assert ok
PY
  grep '^lfs ' $log | tail -4
  venv-c3/bin/python -E -s llama.cpp-c3/convert_hf_to_gguf.py $D --outtype f16 --outfile $G/$m-f16.gguf >> $log 2>&1
  rm -rf $D
  h=$(sha $G/$m-f16.gguf)   # equality with the screen's F16 is expected; the re-made Q4_0 below is the binding check
  [ "$h" = $f16 ] && say "F16 sha256 $h equals the screen's" || say "WARNING F16 sha256 $h is not the screen's $f16"
  R=$G/$m-requant-q4_0.gguf; "$B/llama-quantize" $G/$m-f16.gguf $R Q4_0 8 >> $log 2>&1
  cmp -s $R $G/$m-q4_0.gguf && [ "$(sha $R)" = $q4 ] || { say "re-made Q4_0 differs from $G/$m-q4_0.gguf: stop"; exit 1; }
  rm -f $R; say "re-made Q4_0 equals $G/$m-q4_0.gguf (sha256 $q4)"
fi

if [ $need_f16 = 1 ]; then
  ppl c3f-$m-f16-b512 $F
  [ "${F16_UB1:-0}" = 1 ] && ppl c3f-$m-f16-ub1 $F -ub 1
  rm -f $F; say "F16 deleted after its runs"
fi
ppl c3f-$m-q4_0-b512 $S
ppl c3f-$m-q4_0-ub1 $S -ub 1

v() { grep 'Final estimate' "$OUT/ppl-$1-wiki.log" 2>/dev/null | sed 's/.*PPL = \([0-9.]*\).*/\1/'; }
say "RESULT $m (host): F16 b512 $(v c3f-$m-f16-b512) | served b512 $(v c3f-$m-q4_0-b512) | served ub1 $(v c3f-$m-q4_0-ub1) | F16 ub1 $(v c3f-$m-f16-ub1)"
awk -v f=$(v c3f-$m-f16-b512) -v q=$(v c3f-$m-q4_0-b512) -v u=$(v c3f-$m-q4_0-ub1) 'BEGIN {
  printf "budget b512 %.4f | budget ub1 %.4f | ceiling %.4f | served/F16 %.4f | served <= ceiling: %s\n",
         q*1.05, u*1.05, f*1.20, q/f, (q <= f*1.20 ? "yes" : "NO") }' | tee -a $log
