#!/bin/sh
# Vollie host reproduction: the claims that need no phone.
#   (a) identity: Vollie's CPU runtime knobs leave per-chunk wikitext-2 PPL unchanged (bit-exact)
#   (b) gate:     full wikitext-2 PPL, stock vs router-weight expert skip GGML_EXPERT_SKIP=118 (R25)
# usage: tools/reproduce.sh [--quick] [workdir]     (workdir default ./reproduce; idempotent, re-run resumes)
#   --quick   identity + a 60-chunk screen of the gate (not the gate)
# env: REPRO_MODEL=path  use an existing copy of the GGUF (sha256 still checked) instead of downloading 3.16 GB
#      REPRO_TEXT=path   use an existing wiki.test.raw
#      REPRO_REPO=url    llama.cpp git source (default github; any clone containing BASE works)
#      THREADS=n         default: online CPUs, at most 8 (PPL does not depend on it)
# needs: git, cmake, a C/C++ compiler, curl, unzip, shasum or sha256sum
set -e
QUICK=0
[ "$1" = "--quick" ] && { QUICK=1; shift; }
ROOT=$(cd "$(dirname "$0")/.." && pwd)
mkdir -p "${1:-./reproduce}"; W=$(cd "${1:-./reproduce}" && pwd)
PATCHES=$ROOT/patches/llama.cpp
BASE=$(cat "$PATCHES/BASE")
NPROC=$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 4)
T=${THREADS:-$(( NPROC < 8 ? NPROC : 8 ))}
URL=https://huggingface.co/HoeioUser/vollie-qwen3.5-4b/resolve/8ab9bf2d710eb7f1bcb80ef2817b06643ce2e6c1/vollie-granite-4.0-h-tiny-d6-e6.gguf
SHA=5dc1daa0cd3e5099bdaa79beeb9e744bf7c65e0993ea62e4671ddd63b2b5cb25
WIKI_ZIP=https://huggingface.co/datasets/ggml-org/ci/resolve/main/wikitext-2-raw-v1.zip
STOCK_REF=9.6051 SKIP_REF=9.7952 BUDGET=9.797 HARD=10.853

if [ $QUICK = 1 ]; then
  echo "== Vollie reproduce --quick: ~10-20 min on a recent laptop (M4: build ~3 min, identity ~3 x 1-2 min, 2 x 60-chunk screen ~3 min each) + 3.16 GB download"
else
  echo "== Vollie reproduce (full): ~1-1.5 h on a recent laptop (M4, 6 threads: build ~3 min, identity ~5 min, 2 full wikitext-2 runs ~30 min each) + 3.16 GB download"
fi
echo "   workdir $W, threads $T"

# 1. llama.cpp at BASE + Vollie patches, host CPU-only build
SRC=$W/llama.cpp
if [ ! -d "$SRC/.git" ]; then
  git clone "${REPRO_REPO:-https://github.com/ggml-org/llama.cpp}" "$SRC"
fi
if [ "$(git -C "$SRC" rev-parse HEAD)" != "$BASE" ]; then
  git -C "$SRC" fetch -q origin "$BASE" 2>/dev/null || true
  git -C "$SRC" checkout -q --detach "$BASE"
fi
for p in "$PATCHES"/0*.patch; do
  if git -C "$SRC" apply --reverse --check "$p" 2>/dev/null; then echo "   patch already applied: ${p##*/}"
  else git -C "$SRC" apply "$p"; echo "   applied ${p##*/}"; fi
done
# arm64: build the SD888's kernel set (dotprod+fp16, no i8mm), the build the logged host numbers came from
ARCH=""
case $(uname -m) in arm64|aarch64) ARCH="-DGGML_NATIVE=OFF -DGGML_CPU_ARM_ARCH=armv8.2-a+dotprod+fp16";; esac
# shellcheck disable=SC2086
cmake -S "$SRC" -B "$SRC/build" -DCMAKE_BUILD_TYPE=Release -DGGML_METAL=OFF -DGGML_BLAS=OFF \
  -DGGML_CUDA=OFF -DGGML_VULKAN=OFF -DGGML_OPENCL=OFF -DLLAMA_OPENSSL=OFF -DLLAMA_BUILD_TESTS=OFF \
  -DLLAMA_BUILD_SERVER=OFF $ARCH > "$W/cmake.log" 2>&1 || { cat "$W/cmake.log"; exit 1; }
cmake --build "$SRC/build" --config Release -j "$NPROC" --target llama-perplexity llama-cli > "$W/build.log" 2>&1 \
  || { tail -40 "$W/build.log"; exit 1; }
PPL=$SRC/build/bin/llama-perplexity
echo "   built $PPL"

# 2. model + text
MODEL=${REPRO_MODEL:-$W/vollie-granite-4.0-h-tiny-d6-e6.gguf}
[ -n "$REPRO_MODEL" ] || curl -fL -C - -o "$MODEL" "$URL"
if command -v sha256sum >/dev/null; then GOT=$(sha256sum "$MODEL" | cut -d' ' -f1)
else GOT=$(shasum -a 256 "$MODEL" | cut -d' ' -f1); fi
[ "$GOT" = "$SHA" ] || { echo "FAIL model sha256 $GOT != $SHA ($MODEL)"; exit 1; }
echo "   model sha256 OK"
TEXT=${REPRO_TEXT:-$W/wikitext-2-raw/wiki.test.raw}
if [ ! -f "$TEXT" ]; then
  curl -fL -C - -o "$W/wikitext-2-raw-v1.zip" "$WIKI_ZIP"
  (cd "$W" && unzip -oq wikitext-2-raw-v1.zip && rm -f wikitext-2-raw-v1.zip)
fi

# 3. runs. Every Vollie knob unset first, so the caller's environment cannot leak in.
KNOBS="GGML_TINY_OP GGML_MMID GGML_Q3K_X4 GGML_Q3K_VEC GGML_Q3K_ILP GGML_SSM_INPLACE GGML_CONV_RING GGML_EXPERT_IO
GGML_EXPERT_SKIP GGML_EXPERT_SKIP_RENORM GGML_EXPERT_TOPP GGML_EXPERT_GAP GGML_EXPERT_MINK GGML_Q3K_PF GGML_EXPERT_ANON
GGML_EXPERT_LOG GGML_CPU_PROF GGML_CPU_AB GGML_HELPER_CPUS GGML_BARRIER_BENCH"
for k in $KNOBS; do unset "$k"; done
mkdir -p "$W/logs"
# run <name> <env assignments...> -- <llama-perplexity args...>; skips a finished log
run() {
  log=$W/logs/$1.log; shift
  if ! grep -q 'Final estimate' "$log" 2>/dev/null; then
    envs=""; while [ "$1" != "--" ]; do envs="$envs $1"; shift; done; shift
    echo "   running $(basename "$log" .log) ..." >&2
    # shellcheck disable=SC2086
    env $envs "$PPL" -m "$MODEL" -f "$TEXT" -c 512 -t "$T" -ngl 0 "$@" > "$log" 2>&1 || { tail -20 "$log" >&2; exit 1; }
  fi
}
chunks() { grep -o '\[[0-9]*\][0-9.]*' "$W/logs/$1.log" | tr '\n' ' '; }
final()  { sed -n 's/.*Final estimate: PPL = \([0-9.]*\).*/\1/p' "$W/logs/$1.log"; }
FAILED=0

STOCK="GGML_TINY_OP=0 GGML_MMID=0 GGML_Q3K_X4=0 GGML_Q3K_VEC=0 GGML_Q3K_ILP=0 GGML_SSM_INPLACE=0 GGML_CONV_RING=0 GGML_EXPERT_IO=0"
# shellcheck disable=SC2086
run id-stock $STOCK -- -b 512 -ub 1 --chunks 3
run id-default -- -b 512 -ub 1 --chunks 3
run id-ring GGML_CONV_RING=1 -- -b 512 -ub 1 --chunks 3
echo
echo "(a) identity, per-chunk PPL, first 3 chunks, -ub 1 (host)"
for r in stock default ring; do printf '  %-8s %s\n' "$r" "$(chunks id-$r)"; done
if [ -n "$(chunks id-stock)" ] && [ "$(chunks id-stock)" = "$(chunks id-default)" ] && [ "$(chunks id-stock)" = "$(chunks id-ring)" ]
then echo "  PASS  stock = defaults = CONV_RING=1 to the last digit"
else echo "  FAIL  per-chunk values differ"; FAILED=1; fi

if [ $QUICK = 1 ]; then G="--chunks 60" S=-q60; else G="" S=""; fi
# shellcheck disable=SC2086
run gate-stock$S -- -b 512 $G
# shellcheck disable=SC2086
run gate-skip118$S GGML_EXPERT_SKIP=118 -- -b 512 $G
a=$(final gate-stock$S) b=$(final gate-skip118$S)
echo
if [ $QUICK = 1 ]; then echo "(b) gate SCREEN ONLY, NOT THE GATE: first 60 chunks, -b 512 (host)"
else echo "(b) gate: full wikitext-2 test, ctx 512, -b 512 (host; owner's M4: stock $STOCK_REF, skip 118 $SKIP_REF)"; fi
awk -v a="$a" -v b="$b" -v bud="$BUDGET" -v hard="$HARD" -v q=$QUICK 'BEGIN {
  printf "  %-22s %s\n", "EXPERT_SKIP unset", a
  printf "  %-22s %s  (%+.2f %% vs stock)\n", "EXPERT_SKIP=118", b, (b / a - 1) * 100
  if (q) exit 0
  printf "  %s  <= %s (+2 %% speed-rule budget)\n", (b <= bud ? "PASS" : "FAIL"), bud
  printf "  %s  <= %s (+20 %% hard limit)\n", (b <= hard ? "PASS" : "FAIL"), hard
  exit !(b <= bud && b <= hard) }' || FAILED=1
[ $QUICK = 1 ] && echo "  screen only, not the gate (run without --quick for the full set)"
echo "logs: $W/logs"
exit $FAILED
