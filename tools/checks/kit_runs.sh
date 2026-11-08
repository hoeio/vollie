#!/bin/sh
# Portable entry point for outside readers: tools/reproduce.sh (this check assumes the private lab/ layout).
# Vollie: kit replay check, r15 run list (vollie/lib/research.dart, R46 item 2), one process, knobs reset between runs like
# research.dart setEnv: every run starts from the phone's saved settings GGML_TINY_OP=262144 GGML_EXPERT_IO=1 and the served
# default 117 + head2 q4_0/2048 + sampler bypass (Research.served); the run's own env overrides them. (r14: lab/logs/r42/kit_runs_r14.sh.)
# Runs: 1 baseline 32 (after the load's one-token warm-up), 2 bench (GGML_CPU_BENCH=<this kit's value>), 3 warm-up story 200 (default),
# 4 A/A null per token (knob:MMID_FUSE+CHUNK2:0:0, GGML_CPU_AB_TOKEN=1), 5-7 fuse per token, [KERN=1: kern per token x2], then block
# ABBA / BAAB of the fuse, official 32 x2, then (R46 amendment 1 (4): last) block ABBA / BAAB of knob:ROUTER_CAST:0:1 (C1-F16) and 32 with
# GGML_ROUTER_CAST=f16 (C1 ids are reported, not checked: a numerics knob).
# lab/checks/kit_check.py fails (exit 1) unless: the bench prints its two lines in run 2 only; every per-token run prints tok=1..200
# in A B B A order with every field, the knob's value per side, and on every fused token the fused graph of side A's counts; the
# block runs print the warm-up then 8-graph blocks in their order (router: rcast f32 on A, f16 on B); no profile line outside the
# A/B runs; no head2 copy inside a run; every exact run's greedy token ids equal the warm-up story's.
# Then the bench re-arm check: three one-token runs with GGML_CPU_BENCH 7, 7, 8 print bench lines in runs 1 and 3 only.
# Usage: sh lab/checks/kit_runs.sh [threads]   (KERN=1: with the kernel line; JSON=<path>: also write the host run as kit JSON)
cd "$(dirname "$0")/../.."
L=$PWD/lab/llama.cpp-tiny/build-888/bin; I=lab/llama.cpp-tiny
c++ -std=c++17 -O2 -o lab/checks/kit_runs lab/checks/kit_runs.cpp -I$I/include -I$I/ggml/include \
    -L$L -lllama -lggml -lggml-base -Wl,-rpath,$L || exit 2
S=GGML_TINY_OP=262144,GGML_EXPERT_IO=1,GGML_EXPERT_SKIP=117,GGML_HEAD2=2048,GGML_HEAD2_Q=q4_0,VOLLIE_SAMPLER_BYPASS=1; P=GGML_CPU_PROF=1
TOK=$P,GGML_CPU_AB_TOKEN=1; FU=GGML_CPU_AB=knob:MMID_FUSE+CHUNK2:0:1
K=; [ "$KERN" = 1 ] && K="200|$S,$TOK,GGML_CPU_AB=knob:Q3K_KERN:0:1 200|$S,$TOK,GGML_CPU_AB=knob:Q3K_KERN:0:1"
KIT_NOREPACK=1 lab/checks/kit_runs lab/models/var/d6-q3k-e6.gguf ${1:-6} \
    "w32|$S" "1|$S,GGML_CPU_BENCH=$(date +%s)" "200|$S" "200|$S,$TOK,GGML_CPU_AB=knob:MMID_FUSE+CHUNK2:0:0" \
    "200|$S,$TOK,$FU" "200|$S,$TOK,$FU" "200|$S,$TOK,$FU" $K \
    "200|$S,$P,$FU" "200|$S,$P,$FU,GGML_CPU_AB_BAAB=1" \
    "32|$S" "32|$S" \
    "200|$S,$P,GGML_CPU_AB=knob:ROUTER_CAST:0:1" "200|$S,$P,GGML_CPU_AB=knob:ROUTER_CAST:0:1,GGML_CPU_AB_BAAB=1" \
    "32|$S,GGML_ROUTER_CAST=f16" 2>&1 |
  grep -E '^run |^warmup |^cpu-prof|^cpu-bench|^ids ' |
  python3 -I lab/checks/kit_check.py $([ "$KERN" = 1 ] && echo --kern) $([ -n "$JSON" ] && echo --json "$JSON") || exit 1
n=$(KIT_NOREPACK=1 lab/checks/kit_runs lab/models/var/d6-q3k-e6.gguf ${1:-6} "1|GGML_CPU_BENCH=7" "1|GGML_CPU_BENCH=7" "1|GGML_CPU_BENCH=8" 2>&1 |
    grep -E '^run |^cpu-bench q3k threads' | awk '/^run /{r=$2; next} {printf "%s ", r}')
[ "$n" = "1 1 3 3 " ] && echo "bench re-arm: lines in runs 1 1 3 3 (same value twice runs once, a new value runs again): PASS" ||
  { echo "bench re-arm: lines in runs '$n': FAIL"; exit 1; }
