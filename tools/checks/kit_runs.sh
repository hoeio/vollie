#!/bin/sh
# Portable entry point for outside readers: tools/reproduce.sh (this check assumes the private lab/ layout).
# Vollie: kit replay check, r13b run list (vollie/lib/research.dart), one process, knobs reset between runs like research.dart
# setEnv: every run starts from the phone's saved settings GGML_TINY_OP=262144 GGML_EXPERT_IO=1 (setTinyOps / setExpertIo)
# and the served default 117 + head2 q4_0/2048 + sampler bypass (Research.served). Run 1 = baseline official 32 (after the
# load's one-token warm-up), run 2 = GGML_CPU_BENCH=1, one token. (The r13 ABBA version: lab/logs/r39/kit_runs_r13.sh.)
# Fails (exit 1) unless run 2 prints exactly the two cpu-bench q3k lines (K=1536 rows=512, K=512 rows=1536) and no other run
# prints one; on each: old new int8 q4x4 scalar hot cold cold_touch > 0, every exact_*=1, cpu=<a>-<b> migr=<n>
# cpu_ok=<n>/21 pinned=0|1 mhz=<min>/<max>, no cold_thp / anon_huge_kb; instrument=ok|invalid on the K 512 line only.
# Usage: sh lab/checks/kit_runs.sh [threads]
cd "$(dirname "$0")/../.."
L=$PWD/lab/llama.cpp-tiny/build-888/bin; I=lab/llama.cpp-tiny
c++ -std=c++17 -O2 -o lab/checks/kit_runs lab/checks/kit_runs.cpp -I$I/include -I$I/ggml/include \
    -L$L -lllama -lggml -lggml-base -Wl,-rpath,$L || exit 2
S=GGML_TINY_OP=262144,GGML_EXPERT_IO=1,GGML_EXPERT_SKIP=117,GGML_HEAD2=2048,GGML_HEAD2_Q=q4_0,VOLLIE_SAMPLER_BYPASS=1
KIT_NOREPACK=1 lab/checks/kit_runs lab/models/var/d6-q3k-e6.gguf ${1:-6} "w32|$S" "1|$S,GGML_CPU_BENCH=1" 2>&1 |
  grep -E '^run |^warmup |^cpu-prof|^cpu-bench' |
  awk 'function f(k,  a,b) { if (!split($0,a," " k "=")) return ""; split(a[2],b," "); return b[1] }
       /^warmup |^run /{ r=$2; print; next }
       /^cpu-bench/ { print "  " $0; nb[r]++
                      if (r != 2 || $0 !~ /^cpu-bench q3k threads=1 K=(1536 rows=512|512 rows=1536) /) bad = "bench line run " r
                      if (f("cpu") !~ /^-?[0-9]+--?[0-9]+$/ || f("migr") !~ /^[0-9]+$/ || f("cpu_ok") !~ /^[0-9]+\/21$/ || f("pinned") !~ /^[01]$/ || f("mhz") !~ /^-?[0-9]+\/-?[0-9]+$/) bad = "bench core fields run " r
                      if ($0 ~ / cold_thp=| anon_huge_kb=/) bad = "bench: dropped fields present"
                      if (($0 ~ / K=512 /) != (f("instrument") ~ /^(ok|invalid)$/)) bad = "bench instrument field"
                      n = split("new int8 scalar hot cold cold_touch", ex, " ")
                      for (i = 1; i <= n; i++) if (f("exact_" ex[i]) != 1) bad = "bench exact_" ex[i] " run " r
                      n = split("old new int8 q4x4 scalar hot cold cold_touch", tm, " ")
                      for (i = 1; i <= n; i++) if (!(f(tm[i]) > 0)) bad = "bench " tm[i] " run " r
                      next }
       { print "  " $0; bad = "profile line in run " r }
       END { if (nb[2] != 2) bad = bad ? bad : "bench lines run 2: " nb[2]
             print (bad ? "FAIL: " bad : "PASS"); exit (bad != "") }'
