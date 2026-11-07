#!/bin/sh
# Portable entry point for outside readers: tools/reproduce.sh (this check assumes the private lab/ layout).
# Vollie: kit replay check, r13 run list (vollie/lib/research.dart), one process, knobs reset between runs like research.dart
# setEnv: every run starts from the phone's saved settings GGML_TINY_OP=262144 GGML_EXPERT_IO=1 (setTinyOps / setExpertIo)
# and the served default 117 + head2 q4_0/2048 + sampler bypass (Research.served), and the run's own env overrides them.
# Story prompt for every run (official 32 = 32 tokens). Run 1 gets the app's one-token warm-up first (the load's); no later
# run changes the module copies, so no other warm-up and no copy is ever made inside a run.
# R25: ABBA blocks (GGML_CPU_AB_WARM 16 / _BLOCK 8). r13 (R39): runs 3-5 = ABBA MMID_TOUCH:0:1 in block orders ABBA, BAAB
# (GGML_CPU_AB_BAAB=1), ABBA.
# Fails (exit 1) unless: run 2 (GGML_CPU_BENCH=1, one token) prints exactly the two cpu-bench q3k lines (K=1536 rows=512,
# K=512 rows=1536) with every timing > 0 (old new int8 q4x4 scalar hot cold cold_touch cold_thp), hot = new, every exact_*=1,
# and mhz= / pinned= / anon_huge_kb= fields, and no other run prints one; every ABBA run prints one warm-up line (ab=A blk=0,
# 16 graphs) then 8-graph blocks in its order; A touch=0, B touch=1, kern=1, skip 117, head2 2048 on both sides; no profile
# line in runs 1, 2, 6, 7; no head2 copy made inside a run (the requant line prints only with GGML_CPU_PROF, so only the
# profiled ABBA runs are checked for it).
# Usage: sh lab/checks/kit_runs.sh [threads]
cd "$(dirname "$0")/../.."
L=$PWD/lab/llama.cpp-tiny/build-888/bin; I=lab/llama.cpp-tiny
c++ -std=c++17 -O2 -o lab/checks/kit_runs lab/checks/kit_runs.cpp -I$I/include -I$I/ggml/include \
    -L$L -lllama -lggml -lggml-base -Wl,-rpath,$L || exit 2
S=GGML_TINY_OP=262144,GGML_EXPERT_IO=1,GGML_EXPERT_SKIP=117,GGML_HEAD2=2048,GGML_HEAD2_Q=q4_0,VOLLIE_SAMPLER_BYPASS=1; P=GGML_CPU_PROF=1
AB=GGML_CPU_AB=knob:MMID_TOUCH:0:1
KIT_NOREPACK=1 lab/checks/kit_runs lab/models/var/d6-q3k-e6.gguf ${1:-6} \
    "w32|$S" "1|$S,GGML_CPU_BENCH=1" \
    "200|$S,$P,$AB" "200|$S,$P,$AB,GGML_CPU_AB_BAAB=1" "200|$S,$P,$AB" "32|$S" "32|$S" 2>&1 |
  grep -E '^run |^warmup |^cpu-prof|^cpu-bench' |
  awk 'function f(k,  a,b) { if (!split($0,a," " k "=")) return ""; split(a[2],b," "); return b[1] }
       /^warmup /{ w=$2; inrun=0; print; next }
       /^run /{ r=$2; inrun=1; print; next }
       /^cpu-bench/ { print "  " $0; nb[r]++
                      if (r != 2 || $0 !~ /^cpu-bench q3k threads=1 K=(1536 rows=512|512 rows=1536) / || f("mhz") !~ /^-?[0-9]+\/-?[0-9]+$/ || f("pinned") !~ /^[01]$/ || f("anon_huge_kb") == "") bad = "bench line run " r
                      n = split("new int8 scalar cold cold_touch cold_thp", ex, " ")
                      for (i = 1; i <= n; i++) if (f("exact_" ex[i]) != 1) bad = "bench exact_" ex[i] " run " r
                      n = split("old new int8 q4x4 scalar hot cold cold_touch cold_thp", tm, " ")
                      for (i = 1; i <= n; i++) if (!(f(tm[i]) > 0)) bad = "bench " tm[i] " run " r
                      if (f("hot") != f("new")) bad = "bench hot != new"
                      next }
       /^cpu-prof head2 requant/ { print "  " $0; if (inrun) bad = "head copy made inside run " r; else h2w[w]++
                                   if ($0 !~ /-> q4_0x4x[48] /) bad = "head copy not repacked, warm-up " w; next }
       /^cpu-prof (expq4|willneed)/ { print "  " $0; bad = "unexpected line run " r; next }
       { blk=f("blk"); side=f("ab"); g=f("graphs"); h2=f("head2"); sk=f("skip"); kn=f("kern"); tc=f("touch")
         if (f("majflt") == "" || f("minflt") == "" || f("head") == "" || kn == "" || tc == "") bad = "missing fields on a line of run " r
         printf "  %s%s graphs=%s skip=%s kavg=%s head2=%s kern=%s touch=%s majflt=%s minflt=%s us/token=%s\n", (side != "" ? "ab=" side " " : ""), (blk != "" ? "blk=" blk : ""),
                g, sk, f("kavg"), h2, kn, tc, f("majflt"), f("minflt"), f("us/token")
         if (r < 3 || r > 5) { bad = "profile line in run " r; next }
         if (blk == 0) { warm[r]++; if (side != "A" || g != 16) bad = "run " r " warm-up line" }
         else { p = (blk-1)%4; want = (p == 1 || p == 2) ? (r == 4 ? "A" : "B") : (r == 4 ? "B" : "A")
                if (side != want || g != 8 || blk != last[r]+1) bad = "run " r " block order at blk=" blk
                n_[r side]++ }
         last[r] = blk
         if (sk !~ /^117\// || h2 != 2048 || kn != 1) bad = "run " r " skip " sk " head2 " h2 " kern " kn
         if (tc != (side == "B" ? 1 : 0)) bad = "run " r " " side " touch " tc }
       END { for (r = 3; r <= 5; r++) if (!warm[r] || n_[r "A"] < 10 || n_[r "B"] < 10) bad = bad ? bad : "missing ABBA lines run " r
             if (nb[2] != 2) bad = bad ? bad : "bench lines run 2: " nb[2]
             print (bad ? "FAIL: " bad : "PASS"); exit (bad != "") }'
