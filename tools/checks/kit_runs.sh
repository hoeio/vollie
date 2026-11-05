#!/bin/sh
# Portable entry point for outside readers: tools/reproduce.sh (this check assumes the private lab/ layout).
# Vollie: kit replay check, r6 run list (vollie/lib/research.dart), one process, knobs reset to '' between runs like
# research.dart setEnv; every run starts from the phone's saved settings GGML_TINY_OP=262144 GGML_EXPERT_IO=1 (what
# setTinyOps / setExpertIo restore) and the run's own env overrides them. Story prompt for every run (official 32 = 32 tokens).
# R3: A/B window carry-over. R25: ABBA blocks (GGML_CPU_AB_WARM 16 / _BLOCK 8). R26: adv= (expert slices advised per token).
# Fails (exit 1) unless: every ABBA run prints one warm-up line (ab=A blk=0, 16 graphs) then 8-graph blocks A B B A ...;
# EXPERT_SKIP:0:118 A lines skip=0/0% kavg=6.00, B lines skip=118 kavg 3.8-5.0, pooled B mean 4.0-4.5, B adv <= 0.85 x A adv
# (skip-aware prefetch); CONV_RING:1:0 and SSM_INPLACE:1:0 A nodes < B nodes; vanilla lines tiny=0 eio=0 q3k=0 x4=0 ilp=0
# mmid=0 adv=0; eio 0 lines eio=0 adv=0; other always-on lines eio=1 adv>0, no ab=, 32 graphs; skip 118 kavg < 5, control 6.00.
# Usage: sh lab/checks/kit_runs.sh [threads]
cd "$(dirname "$0")/../.."
L=$PWD/lab/llama.cpp-tiny/build-888/bin; I=lab/llama.cpp-tiny
c++ -std=c++17 -O2 -o lab/checks/kit_runs lab/checks/kit_runs.cpp -I$I/include -I$I/ggml/include \
    -L$L -lllama -lggml -lggml-base -Wl,-rpath,$L || exit 2
S=GGML_TINY_OP=262144,GGML_EXPERT_IO=1; P=GGML_CPU_PROF=1
V=GGML_TINY_OP=0,GGML_MMID=0,GGML_Q3K_X4=0,GGML_Q3K_VEC=0,GGML_Q3K_ILP=0,GGML_SSM_INPLACE=0,GGML_CONV_RING=0,GGML_EXPERT_SKIP=0,GGML_EXPERT_IO=0
lab/checks/kit_runs lab/models/var/d6-q3k-e6.gguf ${1:-6} \
    "32|$S" "32|$S" "32|$S,$V" "200|$S,$P,$V" \
    "200|$S,$P,GGML_CPU_AB=knob:CONV_RING:1:0" "200|$S,$P,GGML_CPU_AB=knob:EXPERT_SKIP:0:118" \
    "200|$S,$P,GGML_CPU_AB=knob:EXPERT_SKIP:0:118" "200|$S,$P,GGML_CPU_AB=knob:SSM_INPLACE:1:0" \
    "200|$S,$P,GGML_EXPERT_SKIP=118" "200|$S,$P" "200|$S,$P,GGML_EXPERT_IO=0" \
    "200|$S,$P,GGML_EXPERT_SKIP=118,GGML_EXPERT_IO=0" "32|$S,GGML_EXPERT_SKIP=118" 2>&1 | grep -E '^run |^cpu-prof' |
  awk 'function f(k,  a,b) { if (!split($0,a," " k "=")) return ""; split(a[2],b," "); return b[1] }
       /^run /{ r=$2; print; next }
       { blk=f("blk"); side=f("ab"); g=f("graphs"); k=f("kavg"); adv=f("adv"); nd=f("nodes")
         printf "  %s%s graphs=%s eio=%s skip=%s kavg=%s adv=%s nodes=%s us/token=%s\n", (side != "" ? "ab=" side " " : ""), (blk != "" ? "blk=" blk : ""),
                g, f("eio"), f("skip"), k, adv, nd, f("us/token")
         lines[r]++
         if (r >= 5 && r <= 8) { # ABBA runs
           if (blk == 0) { warm[r]++; if (side != "A" || g != 16) bad = "run " r " warm-up line" }
           else { want = ((blk-1)%4 == 1 || (blk-1)%4 == 2) ? "B" : "A"
                  if (side != want || g != 8 || blk != last[r]+1) bad = "run " r " block order at blk=" blk
                  if (side == "A") { na[r]++; sa[r] += adv; nda[r] += nd } else { nbb[r]++; sbb[r] += adv; ndb[r] += nd } }
           last[r] = blk
           if (r == 6 || r == 7) {
             if (blk > 0 && side == "A" && (k != "6.00" || f("skip") != "0/0%")) bad = "run " r " A kavg " k
             if (blk > 0 && side == "B") { if (k < 3.8 || k > 5.0 || f("skip") !~ /^118\//) bad = "run " r " B kavg " k; nk++; sk += k } } }
         else if (r == 4) { if (!/ tiny=0 eio=0 q3k=0 x4=0 ilp=0 / || !/ mmid=0 / || adv != 0) bad = "vanilla line" }
         else if (r == 11 || r == 12) { if (f("eio") != 0 || adv != 0 || / ab=/) bad = "eio 0 line run " r }
         else if (r >= 9) { if (f("eio") != 1 || adv <= 0 || / ab=/ || g != 32) bad = "always-on line run " r
                            if (r == 9 && k >= 5) bad = "skip 118 kavg " k; if (r == 10 && k != "6.00") bad = "control kavg " k } }
       END { for (r = 5; r <= 8; r++) if (!warm[r] || nbb[r] < 10) bad = bad ? bad : "missing ABBA lines run " r
             for (r = 9; r <= 12; r++) if (!lines[r]) bad = bad ? bad : "missing lines run " r
             if (lines[1] || lines[13]) bad = bad ? bad : "profile line without GGML_CPU_PROF"
             mk = nk ? sk/nk : 0; if (mk < 4.0 || mk > 4.5) bad = bad ? bad : "B mean kavg"
             for (r = 5; r <= 8; r++) { am[r] = na[r] ? sa[r]/na[r] : 0; bm[r] = nbb[r] ? sbb[r]/nbb[r] : 0
                                        an[r] = na[r] ? nda[r]/na[r] : 0; bn[r] = nbb[r] ? ndb[r]/nbb[r] : 0
                                        printf "run %d: A adv %.1f nodes %.0f | B adv %.1f nodes %.0f\n", r, am[r], an[r], bm[r], bn[r] }
             if (!(an[5] < bn[5]) || !(an[8] < bn[8])) bad = bad ? bad : "A/B nodes (CONV_RING / SSM_INPLACE)"
             for (r = 6; r <= 7; r++) if (!(am[r] > 0 && bm[r] <= 0.85*am[r])) bad = bad ? bad : "skip-aware adv run " r
             printf "skip 118 B mean kavg %.2f; adv B/A %.3f %.3f; %s\n", mk, am[6] ? bm[6]/am[6] : 0, am[7] ? bm[7]/am[7] : 0, bad ? "FAIL: " bad : "PASS"
             exit (bad != "") }'
