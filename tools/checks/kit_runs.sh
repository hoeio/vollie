#!/bin/sh
# Vollie: R3 kit-bug check (A/B window carry-over). One process replays the device's r3 run order: five
# A/B runs, knob:EXPERT_SKIP:0:115, a plain 200-token run, then plain GGML_EXPERT_SKIP=115 as reference.
# Fails (exit 1) unless the plain run shows skip=0 kavg=6.00 in every window and the A/B B side's kavg
# is within 0.15 of the plain-skip run's first window (same tokens). Usage: sh lab/checks/kit_runs.sh [threads]
cd "$(dirname "$0")/../.."
L=$PWD/lab/llama.cpp-tiny/build-888/bin; I=lab/llama.cpp-tiny
c++ -std=c++17 -O2 -o lab/checks/kit_runs lab/checks/kit_runs.cpp -I$I/include -I$I/ggml/include \
    -L$L -lllama -lggml -lggml-base -Wl,-rpath,$L || exit 2
P=GGML_CPU_PROF=1
lab/checks/kit_runs lab/models/var/d6-q3k-e6.gguf ${1:-6} \
    "72|$P,GGML_CPU_AB=knob:Q3K_VEC:0:1" "72|$P,GGML_CPU_AB=knob:Q3K_ILP:0:1" "72|$P,GGML_CPU_AB=knob:Q3K_PF:0:1" \
    "72|$P,GGML_CPU_AB=knob:Q3K_ILP:0:1" "72|$P,GGML_CPU_AB=knob:Q3K_PF:0:1" "72|$P,GGML_CPU_AB=knob:EXPERT_SKIP:0:115" \
    "200|$P" "72|$P,GGML_EXPERT_SKIP=115" 2>&1 | grep -E '^run |^cpu-prof' |
  sed -E 's/^(cpu-prof[^ ]*( ab=.)? graphs=[0-9]+).* skip=([^ ]*) kavg=([^ ]*).*(MUL_MAT_ID [0-9]+:[0-9]+\+[0-9]+).*/\1 skip=\3 kavg=\4 \5/' |
  awk '/^run /{r=$2; print; next} {print "  " $0}
       r==6 && /ab=B/ {split($0,a,"kavg="); split(a[2],b," "); abk=b[1]}
       r==7 && !/skip=0\/0% kavg=6.00/ {bad=1}
       r==8 && ref=="" {split($0,a,"kavg="); split(a[2],b," "); ref=b[1]}
       END {d=abk-ref; if (d<0) d=-d; printf "A/B B kavg=%s plain-skip kavg=%s plain run clean=%s\n", abk, ref, bad?"NO":"yes";
            exit (bad || abk=="" || d>0.15)}'
