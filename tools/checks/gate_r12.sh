#!/bin/sh
# Vollie r13: tools/research_parse.py --gate on the r12 device JSON must reproduce Bonnie's worked example
# (reports/memos/round13_plan.md §B): Q3K_KERN run 1 counted A {1 8 9 12 13 16 17} / B {6 7 10 11 14 15}, MUL_MAT_ID
# -6.27 ± 3.10, pooled verdict unresolved. Exit 1 otherwise.
cd "$(dirname "$0")/../.."
out=$(python3 -I tools/research_parse.py reports/device/r12_2026-11-07.json --gate 10 | sed -n '/== block gate/,/EXPERT_SKIP/p')
echo "$out"
echo "$out" | grep -q 'run 1: .*counted A {1 8 9 12 13 16 17} B {6 7 10 11 14 15}' &&
echo "$out" | grep -q 'run 1   counted A 7 / B 6  majflt A 5.4 B 6.4  MUL_MAT_ID -6.27 ± 3.10' &&
echo "$out" | grep -A1 '^    pooled' | grep -q -- '-> unresolved' && echo PASS || { echo FAIL; exit 1; }
