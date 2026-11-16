# memo-25: routing prior toward recently used experts (memo-21 item 2), host study

Date: 2026-11-16. Author: Tom (`claude-opus-5-5`); filed by the lead. Not pre-registered and not an R-entry:
every number is *host* (Apple M4, builds without KleidiAI) or *estimated*. Nothing here is a phone number.
Raw logs, job lists and exact commands: `lab/logs/r64-prior/queue.txt` (untracked; the label "r64" is a folder
name only, R64 is not assigned).

## Verdict

The prior fits the +5 % budget on both models, alone and on top of num. It is a residency-only lever: bytes
computed per token do not change, so with pages resident it is worth 0. At the phone state R62 measured
(1.9-2.4 GB of the file cached at run start, 7-14 MB of file read per decode run, majors mostly zram swap-ins,
memo-23/24) it is worth about 0.2-0.7 ms/token on OLMoE and about 0 on Granite (*estimated*). It pays only on
low-file-cache days.

## Rule (patch 0007, numerics change, off unless set)

- `VOLLIE_PRIOR=<lambda*1000>` (unset: off; `0`: counters only, bit-identical), `VOLLIE_PRIOR_N` (default 4),
  `VOLLIE_PRIOR_J` (default 0, always keep the top J by true probability).
- For layer l at KV position p, H = experts selected at l at positions p-1..p-N (before skip). Top-k is taken
  over probs x e^lambda for e in H (= +lambda on the logit for softmax gating). Only `ffn_moe_argsort-<l>` is
  replaced; mixing weights stay unbiased, so 0003's skip composes unchanged.
- Decisions are made by thread 0 in row order over a per-layer position-keyed ring, so a 512-token batch decides
  what 512 one-token graphs decide given equal probs. Checked: `tools/c2_prior_trace.py replay` differs from the
  op in 0 of 15,903 rows (OLMoE) and 0 of 40,527 (Granite) at `-b 512`, `-ub 32`, `-ub 1`.
- Outcomes still diverge more than stock between settings (expert sets differing between `-b 512` and `-ub 1`:
  7.5 % stock, 29-31 % at lambda=1 N=4, OLMoE): the both-settings gate is necessary.
- Granite has no ROPE node (NoPE); position = admitted KV cells in the mask row - 1 (byte-identical to rope
  positions on OLMoE).
- Not covered: KV state loaded from a file (history starts empty), interleaved multi-sequence batches.
- Identity: knob unset and `VOLLIE_PRIOR=0` equal the 0001-0006 tree per chunk on 3 chunks at all three
  settings, both models, off and num; 60-chunk stock equals R50 and R59.

## Full sets (*host*, `VOLLIE_PRIOR=500 VOLLIE_PRIOR_N=16`, same on both models)

| model and arm | `-b 512` | `-ub 1` | reference | budget |
|---|---|---|---|---|
| OLMoE num + prior | 12.2454 | 12.2435 | num 12.1053 / 12.0946 | 12.3003, inside at both |
| OLMoE prior alone | 11.8587 (+1.23 %) | not run | stock 11.7146 | inside at `-b 512` only |
| Granite num + prior | 9.5487 | 9.5478 | num 9.5173 / 9.5076 | 9.6879, inside at both |
| Granite prior alone | 9.3172 (+0.98 %) | not run | stock 9.2266 | inside at `-b 512` only |

The prior-alone rows lack `-ub 1` and are not gated.

## Frontier (60 chunks, *host*; re-read columns *estimated* with memo-18's LRU simulator)

- OLMoE: lambda .25 costs about 0 %, .5 N16 +0.76 % (`-ub 1` +1.10), .75 +2.55 %, 1 +7.5 to +11.4 %, 2 +178 % or
  more. Re-read per 32 tokens at 1.8 GB: stock 1766 MB, .5 N16 1021 MB; num 1445, num + .5 N16 971.
- Granite: .25 N16 +0.90 %, .5 N16 +1.50 %, .75 +2.51 %, 1 +3.6 to +4.9 %. With num its decode set already fits
  at 2.0 GB or more (num 43 MB, num + .5 N16 7 MB at 1.8 GB).
- J=2 did not help either model. Halving the 32-token working set (memo-21's target) needs lambda 1-2, far
  outside the budget.
- Consecutive-token expert overlap (memo-21 §8 open question): OLMoE 0.375, Granite 0.49.

## Estimated decode gain (memo-18 rates: 1.4 ms/MB OLMoE, 2.5 Granite, per 32-token run)

| change | 1.6 GB | 1.8 GB | 2.0 GB | 2.2 GB | 2.4 GB |
|---|---|---|---|---|---|
| OLMoE num -> num + prior | -21.8 | -20.7 | -16.6 | -14.0 | -12.3 |
| Granite num -> num + prior | -8.0 | -2.8 | 0 | 0 | 0 |

These assume LRU behaviour at that cache size, which R62 does not show.

## What generalised and what did not

- Generalised: the rule, code path and setting are the same on both models; both fit the budget.
- Did not: PPL cost per unit lambda (OLMoE ~0 % at .25, Granite +0.9 %), and residency value (large for OLMoE,
  small for Granite).

## Corrections to the brief

- memo-21's 0.08 ms per expert MB is a compute rate and does not apply to this lever.
- memo-21 §N2's kill rule (distinct-slice drop >= 25 %) would kill it: the drop is 7-22 % at the chosen
  setting, while the simulated re-read drop is 33-90 % at 1.8 GB. A pre-registration must pick one.
- zsh caution: `env $E` does not split a multi-variable string; use bash (the runners do; no result affected).

## Suggested next step

If a phone test is wanted: a cheap kit arm, num against num + `VOLLIE_PRIOR=500 VOLLIE_PRIOR_N=16`, recording
decode `ru_inblock` and `pswpin`. Expected: no effect unless the file cache is low.

## Files

`patches/llama.cpp-c1-olmoe/0007-c2-routing-prior.patch` (ggml-cpu.c, +316, applies on 0001-0006),
`tools/c2_prior_trace.py` (stats, cmp, sim, replay, selftest), README section for 0007.
