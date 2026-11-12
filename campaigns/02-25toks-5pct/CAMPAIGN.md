# Campaign 2: 25 tok/s inside +5 % perplexity

**Status: planning. Nothing in this campaign has been measured.** Every number below is *estimated* or comes
from campaign 1, and is labelled. Claim rule and budget are **drafts** until the first `prereg-R50` tag exists.

## The question

> Can a Snapdragon 888 phone, running a 7B-class model whose file is not changed, reach 25 tok/s by runtime
> optimisation alone, with perplexity at most +5 % over the served file's stock perplexity?

Campaign 1 answered the same question for 20 tok/s and +2 % with *yes*
(median 20.51, *device*, R49; [../../README.md](../../README.md)). This campaign takes that as given.

## Terms

| item | value | note |
|---|---|---|
| device | Samsung Galaxy Z Flip 3, Snapdragon 888, 8 GB | same phone as campaign 1 |
| model | **undecided**, see [candidates.md](candidates.md) | one frozen GGUF per model |
| quality gate | wikitext-2 test, ctx 512, batch 512, all chunks | as campaign 1 |
| budget | PPL ≤ stock served PPL × 1.05, one budget for everything that changes numerics (expert skip, two-stage head, dense numerics) | absolute value per model, see below |
| hard ceiling | PPL ≤ F16 PPL × 1.20 | unchanged |
| metric | llamadart `decodeTokensPerSec`, official prompt, 32 tokens, greedy, kit cool-down gate | unchanged, so the benchmark's definition does not change |
| claim rule (**draft**) | median of 10 plain runs ≥ 25.0 tok/s | fixed before the run, never a best run |

Budget values known so far:

| model | F16 | stock served | budget (× 1.05) | hard (× 1.20) |
|---|---|---|---|---|
| Granite-4.0-H-Tiny (served file d6-e6) | 9.0438 | 9.6051 | **10.0854** | 10.853 |
| other candidates | to be measured on the host | | | |

For Granite the served config of campaign 1 already spends 9.7886 (+1.91 %), so about +3.0 % of the new budget
is unspent there.

## What campaign 1 already priced (Granite, same budget)

In round 14 the owner opened the same +5 % cap for dense structural numerics (R41 decisions, 2026-11-07). The
host screens in R42 and R43 closed that line: router F32 → F16 passed but is worth about −0.32 ms; a low-rank
head screen failed on recall (closed at every rank); shared-expert pruning at 12.5 % cost about +6.7 %
(kill limit 1.012, measured 1.068); Mamba-2 head pruning was judged not worth its PPL (about 12 % per ms) and
was not tried. The cap was left unspent. See `reports/03_final.md` R42 and R43 and
[feasibility.md](feasibility.md).

## The arithmetic (estimated)

Printed 20.51 tok/s corresponds to a steady token of about 46.5 ms (graph 45.5 + outside 1.05). Printed
25 tok/s needs about 38 ms steady, so **about −8.5 ms (−18 %)**. The known levers inside the budget add up to
roughly −4 to −5 ms for Granite. Details and the model comparison are in [feasibility.md](feasibility.md).

## Open decisions (owner)

1. **Model set.** Options: (a) Granite, OLMoE-1B-7B, LFM2.5-8B-A1B; (b) the same plus a famous-brand dense
   control (Gemma 4 E2B) as a documented "ceiling" row; (c) add Ling-mini-2.0 / Ring-mini-2.0 as a memory-limit
   experiment. See [candidates.md](candidates.md). Dense 7B models are excluded by bandwidth, not by choice.
2. **"7B" strictness.** Does "7B-class" mean total parameters between about 7 and 9 B?
3. **Metric.** Keep the benchmark's definition (its time includes one final decode whose logits are unused,
   about 3 %, *estimated*, R48) or report both? Changing it must be labelled as a change.
4. **Budget scope.** One total budget (this file's draft) or a split like campaign 1's (+2 % speed-rule and a
   separate +5 % for dense numerics)?

## Plan

1. Freeze campaign 1 at commit `686f17d` (a `goal1-final` tag there is recommended; pushing tags was refused from the session that wrote this) and add this structure. *(this change)*
2. Decide the model set; write one card per model under [../../models/](../../models/) (`_TEMPLATE.md`).
3. Host work only, no phone: check llama.cpp support and the pinned `BASE` for each model; F16 and stock PPL;
   a PPL-versus-milliseconds frontier for expert skip, two-stage head and any dense requantisation.
4. Stop rule: if the best package inside the budget is worth less than about −6 ms (*estimated*), declare "no"
   with numbers and do not spend phone rounds.
5. Otherwise pre-register round 17 onward in [prereg/](prereg/TEMPLATE.md), commit, tag, then run on the phone.
   The R-numbers continue from campaign 1's R49, so the first entry here is **R50**.

## Repository work this campaign needs

- A new patch series on a newer llama.cpp `BASE` if a candidate's architecture is not supported at
  `620d2225c`; `patches/llama.cpp/0001-cpu-runtime.patch` (5,699 lines) is not edited. Splitting it by lever is
  part of that work.
- `tools/reproduce.sh` reads its model hash and reference values from environment variables (defaults are
  Granite's, so campaign 1 behaves as before).
- The private `lab/` layout that some tools still assume (`ppl.sh`, `host_bench.sh`, `checks/`) is replaced by
  arguments when those tools are first used for a new model.
