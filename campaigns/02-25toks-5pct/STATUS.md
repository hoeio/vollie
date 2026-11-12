# Campaign 2 status and hand-off

**Read this first, update it last.** Every session appends a line to the session log below and refreshes the
sections above it. Last updated: 2026-11-12 (session 1).

## State in one paragraph

Campaign 2 asks whether 25 tok/s is reachable on the Snapdragon 888 phone with a frozen 7B-class model and a
+5 % perplexity budget. The structure, feasibility arithmetic and candidate list exist. **Nothing has been
measured, no model is chosen, no pre-registration exists.** The honest current view (*estimated*): on Granite the
known levers inside the budget add up to about −4 to −5 ms of the −8.5 ms needed, so about 23 tok/s at best;
OLMoE-1B-7B is the most plausible model for 25, LFM2.5-8B-A1B the least. See
[feasibility.md](feasibility.md).

## Decisions already made by the owner

- Goal: *"On the SD888, with the 7B model untouched, can runtime optimisation reach 25 tok/s within at most
  +5 % perplexity?"* Campaign 1's answer (yes at 20 tok/s, +2 %) is taken as settled.
- The project is expected to be very long; thoroughness over speed.
- Prefer models people know, as far as feasibility allows (the owner saw the candidate analysis and said it was
  good, but **did not pick a set**).
- Repository: one repository, campaign structure (campaign 1 frozen, campaign 2 added beside it). Approved and
  merged (PR #2).
- Dense 7B models are excluded because the bandwidth and compute arithmetic rules them out
  ([feasibility.md](feasibility.md) §1), not by preference. There is no Qwen model in the 7 to 9 B sparse class.

## Open decisions (need the owner)

1. **Model set.** (a) Granite, OLMoE-1B-7B, LFM2.5-8B-A1B; (b) the same plus Gemma 4 E2B as a documented
   "famous dense" ceiling row; (c) add Ling-mini-2.0 / Ring-mini-2.0 as a memory-limit experiment.
   *Recommendation given:* OLMoE as the main model, Granite as the baseline, LFM2.5 as the generalisation check.
2. Whether "7B-class" means total parameters of about 7 to 9 B.
3. Metric: keep the benchmark's definition (includes one unused final decode, about 3 %, *estimated*, R48), or
   report both, clearly labelled.
4. Budget scope: one total +5 % (the draft) or campaign 1's split (+2 % speed rule, separate +5 % for dense
   numerics).

## Next actions, in order

1. Ask the owner for decisions 1 to 4 (use the options above). Record the answers here and in `CAMPAIGN.md`.
2. For each chosen model, write `models/<name>.md` from [../../models/_TEMPLATE.md](../../models/_TEMPLATE.md).
   Verify llama.cpp support and the architecture name against the pinned `BASE` (`620d2225c`) and the current
   llama.cpp; web-search it, the situation changes quickly.
3. Ask the owner (host, Apple M4) for the F16 and stock-file perplexity of each model, full wikitext-2, ctx 512,
   batch 512, via `tools/ppl.sh` or `tools/reproduce.sh` with the `REPRO_*` variables. Record the budgets in the
   model cards.
4. Host frontier (owner runs, you write the plan and the parser): PPL versus estimated milliseconds for expert
   skip thresholds (Granite: 117 to about 138), the two-stage head, and a dense requantisation at load (the
   only unpriced lever, see [knowledge.md](knowledge.md) §5).
5. **Stop rule:** if the best package inside the budget is worth less than about −6 ms (*estimated*), write the
   "no" with numbers into the R-log and do not spend phone rounds.
6. Otherwise write `prereg/R50-<title>.md` from the template, commit it, ask the owner to tag `prereg-R50`, and
   only then ask for the phone run.
7. For a model whose architecture is not supported at `BASE`: a new patch series on a newer llama.cpp commit under
   `patches/llama.cpp-<base>/`, split by lever; check that it applies; do not edit `0001`.

## Things to be careful about

- The owner's +5 % cap was already opened once in campaign 1 (R41 to R43) and the dense-numerics line closed with
  the cap unspent. Do not re-propose the closed items without new evidence: low-rank head screen, shared-expert
  pruning, a hard vocabulary subset, dynamic channel skipping, sub-4-bit dense containers without a kernel.
  List and numbers: [knowledge.md](knowledge.md) §4.
- Expert skip near the cap raises `en.txt` perplexity much more than wikitext-2 (130: +6.1 % versus +3.67 %).
- Single phone runs vary about ±20 %. Small effects need the in-run ABBA A/B and a resolvable σ.
- The first run after memory pressure is slow (cold start, page residency); the claim is about the median of
  warm, gated runs.
- Do not claim a result from a host measurement or an estimate.

## Session log

| date | session | what was done |
|---|---|---|
| 2026-11-12 | 1 | Read the repository; analysed feasibility; researched candidate models; created the campaign structure (PR #2, merged); wrote this hand-off (`CLAUDE.md`, `STATUS.md`, `knowledge.md`). No measurements. |
