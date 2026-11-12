# Campaign 2 status and hand-off

**Read this first, update it last.** Every session appends a line to the session log below and refreshes the
sections above it. Last updated: 2026-11-12 (session 3).

## State in one paragraph

Campaign 2 asks whether 25 tok/s is reachable on the Snapdragon 888 phone with a frozen 7B-class model and a
+5 % perplexity budget. The owner's decisions are in (session 3): three models with stock weights, one +5 %
budget, metric unchanged. **Nothing has been measured, no served file is frozen, no pre-registration exists.**
All three models are supported at the pinned `BASE` (web check 2026-11-12), so no new llama.cpp base is needed
for support. The next step is the owner's host screen that fixes one served file per model
([reports/memo-01-candidates-and-host-screen.md](reports/memo-01-candidates-and-host-screen.md)). The honest
current view (*estimated*): Granite about 23 tok/s at best; OLMoE the most plausible for 25, but its byte
advantage over Granite is only about 11 to 25 % (the layer count, 16 against 40, is the larger advantage) and
its small experts may quantise badly; LFM2.5 reads about 1.3 times Granite's bytes and is the least likely.

## Decisions already made by the owner

- Goal: *"On the SD888, with the 7B model untouched, can runtime optimisation reach 25 tok/s within at most
  +5 % perplexity?"* Campaign 1's answer (yes at 20 tok/s, +2 %) is taken as settled.
- The project is expected to be very long; thoroughness over speed.
- Prefer models people know, as far as feasibility allows.
- Repository: one repository, campaign structure (campaign 1 frozen, campaign 2 added beside it).
- Dense 7B models are excluded because the bandwidth and compute arithmetic rules them out
  ([feasibility.md](feasibility.md) §1), not by preference.
- **2026-11-12, session 3** (full text in [CAMPAIGN.md](CAMPAIGN.md), "Owner decisions"):
  1. Model set (a): Granite-4.0-H-Tiny, OLMoE-1B-7B, LFM2.5-8B-A1B.
  2. "7B-class" = total parameters of about 7 to 9 B.
  3. Metric: the benchmark's definition, unchanged; no second metric.
  4. Budget: one total +5 % over the served file's stock PPL; hard ceiling F16 × 1.20.
  5. **Stock weights, not abliterated.** Campaign 1's Granite file (d6-e6) is therefore not a campaign-2 file
     and its PPL table and budget value (10.0854) do not apply; a stock-weights Granite file is measured anew.
  6. Served file per model is fixed after a host screen (publisher Q4_0 against a campaign-1-style mix), by a
     rule written before the screen (memo-01, part C).
  7. Base or instruct: no owner preference; the lead chose Instruct (OLMoE `0125-Instruct`).
  8. LFM2.5 stays for research use only (licence "lfm1.0" is not open source), third in priority.

## Open decisions (need the owner)

None at the moment. The next one comes from the host screen: if no file of a model passes the selection rule,
that model goes back to the owner.

## Next actions, in order

1. **Owner (host, Apple M4):** run the host screen in memo-01 part B, in the order OLMoE, Granite, LFM2.5: F16,
   publisher Q4_0 and the mix, full wikitext-2, ctx 512, batch 512; send back the `Final estimate` lines, file
   sizes and sha256 values. Also still to do by the owner: tag `goal1-final` at `686f17d`.
2. Apply the selection rule (memo-01 part C) as written; write `models/<name>.md` for each chosen file from
   [../../models/_TEMPLATE.md](../../models/_TEMPLATE.md) with hash, F16, served PPL, budget and hard ceiling.
3. Bonnie and Tom: per-token cost structure of each chosen file (`tools/cost.py` logic; the owner runs repository
   tools, or read the GGUF header with own short code) and a price per lever in bytes and ms (*estimated*).
   Which campaign-1 levers exist for `olmoe` and `lfm2moe` at all (expert skip, fused expert op, Q3_K x4 kernel,
   two-stage head) has to be read from the patch, not assumed: `0001` was written for `granitehybrid`.
4. Host frontier (owner runs, the team writes the plan and the parser): PPL against estimated milliseconds for
   expert skip, the two-stage head and a dense requantisation at load ([knowledge.md](knowledge.md) §5).
5. **Stop rule:** if the best package inside the budget is worth less than about −6 ms (*estimated*) of what the
   model needs, write the "no" with numbers into the R-log and do not spend phone rounds on that model.
6. Otherwise write `prereg/R50-<title>.md` from the template (agent models: clyde `claude-fable-5-2`, bonnie and
   tom `claude-opus-5-5`, annie `claude-sonnet-5-5`), commit it, ask the owner to tag `prereg-R50`, and only then
   ask for the phone run.
7. New patch work goes into a new series beside `0001`, split by lever; `0001` is not edited.

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
- "Stock" has two meanings here: stock weights (the publisher's model, not abliterated) and stock served PPL
  (the served file run by the unmodified runtime). Write which one is meant.
- A weaker served file gets a looser absolute budget (+5 % of a larger number). The guards are the selection
  rule (served ≤ F16 × 1.08) and the hard ceiling (F16 × 1.20); do not pick a file because its budget is roomy.
- llama.cpp issue #29002 (co-batched sequences corrupt the LFM2 short-conv state on CPU) bears on LFM2.5's
  perplexity and identity checks; see memo-01.
- Subagents defined in `.claude/agents/` are only listed if the files exist when the session starts. In session 3
  they appeared late; the first research task ran on a general-purpose stand-in given `bonnie.md`
  (`claude-opus-5-5`). After they loaded, all four stated their pinned IDs.

## Session log

| date | session | what was done |
|---|---|---|
| 2026-11-12 | 1 | Read the repository; analysed feasibility; researched candidate models; created the campaign structure (PR #2, merged); wrote this hand-off (`CLAUDE.md`, `STATUS.md`, `knowledge.md`). No measurements. |
| 2026-11-12 | 2 | Pinned the four-agent team in `.claude/agents/` (full model IDs) and described it in `CLAUDE.md`; added the "Agent models" field to the prereg template; added the campaign 1 Phase 0 model history to `candidates.md`. Pushed to `main` at the owner's instruction. No measurements. |
| 2026-11-12 | 3 | Team check: all four agents state their pinned model IDs (`CLAUDE_CODE_SUBAGENT_MODEL` unset). Owner answered the four open decisions plus file variant, served-file procedure and LFM2.5 licence (see above). Web re-check of the three candidates: all supported at `BASE` `620d2225c`. Wrote memo-01 (candidate re-check, host screen plan with commands, selection rule). No measurements. |
