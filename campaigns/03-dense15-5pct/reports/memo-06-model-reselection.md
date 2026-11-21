# c3 memo-06: reselection of three dense subject models for the 12 tok/s goal

Date: 2026-11-21. Status: **open (joint round 4)**. Section 1 is the lead's selection rule, written before any
candidate is priced or screened; it does not change after results. Sections 2 onward are filled by Bonnie, Ebony,
Tom and the lead. Nothing in this memo is measured on the phone.

Trigger: CAMPAIGN.md Amendment 1 (owner, 2026-11-21): goal 12 tok/s, exactly three dense 7B-class models from
different families on the same Snapdragon 888 phone, met only when all three pass; subjects reselected.

## 1. Selection rule (lead, 2026-11-21, written first)

**Eligibility (all must hold):**
1. Campaign 3's definition of dense 7B-class (CAMPAIGN.md): decoder-only, no expert routing, 6.0 to 8.5 B total
   parameters as published (embeddings included), publisher's stock instruct weights (unquantized QAT checkpoints
   count; fine-tunes, pruned or distilled derivatives do not).
2. Supported by llama.cpp at the campaign base `620d2225c`, or at a later base the lead proposes (an owner-accepted
   base change is listed as a deviation in each pre-registration). Support is checked on the web and in the tree, not
   recalled.
3. Weights downloadable from the publisher (Hugging Face) under a licence that permits this research use.
4. The served file at the chosen quantisation fits the phone: file plus KV at ctx 512 plus app at most 6.0 GB
   (*estimated*), so it can be held resident on 8 GB.
5. Three **different families**: different publisher and different pretraining lineage (two releases of one
   publisher's line count as one family).

**Ranking (applied in this order to the eligible pool; ties go to the next criterion):**
1. Estimated tok/s on the phone with campaign 3's priced levers (memo-01, -03, -05 methods), at the read rate R69
   measures; until R69 reports, at r = 19 and r = 35 GB/s both. The quantity is bytes and compute read per token
   at the served quantisation that stays inside F16 x 1.20.
2. Exploitable runtime sparsity: published or *host*-screened activation sparsity inside the +5 % budget at both
   `-b 512` and `-ub 1` (memo-02's column screen is the reference method).
3. Per-token fixed cost: LM-head and embedding bytes (vocabulary x width), KV width (GQA ratio).
4. Recency and quality as published (a model nobody would run is a weaker subject, not an ineligible one).

**Process rules:**
- The previous set (Mistral-7B-Instruct-v0.3, Qwen2.5-7B-Instruct; reserves Llama-3.1-8B, Yi-1.5-6B) is part of the
  pool, with no preference.
- Every candidate that is priced or screened is reported, not only the three chosen.
- No candidate is measured on the phone before the three are chosen and their served files frozen.
- The lead chooses the three and records two reserves; the owner has delegated the choice (2026-11-21: "proceed
  without asking"); the owner can still replace a model at any time.
- Honest framing: memo-03 put the strict-dense best case near 12 tok/s (*estimated*). 12 on all three is a hard
  target; a model that cannot reach it is reported with numbers, and the rule above is not bent to pass.

## 2. Candidate pool (Bonnie)

## 3. Pricing at 12 tok/s per candidate and challenge (Bonnie, Ebony)

## 4. Host screens (Tom)

## 5. Re-plan at 12 tok/s: levers, rounds, kill lines (lead)

## 6. Choice of the three and reserves (lead)
