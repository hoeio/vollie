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

Full arithmetic and reasoning: [memo-06-work/clyde-replan.md](memo-06-work/clyde-replan.md). Every number here is
*estimated*. R69's bands are the inputs until R69 reports.

**5.1 What 12 tok/s requires.** The token must take at most 83.3 ms, with a fixed cost O of about 8 ms (6 to 12).
Two lines must hold at once:
- **Byte line:** bytes read per token ≤ r × 75 ms. That is 1.43 GB at r 19, 1.81 at 24, 2.64 at 35 and 3.39 at 45.
- **Weight line:** touched weights × the kernel's cost per weight ≤ about 5.4 G stock Q4_0 weights. This is at
  R69's central c_b of 40.4 GB/s on X1 + 2 A78; the band is 4.7 to 6.0.

What the lines mean:
- A dense Q4_0 body is out on bytes. At 7.0 B multiplied weights (4.0 GB) it needs r ≥ 54 GB/s, which is above
  the bus. At 6.0 B (3.4 GB) it needs 45 GB/s.
- **Every 6.0 to 8.5 B dense model is also out on compute.** At the stock Q4_0 cost per weight it stays below 12
  tok/s at any read rate: about 9.5 tok/s at 7.0 B multiplied and 10.9 at 6.0 B.
- Fewer bits shrink the bytes but not the compute, unless the low-bit kernel costs no more per weight than Q4_0.
  Q3_K costs about twice as much (*device*).

**The corner that reaches 12 on one subject** needs all three of these:
1. A served file of about 3.1 to 3.7 bpw. The floor is the F16 × 1.20 ceiling, about 3.05 bpw on Llama-3-8B
   (*published*). The top is residency at C about 3.1 to 3.4 GB.
2. Column sparsity at d about 0.60, with an exact low-bit column kernel at k ≤ 1.3 (P 7.0 B) to 1.5 (P 6.0 B).
   The best measured kernel today is K1 on Q4_0 at k 1.53 to 1.64 (*host*).
3. r of about 20 to 28 GB/s or more, or a GPU lane or 4th core that adds compute.

**Lead's prior:** P(all three pass) is about 0.05 to 0.15. This is a judgement, not a price; §3 replaces it. It is
told to the owner now.

**Levers this puts first** (Bonnie and Ebony price them in §3):
- a fast arm64 kernel for one sub-4-bit type, bit-identical to ggml's reference for that type;
- an exact column kernel for that type;
- the GPU as a compute lane (R69 Rule D);
- X1 + 3 A78 threading.

**5.2 R69 under the amendment.** R69 loads no model, so its measurements and bands stay valid. Its Rules A to C
and §6 named the old subjects and 15 tok/s. Before any R69 data exists, they are adapted in
[prereg/R69-amendment-1.md](../prereg/R69-amendment-1.md):
- **Rule A** is scored at reference densities 0.60 and 0.65. Each new subject gets a derived verdict, with the
  formula fixed now. It does not apply to non-Q4_0 files.
- **Rule B's** inner kill becomes "any of the three subjects".
- **Rule C's** consequences are restated for any Q4_0 body of 6 B or more. Below 3.4 GB, subjects are frozen at
  files that fit C.
- The **goal** reads 12 on each subject.
- **Byte and weight lines** are reported from R69's numbers.

Nothing that is measured or predicted changes.

**5.3 Rounds and kill lines.** The next free R-number after R69 is R72; the next campaign-3 kit is r32.

| stage | content | kill line |
|---|---|---|
| S0 (now) | §2 to §4; choice (§6) at R69's r if reported, else at r 19 and 35 | – |
| S1 = R69 | probe kit r29 + amendment 1; no model; can run now | **K0** per subject: no priced lever set meets both lines at R69's r, c_b, C → "out of reach by arithmetic", written with numbers |
| S2 = R72 | kit r32, no model: D2 rows (K1, Y1b), per-type gemv rows on synthetic arenas (Q4_0, Q4_K, IQ4_NL, Q3_K, IQ3_S, Q2_K at least), X1 + 3 A78, D3 header | **K-type**: a type with no room under the weight line for any subject is dropped from the freeze |
| S3 | freeze per subject (*host*, owner's laptop). Quantise from the publisher's BF16 with stock `llama-quantize`. Measure full-set F16 and stock PPL. Check stock ≤ F16 × 1.20 and resident ≤ C − 0.15 GB. Take the lowest priced token time. Write the budget (stock × 1.05, absolute) in the model card, with the sha256s | **K-F**: no type fits the ceiling and C → reported; the lead proposes a reserve, only the owner replaces |
| S4 | baseline: all three subjects in one session; D1 residency over 20 passes; median of 10 at campaign 2's best bit-identical configuration | **K1** per subject: baseline (*device*) × upper-band gains of the open levers < 12.0 → "out of reach with priced levers" |
| S5 | lever rounds, one pre-registration each. Numerics levers first pass full-set PPL at `-b 512` and `-ub 1` per subject. **Every lever is measured on at least two of the three subjects in the same session** (all three when the kit holds them), in-run ABBA, reported per model with what generalised and what did not | each lever's own kill line; K1 re-applied after each round |
| S6 | go block: median ≥ 12.15 and at most 2 of 10 runs below 12.0, uncapped clocks. Then one claim per subject: N = 10 plain runs, positions fixed, no run dropped, power saving off, deep-sleeping list recorded, charger out; claim configuration within budget at both settings and ≤ F16 × 1.20 | pass per subject: median ≥ 12.0. **Goal: all three pass.** A failed claim stays in the log; another attempt needs a new pre-registration and a changed configuration |

**5.4 Owner questions (a) to (d) from before the amendment.** These are lead decisions, provisional until the
owner confirms them (owner delegated, 2026-11-21). Reasons are in the work file §3.
- **(a)** Y1a side file is a weight-derived artifact: **yes**. COMMON_RULES 1 (R42 amendment 2) and the 2026-11-19
  sidecar regime already cover it.
- **(b)** Twice the storage: **yes, for the research kit**. Free storage is recorded per round.
- **(c)** A column-major copy: **yes**, as an exact re-layout. A 1-bit proxy: **yes**, as lossy derived data
  inside the budget with its own PPL check. This is the least certain of the four.
- **(d)** The served Q4_0 file is **superseded**. Each subject is quantised by us from the publisher's BF16 at the
  base, into one file, with every sha256 recorded and no third-party GGUF. The type is chosen in S3, not assumed
  to be Q4_0.

## 6. Choice of the three and reserves (lead)
