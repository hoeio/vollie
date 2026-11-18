# Campaign 3: a dense 7B-class model at 15 tok/s inside +5 % perplexity

**Status: planning (2026-11-18). Nothing in this campaign has been measured.** Every number below is *estimated*
or quoted from campaigns 1 and 2 with its label. The terms were set by the owner on 2026-11-18. Campaign 2 is still
open and is finished first (R68).

## The question

> Can the same Snapdragon 888 phone run a **dense** 7B-class model, whose file is not changed, at 15 tok/s by
> runtime changes alone, with perplexity at most +5 % over the served file's stock perplexity?

## Terms (owner, 2026-11-18)

| item | value | note |
|---|---|---|
| device | Samsung Galaxy Z Flip 3, Snapdragon 888, 8 GB | same phone as campaigns 1 and 2 |
| goal | **15 tok/s**, median of 10 plain runs | owner first named 20 tok/s, then lowered it to 15 after the lead's arithmetic (§ Arithmetic) |
| model | dense 7B-class, stock weights; definition below | at least two models (generality rule) |
| file | not changed; the runtime may use it in any way, for example a dense model run like a MoE (activation sparsity, neuron or column skipping, layer or head skipping, clustering experts at runtime) | owner: "the model is not touched, how it is used is free" |
| served quantisation | free, chosen once per model and frozen before that model's first device round | owner: "free inside the budget" |
| budget | PPL ≤ stock served PPL × 1.05, one budget for everything that changes numerics, at both `-b 512` and `-ub 1` | as campaign 2 |
| hard ceiling | PPL ≤ F16 PPL × 1.20, for the served file at stock and for every configuration | as campaign 2; it is what bounds a low-bit served file |
| quality gate | wikitext-2 test, ctx 512, batch 512, all chunks | as campaigns 1 and 2 |
| metric | llamadart `decodeTokensPerSec`, official prompt, 32 tokens, greedy, kit protocol (r16 gate) | unchanged. A method that drafts or commits several tokens per step counts committed tokens, keeps greedy output identical, and is scored on a fixed prompt set as well as the official prompt |
| claim rule | median of 10 plain runs ≥ 15.0 tok/s, positions fixed before the run, never a best run; power saving off, deep-sleeping list state recorded, charger out | campaign 2's go rule (memo-31 §4) applies before a claim round |
| start | campaigns 1 and 2's final state: runtime series, kit, protocol, lessons | as campaign 2's "start from Vollie 1" rule |

### "Dense 7B-class" (lead's definition, accepted by the owner)

- A decoder-only transformer whose published architecture has no expert routing: every token uses every FFN weight
  at stock.
- 6.0 to 8.5 billion parameters in total, embeddings included, as published.
- Stock (instruct) weights from the publisher; no fine-tune, no pruned or distilled derivative.
- Runs on llama.cpp at the campaign's base commit, or at a later base that the lead proposes and the owner accepts.
- At least two models from different families; each stays a full subject (COMMON_RULES).

*Added 2026-11-19: Owner, 2026-11-19: "Only unquantized QAT checkpoints count as stock weights." Lead's reading: a publisher's unquantized QAT checkpoint (for example Google's Gemma QAT release in BF16/F16) is stock weights for campaign 3; pre-quantized QAT files (such as a publisher's Q4_0 GGUF) are not. The served file is quantized by us from the unquantized QAT checkpoint and frozen as usual; stock PPL and F16 PPL are measured on that checkpoint. Whether Gemma 4 E4B counts as "dense 7B-class" is still open with the owner.*

*Added 2026-11-19: Gemma 4 E4B is not a subject (owner, 2026-11-19). Reason, in the owner's words: "it is hard to see it as 'dense 7B-class'. In compute and intelligence it is about a 4.5B dense model; in storage and memory, about an 8B model." It sits outside the 6.0 to 8.5 billion definition above on the quantity that matters for speed, so the question left open in the previous note is closed.*

## Arithmetic (*estimated*; campaign 2 `feasibility.md` §1)

- Decode reads every weight once per token. A 7.5 B model at Q4_0 is about 4.0 GB per token; at about 2.5 bits per
  weight, about 2.3 GB.
- 15 tok/s needs about 60 GB/s at Q4_0. Measured dense read rate on the phone: about 19 GB/s (campaign 1, *device*,
  most likely with power saving on). Nominal LPDDR5 peak: about 51 GB/s (recalled, not checked).
- Bytes per token for 15 tok/s: about 1.3 GB at 19 GB/s, about 2.3 GB at 35 GB/s.
- Compute: an int8 dot-product ceiling for three big cores puts a 7B dense model near 10 tok/s at Q4 (campaign 2
  feasibility; it was priced with power saving on).
- Draft-model speculation lost on this CPU (R20: compute-bound at any batch above one); n-gram, EAGLE-3 and MTP
  drafting lost on the host (campaign 1).
- So 15 tok/s needs roughly a **2 to 3 times** cut in bytes and compute per token. The cut has to come from a
  low-bit served file inside the hard ceiling, from runtime sparsity inside the +5 % budget, or from both. Whether
  that is possible is the first research question.

## Open (to be settled before the first pre-registration)

1. Candidate models (web check required; `candidates.md`).
2. The phone's dense read rate and int8 ceiling **with power saving off** (a short device probe).
3. Which runtime sparsity methods fit the budget at both `-b 512` and `-ub 1` (host screens).
4. The prompt set for multi-token methods, if any are tried.
5. R-numbers continue from campaign 2's last entry.
