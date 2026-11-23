# Campaign 3: a dense 7B-class model at 15 tok/s inside +5 % perplexity

**Status: planning (2026-11-18). Nothing in this campaign has been measured.** Every number below is *estimated*
or quoted from campaigns 1 and 2 with its label. The terms were set by the owner on 2026-11-18. Campaign 2 is still
open and is finished first (R68).

*Correction 2026-11-21: campaign 2 is closed (owner, 2026-11-21; goal met on OLMoE, see campaign 2 CAMPAIGN.md
Result). The sentence "Campaign 2 is still open and is finished first (R68)" was true until then.*

## Amendment 1 (owner, 2026-11-21): goal 12 tok/s on three dense models

Owner, 2026-11-21: "Change the goal to 12 tok/s, run three different dense 7B models on the same chipset, and the goal
is met when all three pass. The subject models need to be reselected." This replaces the goal and model rows of the
terms table below; every other term (file not changed, +5 % budget at `-b 512` and `-ub 1`, F16 x 1.20 ceiling,
quality gate, metric, claim rule, start state, calibration tables) is unchanged.

| item | before (2026-11-18) | from 2026-11-21 |
|---|---|---|
| goal | 15 tok/s, median of 10 plain runs | **12 tok/s**, median of 10 plain runs, **on each of three models** |
| models | at least two, different families | **exactly three** dense 7B-class models (definition below), **different families**; the campaign goal is met only when **all three** pass the claim rule |
| subjects | Mistral-7B-Instruct-v0.3, Qwen2.5-7B-Instruct; reserves Llama-3.1-8B, Yi-1.5-6B | **reselected** (memo-06); the earlier set is a candidate pool, not a decision |
| chipset | Snapdragon 888 (Z Flip 3) | same; all three models on the same phone |

Lead's notes:
- The goal was lowered by the owner after estimates were known (memo-03: strict dense best about 12 tok/s,
  *estimated*). It is recorded as an owner decision, not as a result; no estimate is re-scored as a pass.
- The folder keeps its name `03-dense15-5pct` (files are not renamed); the goal is the one in this amendment.
- A model that does not reach 12 is reported as a result; it is replaced only by the owner (COMMON_RULES).
- *Note 2026-11-22: subjects chosen (memo-06 section 6): Yi-1.5-6B-Chat, Mistral-7B-Instruct-v0.3, DeepSeek-LLM-7B-chat; reserves Llama-2-7b-chat, Qwen2.5-7B-Instruct. ChatGLM3-6B excluded by the owner.*
- *Correction 2026-11-22: subject 2 is Llama-2-7b-chat, not Mistral (lead withdrew a post-hoc tie reading; memo-06 section 6). Subjects: Yi-1.5-6B-Chat, Llama-2-7b-chat, DeepSeek-LLM-7B-chat; reserves Mistral-7B-Instruct-v0.3, Qwen2.5-7B-Instruct.*
- Selection of the three is done before any of them is measured on the device, by a rule written first
  (memo-06 section 1), and frozen together with each served file.

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
| claim rule | median of 10 plain runs ≥ 15.0 tok/s (*note 2026-11-21: read 12.0, per model, per Amendment 1*), positions fixed before the run, never a best run; power saving off, deep-sleeping list state recorded, charger out | campaign 2's go rule (memo-31 §4) applies before a claim round |
| start | campaigns 1 and 2's final state: runtime series, kit, protocol, lessons | as campaign 2's "start from Vollie 1" rule |

*Owner decision 2026-11-19 (calibration-derived tables: thresholds and hot-first orders).* Owner: "Treat them as data read at runtime, but manage them as derived artifacts attached to the weights, not as code." Lead's reading:

- Each table is a sidecar file next to the served model, not compiled into the runtime.
- It is generated by a committed script from the served file and a fixed calibration text (wikitext-2 train), both named in the table's header.
- The table records the served file's sha256 and its own sha256.
- The runtime refuses a table whose model sha does not match.
- Every pre-registration lists the table's sha256 next to the model's.
- A new table is a new artifact and needs its own PPL check at `-b 512` and `-ub 1`.

*Owner decision 2026-11-22 (trained guess heads).* Owner: "3번은 허용합니다. 모델마다 헤드를 학습해서 진행하세요." (Lever M of
memo-09: a small next-token guess head, trained on a host, used only to propose tokens for exact lookahead.) Lead's reading:

- A head is a sidecar artifact like the tables above: one per served model, trained by a committed, model-generic
  script from the served model and a fixed, named training text; it records the served file's sha256 and its own; the
  runtime refuses a head whose model sha does not match, and decodes normally without one.
- The head only proposes; the model verifies. Greedy output must stay identical, checked on the fixed prompt set and
  the official prompt. It changes no numerics of the served model, so it spends no PPL budget; the identity check is
  greedy-token identity, not perplexity.
- Every pre-registration lists the head's sha256 next to the model's. Training data and its licence are named in the
  head's header. A model swap needs a new head (memo-09; owner told 2026-11-22).
- The served model file itself stays unchanged.

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

*Added 2026-11-19: RWKV-7 G1 is not a subject (owner, 2026-11-19): "There is no need to include RWKV-7 G1; the subject models and their reserves are enough."*

*Owner decisions 2026-11-22 (after memo-10):* "BOS 설정은 허용하지 않습니다. HoeioUser/vollie3에 올리는 걸 허용합니다. 다만,
실제 돌파구가 나올 때까지 R73을 허용하지 않겠습니다. 진전이 확실한 것들에만 테스트를 허용합니다." Lead's reading:

- **DeepSeek is served unmodified:** `deepseek7b-q4_0.gguf` (sha256 `f403ddc2…5118`), not the one-field BOS copy
  (`f195d24f…a9ec`). The freeze, budget and every later number use that file as it is. Host values measured on the BOS
  copy stay as history and are not reused for the budget. A runtime that inserts BOS on its own is not used either,
  unless the owner allows it.
- **Public hosting:** the three frozen files may go to a new public repo `HoeioUser/vollie3`, one folder per model with
  the publisher's licence. Upload only after the freeze checks pass and when a round needs the files on the phone.
- **Device rounds are held:** R73 (the first real decode) does not run until there is a real breakthrough. A device test
  is proposed only for a lever whose progress is certain on host evidence, with that evidence named in its
  pre-registration. Host research, host checks and the freeze continue.

*Owner decision 2026-11-22 (later):* "GPU 측정은 예외로 허용합니다. x3 프로브 준비하세요." The x3 GPU probe (design:
`reports/memo-09-work/tom-host.md` §1.5) may run on the phone as an exception to the device hold. It is the round after
R73's number: **R74** (R73 stays reserved for the held first real decode). Scope is the GPU lane only; no model is loaded.

*Owner decision 2026-11-23:* "측정 시도 허락합니다" (in answer to the lead's report of memo-12: the GPU join cost, with
device cells listed in `reports/memo-12-work/bonnie-joins.md` §5 and `ebony-joins.md` §7.1). The X4 join probe may run
on the phone as a second exception to the device hold: **R75**, kit r35. Scope is the GPU lane's launch and join cost
(and its residency at a larger GPU share); no model is loaded. R73 stays reserved for the held first real decode.
