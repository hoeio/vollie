# c3 memo-12 work, round 23 (Bonnie): the fixed prompt set; what the claim prompt itself gives a guess head; what a load is made of

- **Date:** 2026-11-24. **Author:** bonnie, `claude-fable-5-2`. Follows [bonnie-joins-r22.md](bonnie-joins-r22.md). Not
  committed. No verdict here. **Ebony has not reviewed this file.**
- **Labels.** *device*, *host*, *estimated*, *source* as before. Triples are Yi / Llama-2 / DeepSeek.
- **New *host* numbers, all read from files that already exist** (Tom's dumps under `lab/c3/head/`, with my own
  short code): prompt lengths in tokens per subject, and tokens per pass on the official prompt recomputed from
  his saved guesses. No model ran, no dump was made. The stack re-prices are *estimated*
  (`bonnie_floor_r14.py` functions, run inline). No web source.

## 0. Answer first

1. **The fixed prompt set is defined and adopted by the lead in the R73 plan, but four things about scoring on
   it are not written anywhere:** how many tokens per prompt, whether each prompt is gated, how 32 prompts
   become one number, and what the number must show. §1.4 proposes each.
2. **Its prompts are short** (*host*): with the chat wrapper, 25 to 383 tokens on Yi (median 40) and 14 to 391 on
   Llama-2 (median 29). With 128 new tokens the longest needs 511 / 519 positions, so **n_ctx 512 fails on one
   Llama-2 prompt by 7 tokens**; the plan's 1,024 is right. With 32 new tokens per prompt everything fits 512.
3. **The set and the claim prompt are different regimes.** On the set a decode step sits at about 100 positions
   and a run lasts 13 s; on the claim prompt it sits at about 200 positions and lasts 3 to 4 s. Attention is
   cheaper on the set, heat is worse.
4. **A finding that changes a price: the claim is scored on the official prompt, and on that prompt Llama-2's
   head is right 6 times in 31** (*host*, Tom's saved guesses, recomputed today). Tokens per pass there: **Yi
   1.39, Llama-2 1.14** with the best heads, against the 1.33 and 1.45 from the prompt set that every stack
   uses. With prompt lookup in front of the head Llama-2 reaches 1.33 to 1.39 (Tom's table).
5. **Re-priced on the claim prompt** (*estimated*; r14's rung 6 at K 300, `linear` / `pair`):

   | | as the stacks have it (set) | on the official prompt |
   |---|---|---|
   | Yi | 10.82 / 11.10 | **11.31 / 11.61** (head) |
   | Llama-2 | 10.80 / 11.12 | **8.60 / 8.85** (best head alone); 9.95–10.38 / 10.25–10.70 (lookup, then head) |
   | DeepSeek | 10.35 / 10.64 | unknown: no head, no dump |

   Llama-2 with its best head alone is barely above the width-1 lane (8.53). **For the claim, the guess source
   has to be chosen on the official prompt's own behaviour, and that is 31 guesses per model.**
6. **A load is one number in every report (`reload_ms`); its parts were never timed.** §3 lists the steps from
   source with an estimate each and the three fields a kit would need to split it.

## 1. The fixed prompt set

### 1.1 Definition (*source*)

| | |
|---|---|
| file | `lab/c3/head/prompts/eval.txt`, sha256 `ff1f32e3…72c0`, 32 prompts, 11,125 bytes |
| rule | databricks-dolly-15k at HF commit `bdd27f4d`; record index ≡ 7 (mod 100); at most 2,500 characters; the first 32 in file order (`tools/c3_head_prompts.py`) |
| adopted | by the lead, R73 plan §6: "used for every multi-token method, beside the official prompt" |
| conditions written there | nothing is tuned on it from adoption on; host values on it keep their label; it is committed with generator and attribution (CC BY-SA 3.0); **runs on it use nCtx 1024** |
| why it exists | `CAMPAIGN.md`, metric row: a multi-token method "is scored on a fixed prompt set as well as the official prompt" |
| R73 | does not run it (plan line 144: "no fixed-prompt-set runs") |

### 1.2 Lengths (*host*; from `eval.ptok` and `eval.rows` of Tom's CPU dumps, chat-formatted as decoded)

| | Yi | Llama-2 | DeepSeek |
|---|---|---|---|
| prompt characters: min / median / max | 21 / 90 / 1,472 | the same text | the same text |
| prompt tokens with the wrapper: min / median / 90th / max | 25 / 40 / 272 / 383 | 14 / 29 / 280 / 391 | no dump; about as Yi (the official prompt is 147 tokens on both before the wrapper, Ebony r4) |
| sum over the 32 prompts | 3,276 | 3,104 | about 3,300 (*estimated*) |
| reply length in the dumps (cap 128): min / median | 14 / 128 | 12 / 128 | – |
| longest prompt + 128 new tokens | 511 | **519** | about 510 to 520 |
| longest prompt + 32 new tokens | 415 | 423 | about 420 |
| the official prompt, with the wrapper | 167 | 186 | 153 by Ebony's hand wrapper |

- Tom's dumps measure the app-side token counts that Ebony's hand wrappers estimated: 167 and 186, exactly her
  figures.
- Most replies run to the cap; a few stop early (an end-of-text token), so a prompt does not always yield its
  full count.

### 1.3 What scoring on the set implies

| | official prompt (the claim) | prompt set, 128 tokens each | prompt set, 32 tokens each |
|---|---|---|---|
| n_ctx | 512 | 1,024 (519 positions needed) | 512 |
| KV buffer, Q8_0 | 17 / 136 / 128 MiB | 34 / 272 / 255 MiB allocated; at most half in use | 17 / 136 / 128 MiB |
| decode positions | 167–199 / 186–218 | median prompt: 40–168 / 29–157; longest: 383–511 / 391–519 | median: 40–72 / 29–61 |
| attention per token (r15, *estimated*) | 2.5 / 4.5 / 4.2 ms | about half of that at the median prompt, 2.5 × at the longest | about a quarter |
| decode seconds per prompt at 10 tok/s | 3.2 | 12.8 | 3.2 |
| clock state (Ebony r3, r4) | ends before the first cap step in the centre case | past the 15 s mark with the prefill: the sustained state, C about 35 | as the official prompt |
| gates per arm at one per prompt | 1 per run | 32, each 60 to 180 s after a 13 s run | 32, shorter |
| phone time per arm | – | 7 min of decode + 30 to 95 min of gates | 2 min + 15 to 50 min |

- **The set at 128 tokens measures a hotter, shallower decode than the claim.** A method can look better on it
  (cheaper attention, and its tokens per pass were tuned on this kind of text) or worse (capped clocks).
- **Residency:** n_ctx 1,024 doubles the cache allocation; in use it stays under 0.14 GB. No new problem.
- The R73 plan's nCtx 512 for the official block and 1,024 for the set means a reload between them (context
  size is a load-time setting in the app: `assistant.dart:460`).

### 1.4 What is not written, with a proposal for each

| open point | proposal | why |
|---|---|---|
| tokens per prompt | **128**, as the host numbers were made, stopping early at end-of-text | tokens per pass on the set (1.33 / 1.45) were measured at 128; 32 would be a different statistic |
| gating | one r16 gate before each prompt, as before each official run; report the gate series | the alternative (32 prompts back to back) is a sustained run, a different test. If phone time forbids 32 gates per arm, gate every prompt of a fixed 8-prompt subset instead and say so beforehand |
| the number | **committed tokens ÷ decode seconds, summed over the prompts of an arm**; also the median of the per-prompt rates | a per-prompt median hides the short replies; the sum is what a user waits |
| what it must show | not a second 12.0 line (the terms give the set no threshold). Proposal: **the method's arm ≥ the stock arm on the set, in the same session, ABBA**, and greedy text identical per prompt | the set is there so that a method tuned on the claim prompt cannot pass by fitting 31 guesses |
| identity | the method's reply text equals the stock arm's per prompt, both arms on the phone | the terms: "keeps greedy output identical" |
| subset and order | fixed before the run, in file order | – |

## 2. The claim prompt and the guess head

### 2.1 What the saved guesses say (*host*; `official.rows` and `<head>.pt.official.pred.npy`, 32 decoded tokens, 31 guesses)

| model, head | hits of 31 | top-1 | tokens per pass | the same head on the prompt set |
|---|---|---|---|---|
| Yi, `tok`, 2,117 prompts (its best) | 12 | 0.387 | **1.391** | 0.379; 1.326 |
| Yi, `tok`, 1,500 prompts | 13 | 0.419 | 1.391 | 0.368; 1.33 |
| Llama-2, `tok`, 3,000 prompts (its best on the set) | 6 | 0.194 | **1.143** | 0.497; 1.448 |
| Llama-2, `tok`, 1,500 prompts | 9 | 0.290 | 1.231 | 0.453; 1.41 |
| Llama-2, plain, 3,000 prompts | 8 | 0.258 | 1.231 | 0.476; 1.44 |
| Llama-2, prompt lookup first, then the 1,500 head (Tom §4.2) | – | 0.548 | 1.39 | 0.440; 1.41 |
| Llama-2, lookup first, then the 3,000 head (Tom §4.2) | – | 0.516 | 1.33 | 0.460; 1.43 |
| Yi, lookup first, then its head (Tom §4.2) | – | 0.323 | 1.28 | 0.403; 1.37 |

- These reproduce Tom's table (his "official top-1" and "official tokens/pass" columns). He warned that 31
  guesses are "not a basis for a verdict" on the head. That stands. **But the claim rule is the median of ten
  runs of exactly this prompt, greedy: its 31 guesses are the same in every run.** For the claim, the official
  prompt's tokens per pass is not a noisy sample; it is the number.
- Llama-2's official reply copies from the prompt's notes (Tom): a head trained on free chat guesses it badly,
  and a lookup in the prompt guesses it well. More training data made the head better on the set and worse here
  (0.290 → 0.194).

### 2.2 Price (*estimated*; rung 6 of r14: C5 out, campaign 2's two-stage head, head knob, O 5.2, K 300, cg 1.26)

| subject, guess source | tokens per pass | `linear` | `pair` | `pair` at K 0 |
|---|---|---|---|---|
| Yi, as the stacks have it (set) | 1.326 | 10.82 | 11.10 | 13.68 |
| **Yi, official, head** | 1.391 | **11.31** | **11.61** | 14.29 |
| Yi, official, lookup then head | 1.28 | 10.47 | 10.75 | 13.25 |
| Llama-2, as the stacks have it (set) | 1.448 | 10.80 | 11.12 | 13.45 |
| **Llama-2, official, best head alone** | 1.143 | **8.60** | **8.85** | 10.73 |
| Llama-2, official, 1,500 head alone | 1.231 | 9.24 | 9.51 | 11.52 |
| Llama-2, official, lookup then the 3,000 head | 1.33 | 9.95 | 10.25 | 12.40 |
| **Llama-2, official, lookup then the 1,500 head** | 1.39 | **10.38** | **10.70** | 12.94 |
| DeepSeek, assumed | 1.385 | 10.35 | 10.64 | 12.70 |
| width 1 on the same lane, for scale | 1 | 9.17 / 8.53 / 8.45 | the same | – |

- **Yi gains half a tok/s on the claim prompt; Llama-2 loses 0.4 to 2.3**, depending on the guess source.
- Lookup costs less per guess than a head evaluation (no 2.6 ms of head blocks when it has a candidate); the
  table does not credit that. It would add about 0.1 to 0.2 tok/s to the lookup rows.
- **A source chosen because it does well on this one prompt is exactly what the fixed prompt set exists to
  check.** The honest design: fix the rule (for example "lookup when it has a candidate, else the head") on the
  training split, then report both the official prompt and the set. Tom's per-step chooser (not tried) is the
  open host test.
- DeepSeek's official-prompt behaviour is unknown and is the binding subject in every stack.

## 3. What a load is made of (next question 2)

### 3.1 What the reports hold

One field: `reload_ms` (kit code `research.dart:1463`), the wall time of the app's `_loadOn` when a run needed a
different load. Campaign 1 and 2: 5.6 to 12.9 s on Granite, 2.6 to 11.0 s on OLMoE (*device*, r22). Beside it
`mem_before_load` and the resident megabytes before the load. No per-step time. `ttft_ms` is recorded per run
(`research.dart:1652`) and holds the prompt's prefill, not the load.

### 3.2 The steps, from source, with an estimate for a dense subject under choice B (*estimated*)

| step | where | what it costs | Yi / Llama-2 / DeepSeek |
|---|---|---|---|
| wait for a pending chat-state save; recall saved turns | `assistant.dart:524–525` | nothing in a kit run | 0 |
| drop the previous model and context | the library's `setModel` | unmap, free: fast unless the copies are large | under 0.5 s |
| open and map the file, read the header and the vocabulary | llama.cpp loader | 64,000 / 32,000 / 102,400 vocabulary entries into tables | 0.3 / 0.2 / 0.5 s |
| allocate the weight buffers and fill them: read 3.4 to 3.8 GB from storage (if not cached) and pack | `kleidiai.cpp` `set_tensor` | r9: pack 0.7 to 1.5 GB/s; the cold read is not measured | 3 to 20 s |
| re-quantise the head to int8 | `kleidiai.cpp:1529–1596` | a scalar float pass over 262 / 131 / 419 million weights | 1 to 4 s |
| create the context: KV buffer, three graph reservations | `llama-context.cpp:715–830` | planning only, no compute; the buffers are not touched yet | 0.1 to 0.3 s |
| clear the KV | at the first prompt (campaign 2 memo-24) | with `VOLLIE_KV_DONTNEED=1` a release, not a write | about 0 |
| chat template, sampler chain | the library | small | under 0.1 s |
| **sum** | | | **about 5 to 25 s**, against r22's 11 to 31 s by scaling campaign 2 |

- The two estimates overlap and neither is a measurement. Campaign 2's 7 to 9 s for models with a 0.24 to
  0.54 GB copy says there is a fixed part of several seconds that this list does not explain; the unload of the
  previous model and the file system are my candidates.
- **Fields that would split it in the next kit that loads a model** (three numbers, no new code in the runtime
  beyond reading llama.cpp's own timer): the library's model-load time (`t_load` from the context's performance
  counters), the wall time of `setModel` around it, and the wall time of the unload before it. With the
  `c3-pack` line of r21 the copy's share is then known.

## 4. Next questions, ranked

1. **Fold R75 into r18's draft** when its report is in the repository.
2. **The claim prompt's guesses, properly:** for each subject and each guess source that exists (head, lookup,
   lookup then head), the per-step hits on the official prompt, and what a rule fixed on the training split
   would have chosen. All from Tom's saved files; DeepSeek needs its dump first. Mine.
3. **What the terms should say about the set** (§1.4): a short text for the lead to put to the owner or into a
   pre-registration. Mine.
4. **Re-read r15 to r17 against Tom's host decode and KV screen** when they exist.
5. **DeepSeek's head; cold read rate; fixed share.**

## 5. Limits

- §1.2's token counts are for Tom's chat formatting on the laptop; the app applies the model's own template
  through its library and may differ by a few tokens. DeepSeek's are estimates.
- §2.1 is 31 guesses per head; that is the whole claim prompt, and it is also all there is.
- §2.2 changes one input (tokens per pass) in a stack whose other inputs are r14's; it inherits every guess
  listed there.
- "Lookup" numbers are Tom's table; I did not recompute them.
- §3.2 is a list from source with estimates; no load of a dense model has been timed on the phone.

## 6. Sources (2026-11-24)

Repository: `campaigns/03-dense15-5pct/CAMPAIGN.md` (metric row); `reports/memo-10-work/clyde-r73-plan.md` (§2, §6,
lines 144 to 172); `reports/memo-09-work/tom-head.md` (§3, §4.1, §4.2); `tools/c3_head_prompts.py`,
`tools/c3_head_dump.cpp` (file formats); `lab/c3/head/prompts/eval.txt`, `eval.idx`, `official.txt`;
`lab/c3/head/{yi,llama2}/eval.ptok`, `eval.rows`, `official.ptok`, `official.rows`, `head*.pt.official.pred.npy`;
`lab/app-c3/lib/assistant.dart` (lines 460, 518 to 546), `lib/research.dart` (lines 1463, 1652);
`lab/llama.cpp-c3r32/src/llama-context.cpp` (lines 715 to 830), `ggml/src/ggml-cpu/kleidiai/kleidiai.cpp` (lines 1529
to 1596); campaign 2 `reports/memo-24-anon-memory-and-rest.md`; my r9, r14, r15, r22; [ebony-joins-r3.md](ebony-joins-r3.md),
[ebony-joins-r4.md](ebony-joins-r4.md).
