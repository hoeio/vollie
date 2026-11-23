# c3 memo-12 work (Ebony, round 13): review of Bonnie r26, the claim run's chat wrapper against Tom's dumps, and two things the wrapper check turned up

Date: 2026-11-24. Author: Ebony (`claude-fable-5-2`). Not committed; the lead commits. No verdict here; the
terms are the owner's.
Follows [ebony-joins-r12.md](ebony-joins-r12.md). Reviews [bonnie-joins-r26.md](bonnie-joins-r26.md) (read in full).

Labels as before. Triples are Yi / Llama-2 / DeepSeek. New *host* numbers: arithmetic on Tom's existing dumps and
on the three GGUF headers. Nothing ran on a model.

## 0. Answer first

1. **Bonnie's quotations are word for word** (checked against `CAMPAIGN.md` and the R73 plan). One thing she
   says the terms leave open is in fact half written: the lead's reading of the owner's decision on guess heads
   already requires greedy identity on the fixed set. What is missing is only a speed consequence.
2. **I agree with option B and her paragraph, with one change and one addition.** Change: run the set at **32
   new tokens per prompt**, the claim's own regime, not 128. Addition: the same guess source with the same
   parameters on the set and on the official prompt, in one session. §1.2.
3. **A fact for the owner that her note does not state: on Llama-2 the official prompt is not a lucky text for a
   guess head, it is the unluckiest one we have.** Over the first 32 tokens of each set prompt, rule T gives 1.28
   to 1.78 tokens per pass (median 1.60), and the head alone is never under 1.185. The official prompt gives
   1.33 with rule T and 1.14 with the head alone. Keeping the claim on that prompt can fail a method that works on every other text. §1.3.
4. **The claim run has no unwrapped path.** The app's `_complete` sends the official prompt as one user chat
   message through the model's own template. Bonnie's sheet calls it "the plain completion call"; it is plain
   only in having no system prompt and no history. So "[INST] format against plain completion" is not a choice
   the claim offers, and Llama-2's copying is measured in the format the claim uses.
5. **Tom's dumps match that wrapper for Yi and Llama-2 as far as the source can say, and will not match it for
   DeepSeek.** The app renders the template with its own Jinja engine, and DeepSeek's template begins with
   `{{ bos_token }}`. Tom's tool uses llama.cpp's built-in formatter, which writes "User: … Assistant:" with no
   BOS, and the unmodified file has `add_bos_token` = 0. **So the phone's DeepSeek prompt starts with BOS and
   the tool's does not.** E-T23 must be changed before it runs. §2.2.
6. **A property of Yi's served file, found on the way:** its `<|im_start|>` is a normal token in the GGUF, not
   a control token, so llama.cpp spells the marker out as six text tokens. Every chat prompt Yi has seen in
   this project, on the host and (by the same tokenizer) on the phone, has its turn markers as text. §2.3.

## 1. Review of Bonnie r26

### 1.1 The quotations

| her quotation | source | result |
|---|---|---|
| metric row: "llamadart `decodeTokensPerSec`, official prompt, 32 tokens, greedy, kit protocol (r16 gate)" … "A method that drafts or commits several tokens per step counts committed tokens, keeps greedy output identical, and is scored on a fixed prompt set as well as the official prompt" | `campaigns/03-dense15-5pct/CAMPAIGN.md` line 51 | word for word |
| claim rule: "median of 10 plain runs ≥ 15.0 tok/s (*note 2026-11-21: read 12.0, per model, per Amendment 1*), positions fixed before the run, never a best run; power saving off, deep-sleeping list state recorded, charger out" … "campaign 2's go rule (memo-31 §4) applies before a claim round" | line 52 | word for word |
| Amendment 1: "Change the goal to 12 tok/s, run three different dense 7B models on the same chipset, and the goal is met when all three pass." | lines 12 to 13 | word for word; the owner's next sentence ("The subject models need to be reselected.") is left out, which changes nothing here |
| the lead's reading of the head decision: "The head only proposes; the model verifies. Greedy output must stay identical, checked on the fixed prompt set and the official prompt." | lines 69 to 70 | word for word |
| R73 plan §6: "used for every multi-token method, beside the official prompt" | `memo-10-work/clyde-r73-plan.md` line 151 | word for word |

Two texts she does not quote and that bear on it:
- `COMMON_RULES.md` rule 11: "A claim is 'the median of N plain runs of the official benchmark is at least X',
  fixed before the run, never a best run." So the claim's shape names "the official benchmark" in the shared
  rules as well as in the campaign's terms. Her option C or D would touch a common rule, not only this
  campaign's table.
- Campaign 2's metric row gives the reason for "unchanged": "so the benchmark's definition does not change"
  (`campaigns/02-25toks-5pct/CAMPAIGN.md` line 25). She cites the line; the reason is worth having in front of
  the owner.

**Her statement "what 'scored on a fixed prompt set as well' requires: no number, no comparison, no
consequence"** is right for speed. For identity it is not open: lines 69 to 70 already make greedy identity on
the set a requirement for a guess head. Her paragraph repeats it, which is harmless.

### 1.2 Would I propose a different set condition?

Her condition: the method's rate on the set (committed tokens ÷ decode seconds) is at least the stock rate, same
session, ABBA, identical text; rules chosen on training data only; both tokens-per-pass figures reported.

**I would keep it and change the set's run length.**

| | her proposal (r23 §1.4, r26) | mine |
|---|---|---|
| new tokens per set prompt | 128 | **32** |
| why | the host numbers were made at 128 | it is the claim's regime: a 3-second gated run at full clocks, not a 13-second run that ends capped (her r23 §1.3) |
| does the statistic change? (*host*, the first 32 tokens of each set prompt against all 128) | – | little: head 1.301 against 1.332 (Yi), 1.485 against 1.453 (Llama-2); rule T 1.379 against 1.391, 1.540 against 1.486 |
| phone time per arm | 7 min of decode, 30 to 95 min of gates | 2 min of decode, the same 32 gates; **an 8-prompt subset at 32 tokens is 8 gates** |
| what an 8-prompt subset can resolve | – | per-prompt tokens per pass spreads by about ±0.15 (below), so the mean of 8 is known to ±0.05; a method's gain over stock is 10 to 30 %, so "not slower than stock" is decided with room |

Additions to her paragraph:
1. "The same guess source, with the same parameters, is used on the set and on the official prompt in that
   session." Without it a lookup could be switched on for the claim only (r11 §1.3).
2. "Identity is between two arms of the same build, cache type and prompt batching" (r11 §1.2a: the official
   reply itself changes across those on Yi).
3. The stronger form she mentions and declines ("the gain on the set is at least half the gain on the official
   prompt"): I decline it too, for another reason. On Llama-2 the set's gain is the larger one, so the
   condition would be idle there, and on a subject whose official reply happens to copy it would punish an
   honest method.

### 1.3 What her note should also tell the owner

*Host*; tokens per pass over the first 32 tokens of each of the 32 set prompts (so each prompt is scored exactly
as the official one is), best heads, rule T at match length ≥ 2:

| | Yi | Llama-2 |
|---|---|---|
| set, per prompt, rule T: minimum / median / maximum | 1.07 / 1.36 / 1.88 | 1.28 / 1.60 / 1.78 |
| set, pooled: head alone / rule T | 1.301 / 1.379 | 1.485 / 1.540 |
| **official prompt: head alone / rule T** | 1.391 / 1.455 | **1.143 / 1.333** |
| where the official prompt sits among the 30 set prompts that reach 32 tokens | above the median (22 or 23 of 30 are at or below it) | **head alone: below all 30 (their lowest head-alone value is 1.185); rule T: only 2 of 30 are at or below it** |

- A single 32-token reply moves tokens per pass by ±0.3 around a model's mean. The claim reads one such reply.
- For Yi the official reply is an ordinary draw, on the good side. For Llama-2 it is the worst text we hold for its head: the reply
  copies the notes, the head was trained on free chat, and more training made it worse there (Tom).
- **So the risk runs both ways.** Her note stresses that a method could be fitted to the paragraph. The numbers
  say the opposite case is the live one: a method that is good on 32 of 32 other prompts can miss 12.0 on
  Llama-2 because of this paragraph. Under option B that miss stands. That is a consequence of keeping the
  benchmark's definition, and the owner should choose B knowing it.
- I do not propose C or D. I would add one line to any result: the method's rate on the set at 32 tokens per
  prompt, beside the claim, so that a miss or a pass on the official prompt can be read against the general
  case.

## 2. The claim run's wrapper and Tom's dumps (my question 3)

### 2.1 What the claim run sends (*source*)

- `lab/app-c3/lib/assistant.dart` lines 555 to 566: `_complete(prompt, maxTokens)` calls
  `_engine.create([LlamaChatMessage.fromText(role: user, text: prompt)], enableThinking: false,
  GenerationParams(maxTokens, temp: 0))`. One user message, no system message, the template from the model file
  (`chatTemplate: _template`, null for these subjects).
- `research.dart` line 16: run kind `official32` is this call with `_benchPrompt` and 32 tokens.
- llamadart renders the template with its own Jinja engine (`lib/src/core/template/`), with `bos_token` taken
  from the file's metadata (`chat_template_handler.dart` line 205), and does not add a second BOS when the
  rendered prompt already starts with it (`llama_cpp_service.dart` lines 6335 to 6340).
- So there is no raw-completion path in a claim run. The wrapper is each model's own template.

### 2.2 Against Tom's tool (`tools/c3_head_dump.cpp`: `llama_chat_apply_template`, then `llama_tokenize` with special tokens added and parsed)

| | the file's template (*host*, GGUF) | Tom's tool renders | the app renders | same tokens? |
|---|---|---|---|---|
| Llama-2 | `bos_token + '[INST] ' + content + ' [/INST]'`; `add_bos_token` 1 | built-in "llama2": `[INST] … [/INST]`, BOS added by the tokenizer | `<s>[INST] … [/INST]`, no second BOS | expected yes. The dump starts `<s>`, `▁[`, `INST`, `]` and ends `▁[`, `/`, `INST`, `]`; 186 tokens |
| Yi | `'<|im_start|>user\n' + content + '<|im_end|>\n<|im_start|>assistant\n'`; `add_bos_token` 0 | built-in ChatML, no BOS | the same string, no BOS | expected yes; 167 tokens |
| **DeepSeek** | `{{ bos_token }}` then `'User: ' + content + '\n\n'`, then `'Assistant:'`; **`add_bos_token` 0** (the unmodified file) | built-in "deepseek2" (matched by the template's `'Assistant: ' + message['content'] + eos_token`): `User: …\n\nAssistant:` with **no BOS**, and the tokenizer adds none | **BOS**, then the same text | **no: the phone's prompt has one token more, and it is BOS** |

- "Expected yes" is a reading of two renderers, not a measurement. No device report holds a prompt token count
  for these subjects. **The first kit that loads them should print the prompt's token count; 167 and 186 are
  the values to match.** A hash of the token ids would settle it completely.
- **DeepSeek.** BOS has already mattered for this model once: the freeze was prepared with and without it, the
  owner chose the unmodified file (2026-11-22), and the two differ in perplexity (8.2231 without, 8.3095 with,
  40 chunks at `-b 512`, *host*, `lab/c3/logs/ppl-deepseek7b-*-q4_0-b512-40.log`). A dump
  without BOS is a dump of a prompt the phone never sends. For E-T23 (r12 §2):
  - add a mode in which the tool takes prompts that are already rendered, and tokenizes them with special
    tokens parsed and BOS added only if the text does not start with it (the app's rule);
  - render the official prompt and the 32 set prompts from each file's own template with a Jinja renderer
    (`jinja2` 3.1.6 is in `lab/venv-c3`, with `bos_token` and `eos_token` from the GGUF);
  - check: for Llama-2 and Yi the rendered route gives the same token ids as the existing dumps' `.ptok` (186
    and 167 tokens). If it does not, the "expected yes" above is wrong and that is the finding;
  - for DeepSeek the first token must be id 100000.
- The same holds for the head task (r5 §2.3 step 2a): DeepSeek's training dumps should use the rendered route,
  or its head is trained on a prompt format the phone does not use.

### 2.3 Yi's turn marker (*host*; a property of the served file)

- In `yi15-6b-q4_0.gguf` the token types are: id 6 `<|im_start|>` **normal** (type 1); id 7 `<|im_end|>` control
  (type 3); id 8 `<|im_sep|>` normal.
- llama.cpp matches whole special tokens only among control and user-defined tokens. So `<|im_end|>` becomes one
  token and `<|im_start|>` is spelled out: the dump's prompt begins `▁<`, `|`, `im`, `_`, `start`, `|>`, `user`,
  `\n` and ends `<`, `|`, `im`, `_`, `start`, `|>`, `ass`, `istant`, `\n`.
- The official prompt on Yi is 167 tokens of which 12 are these spelled-out markers.
- *External*, one search, summaries only: the same has been reported for sibling models. On
  Yi-Coder-9B-Chat it was fixed upstream by adding `<|im_start|>` as a special token in `tokenizer_config.json`
  (<https://huggingface.co/01-ai/Yi-Coder-9B-Chat/discussions/4>); a discussion on Yi-1.5-34B-Chat notes that
  `<|im_end|>` is a special token and `<|im_start|>` is not
  (<https://huggingface.co/01-ai/Yi-1.5-34B-Chat/discussions/6>). I found no thread for Yi-1.5-6B-Chat itself and
  did not open the threads. Whether the publisher's own tokenizer also splits the marker, so that the model was
  trained this way, is not settled by what I read.
- What it means here:
  - The quality gate is untouched (perplexity on raw text has no markers).
  - The phone uses the same tokenizer on the same file, so the claim run sees the same 167 tokens as the dump.
    Dump and claim agree.
  - If the model was trained with single-token markers, every chat-format measurement on Yi (the reply, the
    head's top-1, tokens per pass) is of a model reading a slightly foreign prompt. The served file is frozen;
    a runtime that mapped the text to id 6 would change the prompt's tokens and the reply. That is a question
    for the lead and the owner, not a lever I propose.
- A check that costs minutes when the laptop is free: tokenize the rendered official prompt with the publisher's
  tokenizer files and compare with the dump's ids.

### 2.4 Does the wrapper drive Llama-2's copying?

It cannot be answered from files: it needs the official prompt decoded once without the wrapper (20 seconds on
an idle laptop). It no longer matters for the claim, because the claim uses the wrapper (§2.1). I leave it as a
low question.

## 3. Lists (additions and one change)

| # | test | where | cost | closes when |
|---|---|---|---|---|
| **E-T23, changed** | the DeepSeek dump takes rendered prompts (§2.2); its first prompt token is BOS; the rendered route reproduces Llama-2's and Yi's `.ptok` | host, Tom | + about 20 lines and a 15-line render script | Llama-2 or Yi ids differ from the dumps → the app's wrapper is not the dumps'; all existing chat-format numbers get a note |
| E-T25 | the first model-loading kit prints the official prompt's token count per subject (and a hash of the ids) | device, free | 0 | 167 / 186 / (DeepSeek's rendered count) matched → dumps and claim run agree on the prompt |
| E-T26 | Yi's rendered official prompt through the publisher's tokenizer against the dump's ids | host, minutes, needs the publisher's tokenizer files (a small download, the lead's call) | – | the publisher's tokenizer gives id 6 → the served file's chat prompts differ from the model's training format; a note for the owner |

No new try item.

## 4. Next questions, ranked

| # | question | state |
|---|---|---|
| 1 | Tom's KleidiAI stock result (flags, E-T13, E-T16) against the bands | waiting for his file |
| 2 | DeepSeek's dump with the rendered prompt (E-T23) | Tom's; the spec change is §2.2 |
| 3 | Is Yi's marker split by the publisher's tokenizer too (E-T26)? | a small download; the lead's call |
| 4 | How long is a set block on the phone at 32 tokens per prompt and 8 prompts, per subject? | arithmetic on R74's and R75's gate series; with Bonnie's r26 question 3 |
| 5 | The scoring-script columns (E-T21, E-T24): chooser against rule T on held-out rows | Tom |
| 6 | Llama-2's copying without the wrapper (§2.4) | low; 20 seconds on an idle laptop |

## 5. Sources (2026-11-24)

Repository: `campaigns/03-dense15-5pct/CAMPAIGN.md` (lines 10 to 15, 51 to 52, 64 to 72);
`campaigns/02-25toks-5pct/CAMPAIGN.md` (lines 23 to 26); `campaigns/COMMON_RULES.md` (rule 11);
`reports/memo-10-work/clyde-r73-plan.md` (line 151); `lab/app-c3/lib/assistant.dart` (lines 500 to 566, 630),
`research.dart` (lines 16, 1351, 1606); `lab/app-c3/third_party/llamadart/lib/src/core/engine/engine.dart`,
`core/template/chat_template_handler.dart` (line 205), `backends/llama_cpp/llama_cpp_service.dart` (lines 6335 to
6340); `lab/llama.cpp-base/src/llama-chat.cpp` (lines 160 to 187, 533 to 547); GGUF headers of the three served
files (templates, BOS flags, token types); `lab/c3/head/{yi,llama2}/official.ptok`, `eval.*`, the saved guesses.

External (search summaries only, 2026-11-24): the two Hugging Face discussions linked in §2.3.
