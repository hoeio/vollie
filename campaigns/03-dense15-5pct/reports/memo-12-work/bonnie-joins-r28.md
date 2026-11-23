# Bonnie, memo-12 work, round 28: r27's drivers stored; three rows of the claim sheet (identity, prefix reuse, DeepSeek's BOS)

2026-11-24. Labels: *device*, *host* (laptop, relative only), *estimated*, *source* (code or file read). No device
claim. No web search (nothing is stated about upstream's current state). R75's report and `tom-kai-stock.md` are
still not in the repository; `ebony-joins-r14.md` (the owner question on DeepSeek's BOS) is not in the folder yet
and is referenced by name only.

## 0. Answer first

1. **Every number of r27 now reproduces from the repository:** `bonnie_r27_tables.py` (new, same folder) prints
   r27 §1.1's half-against-float figure, §1.4's spread table and §2.5's two tables, using
   `bonnie_chooser_r27.py`'s functions unchanged. Run today: all values equal r27's as printed (*host*).
2. **The claim sheet (r22 §1) gets one rewritten row and three new ones; it has 41 rows now, 8 of them open.**
   The rows are in §2 in the sheet's own format, ready to paste.
3. **The identity row was not testable as r22 wrote it.** "Greedy output identical" is now a comparison of two
   arms inside one session, on the same CPU module, cache type and prompt batching, by a printed hash of the
   reply's token ids. A host reply is not a reference for the phone's (Ebony r11: Yi's official reply changes
   with prompt batching alone).
4. **DeepSeek's first prompt token is BOS on the phone, by the file's own template,** although the file's
   `add_bos_token` is 0 (*source* and *host* header read, mine, independent of Ebony r13). Whether the owner's
   "BOS 설정은 허용하지 않습니다" covers a BOS that the unmodified file's template writes is the owner's to say;
   Ebony's r14 carries the question. This memo only records the row, what each answer changes, and that
   **nothing in the quality gate depends on it** (§2.3).

## 1. The stored drivers

`bonnie_r27_tables.py`, about 50 lines. Run: `lab/venv-c3/bin/python3 -I -B bonnie_r27_tables.py <repo root>`,
about 2 minutes. It reads only the trainer's held-out training prompts.

| output | r27 place | reproduces |
|---|---|---|
| (a) half against float value sum, 64 rows: median 1.7e-3 (1.0e-3 to 2.6e-3) | §1.1, §1.5 row 2 | yes |
| (b) hits per 31 steps, Llama-2: first windows n 144, mean 16.12, sd 5.06; all windows n 737, 14.62, 4.88; share ≤ 6: 0.021, 0.022 | §1.4 | yes |
| (c) P2 ÷ P1 from the stack model, 8 lines (1.094 to 1.253) | §2.5, second table | yes |
| (c) the sweep, 14 lines | §2.5, first table | yes |

Not stored, because they are one-line reads: the GGUF head counts of r27 §1.3 and the `CMakeCache.txt` settings of
§1.1. r27's sentence "they are not stored as files" is superseded by this section.

## 2. Claim sheet rows

Status letters as in r22: T = terms, P = R73 plan, K = kit as it stands, A = app default, O = open.

### 2.1 Row 6, rewritten (it was one row; it is 6 and 6a now)

| # | item | value | source | status |
|---|---|---|---|---|
| 6 | multi-token methods: what "greedy output identical" is compared with | the method arm's 32 committed token ids equal the stock arm's, **both arms run in the same session with (a) the same CPU module build, (b) the same cache type and flash-attention setting, (c) the same prompt batching** (the prompt as one batch, the same micro-batch). The kit prints, per run, the prompt's token count, a hash of the prompt's ids and a hash of the reply's ids; identity is equality of the reply hashes of every method run with every stock run of the block | `CAMPAIGN.md` (metric) for the term; Ebony r11 §1.2a for the three conditions; Ebony r13 E-T25 for the printed count and hash | T for the term; **the three conditions and the hash are a reading the lead has to adopt** |
| 6a | multi-token methods: what is counted and where it is scored | committed tokens; the fixed prompt set as well as the official prompt; the same guess source with the same parameters on both in one session; the kit prints passes, guesses made and accepted by source, passes without a guess | `CAMPAIGN.md` (metric); r24 §3, r26 (option B, with the owner); Ebony r11 §1.3 item 3, E-T22; r27 §2.4 | T for the count; O for how the set is scored (with the owner) |

Why each condition is there (all *host*, Ebony's measurements; the mechanism checked in *source* in r27 §1.1):

| condition | what happens without it |
|---|---|
| same build | two builds differ by 0.05 to 0.07 rms per logit after a few layers; a step decided by less flips (Llama-2's official reply has two steps under 0.13) |
| same cache type | F16 and Q8_0 caches take different attention arithmetic |
| same prompt batching | with an F16 cache a prompt of 64 or more tokens fed whole takes the tiled float path, fed in pieces the half path; Yi's reply changed at token 12 from this alone |

- With the phone's Q8_0 cache condition (c) is believed to be free (one attention path at every batch size,
  Ebony r8); it stays in the row because the cache type is itself open (row 25).
- **A consequence for the sheet's own numbers:** 167 and 186 prompt tokens and the host's tokens per pass (1.143,
  1.391; r23) are host figures of one path. They are expectations for the first kit, not references.
- What makes identity hold for a width-2 method is the node-exact kernel (E-T15). The hash is the phone's check
  that it did.

### 2.2 New row: prompt prefix reuse

| # | item | value | source | status | freeze |
|---|---|---|---|---|---|
| 39 | prompt prefix reuse | **off**: every run decodes the whole prompt again (`reusePrefix = false`, passed as `reusePromptPrefix`); the report's `saved` block records it as `reuse_prefix` | `lab/app-c3/lib/research.dart:102, 469, 1090` (read by me today) | K | – |

- It belongs with row 31 (the cache is cleared before every prompt): the two together mean each scored run
  starts with a full prefill of 167 / 186 tokens and its heat.
- A kit that turned it on would shorten the prefill to nothing on repeated runs and change the thermal state
  at decode start. It would be a protocol change and needs a pre-registration line, not a silent default.

### 2.3 New row: the first prompt token (BOS) per subject

| # | item | Yi | Llama-2 | DeepSeek | source | status | freeze |
|---|---|---|---|---|---|---|---|
| 40 | BOS at the start of the claim prompt | none | BOS (id 1), written by the template; the tokenizer does not add a second | **BOS (id 100000), written by the file's template** (`{{ bos_token }}` is its first output) | GGUF headers (*host*, read by me today); `llama_cpp_service.dart:6112–6113, 6324–6340` (special tokens are added only when the rendered prompt does not already start with BOS) | **O: owner** (Ebony r14) | raw text; no BOS on DeepSeek (`add_bos_token` 0) |

What I read myself (*host*, the three served files):

| file | `add_bos_token` | BOS id | template begins with |
|---|---|---|---|
| `yi15-6b-q4_0.gguf` | 0 | 1 | the system-message test, then `<|im_start|>…`; no `bos_token` |
| `llama2-7b-q4_0.gguf` | 1 | 1 | the system-message test, then `bos_token + '[INST] '` |
| `deepseek7b-q4_0.gguf` | 0 | 100000 | `{{ bos_token }}`, then `'User: ' + content + '\n\n'`, then `'Assistant:'` |

This agrees with Ebony r13 §2.2. The decision it touches is the owner's of 2026-11-22 (`CAMPAIGN.md` lines 115 to
122): the unmodified file is served, and "a runtime that inserts BOS on its own is not used either, unless the
owner allows it" (the lead's reading). Here the runtime inserts nothing on its own: it renders the template
stored in the unmodified file. I do not decide which reading holds. What each answer costs:

| | the template's BOS stays (the app as it is) | the claim prompt must have no BOS on DeepSeek |
|---|---|---|
| app or kit change | none | the kit strips the template's first token for this subject, or passes a template without it: a per-model special case, against "kits are model-generic" |
| file | unchanged | unchanged |
| quality gate (rows 7, 8) | **unchanged: perplexity is on raw text, 8.2231 at 40 chunks, no BOS** (*host*, `lab/c3/logs/ppl-deepseek7b-q4_0-b512-40.log`) | unchanged |
| speed metric | one more prompt token in the prefill; nothing in the 32 decoded tokens | – |
| what must match the phone | every DeepSeek dump (official prompt, set, head training) starts with id 100000; Tom's tool as it is writes none (Ebony's E-T23 change) | Tom's tool as it is |
| reply and tokens per pass | those of the prompt with BOS | those of a prompt the model's own template never produces |

- The BOS copy's 8.3095 (*host*) is history under the owner's decision and is not a budget; I cite neither it
  nor a reason to revisit the file.
- Until the owner answers, **no DeepSeek dump should be made** (either answer invalidates the other's dump),
  and the first kit that loads DeepSeek should print the prompt's first token id next to the count of row 6.

### 2.4 Changes to r22's other rows and lists

| r22 place | change |
|---|---|
| row 3 (prompt) | add: with the wrapper 167 (Yi) / 186 (Llama-2) tokens on the host dumps; DeepSeek not dumped. Expectations for the kit's printed count, not references |
| row 29 (chat template) | add: "see row 40 for the first token" |
| status counts | 41 rows: 12 T, 18 P or K, 3 A, 8 O (r22: 38 rows, 6 O). New open: row 6a's set scoring (with the owner), row 40 |
| §1.6 open decisions | insert as decision 2, after the load path: **"DeepSeek's first token (row 40): owner; blocks every DeepSeek dump and its head."** Insert as decision 3: "how the fixed set is scored (row 6a): owner; option B of r26 is proposed". The lead's adoption of row 6's three conditions is not a decision that blocks a round; it blocks the first width-2 pre-registration |

## 3. Lists (additions only)

| # | test | where | cost | closes when |
|---|---|---|---|---|
| B-T28a | the first model-loading kit prints, per run: prompt token count, first prompt token id, hash of the prompt ids, hash of the 32 reply ids (extends Ebony's E-T25) | device, free | 0 | counts differ from 167 / 186 → the host dumps are of another prompt; reply hashes differ between stock runs of one block → the stock arm is not deterministic and row 6 cannot be scored as written |

The second closing condition is new and worth a sentence: nothing so far shows that ten stock runs of one block
give one reply on the phone. Single decodes are bit-equal across 2 and 4 threads on the host (Ebony r11), so I
expect they do; P 0.9 (*estimated*). If they do not, identity has to be defined against the set of stock replies.

No try item.

## 4. Next questions, ranked

| # | question | state |
|---|---|---|
| 1 | P2 ÷ P1 from records that exist: R72 and R74 timed the two-column CPU kernel and the GPU at one and two columns; do they already bound the ratio that r27 §2.5 reads (gate at 1.20, width 2 closed near 1.33 / 1.42), before any width-2 kit? | *device* records and arithmetic; **the top item I can do now** |
| 2 | R75 into r18's draft, r25's script and r27 §2.1's costs | waits for the report |
| 3 | DeepSeek's lookup arms and first guess numbers | waits for the owner's answer on row 40, then Tom's dump |
| 4 | Do ten stock runs of a block give one reply on the phone (B-T28a)? Campaign 2's device JSONs may hold reply texts for OLMoE and Granite: a read of existing files | minutes, *device* records |
| 5 | Softmax time on the A78 against the free logit gap | only if question 1 says the gate pays |
| 6 | r15's O against Tom's host decode and KV screen | waits for Tom |

## 5. Sources (2026-11-24)

Repository: `bonnie-joins-r22.md` (§1), `bonnie-joins-r27.md`; `ebony-joins-r11.md` (§1.2a, §1.3, §1.4),
`ebony-joins-r13.md` (§2); `CAMPAIGN.md` lines 110 to 124; `lab/app-c3/lib/research.dart` (lines 100 to 102, 459 to
469, 1090); `lab/app-c3/third_party/llamadart/lib/src/backends/llama_cpp/llama_cpp_service.dart` (lines 6108 to
6113, 6324 to 6340, 6358); GGUF headers of the three served files (`tokenizer.chat_template`,
`tokenizer.ggml.add_bos_token`, `tokenizer.ggml.bos_token_id`); file names under `lab/c3/logs`. The value 8.2231
is quoted from my r9 and Ebony r13, not re-read from the log. No external source this round.
