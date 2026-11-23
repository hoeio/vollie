# c3 memo-12 work (Ebony, round 15): what stops a reply on the phone, and campaign 2's two models through the same renderer check

Date: 2026-11-24. Author: Ebony (`claude-fable-5-2`). Not committed; the lead commits. No verdict here.
Follows [ebony-joins-r14.md](ebony-joins-r14.md).

Labels as before. New *host* results: the app library's renderer on five files' templates (`ebony_render.dart`,
extended: stop strings and a two-turn prompt); `llama-tokenize` counts on campaign 2's two files; counts read
from campaign 2's device reports (*device*). No model decode was run.

## 0. Answer first

1. **A reply stops on the phone when the model emits its end token id, and that part is sound on all five
   models.** The generation loop ends on `llama_vocab_is_eog(token)`, which reads the ids from the file. The
   library's text defaults play no part in it.
2. **The `eos_token` side of the library default exists and is the mirror of r14's:** when a template writes
   `eos_token` after an assistant turn, the library writes the literal `</s>`. That happens only when the prompt
   holds an earlier assistant turn. **The official benchmark is one user message, so no claim run of any
   campaign contains it.** DeepSeek and OLMoE would get `</s>` as stray text in a multi-turn chat.
3. **Campaign 2's OLMoE runs, the R70 claim among them, were made with a stray `<s>` at the start of the
   prompt, and the phone's own reports show it.** OLMoE's template begins with `bos_token`; its BOS text is not
   `<s>`; the rendered prompt tokenizes to **156** tokens with the stray text, 154 without, 155 with the real BOS
   token (*host*). Campaign 2's device reports say `prompt_tokens` = **156** on all 252 OLMoE runs I read
   (*device*). Granite's template uses no `bos_token`: 177 tokens rendered, 177 on the phone.
4. **That turns r14's reading into a measured fact for one model:** the app really sends what the library's
   renderer produces on the laptop. For DeepSeek the same mechanism predicts 156 tokens on the phone (its own
   count, by coincidence the same number).
5. **It changes no campaign-2 result.** The metric is decode tokens per second; the prompt was two tokens longer
   than a clean one. The quality gate runs on raw text. What it changes is a sentence: OLMoE's official reply in
   campaign 2 is the reply to a prompt with three stray characters, not to the publisher's format. Whether the
   closed campaign gets a dated note is the lead's call.

## 1. What stops a reply (my question 3)

### 1.1 The loop (*source*, llamadart 0.15.6 as vendored, `backends/llama_cpp/llama_cpp_service.dart`)

| stop | where | depends on the library's text defaults? |
|---|---|---|
| the sampled token is an end-of-generation token: `llama_vocab_is_eog(vocab, token)` | lines 6913, 7060, 7400 | **no**: llama.cpp answers from the file's token ids and types |
| the token limit (`maxTokens`; 32 in the benchmark) | the same loop | no |
| a stop string appears in the decoded text (`StopSequenceBuffer`) | lines 6892, 7005 | the strings come from the template handler (below) |

The stop strings a handler returns (`generic_handler.dart` lines 57, 98 to 115): `<end_of_turn>`, `<|im_end|>` or
`<|end|>` if the template contains one of them; otherwise the default list, which is `['<|im_end|>']`.

### 1.2 Per model (*host*: GGUF headers; the renderer's output)

| model | end token (id, text, type) | template writes `eos_token`? | stop strings from the handler | what ends a reply |
|---|---|---|---|---|
| Yi | 7, `<|im_end|>`, control | no (it writes the text `<|im_end|>` itself) | `<|im_end|>` | id 7. In Tom's set dumps the 12 replies that end before 128 tokens all end with id 7 (*host*) |
| Llama-2 | 2, `</s>`, control | yes | `<|im_end|>` (the default; never produced) | id 2 (9 early endings in the dumps, all id 2) |
| DeepSeek | 100001, `<｜end▁of▁sentence｜>`, control | yes | `<|im_end|>` (default) | id 100001 |
| Granite (campaign 2) | 100257, `<|end_of_text|>` | no | `<|im_end|>` (default) | id 100257 |
| OLMoE (campaign 2) | 50279, `|||IP_ADDRESS|||` | yes | `<|im_end|>` (default) | id 50279 |

- **So the answer to "what stops a DeepSeek / Yi / Llama-2 reply": the model's own end token id, checked by
  llama.cpp.** No stray text is involved. A 32-token run rarely reaches it; the set at 128 tokens does, and it
  will end where the dumps end, given the same reply.
- The default stop string `<|im_end|>` on models that never write it is harmless.
- One case to know for Yi: its prompt carries `<|im_start|>` as six text pieces (r14 §2). If the model answered
  in kind and spelled `<|im_end|>` out as text instead of emitting id 7, the stop string would still catch it,
  because the handler takes that string from Yi's template. In the dumps it emits id 7.

### 1.3 Where `eos_token` becomes stray text (*host*, the renderer on a two-turn prompt: user "Hi", assistant "Hello.", user "Bye")

| model | rendered |
|---|---|
| Yi | `<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\nHello.<|im_end|>\n<|im_start|>user\nBye<|im_end|>\n<|im_start|>assistant\n` |
| Llama-2 | `<s>[INST] Hi [/INST] Hello. </s><s>[INST] Bye [/INST]` |
| DeepSeek | `<s>User: Hi\n\nAssistant: Hello.</s>User: Bye\n\nAssistant:` |
| OLMoE | `<s><|user|>\nHi\n<|assistant|>\nHello.</s>\n<|user|>\nBye\n<|assistant|>\n` |
| Granite | role markers and `<|end_of_text|>` written by the template itself; a default system message; no `bos_token` or `eos_token` |

- Llama-2: `<s>` and `</s>` are its real tokens, so the defaults happen to be right.
- DeepSeek and OLMoE: neither `<s>` nor `</s>` is in the vocabulary (*host*), so both become text pieces. In a
  chat with history the model sees `</s>` as letters where it was trained to see its end token.
- **This is outside every benchmark run** (one user message, prefix reuse off). It is in the app's chat path, so
  it belongs with the owner's question of r14 §1.5, not beside it: one fix (give the template the vocabulary's
  real token texts, or empty strings) settles both tokens.

## 2. Campaign 2's models through the renderer (my question 4)

### 2.1 Host

`lab/granite/granite-Mi.gguf`, `lab/olmoe/olmoe-Mi.gguf`; the same renderer; the official prompt as one user
message; `llama-tokenize` of the campaign tree on each file.

| | Granite | OLMoE |
|---|---|---|
| BOS id and text; `add_bos_token` | 100257 `<|end_of_text|>`; 0 | 50279 `|||IP_ADDRESS|||`; 0 |
| template uses `bos_token` / `eos_token` | no / no | **yes / yes** |
| llamadart's handler | its Granite handler (format 11) | generic (format 0) |
| rendered prompt begins | `<|start_of_role|>system<|end_of_role|>You are a helpful assistant. …` | **`<s><|user|>\nRead these notes, …`** |
| tokens: as the app renders | **177** | **156** (first ids 29, 84, 2730: `<`, `s`, `><`) |
| tokens: the same without the stray text | – | 154 |
| tokens: with the real BOS token instead | – | 155 (first id 50279) |

### 2.2 Device

`prompt_tokens` in campaign 2's device reports (12 JSON files under
`campaigns/02-25toks-5pct/reports/device/`, the last twelve by file name; every run in them that has the field):

| model | run kind | `prompt_tokens` | runs |
|---|---|---|---|
| OLMoE | `official32` | **156** | 244 |
| OLMoE | `profiled32` | 156 | 8 |
| Granite | `official32` | **177** | 196 |
| Granite | `profiled32` | 177 | 8 |

- **OLMoE's 156 is the count with the stray `<s>`, not 154 and not 155.** The phone sent `<`, `s`, `><` … at the
  start of every OLMoE benchmark prompt.
- Granite's 177 equals the host's rendering, which has no stray text.
- Campaign 2's app (`lab/app-c2`) vendors the same llamadart 0.15.6 with the same default (*source*).
- I read 12 files, not every report of the campaign. The count is the same in all of them.

### 2.3 Does any campaign-2 device number rest on it?

| number | rests on the stray text? |
|---|---|
| decode tok/s, every OLMoE round including the R70 claim (27.21) | the prompt is 2 tokens longer than a clean one and the first three tokens are text. The metric times decoding. No effect on the rate that I can argue; none measured |
| stock against method comparisons (ABBA) | both arms had the same prompt |
| perplexity gates and budgets | no: raw text, no template |
| the reply's text, expert routing on the reply, anything scored per token of the reply | yes in principle: it is the reply to that prompt. Campaign 2 claimed speed, not text |
| Granite | no stray text at all |

- So: **no campaign-2 verdict changes.** A dated note in campaign 2's record ("OLMoE's benchmark prompt carried a
  literal `<s>` from the chat library's default; `prompt_tokens` 156") would be accurate; it is the lead's to
  write or not. I have not edited any campaign-2 file.
- For campaign 3 the device confirmation is the useful part: the renderer on the laptop reproduces what the
  phone sends, to the token.

## 2a. What DeepSeek's head would have to do on the claim prompt (next question 4 of §4, done at once; *estimated*)

r14 §1.4 gave DeepSeek's lookup steps on its official reply (prompt a). Under rule T the lookup takes the 11
steps with a match of two or more tokens (9 right) and a head takes the other 20. With the head right
independently at rate p on those steps (20,000 draws each):

| head top-1 on the non-lookup steps | 0 (lookup alone) | 0.20 | 0.30 | 0.38 | 0.44 | 0.50 |
|---|---|---|---|---|---|---|
| tokens per pass over the 32 tokens | 1.185 | 1.35 | 1.43 | 1.49 | 1.53 | 1.58 |

- **The 1.385 every stack assumes for DeepSeek needs a head that is right on about a quarter of the steps the
  lookup does not cover.** On their own official replies the trained heads got 5 of 19 such steps on Llama-2
  (0.26) and about 0.4 on Yi (*host*, from r12's step strings).
- So on this reply rule T with an ordinary head lands between 1.35 and 1.5; the assumption is not optimistic
  here, provided a head exists. P(DeepSeek reaches 1.385 on the claim prompt with rule T) about 0.55, my
  judgement.
- Limits: the reply is the laptop's, for prompt a; the other two prompts gave the same lookup count (r14);
  independence between steps is assumed; DeepSeek has no head.

## 3. Lists (additions only)

| # | test | where | cost | closes when |
|---|---|---|---|---|
| E-T28 | before any kit loads a subject: render its official prompt with `ebony_render.dart`, tokenize it on the served file, and write the count into the pre-registration; the report's `prompt_tokens` must equal it | host, then device | minutes; free on the phone | a different count → the phone's prompt is not the one the host work used |

No try item.

## 4. Next questions, ranked

| # | question | state |
|---|---|---|
| 1 | The owner's answers to r14 §1.5 (DeepSeek's prompt; now with the `eos_token` side and OLMoE's precedent) and §2.3 (Yi's marker) | with the lead |
| 2 | Tom's KleidiAI stock result (flags, E-T13, E-T16) against the bands | waiting for his file |
| 3 | Granite's default system message: the app's Granite handler adds "You are a helpful assistant. …" when none is given. Do the three subjects' templates add anything of the kind that the dumps lack? | checked for the official prompt in r14 (ids identical for Llama-2 and Yi); nothing further |
| 4 | What head top-1 would bring DeepSeek from the lookup's 1.19 to 1.23 tokens per pass to the 1.385 the stacks assume | **done, §2a:** about 0.25 on the steps the lookup does not cover |
| 5 | R75 free reads; the scoring-script columns (E-T21, E-T24) | R75; Tom |

## 5. Sources (2026-11-24)

Repository: `lab/app-c3/third_party/llamadart` 0.15.6: `backends/llama_cpp/llama_cpp_service.dart` (lines 6887 to
7060, 7400), `stop_sequence_buffer.dart`, `core/template/handlers/generic_handler.dart` (lines 57 to 115);
`lab/app-c2/third_party/llamadart/pubspec.yaml` and its `generic_handler.dart` line 81; GGUF headers of the five
files; `lab/c3/head/{yi,llama2}/eval.rows`; campaign 2's device JSON files (read only).
No external source this round.
