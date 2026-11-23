# c3 memo-12 work (Ebony, round 14): the BOS question for the owner, stated precisely; Yi's turn marker checked against the publisher's tokenizer; a correction to r13

Date: 2026-11-24. Author: Ebony (`claude-fable-5-2`). Not committed; the lead commits. No verdict and no
recommendation on the rule: the terms are the owner's.
Follows [ebony-joins-r13.md](ebony-joins-r13.md).

Labels as before. Triples are Yi / Llama-2 / DeepSeek.
New *host* results this round:
- the app library's own template renderer run on the three files' templates (`ebony_render.dart`, Dart on the
  laptop, no model; run from `lab/app-c3` with its package configuration; its input is a JSON of each file's
  header strings and the official prompt, made with a few lines of Python that I did not keep);
- token counts of the rendered prompts (`llama-tokenize` on the served files);
- three 32-token greedy decodes of DeepSeek's official prompt, two threads at low priority
  (`ebony_deepseek_prompt.py`);
- Yi's publisher tokenizer files (five small files, no weights, downloaded today into the untracked
  `lab/c3/dl/yi15-tokenizer/`) compared with Tom's dump.

Nothing ran on the phone.

## 0. Answer first

1. **Correction to r13 §0 item 5 and §2.2: the phone's DeepSeek prompt does not start with a BOS token. It
   starts with the three characters `<s>` as ordinary text.** I ran the app library's renderer instead of
   reading it. The file's template asks for `bos_token`; the library looks for the token's text in a metadata
   key the file does not have, and falls back to the literal string `<s>`. DeepSeek's vocabulary has no such
   token, so it becomes three text tokens (`<`, `s`, `>`), and no BOS token is added.
2. **So there are three candidate prompts for DeepSeek, and the claim run as built would use the one nobody
   chose:**

   | | first tokens | prompt tokens (*host*) | who produces it |
   |---|---|---|---|
   | a | `<`, `s`, `>`, `User`, `:` … | 156 | the app today (llamadart 0.15.6 as vendored) |
   | b | `User`, `:` … | 153 | llama.cpp's built-in formatter, which Tom's tool uses |
   | c | BOS (id 100000), `User`, `:` … | 154 | the file's template rendered as its publisher wrote it |

3. **The three give three different official replies** (*host*, 32 tokens each; §1.4). Stock speed is not
   affected. Everything that depends on the reply's text is: tokens per pass, the head's training format.
4. **For Llama-2 and Yi the app's prompt is exactly Tom's dump: identical token ids, 186 and 167** (*host*).
   r13's "expected yes" is now measured on the host.
5. **Yi's turn marker: the publisher's tokenizers make it one token; our served file does not.** All three
   publisher tokenizers (the SentencePiece model, the slow tokenizer, `tokenizer.json`) give id 6 for
   `<|im_start|>` and 156 tokens for the official chat prompt. llama.cpp on the served file gives 167 tokens and
   never id 6. The model was trained on a format our runtime does not feed it. §2.
6. **First figure of any kind for DeepSeek on the claim prompt** (*host*): a prompt lookup alone gives 1.23
   tokens per pass (1.19 with matches of two tokens or more), the same under all three prompts. Its reply
   contains one run of nine tokens copied from the notes. The stacks assume 1.385. §1.4.

## 1. The BOS question

### 1.1 What the owner decided, and the lead's reading (`CAMPAIGN.md` lines 115 to 122)

Owner, 2026-11-22: "BOS 설정은 허용하지 않습니다." Lead's reading: DeepSeek is served unmodified
(`deepseek7b-q4_0.gguf`, sha256 `f403ddc2…5118`), not the one-field BOS copy; "A runtime that inserts BOS on its own
is not used either, unless the owner allows it."

That decision was taken about the **perplexity runs** (whether the file's `add_bos_token` flag should be set so
that every chunk starts with BOS). The chat template was not in front of anyone then.

### 1.2 What the served file says (*host*, GGUF header of `deepseek7b-q4_0.gguf`)

| field | value |
|---|---|
| `tokenizer.ggml.bos_token_id` | 100000, text `<｜begin▁of▁sentence｜>`, a control token |
| `tokenizer.ggml.add_bos_token` | **0** (false): the tokenizer adds no BOS by itself |
| `tokenizer.chat_template` | begins `{{ bos_token }}`, then `'User: ' + content + '\n\n'` per user turn, then `'Assistant:'` |
| a key holding the BOS token's text (`tokenizer.ggml.bos_token`) | absent, as in every GGUF: the standard key is the id |

So the file gives two instructions that differ: "do not add BOS to text" (the flag) and "a chat prompt begins
with BOS" (the template). The publisher's model card says its tokenizer adds the BOS token before the input
(r5 §2.6).

### 1.3 What each renderer does with it

**The freeze and every perplexity run** (`llama-perplexity`, raw text, no template): BOS is governed by the flag,
so **no BOS**. The budget's reference was measured that way: 8.2231 on 40 chunks at `-b 512`; the BOS copy read
8.3095 (*host*, existing logs). Nothing in this section touches those numbers.

**The app** (*source* and *host*):
- `_complete` → `_engine.create([one user message])` → llamadart's `ChatTemplateEngine.render`.
- The template matches none of llamadart's named formats, so it goes to `GenericHandler`, which renders the
  file's Jinja template with `bos_token` set to `metadata['tokenizer.ggml.bos_token'] ?? '<s>'`
  (`generic_handler.dart` line 81). The metadata map is the file's key-value list as strings
  (`llama_cpp_service.dart` lines 7814 to 7836); it has no such key.
- **Run on the laptop** (`ebony_render.dart`, the package as vendored in `lab/app-c3`, metadata taken from the
  three GGUF headers, the official prompt as one user message):

  | file | rendered prompt begins | ends |
  |---|---|---|
  | Yi | `<|im_start|>user\nRead these notes, …` | `…behind.<|im_end|>\n<|im_start|>assistant\n` |
  | Llama-2 | `<s>[INST] Read these notes, …` | `…the salt behind. [/INST]` |
  | DeepSeek | **`<s>User: Read these notes, …`** | `…the salt behind.\n\nAssistant:` |

- Tokenizing (`llama_cpp_service.dart` lines 6358 to 6369): special tokens are parsed; BOS is added by the
  tokenizer only if the prompt does not already start with the vocabulary's BOS text, and then only if the
  file's flag allows it.
  - Llama-2: the text `<s>` **is** its BOS token; it is parsed as id 1 and no second one is added. Correct by
    coincidence of the default.
  - Yi: its template uses no `bos_token`; flag 0; no BOS.
  - DeepSeek: `<s>` is not its BOS text and not a token at all (*host*: no `<s>` in its 102,400-entry
    vocabulary). It is tokenized as `<` (27), `s` (82), `>` (29). The flag is 0, so no BOS token follows.
- Limit: this is the library's renderer run on the laptop with the file's metadata. The phone's own count is in
  every kit report (`prompt_tokens`, `research.dart` line 1650). **153, 154 or 156 on DeepSeek's official run
  says which prompt the phone sent.**

**Tom's dump tool** (`llama_chat_apply_template`): the template is recognised as llama.cpp's built-in
"deepseek2" format, which writes `User: …\n\nAssistant:` and no BOS. Prompt b.

**llama.cpp's own Jinja route** (the server's, not used by anything of ours) passes the vocabulary's real BOS
text to the template. Prompt c. (*Source* knowledge of upstream's behaviour, not run today.)

### 1.4 Does it matter? (*host*; `ebony_deepseek_prompt.py`; the freeze-flag repack build, Q8_0 cache, the prompt as one batch, 32 greedy tokens)

| prompt | reply (32 tokens) |
|---|---|
| a, literal `<s>` | "The ocean is salty due to the continuous process of evaporation and precipitation. As rivers carry dissolved minerals from rocks into the sea, the minerals are eventually left behind as" |
| b, no BOS | "The ocean is salty because of the continuous process of evaporation and precipitation. As rivers carry dissolved minerals from rocks into the sea, the minerals are left behind when the" |
| c, BOS token | "The ocean is salty due to the continuous process of evaporation and precipitation. Rivers carry dissolved minerals from rocks into the sea, and as the water evaporates, it" |

- Tokens equal between replies: a and b 26 of 32 (first difference at token 4); a and c 14 (token 14); b and c
  12 (token 4). Prompt b has a step decided by 0.003 of a logit (token 7).
- All three are sensible. None is "the model's reply"; each is the reply to its prompt.
- A prompt lookup on each: right at 11 steps, among them nine in a row ("rivers carry dissolved minerals
  from rocks into the sea"). Passes for 32 tokens: **26 with any match length (1.231 tokens per pass), 27 with
  matches of two or more (1.185)**, the same for a, b and c. 13 of the 32 reply tokens lie in a run copied from
  the prompt.
- This is the laptop's reply on the repack path. By r11 §1.2a the phone's may differ at a near-tie.

What does not depend on the choice: perplexity, the budget, stock tok/s (the metric times decoding; the prompt
is 3 tokens longer or shorter).
What does: the reply's text, hence tokens per pass of any multi-token method; the prompts a guess head is
trained on; and whether Tom's dumps describe the prompt the phone sends.

### 1.5 The question, and the options

**Question for the owner:** in a chat-format run of DeepSeek (the official benchmark is one), what are the first
tokens of the prompt?

| option | prompt | what has to change | relation to the 2026-11-22 decision | what else it brings |
|---|---|---|---|---|
| A | as the app renders today: the text `<s>`, no BOS token (156 tokens) | nothing | no BOS token is inserted. But the prompt carries three stray text tokens that neither the file nor the publisher asks for | Tom's tool must be made to produce the same prompt, or its dumps describe another one |
| B | no BOS and no stray text (153 tokens) | one line in the app's use of the library (or a patch to its default): render `bos_token` as empty when the file gives no text for it | the literal reading of "BOS is not allowed"; the same convention as the perplexity runs | equals Tom's tool as it is; the file's template is not followed in its first element |
| C | the BOS token, as the file's template is written (154 tokens) | the app passes the vocabulary's BOS text to the template; Tom's tool renders the template instead of using the built-in formatter | the runtime would put a BOS token in the prompt. It would do so because the served file's own template says so, not "on its own"; whether that is inside the decision's words is exactly the owner's call | the publisher's format; the format its model card describes |

What I can ground, and nothing more:
- A is not something anyone decided; it is a library default (`'<s>'`) meeting a file whose BOS text is
  different. Whichever option is chosen, the pre-registration of the first model-loading round should say which
  prompt is used, and the kit's `prompt_tokens` should be checked against it.
- B and C are both changes in the app or its library, not in the served file.
- The decision changes no perplexity figure and no stock speed.
- Host work that depends on it and is queued: the DeepSeek dump (E-T23) and DeepSeek's head. Both should wait
  for the answer or be run for the chosen prompt only; a head trained on b is trained on a prompt the app does
  not send today.
- The same library default touches any model whose template uses `bos_token` and whose BOS text is not `<s>`.
  Of the three subjects that is DeepSeek only. Campaign 2's models were not checked.

## 2. Yi's turn marker (E-T26)

### 2.1 What was downloaded

From `https://huggingface.co/01-ai/Yi-1.5-6B-Chat/resolve/main/`, 2026-11-24, into `lab/c3/dl/yi15-tokenizer/`
(untracked; `lab/` is ignored by git):

| file | bytes | sha256 (first 16 hex) |
|---|---|---|
| `tokenizer_config.json` | 1,673 | `a877a66153e25d07` |
| `tokenizer.json` | 3,602,758 | `a13ccc285aea27f5` |
| `special_tokens_map.json` | 567 | `27ce6fefbeb38d97` |
| `tokenizer.model` | 1,033,105 | `386c49cf943d71aa` |

(`added_tokens.json` does not exist there.) No weights. `tokenizer.model` is the SentencePiece vocabulary; I
took it because the config names the slow tokenizer class, which reads that file.

### 2.2 Result (*external* files, *host* computation; the official prompt in Yi's chat template)

| tokenizer | tokens | `<|im_start|>` as id 6 | first tokens |
|---|---|---|---|
| publisher, SentencePiece model alone | 156 | 2 times | `<|im_start|>`, `user`, `\n`, `Read`, … |
| publisher, slow tokenizer (`LlamaTokenizer`, the class its config names) | 156 | 2 | the same |
| publisher, `tokenizer.json` | 156 | 2 | the same |
| **llama.cpp on our served file (Tom's dump; equal to the app's prompt, §0 item 4)** | **167** | **0** | `▁<`, `|`, `im`, `_`, `start`, `|>`, `user`, `\n`, `Read`, … |

- `<|im_end|>` is id 7 once in all four.
- **Why** (*host*, the files): in the publisher's `tokenizer_config.json` the list of added tokens holds ids 0,
  1, 2 and 7 only. `tokenizer.json` lists id 6 as an added special token, and the SentencePiece model emits it
  as one piece. The GGUF's token-type table has id 7 as a control token and id 6 as a normal one (r13 §2.3).
  llama.cpp matches whole special tokens only among control and user-defined entries, so it spells id 6 out.
  The conversion appears to have taken the token types from the config's short list. I did not re-run the
  conversion.
- **So r13's open point is settled:** the publisher's own tokenizers do not split the marker. The model was
  trained with one token where our runtime feeds six, at the start of the user turn and of the assistant turn.

### 2.3 What it touches and what it does not

- Not the quality gate (raw text), not the budget, not stock tok/s.
- Every chat-format number on Yi: the official reply, the 32 set prompts, the trained head's top-1 and tokens
  per pass. They are self-consistent (the head was trained on the same prompts the phone will send), but they
  describe the model reading a prompt it was not trained on.
- The official prompt is 11 tokens longer than it would be (167 against 156).
- It is the same kind of question as §1: the served file is frozen and its metadata is what it is; a runtime
  could map the marker's text to id 6 without touching the file; whether it should is the owner's call. If it
  did, Yi's dumps and head would have to be redone on the new prompts.
- A host test that would show the size (20 seconds of CPU, not run: it needs a decision on whether the answer
  matters): the official prompt decoded from the publisher's 156 ids against the dump's 167.

## 3. Corrections to r13

| r13 said | now |
|---|---|
| §0 item 5, §2.2: "the phone's DeepSeek prompt starts with BOS and the tool's does not" | the phone's starts with the text `<s>` (three tokens) and has no BOS token; the tool's has neither. Three prompts exist (§0 item 2) |
| §2.2: Llama-2 and Yi "expected yes" | identical ids on the host: the app library's rendered prompt, tokenized by llama.cpp on the served file, equals Tom's `.ptok` (186 and 167 tokens) |
| §2.3: whether the publisher's tokenizer also splits Yi's marker "is not settled" | settled: it does not (§2.2) |
| E-T23's change: "its first prompt token is BOS" | E-T23 waits for the owner's answer to §1.5, then uses that prompt |

## 4. Lists (changes and additions)

| # | test | where | cost | closes when |
|---|---|---|---|---|
| E-T23, changed again | the DeepSeek dump uses the prompt the owner chooses in §1.5 (a, b or c) | host, Tom | as before | – |
| E-T25, sharpened | the first kit that loads DeepSeek: `prompt_tokens` of the official run | device, free (the field exists) | 0 | 156 → prompt a; 153 → b; 154 → c |
| E-T27 | Yi's official prompt decoded from the publisher's 156 ids against the served file's 167 | host, CPU, 20 seconds | – | the two replies and the lookup steps side by side; only if the lead wants the size before asking the owner |

No try item: both §1 and §2 are decisions, not levers.

## 5. Next questions, ranked

| # | question | state |
|---|---|---|
| 1 | The owner's answer to §1.5 (DeepSeek's prompt) and, if the lead puts it, to §2.3 (Yi's marker) | with the lead |
| 2 | Tom's KleidiAI stock result (flags, E-T13, E-T16) against the bands | waiting for his file |
| 3 | Does the same library default (`'<s>'`) or another silent substitution touch the `eos_token` side: what stops a DeepSeek or Yi reply on the phone? | *source* and the renderer, minutes; a 32-token run never reaches it, the prompt set at 128 tokens does |
| 4 | Campaign 2's two models through the same renderer check | host, minutes; only to know whether the closed campaign's prompts were what was assumed |
| 5 | DeepSeek's lookup on the claim prompt is 1.19 to 1.23; with rule T it needs a head. What head top-1 would bring it to the 1.385 the stacks assume? | arithmetic on §1.4's step string |
| 6 | R75 free reads; the scoring-script columns (E-T21, E-T24) | R75; Tom |

## 6. Sources (2026-11-24)

Repository: `campaigns/03-dense15-5pct/CAMPAIGN.md` (lines 115 to 122);
`lab/app-c3/third_party/llamadart` 0.15.6: `lib/src/core/template/chat_template_engine.dart` (lines 186 to 232, 852
to 863), `handlers/generic_handler.dart` (lines 60 to 84), `chat_format.dart` (`detectChatFormat`),
`backends/llama_cpp/llama_cpp_service.dart` (lines 6318 to 6369, 7814 to 7836); `lab/app-c3/lib/assistant.dart`
(lines 555 to 566), `research.dart` (line 1650); `lab/llama.cpp-base/src/llama-chat.cpp` (lines 183 to 184, 533 to
547); GGUF headers and vocabularies of the three served files; `lab/c3/head/{yi,llama2}/official.ptok`;
`lab/c3/logs/ppl-deepseek7b-*-q4_0-b512-40.log`.

External: the four tokenizer files of `01-ai/Yi-1.5-6B-Chat` listed in §2.1 (downloaded and read today).
