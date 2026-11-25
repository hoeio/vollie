# c3 memo-12 work: E-T23, the DeepSeek dump with prompt a

- **Date:** 2026-11-25. **Author:** tom, `claude-opus-5-5`. Not committed.
- **Spec:** [ebony-joins-r12.md](ebony-joins-r12.md) §2, changed by [ebony-joins-r13.md](ebony-joins-r13.md) §2.2 and
  [ebony-joins-r14.md](ebony-joins-r14.md) §1.5 / §4. The owner chose prompt **a** on 2026-11-24 (`CAMPAIGN.md`): the
  app's rendering, the text `<s>` and then the file's template, with no BOS token.
- **Labels.** Everything here is *host* (owner's Apple M4 Pro laptop, CPU only, 8 threads, one job at a time, load
  average 2 to 4 from other sessions). No phone run. The replies are the laptop's. By r11 §1.2a the phone's reply can
  differ wherever there is a near-tie.

## 1. Tool change (`tools/c3_head_dump.cpp`)

Three knobs. A fifth output file is always written.

| knob / file | what |
|---|---|
| `C3_HEAD_RENDERED=1` | each line is a prompt the app has already rendered. Special tokens are parsed. BOS is added only if the text does not start with the vocabulary's BOS text, and then only if the file's `add_bos` flag allows it. That is the app's rule (`llama_cpp_service.dart` 6358 to 6369, r14 §1.3) |
| `C3_HEAD_ONEBATCH=1` | the whole prompt goes in one decode, and its last row is written as step 0, as the app feeds it. Needs `n_par` 1 |
| `C3_HEAD_KV=q8_0` | K and V cache Q8_0, flash attention on (the app's setting). `f16` or unset leaves the default |
| `<out>.gap` | float32 per row: the top logit minus the second |

- **Self-check, knobs unset:** first 4 set prompts of Llama-2, `n_gen` 32, `n_par` 1, 8 threads. The old tool (the
  file before the change, built against the same libraries) and the new tool give byte-identical `.rows`, `.h16`,
  `.skip`, `.ptok` and `.topk` (`cmp`).
- **Limit of `ONEBATCH`:** the tool keeps `embeddings = true`, so every prompt row is an output and the head runs over
  156 rows instead of one. That can change the last row's logits at the rounding level. Ebony's script outputs only the
  last row, and on DeepSeek it gives the same 32 tokens as run 3 (§3), so this did not matter here.

**Render script:** `tom_et23_render.dart`. It uses the app's own engine (llamadart as vendored in `lab/app-c3`,
`ChatTemplateEngine.render`, one user message, `addAssistant: true`). The only metadata it passes is the GGUF's
`tokenizer.chat_template`. The files have no `tokenizer.ggml.bos_token` key, so the engine falls back to `<s>`, as on
the phone. Rendered files: `lab/c3/head/prompts/official-deepseek-a.txt` and `eval-deepseek-a.txt`.

**Check of the rendered route** (r13 §2.2): for Llama-2 and Yi, the rendered prompts through the new tool give `.ptok`
files byte-identical to the existing dumps' `official.ptok` and `eval.ptok`, all 33 prompts (186 and 167 tokens on the
official prompt). So the app's wrapper and the dumps agree on those two models.

**Builds:** R = `lab/llama.cpp-c3/build` libraries (the freeze flags; the same libraries as `lab/c3/head/cpu`). P =
`lab/llama.cpp-c3/build-kai` (the same flags plus KleidiAI), run with `GGML_KLEIDIAI_SME=0`. All three P logs print
`kleidiai: primary q4 kernel feature DOTPROD`, `SME disabled` and a `CPU_KLEIDIAI model buffer size` line
(DeepSeek 3,658.06 MiB), with `flash_attn = enabled` and K at q8_0. They also print `primary q8 kernel feature I8MM`.
The phone has no i8mm, but none of these files has Q8_0 weights, so that kernel is not on their path (reading, not
checked).

## 2. Runs (`lab/c3/models/deepseek7b-q4_0.gguf`, sha256 starts `f403ddc29802`, unmodified)

| # | build | cache | feeding | prompts | new tokens | output (`lab/c3/head/`) | wall |
|---|---|---|---|---|---|---|---|
| 1 | R | F16 (default) | as before | official, prompt a | 32 | `deepseek/official.*` | 3 s |
| 2 | R | F16 | as before | 32 set prompts, prompt a | 128 | `deepseek/eval.*` (2,765 rows, none skipped) | 76 s |
| 3 | P | Q8_0 | one batch | official, prompt a | 32 | `deepseek/official-p.*` | – |
| 4 | – | – | – | not run (optional) | – | – | – |
| 5 | P | Q8_0 | one batch | official, rendered, Llama-2 and Yi | 32 | `llama2/official-p.*`, `yi/official-p.*` | – |

Analysis: `tom_et23.py` (results 1 to 4). Results 5 to 7 come from `bonnie_head_paths.py` (one dictionary entry added
for `deepseek`) and `ebony_head_outliers.py` (unchanged; it already reads DeepSeek's dump once the dump exists).

## 3. The seven results of r12 §2.4 (all *host*)

1. **Prompt length: 156 tokens.** The first ids are 27, 82, 29 (`<`, `s`, `>`), then 5726 (`User`). That is prompt a,
   as r14 counted it. The kit's `prompt_tokens` should read 156 (E-T25).
2. **Official reply, both configurations:**
   - R (F16, fed as before): "The ocean is salty due to the continuous process of evaporation and precipitation. Rivers
     carry dissolved minerals from rocks into the sea, and as the water evaporates, the"
   - P (KleidiAI, Q8_0, one batch): "… precipitation. As rivers carry dissolved minerals from rocks into the sea, the
     minerals are eventually left behind as". This is **the same as Ebony's r14 reply for prompt a**, all 32 tokens
     (her run used the repack build with Q8_0 and one batch).
   - **Equal tokens: 14 of 32. First difference at token 14** ("Rivers" against "As"). It is a near-tie on both paths:
     gaps 0.217 (R) and 0.092 (P).
   - Smallest gaps: R 0.193 (token 31), 0.217 (14), 0.327 (24). P 0.033 (token 28), 0.092 (14), 0.260 (27).
   - Run 5, beside it: **Llama-2's P reply equals its dump on 32 of 32** (smallest gap 0.077, token 14), as Ebony's r11
     result predicted on a clean build. **Yi's differs from its dump at token 17** (18 of 32 equal; smallest gap 0.033,
     token 25). Yi was expected to differ.
3. **Lookup on the official reply** (Ebony's lookup: longest match 3, 2, 1; one guess per pass):
   - R step string: `-0 L1 L2 x3 -0 -0 x1 -0 -0 -0 x1 x1 -0 x1 -0 L1 L2 L3 L3 L3 L3 L3 L3 L3 x3 -0 x1 -0 -0 -0 x1`.
     P differs at one step.
   - **Tokens per pass, lookup alone, on both replies: 1.231 at any match length (26 passes), 1.185 at length 2 or
     more (27), 1.143 at length 3 only (28).** These equal r14's 1.231 and 1.185.
   - E-T23's closing line, "lookup alone under 1.10 → the claim-prompt price rests on the head alone", is **not
     triggered** (1.231). It is still well below the 1.385 every stack assumes.
4. **The set** (run 2: 32 prompts, 2,765 rows). Lookup precision by match length: 1 token 215 of 778 (0.276);
   2 tokens 147 of 324 (0.454); 3 tokens 370 of 555 (0.667). Tokens per pass, lookup alone: 1.203 at any length, 1.127
   at length 2 or more, 1.087 at length 3. Smallest gap 0.0008; 59 rows have a gap under 0.1.
5. **Hidden state range ÷ rms** (`bonnie_head_paths.py`): **median 24.9, p90 39.9, max 51.2.** Llama-2 is 10.9 and
   Yi 41.3, from the same script today.
   - Head-path proxy, mean −log p(greedy) of the KleidiAI path against ggml's: **DeepSeek +0.00171**; Llama-2 +0.00008;
     Yi +0.00536. DeepSeek's is almost all from the activations (+0.00162; weights +0.00010).
   - Activation error ÷ rms: ggml Q8_K 0.0148; KleidiAI per vector median 0.0283, p90 0.0451.
   - **E-T10 reading.** DeepSeek's measured stock shift is already on file: +0.69 % at `-b 512` and +0.67 % at `-ub 1`
     (*host*, 40 chunks, [tom-kai-stock.md](tom-kai-stock.md)). That is inside Ebony's band of +0.3 to +1.2 %.
     Yi's shift is +0.50 % against a proxy of +0.00536. Scaling the same way, DeepSeek's +0.00171 would account for
     about a third of its shift (*estimated*, a ratio of proxies, not a decomposition). So on DeepSeek most of the
     stock shift would come from the KleidiAI body, not from the head. That reading needs a run with the head alone
     off KleidiAI to confirm.
6. **Outlier dimensions** (`ebony_head_outliers.py`). `|output_norm.weight|` median 0.32, largest 0.6 (dim 1140). No dim
   is above 2 × median. Of the top 8 dims by data, 3 are in the gain's top 8 and 5 in its top 32. Float-column repair,
   change of −log p against ggml's path: k 0 +0.00171; k 2 +0.00088; k 4 +0.00089; k 8 +0.00077; k 16 +0.00083;
   k 32 +0.00083. **Two dims halve the head's proxy shift and more dims add nothing.** (The same run gives Yi
   +0.00536 at k 0 and +0.00170 at k 32; Llama-2 is near zero throughout.)
7. **Base check:** the dequantised Q6_K head (float) picks the dumped token on **0.9942** of the 2,765 rows. That is at
   or above 0.99, so it passes. Llama-2 is 0.9995 and Yi 0.9921 on the same script.

**Not done:** `ebony_head_int8.py` needs a trained head `.pt`, and DeepSeek has none, so its part (a) (KLD) did not run.
The DeepSeek head training dumps (r5 §2.3 step 2a) are still queued. Run 4 is optional and was not run.

## 4. Files

- `tools/c3_head_dump.cpp` (changed, three knobs plus `.gap`; knobs unset byte-identical).
- `campaigns/03-dense15-5pct/reports/memo-12-work/tom_et23_render.dart`, `tom_et23.py` (new);
  `bonnie_head_paths.py` (one dictionary entry added).
- Data, untracked: `lab/c3/head/deepseek/{official,official-p,eval}.*`, `lab/c3/head/llama2/official-p.*`,
  `lab/c3/head/yi/official-p.*`, `lab/c3/head/prompts/{official,eval}-deepseek-a.txt`.
