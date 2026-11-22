# memo-09 work file, Tom: trained guess head (T5), model-free guesses (T1), batch-2 identity (T1b)

2026-11-22, tom (`claude-opus-5-5`), on the owner's laptop (Apple M4 Pro, 24 GB). Not committed. Every number here
is *host* unless it is marked *estimated* (arithmetic) or quoted as *device* (R72). Nothing ran on the phone.
Owner decision 2026-11-22 (CAMPAIGN.md, "trained guess heads") allows lever M; this file is the first measurement.
Ebony's review (memo-09-gap.md §10) asked for the ridge probe first, chat-domain training, head cost as
head ÷ (1 + p), and T1 / T1b; all are here.

## 1. Verdict

- **T5 kill line (memo-09 §7.1: head top-1 < 0.50): missed on both models. Head top-1 on the fixed prompt set:
  Yi 0.379 (2,117 prompts), Llama-2 0.497 (3,000 prompts), best variant each, *host*.** Llama-2 misses by 0.003 and
  is still gaining 0.04 per doubling of data; Yi gains 0.02 per doubling and is far off. Medusa's 0.60 (a blog figure, §10.5) is not reached with
  the data a laptop session can make, and the learning curves say it would take far more (§4.3).
- **Ebony's ridge line (ridge top-1 < 0.35): missed too** (Yi 0.22, Llama-2 0.27). The trained block is far above
  the ridge map, so the ridge is a weak lower bound on these models.
- Read as written, both lines close M. What the head does give is still the best single guess source measured
  today: tokens per pass 1.33 (Yi) and 1.45 (Llama-2) on the prompt set, against 1.25 / 1.29 for the best
  model-free arm (T1 arm e). Priced on the phone (§5, *estimated*) the head is worth about −14 to −22 ms per token on
  Yi and −29 to −36 ms on Llama-2, if a batch-2 kernel with kB 1.08 to 1.15 exists. That is a lever of
  roughly 1 to 2 tok/s (Yi 6.90 → 7.7 to 8.2, Llama-2 6.21 → 7.7 to 8.2 printed), not the 12 tok/s stack Bonnie's M row assumed (p 0.60).
- **T1b: batch 2 is bit-identical to batch 1 on both stock CPU paths** (ggml repack q4_0_4x4 and KleidiAI v1.31.0
  dotprod), 128 of 128 rows on each model. Ebony's worry that KleidiAI's m = 2 gemm path changes the last bits did
  not show on the host.
- **T1** (model-free, one guess): on the prompt set the best arm (e, last-logit rescoring) gives p 0.30 (Yi) and
  0.33 (Llama-2). On the official 32-token run, prompt lookup gives 0.42 on Llama-2 (it copies the notes) and
  0.065 on Yi (it paraphrases). Yi's official run is below Bonnie's p < 0.20 line for every model-free arm (best: the
  counted table f, 0.19); see §6.

## 2. What was built (all untracked data under `lab/c3/head/`; scripts under `tools/`, not committed)

| file | what |
|---|---|
| `tools/c3_head_dump.cpp` | greedy chat decode against libllama; writes the post-norm hidden state (the vector the output head reads) per decoded row, the greedy token, the prompt tokens and the row's top-64 logits. Model's own chat template, BOS from the file. `C3_HEAD_NGL=99` runs Metal (training dumps only); default is CPU, as the phone |
| `tools/c3_head_build.sh` | copies a tree's libraries and builds a tool against them (so a rebuild of the tree cannot swap them mid-run) |
| `tools/c3_head_prompts.py` | prompt files: fixed split of databricks-dolly-15k (rule in the file) |
| `tools/c3_head_train.py` | ridge probe (closed form), Medusa-1 block training (PyTorch MPS), scoring on the CPU in fp32: top-1, shortlist top-1, simulated tokens per pass; `--tok` variant, `--rank`, `--max-prompts`, `--load` |
| `tools/c3_head_t1.py` | T1 arms a to f, t2, e; with `--head` also the head (h) and lookup-then-head (ah) |
| `tools/c3_head_b2check.cpp` | T1b: batch-1 against batch-2 logits, bitwise |
| `tools/c3_head_show.py` | prints a dump's greedy text |

Each script has a self-check (`selftest()` in the Python files). Libraries: `lab/llama.cpp-c3` (`26f78f925`, c2
series 0001 to 0009, CPU, repack on, KleidiAI off) for dumps; `lab/llama.cpp-base` (`620d2225c`) for T1b, built as
is (repack) and with KleidiAI v1.31.0 in `lab/c3/head/build-kai`. The base tree shows one deleted file in its
working copy (`ggml/src/ggml-cpu/vollie-release.h`), untouched by me.

Served files (sha256 checked by the training script and stored in each head's metadata):
Yi `yi15-6b-q4_0.gguf` `702cc2eb…a299`, Llama-2 `llama2-7b-q4_0.gguf` `8eb4952f…d4bf`. Both output heads are
Q6_K; the training script dequantizes them exactly (gguf-py) and checks itself against llama.cpp: argmax of the
dequantized head on the stored hidden state equals llama.cpp's greedy token in 0.9999 / 1.0000 of training rows and
0.9929 / 0.9994 of the CPU prompt-set rows (fp16-stored hidden states; near-ties).

## 3. Data and licences

- **Training text: the model's own greedy chat outputs** (self-distillation, as Ebony asked), on prompts from
  **databricks-dolly-15k**, licence CC BY-SA 3.0 (HF tag, read 2026-11-22), file at HF commit `bdd27f4d`, sha256
  `2df90833…94ec` (13.1 MB; the only download).
- Split (fixed, `tools/c3_head_prompts.py`): prompt set = record index ≡ 7 (mod 100), ≤ 2,500 characters, first 32
  in file order (`eval.txt` sha256 `ff1f32e3…72c0`); training = all other records, instruction text not equal to a
  prompt-set instruction, shuffled with `random.Random(0)`; `train1500.txt` = first 1,500 (`19c1b9fb…f203`),
  `train1500b.txt` = next 1,500 (`5c875edc…688fe`). Disjoint by construction (asserted).
- **The 32-prompt set is my provisional "fixed prompt set"**: CAMPAIGN.md asks for one and none existed. The lead
  should adopt it (or another) in a pre-registration before any device round scores a multi-token method.
- Official prompt: `tools/c2_official_prompt.txt` = the app's `_benchPrompt` (checked against
  `lab/app-c3/lib/assistant.dart`), one user message, no system prompt, the model's own template; 32 tokens.
- Model licences (HF tags): Yi-1.5-6B-Chat apache-2.0. Llama-2-7b-chat "llama2": the Llama 2 licence forbids using
  outputs to improve other LLMs *except Llama 2 or its derivatives*. The Llama-2 head is trained on Llama-2 outputs
  for Llama-2 only, which is my reading of the exception; the lead should confirm before a head ships.
- Training dumps ran on Metal, 16 sequences batched (greedy text may differ from single-stream CPU in places; it is
  still the model's own output). Prompt-set and official dumps ran on the CPU, one sequence, as the phone does.
- Sizes: Yi 1,500 prompts → 226,305 rows (1.85 GB of fp16 hidden states); 617 more prompts of the second file →
  89,073 rows (run stopped to free memory); Llama-2 1,500 → 237,638 rows. 192 new tokens per prompt at most.
- **A bug found and fixed before any reported number:** the first dump refilled a finished slot inside the output
  loop, which overwrote the step's outputs for later slots. Those dumps were deleted and redone.
- **A host trap worth recording:** full-vocabulary argmax computed on MPS in batched scoring gave wrong answers
  (top-1 0.11 for a head that scores 0.33 on the CPU). Training stays on MPS; all scoring runs on the CPU in fp32.
  The laptop's swap was 14 to 19 GB full during the session; CPU dumps run beside a Metal job slowed by 50×, so
  every CPU dump was run alone.

## 4. T5: the head

Head: g(h) = h + SiLU(W h + b), then the served file's own Q6_K output head, frozen; W zero-initialised (Medusa-1);
trained on (h_j, y_{j+1}) pairs: the hidden state of row j against the greedy token of row j+1, which is the guess
the lookahead pass needs (it feeds [y_j, g] and checks g against y_{j+1}). Cross-entropy, AdamW, lr 1e-3
one-cycle, batch 256, 2 epochs. Variant `tok` (EAGLE-like, not Medusa-1): the block also reads the input embedding
of y_j, the token row j itself produced, which is known when the guess is made: g = h + SiLU(W h + Wt E[y_j] + b).
That changes the head's shape, so it is a lead's call whether it fits the owner's decision; I report it beside
Medusa-1.

### 4.1 Results (*host*; "prompt set" = the 32 provisional prompts, CPU dumps, 128 tokens each)

| model | head | train pairs | held-out top-1 | **prompt-set top-1** | prompt-set tokens/pass | official top-1 (31 guesses) | official tokens/pass | train time |
|---|---|---|---|---|---|---|---|---|
| Yi | ridge (closed form) | 200,000 | 0.217 | – | – | 0.194 | 1.23 | 120 to 175 s |
| Yi | Medusa-1, full rank | 213,186 | 0.345 | 0.346 | 1.30 | 0.452 | 1.52 | 393 s |
| Yi | same, 2,117 prompts | 298,208 | 0.351 | 0.355 | 1.31 | 0.387 | 1.39 | 162 s |
| Yi | `tok` | 213,186 | 0.362 | 0.368 | 1.33 | 0.419 | 1.39 | 303 s |
| Yi | `tok`, 2,117 prompts | 298,208 | 0.377 | **0.379** | 1.33 | 0.387 | 1.39 | 189 s |
| Yi | rank 512 | 213,186 | 0.322 | 0.334 | 1.30 | 0.290 | 1.28 | 259 s |
| Yi | 1 epoch | 213,186 | 0.318 | 0.324 | 1.29 | 0.387 | 1.39 | 154 s |
| Yi | weight decay 0.5 | 213,186 | 0.346 | 0.343 | 1.30 | 0.452 | 1.39 | ~300 s |
| Llama-2 | ridge | 200,000 | 0.260 | 0.265 | 1.24 | 0.226 | 1.23 | 32 s |
| Llama-2 | Medusa-1, full rank | 224,031 | 0.420 | 0.434 | 1.39 | 0.226 | 1.23 | 71 s |
| Llama-2 | `tok` | 224,031 | 0.440 | **0.453** | 1.41 | 0.290 | 1.23 | 81 s |

| Llama-2 | Medusa-1, 3,000 prompts | 443,878 | 0.444 | 0.476 | 1.44 | 0.258 | 1.23 | 124 s |
| Llama-2 | `tok`, 3,000 prompts | 443,878 | 0.471 | **0.497** | 1.45 | 0.194 | 1.14 | 157 s |

- Held-out = 5 % of the training prompts, never trained on. The prompt-set and held-out values agree within 0.01 to
  0.02, so the prompt set is not easier or harder than Dolly in general.
- The official prompt gives 31 guesses per model; its top-1 moves by 0.03 per hit and is not a basis for a verdict.
- Train time is wall time on MPS with other jobs on the laptop (Yi 162 to 393 s for the same work); dumps took
  about 30 to 45 minutes per 1,500 prompts on Metal (118 to 135 rows per second when alone).
- Shortlist (argmax inside the 4,096 / 8,192 / 16,384 most frequent training targets): within ±0.01 of full
  vocabulary from 8,192 up (Yi `tok` 2,117: 0.373 / 0.378 at 8,192 / 16,384; Llama-2 `tok`: 0.449 / 0.452). The
  phone can use an 8,192-row shortlist of the served head at almost no loss.
- Overfitting is the limit: Yi full-rank training top-1 0.78 against 0.33 held-out (CPU check, 3,000 rows each).
  Weight decay, one epoch and rank 512 do not help; more data does, slowly (next section).

### 4.2 Combined with lookup (T1 arm ah: prompt lookup when it has a candidate, else the head; `tok` heads, 1,500 prompts)

| model | prompt set p / tokens per pass | official p / tokens per pass |
|---|---|---|
| Yi | 0.403 / 1.37 | 0.323 / 1.28 |
| Llama-2 | 0.440 / 1.41 | **0.548 / 1.39** |
| Llama-2, `tok` 3,000 prompts | 0.460 / 1.43 (head alone 0.497 / 1.45) | 0.516 / 1.33 (head alone 0.194 / 1.14) |

Lookup first helps Yi on the set and Llama-2 on the official prompt (which copies its notes); it hurts Yi's
official run. A rule that picks per step (for example the head's own confidence) was not tried.

### 4.3 Learning curve: how much data 0.50 would take (*host* points, *estimated* extrapolation)

Prompt-set top-1, full-rank Medusa-1 head, 2 epochs:

| model | 102 to 109 k pairs | 213 to 224 k | 298 k |
|---|---|---|---|
| Yi | 0.316 | 0.346 | 0.355 |
| Llama-2 | 0.394 | 0.434 | 444 k: 0.476 |

- Yi gains 0.03 for the first doubling and about 0.02 per doubling after. Reaching 0.50 from 0.355 takes 5 to 7
  more doublings: 10 to 40 M training tokens (*estimated*), against 15 k Dolly prompts (about 2.2 M tokens of answers
  at 150 tokens each) and about 120 rows per second on this laptop (23 to 90 hours per model at 8 KB a row,
  80 to 320 GB of hidden states). Not a laptop job, and the prompt source runs out first.
- Llama-2 gains a steady 0.04 per doubling (0.394, 0.434, 0.476). The `tok` head at 444 k pairs is at 0.497, 0.003
  under the line. One more doubling (about 6,000 prompts, 0.9 M pairs, about 2 hours of Metal dumps and 7 GB of
  hidden states on this laptop) would put it near 0.52 to 0.54 (*estimated*, if the slope holds). Dolly has enough
  prompts for that; it does not have enough for Yi.

## 5. Phone cost and price (*estimated*, from R72 *device* inputs)

Per lookahead pass the guess costs one block plus one pass over the output head (or a shortlist of it); per
committed token that is guess ÷ tokens-per-pass, as Ebony asked. Every committed token still pays its own head once.

    ms per token = (body × kB + O + guess) ÷ tau + head          printed = 950 ÷ ms (memo-09 §2)

Inputs: body 120.6 / 140.6 ms, head 9.04 / 4.40 ms (R72 L7, *device*), O 8 ms; head rate about 24 GB/s
(215 MB Q6_K in 9.04 ms), so a 4,096 × 4,096 block in Q8_0 (17.8 MB) costs about 0.74 ms and the `tok` variant two
of them; an 8,192-row shortlist of the head costs 9.04 × 8,192 ÷ 64,000 = 1.16 ms (Yi) and 1.13 ms (Llama-2).

| model | guess | guess ms | tau (prompt set) | kB 1.08 | kB 1.15 | stock |
|---|---|---|---|---|---|---|
| Yi | `tok`, shortlist 8,192 | 2.64 | 1.326 | 115.3 ms (−22.3) → 8.24 | 121.7 (−16.0) → 7.81 | 137.6 → 6.90 |
| Yi | Medusa-1, shortlist 8,192 | 1.90 | 1.305 | 116.4 (−21.2) → 8.16 | 122.9 (−14.7) → 7.73 | |
| Yi | `tok` + lookup (ah), full head | 10.52 | 1.369 | 117.7 (−19.9) → 8.07 | 123.9 (−13.8) → 7.67 | |
| Llama-2 | Medusa-1, shortlist 8,192 | 1.87 | 1.386 | 121.1 (−31.9) → 7.85 | 128.2 (−24.8) → 7.41 | 153.0 → 6.21 |
| Llama-2 | Medusa-1, full head | 5.14 | 1.391 | 123.0 (−30.0) → 7.72 | 130.1 (−22.9) → 7.30 | |
| Llama-2 | `tok` 3,000 prompts, shortlist 8,192 | 2.61 | 1.448 | 116.6 (−36.4) → 8.15 | 123.4 (−29.6) → 7.70 | |
| Llama-2 | Medusa-1 3,000 prompts, shortlist 8,192 | 1.87 | 1.432 | 117.3 (−35.7) → 8.10 | 124.2 (−28.8) → 7.65 | |

- **A smaller head is needed on Yi and not on Llama-2:** Yi's full-vocabulary guess (9.04 + 0.74 ms per pass) eats
  most of what the shortlist row keeps; on Llama-2 the full head costs 3 ms more per pass and gives almost the same.
  Low rank is not needed for cost (the block is under 1 ms) and rank 512 loses 0.01 to 0.02 of top-1.
- Batch 2 can commit at most 2 tokens per pass, so the lever alone cannot reach 12 tok/s (79.2 ms) on Yi: it
  would need tau 2.1. It is a stack element.
- **kB is the open half.** T1b shows the stock paths are exact at batch 2 but do not share the weight read (ggml
  repack runs the gemv once per position; Ebony §10.2), so stock kB is near 2 and the head loses on stock. Every
  price above assumes the batch-2 kernel of T2 that does not exist yet.
- Sidecar on the phone: the block in Q8_0 (17.8 MB, `tok` 35.7 MB), the shortlist as 8,192 row indices into the
  served head (32 KB), both sha256 next to the model's. The training script writes the model's sha256 and the
  training text into the head's metadata; a GGUF-style sidecar writer and the runtime reader are not written.

## 6. T1: model-free single guess (*host*; one guess per pass; b2 = share of passes with a candidate)

Arms (`tools/c3_head_t1.py`): a prompt lookup (3, 2, 1-gram), b prompt bigram, c self-successor (Token Recycling,
one node), d = a then c then b, f counted 3-gram (fallback 2-gram) table from the model's own greedy outputs on the
1,500 training prompts (Ebony asked for "the calibration text"; this is the chat text the head uses), t2 second-best
token of the last logits, e = candidates of a, b, c, f ranked by the last logits' top-64.

| arm | Yi set: coverage / acc / **p** / tau | Llama-2 set | Yi official | Llama-2 official |
|---|---|---|---|---|
| a lookup | 0.604 / 0.444 / **0.268** / 1.20 | 0.582 / 0.437 / **0.254** / 1.19 | 0.452 / 0.143 / 0.065 / 1.03 | 0.677 / 0.619 / **0.419** / 1.28 |
| b prompt bigram | 0.384 / 0.365 / 0.140 / 1.10 | 0.370 / 0.407 / 0.150 / 1.11 | 0.387 / 0.083 / 0.032 / 1.03 | 0.645 / 0.600 / 0.387 / 1.28 |
| c self-successor | 0.418 / 0.301 / 0.126 / 1.10 | 0.406 / 0.301 / 0.122 / 1.10 | 0.129 / 0 / 0 / 1.00 | 0.065 / 0 / 0 / 1.00 |
| d priority | = a | = a | = a | = a |
| f counted table | 0.954 / 0.243 / 0.232 / 1.21 | 0.977 / 0.285 / 0.279 / 1.26 | 0.968 / 0.200 / 0.194 / 1.23 | 1.000 / 0.129 / 0.129 / 1.10 |
| t2 last logit #2 | 1.000 / 0.065 / 0.065 / 1.06 | 1.000 / 0.061 / 0.061 / 1.06 | 0.032 / 1.03 | 0 / 1.00 |
| **e rescoring** | 0.699 / 0.429 / **0.300** / 1.25 | 0.695 / 0.479 / **0.333** / 1.29 | 0.516 / 0.125 / 0.065 / 1.03 | 0.710 / 0.591 / 0.419 / 1.28 |

- Against memo-09's line (p < 0.20 on the official prompt or the set, either model): **Yi's official run is under it
  for every arm (the best is f at 0.194).** Read as written, L with a model-free guess is closed on Yi. Llama-2
  passes on both. Against Ebony's joint line ((1 + p) ÷ kB < 1.05): Yi official fails at any kB ≥ 1.0 with a
  model-free guess.
- The head (§4) lifts Yi's official run to 0.39 to 0.45 (31 guesses), above both lines; that is the head's real
  contribution on Yi.
- Coverage and accuracy are printed apart as asked: lookup covers 0.58 to 0.68 of steps; with a confidence gate the
  uncovered steps run at batch 1 (b2 0.50 to 0.56 for arm a).

## 7. T1b: batch 2 against batch 1, bitwise (*host*)

`tools/c3_head_b2check.cpp`: per prompt, A = prompt then 32 greedy steps one token per decode; B = same prompt, then
the same 32 tokens two per decode, both positions output; each B row against A's row at the same position.
Prompts: the official prompt and the first three of the prompt set; 4 threads; `GGML_KLEIDIAI_SME=0`.

| path | Yi | Llama-2 |
|---|---|---|
| ggml repack (base `620d2225c`; chose `q4_0_4x4` for the body and `q6_K_8x4` for the head, the build has no i8mm) | 128 of 128 rows bit-identical, max |diff| 0 | 128 of 128 |
| KleidiAI v1.31.0 (log: "primary q4 kernel feature DOTPROD", 3,220 MB (Yi) and 3,599 MB (Llama-2) in the CPU_KLEIDIAI buffer) | 128 of 128 | 128 of 128 |

- So greedy identity at batch 2 holds on both stock paths on the host. On the phone the same kernels are chosen
  (dotprod, no i8mm on the A78), but this was not run there.
- Not tested: the new batch-2 kernel of T2 (it does not exist); KV-cache drop after a miss (the check feeds only
  true tokens, so it does not exercise the rollback).

## 8. Open, for the lead

1. M by the written lines is closed (head top-1 0.38 / 0.497 < 0.50; ridge < 0.35). Llama-2 sits at the line and
   one more data doubling (about 2 laptop hours) would likely cross it; that is a re-test the lead may order or not,
   not a re-score. The head is still the best guess
   source measured and turns Yi's official run from p 0.07 to about 0.4. Whether to keep it as part of lever L (a
   guess source, priced at −14 to −32 ms *estimated*, needing T2's kernel) is the lead's call; I do not re-score.
2. The `tok` variant reads the token embedding: is that inside the owner's "trained guess head" decision?
3. The provisional fixed prompt set (§3) needs adopting in a pre-registration.
4. Llama-2 licence reading (§3).
5. DeepSeek not run (two-model rule met by Yi and Llama-2; same scripts apply, head time 13.88 ms makes the
   shortlist necessary there).
