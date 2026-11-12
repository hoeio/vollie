# Candidate models

Status: **no model is chosen yet** (open decision 1 in [CAMPAIGN.md](CAMPAIGN.md)). Researched 2026-11-12 from the
sources listed at the end. Total/active parameter counts are the publishers'; fit and speed statements are
*estimated* (see [feasibility.md](feasibility.md)). "Well known" is a judgement about the maker, not a measure.

Selection logic: a sparse model with total parameters of about 7 to 9 B (it must fit an 8 GB phone with about
2 GB of page cache) and about 1 to 1.5 B active, supported by llama.cpp, with the largest name recognition
available in that class.

## Tier A: 7 to 9 B sparse models

| model | maker | total / active | structure | notes |
|---|---|---|---|---|
| Granite-4.0-H-Tiny | IBM | 7B / about 1B | 36 Mamba-2 + 4 attention layers, 64 experts | campaign-1 model; all patches apply; 25 tok/s unlikely under the budget |
| OLMoE-1B-7B | AI2 | 6.9B / 1.3B | 16 layers, 8 of 64 experts per layer | released 2024-09; fewest ops per token; small experts may quantise worse |
| LFM2.5-8B-A1B | Liquid AI | 8.3B / 1.5B | 24 layers (18 gated conv + 6 GQA), 128K vocabulary, 128K context | released 2026-05-28; llama.cpp GGUF from launch; the publisher's "about 30 tok/s on a phone" names no device |

## Tier B: higher risk, or not 7B

| model | maker | total / active | notes |
|---|---|---|---|
| Ling-mini-2.0 / Ring-mini-2.0 | inclusionAI (Ant Group), MIT | 16.3B / 1.4B (789M non-embedding), 20 layers, 8 of 256 experts | smallest compute per token, but about 6 to 7 GB at 3 bits on an 8 GB phone: the page-cache problem of campaign 1 (R9) gets worse. llama.cpp support must be checked per variant: a Ring-mini GGUF for llama.cpp exists (architecture `bailingmoe2`), a Ling-mini GGUF found was for `ik_llama.cpp` only |
| SmallThinker-4B-A0.6B | SJTU / PowerInfer | 4B / 0.6B, 32 layers, 4 of 32 experts | reported above 20 tok/s on ordinary CPUs at Q4_0 with about 1 GB; not 7B and little known; useful as a ceiling experiment |

## Tier C: well known, excluded (with the reason)

| model | reason |
|---|---|
| Llama 3.1 8B, Mistral 7B, Qwen3-8B, Qwen3.5-9B, Granite 4.1 8B | dense: bandwidth and compute, see [feasibility.md](feasibility.md) §1 |
| Gemma 4 E2B / E4B | dense with per-layer embeddings: 2.3B effective (5.1B with embeddings) and 4.5B effective (8B); about 1.3 GB per token at 4 bits puts the ceiling near 14 to 19 tok/s here (*estimated*). Kept as an optional "famous dense" control row |
| Gemma 4 26B-A4B | 25.2B total, 3.8B active: does not fit |
| gpt-oss-20b | 21B total, 3.6B active: does not fit |
| Qwen3.5-35B-A3B, Qwen3-30B-A3B | 3B active but 30B+ total: does not fit; Qwen3.5's small models are all dense |
| Qwen1.5-MoE-A2.7B | 2.7B active: more than twice Granite's bytes per token |

There is no Qwen model in the 7 to 9 B sparse class. Not verified here: ZAYA1-8B (named in a comparison page
title only) and MobileMoE (a 2026 paper, 0.3 to 0.9B active; no public weights were checked).

## Campaign 1 history (Phase 0, from [../../reports/00_model_choice.md](../../reports/00_model_choice.md))

Read this before proposing a model: part of the field was already tested.

- **Ling-3.0-tiny** (7.9B / 1.3B active, MIT, KDA+MLA hybrid; i1-Q4_0, abliterated variant): measured on the host and
  **rejected on quality**: bpb 0.906 on wikitext-2 against 0.750 for Granite (*host*). Its Q4_0 quality is the
  problem, so it needs a better quantization before it is a candidate.
- Rejected without download: dense 3B Granite 4.1/4.2 (about 2 GB per token), Gemma 4 E2B (about 1.2 to 1.3 GB per
  token, 262k vocabulary head), a 1.3B-A0.6B MoE. Rejected in earlier sessions (reasons are in private notes, not in
  the repository): LFM2.5-8B-A1B, SmallThinker, Qwen3.5-2B/0.8B, Trinity Nano. Re-test LFM2.5 only with new evidence.
- Every served file in campaign 1 was an **abliterated** variant. Whether campaign 2 uses a stock or an abliterated
  file is an owner decision, and it interacts with "a model people know".
- The campaign 1 comparison used bits per byte (comparable across tokenizers); use it again when models differ.

## Sources

- OLMoE: <https://arxiv.org/html/2409.02060v2>; GGUF pruned variant card <https://huggingface.co/Local-Axiom-AI/llama-1B-5B>
- LFM2.5-8B-A1B: <https://liquid.ai/blog/lfm2-5-8b-a1b>; <https://thedailypac.blogspot.com/2026/05/liquid-ai-releases-lfm25-8b-a1b-on.html>
- Gemma 4: <https://lmstudio.ai/models/gemma-4>
- Ling-mini-2.0: <https://huggingface.co/NarsAI/mini-2.0>; <https://docs.nvidia.com/nemo/automodel/latest/model-coverage/large-language-models/ling-2-0>
- Ring-mini-2.0 GGUF: <https://hug-proxy.pages.dev/redponike/Ring-mini-2.0-GGUF>
- SmallThinker: <https://arxiv.org/abs/2507.20984v2>
- Qwen3.5 family: <https://www.deeplearning.ai/the-batch/alibabas-latest-flagship-models-are-open-weights-moe-performers-in-sizes-from-less-than-1b-parameters>
- Granite 4.1 8B (dense) vs LFM2.5: <https://tinyweights.dev/posts/granite-4-1-8b-vs-lfm2-5-8b-a1b/>
