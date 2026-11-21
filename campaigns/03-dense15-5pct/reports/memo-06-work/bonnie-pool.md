# c3 memo-06 work file, Bonnie: candidate pool and pricing for the 12 tok/s, three-model goal

Agent model: `claude-fable-5-2` (Bonnie). Date 2026-11-21. Draft for the lead; the choice of the three is the lead's.
Selection rule: [memo-06 section 1](../memo-06-model-reselection.md), followed as written. **Nothing here is measured
on the phone, and no perplexity was run.**

Labels: *published* = the publisher's `config.json`, model card, licence file or the Hugging Face API
(`safetensors.total`), all read today; *host* = a read of the base tree `lab/llama.cpp-base` at `620d2225c`
(2026-10-15) or a GGUF header; *device* = a stored campaign 1 or 2 phone number, named by R-number; *estimated* = my
arithmetic. Every tok/s number is *estimated* and is the benchmark's printed value (0.95 x steady).

## 0. Summary

1. **28 eligible stock models from 22 families are priced; the per-token read spans
   3.33 to 4.79 GB at Q4_0** (*estimated* from *published* shapes; the shapes reproduce every published parameter
   count within 0.6 %).
2. **No candidate reaches 12 tok/s at r = 19 GB/s with any priced lever: the best is Yi-1.5-6B at 6.7 to 7.8.** At
   r = 35 the priced band tops out at 11.5 (Yi-1.5-6B), 11.2 (ChatGLM3-6B), 10.4 (DeepSeek-LLM-7B) and 10.2
   (Llama-2-7B, Granite-7B). 12.0 appears only in the k = 1.2 kernel corner, which the host did not reach (K1:
   1.53 to 1.64, memo-05).
3. **The ranking is the order of weights read per token.** The six fastest are 2023 to mid-2024 models with a
   Llama-2-shaped block (FFN 11,008) or fewer layers. Every 2025 to 2026 8B release reads 4.3 to 4.8 GB and prices
   2.5 to 3.2 tok/s lower at the top of the band at r = 35.
4. **No eligible model has a ReLU-family FFN.** All are SiLU-gated (SwiGLU) except Apertus (xIELU). There is no
   published exact-zero sparsity to rank on; criterion 2 rests on the host column screen (memo-02 H3), which
   exists for Mistral (40 %) and Qwen2.5 (35 %) only.
5. **Q4_0 stays the served type for every candidate.** Fewer bits read fewer bytes but have no arm64 dotprod
   kernel at the base (memo-01 section 2.3): Q3_K runs at half the `sdot` rate (*device*, R41) and is compute-bound
   below Q4_0 from about r = 19 upwards.
6. Proposed top five (strict rule, one per family): **Yi-1.5-6B-Chat, ChatGLM3-6B, Llama-2-7b-chat,
   Granite-7b-instruct, DeepSeek-LLM-7B-chat.** Each carries a flag in section 5.

## 1. Method of the web check

- For about 95 publisher accounts the Hugging Face API was listed and every model with 5.8 to 9.0 B parameters was read
  (`/api/models?author=...&expand[]=safetensors`); candidates were then read one by one (API record, `config.json`,
  card, licence file). Gated repositories (Meta, Google, Mistral, Cohere) give the parameter count through the
  API; their shapes come from an ungated mirror's `config.json` or a public GGUF header (*host*).
- llama.cpp support at the base: the converter class of each `architectures` entry is looked up in
  `lab/llama.cpp-base/conversion/__init__.py` and the architecture in `src/llama-arch.cpp` (*host*, tree read).
  **Not checked: tokenizer pre-hash, and an actual load of a GGUF at the base.** That is a host item for Tom.
- Limits: an account I did not list is not covered; the listing is capped at 1,000 models per account (Google and
  NVIDIA hit the cap, sorted by the API's default order).

## 2. Eligible pool (*published* unless marked)

All are decoder-only transformers with no expert routing, 6.0 to 8.5 B, stock chat or instruct weights.
"GQA" = attention heads / key-value heads. "Base" = architecture name at `620d2225c` (*host*).

| model (HF repo) | released | total params | layers / width / FFN / vocab | GQA | activation | tied head | licence | base | family and lineage |
|---|---|---|---|---|---|---|---|---|---|
| `01-ai/Yi-1.5-6B-Chat` | 2024-05 | 6,061,035,520 | 32 / 4096 / 11008 / 64,000 | 32/4 | SiLU | no | Apache-2.0 | `llama` | 01.AI; Yi continued (+500 B tokens) |
| `zai-org/chatglm3-6b` (was `THUDM/`) | 2023-10 | 6,243,584,032 | 28 / 4096 / 13696 / 65,024 | 32/2 | SiLU | no | ChatGLM3-6B licence: free for academic research, commercial use by registration | `chatglm` | Zhipu / Tsinghua; ChatGLM line |
| `meta-llama/Llama-2-7b-chat-hf` | 2023-07 | 6,738,417,664 | 32 / 4096 / 11008 / 32,000 | 32/32 | SiLU | no | Llama 2 community (gated, manual) | `llama` | Meta; Llama |
| `ibm-granite/granite-7b-instruct` | 2024-05 | 6,738,481,152 | 32 / 4096 / 11008 / 32,008 | 32/32 | SiLU | no | Apache-2.0 | `llama` | IBM; `ibm/granite-7b-base`, LAB-tuned by IBM. Whether the base was pretrained by IBM from scratch is on the card of the base, not read |
| `deepseek-ai/deepseek-llm-7b-chat` | 2023-11 | 6.91 B (*estimated* from the shape; the API has no count) | 30 / 4096 / 11008 / 102,400 | 32/32 | SiLU | no | DeepSeek licence (custom; card: released "to foster research") | `llama` | DeepSeek; from scratch, 2 T tokens |
| `allenai/OLMo-7B-0724-Instruct-hf` | 2024-07 | 6,888,095,744 | 32 / 4096 / 11008 / 50,304 | 32/32 | SiLU | no | Apache-2.0 | `olmo` | AI2; OLMo 1 |
| `cyberagent/calm2-7b-chat` | 2023-11 | 7,008,948,224 | 32 / 4096 / 11008 / 65,024 | 32/32 | SiLU | no | Apache-2.0 | `llama` | CyberAgent (Japanese) |
| `skt/A.X-3.1-Light` | 2025-07 | 7,264,800,768 | 32 / 4096 / 10880 / 102,400 | 32/32 | SiLU | no | Apache-2.0 | `llama` | SKT; from scratch, 1.65 T tokens (Korean first) |
| `allenai/Olmo-3-7B-Instruct` | 2025-11 | 7,298,011,136 | 32 / 4096 / 11008 / 100,278 | 32/32 | SiLU | no | Apache-2.0 | `olmo2` (class `Olmo3ForCausalLM` registered) | AI2; Olmo 3 |
| `allenai/OLMo-2-1124-7B-Instruct` | 2024-12 | 7,298,617,344 | 32 / 4096 / 11008 / 100,352 | 32/32 | SiLU | no | Apache-2.0 | `olmo2` | AI2; OLMo 2 |
| `mistralai/Mistral-7B-Instruct-v0.3` | 2024-05 | 7,248,023,552 | 32 / 4096 / 14336 / 32,768 | 32/8 | SiLU | no | Apache-2.0 | `llama` | Mistral |
| `BSC-LT/salamandra-7b-instruct` | 2024-09 | 7,768,117,248 | 32 / 4096 / 11008 / 256,000 | 32/8 | SiLU | no | Apache-2.0 | `llama` | BSC; from scratch (European languages) |
| `tiiuae/Falcon3-7B-Instruct` | 2024-11 | 7,455,550,464 | 28 / 3072 / 23040 / 131,072 | 12/4 (head 256) | SiLU | no | TII Falcon-LLM 2.0 | `llama` | TII; Falcon 3 |
| `trillionlabs/Tri-7B` | 2025-07 | 7,526,944,768 | 32 / 4096 / 11008 / 128,256 | 32/32 | SiLU | no | Apache-2.0 | `llama` | Trillion Labs (Korean) |
| `sapienzanlp/Minerva-7B-instruct-v1.0` | 2024-12 | 7,399,542,784 | 32 / 4096 / 14336 / 51,264 | 32/8 | SiLU | no | Apache-2.0 | `llama` | Sapienza; from scratch (Italian) |
| `Qwen/Qwen2.5-7B-Instruct` | 2024-09 | 7,615,616,512 | 28 / 3584 / 18944 / 152,064 | 28/4 | SiLU | no | Apache-2.0 | `qwen2` | Alibaba Qwen |
| `internlm/internlm2_5-7b-chat` | 2024-06 | 7,737,708,544 | 32 / 4096 / 14336 / 92,544 | 32/8 | SiLU | no | weights open for academic research; commercial by application | `internlm2` | Shanghai AI Lab |
| `LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct` | 2024-12 | 7,818,448,896 | 32 / 4096 / 14336 / 102,400 | 32/8 | SiLU | no | EXAONE licence: research only | `exaone` | LG |
| `FreedomIntelligence/openPangu-Embedded-7B` | 2025-08 | 8,030,890,112 | 34 / 4096 / 12800 / 153,376 | 32/8 | SiLU | no | openPangu licence (custom, not read) | `pangu-embedded` | Huawei; the upload is on a third party's account |
| `mistralai/Ministral-8B-Instruct-2410` | 2024-10 | 8,019,808,256 | 36 / 4096 / 12288 / 131,072 | 32/8 | SiLU | no | Mistral Research Licence | `llama` | Mistral (same family as Mistral 7B) |
| `meta-llama/Llama-3.1-8B-Instruct` | 2024-07 | 8,030,261,248 | 32 / 4096 / 14336 / 128,256 | 32/8 | SiLU | no | Llama 3.1 community (gated, manual) | `llama` | Meta (same family as Llama 2) |
| `kakaocorp/kanana-1.5-8b-instruct-2505` | 2025-05 | 8,030,285,824 | 32 / 4096 / 14336 / 128,259 | 32/8 | SiLU | no | Apache-2.0 | `llama` | Kakao. The shape and vocabulary are Llama 3.1's plus three tokens: lineage must be confirmed before it counts as its own family |
| `swiss-ai/Apertus-8B-Instruct-2509` | 2025-09 | 8,053,338,176 | 32 / 4096 / 21504 (two matrices, not three) / 131,072 | 32/8 | **xIELU** | no | Apache-2.0 | `apertus` | Swiss AI (EPFL, ETH, CSCS); from scratch |
| `Qwen/Qwen3-8B` | 2025-04 | 8,190,735,360 | 36 / 4096 / 12288 / 151,936 | 32/8 | SiLU | no | Apache-2.0 | `qwen3` | Alibaba Qwen (same family as Qwen2.5) |
| `openGPT-X/Teuken-7B-instruct-v0.6` | 2025-07 | 7,452,725,248 | 32 / 4096 / 13440 / 250,680 | 32/2 | SiLU | **yes** | CC-BY-NC-4.0 | `llama` | OpenGPT-X; from scratch (European languages) |
| `openbmb/MiniCPM4-8B` | 2025-06 | 8,185,253,888 | 32 / 4096 / 16384 / 73,448 | 32/2 | SiLU | no | Apache-2.0 | `minicpm` | OpenBMB / Tsinghua |
| `CohereLabs/c4ai-command-r7b-12-2024` (same shape: `aya-expanse-8b`) | 2024-12 | 8,028,033,024 | 32 / 4096 / 14336 / 256,000 (*host*, GGUF header) | 32/8 | SiLU | **yes** | CC-BY-NC-4.0 (gated) | `cohere2` (`command-r` for Aya) | Cohere |
| `ibm-granite/granite-3.3-8b-instruct` | 2025-04 | 8,170,864,640 | 40 / 4096 / 12800 / 49,159 | 32/8 | SiLU | **yes** | Apache-2.0 | `granite` | IBM (same family as Granite 7B) |

Eligible but not priced: `inceptionai/Jais-2-8B-Chat` (8,090,401,280; Apache-2.0; gated, so its shape was not
read; `Jais2ForCausalLM` is registered at the base); `baichuan-inc/Baichuan2-7B-Chat` (2023; 32 / 4096 / 11008 /
125,696, no published count, about 7.5 B *estimated*; community licence; `baichuan`).

Same family, same shape as a listed row (not separate candidates): Yi-6B-Chat, Llama-3-8B-Instruct,
Mistral-7B-Instruct-v0.1/v0.2 (7.242 B, vocabulary 32,000), Qwen2-7B-Instruct, Qwen1.5-7B-Chat (7.721 B),
internlm2-chat-7b, EXAONE-3.0-7.8B, Granite-3.0/3.1/3.2-8B, MiniCPM4.1-8B, OLMo-7B-Instruct (0424).

Activation and sparsity: none of the 28 has a ReLU, ReLU² or other exact-zero activation. Apertus's xIELU is smooth
and non-zero for negative inputs (*published*, model card and the base's `apertus.cpp`); it is not a sparsity
source. The only exact-zero 7B-class models found are RWKV-7 G1 (ruled out by the owner) and ReLU-fied fine-tunes
(ProSparse and the like), which are not stock weights.

## 3. Excluded, with the reason (*published* counts)

| model | total params | reason |
|---|---|---|
| `google/gemma-7b-it`, `gemma-1.1-7b-it` | 8,537,680,896 | above 8.5 B (by 0.04 B); also GELU, vocabulary 256,000, tied |
| `google/gemma-4-E4B-it` (7,996,156,490), `gemma-3n-E4B-it` (7.85 B), `nvidia/NVIDIA-Nemotron-3.5-Bolt-E4B` (7.996 B) | | owner, 2026-11-19: E4B is not dense 7B-class. Gemma 3n E4B and the Nemotron E4B have the same per-layer-embedding design or are built on it; listed under the same ruling, for the lead to confirm |
| `01-ai/Yi-1.5-9B-Chat` | 8,829,407,232 | above 8.5 B |
| `internlm/internlm3-8b-instruct` | 8,804,241,408 | above 8.5 B (48 layers) |
| `zai-org/glm-4-9b-chat` | 9,399,951,392 | above 8.5 B |
| `ibm-granite/granite-4.1-8b`, `granite-4.2-8b` | 8,791,592,960 | above 8.5 B (untied 100,352-token vocabulary on the 3.3 body) |
| `mistralai/Ministral-3-8B-Instruct-2512` | 8,918,026,240 | above 8.5 B as published (vision tower included); `mistral3` |
| `Qwen/Qwen3.5-9B` | 9,653,104,368 | above 8.5 B; hybrid (Gated DeltaNet) with a vision encoder. Qwen3.5 has no dense text model between 4B and 9B |
| `QCRI/Fanar-1-9B-Instruct` | 8.784 B | above 8.5 B |
| `pfnet/plamo-3-8b-fin-instruct`, `plamo-3-translate-8b` | 8.532 B | above 8.5 B; domain fine-tunes. `plamo-3-nict-8b-base` (8.091 B) is a base model with no instruct release |
| `microsoft/Phi-3-small-8k-instruct` | 7,392,274,432 | not supported: no `Phi3SmallForCausalLM` converter at the base (block-sparse attention, GEGELU); upstream issue #7439 |
| `tencent/Hunyuan-7B-Instruct` | 7,504,568,320 | criterion 3: the licence "does not apply in the European Union, United Kingdom and South Korea" and forbids use outside its territory. Supported at the base (`hunyuan-dense`); eligible only if the lead reads the licence otherwise |
| `XiaomiMiMo/MiMo-7B-RL` | 7,833,409,536 | class `MiMoForCausalLM` is not registered at the base (only MiMo V2); a reasoning model, no instruct release. Community GGUFs exist; unsure, Tom's load test would settle it |
| `tiiuae/Falcon-H1-7B-Instruct`, `Falcon3-Mamba-7B`, `Zyphra/Zamba2-7B-Instruct`, `allenai/Olmo-Hybrid-7B`, `nvidia/Nemotron-H-8B`, `NVIDIA-Nemotron-Nano-9B-v2` | 7.3 to 8.9 B | hybrid or pure state-space models, not decoder-only transformers by the definition (Falcon-H1 and Nemotron-H are supported at the base; lead's call if the definition is read wider) |
| `nvidia/Nemotron-Labs-Sigma-8B-instruct` (8.492 B), `tencent/WeDLM-8B-Instruct`, `nvidia/Efficient-DLM-8B` | | diffusion language models, not autoregressive decoders; no converter at the base |
| `microsoft/Phi-mini-MoE-instruct`, `arcee-ai/Trinity-Nano-Preview`, `LiquidAI/LFM2.5-8B-A1B`, OLMoE, Granite-4.0-H-Tiny | | expert routing |
| `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B` / `-Llama-8B`, `DeepSeek-R1-0528-Qwen3-8B`, `nvidia/Mistral-NeMo-Minitron-8B-Instruct`, `speakleash/Bielik-Minitron-7B` | | distilled or pruned derivatives |
| `skt/A.X-4.0-Light` | 7,259,624,960 | built on Qwen2.5 (*published*, card): a derivative, and the Qwen lineage |
| Llama-3.1 and Qwen fine-tunes by other publishers (SEA-LION, Tulu 3, Nemotron-Nano-8B-v1, UserLM-8b, Poro 2, CAT-8B, HuatuoGPT, lightonai Qwen3-8B-*, and the like) | | fine-tunes of another publisher's weights |
| code and maths models (CodeLlama-7b-Instruct, deepseek-coder-6.7b-instruct, CodeQwen1.5-7B-Chat, Qwen2.5-Math-7B) | | not listed as subjects: the gate is wikitext-2 prose. deepseek-coder-6.7b (6,740,512,768, Llama-2 shape) would price like Llama-2 |

## 4. Pricing (*estimated*)

**Method** (memo-03 section 9, unchanged; script `bonnie_price.py` beside this file prints both tables and
asserts that each shape reproduces the published parameter count):

    steady ms = max( (B_H + B_L x (1 - 0.889 s)) / r ,  (W_H + W_L x (1 - s) x k) / c ) + O
    printed tok/s = 0.95 x 1000 / steady ms

- Bytes: Q4_0 body at 4.5 bits per weight, head at Q6_K (6.5625), which is what `llama-quantize` writes for a
  Q4_0 file (*host*, memo-02 H1; a tied head is the Q6_K embedding, *host*, Command R7B header). Cross-check with
  headers: Mistral 4.04 against 4.037 GB, Qwen2.5 4.12 against 4.119, Command R7B 4.79 against 4.801.
- c = 69.9 G weights/s (three threads, power saving off clocks, in situ; *estimated* from *device* benches R37 and
  R41). O = 5 to 10 ms.
- "stock" = the stock Q4_0 path. "levers" = the better of stock and the exact column kernel with input thresholds:
  low end s = 0.30, k = 2.2, O = 10; high end s = 0.40, k = 1.55, O = 5. k's range is memo-05's device price (1.55
  to 2.2; *host* K1 1.53 to 1.64). s = 0.30 to 0.40 is **assumed** for every model that has no H3 screen (*host*:
  Mistral 40 %, Qwen2.5 35 %).
- "k = 1.2 corner" is memo-03's best case (s = 0.40, k = 1.2, O = 5), kept so the table can be read against the
  "about 12" of memo-03. No kernel at 1.2 exists.
- Multi-stream GEMV (pre-registered centre 1.05 x), a fourth thread and the CPU + GPU split are not in the bands.
- Order: the top of the lever band at r = 19 plus that at r = 35 (the two orders agree except inside ties).

| # | model | family | total B | W/token G | Q4_0 GB/token | head share | stock r=19 | levers r=19 | stock r=35 | levers r=35 | k=1.2 corner r=35 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | Yi-1.5-6B-Chat | 01.AI | 6.061 | 5.80 | 3.33 | 6 % | 5.1 to 5.3 | 6.7 to 7.8 | 9.0 to 9.5 | 9.0 to 11.5 | 13.9 |
| 2 | chatglm3-6b | Zhipu | 6.244 | 5.98 | 3.43 | 6 % | 5.0 to 5.1 | 6.5 to 7.6 | 8.8 to 9.2 | 8.8 to 11.2 | 13.5 |
| 3 | Llama-2-7b-chat | Meta | 6.738 | 6.61 | 3.75 | 3 % | 4.6 to 4.7 | 6.1 to 7.1 | 8.1 to 8.5 | 8.1 to 10.2 | 12.6 |
| 4 | granite-7b-instruct | IBM | 6.738 | 6.61 | 3.75 | 3 % | 4.6 to 4.7 | 6.1 to 7.1 | 8.1 to 8.5 | 8.1 to 10.2 | 12.6 |
| 5 | deepseek-llm-7b-chat | DeepSeek | 6.910 | 6.49 | 3.76 | 9 % | 4.6 to 4.7 | 5.9 to 6.8 | 8.1 to 8.5 | 8.1 to 10.4 | 12.2 |
| 6 | OLMo-7B-0724-Instruct | AI2 | 6.888 | 6.68 | 3.81 | 4 % | 4.5 to 4.6 | 6.0 to 6.9 | 8.0 to 8.3 | 8.0 to 10.1 | 12.4 |
| 7 | calm2-7b-chat | CyberAgent | 7.009 | 6.74 | 3.86 | 6 % | 4.5 to 4.6 | 5.9 to 6.8 | 7.9 to 8.2 | 7.9 to 10.0 | 12.1 |
| 8 | A.X-3.1-Light | SKT | 7.265 | 6.85 | 3.96 | 9 % | 4.4 to 4.5 | 5.7 to 6.5 | 7.7 to 8.0 | 7.7 to 9.8 | 11.7 |
| 9 | Olmo-3-7B-Instruct | AI2 | 7.298 | 6.89 | 3.98 | 8 % | 4.3 to 4.4 | 5.6 to 6.5 | 7.7 to 8.0 | 7.7 to 9.8 | 11.6 |
| 10 | OLMo-2-1124-7B-Instruct | AI2 | 7.299 | 6.89 | 3.98 | 8 % | 4.3 to 4.4 | 5.6 to 6.5 | 7.7 to 8.0 | 7.7 to 9.8 | 11.6 |
| 11 | Mistral-7B-Instruct-v0.3 | Mistral | 7.248 | 7.11 | 4.04 | 3 % | 4.3 to 4.4 | 5.7 to 6.6 | 7.6 to 7.9 | 7.6 to 9.5 | 11.8 |
| 12 | salamandra-7b-instruct | BSC | 7.768 | 6.72 | 4.05 | 21 % | 4.3 to 4.4 | 5.3 to 6.0 | 7.6 to 7.9 | 7.6 to 10.0 | 10.8 |
| 13 | Falcon3-7B-Instruct | TII | 7.456 | 7.05 | 4.07 | 8 % | 4.2 to 4.3 | 5.5 to 6.4 | 7.5 to 7.8 | 7.5 to 9.6 | 11.4 |
| 14 | Tri-7B | Trillion Labs | 7.527 | 7.00 | 4.07 | 11 % | 4.2 to 4.3 | 5.5 to 6.3 | 7.5 to 7.8 | 7.5 to 9.6 | 11.3 |
| 15 | Minerva-7B-instruct-v1.0 | Sapienza | 7.400 | 7.19 | 4.10 | 4 % | 4.2 to 4.3 | 5.6 to 6.5 | 7.5 to 7.8 | 7.5 to 9.4 | 11.6 |
| 16 | Qwen2.5-7B-Instruct | Alibaba Qwen | 7.616 | 7.07 | 4.12 | 11 % | 4.2 to 4.3 | 5.4 to 6.2 | 7.4 to 7.7 | 7.4 to 9.5 | 11.1 |
| 17 | internlm2_5-7b-chat | Shanghai AI Lab | 7.738 | 7.36 | 4.24 | 7 % | 4.1 to 4.2 | 5.3 to 6.1 | 7.2 to 7.5 | 7.2 to 9.2 | 11.0 |
| 18 | EXAONE-3.5-7.8B-Instruct | LG | 7.818 | 7.40 | 4.27 | 8 % | 4.0 to 4.1 | 5.3 to 6.1 | 7.2 to 7.5 | 7.2 to 9.1 | 10.9 |
| 19 | openPangu-Embedded-7B | Huawei | 8.031 | 7.40 | 4.33 | 12 % | 4.0 to 4.1 | 5.2 to 5.9 | 7.1 to 7.4 | 7.1 to 9.1 | 10.6 |
| 20 | Ministral-8B-Instruct-2410 | Mistral | 8.020 | 7.48 | 4.35 | 10 % | 4.0 to 4.1 | 5.2 to 5.9 | 7.1 to 7.4 | 7.1 to 9.0 | 10.6 |
| 21 | Llama-3.1-8B-Instruct | Meta | 8.030 | 7.50 | 4.36 | 10 % | 4.0 to 4.1 | 5.2 to 5.9 | 7.1 to 7.3 | 7.1 to 9.0 | 10.6 |
| 22 | kanana-1.5-8b-instruct | Kakao | 8.030 | 7.50 | 4.36 | 10 % | 4.0 to 4.1 | 5.2 to 5.9 | 7.1 to 7.3 | 7.1 to 9.0 | 10.6 |
| 23 | Apertus-8B-Instruct-2509 | Swiss AI | 8.053 | 7.52 | 4.37 | 10 % | 4.0 to 4.0 | 5.1 to 5.9 | 7.1 to 7.3 | 7.1 to 9.0 | 10.6 |
| 24 | Qwen3-8B | Alibaba Qwen | 8.191 | 7.57 | 4.42 | 12 % | 3.9 to 4.0 | 5.1 to 5.8 | 7.0 to 7.2 | 7.0 to 8.9 | 10.4 |
| 25 | Teuken-7B-instruct-v0.6 | OpenGPT-X | 7.453 | 7.45 | 4.46 | 19 % | 3.9 to 4.0 | 4.9 to 5.5 | 6.9 to 7.2 | 6.9 to 9.0 | 9.9 |
| 26 | MiniCPM4-8B | OpenBMB | 8.185 | 7.88 | 4.51 | 5 % | 3.8 to 3.9 | 5.1 to 5.8 | 6.8 to 7.1 | 6.8 to 8.6 | 10.5 |
| 27 | granite-3.3-8b-instruct | IBM | 8.171 | 8.17 | 4.65 | 4 % | 3.7 to 3.8 | 5.0 to 5.7 | 6.7 to 6.9 | 6.7 to 8.3 | 10.3 |
| 28 | c4ai-command-r7b-12-2024 | Cohere | 8.028 | 8.03 | 4.79 | 18 % | 3.6 to 3.7 | 4.6 to 5.2 | 6.5 to 6.7 | 6.5 to 8.4 | 9.3 |


Reading:
- At r = 19 every candidate is byte-bound; the lever band tops out at 7.8. At r = 35 every candidate is
  compute-bound on the stock path (6.5 to 9.5), and the column kernel's cost k decides.
- A large head costs twice: it is not sparsified and it is Q6_K. Salamandra reads fewer weights than Mistral and
  still prices lower with levers (head 21 % of the token); Command R7B and Teuken are last or near it for the same
  reason. Per-token head bytes run from 0.11 GB (Llama-2, Granite-7B, Mistral) to 0.86 GB (Command R7B).
- KV width (criterion 3) is small beside the body at ctx 512: 0.01 to 0.27 GB of F16 cache. Models with 32
  key-value heads (Llama-2, Granite-7B, DeepSeek, OLMo, A.X) carry four full attention matrices, 25 % of each
  block. That is in the counts above. memo-02 found attention inputs the more tolerant group, so these may screen
  better than uniform s suggests (not priced).

**What 12 printed needs** (steady 79.2 ms; 74 ms beside O = 5; *estimated*): at most 5.17 G weight-equivalents of
compute and 74 ms of bytes.

| model | s = 0.35 | s = 0.40 | s = 0.45 |
|---|---|---|---|
| Yi-1.5-6B | k ≤ 1.36, r ≥ 31.9 | k ≤ 1.48, r ≥ 30.0 | k ≤ 1.61, r ≥ 28.2 |
| ChatGLM3-6B | k ≤ 1.32, r ≥ 32.9 | k ≤ 1.43, r ≥ 30.9 | k ≤ 1.56, r ≥ 29.0 |
| Llama-2-7b / Granite-7b | k ≤ 1.20, r ≥ 35.4 | k ≤ 1.30, r ≥ 33.2 | k ≤ 1.42, r ≥ 31.0 |
| Mistral-7B-v0.3 | k ≤ 1.11, r ≥ 38.0 | k ≤ 1.20, r ≥ 35.7 | k ≤ 1.31, r ≥ 33.3 |

So even the smallest body needs all three of: a read rate of about 28 to 32 GB/s (memo-04 centre: 24), a column
frontier of 40 to 45 % inside +5 % at both batch settings, and a device kernel at or under the bottom of the host
range. The third model of any trio (Llama-2 shape, 3.75 GB) needs k ≤ 1.3 to 1.4, which no kernel has shown. On
these prices 12 tok/s on all three is not reached by the priced levers; a lever outside this table (Ebony's half)
would have to carry it.

**Served quantisation** (GB per token, *estimated*: Q4_0 bytes scaled by the file-size ratios of llama.cpp's
published Llama-3-8B table; "file + KV + app" is criterion 4 at Q4_0 with an F16 cache at ctx 512 and 0.4 GB for
the app, *estimated*):

|---|---|---|---|---|---|---|---|
| Yi-1.5-6B-Chat | 3.33 | 3.51 | 3.18 | 2.87 | 2.62 | 2.34 | 3.91 |
| chatglm3-6b | 3.43 | 3.62 | 3.27 | 2.96 | 2.70 | 2.41 | 4.00 |
| Llama-2-7b-chat | 3.75 | 3.96 | 3.58 | 3.23 | 2.96 | 2.64 | 4.49 |
| granite-7b-instruct | 3.75 | 3.96 | 3.58 | 3.23 | 2.96 | 2.64 | 4.49 |
| deepseek-llm-7b-chat | 3.76 | 3.97 | 3.59 | 3.24 | 2.96 | 2.64 | 4.65 |
| OLMo-7B-0724-Instruct | 3.81 | 4.02 | 3.64 | 3.29 | 3.00 | 2.68 | 4.60 |
| calm2-7b-chat | 3.86 | 4.07 | 3.68 | 3.33 | 3.04 | 2.71 | 4.68 |
| A.X-3.1-Light | 3.96 | 4.18 | 3.78 | 3.41 | 3.12 | 2.78 | 4.86 |
| Olmo-3-7B-Instruct | 3.98 | 4.20 | 3.80 | 3.43 | 3.14 | 2.80 | 4.88 |
| OLMo-2-1124-7B-Instruct | 3.98 | 4.20 | 3.80 | 3.43 | 3.14 | 2.80 | 4.88 |
| Mistral-7B-Instruct-v0.3 | 4.04 | 4.26 | 3.85 | 3.48 | 3.18 | 2.84 | 4.58 |
| salamandra-7b-instruct | 4.05 | 4.27 | 3.86 | 3.49 | 3.19 | 2.85 | 5.11 |
| Falcon3-7B-Instruct | 4.07 | 4.29 | 3.88 | 3.51 | 3.21 | 2.86 | 4.76 |
| Tri-7B | 4.07 | 4.30 | 3.89 | 3.51 | 3.21 | 2.86 | 5.04 |
| Minerva-7B-instruct-v1.0 | 4.10 | 4.32 | 3.91 | 3.53 | 3.23 | 2.88 | 4.68 |
| Qwen2.5-7B-Instruct | 4.12 | 4.34 | 3.93 | 3.55 | 3.24 | 2.89 | 4.85 |
| internlm2_5-7b-chat | 4.24 | 4.47 | 4.04 | 3.65 | 3.34 | 2.98 | 4.92 |
| EXAONE-3.5-7.8B-Instruct | 4.27 | 4.50 | 4.07 | 3.68 | 3.36 | 3.00 | 4.97 |
| openPangu-Embedded-7B | 4.33 | 4.56 | 4.13 | 3.73 | 3.41 | 3.04 | 5.15 |
| Ministral-8B-Instruct-2410 | 4.35 | 4.59 | 4.15 | 3.75 | 3.43 | 3.06 | 5.12 |
| Llama-3.1-8B-Instruct | 4.36 | 4.60 | 4.16 | 3.76 | 3.43 | 3.06 | 5.12 |
| kanana-1.5-8b-instruct | 4.36 | 4.60 | 4.16 | 3.76 | 3.43 | 3.06 | 5.12 |
| Apertus-8B-Instruct-2509 | 4.37 | 4.61 | 4.17 | 3.76 | 3.44 | 3.07 | 5.14 |
| Qwen3-8B | 4.42 | 4.66 | 4.21 | 3.81 | 3.48 | 3.11 | 5.24 |
| Teuken-7B-instruct-v0.6 | 4.46 | 4.70 | 4.25 | 3.84 | 3.51 | 3.13 | 4.87 |
| MiniCPM4-8B | 4.51 | 4.76 | 4.30 | 3.89 | 3.56 | 3.17 | 5.10 |
| granite-3.3-8b-instruct | 4.65 | 4.90 | 4.43 | 4.01 | 3.66 | 3.27 | 5.13 |
| c4ai-command-r7b-12-2024 | 4.79 | 5.05 | 4.57 | 4.13 | 3.77 | 3.36 | 5.25 |

- Criterion 4 holds for all (3.9 to 5.3 GB against 6.0). It does not count the anonymous repack copy: with it
  every candidate is 7 GB or more, so the no-copy path (memo-05 Y1b or Y1c) is a precondition for all of them.
  Residency itself is open until R69 (largest set held so far: 2.67 GB, *device*, R58).
- Kernels at the base on this CPU (*host*, memo-01 section 2.3): Q4_0 4x4, IQ4_NL 4x4, Q4_K/Q5_K/Q6_K 8x4 have a
  dotprod repack path; only Q4_0's rate is a *device* number. Q3_K, IQ3_*, IQ4_XS and IQ2_* have none.
- Q3_K_M reads 14 % fewer bytes but computes at half the rate (*device*, R41): Yi-1.5-6B would be 2.87 GB and
  about 5.5 tok/s at any r above 19 (*estimated*), under its Q4_0 stock price from r = 20. IQ3 types are slower
  again (0.22 of Q4_0, *published*, M2). Q4_K_M reads 5.5 % more than Q4_0. IQ4_NL reads the same bytes as Q4_0 and
  has a 4x4 kernel: it is the one alternative worth a host PPL row, because it would buy quality room, not speed.
- Whether each model's Q4_0 is inside F16 x 1.20 is unmeasured. *Published* for Llama-3-8B: +7.5 % without an
  imatrix; *host* for Mistral and Qwen2.5: Q4_0 is 1.02 to 1.04 of Q8_0 (memo-02).

## 5. Ranking and proposed top five

Criterion 1 orders the pool as the table in section 4. Criterion 2 cannot reorder it yet (two models screened).
Criterion 3 is already inside the price. Criterion 4 only breaks ties, and there is one: Llama-2-7b-chat and
Granite-7b-instruct have the same shape.

| rank | model | family | released | levers r = 19 | levers r = 35 | flags |
|---|---|---|---|---|---|---|
| 1 | Yi-1.5-6B-Chat | 01.AI | 2024-05 | 6.7 to 7.8 | 9.0 to 11.5 | a reserve of the earlier set; Apache-2.0; smallest body in the range |
| 2 | ChatGLM3-6B | Zhipu | 2023-10 | 6.5 to 7.6 | 8.8 to 11.2 | academic-research licence; own graph (`chatglm`: biased q/k/v, prefix tokens), so a base load and identity check come first; trained mostly on Chinese, wikitext PPL will be high (budgets are relative, the ceiling is not) |
| 3 | Llama-2-7b-chat | Meta | 2023-07 | 6.1 to 7.1 | 8.1 to 10.2 | gated (manual approval); TEAL *published* +3.0 % at 40 % on FP16, the most tolerant of its three models |
| 4 | Granite-7b-instruct | IBM | 2024-05 | 6.1 to 7.1 | 8.1 to 10.2 | Apache-2.0, ungated; lineage of `granite-7b-base` to be confirmed (it must not be Llama-2 weights) |
| 5 | DeepSeek-LLM-7B-chat | DeepSeek | 2023-11 | 5.9 to 6.8 | 8.1 to 10.4 | custom licence; 30 layers; head 9 % |
| 6 | OLMo-7B-0724-Instruct | AI2 | 2024-07 | 6.0 to 6.9 | 8.0 to 10.1 | superseded by OLMo 2 and Olmo 3 in the same family |
| 7 | calm2-7b-chat | CyberAgent | 2023-11 | 5.9 to 6.8 | 7.9 to 10.0 | Japanese |
| 8 | A.X-3.1-Light | SKT | 2025-07 | 5.7 to 6.5 | 7.7 to 9.8 | newest model in the top ten with Olmo 3 |
| 9 | Olmo-3-7B-Instruct | AI2 | 2025-11 | 5.6 to 6.5 | 7.7 to 9.8 | |
| 11 | Mistral-7B-Instruct-v0.3 | Mistral | 2024-05 | 5.7 to 6.6 | 7.6 to 9.5 | earlier subject; *host* 40 % |
| 16 | Qwen2.5-7B-Instruct | Alibaba | 2024-09 | 5.4 to 6.2 | 7.4 to 9.5 | earlier subject; *host* 35 % |
| 21 | Llama-3.1-8B-Instruct | Meta | 2024-07 | 5.2 to 5.9 | 7.1 to 9.0 | earlier reserve |

**Proposed top five: Yi-1.5-6B-Chat, ChatGLM3-6B, Llama-2-7b-chat, Granite-7b-instruct, DeepSeek-LLM-7B-chat**
(three subjects and two reserves in rank order; five families).

Reasons:
1. They are the five lowest per-token reads from five families, and the gap is not small: rank 1 reads 3.33 GB,
   rank 5 reads 3.76, the earlier lead subjects 4.04 and 4.12. At r = 35 that is 0.9 to 2.0 tok/s at the top of
   the band.
2. Yi-1.5-6B and ChatGLM3-6B are the only two with an "s = 0.45, k ≤ 1.56 to 1.61" row inside the host kernel
   range. Every other model needs a kernel nobody has measured.
3. Llama-2-7b has the one *published* 40 % column result inside the budget with margin (TEAL, FP16 weights).

What the rule does not see, for the lead (criterion 4 is last, so none of this moves the ranking):
- Ranks 1 to 7 are 2023 to mid-2024 models. The newest models that price within 1.7 tok/s of rank 1 at r = 35
  are A.X-3.1-Light and Olmo-3-7B-Instruct (2025), then Mistral-7B-v0.3 and Falcon3-7B.
- Licences: ChatGLM3, InternLM2.5, EXAONE, Ministral 8B, Command R7B and Teuken are research or non-commercial.
  That meets criterion 3 but such a model could not ship in the Play build.
- Disk: the laptop had 1.5 to 2.2 GB free while this was written (*host*, `df`). No candidate file (3.3 to
  4.8 GB) fits until space is freed; one header fetch failed for that reason.

## 6. Host screens that would move the ranking most (for Tom)

1. **H3 column frontier on ranks 1 to 9** (40 chunks, `-b 512` and `-ub 1`, thresholds from wikitext-2 train; the
   memo-02 tool is model-generic through the `llama` graph, ChatGLM3 and OLMo need a check). Five points of s are
   worth as much as the whole spread of the top five: at r = 35, s 0.35 to 0.45 moves Yi's kernel line from 1.36
   to 1.61. A model that screens at 30 % drops under one that screens at 45 % whatever its bytes. Per-group cells
   (attention 50 %) matter most for the five full-attention models.
2. **Q4_0 against F16 (or the Q8_0 proxy) per candidate**, with IQ4_NL as a second row: a candidate whose Q4_0 is
   outside F16 x 1.20, or near it, is out or needs Q4_K (5.5 % more bytes).
3. **Load and identity at the base** for the non-`llama` graphs (ChatGLM3, OLMo 1, `olmo2` with Olmo 3): GGUF
   converts, loads at `620d2225c`, per-chunk PPL equal with the runtime series' knobs unset.
4. **H1 header inventory on the actual files** once they exist: confirms the head type (Q6_K assumed) and that
   every row count is a multiple of 4 or 8 for the repack.

## 7. Not done, and where I may be wrong

- No GGUF of the top five was opened; their bytes are arithmetic on `config.json`. The check is that every shape
  reproduces its published count within 0.6 %.
- s is assumed for 26 of 28 models. If older, less trained models are more tolerant (TEAL's Llama-2 row hints at
  it) the top five gain; if not, the bands' top ends are optimistic.
- c is scaled from benches at other clocks; r is unmeasured with power saving off (R69).
- Jais-2-8B and Baichuan2-7B are not priced. Lineage is unverified for Granite-7B and Kanana 1.5.
- The account list is mine; a publisher I did not think of is missing.

## 8. Sources (read 2026-11-21)

- Hugging Face API and files, per model: `https://huggingface.co/api/models/<repo>`,
  `https://huggingface.co/<repo>/resolve/main/config.json` for every repo named in sections 2 and 3, for example
  <https://huggingface.co/01-ai/Yi-1.5-6B-Chat>, <https://huggingface.co/zai-org/chatglm3-6b>,
  <https://huggingface.co/meta-llama/Llama-2-7b-chat-hf>, <https://huggingface.co/ibm-granite/granite-7b-instruct>,
  <https://huggingface.co/deepseek-ai/deepseek-llm-7b-chat>, <https://huggingface.co/allenai/Olmo-3-7B-Instruct>,
  <https://huggingface.co/skt/A.X-3.1-Light>, <https://huggingface.co/Qwen/Qwen3.5-9B>,
  <https://huggingface.co/ibm-granite/granite-4.2-8b>, <https://huggingface.co/mistralai/Ministral-3-8B-Instruct-2512>,
  <https://huggingface.co/nvidia/Nemotron-Labs-Sigma-8B-instruct>.
- Licence texts: <https://huggingface.co/tencent/Hunyuan-7B-Instruct/blob/main/LICENSE>,
  <https://huggingface.co/THUDM/chatglm3-6b/blob/main/MODEL_LICENSE>,
  <https://huggingface.co/LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct/blob/main/LICENSE>.
- GGUF headers (*host*, first megabytes only): <https://huggingface.co/bartowski/c4ai-command-r7b-12-2024-GGUF>,
  <https://huggingface.co/bartowski/aya-expanse-8b-GGUF>.
- Phi-3-small: <https://huggingface.co/LiteLLMs/Phi-3-small-8k-instruct-GGUF/discussions/1> (points to llama.cpp
  issue #7439), <https://huggingface.co/spaces/ggml-org/gguf-my-repo/discussions/69>.
- Qwen3.5 sizes: <https://huggingface.co/Qwen/Qwen3.5-9B>, <https://huggingface.co/Qwen/Qwen3.5-4B>.
- llama.cpp quantisation table: <https://raw.githubusercontent.com/ggml-org/llama.cpp/master/tools/perplexity/README.md>
  (as quoted in memo-01-work/bonnie-draft.md section 2.1; not re-read today).
- TEAL: <https://arxiv.org/html/2408.14690v1> (as quoted in memo-01-work/bonnie-draft.md section 3; not re-read today).
- Repository: memo-01, memo-02 (H1, H3), memo-03 section 9, memo-05; `lab/llama.cpp-base` at `620d2225c`
  (`src/llama-arch.cpp`, `conversion/`).
