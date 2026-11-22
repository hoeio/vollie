# c3 memo-06: reselection of three dense subject models for the 12 tok/s goal

Date: 2026-11-21. Status: **open (joint round 4)**. Section 1 is the lead's selection rule, written before any
candidate is priced or screened; it does not change after results. Sections 2 onward are filled by Bonnie, Ebony,
Tom and the lead. Nothing in this memo is measured on the phone.

Trigger: CAMPAIGN.md Amendment 1 (owner, 2026-11-21): goal 12 tok/s, exactly three dense 7B-class models from
different families on the same Snapdragon 888 phone, met only when all three pass; subjects reselected.

## 1. Selection rule (lead, 2026-11-21, written first)

**Eligibility (all must hold):**
1. Campaign 3's definition of dense 7B-class (CAMPAIGN.md): decoder-only, no expert routing, 6.0 to 8.5 B total
   parameters as published (embeddings included), publisher's stock instruct weights (unquantized QAT checkpoints
   count; fine-tunes, pruned or distilled derivatives do not).
2. Supported by llama.cpp at the campaign base `620d2225c`, or at a later base the lead proposes (an owner-accepted
   base change is listed as a deviation in each pre-registration). Support is checked on the web and in the tree, not
   recalled.
3. Weights downloadable from the publisher (Hugging Face) under a licence that permits this research use.
4. The served file at the chosen quantisation fits the phone: file plus KV at ctx 512 plus app at most 6.0 GB
   (*estimated*), so it can be held resident on 8 GB.
5. Three **different families**: different publisher and different pretraining lineage (two releases of one
   publisher's line count as one family).

**Ranking (applied in this order to the eligible pool; ties go to the next criterion):**
1. Estimated tok/s on the phone with campaign 3's priced levers (memo-01, -03, -05 methods), at the read rate R69
   measures; until R69 reports, at r = 19 and r = 35 GB/s both. The quantity is bytes and compute read per token
   at the served quantisation that stays inside F16 x 1.20.
2. Exploitable runtime sparsity: published or *host*-screened activation sparsity inside the +5 % budget at both
   `-b 512` and `-ub 1` (memo-02's column screen is the reference method).
3. Per-token fixed cost: LM-head and embedding bytes (vocabulary x width), KV width (GQA ratio).
4. Recency and quality as published (a model nobody would run is a weaker subject, not an ineligible one).

**Process rules:**
- The previous set (Mistral-7B-Instruct-v0.3, Qwen2.5-7B-Instruct; reserves Llama-3.1-8B, Yi-1.5-6B) is part of the
  pool, with no preference.
- Every candidate that is priced or screened is reported, not only the three chosen.
- No candidate is measured on the phone before the three are chosen and their served files frozen.
- The lead chooses the three and records two reserves; the owner has delegated the choice (2026-11-21: "proceed
  without asking"); the owner can still replace a model at any time.
- Honest framing: memo-03 put the strict-dense best case near 12 tok/s (*estimated*). 12 on all three is a hard
  target; a model that cannot reach it is reported with numbers, and the rule above is not bent to pass.

## 2. Candidate pool (Bonnie)

Full draft, sources and script: [memo-06-work/bonnie-pool.md](memo-06-work/bonnie-pool.md). Read 2026-11-21 from the
Hugging Face API, each `config.json`, cards and licence files (*published*); llama.cpp support from the base tree
`620d2225c` (*host*: converter class and architecture are present; tokenizer hash and an actual load are not
checked). About 95 publisher accounts were listed for models of 5.8 to 9.0 B.

**Eligible (28 models, 22 families).** Shape = layers / width / FFN / vocabulary; GQA = heads / key-value heads.
All have a SiLU-gated FFN except Apertus (xIELU, two FFN matrices). **None has a ReLU-family activation**, so there
is no published exact-zero sparsity to rank on.

| model | released | total params | shape | GQA | tied head | licence | base arch | family |
|---|---|---|---|---|---|---|---|---|
| Yi-1.5-6B-Chat | 2024-05 | 6.061 B | 32 / 4096 / 11008 / 64,000 | 32/4 | no | Apache-2.0 | `llama` | 01.AI |
| ChatGLM3-6B | 2023-10 | 6.244 B | 28 / 4096 / 13696 / 65,024 | 32/2 | no | academic research; commercial by registration | `chatglm` | Zhipu |
| Llama-2-7b-chat | 2023-07 | 6.738 B | 32 / 4096 / 11008 / 32,000 | 32/32 | no | Llama 2 (gated) | `llama` | Meta |
| Granite-7b-instruct | 2024-05 | 6.738 B | 32 / 4096 / 11008 / 32,008 | 32/32 | no | Apache-2.0 | `llama` | IBM (lineage of its base to be confirmed) |
| DeepSeek-LLM-7B-chat | 2023-11 | 6.91 B (*estimated*) | 30 / 4096 / 11008 / 102,400 | 32/32 | no | DeepSeek licence | `llama` | DeepSeek |
| OLMo-7B-0724-Instruct | 2024-07 | 6.888 B | 32 / 4096 / 11008 / 50,304 | 32/32 | no | Apache-2.0 | `olmo` | AI2 |
| calm2-7b-chat | 2023-11 | 7.009 B | 32 / 4096 / 11008 / 65,024 | 32/32 | no | Apache-2.0 | `llama` | CyberAgent |
| A.X-3.1-Light | 2025-07 | 7.265 B | 32 / 4096 / 10880 / 102,400 | 32/32 | no | Apache-2.0 | `llama` | SKT |
| Olmo-3-7B-Instruct; OLMo-2-1124-7B-Instruct | 2025-11; 2024-12 | 7.298 B; 7.299 B | 32 / 4096 / 11008 / 100,278; 100,352 | 32/32 | no | Apache-2.0 | `olmo2` | AI2 |
| Mistral-7B-Instruct-v0.3 | 2024-05 | 7.248 B | 32 / 4096 / 14336 / 32,768 | 32/8 | no | Apache-2.0 | `llama` | Mistral |
| salamandra-7b-instruct | 2024-09 | 7.768 B | 32 / 4096 / 11008 / 256,000 | 32/8 | no | Apache-2.0 | `llama` | BSC |
| Falcon3-7B-Instruct | 2024-11 | 7.456 B | 28 / 3072 / 23040 / 131,072 | 12/4 | no | TII Falcon-LLM 2.0 | `llama` | TII |
| Tri-7B | 2025-07 | 7.527 B | 32 / 4096 / 11008 / 128,256 | 32/32 | no | Apache-2.0 | `llama` | Trillion Labs |
| Minerva-7B-instruct-v1.0 | 2024-12 | 7.400 B | 32 / 4096 / 14336 / 51,264 | 32/8 | no | Apache-2.0 | `llama` | Sapienza |
| Qwen2.5-7B-Instruct | 2024-09 | 7.616 B | 28 / 3584 / 18944 / 152,064 | 28/4 | no | Apache-2.0 | `qwen2` | Alibaba Qwen |
| internlm2_5-7b-chat | 2024-06 | 7.738 B | 32 / 4096 / 14336 / 92,544 | 32/8 | no | academic research; commercial by application | `internlm2` | Shanghai AI Lab |
| EXAONE-3.5-7.8B-Instruct | 2024-12 | 7.818 B | 32 / 4096 / 14336 / 102,400 | 32/8 | no | research only | `exaone` | LG |
| openPangu-Embedded-7B | 2025-08 | 8.031 B | 34 / 4096 / 12800 / 153,376 | 32/8 | no | openPangu licence (not read) | `pangu-embedded` | Huawei |
| Ministral-8B-Instruct-2410 | 2024-10 | 8.020 B | 36 / 4096 / 12288 / 131,072 | 32/8 | no | Mistral Research Licence | `llama` | Mistral |
| Llama-3.1-8B-Instruct | 2024-07 | 8.030 B | 32 / 4096 / 14336 / 128,256 | 32/8 | no | Llama 3.1 (gated) | `llama` | Meta |
| kanana-1.5-8b-instruct | 2025-05 | 8.030 B | 32 / 4096 / 14336 / 128,259 | 32/8 | no | Apache-2.0 | `llama` | Kakao (Llama 3.1's shape and vocabulary plus three tokens: lineage to be confirmed) |
| Apertus-8B-Instruct-2509 | 2025-09 | 8.053 B | 32 / 4096 / 21504 / 131,072 | 32/8 | no | Apache-2.0 | `apertus` | Swiss AI |
| Qwen3-8B | 2025-04 | 8.191 B | 36 / 4096 / 12288 / 151,936 | 32/8 | no | Apache-2.0 | `qwen3` | Alibaba Qwen |
| Teuken-7B-instruct-v0.6 | 2025-07 | 7.453 B | 32 / 4096 / 13440 / 250,680 | 32/2 | **yes** | CC-BY-NC-4.0 | `llama` | OpenGPT-X |
| MiniCPM4-8B | 2025-06 | 8.185 B | 32 / 4096 / 16384 / 73,448 | 32/2 | no | Apache-2.0 | `minicpm` | OpenBMB |
| Command R7B (12-2024); Aya Expanse 8B | 2024-12 | 8.028 B | 32 / 4096 / 14336 / 256,000 (*host*, header) | 32/8 | **yes** | CC-BY-NC-4.0 (gated) | `cohere2`; `command-r` | Cohere |
| Granite-3.3-8b-instruct | 2025-04 | 8.171 B | 40 / 4096 / 12800 / 49,159 | 32/8 | **yes** | Apache-2.0 | `granite` | IBM |

Eligible, not priced: Jais-2-8B-Chat (8.090 B, gated, shape not read), Baichuan2-7B-Chat (no published count).

**Excluded:**

| model | reason |
|---|---|
| Gemma 7B-it, Gemma 1.1 7B-it | 8.538 B, above 8.5 B |
| Yi-1.5-9B (8.829 B), InternLM3-8B (8.804 B), GLM-4-9B (9.400 B), Granite 4.1-8B and 4.2-8B (8.792 B), Ministral 3 8B (8.918 B with its vision tower), Qwen3.5-9B (9.653 B, hybrid), Fanar-1-9B (8.784 B), PLaMo 3 8B fin and translate (8.532 B, domain fine-tunes) | above 8.5 B |
| Gemma 4 E4B, Gemma 3n E4B, Nemotron 3.5 Bolt E4B | owner's E4B ruling (2026-11-19); the last two by the same reasoning, for the lead to confirm |
| Phi-3-small-8k-instruct (7.392 B) | no converter at the base (`Phi3SmallForCausalLM`; upstream issue #7439) |
| Hunyuan-7B-Instruct (7.505 B) | licence excludes the EU, the UK and South Korea from its territory (criterion 3); supported at the base |
| MiMo-7B-RL (7.833 B) | its class is not registered at the base (only MiMo V2); no instruct release |
| Falcon-H1-7B, Falcon3-Mamba-7B, Zamba2-7B, Olmo-Hybrid-7B, Nemotron-H-8B | hybrid or state-space, not decoder-only transformers |
| Nemotron-Labs-Sigma-8B, WeDLM-8B, Efficient-DLM-8B | diffusion language models; no converter at the base |
| Phi-mini-MoE, Trinity-Nano, LFM2.5-8B-A1B | expert routing |
| DeepSeek-R1-Distill 7B and 8B, Mistral-NeMo-Minitron-8B, Bielik-Minitron-7B | distilled or pruned |
| A.X-4.0-Light, and every Llama-3.1 or Qwen fine-tune by another publisher | derivatives of another publisher's weights |
| code and maths models (deepseek-coder-6.7b-instruct and the like) | not proposed: the gate is prose |

## 3. Pricing at 12 tok/s per candidate and challenge (Bonnie, Ebony)

### 3.1 Bonnie's prices (*estimated*; method of memo-03 section 9, script `memo-06-work/bonnie_price.py`)

- Bytes are Q4_0 for the body and Q6_K for the head, from the published shapes. The shapes reproduce every
  published parameter count within 0.6 %, and three headers agree within 0.3 % (*host*).
- tok/s is the printed value. "stock" is the stock path. "levers" is the better of stock and the exact column
  kernel with input thresholds, from (s 0.30, k 2.2) to (s 0.40, k 1.55); s is **assumed** for all but Mistral
  (40 %, *host*) and Qwen2.5 (35 %, *host*). The last column is memo-03's k = 1.2 corner; no kernel at 1.2 exists
  (*host* K1: 1.53 to 1.64).
- c = 69.9 G weights/s on three threads. Multi-stream GEMV, a fourth thread and the CPU + GPU split are not in
  the bands.

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

**Reading.**
- **No candidate reaches 12 at r = 19; the best is Yi-1.5-6B at 6.7 to 7.8.** At r = 35 the priced band tops out
  at 11.5 (Yi-1.5-6B) and 11.2 (ChatGLM3-6B); 12 appears only in the k = 1.2 corner.
- 12 printed needs at most 74 ms of bytes and 5.17 G weight-equivalents of compute. Yi-1.5-6B gets there with
  s = 0.40, k ≤ 1.48 and r ≥ 30.0, or s = 0.45, k ≤ 1.61 and r ≥ 28.2. A Llama-2-shaped body needs k ≤ 1.30 at
  s = 0.40 and r ≥ 33.2; Mistral needs k ≤ 1.20 and r ≥ 35.7.
- The order is the order of weights read per token. A large head costs twice (not sparsified, Q6_K): the head
  share runs from 3 % to 21 %.
- **Q4_0 stays the served type for every candidate.** Q3_K_M reads 14 % fewer bytes at half the kernel rate
  (*device*, R41) and loses from about r = 20; IQ3 and IQ4_XS have no arm64 kernel at the base; Q4_K_M reads 5.5 %
  more. IQ4_NL (same bytes, 4x4 kernel) is worth a host PPL row for quality room. Whether each Q4_0 is inside
  F16 x 1.20 is unmeasured.
- Criterion 4 holds for all at Q4_0: file plus KV plus app is 3.9 to 5.3 GB. With the anonymous repack copy every
  candidate is 7 GB or more, so a no-copy path (memo-05 Y1b or Y1c) is a precondition for all of them.

**Ranking by section 1 and proposed top five** (one per family, in rank order): **Yi-1.5-6B-Chat (01.AI),
ChatGLM3-6B (Zhipu), Llama-2-7b-chat (Meta), Granite-7b-instruct (IBM), DeepSeek-LLM-7B-chat (DeepSeek).** Then
OLMo-7B-0724-Instruct, calm2-7b-chat, A.X-3.1-Light, Olmo-3-7B-Instruct. The earlier set ranks 1 (Yi-1.5-6B),
11 (Mistral), 16 (Qwen2.5) and 21 (Llama-3.1).

Flags for the lead:
- ChatGLM3-6B: academic-research licence, its own graph (`chatglm`), trained mostly on Chinese.
- Llama-2-7b-chat is gated; Granite-7b-instruct has the same shape, and its base's lineage must be confirmed.
- Ranks 1 to 7 are 2023 to mid-2024 models. Criterion 4 is last, so the ranking does not see this. The newest
  models within 1.7 tok/s of rank 1 at r = 35 are A.X-3.1-Light and Olmo-3-7B-Instruct.
- Research-only or non-commercial licences (ChatGLM3, InternLM2.5, EXAONE, Ministral 8B, Command R7B, Teuken)
  meet criterion 3 but could not ship in the Play build.
- The laptop had 1.5 to 2.2 GB of disk free while this was written (*host*); no candidate file fits until space
  is freed.

**Host screens that would move the ranking most (for Tom):**
1. The H3 column frontier on ranks 1 to 9 at `-b 512` and `-ub 1`. Five points of s are worth the whole spread
   of the top five.
2. Q4_0 against F16 (or the Q8_0 proxy) per candidate, with an IQ4_NL row.
3. Load and identity at the base for the non-`llama` graphs (ChatGLM3, OLMo 1, Olmo 3 on `olmo2`).
4. The H1 header inventory on the real files (head type, row counts for the repack).

### 3.2 Ebony's challenge

Ebony, 2026-11-21. Full text, tables and sources: [memo-06-work/ebony-levers.md](memo-06-work/ebony-levers.md).
Nothing was run. Numbers are *published* (source claims, read 2026-11-21), quoted *host* results of this
repository, or *estimated* with t = max(B ÷ r, W ÷ c) + 8 ms at R69's pre-registered bands (r 24, upper case 35;
c 40.4 GB/s; four cores 50.3). **My tok/s are steady values; printed is 0.95 x (Bonnie's convention, adopted
below).**

**Scout.** Nothing published in 2025 to 2026 changes the arithmetic.
- *Sparsity.* Prox (arXiv 2607.27591, 2026-07) is whole-neuron FFN skipping ranked by a low-bit proxy, the parked
  "row skip" of memo-05. Its PPL table (*published*, FP16 weights): Qwen3-8B 60 % FFN sparsity at +4.9 %,
  Mistral-7B 50 % at +4.1 %, Llama-3-8B +5.6 % already at 40 %, Gemma-3-12B (GeGLU) collapses at 40 %. WINA needs
  a weight transform (not file-unchanged; its score alone is a side table). R-Sparse adds low-rank compute.
  SGA-Sparse is trained.
- *Kernels.* No 4-bit ARM kernel beats repacked `sdot` on a core without i8mm or SME2. The best published NEON
  kernels run 3-bit types at 1.5 x (Q3_K) to 3 x (IQ3) per weight (ik_llama.cpp, M2 Max). LUT kernels (T-MAC,
  T-SAR) remain 1 to 2 bit.
- *Lossless lookahead* (Token Recycling, Cacheback, UniSpec): pays only while byte-bound (batch 2 at r 24:
  +0.5 to +1.3 tok/s at acceptance 0.4), is negative when compute-bound, and does not stack with sparsity.
- *Head.* No exact method found; the HNSW head (arXiv 2608.27460) is approximate with −inf misses. Campaign 2's
  two-stage head is still the tool.

**Stack and price (*estimated*, steady).** I agree with §5.1's weight line. With this repository's kernel costs,
proxy neuron skipping at s 0.5 and k 1.8 saves 24 % of the token's bytes and **no compute** (FFN compute factor
1.07); it cuts compute only at k ≤ about 1.5 (factor 0.77 at s 0.6). Stack = no-copy layout, two-stage head,
`down` columns (K1), whole-neuron skip of `gate` + `up`, then a 4th big core.

| body (multiplied G; GB at Q4_0) | dense r 24 / r 35 | stack, central: r 24 / r 35 / r 35 + 4 cores | stack, optimistic |
|---|---|---|---|
| Yi-1.5-6B class (5.80; 3.26) | 7.0 / 9.9 | 8.9 / 10.8 / **12.6** | 9.8 / **13.4** / **13.8** |
| Llama-2-7B class (6.61; 3.72) | 6.1 / 8.8 | 7.9 / 9.6 / 11.3 | 8.7 / 11.9 / **12.3** |
| Mistral-7B-v0.3 (7.11; 4.04) | 5.7 / 8.1 | 7.4 / 8.8 / 10.5 | 8.2 / 11.2 / 11.6 |
| Qwen2.5-7B (7.07; 4.12) | 5.6 / 8.0 | 7.3 / 8.7 / 10.3 | 8.0 / 10.9 / 11.4 |
| Llama-3.1-8B (7.50; 4.5) | 5.1 / 7.3 | 6.6 / 8.0 / 9.4 | 7.3 / 10.0 / 10.3 |

Printed (x 0.95): the 5.8 G corners are 12.0 / 12.7 / 13.1 and Llama-2's 12.3 is 11.7. Residency is not in these
rows; each needs C of 3.8 to 5.2 GB (body plus proxy) against R69's central 3.1. **I do not see 12 on a 7.0 G+
body.** My judgement: P(12) about 0.07 on a 5.8 G body, about 0.02 on a 7.1 G body, P(all three) at most 0.02.
The lead's prior is 0.05 to 0.15; both stand.

**New proposal.** Skip neurons in groups of four rows of the stock 4x4 repack: kept weights cost k = 1.0, no new
SIMD kernel, bit-identical at an empty skip list. Reordering rows is exact, so groups can be formed offline from
co-activation. What decides it is the share of groups skippable inside the budget (prediction before data: 20 to
35 % in file order, 30 to 45 % clustered; P(≥ 40 % on two models) 0.3).

**Model properties, in order.**
1. Multiplied weights per token (total minus lookup-only embeddings): about 1 tok/s per 0.5 G near 12. A large
   **untied** vocabulary is an advantage under the "embeddings included" definition; a tied one is not.
2. Neuron-skip tolerance, host-screened per candidate; published order Qwen > Mistral > Llama-3, GeGLU fails;
   Llama-2 has two published values that disagree.
3. FFN share of bytes (strong GQA helps through attention bytes, not KV).
4. Quantisation robustness under 4.5 bpw, forced by residency; a mixed file (3-bit on `gate` + `up` only) is the
   form that does not spend the compute line.
5. Head type and tying.
6. Activation: no ReLU-family eligible model (Bonnie's pool confirms).

**Challenge to §5.1's corner.** A low-bit **column** kernel at k ≤ 1.3 to 1.5 is about 2.3 to 3 by the published
and measured factors (3-bit dense 1.5 x or more, times K1's 1.53 to 1.64, *host*), and K-types have no exact
column re-layout. I put it under 0.03 per subject and set the mixed file with row skipping beside it.

**Challenge to §2 and §3.1** (details: ebony-levers.md §4.2; the seven checks were written before reading her
draft).
- Agreed: her W per token matches mine; ranks 1 and 2 (Yi-1.5-6B, ChatGLM3-6B). I adopt "printed = 0.95 x
  steady"; §5.1's 5.4 G weight line is 5.17 G in that convention. The three files need one convention.
- Residency is in none of the prices: at R69's central C 3.1 GB no Q4_0 candidate is resident (smallest 3.33 GB).
  A column "largest type that fits C, and its price" is missing.
- The served-quantisation table in bonnie-pool.md §4 has no header row.
- Llama-2's rank leans on TEAL's own +3.0 % at 40 %; LaRoSA's re-run has +15.4 % at 25 % (*published*, memo-01
  work §2.2). No top-five model is screened; s is assumed for 26 of 28.
- Third subject: at 12 the weight line binds, so order by W puts DeepSeek-LLM-7B (6.49 G, 30 layers) ahead of
  Llama-2 / Granite-7B (6.61 G). Her own r 35 column agrees (10.4 against 10.2).
- The head penalty is only half a model property: the two-stage head reads a Q6_K head at 4.5 bits (Salamandra
  about +0.3 tok/s at r 19, *estimated*).
- One c (39.3 GB/s) and no four-core row: Yi's "k ≤ 1.48" is about 1.3 at c 35 and about 1.85 at four cores.
- Disk: 1.5 to 2.2 GB free (*host*, her `df`) blocks every host screen of both lists.

**Test list (host, Tom).**

| # | test | closes when |
|---|---|---|
| T1 | group-of-four neuron rule (E1's tool), file and clustered order, 20 to 50 % of groups, both batch settings | under 30 % inside +4.5 % on either model, clustered |
| T2 | proxy ranking with 1- and 2-bit copies (memo-05 E3, never run), `down` input 40 / 50 / 60 % | 1-bit under 40 % on either model |
| T3 | weight-norm score without the transform (memo-04 work T6, never run) | no 5-point move on either model |
| T4 | family screen on ranks 1 to 7: E1's `down`-input cells and H3's uniform cells | gives criterion 2 a number; under 50 % `down` input = "no neuron lever" |
| T5 | per-type gemv cost per weight (Q4_0x4, Q4_K, IQ4_NL, Q3_K, IQ3_S), host, before S2's device rows | 3-bit above 1.3 x in the best kernel closes the all-3-bit corner |
| T6 | 1-bit proxy kernel (`TBL`) against `sdot`, hot and cold | above 0.35 x per weight → reprice; FFN compute factor ≥ 1.0 closes proxy levers |
| T7 | multiplied-weight inventory per chosen candidate (real GGUF headers) | no kill; feeds criteria 1 and 3 |
| T8 | draft-size-1 acceptance on the prompt set, only if R69 reads r under 28 | under 0.30 on either model |

**Try list (device, later; each pre-registered).**

| # | lever | gate | class | *estimated* gain |
|---|---|---|---|---|
| Y8 | skip list for groups of four rows, with a row-permutation side file | T1, T2, T6 | bit-identical at an empty list; numerics above | +1.7 to +2.9 tok/s over `down` columns |
| Y9 | mixed served file (3-bit `gate` + `up`, 4-bit elsewhere) with row skipping | T5, H2, S3 | served-file choice | about 0.3 to 0.4 GB less resident on a 5.8 G body |
| Y10 | X1 + 3 A78, A55s on proxy and head by weighted chunks | R69, S2 | bit-identical | x 1.13 to 1.25 on compute when compute-bound, 0 when byte-bound |
| Y11 | campaign 2's two-stage head per subject | none | numerics, +0.01 % (*host*, c2) | +0.1 to +0.3 tok/s at 100 K+ vocabularies |
| Y12 | batch-2 lookahead | T8, r under 28 | bit-identical if gemm = gemv holds on the phone | +0.5 to +1.3 while byte-bound |

## 4. Host screens (Tom)

Full record: [memo-06-work/tom-screens.md](memo-06-work/tom-screens.md). All numbers are *host* or *estimated*.

**Update 2026-11-22 (owner freed disk).** The five provisional models are screened (tom-screens §5, §6). Method:
- Publisher weights at pinned commits, LFS sha256 checked, converted to F16 and quantised to Q4_0 here.
- 40 chunks; the gate's budget is stock Q4_0 × 1.05, and the pass line is +4.5 %.
- The bullets after this update are the state of 2026-11-21, kept as written.

| model (§6 slot) | Q4_0 / F16 | uniform column sparsity inside +4.5 % at both `-b 512` and `-ub 1` | ΔPPL (`-b 512` / `-ub 1`) | next level up (`-b 512`) |
|---|---|---|---|---|
| Yi-1.5-6B-Chat (subject 1) | ×1.0236 | **40 %** | +1.55 / +2.79 % | 45 %: +5.11 % |
| ChatGLM3-6B (subject 2) | ×1.0116 | **20 %** | +3.56 / +3.51 % | 30 %: +4.32 % at `-b 512` but +5.05 % at `-ub 1` |
| DeepSeek-LLM-7B-chat (subject 3) | ×1.0367 | **20 %** | +3.26 / +3.48 % | 30 %: +5.13 % |
| Llama-2-7b-chat (reserve 1) | ×1.0302 | **20 %** | +4.20 / +4.31 % | 30 %: +5.69 % |
| Granite-7b-instruct (reserve 2) | ×1.0263 | **15 %** | +4.25 / +4.20 % | 20 %: +5.73 % |
| memo-02: Mistral-7B-v0.3 / Qwen2.5-7B | ×1.0195 / ×1.0357 (Q8_0 proxy) | 40 % / 35 % | | |

- **Slot 3:**
  - DeepSeek and Llama-2 share the frontier at 20 % (no 25 % level in the tables).
  - DeepSeek has about 0.9 points more margin there.
  - Llama-2 has the smaller Q4_0 loss.
  - Criterion 2 leans to DeepSeek, narrowly.
- **Caveats (tom-screens §5.1, §5.3):**
  - DeepSeek's GGUF has `add_bos = false` (a converter rule), so every DeepSeek PPL is scored without BOS.
  - ChatGLM3 was converted without executing its custom code; the tokenizer was checked against sentencepiece.
  - ChatGLM3's stock PPL (33.0) is about 4.5 times the others and is unexplained. The `[gMASK]sop` prefix makes it
    worse (8 chunks: 48.26 → 53.36). Its level should be checked before it is frozen.
- **Criterion 2 spreads by a factor of 2.7**: Granite 15 %, DeepSeek, Llama-2 and ChatGLM3 20 %, Yi 40 %. The
  column lever is worth much less on the 2023-era MHA models than memo-02 measured on Mistral and Qwen2.5.
- **Ebony T1, finished** (groups of four neurons, exact score), inside +4.5 % at both settings:

  | model | file order (`-ub 1`) | clustered (`-ub 1`) |
  |---|---|---|
  | Mistral | 30 % (+3.60 %) | about 33 % (+3.51 %) |
  | Qwen2.5 | about 37 % (+4.03 %) | about 36 % (+2.92 %) |

  Against Ebony's pre-data band:
  - File order 20 to 35 %: hit on Mistral; Qwen2.5 missed high by 2 points.
  - Clustered 30 to 45 %: hit, low half.
  - ≥ 40 % on two models: did not happen.
  - By realised share, clustering moves the frontier by 0 to 3 points.

- **Tools.** `tools/c3_colskip.py` and `tools/c3_h3_grid.sh` are now model-generic.
  - Two silent errors are fixed. On post-norm blocks (`olmo2`: OLMo-2, Olmo-3) the old tool zeroed the residual
    stream, and it never calibrated the gate/up input. OLMo-2 1B, uniform 30 %: old PPL 16.31, fixed 13.57
    (stock 13.63).
  - A consumer plan (in place, restore after the last mat-mul, or refuse) is printed per model.
  - Identity with `llama-perplexity` holds on llama, gemma3, exaone, stablelm, qwen2 and olmo2 at `-b 512` and
    `-ub 1`.
  - memo-02's Mistral per-chunk values are reproduced bit for bit.
  - Parallel blocks (`cohere2`, Command R7B) are refused, not mis-screened.
  - No runtime patch was touched.
- **Header inventory (T7), 24 candidates, no download.** Q4_0 bytes per token (*estimated*):

  | candidate | GB/token |
  |---|---|
  | Yi-1.5-6B | **3.33** |
  | ChatGLM3-6B | **3.43** |
  | Llama-2-7B, Granite-7b-instruct | 3.75 |
  | DeepSeek-LLM-7B | 3.76 |
  | A.X-3.1-Light | 3.96 |
  | Olmo-3-7B | 3.98 |
  | Mistral-7B-v0.3 (memo-02) | 4.04 |
  | Qwen2.5-7B (memo-02) | 4.12 |
  | Llama-3.1-8B | 4.36 |

  Every architecture is in llama.cpp at `620d2225c`. Teuken, Command R7B and Granite-3.3 have tied heads.
- **H3 column frontier and the Q4_0 vs Q8_0/F16 ratio: not screened: disk** for every new candidate, including
  all of Bonnie's top five. No pool file is on the laptop, and free disk was 2.2 to 6.1 GB against the 5 GB floor.
  One model at a time needs about 16 GB free (Q4_0 + Q8_0 + floor), or about 10 GB for H3 alone.
- The only screened models are still memo-02's (40 chunks, at `-b 512` and `-ub 1`):

  | model | inside the margin | Q4_0 / Q8_0 |
  |---|---|---|
  | Mistral-7B-v0.3 | 40 % | x 1.0195 |
  | Qwen2.5-7B | 35 % | x 1.0357 |

- **Ebony T1** (whole-neuron skip in groups of four, exact score; *host*, **partial**):
  - Mistral-7B-v0.3 at `-b 512`, 40 chunks: about **30 %** of groups inside +4.5 % in file order (+3.65 %) and about
    **33 to 35 %** clustered (30 %: +2.60 %; 38 % realised: +4.78 %).
  - Both inside Ebony's pre-data bands. ≥ 40 % is not reached on Mistral.
  - Not run: Qwen2.5 and `-ub 1`. The Qwen2.5 calibration filled swap and the disk (about 0.23 GB free), and
    was stopped.
  - So T1 is not yet a two-model result.
- Selection criterion 2 (sparsity) therefore has no number for any new candidate. If the choice cannot wait for
  disk, it rests on criteria 1 and 3.

## 5. Re-plan at 12 tok/s: levers, rounds, kill lines (lead)

Full arithmetic and reasoning: [memo-06-work/clyde-replan.md](memo-06-work/clyde-replan.md). Every number here is
*estimated*. R69's bands are the inputs until R69 reports.

**5.1 What 12 tok/s requires.** The token must take at most 83.3 ms, with a fixed cost O of about 8 ms (6 to 12).
Two lines must hold at once:
- **Byte line:** bytes read per token ≤ r × 75 ms. That is 1.43 GB at r 19, 1.81 at 24, 2.64 at 35 and 3.39 at 45.
- **Weight line:** touched weights × the kernel's cost per weight ≤ about 5.4 G stock Q4_0 weights. This is at
  R69's central c_b of 40.4 GB/s on X1 + 2 A78; the band is 4.7 to 6.0.

What the lines mean:
- A dense Q4_0 body is out on bytes. At 7.0 B multiplied weights (4.0 GB) it needs r ≥ 54 GB/s, which is above
  the bus. At 6.0 B (3.4 GB) it needs 45 GB/s.
- **Every 6.0 to 8.5 B dense model is also out on compute.** At the stock Q4_0 cost per weight it stays below 12
  tok/s at any read rate: about 9.5 tok/s at 7.0 B multiplied and 10.9 at 6.0 B.
- Fewer bits shrink the bytes but not the compute, unless the low-bit kernel costs no more per weight than Q4_0.
  Q3_K costs about twice as much (*device*).

**The corner that reaches 12 on one subject** needs all three of these:
1. A served file of about 3.1 to 3.7 bpw. The floor is the F16 × 1.20 ceiling, about 3.05 bpw on Llama-3-8B
   (*published*). The top is residency at C about 3.1 to 3.4 GB.
2. Column sparsity at d about 0.60, with an exact low-bit column kernel at k ≤ 1.3 (P 7.0 B) to 1.5 (P 6.0 B).
   The best measured kernel today is K1 on Q4_0 at k 1.53 to 1.64 (*host*).
3. r of about 20 to 28 GB/s or more, or a GPU lane or 4th core that adds compute.

**Lead's prior:** P(all three pass) is about 0.05 to 0.15. This is a judgement, not a price; §3 replaces it. It is
told to the owner now.

**Levers this puts first** (Bonnie and Ebony price them in §3):
- a fast arm64 kernel for one sub-4-bit type, bit-identical to ggml's reference for that type;
- an exact column kernel for that type;
- the GPU as a compute lane (R69 Rule D);
- X1 + 3 A78 threading.

**5.2 R69 under the amendment.** R69 loads no model, so its measurements and bands stay valid. Its Rules A to C
and §6 named the old subjects and 15 tok/s. Before any R69 data exists, they are adapted in
[prereg/R69-amendment-1.md](../prereg/R69-amendment-1.md):
- **Rule A** is scored at reference densities 0.60 and 0.65. Each new subject gets a derived verdict, with the
  formula fixed now. It does not apply to non-Q4_0 files.
- **Rule B's** inner kill becomes "any of the three subjects".
- **Rule C's** consequences are restated for any Q4_0 body of 6 B or more. Below 3.4 GB, subjects are frozen at
  files that fit C.
- The **goal** reads 12 on each subject.
- **Byte and weight lines** are reported from R69's numbers.

Nothing that is measured or predicted changes.

**5.3 Rounds and kill lines.** The next free R-number after R69 is R72; the next campaign-3 kit is r32.

| stage | content | kill line |
|---|---|---|
| S0 (now) | §2 to §4; choice (§6) at R69's r if reported, else at r 19 and 35 | – |
| S1 = R69 | probe kit r29 + amendment 1; no model; can run now | **K0** per subject: no priced lever set meets both lines at R69's r, c_b, C → "out of reach by arithmetic", written with numbers |
| S2 = R72 | kit r32, no model: D2 rows (K1, Y1b), per-type gemv rows on synthetic arenas (Q4_0, Q4_K, IQ4_NL, Q3_K, IQ3_S, Q2_K at least), X1 + 3 A78, D3 header | **K-type**: a type with no room under the weight line for any subject is dropped from the freeze |
| S3 | freeze per subject (*host*, owner's laptop). Quantise from the publisher's BF16 with stock `llama-quantize`. Measure full-set F16 and stock PPL. Check stock ≤ F16 × 1.20 and resident ≤ C − 0.15 GB. Take the lowest priced token time. Write the budget (stock × 1.05, absolute) in the model card, with the sha256s | **K-F**: no type fits the ceiling and C → reported; the lead proposes a reserve, only the owner replaces |
| S4 | baseline: all three subjects in one session; D1 residency over 20 passes; median of 10 at campaign 2's best bit-identical configuration | **K1** per subject: baseline (*device*) × upper-band gains of the open levers < 12.0 → "out of reach with priced levers" |
| S5 | lever rounds, one pre-registration each. Numerics levers first pass full-set PPL at `-b 512` and `-ub 1` per subject. **Every lever is measured on at least two of the three subjects in the same session** (all three when the kit holds them), in-run ABBA, reported per model with what generalised and what did not | each lever's own kill line; K1 re-applied after each round |
| S6 | go block: median ≥ 12.15 and at most 2 of 10 runs below 12.0, uncapped clocks. Then one claim per subject: N = 10 plain runs, positions fixed, no run dropped, power saving off, deep-sleeping list recorded, charger out; claim configuration within budget at both settings and ≤ F16 × 1.20 | pass per subject: median ≥ 12.0. **Goal: all three pass.** A failed claim stays in the log; another attempt needs a new pre-registration and a changed configuration |

**5.4 Owner questions (a) to (d) from before the amendment.** These are lead decisions, provisional until the
owner confirms them (owner delegated, 2026-11-21). Reasons are in the work file §3.
- **(a)** Y1a side file is a weight-derived artifact: **yes**. COMMON_RULES 1 (R42 amendment 2) and the 2026-11-19
  sidecar regime already cover it.
- **(b)** Twice the storage: **yes, for the research kit**. Free storage is recorded per round.
- **(c)** A column-major copy: **yes**, as an exact re-layout. A 1-bit proxy: **yes**, as lossy derived data
  inside the budget with its own PPL check. This is the least certain of the four.
- **(d)** The served Q4_0 file is **superseded**. Each subject is quantised by us from the publisher's BF16 at the
  base, into one file, with every sha256 recorded and no third-party GGUF. The type is chosen in S3, not assumed
  to be Q4_0.

## 6. Choice of the three and reserves (lead)

**Provisional, 2026-11-21 (lead).** Applied section 1 to Bonnie's pool (section 2), Ebony's challenge (3.2) and
Tom's header inventory (4). All tok/s below are *estimated*; no candidate is screened yet.

| slot | model | family | Q4_0 GB/token | levers at r 35 (Bonnie) | why |
|---|---|---|---|---|---|
| subject 1 | Yi-1.5-6B-Chat | 01.AI | 3.33 | 9.0 to 11.5 | rank 1 on criterion 1 at both r |
| subject 2 | ChatGLM3-6B | Zhipu | 3.43 | 8.8 to 11.2 | rank 2; research licence fits criterion 3 (not shippable in the Play build) |
| subject 3 | DeepSeek-LLM-7B-chat | DeepSeek | 3.76 | 8.1 to 10.4 | fewest multiplied weights after 1 and 2 (6.49 G, Ebony 3.2); ahead of Llama-2 at r 35, behind it at r 19 |
| reserve 1 | Llama-2-7b-chat | Meta | 3.75 | 8.1 to 10.2 | criterion 1 near-tie with subject 3; lower head share |
| reserve 2 | Granite-7b-instruct | IBM | 3.75 | 8.1 to 10.2 | same shape as Llama-2; from-scratch lineage still to confirm |

- Slot 3 is a near-tie on criterion 1 (DeepSeek leads at r 35, Llama-2 at r 19). It is settled by criterion 2: the
  H3 column frontier at `-b 512` and `-ub 1` on both (Tom). Until then the order above stands.
- The choice becomes final, and each served file is frozen (S3, Clyde section 5), only after the H3 and
  Q4_0-vs-F16 screens of all five. These are blocked by the laptop disk (Tom section 4: about 0.2 GB free, swap
  full); about 16 GB free per model is needed.
- None of the five is priced at 12 tok/s with the levers priced so far (Bonnie 3.1, Ebony 3.2, Clyde 5). The set
  is chosen to give 12 its best chance, not because 12 is predicted. Every subject stays a full subject.
- Mistral-7B-Instruct-v0.3 and Qwen2.5-7B-Instruct leave the subject set (ranks 11 and 16). Their memo-02/05
  host results stay as cross-model evidence for levers.

**Owner decision, 2026-11-22:** "ChatGLM3 is excluded. The three models come first." ChatGLM3-6B leaves the pool
(stock PPL 33.0 unexplained, research-only licence). Lead's consequence: the provisional table above is superseded.
- Subject 1: **Yi-1.5-6B-Chat** (rank 1 with measured s 0.40 to 0.42; firm).
- Subjects 2 and 3: chosen by criterion 1 from Mistral-7B-Instruct-v0.3 (s 0.40), Qwen2.5-7B-Instruct (s 0.35),
  Llama-2-7b-chat and DeepSeek-LLM-7B-chat, re-priced with the k,v-dense frontiers (memo-07-work/tom-split.md:
  both pass 30 % with attn k and v dense, *host*). Without that rule the order is Mistral, Qwen2.5 (Bonnie,
  memo-07-work/bonnie-reprice.md); Ebony's hypothesis-A price puts Llama-2 close to them. Tom finishes the k,v-dense
  35/40/45 % cells, Bonnie re-prices, and the lead then fixes slots 2 and 3 and two reserves here.

**Final choice, 2026-11-22 (lead; owner delegated).** Inputs: Bonnie memo-07-work/bonnie-reprice.md §7 (*estimated*),
Tom tom-split.md §5 (*host*).

| slot | model | family | rule inside budget (*host*, both settings) |
|---|---|---|---|
| subject 1 | Yi-1.5-6B-Chat | 01.AI | 42 % uniform |
| subject 2 | Mistral-7B-Instruct-v0.3 | Mistral AI | 40 % uniform |
| subject 3 | DeepSeek-LLM-7B-chat | DeepSeek | 40 % with attn k and v dense (33.4 % of weights) |
| reserve 1 | Llama-2-7b-chat | Meta | 40 % with k and v dense, 0.5 point under the line |
| reserve 2 | Qwen2.5-7B-Instruct | Alibaba | 35 % uniform |

How the rule was applied:
- Criterion 1 by Bonnie's strict order is Llama-2, DeepSeek, Mistral, Qwen2.5. The gaps among the first three
  (Llama-2 vs Mistral -0.1 to +0.7 tok/s, Llama-2 vs DeepSeek within 0.3) are smaller than the error of the
  estimates, so the lead reads them as a tie. Section 1 left "tie" undefined; this reading is a lead judgement made
  after the prices were known, recorded as such. Qwen2.5 trails by more and is not in the tie.
- Criterion 2 breaks the tie: Mistral skips the largest share (40 %, +2.44 / +2.92 %), DeepSeek beats Llama-2 at the
  same share with more margin (+2.95 / +3.54 % against +3.97 / +3.95 %).
- Under the strict reading the subjects would be Yi, Llama-2, DeepSeek; that alternative is recorded.
- Side effect, not a criterion: the set holds one GQA 6B, one GQA 7B with sliding-window history and one full-MHA
  7B, so the k,v-dense rule is tested on one subject rather than carried by two.

For the freeze (S3): DeepSeek's screens ran without BOS (`add_bos = false` from the converter); the served file
sets BOS explicitly and its stock and F16 PPL are measured that way. No subject is priced at 12 tok/s with the
priced levers (*estimated*); every subject stays a full subject.

