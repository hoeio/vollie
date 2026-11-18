# Campaign 3 memo-01, Ebony's draft: candidates, 2025-2026 methods, mechanisms, first tests

Agent model: claude-fable-5-2 (Ebony). Date 2026-11-18. Repository HEAD when written: `16a3d98`.

**Nothing here ran on a phone or on the host.** Sources are web pages fetched or searched on 2026-11-18 and the
repository's own documents. Labels: *device* = quoted from a stored device result (named); *host* = quoted from a
host entry; *estimated* = my arithmetic; "source" = a published claim not reproduced here. Web pages were read
through a summarising fetch tool; where only an abstract or a search summary was seen, it says so. Pricing in
bytes and milliseconds is Bonnie's lane: every *estimated* figure below is a first guess for her to replace.
This draft proposes; verdicts, pre-registrations and the model set are the lead's and the owner's.

## 0. Summary

1. **Residency comes before bandwidth.** A dense model reads its whole body every token, so the whole body must
   stay resident. The largest page cache this phone has shown is 2.4 to 3.2 GB (*device*, R58, R67); a 7 B body
   at Q4_0 is about 4.0 GB (*estimated*), and Llama-3.1-8B's Q4_0 file is 4,438 MiB (source). On top of that,
   llama.cpp repacks every Q4_0 matrix into anonymous memory (ggml 4x4 repack, or KleidiAI on the phone build),
   which for a dense Q4_0 file is a second copy of the whole body. **The stock baseline of a dense Q4_0 7 B model
   on this phone may be a thrash number, not a bandwidth number.** The first device probe is a capacity probe.
2. **For stock SiLU models, the published training-free methods that fit a +5 % budget are all input-side**
   (TEAL, DIP, LaRoSA: they drop weight **columns** by activation magnitude, no predictor). Output-side methods
   (CATS, Deja Vu predictors, CMoE's dense-to-MoE conversion) lose by wide margins on the same models. The file's
   quant blocks run along the input, so a dropped column saves no DRAM traffic in the stock layout. **The core
   build of this campaign would be a column-major derived layout with an integer kernel**, and for Q4_0 that
   layout can be exact (§3.1); memo-21 A4 closed input sparsity because it assumed no exact layout exists.
3. **Published tolerance inside +5 % is about 25 to 40 % model-wide sparsity, and it depends on the model:**
   Mistral-7B tolerates the most, Llama-3-8B the least, Qwen2.5-7B is between (§2.2). All of those numbers are on
   FP16 weights, and the PPL protocol of TEAL is easier than ours.
4. **My arithmetic says 40 % column sparsity alone gives about 1.4 times on a 7 B body, not 2 to 3 times**
   (*estimated*, §3.4): the column kernel costs more per touched weight than `sdot`, so the token becomes
   compute-bound. On a true 7 B body I do not see 15 tok/s from any published method. I say so now, with the
   numbers, for Bonnie to check.
5. **Candidates.** Recommended pair from different families: **Mistral-7B-Instruct-v0.3** (best published
   sparsity tolerance, smallest head) and **Qwen2.5-7B-Instruct** (28 layers, 87 % of the body is FFN, published
   tolerance data). Two questions for the owner change the picture more than any lever: whether **Gemma 4 E4B**
   (8 B with embeddings, 4.5 B effective, dense) is inside the definition, and whether a natively sparse
   non-transformer (**RWKV-7 G1 7.2B**, ReLU² FFN) may be a subject. No ReLU-family dense transformer of 6.0 to
   8.5 B with stock instruct weights and current llama.cpp support was found.
6. Speculative decoding stays closed: the regime where a second column is nearly free (dense and byte-bound)
   is the regime the sparsity lever leaves (§3.5). One probe column settles it for free.

## 1. Candidate models

Architecture rows marked "config" were read from the publisher's `config.json` on Hugging Face on 2026-11-18;
"recalled" rows are from memory and must be checked against the GGUF header before a card is written (host,
free). "Body" = weights read per token = all layers plus the output head, input embedding excluded;
bytes at Q4_0 = body × 4.5 bits (*estimated*). FFN share is of the layers' weights (*estimated* from the
dimensions).

| model | params | layers, d, FFN | FFN act. | attention | vocab, head | licence | body, GB/token at Q4_0 | basis |
|---|---|---|---|---|---|---|---|---|
| Mistral-7B-Instruct-v0.3 | 7.25 B | 32, 4096, 14336 (SwiGLU) | SiLU | GQA 32 / 8 | 32,768; untied; head 0.13 B | Apache 2.0 | 7.11 B, 4.0 | config values confirmed through mirrors (search summary); official repo is gated |
| Qwen2.5-7B-Instruct | 7.62 B | 28, 3584, 18944 | SiLU | GQA 28 / 4 | 152,064; untied; head 0.55 B | Apache 2.0 | 7.07 B, 4.0; FFN 87 % of layers | config |
| Qwen3-8B | 8.19 B | 36, 4096, 12288 | SiLU | GQA 32 / 8 | 151,936; untied; head 0.62 B | Apache 2.0 | 7.57 B, 4.3 | config |
| Llama-3.1-8B-Instruct | 8.03 B | 32, 4096, 14336 | SiLU | GQA 32 / 8 | 128,256; untied; head 0.53 B | Llama 3.1 community licence | 7.50 B, 4.2 | recalled; file sizes from arXiv 2601.14277 |
| Yi-1.5-6B-Chat | 6.06 B | 32, 4096, 11008 | SiLU | GQA 32 / 4 | 64,000; untied; head 0.26 B | Apache 2.0 | 5.80 B, 3.3 | config; 2024-05 model, little known |
| Granite 4.1 8B | "8B" (one listing says 8.8 B: check) | 40, 4096, 12800 | SiLU | GQA 32 / 8 | tied embeddings | Apache 2.0 | about 4.6 if 8.2 B | IBM blog through a search summary; released 2026-04-29; GGUF not confirmed here |
| Ministral 3 8B Instruct 2512 | 8.4 B language + 0.4 B vision | recalled 36 layers | SiLU | GQA | 131 k | Apache 2.0 by one third-party listing only | about 4.6 | search summary; 8.4 to 8.9 B total sits at or over the 8.5 B limit |
| Gemma 4 E4B-it | "4.5 B effective, about 8 B with embeddings" | not read | not read (Gemma family: gated GELU, recalled) | not read | 262 k (recalled) | Apache 2.0 (model cards, search summary) | about 2.5 if 4.5 B is all read (*estimated*) | search summary of the model card; llama.cpp lists `gemma4` |
| RWKV-7 G1 7.2B | 7.20 B | 32, 4096, FFN 16384 (two matrices, no gate) | **ReLU²** (paper pseudocode: `W_v @ relu(k).square()`) | none (recurrent) | 65,536 | Apache 2.0 | about 4.0 | HF card of the 2026-08-05 checkpoint; arXiv 2503.14456 through a search summary; **not a transformer** |

Published GGUF quants (search results, 2026-11-18): Ministral 3 8B (Mistral's own collection and unsloth),
Gemma 4 (unsloth), RWKV-7 G1 7.2B (a community Ollama repository at Q4_K_M, 4.27 GiB; llama.cpp merged RWKV-7 in
2025-03). For Mistral 7B v0.3, Qwen2.5-7B, Qwen3-8B, Llama 3.1 8B and Yi-1.5-6B I did not fetch a GGUF page
today; from memory the usual community ladders (Q2_K to Q8_0, IQ quants, imatrix variants) exist, and Qwen
publishes its own GGUF. **Not verified today; and since the served quantisation is ours to choose, the campaign
can quantise from the publisher's safetensors as campaign 2 did.**

llama.cpp support: the architecture table of `src/llama-arch.cpp` on master (read through the fetch tool,
2026-11-18) contains `llama`, `qwen2`, `qwen3`, `gemma3n`, `gemma4`, `granite`, `rwkv7`, `mistral3`, `exaone4`.
Whether each exists at the campaign base `620d2225c` is a desk check for Tom.

**Activation sparsity, as known:**

- Every in-definition candidate has a SiLU-gated (or GELU-gated) FFN: **no natural zeros**. What is published is
  magnitude compressibility: PowerInfer-2 cites "about 50 %" for SiLU models (after CATS and CHESS), TEAL reaches
  40 to 50 % on Llama and Mistral (§2.2).
- **No ReLU-family model fits the definition today.** Persimmon-8B (squared ReLU by recall; I could not fetch
  Adept's page, HTTP 403) is 9.3 B by recall and `persimmon` is no longer in llama.cpp's architecture table. OPT-6.7B
  (ReLU) is not an instruct model and has no llama.cpp support. The ReLU-fied 7 B models (ProSparse-Llama-2-7B,
  TurboSparse-Mistral-7B, Bamboo-7B, SparseQwen2-7B) are fine-tuned derivatives, excluded by the definition, and
  ship for PowerInfer's own format.
- RWKV-7's FFN is ReLU², so its second FFN matrix (one third of the model, *estimated* from 8 d² of 12 d² per
  layer) sees exact zeros. **The zero fraction is not published** (search summary); it is a ten-minute host count.

**Recommendation.**

1. **Mistral-7B-Instruct-v0.3** (Mistral family). Best published tolerance to input sparsity in three independent
   papers (§2.2; the papers used Mistral-7B without naming v0.3, so this must be re-measured), the smallest head
   (0.13 B: the head is 2 % of the token), Apache 2.0, plain `llama` architecture.
2. **Qwen2.5-7B-Instruct** (Qwen family). 28 layers (fewest sync points), 87 % of the layer weights are FFN, a
   0.55 B head, published tolerance (TEAL 40 %: +5.1 %; LaRoSA table).
   Qwen3-8B is the newer alternative; no sparsity data was found for it and it has 36 layers.
3. Reserve: **Llama-3.1-8B-Instruct** as the "model people know"; it is the least tolerant in every table, so it
   is the honest hard case. **Yi-1.5-6B-Chat** is the smallest body inside the definition (3.3 GB at Q4_0) if
   the capacity probe says nothing larger can stay resident.
4. **Two questions for the owner, through the lead:**
   - **Gemma 4 E4B.** It meets the letter of the definition (dense, about 8 B in total with embeddings, stock
     instruct weights, llama.cpp support) and reads only its 4.5 B "effective" part per token. It is the one
     candidate where my arithmetic comes near 15 tok/s (§3.4). Whether a model whose extra 3.5 B are lookup tables
     is "dense 7B-class" in the owner's sense is not mine to decide. Campaign 2 counted class by total parameters.
     "Reaches" was too strong in my first version: see the corrected row in §3.4 (12 to 14 at my inputs).
   - **RWKV-7 G1 7.2B.** Outside the definition (not a transformer; the card calls the checkpoint a base model
     with a chat template). It is the only current, llama.cpp-supported 7 B model with a ReLU-family FFN. Worth a
     subject only if the host count shows most of the ReLU² outputs are zero.

## 2. Methods, 2025-2026

### 2.1 What exists, and phone numbers where they exist

| work | date | idea | trained? | measured on a phone | transfers here? |
|---|---|---|---|---|---|
| PowerInfer-2 ([arXiv 2406.06282v3](https://arxiv.org/html/2406.06282v3)) | 2024-06, v3 later | hot neurons dense on the NPU, cold neurons sparse on CPU cores, neuron bundles in flash, predictor | needs ReLU-fied models (Bamboo, TurboSparse) | OnePlus 12 (24 GB), CPU + NPU: Bamboo-7B 11.1 tok/s at 50 % FFN offload; **Mistral (SiLU)-7B 5.3 tok/s** at 50 % offload; in-memory Bamboo-7B 2.24 × llama.cpp. Memory rate of that phone: CPU 43.9 GB/s, CPU + NPU 59.6 GB/s | No. Hybrid NPU path, derivative models. Its SiLU row (5.3 tok/s on a phone two generations newer) is the nearest published number to our case |
| LLM in a Flash (Apple, 2023) | – | row-column bundles, sliding window, low-rank predictor | ReLU models | none (simulated / laptop) | idea of bundling only |
| ActiveFlow ([arXiv 2504.08378v2](https://arxiv.org/html/2504.08378v2)) | 2025-04 | top-K input sparsity, cross-layer preloading, hot-weight cache, DRAM-flash swap; **built on llama.cpp's CPU backend, 4-bit** | sparsity is predictor-free, but its accuracy uses self-distillation (10 h on 4 × A100): a fine-tune | Pixel 6 (X1 + A76 + A55, 8 GB: close to our phone): LLaMA-2-7B at full-weight speed with 40 % less memory, 1.9 × at its "75 %" setting (absolute tok/s not in the text read); OnePlus 12: 5.9 tok/s at 75 % | The mechanism yes (big cores compute, little cores load). The accuracy no: without distillation it is TEAL's curve. Llama-3-8B with distillation: 6.09 → 6.97 at 50 % (+14 %) |
| DIP and DIP-CA, Qualcomm ([arXiv 2412.01380v2](https://arxiv.org/html/2412.01380v2)) | 2024-12 | top-K on \|x\| for `up` / `gate` columns, top-K on \|GLU\| for `down` columns; **cache-aware mask**: a column already in DRAM gets its score kept, others × γ = 0.2 | no (LoRA optional) | none: simulated (DRAM 60 GB/s, flash 1 GB/s); Llama-3-8B 0.57 → 0.89 tok/s at +0.5 PPL | Yes, both parts. The cache-aware mask is a residency lever that is new to this repository |
| TEAL ([arXiv 2408.14690v3](https://arxiv.org/html/2408.14690v3)) | ICLR 2025 | per-tensor magnitude thresholds on the input of all seven matrices | no | none (GPU: 1.42 to 1.53 × at 40 %) | Yes as a rule; kernel must be ours |
| LaRoSA ([arXiv 2507.01299v2](https://arxiv.org/html/2507.01299v2)) | ICML 2025, v2 2026-01 | rotate activations by a per-layer PCA basis, then top-K | no gradient, but **rotations are merged into the weights** and adapters `Q_l^T Q_{l+1}` run between layers | none (A100: 1.30 to 1.40 × at 50 %; 0.88 to 0.92 × at 0 %) | Not as published: merging means dequantise, rotate, requantise (a second quantisation of every tensor), and the adapters add two d × d products per layer, about 1 GB per token at 8 bits (*estimated*) |
| SpQt, "Enabling Dynamic Sparsity in Quantized LLM Inference" ([arXiv 2511.04477v1](https://arxiv.org/html/2511.04477v1)) | 2025-11 | **our exact problem**: Q4_K superblocks are row-wise, sparse columns scatter over them; a "zigzag" layout groups weights along columns; TEAL thresholds; dense kernels in prefill, sparse in decode | no | Apple GPUs only (A19 Pro 1.55 ×, M2 1.46 ×); sparse GEMV 1.22 / 1.51 / 1.78 × at 25 / 40 / 50 % | The finding yes (naive skipping on the stock layout: under 0.2 ×; a column layout is required). The code no (8 K lines of Metal on b5711). **Its PPL is on a quantised model**: Llama-2-7B Q4_K 5.28 → 5.27 (25 %), 5.65 (50 %, +7.0 %), 7.39 (65 %) |
| SVD contextual-sparsity predictors ([arXiv 2603.14110v1](https://arxiv.org/html/2603.14110v1)) | 2026-03 | training-free low-rank predictor of the gate sign, rank about 2 % of the FFN width | predictor no; **models are ReLU-fied** | none (RTX 3090, 1.6 to 1.9 × end to end) | Only for a ReLU-family model. Cost: r (d + D) weights per FFN |
| CMoE ([arXiv 2502.04416v2](https://arxiv.org/html/2502.04416v2)) | 2025-02 | dense → MoE without training: shared experts from always-on neurons, routed experts by balanced clustering, router from representative neurons | no (LoRA optional) | none | **No: training-free PPL is far outside any budget.** Llama-3-8B 6.14 → 10.16 with 75 % of the FFN active, 143 with 25 %; Llama-2-7B 5.27 → 7.02 and 62.3 |
| MLPMoE (arXiv 2511.21089) | 2025-11 | static slicing of the MLP into experts | no | – | search summary only; not read |
| CoreInfer, SelectInfer ([arXiv 2607.18081](https://arxiv.org/abs/2607.18081)) | 2024-10; 2026-07 | fix the neuron set per prompt at the prefill; profile task neurons offline | no | SelectInfer: "edge devices", **abstract only**, no numbers seen | Idea worth a host count (§4, H5): a set fixed per prompt has no per-token predictor and is good for residency |
| NimbleEdge sparse_transformers ([GitHub](https://github.com/nimbleedge/sparse_transformers)) | 2025-06 | Deja Vu-style predictor, CPU kernels in Torch | predictor trained | laptop CPU only: Llama-3.2-3B 0.7 → 1.3 tok/s at about 20 % sparsity, fp16 | No. GGUF "coming soon" (maintainer reply, search summary) |
| Sakana TwELL ([page](https://sakana.ai/twell/)) | 2026-05 | sparse packing format for trained-sparse LLMs, GPU | yes | none | No |
| llama.cpp itself | – | no pull request for activation sparsity or sparse FFN found (one search of weekly digests; GitHub not searched directly) | | | nothing to carry over |

MNN, ExecuTorch and mllm were not searched this round (time); none of the sparsity papers above builds on them.

### 2.2 Perplexity cost of training-free sparsity on stock 7 B models (source; FP16 unless noted)

Percent over each paper's own dense value (*estimated* from the published PPLs). Sparsity is model-wide for TEAL
and LaRoSA, FFN-only for DIP.

| model | TEAL, own table (25 / 40 / 50 %) | TEAL as re-run by LaRoSA (25 / 40 / 50 %) | LaRoSA (25 / 40 / 50 %) | DIP, FFN only (40 / 50 %) | output-side at 50 %: CATS / Deja Vu (DIP's table) |
|---|---|---|---|---|---|
| Llama-3-8B | +1.2 / +5.8 / +13.6 | +3.9 / +11.4 / +23.3 | +1.6 / +7.7 / +17.8 | +8.5 / +20.2 | +360 / +54 |
| Mistral-7B | +1.8 / +4.3 / +7.9 | +4.1 / +12.6 / +35.0 | +0.6 / +2.4 / +5.8 | +2.7 / +6.9 | +1,052 / +18.9 |
| Qwen2.5-7B | – | +1.2 / +5.1 / +14.0 | +0.7 / +3.6 / +8.3 | – | – |
| Llama-2-7B | +0.4 / +3.0 / +7.1 | +15.4 / +17.0 / +24.3 | +0.7 / +3.1 / +7.3 | – | – |
| Llama-2-7B **Q4_K** (SpQt) | about 0 at 25 %, +7.0 at 50 % | | | | |

Readings:

- **The two TEAL columns disagree by a factor of two or more** (protocol: TEAL's own evaluation sparsifies only the
  second half of each context and warns that sparsifying the first tokens "causes much more degradation"). Our
  gate also scores the second half of each 512-token chunk, but a runtime rule must also decide what it does at
  the first positions. **A position rule (first N tokens dense) is part of the lever**, behaves the same at
  `-b 512` and `-ub 1`, and costs nothing in the 32-token decode metric.
- Inside +5 %, plain thresholds give about 25 to 40 % (model-dependent); rotation would give 40 to 48 % but does
  not transfer (§2.1).
- **Output-side methods are an order of magnitude worse on stock SiLU models.** So the per-token predictor
  question mostly disappears: the methods that fit need no predictor. It returns only for a ReLU-family model.
- TEAL reports that sparsity and quantisation errors "compound somewhat independently" (plots only); SpQt's Q4_K
  row is the only quantised table found. Our host screen (H3) is on the served quantisation for that reason.

## 3. Mechanisms on this phone

### 3.1 Do skipped rows or columns save DRAM traffic?

| what is skipped | stock ggml Q4_0 (blocks of 32 along the input) | ggml 4x4 repack and KleidiAI (4 rows interleaved, blocks along the input) | Q3_K / Q4_K (superblocks of 256 along the input) | a column-major derived layout |
|---|---|---|---|---|
| a whole output row (an FFN neuron in `gate` / `up`; an output channel) | saved: a row is contiguous | saved only in groups of 4 rows, unless the repack sorts rows so that cold rows share groups (the repack is ours to write) | saved: rows are contiguous | not saved |
| an input column (TEAL, DIP; an FFN neuron in `down`) | **not saved**: the column's nibbles are spread over every row's block; skipping needs a whole 32-column block to be zero, and then skips one 18-byte piece per row inside a stream the prefetcher reads anyway | not saved, same reason (one 72-byte piece per 4 rows) | not saved; campaign 1 measured whole-block sparsity of 0.7 to 12 % (*host*, R29) and SpQt measured under 0.2 × for naive skipping (source) | **saved**: a column is contiguous |

So the methods that fit the budget need the layout the file does not have. Three points for a derived layout:

1. **For Q4_0 an exact column-major layout exists** (my design, *estimated*, not built). Keep the 4-bit integers
   as they are, stored column by column, and keep the fp16 scales untouched as a per-(row, 32-column block) table.
   For each block of 32 columns: for every active column j, add `xq_j × q[:, j]` into an int16 accumulator per row
   (127 × 8 × 32 = 32,512 fits); at the end of the block, `y_r += float(acc_r) × d[r, block] × dx[block]`. The
   integer sums are those of the stock Q4_0 × Q8_0 kernel with the zero terms dropped, and the float steps can be
   kept in the stock order. So at threshold 0 it should be bit-identical to the non-KleidiAI kernel, and a
   threshold is then a declared numerics change. This is why I reopen memo-21 A4 ("an input-major copy of a
   quantised tensor is a requantisation, not exact"): for Q4_0 it need not be. For K-quants (6-bit sub-scales,
   `hmask`) the same idea is heavier and unpriced.
2. **Scales are always read**: 16 bits per 32 weights, 11 % of Q4_0's bytes. At density ρ the bytes read are
   about 0.89 ρ + 0.11 of the dense bytes (*estimated*): 0.64 at 40 % sparsity, 0.56 at 50 %.
3. **Campaign 1's negative (R29: an fp32 axpy over kept channels ran +2 to +5 ms slower than the byte-rate GEMV)
   was an fp32 copy, eight times the bytes of a nibble column.** It does not price an int4 column kernel. But the
   direction of the warning stands: per touched weight the column kernel is more work than `sdot`. My count is
   about 12 NEON ops per 32 weights (unpack, offset, two widening multiply-adds, twice) against about 7 for
   Q4_0x4, so roughly 1.7 to 2 times per touched weight (*estimated*; H6 measures it).

Exact zeros: the stock path quantises activations to int8 per block of 32, so a value below 1 / 254 of its block
maximum is already an exact zero and can be skipped with no numerics change. For a Gaussian-like block that is
under 1 % of entries (*estimated*); R29's whole-block count supports "small". It is a counter in H3, and it is
the whole lever for a ReLU-family model.

### 3.2 Prediction overhead per token (*estimated*)

| method | work per token | cost |
|---|---|---|
| magnitude threshold on inputs (TEAL) | one compare per activation: about 39 k per layer for a Llama-shaped block, 1.2 M per token; plus index lists | under 0.5 ms with NEON compaction; a scalar index loop could reach 1 to 2 ms (worth writing well) |
| top-K (DIP) | a selection per tensor | more than a threshold, same numerics idea; thresholds calibrated to a target K are the cheap form |
| cache-aware mask (DIP-CA) | one bit per column (resident or not) | the bit must come from our own bookkeeping, not from `mincore` per token |
| low-rank predictor (Deja Vu, SVDP), rank 2 % | r (d + D) weights per FFN: Mistral 352 × 18,432 = 6.5 M per layer, 208 M per token | 0.2 GB at 8 bits, resident and read every token: about 8 to 10 ms at 21 GB/s, 5 % of the stock body's bytes. Not worth it on stock SiLU models (§2.2) |
| sign-bit predictor (SparseInfer-style; recalled, not searched today) | 1 bit per weight of the first FFN matrix | 22 % of that matrix's Q4_0 bytes; only for a ReLU-family model |
| neuron set fixed at the prefill (CoreInfer) | none per token | free; PPL unknown, and it behaves differently at `-b 512` and `-ub 1` by construction, so it must pass both |

### 3.3 Interaction with campaign 2's residency, pin and touch work

- **Everything a dense token reads must be resident, every token.** Campaign 2's working sets were 2.0 to 2.7 GB
  and already lost pages in gate waits (*device*, R60 to R67). A Q4_0 7 B body is 4.0 GB. Per-token magnitude
  sparsity does not shrink the resident set by itself: the union of active columns over a few tokens is
  everything, unless the activity is skewed (H5 counts it) or the mask is biased to resident columns (DIP-CA).
- **The repack copy.** On this phone build KleidiAI takes Q4_0 matrices into its own anonymous buffer (found on
  OLMoE's head, STATUS 2026-11-15), and without KleidiAI ggml's 4x4 repack does the same. For OLMoE that was
  about 225 MB; **for a dense Q4_0 file it is the whole body, about 4 GB of anonymous memory**, with
  `RLIMIT_MEMLOCK` at 64 MiB (*device*, r22) and zram behind it. A K-quant file is read from the mapping (no ARM
  repack that I know of at the base; Tom to confirm), at a slower kernel. Either way the first exact lever is
  "the layout the kernel wants lives in a file": a runtime-derived artefact in the app's internal storage,
  mapped `PROT_EXEC` (R67: accepted on internal f2fs, residency criterion met on both models). It costs storage
  (one more body) and the owner ruled such artefacts are runtime (R42 amendment 2).
- **Pin, pool, touch, nice** (R67, R68) are properties of the phone and carry over unchanged. A dense token has
  far fewer, far larger ops than a MoE token (about 9 matrix products per layer, each tens to hundreds of MB),
  so sync and dispatch pools shrink in share and the X1's unequal share (c2 Y4's later arm) grows in value.
- **Heat.** A dense token keeps three or four big cores and the memory bus saturated for the whole token; the
  32-token run lasts 2 to 6 s instead of 1.5 s. Clock caps under sustained load are unmeasured for that shape.

### 3.4 Where this leaves the arithmetic (*estimated*; for Bonnie to replace)

Inputs: Q4_0x4 51 ps per weight on an A78 at 2,112 MHz (*device*, campaign 1 bench), scaled by 0.85 for the
2,419 / 2,841 MHz seen with power saving off (*device* clocks, R67; the scaling is an assumption); dense read
rate 21 to 27 GB/s (*device*, 19 to 22.8 in campaign 1, 25 to 27 implied by OLMoE's head op); column kernel at
1.85 × `sdot` per touched weight; three threads; everything resident.

| case | bytes per token | byte time | compute time | tok/s |
|---|---|---|---|---|
| Mistral-7B Q4_0, stock | 4.0 GB | 148 to 190 ms | about 103 ms | 5.3 to 6.8 |
| the same, 40 % column sparsity | 2.56 GB | 95 to 122 ms | about 114 ms | 8.2 to 8.8 |
| Yi-1.5-6B Q4_0, 40 % | 2.1 GB | 78 to 100 ms | about 93 ms | 10.0 to 10.8 |
| Gemma 4 E4B Q4_0, stock (if 4.5 B is read) | 2.5 GB | 93 to 119 ms | about 65 ms | 8.4 to 10.8 |
| the same, 40 % on the layers, head dense (0.67 B if 262 k × 2,560: recalled, unverified) | about 1.76 GB | 65 to 84 ms | about 71 ms | 12 to 14 |

**These rows are ceilings, not predictions.** The one dense device number in the repository is below its own
byte arithmetic: Qwen3.5-4B Q4_0, 2.23 GB per token, printed 5.9 to 6.6 tok/s on this phone with the stock
runtime (*device*, 2026-10-25, `reports/02_methods.md`), an effective 13 to 15 GB/s. Scaled on bytes, the stock
rows above would be 3.3 to 3.7 (Mistral) and 5.3 to 5.9 (Gemma 4 E4B). How much of that gap campaign 1's runtime
and power saving off recover on a dense graph is what P2 and the first baseline round measure.

15 tok/s is 66.7 ms per token. On a 7 B body the bytes must fall to about 1.4 to 1.8 GB (35 to 45 % of Q4_0's)
**and** the compute must fit, which with the column kernel means touching about a third of the weights. No
published training-free method gives 65 % model-wide sparsity inside +5 % on a stock SiLU model (TEAL at 65 %:
+27 to +54 %). A lower-bit file moves bytes but not compute: the Q3_K kernel is 102 ps per weight (*device*),
twice Q4_0x4, and Llama-3.1-8B's Q3_K_S is already at F16 × 1.224 (source: 8.96 against 7.32, quantisation
recipe not stated), over the hard ceiling; Q3_K_M is × 1.087 at 3.8 GiB.

### 3.5 Speculation: why it stays closed

R20 closed speculation because the CPU was compute-bound above batch one. A dense Q4_0 model is byte-bound at
batch one, so a second column should cost less than a second token: that is new. But by the numbers above the
compute of one column is already 0.55 to 0.7 of the byte time, so two columns cost about 1.1 to 1.4 tokens, and
prompt lookup gave 1.20 to 1.27 tokens per evaluation on the official prompt (*device* / *host*, R16). Net about
zero; and any sparsity lever moves the token to compute-bound, where R20 applies again. The probe P2 prints the
two-column cost anyway; a ratio under 1.08 would reopen it for the dense, unsparsified case only.

## 4. Proposals for Tom

Cost: S = under half a day, M = 1 to 2 days, L = more. B = bit-identical (three-setting identity check), N =
declared numerics change inside the budget at `-b 512` and `-ub 1`. All gains *estimated*.

### 4.1 TEST list

| # | test | where, cost | decides | closing result |
|---|---|---|---|---|
| P1 | **Resident-capacity probe (first device probe).** In the kit: map N GB of existing model files (`PROT_EXEC`, internal storage), read them through in a loop for 20 passes at N = 2.5, 3.0, 3.5, 4.0, 4.5; print seconds per pass, major faults, `Cached`, the app's swap. No new download. Power saving off, recorded. | device, S; minutes of phone time | the largest body that can be read every token; whether a Q4_0 7 B file is possible at all | the last N whose passes run at DRAM speed without major faults. Under 3.3 GB → no 6 to 8.5 B body at Q4_0 fits; the served file must be about 3 bits or the model must be Gemma-4-E4B-like, and that goes to the owner with the number |
| P2 | **Dense rate and compute ceiling, power saving off.** `GGML_CPU_BENCH` rows for Q4_0x4, int8 and Q3_K per core class; a 256 MB Q4_0x4 stream with 1, 2, 3, 4 threads (A78 only, A78 + X1); the same with two right-hand columns; clocks and caps before and after; 30 s sustained to see the cap under a dense load. | device, S to M; rides with P1 | CAMPAIGN.md open item 2; the byte and compute inputs of every price; §3.5 | numbers, not a verdict. Two-column cost under 1.08 × reopens speculation for the unsparsified case |
| H1 | **Byte inventory from GGUF headers** for each candidate: bytes read per token by tensor group, head size, tied or not, which tensors the phone build repacks and into what memory. Includes Gemma 4 E4B's real per-token bytes. | host desk, S | §1's "recalled" cells; the repack-copy claim of §3.3 | – |
| H2 | **Served-quantisation ladder** per chosen model: F16, Q4_0, Q4_K_M, Q3_K_M, Q3_K_S, IQ3_XXS, with and without imatrix; rule for choosing written first (as memo-01 part C of campaign 2). | host, a night per model (lead) | the lowest-bit file inside F16 × 1.20; the stock PPL that defines the budget | – |
| H3 | **Input-sparsity tolerance on the served quantisation**, no new kernel: `tools/dense_colskip.py` (campaign 1; it zeroes inputs below a threshold before the stock product, which is the numerics of skipping) generalised from Granite's tensor groups to all seven matrices of a transformer block, or the same as an environment knob. Thresholds from wikitext-2 **train**. Grid: 20 / 30 / 40 / 50 % for all seven matrices, and per group (attention inputs, `gate` + `up`, `down`); first-N-tokens-dense with N = 0, 1, 8; 40 to 60 chunks at `-b 512` and `-ub 1`; `en.txt`. Counters: realised sparsity, exact int8 zeros. | Tom S to M (patch), lead: a night per model | whether the column lever is worth its kernel on each model; the position rule | largest model-wide sparsity inside +5 % with 0.5 points of margin is under 25 % on both models → the column lever is closed for stock SiLU models. Published expectation: Mistral 35 to 40 %, Qwen2.5 about 35 %, Llama-3 25 to 30 % |
| H4 | **ReLU² zero count** on RWKV-7 G1 7.2B (fraction of exact zeros entering the second FFN matrix, per layer), only if the lead wants the question put to the owner. | host, minutes after a download | whether to ask the owner about RWKV | under 70 % zeros → not worth a definition change |
| H5 | **Stability of the active set** (rides on H3's knob): union of active columns over 32 decode tokens; share of active mass covered by a fixed hot half chosen on train text; by a set fixed at the prefill. | host, S | whether sparsity can shrink the resident set (DIP-CA, CoreInfer-style); whether the derived file should be ordered hot-first | union over 32 tokens above 90 % of columns and hot-half coverage under 70 % → sparsity is a bandwidth lever only; residency must come from the file size |
| H6 | **Column kernel microbench**: exact int4 column kernel against Q4_0x4 on a 14336 × 4096 matrix at density 100 / 60 / 50 %; identity of the result at threshold 0 against the non-KleidiAI kernel; host first, then a `GGML_CPU_BENCH` row on the A78. | Tom M; device minutes | §3.1 point 3, and so §3.4 | on the A78, above 2.2 × `sdot` per touched weight → at 40 % sparsity the token is slower than dense; closed unless P2 shows the phone far more byte-bound than assumed |
| H7 | **Sublayer ablation table**: skip one attention or one FFN sublayer at a time, 20 chunks, both models. | host, no code beyond a knob; a night | whether any whole sublayer is nearly free on a 32-layer dense model (R21's +2.5 % was Granite's cheapest layer) | cheapest sublayer above +1.0 % on both → closed |

### 4.2 TRY list

| # | lever | work | gain | label | generic? | needs first |
|---|---|---|---|---|---|---|
| Y1 | **Kernel layout in a file, not in anonymous memory**: the 4x4 (or column-major) layout written once to internal storage and mapped `PROT_EXEC`; no repack copy. | Tom M | removes up to one body of anonymous memory; on this phone probably the difference between running and thrashing | B | yes, any Q4_0 model | H1, P1 |
| Y2 | **Two-stage head**, campaign 1's and campaign 2's patches on a dense model. *Corrected after reading Bonnie's draft: the shipped head screens every row with a Q4_0 copy and saved bytes only because the served head was Q6_K. If the served head is Q4_0 it saves nothing; screens below 4 bits were compute-bound (R29).* | Tom S | 0 for a Q4_0 head; the Q6_K-to-Q4_0 ratio (about −30 % of the head's bytes) only if H2 picks a higher-bit head. The head is 7 to 15 % of the token on the large-vocabulary candidates and has no proven row-skip lever (vocabulary subset and low-rank screen are closed, R43) | N (+0.01 % on Granite, *host*) | rule yes; value depends on the served head type | H1, H2 |
| Y3 | **Exact column-major Q4_0 kernel** (§3.1) for the matrices H3 selects, with the position rule. | Tom L | none by itself (B at threshold 0); it is what makes Y4 pay | B | yes | H6 |
| Y4 | **Input-magnitude thresholds** on Y3, per tensor, calibrated on train text. | Tom S on top of Y3 | 7 B body at 40 %: about 1.3 to 1.5 × over stock (§3.4); less if H3 allows less | N | rule yes; thresholds per model | H3 |
| Y5 | **Cache-aware mask** (DIP-CA) and hot-first file order. | Tom M | residency only: fewer distinct columns per run | N (mask), B (order) | yes | H5, P1 |
| Y6 | **Lower-bit served file** chosen by H2. | lead; no build | bytes down by the bit ratio; compute up if the kernel is Q3_K's | file choice, frozen before the first device round | per model | H2, P1, P2 |
| – | Not ranked, behind a decision or a test: unequal X1 share (c2 Y4), sublayer skip (H7), sign predictor for a ReLU-family model (H4, owner), LaRoSA with one global rotation and no adapters (requantisation cost unknown) | | | | | |

### 4.3 Rules questions for the lead

1. Are calibration-derived side tables (thresholds, a hot-first order) computed on the phone from a fixed text
   "runtime" in the sense of R42 amendment 2? I assume yes when the phone computes them from the served file.
2. Which build defines stock numerics for a dense Q4_0 model: the KleidiAI path (the phone's) or ggml's? The
   exact column kernel can match ggml's, not KleidiAI's (campaign 2 met this with OLMoE's head).
3. The two definition questions of §1 (Gemma 4 E4B, RWKV-7).

## 5. Sources (all read 2026-11-18)

- PowerInfer-2: <https://arxiv.org/html/2406.06282v3>. ActiveFlow: <https://arxiv.org/html/2504.08378v2>.
  DIP: <https://arxiv.org/html/2412.01380v2>. TEAL: <https://arxiv.org/html/2408.14690v3>. LaRoSA:
  <https://arxiv.org/html/2507.01299v2> (first 100 k characters). SpQt: <https://arxiv.org/html/2511.04477v1>.
  SVD predictors: <https://arxiv.org/html/2603.14110v1>. CMoE: <https://arxiv.org/html/2502.04416v2>.
  SelectInfer (abstract only): <https://arxiv.org/abs/2607.18081>. TwELL: <https://sakana.ai/twell/>.
  NimbleEdge (search summary): <https://github.com/nimbleedge/sparse_transformers>.
- Llama-3.1-8B quantisation ladder (F16 7.32, Q4_0 7.74, Q4_K_M 7.56, Q3_K_M 7.96, Q3_K_S 8.96; sizes):
  <https://arxiv.org/html/2601.14277v1>.
- Configs: <https://huggingface.co/Qwen/Qwen3-8B/raw/main/config.json>,
  <https://huggingface.co/Qwen/Qwen2.5-7B-Instruct/raw/main/config.json>,
  <https://huggingface.co/01-ai/Yi-1.5-6B-Chat/raw/main/config.json>; Mistral-7B-Instruct-v0.3 through mirrors
  such as <https://huggingface.co/TitanML/Mistral-7B-Instruct-v0.3-AWQ-4bit/blob/main/config.json> (search summary).
- RWKV-7: <https://huggingface.co/RWKV/RWKV7-7.2B-20260805>, <https://arxiv.org/pdf/2503.14456> (search summary),
  <https://ollama.com/shoumenchougou/rwkv7-g1l:7.2b> (search summary).
- Gemma 4 E4B (search summary): <https://huggingface.co/google/gemma-4-E4b-it>, <https://huggingface.co/unsloth/gemma-4-e4b>.
  Granite 4.1 (search summary): <https://huggingface.co/blog/ibm-granite/granite-4-1>. Ministral 3 (search summary):
  <https://huggingface.co/unsloth/Ministral-3-8B-Instruct-2512-GGUF>.
- llama.cpp architecture table: <https://raw.githubusercontent.com/ggml-org/llama.cpp/master/src/llama-arch.cpp>
  (first 100 k characters, through the fetch tool). Persimmon support added in #3410 (search summary); Adept's
  page returned HTTP 403.
- KleidiAI path for Q4_0 (search summary): <https://learn.arm.com/learning-paths/mobile-graphics-and-gaming/performance_llama_cpp_sme2/kleidiai_integration/>.

## 6. Review of Bonnie's pricing (`bonnie-draft.md`, read 2026-11-18 after §0 to §5 were written)

The two drafts were written independently and agree on the frame: resident bytes is a wall the campaign
arithmetic did not name; the budget buys 25 to 40 % input (column) sparsity and the structured, kernel-friendly
forms cost many times the budget; 15 tok/s on a body of 7.1 B or more is not reachable by anything priced; the
one opening is a model that reads about 4.5 B per token; two device probes decide most of it. Her compute
ceiling (9.8 tok/s for the Mistral shape at S2, three threads) and mine (103 ms) are the same number.

**Where her draft corrected mine** (edited in place above, marked):

1. **Two-stage head.** I priced it as skipping head bytes. It screens every row with a Q4_0 copy; with a Q4_0
   served head it saves nothing. My Gemma row fell from "13 to 16" to "12 to 14" and Y2 is rewritten.
2. **The dense device anchor** (Qwen3.5-4B Q4_0, 2.23 GB per token, 5.9 to 6.6 tok/s, *device*). I had not used
   it. It puts every stock row of my §3.4 about 35 to 45 % too high as a prediction; added as a caveat.
3. `tools/dense_colskip.py` exists; H3 should reuse it (her T4) and not start with a new knob.
4. Her llama.cpp table is the better ceiling source (imatrix stated, IQ quants included): for Llama 3 8B nothing
   under about 3.3 GB passes F16 × 1.20 with room left. My second source agrees where they overlap (Q3_K_S fails
   in both: × 1.220 and × 1.224).

**Where I disagree or would change her pricing:**

1. **"An input-major copy is a requantisation" and "it accumulates in float, 0.94 instructions per weight" are
   not necessary for Q4_0** (§3.1). The 4-bit integers can be stored column-major unchanged, with the fp16 scales
   kept per (row, 32-column block); the kernel accumulates integers per block and does one float step per row
   and block, as the stock kernel does. So C3's "copy re-quantised" budget item can be zero, and the lever can
   be checked for identity at threshold 0. Campaign 1's 0.94 count was for an fp32 or float-accumulating axpy.
2. **Kernel cost per kept weight: her 2.6 × (0.44) against my 1.7 to 2 ×, and neither is benched.** Her 0.44
   kernel interleaves four kept columns in registers to feed `sdot`. Mine multiplies one column's nibbles by one
   int8 activation with widening multiply-adds into int16 row accumulators: about 12 ops per 32 weights as I
   first counted, about 6 to 8 if the nibbles stay unsigned and the −8 offset becomes one correction per block
   (Σ x_j (u − 8) = Σ x_j u − 8 Σ x_j), plus the load and store of the accumulators, which I have not counted and
   which may dominate. Against her 0.17 baseline that is 1.1 to 2.2 ×; against memo-21's 0.22, 0.9 to 1.7 ×. **The
   honest band is 1.2 to 2.6 ×**, and at 60 % kept the token gains about 1.5 × while the kernel costs up to about 1.5 × `sdot`, less and less
   above that, and nothing near 2.4 to 3 × (*estimated*, with dense compute at 0.55 to 0.7 of the byte time). That is
   the whole of C3, so the kernel microbench (her Y2, my H6) should come before the PPL screen is trusted as a
   speed number, and should include the integer-accumulator variant. I would move it from her try list to the
   test list.
3. **Her break-even for 4-neuron groups assumes independent masks** (1 − 0.4⁴ = 97 % of groups kept). If
   neurons are ordered by how often they are active, or grouped by co-activation, groups are far from
   independent; that is what PowerInfer's hot and cold split and DIP-CA's cache bias rely on. Whether stock SiLU
   activations are skewed enough is my H5; it should run in the same pass as her T4's 4-group and 32-block masks,
   with a sorted-order variant, or T4 will measure the worst ordering only.
4. **Residency ceiling: the 3.93 GB kill (R9) was an anonymous repack copy, and the 1.6 to 2.4 GB is from days
   with power saving on and no exec mapping.** File pages are evicted, not killed; R58 and R67 show `Cached` of
   2.4 to 3.2 GB and an exec-mapped model losing far fewer pages in a wait (*device*; R67's own caveat about
   wait lengths stands). So W* for a file-backed, exec-mapped body may sit above her 2.4 and the failure mode is
   a slow pass, not a kill. T1 / P1 should run both arms (file mapping with `PROT_EXEC`, and anonymous) and
   report seconds per pass, since a kill line alone would miss it.
5. **Read rate.** She uses 18.9 to 24.3 GB/s; I used 21 to 27, the top from OLMoE's head op (103 MB in 3.8 to
   4.2 ms, *device* R60, memo-21). Hers is the better-sourced band and the Qwen anchor is lower still (13 to
   15 GB/s effective). The gap between a single op's rate and a whole dense token's rate is itself a pool nobody
   has split (sync, small ops, heat over a 3 to 6 s run); the first baseline round should print it per op.
6. **C2 and C6 lean on "kernel-free skips, about 10 to 13 % of the token inside +5 %" with no PPL source.** The
   only number we own is R21's +2.5 % for the cheapest layer of a 40-layer hybrid. I would not carry C6's "15 at
   33 to 35 GB/s" into the joint memo until the sublayer table (her Y3, my H7) exists; without it C5 needs
   40 GB/s.
7. **Four threads and A55 helpers re-opened: agreed, with one caution from her own constants.** Three threads
   read 18.9 to 24.3 GB/s and four read 22.7 (*device*): the bus looks saturated at three. A fourth core then
   helps only where the token is compute-bound, which is exactly the sparse-column case and the 3-bit case, not
   the stock Q4_0 case. The probe should say which regime each arm is in.
8. **ExpertWeaver (her request to me).** The abstract and the HTML (first 100 k characters) have no perplexity.
   Training-free, 25 % structured sparsity, average accuracy on five tasks: Llama-3-8B 72.8 → 60.4 (CMoE 57.2),
   Qwen2.5-7B 75.2 → 67.0 (source: <https://arxiv.org/html/2602.15521v1>). MMLU falls from 65.3 to 47.0. That is
   better than CMoE and still far outside a +5 % perplexity budget by any reading; I would close structured
   dense-to-MoE conversion for stock weights with CMoE's numbers and this one.
9. Small: her §2.3 says KleidiAI takes Q4_K and Q6_K as well as Q4_0 and Q8_0. Campaign 2 found the Q6_K head
   taken on the phone, so that part is confirmed (*host*, STATUS 2026-11-15); the Arm learning path I read today
   names only Q4_0 and Q8_0 (search summary). Tom's H1 inventory settles what the phone build takes.

**Joint position I would sign:** P(15 tok/s) on a Llama- or Mistral-shaped body below 0.05; on a 4.5 B-per-token
model 0.15 to 0.3, decided by the read-rate probe, the kernel microbench and the owner's reading of the
definition. Order of work: capacity and rate probes on the phone (no model needed), inventory and quantisation
ladder on the host, kernel microbench, then the PPL screen with element, sorted-group and block masks.
