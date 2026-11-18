# Campaign 3 memo-03, Bonnie's draft: Gemma 4 E4B in full, and the strict dense 7B path as a function of the read rate

Agent model: claude-fable-5-2 (Bonnie). Date 2026-11-18. Repository HEAD when written: `6a8b7fb`. Round 2 of the
joint work; my lane is [memo-01](../memo-01-feasibility.md) §7 item 3 (Gemma 4 E4B) and a re-price of the best
strict dense 7B path with memo-01's corrections.

**Nothing here ran on the phone, and no perplexity or speed was measured.** Labels:

- *device*: a stored campaign 1 or 2 phone measurement, named by R-number.
- *host*: read on the laptop today. Two kinds: the llama.cpp tree at the campaign base (`lab/llama.cpp-base`,
  `620d2225c`), and **GGUF headers** (the first 32 MB of each file, fetched by HTTP range request from Hugging
  Face and parsed with a 40-line reader; no weights were downloaded). A header gives exact tensor shapes and
  types, so element counts and bytes are exact.
- *published*: a number from a cited source, measured by others.
- *estimated*: my arithmetic.

Gemma 4 E4B is a **provisional** subject, held for the owner. Verdicts are the lead's.

## 0. Summary

1. **The header inventory replaces "4.5 B read per token, 2.5 GB".** Gemma 4 E4B multiplies **4,644 M weights per
   token** (*host*, headers): 3,303 M FFN, 587 M attention, 671 M output head, 83 M per-layer-embedding gates
   and projection. The 2,819 M per-layer embedding table is a lookup of one row per token and stays on storage.
2. **The published Q4_0 files read 2.83 to 2.99 GB per token, not 2.5** (*host*), because the tied head is
   stored at Q8_0, Q6_K or Q4_K and one file keeps the gates in F32. A campaign mix with every multiplied tensor
   at Q4_0 reads **2.61 GB** (*estimated* from the header shapes).
3. **Band for that mix on three threads** (*estimated*, steady tok/s; printed is 0.95 of it):

   | read rate | 19 GB/s | 24 GB/s | 33 GB/s | 51 GB/s | `sdot` ceiling |
   |---|---|---|---|---|---|
   | power saving off clocks (2,419 / 2,841) | 6.7 to 7.0 | 8.3 to 8.7 | 11.0 to 11.7 | 12.7 to 13.8 | 12.7 to 13.8 |
   | campaign 1 clocks (2,112) | 6.7 to 7.0 | 8.3 to 8.7 | 10.7 to 11.4 | 10.7 to 11.4 | 10.7 to 11.4 |

   **Exact Gemma 4 E4B does not reach 15 tok/s on three threads at any read rate**: the 4-bit `sdot` kernel caps
   it at 12.7 to 13.8. My round-1 row "15.5 at 40 GB/s or more" did not count the head and is withdrawn.
4. **A quality risk that can remove the subject before any speed question** (*published*, §5): on the stock E4B
   base weights a llama.cpp user measured Q4_K_M at 3.24 times the BF16 perplexity and Q5_K_M at 1.17 times;
   community KL figures for stock Q4_K_M files are 0.07 to 0.28. Only Google's QAT checkpoint is reported near
   F16 at four bits (1.03 to 1.08 times, against its own BF16). Whether a QAT checkpoint is "stock weights" is
   an owner question, and it may decide whether Gemma 4 E4B has any 4-bit file inside F16 × 1.20.
5. **Outside check** (*published*): Qualcomm's own model page lists Gemma 4 E4B Q4_0 at 14.9 to 17.2 tok/s on the
   Snapdragon 8 Elite, a chip three generations newer. A Termux llama.cpp report on that chip has 7 to 8 tok/s.
   I lower my chance of 15 tok/s on this subject from 0.2 to 0.3 (round 1) to **0.03 to 0.08** (*estimated*,
   judgement).
6. **Two exact costs found in the base tree that no memo has priced** (*host* source read, ms *estimated*): the
   `per_layer_model_proj` tensor is BF16 in three of four published files and the base has no NEON BF16 dot
   (scalar loop, about 8 ms per token); the final logit soft-cap is a scalar `tanhf` over 262,144 logits (about
   2 to 4 ms).
7. **Carry-over:** pin, exec map, touch (with a tensor-range limit) and kv clear carry over. The two-stage head
   carries over and is worth more here than on any earlier model, but only as a way to keep a 6- or 8-bit head
   at the Q4_0 head's speed. Expert skip has no counterpart.
8. **Gemma 3n E4B is the only other effective-parameter model in the 6 to 8.5 B range** (*host*, header: 6.9 B
   text weights, 4,485 M multiplied per token). It has one thing Gemma 4 lacks: an exact, stock 95 % activation
   cut in its first ten FFN layers. It does not change the verdict (§8), and it is the same family.
9. **Strict dense 7B (§9):** with memo-01's kernel band the best corner is 11.9 to 12.7 tok/s (40 % sparsity,
   kernel at 1.2 ×, power saving off, read rate 35 GB/s or more). At the measured 19 to 24 GB/s it is 6.8 to
   8.8. If the kernel costs 1.85 ×, the path stops at 8.1 to 8.6 from 23 GB/s on, whatever the bus gives. §9 has
   the formula, so that P2's number reads off the result.

## 1. Architecture (*host*: GGUF header of `ggml-org/gemma-4-E4B-it-GGUF` Q4_0 and the publisher's `config.json`, read today)

| item | value |
|---|---|
| architecture string | `gemma4`; `general.size_label` 7.5B (text weights; the published "8 B with embeddings" adds a 150 M vision and a 305 M audio encoder and a 77 M MTP drafter, *published*, technical report Table 1) |
| layers | 42 |
| width, FFN | 2,560 and 10,240 (4 ×); gated FFN, `gelu_pytorch_tanh`, no experts (`enable_moe_block` false) |
| FFN weights | 3 × 2,560 × 10,240 = 78.64 M per layer, **3,303.0 M** in all |
| attention heads | 8 heads, 2 KV heads (grouped) |
| local layers (35) | sliding window 512; head size 256; q and o 2,560 × 2,048, k and v 2,560 × 512 |
| global layers (7: indices 5, 11, 17, 23, 29, 35, 41) | full attention; head size 512; q and o 2,560 × 4,096, k and v 2,560 × 1,024 |
| shared KV | the last 18 layers (24 to 41) reuse earlier keys and values; only layers 0 to 23 compute k and v |
| attention weights used | **587.2 M** (q 256.9, o 256.9, k 36.7, v 36.7) |
| per-layer embeddings (PLE) | table `per_layer_token_embd` 262,144 × 10,752 (42 layers × 256) = **2,818.6 M**, a lookup. Per layer: `inp_gate` 2,560 × 256 and `proj` 256 × 2,560 (55.1 M in all). Once per token: `per_layer_model_proj` 2,560 × 10,752 = 27.5 M |
| vocabulary, head | 262,144; `token_embd` 262,144 × 2,560 = **671.1 M**, tied: the same tensor is the input lookup and the output head |
| logit soft-cap | 30.0 (scale, tanh, scale over all 262,144 logits) |
| positions | RoPE, base 10,000 local and 1,000,000 global |
| context | 131,072 |
| KV size (*estimated* from the shapes, F16) | 20 local layers × 1,024 values + 4 global layers × 2,048 values = 28,672 values = **57.3 kB per position**; 10.7 MB at the benchmark's 187 positions; local layers never hold more than the window |

How a PLE enters a layer (*host*, `src/models/gemma4.cpp` at the base): after the FFN residual, the hidden
state goes through `inp_gate` (2,560 → 256), GELU, an element-wise product with the layer's 256-wide slice of
the looked-up row, `proj` (256 → 2,560), a norm and a residual add. The looked-up row is first added to
`per_layer_model_proj` × (token embedding). So per token the table costs one row read (10,752 values, 6.0 kB at
Q4_0) and the PLE path costs 83 M multiplied weights (1.8 % of the token).

The technical report's totals agree: embedder 670 M + 2,820 M, "einsums" 3,940 M (*published*, arXiv 2607.02770
Table 1); my header sum for FFN + attention + PLE gates is 3,945 M.

## 2. llama.cpp support: the campaign base against today

| item | base `620d2225c` (0.7.0, 2026-10-15) | master today (build b12503, 2026-11-18) |
|---|---|---|
| `gemma4` architecture, E4B (42 layers) | **present** (*host*: `src/models/gemma4.cpp`, 534 lines; E2B, E4B, 26B-A4B and 31B share it) | present |
| PLE table read lazily from the mapping, rows prefetched | present: `TENSOR_READ_LAZY` ("requires mmap for now") and `llama_prefetch_rows` on `per_layer_tok_embd` (*host*) | present |
| shared-KV layers, sliding-window KV | present (*host*) | present |
| tied head | the output is created first as a duplicate of `token_embd` (*host*) | PR #31394 (merged 2026-10-30) creates `token_embd` first. It removes a second 216 MiB copy on E2B **only when mmap is off**; with mmap nothing changes (*published*, PR text) |
| prefill | every layer runs every row | PR #31356 (merged 2026-11-03): rows are dropped before the shared-KV layers. E4B prompt processing 1.6 × on an M4 Pro (805.9 → 1,318.6 tok/s); **decode unchanged** (19.73 → 19.53 tok/s on a Snapdragon HTP) (*published*, PR text) |
| MTP drafter (`gemma4-assistant`) | architecture present in the tree (*host*); PR #23398 said E2B and E4B were not covered at that time (*published*, search summary) | ggml-org now ships `mtp-gemma-4-E4B-it` GGUFs (60 to 172 MB) (*host*, repository listing) |
| BF16 mat-mul on arm64 | **scalar**: `ggml_vec_dot_bf16` has AVX, RISC-V and POWER paths and no NEON path (*host*) | not checked |
| logit soft-cap | `ggml_tanh` = a scalar `tanhf` loop over the vocabulary (*host*) | not checked |
| repack kernels with dotprod | Q4_0 4x4, Q4_K, Q5_K, Q6_K, Q8_0 4x4, IQ4_NL (*host*, `repack.cpp`); none for Q3_K or the IQ2/IQ3 types | not checked |

Reading: **the base runs Gemma 4 E4B as it stands; no rebase is needed for decode speed.** The one upstream
change that matters to a user is #31356, and it changes time to first token, which the campaign's metric does
not count. Two base facts need Tom's check on the phone build: where the tied head lands when the repack buffer
type is on (the duplicate rule is per buffer type, so I expect a repacked anonymous copy of the head beside the
mapped tensor), and whether KleidiAI takes it (it took OLMoE's Q6_K head, campaign 2 R60).

## 3. Bytes read per token by tensor group (*host*: headers; MB = 10⁶ bytes)

| group | weights (M) | ggml-org Q4_0 | unsloth Q4_0 | bartowski Q4_0 | Google QAT Q4_0 | unsloth Q4_K_M | unsloth Q3_K_M | unsloth UD-IQ3_XXS | campaign mix, all Q4_0 (*estimated*) |
|---|---|---|---|---|---|---|---|---|---|
| FFN gate, up, down | 3,303.0 | 1,857.9 | 1,866.1 | 1,866.1 | 1,857.9 | 1,999.9 | 1,572.0 | 1,283.9 | 1,857.9 |
| attention q, k, v, o | 587.2 | 330.3 | 330.3 | 360.0 | 330.3 | 334.0 | 291.6 | 233.6 | 330.3 |
| head = `token_embd` | 671.1 | 713.0 (Q8_0) | 377.5 (Q4_K) | 550.5 (Q6_K) | 550.5 (Q6_K) | 377.5 (Q4_K) | 288.4 (Q3_K) | 288.4 (Q3_K) | 377.5 |
| PLE `inp_gate` + `proj` | 55.1 | 31.0 | 220.2 (F32) | 31.0 | 31.0 | 220.2 (F32) | 220.2 (F32) | 220.2 (F32) | 31.0 |
| `per_layer_model_proj` | 27.5 | 55.1 (BF16) | 55.1 (BF16) | 55.1 (BF16) | 55.1 (F16) | 55.1 (BF16) | 55.1 (BF16) | 55.1 (BF16) | 15.5 |
| norms and scales (F32) | 0.6 | 2.2 | 2.3 | 2.3 | 2.2 | 2.3 | 2.3 | 2.3 | 2.2 |
| **read per token** | **4,644.4** | **2,989.5** | **2,851.4** | **2,865.0** | **2,827.0** | **2,988.9** | **2,429.6** | **2,083.3** | **2,614.7** |
| PLE table (one row per token) | 2,818.6 | 1,585.4 (Q4_0) | 1,937.8 (Q5_K) | 2,312.1 (Q6_K) | 2,312.1 (Q6_K) | 1,937.8 | 1,585.4 (Q4_K) | 1,585.4 | 1,585.4 |
| k, v of the 18 sharing layers (in the file, never used) | 55.1 | absent | 31.0 | 31.0 | absent | 34.7 | 27.3 | 31.0 | absent |
| file size | | 4,591 | 4,836 | 5,224 | 5,155 | 4,977 | 4,058 | 3,717 | about 4,215 |

What stays on storage or is looked up:

- **The PLE table: 2,818.6 M weights, 37.8 % of the text model, 1.59 to 2.31 GB of the file.** Per token one row
  is read: 6.0 kB at Q4_0, 8.8 kB at Q6_K, two or three 4 kB pages. Over the benchmark's 187 positions that is
  under 1.5 MB with read-ahead off and about 24 MB with the default 128 kB read-ahead (*estimated*).
- The input lookup: one 2,560-value row of `token_embd` per token.
- The unused k and v tensors in two of the publishers' files.
- **Everything else is multiplied every token**: 4,644 M weights. The file is 35 to 44 % cold; the model is not
  "4.5 B body plus a head", it is 3.97 B of layers plus a 0.67 B head.

Three things in this table that the round-1 prices did not have:

1. **The head is 14.4 % of the multiplied weights** and 13 to 24 % of the bytes read. In the ggml-org file it is
   Q8_0: 713 MB per token, which is 30 to 38 ms at 19 to 24 GB/s (*estimated*).
2. **The serving type of two small tensors moves the token by milliseconds.** unsloth's F32 gates add 189 MB
   (8 to 10 ms at 19 to 24 GB/s, *estimated*). A BF16 `per_layer_model_proj` runs in the scalar loop at the base:
   27.5 M elements at about one element per two core cycles on three threads is **7.6 to 8.7 ms per token**
   (*estimated*; unbenched). Google's file has it in F16, which has a NEON path. A mix that stores it as Q4_0 or
   Q8_0 pays under 1 ms.
3. **The 3-bit files do not reach the bytes their name suggests** (2.08 to 2.43 GB), and none of their layer
   types has a byte-rate kernel on this CPU (memo-01 §2).

## 4. Residency on the 8 GB phone (*estimated* from *device* facts of campaigns 1 and 2)

| quantity | value | against what the phone has shown |
|---|---|---|
| file on internal storage (needed for the exec mapping, R67) | 4.2 to 5.2 GB | storage only |
| bytes that must be resident every token, campaign mix | **2,615 MB** | the largest model set held resident is 2,672 MB (*device*, R58, OLMoE); `Cached` before runs was 2.4 to 3.2 GB (*device*, R58 to R67) |
| the same, published files | 2,827 to 2,990 MB | **above** anything held so far |
| if the phone build repacks into anonymous memory (stock behaviour) | about 2.56 GB anonymous for layers and head, plus the app's own 0.5 to 0.7 GB (*device*, campaign 2 memo-23) | ran at 2.27 GB repacked (*device*, 2026-10-25); killed at 3.93 GB (*device*, R9). 2.56 GB is between the two and **untested** |
| if the kernel layout is a mapped file (memo-01 Y1) | 2.6 GB file-backed, plus 2.6 GB of storage for the artefact | file pages are evicted, not killed: the failure is a slow token, 0.7 to 2.7 ms per lost MB (*device* fits, memo-18) |
| KV | 10.7 MB used at 187 positions; about 110 MB allocated at the kit's `n_ctx` 4,096, touched only as used | small |
| PLE rows | under 25 MB | small; one flash read of 2 to 3 pages per new token, 0.1 to 0.4 ms each when cold (*estimated*; under 0.6 % of a 63 ms token) |

- **Gemma 4 E4B sits exactly at the edge of the resident set this phone has held, and the published files sit
  over it.** P1 (resident-capacity probe) decides this before any speed question, as for the 7B bodies; its
  3.0 GB step is the one that matters here.
- A hazard specific to this model: **anything that touches "the whole file" pulls 1.6 to 2.3 GB of PLE table
  through the page cache and evicts the layers.** Campaign 2's touch helper falls back to "every page" when it
  has no page set (memo-32 §1.1). On this model the page set must be built from tensor ranges, and the table,
  the unused k and v, and the non-hot head rows must be left out. Read-ahead on the table's range should be off
  (`MADV_RANDOM`).
- The technical report's own mobile figure is 2.3 GB of quantised weights for E4B with a mix of 2-bit and 4-bit
  weights and int8 activations (*published*, Table 3). That is another file format and is not available to
  llama.cpp.

## 5. Published perplexity of its quantisations

None of these is our gate (ctx 512, all chunks, our own F16 run). All *published*.

| source | weights | what was measured | result |
|---|---|---|---|
| llama.cpp issue #22407 (commit `f535774`; closed as stale 2026-06-10, cause not found) | **E4B base**, stock, no imatrix stated | wikitext-2 test, `llama-perplexity` | BF16 7.1099; Q8_0 7.1428 (× 1.005); Q6_K 7.3947 (× 1.040); Q5_K_M 8.3111 (× **1.169**); Q4_K_M 23.06 (× **3.24**); Q3_K_M 919.5 |
| same issue, for scale | E2B base | same | BF16 8.3238; Q4_K_M 9.1381 (× 1.098); Q3_K_M 12.693 (× 1.525) |
| localbench (62 files, about 250,000 tokens of chat-format text, reference unsloth BF16) | E4B-it, stock | mean KL over the top 40 tokens | Q4_K_M: ggml-org 0.208, lmstudio-community 0.207, bartowski (imatrix) 0.072; Q8_0 0.003 to 0.006; UD-IQ2_M agrees with the top token 78.8 % of the time |
| two community model cards (search summary) | E4B-it derivatives | KL against F16 | plain Q4_K_M 0.257 to 0.280, top token equal 79.6 to 79.8 %; with imatrix 0.088 |
| a third-party rebuild card (xbill9) | **Google's QAT** Q4_0 GGUF, and a rebuild with QAT-grid scales | wikitext-2 test, 16 chunks of 512, against a BF16 of the QAT weights | Google's file: PPL × 1.0816 ± 0.0089, KL 0.0353, top token equal 90.6 %; rebuild: × 1.0256 ± 0.0052, KL 0.0010 |

Reading (*estimated*):

- **Stock Gemma 4 E4B looks unusually fragile below six bits.** A KL of 0.2 for Q4_K_M is ten times what the
  same type costs on Llama-class models, and the one wikitext table has Q5_K_M at +16.9 %, already near the
  hard ceiling, and Q4_K_M far outside. The table is for the base model without an imatrix, from one user, with
  the cause not found; the instruct model with an imatrix is better by KL (0.07 to 0.09) and has **no published
  perplexity**.
- If that holds on our gate, the smallest stock file inside F16 × 1.20 is Q5_K-class: 4,644 M × 5.5 bits =
  **3.19 GB per token** (*estimated*), which is over the residency edge and 22 % more bytes than the mix priced
  below.
- Google's QAT checkpoint is the only route with a reported four-bit cost near F16, and the +8.2 % of Google's own
  file is against the QAT weights' BF16, not the stock model's. Under campaign 3's definition ("stock (instruct)
  weights from the publisher; no fine-tune") it needs the owner's ruling. This is round 1's question (c), and it
  now carries the whole subject.
- So **H2 (the quantisation ladder) on Gemma 4 E4B is the first test, before H1's remaining items and before any
  kernel work**: stock-it Q4_0 with and without imatrix, Q5_K_M, Q6_K, and the QAT file, 40 chunks first, against
  Q8_0 as the F16 proxy.

## 6. Priced tok/s band on the Snapdragon 888

Model (*estimated*): steady token = max(bytes ÷ read rate, weights ÷ `sdot` rate) + O; printed = 0.95 × steady.

- `sdot` rate, three threads in situ: 57.0 G weights/s at campaign 1's clocks (S1: 2,112 MHz), 69.9 with power
  saving off (S2: caps 2,419 / 2,841, R67); four threads at S2 89.1. From the *device* benches R41 and R37,
  scaled by clock and by the 0.87 in-situ factor (*estimated*, memo-01 draft §1).
- O = everything that is not a weight product: attention over the KV (0.8 to 1.0 ms at 187 positions; 2.1 to
  2.7 ms at 512), norms, GELU, the soft-cap `tanhf` (2 to 4 ms), sampling over 262,144 logits, thread syncs of
  about 380 large products, one PLE page read. **6 to 12 ms** with campaign 1's runtime (*estimated*; not
  measured on any dense model). With the stock runtime the graph has about 2,000 to 2,400 nodes, and at
  campaign 1's 23 µs per barrier that alone is about 45 to 55 ms (*estimated* from *device* R4).

Steady tok/s; the range in each cell is O = 12 to 6 ms.

| case | bytes (MB) | weights (M) | clocks | 19 GB/s | 24 GB/s | 33 GB/s | 51 GB/s | `sdot` ceiling |
|---|---|---|---|---|---|---|---|---|
| E0: ggml-org Q4_0 as published (Q8_0 head priced at 1.5 × `sdot` per weight, an assumption; BF16 projection scalar) | 2,990 | 4,980 equivalent | S1 | 5.6 to 5.8 | 6.9 to 7.2 | 9.0 to 9.6 | 9.3 to 9.9 | 9.3 to 9.9 |
| | | | S2 | 5.6 to 5.8 | 6.9 to 7.2 | 9.0 to 9.6 | 10.9 to 11.8 | 10.9 to 11.8 |
| **E1: campaign mix, every multiplied tensor Q4_0** | 2,615 | 4,644 | S1 | 6.7 to 7.0 | 8.3 to 8.7 | 10.7 to 11.4 | 10.7 to 11.4 | 10.7 to 11.4 |
| | | | S2 | 6.7 to 7.0 | 8.3 to 8.7 | 11.0 to 11.7 | 12.7 to 13.8 | 12.7 to 13.8 |
| E1 on four threads (lost twice on MoE models, R18 and R62; untested on a dense body) | 2,615 | 4,644 | S2 | 6.7 to 7.0 | 8.3 to 8.7 | 11.0 to 11.7 | 15.6 to 17.2 | 15.6 to 17.2 |
| E2: a 6-bit head served (Google QAT file) + two-stage head (Q4_0 screen, 2,048 exact rows) + F16 projection | 2,659 | 4,650 | S2 | 6.6 to 6.9 | 8.1 to 8.6 | 10.8 to 11.6 | 12.7 to 13.8 | 12.7 to 13.8 |
| E3: E1 + 25 % column sparsity on the layers, kernel 1.2 × | 2,128 | 4,255 | S2 | 8.1 to 8.5 | 9.9 to 10.6 | 13.1 to 14.2 | 13.7 to 15.0 | 13.7 to 15.0 |
| E3: 25 %, kernel 1.85 × | 2,128 | 6,151 | S2 | 8.1 to 8.5 | 9.9 to 10.6 | 10.0 to 10.6 (E1 is better) | E1 is better | – |
| E3: 40 %, kernel 1.2 × | 1,837 | 3,555 | S2 | 9.2 to 9.7 | 11.3 to 12.1 | 14.8 to 16.2 | 15.9 to 17.6 | 15.9 to 17.6 |
| E3: 40 %, kernel 1.85 × | 1,837 | 5,072 | S2 | 9.2 to 9.7 | 11.3 to 12.1 | 11.8 to 12.7 | E1 is better | – |
| E3: 40 %, kernel 2.6 × | 1,837 | 6,822 | S2 | 9.1 to 9.7 | 9.1 to 9.7 | E1 is better | E1 is better | – |

Column bytes: the column-major Q4_0 copy keeps one fp16 scale per row and 32-column block, and those are read
whatever columns are skipped, so a skipped fraction s saves 8/9 × s of the layer bytes, not s. Memo-01 did not
count this.

What the table says:

1. **Printed 15.0 needs a steady token of 63.2 ms.** For E1 that is a read rate of **46 to 51 GB/s** and at
   most 3,580 to 4,000 M weights at S2. E1 has 4,644 M. So even on a bus at its nominal peak, three threads must
   not compute 14 to 23 % of the weights.
2. Paths whose range contains 15 printed (15.8 steady), all at S2: four threads with 46 GB/s or more (two
   unmeasured things at their best values); or 40 % sparsity with a kernel at 1.2 × and 33 GB/s or more.
   **Published tolerance for 40 % exists only for SiLU Llama-class models; nothing is published for Gemma's GELU
   FFN**, and its fragility under quantisation (§5) is not encouraging.
3. At the read rate campaign 1 measured (19 to 24 GB/s), every row is 5.6 to 12.1.
4. **Stock runtime prediction for a first baseline:** scaling the one dense *device* anchor (Qwen3.5-4B Q4_0,
   2.23 GB per token, 5.9 to 6.6 tok/s printed, stock runtime, power saving most likely on) by bytes gives 4.4 to
   4.9 tok/s for the ggml-org file and 5.0 to 5.6 for the campaign mix (*estimated*). An Exynos 1380 phone is
   reported at 3.4 to 3.6 tok/s with llama.cpp (*published*, one blog).

Two small exact levers this model adds (both *estimated*, both for Tom's microbench before anyone builds):

- **BF16 projection:** serve it as Q4_0 or Q8_0 (a file choice, free), or add a NEON BF16 dot (a declared
  numerics change unless it keeps the scalar loop's summation order). About −7 ms per token against the published
  files; zero against a mix that never had BF16.
- **Soft-cap fast path for greedy decode:** tanh is monotone, so the arg-max can be taken before it. To stay
  exact when two logits round to the same value after tanh, apply the stock scale, `tanhf`, scale to the
  candidates within a margin of the maximum and take the stock arg-max among them. About −2 to −4 ms per token;
  the perplexity path is untouched. Same class as campaign 1's sampler fast path (R33).

## 7. Which campaign 2 levers carry over

| lever (campaign 2 state) | carries over? | price on Gemma 4 E4B (*estimated*) |
|---|---|---|
| **pin** (patch 0010, persistent pool and per-worker mask; +1.94 ± 0.69 tok/s on OLMoE, *device*, R67) | yes, model-generic | a condition, not a gain on top of §6: the bands assume three pinned big cores. A worker on an A55 for one run is a 2 to 4 × slower token, since every product is long |
| **exec map** (patch 0011; keeps pages through a wait, *device*, R67; speed not shown) | yes | worth more than on a MoE model, because every byte is needed every token. Losing 240 MB in a wait (R62's median) is 170 to 650 ms of flash reads, 5 to 20 % of a 32-token run. Needs the 4.2 to 5.2 GB file on internal storage |
| **touch** (patch 0012; R68 not run yet) | yes, **with a tensor-range page set** (§4). The page set is 640,000 pages, 1.0 to 1.3 × OLMoE's | unpriced until R68 reports the cost of a pass |
| **kv clear** (patches 0008 and 0009; OLMoE time to first token −704 ± 149 ms, *device*, R66) | yes, generic | time to first token only, about −0.1 to −0.3 s for about 110 MB of KV; zero on the decode metric |
| **two-stage head** (campaign 1; +0.01 % PPL with 2,048 exact rows on Granite, *host*) | yes, and the head is larger here than on any earlier model | against a Q4_0 head: nothing (memo-01). Against a Q6_K or Q8_0 head: −173 to −335 MB per token (3 to 18 ms at 51 to 19 GB/s), which is E2: **it buys a better head at the Q4_0 head's speed, it does not beat E1.** The 2,048-row rule was set on a 100 k vocabulary and needs its own host check at 262 k. A screen under four bits is closed (R29). The screen is a 377 MB derived artefact. KleidiAI may take the head first (R60, patch 0006) |
| **skip** (expert skip by router weight) | no: there are no experts and no router | its dense counterpart is column sparsity (H3, H6), unmeasured on a GELU model |
| campaign 1's small-ops-on-one-thread and sampler fast path | yes | most of the gap between the stock O (45 to 55 ms) and 6 to 12 ms. This is the largest single carry-over and it is the same for the 7B bodies |

## 8. Other "effective-parameter" dense models in the 6 to 8.5 B range

Searched today: Google's is the only family that ships per-layer embeddings. LongCat-Flash-Lite (n-gram
embeddings, arXiv 2601.21204) is a 68.5 B MoE and outside the class. Gemma 4 E2B is 5.1 B in total and outside
the range. One model remains.

**Gemma 3n E4B** (*host*: header of `unsloth/gemma-3n-E4B-it-GGUF` Q4_0 and `src/models/gemma3n.cpp` at the base).

| item | value |
|---|---|
| size label | 6.9 B text weights ("8 B raw" with its encoders, *published*); llama.cpp `gemma3n` is in the base |
| layers, width, FFN | 35; 2,048; 16,384 (8 ×) |
| multiplied per token | FFN 3,523 M; attention 336 M (15 of 35 layers share KV); head 537 M (tied, 262,144 × 2,048); PLE gates 37 M; AltUp and LAuReL extras 53 M (F16): **4,485 M** |
| looked up | PLE table 262,144 × 8,960 = 2,349 M (1.76 GB in this file) |
| bytes per token, all Q4_0 with the F16 extras | about 2,600 MB (*estimated*) |
| **stock activation sparsity** | in layers 0 to 9 the gate pre-activation passes through ReLU(x − mean − 1.645 σ) before GELU (*host*: `gaussian_topk`, `n_layer_sparsity = 10`; the file carries the scale). About 95 % of the 16,384 gate outputs of those layers are **exact zeros in the stock model** |

- Priced (*estimated*): those zeros allow an exact skip of 95 % of `ffn_up`'s rows (free in the file's row-major
  layout once the gate is computed first) and 95 % of `ffn_down`'s columns (needs the column kernel, which at
  5 % density wins at any cost in memo-01's band). That removes about 620 M of 4,485 M weight products and about
  350 MB of 2,600 MB, half of it only with a 189 MB column-major copy of ten `ffn_down` tensors.
- Band, O = 8 to 14 ms (the AltUp graph has more small ops), S2: stock 6.6 to 6.9 | 8.2 to 8.6 | 10.8 to 11.5 |
  12.8 to 13.9 at 19 | 24 | 33 | 51 GB/s; with the exact skip 7.6 to 7.9 | 9.3 to 9.8 | 12.2 to 13.1 | 14.4 to
  15.8, `sdot` ceiling 14.4 to 15.8.
- It is the same order as Gemma 4 E4B, with one **bit-identical** 12 to 14 % lever that Gemma 4 does not have.
  Its perplexity under quantisation was not found published. It is an older and weaker model (2025-06), and
  **it is the same family as Gemma 4 E4B, so the pair does not meet the "two families" rule.**

## 9. The strict dense 7B path as a function of the read rate

Subjects: Mistral-7B-Instruct-v0.3 and Qwen2.5-7B-Instruct, Q4_0. Shapes are *estimated* from the published
configurations as recalled (H1 confirms them from headers).

| | layer weights W_L (M) | head W_H (M) | layer bytes B_L (MB) | head bytes B_H (MB) |
|---|---|---|---|---|
| Mistral-7B-v0.3 (32 layers, 4,096 wide, FFN 14,336, vocabulary 32,768) | 6,979 | 134 | 3,926 | 75 |
| Qwen2.5-7B (28 layers, 3,584 wide, FFN 18,944, vocabulary 152,064) | 6,526 | 545 | 3,671 | 307 |

**Formula** (*estimated*). r = dense read rate in GB/s (P2), c = `sdot` rate in G weights/s (P2; 57.0 at S1, 69.9
at S2 from campaign 1's benches), s = column sparsity on the layers, k = column-kernel cost per kept weight
relative to `sdot` (H6; band 1.2 to 2.6), O = 5 to 10 ms:

    steady ms = max( (B_H + B_L × (1 − 0.889 s)) / r ,  (W_H + W_L × (1 − s) × k) / c ) + O
    printed tok/s = 0.95 × 1000 / steady ms

- With s = 0, k = 1 it is the stock path. The lever is used only where it beats the stock path.
- The first term falls with r; the second does not. Their crossing r* is the read rate above which a faster bus
  buys nothing.

Steady tok/s at S2 (power saving off), O = 10 to 5 ms. "stock" in a cell means the stock path is faster there.

| Mistral-7B-v0.3 | 19 GB/s | 24 GB/s | 33 GB/s | 51 GB/s | plateau from r* |
|---|---|---|---|---|---|
| stock Q4_0 | 4.5 to 4.6 | 5.7 to 5.8 | 7.6 to 7.9 | 8.9 to 9.4 | 8.9 to 9.4 from 39 GB/s |
| 25 %, k = 1.2 | 5.7 to 5.9 | 7.1 to 7.4 | 9.5 to 10.0 | 9.8 to 10.3 | 9.8 to 10.3 from 34 |
| 25 %, k = 1.85 | 5.7 to 5.9 | 6.6 to 6.9 | stock | stock | 6.6 to 6.9 from 22 |
| 25 %, k = 2.6 | 4.8 to 5.0 | stock | stock | stock | 4.8 to 5.0 from 16 |
| 40 %, k = 1.2 | 6.8 to 7.0 | 8.4 to 8.8 | 11.2 to 11.9 | 11.9 to 12.7 | **11.9 to 12.7 from 35** |
| 40 %, k = 1.85 | 6.8 to 7.0 | 8.1 to 8.5 | 8.1 to 8.5 | stock | 8.1 to 8.5 from 23 |
| 40 %, k = 2.6 | 6.0 to 6.1 | 6.0 to 6.1 | stock | stock | 6.0 to 6.1 from 17 |
| 40 %, k = 1.2, four threads | 6.8 to 7.0 | 8.4 to 8.8 | 11.2 to 11.9 | 14.7 to 15.9 | 14.7 to 15.9 from 46 |

| Qwen2.5-7B | 19 GB/s | 24 GB/s | 33 GB/s | 51 GB/s | plateau from r* |
|---|---|---|---|---|---|
| stock Q4_0 | 4.6 to 4.7 | 5.7 to 5.9 | 7.7 to 8.0 | 9.0 to 9.4 | 9.0 to 9.4 from 39 |
| 25 %, k = 1.2 | 5.7 to 5.8 | 7.1 to 7.3 | 9.5 to 9.9 | 9.8 to 10.3 | 9.8 to 10.3 from 34 |
| 25 %, k = 1.85 | 5.7 to 5.8 | 6.8 to 7.0 | stock | stock | 6.8 to 7.0 from 23 |
| 40 %, k = 1.2 | 6.6 to 6.9 | 8.2 to 8.6 | 11.0 to 11.6 | 11.8 to 12.5 | **11.8 to 12.5 from 36** |
| 40 %, k = 1.85 | 6.6 to 6.9 | 8.2 to 8.6 | 8.2 to 8.6 | stock | 8.2 to 8.6 from 24 |
| 40 %, k = 2.6 | 6.1 to 6.3 | 6.1 to 6.3 | stock | stock | 6.1 to 6.3 from 17 |
| 40 %, k = 1.2, four threads | 6.6 to 6.9 | 8.2 to 8.6 | 11.0 to 11.6 | 14.5 to 15.7 | 14.5 to 15.7 from 46 |

At campaign 1's clocks (S1) every plateau is about 0.83 of the S2 value and begins at 0.82 of its r*.

**How P2's number decides:**

| P2 reads | stock 7B | best strict path (40 %, k = 1.2) | what it means |
|---|---|---|---|
| r ≤ 24 GB/s | ≤ 5.9 | ≤ 8.8 | bytes bind everywhere. The kernel cost hardly matters below k = 1.85; only fewer bytes or a second reader helps. The strict path ends near 8 to 9 |
| 24 < r < 35 | 5.9 to 8.2 | 8.8 to 12.4 | the kernel cost decides: at k = 1.85 the path stops at 8.1 to 8.6 whatever r is |
| r ≥ 35 | 8.2 to 9.4 | **11.9 to 12.7 and no more** | compute binds. More bus gives nothing. Only a fourth thread or fewer computed weights moves it |

- **15 printed (15.8 steady) is not in any three-thread cell.** The only cell that touches it needs all of: 40 %
  sparsity inside +5 % on both models (*published* for Mistral on FP16 weights at +4.3 %, the whole budget;
  nothing for Qwen2.5), a kernel at 1.2 × (the bottom of an unbenched band), a fourth thread that scales (lost
  twice on this phone), 46 GB/s (twice what was measured), and 2.6 GB resident beside a 3.9 GB derived copy. I
  put that below 0.02 (*estimated*, judgement).
- **H6's number matters as much as P2's.** At k = 1.85 the path stops at 8.1 to 8.6 from 23 GB/s and falls under
  the stock path near 36 GB/s; at k = 2.6 it stops at 6.0 to 6.3 from 17 GB/s. Memo-01's kill line for H6 (2.2 ×) is too loose for Mistral and Qwen2.5 at
  40 %: at k = 2.2 the path gives 7.0 to 7.3 (*estimated*), under the stock path from 30 GB/s. I would set it at
  **1.6 ×**, where 40 % still gives 9.2 to 9.7 at 33 GB/s.
- Residency is unchanged from memo-01: the column-major copy is a 3.7 to 3.9 GB artefact, the union of the active
  columns over a run is all of it (until H5 shows otherwise), and the phone has not held more than 2.67 GB.

## 10. Proposals for the joint lists

Test list (each decides a number above):

| # | test | decides | closes if |
|---|---|---|---|
| G1 | **Quantisation ladder on Gemma 4 E4B first** (H2, this model): stock-it Q4_0 with and without imatrix, Q4_K_M, Q5_K_M, Q6_K, Google's QAT Q4_0; Q8_0 as the F16 proxy; 40 chunks, then all chunks for the survivors; at `-b 512` and `-ub 1` | whether any stock file of 4.5 bits per weight or less is inside F16 × 1.20 | no stock file at or under 4.5 bpw passes: the subject needs the QAT ruling or a 5.5 bpw file (3.19 GB per token, band × 0.82) |
| G2 | Head-type ladder on the same model: Q4_0, Q6_K, Q8_0 head with the layers fixed; two-stage head with 1,024, 2,048, 4,096 exact rows | E1 against E2; whether 2,048 rows hold at a 262 k vocabulary | Q4_0 head costs under 0.5 %: E2 is not needed |
| G3 | Host microbench, aarch64 container: BF16 against F16 and Q8_0 for a 10,752 × 2,560 product; `tanhf` over 262,144 values | the two §6 levers | under 1 ms each on the host scaled to the A78: drop them |
| G4 | H1 on the phone build: which Gemma tensors the repack and KleidiAI take, and whether the tied head becomes an anonymous copy | §4's anonymous figure | – |
| G5 | P1 with a 2.6 and a 3.0 GB step, the PLE range excluded | §4 | 2.6 GB not held at DRAM speed: the subject fails on residency |
| G6 | H3 on Gemma 4 E4B's GELU FFN (input-sparsity tolerance), only if G1 passes | E3 | under 25 % inside +5 %: E3 closes |
| G7 | Gemma 3n E4B: the measured fraction of exact zeros per layer on wikitext-2 train (host, minutes) | §8's 95 % | under 85 %: the exact skip is under 10 % of the token |
| S1 | H6's kill line moved to 1.6 × `sdot` per kept weight | §9 | – |

Try list:

1. The campaign mix (every multiplied tensor Q4_0, PLE table Q4_0, projection Q4_0 or Q8_0), built with
   `llama-quantize` tensor-type overrides from the stock weights. It is the E1 row and costs a host run.
2. Soft-cap fast path and the gate-first FFN order (one pass, fewer barriers).
3. A drafted second token checked in one batch of two, **only if P2 reads under about 28 GB/s** (see §11,
   point 3).

Questions for the owner, through the lead:

- (a) Is Gemma 4 E4B "dense 7B-class"? Now with numbers: 7.5 B text weights, 4.64 B multiplied per token.
- (b) Is Google's QAT checkpoint "stock weights"? §5 suggests the answer decides whether the subject exists.
- (c) Do Gemma 4 E4B and Gemma 3n E4B count as two families? I read the rule as no.

## 11. Review of Ebony's draft (read 2026-11-18, after §0 to §10 were written)

Her draft closes the Hexagon and stock-OpenCL lanes with sources, corrects memo-01 §7 on what campaign 1's GPU
number was, and proposes the two benches (X1, X2) that decide whether the bus has headroom. I agree with all of
that. Points where a number should change:

1. **"The provisional 4.5 B subject, 2.53 GB per token, needs 38 GB/s" (her §0 item 1, §1.4, §6).** By the
   headers it is 4,644 M weights and 2.61 GB at best, 2.83 to 2.99 GB in the published files (§3). With the
   non-product time counted, 15 printed needs 46 to 51 GB/s (§6), and on three threads the compute cap is 12.7 to
   13.8 whatever the bus does.
2. **Her CPU + GPU row for that subject, "11.5 to 15.8 by bytes ... the first lever whose range contains 15".**
   Re-priced with her own inputs (*estimated*): combined 29 to 40 GB/s on 2,615 MB is 65 to 90 ms; O is 6 to
   12 ms; her join cost of 20 to 100 µs applies to about 380 mat-vecs on this model (42 layers × 9), 7.6 to
   38 ms, or 2.5 to 12.7 ms if only the 126 FFN products and the head are split (84 % of the bytes). Token 74 to
   115 ms: **8.7 to 13.6 tok/s steady, 8.3 to 12.9 printed.** The range no longer contains 15. It is still the
   largest lever of her round, and on this subject the CPU share's compute cap does lift as she says.
   - The residency side she asks me to price: a 0.4 GPU share is 1.05 GB of driver memory on this model. With the
     app's 0.5 to 0.7 GB that is 1.6 to 1.8 GB that cannot be evicted, below what has run (2.27 GB repacked,
     *device*) and it leaves 1.57 GB file-backed, well inside the 2.67 GB held (*device*, R58). On the numbers we
     have, the split helps residency on Gemma 4 E4B. On a 7B body the same share is 1.6 GB of driver memory plus
     2.4 GB of file pages, and I cannot price which of the two the phone gives up first; P1 with a pinned-memory
     arm would.
3. **"Self-drafted second column ... stays closed" (her §4).** Her reason is right for a compute-bound token, and
   R20 measured exactly that on Granite. A dense token at 19 to 24 GB/s is the opposite case: bytes take 109 to
   138 ms and compute 66 ms (E1, S2), so a second column rides on bytes already being read. Price
   (*estimated*): step = max(2,615 / r, 2 × 4,644 / 69.9 = 133 ms) + O + drafter; tokens per step = 1 + a, a = the
   chance the drafted token is the greedy one.

   | read rate | single token | batch of two, a = 0.5 | a = 0.7 |
   |---|---|---|---|
   | 19 GB/s | 6.7 to 7.0 | 10.1 | 11.4 |
   | 24 GB/s | 8.3 to 8.7 | 10.4 | 11.8 |
   | 33 GB/s | 11.0 to 11.7 | 10.4 | 11.8 |

   It is a hedge, not an addition: it is worth 1.2 to 1.6 × if P2 reads low and nothing if P2 reads 33 GB/s or
   the GPU split works, as she says. Gemma 4 ships a publisher drafter for this (the 77 M MTP head; ggml-org has
   GGUFs for E4B). Its acceptance rate on greedy decode I did not find; it is a second publisher file (owner
   question); and greedy identity between a batch of two and a batch of one must be shown, since ggml's
   one-column and multi-column kernels are different code. I would move it from "closed" to "priced, gated by
   P2", with the lead to decide whether R20 is re-opened.
4. **Per-token FFN-sublayer skip (her rank 3), on Gemma.** The FFN is 71 % of this model's multiplied weights
   (3,303 of 4,644 M), so 5 to 15 % of FFN blocks is 3.6 to 10.7 % of the token, × 1.04 to 1.12: her range
   stands. Gemma 4's FFN output passes a post-norm before the residual add, so the "input and output nearly
   parallel" rule needs the norm in the comparison.
5. **Outside check she can add to §2.** Qualcomm's model page for Gemma 4 E4B (Q4_0, a llama.cpp-based runtime)
   has 14.9 to 17.2 tok/s at context 512 on the Snapdragon 8 Elite and 16.9 to 17.6 on the 8 Elite Gen 5
   (*published*; the page does not say which rows are CPU, GPU or NPU). The goal for this phone equals the
   vendor's own figure for chips three and four generations later.
6. **Her §3.1 reading that a CPU clock cap can lower the DDR floor** makes P2 the bus test as well as the clock
   test. I agree, and §6 and §9 are written in r for that reason. One addition to X1: print the Q4_0 4x4 `sdot`
   rate on a buffer that fits L2 beside the 512 MB rate, per core, in the same run, so that c and r come from
   one session and the same clocks.
7. **Multi-stream GEMV "× 1.0 to 1.3, then the `sdot` cap".** Consistent with §9: for a 7B body the cap starts at
   39 GB/s at S2, so the whole × 1.3 is usable from 19 to 24. For Gemma 4 E4B the cap starts at 39 GB/s as well
   (2,615 MB ÷ 66.4 ms).
8. Small: her baseline "2.53 GB" and "4.5 B" should be replaced by §3's figures wherever they appear, and her
   rank table should carry a second column for the 2.61 GB body, since the ranks differ there (the batch of two
   and the split both gain more on it than on a 7B body).

Joint position I would sign (*estimated*): for a 7.1 B body nothing in either draft changes memo-01's verdict;
the best stack is 9 to 12.7 tok/s in a corner that needs four unmeasured things at once. For Gemma 4 E4B the
exact band is 6.7 to 13.8 by read rate, the best priced stack is 12 to 14, and whether the subject exists at
four bits is an open perplexity question that comes before all of it.

## 12. Sources (read 2026-11-18)

- Gemma 4 E4B publisher configuration: <https://huggingface.co/google/gemma-4-E4B-it> (`config.json`, fetched directly)
- GGUF headers (first 32 MB, parsed locally): <https://huggingface.co/ggml-org/gemma-4-E4B-it-GGUF>,
  <https://huggingface.co/unsloth/gemma-4-E4B-it-GGUF>, <https://huggingface.co/bartowski/google_gemma-4-E4B-it-GGUF>,
  <https://huggingface.co/google/gemma-4-E4B-it-qat-q4_0-gguf>, <https://huggingface.co/unsloth/gemma-3n-E4B-it-GGUF>
- Gemma 4 technical report, arXiv 2607.02770: <https://arxiv.org/html/2607.02770v1> (fetch tool)
- Gemma 4 model card: <https://ai.google.dev/gemma/docs/core/model_card_4> (search summary)
- llama.cpp releases: <https://github.com/ggml-org/llama.cpp/releases> (fetch tool); PR #31356
  <https://github.com/ggml-org/llama.cpp/pull/31356>; PR #31394 <https://github.com/ggml-org/llama.cpp/pull/31394>;
  PR #23398 <https://github.com/ggml-org/llama.cpp/pull/23398> (search summary)
- llama.cpp issue #22407, perplexity of Gemma 4 E4B base quantisations: <https://github.com/ggml-org/llama.cpp/issues/22407> (fetch tool)
- localbench, Gemma 4 E4B GGUF quality: <https://localbench.substack.com/p/gemma-4-e4b-gguf-quality-benchmark> (fetch tool; its tables are in images and were not read)
- QAT rebuild card: <https://huggingface.co/xbill9/gemma-4-E4B-it-qat-q4_0-exact-gguf> (fetch tool); Google's QAT post:
  <https://www.blog.google/innovation-and-ai/technology/developers-tools/quantization-aware-training-gemma-4/> (search summary)
- Qualcomm model page: <https://huggingface.co/qualcomm/Gemma-4-E4B-it> (fetch tool)
- Phone reports (search summaries, not opened): <https://dev.to/baiju_rajyaguru_a70384dfd/i-ran-gemma-4-on-a-mid-range-android-phone-heres-what-actually-happened-40ge>,
  <https://dev.to/alichherawalla/how-to-run-gemma-4-on-your-android-phone-in-2026-locally-no-cloud-1e39>
- LongCat-Flash-Lite, arXiv 2601.21204: <https://arxiv.org/html/2601.21204v2> (search summary)
- Repository: `lab/llama.cpp-base` at `620d2225c` (`src/models/gemma4.cpp`, `src/models/gemma3n.cpp`,
  `src/llama-model-loader.cpp`, `ggml/src/ggml-cpu/vec.cpp`, `vec.h`, `repack.cpp`); campaign 2 `STATUS.md`,
  `knowledge.md`, memo-18, memo-23, memo-32; memo-01 and its two drafts.

## 13. Not done, and where I may be wrong

- No perplexity was run. §5 rests on one user's table for the **base** model and on KL figures from community
  pages; G1 replaces it.
- The BF16 and `tanhf` costs are instruction-level estimates from the source, not benches. I did not check
  whether master has added a NEON BF16 path.
- The Q8_0 head's kernel cost (1.5 × `sdot` per weight in row E0) is an assumption.
- O (6 to 12 ms) has never been measured on a dense model with campaign 1's runtime. If it is 20 ms, every cell
  of §6 falls by about 1 tok/s.
- Mistral's and Qwen2.5's shapes are from recalled configurations, not headers.
- The Qualcomm rows do not say which processor ran them.
- I did not find the MTP drafter's acceptance rate, and I did not search for Gemma 3n's quantised perplexity
  beyond one query.
- The header reader counts bytes from element counts and bits per weight. Its total for the ggml-org file is
  4,575.0 MB of tensors against a 4,590.8 MB file; the difference is the 15.8 MB header.
