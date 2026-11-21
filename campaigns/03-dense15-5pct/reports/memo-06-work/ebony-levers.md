# memo-06 work file, Ebony: levers for 12 tok/s on a dense 7B-class body, and what they ask of the model

2026-11-21, Ebony (`claude-fable-5-2`). **Nothing here was run on the phone or on the host.** Labels: *published* =
a source's claim, not reproduced (with how the source was read); *host* / *device* = quoted from a stored result of
this repository (named); *estimated* = my arithmetic. Web pages were read on 2026-11-21 through a summarising fetch
tool; "abstract only" and "search summary" are said where true. Pricing is Bonnie's lane: my *estimated* figures
are first guesses for her to replace. Verdicts and the choice of subjects are the lead's.

Token model used throughout (memo-04-work §6, clyde-replan §1): t = max(B ÷ r, W ÷ c) + O, with B the bytes read,
W the compute in stock-Q4_0-`sdot` byte equivalents, O = 8 ms, r = 24 GB/s (R69 band 19 to 30; 35 shown as the
upper case), c = 40.4 GB/s on X1 + 2 A78 (band 35 to 45) and 50.3 for four big cores scaling perfectly (memo-05
work §4). All four inputs are pre-registered bands, not measurements; R69 has not run.

## 0. Short answers

1. **Nothing published in 2025 to 2026 changes the arithmetic.** No new training-free method cuts compute on a
   stock SwiGLU model inside +5 % by more than the methods already screened here, and no runtime (llama.cpp,
   ik_llama.cpp, ExecuTorch, MNN, T-MAC, bitnet.cpp) has a 4-bit ARM kernel that beats the repacked `sdot` path per
   weight on a core without i8mm or SME2 (§1).
2. **I agree with the lead's weight line** (clyde-replan §1): at c 40.4 the token affords 5.4 G `sdot`-weights, and
   every eligible model multiplies 5.8 G or more. So at 12 tok/s the binding quantity is **compute per token**,
   and a lever that cuts bytes but not compute (a lower-bit file, down-column sparsity at k 1.8) cannot reach 12
   by itself on any 6 B+ body.
3. **The one published idea that cuts both** is whole-neuron FFN skipping ranked by a cheap proxy (Prox, 2026-07;
   this repository's parked "row skip", memo-05 §3). Its PPL table is now readable (§1.1) and it changes the
   model ranking: Qwen3-8B holds 60 % FFN sparsity at +4.9 %, Mistral-7B 50 % at +4.1 %, Llama-3-8B fails at 40 %
   (+5.6 %), and the GeGLU model (Gemma-3-12B) collapses at 40 % (all *published*, FP16 weights, one-decimal PPLs).
   But priced with this repository's measured kernel costs it is **compute-neutral or worse** unless the kept
   rows run at k ≤ about 1.5 (§2.2). The new proposal of this file is a form with **k = 1.0**: skip neurons in
   groups of four rows of the stock 4x4 repack (§2.3). Whether enough groups are skippable is a host screen.
4. **Best stack and its price** (§2.4, *estimated*): the smallest body, no-copy layout, whole-neuron skip with a
   1-bit proxy, `down` columns through K1, two-stage head, and a 4th big core. On a 5.8 G-multiplied body
   (Yi-1.5-6B class) it gives 8.7 to 9.8 tok/s at r 24 and 12.0 to 13.8 only at r ≥ 30 to 35 with k ≤ 1.5. On a
   7.1 G body (Mistral, Qwen2.5) the same stack gives 7.3 to 8.2 at r 24 and at most 10.3 to 11.6 at r 35 with
   four cores. **I do not see 12 on a 7.0 G+ body.** My judgement, not a price: P(12) about 0.10 for a 5.8 G body,
   about 0.02 for a 7.1 G body, and P(all three) at most 0.02 unless three families each offer a body at or
   under about 6.2 G multiplied weights. That is below the lead's prior (0.05 to 0.15); both are recorded.
5. **Model properties, in order of weight** (§3): (1) multiplied weights per token, where a large **untied**
   vocabulary helps (its embedding counts toward the 6.0 B floor but is never multiplied); (2) host-screened
   neuron-skip tolerance, which is family-specific; (3) FFN share of the bytes; (4) quantisation robustness below
   4.5 bpw, because residency (C about 3.1 GB, R69 band) excludes every Q4_0 body of 6 B+ before speed is even
   asked. GQA matters through attention bytes, not through KV (0.3 to 0.6 % of the token). No ReLU-family stock
   dense transformer of 6.0 to 8.5 B was found; non-gated GELU models (Falcon-7B, MPT-7B) exist in the base tree
   and are unscreened.
6. Lossless lookahead stays closed for the 12 path: it pays only while the token is byte-bound, and every path to
   12 ends compute-bound (§1.3).

*Note added after reading `bonnie-pool.md` (same day; the text above and §2 are left as written): every tok/s in
§0 and §2 is a **steady** value. Bonnie prices the printed benchmark value as 0.95 x steady, and I adopt that
(§4.2): read my figures 5 % lower (the 5.8 G corners 12.0 / 12.7 / 13.1; Llama-2's 12.3 is 11.7) and my P(12) on a
5.8 G body as about 0.07, not 0.10. Her pool has no GELU or ReLU-family eligible model, so property 6 and T4's
GELU count fall away.*

## 1. Scout (read 2026-11-21)

Not repeated: the rows of memo-01 work §2.1, memo-04 work §1 and memo-05 work §2 (TEAL, DIP, LaRoSA, SpQt, CATS,
CMoE, PowerInfer-2, ActiveFlow, NorSA, SpAx, T-MAC on Q4_0, MNN-LLM). Only what is new or newly readable.

### 1.1 Training-free activation sparsity

| work | date, how read | claim (*published*) | models | fit to our rules | *estimated* effect on the SD888 |
|---|---|---|---|---|---|
| **Prox** ([arXiv 2607.27591](https://arxiv.org/html/2607.27591)) | 2026-07; HTML read (first 100 K of 109 K characters) | Stage 1: input sparsity and INT4 proxies of `gate`/`up` rank the FFN channels; stage 2 computes only the chosen channels in all three matrices. 1.51 to 1.99 x decode at 60 to 70 % FFN sparsity, A6000 GPU, FP16. Proxy storage +12 %. No CPU number | ten: Qwen3-1.7B/4B/8B/14B, Qwen3.5-4B/9B, Ministral-3.3B, Mistral-7B, Llama-3-8B, Gemma-3-12B | File unchanged if the proxy is a derived side file (lead's provisional answer (c)). Numerics change, per token and per neuron, so the same rule at `-b 512` and `-ub 1`. Their proxy is 4-bit against FP16; ours must be 1 to 2 bit against a 4.5-bit file | §2.2: bytes x 0.72 to 0.77 of the token; compute x 0.83 to 1.05. Alone +1.5 to +3 tok/s at r 24, and 0 to +1.5 when compute-bound |
| WINA ([arXiv 2505.19427](https://arxiv.org/html/2505.19427)), ICLR 2026 | HTML read | score \|x_j\| x column norm; "up to 2.94 %" better average accuracy than TEAL at equal sparsity; 60 to 64 % fewer GFLOPs at 65 %. **No wikitext PPL in the paper.** It needs a one-off orthogonal transform W' = WV of the stored weights | Llama-2-7B, Llama-3-8B, Mistral-7B, Phi-4-14B | The transform is a second quantisation of the file: **not** file-unchanged. The score alone is a side table of column norms and fits | Unknown; a few points of sparsity at best. One host cell decides (T3; memo-04 work T6 was never run) |
| R-Sparse ([arXiv 2504.19449](https://arxiv.org/html/2504.19449v1)), ICLR 2025 | HTML read | sparse input channels plus a rank-r SVD path for the dropped ones; 50 % model-wide with accuracy 64.06 / 66.20 / 68.39 against dense 65.88 / 69.44 / 69.89; up to +42 % speed on an A6000, FP32. PPL only in a figure. Per-model evolutionary search, about 1 GPU-hour | Llama-2-7B, Llama-3-8B, Mistral-7B | The low-rank factors are extra weights read and multiplied every token, r(m+n) per matrix; on a compute-bound CPU they are cost, not saving | Not proposed: adds compute where compute binds |
| SGA-Sparse ([arXiv 2611.11223](https://arxiv.org/abs/2611.11223)) | 2026-11; abstract only | trains the SwiGLU gate as a top-k router, 75 to 85 % FFN sparsity. Says training-free routing by the native gate is "substantially worse" than by the up activation at high sparsity, and PTQ breaks its checkpoints | Llama 1B to 8B, retrained | Trained weights: not stock. No release seen | None. It agrees with E2 (gate rule closed, *host*) |
| SpAx ([arXiv 2610.02598](https://arxiv.org/html/2610.02598)) | search summary | 2.06 x mean at 4-bit on CPU-offloaded weights, PPL tolerance 10 % | – | Twice our budget | Not proposed (memo-05 §3) |

Prox, WikiText PPL (*published*; 128 validation sequences, context 2,048, FP16 weights, sparsity is "effective"
FFN sparsity including the proxy cost; my percentages from one-decimal PPLs, so each carries about ±1 point):

| model | dense | 40 % | 50 % | 60 % | 70 % | largest level inside +5 % |
|---|---|---|---|---|---|---|
| Qwen3-8B | 8.2 | 8.3 (+1.2 %) | 8.4 (+2.4 %) | 8.6 (+4.9 %) | 9.4 | 50 to 60 % |
| Mistral-7B | 4.9 | 5.0 (+2.0 %) | 5.1 (+4.1 %) | 5.3 (+8.2 %) | 5.8 | 50 % |
| Llama-3-8B | 7.1 | 7.5 (+5.6 %) | 7.8 | 8.2 | 9.5 | under 40 % |
| Qwen3.5-9B (above 8.5 B if dense; not checked) | 7.4 | 7.6 (+2.7 %) | 7.6 | 7.8 (+5.4 %) | 8.2 | 50 % |
| Gemma-3-12B (GeGLU) | 7.4 | 36.0 | 185 | 946 | – | none; TEAL 807 at 40 % |

Readings: (a) it agrees with this repository's E1 (*host*: exact-product ranking holds 60 % on Mistral and
Qwen2.5, 70 % fails), with the proxy costing about 10 points on Mistral; (b) the family order is Qwen > Mistral >
Llama-3, the same as memo-01's TEAL tables; (c) **the GeGLU family is not a neuron-skip subject as published**
(the paper warns Gemma's PPL is not comparable across families; the within-model blow-up is what I read). (d)
Prox was not run on a quantised 4-bit body in a table (one figure for Qwen3-8B, no numbers in the text).

### 1.2 Kernels and runtimes

| work | date, how read | claim (*published*) | for us |
|---|---|---|---|
| ik_llama.cpp CPU comparison ([discussion 164](https://github.com/ikawrakow/ik_llama.cpp/discussions/164)) | 2024-12-24; page read | M2 Max, 4 threads, Llama-3.1-8B, tg128 tok/s, ik fork: Q4_0 23.43, IQ4_XS 23.36, Q4_K_S 22.49, Q2_K_S 23.49, **Q3_K_S 15.29**, IQ3_XXS 8.54, IQ3_S 6.61 (stock llama.cpp: 23.89, 16.03, 17.55, 13.91, 11.10, 6.26, 5.34) | The best published NEON kernels put 3-bit types at 0.65 x (Q3_K) and 0.28 to 0.36 x (IQ3) of Q4_0's token rate with fewer bytes, so at least 1.5 x per weight. IQ4_XS (4.25 bpw) is at parity. This prices the lead's "fast sub-4-bit kernel" lever: §4 |
| llama.cpp | 2026; search summary | repack work since the base: Q4_K dotprod gemm (#17494, merged 2025-11-27), Q6_K dotprod (#19360), Q5_K i8mm (#18860), Q4_K gemv (#15719, open). No Q4_0 gemv change, no activation sparsity, no Q3_K/IQ3 arm64 repack found | Nothing to carry for Q4_0. Q4_K/Q6_K repack matters only for a K-type served file or head |
| T-SAR (DATE 2026, [paper 705](https://past.date-conference.com/proceedings-archive/2026/DATA/705.pdf)); TU Delft thesis 2026 | search summary | T-SAR: ternary LUT GEMV on x86 AVX2, 1.1 to 86 x over T-MAC and bitnet.cpp. The thesis: the LUT advantage is conditional and narrows when tables leave the fast cache | Ternary or 1 to 2 bit only. No 2026 4-bit LUT kernel found. T-MAC on Q4_0 stays closed (memo-05 §3). **A LUT kernel is the natural kernel for a 1-bit proxy** (§2.2, T6) |
| Stage-level NPU study ([arXiv 2605.27435](https://arxiv.org/html/2605.27435)) | 2026-05; HTML read | Snapdragon 8 Gen 3, llama.cpp b7588, Q4_0 Llama-3.1-8B and Qwen3-8B: the Hexagon v75 is 1.55 to 1.67 x faster on decode MUL_MAT but only 1.05 to 1.2 x end to end | Confirms the Hexagon closure (memo-03) on a chip three generations newer |
| Qualcomm QMX post ([2026-04](https://www.qualcomm.com/developer/blog/2026/04/llama-models-acceleration-on-cpu-qmx)), Arm SME2 learning path | search summary | matrix extensions speed up llama.cpp on a Snapdragon 8 Elite Gen 5 | Not on the SD888 (no SME2, no i8mm) |
| ExecuTorch + KleidiAI, MNN | search summary | 1 B to 3 B models only; no 7 B phone decode number found | Nothing to carry |

### 1.3 Lossless multi-token methods (greedy output identical)

| work | date, how read | claim (*published*) | for us |
|---|---|---|---|
| Token Recycling ([arXiv 2408.08696](https://arxiv.org/abs/2408.08696v3)), ACL 2025 | search summary | training-free, under 2 MB, about 2 x on GPUs, with a draft **tree** checked by tree attention | A tree is a batch of tens of tokens: R20's regime (0.48 to 0.76 x, *device*) |
| Cacheback ([arXiv 2511.21699](https://arxiv.org/abs/2511.21699)), EMNLP 2025 | abstract only | drafts from LRU n-gram tables | same family as campaign 1's n-gram drafting (lost, *host*) |
| UniSpec ([ACL 2026](https://aclanthology.org/2026.acl-long.285/)) | abstract only | training-free, "device-calibrated" draft size, up to 2.6 x over other training-free methods; hardware not named | The calibration idea is right for us: draft size 1 |

The only regime where any of these can win here (*estimated*): one extra candidate verified in the same pass
(batch 2) costs max(B ÷ r, 2W ÷ c). On a byte-bound dense body at r 24 that is 1.19 x the pass time for 1 + a
tokens. Break-even a = 0.17; at a = 0.4 a 5.8 G body goes from 7.0 to 8.3 tok/s and a 7.1 G body from 5.7 to 6.7.
memo-05 T4 found gemm equal to gemv per element (*host*), so batch 2 can be bit-identical. But: every stack that
approaches 12 is compute-bound, and there batch 2 costs 2 x for 1 + a tokens. **It does not stack with sparsity
and it is not on the path to 12.** It stays under memo-03's rule (reopen only if R69 reads r under 28 and no
compute-cutting lever survives).

### 1.4 LM head

| work | date, how read | claim (*published*) | for us |
|---|---|---|---|
| HNSW output head ([arXiv 2608.27460](https://www.alphaxiv.org/abs/2608.27460.md)) | 2026-08; summary page | approximate MIPS over the output rows on a CPU: head over 12 x faster at batch 1, recall@2 97.3 to 99.9 %, index +5 to 10 % bytes; models of 0.27 to 1.7 B only | Approximate, and unretrieved logits are −inf: infinite PPL on a miss, the same failure as campaign 2's hard vocabulary subset (closed). Random graph walks over a 0.3 to 0.5 GB head are cache misses, not streams |
| FlashSampling ([arXiv 2603.15854](https://arxiv.org/pdf/2603.15854)) | search summary | exact, but computes every logit | nothing |
| VocabTrim ([arXiv 2506.22694](https://arxiv.org/html/2506.22694v2)) | search summary | prunes the **drafter's** head | nothing without a drafter |

No exact bound-certified argmax method was found (one search). I do not propose one: norm bounds on rows are
loose in 3,584 to 4,096 dimensions (*estimated*, not measured). **Campaign 2's two-stage head (a Q4_0x4 screen of
all rows, exact rows for the top 2,048; +0.01 % PPL, *host*) is still the right tool** and is model-generic. What
it buys depends on the model (§3, property 5): Qwen2.5 about 0.14 GB of 4.12 (*estimated*: a Q6_K head read at
4.5 bits), Mistral 0.03 GB.

## 2. Lever stack for 12 tok/s

### 2.1 Three walls (*estimated* from R69's pre-registered bands)

| wall | line at 12 tok/s | a 5.8 G-multiplied Q4_0 body (3.26 GB) | a 7.1 G body (4.04 GB) |
|---|---|---|---|
| compute | W ≤ 3.04 GB-equivalent at c 40.4 (3.79 at four cores) | 3.26: over by 7 % | 4.04: over by 33 % (7 % with four cores) |
| bytes | B ≤ 1.81 GB at r 24, 2.64 at r 35 | over by 80 % / 23 % | over by 123 % / 53 % |
| residency | body + side files ≤ C − 0.15; C central 3.1 GB | 3.26: does not fit | 4.04: does not fit |

Dense, no lever: 7.0 tok/s (5.8 G body) and 5.7 (7.1 G) at r 24; 9.9 and 8.1 at r 35; compute caps them at 11.3
and 9.3.

### 2.2 Whole-neuron skip as published (Prox), with this repository's kernel costs

Per FFN, at neuron sparsity s, proxy at a quarter of the `gate` + `up` bytes and priced at `sdot` cost per byte
(unbenched), row kernel cost k_r on kept rows, K1 column kernel k_c on `down`:
- bytes factor = 2/3 x ((1 − s) + 0.25) + 1/3 x (2/18 + 16/18 x (1 − s));
- compute factor = 2/3 x ((1 − s) x k_r + 0.25) + 1/3 x (1 − s) x k_c.

| case | s | k_r, k_c | FFN bytes | FFN compute |
|---|---|---|---|---|
| central | 0.5 (Prox's Mistral level) | 1.8, 1.8 (Y1b cold on short rows was 1.45 to 3.3, *host*; K1 1.55 to 2.2, *estimated* device) | 0.69 | **1.07** |
| optimistic | 0.6 (E1's exact-product level) | 1.5, 1.5 (both kernels at their *host* hot cost) | 0.59 | 0.77 |
| pessimistic | 0.5 | 2.1, 2.1 | 0.69 | 1.22 |

**At the central costs the lever saves 24 % of the token's bytes and no compute.** It helps while the token is
byte-bound and stops helping exactly where 12 tok/s lies. Token rates (*estimated*):

| body (multiplied G, GB at Q4_0, FFN share) | central: r 24 / r 35 / r 35 + 4 cores | optimistic: r 24 / r 35 / r 35 + 4 cores |
|---|---|---|
| Yi-1.5-6B class (5.80, 3.26, 0.75) | 8.9 / 10.8 / **12.6** | 9.8 / **13.4** / **13.8** |
| Llama-2-7B class (6.61, 3.72, 0.76) | 7.9 / 9.6 / 11.3 | 8.7 / 11.9 / **12.3** |
| Mistral-7B-v0.3 (7.11, 4.04, 0.785; *host* H1) | 7.4 / 8.8 / 10.5 | 8.2 / 11.2 / 11.6 |
| Qwen2.5-7B (7.07, 4.12, 0.78; *host* H1) | 7.3 / 8.7 / 10.3 | 8.0 / 10.9 / 11.4 |
| Llama-3.1-8B (7.50, 4.5, 0.75) | 6.6 / 8.0 / 9.4 | 7.3 / 10.0 / 10.3 |
| Qwen3-8B (7.57, 4.4, 0.72) | 6.7 / 8.2 / 9.5 | 7.3 / 10.1 / 10.4 |

`down` columns alone (s 0.6, k_c 1.8; E1 fits on both models, *host*): 7.9 / 11.3 on the 5.8 G body, 6.5 / 9.3 on
Mistral at r 24 / r 35. Residency is not in these rows: every one of them needs C of 3.8 to 5.2 GB (body plus
proxy), which R69's band gives a small chance.

### 2.3 New: skip neurons in groups of four rows of the stock repack (k = 1.0)

- **What.** The phone's Q4_0 kernel reads `gate` and `up` as groups of four interleaved rows. Skipping a whole
  group needs no new SIMD kernel: the outer loop takes a list of kept groups. Kept weights cost exactly k = 1.0,
  skipped groups save all their bytes including the scales, and at an empty skip list the result is bit-identical
  by construction. (memo-02 §8 measured the same shape of kernel on input blocks at 0.93 x per touched weight,
  bit-identical, *host*.)
- **Row order is free.** Reordering the rows of `gate` and `up` (and the matching columns of `down`) is exact for
  the row side (memo-04 work §3.1: "that side is bit-exact"). So the groups of four can be chosen offline, by a
  committed script from a calibration text: neurons whose |SiLU(g) x u| rise and fall together go into one group.
  Only the `down` side then needs K1 (columns in the permuted order) or stays dense.
- **What decides it.** The share of groups that can be skipped inside the budget. With independent neurons at
  s = 0.6 only 13 % of groups are fully skippable; a group rule (skip when the group's largest proxy score is
  under a threshold) skips more but drops live neurons. memo-05 T5 found no stable hot set (h 0.002 to 0.005,
  *host*), which says a static order alone will not do it; co-activation is a different statistic and is not
  measured. My prediction, written before any data: file order 20 to 35 % of groups inside +4.5 % on both models,
  clustered order 30 to 45 %, P(≥ 40 % on two models) about 0.3.
- **Price at 40 % of groups** (*estimated*; proxy bytes 0.25, proxy compute 0.2 of `gate` + `up` with a LUT
  kernel, `down` through K1 at s 0.6, k 1.8): FFN bytes 0.72, FFN compute 0.77. The 5.8 G body: 8.7 tok/s at
  r 24, 12.2 at r 35 on three cores. Mistral: 7.2 and 10.2 (still byte-bound at r 35, so a 4th core adds nothing there). Same ceiling as §2.2's
  optimistic case, but without betting on a cold row kernel at 1.5.
- **Class.** Numerics change (declared), one budget, same rule at both batch settings. Side files: the
  permutation (exact), the proxy (lossy, ranking only) and thresholds, all under the sidecar regime.

### 2.4 The stack I would bet on, in build order (*estimated* tok/s on the 5.8 G body / on Mistral, r 24 → r 35)

| # | lever | class | generic? | cumulative tok/s |
|---|---|---|---|---|
| 0 | campaign 2's state (pin, exec map, touch, kv clear), dense | bit-identical | yes | 7.0 → 9.9 / 5.7 → 8.1, **if resident** |
| 1 | no-copy layout (Y1a or Y1b) or a served file that fits C | bit-identical (Y1c: declared) | yes | makes row 0 real; no gain by itself |
| 2 | two-stage head (campaign 2) | numerics, +0.01 % (*host*, c2) | yes; gain scales with vocab x width | +0.1 to +0.3 on large-vocab models, about 0 on 32 K to 64 K vocabularies |
| 3 | `down` columns, K1 at TR 128, s 0.6 | numerics | yes (E1 fits both, *host*) | 7.9 → 11.3 / 6.5 → 9.3 |
| 4 | whole-neuron skip of `gate` + `up` rows: groups of four (§2.3) or proxy rows (§2.2) | numerics; shares the budget with 3 (same neuron set, so about the same PPL) | to be screened per family | 8.7 to 9.8 → 12.2 to 13.4 / 7.2 to 8.2 → 10.2 to 11.2 |
| 5 | a 4th big core, and A55s on the proxy and the head (weighted chunks) | bit-identical | yes; closed for MoE (R18), open for dense | only when compute-bound: up to x 1.13 to 1.25 on W; 13.8 / 11.6 at r 35 |
| – | GPU compute lane (R69 Rule D) | numerics | – | the lead's lever; not priced here |
| – | batch-2 lookahead | bit-identical if gemm = gemv holds on the phone | yes | off the path (§1.3) |

So: **12 needs rows 0 to 4 on the smallest body with r ≥ about 30, or rows 0 to 5 with r ≥ 35 on a 6.6 G body; a
7.1 G body tops out near 10 to 11.6 in the best case I can price.**

## 3. Model properties that matter, for the choice (ranking by *estimated* tok/s per unit of the property)

1. **Multiplied weights per token = total parameters − lookup-only embeddings.** Around 12 tok/s each 0.5 G of
   multiplied weights is about 7 ms, about 1 tok/s. Under the definition "6.0 to 8.5 B, embeddings included", a
   **large untied vocabulary is an advantage**: with total P and embedding V x d, an untied model multiplies
   P − V x d, a tied one multiplies P. Examples (*host* H1 headers): Qwen2.5-7B 7.616 → 7.07 G; Mistral 7.248 →
   7.11 G; Yi-1.5-6B 6.061 → 5.80 G. This runs against reading selection criterion 3 ("LM-head bytes") as
   "smaller vocabulary is better": criterion 1 (tok/s) already has the right sign, and the head is the one tensor
   the two-stage screen makes cheaper. The ideal subject is a model just above 6.0 B total with a 100 K+ untied
   vocabulary; I have not searched for it (Bonnie's pool).
2. **Neuron-skip tolerance**, measured on the host per candidate with E1's cell (`down` input 50 / 60 / 70 % at
   `-b 512` and `-ub 1`). Published order: Qwen3 ≥ Qwen2.5 ≥ Mistral > Llama-2 > Llama-3; GeGLU (Gemma) fails
   (Prox, memo-01 tables). A Llama-3-family body pays twice: largest and least tolerant.
3. **FFN share of the bytes.** Neuron skipping touches only the FFN. Qwen2.5 0.78, Mistral 0.785, Yi 0.75,
   Qwen3-8B 0.72 (*host* H1). Strong GQA helps here (Qwen2.5's attention is 11 % of the bytes, Mistral's 19 %),
   not through KV, which is 0.3 to 0.6 % of the token at the metric's context (memo-05 work §2).
4. **Quantisation robustness under 4.5 bpw.** Residency, not speed, forces it: at C 3.1 GB no Q4_0 body of 6 B+
   fits. But the kernel cost of 3-bit types is at least 1.5 x per weight in the best published NEON code (§1.2),
   so a 3-bit file spends the compute line it cannot afford. A mixed file is the form that may work: 3-bit K-type
   for `gate` and `up` only, where kept rows are 40 to 50 % (so 1.5 x on 0.4 to 0.5 is still under 1.0), 4-bit
   elsewhere. That is a served-file choice (S3) and wants a per-candidate ladder (H2).
5. **Head.** Bytes are vocab x width at the head's type. With the two-stage head the multiplied cost is a 4-bit
   screen plus 2,048 exact rows. A tied head (Gemma style) cannot be re-typed apart from the embedding.
6. **Activation function.** ReLU or ReLU² would make skipping exact and bit-identical, and nothing else on this
   list comes close to that. None found among stock dense transformers of 6.0 to 8.5 B (searched 2026-11-18 and
   again today; RWKV-7 is excluded by the owner). **Unscreened:** non-gated GELU models. Falcon-7B-instruct
   (hidden 4,544, 32 layers, FFN 18,176, GELU, vocabulary 65,024, multi-query attention; *published* config) and
   MPT-7B are in the base tree (`falcon`, `mpt` in `llama-arch.cpp`, *host* read). A GELU output is bounded in
   [−0.17, 0] for every negative pre-activation, so most neurons may sit near zero; no published count for these
   models was found. They are 2023 models (weaker subjects by criterion 4) and their graph differs from H3's tool.
   One host count (T4) says whether to price them.
7. **Depth against width** at equal size: fewer, wider layers mean fewer thread syncs per token (campaign 1's
   lesson) and longer rows. Second order; inside O.

## 4. Challenge

### 4.1 To the lead's corner (clyde-replan §1: a 3.1 to 3.7 bpw file, column sparsity d 0.60, a low-bit column kernel at k ≤ 1.3 to 1.5)

- **The kernel it needs is further than the text suggests.** k is counted against Q4_0 `sdot` per weight. The best
  published NEON kernels for 3-bit types run dense at 1.5 x (Q3_K) to 3 x (IQ3) per weight (ik_llama.cpp, M2 Max,
  *published*; our own Q3_K reads half the `sdot` rate, *device*). A column kernel pays its own factor on top (K1:
  1.53 to 1.64 on Q4_0, *host*). The product is about 2.3 to 3, against a line of 1.3 to 1.5. I put this corner
  under 0.03 per subject.
- **Column sparsity cuts the wrong thing for a K-type.** Q3_K and Q4_K carry 6-bit sub-block scales per 16
  weights along the row; a column layout for them is not the exact re-layout that Q4_0 has (memo-01 work §3.1).
  It would be a new format with its own identity question.
- **The corner I would put beside it:** a mixed served file (property 4) with whole-neuron row skipping, which
  needs no column layout for `gate` and `up` at all. It is §2.4 with smaller bytes. It still tops out near 12 on
  the smallest body only.
- The lead's prior of 0.05 to 0.15 for all three: mine is at most 0.02, because the three walls are independent
  of family and two of the three subjects will have 7 G bodies unless the pool is unusual.

### 4.2 To Bonnie's pool (`bonnie-pool.md`, read 2026-11-21 after §0 to §3 and §4.1 were written)

The seven checks were fixed before I read her draft (first version of this section): multiplied not published
size; weight line beside byte line; residency including side files; FP16 sparsity marked unscreened; 3-bit kernel
cost at least 1.5 x; head at its real type with the two-stage screen; tied against untied.

**Where she is right and I adopt it.**
- Her W per token matches mine where both exist (Yi 5.80, Mistral 7.11, Qwen2.5 7.07, Llama-2 6.61). Checks 1, 5
  and 7 pass. Her pool is far wider than my guesses in §3, and it settles property 6: no eligible ReLU-family
  model, and the 2023 GELU Falcon-7B and MPT-7B are not in her pool (not instruct-eligible or not listed; T4's
  GELU count is dropped unless the lead wants them).
- **Printed against steady.** She prices the benchmark's printed value as 0.95 x steady; my tables and the lead's
  lines (5.4 G weights, B12) are steady. With her factor the weight line is 5.17 G, not 5.4 G, and every tok/s in
  my §2 is 5 % high: the 5.8 G body's corners read 12.0 (central, r 35, four cores) and 12.7 / 13.1 (optimistic),
  Llama-2's 12.3 becomes 11.7. **The three files must use one convention; hers is the conservative one and has a
  measured origin (memo-03 §9), so I take it.** My P(12) on a 5.8 G body drops from 0.10 to about 0.07.

**Where I disagree (both positions stand).**
1. **Residency is not in any price (check 3).** Criterion 4 (file + KV + app ≤ 6.0 GB) passes all 28, but R69's
   pre-registered central C is 3.1 GB, and the smallest Q4_0 body is 3.33 GB. At the central band no row of her
   table is a resident configuration; her text says residency is open but the ranking does not carry it. I ask
   for one more column: the largest served type that fits C − 0.15 at C 3.1 and 3.4, and the price at that type.
   For most candidates it is a 3-bit type at half the `sdot` rate, which is her own 5.5 tok/s row.
2. **The served-quantisation table has no header row**, so its seven columns cannot be read (the text implies
   Q4_0 first, Q3_K_M fourth, criterion 4 last). To be fixed before the merge.
3. **Reason 3 for Llama-2 rests on the favourable one of two disagreeing sources.** TEAL's own table has Llama-2
   at +3.0 % for 40 %; LaRoSA's re-run of TEAL has it at +15.4 % already at 25 % (memo-01 work §2.2, *published*).
   Llama-2 and Granite-7B have 32 key-value heads, so 25 % of each block is attention, where a neuron lever does
   nothing. I would not rank on a published tolerance for any unscreened model; s = 0.30 to 0.40 is an assumption
   for 26 of 28 and the top five contains no screened model.
4. **Order of the third subject.** Her order is the sum of the lever tops at r 19 and r 35. At 12 tok/s the weight
   line binds (her own reading: every candidate is compute-bound at r 35), so I would order by W: Yi 5.80,
   ChatGLM3 5.98, **DeepSeek-LLM-7B 6.49**, Llama-2 / Granite-7B 6.61, OLMo 6.68, Salamandra 6.72, calm2 6.74.
   DeepSeek has 30 layers (fewer thread syncs) and an untied 102 K vocabulary; her own r 35 column already puts it
   above Llama-2 (10.4 against 10.2). Its head is the part a two-stage screen makes cheaper.
5. **"A large head costs twice" is half right (check 6).** It is Q6_K and not sparsified, but campaign 2's
   two-stage head reads it at 4.5 bits (+0.01 % PPL, *host*, c2). Salamandra: 0.86 → about 0.59 GB, about
   +0.3 tok/s at r 19 (*estimated*); it does not reach the top five, but the 21 % penalty is not a property of
   the model. The compute side of the head stays, so her order by W is unaffected.
6. **One c, no band (check 2).** c = 69.9 G weights/s is 39.3 GB/s; the pre-registered centre is 40.4 (band 35 to
   45) and the four-core row is missing. At c 35 the "k ≤" column for Yi at s 0.40 falls from 1.48 to about 1.3;
   at four cores it rises to about 1.85 (*estimated*, scaling her line by c). The kernel requirement is as
   sensitive to c as to s, and R69 measures c.
7. **Disk.** 1.5 to 2.2 GB free (*host*, her `df`) blocks every screen in her §6 and my T1 to T4. This is the
   first thing for the lead to clear; no candidate file fits.

**Net.** I agree with ranks 1 and 2 (Yi-1.5-6B, ChatGLM3-6B, with her flags: `chatglm` graph, licence). For the
third I put DeepSeek-LLM-7B ahead of the Llama-2 shape and ask that the choice wait for T4 on ranks 1 to 7 if disk
allows. Neither of us prices 12 on the third subject: her line needs k ≤ 1.3 to 1.4 at r ≥ 31 to 35, mine needs
four cores and r ≥ 35 at k 1.5.

## 5. Test list (host, for Tom) and try list (device, later)

Each test says what it costs and what closes it. Model-generic tools; every screen on at least two families.

| # | test | cost | closes / opens |
|---|---|---|---|
| T1 | **Group-of-four neuron rule.** E1's tool with the rule "zero all four neurons of a group when the group's largest \|SiLU(g) x u\| is under a per-layer train threshold". Orders: file order; a clustered order from wikitext-2 train (greedy grouping by correlation of \|p\|). Cells 20 / 30 / 40 / 50 % of groups, 40 chunks, `-b 512`, then `-ub 1` on the frontier cell | one calibration pass and 8 cells per model, two models; about E1's cost twice | under 30 % of groups inside +4.5 % on either model in the clustered order → §2.3 closed. At least 40 % on two models → build the skip-list path (bit-identical at an empty list) |
| T2 | **Proxy ranking (memo-05 E3, never run).** Rank neurons by a 1-bit and a 2-bit copy of `gate` and `up` (sign and per-row scale) instead of the exact product; `down` input cells 40 / 50 / 60 % | 6 cells per model | 1-bit under 40 % inside +4.5 % on either model → proxy levers (§2.2, and T1's proxy form) closed; Prox's table predicts Mistral about 50 %, Qwen about 55 to 60 % |
| T3 | **Weight-norm score (memo-04 work T6, never run).** Element rule with \|x_j\| x column norm from the GGUF, no weight transform; at the frontier cells and 5 points above | 4 cells per model | moves no frontier by 5 points on either model → dropped |
| T4 | **Family screen for the pool.** E1's `down`-input cell (50 / 60 / 70 %) and H3's uniform cell on every candidate Bonnie keeps, on the served-type stand-in; for a non-gated GELU candidate, first a count of \|a\| under 0.01 / 0.05 / 0.17 at fc2's input | per candidate: one download (3 to 5 GB, deleted after), about 8 cells; the GELU graph needs a tool change | gives selection criterion 2 a number per candidate. A candidate under 50 % `down`-input sparsity is marked "no neuron lever" |
| T5 | **Per-type cost on the host, before the device rows of S2.** ns per weight of the stock gemv for Q4_0x4, Q4_K, IQ4_XS/IQ4_NL, Q3_K, IQ3_S at one thread on the four shapes; and ik_llama.cpp's Q3_K kernel read for what it does differently | a microbench; no model | a 3-bit type above 1.3 x Q4_0 per weight in the best available kernel → the all-3-bit corner is closed on the host; only the mixed file (property 4) goes on |
| T6 | **1-bit proxy kernel.** `TBL`-based sign-weight gemv over int8 activations against `sdot` per weight, hot and cold | a microbench | above 0.35 x `sdot` per weight → proxy compute is priced at that and §2.2 / §2.3 repriced; if the repriced FFN compute factor is ≥ 1.0 at T1/T2's frontier, the proxy levers close |
| T7 | **Multiplied-weight inventory** with `tools/c3_gguf_inventory.py` on every pool candidate: multiplied weights, tied or untied, FFN share, head type | header reads only | feeds criteria 1 and 3; no kill |
| T8 | **Free-draft acceptance (only if R69 reads r under 28).** Count on the official prompt and the fixed prompt set how often a draft-size-1 guess (last top-2, n-gram cache) equals the greedy token | one logging run per model | a under 0.30 on either model → lookahead stays closed |

Try list (device, after the tests that gate them; each needs its own pre-registration):

| # | lever | gate | class | *estimated* gain |
|---|---|---|---|---|
| Y8 | Skip-list path for groups of four rows in the stock repack, with a row permutation side file | T1, T2, T6 | bit-identical at an empty list; numerics above | §2.3: +1.7 to +2.9 tok/s over row 3 of §2.4 |
| Y9 | Mixed served file: 3-bit K-type on `gate` + `up`, 4-bit elsewhere, with row skipping | T5, H2 ladder, S3 | served-file choice; then as Y8 | residency: body about 0.3 to 0.4 GB smaller on a 5.8 G body; speed as Y8 |
| Y10 | X1 + 3 A78, and weighted chunks that give the A55s the proxy pass and the head screen | R69 (four-core row), S2 | bit-identical | up to x 1.13 to 1.25 on compute when compute-bound; 0 when byte-bound |
| Y11 | Two-stage head from campaign 2 on each subject | none (existing code) | numerics, tiny | +0.1 to +0.3 tok/s on 100 K+ vocabularies |
| Y12 | Batch-2 lookahead | T8 and r under 28 | bit-identical if identity holds | +0.5 to +1.3 tok/s while byte-bound; negative when compute-bound |

## 6. Not checked, and where I may be wrong

- Prox's last 9 K characters (appendix tail) and its quantised-model figure were not read; its PPL protocol
  (context 2,048, validation split) is easier than ours.
- The ik_llama.cpp table is an M2 Max at four threads; whether Q4_0 was byte-bound there is not stated. If it was,
  the 3-bit per-weight costs are higher than 1.5 x, not lower.
- The proxy's compute cost (0.2 to 0.25 of `gate` + `up`) is a guess until T6.
- Falcon-7B's parameter count (about 6.9 B, *estimated* from its config) and whether its head is tied were not
  checked; its config lists `num_kv_heads` 71 beside `multi_query` true.
- Llama-2-7B's row in §2.2 uses 6.61 G multiplied at Q4_0 (*estimated*, no header read).
- UniSpec, Cacheback, SGA-Sparse: abstracts only. Token Recycling, T-SAR, the Delft thesis, QMX, ExecuTorch, MNN,
  llama.cpp pull requests: search summaries only.

## 7. Sources (read 2026-11-21)

- Prox: <https://arxiv.org/html/2607.27591> (2026-07). HTML, first 100 K characters.
- WINA: <https://arxiv.org/html/2505.19427> (2025-05; ICLR 2026). HTML.
- R-Sparse: <https://arxiv.org/html/2504.19449v1> (ICLR 2025). HTML.
- SGA-Sparse: <https://arxiv.org/abs/2611.11223> (2026-11). Abstract.
- SpAx: <https://arxiv.org/html/2610.02598> (2026-10). Search summary.
- ik_llama.cpp CPU comparison: <https://github.com/ikawrakow/ik_llama.cpp/discussions/164> (2024-12-24). Page.
- llama.cpp pull requests #17494, #15719, #19360, #18860: search summary.
- T-MAC: <https://arxiv.org/html/2407.00088v2>; T-SAR: <https://past.date-conference.com/proceedings-archive/2026/DATA/705.pdf> (DATE 2026). Search summaries.
- Stage-level NPU study: <https://arxiv.org/html/2605.27435> (2026-05). HTML.
- Qualcomm QMX: <https://www.qualcomm.com/developer/blog/2026/04/llama-models-acceleration-on-cpu-qmx> (2026-04). Search summary.
- Token Recycling: <https://arxiv.org/abs/2408.08696v3> (ACL 2025). Search summary.
- Cacheback: <https://arxiv.org/abs/2511.21699> (2025-11). Abstract.
- UniSpec: <https://aclanthology.org/2026.acl-long.285/> (ACL 2026). Abstract.
- HNSW output head: <https://www.alphaxiv.org/abs/2608.27460.md> (2026-08). Summary page.
- FlashSampling: <https://arxiv.org/pdf/2603.15854>; VocabTrim: <https://arxiv.org/html/2506.22694v2>. Search summaries.
- Falcon-7B-instruct config: <https://huggingface.co/TheBloke/Falcon-7B-Instruct-GPTQ/blob/main/config.json>. Search summary.
- "ReLU strikes back" (Falcon and MPT use GELU): <https://arxiv.org/pdf/2310.04564v1>. Search summary.
