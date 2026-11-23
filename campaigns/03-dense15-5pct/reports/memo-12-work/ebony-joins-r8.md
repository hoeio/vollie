# c3 memo-12 work (Ebony, round 8): where two runs of the same model first differ, what else the one-scale rule touches, review of Bonnie r12 to r16

Date: 2026-11-24. Author: Ebony (`claude-fable-5-2`). Not committed; the lead commits. No verdict here.
Follows [ebony-joins-r7.md](ebony-joins-r7.md). Reviews [bonnie-joins-r12.md](bonnie-joins-r12.md) to
[bonnie-joins-r16.md](bonnie-joins-r16.md): r16 §0 and §1 read in full; r12 to r15 in their "answer first" sections.

Labels as before. Triples are Yi / Llama-2 / DeepSeek. New *host* numbers: `ebony_batchdiff.py` (new) on Llama-2,
64 tokens, two threads at low priority, on three library sets. No Metal, no phone.

## 0. Answer first

1. **The 0.0007 divergence between `-ub 1` and `-b 512` inside one build comes from the attention op, not from any
   weight matmul.** *Source*: ggml's CPU flash-attention has two arithmetic paths with an F16 cache. With 64 or
   more queries it runs a tiled path in float. Below that it runs a per-query path that converts Q to half and
   accumulates the values in half. *Host*: layer 0's q, k and v matmul outputs are bit-equal between the two
   settings; the first node that differs is the attention output, by 1.4e-3 of its largest value.
2. **Correction to my r6 and r7: the "repack against KleidiAI" divergence I measured on Llama-2 was not the Q4_0
   kernels either.** With the same tree, the repack build and the KleidiAI build give **bit-equal** q, k and v
   outputs in layer 0 (12,288 values, real activations). The first difference is again the attention output
   (7e-5), and the cause is the test build's flags (it was built with BLAS on and no instruction-set flag), not
   KleidiAI. Bonnie's r10 claim about the Q4_0 arithmetic is supported; my r7 "disagree on the size" is withdrawn.
3. **A general property worth one sentence in every numerics reading:** a difference of 7e-5 at one attention
   output and one of 1.4e-3 both end as the same noise at the logits (rms 0.05 to 0.07, KL divergence about 1e-3,
   per-chunk perplexity ±0.5 %, no direction). Rounding of activations to int8 in every matmul amplifies any
   change to that floor. **So a KL divergence or a per-chunk comparison cannot tell a small numerics change from a
   negligible one.** Only the mean over many chunks, or a node-by-node comparison, can.
4. **The one-scale activation rule touches the head only** (my question 4). Each served file has exactly one
   tensor that is not Q4_0 or F32: `output.weight`, Q6_K. The attention op is ggml's own in both builds;
   KleidiAI has no part in it.
5. **On the phone the attention takes a third path, which no host run has used:** the app sets the cache to Q8_0
   (Bonnie r15), and with a quantised cache the tiled float path is never taken. Q is quantised to int8 in blocks
   of 32, the score is an integer dot, the values are converted to float and accumulated in float. That path is
   the same at any batch size.
6. **Bonnie r16 row 9 is right** (checked in the source): the app refuses the faster layouts for these files,
   and only a kit's `_repack` forces them. One addition: the rule counts copied bytes, so the mapped side file is
   what makes the fast path legal under the app's own rule. §3.
7. **Her stripped stacks at K 300 (10.4 to 11.1 tok/s): confirmed by my model within 0.3.** With her r15
   attention terms added, mine reads **10.0 to 10.6**; DeepSeek binds at about 10.0. §4.

## 1. Where two runs of the same model first differ (my question 3)

### 1.1 Source

`lab/llama.cpp-base/ggml/src/ggml-cpu/ops.cpp`, `ggml_compute_forward_flash_attn_ext_f16` (lines 9518 to 9650);
`common.h` lines 9 to 10.

| condition | path | arithmetic with an F16 cache |
|---|---|---|
| queries in the batch ≥ `GGML_FA_TILE_Q` = 64, Q is F32, K and V are F32 or F16 | `…_tiled` | Q stays float; K and V tiles are converted to float; `simd_gemm` in float |
| otherwise | `…_f16_one_chunk`, one query at a time | Q converted to the cache's dot type (half); score by `vec_dot` in half lanes; values accumulated in **half** (`VKQ16`, `ggml_vec_mad_f16`), converted to float at the end |
| one query and 512 or more cache rows | a split variant of the per-query path | the same arithmetic in chunks |

- The weight matmuls do not depend on the batch: ggml's repack gemv and its four-row kernel both run FMUL of the
  two scales, SCVTF #4, FMLA per block (`arch/arm/repack.cpp` lines 230 to 290 and the assembly from line 1990),
  and both quantise activations with the same rule (`ggml_quantize_mat_q8_0_4x4` and `quantize_row_q8_0`: nearest
  even, reciprocal of the stored scale).
- The model's default is flash attention "auto"; on the host it resolved to on: the graph holds a
  `FLASH_ATTN_EXT` node per layer with an F16 cache (*host*, my node listing). This fills the "not in the logs"
  cell of Bonnie r16 row 12.

### 1.2 Measurement (*host*)

`ebony_batchdiff.py`: Llama-2, the first 64 tokens of wikitext-2 test. It records the last token's column of every
two-dimensional float node (553 nodes) in one batched decode, and again after 63 one-token decodes, and compares
bit for bit. A control (the batch run twice) differs in one node, the embedding lookup's scratch output, which is
ignored.

| comparison | nodes that differ | first node that differs | layer 0's q, k, v outputs | logits at the last position |
|---|---|---|---|---|
| one batch of 64 against 64 single decodes, repack build | 543 of 553 | `kqv_out-0` (the attention output), 1.4e-3 of its largest value | **bit-equal** | rms difference 0.054, same top-1 |
| repack build against the KleidiAI build (T1b's libraries), both batched | 542 of 553 | `kqv_out-0`, 7.4e-5 | **bit-equal** | rms difference 0.071 |
| base tree's repack build against the campaign tree's (`lab/llama.cpp-c3`), both batched | **0 of 553** | – | bit-equal | identical |

- Row 1 is the half-against-float accumulation of §1.1. A half sum over a few hundred terms has a relative
  error near 1e-3, as measured.
- Row 3 is a free check that the campaign-2 patch series with every knob unset is stock, on every node.
- Row 2 isolates what my r6 comparison measured. The two builds differ in more than KleidiAI:

  | | base repack build | T1b's KleidiAI build (`lab/c3/head/build-kai`) |
  |---|---|---|
  | `GGML_CPU_KLEIDIAI` | off | on |
  | `GGML_BLAS` | off | **on** |
  | `GGML_CPU_ARM_ARCH` | `armv8.2-a+dotprod+fp16` | **empty** |

  The Q4_0 outputs are equal bit for bit while their inputs are equal. The attention op is the same source in
  both, so its 7e-5 comes from how it was compiled. **A clean KleidiAI reference needs the freeze's flags plus
  KleidiAI and nothing else.** If Tom's stock runs use `build-kai`'s configuration, their mean is still fair (the
  noise has no direction), but any per-chunk identity claim against the freeze is not.

### 1.3 The floor

| change at its source | size at the source | logits, rms difference | measured as (r6) |
|---|---|---|---|
| float order in one attention op (build flags) | 7e-5 | 0.071 | KLD 0.0009 to 0.0012, same top-1 98.0 to 98.8 % |
| half against float accumulation in attention | 1.4e-3 | 0.054 | KLD 0.0007, same top-1 98.8 % |

- Twenty times the cause, the same effect. After a few layers every matmul's int8 rounding has turned the
  difference into a handful of flipped activation steps per vector, and from there it grows to the same level.
- What follows for readings to come:
  - Tom's 0006 emulation (the lane's kernel, 6e-4 to 8e-4 at its source) will show KLD near 1e-3 and per-chunk
    changes of ±0.5 % whatever its true cost. Its **mean** is the result; I expect it inside ±0.1 % at 40 chunks.
  - "Identical per chunk" is a test of bit-equality only. Anything that is not bit-equal looks alike.
  - A real bias (the head's one-scale rule on Yi) shows as a mean shift above that noise: 40 chunks resolve
    about 0.2 % (r7 §2.2).

## 2. What else the one-scale rule touches (my question 4)

| part of the phone's path | who computes it | activations | touched? |
|---|---|---|---|
| 225 / 225 / 211 Q4_0 tensors (the body, the embedding) | KleidiAI Q4_0 chain when the copies are allowed | int8, a scale per 32 values | no |
| `output.weight`, Q6_K: the only other quantised tensor in each file (*host*, GGUF headers) | KleidiAI 8-bit chain | **int8, one scale and zero point per vector** | **yes** (r7) |
| attention | ggml's `FLASH_ATTN_EXT`; KleidiAI has no entry for it | on the phone: Q in Q8_0 blocks of 32, integer score, float value sum | no |
| norms, SwiGLU, adds, RoPE | ggml | float | no |
| F32 and F16 chains of KleidiAI | would take a `MUL_MAT` on a float weight | – | no such node exists in these graphs with flash attention on |

- So r7's bands are the whole of it: the head, on Yi and DeepSeek.
- **The phone's attention path is a separate, known gap** (Bonnie r15, r16 rows 11 and 12). What I add from §1.1:
  with a Q8_0 cache the per-query path is the only one (`kv_is_f32_or_f16` is false, line 9566), so the phone's
  attention is batch-invariant, where the host's two gate settings are not. Its error is of the Q8_0 kind
  (1 part in 127 of each block's maximum on K, V and Q), larger at its source than the half sums of the host. By
  §1.3 it will again look like the floor; whether it has a direction is what the second reference
  (`-fa on -ctk q8_0 -ctv q8_0`) measures. My band for the three subjects: −0.1 to +0.3 % (*estimated*; campaign
  1 judged it "no visible quality loss" on a model with few attention layers).

## 3. Review of Bonnie r16, row 9

Her claim: by the app's own rule a dense 7B file loads without KleidiAI or any repack; only a kit forces it.

Checked (*source*, `lab/app-c3/lib`):
- `device.dart` line 591: `bool canCopyWeights(int copyBytes) => totalRam - copyBytes >= 5000000000;`. The comment
  above it gives the reason: "an 8 GB SD888 was killed for low memory copying the all-Q4_0 Granite (3.6 GB), not
  Qwen3.5-4B (2.3 GB)".
- `assistant.dart` lines 456 to 458: `_loadedRepack = _repack ?? device.canCopyWeights(…)`, with the file's
  length as the copy size unless the file is the OLMoE one (`ModelStore.copyBytes` = 235,505,664).
  Line 467: `useExtraBuffers: _loadedRepack`.
- `research.dart` line 1281: the kit sets `_repack` per run; line 1711 clears it.
- With 7.75 GB reported and a 3.5 to 4.0 GB file, 7.75 − 3.5 = 4.25 < 5. **Refused for all three.** The limit is a
  copy of about 2.75 GB (2.3 GB if the phone reports 7.3 GB, as the comment says some do).

**Agree with the row and with what she draws from it.** Additions:

1. **KleidiAI goes with it.** KleidiAI's layout is one of the extra buffer types, so `useExtraBuffers = false`
   leaves the body on the plain mapped rows and the head on the plain Q6_K dot product. Then rows 1, 9 and 10 of
   her table are one switch on the phone, not three: copies off means no KleidiAI, no int8 head, no one-scale
   rule. **The configuration the shipped rule produces has none of r7's head shift, and is 1.5 times slower**
   (her 4.5 / 4.0 / 4.1 against 6.90 / 6.21 / 6.22, *estimated*).
2. **The rule counts copied bytes, and a mapped side file copies none.** Bonnie's side file (r7, r8) puts the
   KleidiAI layout in file-backed pages. With it the honest argument to `canCopyWeights` is the anonymous part
   only (the head if it stays in KleidiAI, 0.13 to 0.42 GB; nothing if the head knob is on). So the side file is
   not one residency option among three: **it is the only form in which the fast body is allowed by the app's
   rule on all three subjects.** That raises it above the lane in build order: a lane on top of a body the app
   would refuse is a research setting twice over.
3. **R72 agrees with the rule from the device side:** 3.4 GB of anonymous copy held and 4.2 GB did not (*device*).
   Yi's forced copy (3.36 GB) sits on the edge the rule was written to avoid.
4. **For the pre-registration of the first model-loading round** (her §1.4 proposal, agreed): the "stock
   configuration" block should state the value of `_repack` and say in words that it overrides the app's rule,
   so that a reader of the result knows which of the two stocks (4 to 4.5 tok/s or 6 to 7) the gain is against.

## 4. Review of Bonnie r12 to r15, and an independent check of her stack at K 300

| round | her claim | my position |
|---|---|---|
| r12 | sending the head to the repack type wins on budget, time (7.7 / 3.7 / 11.8 ms against 9.04 / 4.40 / 13.88) and bytes | agree on budget and bytes. The time is her estimate from OLMoE's 2,048-wide head; not checked. Her probe cell would settle it |
| r12 | C7 cannot run with the head in KleidiAI; choosing C7 forces the knob | agree with the reasoning; *source* not re-read |
| r13 | C7 as priced ("head × 0.5") is an unbuilt idea; the built two-stage head reads 0.80 / 0.91 / 0.76 of the head's bytes | agree |
| r13 | a one-bit screen misses the right token on 0.66 % of Llama-2 rows | not re-run |
| r13 | **C5 is my "up rule" from memo-07 and was never tested on any model** | **true, and it is mine to say so.** I proposed it; its screen (U4) was never run. It should not have stayed in a centre stack at × 0.93. I agree with removing it until a host screen exists |
| r13 | C5 would split the fused [gate;up] stage and cost a join per layer | agree; I had not priced that |
| r14 | 9 of 24 stack factors are measured on the phone; the device-only floor is 8.34 / 7.76 / 7.64 | agree with the method; the count not re-done |
| r15 | KV in Q8_0 is already what the app does; not a lever | agree (`assistant.dart` lines 471 to 472, read) |
| r15 | O is about 5 / 7 / 7 ms; a second position pays the attention again, 3.2 / 5.3 / 5.0 ms | plausible; not checked. Used below as given |

**The stack at K 300, recomputed** (*estimated*; my pair model `ebony_pair.py` with C5 and the head term made
parameters; C 40; width chosen per subject; tok/s):

| stack | Bonnie r13, `linear` | Bonnie r13, `pair` | mine, no jitter | mine, 60 µs of jitter per split stage |
|---|---|---|---|---|
| as committed (C5 × 0.93, C7 × 0.5) | 11.59 / 11.54 / 11.13 | 11.89 / 11.88 / 11.43 | 12.51 / 12.45 / 11.95 | 11.73 / 11.74 / 11.30 |
| C5 out, C7 as built, head knob on | 10.82 / 10.80 / 10.35 | 11.10 / 11.12 / 10.64 | 11.52 / 11.51 / 10.98 | **10.80 / 10.85 / 10.40** |
| C5 out, no C7, head in the repack type | 10.63 / 10.76 / 10.04 | 10.91 / 11.08 / 10.31 | 11.31 / 11.46 / 10.63 | 10.62 / 10.81 / 10.08 |
| C5 out, no C7, head in KleidiAI | 10.47 / 10.68 / 9.83 | 10.74 / 10.99 / 10.08 | 11.13 / 11.36 / 10.39 | 10.46 / 10.72 / 9.87 |
| second row with r15's O (5 / 7 / 7) and second position (3.2 / 5.3 / 5.0) | not in r13 | not in r13 | 11.26 / 10.99 / 10.52 | **10.58 / 10.39 / 9.98** |

- **Her 10.4 to 11.1 is confirmed.** In the stripped rows my jitter column sits on her `linear` column to within
  0.05 (0.15 in the committed row); my no-jitter column is 0.4 to 0.6 above her `pair` column, which is the
  jitter term I carry and she does not.
- The steps agree better than the levels. Committed → second row: mine −0.9 on each subject, hers −0.8.
  Second → fourth row: mine −0.34 / −0.13 / −0.53, hers −0.36 / −0.13 / −0.56.
- Her r15 terms cost −0.22 / −0.46 / −0.42 in my model, more than her "−0.25 to −0.3" on Llama-2 and DeepSeek.
- **So the stack with no unearned credit, at the K that R75 is expected to show, is 10.0 to 10.9 tok/s**, and it
  still assumes the W-position kernel, the `_mc3` cost at two columns, DeepSeek's untrained head at 0.438, and a
  body that stays resident. DeepSeek binds at about 10.0.

## 5. Lists (additions only)

| # | test | where | cost | closes when |
|---|---|---|---|---|
| E-T13 | `ebony_batchdiff.py` in its cross-build mode on any new host build before it is used as a reference | host | 2 minutes, two threads | every Q4_0 `MUL_MAT` output of layer 0 equal to the freeze build's → the build's weight kernels are the freeze's; the first differing node names what is not |
| E-T14 | the second reference: KleidiAI build with the freeze's flags (`armv8.2-a+dotprod+fp16`, BLAS off), `-fa on -ctk q8_0 -ctv q8_0`, 40 chunks, both batch settings, three subjects | host, Tom | about 30 minutes | it is the phone's path except the 8-bit kernel entry; replaces every estimate in §2 |
| E-T12 (r7) | – | – | – | **closed by §1**: the cause is the attention op's two paths |

No new try item. Build order as I now see it, for the lead: (1) the stock configuration decided and its host
reference run; (2) the side file, because the app's rule refuses the fast body without it; (3) the W-position
kernel with its exactness test; (4) the lane.

## 6. Next questions, ranked

| # | question | state |
|---|---|---|
| 1 | Tom's KleidiAI stock result against r7's bands; which flags his build used (§1.2) | waiting for his file |
| 2 | Does the phone-path attention (Q8_0 cache, per-query path) carry a direction on the three subjects? | E-T14; or a three-chunk KL run now if the laptop is free, which by §1.3 would show only the floor |
| 3 | The cost of attention on the phone's path: it is the per-query loop at every batch size, so a prefill of 150 to 190 tokens does not get the tiled kernel. How long is the prefill's attention then? | **done by arithmetic, small:** a causal prefill of 186 tokens is 186² ÷ 2 = 17,300 query-key pairs per layer, 0.55 M over 32 layers. Each pair is an int8 dot, a Q8_0-to-float conversion and a float multiply-add over 4,096 values, about 1 µs on an A78 (*estimated*; Bonnie r15's 3 to 6.5 ms per decoded token at 200 positions is the same rate). So 0.3 to 0.8 s of thread time, 0.1 to 0.3 s on four threads, against a prefill of 6 to 15 s. r4 §3 stands |
| 4 | Can the W-position kernel's exactness test use `ebony_batchdiff.py` directly (node equality at batch 2 against batch 1)? With a Q8_0 cache the attention is batch-invariant, so every node should be equal; with F16 it is too below 64 queries | host, minutes, once the kernel exists |
| 5 | DeepSeek's dump and proxy (E-T10); R75 free reads | Tom; R75 |
| 6 | Where a real bias can hide below the floor: is there any other one-sided step on the phone's path (clamps, saturating int8 in the Q8_0 cache at large K or V values)? | *source*; `ebony_hmax.py` gives the ranges (q, k, v outputs 10 to 21) |

## 7. Sources (2026-11-24)

Repository: `lab/llama.cpp-base/ggml/src/ggml-cpu/ops.cpp` (lines 8920 to 9160, 9518 to 9650), `common.h` (lines
9 to 10, 90 to 93), `arch/arm/repack.cpp` (lines 69 to 110, 230 to 290, 1970 to 2450), `repack.cpp` (lines 4684 to
4712), `arch/arm/quants.c` (`quantize_row_q8_0`); `lab/llama.cpp-base/src/llama-context.cpp` (lines 4063 to 4085);
`lab/app-c3/lib/device.dart` (lines 580 to 591), `assistant.dart` (lines 455 to 472), `research.dart` (lines 1276
to 1281, 1711), `model_store.dart` (lines 12 to 19); the three builds' `CMakeCache.txt`; GGUF headers of the three
served files; `ebony_pair.py` and `memo-10-work/bonnie_tree.py`; Bonnie's r12 to r16.
No external source this round.
