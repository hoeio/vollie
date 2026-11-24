# c3 memo-12 work (Ebony, round 16): a fresh scouting round for dense 7B decode on a bus-bound phone

Date: 2026-11-24. Author: Ebony (`claude-fable-5-2`). Not committed; the lead commits. No verdict here.
Follows [ebony-joins-r15.md](ebony-joins-r15.md).

Labels: *device*, *host*, *estimated*, *source* as before; *external* = another group's claim, with how much of it
I read. Triples are Yi / Llama-2 / DeepSeek. New *host* number: the entropy of the Q4_0 nibbles in the three
served files (§3.1). New reading of existing *device* data: the phone's OpenCL extension list (§3.3).

## 0. Answer first

1. **Nothing found moves the verdict.** Eight searches and four fetched pages turned up six things that are not
   in the excluded memos. One is a real route to the binding cost (the join), one is a fact about our own tree
   that removes a build, and four close on numbers.
2. **Already in the tree, and in no memo: upstream llama.cpp ships draft-free n-gram speculation, and the app
   library exposes it.** The base tree (`620d2225c`) has `ngram-simple`, `ngram-map-k`, `ngram-map-k4v`,
   `ngram-mod` and `ngram-cache`, with rollback, behind `--spec-type`; llamadart carries the matching settings.
   That is the "prompt lookup" source and the "copy-run" pass of Bonnie r24 and r25, already written and
   maintained. It needs a host comparison with our simulation, not a build. §2.1.
3. **New route to a small join cost: a Vulkan lane.** A token's whole sequence of GPU stages is recorded once
   in a command buffer; each stage is gated by an event the host sets; buffers stay mapped, which Vulkan
   permits. It attacks the same term as R75's recordable queues and persistent kernels by a third mechanism.
   No device evidence exists. Worth +1.0 to +1.2 tok/s if it brings K from 300 to 50 µs (*estimated*); P 0.15.
   To be considered only if R75's two cells fail. §2.2.
4. **Lossless coding of the weights is closed with a measurement:** the nibbles hold 3.77 bits of 4 in all three
   files, so the best possible saving is 5.1 to 5.2 % of the body's bytes (*host*). memo-11 had guessed 10 to
   15 %. §3.1.
5. **A free device answer to my r5 test E-T1:** the phone's extension list (R72) has `cl_qcom_dot_product8` and
   `cl_qcom_ml_ops` and does **not** have `cl_khr_integer_dot_product`. Upstream's integer prefill kernel cannot
   run on this phone as written. §3.3.
6. **External support for one of my own items:** a paper of 2026-10-31 on mobile GPUs names the KV cache's
   layout and packing as one of two causes of slow decode (abstract only). That is the direction of E-Y4
   (r10, r11).

## 1. What I excluded (read first, as asked)

| source | already covered there |
|---|---|
| `memo-11-exotic.md` | retrieval by hidden state; text datastore on flash; layer-skip self-drafts; the prefix-exit ring; draft layers; DDR steps and governors; a third bus master (Hexagon) and the small cores; temporal delta sparsity; an exact int8 GPU lane; huge pages and prefetch; small exact tables; LUT kernels; lossy texture formats; decode work during prefill; non-exact acceptance |
| `memo-11-bytes.md` | re-encoding the frozen Q4_0 to 3 bits at load; TBL / LUT kernels; skipping; low rank; lossless coding of nibbles (closed on decode cost, uncounted) |
| campaign 2 `memo-21-unconventional-levers.md` | LUT inference; XOR / popcount; gemv to gemm; activation-gated skipping; vector-quantisation codebooks; shifts; gate-guided row skip; early commit on the head screen; sign-bit first stages; the stock OpenCL backend; core mixes and the heat gate; residency tricks; bit layouts; reuse across tokens; work-stealing helpers; cache-conditional routing; a helper that warms L3; cooler prefill |
| `memo-12-work` (both of us, 40 rounds) | the CPU + GPU lane and its join cost; recordable queues; persistent kernels with a shared flag; stacked buffers; width 2 with a guess head; the W-position kernel; prompt lookup and rule T; copy-run passes; the side file; the head's placement; float columns for the head; a GPU prefill; a share that follows the cap; the key-major attention walk; the compute-buffer release; `cl_qcom_perf_hint` |

Also read before and not repeated: HeteroLLM, mllm-NPU, PowerInfer-2, MNN-LLM, Lever, AHASD, HeteroMosaic, Hydra,
EAGLE, Medusa, Kangaroo, REST, Token Recycling (r2, r3, r4, memo-10).

## 2. New ideas

### 2.1 Upstream's n-gram speculation is already in our tree (*source*)

`lab/llama.cpp-base/common/` at the base commit:

| piece | what it is |
|---|---|
| `common.h` lines 174 to 187 | speculative types: `draft-simple`, `draft-eagle3`, `draft-mtp`, `draft-dflash`, `draft-dspark`, and five without a draft model: **`ngram-simple`, `ngram-map-k`, `ngram-map-k4v`, `ngram-mod`, `ngram-cache`** |
| `ngram-mod.{h,cpp}`, `ngram-map.{h,cpp}`, `speculative.cpp` | the drafters and the verify-and-roll-back loop |
| `arg.cpp` lines 4295 to 4340 | `--spec-type` (a list), `--spec-ngram-mod-n-min / -n-max / -n-match`, `--spec-ngram-simple-size-n`, … |
| llamadart, `llama_cpp_service.dart` lines 525 to 545 | the app library's speculative configuration: strategies, `ngramSizeN`, `ngramSizeM`, `ngramMinHits`, `ngramMatch`, `ngramTokenMin`, `ngramTokenMax`, static and dynamic cache paths, a draft model path |

- **Mechanism.** `ngram-simple`: when the last N tokens occurred earlier in the context, propose the M tokens
  that followed and verify them in one batched decode; keep the leading ones the model itself produces.
  `ngram-mod`: the same through a hash table, with a minimum and maximum draft length. `ngram-cache`: three
  levels of n-gram statistics, one of them a static file built from a corpus.
- **What it is in our terms:** Tom's lookup arm (a), Bonnie's rule T (its lookup half, with N as the match
  length) and her copy-run pass (M > 1), and with the static cache the "text datastore" of memo-11 §3.2. None of
  our memos mentions that it exists upstream (searched: `campaigns/`, `docs/`, `patches/`).
- **Budget:** exact. A draft is kept only if the model produces it; greedy output is identical by construction,
  to be shown per subject as for any multi-token method.
- **Price:** nothing new in tok/s. It is the price already carried for lookup (r24: Yi 1.03 alone, Llama-2 1.28;
  r14: DeepSeek 1.19 to 1.23 tokens per pass on the claim prompt). **What changes is the cost of getting there:
  the draft source and the rollback are a setting, not a patch.** The width-W verification still runs ggml's
  stock batch path, which is the four-row tile on the CPU (c 2.5, Bonnie r5) and `_mc3` on a lane; the
  W-position kernel is still the build item.
- **Also there: `draft-eagle3`.** Upstream can run an EAGLE-3 draft head. The draft-layer route (memo-10 E5) is
  an owner question for other reasons (a published or trained draft model per subject); its runtime side would
  need no build either. Unchanged otherwise.
- **Its defaults are for another use** (*source*, `common.h` lines 367 to 369): lookup length 12, draft length
  48, minimum hits 1. That suits long verbatim repeats (editing code). Our numbers are for a lookup of 2 or 3
  tokens and a draft of 1; both are settings.
- **Its cost model is ours** (*source*, `examples/speculative-simple/speculative-simple.cpp` lines 274 to 331):
  the last token and the draft go in as one batch; after a partial accept the rejected tail is removed from the
  cache with `llama_memory_seq_rm` and the accepted prefix is kept, so a miss costs the wasted positions and no
  re-decode. The app library has its own entry points for the same steps (`llama_dart_speculative_draft`,
  `_process_batch`, `_accept`; they need `speculativeRollbackTokenMax` > 0, which campaign 2's kit already set
  for its draft-model runs).
- Evidence class: *source* (read today). P(upstream's `ngram-simple` at N = 2 reproduces our simulated lookup
  passes on the official prompt within one pass, per subject) 0.75; the difference to expect is in details (it
  drafts M tokens at once and may use the newest or the most frequent match).

### 2.2 A Vulkan lane with host-signalled events (*external*: the specification; nothing measured)

- **The problem it addresses.** The lane's binding cost is K, the dead time around each GPU stage: 452 µs per
  round trip in R74 (*device*), of which 43 % host driver and 44 % submit-to-start (Bonnie r1). The stripped
  stack needs K ≤ 24 to 90 µs for 12 on all three (Bonnie r13).
- **Mechanism.**
  1. Record once a command buffer that holds every GPU stage of a token: for each stage a
     `vkCmdWaitEvents` on a host-set event, the dispatch, and a `vkCmdSetEvent` when it is done.
  2. Submit it once per token. Per stage the host writes the stage's input into a persistently mapped buffer,
     calls `vkSetEvent`, computes its own rows, and polls `vkGetEventStatus` for the stage's done-event.
  3. No enqueue, no argument setting, no flush per stage.
- **Why it is not the same as what R75 tests.** x4d replays a recorded OpenCL queue (a vendor extension whose
  entry points Tom had to guess); x4e keeps one kernel alive and passes a flag through shared memory, which
  OpenCL 1.2 does not define. Vulkan defines both things this design needs: host-signalled events inside a
  command buffer (the wait's source stage must include the host stage; *external*, specification text through a
  search summary) and use of a buffer while it is mapped.
- **What is unknown, and it is everything that matters:**
  - how the Adreno 660's driver implements a wait on a host event. If it splits the command buffer and waits on
    the CPU side, each stage is a kernel submission again and K is back;
  - whether a compute shader reaches the OpenCL kernel's 21 GB/s. The upstream kernel relies on Adreno
    sub-group controls that have no Vulkan counterpart I know of. One user report has llama.cpp's Vulkan
    backend slower than the CPU on an Adreno 740 (*external*, search summary of an issue; not opened);
  - the half-precision product would be a second implementation with its own numerics check.
- **Budget:** a numerics change of the lane's kind (not bit-identical; inside the budget if the OpenCL lane is).
- **Price** (*estimated*, from Bonnie r13's K sensitivity of about 0.43 tok/s per 100 µs on the stripped
  stack): K 300 → 50 µs is **+1.0 to +1.2 tok/s per subject**; at K 50 the stripped stack reads about 11.9 /
  11.9 / 11.4 (`linear`).
- **P:** 0.15 that events give K ≤ 100 µs on this driver; 0.4 that a compute shader comes within 20 % of the
  OpenCL kernel's rate; together about 0.06.
- **When:** only if R75 shows both x4d (replay ≤ 0.5 × sync, P 0.20 to 0.25) and x4e (flag seen, P 0.55) failing.
  A first cell would be model-free and small in phone time, but large to build: a SPIR-V gemv, a Vulkan
  context beside the OpenCL one, and an event round-trip timer.

### 2.3 What the newest mobile-GPU engine says (*external*, abstract only)

BitLLM (arXiv 2611.00547, submitted 2026-10-31; <https://arxiv.org/abs/2611.00547>; the abstract through the fetch
tool, not the paper): low-bit decode on mobile GPUs is slow because of (a) conversions between quantised and
floating-point data in the hot path and (b) inefficient KV-cache memory access. Its engine keeps the matmuls in
integers, defers the conversions, and re-lays-out and packs the KV cache. Claimed: 1.78 × over llama.cpp, 1.73 ×
over MNN-LLM, 1.58 × over LiteRT-LM, 1.47 × over MLC-LLM on average. No device, model size or token rate in
what I read.

| its point | on our phone |
|---|---|
| KV layout and packing | **the direction of E-Y4** (r10 §1.3: the per-head walk steps a whole cached row between reads). Independent support that the walk is worth measuring (E-T18). No number carries over |
| integer-native matmul on the GPU | the phone has `cl_qcom_dot_product8` and not the Khronos extension (§3.3). memo-11 §3.9 priced an int8 GPU lane and found it dominated. What BitLLM adds is the claim that the conversion is the cost. On the 660 the one-column gemv is bus-bound (21 GB/s against a compute ceiling of 28, *device*, R74), so removing conversions helps only where the GPU is compute-bound: at two columns. If `dev_ratio(2)` fell from 1.25 to 1.10 the width-2 lane gains about 0.1 to 0.2 tok/s (*estimated*) |

P that it changes a price by more than 0.2 tok/s: 0.1. Worth reading in full when the paper's text can be had.

## 3. Closed on numbers this round

### 3.1 Lossless coding of the weights (*host*)

*External*: EntroLLM (arXiv 2505.02380: Huffman-coded quantised weights with parallel decoding on edge devices;
search summary) and MINE (arXiv 2611.09424, 2026-11-06: an ANS coder on desktop GPUs, 2.52 × over the fastest
baseline on an RTX 4060 Ti from a 2.1-bit checkpoint; abstract through the fetch tool).

Our files (*host*; every Q4_0 body tensor, every 16th block, 10.8 to 12.6 M blocks per file):

| | Yi | Llama-2 | DeepSeek |
|---|---|---|---|
| entropy per nibble, order 0 | 3.768 bits | 3.770 | 3.768 |
| the same over byte pairs | 3.767 | 3.769 | 3.767 |
| range over the seven tensor classes | 3.748 to 3.773 | 3.763 to 3.774 | 3.761 to 3.771 |
| bits per weight with the block scale: stored / coded | 4.50 / 4.27 | 4.50 / 4.27 | 4.50 / 4.27 |
| **largest possible saving** | **5.2 %** | **5.1 %** | **5.2 %** |

- A Q4_0 block is scaled to its own maximum, so its nibbles are spread almost evenly. There is very little to
  code away. memo-11's "10 to 15 % at best (not counted on the files)" is now counted: 5 %.
- Even at no decode cost that is 6 to 7 ms of body per token, +0.3 tok/s. The CPU would need about 13 G symbols
  per second per core; the GPU has about 15 % of spare compute at one column (R74). **Closed.** P under 0.01.
- The published gains come from checkpoints quantised for low entropy (2.1 bits). That is a served-file route,
  which is the owner's.

### 3.2 Activation sparsity with a low-rank part (*external*, search summaries)

R-Sparse (ICLR 2025, arXiv 2504.19449): each linear layer as a sparse part driven by the large inputs plus a
low-rank part, no training, "50 % model-level sparsity", up to 43 % faster generation with a custom kernel.
WiSparse (arXiv 2602.14452): 50 % sparsity keeping 97 % of Llama-3.1's quality, 21 % faster.

- This is the column-input family the campaign measured and closed: our subjects tolerate 20 to 40 % inside
  the budget (*host*, memo-02, memo-07), and the column kernel costs 2.6 × per touched weight on the A78, which
  killed it (*device*, R69 rule A).
- The new element is the low-rank compensation, which could raise the tolerated sparsity. It does not change
  the kernel's cost, and "97 % of dense quality" is far outside a +5 % perplexity budget.
- **Closed, same reason as before.** P 0.03 that the compensation moves a subject's frontier past the 62 % the
  kernel needs.

### 3.3 The phone's extensions (*device*, R72's `x2info` text, read today)

| extension | on the phone | consequence |
|---|---|---|
| `cl_khr_integer_dot_product` | **absent** | upstream's `gemm_noshuffle_q4_0_q8_1_dp4a` and its other dp4a kernels are not selectable (`has_integer_dot` reads this name only). **My r5 test E-T1 is answered: the integer prefill kernel is closed on this phone as upstream wrote it** |
| `cl_qcom_dot_product8` | present | the vendor's signed × unsigned form; a kernel would have to be written for it (memo-11 §3.9) |
| `cl_qcom_ml_ops` | present | Qualcomm's own operator library. Its fully-connected operator takes float or 8-bit tensors (*recalled*, not read today): at least twice the bytes per weight of Q4_0. On a bus-bound phone that halves the rate before anything else. Closed by bytes |
| `cl_qcom_compressed_image` | present | hardware texture decompression; lossy, as memo-11 said |

### 3.4 Others

| item | *external* source | why it adds nothing |
|---|---|---|
| SpecBound (ACL 2026 Findings): a self-draft that stops early by layer-wise confidence and pushes the cached states through the remaining layers in one pass | search summary | the prefix-exit ring and the layer-skip drafts of memo-11 §3.3 and §3.4; the draft's layers cost their bytes again, which is what closed them |
| sustained-load studies (arXiv 2603.23640: a Snapdragon 8 Gen 3 phone falls 15 to 22 % from its peak within 8 iterations; an iPhone 41.5 %; a 1-second pause between iterations is not enough to recover) | read through the fetch tool | agrees with our gate records (r4 §4). Its remedies are future work: duty cycling, cooling. A pause is the owner question already listed (r5 §1.3) |
| MTP in llama.cpp (merged 2026-05 and 2026-06) | search summary | needs a model that ships an MTP head; none of the three does |
| decode is "bound by KV-cache access speed" (arXiv 2603.23640, 2606.27906) | summary and fetch | stated, not measured, on newer chips. Our decode is bound by the weights' bytes (R72, R74); the KV walk is a second-order term we have not measured (E-T18) |
| measurement studies of NPUs (arXiv 2605.27435, 2607.05475) | search summary | the Hexagon route is closed here by the shared bus (r2 C.2) |

## 4. Searched and not found (2026-11-24)

- A published token rate, prefill or decode, for any Adreno 6xx phone with a current runtime.
- Any measurement of the latency of a host-signalled Vulkan event on an Adreno driver.
- A draft-free exact method newer than the n-gram family and the self-draft family.
- Anything that reads fewer weight bytes per token exactly, for a frozen 4-bit file, other than what the
  excluded memos hold.
- I could not read the two IWOCL slide decks on the Adreno backend (2025, 2026): PDFs, no reader here.

## 5. Ranked try list

| # | item | after | label | price | P |
|---|---|---|---|---|---|
| 1 | **Use upstream's `ngram-simple` / `ngram-mod` as the lookup source and the copy-run pass** instead of writing one | the host test S-T1 below | bit-identical output (greedy text) | none in tok/s beyond what rule T and r25 already carry; it removes a build item and its bugs | 0.75 that it matches our simulation within one pass |
| 2 | Vulkan lane with host events | R75 showing x4d and x4e both failed; then a model-free device cell | numerics change of the lane's kind | +1.0 to +1.2 tok/s if K goes from 300 to 50 µs | 0.06 |
| 3 | A `cl_qcom_dot_product8` gemv for the two-column case only | R75's `x4 mcsum dev_ratio(2)` reading above 1.3 | exact on the GPU's rows (integer products) if activations are quantised as the CPU does | +0.1 to +0.2 tok/s | 0.1 |

Tests:

| # | test | where | cost | closes when |
|---|---|---|---|---|
| S-T1 | llama.cpp's own speculative example on the base tree, `--spec-type ngram-simple` at N = 2 and 3 with M = 1, then `ngram-mod` with a draft of up to 4, on the official prompt of each subject (the prompt the phone sends, r14) and on 8 set prompts; greedy; print drafted, accepted and passes, and compare the text with a plain run | host, CPU, Tom | a build target and minutes | passes within one of our simulation and identical text → the lookup is "upstream's, configured"; a different text → not exact as upstream wrote it for our settings, which would itself be worth knowing |
| S-T2 | the same types through llamadart's speculative configuration in the kit (no run: a reading of how the kit would set them and whether the app's rollback path is the one campaign 2 exercised) | *source* | an hour | – |

## 6. Next questions, ranked

| # | question | state |
|---|---|---|
| 1 | Does upstream's verify loop feed the draft as one batch, and what does it do on a partial accept? | **done, §2.1:** one batch; the rejected tail is removed from the cache, nothing is decoded again. The cost model is ours |
| 2 | The owner's answers (DeepSeek's prompt, Yi's marker); Tom's KleidiAI stock result | waiting |
| 3 | BitLLM's full text: what its KV packing is, and on which GPUs | when a readable copy exists |
| 4 | Is there any record of `vkCmdWaitEvents` with a host event on Adreno 6xx (driver source, bug trackers)? | one more search; low odds |
| 5 | R75 free reads; E-T18 | R75; a later kit |

## 7. Sources (2026-11-24)

Repository: the three excluded memos (their route tables and headings, §1); `lab/llama.cpp-base/common/common.h`
(lines 174 to 187, 378 to 392), `arg.cpp` (lines 4295 to 4340), `ngram-mod.h`;
`lab/app-c3/third_party/llamadart/lib/src/backends/llama_cpp/llama_cpp_service.dart` (lines 520 to 545);
`campaigns/03-dense15-5pct/reports/device/r72-c3probe-r33-2026-11-22T1759.json` (extension names); the three served
GGUF files (nibble counts); Bonnie's r13 (K sensitivity), r24, r25.

External, with what I read of each:
- BitLLM, arXiv 2611.00547: abstract page through the fetch tool. <https://arxiv.org/abs/2611.00547>
- MINE, arXiv 2611.09424: abstract page through the fetch tool. <https://arxiv.org/abs/2611.09424>
- "LLM Inference at the Edge … Under Sustained Load", arXiv 2603.23640v2: the HTML through the fetch tool.
  <https://arxiv.org/html/2603.23640v2>
- Search summaries only: EntroLLM (<https://arxiv.org/abs/2505.02380>); R-Sparse
  (<https://arxiv.org/abs/2504.19449>); WiSparse (<https://arxiv.org/abs/2602.14452>); SpecBound
  (<https://aclanthology.org/2026.findings-acl.802.pdf>); llama.cpp pull requests 22673, 23269, 23398 (MTP and
  n-gram types; <https://github.com/ggml-org/llama.cpp/pull/23269>); the Vulkan reference page for
  `vkCmdWaitEvents` (<https://docs.vulkan.org/refpages/latest/refpages/source/vkCmdWaitEvents.html>); a
  llama.cpp issue on Vulkan on an Adreno 740 (<https://github.com/ggml-org/llama.cpp/issues/20603>); arXiv
  2606.27906, 2605.27435, 2607.05475.
