# c3 memo-04 work: block-granular input sparsity (Ebony's draft)

Date: 2026-11-19. Author: Ebony (`claude-fable-5-2`). For the lead and Bonnie; not a verdict.

**Labels.** *published* = from a paper (how it was read is stated in §6). *host* = Tom's logs under `lab/c3/logs/`
(relative only). *estimated* = my arithmetic or a synthetic model. Nothing here was measured on the phone, and I ran
no model. The only things I computed: paired standard errors from Tom's existing `PERCHUNK` lines, a fit to his H6
table, and a synthetic energy model (no model activations).

**What I read of Tom's running H3b/H6b work** (not duplicated): the uncommitted diffs of `tools/c3_colskip.py` and
`tools/c3_colkern_bench.c`, `lab/c3/queue4.sh`, and `lab/c3/logs/h6b-blockskip.log`. No H3b perplexity log existed
when this was written, so the prediction in §2.3 was made before any block PPL.

## 0. Short answers

1. **Literature.** No paper found measures what the lead proposes (zeroing whole contiguous groups of 32 or 64
   input channels by group norm) on an LLM. The only structured activation sparsity that is published is N:M
   (keep N of every M consecutive inputs), which is a different constraint, and its cost over unstructured at equal
   50 % sparsity is large at small M and shrinks at M = 16 or 32 (§1). No paper found permutes channels to cluster
   salient activations; channel permutation exists only for static N:M weight pruning.
2. **Permutation.** It is bit-identical only if it stays inside a Q4_0 block (or reorders whole blocks while
   the float steps keep the file's block order). Any permutation that moves a column to another block needs the
   weights requantised from their Q4_0 values. That is lossy at threshold 0, I price it at about +2 points
   (Mistral) and +3.5 points (Qwen2.5) of the 5-point budget (*estimated*), and what it buys is bounded by how
   static the channel magnitudes are, at which point it is close to static channel pruning (§3).
3. **memo-02 §3 and §4.** No error that changes a closed/open reading. Six things to correct or add (§4): the
   "7 of 8" count is wrong and the dense-first effect on Mistral is statistically real, not noise; the frontier
   cells have a paired standard error of 0.4 to 0.7 points; the column layout does not skip the scales, so the bytes
   at 40 % sparsity are 64 %, not 60 %; H6's cost is a fixed part plus a per-weight part; prefill in the column
   layout is not benched; the speed text and the gate text differ.
4. **The lead's premise needs one correction.** In the stock 4x4 repack a 32-input block is 72 contiguous bytes
   per group of 4 rows, at a pitch of 72 × K/32 bytes. A skip list on the stock layout therefore saves compute in
   proportion but almost no DRAM traffic (§2.1). Which layout is right depends on the phone's read rate (P2).

## 1. Literature, 2024 to 2026 (searched 2026-11-19)

| work | date | granularity | what is measured against unstructured at equal sparsity | read how |
|---|---|---|---|---|
| N:M activation benchmark, "Motivating Next-Gen Accelerators with Flexible N:M Activation Sparsity" ([arXiv 2509.22166v3](https://arxiv.org/html/2509.22166v3)) | v3 2026-02-28 | N:M on linear-layer inputs, all at 50 %: 2:4, 4:8, 8:16, 16:32 | **the only equal-sparsity PPL table found** (below) | fetch summary of the HTML |
| Amber Pruner ([arXiv 2508.02128v1](https://arxiv.org/html/2508.02128v1)) | 2025-08 | N:M (2:4, 4:8, 8:16) on inputs of q, gate, down (k, v, o, up kept dense; some layers skipped); prefill only | no PPL and no unstructured baseline. Accuracy drop, Llama-3.1-8B: magnitude top-k −10.3 / −7.4 / −5.4 % at 2:4 / 4:8 / 8:16; with its weight-norm score −2.3 / −1.8 / −0.7 % | fetch summary |
| R-Sparse ([arXiv 2504.19449v1](https://arxiv.org/html/2504.19449v1)) | ICLR 2025 | per input channel (unstructured across channels); the dropped small inputs go through a rank-r SVD of W | no block variant. No PPL values in the text read (figure only). Speed on an A6000 GPU, FP32. Needs r(m+n) extra weights per matrix, read every token | fetch summary |
| TEAL ([arXiv 2408.14690v3](https://arxiv.org/html/2408.14690v3)) | ICLR 2025 | per element | no structured variant found in two searches | memo-01 read; search |
| WINA ([arXiv 2505.19427v1](https://arxiv.org/html/2505.19427v1)), WiSparse ([arXiv 2602.14452v1](https://arxiv.org/html/2602.14452v1)) | 2025-05; 2026-02 | per element; score \|x_j\| × ‖W column j‖ (WiSparse: norm to a per-layer power) | no block variant ("block" in WiSparse is a transformer block for allocating ratios). Accuracy only, no PPL. Both beat plain magnitude (TEAL) at equal sparsity | WINA: search summary only; WiSparse: fetch summary |
| DIP, DIP-CA ([arXiv 2412.01380v2](https://arxiv.org/html/2412.01380v2)) | 2024-12 | per element (top-K per token); the cache mask is per neuron | no block mask. Its Table 1 has static SparseGPT 2:4 and 4:8 weight pruning as baselines, not as DIP variants | fetch summary |
| SpQt ([arXiv 2511.04477v1](https://arxiv.org/html/2511.04477v1)) | 2025-11 | per column (one column = a run of 256-row superblocks) | no block-of-columns variant, no granularity comparison. **It quantises to Q4_K "under the new layout" as a one-time step**, i.e. from the original weights, not a re-layout of an existing file | fetch summary |
| PermLLM ([arXiv 2510.10136](https://arxiv.org/abs/2510.10136)) | NeurIPS 2025 | learned channel permutation for **static N:M weight** pruning (Wanda, RIA), block size 64 | not activation sparsity; permutation is applied to float weights before pruning | search summary only |
| Ripple ([arXiv 2410.19274](https://arxiv.org/abs/2410.19274)) | 2024-10, v3 2025-10 | places co-activated neurons next to each other **in flash** to cut IOPS | the nearest published idea to "cluster by activation"; it is for storage reads of sparse (ReLU-family) models, and reports I/O gains, not PPL | search summary only |
| PowerInfer-2 ([arXiv 2406.06282v3](https://arxiv.org/html/2406.06282v3)) | 2024-06 | "neuron clusters" as the unit of compute and I/O | ReLU-fied models; clusters are an execution unit, not a skip rule with a PPL cost | search summary; memo-01 |
| SparAMX ([arXiv 2502.12444](https://arxiv.org/abs/2502.12444v1)) | 2025-02 | unstructured **weight** sparsity on x86 AMX | not activation sparsity; listed because it was the only "CPU kernel" hit | search summary only |

**The one equal-sparsity table** (*published*, 2509.22166v3 Tables 10 to 12; wikitext-2, FP16, the paper's own
protocol; percent over dense is my arithmetic, *estimated*). All columns are 50 % activation sparsity with a plain
magnitude rule:

| model | dense | unstructured | 8:16 | 2:4 |
|---|---|---|---|---|
| Llama-2-7B-chat | 6.94 | 7.53 (+8.5 %) | 8.12 (+17.0 %) | 10.23 (+47.4 %) |
| Qwen2.5-7B-Instruct (q, k, v inputs not sparsified) | 7.46 | 8.3 (+11.3 %) | 8.61 (+15.4 %) | 10.06 (+34.9 %) |
| Gemma-3-4B-Instruct | 17.29 | 22.39 (+29.5 %) | 25.31 (+46.4 %) | 35.62 (+106 %) |
| Llama-3.1-8B-Instruct | 7.21 | not reported | 10.32 (+43.1 %) | 16.61 (+130 %) |

The paper gives no PPL for 16:32, only an accuracy drop on Llama-3.1-8B: 14.35 / 9.29 / 7.38 / 5.40 % for
2:4 / 4:8 / 8:16 / 16:32.

What this says for us, and what it does not:

- N:M forces **uniform** sparsity in every window; block skipping forces **clustered** sparsity. They are opposite
  constraints. N:M gets closer to unstructured as the window grows (it only has to find the 16 smallest of 32);
  block skipping gets further from it as the block grows (§2.2). So the N:M numbers are not a price for the lead's
  lever. **No published price exists for it; Tom's H3b is the first number.**
- Two published side results apply to both the element and the block rule and were not in H3:
  - a weight-norm score (\|x_j\| × ‖W column j‖; Amber Pruner, WINA, WiSparse) beats plain magnitude everywhere it
    was tested (Amber: −10.3 % to −2.3 % at 2:4);
  - a static shift (2509.22166's "S-PTS": prune around a calibrated offset and add the offset's contribution back
    as a bias) cut the average accuracy drop at 8:16 from 5.47 to 0.61 points. I have this from a fetch summary and
    did not see whether the offset is per channel or per layer.
- A fixed count per block (16 of 32) is a kernel-friendly form of the element rule: the column kernel's loop count
  per block becomes constant. Its cost over unstructured is not published at 16:32 in PPL.

Not found: any pull request or issue for activation sparsity in llama.cpp (memo-01's search, not repeated), and
any block-granular CPU kernel for dynamic sparsity. MNN, ExecuTorch and mllm were not searched for this question.

## 2. The lead's lever: whole Q4_0 input blocks

### 2.1 What a skipped block saves in the stock 4x4 layout (*host* facts from the tree, *estimated* traffic)

`repack_q4_0_to_q4_0_4_bl` (`ggml-cpu/repack.cpp` at the lab tree) writes rows in groups of 4, and inside a group
the blocks in input order. One `block_q4_0x4` is 72 bytes (4 fp16 scales, 64 nibble bytes). So the bytes of input
block b are one 72-byte piece per 4 rows, at a pitch of 72 × K/32 bytes (9,216 bytes for K = 4096). They are
contiguous only 72 bytes at a time.

- **Compute**: saved in proportion. Tom's H6b bench (stock layout plus a skip list, `h6b-blockskip.log`, *host*):
  bit-identical to the stock kernel with the blocks zeroed, 1.02 to 1.15 × stock per touched weight at 100 to 50 %
  kept. This part of the lead's claim holds.
- **DRAM traffic**: a 72-byte piece at 72-byte pitch is not aligned to 64-byte cache lines. An isolated skipped
  piece frees a whole line only when it happens to cover one: (72 − 63)/64 = 0.14 lines of the 1.125 it spans, about
  **12 % of its bytes**. Two adjacent skipped blocks (the 64-column unit) free 1.27 of 2.25 lines, about 56 %; four
  adjacent, 78 % (*estimated*, uniform offsets, before prefetch). The hardware prefetcher follows the row stream
  and will fetch most skipped single lines anyway, so the real saving is lower still. The host bench cannot see
  this: the M4 runs this kernel at 52 GB/s on one core (*host*), i.e. compute-bound.
- So on the stock layout the lever is a **compute lever only**. memo-02 §7 prices the phone's dense token as
  read-bound at 19 GB/s (*device* rate, *estimated* split: about 210 ms of reads against 110 to 130 ms of `sdot`).
  If that holds, a skip list on the stock layout moves the token very little.

**The layout that does save traffic** keeps everything else of the idea: store the same `block_q4_0x4` pieces
block-major inside tiles of TR rows, `[tile][input block][TR/4 pieces]`. A skipped block is then TR/4 × 72
contiguous bytes (18 KB at TR = 1024), scales included. The kernel is the stock one with the 4 float accumulators
loaded and stored per piece (TR floats, in L1). The float steps per row still run in the file's block order, so it
is **bit-identical** to stock at threshold 0 and equal to stock with zeroed blocks at any threshold. Its cost is not
benched; I expect 1.1 to 1.3 × `sdot` per touched weight (*estimated*: one load and one store of 16 bytes added to
about 26 instructions). It also serves batched prefill almost unchanged, which the column layout does not (§4.5).

### 2.2 What a block rule costs in input energy (*estimated*, synthetic; no model activations)

Share of the input's energy (sum of x²) dropped at a given sparsity, for groups of g consecutive channels dropped by
group L2 norm with one threshold, K = 4096, 2,000 synthetic tokens:

| input model | g = 1 at 40 % | g = 8 at 40 % | g = 32 at 40 % | g = 64 at 40 % | block-32 sparsity with the energy of element 35 % / 40 % |
|---|---|---|---|---|---|
| iid Gaussian | 3.5 % | 22 % | 31 % | 33 % | 4.3 % / 6.2 % |
| iid Laplace | 1.5 % | 16 % | 26 % | 30 % | 2.6 % / 3.9 % |
| iid Student-t(3) | 1.4 % | 12 % | 19 % | 23 % | 3.3 % / 4.8 % |
| Gaussian × fixed log-normal channel scale, σ = 0.5 | 1.8 % | 16 % | 26 % | 29 % | 3.0 % / 4.4 % |
| the same, channels **sorted by scale** (§3) | 1.8 % | 8.4 % | 10 % | 10 % | 10.9 % / 14.4 % |
| σ = 1.0, unsorted | 0.2 % | 4.3 % | 12 % | 16 % | 1.4 % / 2.1 % |
| σ = 1.0, sorted | 0.2 % | 0.9 % | 1.0 % | 1.0 % | 17 % / 22 % |

Readings:
- With no structure in the channel order, the norm of 32 channels concentrates (its relative spread is about
  √(2/32) = 25 % for Gaussian inputs), so the "lowest" blocks still carry most of an average block's energy.
  Dropping 40 % of blocks drops 19 to 31 % of the energy, against 1.4 to 3.5 % for elements: about ten times more.
- At equal dropped energy, block-32 sparsity is **1 to 6 %** where element sparsity is 35 to 40 %. A 64-wide unit
  is a little worse again. Heavier tails do not rescue it.
- Dropped energy is not perplexity, and real inputs are not iid. The `down` input (the gated product) is the one
  most likely to have block-to-block variation, and H3 found it the tolerant group on both models.

### 2.3 Prediction for H3b, written before its data (*estimated*)

Largest block-32 sparsity inside +4.5 % on 40 chunks, plain block-norm rule, all four groups: **3 to 10 % on both
models**, Qwen2.5 lower than Mistral; block-64 about a point lower. On the `down` input alone 10 to 25 %. I put the
chance that either model holds 20 % model-wide at about 0.1.

**Heads-up for Tom's running grid.** `queue4.sh` runs cells at 20, 30 and 40 % only, and the `thr3` tables hold
quantiles from 0.2 upwards. If the prediction is right, every cell fails and the grid shows no frontier. The
tables need the quantiles 0.02, 0.05, 0.10 and 0.15, and per-group cells (`down` alone). Also: the resume check in
`tools/c3_h3_grid.sh` greps for `dense_first <N>:`, and the block tag is written before the colon, so a restarted
block grid repeats finished cells (harmless, but it doubles lines in the log).

### 2.4 Price on the phone (*estimated*; memo-02 §7's model: token time = the larger of read time and compute time)

Mistral bytes from memo-02 §1 (*host*): 4.037 GB per token, 0.110 GB of it the head. Compute 120 ms (middle of
108 to 132, *estimated* in memo-01).

| configuration | read rate 19 GB/s (*device*, campaign 1) | read rate 35 GB/s (assumed; P2 decides) |
|---|---|---|
| stock dense | read 212 ms, compute 120 → 212 ms, 4.7 tok/s | read 115, compute 120 → 120 ms, 8.3 tok/s |
| column kernel, 40 % element sparsity (H3, H6 fit of §4.4) | read 139, compute 183 → 183 ms, 5.5 tok/s | 183 ms, 5.5 tok/s: slower than dense |
| skip list on the stock layout, block sparsity s | read barely changes → about 4.7 to 4.8 tok/s | compute 120 × (1 − 0.97 s), but the read time of 115 ms then binds: s = 10 % → 115 ms, 8.7 tok/s |
| block-major layout, kernel 1.15 ×, block sparsity s | read 212 × (1 − 0.97 s): s = 10 % → 5.2 tok/s; s = 14 % equals the column kernel at 40 % | compute 138 × (1 − 0.97 s): needs s above 13 % to beat stock dense |

- At 19 GB/s the block lever needs the block-major layout and about **14 % block sparsity to match what the column
  kernel gives at 40 % element sparsity**. That is above my prediction for both models.
- At 35 GB/s the token is compute-bound, the column kernel loses, and the plain skip list on the stock layout is
  the right form: bit-identity is trivial and no new layout is built. The gain is capped at about 4 % (120 → 115
  ms), because the unchanged read time binds next.
- Neither row opens a path to 15 tok/s.

## 3. A calibrated input-channel permutation

### 3.1 Is it exact?

A Q4_0 weight is w = d[r, b] × q, with one fp16 scale d per row r and block b of 32 inputs, and q in −8 to 7. The
stock kernel sums q × x_q exactly in integers over the 32 inputs of a block, then does one float step per
(row, block) in block order.

| permutation | exact? | why | use |
|---|---|---|---|
| inside a block (the 32 columns of block b stay in block b) | **bit-identical** | the integer sum does not depend on order; the scale and the Q8_0 activation scale of the block are unchanged | makes sub-block units: in the 4x4 piece, 8 columns (4g to 4g+3 and 16+4g to 19+4g) share one 16-byte vector and two `sdot`s. Sorting each block's channels by calibrated magnitude puts its 8 coldest in one vector |
| whole blocks reordered **in storage**, float steps still in file order | **bit-identical** | storage order is free when each row's accumulator is kept across blocks (block-major layout) | nothing for skipping (a block is already contiguous there); useful for a hot-first resident set (memo-01 Y5) |
| whole blocks reordered **in the float order** (for example hot-first in the stock row stream) | not bit-identical | float addition is not associative | a declared numerics change of rounding size. Scale: stock itself differs by 0.06 % between `-b 512` and `-ub 1` on Mistral (5.8844 against 5.8810, *host*) |
| **across blocks** (the lead's idea) | **not exact** | a new block holds columns that had 32 different scales per row | see below |

For a cross-block permutation there are three ways to store the weights, and none is free:

1. **Requantise to Q4_0 with new block scales, from the Q4_0 values.** Lossy at threshold 0. The values d × q sit
   on lattices of different pitch; rounding them to a common new scale adds a second quantisation error of about
   the same variance as the first (the new scale is again max/8 of 32 weights of the same row). If the PPL cost of
   quantisation noise is linear in its variance, the cost is about the file's own Q4_0-over-Q8_0 gap again:
   **about +2.0 % on Mistral and +3.6 % on Qwen2.5** (*estimated* from memo-02's ×1.0195 and ×1.0357, *host*,
   40 chunks). The activation side changes too (Q8_0 scales are per block of the permuted order), a much smaller
   term. This is spent before any block is skipped. It can be measured on the host (T3 below).
2. **Keep every weight's original scale.** Then a block has up to 32 scales per row, the integer sum cannot be
   factored, and `sdot` is lost. Restricting the permutation to moving 8- or 16-column pieces keeps `sdot`, but
   the float step is then per piece: the result is no longer bit-identical (reassociation), and each original
   scale column is read once per piece, 4 times at 8 columns: 6.0 bits per weight read instead of 4.5, +33 %
   bytes (+11 % at 16 columns) (*estimated*). Exactness would need all pieces of an original block processed
   together, which is the within-block case again.
3. **Requantise from the float weights in permuted order** (what SpQt does for its layout). That is a different
   served file, not a use of this one; and the residual-stream inputs would need one global permutation. It is an
   owner question, not a runtime lever.

One part is free: the `down` input is the FFN's hidden vector, so permuting it means permuting the **rows** of
`gate` and `up`, and rows are whole Q4_0 block sets. That side is bit-exact and needs no runtime gather (the
activations come out already permuted). Only `ffn_down` (27 to 29 % of the block-matrix bytes) is requantised, so
the threshold-0 cost for a `down`-only permutation is roughly a quarter of the figures above, about +0.5 and +1.0
points (*estimated*, same linear assumption).

### 3.2 What it costs to run (*estimated*)

- Bytes per token: unchanged (4.5 bits per weight) for way 1.
- Runtime gather of the activations for the residual-stream inputs: about 12 K floats per layer, under 1 ms per
  token as a scalar loop and near zero when folded into the Q8_0 quantisation pass. Zero for the `down` input.
- Repack: dequantise, permute, requantise about 4 GB once; it belongs in the layout file of memo-01 Y1 so that it
  is not paid at every load. Not timed.
- Side table: one permutation per matrix input, 2 bytes per channel, under 2 MB.

### 3.3 What it buys, and its ceiling

A fixed permutation can only cluster what is fixed. If channel magnitudes had a static scale and nothing else,
sorting would turn the block rule into dropping the same cold channels at every token: in the synthetic table of
§2.2 the sorted rows drop the same energy as static pruning of the coldest 40 % of channels (10.0 % against
10.5 % at σ = 0.5). The equal-energy block sparsity rises from 1 to 4 % unsorted to 11 to 22 % sorted, for a
static log-scale spread σ of 0.5 to 1.0 (*estimated*, synthetic). Whether real inputs have σ near 0.2 or near 1 is
not known here; it is what memo-01's H5 (not run) would have counted. The token-dependent part of the element mask,
which is what TEAL-style sparsity lives on, is not recoverable by any fixed order unless channels switch on and off
together (Ripple's co-activation, shown only for ReLU-family neurons).

So the honest price: the permutation must add more block sparsity than its requantisation costs. On Qwen2.5,
+3.6 points at threshold 0 leaves about 1 point for sparsity; that is closed before it starts unless only `down`
is permuted. On Mistral about 2.5 points remain. The cheap order of work is T2 (does sorting raise the block
frontier at all, no requantisation needed to find out), then T3 (what the requantisation costs), and no build
before both.

## 4. Review of memo-02 §3 and §4

I found no arithmetic error in the tables: every percentage I recomputed from the logs matches to rounding
(0.63 against 0.64 and 0.99 against 0.98 in two cells), the byte-weighted 42 % and 38 % are right (41.9 % and
38.3 % from §1's bytes), and the −0.35 to +0.48 spread is right. The points below are corrections of readings and
omissions.

### 4.1 The frontier cells have an error bar that the memo does not give (*host*, my computation)

Paired over chunks (per-chunk NLL of the cell minus stock, from the `PERCHUNK` lines; 40 chunks):

| cell | Δ PPL | paired SE | 95 % interval |
|---|---|---|---|
| Mistral 40 %, `-b 512` | +2.44 % | 0.59 pt | +1.3 to +3.6 |
| Mistral 40 %, `-ub 1` | +2.92 % | 0.68 pt | +1.6 to +4.3 |
| Mistral 45 %, `-b 512` | +4.96 % | 0.71 pt | +3.5 to +6.4 |
| Qwen2.5 35 %, `-b 512` | +3.10 % | 0.49 pt | +2.1 to +4.1 |
| Qwen2.5 35 %, `-ub 1` | +2.75 % | 0.67 pt | +1.4 to +4.1 |
| Qwen2.5 40 %, `-b 512` | +4.51 % | 0.67 pt | +3.2 to +5.9 |
| Mistral mixed 42 % | +3.78 % | 0.57 pt | +2.6 to +5.0 |
| Qwen2.5 mixed 38 % | +3.95 % | 0.77 pt | +2.4 to +5.5 |

The two frontier claims (Mistral 40 %, Qwen2.5 35 %) hold on these 40 chunks with the interval inside +4.5 %.
"Qwen2.5 at 40 % is 0.01 point outside" reads a difference a hundred times smaller than the error; it is
undecided, as is Mistral 45 %. The move of Qwen2.5 40 % from +7.2 % (16 chunks) to +4.5 % (40) is a change of
chunk set, not this noise; memo-02 says so and the full set decides.

### 4.2 The dense-first reading is wrong in its count and too weak in its conclusion (*host*, my computation)

memo-02 §3.4: "worse in 7 of 8 cells". From the table in §3.1 it is worse in **12 of 16** cells (6 of 8 for N = 1,
6 of 8 for N = 8) and better in all four 50 % cells. Paired against the N = 0 cell on the same 16 chunks:

| | 20 % | 30 % | 40 % | 50 % |
|---|---|---|---|---|
| Mistral, N = 1 minus N = 0 | +0.92 pt (t = 4.6) | +0.51 (1.2) | **+1.87 (4.4)** | −0.87 (−1.1) |
| Mistral, N = 8 minus N = 0 | +0.77 (4.4) | +0.08 (0.2) | +1.02 (3.0) | −0.37 (−0.5) |
| Qwen2.5, N = 1 minus N = 0 | +0.70 (1.6) | +0.46 (0.7) | +0.58 (1.1) | −1.03 (−0.6) |
| Qwen2.5, N = 8 minus N = 0 | +0.15 (0.3) | +1.01 (1.9) | +0.41 (0.6) | −1.96 (−1.4) |

- On Mistral, leaving position 0 (BOS) dense makes 20 % and 40 % sparsity significantly **worse**, by 1 to 2
  points. That is the size of the frontier margin, and it is the opposite of what TEAL reports. On Qwen2.5 no cell
  is significant. So this did not generalise: "does not help on either model" is right, but "a real effect on
  Mistral, none detectable on Qwen2.5" is the finding. Qwen2.5's file has `add_bos_token = false`, so its position
  0 is a text token; the two models are not running the same experiment at N = 1.
- It is not explained. One trace in the logs: with BOS dense, Mistral's realised sparsity of the `o` input falls
  from 0.4196 to 0.4121 at the same thresholds. One position out of 512 cannot move the mean by that much, so
  sparsifying BOS changes the activations of the other 511 positions.
- Consequence for the lever: the measured frontier is for a runtime that sparsifies **every** position, the
  prompt's first token included. A design that runs prefill dense (SpQt does) is a different rule with a different
  cost on Mistral, about +1 to +2 points at 40 % on 16 chunks. The pre-registration should fix which one is built,
  and the 40-chunk frontier should be re-run for it if it is the dense-prefill one.

### 4.3 The column layout does not skip the scales, so the byte saving is overstated

§4's layout keeps TR fp16 scales per (tile, block) and skips only the nibble columns. Bytes read at kept fraction d
are (2 + 16 d)/18 of the matrix, not d: **64.4 % at 40 % sparsity, 68.9 % at 35 %** (*estimated* from the layout).
§7's "at 40 % sparsity 2.4 GB = 126 ms" should be 0.644 × 3.926 + 0.110 (head, not sparsified) = 2.64 GB = 139 ms
on Mistral. On Qwen2.5 the head is 0.447 GB (11 %), so 35 % sparsity reads 0.689 × 3.672 + 0.447 = 2.98 GB of
4.119 GB: a 28 % cut in bytes, not 35 %. The conclusion of §7 ("roughly level") does not change, because compute
binds in that row.

### 4.4 "Cost per touched weight" hides a fixed part (*host*, my fit to the H6 table)

The TR = 1024 row fits time / stock dense = **0.56 + 1.63 × density** (residuals under 0.04 at all five densities).
So the kernel costs 1.63 × `sdot` per touched weight plus a fixed 0.56 of a stock dense pass that no element
sparsity removes (the float step, the scale reads and the int16 clear per (row, block)). The rising "2.2 × to
3.0 ×" in the table is this fixed part divided by a shrinking density. Two uses:
- The kill line of memo-03 §5 is stated per touched weight; it is better applied to the whole-pass ratio (1.54 at
  60 % kept, 1.62 at 65 %).
- Skipping a whole block removes the fixed part as well. That is the real case for a block rule, and it argues
  for a **two-level rule in the column kernel** (skip the block when its norm is tiny, else skip elements), which
  the `nk[b] == 0` branch already implements; random masks never exercise it. With real masks some blocks will be
  empty: the H3 tool can count them at the H3 thresholds at no cost.

Smaller points on §4: TR was tried at 256 and 1024 only, and larger was faster (2.67 → 2.19 dense); 2048 and 4096
still fit an A78's 64 KB L1 (size recalled, not checked; int16 and float accumulators 6 × TR bytes) and are one compile flag each. The bench
cannot show prefetch overshoot at the end of each kept run (512-byte columns, random gaps); on a read-bound phone
that is extra traffic the 64.4 % does not include. I checked the exactness argument (integer sum order-free,
8 × 127 × 32 = 32,512 below 2^15, the same fused step, an all-zero block adds nothing because the accumulator can
never be −0.0) and found no hole.

### 4.5 Prefill in the column layout is not benched

Only one layout can be resident (memo-02 §1: the repack already doubles the set). With the column layout in place
of the 4x4 copy, batched prefill and the `-b 512` gate run through it too. Masks differ per token, so a batch
cannot skip columns; it runs dense at 2.19 × stock gemv or more (*host*, the density-1.0 row), against a stock
gemm that is faster per token than gemv. Prefill is not in `decodeTokensPerSec`, but time to first token and the
length of every `-b 512` gate run are. The block-major 4x4 layout of §2.1 does not have this problem.

### 4.6 Smaller omissions in §3

- **Speed and quality are read on different texts.** Thresholds are absolute and calibrated on wikitext; the
  speed metric runs the official prompt. Realised sparsity on the official prompt (and on `en.txt`, which was in
  memo-01's H3 specification) was not counted. It should be, before any speed is predicted from "40 %".
- **Score.** H3 uses plain magnitude. The published weight-norm score (§1) is one multiply per input and could move
  the frontier by a few points of sparsity on both models. Not tested.
- **Allocation.** "Per-group mixes did not beat uniform" rests on one mix per model. TEAL and WiSparse allocate
  per layer (greedy or search), which was not tried. The sentence should say "the one mix tried".
- **Exact zeros.** The calibration logs show 0.9 to 6.1 % of inputs already quantise to 0 in stock Q8_0
  (Mistral: `down` 3.9 %, `gu` 0.9 %, `o` 1.1 %, `qkv` 4.2 %; Qwen2.5: 6.1 / 1.0 / 1.2 / 2.2 %; *host*, train
  text). memo-02 does not report them. Skipping these is bit-identical; it is small, as memo-01 expected.

## 5. Candidates for the joint lists (for Bonnie's pricing and the lead's decision)

### Test list (host unless marked)

| # | test | cost | closes when |
|---|---|---|---|
| T1 | H3b with low cells: block-32 and block-64 at 2, 5, 10, 15 %, all groups and `down` alone, 40 chunks, both models, `-b 512`; the winning cell again at `-ub 1` | recalibrate with the low quantiles, then about the time of the present grid | largest block-32 sparsity inside +4.5 % is under 5 % on both models → the block rule is closed as a traffic lever (it stays as the free empty-block branch of §4.4) |
| T2 | Sorted-block masks, simulation only: per input, sort channels by calibrated RMS, form groups of 32 in that order, zero whole groups by norm. No requantisation is needed to measure the PPL side. Also print the per-channel log-RMS spread (the σ of §2.2) | one calibration counter and the T1 grid again | sorted frontier minus unsorted frontier under 5 points of block sparsity on both models → the permutation is closed without building anything |
| T3 | Requantisation cost at threshold 0: dequantise each Q4_0 matrix, permute columns (T2's order), requantise to Q4_0, undo the permutation, store in a Q8_0 container (7.7 GB, fits the disk floor; the container's own error is small and is measured by doing the same with the identity permutation), 40-chunk PPL | a gguf-py script, two files, two PPL runs per model | above +2 points on a model → way 1 of §3.1 is closed for that model. Run only if T2 passes |
| T4 | Block-major 4x4 kernel bench in `c3_colkern_bench.c`: identity against stock, cost at kept 1.0 / 0.9 / 0.8 | small addition to the bench | above 1.3 × per touched weight on the host → use the stock layout and skip list only |
| T5 | Counters on the existing H3 cells: share of empty 32-blocks at the element thresholds; realised sparsity on the official prompt and `en.txt` | a counter in the tool, no new PPL | informs §4.4 and §4.6 |
| T6 | Weight-norm score and static shift for the element rule, at the present frontier cells (Mistral 40 and 45 %, Qwen2.5 35 and 40 %) | column norms from the GGUF, one offset table from calibration; four cells per model | neither moves the frontier by 5 points on either model → dropped |
| P2 (device, already planned) | add one cell to the probe kit: the stock q4x4 kernel over the arena with a 10 % and a 20 % block skip list, 32-wide and 64-wide, on the stock layout | two extra cells, a few seconds | GB/s of the weight bytes touched and ms per pass: shows directly whether skipped 72-byte pieces save traffic on the A78 (§2.1) |

### Try list (after the tests; none is to be built on today's evidence)

| # | lever | label | price (*estimated*) | gate |
|---|---|---|---|---|
| Y7 | Skip list for all-zero Q8_0 blocks and blocks under a calibrated norm, stock layout | bit-identical at threshold 0; numerics change above it | compute × (1 − s); traffic about unchanged; at most about 4 % and only if P2 shows the token compute-bound | T1, P2 |
| Y8 | Block-major 4x4 layout with the same skip list | bit-identical layout; numerics change above threshold 0 | at 19 GB/s: 4.7 → 5.2 tok/s at s = 10 %; needs s = 14 % to equal the column kernel at 40 % | T1, T4, P2 |
| Y9 | Two-level column kernel: empty blocks skipped whole, elements otherwise | as Y3/Y4 | removes the fixed 0.56 for empty blocks only | T5 |
| Y10 | Within-block channel sort and 8-column units | bit-identical at threshold 0 | group-8 sparsity at equal energy is 8 to 11 % against 4 to 6 % for 32 (synthetic); kernel unbenched | T1 shows block-32 under 10 % and Y8 is still wanted |
| Y11 | Cross-block permutation with requantisation, `down` input first | **numerics change at threshold 0** (lossy second quantisation) | −0.5 to −1.0 point of budget for `down` only, −2.0 to −3.6 for all inputs, before any skip | T2, T3 |

Generic across models: every item works on any Q4_0 file with 32-input blocks; tables are per file. Per-model
expectations differ (Qwen2.5 has less budget left after any requantisation).

## 6. Sources and how each was read (2026-11-19)

All arXiv pages below were read through a fetch tool that returns a summary of the HTML, not through the PDF; table
values are as that summary reported them and were not checked a second time. "Search summary only" means I did not
open the paper.

- N:M activation benchmark: <https://arxiv.org/html/2509.22166v3> (v3, 2026-02-28). Fetch summary.
- Amber Pruner: <https://arxiv.org/html/2508.02128v1> (2025-08). Fetch summary.
- R-Sparse: <https://arxiv.org/html/2504.19449v1> (2025-04, ICLR 2025). Fetch summary; no PPL values in the text.
- SpQt: <https://arxiv.org/html/2511.04477v1> (2025-11). Fetch summary.
- DIP: <https://arxiv.org/html/2412.01380v2> (2024-12). Fetch summary.
- WiSparse: <https://arxiv.org/html/2602.14452v1> (2026-02). Fetch summary.
- WINA: <https://arxiv.org/html/2505.19427v1> (2025-05). Search summary only.
- PermLLM: <https://arxiv.org/abs/2510.10136> (2025-10, NeurIPS 2025). Search summary only.
- Ripple: <https://arxiv.org/abs/2410.19274> (2024-10). Search summary only.
- PowerInfer-2: <https://arxiv.org/html/2406.06282v3>. Search summary; memo-01.
- SparAMX: <https://arxiv.org/abs/2502.12444v1> (2025-02). Search summary only.
- TEAL: <https://arxiv.org/html/2408.14690v3>. As read for memo-01.
- Repository: `campaigns/03-dense15-5pct/reports/memo-02-host-screens.md`; `lab/c3/logs/h3-*.log`,
  `h6-colkern.log`, `h6b-blockskip.log`, `calib2-*.log`; `lab/llama.cpp-c3/ggml/src/ggml-cpu/repack.cpp`
  (`repack_q4_0_to_q4_0_4_bl`); `tools/c3_colskip.py`, `tools/c3_colkern_bench.c`, `tools/c3_h3_grid.sh`.
- The synthetic model of §2.2 is a 30-line numpy script, not kept in the repository; its definition is the first
  paragraph of §2.2 (seed 0).
