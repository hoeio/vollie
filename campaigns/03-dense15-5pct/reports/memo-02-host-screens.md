# c3 memo-02: host screens H1, H3, H6, H7 (+ X4), and the device probe kit design

Date: 2026-11-18. Author: Tom (`claude-opus-5-5`), on the owner's laptop (Apple M4 Pro, 8 P + 4 E cores). Every
number here is ***host*** (relative only) unless it is marked *estimated* or quoted with its label. Nothing in this
memo was measured on the phone. Written as results arrived (2026-11-18 21:10 to 2026-11-19 02:00 KST); final.

## 0. Setup

- **Lab tree** `lab/llama.cpp-c3` (not tracked): base `620d2225c` + campaign 2's series `patches/llama.cpp-c2/0001` to
  `0009` applied in order (committed locally in the lab clone as "c2 series 0001-0009"). Build: CPU only, as the c2
  host tree: `cmake -DCMAKE_BUILD_TYPE=Release -DGGML_METAL=OFF -DGGML_NATIVE=OFF
  -DGGML_CPU_ARM_ARCH=armv8.2-a+dotprod+fp16 -DGGML_CPU_KLEIDIAI=OFF -DGGML_BLAS=OFF`. With that arch flag the host
  takes the phone's repack path (Q4_0 -> 4x4, dotprod, no i8mm). No c2 knob is set anywhere in this memo, so every
  run is the stock graph.
- **No runtime change was made.** All screens are numerics simulations through llama.cpp's `cb_eval` hook (the
  method of campaign 1's `tools/dense_colskip.py`), so `patches/llama.cpp-c3/` does not exist yet. The first c3
  patch will be the column kernel if H3 and H6 call for it.
- **Files** (downloaded to `lab/c3/models`, sha256 checked against the Hugging Face LFS ids):

| model | file (repo @ commit) | sha256 | role |
|---|---|---|---|
| Mistral-7B-Instruct-v0.3 | `Mistral-7B-Instruct-v0.3-Q4_0.gguf` (second-state/Mistral-7B-Instruct-v0.3-GGUF @ `1b61eaeb`) | `897f7877…c31c0e32` | served-quantisation stand-in (plain Q4_0, Q6_K head); bartowski has no Q4_0 for this model |
| Mistral-7B-Instruct-v0.3 | `Mistral-7B-Instruct-v0.3-Q8_0.gguf` (same repo and commit) | `3776268f…884b0d9a0` | **F16 proxy** (Q8_0, labelled) |
| Qwen2.5-7B-Instruct | `qwen2.5-7b-instruct-q4_0-0000{1,2}-of-00002.gguf` (Qwen/Qwen2.5-7B-Instruct-GGUF @ `bb5d59e0`, the publisher's own) | `09cda9ec…08a6a2`, `db92b5c3…e564` | served-quantisation stand-in (Q4_0, Q6_K head) |
| Qwen2.5-7B-Instruct | `qwen2.5-7b-instruct-q8_0-0000{1,2,3}-of-00003.gguf` (same repo and commit) | `26e30cb4…`, `e93856ce…`, `8f29c0e4…` (match the LFS ids) | **F16 proxy** (Q8_0, labelled) |

  All model files were deleted after use (disk floor 10 GB; 22 GB free at the end). F16-proxy PPL, 40 chunks,
  `-b 512` (`llama-perplexity -c 512 -b 512 --chunks 40`, logs `lab/c3/logs/ppl-*-q8_0-b512-40.log`): Mistral
  Q8_0 **5.7718** (Q4_0 5.8844 = Q8_0 × 1.0195); Qwen2.5 Q8_0 **6.7178** (Q4_0 6.9575 = × 1.0357). Both Q4_0 files
  are far inside the F16 × 1.20 ceiling (with Q8_0 standing in for F16).

  The served quantisation of each model is still to be chosen (H2, not run here); every budget below is therefore
  "this Q4_0 file × 1.05" on the same chunks, a stand-in for the real budget.
- **Text**: wikitext-2-raw test (gate text) and train (calibration only), copied from the sibling `vollie` folder.

## 1. H1: byte inventory from GGUF headers

Tool: `tools/c3_gguf_inventory.py` (new). For a URL it fetches only the header with HTTP range requests (8 MB, doubled
until the tensor table is complete) into a sparse file, so no model was downloaded for this table. "Per token" is every
tensor except the lookup tables (`token_embd`, Gemma's `per_layer_token_embd`), plus the embedding when the head is
tied. "Repacked" = what the phone build (arm64 dotprod, no i8mm, `repack.cpp` at `620d2225c`) copies into anonymous
memory: Q4_0/IQ4_NL/Q8_0/MXFP4 with rows % 4, Q4_K/Q5_K/Q6_K with rows % 8, never a lookup table.

```sh
python3 -I tools/c3_gguf_inventory.py lab/llama.cpp-c3/gguf-py <url-or-path>[+<split 2>] ...   # log: lab/c3/logs/h1-inventory.txt
```

| model (file read) | params, B | layers, d, FFN | head | bytes/token, GB | FFN / attn / head, GB | repacked copy, GB |
|---|---|---|---|---|---|---|
| Mistral-7B-v0.3 (second-state Q4_0) | 7.248 | 32, 4096, 14336 | untied Q6_K, 32,768 rows | **4.037** | 3.171 / 0.755 / 0.110 | 4.036 |
| Qwen2.5-7B (Qwen Q4_0, 2 splits) | 7.616 | 28, 3584, 18944 | untied Q6_K, 152,064 rows | **4.119** | 3.208 / 0.463 / 0.447 | 4.118 |
| Llama-3.1-8B (bartowski Q8_0) | 8.030 | 32, 4096, 14336 | untied Q8_0, 128,256 | 7.975 at Q8_0; *estimated* 4.5 at Q4_0 + Q6_K head | 5.99 / 1.43 / 0.56 | all |
| Yi-1.5-6B (bartowski Q8_0) | 6.061 | 32, 4096, 11008 | untied Q8_0, 64,000 | 6.162 at Q8_0; *estimated* 3.4 at Q4_0 | 4.60 / 1.28 / 0.28 | all |
| Qwen3-8B (Qwen Q8_0) | 8.191 | 36, 4096, 12288 | untied Q8_0, 151,936 | 8.042 at Q8_0; *estimated* 4.4 at Q4_0 + Q6_K head | 5.78 / 1.60 / 0.66 | all |
| Gemma 4 E4B (ggml-org Q4_0) | 7.463 | 42, 2560, 10240 | **tied** to a Q8_0 embedding, 262,144 rows | **2.990** (head 0.713 of it) | 1.858 / 0.330 / 0.713 (+0.088 PLE proj, norms) | 2.219 (head not repacked) |
| Gemma 4 E4B (Google QAT Q4_0) | 7.463 | same | tied to a Q6_K embedding | **2.827** (head 0.551) | 1.858 / 0.330 / 0.551 | 2.219 |
| RWKV-7 G1i 7.2B (Q8_0) | 7.200 | 32, 4096, 16384 | untied Q8_0, 65,536 | 7.644 at Q8_0 | cmix 4.56 / tmix 2.79 / 0.29 | 7.13 |

Findings (*host*, from headers):
- Both lead candidates read about 4.0 to 4.1 GB per token at Q4_0, and **all of it is repacked** into an anonymous
  copy on the phone build (the Q6_K heads too): the repack doubles the resident set unless memo-01 Y1 (layout in a
  file) is done. Qwen2.5's head is 11 % of the token (0.447 GB of Q6_K), Mistral's 3 %.
- Gemma 4 E4B: the per-layer table (`per_layer_token_embd`, 262,144 × 10,752, 1.585 GB at Q4_0 or 2.312 GB at Q6_K)
  is a lookup; per token the file reads 2.83 to 2.99 GB, of which the tied head is 0.55 to 0.71 GB (19 to 24 %),
  and the tied head is not repacked (it lives with the embedding). This agrees with memo-03 §3.
- The base `620d2225c` has `gemma4`, `gemma3n`, `qwen2`, `qwen3`, `rwkv7` in `src/llama-arch.cpp`.

## 2. Identity of the screening tool (*host*)

`tools/c3_colskip.py` (new) scores exactly like `llama-perplexity` (ctx 512, BOS replaces the first token of each chunk,
tokens 257 to 511 scored). Campaign 1's `dense_colskip.py` did not replace the first token by BOS; it was right for
Granite only. Checks on Mistral Q4_0:

| check | llama-perplexity | tool, no edit | tool, every input edited with threshold 0 |
|---|---|---|---|
| `-b 512`, 4 chunks | [1]5.2873 [2]5.4066 [3]6.3280 [4]7.0083 | same 4 values | same 4 values; per-chunk NLL equal to the last bit to the no-edit run |
| `-b 512 -ub 1`, 2 chunks | [1]5.2980 [2]5.4042 | same | same; per-chunk NLL bit-equal |

So the hook (and the graph splits it causes) does not change the numerics, and a zeroed input is the only change in
every cell below.

## 3. H3: input-sparsity tolerance (*host*)

**Method.** Per weight input (4 per block: `qkv` = the input of `attn_q/k/v`, `o` = of `attn_output`, `gu` = of
`ffn_gate/up`, `down` = of `ffn_down`; 128 inputs on Mistral, 112 on Qwen2.5) one absolute threshold t is
calibrated on **wikitext-2 train** (16 chunks of 512 tokens, 32,768 random |x| samples per input per chunk, the
quantile at the target sparsity, TEAL-style). On the test text every input with |x| < t is set to 0 before the stock
mat-mul; that is exactly what skipping its weight column computes. The rule is element-wise and does not depend on
the batch, but the stock kernels do, so `-b 512` and `-ub 1` are both run. "dense first N" keeps the tokens at
positions < N of each chunk unsparsified. Realised sparsity on the test text is counted per group.

```sh
python3 -I tools/c3_colskip.py calib lab/llama.cpp-c3/build/bin <model> lab/c3/text/wiki.train.raw 16 lab/c3/thr2-<m>-q4_0.npz
tools/c3_h3_grid.sh lab/llama.cpp-c3/build/bin <model> lab/c3/thr2-<m>-q4_0.npz lab/c3/text/wiki.test.raw <chunks> <512|1> <log> [cells]
#   cell = <sparsity>:<all|qkv,o|gu|down|qkv=a,o=b,gu=c,down=d>:<dense first N>; logs lab/c3/logs/h3-<m>-<b512|ub1>-<chunks>.log
```

(The 16-chunk grid used the first table, `thr-*.npz`, with levels 0.2/0.3/0.4/0.5/0.6; the 40-chunk cells use
`thr2-*.npz`, which adds 0.35 and 0.45. Same samples, same seed, so the shared levels are the same thresholds.)

Budget used here (stand-in, §0): this Q4_0 file's PPL on the same chunks × 1.05; the memo-01 pass line is +4.5 %
(0.5 points of margin).

### 3.1 Short grid, 16 chunks, `-b 512`

Stock PPL: Mistral 7.7493, Qwen2.5 8.2341 (Q8_0 proxy: Mistral 7.5864, Qwen2.5 7.8862).

| cell | Mistral | Qwen2.5 |
|---|---|---|
| all four groups 20 % | −0.39 % | +0.73 % |
| all 30 % | +0.23 % | +2.00 % |
| all 40 % | +1.96 % | +7.22 % |
| all 50 % | +10.34 % | +20.96 % |
| all 20 / 30 / 40 / 50 %, dense first 1 | +0.53 / +0.75 / +3.89 / +9.39 % | +1.44 / +2.48 / +7.84 / +19.72 % |
| all 20 / 30 / 40 / 50 %, dense first 8 | +0.38 / +0.32 / +3.01 / +9.94 % | +0.88 / +3.04 / +7.65 / +18.61 % |
| attention only (`qkv`, `o`) 40 / 50 % | −0.15 / +1.05 % | +1.24 / +3.96 % (20 / 30 %: +0.09 / +0.22) |
| `gate`+`up` input only 40 / 50 % | +2.10 / +8.27 % | +7.12 / +16.41 % (20 / 30 %: +1.29 / +2.70) |
| `down` input only 40 / 50 % | +0.63 / +1.21 % | −0.27 / +0.70 % (20 / 30 %: +0.24 / −0.07) |

Realised sparsity on the test text is within 1.5 points of the target in every cell (e.g. Mistral 40 %: down 0.413,
gu 0.397, o 0.420, qkv 0.396).

### 3.2 Frontier, 40 chunks, `-b 512`

Stock PPL: Mistral **5.8844** (Q8_0 proxy 5.7718, Q4_0 = Q8_0 × 1.0195), Qwen2.5 **6.9575**.

| cell | Mistral | Qwen2.5 |
|---|---|---|
| all 30 % | – | +1.09 % |
| all 35 % | +0.99 % | +3.10 % |
| all 40 % | +2.44 % | +4.51 % |
| all 45 % | +4.96 % | – |
| mixed: attention 50 %, `down` 50 %, `gate`+`up` 35 % (Mistral) / 30 % (Qwen) | +3.78 % (byte-weighted 42 %) | +3.95 % (byte-weighted 38 %) |

### 3.3 Frontier, 40 chunks, `-ub 1`

| cell | Mistral (stock 5.8810) | Qwen2.5 (stock 6.9607) |
|---|---|---|
| all 30 % | – | +1.17 % (`-b 512`: +1.09 %) |
| all 35 % | – | +2.75 % (`-b 512`: +3.10 %) |
| all 40 % | +2.92 % (`-b 512`: +2.44 %) | – |

`-ub 1` differs from `-b 512` by −0.35 to +0.48 points at the same thresholds; realised sparsity is the same to
three decimals (the rule is per element). Both settings are inside the margin at the frontier cells.

### 3.4 Reading (*host*)

- **Largest model-wide sparsity inside +5 % with 0.5 points of margin (40 chunks, at `-b 512` and at `-ub 1`):
  Mistral 40 % (+2.44 % / +2.92 %), Qwen2.5 35 % (+3.10 % / +2.75 %)**; Qwen2.5 at 40 % is +4.51 %, 0.01 point outside the margin. Both are above
  memo-01's 25 % kill line, so **the column lever is not closed by H3** on either model. Published expectation
  (memo-01 H3) was Mistral 35 to 40 %, Qwen2.5 about 35 %: met.
- **What generalised (both models):** the `gate`/`up` input is the sensitive group and the `down` input the
  tolerant one (50 % on `down` alone costs about 1 %); attention inputs take 40 % for about 1 %. Keeping the first
  1 or 8 tokens dense does **not** help on either model at 16 chunks (it is worse in 7 of 8 cells [*correction 2026-11-19 (Ebony): 12 of 16 cells, N = 1 and 8 at four levels on two models; worse in every cell up to 40 %, better only at 50 %*]); the position
  rule is not needed.
- **What did not generalise:** the level. Qwen2.5 is about 5 points less tolerant than Mistral. The 16-chunk grid
  also overstated Qwen2.5's cost at 40 % (+7.2 % on 16 chunks against +4.5 % on 40), so 16 chunks rank cells but
  do not decide a frontier; the claim-relevant number is the full test set on the owner's laptop.
- **Per-group mixes did not beat uniform sparsity per byte** on either model (Mistral: 42 % byte-weighted mixed
  +3.78 % against uniform 40 % +2.44 % and 45 % +4.96 %, about the uniform line; Qwen2.5 the same).
- **Small negative deltas** (for example Mistral 20 %: −0.39 % on 16 chunks) are within what any perturbation of this
  size does on 16 chunks; they are not a gain.

## 4. H6: column-kernel microbench (*host*, single thread, relative only)

Tool: `tools/c3_colkern_bench.c` (new, standalone). The stock kernel is ggml's `ggml_gemv_q4_0_4x4_q8_0` (NEON
dotprod path, the phone's non-KleidiAI Q4_0 decode kernel) copied with its repack. The column kernel keeps the
file's Q4_0 blocks and scales but stores them input-major in tiles of TR rows: per (tile, 32-input block) the TR
fp16 scales, then 32 columns of TR nibbles, so a dropped input skips TR/2 contiguous bytes. Per block the kept
columns are summed exactly in int16 (|w| <= 8, |x| <= 127, 32 terms < 2^15), then `acc = fma(acc, isum, dx*dw)` in
block order, which is the stock kernel's float step.

```sh
clang -O3 -march=armv8.2-a+dotprod+fp16 -DTR=1024 -o lab/c3/bench/colkern1024 tools/c3_colkern_bench.c
lab/c3/bench/colkern1024 14336 4096 30; lab/c3/bench/colkern1024 4096 14336 30      # log: lab/c3/logs/h6-colkern.log
```

**Identity: the column kernel is bit-identical to the stock 4x4 kernel** (all 14336 or 4096 outputs, `memcmp`) at
density 1.0 and at every density against the stock kernel fed the same zeroed q8 inputs, on both shapes and both
tile sizes. So Y3 can be an exact (B) change against ggml's kernel; against KleidiAI it would be a declared
numerics change (memo-01 §4).

| shape (rows x inputs), TR | density 1.0 | 0.8 | 0.6 | 0.5 | 0.4 |
|---|---|---|---|---|---|
| 14336 x 4096, TR 1024: time / stock dense | 2.19 | 1.85 | 1.56 | 1.42 | 1.21 |
| same: cost per touched weight, x stock | 2.19 | 2.34 | 2.63 | 2.82 | 3.02 |
| 4096 x 14336, TR 1024: time / stock dense | 2.19 | 1.88 | 1.58 | 1.40 | 1.20 |
| same: cost per touched weight | 2.19 | 2.34 | 2.64 | 2.80 | 3.04 |
| 14336 x 4096, TR 256: per touched weight | 2.67 | 2.77 | 2.91 | 3.02 | 3.23 |

Stock: 0.63 ms per 33 MB matrix on one M4 P-core (about 52 GB/s, *host*). Random masks (independent per input).

Reading (*host*):
- **On the host the column kernel is slower than the dense stock kernel at every density down to 40 %**
  (1.20 x at 40 %). Per touched weight it costs 2.2 x (dense) to 3.0 x (40 % density) of `sdot`.
- The cost rises as density falls because the per-(row, block) float step and the scale reads do not shrink with
  density; that step is what makes the kernel exact. The int part uses `smlal` (8 MACs per instruction plus an
  int16 read-modify-write in L1) against `sdot`'s 16. A first variant that kept the sums in registers and walked
  32 columns at 512-byte stride was slower (3.6 x dense; not kept).
- Against memo-03 §5's read-rate kill line (break-even about 3.5 x at 19 GB/s, about 1.7 x from 39 GB/s): the host
  ratio at the densities H3 allows (60 to 65 % kept, §3) is **about 2.6 x** (2.63 measured at 60 %, 65 % interpolated at about 2.55), under the 19 GB/s line and over the
  39 GB/s line. The phone value (A78, `GGML_CPU_BENCH`-style row) is still needed; the host does not decide it.

## 5. H7: sublayer ablation and X4 per-token FFN skip

### 5.1 X4: per-token FFN-sublayer skip (memo-03 §5; *host*, 16 chunks, `-b 512`)

**Rule.** For token t at layer L, r = ‖FFN output‖ / ‖residual entering the FFN‖ (L2 per token). The FFN of (t, L)
is skipped (adds nothing) when the rule's r is below one global threshold, calibrated on train so that a target
fraction of all (token, layer) pairs falls below it. `oracle` uses r of the same (t, L), which is known only after
the FFN ran: it is an **upper bound** for any predictor and saves no bytes itself. `prev` uses r of (t, L−1), which is
causal (the FFN of L need not be read); layer 0 never skips. "BOS kept" = the first position never skips
(`C3_DENSE_FIRST=1`).

```sh
[C3_DENSE_FIRST=1] python3 -I tools/c3_colskip.py ffnskip lab/llama.cpp-c3/build/bin <model> lab/c3/text/wiki.test.raw 16 512 lab/c3/thr2-<m>-q4_0.npz <oracle|prev> <0.05|0.1|0.2>
#   logs lab/c3/logs/x4-<m>-b512-16.log; stock 16 chunks: Mistral 7.7493, Qwen2.5 8.2341
```

| rule, target | Mistral: skipped (test), ΔPPL | Qwen2.5: skipped, ΔPPL |
|---|---|---|
| oracle 5 / 10 / 20 % | 3.6 % +0.93 % · 6.1 % +2.76 % · 11.5 % +11.6 % | 2.7 % **+20.5 %** · 4.8 % +24.3 % · 9.2 % +35.5 % |
| oracle, BOS kept | 3.4 % +1.28 % · 5.9 % +3.11 % · 11.3 % +11.5 % | 2.4 % +1.77 % · 4.6 % +4.89 % · 9.1 % +11.2 % |
| prev 5 / 10 / 20 % | 4.1 % +4.10 % · 6.9 % +68 % · 12.6 % +99 % | 3.1 % +3.96 % · 5.7 % +11.2 % · 10.6 % +25.5 % |
| prev, BOS kept | 3.9 % +4.34 % · 6.7 % +68 % · 12.5 % +99 % | 3.0 % +5.79 % · 5.6 % +13.4 % · 10.6 % +28.4 % |

Reading (*host*): **X4 is closed on both models.** Even the oracle bound stays inside +4.5 % only up to about 5 to
6 % of (token, layer) pairs on Mistral and about 4 % on Qwen2.5 (BOS kept); the causal rule only up to about 3 to 4 %.
At 78 % FFN share of the token's bytes (§1) that is at most about 3 % of the bytes per token (*estimated*), against
a +4 % PPL cost. What generalised: the cost curve is steep on both models (doubling the skip fraction costs 3 to 15
times as much). What did not: on Qwen2.5 skipping the first token's FFN alone costs about +18 % (the oracle row
without BOS kept), because that token's residual norm is very large (its r is tiny) and it is the attention sink;
Mistral does not show this. `-ub 1` was not run for X4: the lever is closed at `-b 512` already, at an upper bound.

### 5.2 H7: sublayer ablation (*host*, 8 chunks, `-b 512`, every sublayer of both models)

The output of one `attn_output` or `ffn_down` mat-mul is set to 0, so that sublayer adds nothing to the residual.
Stock on the same 8 chunks: Mistral 7.2286, Qwen2.5 7.1072 (first 8 per-chunk NLLs of the 16-chunk stock run).

```sh
python3 -I tools/c3_colskip.py ablate lab/llama.cpp-c3/build/bin <model> lab/c3/text/wiki.test.raw 8 512 <attn|ffn> <layer[,layer...]>
#   logs lab/c3/logs/h7-<m>-b512-8.log, sorted tables lab/c3/logs/h7-<m>-table.txt; sets: lab/c3/logs/h7set-mistral-b512-20.log
```

| | Mistral | Qwen2.5 |
|---|---|---|
| cheapest attention sublayers | attn 5 −1.20 %, attn 4 −0.70 %, attn 28 −0.27 %, attn 19 −0.05 %, attn 2 +0.05 %; 10 of 32 at or under +1.0 % | attn 25 +1.51 %, attn 17 +2.71 %, attn 16 +3.31 %; none under +1.0 % |
| cheapest FFN sublayers | ffn 8 +1.74 %, ffn 12 +1.80 %, ffn 10 +1.80 % | ffn 18 +6.57 %, ffn 16 +7.43 %, ffn 5 +7.49 % |
| most expensive | ffn 0 (PPL × 10⁴), ffn 1 +1374 % | attn 0 (× 1,455), attn 27 +209 %, ffn 0 +170 % |

Sets on Mistral, 20 chunks (stock 7.4696): attn 5 alone **+0.39 %**; attn 5 + 4 **+11.7 %**; the 5 cheapest +15.7 %;
the 10 cheapest +53 %. Single costs do not add up: two individually free sublayers together cost +11.7 %.

Reading (*host*): **closed on Qwen2.5 by memo-01's rule** (cheapest sublayer +1.51 %, over +1.0 %). On Mistral one
attention sublayer is free on its own, but sublayers do not compose, so the lever is worth one attention sublayer,
about 24 MB, 0.6 % of the token's bytes (*estimated*). **Closed in practice on both**; what generalised: FFN sublayers
are never free on a dense 7B (cheapest +1.7 % and +6.6 %), and layer 0 is essential on both. What did not: Mistral has
cheap attention sublayers in layers 2 to 5 and 18 to 28, Qwen2.5 has none. `-ub 1` was not run (closed at `-b 512`).

### 5.3 H4: RWKV-7 G1 ReLU² exact-zero count (*host*)

File: `rwkv7-g1i-7.2b-Q4_K_M.gguf` (shoumenchougou/RWKV7-G1i-7.2B-GGUF @ `290fa74e`, sha256 `bb957c0f…3ee129ff0`;
a community quant, Q4_K_M because the Q8_0 did not fit the disk floor; downloaded, counted, deleted). Text: 4 chunks
of wikitext-2 **test** (the count needs no calibration split), `-b 512`.

```sh
python3 -I tools/c3_colskip.py calib lab/llama.cpp-c3/build/bin <rwkv7 gguf> lab/c3/text/wiki.test.raw 4 lab/c3/thr-rwkv7.npz 4
```

- **90.5 % of the inputs of `channel_mix_value` are exactly 0.0** (the ReLU² output), per layer 78 % (layer 31) to
  99 % (layer 0), lowest in layers 16 to 19 (83 to 84 %) and the last layer. The input of `channel_mix_key` has no
  exact zeros.
- Above memo-01's 70 % line, so **the owner question on RWKV-7 is worth asking.** Bytes (*estimated* from §1):
  `channel_mix_value` is 16384 x 4096 per layer, half of the channel mix, about 2.28 GB of 7.64 GB per token at Q8_0
  (30 %). Skipping its zero inputs is **exact** (no numerics change), but it is an input-side skip, so it needs
  the column layout and kernel of H6 (or a row-gather of the transposed matrix); the 90 % zeros make H6's per-weight
  cost far less important than on the SiLU models (10 % kept).
- Not measured: the union of nonzero channels over 32 decode tokens (H5's question, for residency), RWKV-7's stock
  PPL, and the same count on a Q8_0 file.

## 6. Device probe kit design (P1, P2, X1, X2): built 2026-11-19 as kit r29 (see §6.1)

Lead's memo-03 §5: one kit, about 15 phone-minutes, built after the host screens. This is the design; nothing below
is built or measured. It starts from campaign 2's kit r28 (same package line, run protocol, r16 cool-down gate,
power-saving state and charger state recorded per block) and adds one runtime knob and one kit page.

**Runtime side: `patches/llama.cpp-c3/0001-probe.patch` (planned).** One knob, `VOLLIE_PROBE=<list>`, off when
unset; when set, the CPU module runs the listed probes once at backend init and prints one `c3-probe:` line per
cell (stderr, `VOLLIE_LINES` as in r28). No graph or kernel changes, so the knob is trivially identity-safe (it
runs before any graph); the identity check is still done (off = stock per chunk, three settings).

| block | what it does | lines | phone time (*estimated*) |
|---|---|---|---|
| P1 resident capacity | write a scratch file of N GB to the app's internal storage once (N = 2.5, 3.0, 3.5, 4.0, 4.5; deleted at the end), map it `PROT_READ` and, as a second arm, `PROT_EXEC` (c2's exec map), read it through 20 times with 3 threads; per pass: seconds, GB/s, major faults (`getrusage`), `Cached`, `SwapFree`, the app's `VmSwap` | `c3-probe: p1 n=<GB> map=<r\|x> pass=<i> s= gbs= majflt= cached= swapfree= vmswap=` | 4 to 5 min (the write is about 1 GB/s; the passes run at DRAM speed when resident) |
| P2 dense read rate | a 1 GB anonymous arena: (a) plain `ldp` read loop and (b) the stock `ggml_gemv_q4_0_4x4_q8_0` over 14336 x 4096 slices walked cold through the arena, with 1, 2, 3 A78 threads, X1 alone, X1 + 3 A78, and 3 A78 + 4 A55; each cell 2 s; clocks of all three policies before and after | `c3-probe: p2 kern=<read\|q4x4> cores=<set> gbs= ns_per_w= mhz=` | 2 min |
| P2 `sdot` ceiling | the same gemv on a 256 KB hot slice (L2-resident) per core class, 1 thread pinned (campaign 1's r13b pinning and `sched_getcpu` checks) | `c3-probe: p2hot core=<a78\|x1\|a55> ns_per_w=` | 1 min |
| X1 bus headroom (CPU) | multi-stream: 1, 2, 4 independent streams per thread on the 3 A78s (separate 256 MB regions, interleaved loads), to see whether more outstanding misses lift the read rate | `c3-probe: x1 streams=<k> cores=<set> gbs=` | 1 min |
| X2 GPU read rate | OpenCL through `dlopen("libOpenCL.so")` (as campaign 1's GPU runs): a `float4` sum kernel over a 512 MB buffer (`CL_MEM_ALLOC_HOST_PTR`), and a Q4_0 dequant-dot kernel on the same 14336 x 4096 matrix; then **CPU and GPU together**: the P2 3-A78 q4x4 cell running while the GPU kernel runs, both rates reported | `c3-probe: x2 kern=<read\|q4> gbs=` and `c3-probe: x1gpu cpu_gbs= gpu_gbs= sum_gbs=` | 2 min |
| gates | kit r16 cool-down gate before each block; heat series and clocks recorded | as r28 | 4 to 5 min |

Total about 15 minutes (*estimated*). Order: P2 and X1 first (cool phone), then X2, then P1 last (P1 fills memory and
may evict other apps; it is the only block that can disturb the phone state, so nothing timed follows it). Power
saving off, charger out, deep-sleeping list recorded (campaign 2's claim conditions). The parser is a new
`tools/c3_probe_parse.py` that tabulates the lines; the prediction bands come from memo-03 (19 to 51 GB/s, 39 GB/s
for a same-bus Adreno, *published*) and are written into the pre-registration before the run, not here.

What each block decides: P1 the largest body that can be read every token (memo-01 P1 kill: under 3.3 GB no Q4_0
6 to 8.5 B body fits); P2 the read rate r that sets H6's kill line (memo-03 §5) and reopens R20 under 28 GB/s; X1 and
X2 whether a CPU + GPU split (memo-03 §2 rank 1) or multi-stream GEMV (rank 2) has bus headroom to use.

### 6.1 Build (2026-11-19), with the lead's six fixes

The design above was approved; the build differs from it as follows (the table above is kept as written).

- **Runtime:** `patches/llama.cpp-c3/0001-probe.patch`, on top of the r28 runtime (`patches/llama.cpp` 0001-0003 +
  `patches/llama.cpp-c1-olmoe` 0001-0012). The probe is a libllama call, `llama_vollie_probe(block, dir, out, cap)`,
  not a backend-init hook: the kit calls it per block, so the r16 gate can run between blocks. Off unless
  `VOLLIE_PROBE` is set: it returns -1 and does nothing. It is a new file only, so nothing on the model path changes.
- **Identity (*host*, Mistral Q4_0):** with the knob unset, `llama-perplexity` running values are identical to the r28
  runtime and to stock `620d2225c`:
  - `-b 512`, 4 chunks: [1]5.2873 [2]5.4066 [3]6.3280 [4]7.0083;
  - `-ub 32`, 4 chunks: [1]5.3060 [2]5.4174 [3]6.3351 [4]7.0187;
  - `-ub 1`, 2 chunks: [1]5.2980 [2]5.4042.
  With `VOLLIE_PROBE=1` at `-b 512`, also identical. Logs: `lab/c3/logs/ident-probe-*.txt`.
- **The six fixes:**
  1. P1 steps are 2.7, 3.0, 3.4, 4.2 and 4.6 GB. The scratch file is written once at the largest step that leaves
     0.5 GB free.
  2. P2 adds an X1 + 2 A78 cell (`7,4-5`). The read and q4x4 sets are 4, 4-5, 4-6, 7, 7+4-5, 4-7, and 0-6.
  3. `p2hot` working sets: A78 256 KB, X1 512 KB, A55 64 KB (half of each class's L2).
  4. Every P2, X1 and X2 cell runs 1 s of warm-up, then 2 s measured. The CPU and GPU share one window in `x1gpu`.
  5. New P2 rows `p2col`:
     - an on-phone identity check: the H6 column kernel and the H6b skip-list kernel against the stock kernel,
       memcmp on one 1024 x 4096 matrix at 60 % kept;
     - then stock, column and skip-list cells at 60 % kept, on 1 A78 and on 3 A78. The parser prints column/stock and
       skip/stock per touched weight.
  6. `x1gpu` lines name `gpu_kern=read|q4` and `cpu_kern=q4x4`. Both GPU kernels are run together with the CPU.
- **App:** `patches/app-c3/0001-kit-r29-probe.patch`, the button "Run c3 probe r29". The R68 run list is unchanged.
- **APK:** `lab/dist-c3/vollie-c3-probe-r29-1.29.0-29.apk`, 16,430,946 B, sha256
  `02736b568503624fa036f86c994c5407516605522f1db901ed2f0d58b866eaf9`. It is package `com.insaneutils.vollie.c2`,
  versionCode 29, so it installs over r28. Build record: `lab/dist-c3/BUILD-r29.txt`.
  - The APK's `libllama.so` exports `llama_vollie_probe`.
  - The CPU module's sha256 differs from r28's although no CPU-module source changed. Builds are not byte-reproducible
    (the temporary build path is embedded), so the per-chunk check above is the identity evidence.
  - Not uploaded, not committed.
- **Parser:** `tools/c3_probe_parse.py`, whose `--selftest` passes. A host functional run of all four blocks went
  through the parser (`lab/c3/probe-host.json`; host numbers mean nothing for the phone). On the host the
  `PROT_EXEC` file mapping fails, which is expected on macOS. The Apple GPU ran through the same OpenCL path.
- **Phone time (*estimated*):**
  - P2: 23 cells x 3 s plus fills, about 75 s.
  - X1: 9 s.
  - X2: about 15 s.
  - P1: write 4.6 GB in about 4 s, then 5 steps x 2 maps x 20 passes, capped at 45 s per map. That is about 40 s if
    resident, at worst 7.5 min.
  - Gates: 0 to 6 min.
  - **Total: about 4 to 6 min typical, at worst about 16 min**, under the 20 min limit.
  - The six fixes add about 1 min: X1 + 2 A78 6 s, warm-ups about 30 s, the column row about 20 s.

## 7. What closed, what stayed open (*host*; per model)

| test | Mistral-7B-v0.3 | Qwen2.5-7B | generalised? |
|---|---|---|---|
| H1 bytes | 4.04 GB/token at Q4_0, all repacked | 4.12 GB/token, all repacked, head 11 % | yes: both about 4 GB, both doubled by the repack copy unless Y1 |
| H3 column sparsity | **open**: 40 % inside the margin at both settings | **open**: 35 % inside the margin | yes in direction (gate/up input is the sensitive group, down tolerant, dense-first not needed); no in level (Qwen about 5 points lower) |
| H6 column kernel | exact (bit-identical to ggml's 4x4 kernel); host cost per touched weight about 2.6 x at 60 to 65 % kept, slower than dense on the host at every density | same kernel, same shape class | kernel is model-generic; the decision waits for the phone's r and an A78 row (memo-03 §5 line: under 3.5 x at 19 GB/s, under 1.7 x from 39 GB/s) |
| H7 sublayer skip | closed in practice: one free attention sublayer, pairs cost +11.7 % | **closed** by rule (cheapest +1.51 %) | yes: FFN never free, layer 0 essential |
| X4 per-token FFN skip | **closed**: oracle bound about 6 % of pairs inside +4.5 %, causal 4 % | **closed**: oracle 4 % (BOS kept), causal 3 % | yes: at most about 3 % of the bytes per token at a +4 % cost |
| H4 RWKV-7 zeros | – | – | 90.5 % exact zeros into `channel_mix_value`: the owner question is worth asking |
| Q8_0 proxy vs Q4_0 | Q4_0 = Q8_0 × 1.0195 | × 1.0357 | both far inside F16 × 1.20 |

**What the numbers say about 15 tok/s** (*estimated*, from the above and memo-03's model): the only lever these
screens leave open is column sparsity at 35 to 40 %, and it needs a kernel that on the host costs about 2.6 x
`sdot` per touched weight at those densities, against a break-even of 1.7 x to 3.5 x depending on the phone's read
rate. At the 19 GB/s *device* rate of campaign 1 that would at best hold the token time roughly level (*estimated*:
dense = 4.0 GB / 19 GB/s = 210 ms read, about 110 to 130 ms of `sdot` compute from memo-01's 7.6 to 9.3 tok/s cap;
at 40 % sparsity 2.4 GB = 126 ms read [*correction 2026-11-19 (Ebony): the column layout still reads every scale, so 40 % sparsity reads 64 % of the bytes (2,048 + 0.6 × 16,384 of 18,432 B per chunk), about 2.6 GB and 135 ms, not 60 %*] but 0.6 × 2.6 = 1.6 times the compute, about 180 to 210 ms, if the host
ratio carries to the A78); it does not open a path to 15 tok/s on a 4 GB body. The P2 read
rate decides whether it is worth building.

**Not done** (and why):
- H2 (served-quantisation ladder) and the Gemma 4 E4B per-group ladder: not on this list or not affordable on this
  disk (a clean ladder needs the BF16 safetensors, about 15 GB; requantising from Q8_0 is not the publisher's file).
- H5 (active-set stability): not run.
- Llama-3.1-8B and Yi-1.5-6B (reserves): header bytes only (§1).
- The device probe kit (§6): designed, not built. The host screens are done; building it (runtime knob, kit page,
  parser, APK) is the next code task once the lead confirms the design.
- Full-set (all chunks) PPL of any cell: not run; every PPL here is 8, 16, 20 or 40 chunks.

## 8. H3b and H6b: block-granular input sparsity (lead's idea, added 2026-11-19; *host*)

### 8.1 H6b: skip-list and block-major kernels (*host*, single thread, other jobs paused during the run)

Tool: `tools/c3_colkern_bench.c` (extended). The new kernels are:
- **skip list:** the stock `ggml_gemv_q4_0_4x4_q8_0` loop visits only the kept input blocks, in the stock layout.
- **block-major:** the same 4x4 groups stored `[block][row group]`. A dropped input block is then N/4 x 72 B
  contiguous. Blocks are outer and row groups inner, with one float accumulator per row; per row the blocks are
  still added in increasing order with the stock float step.

"Lines touched" is the fraction of the matrix's distinct 64-byte lines that the kept blocks cover, counted exactly
for each layout. Blocks were dropped at random positions.

```sh
clang -O3 -march=armv8.2-a+dotprod+fp16 -o lab/c3/bench/colkern1024 tools/c3_colkern_bench.c
lab/c3/bench/colkern1024 14336 4096 30; lab/c3/bench/colkern1024 4096 14336 30     # log: lab/c3/logs/h6b-blockskip.log
```

**Both kernels are bit-identical to the stock kernel** fed x with the dropped blocks zeroed (memcmp of all outputs),
at every kept fraction, on both shapes.

| kept (realised), 14336 x 4096 | skip list: time / dense | per touched weight | lines touched | block-major: time / dense | per touched weight | lines touched |
|---|---|---|---|---|---|---|
| 1.00 | 1.02 | 1.02 | 1.000 | 0.93 | 0.93 | 1.000 |
| 0.91 | 0.92 | 1.01 | 0.979 | 0.85 | 0.93 | 0.914 |
| 0.67 | 0.70 | 1.04 | 0.868 | 0.63 | 0.93 | 0.672 |
| 0.64 | 0.68 | 1.06 | 0.812 | 0.60 | 0.93 | 0.641 |
| 0.52 | 0.56 | 1.09 | 0.729 | 0.48 | 0.93 | 0.516 |

The 4096 x 14336 shape gives the same picture: skip list 1.01 to 1.18 per touched weight, block-major 0.93 to 0.95.

Reading (*host*):
- **The lead's cost prediction holds.** The skip list costs 1.01 to 1.09 x per touched weight on the stock layout,
  against 2.2 to 3.0 x for the exact column kernel (§4).
- **Ebony's byte correction holds in direction.** In the stock layout, at 64 % of blocks kept, the kept blocks still
  touch 81 % of the lines. That saves about half of the skipped bytes at 64-byte line granularity, not all of them.
  The phone's prefetchers and DRAM bursts may fetch more than the lines read, so the real saving can be lower; the
  `p2col kern=skip` row of the probe measures the stock-layout case.
- **The block-major layout touches exactly the kept fraction** of the lines. It is bit-identical and runs at 0.93 x
  per touched weight on the host. It is the layout question of Y1 (a derived, file-backed layout).


### 8.2 H3b: block-granular input sparsity, PPL (*host*, 40 chunks of wikitext-2 test, Q4_0 files)

Method: same cb_eval tool and test chunks as §3. Thresholds are quantiles of the per-block L2 norm (`b32:`/`b64:`
arrays of `thr4-*.npz`, calibrated on wikitext-2 train, 16 chunks); a block whose norm is below the threshold is
zeroed whole before the stock mat-mul, which is what a kernel that skips that input block computes. "q" is the
train quantile; the realised zeroed fraction on the test text is given because it drifts above q on `down`
(zeroing upstream changes later activations).

```sh
C3_BLOCK=32 tools/c3_h3_grid.sh lab/llama.cpp-c3/build/bin <model> lab/c3/thr4-<m>-q4_0.npz lab/c3/text/wiki.test.raw 40 <512|1> <log> <q>:<all|down>:<dense first N>
#   logs lab/c3/logs/h3b-<m>-<b512|ub1>-40-blk<32|64>.log (RESULT + PERCHUNK lines); C3_BLOCK=64 for 64-wide blocks
```

Stock: Mistral 5.8844 (`-b 512`) / 5.8810 (`-ub 1`); Qwen2.5 6.9575 / 6.9607. Pass line +4.5 % (budget +5 %, §3).

| cell (block 32 unless stated) | Mistral `-b 512` | realised (down / gu / o / qkv) | Qwen2.5 `-b 512` | realised |
|---|---|---|---|---|
| all, q 0.02 | +7.76 % | .025 / .016 / .031 / .016 | +20.89 % | .029 / .018 / .006 / .016 |
| all, q 0.05 | +7.30 % (`-ub 1` +7.27 %) | .066 / .038 / .056 / .039 | +28.38 % (`-ub 1` +28.74 %) | .070 / .041 / .018 / .038 |
| all, q 0.10 / 0.15 | +20.65 / +41.99 % | down .145 / .231 | +82.99 / +315.9 % | down .149 / .250 |
| all, q 0.2 / 0.3 / 0.4 | +74.7 % / x12.7 / x274 | down .317 / .610 / .787 | x16.3 / x486 / x934 | down .374 / .612 / .662 |
| all, q 0.02 / 0.05 / 0.10, dense first 1 | +12.56 / +4.12 / +18.14 % | down .020 / .055 / .125 | +2.11 / +5.59 / +36.90 % | down .024 / .066 / .147 |
| `down` only, q 0.05 | **+1.82 %** | .052 | **+1.23 %** | .045 |
| `down` only, q 0.10 | **+3.72 %** (`-ub 1` **+3.74 %**) | .099 | **+1.98 %** (`-ub 1` **+1.98 %**) | .087 |
| `down` only, q 0.2 / 0.3 / 0.4 | +8.58 / +15.22 / +24.26 % | .187 / .270 / .352 | +4.79 / +9.15 / +15.57 % | .168 / .247 / .327 |
| block 64, all, q 0.05 / 0.1 / 0.2 | +10.90 / +30.49 / +157.9 % | | +48.09 / +204.1 / x76.7 | |
| block 64, `down` only, q 0.2 | +11.44 % | .181 | +6.95 % | .164 |

(The "dense first 1" rows were measured in the order Mistral 0.02 / 0.05 / 0.10, then Qwen2.5; they keep position 0 of
each chunk unsparsified.)

Reading (*host*):
- **Per-model frontier inside +4.5 % at both `-b 512` and `-ub 1`: `down` input only, Mistral about 10 % of `down`
  blocks (+3.72 % / +3.74 %; 19 % fails at +8.58 %), Qwen2.5 about 9 % (+1.98 % / +1.98 %; 17 % is +4.79 %, inside
  the budget but outside the margin).** Every all-groups cell fails on both models without the dense first token; with
  it, the only all-groups cells inside the line are Mistral q 0.05 (+4.12 %, realised 3 to 5 %) and Qwen2.5 q 0.02
  (+2.11 %, realised 1 to 2 %). Neither model holds block-32 sparsity of 20, 30 or 40 % on any group set; the
  lead's 20/30/40 % cells are out by factors, not percent.
- **What generalised:** block granularity is about ten times harsher than element granularity on both models
  (element 35 to 40 % inside the line, §3.4; block about 2 to 5 % model-wide), as Ebony's energy argument predicted;
  `down` is the tolerant group on both; 64-wide blocks are worse than 32 on both; `-ub 1` equals `-b 512` within
  0.4 points on all four cells run at both.
- **What did not generalise:** the first token. Keeping position 0 dense rescues Qwen2.5's low-q all-groups cells
  (+20.9 → +2.1 % at q 0.02) but not Mistral's (+7.8 → +12.6 %). Mistral's all-groups cost is not even monotone in q
  (q 0.02 is worse than q 0.05): it comes from single chunks; at q 0.02 with dense first 1 chunk 32 alone has
  nll 5.84 against at most 2.51 for every other chunk. A rule whose cost is one blown-up chunk out of 40 is not
  safe to ship at any setting this grid found; only the `down`-only cells are smooth in q on both models.
- **Score against Ebony's pre-data prediction (memo-04 work, ebony-blocksparse §2.3):** all four groups, "3 to 10 %
  on both models, Qwen2.5 lower": **missed low** (no all-groups cell passes on either model without the dense
  first token; with it, about 3 to 5 % on Mistral and 1 to 2 % on Qwen2.5, the latter below the band); block 64
  "about a point lower": worse than that (fails at q 0.05 on both). `down` alone, "10 to 25 %": **at the bottom of the
  band** (about 10 % and 9 %). Chance of 20 % model-wide, 0.1: it did not happen.
- **Price (*estimated*):** `down` is 26.9 % of a Mistral layer's weight bytes and 29.1 % of Qwen2.5's (shapes from
  §1). The `down`-only frontier therefore skips about 2.7 % (Mistral) and 2.5 % (Qwen2.5) of the weight bytes, and only
  in the block-major layout (§8.1); in the stock layout the skip list saves about half of that at line granularity.
  That is under 3 % of token time either way. **Block-granular input sparsity is closed as a speed lever on both
  models**; element-granular column sparsity (§3, §4) stays as it was.
