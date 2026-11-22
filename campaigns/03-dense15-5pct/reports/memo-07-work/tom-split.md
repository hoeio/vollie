# memo-07 work: Tom's split-location cells (test 2), the missing grid cells (test 3), and K1's layout

Date: 2026-11-22. Author: Tom (`claude-opus-5-5`). Every PPL here is *host* (the owner's laptop, Apple M4 Pro, lab
build of the campaign-3 runtime, Q4_0 files of memo-06 §5). Each one is 40 chunks of wikitext-2 test at ctx 512, scored
like `llama-perplexity`, at `-b 512` (ub 512) and at `-ub 1`. Deltas are against stock at the **same setting** (as in
tom-screens.md §5). The budget is stock Q4_0 × 1.05 with a pass line of +4.5 %. Weight shares are *estimated*
(arithmetic on the GGUF tensor sizes). Jobs ran serially, one 7B process at a time, under the 5 GB disk watchdog.

## 0. Setup

- **Tool change** (`tools/c3_colskip.py`):
  - **`C3_DENSE_W=attn_k,attn_v`** takes those weights out of the target set, so their mat-muls read the unedited
    input. Llama-2's graph reads `attn_norm` in the order Qcur, Vcur, Kcur, the same at ub 512 and at ub 1. The consumer
    plan therefore zeroes the shared `qkv` input for `attn_q` and restores it after Qcur. The printed plan reads
    `restore:qkv` in every layer of every k, v dense cell. A graph that read K or V before Q would be refused, not run
    wrongly.
  - **The tag** gains ` densew <names>`.
  - **The quantile set** `QS` gains 0.25 and 0.42.
- **Check of the mechanism** (Llama-2, 2 chunks, ub 512). `C3_DENSE_W=attn_q,attn_k,attn_v` with groups `all` gives
  per-chunk NLL bit-identical to groups `o,gu,down` (1.7911861779847056, 1.9606204103102398 for both). With only k and v
  dense, the values differ (1.7961…, 1.9303…), so q is still sparsified.
- **New threshold tables**, `lab/c3/thr6-<m>-q4_0.npz`:
  - calibrated as before on 16 chunks of wikitext-2 train;
  - at every level they share with the thr2 tables of the phase-2 screens they are equal, element for element (Llama-2
    417 / 417 arrays, DeepSeek 421 / 421, Yi 449 / 449), so the 25 % and 42 % cells sit on the same calibration as the
    20 / 30 / 40 / 45 % cells;
  - sha256: Llama-2 `e0fe4ca8…a8e92d`, DeepSeek `ce582097…31bf01`, Yi `1e08e17d…384250`.
- **Logs** (untracked): `lab/c3/logs/u1-<model>-<b512|ub1>-40.log`. Per-chunk NLL is in the `PERCHUNK` lines. The
  references are the phase-2 logs `h3-<model>-<b512|ub1>-40.log`.
- **Share of the block's mat-mul weights in k and v** (*estimated*, from the GGUF headers):

  | model | attention | k and v share of weights | q share of weights |
  |---|---|---|---|
  | Llama-2-7B | MHA, 32/32 | 0.166 | 0.083 |
  | DeepSeek-7B | MHA, 32/32 | 0.166 | 0.083 |
  | Yi-1.5-6B | GQA, 32/4 | 0.024 | 0.097 |

  So k, v dense at 30 % skips a nominal 0.30 × 0.834 = **25.0 %** of the weights on Llama-2 and DeepSeek, and
  0.30 × 0.976 = **29.3 %** on Yi. On the MHA models, **the share-matched comparison is the uniform 25 % cell of
  test 3**.
- **Realised zeroed fractions** are within 0.01 to 0.03 of nominal in every cell (logs). `o` runs about 0.02 high at
  25 % and at 42 %.

## 1. Test 2: uniform 30 %, k and v inputs dense

Predictions written before the cells:
- **Ebony** (ebony-compute.md §2.3, U1 and list item 1): hypothesis A (context error through full-size k and v), P 0.40.
  - "k and v dense lifts the full-attention frontier to ≥ 0.30 → hypothesis A", and "the frontier of the rest should
    approach 0.35".
  - On GQA (Yi), A implies a small change.
- **Plain 30 %** (phase 2), the reference: Llama-2 +5.69 %, DeepSeek +5.13 %, Yi −0.71 % at `-b 512`. No plain 30 %
  cell was run at `-ub 1` on these three.

| model | setting | stock | k, v dense 30 % | Δ vs stock (SE) | plain 30 % Δ | paired Δ, k, v dense − plain (SE) | Ebony's prediction | verdict |
|---|---|---|---|---|---|---|---|---|
| Llama-2 | `-b 512` | 7.4071 | 7.5031 | **+1.30 %** (0.33) | +5.69 % | −4.16 % (0.60) | inside (A) | inside; hit |
| Llama-2 | `-ub 1` | 7.4075 | 7.4974 | **+1.21 %** (0.35) | – | – | inside (A) | inside; hit |
| DeepSeek | `-b 512` | 8.2231 | 8.3099 | **+1.06 %** (0.31) | +5.13 % | −3.88 % (0.43) | inside (A) | inside; hit |
| DeepSeek | `-ub 1` | 8.2192 | 8.3252 | **+1.29 %** (0.24) | – | – | inside (A) | inside; hit |
| Yi | `-b 512` | 7.6966 | 7.7668 | +0.91 % (0.48) | −0.71 % | **+1.63 % (0.41)** | small change | inside; the change is 4 SE and has the wrong sign |
| Yi | `-ub 1` | 7.6817 | 7.8077 | +1.64 % (0.58) | – | – | small change | inside |

SE is the standard error over the 40 per-chunk NLL values, in PPL percent. The paired column takes the per-chunk
difference between two cells, so chunk difficulty cancels.

## 2. Test 3: the missing grid cells

Each prediction is Bonnie's straight line between the neighbouring `-b 512` cells (bonnie-reprice.md §1, her method). The
value at the level is my arithmetic (*estimated*).

| model | level | setting | PPL | Δ vs stock (SE) | straight-line prediction | verdict |
|---|---|---|---|---|---|---|
| Llama-2 | 25 % | `-b 512` | 7.7660 | **+4.84 %** (0.54) | +4.95 % (crossing near 22 %) | **outside** the pass line; budget +5 % not exceeded |
| Llama-2 | 25 % | `-ub 1` | 7.7377 | +4.46 % (0.49) | – | inside, 0.04 point under the line |
| DeepSeek | 25 % | `-b 512` | 8.5353 | +3.80 % (0.36) | +4.20 % (crossing near 27 %) | inside |
| DeepSeek | 25 % | `-ub 1` | 8.5297 | +3.78 % (0.40) | – | inside |
| Yi | 42 % | `-b 512` | 7.8992 | +2.63 % (0.72) | +2.97 % | inside |
| Yi | 42 % | `-ub 1` | 7.9851 | +3.95 % (0.69) | – | inside |

**Frontiers at uniform sparsity, both settings** (40 chunks, *host*):
- **Llama-2 stays at 20 %.** 25 % fails at `-b 512` by 0.34 point; it is on the line, not clear of it.
- **DeepSeek moves to 25 %.** 30 % fails.
- **Yi moves to 42 %.** 45 % fails at `-b 512` with +5.11 %.

The straight line was 0.1 to 0.4 point pessimistic on all three.

## 3. Reading

1. **Hypothesis A holds on both MHA models, and the share-matched cell shows it is the location, not the byte count.**
   At the same nominal skipped share (25.0 %), the two layouts compare as follows (*host*):

   | model | k, v dense 30 %, `-b 512` / `-ub 1` | uniform 25 %, `-b 512` / `-ub 1` |
   |---|---|---|
   | Llama-2 | +1.30 / +1.21 % | +4.84 / +4.46 % |
   | DeepSeek | +1.06 / +1.29 % | +3.80 / +3.78 % |

   Leaving k and v dense removes about three quarters of the loss of plain 30 % (Llama-2 −4.16 %, SE 0.60; DeepSeek
   −3.88 %, SE 0.43). q is still sparsified in these cells, so the intolerant part is the k and v inputs, not the
   attention inputs as a group. That counts against hypothesis C in its "attention inputs" form. The cells do not
   separate k from v.
2. **Ebony's pass condition is met as written** ("lifts the full-attention frontier to ≥ 0.30"): 30 % with k and v dense
   passes at both settings on both MHA models, with 3.2 to 3.4 points to spare. Her stronger statement ("should
   approach 0.35") is **not tested**: no k, v dense cell above 30 % was run. If the lead wants that frontier, it costs
   2 to 3 levels per model at both settings: 35 / 40 / 45 % on Llama-2 and DeepSeek, about 15 min per level and model
   serially.
3. **Yi: a small change, as A predicts for GQA, but in the unexpected direction.** k and v dense is +1.63 % (SE 0.41)
   worse than plain 30 %, while k and v are only 2.4 % of Yi's weights. The plain cell is the odd one: it sits under
   stock (−0.71 %, SE 0.46), and 35 % is +0.73 %. One reading is that sparsifying Yi's small k and v inputs helps at
   this level. I have no mechanism and did not test one. This does not change Yi's frontier (42 % uniform, inside).
4. **Price consequence** (*estimated*, for Bonnie): with k, v dense the MHA models skip 25.0 % of the weights at +1.1
   to +1.3 %. Uniform sparsity reaches 20 % (Llama-2) or 25 % (DeepSeek) at +3.3 to +4.3 %. Higher k, v dense levels
   are unmeasured.
   - The skipped share of k, v dense at level s is 0.834 s on these two models. A build skips columns only in q, o,
     gate/up and down; k and v run the stock kernel.
   - DeepSeek's GGUF has `add_bos = false` (converter rule, tom-screens.md §5). This caveat carries over unchanged.

## 4. Question 2, no run: which layout does K1 at TR 128 read, and must the row layout also be resident?

Source: `patches/llama.cpp-c3/0002-d-rows.patch`, functions `k1` and `d2_identity`, constants `KT` and `KB`.

- **K1 reads its own column-tiled layout.** This is neither the file's row-major Q4_0 blocks nor ggml's 4x4 repack.
  - Rows are cut into tiles of 128 (`KT`). For each (tile, 32-input block) pair there is one chunk of `KB` = 128 × 18 =
    2,304 B: first the 128 fp16 scales of that block (256 B), then 32 columns of 64 B. Byte k of column j holds row
    (k/16)·32 + k%16 in its low nibble and that row + 16 in its high nibble, as the file's unsigned nibble. Chunks are
    stored tile-major, then block.
  - The layout holds the same bytes as the row form: 18 B per 32 weights, every nibble and every scale once. It is a
    lossless permutation of the file's data. `d2_identity` builds it from row blocks and checks K1 bit for bit against
    the stock 4x4 gemv with `memcmp`.
- **The row layout does not have to be resident for K1.** The kernel reads only its own chunks.
  - The K1 layout can be the single resident copy of a matrix, in the place of the 4x4 repack copy that stock holds
    today. Both are anonymous copies built at load from the mmap'd file, of the same size. The mmap'd file pages are
    clean page cache and can be reclaimed, as at stock.
  - **Prefill does not force a second copy either.** memo-05 T4 (tom-host.md §3, *host*) showed that the 4x4 gemm
    equals the gemv row for row, bit for bit, and that K1 equals the gemv. Prefill can therefore run through K1 one
    token at a time with no declared change.
  - The cost of that is speed, not residency. Per-token prefill reads the matrix once per token unless a batched
    kernel over the K1 layout exists, and none is written. Under column sparsity, each prefill token has its own
    kept set in any case.
  - Keeping the stock gemm for prefill on the same tensor would need the 4x4 copy as well. That is double residency:
    about 3.5 GB × 2 on Yi's 3.48 GB file (*estimated*). It does not fit 8 GB beside the OS. A build should not do it.
- **Per tensor, one layout.** Two cases:
  - Under the k, v dense rule of §1, `attn_k` and `attn_v` keep the stock 4x4 layout and kernel. Every other matrix
    can carry the K1 layout. They are different tensors, so no tensor needs two layouts.
  - A row lever on the same matrix (Y1b's row-list skip, or T1 / C5 skipping gate/up rows) reads the row layout. Such a
    matrix needs that layout, not K1's. Combining a row-skip and a column-skip on the **same** tensor would need both
    copies, or a K1 kernel that also skips 128-row tiles; that is not written.
- **Shape rule:** K1 needs the row count to be a multiple of 128. Every mat-mul of Llama-2, DeepSeek and Yi meets it:
  4096, 11008 (= 86 × 128), and 512 (Yi's k and v).
- **What the patch does not decide:** 0002 is a probe on synthetic 512 MB arenas. It loads no model and is not wired
  into ggml, so residency in a real build is a design choice. The design above needs no second copy.

**Answer to Ebony §8 item 6:** her assumption holds. K1 reads a column-tiled copy, and the row layout need not be
held. A K1 build has the stock resident size, not stock plus a column copy, provided prefill runs through K1 (per token)
and no tensor is both row-skipped and column-skipped. So the Yi corner does not close on residency. Her factor
"P(resident at 3.3 GB plus the column copy) 0.3" can be re-read as "resident at the stock size".

## 5. Follow-up (2026-11-22): k, v dense frontier on the MHA models, and k against v

The lead asked for k, v dense at 35 / 40 / 45 % on Llama-2 and DeepSeek at both settings, 40 chunks, stopping a model at
its first failing level, then k-only and v-only dense at 30 % on Llama-2 at `-b 512`. Runs were serial and used the thr2
tables. A model stopped at the first level over +4.5 % at either setting; when `-b 512` already failed, its `-ub 1` cell
was not run.

**Validity after the container restart.** The queue finished before the restart (`split2 done` 08:24:15 in
`lab/c3/logs/split.log`), so no cell was lost.
- All 13 cells are in `lab/c3/logs/u1-<model>-<b512|ub1>-40.log`, each with 40 per-chunk values.
- The `.err` files hold no error or traceback.
- The printed consumer plans are as designed: `restore:qkv` for k, v dense and for k-only, `toggle3:qkv` for v-only
  (see the tool change below).

### 5.1 k, v dense frontier (*host*, Δ vs stock at the same setting, SE over 40 chunks)

| model | level | nominal skipped share (*estimated*) | `-b 512` | `-ub 1` | verdict |
|---|---|---|---|---|---|
| Llama-2 | 30 % | 25.0 % | +1.30 % (0.33) | +1.21 % (0.35) | inside (§1) |
| Llama-2 | 35 % | 29.2 % | +2.57 % (0.42) | +2.50 % (0.43) | inside |
| Llama-2 | 40 % | 33.4 % | +3.97 % (0.57) | +3.95 % (0.53) | inside |
| Llama-2 | 45 % | 37.5 % | **+6.74 %** (0.69) | not run | outside: stop |
| DeepSeek | 30 % | 25.0 % | +1.06 % (0.31) | +1.29 % (0.24) | inside (§1) |
| DeepSeek | 35 % | 29.2 % | +2.18 % (0.35) | +1.79 % (0.40) | inside |
| DeepSeek | 40 % | 33.4 % | +2.95 % (0.56) | +3.54 % (0.54) | inside |
| DeepSeek | 45 % | 37.5 % | **+5.21 %** (0.63) | not run | outside: stop |

PPL values: Llama-2 7.5975 / 7.5928 (35 %), 7.7010 / 7.7000 (40 %), 7.9061 (45 %, `-b 512`). DeepSeek 8.4024 / 8.3665
(35 %), 8.4659 / 8.5105 (40 %), 8.6512 (45 %, `-b 512`).

- **The k, v dense frontier is 40 % on both MHA models at both settings.** It skips a nominal 33.4 % of the block's
  mat-mul weights (*estimated*, 0.834 × 0.40). The uniform frontier skips 20 % on Llama-2 and 25 % on DeepSeek (§2).
- **Ebony's "approach 0.35" is met and passed.** The frontier lies between 0.40 and 0.45. Llama-2's 40 % cell is 0.5
  point under the line, so its margin is thin; DeepSeek's margin is 1.0 to 1.6 points.

### 5.2 k against v (Llama-2, 30 %, `-b 512`)

| cell | PPL | Δ vs stock (SE) | paired vs plain 30 % (SE) |
|---|---|---|---|
| plain 30 % (phase 2) | 7.8284 | +5.69 % (0.67) | – |
| k dense only | 7.6412 | +3.16 % (0.49) | −2.39 % (0.48) |
| v dense only | 7.6929 | +3.86 % (0.38) | −1.73 % (0.61) |
| k and v dense (§1) | 7.5031 | +1.30 % (0.33) | −4.16 % (0.60) |

- **Both inputs carry part of the loss, and the parts add.**
  - Keeping k dense removes 2.4 points and keeping v dense 1.7 points. The sum, 4.1 points, matches the 4.2 points of
    keeping both.
  - k-only against v-only is −0.67 % (SE 0.45, paired): k is the larger part, but at 1.5 SE this is not resolved.
  - Adding v to k-only still removes 1.81 % (SE 0.39). Neither one alone is enough.
- **For a build:** leaving only one of the two dense does not keep 30 % inside with margin. Both single cells are inside
  (+3.2 and +3.9 %), but with less room than k and v together.

### 5.3 Tool change and identity re-check

`tools/c3_colskip.py` (uncommitted): the consumer plan now handles an unedited reader that runs **between** two target
readers. That is the v-only case: Llama-2's read order is Qcur, Vcur, Kcur, and with only v dense, K is still a target.
- The plan restores the shared input before each unedited reader and puts the edited copy back before the next target
  reader. After the last target it restores for good.
- It is refused only if an unedited reader runs before the first target. The plan prints this as `toggle<n>:<group>`.
- Toggle nodes are keyed by name and op; the tool asserts that this key is unique in the graph.
- The single-restore path (`restore:`) is unchanged in effect.

Checks (*host*, Llama-2 Q4_0):
- **Known answer for v-only** (1 chunk, layer 0): Qcur and Kcur equal the plain 30 % cell bit for bit; Vcur equals the
  stock cell bit for bit; plain 30 % Vcur differs from stock.
- **Identity, re-run after the edit with the current tool** (2 chunks, ub 512):
  - `C3_DENSE_W=attn_q,attn_k,attn_v` with groups `all` gives per-chunk NLL 1.7911861779847056, 1.9606204103102398.
    That is bit-identical to groups `o,gu,down`, and identical to §0's pre-edit values.
  - k, v dense 30 % gives 1.7961652146235187, 1.930309286603152, identical to §0's pre-edit values. So the 30 % cells
    of §1, run before the edit, and the 35 to 45 % cells, run after it, come from the same computation.
  - Table `-`, sparsity 0 gives running PPL [1]5.8104, [2]6.3386, the same as `llama-perplexity -c 512 -b 512
    --chunks 2` on this build.
