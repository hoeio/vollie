# Two-stage lm_head: evaluation fixed, table rerun (Bonnie, r27, 2026-11-05, host only)

Question (Clyde): the 2026-11-04 screening of a two-stage lm_head (cheap screen over all 100,352 rows → exact
Q6_K logits for the top-M candidates) printed reference PPLs of 18.69 and 13.16 against llama-perplexity's
9.6051 on the same text, and two-stage PPLs in the millions. Find the root causes, make the reference reproduce
llama-perplexity to < 1 %, then rerun the table. Everything here is **host** (M4 Pro) unless marked *estimated*
(device ms from R20/R25/R27 bandwidth). Raw logs `lab/logs/r27/head2stage/`; scripts `lab/tools/head2stage*.py`.

## a) Root causes

**Logits route (`head2stage.py`, 18.69).** `--save-all-logits` does not store logits: `log_softmax()` in
`tools/perplexity/perplexity.cpp:79-104` stores a uint16 *log-softmax floored at max−16 nats*, so 92.2 % of the
entries are 0 ("≤ floor") and only 7.8 % carry a value. The script inverted this to recover h by least squares on
the uncapped rows (1,537 unknowns from ≤ 2,560 top-ranked rows), which is ill-posed: the fit explains the top rows
(held-out residual 0.008 nats) and says nothing about the directions the other 92 % of rows probe, so the
"exact" logits of those rows came out wrong (reference 18.69; 688 of 2,040 positions discarded as unfittable).
The dump needs no inversion for the reference: scoring the true token's stored value directly gives, per chunk,
`[1]5.5137 [2]7.3175 [3]7.0292 [4]7.3527 … [8]9.1743` vs llama-perplexity `5.5137 7.3175 7.0292 7.3567 … 9.2129`
(−0.4 % at 8 chunks; the whole gap is 3 positions where the true token sits below the −16 floor). The route can
give a reference but never h, so it cannot replay a screen. Retired. (`check_old_h.txt`)

**Hidden-state route (`head2stage_h.py`, 13.16).** h is right: `/embedding --pooling none --embd-normalize -1`
returns `result_norm` (after `output_norm`, before `model.output`), the head is tied `token_embd.weight` Q6_K
(126,443,520 B), `logit_scale` 6.0 is in the file and was applied, no BOS (`add_bos_token=false`), and the
tokens of all 8 chunks are identical to the dump's. Two evaluation mistakes: (1) **it scored all 511 positions of
every chunk**; llama-perplexity scores only positions 256..510 (`first = n_ctx/2`). Second half only → 9.1328.
(2) **it used fp32 activations**; ggml computes Q6_K × Q8_K, i.e. h is quantised per 256-block to int8 before the
dot. Simulating `quantize_row_q8_K` on h → **9.2129 = llama-perplexity's [8] to four decimals**, per-position NLL
difference median 0.0000, p90 0.0001 (`q8check_old_h.txt`; the only remaining differences are the 3 floored
positions). The IncompleteRead in `fetch_h.txt` was a later, failed re-fetch; the `h.npy` actually used held 8
chunks. The two-stage PPLs "in the millions" were not a fallback bug — the script already gave non-candidates
the sketch logit — they are the honest result of a **rank-r SVD screen**: `token_embd` has a flat spectrum (rank
128/256/512 capture 21 / 32 / 51 % of the energy), true-token recall is 0.29-0.75, and a missed true token gets a
sketch logit 15-25 nats below its exact one. SVD screens are dead for a tied, un-normalised head.

**Fixed reference (60 chunks, 15,300 positions, Q8_K activations, second half of each 512 chunk): 9.8702 =
llama-perplexity's [60] 9.8702 (`lab/logs/r26/ppl60-base-wiki.log`), with [1] 5.5137, [2] 7.3175, [4] 7.3567,
[8] 9.2129, [16] 10.3234 (log 10.3235), [32] 8.8666 (log 8.8667). Greedy argmax == true token on 52.0 %.**
Pipeline: `head2stage_fetch.py` (llama-server `lab/logs/r26/bin-r5`, `--embeddings --pooling none -c 512 -b 512
-ub 512`, whole-file tokenisation as llama-perplexity) → `head2stage.py`.

## b) Two-stage table (60 chunks; `head2stage60.txt`)

Exact head = Q6_K rows (1,260 B each). Screen scores all rows; the top-M get exact logits; the rest keep the
screen logit (what the runtime would hand to the sampler and to llama-perplexity). Bytes = screen + M × 1,260 B.
Device ms *estimated* at the dense-`MUL_MAT` rate: 560 MB in 23.1 (R25) … 25.6 ms (R20) = 0.041-0.046 ms/MB; the
Q6_K head alone is 126.4 MB ≈ **5.2-5.8 ms**. R27's clock test showed the dense products bandwidth-bound
(×1.08 at ×0.74 clock), so bytes are the right currency here; compute of the screen (same 154 M MACs at
Q2_K/Q3_K/Q4_0 rate) and the top-M selection are not in the byte number — see the ms discussion below.

| screen (MB) | screen alone PPL | M | argmax recall | true-token recall | PPL (2-stage) | ΔPPL | MB/token | ms saved (est.) |
|---|---|---|---|---|---|---|---|---|
| SVD r=128 fp16 (25.7) | — | 1,024-32,768 | 0.31-0.63 | 0.29-0.64 | 1.9e4-6.9e5 | ×10³-10⁵ | 27-67 | dead |
| SVD r=256 fp16 (51.4) | — | 1,024-32,768 | 0.40-0.70 | 0.38-0.71 | 5.2e3-1.9e5 | dead | 53-93 | dead |
| SVD r=512 fp16 (102.8) | — | 1,024-32,768 | 0.49-0.75 | 0.47-0.75 | 2.0e3-5.0e4 | dead | 104-144 | dead |
| **Q2_K copy (50.6)** | 15.0352 (+52.3 %) | 1,024 | 0.9999 | 0.9818 | 10.0350 | +1.670 % | 51.9 | 3.1-3.4 |
| | | 2,048 | 1.0000 | 0.9890 | 9.9685 | +0.996 % | 53.2 | 3.0-3.3 |
| | | 4,096 | 1.0000 | 0.9942 | 9.9260 | +0.566 % | 55.7 | 2.9-3.2 |
| | | **8,192** | **1.0000** | **0.9977** | **9.8973** | **+0.275 %** | **60.9** | **2.7-3.0** |
| | | 16,384 | 1.0000 | 0.9993 | 9.8765 | +0.065 % | 71.2 | 2.3-2.5 |
| | | 32,768 | 1.0000 | 0.9998 | 9.8722 | +0.021 % | 91.9 | 1.4-1.6 |
| **Q3_K copy (66.2)** | 11.0980 (+12.4 %) | 1,024 | 1.0000 | 0.9867 | 9.9024 | +0.326 % | 67.5 | 2.4-2.7 |
| | | 2,048 | 1.0000 | 0.9920 | 9.8908 | +0.210 % | 68.8 | 2.4-2.6 |
| | | 4,096 | 1.0000 | 0.9956 | 9.8839 | +0.139 % | 71.4 | 2.3-2.5 |
| | | **8,192** | **1.0000** | **0.9983** | **9.8757** | **+0.056 %** | **76.6** | **2.1-2.3** |
| | | 16,384 | 1.0000 | 0.9997 | 9.8704 | +0.003 % | 86.9 | 1.6-1.8 |
| Q4_0 copy (86.7) | 10.3091 (+4.45 %) | 2,048 | 1.0000 | 0.9925 | 9.8707 | +0.006 % | 89.3 | 1.5-1.7 |
| | | 8,192 | 1.0000 | 0.9988 | 9.8721 | +0.020 % | 97.0 | 1.2-1.3 |
| | | 16,384 | 1.0000 | 0.9997 | 9.8694 | −0.007 % | 107.3 | 0.8-0.9 |

Reading: a low-bit *copy of the same rows* is a near-perfect screen (argmax recall 1.0000 on 15,300 positions
for every M ≥ 2,048 — the greedy output would be identical in every scored position), while a low-rank sketch is
useless. The PPL cost comes only from the 0.1-1 % of positions where the true token is outside the set and gets
the screen's logit; it falls as the screen gets finer (Q2_K → Q4_0) and as M grows. The Q4_0 head *alone* fails
the gate (R-log D2, +10.3 % full set); as a screen in front of 2,048 exact rows it is +0.006 %.

**Full wikitext-2 (565 chunks, 144,075 positions — the gate's own set, not an extrapolation; `full/head2stage565.txt`,
`full/head2stage565_b.txt`).** Reference = **9.6051 = the stock gate number** ([1] 5.5137 … [60] 9.8702 … [565] 9.6051);
argmax == true token 52.2 %. Q2_K head alone 14.5991 (+52.0 %), Q3_K alone 10.8331 (+12.8 %), Q4_0 alone 10.0137 (+4.25 %).

| screen | M | argmax recall | true-token recall | PPL (full set) | ΔPPL vs 9.6051 | MB/token | ms saved, byte term (est.) |
|---|---|---|---|---|---|---|---|
| Q2_K | 4,096 | 1.0000 | 0.9949 | 9.6499 | +0.467 % | 55.7 | 2.9-3.2 |
| **Q2_K** | **8,192** | **1.0000** | **0.9980** | **9.6237** | **+0.193 %** | **60.9** | **2.7-3.0** |
| **Q2_K** | **16,384** | **1.0000** | **0.9994** | **9.6108** | **+0.059 %** | **71.2** | **2.3-2.5** |
| Q3_K | 4,096 | 1.0000 | 0.9962 | 9.6137 | +0.090 % | 71.4 | 2.3-2.5 |
| **Q3_K** | **8,192** | **1.0000** | **0.9987** | **9.6078** | **+0.028 %** | **76.6** | **2.1-2.3** |
| Q4_0 | 2,048 | 1.0000 | 0.9932 | 9.6062 | +0.011 % | 89.3 | 1.5-1.7 |
The 60-chunk screen predicted +0.275 / +0.065 % for Q2_K 8,192 / 16,384 and +0.056 % for Q3_K 8,192; the full set says +0.193 / +0.059 / +0.028 %. The
argmax never left the candidate set in 144,075 positions for any M ≥ 4,096: a greedy decode through the two-stage
head prints the same tokens as the exact head on this text.

**ms on the phone (estimated, not measured).** Byte term for Q2_K/M 8,192: 126.4 → 60.9 MB ≈ −2.7 to −3.0 ms.
Against it: (i) the screen is a Q2_K × Q8_K `MUL_MAT` of the same 154 M MACs — on a bandwidth-bound op this is
hidden, but Q2_K's per-MAC kernel is not faster than Q6_K's, so the compute floor of the head (R20's additive fit:
compute ≈ 24 % of the dense op at one token → ≈ 1.2-1.5 ms for the head's share) stays; (ii) the M exact rows are
a gather of 8,192 × 1,260 B random reads from a page-cache-hot tensor, ≈ 10.3 MB, call it 0.4-0.7 ms; (iii) a
top-M selection over 100,352 floats, ≈ 0.3-0.5 ms single-threaded (nth_element), parallelisable. Honest band:
**−2.0 to −2.5 ms/token for Q2_K/8,192, −1.5 to −2.0 for Q3_K/8,192, ≤ −1.5 for any Q4_0 setting.** Memory: a
Q2_K copy of `token_embd` is 50.6 MB of anon (or an app-private cache file) built at load; the served file is
untouched; the one-time Q2_K quantisation of 154 M weights is seconds on three A78s (*estimated*, not timed).

## c) Verdict

**Yes alone, no on top of skip 118.** Q2_K screen + 8,192 exact rows is **+0.19 % on the full set (9.6237 ≤ 9.797)**
with an *estimated* **−2.0 to −2.5 ms/token** (byte term −2.7 to −3.0 less the screen's compute floor and the top-M
selection); Q2_K + 16,384 is +0.06 % for −1.8 to −2.2. But the +2 % speed-rule budget is already spent by the
served skip rule: 118 = 9.7952 on the full set, 0.02 % under 9.797, so no (screen, M) stacks on 118 inside the
budget (Q3_K/8,192 at +0.028 % → 9.7979, still over by 0.001); on **skip 116 (9.7787)** the pair 116 + Q2_K/16,384 ≈ 9.7845 or 116 + Q3_K/8,192 ≈ 9.7814 (*estimated*, multiplicative) fits, trading
≈ +0.5-0.8 ms of experts (116 vs 118, R25) for −1.8 to −2.2 of head: **net ≈ −1.0 to −1.7 ms/token**, i.e. below the
≥ 2 ms bar unless the owner re-prices the budget. It is the first lever found that cuts dense-`MUL_MAT` *bytes*
inside the rules and the frozen file (−55 to −65 MB/token of the 560), it is model-agnostic (any tied or untied
head; the screen is a runtime re-quantisation), and its output is argmax-identical on 144 k positions — so I
recommend it goes to the owner as a budget question, not into the kit as a default. Cheapest falsifier on the
device: Tom's `GGML_HEAD2` behind an env var, profiled `MUL_MAT` sum before/after in one ABBA run (pre-registered
−2.0 to −2.5 ms at 8,192), full-set `tools/ppl.sh` at 9.6237 ± 0.01.

## Spec for Tom (if Clyde keeps it)

`GGML_HEAD2=<M>` in the CPU module: on the `MUL_MAT` whose `src0` is the model's output tensor and `ne11 == 1`,
(1) at first use quantise `src0` to Q2_K (`ggml_quantize_chunk`, type selectable `GGML_HEAD2_TYPE=Q2_K|Q3_K`) into
an anon buffer; (2) run the screen `vec_dot` over all rows into the destination; (3) `nth_element` the top-M row
ids; (4) recompute those M rows with the Q6_K `vec_dot` in place. Everything else, including `ggml_scale(1/6)`,
unchanged. Check: `tools/ppl.sh` with `GGML_HEAD2=8192` vs stock on the full set (must land at the row's ΔPPL ±
0.1 %), greedy 32-token official bench identical to stock (argmax recall 1.0000), and the profiler's `MUL_MAT` sum
on the phone before/after (pre-registered −2.0 to −2.5 ms).
