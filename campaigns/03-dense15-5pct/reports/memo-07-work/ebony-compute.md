# memo-07 work file, Ebony: the family split, and levers that cut compute per token

Ebony (`claude-fable-5-2`). **Nothing here was run on the phone; I ran nothing on the host either.** Labels:
*host* = Tom's logs under `lab/c3/logs/` (read-only) or earlier memos; *published* = a source's claim, with how it
was read; *estimated* = my arithmetic. Printed tok/s = 0.95 x steady (memo-06 convention); tables say which.

## 1. Predictions, written first (2026-11-22 04:55 KST)

**What I had seen when writing this section:** the lead's message (H3 frontiers: Yi-1.5-6B 0.40 / 0.40,
DeepSeek-7B 0.20 / 0.20, Llama-2-7B 0.20 / 0.20, Granite-7B 0.15 / pending, Mistral 0.40, Qwen2.5 0.35; T1 on
Mistral at `-b 512`: about 30 % of groups in file order, 33 to 35 % clustered) and the tail of
`h3-frontiers.log`, which I listed before writing and which already held **ChatGLM3: 0.35 at `-b 512`, 0.20 at
`-ub 1`** and Granite 0.15 at `-ub 1`. I had not opened any `t1-*` log, any per-cell H3 log, or Bonnie's
re-price.

**ChatGLM3 H3: no blind prediction is possible; I saw the result.** For the record only, and not to be scored:
from its 32/2 grouped attention I would have said 0.30 to 0.35 at both settings. The 0.15-point gap between
`-b 512` and `-ub 1` is not what I would have predicted for any model (memo-02: the two settings agree within
0.5 point of PPL at equal thresholds), so I read it as a tool or graph question before a model property (§2.4).

**T1 group-of-four frontier, largest share of groups inside stock x 1.05, `-b 512`, clustered order** (file order
about 3 to 5 points lower on every model):

| model | prediction (band) | P(≥ 30 %) | reasoning in one line |
|---|---|---|---|
| Qwen2.5-7B | 37 % (33 to 42) | 0.85 | E1's element frontier on `down` input was higher than Mistral's (*host*, memo-05) |
| Yi-1.5-6B | 32 % (27 to 37) | 0.6 | same H3 frontier as Mistral; narrower FFN (11,008), so each group is a larger share of the layer |
| ChatGLM3-6B | 27 % (20 to 33) at `-b 512`; `-ub 1` within 3 points of it if the H3 gap is a tool effect, 10 points lower if it is real | 0.35 | between the two classes |
| Llama-2-7B | 22 % (15 to 28) | 0.1 | see below |
| DeepSeek-LLM-7B | 22 % (15 to 28) | 0.1 | same shape and class as Llama-2 |
| Granite-7B | 18 % (12 to 25) | 0.05 | lowest H3 frontier |

**The structural prediction behind the table.** T1 zeroes FFN neurons only (the `down` input); H3 zeroes
residual-stream inputs as well. I expect the family split to sit mostly in the residual-stream inputs, so **T1's
split is narrower than H3's**: the Llama-2 class reaches about 0.6 to 0.7 of Mistral's group share, not 0.4 to 0.5.
Falsified if Llama-2 and DeepSeek are both under 15 %.

**Discriminating cells I predict (for §2; each is one or two cells per model):**
1. Per-group H3 cells on the Llama-2 class: the `down`-input cell holds 50 % inside the budget (P 0.6), and the
   damage is in `gate` + `up` and attention inputs.
2. First-token-dense (N = 1) on the Llama-2 class moves the uniform frontier up by at least 10 points (P 0.4). On
   Mistral and Qwen2.5 it did nothing (memo-02).
3. Keeping layers 0 to 3 dense moves it by at least 10 points (P 0.45).

**Scored the same day, after the section above was written** (Tom's `memo-06-work/tom-screens.md` §6, *host*):
Qwen2.5 T1 clustered about 36 % (target 0.40 at +2.74 %, 0.45 at +4.59 %), file order about 37 %. Prediction
37 % (33 to 42): **hit**. My note "file order 3 to 5 points lower" is a **miss** on Qwen2.5 (file order is equal or
higher by realised share). T1 has not been run on Yi, ChatGLM3, DeepSeek, Llama-2 or Granite; those rows stand
unscored.

## 2. The family split in H3 (*host* frontiers: Yi 0.40, Mistral 0.40, Qwen2.5 0.35; DeepSeek 0.20, Llama-2 0.20, Granite 0.15; ChatGLM3 0.35 / 0.20)

### 2.1 What the activations rule out (*host*, my reading of Tom's threshold tables `lab/c3/thr2-<m>-q4_0.npz`)

The tables hold quantiles of |x| per layer and input group, from 16 chunks of wikitext-2 train. Median over
layers of two shape ratios:

| model | qkv input: q20 / q50, q40 / q50 | gate+up input | `down` input | o input |
|---|---|---|---|---|
| Yi-1.5-6B | 0.371, 0.774 | 0.372, 0.775 | 0.266, 0.703 | 0.311, 0.719 |
| Mistral-7B | 0.369, 0.772 | 0.371, 0.775 | 0.266, 0.707 | 0.331, 0.732 |
| Qwen2.5-7B | 0.365, 0.769 | 0.370, 0.774 | 0.287, 0.720 | 0.315, 0.722 |
| ChatGLM3-6B | 0.371, 0.774 | 0.371, 0.775 | 0.262, 0.701 | 0.320, 0.724 |
| DeepSeek-7B | 0.369, 0.774 | 0.370, 0.775 | 0.258, 0.698 | 0.310, 0.715 |
| Llama-2-7B | 0.369, 0.774 | 0.370, 0.775 | 0.254, 0.695 | 0.337, 0.735 |
| Granite-7B | 0.366, 0.773 | 0.369, 0.775 | 0.254, 0.694 | 0.331, 0.735 |
| half-normal, for reference | 0.376, 0.777 | | | |

- **The pooled distribution of the residual-stream inputs is the same in all seven models, and it is Gaussian up
  to the median.** At equal sparsity the same share of input mass is removed in every model. So the split is
  **not** in how sparse the inputs are. It is in how much each model's output moves for the same relative input
  error.
- The `down` input is a little more peaked in the three intolerant models (0.254 to 0.258 against 0.266 to
  0.287). That would make them slightly *more* tolerant on that group, not less.
- Limits of this reading: quantiles are pooled over channels and positions, so per-channel scale structure and
  the first positions are invisible; the tables stop at the 0.8 quantile (no tail).

### 2.2 What is published

- **Llama-2's two published values are the two sides of this split.** TEAL's own table sparsifies only the second
  half of each context and has Llama-2-7B at +0.4 % (25 %) and +3.0 % (40 %). LaRoSA's re-run of TEAL sparsifies
  everything and has +15.4 % at 25 % (memo-01 work §2.2). With the same change of protocol Mistral moves from
  +1.8 to +4.1 % and Llama-3 from +1.2 to +3.9 %. So on Llama-2 the damage comes from sparsifying the **context the
  scored tokens attend to**, far more than on the grouped-attention models. TEAL says so in words ("sparsifying
  the initial tokens causes much more degradation"). Tom's 0.20 is the LaRoSA side. (*published*; both tables as
  quoted in memo-01, not re-read today.)
- Prox's order (Qwen3 ≥ Mistral > Llama-3) has no model of the intolerant class, so it does not speak to it.
- No paper I have read explains a split by attention type; I did not search for one today beyond what memo-06
  cites.

### 2.3 Hypotheses, with what each predicts (none is shown)

| # | hypothesis | for | against | deciding cell |
|---|---|---|---|---|
| A | **Context (KV) error.** In full multi-head attention the k and v matrices are full size (a sixth of the body each pair) and every one of 32 independent key/value heads is written from the sparsified input; the error of all earlier positions is read back by every later token | the split is exactly MHA (DeepSeek, Llama-2, Granite: 32/32) against GQA (Yi 32/4, Mistral 32/8, Qwen2.5 28/4), 6 of 6; Yi and Llama-2 share shape and differ in this; TEAL-against-LaRoSA on Llama-2 | no mechanism is published; n = 6 and era is confounded (all three MHA models follow the 2023 Llama-2 recipe); ChatGLM3 (32/2) is 0.35 at `-b 512` but 0.20 at `-ub 1` | uniform 30 % with the k and v inputs left dense, on Llama-2 and DeepSeek |
| B | **First positions.** The attention-sink token's large activations are written by early layers and set a bias for all later tokens (Sun et al., "massive activations", *recalled*, not re-read) | TEAL's remark; Llama-2 is the model that paper studies | DeepSeek was screened without BOS and is as intolerant; dense-first did nothing on Mistral and Qwen2.5 (memo-02) | first N = 1, 8 tokens dense; layers 0 to 3 dense |
| C | **One input group carries it** (for example the attention inputs, which are a third of these bodies) | Tom's remark that MHA bodies have more attention | memo-02 found attention inputs the tolerant group on Mistral and Qwen2.5 | the four per-group cells at 30 % on Llama-2 and on Yi (same shape) |
| D | Training recipe or data scale, nothing structural | cannot be excluded | Mistral is a 2023 model and is tolerant | none on the host; it is what remains if A to C fail |

My weights before the cells: A 0.40, C (some other single group) 0.20, B 0.15, D 0.25. If A holds, k and v dense
costs the MHA models a sixth to a third of the lever's bytes and the frontier of the rest should approach 0.35.

### 2.4 ChatGLM3

- Seen before predicting (§1). The split between `-b 512` (0.35) and `-ub 1` (0.20) is 0.7 point of PPL on a flat,
  non-monotone curve (30 %: +4.32 / +5.05 %; 35 %: +4.21 / +4.74 %; 40 %: +4.63 %), all within about 1 point of the
  line. On this model the frontier is decided by noise of the 40-chunk screen, not by the model: **I read both
  numbers as "about 0.20 with margin, 0.30 to 0.40 on the line"**, not as a batch effect.
- Its stock PPL of 33 (*host*) says the gate text is far from what the model expects without its prefix (Tom §5.3,
  result still a placeholder). A budget of +5 % of 33 is 1.7 PPL; whether that measures the model is the lead's
  question. I would not freeze ChatGLM3 before the prefix check has a number.
- `tom-screens.md` still has unfilled placeholders (CHATGLM_B512_20, GLM_PREFIX_RESULT, T1_QF, T1_QC).

*Note after Tom's final file (2026-11-22, `tom-screens.md` §5.3, *host*): the prefix check is in and makes stock PPL
worse (48.26 without, 53.36 with, 8 chunks), so the prefix is not the cause and the placeholders are filled. The
lead scores ChatGLM3's frontier as 0.20. The level of 33 is unexplained and the conversion is unchecked against
the publisher's code. My §1 statement on ChatGLM3 remains post-data and unscored.*

## 3. Levers that cut compute, priced (*estimated*)

Convention: printed = 0.95 x steady; c = 69.9 G `sdot`-weights/s on X1 + 2 A78 (Bonnie's value; R69 band 62 to
80); O = 8 ms. 12 printed needs 71.2 ms for body and head: **4.98 G weight-equivalents on three cores** and
**B ≤ 1.71 GB at r 24, 2.49 GB at r 35**. Both lines must hold.

### 3.1 The levers

| # | lever | class | what it does to W | what it does to B | status and price |
|---|---|---|---|---|---|
| C1 | **4th big core** (X1 + 3 A78) | bit-identical | c x 1.15 (1.00 to 1.25): my haircut for a fourth thread (R18 and R62 measured losses). The pre-registered hot sum is 50.3 against 40.4 GB/s, x 1.245, at the same 0.87 efficiency (memo-04 work); Bonnie uses that. *Correction after reading R69 §3: the 1.13 I first quoted here is the cold q4x4 ratio over 3 A78, a read-rate figure, not a compute figure* | r also moves: pre-registered cold q4x4 on X1 + 3 A78 is 27 (21 to 35) against 24 GB/s | R69 measures it. Closed for MoE (R18, sync cost per expert); a dense token has about 230 large mat-vecs, so the sync share is smaller. **The largest single compute lever** |
| C2 | A55s as weighted row chunks of the same mat-vecs | bit-identical | up to +21 % of c at perfect scaling (0.33 cycles per weight, R69 band); I price +5 % (0 to +12 %) after sync | none | R69's "3 A78 + 4 A55" cell. Lost on MoE (R18) |
| C3 | **GPU as a compute lane**: a fixed row share of `gate` + `up` (one dispatch) and of `down` (one dispatch) per layer | numerics (float order) | at R69's X2 centre (10 GB/s-equivalent, 3 to 22) the GPU's balanced share is about 20 % of the FFN: about 12 ms saved, minus 64 dispatches at about 0.1 ms: **x 1.07 (1.00 to 1.20)** on the token | none | R69 Rule D. Splitting every mat-vec (224 dispatches) loses to dispatch cost (*estimated*, dispatch time not measured) |
| C4 | Group-of-four neuron skip (memo-06 Y8) at the **measured** frontier, g = 0.33 (Mistral) to 0.36 (Qwen2.5), *host*, exact-product ranking | numerics | with a 1-bit proxy at 0.2 of `gate` + `up`: FFN x 0.90 to 0.94, token x 0.92 to 0.95. **This is a re-price of my own Y8 downwards**: memo-06 assumed 40 % and gave +1.7 to +2.9 tok/s; at the measured share the proxy eats most of it | FFN x 0.75 | +0.4 to +0.8 tok/s. Not worth a proxy file unless T6 puts the proxy under 0.1 |
| C5 | **Up rule**: compute `up` fully, rank neurons by \|u\| (times a calibrated per-neuron gate scale), skip the groups' `gate` rows (k = 1.0) and `down` columns. No proxy, no side copy, no extra resident bytes | numerics | FFN x (1/3 + (1 − g)/3 + (1 − g) k_c/3): 0.91 at g 0.33, k_c 1.6; token x 0.93 | FFN x 0.79 | **Untested.** SGA-Sparse's abstract says the up activation routes better than the gate in stock models (*published*, abstract only); E2 closed only the gate rule. +0.6 to +0.8 tok/s when compute-bound |
| C6 | **Units of eight input channels inside the 4x4 repack.** One 16-byte vector of a Q4_0x4 block holds 8 columns of 4 rows (memo-04 work §3.1). Skip the vector load, unpack and two `sdot`s when all 8 inputs are under threshold. Kept weights run the stock instructions | bit-identical at threshold 0 (integer sums; an in-block reorder is also exact); numerics above | applies to **every** body matrix, not only the FFN: x (1 − u) x about 1.05 | x (1 − 0.89 u) | **Untested**, kernel and frontier. Between element sparsity (0.15 to 0.40) and blocks of 32 (0.09 to 0.10, closed). My prediction before data: u = 0.18 (0.12 to 0.25) on Yi and Mistral, 0.15 on Qwen2.5, 0.08 (0.04 to 0.12) on the Llama-2 class; P(≥ 0.20 on two models) 0.25 |
| C7 | Head screen with input sparsity: the two-stage head's screen reads only the largest 30 % of the final hidden vector's channels (K1), then exact rows for the top 2,048 | numerics | head x about 0.5 | head x about 0.4 | untested; head is 2 to 8 % of W, so +0.1 to +0.35 tok/s; generic |
| – | K1 element columns at the H3 frontier | numerics | body x (1 − s) x 1.6: **0.96 at s 0.40, 1.28 at s 0.20** | body x (1 − 0.889 s): 0.64 / 0.82 | a byte lever. Compute-neutral on Yi and Mistral, **compute-negative on the Llama-2 class at any measured k** (it needs k < 1 ÷ (1 − s) = 1.25) |
| – | a sub-4-bit served file | – | x 1.5 or more per weight (*published*, *device*) | x 0.75 to 0.85 | spends the compute line (memo-06 §3.2) |

C4, C5, C6 and the K1 columns share the one +5 % budget; each frontier above was measured or predicted alone.

### 3.2 Price per body (printed tok/s; "r needed" is the read rate at which the byte line stops binding)

Shares used: head 4.5 / 4.5 / 6.5 / 2.0 / 1.9 / 7.7 % of W. Column s is Tom's H3 frontier (*host*); u is my
unscreened prediction for C6.

| model (W G, B GB, s, u) | dense, 3 cores | C1 | C1 + K1 columns (k 1.6) | C1 + C6 | C1 + C6 + C3 + C7 |
|---|---|---|---|---|---|
| Yi-1.5-6B (5.80, 3.33, 0.40, 0.18) | 10.4, r 40 | 11.9, r 46 | **12.3, r 32** | 13.5, r 45 | 14.6, r 49 |
| ChatGLM3-6B (5.98, 3.43, 0.20, 0.12) | 10.2, r 40 | 11.5, r 46 | 9.3, r 30 | 12.3, r 45 | 13.4, r 48 |
| DeepSeek-7B (6.49, 3.76, 0.20, 0.08) | 9.4, r 40 | 10.7, r 47 | 8.6, r 31 | 11.0, r 45 | 12.1, r 48 |
| Llama-2-7B (6.61, 3.75, 0.20, 0.08) | 9.3, r 40 | 10.5, r 46 | 8.4, r 30 | 10.9, r 44 | 11.7, r 47 |
| Mistral-7B (7.11, 4.04, 0.40, 0.18) | 8.7, r 40 | 9.8, r 46 | 10.2, r 31 | 11.3, r 45 | 12.1, r 48 |
| Qwen2.5-7B (7.07, 4.12, 0.35, 0.15) | 8.7, r 41 | 9.9, r 47 | 9.6, r 33 | 10.9, r 46 | 12.0, r 49 |

The same columns at the read rates R69 is expected to find (printed tok/s at r 24 / r 35):

| model | dense or C1 | C1 + K1 columns | C1 + C6 | C1 + C6 + C3 + C7 |
|---|---|---|---|---|
| Yi-1.5-6B | 6.5 / 9.2 | 9.4 / **12.3** | 7.5 / 10.7 | 7.7 / 10.9 |
| ChatGLM3-6B | 6.3 / 9.0 | 7.5 / 9.3 | 7.0 / 9.9 | 7.1 / 10.1 |
| DeepSeek-7B | 5.8 / 8.2 | 6.8 / 8.6 | 6.1 / 8.8 | 6.3 / 9.0 |
| Llama-2-7B | 5.8 / 8.3 | 6.9 / 8.4 | 6.2 / 8.8 | 6.2 / 8.9 |
| Mistral-7B | 5.4 / 7.7 | 8.0 / 10.2 | 6.3 / 9.0 | 6.4 / 9.1 |
| Qwen2.5-7B | 5.3 / 7.6 | 7.2 / 9.6 | 6.0 / 8.5 | 6.2 / 8.8 |

### 3.3 Reading: the two lines close on each other

- **Every lever that cuts compute leaves the bytes almost whole, and then needs r of 44 to 49 GB/s**, at or above
  the reachable bus ceiling (memo-03: about 45). Every lever that cuts bytes (K1 columns) leaves compute where it
  was or raises it. The budget is shared, so the two kinds cannot both run at their frontiers.
- **One cell reaches 12: Yi-1.5-6B with the 4th core at x 1.15, K1 columns at s 0.40 and a device k ≤ 1.6, and
  r ≥ 32** (12.3 printed). R69's pre-registered band for r is 19 to 30. My joint chance: P(r ≥ 32) 0.12 x
  P(4th core ≥ 1.15) 0.4 x P(device k ≤ 1.6) 0.35 x P(resident at 3.3 GB plus the column copy) 0.3, with some
  positive dependence: **about 0.01 to 0.02 for Yi.** That is under my memo-06 figure (0.07), because the
  measured group share came in at the low end and the column kernel does not cut compute.
- **No cell reaches 12 on a second or third model at r ≤ 35.** The best second model at r 35 is Mistral (10.2),
  then ChatGLM3 with the unscreened C6 (9.9 to 10.1), Qwen2.5 (9.6). The Llama-2 class tops out at 8.6 to 9.0.
- **P(all three subjects pass) is under 0.005 in my judgement.** The lead's prior was 0.05 to 0.15. With the H3
  frontiers and T1 in hand I think the goal as written (12 on each of three) should be reported to the owner as
  out of reach by the priced levers, with these tables, while R69 is still awaited: R69 can move Yi's cell, not
  the second and third models.
- **For the choice of subjects.** By these prices the tolerant grouped-attention models (Mistral, Qwen2.5) are
  worth more than the Llama-2 class at r 35 (10.2 and 9.6 against 8.4 to 9.0) and at r 24 (8.0 and 7.2 against 6.8
  to 6.9), although they multiply 0.5 to 0.6 G more weights. Criterion 2 has overtaken criterion 1's order.
  Criterion 1 is "estimated tok/s with the priced levers", so this is the rule as written, not a change to it.

*Note added with §4: a fourth core also changes r. R69 pre-registers cold q4x4 on X1 + 3 A78 at 27 GB/s (21 to 35).
At the pre-registered pair (c x 1.15 to 1.245, r 27) Yi's column corner is byte-bound at 10.5 printed; 12.3 needs
the four-core read rate itself to be 32 or more, at the top of that band.*

## 4. Challenge to Bonnie's re-price

Her draft (`bonnie-reprice.md`) was read on 2026-11-22 after §1 to §3 and §5 above were written. She had seen only
my §1 when she wrote hers.

**Agreed, and where she corrected me.**
- The screened order and the trio by the rule: Yi-1.5-6B, Mistral-7B-v0.3, then ChatGLM3-6B or Qwen2.5-7B. My
  §3.3 reaches the same from the compute side. Our Yi corners agree: she needs r ≥ 31.2 and k ≤ 1.84 on four
  cores (top 13.7 at r 35); I have 12.3 at k 1.6 with a smaller four-core factor and r ≥ 32.
- **Her point 2 stands against my memo-06 §2.3**: I priced groups at 40 % and `down` at s 0.6 through K1 as if one
  measured cost covered both. A group rule drops live neurons that E1's set keeps, so the joint cost is unscreened.
  Her J5 is the right test. My §3.1 here no longer adds them.
- Her points 3 and 4 (T1 is an upper bound; my 0.5 and 0.6 cases are past the measured frontier): conceded; §3.1
  C4 is my own downward re-price of Y8, to the same size as hers (0.1 to 0.5 tok/s).
- Her point 5 (tolerance now outweighs 0.3 GB of weights): agreed, §3.3 last bullet.
- c: she moved to 71.8 G weights/s (40.4 GB/s); my §3 tables use her earlier 69.9. Mine read 0.2 to 0.3 tok/s low
  against hers for that reason alone. The merged memo should use 71.8.

**Where I disagree (both positions stand).**
1. **Her four-core columns move c and hold r.** "r 35, 4 cores" uses the pre-registered hot sum (89.4 G, x 1.245)
   but the read rate of the column. The same pre-registration puts the cold q4x4 rate on X1 + 3 A78 at 27 GB/s
   (21 to 35) against 24 on three threads. A fourth core is one lever with two effects, and the pair R69 predicts
   is (c 89.4, r 27), not (89.4, 35). At that pair Yi with columns is byte-bound: 2.22 GB at 27 GB/s is 82 ms,
   **10.5 printed** (*estimated*), not 13.7. Her 13.7 is the top of a band whose r is above R69's 80 % band; the
   table should carry a "four cores at R69's four-core r" column.
2. **x 1.245 is perfect scaling at the three-thread efficiency.** Campaign 1 and 2 lost with a fourth compute
   thread (R18, R62, *device*, MoE). I price x 1.15 (1.00 to 1.25). At 1.15 her Yi "k max, 4 cores" falls from
   1.84 to about 1.68, still inside K1's device price; Mistral's from 1.49 to about 1.37, under the host floor.
   So the second subject depends on which of us is right, and only R69's core-set cells say.
3. **"The type that fits C" is priced dense only (last column of her §3).** At C 3.1 she gives 5.6 tok/s for Yi on
   Q3_K_M and stops. That is the honest floor, but the table hides that **no sparsity lever exists for a K-type**
   (no exact column re-layout, memo-06 §3.2), so at C 3.1 the levers of both files do not apply at all. It should
   say so in the reading: at the central C the campaign has no priced lever, not a slower one.
4. **Midpoint ranking mixes two different uncertainties.** Her order is the midpoint of the band at r 19 plus
   r 35. The band's low end is k 2.2, where the column kernel is not used and the price falls back to stock, so
   for the 20 % models the "band" is one number and for the 40 % models it is wide. Ranking on midpoints
   understates Mistral's and Yi's lead when k is good and overstates it when k is bad. It does not change the
   order today; it will when two models are closer. I would rank at R69's measured k once it exists and say so
   now.
5. **Her J4 (screen one more full-attention and one more grouped model) before U1.** I would run U1 first: one
   cell (k and v inputs dense) on Llama-2 tests the mechanism directly on files already on disk, and if it holds
   it re-prices five pool models without downloading two more. J4 is the right second step.
6. **ChatGLM3.** She keeps it at rank 3 at s 0.20 "if its screen stands". Tom's prefix check is in since she
   read: the prefix makes stock PPL worse (48.26 → 53.36, 8 chunks, *host*), and the cause of the 33 level is
   open, with the conversion not checked against the publisher's code. A model whose stock level is unexplained
   should not hold a slot; by the rule as she applies it the third is Qwen2.5 until the level is verified
   independently. Her J1 as written is overtaken by that result.
7. **Not adopted by her: the head.** Agreed that it is small (0.1 to 0.35 tok/s). I keep it on the try list
   because it is generic, bit-cheap and already built (campaign 2), and Qwen2.5, if it becomes subject 3, has the
   largest head of the screened set (7.7 % of W).

**What neither of us prices.** My C6 (units of eight) is the only untested lever in either file that cuts
compute on every matrix with a stock-cost kernel, and it needs r ≥ 44 by my own table. Her tables and mine agree
on the end state: one corner for Yi, none for a second model at r ≤ 35.


## 5. Test list and try list (my half; joint list with Bonnie in memo-07)

Tests (host, Tom; no 7B job runs beside another; each on at least two families):

| # | test | cost | closes / opens |
|---|---|---|---|
| U1 | **Where the split sits.** On Llama-2 and Yi (same shape), then DeepSeek: (a) the four per-group cells at 30 %; (b) uniform 30 % with the k and v inputs dense; (c) first N = 1, 8 tokens dense; (d) layers 0 to 3 dense | about 8 cells per model at `-b 512`, the two best at `-ub 1` | k and v dense lifts the MHA frontier to ≥ 0.30 → hypothesis A, and the MHA models are repriced with k, v dense. Nothing moves it by 10 points → the split is taken as a model property (D) and the Llama-2 class is priced at 0.20 for good |
| U2 | **C6 frontier.** H3's tool with the rule "zero the 8 channels of a unit when the unit's largest \|x\| is under threshold", all four input groups, units as in the 4x4 repack, file order and in-block sorted by calibrated magnitude; cells 10 / 15 / 20 / 25 % | 8 cells per model, Yi, Mistral, Llama-2 | under 0.12 on Yi and Mistral → C6 closed. ≥ 0.18 on two models → U3 |
| U3 | **C6 kernel.** Unit skip in the stock `gemv_q4_0_4x4`: identity at an empty mask (`memcmp`), ns per touched weight at u 0.1 / 0.2 / 0.3, hot and cold | a microbench | k > 1.15 at u 0.2 cold → closed |
| U4 | **Up rule (C5).** T1's tool with the score \|u\| and \|u\| x mean \|SiLU(g)\| per neuron, groups of four, 20 / 30 / 40 % | 6 cells per model, Mistral and Qwen2.5, then Yi | under 25 % of groups inside +4.5 % on either model → closed |
| U5 | T1 on Yi and Llama-2 (clustered and file order), to score §1 | 6 cells per model | scores the prediction; under 15 % on Llama-2 falsifies "the split is in the residual inputs" |
| U6 | ChatGLM3 with its prefix (Tom §5.3), then its H3 frontier again on 80 chunks | one stock run, 4 cells | stock PPL with prefix under 15 → the gate question goes to the lead with a number |
| T6 (memo-06) | 1-bit proxy kernel | microbench | only if U4 fails and C4 is still wanted |

Tries (device, after R69; each pre-registered):

| # | lever | gate | class | *estimated* gain (printed) |
|---|---|---|---|---|
| Z1 | X1 + 3 A78 for the dense body (C1), then weighted A55 chunks (C2) | R69's core-set cells | bit-identical | x 1.15 (1.00 to 1.25) on compute; 0 while byte-bound |
| Z2 | K1 columns at the H3 frontier, on the tolerant models only | D2 device k ≤ 1 ÷ (1 − s) | numerics | Yi 9.4 at r 24, 12.3 at r 35 with Z1 |
| Z3 | C6 unit skip | U2, U3 | bit-identical at threshold 0 | +1 to +1.6 when compute-bound; +1.0 at r 24 |
| Z4 | GPU compute lane on two dispatches per layer (C3) | R69 Rule D, X2 kernel ≥ 10 GB/s | numerics | x 1.07 (1.00 to 1.20) |
| Z5 | Up-rule group skip (C5) | U4 | numerics | +0.6 to +0.8 when compute-bound |
| Z6 | Head screen with input sparsity (C7) on the two-stage head | none | numerics | +0.1 to +0.35 |

## 6. Not checked

- No per-group or position cell exists for the MHA models; §2.3 is hypotheses.
- "Massive activations" (Sun et al.) and the SGA-Sparse claim are from memory and an abstract.
- GPU dispatch cost (0.1 ms) and C6's kernel cost (1.05) are guesses; U3 and R69 replace them.
- The 4th-core factor is derived from R69's bands, not measured.
- ChatGLM3's and DeepSeek's group shares of W were taken from shapes, not headers.

## 7. Joint test list and try list (merged from Bonnie's §6 and my §5; she has not reviewed the merge)

Order is by how much each item can move the choice or a price. "B" and "E" mark whose item it is; where we order
differently both orders are given.

**Test list (host, Tom).**

| # | test | from | cost | decides / closes when |
|---|---|---|---|---|
| 1 | Where the split sits: on Llama-2 and Yi, then DeepSeek: per-group cells at 30 %; uniform 30 % with k and v inputs dense; first N = 1, 8 dense; layers 0 to 3 dense; E1 `down` input 40 / 50 / 60 % | E U1, B J3 + J8 | about 12 cells per model | k, v dense lifts the full-attention frontier to ≥ 0.30 → that class is repriced. No 10-point move from any cell → the class stays at 0.15 to 0.20. Llama-2 class under 50 % on `down` → no FFN lever for it |
| 2 | 25 % cell on DeepSeek and Llama-2; 42 % on Yi | B J2 | 6 cells | replaces interpolation; no kill |
| 3 | ChatGLM3: independent check of the stock level (needs the lead's or owner's consent to run the publisher's code), then H3 on 80 chunks | B J1 (overtaken by Tom's prefix result), E U6 | one check, 4 cells | level not reproduced, or still unexplained → not a subject on this gate; Qwen2.5 takes the slot |
| 4 | One newer full-attention model and one unscreened grouped-attention model through H3 | B J4 (she: before item 1; I: after it) | 2 downloads | both land in their class's range → stop screening the pool |
| 5 | Units of eight channels (C6): frontier at 10 / 15 / 20 / 25 %, file order and in-block sorted, on Yi, Mistral, Llama-2; then the kernel microbench | E U2, U3 | 8 cells per model; a microbench | under 0.12 on Yi and Mistral, or k > 1.15 cold → closed |
| 6 | Joint cell: T1 groups at 20 to 30 % plus `down` element thresholds, Mistral and Qwen2.5 | B J5 | 8 cells | no cell beats E1's 60 % in bytes saved → Y8 closes behind E1 |
| 7 | Up rule: groups ranked by \|u\| (and \|u\| x a calibrated gate scale), 20 / 30 / 40 % | E U4 | 6 cells per model | under 25 % of groups on either model → closed |
| 8 | Per-type gemv cost (Q4_0x4, IQ4_NL, Q3_K, IQ3_S) and Q3_K_M / IQ4_XS PPL against F16 on Yi and Mistral | B J7, E memo-06 T5 | microbench, 4 PPL rows | 3-bit above 1.3 x in the best kernel → the fits-C path stays under 8 tok/s |
| 9 | T1 on Yi and Llama-2 | E U5 | 6 cells per model | scores my §1; under 15 % on Llama-2 falsifies "the split is in the residual inputs" |
| 10 | 1- and 2-bit proxy ranking and the proxy kernel | E memo-06 T2, T6; B J6 | cells and a microbench | only if items 6 and 7 leave a proxy lever worth pricing |

**Try list (device; each with its own pre-registration).**

| # | lever or measurement | from | gate | subjects | *estimated* gain (printed) |
|---|---|---|---|---|---|
| 1 | R69 as pre-registered | B D0 | none | none loaded | sets r, c, C, k and the four-core pair (c, r) for every table |
| 2 | Residency of the real Q4_0 files over 20 passes, no-copy path | B D1 | R69: 3.4 step held | Yi, then Mistral | none by itself; without it no Q4_0 price is real |
| 3 | K1 columns at the *host* frontier | B D2, E Z2 | device k < 1 ÷ (1 − s) for a compute-neutral lever; full-set PPL at both settings | Yi, Mistral, third subject | r 24: Yi +1.6 to +3.1, Mistral +1.3 to +2.8 (B); with four cores at r ≥ 32 Yi 12.3 (E); at s 0.20: +0.3 to +1.3 at r 24, 0 or negative above |
| 4 | X1 + 3 A78, then weighted A55 chunks | B D3, E Z1 | R69 core-set cells | all | compute x 1.15 (E) to x 1.245 (B), read rate x 1.13; 0 to +2.0 depending on which line binds |
| 5 | Units-of-eight skip | E Z3 | test 5 | Yi, Mistral | +1 to +1.6 when compute-bound, +1.0 at r 24 |
| 6 | GPU compute lane, two dispatches per layer | E Z4 | R69 Rule D | all | x 1.07 (1.00 to 1.20) |
| 7 | The type that fits C, if the 3.4 step is not held | B D4 | test 8 | Yi | 5.6 (7.3 with a 1.5 x kernel); a floor, with no sparsity lever on it |
| 8 | Group skip: up rule or Y8 | E Z5, B D5 | tests 6, 7, 10 | Mistral, Qwen2.5, Yi | −0.3 to +0.8 |
| 9 | Two-stage head, with an input-sparse screen | E Z6, memo-06 Y11 | none | all | +0.1 to +0.35 |

## 8. Answers to Bonnie's challenges (her final §4), and what is agreed (2026-11-22)

**Agreed conventions for the merged memo.**
- **c = 40.4 GB/s of Q4_0 = 71.8 G weights/s** on X1 + 2 A78 (R69's pre-registered centre, the number R69 scores).
  The weight line at 12 printed is 5.11 G. My §3 tables were computed at 69.9 and read 0.2 to 0.3 tok/s low in
  compute-bound cells; they are left as written and this note governs.
- **Fourth core: centre x 1.15 (1.00 to 1.25)** on compute, with x 1.245 as the perfect-scaling end; she accepts
  this centre. I add, and ask that the merged table carry it: the fourth core's read rate is R69's own four-core
  cell (pre-registered 27 GB/s, 21 to 35), not the column's r.
- Printed = 0.95 x steady; O = 8 ms (6 to 12).

**Her challenges, one by one.**
1. **C6, u too high: conceded on the centre, kept on the kill line.** I checked her arithmetic: for independent
   half-normal channels u = 0.18 puts the threshold at the 0.81 quantile and removes 8.1 % of the input energy,
   against 3.5 % at the element frontier s 0.40 (*estimated*, my recomputation; it matches hers). My 0.18 was a
   guess between two measured points, not a calculation, and hers is the better centre under independence.
   What I keep: independence is not what the block screen showed. Blocks of 32 held 9 to 10 % on the `down`
   input (*host*, memo-02 §8), and the same equal-energy rule under independence gives about 5 % for a block of
   32 (*estimated*), so real inputs beat independence by about a factor of two on that group. A unit of eight is
   never worse than a block of 32 on the same input. **My §3.1 prediction (0.18; 0.12 to 0.25) stands as written
   for scoring; my amended view is 0.13 (0.08 to 0.18) on Yi and Mistral and 0.04 on the Llama-2 class, hers is
   0.10 (0.06 to 0.15) and 0.02.** The kill line of U2 stays at 0.12. At u 0.10 to 0.13 the lever is worth about
   +0.5 to +0.8 tok/s when compute-bound (*estimated*), which is under the cost of a new kernel path: if U2 lands
   where she predicts, C6 closes and I will not argue for it.
2. **Drop the C6 and C3 + C7 columns from the headline table: agreed.** They need r of 44 to 49 and are kept only
   as §3.2's detail.
3. **Groups price under E1: agreed**, and conceded in §4 (the double use of the budget in memo-06 §2.3).
4. **Up rule: agreed.** The lever is the `gate` third only, about 9 % of FFN compute; `down` through K1 adds
   bytes saved and costs compute. U4 runs after U1.
5. **Hypothesis A changes slot 3 only: agreed**, and her price for Llama-2 with k and v dense (8.0 to 9.6 at r 35)
   is the right size. It makes U1 (b) the first cell to run on the files already on disk.
6. **Layout behind my P(resident) 0.3:** I assumed K1 reads a column-major copy and the row layout is not held
   (prefill then runs through the column layout or a per-token path). I did not check the patch. If both layouts
   must be resident the Yi corner closes on residency alone. Tom states which; her request stands as mine.

**Still disagreed (recorded in §4 above, not repeated):** her four-core columns at a fixed r; whether ChatGLM3
may hold rank 3 before its level is verified (she now makes the verification test 1, which I accept as the
resolution: until it passes, Qwen2.5 is third).

**Joint lists: I confirm Bonnie's final §6 as the joint list**, in her order, in place of my §7 above, with three
amendments:
- Test 1 (ChatGLM3 with the publisher's code) waits for consent; **test 2 (U1, cell (b) first) starts without
  waiting for it.**
- Test 6's kill line reads "under 0.12 on Yi and Mistral → closed" with both predictions beside it (hers 0.10,
  mine 0.18 as written, 0.13 amended).
- Try 4 (X1 + 3 A78) records the four-core read rate beside the compute gain, and every "4 cores" price in the
  merged memo uses that pair.

## 9. Request from Bonnie (2026-11-22)

Please check `bonnie-reprice.md` section 7 (slots 2 and 3 after the ChatGLM3 exclusion; agreed conventions, four-core
price at R69's four-core r of 27 and 35) and leave a short note here. My result: by criterion 1 the order is
Llama-2 (k, v dense, 40 %), DeepSeek (same), Mistral (0.40), Qwen2.5 (0.35), with margins of 0.1 to 0.7 tok/s,
inside the estimate error. Please say whether you read Llama-2 against Mistral as a tie for criterion 2.

## 9. Bonnie's addendum (§7) checked, and a challenge to the lead's final choice (memo-06 §6, `3348e3b`) (2026-11-22)

### 9.1 Bonnie §7: reproduced

I recomputed her table with the agreed conventions (c 71.8 G weights/s, O 12 / 6 band ends, printed 0.95 x steady,
head at Q6_K from header shapes, k and v at k = 1). Every cell I recomputed agrees with hers to 0.1 tok/s: the
levers bands at r 19 / 24 / 35 for all four models, k max on three cores (1.20 / 1.21 / 1.19 / 1.08) and four
cores (1.44 / 1.46 / 1.37 / 1.26), and r min (37.5 / 38.6 / 37.1 / 41.8). Her "no candidate for slots 2 and 3
reaches 12" stands. **Check passed, no correction.** I did not recompute her "4 cores at r4" column.

One thing her ranking key hides (not an error): it sums the two band ends, so it averages over the kernel cost.
The order depends on k in one direction (printed tok/s, *estimated*, O 8, at r 24 / r 35):

| K1 device k | Llama-2 (40 %, k, v dense) | DeepSeek (same) | Mistral (40 %) | Qwen2.5 (35 %) | Llama-2 at 35 %, if its 40 % cell fails on the full set |
|---|---|---|---|---|---|
| 1.55 | 8.0 / 10.0 | 7.8 / 10.2 | 8.0 / 9.5 | 7.2 / 8.9 | 7.6 / 9.4 |
| 1.8 | 8.0 / 9.0 | 7.8 / 9.1 | 8.0 / 8.3 | 7.2 / 7.8 | 7.6 / 8.4 |
| 2.2 | 7.7 / 8.3 | 7.8 / 8.2 | 6.9 / 7.7 | 6.5 / 7.6 | 7.2 / 8.3 |
| 2.6 (R69's pre-registered P2col centre) | 6.7 / 8.3 | 6.9 / 8.2 | 5.9 / 7.7 | 5.6 / 7.6 | 6.3 / 8.3 |

Why: under the k, v dense rule a sixth of the body runs at k = 1, so a dear kernel costs the MHA models less.
Mistral is level only at k ≤ about 1.6. From k 1.8 upward the two MHA models lead it by 0.5 to 1.0 tok/s at r 24
and by 0.6 to 0.8 at r 35, and that holds even if Llama-2 drops to 35 %.

### 9.2 Challenge to the lead's tie reading

**I think the tie reading is wrong**, for three reasons. The lead's choice is recorded as a lead judgement and
the strict alternative is on the record; this is my position beside it.

1. **The error invoked is common-mode.** c, r and k are R69 centres, but an error in them moves all four models
   together. It flips the order only through k, and §9.1 shows which way: the gap is a sign that depends on one
   parameter, not noise. Inside the k band both files use (1.55 to 2.2) Mistral never leads, and at R69's own k
   centre (2.6) it trails by 0.8 to 1.0 at r 24. A tie should be read off differential uncertainty, and the one
   differential that matters favours the MHA models.
2. **The differential uncertainties that exist do not reach Mistral either.** Llama-2's 40 % cell is 0.5 point
   under the line. One grid step down (35 %) leaves it level with or above Mistral from k 1.8 upward (last
   column).
3. **Both tie-breaks inside the lead's reading are weak.**
   - Criterion 2 puts DeepSeek ahead of Llama-2 on margin: +2.95 / +3.54 % against +3.97 / +3.95 %, each with
     SE about 0.55 (*host*, tom-split §5.1). That is about 1.3 SE at `-b 512` and 0.5 SE at `-ub 1`.
   - DeepSeek's cells were screened **without BOS**, Llama-2's with it, and the served DeepSeek file sets BOS
     (lead's freeze note). The attention-sink token is exactly what a BOS changes, so DeepSeek's margin is not
     like-for-like.

**What follows by the rule as written** (section 1: R69 is not waited for, and criterion 1 comes first): subjects
Yi-1.5-6B, Llama-2-7b-chat, DeepSeek-LLM-7B; reserves Mistral-7B-v0.3, then Qwen2.5-7B. The lead's set swaps
Mistral in for Llama-2. Neither set changes the verdict: no subject 2 or 3 reaches 12 with priced levers, so the
swap moves 0.0 to 1.0 tok/s of a sub-12 result and the generality report, not a pass. The lead's side effect (one
GQA 7B in the set, Mistral's host evidence from E1, T1 and K1) is a real merit, but it is not a criterion.

**If the lead keeps the choice, I ask for three things before the freeze (S3)**, each a host cell or two:
- (a) DeepSeek's 40 % k, v dense cell rerun **with BOS**, at both settings. If it fails, slot 3 passes to
  Llama-2 by the rule, with no new judgement.
- (b) Mistral's 40 % frontier re-screened on our own quantised file. memo-02 used the second-state stand-in;
  S3 requires a file quantised by us.
- (c) A dated note in memo-06 §6: the tie holds only at k ≤ about 1.6, and R69's D2 row is the measurement that
  would have decided it.
