# c3 memo-10: how wide can a verify pass go? Trees and multi-candidate verification at widths 2 to 8

Date: 2026-11-22. Author: Bonnie (`claude-fable-5-2`). **Ebony has not reviewed this**: this session had no tool to
start her, so §9 lists what I ask her to attack. Not committed. Nothing here is measured by me. Every price is
*estimated* (arithmetic); quoted inputs carry their own label (*device* R72, *host* tom-head.md, *published*).
Script with self-checks: [memo-10-work/bonnie_tree.py](memo-10-work/bonnie_tree.py) (reproduces R72 §5's
6.90 / 6.21 / 6.22, Tom's 115.3 ms, and Ebony's kB 1.06 / 1.22).

Inputs: [memo-09-gap.md](memo-09-gap.md) §1 to §10, [memo-09-work/tom-head.md](memo-09-work/tom-head.md),
[R72-result.md](R72-result.md). Lead decisions used: the `tok` head variant is inside the owner's trained-head
decision; the Llama 2 licence reading is accepted.

"Width" W = positions in one pass: the last committed token plus W − 1 candidates. Goal: 79.2 ms per token.

## 1. Verdict

- **The best width is 2, on every subject, at every clock state priced.** Width 3 loses 0.8 to 1.0 tok/s against
  width 2 at the centre, width 4 loses 1.4 to 1.7, widths 6 and 8 are slower than stock.
- **Compute overtakes the bus at 1.75 positions at the centre** (1.4 at the L4 clock caps, 2.5 at full clocks with
  the X1 on-core). The body is bus-bound at one position with only 1.3 to 1.7 × of compute slack, and the first
  extra position spends 0.45 of it. From there each further position costs 0.23 to 0.32 of a body pass.
- **No width reaches 12 tok/s on any subject at the centre.** Best stack (width 2, the trained head, GPU lane g 8,
  C7, C5, O 5.2): **9.8 / 9.5 / 9.3** tok/s (Yi / Llama-2 / DeepSeek); with Ebony's inputs 9.3 / 9.0 / 8.8. With
  every band at its top (g 15, full clocks): 11.3 / 10.9 / 10.6.
- A wider pass needs a better guesser, and no tree shape replaces that: width 3 beats width 2 only when head
  top-1 is at least 0.50 (full clocks) to 0.68 (centre); width 4 needs 0.80. Tom measured 0.379 and 0.497 (*host*).
- **What does add a little: a confidence gate** (width 2 only when the head is confident, else width 1): about
  −2 to −6 ms per token, +0.1 to +0.4 tok/s, under an assumed confidence distribution that Tom can replace with
  a count from his existing dumps.
- This agrees with campaign 1's R20 (*device*: "verifying k tokens costs about k × one token", compute-bound above
  batch one) once the kernel shares the weight read: R72's bus-bound finding buys one cheap extra position, not a
  tree.
- P(all three at 12): about 0.002 (my judgement; 0.005 in memo-09, lowered because the measured head is below the
  0.60 that memo-09's M row assumed, and widening does not make up for it).

## 2. Pass cost by width, kB(W)

Model (Ebony's two-point fit, memo-09 §10.3, extended to W positions):

    t(c) = max(1/R, c/C) + 0.25 × min(1/R, c/C)        kB(W) = t(c(W)) ÷ t(1)
    R = 29.9 GB/s read (*device*, R72 block a);  C = hot compute rate of the four threads, GB/s
    c(W) = 1 + 0.45 × (W − 1): the Q4_0 unpack is shared, the sdot and float steps repeat per position

- 0.45 per extra position: memo-09 §5.1 and §10.3 (agreed band 0.40 to 0.60, operation count on the A78's two
  vector pipes; A78 at 0.118 cycles per weight, *device*, R72 L3). The KleidiAI gemm at four rows was read by Ebony
  as 2.3 to 2.6 × a gemv (*estimated* from assembly); the line gives 2.35. No device number exists for any W above 1.
- C: three A78 hot on-core 34.6 GB/s (*device*, L3); the X1 at 20.5 GB/s is a prediction never measured on-core.

| state | C (GB/s) | W=2 | W=3 | W=4 | W=6 | W=8 | compute = bus at W |
|---|---|---|---|---|---|---|---|
| full clocks, X1 at its prediction | 55 | 1.05 | 1.13 | 1.34 | 1.78 | 2.21 | 2.9 |
| full clocks, less 10 % (Ebony) | 50 | 1.06 | 1.21 | 1.44 | 1.91 | 2.38 | 2.5 |
| **centre** | **40** | **1.12** | **1.41** | **1.69** | **2.26** | **2.82** | **1.75** |
| L4 caps (A78 1,555 MHz), X1 on-core | 35 | 1.23 | 1.54 | 1.86 | 2.49 | 3.13 | 1.4 |
| L4 caps, X1 thread on a small core | 24 | 1.37 | 1.75 | 2.12 | 2.87 | 3.62 | below 1 |

- **Centre C = 40** is chosen so that kB(2) = 1.12, midway between memo-09's two centres (mine 1.08, Ebony's 1.15).
  I have moved towards hers: a decode is sustained, and the L4 arms, the nearest thing measured, ran capped.
- Beyond the crossover a position costs 0.45 ÷ C of a full read: 0.23 (C 50), 0.28 (C 40), 0.32 (C 35) of a stock
  body pass. On Yi that is 28 to 38 ms per added position.
- Other per-position costs (charged as `xpos`, *estimated*): 0.5 ms (Yi) and 1.5 ms (Llama-2, DeepSeek) per extra
  position for attention arithmetic, norms and activations. The KV read itself is shared across positions: ggml's
  generic `mul_mat` walks 16 × 16 blocks (read in `lab/llama.cpp-base/ggml/src/ggml-cpu/ggml-cpu.c`, line 1203),
  so K and V rows are read once for up to 16 positions. Not measured.

## 3. The tree mask and the KV rollback

Read in `lab/llama.cpp-base` (base `620d2225c`):

- A tree needs no new attention code. Each candidate goes into the batch with its own sequence id; the prefix cells
  carry all ids. `llama_kv_cache::seq_cp` inside one stream is metadata only ("since both sequences are in the same
  stream, no data copy is necessary", `src/llama-kv-cache.cpp` line 486). `examples/speculative/speculative.cpp`
  already verifies a tree this way (`n_parallel`, `p_split`, lines 463 to 470).
- **It must run with the unified KV buffer.** With one stream per sequence, `seq_cp` copies buffers and asserts on a
  partial range (line 528), and the context is divided by `n_seq_max` (`src/llama-context.cpp` line 382).
- Per pass: one `seq_cp` per candidate before, `seq_keep` / `seq_rm` after; the mask is W × n_kv floats. I price all
  of it at under 0.1 ms per pass (*estimated*; not measured). It is not what limits the width.
- What a miss costs is the pass time, not the rollback: a rejected candidate's KV cells are freed by metadata.
- Not covered by any check yet: sibling candidates share one position with different sequence ids. T1b fed only true
  tokens in a chain (tom-head.md §7).

## 4. Acceptance: what is measured and what is assumed

| quantity | Yi | Llama-2 | DeepSeek | label |
|---|---|---|---|---|
| head top-1, prompt set (`tok`, best run) | 0.379 | 0.497 | 0.438 | *host* (Tom); **DeepSeek: no head exists, mean of the two, *estimated*** |
| tokens per pass at width 2 | 1.326 | 1.448 | 1.385 | *host*; DeepSeek *estimated* |
| d = (tokens per pass − 1) ÷ top-1 | 0.86 | 0.90 | 0.88 | from the two rows above |
| top-5 | 0.62 | 0.73 | 0.68 | ***estimated*, not measured** |
| top-2 / top-3 / top-7 | 0.47 / 0.54 / 0.68 | 0.59 / 0.65 / 0.78 | 0.53 / 0.59 / 0.73 | *estimated* |
| P(depth n + 1 top-1 right, given depth n right) | 0.379 | 0.497 | 0.438 | **assumed** equal to top-1; band 0.7 to 1.15 × |

- **Top-k source:** the only published pair for a Medusa-type head is Together AI's post (Vicuna, in-domain chat):
  top-1 "approximately 60 %", top-5 "over 80 %". I carry that step (+0.98 in log-odds) to Tom's top-1 and spread
  ranks 2 to 5 by 1 ÷ rank. The Medusa paper has no per-head table (memo-09 §10.5; one search today found none
  either). Band on top-5: ±0.07. Tom's dumps hold each row's top-64 logits, so the real numbers are a count away.
- **d is below 1** because hits come in runs: easy stretches are crossed two tokens at a time, so passes start more
  often in hard stretches than rows do. I apply the measured d to every tree. For chains this is the cautious side
  (runs help a chain).
- **Depth 2 and deeper** need more heads (Medusa: one head per depth, trained on (h_j, y_{j+2})) or a chained head
  (EAGLE-like). Neither exists. EAGLE's published acceptance by depth for Vicuna-7B falls slowly (0.79, 0.74, 0.72,
  0.73, 0.67 on MT-bench, temperature 0), but that draft is a whole decoder layer of 0.24 B parameters with
  top-1 near 0.8; EAGLE-3's authors report that a second draft token can fall sharply (search summary).
- The official 32-token run gives 31 guesses per model (top-1 0.39 to 0.45 on Yi, 0.19 to 0.29 on Llama-2, *host*);
  one hit moves it by 0.03. The prompt set is the basis here; the claim run will be noisier than these centres.

## 5. Prices by width

    ms per committed token = (body × lane × kB(W) × c5 + dispatch + O + guess × depths + xpos × (W − 1)) ÷ tau + head × hs
    tau = 1 + d × (sum of the path probabilities of the W − 1 candidate nodes);   lane = 26 ÷ (26 + 0.64 × g ÷ c(W))
    guess = 2.64 / 2.61 / 2.59 ms per head evaluation (two blocks + an 8,192-row shortlist; tom-head.md §5)

### 5.1 Shapes compared (lever alone, centre C 40): tokens per pass / printed tok/s

| W | shape | Yi | Llama-2 | DeepSeek |
|---|---|---|---|---|
| 1 | stock | 1.00 / 6.90 | 1.00 / 6.21 | 1.00 / 6.22 |
| **2** | **one guess** (batch 2) | **1.33 / 7.94** | **1.45 / 7.79** | **1.39 / 7.38** |
| 3 | top-2 at one depth | 1.41 / 6.88 | 1.53 / 6.65 | 1.47 / 6.39 |
| 3 | chain of 2 (two depths) | 1.45 / 6.99 | 1.67 / 7.17 | 1.55 / 6.66 |
| 4 | top-3 at one depth | 1.46 / 6.05 | 1.58 / 5.78 | 1.52 / 5.61 |
| 4 | two depths: 2 + 1 | 1.53 / 6.25 | 1.75 / 6.32 | 1.64 / 5.94 |
| 4 | chain of 3 | 1.50 / 6.05 | 1.78 / 6.36 | 1.63 / 5.86 |
| 6 | top-5 at one depth | 1.53 / 4.87 | 1.65 / 4.58 | 1.59 / 4.51 |
| 6 | two depths: 4 + 1 | 1.62 / 5.10 | 1.84 / 5.06 | 1.73 / 4.83 |
| 6 | Medusa-style greedy tree (3+1+1; Llama-2 2+1+1+1) | 1.63 / 5.07 | 1.92 / 5.18 | 1.76 / 4.88 |
| 8 | top-7 at one depth | 1.58 / 4.07 | 1.70 / 3.80 | 1.64 / 3.78 |
| 8 | two depths: 5 + 2 | 1.69 / 4.30 | 1.92 / 4.24 | 1.80 / 4.09 |
| 8 | Medusa-style greedy tree (5+1+1; 4+1+1+1; 4+2+1) | 1.70 / 4.31 | 2.01 / 4.39 | 1.84 / 4.15 |

- "Medusa-style greedy tree" = the W − 1 nodes with the highest path probability (Medusa's own construction).
- A chain beats siblings at every width up to 4: the second-ranked sibling is right in 0.09 of passes, the first
  node of the next depth in 0.14 to 0.25. Neither pays for a position that costs 0.28 of a body pass.
- **EAGLE-2 dynamic trees:** EAGLE-2 reranks 60 draft nodes per pass at depth 6 and gains tau 3.94 → 4.98 over
  EAGLE's static tree (Vicuna-7B, MT-bench, *published*). Its regime is a GPU, where 60 positions cost about one.
  Here every node costs 0.28 of a pass, so the only part that transfers is choosing the width per step from the
  head's confidence, which EAGLE-2 reports to be calibrated (confidence below 0.05: acceptance about 0.04; above
  0.95: about 0.98). At our accuracy the choice is between width 1 and width 2: the "gated" row below.

### 5.2 Best shape per width, lever alone, three clock states (printed tok/s)

| W | full clocks (C 50) | centre (C 40) | L4 caps (C 35) |
|---|---|---|---|
| 1 | 6.90 / 6.21 / 6.22 | 6.90 / 6.21 / 6.22 | 6.90 / 6.21 / 6.22 |
| **2** | **8.35 / 8.22 / 7.75** | **7.94 / 7.79 / 7.38** | **7.36 / 7.20 / 6.86** |
| 3 | 7.97 / 8.22 / 7.55 | 6.99 / 7.17 / 6.66 | 6.45 / 6.60 / 6.16 |
| 4 | 7.19 / 7.34 / 6.79 | 6.25 / 6.36 / 5.94 | 5.75 / 5.84 / 5.48 |
| 6 | 5.92 / 6.01 / 5.63 | 5.10 / 5.18 / 4.88 | 4.66 / 4.73 / 4.48 |
| 8 | 5.03 / 5.13 / 4.82 | 4.31 / 4.39 / 4.15 | 3.93 / 4.00 / 3.79 |
| 1 or 2, gated | 8.42 / 8.26 / 7.81 | 8.12 / 7.90 / 7.52 | 7.74 / 7.45 / 7.15 |

- Width 3 ties width 2 in one cell only (Llama-2, full clocks, 8.22 both).
- The gate's row assumes the head's top-1 confidence is calibrated and spread as a Beta distribution with the
  measured mean and concentration 1.5. **That distribution is an assumption**, not a measurement. The gate pays
  most where kB is worst (caps: +0.25 to +0.38 tok/s), because it removes the passes that were going to miss.

### 5.3 Stacks (ms per token → printed tok/s; best shape per width)

Small levers as memo-09 §4: G = GPU lane g 8 GB/s (its batch-W cost: g ÷ c(W)), C7 head × 0.5, C5 body × 0.93,
O 8 → 5.2 ms.

| stack | W | Yi | Llama-2 | DeepSeek |
|---|---|---|---|---|
| exact only: tree + O 5.2, C 40 | 2 | 117.5 → 8.08 | 120.0 → 7.92 | 126.7 → 7.50 |
| | 3 | 134.0 → 7.09 | 130.9 → 7.26 | 140.9 → 6.74 |
| | 4 | 150.1 → 6.33 | 147.7 → 6.43 | 158.2 → 6.01 |
| | 6 | 184.7 → 5.14 | 182.1 → 5.22 | 192.9 → 4.92 |
| | 8 | 218.8 → 4.34 | 215.0 → 4.42 | 227.4 → 4.18 |
| | gated | 114.9 → 8.27 | 118.3 → 8.03 | 124.4 → 7.64 |
| **+ G (8, dispatch 3) + C7 + C5 + O 5.2, C 40** (my inputs) | 1 | 106.4 → 8.93 | 119.6 → 7.94 | 116.8 → 8.14 |
| | **2** | **96.8 → 9.82** | **100.1 → 9.49** | **102.7 → 9.25** |
| | 3 | 113.2 → 8.39 | 111.8 → 8.50 | 117.3 → 8.10 |
| | 4 | 128.7 → 7.38 | 128.2 → 7.41 | 133.9 → 7.09 |
| | 6 | 161.4 → 5.89 | 161.1 → 5.90 | 167.1 → 5.69 |
| | 8 | 193.6 → 4.91 | 192.1 → 4.94 | 199.6 → 4.76 |
| | gated | 93.1 → 10.20 | 97.6 → 9.73 | 99.5 → 9.55 |
| same with Ebony's inputs (dispatch 8, kB(2) 1.15) | 2 | 102.5 → 9.27 | 105.6 → 8.99 | 108.3 → 8.77 |
| | 3 | 119.1 → 7.98 | 117.3 → 8.10 | 123.0 → 7.72 |
| | gated | 98.2 → 9.68 | 102.6 → 9.26 | 104.5 → 9.09 |
| top of every band: G (15) + C7 + C5 + O 5.2, C 50 | 2 | 84.4 → 11.25 | 86.9 → 10.93 | 89.9 → 10.57 |
| | 3 | 92.6 → 10.26 | 91.1 → 10.43 | 96.5 → 9.84 |
| | 4 | 105.5 → 9.00 | 105.1 → 9.04 | 110.4 → 8.60 |
| | gated | 82.0 → 11.59 | 85.2 → 11.15 | 87.7 → 10.84 |

**Is there a width at which all three reach 12 at the centre? No.** Tokens per pass needed for 12 against what the
measured head gives:

| stack | W | needed: Yi / Llama-2 / DeepSeek | have (best shape) |
|---|---|---|---|
| exact only, C 40 | 2 | 2.05 / 2.24 / 2.39 | 1.33 / 1.45 / 1.39; **above the ceiling of 2** |
| | 3 | 2.55 / 2.79 / 2.98 | 1.45 / 1.67 / 1.55 |
| | 4 | 3.04 / 3.34 / 3.58 | 1.53 / 1.78 / 1.64 |
| + G (8) + C7 + C5 + O 5.2, C 40 | 2 | 1.64 / 1.84 / 1.84 | 1.33 / 1.45 / 1.39 |
| | 3 | 2.07 / 2.35 / 2.34 | 1.45 / 1.67 / 1.55 |
| | 4 | 2.51 / 2.85 / 2.84 | 1.53 / 1.78 / 1.64 |
| | 8 | 4.24 / 4.86 / 4.84 | 1.70 / 2.01 / 1.84 |
| top of every band | 2 | 1.42 / 1.59 / 1.59 | 1.33 / 1.45 / 1.39 |
| | 3 | 1.68 / 1.89 / 1.89 | 1.45 / 1.67 / 1.55 |

- The need grows by about 0.45 to 0.5 tokens per added position; the measured head adds 0.05 to 0.22. The gap
  widens with width at every stack.
- Without the GPU lane, 12 needs more than 2 tokens per pass at width 2, which cannot happen. So the lead's "M
  alone prices about 8 tok/s" holds, and no tree lifts it.
- With every band at its top, width 2 needs top-1 of about 0.49 / 0.66 / 0.67 (tau − 1, divided by d). Llama-2's
  head is at 0.497.

### 5.4 P (my judgement)

| claim | P |
|---|---|
| width 2 with the trained head delivers its centre alone (7.9 / 7.8 / 7.4) on the device | 0.30 (needs T2's kernel at kB ≤ 1.12 sustained, and the host top-1 to hold) |
| the gate adds 2 ms or more on all three, given width 2 works | 0.5 |
| width 3 beats width 2 on any subject on the device | 0.05 |
| width 4 or more beats width 2 on any subject | under 0.01 |
| the centre stack row (9.8 / 9.5 / 9.3) is reached on all three | 0.02 (L kernel 0.30 × G 0.25 × C7 and C5, not independent) |
| any subject at 12 | 0.02 (Yi, top of every band plus a head near 0.5) |
| all three at 12 | 0.002 |

## 6. What would change the answer

1. **A much better guesser.** Width at which the best shape is found, by head top-1 (Yi's body, d 0.88):

   | clocks | width 2 beats 1 | width 3 beats 2 | width 4 beats 3 |
   |---|---|---|---|
   | C 50 | 0.10 | 0.50 | 0.80 |
   | C 40 | 0.16 | 0.68 | 0.81 |
   | C 35 | 0.27 | 0.69 | 0.82 |

   An EAGLE-size draft (one decoder layer, 0.24 B parameters, *published* top-1 0.74 to 0.82 on its own chat data)
   priced as a what-if, guess = one layer of the body (3.8 to 4.4 ms) plus the shortlist:

   | assumed top-1 | alone, C 40 | centre stack | top of every band |
   |---|---|---|---|
   | 0.60 (width 2) | 8.98 / 8.14 / 7.98 | 11.12 / 9.89 / 10.01 | 12.70 / 11.35 / 11.39 |
   | 0.75 (width 3; width 2 for the MHA pair in the centre stack) | 9.95 / 9.03 / 8.77 | 12.05 / 10.73 / 10.81 | 14.58 / 13.03 / 12.94 |

   - So the only priced cell with all three above 12 is: top-1 0.75, the GPU lane at g 15, full clocks, C7 and C5.
   - Against it: Tom's block reached 0.38 to 0.50 where the same source promised 0.60; the published 0.75 took
     ShareGPT-scale data and days of GPU time; the layer is 0.14 GB (Q4_0) of extra file, read every pass; campaign 1
     lost with EAGLE-3 on the host (`reports/02_methods.md` row 10, on the MoE, where verification grew with the
     union of experts; that reason does not carry to a dense model, the training cost does).
   - Whether a 0.24 B-parameter layer is "a small guess head" under the owner's 2026-11-22 decision is the lead's
     question, not mine. I do not propose it; I record what accuracy the goal would need.
2. **Cheaper extra positions.** At 0.30 per position instead of 0.45 the best width is still 2 (8.43 / 8.30 / 7.82
   alone, C 40). The width does not move inside the band 0.30 to 0.60; the price of width 2 moves by ±0.5 tok/s.
3. **More compute beside the same read.** The four A55 cores ran at 3.95 G weights/s each when a pinned thread
   landed on them (*device*, R72): 8.9 GB/s together, which would lift C from 40 to about 49 if they computed the
   extra position from weights the big cores have just pulled into the shared L3. Campaign 1 R18 lost 13 % with an
   A55 helper (*device*; an A55 in every sync). I price it at −5 ms at best and P 0.05; listed as Y9, last.

## 7. Lists for Tom (through the lead)

### 7.1 Test list (all host, free; each decides a question)

| # | test | cost | closes when |
|---|---|---|---|
| T7 | **Top-k acceptance.** From the existing prompt-set dumps and `tok` heads (Yi, Llama-2): head top-1, 2, 3, 5, 7 against the greedy next token, row-weighted, and simulated tokens per pass for top-2 and top-3 at one depth | a few lines in `tools/c3_head_train.py` scoring | top-2 − top-1 < 0.20 on both → sibling shapes closed (a third position must be the right one in about 0.20 of passes to pay at C 50, about 0.30 at C 40; my estimate is 0.09) |
| T8 | **Confidence calibration and the gate.** Histogram of the head's top-1 probability in ten bins with the hit rate per bin; simulated tokens per pass and share of width-2 passes for thresholds 0.05 to 0.5; print the pass-weighted and row-weighted accuracy separately | same script | gain of the best threshold under 1.5 % of ms per token at kB 1.12 and 1.23 → gate not built |
| T9 | **Depth 2.** Train a second head on (h_j, y_{j+2}) with the same script and data; report P(depth 2 right, given depth 1 right) and simulated tokens per pass for a chain of 2 | minutes per model (blocks trained in 71 to 393 s, *host*) | conditional top-1 < 0.50 on Llama-2 → width 3 closed (it needs about 0.50 at full clocks) |
| T10 | **Tree identity.** Extend `tools/c3_head_b2check.cpp`: batches of 3, 4 and 8 with sibling candidates on separate sequence ids over a shared prefix (unified KV), wrong candidates included; every position's logits bitwise against the one-token decode of the same path; then `seq_rm` of the rejected cells and 32 more greedy steps bitwise against the plain run. ggml repack (gemv to 3 positions, gemm from 4) and KleidiAI | minutes | any difference → the tree (and the rollback of Y1) is not exact as built; it also decides whether width 4 can use the stock gemm |
| T11 | **DeepSeek head** with the committed scripts (two-model rule is met, but DeepSeek's price above rests on a mean) | about 1 hour of dumps | none; replaces an estimate |
| T2 (memo-09, device) | add a **batch-3 arm** to the kB cell, same 60 s arms | one more arm | kB(3) ÷ kB(2) > 1.15 sustained → width 3 closed whatever the head |

### 7.2 Try list (build only after its test; env-gated, off by default, model-generic, two subjects)

| # | lever | after | label | price (Yi / Llama-2 / DeepSeek) | P |
|---|---|---|---|---|---|
| Y1 + Y5 | **Width 2 with the trained head** (memo-09's items, unchanged): new two-vector Q4_0 kernel, `tok` head sidecar with an 8,192-row shortlist, KV drop on a miss, lazy second head | memo-09 T2, T1b; T10 for the rollback | exact output | −17.9 / −31.1 / −24.0 ms alone at C 40 (7.94 / 7.79 / 7.38 tok/s) | 0.30 |
| Y6 | **Confidence gate** on Y1: width 1 when the head's top-1 probability is under a threshold fixed on the training text | T8 | exact output; runtime only | −2.7 / −1.7 / −2.3 ms at C 40; −6.3 / −4.4 / −5.5 at caps | 0.5 given Y1 |
| Y7 | Width 3 as a chain of 2 with a second head | T9, T10, T2's batch-3 arm | exact output | +16 / +11 / +14 ms against Y1 at C 40 (a loss); 0 to +6 at full clocks | 0.05 that it beats Y1 anywhere |
| Y8 | Widths 4 to 8, any shape | not proposed | – | a loss at every state priced | under 0.01 |
| Y9 | A55 cores compute the second position from L3 | Y1 on the device first | exact output | −5 ms at best | 0.05 |

Order I propose: T7, T8 and T10 now (no new data needed). T9 and T11 when the laptop is free. Nothing here changes
the phone order of memo-09 §10.10; T2 gains one arm.

## 8. Not done, and limits

- No host or device run. Top-k, depth-2 acceptance and the confidence distribution are estimates; DeepSeek has no
  head. kB above width 1 has no device number at all, and the X1's hot rate was never measured on-core.
- The fit behind kB is two points (Ebony's caution stands).
- The Medusa and EAGLE numbers are from GPU papers on other models; only the shapes of their trees are used.
- The mask and rollback cost is read from the code, not timed.

## 9. For Ebony's review

1. c(W) = 1 + 0.45 × (W − 1): is the per-position cost really linear to 8, or does a 4- or 8-row kernel do better
   per position than the two-vector kernel (register pressure on 32 NEON registers; the KleidiAI 16x4 block)?
2. The centre C = 40: I moved kB(2) from 1.08 to 1.12. Does she hold 1.15?
3. Top-k from one blog pair (+0.98 log-odds): is there any published per-head top-k table for Medusa, Hydra, or a
   later multi-head method on a 7B chat model?
4. Depth 2: published conditional acceptance for a block-size head (not a full layer), 2025 to 2026.
5. Is there a 2025 to 2026 guesser between Tom's block (36 M parameters, 0.38 to 0.50) and an EAGLE layer (0.24 B,
   0.75 published) that trains on a laptop? §6 item 1 shows this is the quantity that matters, not the tree.
6. The gate: does a threshold fitted on the training text count as a calibration-derived value under the
   2026-11-19 decision, as I read it?

## 10. Sources (read 2026-11-22)

- EAGLE, arXiv 2401.15077v3 (fetched; Tables 2 and 8, draft size 0.24 B, chain 3.20 against tree 3.94):
  <https://arxiv.org/html/2401.15077v3>
- EAGLE-2, arXiv 2406.16858v1 (fetched; 60 draft tokens, depth 6, calibration figures, Table 1):
  <https://arxiv.org/html/2406.16858v1>
- EAGLE-3, arXiv 2503.01840 (search summary only): <https://arxiv.org/pdf/2503.01840>
- Medusa, arXiv 2401.10774 (tree construction; no per-head accuracy; search summary today, full read in memo-09
  §10.5): <https://arxiv.org/pdf/2401.10774>
- Together AI, Medusa post (top-1 about 60 %, top-5 over 80 %; quoted from memo-09 §10.5, not re-read today):
  <https://together.ai/blog/medusa>
- Repository: `lab/llama.cpp-base/src/llama-kv-cache.cpp`, `src/llama-context.cpp`,
  `examples/speculative/speculative.cpp`, `ggml/src/ggml-cpu/ggml-cpu.c` (base `620d2225c`);
  `docs/negative-results.md` and `reports/02_methods.md` (R20, R18, EAGLE-3 on the host).

## Ebony review

Date: 2026-11-22. Reviewer: Ebony (`claude-fable-5-2`). Appended, not committed. §1 to §10 above are Bonnie's and
are unchanged; the header's "Ebony has not reviewed this" was true until this section. Labels: *host* = this
laptop (Apple M4 Pro), relative only; *estimated* = arithmetic or an instruction count; *published* = a paper's
number. Nothing here is a phone number. Work files (untracked):
[memo-10-work/ebony_gw.c](memo-10-work/ebony_gw.c) (a W-position Q4_0 kernel that unpacks each weight block once),
[memo-10-work/ebony_reprice.py](memo-10-work/ebony_reprice.py) (her model with c(W) swapped; asserts her 7.94 / 7.79
/ 7.38 and 9.82 / 9.49 / 9.25 first).

### E1. Verdict

- **Her verdict stands: the best width is 2, and no width or stack reaches 12 on any subject at the centre.** With
  my inputs the best centre stack is **9.3 / 9.1 / 8.8** tok/s (Yi / Llama-2 / DeepSeek), gated 9.7 / 9.3 / 9.1.
- **The attack on c(W) fails. An unpack-once kernel does not make the extra position cheap**, because the unpack
  is the small part of a Q4_0 gemv. Counted on the compiled kernel, the marginal position costs **0.52** of a stock
  gemv (hers: 0.45), constant from width 2 to 8. The M4 gives 0.42 to 0.49 (*host*). A marginal cost of 0.15 is
  not reachable with exact output: 8 SDOT per 128 weights per position alone are 0.26 to 0.32 of a stock gemv.
- One opening remains, for width 2 only: if the A78 runs the stock kernel below its issue limit (R72 says it is
  not issue-bound on K1), the first extra position could cost as little as 0.19. That is a bound, not a centre.
  Even there no stack reaches 12 at the centre (10.4 to 10.8 / 10.1 to 10.5 / 9.8 to 10.2), and width 3 still
  needs top-1 of 0.60.
- **The draft-layer what-if reaches 12 on Yi only** at the centre (12.0 / 10.7 / 10.8 at top-1 0.75 with her
  inputs; 11.4 / 10.2 / 10.3 with mine). It is a draft model, not a guess head (E5), and it cannot be trained to
  its published accuracy on the laptop.
- **P(all three at 12): about 0.002 to 0.003** (my judgement; hers 0.002). Agreed.

### E2. kB(W): the unpack-once split (her §2, §9 items 1 and 2)

**What I built to count it.** `ebony_gw.c`: the stock 4x4 gemv's block loop with the four `shl` and four `and`
done once per weight block, the weight scales converted once, and then, per position, the stock sequence (8 SDOT
into one sum, SCVTF, FMUL, FMLA) in the stock order. The activation scales are converted to float once per call
(exact).

- **Identity (*host*):** every position's output equals the stock gemv's on the same input, by `memcmp`, at widths
  1, 2, 3, 4, 6 and 8 (256 rows × 4,096 columns, random weights and activations). This is not Tom's T10; it shows
  only that the split keeps the per-column operation order.
- **A78 issue count (*estimated*):** NDK r28 clang, `-O3 -mcpu=cortex-a78`, counted with `tools/c3_a78_pipes.py`
  (SCVTF and FCVTL as two V0 cycles, as in memo-08):

  | kernel | V0 / V1 / either, per 128 weights | slots | floor, cycles | against stock |
  |---|---|---|---|---|
  | stock 4x4 gemv | 6 / 4 / 15 | 25 | 12.5 | 1.00 (Tom's count, reproduced) |
  | W = 1 | 4 / 4 / 15 | 23 | 11.5 | 0.92 |
  | W = 2 | 6 / 4 / 26 | 36 | 18.0 | **1.44** |
  | W = 3 | 8 / 4 / 37 | 49 | 24.5 | 1.96 |
  | W = 4 | 10 / 4 / 48 | 62 | 31.0 | 2.48 |
  | W = 6 | 14 / 4 / 71 | 89 | 44.5 | 3.56 |
  | W = 8 | 18 / 4 / 94 | 116 | 58.0 | 4.64 |

  - The split: **10 slots shared** (8 unpack, 2 for the weight-scale FCVTL) and **13 per position** (8 SDOT, the
    zeroing MOVI, SCVTF at 2, FMUL, FMLA). The unpack is 8 of the stock kernel's 25 slots. The dot products and
    the float step, which no sharing removes, are 13.
  - Every width binds on the total, not on V0 or V1, so the line is straight: c(W) = (10 + 13 W) ÷ 25.
- **M4 timing (*host*, hot, one thread, best of 5):** stock 0.0105 ns per weight; W = 1 to 8 at 0.94, 1.42, 1.93,
  2.43, 3.43, 4.43 × stock. Marginal position 0.42 to 0.49. A core with four vector pipes shows the same split, so
  the X1 will not differ in kind.
- **§9 item 1, linear to 8: yes.** Register pressure does not bind: only the 8 unpacked vectors and one float
  accumulator per position live across a block (16 registers at width 8); the activations and the integer sum are
  temporaries. A 4- or 8-row block does nothing for the per-position cost, which is fixed by 8 SDOT per 128
  weights.
- **Against the device rate.** The A78 measured 0.118 cycles per weight, 15.1 cycles per block, against the floor
  of 12.5 (*device* rate, *estimated* floor). Two readings of the 2.6 cycles in between:
  - they scale with the work: c(2) = 1.44, marginal 0.52 (**my centre**; it equals her 1.45 at width 2 and is
    dearer above);
  - they are slack that a second, independent SDOT chain can fill, so width 2 runs at its own floor:
    c(2) = 18 ÷ 15.1 = **1.19**, and every later position adds 6.5 cycles, 0.43. This is the lower bound.
  - I cannot choose between them from the data we have. R72's K1m miss (the pipe model predicted 0.79, the
    phone gave 0.983) says the A78 is not issue-bound on that kernel, which makes the second reading possible. One
    hot single-thread cell at widths 1 to 4 decides it in seconds (T2 below).

**Re-priced** (her model, only c(W) changed; printed tok/s, Yi / Llama-2 / DeepSeek; *estimated*):

| c(W) | c at W = 2, 3, 4, 8 | kB at C 40: W = 2, 3, 4 | best width alone, C 40 | centre stack, her inputs | centre stack, my inputs | top of every band |
|---|---|---|---|---|---|---|
| Bonnie, 1 + 0.45 (W − 1) | 1.45, 1.90, 2.35, 4.15 | 1.12, 1.41, 1.69 | 2: 7.94 / 7.79 / 7.38 | 2: 9.82 / 9.49 / 9.25 | 2: 9.27 / 8.99 / 8.77 | 2: 11.25 / 10.93 / 10.57 |
| **count, (10 + 13 W) ÷ 25** | 1.44, 1.96, 2.48, 4.64 | 1.12, 1.45, 1.77 | 2: 7.98 / 7.83 / 7.42 | 2: 9.87 / 9.55 / 9.30 | **2: 9.32 / 9.05 / 8.81** | 2: 11.28 / 10.96 / 10.59 |
| lower bound, 1.19 then + 0.43 | 1.19, 1.62, 2.05, 3.77 | 1.03, 1.23, 1.50 | 2: 8.55 / 8.42 / 7.93 | 2: 10.81 / 10.49 / 10.17 | 2: 10.36 / 10.10 / 9.78 | 11.99 / 11.74 (W 3) / 11.24 |
| what-if 0.30 flat | 1.30, 1.60, 1.90, 3.10 | 1.05, 1.22, 1.41 | 2: 8.43 / 8.30 / 7.82 | 2: 10.56 / 10.23 / 9.93 | 2: 10.09 / 9.82 / 9.53 | 11.65 / 11.79 / 11.04 |
| what-if 0.15 flat (**not reachable**) | 1.15, 1.30, 1.45, 2.05 | 1.02, 1.05, 1.12 | 3: 8.96 / 9.29 / 8.45 | 3: 11.19 / 11.42 / 10.72 | 3: 10.71 / 10.97 / 10.30 | 12.47 / 12.61 / 11.78 |

- Per width, alone at C 40, with the count: 7.98 / 7.83 / 7.42 (W 2), 6.83 / 7.00 / 6.51 (W 3), 6.00 / 6.10 / 5.71
  (W 4), 4.77 / 4.85 / 4.59 (W 6), 3.98 / 4.05 / 3.84 (W 8). Widths 6 and 8 are slower than stock, as she says.
- "Compute overtakes the bus at 1.75 positions": **agreed** with the count (1.77 at C 40); 2.3 on the lower bound.
- Even the 0.15 row, which the count rules out, gives no subject at 12 at the centre. **Cheaper positions are
  not what is missing; accepted tokens are.** Width 3 beats width 2 at head top-1 of 0.55 / 0.75 / 0.76 with the
  count (C 50 / 40 / 35) and 0.17 / 0.60 / 0.74 on the lower bound. Tom measured 0.379 and 0.497 (*host*).
- **§9 item 2, the centre C: I hold kB(2) = 1.15, a small split.** The three body times in her script come from
  the `4-7` arms of R72 L4, and those ran with the A78 cap at 1,555 to 1,785 MHz on all three subjects. A batch of
  two priced against those bodies belongs at the same clocks, C about 35, where kB(2) is 1.22. C 40 prices the
  second position at better clocks than the stock time it is divided by. T2 decides; the difference between 1.12
  and 1.15 is 0.15 tok/s.
- A by-product, not a lever: width 1 of this kernel is 0.92 (count) and 0.94 (*host*) of the stock gemv's
  compute, exact. Stock is bus-bound, so this is worth under 1 ms per token.

### E3. Top-k from the Medusa post (her §4, §9 items 3 and 4)

- **The transfer is weak, and the conclusion does not depend on it.** It is one blog sentence, in-domain, for a
  different head, carried by one log-odds step. But sibling shapes need the second-ranked guess to be right in 0.20
  to 0.30 of passes. With top-1 at 0.38 to 0.50, the second rank would have to hold a third to a half of all the
  remaining mass. Her 0.09 could be 0.15 and nothing changes. T7 is a count on existing dumps and settles it.
- One caution for T7: it must rank the **head's** outputs. The top-64 logits stored in Tom's dumps are the model's
  own for that row, not the head's guess for the next one.
- **§9 item 3, a published per-head top-k table: not found.**
  - Medusa: none in the paper (memo-09 §10.5).
  - Hydra (arXiv 2402.05109v2, fetched): no per-head accuracy. Acceptance length only in figures; "up to 0.46
    tokens" over Medusa from making heads depend on the earlier candidates.
  - Beagle (arXiv 2505.24544v4, fetched): per-position acceptance only as figures.
- **§9 item 4, depth 2 for a block-size head: not found.** The only numbers are for a full layer. EAGLE Table 2
  (*published*, MT-bench, temperature 0): Vicuna-7B 0.79, 0.74, 0.72, 0.73, 0.67; **Llama-2-chat-7B 0.76, 0.69,
  0.67, 0.68, 0.68**. Depth 2 given depth 1 is about 0.92 × the first. EAGLE-3 says EAGLE's rate "drops
  significantly" with more self-predicted inputs on Llama-3.1-8B (text; the values are in a figure). Hydra's
  premise is that independent heads lose accuracy with depth. Her band of 0.7 to 1.15 × is fair; I would centre it
  at 0.85 for a second independent head. T9 measures it.

### E4. A guesser between the block and the layer (her §9 item 5)

Nothing found that trains at laptop data scale. Everything published sits at ShareGPT scale or above:

| method | draft | training | result | read |
|---|---|---|---|---|
| EAGLE (2401.15077v3) | FC + one decoder layer, 0.24 B | ShareGPT, about 68,000 dialogues; "an RTX 3090 node in 1 to 2 days" (the repository says 8 × RTX 3090) | Llama-2-chat-7B tau 3.62, first-token 0.76 | fetched |
| Hydra++ (2402.05109v2) | 4-layer MLP heads + one extra decoder layer | ShareGPT, 8 × A100-80GB, 10 epochs | 1.27 × Medusa's throughput at 7B | fetched |
| Beagle (2505.24544v4) | one cross-attention block + MLP, "fewer parameters than EAGLE" (no count) | ShareGPT, over 60,000 dialogues; one 24 GB GPU is enough for a 7B target | tau 4.1 against EAGLE-2's 4.0 (Vicuna-7B, MT-bench) | fetched |
| EAGLE-3 (2503.01840) | one decoder layer, multi-level features | ShareGPT + UltraChat (about 532,000 entries), answers regenerated by the target | gains with data where EAGLE does not (figures only) | fetched |
| SpecForge (2603.18567) | EAGLE-3 trainer | 8 × H200; 1.4 M conversations; tau 2.82 → 3.48 on Math500 from 5 × the data | no Llama-2, Yi or DeepSeek-LLM draft | fetched |
| FastEagle (2509.20416), SpecVocab (2602.13836) | one-pass cascaded drafter; a shortlist for the draft's output head (+8.1 % over EAGLE-3) | – | – | search summary only |

- Tom's head has 36 M parameters and overfits (training top-1 0.78 against 0.33 held out). **Data is his limit,
  not the block's size.** A bigger guesser on the same 0.3 to 0.4 M rows is not the next step.
- A cheaper way to more rows, as T12: teacher-forced dumps of fixed chat text at `-b 512` instead of greedy
  generation. Keep the rows where the text's next token equals the model's greedy token; the label is the next
  row's greedy token. EAGLE reports a fixed dataset costs little against regenerated answers (2.78 × against
  2.88 ×, *published*). Tom's slope is +0.04 (Llama-2) and +0.02 (Yi) per doubling, so two doublings would put
  Llama-2 near 0.57 and Yi near 0.42 (*estimated*). It moves Llama-2 over 0.50 and leaves Yi short.

### E5. The EAGLE-size draft layer (her §6 item 1)

**Phone cost per draft step** (*estimated*; hers is one body layer plus the shortlist):

| part | Yi | Llama-2 | DeepSeek |
|---|---|---|---|
| one decoder layer = body ÷ layers (32 / 32 / 30) | 3.77 | 4.39 | 4.36 |
| the input FC, 2 × 4,096 → 4,096, 33.5 M weights, as Tom's block | 0.73 | 0.73 | 0.73 |
| 8,192-row shortlist of the served head | 1.16 | 1.13 | 1.11 |
| **per draft step** | **5.7** | **6.2** | **6.2 ms** |

- She left out the FC (0.7 ms). Also not priced by either of us: the layer must pass over every committed token's
  true feature to keep its own KV (one batched call per pass, about +0.5 to 1 ms), and a draft file in Q8_0 would
  double the layer's read.
- **Memory.** 0.236 B weights in Q4_0 are 0.13 GB, read on every pass. R69 measured 4.2 GB of file-backed
  capacity (*device*). Llama-2 and DeepSeek would stand at 4.15 and 4.14 GB with it.
- **Re-priced** (chain, depth 2 at 0.92 × the first, d 0.90; best width; printed tok/s):

  | first-token top-1 | c(W) | centre stack, her inputs | centre stack, my inputs | top of every band |
  |---|---|---|---|---|
  | 0.60 | count | 2: 11.12 / 9.89 / 10.01 | 2: 10.51 / 9.38 / 9.50 | 2: 12.65 / 11.31 / 11.35 |
  | 0.75 | count | 2: **12.04** / 10.74 / 10.82 | 2: 11.38 / 10.19 / 10.27 | 3: 13.84 / 12.36 / 12.32 |
  | 0.75 | lower bound | 3: 13.25 / 11.82 / 11.82 | 2: 12.61 / 11.34 / 11.35 | 3: 15.77 / 14.17 / 13.97 |

  Her table is reproduced within 0.1 at the centre; at 0.75 with every band at its top mine is 0.6 to 0.7 lower. **All three pass 12 only with every band at its top**, as she found.
  At the centre Yi alone passes, and only with her inputs or on the lower bound.
- **Does the published acceptance hold for our subjects?**
  - Llama-2-chat-7B is one of EAGLE's own targets: 0.76 first token, tau 3.62 (*published*, fp16 model, MT-bench).
    Official weights exist (`yuhuili/EAGLE-llama2-chat-7B`, 0.24 B). So for one subject the number can be
    **measured** on our Q4_0 file and prompt set without training (T13). I expect 0.65 to 0.75 there (judgement):
    the prompts are Dolly, the model is quantised, and the test is the greedy token.
  - Yi-1.5-6B-Chat and DeepSeek-LLM-7B-chat: no published draft of any kind found (EAGLE's and EAGLE-3's lists
    read; `EAGLE3-DeepSeek-R1-Distill-LLaMA-8B` is a different model). They would have to be trained.
- **Training cost on the M4 Pro (*estimated*):**
  - Data: about 68,000 dialogues. At 1,000 to 2,000 tokens each that is 70 to 140 M positions (my arithmetic; the
    paper gives no token count), 150 to 300 × Tom's largest set.
  - Features: at Tom's 118 to 135 rows per second, 6 to 14 days per model for one pass; stored at 8 KB a row, 0.5
    to 1.1 TB. The laptop was at 0.2 GB free this week.
  - Training: 1 to 2 days on a node of RTX 3090s. If the M4 Pro is a quarter of one 3090 (judgement, not
    measured), that is weeks per model, for two or three models.
  - **Not a laptop job.** A laptop-sized run (1 to 2 M rows) is possible in a day or two. Its accuracy is unknown;
    my guess is 0.45 to 0.60, since Tom's block overfits on that much data already.
- **Is it a "guess head"? My reading: no, it is a draft model**, and it goes to the owner before any work.
  - The owner allowed option 3 of memo-09, recorded as "a small next-token guess head". This is a transformer
    layer with attention and its own KV state, called once per draft depth. It has 3.4 to 3.9 % of the subject's
    parameters (Tom's block: 0.5 %).
  - That is the class campaign 1 measured in R20 (*device*, 0.48 to 0.76 × the plain rate) and that
    `reports/02_methods.md` row 10 lists as lost on the host with EAGLE-3. Bonnie is right that the loss reason
    there (the union of experts) does not carry to a dense model. The classification does.
  - R69's Rule B scored KILL for a drafted second token, with an inside line of a match rate under 0.4 on any
    subject. Tom's Yi head is at 0.379 (*host*), under that line as written. Memo-09 §10.6 asked the lead to write
    the reopening and its line before T2 runs on the phone; that still stands, for the head as well as for a layer.
  - The published Llama-2 weights are a third party's file: usable for a host measurement, but "a different file
    or model is proposed to the owner first" applies to shipping them.
- **I do not propose it either.** P(first-token top-1 of 0.75 or more on all three with training we can afford):
  0.02. And at 0.75 the centre is still 10.2 to 10.8 on the two MHA subjects.

### E6. The gate (her §9 item 6) and smaller points

- **Agreed: a threshold fitted on the training text is a calibration-derived value.** The 2026-11-19 decision
  names thresholds. It belongs in the head's sidecar header with the shas, from the committed script. Output stays
  exact, so the check is greedy identity, not PPL. The lead confirms.
- The gate's gain rests on an assumed Beta distribution. T8 replaces it; until then −2 to −6 ms is a guess.
- §3 (mask, rollback, unified KV): read and agreed. T10 is the right check.
- §5.4, P(width 2 delivers its centre alone) 0.30: I say 0.25, for the clocks argument in E2.

### E7. Verdicts

| item | Bonnie | Ebony | verdict |
|---|---|---|---|
| marginal position, exact kernel | 0.45 (0.30 to 0.60), linear | 0.52 by count, 0.42 to 0.49 *host*; lower bound 0.19 for the first, 0.43 after | **agreed**; the 0.15 hypothesis is closed by the count |
| compute overtakes the bus at | 1.75 positions (C 40) | 1.77; 2.3 on the lower bound | **agreed** |
| centre kB(2) | 1.12 (C 40) | 1.15; bodies were measured at caps | **split**, small; T2 |
| best width | 2 | 2 on every c(W) the count allows | **agreed** |
| top-k transfer | +0.98 log-odds, top-2 gain 0.09 | weak source, conclusion insensitive | **agreed**; T7 |
| depth 2 conditional | 1.0 × top-1 (0.7 to 1.15) | 0.85 × for an independent head; 0.92 × *published* for a layer | **agreed band** |
| draft layer, cost per step | 3.8 to 4.4 ms + shortlist | 5.7 / 6.2 / 6.2 ms | **split**, +1 ms |
| draft layer, rule | the lead's question | a draft model; the owner's question; Rule B's line unwritten | **split** on who decides |
| gate as a calibration value | yes | yes | **agreed** |
| any width or stack at 12 on all three, centre | no | no | **agreed** |
| P(all three at 12) | 0.002 | 0.002 to 0.003 | **agreed** |

### E8. Merged lists for Tom (through the lead)

**Tests** (T7 to T11 are hers, unchanged unless marked +):

| # | test | where | cost | closes when |
|---|---|---|---|---|
| T7 | Top-k of the head on the prompt-set dumps. **+** rank the head's outputs, not the stored model logits | host | a few lines | top-2 − top-1 < 0.20 on both → sibling shapes closed |
| T8 | Confidence histogram and gate thresholds | host | same script | best threshold gains under 1.5 % at kB 1.12 and 1.23 → gate not built |
| T10 | Tree identity at batches of 3, 4, 8 with sibling sequence ids and `seq_rm`. **+** an arm on the W-position kernel once it is in the tree (`ebony_gw.c` is a starting point and is exact on random data only) | host | minutes | any difference → not exact as built |
| T2 | Device kB cell. **+** build the kernel with the width as a parameter. **+** hot single-thread cells on cpu 4 at full clock, widths 1 to 4: this is c(W) itself. **+** 60 s arms at widths 1, 2, 3 on `4-7` at subject shapes, with caps and placement logged | device, model-free | one kernel; about 8 phone-minutes | hot c(2) ≥ 1.35 → the count holds and width 3 is closed whatever the head; hot c(2) ≤ 1.25 → re-price on the lower bound (width 2 gains about 0.6 tok/s, width 3 then needs top-1 0.60 at C 40); sustained kB(2) > 1.20 → her line |
| T9 | Depth-2 head | host | minutes per model | **only if** T2 gives c(2) ≤ 1.25 **or** a head passes 0.55; otherwise width 3 is closed without it |
| T11 | DeepSeek head | host | about 1 hour | replaces an estimate |
| T12 | **+** Teacher-forced dumps of fixed chat text at `-b 512` (E4), two more doublings of pairs, same head | host | rows per second to be measured first; hours | Yi under 0.45 at 1.2 M pairs → Yi's head stays where it is; reports the slope either way |
| T13 | **+** Optional: the published EAGLE layer for Llama-2-chat-7B on Tom's prompt-set dumps (first-token and depth-2 top-1 against the Q4_0 greedy token). Check first that its input feature is the post-norm state Tom stored | host; a download of about 0.5 GB, the lead's call | 1 to 2 hours | under 0.65 → the 0.75 what-if row is dropped. It cannot open the goal: at 0.75 Llama-2 prices 10.2 to 10.7 at the centre |

**Try list** (build only after its test; env-gated, off by default, model-generic, two subjects):

| # | lever | after | label | price alone at the centre (Yi / Llama-2 / DeepSeek) | P |
|---|---|---|---|---|---|
| Y1 + Y5 | Width 2 with the trained head and the W-position kernel | T2, T10; the lead's written Rule B reopening | bit-identical output | Bonnie (C 40): −17.9 / −31.1 / −24.0 ms → 7.94 / 7.79 / 7.38. Ebony (C 38.6): −16.2 / −29.2 / −22.1 → 7.82 / 7.67 / 7.27; at the caps (C 35): −9.3 / −21.8 / −14.9 → 7.40 / 7.24 / 6.90 | 0.30 / 0.25 |
| Y6 | Confidence gate on Y1 | T8 | bit-identical output; threshold in the sidecar | −2 to −6 ms (hers; the distribution is assumed) | 0.5 given Y1 |
| Y7 | Width 3, chain of 2 | T2 at c(2) ≤ 1.25, then T9 | bit-identical output | a loss of 14 to 20 ms against Y1 with the count (hers: 11 to 16) | 0.05; 0.03 mine |
| Y8 | Widths 4 to 8 | not proposed | – | a loss at every c(W) the count allows | under 0.01 |
| Y9 | A55 cores for the second position | Y1 on the device | bit-identical output | −5 ms at best | 0.05 |
| Y10 | Draft layer (EAGLE-size) | **not proposed**; an owner question first (E5) | bit-identical output; a 0.13 GB sidecar | what-if at top-1 0.75: centre stack 12.0 / 10.7 / 10.8 (her inputs), 11.4 / 10.2 / 10.3 (mine) | 0.02 that 0.75 is reached on all three |

Order: T7, T8, T10 now, as she proposes. T12 before T9 and T11 (it changes what the heads are). T2's hot cells
ride in the kit that carries the kB cell. Nothing changes the phone order of memo-09 §10.10.

### E9. Sources of this review (read 2026-11-22) and limits

- EAGLE, arXiv 2401.15077v3 (fetched; Tables 2, 5, 6): <https://arxiv.org/html/2401.15077v3>
- EAGLE-3, arXiv 2503.01840 (fetched; acceptance and data-scale values are in figures only): <https://arxiv.org/html/2503.01840>
- EAGLE repository (fetched; weight lists, "within 1-2 days ... on 8x RTX 3090"): <https://github.com/SafeAILab/EAGLE>
- Hydra, arXiv 2402.05109v2 (fetched): <https://arxiv.org/html/2402.05109v2>
- Beagle, arXiv 2505.24544v4 (fetched): <https://arxiv.org/html/2505.24544v4>
- SpecForge, arXiv 2603.18567 (fetched): <https://arxiv.org/html/2603.18567>
- FastEagle, arXiv 2509.20416; SpecVocab, arXiv 2602.13836 (search summaries only).
- Repository: `lab/llama.cpp-base/ggml/src/ggml-cpu/arch/arm/repack.cpp` lines 230 to 289 (the stock gemv),
  `tools/c3_a78_pipes.py`, memo-08-work `tom-k1.md` §3 and `bonnie-path.md` §10.2, R72-result.md, R69 amendment 1.

Limits: the web pages were read through a summarising fetch, not line by line. The A78 numbers are an issue
count, not a simulation or a measurement; the SDOT chain latency on the A78 was not looked up. The M4 timing is
one run on a busy laptop. The kernel is a scratch file, checked on random data, not in the tree. No device run.
