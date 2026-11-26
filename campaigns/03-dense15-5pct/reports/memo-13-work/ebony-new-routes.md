# c3 memo-13 work (Ebony): is there another way to compute a decode step? Unpriced routes, host screens, bus relief

- **Date:** 2026-11-26. **Author:** ebony, `claude-fable-5-2`. Not committed. No verdict here.
  **Bonnie has not reviewed this file.** Her part is [bonnie-cpu-ladder.md](bonnie-cpu-ladder.md); I read its
  "Answer first" and §5 (cost model) and did not wait for the rest.
- **Labels.** *device* (phone, quoted from R69 / R72 / R74), *host* (this laptop, Apple M4 Pro; perplexity ratios
  from my own numpy forward pass, a screen, **not** `llama-perplexity`), *estimated* (arithmetic), *external*
  (another group's claim, with how much of it I read), judgement. Triples are Yi / Llama-2 / DeepSeek.
  Goal: 12 tok/s printed = 79.2 ms per token.
- **Scripts (work files, this folder):**
  - [ebony_seqsel.py](ebony_seqsel.py): a float32 numpy forward pass of a llama-architecture GGUF (weights
    dequantised from the served file), `llama-perplexity`'s protocol (chunks of 512, first 256 are context, 255
    scored). Self-check against `llama-perplexity` on the same chunks, running PPL: Llama-2 5.7928 / 6.3326 /
    7.0048 / 7.9895 against 5.8104 / 6.3386 / 7.0099 / 7.9916 (−0.02 to −0.30 %); Yi 5.0138 / 6.9578 / 7.3954
    against 5.0145 / 6.9742 / 7.4150; DeepSeek 5.1101 / 6.9326 / 6.5785 against 5.1429 / 6.9611 / 6.5990
    (−0.3 to −0.6 %). The gap is the float activations (llama.cpp quantises them to 8 bits). Every ratio below
    is against this script's own dense run on the same chunks.
  - [ebony_routes.py](ebony_routes.py): prices; loads `memo-11-work/bonnie_bytes.py` unchanged (self-check: stock
    6.90 / 6.21 / 6.22). C 40, O 8, packed-layout baseline unless said.
- **What the screen is and is not.** 5 to 8 chunks (1,275 to 2,040 scored tokens) per subject, two runs for the surviving rules; the standard error
  of a ratio is 0.4 to 1.1 % for the small effects and is printed by the script. It decides "far outside" and
  "plausibly inside". It does not replace the gate (full set, `llama-perplexity`, `-b 512` and `-ub 1`).
- **Baseline caveat (Bonnie's Answer 1).** The owner's adopted body path is choice A (plain mapped rows); the
  prices here are differences in ms on the packed-layout model (stock 137.6 / 153.0 / 152.7 ms). I say per item
  whether it works in place on path A or needs a derived copy.

## Answer first

1. **No. Nothing found reaches 12 tok/s on any subject, and nothing changes the model of cost by the factor
   needed.** I took the owner's question ("we may need to rethink the computation scheme itself") as the main
   axis, tested five schemes against the three served files on the host, and searched 2025 to 2026 work. Four
   schemes are closed by measurements made today; one survives as a partial bus relief.
2. **Computing across tokens is closed on all three subjects (*host*).**
   - Consecutive tokens' activations are not close: the squared change of the FFN input between token t−1 and
     t is 1.21 / 1.20 / 1.24 × the squared input itself (mean over layers; no layer under 0.72), and 1.58 /
     1.46 / 1.46 at the `down` input. A delta has more energy than the thing it replaces. Even the nearest
     earlier position in a 256-token context is 0.78 / 0.85 away (Yi / Llama-2; DeepSeek not run).
   - Reusing the previous token's layer output (LoRA-Drop without its trained LoRA) in 8 middle layers on every
     second token multiplies perplexity by 21 / 17 / 10. Plainly skipping those layers is ×1.7 / ×2.2 / ×1.6;
     reuse is worse than skipping.
   - Choosing the FFN channels once per sequence from the prompt (GRIFFIN, GLASS, DART; none in our memos):
     dropping 12.5 % of the channels costs +9.9 / +5.5 / +9.8 %, and 25 % costs +31 / +20 / +20 %. A fixed
     corpus-wide set does the same (+29 / +21 / +18 % at 25 %). There is no per-sequence structure to use at
     our budget.
3. **The models do tolerate dropping half of the FFN channels per token, if the mask is known:** by |h|
   (the exact product), 50 % costs +1.8 / +1.1 / +1.1 % and 65 % costs +5.1 / +5.0 / +3.7 %. This is the first
   time the element rule is measured on the three subjects (memo-05's E1 was Mistral and Qwen2.5).
   **The mask is the problem.** A 1-bit copy of `gate` and `up` as the ranker (the Prox design of 2026-07,
   not in our memos) fails: +37 / +13 / +8 % at 50 %. A 2-bit copy passes on DeepSeek only (+16 / +5.4 /
   +1.8 % at 50 %) and costs as many bytes and more arithmetic than it saves on a 4-bit file. **Two-stage FFN
   is closed.**
4. **What survives, with evidence on all three (*host* screen):** a proxy-free FFN rule in two steps.
   Compute `up` first and skip the 30 % of channels with the smallest |up| (their `gate` rows: exact rows, no
   ranker, no copy): +0.5 to +0.8 / +1.9 to +2.1 / +1.3 to +1.5 %. Then, with h known, send only the top half
   of the channels by |h| through `down`. Both steps together ("UD"): **+2.6 / +2.1 / +2.0 %** (6 chunks, s.e.
   0.7 to 1.2). Price (*estimated*): rows alone −8.9 / −8.9 / −8.3 ms per token, in place on the owner's path;
   with the `down` columns through the column kernel K1 at its *device* cost **−21.9 / −21.8 / −20.3 ms
   (8.21 / 7.24 / 7.18 tok/s alone)**. It stays bus-bound because the body's arithmetic does not rise (k 0.96
   to 0.97) while its bytes fall to 0.79 / 0.82 / 0.82.
5. **One new bit-identical item:** the Q4_0 block scales. A body tensor holds at most 1,949 distinct fp16 scale
   values (*host*, every fourth layer; median about 900), so a 12-bit index is lossless: −2.8 % of the body's
   bytes, −2.2 / −2.6 / −2.5 ms. A one-byte code is closed (single blocks off by 0.7 to 2.0 ×; PPL × 16 on Yi,
   *host*). With UD: **−23.5 / −23.9 / −22.2 ms**, about a fifth of the bus term. That is the closing section's
   answer to "ease the bus, not remove it".
6. **Rank of the routes** (§5). 1: UD (rows first, then columns). 2: lossless scale index. 3: nothing else is
   open. Closed today: delta, layer reuse, per-sequence channels, two-stage FFN by a low-bit ranker, activation
   low rank, one-byte scales. Closed by fact: SVE / SME / i8mm on this SoC (the A78 and X1 are Armv8.2 with
   dotprod only; *external*, GCC's core table through a search summary).
7. **The assumption I would challenge:** "R72 parked the column kernel, so no column rule is on the ladder."
   R72 parked K1 as a **uniform rule on every matrix** at keep 0.58 to 0.65 (R72 L4), where the body's arithmetic
   rises by 20 to 40 % and the token turns compute-bound. On `down` alone at keep 0.5, with a mask that is exact and
   free once `gate` and `up` exist, the body's arithmetic is × 1.01 and its bytes × 0.87 to 0.89: −13.0 /
   −13.0 / −12.0 ms at the very cost R72 measured (2.1 per touched weight). That cell was never priced on the
   three subjects (memo-07's E1 row was Mistral and Qwen2.5, and its perplexity cell for the subjects was a
   pending task). Second, smaller: an FFN row rule does not need groups of four or a ranker (memo-07 C4 / C5;
   r14 "rests on nothing"): per element, |up| alone is enough for 30 %.
8. P(all three at 12 with anything in this memo): under 0.002 (judgement), unchanged. With C7 and O 5.2 the
   new items reach 8.90 / 7.65 / 7.87 tok/s (*estimated*, memo-11's model; they do not multiply with width 2).

## 1. What was already covered (not re-scouted)

memo-09 (lookahead, heads, Hexagon), memo-10 (tree verify, W-position kernel, draft layer), memo-11-bytes
(low-bit re-encode, LUT / TBL kernels, layer and FFN skip, SVD, lossless nibbles), memo-11-exotic (retrieval,
layer-skip self-drafts, prefix-exit ring, DDR steps, huge pages, delta sparsity on the final state only), my
r16 (upstream n-gram speculation, nibble entropy 3.77 bits, R-Sparse, SpecBound, MTP), memo-02 / 05 / 07 (column
sparsity, K1, groups of four, sublayer skip), campaign 2 memo-21 (XOR / popcount, VQ, cross-token reuse "x changes
every token", closed there without a number), docs/negative-results.md. Where I reopen one of these, the reason
is stated in its row.

## 2. The model of cost, and the five schemes

A decode step costs max(bytes ÷ R, weight-touches × k ÷ C) plus about a quarter of the smaller term, per pass,
divided by tokens per pass (memo-10's reader; Bonnie §5: bus term 104 / 122 / 114 ms, and a scheme must cut
weight-touches per committed token to 0.54 / 0.48 / 0.45 in bytes and arithmetic at once). A different way to
compute a step has to do one of five things. Each row: what would change in the cost model, what I measured or
read, and its state.

| # | scheme | what changes in the cost model | evidence today | price against 79.2 ms | state |
|---|---|---|---|---|---|
| S1 | **Across tokens: delta.** y_t = y_(t−1) + W (x_t − x_(t−1)), or a delta against any earlier position | weight-touches fall to the share of inputs that moved; bytes too with a column layout | *host*, all layers, context half: squared change ÷ squared norm 1.21 / 1.20 / 1.24 at the FFN input (range over layers 0.72 to 1.72) and 1.58 / 1.46 / 1.46 at the `down` input. Nearest earlier position (3 chunks): 0.78 / 0.85 at the FFN input, 0.89 / 0.86 at the `down` input | none: the delta is a larger input than the original | **closed (*host*)**. Memo-11-exotic had the final layer only; Bonnie §5.3 asked for this count |
| S2 | **Across tokens: reuse a layer's output** from the previous token (LoRA-Drop, arXiv 2601.02569, 2026-01: "up to 2.6 ×") | passes per token fall for the reused layers: 8 of 32 layers on every second token is −12.5 % of bytes and arithmetic | *host*: ×21.1 / ×16.5 / ×10.2 perplexity; plain skip of the same layers ×1.70 / ×2.22 / ×1.57. *External* (read): the paper's gain rests on a LoRA trained on about 15 B tokens with the base frozen; no result without it; average task scores, no perplexity | −15 / −18 / −16 ms if it were free | **closed (*host*)** without training; with the LoRA it is a trained change of numerics at a scale we cannot run: not proposed |
| S3 | **Across tokens: one channel set per sequence**, chosen from the prompt (GRIFFIN, ICML-W 2024; GLASS, arXiv 2508.14302; DART, arXiv 2601.22632) | true skipping with no ranker and no column kernel: the kept rows and columns could be packed once per prompt; b and k fall together | *host*: 12.5 % of channels +9.9 / +5.5 / +9.8 %; 25 % +30.8 / +19.7 / +19.9 %; 50 % ×2.28 (Llama-2). Statistic from the last 64 prompt tokens: +26 % at 25 % (Llama-2). A corpus-wide fixed set: +29.1 / +20.9 / +18.0 % at 25 %. *External* (read): GRIFFIN reports Rouge and F1 at 50 %, no perplexity; GLASS's own table has GRIFFIN at KL 0.65 to 0.94 | at the tolerated share (under 10 %): under −7 ms | **closed (*host*)** on all three |
| S4 | **Another arithmetic domain** (tables, bit-serial, integer decompositions, index structures such as RSR, arXiv 2411.06360 / 2603.27462) | k falls; bytes do not | Closed by Tom's U2 for 2- and 3-bit tables and by Bonnie §5.3 for the same bytes (zero arithmetic gives 7.99 / 7.10 / 7.05). What is different here: I tested the one place where an inexact domain could pay, as the **ranker** of a two-stage FFN (S5). RSR (*external*, search summary) is for binary and ternary matrices and its baselines are unvectorised; on four bit-planes it is one scalar load and add per 3 weights against `sdot`'s 16 weights per instruction | – | **closed by arithmetic**, no test |
| S5 | **Two-stage FFN** (Prox, arXiv 2607.27591, 2026-07, read to 100,000 characters): rank the channels with a cheap copy of `gate` and `up`, then run only the kept channels exactly | bytes(FFN) = 2/3 × proxy + (1 − skip); arithmetic = proxy + kept rows + kept `down` columns | *host*, oracle mask by \|h\|: 50 % +1.8 / +1.1 / +1.1 %; 65 % +5.1 / +5.0 / +3.7 %. 1-bit ranker: 50 % +37.2 / +12.8 / +8.1 %; 65 % +22.7 % (Llama-2). 2-bit ranker: 50 % +16.1 / +5.4 / +1.8 %; 65 % +24.6 / +9.3 / +6.2 %. 1-bit weights with the input cut to 7 levels (the popcount form): ×1.69 at 50 % (Llama-2). *External*: Prox reports Qwen3-8B wikitext 8.2 → 8.6 at 60 % with an INT4 ranker beside FP16 weights, mask overlap 82 % | with a 1-bit popcount ranker at skip 0.65, had it held: 94.4 / 109.9 / 112.6 ms (10.06 / 8.64 / 8.44). With the 2-bit ranker at 0.50: 134.2 / 146.3 / 146.5 ms, and it passes on DeepSeek only | **closed (*host*)**: the ranker that keeps the order needs 2 bits or more, and on a file that is already 4-bit that is half the bytes of what it saves and more arithmetic. Does not generalise (Yi fails) |
| S6 | **Runtime-derived low rank / factored path** | weight-touches fall to rank ÷ width | *host*: the FFN input has no low-rank part. Mean direction plus the top 256 principal directions (fitted on half the chunks, scored on the rest) hold 0.33 to 0.42 / 0.31 to 0.34 / 0.33 to 0.37 of its energy past layer 0 (layer 0: 0.51 / 0.62 / 0.82, mostly the mean). *External*: SVD-LLM V2 +25 % at 20 % fewer parameters (memo-11-bytes) | under −6 ms inside the budget (Bonnie §5.3) | **closed**; the activation spectrum is new and agrees |
| S7 | **One read for several tokens** beyond width 2 | passes per token | Token t+1's first input is token t's output through every layer, so only guessed tokens can share a read (Bonnie §5.3, memo-10). Nothing in 2025 to 2026 changes the guesser: UNISPEC (ACL 2026) and SSSD are n-gram drafts, SDFP (arXiv 2602.05499) is a layer-pruned self-draft (*external*, search summaries) | – | closed (memo-10, memo-11-exotic) |

**What S5 teaches, and what is left of it.** The tolerance exists (half the channels per token); the knowledge of
which half costs `gate` and `up`. The cheapest piece of that knowledge is one of the two matrices in full:

| rule (*host*, ratio to dense) | Yi (5 chunks) | Llama-2 (8) | DeepSeek (5) |
|---|---|---|---|
| U30: `up` first, drop the 30 % smallest \|up\| | 1.005 ± 0.011 | 1.019 ± 0.008 | 1.013 ± 0.005 |
| U50 | – | 1.076 | – |
| T30: `gate` first, drop the 30 % smallest \|SiLU(gate)\| (CATS's rule) | 1.033 | 1.012 | 1.010 |
| T40 | 1.071 | 1.041 | 1.034 |
| T50 | 1.126 | 1.093 | 1.088 |
| D50: both first, drop the half with the smallest \|h\| (`down` columns only) | 1.018 ± 0.005 | 1.011 ± 0.007 | 1.011 ± 0.008 |
| D65 | 1.051 | 1.050 | 1.037 |
| second run, 6 chunks each: U30 | 1.008 ± 0.010 | 1.021 ± 0.011 | 1.015 ± 0.005 |
| second run: D50 | 1.020 ± 0.004 | 1.012 ± 0.008 | 1.012 ± 0.007 |
| **second run: UD = U30, then keep the top half of all channels by \|h\| for `down`** | **1.026 ± 0.009** | **1.021 ± 0.012** | **1.020 ± 0.007** |

- U30 is the rule that holds on all three (worst +2.1 %). T30 is better on the two MHA subjects and worse on Yi.
  The rule is per element, not per group of four.
- UD costs little more than its dearer half: the channels U30 drops are mostly among the half D50 drops.
- *External* (search summary): CATS claims "about 99 % of base performance at 50 %" on Mistral-7B and
  Llama-2-7B; on our gate its rule costs +9 to +13 % at 50 %. Task scores are not our budget.

## 3. Prices of what survives (*estimated*; `ebony_routes.py`; C 40; ms per token and printed tok/s)

Kernel inputs: exact rows taken four at a time from any rows, 1.15 × the packed gemv (judgement; unbuilt);
column kernel K1 2.1 per touched weight (*device*, R72 L1: 1.992 hot, 2.239 cold, A78); FFN share of the body
bytes 0.782 / 0.668 / 0.668 (*host*, tensor table).

| lever | body bytes b / arithmetic k | Yi | Llama-2 | DeepSeek |
|---|---|---|---|---|
| stock | 1 / 1 | 137.6, 6.90 | 153.0, 6.21 | 152.7, 6.22 |
| R1: up-first rule 0.30, `gate` rows only | 0.92 / 0.95 (Yi), 0.93 / 0.96 | 128.7, 7.38 (−8.9) | 144.1, 6.59 (−8.9) | 144.4, 6.58 (−8.3) |
| R3: D50, `down` at half its columns through K1 | 0.87 / 1.01, 0.89 / 1.01 | 124.6, 7.62 (−13.0) | 140.0, 6.78 (−13.0) | 140.6, 6.76 (−12.0) |
| **UD = R1 + R3** | 0.79 / 0.96, 0.82 / 0.97 | **115.7, 8.21 (−21.9)** | **131.2, 7.24 (−21.8)** | **132.4, 7.18 (−20.3)** |
| UD at C 33 (the 1,555 MHz cap) | | 122.2 (−15.4) | 136.5 (−16.5) | 137.3 (−15.4) |
| R2: lossless 12-bit scale index | 0.972 / 1.03 | 135.4 (−2.2) | 150.4 (−2.6) | 150.2 (−2.5) |
| **UD + R2** | | **114.1, 8.33 (−23.5)** | **129.1, 7.36 (−23.9)** | **130.5, 7.28 (−22.2)** |
| UD + R2 with C7 and O 5.2 | | 106.7, 8.90 | 124.1, 7.65 | 120.7, 7.87 |
| for scale: no arithmetic at all on stock bytes | 1 / 0 | 121.1, 7.85 | 134.1, 7.08 | 136.3, 6.97 |
| for scale: a free, perfect ranker at 65 %, rows only, C7, O 5.2 | | 69.0, 13.76 | 86.9, 10.93 | 86.1, 11.03 |

- UD is worth more than removing **all** arithmetic from the stock path. It is still a third of the gap.
- At capped clocks (C 33) UD keeps two thirds of its price; R2's extra table read then costs what it saves
  (UD + R2 at C 33: 124.3 / 138.8 / 139.5), so R2 is a full-clock item.
- The last row is the reason no FFN scheme can carry the two MHA subjects: attention is a third of their body
  and no rule on file touches it inside the budget (uniform columns 20 %, memo-06).
- These levers do not multiply with width 2: a second position has its own masks, so the shared read shrinks
  and K1 runs once per position. I did not price that stack; Bonnie's ladder should, with her O and
  second-position inputs.
- One budget: UD spends +2.0 to +2.6 % of the +5 % on the screen, before the KleidiAI stock shift (+0.5 / +0.07 /
  +0.69 %, Tom) and anything else. U30 alone spends +0.5 to +2.1 %.
- Bytes on the phone: K1 needs `down` column-major, 0.81 / 0.81 / 0.76 GB as a derived copy (side file or
  anonymous); the original `down` pages are then never read, so residency does not grow. R1 needs nothing.

## 4. The frontier search (2026-11-26), beyond the rows above

| item | source, and how much I read | on our phone |
|---|---|---|
| Prox | arXiv 2607.27591 (HTML, first 100,000 characters) | S5; closed on our files |
| GRIFFIN, GLASS, DART, TDA | arXiv 2404.01365 (read), 2508.14302 (read), 2601.22632 and 2408.11393 (search summaries) | S3; closed |
| LoRA-Drop | arXiv 2601.02569 (HTML read; its tables disagree with each other on speed) | S2; closed without training |
| DeltaLLM | arXiv 2507.19608 (read): delta coding of attention keys only, 1 B and 2 B models, no speed measured, no linear-layer bytes | nothing for a body bound by weight bytes; S1 closes the general form |
| RSR, RSR-core, SSR | arXiv 2411.06360, 2603.27462, DATE 2026 (search summary) | S4; binary and ternary only |
| SVD predictors for gated FFNs | arXiv 2603.14110 (search summary): for ReLU-gated models, with calibrated biases | our subjects are SiLU; S6's spectrum says a low-rank ranker of the input sees under half its energy |
| UNISPEC, SSSD, SDFP, LinguaSpec | ACL 2026, arXiv 2602.05499 (search summaries) | S7; n-gram and self-draft families already priced |
| Arm's "llama-roofline" learning path | learn.arm.com (search summary): decode tok/s = bandwidth ÷ bytes per token; a Raspberry Pi 5 at 13.8 GB/s | the same cost model, stated by the vendor; no method |
| Arm SME2 / SVE / i8mm paths in llama.cpp and KleidiAI | GCC core table (search summary): Cortex-X1 is Armv8.2-A with fp16, rcpc, dotprod; the A78 the same class | **not applicable**: no SVE, SME, i8mm or bf16 on the Snapdragon 888. The dotprod path is what we run. I did not find the phone's own hwcap line in a kit log; one line in the next kit would confirm |
| ik_llama.cpp | discussion 164 (search summary): mainline's Q4_0 on NEON is called competitive; its gains are in other types and in batched prefill | no decode lever for Q4_0; it is a second source for a multi-position kernel if Y1 is ever built |
| KV and attention at short context | r15, Bonnie §1 (5 / 7 / 7 ms) | nothing new found; DeltaLLM is the only 2025 to 2026 item and it has no measurement |
| phone runtimes (MNN, MLC, ExecuTorch, PowerInfer-2) | one search, no 7B CPU decode number newer than memo-11-exotic's (8.7 tok/s, llama.cpp CPU, Qwen2.5-7B, Snapdragon 8 Gen 3) | nothing new |

Searched and not found: a training-free cross-token method with a perplexity number; a ranker for SiLU-gated
FFNs cheaper than 2 bits per weight; any dense 7B CPU decode on a Snapdragon 888 with a current runtime.

## 5. Ranked routes, with the cheapest host test that kills each

| rank | route | file unchanged? budget | price alone (ms; Yi / Llama-2 / DeepSeek) | cheapest kill test (host) | P it delivers its price on the phone |
|---|---|---|---|---|---|
| 1 | **UD**: up-first row rule (30 % of `gate` rows), then `down` at the top half of the channels by \|h\| through K1 | yes; numerics, one budget; the rule is per token, so `-b 512` and `-ub 1` see the same rule | rows −8.9 / −8.9 / −8.3; columns −13.0 / −13.0 / −12.0; both −21.9 / −21.8 / −20.3 | E-T25: the rule in llama.cpp, full set, both settings. Kill: over same-build stock × 1.035 on any subject. Then E-T28 (K1 on the `down` shape at keep 0.5) | rows 0.45; columns 0.3 (K1's cost is *device*; the perplexity and the cold cost on a column-major copy are not) |
| 2 | **R2 lossless scale index** | yes (derived copy); bit-identical | −2.2 / −2.6 / −2.5 | E-T26: count distinct scales in every body tensor (needs ≤ 4,096) | 0.4; needs a packed or side copy, which path A does not have; nothing at capped clocks |
| – | two-stage FFN, per-sequence channels, layer reuse, delta, low rank, one-byte scales | – | – | done (§2, §7) | closed |

## 6. The assumption I challenge

**"R72 parked the column kernel, so the ladder carries no column rule."** (R72 rule A; Bonnie §5.3: "the kernel
that skips them costs what it saves".)

- What R72 parked: K1 as a uniform rule on every matrix of a layer, keep 0.58 to 0.65 (L4), 2.0 to 2.2 per
  touched weight. Arithmetic 1.2 to 1.4 × stock, so the layer turned compute-bound: t_rule ÷ t_stock 0.95 to
  0.99 (*device*).
- What was not priced: K1 on `down` alone. `down` is 26 / 22 / 22 % of the body's bytes; at keep 0.5 and 2.1
  per touched weight it costs 1.05 of its stock arithmetic, so the body's arithmetic is × 1.01 and its bytes
  × 0.87 / 0.89 / 0.89. The token stays bus-bound: −13.0 / −13.0 / −12.0 ms (*estimated*; −8.9 / −10.1 / −9.4
  at C 33).
- Why keep 0.5 is available there and nowhere else: the mask for `down` is exact and free (h is computed
  anyway), and the three subjects hold +1.1 to +2.0 % at 50 % (*host*, two runs). Uniform input columns on the
  other matrices hold 40 / 20 / 20 % (memo-06).
- It is memo-05's E1 cell; memo-07 priced it for Mistral and Qwen2.5 and left the subjects' perplexity as a
  task (B2). So this reopens a parked family, and the difference this time is: one matrix, keep 0.5, measured
  perplexity on the three subjects.
- Risk: the *device* K1 figure is for a hot and a cold arena of the probe's shapes; a column-major `down` at
  keep 0.5 skips every second 2 KB column, which the prefetcher may over-read. P 0.3 that the phone gives 70 %
  of the price.

**Second, smaller: "an FFN skip rule is a group-of-four rule with a ranker, worth 7 % of body on no evidence"**
(memo-07 C4 / C5; r14 §3). The group of four is a property of the packed layout; choice A reads plain rows.
Per element, |up| alone ranks well enough for 30 % on all three. C5's size (× 0.93 of body) was about right;
its evidence class and its layout assumption were not.

## 7. Closing section: easing the memory bus (owner, 2026-11-26)

"Devise a new way to ease the bus bottleneck; it need not remove it." Inputs: bus term 104.0 / 121.7 / 114.4 ms
(Bonnie §5.1); CPU alone 29.9 GB/s, of a nominal 51.2 (*device* R72; nominal *external*). Two axes.

### 7.1 Fewer DRAM bytes per committed token (file on disk unchanged)

| # | method | mechanism | bus relief, ms (Yi / Llama-2 / DeepSeek) | arithmetic cost | evidence, and the cheapest kill test |
|---|---|---|---|---|---|
| B1 | **Up-first row rule** (R1) | `up` in full; the 30 % of channels with the smallest \|up\| skip their `gate` row. In place on path A: no copy | −8.2 / −8.2 / −7.7 of bus term; −8.9 / −8.9 / −8.3 per token | falls too (k 0.95) if a four-row kernel on arbitrary rows runs near the packed one | *host* screen on all three (§2). Kill: E-T25, E-T27 |
| B2 | **`down` at the top half of its columns** (R3, D50), mask exact and free | a column-major copy of `down` (0.81 / 0.81 / 0.76 GB, side file or anonymous; the original `down` pages are then never read) | −13.5 / −13.6 / −12.7 of bus term; −13.0 / −13.0 / −12.0 per token | body × 1.01 (K1 at 2.1, *device*) | *host*: +1.8 to +2.0 / +1.1 to +1.2 / +1.1 to +1.2 %. Kill: E-T25's D cell; E-T28. With B1 (UD): −21.7 / −21.7 / −20.4 of bus term, +2.6 / +2.1 / +2.0 % |
| B3 | **Lossless scale index** | Q4_0 spends 2 of every 18 bytes on an fp16 scale. A body tensor has at most 1,949 distinct values (median about 900; entropy 8.5 bits per tensor on average, 10.1 at most; *host*, every fourth layer, 43 to 51 M blocks). A 12-bit index into a per-tensor table (two blocks share three bytes) is bit-identical; decode is one L1 table read per block | −2.9 / −3.4 / −3.2 of bus term; −2.2 / −2.6 / −2.5 per token | about +3 % (one more load per 32 weights): nothing left at capped clocks | new; nibbles were counted in r16, scales were not. A one-byte code is **closed**: a sign bit and 127 Lloyd levels of the log scale has 1 % rms error but single blocks off by 0.7 to 2.0 ×, and Yi's PPL goes 8.34 → 134.8 (*host*, 8 chunks). Kill: E-T26. Needs a derived copy: not on path A as adopted |
| B4 | lossless coding of the nibbles | 3.77 bits of 4 (r16) | −5 at most | a decoder at 13 G symbols/s per core does not exist | closed (r16) |
| B5 | **what stays in cache across tokens** | L2 0.5 MB × 3 + 1 MB, L3 4 MB: 6.5 MB against 3.1 to 3.6 GB per token; the 3 MB system cache is not reachable from the CPU (*external*, AnandTech, campaign 2 memo-21) | under −0.25 | – | closed by arithmetic. Small exact tables (layer-1 q, k, v by token id) are memo-11-exotic's −1 ms |
| B6 | `down` at 35 % of its columns (D65) | the same mask, deeper | a further −4 of bus term | – | *host*: +5.1 / +5.0 / +3.7 %: takes the whole budget; not proposed |
| B7 | a cheap ranker for 50 to 65 % | S5 | −25 to −40 if it existed | – | closed (*host*): 1-bit fails, 2-bit pays nothing and fails on Yi |
| B8 | lower-bit in-memory forms inside +5 % | memo-11-bytes: 3.3 to 3.9 bpw, kernel dearer than `sdot` (R72 L5, Tom U2) | – | – | closed there; nothing new |
| B9 | an exact predictor for skipped rows | SiLU has no zero and attention has no exact bound; the only exact screen is the head's (C7, on file) | – | – | none exists |

### 7.2 More of the bus actually used

*Device* rows: read on three A78 28.79 GB/s; packed gemv on three A78 30.09; X1 + three A78 28.56 (R69, reduced
clocks); read 29.90 and ggml cold 26.67 on 4-7 (R72); three A78 + four A55 **24.75** (R69); GPU alone 30.86
(R69) to 31.1; CPU + GPU 46.5 (R74). So the DRAM delivers 46.5 GB/s to two masters and about 30 to either one.

| # | method | what the rows say | relief | test |
|---|---|---|---|---|
| U1 | more readers on the CPU side (fourth core, small cores, more streams) | r4 ≈ r3; adding the A55s lowers the sum to 24.75 | none; negative with the small cores | done (*device*) |
| U2 | close the gap between the kernel and the port (26.67 → 29.90 GB/s) | the body is 120.6 / 140.6 / 130.8 ms against a bus term of 104.0 / 121.7 / 114.4: the difference is arithmetic sharing the time, not unused bus | bound: −16.5 / −18.9 / −16.4 per token with no arithmetic at all | not a lever by itself; R1's k < 1 buys a little of it |
| U3 | bus time during the phases that read no weights (attention, KV, sampling: O 5 to 7 ms; about 32 gaps per token) | a helper could pull the next op's first bytes while attention runs, but only into L2 / L3: at most 4 MB per gap, 128 MB per token, 3.7 % of the bytes | −4 at best; campaign 2's N3 priced the same idea at 0 to −3 | device only; P 0.2. Not proposed |
| U4 | a second master as a mover (GPU or DSP fetches, CPU computes) | the CPU cannot read what another master fetched without a second DRAM read; the system cache is not CPU-reachable | none | closed by fact |
| U5 | page size | choice A is a file mapping: 4 KB pages, the prefetcher restarts at each page. Transparent huge pages apply only to an anonymous derived copy (`madvise`), and only if the phone's kernel allows it; the state was never read (rider T6 of memo-09) | unknown; the room is inside U2's gap, so a few ms at most | device read-out of one sysfs line in any future kit; P 0.15 for −3 ms |
| U6 | prefetch distance, stream count, chunking | campaign 1 closed software prefetch (R23) and finer chunks (R33); R69's multi-stream gains were 0.92 and 0.86 | none | done |
| U7 | DRAM row / bank locality | physical placement of page-cache pages is not under an app's control; huge pages are the only handle (U5) | – | – |
| U8 | the bus vote (DDR frequency) | r19: on paper the vote follows traffic and should be at its top step | P 0.15 that it is not (memo-11-exotic) | rider T6 |

**Ranking of the bus-relief methods:** B1 (free, in place, screened) > B2 (the largest; needs K1 on a
column-major `down`) > B3 (bit-identical, small, needs a copy, full clocks only) > U5 (a read-out) > the rest
(closed or bound). **Total honest relief: −23.5 / −23.9 / −22.2 ms per token with B1 + B2 + B3 (about a fifth of
the bus term, 8.33 / 7.36 / 7.28 tok/s alone); −8.9 / −8.9 / −8.3 ms with B1 alone on the owner's path as
adopted.** Nothing on the second axis is a lever: the CPU's port, not its access pattern, sets the 30 GB/s.

## 8. Lists for Tom (through the lead)

### 8.1 Test list (all host)

| # | test | cost | closes when |
|---|---|---|---|
| E-T25 | **UD in llama.cpp's screen build** (the tool that made T1). Env-gated numerics on the FFN: (a) per token, zero h for the 30 % of channels with the smallest \|up\| (also 20 %, 40 %); (b) keep only the top half of the channels by \|h\| before `down` (also 40 %, 60 % dropped); (c) both. 40 chunks at `-b 512` and `-ub 1`, then the full set for the chosen cell; all three subjects; repack and KleidiAI builds | a few lines in the existing screen patch; about 14 PPL runs per subject | (c) over same-build stock × 1.035 on any subject at either setting → fall back to (a) or (b) alone; (a) over × 1.03 at 30 % → 20 % or close. My screen predicts (c) +2.6 / +2.1 / +2.0 %, (a) +0.5 to +2.1 %, (b) +1.1 to +2.0 % |
| E-T26 | **Scale index.** Count distinct fp16 scales in every Q4_0 body tensor of the three files | a script, minutes | any tensor over 4,096 → the 12-bit form needs an escape or is dropped. I counted every fourth layer: max 1,949 |
| E-T27 | **Arbitrary-row four-row kernel.** Host bench of a Q4_0 gemv that takes four row pointers (plain rows, and a per-row packed form) against the packed 4x4 gemv, hot and cold, `up` shape, with 30 % of rows skipped by a mask | one scratch kernel (Bonnie's Y1b is the starting point, `patches/llama.cpp-c3/0002-d-rows.patch`) | over 1.3 × per touched weight cold → R1's price falls to about −4 ms and it is not worth a build |
| E-T28 | **K1 on the `down` shape at keep 0.5**, column-major copy, kept columns chosen by a real mask (every dumped h from E-T25, not a random or a contiguous set), hot and cold, host | Tom's K1 bench, one new cell | over 1.6 per touched weight hot on the M4 (its K1 was 1.53 to 1.64 at TR 128, memo-05) or cold ÷ hot over 1.3 → B2's price is cut and it needs a device cell before any build |
| – | `ebony_seqsel.py` is reusable for any per-token FFN or attention rule: about 4 s per layer per variant at 5 chunks | – | – |

No device test is asked for.

### 8.2 Try list (after its test; env-gated, off by default, model-generic, all three subjects)

| # | lever | after | label | price alone | P |
|---|---|---|---|---|---|
| E-Y12 | Up-first row rule (R1 / B1) in the decode path: `up`, mask, `gate` on kept rows | E-T25 (a), E-T27 | numerics change, one budget, same rule at `-b 512` and `-ub 1` | −8.9 / −8.9 / −8.3 ms | 0.45 |
| E-Y13 | `down` through K1 on a column-major derived copy, top half of the channels by \|h\| (R3 / B2) | E-T25 (b) or (c), E-T28, and a decision on a derived copy of `down` (0.76 to 0.81 GB) | numerics change, the same budget | −13.0 / −13.0 / −12.0 ms; with E-Y12 −21.9 / −21.8 / −20.3 | 0.3 |
| E-Y14 | 12-bit scale index in the packed or side layout (R2 / B3) | E-T26, and a layout that is a copy (choice B or C) | bit-identical | −2.2 / −2.6 / −2.5 ms at full clocks | 0.4 |

## 9. Ranked next questions (the lead decides; I do not continue)

1. E-T25 on the real gate: does UD hold at `-ub 1` and `-b 512` on the full set, on the KleidiAI build? Everything
   in §3 hangs on it.
2. E-T28: what does K1 cost on a column-major `down` when every second column is skipped by a real mask, cold?
   It decides whether B2's −13 ms needs a device cell.
3. Attention on the MHA subjects (a third of the body): a per-token head rule (skip a head's `q` rows and `o`
   columns by an oracle on the head's output norm) can be screened in `ebony_seqsel.py` in one run. Prior that
   25 % of heads fit +2 %: 0.2.
4. UD and width 2 together: two positions have two masks. Price the stack with Bonnie's inputs before anyone
   builds both.
5. U30's mask against T30's: Yi prefers `up` first, the MHA subjects `gate` first. A per-model choice is
   model-specific tuning; leave at `up` first unless E-T25 (a) fails.
6. The phone's hwcap line and the THP mode: two lines in whatever kit runs next (no round for them).

## 10. Not done, limits

- Every perplexity ratio is from a float forward pass on 5 to 8 chunks, context half dense. Not the gate.
- The delta, nearest-position and spectrum statistics are from the dense context half of the same chunks;
  nearest-position was run on Yi and Llama-2 only (3 chunks).
- Per-sequence rules were screened with a 256-token (and 64-token) prompt; the official prompt is 156 tokens or
  fewer, which can only be worse.
- Kernel costs for arbitrary rows (1.15) and the scale table (+3 %) are counts, not measurements.
- Prices use memo-11's model (O 8, C 40, packed baseline), not Bonnie's memo-13 inputs (O 5 / 7 / 7, C 50, path
  A); the differences in ms carry over, the tok/s do not.
- Prox was read to 100,000 of 109,019 characters; LoRA-Drop's tables are inconsistent in the source; GRIFFIN's
  figures (perplexity against prompt length) are not readable as numbers.

## 11. Sources (2026-11-26)

- GRIFFIN, arXiv 2404.01365v3 (read): <https://arxiv.org/html/2404.01365v3>
- GLASS, arXiv 2508.14302v1 (read): <https://arxiv.org/html/2508.14302v1>
- DART, arXiv 2601.22632 (search summary): <https://arxiv.org/html/2601.22632v1>
- Prox, arXiv 2607.27591v1 (HTML, first 100,000 characters): <https://arxiv.org/html/2607.27591v1>
- LoRA-Drop, arXiv 2601.02569v1 (read): <https://arxiv.org/html/2601.02569v1>
- DeltaLLM, arXiv 2507.19608v1 (read): <https://arxiv.org/html/2507.19608v1>
- CATS, arXiv 2404.08763 (search summary): <https://arxiv.org/html/2404.08763v4>
- TDA, arXiv 2408.11393; SVD contextual sparsity predictors, arXiv 2603.14110 (search summaries):
  <https://arxiv.org/html/2408.11393v1>, <https://arxiv.org/html/2603.14110v1>
- RSR, arXiv 2411.06360; RSR-core, arXiv 2603.27462 (search summary): <https://arxiv.org/pdf/2411.06360>,
  <https://arxiv.org/abs/2603.27462>
- UNISPEC (ACL 2026), SDFP arXiv 2602.05499, SSSD arXiv 2411.05894 (search summaries):
  <https://aclanthology.org/2026.acl-long.285.pdf>, <https://arxiv.org/abs/2602.05499>,
  <https://arxiv.org/html/2411.05894v2>
- Arm learning path "Measure LLM decode bandwidth limits on Arm with llama-roofline" (search summary):
  <https://learn.arm.com/learning-paths/cross-platform/llm-decode-roofline/how-decode-uses-memory/>
- GCC `aarch64-cores.def` (Cortex-X1 feature flags; search summary):
  <https://gnu.googlesource.com/gcc/+/33fdbbe4ce6055eb858096d01720ccf94aa854ec/gcc/config/aarch64/aarch64-cores.def>
- ik_llama.cpp discussion 164 (search summary): <https://github.com/ikawrakow/ik_llama.cpp/discussions/164>
- Repository: R69-result.md, R72-result.md, memo-05, memo-07-work, memo-11-bytes.md, memo-11-exotic.md,
  memo-12-work/ebony-joins-r16.md, bonnie-joins-r14.md, bonnie-cpu-ladder.md (Answer first, §5),
  campaign 2 memo-21; the three served GGUF files; `lab/c3/logs/ppl-*-q4_0-b512-40.log` (reference chunks).
