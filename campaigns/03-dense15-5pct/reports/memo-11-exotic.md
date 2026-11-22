# c3 memo-11 (exotic): new routes to 12 tok/s. Retrieval, self-speculation, all compute units, memory system

Date: 2026-11-22. Author: Ebony (`claude-fable-5-2`). **Bonnie has not reviewed this**; §7 lists what I ask her to
attack. Not committed. Her memo of the same round (bytes and compute re-encoding, skipping: `memo-11-bytes.md`) is
not duplicated here; §5 is the interface between the two.

Labels: *device* = R72 / R69 values quoted; *host* = this laptop (Apple M4 Pro), relative only; *estimated* =
arithmetic; *published* = a paper's number, with its read status in §8. Two things in this memo are measured by me
on the host (§3.1, §3.8); everything else is *estimated*. Nothing here is a phone number of mine.

Work files (untracked): [memo-11-work/ebony_exotic.py](memo-11-work/ebony_exotic.py) (prices; Bonnie's model of
memo-10 unchanged, self-checked against her 6.90 / 6.21 / 6.22), [memo-11-work/ebony_knn.py](memo-11-work/ebony_knn.py)
(retrieval store on Tom's dumps).

Subjects: Yi-1.5-6B, Llama-2-7B, DeepSeek-LLM-7B (served unmodified). Goal 12 tok/s printed = 79.2 ms per token.
Every triple below is Yi / Llama-2 / DeepSeek.

## 1. Verdict

- **No route found here is a breakthrough** by the definition asked for (centre price at 12 or more on all three,
  and the key uncertainty settled on the host). I searched the four areas named by the lead and priced eleven
  routes. The best new one alone is about 8.6 / 7.9 / 7.7 tok/s at its centre.
- **Why, in one sentence:** on this phone a token is bounded by bytes over one bus port (29.9 GB/s, *device*), an
  extra verified position costs about half a pass of arithmetic, and every guesser that is cheap in bytes stops near
  top-1 0.4 to 0.5. No trick in areas (a) to (d) changes one of those three facts by the factor needed.
- **Three things are closed by this memo** (two by host measurement, one by price):
  - a retrieval store of the model's own outputs, keyed by the hidden state, is a worse guesser than Tom's head on
    both models (*host*, §3.1);
  - layer-skipping self-speculation (Draft & Verify, SWIFT, CLaSp, ConfLayers) is dominated by the head on a
    bus-bound phone, because a draft token's bytes cross the bus (§3.3);
  - the CPU alone cannot be taken from 30 to about 44 GB/s by anything an app controls: the same single-processor
    ratio (0.6 of nominal) is published for a Snapdragon 8 Gen 3 (§3.6).
- **One new route is worth one host test:** prefix-exit self-speculation ("ring", §3.4). It is Kangaroo re-cut for
  a bus-bound phone. It beats the head only if shallow features lift top-1 by 0.07 or more.
- **What a breakthrough would have to look like** is now a two-number condition that is fully host-testable (§5):
  a guesser's top-1 on the prompt set, and a body-time factor b from Bonnie's lane. With the measured heads, b must
  be at most 0.67 / 0.65 / 0.62 with the head screen C7 (0.61 / 0.59 / 0.57 at the L4 clock caps). With a guesser
  at 0.75 it is 0.82 / 0.73 / 0.73.
- P(a route of this memo, with memo-09 and memo-10 levers, reaches 12 on all three at its centre): about 0.01 (my
  judgement). P(all three at 12) overall stays at 0.002 to 0.003; nothing here moves it.

## 2. The definition, and the map of what must change

"Breakthrough" (lead, 2026-11-22): a route whose centre price reaches 12 on all three and whose key uncertainty
can be settled on the host. The second half matters because the owner holds device rounds until progress is certain
on host evidence (CAMPAIGN.md, 2026-11-22).

What bounds a token (inputs: R72 *device*; memo-10 for the pass model):

| quantity | value | what would have to change |
|---|---|---|
| bytes per token over the CPU's port | 3.11 / 3.64 / 3.42 GB at 29.9 GB/s read; ggml runs at 26.0 to 26.7 | fewer bytes (Bonnie), or a second port (GPU: read sum 46.5 GB/s, *device*) |
| arithmetic per extra verified position | 0.45 to 0.52 of a gemv (memo-10 E2, counted) | cannot fall below 0.26 to 0.32 with exact output; more compute only from A55 or GPU |
| guess accuracy at one cheap guess | 0.379 / 0.497 / 0.438 (*host*; DeepSeek *estimated*) | 0.68 for width 3 to pay, 0.75 to matter in a stack (memo-10 §6) |

Ceilings, to show which combinations can reach 12 at all (*estimated*, `ebony_exotic.py` §4, §1):

| combination | tok/s | uncertain part | host-testable? |
|---|---|---|---|
| perfect GPU lane on stock bytes (all bytes at 46.5 GB/s) | 11.3 / 10.4 / 10.0 | GPU kernel rate | no |
| perfect GPU lane + measured head at width 2 + O 5.2 | 13.2 / 13.5 / 12.1 | GPU kernel at bus rate while doing twice the arithmetic | no |
| CPU only, exact, head at width 2, any guesser | under 12 unless b ≤ 0.6 to 0.8 (§5) | guess top-1 and a byte cut | **yes, both** |

So every combination that reaches 12 contains either the GPU kernel rate (device only) or a byte cut (PPL, host).
Guess accuracy alone cannot do it: without a byte cut or the GPU, width 2 would need more than 2 tokens per pass
(memo-10 §5.3). That shapes the rest of this memo: my routes can supply the first number of §5, not the goal.

## 3. Routes

Prices are for the lever alone against stock 6.90 / 6.21 / 6.22 (137.6 / 153.0 / 152.7 ms), C 40 unless said.
P = probability that the route delivers its centre on the device inside the rules (my judgement).

| # | route | area | centre price, tok/s | P | host test that would make it certain | verdict |
|---|---|---|---|---|---|---|
| 3.1 | Retrieval store keyed by hidden state (kNN) | a | below the head: tokens per pass 1.27 / 1.38 against 1.33 / 1.45 | – | done | **closed (*host*)** |
| 3.2 | Text datastore on flash (REST, suffix automaton) | a | within ±0.1 of the head; as a second opinion +0.0 to +0.2 | 0.2 for +2 ms | T1 arm e already measures it at small size | not proposed |
| 3.3 | Layer-skip self-draft, no training (SWIFT class) | a | 7.65 / 7.05 / 6.76 at acceptance 0.9, skip 0.45 | 0.3 | none needed | **closed by price**: head gives 7.94 / 7.79 / 7.38 |
| 3.4 | **Prefix-exit ring** (exit after 2 layers, re-read the prefix, ride the deep layers at width 2) | a | 8.56 / 7.87 / 7.67 at top-1 0.60; 9.00 / 8.32 / 8.06 at 0.70 | 0.10 for 0.60 on all three | E-T3: exit-feature head top-1 | **open, one host test** |
| 3.5 | Draft layer (EAGLE size) | a | memo-10 E5: centre stack 11.4 / 10.2 / 10.3 at 0.75 | 0.02 | T13 (published Llama-2 layer on our file) | owner question, unchanged |
| 3.6 | CPU port above 30 GB/s (DDR step, governor) | c | what-if 8.2 / 7.4 / 7.4 at 38 GB/s; 8.7 / 7.9 / 7.8 at 46 | 0.03 | none (device read-out, rider T6) | **closed as a lever**; rider stays |
| 3.7 | Third bus master (Hexagon), A55 for head or attention | b | under +0.2 | 0.05 | none | closed by arithmetic |
| 3.8 | Temporal delta sparsity (update only inputs that moved since the last token) | d | – | 0.02 | weak hint done; mid-layer screen is Bonnie's kind | **not proposed** (*host* hint against) |
| 3.9 | Exact GPU lane on an int8 copy (`cl_qcom_dot_product8`) | b | 7.4 / 6.8 / 6.7 at its ceiling | 0.15 | identity only | dominated by the Q4_0 lane's upper band; note kept |
| 3.10 | Huge pages, prefetch distance, read order | c | at most +0.3 to +0.5 (the gap from 26.0 to 29.9) | 0.15 | none (page size and prefetchers differ on the host) | device rider only |
| 3.11 | Small exact tables (layer-1 q, k, v by token id), LUT kernels (T-MAC) | d | about −1 ms; −6 ms | 0.1 | host kernel bench | not worth a build |

### 3.1 Retrieval store keyed by the hidden state (*host*, measured today)

Question: Tom's head overfits (training top-1 0.78 against 0.33 held out). A non-parametric store does not overfit
in that way: is the nearest stored state a better guess, or a useful second opinion?

Setup: store = every training pair of Tom's greedy dumps (key: unit-length post-norm hidden state of row j; value:
greedy token of row j + 1). Queries: the prompt-set dump (never in the store). Cosine, exact search.
`ebony_knn.py`, self-check included; 33 to 49 s per model.

| arm | Yi (store 313,261 pairs; 3,531 queries) | Llama-2 (467,907; 3,606) |
|---|---|---|
| Tom's `tok` head | top-1 0.379, tokens per pass 1.332 | 0.497, 1.453 |
| nearest key | 0.225, 1.196 | 0.335, 1.288 |
| vote of 16 | 0.257, 1.225 | 0.364, 1.309 |
| nearest key among rows that produced the same token | 0.295, 1.268 | 0.413, 1.380 |
| head or store right (upper bound of any chooser) | 0.470 | 0.595 |
| similarity gate fitted on even prompts, scored on odd | 0.386 = head alone (store used in 1.4 % of rows) | 0.507 = head alone (0 %) |

- **Closed as a guess source on both models.** At every similarity level but one (Yi above 0.9, 3 % of rows) the
  head is right more often than the store. A fitted gate never prefers the store.
- A side result for memo-10's gate (Y6, T8): when head and store agree (0.258 / 0.376 of rows) the head is right in
  0.795 / 0.839; when they differ, in 0.235 / 0.291. Agreement is a strong confidence signal. The head's own
  probability may carry the same information for nothing; T8 should print both.
- The upper bound 0.47 / 0.60 says that even a perfect chooser between two guessers of this kind stays under the
  0.68 that width 3 needs.
- Limit: the store is 0.3 to 0.5 M pairs. REST's own table gives +0.12 × speed (1.57 to 1.69 ×, Vicuna-7B, GPU,
  *published*) for a store 26 × larger. Nothing suggests a different sign at 10 × our size.

### 3.2 Text datastore on flash (REST, SAM-Decoding)

- *Published*: REST with a 12 GB UltraChat store and a tree of up to 64 draft tokens gives 1.69 × on Vicuna-7B
  (MT-bench, greedy, A6000); 1.57 × with 465 MB. SAM-Decoding (ACL 2025) reports 18 % over PLD and REST. Neither
  gives a one-guess number. On a GPU 64 draft positions cost about one pass; here the second position alone costs
  0.12 to 0.23 of a pass, so only the first token of the retrieved continuation counts.
- Tom's arm f (counted 3-gram table from the model's own outputs) and arm e (candidates ranked by the last logits)
  are this route at small size: p 0.23 to 0.33 on the set, below the head (*host*, tom-head.md §6).
- **A rule point for the lead.** A store that contains the model's answer to the official prompt or to a prompt-set
  prompt is a lookup of the benchmark, not a lever. Tom's split already excludes them; any larger store must keep
  that exclusion, and the pre-registration should say so.
- Not proposed. If T12 (more head data) is run, the same text can fill arm f at no extra cost.

### 3.3 Layer-skipping self-speculation without training (Draft & Verify, SWIFT, CLaSp, DEL, ConfLayers, KnapSpec)

- The draft is the same frozen model with 40 to 50 % of its layers skipped; no new weights. *Published* acceptance
  per token is high (ConfLayers Table 2: 0.78 to 0.97 at skip 44 to 50 %, models of 34 B and 72 B).
- The papers assume a draft token costs its share of the arithmetic and a verify pass costs one pass at any width.
  Here a draft token's layers are bytes over the bus: 0.55 of a pass each, and the verify pass at width 2 costs 1.12.
- Price (acceptance 0.9, skip 0.45, one draft token, C 40): 7.65 / 7.05 / 6.76. At 0.97 and skip 0.50:
  8.11 / 7.51 / 7.15. Tom's head at width 2: 7.94 / 7.79 / 7.38. **Dominated on Llama-2 and DeepSeek at any
  published acceptance, and on Yi unless acceptance is 0.97.**
- The published GPU results at our model size agree: ConfLayers Table 1 gives 0.75 to 1.11 × on Llama-3-8B and
  0.93 to 1.15 × on Llama-2-13B for DEL, SWIFT and its own method (*published*, read).
- Closed by price. No test asked.

### 3.4 Prefix-exit ring (new; Kangaroo re-cut for a bus-bound phone)

**Mechanism.** One sweep through the weights:
1. the committed token goes through the first l layers alone (l = 2);
2. a small exit head reads its layer-l state, together with the previous position's final state and the token
   embedding (what Tom's head reads now), and guesses the next token;
3. the guess goes through the same l layers (a re-read of l ÷ L of the body, 6 % at l = 2);
4. both positions ride layers l + 1 to L at width 2, sharing the weight read, as in Y1.

The guess is made after the model has seen the current token in context for l layers, which the head of memo-09
cannot. The prefix states of the guessed position are exact, so nothing is recomputed on a hit. Output is exact
(same check as Y1). It needs Y1's W-position kernel and one sidecar head per model, the class the owner allowed on
2026-11-22, provided the exit head stays a block and not an attention layer (below).

**Price** (`ebony_exotic.py` §2; exit head cost 2.8 ms + shortlist; d as measured):

| exit-head top-1 | alone, C 40 | alone, C 35 (L4 caps) | centre stack (G 8, dispatch 8, C7, C5, O 5.2) |
|---|---|---|---|
| 0.50 | 8.11 / 7.43 / 7.28 | 7.58 / 6.92 / 6.82 | – |
| 0.60 | **8.56 / 7.87 / 7.67** | 8.00 / 7.34 / 7.19 | 10.1 / 9.1 / 9.2 |
| 0.70 | 9.00 / 8.32 / 8.06 | 8.42 / 7.75 / 7.56 | 10.6 / 9.6 / 9.7 |
| 0.80 | 9.50 / 8.85 / 8.49 (width 3) | 8.87 / 8.23 / 7.95 | 11.1 / 10.2 / 10.1 |
| head of memo-09 at width 2, for comparison | 7.94 / 7.79 / 7.38 | 7.36 / 7.20 / 6.86 | 9.3 / 9.0 / 8.8 |

- Break-even against the plain head: the ring's top-1 must exceed the head's by about 0.05 × (1 + p), that is
  **+0.07**. Exit at layer 4 costs 0.3 to 0.4 tok/s more than layer 2 at the same accuracy.
- **No cell reaches 12**, alone or in the centre stack, even at 0.80. It is a better first number for §5, not a
  route to the goal by itself.
- What is known about the accuracy. Kangaroo (*published*, read): Vicuna-7B, exit at layer 2 of 32, adapter of one
  attention block plus two norms (67 M parameters), trained 10 epochs on ShareGPT; committed tokens per verify
  1.41 to 2.22 by task (2.22 on MT-bench), drafting stopped below confidence 0.6. Top-1 is not published. From 2.22
  tokens per verify with drafts of up to 6 I read a first-token accuracy of 0.55 to 0.70 (*estimated*).
- Two differences from Kangaroo, in opposite directions: our head also reads the previous position's final state
  (should help); it is a block trained on laptop-scale data, not an attention adapter trained on ShareGPT (should
  hurt). An attention adapter with its own state is the "draft model" question of memo-10 E5, not this route.
- My prior for a block on [final state of j − 1, token embedding, layer-2 state of j]: 0.45 / 0.57 / 0.50.
  P(≥ 0.60 on all three) 0.10; P(≥ 0.70 on all three) 0.03.

### 3.5 Draft layer and newer trained drafters

- Nothing new that trains at laptop scale. Lyanna (2026-02, *published*, read): hidden-state drafter trained on
  68 K ShareGPT samples for 20 epochs, Llama-2-7B conditional acceptance 0.91 / 0.80 / 0.70 for the first three
  steps (tree acceptance, not top-1). DFlash (block diffusion drafter) is named in SpecSearch's abstract
  (2026-11-17) with 3.5 × on a GPU; not read.
- These confirm memo-10 E4: accuracy near 0.75 exists only at ShareGPT scale with a drafter of 0.1 to 0.25 B
  parameters. The owner question of memo-10 E5 is unchanged; T13 still bounds it for Llama-2 at no training cost.

### 3.6 Memory system: why 30 and not about 44 GB/s

- *Device*: CPU alone 29.9 GB/s (r4 ≈ r3, so not a per-core limit), GPU alone 31.1, both together 46.5 of a
  nominal 51.2 (4 × 16-bit LPDDR5 at 3,200 MHz; nominal *published*).
- *Published* (HeteroInfer, SOSP 2025, read): on a Snapdragon 8 Gen 3, "each individual processor (CPU, GPU, or
  NPU) achieves only 40 to 45 GB/s" of a 68 GB/s peak under decoding, and GPU with NPU together 59.5. That is 0.59
  to 0.66 of nominal for one processor, against our 0.58 to 0.61. **The same ratio two SoC generations apart reads
  as a per-processor limit of the SoC's memory path, not a setting of this phone.** The paper gives no root cause.
- The alternative I checked: a DDR clock one or two steps low. The Snapdragon 865 table (read) has steps at 2,092
  and 2,736 MHz; 2,092 MHz is 33.5 GB/s nominal, and 0.91 of it is 30.5, which would fit both single-processor
  cells. Against it: the DDR vote follows measured traffic (bwmon), and 30 GB/s of traffic is far above the top
  entry of the table, so the clock should already be at its top step. P(the DDR step explains the 30) 0.15; P(an
  app could then raise it) under 0.2 of that. The 888's own table was not read (the page returned an error).
- What-if, if the CPU could read 46 GB/s: the body turns compute-bound at once, 8.7 / 7.9 / 7.8 tok/s. **Even the
  optimistic answer to "why not 44" is not a route to 12**, and it removes the slack that width 2 spends.
- The read-out (DDR devfreq nodes, memo-09 T6) stays a free rider for the next kit, as a fact to record. No host
  test exists: the M4's memory path says nothing about the 888's.
- Huge pages, prefetch distance, read order (3.10): the room is the gap between ggml cold (26.0 to 26.7) and the
  read (29.9), at most 11 %. Campaign 1 closed software prefetch (R23) and finer chunks (R33). THP state is in T6.
  Android's performance-hint API sends hints for clock and core choice; I found nothing that says it lifts a
  lowered `scaling_max_freq` (documentation pages, search summary). The cap is Bonnie's "clocks" row, unchanged.

### 3.7 All compute units at once

- **Bus first.** CPU + GPU already read 46.5 of 51.2 GB/s. A third master (Hexagon) can add at most 10 % of bytes,
  and memo-09 priced its lane at −7 to −9 ms with P 0.10. Unchanged.
- **A55 cluster for the head.** The head could run on the four A55s while the big cores start the next sweep on a
  shortlist argmax, with a rollback on mismatch (exact). It saves nothing: the head's 0.2 to 0.35 GB cross the same
  port during the sweep, and the sweep is bus-bound, so the 8 to 14 ms come back as a slower body. Closed by
  arithmetic. Attention and KV are under 3 % of the bytes (memo-08), so the same holds.
- **A55 for part of the second position** is memo-10's Y9 (−5 ms, P 0.05). No change; four A55s have the
  arithmetic for about a third of one position (8.9 of 26 GB/s).
- **GPU lane.** Two notes, no new price:
  - *Published* (arXiv 2607.05475, read): Qwen2.5-7B decode on a Snapdragon 8 Gen 3 phone: CPU llama.cpp 8.7, GPU
    llama.cpp 5.4, GPU MNN 11.7, GPU MLC 8.0 tok/s. The upstream OpenCL kernel is not the fastest Adreno kernel even
    on a verified GPU; MNN's is twice as fast there. If T3 is ever run, MNN's OpenCL 4-bit kernel is a second
    source. No Adreno 660 number exists in any of the three papers read.
  - **An exact lane is possible** (3.9): the phone reports `cl_qcom_dot_product8` (*device*, R72 x2info). A lane
    that holds its rows as int8 with the Q4_0 block scales can do the CPU kernel's integer sums and the same float
    steps, so it need not spend the PPL budget. Its cost is bytes: 34 against 18 per block, so at the GPU's whole
    bus share it replaces 13.7 GB/s of Q4_0 work and takes 0.68 GB/s from the CPU per GB/s it replaces. Ceiling
    body × 0.86, 7.4 / 6.8 / 6.7 tok/s; the Q4_0 lane at g 8 gives × 0.84. It also needs 1.9 × the lane's bytes in
    GPU memory. Dominated unless the budget is the binding constraint. Kept as a note for G, not a lever.

### 3.8 Temporal delta sparsity (my idea; weak host hint against)

- Idea: keep, per matrix, the product for a stale input vector and update only the columns whose input moved by
  more than a threshold since they were last updated. If consecutive tokens' inputs were close, far fewer columns
  than plain magnitude thresholding would be touched (memo-02's column sparsity kept 0.60 to 0.65).
- Hint (*host*, prompt-set dumps, final post-norm state only, the one layer stored): cosine between consecutive
  tokens 0.51 / 0.35 (Yi / Llama-2); the squared change is 1.02 / 1.30 × the squared state. At equal threshold the
  delta rule keeps **more** coordinates than the magnitude rule (0.51 against 0.48 on Yi, 0.64 against 0.60 on
  Llama-2 at half the RMS). Subtracting the mean vector gains 0.01.
- This is one layer, and not an FFN input. Mid-layer inputs were not stored. With the column kernel at 1.5 to
  1.6 × per touched byte (*device*, R72), the rule would need keep ≤ 0.35 to matter. I put that at P 0.02 and do
  not ask for a screen. If Bonnie's screens dump mid-layer inputs for another reason, the same three lines answer it.

### 3.9 to 3.11 Small items

- Layer-1 q, k, v by token id: these three products depend only on the token. A table on flash (64,000 × 3 × 4,096
  values for Yi) removes 0.9 % of the body's bytes: about −1 ms. Exact if stored as the kernel's own floats.
- LUT kernels (T-MAC): a derived layout computed on the phone is runtime by rule 1. For 4-bit weights the published
  gain is in arithmetic, which is not our bound at one position; it would lift C and so lower kB(2) by about
  0.05 (−6 ms), with a second copy of the body in anonymous memory that Llama-2 and DeepSeek cannot hold (R72
  anon42). The table quantisation is a numerics change. Not proposed.
- Lossy GPU texture formats for weights (hardware decompression): weights are noise-like, the formats assume
  smooth blocks. Not priced; P under 0.01.
- **Rule-blocked, not proposed:** doing decode work during prefill (the metric times decode only), and typical or
  top-k acceptance of drafts (the terms ask for identical greedy output).

## 4. What I searched and did not find (2026-11-22)

- A one-guess accuracy for any retrieval method, or for Kangaroo's adapter.
- Any measurement of speculative decoding on a phone CPU. The nearest is a laptop study (arXiv 2607.17283, read):
  llama.cpp Metal verification of a 7B Q4_K_M target costs 51.0 / 74.7 / 164.9 / 218.5 ms for 2 / 3 / 5 / 7 tokens,
  then a flat 141 ms from 9 up, because small batches run a matrix-vector kernel per position. That is the same
  code path memo-09 §10.2 found on the CPU, seen from the outside.
- An Adreno 660 token rate, a measured Snapdragon 888 read rate, or the 888's DDR table (one fetch failed).
- A training-free method newer than those in memo-09 and memo-10 that lifts a single guess above 0.5.

## 5. The interface with Bonnie's lane: a host-testable condition for 12

A CPU-only, exact stack is: a guesser of top-1 a at width 2 (width 3 where it pays), O 5.2, and a body whose time
is b × stock. Largest b at which the subject prints 12.0 (*estimated*, `ebony_exotic.py` §1; guess cost 2.6 ms for
a head, 5.7 to 6.2 ms for a draft layer; d 0.86 to 0.90 as measured):

| guesser top-1 | full head, C 40 | with C7 (head × 0.5), C 40 | with C7, C 35 (L4 caps) | with C7 and the GPU lane at g 8, dispatch 8 |
|---|---|---|---|---|
| measured heads (0.379 / 0.497 / 0.438) | 0.62 / 0.63 / 0.55 | **0.67 / 0.65 / 0.62** | 0.61 / 0.59 / 0.57 | 0.68 / 0.66 / 0.62 |
| 0.60 | 0.70 / 0.65 / 0.59 | 0.75 / 0.67 / 0.66 | 0.69 / 0.61 / 0.61 | 0.77 / 0.69 / 0.68 |
| 0.75 | 0.77 / 0.71 / 0.65 | 0.82 / 0.73 / 0.73 | 0.75 / 0.67 / 0.67 | 0.85 / 0.76 / 0.75 |
| 0.85 | 0.85 / 0.79 / 0.72 | 0.91 / 0.82 / 0.80 | 0.83 / 0.74 / 0.73 | – |

- Reading: **every 0.10 of top-1 buys about 0.04 to 0.05 of b.** The guesser is the weaker of the two numbers; the
  byte cut carries the goal. A body-time factor of 0.65 on Llama-2 and 0.62 on DeepSeek with today's heads would be
  a breakthrough, and both numbers (PPL of the cut, top-1 of the head) are host measurements.
- b here is a time factor. A cut that removes weights (skipping) lowers arithmetic with the bytes and keeps kB(2);
  a re-encoding that keeps the weights but shrinks their bytes leaves the arithmetic, so the body turns
  compute-bound sooner and kB(2) rises. Her model splits b and k; mine does not. Her numbers decide.
- The GPU lane at its centre moves the condition by 0.01 to 0.03 only, because its dispatch cost is paid per pass.

## 6. Lists for Tom (through the lead)

### 6.1 Test list

| # | test | where | cost | closes when |
|---|---|---|---|---|
| E-T1 | Retrieval store keyed by hidden state, against the head | host | **done** (§3.1) | closed on Yi and Llama-2 |
| E-T2 | Delta against magnitude thresholding on final states | host | **done** (§3.8), weak | not proposed further |
| E-T3 | **Exit-feature head.** `tools/c3_head_dump.cpp` also writes each row's layer-2 and layer-4 state; `tools/c3_head_train.py` gets an input variant [final state of row j − 1, embedding of its token, layer-l state of row j] with target the token row j produces; score top-1 and tokens per pass on the prompt set, Yi and Llama-2 | host | one dump change, a re-dump (30 to 45 min per 1,500 prompts per model), minutes of training | gain over the `tok` head under 0.07 on either model → ring closed. Top-1 ≥ 0.60 on both → price §3.4 and train DeepSeek |
| E-T4 | T13 of memo-10 (the published EAGLE layer for Llama-2 on our Q4_0 greedy tokens), raised from optional | host | 1 to 2 hours, a 0.5 GB download (lead's call) | under 0.65 → the 0.75 row of §5 has no known source on any subject |
| E-T5 | T8 of memo-10 with one more column: head and store agreement as the gate signal (§3.1) | host | a few lines; the store search takes under a minute | agreement gate no better than the head's own probability → dropped |
| T6 | Unchanged rider: DDR devfreq nodes and THP mode in the next kit that runs for another reason | device | a few lines | a fact for §3.6; not a reason for a round |

No device test is asked for by this memo.

### 6.2 Try list

| # | lever | after | label | price alone (Yi / Llama-2 / DeepSeek) | P |
|---|---|---|---|---|---|
| Y11 | Prefix-exit ring on Y1's kernel: exit head sidecar per model, prefix re-read, width 2 from layer 3; env-gated, off by default, model-generic, two subjects | E-T3 at ≥ 0.60 on both; Y1's own tests (T2, T10) | bit-identical output (greedy identity on the prompt set and the official prompt) | at 0.60: −26.6 / −32.3 / −28.8 ms (8.56 / 7.87 / 7.67); against Y1 + Y5: −8.7 / −1.2 / −4.9 ms | 0.10 |
| – | Exact int8 GPU lane | only if G is built and the PPL budget binds | bit-identical intended | ceiling body × 0.86 | 0.15 |

Nothing else in this memo is proposed for a build.

## 7. For Bonnie's review

1. §3.4 cost model: the prefix is re-read at width 1 at the bus rate (2 layers are about 0.2 GB, far above L3), and
   the deep layers use her kB(W). Is a re-read of 6 % per drafted token right, or does the prefix at width 1 leave
   compute idle that I should credit?
2. §5: b as one time factor. Please replace the table with her b and k split where the two differ by more than 0.03.
3. §3.6: P 0.15 for the DDR step, and my reading of HeteroInfer's single-processor figure as a per-processor limit.
4. §3.3: is there a layer-skip draft whose skipped layers' bytes are not read (they are not), but whose kept
   layers are shared with the verify pass? I found none: only a prefix can be shared, which is §3.4.
5. §3.7: the claim that overlapping the head on the A55s saves nothing because the sweep is bus-bound.

## 8. Sources (read 2026-11-22 unless said)

- Kangaroo, arXiv 2404.18911 (HTML read; figures not readable): <https://arxiv.org/html/2404.18911>
- ConfLayers, arXiv 2604.14612v1 (read): <https://arxiv.org/html/2604.14612v1>
- SWIFT, arXiv 2410.06916; CLaSp, arXiv 2505.24196; KnapSpec, arXiv 2602.20217 (search summaries; the KnapSpec PDF
  text could not be extracted): <https://arxiv.org/pdf/2410.06916>, <https://arxiv.org/abs/2505.24196>,
  <https://arxiv.org/pdf/2602.20217>
- REST, arXiv 2311.08252 (read): <https://arxiv.org/html/2311.08252>
- SAM-Decoding, arXiv 2411.10666, ACL 2025 (search summary): <https://arxiv.org/html/2411.10666v3>
- Lyanna, "Make Every Draft Count", arXiv 2602.21224v1, 2026-02-02 (read): <https://arxiv.org/html/2602.21224v1>
- SpecSearch, arXiv 2611.22296, 2026-11-17 (abstract only): <https://arxiv.org/abs/2611.22296>
- "Lossless but Not Free", arXiv 2607.17283, 2026-07-19 (read): <https://arxiv.org/html/2607.17283>
- Verifier skipping, arXiv 2608.14787 (abstract only; a lossy policy, not usable here): <https://arxiv.org/abs/2608.14787>
- HeteroInfer, "Characterizing Mobile SoC for Accelerating Heterogeneous LLM Inference", arXiv 2501.14794v2, SOSP
  2025 (read): <https://arxiv.org/html/2501.14794v2>
- "Is Your NPU Ready for LLMs?", arXiv 2607.05475v1 (first 100,000 characters read): <https://arxiv.org/html/2607.05475v1>
- "When NPUs Are Not Always Faster", arXiv 2605.27435v1 (read): <https://arxiv.org/html/2605.27435v1>
- Snapdragon 865 device tree, DDR table and bus nodes (read; the page was cut before the GPU bus table):
  <https://android.googlesource.com/kernel/msm-extra/devicetree/+/bac014853ace9398eb34866209e1eb183c7fdfbe/qcom/kona.dtsi>
- Snapdragon 888 DDR table, Fairphone kernel commit (search summary only; the fetch returned HTTP 429):
  <https://gitlab.e.foundation/e/devices/android_kernel_fairphone_fp5/-/commit/3a2d0a2e08966a94c7219677ba0644fca4c95200>
- Android performance hint and thermal APIs (search summary): <https://source.android.com/docs/core/perf/performance-hint-api>
- T-MAC, arXiv 2407.00088 (search summary): <https://arxiv.org/html/2407.00088v2>
- Repository: R72-result.md, memo-09-gap.md, memo-10-tree.md, memo-09-work/tom-head.md, `lab/c3/head/` dumps
  (untracked), docs/negative-results.md, memo-02-host-screens.md.

Not done: no device run; no layer-l dump (E-T3 is Tom's); DeepSeek has no head and no dump, so §3.1 covers two
subjects; the store was not grown beyond Tom's training rows; the 888's own DDR table, Kangaroo's figures and
KnapSpec's text were not read; Bonnie's memo of this round was not read (it did not exist when I wrote this).
