# c3 memo-12 work, round 13 (Bonnie): what C7 and C5 really are, what supports them, and the stacks without credit they have not earned

- **Date:** 2026-11-23. **Author:** bonnie, `claude-fable-5-2`. Follows [bonnie-joins-r12.md](bonnie-joins-r12.md). Not
  committed. No verdict here. **Ebony has not reviewed this file.**
- **Labels.** *device*, *host*, *estimated*, *source* as before. Triples are Yi / Llama-2 / DeepSeek.
- **Scripts:** [bonnie_stack_r13.py](bonnie_stack_r13.py) (`python3 -I -B`; reproduces r5 §2.5's rows as its
  self-check); [bonnie_head_screen.py](bonnie_head_screen.py) (`lab/venv-c3/bin/python3 -I -B … <repo root>`).
- **New *host* numbers:** the head-screen test of §1.3, on Tom's existing dumps and the served heads, in numpy. No
  llama.cpp run, no dump made, nothing on the phone. No web source.

## 0. Answer first

1. **C7 is an unbuilt idea, and its one test was never run.** It is memo-08's "two-stage head, bit-plane screen":
   a proxy of about one bit per weight, activations in 3 bit-planes, the top 2,048 rows re-scored exactly. Its
   cost was a guess ("0.3 × Q4_0 per weight"), memo-09 rounded it to "head × 0.5", and its test (memo-08 T4) is
   not in any log. The proxy copy is also owner question (c), still open.
2. **A first test today says the one-bit screen does not generalise** (*host*, proxy on the 32-prompt set): on Yi
   it finds the right token every time; on Llama-2 it misses it in 0.66 % of rows at any K up to 8,192 and costs
   +0.046 in log-probability, about the whole budget (0.049) on the head alone. A Q4_0 copy (campaign
   2's built design) is clean on both, and saves only 9 to 24 % of the head's bytes.
3. **C5 is Ebony's "up rule" from memo-07, and it has never been tested on any model.** Its screen (U4) was not
   run. The nearest evidence points against it fitting beside anything else: the best possible ranking of FFN
   groups used the whole budget at 33 to 35 % on Mistral (*host*), and Llama-2 and DeepSeek are the intolerant
   subjects.
4. **C5 also collides with the lane.** It needs `up` finished before it can choose `gate` rows, so the fused
   [gate;up] stage splits in two and each layer pays one more join. At K 300 that takes back most of what C5
   saves (*estimated*).
5. **The stacks without unearned credit** (*estimated*; width 2, GPU 19.5 GB/s beside the CPU, C 40; K in µs):

   | stack | K 300, `linear` | K 300, `pair` | K for 12.00 on all three, `linear` / `pair` |
   |---|---|---|---|
   | as committed (C5 × 0.93, C7 × 0.5) | 11.59 / 11.54 / 11.13 | 11.89 / 11.88 / 11.43 | 164 / 218 |
   | C5 out, C7 as campaign 2 built it, head knob on | 10.82 / 10.80 / 10.35 | 11.10 / 11.12 / 10.64 | **24 / 90** |
   | C5 out, no C7, head in the repack type | 10.63 / 10.76 / 10.04 | 10.91 / 11.08 / 10.31 | **none / 32** |
   | C5 out, no C7, head in KleidiAI | 10.47 / 10.68 / 9.83 | 10.74 / 10.99 / 10.08 | none / none |

   **The 12-on-all-three line moves from K ≤ 164 to 218 µs to K ≤ 24 to 90 µs at best, and to nothing without a
   head screen.** At the K that R75 is expected to show (about 300) the stacks lose 0.8 tok/s and sit at 10.4 to
   11.1. DeepSeek binds in every row.
6. **Yi and Llama-2 alone** keep a line: K ≤ 111 to 130 µs (`linear`), 174 to 181 µs (`pair`), with the built
   two-stage head.

## 1. C7: the head screen

### 1.1 Where the row comes from (*source*: the memos)

| memo | text | what it is |
|---|---|---|
| memo-08 work, `bonnie-path.md` §3 (mine) | "Two-stage head, bit-plane screen (3 planes of x; top 2,048 rescored exactly at Q6_K)"; bytes −0.175 / −0.084 / −0.285 GB (81 % of the head); compute "screen 0.3 × Q4_0 per weight (guess)"; P 0.35 | a screen on a proxy of about one bit per weight, with activations cut to three bit-planes |
| the same, test T4 | "recall of the greedy top-1 within the screen's top 2,048, on 2,000 wikitext positions; PPL at both settings; Yi and DeepSeek. Pass: recall ≥ 0.999 and PPL ≤ +0.1 % on both; else close" | **never run** (no result in any memo or log I searched) |
| memo-09 §4 | "C7, head screen (head × 0.5)", band × 0.3 to 0.7, P 0.35 | the same lever, with a rounder factor |
| memo-10 and every stack since, the scripts' `hs = 0.5` | "C7 head × 0.5" | inherited |

- So the × 0.5 is my own guess from memo-08, never revisited. It is not campaign 2's two-stage head (a Q4_0 copy,
  r12 §2), which is the only head screen that exists as code.
- A proxy copy of the head in another format is a "non-stock layout" in the sense of owner question (c).

### 1.2 What a screen has to do under our gate

The gate scores the text's token, which is often not the model's first choice. A two-stage head gives exact
logits to the K rows its screen ranks highest; every other row keeps the screen's logit. So the screen needs two
things: the wanted token inside its top K, and tail logits good enough not to distort the normalisation.

### 1.3 A first test on what we have (*host*; proxy, not perplexity)

`bonnie_head_screen.py`: every third dumped row (about 1,200 per model), exact logits from the served Q6_K head
in float; target = the exact argmax. "Change" is the change of −log p(target) when the screen's top K get exact
logits and the rest keep the screen's.

| screen | Llama-2: recall at K 1,024 / 2,048 / 8,192 | Llama-2: change | Yi: recall | Yi: change |
|---|---|---|---|---|
| 1 bit per weight (sign × the row's mean magnitude), float x | 0.9934 / 0.9934 / 0.9934 | **+0.046** / +0.047 / +0.049 | 1.000 at every K | −0.001 / 0.000 / 0.000 |
| the same with x in 3 bits | 0.9934 at every K | +0.046 / +0.046 / +0.049 | 1.000 | −0.002 / −0.001 / 0.000 |
| Q4_0 copy of the head, float x | 1.000 at every K | −0.001 / 0.000 / 0.000 | 1.000 | 0.000 |

- **Llama-2 fails memo-08's own pass line** (recall ≥ 0.999) with the one-bit proxy, and a larger K does not
  help: the eight missed rows are outside the top 8,192. Its tail logits are also wrong enough to move the
  log-probability by 0.046, where the whole +5 % budget is about 0.049.
- **Yi passes.** Its hidden state is dominated by a few huge components (r11), which a sign proxy captures.
- By the two-purposes rule a lever is reported per model with what generalised: **this one does not.**
- Limits: my proxy is the crudest one-bit code (no per-row bias, no learned scale); the target is the greedy
  token, kinder than the gate's; DeepSeek has no dump. A better one-bit code might pass Llama-2. I would not put
  P above 0.1 on "a one-bit screen inside the budget on all three".

### 1.4 The factor that has evidence

Campaign 2's two-stage head (Q4_0 copy, exact top K, *source* and *device*): clean numerics here (table above;
campaign 2: +0.01 % on its model, *host*), bytes 0.80 / 0.91 / 0.76 of the Q6_K head at K 1,024 (r12 §2,
*estimated*), no measurable gain on OLMoE's phone runs (R62, *device*). With the head bus-bound (r12 §1.3) the
factor on time is the factor on bytes. It needs the head knob, and 147 / 74 / 236 MB more anonymous memory.

| head variant | ms per token (Yi / Llama-2 / DeepSeek) | basis |
|---|---|---|
| as committed: KleidiAI head × 0.5 | 4.5 / 2.2 / 6.9 | memo-09's guess |
| KleidiAI head | 9.04 / 4.40 / 13.88 | *device*, R72 |
| repack head | 7.7 / 3.7 / 11.8 | r12, *estimated* |
| repack head with campaign 2's two-stage head | 6.2 / 3.4 / 9.0 | × 0.80 / 0.91 / 0.76, *estimated* |

## 2. C5: the gate-row rule

### 2.1 Where the row comes from (*source*: the memos)

- **memo-07 work, `ebony-compute.md` C5, "Up rule":** compute `up` in full; rank FFN neurons by |u| (times a
  calibrated per-neuron gate scale); skip the `gate` rows and the `down` columns of the lowest groups of four.
  "No proxy, no side copy, no extra resident bytes." FFN × 0.91 at a skipped share of 0.33, token × 0.93.
  **"Untested."**
- **Its screen is U4** (same memo): the group tool with the score |u|, 20 / 30 / 40 %, Mistral and Qwen2.5,
  then Yi; closed if under 25 % of groups fit inside +4.5 % on either model. **Not run.** R72-result §"next"
  still lists "C5 gate-row … frontiers on Yi and Llama-2" as work to do.
- **My own memo-07 note** (`bonnie-reprice.md` item 4): the rule computes all of `up`, the `down` part costs
  more than dense at that share, so "the whole lever is the `gate` third only"; "worth U4 only after U1".
- memo-09 took it into the stack as "body × 0.93, P 0.30".

### 2.2 Perplexity evidence on the three subjects, and on which path

**None.** What exists nearby, all *host*, all on the lab build (repack path, r10):

| evidence | says |
|---|---|
| T1, groups of four ranked by the exact product (an upper bound for any proxy), Mistral, `-b 512` | about 30 % of groups in file order, 33 to 35 % clustered, inside +4.5 %. So the best possible version of this lever uses nearly the whole budget at the share C5 assumes |
| E2, the gate rule (rank by the gate's activation), 40 chunks | Mistral fails at 30 % (+6.34 %), Qwen2.5 fits (+3.73 %): closed, did not generalise |
| H3, uniform column sparsity on the subjects | Yi 40 %; Llama-2 and DeepSeek 20 % (30 % fails at +5.69 % and +5.13 %) |
| the k,v-dense rule, full set | Llama-2 +3.40 %, DeepSeek (BOS file) +3.55 % at 40 % elsewhere |

- Neither Mistral nor Qwen2.5 is a subject, and the two intolerant subjects are the ones that bind the goal.
- The budget is one budget. A lane already spends some on the GPU's arithmetic (Tom's 0006, not reported), and
  the head path may spend up to a fifth on Yi if the knob is not set (r11).
- **My P that C5 fits inside the budget on all three, beside a lane: 0.1** (was 0.30 as a stand-alone price).

### 2.3 C5 on a lane costs a join

The lane's stage list is q‖k‖v, o, gate‖up, down: four joins per layer, with gate and up fused in one launch
(level stacking, the design every stack since r2 assumes). C5 must see `up`'s output before it knows which
`gate` rows to compute. So gate and up become two stages: five joins per layer.

- One more join per layer costs about 0.56 × K per layer in the stage model (dead time less the CPU's work
  during it, plus slack): at K 300, 0.17 ms × 32 layers = 5.4 ms per pass.
- C5's saving is 7 % of the lane's byte time, 5.3 / 6.2 / 5.7 ms per pass (the body at K 0 is 75.5 / 88.1 / 81.9 ms
  in `pair`).
- **At K 300 the join takes back 85 to 100 % of it; at K 150 C5 keeps about half its price; only near K 0 is it
  worth its 7 %**
  (*estimated*). The stacks' "× 0.93" ignores this at every K.

## 3. The stacks, re-run

`bonnie_stack_r13.py`; width 2 with the W-position CPU kernel, trained head at its *host* top-1 (DeepSeek's
assumed), O 5.2, GPU at 19.5 GB/s beside the CPU, C 40; `linear` and `pair` as in r5 §2.5. tok/s, Yi / Llama-2 /
DeepSeek. All *estimated*.

**`linear`**

| stack | K 0 | K 150 | K 300 | K for 12.00 |
|---|---|---|---|---|
| as committed: C5 0.93, C7 × 0.5 | 14.24 / 13.90 / 13.25 | 12.78 / 12.61 / 12.10 | 11.59 / 11.54 / 11.13 | 245 / 233 / 164 |
| C5 out, C7 × 0.5 | 13.39 / 13.05 / 12.49 | 12.09 / 11.90 / 11.46 | 11.02 / 10.95 / 10.59 | 162 / 136 / 68 |
| C5 in, C7 as built | 13.90 / 13.67 / 12.89 | 12.50 / 12.42 / 11.79 | 11.36 / 11.38 / 10.87 | 212 / 208 / 119 |
| **C5 out, C7 as built, head knob on** | 13.09 / 12.84 / 12.17 | 11.85 / 11.73 / 11.19 | 10.82 / 10.80 / 10.35 | **130 / 111 / 24** |
| C5 out, no C7, head in the repack type | 12.82 / 12.78 / 11.74 | 11.62 / 11.68 / 10.83 | 10.63 / 10.76 / 10.04 | 99 / 104 / none |
| C5 out, no C7, head in KleidiAI | 12.59 / 12.66 / 11.45 | 11.43 / 11.59 / 10.57 | 10.47 / 10.68 / 9.83 | 73 / 89 / none |

**`pair`**

| stack | K 0 | K 150 | K 300 | K for 12.00 |
|---|---|---|---|---|
| as committed | 14.89 / 14.57 / 13.84 | 13.22 / 13.09 / 12.52 | 11.89 / 11.88 / 11.43 | 286 / 284 / 218 |
| C5 out, C7 × 0.5 | 14.02 / 13.68 / 13.05 | 12.53 / 12.36 / 11.87 | 11.32 / 11.28 / 10.89 | 212 / 197 / 132 |
| C5 in, C7 as built | 14.52 / 14.32 / 13.44 | 12.93 / 12.88 / 12.19 | 11.65 / 11.71 / 11.16 | 255 / 260 / 176 |
| **C5 out, C7 as built, head knob on** | 13.68 / 13.45 / 12.70 | 12.26 / 12.18 / 11.58 | 11.10 / 11.12 / 10.64 | **181 / 174 / 90** |
| C5 out, no C7, head in the repack type | 13.39 / 13.39 / 12.24 | 12.02 / 12.13 / 11.19 | 10.91 / 11.08 / 10.31 | 153 / 167 / 32 |
| C5 out, no C7, head in KleidiAI | 13.14 / 13.26 / 11.92 | 11.82 / 12.02 / 10.92 | 10.74 / 10.99 / 10.08 | 128 / 153 / none |

Reading:
- **C5 is worth 0.54 to 0.87 tok/s in these tables, and it is the lever with the least behind it.** The "C5 in"
  rows also over-credit it, because they do not charge the fifth join of §2.3.
- **C7 at × 0.5 against the built two-stage head: 0.15 to 0.3 tok/s.** DeepSeek pays most; its head is the
  largest.
- **The head knob is worth 0.1 to 0.3 tok/s** (the last two rows), most on DeepSeek, if r12's time estimate
  holds.
- **The line for 12 on all three**, set by DeepSeek:

  | | `linear` | `pair` |
  |---|---|---|
  | as committed | K ≤ 164 µs | K ≤ 218 µs |
  | C5 out, built two-stage head, knob on | K ≤ 24 µs | K ≤ 90 µs |
  | C5 out, no head screen, knob on | none | K ≤ 32 µs |

  R75's merged band for the in-situ K is 290 to 300 µs (170 to 400), before the hand-off is added. **So on the
  levers that have evidence, no stack I hold prices 12 on all three at a K inside R75's band**, in either
  model. Before this round the committed `pair` stack did so at the band's low edge (K ≤ 218).
- Every row still assumes the unbuilt pieces of the lane itself: the W-position CPU kernel at 1.44, `_mc3` near
  1.25, the trained head, a design that fuses stages. Those are R75's and Tom's to settle; this round only
  removes the two levers that were riding along.
- The stacks' DeepSeek head top-1 is still the assumed 0.438.

## 4. Next questions, ranked

1. **What replaces C5's 7 % of body, if anything?** Candidates already on the books: the k,v-dense column rule
   needs K1 (parked); per-group levels (memo-08, P 0.5, +0.03 of body); a lane share tuned per stage. Price each
   on the lane's stage model with its join cost, as §2.3 does for C5. Mine.
2. **U4 for the record** (Tom, host, minutes per cell with the existing group tool): the up rule at 20 / 30 % on
   Yi and Llama-2, on a KleidiAI build. It closes or revives C5 with data instead of my P.
3. **memo-08's T4 on the three subjects** with a better one-bit code (per-row scale and bias fitted on training
   rows), scored on the text's token, not the greedy one. Only if the owner's answer to (c) allows a proxy copy.
4. **DeepSeek's head:** it binds every line above, and both its top-1 and its hidden-state statistics are
   assumed. Tom's H2 dump gives the second; training its head gives the first.
5. **The stacks' own audit:** list every factor in `bonnie_tree.py` and its successors with its evidence class
   (device, host on a subject, host on another model, guess), as this round did for two of them. Mine; it is
   the table the lead needs before any claim about distance to 12.
6. **Cold read rate of the served file; fixed share: which C**, after R75.

## 5. Limits

- §1.3 is a proxy on chat prompts with the greedy token as target, on two subjects, with the simplest one-bit
  code. It can close "a naive one-bit screen", not "every proxy screen".
- §1.4's byte factors assume K 1,024 and campaign 2's count of exact rows; the repack head's time is r12's
  estimate.
- §2.2 lists evidence from other models and other rules; C5 itself has no data. My P is a judgement.
- §2.3 uses the stage model's own join cost; the fused-stage design it protects is itself unbuilt.
- §3 inherits everything in `bonnie_tree_r2.py` and the r5 `linear` variant.

## 6. Sources (2026-11-23)

Repository: `reports/memo-08-work/bonnie-path.md` (§3 lever table, test T4, lines 139, 355, 377);
`reports/memo-09-gap.md` (§4, lines 82 to 83, 95 to 101); `reports/memo-07-work/ebony-compute.md` (C5, U4, lines 133,
256, 269, 295) and `bonnie-reprice.md` (item 4, lines 225 to 236, 278); `reports/memo-05-y1-column-kernels.md` (E2);
`reports/memo-06-work/tom-screens.md` (H3 frontiers); `reports/memo-08-work/tom-k1.md` §6; `reports/R72-result.md`
line 336; `reports/memo-10-tree.md` line 159; campaign 2 `reports/README.md` (R62);
`patches/llama.cpp-c1-olmoe/0004-c2-head2-untied.patch`; `lab/c3/head/{llama2,yi}/eval.*` and
`lab/c3/models/{llama2-7b,yi15-6b}-q4_0.gguf` through `bonnie_head_screen.py`; `bonnie_tree_r2.py`,
`bonnie_joins_r5.py`; [bonnie-joins-r12.md](bonnie-joins-r12.md).
