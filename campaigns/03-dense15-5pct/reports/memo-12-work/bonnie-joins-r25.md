# c3 memo-12 work, round 25 (Bonnie): copy-run passes, priced on training data and read on the set and the claim prompt

- **Date:** 2026-11-24. **Author:** bonnie, `claude-fable-5-2`. Follows [bonnie-joins-r24.md](bonnie-joins-r24.md). Not
  committed. No verdict here. **Ebony has not reviewed this file.**
- **Labels.** *host*, *estimated* as before. Pairs are Yi / Llama-2; DeepSeek has no dump.
- **Script:** [bonnie_copyrun.py](bonnie_copyrun.py) (`lab/venv-c3/bin/python3 -I -B … <repo root>`). Counts are
  *host* (Tom's dumps; no model ran). Speeds are *estimated*: the counts put through the lane's time model of r14
  rung 6. No web source.
- **Guard (r24 §1.5), as applied here.** The policy is chosen on training prompts only (every third of the 1,500:
  500 prompts, 74,089 / 79,572 steps) and on the heads' held-out top-1. The set is then read with that policy:
  **this is the set's second look** (r24's rule T was the first). The official prompt is read, not used.

## 0. Answer first

1. **Copy-run passes are a small lever: +0.1 to +0.2 tok/s on the set, and between −0.4 and +0.4 on the claim
   prompt depending on what a wide pass costs.** They do not change any line for 12.
2. **Why small:** a matched run usually stops early. On training prompts, after a 3-token match the next token is
   right 0.52 / 0.57 of the time, the next two 0.32 / 0.36, the next three 0.20 / 0.24. Only matches of 8 or more
   tokens (3 to 4 % of steps) continue for three tokens more than half the time. And a width-3 or width-4 pass
   costs 26 to 60 % more than a width-1 pass.
3. **The policy the training data choose** (cost model of the stacks, `pair`): no lookup at match length 1
   (the head guesses); width 2 at length 2; width 3 at lengths 3 and 4; width 4 from length 4 or 5 on. The exact
   switch points move by one with the cost model.
4. **It depends on two unmeasured costs,** the GPU's time at three and four columns (R75 `x4m`) and the second
   and third positions' attention (r15). With either at the high end of its band the wide passes stop paying.
5. **Recommendation:** do not build it now. Keep r24's rule T (lookup at match length ≥ 2, else the head, width
   2). After R75, re-run this script with the measured `dev_ratio(3)` and `dev_ratio(4)`; it takes a minute.

## 1. The lever

At a decode step the context ends in some tokens that also occur earlier (in the prompt or in the reply so far).
The lookup proposes the token that followed them. A copy-run pass proposes the next W − 1 tokens that followed,
as a chain, and feeds them with the current token in one pass of width W. The model's own greedy output at each
position is compared with the next proposed token; the pass commits one token plus one for every leading
proposal that was right. Nothing is committed that the model did not produce, so the output is the stock greedy
text.

It needs the lane's width-W pieces (the W-position CPU kernel, `_mc3` at 3 or 4 columns) and nothing trained.

## 2. Training data (*host*; 500 training prompts per model)

Match length n = how many trailing tokens match at the newest earlier place with the longest match (capped at
8). "r right" = the first r continuation tokens equal the model's next r tokens.

| n | Yi: share of steps | 1 right | 2 right | 3 right | Llama-2: share | 1 right | 2 right | 3 right |
|---|---|---|---|---|---|---|---|---|
| 1 | 0.309 | 0.171 | 0.077 | 0.043 | 0.287 | 0.230 | 0.115 | 0.066 |
| 2 | 0.104 | 0.429 | 0.226 | 0.136 | 0.107 | 0.474 | 0.271 | 0.168 |
| 3 | 0.050 | 0.523 | 0.318 | 0.203 | 0.057 | 0.573 | 0.359 | 0.235 |
| 4 | 0.028 | 0.609 | 0.388 | 0.268 | 0.035 | 0.620 | 0.404 | 0.286 |
| 5 | 0.018 | 0.629 | 0.432 | 0.317 | 0.023 | 0.645 | 0.456 | 0.334 |
| 6 | 0.011 | 0.689 | 0.494 | 0.384 | 0.015 | 0.710 | 0.517 | 0.391 |
| 7 | 0.008 | 0.722 | 0.560 | 0.425 | 0.011 | 0.730 | 0.553 | 0.439 |
| 8 or more | 0.032 | 0.820 | 0.674 | 0.565 | 0.044 | 0.812 | 0.668 | 0.554 |
| head, held-out top-1 | | 0.377 | | | | 0.471 | | |

- A match exists at 56 % / 58 % of steps, but more than half of those are 1-token matches, which are worse
  than the head.
- The lookup beats the head's first guess from n = 2 on. It needs n ≥ 6 before two tokens in a row are right
  half the time.
- (This lookup takes the longest match up to 8 tokens; r24's took the newest match of 3, else 2, else 1. The
  two agree on which token is proposed in most steps; the precisions at n = 3 differ because r24's "3" meant
  "3 or more".)

## 3. Policy and cost model

**Cost of a pass** (*estimated*; ms; lane at K 300 as r14 rung 6: C5 out, head knob, two-stage head):

| cost model | Yi: width 1 / 2 / 3 / 4 | Llama-2: width 1 / 2 / 3 / 4 |
|---|---|---|
| the stacks' costs (O 5.2, 0.5 / 1.5 ms per extra position), `pair` | 95.9 / 102.6 / 121.0 / 140.4 | 107.7 / 116.2 / 138.5 / 162.0 |
| the same, `linear` | 95.9 / 105.7 / 122.4 / 142.1 | 107.7 / 119.9 / 140.1 / 164.0 |
| r15's costs (O 4.8 / 6.9, 3.2 / 5.3 ms per extra position), `pair` | 95.5 / 104.9 / 126.0 / 148.1 | 109.4 / 121.7 / 147.8 / 175.1 |

GPU time factors at 2 / 3 / 4 columns: 1.26 / 1.61 / 1.96 (the merged centres of r5; **guesses until R75**).
CPU at W positions: 1.45 / 1.9 / 2.35 (the W-position kernel, laptop count). Plus the head's time per committed
token, and 2.6 ms when the head makes the guess.

**Policy** = for each match length, the action with the best tokens per millisecond on the training prompts
(the head at width 2, or a copy at width 2, 3 or 4), found by iterating on the training set's own rate:

| cost model | n = 0, 1 | 2 | 3 | 4 | 5 | 6, 7, 8+ |
|---|---|---|---|---|---|---|
| stacks' costs, `pair`, Yi | head | copy, width 2 | 3 | 4 | 4 | 4 |
| stacks' costs, `linear`, Yi | head | 3 | 3 | 4 | 4 | 4 |
| r15's costs, `pair`, Yi | head | 2 | 3 | 3 | 4 | 4 |
| stacks' costs, `pair`, Llama-2 | head | 2 | 3 | 3 | 4 | 4 |
| stacks' costs, `linear`, Llama-2 | head | 3 | 3 | 3 | 4 | 4 |
| r15's costs, `pair`, Llama-2 | head | 2 | 3 | 3 | 4 | 4 |

## 4. Results (*estimated* tok/s from *host* counts; tokens per pass in brackets)

| cost model | prompt | head only | rule T (r24: copy at width 2 when n ≥ 2) | copy-run policy |
|---|---|---|---|---|
| Yi, stacks' costs, `pair` | official | 11.28 (1.348) | 11.76 (1.409) | 11.76 (1.409) |
| | set | 11.16 (1.333) | 11.70 (1.394) | **11.85** (1.459) |
| Yi, `linear` | official | 10.99 | 11.46 | 11.39 (1.409) |
| | set | 10.87 | 11.40 | **11.58** (1.481) |
| Yi, r15's costs | official | 11.05 | 11.53 | 11.53 |
| | set | 10.94 | 11.47 | **11.57** (1.450) |
| Llama-2, stacks' costs, `pair` | official | 8.89 (1.148) | 10.03 (1.292) | **9.66** (1.292) |
| | set | 11.14 (1.451) | 11.48 (1.488) | **11.61** (1.552) |
| Llama-2, `linear` | official | 8.63 | 9.73 | **10.12** (1.409) |
| | set | 10.82 | 11.15 | **11.32** (1.584) |
| Llama-2, r15's costs | official | 8.51 | 9.60 | **9.20** (1.292) |
| | set | 10.67 | 10.99 | **11.08** (1.552) |

- **On the set (second look): +0.09 to +0.18 tok/s over rule T on both models, in all three cost models.** About
  1 to 1.5 %.
- **On the claim prompt: nothing on Yi; on Llama-2 −0.37, +0.39 and −0.40** in the three cost models. With 31
  steps, whether two or three wide passes land on a run that continues decides the sign. Under `pair` the
  policy widens passes that then commit no more than width 2 would have (same tokens per pass, more
  milliseconds); under `linear` it happens to choose width 3 at n = 2 and catches a run.
- **Numbers here are not comparable with r24's table.** This script does not count the reply's first token (it
  comes from the prompt's pass), charges the head per committed token, and uses the longest-match lookup. Read
  the columns against each other, not against r24.

## 5. Reading

- The lever is real and exact, and small: a run is usually over after one or two tokens, and width costs.
- It leans on the two costs nobody has measured. If R75 reads `dev_ratio(3)` near 1.3 and `dev_ratio(4)` near
  1.6, wide passes are cheaper than assumed here and the set gain roughly doubles (*estimated*, not run). If
  they read 2.0 and 2.6, the policy collapses to rule T.
- It makes the claim prompt's result depend on a handful of wide passes. A claim built on that is fragile even
  when the rule was chosen cleanly.
- **So:** rule T stays the guess rule to carry. Copy-run is a second-order addition to revisit with R75's
  numbers.
- One thing it shows for free: **Llama-2's claim-prompt weakness is not cured by width.** Its reply copies in
  runs of 2 to 6 tokens with breaks between; the best this gets is 1.29 to 1.41 tokens per pass, against 1.45
  to 1.58 on the set.

## 6. Next questions, ranked

1. **Fold R75 into r18's draft and into this script** when its report is in the repository (`dev_ratio` at 3 and
   4 columns is the input this round lacks).
2. **The same analyses for DeepSeek** when its dump and head exist.
3. **Is the official prompt a fair claim prompt for a multi-token method at all?** It was fixed in campaign 1
   for a single-token runtime; with guesses its 31 steps decide a subject's result (Llama-2: 1.14 to 1.41
   tokens per pass by source and width). A short note for the lead and the owner: what the terms fix, what
   varies, and whether the claim should be read with the set beside it (r24 §3 line 4). Mine.
4. **Re-read r15 to r17 against Tom's host decode and KV screen** when they exist.
5. **Cold read rate; fixed share.**

## 7. Limits

- Two models; DeepSeek absent.
- The time model is r14's rung 6 with guessed GPU factors at 3 and 4 columns and guessed per-position costs.
  Every tok/s here inherits that.
- The policy is chosen on 500 training prompts per model with the head represented by one number (held-out
  top-1); on the set and the official prompt the head's real guesses are used.
- The set has now been read twice with rules I wrote (r24, this round). Both readings came after the rules were
  fixed on training data; neither changed a rule.
- The simulation commits at most W tokens per pass and does not model a rollback cost for rejected positions
  beyond their time.

## 8. Sources (2026-11-24)

Repository: `lab/c3/head/{yi,llama2}/train1500.{rows,ptok}`, `eval.{rows,ptok}`, `official.{rows,ptok}`,
`head2117-tok.pt.*.pred.npy`, `head3000-tok.pt.*.pred.npy`; `reports/memo-09-work/tom-head.md` (§4.1, held-out
top-1); `tools/c3_head_t1.py` (the lookup this extends); `bonnie_floor_r14.py`, `bonnie_stack_r13.py`,
`bonnie_tree_r2.py` (time model); my r5 (column factors), r14, r15, r24.
