# c3 memo-12 work, round 24 (Bonnie): the claim prompt's guesses, source by source; a chooser fixed on training data; a proposed text on scoring the fixed set

- **Date:** 2026-11-24. **Author:** bonnie, `claude-fable-5-2`. Follows [bonnie-joins-r23.md](bonnie-joins-r23.md). Not
  committed. No verdict here. **Ebony has not reviewed this file.**
- **Labels.** *host*, *estimated* as before. Pairs are Yi / Llama-2; DeepSeek has no head and no dump and is absent
  from every table of §1.
- **Script:** [bonnie_claim_guess.py](bonnie_claim_guess.py) (`lab/venv-c3/bin/python3 -I -B … <repo root>`; numpy;
  reads Tom's dumps and saved head guesses; trains nothing, runs no model). All of §1 is *host*. §2's stack
  prices are *estimated* (`bonnie_floor_r14.py` functions, run inline). No web source.
- **A caution about my own order of work.** I saw the official prompt's results (r23) before I wrote the rule of
  §1.3. Its threshold comes from training data only, but the idea of gating the lookup by match length came
  after seeing that lookup helps on the claim prompt. §1.5 says what that does and does not contaminate.

## 0. Answer first

1. **On the claim prompt the sources split by model** (*host*, tokens per pass over its 32 tokens):

   | source | Yi | Llama-2 |
   |---|---|---|
   | best head alone | 1.391 | 1.143 |
   | prompt lookup alone | 1.032 | 1.280 |
   | lookup when it has any candidate, else the head (Tom's `ah`) | 1.333 | 1.333 |
   | **lookup when its match is 2 or 3 tokens long, else the head ("rule T")** | **1.455** | **1.333** |
   | either source right (an upper bound, not a rule) | 1.455 | 1.391 |

2. **Rule T is fixed by training data alone:** on 1,500 training prompts a lookup with a 3-token match is right
   0.62 / 0.65 of the time and with a 2-token match 0.42 / 0.47; with a 1-token match 0.17 / 0.23. The heads'
   held-out top-1 is 0.377 / 0.471. "Use the lookup where it beats the head on training data" gives the same
   rule for both models: lengths 2 and 3.
3. **It also improves the fixed prompt set, on both models:** 1.332 → 1.391 (Yi), 1.453 → 1.486 (Llama-2). A rule
   that helped only the claim prompt would be suspect; this one helps both, which is what the set is for.
4. **Price on the claim prompt** (*estimated*; r14's rung 6, K 300, `linear` / `pair`): Yi **11.79 / 12.10**
   (r23: 11.31 / 11.61 with the head alone); Llama-2 **9.97 / 10.27** (r23: 8.60 / 8.85). Llama-2 stays the
   weak one on the claim prompt: at K 150 it is 10.84 / 11.25, and its line for 12 is K ≤ 51 µs in `pair`.
5. **The guard I propose:** the guess rule and the head are chosen on training and held-out data, written into
   the pre-registration with their numbers, and the claim prompt and the set are only read. §1.5.
6. **A text on scoring the fixed set, for the owner, is §3:** six sentences that fix tokens per prompt, gating,
   the number, the comparison, identity, and what the set is not.

## 1. The claim prompt, source by source

### 1.1 Sources that exist (all computable from Tom's files)

| source | what it guesses for the next-next token | cost per guess |
|---|---|---|
| head (`h`) | the trained block on the last hidden state, then the served head on a shortlist | about 2.6 ms (*estimated*, tom-head §5) |
| lookup (`a`) | the newest earlier place in prompt + reply where the last 3 (else 2, else 1) tokens occurred; propose what followed | microseconds |
| lookup, else head (`ah`) | the lookup's guess if it has one, else the head's | the head's cost only when the lookup has none |
| counted table, prompt bigram, self-successor, rescoring (`f`, `b`, `c`, `e`) | Tom §6 | small |

Tom's table has `f`, `b`, `c`, `e` on both prompts; none beats the lookup or the head where it matters (his
§6), so they are not carried further here.

### 1.2 The 31 steps of the official prompt (*host*)

One entry per step: `H` the head is right, `L` the lookup is right, `x` the lookup proposed something wrong,
`-` nothing or wrong head; the digit is the lookup's match length.

| | steps 1 to 31 |
|---|---|
| Yi (`head2117-tok`) | `--0 -x1 H-0 --0 -x1 H-0 H-0 H-0 --0 H-0 -x1 H-0 H-0 --0 --0 --0 H-0 --0 -x1 -x1 H-0 HL1 -L2 -x3 -x1 -x1 --0 Hx1 -x1 -x1 Hx1` |
| Llama-2 (`head3000-tok`) | `Hx1 H-0 -L1 -L2 -L3 -x3 --0 --0 -x1 -L1 -L2 -L3 HL3 -L3 -L3 -x3 --0 --0 --0 --0 -x1 HL1 -L2 -x3 -x1 --0 H-0 H-0 -x1 -L1 -L2` |

- **Yi's reply is its own words:** the lookup proposes 14 times and is right twice; the head is right 12 times.
- **Llama-2's reply repeats the prompt in runs:** the lookup proposes 21 times and is right 13 times, in four
  runs; the head is right 6 times and mostly elsewhere.
- A 1-token match is the lookup's weak case on both (Yi: 12 proposals, 1 right; Llama-2: 9, 4 right; at lengths 2 and 3 Llama-2 is 9 right of 12).

### 1.3 A chooser fixed on the training split

- **Data:** the 1,500 training prompts' greedy dumps (`train1500`: 224,805 / 236,138 steps). No head guesses are
  saved for training or held-out rows, so the head enters through Tom's held-out top-1 (0.377 / 0.471).
- **Lookup precision by match length, training prompts** (*host*):

  | match length | Yi: precision (share of steps) | Llama-2 |
  |---|---|---|
  | 3 | 0.622 (0.144) | 0.653 (0.183) |
  | 2 | 0.422 (0.105) | 0.474 (0.108) |
  | 1 | 0.172 (0.312) | 0.230 (0.290) |
  | head, held-out | 0.377 | 0.471 |

- **Rule T:** at each step use the lookup's guess if its match length has a training precision above the head's
  held-out top-1; else the head's guess. For both models that is "length 2 or 3". (Llama-2's length 2 clears the
  head by 0.003, which is inside noise; "length 3 only" is the cautious variant and is in the table below.)
- **Head choice by the same data:** the head with the best held-out top-1: `head2117-tok` (Yi), `head3000-tok`
  (Llama-2).

### 1.4 Results (*host*; tokens per pass; in brackets hits ÷ guesses made)

| rule | Yi, official | Yi, set | Llama-2, official | Llama-2, set |
|---|---|---|---|---|
| head alone | 1.391 (9/23) | 1.332 (888/2,649) | 1.143 (4/27) | 1.453 (1,134/2,486) |
| lookup alone | 1.032 (1/13) | 1.202 | 1.280 (7/14) | 1.185 |
| lookup if any, else head (`ah`) | 1.333 | 1.377 | 1.333 | 1.429 |
| lookup at length 3, else head | 1.391 | 1.378 | 1.231 | 1.483 |
| **rule T: lookup at length ≥ 2, else head** | **1.455** (10/22) | **1.391** (1,002/2,537) | **1.333** (8/24) | **1.486** (1,189/2,432) |
| either right (bound) | 1.455 | 1.461 | 1.391 | 1.564 |

With the other saved heads for Llama-2 (not the ones the training data would pick): rule T gives 1.391 on the
official prompt and 1.447 to 1.465 on the set.

- Rule T is at the bound on Yi's claim prompt and one hit short of it on Llama-2's.
- **`ah`, the rule in Tom's table, is worse than the head on Llama-2's set (1.429 against 1.453)** because it
  takes every 1-token match. Gating by length removes that loss and keeps the gain.
- The step counts differ from "6 hits of 31" in r23: a pass that commits two tokens makes no guess at the
  second, so fewer guesses are made than there are steps.

### 1.5 The guard

What was chosen on what:

| choice | chosen on | seen before choosing |
|---|---|---|
| the heads' weights | Dolly training prompts | – |
| which head per model | held-out top-1 (Tom) | Tom also printed set and official values in the same table |
| the lookup's form (3, 2, 1-gram, newest match) | Tom wrote it before any score (his tool's self-test) | – |
| **gating the lookup by match length** | the threshold: training precision against held-out top-1 | **I had seen the official prompt's result (r23) and Tom's set table** |

- So the threshold is clean and the idea is not. The set's improvement (§0 item 3) is evidence that the idea is
  not a fit to 31 guesses, but the set has now been looked at with this rule once.
- **Proposal for the pre-registration of the first round that scores a multi-token method:**
  1. The guess rule and the head are named with the training and held-out numbers that select them (the table
     of §1.3), and nothing else selects them.
  2. From then on neither the official prompt nor the set is used to change the rule, a threshold or a head.
  3. Both are reported: the official prompt (the claim) and the set (the check), with the same rule.
  4. If a later rule is proposed, it is selected on training data again and its set result is reported as a
     second look, labelled so.
- **For Tom, to remove the weak point of §1.3** (minutes, existing files): save the head's guesses on the 5 %
  held-out training prompts (`c3_head_train.py` writes them only for the evaluation labels today), so that the
  chooser can be scored end to end on held-out data instead of through one summary number. And the same dumps
  for DeepSeek once its head exists.

## 2. Price (*estimated*; r14's rung 6: C5 out, campaign 2's two-stage head, head knob, O 5.2, cg 1.26)

| subject, rule, prompt | tokens per pass | K 300, `linear` / `pair` | K 150 | K for 12.00, `pair` |
|---|---|---|---|---|
| Yi, head, set (the stacks so far) | 1.332 | 10.86 / 11.15 | 11.89 / 12.31 | 187 µs |
| Yi, rule T, set | 1.391 | 11.31 / 11.61 | 12.38 / 12.81 | 248 |
| Yi, head, official | 1.391 | 11.31 / 11.61 | 12.38 / 12.81 | 248 |
| **Yi, rule T, official** | 1.455 | **11.79 / 12.10** | 12.90 / 13.35 | **313** |
| Llama-2, head, set (the stacks so far) | 1.453 | 10.84 / 11.16 | 11.77 / 12.22 | 179 |
| Llama-2, rule T, set | 1.486 | 11.07 / 11.40 | 12.03 / 12.48 | 214 |
| Llama-2, head, official | 1.143 | 8.60 / 8.85 | 9.34 / 9.70 | none |
| **Llama-2, rule T, official** | 1.333 | **9.97 / 10.27** | 10.84 / 11.25 | **51** |

- The lookup's guesses cost no head evaluation (rule T uses it at about a quarter of the steps on training
  data); the table charges the head's 2.6 ms at every guess. Crediting it would add about 0.05 to 0.1 tok/s.
- r15's corrections (O, the second position's attention) are not in this table; they take about 0.2 / 0.45
  tok/s off (r15, *estimated*).
- **Reading:** on the claim prompt Yi's priced lane now has a line for 12 inside R75's expected band (K ≤ 313 µs
  in `pair`), and Llama-2's does not (K ≤ 51 µs). The three-subject goal is still set by the weakest subject on
  the claim prompt, which is now Llama-2 by this term, with DeepSeek unknown.

## 3. A proposed text on scoring the fixed prompt set (for the owner, through the lead)

The terms say a multi-token method "is scored on a fixed prompt set as well as the official prompt" and say no
more. Proposed addition, in the terms' own style:

> **Fixed prompt set (proposal).** The set is `prompts/eval.txt` (32 prompts, sha256 `ff1f32e3…72c0`), fixed by
> the rule in its README; nothing is tuned on it.
> 1. A method is run on it with the same runtime settings as on the official prompt, 128 new tokens per prompt
>    (fewer if the model ends its reply), context 1,024, greedy.
> 2. Each prompt starts after the kit's cool-down gate, as an official run does. If a full pass is too long for
>    one session, a subset fixed in the pre-registration (the first 8 in file order) is used and named.
> 3. The score of an arm is its committed tokens divided by its decode seconds, summed over the prompts; the
>    median of the per-prompt rates is reported beside it.
> 4. The method's arm and the stock arm run in the same session in ABBA order. **The method passes the set if
>    its score is at least the stock arm's.** The set has no tok/s line of its own; the 12 tok/s claim is on the
>    official prompt.
> 5. The method's reply text must equal the stock arm's on every prompt (greedy identity). A prompt where it
>    differs fails the round for that method.
> 6. Guess rules, heads and thresholds are chosen on training and held-out data only, named with those numbers
>    in the pre-registration; the official prompt and the set are read, never used to choose.

Why each line:
- 1 and 3: the host numbers behind every prediction were made at 128 tokens and as tokens per pass over all
  rows; a different statistic on the phone could not be compared with them.
- 2: 32 gated prompts are 30 to 95 minutes per arm (r23, *estimated*); the subset keeps the rule honest when
  time is short.
- 4: without a comparison the set is a number with no meaning; with a 12 tok/s line it would be a second goal
  the owner did not set. "Not slower than stock on text it was not tuned on" is the property the set exists to
  check.
- 5: the terms already require identical greedy output; this says where it is checked.
- 6: §1.5.

Open in the text, for the owner or the lead to choose: whether a failed set (line 4) voids a claim made on the
official prompt, or is reported beside it. I would say it voids it: a method that is slower than stock away from
the claim prompt has not met "generalises".

## 4. Next questions, ranked

1. **Fold R75 into r18's draft** when its report is in the repository.
2. **The same analysis for DeepSeek** as soon as its dump and head exist; until then every three-subject line
   rests on an assumed 1.385.
3. **Does the lookup carry to width 3 or to a longer copy?** Where Llama-2 copies a run, the next several
   tokens are known at once (a 3-token match usually continues). A copy-run pass that commits more than two
   tokens is a different lever from a guess head, model-free and exact. Price it from the same files: the
   distribution of copy-run lengths on training prompts, the set and the claim prompt. Mine.
4. **Re-read r15 to r17 against Tom's host decode and KV screen** when they exist.
5. **Cold read rate; fixed share.**

## 5. Limits

- Two models, 31 claim-prompt guesses each. The training-split precisions are solid (hundreds of thousands of
  steps); the claim-prompt columns are what they are.
- The head enters rule T's selection through one number per model (held-out top-1), not per step.
- Rule T's idea was formed after seeing claim-prompt results (§1.5).
- §2 changes one input of r14's rung 6 and inherits the rest.
- The lookup here is my re-implementation of Tom's arm `a`; its numbers reproduce his `ah` rows on the set to
  within 0.001 to 0.002 where the heads match (1.377 / 1.429 against his 1.37 / 1.43).

## 6. Sources (2026-11-24)

Repository: `reports/memo-09-work/tom-head.md` (§4.1, §4.2, §6); `tools/c3_head_t1.py` (arm definitions, lines 65
to 104); `lab/c3/head/{yi,llama2}/train1500.{rows,ptok}`, `eval.{rows,ptok}`, `official.{rows,ptok}`,
`head*.pt.{evalset,official}.pred.npy`; `campaigns/03-dense15-5pct/CAMPAIGN.md` (metric row);
`reports/memo-10-work/clyde-r73-plan.md` §6; my r14, r15, r23.
