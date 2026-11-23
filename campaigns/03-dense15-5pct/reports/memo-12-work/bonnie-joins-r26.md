# c3 memo-12 work, round 26 (Bonnie): is the official prompt a fair claim prompt for a multi-token method? A note for the lead and the owner

- **Date:** 2026-11-24. **Author:** bonnie, `claude-fable-5-2`. Follows [bonnie-joins-r25.md](bonnie-joins-r25.md). Not
  committed. No verdict here; the terms are the owner's. **Ebony has not reviewed this file.**
- **Labels.** *host* = from Tom's dumps on the laptop (r23 to r25); *estimated* = arithmetic. Nothing new was
  measured this round; I printed the two dumped replies with Tom's `tools/c3_head_show.py`. No web source.

## 0. The note in five lines

1. The terms put the claim on one prompt and 32 tokens, and add the fixed set only as "scored … as well".
2. That prompt is one instruction and **the same sentence six times**. For a single-token runtime its content
   does not matter. For a method that commits several tokens per pass it decides the result.
3. On it, a subject's tokens per pass range from 1.03 to 1.46 by guess source, and the ranking of sources
   reverses between the two subjects that have data (*host*).
4. It is fair in the sense that it was fixed long before any of this and is the same for every method. It is
   not a fair measure of whether a method works in general. The set is what measures that.
5. **Proposal: keep the claim where the terms put it, and make the set a condition** (the method is not slower
   than stock on the set, same session, identical text), with guess rules chosen on training data only. One
   paragraph for the terms is in §4.

## 1. What the terms say, word for word

`campaigns/03-dense15-5pct/CAMPAIGN.md`, terms table:

> | metric | llamadart `decodeTokensPerSec`, official prompt, 32 tokens, greedy, kit protocol (r16 gate) |
> unchanged. A method that drafts or commits several tokens per step counts committed tokens, keeps greedy
> output identical, and is scored on a fixed prompt set as well as the official prompt |

> | claim rule | median of 10 plain runs ≥ 15.0 tok/s (*note 2026-11-21: read 12.0, per model, per Amendment 1*),
> positions fixed before the run, never a best run; power saving off, deep-sleeping list state recorded, charger
> out | campaign 2's go rule (memo-31 §4) applies before a claim round |

Amendment 1 (owner, 2026-11-21): "Change the goal to 12 tok/s, run three different dense 7B models on the same
chipset, and the goal is met when all three pass."

The owner's decision on trained guess heads (2026-11-22), in the lead's reading: "The head only proposes; the
model verifies. Greedy output must stay identical, checked on the fixed prompt set and the official prompt."

The R73 plan, §6 (lead): the set is "used for every multi-token method, beside the official prompt".

**What this fixes:** the prompt, the token count, greedy decoding, the run protocol, ten runs, the median, 12.0
per model. **What it does not fix:** what "scored on a fixed prompt set as well" requires. No number, no
comparison, no consequence.

## 2. What the official prompt is

`tools/c2_official_prompt.txt` (702 bytes; the app's `_benchPrompt`):

> Read these notes, then write one paragraph about why the ocean is salty.
> Note 1: rivers carry dissolved minerals from rocks into the sea, and evaporation leaves the salt behind.
> Note 2: rivers carry dissolved minerals from rocks into the sea, and evaporation leaves the salt behind.
> … (Notes 3 to 6: the same sentence)

It was chosen in campaign 1 to time a decode: long enough to give a context of 150 to 190 tokens, short enough
to run in seconds. Campaigns 1 and 2 measured single-token runtimes, for which the reply's words do not matter:
every token costs one pass.

The first 32 tokens of the two replies we have (*host*, greedy, the served files):

| subject | reply |
|---|---|
| Yi | "The ocean's saltiness can be attributed to the continuous process of mineral-rich water from rivers flowing into the sea and the subsequent evaporation of water, leaving" |
| Llama-2 | "The ocean is salty because of the dissolved minerals from rocks that are carried by rivers into the sea. Through a process of evaporation" |

Yi paraphrases the notes. Llama-2 reuses their phrases.

## 3. What varies on it once a pass can commit two tokens (*host*; tokens per pass over the 32 tokens)

| guess source (all exact: a guess is kept only if the model produces it) | Yi | Llama-2 | the same on the 32-prompt set, Yi / Llama-2 |
|---|---|---|---|
| none (stock) | 1.00 | 1.00 | 1.00 / 1.00 |
| prompt lookup alone | 1.03 | 1.28 | 1.20 / 1.19 |
| the trained head alone | **1.39** | **1.14** | 1.33 / 1.45 |
| lookup when its match is 2 or 3 tokens long, else the head (r24's rule, fixed on training data) | 1.46 | 1.33 | 1.39 / 1.49 |

- **The same head is the better source on one subject and the worse on the other**, because one reply
  paraphrases and the other copies. On the set the head leads on both.
- **31 guesses decide it.** One hit more or less moves tokens per pass by about 0.03 and the claim by about
  0.25 tok/s near 11 (*estimated*).
- Because decoding is greedy and the prompt fixed, the ten claim runs repeat the same 31 guesses. The median of
  ten runs averages the phone's state, not the text.
- In tok/s (*estimated*, r24 §2, lane at K 300): Yi 11.3 to 12.1, Llama-2 8.6 to 10.3, by source. On the set
  both subjects sit at 10.8 to 11.6. **The claim prompt spreads the subjects apart; the set does not.**
- DeepSeek has no head and no dump; where it falls is unknown.

So:

| question | answer |
|---|---|
| Is the prompt neutral between methods? | No. Six identical sentences reward copying when the reply copies, and say nothing when it paraphrases |
| Is it neutral between subjects? | No. It depends on each model's phrasing of one paragraph |
| Can a method be fitted to it? | Yes: 31 guesses, known in advance. r24's guard (choose on training data only) is what prevents that |
| Was it chosen to favour anything? | No. It predates every multi-token idea in this campaign, and it is the same for all |
| Does a pass on it show that a method works on other text? | No. That is what the set shows |

## 4. Options, and the one I propose

| option | what changes | for | against |
|---|---|---|---|
| A. Leave the terms as they are | nothing | simplest; the benchmark's definition is untouched | "scored … as well" has no consequence: a method could pass the claim by its fit to one paragraph and be slower than stock elsewhere |
| **B. Claim on the official prompt; the set is a condition** | one paragraph (below) | keeps the benchmark and its history comparable; makes "generalises" a tested property; costs one more block per subject | a longer round (r23: a gated pass of the set is 30 to 95 minutes per arm at 128 tokens; a pre-registered 8-prompt subset is the fallback) |
| C. Claim on the set | the metric row | measures the general case | changes the benchmark's definition against "unchanged"; numbers stop being comparable with campaigns 1 and 2; a 128-token run is a hotter regime than the claim's |
| D. Another official prompt | the prompt | removes the six repeated sentences | the same objection as C, and any single prompt has the same weakness at 31 steps |

**Proposed paragraph for the terms (option B), for the owner to accept, change or refuse:**

> **Multi-token methods.** The claim stays on the official prompt as written. A method that commits several
> tokens per pass must also be run on the fixed prompt set in the same session as the stock runtime, in ABBA
> order, with greedy output identical to stock on every prompt. The claim holds only if the method's rate on
> the set (committed tokens ÷ decode seconds over the set) is at least the stock rate. Guess rules, heads and
> thresholds are chosen on training and held-out data only and are named with those numbers in the
> pre-registration. The result reports tokens per pass on the official prompt and on the set side by side.

What it does and does not do:
- It does not move the goal. 12.0 is still read on the official prompt, per model.
- It does not ask the set for 12.0. The set is a hotter, longer run (r23); a second threshold there would be a
  new goal.
- It closes the gap named under option A.
- It leaves open one choice for the owner: whether the set's condition is "not slower than stock" (my
  proposal) or something stronger, such as "the method's gain on the set is at least half its gain on the
  official prompt". The stronger form would catch a method that is neutral on the set and large on the claim
  prompt; by today's numbers no source is in that corner (the set gains are the larger ones on Llama-2), so I
  would not add it.

One sentence I would also put in any result that passes: how much of the claim's gain came from guesses copied
out of the prompt and how much from the head, since on this prompt the two are very different things.

## 5. Next questions, ranked

1. **Fold R75 into r18's draft and into r25's script** when its report is in the repository.
2. **DeepSeek's reply to the official prompt, and its dump:** which side of §3 it falls on decides the
   three-subject picture more than any kernel. Tom's H2 dump plus a head.
3. **How long a set block really is on the phone,** per subject, with the gate: r23's 30 to 95 minutes per arm
   is an estimate from R74's gate series; R75's gates after lane-like load refine it. Mine, after R75.
4. **Re-read r15 to r17 against Tom's host decode and KV screen** when they exist.
5. **Cold read rate; fixed share.**

## 6. Limits

- Two subjects have data; DeepSeek has none. "The ranking reverses between subjects" is a statement about two.
- §3's tokens per pass are 31 guesses each, from saved guesses on the laptop; the phone's greedy text could
  differ from the laptop's in a token (different kernels), which would change the 31 steps.
- The tok/s ranges are r24's estimates and carry everything r14 lists.
- Whether option B is acceptable is the owner's question; I price it, I do not decide it.

## 7. Sources (2026-11-24)

Repository: `campaigns/03-dense15-5pct/CAMPAIGN.md` (Amendment 1, lines 10 to 15; terms table, lines 51 to 52;
owner decision 2026-11-22, lines 64 to 72); `campaigns/02-25toks-5pct/CAMPAIGN.md` line 25;
`reports/memo-10-work/clyde-r73-plan.md` §6; `tools/c2_official_prompt.txt`; `lab/c3/head/{yi,llama2}/official.rows`
printed with `tools/c3_head_show.py`; `reports/memo-09-work/tom-head.md` (§4, §6); my r23, r24, r25.
