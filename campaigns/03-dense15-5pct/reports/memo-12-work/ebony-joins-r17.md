# c3 memo-12 work (Ebony, round 17): what sets and what lifts the CPU caps around a dense prefill

Date: 2026-11-24. Author: Ebony (`claude-fable-5-2`). Not committed; the lead commits. No verdict here.
Follows [ebony-joins-r16.md](ebony-joins-r16.md). Read for it: [bonnie-joins-r31.md](bonnie-joins-r31.md) §0 and §1.1,
[tom-kai-stock.md](tom-kai-stock.md) in full.

Labels as before. Triples are Yi / Llama-2 / DeepSeek. New readings of existing *device* data: the cap probe of
2026-11-23 second by second, and the cell at which each cap step arrived in R72's and R74's blocks. Nothing ran.

## 0. Answer first

1. **The caps are a ladder that the phone walks every 10 s, and they come back in one jump.** In the cap probe
   (*device*): from full clocks on a warm phone (headroom 0.76), three spinning threads were at 2,150 MHz after
   10 s, 1,785 after 20 s, 1,670 after 30 s and 1,555 after 40 s. The 2,150 rung appears at thermal status 0
   (once with nothing running); the deeper ones come with status 1. When the status returns to 0 the caps go
   back to 2,419 / 2,841 at once, even under load.
2. **From a gated start the first step takes longer: 17 to 36 s of multi-core load** (*device*, four blocks of
   R72 and R74; §1.2). A dense claim run is a prefill of 6 to 15 s and a decode of 3 to 4 s: 9 to 19 s.
   **So I do not expect dense runs to sit at 1,555.** My odds for a gated run: no cap during the decode 0.55;
   2,150 at some point 0.30; 1,785 0.12; 1,555 0.03. Later in a ten-run block, when gates time out near 0.70,
   the first row falls to about 0.4 and the rest move up one step.
3. **This is where I differ from Bonnie r31** (her P 0.6 for "cap below maximum in most runs", C nearer 34 than
   52). Her precedent, R72's L4 cells at 1,555, were cells 18 and later of a block that had run 67 s without a
   gate; the first nine cells of the same block ran at full caps. Both positions are in §1.3. R73 or any
   model-loading round settles it with one field per run.
4. **The one lever that removes the problem is not a kernel: it is reusing the prompt's cache between runs.**
   The official prompt is identical in every run. With the app library's prefix reuse on, runs 2 to 10 decode
   one prompt token instead of 150 to 190, and the big cores start the decode as cold as the gate left them.
   It is a runtime feature, not a phone setting; but the kit has it off by design (as r16), so it changes the
   protocol and is **an owner question**. Worth the whole cap loss: +0.15 tok/s by my odds, +0.5 by Bonnie's
   (*estimated*). §2.
5. **Everything that stays inside the rules without a decision is small:** prefill without the X1, a paced
   prefill, a lane prefill: +0.03 to +0.10 tok/s each (*estimated*), and the first two lengthen the wait for
   the first token. Bonnie's self-timing gate (r31 item 8) is the right exact answer to whatever cap state a
   run finds; it adapts instead of preventing.
6. **Tom's first stock result is inside my band.** Yi at `-b 512` on a clean KleidiAI build: +0.50 % (*host*).
   My r7 band, committed before the run: +0.4 to +1.5 %; the r10 revision: +0.3 to +1.2 %. Llama-2 and DeepSeek
   are not run yet. One note on his node check in §4.
7. **BitLLM, now read in full** (my next question 2): small models on newer Adreno and Mali GPUs; nothing in it
   changes a price here. §5.

## 1. What is known about the caps

### 1.1 The cap probe (*device*; `device/r74-capprobe-r34-2026-11-23T1736.json`, campaign 2's T-A design: one sample per second for 360 s; one spinner thread in 0 to 60 s, 120 to 180 s and 240 to 300 s; from 180 s to the end three threads spin on cpus 4 to 7)

Every cap change in the 360 samples (they all fall on a 10-second grid, so the phone's thermal service acts
every 10 s):

| t, s | A78 / X1 cap, MHz | thermal status | headroom | running at that moment |
|---|---|---|---|---|
| 0 | 1,420 / 1,420 | 3 | 0.97 | one spinner; the phone arrives hot |
| 14 | 1,555 / 1,555 | 2 | 0.86 | one spinner |
| 44 | 1,555 / 1,555 | 1 | 0.82 | one spinner |
| 94 | **2,419 / 2,841** | **0** | 0.78 | nothing (rest minute): caps restored in one step |
| 104 | 2,150 / 2,150 | 0 | 0.75 | **nothing** |
| 124 | 1,785 / 1,785 | 1 | 0.95 | one spinner since 120 s |
| 134 | 1,670 / 1,670 | 1 | 0.96 | one spinner |
| 184 | **2,419 / 2,841** | 0 | 0.76 | three threads since 180 s: restored again |
| 194 | 2,150 / 2,150 | 0 | 0.76 | three threads |
| 204 | 1,785 / 1,785 | 1 | 0.89 | three threads |
| 214 | 1,670 / 1,670 | 1 | 1.03 | three threads |
| 224 | 1,555 / 1,555 | 1 | 1.00 | three threads |
| 354 | 2,419 / 2,841 | 0 | 0.77 | **three threads still spinning** (and a fourth until 300 s) |

Reading:
- **A ladder, one rung per 10 s:** 2,150 → 1,785 → 1,670 → 1,555 (1,420 was seen at status 3). From the first
  rung on, the X1 is capped to the A78's value (2,841 → 2,150 is −24 %; 2,419 → 2,150 is −11 %).
- **Release is all at once,** at thermal status 0 with headroom near 0.77.
- **Under steady three-thread load the caps cycle:** 40 s down the ladder, about 130 s at 1,555, then a full
  release while the load is still on (354 s). At 1,555 three threads make little enough heat for the phone to
  cool to status 0.
- **The rung at 2,150 is not only a matter of load.** Twice it followed a release by 10 s, once with nothing
  running (104 s) and with headroom falling. So it belongs to a lower temperature line than status 1, and a
  phone that has just come back from a cap sits between the two lines. The deeper rungs came with status 1.
- This probe started hot (status 3) and never went below headroom 0.75. It shows the mechanism, not a claim
  run: a gate at 0.65 starts lower. §1.2 is the record for that.

### 1.2 From a gated start (*device*; the cell at which each cap first appears; cell lengths from the block's time ÷ cells, so each figure is good to a few seconds, and cells include ramps and buffer builds)

| block | load | headroom at the gate: start → end | first 2,150 | 1,785 | 1,555 |
|---|---|---|---|---|---|
| R72 `l4` | 3 and 4 threads, layer sweeps | 0.73 → 0.64 | cell 10 of 63, about **36 s** | cell 13, 47 s | cell 18, 67 s; it stays there |
| R72 `l23` | 3 and 4 threads | 0.96 → 0.65 | cell 6 of 14, about **21 s** | – | – |
| R72 `l7` | 3 threads on the head (lighter duty) | 0.70 → 0.65 | cell 13 of 18, about 75 s | – | – |
| R72 `l1` | one thread | 0.86 → 0.65 | never in 523 s | – | – |
| R74 `x3c` | 4 threads and the GPU | 0.90 → 0.65 | about **17 s** | 25 to 35 s | – |
| R74 `x3g` | the GPU alone | 0.71 → 0.65 | never in 32 s | – | – |
| campaign 2's claim-format runs (Bonnie r31 item 4) | 3 threads, about 3 s in all | gated | caps at maximum before and after 287 of 312 runs | | |

- **A gate buys 17 to 36 s of multi-core load before the first step,** less when the phone was hot before the
  gate (`l23`, `x3c`) and less with the GPU on.
- One thread never trips it. A light three-thread duty takes more than a minute. The GPU alone does not trip
  the CPU's caps in 32 s.
- External: nothing found on this phone's thermal steps or their timing (one search today; reviews give
  long-run averages only).

### 1.3 What that means for a dense claim run, and where Bonnie and I differ

A run from the gate: prefill 6 to 15 s on four big cores, then 3 to 4 s of decode (CPU, or CPU and GPU with a
lane). Total 9 to 19 s against an onset of 17 to 36 s.

| state during the decode | Ebony, early in a block | Ebony, late in a block (gates timing out near 0.70) | Bonnie r31 |
|---|---|---|---|
| full caps throughout | 0.55 | 0.40 | 0.4 ("cap below maximum in most runs, P 0.6") |
| 2,150 at some point | 0.30 | 0.35 | – |
| 1,785 | 0.12 | 0.18 | – |
| 1,555 | 0.03 | 0.07 | "C nearer 34 than 52", which is the 1,555 state |

- We agree on how often a cap appears. **We differ on its depth.** 1,555 needs about 40 s past the first step
  or a start at status 1; a gated 9-to-19 s run does not get there unless the gate itself failed.
- Her precedent (R72's L4 body cells at 1,555) is real but belongs to minute two of an ungated block (§1.2).
- A slow prefill feeds itself: once capped, the prefill takes up to 1.5 × as long (it is compute-bound), which
  adds heat before the decode. That is the tail in my table.
- In her terms (r31 item 2): full clocks are C 52 to 59, the 1,555 state C 32 to 34. By the clock ratios the
  2,150 state is about C 42 to 46 (*estimated*), which is the stacks' old 40.
- **Price of the cap state as it stands** (*estimated*; her r31 item 3 gives the width-2 ratio at the two ends;
  I take the body as 80 % of a token and 2,150 as a third of the way to 1,555): a run at 2,150 loses about 0.2
  tok/s on the priced lane and 0.4 on the W-kernel designs; at 1,555 about 0.6 and 1.3. Weighted by my odds:
  **−0.15 to −0.3 tok/s on the block's mean**; by hers, −0.4 to −0.8. A width-1 pass loses nothing while all
  four threads are on their cores (her item 1).
- What settles it: `max_mhz` at the first and last decoded token of each official run. The kit's lines carry
  the field already.

## 2. What lifts or holds the caps, and what the rules allow

"Inside the rules" here means: a change in the runtime, not in the phone's settings, that needs no new decision.
"Owner" means it changes the benchmark's protocol or what a user waits for.

| # | lever | mechanism | rules | effect on the cap odds | price on the claim block (*estimated*) |
|---|---|---|---|---|---|
| 1 | **Reuse the prompt's cache between runs** | llamadart's `reusePromptPrefix`: the official prompt is the same every run, so from the second run on the prefill is one token (`cached_prompt_tokens` is already a report field). The kit sets `reusePrefix = false` (`research.dart` line 102, "as r16") | **owner**: a runtime feature, but it changes what the benchmark runs before the timed decode | the decode starts at the gate's state: full caps in about 0.9 of runs | recovers the whole loss of §1.3: **+0.15 to +0.3 tok/s by my odds, +0.4 to +0.8 by Bonnie's**; each run 6 to 15 s shorter, so shorter gates too |
| 2 | A pause between prefill and decode | wait until the thermal status is 0 and headroom is back under the gate's line | **owner** (r5 §1.3): the user waits | as lever 1 if the pause is 30 to 90 s (§1.1: recovery is one jump, but only at status 0) | as lever 1, at a longer wait for the first token |
| 3 | Prefill without the X1, or on two cores | the same kernels on fewer cores; the X1 is the least efficient core per unit of work | inside (bit-identical); campaign 2 memo-21 N6 called the longer wait an owner matter | energy before the decode falls 10 to 20 %; time rises 25 to 100 % | +0.02 to +0.05 |
| 4 | A paced prefill (work, rest, work) | `l7` shows a lighter duty takes 75 s to the first step where full duty takes 21 to 36 s | inside (bit-identical); the wait for the first token doubles | the prefill itself stops tripping the step; the decode still starts on a warm chip | +0.05 to +0.10 |
| 5 | A lane prefill (CPU and GPU together) | r5 §1.1, design P2: 1.4 to 1.6 × faster, so less time under load | numerics change in the prefill (r5 §1.4) | a few seconds less load | +0.02 to +0.06 |
| 6 | A GPU-only prefill | r5: the GPU alone does not trip the CPU's caps (`x3g`), but it is no faster and the chip is as warm | numerics change | r5: P(full caps) 0.55 → 0.65 | +0.03 to +0.10 |
| 7 | Fewer threads in the decode when a cap is on | at one position three A78 read as much as four threads while uncapped (27.75 against 26.67 GB/s, *device*, R72 `l2`); the fourth thread matters only under a cap or at width 2 | inside (bit-identical) | lower power in the decode | under +0.05; folds into lever 8 |
| 8 | Bonnie's self-timing gate (r31 item 8) | the runtime times its own width-1 and width-2 passes and chooses | inside (exact) | none: it adapts to the cap | it turns a capped run's loss at width 2 into the smaller width-1 loss; hers to price |
| 9 | A stricter gate (0.60 instead of 0.65) | a cooler start | **owner**: the r16 gate is part of the metric | onset later by an unknown amount; many more time-outs (r4 §4) | not priced |
| 10 | Android's performance-hint API | asks for clock and placement | inside | none on a thermal cap (memo-11 §3.6) | 0 |
| – | cooling the phone, a charger rule, a case | – | outside: the user's phone | – | – |

Reading:
- **Levers 1 and 2 are the only ones that change the picture, and both are the owner's.** Lever 1 is the
  cleaner of the two: no wait, nothing new to build (the switch exists), and a chat app really does keep the
  cache between turns. What it changes is the benchmark: campaigns 1 and 2 timed the decode after a full
  prefill of the prompt, and the terms say "kit protocol (r16 gate)". If it were allowed, every earlier
  number would need "prefix reuse off" beside it for comparison.
- Levers 3 to 7 together are worth at most 0.1 to 0.2 tok/s and two of them cost the user seconds.
- Whatever is chosen, the first model-loading round should print the cap at the first and the last decoded
  token, the prefill's seconds, and `cached_prompt_tokens`, per run.

## 3. Lists (additions only)

| # | test | where | cost | closes when |
|---|---|---|---|---|
| E-T29 | per official run of the first model-loading round: `max_mhz` at the first and last decoded token, prefill seconds, thermal status before and after | device, free (fields exist or are one line) | 0 | full caps in ≥ 7 of 10 runs per subject → the cap is a tail, C 52 to 59 holds, and levers 1 to 7 are not needed; 1,555 in ≥ 3 of 10 → Bonnie's reading holds and lever 1 goes to the owner with numbers |
| E-T30 | one extra arm in that round, if the owner allows the question to be tested at all: the same ten runs with prefix reuse on | device | one more block per subject | the difference in decode tok/s between the arms is lever 1's price, measured |

| # | try | after | label | price | P |
|---|---|---|---|---|---|
| E-Y6 | prefix reuse in the claim protocol | the owner's answer; E-T29 | bit-identical (the same tokens are decoded from the same cache) | +0.15 to +0.8 tok/s depending on whose cap odds hold | 0.3 that the owner allows it |
| E-Y7 | paced prefill | E-T29 showing caps in ≥ 3 of 10 runs and lever 1 refused | bit-identical | +0.05 to +0.10 | 0.3 |

## 4. Tom's stock result and node check

- **Yi, `-b 512`, 40 chunks, clean KleidiAI build: 7.7347 against 7.6966, +0.50 %** (*host*). Inside my r7 band
  (+0.4 to +1.5 %, the one that is scored) and the r10 revision (+0.3 to +1.2 %). The partial `-ub 1` reading
  (+0.56 % at chunk 9) is not a score. By r9 §1.2 the two settings should be **identical per chunk** on that
  build only with the Q8_0 cache; Tom's runs use the default F16 cache, so they need not be.
- His build is the one r8 asked for (the freeze's flags, BLAS off, KleidiAI the only difference).
- **His E-T13 run adds something:** with clean flags and 16 tokens, all of layer 0 is bit-equal between the
  builds, attention included. In my r8 run on the flawed libraries the attention of layer 0 already differed.
  So the 7e-5 I saw there was the flags, as I wrote. His first difference is the attention output of layer 1
  (6e-4), with layer 1's q, k and v equal **in the last column**. As he says, the tool sees one column; the
  difference must enter through another position's K or V, or through layer 0's output at another position.
  The tool change of r9 §1.3 (compare every column) would name the node. It matters because it is the first
  sign of where the two paths part in the body, if they do: 6e-4 is the size of a half-precision rounding of a
  cached value, not of a float reordering.

## 5. BitLLM in full (my next question 2; *external*, arXiv 2611.00547v1, HTML through the fetch tool, the first 100,000 characters)

- Devices: Snapdragon 8 Gen 3 (Adreno 750), 8 Gen 2 (Adreno 740), Tensor G3 (Mali-G715). Models of 1.2 to 2.8 B
  parameters. Token rates are in figures only.
- KV layout: which axis is contiguous depends on the context length (the head dimension for under about 256
  tokens, the token dimension for over about 512), with append-only segments. Its layout step gives 1.34 to
  1.37 × at 4-bit weights on those models.
- Integer-native kernels use the GPU's 8-bit dot-product instruction and defer the conversion to float to the
  end of a tile.
- Nothing on throttling or on the cost of CPU and GPU hand-offs; the whole graph runs on the GPU.
- **For us:** it is a whole-graph GPU engine on small models, where attention is a large share of a token. On a
  7 B body the attention is about 5 % of a token, so its layout gain does not carry over as a ratio. It
  supports measuring the KV walk (E-T18) and nothing more. No price changes.

## 6. Next questions, ranked

| # | question | state |
|---|---|---|
| 1 | The owner's answers: DeepSeek's prompt, Yi's marker, and now prefix reuse (§2 lever 1) | with the lead |
| 2 | Tom's remaining five stock runs (Llama-2 ±0.05 % and DeepSeek +0.3 to +1.2 % are the open halves of the prediction), and his node check with every column compared | Tom |
| 3 | Which temperature line sets the 2,150 rung (it appeared at status 0, once with no load), and at what headroom does a gated phone sit relative to it? | *device* data only: the headroom at each run's start against whether 2,150 appears, from campaign 2's reports (they hold both) and R75's `x4 therm` rider; next for me |
| 4 | Any record of a host-signalled Vulkan event on Adreno 6xx | one search; low odds |
| 5 | R75 free reads; E-T18 | R75; a later kit |

## 7. Sources (2026-11-24)

Repository: `device/r74-capprobe-r34-2026-11-23T1736.json` (all 360 samples), `device/r72-c3probe-r33-…json` and
`device/r74-c3probe-r34-…json` (`max_mhz` per cell, block times); `lab/app-c3/lib/research.dart` (lines 102, 459 to
469, 1650 to 1651); campaign 2 `memo-21-unconventional-levers.md` (N6); Bonnie's r31; Tom's `tom-kai-stock.md`.

External: BitLLM, arXiv 2611.00547v1 (<https://arxiv.org/html/2611.00547v1>), read through the fetch tool. One search
for this phone's thermal steps: reviews only (GSMArena, AnandTech, Notebookcheck summaries), no step timing.
