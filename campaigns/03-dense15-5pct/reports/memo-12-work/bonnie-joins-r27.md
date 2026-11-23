# Bonnie, memo-12 work, round 27: review of Ebony r8 to r11; the per-step chooser (E-T21) as a rule family

2026-11-24. Labels: *device* (phone), *host* (laptop, relative only), *estimated* (arithmetic), *source* (code
read). No device claim is made here. No web search was needed (nothing is stated about upstream's current state).
State of the inputs I wait for: R75's report and `tom-kai-stock.md` are not in the repository.

Script: `bonnie_chooser_r27.py` (same folder; reads Tom's dumps and saved heads, runs no model; about 1 minute).
**It reads only the trainer's held-out training prompts. The fixed prompt set and the official prompt were not
read in this round.**

## 0. Answer first

1. **Ebony r8 to r11: I agree with every main claim.** Seven independent checks, all confirm (§1.5). Three places
   where I differ in size, none in direction:
   - r11's "3.4 standard deviations, not chance" for Llama-2's 6 hits of 31 uses the binomial spread. Hits within
     a prompt are correlated: on the 144 held-out training prompts the spread of hits in the first 31 steps is
     5.06, not 2.78 (*host*). Six is **2.0 standard deviations below, and 2.1 % of held-out prompts do as badly.**
     The official prompt is a hard text for that head, one prompt in fifty, not one in three thousand.
   - r10's band for the key-major walk (0 to 5 ms) is too wide at the top. My band: **0 to 3 ms, centre under
     1 ms** (*estimated*, §1.3).
   - r8's "the emulation's mean will be inside ±0.1 % at 40 chunks" is a one-sigma statement by her own numbers
     (per-chunk ±0.5 %, 40 chunks: 0.08 %). A pass line should be ±0.2 %.
2. **The chooser: built as a five-number rule family, fitted and cross-checked on held-out training data. It is
   worth little over rule T.** (*host* counts, *estimated* rates; §2.3)

   | | head always | rule T (r24) | chooser, cross-fitted | chooser over rule T |
   |---|---|---|---|---|
   | Yi, tok/s | 10.78 | 11.07 | 11.09 | **+0.14 %** (folds −0.27 to +0.62) |
   | Llama-2, tok/s | 10.59 | 10.89 | 10.99 | **+0.95 %** (folds +0.61 to +1.45) |

   (`pair` model; `linear` gives +0.20 % and +1.26 %.) In tokens per pass: Yi 1.370 → 1.368, Llama-2 1.456 →
   1.467. **By Ebony's own closing line for E-T21 (under +0.02 tokens per pass → no chooser) it closes on both
   subjects.** By my rate line (§2.4) it fails on Yi and is borderline on Llama-2.
3. **What the fit did confirm is rule T itself, on prompts it was not derived from in this form:** +2.7 % on Yi
   and +2.8 % on Llama-2 over the head alone (*estimated* rate from *host* counts, held-out prompts).
4. **"No guess" is worth almost nothing at the priced pass costs.** A width-2 pass costs 1.09 to 1.15 of a
   width-1 pass (*estimated*), so a guess pays above a hit probability of 0.11 to 0.18. Only the head's lowest
   confidence decile (precision 0.09 to 0.10) is below that. The gate becomes a real lever only if the phone
   shows the second position to cost much more than priced: **at P2 ÷ P1 of 1.20 or more a one-number gate on
   rule T is worth +1 to +2.5 %, and width 2 stops paying at all near 1.33 (Yi) and 1.42 (Llama-2)** (§2.5). The
   stack model gives 1.09 to 1.15 with the batched GPU kernel and 1.18 to 1.25 with two launches.
5. Recommendation for the lead: **keep rule T as the guess rule; do not spend Tom's time on E-T21 beyond the
   column his held-out dump already needs.** The family and its fitting procedure are written in §2.4 so that
   it can be pre-registered unchanged if a later device number changes point 4.

## 1. Review of Ebony r8 to r11

### 1.1 r8 (attention divergence, the build-flag trap, the floor)

| her claim | my position | check |
|---|---|---|
| the `-ub 1` against `-b 512` difference inside one build comes from the attention op: tiled float path at 64 or more queries, per-query half accumulation below | **agree** | *source*: `ops.cpp` line 9566, `kv_is_f32_or_f16`; `common.h` line 9, `GGML_FA_TILE_Q 64`. My own simulation of a half against a float weighted sum over 64 rows of 128 values: median difference 1.7e-3 of the largest value (10 to 90 %: 1.0e-3 to 2.6e-3, *host*). Her measured 1.4e-3 is inside |
| `lab/c3/head/build-kai` differs from the freeze build in three settings, not one | **agree** | *source*, both `CMakeCache.txt`: `GGML_BLAS` ON against OFF; `GGML_CPU_ARM_ARCH` empty against `armv8.2-a+dotprod+fp16`; KleidiAI ON against OFF |
| Q4_0 outputs bit-equal between the two builds while inputs are equal | agree; it is my r10 by measurement instead of by reading | not re-run |
| every small difference ends at the same logit floor (rms 0.05 to 0.07), so only many-chunk means or node comparisons discriminate | **agree, and it is the most useful result of the four memos** | not re-run |
| "the emulation's mean inside ±0.1 % at 40 chunks" | **the size is one sigma.** ±0.5 % per chunk over 40 chunks is 0.08 % on the mean. As an expectation fine; as a pass line use ±0.2 %, which is what her own §1.3 says 40 chunks resolve | arithmetic |
| the one-scale rule touches the head only | agree (my r11) | – |
| on the phone the attention takes the per-query path at every batch size | agree: with a Q8_0 cache `kv_is_f32_or_f16` is false | *source*, same line |
| phone-path attention band −0.1 to +0.3 % | no view until E-T14; I would not put a centre on it | – |
| the side file is the only form in which the fast body is legal under the app's rule | agree; it is what r17 concluded, said more sharply | – |
| build order: stock configuration, side file, W-position kernel, lane | **agree** | – |

### 1.2 r9 (E-T15, one-sided steps, the BLAS trap)

| her claim | my position | check |
|---|---|---|
| node comparison at decode sizes 2 and 4 against singles is a valid exactness test; zero differing nodes on KleidiAI libraries | **agree; E-T15 should replace the random-data test as the first gate** | not re-run (the host is busy with Tom's 40-chunk runs) |
| with BLAS compiled in, a batched matmul on a plain weight goes to BLAS with float activations | **agree** | *source*: `ggml-blas.cpp` lines 420 to 432: taken when all three sizes are 32 or more and the weight type has a `to_float`; lines 70 to 103 convert the weight rows to float. Her 3e-3 at the first node is the missing int8 activation step |
| the repack Q6_K head at four or more rows differs in last bits | plausible; not checked | – |
| no one-sided step other than the head's clamp, which changes nothing | agree with the table; the clamp count not re-run | – |
| requests on r18: idle anonymous memory, where the head's bytes live, two kernel lines | accepted in r19 and r20 already | – |

**Two additions to E-T15's specification.**
1. **The kernel entry line is a stop condition, not a note.** The laptop has the i8mm feature
   (`hw.optional.arm.FEAT_I8MM` = 1, *host*); the phone's A78 does not. Every existing host KleidiAI log prints
   `primary q4 kernel feature DOTPROD` (10 lines in `lab/c3/head/b2-kai` and the logs, *host*), so today's
   builds take the phone's entry. A clean rebuild with other flags could pick another entry. If the line does
   not say DOTPROD, the run tests another kernel and must stop.
2. **Run 6 (the official prompt) decides nothing about the phone's reply** by her own r11 §1.2a (Yi's reply
   changes with prompt batching). Feed the prompt as one batch, as the app does, and say so in the run's line.

### 1.3 r10 (width 2 in the attention, key-major walk, head repair)

| her claim | my position | check |
|---|---|---|
| at width 2 only the value conversion is shared exactly | **agree** | – |
| a second position costs 0.57 to 0.62 of the first | **0.57 to 0.66** by her own table: (105 − 45) ÷ 105 and (105 − 36) ÷ 105. Nothing rests on it | arithmetic |
| Yi converts each value row eight times per query | **agree** | *host*, GGUF headers read by me: Yi `head_count` 32, `head_count_kv` 4; Llama-2 and DeepSeek 32 and 32 |
| one conversion per group saves 0.5 to 1.0 ms per token on Yi | agree as an estimate; it helps the subject that binds least | – |
| the cache is 1.7 MB per layer and 56 MB over 32 layers at 200 positions | **agree**: 200 × 32 × 136 B × 2 = 1.74 MB; × 32 = 55.7 MB | arithmetic |
| key-major order is worth 0 to 5 ms per token | **0 to 3 ms, centre under 1 ms** (*estimated*), below | – |
| head repair: 16 float columns bring Yi's loss from +0.0041 to +0.0007 | not re-run. **Her stock figures agree with mine independently:** her +0.0041 against the float head and −0.0013 for ggml's path differ by 0.0054; my r11, against ggml's path with my own quantiser, was +0.00536 | cross-check |
| the proxy favours the path that chose the token (0.0013 on Yi) | **agree; it applies to my r11 and r12 figures.** My r11 prediction for Yi (+0.3 to +1.0 %) stands as committed and is scored as written | – |
| use the head knob; float columns only as a fallback | **agree** | – |

**Why I cut the key-major band.**
- Bytes: a token reads the whole cache once, 56 MB on Llama-2 at 200 positions. At a bus rate of 25 GB/s that
  is 2.2 ms of bus time in any order (*estimated*). The body reads 3.6 GB, so the cache is 1.5 % of a token's
  bytes.
- If today's walk truly waited for memory at each step (her premise at its worst), 205,000 steps × 5 lines ×
  about 100 ns is 100 ms of thread time, 25 ms on four threads (*estimated*). r15's whole O for Llama-2 is 6.9 ms
  with a band of 5 to 10, and that band comes from token time minus matmuls minus head, so 25 ms is excluded
  and the misses mostly overlap already.
- What is left for the order to remove: at the top of r15's band, 10 − (6.9 − 4.5) = 7.6 ms of attention
  against 4.5 ms of compute, so **at most about 3 ms**; at the centre nothing.
- r15's O is itself an estimate built on *device* token times, so this is a bound on a bound. E-T18 is still the
  right cell and costs 10 s; I only lower what the lead should expect from it. Her P 0.35 that the ratio reads
  above 1.15: mine 0.25.

### 1.4 r11 (official-prompt counts, reply stability, the guard, E-T19)

| her claim | my position | check |
|---|---|---|
| my r23 counts reproduce (Llama-2 6 of 31, Yi 12 of 31, set 1,791 of 3,606) | – | hers is the check |
| Llama-2's 6 of 31 is 3.4 standard deviations below the set's rate | **the spread is wrong; 2.0, see below** | *host*, held-out prompts |
| tokens per pass on 32 tokens moves in steps of 32 ÷ passes | **agree**: 32 ÷ 28, 27, …, 22 = 1.143, 1.185, 1.231, 1.280, 1.333, 1.391, 1.455 | arithmetic |
| Yi's official reply is not stable across host paths, Llama-2's is (P 0.7) | not re-run. The mechanism holds in *source*: a prompt of 167 tokens fed whole takes the tiled path, fed without its last token and then one token it takes both paths, and token 12 hangs on 0.03 of a logit | – |
| "greedy output identical" can only be asked inside one build, cache type and prompt batching | **agree; this belongs in the claim sheet** as a condition of the identity row | – |
| the guard needs: variant chosen by held-out score, the family written before the next look, the same source and parameters on the set and the official prompt in one session | **agree with all three.** §2 is written to them | – |
| key-major does nothing for the prefill; the tile is not proposed | agree | – |
| E-T19 host specification | agree as written. Its `work_size` warning is the important line | – |
| add "prompt prefix reuse: off" to the claim sheet | **agree; accepted** (`research.dart` line 102) | – |

**The spread of hits in 31 steps** (*host*; Llama-2, `head3000-tok`, the trainer's held-out prompts, none read by
the head's training):

| sample | n | mean hits | standard deviation | binomial value | share with 6 or fewer |
|---|---|---|---|---|---|
| first 31 guess steps of each prompt | 144 | 16.12 | **5.06** | 2.78 | 0.021 |
| all disjoint 31-step windows | 737 | 14.62 | 4.88 | 2.78 | 0.022 |

- The variance is 3.3 times the binomial one: a reply is easy or hard as a whole.
- So 6 of 31 is the second percentile of ordinary prompts. That is still unusual, and Ebony's conclusion for the
  claim stands (the official prompt is among the worst texts for Llama-2's head). What changes is the reading
  of **any** 32-token count: one more or one fewer pass (0.05 tokens per pass) is far inside the noise of the
  prompt, and the fixed set's 32 prompts give a standard error of about 5.06 ÷ √32 ÷ 31 = 0.03 on the hit rate.

### 1.5 The checks in one list

| # | memo | what I checked | result |
|---|---|---|---|
| 1 | r8 | the attention gate in source | confirmed |
| 2 | r8 | half against float sum, my simulation | 1.7e-3 against her 1.4e-3 |
| 3 | r8, r9 | the two builds' three settings; BLAS takes batched plain matmuls with float activations | confirmed |
| 4 | r9 | which kernel entry a host KleidiAI build takes | DOTPROD in all 10 log lines; made a stop condition |
| 5 | r10 | Yi's 4 key-value heads; cache bytes | confirmed |
| 6 | r10 | stock head loss on Yi against my r11 | 0.0054 against 0.00536 |
| 7 | r11 | the 3.4 | **2.0** with the measured spread |

## 2. The per-step chooser (E-T21)

### 2.1 What a step knows, and what a guess costs

After each pass the runtime has the last verified hidden state. It can:
- look in the prompt and the reply so far for the newest earlier match of the last 3, 2 or 1 tokens: guess `g`,
  match length `n` (0 when none). Cost: a scan of at most 512 tokens, microseconds;
- run the guess head: guess `h`, and a confidence `c`. Cost G = 2.6 ms per pass (*estimated*, the stack's value).

| confidence | extra cost per pass | AUC for "the head's guess is right" (*host*, held-out) |
|---|---|---|
| gap between the two largest logits | none: it falls out of the argmax scan | Yi 0.746, Llama-2 0.793 |
| top-1 probability (a softmax over the vocabulary) | 32,000 / 64,000 / 102,400 exponentials: 0.05 to 0.3 / 0.1 to 0.6 / 0.15 to 1.0 ms (*estimated*; not timed) | Yi 0.772, Llama-2 0.824 |

Bytes: none beyond the head; the rule is five numbers.

**The objective is tokens per second, not tokens per pass.** With P1 and P2 the times of a width-1 and a width-2
pass and the exact head paid per committed token, a guess with hit probability p pays when

    p > p* = (P2 − P1) ÷ (time per token − exact head time)

| | P1 | P2, `pair` / `linear` | P2 ÷ P1 | p* |
|---|---|---|---|---|
| Yi | 95.9 ms | 104.9 / 108.0 | 1.09 / 1.13 | 0.11 / 0.15 |
| Llama-2 | 107.7 ms | 119.8 / 123.5 | 1.11 / 1.15 | 0.14 / 0.18 |

All *estimated*: r14's priced lane (K 300, O 5.2, two GPU columns at the `_mc3` factor), plus the second
position's attention from r15 less r17's shared conversion (2.3 / 3.6 ms). These four numbers are the weakest
input of this section; R75 and the first width-2 kit replace them.

### 2.2 What the held-out data says (*host*; Yi 105 prompts, 15,053 guess steps; Llama-2 150 prompts, 24,029)

The script reproduces the trainer's held-out top-1 (0.3764 and 0.4709 against Tom's 0.377 and 0.471), which is
its check that it reads the same prompts and the same head.

Head precision by decile of its top-1 probability:

| decile | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| Yi | 0.10 | 0.16 | 0.18 | 0.23 | 0.30 | 0.34 | 0.41 | 0.53 | 0.65 | 0.86 |
| Llama-2 | 0.09 | 0.15 | 0.23 | 0.30 | 0.39 | 0.47 | 0.57 | 0.71 | 0.84 | 0.97 |

- The head is over-confident (a stated 0.98 to 1.00 is right 0.86 of the time on Yi) but monotone. A threshold
  on it is meaningful; its value is not a probability.

By lookup state:

| | share of steps, Yi / Llama-2 | lookup precision | head precision | they agree (precision then) | when they disagree: lookup right / head right |
|---|---|---|---|---|---|
| no candidate | 0.448 / 0.430 | – | 0.356 / 0.465 | – | – |
| match length 1 | 0.315 / 0.290 | 0.160 / 0.223 | 0.394 / 0.474 | 0.10 / 0.13 (0.70 / 0.83) | 0.10 / 0.13 against 0.36 / 0.42 |
| match length 2 | 0.100 / 0.107 | 0.408 / 0.469 | 0.368 / 0.480 | 0.21 / 0.27 (0.85 / 0.89) | 0.29 / 0.31 against 0.24 / 0.33 |
| match length 3 | 0.137 / 0.173 | 0.646 / 0.637 | 0.410 / 0.476 | 0.30 / 0.34 (0.92 / 0.93) | 0.53 / 0.49 against 0.19 / 0.25 |

- **Length 3: the lookup wins clearly. Length 1: the head wins clearly. Length 2 is a draw** (Llama-2: 0.469
  against 0.480). Rule T takes the lookup at length 2 and is right to, because it then skips the guess head
  (2.6 ms).
- Agreement is strong evidence (0.85 to 0.93 at lengths 2 and 3) but changes no choice: both give the same
  guess.
- The only room for a chooser is the disagreeing steps at length 2 and 3 where the head is very sure, and the
  lowest decile where no guess is better. That is under a tenth of the steps.

### 2.3 The family, and its fit

**Family F** (five numbers per model: c0, c1, c2, c3, s):

    n = 0:                 guess h if c ≥ c0, else no guess (a width-1 pass)
    n ≥ s (s in 2, 3; 0 = never):   guess g; the guess head is not run
    otherwise, g = h:      guess it
    otherwise:             guess h if c ≥ c_n, else g

- Rule T is the member (0, 0, –, –, 2). "The head always" is (0, 0, 0, 0, 0). So the family is compared with both
  on equal terms.
- Thresholds are taken from the deciles of c on the fitting prompts, plus "always" and "never": 11 values each.

**Fit:** coordinate search for the largest tokens per second under §2.1's costs. **Cross-fitting:** the held-out
prompts are cut into five parts by prompt; the thresholds come from four parts and are scored on the fifth.

| | Yi, `pair` | Yi, `linear` | Llama-2, `pair` | Llama-2, `linear` |
|---|---|---|---|---|
| width 1, no guessing (tok/s) | 9.17 | 9.17 | 8.53 | 8.53 |
| the head always | 10.78 | 10.51 | 10.59 | 10.30 |
| rule T | 11.07 | 10.80 | 10.89 | 10.58 |
| F on top-1 probability, cross-fitted, over the head | +2.82 % | +2.88 % | +3.75 % | +4.06 % |
| **the same over rule T** (range of the five parts) | **+0.14 %** (−0.27 to +0.62) | **+0.20 %** (−0.14 to +0.77) | **+0.95 %** (+0.61 to +1.45) | **+1.26 %** (+0.93 to +1.77) |
| F on the logit gap, over rule T | −0.11 % | −0.10 % | +0.50 % | +0.60 % |
| thresholds fitted on all held-out prompts (c0, c1, c2, c3, s), top-1 probability | 0.23, 0.23, 0.81, 0.91, 3 | 0.32, 0.23, 0.81, 0.91, 3 | 0.21, 0.31, 0.87, 0.87, 0 | 0.31, 0.31, 0.87, 0.87, 0 |

Rates are *estimated* from *host* counts. The softmax's own cost (0.1 to 0.5 % of a pass) is **not** subtracted.

- Yi: nothing. The fit lands on rule T with one small change (lookup taken blind only at length 3).
- Llama-2: about +1 %, +0.1 tok/s, less the softmax: **+0.08 to +0.12 tok/s** (*estimated*). With the free
  confidence (the gap) half of that.
- The thresholds at lengths 2 and 3 do not move between the two cost models; c0 does, as it should (it is the
  break-even p*).
- Rule T against the head alone: +2.7 % and +2.8 %. In r24 its lengths were chosen on training prompts by
  precision; here it is scored by rate on prompts no head saw. It holds.

### 2.4 The pre-registrable text

**Rule family.** F of §2.3, with the confidence named (top-1 probability or logit gap) and the lookup of
`tools/c3_head_t1.py` arm a.

**Fitting data.** Training-split prompts that the head was not trained on. The trainer's held-out prompts
(105 / 150) have now been used three times: to choose the head variant, by r24 indirectly, and by this pilot.
**The pre-registered fit therefore uses Tom's queued held-out guess dump (fresh training-split prompts), with
the head's confidence column**; this pilot's prompts become the cross-check, not the selector.

**Fitting procedure.** `bonnie_chooser_r27.py` at a named commit: deciles of c as the grid, coordinate search,
five-fold cross-fitting by prompt, the objective of §2.1. **The pass costs P1, P2, G and the head time are
device numbers** from the newest width-2 round, written into the pre-registration before the fit is run. Until
they exist, what is pre-registered is the script's hash and the cost source, not the five numbers.

**Adoption line (my proposal; replaces "+0.02 tokens per pass", which cannot see a no-guess decision or a
skipped guess head).** F replaces rule T only if, on the fresh prompts, its cross-fitted rate is at least 1.0 %
above rule T's on **each of two subjects**, after the confidence's own cost. Otherwise the rule stays rule T on
every subject. One rule for all subjects of a session; no per-subject mix chosen afterwards.

**Reads.** After adoption or rejection, one scoring on the fixed set (its **third** look, labelled so) and one
on the official prompt (already seen by all of us). Neither changes a number.

**What a kit prints with any guess rule** (with Ebony's E-T22): passes; guesses made by source (lookup, head);
guesses accepted by source; passes without a guess; the five numbers and the confidence's name.

**What cannot go wrong.** A guess rule only decides what is verified. Every committed token comes from the
exact pass, so the reply is identical under any member of F; the identity check of the claim sheet covers it.

**Prediction, written before the fresh dump exists:** F over rule T, Yi −0.3 to +0.6 %, Llama-2 +0.4 to +1.5 %,
DeepSeek unknown (no head). **P(the adoption line is met) 0.15.** P(rule T stays at least +2 % above the head
alone on both subjects on fresh prompts) 0.85.

### 2.5 When the gate matters (next question 1, done at once)

The same held-out prompts, P2 ÷ P1 swept with everything else fixed (`pair` width-1 costs; *estimated* rates from
*host* counts; gains cross-fitted, over rule T):

| P2 ÷ P1 | p* | Yi: width 1 / rule T (tok/s) | Yi: rule T + gate / full F | Llama-2: width 1 / rule T | Llama-2: rule T + gate / full F |
|---|---|---|---|---|---|
| 1.05 | 0.06 | 9.17 / 11.48 | −0.05 % / −0.03 % | 8.53 / 11.49 | +0.22 % / +0.75 % |
| 1.10 | 0.12 | 9.17 / 11.02 | +0.24 % / +0.16 % | 8.53 / 11.00 | +0.54 % / +0.93 % |
| 1.15 | 0.18 | 9.17 / 10.59 | +0.52 % / +0.59 % | 8.53 / 10.55 | +0.94 % / +1.29 % |
| **1.20** | 0.23 | 9.17 / 10.19 | **+1.01 % / +1.07 %** | 8.53 / 10.14 | **+1.23 % / +1.80 %** |
| 1.30 | 0.32 | 9.17 / 9.49 | +2.29 % / +2.67 % | 8.53 / 9.40 | +2.45 % / +3.11 % |
| 1.40 | 0.40 | 9.17 / 8.87 | +3.37 % / +4.71 % | 8.53 / 8.76 | +3.60 % / +4.67 % |
| 1.50 | 0.47 | 9.17 / 8.33 | +5.07 % / +6.64 % | 8.53 / 8.21 | +5.07 % / +6.42 % |

"Rule T + gate" is rule T with one fitted number: no guess when the head's confidence is under c0.
The sweep and the §1.4 spread table are short drivers around the script's own functions (`held_steps`, `sim`,
`make`, `fit`, `costs`) run from my scratch folder; they are not stored as files. `make((v, v, 0, 0, 2))` is the
gated rule T.

What the stack model gives for the ratio (*estimated*):

| | `_mc3` at two columns (cg 1.26), `pair` / `linear` | two GPU launches (cg 2.0), `pair` / `linear` |
|---|---|---|
| Yi, K 300 | 1.09 / 1.13 | 1.19 / 1.22 |
| Yi, K 452 (as R74 measured) | 1.11 / 1.13 | 1.18 / 1.21 |
| Llama-2, K 300 | 1.11 / 1.15 | 1.21 / 1.25 |
| Llama-2, K 452 | 1.12 / 1.14 | 1.20 / 1.23 |

- **The reading line for the first width-2 device round: P2 ÷ P1 at or above 1.20 → the one-number gate is worth
  building (+1 to +2.5 %); below 1.15 → rule T as it is.** If the batched GPU kernel does not deliver its
  two-column factor on the phone, the ratio is 1.18 to 1.25 and the gate pays.
- **Width 2 stops paying at all at a ratio near the tokens per pass:** about 1.33 on Yi and 1.42 on Llama-2
  without a gate. That is the number the lane's width-2 kit must print before any guess rule is discussed, and
  it is a kill line for width 2 that the committed stack tables do not show.
- Most of the gain at high ratios comes from the single gate number; the other three thresholds add 0.1 to
  1.5 points. So the family worth pre-registering if the ratio is high is "rule T + gate", one number, with the
  free confidence tried first.

## 3. Lists (additions only)

**Tests**

| # | test | where | cost | closes when |
|---|---|---|---|---|
| B-T27a | `bonnie_chooser_r27.py` on Tom's fresh held-out dump, with device pass costs | host | 1 minute after the dump | §2.4's adoption line |
| B-T27c | the first width-2 kit prints the time of a width-1 pass and of a width-2 pass on the same lane state, so that P2 ÷ P1 is a *device* number | device, with that kit | 0 | ≥ 1.20 → build the gate; ≥ tokens per pass → width 2 is closed for that subject |
| B-T27b | E-T15's kernel-entry line as a stop condition; its run 6 with the prompt fed as one batch | host, with E-T15 | 0 | – |

**Try:** nothing new. E-T21 moves from "test" to "written, expected to close"; E-Y4's expected value is lowered
(§1.3), its cell E-T18 unchanged.

## 4. Next questions, ranked

| # | question | state |
|---|---|---|
| 1 | At what second-position cost does the no-guess gate become worth 1 %? | **done, §2.5:** at P2 ÷ P1 of 1.20; width 2 itself stops paying near 1.33 (Yi) and 1.42 (Llama-2) |
| 2 | R75 into r18's draft, r25's script and §2.1's costs | waits for the report |
| 3 | The lookup arms on DeepSeek (Ebony r11 question 3): rule T needs no head for its lookup half, so a CPU dump of its official prompt and set gives DeepSeek's first guess numbers | Tom's dump |
| 4 | The claim sheet's identity row rewritten with Ebony's three conditions (same build, cache type, prompt batching) and the prefix-reuse row | small; with the next sheet revision |
| 5 | Time of a 32,000 to 102,400 entry softmax on the A78 against the free logit gap | only if question 1 revives the gate |
| 6 | r15's O against Tom's host dense decode and the KV screen, then E-T18's expectation | waits for Tom |

## 5. Sources (2026-11-24)

Repository: `ebony-joins-r8.md` to `ebony-joins-r11.md`; `lab/llama.cpp-base/ggml/src/ggml-cpu/ops.cpp` (lines
9555 to 9580), `common.h` (lines 9 to 10); `lab/llama.cpp-base/ggml/src/ggml-blas/ggml-blas.cpp` (lines 70 to 103,
396 to 432); `lab/llama.cpp-c3r32/ggml/src/ggml-cpu/kleidiai/kleidiai.cpp` (lines 84 to 97, 315 to 330, 403);
`lab/c3/head/build-kai/CMakeCache.txt`, `lab/llama.cpp-c3/build/CMakeCache.txt`; KleidiAI log lines under
`lab/c3/head/b2-kai` and `lab/c3/logs`; GGUF headers of the three served files; `tools/c3_head_train.py` (the
held-out split and the head's form); `lab/c3/head/{yi,llama2}/train{2117,3000}.{rows,h16}`, `train1500{,b}.ptok`,
`head2117-tok.pt`, `head3000-tok.pt`; my `bonnie_stack_r13.py` and the scripts it loads; my r14, r15, r17, r23,
r24. No external source this round.
