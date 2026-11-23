# c3 memo-12 work (Ebony, round 12): how much of the official reply is a copy, the DeepSeek dump as a task for Tom, review of Bonnie r24 and r25

Date: 2026-11-24. Author: Ebony (`claude-fable-5-2`). Not committed; the lead commits. No verdict here.
Follows [ebony-joins-r11.md](ebony-joins-r11.md). Reviews [bonnie-joins-r24.md](bonnie-joins-r24.md) (§0, §1.1 to
§1.3 read) and [bonnie-joins-r25.md](bonnie-joins-r25.md) (§0 and §1 read). r26 exists and is not reviewed here.

Labels as before. Triples are Yi / Llama-2 / DeepSeek. New *host* numbers: arithmetic with my own code on Tom's
existing files (`official.*`, `eval.*`, the saved guesses). Nothing was run on a model: the laptop is busy.

## 0. Answer first

1. **Llama-2's official reply is a little more than half copy: 17 of its 32 tokens lie in four runs that occur
   word for word in the prompt** (run lengths 4, 7, 3, 3; the first starts at reply token 2). Yi's reply has one
   run of 3. That is why a prompt lookup is right 13 times in 21 proposals on Llama-2 and 2 in 14 on Yi.
2. **Bonnie's step-by-step table and her rule T numbers reproduce exactly from my own lookup** (both official
   strings, character for character; the set at 1.486 and 1.391).
3. **Rule T is sound for match length 3 and decided by a coin flip at length 2 on Llama-2, and that coin flip is
   worth two passes on the claim prompt.** Lookup at length 2 beats Llama-2's head by 0.003 on training data
   (her own remark) and loses by 0.011 on the set once the head is scored on the same steps (0.469 against
   0.480). With "length 3 only" Llama-2's claim prompt reads 1.231 tokens per pass, not 1.333. So Llama-2's
   claim-prompt price is a range, **1.231 to 1.333**, until the rule's criterion is written with a margin. §1.3.
4. **Her comparison uses the head's average hit rate; the right one is the head's rate on the steps where the
   lookup has a match.** On the set the head is better there than on average (Llama-2 0.536 at length-3 steps
   against 0.497 overall; Yi 0.435 against 0.379). Lookup still wins at length 3 by 0.15 to 0.24. The fix is one
   column in Tom's scoring script. §1.3.
5. **r25 (copy-run passes): agreed, do not build now.** One consequence of §1.1 supports her: at width 2 a run
   of 7 copied tokens can save at most 3 passes, because a hit commits two tokens and the next guess is unused.
6. **The DeepSeek dump is specified for Tom in §2:** one tool change (feed the prompt as one batch; cache type
   as a parameter), two configurations, about 5 to 20 minutes of CPU and 35 MB. It yields seven things, among
   them the hidden states that E-T10 and E-T20 wait for.

## 1. The official reply and the lookup (my question 3; review of r24)

### 1.1 How much is copied (*host*; `official.ptok` and `official.rows` of Tom's dumps)

For each reply token: the length of the longest run of reply tokens containing it that occurs contiguously in
the prompt (0 = no run of two or more).

| | prompt tokens | reply tokens in a prompt run of ≥ 2 / ≥ 3 / ≥ 4 | run length per reply token (32 tokens) |
|---|---|---|---|
| Llama-2 | 186 | **17 / 17 / 11** | `00444400077777770000033300000333` |
| Yi | 167 | 3 / 3 / 0 | `00000000000000000000033300000000` |

- Llama-2 copies in four runs separated by 3 to 5 of its own tokens. The copy starts at the third reply token,
  so a lookup needs no warm-up.
- Of the lookup's 21 proposals on Llama-2, 20 found their match in the prompt and one in the reply itself.
- **What width 2 can take from a run.** Inside the 7-token run the lookup is right at six consecutive steps
  (`L1 L2 L3 L3 L3 L3`), but a hit commits two tokens and the guess made for the second is never used. So six
  hits save three passes. Over the whole reply: lookup right 13 times, 7 passes saved (32 → 25). This is the
  room a copy-run pass (her r25) would aim at, and it is 3 to 4 passes on this prompt at best.

### 1.2 Her numbers, recomputed

My lookup (newest earlier place where the last 3, else 2, else 1 tokens occurred, over prompt plus reply so far;
propose what followed) on the same files:

| | Yi | Llama-2 |
|---|---|---|
| official: step strings of her §1.2 | identical | identical |
| official, tokens per pass: head / lookup / rule T / either right | 1.391 / 1.032 / 1.455 / 1.455 | 1.143 / 1.280 / 1.333 / 1.391 |
| set, tokens per pass: head / lookup / rule T | 1.332 / 1.202 / **1.391** | 1.453 / 1.185 / **1.486** |
| set, "length 3 only" | 1.378 | 1.483 |

All equal to hers.

### 1.3 Rule T: what holds and what is thin

Rule T: use the lookup's guess when its match is 2 or 3 tokens long, else the head's. She fixes it on training
data: lookup precision by match length against the head's held-out top-1.

**The head's rate on the same steps** (*host*, the prompt set, where the heads' guesses are saved; this is a
check of her assumption, not a new selection):

| match length at the step | Yi: steps; head; lookup | Llama-2: steps; head; lookup |
|---|---|---|
| 0 (no proposal) | 1,400; 0.350; – | 1,506; 0.493; – |
| 1 | 987; 0.376; 0.253 | 986; 0.479; 0.234 |
| 2 | 393; 0.382; **0.489** | 369; **0.480**; 0.469 |
| 3 | 751; 0.435; **0.671** | 745; 0.536; **0.689** |

- **Length 3: the lookup wins on both models by a wide margin,** also against the head's higher rate on those
  steps. This part of the rule is safe.
- **Length 1: the head wins on both.** Safe.
- **Length 2: Yi's lookup wins by 0.11; Llama-2's is a tie** (0.469 against 0.480 here; 0.474 against 0.471 in
  her training table). Nothing in the data picks a side for Llama-2.
- **What the tie is worth** (*host*, the official strings):

  | rule | Yi: passes; tokens per pass | Llama-2: passes; tokens per pass |
  |---|---|---|
  | lookup at length ≥ 2 (rule T) | 22; 1.455 | 24; **1.333** |
  | lookup at length 3 only | 23; 1.391 | 26; **1.231** |
  | lookup at any length | 24; 1.333 | 24; 1.333 |

  On the set the two variants differ by 0.003 on Llama-2. On the claim prompt they differ by two passes,
  because all four of its length-2 lookups happen to be right. In her price table that is 9.97 / 10.27 against
  about 9.24 / 9.51 tok/s (*estimated*, her r23 row for 1.231).

**So, for the guard:**
1. Write the criterion with a margin and with the head scored on the same steps: "the lookup is used at a match
   length where its precision exceeds the head's top-1 **at steps of that length** by at least 0.02, both on
   held-out training prompts". That needs the head's guesses saved for the held-out rows, which the scoring
   script does not do today (one column; with E-T21's probability column it is the same change).
2. By that criterion today's evidence gives length ≥ 2 for Yi and length 3 only for Llama-2. If one rule for
   every model is wanted (the campaign's model-generic principle), pool the models before comparing; pooled,
   length 2 wins by about 0.05 (0.479 against 0.43 on the set) and the rule is "≥ 2" for all.
   **Either is defensible. It must be chosen before the next look, and the pre-registration should print the
   claim-prompt value under both variants.**
3. The family "lookup with a length threshold" appeared after the official steps had been laid out token by
   token (her §1.2 precedes her §1.3). That is post hoc for the family, even though its parameters come from
   training data. What repairs it is what she already shows: the same rule improves the set on both models
   (+0.059 and +0.033 tokens per pass). I accept it on that ground and would say so in the pre-registration.
4. From r11 §1.2a: Yi's official reply is not stable across host configurations, so Yi's 1.455 is one draw.
   Llama-2's reply was the same on the phone-like path.

### 1.4 r25 (copy-run passes)

| her claim | my position |
|---|---|
| a copy-run pass (propose the next W − 1 tokens after a match and verify them in one pass of width W) is worth +0.1 to +0.2 tok/s on the set and −0.4 to +0.4 on the claim prompt | agree with the size; not re-run. §1.1 gives the ceiling on the claim prompt from the reply itself: 3 to 4 passes on Llama-2, none on Yi |
| matched runs usually stop early: after a 3-token match the next token is right 0.52 / 0.57, the next two 0.32 / 0.36 | consistent with the set's 0.67 to 0.69 at length 3 (hers is on training prompts) |
| it depends on the GPU's time at three and four columns and on the extra positions' attention | agree; by r10 §1.2 a further position's attention is 0.57 to 0.62 of the first once the value conversion is shared |
| do not build it now; re-run after R75 | **agree** |
| the output is the stock greedy text, because nothing is committed that the model did not produce | agree, with the condition of r11 §1.2a: identity holds against a stock arm of the same build, cache type and prompt batching |

## 2. The DeepSeek dump: a task for Tom (my question 2)

DeepSeek binds in every stack and is the one subject with no dump. This task trains nothing. It is the CPU part
of the head task (r5 §2.3 step 2a) pulled forward, plus the phone-like configuration.

### 2.1 Tool change (`tools/c3_head_dump.cpp`, about 20 lines)

| change | why |
|---|---|
| `C3_HEAD_ONEBATCH=1`: decode all prompt tokens in one call with output on the last, and write its row as step 0; then single decodes | the app feeds the prompt as one batch (Bonnie r16 row 14). The tool today feeds the prompt without its last token, then the last token alone. With an F16 cache those two take different attention paths at 64 or more tokens, and on Yi that alone changed the official reply at token 12 (r11 §1.2a) |
| `C3_HEAD_KV=<f16\|q8_0>`: the cache type for K and V; `q8_0` also sets flash attention on | the app's setting is Q8_0 |
| a fifth output file `<out>.gap`: float32 per row, the difference between the two largest logits | near-ties decide whether a reply survives a change of path; the saved top-64 ids do not hold the values |
| self-check: with both knobs unset the outputs are byte-identical to the present tool's on the first 4 set prompts of Llama-2 | the existing dumps stay comparable |

### 2.2 Builds

| name | what | state |
|---|---|---|
| R (reference) | the freeze's flags: `armv8.2-a+dotprod+fp16`, BLAS off, repack on, KleidiAI off. It is `lab/llama.cpp-c3/build`, the libraries Tom's CPU dumps already use | exists |
| P (phone-like) | the same flags plus KleidiAI, run with `GGML_KLEIDIAI_SME=0`; the log must print `primary q4 kernel feature DOTPROD` and a `CPU_KLEIDIAI model buffer size` line | the build Tom is making for the stock runs. **Not** `lab/c3/head/build-kai` (BLAS on, no instruction-set flag: r8, r9) |

If the tool is started under `nice` or `time`, set the library path with `/usr/bin/env` directly before the
binary (r6).

### 2.3 Runs (file `lab/c3/models/deepseek7b-q4_0.gguf`, the unmodified one; one sequence; 8 threads on an idle laptop)

| # | build | cache | prompt feeding | prompts | new tokens | purpose |
|---|---|---|---|---|---|---|
| 1 | R | F16 | as today | `official.txt` | 32 | comparable with Yi's and Llama-2's official dumps |
| 2 | R | F16 | as today | `eval.txt` (32 prompts) | 128 | comparable with their set dumps; the hidden states for the head scripts |
| 3 | P | Q8_0 | one batch | `official.txt` | 32 | the phone-like reply |
| 4 | P | Q8_0 | one batch | `eval.txt` | 128 | only if time allows; the phone-like set |
| 5 | P | Q8_0 | one batch | `official.txt` on Yi and Llama-2 | 32 | 20 seconds each; replaces my r11 runs, which used the flawed KleidiAI libraries |

Time (*estimated*): run 2 is 4,096 decoded tokens and about 3,300 prompt tokens. At 15 to 20 tok/s on eight idle
threads that is 4 to 6 minutes; on two threads beside other jobs about 20 (my single decodes ran at 3.4 tok/s
on two threads). Runs 1, 3 and 5 are under a minute together. Run 4 as run 2.
Disk: 8.2 KB per row: 34 MB per set dump, 0.3 MB per official dump.

### 2.4 What it yields

| # | result | from | used by |
|---|---|---|---|
| 1 | DeepSeek's official prompt length with its own template | run 1's log | replaces my hand count of 153 (r4 §3) |
| 2 | its 32-token official reply on both configurations, and whether they are the same; the smallest top-2 gaps | runs 1, 3, `.gap` | r11 §1.2a for the third subject |
| 3 | how much of its official reply is copied from the prompt; the lookup's step string; tokens per pass for lookup alone and for "length ≥ 2" and "length 3" without a head | run 1 or 3, my code of §1 (minutes) | **the first measured tokens-per-pass figure of any kind for DeepSeek on the claim prompt.** A lookup alone is a floor; every stack assumes 1.385 |
| 4 | the same on the set: lookup precision by match length | run 2 | rule T's lookup half for the third model |
| 5 | the final hidden state's range ÷ rms on chat rows, and Bonnie's head-path proxy | run 2, `bonnie_head_paths.py`, `ebony_head_int8.py` (add the model's name to each) | **E-T10**: the prediction of DeepSeek's stock shift (my band +0.3 to +1.2 %) from its own states |
| 6 | the outlier dimensions and the float-column repair | run 2, `ebony_head_outliers.py` | **E-T20** |
| 7 | the base check for a later head: the dequantised head picks the dumped token on ≥ 0.99 of rows | run 2, the training script's `base` | r5 §2.4, first row |

What it does not yield: a head's top-1 for DeepSeek. That needs the training dumps (r5 §2.3 step 2a: 1 to 1.5 h
of Metal and 3.7 GB), which stay queued.

### 2.5 Checks

- Runs 1 and 2: the tool's log line "prompt 0 formatted" shows DeepSeek's own template with no system prompt
  (its model card says a system prompt is not supported; r5 §2.6).
- Run 3 against run 1: the number of equal reply tokens and the first difference, as my `ebony_official_path.py`
  prints. A difference is a result, not an error.
- Run 5 on Llama-2 should give the dump's 32 tokens if my r11 result holds on a clean build; on Yi it is
  expected to differ.

## 3. Lists (additions only)

| # | test | where | cost | closes when |
|---|---|---|---|---|
| E-T23 | the DeepSeek dump, runs 1 to 3 and 5 (run 4 optional) | host, Tom, CPU | 20 lines; 5 to 20 minutes | it yields the seven results of §2.4; lookup-alone on the official prompt under 1.10 → DeepSeek's claim-prompt price rests on its head alone |
| E-T24 | head guesses saved for the held-out training rows; lookup against head by match length on those rows | host, Tom's scoring script | one column, minutes | the criterion of §1.3 item 1 gives the rule per model and pooled; both printed in the pre-registration |

No new try item.

## 4. Next questions, ranked

| # | question | state |
|---|---|---|
| 1 | Tom's KleidiAI stock result (flags, E-T13, E-T16) against the bands | waiting for his file |
| 2 | DeepSeek's dump (E-T23) and what its official reply gives a lookup | specified; Tom's |
| 3 | Can a lookup use more of a copied run at width 2? | **closed by reading the rule:** after an accepted guess the next pass starts two tokens on and looks up from there, which is what the simulation already does. The loss inside a run is the unused second guess of each accepted pass (§1.1), and only a wider pass recovers it (r25) |
| 4 | Does the app's chat wrapper put the official prompt's notes where a lookup can match across the wrapper tokens, and is Llama-2's copy rate specific to its `[INST]` format? | host, on `official.ptok`; small |
| 5 | The chooser by the head's confidence (E-T21) against rule T, on held-out rows | Tom's column first |
| 6 | Review of Bonnie r26; R75 free reads | next round; R75 |

## 5. Sources (2026-11-24)

Repository: `lab/c3/head/{yi,llama2}/official.{ptok,rows}`, `eval.{ptok,rows}`, `head2117-tok.pt.*.pred.npy`,
`head3000-tok.pt.*.pred.npy`; `tools/c3_head_dump.cpp` (read in r5); Bonnie's r23 to r25. No external source
this round.
