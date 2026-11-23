# c3 memo-12 work (Ebony, round 11): review of Bonnie r21 to r23, the claim prompt's guesses checked, key-major order in the prefill, the host specification for the attention walk

Date: 2026-11-24. Author: Ebony (`claude-fable-5-2`). Not committed; the lead commits. No verdict here.
Follows [ebony-joins-r10.md](ebony-joins-r10.md). Reviews [bonnie-joins-r21.md](bonnie-joins-r21.md),
[bonnie-joins-r22.md](bonnie-joins-r22.md) (their "answer first" sections and the parts quoted) and
[bonnie-joins-r23.md](bonnie-joins-r23.md) (§0 to §2 read in full; §3 not read).

Labels as before. Triples are Yi / Llama-2 / DeepSeek. New *host* numbers: Tom's saved guesses recomputed with my
own code; one thread-count comparison with `ebony_batchdiff.py`; four greedy decodes of the official prompt
(`ebony_official_path.py`, two threads). No model was trained, no Metal, no phone.

## 0. Answer first

1. **Bonnie's official-prompt finding is right; I get her numbers from Tom's files with my own code.**
   Llama-2's best head on the set (`head3000-tok`) is right 6 times in 31 on the official prompt, 1.143 tokens
   per pass; on the set 1,791 of 3,606, 1.453. Yi's best: 12 of 31, 1.391; on the set 0.379, 1.332.
2. **It is not chance.** At the set's rate Llama-2's head would get 15.4 of 31 with a standard deviation of 2.8;
   6 is 3.4 below. The official prompt is a different kind of text for that model (its reply copies the prompt's
   notes, as Tom found).
3. **One limit on "it is the number, not a sample":** the 31 guesses are fixed on the path that made the dump.
   Checked today (*host*, §1.2a): on a phone-like path (KleidiAI libraries, Q8_0 cache) **Llama-2's 32-token
   reply is the dump's**, though two steps are near-ties; **Yi's is not**, and Yi's reply changes even on the
   dump's own build when the prompt is fed as one batch. So Llama-2's 6 of 31 probably is the phone's (P 0.7);
   Yi's 12 of 31 probably is not. The phone's own count is fixed run to run and has to be read from the phone.
4. **Her guard is right and needs two more sentences.** The official prompt has already been seen, and the
   prompt set has already been used to pick the "best variant" of each head. So neither can be the place where
   a guess source is chosen now. §1.3 says what is left.
5. **Tokens per pass on 32 tokens moves in steps.** It can only be 32 ÷ (number of passes): 1.143, 1.185, 1.231,
   1.280, 1.333, 1.391, 1.455. Yi's heads with 12, 13 and 14 hits all read 1.391. One pass is 0.05, about 0.4
   tok/s in the stack.
6. **Key-major order does nothing for the prefill** (my question 4): there the op already re-reads one head's
   rows for every query, so they are hot. What can be shared exactly is the value conversion across a tile of
   queries, worth about 0.15 to 0.2 s of a prefill (*estimated*), outside the metric. Not proposed.
7. **The host specification for the attention walk (E-T19) is §3.** One fact for it, measured today: stock single
   decodes with the Q8_0 cache are bit-equal at two and at four threads on every node (*host*, 48 tokens).
8. **r21 and r22: agreed.** Her side-file key already carries what I would have asked for (the kernel names and
   packing parameters). One row to add to the claim sheet in §1.4.

## 1. Review of Bonnie r23

### 1.1 The numbers, recomputed (*host*; `official.rows`, `eval.rows` and each `<head>.pt.<set>.pred.npy` under `lab/c3/head/`)

A guess at row j is a hit when it equals row j + 1's token; tokens per pass by Tom's rule (a hit commits two).

| model, head | official: hits of 31; tokens per pass | prompt set: hits; top-1; tokens per pass |
|---|---|---|
| Yi, `head2117-tok` | 12; **1.391** | 1,338 of 3,531; 0.379; 1.332 |
| Yi, `head-tok` (1,500 prompts) | 13; 1.391 | 1,298; 0.368; 1.325 |
| Yi, `head2117-full` | 12; 1.391 | 1,255; 0.355; 1.310 |
| Yi, `head-wd` | 14; 1.391 | 1,212; 0.343; 1.302 |
| Yi, `head-r512` | 9; 1.280 | 1,178; 0.334; 1.298 |
| Llama-2, `head3000-tok` | **6; 1.143** | 1,791 of 3,606; 0.497; 1.453 |
| Llama-2, `head3000-full` | 8; 1.231 | 1,716; 0.476; 1.436 |
| Llama-2, `head-tok` (1,500 prompts) | 9; 1.231 | 1,632; 0.453; 1.406 |

- Every cell of her §2.1 that I can form from the files agrees (her 1.448 on the set is 1.453 in my count, a
  difference in how the last row of a prompt is treated; nothing rests on it).
- The lookup rows of her table are Tom's; I did not recompute them (they need his lookup code).
- **Significance.** Llama-2, `head3000-tok`: expected 15.4 hits at 0.497, standard deviation 2.78; 6 is −3.4.
  `head-tok`: expected 14.0 at 0.453; 9 is −1.8. Yi: expected 11.7 at 0.379; 12 is on it.
- **Granularity.** With 32 tokens, tokens per pass is 32 ÷ passes. Two hits in a row count once (the second
  guess is never used), which is why Yi's 12, 13 and 14 hits give the same 23 passes. A difference of one pass
  is 0.05 in tokens per pass.

### 1.2 "For the claim it is not a noisy sample; it is the number"

True on one path. Three things decide whether the host's count is the phone's:
- **The path.** The phone's head and cache differ from the dump's (KleidiAI int8 head or the knob; Q8_0 cache).
  Per position the two paths pick the same token with probability 0.98 to 0.99 on Llama-2 (*host*, r6) and the
  heads alone with 0.979 on Yi (*host*, r7). Over 32 positions that is 0.5 to 0.7 for an identical reply.
- **After a divergence** the reply is another text of the same kind, and the head's hits on it are a new draw
  at the official prompt's rate (low for Llama-2, about the set's rate for Yi).
- **The template.** Tom checked the dump's prompt against the app's `_benchPrompt` and used the model's own
  template with no system prompt. The phone's token count per run (167 and 186 by his dumps) is the check.

So: on the phone the count is one fixed number per configuration, and a kit that runs the official prompt with
the head should print it (guesses made, guesses accepted, passes). Until then 1.143 and 1.391 are the host's
count with P 0.5 to 0.7 of being the phone's. Her re-price stands as the centre.

### 1.2a The official reply on a phone-like path (next question 2 of §5, done at once; *host*)

`ebony_official_path.py` (new): the dumped prompt tokens fed in one batch, then 32 greedy single decodes; the
reply compared with Tom's dump token by token; the gap between the two largest logits printed at every step.

| subject | build, cache | reply tokens equal to the dump | first difference | steps with a top-2 gap under 0.3 |
|---|---|---|---|---|
| Llama-2 | repack, F16 (the dump's path) | 32 of 32 | – | token 14: 0.091; token 26: 0.049 |
| Llama-2 | KleidiAI libraries, Q8_0 cache | **32 of 32** | – | token 14: 0.077; token 26: 0.128 |
| Yi | repack, F16 | **14 of 32** | token 12 | token 12: 0.032 |
| Yi | KleidiAI libraries, Q8_0 cache | 18 of 32 | token 17 | tokens 12, 16, 17, 18: 0.13 to 0.19; token 25: 0.033 |

- **Llama-2's reply is the same on the phone-like path, so its 6 of 31 carries over as far as the host can
  say.** Two of its steps are decided by 0.05 to 0.13 of a logit, and the paths differ by 0.05 to 0.07 rms per
  logit (r8 §1.2), so this is not safe: P(the phone gives the same 32 tokens) about 0.7.
- **Yi's reply is not stable even on the dump's own build.** My run differs from Tom's dump from token 12 on the
  same libraries and cache type. The only difference is how the prompt was fed: his tool decodes the prompt
  without its last token and then the last token alone; mine feeds all 167 at once, and with 64 or more queries
  the attention takes its float path (r8 §1.1). Token 12 is decided by 0.03 of a logit. On the phone-like path
  the reply leaves the dump at token 17.
- So for Yi the host's 12 of 31 belongs to one of at least three replies. Its tokens per pass on the phone is a
  new draw at Yi's rate (0.38: 1.28 to 1.39), not the 1.391 of the dump. Bonnie's Yi row (+0.5 tok/s on the
  claim prompt) should be read as "between the set's 10.8 to 11.1 and 11.3 to 11.6".
- **A wider consequence for the claim rule's "greedy output identical":** identity can only be asked between
  two arms of the same build, cache type and prompt batching. Across any of those three the official reply
  itself can change at a near-tie. The app feeds the prompt as one batch (her r16 row 14); a host comparison
  has to do the same.
- The KleidiAI libraries here are T1b's, with the build flags r8 and r9 warn about. For this comparison they
  only make the two paths differ a little more.

### 1.3 Her guard against choosing a guess source on the claim prompt

Her text: a source chosen because it does well on this one prompt is what the fixed prompt set exists to check;
fix the rule on the training split, then report both.

**Agree. What it still needs:**
1. **Both test texts are already used.** Tom's table of official-prompt results has been read by all of us, and
   "best variant each" was picked by prompt-set top-1 (r1500 against r3000, `tok` against plain). From here on a
   choice justified by either is post hoc. The R73 plan's condition "nothing is tuned on it from adoption on"
   should be applied to the head variants too: **the variant is the one with the best held-out score on the
   training split** (Tom's 5 % held-out prompts, never trained on), and that happens to be the same heads
   (Llama-2 `tok` 3,000: 0.471; Yi `tok` 2,117: 0.377, *host*, his table). Write that as the rule, so the set
   stops being the selector.
2. **The rule family has to be written before the next look**, in a pre-registration, with its parameters:
   for example "lookup when the prompt holds a candidate of at least n tokens, else the head", n chosen on the
   training split by tokens per pass. A per-step chooser by the head's own confidence is the better family
   (my r4 §2.2); it needs the head's probability per row, which the saved guesses do not hold. It is a host
   test of minutes once the scoring script writes that column.
3. **What the set must show** (her §1.4 proposal: the method's arm at least the stock arm on the set, ABBA,
   identical text): agree. I would add the mirror condition for the claim: **the same source, with the same
   parameters, is used on the set and on the official prompt in one session.** A lookup that is switched on only
   for the claim run is the case the guard is for.
4. A source that copies from the prompt is a legitimate, model-free lever. Nothing in the rules is against it.
   It is the selection that needs the guard, not the lever.

### 1.4 r21 and r22

| her claim | my position |
|---|---|
| r21: choice C is a shared helper, two call sites and the compute-buffer release, about 300 lines, every piece behind a knob | agree as a design |
| r21: the file's validity rests on a key (the model's sha256) and a **variant** string: KleidiAI version, the selected Q4_0 and 8-bit kernel names, `nr`, `kr`, block length; for the repack type the layout name | **agree; this is the point I would have raised.** The packed layout differs between the dotprod and i8mm entries, so a file built on one core type must not be read on another. It is covered |
| r21: identity matrix (knob off, first run, second run, per chunk at three batch settings, two subjects, on the KleidiAI build) | agree. **Add one faster line:** `ebony_batchdiff.py` with the file mapped against anonymous, every node equal (2 minutes; it also catches a wrong offset in the first matmul, where a per-chunk value would only look odd) |
| r21: nothing else in a context is anonymous and large (20 / 15 / 40 MiB) | not checked |
| r22: the claim-run sheet, 38 rows, 6 open; three corrections to the R73 plan | agree with the three corrections (the forced copies are not "stock"; the cache is Q8_0; the reference differs in four ways) |
| r22: open decisions in blocking order (load path; head placement; second reference; cache type; four small ones) | agree with the order |
| r22: load time 11 to 31 s today, 7 to 9 s mapped | not checked; outside the metric |

**One row to add to the sheet:** "prompt prefix reuse: off" (`research.dart` line 102, `reusePrefix = false`,
*source*). Every official run decodes its 150 to 190 prompt tokens again. That is what my prefill heat arithmetic
(r4 §3, r5 §1.3) assumes, and a kit that turned it on would change the heat before every decode.

## 2. Key-major order in the prefill (my question 4)

*Source*: the same per-query function runs for every query when the cache is Q8_0 (r8 §1.1). The work index is
`ir = iq1 + neq1 × iq2`: head by head, and inside a head query by query, each query walking all cached rows.

| | decode (one query) | prefill (186 queries) |
|---|---|---|
| bytes of one head's keys and values | 200 × 272 B, read once | 186 × 272 B = 50 KB, read by each of 186 queries in turn |
| cache state of those bytes | cold for every head (r10 §1.3 b) | cold for the first query of a head, then in the first- or second-level cache for the other 185 |
| row steps per layer | 32 × 200 | 32 × 186² ÷ 2 = 553,000 |
| value conversions of one row and head | 1 (8 on Yi) | one per later query: up to 186 |

- **The order is already the right one for memory in the prefill.** Putting the row loop outside all 186 queries
  would need 186 accumulators per head (95 KB) hot at once, which is worse.
- **What is repeated is the value conversion.** A tile of T queries (8 fits the first-level cache: 4 KB of
  accumulators) walking the rows together converts each row once per tile. Exact, by the argument of r10 §1.2.
  Saving: 7 ÷ 8 of the conversion, 0.30 to 0.37 of the prefill's attention.
- Price (*estimated*): the prefill's attention is about 0.5 s on four threads at 186 tokens (r9 §0 item 6), so
  **0.15 to 0.2 s per prefill**, 2 to 3 % of a 6 to 15 s prefill. It is outside the metric and too small to
  change the heat at decode start. **Not proposed.** If E-Y4 is written as "a row loop over a set of (query,
  head) states", the tile comes with it at no extra cost; that is the only reason to remember it.

## 3. Host specification for the attention walk (E-T19; for Tom, through the lead)

**What is built.** In `ggml_compute_forward_flash_attn_ext_f16_one_chunk` (`ggml-cpu/ops.cpp`), behind
`VOLLIE_FA_KEYMAJOR=1`, off by default:

1. For the rows `[ir0, ir1)` a thread owns, when they are one query and several heads: keep per head the
   running maximum, the running sum, the float accumulator (DV floats) and the quantised query (DK values in
   the cache's dot type).
2. Loop the cached rows `ic` outside and the heads inside. Per `ic`: read the mask once per mask row; convert
   the value row once per **distinct value head** (`iv2`), not once per query head; then for each head do
   exactly today's steps (score, scale, soft cap, mask, maximum and sum update, weighted add).
3. Everything after the row loop (sinks, partials, the division by the sum, the store) stays per head as it is.
4. **`work_size` grows.** Today a thread has DK + 2 × DV floats. The new path needs (number of heads the thread
   owns) × (DV + DK + 2) + DV. The planner's size for `FLASH_ATTN_EXT` must be raised with the knob on, or the
   path writes past its buffer. (The same kind of trap as condition 5 of r5 §3.2.)
5. Other paths are untouched: the tiled path (64 or more queries with an F32 or F16 cache), and batches of more
   than one query (they keep today's order; §2 above).

**Why it is exact.** For a given (query, head), the sequence of floating-point operations and their operands is
unchanged: the same rows in ascending order into the same accumulator. Only the interleaving between heads
changes, and heads share no state. The converted value row is a function of the cached bytes alone.

**Exactness test (pass = zero differing nodes, `node_1` excluded):** `ebony_batchdiff.py` in its save mode, knob
off against knob on, same build, same thread count.

| # | subject | cache | decodes | positions | threads | covers |
|---|---|---|---|---|---|---|
| 1 | Llama-2 | Q8_0 | singles (`EBONY_B=1`) | 64 | 2 | the phone's path, no grouping |
| 2 | Yi | Q8_0 | singles | 64 | 2 | grouped heads: one conversion for eight heads |
| 3 | Llama-2 | Q8_0 | singles | 64 | 4, then 1 | the thread split (a thread owns 8 heads, then 32) |
| 4 | Llama-2 | Q8_0 | singles | 300 | 2 | past 256 cached rows the op takes its split-cache variant with partial results; the new order must not change it |
| 5 | Llama-2 | F16 | singles | 64 | 2 | the half-precision accumulator branch |
| 6 | Llama-2 | Q8_0 | decodes of 2 (`EBONY_B=2`) | 64 | 2 | more than one query: the knob must change nothing |
| 7 | Llama-2, Yi | Q8_0 | the official prompt with the chat template, then 32 singles | about 200 | 2 | the claim's shape |

- Known about stock, so that a failure is read correctly: single decodes with the Q8_0 cache are bit-equal at 2
  and 4 threads on all 585 nodes (*host*, Llama-2, 48 tokens, today). With an F16 cache, decodes below 64
  queries are equal to singles (r9 §1.2).
- The tool change asked for in r9 §1.3 (compare every column; exit 1 with the first differing node) serves here
  too. For run 4 the tool's context must be raised from 512 if more than 512 positions are wanted; 300 fits.
- Then the usual: per-chunk perplexity knob on against off at `-ub 1` with `-ctk q8_0 -ctv q8_0`, 4 chunks, two
  subjects (identical values); `-b 512` must be untouched by construction.

**Timing arms on the laptop (relative only; they check the cell, not the phone):** a model-free graph of one
`FLASH_ATTN_EXT` node on a Q8_0 cache of Llama-2's shape and of Yi's, 200 and 500 cached rows, one query, 4
threads:
(i) as it is, with a 64 MB buffer read between calls to evict the cache;
(ii) as it is, called back to back;
(iii) key-major, evicted as (i).
Report ns per (head, row) step for each. On the M4, (i) ÷ (ii) near 1 is expected and says nothing about the
A78. What the laptop can show is (iii) ÷ (i) on Yi: the grouped conversion alone should give 0.65 to 0.72.

**The device cell (E-T18) is the same three arms** in a kit that already runs ggml graphs: about 10 s of phone
time. Its line: (i) ÷ (ii) ≤ 1.15 on Llama-2's shape closes the reordering for the full-attention subjects; Yi's
grouping stands or falls by (iii) ÷ (i) on Yi's shape.

Cost for Tom: about 80 lines in one function and its `work_size`; the tests are 7 runs of 2 to 4 minutes and
two perplexity pairs. It changes the CPU module, so a kit that carries it cannot ship r33's library unchanged.

## 4. Lists (additions only)

| # | test | where | cost | closes when |
|---|---|---|---|---|
| E-T21 | the head's probability per row written by the scoring script; then a per-step chooser (lookup, head, or no guess) fitted on the training split's held-out prompts, scored once on the set and on the official prompt | host, Tom's tools | minutes | under +0.02 tokens per pass on the held-out prompts → no chooser; the rule stays "the head" |
| E-T22 | a kit that runs the official prompt with a guess source prints guesses made, accepted and passes | device, with the first width-2 kit | 0 | the phone's count differs from the host's → the host's 1.143 / 1.391 are replaced by the phone's |

No new try item. E-Y4's price for Yi is unchanged; its prefill tile is not proposed.

## 5. Next questions, ranked

| # | question | state |
|---|---|---|
| 1 | Tom's KleidiAI stock result (flags, E-T13, E-T16) against the bands | waiting for his file |
| 2 | Is the official reply the same on the phone's path as in the dump? | **done, §1.2a:** Llama-2 yes on the host's phone-like path; Yi no |
| 3 | DeepSeek on the official prompt: with no head and no dump its tokens per pass is the 1.385 everyone assumes. A CPU dump of the official prompt and the 32 set prompts (no training) would at least give the lookup arms and the copy rate | host, CPU, Tom's dump tool, about 15 minutes |
| 4 | How much of Llama-2's official reply is copied from the prompt, and does the copy start late enough for a lookup to need a warm-up? | host, on Tom's `official.rows` and `official.ptok`, minutes |
| 5 | The chooser of E-T21 | Tom |
| 6 | R75 free reads | R75 |

## 6. Sources (2026-11-24)

Repository: `lab/c3/head/{yi,llama2}/official.rows`, `eval.rows`, every `*.pt.official.pred.npy` and
`*.pt.evalset.pred.npy`; `memo-09-work/tom-head.md` §4.1, §4.2; `lab/app-c3/lib/research.dart` (lines 16, 102,
217, 459 to 469); `lab/llama.cpp-base/ggml/src/ggml-cpu/ops.cpp` (the per-query function and its caller);
Bonnie's r21 to r23. No external source this round.
