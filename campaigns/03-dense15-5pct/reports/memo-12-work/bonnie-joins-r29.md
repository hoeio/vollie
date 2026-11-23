# Bonnie, memo-12 work, round 29: the width-2 pass cost ratio from the records that exist; do repeated stock runs give one reply

2026-11-24. Labels: *device* (phone), *host*, *estimated* (arithmetic), *source*. No web search (nothing is stated
about upstream's current state). R75's report and `tom-kai-stock.md` are still not in the repository.

Script: `bonnie_ratio_r29.py` (same folder; `python3 -I -B bonnie_ratio_r29.py`, seconds; it asserts two cells of
r27 §2.5 before printing). The reply count of §2 is a read of committed device JSONs with a few lines of my own
code, not stored; the fields are named so that it can be redone.

## 0. Answer first

1. **No device record times anything at two positions.** R72 and R74 measured one column everywhere (§1.1). So
   the records bound P2 ÷ P1 only through their one-column inputs, and the bound is wide: **1.06 to 1.52 on the
   lane** (*estimated*). The priced value, 1.09 to 1.15, sits near the bottom of it because it assumes both
   unbuilt kernels.
2. **Which unbuilt piece carries the ratio** (*estimated*, K 300, `pair` / `linear`, Llama-2; Yi is 0.02 lower,
   DeepSeek equals Llama-2):

   | what exists | P2 ÷ P1 | against r27's lines (gate at 1.20; width 2 pays below about 1.33 Yi / 1.42 Llama-2) |
   |---|---|---|
   | W-position CPU kernel and batched GPU kernel (as priced) | 1.11 / 1.15 | rule T as it is |
   | W-position kernel, GPU by two launches | 1.21 / 1.25 | **gate pays** |
   | one-row CPU kernel run twice, batched GPU kernel | 1.24 / 1.25 | **gate pays** |
   | neither: one-row kernel twice, two launches | 1.43 / 1.45 | **width 2 does not pay** on any subject |
   | stock batch 2 as llama.cpp has it (KleidiAI's four-row tile) | 1.60 / 1.62 | width 2 loses 15 to 20 % |

   **Either kernel missing puts the ratio at the gate's line; both missing closes width 2.** The W-position
   kernel is therefore not one lever among several: without it and without the batched GPU kernel there is no
   multi-token method to score. This is the order the lead already has (kernel before lane); the table is its
   price.
3. **Without a lane (CPU alone) the ratio hangs on one number that has no clean device value:** the hot compute
   rate C. With the W-position kernel: 1.10 at C 46, 1.17 at C 40, 1.30 at C 34.6 (Llama-2). 34.6 GB/s is the
   only *device* part of C (three A78, R72 L3); the X1's share was never read on-core. At 34.6, width 2 on the CPU
   alone is at Yi's break-even.
4. **What R75 will and will not settle.** Its `x4m` cell gives cg (the batched GPU kernel at two columns) and its
   ggml two-column cell gives the tile's cost; both pick a row of the table. It does not time the W-position
   kernel, which does not exist in the tree. Reading line for cg, with the W-position kernel assumed: the ratio
   reaches 1.20 at cg about 1.9 (`pair`) or 1.6 (`linear`).
5. **Ten stock runs do give one reply, on the two campaign-2 subjects:** 560 official-prompt runs in 12 cells of
   (model, arm, thread count), over up to 8 sessions per cell, **one reply text per cell, no exception**
   (*device*, §2). The same text at 3 and at 4 threads, and the same from the stock runtime and from the patched
   runtime with its knobs off. I raise my P for the dense subjects from 0.9 to 0.97.
6. Two things the kits already record, so r28's B-T28a shrinks: `prompt_tokens` per run (the count) and `text`
   (the reply). Only the first token id and the id hashes are new fields.

## 1. P2 ÷ P1

### 1.1 What the records hold (*device*) and what they do not

| input of a width-2 pass | record | value | at two positions |
|---|---|---|---|
| body, one position, stock | R72 L4 | 120.6 / 140.6 / 130.8 ms | not measured |
| CPU alone; CPU beside the GPU; GPU (one column) | R74 `togsum` | 25.64; 19.77; 21.53 GB/s | not measured |
| GPU device-time rate; bus sum | R74 `gpu_busy`, fitted | 24.6; 46.1 GB/s | – |
| join cost | R74 `x3 rt` | 452 µs | believed independent of width |
| hot compute, three A78 | R72 L3 | 34.6 GB/s | gives the one-row kernel twice; the X1 was off-core in every hot cell |
| head | R72 L7 | 9.04 / 4.40 / 13.88 ms | paid per committed token, outside the ratio |
| attention, O | none: no dense decode has run | – | – |

R72's K1, K1m and k1m4 cells are column kernels for sparse inputs, one position; they are not the W-position
kernel. So the four factors that make a second position cheap or dear are all non-device:

| factor | centre (class, r14) | corner that needs nothing new |
|---|---|---|
| κ: CPU cost of two positions over one | 1.45, W-position kernel (host, random data, M4) | 2.0: the one-row kernel twice. Stock llama.cpp does worse: its batch 2 takes KleidiAI's four-row tile, 2.60 by issue count (r5, *source*) |
| cg: GPU time at two columns over one | 1.26, batched kernel (model) | 2.0: two launches of R74's kernel |
| C: hot CPU compute rate | 40 GB/s (guess with a device part) | 34.6 (*device*, three A78) |
| second position's attention | 2.3 / 3.6 / 3.4 ms with the shared conversion (*estimated*, r15, r17) | 3.2 / 5.3 / 5.0 without |

### 1.2 The ratio at the corners (*estimated*; `pair` / `linear`; Yi | Llama-2 | DeepSeek)

| corner | K 300 µs | K 452 µs (as R74 measured) |
|---|---|---|
| low: W kernel, batched GPU kernel at no extra cost (cg 1.0), C 46 | 1.06 / 1.06 \| 1.07 / 1.07 \| 1.07 / 1.07 | the same |
| **as priced:** W kernel, cg 1.26, C 40 | 1.09 / 1.13 \| 1.11 / 1.15 \| 1.11 / 1.15 | 1.11 / 1.13 \| 1.12 / 1.14 \| 1.12 / 1.14 |
| W kernel, two GPU launches | 1.19 / 1.22 \| 1.21 / 1.25 \| 1.21 / 1.25 | 1.18 / 1.21 \| 1.20 / 1.23 \| 1.20 / 1.23 |
| one-row kernel twice, batched GPU kernel | 1.22 / 1.23 \| 1.24 / 1.25 \| 1.24 / 1.25 | 1.27 / 1.28 \| 1.27 / 1.27 \| 1.27 / 1.27 |
| nothing new: one-row twice, two launches, no shared conversion | 1.39 / 1.41 \| 1.43 / 1.45 \| 1.43 / 1.45 | 1.40 / 1.42 \| 1.42 / 1.44 \| 1.42 / 1.44 |
| the same at C 34.6 | 1.47 / 1.48 \| 1.51 / 1.52 \| 1.51 / 1.52 | 1.48 / 1.50 \| 1.50 / 1.51 \| 1.50 / 1.51 |
| stock batch 2 (tile 2.60), two launches | 1.56 / 1.58 \| 1.60 / 1.62 \| 1.60 / 1.62 | 1.58 / 1.59 \| 1.59 / 1.61 \| 1.59 / 1.61 |

Without a lane (CPU alone at its bus rate, 25.64 GB/s):

| | Yi | Llama-2 | DeepSeek |
|---|---|---|---|
| W kernel, C 46 | 1.08 | 1.10 | 1.10 |
| W kernel, C 40 | 1.16 | 1.17 | 1.17 |
| W kernel, C 34.6 | 1.29 | 1.30 | 1.30 |
| one-row kernel twice, C 40 | 1.50 | 1.52 | 1.52 |
| stock batch 2 (tile), C 40 | 1.87 | 1.89 | 1.89 |

Reading:
- K hardly moves the ratio (the joins are paid once per pass at either width). The ratio is a statement about
  the two kernels and C.
- **The break-even ratios are r27's tokens per pass on held-out prompts, 1.34 (Yi) and 1.42 (Llama-2);
  DeepSeek has no head, so its line is unknown** (1.385 if it is the mean of the others).
- The stock tile row is why "just decode two tokens per batch" was never a lever: at 1.6 it is slower than
  single decodes with any guess source we have. This is r5's count turned into a pass time; R75's ggml
  two-column cell is its device check.
- On the CPU alone the W-position kernel leaves width 2 worth something only if C is 40 or more. That makes the
  X1's on-core hot rate a blocking unknown for any no-lane width-2 design (choice A's world, r17).
- The two-term reader (the bus-against-compute rule every cell uses) is a fit through two phone points (r14
  factor 16). Every ratio above moves with it; I have no second form to bracket it with, and say so.

### 1.3 What follows for the guess rule

| R75 and kernel state | ratio | rule |
|---|---|---|
| both kernels as priced | 1.09 to 1.15 | rule T |
| one of the two missing or slower than priced | 1.19 to 1.28 | **rule T + gate** (one number, r27 §2.5): +1.0 to +2.3 % (*estimated*) |
| both missing | 1.39 or more | no width-2 method; the campaign is back at rung 4 of r14 (width 1 on the lane) |

So the gate is not dead, as r27 §0 item 5 read: it is the fallback for the middle row, and the middle row is
where a first width-2 build is likely to land (P 0.5 that the first device ratio is 1.19 or more). I keep r27's
recommendation that Tom builds nothing for it now; the one number is fitted in a minute when the device ratio
exists.

## 2. Do repeated stock runs give one reply? (r28 next question 4)

*Device*: every committed report under `campaigns/02-25toks-5pct/reports/device/` and `reports/device/` whose
`runs[]` carry `text`, `model`, `arm`, `kind` = `official32` and `tokens` = 32. Cell = (model, arm, threads,
repack, loaded file, prompt tokens).

| model | arm | threads | runs | report files | distinct reply texts |
|---|---|---|---|---|---|
| OLMoE | off | 3 | 192 | 8 | **1** |
| OLMoE | off | 4 | 4 | 1 | 1, the same text |
| OLMoE | stock (stock runtime) | 3 | 12 | 1 | 1, the same text |
| OLMoE | c1 (campaign 1's runtime) | 3 | 24 | 2 | 1, the same text |
| OLMoE | num (numerics arm) | 3 | 68 | 6 | 1, another text |
| OLMoE | num | 4 | 4 | 1 | 1, as num at 3 threads |
| Granite | off | 3 | 32 | 3 | **1** |
| Granite | off | 4 | 4 | 1 | 1, the same text |
| Granite | stock | 3 | 12 | 1 | 1, the same text |
| Granite | num | 3 | 180 | 8 | 1, another text |
| Granite | num | 4 | 4 | 1 | 1, as num at 3 threads |
| Granite | c1 | 3 | 24 | 2 | 1, a third text |

560 runs, 12 cells, warm-ups and scored runs together. Prompt tokens are 156 (OLMoE) and 177 (Granite) in every
run.

- **Inside a cell the reply never changed**, across sessions days apart, heat states and clock caps.
- **Thread count did not change it** (3 against 4), which agrees with Ebony's host finding for single decodes.
- **A different runtime build did not change it when the arithmetic is declared identical** (stock, and the
  patched runtime with knobs off). That is the bit-identity rule seen from the phone.
- **Arms that change numerics give another reply**, as they should: on a 32-token reply a declared numerics
  change is visible as text. For Granite, campaign 1's runtime gives a third text (it carries campaign 1's own
  numerics settings).
- Campaign 1's early reports (R1 to R12) show two or three texts per prompt, but their runs are different knob
  arms without an `arm` field (expert skip, the two-stage head); the controls among them agree. Not evidence
  against.

Limits: it is text, not token ids (two id sequences with one text are possible in principle, not in 32 tokens of
this kind); two sparse models, not the dense subjects; the same app, cache type and prompt batching throughout.
The dense subjects add nothing to the stock path that could make it non-deterministic (one more attention layer
type is not involved; the path is the same per-query attention). **P(ten stock runs of a dense subject give one
reply) 0.97** (*estimated*).

For the claim sheet's row 6 (r28): the stock arm is a fixed reference inside a session, so "method reply equals
stock reply" is a clean test, and **it can already be scored from the `text` field of today's kit with no new
field**. The id hashes of B-T28a remain worth adding (they are what the term says), but they do not block.

## 3. Lists (additions only)

| # | test | where | cost | closes when |
|---|---|---|---|---|
| B-T29a | when R75 is in: read cg from `x4m` and the tile's cost from the ggml two-column cell, and place both in §1.2's table | arithmetic on *device* | minutes | tile under 2.0 × a gemv on the phone → r5's 2.60 is wrong and "stock batch 2" moves up the table; cg above 1.9 → gate row even with the W kernel |
| B-T29b | a hot cell with a thread verified on cpu 7 (the X1), same kernel as R72 L3, so that C has a four-core device value | device, a rider of about 5 s in any probe kit | 0 | C under 37 → no-lane width 2 is at break-even on Yi; C above 43 → the priced ratios are high by 0.03 to 0.05 |

No try item. r27's B-T27c (the first width-2 kit prints both pass times) stays the direct measurement; §1 is what
to expect from it.

## 4. Next questions, ranked

| # | question | state |
|---|---|---|
| 1 | Is there an on-core X1 hot reading anywhere in R69, R72's `ramp_ok=1` subset or R74 that gives C a four-core device value without a new cell (B-T29b)? | a read of committed JSONs; **the top item I can do now** |
| 2 | R75 into r18's draft, r25's script, r27 §2.1 and §1.2 here (B-T29a) | waits for the report |
| 3 | The two-term reader bracketed: a second functional form through the same two phone points, to put a band on every ratio of §1.2 | arithmetic; small |
| 4 | DeepSeek's lookup arms and break-even line | waits for the owner's answer on r28 row 40, then Tom |
| 5 | r15's O and the second position's attention against Tom's host decode and KV screen | waits for Tom |
| 6 | Softmax time on the A78 | only if the gate is built on the top-1 probability; the free logit gap is tried first |

## 5. Sources (2026-11-24)

Repository: `R72-result.md` (§3, §7), `R74-result.md` (§4, §6); my r5 (tile count), r14 (factor table), r15, r17,
r27, r28; `bonnie_stack_r13.py` and the scripts it loads; device JSONs `r17` to `r31` and the `r30-chain` files
under `campaigns/02-25toks-5pct/reports/device/`, `r1` to `r16` under `reports/device/` (fields `runs[].text`,
`model`, `arm`, `kind`, `threads`, `repack`, `loaded_file`, `prompt_tokens`, `tokens`). No external source this
round.
