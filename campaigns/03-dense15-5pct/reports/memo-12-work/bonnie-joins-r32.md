# Bonnie, memo-12 work, round 32: the coupling term refitted on every cold ggml Q4_0 cell; the body at the cap and at full clocks; how caps moved in campaign 2's runs

2026-11-24. Labels: *device* (phone), *host*, *estimated* (arithmetic), *source*. No web search (nothing is stated
about upstream's current state). R75's report is still not in the repository.

Scripts (same folder, both new, both read committed JSONs only):
- `bonnie_coupling_r32.py <repo root>`: the fit, the per-cell table, the body, the width-2 ratios (seconds).
- `bonnie_caps_r32.py <repo root>`: caps around campaign 2's runs (seconds).

## 0. Answer first

1. **The refit lowers the coupling term from 0.25 to 0.17** (*estimated* from 17 *device* cells; leave-one-group-out
   range 0.14 to 0.21). Root-mean-square error of the rule on the cells: 4.2 % fitted, 5.8 % as committed. The
   per-A78 hot rate that fits best is 11.3 GB/s at 2,227 MHz against the probe kernel's measured 11.54, so that
   input stands.
2. **The rule still misses the full-clock cells by 6 to 8 %, in every variant, and always in the same
   direction:** it expects the CPU to read faster at full clocks than at the 1,555 cap, and R74's cells do not
   (25.4 to 25.9 GB/s against 25.8 to 26.1). Those cells are one warm round on another arena, so I do not build
   a ceiling into the rule; I say that above about 2,150 MHz the phone shows no gain the rule can be trusted for.
3. **The body restated** (*estimated*; four threads on their cores; Yi / Llama-2 / DeepSeek, ms per token):

   | clock state | body |
   |---|---|
   | cap 1,555 MHz: the measured state (*device*, R72 L4) | 120.6 / 140.6 / 130.8 |
   | the fit at that cap | 119.3 / 139.5 / 130.8 |
   | full clocks, by the fit | 113.3 / 132.6 / 124.3 |
   | full clocks, by R74's own CPU-alone rate (25.64 GB/s) | 121.4 / 142.1 / 133.2 |
   | cap 1,555 with three A78 only, by the fit (*device*: 146.5 / 170.6 / 159.3) | 149.3 / 174.6 / 163.7 |

   **The stack's body is a capped-clock number and full clocks improve it by 0 to 6 %, no more.** The stock
   decode rate is therefore safe against the cap as long as the fourth thread is on the X1; losing that thread
   costs 21 to 25 % of the body.
4. **With 0.17 the width-2 ratios of r29 to r31 fall by 0.02 to 0.05** (§1.4). The priced lane is 1.07 to 1.16
   over the whole clock range, the lane with two GPU launches 1.10 to 1.31, the W-position kernel without a lane
   1.07 to 1.33. This is the opposite direction to r30's second form, which the same cells disfavoured (r31
   §1.4). **My centre for the priced lane's ratio is now 1.11 (band 1.07 to 1.18).**
5. **Caps in campaign 2, power saving off** (*device*, 384 runs in 7 sessions):
   - A gated run started below the maximum cap 4 times in 328.
   - **A run that started at the maximum with thermal headroom under 0.70 never had its cap lowered while it
     ran: 0 of 280.** Almost all of those runs were under 8 s long.
   - Without a gate, back to back: the cap was lowered by the end of the second run in 22 % of chains (about
     11 s of continuous load) and by the end of the third in 52 % (about 16 s).
   - Once lowered, it was back at the maximum after a 60 to 100 s wait in 18 of 20 cases, and never without a
     wait (0 of 9).
   - The first step down is 2,150 MHz (11 of 17 lowered starts); 1,555 was seen 4 times.
6. **What that means for a dense claim run.** Its load is one prefill and one decode from a gated start, 9 to
   20 s (*estimated*). Campaign 2 has almost no run that long (6 runs over 8 s among those that started at the maximum), so the nearest evidence is
   the ungated chains: **P(the cap is lowered before the decode ends) about 0.35** (*estimated*; I said 0.6 in r31
   and lower it), and the likely step is 2,150, where C is still near 50 and the ratios are the full-clock ones.
   **A 1,555 cap inside a gated claim run is the unlikely case (P 0.1), not the planning case.** r31's table for
   that cap stays as the bad end.
7. Power saving holds both clusters at 2,035 MHz for a whole session (5 sessions, 216 runs set aside). The claim
   rule has it off; the sheet's row 11 is what keeps C out of that state.

## 1. The refit

### 1.1 Cells, model, parameters

Every cold ggml Q4_0 cell on the KleidiAI buffer type in the committed campaign-3 probe reports (*device*): R72 L4
`arm=ggml_stock` (9: three subjects × three thread sets), R72 L2 `kern=ggml_q4_0` (3), R74 `x3 tog arm=cpu` (2),
R72 L5 `type=q4_0 regime=cold` on cpu 4 with the thread verified (3). R69 and R69b have no ggml cell (their
kernels are the probe's own).

    time per byte = max(1/R, 1/C) + α × min(1/R, 1/C)
    C = h × (A78 MHz summed over the A78 threads + x × X1 MHz) ÷ 2,227

- R: 29.9 GB/s for three or four threads (R72 block `a`; R69 and R69b read 28.8 to 31.4), 21.4 for one A78 (R69,
  R69b: 22.15, 20.59). Taken as independent of the clock.
- Each thread is taken at the top of its policy's `mhz_range` in that cell.
- The X1 thread is taken as on its core in the four-thread ggml cells; those cells verify only their first
  thread. In the 7,4,5 cells the first thread is cpu 7 and it is verified.

| fit | α | h (GB/s per A78 at 2,227 MHz) | x (X1 over A78, per MHz) | rms relative error |
|---|---|---|---|---|
| as committed | 0.25 | 11.54 | 1.44 | 5.8 % |
| α only | 0.19 | 11.54 | 1.44 | 4.3 % |
| **α and h** | **0.17** | **11.30** | 1.44 | **4.2 %** |
| α, h and x | 0.17 | 11.30 | 1.40 | 4.2 % |

Leave one group out (α and h refitted without it, error on it): without R72 L4 α 0.16, 3.7 %; without R72 L2
α 0.21, 7.4 %; without R74 α 0.14, 8.5 %; without R72 L5 α 0.175, 2.8 %.

### 1.2 Cell by cell (*device* measured; *estimated* fitted)

| cell | set | A78 / X1 MHz | measured GB/s | fit (α 0.17, h 11.3) | committed | C in the fit |
|---|---|---|---|---|---|---|
| R72 L4, Yi | 4,5,6 | 1,555 / – | 21.26 | 20.86 | 20.11 | 23.7 |
| R72 L4, Llama-2 | 4,5,6 | 1,555 / – | 21.36 | 20.86 | 20.11 | 23.7 |
| R72 L4, DeepSeek | 4,5,6 | 1,555 / – | 21.44 | 20.86 | 20.11 | 23.7 |
| R72 L4, Yi | 7,4,5 | 1,555 / 1,555 | 21.86 | 23.51 | 22.50 | 27.1 |
| R72 L4, Llama-2 | 7,4,5 | 1,555 / 1,555 | 24.35 | 23.51 | 22.50 | 27.1 |
| R72 L4, DeepSeek | 7,4,5 | 1,555 / 1,555 | 23.98 | 23.51 | 22.50 | 27.1 |
| R72 L4, Yi | 4,5,6,7 | 1,555 / 1,555 | 25.83 | 26.11 | 24.73 | 35.0 |
| R72 L4, Llama-2 | 4,5,6,7 | 1,555 / 1,555 | 25.91 | 26.11 | 24.73 | 35.0 |
| R72 L4, DeepSeek | 4,5,6,7 | 1,555 / 1,555 | 26.11 | 26.11 | 24.73 | 35.0 |
| R72 L2, arena | 4,5,6 | 2,227 / – | 27.75 | 26.00 | 24.59 | 33.9 |
| R72 L2, arena | 7,4,5 | 2,227 / 2,150 | 28.44 | 26.40 | 25.10 | 38.3 |
| R72 L2, arena | 4,5,6,7 | 2,227 / 2,150 | 26.67 | 27.12 | 26.06 | 49.6 |
| R72 L5, 4,096 inputs, pass 1 | 4 | 2,419 / – | 11.02 | 11.18 | 10.93 | 12.3 |
| R72 L5, 4,096 inputs, pass 2 | 4 | 2,419 / – | 11.17 | 11.18 | 10.93 | 12.3 |
| R72 L5, 11,008 inputs, pass 2 | 4 | 2,419 / – | 10.77 | 11.18 | 10.93 | 12.3 |
| R74 `tog`, position 0 | 4,5,6,7 | 2,419 / 2,841 | 25.86 | 27.47 | 26.53 | 57.6 |
| R74 `tog`, position 5 | 4,5,6,7 | 2,419 / 2,841 | 25.43 | 27.47 | 26.53 | 57.6 |

What the residuals say:
- **The capped cells fit to 2 % except one** (Yi on 7,4,5, −7 %; the same set reads 24.0 and 24.4 on the other
  two subjects).
- **R72 L2's three-thread arena cells are 7 % above the fit and above the four-thread cell of the same block.**
  Three threads at 2,227 MHz reading 27.8 to 28.4 GB/s are within 5 to 7 % of the read rate itself. Either the
  coupling is smaller still when the bus binds (0.09 by r31's reading), or the read rate there was above 29.9.
- **R74's two cells are 6 to 8 % below the fit.** Warm start (headroom 0.70 to 0.74), another round, an arena of
  576 MB walked while a feeder thread exists in the kit. No variant of the rule reaches them without spoiling
  L2.
- So the four clock states do not lie on one curve of this form to better than about ±7 %. **The rule is good
  to ±7 % on a width-1 rate; I would not quote it finer.**
- Limits: 17 cells but four clock states and two rounds; the nine L4 cells share a block; R is one number.

### 1.3 The body

Bytes per token from R72 L4's four-thread cells (body ms × GB/s): 3.11 / 3.64 / 3.41 GB. The table is §0 item 3.
Two readings of full clocks are given because the fit and the one full-clock measurement disagree by 7 %; the
truth for a dense decode is between them until a dense decode runs.

In stock tokens per second (950 ÷ (body + 8 + head), heads as R72): 6.90 / 6.21 / 6.22 at the cap (as committed)
and 7.3 to 6.9 / 6.5 to 6.1 / 6.5 to 6.1 at full clocks (*estimated*). **No clock state lifts stock by more than
0.4 tok/s.**

### 1.4 The width-2 ratio with the fitted coupling (*estimated*; Llama-2; K 300; `pair` / `linear`)

| C | α | lane, as priced | lane, W kernel, two launches | no lane, W kernel | W kernel alone |
|---|---|---|---|---|---|
| 32.2 (cap 1,555, low) | 0.25 | 1.18 / 1.19 | 1.32 / 1.34 | 1.38 | 1.35 |
| | **0.17** | 1.16 / 1.16 | 1.30 / 1.31 | 1.33 | 1.30 |
| 35.0 (cap 1,555, fit) | 0.25 | 1.15 / 1.16 | 1.28 / 1.30 | 1.29 | 1.27 |
| | **0.17** | 1.13 / 1.15 | 1.25 / 1.27 | 1.24 | 1.21 |
| 40 (the stacks) | 0.25 | 1.11 / 1.15 | 1.21 / 1.25 | 1.17 | 1.14 |
| | **0.17** | 1.09 / 1.13 | 1.18 / 1.23 | 1.11 | 1.08 |
| 46 (cap 2,035 to 2,150) | 0.25 | 1.09 / 1.13 | 1.15 / 1.23 | 1.10 | 1.06 |
| | **0.17** | 1.08 / 1.12 | 1.12 / 1.22 | 1.08 | 1.04 |
| 57.6 (full clocks) | 0.25 | 1.09 / 1.13 | 1.12 / 1.23 | 1.08 | 1.05 |
| | **0.17** | 1.07 / 1.12 | 1.10 / 1.22 | 1.07 | 1.03 |

- At α 0.14 (the low end of the leave-out range) every cell is 0.00 to 0.02 lower again.
- Width 1 is anchored on device rates in every cell, so the body and rungs 0 to 4 of r14 do not move. Only the
  second position's price moves: about +0.2 to +0.4 tok/s on the width-2 rungs against the committed 0.25
  (*estimated*: 2 to 4 points of a 120 ms pass at 10.9 tok/s).
- The `linear` model's ratio hardly depends on C above 40 (1.12 to 1.13 on the priced row). What is left there
  is the GPU's two-column factor, which R75 measures.
- **I do not change the committed stacks.** The proposal for the lead: new estimates use α 0.17 with 0.14 to 0.25
  as the band and state C with its clock state; r14's rungs are re-run once, after R75, with both.

## 2. Caps in campaign 2's runs (r31 next question 3)

### 2.1 Data

*Device*: every run with `cap_before` and `cap_after` in the kit JSONs under
`campaigns/02-25toks-5pct/reports/device/` (600 runs, 12 sessions). The caps are the two fast clusters'
`scaling_max_freq`. Five sessions sat at 2,035 / 2,035 throughout (r22, r23, r25, r26 and r28-c1; the last
records power saving on); they are set aside: 216 runs. The rest: 384 runs, 7 sessions, power saving off.

### 2.2 Results

| question | answer (*device*) |
|---|---|
| cap at the start of a gated run | below the maximum in 4 of 328 |
| cap at the start of an ungated run | below the maximum in 13 of 56 |
| by thermal headroom at the start | 0.60 to 0.70: 2 of 282 below; 0.70 to 0.80: 2 of 79; 0.80 to 0.90: 9 of 10; above 0.90: 4 of 13 |
| a run that starts at the maximum: lowered while it runs? | headroom under 0.70 at the start: **0 of 280**; 0.70 or more: 16 of 87 |
| the same by run length | under 5 s: 6 of 213; 5 to 6 s: 3 of 107; 6 to 8 s: 3 of 41; 8 to 12 s: 4 of 5 |
| ungated chain from the maximum, second run (about 6 s of load before it) | lowered by its end: 5 of 23 |
| the same, third run (about 10 s before it) | lowered at its start: 3 of 21; by its end: 11 of 21 |
| after a run that ended lowered, next run after no wait | back at the maximum: 0 of 9 |
| after a wait of 30 to 60 s | 2 of 3 |
| after a wait of 60 to 100 s | 18 of 20 |
| levels when lowered at a start | 2,150: 11; 1,785: 2; 1,555: 4 |

Reading:
- **The cap follows the headroom reading with a threshold near 0.8,** and the kit's gate (0.65) keeps a run's
  start far enough below it that a run of up to 8 s never crossed it.
- **Continuous three-thread load crosses it after roughly 11 to 16 s** from an ungated (warmer) start: a fifth
  of chains by 11 s, half by 16 s.
- **Recovery takes a wait of about a minute or more,** which is what the gate's wait already is in most runs.
- The 8 to 12 s row (4 of 5) is five runs that started warm (median headroom 0.74); it is not a rate for gated
  runs.

### 2.3 For a dense claim run (*estimated*)

- Load per run: a prefill of 167 to 186 tokens, then 32 decoded tokens at 6 to 12 tok/s: about 9 to 20 s
  (prefill 6 to 15 s is Ebony's range, r4 and r11; not re-priced here), on four threads, not three.
- From a gated start the crossing should come later than in the ungated chains, and four threads heat faster
  than three; I take those as cancelling and read the chain rows directly: **P(lowered before the decode ends)
  0.2 at 11 s, 0.5 at 16 s; 0.35 for the block as a whole.**
- The decode is the last 3 to 5 s of the run, so a cap that drops does so during or just before the timed part:
  it would show as a slow tail of tokens, not as a slow run from the start.
- The first step is 2,150 MHz. By §1.3 that costs the body under 3 % and leaves the width-2 ratio at its
  full-clock value. Only a second step (1,785 or 1,555) reaches r31's capped table.
- What decides it is the prefill's length, which is why the prefill is next question 1.

## 3. Lists (additions only)

| # | test | where | cost | closes when |
|---|---|---|---|---|
| B-T32a | B-T31a's fields, plus the cap sampled once a second during the run (the kit reads it only before and after today) | device, with the first dense decode kit | a few lines in the kit | cap lowered inside the decode in more than 3 of 10 runs → the claim's median is a capped-clock number and C 46 to 50 is the planning value; a step to 1,555 in any run → r31's capped table applies to that run |
| B-T31b | unchanged (one cold ggml arena cell at two cap states); §1.2 is the reason it is still wanted | – | – | – |

No try item. B-Y31 (the self-timed gate) is unaffected: it needs no clock model.

## 4. Next questions, ranked

| # | question | state |
|---|---|---|
| 1 | **How long is a dense prefill on the phone, and at what rate does it heat?** Priced from campaign 2's *device* prefill rates (`prefill_tps`, `ttft_ms` with the bytes per token of OLMoE and Granite) and the tile kernel's cost, per subject at four threads; it sets the 9 to 20 s of §2.3 and with it the cap risk | committed JSONs and arithmetic; **the top item I can do now** |
| 2 | R75 into r18's draft, r25's script, r27 §2.1, r29 §1.2 and §1.4 here | waits for the report |
| 3 | Headroom over a run: campaign 2 records `heat` and `heat_after` on ungated runs. Headroom gained per second of load, by thread count, gives the time from 0.65 to the 0.8 threshold directly | committed JSONs; second in line |
| 4 | DeepSeek's lookup arms and break-even line | waits for the owner on r28 row 40, then Tom |
| 5 | Llama-2's and DeepSeek's stock shifts against r11 | waits for Tom's five runs |
| 6 | r15's O and the second position's attention against Tom's host decode and KV screen | waits for Tom |

## 5. Sources (2026-11-24)

Repository: `device/r72-c3probe-r33-2026-11-22T1759.json` (lines `l2`, `l4`, `l5`, block `a`),
`device/r74-c3probe-r34-2026-11-23T1743.json` (lines `x3 tog`), `device/r69-…` and `r69b-…` (lines `p2 kern=read`);
the campaign-2 kit JSONs (`runs[].cap_before`, `cap_after`, `heat`, `gate`, `cool_ms`, `wall_ms`, `ttft_ms`;
`power_start`); `R72-result.md` §5; `memo-09-gap.md` §10.3; my r14, r29, r30, r31; `bonnie_reader_r30.py`,
`bonnie_ratio_r29.py`. No external source this round.
