# Bonnie, memo-12 work, round 31: C at the clocks of a claim block; where the decode threads sat in campaign 2; the r11 Yi prediction scored

2026-11-24. Labels: *device* (phone), *host*, *estimated* (arithmetic), *source*. No web search (nothing is stated
about upstream's current state). R75's report is still not in the repository.

Scripts (same folder): `bonnie_threads_r31.py` (new; `python3 -I -B bonnie_threads_r31.py <repo root>`). The clock
and placement reads of §1 are of named fields in the committed probe JSONs; the ratios come from
`bonnie_reader_r30.py`'s functions at other values of C.

## 0. Answer first

1. **The two device anchors of the stack were taken at opposite ends of the clock range, and read the same.**
   (*device*)

   | anchor | clocks | threads on their cores | four-thread ggml Q4_0 rate, cold |
   |---|---|---|---|
   | R72 L4 body (120.6 / 140.6 / 130.8 ms) | **A78 and X1 both capped at 1,555 MHz** | first thread verified; the neighbouring probe arm of the same set shows 4 of 4 | 25.83 / 25.91 / 26.11 GB/s |
   | R74 `tog` CPU alone (25.64 GB/s) | **full: 2,419 / 2,841 MHz**, caps at the hardware maximum | 4 of 4 in all 30 samples, cpu 7 included | 25.86 and 25.43 |

   So a width-1 pass is bus-bound over the whole range, and the model's assertion (C above 31) holds **when the
   fourth thread is on the X1**. With three A78 alone at the 1,555 cap it does not: those cells read 21.3 to
   21.9 GB/s and bodies of 142 to 171 ms (*device*).
2. **C is not one number. It is 32 to 34 GB/s at the 1,555 cap and 52 to 59 at full clocks** (*estimated* from
   *device* parts, all four threads on-core; §1.2). The stacks' 40 is neither state.
3. **That makes the cap state the largest single factor of the width-2 ratio** (*estimated*; Llama-2, K 300,
   form A; `pair` / `linear`):

   | | cap 1,555 (C 32 to 34) | full clocks (C 52 to 56) | r27's lines |
   |---|---|---|---|
   | lane, both kernels as priced | 1.16 to 1.19 | 1.09 / 1.13 | gate at 1.20 |
   | lane, W kernel, two GPU launches | **1.29 to 1.34** | 1.12 / 1.23 | width 2 pays below 1.34 (Yi), 1.42 (Llama-2) |
   | no lane, W kernel | **1.31 to 1.38** (Yi 1.29 to 1.36) | 1.08 to 1.09 | |

   At full clocks every width-2 design is cheap. At the cap the priced lane sits on the gate's line and the two
   cheaper designs sit on Yi's break-even.
4. **In campaign 2's real decodes the X1 was used every time once the pool and mask were on:** 312 runs, 619
   samples with all workers running, cpu 7 held in 100 % of them, no thread on cpus 0 to 3 (*device*, §2). The
   probes' problem (the cpu-7 thread pushed off its core) does not show in the app's decode path. Caps were at
   the hardware maximum before and after 287 of those 312 runs.
5. **But those were 3 threads, a sparse model and 1.3 to 1.6 s of decode.** A dense subject runs 4 threads
   through a prefill of 6 to 15 s first. R72's own L4 cells are the nearer precedent: after 1.5 s cells at four
   threads the cap sat at 1,555 for Llama-2 and DeepSeek throughout. **My expectation for a dense claim block:
   cap below maximum in most runs (P 0.6), so C nearer 34 than 52.**
6. **A correction to r30 §2.** There is one verified on-core X1 reading of the Q4_0 probe kernel; it is cold,
   not hot, and I missed it: 18.91 GB/s on cpu 7 at 2,841 MHz against 11.1 to 11.3 on cpu 4 at 2,419, single
   thread (*device*, R72 L1, 11,008 inputs). **X1 = 1.69 × an A78, 1.44 per MHz.** r30's "no hot reading" stands;
   its "1.4 to 1.5 on other kernels" now has a Q4_0 companion that is a little higher.
7. **r11's Yi prediction: inside.** One paragraph, §3.
8. **A design that follows from item 3 (try list): the gate should time itself.** The runtime knows the duration
   of its own width-1 and width-2 passes, so the break-even hit probability can be computed live from a running
   mean instead of being fitted offline for one assumed ratio. It follows the cap without reading any system
   file, and cannot change the reply.

## 1. C at the clocks of a claim block (r30 next question 1)

### 1.1 The records (*device*)

R72, L4, `arm=ggml_stock` (the cells that give the stack's body):

| subject | set | body ms | GB/s | cap, A78 / X1 | `ramp_ok` |
|---|---|---|---|---|---|
| Yi | 4,5,6 | 146.49 | 21.26 | 1,555 | 1 |
| Yi | 7,4,5 | 142.49 | 21.86 | 1,555 / 1,555 | 1 |
| Yi | 4,5,6,7 | **120.55** | 25.83 | 1,555 / 1,555 | 1 |
| Llama-2 | 4,5,6 | 170.56 | 21.36 | 1,555 | 1 |
| Llama-2 | 7,4,5 | 149.62 | 24.35 | 1,555 / 1,555 | 1 |
| Llama-2 | 4,5,6,7 | **140.61** | 25.91 | 1,555 / 1,555 | 1 |
| DeepSeek | 4,5,6 | 159.26 | 21.44 | 1,555 | 1 |
| DeepSeek | 7,4,5 | 142.39 | 23.98 | 1,555 / 1,555 | 1 |
| DeepSeek | 4,5,6,7 | **130.77** | 26.11 | 1,555 / 1,555 | 1 |

- The ggml cells print placement for the first thread only (`cpu0_ok=1`, no off-core call). The probe's own
  stock arm on 4,5,6,7 at position 5, same cap, has all four on-core (`cpu_ok=4/4`) and reads 27.55 / 25.66 /
  27.51 GB/s.
- Three A78 at this cap are 4.5 GB/s slower than four threads. At full clocks they are not: R72 L2's ggml cell
  on 4,5,6 at a 2,150 cap reads 27.75 GB/s, above the four-thread cell (26.67). **The fourth thread matters
  only when the clock is capped,** which is the bus-against-compute crossing seen from the phone.

R74, `x3 tog`, CPU arms (positions 0 and 5): 25.86 and 25.43 GB/s; `max_mhz` p4 2,419 of 2,419 and p7 2,841 of
2,841; `mhz_range` A78 2,419, X1 2,841; `thr` off-core count 0 of 30 for cpus 4, 5, 6 and 7; `ramp_ok=1`; thermal
headroom 0.70 to 0.74 (a warm start, as R74's result says).

### 1.2 C in the two states (*estimated* from *device* parts)

| part | value | from |
|---|---|---|
| one A78, hot, at 2,227 MHz | 11.54 GB/s | R72 L3 (20.52 G weights/s) |
| X1 over an A78 | 1.69 at 2,841 against 2,419 MHz; 1.44 per MHz | R72 L1 cold, one verified line; the plain-kernel cells of r30 give 1.4 to 1.5 at the same clocks |

| state | three A78 | X1 | C |
|---|---|---|---|
| both clusters capped at 1,555 MHz | 24.2 | 8.1 (equal to an A78 per MHz) to 10.4 (1.44 per MHz, R72 L1 scaled) | **32 to 34.5** |
| the same, fourth thread off its core | 24.2 | 2.2 (R72 L3's small-core rate) | 26.4 |
| full clocks, 2,419 / 2,841 | 37.6 | 14.7 to 21.2 | **52 to 59** |
| R72 L3 as it ran (2,227, fourth thread off-core) | 34.6 | 2.2 | 36.9 (*device*) |

The X1 ratio is a cold single-thread reading turned into a hot rate, and the A78 rate is the probe's kernel, not
KleidiAI's. Both are the best the records give.

### 1.3 The ratios at those values (*estimated*; Llama-2; K 300; `pair` / `linear`)

| C | form | lane, as priced | lane, W kernel, two launches | no lane, W kernel (Yi) | W kernel alone, two positions over one |
|---|---|---|---|---|---|
| 32.2 | A | 1.18 / 1.19 | 1.32 / 1.34 | 1.38 (1.36) | 1.35 |
| 32.2 | B, p 2.35 | 1.18 / 1.19 | 1.30 / 1.32 | 1.31 (1.30) | 1.29 |
| 34.4 | A | 1.16 / 1.17 | 1.29 / 1.31 | 1.31 (1.29) | 1.28 |
| 34.4 | B | 1.17 / 1.18 | 1.28 / 1.31 | 1.28 (1.26) | 1.25 |
| 40 | A | 1.11 / 1.15 | 1.21 / 1.25 | 1.17 (1.16) | 1.14 |
| 52 | A | 1.09 / 1.13 | 1.12 / 1.23 | 1.09 (1.08) | 1.06 |
| 52 | B | 1.11 / 1.15 | 1.17 / 1.26 | 1.14 (1.12) | 1.11 |
| 56 | A | 1.09 / 1.13 | 1.12 / 1.23 | 1.08 (1.07) | 1.05 |

- **The lane is the design least hurt by a cap.** Beside the GPU the CPU's bus share is smaller, so its second
  position stays hidden behind the bus longer: 1.09 to 1.19 over the whole range, against 1.08 to 1.38 without
  a lane.
- **At the cap, r30's form question disappears** (the two forms agree to 0.01 on the priced row) and the clock
  question replaces it.
- For the claim: if a dense block runs capped, the priced lane is at 1.16 to 1.19 and the one-number gate of
  r27 §2.5 is worth about +0.5 to +1 %; if the GPU kernel is by two launches, width 2 is at Yi's break-even.

### 1.4 What the same records say about the rule itself

The rule's coupling term (0.25 × the smaller of bus and compute time) can be read off four cold ggml cells, taking
the read rate as 29.9 GB/s and C from §1.2 (*estimated* from *device*):

| cell | C taken | measured GB/s | coupling that reproduces it |
|---|---|---|---|
| R72 L4, four threads, cap 1,555 | 32 to 34.5 | 25.8 to 26.1 | 0.16 to 0.17 |
| R72 L4, three A78, cap 1,555 (compute-bound) | 24.2 | 21.3 to 21.4 | 0.16 |
| R72 L2, three A78, cap 2,150 | 34.6 | 27.75 | 0.09 |
| R74 `tog`, four threads, full clocks, warm | 52 to 59 | 25.64 | 0.29 to 0.33 |

- The committed 0.25 is inside the spread, and the spread is wide. The two capped cells, which are the cleanest
  (same block, same cap, `ramp_ok=1`), agree on 0.16 to 0.17.
- **Form B is disfavoured by the same cells:** it predicts the four-thread rate to rise by 17 % from the cap to
  full clocks, form A by 8 %, and the phone shows about 0 (25.8 to 26.1 against 25.64; different cell shapes and
  a warm start on the full-clock side, so ±4 % is the most I would claim). r30's bracket is therefore generous
  on the high side.
- I have not refitted anything. A refit across cells with unequal shapes is next question 1, and its honest
  form is a model-free cell that runs one arena at whatever caps the phone happens to be in and prints them.

## 2. Where the decode threads sat in campaign 2 (r30 next question 4)

*Device*: every `official32` run with `thread_samples` in the kit JSONs under
`campaigns/02-25toks-5pct/reports/device/`. A sample is (time, running workers, the cpu of each). Only samples
with all workers running are counted.

| model | threads | pool and mask | runs | samples | cpu 7 held | a thread on cpus 0 to 3 | two workers on one cpu | caps at maximum before and after |
|---|---|---|---|---|---|---|---|---|
| OLMoE | 3 | `VOLLIE_POOL=1`, mask 0xf0 | 180 | 362 | **1.00** | 0.00 | 0.02 | 164 of 180 |
| Granite | 3 | the same | 132 | 257 | **1.00** | 0.00 | 0.03 | 123 of 132 |
| OLMoE | 3 | no pool | 96 | 181 | 0.98 | 0.22 | 0.06 | 29 of 96 (58 at 2,035) |
| Granite | 3 | no pool | 96 | 23 | 1.00 | 0.13 | 0.22 | 27 of 96 (43 at 2,035, 15 at 1,555) |
| OLMoE | 4 | no pool | 8 | 17 | 0.94 | 0.82 | 0.18 | 0 of 8 (all 2,035) |
| Granite | 4 | no pool | 8 | 2 | 1.00 | 0.00 | 1.00 | 0 of 8 |

- **With the pool and the mask, three workers sat on three of cpus 4 to 7 and one of them was always cpu 7.** X1
  clock at the end of the run: 2,841 MHz (median; at its floor in 2 of 312 runs).
- **Without the pool** a worker was on a small core in 13 to 22 % of samples, and at four threads in 82 % on
  OLMoE. The four-thread rows are without the pool only: **no committed record shows four pooled workers on four
  cores in a real decode.** That is the configuration the R73 plan sets (r22 row 28).
- The no-pool runs sat mostly at a 2,035 cap and the pooled runs at the maximum. The sessions differ (the
  no-pool rows are the earlier rounds), so I do not read that as an effect of the pool.
- For C: the placement half of the X1 question is answered for three pooled threads (the X1 is used), open for
  four. The clock half is §1: campaign 2's short sparse decodes kept full clocks in 92 % of pooled runs; nothing
  says a dense block will.

## 3. The r11 prediction for Yi, scored

r11 (committed before any run) predicted for the 40-chunk stock run on a KleidiAI build: Yi +0.3 to +1.0 % over
the freeze build's value, Llama-2 within ±0.05 %, DeepSeek unknown. `tom-kai-stock.md` has one finished run: Yi at
`-b 512`, 7.7347 against 7.6966, **+0.50 %** (*host*). **Inside the band**, and at the proxy's own figure
(+0.00536 in −log p is +0.54 %). Tom's note quotes my band as "−0.3 to +1.0 %"; r11's text is +0.3 to +1.0 %, and
the result is inside either. With 40 chunks resolving about 0.2 %, the reading is "+0.3 to +0.7 %", so it also
sits inside Ebony's revised band (+0.3 to +1.2 %) and cannot tell the two apart. **Unscored:** Yi at `-ub 1` (a
partial at chunk 9 reads +0.56 %, which is not a score); Llama-2 at both settings, which is the sharper half of
the prediction (±0.05 % is below what 40 chunks resolve, so it can fail but hardly pass on 40 chunks alone);
DeepSeek at both. One caution for the reading on the phone: Tom's log shows the host's 8-bit head kernel entry
is the i8mm one, and the A78 will take the dotprod one. The integer arithmetic is the same by construction
(*source*, not checked on a node), so I expect the same values; E-T13 on the head's node would show it.

## 4. Lists (additions only)

**Tests**

| # | test | where | cost | closes when |
|---|---|---|---|---|
| B-T31a | the first dense decode kit records, per run, the caps before and after, the per-cpu clocks at run end, and `thread_samples` for the four workers (all exist as kit fields today) | device, free | 0 | cap below maximum in most runs → C 32 to 34 is the planning value; a worker off cpus 4 to 7 → C 26 and width 1 is itself compute-bound |
| B-T31b | one cold ggml Q4_0 arena cell, three and four threads, run at the start and at the end of a block so that two cap states are seen, caps printed | device, rider, about 6 s | 0 | gives the coupling term of §1.4 on one shape |

**Try**

| # | lever | after | label | price | P |
|---|---|---|---|---|---|
| B-Y31 | **self-timed gate:** the runtime keeps a running mean of its own width-1 and width-2 pass times and of the exact head time, computes p* = (P2 − P1) ÷ (time per token − head) live, and makes a guess only when the source's calibrated hit rate for the step (r27's table: match length, confidence bin) is above it | the first width-2 build | bit-identical (it decides what is verified, never what is committed) | bytes 0; time: two clock reads per pass; value: r27 §2.5's gate gain at whatever ratio the phone is in, without choosing a ratio beforehand: 0 at full clocks, +1 to +2.5 % at 1.20 to 1.30, and it stops width 2 by itself above break-even | 0.5 that it is worth 1 % in a dense block |

The calibration table of B-Y31 is fitted on training data as in r27 §2.4; only the threshold moves at run time.
One width-1 pass every N passes keeps P1 measured when every step guesses (N is a parameter; 16 costs under
0.5 %).

## 5. Next questions, ranked

| # | question | state |
|---|---|---|
| 1 | The coupling term refitted on every cold ggml Q4_0 cell of R69, R72 and R74 with its own clocks and thread set (about 15 cells), and the stack's body at full clocks and at the cap from that fit | a read of committed JSONs and arithmetic; **the top item I can do now** |
| 2 | R75 into r18's draft, r25's script, r27 §2.1, r29 §1.2 | waits for the report |
| 3 | How long after the gate does a cap stay lifted in campaign 2's runs (cap before against cap after, by position in block and by prefill length)? It bounds what a dense prefill of 6 to 15 s does to the cap | committed JSONs; second in line |
| 4 | DeepSeek's lookup arms and break-even line | waits for the owner on r28 row 40, then Tom |
| 5 | Llama-2's and DeepSeek's stock shifts against r11 when Tom's five runs finish | waits for Tom |
| 6 | r15's O and the second position's attention against Tom's host decode and KV screen | waits for Tom |

## 6. Sources (2026-11-24)

Repository: `device/r72-c3probe-r33-2026-11-22T1759.json` (lines `l1`, `l2`, `l3`, `l4`; fields `arm`, `set`,
`body_ms`, `gbs`, `cpu_ok`, `cpu0_ok`, `thr`, `mhz_range`, `max_mhz`, `ramp_ok`), `device/r74-c3probe-r34-2026-11-23T1743.json`
(lines `x3 tog`, `x3 togsum`, `x3 cond`), `R72-result.md` (§2, §5), `R74-result.md` (§2); the campaign-2 kit JSONs
(`runs[].thread_samples`, `threads`, `env`, `cap_before`, `cap_after`, `mhz_after`); `tom-kai-stock.md`; my r11,
r22, r27, r29, r30; `bonnie_reader_r30.py`. No external source this round.
