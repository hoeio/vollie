# Bonnie, memo-12 work, round 34: why R75's row split read 0.923 where the stage model said 0.663

2026-11-25. Labels: *device* (phone), *host*, *estimated* (arithmetic), *source*. Inputs: R75-result.md (both
sessions), `reports/device/r75a1-c3probe-r36-2026-11-24T1345.json` (session 2, x4s and x4m lines), R74's JSON for
one comparison, and the probe source (`patches/llama.cpp-c3/0007-x4-join-probe.patch`, function `x4s`). Two web
searches were made for §3.4 only (kernel mechanisms; hypotheses, not findings). Ebony's r18 was not read before
§0 to §6 were written; §7 compares.

Script (same folder, new): `python3 -I -B bonnie_split_r34.py <repo root>` (seconds; loads `bonnie_floor_r14.py`
and through it the earlier scripts unchanged; its self-check reproduces R75 S2.10's rung 2 and rung 7 and the 0.663).

## 0. Answer first

1. **The 0.923 is not one number.** The 22 arms of x4s ran through four phone states in 63 s (§1). The two arms
   averaged into "share 0.55" are a throttle-onset arm (pos 8, 1.880 ms) and a capped arm (pos 13, 1.508 ms), and
   they are divided by the mean of a cool cpu arm (pos 0, 1.776 ms) and a fully capped one (pos 21, 1.895 ms). All
   *device*. ABBA did not cancel the drift because the drift was not linear: slow, fast, slow.
2. **Three terms make the gap, and only the third is the one the lead named** (*estimated*, §2):

   | step | best ratio |
   |---|---|
   | stage model as priced (K 149 beside the rows, no launch term, no end slack) | 0.663 |
   | the same with the rates of the cool arms (pos 1 to 4) | 0.660 |
   | + the launch is serial in the probe: `enq_us` 214 µs with all four CPU threads idle | 0.749 |
   | + end slack near balance, as the arms show it (fit on five arms) | 0.832 |
   | the same cell in the capped state C (pos 12 to 14) | 0.816 |
   | the same cell at throttle onset, state B (pos 5 to 9) | 1.005 |
   | printed (a B arm and a C arm over an A and a D cpu arm) | 0.923 |

3. **At cool clocks the CPU rows beside the GPU ran as priced.** In pos 1 to 4 (no cap, GPU at 840 MHz) the CPU
   reads 18.5 to 22.3 GB/s while the GPU kernel runs, the GPU 24.2 to 26.0, the sum 44.6 to 47.9 (*estimated* from
   *device* lines; the committed bus sum is 46.1). S10 at share 0.25 in that state is 1.10 against 1.14 priced.
   The 1.62 at share 0.50 is the mean of a throttle-onset arm and a capped arm (1.82 and 1.42 on the printed
   means; 2.02 and 1.46 on the measured window against the cool cpu arm); no arm at 0.50 ran cool. The model's cool value there is 1.32 (*estimated*).
4. **What was wrong in my pricing, in size order:** (a) no serial launch term: the lead thread spends 200 to 245 µs
   in `launch` + `clFlush` before the pool starts, and K_meas (GPU QUEUED to START) is that same interval within
   20 µs, so the "GPU dead time" was priced as time the CPU works through and the phone spent it with the CPU
   idle; (b) the end slack: 80 to 270 µs when the GPU ends within 0.4 ms of the CPU rows, against my 0.107 × K
   (17 µs) and Ebony's 60 µs; (c) nothing for the state: under throttling the CPU rate beside the GPU halves
   (9 to 14 GB/s) while the GPU keeps 21 to 24.
5. **Refit of the CPU-beside-GPU term** (§3): CPU slowdown while the GPU kernel runs, cool **1.38 (1.28 to 1.54)**
   against 1.45 priced (inside); capped state C **1.57 (1.55 to 1.61)**; throttled states B and D **2.2 to 2.8**.
   As a split factor (CPU-alone rate ÷ the sum beside each other): cool **0.61 (0.58 to 0.64)** against 0.589
   committed; C 0.665; D 0.775; B 0.85. All *estimated* from *device* lines.
6. **K_eff 162.3 is a mixture too.** The rule's two shares took one state-B arm (191, 179 µs) and one state-C arm
   (114, 112 µs). The four cool arms read 206 to 209 µs (their `wait_ms` is under 0.1, so the rule left them out).
   Cool in-situ K is **207 + 13.3 = 220 µs**, and it is serial as built.
7. **Ladder** (§4, *estimated*): the lead's rungs at K_eff 162.3 hold only if the launch is hidden and the slack
   is 17 µs. As built in x4s at cool clocks the lane at width 1 is **7.8 to 7.9 / 7.0 to 7.2 / 7.0 to 7.1 tok/s**
   (stock 6.90 / 6.21 / 6.22): **+0.8 to +1.0**, not +0.5 and not +2.7. The committed top rung (7) is 9.4 to 10.0
   / 9.0 to 9.6 / 8.8 to 9.3 as built. **No rung reaches 12 on all three unless the persistent kernel F exists in
   a fast version, the end slack is zero, and the `pair` contention form holds**; with slack 60 µs DeepSeek stays
   at 10.5 to 11.7.
8. **P(12 on all three) stays about 0.002 (0.0005 to 0.005, judgement)**, §4.3. R75's split cell is not new
   evidence against it once the states are separated; the serial launch and the slack are, and they are what F
   would have to remove.
9. **Artefact or physics** (§5): the serial launch is the probe's design (but any lead on a worker pays some of
   it); the mixed states are the probe's arm order on a phone that was hot at the start (the gate timed out at
   0.735); the collapse of the CPU rate in states B and D is the phone, and the file cannot say which mechanism.
10. **Cheapest cell that separates them** (§6, proposal only): the same lead cell, shares 0.40 to 0.55, each
    split arm bracketed by a cpu arm in the same second and by a 0.1 s compute canary, run once from a passed
    gate and once hot. About 25 s of load. My prediction for the cool run is a best ratio of 0.83 (0.78 to 0.86).

## 1. The four states of x4s (*device* lines; the grouping is mine)

One arm is 0.5 s warm-up and 1.5 s measured, plus ring allocation; 22 arms in 63 s. The gate before the block timed
out at headroom 0.735 after x4m's 61 s of load. Battery level 15 % before the block.

| state | pos | CPU cap (policy 4 / 7), MHz | GPU max, MHz | `hr` | cpu arm, same state |
|---|---|---|---|---|---|
| A cool | 0 to 4 | 2,419 / 2,841 (none) | 840 | 0.753 to 0.793 | pos 0: 1.776 ms, 28.57 GB/s |
| B throttle onset | 5 to 10 | 2,419 → 2,150 / 2,841 → 2,150; cpu 7 seen at 1,785 to 2,150 | 840 | 0.793 to 0.903 | none |
| C capped | 11 to 14 | 1,785 / 1,785 | 676 | 0.987 | none |
| D capped low | 15 to 21 | 1,420 / 1,420 | 676, then 491 (pos 20, 21) | 1.033 to 1.057 | pos 21: 1.895 ms, 26.77 GB/s |

Per lead arm, from the script (cpu = `ms_per_mat` − `enq_us` − `wait_ms`, because the printed `cpu_ms` is a mean
that includes the warm-up; rb = CPU GB/s while the GPU kernel runs, taking the CPU at its alone rate for the rest;
all *estimated* from *device* fields):

| pos | state | share | `ms_per_mat` | enq µs | cpu ms | wait ms | K µs | GPU GB/s | rb GB/s | rb + GPU |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | A | 0.20 | 1.728 | 216 | 1.511 | 0.001 | 209 | 25.6 | 22.3 | 47.9 |
| 2 | A | 0.25 | 1.683 | 214 | 1.467 | 0.002 | 207 | 24.2 | 21.2 | 45.4 |
| 3 | A | 0.30 | 1.622 | 213 | 1.408 | 0.001 | 206 | 25.2 | 20.6 | 45.8 |
| 4 | A | 0.35 | 1.610 | 214 | 1.395 | 0.001 | 207 | 26.0 | 18.5 | 44.6 |
| 5 | B | 0.40 | 1.853 | 230 | 1.619 | 0.004 | 218 | 23.6 | 10.3 | 34.0 |
| 6 | B | 0.45 | 1.830 | 234 | 1.589 | 0.007 | 221 | 23.5 | 10.8 | 34.3 |
| 7 | B | 0.50 | 2.017 | 203 | 1.796 | 0.018 | 186 | 22.4 | 5.8 | 28.2 |
| 8 | B | 0.55 | 1.880 | 210 | 1.621 | 0.049 | 191 | 23.1 | 9.2 | 32.3 |
| 9 | B | 0.60 | 1.964 | 200 | 1.596 | 0.168 | 179 | 21.8 | 10.5 | 32.3 |
| 12 | C | 0.60 | 1.659 | 126 | 1.184 | 0.349 | 114 | 23.9 | 17.2 | 41.0 |
| 13 | C | 0.55 | 1.508 | 124 | 1.232 | 0.152 | 112 | 24.6 | 17.7 | 42.3 |
| 14 | C | 0.50 | 1.518 | 147 | 1.294 | 0.077 | 135 | 23.6 | 17.9 | 41.5 |
| 15 | D (cap falling) | 0.45 | 1.593 | 233 | 1.345 | 0.015 | 222 | 24.0 | 18.3 | 42.3 |
| 16 | D | 0.40 | 1.819 | 239 | 1.575 | 0.005 | 226 | 22.6 | 13.7 | 36.3 |
| 17 | D | 0.35 | 1.900 | 238 | 1.660 | 0.002 | 226 | 23.5 | 11.6 | 35.1 |
| 18 | D | 0.30 | 1.974 | 245 | 1.727 | 0.002 | 233 | 22.5 | 10.9 | 33.4 |
| 19 | D | 0.25 | 1.985 | 239 | 1.744 | 0.002 | 227 | 21.2 | 12.3 | 33.5 |
| 20 | D | 0.20 | 2.065 | 238 | 1.825 | 0.002 | 227 | 17.6 (491 MHz) | 12.5 | 30.1 |

The feeder arms, same configuration, 2.4 s apart: pos 10 (state B) 2.022 ms, pos 11 (state C) 1.516 ms (*device*).

Readings from the table:
- **Lower caps did not make the cell slower; state C is the fastest.** The order is A good, B bad, C good, D bad.
  The loss is not a function of the printed CPU cap.
- **The GPU hardly moves** (21 to 26 GB/s over all four states, 17.6 at 491 MHz). The CPU beside it moves by a
  factor of two. The bus sum falls from 45 to 48 (A) to 28 to 36 (B, D).
- **The CPU alone is nearly untouched by the cap**: 26.77 GB/s at 1,420 MHz against 28.57 at full clocks, 6 %
  (*device*, pos 21 and pos 0). r32's refit of the coupling (α 0.17) gives 25.8 and 27.5 for these two cells
  (*estimated*), both 4 % low, so the cell is one more out-of-sample point for α 0.17 against 0.25 (24.4 at the cap).
- **The printed `enq_us` + `cpu_ms` + `wait_ms` equal `ms_per_mat` within 15 µs in the eight arms whose state
  held (pos 4, 12 to 14, 16 to 19)**, and miss by up to 130 µs where the warm-up and the window differ (pos 1, 7).
  The launch is in series with the rows.

## 2. The cell, term by term

### 2.1 The launch is serial as built (*source*, *device*)

`x4s`, lead mode, per matrix: `launch(...)`, `clFlush`, **then** `ggml_backend_graph_compute` on the pool, then
`spin_ev`. The pool on cpus 5, 6, 7 has nothing to do until the lead returns from the flush.
- The phone shows it: workers' CPU-over-wall is 0.84 to 0.88 in every lead arm with no wait, and 1 − enq ÷
  `ms_per_mat` is 0.87 to 0.88. The `P` label the parser puts on cpus 5 to 7 in lead arms is this idle time
  (CW_MIN 0.9), not preemption. The cpu arms read 0.99 and `C`.
- `enq_us` is 213 to 216 µs in state A, 200 to 234 in B, 124 to 147 in C, 233 to 245 in D. In session 1 the same
  calls cost 12 to 38 µs per launch without a flush and 92 to 211 with one, on an idle phone (x4a). I have no
  explanation for C's lower value; it is not the GPU duty (pos 9 at 73 % busy reads 200, pos 14 at 72 % reads 147).
- `gpu_ms` − `gpu_dev_ms` (the rule's K) equals `enq_us` minus 7 to 21 µs in all 18 arms. The GPU starts when the
  flush returns. **K_meas measures the flush**, and in the probe the CPU does not work through it.

The stack's stage form is t = K + (B − ca·K) ÷ (cb + g): the CPU runs at its alone rate during K. With the launch
serial the stage costs Lh + B ÷ (cb + g). One microsecond of serial launch costs what 2.4 µs of K cost in the
stack (1 ÷ (1 − 0.589)). 214 µs serial is K 520 in the stack's terms: no better than R74's feeder (452).

That is what the phone says too: lead at share 0.25 in state A reads 0.948 of the cpu arm (pos 2 over pos 0);
the feeder at 0.25 in state C reads 0.854 over pos 0 and 0.828 over its own state (pos 11). My S2 was 0.86.

### 2.2 End slack (*device*; the fit is *estimated*)

`wait_ms` is not max(0, GPU end − CPU end). With the GPU's END taken from `gpu_ms`:

| pos | GPU ended before the CPU rows by | `wait_ms` |
|---|---|---|
| 8 | 0.41 ms | 0.049 |
| 14 | 0.21 | 0.077 |
| 9 | 0.20 | 0.168 |
| 13 | 0.09 | 0.152 |
| 12 | −0.10 (GPU late) | 0.349, so 0.25 beyond the lateness |
| 7, 15 | 0.66, 0.39 | 0.018, 0.015 |

The lead's voluntary context switches go with it: about 0 per matrix in pos 1 to 6, 0.36 in pos 8, 0.57 to 0.65
in pos 9, 13, 14, 1.00 in pos 12. The lead blocks inside `clGetEventInfo` when it polls an event that is not yet
complete, and pays a wake-up. Session 1 saw the same on cpu 4 (`finish` and `wfe` near 960 µs, spin bimodal).
Fit used by the script: 0.2 ms at balance, falling to 0 when the GPU ends 0.4 ms early. This pushes the best share
to the CPU-bound side (0.42 in state A) and costs about 0.08 of ratio at the optimum. My slack (0.107 × K = 17 µs)
was too small by a factor of 5 to 15; Ebony's 60 µs was nearer.

### 2.3 The cell in each state (*estimated*; ratio to the cpu arm of the same state)

| state | as built | launch hidden behind the rows | F (K 18 µs, slack 60) | no launch, no slack |
|---|---|---|---|---|
| A cool | **0.832** (share 0.42) | 0.786 | 0.665 | 0.629 |
| B throttle onset | 1.005 | 0.933 | 0.885 | 0.851 |
| C capped | 0.816 | 0.785 | 0.707 | 0.671 |
| D capped low | 0.951 | 0.885 | 0.810 | 0.777 |

Band on the cool as-built value: **0.78 to 0.86** (rb 18.5 to 22.3, slack 0.1 to 0.27 ms). The model reproduces
every lead arm it was fitted on within 8 % (script self-check; pos 7, 15 and 20 are left out because the cap or the
GPU clock changed during the arm). The best measured arm of each state over the cpu arm of that state: A 0.907
(share 0.35, the highest share that ran cool), C 0.823 (0.55), D 0.841 (pos 15, cap falling), B 1.021.

"Launch hidden" means the pool starts the CPU rows while the lead is in the flush, and the lead then joins. It
needs rows that are claimed finely enough; §9 reads the source: they are claimed, but too coarsely by default.

## 3. What each of the lead's five items contributes

### 3.1 CPU clocks and caps

- The caps do not explain the loss by themselves: the CPU alone loses 6 % at 1,420 MHz, and state C (1,785) is
  faster than state B (2,150 to 2,227).
- They mark the states. Every arm at share 0.40 or more ran in B, C or D. No arm above 0.35 ran cool.
- R74's split cell also ran under caps (2,150 from its first arm, 1,785 from the fourth; battery 67 %, *device*),
  12 to 20 s after its together cell began. Lane-like load reached a CPU or GPU cap within about 12 to 35 s in three
  blocks out of three (R74 x3c, R75 x4m, R75 x4s; *estimated* from arm positions). That is r33's heat question, now with the GPU on.

### 3.2 DRAM contention (the refit)

CPU rate while the GPU kernel runs, against the CPU alone in the same state (*estimated* from *device*):

| state | cells | CPU slowdown | band | split factor ca ÷ (cb + g) |
|---|---|---|---|---|
| cool | x4s pos 1 to 4; x4m one-column together cell (26.38 → 20.0 while the GPU runs, GPU at 676); R74 tog (25.64 → 18.9) | **1.38** | 1.28 to 1.54 (tog cells 1.32, 1.36) | **0.61** (0.58 to 0.64) |
| capped, C | pos 12 to 14 | 1.57 | 1.55 to 1.61 | 0.665 |
| throttled, D | pos 16 to 20 | 2.2 | 2.0 to 2.5 | 0.775 |
| throttle onset, B | pos 5, 6, 8, 9 (pos 7: 4.9) | 2.8 | 2.6 to 3.1 | 0.85 |

- The committed term (X 0.45, so 1.45; factor 0.589) is inside the cool band. **The cool refit moves the ladder by
  −0.2 tok/s, not more** (§4).
- Within state A the slowdown rises arm by arm (1.28, 1.35, 1.39, 1.54 at shares 0.20 to 0.35). Shares and time
  are confounded there; state D, where the shares run the other way, shows no such trend (2.0 to 2.5, flat). I
  read it as the phone warming, at 0.5.
- In B and D the committed two-term rule fails by a factor of about two with any clock I put in: at the 1,420
  cap and a 25 GB/s bus share it gives 22 GB/s for the CPU; the phone gives 11 to 14. The GPU's 21 to 24 GB/s
  did not rise to take what the CPU lost. So this is not the bus being shared differently; the sum fell.

### 3.3 The feeder

- pos 10 and pos 11 are the same arm 2.4 s apart: 2.022 and 1.516 ms. The feeder cell's printed 0.964 is the mean
  of a state-B and a state-C arm over the A and D cpu arms, as the lead cell's is.
- In state C the feeder cell at 0.25 reads 0.854 over pos 0 (S2 priced 0.86). Where there is no serial launch and
  the state is good, the R74 fit holds.
- In pos 10 the CPU threads were on their cores (CPU-over-wall 0.965 to 0.984) and the GPU ran about a quarter of
  the time, yet the rows took 2.02 ms against 1.33 for the same rows alone at state A's rate. The CPU was slow
  when the GPU was not running. State B is a state of the CPU side, not only contention.

### 3.4 Thread placement (`thr`)

- No `M` label on any arm: every thread stayed on its core.
- `P` on cpus 5 to 7 in lead arms is the serial launch (§2.1). `U` is the cap.
- Involuntary switches on cpus 5 and 6 run at 100 to 800 per arm in lead and feeder arms and 0 to 5 in cpu arms:
  something of the driver's wakes on those cores. It does not track the loss (pos 1: 773 on cpu 6, the best
  state; pos 12: 2). Time lost to it is at most 3.5 % of one thread (pos 10, cpu 6).
- The KleidiAI Q4_0 path does not give fixed shares: it claims row chunks, 16 for four threads (§9). A thread
  that is slow or late still holds the end back by up to one chunk.

What states B and D are, the file cannot say. Three candidates, none shown by a line of R75:
- **Clock below the printed one.** The big cores are limited in hardware under heat or current, and the node the
  probe reads shows the requested step. Upstream's `qcom-cpufreq-hw` takes hardware throttling through an
  interrupt and reports it as pressure on capacity; whether `scaling_cur_freq` shows it depends on the kernel
  version (web search today; the phone's vendor kernel was not read). Fits: the loss is on the CPU only, and pos 10
  is slow without the GPU. Against: pos 21 alone at the 1,420 cap is fine, so it would have to engage only when
  both run (current).
- **DRAM frequency voted down.** Qualcomm's bus governors vote the DDR and cache-bus frequency from measured CPU
  traffic and from a CPU-frequency map (web search today, Qualcomm's documentation; SM8350's tables were not
  read). A CPU that draws less beside the GPU, at a low clock, could lose its vote. Fits D against C. Does not
  fit B (higher clocks than C).
- **Memory reclaim during the block.** MemAvailable was 2.2 GB with 3.0 GB in zram at the start, and every arm
  allocates and frees its rings. x4r saw swap move by 0.7 GB in 21 s. No per-arm memory line exists in x4s.

Battery was at 15 % in x4s and 67 % in R74's capped split, so a low battery is not needed for the caps.

### 3.5 Session drift

- Session 1's cpu-4 spin `ovh_us` 401.4 against session 2's 131 to 140 (flag set by the lead). Inside session 2,
  the same launch reads 214 µs in x4s state A and 124 to 147 in state C. The launch cost moves by a factor of
  1.7 inside one block and 3 between sessions. No K from R75 is a constant of the phone.
- K_meas 149.0 is the mean of state-B arms (191, 179) and state-C arms (114, 112). Cool arms read 206 to 209.
  For a claim run from the gate I would use **K 207 (124 to 245) + H 13.3**, serial as built.

## 4. The ladder (*estimated*; tok/s Yi / Llama-2 / DeepSeek)

`bonnie_floor_r14.py` unchanged; `t2.body` replaced by the same stage loop with a serial launch term and a fixed
slack; rates as committed or refitted; α 0.25 and 0.17 (r32). Width 2 is two launches of the general kernel
(cg 2.0), because `_mc3` read 3.316 and is dropped; +40 µs for the second enqueue (*estimated*). Rung numbers are
R75 S2.10's.

### 4.1 Cool refit centre (cb 18.9, g 23.1, factor 0.610); `linear` | `pair`, α 0.17

| join | rung 2: width 1, O 8, measured head | rung 6: width 2, two-stage head | rung 7: + C5 0.93, head × 0.5 |
|---|---|---|---|
| stock (rung 0) | 6.90 / 6.21 / 6.22 | | |
| lead's form: K_eff 162.3 beside the rows, slack 17 µs | 9.41 / 8.75 / 8.53 | 10.64 / 10.45 / 10.05 \| 11.20 / 11.04 / 10.58 | 11.49 / 11.25 / 10.87 \| 12.09 / 11.88 / 11.44 |
| **as built in x4s, cool**: serial 207 + 13.3, slack 115 | **7.76 / 7.04 / 6.98** | 9.01 / 8.64 / 8.40 \| 9.25 / 8.88 / 8.61 | 9.70 / 9.28 / 9.06 \| 9.96 / 9.50 / 9.26 |
| as built, state C's launch (124 + 13.3) | 8.12 / 7.48 / 7.38 | 9.19 / 8.87 / 8.60 \| 9.58 / 9.32 / 9.01 | 9.90 / 9.48 / 9.25 \| 10.27 / 9.93 / 9.65 |
| launch hidden (unbuilt): serial 52 + 13.3, GPU dead 155, slack 115 | 8.17 / 7.55 / 7.44 | 9.32 / 9.03 / 8.75 \| 9.67 / 9.44 / 9.12 | 10.01 / 9.63 / 9.38 \| 10.38 / 10.07 / 9.78 |
| F, fast persistent kernel (unbuilt): K 17.95, slack 60 | 9.57 / 8.89 / 8.65 | 10.62 / 10.43 / 10.03 \| 11.27 / 11.11 / 10.63 | 11.46 / 11.23 / 10.85 \| 12.17 / 11.95 / 11.51 |
| F, slack 0 (upper bound) | 10.38 / 9.58 / 9.26 | 11.35 / 11.08 / 10.61 \| 12.10 / 11.84 / 11.29 | 12.32 / 11.98 / 11.53 \| **13.14 / 12.81 / 12.28** |

With the committed rates (factor 0.589) every cell is 0.1 to 0.3 higher; at the bad end of the cool band (0.640)
0.1 to 0.2 lower; at α 0.25 the width-2 cells are 0.2 to 0.4 lower. The script prints all of them.

### 4.2 Which rungs reach 12.00 on all three

Four cells of the script's 288, all of them **rung 7 with F at slack 0 in `pair`**: committed rates at α 0.25
(13.01 / 12.67 / 12.16) and α 0.17 (13.40 / 13.07 / 12.51); cool refit centre at α 0.17 (13.14 / 12.81 / 12.28);
bad end at α 0.17 (12.96 / 12.62 / 12.11). None in `linear`. None with slack 60 (DeepSeek 10.5 to 11.7). None
with any launch the phone has run.

### 4.3 P(12 on all three), judgement

What the passing cells need, with my probability for each given the ones before:
- a persistent, row-claiming version of the fast kernel that keeps about 24 GB/s (the probe's ran at 1.83): 0.3;
- end slack near zero in situ, not 60 to 115 µs (a shared-memory flag removes the driver's wake-up; the CPU-side
  jitter stays unless both sides claim rows): 0.4;
- `pair`, not `linear`, at heavy draw on both sides (R75 read `pair` at light draw only; the heavy one-column
  cell lost 21 % as R74 did): 0.4;
- the W-position CPU kernel at its priced cost: 0.5; C5 at 0.93 inside the budget: 0.5;
- the head at half its measured time with the guess accuracy the stack assumes (the trained head's top-1 was
  below the line): 0.3;
- the claim run stays in state A (short run from a passed gate): 0.8; file-backed residency on Llama-2 and
  DeepSeek: 0.7.

Product 0.002. The terms are not independent, so **0.002, band 0.0005 to 0.005**. The books say 0.002 to 0.003
and the lead about 0.001; I see no reason in R75's split cell to move it. The lane is worth **more than +0.5
tok/s at cool clocks on all three even as built (+0.8 to +1.0)**, about +1.2 to +1.4 with the launch hidden,
+2.4 to +2.7 with F at slack 60, and nothing in state B.

## 5. Artefact or physics

| item | which | why |
|---|---|---|
| arms of one share in different states; cpu arms at the two ends | **probe** (arm order, 63 s block, hot start after x4m) | pos 10 and 11; §1 |
| printed `cpu_ms`, `wait_ms`, `enq_us` include the 0.5 s warm-up; S10 is built on `cpu_ms` | **probe** | the patch's own comment; pos 7 window 1.796 against printed 1.668 |
| launch in series with the rows, all four threads | **probe design**; partly physics | a lead on a worker always loses its own 124 to 245 µs; the other three threads need not |
| launch 124 to 245 µs in situ against 12 to 38 isolated | **physics** (in situ) | the flush is the cost; x4a's `batch8flush` 92 to 211 |
| end slack 80 to 270 µs near balance | **physics of this join** (event polling through the driver) | lead's voluntary switches; F's flag would not have it |
| CPU slowdown 1.28 to 1.54 beside the GPU, cool | **physics**, as priced | three cells, two sessions |
| CPU slowdown 2.2 to 2.8 in states B and D | **physics of a throttled phone**; mechanism unknown | §3.4; not the claim state if the gate is passed and the run is short |
| caps within about 12 to 35 s of lane-like load | **physics**; a cost of the lane the stack does not carry | R74 x3c, R75 x4m, x4s |
| K_meas 149.0 | **mixture** | §3.5 |

## 6. The cheapest cell that tells them apart (proposal; needs its own pre-registration and the owner's go)

Kit r36's x4s code with a different arm order and three printed additions. No new kernel.

- **Arms:** triplets cpu (0.5 s) | lead split (1.0 s) | cpu (0.5 s), no warm-up arm between them, at shares 0.40,
  0.45, 0.50, 0.55; then one feeder triplet at 0.25. Each triplet's ratio uses its own two cpu arms.
- **Canary:** after every arm, 0.1 s of the hot sdot loop on cpu 5 (L2-resident, R72's P2 cell). Its rate follows
  the true clock and not the bus. A cpu arm that keeps its rate while the canary falls is a hidden clock limit; a
  cpu arm that falls while the canary holds is the bus or memory.
- **Printed:** medians and p90 per matrix of enq, cpu, wait over the measured window only; `/proc/vmstat`
  `pgscan`, `pswpout`, `allocstall` and per-cpu `/proc/stat` deltas per arm; `cpuinfo_cur_freq` beside
  `scaling_cur_freq` if readable.
- **Two passes:** (1) cool: after a passed r16 gate, 10 s of rest between triplets, void if any cap or GPU
  `thermal_pwrlevel` above 0 appears; about 10 s of load. (2) hot: straight after 30 s of the one-column together
  cell, the same triplets without rests; about 15 s.
- **What it decides:** pass 1 best ratio ≤ 0.86 → the 0.923 was state mixing and the serial launch, the cool
  model stands (my prediction **0.83, 0.78 to 0.86**, best share 0.40 to 0.50; rb 18.5 to 22.3; enq 200 to 220).
  Pass 1 ≥ 0.90 with no cap → the CPU term is wrong at high share even cool, and the ladder's lane rungs fall to
  the lead's +0.5. Pass 2 with the canary says which of §3.4's candidates states B and D are.
- **Cost:** about 25 s of load, 2 to 6 min with gates. Bytes 0. It changes no price above by itself; it removes
  the doubt about which column of §4 is the phone.

## 7. Against Ebony's r18 (read after §0 to §6 were written; nothing above was changed for it)

**Where the two agree, reached separately:**

| | Bonnie r34 | Ebony r18 |
|---|---|---|
| the 0.923 mixes phone states; the slow stretch is pos 5 to 10 | yes (state B) | yes ("bad stretch") |
| the cell without it | 0.83 (0.78 to 0.86), model on the cool arms | 0.82 (0.80 to 0.85), the pos 13 arm |
| CPU slowdown beside the GPU, cool | 1.38 (1.28 to 1.54) | 1.36 (1.28 to 1.45) |
| the committed coupling is inside the cool band | yes | yes (1.353) |
| the launch is in series with the rows and K_meas is the flush | yes | yes |
| a tail after the GPU's END before the spin sees it | 80 to 270 µs (slack fit) | 150 to 350 µs |
| lane as probed, width 1, cool | 7.76 / 7.04 / 6.98 | 7.59 / 6.82 / 6.78 |
| F, rung-7 form, `pair`, slack 60 | 11.83 / 11.61 / 11.19 (refit, α 0.25) | 11.86 / 11.66 / 11.24 |
| 12 on all three | only F at slack 0 in `pair` | nothing |
| states B and D | not explained by the file | not explained by the file |

**Where they differ, and what I make of it:**
1. **Size of the as-probed join.** Hers: 400 µs serial (340 to 570), read off the three GPU-critical arms. Mine:
   220 µs serial + 115 µs slack = 335 µs at the best share, which sits on the CPU-bound side where the slack is
   smaller. Her arms are in states B and C; mine is the cool launch (207) with a slack fitted on the same arms.
   The 0.2 tok/s between our as-probed rows is this. I keep 335 as the centre and take her 400 as the upper half
   of the band; her T3 (percentiles of END to seen) settles it.
2. **Fixed term in the capped state.** She fits the 1,420 arms as 151 µs fixed + slowdown 1.53; I put it all in
   the rate (2.2). Her form fits the five arms better (the excess grows by about 100 µs over 0.33 ms more overlap,
   where a pure rate gives 180). **I take her form for state D.** The cell values are the same either way.
3. **Fixed term in the cool arms.** Her 47 µs + 1.36 is fitted on the printed `cpu_ms`, which includes the 0.5 s
   warm-up. On the measured window (`ms_per_mat` − enq − wait) the four cool arms give excess 90, 135, 165, 241 µs
   at overlaps 0.391 to 0.680: no positive fixed term, and a slowdown that rises from 1.28 to 1.54. Same centre;
   the difference is whether the cool arms were already warming. Her T4 in both share orders answers it.
4. **Why the launch is 124 to 147 µs in pos 12 to 14.** Her reading: a launch into a GPU that has just finished
   is 100 µs cheaper. Counter-example in the file: pos 9 (GPU critical, wait 0.168 ms, launch straight after)
   reads 200. I leave it unexplained; her T10 is the right cell.
5. **Design G is better than my "launch hidden" row.** Mine hides the launch and keeps the event spin (slack 115):
   8.17 / 7.55 / 7.44. Hers adds a done flag written by a trailing kernel, which removes the driver from the end
   of the join: 8.41 / 7.85 / 7.72. Bit-identical, uses the upstream kernel. **I withdraw my row in favour of G**
   and agree with her order: G before F. Her F-over-G line (only at a persistent rate of 0.9 or more) is the
   right form of the question; my §4.3 has 0.3 for that rate.
6. **Her P4 to P6 (kgsl timer, the screen, spin heat) are costs of F that my §4.3 does not carry.** With them my
   0.002 is, if anything, high. I do not move it; the band covers it.

**One list for the lead** (tests first; device items need the owner and a pre-registration):
- Tests: her T1 (pool stall, host) and T2 (done flag on ordinary kernels, host) first; then her T4 as the device
  cell, **with my §6 additions** (each split arm bracketed by cpu arms in the same second, the compute canary,
  per-arm memory and per-cpu lines, a cool pass and a hot pass) and her T3 percentiles; her T5 if the canary
  points at the bus; T6 to T9 only after G.
- Try: G (her L1 + L2); L6, the self-timed split, costs nothing and would have refused the split in 9 of the 18
  lead arms; F only if T7 reads 0.9 or more and T9 passes.

## 8. Next questions, ranked

1. **Does the pool claim rows or give fixed shares, and can the lead join a graph that is already running?**
   (*source*, campaign 1's pool and KleidiAI's Q4_0 path.) It decides whether "launch hidden" is a 20-line change
   or a new scheduler, and it is worth +0.4 tok/s at width 1 on all three (§4.1). Mine; I continue with this.
2. **F's two open costs:** the rate of a persistent fast kernel and the in-situ slack of a flag join when both
   sides claim rows. Host design with Tom (R75's next step 3); every passing cell of §4.2 hangs on them.
3. **Heat with the GPU on:** seconds to the first cap at lane power from a passed gate, from R74 and R75's gate
   and `hr` series, set against r33's CPU-only figures. Decides whether a claim run stays in state A.
4. **States B and D:** which mechanism (§3.4). Needs §6's canary; nothing more can be had from the file.
5. **The flush itself:** why 124 to 147 µs in state C and 214 in state A; whether `cl_qcom_perf_hint` or a
   recordable queue (44.3 µs per launch replayed, x4d) removes most of the serial launch without F.
6. **α:** add pos 0 and pos 21 to r32's fit (two cells, both 4 % above it).

## Sources

- Hardware throttling in upstream `qcom-cpufreq-hw` (LMh interrupt, reported as capacity pressure):
  https://lore.kernel.org/linux-pm/20210624115813.3613290-4-thara.gopinath@linaro.org/ and the driver source
  https://gbmc.googlesource.com/linux/+/refs/heads/dev-6.6/drivers/cpufreq/qcom-cpufreq-hw.c
- Qualcomm bus governors (BWMON traffic vote, static CPU-frequency map for LLCC and DDR):
  https://dragonwingdocs.qualcomm.com/System/Kernel/configure-the-dynamic-voltage-and-frequency-scaling-dvfs-governors
  and https://docs.qualcomm.com/doc/80-70015-10/topic/2-performance-features.html
- Lahaina LLCC and DDR devfreq nodes in a vendor tree:
  https://gitlab.e.foundation/e/devices/android_kernel_fairphone_fp5/-/commit/3a2d0a2e08966a94c7219677ba0644fca4c95200

## 9. Next question 1, first pass: does the pool claim rows, and can the lead join late? (*source*, *estimated*)

Read: `lab/llama.cpp-c3` (commit `26f78f925` on base `620d2225c`), `ggml/src/ggml-cpu/kleidiai/kleidiai.cpp`, the
Q4_0 forward, and `ggml_barrier` in `ggml-cpu.c`. *Source* facts:
- The weight rows are cut into **nth × `chunk_multiplier` chunks; the multiplier is 4 by default** and is read
  from the environment variable `GGML_KLEIDIAI_CHUNK_MULTIPLIER`. Four threads: 16 chunks.
- Thread i starts on chunk i; after that every thread takes the next free chunk
  (`ggml_threadpool_chunk_add`). So rows are claimed, with one reserved first chunk per thread.
- One `ggml_barrier` stands before the claim loop ("publishes both LHS packing and the initialized dynamic chunk
  queue"). All threads must reach it. A launch placed on thread 0 **before** that barrier stalls the other three
  as x4s did; placed **after** it, the other three run on.

What it means for "launch hidden" and for Ebony's L1 (*estimated*, the script's §4; state A, share 0.42, CPU rows
1.269 ms on four threads, lead 0.214 ms late):

| chunk multiplier | CPU rows end, ms after the workers start |
|---|---|
| serial launch, as x4s | 1.483 |
| 4 (default) | **1.483** |
| 8 | 1.428 |
| 16 or 32 | 1.348 |
| ideal (the lead's 0.214 ms shared by four) | 1.323 |

- **At the default multiplier, hiding the launch gains nothing.** The lead starts its reserved chunk late and
  still takes a quarter of the chunks; its last one ends 0.214 ms after the others'. The gain needs a
  multiplier of 16: 135 of the 160 µs that are there.
- So the lever is two parts: the launch on thread 0 after the op's barrier (a hook in the Q4_0 forward or a small
  custom op; about 30 lines, *estimated*), and `GGML_KLEIDIAI_CHUNK_MULTIPLIER=16` for lane stages. Both are
  expected bit-identical (row ranges move, the row arithmetic does not), to be shown per chunk by Tom's identity
  run at `-ub 1`, `-ub 32` and `-b 512`. Bytes 0.
- Price, width 1, cool: serial launch 207 + 13.3 becomes about 75 + 13.3 (135 µs of 160 recovered, the rest is
  the chunk tail), GPU dead 155: within 0.05 tok/s of §4.1's "launch hidden" row (8.17 / 7.55 / 7.44), so
  **about +0.4 to +0.5 tok/s over as built**; with Ebony's done flag on top it is her G.
- A side effect worth one host check: with 16 chunks any thread that is slower than the others (a capped X1, a
  preempted worker) costs the whole pool up to one chunk, 0.3 to 0.45 ms on the fused FFN matrix, in stock decode
  too. A multiplier of 16 bounds that at a quarter of it. It cannot be seen on the host; it is a knob for the next
  device A/B, on at least two subjects.

Test for Tom (host, 1 h): Ebony's T1 with one addition: the 200 µs stall on thread 0 placed after the Q4_0
barrier, at multipliers 4 and 16. Pass: CPU rows end within 60 µs of the no-stall arm at 16 and about 200 µs
late at 4. If 16 is not better than 4 on the host shim, my chunk reading is wrong and L1 needs a different hook.

Ranked next questions after this pass: (1) F's rate and slack (§8 item 2); (2) heat with the GPU on (§8 item 3);
(3) whether a recordable queue or `cl_qcom_perf_hint` shortens the flush (§8 item 5); (4) states B and D (needs
§6's cell); (5) α with pos 0 and pos 21. I continue with (2), because (1) is Ebony's and Tom's ground after her
§8 and it needs a build.

