# c3 memo-12 work, round 2 (Bonnie): Ebony's challenges resolved, R75 bands, what a perfect lane still needs

- **Date:** 2026-11-23. **Author:** bonnie, `claude-fable-5-2`. Follows [bonnie-joins.md](bonnie-joins.md) (r1,
  committed `8bbabb7`, not edited) and answers [ebony-joins.md](ebony-joins.md) §6 and §6.1. Not committed. No
  verdict here; the lead decides. **Ebony has not reviewed this file**; §2 is handed to her.
- **Labels.** *device* = printed by the phone (R74 JSON unless R72 is named). *derived* = arithmetic on *device*
  fields of the same cell. *estimated* = model output or judgement. *external* = a public source. *source* = read in
  the local upstream tree at the campaign base (`lab/llama.cpp-c3`, base `620d2225c`).
- **Script:** [bonnie_joins_r2.py](bonnie_joins_r2.py) (`python3 -I`; loads `bonnie_joins.py` and memo-10's
  `bonnie_tree.py` unchanged; self-checks R74-result §4 and the three shares of the R74 split cell).
- Triples are Yi / Llama-2 / DeepSeek. 12 tok/s = 79.17 ms. K = GPU dead time per forced join, µs.

## 0. Answer first

1. **Ebony's three challenges: accepted, with one reservation.** B+ low end 150 µs: accepted, and its centre moves
   350 → 300. A is not additive with B+: accepted (A = 270). C centre 220, band 100 to 500: accepted; I keep a
   higher probability than hers that a GPU power state is part of submit→start (0.35 against 0.2). The ladder's top
   falls from 11.45 / 11.19 / 10.82 (r1's C) to **10.87 / 10.68 / 10.35** (*estimated*, memo-10 model form).
2. **One of my own r1 arguments is withdrawn.** I wrote that the split cell argues against the launcher-clock
   explanation. The probe's feeder spins only between matrices and sleeps in `clFinish` (code read; its CPU share was
   0.55 over the block, *device*). It was not a full-clock launcher.
3. **R75's most likely outcome (*estimated*):** best `ovh_us` in x4b 215 µs (80 % band 140 to 280); in-situ K in
   x4s about 300 (200 to 380). That prices the lane at about **10.4 / 10.2 / 9.9** in the model form R74 used. P(best
   `ovh_us` ≤ 150) 0.12; P(≤ 100) 0.02.
4. **A correction to my memo-10 model (mine, unreviewed): the width-2 lane term penalises the GPU twice.** It
   divides the GPU rate by c(2) = 1.45 and then multiplies the whole lane by kB(2) = 1.12. Without the second
   penalty the zero-dead-time stack is **12.60 / 12.24 / 11.77**, not 12.23 / 11.87 / 11.44 (*estimated*). R74-result
   §4's stack column used the model as committed and carries this (+0.25 to +0.37 tok/s at every K).
5. **What decides the perfect lane is one unmeasured number: cg, what a second position costs the GPU.** The model
   assumed 1.45 by analogy with the CPU. *Device* rates say the GPU kernel runs at 0.91 of the GPU's own read-only
   rate alone and at 1.00 of it beside the CPU, so it is bus-bound in the lane and a second column may be nearly
   free: cg 1.07 to 1.72, centre 1.2 (*estimated*). Upstream already ships the kernel
   (`kernel_gemv_noshuffle_q4_0_f32_mc3`, 2 to 4 columns on the same weight image, *source*).
   - cg 1.2: stack 13.58 / 13.24 / 12.66 at K 0; **12 on all three at K ≤ 104 µs**.
   - cg 1.08: 12 on all three at K ≤ 184 µs. cg 1.45: Yi and Llama-2 at K ≤ 85 / 39, DeepSeek at no K.
   - cg 2.0 (two launches per stage): 11.26 / 10.90 / 10.56 at K 0. Nothing reaches 12.
   All with every unbuilt lever at its centre (trained head at width 2, C7, C5, O 5.2). A price, not a plan.
6. **Requests for kit r35, if it is not frozen (lead's call):** (a) a two-column cell with the `_mc3` kernel,
   `dev_us` at n_cols 2 against n_cols 1 on 11008×4096: it measures cg; (b) x4s shares up to 0.55 and 0.60: at
   K ≤ 300 the fused stage is CPU-bound at every share up to 0.50, so the sweep as briefed cannot see its optimum.

## 1. Ebony's challenges (her §6, §6.1)

### 1.1 Evidence added from the R74 lines (all *device* or *derived*)

| # | fact | reading |
|---|---|---|
| N1 | `x3 tog` gpu arm: `gpu_busy` 86 and 89 % at 1,031 and 1,008 µs per gemv → device time 887 and 897 µs; `x3 alone` printed `dev_us` 895.6 | Ebony's F5 rested on an unmeasured `dev_us`. `gpu_busy` confirms it. Gap per queued launch 111 to 144 µs, against 311 µs for the same code in `x3 alone` |
| N2 | `x3 tog` **both** arm: 21.53 GB/s = 1,124 µs per gemv at `gpu_busy` 86 to 89 % → device 967 to 1,000 µs, gap 124 to 157 µs | The queued gap stays near 140 µs with four A78s busy. Bus contention does not explain the split cell's 500 µs |
| N3 | Probe code, `split`: the feeder loops `if (q == seen) continue;`, then launch + `clFinish`. `feeder_cpu_over_wall` 0.55 over the block, about 0.40 in the split arms (two of eight arms have no GPU work and spin fully) | The feeder is on-CPU about 270 to 300 µs per matrix and asleep in `clFinish` for the rest. Policy 0 read 1,094 to 1,804 MHz. **Not a full-clock launcher.** My r1 sentence "evidence against the clock explanation" is withdrawn |
| N4 | Queued gap in `x3 alone` and sync1: 252 to 320 µs. `x3 rt`: submit→start 199 + rest 57 = 256 µs | The gap per queued launch equals submit→start plus end handling. The 196 µs of queued→submit is the part a queue hides; the rest is paid per kernel even when queued |
| N5 | Little-cluster clock: 691 to 1,094 MHz in `x3 alone`, 806 to 1,804 in the `tog` gpu arms. Feeder CPU per launch 179 → 94 to 117 µs (× 0.52 to 0.65). Non-feeder part of the gap 132 → about 20 to 30 µs | Host work scales about with the clock. Submit→start plus end handling fell by more than the clock ratio. Either that part also runs on the little cluster (kernel threads, interrupts), or the GPU stayed awake. R74 cannot tell |
| N6 | *External, search summary only:* an older Qualcomm kgsl tree sets the a6xx inter-frame power collapse main hysteresis to 0x5DC cycles = 78.1 µs. Tree and GPU model not confirmed for this phone | Every gap R74 shows (111 to 450 µs) is longer than 78 µs. A collapse and wake inside each gap is possible. This is why I do not take P(power state) down to 0.2 |

### 1.2 Resolutions

| challenge | position | band after (K µs, centre and 80 % band) |
|---|---|---|
| **B+ low end 150** (her F5) | **Accept.** N1 and N2 confirm F5 and show it holds beside a busy CPU. The cell is a queued gap, not a join; a join adds end detection (about 8 µs by event spin, 37 by `clFinish`, *estimated*). 150 is a band end, not a floor. Centre also moves, on N2 and N3 | 300 (150 to 500); was 350 (250 to 500) |
| **A not additive with B+** | **Accept.** Both act on queued→submit (196 µs). After B+ at most 40 to 110 µs of it is left to hide. Same step size as hers (−30) | 270 (130 to 500); was 290 (205 to 500) |
| **C centre 220, range 100 to 500; double count** | **Accept the band and the logic.** F5 cannot tell a hot launcher from a hot GPU, so one of my two steps was empty. The low end 35 is withdrawn. **Reservation:** P(a GPU power state is ≥ 80 µs of submit→start) 0.35, hers 0.2 (N6, weak source) | 220 (100 to 500); was 120 (35 to 500) |
| Polling saves ≤ 57 to 172 µs (§6.1) | Agree; B unchanged | 470 (400 to 560) |
| Pre-enqueue on the stock queue is what sync1 already ran (§6.2) | Agree (N4). A needs the user-event gate (x4f) or a recording (x4d) | – |
| Whole matrices per level with the weighted dead time (§6.4) | Done in r1 (D1 to D3, same κ). Unchanged: 0.3 to 1.0 tok/s below free row shares. New: a fused [gate; up] at share 0.50 **is** "gate on the GPU beside up on the CPU", so x4s at 0.50 measures D1's FFN level | – |
| Additive or weighted form (§6.5) | Weighted, as hers. κ 0.52 stands as an effective value: about 0.40 structural (the CPU runs at its alone rate while the GPU is dead) plus about 0.11 for join slack (R74 `wait_ms` 35 to 68 µs at K ≈ 500) | – |
| "Attack mine" (§6.6): the 0.52 assumes the CPU always has rows of the level | Holds in a row split by construction. It fails for a whole-matrix level with one matrix (`o`, `down`): priced in D | – |

### 1.3 Two terms neither memo had

1. **A launcher that is also a worker pays the launch in series.** With launch time L_h taken from the CPU side:
   t = lane × A + κ × K + (1 − κ) × L_h, or K_eff = K + 0.92 × L_h (*estimated*). Y13 ("launch from the pool's lead
   thread on a big core") moves 196 µs of overlapped little-core work into about 65 µs of serial big-core work:
   0.52 × 196 = 102 µs against 0.52 × 65 + 0.48 × 65 = 65 µs. **Y13 gains about 40 µs per join from the host
   part**; the rest of its price needs submit→start to fall too. Giving the launching thread fewer rows cuts the
   serial term to about a quarter.
2. **Which core runs the kernel side is unknown, and it changes the best launcher.** If submit→start follows the
   little cluster's state (N5), a spin-waiting launcher on cpus 0-3 beats one on cpu 4. Split of my belief: little
   cluster 0.5, launcher's own core 0.3, GPU or fixed 0.2. x4b's pair (0-3 spin, cpu 4 spin) decides; a spinner
   thread on 0-3 beside a cpu 4 launcher would decide it more cleanly.

### 1.4 Price table after the review (memo-10 model form, as R74-result §4; *estimated*)

| design (Ebony's name) | K µs | P | centre stack (worst to best) | r1 centre |
|---|---|---|---|---|
| E, shares re-balanced, one launch per level (Y12) | 500 (452–560) | 0.6 | 10.01 (9.59–10.31) / 9.72 (9.17–10.12) / 9.47 (8.96–9.83) | same |
| B, E + event spin | 470 (400–560) | 0.7 | 10.08 / 9.82 / 9.55 | same |
| B+, launcher off the cold A55: fast core, hot little cluster, or recording (Y13, Y14) | 300 (150–500) | 0.35 | 10.51 (9.71–11.42) / 10.37 (9.34–11.17) / 10.06 (9.11–10.79) | 10.38 / 10.20 / 9.91 |
| A, B+ + launch-ahead behind a user event | 270 (130–500) | 0.3 | 10.63 (9.71–11.52) / 10.47 (9.34–11.26) / 10.16 (9.11–10.88) | 10.55 / 10.40 / 10.09 |
| C, A + submit→start removed | 220 (100–500) | 0.2 | 10.87 (9.71–11.68) / 10.68 (9.34–11.39) / 10.35 (9.11–11.00) | 11.45 / 11.19 / 10.82 |
| F, persistent kernel (Y15) | 30 (5–100) | 0.10 | 12.02 (11.41–12.20) / 11.69 (11.16–11.85) / 11.27 (10.78–11.41) | same, P was 0.08 |
| Y16, FFN column partition (one join per FFN) | as its base design | 0.1 | saves κ × (K − 2 × queued gap) − about 15 µs per layer: +0.15 tok/s at K 300 and gap 100; nothing when the gap ≥ K ÷ 2 | not priced |
| Z, zero dead time | 0 | – | 12.23 / 11.87 / 11.44 | same |

Remaining disagreement with Ebony: none on bands. P(power state) 0.35 against 0.2; x4a decides.
With the §3.1 correction every row rises by 0.25 to 0.37 tok/s; the mapping table in §2.5 gives both forms.

## 2. R75 prediction bands (for the pre-registration; all *estimated*; centre and 80 % band)

Method: R74's `x3 rt` at 2,816 rows as the anchor (queued→submit 196, submit→start 199, args 20, end→return 37 µs);
a Monte Carlo over the launcher's speed-up, the three hypotheses of §1.3 and a ±15 % state term (`bonnie_joins_r2.py`
§2, 20,000 draws, seed 75); bands widened by hand where the model is thin. The split cells use a duty-aware model
fitted to R74's three shares (CPU slowed by 1 + 0.45 × GPU duty; GPU device time × 1.10 beside the CPU; device time
14 + 0.0801 × rows µs). Field names are those of Tom's probe source as read today in `lab/llama.cpp-c3r35`.

### 2.1 Gating events

| event | P | source |
|---|---|---|
| x4i: `cl_qcom_recordable_queues` in the extension string | 0.99 | R72 `x2info` printed it (*device*) |
| x4i: `x4 recq present=1` with `max_size` > 0 | 0.80 | MNN creates such a queue on Adreno (*external*, Ebony §3); no spec text found |
| x4i: `svm_coarse=1` / `svm_fine_buffer=1` / `svm_fine_system=1` | 0.85 / 0.30 / 0.05 | OpenCL 2.0 device; judgement |
| x4d runs: recording made and replayed, `err=0` | 0.55 | 0.8 × 0.7; the API is used without its spec |
| x4d passes Ebony's line (replay overhead ≤ 0.5 × sync1's) | 0.25 | 0.55 × 0.45 |
| x4c: identity `ok=1` on both stacked shapes | 0.90 | general kernel takes M; 50.7 MB against `max_alloc_mb` 923 (*device*) |
| x4e: flag seen by a running kernel within 1 ms, any mode | 0.35 | judgement; undefined in OpenCL |
| x4e: all three of Ebony's lines pass (persistent kernel works) | 0.10 | 0.35 × 0.6 × 0.6, rounded down for hangs |
| x4e: a child hangs or times out in at least one sub-test | 0.35 | judgement |
| x4a: `submit_us(i)` ≥ `end_us(i−1)` for most i ≥ 2 without flush (the driver serialises) | 0.60 | N4; judgement |
| x4b: best `ovh_us` ≤ 250 / ≤ 150 / ≤ 100 | 0.80 / 0.12 / 0.02 | Monte Carlo |
| x4b: `submit_to_start_us` on 0-3 spin is at least 30 µs below cpu 4 spin | 0.50 | §1.3 item 2 |
| x4f: set→START ≤ submit→start + 30 µs (the driver prepares at enqueue) | 0.40 | N4 leans against |
| x4r: held by R74 §5's three criteria | 0.35 | 4.1 GB lies between anon34 + 0.85 GB (held, slowed at the end) and anon42 (not held, R72) |
| x4r: process killed | 0.10 | judgement |

### 2.2 Cells

| cell, field | centre | 80 % band | anchor |
|---|---|---|---|
| **x4a** batch 8 × 2816×4096, launcher 0-3, one `clFinish`: `gap_p50_us` | 200 | 145 to 270 | R74 queued gaps 199 (512 rows) to 311 |
| same, launcher cpu 4 at full clock | 210 | 140 to 285 | P(below the 0-3 value by ≥ 30) 0.35 |
| same, `clFlush` after each enqueue: change of `gap_p50_us` | −10 | −60 to +30 | P(gap < 60) 0.07 |
| x4a `s2b_first_us` (first launch, 0-3) | 190 | 130 to 220 | R74 199 |
| x4a `enq_us` per launch, 0-3 / cpu 4 | 125 / 62 | 85 to 185 / 35 to 115 | R74 feeder CPU 94 to 184 |
| x4a Llama-2 layer of 7: (wall − Σ dev) ÷ 7, 0-3 / cpu 4 | 235 / 215 | 150 to 300 / 130 to 290 | R74 251.9 |
| **x4b** `ovh_us`, 0-3: finish / spin / wait-for-events | 400 / 265 / 395 | 320–500 / 190–360 / 310–500 | R74 452.0 |
| x4b `ovh_us`, cpu 4: finish / spin / wait-for-events | 310 / 255 / 305 | 225–410 / 150–345 / 220–410 | – |
| x4b `ovh_us`, cpu 7: finish / spin / wait-for-events | 295 / 250 / 290 | 215–390 / 145–335 / 210–390 | – |
| x4b best `ovh_us` over the nine cells | **215** | 140 to 280 | – |
| x4b `queued_to_start_us`: 0-3 finish / 0-3 spin / cpu 4 spin | 340 / 240 / 240 | 275–425 / 175–330 / 140–320 | R74 394.9 |
| x4b `submit_to_start_us`: 0-3 finish / 0-3 spin / cpu 4 spin | 162 / 120 / 177 | 125–200 / 75–195 / 75–220 | R74 199.0; bimodal across hypotheses |
| x4b p90 − p50: finish cells / spin cells | 150 / 90 | 60–300 / 30–250 | R74 155 |
| x4b launcher CPU per call, finish cells on 0-3 | 300 | 200 to 450 | R74 sync7 feeder 270 to 509 |
| x4b `dev_us` | 240 | 225 to 265 | R74 239.4 |
| **x4f** `set_to_start_p50_us`, 0-3 / cpu 4 | 190 / 210 | 110 to 280 | plain queued→start 240 on the same launcher |
| x4f `rtargs`: `ovh_us` at 3 args − at 15 args | −8 | −30 to +5 | inside R74's 57 µs rest |
| **x4c** gate‖up 22016×4096 cold GB/s; `gate_up_vs_11008` | 24.3; 1.14 | 21.5–27.0; 1.04–1.20 | device time 1,777 µs + one gap of 311 (125 in the `tog` state) |
| x4c q‖k‖v 12288×4096 cold GB/s; `qkv_vs_4096`; against 11008×4096 | 21.6; 1.42; 1.02 | 19.0–25.5; 1.25–1.60; 0.93–1.10 | R74 cold 15.06 (4096²), 21.01 |
| **x4d** `ovh_us_per_launch_replay` (given it runs); `replay_vs_sync1` | 138; 0.87 | 26 to 230; 0.74 to 0.97 | 60 if the driver serialises today, else submit→start + 37 |
| **x4e** flag `rtt_p50_us` (given seen) | 30 | 5 to 300 | judgement |
| x4e counter `iv_p50_us` (given seen) | 10 | 1 to 100 | judgement |
| x4e looped gemv `q4p48_vs_q4n` (given it runs) | 0.85 | 0.50 to 1.02 | judgement |
| **x4s** `t_cpu_alone_ms` (22016×4096) | 1.90 | 1.80 to 2.05 | 2 × R74's 0.948 |
| x4s feeder anchor, share 0.25: ratio; `gpu_ms`; `wait_ms` | 0.86; 1.00; 0.02 | 0.83–0.90; 0.85–1.15; 0–0.06 | CPU-bound at this share: the GPU ends 0.6 ms early |
| x4s lead mode (cpu 4, flush, spin): best ratio in 0.20 to 0.50 | **0.735** | 0.69 to 0.80 | best share 0.50 (P 0.65), 0.45 (0.25), lower (0.10) |
| x4s lead mode, share 0.20 / 0.35 ratio | 0.92 / 0.82 | 0.88–0.96 / 0.79–0.86 | CPU-bound: nearly independent of K |
| x4s lead mode, share 0.50: `gpu_ms`; `gpu_dev_ms`; `wait_ms`; `enq_us` | 1.29; 0.985; 0.04; 65 | 1.15–1.42; 0.90–1.08; 0–0.20; 35–125 | in-situ K = `gpu_ms` − `gpu_dev_ms` = 300 (200 to 380) |
| **x4r** (if held) `cpu_gbs_median`; majflt median | 18.5; 0 | 17.5 to 19.5; 0 to 5 | R74 18.72; unconditional values are bimodal, no band |

### 2.3 Per design: the value that kills it and the value that makes it worth a build

"Worth a build" = the implied K gives at least +0.5 tok/s over E on all three (in-situ K ≤ 300). "Kill" = under
+0.15 tok/s over E, or the mechanism is absent. Ebony's lines (her §7.1) are kept where she wrote one.

| design | cell | kill | worth a host or lane build |
|---|---|---|---|
| E / Y12 level stacking | x4c; x4s | identity `ok=0`, alloc error, or stacked rate < 0.9 × the same session's 11008×4096 cold rate (hers); or best fused ratio ≥ 0.95 | identity ok, rate ≥ 1.0 ×, and fused ratio ≤ 0.85 at any share |
| B event spin | x4b, same core | spin saves < 30 µs of `ovh_us` against finish | saves ≥ 80 µs |
| B+ / Y13 fast or hot launcher | x4b | best `ovh_us` > 350 | best `ovh_us` ≤ 250. Hers: > 150 means "λ stays 250 to 450"; ≤ 150 opens her lower rows |
| A launch-ahead | x4f | set→START ≥ 300 µs (hers), or less than 60 µs below the same launcher's plain queued→start | set→START ≤ submit→start + 30 µs |
| C submit→start removable | x4a | `gap_us` ≥ 150 for i ≥ 2 in every case **and** `submit_us(i)` < `end_us(i−1)` (submitted early, still late: a per-command device cost) | `gap_us` < 50 for i ≥ 2 in any case (a first-dispatch cost). If `submit_us(i)` ≥ `end_us(i−1)`: undecided, x4d decides |
| Y14 recordable replay | x4i, x4d | no queue, or replay overhead > 0.5 × sync1's (hers) | `ovh_us_per_launch_replay` ≤ 100 |
| F / Y15 persistent kernel | x4e | flag not seen within 1 ms, a hang, counter median > 50 µs, or rate < 0.8 × (hers) | all three pass and flag + counter latency ≤ 60 µs together |
| Y16 FFN column partition | x4a against x4b | queued `gap_p50_us` ≥ 0.5 × the round-trip `ovh_us` of the same launcher | gap ≤ 0.25 × |
| D whole matrices | x4s at share 0.50 | ratio at 0.50 more than 0.03 above the best ratio | ratio at 0.50 is the best: no stacked buffer needed for the FFN |
| any design at Llama-2 or DeepSeek size | x4r | not held | held |

### 2.4 From a cell value to the in-situ K

- From x4b: K = `ovh_us` of the cell the design uses + 40 µs (R74: 452 alone, 480 to 557 in the split; band +0 to
  +100). If the launcher is a worker: + 0.92 × `enq_us`.
- From x4s directly (preferred): K = `gpu_ms` − `gpu_dev_ms` at the best share. Cross-check with the ratio r at an
  interior optimum: K_eff = (r − 0.621) × `t_cpu_alone_ms` ÷ 0.521.
  **If the best ratio sits at the edge share 0.50 with `wait_ms` near 0, the cell bounds K from above only.**

### 2.5 From K to prices (*estimated*)

P_G = K0 + lane only. "Stack as R74" = `bonnie_tree.best` as committed. "Stack r2" = with the correction of §3.1 at
cg 1.45 and at cg 1.2 (K cost in ms taken from the committed model; approximate away from cg 1.45).

| in-situ K | P_G | stack as R74 | stack r2, cg 1.45 | stack r2, cg 1.2 |
|---|---|---|---|---|
| 0 | 10.34 / 9.53 / 9.22 | 12.23 / 11.87 / 11.44 | 12.60 / 12.24 / 11.77 | 13.58 / 13.24 / 12.66 |
| 30 | – | 12.02 / 11.69 / 11.27 | 12.38 / 12.06 / 11.60 | 13.33 / 13.02 / 12.46 |
| 100 | 9.64 / 8.93 / 8.69 | 11.57 / 11.30 / 10.92 | 11.90 / 11.64 / 11.22 | 12.77 / 12.53 / 12.03 |
| 150 | 9.32 / 8.66 / 8.45 | 11.27 / 11.03 / 10.67 | 11.58 / 11.36 / 10.97 | 12.41 / 12.21 / 11.73 |
| 200 | 9.03 / 8.41 / 8.22 | 10.98 / 10.78 / 10.44 | 11.28 / 11.09 / 10.72 | 12.06 / 11.90 / 11.45 |
| 300 | 8.53 / 7.98 / 7.83 | 10.51 / 10.37 / 10.06 | 10.79 / 10.65 / 10.32 | 11.50 / 11.40 / 11.00 |
| 360 | – | 10.36 / 10.17 / 9.88 | 10.61 / 10.44 / 10.13 | 11.30 / 11.15 / 10.78 |
| 452 | 8.12 / 7.50 / 7.40 | 10.13 / 9.87 / 9.60 | – | – |
| 500 | 8.01 / 7.36 / 7.27 | 10.01 / 9.72 / 9.47 | 10.26 / 9.97 / 9.70 | 10.90 / 10.62 / 10.29 |

K for 12.00: as R74, Yi ≤ 33, the others at no K. r2 at cg 1.45: 85 / 39 / none. At cg 1.2: 209 / 183 / 104.
At cg 1.08: 288 / 269 / 184.

### 2.6 Notes on the kit, for the lead and Tom (not conditions of my bands)

1. **Share sweep.** With K ≤ 300 the fused stage is CPU-bound at every share up to 0.50 (script, x4s table), so
   the best ratio lands on the edge and K is only bounded. Shares 0.55 and 0.60 would show the optimum.
2. **Two-column cell (cg).** `kernel_gemv_noshuffle_q4_0_f32_mc3` is in the same source file the probe already
   builds. One cell: 11008×4096 cold, `dev_us` at n_cols 2 against the one-column kernel. This is the number §3
   turns on. Band if it is run: cg 1.2 (1.05 to 1.7); P(cg ≤ 1.2) 0.5; P(cg ≤ 1.38) 0.7 (*estimated*).
3. A spinner on cpus 0-3 beside the cpu 4 launcher would separate the hypotheses of §1.3.

## 3. A perfect lane still misses on Llama-2 and DeepSeek: what else must move

### 3.1 First, the model: the width-2 lane term (correction to memo-10, mine)

`bonnie_tree.ms` computes `body × lane(W) × kB(W)` with lane(W) = 26 ÷ (26 + 0.64 × g ÷ c(W)).
- The GPU rate is divided by c(2) = 1.45, and the product by kB(2) = 1.12. The GPU pays for the second position
  twice. At memo-10's g = 8 this was small; at R74's g_eff = 24.8 it is not.
- The CPU's kB(2) was fitted alone (bus share 30 GB/s). Beside the GPU its bus share is 22.6 GB/s (*derived* from
  19.77 GB/s, same model), so more of the second position hides under the bus: × 1.056, not × 1.12 (*estimated*).
- Corrected body factor at width 2: 25.64 ÷ (18.73 + 21.53 ÷ cg) = **0.764** at cg 1.45, against 0.791 as
  committed. Stack at zero dead time: **12.60 / 12.24 / 11.77**.
- I have not edited `bonnie_tree.py`. Ebony: please attack this first.

### 3.2 cg, the cost of the GPU's second position (never measured; the largest term)

| evidence | value | label |
|---|---|---|
| GPU read-only rate, alone / beside the CPU | 31.14 / 25.81 GB/s | *device*, R72 §7 |
| GPU gemv device rate, 11008×4096 cold, alone / beside the CPU | 28.32 / about 25.7 GB/s | *device* / *derived* (N2) |
| so the gemv runs at | 0.91 of its read rate alone, 1.00 beside the CPU | *derived* |
| upstream's multi-column gemv: weights loaded once per block, dequantise and accumulate repeated per column | compute per byte about × 1.9 at two columns | *source* (`gemv_noshuffle_q4_0_f32.cl`, `_mc3`) |
| the CPU's two-term model applied to the GPU: compute-alone rate | 78 GB/s → cg **1.07** | *estimated* |
| worst case, the GPU exactly compute-bound at 28.3 GB/s | cg **1.72** | *estimated* |
| two launches per stage instead of one kernel | cg 2.0, and K twice | – |

| cg | stack at K 0 | K for 12.00 |
|---|---|---|
| 1.08 | 14.20 / 13.87 / 13.22 | 288 / 269 / 184 |
| 1.2 | 13.58 / 13.24 / 12.66 | 209 / 183 / 104 |
| 1.3 | 13.15 / 12.80 / 12.27 | 156 / 122 / 44 |
| 1.45 (assumed so far) | 12.60 / 12.24 / 11.77 | 85 / 39 / none |
| 1.7 | 11.89 / 11.53 / 11.13 | none |
| 2.0 | 11.26 / 10.90 / 10.56 | none |

cg needed for 12 at K 0: ≤ 1.66 / 1.53 / 1.38. At K 100: 1.42 / 1.34 / 1.21. At K 200: 1.22 / 1.17 / 1.06. At
K 300: 1.06 / 1.05 / none.

**Reading.** With cg near 1.1 to 1.2, all three reach 12 when the join is cut to about 100 to 180 µs, which is
B+'s low end and C's band, not F's. That is the first combination in this campaign whose centre reaches 12 on all
three without a persistent kernel. It still needs: the trained head at width 2 (top-1 0.379 / 0.497 / 0.438,
DeepSeek's not measured), C7, C5, O 5.2, a GPU share of about 0.5 of the body (1.6 to 1.9 GB of GPU buffers, x4r),
and the half-precision numerics inside the budget at both batch settings. My P(cg ≤ 1.2 and in-situ K ≤ 150) is
about 0.5 × 0.12 = 0.06; with the unbuilt levers at their own P the joint stays near 0.01.

### 3.3 The other levers, at a perfect lane, one at a time (model as committed; ms per token → tok/s)

| lever | change | Yi | Llama-2 | DeepSeek | value needed alone for 12 (L2 / DS) |
|---|---|---|---|---|---|
| centre | – | 77.69 → 12.23 | 80.03 → 11.87 | 83.07 → 11.44 | gap −0.86 / −3.90 ms |
| lane factor (device-rate lane, no launch gaps) | 0.621 → 0.590 | −2.64 → 12.66 | −2.82 → 12.30 | −2.74 → 11.83 | 0.612 / 0.577 |
| the same at R72's read ceiling 46.5 GB/s | → 0.551 | 13.26 | 12.91 | 12.37 | – |
| head screen C7 deeper | hs 0.5 → 0.3 | −1.81 → 12.52 | −0.88 → 12.00 | −2.78 → 11.83 | 0.30 / 0.22 |
| head on the lane (Q6_K gemv exists upstream, *source*; one join per token) | head × 0.62 | 12.50 | 12.00 | 11.81 | – |
| C5 up rule | 0.93 → 0.90 | −2.16 → 12.58 | 12.22 | 11.75 | 0.919 / 0.878 |
| O | 5.2 → 4.0 / 3.0 ms | 12.37 / 12.50 | 11.99 / 12.10 | 11.56 / 11.66 | 3.95 / none |
| head top-1 | – | – | – | – | 0.515 / 0.523 (measured 0.497; DeepSeek assumed 0.438) |
| clocks (C 40 → 50 GB/s) | – | 12.87 | 12.52 | 12.02 | 40.8 / 47.9 |
| width 3 | W 2 → 3 | 10.02 | 10.15 | 9.61 | loses; at C 50: 11.44 / 11.63 / 10.92 |
| no head at all (width 1) | – | 11.97 | 10.73 | 10.84 | Yi is within 0.03 of 12 without any guesser |

Combinations (K 0): lane 0.590 + hs 0.3 → 12.97 / 12.45 / 12.25. hs 0.3 + O 4.0 → 12.67 / 12.13 / 11.96.
lane 0.590 + hs 0.3 + O 4.0 → 13.13 / 12.58 / 12.39. Each is lower by 0.66 / 0.57 / 0.52 at K 100 and by
0.21 / 0.18 / 0.16 at K 30.

Notes:
- **Lane factor.** R74's 1.611 holds the `tog` state's launch gap (about 140 µs per 1 ms kernel, N1). The lane
  without gaps is 0.57 (from `tog` and `gpu_busy`) to 0.63 (from the split cell's CPU times): centre 0.60, not
  better known. x4s prints `gpu_dev_ms` and `cpu_ms` per share and will narrow it. K in this file is all GPU dead
  time, so a design with K → 0 gets the gap-free lane too.
- **GPU work other than gemv rows.**
  - The head: one stage, one join per token, so its lane costs κ × K once (0.26 ms at K 500). Worth 1.7 / 0.8 /
    2.6 ms after C7. Cheap in joins, needs the Q6_K kernel's numerics inside the same budget.
  - Attention, norms, activations (inside O): each is tens of µs of CPU work. A stage pays only if A > K (§1.2), so
    they stay on the CPU at any K above about 30 µs. With a persistent kernel they could move; not priced.
- **Nothing in this table is new except the lane factor, the head on the lane and the width-1 reading for Yi.**

## 4. Lists for Tom (through the lead)

**Test list**
1. Host: recompute the stack with the lane term of §3.1 after Ebony's review (a two-line change in a copy of
   `bonnie_tree.py`); add a dated note to R74-result §4 if she agrees.
2. Host: read `_mc3`'s launch code (local size, image layout of the two columns) and confirm it runs on the probe's
   existing weight image without a second copy.
3. Device, R75 if still open: the two-column cell (§2.6 item 2); x4s shares 0.55 and 0.60.
4. Host numerics (extends R74 §7 item 1): the half-precision emulation at width 2 gives the same per-column values
   as at width 1 (the kernel comment claims per-column identity).

**Try list (after R75)**
1. If x4b's best `ovh_us` ≤ 250: lane with the best launcher and wait mode, level-stacked buffers, static shares.
2. If cg ≤ 1.3: the lane's GPU side on `_mc3` for the width-2 pass. This comes before any work on F.
3. Head on the lane with upstream's Q6_K gemv, one join per token.

## 5. Next questions (ranked), and what I did on each

1. **cg.** Done on the host side: §3.2 (evidence from R72 and R74 rates, upstream kernel found and read). Open: the
   device number.
2. **Is the width-2 correction right?** Done: §3.1; needs Ebony.
3. **Gap-free lane factor.** Bounded 0.57 to 0.63 (§3.3); x4s decides.
4. **Which core runs the kernel side** (§1.3 item 2). Predicted in §2.1; x4b decides.
5. **Head on the lane.** Priced (§3.3). Open: Q6_K kernel numerics; whether `q6_K_f32_tiled_mc3` covers two columns.
6. **Yi at width 1.** 11.97 at a perfect lane with C7, C5, O 5.2 and no trained head. Open: the same at finite K
   (not priced; the width-1 lane has cg = 1 by definition, so its K line may be easier than width 2's on Yi).
7. **O.** Still an allowance (8 ms, unmeasured; R73 held). Each ms of O is 0.11 to 0.15 tok/s near 12.
8. **Residency at 1.6 to 1.9 GB of GPU buffers** for Llama-2 and DeepSeek: x4r covers 1.5 GB.

## 6. Limits

- Every band in §2 rests on one R74 session without the cool start, and on a three-way hypothesis split that is my
  judgement. The Monte Carlo gives coherent bands, not calibrated ones.
- §3.1 and §3.2 change a model I wrote; no one has reviewed them. cg is unmeasured. The K cost at cg other than
  1.45 is carried over from the committed model.
- The duty-aware split model is fitted to three shares of one shape and reads 0.03 to 0.05 ms low near balance
  (join slack is not in it).
- The stack still needs every unbuilt lever at its centre. It is a price, not a plan (prereg R74 §5).

## 7. Sources (read 2026-11-23)

Repository: R74 JSON (raw `x3` lines), R72 JSON (`x2info`, `x2 kern=read`), [R74-result.md](../R74-result.md),
[R72-result.md](../R72-result.md) §7, `patches/llama.cpp-c3/0005-x3-gpu-probe.patch` (`batch`, `roundtrip`, `split`),
`lab/llama.cpp-c3r35/src/llama-vollie-probe.cpp` (x4 line formats, read only), `memo-10-work/bonnie_tree.py`,
[memo-09-gap.md](../memo-09-gap.md) §4, [ebony-joins.md](ebony-joins.md).

Upstream at the base (*source*, local): `ggml/src/ggml-opencl/kernels/gemv_noshuffle_q4_0_f32.cl`
(`kernel_gemv_noshuffle_q4_0_f32_mc3`), `gemm_noshuffle_q4_0_f32.cl` (half-precision accumulators, tuned for 128
columns; not proposed), `gemv_noshuffle_q6_k_f32.cl`; `ggml-opencl.cpp` lines 1225 to 1278 (the `_mc3` family).

External:
- kgsl inter-frame power collapse hysteresis, 78.1 µs in an older tree (**search summary only**; the page returned
  HTTP 429 when fetched):
  https://gitlab.e.foundation/e/devices/android_kernel_fairphone_FP3/-/commit/4edfb6b4b99533f3fb92fd162f91bbf4be587d54
- Upstream IFPC patches for drm/msm, 2025 (search summary): https://lists.freedesktop.org/archives/freedreno/2025-July/039839.html
- llama.cpp release notes naming the Q6_K Adreno kernels and "decode GEMV and medium-batch GEMM optimizations
  (speculative decoding/MTP)" (search summary): https://newreleases.io/project/github/ggml-org/llama.cpp/release/b8493 ,
  https://newreleases.io/project/github/ggml-org/llama.cpp/release/b10791
- Kim, Oh, Yi, "Minimizing GPU Kernel Launch Overhead in Deep Learning Inference on Mobile GPUs", HotMobile 2021
  (abstract only; no per-launch figure found): https://pure.uos.ac.kr/en/publications/minimizing-gpu-kernel-launch-overhead-in-deep-learning-inference-/
- Khronos registry, qcom extensions (no recordable-queue text): https://registry.khronos.org/OpenCL/extensions/qcom/

Searched and not found: the `cl_qcom_recordable_queues` specification; a measured Adreno submit→start latency; a
measured cost of a two-column gemv on an Adreno 6xx.
