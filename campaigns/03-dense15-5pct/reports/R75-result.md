# R75 result (partial): x4 GPU join probe, kit r35, hung in block x4f

- **Date:** 2026-11-24. Run started 10:35:52 local; the app hung in x4f and the owner force-stopped it and saved the
  report at about 13:14 (owner's account). **Status: partial.** x4i, x4a and x4b are complete and scored below;
  x4f is void; x4c, x4d, x4e, x4m, x4s and x4r never ran. The unmeasured blocks are re-run under
  [R75-amendment-1](../prereg/R75-amendment-1.md); its cells are scored in a later section of this file.
- **Labels:** every value the phone printed is *device*. Differences, ratios, medians over cells and the Brier score
  are arithmetic on *device* values. Prices and K values are *estimated*.
- **Sources:**
  - [reports/device/vollie-c3probe-r35-2026-11-24T1035.json](../../../reports/device/vollie-c3probe-r35-2026-11-24T1035.json)
    (committed in `d51039f`), sha256 `7560b6ef6ca813cd7c958e6627d32561eb684a84f3901325b3e8959fc1166885`. JSON fields:
    `kit=r35`, `app=1.35.0+35`, `running_block=x4f`, `finished=null`, `power_end=null`, three blocks saved.
    The prereg (§8 step 5) asks for the file under `campaigns/03-dense15-5pct/reports/device/`; that copy is Annie's.
  - Pre-registration [R75](../prereg/R75-x4-gpu-join-probe-device.md) (commit `2588f40`, tag `prereg-R75`).
  - Parser `tools/c3_probe_parse.py` (x4report) as committed; the per-event `x4 tl` reads (V12, design C) and the
    cell arithmetic re-done by the lead with short `python3 -I` code, not committed.
- **Scored by:** clyde (lead), `claude-opus-5-5`.

## 1. Void checks (§5.5)

| void | state | source |
|---|---|---|
| APK hash differs | not triggered as far as the file shows (`kit=r35`, `app=1.35.0+35`). *Owner: please confirm the sha256 you checked against `r35/SHA256SUMS`.* | JSON header |
| power saving on | not triggered | `power_start.powerSave=false`; `power_before.powerSave=false` in x4i, x4a, x4b |
| charger plugged | not triggered | `plugged_start=false`; `plugged_before=false` in x4i, x4a, x4b |
| process killed during a block other than x4r | **triggered for x4f.** The owner force-stopped the app while `running_block=x4f`. By §5.5 "that block only is void (the earlier blocks are saved)" | `running_block=x4f`, no x4f block in the file |

x4c, x4d, x4e, x4m, x4s and x4r did not start. They are neither void nor killed: they are **not measured (x4f
hang)** under §4's "a row whose cell did not run ... is not measured, not a miss". In particular x4r is **not** the
"killed in x4r" outcome of §5.4: it never began.

## 2. The hang, against the prereg's own lines

- **What happened (*device*, from the file):** x4b ended (`x4b end s=20.2`); the next block x4f began and never
  ended. The r16 gate before x4f is capped at 180 s, so a 2.5 h stop is inside x4f itself, not its gate. No x4f
  line was saved (lines are saved at the end of a block).
- **Probable cause (lead's reading of `0007-x4-join-probe.patch`, function `x4f`; not a measurement):** in the gate
  iteration the kernel is enqueued behind a user event and flushed; then `clSetUserEventStatus` releases it and
  `spin_ev()` polls the kernel's event with no further `clFlush` and no deadline. The first gated iteration is the
  second of the unprinted warm-up pairs (i = −9). If the Adreno driver does not submit a command released after its
  flush, the poll never ends. x4f runs in the parent process, so no child timeout applies. Tom checks this in kit
  r36.
- **Did any prereg line predict it?** No.
  - V11 ("a child hangs or times out", P 0.35) is about the child blocks x4d and x4e (`x4 child rc=-1001`). Neither
    ran. The one child that ran, `x4k_c` in x4i, ended `rc=0 timeout=0` in 0.12 s. **V11: not measured.**
  - §5.1's child-timeout rules and §5.5's last line cover children only.
  - §2's memory note expects a possible kill in x4r, the last block. It says nothing of x4f.
  - No line priced a hang in a parent block, and no line gave the owner a time after which to stop the app. That is
    a gap in the pre-registration. It is reported here and closed by a pre-written rule in amendment 1 (§5 there).
- **Wall time.** Prereg §2: about 3.6 to 7 min of blocks plus up to ten gates of 180 s, so 4 to 37 min
  (*estimated*). Measured part (*device*): gates 85.1 + 0.1 + 0.0 s and blocks 0.96 + 1.16 + 20.20 s, so 107.6 s
  in total; x4f began at about 10:37:40. The run then stood in x4f until about 13:14. That is about 2 h 36 min of
  hang and 2 h 38 min in all, 4.3 times the prereg's upper bound (arithmetic on the owner's stop time).

## 3. Conditions (not voids; reported beside the numbers)

- **Cool start not shown.** The x4i gate began at headroom 0.873 and took 85.1 s to drop below 0.65 (18 samples).
  The x4a and x4b gates passed at once (0.643, 0.644). Battery 29.0 °C at start, 28.7 °C before each block. The
  prereg asks for 30 min of rest with no other probe before it. The file cannot show the rest. *Owner: how long did
  the phone rest before 10:35?*
- **Thermal lines (*device*):** `x4 therm` status 0, `thermal_pwrlevel=0` and `gpu_max_mhz=840` before and after
  x4i, x4a and x4b. `gpu_mhz` was unreadable (−1). `x3 cond` read `devfreq_cur` 315 MHz before x4i and 840 MHz
  before x4a and x4b.
- **CPU clocks:** `ramp_ok=1` on every timed cell except the three launcher-7 cells of x4b (`ramp_ok=0`, `ramp_s`
  2.50).
- **Launcher 7 ran on cpu 4.** All three cpu-7 cells read `lplace=n/n@4`: every sample of the launcher thread was
  on cpu 4, not cpu 7. cpu 7 stayed at 844 MHz in `mhz_range`. So B3 is a second cpu-4 measurement, not a cpu-7
  one. Scored as built (§5.5: "the rules use the cells as built"); Tom should check the pinning.
- **Drift inside x4b.** The 0-3 `finish` anchor read `ovh_us` 350.4 at the start (rep 0) and 451.5 at the end
  (rep 1): +101.1 µs, +28.9 % (arithmetic). Rep 1 equals R74's 452 at the same cell.
- **Memory at start (*device*, `hdr mem`):** MemAvailable 2.81 GB; swap 4.19 GB total, 0.99 GB free (about 3.2 GB
  swapped out before the run). It does not touch x4i to x4b. It matters for x4r on the re-run.
- Deep-sleeping list **on**. `model_unloaded=olmoe-Mi.gguf`. Battery level 80 %.
- PSI was unreadable (errno 13), as in R74.

## 4. Events (§4.1), as written

| # | outcome (*device*) | P (merged) | scored |
|---|---|---|---|
| V1 | `cl_qcom_recordable_queues` is in the extension string | 0.99 | **yes** |
| V2 | `x4 recq present=1 max_size=512` | 0.80 | **yes** |
| V3 | `svm_coarse=1` / `svm_fine_buffer=1` / `svm_fine_system=0` | 0.90 / 0.60 to 0.70 / 0.05 | **yes / yes / no** |
| V4 | `svm_atomics=1` | 0.65 | **yes** |
| V5 | x4d did not run | 0.55 / 0.20 to 0.25 | not measured (x4f hang) |
| V6 | x4c did not run | 0.90 | not measured (x4f hang) |
| V7 to V10 | x4e did not run | – | not measured (x4f hang) |
| V11 | x4d and x4e did not run; the one child that ran (`x4k_c`) had `rc=0 timeout=0` | 0.35 | not measured (x4f hang); see §2 |
| V12 | the driver does **not** serialise: `submit_us(i) ≥ end_us(i−1)` for 0 of 35 events (i ≥ 1) of `batch8` at launcher 0-3 and 0 of 35 at cpu 4. Within a rep all events share one `submit_us`: the batch is submitted at once | 0.50 to 0.60 | **no** |
| V13 | best `ovh_us` of the nine rep-0 cells = **188.6** µs (0-3 `spin`): ≤ 250 yes, ≤ 150 no, ≤ 100 no | 0.75 to 0.80 / 0.12 to 0.20 / 0.02 to 0.03 | **yes / no / no** |
| V14 | 0-3 `spin` `submit_to_start_us` 18.6 vs cpu 4 `spin` 66.5: 47.9 µs below (≥ 30) | 0.45 to 0.50 | **yes** |
| V15 | x4f void | 0.30 to 0.40 | not measured (x4f void) |
| V16, V17 | x4m, x4s did not run | – | not measured (x4f hang) |
| V18 | x4r did not run (not "killed": it never started) | – | not measured (x4f hang) |
| V19 to V21, V23, V24 | x4s, x4m did not run | – | not measured (x4f hang) |
| V22 | `gpu_max_mhz` 840 on every printed line; `thermal_pwrlevel=0` on every `x3 cond`. Only three of ten blocks printed | 0.15 / 0.10 (B) | **no / no**, on the lines that exist (partial) |

**Brier score** (arithmetic; §4.1 asks for one; the prereg does not say how a range is scored, so both ends are
given): over the 13 scored outcomes above (V3 as three, V13 as three, V22 as two), 0.066 at the most favourable end
of each range and 0.087 at the least favourable. Without V22 (partial), over 11 outcomes: 0.075 to 0.100. The
largest single terms are V12 (0.25 to 0.36), V14 (0.25 to 0.30), V3 fine buffer (0.09 to 0.16) and V4 (0.12).

## 5. Cells (§4.2), as written

Scored on the merged band ("union band"). Values *device*; differences arithmetic.

| # | value | band | scored |
|---|---|---|---|
| A1 | `gap_p50_us` `batch8` 0-3 = **1.0** | 110–270 | **miss, low** (about 1/200 of the centre 200) |
| A2 | `gap_p50_us` `batch8` cpu 4 = **1.0** | 100–285 | **miss, low** |
| A3 | `batch8flush` − `batch8` `gap_p50_us`: 0-3 4.1 − 1.0 = +3.1; cpu 4 4.1 − 1.0 = +3.1 | −60 to +30 | **inside** (both) |
| A4 | `s2b_first_us` 0-3 = 186.9 | 130–220 | **inside** |
| A5 | `enq_us` per launch, `batch8`: 0-3 **38.4**; cpu 4 **12.3** | 85–185 / 35–115 | **miss, low / miss, low**. The row names no case; `batch8` is scored as A1's case. Beside it (not scored): `layer7` 71.3 / 38.2, `batch8flush` 210.5 / 91.9 |
| A6 | `layer7` (`wall_us` − `dev_sum_us`) ÷ 7: 0-3 (6080.8 − 4125.0) ÷ 7 = **279.4**; cpu 4 (4961.3 − 4162.1) ÷ 7 = **114.2** | 150–300 / 130–290 | **inside / miss, low** |
| A7 | `layer7` − `batch8` `gap_p50_us`: 0-3 1.0 − 1.0 = 0.0; cpu 4 0.0 | 0 to +120 | **inside** (at the lower edge; both launchers) |
| B1 | 0-3 `ovh_us`: finish **350.4** / spin **188.6** / wfe **461.4** | 320–500 / 190–360 / 310–500 | **inside / miss, low (1.4 µs under) / inside** |
| B2 | cpu 4: finish **962.5** / spin **401.4** / wfe **960.3** | 225–410 / 130–345 / 220–410 | **miss, high / miss, high / miss, high** |
| B3 | "cpu 7" (ran on cpu 4, §3): finish **963.3** / spin **314.6** / wfe **666.7** | 215–390 / 130–345 / 210–390 | **miss, high / inside / miss, high** |
| B4 | **best `ovh_us` of nine = 188.6** (0-3 spin) | 120–290 | **inside** |
| B5 | `queued_to_start_us`: 0-3 finish **109.2** / 0-3 spin **93.1** / cpu 4 spin **96.4** | 275–425 / 175–330 / 140–320 | **miss, low** (all three) |
| B6 | `submit_to_start_us`: 0-3 finish **19.7** / 0-3 spin **18.6** / cpu 4 spin **66.5** | 125–200 / 75–195 / 75–220 | **miss, low** (all three) |
| B7 | `p90_us` − `p50_us`, median over launchers 0-3 / 4 / 7: finish 697.5 / 21.9 / 20.8 → **21.9**; spin 13.0 / 665.2 / 663.3 → **663.3** | 60–300 / 30–250 | **miss, low / miss, high** |
| B8 | `lcpu_us` 0-3 finish **185.1**; `dev_us` **232.8** (all ten cells 232.6 to 235.2) | 200–450; 225–265 | **miss, low; inside** |
| F1, F2 | x4f void | – | not measured (x4f void) |
| C1, C2 | x4c | – | not measured (x4f hang) |
| D1 | x4d | – | not measured (x4f hang) |
| E1 to E3 | x4e | – | not measured (x4f hang) |
| M1, M2, T1 to T11 | x4m | – | not measured (x4f hang) |
| S1 to S10, H1 | x4s | – | not measured (x4f hang) |
| R1 | x4r | – | not measured (x4f hang) |

**Tally** (each scored value counted once; arithmetic): 31 values, **11 inside, 20 misses (14 low, 6 high)**.
- Inside: A3 ×2, A4, A6 (0-3), A7 ×2, B1 finish, B1 wfe, B3 spin, B4, B8 `dev_us`.
- Low: A1, A2, A5 ×2, A6 (cpu 4), B1 spin, B5 ×3, B6 ×3, B7 finish, B8 `lcpu_us`.
- High: B2 ×3, B3 finish, B3 wfe, B7 spin.

**What the misses say, in one line each (reading, not a rule):**
- Once commands are queued, the GPU runs them back to back: 1 µs between kernels (A1, A2, V12). The pricing
  assumed a per-launch gap of about 200 µs. It is not there.
- The host side is cheaper than priced (A5, B5, B6 low): enqueue 12 to 38 µs, submit-to-start 19 µs at 0-3.
- The big-core launcher is worse, not better, when it sleeps: on cpu 4, `finish` and `wfe` both cost about 960 µs
  with `p50` 1,190 µs and a tight `p90`. Looks like a wake-up of about 1 ms on the big core (B2, B3; reading only).
  `spin` on cpu 4 is bimodal (p50 427, p90 1,093).
- The best join is 0-3 `spin` at 188.6 µs. The prediction for it was 190–360; it is 1.4 µs below the band.

## 6. Decision rules (§5.1), on the inputs that exist

"Worth a build" in §5.1 has two parts. The table criterion, and the definition above it: the design's implied K
gives at least +0.5 tok/s over the measured in-situ K on all three subjects. The in-situ K (K_meas, §5.3) comes
from x4s, which did not run. So no design can be called "worth a build" yet. Where the table criterion is met it is
written **"criterion met, K test pending (x4s)"**.

| design | inputs (*device*) | verdict |
|---|---|---|
| **B** event spin | finish − spin, same launcher: 0-3 350.4 − 188.6 = **161.8**; cpu 4 962.5 − 401.4 = **561.1**; "7" (cpu 4) 963.3 − 314.6 = **648.7** µs | not killed (≥ 30 on every launcher). Criterion (≥ 80 on any launcher) **met, K test pending (x4s)**. Against the rep-1 anchor the 0-3 saving is 262.9 µs (drift, §3) |
| **B+ / Y13** | best `ovh_us` 188.6 | not killed (≤ 350). Criterion (≤ 250) **met, K test pending**. Ebony's note: > 150, so "λ stays in 250 to 450, reported". Her lower rows (≤ 150) are not opened |
| **A** launch-ahead | needs x4f | **not measured (x4f void)** |
| **C** submit→start removable | per-event median `gap_us` over i ≥ 2: `batch8` 1.0 / 1.0, `layer7` 1.0 / 1.0, `batch8flush` 3.8 / 4.1 µs (0-3 / cpu 4); `submit_us(i) < end_us(i−1)` for 34 of 35 and 35 of 35 events | not killed. Criterion (median < 50 in any case) **met**. §5.3 names no design K for C, so C has no price of its own. As a fact: the per-command device cost the kill line looked for is about 1 µs |
| **Y14** recordable replay | `x4 recq present=1 max_size=512 err=0`; x4d did not run | the x4i branch of the kill is not hit; the rest is **not measured (x4f hang)** |
| **E / Y12**, **F / Y15**, **D**, **cg**, hand-off, residency | x4c, x4e, x4s, x4m, x4r | **not measured (x4f hang)** |
| **Y16** FFN column partition | `batch8` cpu 4 `gap_p50_us` 1.0 vs 0.25 × cpu 4 `finish` `ovh_us` 962.5 = 240.6 | not killed. Criterion (gap ≤ 0.25 × `ovh_us`) **met, K test pending**. The rule's input is a cpu-4 `finish` cell that is 2.7 times the 0-3 one (B2) |

**§5.2 (together model), §5.3 (K_meas, H, K_eff, cg, prices) and the priceable rule:** not measured. They need
x4m and x4s. The priceable rule is **not scored**.

Beside it, descriptive and *estimated*: the cross-check K_x4b = best `ovh_us` + 40 = **228.6 µs**, before any
hand-off H. At cg 1.25 the DeepSeek line is 168 µs (`linear`) and 138 µs (`pair`, Ebony slack). K_x4b is above
both, even at H = 0. Only at cg ≤ 1.0 in `linear` (253) would it fall under. This is not the rule's K. The rule
reads K_meas from x4s.

## 7. The memo-12 kill lines X4-a and X4-b (Ebony, `ebony-joins.md` §7.1; inputs printed by the parser)

- **X4-a** ("gaps END(i)→START(i+1) ≥ 150 µs with the launcher on a big core at full clock → the cost is inside
  the driver or GPU per launch; pre-enqueue and polling are closed"): **not triggered.** `gap_p50_us` at cpu 4 is
  1.0 / 1.0 / 4.1 µs (`batch8` / `layer7` / `batch8flush`, *device*). The per-launch device-side cost is about 1 µs.
  Pre-enqueue (A) and polling (B) stay open. The cost of a join is the host round trip: enqueue, submit, start of
  the first kernel, and the wait.
- **X4-b** ("best `ovh_us` > 150 µs → λ stays in 250 to 450 µs and the lower rows are out of reach by host-side
  means"): **triggered.** Best `ovh_us` is 188.6 µs (0-3 spin, *device*). As written, the lower rows (λ ≤ 150) are
  out of reach by host-side means. The line's own range (λ 250 to 450) is wider than the data: K_x4b = 228.6 µs
  (*estimated*) is below 250. The line is read as written; the gap between its range and K_x4b is noted, not used.
- **Still open:** A (launch-ahead, x4f), Y14 replay (x4d), F persistent kernel (x4e), E stacking (x4c), D, cg,
  the contention model and the in-situ K with the hand-off (x4m, x4s), and residency at 1.5 GB of GPU buffers (x4r).
  K_meas decides every "worth a build" above.

## 8. Next

- Re-run of x4f, x4c, x4d, x4e, x4m, x4s and x4r on kit r36 under
  [R75-amendment-1](../prereg/R75-amendment-1.md). Predictions and rules are unchanged; the amendment adds a
  harness deadline and a parent-hang rule.
- Tom: the x4f flush and deadline fix; the cpu-7 pinning (`lplace=@4`).
- Owner: the APK sha256 checked for r35, and the rest time before 10:35.
