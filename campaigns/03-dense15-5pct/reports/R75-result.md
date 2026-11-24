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

## Owner confirmations 2026-11-24

*Dated note, appended by Annie from the owner's words relayed by the lead. No score above is changed.*

1. APK checksum. Owner: "SHA 맞습니다." The r35 APK sha256 the owner checked before R75 is
   `465d97bb7932fc4c9b7c675e1b3e2b78536466f9dec3442d4b8c59405f558af5` (as prereg section 8). This closes the open item
   "the APK sha256 checked for r35" in section 8 (Next).
2. Rest time. Owner: "실온에서 30분 쉬게 했습니다." The phone rested 30 min at room temperature before the 10:35 start,
   so the prereg's 30-min cool was kept. This closes the open item "the rest time before 10:35" in section 8. The
   first gate's 85 s wait at headroom 0.873 happened despite this rest; the note records that and explains nothing.


## Session 2 (kit r36, amendment 1)

*Appended 2026-11-24 by clyde (lead, `claude-opus-5-5`). Sections 1 to 8 above (session 1) are not changed; where
session 2 adds to them, a dated note in this section says so.*

- **Run:** 2026-11-24 13:45:29 to 13:59:17 local (13 min 48 s), `finished` set, `running_block=null`,
  `stopped_early=false`, no `x4 hang` line. Blocks 191.6 s, gates 635.9 s (*device*, sums of the JSON's `ms` and
  gate series). The amendment's estimate was 3.5 to 31 min (*estimated*): inside.
- **Sources:**
  - [reports/device/r75a1-c3probe-r36-2026-11-24T1345.json](device/r75a1-c3probe-r36-2026-11-24T1345.json)
    (committed in `b974b5c`), sha256 `e46e3446d31dc40f0f3f8c19d91c389728c1f00c226bff2cb92961a41555d513`. JSON
    fields: `kit=r36`, `app=1.36.0+36`, eight blocks x4i, x4f, x4c, x4d, x4e, x4m, x4s, x4r.
  - [R75-amendment-1](../prereg/R75-amendment-1.md) (commit `a49db43`) on top of the pre-registration (commit
    `2588f40`, tag `prereg-R75`).
  - Parser `tools/c3_probe_parse.py` as committed (x4report with the r36 fields). Two-arm means, per-pass x4r medians
    and the prices were re-done by the lead with short `python3 -I` code reading the JSON and loading the committed
    memo-12 scripts unchanged; that code is not committed.
- **Labels:** every value the phone printed is *device*. Differences, ratios, means of arms and medians are
  arithmetic on *device* values and are written without a label when the *device* inputs sit beside them. K_meas,
  K_eff, design K, every tok/s and every stack are *estimated*.

### S2.1 Void checks (§5.5, APK hash against `r36/SHA256SUMS`)

| void | state | source |
|---|---|---|
| APK hash differs | not triggered as far as the file shows (`kit=r36`, `app=1.36.0+36`). The owner's sha256 check of the r36 APK is not in the file. *Owner: please confirm you checked `5d852dea…6bda`.* | JSON header |
| power saving on | not triggered | `powerSave=false` at start, before every block, at end |
| charger plugged | not triggered | `plugged_start=false`, `plugged_before=false` in all eight blocks |
| process killed during a block | not triggered | `finished` set, all eight blocks saved |

No block is void.

### S2.2 Conditions (not voids)

- **Rest before the run: assumed** to be 30 min as in session 1 (the owner's account relayed by the lead). The
  JSON cannot show it and does not contradict it: battery 26.4 °C at start (*device*). The first gate began at
  headroom 0.783 and took 100.1 s to drop below 0.65 (session 1: 0.873 and 85.1 s).
- **Gates:** x4f and x4e passed at once; x4c 45.1 s, x4d 85.1 s, x4m 45.1 s. **x4s and x4r gates timed out** at
  180 s (headroom 1.148 → 0.735 and 1.115 → 0.705): x4m's 61 s of load heated the phone and x4s started hot.
- **Thermal lines** (`x4 therm`, *device*): status 0, `thermal_pwrlevel` 0, `gpu_max_mhz` 840 before and after
  x4i, x4f, x4c, x4d, x4e and before x4m, x4s, x4r. **After x4m: status 2, `thermal_pwrlevel` 3,
  `gpu_max_mhz` 676. After x4s: status 3, `thermal_pwrlevel` 6, `gpu_max_mhz` 491.** After x4r: back to 0 / 0 / 840.
  `x3 cond` read `thermal_pwrlevel=0` before every block.
- **Which cells ran under a GPU cap below 840 MHz** (`gpu_max_mhz` on the tails, *device*): in x4m, from the last
  CPU arm of the two-column together cell (pos 5) on, so the whole one-column cell and the mixed arm (676); in x4s,
  arms pos 11 to 19 (676), pos 20 and 21 and the whole hand-off cell (491). The CPU caps fell at the same time in
  x4s (policy 4 and 7 maximum 2,150 MHz from pos 6, 1,785 from pos 11, 1,420 from pos 15 on). Every other timed cell: 840, `ramp_ok=1`.
- **Battery level 23 % at start, 6 % at end** (*device*, `power_start.level`, `power_end.level`; 15 % before x4s,
  7 % before x4r). Power saving stayed off. Whether the low level contributed to the caps is not shown by the file.
- **Memory** (*device*, `hdr mem` before x4i): MemAvailable 2.59 GB, swap 4.19 GB total with 1.23 GB free.
- Deep-sleeping list **on**. `model_unloaded=olmoe-Mi.gguf`. PSI unreadable (errno 13), as before.
- **Deadlines** (`x4 dl`, *device*): 0 in every block. The no-flush check's own 2 s waits are not deadlines and are
  not counted (`tom-r36-build.md`).
- **`x4 pin`** (x4i, idle, *device*): `cpu4=ok@4 cpu5=e22 cpu6=ok@6 cpu7=e22`. *Note 2026-11-24 to session 1's §3
  ("launcher 7 ran on cpu 4"): the pin to cpu 7 is refused with EINVAL at idle (cpu 5 as well), which fits Tom's
  inferred reading that `sched_setaffinity({7})` failed unseen; session 1's B3 stays scored as built.*
- **x4i against session 1** (descriptive, amendment §3): every x4i line is identical (extensions, `x4 dev`,
  `x4 recq`, `x4 entry`, `x4 kai`, the build and load lines, identity at 2,816 rows `max_rel_err` 6.723e-04, `ok=1`)
  except the install path inside `x4 info` and r36's new `x4 pin` and `x4 dl` lines. x4i ran clean (amendment §5
  item 4).
- **Session drift flag (amendment §3): SET.** The named pair is x4f's cpu-4 spin cells against session 1's x4b
  `lcpu=4 wait=spin` cell. `ovh_us`: session 2 `rtargs` 140.3 / 135.5 / 132.6 / 130.6 (mean 134.75) against
  session 1's 401.4: −66 %. `queued_to_start_us` 24.0 to 24.3 against 96.4: −75 %. The plain arm's
  `set_to_start_p50_us` 147.0 is also 63 % below 401.4. (Against the 0-3 spin anchor 188.6 the `rtargs` mean is
  −29 %.) Every rule below that mixes the two sessions carries the flag; the rules are applied as written.
  Session 2's cpu-4 spin is tight (`p90` 368.8 to 382.6 against `p50` 359.5 to 364.7); session 1's was bimodal.

### S2.3 Events (§4.1 and amendment §5 item 5), as written

| # | outcome (*device*) | P (merged) | scored |
|---|---|---|---|
| V5 | x4d ran: `x4 rec setup ... ok=1`; replay `ovh_us_per_launch_replay` 44.3 ≤ 0.5 × sync1's 124.5 = 62.3 | 0.55 / 0.20 to 0.25 | **yes / yes** |
| V6 | `x3 identity` m 22016 `ok=1` (7.713e-04), m 12288 `ok=1` (7.451e-04) | 0.90 | **yes** |
| V7 | `hostptr` and `svm` flag lines `seen=0 ok=0` (kernel timed out, `kernel_exit=-2`) | 0.35 to 0.40 | **no** |
| V8 | `svm_c11` flag `ok=1`, `rtt_p50_us` 0.9 | 0.55 | **yes** |
| V9 | `x4 pk c11 test=flag err=0`; `svm_c11` flag line `ok=1`, seen 200 of 200 | 0.55 / 0.75 | **yes / yes** |
| V10 | all three F lines pass in `svm_c11` (§5.1 F "worth": 0.9 + 3.75 = 4.65 µs ≤ 60; (iv) 0.985 / 0.986, `same_rows=1`) | 0.10 to 0.15 | **yes** |
| V11 | children `x4d_c`, `x4e1_c`, `x4e2_c`, `x4e3_c` all `rc=0 timeout=0` (session 1's `x4k_c` too) | 0.35 | **no** |
| V15 | gate `set_to_start_p50_us` 381.7 against plain `submit_to_start_p50_us` 6.9 + 30 = 36.9 | 0.30 to 0.40 | **no** |
| V16 | `_mc3` created / runs (`err=0`) / identity `ok=1` on every column at n 2, 3, 4 / all | 0.92 / 0.85 / 0.88 to 0.90 / 0.75 | **yes / yes / yes / yes** |
| V17 | lead `ratio` at the best share 0.55 is 0.923; 0.963 at 0.50 and 0.987 at 0.60 | 0.85 to 0.90 | **yes** |
| V18 | x4r held (S2.6) / killed: `x4 res done err=0`, `x4r end s=21.2` | 0.25 to 0.35 / 0.10 to 0.15 | **yes / no** |
| V19 | `ahp_persist match=1`; best `extra_us` 13.3 (`write`, `match=1`) ≤ 20 | 0.55 to 0.60 / 0.50 | **yes / yes** |
| V20 | `x4 tog2sum cpu_alone_gbs` 18.33 < 19 | 0.85 to 0.90 | **yes** |
| V21 | tile reading from `x4 tog2sum` is `pair` (S2.5) | 0.50 / 0.25 to 0.30 / 0.20 to 0.25 | **pair yes / undecided no / carried-over no** |
| V22 | `gpu_max_mhz` 676 and 491 on x4m and x4s tails; `x3 cond thermal_pwrlevel` 0 on all eight (the `x4 therm` lines read 3 and 6 after x4m and x4s, a different field) | 0.15 / 0.10 (B) | **yes / no** over both sessions. *Note 2026-11-24 to session 1's §4: V22's first part, "no on the lines that exist (partial)", becomes yes with session 2's lines* |
| V23 | the two-column part ran (`x4 tog setup n_cols=2`, no `ok=0`) | 0.25 | **no** |
| V24 | `x4 togmix choice gpu_n_cols=4`; `both_cpu_vs_tog1_both_cpu` 1.225: ≥ 1.15 / ≤ 1.05 | 0.50 / 0.25 | **yes / no** |
| V25 | no wait reached the 2 s deadline (`x4 dl n=0` in all eight blocks; the no-flush check's designed 2 s waits are not deadlines) | 0.15 (lead) | **no** (kept out of the Brier score) |
| V26 | no parent block hung; the owner did not stop the app | 0.05 (lead) | **no** (kept out of the Brier score) |

V1 to V4, V12 to V14 are session 1's and are not re-scored (amendment §3).

### S2.4 Cells (§4.2), as written

Scored on the merged (union) band. Values *device*; means of the two ABBA arms and ratios arithmetic.

| # | value | band | scored |
|---|---|---|---|
| F1 | gate `set_to_start_p50_us` **381.7** (includes the post-release `clFlush`, amendment §4) | 110–280 | **miss, high** |
| F2 | `rtargs enq_us`, mean at args 2 (11.7, 11.7) − mean at args 15 (12.5, 12.9) = **−1.0** | −22 to −3 | **miss, high** |
| C1 | `gate_up_gbs` **23.32**; `gate_up_vs_11008` **0.966** | 21.5–27.0; 1.00–1.20 | **inside; miss, low** |
| C2 | `qkv_gbs` **25.65**; `qkv_vs_4096` **1.475**; `qkv_gbs` ÷ `m11008_gbs` 25.65 ÷ 24.13 = **1.063** | 19.0–25.5; 1.25–1.60; 0.93–1.10 | **miss, high (0.15 over); inside; inside** |
| D1 | `ovh_us_per_launch_replay` **44.3**; `replay_vs_sync1` **0.356** | 26–250; 0.55–0.97 | **inside; miss, low** |
| E1 | 1.x modes never saw the flag or the counter | – (given seen) | **not measured** |
| E2 | `svm_c11`: flag `rtt_p50_us` **0.9**; counter `iv_p50_us` **3.75** | 3–200; 1–60 | **miss, low; inside** |
| E3 | `q4p48_vs_q4n` **0.985** | 0.50–1.02 | **inside** |
| M1 | `dev_ratio` n 2 / 3 / 4 = **3.316 / 3.912 / 4.528** | 1.05–1.80 / 1.15–2.6 / 1.35–3.4 | **miss, high** (all three) |
| M2 | `wall_ratio` n 2 **2.681**; `max_rel_err` n 2 col 0 **6.748e-04**, col 1 **8.197e-04** | 1.04–1.65; 6e-4 to 8e-4 | **miss, high; inside; miss, high.** At n 3 and 4 the third column reads 9.319e-04 (descriptive) |
| T1 | one column: `cpu_alone_gbs` **26.38**; `both_cpu_gbs` **20.80** / `both_gpu_gbs` **19.72**; `sum_vs_cpu_alone` **1.536** | 24.5–27.0; 18.5–21.0 / 19.5–24.5; 1.50–1.72 | **inside** (all four). GPU cap 676 MHz during the cell (S2.2) |
| T2 | two columns `cpu_alone_gbs` **18.33** | 12.2–18.0 | **miss, high** |
| T3 | 18.33 ÷ 26.38 = **0.695** | 0.47–0.72 | **inside** |
| T4 | `gpu_alone_gbs` **7.76** | 14.5–23.5 | **miss, low** |
| T5 | `both_cpu_gbs` **17.50** | 11.7–17.5 | **inside** (at the upper edge) |
| T6 | `both_gpu_gbs` **7.96** | 13.5–22.0 | **miss, low** |
| T7 | `sum_gbs` **25.47** | 27.5–38.5 | **miss, low** |
| T8 | GPU ratio 7.96 ÷ 7.76 = **1.026** (cross-check device time 2948.1 ÷ 3013.7 = 0.978) | 0.85–1.00 | **miss, high** |
| T9 | sum ratio 25.47 ÷ (18.33 + 7.76) = **0.976** | 0.85–1.00 | **inside** |
| T10 | `sum_vs_cpu_alone` **1.389** | 1.9–2.8 | **miss, low** |
| T11 | mixed arm at `gpu_n_cols=4`: **1.225** | 1.00–1.30 | **inside** |
| S1 | `t_cpu_alone_ms` **1.835** | 1.80–2.05 | **inside** |
| S2 | feeder 0.25: `ratio` **0.964**; `gpu_ms` **1.201**; `wait_ms` **0.004** | 0.83–0.90; 0.85–1.15; 0–0.06 | **miss, high; miss, high; inside** |
| S3 | best lead `ratio` **0.923** at share **0.55** | 0.67–0.79 (share centre 0.50, P 0.60 to 0.78) | **miss, high**; the best share is not 0.50 (descriptive, not a band) |
| S4 | lead 0.20 / 0.35 `ratio` **1.033 / 0.956** | 0.88–0.96 / 0.79–0.86 | **miss, high / miss, high** |
| S5 | lead 0.50: `gpu_ms` **1.265**; `gpu_dev_ms` **1.104**; `wait_ms` **0.048**; `enq_us` **175.0** | 1.15–1.42; 0.90–1.10; 0–0.25; 35–125 | **inside; miss, high (0.004 over); inside; miss, high** |
| S6 | lead 0.55: `ratio` **0.923**; `gpu_ms` **1.326**; `wait_ms` **0.1005**; `gpu_dev_ms` **1.1745** | 0.72–0.84; 1.28–1.46; 0.11–0.29; 0.98–1.22 | **miss, high; inside; miss, low; inside** |
| S7 | lead 0.60: `ratio` **0.987**; `gpu_ms` **1.480**; `wait_ms` **0.2585**; `gpu_dev_ms` **1.333** | 0.77–0.90; 1.38–1.56; 0.29–0.47; 1.07–1.33 | **miss, high; inside; miss, low; miss, high (0.003 over)** |
| S8 | **K_meas 149.0 µs** (S2.7) | 170–400 | **miss, low** |
| S9 | 12,160 × 4,096 × 0.5625 B ÷ 1.1745 ms = **23.85 GB/s** | 22.5–27.5 | **inside** |
| S10 | `cpu_ms` ÷ ((1 − share) × 1.835): 0.25: 1.6295 ÷ 1.376 = **1.184**; 0.50: 1.485 ÷ 0.9175 = **1.619** | 1.08–1.22 / 1.20–1.40 | **inside / miss, high** |
| H1 | `extra_us` `write` **13.3** / `map` **88.3** / `ahp_map` **91.5** / `ahp_persist` **27.0** (all `match=1`) | 40–200 / 30–170 / 20–130 / 0–30 | **miss, low / inside / inside / inside** |
| R1 | x4r: `cpu_gbs_median` **21.01**; majflt per pass median **0** | 17.5–19.5; 0–5 | **miss, high; inside** |

**Tally, session 2** (each scored value once): 61 values, **28 inside, 33 misses (11 low, 22 high)**; E1 not measured.
**Both sessions:** 92 values, **39 inside, 53 misses (25 low, 28 high)**.

What the misses say, one line each (reading, not a rule):
- `_mc3` is slow, not cheap: at two columns its device time is 2,951 µs against 890 µs for the general kernel at
  one (8.6 against 28.5 GB/s of weight bytes). Every row that leaned on it (M1, M2, T4, T6, T7, T10) missed.
- The row split gains far less than priced (S2 to S4, S6, S7 high): best 0.923 against a centre of 0.72. S10 at
  share 0.50 (1.62 against 1.30) says why: the CPU rows run much slower beside the GPU than the pricing assumed.
- The join itself is cheaper than priced (S8 low, D1 low, H1 `write` low, session 1's B5 and B6 low).
- Stacking gate‖up does not raise the GPU rate (C1 ratio 0.966).

### S2.5 Decision rules: designs (§5.1), the together model (§5.2)

Amendment §5 items 1 and 2: no deadline in any cell, no hang, so every rule is applied to the cells as printed.
Inputs from session 1 are marked (s1) and carry the drift flag (S2.2).

| design | inputs | verdict |
|---|---|---|
| **A** launch-ahead | gate `set_to_start_p50_us` 381.7; s1 x4b cpu 4 spin `queued_to_start_us` 96.4 (s1); plain `submit_to_start_p50_us` 6.9 | **killed**: 381.7 ≥ 300 (session 2 alone), and it is not 60 µs below 96.4 (s1, drift flag). The flush is part of the mechanism (amendment §4). Beside it: gate `queued_to_submit_p50_us` 1,018.9 ≈ `ahead_us` 1,000 and gate `submit_to_start_p50_us` 174.8 against plain 6.9: the driver submits only at release plus flush and prepares nothing at enqueue. A launch behind a gate costs 381.7 against 147.0 for a plain one |
| **E / Y12** stacking | identity ok on both; `gate_up_vs_11008` 0.966; best lead `ratio` 0.923 | not killed (0.966 ≥ 0.90; 0.923 < 0.95). Criterion not met (0.966 < 1.00; 0.923 > 0.85): **open, not worth a build now**. q‖k‖v: identity ok, 1.063 ≥ 0.90: **not closed**. Note: `qkv_vs_4096` 1.475 rests partly on a slow second 4096 arm (19.64 then 15.14 GB/s) |
| **Y14** recordable replay | `x4 recq present=1`; `x4 rec setup ok=1`; replay 44.3, sync1 124.5 | not killed (44.3 ≤ 62.3). Criterion (≤ 100) met. K test (S2.7): +0.50 / +0.43 / +0.40 tok/s at the primary `pair` price, under +0.5 on Llama-2 and DeepSeek: **open, not worth a build now** |
| **F / Y15** persistent kernel | C11 ran; `svm_c11` flag `ok=1` 0.9 µs; counter `visible_during_run=1` 3.75 µs; `q4p48_vs_q4n` 0.985, `q4p192_vs_q4n` 0.986, `same_rows=1` | not killed (none of (i) to (iv)). Criterion met (4.65 ≤ 60, (iv) passes). K test (S2.7): +0.70 / +0.61 / +0.56 tok/s at the primary `pair` price: **worth a build** (host list for a lane design; not a device go). With Ebony's slack beside it: +0.45 / +0.39 / +0.36; in `linear`: +0.60 / +0.52 / +0.48. **Caveat (reading):** the probe's gemv runs at 1.86 (q4n) and 1.83 (q4p) GB/s, 0.076 of the general kernel's 24.13 (x4c). (iv) as written compares the persistent loop with the same simple kernel, so it passes; a persistent version of the fast kernel does not exist and has not been measured |
| **B** event spin | s1: savings 161.8 / 561.1 / 648.7 µs | not killed; criterion met (s1). Design K 188.6 (s1) + 40 + 13.3 = 241.9 µs > K_eff 162.3: −0.35 / −0.31 / −0.28 tok/s. **open, not worth a build** (drift flag). Session 2's own cpu-4 spin (`ovh_us` 130.6 to 140.3) would give 184 to 194 µs, also above K_eff |
| **B+ / Y13** | s1 best `ovh_us` 188.6 | same as B: **open, not worth a build** (drift flag) |
| **C** submit→start | s1 | criterion met (s1); §5.3 names no design K for C, so no "worth a build". **open** |
| **Y16** | s1 | criterion met (s1); no design K in §5.3. **open** |
| **D** whole matrices | lead `ratio` at 0.50 0.963; best 0.923 | **killed**: 0.963 − 0.923 = 0.040 > 0.03 |
| **cg** | `x4 mcsum n_cols=2 dev_ratio` 3.316 | **> 1.70: the width-2 route to 12 is closed** (the lane stays a 10 to 11 tok/s lever by §5.1's own words). **≥ 2.0: `_mc3` is no better than two launches; dropped.** Width 3 (descriptive): 3.912, not ≤ 1.3 |
| **hand-off** | `write` 13.3, `map` 88.3, `ahp_map` 91.5, `ahp_persist` 27.0, all `match=1` | **H = 13.3 µs** (`write`) |
| **any design at Llama-2 or DeepSeek size** | x4r (S2.6) | **held** |

**x4 kai** (descriptive): `kernel=DOTPROD`, `q4_buft=CPU_KLEIDIAI`, as in session 1, so r5 §2.1's tile cost holds and
T2 to T5 are read without a note.

**§5.2, the together model:**
1. Tile: T8 1.026 ≥ 0.93 → **`pair`**. T2 18.33 < 19, so **"`pair` at light CPU draw"**. (T9 0.976 ≥ 0.94 also.)
2. Mixed arm: `gpu_n_cols=4`, `dev_ratio(4)` 4.528 ≥ 1.3, so informative; 1.225 ≥ 1.20 → **`pair`**.
3. Model for the stack: mixed `pair` and tile `pair` → **`pair`**.
4. T3 0.695 < 0.80: every width-2 stack is "conditional on building the W-position kernel".

Reading beside it (not a rule): both readings come from cells where the GPU drew little, 7.8 GB/s in the tile cell
and 5.0 GB/s in the mixed arm, because `_mc3` is slow. "`pair`" here is `pair` at light draw on both sides. The
one-column cell, where the GPU draws 19.7 GB/s (at a 676 MHz cap), cost the CPU 21 % (26.38 → 20.80), as R74 did.

### S2.6 Residency (x4r, §5.4)

- **Held**, all three criteria (*device* inputs): median CPU read over passes 5 to 109 = **21.02 GB/s** ≥ 0.80 ×
  (29.90 − 0.36 × 19.57) = 18.28; median majflt per pass **0** ≤ 50 (445 in 110 passes, 38 passes with any);
  `x4 res done err=0`.
- Footprint (*device*): 59 copies, `gpu_mb=1427` (MiB, 1.50 GB) of GPU buffers (`kgsl_page_alloc` 1.75 GB), 2.6 GB anonymous, VmRSS
  4.11 GB. MemAvailable 2.21 GB before, 0.83 after the GPU allocation, 0.41 after the fill, 0.51 after. GPU read
  19.57 GB/s; `gpu_max_mhz` 840 on every pass.
- Reading (§5.4 "held"): 4.1 GB with 1.5 GB pinned stayed resident for 15.4 s; it covers Llama-2's and DeepSeek's
  totals and their GPU share; evidence for, not proof.
- Beside it (descriptive, not a criterion): the last ten passes fell from 16 to 4.64 GB/s with majflt 0 on the last
  pass; and swap free rose from 1.33 to 2.07 GB during the block, which suggests that other processes' pages were
  dropped or the processes killed while the arm held. Neither is explained by the file.

### S2.7 K, hand-off, prices, the priceable rule (§5.3; *estimated* from *device* inputs)

- **K_meas.** Lead shares with a two-arm mean `wait_ms` > 0.1: 0.55 (0.1005) and 0.60 (0.2585). Two-arm mean of
  (`gpu_ms` − `gpu_dev_ms`): 0.55: 1.326 − 1.1745 = 151.5 µs; 0.60: 1.480 − 1.333 = 146.5 µs. **K_meas = 149.0 µs.**
  Per arm the first half (GPU at 840 MHz) reads 179 to 191 µs and the second half (GPU capped at 676 MHz, CPU at 1,785) 112 to 114 µs; the rule takes
  the means as written.
- **H = 13.3 µs. K_eff = K_meas + H = 162.3 µs.**
- Cross-checks (descriptive): K_x4b = 188.6 (s1) + 40 = 228.6 µs (drift flag); session 2's cpu-4 spin `ovh_us` +
  40 = 171 to 180 µs; **K_r = (0.923 − 0.621) × 1.835 ÷ 0.521 = 1,064 µs**. K_r is seven times K_meas: the stage
  model with K 149 predicts a best ratio of 0.663 for this cell and the phone gave 0.923. The split loses its gain
  to the CPU side (S10), not to the join.
- **Design K** for designs that passed "worth a build": F: 0.9 + 3.75 + 13.3 = **17.95 µs** (K_best, conditional on
  building a persistent lane kernel). For comparison, Y14 44.3 + 13.3 = 57.6 µs; B / B+ 241.9 µs.
- **cg = 3.316** (*device* ratio).

Prices (scripts as committed, `python3 -I`; at C 40, trained head, C7 head × 0.5, C5 body × 0.93, O 5.2, every
unbuilt lever at its centre; tok/s Yi / Llama-2 / DeepSeek; gap = ms per token − 79.2 at K_eff):

| model | K_eff 162.3 | gap (ms) | K_best 17.95 (F, conditional) |
|---|---|---|---|
| `pair`, Bonnie's slack | 10.91 / 10.66 / 10.33 | +7.9 / +9.9 / +12.8 | 11.61 / 11.27 / 10.89 |
| `pair`, Ebony's slack (J0 60) | 10.42 / 10.23 / 9.93 | +12.0 / +13.7 / +16.5 | 10.87 / 10.62 / 10.29 |
| `linear` (beside, not named by §5.2) | 10.59 / 10.33 / 10.03 | +10.5 / +12.8 / +15.5 | 11.19 / 10.84 / 10.50 |
| `old` (R74's form) | 11.20 / 10.97 / 10.61 | +5.7 / +7.4 / +10.3 | – |

At K 0 and cg 3.316, `pair` with Ebony's slack gives 10.93 / 10.67 / 10.34 and `linear` 11.27 / 10.91 / 10.57: no K
reaches 12 at the measured cg.

**Priceable rule (scored): NOT PRICEABLE.** The model is `pair`, so the Ebony-slack column applies. At cg 3.316 the
table has no DeepSeek value (its last rows already read "none" at cg 1.70 and 2.00), and the script confirms that
no K, not even 0, reaches 12 on DeepSeek. K_eff 162.3 and K_best 17.95 are both without a line.

### S2.8 The memo-12 kill lines X4-c to X4-f (Ebony, `ebony-joins.md` §7.1; reported, not prereg rules)

- **X4-c** (identity `ok=0`, alloc error, or stacked rate < 0.9 × the 11008 rate → Y12 closed): **not triggered**
  (identity ok, 0.966).
- **X4-d** (queue cannot be created, or per-launch overhead not under 0.5 × sync1's → Y14 closed): **not
  triggered** (44.3 against 0.5 × 124.5; the line's printed sync1 values 293 / 252 / 261 are older estimates).
- **X4-e** (flag not seen within 1 ms or the child hangs; or (2) median > 50 µs; or (3) < 0.8 × g → Y15 closed):
  (1) and (2) **not triggered** on `svm_c11` (0.9 µs, 3.75 µs, no hang). (3) as Ebony wrote it, against `x3 alone`:
  1.83 ÷ 24.13 = 0.076, **triggered**; against the prereg's (iv) measure (q4p ÷ q4n) it passes. The probe's gemv is
  not upstream's kernel, so (3) as written measures the simple kernel's rate, not the loop's cost. Y15 stays open
  by the prereg's rule, with no fast persistent kernel measured.
- **X4-f** (completion→START ≥ 300 µs → launch-ahead hides nothing): **triggered** (381.7).
- X4-a and X4-b: session 1 (§7). X4-b's reading ("λ stays in 250 to 450 µs") is contradicted in situ: K_meas is
  149.0 and session 2's cpu-4 spin `ovh_us` 130.6 to 140.3 (drift flag).
- The cause check `x4 la nf` (*device*): `done_without_flush` 0 / 0 / 0 after 2 s each, `done_after_flush` 1 / 1 / 1
  (5,642.8 / 2,000.0 / 7,578.0 µs after the flush). Adreno does not submit a command released after its flush until
  another flush. **This confirms on the phone the cause of session 1's x4f hang** (§2's reading, Tom's shim).

### S2.9 Combined R75 verdict (both sessions)

- **Complete.** Every block of the pre-registration has run: x4i, x4a, x4b in session 1; x4f, x4c, x4d, x4e, x4m,
  x4s, x4r in session 2 (x4i repeated as a build step). Session 1's x4f is void; session 2's x4f replaces it under
  the amendment. No block of session 2 is void.
- **Cells:** 92 scored values, 39 inside, 53 misses (25 low, 28 high).
- **Designs:** A killed; D killed; width-2 route to 12 closed by cg 3.316, `_mc3` dropped; F worth a build (primary
  `pair` price, conditional, with the kernel caveat); Y14, E, B, B+ open, not worth a build now; C and Y16 criterion
  met with no design K; q‖k‖v stacking not closed.
- **Contention model:** `pair` (at light draw on both sides).
- **K_eff 162.3 µs** (K_meas 149.0 + H 13.3), *estimated* from *device* inputs, below the merged band 170 to 400.
- **Residency at 4.1 GB with 1.5 GB of GPU buffers: held.**
- **12 tok/s on all three: not priceable** through the lane after R75 (§5.3 rule).
- **Kill lines:** X4-a not triggered, X4-b triggered (s1; contradicted in situ), X4-c and X4-d not triggered,
  X4-e (3) triggered as written, X4-f triggered.
- **§7, what the result means:** the "K_eff above the line" case (there is no line at cg 3.316), together with
  "cg > 1.70". The campaign verdict stands: 12 tok/s on all three is out of reach by priced levers. F goes on the
  host list.
- **Brier score** (§4.1 outcomes of both sessions; V3, V5, V9, V13, V16, V18, V19, V21, V22, V24 counted per part;
  V25 and V26 kept out; ranges scored at both ends): **0.142 to 0.163 over 39 outcomes.** Session 2 alone (26
  outcomes): 0.154 to 0.175. Session 1 with V22's first part now yes (13): 0.119 to 0.141 (§4's 0.066 to 0.087 was
  with V22 no). Largest terms: V10 (0.72 to 0.81: the persistent-kernel lines all passed), V22 first part (0.72), V5
  replay passes (0.56 to 0.64), V18 held (0.42 to 0.56). Lead's V25 and V26 (both no): 0.0225 and 0.0025.

### S2.10 What R75 means for 12 tok/s on Yi, Llama-2 and DeepSeek (*estimated*; arithmetic, not scored)

This part is the lead's reading. It is not a rule of the pre-registration and it scores nothing.

**What R75 gives (in situ):** K_eff 162.3 µs (better than the 300 the stacks used), cg 3.316 for upstream's
`_mc3` (much worse than 1.25), `pair` at light draw, residency held at 4.1 GB, and a row split that saves 7.7 % at
its best share on the largest matrix (much worse than the 28 % priced).

**The honest ladder** (`bonnie_floor_r14.py` loaded unchanged; rungs re-run at K_eff 162.3; width 2 by two launches
of the general kernel, cg 2.0, because `_mc3` is dropped; coupling α 0.25 as committed and 0.17 as refitted in
`bonnie-joins-r32.md`; tok/s Yi / Llama-2 / DeepSeek):

| rung | what | `linear`, α 0.25 | `pair`, α 0.25 | `pair`, α 0.17 |
|---|---|---|---|---|
| 0 | stock, CPU alone | 6.90 / 6.21 / 6.22 | same | same |
| 1 | lane, width 1, K 452 (R74), O 8, head as measured | 8.34 / 7.76 / 7.64 | same | same |
| 2 | = 1 at K_eff 162.3 | 9.61 / 8.97 / 8.72 | same | same |
| 3 | + head in the repack type | 9.75 / 9.03 / 8.89 | same | same |
| 4 | + O 5.2 | 10.04 / 9.27 / 9.13 | same | same |
| 5a | width 2, cg 2.0 (two launches), W-position CPU kernel, trained head | 10.48 / 10.44 / 9.78 | 10.85 / 10.85 / 10.12 | 11.16 / 11.18 / 10.40 |
| 5b | width 2 with `_mc3` at the measured cg 3.316 | 9.66 / 9.56 / 9.02 | 9.94 / 9.87 / 9.28 | 10.25 / 10.19 / 9.56 |
| **6** | **5a + campaign 2's two-stage head (r14's "priced lane")** | **10.66 / 10.48 / 10.07** | **11.05 / 10.89 / 10.44** | **11.37 / 11.22 / 10.73** |
| 7 | 5a + C5 × 0.93 + head × 0.5 (committed form) | 11.50 / 11.28 / 10.89 | 11.91 / 11.71 / 11.29 | 12.26 / 12.07 / 11.61 |
| 7F | = 7 at F's K 17.95 (unbuilt persistent fast kernel) | 12.40 / 12.07 / 11.61 | 12.98 / 12.65 / 12.13 | 13.36 / 13.03 / 12.48 |

(`pair` in this table uses Bonnie's slack, as r14 did. With Ebony's slack, rung 7F at α 0.25 reads 12.05 / 11.84 /
11.40.) For comparison, rung 7 at the old centre cg 1.26 would have read 12.67 / 12.52 / 12.01 (`linear`): the move from cg 1.26 to the two-launch fallback (cg 2.0, because the measured `_mc3` is worse) took 1.2 tok/s off
the committed form.

**Why the ladder is optimistic after R75.** Every rung from 2 up prices the lane with the stage model. R75 tested
that model in one cell and it failed by a wide margin: at K 149 the model gives a best split ratio of 0.663 for
gate‖up and the phone gave 0.923 (S2.7, K_r 1,064 µs). If the measured ratio held for the whole body, the lane at
width 1 would give about **7.40 / 6.68 / 6.66 tok/s** (body × 0.923, O 8, the measured head), +0.44 to +0.50 over
stock and below rung 1. That figure is crude (one fused matrix, half the arms under caps, small matrices would gain
less), but it is the only lane number in the file that rests on a split the phone ran.

**Lead's judgement.**
- The lane at its centres remains a 10 to 11 tok/s lever on paper (rung 6: 10.1 to 11.4). On the phone's split it is
  a +0.5 tok/s lever. Neither reaches 12 on any subject without at least two unbuilt pieces (a persistent fast GPU
  kernel and the W-position CPU kernel) plus C5 and the half-size head, and none reaches it on DeepSeek with
  Ebony's slack.
- P(12 on all three) is now below the 0.002 to 0.003 on the books (judgement; about 0.001).
- The goal is out of reach by every priced lever. That is the verdict, with these numbers. No rule was moved.

**Ranked next steps** (nothing here is a device go; device rounds stay held by the owner):
1. **Bonnie and Ebony, joint round (research, no device): the split gap.** From R75's own lines (per arm, thread
   labels, clocks, uncapped against capped halves, S10, the one-column cell, x4r), explain why the split saves 7.7 %
   where the stage model says 34 %. Refit the CPU-beside-GPU term, re-run the ladder with R75's K, cg and split, and
   say whether the GPU lane is still worth more than +0.5 tok/s anywhere. Every lane rung rests on this. Ebony
   reviews.
2. **Tom, host: `_mc3` as built.** Check the probe's launch of `kernel_gemv_noshuffle_q4_0_f32_mc3` (work-group
   256, `nsg` 4, global size, layout) against upstream's own dispatch. The question is whether 2,951 µs at two
   columns is the kernel or the probe. The rule stands either way; the answer says whether a two-column kernel is a
   dead idea or an unbuilt one.
3. **Tom, host: price F's missing piece.** A persistent, row-claiming version of the general noshuffle kernel (the
   only design that passed "worth a build"). Design and host build only, behind a knob, no device.
4. **Lead to owner:** report R75 and the verdict in Korean with these numbers. Ask whether to keep the lane research
   (items 1 to 3), to realign the campaign, or to close it.
5. **R76, x5 kit r37 (Bonnie r18; r36 is taken by this re-run):** still relevant, because the residency of the
   file-backed body under choice A (the owner's path) is a precondition of any decode on Llama-2 and DeepSeek, lane
   or not. x4r's held arm is anonymous and shows nothing about file-backed pages. It does not change the 12 verdict,
   so it waits for the owner's permission and its own pre-registration.
6. **Only if item 1 finds a probe artefact:** a small split cell at cool clocks (CPU rows alone on three threads
   against beside the GPU), with its own pre-registration.
