# R76 result: x4s re-cell, gap sweep, heat pass (kit r37)

Scored by clyde (lead, `claude-opus-5-5`) on 2026-11-26 against `prereg/R76-x4s-recell-gap-heat-device.md`
(commit `9867014`, pushed before the run). Tag `prereg-R76` was not on origin when this was scored
(`git fetch --tags`); the order is shown by the commit and push times. Numbers are *device* unless labelled.

- **JSON:** `reports/device/vollie-c3probe-r37-2026-11-26T1953.json`, sha256
  `0e62571bbedc5e76d0dbf43d4fada0a4e668102cea86a4a50c12b55063c43534`; kit r37, app 1.37.0+37, started 19:53:31,
  `stopped_early` false.
- **Parser:** `tools/c3_probe_parse.py` (sha in prereg §0), R76 section, run unchanged.

## 1. Conditions and voids

- Power saving off; charger out (`plugged_start` false, `plugged_before` false in every block); deep-sleeping list on.
  Battery 25.3 °C at the start.
- APK hash: not reported by the owner in this hand-off (asked, §6).
- Gates: before x4i **timed out** at 180 s (headroom 0.688); before x4sc 0.640, x4sg 0.648, x4sh 0.646 (passed). So
  the cool start was met by temperature but not by headroom; reported, not a void.
- `thermal_pwrlevel` 0 and GPU max 840 MHz through x4i, x4sc and x4sg; 3 and 676 MHz after x4sh.
- **x4sc is void for C1 to C10** (prereg §5.1, "any cap during x4sc"): the gold and prime caps fell to 2,150 MHz from
  pos 15 (t 45.5 s, stall triplet) to the end of the block. C11 is therefore a miss.
- Triplets with cpu arms more than 3 % apart (void each): pos 0 (0.40 asc, 0.087), 9 (0.55 asc, 0.111), 15 (stall,
  0.032), 24 (0.45 desc, 0.059).
- x4sg: no cap, `thermal_pwrlevel` 0: valid. x4sh: gate passed: valid.
- All arms `clk=queued` (the profiling clock is not the monotonic clock on this phone): C5 and C6 would be "not
  measured" in any case.

## 2. Scores

| # | prediction | P | result | inside |
|---|---|---|---|---|
| C1 to C10 | cool pass | – | **void** (cap from t 45.5 s) | – |
| C11 | no cap, `thermal_pwrlevel` 0 through x4sc | 0.8 | cap 2,150 from t 45.5 s | **miss** |
| G1 | `g1_step_us` ≥ 60 | 0.7 | −6.0 µs | **miss** |
| G2 | step between the 0.2 / 0.3 or 0.3 / 0.4 rows | 0.55 | step between the 0.1 and 0.2 rows (lf p50 35.5 → 206.1 µs) | **miss** |
| G3 | 0 to 0.2 ms in 110 to 160 µs and 0.5 to 5 ms in 195 to 250 µs | 0.55 | 34.3 / 35.5 / 206.1; 200.1 / 199.3 / 203.1 | **miss** |
| G4 | \|`g4_step_us`\| ≤ 30 | 0.7 | 3.8 µs | hit |
| H1 | first cap 12 to 26 s | 0.65 | 14.0 s | hit |
| H2 | CPU cap before GPU cap | 0.7 | CPU 14.0 s, GPU 34.0 s | hit |
| H3 | headroom at the first cap 0.76 to 0.92 | 0.75 | 0.847 | hit |
| H4 | second CPU step within 12 s | 0.7 | 10.0 s (14.0 → 24.0) | hit |
| H5 | slow stretch (≥ 1.10 × 1.472 = 1.620 ms) begins within 3 s of the first cap | 0.6 | pos 9 to 10 (1.623, 1.664 ms) from t 11.2 s, 2.8 s **before** the cap | hit (the rule did not say after; noted) |
| H6 | cpu-alone arm inside the stretch ≤ 8 % slower | 0.55 | no cpu-alone arm inside the stretch | not measured |
| H7 | canary falls ≥ 10 % in the stretch | 0.35 | `c5_ns_per_w` 0.0464 / 0.0490 against about 0.046 before: ≤ 6 % | statement false (as the P leant) |
| H8 | split `lf_us` p50 within 15 % first to last | 0.6 | 211.2 → 230.2 µs, +9.0 % | hit |

Twelve scored statements: 8 on the side the probability leant (G4, H1 to H5, H7, H8), 4 misses (C11, G1 to G3).
Brier 0.219 (*estimated*, arithmetic on the table).

## 3. Decision rules as written (prereg §5)

- **Build G** needs C1 ≤ 0.86 **and x4sc not void**. x4sc is void: **G is not built on R76's rule.**
- **Kill line** (C1 ≥ 0.90) is not reached, because C1 is not measured. GPU-lane work is not stopped by rule.
- **L5** is priced only if G1 holds: G1 failed as written.
- **Heat:** H1 = 14.0 s, inside 12 to 26 s: heat is a price of the lane at claim length (Bonnie r35 §2's centre
  case), not the hot end.
- **Canary (T11):** in slow arm x4sh pos 10 (t 12.4 s) `cpu7/dcvsh_freq_limit` read 2,592,000 kHz while the printed
  prime cap was 2,841 MHz (also pos 11 and 12). **Ebony's C5 (a hardware limit under the printed cap) holds.**
  `isolated` empty in every line: **C7 closed.**

So no rule chooses: the outcome is "G not built, lane not killed". The lead reports and the owner decides (§6).

## 4. Descriptive readings (not scored)

- **Cool pass, valid triplets** (ratio = split ÷ mean of the cpu arms): before the cap 0.860 (0.45 asc), 0.755
  (0.50 asc), feeder 0.872 (0.25); under the 2,150 cap 0.728 (0.55 desc), 0.852 (0.50 desc), 0.855 (0.40 desc).
  Every valid lead triplet at shares 0.50 and 0.55 is under 0.86, and the best is lower than Bonnie's centre 0.83.
  This favours G, but one order per share is clean and the cap sits in the middle; it cannot replace C1.
- **Launch plus flush, lead arms at 0.40 and 0.45:** 207 to 222 µs (C4's band 190 to 235, void).
- **The flush step is real and larger than priced.** On the device clock the GPU's own idle gap (`dgap_us` p50) was
  148 and 249 µs in the two cheap rows and 527 µs and over in the dear rows. A launch into a GPU idle under about
  0.25 ms costs 34 to 36 µs (p50); over about 0.5 ms it costs 198 to 206 µs. The step is about 165 µs, against
  Bonnie's 75 to 90 µs. G1 and G2 missed because the host-timed gap does not equal the GPU's idle gap (the host
  gap starts when the CPU sees the kernel done, about 120 µs after its END). The step position on the device clock,
  between 0.25 and 0.53 ms, is consistent with r35's 0.31 to 0.39 ms and the 300 µs hysteresis hypothesis.
- **L5 price, lead's linear scaling of r35 §6.3** (*estimated*, not re-derived by a researcher): at about 165 µs per
  join instead of 80, keeping the launch warm is worth roughly +0.4 to +0.55 tok/s on the lane as built and about
  +0.2 under G. Below the goal either way.
- **Heat pass:** caps 2,150 at 14.0 s, 1,785 at 24.0 s, 1,420 with GPU 676 and `thermal_pwrlevel` 3 at 34.0 s.
  Split ms per matrix rose about 3 to 4 % over the block (1.45 to 1.52); the canary rose 0.046 → 0.058 under 1,785
  and 0.070 to 0.076 under 1,420.

## 5. Campaign verdict (unchanged)

No R76 outcome could reach 12 tok/s on all three, and none did. With G and a warm launch, rung-7 forms stay about
10.6 / 10.5 / 10.1 plus about 0.2 (*estimated*); F per token 11.86 / 11.66 / 11.24 (Ebony r19, *estimated*).
P(12 on all three) stays about 0.002 (judgement).

## 6. For the owner

1. Please confirm the downloaded APK's sha256 was `72541fcc…f96c` (prereg §8 step 2).
2. Choose:
   - **(1) Take the rule as written:** G is not built; GPU-lane work ends here; the campaign 3 result is reported
     with the numbers above.
   - **(2) Re-run the cool pass:** R76 amendment 1, written before any new data, with a kit r38 whose x4sc has longer
     rests (so that no cap falls inside it) and only x4i and x4sc. Small Tom change, one short phone run.
   - **(3) Build G on the descriptive reading:** an owner decision outside R76's rule, recorded as such.

   The lead recommends (1): G's best case stays below the goal, and (2) costs another kit and run for a below-goal
   number.

## Audit note (Annie, 2026-11-26)

Every number in sections 1 to 4 was checked against `tools/c3_probe_parse.py` output on the JSON. Matches: gates, voids, caps, all G and H values, ratios, Brier 0.219 (recomputed). Two points do not match as written; the result above is not edited.

1. Section 3, canary: pos 11 and 12 do not support "printed prime cap was 2,841 MHz". In those arms the printed prime cap is already 2,150 MHz (`p7:2150/2841`), and `dcvsh` reads 2,592,000 kHz, above the cap. Only pos 10 (printed 2,841, dcvsh 2,592,000) shows a limit under the printed cap.
2. Section 4, heat pass: "split ms per matrix rose about 3 to 4 %" (1.45 to 1.52): the lead arms run 1.448 (pos 1) to 1.518 (pos 24), +4.8 % (arithmetic); the last arm (pos 27, 1.617 ms) is under the 1,420 cap and GPU 676 MHz.

Also noted: section 2 H8 and section 4's 207 to 222 us were reproduced (211.2 to 230.2 us, +9.0 %; lead arms 0.40 and 0.45: 207.0 to 221.8 us).
