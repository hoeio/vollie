# R77 result: UD and Y1b kernel cells on the phone (kit r38)

Scored by clyde (lead, `claude-opus-5-5`) on 2026-11-28 against `prereg/R77-ud-y1b-kernel-cells-device.md` (commit `67ec3f6`,
pushed before the run). Tag `prereg-R77` was not on origin when this was scored. Numbers are *device* unless labelled.

- **JSON:** `reports/device/vollie-c3probe-r38-2026-11-28T1617.json`, sha256
  `9125c67f09a6e1d79848900f1fa3307c0c652c894274fbde82f9d99504ce7569`; kit r38, app 1.38.0+38, started 16:17:24, not stopped early.
- **Parser:** `tools/c3_probe_parse.py` (sha in prereg §0), R77 section, unchanged.

## 1. Conditions and voids

- Power saving off, charger out (`plugged_start` false), battery 27.2 °C, level 75 %. Gates passed before all three blocks
  (0.649 / 0.647 / 0.639).
- No cap through `udk1` and `udr1`; `y1bd` ended at caps 1,555 MHz and `thermal_pwrlevel` 5 (canaries above 0 from y1bd pos 26).
- Identity: `udk1` 12/12, `udr1` y1b 1 and 12/12, `y1bd` 2/2: all equal.
- **Prereg §4.1 voids a triplet if any of its arms has `ramp_ok=0`, an off-core label, or a canary with `thermal_pwrlevel` > 0.**
  On the 4-7 set **every** probe stock arm carries an off-core label (one of the four threads marked M, often `7:MU`; parser
  "arms off-core (*)"), so every 4-7 triplet of all three blocks is void. On cpu 4 alone: `udk1` clean (labels `4:C`, ramp_ok 1);
  `udr1` void (labels `4:MU`/`4:U`, ramp_ok 0); `y1bd` valid only for 4096² (pos 10 to 16), the rest void by the canary rule.
- The same pattern (the cpu-7 thread off-core in most cells) was recorded in R72 (`R72-result.md` §2). The prereg should have
  allowed for it; it did not, and it is scored as written.

## 2. Scores

| # | prediction | P | result | inside |
|---|---|---|---|---|
| D1 | K1 `k` on `down`, cpu 4, median | 0.7 | 2.613 / 2.518 / 2.543 → median **2.543** (band 2.2 to 3.1) | hit |
| D2 to D8 | 4-7 cells | – | **void** (off-core label in every 4-7 triplet) | not measured |
| D9 | identity all equal | 0.95 | yes | hit |

Two scored statements, both hits; Brier 0.046 (*estimated*, arithmetic).

## 3. Decision rules as written

All three rules (R3, R1, Y1b) are written on the 4-7 set. Every 4-7 triplet is void, so **no rule fires**: R3 is neither
built nor dead, R1 is not killed by rule, Y1b is not adopted. The lead reports; the owner decides (§5).

## 4. Descriptive readings (not scored; void cells are marked)

| cell | cpu 4 alone | 4-7 (void: off-core labels) | rule line |
|---|---|---|---|
| K1 `down` keep 0.5, `k` (Yi / Llama-2 / DeepSeek) | 2.61 / 2.52 / 2.54 (clean) | 1.97 / 1.98 / 2.83 (two also `stock_diff` > 0.03) | R3 alive ≤ 3.15, dead > 4.5 |
| `rows` (Y1b on 70 % `gate` rows), `k` | 7.7 / 7.5 / 7.9 (labels U, ramp_ok 0) | 2.39 / 2.36 / 2.34 | R1 dead > 2.18 |
| dense Y1b on `gate`, `t_vs_stock` | 7.30 (labels U) | 2.49 | – |
| Y1b ÷ ggml plain, 4096² / 11008 × 4096 / 4096 × 11008 | 0.90 (clean) / 0.84 / 0.74 (canary > 0) | 2.11 / 2.38 / 1.67 | adopt ≤ 0.90 median and ≤ 1.00 each |
| ggml plain ÷ probe stock, same shapes | 1.94 / 1.92 / 1.83 | 0.89 / 1.02 / 0.92 (`stock_diff` 0.17 to 0.35) | – |
| ggml KleidiAI ÷ probe stock | 1.06 / 1.03 / 1.04 | 0.89 / 0.81 / 0.66 | – |

Readings:
- **K1 on `down` (R3, A′):** the clean one-thread cell reproduces R72 L1 (2.621 with a random mask) within 4 %, and every K1
  reading, clean or void, is under R3's strictest break-even (3.15). Nothing here argues against R3; the rule cell is missing.
- **R1:** every reading is above its kill line (2.18): 2.34 to 2.39 at four threads, about 7.5 on one thread. R1 looks dead on
  the phone as on the host.
- **Y1b:** at four threads Y1b is 1.7 to 2.4 times slower than ggml's plain path on all three shapes. On one thread it is 0.74 to
  0.90 of plain. The four-thread decode is what matters; Y1b looks parked.
- **New, and the largest reading of the round:** at four threads ggml's **plain** (`--no-repack`, choice A's path) mat-vec ran at
  0.89 to 1.02 of the probe's repack-type stock, against 1.83 to 1.94 on one thread. If that holds, choice A's stock at four
  threads is close to the packed stock (bus-bound either way), and Bonnie's choice-A stock of about 5.1 / 4.6 / 4.5 tok/s
  (*estimated* from one-thread ratios) would be pessimistic. These are void triplets with stock arms 17 to 35 % apart, so this is
  a lead, not a number.

## 5. For the owner

Choose:
1. **Take the descriptive readings as direction only, and move to the first decode round (R73)** on choice A: measure choice A's
   stock decode on the three subjects directly (it has never run on the phone), with UD's A′ half as an arm only if Tom builds
   it. R1 and Y1b leave the list on R77's readings (recorded as descriptive, not as rule outcomes).
2. **Re-run R77 under an amendment** that scores on cpu 4 alone and on the 4-7 set with the off-core rule replaced by the
   per-thread rate check R72 used, written before any new data. Same kit, one short run.
3. Stop here and record.

The lead recommends **1**: the question that now matters most is choice A's real decode rate, which no probe cell can give.
