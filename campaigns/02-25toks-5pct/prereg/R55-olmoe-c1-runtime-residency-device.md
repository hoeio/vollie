# Pre-registration R55: OLMoE on campaign 1's runtime, with residency measured (kit r19)

Committed and pushed **before** the APK is handed to the owner. Changes go into `R55-amendment-1.md`.

- **Date / author:** 2026-11-14, clyde (lead).
- **Campaign / model / file sha256:** campaign 2; OLMoE-1B-7B-0125-Instruct; `olmoe-Mi.gguf`,
  `5abf710b775cd64fd89d74a783c8a7056a409ba795cd401b9dd448ef8e444abb`.
- **Agent models** (full IDs): clyde `claude-fable-5-2`; bonnie `claude-opus-5-5` (memo-15, the bands);
  tom `claude-opus-5-5` (memo-13, memo-14, the kit); annie `claude-sonnet-5-5`.
- **Build under test:** APK `vollie-c2-1.19.0+19-research-r19.apk`, 17,162,019 bytes, sha256
  `9b1cd9a340e462163e9c46ce8e3fea2c966cfb286b01ab14beaaa5d993bdc668`, package `com.insaneutils.vollie.c2` (update
  over r17/r18). Runtime: llama.cpp `620d2225c` + **campaign 1's** `patches/llama.cpp/0001` to `0003`, unchanged
  (owner, 2026-11-14: use campaign 1's Vollie as well); app patches `patches/app-c2/0001` to `0003`.
- **Device:** the owner's Galaxy Z Flip 3 (Snapdragon 888, 7.7 GB). Repack on. `GGML_REPACK_RELEASE=1` at the
  kit's own load, so it covers every row.
- **Labels:** *estimated* (memo-15 §3) unless a value names a report.

## 1. What is tested

Three rows on one module, 5 official runs and 1 per-token run each, in the order a, b, c, a, b, c
([memo-14](../reports/memo-14-kit-r19.md)), after one unscored `cold` run:

- `c1_off`: every default-on knob of campaign 1's patch set to 0 (the stock arithmetic);
- `c1_ship`: campaign 1's shipping settings without the Granite numerics (no skip, no two-stage head);
- `c1_ship_io`: the same with `GGML_EXPERT_IO=5` (routed read-ahead plus next-layer guess).

New in the kit: a working cool-down gate (headroom below 0.65; up to 10 minutes before the first timed run,
90 s before the others; a missing reading is read again, never taken as cool), and the page-cache residency of
the model file (mincore, experts and rest) before and after every run. No numerics change: all three rows gave
the stock greedy text on the host.

## 2. Budget accounting

| item | PPL | % over stock |
|---|---|---|
| stock served file (R50, *host*) | 11.7146 | 0 |
| all three rows | 11.7146 (exact levers; identical to the base on 4 chunks, memo-13 §7) | 0 |
| budget / hard ceiling | 12.3003 / 13.1304 | |

Identity: the reply text of all 18 scored runs and of `cold` must be the same.

## 3. Prediction

The phone's state decides the result and the kit cannot choose it, so the prediction is a table by state.
State of a run, from the report: *resident* = mincore of the model file ≥ 2.6 GB before the run; *full clock*
= A78 ≥ 2,000 MHz at run end; *throttled* = 1,075 to 1,450 MHz.

Printed tok/s per run (steady ms in brackets):

| row | resident, full clock | resident, throttled | not resident, full clock | not resident, throttled |
|---|---|---|---|---|
| `c1_off` | 14.5, 13.0 to 16.0 (66) | 9.2, 7.0 to 11.8 (105) | 4.3, 2.8 to 5.8 (224) | 3.7, 2.4 to 5.2 (262) |
| `c1_ship` | 19.9, 18.4 to 21.7 (48) | 13.2, 10.1 to 17.0 (73) | 9.2, 5.1 to 12.1 (104) | 7.4, 4.2 to 10.5 (129) |
| `c1_ship_io` | 19.8, 17.9 to 21.8 (48) | 13.1, 9.9 to 17.1 (73) | 9.8, 5.1 to 13.5 (98) | 7.8, 4.1 to 11.5 (123) |

Runs between the states (clock 1,450 to 2,000 MHz, or mincore 2.0 to 2.6 GB) are scored against memo-15 §1.2's
formula with the run's own clock and mincore, band ± 25 %.

**Which state is expected:** not resident (probability 0.8 to 0.9); mincore 1.5 to 2.1 GB; full clock for the
first run or two, then 1,440 to 1,800 MHz. So the expected medians are `c1_off` 3.5 to 4.5, `c1_ship` 7 to 10,
`c1_ship_io` 7.5 to 10.5 tok/s. Ratios that should hold whatever the day: steady ms of `c1_ship` / `c1_off` is
0.68 to 0.80 if resident and 0.45 to 0.55 if not.

## 4. Pass rule and kill rule

- **Each row is scored per state:** the median of its runs in a state against that cell's band; a cell with
  fewer than two runs is reported, not scored. Hit or miss as written; the "expected state" is scored separately
  (hit if at least 12 of the 18 runs are "not resident").
- **Residency question (the reason for this round):** report mincore before each run against the 2,725 MB the
  benchmark touches (memo-13, *host*). If no run reaches 2.6 GB, memo-15's reading stands: on this phone in this
  state OLMoE is fault-bound whatever the kernels do.
- **Kernel-level estimates apply (memo-08) only if** at least three `c1_ship` runs are resident at full clock.
  Then their median is compared with 19.9 tok/s and the lever work continues from there.
- **Nothing here passes or fails 25 tok/s.** A decision about the goal follows from the residency numbers and is
  the owner's (state-conditioned claim, another file, private expert memory, or "no"); the lead writes the
  options with this round's numbers.
- **Void:** hash mismatch (model, libraries); reply texts differ between rows; the gate records no headroom
  reading at all (then the gate is still broken).

## 5. Claim rule

Not a goal claim.

## 6. Hygiene counts

Per run: the gate's headroom series and wait; `gate_timeout` recorded, and such runs are kept but flagged;
clocks at run end; major and minor faults; mincore before and after. A run is "fault-bound" by the parser's
flag. Flags do not remove a run here, because the state is what is being measured.

## 7. What a miss means

- Rows faster than their cell: the I/O term of memo-15 is too pessimistic; refit with this report (it was fitted
  on r17 alone).
- Rows slower: look first at the gate record and the clocks, then at mincore.
- Resident runs appear (against the expectation): the phone's state is better than its history suggests; the
  next kit then runs the lever comparison of the withdrawn R54 on this runtime.
