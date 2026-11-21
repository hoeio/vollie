# R71 amendment 1: S2 (c1) and S3 (stock) run on kit r31 by "Resume chain" (the r30 restart was blocked)

Date: 2026-11-21. Author: clyde (lead). Written and pushed before any S2 or S3 data exists. S1 and the cool-down
(kit r30, 2026-11-20) are scored and stand as they are (`reports/memo-33-work/score-r71-S1.txt`).

## Why

On the phone the chain stopped at the restart into c1 ("restart into c1 did not happen within 30 s"). Tom's
diagnosis is an inference from the cool-down sample timing plus an emulator reproduction. There is no phone logcat.

- r30's `researchRun` cleared the keep-screen-on flag at the end of S1.
- The screen timed out during the cool-down. The 10 s samples stretch from 591 s on.
- Android 15 then blocked the restart activity, because the app had no visible window (background-activity-launch
  block, `BAL_BLOCK`).
- The 30 s watchdog fired 49 min late, because the phone was suspended.

## Change

| field | value |
|---|---|
| APK | `vollie-c2-1.31.0+31-research-r31.apk` (HF `r31/vollie-r31.apk`), 20,807,802 bytes |
| sha256 | `fd9fec6ad00dd34354b24ef28409e5ab065aa4c9593bab6583656441cff0ea61` |
| package | `com.insaneutils.vollie.c2`, versionCode 31, cert `c49dde85…`; installs over r30 and keeps r30's chain state |
| app patch | `0014-kit-r31.patch` `05fb13d54be7876289dcd27d67f68cdaf7bbb1f4fbad0c752d2356890ee05625` (on top of 0001 to 0013) |
| native libraries | all 16 other than `libapp.so` byte-identical to r30 (lead re-checked), so R71 §0's runtime table stands |

What 0014 changes:

- The chain keeps the screen on through every segment.
- A restart wakes the screen and shows over the lock screen before it starts the restart activity.
- The watchdog runs on the wall clock and tries up to 3 times.
- A "Resume chain" path continues a chain that stopped at a restart. It never re-runs a finished segment.

No numerics, plan, env or protocol of a block changes.

**How S2 and S3 run.**

1. The phone state is that of R71 §1: power saving off, the list on, charger out, battery ≥ 60 %, Flip open.
2. The phone first rests 30 min or more with the screen off.
3. The owner opens the app and taps **Resume chain**.
4. The chain runs S2 (c1), restarts into stock, and runs S3.

There is no new cool-down before S2; the rest of step 2 stands in for it. S2 and S3 therefore run on another day
than S1. Comparisons between S1 and S2/S3 are reported as cross-day, not as same-sitting comparisons.

**Void additions:** the APK hash differs from the table above; a segment report's `kit` is not `r31`; any
`runtime_check` fails. The scorer accepts kit `r31` reports (one-word change, no rule changed).

**Predictions** for S2 and S3 are R71 §3's, unchanged. So is the restart part of the prediction "the chain finishes
with no stop" (P 0.75): it is already scored as a MISS for r30 and is not rescored.

Nothing else changes.
