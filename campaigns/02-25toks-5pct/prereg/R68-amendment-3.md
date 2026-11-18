# R68 amendment 3: the last nine readings, fixed before any data is seen

Date: 2026-11-18. Author: clyde (lead). Written before the lead or any agent has seen R68 data. Its source is the nine
items that Bonnie's scorer (`reports/memo-33-work/score_r68.py`) still prints as `AMBIGUOUS` after amendment 2.
Each item is decided as follows. The scorer's current reading is adopted except where a different reading is
stated. No prediction or threshold changes.

1. **A2, "`wait_lost_mb` not higher with T".** The paired mean over quads (T minus t) must be ≤ 0. This is a
   reporting condition for "stays on". It adds no pass route.
2. **A1, `RssFile` fallback.** The value is in kB, as `/proc/<pid>/status` reports it. It is converted as
   kB × 1024 / 10^6 to match `want_mb`.
3. **A8.** The rule for combining two sessions applies to nice as it does to touch.
4. **Nice.** If the run-queue route passes and the tok/s kill also fires, the kill wins.
5. **B4 mirror.** The mirror control blocks the route only when the mirror itself has at least 3 qualifying pairs.
   With fewer, it is reported as "mirror not testable" and does not block.
6. **D2.** The "fewer than 3 quads" rule is applied after the B3 restore drops.
7. **C5.** If b5's num-arm check fails, b5 is not eligible for the go rule. Its level is still reported.
8. **C6.** "First" means earliest by the report's `started` time. Session B's night is 2026-11-18 18:00 to
   2026-11-19 12:00 KST.
9. **B1.** A tid that is absent from `workers_after` does not count against permission when its `c2-nice` line reads
   −10 and `f0`. The absence is listed.

To be confirmed on the first real file, not here: the layout of the `workers_after` rows, and that the full model
hashes are taken from r27's `expected_sha256`. If either differs, the scorer is corrected to the real layout. That
correction is a code fix, not a change of rule.
