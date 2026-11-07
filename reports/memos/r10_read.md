# r10 read (Bonnie, 2026-11-07) — answers to Clyde's four questions on `reports/device/r10_2026-11-07.json`

Labels: *device* = that JSON (thrashing day: avail 1,238 MB, swap in use; ABBA sides 79-83 majflt/token vs r8's 3-11); *host* = r34 disassembly / `gen_kern.py`; *estimated* = arithmetic.
Per-block table + regressions: `python3 -I` over the JSON (23 ABBA blocks + 12 always-on windows, slot µs = `MUL_MAT_ID` compute / (40·kavg)).

## 1. The (i) line: unscorable, not "not decided" — and write the re-run cap

- Point estimate `MUL_MAT_ID` compute B − A **−0.21 ± 6.17 ms** (*device*); by the letter it sits in "falsified > −1.5". But every block of both sides ran at
  79-168 majflt/token; the lowest-fault block (38.6) still costs **246 µs/slot vs the clean-day 116-120** (*device*): the line is 2× fault inflation, and the
  per-block residual after regressing on faults is **44-55 µs/slot** against a predicted −20..−35 — the effect is 0.5-0.7 σ *per block*, 1 σ on the mean.
- The §9.9 fault-match letter (B ≤ A + 10) was met (78.6 / 82.9), so the letter does not rescue it. What does is the **R33 (ii) precedent**: an ABBA whose
  sides sit at > 10 majflt/token on every block was ruled *unscorable*. Apply the same standard: **unscorable**, point estimate logged, thresholds untouched.
  "Not decided" invites "we keep rolling until it passes"; "unscorable by the R33 rule" is a pre-existing standard. Honest only with a cap.
- **Noise clause for r11 (pre-register):** scorable iff (a) all scored blocks on both sides at the same clock (drop other-clock blocks; ≥ 4 blocks/side remain);
  (b) both sides' mean majflt ≤ 10/token; (c) σ(Δ) on the scored line ≤ 1.0 ms (the 1.5 ms pass→falsify gap = 1.5 σ). Otherwise unscorable; **at most two
  re-runs**, after which the pooled estimate is scored whatever σ. The identical-routing always-on windows (below) are *reported* as a cross-check, not scored.

## 2. What the 0.724-0.737 microbench says for the 3-thread Δ at 2,112

- Cache-hot old kernel: 144.2 ns/row ÷ 6 = **24.0 ns per 256-weight row-block** (*device*, cpu5 at 2,112-2,419; `mhz_after` 2,112). In situ the slot is
  116-120 µs / 3,072 row-blocks = **38-39 ns/row-block** (*device*, R8). So the kernel is **≈ 63 % of the slot** (73.7 µs); 42-46 µs is stream / chunk
  atomics / arrival that no kernel touches — my memo's 12-16 µs stall was too small (it agrees with R27's "65-80 % scales with clock"; I had read the high end).
- Estimated Δ: kernel 73.7 × 0.73 = 53.8 µs → slot **96-100 µs**, −18..−22 µs/slot × 164 slots = **−3.0 to −3.6 ms**; if 20-30 % of the 'other' overlaps and
  shrinks too, **−3.6 to −4.5**. **Central −3.6, band −3.0..−4.5** (*estimated*). Pass (≤ −3.0) at the edge; the band's upper half (−5..−6.5) is now unlikely.
  Steady 53.0 − 3.6 = 49.4 → printed fault-free **19.8** (*estimated*): coin flip, slightly below.
- Cross-check (*device*, informative only): the always-on pair has **identical kavg per window** (4.20/4.23/4.19/4.08/4.01/4.01) and identical text = the same
  routing; fault-matched windows: w4 (2,112; majflt 42 vs 44) **−4.0 ms**, w6 (1,881; 35 vs 42) −3.8 → ≈ −2.9 at 2,112. Consistent with −3..−4, not with −6.
- Issue model: ops 377/562 = 0.671 but measured 0.73. If the 69 in-loop `movi` were free (zero-idiom at rename), the old kernel's real count is 493 → 377/493 =
  **0.765**; the measured 0.73 sits between → **"2 pipes, `movi` free"** fits; the `movi` removal bought less than counted. Check: q4x4 8.84 ns/row-block × 2.1-2.4
  GHz = 19-21 cycles for 44 NEON ops = 2.1-2.3 ops/cycle (*device*) — at or just above 2 pipes; old 50-58 cycles for 124 non-`movi` ops = 2.1-2.5. The bench
  clock is unknown to ±13 % (ask Tom to log the core's MHz with the line); at 2,112 both exceed 2/cycle, so either the clock ramped to 2,419 or the A78 dual-
  issues more than I model. Ratios stand either way.

## 3. (iii) scalar, int8, q4x4

- **Scalar is built on the new kernel** (`gen_kern.py scalar` = the generated asm + SWAR planes 0-1 of the next block; *host*). So the ratio that prices "below
  the NEON floor" is **scalar/new = 0.937 (K 1,536) and 1.104 (K 512)** (*device*); MAC-weighted (gate+up 2/3 at K 1,536, down 1/3 at K 512) = **0.99**.
  **Round 11 opens by the letter (0.691 < 0.75 at one shape) and closes on the substance:** the assist is not additive — it shares the unpack work, gains 6 % on
  long rows and loses 10 % on short ones (per-row double-buffer prologue). Recommend: closed; not worth a round.
- **int8/old 0.54 vs 0.40-0.45:** a working-set artefact, not an A78 fact. K 1,536 × 512 rows int8 = **786 KB** (> A78 L2 512 KB → L3) vs Q3_K 338 KB (L2);
  786 KB / 40 µs = **19.7 GB/s** = one core's L3 rate (*device*). The "unpack-free compute floor" was not measured; it is ≈ 30 NEON/row-block → ~0.26 of old
  (*estimated*). Irrelevant in situ (2.7 MB/slot = 118 µs byte floor, as in the memo).
- **old/q4x4 2.72 vs 2.9-3.2:** old is faster cache-hot than my count (the free `movi`), q4x4 is exactly at its 2-pipe count. The 4-bit container's prize is
  unchanged because in situ it is **byte-bound, not issue-bound**: 1.35 MB/slot / 23 GB/s = 59 µs → slot 62-68 vs the new kernel's 96-100 → a further −26..−32
  µs/slot = **−4.3..−5.2 ms** (*estimated*) for +33 % bytes and 3.45 GB anon. Still only for ≥ 12 GB phones; nothing here moves it.

## 4. Missed in the JSON

1. **Bit-identity confirmed on device**: both always-on pairs print identical text for 200 and 32 greedy tokens with identical per-window kavg — `kern=1` is
   bit-identical on real routing; it can join the default without a gate. Worth a results.csv row.
2. **ABBA position bias**: `SSM_SCAN` is +0.34 ± 0.16 on side B in *both* ABBAs (kern and skip), an untouched op → the B side carries ≈ +0.3 ms of systematic
   bias on this kit order. Small, but it means a "0 ± 0.3" clause on untouched lines is at its floor; keep it in mind for sub-ms items.
3. Compute line = thread 0's time (`MUL_MAT_ID` = `mmid_thr[0]` every block); thread 0 runs 5-15 % above threads 1-2 here vs 1-5 % on r8 — fault-driven, not a
   crumb (*device*).
4. The ABBA's last third ran at 1,881; the parser pools it with 2,112 blocks — part of the ±6 σ is clock mixing. The r11 clause (a) removes it.
5. `official 32 · default + kern` 8.8 vs 10.9 tok/s: minflt 100,894 vs 71,665, `mhz_after` 1,804 cluster — the day, not the kernel; do not quote.

**Verdicts:** (i) unscorable (R33 rule), re-run under the clause above, my central −3.6 (band −3.0..−4.5, *estimated*), pass likely but at the edge, 20 printed
≈ 19.8 fault-free; (iii) scalar closed on substance (scalar/new 0.99 MAC-weighted); int8 and q4x4 misses are bench artefacts, the container prize is unchanged.
