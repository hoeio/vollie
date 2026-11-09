# What got there: the levers that survived

This is the lever table that used to sit in the README, moved here word for word. The chart in the
[README](../README.md) ([source of its numbers](img/make_charts.py)) is drawn from this table.

The levers that survived, in the served config. Effects are *device* in-run A/Bs in ms per token unless
labelled; minus is faster.

| lever | output | device effect | log |
|---|---|---|---|
| bit-identical runtime stack: sync-skipping around small ops, no-barrier expert op, 4-row Q3_K kernel with independent chains, SSM state in place, expert prefetch | identical | steady token 91.0 → 63.9 ms vs stock settings (ratio 1.42); official 32 9.4 → 12.0 | R25 |
| expert skip at threshold 117 (experts with a small normalised router weight are not computed) | PPL +1.89 % alone (9.7869, *host*) | at threshold 118: token −7.90 ± 2.12, `MUL_MAT_ID` −8.28 ± 1.14 at 2,112 MHz | R25, R31 |
| two-stage output head: a runtime Q4_0 copy screens all rows, the top 2,048 are recomputed exactly | PPL +0.01 % alone (9.6061, *host*); with skip 117: 9.7886 | head op −1.70 ± 0.13 (on skip 118); −1.60 ± 0.14 in the 118 → 117 + head swap | R31 |
| sampler fast path: the repetition penalty visits only the ≤ 64 penalised tokens | identical | outside-graph time 1.90 → 1.13 ms per token (at 1,555 MHz) | R33 |
| in-register Q3_K expert kernel | identical (greedy text identical on the device, four times) | compiled default since R35. **Falsified** under the pre-registered pooled rule in R38: `MUL_MAT_ID` +1.84 ± 4.22 on fault-heavy days; the verdict stands. A new pre-registered experiment **passed** in R47: −3.530 ± 0.117, measured on throttled clocks (1,670 and 1,555 MHz); ≈ −2.67 at 2,112 MHz by clock scaling (*estimated*) | R35, R38, R47 |
| fused expert op: gate, up and SwiGLU in one op, two chunks per claim | identical | `MUL_MAT_ID` + `GLU` −0.802 ± 0.100; compiled default from the r16 build | R47, R48 |
