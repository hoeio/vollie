# Clyde, R76 plan (draft, not a pre-registration)

2026-11-26. Lead's reading of Bonnie r35 and Ebony r19, folded into the R76 candidate (kit r37). Nothing here is
scored and nothing is measured. Labels as in the memos; every tok/s below is *estimated*.

## 1. What r35 and r19 change

| item | before | after r35 / r19 | source |
|---|---|---|---|
| DDR follows the capped CPU clock (states B, D) | open | closed: the GPU's own bus vote holds the bus up; the CPU side draws less | r19 §0.1 to 0.3 |
| state-C coupling | 1.40 | 1.50 (1.38 to 1.60); cool 1.36 unchanged | r19 §4.1 |
| flush cost | 124 to 245 µs, unexplained | a step at a GPU idle gap of about 0.3 ms (124 to 147 below, 200 to 245 above, n 3 / 15, no overlap); candidate: inter-frame power collapse | r35 §6, r19 §0.8 |
| first cap at lane power | r33: 12 to 16 s | 17 s (13 to 24) from a passed gate; CPU-only 30 s | r35 §0.1 |
| heat price of a 12.0 claim | not priced | about 0.4 tok/s of margin on Llama-2 at the hot end with a lane, 0.05 without | r35 §0.4 |
| kgsl 200 ms timer | "a per-token kernel is near it" | not armed on the A660; only the 2,000 ms timeout | r19 §0.5 |
| F | per token only | also per block of 4 layers: +0.88 / +0.82 / +0.73 over G at full rate, about 0 at 0.8; P(F beats G by 0.5) 0.18 | r19 §4.2 |
| perf hint, recordable queue | open | perf hint 0 µs (HIGH is the default); recording +0.02 under G | r35 §0.6 to 0.7 |

Order of work unchanged: **G before F.**

## 2. Where the goal stands (lead's verdict input, not changed by this round)

Best priced rung-7 forms (Ebony r19 §4.2, width 1, cool): G 10.59 / 10.46 / 10.14; F per 4 layers 11.51 / 11.35 /
10.96; F per token 11.86 / 11.66 / 11.24. **No priced design reaches 12 on all three, built or not.** P(12 on all
three) about 0.002 (0.0005 to 0.005, Bonnie, judgement). The heat margin (§1) pushes the lane rungs further down
on Llama-2. **No outcome of R76 can change this verdict.** R76 can only decide whether G is worth building as a
result below the goal (about +0.8 to +1.0 tok/s over the lane as probed, cool).

## 3. R76 candidate (kit r37), smallest form

Kit r36's x4s code, no new kernel, no model, 0 bytes (r35 §4.1). Three cells, each after its own gate:

1. **C, cool pass** (about 22 s): triplets cpu | lead split | cpu at shares 0.40 to 0.55, ascending then
   descending; a feeder triplet at 0.25; Ebony's stall arm. Decides the cool split ratio (merged band 0.83,
   0.76 to 0.88) and so whether the lane rungs stand.
2. **G, gap sweep** (5 s): launch plus flush at idle gaps 0, 0.1, 0.2, 0.3, 0.4, 0.5, 1, 5 ms (Ebony T10).
   Decides the power-collapse step and L5's price.
3. **H, heat pass** (45 s at most), last. Optional: it prices the claim's heat margin, which matters only if a
   lane result is claimed.

Printed per arm: a time stamp, launch and flush apart, p10 / p50 / p90, the cpu-5 canary, and Ebony T11's three
lines (`dcvsh_freq_limit`, `isolated`, register-spin canary). Bands, voids and outcome table: r35 §4.2 to 4.3 as
written; the prereg copies them and adds kill lines.

Proposed kill line for the lane's as-built rungs: C1 (best cool ratio) at 0.90 or over. Proposed decision rule
for G: build G only if C1 ≤ 0.86 (Bonnie's outcome row).

Dropped from R76 to keep it small: T5, T5b (bus; closed on paper by r19), T9 (F lifetimes; F is not next), T12
(owner adb read; optional at any time).

## 4. Cost of going ahead

Tom: kit r37 build (app cells, no runtime kernel). Lead: pre-registration with the kit's sha256 values. Owner:
tag `prereg-R76`, HF token for `r37/`, one run of 8 to 12 min with a 30 min cool start. Researchers: none until
the result is in.

## 5. Question to the owner (asked 2026-11-26)

Because no R76 outcome reaches 12 on all three: (a) build and run R76 to decide whether G is worth building as a
below-goal result; (b) stop GPU-lane work and report campaign 3 as not reachable with the numbers above;
(c) realign the goal. The lead recommends (b) or (c); (a) only if a below-goal lane result is wanted.
