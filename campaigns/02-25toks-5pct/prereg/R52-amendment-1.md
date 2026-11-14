# R52 amendment 1: the baseline may be taken from kit r18

- **Date / author:** 2026-11-14, clyde. Written before any phone report exists.
- **Why:** kit r18 (R54) was finished before the owner ran kit r17, and r18 contains r17's groups (`cold`,
  `stock`, `norepack`) on a runtime whose arithmetic with every knob unset is the same. Each round costs the
  owner an install and a run of up to an hour, so one round is better than two.
- **Rule:** if the owner runs kit r18 without having run r17, R52 is scored on r18's `stock` group (3 runs
  instead of 5) with R52's band unchanged (13.8 tok/s, 12.1 to 15.0), and R52's "void" condition "fewer than
  three `stock` runs meet hygiene" stays as written. If the owner runs both, R52 is scored on r17 as written
  and r18's `stock` group is the anchor of R54 only. The `norepack` group of r18 runs with the exact stack on,
  so R52's second prediction (repack off slower by 0 to 5 ms on the stock runtime) is scored only from r17.
