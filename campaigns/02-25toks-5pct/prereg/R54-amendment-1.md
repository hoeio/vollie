# R54 amendment 1: kit r18 is withdrawn before any run

- **Date / author:** 2026-11-14, clyde. No r18 report exists; the owner was asked not to run it.
- **Why:** R52 (kit r17) showed that on this phone, in its state of 2026-11-14, OLMoE's decode is bound by page
  faults and thermal throttling (3.34 tok/s, memo-13), and that the kit's cool-down gate does not wait (a bug that
  r18 shares). Under those conditions R54's bands, which assume a resident model at full clock, cannot be scored
  and its A/B runs would measure flash, not levers.
- **Rule:** R54 as written is not run. If an r18 report arrives anyway, it is recorded as data and not scored
  against R54's bands. The lever questions of R54 move to a new pre-registration for the next kit, written after
  the residency question is answered.
