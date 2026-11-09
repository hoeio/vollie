# What did not work

Each of these was measured and closed; the numbers are in the log. Labels follow [RULES.md](../RULES.md):
*device* = measured on the phone, *host* = measured on a laptop (Apple M4), relative or machine-independent
only. "R" numbers are entries in the [R-log](../reports/03_final.md).

- Draft-model speculation and batched verification: 0.48-0.76x the plain rate on the device; the SD888
  CPU is compute-bound at any batch above one (R20). n-gram, EAGLE-3 and MTP drafting lost on the host.
- Early exit: the cheapest layer to drop already costs +2.5 % perplexity (*host*, R21).
- 4 compute threads, and the expert-guess helper on an A55 (R18).
- Expert skip above threshold 118: 119 fails the +2 % budget (9.8087, *host*, R25).
- Anonymous-memory hot-expert cache and Q3_K software prefetch (R23).
- Finer dense mat-mul chunks (R33).
- Static Q4_0 copy of the 4 most-routed layers' experts: over budget, no device effect (R32, R33).
- Q2_K / Q3_K screens for the two-stage head: compute-bound, slower than the stock head (R29).
- Touching expert pages ahead of the kernel (`MMID_TOUCH`): parked (R41).
- Dense structural numerics under a separate +5 % budget: router cast to F16 changed the device's greedy
  text (R47), low-rank head screen and shared-expert pruning failed perplexity (R43).
- OpenCL on the Adreno 660: 0.7-2.9 tok/s, always below the CPU (*device*, see "Didn't" in
  [reports/03_final.md](../reports/03_final.md)).

The list is copied from the README's former "What did not work" section. Only the relative link was adjusted
for this folder; nothing here is new analysis.
