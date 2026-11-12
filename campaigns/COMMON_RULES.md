# Rules shared by every campaign

These carry over from [../RULES.md](../RULES.md) (campaign 1), which stays as written. A campaign adds its own
goal, budget and claim rule in its `CAMPAIGN.md`; it does not weaken anything here.

1. **The served model file is frozen.** Speed comes from the runtime: kernels, threading, scheduling, how
   weights are read. A different file or a different model is proposed to the owner first, with its hash.
   A derived artefact that the runtime computes once on the phone from the served file and caches in the app's
   files counts as runtime (owner's answer, R42 amendment 2).
2. **Quality gate.** `llama-perplexity`, wikitext-2-raw test set, context 512, batch 512, all chunks
   (`tools/ppl.sh`). Each model has its own reference values (F16, stock served file) in its card under
   [../models/](../models/). The test does not change mid-campaign. English prose (`en.txt`) is reported as a
   sanity check, not as the gate.
3. **Budgets are written before results and do not move after them.** A request to raise one is declined
   (campaign 1, 2026-11-05). A budget is stated as an absolute PPL per model.
4. **Bit-identical unless declared.** Every runtime change that is not declared a numerics change gives
   identical per-chunk perplexity at `-ub 1`, `-ub 32` and `-b 512`, and is off unless switched on.
5. **Fix it in the runtime, never in the user's phone.** No asking users to reboot, free memory or change
   system settings.
6. **Measure on the device.** Host numbers are relative only; the phone hid the two largest effects of
   campaign 1 (thread-sync cost, page-cache thrash).
7. **Label every number:** *device*, *host* or *estimated*. Never present a host number or an estimate as a phone
   number.
8. **Compare under the same heat.** Wait for the cool-down gate before each timed run, record clocks and page
   faults next to every timing, and score small effects with in-run ABBA A/B.
9. **Predict first.** Prediction, pass rule and kill rule go into `prereg/` and are committed and tagged before
   the run (see [README.md](README.md)). Results are appended to the campaign's R-log, failures and misses
   included. History is not rewritten; corrections are dated notes.
10. **Say so with numbers when the goal looks out of reach.** A rule is never bent to make a number pass.
11. **Claim rule shape.** A claim is "the median of N plain runs of the official benchmark is at least X", fixed
    before the run, never a best run.
