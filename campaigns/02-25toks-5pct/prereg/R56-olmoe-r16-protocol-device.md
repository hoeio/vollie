# Pre-registration R56: OLMoE under campaign 1's kit protocol (kit r20)

Committed and pushed **before** the APK is handed to the owner. Changes go into `R56-amendment-1.md`.

- **Date / author:** 2026-11-14, clyde (lead).
- **Starting point (owner's rule "start from Vollie 1"):** campaign 1's kit r16 and its report
  `reports/device/r16_2026-11-08.json` (the R49 session). This round repeats r16's run list, waits and gate on
  OLMoE with campaign 1's runtime and shipping settings, and compares run k here with run k there.
- **Campaign / model / file sha256:** campaign 2; OLMoE-1B-7B-0125-Instruct; `olmoe-Mi.gguf`,
  `5abf710b775cd64fd89d74a783c8a7056a409ba795cd401b9dd448ef8e444abb`.
- **Agent models** (full IDs): clyde `claude-fable-5-2`; tom `claude-opus-5-5` (memo-17, kit, keep-warm patch);
  bonnie `claude-opus-5-5` (memo-15 fit, used below); annie `claude-sonnet-5-5`.
- **Build under test:** APK `vollie-c2-1.20.0+20-research-r20.apk`, 17,164,163 bytes, sha256
  `7981aaa682a703752c4b82bd63a269c453368c490bc619da1ce6009599883b93`, package `com.insaneutils.vollie.c2` (update
  over r19). Runtime: llama.cpp `620d2225c` + campaign 1's `patches/llama.cpp/0001` to `0003` +
  `patches/llama.cpp-c1-olmoe/0001-keep-warm.patch` (off unless a run sets it); app patches `patches/app-c2/0001`
  to `0004`. The report carries all patch and library hashes.
- **Labels:** *device* where a value names r16 or r19; *estimated* otherwise.

## 1. What is tested

48 runs ([memo-17](../reports/memo-17-kit-r20.md)), configuration `c1_ship` (campaign 1's shipping settings
without the Granite numerics) unless stated:

- **Part A, r16's protocol, 21 runs:** baseline, two 200-token stories, groups 1 to 4 (plain, profiled,
  profiled, plain; 20 s pause before groups 2 and 4), two plain runs. Every run gated as in r16: go below 0.65,
  or after 90 s. The ten plain runs at positions 4, 7, 8, 11, 12, 15, 16, 19, 20, 21 are the R49 claim positions.
- **Part B, r16's group 2 again with keep-warm on** (a helper re-reads one byte per resident expert page every
  10 s while the app is idle).
- **Part C, warm block, no gate wait:** one unscored warm-up and 10 runs back to back; then warm-up + 5 with
  `EXPERT_IO=5`; then warm-up + 5 with every knob off (`c1_off`).

No numerics change anywhere (keep-warm and all rows gave the stock text and identical per-chunk perplexity on
the host). PPL of every configuration: 11.7146 (R50), budget 12.3003.

## 2. Prediction

The phone's state is not ours to set, so predictions are stated per state, read from the report: *cool* =
headroom at the gate ≤ 0.70 and A78 ≥ 2,000 MHz at run end; *hot* = cap 1,555 MHz (as all of r19).

1. **State:** the phone is hot for all of part A with probability 0.6 (r19: 0.774 was the lowest headroom in ten
   idle minutes); if it starts cooling as r16 did, cool from about run 6.
2. **Part A, plain runs after a timed-out gate (hot):** median 6.5 to 9.0 tok/s (r19's `c1_ship`: 7.20), model
   pages before the run 0.4 to 1.4 GB.
3. **Part A if cool (the r16 situation):** the claim-position median is 16 to 21 tok/s, central 19; run k is
   within −25 % to +5 % of r16's run k (Granite) from the first run in which both the clock is 2,112 MHz and at
   least 1.9 GB of the model is resident before the run.
4. **Part B, keep-warm on (hot):** model pages before a run higher than in part A's group 2 by at least 0.5 GB,
   and tok/s higher by 20 to 60 %; headroom at the gate not higher than without it by more than 0.03.
5. **Part C, warm block, `c1_ship` (hot):** median of the 10 runs **11 to 15 tok/s, central 13** (r19's cleanest
   tokens: 73 ms at this cap; campaign 1's Granite in this state, r16 run 1: 12.27 tok/s); model pages before
   each run ≥ 2.0 GB and fewer than 5,000 major faults in at least 7 of the 10 runs. If cool: 17 to 21,
   central 19.
6. **Part C ratios (either state):** `EXPERT_IO=5` within ± 10 % of `c1_ship`; `c1_off` at 55 to 80 % of
   `c1_ship`'s tok/s.

## 3. Pass rule and kill rule

- Each numbered prediction is scored hit or miss on its own, in the state the report shows; a state with fewer
  than three runs is reported, not scored.
- **The comparison with Vollie 1 is made at equal state only:** hot OLMoE against r16's run 1 (12.27 tok/s),
  cool and resident OLMoE against r16's runs 6 to 21 (median of its claim positions 20.51).
- **OLMoE matches Vollie 1** if the part C `c1_ship` median reaches 12.3 tok/s when hot, or the claim-position
  median of part A reaches 19.5 when cool. Then the lever work (skip rule, untied head) goes on top of campaign
  1's runtime and is measured against this baseline.
- **OLMoE is slower than Vollie 1 in the same state** if part C is below 10 tok/s (hot) with the model resident
  (≥ 2.0 GB before the runs); then the cause is sought in the per-token profile before anything else.
- **Nothing here passes or fails 25 tok/s.**
- **Void:** hash mismatch; texts differ between configurations; part A's gate records no headroom readings.

## 4. Hygiene counts

Per run: headroom readings at the gate, wait, timeout flag; clock cap before and after; model pages resident
(mincore) before and after; major and minor faults; reply text. For part B: the keep-warm pass count and its
CPU time. Flags do not remove runs; the state is part of the result.

## 5. What a miss means

- Part C far below 11 (hot) although resident: the per-token profile decides whether attention, the head or
  the expert op is the term; no lever is planned before that.
- Part B shows no gain or more heat: keep-warm is closed as a lever.
- The phone cools and part A follows r16: campaign 1's method carries over to OLMoE unchanged, and the next
  round is the lever comparison on this runtime.
