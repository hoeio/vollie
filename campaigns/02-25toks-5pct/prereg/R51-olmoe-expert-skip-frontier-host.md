# Pre-registration R51: OLMoE expert-skip frontier (host)

Committed **before** the sweep starts. Do not edit afterwards; if a rule has to change, add `R51-amendment-1.md`.
This is a host round (no phone); it is pre-registered because its result decides where phone rounds go.

- **Date / author:** 2026-11-13, clyde (lead).
- **Campaign / model / file sha256:** campaign 2; OLMoE-1B-7B-0125-Instruct; `olmoe-Mi.gguf`,
  `5abf710b775cd64fd89d74a783c8a7056a409ba795cd401b9dd448ef8e444abb`.
- **Agent models** (full IDs used this round): clyde `claude-fable-5-2`; bonnie `claude-opus-5-5` (memo-03);
  tom `claude-opus-5-5` (memo-02, memo-04, the patch); annie `claude-sonnet-5-5`.
- **Build under test:** llama.cpp `620d2225c` + `patches/llama.cpp-c2/0001-olmoe-expert-skip.patch` only (campaign
  1's `0001` is not applied), CPU, `armv8.2-a+dotprod+fp16`; host Apple M4 Pro; `tools/c2_skip_sweep.sh`, 8 threads.
- **Labels:** every number in this file is *estimated*, or *host* where it names R50 or a 4-chunk smoke check.

## 1. What is tested

Expert skip by router weight on OLMoE, a declared numerics change, off unless a `VOLLIE_SKIP_*` variable is set
([memo-04](../reports/memo-04-olmoe-expert-skip.md)). Rules: `SUM=t` (skip if v < t‰ of the sum of the 8 selected
weights), `MASS=p` (keep by rank until p‰ of that sum is covered); no renormalisation; `MINK=1`.

Settings, fixed here: `stock SUM=45 SUM=55 SUM=65 SUM=80 MASS=950 MASS=925 MASS=900 MASS=850`. Full wikitext-2
test, ctx 512, batch 512, all 560 chunks. Further settings may be added afterwards only **between** two measured
points that bracket the budget, to locate the crossing; they are reported as added.

## 2. Budget accounting

| item | PPL | % over stock |
|---|---|---|
| stock served file (R50, *host*) | 11.7146 | 0 |
| config before this round | 11.7146 | 0 |
| config under test | to be measured per setting | |
| budget (stock × 1.05) | 12.3003 | +5.00 |
| hard ceiling (F16 10.9420 × 1.20) | 13.1304 | +12.09 |

Identity check required: the sweep's `stock` row (patched build, no knob) must print PPL = 11.7146 ± 0.09601, the
same digits as R50. If it does not, the round is void and the patch goes back to Tom. Per-chunk identity at
`-ub 1`, `-ub 32` and `-b 512` was shown on 4 chunks only (memo-04).

## 3. Prediction

Quantity: **f\***, the largest fraction of expert bytes skipped per token (the patch's counter) among the swept
settings whose full-set PPL is ≤ 12.3003.

- Central **10 %**, band **6 % to 15 %**.
- Basis: 4-chunk smoke values (*host*, not results): `SUM=65` skipped 10.5 % at +4.4 %, `MASS=900` 13.8 % at
  +6.9 %; one published prior for a fixed top-k cut (arXiv 2509.23012: 8→7 experts +1.6 %, 8→6 +6.1 %, not GGUF).
- Second prediction: at equal bytes skipped, `SUM` gives lower PPL than `MASS` (band: by 0.5 to 3 % of stock PPL at
  about 12 % skipped).
- Resolution: each PPL carries about ± 0.096 (± 0.8 %), but settings are compared on the same text, so the order
  of settings is far better resolved than that; f\* is resolved to the spacing of the settings (about 3 points).

In milliseconds (memo-03, *estimated*, not part of the scoring): 1 % of expert bytes is about 0.29 ms on the
phone; OLMoE needs about 16 % (0 to 41 %) to reach 38.0 ms steady.

## 4. Pass rule and kill rule

- **Prediction scored as hit** if 6 % ≤ f\* ≤ 15 %; otherwise a miss, written as such.
- **Lever sufficient at the central estimate:** f\* ≥ 16 %.
- **Lever closed as a main lever:** f\* < 5 % (worth less than about 1.5 ms, *estimated*). OLMoE then reaches
  25 tok/s only if the device baseline lands at the fast end of memo-03's band.
- **Unresolved:** the budget crossing lies between two settings more than 5 points of bytes apart; add settings
  between them as allowed in §1.

No setting is served on the strength of this round alone: `en.txt` is checked for the chosen setting first
(sanity, not the gate; campaign 1 saw skip move `en.txt` more than wikitext-2), and a served configuration is
fixed in a device pre-registration.

## 5. Claim rule

Not a goal claim.

## 6. Hygiene counts

Every log: 560 chunks, `n_seq=1`, and a `VOLLIE_SKIP` counter line; the `stock` row's counter must show 0 pairs
skipped. A log without `Final estimate` gives no result.

## 7. What a miss means

- f\* ≥ 16 %: write the device pre-registration for the OLMoE baseline and then the skip package.
- 5 % ≤ f\* < 16 %: the device baseline of `olmoe-Mi` on the exact stack is the next round (memo-03 predicts
  43.5 ms, band 39.1 to 50.4); it decides how much skip is needed. In parallel, price what else can close the
  gap (untied two-stage head, a Q4_0-expert file if the owner allows a second file).
- f\* < 5 %: say so to the owner with the numbers before any phone round is spent on OLMoE.
