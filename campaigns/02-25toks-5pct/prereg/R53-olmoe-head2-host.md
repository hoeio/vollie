# Pre-registration R53: untied two-stage head on OLMoE (host)

Committed and pushed before the sweep starts. Changes go into `R53-amendment-1.md`. Host round, no phone.

- **Date / author:** 2026-11-14, clyde (lead).
- **Campaign / model / file sha256:** campaign 2; OLMoE-1B-7B-0125-Instruct; `olmoe-Mi.gguf`,
  `5abf710b775cd64fd89d74a783c8a7056a409ba795cd401b9dd448ef8e444abb`.
- **Agent models:** clyde `claude-fable-5-2`; tom `claude-opus-5-5` (patches 0008, 0009, memo-11);
  bonnie `claude-opus-5-5` (memo-08); annie `claude-sonnet-5-5`.
- **Build under test:** llama.cpp `620d2225c` + `patches/llama.cpp-c2/0001` to `0009`, CPU,
  `armv8.2-a+dotprod+fp16`, host Apple M4 Pro, `tools/c2_head2_sweep.sh`, 8 threads, repack **on** (the path the
  kit uses for OLMoE and the path of R50/R51).
- **Labels:** *host* where a value names R50/R51 or a 4-chunk smoke check, *estimated* otherwise.

## 1. What is tested

`VOLLIE_HEAD2=K` ([memo-11](../reports/memo-11-kit-runtime-and-head.md)): the output head is screened and only
K candidate rows (with repack on: every touched 8-row group) are computed exactly; a declared numerics change,
off by default, charged to the one +5 % budget. Settings, fixed here: `stock K=1024 K=2048 MASS=900
K=1024,MASS=900 K=2048,MASS=900`. Full wikitext-2 test, ctx 512, batch 512, 560 chunks. `K=4096` is left out:
with repack on it reads more head bytes than the stock head (memo-11).

## 2. Budget accounting

| item | PPL | % over stock |
|---|---|---|
| stock served file (R50) | 11.7146 | 0 |
| skip `MASS=900` alone (R51) | 12.1031 | +3.32 |
| budget (stock × 1.05) | 12.3003 | +5.00 |
| hard ceiling (F16 × 1.20) | 13.1304 | |

Two identity checks come with this sweep, because its build is the **full** series: the `stock` row (no knob)
must print 11.7146 ± 0.09601, and the `MASS=900` row must print 12.1031 ± 0.10045, R51's digits. Either
failing voids the round and sends the series back to Tom; it would also mean that 0002 to 0009 are not
identical with their knobs unset on the full set.

## 3. Prediction

- `K=1024` and `K=2048` alone: within ± 0.10 % of stock (central +0.01 %). Basis: the 4-chunk smoke check moved
  −0.10 % to +0.02 % and the true top-1 was always among the screened rows; campaign 1's tied head added +0.01 %.
- With `MASS=900`: +3.32 % ± 0.12 points for both K (the two changes add, no interaction).
- Head bytes per token (*estimated*, memo-11): K=1024 about 69.7 MB against 84.5 MB stock, so the lever is small,
  about 0.6 ms on the phone at the dense byte rate.

## 4. Pass rule and kill rule

- **Pass for a K:** alone ≤ +0.15 % and with `MASS=900` ≤ 12.3003. The smaller passing K is the one used in the
  lever kit.
- **Kill for a K:** alone > +0.50 %, or with `MASS=900` > 12.3003. If both K are killed the head lever is closed
  for OLMoE.
- **Between** (+0.15 % to +0.50 % alone, inside the budget with skip): usable, but the head's cost is then
  weighed against its saving in the device A/B before it is served.
- Prediction scored hit only if all four head rows are inside the bands of §3.

## 5. Claim rule

Not a goal claim.

## 6. Hygiene counts

Every log: 560 chunks, `n_seq=1`, a `CPU_REPACK` buffer line (repack on); head rows carry the head2 counter line
and skip rows the skip counter line.

## 7. What a miss means

A pass puts `HEAD2=K` into the lever kit as its own A/B and into the package run. A kill removes it; the
package is then stack plus skip only, which memo-08 puts about 0.6 to 1.4 ms slower (*estimated*).
