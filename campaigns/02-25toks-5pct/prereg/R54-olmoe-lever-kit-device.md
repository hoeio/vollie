# Pre-registration R54: OLMoE levers on the phone (kit r18)

Committed and pushed **before** the APK is handed to the owner. Changes go into `R54-amendment-1.md`.

- **Date / author:** 2026-11-14, clyde (lead).
- **Campaign / model / file sha256:** campaign 2; OLMoE-1B-7B-0125-Instruct; `olmoe-Mi.gguf`,
  `5abf710b775cd64fd89d74a783c8a7056a409ba795cd401b9dd448ef8e444abb`.
- **Agent models** (full IDs): clyde `claude-fable-5-2`; bonnie `claude-opus-5-5` (memo-08, all bands);
  tom `claude-opus-5-5` (patches 0001 to 0009, memo-07, memo-11, kit r18, memo-12); annie `claude-sonnet-5-5`.
- **Build under test:** APK `vollie-c2-1.18.0+18-research-r18.apk`, 17,128,219 bytes, sha256
  `066e80455ce626be8fe3385a05973d28c8f9aa9baf72bb49259fd3a144774cfa`, package `com.insaneutils.vollie.c2`
  (an update over r17; beside the owner's app). Runtime: llama.cpp `620d2225c` + `patches/llama.cpp-c2/0001` to
  `0009` at commit `45ce55d`, every knob unset unless a run sets it; app patches `patches/app-c2/0001`, `0002`.
  The report carries the sha256 of every patch and of both libraries.
- **Device:** the owner's Galaxy Z Flip 3 (Snapdragon 888, 8 GB). Repack on except in group `norepack`.
- **Labels:** every number here is *estimated* (memo-08 §2, §3, §9) unless it says *host*.
- **No phone result exists yet.** R52 (kit r17) has not been run when this is written; these bands come from
  the cost model alone, not from a measured baseline.

## 1. What is tested

The run list of [memo-12](../reports/memo-12-kit-r18.md): `cold` × 1 (unscored); `stock` × 3; `stack` × 5 (all
six exact levers on); `stack_prof` × 2; one in-run ABBA A/B per lever with the rest of the stack on (`ab_tiny`,
`ab_x4`, `ab_x4_kernel`, `ab_nb`, `ab_fuse`, `ab_io`, `ab_sampler`; 3 generations of 400 tokens each, blocks of 8);
`skip` × 5 and `ab_skip` (`MASS=900`); `ab_head2` (`HEAD2=1024`); `package` × 5 (stack + skip + head2);
`norepack` × 2. Official prompt, 32 tokens, greedy, the kit's cool-down gate, app thread pick.

## 2. Budget accounting

| item | PPL (*host*, full set) | % over stock |
|---|---|---|
| stock served file (R50) | 11.7146 | 0 |
| exact stack (0002 to 0007) | 11.7146 by construction: bit-identical on 4 chunks (memo-07); the full-set identity row is part of R53 | 0 |
| stack + skip `MASS=900` (R51; no-repack path 12.1032) | 12.1031 | +3.32 |
| stack + skip + `HEAD2=1024` | **not measured**: R53 is queued, not run | |
| budget (stock × 1.05) | 12.3003 | +5.00 |
| hard ceiling (F16 × 1.20) | 13.1304 | |

Consequence: the `package` and `ab_head2` runs are measurements only. They count toward a served configuration
only if R53 passes `K=1024` as written there. The `skip` group is inside the budget by R51.

Identity required in the report: the reply text of `stock` equals that of `stack` and of every exact-lever A/B
run (the kit's host check showed this); texts within `skip` are equal; texts within `package` are equal.

## 3. Prediction

Official groups, median of the group's hygienic runs, printed `decodeTokensPerSec`; steady ms per token in
brackets, conversion printed ≈ 0.995 × 31,000 / (32 × (T + 0.3) + 12):

| group | steady ms (central, band) | printed tok/s (central, band) |
|---|---|---|
| `stock` | 69.4 (63.5 to 78.9) | 13.8 (12.1 to 15.0) |
| `stack` | 47.5 (43.3 to 50.5) | **20.0 (18.8 to 21.9)** |
| `skip` | 43.6 (38.1 to 47.8) | 21.8 (19.9 to 24.9) |
| `package` | 42.5 (36.9 to 46.6) | **22.3 (20.4 to 25.7)** |

In-run A/B, B − A in ms per token (B = lever on), with the rest of the stack on:

| run | central | band | source |
|---|---|---|---|
| `ab_tiny` | −1.1 | −2.3 to −0.4 | memo-08 §2 |
| `ab_nb` (no-barrier and fused together, see memo-12) | −8.0 | −14.8 to −3.3 | memo-08 §2, NB plus fuse |
| `ab_fuse` | −1.0 | −1.8 to −0.3 | memo-08 §2 |
| `ab_x4` (0 against 2) | −13.1 | −18 to −7.5 | memo-08 §2, both kernel steps |
| `ab_x4_kernel` (1 against 2) | −6.8 | −9 to −4.5 | memo-08 §2 |
| `ab_io` | +0.3 | −1 to +1 | memo-08 §2; read on all blocks |
| `ab_sampler` | −0.3 | −1.0 to 0 | lead's guess, no model term; read on wall time |
| `ab_skip` | −3.9 | −5.2 to −2.7 | 13.39 % of expert bytes at 0.29 ms per 1 % |
| `ab_head2` | −0.6 | −1.4 to −0.2 | memo-11 bytes at the dense rate |

Memo-08 priced the levers in switch-on order; an A/B with the rest of the stack on measures the marginal
effect, which can be smaller for levers that share a cost. A band missed for that reason is still a miss.

Resolution (memo-12 §3): about 72 blocks per side; with campaign 1's clean-block spread, σ about 0.27 to 0.40 ms.

## 4. Pass rule and kill rule

- **Each row of §3 is scored hit or miss on its band.** For an A/B, B − A ± σ over clean blocks; an effect
  predicted below 1 ms with σ > 0.4 is "unresolved", not a miss (campaign 1's rule).
- **Goal reachable (memo-08 §9):** steady token of `stack` ≤ 42.95 ms (from `stack_prof` tokens 4 to 32, and
  consistent with the `stack` median). Then the claim round is pre-registered.
- **Stop line:** steady token of `stack` ≥ 49.0 ms **and** `package` median < 23.0 tok/s. Then the lead writes
  "no on OLMoE at this budget" with the measured numbers and proposes nothing further without new evidence.
- **Between:** the measured per-lever table decides what is worth building next; that plan is written before
  any further round.
- **A lever whose A/B shows B slower than A by more than σ** is taken out of the stack for later rounds.
- **Void:** hash mismatch (model, libraries, patches); `stock` and `stack` texts differ (the stack is then not
  exact on the phone, which is itself a finding to be reported first); fewer than three hygienic runs in `stack`.

## 5. Claim rule

Not a goal claim. If `package` or `skip` reaches a median of 25.0 tok/s here, that is **not** the claim: the
claim is a separate pre-registration, "the median of 10 plain runs of the official benchmark is at least
25.0 tok/s", on a configuration whose full-set PPL is inside the budget.

## 6. Hygiene counts

As R52 §6, per run: clocks recorded, no major faults continuing past the third decode token, gate not timed
out; excluded runs are listed. A/B: warm-up blocks dropped, clean-block count reported per side. Counters: skip
runs show about 13.4 % of expert bytes skipped (band 12.5 to 14.5 % on this prompt; the host value on wikitext
is 13.39 %); `repack_loaded` true except in `norepack`.

## 7. What a miss means

- `stock` outside its band: the cost model is off at its base; every later band is then read relative to the
  measured `stock`, and Bonnie refits before anything else is planned.
- `stack` slower than 50.5 ms: look at `stack_prof`'s class breakdown for the term (experts, attention, head,
  sync) before deciding; the stop line above applies if `package` is also below 23.0.
- `stack` faster than 43.3 ms: memo-08's second correction was wrong in the favourable direction; go to the
  claim pre-registration with skip, and the head only if R53 passes.
