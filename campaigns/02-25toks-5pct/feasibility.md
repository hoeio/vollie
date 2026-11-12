# Feasibility: where 25 tok/s stands before any measurement

Everything here is *estimated* unless it says *device* or *host* and names a campaign-1 entry. It exists to pick
experiments, not to claim results.

## 1. Why the model must be a sparse one

Decode reads (and computes with) the active weights once per token. A dense 7B model at 3 bits is about 3 GB
per token, so 25 tok/s needs about 75 GB/s. The device measured about 19 GB/s on dense weights (campaign 1,
`findings.md`); the SoC's nominal LPDDR5 peak is about 51 GB/s (recalled, not checked here), which is also
below 75. An int8 dot-product ceiling for three big cores puts a 7B dense model near 10 tok/s even before
bandwidth. Speculative decoding does not help: campaign 1 found the CPU compute-bound at any batch above one
(R20), and early exit cost +2.5 % perplexity for the cheapest layer (R21).

So dense 7B is not "hard", it is excluded by the arithmetic. Only the ways to read fewer bytes per token remain
(activation skipping, lower bits), and every known one either failed or exceeded the budget (see §3).

## 2. The target in milliseconds

| quantity | value |
|---|---|
| campaign-1 result | median 20.51 tok/s printed (*device*, R49) |
| steady token behind it | about 45.5 ms graph + 1.05 ms outside (*device*, R49) |
| 25 tok/s printed | about 38 ms steady (*estimated*: 32 tokens + ramp + one unused final decode) |
| needed | about −8.5 ms, −18 % |

## 3. What the +5 % budget can buy on Granite

| lever | estimated saving | basis |
|---|---|---|
| expert skip up to the cap (117 → about 138; 130 is +3.67 %, 140 is +5.25 %, *host* full set, R24/R25 correction) | −3 to −4 ms | extrapolated from the R25 costing (115: 54.2, 130: 51.5, 140: 49.9 ms steady); uses the **whole** budget |
| router F32 → F16 | −0.3 ms | R42/R43 host screen |
| remaining exact crumbs (barrier waits) | −0.5 to −1 ms | R43 text; the fused expert op already took −0.8 ms |
| drop the unused final decode from the metric | +3 % printed | changes the benchmark's definition: must be reported separately |
| dense requantisation at load (Q4_0 → lower bits, runtime artefact) | up to about −5 ms for 24 % fewer dense bytes | **unpriced**; double quantisation error is expected to exceed the budget; needs a host PPL screen |
| low-rank head screen, shared-expert pruning, Mamba-2 head pruning | none | closed or not worth it in R43 |

Sum of the priced levers: about −4 to −5 ms, so roughly 23 tok/s at best on Granite, with large uncertainty.
25 tok/s needs about −4 ms more from something not yet priced. Expert skip also moves `en.txt` much more than
wikitext-2 (skip 130: +6.1 % versus +3.67 %), which matters for the sanity check.

## 4. Model comparison by active bytes (very rough)

Scaling Granite's measured breakdown by active bytes and op count. Not a measurement.

| model | total / active | layers | expected steady token | 25 tok/s |
|---|---|---|---|---|
| Granite-4.0-H-Tiny | 7B / about 1B | 40 (36 Mamba-2 + 4 attention) | about 44 ms with the whole budget | unlikely |
| OLMoE-1B-7B | 6.9B / 1.3B; 8 of 64 experts | 16 | about 26 to 34 ms | plausible |
| LFM2.5-8B-A1B | 8.3B / 1.5B; 128K vocabulary | 24 (18 conv + 6 attention) | about 40 to 55 ms | hard |
| Ling-mini-2.0 | 16.3B / 1.4B (789M non-embedding); 8 of 256 experts | 20 | compute small; memory about 6 to 7 GB at 3 bits | open (memory) |

OLMoE's advantage is structural: 16 layers instead of 40 means far fewer small ops and syncs per token, which
was half of the token on this phone before campaign 1's first fix. Its risks are an older model and small
experts (ff 1,024) that may lose more to 3-bit quantisation than Granite's; both are unmeasured.

## 5. What would change this picture

- A dense requantisation or activation-skipping scheme that stays inside +5 % and removes about a quarter of
  the dense bytes.
- A candidate whose active bytes per token are well below Granite's (OLMoE is the first to test).
- A fix for cold-start page residency (campaign 1's open item) only helps the first run, not the median of
  warm runs.
