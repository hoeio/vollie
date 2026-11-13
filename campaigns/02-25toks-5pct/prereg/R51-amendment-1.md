# R51 amendment 1: the same frontier on the second OLMoE file

- **Date / author:** 2026-11-13, clyde. Written and committed before any sweep log of either file exists
  (the R51 sweep is queued behind the LFM2.5 reference run and has not started).
- **Why:** the owner added the publisher's Q4_0 as a second OLMoE served file (CAMPAIGN.md, owner decision 9).
  R51 itself is unchanged.
- **Agent models:** as in R51.

## What is added

The R51 sweep, same build, same settings (`stock SUM=45 SUM=55 SUM=65 SUM=80 MASS=950 MASS=925 MASS=900 MASS=850`),
same text and flags, on `OLMoE-1B-7B-0125-Instruct-Q4_0.gguf`
(`df43bd738dfdcbe889c5f854120c271e9e53e2c624f212ac2523f8bc11ae9bf3`). It runs after the olmoe-Mi sweep.

| item | PPL | % over stock |
|---|---|---|
| stock served file (R50, *host*) | 11.3784 | 0 |
| budget (stock × 1.05) | 11.9473 | +5.00 |
| hard ceiling (F16 10.9420 × 1.20) | 13.1304 | +15.40 |

Identity check: the `stock` row must print 11.3784 ± 0.09306 (R50's digits), counter 0 pairs skipped; otherwise
this part is void.

## Prediction

f\*(Q4_0), defined as in R51 against this file's budget 11.9473: central **11 %**, band **6 % to 16 %**. Reason
(*estimated*): the same router decides, so the frontier should have the same shape; Q4_0 experts carry less
quantization error, which may leave slightly more room. Second prediction: at each setting the fraction of
bytes skipped differs from olmoe-Mi's by less than 2 points (the router tensors are F32 in both files, but the
hidden states differ).

## Rules

Scored as in R51 §4 (hit if 6 % ≤ f\* ≤ 16 % here). The thresholds for "sufficient" (16 %) and "closed" (5 %)
are **not** carried over in milliseconds: memo-03's 0.29 ms per 1 % is for Q3_K experts, and no rate exists for
Q4_0 experts. For this file the frontier is reported in bytes only; its value in milliseconds waits for a
device measurement of the Q4_0 expert op.
