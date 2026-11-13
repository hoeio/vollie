# Memo 03: lever price list per model, in bytes and milliseconds

Agent model: claude-opus-5-5

Date 2026-11-13. Author: Bonnie (research). **Nothing in this memo is a measurement of a campaign-2 file.** Every
number is *estimated* unless it says *device* (campaign 1, with its R-number) or *host*. The only data I read
myself are campaign 1's device JSON `reports/device/r16_2026-11-08.json` (R49's session), with my own short code;
numbers from that reading are labelled *device, r16 JSON, my reading* and are not a new measurement. No job was run
apart from that read and a small arithmetic script.

Inputs: STATUS.md, knowledge.md, feasibility.md, memo-02, the two model cards, R25, R27, R31-R33, R42, R43, R47-R49
in `reports/03_final.md`, and `reports/05_compute_less.md` §7.3, §8, §9.

## 0. The short version

| model | best package inside +5 % (all *estimated*) | steady ms | printed tok/s | gap to 38.0 ms |
|---|---|---|---|---|
| Granite-Mi (stock weights) | exact stack + head2 + skip up to the budget + waits | 43.2 (40.6-46.4) | 22.3 (20.5-23.9) | **+5.2 (+2.6 to +8.4) short** |
| OLMoE-Mi | exact stack + untied head2 + waits, **no skip** | 42.2 (36.6-49.4) | 22.8 (19.3-26.5) | +4.2 (−1.4 to +11.4) |
| OLMoE-Mi | the same + expert skip of 20 % of expert bytes | 37.0 (30.5-44.8) | 26.0 (21.3-31.7) | −1.0 (−7.5 to +6.8) |
| LFM2.5 mix (no file chosen) | exact stack + head2 + skip 30 % | 47.3 (38.0-57.7) | 20.3 (16.6-25.5) | +9.3 (0 to +19.7) short |

- **feasibility.md §4 was too optimistic for OLMoE** (it said 26 to 34 ms). The expert product is compute-bound.
  It runs at about 0.083 ms per MB of Q3_K (*device*-derived, R49), twice the dense byte rate. OLMoE's 346 MB of
  experts per token is therefore about 29 ms by itself. Its exact-stack token is about 43.5 ms, close to Granite's.
- **OLMoE reaches 25 if the skip rule can remove about 16 % of the expert bytes inside the budget** (band 0 to
  41 %, set by how uncertain the stack is). The only published OLMoE prior is a fixed top-k cut, from k = 8 to 7
  (12.5 % of bytes): wikitext PPL +1.6 %. A cut to 6 experts (25 %) gave +6.1 % (MoE-PHDS, see Sources). The model is
  **undecidable before a host frontier and a device baseline, and plausible at the central value.**
- **Granite misses by about 5 ms with the whole budget spent.** By my reading of the stop rule (a shortfall of
  about 6 ms or more is a "no") it sits just inside the line at the central value. No priced lever is left to
  close the gap. The only unpriced one is the dense requant at load, and it would have to share the same +5 %.
- **LFM2.5 fails the stop rule.** It needs about 60 % of its expert bytes skipped with k = 4, and its head alone is
  9 ms.
- **The most valuable next measurement:** a device baseline of `olmoe-Mi` with the profiled official 32 and the kit's
  `cpu-bench` at OLMoE shapes (§6).

## 1. The cost model

### 1.1 Terms

At 2,112 MHz on the fast cores, 3 threads, fault-free tokens, the official benchmark's context:

T_steady = Σ dense bytes × r_dense + head term + expert bytes × r_exp + n_ops × w_op + n_barriers × s_bar
+ attention + t_outside

| term | what it is | value | source |
|---|---|---|---|
| r_dense | dense Q4_0 `MUL_MAT`, bandwidth-bound | 0.0398 to 0.0411 ms/MB | 0.0411 *estimated* R25; 0.0398 fitted to R49 below |
| r_head2 | head2 screen + exact rows | 0.041 ms/MB (byte rate) | *device* R49: 3.66 ms for 89.3 MB |
| stock head (Q6_K) | plain Q6_K head | 0.042 ms/MB central (0.041 to 0.055) | 0.042 = R49 head2 3.66 + R31 head2 Δ 1.65 over 126.4 MB; 0.055 = R29's 6.5-7.0 ms (*device*) |
| router (F32) | 40 small `MUL_MAT`s | 0.79 ms for 15.7 MB (about 20 µs per op) | *device*, r16 JSON, my reading |
| r_exp | Q3_K expert `MUL_MAT_ID`, fused gate+up, in-register kernel. **Compute-bound, scales as 1/clock** (R27) | 0.083 ms/MB (band ±10 % for another shape: 0.075 to 0.091) | fitted to R49 below; per-row-block cost does not depend on K |
| w_op | wait at the end of a big op | 6.5 µs per dense `MUL_MAT`, 6.6 µs per `MUL_MAT_ID` | *device*, r16 JSON: 1.71 ms / 265 and 0.53 ms / 80 |
| s_bar | everything else (small ops, `CPY`, `SSM_CONV`, norms, Granite's 4 FA ops, their waits), lumped per barrier | 7.7 µs per barrier | fitted: 5.15 ms / 666 barriers |
| attention | flash attention over the KV cache | KV bytes × r_dense + per-op cost | §2.3 |
| t_outside | sampling, detokenisation, app | 1.05 ms (Granite, 100,352 vocabulary, sampler bypass) | *device* R49 |
| page faults | not in the steady model | 0 by definition | R49: fault-free tokens exist only when the model is resident |

Stock-runtime rates (vanilla knobs, R25 vanilla row, *device*). Dense 25.9 ms for 562.5 MB is 0.046 ms/MB. The
experts took 33 to 38 ms of compute for 243 MB at k = 6, i.e. 0.131 to 0.161 ms/MB. Their wait was 11 to 16 ms
(the op's internal barrier). The remainder, 8.9 ms over 1,926 barriers, is about 4.6 µs per barrier.

### 1.2 Calibration on Granite and residual

R49, fault-free decode tokens 4 to 32 of the eight profiled official runs at 2,112 MHz (N = 125; *device*, r16
JSON, my reading). The medians:

| op | compute + wait (ms) |
|---|---|
| token, in graph | 45.81 (R49 text: about 45.5) |
| `MUL_MAT` (265 ops, includes head 3.66 and router 0.79) | 21.20 + 1.71 |
| `MUL_MAT_ID` (80) | 14.22 + 0.53 |
| `SSM_SCAN` (36) | 2.61 + 0.18 |
| `GLU` (76) | 0.20 + 0.01 |
| rest, by difference | 5.15 |

The fit gives four rates:

- **Dense rate.** The non-head, non-router `MUL_MAT` is 16.75 ms for 420.5 MB, so r_dense = 0.0398 ms/MB.
  R25's 0.0411 predicts 17.28: residual −0.5 ms, −3 %.
- **Head2.** 3.66 ms measured against 89.3 × 0.0411 = 3.67 ms predicted: residual 0.
- **Expert rate.** k̄ at threshold 117 is 4.02 to 4.29: *host* 4.24 (R43); *device* 4.21 at 118 and 4.29 at 115
  (R25). That makes 161 to 172 slots × 1.0138 MB, so **r_exp = 0.082 to 0.087 ms/MB, about 84 µs per slot**, against a
  byte floor of 42 µs. This is a fit, not a prediction.
- **Rest.** Fitted by construction.

The calibration is a fit to one state, so it reproduces R49 by construction: the model gives 46.7 (44.4 to 49.0)
for the R49 configuration against 46.55 *device*. **Out-of-sample check.** I added the measured lever deltas back
onto R49 to predict R25's k = 6 control:

- the in-register kernel, +3.8 (R47's −2.67 at about 168 slots, scaled to 240 slots);
- the fused op, +1.0;
- `CONV_RING`, +0.8;
- head2 and the skip back, +6.9.

The prediction is 58.2 ms in graph. The measurement was 62.2 to 63.9 ms (*device* R25). **Residual −4 to −6 ms
(the model is 6 to 9 % fast).** R25 was a swapping day (34 major faults per token), so most of that is plausibly the
fault tax, which the model leaves out by definition. Read every band below as a clean-day band. A faulting day adds
time that the runtime levers here do not remove.

### 1.3 What transfers to OLMoE and what does not

- **Transfers (by mechanism):**
  - r_dense: Q4_0 attention weights, the same kernel and the same bus;
  - r_head2: the byte-rate screen;
  - w_op;
  - t_outside, scaled by vocabulary;
  - the skip coefficient: a skipped slot removes 0.89 to 0.96 of its share of `MUL_MAT_ID` (*device* R25, R27).
- **Transfers with a ±10 % band:** r_exp. OLMoE's experts are Q3_K too, and the cost per 256-weight row-block does
  not depend on K (§9 of `05_compute_less.md`). The shapes differ, though: K = 2,048 / 1,024 against 1,536 / 512.
  There are 1,024 or 2,048 rows per expert, so the chunk balance is different and the activation is 2.3 KB in L1.
- **No OLMoE analogue:**
  - `SSM_SCAN` (2.8 ms on Granite);
  - `SSM_INPLACE` and `CONV_RING` (Granite's −4.5 and −0.8, *device* R25/R27);
  - the shared expert (106.7 MB, 4.4 ms);
  - the Mamba-2 part of the 666 barriers.
- **New on OLMoE with no Granite calibration:** 16 full-MHA attention layers over a growing KV cache (§2.3). The
  node and barrier count per token has to be counted, not derived (memo-02 §3).
- **The page cache.** Granite's experts are 2,595 MB; OLMoE's are 2,768 MB, against about 2.1 GB of page cache on
  this phone plus about 0.3 GB of anonymous memory that OLMoE frees (memo-02 §5). R49 shows fault-free Granite tokens
  with a 2.6 GB expert set, so OLMoE can plausibly be resident. That has not been checked.

### 1.4 Steady ms to printed tok/s

llamadart prints decode tok/s ≈ 31 / `decode_ms`. Here `decode_ms` holds 32 decode graphs, because the 32nd graph's
logits are never used (R48 amendment 1). In r16's fault-free plain runs, `decode_ms` was 1,473 to 1,513 ms against
32 × 46.55 ms (*device*, r16 JSON). So **printed = 31,000 / (32 · T + R)**, where the ramp term R is 0 to 25 ms
(central 10). `decode_tps` is about 0.995 of 31 / `decode_ms`; that difference is ignored. **25 printed needs
T ≤ (1,240 − R) / 32 = 37.97 to 38.75 ms; I use 38.0.** The printed bands below take the low T with R = 0 and the
high T with R = 25.

## 2. Per-model tables

Band columns: **low / central / high** in ms (low = fast). A lever's Δ is written as smallest / central / largest
saving. The PPL charge counts against the model's single +5 % budget and is 0 for bit-identical levers. Every row
is *estimated* unless labelled.

### 2.1 Granite-Mi (stock weights; same types and bytes as campaign 1's d6-e6)

The starting point is the R49 state (*device*, d6-e6 file) with the two numerics levers removed, because in campaign
2 they are charged to the budget.

| step | Δ (ms) | steady after (ms) | printed (tok/s) | PPL charge | basis |
|---|---|---|---|---|---|
| stock runtime | | 92.5 (91.0 in graph + about 1.5 outside) | 10.4 | 0 | *device* R25 (vanilla knobs, d6-e6, 2,112 MHz) |
| exact stack at R25 (tiny-op, `MMID=3`, x4 + ILP, `SSM_INPLACE`, prefetch) | −27 | 63.9 in graph | | 0 | *device* R25 |
| + `CONV_RING`, in-register kernel, fused op, sampler bypass (= R49 without skip and head2) | see note | **53.6 (52.8 / 53.6 / 55.4)** | 18.0 (17.3-18.4) | 0 | R49 in graph 45.65, plus head2 back +1.65 (R31 *device* Δ), plus skip 117 back +5.3 (4.8-6.6), plus outside 1.05 |
| + head2 (Q4_0 screen, 2,048 rows) | −1.5 / −1.65 / −1.8 | 52.0 (51.0-53.9) | 18.5 | +0.01 % on d6-e6 (*host* R31); unmeasured on this file | *device* R31 |
| + expert skip up to the budget (threshold about 138; share of slots 42 / 45 / 47 %) | −7.1 / −8.2 / −9.4 | 43.8 (41.6-46.7) | 22.0 | the rest of the budget, about +4.9 % | campaign-1 curve on other weights: 130 +3.67 %, 140 +5.25 % (*host* R25); k̄ 3.61 / 3.20 (*host*); **must be re-screened on this file** |
| + remaining exact crumbs (waits) | −0.3 / −0.6 / −1.0 | **43.2 (40.6-46.4)** | **22.3 (20.5-23.9)** | 0 | R43 text, unbuilt |
| (router F32 → F16, closed in campaign 1, see §3) | −0.40 / −0.49 / −0.57 | 42.7 | 22.5 | −0.02 % on d6-e6 (*host* R43) | *device* R47: router op −0.485 ± 0.089 |

The R25-to-R49 levers cannot be split row by row on one consistent clock without double counting, so the table
jumps from the R25 measurement to the R49-derived state. The skip saving is smaller than feasibility.md §3 assumed,
because the in-register kernel made each expert slot about 25 % cheaper. Expert time at k = 6 is now about 19.5 ms,
not 26.7 (R25 §7.3).

### 2.2 OLMoE-Mi (590 MB per token: dense 152 + experts 346 + router 8 + head 85)

**Stock runtime (vanilla knobs, the same app configuration):** dense 243.9 MB × 0.046 = 11.2 ms; experts
346 × 0.131 / 0.146 / 0.161 = 45 / 51 / 56; expert waits 5 / 10 / 19 (per op at 112 µs × 48 ops, up to scaled by
bytes); rest 600 / 750 / 900 barriers × 4.6 µs; attention 0.5 / 1.0 / 2.0; outside 0.8 / 1.1 / 1.5. Sum
**77.3 (65.6-93.8) ms, printed 12.5 (10.2-14.8)**. OLMoE on the stock runtime is not much faster than Granite (92.5):
its expert bytes dominate and the stock Q3_K kernel is slow.

**Exact stack, term by term:**

| term | MB per token | ms low / central / high |
|---|---|---|
| attention q/k/v/o Q4_0 | 151.0 | 6.01 / 6.10 / 6.21 |
| router F32 (16 ops, 524 KB each) | 8.4 | 0.35 / 0.40 / 0.45 |
| head `output.weight` Q6_K, stock path (head2 does not engage, memo-02) | 84.5 | 3.47 / 3.55 / 4.68 |
| dense waits (81 `MUL_MAT`) | | 0.40 / 0.53 / 0.70 |
| experts Q3_K, k = 8 (fused gate+up, kernel) | 346 | **25.9 / 28.6 / 31.5** |
| expert waits (32 `MUL_MAT_ID`; high = scaled by bytes) | | 0.20 / 0.40 / 1.10 |
| small ops + syncs (200 / 275 / 350 barriers × 7.7 µs; not counted yet) | | 1.54 / 2.12 / 2.70 |
| attention (§2.3) | 11.6 to 21.9 of KV | 0.40 / 0.90 / 2.00 |
| outside the graph (vocabulary 50,304, sampler bypass) | | 0.75 / 0.90 / 1.05 |
| **total, exact stack (E0)** | **about 612 incl. KV** | **39.1 / 43.5 / 50.4** → printed 22.1 (18.9-24.8) |

**Levers in turn (applicability from memo-02):**

| step | Δ (ms) | steady after (ms) | printed | PPL charge | basis |
|---|---|---|---|---|---|
| stock runtime | | 77.3 (65.6-93.8) | 12.5 | 0 | above |
| tiny-op, `MMID=3`, prefetch (as is) | about −9 to −11 | | | 0 | expert waits 10 → 0.4, rest 3.4 → 2.1 (Granite's −13 of waits, R25, scaled) |
| x4 + ILP + in-register kernel + fused op (as is) | about −22 | | | 0 | 0.146 → 0.083 ms/MB on 346 MB |
| sampler bypass (as is) | about −0.2 | **E0: 43.5 (39.1-50.4)** | **22.1** | 0 | 1.90 → 1.13 on 100,352 entries (*device* R33), halved for 50,304 |
| `SSM_INPLACE`, `CONV_RING`, shared-expert items | n/a | | | | memo-02 |
| head2 for an untied, repacked head (**new code**, 30-40 lines, option A) | −0.9 / −1.0 / −2.1 | 42.5 (37.0-49.5) | 22.6 | 0 to +0.2 %, central +0.05 % (unmeasured; 2,048 rows = 4.1 % of the vocabulary, Granite's 2.0 %) | head 84.5 MB → 58.0 screen + 3.4 exact rows = 61.4 MB; Δ by R31's measured 0.0445 ms per MB saved, up to the R29 rate |
| waits crumbs | −0.1 / −0.25 / −0.4 | **42.2 (36.6-49.4)** | **22.8 (19.3-26.5)** | 0 | Granite's −0.3 to −1.0 × 275 / 666 barriers |
| expert skip, fraction f of expert bytes (**small code**, a p/S rule per memo-02) | −0.925 · f · 28.6 (coefficient 0.89-0.96) | see below | | **unknown: the host frontier** | coefficient *device* R25/R27 |
| (router F16, closed in campaign 1) | −0.15 / −0.26 / −0.30 | −0.26 on any row | | ≈ 0 (Granite −0.02 %) | 4.2 MB × R47's 0.061 ms/MB |

**Expert skip as a function of f** (every OLMoE expert is 2.70 MB, so the byte fraction equals the slot fraction).
The row above is the starting point (42.2):

| f (expert bytes skipped) | skip Δ (ms) | steady (ms) | printed tok/s | gap to 38.0 |
|---|---|---|---|---|
| 10 % | −2.3 / −2.7 / −3.0 | 39.6 (33.6-47.1) | 24.3 (20.2-28.9) | +1.6 |
| 15 % | −3.5 / −4.0 / −4.5 | 38.3 (32.0-45.9) | 25.1 (20.7-30.2) | +0.3 |
| 20 % | −4.6 / −5.3 / −6.1 | 37.0 (30.5-44.8) | 26.0 (21.3-31.7) | −1.0 |
| 25 % | −5.8 / −6.6 / −7.6 | 35.6 (29.0-43.6) | 27.0 (21.8-33.4) | −2.4 |
| 30 % | −6.9 / −7.9 / −9.1 | 34.3 (27.5-42.4) | 28.0 (22.4-35.2) | −3.7 |
| 35 % | −8.1 / −9.3 / −10.6 | 33.0 (26.0-41.3) | 29.1 (23.0-37.3) | −5.0 |

**f needed for 38.0 ms: 16 % central, 0 % in the fast corner, 41 % in the slow corner.**

**Host frontier that must be measured** (Tom writes the rule; the owner runs it):

- **Run:** PPL against f on the served `olmoe-Mi`. Full wikitext-2 test set (560 chunks, ctx 512, b 512), with the
  patched build and the knob off checked bit-identical first.
- **Points:** f about 10, 15, 20, 25, 30 and 35 %, for both readings of the weight: raw p and p/S (memo-02).
- **Per point, report:** PPL, k̄ on wikitext and on the official prompt (f on the benchmark prompt is what buys
  milliseconds), and the per-layer skip share.
- **References and checks:** a fixed top-7 run as the reference point (the paper's +1.6 % was on another checkpoint
  in another setting); `en.txt` at the edge.
- **Pass:** PPL ≤ 12.3003 minus the head2 allocation.
- **Reading:** if the frontier gives f ≥ 20 % inside the budget, OLMoE passes at the central value. If it gives
  f < 10 %, OLMoE needs the fast corner of the E0 band.

### 2.3 Attention at the benchmark's context (OLMoE)

- **Sizes.** 16 layers × 16 KV heads × 128 = 2,048 per K and per V. At f16 that is 128 KiB per position (card). The
  app serves a q8_0 KV cache with flash attention (R25 footnote), so about 68 KiB per position.
- **Context.** The decode runs at positions 155 to 187, mean about 171. That is 11.6 MB (q8_0) to 21.9 MB (f16) read
  per token: 0.5 to 0.9 ms at the dense byte rate.
- **Compute.** About 11 M MACs, small.
- **Per-op cost.** 16 FA ops against Granite's 4; the cost per op on the device is unknown (R21 §5.4 found a
  split-KV effect at `-ub 1` on the host).
- **Band** 0.4 / 0.9 / 2.0 ms. Granite has no comparable term: its 4 attention layers sit inside the fitted "rest".

### 2.4 LFM2.5-8B-A1B (short; no served file chosen)

The mix, from memo-02's header numbers: experts 416 MB (k = 4, 4.73 MB each), dense Q4_0 238.3 + Q4_1 18.4 MB, tied
Q6_K head 215 MB, router 5.8 MB, 6 GQA and 18 short-conv layers.

| | steady (ms) | printed |
|---|---|---|
| exact stack E0 (experts 34.4, dense 10.4, head 9.0, rest 3.3, outside 1.2, other 1.8) | 60.1 (55.2-68.8) | 16.0 (13.9-17.6) |
| + head2 (screen 147.5 + rows 3.4 MB: −2.6 / −2.85 / −5.7) + waits | 56.9 (48.9-66.0) | 16.9 |
| + skip 20 % / 30 % / 45 % of expert bytes | 50.5 / 47.3 / 42.5 | 19.1 / 20.3 / 22.6 |

25 needs about 60 % of LFM2.5's expert bytes skipped with k = 4. For comparison, Granite at k = 6 bought 45 % with
the whole budget. The publisher Q4_0 file (about 1.0 GB per token) is not priced in detail. Its experts would be
Q4_0, and the same point as §3's file note applies. `CONV_RING`'s generic path and llama.cpp #29002 stay open
(memo-01, memo-02).

## 3. Levers outside the price lists, and why

- **Router F32 → F16.** Closed in campaign 1: device C1 was VOID because the greedy text diverged (R47), and
  knowledge.md §4 lists it. In campaign 2 a declared numerics change is allowed inside the one budget, which may be
  new evidence. It is worth −0.49 (Granite, *device* R47) and −0.26 (OLMoE). I left it out of every package; the
  lead decides whether it reopens.
- **The file has Q4_0 experts (not a runtime lever; the owner decides).** The Q3_K expert op is issue-bound at about
  twice its byte floor (*device*, §9). Q4_0 needs about 0.17 NEON ops per MAC against 0.55 for Q3_K. OLMoE's
  publisher Q4_0 file (R50 row P, PPL 11.3784, *host*) reads 453 MB of experts per token. At 0.047 / 0.055 / 0.070
  ms/MB that is 21.3 / 24.9 / 31.7 ms, against the mix's 25.9 / 28.6 / 31.5: **Δ −4.7 / −3.7 / +0.2 ms.** The rate for
  the plain, non-repacked Q4_0 `MUL_MAT_ID` on the A78 is unmeasured. R33's repacked 4-layer copy was unscorable on
  the device (−0.26 ± 4.0). The file also maps 3.6 GB of experts against about 2.4 GB of cache, so faults are likely.
  It is listed only so that the lead can see it. R50's rule chose the file by PPL, and changing the file is the
  owner's decision.
- **Not re-proposed (closed list):** the low-rank head, shared-expert pruning, a vocabulary subset, dynamic channel
  skipping, sub-4-bit dense containers, speculative decoding, early exit (knowledge.md §4).
- **Dense requantisation at load (unpriced, knowledge.md §5).** On Granite, 24 % fewer dense bytes is about −4.3 ms
  at 0.0398 ms/MB. It shares the budget with skip, so it needs a joint frontier. On OLMoE the dense part is only
  6.1 ms, so it is worth at most about −1.5 ms there.

## 4. The stop rule

I read STATUS item 6 as: **stop if the best package inside the budget is still ≥ about 6 ms above the 38.0 ms
target.** That reading is mine; the lead decides.

| model | shortfall, central (band) | reading |
|---|---|---|
| Granite-Mi | +5.2 ms (+2.6 to +8.4); +4.7 with the router cast | **Borderline: inside the 6 ms line at the central value, outside it in the slow third of the band.** Every priced lever is in and the whole budget is spent on skip. Only an unpriced dense requant (sharing the same budget) could close the gap. I would not spend a phone round on Granite for 25 before a joint skip + requant host frontier exists |
| OLMoE-Mi | +4.2 without skip; −1.0 at f = 20 %; f needed 16 % (0 to 41 %) | **Passes at the central value; undecidable until the host frontier (the f inside 12.3003) and a device baseline (the E0 band) exist** |
| LFM2.5 | +9.3 at f = 30 %; f needed about 60 % | **Fails** (*estimated*, no file chosen) |

## 5. The single most valuable next measurement

**A device baseline of `olmoe-Mi` on the campaign-1 runtime** (all default-on, bit-identical levers, after their host
identity check on `olmoe`, memo-02 §6.1), recorded with:

- the profiled official 32 (per-token op table, nodes, barriers, majflt, clock);
- the kit's `cpu-bench` at OLMoE's shapes (K 2,048 and 1,024; a 5-line change, memo-02);
- if cheap, a plain Q4_0 `vec_dot` row at the same shapes.

Why this one:

- **Width.** The E0 band (39 to 50 ms) is wider in milliseconds than the plausible skip supply (f of 15 to 35 % is
  about ±2.6 ms).
- **What it settles.** The run fixes r_exp on the real shapes (28.6 ± 2.8 ms), the attention and barrier terms, the
  head path, and whether 2.77 GB of experts stays resident.
- **What it is checked against.** Predicted steady: 43.5 (39.1-50.4) ms at 2,112 MHz on fault-free tokens. Printed:
  22.1 (18.9-24.8). Falsified if the fault-free steady is above 52 or below 37.
- **In parallel.** The host frontier of §2.2 needs no phone and should run beside it.
- **Same-day anchor.** A Granite-Mi baseline in the same session re-anchors the calibration. Predicted 53.6
  (52.8-55.4), exact stack only. Skip and head2 stay off, because their PPL on this file is unmeasured.

## 6. Assumptions and what would falsify them

| assumption | falsified by |
|---|---|
| The Q3_K expert cost per MB transfers to OLMoE's shapes within ±10 % | the device `MUL_MAT_ID` per slot outside 1.6 to 2.0 ms per layer-token (8 slots; 28.6 / 16 central), or `cpu-bench` ns per row-block at K 2,048 differing > 10 % from K 1,536 |
| The expert op stays compute-bound (time ∝ 1/clock) | an OLMoE `MUL_MAT_ID` that does not move with a clock change (as R27 did on Granite) |
| The skip coefficient is 0.89 to 0.96 | an ABBA skip Δ per skipped share outside that band |
| OLMoE's barriers per token are 200 to 350 at 7.7 µs | the counted `barriers=` field outside that range, or rest > 3 ms |
| Attention is 0.4 to 2.0 ms at about 171 positions | profiled FA time outside that band |
| The head runs at 0.041 to 0.055 ms/MB; the untied head2 saves 0.9 to 2.1 ms | the `head=` field on OLMoE outside 3.4 to 4.7 ms |
| OLMoE is resident at steady state (2.77 GB of experts, about 2.4 GB of cache) | majflt per token > 10 on warm, gated runs; then the steady model does not apply and residency becomes the first item |
| The fixed top-k prior bounds the threshold rule from below (a threshold skips the least useful slots first) | a host frontier where p/S skipping at 12.5 % costs more than the fixed top-7 run |
| Granite's skip curve on stock weights resembles d6-e6's (+5 % at about 45 % share) | the re-screen at 115 to 140 on `granite-Mi` |
| Clean-day bands: R25's out-of-sample residual (−4 to −6 ms) is the fault tax | a fault-free device token on either model that is 4 ms or more above the band's high end |

## 7. Not verified

1. Every OLMoE and LFM2.5 number: no device or host timing of either model exists.
2. k̄ at threshold 117 on the official prompt; I used R43's host story k̄ and R25's device k̄.
3. Whether the app keeps OLMoE's dense weights and head repacked or on the mmap (R25 footnote against memo-02 §5).
   This moves the head term within its band.
4. Whether the head field is inside the r16 `MUL_MAT` total (I assumed so; the bytes agree).
5. Node, barrier and FA counts for OLMoE (memo-02 §3). Whether flash attention is on in the phone build for OLMoE.
6. The MoE-PHDS numbers are for OLMoE-1B-7B-0125 base after SFT at k = 8 (their "Oracle" row), not our instruct
   file, and not at Q3_K. Their percentages are computed from their Table 11.
7. The ramp term R = 0 to 25 ms comes from Granite; OLMoE's ramp is unmeasured.
8. The outside-graph time on a 50,304 vocabulary.
9. The rate of the plain Q4_0 `MUL_MAT_ID` on the A78 (§3 file note).

## Sources

- MoE-PHDS, "One MoE checkpoint for flexible runtime sparsity", arXiv 2509.23012, Table 11: OLMoE-1B-7B-0125 with k
  reduced at inference. WikiText PPL 9.407 (k 8), 9.557 (k 7), 9.985 (k 6), 10.975 (k 5), 13.032 (k 4).
  <https://arxiv.org/html/2509.23012>
- The web search (2026-11-13) found no OLMoE study with a threshold-skip PPL curve. NAEE (arXiv 2402.14800) is on
  Mixtral; LExI (arXiv 2509.02753) uses static per-layer k. <https://www.arxiv.org/pdf/2402.14800>,
  <https://arxiv.org/html/2509.02753v1>
