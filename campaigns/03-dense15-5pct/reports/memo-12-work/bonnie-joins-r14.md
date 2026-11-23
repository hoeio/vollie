# c3 memo-12 work, round 14 (Bonnie): an audit of every factor in the stack, the stack as a ladder by evidence class, and what could stand in for C5

- **Date:** 2026-11-23. **Author:** bonnie, `claude-fable-5-2`. Follows [bonnie-joins-r13.md](bonnie-joins-r13.md). Not
  committed. No verdict here. **Ebony has not reviewed this file.**
- **Labels.** *device*, *host*, *estimated*, *source* as before. Triples are Yi / Llama-2 / DeepSeek.
- **Script:** [bonnie_floor_r14.py](bonnie_floor_r14.py) (`python3 -I -B`; its self-check reproduces stock 6.90 /
  6.21 / 6.22 and the committed `pair` stack 11.89 / 11.88 / 11.43). Every output is *estimated* arithmetic.
- Nothing was measured this round. No web source.

## 0. Answer first

1. **Of the 24 factors the stack uses, 9 are measured on the phone, 3 on the laptop on a subject, 2 on the laptop
   on something else, and 10 are guesses or designs that do not exist yet** (§1). The guesses carry most of the
   distance from 7 to 12.
2. **The floor, built from phone measurements only:** the GPU lane at width 1 with the join cost as R74 measured
   it (452 µs), the measured head, the 8 ms allowance.

   | tok/s, Yi / Llama-2 / DeepSeek | floor (device only) | committed stack at K 300 |
   |---|---|---|
   | `linear` and `pair` (equal at width 1) | **8.34 / 7.76 / 7.64** | 11.59 / 11.54 / 11.13 and 11.89 / 11.88 / 11.43 |
   | `bonnie` (rigid) | 8.19 / 7.59 / 7.48 | 11.14 / 11.05 / 10.68 |

   Stock is 6.90 / 6.21 / 6.22. **So the phone's own numbers support +1.4 to +1.5 tok/s over stock; the other 3.2
   to 4.1 tok/s of the committed stack rest on things not yet measured or not yet built.** Even this floor
   assumes the body stays resident, which is shown for Yi only.
3. **Laptop measurements on a subject add nothing to the floor by themselves.** The only ones are the trained
   head's hit rate and tokens per pass (Yi, Llama-2). They pay only through a width-2 pass, which needs two
   unmeasured kernels.
4. **The ladder** (§2) shows where the tok/s come from. The largest single step is width 2 with `_mc3`:
   +1.7 to +2.6 tok/s, resting on the W-position CPU kernel (laptop timing on random data), on the GPU's
   two-column cost (R75 will measure it) and on DeepSeek's assumed head.
5. **Nothing on the books replaces C5's 7 % of body with comparable evidence** (§3). The candidates that cost no
   join are small: a better head (+0.35 tok/s per +0.05 of top-1), KV in Q8_0 (about +0.2 tok/s on Llama-2 and
   DeepSeek, nothing on Yi), the head on the lane (+0.06 to +0.22). None is measured on a subject.

## 1. The audit: every factor in the stack scripts

Scripts: `memo-10-work/bonnie_tree.py`, `memo-12-work/bonnie_joins.py`, `bonnie_tree_r2.py`,
`bonnie_joins_r5.py` (`rates_linear`), `bonnie_stack_r13.py`. Classes: **D** = measured on the phone;
**Hs** = laptop, on a subject; **Ho** = laptop, on another model or on synthetic data; **G** = guess, model
choice or unbuilt design.

| # | factor | value | class | where from | note |
|---|---|---|---|---|---|
| 1 | body ms per token, stock | 120.6 / 140.6 / 130.8 | D | R72 L4: one layer of the subject's shapes on a cold arena × layers | a probe, not a decode; taken at capped clocks (1,555 to 1,785 MHz) |
| 2 | head ms | 9.04 / 4.40 / 13.88 | D | R72 L7 | head in KleidiAI |
| 3 | read rate R | 29.9 GB/s | D | R72 | – |
| 4 | CPU alone / CPU beside GPU / GPU, one column | 25.64 / 19.77 / 21.53 GB/s | D | R74 `togsum` | one cell, about 10 s |
| 5 | GPU device-time rate beside the CPU; bus sum | 24.6; 46.1 GB/s | D (derived) | R74 `gpu_busy`; fitted | – |
| 6 | join cost K as it is today | 452 µs | D | R74 `x3 rt` | one design (feeder on small cores, `clFinish`) |
| 7 | graph sync inside O | 0.21 ms | D | R72 L6 | – |
| 8 | the CPU keeps its rate beside a GPU that draws almost nothing | – | D | R69, R72 `x1gpu` | the low end of `linear` |
| 9 | head ratio repack ÷ KleidiAI | 0.85 | D, another model | campaign 2 R60 and R62 (OLMoE) | used for the head knob |
| 10 | head top-1 | 0.379 / 0.497 / – | Hs | `tom-head.md` | 32 prompts; DeepSeek's 0.438 is a guess (#20) |
| 11 | tokens per pass at width 2 (gives d) | 1.326 / 1.448 / – | Hs | the same | – |
| 12 | Q4_0-copy head screen is numerically clean | recall 1.000 | Hs | r13 §1.3 (proxy) | Yi and Llama-2 |
| 13 | W-position CPU kernel, cost at two positions | 1.42 to 1.45 × a gemv | Ho | memo-10 E2: assembly count and M4 timing on random data | not built into llama.cpp; exact on random data only |
| 14 | printed = 0.95 × steady | 950 ÷ ms | Ho (D on other models) | campaign 2's kits | not checked on a dense model |
| 15 | hot compute rate C | 40 GB/s | G with a D part | three A78 hot 34.6 (D, R72 L3); the X1's share is a prediction never measured on-core; 40 is "midway" | pair is flat above 40, both lose below |
| 16 | two-term reader (max + 0.25 × min) | – | G (fit) | Ebony's fit through two phone points | sets every bus-against-compute trade |
| 17 | overhead O | 8 ms; 5.2 in stacks | G | an allowance; 5.2 = 0.21 measured + 5 guessed | no dense decode has run on the phone |
| 18 | GPU two-column cost cg | 1.25 to 1.26 | G | r3 model; R75 `x4m` measures it | 2.0 needs no new kernel |
| 19 | join cost a design can reach | 300 µs centre | G | r2 Monte Carlo on R74's breakdown; R75 measures it | hand-off not included |
| 20 | DeepSeek's head: top-1 0.438, d 0.88 | – | G | mean of the other two | its head was never trained |
| 21 | guess-head cost; extra per position | 2.6 ms; 0.5 / 1.5 / 1.5 ms | G (arithmetic on D inputs) | `tom-head.md` §5; memo-10 | – |
| 22 | four fused stages per layer, row split, slack 0.107 × K | – | G (design) | r1 to r3; slack fitted on R74's split cell | no lane exists |
| 23 | `linear` and `pair` between their end points | – | G | r5 §2.5; Ebony r2 | R75's together cell and mixed arm |
| 24 | C5 × 0.93; C7 × 0.5 | – | G | r13 | removed in the rows marked so |

Not factors of the arithmetic, but conditions every row assumes:
- **the body is resident** (D for Yi; not shown for Llama-2 and DeepSeek, R72);
- **the numerics fit the one budget** (Tom's 0006 not reported; the phone's stock path under measurement);
- **the clocks of a claim run** (Ebony r3 and r4: C ≥ 40 with P 0.55).

## 2. The ladder

`bonnie_floor_r14.py`. Each rung adds one thing; its class is the weakest factor it needs. tok/s, Yi / Llama-2 /
DeepSeek. All *estimated* arithmetic.

| rung | what is added | weakest class | `linear` | `pair` | `bonnie` (rigid) |
|---|---|---|---|---|---|
| 0 | stock, CPU alone | D | 6.90 / 6.21 / 6.22 | same | same |
| **1** | **GPU lane, width 1, K 452 as measured, O 8, measured head** | **D** | **8.34 / 7.76 / 7.64** | same | 8.19 / 7.59 / 7.48 |
| 2 | K 300 (a launch design that does not exist; R75) | G | 8.81 / 8.27 / 8.10 | same | 8.61 / 8.04 / 7.90 |
| 3 | head in the repack type | D on another model | 8.93 / 8.32 / 8.24 | same | 8.71 / 8.09 / 8.03 |
| 4 | O 5.2 | G | 9.17 / 8.53 / 8.45 | same | 8.94 / 8.29 / 8.23 |
| 5a | width 2: trained head, W-position CPU kernel, GPU by two launches of the measured kernel (cg 2.0) | Ho (kernel); G for DeepSeek's head | 9.85 / 9.87 / 9.28 | 10.13 / 10.19 / 9.56 | 9.23 / 9.17 / 8.68 |
| 5b | `_mc3` at cg 1.26 instead | G | 10.63 / 10.76 / 10.04 | 10.91 / 11.08 / 10.31 | 10.22 / 10.29 / 9.64 |
| 6 | campaign 2's two-stage head | Hs for numerics; bytes *estimated* | 10.82 / 10.80 / 10.35 | 11.10 / 11.12 / 10.64 | 10.39 / 10.32 / 9.92 |
| 7 | the committed stack: C5 × 0.93, head × 0.5 | G | 11.59 / 11.54 / 11.13 | 11.89 / 11.88 / 11.43 | 11.14 / 11.05 / 10.68 |

Steps, in tok/s (`pair`):

| step | Yi | Llama-2 | DeepSeek | rests on |
|---|---|---|---|---|
| lane at width 1 as measured | +1.44 | +1.55 | +1.42 | R74 |
| K 452 → 300 | +0.47 | +0.51 | +0.46 | a design; R75 |
| head knob | +0.12 | +0.05 | +0.14 | campaign 2's ratio; task H3 |
| O 8 → 5.2 | +0.24 | +0.21 | +0.21 | a guess; R73 |
| width 2 by two launches | +0.96 | +1.66 | +1.11 | W-position kernel; Tom's heads; DeepSeek assumed |
| `_mc3` | +0.78 | +0.89 | +0.75 | R75 `x4m` |
| two-stage head | +0.19 | +0.04 | +0.33 | r13 |
| C5 and C7 at × 0.5 | +0.79 | +0.76 | +0.79 | nothing (r13) |

- **The floor is rung 1: 8.3 / 7.8 / 7.6.** It is R74's own price put through the stage model: R74-result wrote
  7.86 / 7.39 / 7.30 with a flat dispatch term; the stage model lets small stages stay on the CPU, which is
  worth about 0.4.
- **At width 1 a lane cannot reach 12 at any join cost:** with K 0 the rung-1 form gives 10.79 / 9.98 / 9.61
  (`pair`). Width 2 is what the goal needs, and width 2 is where the unmeasured factors are.
- **What turns guesses into measurements, in the order of the ladder:** R75 (K, cg, the bus model), Tom's
  W-position kernel inside llama.cpp, a dense decode (O, clocks: R73), DeepSeek's head, the residency arm (r8).
- With every one of them at its centre the stack is rung 6: **10.8 / 10.8 / 10.4 (`linear`) to 11.1 / 11.1 / 10.6
  (`pair`)**. That is the number I would now quote as "the priced lane", in place of rung 7.

## 3. What could stand in for C5's 7 % of body

C5 was worth 5.3 / 6.2 / 5.7 ms per pass at K 0, about 0.8 tok/s in the committed stack (r13). Candidates, with
what each costs in joins. All *estimated*.

| candidate | gain | joins | evidence | P that it delivers on all three |
|---|---|---|---|---|
| **a better guess head** (more self-distilled rows, soft targets; Ebony r4 §2, r5 §2) | +0.35 tok/s per +0.05 of top-1; her corrected data route gives one to two doublings: Llama-2 +0.04 to +0.08, Yi +0.02 to +0.06 | none | Hs for the trend (Tom's learning curve); nothing for DeepSeek | 0.4 for +0.3 tok/s on Yi and Llama-2; DeepSeek unknown |
| **KV in Q8_0** (`-ctk q8_0 -ctv q8_0`) | the F16 KV at 200 positions is 13 / 105 / 98 MB read per token; Q8_0 halves it: about 0.25 / 2.0 / 1.8 ms, **+0.03 / +0.25 / +0.22 tok/s** near 10.8 | none (attention stays on the CPU) | G: byte arithmetic; memo-08 gave it P 0.8 on numerics, never screened on a subject | 0.5; it helps exactly the two binding subjects and does nothing for Yi |
| the head on the lane (r4 §2) | 1.0 / 0.4 / 1.7 ms after a head screen, 2.2 / 1.0 / 3.5 without: +0.06 to +0.22 tok/s | one per token | G; the Q6_K GPU kernel was never built on the phone | 0.3 |
| per-stage GPU shares | already in the stage model | – | – | – |
| a gated claim run at full clocks (C 45 to 50) | +0.1 to +0.3 in `pair`, more in `linear` | none | D for the cap record; G for C | 0.55 per run (Ebony r4) |
| an FFN row rule that does not need `up` first (rank `gate` rows by a proxy of the input) | up to C5's 7 %, less the proxy's own bytes | none, if the proxy runs on the CPU before the launch | G; the group rule with a proxy was priced on Mistral and Qwen2.5 only; E2 did not generalise | 0.05 |
| `down`-input sparsity (E1, 60 %) | 14 % of the token's bytes on other models | none | Ho (Mistral, Qwen2.5); needs a column kernel, and K1 is parked on the phone | 0.05 |
| per-group levels (memo-08) | +0.03 of body | none | G; needs the column kernel too | 0.05 |

- **Nothing here is both about 7 % of body and supported.** The two with some support (the head, KV) together
  are worth about +0.3 to +0.6 tok/s on Llama-2, less on the others.
- **KV in Q8_0 is the one new line.** It was in memo-08's table and dropped out of the stacks because the body
  was measured without attention. Its bytes sit inside O, which nobody has measured. A dense decode (R73) with
  the KV type as an arm would price it; a host screen of its perplexity on the three subjects costs minutes.
  llama.cpp ties KV quantisation to its flash-attention path; whether the kit's runtime runs that path on the
  phone is a source question I have not read.
- The row rule by a proxy is listed for completeness. It is C4-like (memo-07) and inherits C5's lack of
  evidence.

## 4. Next questions, ranked

1. **KV in Q8_0, read properly:** does the kit's runtime support it (flash attention on the CPU path, the app's
   context options), what does the attention read cost on the phone by R72's rates, and a host screen
   specification for Tom on the three subjects at both batch settings. Mine for the source part.
2. **What is in O?** List the non-matmul work of a dense token (attention over the KV, norms, residual adds,
   RoPE, sampling, graph sync) with bytes and an estimate each, so that R73 has bands for it and the 5 ms guess
   is replaced. Mine.
3. **DeepSeek's head** (Tom's H2 dump, then training): it is the binding subject in every row and the only one
   whose width-2 gain is assumed.
4. **Re-read the ladder after R75:** K, cg and the bus model move rungs 2, 5 and the choice of column.
5. **Cold read rate of the served file; fixed share: which C**, after R75.

## 5. Limits

- The classes in §1 are my reading of where each number came from; a factor measured once in one cell is "D"
  all the same.
- The floor uses the stage model's choice of which stages to split; that choice is a design, though each input
  is measured.
- Rung 3's head time and rung 6's byte factors are r12's and r13's estimates.
- §3's gains are byte arithmetic at the phone's read rate; none is a measurement.
- Every rung holds the body at its R72 value and C at 40.

## 6. Sources (2026-11-23)

Repository: `reports/memo-10-work/bonnie_tree.py`, `reports/memo-12-work/bonnie_joins.py`, `bonnie_tree_r2.py`,
`bonnie_joins_r5.py`, `bonnie_stack_r13.py` (read in full for the factor list); `reports/R72-result.md` (§5, L3,
L4, L6, L7), `reports/R74-result.md` (§4), `prereg/R72-probe-k1-cores-types-anon-device.md` (lines 118 to 119, 140);
`reports/memo-09-gap.md` (lines 31 to 37), `reports/memo-10-tree.md` (lines 47 to 61, 109 to 111);
`reports/memo-09-work/tom-head.md` (§4, §5); `reports/memo-08-work/bonnie-path.md` (§3 lever table);
`reports/memo-07-work/ebony-compute.md` and `bonnie-reprice.md`; [ebony-joins-r3.md](ebony-joins-r3.md),
[ebony-joins-r4.md](ebony-joins-r4.md), [ebony-joins-r5.md](ebony-joins-r5.md); my r4, r5, r12 and r13.
