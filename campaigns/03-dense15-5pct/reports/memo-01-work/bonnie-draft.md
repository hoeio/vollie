# Campaign 3 memo-01, Bonnie's draft: pricing a dense 7B-class model at 15 tok/s

Agent model: claude-fable-5-2. Date 2026-11-18. Lane: pricing (bytes, compute, ms). Ebony's lane: candidates and
frontier. **Nothing here was measured for campaign 3.** Labels: *device* = a campaign 1 or 2 phone measurement,
named by R-number; *host* = laptop or a read of the source tree; *estimated* = arithmetic; *published* = a number
from a cited paper or README (not ours, other test settings). Verdicts are the lead's.

## 0. Summary

1. **No priced combination reaches 15 tok/s on a model with 7.1 to 7.5 billion weights read per token.** Central
   band 4 to 8 tok/s, optimistic corner 12 to 13.5 (*estimated*, §5). Three walls bind at once: resident bytes,
   bandwidth, and integer compute.
2. **Resident bytes is a wall the campaign arithmetic does not yet name.** A dense model reads its whole file every
   token. The largest model set this phone has held resident is 2.67 GB (*device*, c2 R58), with `Cached` up to
   3.2 GB (*device*, c2 R-log), and the app was killed at 3.93 GB of repacked copy (*device*, R9). Every file of an 8B model inside
   F16 × 1.20 is 3.3 GB or more (*published*, §2).
3. **Bits below four do not buy speed on this CPU.** The only kernel class that runs near the byte rate is the
   4-bit repacked `sdot` gemv. Q3_K runs at half its rate (*device*, R41); IQ2 and IQ3 have no arm64 repack kernel
   at the base commit (*host*, tree read) and fail the hard ceiling below about 3.05 bpw anyway.
4. **The +5 % budget buys about 25 to 40 % unstructured input sparsity** (*published*: TEAL, DIP, SpQt). That is
   column skipping, which needs a re-laid copy and a kernel that nobody has built. The designs on paper cost 1.7
   to 2.6 times more per kept weight than `sdot` gemv (*estimated*; mine and Ebony's), so 40 % sparsity is worth
   between 1.05× and 1.5× on the token, not 2 to 3×. Structured variants that the
   kernel likes (expert clusters, CATS row skip, layer drop) cost 6 to 13 times the budget at the sparsity needed
   (*published*).
5. **One opening:** the lead's class definition counts embeddings. A model with 8B total and about 4.5B weights
   read per token (Gemma 4 E4B, *published* by its maker) prices at 7.5 to 13.8 tok/s exact and reaches 15 only
   if the device read rate is 33 GB/s or more (*estimated*). Whether that is "dense 7B-class" is the owner's call.
6. Two device probes decide almost everything, and neither needs a new model round: resident-bytes ceiling, and
   dense read rate plus `sdot` rate with power saving off (§6).

## 1. Constants used

| constant | value | label and source |
|---|---|---|
| dense read rate, 3 threads | 18.9 to 24.3 GB/s (0.0411 ms/MB at 2,112 MHz; head op 18.9) | *device*, R25, R29; power saving most likely on |
| read rate, 4 threads | 22.7 GB/s | *device*, R13/R14 |
| LPDDR5 nominal peak | 51.2 GB/s | recalled, not checked (as in CAMPAIGN.md) |
| repacked Q4_0 4x4 `sdot` gemv, cache-hot, 1 thread | A78 at 2,112 MHz: 78.58 ns per 1,536-weight row = 19.5 weights/ns; X1 at 2,841: 43.23 ns = 35.5 weights/ns | *device*, R41 (r13b, verified core) and R37 (r11, the X1) |
| Q3_K in-register kernel | A78: 156.86 ns per row = 9.8 weights/ns; X1: 87.13 = 17.6 | *device*, same benches |
| kernel share of in-situ time | 0.87 | *device*, R41 (expert op); applied to dense as an assumption |
| clocks | S1 = 2,112 on all three compute cores (campaign 1 days); S2 = caps 2,419 / 2,841 (R67, power saving off) | *device* caps; rates scaled by clock are *estimated* |
| flash re-read | 1.2 to 2.7 ms/MB fitted in decode; 0.7 ms/MB sequential assumed | *device* fits and assumption, c2 memo-18 |
| steady token for 15 tok/s printed | about 63 ms (66.7 ms less about 5 % for ramp and the unused final decode) | *estimated*, scaled from c2 feasibility §2 |

Three-thread `sdot` ceiling (X1 + 2 A78, *estimated* from the device benches): S1 65.5 G weights/s cache-hot, 57.0
in situ; S2 80.3 and 69.9. Four threads at S2: 102.7 and 89.3, but four threads lost on both MoE models
(*device*, R18, R62); for a dense model that is a different regime and is untested.

Model shapes (*estimated*: arithmetic on the published configurations as I recall them; Tom confirms with a GGUF
tensor inventory before any number is reused). Weights read per token exclude the input embedding (a lookup).

| shape | per token | FFN | attention | output head |
|---|---|---|---|---|
| Llama-3.1-8B (32 layers, 4,096 wide, FFN 14,336, vocabulary 128,256) | 7.50 B | 5.64 B (75 %) | 1.34 B (18 %; q and o 1.07, k and v 0.27) | 0.53 B (7 %) |
| Mistral-7B-v0.3 (same blocks, vocabulary 32,768) | 7.11 B | 5.64 B | 1.34 B | 0.13 B |
| Gemma 4 E4B | 4.5 B "effective" of 8 B total (*published*: Gemma 4 technical report and model card); tensor inventory not checked | | | |

## 2. Part 1: bytes and compute per token by quantisation

### 2.1 What the hard ceiling allows (Llama 3 8B, *published*)

llama.cpp's own table (`tools/perplexity/README.md`, wikitext-2 test, context not stated; F16 6.2332, so
F16 × 1.20 = 7.4798). File size includes the input embedding.

| file type | imatrix | GiB | PPL | over F16 | inside × 1.20 | room left for the +5 % budget |
|---|---|---|---|---|---|---|
| q4_K_M | yes | 4.58 | 6.3829 | +2.4 % | yes | full 5 % |
| q4_K_S | yes | 4.37 | 6.4097 | +2.8 % | yes | full |
| iq4_XS | yes | 4.14 | 6.4597 | +3.6 % | yes | full |
| q4_0 | no | 4.34 | 6.7001 | +7.5 % | yes | full |
| q3_K_M | yes | 3.74 | 6.7343 | +8.0 % | yes | full |
| iq3_S | yes | 3.42 | 6.9655 | +11.7 % | yes | full |
| iq3_XS | yes | 3.28 | 7.1630 | +14.9 % | yes | +4.4 % (then the ceiling binds) |
| iq3_XXS | yes | 3.05 | 7.4584 | +19.7 % | by 0.3 % | +0.3 % |
| q3_K_S | yes | 3.41 | 7.6029 | +22.0 % | **no** | |
| iq2_M | yes | 2.74 | 8.6008 | +38.0 % | **no** | |
| q2_K | yes | 2.96 | 8.6478 | +38.7 % | **no** | |
| iq2_S / iq2_XS / iq2_XXS | yes | 2.56 / 2.43 / 2.24 | 9.65 / 10.76 / 14.09 | +55 / +73 / +126 % | **no** | |

Reading: for an 8B model the **floor is about 3.05 to 3.3 GiB (3.3 to 3.5 GB)**; the whole IQ2 and Q2 family is
out by 18 points or more. Our gate is ctx 512 and our own F16 run; the ratios will move a little, the ordering
will not. IQ3_XXS is a coin flip at the ceiling and leaves no budget.

### 2.2 Ceilings (*estimated*; pure type on every mat-mul tensor; real mixes are 3 to 8 % larger)

tok/s if bytes were the only limit, steady token, Llama-3.1-8B shape (Mistral shape: × 1.055):

| type (bpw) | GB per token | 19 GB/s | 24 GB/s | 35 GB/s | 51 GB/s | inside F16 × 1.20 |
|---|---|---|---|---|---|---|
| Q4_0, Q4_K (4.5) | 4.22 | 4.5 | 5.7 | 8.3 | 12.1 | yes |
| IQ4_XS (4.25) | 3.99 | 4.8 | 6.0 | 8.8 | 12.8 | yes |
| Q3_K, IQ3_S (3.44) | 3.22 | 5.9 | 7.4 | 10.9 | 15.8 | IQ3_S yes, Q3_K_S no |
| IQ3_XXS (3.06) | 2.87 | 6.6 | 8.4 | 12.2 | 17.8 | edge |
| Q2_K (2.63) | 2.46 | 7.7 | 9.7 | 14.2 | 20.7 | no |
| IQ2_XS (2.31) | 2.17 | 8.8 | 11.1 | 16.1 | 23.5 | no |
| IQ2_XXS (2.06) | 1.93 | 9.8 | 12.4 | 18.1 | 26.4 | no |

So **bytes alone allow 15 tok/s only at 51 GB/s with a 3.4 bpw file, or with a file that fails the ceiling.** The
measured rate is 19 to 24.

tok/s if integer compute were the only limit (in situ; cache-hot in brackets):

| kernel | S1, 3 threads | S2, 3 threads | S2, 4 threads (untested) |
|---|---|---|---|
| repacked 4-bit `sdot` gemv, Llama-8B shape | 7.6 (8.7) | 9.3 (10.7) | 11.9 (13.7) |
| same, Mistral shape | 8.0 (9.2) | 9.8 (11.3) | 12.6 (14.4) |
| same, 4.5 B per token | 12.7 (14.6) | 15.5 (17.8) | 19.9 (22.8) |
| Q3_K in-register kernel, Llama-8B shape | 3.8 (4.4) | 4.6 (5.3) | 5.9 (6.8) |

This is the "int8 ceiling near 10 tok/s" of c2 feasibility §1, now with its source: **7.6 to 9.3 tok/s for an 8B
model on three cores, below 15 in every clock state**, 11.9 with a fourth core that has lost twice on this phone.
Weights that fit in the 63 ms budget: 3.6 G (S1), 4.4 G (S2), 5.6 G (S2, four threads). An 8B model has to
**not compute 41 to 52 % of its weights** even with the best kernel and infinite bandwidth.

### 2.3 Which types have a byte-rate kernel on this CPU (*host*, tree read at base `620d2225c`)

`ggml-cpu/repack.cpp` selects a dotprod NEON path for Q4_0 (4x4), Q4_K (8x4), Q5_K, Q6_K, IQ4_NL (4x4), Q8_0,
MXFP4 and Q1_0. There is **no repack path for Q3_K, IQ3_*, IQ2_* or Q2_K on arm64**. KleidiAI takes Q4_0, Q4_K,
Q8_0 and Q6_K. Upstream history agrees: Q4_K dotprod repack came in PR #17494 (2025-11), Q6_K in #19360.
The SD888 has dotprod and no i8mm.

- Only Q4_0's rate is a *device* number. Q4_K 8x4 and IQ4_NL 4x4 on the A78 are unmeasured; the ik_llama.cpp
  wiki (M2 Max, 2024, prompt processing, *published*) has IQ2_XS at 0.35 and IQ3_XXS at 0.22 of Q4_0's rate.
- A 3-bit file therefore runs at 3.8 to 4.6 tok/s by compute (table above), **slower than Q4_0's byte-bound 4.5
  to 5.7**. An exact nibble container for Q3_K (campaign 1's option 3) runs at the `sdot` rate but is 4.5 bpw
  again: no byte gain.
- Every repack is a second copy in anonymous memory. For a 7 to 8B file that is 4.0 to 4.2 GB, above the 3.93 GB
  that was killed (*device*, R9). The repacked layout would have to be written once to the app's files and
  mapped (a derived artefact, allowed by COMMON_RULES 1; about 4 GB of storage).

### 2.4 The resident-bytes wall

| fact | value | label |
|---|---|---|
| page cache the phone kept for the model | 1.6 to 2.4 GB on the memo-18 days; largest resident model set 2,672 MB (OLMoE, its whole working set); `Cached` before runs 2.4 to 3.2 GB | *device*, c2 memo-18, R58, R-log lines for R58 to R67 |
| system `MemAvailable` / `Cached` | about 2.0 / 2.1 GB | *device*, R9 |
| largest weight set that ran resident | 2.27 GB repacked Qwen3.5-4B, 5.9 to 6.6 tok/s | *device*, 2026-10-25 anchor |
| smallest that was killed | 3.93 GB repacked Granite | *device*, R9 |
| `mlock` limit | 64 MiB | *device*, r22 |

The ceiling W* is somewhere in 2.7 to 3.9 GB and is **unmeasured**. A dense token needs all of its bytes; the
shortfall is re-read from flash every token at 0.7 ms/MB at best: 1 GB short is +700 ms per token (*estimated*).
Activation sparsity does not help here, because the union over tokens is the whole file (OLMoE's case in memo-18).
Consequences (*estimated*):

- Llama-8B or Mistral at 4.5 bpw (4.0 to 4.2 GB): resident only if W* ≥ 4.0, above the R9 kill. **Likely never
  resident.**
- The same at 3.06 to 3.44 bpw (2.7 to 3.2 GB): resident if W* ≥ 2.7 to 3.2. Open.
- 4.5 B per token at 4.5 bpw (2.53 GB): inside the 2.67 GB that has been held, before KV and buffers. The only
  case with device precedent.

## 3. Part 2: runtime sparsity, priced

Per layer of the Llama shape: 218 M weights = 2.9 % of the token = 123 MB at 4.5 bpw = **5.1 to 6.5 ms at 19 to
24 GB/s** (*estimated*). The whole token is 176 to 222 ms by bytes. All PPL figures below are *published*, on
FP16 weights unless stated, wikitext-2, not our gate.

| method | what is skipped | PPL cost reported | bytes real on this CPU? | ms per token here (*estimated*) |
|---|---|---|---|---|
| **TEAL** (ICLR 2025), magnitude threshold on the input of all 7 matrices | columns | Llama-3-8B 25 / 40 / 50 %: +1.2 / **+5.8** / +13.6 %; Mistral-7B +1.8 / **+4.3** / +7.9 %; Llama-2-7B +0.4 / +3.0 / +7.1 % | no in the served layout (see below); yes with an input-major copy | 40 % skipped: bytes −70 to −89 ms, compute +5 to +74 ms; token (max of the two) **−8 to −80 ms**, set by the column kernel's cost (2.6× to 1.7× `sdot`) |
| **SpQt** (arXiv 2511.04477, 2025-11), TEAL thresholds on Q4_K inside llama.cpp, column-wise "zigzag" superblocks | columns, decode only | Llama-2-7B Q4_K 5.28 → 5.27 (25 %), 5.65 (50 %, +7.0 %), 7.39 (65 %) | yes, but the kernel is Metal on Apple GPUs: 1.33 to 1.55× at 50 %; a naive sparse kernel was below 0.2× | no CPU kernel exists; same pricing as TEAL |
| **DIP** (Qualcomm, arXiv 2412.01380), top-K on FFN inputs and on the GLU output | columns of gate, up, down | FFN density 60 %: Llama-3-8B +8.5 %, Mistral-7B **+2.7 %**; 50 %: +20 %, +6.9 % | as TEAL; their throughput is a simulator with flash and no compute model | FFN only: 75 % of the token |
| **R-Sparse** (ICLR 2025), sparse inputs plus a low-rank path | columns, plus an SVD factor per matrix | "comparable" at 50 % on task accuracy; PPL only in a figure; with 4-bit GPTQ the average drops 67.3 → 65.8 | needs stored SVD factors (a derived artefact made on a GPU: rules question) and a second mat-mul path | not priced: two kernels per product |
| **CATS** (2024), threshold on the gate output | rows of up (free in the served layout), columns of down | training-free: Llama-3-8B 25 %: **+15.5 %**, 40 %: collapse; Mistral 25 %: +19 % (TEAL Table 1); DIP's table: 60 % density +54 % | rows yes | over budget before it saves 8 % of the token |
| **CMoE** (arXiv 2502.04416), FFN neurons clustered into experts, analytic router, training-free | whole experts (contiguous after a re-lay) | 75 % of FFN active: Llama-2-7B 5.27 → 7.02 (**+33 %**), Llama-3-8B 6.14 → 10.16 (**+65 %**); 25 % active: 62 and 143; with LoRA 5.69 and 8.26 | yes, this is the kernel-friendly form | −19 % of the token at 6 to 13 times the budget |
| **ExpertWeaver** (arXiv 2602.15521, ICML 2026), GLU-pattern experts, training-free | whole experts | reported better than CMoE at 25 % sparsity on tasks; PPL not retrieved | yes | Ebony: fetch the PPL table; the only structured method not yet priced out |
| **Attention-layer drop** (He et al., arXiv 2406.15786), static, by similarity | q, k, v, o of a layer: 42 M = 0.56 % of the token each | task accuracy only; no PPL for an 8B model found | yes, static, also leaves the resident set | 1.0 to 1.3 ms per dropped layer; 8 layers 8 to 10 ms (4.5 %) |
| **Layer skip** (static, or D3, arXiv 2503.08524) | a whole block: 2.9 % | no single-layer PPL for an 8B model retrieved; campaign 1's own cheapest layer cost +2.5 % (*host*, R21, Granite) | yes, static | 5.1 to 6.5 ms per layer; +5 % buys one or two |
| FlexiDepth, DiffSkip, ExpertNavigator, DIP+LoRA, CMoE+LoRA | | need trained routers or adapters | outside "stock weights" unless the owner rules otherwise | not priced |
| **Head-row skipping** (campaign 1's two-stage head) | rows of the output head | campaign 1: +0.01 % with exact top-2,048 (*host*, R31) | yes | Llama shape: head is 295 MB = 12 to 16 ms; a screen below 4 bits was compute-bound (R29), so at most −4; Mistral: head is 3 ms in total |
| Attention head skipping (dynamic) | q rows and o columns of a head | no training-free PPL source found | q rows yes; o columns no | q and o are 14 % of the token; half of the heads is at most −7 %, unpriced in PPL |

### Why column skipping does not pay here (campaign 1's finding, re-applied)

*Note added after reading Ebony's draft: the second and third bullets and the two kernels below are superseded
in part by her integer column kernel (review, point 2). The band 1.7 to 2.6× per kept weight and C3's 5.1 to 9.3
tok/s stand; the float-accumulation argument does not hold for a re-laid Q4_0 copy that keeps its row scales.*

- The served layout is row-major with quant blocks along the input. A skipped input removes 4 bits from every
  block of every row: no cache line is saved. Campaign 1 measured this on Granite: 46 % of `ssm_out` inputs were
  skippable per element, 0.7 to 12 % as whole 32-blocks (*host*, `05_compute_less.md` §8). The published
  thresholds are per element too.
- An input-major copy makes the bytes real, but the product becomes an axpy. With per-block scales it accumulates
  in float: 0.94 NEON instructions per weight against 0.17 for the 4x4 `sdot` gemv (*host* count, campaign 1,
  `dense_bytes_device.md`); that priced `ssm_out` at +2 to +5 ms and closed it.
- Two unbuilt kernels could do better (*estimated* instruction counts, not benched): four kept columns
  interleaved in registers and fed to `sdot`, about 0.44 per weight (2.6×); or pre-interleaved 4-neuron groups
  with one float fix-up per block, about 0.30 (1.75×). Break-even kept fraction: 39 % and 57 %. The budget
  allows 60 to 75 % kept. So the compute side gets worse while the byte side gets better, and at Q4_0 the two
  sides are within a factor 1.3 to 2 of each other (compute 107 to 132 ms, bytes 176 to 222 ms).
- **The 0.30 kernel only serves masks made of fixed groups of four neurons.** With per-element masks at 60 % kept,
  a fixed group has at least one kept member 97 % of the time (1 − 0.4⁴, independence assumed), so nothing is
  skipped. Per-element masks, the only kind the papers price, need the 0.44 kernel. What a 4-group mask costs in
  PPL is unknown; campaign 1's 32-block result says structure is expensive. T4 measures it.
- Worked case, Mistral shape, 40 % input sparsity (the whole budget, +4.3 % *published*), 0.30 kernel: bytes
  4.00 → 2.40 GB = 100 to 126 ms; compute 102 to 125 ms × 0.6 × 1.75 = 107 to 131 ms. Token 107 to 131 ms against
  167 to 211 ms dense: 7.6 to 9.3 tok/s, **but only if a 4-group mask costs no more PPL than a per-element one**,
  the kernel meets its count, the copy is resident and its re-quantisation is free. With the 0.44 kernel, which
  is what a per-element mask needs: compute 159 to 195 ms, **5.1 to 6.3 tok/s** against 4.7 to 6.0 dense. With
  float axpy: slower than dense.
- A rule that sparsifies decode but not prefill (SpQt does exactly that) falls under the owner's 2026-11-14 rule:
  inside the budget at both `-b 512` and `-ub 1`.

## 4. What is closed already and stays closed

Draft speculation (R20), early exit as priced on Granite (R21), sub-4-bit screens (R29), dense column skip with
a float axpy (R29 addendum), OpenCL on the Adreno 660. I re-open two with a stated reason: **four threads** and an
**A55 helper**, because the dense token is one long compute-or-byte-bound product and not hundreds of small ops;
both go on the try list as cheap device probes, not as predictions.

## 5. Part 3: combinations

All *estimated*; printed tok/s is about 0.95 of the steady figure. "Resident" assumes §2.4's wall is passed.

| # | combination | steady tok/s, central band | best corner and what it needs | budgets |
|---|---|---|---|---|
| C1 | 7.1 to 7.5 B per token, Q4_0 or Q4_K repacked, exact | 4.5 to 6.0; by the Qwen anchor scaled on bytes 3.1 to 3.7 | 12.2 to 12.7 at 51 GB/s with four threads | served +2.4 to +7.5 % over F16; **not resident (4.0 to 4.2 GB)** |
| C2 | C1 + one or two static layers + attention-layer drops + head screen (kernel-free, about 10 % of the token) | 5.0 to 6.6 | 13.5 | whole +5 %; not sourced for PPL |
| C3 | C1 + 40 % input sparsity with a new column kernel at 1.7 to 2.6× `sdot` per kept weight | 5.1 to 9.3 (the kernel count decides; see the review, point 2) | about 10 at S2 | whole +5 % on Mistral, over on Llama-3-8B; copy re-quantised; not resident unless the copy replaces the file |
| C4 | 8B at IQ3_XS or IQ3_S (3.3 to 3.5 GB) with today's kernels | 2 to 4.6 (compute) | 5.9 with four threads and a Q3_K-class kernel | ceiling leaves +4.4 % or +5 %; residency open |
| C5 | 4.5 B per token (8 B total with embeddings), Q4_0 repacked, exact | 7.5 to 9.5 at 19 to 24 GB/s | 13.8 at 35 GB/s; **15.5 at ≥ 40 GB/s and S2 clocks**; 19.9 with four threads | Q4_0's own PPL against F16 unmeasured; 2.53 GB: residency open |
| C6 | C5 + 13 % kernel-free skip (static layers, attention drops) | 8.5 to 11 | **15 at 33 to 35 GB/s and S2 clocks** | most of the +5 % |

**Numbered statement.**

1. For a model that reads 7.1 B weights or more per token, the best priced combination is C3 at 5.1 to 9.3 tok/s
   and the physical corner is 12 to 13.5. **15 tok/s is not reachable**; it would need at once a resident 4 GB,
   at least 32 to 39 GB/s, and 41 to 52 % of the weights not computed at no kernel cost, where the budget buys 25
   to 40 % at a kernel cost of 1.7 to 2.6× per kept weight.
2. The lowest per-token load that still allows 15 at S2 clocks on three cores is **4.4 G weights** (compute) and
   **2.2 GB at 35 GB/s, 1.5 GB at 24 GB/s** (bytes). A 6.0 B model at the floor of the class definition (about
   5.6 to 5.8 B per token) is still 25 % above that.
3. Only a model like C5 can get there, and only if the power-saving-off probe shows a read rate of 33 GB/s or
   more. At the campaign-1 rate (19 to 24) it is 7.5 to 11 tok/s.
4. I would put the chance of 15 tok/s on a Llama- or Mistral-shaped model below 0.05, and on a C5/C6 model at
   about 0.2 to 0.3, almost all of it riding on the read-rate probe (*estimated*, judgement).

## 6. Proposed test list and try list (for the joint memo)

Test list (decides a number already in this memo):

- **T1 resident-bytes ceiling** (*device*, existing kit): map any 3.0, 3.5 and 4.2 GB GGUF, read it end to end ten
  times, record mincore, `RssFile`, major faults, swap and the read rate per pass. Kill line for a 7B at 4.5 bpw:
  resident below 4.0 GB.
- **T2 read rate and `sdot` rate with power saving off** (*device*): the r13b `cpu-bench` (q4x4, per core,
  per clock) plus a pure sequential-read bench at 1, 2, 3 and 4 threads over 1 GB. Decides §1's S2 column and
  whether 33 GB/s exists.
- **T3 kernel bench by type** (*device*, same bench): Q4_K 8x4, IQ4_NL 4x4, KleidiAI Q4_0, against q4x4. Decides
  whether a +2.8 % file (Q4_K_S) costs speed against a +7.5 % one (Q4_0).
- **T4 host PPL screen** with `tools/dense_colskip.py` generalised to FFN and attention inputs, on two models, at
  `-b 512` and `-ub 1`: per-element thresholds for 25 / 40 % and the same with 4-group and 32-block masks.
  Decides whether the published FP16 costs hold on the served quantisation and what structure costs.
- **T5 tensor inventory** of each candidate (`tools/cost.py`): replaces §1's shape arithmetic.

Try list (cheap, may surprise):

- Y1 four threads, and three threads plus A55 helpers, on a dense file (re-opened, §4).
- Y2 host microbench of the column kernels (Ebony's integer kernel first; my 0.44 and 0.30 designs only if it
  misses 2.2×) before anyone writes a spec.
- Y3 static attention-layer drop and one-layer skip PPL on two models (host).
- Y4 ExpertWeaver's training-free PPL at 25 % (Ebony), the one structured method not priced out.

Questions for the owner, through the lead: (a) does a model with 8 B total and 4.5 B read per token count as
"dense 7B-class"; (b) does an artefact trained off the phone with the weights frozen (a router, SVD factors,
thresholds) count as runtime; (c) is a publisher's QAT checkpoint "stock weights".

## Sources

- llama.cpp quantisation table: <https://raw.githubusercontent.com/ggml-org/llama.cpp/master/tools/perplexity/README.md>
- TEAL: <https://arxiv.org/html/2408.14690v1>
- SpQt, "Enabling Dynamic Sparsity in Quantized LLM Inference": <https://arxiv.org/html/2511.04477v1>
- DIP: <https://arxiv.org/html/2412.01380v2>
- R-Sparse: <https://arxiv.org/html/2504.19449v1>
- CMoE: <https://arxiv.org/html/2502.04416v2>
- ExpertWeaver: <https://arxiv.org/abs/2602.15521>
- Attention drop: <https://arxiv.org/abs/2406.15786>; D3: <https://arxiv.org/html/2503.08524v1>; FlexiDepth: <https://arxiv.org/html/2503.23798v3>
- Q4_K dotprod repack: <https://github.com/ggml-org/llama.cpp/pull/17494>; Q6_K: <https://app.semanticdiff.com/gh/ggml-org/llama.cpp/pull/19360/overview>
- i-quant speed on NEON (2024): <https://github-users.yhajoa.workers.dev/ikawrakow/ik_llama.cpp/wiki/July-2024:-Prompt-processing-perfomance-comparison>
- Gemma 4 E4B: <https://huggingface.co/papers/2607.02770.md>, <https://huggingface.co/google/gemma-4-E4B-IT>
- Repository: `reports/03_final.md` (R9, R20, R21, R25, R29, R37, R41), `reports/05_compute_less.md` §8,
  `campaigns/02-25toks-5pct/reports/memo-18-residency-options.md`, `lab/llama.cpp-base` at `620d2225c`.

Not done: no host run, no device run, no GGUF opened; R-Sparse's and ExpertWeaver's PPL tables and any
single-layer-removal PPL for an 8B model were not retrieved.

## Review of Ebony's draft (read 2026-11-18, after my sections were written)

We reached the same four conclusions separately: residency first, only input-side methods fit +5 %, 40 % sparsity
is about 1.4× and not 2 to 3×, and no published method gives 15 tok/s on a 7 B body. Points where one of us has to
change:

1. **Residency figures: hers are the fuller set, I corrected mine.** I had quoted only memo-18's 1.6 to 2.4 GB.
   The R-log has `Cached` up to 3.2 GB and a resident model set of 2,672 MB (*device*, R58 onward). §2.4 and the
   summary now say so. Her "2.4 to 3.2 GB page cache" mixes system `Cached` with what the model held; the model
   figure is 2.67 GB. The probe (her P1, my T1) should also hold the KV cache and buffers (0.2 to 0.6 GB of
   anonymous memory, *estimated*) while it reads, or it will overstate the ceiling.
2. **Her column kernel (§3.1) is better than the ones I priced, and I withdraw part of my §3.** I had carried over
   campaign 1's reading that a column product must accumulate in float. For Q4_0 that is wrong when the nibbles
   are re-laid and the row scales are kept: all 32 columns of a block share a row's scale, so the sum over kept
   columns is an integer and the float step is once per row and block. My "four kept columns interleaved for
   `sdot`" (2.6×) and "fixed 4-groups" (1.75×, with its mask caveat) are then not needed. Her count is 12 NEON ops
   per 32 weights against 7; mine for the same loop is 8 plus accumulator traffic: the int16 sums of 4,096 to
   14,336 rows do not stay in registers, so each `smlal` pair is a load and a store unless the loop is turned
   inside out, which makes up to 32 read streams. I keep 1.7 to 2.6× as the band and C3 at 5.1 to 9.3 tok/s. Her
   H6 with a 2.2× kill line is the right test, and it should run on the A78 with the real row counts.
3. **"Bit-identical at threshold 0" is doubtful.** The stock NEON dot converts four int32 lane sums to float per
   block and adds them into two alternating float accumulators; an int16 total per row and block gives the same
   integers but not the same float rounding unless the lane split and order are copied, which costs four
   accumulators per row. And the phone's stock numerics are KleidiAI's (her own rules question 2, and R61's
   ±0.5 %). I would pre-register the kernel as a declared numerics change with a small expected PPL shift, not as B.
4. **Her Gemma 4 E4B row "13 to 18 tok/s" is too high by her own inputs.** Dense compute is 65 ms; 40 % sparsity
   with a 1.85× kernel is 65 × 0.6 × 1.85 = 72 ms, not 55 to 65, so the token is 72 to 76 ms: 13 to 14 tok/s.
   It also assumes 35 to 40 % tolerance on a model none of the papers measured. My C5 and C6 (7.5 to 11 central,
   15 only at 33 GB/s or more with kernel-free skips) are the lower and, I think, the safer reading. Both of us
   need H1's inventory: a 262 k vocabulary head could be a tenth of that model's token.
5. **Compute time: same number, different route.** Her 103 ms for Mistral is three A78s at 0.85 of the 2,112 MHz
   cost; mine is X1 + two A78s at S2 with the 0.87 in-situ factor (102 ms). At campaign-1 clocks mine is 125 ms.
   The memo should carry both clock states until P2 is back.
6. **Read rate.** She uses 21 to 27 GB/s (OLMoE's head op), I use 19 to 24 (campaign 1). Neither is a dense-model
   number with power saving off. Not worth arguing before P2.
7. **Quantisation ladder: her source and mine agree.** arXiv 2601.14277 has Llama-3.1-8B Q3_K_S at × 1.224 and
   Q3_K_M at × 1.087; llama.cpp's table has × 1.220 and × 1.080 to 1.105. Mine adds the IQ3 rows (IQ3_XS × 1.149,
   IQ3_XXS × 1.197) and the kernel fact that none of them has an arm64 repack path at the base. Her "no ARM repack
   for K-quants that I know of" is out of date: Q4_K, Q5_K and Q6_K have dotprod paths in `620d2225c` (§2.3).
8. **H3 cannot change the answer for a 7 B body.** With her kernel and the best published tolerance the token is
   at most 1.5× faster. H3's closing line (under 25 %) decides how far below 15, not whether. So I would order the
   work: P1 and P2 (device, small), H1, H6 (host microbench), then the owner's answer on Gemma 4 E4B, and only
   then H3 and the large kernel (her Y3), on the model set that answer leaves.
9. **PowerInfer-2's SiLU row** (Mistral-7B, 5.3 tok/s on a phone with 43.9 GB/s, source) is a useful outside
   check on statement 1 of §5, with the caveat that it was run with half of the FFN in flash.
10. Candidates: no objection to Mistral-7B-v0.3 plus Qwen2.5-7B. By pricing, Qwen2.5's 0.55 B head is 0.31 GB =
    13 to 16 ms (*estimated*), the one place campaign 1's two-stage head earns something on a dense model.
    Yi-1.5-6B (3.3 GB at Q4_0) is the only in-definition transformer near the residency ceiling, but it is 25 %
    over the compute budget of §5 statement 2.

Proposed merged lists for the joint memo: tests P1 (= T1), P2 (= T2 and T3), H1 (= T5), H6 (= Y2), H2, H7 (= Y3),
then H3 and H5; tries Y1 (layout in a file), my four-thread and A55 probe (inside P2), Y2 (head), and Y3 + Y4 only
after H6 and the owner's answer.
