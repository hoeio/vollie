# c3 memo-12 work (Ebony, round 9): the W-position kernel's exactness test as a node comparison, one-sided steps on the phone's path, review of Bonnie r17 and r18

Date: 2026-11-24. Author: Ebony (`claude-fable-5-2`). Not committed; the lead commits. No verdict here.
Follows [ebony-joins-r8.md](ebony-joins-r8.md). Reviews [bonnie-joins-r17.md](bonnie-joins-r17.md) and
[bonnie-joins-r18.md](bonnie-joins-r18.md) (both read; r18's last lines after its step table not seen).

Labels as before. Triples are Yi / Llama-2 / DeepSeek. New *host* numbers: `ebony_batchdiff.py` (two modes added)
on Llama-2 with the base tree's repack build and T1b's KleidiAI libraries, two threads at low priority; one
clip count on Tom's head dumps. No Metal, no phone.

## 0. Answer first

1. **Yes: the exactness test can be a node comparison, and the method already passes on both stock paths.**
   32 tokens decoded two at a time and four at a time give every one of 553 nodes bit-equal to single decodes on
   the KleidiAI libraries (*host*, Llama-2). The W-position kernel passes when the same run with its knob on
   still reads zero differing nodes. Specification for Tom in §1.3. It replaces part A of r5 §3.3 as the first
   test; parts B to D stay.
2. **With the app's Q8_0 cache, a batch of 64 and 64 single decodes are bit-equal on every node** (*host*, the
   KleidiAI libraries; 585 nodes). So on the phone's configuration `-b 512` and `-ub 1` are the same arithmetic
   for stock. **A free check for the second reference: its two settings must give identical per-chunk values.**
   If they do not, something in that build is not what we think.
3. **One exception found on the way:** on ggml's repack path the Q6_K head at four or more rows differs from the
   one-row result in the last bits (1.9e-6, half of the logits). It is the last op, so nothing amplifies it.
   It matters only for a claim of bit-equal logits at width 4 or above on the repack head.
4. **No other one-sided step on the phone's path** (§2). Every quantiser in the body and in the cache is
   symmetric and cannot saturate at the ranges measured. The head's asymmetric quantiser does clip, on 0.1 % of
   rows by one step, with no effect on log-probability to six decimals. The one-scale rule of r7 remains the
   only source of bias I can find.
5. **Bonnie r17 (choices A, B, C): agreed.** It is the argument of my r8 §3 carried through with prices.
   **r18 (kit r36): agreed as a design, with three requests** (§3): more anonymous memory in the lane steps, a
   decision on where the head's bytes live, and the two kernel-feature lines in the report.
6. **Correction to my r8:** I priced a query-key pair in the phone's attention at about 1 µs. By her r17 count it
   is about 3.4 µs over 32 heads. The prefill's attention is then 0.5 s on four threads, not 0.1 to 0.3 s. Still
   small against a 6 to 15 s prefill.

## 1. The exactness test as a node comparison (my question 3)

### 1.1 What the tool does now

`ebony_batchdiff.py` hooks every graph node through llama.cpp's evaluation callback, keeps the last token's column
of every two-dimensional float node, and compares two ways of decoding the same tokens bit for bit.

| mode | first run | second run |
|---|---|---|
| default | one batch of N | N single decodes |
| `EBONY_B=<B>` (new) | N tokens in decodes of B; the last decode's last column | N single decodes |
| `EBONY_Q8KV=1` (new) | either of the above with a Q8_0 cache and flash attention on, as the app sets them | – |
| `EBONY_SAVE=<file>` | saves the first run, for a comparison across builds | – |

One node is always excluded by eye: `node_1`, a second embedding lookup whose buffer is scratch (it differs even
when the batch is run twice).

### 1.2 The method on stock (*host*, Llama-2, first tokens of wikitext-2 test)

| build | cache | comparison | nodes that differ (of) | logits |
|---|---|---|---|---|
| repack | F16 | decodes of 2 against singles, N 32 | **0** (553) | equal |
| repack | F16 | decodes of 4 against singles, N 32 | 1: `result_output` (the head), 1.9e-6, 15,505 of 32,000 entries | rms difference 2.2e-7 |
| KleidiAI | F16 | decodes of 2 against singles, N 32 | **0** (553) | equal |
| KleidiAI | F16 | decodes of 4 against singles, N 32 | **0** (553) | equal |
| repack | Q8_0 | one batch of 64 against singles | 1: `result_output`, 1.9e-6 | rms difference 2.0e-7 |
| KleidiAI | Q8_0 | one batch of 64 against singles | **0** (585) | equal |

- Rows 1 and 3 reproduce Tom's T1b (batch 2 equals batch 1 on both stock paths) on every node, not only on the
  logits.
- Rows 2 and 5: ggml's repack path sends four or more rows of the Q6_K head to its tile kernel, whose float
  order is not the one-row kernel's. KleidiAI's head is equal at every width tried.
- Row 6 is item 2 of §0. With an F16 cache the same comparison differs in 543 nodes (r8 §1.2).
- The KleidiAI libraries here are T1b's (BLAS on, no instruction-set flag). The comparison is inside one build,
  so the flags cannot matter; the reference build of §1.3 should still be the clean one.

### 1.3 Specification for Tom (one task; host; after the kernel is in the tree)

**Build.** The freeze's flags plus KleidiAI and nothing else: `GGML_CPU_ARM_ARCH=armv8.2-a+dotprod+fp16`, BLAS
off, `GGML_CPU_KLEIDIAI=ON`, run with `GGML_KLEIDIAI_SME=0`. The log must show
`kleidiai: primary q4 kernel feature DOTPROD` and `CPU_KLEIDIAI model buffer size`. First run E-T13 (r8): layer
0's q, k and v outputs equal to the freeze build's.

**Tool.** `ebony_batchdiff.py` as it is, plus two small changes:
1. compare **every** column of the last decode, not only the last one (the second position of a width-2 pass is
   the one the kernel adds);
2. exit with status 1 when any node other than `node_1` differs, and print the first differing node's name, op,
   position, row, and both values in hex.

**Runs** (two subjects at least; all three is 6 more minutes):

| # | knob | cache | `EBONY_B` | N | pass |
|---|---|---|---|---|---|
| 1 | off | Q8_0 | 2 | 64 | 0 differing nodes (the method's own check on this build) |
| 2 | **on** | Q8_0 | 2 | 64 | **0 differing nodes** |
| 3 | on | Q8_0 | 3, then 4 | 60, 64 | 0 differing nodes (only if widths 3 and 4 are built) |
| 4 | on | F16 | 2 | 64 | 0 differing nodes (below 64 queries the attention is the per-query path) |
| 5 | on against off | Q8_0 | 2 | 64 | with `EBONY_SAVE`: the two saved runs equal node for node (the kernel against the stock tile, same decode pattern) |
| 6 | on | Q8_0 | 2 | 192, from the official prompt with each subject's chat template | 0 differing nodes (control tokens and Yi's 5,000-size activations are in the input) |

- Cost: about 2 minutes per run on two threads; under an hour with the build.
- What it covers that random data did not: every layer's real activation range, every matrix shape, the work
  buffer's size (a wrong `work_size` shows as garbage in the first `MUL_MAT` node), the head.
- What it does not cover, so r5 §3.3 keeps them: edge vectors that real text does not produce (part B: zero
  blocks, infinities, the clamp), thread counts above two and a row range that does not start at 0 (add
  `threads=4` to runs 2 and 5), the KV rollback after a missed guess (T10), and the phone (part D).
- Reading: any differing node in runs 2 to 6 → not exact as built. The first differing node names the op.

## 2. One-sided steps on the phone's path (my question 5)

A step can bias perplexity below the noise floor of r8 §1.3 only if its error has a direction. I went through
every place where the phone's stock path rounds, clamps or changes type.

| step | where (*source*) | rounding | can it saturate? | one-sided? |
|---|---|---|---|---|
| activations of the Q4_0 chain | `kai_lhs_quant_pack_qsi8d32p_f32.c` lines 91 to 110: scale = block maximum ÷ 127, `roundf`, cast to int8 | half away from zero, symmetric | no: the maximum maps to ±127 | no |
| that block's scale, stored as half | same file, line 95 | nearest | above 8.3e6 or below 8e-6 of block maximum. Largest input measured: 5,309 (*host*, r6) | no |
| Q4_0 weights | the file's nibbles and half scales, re-ordered | none | – | no (the format's range −8 to 7 is the same on every path) |
| integer dot and `scvtf #4` | the gemv's assembly | exact | int32 holds 32 × 127 × 128 with room | no |
| float accumulation per block | FMUL, FMLA | nearest | – | no |
| the clamp to ±FLT_MAX | `fmax`, `fmin` at the end of each kernel | – | only an infinite sum, which needs a value above 8.3e6 | no |
| K and V into the Q8_0 cache | `quantize_row_q8_0`, blocks of 32 | nearest even, symmetric | no. K and V outputs measured: up to 21 | no |
| Q for the score | the same quantiser, per query | symmetric | no | no |
| V back to float, value sum in float | `dequantize_row_q8_0`, `ggml_vec_mad_f32` | exact, then nearest | – | no |
| **head weights, int8 with one scale per row** | `kleidiai.cpp` lines 1560 to 1580 | `lround`, symmetric; clamp at ±127 | only the row's own maximum, which maps to 127 | no; the error is larger than Q6_K's (0.9 to 1.0 % of row rms, Bonnie r11) but has no sign |
| **head activations, one scale and zero point per vector** | `kai_lhs_quant_pack_qai8dxp_f32.c` | half away; then + zero point, **clamp to −128…127** | **yes**: rounding the zero point can push the vector's extreme component one step out | see below |

- **The head's clip, measured** (*host*, Tom's dumps, my quantiser with and without the clamp): it happens on
  0.1 % of rows (3 of 3,563 on Yi, 2 of 3,638 on Llama-2), always one component by one step, on Yi's dimension
  3889 twice. Change of −log p(dumped token): +0.000000. Not a source of bias.
- **So the bias of the head path is the resolution loss itself** (r7 §2.4: a few fixed, very large dimensions
  set the step for all 4,096), not a clip and not a rounding rule. Noise in the logits always costs
  log-probability on average, which is why this one has a sign and the body's symmetric errors show none: the
  head's error enters after the last layer, so it is not washed into the floor, and it is five times the block
  quantiser's on Yi.
- Not examined: the plain path of choice A (ggml's row-at-a-time kernels). Its quantisers are the same
  `quantize_row_q8_0` and the Q8_K one for the head, both symmetric.
- The KleidiAI source read here is the lab copy that Bonnie identifies as v1.31.0; the shipped library is built
  from the same pinned tag. I did not disassemble the phone's module.

## 3. Review of Bonnie r17 and r18

### 3.1 r17

| her claim | my position |
|---|---|
| A (the app's rule) is the only choice the phone can do today; B (forced anonymous copies) is what every speed number assumes and what the evidence is against for Llama-2 and DeepSeek; C (side file) is the only one both fast and inside the rules | **agree.** The source lines are the ones I checked in r8 §3 |
| A's stock speed 4.3 / 3.9 / 3.9 tok/s (body × 1.62, head × 1.7) | agree as an estimate; r18's `x5p` measures it |
| B would need the app's bound lowered to where campaign 1's phone was killed; "not consistent as it stands" | agree. One sentence I would add for the lead: **every campaign-3 stack printed so far, mine included, is a choice-B or choice-C number.** Under A the lane is width 1 only and the best stack is her 8.0 to 8.7 at K 300 |
| P(resident through a claim block): A 0.85 / 0.65 / 0.55, B 0.45 / 0.15 / 0.15, C 0.85 / 0.65 / 0.60; with a lane 0.85 / 0.45 / 0.40 | judgements; mine are within 0.1 of each |
| the Q8_0 value conversion is vectorised in the shipped `libggml-base.so`, 36 to 45 ns per 128-value row, 30 to 45 % of the attention's per-row work | not checked (I did not disassemble). Used in §0 item 6 as given |
| sharing the converted row at width 2 saves 0.9 / 1.7 / 1.6 ms per pass; exact | agree it is exact: each query keeps its own operations. By §1.2 a node comparison at width 2 would show it in two minutes |
| an F16 value cache removes the conversion | agree; it is a numerics change towards the freeze, and it brings the half-precision value sum back (r8 §1.1), so the phone's attention would stop being batch-invariant at 64 queries and above. The gate's two settings would differ again |

### 3.2 r18 (kit r36, round R76)

**Agreed as a design.** It measures the three things the A / B / C decision rests on and loads no model.
Requests, in order of weight:

1. **The anonymous part of the lane and whole-body steps is low.** She lowered it from 0.25–0.30 to 0.15 GB
   because the cache is Q8_0. Beside the cache (17 / 136 / 128 MiB at n_ctx 512) the context holds a compute
   buffer of 102 to 105 MiB at a micro-batch of 512 (*host*, every perplexity log of mine: "CPU compute buffer
   size is 105.02 MiB"), and the app sets micro-batch 512 (her r16 row 14). That is 0.24 GB on Llama-2 before
   the app's own memory. **I would keep 0.25 GB in `x5s2`, `x5s3`, `x5s4` and `x5sA`.** At n_ctx 4,096, the
   app's default, the cache alone is 1 GiB on two subjects; the pre-registration should say the steps stand for
   n_ctx 512 only.
2. **Where the head's bytes live has to be fixed before the sizes are.** r12 recommends the head in the repack
   type; its copy is anonymous (0.2 / 0.1 / 0.3 GB, her r17 §1.4). The steps for choice C carry the side file at
   the KleidiAI buffer's full size, head included, and no anonymous head. Either the side file also holds the
   head in whatever layout is chosen (then the steps are right as drawn), or `x5s3` and `x5s4` need 0.3 GB more
   anonymous. The first is cheaper and I would specify it: the side file is "every packed tensor", not "the
   KleidiAI buffer".
3. **Carry r35's `x4 kai` and `x4 kai log` lines into `x5p`,** so that the report of the first round that times
   the head in three buffer types also says which 8-bit kernel entry the phone took.
4. Smaller:
   - `x5 pl head` for `CPU_KLEIDIAI` times the int8 head. Its input should be a vector with a realistic range
     (it does not change the time, but an identity line against float64 on that cell, with a Yi-like input of
     range 40 × rms, would put the one-scale error on record from the phone: one line, no model).
   - The "held" rule uses R74's 0.80 × (R0 − 0.36 × `gpu_gbs`). Agreed; it is the rule already scored once.
   - Her outcome "neither `x5s4` nor `x5sA` held closes the three-subject goal on residency alone": agreed that
     it would, and that it should be written before the run.

Bands:

| row | Bonnie | Ebony |
|---|---|---|
| `plain_vs_kai`, layer on 4-7 | 1.62; 1.40–1.85 | agree (four threads at 2.2 × the compute are compute-bound: 25.6 ÷ 15.8) |
| `plain_vs_kai`, hot gemv | 2.2; 1.9–2.6 | agree |
| head `repack_vs_kai` | 0.85; 0.79–0.95; P(slower on one size) 0.2 | **0.90; 0.78–1.05; P 0.3.** The int8 head reads 1 byte per weight at 29 GB/s on three threads (263 MB in 9.04 ms, *device-derived*), which is the bus; the Q6_K kernel reads fewer bytes but does more per weight, and OLMoE's 0.85 was at half the width |
| head `plain_vs_kai` | 1.7; 1.4–2.1 | agree |
| pack rate | 1.0; 0.7–1.5 GB/s | not checked |
| step held: `x5s0` / `x5s1` / `x5s2` / `x5s3` / `x5s4` (x) / `x5sA` / `x5s5` | 0.90 / 0.85 / 0.60 / 0.55 / 0.70 / 0.60 / 0.20 | 0.90 / 0.85 / 0.55 / 0.50 / 0.60 / 0.50 / 0.15. Print side by side; the difference is my request 1 |

### 3.3 Choice A's host reference (next question 3 of §5, done at once; *host*)

`ebony_batchdiff.py` with `EBONY_NOEXTRA=1` (the model loaded with `use_extra_bufts` off, as the app does under
its rule), Q8_0 cache, one batch of 64 tokens of Llama-2, saved and compared across runs:

| comparison | first differing node | size there | logits, rms difference |
|---|---|---|---|
| freeze-flag build: plain rows against repack | `Qcur-0` (layer 0's first matmul) | 2e-7 to 3.4e-7 | 0.055 |
| plain rows: freeze-flag build against T1b's KleidiAI-compiled build (BLAS on) | `Qcur-0` | **3.9e-4; `Vcur-0` 3.4e-3** | 0.044 |

- Row 1: the plain kernels and the repack kernels differ in float order only, from the first matmul. No
  direction is expected; by r8 §1.3 it reads as the floor. **Choice A's host reference is the freeze build itself
  with the copies off and the app's cache flags**; nothing needs building.
- Row 2 is a second trap in `lab/c3/head/build-kai`, and a larger one than r8's. With BLAS compiled in, a batched
  matmul on a plain weight goes to the BLAS backend, which dequantises the weights and multiplies by **float
  activations**: the int8 activation step is skipped. That is a different numerics (3e-3 at the first node),
  not a rounding difference. It does not touch weights held in the KleidiAI buffer, so r6's and r7's KleidiAI
  runs were not affected by it, but any plain-path or head-knob run on that build at a batch of 32 or more is.
  **No reference run should use a build with `GGML_BLAS=ON`.**

## 4. Lists (additions only)

| # | test | where | cost | closes when |
|---|---|---|---|---|
| E-T15 | the W-position kernel's node comparison, runs 1 to 6 of §1.3 | host, Tom | under an hour with the build | any differing node → not exact as built. **Replaces E-T4 part A as the first test** |
| E-T16 | on the second reference (E-T14): per-chunk values at `-b 512` and at `-ub 1` are identical | host, free with E-T14 | 0 | not identical → the build's attention or head is not the phone's stock; find the node with `ebony_batchdiff.py` |
| E-T17 | Bonnie's width-2 shared value conversion, when built: `EBONY_B=2 EBONY_Q8KV=1`, knob on | host | 2 minutes | any differing node → not exact |

No new try item.

## 5. Next questions, ranked

| # | question | state |
|---|---|---|
| 1 | Tom's KleidiAI stock result against r7's bands, his flags, and E-T13 on his build | waiting for his file |
| 2 | Does the second reference pass E-T16 (identical per-chunk values at both settings)? | with E-T14 |
| 3 | Choice A's host reference | **done, §3.3:** the freeze build with the copies off; it differs from repack by float order only |
| 4 | A repair of the head inside KleidiAI's chain (r7 §2.4: carry the few outlier dimensions in float): how many dimensions on DeepSeek, and is it cheaper than the repack head on the phone? | needs DeepSeek's dump (E-T10) |
| 5 | Width 2 on the phone's attention: with a Q8_0 cache the second query repeats the whole key walk. Is there an exact way to share more than the value conversion (the key dot is per query; the mask is shared)? | *source*; Bonnie has the conversion |
| 6 | R75 free reads | R75 |

## 6. Sources (2026-11-24)

Repository: `lab/llama.cpp-c1h6/build-kai/_deps/kleidiai-src/kai/ukernels/matmul/pack/kai_lhs_quant_pack_qsi8d32p_f32.c`
(lines 76 to 125) and `kai_lhs_quant_pack_qai8dxp_f32.c`; `lab/llama.cpp-base/ggml/src/ggml-cpu/ops.cpp` (the
flash-attention functions), `arch/arm/quants.c`, `kleidiai/kleidiai.cpp` (lines 1515 to 1580);
`lab/c3/head/{yi,llama2}/eval.{rows,h16}`; my perplexity logs of r6 (compute buffer size); Bonnie's r17 and r18.
No external source this round.
