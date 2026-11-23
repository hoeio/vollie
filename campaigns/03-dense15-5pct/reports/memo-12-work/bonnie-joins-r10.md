# c3 memo-12 work, round 10 (Bonnie): every campaign-3 perplexity figure and its path; KleidiAI against repack, read at source level

- **Date:** 2026-11-23. **Author:** bonnie, `claude-fable-5-2`. Follows [bonnie-joins-r9.md](bonnie-joins-r9.md). Not
  committed. No verdict here. **Ebony has not reviewed this file.**
- **Labels.** *device*, *host*, *estimated*, *source* as before. Triples are Yi / Llama-2 / DeepSeek.
- Nothing ran this round. §1 is a reading of the ledger, the memos and the logs' build records; §2 is source
  reading. No web source.

## 0. Answer first

1. **Every campaign-3 perplexity figure comes from one build: `lab/llama.cpp-c3/build`, KleidiAI off, repack on,
   `armv8.2-a+dotprod+fp16`.** That holds for the freeze, every screen, the full-set checks and every identity
   check. Not one figure is on the phone's path.
2. **What needs a second reading on a KleidiAI build:** the six served stock values (they define the second
   reference); Tom's 0006 emulation when it reports (it is the one live numerics lever); and every "identical
   per chunk with the knob unset" check of a patch that will ship in a model-running kit. **What does not:** the
   F16 values and ceilings, the Q4_0-to-Q8_0 ratios, and the screens of closed or parked levers.
3. **The 0006 emulation cannot run on a KleidiAI build as written** (*source*, its README): it patches ggml's
   generic matmul and the repack type. KleidiAI takes the Q4_0 weights first and stores them in its own layout.
   A second reading needs a port, or a knob that makes KleidiAI decline the tensors.
4. **For Q4_0 the two paths are almost the same arithmetic** (*source*). Same int8 activations per block of 32
   with the same scale, same exact integer dot, same three float steps in the same order. I found **one**
   difference: rounding of an activation that falls exactly half-way between two int8 values (KleidiAI rounds
   away from zero, ggml to even). That is a few values per token and has no direction.
5. **The head is the one-sided part.** KleidiAI re-quantises the Q6_K head to int8 per row; that only adds
   error. Campaign 2 measured it on OLMoE: +0.03 % (*host*, R61 rows 1 against 2).
6. **So I revise r9's band down:** expected shift for our subjects **+0.03 %, band −0.05 to +0.15 %**
   (*estimated*), not +0.3 % (−0.3 to +1.0). r8's first guess was nearer. **But Granite's +0.57 % in R61 is not
   explained by anything I read**: its KleidiAI share was Q4_0 only, the head was not taken. Until that is
   explained I keep P 0.25 that a dense subject moves by more than 0.2 %. The 40-chunk run of r9 §3 decides.

## 1. Campaign-3 perplexity figures and the path each came from

All *host*, Apple M4 Pro. "The lab build" = `lab/llama.cpp-c3/build` at `26f78f925` (base `620d2225c` + the
campaign-2 series), configured `-DGGML_CPU_KLEIDIAI=OFF -DGGML_CPU_ARM_ARCH=armv8.2-a+dotprod+fp16
-DGGML_METAL=OFF -DGGML_NATIVE=OFF -DGGML_BLAS=OFF` (memo-02 lines 10 to 13; `CMakeCache.txt` read today). On
that build Q4_0 runs ggml's repack 4x4 dotprod kernels.

| figures | values | source | tool on the lab build | kind | second reading on a KleidiAI build? |
|---|---|---|---|---|---|
| S3 freeze, served Q4_0, full set | `-b 512` 8.3875 / 7.7435 / 8.7311; `-ub 1` 8.3855 / 7.7442 / 8.7306 | `results.csv`; `lab/logs/p/ppl-c3f-*` | `llama-perplexity` through `tools/ppl.sh` | absolute; **they define the budgets** | **yes.** These are the second reference campaign 2's rule asks for (r9 §1) |
| S3 freeze, F16, full set | 8.1773 / 7.4932 / 8.4115; ceilings 9.8128 / 8.9918 / 10.0938 | the same | the same | absolute | no. F16 weights are not taken by KleidiAI without SME, so the phone's build runs them as the lab build does (*source*, `supports_op`, `get_tensor_traits`) |
| budgets | 8.8069 / 8.1307 / 9.1677 (`-b 512`); 8.8048 / 8.1314 / 9.1671 (`-ub 1`) | `results.csv` | arithmetic on the freeze | absolute | no: rule 3, they stand |
| Q4_0 ÷ Q8_0 and Q4_0 ÷ F16 ratios, 40 chunks | Mistral 1.0195, Qwen2.5 1.0357 (memo-02); pool 1.01 to 1.04 (memo-06, memo-07) | memo-02 §0; `tom-screens.md` | `llama-perplexity` | ratio; file choice | no. The files are chosen and frozen |
| column sparsity frontiers (H3), 40 chunks, both batch settings | Yi 40 %; ChatGLM3, DeepSeek, Llama-2 20 %; Granite-7B 15 %; cells +1.55 to +5.73 % | `tom-screens.md`; `lab/c3/logs/h3-*` | `tools/c3_colskip.py` with the lab build's libraries | delta against same-build stock | no. The column kernel was killed by R69 (rule A, reference and Yi) and K1 is parked |
| k,v-dense rule, 40 chunks | Llama-2 and DeepSeek 40 % inside; cells in `tom-split.md` | memo-07 work | the same | delta | not now: K1 is parked (R72). **Yes if it is revived** |
| k,v-dense rule, full set, `-b 512` | Llama-2 8.0065 (+3.40 %); DeepSeek with BOS 9.1480 (+3.55 %) | `tom-k1.md` §6; `lab/c3/logs/full-*` | the same | delta and absolute | as above. Note the DeepSeek value is on the BOS file, which the owner later ruled out |
| `--no-repack` (Y1c), 40 chunks | −0.014 % Mistral, +0.017 % Qwen2.5 | memo-05 T3 | `llama-perplexity -nr` | delta | no: closed for width 2 (r6) |
| `down`-only sparsity (E1) and gate rule (E2), 40 chunks | E1 60 %: +3.41 / +1.92 % and +3.35 / +2.32 %; E2 closed | memo-05; `lab/c3/logs/e1e2-*` | `c3_colskip.py` | delta | no: not in any stack |
| block sparsity, sublayer and FFN skip, four-row groups | closed or at most 3 to 5 % of bytes | memo-02 §8, memo-06, memo-11 | the same | delta | no: closed |
| Q3_K re-encode (memo-11 bytes) | +0.8 % to +5.3 % against Q4_0 | memo-11 | `llama-quantize`, `llama-perplexity` | delta | no: closed |
| **0006 emulation of the GPU gemv** (the lane's numerics) | **not reported yet** (`tom-numerics.md` does not exist) | `patches/llama.cpp-c3/README.md` | a patched lab build | delta against same-build stock | **yes, and it needs work first** (§0 item 3) |
| identity checks, knob unset (0002 d-rows, 0006) | "identical per chunk at `-b 512`, `-ub 32`, `-ub 1`" on Mistral, Llama-2, Yi | STATUS; the patches' README | `llama-perplexity` | identity | **yes for anything that ships in a kit that loads a model.** No kit has loaded one yet, so nothing shipped rests on them |
| head top-1 and tokens per pass | 0.379 / 0.497; 1.33 / 1.45 | `tom-head.md` | a dump tool on the lab build; training dumps on Metal | not perplexity | no for the budget (a guess is verified exactly). The hit rate on the phone's path is unmeasured; it should differ by noise only |
| C5 (gate-row rule) and C7 (head screen), used in every stack | I found no three-subject perplexity figure for either | memo-09 §4 lists them as "numerics" | – | – | they need their first screens before any kit; run them on a KleidiAI build from the start |

Two things follow for the books:
- `results.csv`'s `build` column says "stock llama.cpp" for the freeze rows. It should name the build option
  (KleidiAI off), as campaign 2's R61 rows do after its correction. Annie's, if the lead agrees.
- A lever's per-cent delta moved little between builds in R61 (OLMoE 0.10 and 0.00 points; Granite 0.71 and 0.46
  points, on the favourable side; *host*). So a delta screened on the lab build is a fair first screen; it is
  not the reading a device round rests on.

## 2. KleidiAI against ggml's repack path for Q4_0, at source level

Read: KleidiAI v1.31.0 `kai_lhs_quant_pack_qsi8d32p_f32.c` (the activation routine the dotprod entry uses) and
the two kernels' assembly (r5); ggml `arch/arm/quants.c` (`quantize_row_q8_0`), `arch/arm/repack.cpp`
(`ggml_quantize_mat_q8_0_4x4`, `ggml_gemv_q4_0_4x4_q8_0`, `ggml_gemm_q4_0_4x4_q8_0`).

| step | KleidiAI | ggml repack | same? |
|---|---|---|---|
| activation block | 32 values | 32 values | same |
| scale | `abs_max / 127` in float | `amax / 127` in float | same |
| reciprocal | `1.0F / scale`, 0 if the scale is 0 | `1.0f / d`, 0 if d is 0 | same |
| stored scale | rounded to half | rounded to half | same (both quantise with the float scale and multiply back with the half one) |
| rounding to int8 | `roundf(x × rep)`: ties away from zero | `vcvtnq_s32_f32`: ties to even | **differs at exact ties only** |
| weights | nibbles XOR 0x88, used as `<< 4` and `& 0xF0` | the same | same |
| integer dot | 8 SDOT into a zeroed sum per block | 8 `vdotq_laneq` into a zeroed sum | same, exact |
| float per block | FMUL (weight scales × activation scale), SCVTF with 4 fractional bits, FMLA | `vmulq_f32` of the two scales, `vcvtq_n_f32_s32(…, 4)`, `vfmaq_f32` | same steps, same order |
| after the last block | FMAX and FMIN against ±FLT_MAX | none | differs only for an infinite sum |
| block order | ascending | ascending | same |
| batch kernel | 16-row and 4-row tiles in assembly | an assembly kernel of the same shape | not compared line by line; each path's batch 2 equals its own batch 1 bit for bit (*host*, Tom's T1b) |
| Q6_K head | dequantised, re-quantised to int8 with one scale per row | kept as Q6_K (8x4 layout) | **differs: a second quantisation** |

- **Ties.** An exact tie needs `x × rep` to land on a half. With 24 bits of mantissa and magnitudes up to 127
  that is about one value in 100,000 (*estimated*); a token has about 1.3 million activation values across its
  matmuls, so about ten int8 values per token differ by one step. On half of the ties ggml's rule already
  rounds the magnitude up, so KleidiAI's value is one step larger in magnitude on about five values per token.
  That is a bias in name only.
- **Clamp.** It matters only where ggml would produce an infinity. Not a perplexity term.
- **So for the Q4_0 body the expected shift is noise around zero.** OLMoE with the head kept out of KleidiAI
  showed exactly that: −0.012 % and +0.022 % (*host*, R61 row 2 against the build without KleidiAI).
- **The head is one-sided.** Re-quantising can only add error to the served head. Its size by resolution: int8
  with a per-row maximum against Q6_K's 6 bits with a scale per 16 values; similar step sizes on ordinary rows,
  coarser on rows with outliers. Measured once: OLMoE +0.028 % and +0.033 % (*host*, R61 row 1 against row 2).
  Our heads are 64,000 / 32,000 / 102,400 rows against OLMoE's about 50,000; I expect the same order.

### 2.1 What this reading does not explain

**Granite moved by +0.57 % and +0.50 % in R61**, and its KleidiAI share was 541 MiB of Q4_0 with the tied head
not taken (*host*; campaign 2 `STATUS.md` line 143, model card). By the table above that should have been noise.
I found no explanation in campaign 2's memos. Candidates, none checked:
1. Granite is a hybrid model; some of its matmuls may reach KleidiAI in a shape the dense subjects never use
   (three-dimensional activations, small row counts), where a different branch packs or chunks differently.
2. Ties are not rare on its activations (values that are exact fractions of the block maximum).
3. The batch kernels differ in a float step I did not compare. This would not explain the `-ub 1` value.
4. Something outside Q4_0 changes when KleidiAI is compiled in (a different default somewhere in the build).

Because of this the band keeps a tail: **P(shift within ±0.1 % on all three) 0.65; P(some subject beyond ±0.2 %)
0.25.**

### 2.2 A check that would settle the Q4_0 part without a perplexity run

Ebony's shadow check (her r5 §3.3 part A) compares two kernels on real activations inside
`compute_forward_qx`. The same hook with ggml's repack gemv as the second kernel, on the KleidiAI build, counts
how many outputs differ and by how much, per tensor, over one prompt. If the count is at the tie rate, §2 holds.
Minutes on the host once her hook exists; Tom's.

## 3. Next questions, ranked

1. **Why did Granite move?** One host comparison on the campaign-2 file: the per-tensor shadow check of §2.2, or
   simply the two builds' per-chunk values on 4 chunks with the op list of what KleidiAI took. It bounds the
   tail of §0 item 6 and is the only thing here that could make the phone's stock differ by tenths of a per
   cent. Tom's; I can prepare the op-shape reading from Granite's graph in source.
2. **Port or bypass for the 0006 emulation on a KleidiAI build** (§0 item 3): which is less code, and what the
   mixed arm (CPU rows on KleidiAI, GPU rows emulated) needs. Source reading, mine.
3. **First screens for C5 and C7 on the three subjects.** Both sit in every stack at centre and neither has a
   perplexity figure here. If C5 does not fit the budget the stacks lose 7 % of body.
4. **The head knob** (r9 question 3): with the head's shift at about +0.03 %, keeping it in KleidiAI costs
   almost no budget and 131 to 421 MB of buffer; sending it to repack changes its kernel. Price both.
5. **Cold read rate of the served file on the phone** (a kit line).
6. **Fixed share: which C to tune for**, after R75.

## 4. Limits

- §1 lists figure families, not every cell; the logs named hold the cells. I read build records and memo text,
  not every log header.
- "No figure found" for C5 and C7 means I did not find one in the memos and logs I searched; Tom may hold one.
- §2 compares the batch-1 kernels step by step and the batch kernels only through Tom's identity result.
- The tie rate is an estimate from mantissa width, not a count.
- §2.1's candidates are guesses.

## 5. Sources (2026-11-23)

Repository: `campaigns/03-dense15-5pct/results.csv`; `reports/memo-02-host-screens.md` (lines 10 to 36),
`memo-05-y1-column-kernels.md`, `memo-05-work/tom-host.md` (§4, §7, §8), `memo-06-work/tom-screens.md`,
`memo-07-work/tom-split.md`, `memo-08-work/tom-k1.md` (§6), `memo-09-gap.md` (§4), `memo-09-work/tom-head.md`,
`memo-11-bytes.md`; `patches/llama.cpp-c3/README.md` (0006); `lab/llama.cpp-c3/build/CMakeCache.txt`;
`lab/c3/logs/` and `lab/logs/p/` (file names); campaign 2 `STATUS.md` (lines 141 to 150), `reports/README.md`
(R61), `models/granite-4.0-h-tiny-stock-mi.md`, `models/olmoe-1b-7b-0125-instruct-mi.md`.
Source trees: KleidiAI v1.31.0 `kai/ukernels/matmul/pack/kai_lhs_quant_pack_qsi8d32p_f32.c`;
`lab/llama.cpp-c3/ggml/src/ggml-cpu/arch/arm/quants.c` and `arch/arm/repack.cpp`;
`lab/llama.cpp-c3r32/ggml/src/ggml-cpu/kleidiai/kleidiai.cpp` and `kernels.cpp`.
