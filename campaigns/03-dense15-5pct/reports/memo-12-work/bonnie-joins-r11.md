# c3 memo-12 work, round 11 (Bonnie): why Granite moved in R61, what that predicts for the three subjects, and the 0006 emulation on a KleidiAI build

- **Date:** 2026-11-23. **Author:** bonnie, `claude-fable-5-2`. Follows [bonnie-joins-r10.md](bonnie-joins-r10.md). Not
  committed. No verdict here. **Ebony has not reviewed this file.** A second Tom is running the 40-chunk KleidiAI
  stock runs (r9 §3); I have not seen `tom-kai-stock.md` and nothing here uses it.
- **Labels.** *device*, *host*, *estimated*, *source* as before. Triples are Yi / Llama-2 / DeepSeek.
- **New *host* numbers, all arithmetic on existing files, no llama.cpp run:**
  - per-chunk comparison of R61's logs with the earlier logs (§1.2);
  - Granite's tensor list from its GGUF header (§1.1);
  - the head's weight re-quantisation error per model ([bonnie_head_requant.py](bonnie_head_requant.py));
  - a simulation of the head on both paths over Tom's hidden-state dumps for Llama-2 and Yi
    ([bonnie_head_paths.py](bonnie_head_paths.py)). It is a **proxy**, not a perplexity.
- No web source. Nothing ran on the phone.

## 0. Answer first

1. **Granite's head was in KleidiAI in R61.** Campaign 2's text says the tied head was not taken; the buffer
   says otherwise: Granite's Q4_0 tensors are 392.9 MiB, the head as int8 per row is 148.1 MiB, and the log's
   KleidiAI buffer is 541.08 MiB (*host*). (My own campaign-2 memo-24 already noted this; r10 repeated the wrong
   line.) So Granite's +0.57 % is not a Q4_0 puzzle. **Correction to r10 §2.1.**
2. **The head on KleidiAI's path differs in two ways, and r8 named only one.** Weights: int8 with one scale per
   row. **Activations: asymmetric int8 with one scale and zero point for the whole hidden vector** (*source*,
   `kai_lhs_quant_pack_qai8dxp_f32.c`). ggml quantises the same vector in blocks (32 or 256 values). A hidden
   state with a few very large components loses most of its resolution under one scale.
3. **That predicts a per-model shift, and Yi is the exposed one** (*host*, proxy on the 32-prompt set):

   | | hidden state range ÷ rms, median | activation error ÷ rms: ggml / KleidiAI | change of −log p(greedy token), KleidiAI against ggml |
   |---|---|---|---|
   | Llama-2 | 10.9 | 0.9 % / 1.2 % | **+0.00008** |
   | Yi | 41.3 | 2.4 % / 4.7 % | **+0.00536** (activations +0.0047, weights +0.0006) |
   | DeepSeek | no dump exists | – | not predicted |

   A change of 0.005 in mean log-probability is about half a per cent of perplexity if it carries over to the gate
   text. **My prediction for the stock run: Yi +0.3 to +1.0 %; Llama-2 within ±0.05 %; DeepSeek unknown, +0.0 to
   +1.0 %.** This replaces r10's single band.
4. **The Q4_0 body is not the cause, on the evidence there is.** OLMoE with its head out of KleidiAI: mean
   per-chunk change −0.012 %, 51 % of chunks up, t −0.3 (*host*). Granite: +0.573 %, 62 % up, t +6.9.
5. **Cheapest checks for Tom, in order** (§1.5): read Yi against Llama-2 in the run he is doing (free); one more
   40-chunk run of Yi with the head sent back to the repack type (minutes, needs the kit's tree); for Granite,
   dump 32 prompts of hidden states with his existing tool and run my script (no build).
6. **If Yi moves as predicted, the head knob stops being a detail.** With the head in KleidiAI Yi's phone-path
   stock uses about a tenth of the +5 % budget before any lever. `VOLLIE_HEAD2_REPACK=1` removes it; its price in
   head milliseconds is not measured.
7. **The 0006 emulation on a KleidiAI build:** the all-GPU arm needs no port. A two-line decline in KleidiAI's
   `supports_op` makes it run, and its result should equal the lab build's per chunk, because the emulated
   arithmetic uses neither CPU path. The **mixed arm** (CPU rows on KleidiAI, GPU rows emulated) needs about 60
   to 80 lines inside `compute_forward_qx`. Design in §2.

## 1. Why Granite moved

### 1.1 What KleidiAI held (*host*, GGUF header of `lab/granite/granite-Mi.gguf` and R61's load log)

| tensors | type | size |
|---|---|---|
| `ssm_in` (1536 → 6448) × 36, `ssm_out` (3072 → 1536) × 36 | Q4_0 | 191.3 + 91.1 MiB |
| shared-expert `gate`, `up`, `down` | Q4_0 | 33.8 + 33.8 + 29.5 MiB |
| attention q, k, v, o × 4 layers | Q4_0 | 13.6 MiB |
| **all Q4_0** | | **392.9 MiB** |
| `token_embd` (1536 × 100,352), tied head | Q6_K | 120.6 MiB in the file; **148.1 MiB as int8 with 12 bytes per row** |
| **sum** | | **541.0 MiB**; the log prints `CPU_KLEIDIAI model buffer size = 541.08 MiB` |

- Every Q4_0 shape has inputs in multiples of 32 and rows in multiples of 4. Nothing odd about shapes.
- The same arithmetic holds for OLMoE: 242.83 MiB in the log = 144.0 MiB of Q4_0 + 98.8 MiB of int8 head.
- So "the tied head is not taken" (campaign 2 `STATUS.md` line 143) is wrong for Granite. The tied tensor is
  used twice; the copy the head multiplies by went to KleidiAI.

### 1.2 What R61's own logs say (*host*; my arithmetic on the running values)

Per-chunk log-likelihood recovered from the running perplexity, the build with KleidiAI minus the build without:

| pair, `-b 512` | chunks | mean change | t | chunks that went up | spread per chunk |
|---|---|---|---|---|---|
| Granite, stock | 565 | **+0.573 %** | +6.9 | 62 % | 2.0 % |
| OLMoE, head kept out of KleidiAI | 560 | −0.012 % | −0.3 | 51 % | 1.1 % |
| OLMoE, head in KleidiAI | 560 | +0.016 % | +0.3 | 53 % | 1.1 % |

- OLMoE's Q4_0 body through KleidiAI is noise around zero. So is its head.
- **The spread per chunk is large (1 to 2 %) even where the mean is zero.** That is not what ten rounding ties
  per token would give. Both models route tokens to experts by a top-k choice; a small numeric change flips a
  choice now and then and changes that token a lot. A dense model has no such switch, so I expect its per-chunk
  spread to be far smaller. Tom's run shows it: the spread is in his logs.
- Caveat: R61's runs carried the kit's environment (`GGML_TINY_OP`, `GGML_EXPERT_IO`, `GGML_C2_IO`,
  `VOLLIE_SAMPLER_BYPASS`) and the older runs did not. They were shown identical on 4 chunks only. This
  comparison cannot separate them from KleidiAI; OLMoE's zero mean says they are harmless there.

### 1.3 The mechanism (*source*)

KleidiAI's 8-bit chain, which takes Q8_0 and Q6_K weights (`kernels.cpp` lines 935 to 980, `kleidiai.cpp` lines
500 to 507 and 1529 to 1596):

| | KleidiAI, 8-bit chain | ggml |
|---|---|---|
| weights | dequantised, then int8 with one scale per row (max ÷ 127) | the served Q6_K (6 bits, a scale per 16 values) |
| activations | **one scale and one zero point per vector**: 255 steps over [min(0, min x), max(0, max x)] | blocks: one scale per 256 values for Q6_K's dot products, per 32 for Q8_0's |
| product | exact integer, corrected for the zero point | exact integer |

- For the Q4_0 chain the activation block is 32 values in both (r10 §2). The per-vector rule is only for the
  8-bit chain: `kleidiai_get_block_args` gives it no block length.
- **A vector with a few huge components** has a range many times its rms. One scale over that range leaves the
  ordinary components a handful of levels. ggml's blocks confine the damage to the block that holds the outlier.
- The weight side is small and even across models (*host*, `bonnie_head_requant.py`, 6,000 sampled rows each):
  added error ÷ row rms 1.2 % (OLMoE), 0.8 % (Granite), 0.9 / 1.0 / 1.0 % (Yi / Llama-2 / DeepSeek). It does
  not separate Granite from OLMoE: OLMoE's is the larger one. So Granite's shift is not the weights.
- **By elimination the leading candidate for Granite is the activation rule** on its final hidden state. I
  cannot test it: no hidden-state dump of Granite exists. P 0.6 (judgement). The rest of the probability is with
  causes specific to the hybrid model that I did not find, or with the environment caveat of §1.2.

### 1.4 What it predicts for our subjects (*host*, proxy)

`bonnie_head_paths.py` takes the hidden states Tom dumped for his head work (the vector the head reads, every
decoded row of the 32-prompt set) and the served head, and computes the logits three ways.

| | Llama-2 | Yi |
|---|---|---|
| rows; head | 3,638; 32,000 × 4,096 | 3,563; 64,000 × 4,096 |
| self-check: argmax of the float head = the dumped greedy token | 0.9995 | 0.9921 |
| hidden state range ÷ rms: median, 90th percentile, max | 10.9, 19.4, 88.9 | **41.3, 52.5, 64.2** |
| activation error ÷ rms, ggml's blocks of 256 (median) | 0.86 % | 2.39 % |
| activation error ÷ rms, KleidiAI's one scale (median, 90th percentile) | 1.24 %, 2.20 % | **4.67 %, 5.95 %** |
| mean −log p(greedy token): float / ggml / KleidiAI | 0.12150 / 0.12163 / 0.12171 | 0.24615 / 0.24484 / 0.25020 |
| KleidiAI − ggml | **+0.00008** | **+0.00536** |
| of which weights only / activations only | +0.00006 / +0.00003 | +0.00064 / +0.00472 |

- Yi's final hidden state carries components about 40 times its rms on every row. Llama-2's is far milder.
- **What the proxy is and is not.** It scores the model's own greedy token on chat prompts, with float matmul
  and no kernel. The gate scores the text's token on wikitext. The direction and the ranking should carry over;
  the size need not. That is why the prediction is a band: Yi +0.3 to +1.0 %.
- DeepSeek has no dump (its head was never trained). Its head has the most rows; its hidden state is unknown.
- This is the head only. For the body the expectation stays noise around zero (§1.2, r10 §2).

### 1.5 The cheapest checks that tell the candidates apart (for Tom)

| # | check | cost | says |
|---|---|---|---|
| 1 | In the 40-chunk runs under way: the shift of Yi against Llama-2, and the per-chunk spread | none | Yi up by tenths of a per cent and Llama-2 flat = the head's activation rule. Both flat = my proxy overstates. Both up = something in the body |
| 2 | Yi, 40 chunks, both batch settings, on a KleidiAI build of the kit's tree with `VOLLIE_HEAD2_REPACK=1` | one build of `lab/llama.cpp-c3r32` with KleidiAI on (the knob is only in that tree), 9 minutes of runs | the head goes back to the repack type. If the shift disappears, it was the head. This is also the configuration a kit could ship |
| 3 | Dump Granite's hidden states on the 32 prompts with `tools/c3_head_dump.cpp` (model-generic), add the file and the tensor name `token_embd.weight` to my script | minutes, no build | range ÷ rms of Granite's final state and the proxy. A large value closes R61's open point |
| 4 | The same dump for DeepSeek | minutes | the missing prediction of §0 item 3 |
| 5 | Only if 1 shows the body moving: Ebony's shadow hook with the repack gemv as the second kernel (r10 §2.2) | her hook first | per-tensor counts of differing outputs |

## 2. The 0006 emulation on a KleidiAI build (next question 2)

### 2.1 How 0006 is built (*source*, the patch)

- It hooks the start of `ggml_compute_forward_mul_mat` in `ggml-cpu.c`: for a Q4_0 weight with float activations
  and `VOLLIE_X3_EMU` set, every output is computed from the **plain row-major Q4_0 blocks** as the GPU kernel
  does (`(q − 8) × d` rounded to half, float activations, the kernel's reduction order).
- So that the data stay plain, the repack type declines the tensor when the knob is on (one changed condition in
  `repack.cpp`).
- `VOLLIE_X3_EMU_Q6K` does the same for the head with upstream's Q6_K kernel.

### 2.2 Why it does nothing on a KleidiAI build

KleidiAI's type is asked before the repack type and takes every Q4_0 matmul weight. The data are then in
KleidiAI's layout, and the op is run by KleidiAI's own `compute_forward`, which is tried before the generic
function 0006 hooks. The knob is silently without effect: a perplexity run would return stock.

### 2.3 The all-GPU arm: bypass, no port

- **Change:** in `supports_op` (`kleidiai.cpp` line 1788 on) decline a Q4_0 weight when `VOLLIE_X3_EMU` is on,
  and a Q6_K weight when `VOLLIE_X3_EMU_Q6K` is on. Two conditions, the mirror of 0006's change in `repack.cpp`.
- **Expected result:** per-chunk values equal to the lab build's emulation, because with both types declining
  the tensors are plain in both builds and the emulated arithmetic is the same code. A 4-chunk comparison at
  `-b 512` and `-ub 1` shows it. If they are equal, **the all-GPU arm needs no second full run**; only its
  reference changes (Tom's KleidiAI stock values).
- **One trap:** with `VOLLIE_X3_EMU` on and `VOLLIE_X3_EMU_Q6K` off, the head is on KleidiAI's path in this build
  and on the repack path in the lab build. Then the two builds differ by the head (§1), not by the emulation.
  Compare like with like: the head emulated in both, or the head stock in both and read against each build's own
  stock.

### 2.4 The mixed arm: a small port

A lane computes some rows of a matrix on the GPU and the rest on the CPU. The arm that matches a design with a
fixed share (r6 §3) is: rows [0, r) by the phone's CPU kernel, rows [r, n) emulated.

- **Where:** inside `compute_forward_qx`, after the stock chunk loop has filled `dst` for all rows. Thread 0 then
  overwrites rows [r, n) with emulated values.
- **What it needs:** the plain Q4_0 row for each emulated row. KleidiAI's packed group holds four rows per block
  as four half scales followed by sixteen 4-byte pieces, each byte XOR 0x88 (r9 §4.1; my host harness checks
  exactly this layout). An unpack of one group back into four `block_q4_0` rows is about 20 lines; the existing
  `vollie_x3emu_row_ref_t` then runs unchanged on the float activations in `src1`.
- **Knobs:** `VOLLIE_X3_EMU=1` and `VOLLIE_X3_EMU_SHARE=<fraction>`; emulated rows = round(share × n ÷ 128) × 128
  (the kit's rule), taken from the end. Share 1.0 must reproduce §2.3's arm bit for bit: that is the port's
  self-check.
- **Size:** 60 to 80 lines with the knob parsing. Host only, slow, never in a kit.
- **Is the port needed at all?** For a first screen, no: the CPU rows on KleidiAI and on the repack path are the
  same arithmetic up to rounding ties (r10 §2), so a mixed arm on the lab build is a fair screen. For the
  reading a device round rests on, yes: the rule is the phone's own path.
- **When:** after R75, when a lane design has a share. Not before.

### 2.5 What the check must then cover (with r6 §3 and r10 §1)

| arm | build | settings | read against |
|---|---|---|---|
| all rows emulated, head stock | KleidiAI build with the decline | 40 chunks, then full set, `-b 512` and `-ub 1`, two subjects at least | that build's stock (Tom's run) × 1.05 and the absolute budget |
| the same with the head emulated (`VOLLIE_X3_EMU_Q6K`) | the same | 40 chunks | the same; only if a head on the lane is planned (r4 §2) |
| mixed, at the design's share | KleidiAI build with the port | full set, both settings, all three subjects | the same |

## 3. Next questions, ranked

1. **The head knob for campaign 3** (was question 4; now first if Yi moves): head in KleidiAI (int8 weights,
   one activation scale) against head in the repack type (`VOLLIE_HEAD2_REPACK=1`). Budget: up to about +0.5 %
   on Yi by my proxy. Speed: the repack Q6_K kernel's time for a 64,000-row head on the phone is not measured
   (R72 timed the head in KleidiAI only: 9.04 / 4.40 / 13.88 ms). Buffer: 263 / 131 / 421 MB leave the KleidiAI
   buffer and the side file. Mine to price from source and R72; a probe cell would settle the speed.
2. **DeepSeek's and Granite's hidden states** (checks 3 and 4 of §1.5).
3. **Does C7 (the head screen) interact with this?** A two-stage head scores a shortlist exactly; on which path
   are the two stages computed, and does the first stage inherit the one-scale activations? Source reading.
4. **First screens for C5 and C7** (r10 question 3).
5. **Cold read rate of the served file on the phone**; **fixed share: which C**, after R75.

## 4. Limits

- §1.3's conclusion for Granite is by elimination and a mechanism that is shown on Yi, not on Granite.
- §1.4 is a proxy: greedy-token log-probability on 32 chat prompts with float arithmetic. I simulate KleidiAI's
  quantisers from their source; the kernel's own integer arithmetic (zero-point correction, the float steps) is
  not simulated. The activation rule ignores KleidiAI's handling of ties and of the nudged zero point beyond
  what the source line shows.
- Tom's dumps hold fp16 hidden states; Yi's self-check is 0.9921, so 0.8 % of its rows already disagree with
  the dumped token in float.
- §1.2 recovers per-chunk values from running perplexities printed to four decimals; late chunks are noisier.
- §2 is a design from the patch text; I did not build or run 0006.

## 5. Sources (2026-11-23)

Repository: `lab/granite/granite-Mi.gguf`, `lab/olmoe/olmoe-Mi.gguf`, `lab/c3/models/{yi15-6b,llama2-7b,deepseek7b}-q4_0.gguf`
(headers and head tensors); `lab/logs/r61/r4-granite-off-b512.log` and `.load.log`, `r1-…`, `r2-…`;
`lab/logs/ppl-granite-Mi.log`, `lab/logs/ppl-olmoe-Mi.log`; `lab/c3/head/{llama2,yi}/eval.rows`, `eval.h16`;
campaign 2 `STATUS.md` (lines 141 to 150, 349), `reports/README.md` (R61); `patches/llama.cpp-c3/0006-x3-emu.patch`
and `README.md`; `campaigns/03-dense15-5pct/reports/R72-result.md` (head times).
Source trees: KleidiAI v1.31.0 `kai/ukernels/matmul/pack/kai_lhs_quant_pack_qai8dxp_f32.c`,
`kai_rhs_pack_nxk_qsi8cxp_qsi8cx_neon.c`; `lab/llama.cpp-c3r32/ggml/src/ggml-cpu/kleidiai/kleidiai.cpp` (lines 493
to 510, 1529 to 1596, 1788 to 1860) and `kernels.cpp` (lines 935 to 980); `lab/llama.cpp-c3/ggml/src/ggml-cpu/repack.cpp`
(Q6_K kernels with Q8_K activations); `lab/llama.cpp-c3/src/models/mamba-base.cpp`, `granite-hybrid.cpp`.
