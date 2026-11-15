# Memo 21: unconventional speed levers, checked against sources and priced on both models

Agent model: claude-opus-5-5

Date 2026-11-15. Author: Bonnie (research). **Nothing here ran on a phone or on the host; no speed is measured
here.** It is reading, web research and arithmetic only: a perplexity queue was using the laptop's CPU. Labels:
*device* = read from a kit report (named each time), *host* = Apple M4 Pro, *estimated* = arithmetic, "source: <url>" =
a published claim that was not reproduced here. This memo proposes levers and experiments. It does not decide them:
verdicts are Clyde's, and the items marked "needs owner decision" are the owner's.

Device data used: `device/r22_2026-11-15.json` (R60, raw report; per-token profile lines of the 8 profiled runs, my
arithmetic), `device/r21_2026-11-14_summary.json` (R58), and campaign 1's cpu-bench rows (`docs/runtime-knobs.md`,
`GGML_CPU_BENCH` row, R37/R41). The tensor types were read from the GGUF headers of `lab/olmoe/olmoe-Mi.gguf` and
`lab/granite/granite-Mi.gguf` with a 40-line header reader of my own (*host*, no tensor data read).

## 0. Summary

1. **Where the milliseconds are, *device* (R60 profiled runs, fault-free tokens with at most 5 major faults; my
   medians):**

   | model, arm | token | `MUL_MAT` (incl. head) | head | `MUL_MAT_ID` (experts) | `SSM_SCAN` | rest | outside the graph |
   |---|---|---|---|---|---|---|---|
   | OLMoE off (pos 24, n = 9) | 49.8 | 10.9 + 0.7 wait | 3.9 | 31.1 + 0.3 | – | ≈ 6.8 | 0.66 |
   | OLMoE num, skip only (pos 23, n = 23) | 42.8 | 10.2 + 0.7 | 4.1 | 27.2 + 0.3 | – | ≈ 4.3 | 0.66 |
   | Granite off (pos 11/48) | 52.9 / 54.5 | 21.8-22.1 + 1.8 | 5.2-5.4 | 21.2-21.3 + 0.6 | 2.4 + 0.2 | ≈ 4.8 | 0.88-1.05 |
   | Granite num, skip 130 + head2 (pos 12/47) | 47.5 / 47.0 | 22.0-22.1 + 1.7 | 3.8-4.0 | 13.5-13.9 + 0.6 | 2.6-2.7 + 0.2 | ≈ 6.2 | 0.83-0.95 |

   The `head=` field is part of `MUL_MAT`, not added to it: the op times only add up to the token time that way.
   R58's "expert 27.8, dense 9.6, head 3.5" for OLMoE therefore means non-head dense ≈ 6.1 ms, at the byte rate
   (152 + 8 MB at 0.0411 ms/MB = 6.6 ms, *estimated*). Outside the graph means decode wall time minus the sum of the
   per-token graph times, divided by 32 (*device*, my arithmetic).
2. **The expert op is the only large compute pool left, and it is pure Q3_K kernel time.** Campaign 1's A78 bench
   (*device*, r13b, A78 at 2,112 MHz, 1 thread, cache-cold/hot 1.04) gives 156.9 ns per 1,536-weight row, which is
   102 ps per weight per core. OLMoE decodes 805 M expert weights per token (8 of 64 experts × 16 layers × 6.29 M;
   header). 805 M × 102 ps / 3 threads = **27.4 ms**, against **27.8 ms** measured (R58). Granite: 565 M weights →
   19.2 ms against 21.2 (R60, off). So the expert op runs at about 0.080 to 0.093 ms per MB, 2.0 to 2.3 × the
   dense byte rate (0.0411 ms/MB). Its byte floor is 15.2 to 18.2 ms on OLMoE without skip (346 MB at 22.8 to 19
   GB/s), 12.4 to 15.8 with skip `MASS=900`, and 6.0 to 7.7 ms on Granite at skip 130 (147 MB). **Compute pool
   (*estimated*): OLMoE 11 to 15 ms, Granite 6 to 8 ms.** Everything dense (attention, `ssm_in`/`ssm_out`, shared
   expert, head screen) already runs at the byte rate, so kernels cannot help there.
3. **The other large pool is residency.** In R60's hold runs the mean token (decode wall / 32) was ≈ 50 to 52 ms
   on OLMoE against 40.4 fault-free, and ≈ 51 against 47.3 on Granite (*device*, arithmetic): **fault excess ≈ 10
   to 11 ms (OLMoE) and ≈ 4 to 5 ms (Granite)** per token.
4. **What 25 tok/s needs.** The printed number is about 968.75 / (mean ms per token) (pos 23: 968.75 / 51.5 = 18.8,
   printed 18.74). So 25 tok/s means a mean of **38.75 ms**. With zero faults, the head fixed (−1.3 to −1.5 ms,
   *estimated*) and nothing else, OLMoE is at ≈ 39.0 + 0.7 = 39.7 ms (**24.4 tok/s**). It needs about 1 ms out of an
   11 to 15 ms compute pool. **Granite, with zero faults and its expert op all the way down at the byte floor, is at
   39.6 to 41.3 + 0.9 ms (23.0 to 23.9 tok/s).** It needs 2 to 3.5 ms more from the "rest" (≈ 6.2 ms of norms,
   convolution, state copies) and the waits (≈ 2.5 ms), which this memo does not price. That is the honest
   position with numbers: OLMoE needs residency plus a small compute lever, and Granite needs every pool at its
   floor plus something not yet found.
5. **Ranked shortlist** (§5): (1) opportunistic work-stealing helpers on the idle cores inside the expert op
   (bit-identical; OLMoE 0 to −8 ms, Granite 0 to −4); (2) cache-conditional routing, a declared numerics change
   that shrinks the decode working set (OLMoE −2 to −6 ms at R60's state and much more in low-cache states,
   Granite −1 to −2.5); (3) a LUT Q3_K kernel, either exact with int16 tables (0 to −6 / 0 to −3) or lossy with int8
   tables (−5 to −11 / −3 to −6, declared); (4) helper-thread L3 warm of the next dense op (0 to −2 / 0 to −3);
   (5) crumbs (outside-graph pipelining, 64 MiB `mlock`, 0 to −1 ms each).
6. **Dead, with numbers** (§6): XOR/POPCNT networks and sign-bit screens (head, router), GEMV→GEMM, early commit on
   the head screen (the exact stage is ≈ 0.1 ms), the Adreno 660, runtime VQ re-encoding, TEAL-style input sparsity
   on row-major quant blocks, exact bit-plane early termination (SiLU has no zero), cross-token output reuse, an
   exact nibble re-layout of Q3_K (+29 % bytes against residency), and LUT for byte-bound tensors. The gate-guided
   up-row skip is weak but not dead: −1.4 ms (OLMoE) for most of the remaining budget.

## 1. Facts checked before pricing

| fact | value | label / source |
|---|---|---|
| OLMoE-Mi types | attention q/k/v/o Q4_0 (16 × 4 × 4.19 M), `token_embd` Q4_0, experts gate/up/down **Q3_K** (3 × 922.7 MB), router `ffn_gate_inp` F32 (8.4 MB), `output.weight` **Q6_K** (84.5 MB, untied) | *host*, GGUF header |
| Granite-Mi types | `ssm_in` Q4_0 200.6 MB, `ssm_out` Q4_0 95.6, shared expert gate/up Q4_0, down **Q4_0 × 35 + Q4_1 × 5**, attention Q4_0 (4 layers), experts **Q3_K** (3 × 865.1 MB), router F32 15.7 MB, tied `token_embd` **Q6_K** 126.4 MB | *host*, GGUF header |
| OLMoE head on the phone | KleidiAI repacks `output.weight` to int8 with one scale per row: ≈ 103 MB (50,304 × 2,048 bytes), more than the 84.5 MB Q6_K; head op 3.8 to 4.2 ms ≈ 25 to 27 GB/s | Tom 2026-11-15 (STATUS item A); MB *estimated*; ms *device* R60 |
| compute threads | 3, **not pinned** (in `0001` only the cpu-bench thread is pinned, to cores 4 to 6) | *host*, patch reading (`sched_setaffinity` at lines 2254, 2464 of `0001-cpu-runtime.patch`) |
| per-core Q3_K kernel rate | A78 @ 2,112: 156.9 ns/row (K 1,536); Q4_0x4 78.6; int8 100.3. X1 @ 2,841 (r11): new 87.1. A55 @ 1,804 falling to 1,094 (r12, unpinned): ns/row ≈ 1.7 × the A78's, i.e. ≈ 0.58 × the A78's rate (weak datum) | *device*, campaign 1 (runtime-knobs `GGML_CPU_BENCH` row; `reports/memos/round13_plan.md` §C.1) |
| clocks in R60's profiled runs | cpu4 2,112 / cpu7 2,035 MHz | *device* R60 |
| `RLIMIT_MEMLOCK` | **64 MiB** soft and hard (67,108,864 B) | *device*, `memlock_rlimit` in `r22_2026-11-15.json` |
| SD888 caches | X1 1 MB L2; A78 512 KB L2 each; A55 128 KB L2 each; shared L3 4 MB; **3 MB system-level cache in front of the memory controllers** | source: [AnandTech 16271](https://www.anandtech.com/show/16271), [The Register](https://www.theregister.com/2020/12/02/qualcomm_snapdragon_888/), [Computerbase](https://www.computerbase.de/2020-12/qualcomm-snapdragon-888-hands-on/) |
| **CPU access to the 3 MB SLC** | AnandTech's measurements: "Qualcomm seems to have completely removed access of the CPU cluster to the SoC's system cache, as even the Cortex-A55 cores don't look to have access to it" | source: [AnandTech 16463 p.3](https://www.anandtech.com/show/16463/snapdragon-888-vs-exynos-2100-galaxy-s21-ultra/3) (quoted from the search excerpt; the page itself now redirects and could not be fetched) |
| DRAM | 4 × 16-bit LPDDR5-6400, 51.2 GB/s nominal | source: [AnandTech 16271](https://www.anandtech.com/show/16271) |
| measured DRAM rate for dense ops | 19 to 22.8 GB/s with 3 threads | *device*, campaign 1 R25/R31 |

**For the CPU, the usable on-chip capacity is the 4 MB L3** plus the private L2s. Every "resident in L3" argument
below uses 4 MB, not 7.

## 2. Part A: the owner's list

### A1. LUT inference with NEON `TBL` (T-MAC and relatives)

**What the sources say.** T-MAC (Wei et al., EuroSys 2025) replaces multiplication with bit-serial table lookups:
g = 4 activations give a 16-entry table that fits one NEON register for `vqtbl1q_u8`, and the cost grows linearly
with the weight bits. Its single-thread GEMV kernel speedups over llama.cpp are 11.2 × (1 bit), 5.8 × (2), **4.7 ×
(3)** and **3.1 × (4)**. End to end on a phone (OnePlus 12) it reports 10.19 against 8.24 tok/s at 4 bits (1.24 ×)
and 16.62 against 6.95 at 2 bits. **Its tables are quantised to int8** ("table quantization"), so it is **not
bit-exact**: kernel NMSE is 3.35e-3 against llama.cpp's 3.33e-3, and WikiText-2 PPL on Llama-2-7B-4bit equals
llama.cpp's to two decimals. The optional "fast aggregation" (`vrhadd`) is lossier (PPL 5.96 → 6.38). Source:
[arXiv 2407.00088](https://arxiv.org/html/2407.00088v2). The llama.cpp integration proposal (PR #10181) covered
Q4_0, TQ1_0, TQ2_0 and a new INT_N type, **with K-quants left as a TODO**. Its merge status could not be confirmed
(source: [llama.cpp PR 10181](https://github.com/ggml-org/llama.cpp/pull/10181), seen through a mirror). bitnet.cpp's
TL1 and TL2 are the same family for ternary weights only (source: [arXiv 2410.16144](https://arxiv.org/pdf/2410.16144),
[arXiv 2502.11880](https://export.arxiv.org/pdf/2502.11880)).

**Correction to the brief.** "No gain at 3 to 4 bits" is contradicted by T-MAC as published. Its baselines,
though, were llama.cpp's 2024 kernels with lossy tables. Against Vollie's in-register Q3_K kernel and with exact
tables, the brief's estimate is roughly right (below).

**Operation counts on the A78** (2 NEON pipes; *estimated*; the kernels are issue-bound: the in-register Q3_K kernel
is 377 NEON ops per 4 × 256 weights = 0.37 ops/w, and the bench's 0.216 cycles/w is 85 % of that issue bound):

| path | 3 bits | 4 bits | 6 bits | note |
|---|---|---|---|---|
| `sdot` with decode (today) | Q3_K 0.37 to 0.43 ops/w (measured 102 ps/w) | Q4_0x4 ≈ 0.22 (measured 51 ps/w) | Q6_K ≈ 0.4 | Q3_K is expensive because of its decode (2-bit field + `hmask` plane + 6-bit sub-scales), not because of its bits |
| LUT, **exact** int16 tables (per plane per 16 rows × 4 weights: 2 `TBL` + 2 `zip` + 2 `add`, + index and scale work) | ≈ 0.33 | ≈ 0.43 | ≈ 0.61 | lossless; bit-identical if the per-superblock int32 sums and the float steps are kept in order |
| LUT, **lossy** int8 tables (1 `TBL` + 2 widening adds per plane) | ≈ 0.19 | ≈ 0.25 | ≈ 0.36 | T-MAC's mode; declared numerics change |

**Crossover (*estimated*).** Exact LUT gains 1.0 to 1.3 × only against Q3_K's decode-heavy kernel and loses about
2 × against Q4_0x4. Lossy LUT gains ≈ 2 × at 3 bits and is at parity at 4 and 6 bits. At 6 bits the question is
moot: the only Q6_K tensor (the head) is byte-bound. An exact integer LUT is possible for Q3_K: table entries of 4
int8 activations stay within ±508, so they fit int16, and Q3_K's −4 offset and 6-bit scales are integer
operations. It costs 32 bytes of table per 4 activations (2,048-wide input: 16 KB, which fits L1) and about 6 ops
per plane per 16 rows × 4 weights.

**What it needs.** `TBL` looks up 16 lanes from one table, so the 16 lanes must be 16 rows for the same 4
activations. That requires a **16-row interleaved, bit-plane re-layout** of the experts. Q3_K's native layout
(2-bit values strided by 32 inside `qs`) cannot be indexed directly without bit gathering, which NEON lacks. An
anonymous copy is ruled out by residency (2.8 / 2.6 GB). The rules' own route is a **runtime-derived file** the
runtime writes once to the app's files (COMMON_RULES §1): about the same bytes, and the original expert pages are
then never touched. It needs about **+2.8 GB (OLMoE) / +2.6 GB (Granite) of storage**; the rules allow it, but the
storage is an owner question. The conversion is one read and one write of about 3 GB (≈ 30 to 90 s once,
*estimated*).

**Gain (*estimated*; expert op ≈ kernel time, §0.2; floors from §0.2):**
- exact: OLMoE 27.8 × (1 − 1/1.0..1.3) = **0 to −6.4 ms**; Granite num 13.7 × same = **0 to −3.2 ms**.
- lossy: kernel time ÷ 1.5 to 2.0 by the op counts, capped at the byte floor. OLMoE 27.2 (skip) → 13.6 to 18.1
  against a floor of 12.4 to 15.8, i.e. −9 to −13.6 by the arithmetic. Discounted because op counts on this core
  have been optimistic before (a realised 1.2 to 1.5 ×): **−5 to −11 ms, centre −8**. Granite num 13.7 → 6.9 to 9.1
  (floor 6.0 to 7.7), −4.6 to −6.8 by the arithmetic, **−3 to −6 ms** discounted.
- PPL of the lossy form: unknown. T-MAC's NMSE result implies ≈ 0. A crude worst-case bound (tables rounded to
  int8 at a step of 4 activation units) gives a dot-product error ≈ 4 × the Q8_K activation error, which could be
  tenths of a per cent. Remaining budget: OLMoE 12.3003 − 12.1053 = 0.195 (1.66 % of stock), Granite 9.6879 − 9.5173
  = 0.171 (1.85 %).

**Generalises:** both models (both store experts as Q3_K), and any Q3_K MoE. **Size:** large (layout writer, NEON
kernel, the fused op, the skip mask and the read-ahead re-pointed at the new file). **Cheapest decisive
experiment:** a phone microbench in the existing `GGML_CPU_BENCH` harness (one 16-row interleaved slice, exact and
lossy variants against the current kernel, A78-pinned, clock logged). That is minutes of phone time and no PPL.
Kill if exact < 1.10 × and lossy < 1.40 ×. Only if lossy survives: a host PPL emulation (scalar C with exactly the
lossy arithmetic) at `-b 512` and `-ub 1`. Do not use M4 ratios for the speed: campaign 1's Q2_K screen ran 4.9 to
5.9 × worse relative to the M4 (R29).

### A2. XOR/AND/POPCNT networks on frozen weights

**Evidence for training-free 1 to 2 bit use.** None usable here. A pure XNOR-popcount network also binarises the
activations. Post-training 1-bit weight methods (BiLLM, PB-LLM, ARB-LLM) lose far more than +5 %, and OneBit-type
methods need training. *(Their PPL tables were not re-fetched in this session; listed under "could not verify".)*
The only legitimate use is a **screen with exact recovery** for a selection. In this model the selections are the
head's top-K and the router's top-k.
- **Head:** campaign 1 measured that the screen's denominator, not its recall, is the problem. A 2.6-bit Q2_K screen
  needed M = 16,384 exact rows for +0.059 %, and M = 4,096 gave +0.467 % (R28, *host*). A 1-bit screen is worse
  than Q2_K at every M, and at M ≈ 16 K the exact stage is 16 % to 33 % of the head. **Dead by PPL before
  kernels** (last_3ms memo, (a)).
- **Router:** 0.4 ms (OLMoE) and 0.8 ms (Granite) in total (*device* R60), and routing flips change the text
  (campaign 1's F16 router, R47). **Dead: at most −0.4 / −0.8 ms.**
- **Bit-plane progressive evaluation with an error bound:** for Q3_K, `hmask` alone gives the sign-like half of
  every weight (h = 1 → q ∈ {0..3}, h = 0 → q ∈ {−4..−1}), but the bound on the remaining two planes is
  1.5 · s · Σ|x|, the size of the signal itself. No row can be certified early. Exact early termination in the
  literature (SnaPEA exact mode; bit-serial MSB-first) relies on ReLU's hard zero and on sign-sorted weights (source:
  [SnaPEA, ISCA 2018](https://cseweb.ucsd.edu/~hadi/doc/paper/2018-isca-snapea.pdf), [BitSET](https://www.aminer.cn/pub/65254ff4939a5f4082ce805f/bitset-bit-serial-early-termination-for-computation-reduction-in-convolutional-neural-networks)).
  SiLU has no zero, so in these models only the predictive (lossy) form exists, which is seed (a) below.

### A3. GEMV → GEMM at decode

At batch 1, only drafted tokens produce several right-hand sides against the same weights: speculation (draft
model, n-gram, MTP, EAGLE). That is closed by R20 (0.48 to 0.76 ×) and by host trials. One nuance is new: Q3_K's
kernel is about 83 % decode work (313 of 377 NEON ops are not `sdot`), so a two-column Q3_K kernel would cost about
1.17 × one column, not 2 ×. Dense ops are byte-bound with spare compute (Q4_0x4 compute is ≈ 0.4 × its byte time),
so two columns are nearly free there. A 2-token verify could therefore cost ≈ 1.3 to 1.4 × one token (*estimated*,
union of experts ≈ 1.5 × at 45 to 50 % overlap, campaign 1 R16). But the only drafts that win need ≥ 1.4 tokens per
evaluation at ≤ 2 ms per draft token. Prompt lookup gives 1.20 to 1.27 on the official prompt (R16). A draft model
is another file (an owner decision), another set of resident pages, and 15 ms per token on this CPU (R20).
**Dead under the residency constraint**, and the GEMV → GEMM premise is wrong anyway: the expert kernel is
compute-bound, so extra columns add work that nothing amortises.

### A4. Activation-gated skipping of weight reads

**Input-side (TEAL-like) sparsity on row-major quant blocks: dead.** TEAL reaches 40 to 50 % sparsity at small PPL
cost (Llama-3-8B: 5.87 → 5.94 / 6.21 / 6.67 at 25 / 40 / 50 %), but its kernel loads whole weight **columns** and
needs column-major storage (source: [arXiv 2408.14690](https://arxiv.org/html/2408.14690v3)). Q3_K and Q4_0 store
256- and 32-weight blocks along the input dimension, so a skipped input channel saves no bytes and no `sdot`.
Campaign 1 measured that whole-block sparsity is only 0.7 to 12 % (`ssm_out`, block-32 mask, R29 / §8), and that an
fp32 axpy over input-major copies runs +2 to +5 ms slower than the byte-rate GEMV (`dense_bytes_device.md` §4). An
input-major copy of a quantised tensor is a requantisation, not exact. **Output-side sparsity** is seed (a).

### A5. Vector-quantisation codebooks resident in L3

The cache is 4 MB of L3 for the CPU (§1; the SLC is not reachable). State of the art at 2 bits on Llama-2-7B
WikiText-2: AQLM 6.59 to 6.93, QTIP 6.82, QuIP# 8.22, against FP16 ≈ 5.5. VPTQ claims 0.3 to 0.5 lower PPL than
these at 2.0 to 2.3 bits. At 4 bits QTIP, QuIP# and AQLM are at 5.17 to 5.21 (sources:
[AQLM 2401.06118](https://arxiv.org/html/2401.06118v4), [VPTQ 2409.17066](https://arxiv.org/pdf/2409.17066),
[QTIP review](https://liner.com/review/qtip-quantization-with-trellises-and-incoherence-processing)). AQLM's 1×16
codebooks are 1 MB per layer, which means gathers, not a table in a register. llama.cpp's own VQ-like types
(IQ3_XXS at 3.06 bpw, IQ2_XXS at 2.06) decode through grid lookups. Their NEON path was described by its author
at introduction as "ARM_NEON is pathetic", and later CPU data is thin (source:
[llama.cpp #5196](https://github.com/ggml-org/llama.cpp/pull/5196), [beebopkim benchmarks](https://beebopkim.github.io/2024/03/09/Benchmarks-for-lots-of-quantization-types-in-llama-cpp)).

**Reading.** VQ trades compute for bytes. This phone's expert op is compute-bound, so VQ is the wrong direction for
speed. Re-encoding at runtime from the frozen Q3_K is lossy (double quantisation; even a richer Q4_0 copy of Q3_K
cost +0.13 to +0.17 % per four layers, R32). A lossless VQ of a 3-bit integer stream does not exist below its
entropy. **As a runtime lever: dead.** **As a file change: needs owner decision.** IQ3_XXS experts would read 11 %
fewer bytes (3.06 against 3.44 bpw; OLMoE's experts 2.77 → 2.46 GB, Granite's 2.60 → 2.31 GB, *estimated*). That
helps residency only, at unknown (probably worse) compute and unmeasured PPL.

### A6. Rewriting y = Wx as indexing or shifts

y_r = Σ_v v · (Σ_{i: w_ri = v} x_i) is weight-value bucketing. Done naively it is a scatter or gather with no SIMD
gain. Done well it **is** the LUT of A1. Shift-only products need power-of-two weights, which means a re-encoding
(a file change). **Not a separate lever:** it reduces to A1 or A5.

## 3. Part B: the lead's seeds

### (a) Gate-guided row skip inside the experts

- **Mechanism:** h = silu(W_g x) ⊙ (W_u x). Compute the gate rows exactly (they are computed anyway), drop the up
  rows whose |silu(g)| is below a per-layer threshold, and zero the matching h. W_up rows are contiguous Q3_K rows,
  so the bytes and the `sdot` are really saved. W_down columns are not: blocks run along its input, so no bytes are
  saved there (A4).
- **Attacks:** `MUL_MAT_ID`, at most 1/3 of it (the up third), scaled by the dropped fraction s.
- **Evidence:** CATS, which thresholds silu(gate), costs Llama-3-8B 5.87 → 6.78 PPL (+15.5 %) at 25 % and diverges at
  40 % (TEAL's Table 1, source: [arXiv 2408.14690](https://arxiv.org/html/2408.14690v3); CATS itself:
  [arXiv 2404.08763](https://arxiv.org/html/2404.08763v4)). Campaign 1 measured exactly this rule on Granite's
  shared expert (`up_shexp` rows by silu(gate), 60 chunks, base 9.8702): s = 0.15 → 10.0309 (**+1.63 %**), 0.33 →
  10.4547 (+5.9 %), 0.54 → 11.83 (`05_compute_less.md` §8, *host*). By contrast, thresholding the product h
  (`down_shexp` input) was cheap (24 % at −0.3 %, 37 % at +0.03 %), but that is input sparsity of W_down and saves no
  bytes.
- **Gain (*estimated*):** at s = 0.15, 5 % of expert compute: **OLMoE −1.4 ms, Granite num −0.7 ms**, for ≈ +1.6 %
  PPL, which is about all of the remaining budget (1.66 % / 1.85 %). R59's skip frontier buys more per PPL point
  (OLMoE: 13.4 % of expert bytes for +3.3 %, ≈ 4 % per point against ≈ 3 % here).
- **Numerics:** declared; it acts at the same size at `-b 512` and `-ub 1`. **MMID_FUSE:** compatible. The fused op
  claims row chunks per expert, so inside a chunk the gate rows run first and then only the kept up rows. One
  ordering change, no new barrier.
- **Generalises:** every SwiGLU MoE (both models). **Size:** small to medium.
- **Cheapest experiment:** host PPL, 60 chunks, routed experts of OLMoE at s = 0.10 and 0.15 through a `cb_eval`
  hook (`dense_colskip.py`'s method); kill if s = 0.10 costs > +1.0 %. **Verdict proposal: parked, low.**

### (b) Early commit on the head screen

- The exact stage after the screen is select + 1,024 / 2,048 rows ≈ **0.1 ms** (campaign 1 last_3ms memo: 3.92 ms
  op against a 3.80 ms screen byte floor). The screen itself must finish before any top-1 guess exists. **Overlap
  ≤ 0.1 ms: dead.**
- Other serial work, checked: within a token the chain router → experts → next layer is a true dependency. Outside
  the graph, 0.66 to 0.74 ms (OLMoE) and 0.83 to 1.05 ms (Granite) per token remain (*device*, R60). The part after
  sampling (detokenising, the callback to Dart, the UI) can run alongside the next decode. That is the crumb in §4
  N5.

### (c) Sign-bit or popcount first stage for the head and the router

Dead, by A2: the denominator problem for the head; ≤ 0.4 / 0.8 ms and text flips for the router.

### (d) The Adreno 660 through the bundled OpenCL backend

- llama.cpp's OpenCL backend documents that it "does not work on Adreno 6xx GPUs" / "A6x GPUs in phones are likely
  not supported due to the outdated driver and compiler". Q4_0 is the optimised type and Q6_K is "supported, not
  optimised". No Q3_K or MoE statement was found (source: [OPENCL.md mirror](https://huggingface.co/hekod19045/llama-cuda/blob/main/docs/backend/OPENCL.md),
  [Qualcomm blog](https://www.qualcomm.com/developer/blog/2024/11/introducing-new-opn-cl-gpu-backend-llama-cpp-for-qualcomm-adreno-gpu)).
  Campaign 1 measured 0.7 to 2.9 tok/s (*device*).
- **What breaks:** (1) bit-identity: GPU sums are not the CPU's, so it is a declared change; (2) **the gate cannot
  be run**, because the host has no Adreno and the PPL gate is a host run; (3) memory: the backend repacks Q4_0
  into its own buffers, which is anonymous memory taken from the page cache (residency); (4) overlap: the head is the
  token's last op, so a GPU head overlaps nothing in the same token. At best it beats the CPU's 3.9 ms by reading
  faster than 26 GB/s: **0 to −1.4 ms**, minus launch and sync per token. **Dead** on (1) to (3).

### (e) Heterogeneous cores and the heat gate

Priced as new direction N1 (helpers) and N6 (heat).

### (f) Residency tricks beyond the hold

- `mlock`: **64 MiB allowed on this phone** (*device*, r22). Campaign 1's R-log assumed 64 KB and called it
  unavailable; the report contradicts that. 64 MiB is 2.4 % of OLMoE's 2.67 GB working set. Pinning the slices
  most often lost in a wait would cover part of R60's median hold loss (49 to 120 MB): **0 to −0.5 ms**, both
  models. Crumb N5.
- `MADV_POPULATE_READ` (prefault page tables for pages already cached) exists only from Linux 5.14 (source:
  [linux-mm patch](https://lore.kernel.org/linux-mm/20210823120645.8223-1-david@redhat.com/T/)). The phone's kernel
  version is **not known**: the r22 report's `kernel` field is empty. Before anything is built on it, the kit should
  print `uname -r`.
- Per-expert hot sets and working-set shaping without numerics: memo-18 says OLMoE's 32-token decode touches 790
  of 1,024 expert slices (2,155 MB). No exact policy shrinks that; only routing can, which is N2.
- App-managed anonymous buffers: closed (R23); zram would compress Q3_K data poorly and cost CPU on every fault.

### (g) The Q3_K / Q4_0 / Q6_K bit layouts directly

- An `hmask`-first screen for the gate (42 % of the gate's bytes including scales), finishing only rows that are
  not strongly negative and skipping their up rows. That is (a) plus a screen error, after a plane-separated
  re-layout. For s = 0.15: gate (1 − 0.42)·s + up s ≈ 0.53 s of the expert → **OLMoE −2.2 ms, Granite −1.1** at
  (a)'s PPL plus screen error. Parked with (a).
- An exact nibble re-layout of Q3_K (q2 + 4h stored as 4 bits, Q4-speed decode) would make the expert op
  byte-bound at 4.44 bpw: OLMoE 346 → 447 MB → 19.6 ms (−8). But the expert working set grows by 29 % (OLMoE's
  decode set 2.16 → ≈ 2.8 GB). R55 to R60 show residency is the binding constraint, so **dead** unless residency is
  solved first.
- Q4_0 and Q6_K tensors are byte-bound: no layout trick changes their bytes without requantising.

### (h) Reuse across tokens or layers

- Expert-output caching or delta updates: x changes every token, so exact reuse is impossible. Approximate reuse is
  the early-exit family (R21: cheapest layer +2.5 %). **Dead.**
- Routing predictability: useful for read-ahead (campaign 1: 45 to 50 % of a token's experts were used by the
  previous token, R16 *host*) and for N2, not for compute.

## 4. New directions

### N1. Opportunistic work-stealing helpers on the idle cores, inside the expert op only

- **Mechanism:** 1 to 5 extra threads (the fourth big core; the A55s) that never join a barrier. They wait on a
  flag (`wfe`), and while a `MUL_MAT_ID` op is open they take row chunks from the same counter the three compute
  threads use (`MMID=3` + `CHUNK2` already hand out chunks dynamically). Each output row is still computed by one
  thread with the same instructions, so the change is **bit-identical**: the A55 runs the same armv8.2 + dotprod
  module, and the reduction order does not change, by the same argument that made `MM_CHUNK` exact. A straggler
  bound is required, because a helper preempted in the middle of a chunk would hold the op open. Options: small
  chunks for helpers (16 rows: OLMoE gate/up ≈ 16 × 2,048 × 102 ps / 0.4 ≈ 8 µs on an A55, *estimated*), or a main
  thread recomputes an unfinished helper chunk (same bits, idempotent write).
- **Why R18 does not close it:** R18's 4-thread loss was measured with 1,909 barriers per token and threads that
  joined every barrier (barrier 2.0 µs at 4 threads against 0.30 at 3, *device*). Helpers join none. Today's runtime
  has 666 barriers per token (*device*, R60), and the expert op is 64 % (OLMoE) and 29 to 40 % (Granite) of the
  token. Published phone studies find efficiency cores hurt llama.cpp
  ([arXiv 2410.03613](https://arxiv.org/html/2410.03613v3)), but for barrier-synchronised work. Adaptive core
  selection is an active topic ([MNN-AECS, arXiv 2506.19884](https://arxiv.org/html/2506.19884v1)).
- **Gain (*estimated*):** with f extra A78-equivalents free, the expert op shrinks by 27.2 × f / (3 + f) (OLMoE) and
  13.7 × f / (3 + f) (Granite num), capped at the byte floor. f = 0.5: −3.9 / −2.0; f = 1: −6.8 / −3.4; f = 1.5:
  −9.1 / −4.6. **Range 0 to −8 ms (OLMoE), 0 to −4 ms (Granite).** f comes from the fourth big core (X1 ≈ 1.2 ×
  A78 at 2,035 MHz) when Android leaves it free, plus the A55s at 0.3 to 0.58 × each. The A55 figure is weak: one
  unpinned bench on a falling clock. Android places background work on the A55s; AnandTech reports a hard limit on
  A55 memory access, so their streaming rate is a risk (needed: ≈ 1.7 to 2.4 GB/s per A55 at 0.4 to 0.58 × the
  A78's Q3_K rate).
- **Heat:** an A55 adds about a tenth of an A78's power (not verified here), and shorter decodes offset it. It must
  be logged in the kit (headroom after the run).
- **Generalises:** both models and every MoE on any big.LITTLE phone; it is the one lever that is a property of the
  phone, not of a model. **Size:** medium (≈ 200 to 400 lines in the expert op and the threadpool, plus affinity).
- **Cheapest decisive experiment, zero code:** one kit with per-run alternation of `threads` 3 and 4 on both models
  (the run table already has a `threads` field), plus `GGML_CPU_BENCH` pinned once to cpu 0 (A55) and cpu 7 (X1)
  (the bench's pin set is a one-line change). That gives the free fourth core and the per-core rates at logged
  clocks. Kill the A55 part if an A55's bench rate is < 0.25 × the A78's. Kill the whole lever if 4 threads lose and
  the bench says the fourth core is busy. A host prototype on the M4's efficiency cores can check bit-identity only,
  not speed.

### N2. Cache-conditional routing (shrink the decode working set)

- **Mechanism:** add a bias λ to the router logits of the experts that layer used in the last N tokens, before
  top-k; the mixing weights stay those of the original softmax. Skliar et al. (Qualcomm AI Research, TMLR 2025)
  report this training-free change cuts cache misses by over 50 % at 0.1 to 3 % PPL, with up to 2 × on two phones
  (Qwen1.5-MoE) (source: [arXiv 2412.00099](https://arxiv.org/html/2412.00099v2)).
- **Why it is the residency lever with numerics:** memo-18 shows OLMoE's decode set (2.16 GB, 790 slices) does not
  fit below about 2.2 GB of cache, while Granite's (1.11 GB) does, which is why Granite tolerates low-cache days.
  Halving the distinct slices would move OLMoE to Granite's side of that line. At R60's hold state the fault excess
  is 10 to 11 ms (OLMoE) and 4 to 5 ms (Granite).
- **Gain (*estimated*):** if the excess falls in proportion to misses: **OLMoE −2 to −6 ms, Granite −1 to −2.5**. In
  low-cache states it is worth much more (memo-18's simulator puts OLMoE at 11.5 tok/s against Granite's 19.8 at 2.2
  GB; the gap is the decode set).
- **Numerics:** declared. Use the runtime's own deterministic history (the experts it routed), never the OS page
  cache, so PPL is a function of the text alone. The prior can be applied token by token inside a batch, so it
  behaves the same at `-b 512` and `-ub 1`. Gate at both anyway (owner rule). If a page-cache prior were ever
  wanted, it would make numerics depend on the phone's state: **needs owner decision**.
- **Interplay:** it shares the budget with skip. With skip it may need re-screening: the remaining 1.66 % / 1.85 %
  sets the λ.
- **Generalises:** every top-k MoE (OLMoE, Granite, LFM2.5). **Size:** small in the runtime (top-k with a bias plus
  a small per-layer history ring); the host tooling exists (memo-18 tracer and simulator).
- **Cheapest decisive experiment (host only):** (1) on memo-18's traces, distinct slices per 32-token decode and
  simulated re-read MB for λ ∈ {0.5, 1, 2} and N ∈ {4, 8, 16}, with no PPL needed; (2) PPL on 60 chunks at `-b 512`
  and `-ub 1` for the λ that halves the distinct slices. Kill if that λ costs > +1.0 % on either model, or if the
  distinct slices fall by < 25 %.

### N3. Helper-thread L3 warm of the next dense op

- **Mechanism:** while the expert op of layer l runs (it is compute-bound and uses ≈ 12 GB/s of the ≈ 20 to 23 GB/s
  the cluster reaches), a helper streams the first ≤ 3 MB of layer l + 1's first dense weights into the shared 4 MB
  L3, timed for the op's last quarter so the expert stream does not evict them. Prefetch only: **bit-identical.**
- **Attacks:** dense `MUL_MAT`. Granite's per-layer dense (`ssm_in` 5.57 MB, `ssm_out` 2.65, shared expert 2.8) is
  most of its 18 ms; OLMoE's attention per layer is 9.4 MB.
- **Gain (*estimated*):** upper bound = the warmed MB at the byte rate: Granite 40 × 3 MB × 0.0411 = 4.9 ms, OLMoE
  16 × 3 × 0.0411 = 2.0 ms. With half of the lines evicted or late: **Granite 0 to −3, OLMoE 0 to −1**. Campaign 1's
  in-kernel Q3_K software prefetch lost (R23) because it polluted L1; this is a different mechanism (L3, across
  ops) with the same risk class.
- **Generalises:** both; it is larger on dense-heavy hybrids. **Size:** medium. **Cheapest experiment:** a phone
  microbench (Q4_0x4 GEMV over a 3 MB slice, cold and warmed by another core after a 20 MB stream); kill if the
  warmed op is < 15 % faster.

### N4. LUT Q3_K kernel (exact and lossy)

This is A1, listed here because it is the largest compute lever after N1 and competes for the same pool. **N1 and
N4 stack only down to the byte floor** (OLMoE 12.4 to 15.8 ms, Granite 6.0 to 7.7 ms of expert op).

### N5. Crumbs, bit-identical

- Outside-graph pipelining: hand detokenising, callbacks and the UI to another thread after the argmax:
  **0 to −0.5 ms** (OLMoE 0.66 to 0.74 ms, Granite 0.83 to 1.05 ms outside, *device*).
- Pin the three compute threads to {4, 5, 6, 7} (they are unpinned, §1), so that EAS cannot place one on an A55 in
  the middle of an op. Waits are 1.0 ms (OLMoE) and 2.3 to 2.5 ms (Granite) per token (*device*): **0 to −1 ms**,
  more in variance than in the median.
- `mlock` of ≤ 64 MiB of the slices most often lost in a wait: **0 to −0.5 ms**; costs 64 MiB the kernel cannot
  reclaim.
- **Cheapest experiment:** all three are kit knobs; pinning and `mlock` can share one kit with N1's thread
  alternation.

### N6. Cooler prefill for the heat gate (needs owner decision)

- TTFT is 4.8 to 5.3 s (R58) and the decode about 1.6 to 2 s, so the prefill burns most of a run's energy. TTFT is
  not part of the metric, and running the prefill on fewer or smaller cores uses the same kernels (bit-identical) at
  less heat. The gate is reached sooner, waits are shorter and fewer pages are lost; a capped clock (1,555 against
  2,112 MHz, ×1.36 on compute-bound ops) becomes rarer.
- **Gain:** 0 on a cool day (R60's profiled runs were at 2,112); up to −10 ms per token on capped days
  (*estimated*). **Owner decision:** it makes TTFT 2 to 4 × longer, which users see although the benchmark does not.

## 5. Ranked shortlist

| # | lever | OLMoE ms/token | Granite ms/token | numerics | cheapest decisive experiment |
|---|---|---|---|---|---|
| 1 | N1 opportunistic stealing helpers (fourth big core, A55s) in `MUL_MAT_ID` | 0 to −8 | 0 to −4 | bit-identical | kit: `threads` 3 ↔ 4 per run on both models + cpu-bench pinned to cpu 0 and cpu 7 (no new runtime code) |
| 2 | N2 cache-conditional routing (deterministic history prior) | −2 to −6 at R60's state; large in low-cache states | −1 to −2.5 | declared, same at `-b 512` / `-ub 1` | host: memo-18 trace simulation over λ and N, then 60-chunk PPL at both settings |
| 3 | N4 / A1 LUT Q3_K kernel: exact int16 / lossy int8 | 0 to −6.4 / −5 to −11 | 0 to −3.2 / −3 to −6 | identical / declared | phone microbench in `GGML_CPU_BENCH` (16-row slice, both variants); host PPL emulation only for lossy |
| 4 | N3 helper L3 warm of the next dense op | 0 to −1 | 0 to −3 | bit-identical | phone microbench, warmed against cold 3 MB GEMV |
| 5 | N5 crumbs: outside-graph pipelining, thread pinning, 64 MiB `mlock` | 0 to −2 together | 0 to −2.5 together | bit-identical | kit knobs, shared with #1's kit |

Notes on the ranking. #1 is first because its first test costs no runtime code and no PPL, it is bit-identical, and
it is the only lever that is a property of the phone rather than of a model (the trust purpose). #2 attacks the
largest pool on OLMoE and the decisive one on low-cache days; it is host-testable today, once the CPU is free. #3 is
the largest compute lever but the largest build, and it needs storage. #1, #3 and part of #4 compete for the same
pool: their sum stops at the byte floor. Granite reaches 25 only if, besides zero faults and an expert op at its
byte floor, 2 to 3.5 ms come out of the ≈ 6.2 ms "rest" and ≈ 2.5 ms of waits. **Next pricing task proposed: an
op-by-op account of Granite's "rest"** (norms, `SSM_CONV`, state `CPY`, adds; the R60 profile lines carry only four
op classes).

## 6. Dead, with the reason

| idea | reason (numbers) |
|---|---|
| XOR/POPCNT networks on frozen weights | binarised activations; post-training 1-bit far over +5 %; no training allowed |
| sign-bit / 1-bit head screen | denominator: Q2_K needed M = 16,384 for +0.059 % (R28); 1-bit is worse |
| sign-bit / popcount router screen | router 0.4 / 0.8 ms in total; flips change the text (R47) |
| exact bit-plane early termination | SiLU has no zero; remaining-plane bound ≈ 1.5 · s · Σ|x| ≈ the signal |
| GEMV → GEMM at decode | only speculation creates columns (R20, 0.48 to 0.76 ×); a draft model costs residency; the kernel is compute-bound |
| early commit on the head screen | exact stage ≈ 0.1 ms; the screen must finish first |
| Adreno 660 via OpenCL | A6xx documented as unsupported; 0.7 to 2.9 tok/s measured; cannot be gated on the host; anonymous copies cost residency |
| runtime VQ / codebook re-encoding | lossy double quantisation; gather decode on a compute-bound CPU; the SLC is not CPU-reachable |
| TEAL-style input sparsity | blocks run along the input in Q3_K and Q4_0: no bytes saved; whole-block sparsity 0.7 to 12 % (R29) |
| cross-token output reuse, delta updates | x changes every token; approximate reuse is the early-exit family (+2.5 % for one layer, R21) |
| exact nibble re-layout of Q3_K | −8 ms of compute for +29 % expert bytes, against the binding residency constraint |
| LUT for Q4_0 / Q6_K tensors | they are byte-bound; a LUT reduces ops, not bytes |
| entropy-coded experts | decoding costs ≫ 0.22 cycles per weight |
| y = Wx as indexing / shifts | reduces to the LUT (A1) or to re-encoding (A5) |

Parked with numbers (not dead): gate-guided up-row skip and its `hmask` screen (−1.4 / −0.7 ms and −2.2 / −1.1 ms
for ≈ +1.6 % PPL).

## 7. Needs owner decision (what each would buy)

1. **Storage for a runtime-derived expert file** (A1/N4): +2.8 GB (OLMoE) / +2.6 GB (Granite). The rules allow it
   (COMMON_RULES §1); it buys the LUT kernel, 0 to −11 ms.
2. **A page-cache-state routing prior** instead of a history prior (N2): numerics would depend on the phone's
   state. It buys a little more than the deterministic form; not recommended.
3. **A re-encoded expert file** (IQ3_XXS, VQ): −11 % expert bytes for residency; compute probably worse; PPL
   unmeasured.
4. **Trading TTFT for heat** (N6): 0 to −10 ms per token on capped days; TTFT 2 to 4 × longer.

## 8. What I could not verify

- The AnandTech pages (16463, 16271) redirect now. The SLC-access quote and the 51.2 GB/s figure come from search
  excerpts, not from fetched pages.
- Whether T-MAC's PR #10181 was merged; T-MAC numbers on an SD888; any T-MAC comparison against llama.cpp's
  repacked or KleidiAI kernels.
- BiLLM / PB-LLM / OneBit PPL tables (not fetched in this session).
- The phone's Linux kernel version (`kernel` is empty in r22), hence `MADV_POPULATE_READ`.
- A78 / A55 `TBL` throughput and latency (Arm optimisation guides not fetched); my LUT op counts assume 2 `TBL` per
  cycle on the A78.
- The A55's Q3_K rate: one unpinned campaign-1 bench on a falling clock.
- The ≈ 1.4 ms by which Granite's `MUL_MAT` does not fall when its head falls (off 21.8, num 22.0; head 5.4 → 3.9;
  R60). It is not explained here.
- Whether OLMoE's consecutive-token expert overlap resembles Granite's 45 to 50 % (only Granite was traced, R16).
- QTIP's own paper (only a review page was read); WINA's PPL numbers (its paper reports task accuracy only).

## 9. Where the brief and the sources or the repository disagree

1. "LUT: no gain at 3 to 4 bits." T-MAC publishes 4.7 × (3 bits) and 3.1 × (4 bits) kernel speedups over 2024
   llama.cpp, but with lossy int8 tables. With exact int16 tables against Vollie's Q3_K kernel the estimate holds
   (1.0 to 1.3 ×); against Q4_0x4 an exact LUT loses.
2. "Dense ~9.6, head ~3.5" (OLMoE). These are nested: the head is inside the `MUL_MAT` figure, so non-head dense is
   ≈ 6.1 ms. The R60 profile lines only sum to the token time this way.
3. "3 MB SLC" as a cache for the CPU. AnandTech measured that the SD888's CPU cluster does not reach it; use 4 MB.
4. `mlock`: the brief's 64 MB is right (*device*, r22: 64 MiB). Campaign 1's R-log line ("typically 64 KB, not
   available") and memo-13 / memo-20 ("64 KB, not checked") are contradicted by the report.
5. "3 threads on the big cores": the compute threads are not pinned in `0001` (only the bench is), so they run
   wherever Android's scheduler places them. That is probably the big cores, but it is not enforced.
6. "Campaign 1 found the phone compute-bound": confirmed and sharpened. The expert op equals the bench kernel time
   (27.4 against 27.8 ms on OLMoE); the dense ops are byte-bound. "Compute-bound" applies to the experts only.

## 10. Sources

- T-MAC: https://arxiv.org/html/2407.00088v2
- llama.cpp T-MAC proposal, PR 10181: https://github.com/ggml-org/llama.cpp/pull/10181
- bitnet.cpp: https://arxiv.org/pdf/2410.16144 ; https://export.arxiv.org/pdf/2502.11880
- TEAL: https://arxiv.org/html/2408.14690v3
- CATS: https://arxiv.org/html/2404.08763v4
- WINA: https://arxiv.org/html/2505.19427v1
- Mixture of Cache-Conditional Experts: https://arxiv.org/html/2412.00099v2
- SnaPEA: https://cseweb.ucsd.edu/~hadi/doc/paper/2018-isca-snapea.pdf ; BitSET: https://www.aminer.cn/pub/65254ff4939a5f4082ce805f/bitset-bit-serial-early-termination-for-computation-reduction-in-convolutional-neural-networks
- AQLM: https://arxiv.org/html/2401.06118v4 ; VPTQ: https://arxiv.org/pdf/2409.17066 ; QTIP (review): https://liner.com/review/qtip-quantization-with-trellises-and-incoherence-processing
- llama.cpp IQ3 quants: https://github.com/ggml-org/llama.cpp/pull/5196 ; benchmarks: https://beebopkim.github.io/2024/03/09/Benchmarks-for-lots-of-quantization-types-in-llama-cpp
- llama.cpp OpenCL backend: https://huggingface.co/hekod19045/llama-cuda/blob/main/docs/backend/OPENCL.md ; https://www.qualcomm.com/developer/blog/2024/11/introducing-new-opn-cl-gpu-backend-llama-cpp-for-qualcomm-adreno-gpu
- SD888 caches and memory: https://www.anandtech.com/show/16271 ; https://www.anandtech.com/show/16463/snapdragon-888-vs-exynos-2100-galaxy-s21-ultra/3 ; https://www.theregister.com/2020/12/02/qualcomm_snapdragon_888/ ; https://www.computerbase.de/2020-12/qualcomm-snapdragon-888-hands-on/
- `MADV_POPULATE_READ`: https://lore.kernel.org/linux-mm/20210823120645.8223-1-david@redhat.com/T/
- Mobile LLM core-count study: https://arxiv.org/html/2410.03613v3 ; MNN-AECS: https://arxiv.org/html/2506.19884v1
- Repository: `reports/03_final.md` (R16, R18, R20, R21, R23, R28 to R33, R41, R47), `reports/05_compute_less.md` §8,
  `reports/memos/dense_bytes_device.md`, `reports/memos/last_3ms.md`, `reports/memos/round13_plan.md`,
  `docs/runtime-knobs.md`, memo-18, memo-20, R58 to R60, `device/r22_2026-11-15.json`, the two model cards.
