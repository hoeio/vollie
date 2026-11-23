# c3 memo-12 work (Ebony, round 5): a GPU prefill, the head task for Tom, the W-position kernel's exactness, re-keyed rows, review of Bonnie r5

Date: 2026-11-23. Author: Ebony (`claude-fable-5-2`). Not committed; the lead commits. No verdict here.
Follows [ebony-joins-r4.md](ebony-joins-r4.md); reviews [bonnie-joins-r5.md](bonnie-joins-r5.md) (read in full).
Keyed on [tom-r35-build.md](tom-r35-build.md) as saved at 19:20.

Labels: *device* (printed by the phone), *device-derived* (arithmetic on device fields), *host* (the laptop),
*estimated* (model, count or judgement), *external* (another group's number or text), *source* (read in the local
tree `lab/llama.cpp-c3`, base `620d2225c` + c2 series). Triples are Yi / Llama-2 / DeepSeek.

New measurements in this file, all *host*: a random-data simulation of half-precision accumulation (§1.4,
`ebony_h16acc.py`), the batch-512 CPU rate read from existing perplexity logs (§2.1), Dolly's size (§2.2), free
disk (§2.5). Nothing ran on the phone. I ran no model: the laptop was at load 25 with Tom's perplexity job.

## 0. Answer first

1. **A GPU prefill is not worth building for the claim run.** *Estimated*: it is about as fast as the CPU prefill
   (GPU 12 to 36 tok/s, centre 24; CPU 18 to 26), it moves P(decode starts at C ≥ 40) from 0.55 to about 0.65, and
   that is worth **+0.03 to +0.10 tok/s** in expectation. It costs a second numerics path (below) and a re-layout
   stream. It stays a rider cell in a later kit, not a try-list item.
2. **The numerics of upstream's batched kernel are a different class from the lane's.** *Source*: the decode gemv
   accumulates in float; the prefill gemm accumulates the whole 4,096- or 11,008-term sum in **half precision**.
   *Host*, random data: output error 0.9 % and 1.4 % of the output's rms, about 15 to 20 × the gemv's
   (6e-4 to 8e-4, *device*, R74). It would need its own emulation and its own `-b 512` check. Upstream also holds
   an **integer** prefill kernel (float accumulators, exact integer dots) that needs `cl_khr_integer_dot_product`;
   R75's `x4 ext text=` shows for free whether the 660's driver has it.
3. **Correction to my r4 §2.2: teacher-forced dumps are not 10 to 30 × faster.** The laptop's batch-512 rate is
   174 to 176 rows/s on the CPU (*host*, existing logs) and about 440 on Metal (*external*, search summary)
   against Tom's 118 to 135 greedy rows/s: **1.4 × and about 3.5 ×**. Only rows where the model's own argmax
   equals the text's next token are exact training pairs, about 0.35 to 0.50 of them. The data route's real gain
   is one to two doublings, not four. The single task for Tom is in §2.
4. **DeepSeek licence (read today, quoted in §2.6):** a head trained on the model's hidden states and outputs is a
   "Derivative of the Model" by the licence's own definition. Making it is granted. Distributing it carries four
   duties (use restrictions passed on, a copy of the licence, change notices, attribution kept). Research on the
   laptop is not distribution.
5. **W-position kernel: `ebony_gw.c` follows ggml's repack gemv, but the phone runs KleidiAI.** *Source*, the
   KleidiAI 1x4 kernel's assembly: the float sequence per block is the same (FMUL of the two scales, SCVTF #4,
   FMLA), plus an FMAX / FMIN clamp at the end that my file lacks. So the kernel can be exact, and §3 lists the
   eight things that make it so and a test in four parts.
6. **Bonnie r5: agreed on every merged row,** with her two changes to my reading line accepted. Per-row notes in
   §5. I move my centre for the CPU alone at two columns from 15.5 to 14.5.

## 1. A GPU prefill (my question 6)

### 1.1 What upstream has, and what it would take here (*source*)

- For a Q4_0 weight at ne1 > 1, ggml's OpenCL backend runs `kernel_gemm_noshuffle_q4_0_f32`
  (`ggml-opencl.cpp`, `ggml_cl_mul_mat_q4_0_f32_adreno`, lines 19960 to 20260; `kernels/gemm_noshuffle_q4_0_f32.cl`).
  It first transposes the activations to half floats, then each fibre computes an 8 × 4 tile.
- **It reads the same weight buffers as the decode gemv** (`extra0_q4_0->q` and `->d`, the "noshuffle" layout). So
  the rows a lane already holds as GPU buffers can serve a batched kernel with no second copy.
- A middle path exists: `kernel_gemv_noshuffle_q4_0_f32_mc3` for 2 to 4 columns (x4m's kernel), float sums.
- An integer path exists, off by default (`GGML_OPENCL_Q4_0_DENSE_DP4A`): `kernel_quant_a_q8_1` quantises the
  activations per block of 32 (d = amax ÷ 127), then `kernel_gemm_noshuffle_q4_0_q8_1_dp4a` takes exact integer
  dot products with float4 accumulators. It needs N > 8 and the extension `cl_khr_integer_dot_product`
  (`has_integer_dot`, line 6931; the vendor's `cl_qcom_dot_product8` is not used).
- The tree's own document says phones' A6x drivers are "likely not supported"
  (`docs/backend/OPENCL.md` lines 56, 307). R74 built and ran the four gemv kernels on this phone (*device*); the
  gemm kernels were never tried here.

Three designs:

| | design | memory | what runs where |
|---|---|---|---|
| P0 | CPU prefill (stock) | – | four big cores, KleidiAI's tile at 0.65 × a gemv per position (Bonnie r5 §2.1) |
| P1 | **GPU-only matmuls, streamed.** Only `MUL_MAT` on a Q4_0 weight with ne1 ≥ 9 goes to the GPU; attention, norms and the KV stay in ggml on the CPU. The lane's GPU rows are used in place; the CPU's rows of each layer are re-laid-out into one scratch GPU buffer just before that layer and dropped after it | + 0.12 to 0.25 GB of scratch (one layer's CPU-share rows) | GPU all rows; one small core feeds and converts (the body is 3.4 to 4.0 GB in 5 to 15 s: 0.3 to 0.8 GB/s, inside one core's copy rate) |
| P2 | **Lane prefill.** The lane's own row split at batch N: GPU rows on the batched kernel, CPU rows on KleidiAI's tile | none | both; about 1.4 to 1.6 × the faster side's rate |
| P3 | upstream's whole graph on the GPU, then switch | a second full copy, 3.4 to 4.0 GB | closed by residency: x3r held 3.5 GB in total and slowed at the end (*device*, R74) |

Joins are not a cost here: 7 launches × 32 layers per batch is about 224 launches, 0.07 to 0.10 s at R74's 300 to
450 µs (*device-derived*). The activations move about 1 GB in total per prefill (*estimated*, 186 tokens), 0.1 to
0.2 s.

### 1.2 Prefill time at 147 to 186 tokens (*estimated* unless marked)

| | rate, tok/s (Llama-2 body) | Yi 147 to 167 | Llama-2 179 to 186 | DeepSeek 147 to 153 | basis |
|---|---|---|---|---|---|
| P0, C 40 to 50 | 18 to 22 (Yi 21 to 26) | 6 to 8 s | 8.5 to 10.5 s | 6.5 to 8 s | r4 §3; no dense prefill was ever timed on this phone |
| P0, capped or badly placed | × 0.67 | to 12 s | to 15 s | to 12 s | r4 §3 |
| P1, half-float gemm | **24; 12 to 36** | 5.5 s; 4 to 12 | 7.6 s; 5 to 15.5 | 6.2 s; 4 to 12.5 | below |
| P1, `_mc3` at 4 columns | 12 to 15 | 10 to 14 s | 12 to 15.5 s | 10 to 13 s | R74 device time 117 / 134 / 126 ms per body × `dev_ratio(4)` 1.9 to 2.0 ÷ 4, + 16 ms of launches per token |
| P2, split | 35; 22 to 50 | 4 s; 3 to 7 | 5.3 s; 3.7 to 8.5 | 4.3 s; 3 to 7 | sum of the two sides at their shares |

Basis of the P1 centre, two routes that agree only loosely:
- *External*: Llama-2-7B on an Adreno 730 tablet, "GPU" prefill 46.03 tok/s against 12.87 on the CPU with
  instruction acceleration and 5.89 without (arXiv 2410.03613v5, Table V, read through the fetch tool; **the
  GPU row's backend is not confirmed**; the paper's date is before the Adreno backend entered llama.cpp). The 660
  at 0.6 to 0.75 of a 730 and an older compiler at 0.7 to 1.0 give 20 to 35.
- *Estimated* from R74: the gemv at one column is bus-bound (21 GB/s). If a further column costs 0.25 to 0.33 of a
  gemv's device time (the merged x4m centre) and the half8 tile is 1.5 to 2 × cheaper per column than the float
  `_mc3`, the body takes 29 to 44 ms per token: 23 to 35 tok/s.
- No published prefill rate for any Adreno 6xx with this backend was found (two searches today).
- **R75's `x4 mcsum dev_ratio=` at 2, 3 and 4 columns is the first *device* input to this row.** If
  `dev_ratio(4)` reads ≥ 2.5, the float path is under 10 tok/s and only the half-float tile remains.
- P(P1 with the half-float gemm is faster than P0 on this phone) 0.55. P(the gemm kernel builds and passes an
  identity check on the 660's driver) 0.6: it uses `qcom_reqd_sub_group_size("full")` and half8, where R74's
  gemv kernels used "half" groups.

### 1.3 Heat and the clock state at the start of the decode

What the gates say about heat per second of load (*device*, headroom at the start and at the peak of the gate
that followed each block; every block started from a gate at ≤ 0.65 or a time-out; R69, R69 repeat, R72, R74):

| load before the gate | seconds | headroom at gate start → peak | time to 0.65 |
|---|---|---|---|
| CPU, R69 `x1` (run 1 / repeat) | 9.8 / 9.7 | 0.75 → 0.85 / 0.78 → 0.88 | 70 s / 60 s |
| CPU, R72 `a` | 19.3 | 0.86 → 0.86 | 120 s |
| CPU, R72 `l23` | 58.8 | 0.87 → 0.87 | 120 s |
| GPU alone, R74 `x3g` | 32.2 | 0.90 → 0.91 | 60 s |
| GPU read + CPU and GPU together, `x2` (R69 / repeat / R72) | 14.1 / 14.4 / 14.0 | 0.73 → **1.06** / 0.72 → **1.42** / 0.72 → **0.98** (one sample each, 5 to 10 s after the block) | 50 s / 85 s / 40 s |
| CPU + GPU, R74 `x3c` | 35.5 | 1.17 → 1.21 | not reached in 180 s |

Reading (*device-derived*, crude: different sessions, and headroom is a forecast that jumps):
- 32 s of the GPU alone raised headroom by about 0.25 (0.008 per second). 10 to 19 s of CPU load raised it by 0.20
  to 0.23 (0.011 to 0.021 per second). **Per second, the GPU alone heats this sensor at roughly half the CPU's
  rate, and it cooled to 0.65 in half the time.**
- Blocks with any GPU work show a one-sample spike 5 to 10 s after they end. A gate that reads headroom once at
  the wrong moment would see it; the r16 gate waits through it.
- *External*: the same paper measured energy per prefill token of 0.013 mAh on the GPU against 0.038 on the CPU
  with instruction acceleration (Adreno 730 tablet). Times its rates that is about the same device power on
  either side (0.60 against 0.49 mAh per second). **The GPU saves heat only by finishing sooner.** On our phone it
  does not finish much sooner (§1.2). Bonnie r5 §5 quotes up to 11 W for this GPU under a 3D load (*external*).

Clock state at the first decode token (*estimated*; r4 §3 for P0):

| | load before the decode | headroom rise before the decode | P(C ≥ 40) / P(33 to 40) / P(lower) |
|---|---|---|---|
| P0 | 6 to 15 s, four big cores | +0.08 to +0.20 | 0.55 / 0.35 / 0.10 |
| P1 | 5 to 15 s, GPU + one small core | +0.04 to +0.12 | 0.65 / 0.28 / 0.07 |
| P2 | 3 to 8.5 s, both | +0.05 to +0.13 | 0.62 / 0.30 / 0.08 |

- Why the gain is small: within one gated run the caps react after about 15 s (*device*, R74), so in the centre
  case P0's run also ends before the first cap step. P1 only moves the slow tail.
- Price: C 40 against C 35 is 0.3 to 0.5 tok/s on each subject (Bonnie r5 §4: 0.28 to 0.32 in her model, 0.43 to
  0.48 in `pair`), and about 1 tok/s against C 31. A shift of 0.10 of probability from the middle state and 0.03
  from the low one is **+0.03 to +0.10 tok/s** in expectation, and nothing at the median of a 10-run block unless
  more than half the runs were capped.
- Two side effects, both unmeasured:
  - **For the lane:** the GPU arrives at the decode already at its top clock. R74 read `devfreq_cur=315 MHz` of
    840 before the blocks (*device*). If part of the 200 µs submit-to-start gap is a GPU waking up, a warm GPU
    cuts K. x4b's `submit_to_start_us` across its cells is the first hint (bimodal in Bonnie's band).
  - **Against the lane:** the big cores sleep through a P1 prefill and their governor falls. The runtime must
    spin the four workers for the last 50 to 500 ms of the prefill (the kit's `ramp_s` logic).
  - GPU-on time per run goes from 3 to 4 s to 10 to 19 s. Bonnie's P(the GPU's cap falls somewhere in a 10-run
    block) 0.2 becomes about 0.35 (mine).
- **The cheapest comparator is a pause.** If the whole value is a cooler decode start, 5 to 10 s of rest between
  prefill and decode buys more at no build cost. It changes the run protocol and the user's wait, so it is an
  owner question, not something I propose.

### 1.4 Numerics (*source*, *host*)

| kernel | activations | product | accumulation | class |
|---|---|---|---|---|
| CPU, KleidiAI gemv and tile | int8 per block of 32 | exact integer | float, one FMLA per block | the stock reference on the phone |
| GPU gemv, decode (R74) | float | `(q − 8) × d` in half | **float** (`float2 totalSum`) | Tom's 0006 emulation; identity 6e-4 to 8e-4 (*device*) |
| GPU `_mc3`, 2 to 4 columns | float | half | float | as the gemv |
| **GPU gemm, prefill** | **half** | half | **half** (`half8 c0..c3`, one add per weight, 4,096 or 11,008 adds) | new; not emulated |
| GPU dp4a gemm, prefill | int8 per block of 32 | exact integer | float4 | near the CPU's; the block quantiser differs in rounding (127 ÷ amax on the GPU; 1 ÷ (amax ÷ 127) in KleidiAI's `kai_lhs_quant_pack_qsi8d32p_f32.c`, lines 92 to 93, and in ggml's Q8_0) |

- *Host*, `ebony_h16acc.py` (random Q4_0-like weights and Gaussian or log-normal activations, 4,000 rows each,
  product rounded to half then added; a float32 self-check passes):

  | K | rms error ÷ rms of the output | 99th percentile |
  |---|---|---|
  | 4,096 (q, k, v, o, gate, up) | 0.0091 / 0.0089 | 0.032 / 0.029 |
  | 11,008 (down) | 0.0144 / 0.0142 | 0.048 / 0.049 |

  This is random data; real sums with cancellation can be worse relative to a small output.
- Against Q4_0's own noise: a 16-level block quantiser leaves about 7 to 8 % of the weights' rms (*estimated*:
  step ÷ √12 with the block maximum at about 2.1 σ). Half accumulation adds 1.4 to 3.9 % of that noise power.
  Q4_0 costs +1 to +4 % of perplexity against F16 (*host*, S3 freeze). **My band for the half-float gemm at
  `-b 512`: +0.02 to +0.3 % on every row (P1), a third of that on the lane's share (P2).** P(it stays under
  +0.5 %) 0.85.
- The tail risk is range, not rounding: a half holds at most 65,504. A matmul input or output above that becomes
  infinite. Llama-2-7B is reported to carry activations of about 2,500 in a few fixed dimensions (*external*,
  "Massive Activations in Large Language Models", arXiv 2402.17762, **from memory, not re-read today**). That is
  inside the range; Yi and DeepSeek are unknown. P(some subject overflows) 0.05.
- **The owner's rule on prefill and decode binds here.** With P1 or P2 the `-b 512` check measures the batched
  kernel and the `-ub 1` check measures the lane; both must sit inside the one +5 % budget beside whatever else
  spends it (the k,v-dense rule alone is at +3.40 % and +3.55 % on Llama-2 and DeepSeek, *host*).
  Neither check covers the real run, where the KV comes from the batched kernel and the scored tokens from the
  lane. A third check is needed: first 256 tokens of each chunk at batch, the scored 256 one by one.
- The dp4a path would spend almost nothing (its error is float rounding and rare ties in the int8 rounding; my
  band ±0.02 %) and cannot overflow. P(`cl_khr_integer_dot_product` is in the 660's extension string) 0.3.

### 1.5 What a device cell would show (a rider for the next lane kit, not r35)

`x5p`, model-free, about 70 s of load and four gates:
1. **Build and identity:** the half-float gemm and, if the extension is there, the dp4a pair, at 11008 × 4096 and
   4096 × 11008 with N = 152 and 184, against float64. Pass line for the half-float tile: rms error ≤ 2 % of the
   output's rms and no non-finite value (x4m's 1e-2 line would fail a correct kernel at K 11,008).
2. **Rate:** tokens per second of one Llama-2 layer's seven matrices at N = 184, device time and wall.
   Closes P1 if under 15 tok/s-equivalent of a body.
3. **The heat question itself,** ABBA with a gate before each arm: A = 8 s of the GPU batched kernel, then at
   once 4 s of the CPU gemv arm on 4-7 (x3's `alone` arm); B = 8 s of KleidiAI's tile at N = 184 on 4-7, then the
   same 4 s arm. Read the CPU arm's `gbs`, `mhz_range`, `max_mhz` and `hr` after A against after B, and the gate
   series after each.
   - A's CPU arm ≥ 1.05 × B's, or B's `max_mhz` below hardware maximum while A's is not → a GPU prefill changes
     the decode's start; price it at the measured ratio.
   - Within ±3 % and both at full caps → closed for the claim run.
4. **Free now:** R75's `x4 ext text=` (is `cl_khr_integer_dot_product` there?) and `x4 mcsum dev_ratio=`.

Host tests that come first, each closing a branch:

| # | test | cost | closes when |
|---|---|---|---|
| H-max | largest absolute value into and out of every Q4_0 matmul over 40 chunks, three subjects (a hook beside Tom's activation tools) | minutes | any value ≥ 32,000 closes the half-float gemm on that subject |
| H-gemm | the half-float tile emulated in the CPU backend behind a knob (NEON half arithmetic, 8 columns × 4 rows, both fused and unfused), per-chunk PPL at `-b 512`, 40 chunks, two subjects | about 10 s per chunk (*estimated*) | above +0.5 % on either → P1 closed; P2 only if the lane has budget left |
| H-mix | the mixed check of §1.4 with H-gemm for the first half and 0006's gemv emulation for the scored half | one small tool change | outside the budget → no GPU prefill in front of a lane |

## 2. One task for Tom: teacher-forced dumps and DeepSeek's head (my questions 3 and 4)

### 2.1 Correction: the rate

- *Host*, existing logs (`lab/c3/logs/ppl-llama2-7b-q4_0-b512-full.log`, `ppl-deepseek7b-q4_0-b512-40.log`):
  2.91 and 2.95 s per 512-token pass on 8 CPU threads, **174 to 176 rows/s**. One earlier log shows 8.20 s (62
  rows/s) with other jobs running.
- *External*, search summary of secondary pages citing llama.cpp discussion #4167 (I did not open the thread):
  M4 Pro, LLaMA-7B Q4_0, about 440 tok/s of prompt processing on Metal.
- Tom's greedy dumps: 118 to 135 rows/s on Metal at 16 sequences (*host*).
- So teacher forcing gives **1.4 × on the CPU and about 3.5 × on Metal**, not the 10 to 30 × I wrote in r4 §2.2.
  **By my own r4 line ("under 5 × closes the data route as a laptop job") the route would close on rate.** I
  keep one cheap step of it for a different reason: it reaches text the greedy route cannot (below), and it
  costs one tool mode.

### 2.2 What a teacher-forced row is, exactly

At decode, row j has state h_j and emits y_j; the head guesses y_{j+1}, the token emitted after y_j is fed back.
In a teacher-forced pass over a text t, row j emits a_j (the model's argmax) but row j + 1 is fed t_{j+1}.

- **Strict pair:** (h_j, a_{j+1}) kept only where a_j = t_{j+1}. Then the next input was the model's own choice
  and the pair is what the decode sees, except that the earlier context is the text's, not the model's.
- **Loose pair** (only for the `tok` head): (h_j, E[t_{j+1}]) → a_{j+1} on every row. The head learns "given this
  state and that this token came next, what follows"; at decode the token is always the argmax.
- Strict share: the model's top-1 on the text, 0.35 to 0.50 on human-written answers (*estimated*). It selects
  the easy positions, which is a bias.
- Dolly's answers are 5.38 M characters in 15,011 records (*host*), about 1.3 M tokens: **0.45 to 0.65 M strict
  pairs**, one doubling over Llama-2's 444,000. Dolly's unused prompts would give 1.7 M greedy pairs in about 4 h
  per model at Tom's rate. Beyond that both routes need another corpus, and a download is the lead's call.

### 2.3 The task, in order

Inputs that exist: `tools/c3_head_dump.cpp`, `c3_head_prompts.py`, `c3_head_train.py`, `c3_head_t1.py`;
`lab/c3/head/{metal,cpu}` builds; `lab/c3/head/dolly/databricks-dolly-15k.jsonl` (sha256 `2df90833…94ec`);
`lab/c3/head/prompts/{train1500,train1500b,eval,official}.txt`; Llama-2's and Yi's dumps and heads.

**Step 0, rates (10 minutes, laptop otherwise idle).** Three two-minute runs on Llama-2, Metal:
greedy at `n_par` 16 (Tom's reference), greedy at `n_par` 32, teacher-forced at one sequence per decode.
Print rows/s and, for the third, the strict share.
- Teacher-forced rows/s × strict share < 1.5 × the greedy rate → steps 1 and 2b are dropped; go to step 2a.

**Step 1, tool mode.** `c3_head_dump` gets a teacher-forced mode (the context already runs with
`embeddings = true`, so every position's state and logits are available):
- input: two files in the same line order, the prompts and the answers (`c3_head_prompts.py` writes the answers
  of the same records; the split rule is unchanged, so the 32 prompt-set records stay out);
- per record: the model's chat template on the user message with the assistant header, then the answer's tokens;
  one `llama_decode` of the whole sequence with every position marked for output;
- rows written for the last prompt token and every answer position, in the present format (`.rows`
  {prompt, step, argmax, pos}, `.h16`, `.topk`), plus one new file `.tin`: int32 per row, the text token fed at
  the next position (−1 on the last row);
- `c3_head_train.py`: `--strict` keeps a pair (j, j + 1) only if `rows[j].token == tin[j]`; `--loose` feeds
  E[`tin[j]`] to the `tok` block. One assert in `selftest()` for each.
- Self-check of the mode: on a record whose "answer" is the model's own greedy output from an existing dump,
  every row's argmax must equal the greedy dump's token and the strict share must be 1.0 (states equal within
  fp16 when both run on the CPU).

**Step 2a, DeepSeek's head by the existing route** (the number that replaces my 0.438 assumption):
- file `lab/c3/models/deepseek7b-q4_0.gguf` (the unmodified one, owner 2026-11-22), its sha256 in the head's
  metadata; template from the file;
- `train1500.txt` + `train1500b.txt`, 192 new tokens at most, Metal, 16 sequences → about 450,000 rows;
- train Medusa-1 and `tok`, 2 epochs, as the other two; ridge first;
- prompt-set and official dumps on the CPU, one sequence, **run alone**;
- shortlist at 4,096 / 8,192 / 16,384 (vocabulary 102,400: the full head costs 13.88 ms on the phone, *device*,
  R72, so the shortlist row is the one that prices).

**Step 2b, transfer test on Llama-2** (only if step 0 passed):
- teacher-forced dump over the answers of the same 3,000 training records;
- three heads, all scored on the greedy prompt set as now: (a) strict pairs alone, Medusa-1; (b) greedy 444,000
  + strict, Medusa-1 and `tok`; (c) greedy 444,000 + loose, `tok`.

### 2.4 Checks that prove transfer to greedy decode

| check | pass |
|---|---|
| `base`: the dequantised head on stored states reproduces llama.cpp's greedy token | ≥ 0.999 on training rows, ≥ 0.99 on CPU prompt-set rows (Tom had 0.9999 / 1.0000 and 0.9929 / 0.9994) |
| scoring set | always the CPU, one-sequence, greedy dumps of the 32 prompts and the official prompt; never teacher-forced rows |
| held-out against prompt set | within 0.02 (as Tom found); a teacher-forced held-out score is reported but is not the result |
| step 2b (b) against greedy-only (0.476 Medusa-1, 0.497 `tok`, *host*) | ≥ +0.015 → teacher-forced rows transfer (about half of a greedy doubling's +0.04); ≤ +0.005 → they do not, and the mode is closed |
| step 2b (a) | reported only: it shows the gap between the two kinds of row at equal count |
| end to end, when the width-2 pass exists (T10) | tokens per pass in a real decode within 0.01 of the simulated `tau`, and the text identical to plain greedy on the prompt set |

Bands (*estimated*): DeepSeek prompt-set top-1 at 450,000 pairs **0.43 (0.36 to 0.50)**, unchanged from r4.
Step 2b (b) on Llama-2: +0.02 (0.00 to +0.04), so 0.50 to 0.54 for `tok`. P(passes +0.015) 0.5.
These do not move the stack's verdict: r4 §2.1 needs 0.53 to 0.60 on DeepSeek at K 290.

### 2.5 Disk and time

- Free disk today: 44 GB (*host*, `df`). `lab/c3/head` holds 13 GB.
- Rows cost 8.2 KB each (*host*, Tom: 226,305 rows in 1.85 GB) plus 272 B of ids.
  Step 2a: 3.7 GB. Step 2b: 1.3 M rows, 10.7 GB if every row is kept, **4 to 5.5 GB if only strict rows and their
  successors are written** (a writer-side filter; loose training then needs the full file).
- Time: step 0 10 min; step 1 an hour of code; step 2a 1 to 1.5 h of Metal dumps, 5 minutes of training, 15
  minutes of CPU dumps; step 2b 1 to 2.5 h of dumps (CPU or Metal by step 0) and 10 minutes of training.
  Nothing should start while a perplexity job runs (Tom saw CPU dumps slow 50 × beside a Metal job).

### 2.6 Licence of DeepSeek-LLM-7B-chat for this use (*external*, read 2026-11-23)

The model card says "The use of DeepSeek LLM models is subject to the Model License" and "DeepSeek LLM supports
commercial use" (<https://huggingface.co/deepseek-ai/deepseek-llm-7b-chat>). The licence is the "DEEPSEEK LICENSE
AGREEMENT", "Version 1.0, 23 October 2023"
(<https://raw.githubusercontent.com/deepseek-ai/DeepSeek-LLM/main/LICENSE-MODEL>, read through the fetch tool,
which returned the quoted passages; paragraph 4 came back partly as excerpts).

- Definition: "\"Derivatives of the Model\" means all modifications to the Model, works based on the Model, or any
  other model which is created or initialized by transfer of patterns of the weights, parameters, activations or
  output of the Model, in order to cause the other model to perform similarly to the Model, including - but not
  limited to - distillation methods entailing the use of intermediate data representations or methods based on
  the generation of synthetic data by the Model for training the other model."
  **Our head is this:** it is trained on the model's activations and outputs to predict the model's own tokens.
- Grant: "DeepSeek hereby grants to You a perpetual, worldwide, non-exclusive, no-charge, royalty-free,
  irrevocable copyright license to reproduce, prepare, publicly display, publicly perform, sublicense, and
  distribute the Complementary Material, the Model, and Derivatives of the Model."
- Use: "Use may include creating any content with, finetuning, updating, running, training, evaluating and/or
  reparametrizing the Model."
- Output: "Except as set forth herein, DeepSeek claims no rights in the Output You generate using the Model.
  You are accountable for the Output you generate and its subsequent uses."
- On distribution (paragraph 4): "Use-based restrictions as referenced in paragraph 5 MUST be included as an
  enforceable provision by You in any type of legal agreement (e.g. a license) governing the use and/or
  distribution of the Model or Derivatives of the Model"; "You must give any Third Party recipients of the Model
  or Derivatives of the Model a copy of this License;" "You must cause any modified files to carry prominent
  notices stating that You changed the files;" "You must retain all copyright, patent, trademark, and attribution
  notices".
- Attachment A lists the restricted uses (unlawful use, "For military use in any way", harm to minors, false
  information meant to harm, and seven more).

My reading, for the lead (I am not a lawyer): training and testing the head on the laptop is use, and allowed.
Unlike the Llama 2 licence there is no clause against training on outputs. A head that ships (an app sidecar, a
public repository, a kit sent to anyone outside) is a distributed Derivative and carries the four duties above.
The kit repository is private and the owner's; whether that is "distribution to a third party" is the owner's
reading. The training prompts are Dolly's (CC BY-SA 3.0, as Tom recorded).

## 3. What makes the W-position CPU kernel exact inside llama.cpp (my question 2)

### 3.1 The reference is KleidiAI, not ggml's repack path

- The app runs KleidiAI (R72-result line 136; `android/build_cpu.sh`, `KLEIDI` on by default). The host freeze and
  Tom's head dumps ran ggml's repack path (`REPACK = 1`, KleidiAI off). "Bit-identical" is per path, so the
  kernel is checked against the path the kit serves.
- `ebony_gw.c` was written against `ggml_gemv_q4_0_4x4_q8_0`. *Source*, the KleidiAI gemv
  `kai_kernel_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4x4_1x4_neon_dotprod` (assembly in
  `lab/llama.cpp-c1h6/build-kai/_deps/kleidiai-src`; Bonnie r5 §2.1 says this copy is v1.31.0, the pinned
  version; I did not find a version file in it), per block of 32 weights × 4 outputs:
  FCVTL of both scales, 8 SDOT into one zeroed sum, `fmul` (weight scales × activation scale), `scvtf #4`,
  `fmla`; after the last block `fmax` and `fmin` against the clamp values (−FLT_MAX, FLT_MAX).
- **That is my kernel's float sequence, in the same order** (the FMUL's operands are swapped, which changes no
  bit). Two differences: the clamp, which changes an infinite sum to ±FLT_MAX, and the packed activation layout
  (a half scale followed by the int8 values per block, read with one stride).
- Tom's T1b (*host*): KleidiAI's tile at batch 2 gives the same bits as its gemv at batch 1 on 128 of 128 rows.
  So the tile's float sequence per cell is the gemv's, and a two-row cut of the tile is a second way to the same
  kernel.

### 3.2 The eight conditions

1. **Activations are quantised and packed by the library's own routine, one position at a time**
   (`kai_run_lhs_quant_pack_qsi8d32p_f32` with m = 1, as the gemv path does). Then each position's int8 values
   and scale are the batch-1 bytes by construction. No hand-written quantiser.
2. **The float sequence per output is the gemv's:** one FMUL of the scales, one SCVTF with 4 fractional bits, one
   FMLA per block, blocks in ascending order, then FMAX and FMIN. The integer dots are exact in any order.
3. **No contraction and no re-association by the compiler.** Write the float steps with the explicit intrinsics
   (`vfmaq_f32` where stock has FMLA, never a multiply and an add that could fuse), compile the file with
   `-ffp-contract=off`, and check the object: the list of float instructions per block in the shipped `.so` must
   be FCVTL, FCVTL, FMUL, SCVTF, FMLA per position and nothing else (a short disassembly check, as
   `tools/c3_a78_pipes.py` already reads assembly).
4. **Scales converted once per call are exact** (half to float is exact), so hoisting them changes nothing. This
   is the only hoisting in the kernel besides the unpack.
5. **Work size.** At ne1 = 2 ggml sizes `wdata` for the tile's packing (`tensor_traits::work_size`, `is_gemv`
   false). The new path packs W rows at m = 1; `work_size` must return W × that size, or the pack writes past
   the buffer. This is a memory-safety condition, found by reading, not an arithmetic one.
6. **Chunks and rows.** An output depends only on its own weight row, so column chunks, thread count and the
   lane's row range cannot change bits, provided ranges start on a multiple of 4 outputs (`n_step`) and the
   packed-weight offset comes from the kernel's own `get_rhs_packed_offset`. The subjects' shapes are multiples
   of 4. The output head is Q6_K and stays on its stock kernel.
7. **The same kernel variant on the host and the phone.** The M4 has i8mm and SME; the phone's cores have neither.
   Host checks must run with the dotprod entry selected and print it (Tom's T1b log: "primary q4 kernel feature
   DOTPROD", `GGML_KLEIDIAI_SME=0`). No report of ours prints that line from the phone (Bonnie r5 §2.1); the kit
   that carries the kernel should.
8. **A self-check at load, kept in the build:** one random activation through the stock gemv and through the new
   kernel at W = 1 on a real weight tensor, `memcmp`; on a difference the knob turns itself off and says so. It
   costs milliseconds and catches a KleidiAI version change, which nothing else would.

Outside the kernel, already shown or someone else's test: the rest of the graph at batch 2 is exact on both stock
paths (T1b, *host*); the KV rollback after a missed guess is T10.

### 3.3 Test design for Tom (replaces "an arm on the W-position kernel" in memo-10 T10)

| part | what | where | pass |
|---|---|---|---|
| A, shadow check on real data | behind `VOLLIE_GW_CHECK=1`, in `compute_forward_qx`: for every Q4_0 `MUL_MAT` with ne1 = 1 to 4, run stock into `dst` and the new kernel into scratch, `memcmp`, count; print the first difference (tensor, row, position, both values in hex). Run on two subjects: the official prompt and 4 prompt-set prompts decoded one token at a time and two at a time (T1b's feed), and one perplexity chunk at `-ub 2`, `-ub 3`, `-ub 4` | host, KleidiAI build, dotprod entry; minutes | 0 differences over every call; the count of calls printed |
| B, edge vectors | model-free unit check beside `ebony_gw.c`: an all-zero block (scale 0), one value of 1e30 in a block, +inf and NaN inputs (stock's clamp and NaN behaviour must be reproduced), a denormal scale, 1e6 random cases; W 1 to 4; 1 to 4 threads; a row range that does not start at 0 | host; seconds | `memcmp` equal in every case |
| C, end to end | per-chunk perplexity, knob on against off, at `-ub 1`, `-ub 2` (new: the only setting that runs W = 2), `-ub 32` and `-b 512` (must still take the stock tile), 40 chunks, two subjects; greedy text on the prompt set with the head-driven width-2 pass, misses included | host; under an hour | every per-chunk value identical; text identical |
| D, on the phone | part A's counter in the kit that first carries the kernel: the official prompt, 32 tokens, two subjects; the `kleidiai: primary q4 kernel feature` line in the report; the disassembly check of condition 3 on the shipped library | device; seconds | 0 differences; DOTPROD printed |

- Any difference in A, C or D: not exact as built. Then it is fixed or declared a numerics change with its own
  budget check; it does not ship as bit-identical.
- Part A is the one that matters most: it sees every layer's real activation range, which random data does not.
- Cost of the kernel itself, unchanged from memo-10 E2: 36 slots against 25 (c 1.44) by count; KleidiAI's tile
  is 65 (Bonnie r5). R75's `x4 tog2sum cpu_alone_gbs=` is the first *device* value for the tile.

## 4. My rows on Tom's field names (item 4; nothing in a band changes)

| my row (r3 §1, r4 §1) | printed field (tom-r35-build.md, 19:20) | band |
|---|---|---|
| together, 2 columns: CPU alone | `x4 tog2sum cpu_alone_gbs=` | **14.5; 12.2–18.0** (was 15.5; 13.0–22.5; moved after Bonnie's count of the tile) |
| CPU alone at 2 columns ÷ at 1 | `tog2sum cpu_alone_gbs` ÷ `tog1sum cpu_alone_gbs` | 0.57; 0.47–0.72 |
| GPU alone, 2 columns | `x4 tog2sum gpu_alone_gbs=` | 19.8; 14.5–23.5 |
| both CPU / both GPU / sum | `both_cpu_gbs=` / `both_gpu_gbs=` / `sum_gbs=` | 14.0; 11.7–17.5 / 18.6; 13.5–22.0 / 32.6; 27.5–38.5 |
| **model line:** GPU both ÷ GPU alone | `both_gpu_gbs` ÷ `gpu_alone_gbs`; cross-check `gpu_alone_dev_us` ÷ `both_gpu_dev_us` | 0.94; 0.85–1.00 |
| sum ratio | `sum_gbs` ÷ (`cpu_alone_gbs` + `gpu_alone_gbs`) | 0.95; 0.85–1.00 |
| as printed | `x4 tog2sum sum_vs_cpu_alone=` (divides by the two-column CPU) | 2.25; 1.9–2.8 |
| skip | `x4 tog2 ok=0 reason= then=n_cols_1_only` | P 0.25 (x4m "all" is 0.75) |
| one column | `x4 tog1sum` (same fields) | R74's values; `sum_vs_cpu_alone` 1.61 (1.50–1.72) |
| x4m | `x4 mcsum n_cols= dev_ratio= wall_ratio=`; `x4 mcid max_rel_err= ok=` | 1.3 / 1.65 / 2.0; wall 1.22 at 2 columns; 6e-4 to 8e-4 |
| x4e C11: builds | `x4 pk c11 test= opt= err=0 svm_atomics=`; failure `mode=svm_c11 ok=0 reason=build_c11` | P 0.55 |
| x4e C11: flag seen given it runs | `x4 pk test=flag mode=svm_c11 seen=200 ok=1` | P 0.75 |
| x4e C11 round trip; counter interval | `rtt_p50_us=`; `x4 pk test=counter mode=svm_c11 iv_p50_us= visible_during_run=1` | 20; 3–200. 5; 1–60 |
| hand-off: write and read calls | `x4 handoff mode=write extra_us=` (`write_us` / `read_us` beside it) | 90; 40–200 (35 / 50) |
| hand-off: map and unmap | `mode=map` / `mode=ahp_map` | 70; 30–170 / 50; 20–130 (my 60; 25–150 was one row; split here) |
| hand-off: no call | `mode=ahp_persist extra_us=`, `match=` | 5; 0–30; P(`match=1`) 0.55 to 0.60 |
| hand-off: best ≤ 20 µs and correct | min `extra_us` over modes with `match=1` | P 0.50 |
| x4s lead | `x4 splitsum mode=lead share= ratio=`; K = `gpu_ms` − `gpu_dev_ms` on `x4 split` | as merged |

Notes:
- Tom built the C11 mode with acquire / release where I wrote seq_cst. No band changes: the question is whether a
  running kernel sees the write at all.
- `extra_us` is wall against the `none` mode, so it holds any change in the launch itself; `write_us` + `read_us`
  is the cleaner sum when they disagree (Tom's note says the same).
- The hand-off cell runs without CPU rows on one thread. It prices the moves, not their overlap with CPU work.

## 5. Review of bonnie-joins-r5.md

| her row or claim | my position |
|---|---|
| §0.1, §2.1: at ne1 = 2 the CPU runs KleidiAI's four-row tile, 65 slots against 25, c 2.5 (2.3 to 2.9); my line reference was the F32 function | **agree, and the correction is right:** the Q4_0 call is `compute_forward_qx`. My centre moves to 14.5 GB/s; P(under 19) 0.85 against her 0.90 |
| §1.1 events, every row | **agree as merged.** Two ranges stay ranges: `svm_fine_buffer` (her 0.60, mine 0.70) and x4r held (0.35 / 0.25) |
| §1.2 cells x4a, x4b, x4c, x4d, x4e, x4r | **agree as merged** (both centres printed where we differ) |
| x4f `rtargs` rewritten to `enq_us` at 2 args − 15 args: −10 (−22 to −3) | **agree.** Thirteen `SetKernelArg` calls saved at under 2 µs each |
| x4f on launcher 0-3 dropped (no cell) | agree |
| x4m, x4s, hand-off, C11 rows | agree; my fields in §4 |
| together at 2 columns: 14.1 / 13.7 / 18.5 / 32.5 | agree within 0.5 GB/s of my moved centres |
| `sum_vs_cpu_alone` as printed, 2.30 (1.9 to 2.8) | agree (mine 2.25) |
| §2.3 change 1: below 19 GB/s of CPU alone the GPU ratio decides `bonnie` alone | **agree.** My "and" could not be met under the tile |
| §2.3 change 2: a `pair` reading from the tile cell is `pair` at light CPU draw | **agree.** The tile draws 14 GB/s of bus where a lane draws 20 to 22 |
| §2.4 mixed arm (CPU at 1 column beside the GPU's `_mc3` at 4) | **agree it is the right cheap arm.** My band for its CPU ratio: 1.20 (1.00 to 1.30); P(≥ 1.15) 0.5, P(≤ 1.05) 0.25. If r35 is frozen it belongs in the next kit |
| §4 a share that follows the cap: +0.24 to +0.45 tok/s per step, 0.16 to 0.25 GB | arithmetic agreed. **My position: do not build it.** It makes the output depend on the phone's heat, so no host check reproduces a run. Her fixed share tuned for C 35 to 37 loses 0.2 to 0.3 and keeps one numerics |
| §5 no GPU throttle in 80 s of R74; P 0.05 inside a claim run | agree. A GPU prefill would raise the per-block figure (§1.3) |
| §0.7 width 3 on the lane closed at the merged centre | agree |
| "a prefill batch pays 0.65 × a gemv per position" | agree; it is the basis of my P0 row |

Not reviewed: her §6 and §7 (width 3 detail, the file-backed residency arm); I read §0's summary of them only.

**One thing neither of us has priced and both stacks assume:** every stack since memo-10 uses c(2) = 1.45, a kernel
that exists as a scratch file. §3 is what stands between that file and a device prediction.

## 6. Lists (through the lead; additions only)

**Tests**

| # | test | where | cost | closes when |
|---|---|---|---|---|
| E-T1 | R75 `x4 ext text=`: is `cl_khr_integer_dot_product` listed? | device, free | 0 | absent → the integer prefill kernel is closed on this phone |
| E-T2 | H-max, H-gemm, H-mix (§1.5) | host | minutes; 10 s per chunk; one tool change | as written there |
| E-T3 | Head task, steps 0 to 2b (§2.3) | host | half a day of laptop time, 4 to 15 GB | as §2.4 |
| E-T4 | W-kernel parts A, B, C (§3.3) | host | about an hour after the kernel is in the tree | any difference → not exact as built |
| E-T5 | W-kernel part D and the kernel-feature log line | device, in the kit that carries the kernel | seconds | any difference |
| E-T6 | `x5p` (§1.5) | device, a later kit | about 70 s of load, four gates | CPU arm after GPU prefill within ±3 % of after CPU prefill → closed |

**Try**

| # | lever | after | label | price | P |
|---|---|---|---|---|---|
| E-Y1 | W-position kernel in KleidiAI's arithmetic with the load-time self-check (§3.2) | E-T4 | bit-identical | it is the c(2) 1.45 every width-2 stack assumes; without it stock is 2.5 | 0.7 that parts A to C pass at the first build |
| E-Y2 | Lane prefill (P2) | E-T2, E-T6 and a lane that exists | numerics change (half-float tile on the GPU's rows) or near-identical (dp4a) | prefill 1.4 to 1.6 × faster; decode +0.02 to +0.06 tok/s | 0.3 |
| E-Y3 | GPU-only streamed prefill (P1) | E-T2, E-T6 | numerics change | decode +0.03 to +0.10 tok/s; not proposed for the claim run | 0.15 |

## 7. Next questions, ranked

| # | question | state |
|---|---|---|
| 1 | Is the W-position kernel exact on real activations (part A)? | designed (§3.3); Tom's. The largest unpriced assumption in both stacks |
| 2 | Does `x4 mcsum dev_ratio(4)` put a float-sum GPU prefill above or below 10 tok/s; is the integer-dot extension on the phone? | R75, free |
| 3 | Largest activation into each matmul on the three subjects (H-max) | host, minutes; also tells whether the lane's half product can overflow at decode, which 0006 may not flag |
| 4 | Does a warm GPU start cut the launch gap? `x4 rt submit_to_start_us` early against late cells of x4b, and `gpu_mhz` beside them | R75, free; if yes, a 2 ms GPU warm-up before the first token is a zero-cost lever |
| 5 | Greedy rows/s at 32 sequences against 16 (step 0): is the plain route simply faster than teacher forcing? | host, 4 minutes |
| 6 | The host reference runs ggml's repack path and the phone runs KleidiAI: how far apart are their per-chunk perplexities on the three served files? | host, one `-b 512` and one `-ub 1` run per subject on a KleidiAI build; it bounds what "stock" means in every budget check so far |
| 7 | What does a 5 to 10 s pause between prefill and decode do to the first-token clocks? | an owner question first (§1.3); a kit could log it in two runs |
| 8 | Is a kit sent to the owner's private repository "distribution" under the DeepSeek licence? | the owner's reading |

## 8. Sources (2026-11-23)

Repository: `lab/llama.cpp-c3/ggml/src/ggml-opencl/ggml-opencl.cpp` (lines 285 to 300, 3792 to 3804, 6928 to 6932,
8760 to 9002, 19960 to 20270), `kernels/gemm_noshuffle_q4_0_f32.cl`, `gemm_noshuffle_q4_0_q8_1_dp4a.cl`,
`quant_a_q8_1.cl`, `gemv_noshuffle_q4_0_f32.cl`; `docs/backend/OPENCL.md`;
`ggml-cpu/kleidiai/kleidiai.cpp` (lines 668 to 975, 1136 to 1300), `kleidiai/kernels.cpp` (dotprod entry);
`lab/llama.cpp-c1h6/build-kai/_deps/kleidiai-src/.../kai_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4x4_1x4_neon_dotprod_asm.S`
and `.c`; `tools/c3_head_dump.cpp`, `c3_head_train.py`; `memo-09-work/tom-head.md`; `memo-10-tree.md` E2, E8;
device JSONs of R69, the R69 repeat, R72 and R74 (gate series, read with my own short code);
`lab/c3/logs/ppl-*-b512-*.log`; [tom-r35-build.md](tom-r35-build.md); [bonnie-joins-r5.md](bonnie-joins-r5.md).

External:
- DeepSeek licence, Version 1.0, 23 October 2023:
  <https://raw.githubusercontent.com/deepseek-ai/DeepSeek-LLM/main/LICENSE-MODEL>; model card:
  <https://huggingface.co/deepseek-ai/deepseek-llm-7b-chat>.
- "Understanding Large Language Models in Your Pockets", arXiv 2410.03613v5, Table V and §V-C, §V-D (read through
  the fetch tool; the first 100,000 characters of the page): <https://arxiv.org/html/2410.03613v5>.
- mllm-NPU (arXiv 2407.05858) and HeteroLLM (arXiv 2501.14794): **search summary only** this round; both put the
  prefill on an accelerator, neither reports the decode's clocks afterwards.
- M4 Pro prompt-processing rate: **search summary** of pages citing
  <https://github.com/ggml-org/llama.cpp/discussions/4167>; not opened.
- Not read: the IWOCL 2025 slides on the backend (a PDF, no reader here); Qualcomm's two blog posts (the pages
  returned no text).
- Not found: any prefill rate for an Adreno 6xx phone with llama.cpp's OpenCL backend; any paper measuring the
  CPU's clock state at decode start after a GPU or NPU prefill.
