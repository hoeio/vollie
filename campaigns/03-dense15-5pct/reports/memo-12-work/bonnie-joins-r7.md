# c3 memo-12 work, round 7 (Bonnie): the packed side file as runtime; the batched GPU kernel's half-precision sums; review of Ebony r5 §1.4 and §3

- **Date:** 2026-11-23. **Author:** bonnie, `claude-fable-5-2`. Follows [bonnie-joins-r6.md](bonnie-joins-r6.md);
  reviews [ebony-joins-r5.md](ebony-joins-r5.md) §1.4 and §3 (read in full this round). Not committed. No verdict
  here. **Ebony has not reviewed this file.**
- **Labels.** *device*, *derived*, *host*, *estimated*, *external*, *source* as before. Triples are Yi / Llama-2 /
  DeepSeek.
- Nothing was measured this round. Everything in §1 is *source* (read in `lab/llama.cpp-c3r32`, whose ggml sources
  are the kit's) or *estimated*. No web source: nothing here states the state of a model or of upstream beyond
  the local tree.

## 0. Answer first

1. **The packed side file is a small runtime change with no numerics.** KleidiAI's Q4_0 layout is the served
   bytes re-ordered and XOR-ed with 0x88, at the same 0.5625 bytes per weight. The buffer type today allocates
   anonymous memory and packs into it on every load. Mapping a file instead needs one new allocation path and a
   skip in `set_tensor`. The bytes the kernels read are the same, so per-chunk perplexity must be identical.
2. **Storage per model** (*estimated*): the CPU's rows only.

   | | served file | side file, CPU-only (Q4_0 body + head) | side file beside a lane at GPU share 0.37 (body rows + head) |
   |---|---|---|---|
   | Yi | 3.48 GB | 3.12 + 0.26 = 3.4 GB | 1.97 + 0.26 = 2.2 GB |
   | Llama-2 | 3.83 GB | 3.64 + 0.13 = 3.8 GB | 2.29 + 0.13 = 2.4 GB |
   | DeepSeek | 4.00 GB | 3.41 + 0.42 = 3.8 GB | 2.15 + 0.42 = 2.6 GB |

   So 1.6 to 2.0 × the served file per model; 7.2 to 11.0 GB more for all three. (Tensor arithmetic at 0.5625
   bytes per Q4_0 weight and 1 byte per head weight. R72's 3.36 / 4.02 / 4.01 GB are resident totals from
   `bonnie_path.py`, not these file sizes.)
3. **First-run pack: 5 to 30 s per model** (*estimated*; nothing on the phone has timed it). It is not new work:
   every load packs the whole body today. The side file moves that cost from every load to the first one.
4. **What it buys:** the body becomes file-backed. That is the only state in which Llama-2 and DeepSeek have ever
   been shown to fit (4.2 GB file-backed held, 4.2 GB anonymous did not; *device*, R69, R72), and r6 showed the
   served file itself cannot serve a width-2 lane.
5. **Owner question (a) is already answered in the rules' text; (b) is a real choice.** Rule 1 of
   `COMMON_RULES.md` says a derived artefact that the runtime computes once on the phone from the served file and
   caches in the app's files counts as runtime. What the owner has to decide is in §1.6: storage, where the file
   lives, and what the user sees on first run.
6. **Ebony's half-precision reading is right** (*source*): upstream's batched Q4_0 kernel keeps `half8`
   accumulators and adds K products into them. Her simulated error follows a random-walk formula I derive in §2:
   relative error ≈ 2.0e-4 × √(K ÷ 2), which gives 0.0091 and 0.0148 against her 0.0091 and 0.0144. Her PPL band
   (+0.02 to +0.3 %) is agreed; my centre is +0.1 %.
7. **Her eight conditions for the W-position kernel: agreed, with two additions** (§3): run her shadow check on
   the lane's own row range, and pin the KleidiAI version in the self-check's failure message.

## 1. The packed side file (next question 3): a design the owner can decide on

### 1.1 What the runtime does today (*source*)

- `llama-model.cpp` lines 1897 to 1945: a context's tensors point into the mapped model file only if the buffer
  type is the device's default one. The KleidiAI buffer type is an "extra" type, so its tensors get
  `ggml_backend_alloc_ctx_tensors_from_buft`: one allocation for all of them.
- `kleidiai.cpp` line 1705: that allocation is `ggml_backend_buft_alloc_buffer(ggml_backend_cpu_buffer_type(), size)`,
  plain anonymous memory. The size per tensor comes from `get_alloc_size` (line 1725): a 64-byte-aligned header
  (magic, version, slot count, offsets, sizes) plus the packed size.
- `set_tensor` (line 1687) calls `repack` (line 1494): it writes the header and calls
  `kai_run_rhs_pack_nxk_qsi4c32pscalef16_qsu4c32s16s0`. That routine is scalar C: per block it copies the four
  rows' half scales, then interleaves the four rows' nibbles two bytes at a time and XORs them with 0x88.
- So the packed tensor is a pure function of the served tensor, the kernel's `nr`, `kr`, block length and the
  routine's version. Rows are stored in groups of four, group after group: **a row range that starts on a
  multiple of four is a contiguous byte range**, which is what a lane's CPU share needs.
- On an A78 only one kernel slot is packed (the dotprod entry), so one copy per tensor.

### 1.2 The change

| piece | what | size of the change |
|---|---|---|
| allocation | behind a knob (off by default), `alloc_buffer` maps a file in the app's files directory, shared, instead of allocating anonymous memory. First run: created at `size`, mapped read-write. Later runs: mapped read-only | one function, about 60 lines with error paths |
| skip | `set_tensor` returns at once when the file was found valid. The loader then never touches the served file's body pages on those runs | about 10 lines |
| key | a file header: sha256 of the served file (the app already holds it), KleidiAI version, kernel name, `nr`, `kr`, block length, buffer size, and per tensor: name, offset, size. Checked against what the allocator asks for; any mismatch rebuilds | about 80 lines |
| integrity | written under a temporary name, `msync`, a checksum over the body in the header, then rename. A killed first run leaves no valid file | about 30 lines |
| pressure during the first run | after each tensor is packed: `msync` its range and release the served source pages. Campaign 1 already has this knob for the source side (`GGML_REPACK_RELEASE`, `docs/runtime-knobs.md`) | reuse |
| residency | campaign 2's page hold through the gate wait works on file pages as it does today | none |

- It is not a numerics change: the kernels read the same bytes at the same alignment (offsets inside the buffer
  come from the same allocator). Check, as for any runtime change: per-chunk perplexity identical with the knob
  on and off at `-ub 1`, `-ub 32` and `-b 512`, on two subjects, on the laptop's KleidiAI build with the dotprod
  entry; and identical between a first run and a second run.
- Risk to check in the tree, not assumed: several buffers per model (one per context key) each need their own
  file or one file with sections; and the Q6_K head goes through KleidiAI's 8-bit chain, which makes the head
  larger than its Q6_K source (8 bits per weight against 6.56).

### 1.3 Bytes

- Packed Q4_0 is 72 bytes per 128 weights, the same as the served 18 bytes per 32. The side file is as large as
  the rows it holds, plus 64 bytes per tensor and the header.
- CPU-only: the Q4_0 matmul body, 3.12 / 3.64 / 3.41 GB, plus the head if it stays in this buffer type (0.26 /
  0.13 / 0.42 GB at 8 bits per weight; 64,000 / 32,000 / 102,400 rows of 4,096).
- Beside a lane with a fixed GPU share of 0.37: 1.97 / 2.29 / 2.15 GB of body rows, plus the head. The GPU's rows need no file: their buffers
  are filled from the served file at every load (1.2 to 1.5 GB read and re-laid-out, *estimated* 2 to 5 s).
- Correction to r6 §0 item 2 and §1.3, which gave 2.4 to 2.9 GB for the lane's side file: by tensor arithmetic
  it is 2.2 to 2.6 GB with the head, 2.0 to 2.3 GB without.
- The served file stays on the phone: it is the frozen reference, the source of the GPU's rows, the token
  embedding and everything that is not a Q4_0 matmul.

### 1.4 Milliseconds

| event | today (anonymous) | with the side file | label |
|---|---|---|---|
| first load | read 3.4 to 4.0 GB from storage and pack it | the same, plus writing 2.2 to 3.8 GB back | *estimated* 5 to 30 s; pack rate 0.3 to 1 GB/s on one thread (scalar code), storage not measured for writes of this size except R69's 4.6 GB scratch file |
| every later load | the same read and pack again | map only; pages come in on first touch, 2.2 to 3.8 GB at the storage read rate | *estimated* 2 to 4 s spread over the prefill, against 4 to 13 s of packing saved |
| a pass, body resident | – | the same kernels on the same bytes | no change expected |
| a pass after eviction | swap-in from zram (an anonymous page costs compression work and memory to keep) | re-read from storage (a file page is dropped for free and costs one read) | this is the residency difference; r5 §7's arm measures it |

### 1.5 What decides whether it is worth building

- **For Yi: not needed** (3.36 GB resident, *estimated*, fits in the 3.4 GB that held, *device*; margin 0.04 GB). It would still cut
  every load after the first.
- **For Llama-2 and DeepSeek: required for any repacked body,** lane or not. Without it they run the stock
  kernel at 4.0 to 4.1 tok/s CPU-only or about 8 on a width-1 lane (r6 §1.3, *estimated*), or they thrash.
- **Not yet shown:** that file-backed pages stay resident beside 1.0 to 1.5 GB of GPU buffers (r5 §7's arm,
  P(held at 4.0 GB total) 0.55, my judgement), and that the phone's file-backed capacity of 4.2 GB (R69) holds
  for a map that is read by four threads at 20 GB/s rather than touched once.
- The side file and the arm are independent: the arm uses a scratch file and needs no loader change. **Run the
  arm first; build the side file only if step 2 or 3 of the arm holds.**

### 1.6 What the owner is asked

| question | what it decides | my reading of the options |
|---|---|---|
| (a) Is the packed side file runtime? | whether it may exist at all | Rule 1 already says yes for an artefact "the runtime computes once on the phone from the served file and caches in the app's files". The side file is exactly that, and it holds no information the served file lacks. What is new against the calibration tables: it is built on the phone, not shipped, so its sha cannot be listed in a pre-registration. The key must be the served file's sha plus the packer's version; the pre-registration lists those |
| (b) Is 1.6 to 2.0 × storage per model acceptable? | 2.2 to 3.8 GB more per installed model | Nothing cheaper gives a file-backed repacked body. Not keeping the served file is not an option under rule 1 |
| (b1) Where does it live? | whether Android may delete it | The app's files directory: kept until uninstall. The cache directory may be cleared by the system at any time, which makes a later run a first run again (5 to 30 s more, no wrong result) |
| (b2) What does the user see on first run? | rule 5, "fix it in the runtime, never in the user's phone" | The build happens inside the first load with no user action. A claim run is defined as a warm start already (campaign 2's result line); the first run after install is outside it |
| (c) Does the answer cover a file that is not a plain re-ordering? | later levers (a column-major copy, a 1-bit proxy) | Not needed for this design. Keep it separate: this file is byte-for-byte derived by upstream's own packer |

## 2. Upstream's batched GPU kernel sums in half precision (next question 6)

### 2.1 Source

`ggml-opencl/kernels/gemm_noshuffle_q4_0_f32.cl`, read in full (139 lines):
- accumulators `half8 c0, c1, c2, c3` (an 8 × 4 tile of outputs per fibre);
- activations read with `read_imageh` (half);
- weights dequantised as `((bits & 0xF) − 8) × scale` into a `half4`;
- `c += B × w` for every one of the K inputs, all in half;
- the result is converted to float only at the store.

So every output is a K-term sum carried in an 11-bit mantissa. **Ebony's reading is confirmed.** The decode gemv
and `_mc3` keep float sums (her table), so this is a property of the prefill kernel only.

### 2.2 Size of the error, derived

- One add into a partial sum s rounds to half: rms error about 2.0e-4 × |s| (half an ulp spread over the
  mantissa range).
- For terms of rms σ the partial sum after i adds is about σ√i. Independent rounding errors add in power:
  error² ≈ (2.0e-4)² σ² Σ i ≈ (2.0e-4)² σ² K² ÷ 2.
- The output's rms is σ√K. **Relative error ≈ 2.0e-4 × √(K ÷ 2).**

| K | formula | Ebony's simulation (*host*, random data) |
|---|---|---|
| 4,096 | 0.0091 | 0.0091 / 0.0089 |
| 11,008 | 0.0148 | 0.0144 / 0.0142 |

- Rounding each product to half adds 2.0e-4 of the output, which is nothing beside it.
- **The error scales with the partial sums, not with the output.** Where an output is small because large terms
  cancel, the relative error is larger than the table says by the ratio of the two. Where one input dominates
  (a "massive activation" T), every later add is rounded at T's ulp: error about 2.0e-4 × T × √(K left), up to
  1.3 % of T at K 4,096. That is the same order relative to that output, but large against the other, smaller
  outputs' needs only if T leaks into them; it does not, each output has its own sum.
- Range: a half holds 65,504. An activation or a partial sum above that is infinite. This is her tail risk and
  it is the one to test first (her `H-max` test: minutes).

### 2.3 Price in perplexity (*estimated*)

- Q4_0's own weight noise is about 7 to 8 % of the weights' rms and costs +2.6 to +3.8 % of perplexity against
  F16 on our subjects (*host*, S3 freeze). An output error of 0.9 to 1.5 % rms is 1.4 to 4 % of that noise
  power. Scaled linearly: **+0.04 to +0.15 %; my centre +0.1 %, band +0.03 to +0.4 %.** Hers: +0.02 to +0.3 %.
  Agreed.
- Both of us scale a noise power to a perplexity delta linearly from one anchor. That is the weak step; her
  `H-gemm` emulation (40 chunks, two subjects) replaces it.
- Her point on the rule stands and is the real cost: with a GPU prefill the `-b 512` check sees the batched
  kernel, the `-ub 1` check sees the lane, and neither sees a run whose KV comes from one and whose scored tokens
  come from the other. A third, mixed check is needed. With her own price for the prefill (+0.03 to +0.10 tok/s)
  that is three more full-set runs per subject for almost nothing. **Agreed: not a try-list item.**
- Her integer path (`cl_khr_integer_dot_product`, float accumulators) would not have this error. R75's
  `x4 ext text=` answers whether the phone has the extension.

## 3. Review of ebony-joins-r5.md §3 (the W-position kernel's exactness)

- **§3.1, the reference is KleidiAI, not ggml's repack path: agreed,** and it matters more than it looks. The
  host freeze, Tom's head dumps and every "bit-identical" check so far ran the repack path; the phone runs
  KleidiAI. Her float sequence for the KleidiAI gemv (FCVTL of both scales, 8 SDOT into a zeroed sum, FMUL,
  SCVTF with 4 fractional bits, FMLA, then FMAX and FMIN) is what I read in the same assembly in r5.
- **The eight conditions: agreed.** Notes on three:
  - Condition 1 (activations packed one position at a time by the library's routine): the gemv's packed
    activations are a half scale and 32 int8 values per block; the tile's are four scales and the four rows
    interleaved four bytes at a time (*source*, r5). A W-position kernel on the gemv layout therefore cannot reuse
    the tile's packing. Her choice (pack W times at m = 1) is the one that keeps the bytes.
  - Condition 5 (work size): confirmed in the source. `work_size` sizes `wdata` by `is_gemv`; W packs at m = 1
    need W times the gemv size, which is not what the function returns at ne1 = 2.
  - Condition 7 (same kernel variant on host and phone): the lead has accepted the `x4 kai` line in r35, which
    closes the "no report prints it" gap.
- **Two additions:**
  1. **Run part A on the lane's row range.** The shipped configuration calls the CPU kernel on rows [0, r) of a
     matrix whose other rows are on the GPU, with r a multiple of 128. Part A as written checks whole matrices.
     One more arm with a row range that ends before the last row, on real activations, costs nothing more.
  2. **The self-check of condition 8 should name the KleidiAI version and kernel in its failure line.** A
     version bump is the likeliest cause of a difference, and the side file of §1 has the same dependency, so
     both should print the same key.
- **Not a condition, but a consequence for prices:** part C's `-ub 2` is the first setting anywhere that runs
  width 2 on real data, and with the stock tile it is also the host cell for ggml's batch-2 cost (c 2.5 by my
  count). Tom can read both from the same run.
- Her test design (parts A to D) is adopted as written, with addition 1.

## 4. Next questions, ranked

1. **Does the model hold more than one KleidiAI buffer, and does the Q6_K head grow in it?** (§1.2 risk.) Host:
   one load log with the buffer sizes printed. Decides the side file's exact size and whether the head should
   stay out of it.
2. **Pack rate on an A78** (§1.4): the scalar pack routine, timed on the laptop's arm64 Linux container as a
   relative number, or one line in the next kit (pack 100 MB, print GB/s). Decides 5 s against 30 s.
3. **The file-backed arm of r5 §7 as a kit specification** with field names, so that Tom can build it when the
   lead asks: sizes, order, kill handling.
4. **`H-max` (Ebony):** the largest value into and out of each Q4_0 matmul. It also bounds the lane's own
   half-precision dequantise step, so it is worth minutes even with the prefill closed.
5. **Fixed share: which C to tune for** (r6 question 4), after R75.
6. **A second head for `C2`** (r6 question 2), only if R75 reads `dev_ratio(3)` ≤ 1.3.

## 5. Limits

- §1 is read from source; no patch exists and nothing was built. Line counts for the change are estimates.
- §1.4's times are estimates; the pack routine was read, not timed.
- §2.2 is a random-walk model. It reproduces a random-data simulation; real activations are neither independent
  nor zero-mean.
- §2.3 scales one perplexity anchor linearly.
- I did not re-run `ebony_h16acc.py`; I compared its printed results with the formula.

## 6. Sources (2026-11-23)

Repository: `lab/llama.cpp-c3r32/src/llama-model.cpp` (lines 1895 to 1945), `src/llama-model-loader.cpp` (lines
1695 to 1730), `ggml/src/ggml-cpu/kleidiai/kleidiai.cpp` (lines 518 to 570, 1494 to 1790);
`lab/llama.cpp-c1h6/build-kai/_deps/kleidiai-src/kai/ukernels/matmul/pack/kai_rhs_pack_nxk_qsi4c32pscalef16_qsu4c32s16s0.c`;
`lab/llama.cpp-c3/ggml/src/ggml-opencl/kernels/gemm_noshuffle_q4_0_f32.cl`; `campaigns/COMMON_RULES.md` rule 1;
`docs/runtime-knobs.md` (`GGML_REPACK_RELEASE`); `campaigns/03-dense15-5pct/STATUS.md` (owner questions (a) to
(d)); R69-result.md line 110, R72-result.md lines 194 to 195 and 238; `lab/c3/models/*-q4_0.gguf` (sizes);
[ebony-joins-r5.md](ebony-joins-r5.md) §1.4, §3 and `ebony_h16acc.py` (read, not re-run).
