# c3 memo-11: a breakthrough on bytes and compute per token? Low-bit layouts, LUT kernels, skipping, low rank

Date: 2026-11-22. Author: Bonnie (`claude-fable-5-2`). **Ebony has not reviewed this** (§9 lists what I ask her to
attack). Not committed. Nothing here is measured by me. Every price is *estimated* (arithmetic); quoted inputs carry
their own label (*device* R72, *host* memo-02 / memo-06 / tom-head.md, *published*). Script with a self-check:
[memo-11-work/bonnie_bytes.py](memo-11-work/bonnie_bytes.py) (reproduces R72 §5's 6.90 / 6.21 / 6.22 and memo-10's
width-2 row 7.94 / 7.79 / 7.38).

Subjects: Yi-1.5-6B, Llama-2-7B, DeepSeek-LLM-7B, served Q4_0. Goal 12 tok/s printed = **79.2 ms per token**.
"Breakthrough" as the lead defined it: a lever or stack whose **centre** price reaches 12 on all three and whose
key uncertainty can be settled on the host.

## 1. Verdict

- **No breakthrough.** No lever in this angle, and no stack built on one, reaches 12 on all three at its centre.
- **Fewer bytes alone cannot reach 12 on any subject, whatever the bit width.** At the stock kernel's cost per
  weight (k = 1.0) the compute time alone is over the body budget, even with zero bytes read (§3). R72 said the
  body is bus-bound with 1.3 to 1.7 × of compute slack; a byte cut of 2 × uses all of that slack and more. **A byte
  lever needs a kernel at about half the `sdot` cost per weight.** Without one it stops at 8.4 / 7.6 / 7.5 tok/s
  at 3.44 bpw and 8.8 / 8.0 / 7.9 at 2.625 bpw, and at 10.2 / 9.4 / 9.1 with no bytes at all.
- **What 12 needs from this angle** (§3): body at **2.3 to 2.7 bits per weight** (51 to 59 % of the stock bytes)
  **and** a kernel at k ≤ 0.5, with the head screen C7 and O at 5.2 ms on top.
- **Inside the +5 % budget, a re-encode of the frozen Q4_0 weights stops at about 3.3 to 3.9 bpw** (73 to 86 % of
  the bytes; *published* Llama-2-7B ladder, §4). Best case 8.7 / 8.0 / 7.8 tok/s with a table-lookup (LUT) kernel
  that does not exist here; **slower than stock** (5.7 / 5.1 / 5.1) with ggml's Q3_K kernel as measured in R72.
- **Layer skip, FFN skip, dynamic depth and low rank are closed on numbers**: each buys at most 3 to 5 % of the
  bytes inside the budget (§6).
- **The one route whose arithmetic comes near 12 is not a runtime lever: a 2-bit-class served file** (inside the
  F16 × 1.20 ceiling, which CAMPAIGN.md's terms allow) **with a LUT kernel, C7, O 5.2 and width-2 lookahead**:
  11.6 / 11.3 / 10.9 at the centre, 12.7 / 12.3 / 11.8 only if the body fits 2.625 bpw. It changes the served
  file, so it is the owner's question, and R72's type filter said "Q4_0 stays the working type at the freeze". I
  price it; I do not propose it as decided.
- P(all three at 12): **about 0.005 through this angle** (my judgement; the lookahead memos had 0.002). P(Yi alone
  at 12): about 0.04. Neither figure counts whether the owner would accept a lower-bit served file.

## 2. Inputs used

| input | value | label |
|---|---|---|
| stock body, ggml, 4 threads on 4-7 | 120.6 / 140.6 / 130.8 ms | *device*, R72 §5 |
| body bytes per token | 3.11 / 3.64 / 3.42 GB | *device*, R72 §5 |
| head Q6_K; O | 9.04 / 4.40 / 13.88 ms; 8 ms | *device*, R72 L7; allowance |
| read rate R; hot compute C | 29.9 GB/s; 40 GB/s centre (35 at the L4 caps, 50 at full clocks) | *device*; memo-10 §2 |
| per-type cost, 4-7 hot, against Q4_0 (4096 / 11008 wide) | Q3_K 1.731; IQ3_XXS 1.862; IQ3_S 2.934; IQ4_XS 1.459 / 1.238; Q2_K 1.170 / 0.883 | *device*, R72 L5 |
| Q4_0 ÷ F16 perplexity | × 1.0236 / 1.0302 / 1.0367 | *host*, memo-06 (40 chunks) |
| column sparsity inside +4.5 % | 40 % / 20 % / 20 % | *host*, memo-06 |
| tokens per pass at width 2, trained head | 1.326 / 1.448 / 1.385 (DeepSeek: mean, no head) | *host*, tom-head.md |

Model of a token (memo-10's pass model, with bytes and kernel cost made explicit):

    body(b, k, W) = s × [max(x, y) + 0.25 × min(x, y)]
    x = b × B ÷ R                       b = bytes as a share of the stock Q4_0 body
    y = k × (1 + e × (W − 1)) × B ÷ C   k = compute per weight against the stock Q4_0 sdot kernel
    ms per committed token = (body + O + guess + xpos × (W − 1)) ÷ tau + head × hs;  printed = 950 ÷ ms

s is fitted per subject so that b = k = 1 gives the measured body (0.96 to 0.98).

## 3. The requirement: bytes and kernel cost together

Largest byte share at which one position per pass meets 79.2 ms (bits per weight in brackets; stock is 4.5):

| kernel cost k | head and O | Yi | Llama-2 | DeepSeek |
|---|---|---|---|---|
| 1.0 (stock sdot) | stock, 8 ms | none | none | none |
| 1.0 | C7, 5.2 ms | none (0.26 at C 50) | none | none |
| 0.75 | C7, 5.2 ms | 0.49 (2.2) | 0.18 | 0.19 |
| 0.50 | stock, 8 ms | 0.52 (2.3) | 0.47 (2.1) | 0.43 (1.9) |
| **0.50** | **C7, 5.2 ms** | **0.59 (2.7)** | **0.51 (2.3)** | **0.52 (2.3)** |
| 0.35 | C7, 5.2 ms | 0.62 (2.8) | 0.54 (2.4) | 0.54 (2.4) |

- "none" means the compute time alone exceeds the budget. This is the row that matters: **a Q4_0-cost kernel
  cannot reach 12 with any number of bytes**, at any clock state priced (C 35 to 50).
- R72 §5's "at most 60 / 55 / 50 % of the stock bytes" was the bus term alone. With compute counted the lines are
  the bold row, and they hold only for a kernel twice as cheap as `sdot`.

## 4. Lever A: re-encode the frozen Q4_0 weights at load, inside the +5 % budget

**How many bits fit the budget.** The only ladder I found for one of our subjects (*published*, a third party's
blog, Llama-2-7b-chat, llama.cpp `perplexity` on wikitext-2 **train**, imatrix on the I-types only; relative values
are what I use):

| type | file ÷ Q4_0 file | PPL against F16 | PPL against Q4_0 |
|---|---|---|---|
| Q4_0 | 1.000 | +1.8 % | 0 |
| Q3_K_M (3- and 4-bit mixed) | 0.860 | +2.6 % | **+0.8 %** |
| IQ3_S (3.44 bpw) | 0.770 | +4.0 % | +2.2 % |
| Q3_K_S (3.44 bpw, no imatrix) | 0.770 | +7.2 % | **+5.3 %** |
| IQ3_XS (3.3 bpw) | 0.728 | +5.2 % | +3.3 % |
| IQ3_XXS (3.06 bpw) | 0.675 | +9.5 % | +7.6 % |
| Q2_K (2-bit mixed, no imatrix) | 0.661 | +15.5 % | +13.4 % |
| IQ2_M (2.7 bpw) | 0.616 | +14.8 % | +12.8 % |
| Q2_K_S (no imatrix) | 0.605 | +22.5 % | +20.3 % |
| IQ2_S (2.5 bpw) | 0.574 | +22.1 % | +19.9 % |

- These are quantised from F16. A re-encode from the Q4_0 file starts from weights that already carry Q4_0's
  error, so its cost against Q4_0 is at least the last column, not less. (Campaign 2 R43: re-quantising a Q4
  tensor to Q4 changed 10 to 14 % of its bytes for +0.15 %, *host*.)
- **So the budget holds about 3.3 to 3.9 bpw: 73 to 86 % of the bytes.** A uniform 3-bit body (3.44 bpw) sits on
  the +5 % line and leaves nothing for C7 or any sparsity rule. The codebook types that fit better (IQ3) are the
  ones R72 closed on kernel cost.
- The best published training-free codes do not move this far enough: QTIP at 3.0 bpw without fine-tuning is +5.1 %
  against FP16 on Llama-2-7B and +33 % at 2 bits (*published*, context 4096). They need 6,144 × 2,048 tokens of
  Hessians and a trellis decode, which is neither phone-computable nor `sdot`-shaped. Even at k 0.5 a 3.0 bpw body
  prices 10.1 / 9.3 / 9.0.
- A per-tensor bit table from calibration (3 bits where a layer tolerates it) is the Q3_K_M row: about 14 % fewer
  bytes for about +1 %. It is the only form that leaves budget for anything else.

**Prices** (ms per token against stock; bytes per token Yi / Llama-2 / DeepSeek):

| form | bytes (GB) | kernel | ms | printed tok/s | numerics |
|---|---|---|---|---|---|
| all 3-bit, 3.44 bpw | 2.38 / 2.78 / 2.61 (−0.73 / −0.86 / −0.81) | ggml Q3_K, k 1.73 (*device*) | **+30.2 / +35.2 / +32.7** | 5.66 / 5.05 / 5.12 | whole budget |
| same | same | `sdot` at Q4_0 cost, k 1.0 (no such 3-bit kernel) | −23.9 / −28.0 / −26.0 | 8.36 / 7.60 / 7.50 | whole budget |
| same | same | LUT, k 0.75 (0.60 to 1.0) | **−28.7 / −33.5 / −31.2** | 8.72 / 7.95 / 7.82 | whole budget |
| mixed 3/4-bit, 3.9 bpw | 2.67 / 3.13 / 2.94 (−0.44 / −0.51 / −0.48) | LUT, k 0.85 | −17.0 / −19.9 / −18.5 | 7.88 / 7.14 / 7.08 | about +1 to +2 % |

P(all 3-bit inside the budget on all three at both settings) 0.3; P(a 3-bit kernel at k ≤ 0.8 on the A78) 0.45;
together about 0.15. **Not a breakthrough: its centre is 8.7 / 8.0 / 7.8.**

## 5. The kernel question: does a LUT/TBL kernel beat `sdot` on the A78?

**Mechanism** (T-MAC, EuroSys 2025): group four input columns; per token build a 16-entry table of the sums of
every subset of the four activations; store each weight bit-plane as 4-bit indices; one NEON `TBL` then serves 16
rows × 4 columns × 1 bit. Cost grows with the bit width, and the weights must be a uniform grid (Q4_0, Q3_K, Q2_K
are; the IQ codebook types are not).

**Published** (T-MAC paper and README; baseline llama.cpp b2794, May 2024, before the 4x4 repack kernels):

| measurement | value |
|---|---|
| single-thread kernel, best speed-up over llama.cpp | 3.1 × (4-bit), 4.7 × (3-bit), 5.8 × (2-bit), 11.2 × (1-bit) |
| Raspberry Pi 5 (4 × Cortex-A76), single thread, Llama-2-7B end to end | 2.8 × (4-bit), 6.7 × (2-bit) |
| OnePlus 12 (Snapdragon 8 Gen 3), Llama-2-7B | 4-bit 10.19 against llama.cpp 8.24 tok/s; 2-bit 16.62 against 6.95 |
| M2 Ultra, single thread, Llama-2-7B 4-bit | 7.34 against 5.65 tok/s; PPL 5.96 against 5.96 (tables in int8, kernel NMSE 3.35e-3 against 3.33e-3) |
| "fast aggregation" option | PPL 5.96 → 6.38 (+7 %): outside our budget, not used |
| multi-thread | "primarily constrained by memory bandwidth" |

**My price of k** (*estimated*):
- Count on two vector pipes: per `TBL` one lookup, two widening adds and a share of the index unpack, about 3.2
  micro-ops per 64 weight-bits. That is 0.8 micro-ops per 16 weights per bit, against about 3.8 for the stock
  kernel (0.118 cycles per weight, *device*, R72 L3). Plus a per-row scale multiply per 16-column block.
- **k ≈ 0.85 at 4 bits, 0.75 at 3 bits (0.60 to 1.0), 0.50 at 2 bits (0.35 to 0.75).**
- Cross-check from the published A76 ratios: if today's repacked `sdot` kernel is about 2.4 × the May-2024
  `vec_dot` (my assumption, not measured), 2.8 × and 6.7 × become k ≈ 0.85 and 0.4.
- So a LUT kernel does **not** beat `sdot` at 4 bits (memo-05 closed "T-MAC tables on Q4_0"; that stands), is a
  modest gain at 3 bits, and is about twice as cheap at 2 bits. It is a declared numerics change (int8 tables), or
  exact with 16-bit tables at about 1.6 × the lookups.
- R72's lesson applies: the A78 was not issue-bound on K1, so a count is a prior, not a price. No A78 number
  exists for any LUT kernel; the published ARM numbers are A76, A78AE at 12 threads, and newer cores.
- ggml's own Q2_K kernel already runs at Q4_0's cost when hot (1.17 at 4096, 0.88 at 11008, *device*). BitNet.cpp's
  TL1/TL2 kernels are ternary only. An extra lookahead position costs a LUT kernel its lookups again (e about 0.85
  against 0.45 for `sdot`), so a cheap kernel does not make a wide verify pass cheap.

## 6. Skipping and low rank: closed on numbers

| lever | evidence | bytes inside the budget | price (Yi / Llama-2 / DeepSeek) | state |
|---|---|---|---|---|
| Static layer removal | SLEB: Llama-2-7B at 10 % of blocks, PPL 5.47 → 6.95, **+27 %** (*published*). Campaign 1 R21: cheapest layer +2.5 % (*host*, Granite) | at most one block of 32 (3.1 %), and it takes the whole budget | −3.7 / −4.4 / −4.1 ms | closed |
| Sublayer ablation | memo-02 H7 (*host*, Mistral, Qwen2.5): an FFN is never free; one attention sublayer free on Mistral, none on Qwen2.5; pairs +11.7 % | one attention sublayer: 0.7 to 1 % | about −1 ms | closed; not re-screened on the three subjects |
| Per-token FFN skip (FFN-SkipLLM, router-free dynamic depth) | memo-02 X4 (*host*): the **oracle** holds 4 to 6 % of (token, layer) pairs inside +4.5 %, a causal rule 3 to 4 %. FFN-SkipLLM's 25 to 30 % is on task scores, not perplexity (*published*) | at most 3 % | −3.6 / −4.2 / −3.9 ms | closed |
| Position-aware depth decay (D3, 2025) | "about 1.5 × fewer operations" on generation tasks (*published*, search summary); no PPL at fixed positions | unknown; contradicts X4 and H7 under our gate | not priced | not proposed |
| LayerSkip-style early exit | needs exit heads trained with the model | – | – | outside the rules (not stock weights) |
| Low rank (SVD) of the weights | SVD-LLM V2 at 20 % fewer parameters: LLaMA-7B 5.68 → 7.12 (**+25 %**), Llama-3-8B 6.14 → 8.01 (+30 %) (*published*) | under 5 % at +5 % (my reading of the slope) | under −6 ms, with a second dense kernel | closed |
| Low-bit plus low-rank residual | the factors are dense 8-bit reads: rank 256 on a 4096 × 4096 matrix adds 2.1 MB to a 9.4 MB tensor (+22 %) | negative at 3 bits and above | – | closed on arithmetic |
| Lossless coding of the Q4_0 nibbles | entropy under 4 bits, but no decoder near 20 G weights/s per core | 10 to 15 % at best (not counted on the files) | – | closed on decode cost |

CLADA: I found no paper under that name (one search); not priced.

## 7. The served-file route (the owner's question, priced so that it is on the table)

CAMPAIGN.md's terms: "served quantisation: free, chosen once per model and frozen", bounded by F16 × 1.20; the
+5 % budget is then relative to that file. On the ladder of §4 the lowest types inside the ceiling on Llama-2-7B are
Q2_K (+15.5 %, no imatrix) and IQ2_M (+14.8 %); memo-01 found a floor of about 3.05 bpw on Llama-3-8B, so this is
model-dependent. Yi and DeepSeek are not known.

| stack | Yi | Llama-2 | DeepSeek |
|---|---|---|---|
| Q2_K-mix file (b 0.66), ggml's Q2_K kernel (k 1.11, *device* weighted) | 118.1 → 8.0 | 130.2 → 7.3 | 131.5 → 7.2 |
| Q2_K-mix file, LUT k 0.55 | 94.5 (−43.1) → 10.1 | 102.8 (−50.2) → 9.2 | 105.9 (−46.8) → 9.0 |
| + C7 + O 5.2 | 87.2 → 10.9 | 97.8 → 9.7 | 96.2 → 9.9 |
| **+ width 2, trained head** (e 0.85) | 81.7 → **11.6** | 84.4 → **11.3** | 87.3 → **10.9** |
| the row above at k 0.75 / at C 35 | 9.2 / 10.8 | 8.9 / 10.4 | 8.7 / 10.1 |
| body at 2.625 bpw throughout (b 0.583), LUT k 0.50, + C7 + O 5.2 + width 2 | 75.0 → 12.7 | 77.1 → 12.3 | 80.3 → 11.8 |

- Bytes per token: 2.05 / 2.40 / 2.26 GB at b 0.66; 1.81 / 2.12 / 1.99 GB at 2.625 bpw.
- The last row is outside the ceiling on Llama-2 as published (Q2_K_S +22.5 % without an imatrix); with an imatrix
  it is unknown.
- Side effects: a 2.4 GB body fits the anonymous capacity R72 found held (anon34), which Llama-2 and DeepSeek at
  Q4_0 do not. The upstream GPU kernels are Q4_0 only, so lever G does not stack. The guess heads must be retrained
  on the new file's greedy output. The +5 % budget would be a budget on a model already 15 % worse than F16.
- **By the lead's definition this is not a breakthrough either**: the centre misses on all three at the file that
  is known to fit the ceiling.
- P, each my judgement: a LUT-decodable body at b ≤ 0.62 inside the ceiling on all three 0.2; k ≤ 0.55 sustained on
  the A78 0.4; C7 and width 2 deliver their centres 0.25; all three at 12 given all of that 0.3. Joint about 0.006.

## 8. Lists for Tom (through the lead)

### 8.1 Test list (host only; each closes a question)

| # | test | cost | closes when |
|---|---|---|---|
| U1 | **Re-encode screen.** Requantise the frozen Q4_0 file of Yi and Llama-2 to Q3_K per tensor class (`gate` + `up`, `down`, `q` + `k`, `v` + `o`, all) with stock `llama-quantize --allow-requantize` (and `--tensor-type` if the base has it; I recall it does not at `620d2225c`, Tom to check); 40 chunks at `-b 512` and `-ub 1`; report body bytes saved per class | about 10 quantise runs and 20 PPL runs per model | the classes that fit +4.5 % together save under 20 % of the body bytes on either model → lever A closed |
| U2 | **LUT kernel, host ratio.** A `TBL` gemv for a uniform 2-bit and 3-bit layout (16 rows × 4 columns, int8 tables, int16 accumulation, per-block scale), against the repacked Q4_0 4x4 gemv: ns per weight hot (1 MB) and cold (576 MB), 1 and 4 threads; the micro-op count with `tools/c3_a78_pipes.py`; NMSE against the float result | one scratch kernel, a day | host k(2-bit) > 0.8 or k(3-bit) > 1.0 → LUT kernels closed, and with them §4 and §7. The host is an M4 with four vector pipes, so a pass is a prior for the A78, not a price |
| U3 | **Ceiling ladder, only if the lead takes §7 to the owner.** From each subject's F16 (the S3 freeze makes it anyway): Q3_K_S, Q2_K, Q2_K with `--pure`, each with and without an imatrix from wikitext-2 train; body bpw from the tensor table; PPL at `-b 512`, 40 chunks first | 6 files and 6 PPL runs per subject | no uniform type with a body at or under 3.0 bpw inside F16 × 1.20 on all three → §7 closed |

Order: U2 first (it gates both A and §7 and needs no model). U1 beside it. U3 waits for the lead.

### 8.2 Try list (build only after its test; env-gated, off by default, model-generic, two subjects)

| # | lever | after | label | price at the centre (Yi / Llama-2 / DeepSeek) | P |
|---|---|---|---|---|---|
| Z1 | Load-time 3-bit re-encode of the tolerant classes, as a phone-built, sha-keyed side file, with the LUT kernel | U1, U2 | numerics, one budget, both batch settings | −17 to −29 / −20 to −34 / −19 to −31 ms → 7.9 to 8.7 / 7.1 to 8.0 / 7.1 to 7.8 | 0.15 |
| Z2 | LUT kernel for a 2-bit-class served file | U2, U3, and the owner's decision on the served type | numerics (int8 tables) against ggml's kernel for that type | −43.1 / −50.2 / −46.8 ms → 10.1 / 9.2 / 9.0; stack of §7 11.6 / 11.3 / 10.9 | 0.08 for the stack's centre; 0.006 for 12 on all three |

No device test is asked for. By the owner's 2026-11-22 rule, none of this is "certain on host evidence": the
kernel's A78 cost can only be read on the phone, and U2 gives a prior.

## 9. For Ebony's review

1. §3's "none" row: is the pass model (max + 0.25 × min) right when bytes halve and the pass turns compute-bound?
   The 0.25 coupling was fitted on two points at stock bytes.
2. k for a LUT kernel on the A78: the count, and my assumed 2.4 × between today's repacked kernel and b2794's.
3. A published or measured uniform 3-bit kernel on a two-pipe ARM core at or under the `sdot` cost per weight.
4. Any 2025 to 2026 training-free method holding 2.3 to 2.7 bpw inside +5 % of a 4-bit model on a 7B (I found
   none; QTIP's 2-bit row is +33 %).
5. Whether a dynamic-precision rule (a coarse bit-plane for small activations, full bits for large ones) can beat
   column skipping. My paper count gives 73 % of the bytes with a column kernel, so I did not price it.

## 10. Sources (read 2026-11-22)

- T-MAC repository (fetched): <https://github.com/microsoft/T-MAC>
- T-MAC paper, arXiv 2407.00088v2 (fetched; Tables 4 to 7, Figures 6 to 8 as text): <https://arxiv.org/html/2407.00088v2>
- llama.cpp quantisation ladder for Llama-2-7b-chat (fetched): <https://beebopkim.github.io/2024/03/09/Benchmarks-for-lots-of-quantization-types-in-llama-cpp>
- QTIP, arXiv 2406.11235 (fetched, first 100,000 characters; Tables 3 to 5): <https://arxiv.org/html/2406.11235>
- SLEB, arXiv 2402.09025 (search summary, Table 7): <https://arxiv.org/pdf/2402.09025>
- SVD-LLM V2, arXiv 2503.12340v1 (fetched, Table 2): <https://arxiv.org/html/2503.12340v1>
- FFN-SkipLLM, arXiv 2404.03865; D3, arXiv 2503.08524 (search summaries): <https://arxiv.org/abs/2404.03865>, <https://arxiv.org/abs/2503.08524>
- Phone CPU numbers for a dense 7B:
  - T-MAC Table 7 above (OnePlus 12, Snapdragon 8 Gen 3): 10.19 tok/s at 4 bits.
  - OnePlus 13, llama.cpp, Llama-2-7b-chat Q4_0, 6 threads: 12.5 to 13.2 tok/s (fetched; a third party's model card, SoC not named): <https://huggingface.co/imi2/llama-2-7b-chat-pure-Q4_0-gguf>
  - Snapdragon 8 Gen 3 stage-level study, arXiv 2605.27435 (fetched; no decode tok/s in the text): <https://arxiv.org/html/2605.27435>
  - Qualcomm QMX post, 2026-04 (page returned its title only): <https://www.qualcomm.com/developer/blog/2026/04/llama-models-acceleration-on-cpu-qmx>
  - **Nothing found at over 10 tok/s for a dense 7B on a Snapdragon 888-class CPU.** Both numbers above are on chips two and four generations newer.
- Repository: R72-result.md §4, §5, §11; memo-02 §5; memo-06 §4; memo-09; memo-10; campaign 2 knowledge.md.

Not done: no host run, no device run, no review by Ebony. The Llama-2 ladder is a third party's, on the train
split and not our gate. Web pages were read through a summarising fetch. k for every LUT row is a count. Yi and
DeepSeek have no low-bit perplexity of any kind. The QTIP "no fine-tuning" QuIP# column may include fine-tuning
(the table header is ambiguous). CLADA and LUT-GEMM were not found or not read.
