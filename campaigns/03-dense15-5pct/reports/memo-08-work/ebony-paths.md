# memo-08 work file, Ebony: paths past the priced levers, kills, and bands for R72

2026-11-22, ebony (`claude-fable-5-2`). Draft for joint round 6 (memo-08). Not committed by me; Annie commits.
Nothing here is measured. Every price is *estimated* (arithmetic). Inputs are labelled where quoted: *device*,
*host* or *published*. Pricing script: my scratchpad `paths.py` / `paths2.py` / `l4.py` (Tom can re-run the
formulas below; they are short enough to copy into the memo).

Subjects: Yi-1.5-6B, Llama-2-7B, DeepSeek-LLM-7B (memo-06 §6, `e582606`). Goal: 12 tok/s printed on each.

## 0. Conventions (agreed in memo-07 §8, unchanged)

- steady = max(B ÷ r, W ÷ c) + O; printed = 0.95 × 1000 ÷ steady.
- c = 40.4 GB/s = 71.8 G weights/s on X1 + 2 A78 (R69 centre).
- 4th core: × 1.15 on c, with r4 = 27 GB/s (R69's (X1 + 3 A78) ÷ 3 A78 = 1.13 band on read).
- O = 8 ms.
- Sparse lever bytes: 2/18 (scales, always read) + 16/18 × kept.
- Frontier s: Yi 0.42 uniform; Llama-2 and DeepSeek 0.40 with k and v dense, which skips 33.4 % of weights
  (*host*, memo-07).
- K1 cost per touched weight: k 2.2 / 1.8 / 1.55 (device band; *host* 1.53 to 1.64).
- **Added here:** the KV read per token, F16 at about 190 positions: Yi 0.012 GB, Llama-2 0.100 GB, DeepSeek
  0.094 GB (*estimated*). For the MHA subjects that is 2.7 % and 2.5 % of the token's bytes. The ledger drafts I
  have seen leave it out (§5.1).
- The budget for 12 printed is 71.2 ms of steady time after O.

### What 12 needs, per subject (*estimated*)

| subject | stock B (GB, incl. head and KV) | dense r needed for 12 | W (G) | W ÷ c at 3 cores (ms) |
|---|---|---|---|---|
| Yi-1.5-6B | 3.34 | 47 GB/s | 5.80 | 81 |
| Llama-2-7B | 3.85 | 54 GB/s | 6.61 | 92 |
| DeepSeek-7B | 3.85 | 54 GB/s | 6.49 | 90 |

- Dense, 12 is out of reach on every subject: it needs about 2× R69's r centre, and the compute line alone
  (81 to 92 ms) is above 71.2 ms.
- Therefore **every path to 12 must cut both bytes and compute**. Bytes-only levers (tiers, KV) pay only once the
  compute line is already below the byte line. Compute-only levers (4th core, nibble-pair) pay only once bytes
  are below.
- This is why the master variable is the device r, and the second one is k.

## 1. Candidates the lead named: mechanism, evidence, fit, price, P

Δ is printed tok/s against "K1 at the frontier, 3 cores" (the base row of §3), per subject, at r 24 / r 35.
P is my probability that the lever delivers at least half its centre Δ on the device.

### 1.1 Exact self-speculation (layer-skip or sparse draft of the same model, exact verify): **kill on paper**

- **Mechanism.** The draft is the same model with layers skipped (SWIFT, arXiv 2410.06916v2, read in HTML) or
  with K1 at a high skip. The verify is a dense batch-n pass, so it is greedy-identical.
- **Evidence.**
  - Campaign 1 R20 (*device*): draft speculation at 0.48 to 0.76× plain. The SD888 is compute-bound at any batch
    above 1.
  - SWIFT reports 1.3 to 1.6× on A100-class GPUs, where verify is bandwidth-bound (*published*). That regime does
    not exist here once a sparse lever is on.
  - "Lossless but Not Free" (arXiv 2607.17283; search summary only, not read in full) points the same way for
    compute-bound hosts.
- **Price (*estimated*).**
  - Batch-n verify costs about n × W ÷ c on compute (κ2 ≈ 0.81, κ4 ≈ 0.71 per token only with a batch-shared
    unpack kernel that does not exist).
  - Against sparse plain decode, a free draft must reach acceptance a ≥ 0.38 at r 24 and ≥ 0.65 at r 35.
  - The draft is not free. Self-draft with layer skip plus sparsity prices 5.9 to 7.3 tok/s, against sparse
    plain 7.4 to 8.1 on the MHA subjects.
  - A sparse-target verify (verify with K1 on) loses through the union of the kept columns of n tokens: the
    bytes approach dense at n = 4.
- **Fit.** Fine (greedy-identical). It simply loses.
- **Δ:** −0.5 to −2 tok/s on every subject. **P(gain) 0.05.**
- **Kill line, proposed:** reopen only if R72 gives r < 20 *and* K1 fails (k > 2.2) on the device. This sharpens
  R69's rule B (kill if r ≥ 0.71 × c_b), because rule B was written before the sparse lever existed.

### 1.2 Contextual sparsity with exact fallback: **kill the exact form; price the inexact form as a raise of s**

- **Exact form:** skip, then check, then recompute dense where the check fails.
  - A per-token check that the skipped part did not matter needs the dense product, which is the cost.
  - Checking at the logits (does the greedy token change?) is speculation with a sparse draft (§1.1).
  - **Kill.**
- **Inexact form, "confidence-triggered dense fallback"** (new). Run sparse; when the top-1 logit margin is below
  a threshold, re-run the token dense.
  - This is a numerics rule. It must pass +5 % at `-b 512` and `-ub 1`. Note that at `-b 512` the "token" is a
    batch, so the rule must be defined per row of the batch: a dense re-run of the rows whose margin is low.
    PPL tools score the same rows.
  - Purpose: it lets s go up at the same PPL, because the tokens that carry most of the PPL damage are the
    low-margin ones.
  - **Evidence:** none *published* for this exact rule that I could find; related to the confidence-based early
    exit literature, which campaign 1 closed for layer drops (R21, *host*). R21 was about dropping layers, not
    about a fallback, so it does not close this.
  - **Price.** Raise s by +0.10 at a fallback rate f of 10 % of tokens (re-run dense costs f × dense token):
    - Without the fallback cost: Yi +1.3 to +1.6 / +1.6 to +2.1; Llama-2 +0.8 / +1.1 to +1.4; DeepSeek +0.7 /
      +1.1 to +1.3.
    - With f = 0.10 the dense re-runs take back about 60 % of that on the MHA subjects (a dense token is about
      2× a sparse one).
  - **Net Δ:** Yi +0.5 to +0.8; Llama-2 and DeepSeek +0.3 to +0.5. **P 0.25** (whether +0.10 s is reachable at
    f ≤ 0.10 is a host question).
  - **Host test H4** (§6).

### 1.3 Adreno 660 as a concurrent mat-vec lane: **the decisive unknown; keep, priced**

- **Mechanism.** The GPU takes a share g of the dense rows, here k and v plus the head, with its own Q4_0 kernel
  on the same mapped bytes, while the CPU runs K1 on the rest.
  - The two lanes share the LPDDR5 bus, so the sum is capped by a combined rate.
  - Sync is once per layer: one fence.
- **Evidence.**
  - Campaign 1 (*device*): llama.cpp's OpenCL backend ran the whole model at 0.7 to 2.9 tok/s, always below the
    CPU (docs/negative-results.md).
    - That is the stock backend with MoE ops falling back and a per-op sync. It is not the concurrent lane.
    - It must still lower the prior on any Adreno kernel quality.
  - R69 X2 measures the GPU read and a Q4_0 dot kernel. Pre-registered: read 30 (15 to 42) GB/s, q4 kernel
    10 (3 to 22) GB/s, S = CPU + GPU ÷ CPU 1.25 (0.95 to 1.60).
  - `cl_qcom_dot_product8` exists in the Adreno SDK (v1.3 docs; a device dump lists it at 1.1.1). Whether the
    660's driver exposes it is not documented; `clinfo` on the phone decides (search summary only). The
    standard `cl_khr_integer_dot_product` (OpenCL 3.0.8) is the fallback string to look for.
- *Correction 2026-11-22 (after Bonnie's §6.4): the table below lets the GPU take a share of the sparse bytes; my script did not cap it at the dense rows as the text says. Re-priced with the cap in §10.4. The rows below and the GPU rows of §3 are void.*
- **Price** (GPU on dense rows only, og = 4 ms of fence and launch cost per token, *estimated*):

| subject | base r24 / r35 (k 1.8) | + GPU centre (10 GB/s, S 1.25) | + GPU optimistic (20 GB/s, S 1.6) |
|---|---|---|---|
| Yi | 9.6 / 10.3 | 11.2 / 13.0 | 13.8 / 15.8 |
| Llama-2 | 7.7 / 9.0 | 9.1 / 11.2 | 11.3 / 13.6 |
| DeepSeek | 7.5 / 9.1 | 8.9 / 11.3 | 11.1 / 13.6 |

- **Δ at the centre:** Yi +1.6 to +2.7; Llama-2 +1.4 to +2.2; DeepSeek +1.4 to +2.2.
- **P 0.30:** R69's S ≥ 1.15 times a working kernel at ≥ 8 GB/s, discounted by campaign 1's OpenCL record.
- **Fit.** Bit-identical only if the GPU kernel's sums are exact. An int-dot Q4_0 × Q8_0 kernel can be made
  exact by accumulating per block in int32 and applying the scale in the same order as the CPU.
  - Otherwise it is a numerics change: F16 or F32 order differs. It is then checked like any other.
- **The one lever that moves the MHA subjects most.** No CPU-only path in my tables reaches 12 on Llama-2 or
  DeepSeek at r 24.

### 1.4 A55 cluster prefetch helpers: **kill**

- Campaign 1 R18 (*device*): the A55 expert-guess helper cost −13 %; 5 and 8 threads lost (an A55 in every sync).
- AnandTech's SD888 review (search summary): the A55 cluster's memory access is capped well below the big
  cores'. A prefetch helper must touch lines ahead of three big cores reading at about 24 GB/s.
- R69's A55 at 0.33 cycles per weight (band 0.22 to 0.48) makes it a poor compute lane too: about 4 A55 ≈ 0.4 of
  one A78.
- A55 prefetch into the shared L3 (4 MB) can stay at most a few hundred µs ahead of the big cores. It does not
  raise DRAM rate; it competes for it.
- **Δ ≤ 0, P(gain) 0.05.** R69's bus-limit rule (3 A78 + 4 A55 ÷ 3 A78 ≤ 1.10) closes it formally.

### 1.5 Big pages and TLB: **small; one probe cell, not a build**

- **Mechanism.** A 3.8 GB stream over 4 KB pages touches about 0.93 M pages per token. The A78's L2 TLB covers
  about 4 MB, so every page misses.
- If hardware prefetchers stop at 4 KB page boundaries, each page boundary also restarts the prefetch stream.
- **What is reachable.**
  - File-backed THP on arm64 Linux needs `CONFIG_READ_ONLY_THP_FOR_FS`. Upstream it applied to `VM_DENYWRITE` /
    exec mappings; the 2021 relaxation made it executable mappings of files not open for write (kernel lore,
    Fijalkovich, 2021-04).
  - A model `mmap` is not executable, and it is unknown whether the phone's vendor kernel even has the option.
    So a THP-mapped model file is not reachable from userspace.
  - Anonymous THP (`MADV_HUGEPAGE` on an anonymous copy) is reachable if `transparent_hugepage` is `madvise` or
    `always`. But it is an anonymous copy of the body, and campaign 1 lost its anonymous expert cache (R23) and
    R69's C is the open question. Not for 3.8 GB files at C ≈ 3.1 GB.
  - Contiguous-PTE (64 KB) mappings are a newer kernel feature (6.9+). The phone's kernel is far older.
- **Price.**
  - A page walk that hits cache (sequential PTEs, 8 per line) costs about 10 to 30 ns, mostly overlapped.
    Exposed cost is about 0 to 5 % of r.
  - **Δ about +0 to +0.4 on all subjects, P 0.15.**
- **Test.** One R72 rider cell: cold q4x4 read over a 1 GB anonymous arena with `MADV_HUGEPAGE`, against the
  same arena with `MADV_NOHUGEPAGE`. Read `AnonHugePages` from `/proc/self/smaps` to confirm the hint took. If
  the gain is < 3 %, close.

### 1.6 Cache-aware tiling: **kill for decode**

- Decode mat-vec has no weight reuse at batch 1. The activation vector (4096 or 11008 × Q8 = 4.6 to 12 KB) is
  already L1-resident.
- Tiling pays only for batch > 1 (prefill or verify), and verify is dead (§1.1).
- The one cache effect left is L2-slice placement of the per-thread row ranges for the hot weights, which is
  already in R72 block L3. **Δ 0, P —.**

### 1.7 LUT kernels (T-MAC, NEON `TBL`) against `sdot`: **kill at 4 bits; park as the kernel of a tier**

- **Mechanism.** Weights are split into bit-planes. A table of partial sums of each group of g = 4 activations
  (16 entries, int8) is built once per input vector, and each plane is one `TBL` lookup per 4 weights. Cost is
  linear in bits.
- **Evidence (*published*, T-MAC, EuroSys 2025, arXiv 2407.00088v2, read in HTML).**
  - At 4 bits, end-to-end is 1.2× on OnePlus 12 (8.24 → 10.19 tok/s) and 1.9× on Jetson Orin NX (A78AE). Kernel
    gain is at most 3.1× single-thread at 4 bits, against 11.2× at 1 bit.
  - The LUT is int8-quantized: kernel NMSE is 3.35e-3 against llama.cpp's 3.33e-3; Llama-2-7B-4bit PPL is
    equal at 5.96. Not bit-identical. With fast aggregation (+FA) PPL is 6.38, which is over our budget.
  - The llama.cpp baseline was the version of 2024. It is not our repacked Q4_0x4 + 0001 kernel, which is
    already issue-bound at about 20 V-ops per 128 weights.
- **My count (*estimated*).** At 4 planes: per 128 weights about 2 `TBL` + 2 split ops + 2 widening adds per
  plane, which is about 24 ops, plus scale handling. That is no better than 20. At 2 planes it is about 12 ops.
- **Fit with K1: poor.** The LUT works on groups of 4 columns, so a skip must drop whole groups. memo-07 C6
  (units of eight, amended u ≈ 0.13) showed group skip costs s.
- **At 4 bits: Δ ≈ 0, P 0.1.** Its only role is as the kernel for a 2-bit tier in §2.1, if that tier passes
  quality.

### 1.8 Activation-driven column skip fused into the K1 kernel: **already K1**

- K1 *is* the fused column skip. The open item inside it is the cost per touched weight, k, which is the gather
  and unpack overhead. That is §2.2 (nibble-pair layout).

### 1.9 KV and head tricks

- **KV cache Q8_0** (`-ctk q8_0 -ctv q8_0`; runtime option; numerics change).
  - It halves the KV read: −0.05 GB per token on the MHA subjects.
  - **Δ +0.0 to +0.1** (it is on the byte line only, and the base is often compute-bound). **P 0.5** that it
    passes; it is cheap to carry in the same numerics budget.
  - Worth including only as the last 1 % once a path sits on the byte line.
- **DeepSeek head (102,400 × 4096, Q6_K, 0.344 GB, about 9 % of its token bytes and about 18 ms cold at
  0.8 × r 24).**
  - A two-stage head (cheap screen, exact re-score of the top candidates) was closed in campaign 1 at Q2_K and
    Q3_K because it was compute-bound (R29).
  - On a dense 7B with K1 the token is compute-bound too.
  - A Q4_0 side copy for the screen adds 0.24 GB of resident set when C is the open question.
  - **Δ +0.2 to +0.4 on DeepSeek only, P 0.15, park.**
  - Yi's head (64,000 rows, 0.215 GB) is a smaller version of the same.
  - Head K1 (skip columns of the head's input) is a numerics change on the logits directly. It belongs to the
    same budget; I price it only if R72's H block shows the head is over 10 % of the token.
- **Head tricks for MHA** (k and v are kept dense by the rule): see §2.4. The rule's 16.6 % dense k, v share is
  where the GPU lane goes first.

## 2. New levers (not priced before)

### 2.1 Three-tier bit-plane input sparsity (skip / high-2-bit plane only / full)

*Correction 2026-11-22: the compute cost below is wrong; the plane merge raises k by about × 1.22 to 1.25 (§10.2).*

- **Mechanism.**
  - Re-lay each sparse tensor's nibbles as two planes, high 2 bits and low 2 bits, column-major as K1 reads.
  - Per input column, three choices: skip (|x| small); read only the high plane, an approximate weight
    (4 · q_hi + 1.5 − 8) · d (|x| middle); read both planes (|x| large).
  - Bytes for the middle tier are halved. No extra bytes are stored: it is a re-layout in a side table that
    replaces the repacked layout.
- **Evidence.**
  - *Published*, layout only: Any-Precision LLM (arXiv 2402.10517) reads partial bit-planes of overlaid
    quantized weights with a CUDA kernel that matches standard quantized GEMV throughput. That shows the layout
    is servable. It does not show that the per-column tier is accurate.
  - Quality: none. It is a numerics change.
- **Fit.** The served file is unchanged; the side table is a derived layout; +5 % at `-b 512` and `-ub 1` applies
  to the combined rule (skip + tier).
- **Price** (tier share h = 0.30 of columns at half bytes; compute counted as full, conservative):

| subject | base, r 24 (k 2.2 / 1.8 / 1.55) | + tiers, r 24 | r 35 |
|---|---|---|---|
| Yi | 8.6 / 9.6 / 9.6 | 8.6 / 10.3 / 11.7 | no change (compute-bound) |
| Llama-2 | 7.7 / 7.7 / 7.7 | 7.7 / 8.9 / 8.9 | no change |
| DeepSeek | 7.5 / 7.5 / 7.5 | 7.9 / 8.6 / 8.6 | no change |

- It pays only at low r (byte-bound). **Δ +0 to +2.1 at r 24 (Yi), +0 to +1.2 (MHA); 0 at r 35. P 0.20** (quality unknown; a host screen decides).

### 2.2 Nibble-pair-rows layout for K1 (possibly bit-identical)

*Correction 2026-11-22: withdrawn, see §10.1 (κ ≈ 1.0 on the A78's binding pipe).*

- **Mechanism.**
  - Pack the high and low nibbles of one byte from two *rows* of the same column, instead of two columns of the
    same row.
  - One `AND` and one shift then feed two `SDOT`s with the same activation lanes. That removes the per-block
    unpack shuffle that dominates K1's gather cost.
  - The −8 offset is applied as an integer fix-up (Σx per column block × 8), which is exact.
- **Price.**
  - κ ≈ 0.85 on k (*estimated* from the 20-op count: unpack 8 → about 5).
  - It moves the compute line only: at r 24 it is hidden behind bytes (Δ 0), except Yi at k 2.2 (+1.0). At r 35: Yi +1.4 to
    +1.8, Llama-2 +0.9 to +1.1, DeepSeek +0.5 to +1.2.
- **P 0.35.**
- **Fit.** Bit-identical if the integer sums match the stock kernel's order of float scaling (int32 per block,
  scale once). Tom can check it with `memcmp` like K1's identity lines.

### 2.3 Low-rank compensation for skipped columns (R-Sparse-style)

- **Mechanism.** A rank-r side table approximates W · x_skipped, so s can go up at the same PPL.
- **Price.** Rank 128 costs about +4.3 % compute and +8 % bytes at Q8 (*estimated*). It pays only if it buys
  Δs ≥ 0.08. Same Δ as §1.2's raise of s, minus its cost: about +0.3 to +1.0.
- **P 0.15.** The low-rank screen of campaign 1's head failed (R43); this is a different target (activations
  of skipped columns), but a warning.

### 2.4 DDR devfreq and BWMON readout in the kit (diagnostic that can turn into a runtime lever)

- **Mechanism.** On Snapdragon, DDR frequency is voted by the CPU's bandwidth monitor (BWMON / memlat governors)
  and by a static CPU-frequency → DDR-frequency table (Qualcomm DVFS docs; search summary only).
  - If the DDR clock sits below its top step during decode, r is capped by the governor, not by the bus.
- **Fix in the runtime, not in the phone:** a different access pattern can raise the vote (for example keeping
  the big-core clocks at the steps that map to the top DDR step). The kit cannot write sysfs without root, so the
  readout comes first.
- **Price:** if the DDR clock sits one step low, r could rise 10 to 25 %. **P(it sits low) 0.25.**
- **Test:** a kit rider that logs `/sys/class/devfreq/*/cur_freq` (whichever nodes are readable without root)
  before, during and after each R72 T block.

### 2.5 KV, and the 4th core's real share

Not new levers, but corrections to the ledger. See §5.

## 3. Stacked paths per subject (*estimated*, printed tok/s, k 2.2 / 1.8 / 1.55)

r 24 (r4 27), and r 35 (r4 35). "4c" = 4th core × 1.15 on c with r4. "GPUc" = GPU centre, "GPUo" = optimistic.
"nib" = §2.2 (κ 0.85). "s+0.10" = §1.2 / §2.3.

| path | Yi r24 | Yi r35 | Llama-2 r24 | Llama-2 r35 | DeepSeek r24 | DeepSeek r35 |
|---|---|---|---|---|---|---|
| base (K1, 3 cores) | 8.6/9.6/9.6 | 8.6/10.3/11.7 | 7.7/7.7/7.7 | 7.7/9.0/10.0 | 7.5/7.5/7.5 | 7.9/9.1/10.2 |
| + GPUc | 11.2/11.2/11.2 | 11.6/13.0/14.2 | 9.1/9.1/9.1 | 10.1/11.2/12.1 | 8.9/8.9/8.9 | 10.2/11.3/12.2 |
| + tiers + GPUo | 15.9/16.4/16.4 | 15.9/17.2/18.2 | 12.9/12.9/12.9 | 13.4/14.4/15.2 | 12.5/12.5/12.5 | 13.4/14.4/15.2 |
| nib + s+0.10 + tiers (CPU only) | 11.7/13.6/13.6 | 11.7/13.9/15.7 | 10.0/10.0/10.0 | 10.0/11.5/12.8 | 9.5/9.5/9.5 | 10.1/11.6/12.8 |
| nib + s+0.10 + tiers + 4c | 13.3/15.1/15.1 | 13.3/15.7/17.7 | 11.1/11.1/11.1 | 11.3/13.1/14.0 | 10.6/10.6/10.6 | 11.5/13.2/13.4 |
| … + GPUc | 16.8/16.9/16.9 | 16.8/18.7/20.2 | 12.8/12.8/12.8 | 14.0/15.4/15.9 | 12.3/12.3/12.3 | 14.0/15.2/15.2 |

Reading:
- **Yi** reaches 12 on a CPU-only path (nib + s + tiers) at k ≤ 1.8, at either r. It is the subject most likely
  to pass. Required: K1 at k ≤ 1.8 on the device and one of the s-raising levers.
- **Llama-2 and DeepSeek** reach 12 at r 24 only with the GPU lane (centre or better) on top of every CPU lever.
  At r 35 they reach 12 to 14 on CPU only, with the 4th core and k ≤ 1.8.
  - So for the MHA subjects, **12 needs either r ≥ about 30 GB/s on the device or a working GPU lane**.
  - At r 24 Llama-2 needs s ≥ 0.755 on CPU only, which is far outside its frontier (0.40).
- Joint probability that all three pass, *estimated*: about 0.10 to 0.15. The goal is not out of reach on paper,
  but it hangs on two device unknowns: r (R69) and the GPU lane (R69 X2, then a kernel).

*Correction 2026-11-22: the rows with GPU, nib and tiers above are superseded by §10; the joint P is in §10.3.*

## 4. Kills and parks, one line each (for the memo's ledger)

| item | verdict | why (numbers above) |
|---|---|---|
| exact self-speculation (layer-skip / sparse draft, batch 2/4 verify) | kill | §1.1: −0.5 to −2 tok/s; reopen only if r < 20 and k > 2.2 |
| exact contextual-sparsity fallback | kill | §1.2: the check costs the dense product |
| A55 prefetch / compute helpers | kill | §1.4; R18 *device*; bus cap |
| cache tiling for decode | kill | §1.6: no weight reuse at batch 1 |
| T-MAC / `TBL` LUT at 4 bits | kill | §1.7: ≈ 24 ops per 128 against 20; not bit-identical; groups of 4 hurt s |
| THP / TLB | one probe cell | §1.5: Δ ≤ +0.4; file THP not reachable |
| DeepSeek two-stage head | park | §1.9: R29 compute-bound; resident cost |
| KV Q8_0 | carry as the last 1 % | §1.9 |
| GPU concurrent lane | keep, decisive for MHA | §1.3 |
| bit-plane tiers | host screen | §2.1 |
| nibble-pair K1 | host build, identity check | §2.2 |
| confidence fallback, low-rank compensation | host screen | §1.2, §2.3 |
| DDR devfreq readout | kit rider | §2.4 |

## 5. Challenge to the gap ledger

Bonnie's `bonnie-path.md` was not in `memo-08-work/` when I wrote this (only `clyde-r72-plan.md`). I write the
challenge against the ledger as Clyde's draft states it (§1, L1 to L7). Bonnie: please answer each item in your
file, or tell me where your ledger already covers it.

1. **KV bytes are missing from B.** Llama-2 and DeepSeek are MHA (32 / 30 layers × 4096 for k and v). At about 190
   positions in F16 that is 0.100 / 0.094 GB per token, about 2.7 % / 2.5 % of the token's bytes. It grows with
   the prompt: at 512 positions it is about 7 %. The ledger must carry it per subject and state the position count.
2. **The 4th core is priced at × 1.15 with no prior from campaign 1.** R18 (*device*) found that 4 big-core threads
   lost, "Android's own threads take the 4th core", even at equal clocks. The ledger should carry
   P(4th core helps) ≈ 0.35 and not a point factor. R72's T block decides it. My band: c4 ÷ c3 1.10 (0.95 to 1.22),
   r4 ÷ r3 1.06 (0.95 to 1.20).
3. **Residency comes first.** The served Q4_0 files are 3.48 / 3.83 / 4.00 GB. R69's C centre is 3.1 GB, with
   P(3.4 held) 0.35. If C < 3.8, the two MHA subjects are not resident and every price in this file and in the
   ledger is void for them: the token then pages from flash.
   - The ledger must carry a residency row per subject before any speed row.
   - The gap for MHA may then be a byte-size gap, not a speed gap (only a lower-bit served file, H2, closes it).
4. **The head's compute must sit on the compute line, at its type's cost.** Q6_K is not Q4_0: about 1.3 to 1.5×
   per weight in the stock path (my *estimate*; R72 block G measures it). DeepSeek's head is 0.42 G weights.
   At 1.4× that is 0.59 G-equivalent, about 8 ms of compute at 3 cores. A ledger that counts it at Q4_0 cost is
   2 to 3 ms optimistic on DeepSeek.
5. **O = 8 ms is one number for three graphs.** A dense 7B decode at `-ub 1` has about 225 mat-vecs plus about 100
   other nodes. Campaign 1 R33 (*device*) found that the dense wait is barrier count and arrival skew.
   - My band for L6 (empty barriers only): 3 (1.5 to 6) ms per token. Skew adds to it under K1, because
     per-thread work differs with the input.
   - The ledger should carry O per subject: Llama-2 32 layers, DeepSeek 30, Yi 32. The GQA subject has smaller
     k, v nodes; the layer count is the same.
6. **GPU assumptions.** If the ledger uses R69's S as a multiplier on r, it overstates the lane: the GPU's share
   must run a *dense* kernel, so its bytes are full Q4_0 bytes, not sparse bytes.
   - Campaign 1's 0.7 to 2.9 tok/s OpenCL record (*device*) is not a kill of the concurrent lane, but it is the
     prior for kernel quality: my P 0.30.
7. **One check for the ledger's arithmetic.** At r 24 the K1 rule is byte-bound on every subject for
   k ≤ 1.8 (§3 base row is flat in k). So L1 (k) matters only above r ≈ 28 for Yi and ≈ 30 for MHA. The ledger
   should say which row binds per subject; otherwise R72's L1 will be read as a speed result when it is not.

## 6. Bands for R72 (my half; Bonnie gives hers, the lead merges)

*Estimated*, centre (80 % band).

| row | quantity | Ebony |
|---|---|---|
| L1 | K1 k, cold, one A78, keep 0.58 to 0.60, inputs 4096 / 11008 | 1.9 (1.5 to 2.6); X1 1.7 (1.4 to 2.3); 11008 is 0.1 above 4096 |
| L2 | r4 cold through ggml threads, X1 + 3 A78 | 26 (20 to 33) GB/s; r4 ÷ r3 1.06 (0.95 to 1.20) |
| L3 | c4 ÷ c3, all threads hot at once | 1.10 (0.95 to 1.22) |
| L4 | t_rule ÷ t_stock, cold, best set | Yi 0.70 (0.60 to 0.95); Llama-2 0.75 (0.65 to 0.95); DeepSeek 0.75 (0.65 to 0.95) |
| L5 | Q6_K ÷ Q4_0x4 per weight, hot, 1 thread | 1.4 (1.15 to 1.8) |
| L6 | empty barrier sequence per token, 3 / 4 threads | 3 (1.5 to 6) / 3.5 (1.8 to 7) ms |
| L7 | head cold, 3 threads: Yi / Llama-2 / DeepSeek | 11 (8 to 16) / 5.6 (4 to 8) / 18 (13 to 26) ms |

L4 comes from the token model: for example at r 24, k 1.8: Yi 129.8 → 81.3 ms (0.63); Llama-2 151.8 → 106.8 ms
(0.70); DeepSeek 142.3 → 100.1 ms (0.70). At r 30 and k 2.2 the ratio rises to 0.94 to 0.95, hence the wide upper
band.

## 7. Test list (host, Tom) and try list (device): my draft; joint once Bonnie's file is in

### Test list (host)

| id | item | pass rule (written before results) |
|---|---|---|
| H1 | nibble-pair K1 kernel (§2.2), NEON on M4, `memcmp` identity against stock 4x4 with dropped x zeroed at 4096 and 11008 | identity all 1; host k ≤ 0.90 × current K1 at keep 0.60 |
| H2 | bit-plane tiers (§2.1), PPL screen, 40 chunks, `-b 512` and `-ub 1`, on Yi and Llama-2 (two models, one GQA and one MHA) | at the same PPL as the K1 frontier, bytes ≤ 0.92 × K1's bytes on both models |
| H3 | confidence-triggered dense fallback (§1.2), PPL screen at f = 5 / 10 / 20 % | s + 0.10 at f ≤ 0.10 inside +5 % at both batch settings on two models |
| H4 | low-rank compensation, rank 64 / 128 on the skipped part of gate/up/down (§2.3) | Δs ≥ 0.08 at the same PPL on two models; else close |
| H5 | DeepSeek head: share of token time from a `-ub 1` CPU profile | report only; reopens §1.9 if > 10 % |
| H6 | Adreno Q4_0 × Q8_0 int-dot kernel design and exactness argument (OpenCL C, written, compiled for the phone; not run on host) | ready for the first GPU-lane kit only if R69 rule D keeps rank 1 open |

### Try list (device, riders on R72 or the next kit)

| id | item | reading |
|---|---|---|
| D1 | `clinfo`-style dump of the Adreno 660 extension string (`cl_qcom_dot_product8`, `cl_khr_integer_dot_product`) and the driver version | decides H6's kernel |
| D2 | THP cell: 1 GB anonymous arena, `MADV_HUGEPAGE` against `MADV_NOHUGEPAGE`, cold q4x4 read, `AnonHugePages` logged | close if < 3 % |
| D3 | devfreq readout before / during / after T blocks (whichever DDR / bw nodes are readable) | if DDR sits below top step in ≥ 20 % of T samples, open §2.4 |
| D4 | GPU lane prototype (only after R69 D keeps rank 1 and D1 gives an int-dot path): GPU on k, v and head rows, CPU K1 on the rest, one layer, ABBA | S_layer ≥ 1.15 against the CPU-only layer of block T |

## 8. Sources read this round

- SWIFT, arXiv 2410.06916v2 (HTML, read).
- T-MAC, arXiv 2407.00088v2, EuroSys 2025 (HTML, read; numbers in §1.7 are from its §5.2, Tables 3, 4 and 7).
- Any-Precision LLM, arXiv 2402.10517 (search summary of the HTML pages; layout claim only).
- "Lossless but Not Free", arXiv 2607.17283 (search summary only).
- AnandTech, Snapdragon 888 review, A55 memory cap (search summary only).
- Qualcomm DVFS / BWMON documentation (search summary only).
- Adreno OpenCL SDK release notes and a device extension dump listing `cl_qcom_dot_product8` 1.1.1; Khronos
  `cl_khr_integer_dot_product` (OpenCL 3.0.8) (search summaries only).
- Linux kernel lore, "mm, thp: relax the VM_DENYWRITE constraint on file-backed THPs" (Fijalkovich, 2021-04;
  search summary).
- Repository: docs/negative-results.md (R18, R20, R23, R29, R33, R43, OpenCL), knowledge.md, R69 pre-registration
  §3 and §4, memo-07 files, clyde-r72-plan.md.

## 9. Note from Bonnie (2026-11-22)

`bonnie-path.md` is now in this folder. It answers your §5 items one by one (its §7) and challenges this file
(its §6). The points, briefly:
1. Nibble-pair K1 (§2.2). In the A78 SWOG, V0 binds: SMLAL, plus SCVTF and FCVTL at half rate. The unpack sits on
   the other pipe, so κ is about 1.0. Proposed instead: K1m, an exact magic-number convert.
2. Tiers (§2.1). The plane merge costs k × 1.22 to 1.25 by my count, so tiers lose at r ≥ 30.
3. Joint P. With residency as a gate I get 0.005, against your 0.10 to 0.15.
4. GPU on Yi. Its dense rows are only 13 % of its bytes.
5. Your 4th-core band and your P(helps) 0.35 disagree with each other: the band gives 0.68.
Please answer in this file. The joint lists are proposed in my §8.

## 10. Answer to Bonnie's §6 (2026-11-22, ebony)

1. **Nibble-pair K1: conceded, withdrawn.** My κ 0.85 came from the op mix of the stock 4x4 `sdot` gemv. K1 is an
   element-granular `SMLAL` kernel that binds on V0 (SMLAL, SCVTF, FCVTL), with the unpack on V1, which has room. I
   also missed that TR-128 already packs two rows of one column per byte. κ ≈ 1.0 on the A78 and under 5 % on the
   X1. Her K1m (exact magic-number convert, the only change on V0) replaces my H1.
2. **Tiers merge cost: conceded.** Avoiding the merge with two SMLAL streams (4 · Σ x · hi + Σ x · lo) doubles the
   V0 work on full columns, which is worse. So k rises by × 1.22 to 1.25, and tiers are a low-r lever only. Her
   kernel line (tier k ÷ K1 k ≤ 1.15 on the count, else close) goes into the pass rule.
3. **Joint P: conceded in part.** My 0.10 to 0.15 was conditional on residency, and I did not say so. Unconditional:
   the MHA files need C ≥ about 3.9 GB, P(4.2 held) 0.12, and conditional on that about 0.10 to 0.15. That gives
   **about 0.01 (0.005 to 0.02)**. Bonnie's 0.005 is inside my band, so we agree on the order of magnitude: the
   goal on all three subjects is improbable on this phone unless R69 surprises on C and r.
4. **GPU lane on Yi: conceded, and worse than she says for my file.** With the GPU capped at the dense rows (k, v at
   full bytes plus the head), the dense share of the token's bytes is 0.10 (Yi), 0.26 (Llama-2) and 0.32 (DeepSeek).
   Re-priced, printed tok/s at k 2.2 / 1.8 / 1.55 (*estimated*; GPU centre 10 GB/s and optimistic 20 GB/s, og 4 ms,
   3 cores, no tiers):

| subject | base r 24 | + GPU, r 24 (centre / opt.) | base r 35 | + GPU, r 35 (centre / opt.) |
|---|---|---|---|---|
| Yi | 8.6 / 9.6 / 9.6 | 8.6 / 10.1 / 10.1 (both) | 8.6 / 10.3 / 11.7 | no gain |
| Llama-2 | 7.7 / 7.7 / 7.7 | 8.5 / 9.1 / 9.1 and 8.5 / 9.7 / 9.7 | 7.7 / 9.0 / 10.0 | 8.5 / 10.2 / 11.5 (both about equal) |
| DeepSeek | 7.5 / 7.5 / 7.5 | 8.9 / 8.9 / 8.9 and 9.0 / 10.3 / 10.3 | 7.9 / 9.1 / 10.2 | 9.0 / 10.4 / 11.6 and 9.0 / 10.8 / 12.2 |

   - The dense-row lane is worth +0 to +0.5 on Yi, +0.8 to +2.0 on Llama-2 and +1.4 to +2.8 on DeepSeek. It is
     no longer "decisive" alone.
   - It becomes decisive only with a GPU kernel that reads K1's sparse layout (the GPU takes sparse rows too). That
     kernel is not priced, and given campaign 1's OpenCL record I put P ≤ 0.15 on it. I withdraw §3's reading
     that MHA reaches 12 at r 24 with the GPU lane.
5. **4th-core band against P: conceded, I move both.** R18's loss was on Granite. But a dense 7B decode also has
   about 330 nodes, so the barrier cost is of the same kind. New band: c4 ÷ c3 **1.05 (0.90 to 1.20)**, which gives
   P(> 1.05) ≈ 0.5. My P(helps) becomes 0.5 to match.
6. **Head bit-plane screen (her §6.5) and anonymous body under repack (her §6.6): agreed.** The THP cell is the
   relevant arm; my file-THP paragraph does not matter for the repacked body.

**Joint lists: I confirm Bonnie's §8 with three amendments.**
- Host T1 to T7 confirmed as written. T6 carries my H2 (tiers, with her kernel line) and H3 (confidence fallback).
- **Add T8:** the Adreno Q4_0 × Q8_0 int-dot kernel design with an exactness argument (my H6), written only after
  R69's X2 keeps rank 1 open. It is priced on dense rows only (§10.4). A sparse-row GPU kernel is a design note
  only, not a test.
- **Dropped:** my H1 (replaced by T1, K1m); my H4 low-rank compensation (park, P 0.15, R43 warning); my H5 (covered
  by T4 and R72 L7).
- Device D1 to D5 confirmed as written. D5's GPU prototype runs dense rows (k, v, head) only, and is read against
  §10.4's prices.
