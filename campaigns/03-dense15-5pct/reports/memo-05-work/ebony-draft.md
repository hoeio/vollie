# c3 memo-05 work file, Ebony: multi-stream GEMV, a frontier scan for fewer bytes, and the 15 tok/s corner

Agent model: `claude-fable-5-2` (Ebony). Date 2026-11-19. Repository HEAD when written: `458102c`. Joint round 3
with Bonnie; verdicts and pre-registrations are the lead's. **R69 has not been run. Nothing here was measured.**

Labels: *device* = a stored phone result, named; *host* = memo-02 or a read of the lab tree; *published* = someone
else's number on other hardware; *recalled* = from memory, not confirmed today; *estimated* = my arithmetic. Web
pages were read on 2026-11-19; §7 says for each whether I read the text, a fetch summary, a search summary or an
abstract only. Pricing is Bonnie's lane: my prices are first guesses for her to replace.

## 0. Summary

1. **Multi-stream GEMV: I lower my own memo-03 price.** P(the R69 X1 gain is at least 1.15) is **0.10**, not 0.3.
   Band for the X1 block: **2 streams 1.02 (0.94 to 1.12), 4 streams 1.00 (0.88 to 1.12)** (*estimated*). Bonnie's
   pre-registered band is 1.05 with P 0.22 / 0.27; both positions stand (§1.4). Reason: published single-core copy
   rates of A78-class cores are 10.6 to 14.2 GB/s with a write stream beside the read stream, well above the
   6.3 to 8.1 GB/s each decode thread reads here, and every multi-core number I found is sublinear. The evidence
   points to a limit shared by the cluster, which more streams per thread cannot lift.
2. **No published miss-buffer or prefetch-stream count exists for the A78 or X1** in what I could read. Arm's
   A78 optimization guide (read as text) has no prefetcher section, lists `LDNP` with the same timing as `LDP`,
   and recommends `DC ZVA` only for zeroing. `LDNP` and `DC ZVA` are priced at zero for a read-only GEMV.
3. **One hardware fact from that guide changes the column-kernel price** (§5, review of Bonnie): on the A78,
   `SMLAL` runs only on pipe V0 (1 per cycle), `SDOT` on both (2 per cycle). An exact axpy column kernel has a
   floor of 0.125 cycles per touched weight from `SMLAL` alone, against 0.108 for the whole stock kernel
   (*device*, R41). I put Bonnie's K1 at **k = 2.1 (1.5 to 2.6)**, not 1.8.
4. **Frontier: one new candidate, the rest closed or already priced.** New: *proxy-ranked FFN row skip*
   (Prox, arXiv 2607.27591, 2026-07): rank FFN neurons with a cheap proxy, then compute only the kept **rows** of
   `gate` and `up`. Rows are contiguous in the GGUF, so this needs no column layout and runs on Bonnie's Y1b
   in-place kernel. Unpriced in PPL on our models; three host screens decide it (E1 to E3). T-MAC-style tables do
   not beat `sdot` exactly, KV bytes are 0.3 to 0.6 % of a token, early exit is covered by X4.
5. **15 tok/s: no corner on both models.** On Mistral alone a paper corner exists (15.0 tok/s) that needs five
   independent best cases at once; I put their joint chance under 10⁻⁴ (§4). The best new candidate prices 9 to
   12 tok/s at its optimistic end. memo-01 §1 stands.

## 1. Multi-stream GEMV (memo-03 rank 2)

### 1.1 What is published about the cores (and what is not)

| fact | value | label, how read |
|---|---|---|
| A78 vector pipes | two: V0, V1 | *published*, A78 guide Issue 4.0, text extracted |
| A78 load pipes | three (Load/Store 0/1 and Load 2) | same |
| `SDOT` | latency 2 (1 for the accumulate), 2 per cycle, either pipe | same |
| `SMLAL(2)`, `MLA` | latency 4 (1), **1 per cycle, V0 only** | same |
| `SHL`, `USHR` by immediate | latency 2, **1 per cycle, V1 only** | same |
| `AND`/`EOR`, `TBL` (1 or 2 tables), `FMLA` | 2 per cycle, either pipe | same |
| `LDNP` | listed in the same rows as `LDP` (latency 5, throughput 3/2 for Q pairs); no separate behaviour described | same |
| `DC ZVA` | recommended for `memset` to zero only (§4.4 of the guide) | same |
| hardware prefetcher: streams, distance, 4 KB page crossing | **not in the guide; not found** in two searches. TF-A sources show a `PF_MODE` field in `CPUECTLR_EL1` (erratum 2132060 sets the most conservative mode), nothing on counts | search summary |
| L1/L2 miss buffers (MSHR, fill buffers) of A78 / X1 | **not found** | – |
| DSU master port issuing capability (the cluster's cap on reads in flight) | DSU TRM found, the fetched PDF was truncated; **not read** | – |
| DRAM latency of this SoC | about 100 ns full random | *recalled* (AnandTech's S21 Ultra review; the page now redirects) |

### 1.2 Bandwidth numbers on A78-class cores (*published*; none is a Snapdragon 888)

| system | single big core, tinymembench memcpy / memset (sbc-bench table; unit not printed, MB/s *recalled*) | all cores |
|---|---|---|
| Genio 1200, 4 × A78 at 2.2 GHz | 14,200 / 19,000 | – |
| Jetson AGX Orin, A78AE at 2.2 GHz | 10,600 / 30,350 | STREAM about 54.5 GB/s of a nominal 204.8 (12 cores; forum post, search summary); triad 49.2 GB/s (8 cores) and 71.7 GB/s (12 cores) (OpenBenchmarking, search summary) |
| QRB5165 (Snapdragon 865), A77 at 2.84 GHz | 14,470 / 23,910 | – |
| SC8280XP, X1 at 3.0 GHz | 17,500 to 21,010 / 41,780 | – |
| Neoverse N1 / V2 (Arm learning path) | 20.9 / 37.0 GB/s DRAM, combined load and store; Arm's stated reason for V2: "more outstanding memory requests" | – |

Searched and not found: any STREAM, lmbench or tinymembench result for the Snapdragon 888 (Bonnie found the same on
2026-11-19). One systems result worth knowing for memo-03's rank 1: Sereno (OSDI 2026, text extracted) reports
that phone SoCs give the NPU's memory traffic priority over the CPU's; arbitration between engines on the bus is
not neutral. It measures a Snapdragon 8 Elite and gives no GB/s.

### 1.3 Reading for this phone (*estimated*)

- Each decode thread reads 6.3 to 8.1 GB/s (19 to 24 GB/s on three threads, *device*, campaign 1). The same kernel
  cache-hot runs at 0.5625 B ÷ 0.0512 ns = 11.0 GB/s per A78 at 2,112 MHz (*device*, R41). Cold is 58 to 73 % of hot.
- One A78-class core copies 10.6 to 14.2 GB/s on other SoCs: a read stream at that rate with a write stream beside
  it. At about 100 ns that is 17 to 22 reads in flight on **one** stream. Our stream needs 10 to 17. So the
  per-stream reach of the prefetcher is probably not what binds at our rate.
- Multi-core numbers are sublinear wherever I found them: Orin's 12 cores reach 4.5 to 6 GB/s each against 10.6
  alone; here a fourth thread gave × 1.15, not × 1.33 or more (*device*, R13/R14). A limit shared by the cluster
  (DSU port, L3, memory controller) fits both. More streams per thread do nothing against a shared limit.
- What could still make a gain: page-boundary restarts of the prefetcher (4.0 GB is 1.0 M pages per token), which
  a software prefetch across the boundary would hide. Whether the A78 prefetcher crosses 4 KB pages is not
  published in what I read. Campaign 1's page-latency model was withdrawn for expert slots (R41); nothing
  measured it on a dense stream.
- **R69 already separates the cases; no new cell is needed.** With h = hot A78 rate (`p2hot`), g1 and g3 = cold
  `q4x4` on 1 and 3 A78, a1 and a3 = plain read:

| R69 reading | meaning | multi-stream / prefetch |
|---|---|---|
| g1 ÷ h ≥ 0.90 and g3 ÷ (3 g1) ≥ 0.90 | no memory stall; the token is compute-bound | closed; only more cores help |
| g1 ÷ h ≥ 0.90 and g3 ÷ (3 g1) < 0.85 | shared limit | closed with ranks 1 and 4 unless X2's S says the GPU has its own share |
| g1 ÷ h < 0.80 and a1 ≥ 1.3 × g1 | one stream can go faster than the kernel pulls: a stall inside the kernel's access pattern | prefetch distance is the lever to test (E5) |
| g1 ÷ h < 0.80 and a1 ≈ g1 | one stream is the limit | the X1 gain decides, as pre-registered |

### 1.4 Prices (*estimated*)

| lever | mechanism in the `q4_0_4x4` gemv | numerics | gain on the gemv | P(gain ≥ 1.15) |
|---|---|---|---|---|
| 2 or 4 interleaved row-group streams per thread | two or four `b_ptr` advance together; the x blocks are loaded once per step (saves about 1 of 21 vector µops per extra stream) | bit-identical: each row's integer sums and float order are unchanged | 1.02 (0.94 to 1.12) / 1.00 (0.88 to 1.12) memory side, + 0.02 to 0.04 compute | 0.10 |
| `PRFM PLDL2KEEP` at +1, +4 or +8 KB | one prefetch per 72-byte block on a load pipe; needs ≥ 17 lines of lead at 11 GB/s and 100 ns; +4 KB and more also crosses the page | bit-identical | 1.00 (0.97 to 1.10) | 0.05 |
| `LDNP` | no documented difference from `LDP` on the A78 | bit-identical | 1.00 | not proposed |
| `DC ZVA` | saves the fetch of a line that is about to be overwritten; the gemv writes 16 B per 9 to 32 KB read | – | 1.00 | not proposed |

- Usable gain is min(gain, hot ÷ cold) = at most 1.37 to 1.72 with today's numbers; the X1 block's read loop is an
  upper bound for the kernel (Bonnie, bands §4; agreed).
- Token: conditional on a pass, 5.2 to 6.5 tok/s on a 4.0 GB body; expected value about +2 % (*estimated*).
- **Disagreement, both kept.** Bonnie (pre-registered): 1.05 at both stream counts, P(≥ 1.15) 0.22 / 0.27. Mine:
  1.02 / 1.00, P 0.10 / 0.08. The pre-registration is not touched; R69 scores hers, and mine is scored beside it.
- R23's negative (Q3_K `prfm pldl1keep` lost, *device*) was an L1 prefetch inside 660-byte expert runs. This is an
  L2 prefetch kilobytes ahead in a multi-megabyte stream. It is a different mechanism, and I still price it near 0.

### 1.5 The build, if Rule D asks for it

One patch in the c3 series on top of the r28 runtime (`patches/llama.cpp-c3/0002`), about 80 lines in
`ggml/src/ggml-cpu/arch/arm/repack.cpp`: `VOLLIE_GEMV_STREAMS=<1|2|4>` and `VOLLIE_GEMV_PF=<bytes>`, both off when
unset, read once. Only `ggml_gemv_q4_0_4x4_q8_0` changes (the Q6_K head gemv can follow). Identity per chunk at
`-b 512`, `-ub 32`, `-ub 1`; only `-ub 1` exercises the gemv. Model-generic: every Q4_0 mat-vec, the campaign 2
models included, so it is measured on Mistral and Qwen2.5 in one session with an in-run ABBA.

## 2. Frontier scan: fewer bytes per token, file unchanged

Not repeated here: TEAL, DIP, LaRoSA, SpQt, CATS, CMoE, R-Sparse, WINA, N:M, PowerInfer-2, ActiveFlow, EntroLLM
(memo-01, memo-03, memo-04-work).

| work | date | idea | for us | state |
|---|---|---|---|---|
| **Prox** (arXiv 2607.27591) | 2026-07 | training-free. Stage 1: input sparsity plus **quantized proxy weights** give the magnitude ranking of the SwiGLU intermediate channels. Stage 2: only the selected channels are computed exactly, in all three projections. Claims up to 1.99 × decode at 70 % FFN sparsity on ten models (GPU kernels). **Summary page only; no PPL table seen** | **new candidate, §3** | E1 to E3 |
| NorSA (NeurIPS 2026 poster) | 2026 | dynamic thresholds plus a decorrelating rotation; Llama-3-8B at 50 %: +0.44 PPL, 1.32 × | a rotation is merged into weights or costs d × d products per layer, as LaRoSA (memo-01): a second quantisation of the file | not proposed; abstract only |
| SpAx (arXiv 2610.02598) | 2026-10 | three tiers per input: full weights, a compressed approximation, skip; 2.06 × mean at 4 bits on CPU-offloaded weights; PPL "at most +10 %" | twice our budget, and needs a compressed second copy; the regime is a transfer-bound GPU | not proposed; abstract only |
| T-MAC (arXiv 2407.00088v2) | 2024, EuroSys 2025 | bit-plane tables through `TBL`; 4-bit Llama-2-7B on a OnePlus 12: 10.19 against llama.cpp's 8.24 tok/s; M2 Ultra single thread 7.34 against 5.65 | see below | closed |
| MNN-LLM (arXiv 2506.10443) | 2025-06 | embedding kept on flash (Qwen2-7B: 2.18 GB less DRAM), per-core load balancing; 2.3 × llama.cpp decode on a Xiaomi 14 (search summary) | the embedding trick is what mmap already does for `token_embd` here; the speed is a different file format | nothing to carry |
| llama.cpp | 2026 | one search: no pull request for gemv prefetch, multi-stream or activation sparsity; recent repack work is Q6_K dotprod (#19360, merged) and RVV (#20723) | – | nothing to carry |

**T-MAC on Q4_0, priced (*estimated*).** It is a compute idea and saves no bytes (4 bit-planes are 4 bits).
- Its baseline was llama.cpp b2794 (2024-05), before the 4x4 repack kernels (#5780, 2024-07-10). The phone ratio
  of 1.24 × is against a kernel this runtime no longer uses.
- Its tables are per group of four inputs, built from the activations, and quantised to int8, which is lossy
  (the paper reports NMSE 3.35e-3 against 3.33e-3 and equal PPL). An exact version needs 16-bit entries: two
  `TBL` and two `zip` per plane, about 24 vector µops per 64 weights against about 10 for `sdot`.
- A group of four fixed inputs cannot drop single columns: at 40 % sparsity a group is skippable 2.6 % of the
  time. So it is not a column kernel either. **Closed for Q4_0.**

**Small terms, checked (*estimated* from *host* header shapes):**
- **KV.** Mistral: 32 layers × (k + v) × 8 KV heads × 128 = 65,536 values = 131 kB per position at F16; 24.5 MB
  read per token at 187 positions, **0.6 %** of 4.037 GB. Qwen2.5: 57 kB per position, 10.7 MB, **0.26 %**.
  Negligible at the metric's context; no lever.
- **Embedding.** One row per token, 2 to 3 kB. Nothing.
- **Head.** Mistral 0.110 GB (2.7 %), Qwen2.5 0.447 GB (10.9 %), both Q6_K (*host*). Rows are skippable in the
  stock layout, so campaign 2's two-stage head carries to Qwen2.5 (Bonnie: about 0.3 GB) and does almost nothing
  for Mistral. It does not generalise in size. A Q4_0 head is a served-quantisation choice (H2).
- **Cache-line-granular column skipping in the stock layout.** memo-02 §8 measured it: 64 % of blocks kept still
  touch 81 % of the lines, and block sparsity is closed. Bonnie's b_eff question (her §2.2) is the open form.
- **Per-token early exit.** A subset of X4's skip sets (all layers above L for a token). X4's oracle held only
  4 to 6 % of (token, layer) pairs inside the margin (*host*). Published training-free exits need trained heads
  or adapters (LayerSkip, Kangaroo; *recalled*). Not reopened.

## 3. The new candidate: proxy-ranked FFN row skip

**Idea.** `down`'s input is p_i = SiLU(g_i) × u_i, one value per FFN neuron. H3 found it the tolerant group: 50 %
of it zeroed costs +1.21 % (Mistral) and +0.70 % (Qwen2.5) (*host*, 16 chunks). If the neurons with small |p_i|
are known **before** `gate` and `up` are computed, their rows in both matrices are never read. A row is contiguous
in the GGUF (2,304 B on Mistral), so the saving needs no column layout, only a row-major kernel on the file
mapping: Bonnie's Y1b, with four arbitrary row pointers instead of four adjacent rows.

Why it was not priced before: memo-01 closed output-side methods on CATS and Deja Vu numbers (+15 % and more).
Those rank by the gate alone or by a trained predictor. H3's `down`-only cells are the same skip set ranked by the
exact product, and they are cheap. The open question is how much of that survives a cheap ranking.

| variant | ranking signal | side data | bytes saved per token at neuron sparsity s (FFN = 78 % of the token; each matrix 1.057 GB on Mistral) |
|---|---|---|---|
| A, gate rule | \|SiLU(g_i)\| after the full `gate` | none | s × `up` rows: 0.53 GB (13 %) at s = 0.5 |
| B, proxy | a 1- or 2-bit copy of `gate` and `up` (sign and a per-row scale), read every token | 0.47 GB at 1 bit: a weight-derived artifact with its own sha and check | s × (`gate` + `up`) − proxy: 0.59 GB (15 %) at s = 0.5, 1.01 GB (25 %) at s = 0.7 |
| B + `down` columns | as B, and `down` through a column kernel | as B, plus the column layout | 1.67 GB (41 %) at s = 0.7 |

Price (*estimated*, max model, O = 8 ms, c_b 40.4, Y1b at k = 1.5, column kernel at k = 2.1):
- Variant B at s = 0.7, r = 24: 3.03 GB → 126 + 8 ms, **7.5 tok/s**; with `down` columns 2.37 GB → 107 ms,
  **9.3 tok/s**; at r = 35, 12.3. Variant A at s = 0.5: 3.5 GB → 6.5 tok/s.
- The proxy pass is priced at `sdot` cost per byte. A 1-bit proxy holds 4.5 times the weights per byte, and no
  kernel for it exists; if it costs more, variant B turns compute-bound sooner. E3 does not measure this.
- It shares the one +5 % budget with input-side column sparsity; the two are not additive in PPL.
- Residency: `gate` and `up` are read from the file mapping, so 2.1 GB of the repack copy goes away with Y1b
  whether or not any row is skipped. The proxy adds 0.47 GB.
- Both batch settings: the rule is per token and per neuron, so it should behave as H3 did (within 0.5 point).
- Unknown and decisive: the PPL of the product rule above 50 % (E1), of the gate rule (E2), and how much a 1- or
  2-bit proxy loses (E3). Prox's own numbers were not readable.
- For the lead: is a lossy low-bit copy used only for ranking a "weight-derived artifact" under the 2026-11-19
  decision? It never enters the arithmetic of a kept row. I read it as covered; it is the lead's reading to make.

## 4. Is there a corner where 15 tok/s falls? (*estimated*)

t = max(bytes ÷ r, sdot-equivalent bytes ÷ c) + 8 ms. Mistral, 4.037 GB.

| corner | assumptions, all at once | tok/s |
|---|---|---|
| today's centre | r = 24, c_b = 40.4 | 5.7 |
| every core, no sparsity | r = 45 (the two-engine ceiling of memo-03), X1 + 3 A78 scaling perfectly (c = 50.3 from R41's cycles at R67's clocks, with the pre-registration's 0.87) | 10.2 |
| all eight cores, no sparsity | c about 61 with four A55s at the pre-registered 0.33 cycles per weight | bytes need 60.6 GB/s, above the 51.2 GB/s bus: **impossible** |
| the paper corner | r = 45, four big cores, 40 % column sparsity with b_eff = b = 0.644 on the blocks (2.64 GB), an exact column kernel at k ≤ 1.2, C ≥ 4.6 GB | **15.0** (Mistral); Qwen2.5 at its 35 % (2.98 GB): 13.5 |
| proxy row skip, optimistic | §3 variant B with `down` columns, s = 0.7 inside the budget, r = 35 | 12.3 |

- The paper corner needs: a read rate no CPU-only number supports (19 to 24 measured, 39 *published* for a GPU on
  the same bus); four big cores without the loss R18 and R62 measured; a column kernel under the A78's `SMLAL`
  floor of about 1.5 (§5); no prefetch into holes; and the 4.6 GB step held (pre-registered P 0.06). My joint
  chance is under 10⁻⁴, and it fails on Qwen2.5 even then, so it would be reported as "did not generalise".
- Byte and compute limits bind together. Full density at k = 1 caps at 10 to 12.8 tok/s on three or four big
  cores; the bus forbids 15 at full bytes with any number of cores. So 15 needs at least 25 % fewer bytes at the
  bus ceiling (50 % at r = 30) **and** no more compute per token than today. Inside +5 %, the only byte cuts found
  cost compute (columns, k about 2) or are unpriced (rows, §3).
- **So: no. memo-01 §1 stands with these numbers.** What R69 can change is which 6 to 9 tok/s path is worth
  building, not the verdict.

## 5. Review of Bonnie's draft (`bonnie-draft.md`, read 2026-11-19)

1. **K1's centre is too low on the A78: `SMLAL` is V0-only.** Her count (24 vector µops per 128 weights =
   0.1875 V/w, halved over two pipes) assumes the µops balance. From the guide (§1.1): the 16 `SMLAL` go to V0
   only, the 4 `USHR` to V1 only. V0 carries 16 µops per line, so the integer part is 16 cycles per 128 weights =
   **0.125 cycles per weight**, not 0.094. With her float step (176 µops per strip and block, 0.036 at d = 0.60)
   the floor is 0.161 against the stock floor 0.082: **1.96 ×** by count, not 1.58. With her excess factor,
   **k = 2.1 (1.5 to 2.6)**. Against her lines for Mistral / Qwen2.5: over the full-gain line (1.81 / 1.78), near
   the 1.15-gain line (2.44 / 2.25), under the kill line (2.81 / 2.59). By her own model that is about 7.4 and
   6.6 tok/s at r = 24, not 8.5 and 7.5. The same floor applies to Tom's kernel, so K1's edge over it is the removed
   loads and stores, not the µop count. The host (M4) cannot see this limit: T1 can pass and D2 still fail.
   Her alternatives do not escape it: the `tbl` form and the zip-into-`sdot` form both come to 0.125 cycles per
   weight or worse by the same pipe count. **`sdot` does 32 MACs per cycle on an A78, `smlal` 8.** That gap is
   the hardware reason a column kernel costs about 2 ×, and it will not be coded away on this core.
2. **Her recalled pipe counts are right** (two vector pipes on the A78; `SDOT` 2 per cycle). I could not read the
   X1's guide; the X1 thread may do better than the A78s, which then set the pace.
3. **Y1b is worth more than she prices**: it is also the kernel §3 needs. I would add four arbitrary row pointers
   to T2 (my E6) so that one bench serves both.
4. **Y1b's count**: 4 `SHL` are V1-only, but 31 µops still balance to 15.5 cycles. Her 1.5 holds.
5. **D1 is the best test in either draft**: a real 4 GB body with no anonymous copy and no kernel work. I would
   run it whenever the 3.4 step is held, on Yi-1.5-6B too, since P(4.2 held) is only 0.12.
6. **b_eff**: agreed, and it applies to §3 as well, in a milder form: a skipped row is 36 lines on Mistral, a
   skipped column tile 1 to 8.
7. **Side file as an artifact (§1.3)**: I agree with "same category, stricter handling". A column-major or
   proxy file is not a bijection of the stock layout in the same obvious way; her question 3 should name both.
8. No objection to her bands for Y1a, Y1c or K2, or to her order.

## 6. TEST list and TRY list

| # | test | cost | closes if | opens if |
|---|---|---|---|---|
| E1 | Host: `down` input only at 60 / 70 / 80 %, both models, 40 chunks, `-b 512` and `-ub 1` (existing `tools/c3_h3_grid.sh`, new threshold levels) | Tom, under 1 h | under 60 % inside +4.5 % on either model: §3 cannot beat column sparsity; closed | 70 % on both |
| E2 | Host: gate rule (zero neuron i when \|SiLU(g_i)\| is under a train-calibrated threshold) at 30 / 50 / 70 % | Tom S (one rule in `c3_colskip.py`) | under 40 % inside +4.5 % on both | variant A becomes a try item with no side data |
| E3 | Host: proxy fidelity. Rank by a 1-bit and a 2-bit copy of `gate` and `up` (numpy on the dequantised tensors), mask at E1's frontier s; PPL and recall of the exact kept set | Tom M; only if E1 opens | PPL cost over twice E1's at the same s | within 1.5 × : variant B |
| E4 | R69 readout, no phone time: the four-row table of §1.3 from the `p2`, `p2hot` and `x1` lines | parser addition, 20 lines | rows 1 or 2: rank 2 closed | rows 3 or 4 |
| E5 | Device, next kit, about 30 s: the real gemv with 2 and 4 row-group streams and with `PRFM` at +1 / +4 / +8 KB, 1 and 3 A78, hot and cold. Only if E4 gives row 3 or 4, or the X1 gain is at least 1.10 | 30 lines in the probe file; own pre-registration | every cell under 1.10 | a cell at 1.15 or more: build §1.5 |
| E6 | Host: Bonnie's T2 with four arbitrary row pointers, k at kept fractions 1.0 / 0.5 / 0.3, `memcmp` identity against the 4x4 gemv on the same rows | Tom S on top of T2 | k over 1.7 | – |

**TRY list** (each gated; all *estimated*; generic across models unless noted):

| rank | lever | gate | price on a 4.0 GB body | label |
|---|---|---|---|---|
| 1 | Bonnie's Y1b in-place row kernel, written with four row pointers | her T2, T4; my E6; R69 Rule C | removes 2.1 GB of anonymous copy for `gate` and `up` alone; 0 to +15 ms | bit-identical |
| 2 | Proxy-ranked FFN row skip on that kernel (variant B; A if E2 passes and E3 fails) | E1 to E3; the lead's reading of the proxy file | 6.5 to 9.3 tok/s at r = 24 | numerics change, one budget, checked at `-b 512` and `-ub 1`; threshold and proxy per model |
| 3 | Multi-stream and prefetch knobs in the stock gemv (§1.5) | Rule D as pre-registered, or E5 | × 1.0 to 1.2; P 0.10 | bit-identical |
| – | `LDNP`, `DC ZVA`, T-MAC tables, early exit, SpAx, NorSA, KV work | – | 0 or over budget | not proposed |

## 7. Sources (all read 2026-11-19)

- Arm Cortex-A78 Core Software Optimization Guide, Issue 4.0 (2021-04):
  <https://documentation-service.arm.com/static/60a5413bd63d3c31550c391e>. PDF text extracted locally with a
  short script; the tables come out flattened, and I read the rows named in §1.1.
- sbc-bench results table: <https://github.com/ThomasKaiser/sbc-bench/blob/master/Results.md> (fetch summary; the
  page says big-core tinymembench results, units not printed).
- Arm learning path, single-core bandwidth: <https://learn.arm.com/learning-paths/servers-and-cloud-computing/memory-subsystem/streaming-bandwidth/> (fetch summary).
- Jetson AGX Orin bandwidth: <https://forums.developer.nvidia.com/t/jetson-agx-orin-32gb-measured-memory-bandwidth-much-lower-than-theoretical-spec/331766>
  (HTTP 403; search summary only); OpenBenchmarking listings (search summary only).
- A78 erratum 2132060 and `PF_MODE`: TF-A mirrors, for example <https://git01.mediatek.com/plugins/gitiles/filogic/atf/+/39e0865e7a171f3252a4f2543cc9f80e7eca2af9> (search summary).
- Snapdragon 888 bus, 51.2 GB/s: <https://www.anandtech.com/show/16325> (search summary; the review pages redirect).
- DSU TRM: <https://documentation-service.arm.com/static/5e7e16e0b2608e4d7f0a3030> (truncated; not read).
- Sereno, OSDI 2026: <https://www.usenix.org/system/files/osdi26-xin.pdf> (text extracted locally; abstract and §3 read).
- T-MAC: <https://arxiv.org/html/2407.00088v2> (fetch summary; Tables 4 and 7); repack PR #5780 date from a
  mirrored commit list (search summary).
- Prox: <https://www.alphaxiv.org/abs/2607.27591> (summary page only). NorSA: <https://neurips.cc/virtual/2026/poster/154390>
  (abstract). SpAx: <https://arxiv.org/abs/2610.02598> (abstract). MNN-LLM: <https://arxiv.org/html/2506.10443v1>
  (search summary).
- Repository: memo-01 to memo-03, memo-02 §8, `memo-04-work/`, the R69 pre-registration, `reports/03_final.md`
  (R13/R14, R23, R41), `patches/llama.cpp-c3/0001-probe.patch` (the `x1` block), lab tree
  `ggml/src/ggml-cpu/arch/arm/repack.cpp:230`.
