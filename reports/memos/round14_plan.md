# Round-14 plan: a check of the R42 prediction, and the host screen for dense structural numerics (Bonnie, 2026-11-07 night)

Labels: *device* = `reports/device/r8_2026-11-06.json`, `r12`/`r13_2026-11-07.json` per-block lines (read, not re-scored); *host* = `lab/logs/r27/dense/`
(inventory, `dense_colskip.py` PPL table of 05_compute_less §8), `tools/ppl.sh`; *estimated* = arithmetic. Byte rate used throughout: **0.0411 ms/MB at
2,112 MHz** (R25, device; dense `MUL_MAT` sits on this ridge, so ms ∝ bytes). No device claim is made here; everything in Part 2 is a host screen.

## Part 1 — the R42 prediction (`MMID_FUSE` + two-chunk claim), checked

**What the fuse removes, from the per-block lines (device, clean blocks).** `GLU 116:393+175` µs per token (r12 run 1 blk 1) → **3.4 µs compute per GLU op**;
the 40 expert GLUs are the smallest of the 116 (512 × k̄ ≈ 2,000 floats, already tiny-op, no barrier of their own), so the fuse removes **40 × 3.3-3.5 µs =
−0.13 to −0.14 ms of GLU compute, not the whole 0.39-0.45 R42 writes** — 76 GLU ops (shared expert, SSM gate) stay. Barriers: the gate→up barrier goes,
**746 → 706** ✓ (exact count, hygiene line). `MUL_MAT_ID` ops **120 → 80** in the op line (the parser must sum compute + wait regardless of count). Nodes
1855 → **1775**. Per-slot: one Q8_K quantize instead of two for gate/up (−1 µs × 40 ops/token = −0.04 ms); one op-end tail instead of two for gate/up (the
`MUL_MAT_ID` wait is 0.75 ms/token clean over 120 op-ends = 6.2 µs each → −40 × 3-6 µs = **−0.12 to −0.25**); dst writes halve for gate/up (−0.02). Two-chunk
claim: the atomics term (2-6 µs/slot, §E of `round13_plan.md`) halves → **−0.16 to −0.5 ms** at 164 slots.

| component (ms/token at 2,112, counted blocks) | R42 | my check | source |
|---|---|---|---|
| `GLU` compute | → 0 (−0.39..−0.45) | **−0.13..−0.14** (40 of 116 ops; 76 remain) | device per-op cost 3.4 µs |
| `GLU` wait | — | −0.02..−0.1 (its wait is the next barrier's arrival, 17-175 µs clean, noisy) | device |
| `MUL_MAT_ID` wait (40 op-ends) | inside "≤ +0.1 compute" | **−0.12..−0.25** | device 0.75 ms / 120 |
| quantize + dst | — | −0.06 | estimated |
| two-chunk claim (atomics) | — | **−0.16..−0.5** | estimated (§E) |
| **sum, scored line `MUL_MAT_ID` + `GLU` (compute + wait)** | **−0.6..−1.3, central −0.9** | **−0.5..−1.05, central −0.7** | |

**Dissent 1 (band).** Re-centre to **−0.5..−1.05, central −0.7**; R42's upper half (−1.05..−1.3) assumed the whole GLU line goes to zero, which it cannot.
With pass ≤ −0.5 the pass sits at my band's low edge; I would set **pass ≤ −0.4, falsified > −0.15**, keeping the 0.25 gap.

**Dissent 2 (resolvability — the real problem).** The scored line's per-block sd on *clean* blocks is **2.3-4.5 ms** (device: r8 head2 ABBA 3.1 / 4.5, r8 skip
ABBA 2.4 / 2.3 at ≤ 10 majflt/token; r12 run 1 3.4-6.7). With three alternated runs on an r8-like day (≈ 30 counted blocks per side), σ(Δ) ≈ √(2 × 3² / 30) ≈
**0.78 ms**: the line cannot separate −0.7 from −0.2 at 1σ, and the §B clause (σ ≤ 1.0) would let through a verdict with a 50 % chance of being wrong-signed
inside the band. Resolving −0.7 at 2σ needs σ ≈ 0.3 → ≈ 200 counted blocks per side = 7 clean sessions. **Recommendation, before the build is frozen:**
(a) keep the sum as the *reported* line but require **σ ≤ 0.4** for a pass/falsify verdict (else "unresolved", one re-run session, then parked, as §B);
(b) add two **exact-count hygiene lines that cannot be noisy**: `barriers` 746 → 706 and `nodes` 1855 → 1775 on every B block (a miss = the fuse did not
engage = the run is void, not unresolved); (c) add a **low-variance component line that must move**: `GLU` compute B − A = **−0.12..−0.16** (its per-block sd
is **0.007-0.015** ms → 10σ resolution on 8 blocks); if `GLU` compute does not fall by ≥ 0.10 the fuse is not doing what the build says. The wait and atomics
parts (−0.3..−0.75 together) are the ones the day will hide. Say in R42 that this round's most likely honest outcome is "GLU and counts confirmed, the sum
unresolved" unless the day is r8-clean.

**Kit must log:** `fuse=` and `chunk2=` in the cpu-prof header (per side, like `kern=`); `nodes` / `barriers` per block (present); the `MUL_MAT_ID` op count
(the parser currently keys on name only — fine, but print the count so 80 vs 120 is visible); the parser's `--gate` sum must include `GLU` compute + wait;
class K check per §B.1 (fresh-process `ru_minflt` at 1 thread under `GGML_MMID_TEST1=1`, both knobs on vs off, within 0.5 %; the fuse touches the same rows,
the two-chunk claim the same chunks — the page set is identical by construction, the test is the proof). Identity at a **stated thread count** (R39: PPL
depends on `-t`; chunk 1 at `-ub 1` is 5.5102 at `-t 1`, 5.4336 at `-t 6`).

## Part 2 — dense structural numerics, host screen (cap PPL ≤ 10.0854 = +5 % over stock 9.6051; served 9.7886 at `-t 8` → ≈ +3.0 % left)

### (i) Dense byte map of the served file (host inventory, exact bytes; ms at 0.0411 ms/MB, *estimated*)

| class | tensors | MB/token | ms at 2,112 | note |
|---|---|---|---|---|
| head (tied `token_embd`, Q6_K 126.4) | as served today: head2 = repacked Q4_0x4 screen copy **86.7** + exact Q6_K recompute of 2,048 rows **2.6** | **89.3** | **3.7** | device head op 3.92 (R31) ✓ |
| `ssm_in` Q4_0 (1536 → 6448 = z 3072 + x 3072 + B 128 + C 128 + dt 48), 36 layers | 36 × 5.57 | **200.6** | **8.2** | largest dense block |
| `ssm_out` Q4_0 (3072 → 1536), 36 layers | 36 × 2.65 | **95.6** | **3.9** | |
| shared expert `gate/up/down_shexp` (1024 wide; Q4_0, 5 layers' down in Q4_1) | 35.4 + 35.4 + 31.0 + 4.9 | **106.7** | **4.4** | |
| attention q/k/v/o (4 layers, Q4_0) | 5.3 + 1.8 + 1.8 + 5.3 | **14.2** | 0.6 | |
| router `ffn_gate_inp` **F32** (1536 × 64 × 40) | | **15.7** | **0.65** | F32 in the GGUF |
| norms, conv1d (F32), ssm_a/d/dt | | 3.4 | 0.14 | |
| **total** | | **525.5** | **21.6** | device `MUL_MAT` 22-24 ✓ (R31 "89 + 436 MB") |

### (ii) Candidate cuts (mechanism; bytes; ms; runtime implementation; PPL band — all *estimated*, bands to be replaced by the screen)

| # | cut | mechanism | bytes | ms | load-time implementation (served file untouched; head2 precedent: anon copy, dispatch swap) | PPL band |
|---|---|---|---|---|---|---|
| **C1** | **router copy F32 → F16 (or Q8_0)** | the router gemv reads 15.7 MB of F32 per token; F16 weights round at 2⁻¹¹ relative, far under the softmax-threshold noise of the skip rule | −7.9 (F16) / −11.5 (Q8_0) | **−0.32 / −0.47** | at load: convert `ffn_gate_inp` of every layer into an anon F16 (or Q8_0) copy (7.9 / 4.2 MB), route its `MUL_MAT` to the copy with the stock F16/Q8_0 gemv; `GGML_ROUTER_Q=f16|q8_0`, unset = stock | **F16 +0.00..+0.05 %, Q8_0 +0.0..+0.15 %** (routing flips only at near-ties; k̄ unchanged to ±0.01 — log it) |
| **C2** | **low-rank head2 screen** | head2 already uses an *approximate* screen (Q4_0x4 of all 100,352 rows) to pick 2,048 rows for the exact Q6_K recompute; replace the screen by a rank-r factorisation of the dequantised tied embedding: logits′ = (x·Uᵣ)·Vᵣᵀ with Vᵣ (100,352 × r) stored Q4_0x4, Uᵣ (1536 × r) F16; the exact recompute of the top-2,048 is unchanged, tokens outside the top-2,048 keep their low-rank logit (so PPL stays finite and measures the real path) | r = 768: 86.7 → 43.4 + 2.4 = **−41**; r = 512: → 28.9 + 1.6 = **−56**; r = 384: → 21.7 + 1.2 = **−64** | **−1.7 / −2.3 / −2.6** | the factorisation is the cost: randomised SVD of a 100,352 × 1,536 matrix at rank 768 ≈ 2.4 × 10¹¹ flop ≈ **8-15 s on three A78s** (*estimated*) — too long for TTFT; either computed once per install and persisted as a **runtime-derived artefact in app storage** (like the hot-list file of 04_campaign A6: regenerable, not the served file) — **the owner must say whether a persisted derived file is "runtime"** — or computed in the background during the first session while the stock screen serves. Kernel: Uᵣ projection (F16 gemv, 1.2-2.4 MB) + the existing Q4_0x4 `sdot` screen over r columns instead of 1,536 | **r 768: +0.1..+0.6 %; r 512: +0.3..+1.5 %; r 384: +0.8..+3 %** (the screen's top-2,048 recall decides greedy; the denominators carry the low-rank error) |
| C3 | **static SSM-head pruning** (Mamba-2 heads: 48 × 64 channels per layer) | drop whole heads: 128 of 6,448 `ssm_in` rows (z and x of the head), 64 `ssm_out` columns, the head's conv1d channels, A/D/dt entries; the recurrent state of a pruned head simply does not exist → consistent by construction. **Caveat:** `ssm_norm` is a gated RMSNorm over all 3,072 channels (n_group = 1); removing channels changes the RMS denominator → either renormalise by √(kept/3072) (not exact; part of the screen) or keep the norm statistics from a calibration pass | per head per layer 0.22 MB; 6 of 48 heads × 36 layers = **−47 MB**; 12 of 48 = **−94 MB** | **−1.9 / −3.9** | criterion (host, `cb_eval` hook on the SSM output y per head over `lab/text/en.txt`, never wikitext-2): head importance = mean ‖y_head‖ × ‖W_out[:, head]‖; prune the lowest; verify the chosen set by PPL. Runtime: load-time anon copies of the pruned `ssm_in` / `ssm_out` / conv rows (Q4_0 rows are output-major → row removal is a memcpy of whole Q4_0 rows for `ssm_in`; `ssm_out` column removal needs a re-quant of the 1,536 rows at 2,944 → re-quant error is part of the screen), a per-layer kept-head mask in the SSM ops | **6/48: +0.5..+3 %; 12/48: +2..+8 %** — no fine-tuning; Mamba heads are not redundant the way FFN neurons are |
| C4 | **static shared-expert neuron pruning** (1024 wide) | drop the neurons with the smallest mean |silu(gate)·up| over en.txt (the §8 table's *dynamic* 24 % at +0.3 % is an upper bound on how well a static set can do) | 25 % → **−26.7 MB** | **−1.1** | load-time copies of gate/up rows (row removal) and down columns (re-quant, as C3) | **+0.8..+3 %** |
| — | hard head-vocabulary subset (top-V′ rows by corpus frequency) | **dead by the gate's definition**: any wikitext-2 target outside V′ has log p = −∞ → PPL = ∞; choosing V′ to cover wikitext-2 is choosing on the test set. Only a soft screen (C2) keeps every token scorable | | | | |
| — | dynamic per-token channel skipping (`ssm_out` 46 % small channels, §8) | withdrawn in R29 (an fp32 axpy over kept channels runs +2..+5 ms over the byte-rate gemv); only *static* cuts keep the byte-rate kernel | | | | |
| — | `ssm_in`/`ssm_out`/shexp below 4 bits | no byte-rate container below 4 bits on dotprod cores (R29/R31) — closed | | | | |

**Sum at centrals, if the screen passes:** C1 −0.4 + C2 (r 512) −2.3 + C3 (6/48) −1.9 + C4 −1.1 = **−5.7 ms** for ≈ +0.1 + 0.8 + 1.5 + 1.5 ≈ **+3.9 %** PPL —
over the +3.0 % headroom; the realistic buy inside the cap is **C1 + C2 (+ one of C3/C4 at its small setting) ≈ −3..−4.5 ms**, which is the size of the
kernel's gain. Printed arithmetic (*estimated*, stacks device-unshown terms): 48.7-49.3 (R41) − 0.7 (r14) − 3..−4.5 → 43.5-45.6 → **22-23 fault-free**; not
pre-registered as anything until a screen number exists.

### (iii) PPL screen protocol (host; `tools/ppl.sh`, **`-t 8` stated on every number**; base = served config 9.7886; cap 10.0854)

1. **Simulate before building**: C1-C4 are all expressible as edits to the loaded weights or to the logits — `dense_colskip.py`'s `cb_eval` hook (05 §8, thr 0
   reproduces the base to four decimals) for C3/C4 (zero the pruned rows/channels, apply the norm compensation), a numpy post-process of the saved logits for
   C2 (`dense_headthr_check.py` lineage: compute low-rank logits, replace the top-2,048 by exact, score), an in-place F16 cast for C1. No kernel is written
   until a cut passes.
2. **Order** (cheapest-and-safest first, each 60 chunks then full set for survivors; 60-chunk base 9.8702 is a different number — use ratios to base, not
   absolutes): C1-F16 → C1-Q8_0 → C2 r 768 → C2 r 512 → C2 r 384 → C4 at 12.5 % / 25 % → C3 at 6/48 / 12/48 → the best passing **combination** on the full set
   (the §8 table shows "the sum is not the parts": 0.01 on all targets cost +3.5 % where the parts summed to +1 %).
3. **Pass iff** full wikitext-2 PPL at `-t 8` ≤ **10.0854** for the combination that would be served, **and** `lab/text/en.txt` PPL ratio to base ≤ the wikitext
   ratio + 1 % (R17 precedent: wikitext hid a prose cost), **and** greedy 32 on the two kit prompts stays coherent (numerics changes need not be identical; a
   changed text is logged, a degenerate one fails), **and** k̄ at skip 117 within ±0.03 of 4.02 (C1 only).
4. **Pre-registered PPL bands** (ratios to base, *estimated*): C1-F16 **1.000-1.0005**; C1-Q8_0 1.000-1.0015; C2 r 768 **1.001-1.006**; r 512 1.003-1.015; r 384
   1.008-1.03; C4 12.5 % 1.003-1.012, 25 % 1.008-1.03; C3 6/48 **1.005-1.03**, 12/48 1.02-1.08. Falsifiers: C1-F16 > 1.002 means the router sits on
   threshold ties more than the skip-rule arithmetic says (log k̄); C2 r 768 > 1.01 means the tied embedding has no low-rank structure worth bytes → C2 dead at
   every r; C3 6/48 > 1.04 → Mamba heads are not prunable without training → C3 dead.
5. **Device pricing rule carried from R29/R31**: a byte cut pays only through a kernel at the byte rate on the A78; C1-C4 all end in the stock F16/Q8_0 or
   repacked Q4_0x4 gemv (device-shown at the ridge, R31), so ms ∝ bytes holds; C2's extra Uᵣ projection is 1.2-2.4 MB (≤ 0.1 ms).

### (iv) What makes a cut generalise (goal #4)

- **C1** is the most general: routers are stored F32 in most MoE GGUFs (Mixtral, Qwen-MoE, DeepSeek-V2/3, Granite) and are read every token; "requantise the
  router at load" is a one-rule runtime policy for every MoE on every bandwidth-bound device; 0.3-0.5 ms here, more on larger-expert-count models.
- **C2** generalises to every model whose head is a large share of dense bytes (small models with 128k-256k vocabularies: Llama-3 8B-class heads are 0.5-1 GB
  in Q4; Gemma 256k); the two-stage head2 is already the generic structure — the low-rank screen is a drop-in for its first stage. Its cost (an SVD per model)
  is a one-off per install, which is why the persisted-artefact question must be settled by the owner.
- **C3** is specific to Mamba-2 hybrids (Granite-H, Nemotron-H, Falcon-H1, Bamba) and generalises as a *method* (calibration-based head ranking + state-consistent
  removal), not as numbers; **C4** generalises to every MoE with a shared expert (DeepSeek, Qwen-MoE) and every dense FFN.
- Across devices: all four are byte cuts, so the gain is ∝ bytes/ridge on any bandwidth-bound dense phase and **≈ 0 on compute-bound hosts** (the M4 cannot
  measure the speed, only the PPL — label every ms here *estimated* until a device round).

**Owner questions to carry (not mine to decide):** (1) is a persisted runtime-derived artefact (C2's factorisation, regenerable, in app storage) "runtime" under
the frozen-file rule? (2) C1 changes routing on near-ties — is a bit-exact-in-text but numerics-changed router inside the +5 % cap or does the owner want it
under the +2 % speed-rule budget (it is a different mechanism, but it touches the same decision)?

## F. After the R43 host screen: C1-F16 as a kit item, the C2 clamp variant, the C3/C4 protocol (Bonnie, 2026-11-08)

Labels as above; R43 numbers are *host* (PPL machine-independent); every ms is *estimated* at the device byte rate 0.0411 ms/MB (R25). Owner's rules applied:
no new budget, C1 sits under both bounds, cached runtime-derived artefacts are allowed. Correction to §(ii): C3's bytes were over-counted — one Mamba-2 head
is 128 `ssm_in` rows × 864 B + 64 `ssm_out` columns × 1,536 × 0.5625 B = **166 KB**, so 6/48 heads × 36 layers = **−35.9 MB (−1.5 ms)**, 12/48 = **−71.7 MB (−2.9 ms)**
(was −47 / −94 MB).

**(1) C1-F16 as a kit item.** PPL accounting (host, `-t 4` ≡ `-t 8` at `-b 512`, R43): served 9.7886 → **9.7864** = +1.888 % over stock 9.6051; room under the +2 %
speed-rule bound 9.797 = **0.0106 (+0.11 %)**; under the +5 % cap 10.0854 = **0.299 (+3.06 %)**; en.txt 11.4887 → 11.5062 (+0.15 %, inside wiki + 1 %); k̄ +0.003;
greedy identical on both prompts (slots skipped 11,659 vs 11,616 → routing moves on near-ties, text does not). **Class F** (it changes the router's decision on
ties → it can change which experts, hence which pages, are touched; the host page-set test is not applicable because the set legitimately differs) → the
side-level rule applies (both sides' mean ≤ 10 majflt/token, fault ratio reported), not the block gate. Device prediction, two lines: (a) **the kit must add a
`router=` µs/token field** (the 40 router `MUL_MAT`s, 393 KB F32 each, like `head=`) — today's cost is unmeasured: 15.7 MB at the byte rate = 0.65 ms, but 40 ops
of 393 KB may be fixed-cost-dominated; prediction **`router` B − A −0.25..−0.35 if the op is at the byte rate, −0.10..−0.20 if fixed-cost-dominated; falsified
> −0.05**; (b) `MUL_MAT` total (contains the router) **−0.15..−0.40, central −0.3**; `MUL_MAT_ID` 0 ± 0.3 (k̄ +0.003 = +0.1 slot/token ≈ +0.01 ms, invisible);
head / `SSM_SCAN` 0 ± 0.3. Resolvability: `MUL_MAT` per-block sd ≈ 1.0 clean (r8) → σ ≈ 0.26 at 30/side — a −0.3 effect is 1σ on `MUL_MAT`, which is why the
`router` line (fixed small ops, sd ≈ 0.02-0.05 expected) is the one to score. **Not worth a phone session alone**: ride in the r15 kit as a second ABBA next to
whatever r14's result leaves (ABBA `ROUTER_CAST:0:f16` ×2, alternated), ≈ 3 min of kit time. It is PPL-*negative* and generalises to every MoE GGUF with F32
routers — the result worth having even at −0.3 ms. Default-on candidate only after a device `router` line ≤ −0.1 at ≤ 10 majflt/token.

**(2) The C2 clamp variant — kill it now.** Today's plain head2 keeps the **Q4_0 screen logit** for the 98,304 non-picked rows (patch: "the rest keep the screen
logit"); that is why plain head2 costs +0.01 %: the screen logit is within Q4_0 error of the exact one, so the denominator is right and a target outside the
picked set (rare: in_set 1.0 on the kit prompts, overlap 0.91-0.96) still gets a near-exact log p. The low-rank screen fails on **both** counts: recall (in_set
0.87, overlap 0.33-0.53 at r 768) and denominator. A clamp fixes neither: (i) clamp to the 2,048th exact logit → 98,304 rows at l₂₀₄₈ add ≈ 48 × the picked set's
tail mass to the denominator → PPL ratio ≫ 2 — worse than the plain tail, *estimated* from the count alone; (ii) clamp to −∞ → the denominator loses the true tail
mass (1-3 % of probability; PPL is *flattered*, so a −∞ PPL is **not a valid gate measurement**) and every target outside the picked set has log p = −∞ → PPL = ∞ at
in_set 0.87; (iii) clamp to the low-rank value is R43. The only clamp that could work is "plain screen for recall, low-rank for nothing" — i.e. no byte saving.
Kill rule, if Tom wants one cheap check before closing (no PPL needed): the in_set curve over r on the saved host logits; C2 is alive only if **in_set ≥ 0.995 at
r ≤ 1,024** (−29 MB). With 66.5 % of the energy in 768 directions I expect in_set 0.995 only near r ≈ 1,400 (saving < 8 MB). **C2 closed at every r and every clamp.**

**(3) C3 / C4 host protocol for Tom (no wikitext-2 anywhere in the choice of what to prune; `lab/text/en.txt` ≥ 20k tokens is the calibration text).**
- *C4 first — static shared-expert neuron pruning (1,024 neurons × 40 layers).* Importance of neuron j in layer l = mean over calibration tokens of
  |silu(g_j)·u_j| × ‖W_down[:, j]‖₂ (capture the GLU output = `down_shexp`'s input with the `cb_eval` hook of `dense_colskip.py`; 40 × 1,024 running means, one
  pass). Prune the lowest **12.5 % per layer** (128 neurons), then 25 %. **Simulate the runtime exactly**: zero the pruned GLU channels **and** replace `down_shexp`
  by its dequant → drop columns → re-quant Q4_0 (the kept 896 / 768 inputs re-blocked in 32s; this re-quant error is real and must be in the screen); gate/up row
  removal is a memcpy of whole Q4_0 rows (output-major), no error. Bytes −13.3 / −26.7 MB → **−0.55 / −1.1 ms**. PPL band (ratio to 9.7886): 12.5 % **1.003-1.012**,
  25 % **1.008-1.03**; **kill: 12.5 % > 1.012 on 60 chunks** → C4 dead (then 25 % is not run). Load-time mechanism: anon copies (gate/up rows, re-quantised down),
  a per-layer kept-neuron list in the runtime-derived artefact (allowed), `GGML_SHEXP_PRUNE=<pct>`; the three shexp `MUL_MAT`s dispatch to the copies.
- *C3 second — static Mamba-2 head pruning (48 heads × 36 layers), only if C4's PPL-per-ms ≤ 1 %/ms.* Importance of head h in layer l = mean over calibration
  tokens of ‖y_h‖₂ (the 64 channels of the gated-norm output = `ssm_out`'s input, per head) × ‖W_out[:, h]‖_F. Prune the lowest **6 per layer** (12.5 %), then 12.
  **Two-stage simulation**, both through the hook: (a) zero the head's 64 channels at `ssm_out`'s input (head removed, norm statistics untouched) — a lower bound on
  the cost; (b) the runtime-equivalent: zero the head's z/x rows at `ssm_in`'s output **and** rescale the gated RMSNorm by the per-layer constant
  √(1/(1 − f_l)), f_l = the pruned heads' mean share of the norm's energy on the calibration text (the runtime can only norm over kept channels). Score (b);
  report (a). The recurrent state of a pruned head does not exist → consistent by construction; A/D/dt/conv entries of the head are dropped with it. Bytes −35.9 /
  −71.7 MB → **−1.5 / −2.9 ms**. PPL band: 6/48 **1.005-1.03**, 12/48 **1.02-1.08**; **kill: 6/48 (b) > 1.04** → C3 dead. Load time: `ssm_in` row removal (memcpy of
  Q4_0 rows), `ssm_out` dequant → drop columns → re-quant (kept 2,688 = 84 blocks of 32 ✓ at 6/48; 2,304 at 12/48 ✓), conv/A/D/dt masks, the norm constant;
  `GGML_SSM_HEAD_PRUNE=<n>`; the artefact holds the kept-head lists.
- *Pass for a device round (either):* full-set wikitext-2 ≤ 10.0854 for the **combination** that would be served (C1-F16 + the cut), en.txt ≤ wiki + 1 %, greedy
  coherent on both kit prompts, and **the combination's bytes ≥ 36 MB (≥ −1.5 ms)** — below that the device line cannot be resolved (`MUL_MAT` σ ≈ 0.26 at 30/side)
  and the round is not worth the owner's time.

**Ranking.** 1. **C1-F16** — done on the host, free in PPL, rides in r15. 2. **C4** — half a Tom-day of host simulation, mechanism well understood, −1.1 ms at 25 %
if +0.8..+3 % holds. 3. **C3** — a Tom-day, riskier per ms (SSM heads carry the mixing; no training), −1.5 ms at 6/48. 4. **"Stop here"** is the right call *if* C4's
12.5 % step fails its kill: then the dense line closes at **−0.32 ms** (C1) and the +5 % cap stays unspent — spending it on C3 alone for −1.5 ms (≈ +0.6 tok/s
*estimated*) at +1..+3 % PPL is a poor trade I would not recommend. Everything above is host work; no device round is proposed until a combination clears
both the PPL cap and the −1.5 ms resolvability floor.
