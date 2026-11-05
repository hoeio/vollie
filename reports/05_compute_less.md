# 05 — Computing less per printed token (Bonnie, 2026-11-02, round "r21", host only)

Question (owner, `04_campaign.md` §9): which tokens, layers, experts or operations can be left uncomputed —
or computed once and reused — with a quality cost inside the gate, by a rule that does not depend on this
model or this phone? Every number below is **host** (M4 Pro, wikitext-2 test, ctx 512, 60-chunk screens
unless stated; raw logs `lab/logs/r21/`) or **estimated** from the R18/R20 device profiles. Nothing in §0-§4
was run on the phone; §5-§7 add the device scorecards and pre-registrations of rounds 3-6 (R23-R27), scored in `03_final.md`.

## 0. The arithmetic the ranking is measured against

One decoded token on the phone (R20, 3 threads, profile scale, ≈ 96 ms): `MUL_MAT_ID` 42.5 compute + 8.2
wait (**≈ 8.5 ms per routed expert-slot**, 6 slots), dense `MUL_MAT` 26 + 2 (lm_head Q6_K 126 MB ≈ 6 ms
of it), SSM/small ops ≈ 10, other waits ≈ 7. Best-run scale with the three bit-identical defaults
(13-15 tok/s, estimated): 67-77 ms, experts ≈ 36-42 ms, **6-7 ms per expert-slot**, ≈ 1.7-1.9 ms per layer.

20 tok/s = 50 ms. From 67-77 ms that is **−17 to −27 ms**, i.e. the token has to lose 25-35 % of its
multiply-adds *after* the raw levers. In expert-slots: 2.5-4 of 6 per token. In layers: 9-14 of 40.

Quality budget: +0.5 % working tolerance (9.6531 on the full set), +10 % hard (9.948 vs F16 9.0438;
stock D6 is already +6.2 % over F16, so the hard limit leaves **+3.6 % over stock** = 9.948/9.6051).
(Corrected after R22 / owner's decision of 2026-11-02: hard limit raised to +20 % vs F16 = 10.853; the speed
rule has its own approved budget of +2 % over stock = 9.797 full set. §5 uses these.)
On the 60-chunk screen the baseline is 9.8702 (all three builds identical, so one reference serves).

## 1. Ranked families

| # | family | lever on this testbed | gain at best-run scale (est.) | quality (host) | verdict |
|---|---|---|---|---|---|
| 1 | **(a) adaptive expert count by the router's own weight** | skip slot if normalized w < t (exists: `GGML_EXPERT_SKIP`) | −6 to −8 ms inside +0.5 %; **−12 to −17 ms at k̄ ≈ 4.2 for +1.9 %** (corrected after R22: +1.88 % screen is k̄ 3.86; the +2 % *full-set* budget lands at t ≈ 110, k̄ ≈ 4.3, see §5); 20 tok/s needs k̄ ≈ 3-3.5 → ≈ +5-9 % | adaptive is **3× cheaper in PPL per expert saved than uniform k** (measured) | the only family with the magnitude; price is the owner's call |
| 2 | (e′) **bounded cache of re-laid experts, bit-exact** (top-N per layer, adaptive per session; a full re-laid copy = 2.59 GB anon is impossible on this phone) | per-MAC gap Q3_K vs Q4_0 experts measured 1.58× on host; dense rate is the 4× ceiling | N = 16 (648 MB anon, memory-neutral if same-size): **−9 to −17 ms** at k̄ = 6, −19 to −29 ms **together with #1 at k̄ 4.2** (profile scale) | none if the kernel keeps Q3_K's int sub-block sums and float order | the multiplier for #1; §1(e′) has the per-N table; host microbench first (S5) |
| 3 | (d) cheap-token paths (draft confidence, emit-without-verify) | 350M draft agrees with the target's argmax on 55.7 % of tokens; gate p ≥ 0.98 → 8 % of tokens at 99 % agreement | **negative on this phone**: the target still has to ingest every token (state), so an unverified emit saves only the lm_head (~6 ms) and pays 14-24 ms of draft (R20) | +0.85 % at 8 % coverage, +0.12 % at 6 % | dead here; generalizes only to a device with a free draft lane |
| 4 | (b) adaptive depth / early exit | skip a layer ≈ −1.8 ms; need 9-14 layers/token | dropping even the cheapest single layer (4) *always* costs +2.5 %; layers 36-39 each +52 to +547 %; SSM state of a skipped Mamba layer is never updated | — | dead by arithmetic and by architecture; one cheap falsifier left (§3 S2) |
| 5 | (c) reuse across tokens | 45-50 % of experts repeat (R16) but with a *new input*; output reuse = wrong output | no arithmetic without a near-identical input; prefix/state reuse saves prompt time, not decode MACs | expected far over the gate | dead; falsifier is a 10-line cosine dump (§3 S4) |
| 6 | (a′) static per-layer k (sweep once per model) | 4 in lower half + 6 upper = +1.95 %; reverse +3.63 %; uniform 5 = +1.26 % | same budget as uniform 5 but worse | — | **falsified**: uniform beats both static splits; only the ordering (lower layers tolerate 4 better) survives |
| 7 | (a″) more experts where it is hard (k = 7) | +0.03 % PPL at k = 7 globally | none to redistribute | — | falsified: nothing to gain from extra experts, so adaptive-k is pure saving, not a trade |

### (a) Adaptive expert count — the numbers

Uniform k (v0.8.0, `--override-kv granitehybrid.expert_used_count=int:k`, 60 chunks, −t 6):

| k | PPL | Δ vs 9.8702 | per expert-slot removed |
|---|---|---|---|
| 7 | 9.8728 | +0.03 % | — |
| 6 | 9.8702 | 0 | — |
| 5 | 9.9943 | **+1.26 %** | 1.26 %/slot |
| 4 | 10.5351 | **+6.74 %** | 3.4 %/slot |
| 3 | 12.1213 | +22.8 % | 7.6 %/slot |

Weight-threshold skip (tiny build, `GGML_EXPERT_SKIP=<permille>`, no renorm — R17 showed renorm hurts),
same 60 chunks. Average experts per token k̄ from the router-weight CDF below (generation on two prompts;
the PPL run itself does not print the share — Tom spec S1 fixes that):

| t | share of slots with w < t (story / official) | k̄ ≈ | PPL (60 ch.) | Δ | full set (R17) |
|---|---|---|---|---|---|
| 0.085 | ~7 % | 5.6 | 9.8707 | +0.005 % | — |
| 0.090 | 10-12 % / 9-15 % | 5.3 | 9.8597 | −0.11 % | — |
| 0.095 | ~15 % | 5.1 | 9.9021 | +0.32 % | — |
| 0.100 | 15-18 % (wikitext 21.5 %, R22) | **4.71 measured** (corrected after R22; was est. 5.0) | 9.9158 | +0.46 % | **+0.82 %** (9.6834); en.txt +2.1 % |
| 0.120 | ~30 % (wikitext 35.7 %, R22) | **3.86 measured** (corrected after R22; was est. 4.2) | 10.0554 | **+1.88 %** | — |

Router-weight CDF (share of (token, slot) pairs whose normalized weight is below t; `lab/logs/r21/skipshare.txt`,
greedy, 64 decoded tokens per prompt, bit-exact counter in the module): t = 0.03: 0 %, 0.05: 0-2 %, 0.07: 2-5 %,
0.09: 9-15 %, 0.11: 20-24 %, 0.13: 36-40 %, 0.16: 57-59 %, 0.20: 72-75 %, 0.25: 87 %. The two texts agree
within 1-6 points, so the rule's *coverage* is a property of the router, not the text. There is no bimodality
(no population of "two dominant experts" tokens): a *gap* rule (drop slots with w < g × w_top) will behave
like the threshold rule; the cumulative-weight (top-p) form is the same ordering with a per-token budget.

What the two curves say, machine-independently:
- **At equal k̄ the router-adaptive rule costs ~3.5× less PPL than uniform k** (corrected after R22: the measured
  k̄ are 4.71 / 3.86, so the rule saves *more* slots per PPL point than first estimated): k̄ 4.71 adaptive +0.46 %
  (60 ch.) / +0.82 % (full) = 0.36 %/slot vs uniform 5 1.26 %/slot; k̄ 3.86 adaptive +1.88 % = 0.88 %/slot vs
  uniform 4 3.37 %/slot. The router's weights do know which experts matter — the signal the owner asked about
  works, and it is the model's own. Against the two other router-derived rules (R22, §5) the absolute threshold
  is also the cheapest: 2.1-2.6× less PPL per slot than cumulative mass, 2.1-2.9× less than relative gap.
- The screen under-reads the full set by ~0.35 point at t = 0.10 (+0.46 vs +0.82 %); treat 60-chunk deltas as
  optimistic by that much.
- Extrapolating the adaptive curve (cost grows faster than linear: 1 slot → +0.8 %, 1.8 slots → +1.9 %):
  k̄ ≈ 3.5 ≈ **+5 %**, k̄ ≈ 3 ≈ +8-9 % (estimated) — the first is inside the hard limit, the second at it.
- Speed (estimated, proportional to slots; R18 measured skip 0.10 at −9.1 ms for 8 % skipped on the device,
  i.e. better than proportional, one run): t = 0.10 → −6 to −8 ms (13-15 → 14.5-16.5 tok/s) for +0.82 %;
  t = 0.12 → −12 to −14 ms (→ 16-18.5) for ≈ +1.9 %; 20 tok/s from 13-15 alone needs t ≈ 0.14-0.15
  (k̄ 3-3.5, ≈ +5-9 %). **Combined with the raw ceiling (15-17 tok/s = 59-67 ms) the gap is 9-17 ms = 1.3-2.5
  slots → ≈ +1 to +5 %.** That is the honest price of 20 on this phone: raw work to ~16 tok/s *plus* an
  owner-approved PPL budget of a few per cent spent through the router's weights.
- Generality: the rule reads the router softmax every MoE model already computes; one threshold (or one
  cumulative mass) per model, calibrated once on its validation text. It does not depend on the device.
  What does not generalize automatically is the *price*: Granite's six nearly flat weights (median 0.145,
  mean 0.167) mean every slot carries real mass; a model with peakier routing pays less.

### (d) Cheap-token paths — the numbers

Target (D6) vs Granite-4.0-350m Q8_0 on the same 30 chunks (7,650 scored tokens; `lab/draft_gate.py`;
stock `--kl-divergence` cross-check: same-top 55.67 %, mean KLD 1.19, draft PPL 19.19 vs 8.86):

| gate: draft p_max ≥ τ | coverage | agree with target argmax | wrong emits / 1000 tok | PPL of the mixed policy |
|---|---|---|---|---|
| 0.5 | 40.8 % | 84.4 % | 63.7 | +20.3 % |
| 0.8 | 21.7 % | 95.3 % | 10.2 | +6.7 % |
| 0.9 | 15.8 % | 97.5 % | 3.9 | +3.1 % |
| 0.95 | 12.0 % | 98.6 % | 1.7 | +1.5 % |
| 0.98 | 8.1 % | 99.0 % | 0.8 | **+0.85 %** |
| 0.99 | 5.7 % | 100 % | 0.0 | **+0.12 %** |

The target's *own* confidence (ceiling for any self-confidence rule): p_max ≥ 0.9 on 27 % of tokens with
95 % greedy accuracy; ≥ 0.99 on 14 % at 98.8 %. Greedy accuracy overall 54 %.

Why it is dead on the SD888 regardless of the curve: an emitted-unverified token still has to pass through
the target to update the Mamba states and KV (R20: a 2-token graph costs 2.3× one token — no amortization),
so the saving per gated token is the lm_head (~6 ms) and the argmax, against a draft at 14-24 ms per token
on the same cores plus 24 % lost to its memory. Even at 100 % coverage the sign is negative. A
"catch-up at reduced k" for gated tokens would recover ~25 ms of their expert time but corrupts the state
for every later token — a PPL cost only a two-model harness can measure; not worth building here. Where it
*does* generalize: a device with a free lane for the draft (NPU, idle cluster fast enough to lead) — then
the table above is the quality rule: τ ≈ 0.98-0.99 keeps the printed stream inside +0.5-0.85 % while 6-8 %
of tokens skip the lm_head. On this phone the A55 lane was measured too slow even for prefetch (R18).

### (b) Adaptive depth — the numbers

Single-layer drop sweep (old q4_0 file, en.txt 6 chunks, `lab/logs/layer_sweep.txt`; anchor measured today:
10.5094): layer 4 (cheapest) 10.7733 **+2.5 %**; typical middle layers 11.0-11.8 (+5 to +12 %); layer 32 +24 %;
layers 36 / 37 / 38 / 39: 20.3 / 16.0 / 33.6 / 68.0 = **+93 / +52 / +220 / +547 %**; layer 0 ×5,000. The top
four layers build the representation the tied lm_head reads: there is no exit before layer 40 without a
trained head (Granite has none; training is out of the rules). Skipping *middle* layers per token: one
layer ≈ −1.8 ms; the budget needs 9-14 per token; the cheapest layer costs +2.5 % when always skipped, so a
rule that skipped ten on average cannot stay under +10 %. Architecture: 36 of 40 layers are Mamba-2 — a
skipped layer's recurrent state misses that token forever (transformers can backfill KV; SSMs cannot). The
literature agrees in direction: training-free early exit buys little on modern dense models and needs a
calibration/alignment step (SimLens 2507.17618, "Diminishing returns of early-exit" 2603.23701, WAVE); the
trained variants (LayerSkip, CALM, Mixture-of-Depths) are outside the rules. **Dead.** The one 20-line
falsifier (S2) is listed for completeness; prediction: argmax agreement at exit 36 < 50 %.

### (c) Reuse across tokens

An expert's output depends on its input; consecutive tokens give the FFN a different normalized hidden
state, so "the same expert again" (R16: 45-50 %) saves I/O, not arithmetic — unless x_t ≈ x_{t−1}, which for
different tokens it is not (prediction: median per-layer cosine < 0.8; S4 measures it in 10 lines). Prefix /
state reuse (prompt cache, repeated contexts) saves prompt time, not decode MACs. **Dead** for the goal.

### (e) Everything else that cuts arithmetic per printed token

- **Bit-exact expert re-layout**: see (e′) below — only a *bounded* cache is feasible on this phone.
  Lossy re-layout (Q3_K → Q4_0 values) is out: 64 experts × 40 layers at 4.5 bits = 3.4 GB of anon memory on a
  phone killed at 3.6 GB.
- **Candidate-set lm_head** (E2): −3 to −5 ms, but a miss is silent (you cannot know the argmax left the set
  without the full head), so its quality rule is the same mixed-policy PPL as (d); not measured today.
- **Skipping the shared expert / dense FFN per token**: 107 MB of Q4_0 at bandwidth ≈ 5 ms; no per-token
  signal exists for it (it has no router). Not pursued.
- MTP heads, trained exits, distilled drafts: training — out.

### (e′) Bounded cache of re-laid experts — costed (Clyde's correction: a full re-laid copy cannot exist here)

**Why bounded.** One expert = gate + up + down = 3 × 20.62 MiB / 64 = **1.013 MB at Q3_K** (quant log,
`blk.*.ffn_*_exps` 20.62 MiB each); 64 × 40 = 2,560 experts = **2.59 GB** (2.42 GiB). Anon today ≈ 0.65 GB
(R20); the app died at 3.6 GB; and the page cache would lose the same bytes. So: re-lay only each layer's
hot experts into anon memory, keep the rest mmapped Q3_K.

**1. Coverage (host, `lab/logs/r21/hotset.txt`, from the r16 `GGML_EXPERT_LOG` runs; decode tokens only).**
Share of (token, slot) uses covered by each layer's N most-used experts, mean over 40 layers:

| N | story (199 tok) | official (84) | code (199) | list (199) | **pooled, 681 tok** | anon MB (same-size layout) |
|---|---|---|---|---|---|---|
| 4 | 36.9 % | 39.7 % | 30.7 % | 33.5 % | 21.5 % | 162 |
| 8 | 55.9 % | 58.9 % | 49.1 % | 50.0 % | 34.8 % | 324 |
| 12 | 68.2 % | 71.2 % | 61.9 % | 61.5 % | 45.5 % | 486 |
| 16 | 77.2 % | 80.0 % | 71.3 % | 70.4 % | **54.5 %** | **648** |
| 24 | 88.6 % | 90.7 % | 83.4 % | 83.2 % | 69.3 % | 972 |
| 32 | 94.6 % | 95.9 % | 90.8 % | 91.3 % | 80.5 % | 1,297 |

**The hot set is not text-independent.** Per-layer top-16 sets overlap 50-59 % between the prose texts but only
9-18 % between code and prose; a top-16 learned on one text covers 11-62 % of another's uses (code ↔ story 11-14 %).
The within-text 70-80 % of R16 is partly a short-sample artefact (1,200 uses per layer over 64 experts): pooled
over all four texts top-16 covers 54.5 %. So the cache must be **adaptive** (per-session frequency or LRU over
the last few hundred tokens), and its steady-state hit rate is the within-conversation coverage: **55-75 % at
N = 16**, 70-90 % at N = 24. Layers 10-19 are the most concentrated (top-16 = 0.65-0.74), layers 0-9 and 26-31
the least (0.37-0.48; `hotset.txt` last line) — N could be set per layer, but the pooled curve is what to plan with.

**2. Memory.** Anon bytes = 40 × N × 1.013 MB (column above). If the re-laid format keeps Q3_K's 3.44 bits per
weight (a reordering, like the dense repack copy), the original file pages of cached experts go cold and the
page cache need drops by the same amount: **total memory neutral, but 648 MB (N = 16) moves from evictable to
unevictable** — the same anon size the speculation run added (R20, 1.19-1.27 GB total), and that run lost 24 %
to faults because it did *not* free file pages. A Q4_0-sized layout (+31 %) costs a net +200 MB at N = 16. Rule:
N ≤ 16 on this phone, drop the cache on `onTrimMemory`, re-fill lazily.

**3. Gain (estimated, R20 profile scale: `MUL_MAT_ID` compute 42.5 ms = 240 slot-ops × 0.177 ms, wait 8.2 held
constant).** The assumption: the gap is layout/unpacking, not MACs. **Host falsifier run today** (1 thread, M4,
story 64 tokens, `cpu-prof`, `lab/logs/r21/fmt-*-t1.log`): D6 Q3_K experts 48.8 ms per token for 240 slot-ops
(0.203 ms each, 5.0 GB/s-eq); the old Q4_0-expert file (4 routed) 20.6 ms for 160 slot-ops (**0.129 ms each**,
10.3 GB/s-eq): same MACs per slot, 31 % more bytes, **1.58× less time** → the Q3_K unpack is the cost, not the
arithmetic. But Q4_0 experts still run at half the host's dense rate (23.4 GB/s-eq), so part of the expert gap
is the `MUL_MAT_ID` shape (one activation row per expert, 512-wide), which a layout cannot remove. Band used:
re-laid slot-op at **1.6× faster (measured, conservative) … 3.8× (dense per-byte rate, ceiling)**, i.e. 0.111 … 0.047 ms.
Expert compute = k̄ × 40 × [c · t_fast + (1 − c) · 0.177], c = pooled coverage (steady-state) … within-text (best):

| N | c | k̄ = 6, conservative | k̄ = 6, ceiling | k̄ = 4.2 (+1.9 % PPL), conservative | k̄ = 4.2, ceiling |
|---|---|---|---|---|---|
| 0 (today) | — | 42.5 | 42.5 | 29.8 | 29.8 |
| 8 | 0.35-0.55 | 36.9-33.8 (−6 to −9) | 31.6-25.3 (−11 to −17) | 25.8-23.7 | 22.1-17.7 |
| 16 | 0.55-0.75 | 33.8-30.6 (**−9 to −12**) | 25.5-19.1 (**−17 to −23**) | 23.7-21.4 (**−19 to −21 vs today**) | 17.9-13.4 (**−25 to −29**) |
| 24 | 0.69-0.87 | 31.6-28.7 (−11 to −14) | 21.0-15.3 (−21 to −27) | 22.1-20.1 | 14.7-10.7 |
| 32 | 0.80-0.93 | 29.7-27.7 (−13 to −15) | 17.4-13.5 (−25 to −29) | 20.8-19.4 | 12.2-9.5 |
| 64 (impossible) | 1 | 26.6 (−16) | 11.3 (−31) | 18.6 | 7.9 |

At best-run scale (×0.84) N = 16 alone would be −8 to −19 ms and N = 16 + k̄ 4.2 −16 to −24 ms (17-22 tok/s) **if
the host ratio transferred to the phone. It does not transfer as written — see Clyde's (e″) and the re-derived
band in (e‴): N = 16 is −3.5 to −12 ms at profile scale (median ≈ −6), and with k̄ 4.2 the combination lands at
15.6-20.4 tok/s, 20 only in the top corner.** The table above is kept as the upper bound.

**4. Quality.** None if the new kernel reproduces `ggml_vec_dot_q3_K_q8_K` exactly: per 256-block, the int32
sum of each 16-weight sub-block × its 6-bit scale is order-free (integers), but the per-block float
accumulation `sumf += d_block · d_act · isum` must run in the same block order per row (that is how `Q3K_X4`
stayed bit-identical). Tom checks: `GGML_EXPERT_RELAYOUT=64` (every expert re-laid) vs `0`, per-chunk PPL on
chunks 1-3 at `-ub 1` and `-ub 32`, must match to the last digit (5.5137 / 7.3175 / 7.0292), plus one 72-token
greedy generation byte-identical. Fallback if the fast layout needs a different float order (e.g. two
half-rows summed): declared numerics behind the env var, full-set wikitext-2 PPL within noise of 9.6051
(≤ +0.05 %) and en.txt 11.1001, and the owner told — it is then a non-exact change under the +0.5 % rule.

**5. Rerank.** #1 adaptive k and #2 the bounded re-laid cache stay in that order — #1 is a 20-line knob with the
PPL curve already measured; #2 is a kernel + cache + memory-pressure path — but they are one plan, not two:
the cache multiplies #1's saving, and only together do they put 20 in the band. The cheapest step is the
microbench (S5), which can kill the whole family in an hour if the re-laid dot is not ≥ 1.5× faster per MAC on
the M4.

### (e″) Clyde's check of (e′): what the 1.58× measures, and what the phone will see

Read against the kernel (`lab/llama.cpp-tiny/ggml/src/ggml-cpu/arch/arm/quants.c`, `ggml_vec_dot_q3_K_q8_K_x4`
→ `ggml_q3k_half`): per 256-weight block and row the Q3_K dot issues 16 `sdot`s and about 80 other
instructions — 2 × (8 mask/shift ops for the high bits + 12 and/shift/sub ops for the low bits), 6 `vpadd` + 2
`vmla`, 7 loads, and a scale unpack with a 16-iteration *scalar* subtract loop. Q4_0 is near 1 : 1. So Bonnie's
host number (0.203 vs 0.129 ms per expert, 1.58×) is a true **instruction-count** gap. Two things qualify it
for the phone:

1. **Instruction savings have transferred badly to this device before.** `Q3K_X4` removes activation loads only:
   −10.6 % of `MUL_MAT_ID` compute on the M4, **−2.4 % on the SD888** (R18, in-generation A/B). Transfer ratio
   ≈ 0.23. The per-block arithmetic agrees: at 3 threads the phone spends ≈ 120 cycles per 110-byte block for
   ~100 instructions (0.177 ms per expert, 3,072 blocks per thread), while an A78 issuing two NEON ops a cycle
   would need ~50-60 → **roughly half of the device's expert time is stalls, not instructions.** Scaling the
   1.58× by 0.23 gives ≈ 1.1-1.15× for an unpack-only re-layout on the phone, i.e. −3..−5 ms at N = 16 with a
   55-75 % hit rate, not the −9..−12 "conservative" column.

2. **The stall has a shape.** One expert row is 1,536 weights = 6 blocks = 660 bytes, then a jump of one row
   stride; the x4 kernel reads 4 such short streams. The A78's hardware prefetcher needs several sequential
   accesses to ramp, so each row restarts cold — latency-bound, low memory-level parallelism. The dense weights do
   not have this problem because the KleidiAI repack lays rows out as one long contiguous tile stream; that, not
   the 4-bit unpack, is plausibly why Q4_0 *experts* still ran at half the dense rate on the host (Bonnie's
   observation). On the phone, with a 2.5 GB working set, 4 KB file pages (TLB reach ~1-2 MB) and page-cache
   misses on top, this term should be larger than on the M4.

Consequences for the plan:
- A re-laid cache can win on the phone, but its device value comes from **stream shape + anon memory (no page
  faults, possibly huge pages)**, with unpack a minor third. Bonnie's band should be re-derived with those three
  terms, and the "conservative" column anchored to the device transfer ratio, not to the host.
- The three terms can be separated **without building the cache**, in kit r3, each bit-identical and env-gated
  (Tom, small):
  - `GGML_CPU_AB=knob:Q3K_VEC:0:1` on the device — the R15 kernel is another instruction-only change with a known
    host gain (−4..−8 %); its device delta is a second transfer-ratio point.
  - `GGML_Q3K_PF=1`: software prefetch (`prfm pldl1keep`) of the next rows' blocks inside the x4 loop — attacks
    the stream-shape term with zero memory and no layout; if it buys several ms, the layout's job shrinks to
    unpack and huge pages.
  - `GGML_EXPERT_ANON=<N>`: copy the N most-used experts per layer into anon memory **in the existing layout**
    (plain `memcpy`, pointer redirect in `MUL_MAT_ID`, LRU as in S5) — measures the memory term alone (faults,
    TLB with `MADV_HUGEPAGE` if THP is on) before any kernel work; ~30 lines against S5's kernel.
- Decision rule for building S5(ii): only if (prefetch gain) + (anon-copy gain) + 0.23 × (host unpack gain) at
  the measured hit rate is ≥ 8 ms per token on the device; otherwise the raw path is prefetch + anon copy
  alone, and the 648 MB is better spent on page cache.

(Clyde, host arithmetic + R18 device data; nothing new measured.)

### (e‴) Bonnie's reply to (e″): confirmed on the stall share, pushed back on its cause; new band; LRU replay

**1. Op count and cycles — agree on "≥ 50 % non-issue", disagree on what it is.** Counted in
`ggml_vec_dot_q3_K_q8_K_x4` → `ggml_q3k_half` (per 256-block per row): 16 `sdot`; mask prep 4 `bic`+4 `shl` and
4 `bic`+2 `shr` per half = 28; low-bit unpack 10 + 12 per half = 44; 12 `vpadd` + 4 `vmla`; 2 + 2 `vld` of the q3
bytes and hmask + 2 `shr` on hmask; 4 shared activation loads per row; scale unpack 1 `vld` + 6 `movl` + 1 `vaddv`
+ 1 `fmla` + conversion; plus ~60 scalar ops (12-byte `memcpy`, 4 masked `utmp`, the 16-iteration `-= 32` loop
unless clang vectorizes it). **≈ 16 sdot + ~105 other NEON + ~60 scalar**, a little above Clyde's 80 but the
same picture: `sdot` is 13 % of the NEON work. Cycles: device 0.177 ms / 3,072 blocks = 57.6 ns ≈ **121 cycles at
2.1 GHz** (R18 clocks) ≈ 1.0 NEON op per cycle; an A78 issues 2, the X1 4 → issue floor 30-60 → **yes, ≥ 50 %
of the device's expert time is not issue.** But the **M4 does the same block in 0.203 ms / 9,216 = 22 ns ≈ 99
cycles at 4.5 GHz ≈ 1.2 NEON op per cycle against a 4-wide floor of ~30** — with no page faults, 16 KB pages
and the best prefetcher in the business. The host is at 30 % of its issue bound; the device at ~50 %. So most
of the "stall" is **in the kernel, on both machines**: one `visum` chain per row (4 dependent `vmla` steps per
block, each behind a 3-deep `vpadd` tree and the `sdot`s), four rows per x4 call whose chains an A78's ~160-entry
window cannot fully overlap, and the scalar scale unpack sitting in front of every block. **The device-minus-host
gap is 22 cycles per block ≈ 18 % of expert compute (≈ 7.7 ms at profile scale, k̄ = 6) — that is the upper bound
for Clyde's stream-shape + memory terms together on a quiet day**, unless the M4 suffers the same stride penalty
(cheap to test: S5(i) now times contiguous vs real-stride rows). Why `Q3K_X4` transferred at 0.23: it removed
L1-hit activation loads, which cost issue slots on the 4-wide M4 and nothing on a latency-bound chain — it says
"not issue-bound", not "memory-bound". The same arithmetic says a Q4_0-style unpack (fewer instructions on the
same chain) will also transfer badly: **agree with (e″) that the pure unpack term is ≈ 0.23 × 37 % ≈ 8.5 %** of
cached-slot time.

**2. Three terms, N = 16, profile scale (`MUL_MAT_ID` compute 42.5 ms + 8.2 wait at k̄ = 6; hit rate h = 0.70
from the LRU replay below, 0.55 if static):**

| term | mechanism | needs | k̄ = 6 | k̄ = 4.2 | transfer evidence |
|---|---|---|---|---|---|
| U unpack instructions | fewer ops per block in the cached layout | re-laid format + kernel | 42.5 × 0.085 × h = **−2.5** | −1.8 | R18 `Q3K_X4` 0.23 |
| S stream shape | 4 × 660 B runs → one contiguous tile stream per 4-row group; prefetcher ramps | re-laid format (an anon copy in the old layout does nothing here) or `prfm` | 0 to −2.7 (half of the ≤ 18 % bound × h) | 0 to −1.9 | none yet; bound from device − host cycles |
| M memory | no minor/major faults, no TLB walks over 4 KB file pages for cached slots; protects 70 % of expert bytes from eviction | **anon copy only, any layout** | quiet day −1 to −7 (870 minor + 5-28 major faults/token today, ~1-2 µs and 0.1-0.5 ms each, ×h); pressured day −5 to −15 (R20 lost 24 % to cache loss) | −0.7 to −5 / −3.5 to −10 | R12-R14 fault counts, R20 day swing |
| **N = 16 total** | | | **−3.5 to −12 (median ≈ −6)** quiet day | **−2.5 to −8.5** | |
| D kernel ILP (not a cache term) | independent int32 accumulators per half/row, vectorized scale unpack; same bits | kernel only, **no memory, all 64 experts, every device** | −4 to −11 if latency-bound (host −15..−30 %, device −10..−25 % of compute) | −3 to −8 | to be calibrated by the `Q3K_VEC` A/B |

**Which dominates on the phone: M**, by range and by variance — it is the only term that scales with the day's
memory state, it needs no kernel, and Clyde's `GGML_EXPERT_ANON` measures it alone. U is small and certain; S is
bounded and unproven. The biggest *machine-independent* lever is **D**, which is not a cache at all. Order of
value: M (device-specific, 30 lines) > D (generalizes, kernel) > S > U. **S5(ii) as a re-laid-format cache is
below Clyde's 8 ms rule in the median; build order is re-set in §3.**

Combined with #1: k̄ 4.2 saves 12.7 ms of compute; plus the cache −2.5 to −8.5 → −15 to −21 ms at profile scale,
×0.84 → −13 to −18 at best-run: 67-77 → 49-64 ms = **15.6-20.4 tok/s (estimated)**; add D if it holds (−3 to
−8 at k̄ 4.2 → 18-23 tok/s). The honest statement: 20 needs k̄ ≈ 4.2 **and** two of {anon cache, ILP kernel,
raw waits}; none of the three alone.

**3. The kit-r3 probes — sufficient with one addition and one change of method.** Clyde's three separate U/S/M
as intended. Add **`GGML_Q3K_ILP=1`** (D; see S5), because the `Q3K_VEC` A/B alone gives the transfer ratio but
not the size of the prize. Change: `GGML_EXPERT_ANON` cannot be an in-generation A/B — the cache is state and
needs ~160 tokens to fill (below) — so it runs as **separate reload rows** (anon 0 vs 16, same prompt, ≥ 200
tokens, `minflt`/`majflt` per token and `anon`/`file` logged both sides), with a warm-up generation or a fill
during prompt processing (every expert a prefill touches is read anyway; the copy is then a free by-product).
Pre-registered predictions (device, profile scale, `MUL_MAT_ID` compute 42.5 ms):

| probe | measures | prediction | decides |
|---|---|---|---|
| `GGML_CPU_AB=knob:Q3K_VEC:0:1` | transfer of a chain-shortening change (host −4..−8 % of compute, R15) | **−2 to −4 % (−1 to −2 ms)**; if ≥ 0.5 × host → in-core latency dominates (build D); if ≤ 0.25 × host → memory dominates (build M only) | D vs M order |
| `GGML_Q3K_PF=1` (`prfm pldl1keep` next rows inside x4) | S without layout, all slots | **−0.5 to −3 ms**; < 1 ms or positive → S is not a lever and the re-laid format loses its second reason | S |
| `GGML_EXPERT_ANON=16` (separate rows, warm) | M alone, h ≈ 0.70 | quiet day **−1 to −7 ms (median −3)**, minor faults −60 %, major −70 %; pressured day (MemAvailable < 1.5 GB) −5 to −15; if tok/s *falls* with anon +648 MB → too much for the day, retry N = 8 (324 MB, h ≈ 0.50) | M, and the N for this phone |
| `GGML_Q3K_ILP=1` (if Tom builds it) | D | host −15 to −30 % of compute; **device −10 to −25 % (−4 to −11 ms) if latency-bound, ≤ −3 % if memory-bound** | the generalizing lever |

Go/no-go for S5(ii) (re-laid format + cache) by Clyde's rule: PF (−0.5..−3) + ANON (−1..−7) + 0.23 × unpack
(−2.5) = −4 to −12.5, median ≈ −7 → **likely no-go**; the format is built only if ANON + PF + ILP leave ≥ 8 ms
on the table *and* the S5(i) stride test shows the tile stream matters on the M4.

**4. LRU replay (host, `lab/logs/r21/lru-replay.txt`; per-layer cache of N experts, r16 traces; one expert =
1.013 MB, all three matrices cached together, so byte hit rate = identity hit rate).**

| policy | N | after 64 tok | after 256 | after 512-681 (4 texts concatenated, topic switches) | steady state (last 64 tok of each text) | copies per token |
|---|---|---|---|---|---|---|
| pure LRU (insert on every miss) | 8 | 48.9 % | 49.2 % | 50.0 / 49.6 % | 47-53 % | ~75 — infeasible |
| pure LRU | **16** | **68.1 %** | **70.4 %** | **70.8 / 69.9 %** | **68-77 %** | ~70 — infeasible |
| pure LRU | 24 | 78.7 % | 82.9 % | 82.1 / 81.2 % | 79-86 % | ~50 — infeasible |
| window-LFU, ≤ 4 promotions/token (4 MB memcpy) | 16 | fills in ~160 tok (640 copies) | 56 % overall on the 681-tok session | — | **66-74 %** (35 % on the 84-token official prompt: never fills) | 3.7 |
| window-LFU, ≤ 8/token | 16 | — | 62 % | — | 70-79 % | 6.5 |
| window-LFU, ≤ 4/token | 8 | — | 42 % | — | 45-57 % | 3.5 |
| window-LFU, ≤ 4/token | 24 | — | 63 % | — | 68-81 % | 3.8 |

Pure LRU at N = 16 holds ~70 % from the 64th token on and survives topic switches (the within-text 70-80 % and the
pooled 54.5 % bracket it; recency does the work — R16's 45-50 % repeat from the previous token). But it would copy
~70 MB per token. With a 4 MB/token promotion budget the steady state is the same 66-74 %, **the fill takes ~160
tokens, so a 32-token benchmark on a cold cache sees ~0 %**: the cache must live across requests and fill during
prompt processing. Memory-term planning number: **h = 0.70 at N = 16 (steady), 0.50 at N = 8, 0.80 at N = 24.**

## 2. What was measured today (host, raw in `lab/logs/r21/`)

| probe | binary / knob | result |
|---|---|---|
| uniform k = 3/4/5/6/7, wiki test 60 ch. | v0.8.0 `--override-kv granitehybrid.expert_used_count` | 12.1213 / 10.5351 / 9.9943 / 9.8702 / 9.8728 |
| per-layer split, k = 4 (L0-19) + 6 (L20-39) and the reverse | opt build `LLAMA_EXPERTS_PER_LAYER` (uniform 6 reproduces 9.8702) | 10.0625 (+1.95 %) / 10.2289 (+3.63 %) |
| expert skip t = 0.085/0.09/0.095/0.10/0.12 | tiny build `GGML_EXPERT_SKIP` | 9.8707 / 9.8597 / 9.9021 / 9.9158 / 10.0554 |
| router-weight CDF, 9 thresholds × 2 prompts | tiny `llama-cli` + `GGML_CPU_PROF=1` counter | table in §1(a) |
| target vs 350M draft logits, 30 ch. | v0.8.0 `--save-all-logits` ×2, `--kl-divergence`, `lab/draft_gate.py` | §1(d) table; stock KLD cross-check matches |
| layer-drop anchor | v0.8.0, old q4_0 file, en.txt 6 ch. | 10.5094 |
| build parity | tiny / v0.8.0 / opt on the same 60 chunks | 9.8702 all three; `-t 6` = `-t 8` per chunk |

Compute: ~45 min wall on two 6-thread queues (≈ 1.4 h CPU). The two 1.5 GB logits dumps were deleted after
analysis (disk at 97 %); regenerate with the two `--save-all-logits` commands in `lab/logs/r21/*-dump.log`.
`lab/draft_gate.py` recovers log-probs from the tool's uint16 encoding; its target PPL (8.8433) differs from
the run's 8.8574 by 0.16 % (clamp at max−16 and quantization), fine for curves.

**Falsified today:** static per-layer expert budgets (uniform wins); "more experts where hard" (k = 7 is free
of gain); early exit / depth skipping as a route to 20 (arithmetic + sweep); emit-without-verify on this
device (arithmetic independent of the curve). **Confirmed:** the router's weight is a usable, text-independent
signal for which slots to drop, ~3× more PPL-efficient than uniform k.

## 3. Specs for Tom (lab only, each behind its own env var, default off, bit-identical when off)

**S1 — expert count per token from the router's weights, two rules + a counter that works in PPL runs.**
Where: `lab/llama.cpp-tiny/ggml/src/ggml-cpu/ggml-cpu.c`, `ggml_cpu_skip_prepare()` (≈ line 1588): it already
has the normalized weights `w` ([1, n_ids, n_tok] F32, from `ggml_cpu_moe_weights`) and writes
`ggml_cpu_skip_mask_tl[tok*n_ids + s]`. Add, read next to `GGML_EXPERT_SKIP` (≈ line 4230):
- `GGML_EXPERT_TOPP=<permille>`: per token, sort the n_ids weights descending (n_ids ≤ 8: insertion sort),
  keep the shortest prefix whose cumulative weight ≥ p/1000, mask the rest; never mask the top slot.
- `GGML_EXPERT_GAP=<permille>`: mask slot s if w_s < (g/1000) × w_top. (Lu et al. 2402.14800 style.)
- `GGML_EXPERT_MINK=<k>` (default 1): floor on kept slots for both rules.
- Counter: `ggml_cpu_skip_cnt/total` are only printed by the one-token profiler. Print them once at exit
  (`atexit`, stderr: `expert-skip: skipped=N routed=M share=%.3f`) whenever any skip knob is on, so a
  `llama-perplexity` run reports the average experts per token on the *scored* text.
Verify: `GGML_EXPERT_TOPP=1000` and `GGML_EXPERT_GAP=0` reproduce 5.5137 / 7.3175 / 7.0292 (chunks 1-3,
`-ub 1` and `-ub 32`); `GGML_EXPERT_SKIP=100` still gives 9.9158 on 60 chunks. Then the sweep Bonnie wants,
60 chunks each (≈ 3 min at `-t 8`): TOPP 950 / 900 / 850 / 800 / 750; GAP 300 / 400 / 500 / 600; log PPL and
the printed share → `lab/logs/r21/topp-*.log`, `gap-*.log`. The question it answers: at k̄ ≈ 4.2 and ≈ 3.5,
does cumulative mass or relative gap beat the plain threshold (+1.88 % at k̄ 4.2)?
Device side later: the kit's A/B already takes `EXPERT_SKIP:0:<t>`; add `EXPERT_TOPP` the same way
(`GGML_CPU_AB` parser ≈ line 4271).

**S2 — exit-at-L predictor (falsifier for depth; 15 lines).** Where: `lab/llama.cpp-tiny/src/models/granite-hybrid.cpp`,
layer loop `for (int il = 0; il < n_layer; ++il)` (line 155). Env `LLAMA_EXIT_LAYER=L`: loop to `min(L, n_layer)`
and move the `inp_out_ids` get_rows to the last *executed* layer; `output_norm` + lm_head run unchanged on
`cur`. Run `llama-perplexity --save-all-logits` for L = 36, 38, 39 (30 chunks each) and feed each with the
full-depth dump to `python3 -I lab/draft_gate.py full.logits exitL.logits` — table [A] is then "exit-L
confidence vs agreement with the full model", [B] the coverage. Verify: `LLAMA_EXIT_LAYER=40` reproduces
9.8702 (60 ch.). Prediction (pre-registered): agreement < 50 % at L = 36, < 80 % at L = 39; coverage at
agreement ≥ 99 % below 3 % of tokens. If so, depth is closed for good.

**S3 — confidence-gated emission: do not build on this device.** The host curve is in §1(d); the device
arithmetic is negative at any coverage because the target must ingest every token. Only if a free draft
lane exists (not on SD888): llamadart's `draftSimple` loop would, for draft p_max ≥ τ (τ from the table,
0.98-0.99), append the token to both contexts, decode it in the target *without* the lm_head
(`llama_set_causal_attn`-style skip is not enough; needs a graph flag to drop `result_output` for that
ubatch) and continue; verify never. Quality rule = mixed-policy PPL ≤ +0.5 % on wikitext-2 (the harness is
`lab/draft_gate.py` on two dumps, already there).

**S4 — reuse falsifier (10 lines).** Where: `ggml-cpu.c` expert-log site (≈ line 3673, `ggml_cpu_expert_log &&
ids->ne[1] == 1`). `GGML_EXPERT_LOG=2`: additionally print `xcos <tensor> <cos(src1_t, src1_{t-1})>` for the
gate op of each layer (keep the previous token's quantized/raw `src1` row per layer in a static buffer,
1536 floats × 40). Run the story prompt 72 tokens; report the per-layer median. Prediction: < 0.8 in every
layer → expert-output reuse is closed without a PPL run.

**S5 — expert path, re-ordered after (e″)/(e‴): memory first, kernel ILP second, format last.**
(0) Kit r3 probes (Tom, each env-gated, bit-identical): `GGML_Q3K_PF=1` (`prfm pldl1keep` of the next 4 rows'
blocks inside the x4 loop); `GGML_EXPERT_ANON=<N>` (hottest N experts per layer `memcpy`'d into anon memory in the
existing layout, pointer redirect in `MUL_MAT_ID`, window-LFU over 64 tokens with ≤ 4 promotions per token,
fill allowed during prompt processing, cache persists across requests, `MADV_HUGEPAGE` if THP is on; log
`anon=N/hit%/copies` in the `cpu-prof` line and `minflt`/`majflt` per token in the kit JSON; separate rows, not
A/B); `GGML_Q3K_ILP=1` (`ggml_q3k_half` with two independent int32 accumulators per half and the 4 rows' chains
interleaved, scale unpack vectorized: the int32 sub-block sums are exact in any order, the float step
`sum += d · vaddvq(visum)` per block per row stays → bit-identical; verify chunks 1-3 at `-ub 1`/`-ub 32` and a
72-token greedy generation byte-identical); kit A/B `knob:Q3K_VEC:0:1`. Predictions in (e‴) ¶3.
(i) Microbench, host, no model: a 60-line `lab/relayout_bench.c` that takes one expert's gate matrix (512 × 1536
Q3_K, 1.01 MB; dump it once from D6 with `gguf-py` into `lab/logs/r21/expert.bin`) and one Q8_K activation row,
times 1,000 × the x4 dot over all 512 rows for: (a) rows at the real stride vs (b) the same rows packed
contiguously (sizes S on the host), (c) the ILP variant, (d) a candidate re-laid format (4-row interleave as
`Q3K_X4` walks it, 6-bit sub-block scales pre-expanded to int8, 2-bit and 1-bit planes de-interleaved per
4-row group; same byte count), asserting every row bit-identical, printing µs per MB at 1 thread. Go for (ii)
only if (d) beats (c) by ≥ 1.3× *and* (b) beats (a) by ≥ 1.2× — otherwise the format has no job the kernel and
the anon copy do not already do. Log to `lab/logs/r21/relayout-bench.txt`.
(ii) Only on a go: `GGML_EXPERT_RELAYOUT=<N>` (lab, default 0) in `ggml-cpu.c`: per layer a use counter over
the routed ids (the `eid` log site, ≈ line 3673, already sees them), after `GGML_EXPERT_RELAYOUT_WARM=32`
tokens re-lay the top-N experts' gate/up/down into anon buffers (3 × N × 40 × 0.34 MB), refresh the set every
256 tokens (evict the coldest, re-lay the hottest; one expert per token at most so the copy cost stays
≤ 1 ms); `MUL_MAT_ID` dispatch per expert through a `(layer, expert) → ptr` table: re-laid → new dot,
else stock. Log in the `cpu-prof` line: `relay=N/hit%` (hits over routed slots) and the anon bytes.
Verify as in §1(e′) ¶4; then host A/B `GGML_CPU_AB=RELAYOUT:0:16` on the story and `-t 1` `cpu-prof` to report
`MUL_MAT_ID` µs/token vs today's 48.8 ms (prediction from the microbench ratio × hit rate). Device: kit r3
`RELAYOUT:0:16` A/B with `relay=` and `anon` in the JSON; pre-registered prediction from the §1(e′) table
(conservative column) before the run.

## 4. Where this leaves 20 tok/s

(Corrected after R22 and the owner's budget: the k̄ in this section were estimates; measured k̄ = 4.71 at
t = 0.100 and 3.86 at 0.120, and the approved speed-rule budget is +2 % *full set* (≤ 9.797), which lands at
t ≈ 0.110 and k̄ ≈ 4.3 on wikitext — see §5 for the re-stated band. The text below is kept as written.)

Within +0.5 %: the compute-less families buy ≈ 6-8 ms (one slot on ~15 % of tokens): 13-15 → 14.5-16.5 tok/s,
estimated. 20 is reachable only as **raw levers to ~16 tok/s (the bounded re-laid expert cache of §1(e′),
remaining waits) plus the router-weight rule at k̄ ≈ 3.5-4.5, which costs ≈ +1 to +5 % PPL on wikitext-2 (stays
under the +10 % hard limit; the +0.5 % working tolerance does not survive).** After (e″)/(e‴) the
bounded expert cache is worth −3.5 to −12 ms at profile scale (median ≈ −6; memory term dominant and
day-dependent), so N = 16 + k̄ 4.2 (+1.9 % screen) is **15.6-20.4 tok/s** at best-run scale — 20 only in the top
corner — and reaching it needs k̄ ≈ 4.2 **plus two of** {anon expert cache, ILP Q3_K kernel (device-independent),
the remaining raw waits}. Kit r3 (S5(0)) measures each term separately before anything larger is built. That is the number the owner has to
decide with. The method itself — one threshold on the router's own softmax — is model- and
device-independent; the price is model-dependent and must be measured per model the way it was today
(two 3-minute curves).

Round 3 proposal: ship `GGML_EXPERT_SKIP` as a user-visible "speed vs quality" knob only after the owner
sets the budget; Tom's S1 sweep decides threshold vs top-p vs gap at k̄ 4.2 and 3.5 before any device run;
pre-register the device prediction from R18's one skip data point (8 % skipped → −9.1 ms, better than
proportional) against the proportional estimate above.

## 5. R22 fold-in and kit r3 pre-registration (2026-11-02 evening)

All numbers **host** (R22, `lab/logs/r22/`, 60 wikitext-2 chunks, base 9.8702, `-t 6`) unless marked
*estimated* (device predictions). Nothing new was run for this section; the device predictions are
pre-registered before kit r3.

### 5.1 What R22 measured, per slot saved

k̄ is now counted on the scored text itself (Tom's `expert-budget` counter), not inferred from a generation CDF.
Wikitext drops more slots at a given threshold than the two prompts did (21.5 % vs 15-18 % at t = 0.100; 35.7 %
vs ~30 % at 0.120): **the bench prompt will save fewer slots than wikitext at the same threshold** (§5.3(a)).

| rule, setting | k̄ | slots saved | Δ PPL (screen) | **% per slot saved** |
|---|---|---|---|---|
| threshold SKIP 100 | 4.71 | 1.29 | +0.46 % | **0.36** |
| threshold SKIP 120 | 3.86 | 2.14 | +1.88 % | **0.88** |
| cumulative TOPP 850 / 800 / 750 | 4.80 / 4.42 / 3.94 | 1.20 / 1.58 / 2.06 | +0.92 / +1.10 / +2.33 % | 0.77 / 0.70 / 1.13 |
| gap GAP 300 / 400 | 4.69 / 3.85 | 1.31 / 2.15 | +1.35 / +3.95 % | 1.03 / 1.84 |
| uniform k = 5 / 4 (R21) | 5 / 4 | 1 / 2 | +1.26 / +6.74 % | 1.26 / 3.37 |

Threshold is 2.1-2.6× cheaper per slot than top-p and 2.1-2.9× cheaper than gap at equal k̄, 3.5-3.8× cheaper than
uniform. Why, mechanically: top-p and gap are *per-token relative* rules — on a token whose six weights are flat
they drop slots that carry 0.15 of the mass; the absolute rule never drops a slot above t, whatever its
neighbours look like. Flat-routing models (Granite) are exactly where that matters. SKIP 90 (−0.11 %) and TOPP 950
(−0.14 %, k̄ 5.94) are inside paired-run noise: ~0.4-0.5 slot per token is free at any price. **Ranking unchanged**;
the rule is fixed as Clyde stated it: one absolute weight threshold per model.

### 5.2 Placing the threshold under the +2 % full-set budget (≤ 9.797)

Only one screen→full pair exists (t = 0.100: +0.46 → +0.82 %); it can be read as **additive (+0.35 pt)** or
**multiplicative (×1.78)**. Interpolating the screen curve between 100 and 120 permille (linear and exponential
in t; k̄ linear: −0.0425 per permille):

| t | screen Δ (lin / exp) | k̄ (wikitext) | full set, additive | full set, ×1.78 |
|---|---|---|---|---|
| 105 | +0.81 / +0.65 % | 4.50 | 9.717 / 9.702 | 9.744 / 9.717 |
| **110** | +1.17 / +0.93 % | **4.29** | **9.751 / 9.728** | 9.805 / 9.764 |
| 115 | +1.52 / +1.32 % | 4.07 | 9.785 / 9.766 | 9.866 / 9.831 |
| 120 | +1.88 % (measured) | 3.86 | 9.819 | 9.927 |

**Prediction (pre-registered, host):** SKIP 110 full set = **9.74 ± 0.04** (passes; fails only in the
linear-screen × multiplicative corner at 9.805); SKIP 120 full set = **9.82-9.93 (fails)**; the budget edge is at
t ≈ 115-118. Expected choice: **t = 110 permille, k̄ ≈ 4.3 on wikitext, ≈ 4.6-4.8 on the bench prompt** (R21
generation CDF: 20-24 % of slots below 0.11). Falsifiers: SKIP 120 ≤ 9.797 → the under-read shrinks with t (not
additive); SKIP 110 > 9.797 → the under-read is multiplicative or worse, fall back to 105 (k̄ 4.5). en.txt is not
the gate; predicted +3.5-5 % at 110 (it was +2.1 % at 100), report it, do not act on it.

### 5.3 Kit r3 device predictions (pre-registered; *estimated*)

Units: the kit's own µs/token at 3 threads, best-run profile scale ≈ **96 ms/token** after R20's defaults, with
`MUL_MAT_ID` compute ≈ **35 ms** (5.8 ms per routed slot at k̄ = 6) + wait ≈ 3 ms. Bench tok/s = kit ms × 0.70-0.80
(the bench's 13-15 tok/s best run ↔ 67-77 ms). R18's one skip data point (8 % of slots → −9.1 ms of 101) was 1.9×
proportional, but that excess came largely from `MUL_MAT_ID` wait that `MMID 3` has since removed (16.5 → 2.8 ms).

| probe | prediction | falsified if |
|---|---|---|
| (a) `EXPERT_SKIP 110`, A/B on the official prompt | skipped share printed by the kit **20-24 %** (k̄ 4.6-4.8); proportional **−7 to −8 ms** (22 % of 35), R18-anchored ×1.9 → −14 to −15; **central −8 to −11 ms** (wait falls too: fewer slots → less imbalance) → 96 → 85-88 ms, **+9-13 % tok/s** (bench 14.2-17 from 13-15) | < −5 ms (fixed per-op cost dominates slots) or > −16; share outside 17-27 % (router CDF text-dependent after all) |
| (b) `Q3K_ILP=1` | latency thesis: the ~4-deep `vmla` chain + `vpadd` tree is ≈ half of the 121 cycles/block; two accumulators + 4-row interleave remove ~⅓-½ of the stall → `MUL_MAT_ID` compute **−10 to −25 % = −3.5 to −9 ms** (central −6); host −15 to −30 % | **≤ −3 % (≤ −1 ms) on the device while the host shows ≥ −15 %** → the device stall is memory/stream-shape (Clyde's reading), build M/S, drop D; or host also ≤ −5 % → the chain was not the stall anywhere, the count in (e‴) ¶1 is wrong |
| (c) `Q3K_PF=1` | warm, in-core-bound kernel: **−0.5 to −2 ms (≤ 2 ms)**, central −1 | > −3 ms → the stream-shape term S is real and the re-laid format regains its second reason; > +1 ms → prefetch pollutes L1, drop it |
| (d) `EXPERT_ANON=16` (separate rows, 648 MB anon) | hit rate on the 84+32-token official run **35-45 %** (fill incomplete), steady ≈ 70 % on a ≥ 200-token run; memory term **−0.5 to −3.5 ms at h 0.35-0.45 (central −1.5)**, **−1 to −7 ms at h 0.70 (central −3)**; quiet day. `minflt` per token **−30 % (h 0.4) / −60 % (h 0.7)**, `majflt` −70 % steady; saved ms ≈ 1-2 µs × minor faults removed + 0.1-0.5 ms × major faults removed | confirms M only if **both** faults and ms fall together. Faults fall, ms does not → the term is not faults (TLB at most); neither falls → cache holds the wrong experts (check `hit%` in the prof line); tok/s *falls* → 648 MB too much for the day (R20 pattern: file cache 2.0 → 1.3 GB, faults ×14), retry N = 8 |
| (e) combination for 20 | 20 tok/s bench = 50 ms bench = **≈ 64-71 ms kit scale = −25 to −32 ms of 96**. Central sum of (a)-(d): 9.5 + 6 + 1.5 + 1 = **−18 ms → 78 ms kit → 15.5-17.5 tok/s bench**. 20 needs **two of three at their upper band**: (a) ≥ −12 ms (the R18 better-than-proportional holds) **and** (b) ≥ −8 ms (latency thesis fully true), with (d) ≥ −3; or (a) ≥ −12 and (d) ≥ −7 (a pressured-day memory rescue, which is not a quiet-day win) with (b) ≥ −4 | if (b) ≤ −3 % and (d) ≤ −2 ms on a quiet day, the kit-r3 family tops out at 16-17 tok/s and only a larger PPL budget (t ≈ 130-140, k̄ 3-3.5, +5-9 %) or the remaining raw waits reach 20 |

Honest band, restated: within the owner's +2 % budget the router rule buys **1.2-1.4 slots on the bench prompt**
(1.7 on wikitext), ≈ −8 to −11 ms kit scale. It is the largest single term but not enough alone; §1(e‴) ¶2's
"k̄ 4.2 plus two of {ILP kernel, hot cache, raw waits}" stands with k̄ = 4.3 and the sizes above.

### 5.4 The −t 4 vs −t 6 difference at −ub 1: flash-attention split-KV, not MoE or MUL_MAT_ID

Logs: `id-off-ub1.log` (−t 6) chunks 5.4397 / 7.2637 / 7.0036 vs `id-off-ub1-t4.log` 5.4327 (chunk 1); −ub 32
is 5.5121 / 7.3161 / 7.0267 and was −t 6 = −t 8 per chunk in R21. So the thread dependence exists only on the
one-query path. Code (`lab/llama.cpp-tiny/ggml/src/ggml-cpu/ops.cpp:9567`, read-only):

```
const bool use_split_kv_path = !use_ref && (neq1 == 1 && neq3 == 1) && kv_is_f32_or_f16
                               && (k->type == v->type) && q->type == GGML_TYPE_F32 && nek1 >= 512;
...
const int64_t chunk_size = (nek1 + nth - 1) / nth;   // KV split by thread count
...
ggml_flash_attn_ext_reduce_partials(params, dst, nth, chunk_size);  // online-softmax merge of nth partials
```

With one query token and n_kv ≥ 512 (`llama-kv-cache.cpp:1287` pads n_kv to a multiple of 256, so in a
`-c 512` perplexity run at −ub 1 the path is taken for the second half of every chunk — all 256 scored tokens),
each thread computes an online softmax over its own KV chunk and the partials are merged in float — the chunk
boundaries and the merge order depend on `nth`, so −t 4 ≠ −t 6 ≠ −t 8 at −ub 1, while −ub ≥ 2 takes the
per-query tiled path and is thread-independent. Ruled out by reading: `ggml_cpu_moe_weights` only *locates* the
weight tensor (no arithmetic); `RMS_NORM` / `SUM_ROWS` / softmax split by rows, one row per thread; `MUL_MAT_ID`
with `GGML_MMID 3` splits rows, never a row's K dimension. Four of the 40 layers are attention, so the effect is
small (±0.007 per chunk, well under the 0.35-pt screen→full shift) and irrelevant to generation benchmarks
(72-token greedy runs have n_kv 256 < 512, path not taken).

**1-line test for Tom:** `GGML_EXPERT_SKIP= lab/llama.cpp-tiny/build-888/bin/llama-perplexity -m lab/models/var/d6-q3k-e6.gguf -f lab/text/wikitext-2-raw/wiki.test.raw -c 512 --chunks 1 -ub 1 -fa off -t 4` vs the same at `-t 6` →
prediction: identical to the last digit (and both differ from 5.4327 / 5.4397). Consequence for identity checks:
pin `-t` (as R22 did) or add `-fa off`; the device (3 threads) will never byte-match a 6-thread host run at
−ub 1 once n_kv ≥ 512, which is not a bug in our module.

Sources consulted (direction only, no numbers borrowed): Lu et al., *Not All Experts are Equal* (2402.14800,
ACL 2024: post-hoc dynamic expert skipping by routing weight, ~1.2× on Mixtral); *SimLens* (2507.17618) and
*The Diminishing Returns of Early-Exit Decoding in Modern LLMs* (2603.23701) on training-free early exit.

## 6. R23 device results: what survived, round-4 costing (2026-11-02 night)

Inputs: `reports/device/r3_2026-11-02.json` (device, 3 threads, kit profile scale), Clyde's R23 block in
`03_final.md`, Tom's correction (the cpu-prof window was never reset at a run boundary, so every r3 A/B window
after the first carried leftover graphs of the previous run), and two host runs of mine logged under
`lab/logs/r24/`. Labels: *device* / *host* / *estimated*.

### 6.1 Scorecard of the §5.3 pre-registration

| prediction (§5.3, estimated) | measured (device) | score |
|---|---|---|
| threshold for +2 % full set = 110; edge 115-118 | SKIP 115 → 9.7771 (+1.79 %), k̄ 4.11 (host) | **hit** on the edge, 110 unneeded |
| (a) SKIP A/B −8 to −11 ms, share 20-27 % | the one uncontaminated skip window (run 7's own B tail, 24 graphs): share 27 %, k̄ 4.36, `MUL_MAT_ID` 29.8 → 21.8 (−27 %, exactly proportional to the slots), token 68.5 → 60.0 (−8.5 ms); at clean 27.4-29.5 ms → **−7.4 to −8.0 ms** | **hit, low edge**; the R18 "1.9× proportional" did not recur (its excess was wait, since removed) |
| (b) ILP device `MUL_MAT_ID` −10 to −25 %; host −15 to −30 % | device −18.0 % (ILP#1, near-clean window, but 1,210 majflt/token) and −13.3 % (ILP#2, B diluted 12/32 → true effect larger); **host +1.3 % (M4), +1.5 % cache-hot** | **device hit, host miss** (mine). The thesis "chain latency" was right about the A78 and wrong about the M4, see §6.2 |
| (c) PF −0.5 to −2 ms | +0.6 ms clean; the −3.8 was ILP leakage (16 B graphs ran ilp=1) → PF ≈ 0 or worse | **miss on the mechanism, right on the size**: the stream-shape term S is dead |
| (d) ANON 16: hit 35-45 % at 84+32 tokens, 70 % steady; −0.5 to −3.5 / −1 to −7 ms | hit 62 % in 32 tokens, 75-78 % steady (my replay exact); **−0.7 ms** at h 0.78; warm run −20 % (618 MB evicted the file cache) | **hit on the curve, miss on the value**: the memory term M is dead on this phone, as the "faults down, ms flat" branch said |
| (e) round: −18 ms → 15.5-17.5 tok/s bench; 20 needs two upper-band terms | surviving sum ILP + SKIP ≈ −11 to −12.5 ms at clean scale (not yet measured together); bench 5.4 / 11.2 on a swapping phone | **unresolved** on the bench, **miss** on the 20 condition (ANON gave 0) |
| `-fa off`: −t 4 = −t 6 at −ub 1 | 5.4696 both (host) | hit |

Honest tally: 4 hits, 2 misses (host ILP, ANON value), PF a mechanism miss, (e) unresolved. The two misses
share one cause: I priced in-core terms by op counts without pricing the issue width of each core.

### 6.2 Why the M4 saw nothing and the A78 saw −15 %: both are now at their NEON issue floor

Per-token work at k̄ = 6: 6 × 40 × 3 × (1536·512/256) = **2.21 M Q3_K row-blocks**. Clean `MUL_MAT_ID` 27.9 ms
wall with per-thread 26.3-26.5 ms (`mmid_thr`) at 2.112 GHz (the A78 clock the kit read all session) =
**75.7 cycles per row-block** (device, pre-ILP); after ILP (−15.5 %) ≈ **64**. Op count of the ILP kernel
(`arch/arm/quants.c` `ggml_q3k_half_ilp`, read): per row-block ≈ 84 unpack/mask/`sdot` + 12 `vpadd` + 4 `mul` + 8
widen + 2 `add` + ~10 scale-vector + 3 float ≈ **120 NEON-ALU ops** (loads excluded). The Cortex-A78 has **2
NEON pipes → floor ≈ 60 cycles**; the x4 kernel sat at 75.7 (stall 20 %), ILP took it to 64 (**≈ 94 % of the
floor**). The M4 P-core (4 NEON pipes, floor ≈ 30-34): the cache-hot check gives 24.5 µs per 512 × 1536 = 3,072
row-blocks → 8.0 ns = **36 cycles at 4.5 GHz** (host, `lab/logs/r24/q3k_check_pcore.txt`) — already at its
floor, which is why ILP measured +1.5 %/−0.2 % there. The (e‴) ¶1 count "~1 NEON op per cycle on both" was
right; what differed was the floor (2 vs 4 pipes), not the stall.

Consequence for a second ILP step (8 rows, gate+up interleave, split half-chains): the remaining stall is
64 − 60 ≈ 4 cycles ≈ **≤ −5 % of `MUL_MAT_ID` ≈ ≤ −1.2 ms** (estimated, clean 24 ms post-ILP); 8 rows also
doubles live registers (4 × (acc + 2 hmask + scales) = 16 of 32 vregs already) and would spill. **Predict
−0 to −5 %; not worth building.** Instruction cuts in the unpack are a wash (folding the hmask into a second
`sdot` saves 1 op/vector but adds a `vpadd` tree: 5.5 → 5.8 ops per 16 weights, counted). The Q3_K dot is
closed at ≈ 94 % of what this core can issue; the lever is now fewer blocks (k̄), not faster blocks.

Host proxy for the A78: **none usable.** `taskpolicy -c background` does land the check on the M4 E-cores
(3.8-5.9× slower, 30.5 ns/row-block) but the two runs differ by 1.5× (94 vs 145 µs: background QoS also
changes the clock) and ILP reads ×1.025 and ×0.95 — noise, `lab/logs/r24/q3k_check_ecore_bg.txt`. No
cycle counters on the phone; the kit A/B at clean memory state is the only test.

**Disagreement with R23's decision text:** "ILP −9 to −14 % per token" is the fault-heavy number
(`MUL_MAT_ID` 52-58 ms, 1,000-1,200 majflt/token). At clean state the in-core gain is bounded by the floor:
75.7 → 60-64 cycles = **−15 to −20 % of 27.4-29.5 ms = −4.1 to −5.9 ms, i.e. −6 to −9 % per token**
(estimated). Still the largest bit-exact gain, but 2-4 ms smaller than the R23 text implies.

### 6.3 Task item 3 (A/B under-application) — withdrawn

Tom's host reproduction explains the k̄ 5.33 / share 29 % / −11 % row entirely: 20 of 32 B-side graphs were
leftover unmasked graphs of run 6 (my own arithmetic had the unmasked fraction at 62 % = 20/32 before the
cause was known). There was no knob leak and no under-application; the A/B path and the always-on path give
the same k̄ (host 4.51 vs 4.49, −23 % both). Nothing for Tom here beyond the window reset he already made.

### 6.4 Round-4 costing (profile scale, 3 threads, clean memory state)

Clean baseline: 64.4-68.9 ms per token (story windows 3-8, k̄ 6, ilp 0), of which `MUL_MAT_ID` 27.4-30.8 +
0.4-0.8 wait, dense `MUL_MAT` 23.5-24.2 + 1.5-1.9, `SSM_SCAN` 3.5 + 0.1, `CPY` 2.0-2.3 + 0.7-0.9, the rest ≈ 4.5,
total wait 3.0-4.9. Kit wall = profile + ≈ 1.5 ms (15.03 tok/s = 66.5 ms vs 65.0 profiled). Compounding on
`MUL_MAT_ID` only (the skip removes blocks, ILP speeds the remaining ones):

| from 65.0 / 68.0 ms | ILP (−9 / −12.5 / −16 %) | + SKIP 115 (share 25 / 28 / 30 %) | token (profile) | wall | **tok/s, story** |
|---|---|---|---|---|---|
| low | 27.9 → 25.4 | → 19.0 | 56.1 | 57.6 | 17.3 (16.7 from 68) |
| central | 27.9 → 24.4 | → 17.6 | 54.7 | 56.2 | **17.8** (17.2) |
| high | 27.9 → 23.4 | → 16.4 | 53.5 | 55.0 | 18.2 (17.6) |

(ILP band here is the conservative −9 to −16 %; the §6.2 floor argument allows up to −20 %, which adds ≈ 1 ms.)

**Pre-registered round-4 predictions (estimated):**
- *ILP alone, A/B at clean state (both sides ≤ 20 majflt/token):* `MUL_MAT_ID` −12 to −20 %, token **−3.5 to
  −6 ms** (−5.5 to −9 %). Falsified below −2.5 ms (then the device gain was a fault-state artefact: keep the
  default, it is bit-exact, but drop the kernel from the 20-tok/s arithmetic) or above −7 ms (then the floor
  model is wrong and the second ILP step is live after all).
- *ILP + SKIP 115 vs ILP alone, A/B:* share 25-30 % on the story prompt, k̄ = 6 × (1 − share) ± 0.05 (the
  reset fixed), `MUL_MAT_ID` −(share ± 3 pt) %, token **−5.5 to −7.5 ms**. Falsified below −4.5 with share ≥ 25 %.
- *Combined always-on row (ILP=1, SKIP=115, story 200, windows 2+):* **54-58 ms profile → 17.0-18.5 tok/s**
  story (central 17.8); `MUL_MAT_ID` 16.5-19.5 ms. Falsified outside 52-60 ms.
- *Official 32:* the r2/r3 official-to-story ratio was 0.37-0.78 (5.4 and 11.2 vs 14.4-15.0 on the same
  phone): the 155-token prompt is prefilled with no cached prefix and touches ~all 64 experts of every layer
  (≈ 2.6 GB) while MemAvailable was 1.5-2.8 GB, so decode starts with the cache half-evicted (250-1,200
  majflt/token). Prediction: official **7-14 tok/s on an r3-like day, 15.5-18 on an R13-like day**; it will
  not read 20 even when the story does. "What the user sees" in that bench is the phone's page cache more
  than the runtime; the only runtime lever on it is fewer bytes per token (SKIP cuts expert bytes 28 % →
  fewer re-faults), and the honest fix is to report both the story and the official number.

### 6.5 What else is left, ranked (clean scale, per printed token)

| # | lever | ms (estimated) | quality | cheapest falsifier |
|---|---|---|---|---|
| 1 | SKIP 115 always-on (k̄ ≈ 4.3 on prompts) | **−6 to −8** | +1.79 % full set (approved) | round-4 A/B above |
| 2 | ILP default | −3.5 to −6 | bit-exact | round-4 clean A/B |
| 3 | SSM state round trip: `SSM_SCAN` 3.5 + `CPY` 2.0 + 0.8 wait ≈ 113 MB r/w of F32 state per token (36 × 3072 × 128 × 4 B × 2) + conv/state copies; write the scan output in place and drop the state `CPY` nodes (graph change in the runtime, same float ops) | **−1.5 to −2.5** | bit-exact if the op order is kept | host: `GGML_CPU_PROF` CPY count 144 → ≤ 72, PPL identity |
| 4 | dense `MUL_MAT` waits 1.5-1.9 (265 ops, X1/A78 tail) + CPY/RMS waits ≈ 1 | −0.8 to −1.5 | bit-exact | prof `wait` per op |
| 5 | second ILP step (8 rows / gate+up / half-chains) | −0 to −1.2 | bit-exact | §6.2: kernel at 94 % of the 2-pipe floor; skip it |
| 6 | dense `MUL_MAT` 23.6 ms: 559 MB/token (Q4_0 412 — `ssm_in`/`ssm_out` alone 296 MB = 12.5 ms —, Q6_K lm head 126 MB via the tied `token_embd` ≈ 5.3 ms, Q4_1 5, F32 routers 16) = **23.7 GB/s = the bandwidth ridge** (device). No ILP lever; bytes can only fall by changing numerics (lm-head vocabulary subset, F16 SSM state, re-quantized head at load) | 0 within the rules as a runtime-only, exact change | — | gguf-py dump above; `MUL_MAT` wait 1.6 confirms memory-bound |
| — | expert bytes 243 MB/token at 8.7 GB/s: compute-bound, as established | | | |

### 6.6 Verdict on 20 tok/s printed

With everything that survives (1 + 2 + 3 + 4) the clean-state story band is **55-58 ms profile → 17-18.5
tok/s central, 19.5 at the top corner** (estimated); 20 tok/s needs ≤ 50 ms profile = −15 ms from 65, and the
ranked sum is −12 to −18 with the central at −13.5. **20 printed is at the edge of the band inside the +2 %
budget, not its centre; it is not reachable on the official 32-token bench on an r2/r3-like day at all.**
Moving it to the centre needs the owner's say: SKIP 130 (k̄ 3.44 wikitext, screen +3.4 %, full set ≈ +3.8 to
+4.5 % estimated, inside the +20 % hard limit) adds ≈ −3 ms → 51-54 ms → 18.5-19.6; SKIP 140 (k̄ 3.06, ≈ +5.5 to
+6.5 %) → ≈ 50 ms → 20.0. Those two are the only numbers on the table that reach 20 from here; everything
exact is now priced and short by ≈ 3 ms.

## 7. Round-4 scorecard and round-5 costing (2026-11-03)

Inputs: `reports/device/r4_2026-11-03.json` (device, 3 threads, kit profile scale, swapping day: 18-1,390 majflt
and 390-1,410 minflt per token), `reports/device/r3_2026-11-02.json` run 8 ("story 200 · anon 0": 8 majflt / 67
minflt per token, file cache 1.97 GB — the one clean-day story we own), Clyde's "R24 — round-4 device results",
and my re-analysis log `lab/logs/r25/bonnie_r4_reanalysis.txt` (all arithmetic below is reproducible from it).
Labels: *device* / *host* / *estimated*.

### 7.1 Scorecard of the §6.4 pre-registration

| §6.4 item (estimated, clean state) | measured (device) | score | what the miss teaches |
|---|---|---|---|
| ILP A/B at clean state: `MUL_MAT_ID` −12 to −20 %, token −3.5 to −6 | A 107.0 → B 104.4 with ILP *off* on B; 1,388 majflt/token, file cache 23 MB at start (the day's dirtiest run) | **unscoreable**: the clean precondition (≤ 20 majflt/token both sides) failed by 70×, and the design is order-biased (§7.2) | the kit cannot test a −2 to −5 ms knob with one order on a swapping day; the cost model was not falsified, the instrument was |
| ILP + SKIP 115 vs ILP, A/B: share 25-30 %, k̄ = 6(1 − share) ± 0.05, token −5.5 to −7.5 | share 27 %, k̄ 4.33 (6 × 0.73 = 4.38 ✓); token −15.9 / −20.4; `MUL_MAT_ID` −25.5 / −31.5 % | **hit on routing** (share, k̄), **unscoreable on ms**: the band assumed `MUL_MAT_ID` 28 ms on side A, it was 56-60 (fault-inflated ×2) | the skip's saving is proportional to the *whole* A-side `MUL_MAT_ID`, faults included — the lever is larger on bad days, not smaller (§7.2) |
| 130 / 140 ∝ share ± 20 pt (43 / 49 %) | share 39 / 46 %, k̄ 3.61 / 3.20 (6(1−s) = 3.66 / 3.24 ✓); `MUL_MAT_ID` −33.6 / −39.8 % | **hit** (shares 4 and 3 pt under my screen numbers: prompts route flatter than wikitext, §5.1) | — |
| always-on story 200: 54-58 ms profile, 17-18.5 tok/s (falsified outside 52-60) | steady windows 5-6 **60.8**, printed **14.2** | **miss**, narrow on profile, wide on printed: the pre-registration priced the steady profile and said nothing about warm-up or the kit's own 1.4 ms/token | printed ≠ steady: on this day 7.9 ms/token of warm-up + 1.4 ms/token non-graph overhead sit between them (§7.3) |
| official 32: 7-14 on an r3-like day | 11.3 (baselines 7.1 / 5.3) | hit | — |
| SSM round trip −1.5 to −2.5 (falsified < −0.8) | own ops −4.5 (`CPY` −3.3, `SSM_SCAN` −1.2) | **hit, under-priced ×2**: I priced 113 MB of state traffic at the dense-weight bandwidth; the state `CPY` was running at ≈ 1/3 of that (2.85 + 1.03 wait for 57 MB = 15 GB/s with 3 threads and a barrier per layer) | small strided copies pay per-op cost, not bandwidth; price them from the profile, not from bytes |
| waits −0.8 to −1.5, no new lever | not attempted in round 4 | open | — |

Tally: 3 hits, 1 miss (story band), 2 unscoreable (ILP, skip ms), 1 open. The one real miss is a
printed-vs-profile accounting gap, now closed in §7.3.

### 7.2 Where I disagree with R24: the 0.87 coefficient and the order-bias solve

**The "0.87 × share" is a mixture, not a coefficient.** The four skip pairs do not share one ratio: 0.95, 1.17,
0.86, 0.86 (115#1, 115#2, 130, 140). A two-term model fits all five A/B pairs (four skip + the SSM pair, whose
knob never touches `MUL_MAT_ID`) within 1-2 ms:

    MUL_MAT_ID(side A, window 1) = C + g₁          MUL_MAT_ID(side B, window 2) = (1 − share)·C + r·g₁

with C = compute at k = 6 and g = the day's fault/mapping term, which *does not scale with k̄* within a run (it is
the stream of pages the kernel re-faults as the story moves on). Least squares (*device*, log): **C = 30.0 ms,
r = 0.71**, predicted B = 40.1 / 43.4 / 37.3 / 25.1 / 49.5 against 41.4 / 41.3 / 37.7 / 25.6 / 49.9. With C forced
to the clean value 26.5-27.5 the fit is barely worse (rss 7.5-8.3 vs 6.8) and r stays 0.71. So: the compute cut is
**1.0 × share of C** (as the slots say it must be), and every pair also carries a bias g₁ − g₂ = 0.29 g₁ ≈ 7-9 ms
(3.6 in the warm 140 run). Dividing (share·C + 0.29 g₁) by (C + g₁) is what produces "0.87". Clean-state
coefficient from the only order-free measurement we have: **0.92 on the host** (§6.3: −23 % at share 25 %), i.e. a
fixed per-op part F ≈ 8 % ≈ 2.2 ms at k = 6 (dispatch, Q8 quantisation of the 1536-wide activation, chunk tail
with 13-18 chunks over 3 threads). I use 0.92 (F = 2.2) as central and 1.0 as the low edge in §7.4; the
difference is worth 0.3-1.0 ms depending on threshold, and the ABBA kit (§7.5) measures it directly.

**Clyde's bias solve: right size, wrong confidence.** The solve assumes the same bias b in R23's two `0:1` runs
(1,015-1,210 majflt/token) and R24's `1:0` run (1,388 majflt/token, model fully uncached at start). The pair model
gives b for the R24 run as 7.8 ms = 13.7 % of side A, not 9.7 %; putting 13.7 into the two equations gives x = −10 %
(R24) and −1.9 % (R23) — the two days do not agree on b, so x is known only to **−2 to −19 % of C** (the fit with
C = 27.5 says ILP off cost +6.4 ms on B after bias removal = −23 % of C, which would make R23's −18 % honest, but
that is one filthy run). My position: ILP is **−1 to −5 ms clean, central −2.5 (−9 % of 27.5)**, bit-exact, stays
default, and is the first thing the ABBA kit must settle. On the "two-thirds bias" reading of R23: possible, not
established — the bias in the near-clean r3 PF runs (34-250 majflt/token, 167 minflt/token) was 3-11 % of
`MUL_MAT_ID`, so the bias is set by *minor* faults (pages in cache but unmapped) more than by major ones.

**What the window-1 penalty is made of** (*device*, story-200 skip run, window 1 vs steady, k̄ equal): `MUL_MAT_ID`
+18.2 ms (+86 %), dense `MUL_MAT` +2.5 (+10 %), `SSM_SCAN` +0.5, `CPY` +0.2, waits +5.0, everything else < 0.3.
A clock ramp would hit every op by the same factor; +86 % on the op that touches 2.2 GB of fresh expert pages in
its first 32 tokens vs +10 % on the op that streams the same 560 MB every token is page mapping, not DVFS.
Arithmetic: 32 tokens × 4.3 experts × 40 layers touch ≈ 56 of 62 experts per layer ≈ 2.2 GB ≈ 550k pages; at a
16-page fault-around that is ≈ 34k minor faults ≈ 1,070/token — the kit measured 1,176-1,410 minflt/token on the
A/B runs. At 18 ms/token the mapping cost is ≈ 17 µs per fault-around ≈ 1 µs/page (*estimated*). DVFS is at most
the uniform part: ≤ 10 % × window 1 ≈ ≤ 2.5 ms/token for 32 tokens ≈ **≤ 0.4 ms/token over 200** — and on the
clean day (r3 run 8) window 1 was 68.5 vs steady 67.6: **no measurable warm-up at all** when the pages are mapped.

### 7.3 Corrected steady-state cost model, 3 threads (per op, ms/token)

| op | clean k 6, ILP off (*device*, r3 run 8 w3-8) | r4 control steady w4-6 (*device*) | r4 skip 115 steady w5-6 (*device*) | **round-5 default clean, k 6** (*estimated*) | **+ SKIP 115** (k̄ 4.25) | **+ SKIP 130** (3.61) | **+ SKIP 140** (3.20) |
|---|---|---|---|---|---|---|---|
| `MUL_MAT_ID` + wait | 29.3 + 0.6 | 36.9 + 2.8 | 21.1 + 0.7 | 26.7 + 0.6 | 19.6 + 0.5 | 16.9 + 0.5 | 15.3 + 0.5 |
| dense `MUL_MAT` + wait | 24.0 + 1.7 | 25.6 + 2.5 | 24.4 + 2.0 | 24.0 + 1.7 | same | same | same |
| `SSM_SCAN` + wait | 3.5 + 0.2 | 3.6 + 1.2 | 3.4 + 0.5 | 2.4 + 0.1 | same | same | same |
| `CPY` + wait | 2.1 + 0.8 | 2.6 + 0.9 | 2.3 + 0.9 | 0.2 + 0.3 | same | same | same |
| small ops (MUL, GET_ROWS, ADD, CONCAT, GLU, SSM_CONV, RMS_NORM, FA, UNARY, ARGSORT) + wait | 4.4 + 0.4 | 4.7 + 0.6 | 4.4 + 0.6 | 4.4 + 0.4 | same | same | same |
| **profile total** | **67.6** | **82.1** | **60.8** | **61.4** | **54.2** | **51.5** | **49.9** |

How the estimated columns are built: ILP −9 % ± 5 of 29.3 → 26.7 (±1.5); `SSM_INPLACE` at clean state −3.6
(the r4 −4.5 was measured with the `CPY` fault-inflated: 2.85 vs clean 2.1; `SSM_SCAN` −1.1, `CPY` −1.9, waits
−0.6); skip = F + (26.7 − F)(1 − share) with F = 2.2 and share from the measured k̄ on story prompts (115: 4.09-4.35
→ 0.29; 130: 3.61; 140: 3.20); the fault term g is 0 by definition of "clean". Same model on an **r4-like day**:
take the measured skip-115 steady 60.8 and subtract only the exact `SSM_INPLACE` (−4.5, its ops do not depend on
k̄) → **56.3**; for 130 / 140 scale the measured 21.1 by k̄ (the steady fault term on the skip side was ≈ 0.6-3 ms,
small enough to scale with it) → 18.6 / 16.5 → **53.8 / 51.7**. Clean-day sanity check against r4: control steady
82.1 vs clean 67.6 + ILP − … → the day's fault tax at k = 6 is ≈ 13-15 ms on the steady state, ≈ 3-5 ms at k̄ 4.1
(21.1 measured vs 19.6 + F-model 19.6); the skip's super-linear printed gain (×1.40 at −29 % slots; **withdrawn in R25**: it was the control's bad page-cache day, the in-run ABBA gives ×1.12-1.15) is this
coupling, and it is real for the user on bad days.

### 7.4 What reaches 20 tok/s printed (200-token story, `_benchPrompt`-style greedy)

Printed ms/token = steady profile + warm-up excess + 1.4 (kit/sampling/stream overhead, *device*: decode wall
minus profiled windows, 1.41 skip / 1.33 control). Warm-up excess (Σ (window − steady) × 32 / 200, *device*):
r4 control **15.2**, r4 skip 115 **7.9**, clean day (r3 run 8) **≈ 0.15**. On an r4-like day the excess scales
with the expert bytes of the first window: linear in slots, 130 → ≈ 6.0, 140 → ≈ 4.8 (*estimated*).

| threshold (PPL) | clean day steady → **printed** | r4-like day steady → **printed** | with waits −1.0 and F = 0 (best corner), clean |
|---|---|---|---|
| 115 (+1.79 %, approved) | 54.2 → 55.8 ms = **17.9 tok/s** | 56.3 → 65.6 = **15.2** | 52.6 → 54.2 = 18.5 |
| 130 (+3.60 % as written; +3.67 % over 9.6051, corrected in R25; owner's call) | 51.5 → 53.1 = **18.8** | 53.8 → 61.2 = **16.3** | 49.6 → 51.2 = 19.5 |
| 140 (+5.20 % as written; +5.25 % over 9.6051, corrected in R25; owner's call) | 49.9 → 51.5 = **19.4** | 51.7 → 57.9 = **17.3** | 47.9 → 49.5 = **20.2** |
| control k 6 (round-5 defaults) | 61.4 → 63.0 = 15.9 | 77.6 → 94.2 = 10.6 | — |

Reading: **20 printed needs all three of a clean day, SKIP 140 and the waits lever (or F ≈ 0)**; 130 tops at
19.5 in its best corner; 115 (the approved budget) is 17.5-18.5 clean and ≈ 15 on an r4-like day. The
swapping day costs 2.6-3.1 tok/s at every threshold and no runtime lever recovers it except fewer bytes per
token. Those are the numbers for the owner's threshold decision; I present 130/140 as measured options inside
the +20 % hard limit and outside the +2 % budget, not as a recommendation.

Levers left, priced (*estimated*, clean steady, per printed token):

| # | lever | ms | exact? | cheapest falsifier |
|---|---|---|---|---|
| 1 | threshold 130 / 140 (owner) | −2.7 / −4.3 vs 115 | no (+3.60 / +5.20 % full set, host) | round-5 always-on rows |
| 2 | `SSM_INPLACE` default (done) | −3.6 to −4.5 | yes | ABBA `SSM_INPLACE:1:0` |
| 3 | waits: dense `MUL_MAT` tail 1.7 (thread 0 runs 6-10 % slower than 1-2 in every `mmid_thr` triple → end-of-op chunk stealing), `MUL_MAT_ID` 0.5, small-op waits 0.4 | **−0.8 to −1.5** | yes | prof `wait` per op; `mmid_thr` spread < 3 % |
| 4 | conv state in place (ring buffer): drop `CONCAT` 36 ops 0.59 + its barriers; `SSM_CONV` reads the ring | −0.5 to −0.7 | yes if the conv sum order is kept | host prof `CONCAT` 36 → 0, PPL identity |
| 5 | fuse `RMS_NORM` + weight `MUL` (76 of the 117 norms; ggml's CUDA path already fuses) | −0.3 to −0.5 | yes if kept as two multiplies in the same order | host prof `MUL` 76 → 40, PPL identity |
| 6 | first-window penalty on a swapping day: pre-fault expert pages (`MADV_WILLNEED`/toucher thread on the A55s) | 0 on a clean day (nothing to map); **0 to negative on an r4-like day** — the model (3.1 GB) does not fit the cache the phone had (1.4-2.0 GB available), so readahead evicts what the current layer needs; `mlock`/`MAP_POPULATE` of 2.5 GB is the OOM of `granite-phone-oom` | — | not worth a build; the falsifier is already in hand: clean window 1 = steady |
| 7 | keep the big cores off 710 MHz between prompts (spin/`sched` hint) | ≤ −0.4 over 200 tokens, ≤ 1 token-equivalent per prompt; costs battery and heat (threshold 0.93) | — | per-block MHz log (§7.5) — if the clock is at 2,112 from block 1 it is 0 |
| 8 | expert page touch order within a layer | 0: fault-around is already sequential within an expert; file THP is off on this kernel (`thp:` empty) | — | — |

Sum of the exact levers not yet built (3 + 4 + 5): −1.6 to −2.7 ms ≈ +0.6 to +1.0 tok/s. Nothing exact above 1.5
ms remains; dense `MUL_MAT` is at the bandwidth ridge (R24).

### 7.5 Round-5 kit design

**Choose ABBA blocks within one run, not both orders per knob.** Both-orders needs the bias to be equal across two
runs; r4's two `115` runs had 18 vs 534 majflt/token and 4.5 ms of raw Δ between them — the assumption fails on
the day it matters. Within-run alternation shares the memory state; **ABBA (8 A, 8 B, 8 B, 8 A) cancels a linear
drift exactly**, which plain ABAB does not (r4 window 1 → 2 slope 0.38 ms/token = 3 ms of bias on an early AB
pair). Layout over the existing 200-token story: 16-token warm-up, then 23 blocks of 8 = 11.5 ABBA groups → **92
tokens per side**, ≈ 13 s per A/B run at 15 tok/s.

Resolution (*device*-derived, pessimistic iid bound: per-token sd = window sd × √32): r4 skip steady windows sd
1.56 → 8.8 ms/token → 1σ on B − A at 92/side = **±1.3 ms**; r3 clean windows sd 2.6 → ±2.2; r4 control sd 3.65 →
±3.0. ±1 ms at 1σ needs 155 / 430 / 850 per side respectively — more than one story. So: 92/side settles SSM (4)
and skip (7-11) in one run; ILP (≈ 2.5) and waits (≈ 1) need **two runs pooled (±0.9-1.5)** and must be read on
the `MUL_MAT_ID` / `wait` columns, whose block-to-block noise excludes the dense op's. The iid bound is likely
pessimistic for ABBA (slow drift cancels), so the kit must log per-block ms and the parser must print the sd of
the per-group differences — round 5 measures its own noise and sizes round 6.

Spec for Tom (lab only, C side of the existing `GGML_CPU_AB` handler + parser; nothing in the app's decode path):
`GGML_CPU_AB_WARM=16` and `GGML_CPU_AB_BLOCK=8` (defaults 32/32 reproduce r4); after the warm-up graphs the knob
follows A B B A repeating per block; the prof window closes at each block boundary and the `cpu-prof` line carries
`ab=A blk=n`; `research_parse.py` prints per-group (B̄ − Ā) and their sd alongside the pooled Δ. Dart: the `ab` runs
use the 200-token story (not 72) and sample `scaling_cur_freq` of cpu4 and cpu7 every 8 received tokens into the
run's JSON (`mhz_blocks`). Verify on the host with `lab/checks/kit_runs.sh`: `EXPERT_SKIP:0:115` shows 1 warm-up +
23 block lines, k̄ 6.00 on every A block and 4.2-4.5 on every B block, barriers 746 on both sides.

Run list (≈ 12 min): official 32 ×2 (round-5 defaults: ILP + `SSM_INPLACE`, no skip); ABBA `SSM_INPLACE:1:0`,
`Q3K_ILP:1:0` ×2, `EXPERT_SKIP:0:115`, `0:130`, `0:140`; always-on story 200 at skip 115 / 130 / 140 and control;
official 32 with skip 115. If Tom ships lever 3 (waits) behind an env var: ABBA `MM_STEAL:0:1` ×2.

### 7.6 Pre-registered round-5 predictions (*estimated*; B − A as the parser prints it)

| item | prediction | falsified if |
|---|---|---|
| ABBA `SSM_INPLACE:1:0` (A = on) | **+3.4 to +5.0 ms** (off slower); `CPY` 108 vs 144 nodes, barriers 746 vs 818 on the right sides | < +2.5 or > +6 |
| ABBA `Q3K_ILP:1:0` ×2 pooled (A = on) | **+1.0 to +5.0 ms**, `MUL_MAT_ID` +4 to +18 %; central +2.5 | ≤ 0 (then ILP is 0 and the floor model §6.2 over-counted the stall); > +6 (then R23's −18 % was honest and the bias story is wrong) |
| ABBA `EXPERT_SKIP:0:115` | share 27-31 %, k̄ 6(1 − s) ± 0.05; `MUL_MAT_ID` −(0.9-1.0 × share) at ≤ 100 minflt/token, token **−6.5 to −8.5 clean, −7 to −12 r4-like**; per-group sd ≤ 3 ms | coefficient < 0.8 on a clean day → F > 15 % and the fixed per-op cost becomes a lever |
| ABBA `0:130` / `0:140` | clean **−9 to −11.5 / −10.5 to −13**; proportional to share ± 15 % | not proportional |
| story 200, skip 115, round-5 defaults | clean steady **53-55.5 → printed 17.3-18.2**; r4-like steady 55-58 → printed 14.8-15.8 | clean steady > 57 at ≤ 50 minflt/token |
| story 200, skip 130 | clean 50.5-53 → **18.3-19.3**; r4-like printed 15.8-16.8 | outside by > 1 tok/s |
| story 200, skip 140 | clean 48.5-51 → **19.0-20.3**; r4-like printed 16.5-17.8 | outside by > 1 tok/s |
| story 200 control (k 6, defaults) | clean 60-63 steady; r4-like 75-82 | — |
| warm-up (window 1 − steady, skip 115) | clean ≤ 3 ms (≤ 0.5 ms/token over 200); r4-like 20-30 ms (6-9 ms/token); scales with bytes/token | clean window-1 penalty > 5 ms with ≤ 100 minflt/token → a DVFS term exists after all, lever 7 reopens |
| `mhz_blocks` | cpu4-6 at 2,112 and cpu7 at 2,035 from block 1 on, at most block 0 lower | lower clocks persist into block ≥ 2 → lever 7 is live, price it from the deficit |
| official 32, skip 115 | r4-like 9-13; clean 16-18.5 | — |
| waits lever if built | −0.8 to −1.5; `mmid_thr` spread < 3 % | < −0.4 |
| levers 4 + 5 if built | −0.8 to −1.2, bit-identical | < −0.4 or any PPL digit moves |

Verdict for the owner, unchanged in direction and sharper in number: with everything exact in, **115 prints
17.5-18.5 on a clean day and ≈ 15 on a swapping day; 20 printed needs 140 + a clean day + the last exact
millisecond**. The threshold is the owner's decision; the day is the phone's.

### 7.7 Round-5 pre-registration, final (kit r5 as built; 2026-11-03 night, before the phone run; all *estimated*)

Supersedes §7.6 where they differ. The +2 % budget is final (owner): 130/140 closed; the edge is **118** (host full
set 9.7952, +1.98 %, k̄ 3.98 wikitext vs 4.11 at 115 → ratio 0.968; `lab/logs/r25/threshold.txt`). Device story
k̄ at 115 was 4.09-4.35, so 118 should read **3.95-4.25 → share 29-34 %**; 118 − 115 is ≈ 0.12 in k̄ ≈ 0.6-1.0 ms —
inside the ±1.3 ms 1σ, so the two skip A/Bs are a consistency pair, not a resolvable difference. Kit ABBA WARM 16
/ BLOCK 8 is what §7.5 asked for (the "32/32" was the env default that keeps r4 lines reproducible, not the kit
setting); 22 blocks instead of 23 → 88/side, 1σ ±1.3 → ±1.35 ms: nothing lost. `CONV_RING` is A/B-only in r5, so
the always-on rows do not contain it. `GGML_Q3K_VEC` "" → 0 in r3/r4 lines changed nothing (x4 bypasses it); no
re-scoring. Vanilla rows are priced from logged device pieces only: R20 `MMID 3 + X4` −10.7, R20 tiny 65,536 →
262,144 −9 to −14.5 (×3), R12 near-vanilla profile 80 ms at tiny 65,536 / eio 1 on a good day, R10 149 ms (5.5
tok/s printed) at tiny 0 / eio 0 on a swapping day, ILP −1 to −5, `SSM_INPLACE` −3.6/−4.5; 1,910 vs 818 barriers
at 0.3 µs is only 0.3 ms — the tiny-op cost is the per-op imbalance on 1,092 extra syncs, not latency.

| item (B − A as the parser prints it) | prediction | falsified if |
|---|---|---|
| ABBA `SSM_INPLACE:1:0` (A = on) | **+3.4 to +5.0 ms**; `CPY` 108 vs 144, barriers 746 vs 818 | < +2.5 or > +6 |
| ABBA `Q3K_ILP:1:0` ×2 pooled (A = on) | **+1.0 to +5.0 ms**, central +2.5; `MUL_MAT_ID` +4 to +18 % | ≤ 0 (ILP is 0, §6.2 floor over-counted) or > +6 (R23's −18 % was honest) |
| ABBA `EXPERT_SKIP:0:115` | share 27-31 %, k̄ 4.1-4.4; token **−6.5 to −8.5 clean, −7 to −12 r4-like**; `MUL_MAT_ID` −(0.9-1.0 × share) | coefficient < 0.8 clean (≤ 100 minflt/token) → F > 15 % |
| ABBA `EXPERT_SKIP:0:118` | share **29-34 %, k̄ 3.95-4.25**; `MUL_MAT_ID` 26.7 → 18.8-19.4 (**−7.3 to −7.9**); token **−7 to −9 clean, −7.5 to −12.5 r4-like**; 118 − 115 = −0.4 to −1.2 | share < 27 or > 37 %; token Δ outside −5.5…−10.5 clean |
| ABBA `CONV_RING:0:1` (B = on) | **−0.6 to −1.2 ms** (`CONCAT` 0.45-0.6 → 0, conv `CPY` −0.3 to −0.5, `SSM_CONV` −0.1); nodes 1911 vs 1839, barriers equal | > −0.2 (then the host −1.18 was M4 memory latency, not work) or < −2 |
| story 200, skip 118, r5 defaults, steady | clean **52.5-55 → printed 54-56.5 = 17.7-18.5**; r4-like 54.5-57.5 → printed 63-66.5 = **15.0-16.0** | clean steady > 57 at ≤ 50 minflt/token |
| story 200 control (k 6, r5 defaults) | clean 60-63 steady → printed 15.6-16.3; r4-like 75-82 → 10.2-11.5 | — |
| official 32, skip 118 | r4-like 9-13 (page-cache bound, TTFT tells the day); clean 16-18.5 | — |
| **vanilla story 200** (stock knobs) | clean steady **85-100 → printed 87-103 = 9.7-11.5**; r4-like steady 125-155 (eio 0 → ≈ 3× the majflt, R12), warm-up excess 20-30 → printed 150-185 = **5.4-6.7** (R10 5.5 anchor). Ratio control/vanilla **1.5-1.65 clean, 1.8-2.2 r4-like** | ratio < 1.35 clean → the exact stack (tiny/mmid/x4/ILP/SSM) is over-priced by ≥ 10 ms somewhere; ratio > 2.5 clean → tiny-op 0 costs more than the sync model allows |
| **vanilla official 32** | r4-like **3.5-5.5** (prefill is fault-bound either way; decode all in the warm-up regime); clean 8-10.5. R7's 3.4 (D1, 4t, tiny 0) is the only vanilla-ish device anchor and is consistent | — |
| warm-up (window 1 − steady, skip 118) | clean ≤ 3 ms (≤ 0.5/token over 200); r4-like 18-28 ms (5.5-8.5/token); 118's fewer bytes → ≈ 7 % less than 115 | clean penalty > 5 ms with ≤ 100 minflt/token → DVFS term exists, lever 7 reopens |
| `mhz_blocks` | cpu4 2,112 / cpu7 2,035 from block 1 on, block 0 at most lower | lower clocks persisting into block ≥ 2 → lever 7 live |

*[Closed by R27, 2026-11-05: `eio 0` costs +0.85 ms steady and ×9.6 faults, thread 0 is not slow; see `03_final.md` R27. Kept below as written.]*

**`EXPERT_IO` thread-0 madvise (next exact lever, cannot be A/B'd in one run).** Host: with eio 1 thread 0's
`mmid_thr` is +10 % (7724 vs 6890-7081 µs); device triples show thread 0 +6-10 % in every run (eio 1 saved). With
`MMID` bit-2 stealing the other two threads absorb most of it, so the cost per token is ≈ (thread-0 excess)/3 +
the end-of-op tail: 0.06-0.10 × 19-27 ms → 1.2-2.7 ms of syscall time → **−0.4 to −1.0 ms clean** if the calls
leave the compute threads; more on a swapping day (the readahead walk is longer). The catch: a sleeping helper
woken per layer costs ≈ 40 futex wakes × 20-50 µs = 0.8-2 ms, i.e. the whole gain — so the viable forms are (i)
fewer, larger calls (one range per tensor per layer instead of 3 × k̄), or (ii) issuing from whichever thread
finished its last chunk first, both bit-identical. Falsifier before any build: a round-6 always-on `story 200 ·
eio 0` row on a **clean** day (file cache ≥ 2.5 GB at start, ≤ 50 majflt/token both runs): if eio 0 is not faster
than control by ≥ 0.4 ms steady, or the eio-0 `mmid_thr` triple still shows thread 0 ≥ 3 % slow, the madvise costs
nothing worth moving. On a swapping day eio 0 loses by 10-40 ms and the row says nothing about this lever.

**Where 20 stands with 118 (printed, 200-token story).** Clean steady 53.8 (ILP + `SSM_INPLACE` + 118) → printed
55.2 = **18.1**; with `CONV_RING` as a round-6 default 53.2 → **18.3**; best corner (waits −1.0, madvise −0.7, F = 0)
50.9 → 52.3 = **19.1**. r4-like day: **15.5 ± 0.5**. 20 printed needs 48.6 ms steady; 118 sits 3.8-5.2 ms above
it on a clean day, and the exact levers left (CONV_RING 0.6-1.2, waits 0.8-1.5, madvise 0.4-1.0) sum to 1.8-3.7.
So **under the +2 % budget this lever set tops out at ≈ 18.5-19 clean; 20 needs a new MAC-cutting lever outside
k̄** — the dense `MUL_MAT` (24 ms, 45 % of the token at 118) is the only block big enough, and it is at the
bandwidth ridge (R24). That is the round-6 research question, not a threshold question.

### 7.8 Round-6 pre-registration (kit r6 as scoped; 2026-11-04 evening, before Tom's build is frozen; all *estimated* unless labelled)

Inputs: R25 device block and `reports/device/r5_2026-11-04.json` (parser output regenerated today), §7.7, Tom's
prefetch-before-skip code read (R25 "Open after R25" (2)). Round-6 scope: (1) `GGML_CONV_RING` default on;
(2) skip-aware expert prefetch — the `MADV_WILLNEED` set follows the skip mask (advice only, bit-identical), so at
118 the ≈ 68-76 MB/token of skipped experts (≈ 30 % of the 243 MB advised) is no longer read ahead; (3) kit ABBA
`CONV_RING:1:0`, `EXPERT_SKIP:0:118` ×2 pooled, `SSM_INPLACE:1:0`; always-on story 200: skip 118, control, `eio 0`,
`skip 118 · eio 0`; official 32 skip 118; vanilla official 32 + story 200; baselines ×2. Threshold fixed at 118.

**What R25 changed in my cost model.** (i) The exact stack (tiny-op, `MMID 3`, `Q3K_X4`, ILP, `SSM_INPLACE`) is
worth **−27 ms/token steady** (*device*, vanilla 91.0 → control 63.9), not the 32-37 I priced in §7.7: the
over-price was R20's tiny-op number (−9 to −14.5), a bad-day measurement; the tiny-op lever is ≈ −5 to −8 on a
day like R25's (the vanilla row's 14.6-23.2 ms of waits vs Vollie's 3-7 is where it lives). (ii) ILP is **−1 to −2**
(*device*, ABBA +1.2 ± 5.2 pooled, `MUL_MAT_ID` +3.5 %), the low edge of my +1 to +5; it stays default and out of
every sum. (iii) **Printed = steady + 1.4 + page-cache term**, and the page-cache term is now the dominant
variable on this phone: printed ms/token − last-window steady − 1.4 = 23.2 (skip 118), 5.4 (control), 6.5 (vanilla),
and in each case it equals Σ(window − steady) × 32/200 within 1 ms (22.5 / 5.0 / 5.5) — so the whole term is the
warm-up of windows 1-5, set by the file cache at run start (641 / 1,207 / 652 MB) and the expert bytes touched
(77 / 34 / 109 majflt/token). Steady itself is nearly day-independent now: skip-118 steady 55.4 at 77 majflt/token
vs the cleanest ABBA B blocks 52.9-59.0. (iv) The dense `MUL_MAT` sat at **23-25 ms through the whole stack**
(vanilla 25.9 → control 23.1; 249 weight products carry 559 MB, see §8): it is 42-45 % of the token at 118.
Revised "where 20 stands": clean steady at 118 with r6 defaults (`CONV_RING` −0.7) ≈ **53.5-54.7 → printed
55.4-56.6 = 17.7-18.1**; the priced exact crumbs (waits −0.8 to −1.5, madvise off the compute threads −0.4 to −1.0)
→ 51.5-53.5 → **18.2-19.0**; 20 printed needs 48.6 steady on a clean day, i.e. **−4 to −6 ms more that only the
dense `MUL_MAT` (23-25 ms) or a new budgeted lever can give** (§8). On an r5-like day no steady lever reaches 20:
the page-cache term alone is 5-25 ms/token.

**Day-state criterion (scoring rule, stated before the run).** R25 showed minflt 309-573/token on every run,
including the cleanest, so minflt does not separate days. A run is **clean** if `mem_before file=` ≥ 2.0 GB *and*
majflt ≤ 10 per generated token *and* (window 1 − window 6) ≤ 5 ms (r3 run 8 is the reference: 1.97 GB, 8/token,
+0.9 ms). A run is **r5-like** if `file=` at start is 0.4-1.3 GB, swapfree < 400 MB and majflt 20-120/token. Runs
between the two are scored against both bands and marked "between". Rows are scored in their own band; the
vanilla/control/skip comparison is scored only across runs whose `file=` at start differ by < 400 MB.

**Skip-aware prefetch, priced.** What it removes is advice, not faults: `MADV_WILLNEED` reads uncached pages
asynchronously (those reads are not counted as majflt), so its cost on a clean day is only thread 0's walk
(`mmid_thr` thread 0 +6.5 % at steady in the skip-118 run, 21317 vs 19999/20093; 30 % fewer ranges → ≈ −0.1 to
−0.3 ms steady) and on a swapping day the useless 68-76 MB/token of readahead competes for flash bandwidth and
evicts pages the next layers need. Window 1 of the skip-118 run (*device*, 117.0 vs steady 55.4 = +61.6) was
`MUL_MAT_ID` +34.7, waits +18.5, dense `MUL_MAT` +7.5, rest +0.9; the expert part (≈ 50 of 61.6) is the only part
the fix touches, and 30 % of the bytes behind it are the skipped experts' → **window-1 penalty −12 to −20 ms at an
equal start state** (61.6 → 42-50 at file ≈ 640 MB), steady −0.1 to −0.3, majflt/token −25 to −35 % at an equal
start state (77 → 50-58; the control-run comparison stays order-confounded unless the run order changes). The 2×2
of always-on rows (skip/control × eio 1/0) is the clean test: the *ratio* majflt(skip)/majflt(control) must come
out ≈ 0.65-0.8 at eio 0 (skip touches 30 % fewer expert bytes, dense bytes shared) and, after the fix, ≈ the same
0.65-0.8 at eio 1 (R25 read 2.25 at eio 1, order-confounded; before the fix the eio-1 ratio could only be ≥ 1).

| item (B − A as the parser prints it; always-on rows: steady = last window, printed = kit tok/s) | prediction | falsified if |
|---|---|---|
| ABBA `CONV_RING:1:0` (A = on, B = off) | own ops **+0.6 to +0.9** (`CONCAT` 0 → 0.53-0.57, `SSM_CONV` 0.26 → 0.40-0.42, `CPY` count 72 → 108, total ±0.1); token +0.5 to +1.5 (not resolvable at ±1.3 1σ; score the own-ops line); nodes 1855 / 1927, barriers 746 both | own ops < +0.4 or > +1.2 |
| ABBA `EXPERT_SKIP:0:118` ×2 pooled | share **28-33 %**, k̄ 4.0-4.3; `MUL_MAT_ID` **−7 to −9**, steady-block coefficient 0.9-1.0; token −6.5 to −9.5, pooled 1σ ≤ ±3.5 r5-like, ≤ ±2 clean; B-side `mmid_thr` thread-0 excess ≤ A-side's − 1 pt (shorter walk) | share < 26 or > 36 %; coefficient < 0.8 on a run with ≤ 10 majflt/token |
| ABBA `SSM_INPLACE:1:0` (A = on) | own ops **+3.4 to +5.0** (`CPY` +3.0 to +3.5, `SSM_SCAN` +1.0 to +1.5), barriers 746 → 818 | < +2.5 or > +6 |
| story 200 · skip 118 (r6 defaults), clean | steady **53.5-55.0**; window-1 penalty ≤ 5; printed **17.5-18.3** | steady > 57 or printed < 16.5 in a run that meets the clean criterion |
| story 200 · skip 118, r5-like | steady 54.5-57.5; window-1 penalty **25-50** (R25 61.6); majflt/token 40-65 at file ≈ 640 MB at start (R25 77); printed **12.5-15.5** (13.4-13.8 at R25's exact state) | at file ≤ 700 MB: window-1 penalty ≥ 58 **or** majflt/token ≥ 70 (the prefetch was not what inflated the skip run); printed < 12 |
| story 200 · control (k 6, r6 defaults) | clean steady 60.5-62.5 → printed 15.6-16.1; r5-like steady 62-67, printed 13.5-15 (R25 14.5 at 1,207 MB) | clean steady > 64 |
| story 200 · `eio 0` (prefetch off; the §7.7 madvise falsifier) | **clean: steady −1.0 to +0.3 vs control**, `mmid_thr` spread < 3 % (thread 0 no longer slow) → if eio 0 is *not* ≥ 0.4 faster or thread 0 is still ≥ 3 % slow, the madvise-off-threads lever (−0.4 to −1.0) is dead; **r5-like: majflt ×2-3 vs control** (R25 vanilla 109 vs 34), steady +0 to +10, printed −0.5 to −2.5 tok/s vs control | clean: eio 0 slower by > 1 ms steady at ≤ 10 majflt/token (then willneed pays even from cache and the page-cache model is wrong) |
| story 200 · `skip 118 · eio 0` | majflt/token **0.65-0.8 × the `eio 0` control row** (same day band); steady = skip-118 row ± 1; window-1 penalty ≥ the eio-1 skip row's on an r5-like day (faults now synchronous), equal on a clean day | ratio > 0.95 (the skip does not reduce bytes touched → the mask is applied after the faults somewhere else too) |
| 2×2 ratio test (after the fix) | majflt(skip)/majflt(control) at eio 1 ≈ at eio 0 ≈ **0.65-0.8**, both within ±0.15 of each other when the start states differ by < 400 MB | eio-1 ratio ≥ 1.3 with comparable start states |
| official 32 · skip 118 | r5-like 12-15 (R25 13.9 at 848 MB); clean 16.5-18.5 | — |
| vanilla official 32 / story 200 | official 8-10.5 (R25 9.4); story printed 9.5-11 r5-like, steady 82-97; **ratio control/vanilla 1.35-1.5** steady (R25 1.42) — the r5 numbers must reproduce within ±1 tok/s or the README table is a one-day number | ratio < 1.3 or > 1.6 |
| baselines ×2 (official 32, r6 defaults) | 8-13 r5-like (R25 8.2 / 12.0); 14-16.5 clean | — |
| `mhz_blocks` | 2,112 / 2,035 at every sample, at most one transient | any block ≥ 2 at 710 persisting |
| the skip rule's printed worth, same day, story 200 (skip 118 − control) | clean **+1.9 to +2.5 tok/s**; r5-like −1 to +1.5 (order and start state decide); after the fix the skip row should not lose to the control at an equal-or-better start state | skip prints ≥ 1 tok/s below the control at an equal-or-better `file=` at start |

Verdict pre-registered: round 6 does not move the steady number by more than the `CONV_RING` 0.7 (it is an
exact-crumb and page-cache round). What it can settle is (a) whether the madvise lever exists (`eio 0` clean row),
(b) whether the skip run's bad printed number in R25 was prefetch (the 2×2) or day state, and (c) a third
device confirmation of the skip rule at −7 to −9. The number for the owner stays: **118 prints 17.5-18.3 on a
clean day and 12.5-15.5 on an r5-like one; 20 needs §8.**
