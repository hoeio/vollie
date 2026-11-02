# 05 — Computing less per printed token (Bonnie, 2026-11-02, round "r21", host only)

Question (owner, `04_campaign.md` §9): which tokens, layers, experts or operations can be left uncomputed —
or computed once and reused — with a quality cost inside the gate, by a rule that does not depend on this
model or this phone? Every number below is **host** (M4 Pro, wikitext-2 test, ctx 512, 60-chunk screens
unless stated; raw logs `lab/logs/r21/`) or **estimated** from the R18/R20 device profiles. Nothing here
was run on the phone.

## 0. The arithmetic the ranking is measured against

One decoded token on the phone (R20, 3 threads, profile scale, ≈ 96 ms): `MUL_MAT_ID` 42.5 compute + 8.2
wait (**≈ 8.5 ms per routed expert-slot**, 6 slots), dense `MUL_MAT` 26 + 2 (lm_head Q6_K 126 MB ≈ 6 ms
of it), SSM/small ops ≈ 10, other waits ≈ 7. Best-run scale with the three bit-identical defaults
(13-15 tok/s, estimated): 67-77 ms, experts ≈ 36-42 ms, **6-7 ms per expert-slot**, ≈ 1.7-1.9 ms per layer.

20 tok/s = 50 ms. From 67-77 ms that is **−17 to −27 ms**, i.e. the token has to lose 25-35 % of its
multiply-adds *after* the raw levers. In expert-slots: 2.5-4 of 6 per token. In layers: 9-14 of 40.

Quality budget: +0.5 % working tolerance (9.6531 on the full set), +10 % hard (9.948 vs F16 9.0438;
stock D6 is already +6.2 % over F16, so the hard limit leaves **+3.6 % over stock** = 9.948/9.6051).
On the 60-chunk screen the baseline is 9.8702 (all three builds identical, so one reference serves).

## 1. Ranked families

| # | family | lever on this testbed | gain at best-run scale (est.) | quality (host) | verdict |
|---|---|---|---|---|---|
| 1 | **(a) adaptive expert count by the router's own weight** | skip slot if normalized w < t (exists: `GGML_EXPERT_SKIP`) | −6 to −8 ms inside +0.5 %; **−12 to −17 ms at k̄ ≈ 4.2 for +1.9 %**; 20 tok/s needs k̄ ≈ 3-3.5 → ≈ +5-9 % | adaptive is **3× cheaper in PPL per expert saved than uniform k** (measured) | the only family with the magnitude; price is the owner's call |
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
| 0.100 | 15-18 % | 5.0 | 9.9158 | +0.46 % | **+0.82 %** (9.6834); en.txt +2.1 % |
| 0.120 | ~30 % | 4.2 | 10.0554 | **+1.88 %** | — |

Router-weight CDF (share of (token, slot) pairs whose normalized weight is below t; `lab/logs/r21/skipshare.txt`,
greedy, 64 decoded tokens per prompt, bit-exact counter in the module): t = 0.03: 0 %, 0.05: 0-2 %, 0.07: 2-5 %,
0.09: 9-15 %, 0.11: 20-24 %, 0.13: 36-40 %, 0.16: 57-59 %, 0.20: 72-75 %, 0.25: 87 %. The two texts agree
within 1-6 points, so the rule's *coverage* is a property of the router, not the text. There is no bimodality
(no population of "two dominant experts" tokens): a *gap* rule (drop slots with w < g × w_top) will behave
like the threshold rule; the cumulative-weight (top-p) form is the same ordering with a per-token budget.

What the two curves say, machine-independently:
- **At equal k̄ the router-adaptive rule costs ~3× less PPL than uniform k**: k̄ ≈ 5.0 adaptive +0.46 % (60 ch.)
  / +0.82 % (full) vs uniform 5 +1.26 %; k̄ ≈ 4.2 adaptive +1.88 % vs uniform 4 +6.74 %. The router's weights do
  know which experts matter — the signal the owner asked about works, and it is the model's own.
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

At best-run scale (×0.84) N = 16 alone is −8 to −19 ms; **N = 16 + k̄ 4.2 is −16 to −24 ms, i.e. 67-77 → 45-60 ms =
17-22 tok/s (estimated)** — the first combination on the board whose band contains 20, at +1.9 % PPL (60-chunk
screen; ≈ +2.3 % full-set by the screen's bias) and 648 MB of anon memory.

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

**S5 — re-laid expert microbench, then the bounded cache (build in this order; stop at the first failure).**
(i) Microbench, host, no model: a 60-line `lab/relayout_bench.c` (or a `test-backend-ops`-style case) that takes
one expert's gate matrix (512 × 1536 Q3_K, 1.01 MB; dump it once from D6 with `gguf-py` into
`lab/logs/r21/expert.bin`) and one Q8_K activation row, times 1,000 × `ggml_vec_dot_q3_K_q8_K` over all 512 rows
vs the same values in a candidate layout with its own dot (first candidate: 4-row interleave as `Q3K_X4` already
walks it, 6-bit sub-block scales pre-expanded to int8, 2-bit and 1-bit planes de-interleaved per 4-row group;
same byte count), asserts every row's result bit-identical, prints µs per MB for both at 1 thread. Go/no-go:
≥ 1.5× faster per MAC (the host Q3_K→Q4_0 gap measured today); log to `lab/logs/r21/relayout-bench.txt`.
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

Within +0.5 %: the compute-less families buy ≈ 6-8 ms (one slot on ~15 % of tokens): 13-15 → 14.5-16.5 tok/s,
estimated. 20 is reachable only as **raw levers to ~16 tok/s (the bounded re-laid expert cache of §1(e′),
remaining waits) plus the router-weight rule at k̄ ≈ 3.5-4.5, which costs ≈ +1 to +5 % PPL on wikitext-2 (stays
under the +10 % hard limit; the +0.5 % working tolerance does not survive).** The one combination whose
estimated band contains 20 is N = 16 re-laid experts + k̄ 4.2 (+1.9 % screen): 17-22 tok/s at best-run scale. That is the number the owner has to
decide with. The method itself — one threshold on the router's own softmax — is model- and
device-independent; the price is model-dependent and must be measured per model the way it was today
(two 3-minute curves).

Round 3 proposal: ship `GGML_EXPERT_SKIP` as a user-visible "speed vs quality" knob only after the owner
sets the budget; Tom's S1 sweep decides threshold vs top-p vs gap at k̄ 4.2 and 3.5 before any device run;
pre-register the device prediction from R18's one skip data point (8 % skipped → −9.1 ms, better than
proportional) against the proportional estimate above.

Sources consulted (direction only, no numbers borrowed): Lu et al., *Not All Experts are Equal* (2402.14800,
ACL 2024: post-hoc dynamic expert skipping by routing weight, ~1.2× on Mixtral); *SimLens* (2507.17618) and
*The Diminishing Returns of Early-Exit Decoding in Modern LLMs* (2603.23701) on training-free early exit.
