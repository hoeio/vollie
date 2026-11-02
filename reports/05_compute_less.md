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
| 2 | (e) **in-memory re-layout of the experts, bit-exact** (cheaper unpack of the same 3-bit values) | experts run at 5.7 GB/s-eq vs dense 21.5: floor 11 ms for 243 MB | up to −25 ms at profile scale, −20 at best-run; multiplies with #1 (expert time = k̄ × per-slot cost) | none (same values) | raw-path lever, Tom's; not "compute less" but the thing #1 multiplies with |
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

- **Bit-exact expert re-layout** (ranked #2): the expert path moves 243 MB in 42.5 ms while the dense stream
  moves 560 MB in 26 ms; the gap is unpacking instructions, not MACs. A layout of the same Q3_K values that
  unpacks like Q4_0 (owner's "in-memory re-layout" amendment) is the biggest single lever on the board
  (−20 to −25 ms at profile scale if the per-byte rate matched the dense one) and it multiplies with #1.
  Lossy re-layout (Q3_K → Q4_0 values) is out: 64 experts × 40 layers at 4.5 bits = 3.4 GB of anon memory on a
  phone killed at 3.6 GB.
- **Candidate-set lm_head** (E2): −3 to −5 ms, but a miss is silent (you cannot know the argmax left the set
  without the full head), so its quality rule is the same mixed-policy PPL as (d); not measured today.
- **Skipping the shared expert / dense FFN per token**: 107 MB of Q4_0 at bandwidth ≈ 5 ms; no per-token
  signal exists for it (it has no router). Not pursued.
- MTP heads, trained exits, distilled drafts: training — out.

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

## 4. Where this leaves 20 tok/s

Within +0.5 %: the compute-less families buy ≈ 6-8 ms (one slot on ~15 % of tokens): 13-15 → 14.5-16.5 tok/s,
estimated. 20 is reachable only as **raw levers to ~16 tok/s (bit-exact expert re-layout, remaining waits)
plus the router-weight rule at k̄ ≈ 3.5-4.5, which costs ≈ +1 to +5 % PPL on wikitext-2 (stays under the
+10 % hard limit; the +0.5 % working tolerance does not survive).** That is the number the owner has to
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
