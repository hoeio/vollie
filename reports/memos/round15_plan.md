# Round-15 plan: a measurement design that can resolve the two parked expert-line knobs (Bonnie, 2026-11-08, pre-registration draft)

Labels: *device* = per-block `cpu-prof` lines of `reports/device/r14_2026-11-08.json`, `r14b_2026-11-08.json`, `r12_2026-11-07.json`, `r8_2026-11-06.json`
re-read with `tools/research_parse.py`'s own `gate_blocks` (nothing re-scored; R38 and R45 stand as written); *estimated* = arithmetic. Scratch script
and per-block CSV: `/private/tmp/claude-501/-Users-woobeenbaek-Documents-project-vollie/a4951e57-17a0-5909-8504-21a531eeda04/scratchpad/{anal.py,anal2.py,blocks.csv}`.
"Clean" = a block with majflt/token ≤ 10 at the modal clock (R39 §B, T = 10). Scored line = `MUL_MAT_ID` + `GLU` (compute + wait), ms/token, Δ = B − A.

## A. Why the σ ≤ 0.4 rule failed

**A.1 Variance ladder on r14b's 22 clean blocks (13 A / 9 B; device).** Least squares, residual sd about the side means:

| model | Δ (B − A) | residual sd | variance | removed |
|---|---|---|---|---|
| side only (= R45's −3.08 ± 0.81) | −3.08 ± 0.84 | 1.94 | 3.76 | — |
| + run dummies | −3.21 ± 0.86 | 1.97 | 3.88 | **0 %** (no between-run component on clean blocks) |
| + within-run linear position | −3.37 ± 0.70 | 1.60 | 2.56 | **32 %** (drift −0.21 ± 0.08 ms/block ≈ −5 ms over a 23-block run) |
| + minflt/token | −1.46 ± 0.60 | 1.04 | 1.07 | **72 %** |
| paired adjacent blocks (2k−1, 2k; both clean; n = 6) | −3.61 ± 0.77 | pair-diff sd 1.89 → per-block 1.34 | 1.80 | **52 %** |

The same ladder on r12's 13 clean kernel blocks (`MUL_MAT_ID` compute): side only −5.16 ± 3.20, resid 5.76; + position + majflt −6.55 ± 1.76, resid 3.08; + minflt
alone resid 1.03 (slope 5.5 ± 0.3 ms per 100 minflt/token). r14's 6 clean blocks are too few for any fit (5 parameters on 6 points gives −10 ± 1; disregarded).

**A.2 Reading.** (1) Between-run variance is nil: once a block is clean, which ABBA it came from does not matter — pooling runs and sessions is legitimate.
(2) Within-run drift is a third of the clean-block variance: blocks get ≈ 0.2 ms/block faster as the run proceeds (residency still improving at ≤ 10 majflt).
ABBA cancels linear drift exactly only when both sides have the same mean position; the gate breaks that symmetry (r14b counted blocks: mean position
A 13.5, B 12.9 — small here, but at T = 15 A 14.1 / B 12.9, and the estimate moved to −0.72). An interleaved (paired) design removes this component by
construction: paired adjacent blocks give the same −3.6 at per-block sd 1.34 instead of 1.94. (3) The remaining ≈ 1.0-1.3 ms is **minor-fault stall**, not
arithmetic: the scored line rises **4.9 ± 1.9 (r14), 5.0 ± 0.7 (r14b), 5.5 ± 0.3 (r12), 2.8-4.4 (r8) ms per 100 minflt/token on clean blocks** — the same slope
four days running, 30-50 µs per minor fault, which is the cost of a fault that waits on a readahead page lock (counted minor, not major), not of a map-only fault
(≈ 2 µs). The campaign's "clean" state has always carried 100-250 minflt/token (r8: 99-223), i.e. 3-12 ms/token of residency stall that no block gate on majflt sees.

**A.3 Why the rule could not be met.** σ ≤ 0.4 was fixed from the band width (−0.5..−1.05) when the clean-block sd was believed to be 2.3-4.5 (R42 am. 1).
The realised sd 1.6-2.0 needs ≈ 40 clean blocks/side; three ABBAs give 7-13 on a good day. Two further defects: (a) the rule tied σ to the band, not to the
distance from the thresholds — a −3.1 ± 0.8 result (3.8σ from zero, 3.3σ beyond pass) was declared unresolved by a rule written for −0.7; (b) the block gate
is coarse: a block is 8 tokens, and one 20-ms readahead stall inside an otherwise clean block passes the mean-majflt gate and lands in the scored line
(pair-gated blocks at T = 15 give −1.33 ± 1.19 with pair-diff sd 4.0, versus 1.9 at T = 10: the 10-15 majflt blocks carry 4× the variance). The gate-sensitivity
R45 reported is this coarseness, not a sign that the effect is unreal.

**A.4 The confound the next design must handle.** The fused side draws **0.62 / 0.75 / 0.94 (r14) and 0.74 / 0.80 / 0.75 (r14b)** of side A's minflt/token in
every run, ABBA and BAAB alike; the kernel knob (r12) draws 1.02 / 0.98. So the fuse itself cuts minor faults by ≈ 25 % (it is not a harness or position
effect). At the day's slope that is worth 60 × 0.05 ≈ **−2.9 ms of the clean-block −3.6** (paired blocks: minflt B − A = −57), leaving ≈ −0.7 — the
pre-registered central. The same subtraction on r12's clean kernel blocks (minflt A 183 vs B 132 by selection) turns −6.3 into ≈ −3.8, the R35 central.
*Estimated, a hypothesis:* the fused chunk loop touches each expert page from fewer threads (one chunk claim covers gate and up rows; chunk2 halves claim
boundaries), so fewer concurrent minor faults contend on the same page. Consequence: **minflt cannot be a covariate** (it is on the causal path of the knob),
but it is a cheap, low-variance **confirmation line** (ratio 0.62-0.80 on every fused run) and a real effect the user sees on this phone — and ≈ 0 on a phone
whose model is resident. Section D keeps the two readings apart.

## B. Designs that can resolve −0.5..−3 ms on the expert line

Cost facts (device): one 200-token ABBA story = 23 scored blocks = **13.7-14.0 s wall** (r14b); the kit's inter-run pause is 30-70 s, so kit minutes are
pause-dominated (r14b: 75 s of runs in 5.1 min). Tom: shorten the pause for consecutive `ab` runs if heat allows (heat 0.61-0.65 on r14b).

**(1) More and longer ABBAs after a dropped in-kit warm-up.** Block design unchanged, parser unchanged. Need: 40 clean blocks/side at sd 1.9 for σ 0.4
with the plain estimator; **20/side with the paired-block estimator** (sd 1.34). Yield 7-13 clean/side per 3 runs on a good day → **6 ABBAs after one
dropped warm-up run** reach 20/side on an r14b day; 10 on an r12 day; a thrashing day (r10/r11/r14 runs 1-2) yields ≈ 0 whatever the count. Kit:
7 runs × 14 s ≈ 1.6 min of runs, 5-8 min with pauses. Cannot measure: anything on a thrashing day; the per-token stall structure. Pros: zero code,
comparable to r14/r14b; cons: the gate stays coarse (A.3 b), so the T-sensitivity disease remains.

**(2) Per-token A/B interleaving with per-token profile lines.** Knob flipped per graph in token order A B B A A B B A … (ABBA at token grain, so
A-follows-B as often as B-follows-A and any carry-over cancels to first order). Pair k = tokens (2k−1, 2k), one A one B; statistic d_k = y_B − y_A;
Δ = mean d, **σ = sd(d)/√N**; a pair counts iff **both tokens have majflt = 0** (primary; ≤ 2 secondary) at the modal clock — a sharp gate on real
fault-free tokens, which block means cannot give. Expected σ (*estimated* from A.1): block pairing removed 52 % of the variance at 8-token lag; at
1-token lag the drift share is fully removed and the stall share is halved by the sharp gate → per-token pair-diff sd ≈ 3.5-5.5 ms (iid-token bound
√8 × 1.34 × √2 = 5.4; drift-dominated case ≈ 3.5) → **100 counted pairs give σ 0.35-0.55; 3 × 200-token runs (≈ 150-250 counted pairs on an r14b day)
give σ 0.25-0.45.** Cannot measure: class-F knobs (skip thresholds, caches, prefetch — anything whose state spans tokens) and nothing on a thrashing
day either. Needs Tom: (i) `GGML_CPU_AB_TOKEN=1` — the existing per-graph env toggle re-read at op dispatch, flipped by the harness per token in ABBA
order; (ii) one `cpu-prof` line per token (graphs=1) with `majflt`, `minflt`, `us/token` and the op table — 200 lines/run; (iii) parser `--pairs`:
paired Δ ± σ per line, counted-pair yield, and a **null check**. (iv) Validation inside the kit, pre-registered: an **A/A interleaved run** (knob 0 on
both labels) must give 0 ± σ on every line and minflt ratio 1.00 ± 0.05; the interleaved A-token mean must agree with the block ABBAs' A mean within
0.5 ms (alternation itself is neutral). Where I disagree with Clyde: interleaving does not "fix" the variance — it buys √2 (drift) and a sharp gate;
the limiting resource is still clean tokens, and only warming supplies those. Where I agree: the sharp per-token gate is the only thing that cures the
T-sensitivity, and the paired A/A null is the only clean separation of harness from knob (section C).

**(3) Pooled multi-session rule, pre-registered before session 1.** What counts: every counted pair (or clean block) at the modal clock from every r15
session, no session excluded post hoc, r14/r14b data excluded (collected before the rule). Stopping rule (sequential, one-sided 95 %): **pass iff
Δ + 1.645σ ≤ pass threshold; falsified iff Δ − 1.645σ > falsify threshold; else continue; at most 3 sessions; if neither after 3 → "unresolved,
parked", not scored whatever σ** (R35's "score whatever σ" is exactly what falsified the kernel on fault-confounded sides and is not repeated).
Heterogeneity check: per-session Δ's χ² (A.1 says between-run variance is nil on clean blocks, so pooling is legitimate; a χ² p < 0.05 is reported and
pooling proceeds anyway — pre-registered). This is a rule, not a design; it wraps (1) or (2).

**(4) Fault-slope regression — kill as estimator.** All-block slopes 0.396 ± 0.044 (r14) vs 0.675 ± 0.077 (r14b); clean-block slopes 0.15 ± 0.58 vs
0.52 ± 0.30; adjusted Δ −4.1 ± 2.1 vs −0.3 ± 1.6, residual sd 6.5-8.6 → σ ≥ 1.5 even with 70 blocks, and the slope is not a device constant (it depends
on which tensor's pages fault and on readahead state). The minflt slope *is* stable (A.2) but confounded with the knob (A.4). Both stay as diagnostics.

**Ranking.** 1. **(2) + (3)**: per-token interleaving with a 0-fault pair gate, wrapped in the sequential rule — resolves −0.7 in one good session,
and the A/A null settles section C. 2. **(1) + (3)** — the fallback if Tom's per-token mode is not ready: 6-10 block ABBAs with the paired-block
estimator; it resolves −3 but not −0.7. 3. (4) never. Recommend 1; build 2 into the same kit as two block ABBAs (ABBA/BAAB) for continuity with r14.

## C. The `MUL_MAT` +0.48 on side B

Facts (device, r14b clean blocks): pair-gated `MUL_MAT` B − A **+0.54 ± 0.21** (T = 10), +0.43..+0.77 at every T; per run +0.03 (ABBA) / **+0.41 (BAAB)** /
+0.92 (ABBA) — **no sign flip in BAAB**; by position inside the quartet A = 24.08 / 24.11 / 23.88 / 24.13, B = 24.38 / 24.59 / 24.64 / 24.37 — B is ≈ +0.4
at every position. r14 clean: +0.62 (compute +0.39, wait +0.23). r12's kernel knob: `MUL_MAT` −0.83 on its clean blocks, so R35's "+0.3 side-B floor" is
not a harness constant. Conclusion: **position bias is excluded; it is either a harness/labelling effect common to all knobs-on sides or a real side
effect of the fused op on the dense ops** (+0.5 ms / 40 layers = 12 µs per layer on thread 0's compute — consistent with the fused op's larger per-chunk
working set evicting activation/norm lines, or with bus DCVS stepping down after a shorter expert op; *estimated*). **The one separating check: the A/A
interleaved null run of B(2)(iv)** — if `MUL_MAT` shows 0 ± 0.2 and minflt ratio 1.0 with the knob off on both labels, the +0.5 and the 0.75 belong to the
fuse, and the token line (clean A 53.81 ± 2.29 vs B 51.40 ± 1.93 → −2.4 ± 0.9) is the number that matters, not the expert line alone. Pre-register the
**per-token total** as a second scored line for exactly this reason.

## D. What a resolved verdict would buy (all *estimated*; printed = 1000/steady as in R45; outside-graph ≈ 1.0 ms/token, R35, would take ≈ 0.4 tok/s off each)

Clean steady without either parked term: 52.4-53.0 (R41/R43) − 0.32 (C1-F16, host-passed, unshipped) = **52.1-52.7**. Terms are additive by construction:
the fuse was measured with kern = 1 on both sides; the kernel with the fuse off; C1 touches dense `MUL_MAT` only.

| parked terms at | kern | fuse | steady ms | printed tok/s |
|---|---|---|---|---|
| (i) zero | 0 | 0 | 52.1-52.7 | **19.0-19.2** |
| (ii) pre-registered centrals (R35 / R42 am. 1) | −3.75 | −0.7 | 47.6-48.2 | **20.7-21.0** (R44 wrote 20.4-20.9 with the outside-graph haircut) |
| (iii) clean-block point estimates (R38 post hoc −6.27, 13 blocks; R45 pooled −3.05, 18/10) | −6.27 | −3.05 | 42.8-43.4 | **23.0-23.4** on this phone in its usual residency state |
| (iii′) the same with the minflt share removed (A.4) | ≈ −3.8 | ≈ −0.7..−1.5 | 46.8-48.2 | **20.7-21.4** fault- and stall-free |

The honest statement: (ii) and (iii′) agree — the arithmetic part of both knobs is what was pre-registered, and 20 printed fault-free needs **both** of them
plus C1; the excess in (iii) is residency relief the fuse gives a page-pressured phone, real for this owner, not portable to a resident model.

## E. Rules check

Everything here is measurement and runtime scheduling of the kit: both knobs are bit-exact on device (R35 ×4, R44/R45 greedy identical), default off,
no PPL budget touched. **The in-kit warm-up is the runtime's own work** — a dropped first story run inside the kit; the owner is asked for nothing about
the phone's state (no closing apps, no reboot, no waiting; R25-R33 `madvise`/prefetch levers stay closed; the "keep the file warm while foregrounded"
scheduling idea of R45 is not proposed here). **Excluded as re-litigation:** re-scoring R38's pooled set or R45's 22 blocks under the rules above;
including r14/r14b blocks in the r15 pool; any reading of R38/R45 other than "falsified under its rule" / "unresolved, parked". **One decision for Clyde and
the owner, stated plainly:** R38 says the kernel's device speed claim is "not re-proposed". A fresh pre-registration of `Q3K_KERN:0:1` under the r15
rule on new data is a new experiment, not a re-score — I recommend it (1 min of kit; D needs the term), with the R38 verdict left standing in the log.
If the owner reads it as re-litigation, r15 scores the fuse alone. Kit hygiene for Tom, not research: the one-shot `cpu-bench` guard emitted nothing on
the second kit run in one process (R45) — re-arm per kit run; and the inter-run pause.

## Pre-registration draft for R46 (Clyde to paste and amend)

**Round 15 (measurement round; build adds `GGML_CPU_AB_TOKEN` per-token knob flipping + per-token `cpu-prof` lines + parser `--pairs`; knobs
`MMID_FUSE+CHUNK2` and `Q3K_KERN`, both exact, default off/on as today; no numerics change; C1-F16 rides as a side-level ABBA ×2 with the `router=`
field, §F of `round14_plan.md`).** Kit: baseline ×1; bench (guard re-armed); **warm-up story ×1 (dropped, pre-registered as unscored)**; A/A interleaved
null ×1 (knob off both labels); `MMID_FUSE+CHUNK2:0:1` interleaved ×3 (200 tokens, token order A B B A …); `Q3K_KERN:0:1` interleaved ×2; block ABBA
`MMID_FUSE+CHUNK2:0:1` ×2 (ABBA / BAAB, continuity + hygiene 666/1775/80/76 on every B block); official 32 ×2; ≈ 7-9 min at today's pacing.
**Scored lines (paired, both tokens majflt = 0 at the modal clock):** fuse — `MUL_MAT_ID`+`GLU` **band −0.5..−1.05 arithmetic, pass ≤ −0.4, falsified
> −0.15**; per-token total reported alongside with its own band −0.3..−1.2 (dense side effect allowed for); kernel — `MUL_MAT_ID` compute **band
−3.0..−4.5, pass ≤ −3.0, falsified > −1.5** (R35 thresholds, new data). **Verdict rule (sequential):** pass iff Δ + 1.645σ ≤ pass; falsified iff
Δ − 1.645σ > falsify; else continue to the next session; ≥ 60 counted pairs per line per session for the session to count; at most 3 sessions;
unresolved after 3 → parked, not scored. **Validity, every session:** A/A null 0 ± σ on every line and minflt ratio 1.00 ± 0.05 (else the session is
void); interleaved A-token mean within 0.5 ms of the block ABBAs' A mean; clock modal on counted pairs; hygiene exact. **Confirmation lines:** `GLU`
compute −0.12..−0.18; fused-side minflt ratio 0.65-0.85 on every fused run (fact, mechanism per A.4). **Reported, not scored:** `MUL_MAT` paired Δ on the
fuse line (the C question: ≈ 0 in the null, +0.3..+0.7 in the knob run = fused-op side effect), fault-slope and minflt-slope diagnostics, cleanest-token
floors, official 32 with clock and faults. **Honest expected outcome:** on an r14b-like day both lines resolve in one session (fuse pass at the
arithmetic band's edge or above it, token total −1..−3; kernel pass); on a thrashing day the warm-up is spent and the yield rule carries the verdict to
session 2. Where 20 stands if both pass at their centrals: 47.6-48.2 → 20.7-21.0 printed fault-free (*estimated*, outside-graph haircut ≈ 0.4).
