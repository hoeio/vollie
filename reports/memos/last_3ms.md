# The last 3 ms: what can still close the gap to 20 printed, priced by device kernel class (Bonnie, after R31; 2026-11-06)

Labels: *device* = `reports/device/r8_2026-11-06.json` (2,112 / 2,035 MHz, clean day, 117 + head2 q4_0/2048 story run, windows 5-6
unless stated); *host* = M4 or this memo's two probes (`lab/logs/r31/`); *estimated* = arithmetic. Kernel classes from R31: a
repacked `sdot` gemv reads at **22.5-22.8 GB/s** (byte rate); the Q2_K/Q3_K `vec_dot` class is compute-bound (R29 4.9-5.9 × M4 per row;
experts 116 µs per slot, below).

## 1. Where the 53.0 ms token goes (device, steady) — and the gap is not 2.4-3 ms

| line | ms/token | class / note |
|---|---|---|
| `MUL_MAT` 265 ops | 22.0-22.1 + **1.7-1.9 wait** | 525.4 MB (436 dense + 89.3 head) in 22.0 = **23.9 GB/s → at the byte rate; kernels cannot help, only bytes** |
| `MUL_MAT_ID` 120 ops | 19.3-19.4 + 0.7-0.8 wait | k̄ 4.0 → 164 slots × **116 µs** (skip ABBA: 8.28 ms / 71.5 slots); byte floor of a slot is 1.046 MB / 22.8 GB/s = 46 µs → **2.5 × above bytes: the Q3_K kernel, not I/O** (R23's memcpy anon cache gave −0.7 = noise) |
| `SSM_SCAN` 36 | 2.7 + 0.2-0.5 | |
| small ops (MUL 0.9, ADD 0.55, RMS_NORM 0.37, GLU 0.39, FA 0.3, SSM_CONV 0.26, GET_ROWS 0.19, UNARY/ARGSORT/SCALE/CPY/CONT 0.4) | 3.4 + 0.5 wait | ≈ 1,470 nodes |
| per-node dispatch (us/token − compute − wait) | **1.6-1.7** | 1,855 nodes → 0.9 µs/node (includes the profiler's two clock reads per node) |
| **profile total** | **52.4-53.0** | |
| outside the graph (decode wall 11,852.6 − 6 windows 11,156.8 − 7 tokens at steady 371.1, ÷ 199) | **1.63** | sampler chain + `llama_decode` non-graph + Dart stream; *device, derived* |

Printed arithmetic (199 intervals): fault-free at today's steady → 199 / (199 × (53.0 + 1.63)) = **18.3 tok/s**; the run printed 16.79
because windows 1-4 carried **+978 ms** of start-state excess (547 / 288 / 77 / 86), 75 % of it in `MUL_MAT_ID` (majflt 12.4 → 1.1/token),
12 % in `MUL_MAT`. **20.0 printed needs steady ≤ 48.4 ms with the 1.63 outside-graph cost unchanged** (50.0 only if that cost were
zero). So the honest gap is **4.6 ms/token in the profile, or ≈ 4 ms of profile + ≈ 0.6 ms outside it, plus a fault-free run** — Clyde's
2.4-3 ms is the gap to 20 tok/s *profile*. I disagree with the framing, not the numbers.

## 2. The candidates, priced

**(a) Head further.** Head op 3.92 ms vs the screen's byte floor 86.7 MB / 22.8 GB/s = 3.80 → select + 2,048 Q6_K rows + imbalance
≈ **0.1 ms** *device*; M 1,024 or 512 can save ≤ 0.05-0.1 ms and buys PPL risk against a 0.09 % margin — host probe below. Lower-bit
*repacked* screens on arm64 dotprod, from `repack.cpp` `ggml_repack_get_optimal_repack_type`: **Q4_0 4x4 (dotprod) / 4x8 (i8mm), IQ4_NL 4x4,
MXFP4 4x4, Q8_0 4x4/4x8, Q4_K/Q5_K/Q6_K 8x4, and Q1_0 4x4** (this 0.7.0 tree has 1-bit Q1_0 and 2-bit Q2_0 types). **Q2_K's repack is
AVX-512 only; Q3_K has none; Q2_0 has only a plain `tbl`+`sdot` vec_dot** (plain class: R30 host plain Q4_0 ran 1.6 × the repacked).
Bytes if the kernel were at the byte rate: MXFP4 4.25 bpw → 81.9 MB (−0.2 ms, screen quality below Q4_0 → larger M → ≈ 0 net); IQ4_NL
same bytes as Q4_0; Q2_0 43.4 MB (−1.9 ms floor) but a 2.25-bpw single-scale screen is worse than Q2_K, which needed M 16,384 for
+0.059 % — M 4,096 was +0.467 % with Q2_K (R28); the budget margin is 0.086 %: **dead by PPL before kernels**; Q1_0 20 MB: same argument,
harder. **The head is done at 3.9 ± 0.1 ms; nothing on this list is worth a Tom hour.** (Falsifier if anyone insists: Tom accepts
`q2_0`/`q1_0` in `GGML_HEAD2_Q` = one line; host 60 chunks at M 4,096.)

**(b) Whole-tensor lower-bit copies of the other dense tensors: dead.** Served types (gguf-py dump): `ssm_in` Q4_0 200.6 MB, `ssm_out`
Q4_0 95.6, `gate/up_shexp` Q4_0 70.8, `down_shexp` Q4_0 31.0 + Q4_1 4.9 (5 layers), attention Q4_0 14.2, `token_embd` Q6_K (head2 covers
it), router `ffn_gate_inp` **F32 15.7 MB**, conv/norms F32 3.4. Everything that could be Q4_0 already is, and no byte-rate sub-4-bit
kernel exists on this ISA (above), so bytes cannot move there. The one byte item left: a **Q8_0x4 copy of the F32 router** (15.7 → 4.2 MB,
40 ops × ≈ 12 µs = **−0.3 to −0.5 ms** *estimated* at the byte rate; routing logits change in the 4th digit → near-tie expert flips → not
exact, k̄ and PPL to be measured on the full set; same anon-copy mechanism as head2, `GGML_DENSE_Q=ffn_gate_inp=q8_0`). Sparsity: my
`ssm_out` column skip stays withdrawn (fp32 axpy +2..+5, `dense_bytes_device.md` §4).

**(c) Waits 3.3-3.8 ms.** By line: `MUL_MAT` **1.7-1.9** (6.8 µs/op over 265), `MUL_MAT_ID` 0.7-0.8 (6.5 µs/op), `SSM_SCAN` 0.2-0.5,
`RMS_NORM` 0.3, `ADD` 0.1, rest ≈ 0.1. Barrier latency is 0.30 µs (R18, 3 threads) → ≈ 0.25 ms of the 3.5 is sync; the rest is
imbalance. The dense `MUL_MAT` chunking (`ggml-cpu.c` 1669-1722): at decode `chunk_size = 64` rows, dynamic stealing when ≥ 12 chunks,
else a static 3-way split. Chunk tails: `ssm_out` 1,536 rows = 24 chunks of 110 KB = 4.9 µs each; `shexp` 1,024 rows = 16 × 2.4 µs; `ssm_in`
101 × 2.4 µs; the F32 router (64 rows) and attention k/v (512 rows) fall into the static split. Expected tail ≈ half a chunk per op ≈ 0.5-0.7
ms/token of the 1.8; the remainder is per-core bandwidth jitter the same stealing absorbs if the grain is fine. **`GGML_MM_CHUNK=16` (rows
per chunk when `nr1 == 1`, both `mul_mat` and `mul_mat_id`): bit-exact** (a dst row is one full-K vec_dot on one thread whatever the chunk;
reduction order unchanged), +3 atomics per op. *Estimated* `MUL_MAT` wait 1.8 → 0.7-1.2 (**−0.6 to −1.1**), `MUL_MAT_ID` wait −0.1 to −0.3;
compute ± 0.2 (prefetcher restarts on 14 KB runs). Falsifier: if the wait does not move, the wait is not chunk-tail (arbiter or thread-0
overhead) and chunking is closed. Separately, a different thread split (2 + 1, affinity) is not exact-neutral on this kernel and R18's
4-thread cliff says no.

**(d) `MUL_MAT_ID` at 19.3 ms: the elephant, and what is in-rule.** 116 µs per slot against a Q4_0x4 byte floor of **58 µs** (1.327 MB
Q4_0 per expert at 22.8 GB/s): a repacked Q4_0 copy of the experts would run the expert product at ≈ half — **−9 ms/token *estimated***,
R31's kernel class applied to the one op that is not at the byte rate. The full copy is 3.4 GB anon = R9's OOM, dead on this phone (it
generalises to ≥ 12 GB devices as a runtime technique, not a file change). What fits: (d1) a **static slice** — all experts of n layers
requantised + repacked at load (n = 4: 351 MB anon, 268 M weights ≈ 0.5-0.8 s at R31's requant rate *estimated*); gain n × 4.1 × (40-60 µs) =
**−0.66 to −0.98 ms at n = 4**, −1.3 to −2.0 at n = 8 (700 MB). Bit-exact on the other layers; PPL = the copy's own numerics on those
layers. **Host probe (§4): the all-layer Q4_0 copy costs +0.56 % on the full set (9.6590 vs 9.6051; en.txt +0.64 %)** → the full copy
and the LRU form at h ≥ 0.15 fail the 0.086 % margin on 117 + head2 (116 + head2, 9.7800, leaves 0.17 % → h ≤ 0.3); a slice of ≈ δ/40 per
layer fits at **n ≤ 6** on 117 (+0.056 % at n = 4, +0.11 % at n = 8 fails; n = 8 fits only on 116), to be gated on the full set for the exact layer set. Tom ≈ 1 day: the head2 copy mechanism + `ggml_cpu_head2_q4_0_gemv`
already exist, the `MMID=3` per-expert dispatch needs a per-layer branch and a Q8_0 activation. (d2) the LRU form (R23's `EXPERT_ANON=16`
hit 64-78 % on the device with a memcpy and gained nothing *because the kernel was the same*) would be −4..−6.7 ms at h 0.5-0.7 **minus**
promotions: a Q3_K → Q4_0x4 requant of one expert ≈ 1.5 ms on the device (R31 rate), pure LRU wants 17-20/token → impossible; ≤ 2
promotions/token (window-LFU) → net **−1.5 to −2.5** *estimated*, +440-880 MB anon, thrashing-day negative (R29). Round 10 material, only
if (d1) proves the per-slot number. Exact fusions in `MUL_MAT_ID` (shexp into the routed barrier, gate+up in one op) save ≈ 80 barriers
≈ 0.3-0.5 ms *estimated* — listed under (f).

**(e) Printed vs steady.** The 978 ms of window 1-4 excess is page state (expert pages not resident at the run's start); the head copy is
built in TTFT (0.27 s), never in decode — moving it to load changes TTFT only; expert residency is closed (§7.4 #6: 3.1 GB does not fit the
cache the phone has). In-runtime and free: `MADV_WILLNEED` over the 436 MB dense set once per prompt (`GGML_DENSE_WILLNEED=1`, exact) —
recovers the dense share of the excess (≈ 130 ms per 200 tokens on an r8 day = **+0.1-0.2 tok/s printed**, 0 on a warm cache, more on r7
days). The other printed lever is the **1.63 ms outside the graph**: the bench runs `temp: 0` through the default chain (`llama_cpp_service.dart`
6713-6823: penalties 1.1/64 → top_k 40 → top_p 0.9 → temp 0 → greedy): top_k's 128-bucket partial sort over 100,352 candidates + the
candidate-array build (1.2 MB) ≈ **0.3-0.6 ms/token on the A78** *estimated*. Skipping top_k/top_p/min_p/temp when `temp ≤ 0` and keeping
penalties → argmax is the **same token (exact)**; llamadart change, Tom ≤ half a day. Scored on the printed−profile gap.

**(f) Graph crumbs (exact, fusion).** 1,855 nodes carry 1.6-1.7 ms of dispatch + 0.5 ms of small-op waits + 3.4 ms of small-op compute.
428 `ADD`s (≈ 200 are the per-layer expert-output sum chains), 154 `SCALE`, 117 `RMS_NORM` + 76 `MUL` (norm × weight; the 0.9 ms of `MUL`
= 12 µs per op is unexplained for 3 k-element ops — one host print of the `MUL` shapes may expose a 0.5 ms fusion). Summing the expert
outputs in one op in the same order and fusing norm × weight (two multiplies, same order) removes ≈ 300-400 nodes → **−0.5 to −1.0 ms**
*estimated*, bit-exact if the order is kept (PPL identity is the check). Tom 1-2 days in `llama-model.cpp` + one CPU op.

## 3. Ranked

| # | lever | exact? | Δ ms/token at 2,112 (*estimated* unless noted) | PPL | Tom | generalises? | Clyde's likely objection |
|---|---|---|---|---|---|---|---|
| 1 | (c) `GGML_MM_CHUNK=16` dense + mmid chunk grain | **yes** | **−0.7 to −1.4** (wait lines) | 0 | 0.5 d | any model, any CPU at decode | "R18 said imbalance; MMID=3 already chunks; the 1.8 may be the arbiter" — one ABBA settles it |
| 2 | (e) sampler bypass at temp ≤ 0 + dense `WILLNEED` | **yes** | **−0.3 to −0.6 outside the graph** (+0.1-0.2 tok/s printed from WILLNEED on r8 days) | 0 | 0.5 d | any app on llama.cpp sampling | "not the CPU module; the kit scores profiles" — the goal is printed tok/s |
| 3 | (d1) static Q4_0x4 expert copy, n = 4 layers | no (budgeted) | **−0.66 to −0.98** (n = 8: −1.3 to −2.0, 700 MB) | ≈ +0.056 % (n = 4) / +0.11 % (n = 8, fails on 117), from δ_full **+0.56 %** (full set, host) | 1 d | yes: "requant the compute-bound format into the device's byte-rate format" | "anon cache closed (R23), memory (R9), 0.086 % margin" — R23 closed a *memcpy* cache (same kernel); the new term is kernel class, device-proven in R31 |
| 4 | (f) graph crumbs: expert-sum + norm×weight fusion | yes | −0.5 to −1.0 | 0 | 1-2 d | any MoE graph on CPU | "two Tom days for < 1 ms" |
| 5 | (b) router F32 → Q8_0x4 copy | no | −0.3 to −0.5 | measure (routing flips) | 0.5 d | any MoE with an F32 router | "numerics for 0.4 ms" |
| 6 | (a) head M 1,024 | no | ≤ −0.1 | host probe §4 | 0 | — | not worth the margin (agreed) |
| 7 | (d2) LRU Q4_0x4 expert cache, ≤ 2 promotions/token | no | −1.5 to −2.5 net, +440-880 MB anon | h × 0.56 % → fails +2 % at h ≥ 0.15 on 117; ≤ 0.3 on 116 | 3+ d | yes | closed family, thrashing-day negative — round 10 only if #3 holds |
| — | (a) 2-bit/1-bit screens, (b) other dense copies, `ssm_out` skip, full expert copy (3.4 GB) | | dead (§2) | | | | |

Sum of the exact rows 1 + 2 + 4: **−1.5 to −3.0 ms**, of which 0.3-0.6 outside the graph → fault-free printed **≈ 18.8-19.4**. With #3 at
n = 4: 19.1-19.7; with n = 8: **19.5-20.1**. **20 printed is reachable on paper only with every exact crumb, the expert Q4_0 slice at ≈ 700 MB
anon, and a fault-free run** — and the +2 % budget has 0.086 % left for the slice's numerics (n = 8 ≈ +0.11 %: fails on 117, fits only paired with 116 whose give-back is +0.1..+0.4 ms) — **so 20 printed is not reachable inside
+2 % by this memo's arithmetic; 19.1-19.7 fault-free is.**

**My pick for round 9 (3 items): #1, #3 (n = 4, as a kernel-class cost item like R31's (i)), #2.** #1 and #2 are exact and cheap; #3 is the
only lever family that can reach 20 and its per-slot number must be device-measured before anything larger is designed.

## 4. Host probes run for this memo (`lab/logs/r31/`)

- `head2m/ppl60-s117-q4_0-{2048,1024,512}.log`: 117 + Q4_0r screen at M 2,048 / 1,024 / 512, 60 chunks: **10.0069 / 10.0060 / 10.0128** (M 1,024 −0.01 %, M 512 +0.06 % vs M 2,048; *host*). M 1,024 is
  PPL-free on this subset for ≤ 0.1 ms — a full-set gate (37 min) is owed before it enters a kit as the default M; M 512 eats the margin.
- `exps_q4_0/`: the served file with every `*_exps` dequantised and re-quantised to Q4_0 (`lab/tools/exps_requant.py`, gguf-py reference
  quantiser = `quantize_row_q4_0_ref`, all other tensors byte-identical; host artefact, never served) — the PPL a runtime Q4_0x4 expert copy
  would have. **Full set 9.6590 = +0.56 % vs 9.6051; en.txt 11.1710 = +0.64 %** (*host*; chunk 60 read +0.29 %, so the subset under-read this one
  by 2×: gate slices on the full set only). Double quantisation Q3_K → Q4_0 costs 5 × what the budget has left.

## 5. Round-9 pre-registration draft (Clyde freezes; all *estimated*; ABBA 88 tokens/side as r8; every number with its `mhz_blocks`)

**(i) `MM_CHUNK:64:16` on 117 + head2 (exact).** `MUL_MAT` wait 1.7-1.9 → **Δ −0.5 to −1.1**; `MUL_MAT_ID` wait Δ −0.1 to −0.3; `MUL_MAT`
compute 0 ± 0.3; `head` 0 ± 0.1; token −0.4 to −1.4. **Passes** ≤ −0.5 on the `MUL_MAT` wait column; **falsified** > −0.2 (the wait is not
chunk-tail; chunking closed) or `MUL_MAT` compute > +0.4 (prefetcher penalty; try 32). Identity: PPL chunks 1-3 and greedy 32 equal to the
default to the last digit (host, before the build).

**(ii) `EXPERT_Q4_LAYERS:0:4` on 117 + head2 (numerics, budgeted; copy made before `blk=0`, requant line ≤ 1.5 s, falsified > 3 s).**
`MUL_MAT_ID` compute **Δ −0.66 to −0.98** (16.4 slots × 40-60 µs); wait 0 ± 0.3; `MUL_MAT` 0 ± 0.3; k̄ / share / `adv` unchanged; majflt
fault-matched (B ≤ A + 10; anon +351 MB displaces page cache on thrashing days — do not quote the ratio with swapfree 0). **Passes** ≤ −0.6
(the 0.44 MB gemv is at the byte rate: the expert kernel class is device-proven); **falsified** > −0.4 (small-op gemv is not the head's
class; (d) closed on this device). Gate before the kit: full-set PPL of 117 + head2 + the 4-layer copy ≤ 9.797 (expected 9.7886 × 1.00056 ≈ **9.794**, margin 0.003 — gate it; if > 9.797 the slice is n = 2 (≈ 9.791) or the item is dropped).

**(iii) Printed overhead (exact): sampler bypass at `temp ≤ 0` + `GGML_DENSE_WILLNEED=1`.** Story 200: decode wall − (Σ windows + 7 × steady)
per token **1.63 → ≤ 1.2 ms** (Δ −0.3 to −0.6); greedy text identical; `MUL_MAT` window-1 excess ≤ +1.0 over steady (r8: +2.2). **Falsified**
if the gap stays > 1.4 (the overhead is stream/isolate, not sampling). Printed on the combo, same day as its 117 + head2 control.

Always-on: 117 + head2 + (i) + (iii) story 200 and official 32, with the day's control; expected steady **51.5-52.5** at 2,112 on an r8-like
day (hit band 51.0-53.0), printed fault-free 18.6-19.0. Closed by this memo unless the numbers above say otherwise: lower-bit head screens,
whole-tensor dense copies other than the router, head M < 2,048 as a speed lever.
