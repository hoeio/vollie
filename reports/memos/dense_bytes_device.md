# Dense-byte levers re-priced against device kernel costs (Bonnie, after R29; 2026-11-06)

Question (Clyde): R29 showed the stock head bandwidth-bound and clock-flat on the device while the head2 Q2_K
kernels run at 4.9-5.9 × the M4's per-row cost. Re-price what is left of the dense-byte levers (repacked Q4_0
screen at M 2,048; the §8 `ssm_out` re-laid copy) with that rule, and draft round 8. Labels: *device* = R29
(`reports/device/r7_2026-11-06.json`, 1,555 MHz unless stated), *host* = M4 (`lab/logs/r27/head2/`), *estimated*
= arithmetic below. Nothing here was measured today; `results.csv` gets no row from this memo.

## 1. The rule

*Device:* stock head op `head=` **6.5-7.0 ms** in every control window at 2,112 and 1,555 → 126.4 MB / 6.7 ms =
**18.9 GB/s**, bytes only. Head2 Q2_K/16384 **12.2 ms** fault-free (eio-0 windows) for 71.2 MB = 3.8 ms of bytes
→ 8.4 ms of kernel. *Host:* stock 1.73, Q2_K/8192 2.02, Q2_K/16384 2.50 (`head2_norepack_host.txt`, 3 threads).
**Rule: on the A78 at 1,555 MHz a Q2_K `vec_dot` + strided 1,260-B gather costs s = 4.9-5.9 × the M4 per row;
the stock Q6_K head only bounds s_stock ≤ 3.9** (its compute hides under 6.7 ms of bytes). Compute scales with
clock (×0.736 at 2,112 → s 3.6-4.3 / ≤ 2.9); bytes do not. Every runtime-requantised copy is priced as
op = max(bytes / 18.9 GB/s, M4 compute × s) + gather/select × s.

## 2. Repacked Q4_0 screen, M 2,048 (host: screen-only 1.17 repacked / 1.24 unrepacked; select + 2,048 exact rows 0.12)

Bytes: 100,352 × 864 B = 86.7 MB screen + 2,048 × 1,260 B = 2.6 MB rows = **89.3 MB → byte floor 4.72 ms**
(*estimated*, both clocks) = **−1.8 to −2.3 vs the stock 6.5-7.0: the best case is ≈ −2 ms, not ≤ 1 ms.**

| assumption for the Q4_0-repacked kernel | screen = max(4.59 bytes, 1.17 × s) | select + rows 0.12 × s (floor 0.14 bytes) | op | **Δ head vs 6.5-7.0** |
|---|---|---|---|---|
| **A: s = the Q2_K ratio** 4.9-5.9 (1,555) | 5.7-6.9 (compute-bound) | 0.6-0.7 | 6.3-7.6 | **−0.7 to +1.1** (1,555) |
| A at 2,112 (s 3.6-4.3) | 4.6-5.1 | 0.4-0.5 | 5.0-5.6 | −0.9 to −1.7 |
| **B: s ≤ the stock bound** 3.9 (1,555) | ≤ 4.6 → bytes, 4.59 | 0.15-0.47 | 4.7-5.1 | **−1.4 to −2.3** (1,555) |
| B at 2,112 (s ≤ 2.9) | 4.59 | 0.15-0.35 | 4.7-4.9 | −1.6 to −2.3 |

At 1,555 the two bands are disjoint, so one fault-matched ABBA decides: **Δ head > −0.7 falsifies B** (the Q4_0
dotprod kernel is as bad as Q2_K's on the A78; every requant screen is closed on this device); **Δ head ≤ −1.2
falsifies A** (the kernel runs at the byte rate; the lever is the byte floor). At 2,112 the bands overlap
(−0.9..−1.7 vs −1.6..−2.3) and anything ≤ −0.9 is a working lever regardless. Which is likelier: the repacked
Q4_0 gemv is ggml's best-optimised arm64 path (4×8 interleave, `sdot`), the Q2_K kernel is scale-unpacking-heavy,
and the stock Q6_K is already ≤ 3.9 — I expect B for the screen and A for the gather (strided page-cache reads
do not get faster with a better kernel), i.e. central **≈ −1.5 at 1,555**. Caveat on the host input: 1.17 also
appears as the stock head's A side in Tom's repack-build ABBA (`abba_head2_parse.txt`), so the screen-only
number carries a ±0.07 build-path ambiguity; inside the band.

*Requant on the device (estimated, scaled):* Q2_K of 154 M weights took 5,211-5,521 ms *device* vs 1.2 s *host*
(×4.4). `quantize_row_q4_0` is a single max-abs pass (≈ 5 ops/element) against Q2_K's iterative scale search
(≈ 50-100) → host ≈ 0.1-0.15 s, device **≈ 0.5-0.7 s + 0.1-0.2 s for the 4×8 repack = 0.6-0.9 s**; falsifier: Tom
times the host requant line, × 4.4; > 2 s on the device means the cheap quantiser is not the one being run.
Still inside TTFT unless moved to load (R29 open item 3).

*Serving caveat, not a measurement one:* side B reads 2,048 random rows ≈ 2,700 pages/token = 8.7 % of the Q6_K
tensor; the rest becomes evictable (r7 (iii)/(iv): majflt B/A 3.25-3.97 with swapfree 0, and `MUL_MAT` moved
+7 ms beyond the head op). On a thrashing day the lever's served value is negative unless the exact rows stop
being file-backed; on a clean day (A ≤ 10 majflt/token) the cost is 0. Score (i) only fault-matched.

## 3. PPL (host, full set, log-additive; the R28 rule held to the third digit on both 116 pairs)

Q4_0/2048 alone 9.6062 (×1.000115). **117 + Q4_0/2048 = 9.7869 × 1.000115 = 9.7880, margin 0.009 under 9.797**
(> 0.003: safe to pre-register). 116 + = 9.7798, margin 0.017. **118 + = 9.7963, margin 0.0007** — inside the
rule's own error, a coin flip: gate it on the full set as a bonus (deterministic, so legal if it lands ≤ 9.797),
do not pre-register it as the kit pair. 117 is the pair; 116 is the fallback if 117 + measures > 9.797.

## 4. §8 `ssm_out` column skip, re-priced (input-major re-laid Q4_0 copy, 95.6 → 51.6 MB at thr 0.01, kept 54 %)

Stock *device* cost: 95.6 MB at 18.9-24.2 GB/s = **3.9-5.1 ms** (both clocks; compute hidden, ≤ 1.0-1.6 ms by R27's
25-31 % share). Re-laid byte floor **2.1-2.7 ms**. The skip kernel is an axpy over kept channels (y[1,536] += x_k ·
W[:,k], 48 Q4_0 blocks per channel, each with its own scale) — the per-(channel, block) scales force float
accumulation, so `sdot` is out. Instruction count per 32 MACs (A78, 2 NEON pipes, 3 cores, ideal issue):

| kernel | NEON instr / MAC | compute at 1,555 / 2,112 | op = max(bytes, compute) | **Δ vs stock 3.9-5.1** |
|---|---|---|---|---|
| fp32 axpy (the §8 design): unpack 2, sxtl 4+8, scvtf 8, fmla 8 | ≈ 0.94 (+0.5 ld/st) | 9.2 / 6.8 ms | 6.8-9.2 | **+2 to +5: dead** |
| fp16-accumulate axpy: unpack 2, sxtl 4, scvtf.h 4, fmla.h 4 | ≈ 0.44 | 4.3 / 3.2 | 3.2-4.3 | −1.9 to +0.4 |
| fp16 + `tbl` nibble→f16 LUT: unpack 2, tbl 4, fmla.h 4 | ≈ 0.31 | 3.1 / 2.3 | 2.5-3.1 | −1.0 to −2.4 (unbuilt) |
| stock repacked Q4_0 gemv, for scale | ≈ 0.08 | hidden under bytes | 3.9-5.1 | — |

Cross-check with the M4 rule instead of issue counts: a streaming axpy is not strided, so s is nearer the stock
bound (≤ 3.9) than the gather's 4.9-5.9 — same conclusion, the fp32 kernel is 10-12 × the stock kernel per MAC and
5.5-6.5 × at 54 % of the MACs. **My −1.5 to −2.0 of §8 assumed the kernel ran at the byte rate; it cannot in fp32.**
What revives it is a ≤ 0.3 instr/MAC kernel with fp16 accumulation over ≈ 1,659 terms — a second numerics change on
top of the copy's requant (Q4_0 → fp → Q4_0 along outputs, double quantisation of a 4-bit tensor; PPL unmeasured, my
guess +0.1 to +0.5 %, and the margin under 9.797 after 117 + Q4_0/2048 is 0.09 %). Requant of 170 M weights on the
device: like §2 ≈ 0.7-1.0 s + a cache-blocked transpose of 36 × (3,072 × 1,536) ≈ 0.2-0.4 s → **≈ 1-1.4 s estimated**.
Before any Tom spec, in order: (a) a model-free host microbench of the three axpy variants vs the stock gemv
(instr/MAC, ms at the device's shapes) — 30 minutes, decides whether (b) is worth running; (b) `dense_colskip.py`
with the simulated re-laid copy, fp16 accumulation modelled, 60 then 565 chunks + en.txt (still owed from §8);
(c) only then `GGML_SSMOUT_COLSKIP`. The §8 60-chunk "−2.8 %" stays untrusted until (b).

## 5. Ranking (per printed token, 118 alone = 0; clean day at 2,112 ≈ 54-56 ms = 18 tok/s)

1. **Q4_0-repacked screen, M 2,048, on 117** (118 if the gate lands): head −1.4 to −2.3 (B) / −0.7 to +1.1 (A) at
   1,555, give-back +0.1 to +0.4 → net **−1.0 to −2.2 (B)**; PPL 9.788 est., margin 0.009; cost one Tom type switch
   + repack of the anon copy, requant ≈ 0.6-0.9 s est.; cheapest falsifier one fault-matched ABBA on `head`.
   Reaches 52-54.5 ms at 2,112 = 18.3-19.2 tok/s at best — not 20.
2. **Nothing under +2 %** (118 alone): what we hold if (i) falsifies B; zero risk, zero gain.
3. **`ssm_out` re-laid**: fp32 dead by arithmetic (+2 to +5); the fp16/LUT variant −1.0 to −2.4 only on paper, with
   two unpriced numerics changes and a kernel nobody has written. Below 2 until the microbench says otherwise.

My pick: 1, measured as a cost item first (it is the one experiment that prices every future requant-screen lever
on any device, so it pays even when it closes). Clyde will likely say: assumption A is the one R29 just measured
and the central at 1,555 is ≈ −0.5 after the gather — a round item for a lever that cannot reach 20 and is negative
on thrashing days. Fair; the answer is that (i) costs one ABBA and no serving change.

## 6. Round-8 pre-registration draft (Clyde freezes; all *estimated*; score on `head=`, not `MUL_MAT`)

**(i) Cost-only ABBA, 118 → 118 + Q4_0-repacked/2048** (≈ 9.7963 est., research row like r7 (iv); copy built before
`blk=0`, requant line expected 0.6-0.9 s, falsified > 2 s). Predictions, `head` B − A: at 1,555 **B −1.4 to −2.3 /
A −0.7 to +1.1**; at 2,112 **B −1.6 to −2.3 / A −0.9 to −1.7**. `MUL_MAT` = head Δ ± 0.5 only on a fault-matched
run; `MUL_MAT_ID` 0 ± 0.3; k̄/share/`adv` unchanged; majflt B/A 1.0-1.5 on a clean day (A ≤ 10/token), 2-4 on a
thrashing one (the Q6_K rows going cold, as r7). **Passes** = `head` B − A ≤ −1.2 ms at the run's clock, fault-matched
(B majflt/token ≤ A + 10). **Falsified (lever closed on this device, no (iii)):** `head` B − A > −0.7 ms at 1,555 MHz
(or > −0.9 at 2,112), fault-matched — the repacked Q4_0 screen is compute-bound on the A78 like the Q2_K one (per-row
ratio ≥ 4.9), so no runtime-requantised screen of any type beats the stock head here. Between −0.7 and −1.2: the lever
is real but < 1 ms net after the give-back; reported, not served. Unscorable if side A > 10 majflt/token on every block.
**(ii) Full-set gate, host, before any combo ABBA:** Q4_0/2048 alone = 9.6062 ± 0.001 (runtime == replay);
**117 + = 9.788 ± 0.002**, 116 + = 9.780 ± 0.002, 118 + = 9.796 ± 0.002 (reported, not pre-registered as legal);
en.txt alongside; greedy 32 identical to stock. Falsified if 117 + > 9.797 → the kit pair becomes 116 +.
**(iii) Only if (i) passes:** ABBA 118 → 117 + Q4_0r/2048 (118 + if (ii) lands ≤ 9.797): `head` as (i), `MUL_MAT_ID`
+0.1 to +0.4, net −1.0 to −2.2; token Δ the same ± 2.5 noise; story 200 only if (iii) runs. Every steady number with
its `mhz_blocks`; §7.9's clock-correction rule applies to the token line, the head op is clock-flat.
