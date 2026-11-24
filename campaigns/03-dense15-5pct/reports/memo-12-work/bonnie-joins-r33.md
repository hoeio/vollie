# Bonnie, memo-12 work, round 33: how long a dense prefill is on the phone, and how fast a run uses up its thermal headroom

2026-11-24. Labels: *device* (phone), *host*, *estimated* (arithmetic), *source*. No web search (nothing is stated
about upstream's current state). R75's report is still not in the repository. Estimates use the coupling term
0.17 (band 0.14 to 0.25) and state C with its clock state, as the lead ruled after r32.

Script (same folder, new): `bonnie_heat_r33.py <repo root>` (committed kit JSONs only; seconds). It prints
campaign 2's prefill against decode and the headroom change over a run.

## 0. Answer first

1. **A dense prefill of the official prompt takes 6 to 14 s on Yi, 8 to 18 s on Llama-2 and 6.5 to 15 s on
   DeepSeek** (*estimated*; four threads on their cores). The low end is full clocks with the kernel count taken
   as it is; the high end is the 1,555 cap with a 1.4 allowance taken from campaign 2's device prefills (§1.3).
   Centres at full clocks: **7.6 / 9.8 / 8.0 s**.
2. **The prefill is compute-bound, so it is the one part of a run that the clock cap hits in full.** Its time
   goes as 1 ÷ C: 1.65 times longer at the 1,555 cap than at full clocks. The decode's body loses 0 to 6 % over
   the same range (r32).
3. **A whole claim run is then 11 to 15 s of load on Yi, 13 to 18 s on Llama-2 and 12 to 16 s on DeepSeek at
   stock decode speed** (*estimated*, full clocks to first cap step), all of it on four threads.
4. **Headroom on the phone** (*device*, campaign 2, three threads, 56 ungated runs that record it before and
   after): a first run takes it from 0.75 to 0.80 in 6.3 s, the run straight after from 0.80 to 0.95 in 5.0 s.
   Between 0.7 and 0.8 the median rise is 0.036 per second of load, with a wide spread (0.00 to 0.06). The
   reading saturates near 1.
5. **Put together with r32's chains** (cap lowered by 11 s of continuous load in 22 % of chains and by 16 s in
   52 %, from a start near 0.75): a gated start at 0.65 buys about 3 to 5 s, and a fourth thread costs about a
   fifth of the time. **P(the cap is lowered before the decode ends), per subject: Yi 0.3, Llama-2 0.45,
   DeepSeek 0.35** (*estimated*; r32's block figure was 0.35). Llama-2 is the exposed subject because its prompt
   is the longest in tokens and its body the largest.
6. **What the prefill costs the claim is not its seconds (they are outside the metric) but the heat it leaves
   at decode start.** Two existing ideas gain a second price from this, neither new: a GPU prefill (Ebony r5)
   moves 6 to 18 s of four-core load off the CPU before the timed part, and the kit's "prefix reuse: off" row
   (r28 row 39) is what makes every scored run pay the prefill.
7. **A check of the kernel count against the phone:** the count predicts a prefill-to-decode time ratio per
   token of 0.37 for campaign 2's three-thread runs; the phone shows 0.46 (Granite) and 0.58 (OLMoE) (*device*).
   Both models are sparse and have reasons of their own to be above the count. It is the source of the 1.0 to
   1.4 allowance, and it is the weakest number in this memo.

## 1. The prefill

### 1.1 What a prefill batch costs per position (*source*)

KleidiAI v1.31.0, `kai_matmul_clamp_f32_qsi8d32p4x4_qsi4c32p4x4_16x4_neon_dotprod_asm.S` (the entry the A78 takes
for a batch; read today):
- The 16-row loop (labels 4 to 7) is four copies of the four-row group that r5 counted as the tile: for each 64
  bytes of weights it runs the same loads, 32 SDOT, and the same float steps once per group of four positions,
  re-reading the weight block from the first-level cache for each group.
- So sixteen positions cost four tiles: **65 ÷ 4 = 16.25 issue slots per position per 128 weights against 25 for a
  single decode's gemv: 0.65 × a gemv per position** (r5's figure, now read for the full 16-row loop and not only
  the tail).
- The weights of a matrix pass through memory once per batch, not once per position. For a 186-token prompt
  the bus carries 3.6 GB once: 0.14 s (*estimated*, at 26 GB/s).
- A prompt of 167 or 186 tokens leaves a tail of 7 or 10 positions, run as padded groups of four: under 2 %.

### 1.2 Time (*estimated*)

    prefill ≈ tokens × 0.65 × body bytes ÷ C  +  bus pass  +  attention  +  the rest

| | Yi | Llama-2 | DeepSeek |
|---|---|---|---|
| prompt tokens with the wrapper | 167 (*host*, dump) | 186 (*host*, dump) | about 160 (147 before the wrapper, *host*; the wrapper and the BOS question of r28 row 40 are open) |
| body bytes per token (R72 L4) | 3.11 GB | 3.64 GB | 3.41 GB |
| tokens × 0.65 × bytes | 338 GB | 440 GB | 355 GB |
| at full clocks, C 57.6 | 5.9 s | 7.6 s | 6.2 s |
| at a 2,150 cap, C 49.6 | 6.8 s | 8.9 s | 7.1 s |
| at the 1,555 cap, C 35.0 | 9.6 s | 12.6 s | 10.1 s |
| fourth thread off the X1 at full clocks, C 39.0 | 8.7 s | 11.3 s | 9.1 s |
| bus pass, attention (Ebony r9: 0.1 to 0.3 s), norms and the rest | + 0.4 to 0.7 s | + 0.5 to 0.8 s | + 0.4 to 0.7 s |

- C is r32's: 11.3 GB/s per A78 at 2,227 MHz, the X1 at 1.44 per MHz, all on-core. It is a gemv rate from the
  probe's kernel; the batch kernel's 0.65 is a count, not a timing.
- In tokens per second: Llama-2 24 at full clocks, 15 at the 1,555 cap, before the allowance of §1.3.
- With the allowance (× 1.0 to 1.4) and the additions: **Yi 6 to 14 s, Llama-2 8 to 18 s, DeepSeek 6.5 to 15 s.**
  Ebony's range (6 to 15 s, r4 and r11) is confirmed with a longer top for Llama-2.

### 1.3 The count against campaign 2's prefills (*device*)

`official32` runs with the pool on that started and ended at the maximum cap, without major faults in the way
(the first run after a load reads the file and is excluded):

| model, arm | runs | prompt tokens | prefill | prefill, ms per token | decode, ms per token | ratio |
|---|---|---|---|---|---|---|
| Granite, num | 32 | 177 | 3.28 s | 18.5 | 40.6 | 0.46 (0.45 to 0.46) |
| OLMoE, off | 41 | 156 | 3.40 s | 21.8 | 37.8 | 0.58 (0.57 to 0.58) |
| OLMoE, num | 12 | 156 | 3.05 s | 19.6 | 33.9 | 0.58 |

- What the count predicts for three threads at full clocks, if a prefill position and a decode token touched
  the same bytes: 0.65 × (decode's bus-bound rate, about 26 GB/s) ÷ (C of two A78 and the X1, 46.3) = **0.37**
  (*estimated*).
- The phone is 1.25 times that on Granite and 1.6 times on OLMoE. Reasons that belong to those models and not
  to a dense one: OLMoE's prefill sends each expert a fraction of the batch (about 20 positions of 156 per
  expert), so more of the work is tails and small calls, and it reads all 64 experts where a decode token reads
  8; Granite's state-space layers are not matmul batches at all; both decodes are three threads on files a fifth
  the size, where fixed per-token work is a larger share.
- Reasons that would carry over: activation packing per position, the attention at 150 to 190 queries, and the
  count being a floor.
- **Allowance for a dense prefill: × 1.0 to 1.4, centre 1.2.** It is a judgement between the two device ratios
  and the count. The first dense kit replaces it: `ttft_ms` and `prompt_tokens` are existing fields.

### 1.4 The run as a whole (*estimated*; stock decode; 32 tokens)

| | Yi | Llama-2 | DeepSeek |
|---|---|---|---|
| prefill, centre (full clocks × 1.2 + additions) | 7.6 s | 9.8 s | 8.0 s |
| decode, 32 tokens at stock (6.9 / 6.2 / 6.2 tok/s) | 4.6 s | 5.2 s | 5.2 s |
| **load per run, centre** | **12 s** | **15 s** | **13 s** |
| range (full clocks × 1.0 to first cap step × 1.4) | 11 to 15 s | 13 to 18 s | 12 to 16 s |
| with a method at 12 tok/s the decode is 2.7 s | 10 s | 12.5 s | 10.7 s |

A faster decode shortens the run by 2 to 2.5 s, which is the only way the method itself changes the heat.

## 2. Headroom per second of load (r32 next question 3)

### 2.1 Data (*device*)

Campaign 2, sessions without power saving: 56 runs record the thermal headroom at the start (`heat`) and at the
end (`heat_after`). All are ungated `official32` runs on three threads (the two warm-ups at the head of each
block). Gated runs record only the start. The reading is Android's forecast headroom: higher is hotter, 1.0 is
the throttling point; the kit's gate is 0.65.

| | runs | start | end or change | run length | per second of load (median; 10th to 90th) |
|---|---|---|---|---|---|
| all | 56 | 0.76 | +0.070 | 5.4 s | +0.008 (−0.015 to +0.057) |
| first warm-up of a block | 18 | 0.75 | 0.80 | 6.3 s | +0.005 |
| second, straight after | 18 | 0.80 | 0.95 | 5.0 s | +0.025 |
| start 0.6 to 0.7 | 2 | – | – | – | +0.057 |
| start 0.7 to 0.8 | 34 | – | cap lowered by the end in 12 | – | +0.036 (0.000 to +0.062) |
| start 0.8 to 0.9 | 7 | – | cap lowered in 7 | – | +0.010 |
| start above 0.9 | 13 | – | cap lowered in 7 | – | −0.011 |

- **The rise is not a line.** It is slow or absent in the first seconds of a block (the first warm-up also
  loads the model, and the reading lags), fast between 0.7 and 0.9, and flat or falling above 0.9, where the
  phone has already lowered the cap. Per-second figures from single runs spread from 0 to 0.06.
- **Two warm-ups back to back go from 0.75 to 0.95 in about 11 s:** 0.018 per second on average, on three
  threads.
- Only two runs start under 0.7, so the stretch a gated run starts in is the one with least data.

### 2.2 From a gated start (*estimated*)

| step | value |
|---|---|
| from 0.65 to 0.75, at 0.02 to 0.036 per second | 3 to 5 s |
| from 0.75, r32's chains: cap lowered in 22 % by 11 s and in 52 % by 16 s (three threads) | – |
| so from a gated start, three threads | 22 % by about 15 s, 52 % by about 20 s |
| four threads against three (one more A78 at full load; I take power as + 20 to 30 %) | times ÷ 1.25: **22 % by about 12 s, 52 % by about 16 s** |

Against §1.4's run lengths:

| | Yi | Llama-2 | DeepSeek |
|---|---|---|---|
| load per run, centre | 12 s | 15 s | 13 s |
| **P(cap lowered before the decode ends)** | **0.3** (0.15 to 0.45) | **0.45** (0.25 to 0.6) | **0.35** (0.2 to 0.5) |
| the same with a 12 tok/s method (run 2 to 2.5 s shorter) | 0.2 | 0.3 | 0.25 |

- Device check that bounds this from below: gated runs that started at the maximum with headroom under 0.70
  never had the cap lowered in up to 8 s of three-thread load (0 of 280, r32).
- The likely step when it comes is 2,150 MHz: the decode's body loses under 3 % (r32), the width-2 ratio stays
  at its full-clock value. **What a lowered cap changes most is the prefill of the next run**, and the gate's
  wait of a minute or more restores the cap in 18 of 20 cases (r32).
- The four-thread factor is a guess. No committed run has four pooled workers (r31), so no record shows how
  four threads heat.

### 2.3 What follows

- **For the claim's median of ten:** if the cap drops in 3 to 5 runs of ten and each such run loses up to 3 %
  on its last tokens, the median moves by under 1 %. The cap is a second-order risk for the claim number while
  it stays at its first step. It becomes first-order only with the second step (1,555), where r31's table
  applies, and r32 saw that level 4 times in 384 starts.
- **For width 2:** unchanged from r32. The ratio is near its full-clock value in a gated block.
- **For the protocol:** the 90 s gate limit of the kit (r22 row 15) against the 180 s of the probes matters more
  for a dense subject than it did for campaign 2, because each run adds two to three times the heat. That is
  decision 8 of r22 §1.6, and this section is its price: a gate that times out starts the next run near 0.7 to
  0.75, which is where the ungated chains started.

## 3. Lists (additions only)

| # | test | where | cost | closes when |
|---|---|---|---|---|
| B-T33a | the first dense kit's `ttft_ms` ÷ `prompt_tokens` against §1.2 at the cap it printed | device, free (existing fields) | 0 | more than 1.4 × the count → the batch kernel is slower on the phone than counted, and every prefill-heat figure here is low |
| B-T33b | `heat_after` recorded on gated runs too (the kit reads it only on ungated ones) | device, one line in the kit | 0 | gives the headroom rise from 0.65 on four threads directly and replaces §2.2 |
| B-T32a | unchanged: the cap sampled once a second inside the run | – | – | – |

No try item. Ebony's GPU prefill (her r5) and a decode-only heat budget are the levers this prices; I add no new
one.

## 4. Next questions, ranked

| # | question | state |
|---|---|---|
| 1 | **The gate on a dense block:** from campaign 2's gate series (`gate`, `cool_ms`, and the probes' `series` of headroom every 5 s), how fast does headroom fall while waiting, from 0.95 and from 0.8, and how often would a 90 s limit time out after a 12 to 15 s dense run? It prices r22's decision 8 | committed JSONs; **the top item I can do now** |
| 2 | R75 into r18's draft, r25's script, r27 §2.1, r29 §1.2, r32 §1.4 | waits for the report |
| 3 | Prefill on fewer or other cores: does a three-thread prefill (A78 only, the X1 kept cool) leave the decode a better clock than a four-thread one, at 1.3 times the prefill time? Arithmetic on §1.2 and §2 | small; second in line |
| 4 | DeepSeek's lookup arms and break-even line | waits for the owner on r28 row 40, then Tom |
| 5 | Llama-2's and DeepSeek's stock shifts against r11 | waits for Tom's five runs |
| 6 | r15's O and the second position's attention against Tom's host decode and KV screen | waits for Tom |

## 5. Sources (2026-11-24)

Repository: `lab/llama.cpp-c1h6/build-kai/_deps/kleidiai-src/kai/ukernels/matmul/matmul_clamp_f32_qsi8d32p_qsi4c32p/kai_matmul_clamp_f32_qsi8d32p4x4_qsi4c32p4x4_16x4_neon_dotprod_asm.S`
(labels 3 to 7 and 14 to 18); the campaign-2 kit JSONs (`runs[].ttft_ms`, `prompt_tokens`, `decode_ms`, `tokens`,
`majflt`, `heat`, `heat_after`, `wall_ms`, `cap_before`, `cap_after`, `threads`, `env`); `R72-result.md` §5; my r5
(the tile count), r22, r28, r31, r32; Ebony's r4, r5, r9, r11 as cited. No external source this round.
