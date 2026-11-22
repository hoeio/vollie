# memo-08 work file, Bonnie: the gap ledger, the physical K1 cost, lever prices and the path to 12

2026-11-22, bonnie (`claude-fable-5-2`). Draft for joint round 6 (memo-08). I do not commit; Annie commits.
Nothing in this file is measured. Every price is *estimated*. Quoted inputs carry their own label: *device*, *host*
or *published*. Script: [bonnie_path.py](bonnie_path.py), in this folder. It imports the shapes from
`memo-06-work/bonnie_price.py`, and every number below comes from it unless the text says it is a hand count.

Subjects: Yi-1.5-6B-Chat (GQA), Llama-2-7b-chat and DeepSeek-LLM-7B-chat (both MHA), from memo-06 §6 (`e582606`).

## 0. Conventions and inputs

These are the agreed conventions (memo-07 §8), with three additions that Ebony and I both make.

- steady = max(bytes ÷ r, weight-steps ÷ c) + O, and printed = 0.95 × 1000 ÷ steady.
- 12 printed means a steady time of 79.2 ms, which leaves **71.2 ms** for the body after O = 8 ms.
- c = 40.4 GB/s of Q4_0, which is 71.8 G weights/s on X1 + 2 A78. That puts the 3-core **weight line at 5.11 G**,
  and the 4-core line at 5.88 G (× 1.15).
- r is the R69 band, 24 (19 to 30) GB/s. On four cores it is r4 = 1.13 × r, also R69's band.
- Frontiers (*host*, 40 chunks; tom-split.md §2 and §5):
  - Yi: 0.42 uniform.
  - Llama-2: 0.40 with k and v dense, which skips 33.4 % of the body.
  - DeepSeek: 0.40 with k and v dense; with BOS it is +3.83 % at `-b 512` (clyde-r72-plan.md §5).
- Skipped bytes are 0.889 × s, because the scales are always read.
- **Added: KV read.** F16 KV is read at the mean decode position of the official run, 171 = 155 + 16. The 155-token
  prompt length comes from campaign 2's knowledge.md §6 ("155-token prompt"); it is the Granite count, and the
  subjects' tokenizers will differ by perhaps ±20 %. Ebony uses 190; the difference is 0.01 GB.
- **Added: head compute** at Q6_K = 1.4 × Q4_0 per weight (Ebony's L5 centre; my own is 1.6, see §5).
- **Added: attention compute** = 2 MACs per position, head and channel per layer, at a cost of 2 Q4_0 weight-steps per
  MAC. This is a guess, and it is under 2 % of W at 171 positions.
- **Residency** is the whole body plus the head, plus KV at n_ctx 512, plus the head screen when one is used. Untouched
  embedding rows are not counted. It is tested against R69's capacity steps: P(3.4 GB held) 0.35, P(4.2 GB held) 0.12
  (pre-registered).

## 1. The gap ledger: three numbers per subject that must all close

The first row of each subject is its state today: the host frontier, with K1 as built at the device k I derive in §2
(1.64 on X1 + 2 A78), and three cores.

| subject | bytes read per token (GB): stock / frontier | **bytes gap** at r 24 (r needed) | weight-steps (G): stock / frontier with K1 / with K1m | **compute gap**, 3 cores (4 cores) | **O allowed** for 12 at r 24 / r 30, best CPU stack |
|---|---|---|---|---|---|
| Yi-1.5-6B | 3.34 / 2.18 | **+28 %** (needs r 30.6) | 5.99 / 5.74 / 5.05 | K1 +12 % (−2 %); K1m −1 % (−14 %) | 5.3 / 19.3 ms |
| Llama-2-7B | 3.84 / 2.76 | **+62 %** (needs r 38.8) | 6.75 / 6.68 / 5.98 | K1 +31 % (+14 %); K1m +17 % (+2 %) | **−18.4** / 1.6 ms |
| DeepSeek-7B | 3.84 / 2.83 | **+66 %** (needs r 39.8) | 6.74 / 6.68 / 6.03 | K1 +31 % (+14 %); K1m +18 % (+3 %) | **−12.8** / 5.3 ms |

How to read the columns:
- Each gap is the share by which that line exceeds what 12 allows. A minus sign means the line is already under it.
- **Bytes gap at other r:** Yi +61 / +2 / −13 % at r 19 / 30 / 35; Llama-2 +104 / +29 / +11 %; DeepSeek +109 / +33 / +14 %.
- **Best CPU stack** (for the last column) is K1m, the fourth core, a two-stage head and KV Q8_0 (§3).
- **"O allowed"** is 79.2 ms minus the body time of that stack. A negative value means that at r 24 even O = 0
  does not reach 12.
- **KV included:** at 171 positions Yi reads 0.011 GB, Llama-2 0.090 and DeepSeek 0.084. At 512 positions these
  are 0.034, 0.268 and 0.252.

Reading:

1. **The bytes gap is the wall on every subject. The compute gap comes second, and O is not the problem.**
   - At R69's centre, r 24, the bytes line alone is above 71.2 ms on all three subjects, whatever k is.
   - On the MHA subjects, 12 needs r of about 39 GB/s with today's frontier. That is 76 % of the LPDDR5 peak of
     51.2 GB/s (*published*, 4 × 16-bit at 3200 MHz), and outside R69's 80 % band.
2. **The compute gap closes only with the fourth core and a better K1 (K1m, §2).**
   - Yi passes the compute line on 3 cores only with K1m.
   - The MHA subjects need 4 cores *and* K1m, and even then they sit 2 to 3 % over. Their k and v run at stock
     cost (1.07 G of dense weight-steps), and that sets the floor.
3. **O decides nothing at r 24, because the body alone is over the line.** At r 30 it decides the MHA subjects: their
   allowed O is 1.6 and 5.3 ms against a planned 8 (6 to 12).
   - So R72's barrier row (L6) matters only in the world where r turns out high.

## 2. K1's cost on the phone, from the core's issue rules (primary source)

The question was: what k is physically reachable on the X1 and the A78 with `sdot` only and no `i8mm`?

**Source.** I used the Arm Cortex-A78 Software Optimization Guide (*published*; I read the PDF's instruction
tables). I did not find an X1 guide. Its stand-in is the Cortex-X2 guide (*published*), which keeps X1's 4 × 128-bit
vector back end; the X1 values are therefore **by analogy** and are flagged as such.

| instruction | A78: pipes, throughput per cycle | X2 (for X1): pipes, throughput |
|---|---|---|
| SDOT | V0/V1, 2 | V0 to V3, 4 |
| SMLAL(2) (8b → 16h) | **V0 only, 1** | V0/V2, 2 |
| SCVTF (Q-form, F32) | **V0 only, 1/2** | V0/V2, 1 |
| FCVTL (F16 → F32) | **V0 only, 1/2** | V0/V2, 1 |
| USHR, SSHL, SSHLL | V1 only, 1 | V1/V3, 2 |
| AND, ADD/SUB, FMUL, FMLA | either, 2 | any, 4 |

**Floors.** I count per 128-row strip and 32-input block, with Tom's instruction mix (memo-05-work/tom-host.md §1),
and assign each instruction to its pipes. The keep fraction is d = 0.58.

| kernel | A78 cycles per weight | X1 cycles per weight (by analogy) | what binds |
|---|---|---|---|
| stock 4x4 gemv | 0.086 | 0.043 | the total |
| **K1 as built** | **0.179** | 0.089 | V0 on the A78: SMLAL plus SCVTF and FCVTL at half rate; V0/V2 on the X1 |
| **K1m**: K1 with SCVTF replaced by the exact magic-number convert, an integer ADD of 0x4B400000 and then an FSUB of 12582912.0f, both on any pipe | **0.152** | 0.076 | V0 on the A78 (SMLAL and FCVTL); the total on the X1 |

- **K1m is exact.** Each block's integer sum is under 2²² (at most 15 × 127 × 32 = 60,960), so the float result is
  bit-identical to SCVTF's.
- **SMLAL alone puts a floor of 0.125 cycles per weight on the A78.** No element-granular column kernel can go below
  that on the A78, because there SMLAL issues only on V0, and SDOT needs four columns per lane.
- **I counted the transpose-to-SDOT route as well.** It uses 8 ZIP per 64 weights, and comes to 0.170 on the A78 and
  0.086 on the X1. That is worse than K1m on both cores.

**From floors to k.** R69's pre-registered P2hot puts stock at 0.105 cycles per weight on the A78 and 0.078 on the X1,
which is well above the issue floors (most of all on the X1).

- On the A78, I take k between two cases: K1's cost scales with stock's efficiency (the ratio of the floors), or
  K1 adds stock's stall time on top.
- On the X1, I take k between two cases: K1's extra issue hides fully behind whatever holds stock at 0.078, or it
  adds.
- I combine the cores with dynamic chunks, so that their rates add.

| kernel | A78 k | X1 k | **3 cores (X1 + 2 A78)** | 4 cores |
|---|---|---|---|---|
| K1 as built | 1.89 to 2.08 | 1.15 to 1.60 | **1.64 (1.47 to 1.83)** | 1.71 |
| K1m | 1.63 to 1.77 | 1.00 to 1.42 | **1.43 (1.27 to 1.60)** | 1.48 |

What this says:
- **Tom's paired-pipe floor (1.58×) is not reachable on the A78.** The A78 issues SCVTF and FCVTL at half rate on the
  same pipe as SMLAL. Ebony's reading (1.96×) was closer: as built, the A78 floor is 2.08×.
- **The X1 is where K1 is cheap.** Stock leaves about half of the X1's vector issue idle, so K1's extra instructions
  partly fill that slack. This is *estimated*, and it rests on R69's P2hot X1 prediction (0.078).
- **P(k ≤ 1.55 on 3 cores):** about 0.15 as built and about 0.6 for K1m.
- **The k the memo-07 band assumed** (1.55 at the high end) **needs K1m.**
- **Comparative advantage.** Giving the stock matrices (k, v, head) to the A78s and K1 to the X1 would save about 3 %
  of compute on the MHA subjects (hand count). It needs the two kinds of op to run concurrently, which the graph does
  not do today. It is small, and I do not price it in the stacks.

## 3. Lever prices on each gap, per subject

All Δ values are against "frontier + K1 (k 1.64), 3 cores", at 171 positions:
- Bytes in GB per token, ΔB. The byte line is r × 71.2 ms = 1.71 GB at r 24.
- Compute in G weight-steps, ΔW. The line is 5.11 G.
- Fixed cost in ms, ΔO.

P is my probability that the lever delivers on the device and inside the budget.

| lever | Yi ΔB / ΔW | Llama-2 ΔB / ΔW | DeepSeek ΔB / ΔW | ΔO | band | P | numerics |
|---|---|---|---|---|---|---|---|
| **K1m** (§2) | 0 / −0.69 | 0 / −0.69 | 0 / −0.65 | 0 | k 1.27 to 1.60 | 0.6 | bit-identical |
| **4th core** | r × 1.13; c × 1.15 | same | same | 0 | MC uses Ebony's bands: c 1.10 (0.95 to 1.22), r 1.06 (0.95 to 1.20) | 0.68 for c4 ÷ c3 > 1.05 under that band; Ebony states 0.35 (R18 *device*: 4 threads lost); see §7.2 | bit-identical |
| **Two-stage head, bit-plane screen** (3 planes of x; top 2,048 rescored exactly at Q6_K) | −0.175 / −0.28 | −0.084 / −0.14 | **−0.285 / −0.45** | 0 | screen 0.3 × Q4_0 per weight (guess) | **0.35** | numerics; campaign 2's two-stage head was +0.01 %; campaign 1's low-rank screen failed (R43) |
| **KV Q8_0** (`-ctk/-ctv q8_0`) | −0.005 / 0 | −0.042 / 0 (−0.126 at 512) | −0.040 / 0 (−0.118 at 512) | 0 | – | 0.8 | numerics, small |
| **Per-group levels** (q/o, gate/up, down each at its own level; s +0.03 of the body at the same PPL) | −0.083 / −0.27 | −0.081 / −0.27 | −0.076 / −0.25 | 0 | Δs 0 to +0.06 | 0.5 | numerics; one budget |
| **Confidence fallback** (Ebony §1.2: s +0.10, 10 % of tokens re-run dense) | −0.28 / −0.91, before the re-run cost | −0.27 / −0.89 | −0.25 / −0.83 | 0 | – | 0.25 | numerics |
| **Bit-plane tiers** (Ebony §2.1: 0.30 of columns at half bytes) | −0.415 / **+1.16** | −0.405 / **+1.17** | −0.380 / **+1.10** | 0 | k × 1.22 (my count, §6.2) | 0.20 | numerics |
| **GPU lane** (dense rows only: k, v, head) | only Yi's dense share (0.29 GB) | the dense share 0.71 GB | 0.91 GB | **+3 (1.5 to 6)** | S 1.25 (0.95 to 1.6) | **0.35** | bit-identical if int-exact |
| **THP / big pages** (anonymous body) | r × 1.0 to 1.15 | same | same | 0 | – | 0.3 | none |
| **A55 helpers** | c × 0.95 to 1.12 | same | same | 0 | – | 0.25 | none |
| **Sub-4-bit served type we quantise ourselves** | see below | | | | | **0.05** | a new file |
| **Q4_0 re-quantised with an importance matrix** | budget +1 to +2 points, so s +0.03 to +0.05 | same | same | 0 | – | needs a rule reading | a new file |

How the larger rows are priced:

- **Two-stage head.**
  - I price it as a bit-plane screen, not a Q4_0 copy, so it adds only 33 MB (Yi) to 52 MB (DeepSeek) to residency
    rather than the 0.24 GB of the Q4_0 copy Ebony rejected (§1.9).
  - Whether the screen keeps the true top token in its 2,048 candidates, and what it does to PPL, are unknown. That
    is why P is 0.35.
  - DeepSeek is the subject where it matters: its head is 9 % of its bytes.
- **Sub-4-bit type.** The data (*published*, llama.cpp perplexity README, Llama-3-8B; PPL relative to plain Q4_0,
  bytes relative to Q4_0):
  - Q3_K_M with an importance matrix: PPL +0.5 %, bytes × 0.86.
  - IQ3_S: PPL +4.0 %, bytes × 0.79.
  - Q3_K_S: PPL +13.5 %; it fails F16 × 1.20.

  The costs:
  - The PPL comes out of the same +5 %: IQ3_S alone uses 80 % of it.
  - Its kernels cost 2× Q4_0 per weight (Q3_K, *device*, R41), and my count gives about 2× K1 for a 3-bit column
    kernel (bit-plane unpack about 0.36 V per weight, against 0.06).
  - Columns would no longer stack with it.

  **Only one row gains:** a byte-bound subject at low r with no column rule. That loses to the column rule on every
  subject. **Closed on arithmetic.** The one thing a smaller type still buys is residency: Llama-2 at Q3_K_M is
  3.23 GB and would fit the 3.4 step. That belongs to the freeze decision (memo-06 §5.3), not to speed.
- **IQ4_NL** has the same bytes as Q4_0 and −3.6 % PPL on Llama-3-8B (*published*), and stock has a 4x4 kernel for it.
  - But its codebook values reach ±127, so K1's int16 accumulation overflows after 2 columns.
  - A K1 for it needs widening at every column pair, about 2× SMLAL work.
  - **Closed for the column path.**
- **Q4_0 with an importance matrix.** This keeps the format, K1 and every kernel, and lowers PPL.
  - The gain on 7B models is not in the table I read, so the size (+1 to +2 points) is a guess.
  - It frees budget only if the +5 % is measured against plain Q4_0 rather than against the new file. That reading
    is the lead's and the owner's: it is a different file (CLAUDE.md: "a different file or model is proposed to the
    owner first").
  - Not in the stacks.
- **Prefetch and DRAM page policy.**
  - Software prefetch lost on the host (tom-host.md, `K1-tr*-pf`: k 2.09 to 2.19 against 1.53 to 1.58).
  - The DRAM controller's page policy is not reachable from an app. What the runtime can reach is the access pattern
    (streams per core: R69 X1) and the page size (THP above).
  - Ebony's devfreq readout (§2.4 of her file) is the right first step. I add no price beyond THP.

## 4. Stacks

### 4.1 Central printed tok/s at r 19 / 24 / 30 / 35 (residency assumed held)

| stack | Yi | Llama-2 | DeepSeek |
|---|---|---|---|
| columns + K1 | 7.7 / 9.6 / 10.8 / 10.8 | 6.2 / 7.7 / 9.4 / 9.4 | 6.1 / 7.5 / 9.3 / 9.4 |
| + K1m + 4th core | 8.7 / 10.8 / 13.2 / 13.3 | 7.0 / 8.7 / 10.6 / 11.5 | 6.8 / 8.5 / 10.4 / 11.4 |
| + head screen + KV Q8_0 (best CPU stack) | 9.4 / 11.6 / **14.0** / 14.0 | 7.3 / 9.0 / 11.1 / 11.7 | 7.6 / 9.5 / 11.6 / **12.2** |
| K1m + 4th core + GPU lane | 9.3 / 11.4 / 13.3 / 13.3 | 8.2 / 10.1 / **12.0** / 12.0 | 8.0 / 9.8 / 11.9 / 11.9 |
| best CPU stack + GPU + per-group levels | 9.5 / 11.6 / 14.0 / 14.0 | 8.8 / 10.8 / **12.6** / 12.6 | 9.2 / 11.3 / **13.1** / 13.1 |
| K1m + 4th core + tiers | 10.5 / 11.3 / 11.3 / 11.3 | 8.1 / 9.9 / 9.9 / 9.9 | 7.8 / 9.7 / 10.0 / 10.0 |
| everything except A55 (tiers and fallback in) | **12.3 / 13.7** / 13.7 / 13.7 | 10.6 / **12.4** / 12.4 / 12.4 | 11.0 / **12.9** / 12.9 / 12.9 |

The fourth core uses the conventions here (c × 1.15, r × 1.13). The Monte Carlo below uses Ebony's wider bands.

### 4.2 The cheapest stack whose central estimate reaches 12, per subject

Lever cost is in rough work units (K1m 1, head screen 2, per-group levels 1.5, GPU lane 5, tiers 3, fallback 2).

- **Yi:** the 4th core, the head screen and per-group levels. Central 12.1 at r 24 (with K1 as built, since at
  r 24 the token is byte-bound and k does not matter).
  - This rests on two untested levers (the head screen at P 0.35, per-group levels at P 0.5) and the 4th core
    (P 0.35).
  - It needs the 3.4 GB step. Yi's resident set is 3.36 GB, or 3.40 GB with the screen: no margin at all. Any
    runtime buffer that counts against C makes it fail.
- **Llama-2:** K1m, the 4th core, GPU lane, per-group levels, tiers and fallback. Central 12.2 at r 24. It costs
  about 12.5 units, and takes three untested numerics rules and the GPU lane.
- **DeepSeek:** K1m, the 4th core, head screen, GPU lane, tiers and fallback. Central 12.5 at r 24.
- **At r 30** the cheap stacks suffice:
  - Yi: K1m + 4th core, 13.2.
  - DeepSeek: K1m + 4th core + head screen + KV Q8_0, 11.6; it reaches 12.2 at r 35.
  - Llama-2: K1m + 4th core + GPU, 12.0.

**r at which the full stack** (K1m, 4th core, head screen, KV Q8_0, per-group levels, GPU) **reaches 12 centrally:**

| subject | with GPU | without GPU |
|---|---|---|
| Yi | r ≥ 24.8 | 23.8 |
| Llama-2 | r ≥ 27.1 | 31.7 |
| DeepSeek | r ≥ 25.8 | 30.2 |

For Yi the GPU lane makes things slightly worse: its dense share is small (0.29 GB), and the lane's fixed cost is 3 ms.

### 4.3 The stack with the highest P, and what P is

Method: a Monte Carlo over the pre-registered R69 bands, 20,000 draws.
- r, c_b, O, S, capacity C, plus Ebony's 4-core bands.
- K1 and K1m k uniform in the §2 bands.
- The lever P's of §3.
- Llama-2's 40 % holds on the full set with P 0.8, else 35 %.
- One phone: the capacity draw is shared by the three subjects.

| subject | highest-P stack | P(≥ 12) | P(≥ 12 \| resident) | P(resident) |
|---|---|---|---|---|
| Yi | everything (tiers and fallback in) | **0.10** | 0.28 | 0.35 (3.4 step) |
| Llama-2 | everything | **0.005** | 0.04 | 0.12 (4.2 step) |
| DeepSeek | everything | **0.008** | 0.06 | 0.12 |
| **all three** | everything | **0.005** | about 0.04 given 4.2 held | 0.12 |

- **The goal as amended (all three subjects) has P about 0.005 on today's knowledge, *estimated*.** Two factors
  dominate:
  - Residency. P(4.2 GB held) is 0.12, pre-registered, and both MHA subjects need it.
  - The read rate. P(r ≥ 30) is about 0.1 in R69's band.
- **Given residency, Yi is a real candidate (0.28). The MHA subjects are not (0.04 to 0.06).**
- **I disagree with Ebony's joint 0.10 to 0.15 (her §3).** The difference is in:
  - residency (she says it voids the prices, but leaves it out of her joint);
  - GPU optimistic and tiers taken as available;
  - tiers priced at constant k.

  My lever P's are close to hers (GPU 0.35 against 0.30, tiers 0.20, fallback 0.25), so the gap between 0.005 and
  her 0.10 to 0.15 is mostly residency and the kernel cost of tiers.
- **Where a single device number moves P most:**
  - R69's C: at 4.2 held, the joint P rises about eightfold.
  - R69's r: given residency and r ≥ 30, P(≥ 12) is 0.70 for Yi, 0.18 for Llama-2 and 0.22 for DeepSeek
    (*estimated*, same Monte Carlo).
  - Then k (D2 / R72 L1), which decides K1m's worth.

## 5. What must be measured, and my bands for R72 L1 to L7

**R69 (pre-registered, unchanged):**
- r, r4 and c_b.
- C. One gap: P1 maps files, but with repack on, llama.cpp allocates the body in the CPU buffer (`repack.cpp:5169`
  calls `ggml_backend_cpu_buffer_type()`, host memory), which is anonymous, not file-backed. Capacity for anonymous
  memory may differ under Android's low-memory killer and zram. **I propose an anonymous-arena arm for R72**
  (3.4 / 4.2 GB, touched each pass, same held rule).
- X2 (the GPU), P2col (k on the A78).

**R72 bands (my half; the lead merges with Ebony's §6).** Centre (80 % band), *estimated*.

| row | quantity | Bonnie | Ebony | note |
|---|---|---|---|---|
| L1 | K1 k, one A78, keep 0.58 to 0.60, hot | **2.0 (1.8 to 2.25)** | 1.9 (1.5 to 2.6) | from the SWOG floor (§2); cold 0.1 lower |
| L1 | K1 k, X1, hot | **1.35 (1.1 to 1.7)** | 1.7 (1.4 to 2.3) | we differ: stock's X1 slack |
| L1 | K1m k (if built), A78 / X1 | 1.70 (1.55 to 1.9) / 1.2 (1.0 to 1.5) | – | proposed new row |
| L2 | r4 cold, ggml threads, X1 + 3 A78 | 26 (20 to 33); ÷ r3 1.08 (0.95 to 1.25) | 26 (20 to 33); 1.06 | close |
| L3 | c4 ÷ c3 | 1.12 (0.98 to 1.24) | 1.10 (0.95 to 1.22) | perfect scaling from the R69 rates would be 1.28 |
| L4 | t_rule ÷ t_stock, cold, best set | Yi 0.66 (0.58 to 0.85); Llama-2 0.73 (0.65 to 0.90); DeepSeek 0.74 (0.65 to 0.90) | 0.70 / 0.75 / 0.75 | byte-bound at r 24 |
| L5 | Q6_K ÷ Q4_0x4 per weight, hot, 1 thread | 1.6 (1.25 to 2.0) | 1.4 (1.15 to 1.8) | Q6_K has no 4x4 repack: per-row horizontal adds, 6-bit unpack |
| L6 | empty barriers per token, 3 / 4 threads | 2.0 (1.0 to 4.5) / 2.4 (1.2 to 5.5) ms | 3 / 3.5 ms | about 450 nodes |
| L7 | head cold, 3 threads: Yi / Llama-2 / DeepSeek | 9 (7 to 12) / 4.5 (3.5 to 6) / 14 (11 to 19) ms | 11 / 5.6 / 18 ms | I use r and not 0.8 × r |

## 6. Challenge to Ebony's file (ebony-paths.md)

1. **§2.2, nibble-pair-rows K1: there is no mechanism on the binding pipe.**
   - On the A78, K1 binds on V0: SMLAL plus the half-rate SCVTF and FCVTL (§2). The unpack (AND, USHR) issues on V1
     and on either pipe, and V1 has room (0.136 of 0.152 cycles per weight in K1m).
   - Cutting the unpack from 8 to 5 ops therefore saves about nothing on the A78. On the X1 the saving is under 5 %.
   - Her κ 0.85 needs the unpack on the critical pipe. The lever that moves V0 is the exact convert (K1m).
   - Also: K1's TR-128 layout already packs two rows of one column per byte. "Feeding two SDOTs" does not apply to
     an element-granular column kernel, because SDOT needs four columns per lane.
   - I propose that H1 tests K1m in place of the nibble-pair layout, or both, with the same identity rule.
2. **§2.1, tiers: compute is not "counted as full, conservative". It goes up.**
   - Reading both planes of a full column needs a merge: (lo & 0x33) | ((hi & 0x33) << 2), and the mirror for the
     odd rows. That is about 16 more ops per 128 weights on the full columns, and about 10 more on the middle ones.
   - On the A78 the kernel then binds on the total, not on V0: 0.190 against 0.152 cycles per weight, which is
     k × 1.22–1.25 (my count, *estimated*).
   - With that, tiers lift Yi at r 24 from 10.8 to 11.3 on four cores, and *lower* it at r 30 (13.2 → 11.3).
   - Her §3 rows with tiers are 1 to 3 tok/s high at r ≥ 30, and about 1 high at r 24.
   - Tiers are a low-r lever only. The pass rule of H2 should add a kernel line: tier k ÷ K1 k ≤ 1.15 on the host
     count, else close.
3. **§3, joint P 0.10 to 0.15:** see §4.3. Residency alone caps the MHA subjects at 0.12. Her own §5.3 says this, but
   the joint does not use it.
4. **§1.3, the GPU lane on Yi.** Yi's dense rows (k, v and head) are 0.29 GB, so a GPU that may read only dense rows
   can take at most 13 % of Yi's bytes.
   - At the lane's fixed cost (her og 4 ms, my 3), Yi gains little or nothing: in my model 11.4 against 10.8 at r 24
     on four cores, and the full stack is no better with the GPU than without.
   - Her "+GPUc" row for Yi (11.2 / 13.0) needs the GPU to read sparse rows too. That is a K1-layout GPU kernel,
     which is not priced.
5. **§1.9, the head.** I agree that a Q4_0 screen copy is wrong for residency. A bit-plane screen (3 planes of x
   against sign-bit weights, then exact rescoring) is 33 to 52 MB and about 0.3 × Q4_0 per weight, so the R29
   compute-bound objection is weaker. It is still unproven (R43). I price it at P 0.35 and propose a host test (T4).
6. **Agreed:** speculation is killed (§1.1), A55 is killed (§1.4), and THP is a one-cell probe (§1.5). I add that the
   body is *anonymous* under repack, so the THP cell is the relevant arm and file THP does not matter.

## 7. Answers to Ebony's challenges (her §5)

1. **KV:** accepted and included (§0, §1). At 171 positions it adds 2.3 % (MHA) and 0.5 % (Yi) of bytes, and at 512
   it adds 7 %. KV Q8_0 is priced.
2. **4th core without a prior:** accepted. The Monte Carlo uses her bands. Central rows keep the agreed convention,
   and are labelled.
   - Her two numbers do not agree with each other. Her band, c4 ÷ c3 1.10 (0.95 to 1.22), puts P(c4 ÷ c3 > 1.05)
     at 0.68 (lognormal fit), not her stated P(helps) 0.35.
   - One of the two should move before R72's pre-registration. I keep her band, since R18's loss was on Granite's
     barrier-heavy graph.
3. **Residency first:** accepted, and carried as a gate in every P.
   - Correction to the file sizes used for residency: untouched embedding rows are not resident. Yi is 3.36 GB with
     KV at 512, not 3.48.
   - Under repack the body is anonymous memory, which needs its own capacity arm (§5).
4. **Head at its type's cost:** accepted, at 1.4 (her centre). I think 1.6; R72 L5 decides.
5. **O per subject:** O stays 8 (6 to 12) in the prices. L6 is my 2.0 ms of empty barriers. At r 24, O is not
   binding (§1).
6. **GPU on dense rows at full bytes:** accepted. The model caps the GPU's share at the dense bytes (k, v, head), P
   0.35. That is a little above her 0.30, because R69's X2 decides the first half of it.
7. **Which line binds:** accepted (§1, reading 1). At r 24 the byte line binds on every subject, so R72's L1 is not a
   speed result at r 24. It matters only above about r 28 (Yi) and r 30 (MHA).

## 8. Joint lists (my proposal; Ebony to confirm or amend)

**Test list (host, Tom), ordered by information value**

| # | test | why first | pass / kill (written before results) |
|---|---|---|---|
| T1 | **K1m**: K1 with the magic-number convert; `memcmp` identity against the stock 4x4 at 4096 and 11008, keep 0.58 / 0.60; host k. Also emit the aarch64 assembly built with `-mcpu=cortex-a78` (Android NDK clang) and count V0-only µops per weight. If `llvm-mca` is available, also run it with `-mcpu=neoverse-n1` as the A78 proxy and `-mcpu=neoverse-v1` as the X1 proxy (V1 is the X1's server sibling) | the only lever that moves K1's binding pipe; it costs one code change | identity all 1; V0 µops per weight ≤ 0.155 at d 0.58 in the count; else close K1m |
| T2 | **Full-set checks** before any lever round: Llama-2 at 40 % with k, v dense; DeepSeek with BOS, at both settings (already in the plan) | Llama-2's margin is 0.5 point | as memo-06 §6 |
| T3 | **Per-group levels**: the q/o, gate/up and down inputs at separate levels (grid 30 / 40 / 50 / 60 %, with k and v dense on the MHA subjects), on Yi and Llama-2, 40 chunks, `-b 512` and `-ub 1` | the cheapest byte lever left inside the one budget; E1 showed `down` alone tolerates 60 % | skipped share at the best cell inside +4.5 % at both settings is ≥ frontier + 0.03 on both models; else close |
| T4 | **Bit-plane head screen**: recall of the greedy top-1 within the screen's top 2,048, on 2,000 wikitext positions; PPL at both settings; Yi and DeepSeek | DeepSeek's head is 9 % of its bytes | recall ≥ 0.999 and PPL ≤ +0.1 % on both; else close |
| T5 | **KV Q8_0** at both settings, Llama-2 and DeepSeek (flags only) | cheap; frees 0.04 to 0.13 GB | ≤ +0.1 % on both |
| T6 | Ebony's H2 (tiers) with the kernel line of §6.2 added; H3 (confidence fallback) | the low-r levers | as hers, plus tier k ÷ K1 k ≤ 1.15 in the count |
| T7 | Q4_0 with an importance matrix (wikitext-free calibration text), PPL against plain Q4_0, Yi and Llama-2, 40 chunks | information for the owner's rule reading only | report only |

**Try list (device)**

| # | try | where |
|---|---|---|
| D1 | R69 as pre-registered (r, r4, c_b, C, X2, P2col) | R69 |
| D2 | R72 L1 to L7 with the bands of §5; **add K1m** to D2s if T1 passes | R72 |
| D3 | **Anonymous-arena capacity arm** (3.4 and 4.2 GB, touched every pass, same held rule as P1) | R72 rider |
| D4 | THP cell (Ebony's D2) and the devfreq readout (her D3) | R72 riders |
| D5 | `clinfo` extension dump (her D1). The GPU-lane prototype only after R69's X2 and D5 | R72 rider, then a kit |

## 9. Where I may be wrong

- **The X1 pipe table is taken from the X2 guide by analogy.** If the X1 issues SMLAL on one pipe only, X1 k rises
  by about 0.3 and the 3-core K1 band moves up by about 0.15.
- **The X1's k rests on R69's own P2hot prediction (0.078).** If stock on the X1 runs near its issue floor (0.043),
  X1 k becomes the floor ratio, 2.1 as built and 1.8 for K1m. The 3-core k would then be 1.9 and 1.7.
- **Guesses:**
  - the head screen's cost (0.3) and P (0.35);
  - per-group levels at +0.03;
  - attention cost at 2 per MAC;
  - the GPU lane's 3 ms fixed cost and its compute share;
  - the importance-matrix gain;
  - the tier merge count (a hand count, not compiled).
- **max(bytes, compute) assumes perfect overlap.** If the two partly add, every price here is high, most of all near
  the crossover (Yi at r 30).
- **Official prompt length** is the Granite count (155). Each subject's tokenizer changes the KV term by perhaps
  ±20 %, which is about ±0.02 GB on the MHA subjects.

## Sources read this round

- Arm, Cortex-A78 Core Software Optimization Guide (*published*, PDF; ASIMD integer and FP tables, pipeline
  overview), https://documentation-service.arm.com/static/5f15976320b7cf4bc5246c21
- Arm, Cortex-X2 Core Software Optimization Guide (*published*, PDF; used as the X1 stand-in),
  https://documentation-service.arm.com/static/60d346c7677cf7536a55bae6
- llama.cpp `tools/perplexity/README.md` at the base `620d2225c` (Llama-3-8B per-type PPL and size table, *published*).
- llama.cpp `ggml/src/ggml-cpu/repack.cpp` at the base (repack buffer allocation, line 5169).
- Snapdragon 888 memory: LPDDR5-6400, 4 × 16-bit, 51.2 GB/s peak (AnandTech SD888 deep dive, search summary).
- Repository: memo-05-work/tom-host.md §1 (K1 instruction mix), memo-07-work/tom-split.md, R69 pre-registration
  §1 and §3, clyde-r72-plan.md, ebony-paths.md, campaign 2 knowledge.md (official prompt 155 tokens).
