# c3 memo-13 work (Bonnie): the CPU-only ladder after R76, and the cost model itself

- **Date:** 2026-11-26. **Author:** bonnie, `claude-fable-5-2`. Not committed. No verdict here.
  **Ebony has not reviewed this file**; her part of the round is `ebony-*.md` in this folder (not read).
- **Labels.** *device* (phone), *host* (laptop, relative only), *estimated* (arithmetic), judgement. Triples are
  Yi / Llama-2 / DeepSeek. Goal: 12 tok/s printed = 79.2 ms per token.
- **Script:** [bonnie_cpu_ladder.py](bonnie_cpu_ladder.py) (`python3 -I -B`; loads `memo-10-work/bonnie_tree.py`
  unchanged; self-checks: stock 6.90 / 6.21 / 6.22, r17's 4.3 / 3.9 / 3.9 within 0.15, E-T23's 26 / 27 / 28 passes).
  Every output is *estimated*.
- Nothing was measured this round. One web search (§8); it changed nothing.
- memo-10 (width) and memo-11 (bytes, exotic) are cited, not redone. What moved since: R74 to R76 (clocks, caps,
  heat), E-T23 (DeepSeek lookup), r15 (attention), r17 and the owner's choice A (2026-11-24), r24 (rule T),
  the owner's "prefix reuse allowed" (2026-11-24).

## Answer first

1. **The two stock numbers in the brief (K0 6.81 / 5.88 / 5.84, ggml stock 6.90 / 6.21 / 6.22) both describe a
   layout the owner's adopted load path does not run.** They are the packed (KleidiAI) body. The owner adopted
   choice A on 2026-11-24 (plain mapped rows, the app's memory rule as it is). On that path stock is
   **5.1 / 4.6 / 4.5 tok/s at full clocks, 4.4 / 4.0 / 3.9 at C 40** (*estimated*; r17 wrote 4.3 / 3.9 / 3.9).
   No phone number exists for Q4_0 on plain rows (r6). So the ladder below has two columns.
2. **Best honest CPU-only stacks, claim-length decode at full clocks** (*estimated*; official prompt, 32 tokens):

   | | Yi | Llama-2 | DeepSeek | what must exist |
   |---|---|---|---|---|
   | owner's path A, stock | 5.1 | 4.6 | 4.5 | nothing |
   | **path A, best stack (A4)** | **7.7** | **6.2** | **6.7** | in-place 4-row kernel, its two-position form, head-only copy, guess rule |
   | packed layout, stock (P0, *device*-derived) | 6.90 | 6.21 | 6.22 | choice B or C, which the owner did not adopt |
   | **packed, best stack (P4)** | **9.4** | **7.6** | **7.4** | side file or forced copy, W-position kernel, Yi and Llama-2 heads, two-stage head |

   The gap to 12 is 2.6 / 4.4 / 4.6 tok/s on the packed stack and 4.3 / 5.8 / 5.3 on path A. **CPU alone, no
   subject reaches 12 on any rung.** P(any subject at 12, CPU only) under 0.005; P(all three) under 0.0005
   (judgement).
3. **Heat is not a price of the CPU-only ladder at claim length.** CPU-only first cap: about 30 s (28 to 41,
   three R72 blocks, *estimated* from *device* lines, r35). R76 measured 14.0 s with the GPU on (*device*), inside
   r35's lane band; it did not re-measure the CPU-only figure. With prefix reuse (owner allowed) a scored decode
   is 3.5 to 7 s from a gated start: P(cap inside it) 0.02 or less (judgement). So the ladder is priced at C 50.
   The exception is path A without prefix reuse, where the prefill alone is 17 to 33 s (§3).
4. **The single CPU-only lever with the largest unknown a host test can settle: the in-place 4-row kernel
   ("Y1b", memo-05) as the decode kernel on the owner's path A.** It exists as probe code in
   `patches/llama.cpp-c3/0002-d-rows.patch`, was bit-identical to the packed gemv on the host on four shapes, and
   **was never timed on the phone and never connected to choice A** (R72 dropped its rows; r17 priced A without
   it). Price: **0 bytes** (no copy, no side file, no storage); body −37 / −43 / −40 ms per token at full clocks
   (−57 / −67 / −62 at C 40); **+1.2 / +1.2 / +1.1 tok/s** (+1.6 / +1.5 / +1.4 at C 40) (*estimated*). Its unknown
   is ±0.4 tok/s: the cold cost on long rows (host: 2.1 against 1.5 hot on `down` shapes). §4.
5. **The cost model (§5): a width-1 token is bus-bound on all three, and the bus term alone (104 / 122 / 114 ms)
   is above the whole 79.2 ms.** With zero arithmetic the stock bytes give 8.0 / 7.1 / 7.1 tok/s; with zero bytes
   the stock arithmetic gives 12.4 / 11.2 / 10.7 at full clocks. So no change of a constant on one side reaches
   12. **A scheme must cut weight-touches per committed token to 0.54 / 0.48 / 0.45 of the body, in bytes and in
   arithmetic at once** (0.75 / 0.58 / 0.50 with width 2 at the measured guess rates). Of the four schemes the
   lead named, three are closed by this arithmetic or by evidence on file; one (delta across tokens) has no
   measurement and one cheap host count closes or opens it. My prior that it opens: 0.01.

## 1. Inputs, and what moved since memo-10 and memo-11

| input | value | label | moved by |
|---|---|---|---|
| packed body, 4 threads on 4-7 | 120.6 / 140.6 / 130.8 ms | *device*, R72 | r31: same at the 1,555 cap and at full clocks (R74 25.64 GB/s) |
| head, KleidiAI | 9.04 / 4.40 / 13.88 ms | *device*, R72 | – |
| head, repack type; two-stage (campaign 2's design) | 7.7 / 3.7 / 11.8; × 0.80 / 0.91 / 0.76 | *estimated*, r12 | R62: the two-stage head saved nothing measurable on OLMoE (*device*) |
| O (attention, KV, graph) | 5 / 7 / 7 ms (band 3.5–8 / 5–10 / 5–10) | *estimated*, r15 | replaces "8 → 5.2" |
| second position, extra | 3.2 / 5.3 / 5.0 ms | *estimated*, r15 | replaces 0.5 / 1.5 / 1.5 |
| hot compute rate C | 52 to 59 at full clocks, 32 to 34 at the 1,555 cap | *estimated* from *device* parts, r31 | I use **50** for a claim decode (below) |
| compute follows the cap | canary 0.046 → 0.058 ns per weight under 1,785 MHz (× 1.26; clock ratio 1.36); 0.070 to 0.076 under 1,420 (× 1.52 to 1.65; 1.70) | *device*, R76 | first device support for "C goes with the clock" |
| X1 limit under the printed cap | `dcvsh_freq_limit` 2,592 MHz while the printed cap was 2,841 (one arm; Annie's audit) | *device*, R76 | why 50 and not 52 to 59 |
| tokens per pass, official prompt | Yi 1.455, Llama-2 1.333 (rule T: lookup at match length 2 or 3, else the head) | *host*, r24 | replaces the set's 1.326 / 1.448 for the claim |
| tokens per pass, fixed set | 1.391 / 1.486 (rule T) | *host*, r24 | – |
| DeepSeek, lookup alone, official prompt | 1.231 any match length, 1.185 at 2 or more, 1.143 at 3 | *host*, E-T23 | replaces the assumed 1.385 |
| plain-row stock kernel | 2.2 (1.9 to 2.5) × the packed gemv per weight | *estimated* count and *host*, r6 | – |
| Y1b in place, one position | 1.47 to 1.51 hot, 2.07 to 2.21 cold on long rows (M4); count 35 ÷ 25 = 1.40, 1.24 without four zeroing moves | *host*, *estimated*, memo-05 tom-host §2 | not used since memo-06 |

**A note on the brief.** "E-T23 lookup 1.231 / 1.185 / 1.143" is not a triple over subjects. All three are
DeepSeek, by minimum match length. DeepSeek has no trained head and no head dump (tom-et23.md, "Not done").

**Cost form** (memo-10's, with r15's terms and a share of width-2 passes):

    body(c) = body_R72 × t(c) ÷ t(1),   t(c) = max(1/R, c/C) + 0.25 × min(1/R, c/C),   R 29.9 GB/s
    ms per token = [(1 − s2)(body1 + O) + s2 (body2 + O + xpos) + sh × guess] ÷ tau + head;   printed = 950 ÷ ms

c per weight against the packed gemv: 1.00 packed; 1.45 packed W-position kernel at two positions (memo-10 E2);
2.2 plain stock; 1.45 Y1b; **2.1 for a Y1b-style kernel at two positions** (my count, not compiled: 18 slots
shared, about 17 per position, over 25). Coupling 0.17 (r32's refit) moves every cell by +0.05 to +0.15 tok/s.

## 2. The ladder, no lane rung

Claim state: C 50, official prompt. *Estimated*; class = the weakest factor a rung needs (D phone, Hs laptop on a
subject, Ho laptop on something else, G guess or unbuilt), as in r14.

### 2.1 Packed layout (the layout of R75-result's ladder; needs choice B or C)

| rung | what | class | Yi | Llama-2 | DeepSeek |
|---|---|---|---|---|---|
| P0 | stock, O 8 (R75 rung 0) | D (a probe, not a decode) | 6.90 | 6.21 | 6.22 |
| – | R72's K0 as written (K1 rule body) | D | 6.81 | 5.88 | 5.84 (K1 is parked; not a rung) |
| P1 | head in the repack type | D on another model | 6.97 | 6.24 | 6.31 |
| P2 | O as r15 | G | 7.13 | 6.28 | 6.35 |
| P3 | width 2, W-position kernel; rule T; DeepSeek lookup at 2 or more, width 2 only on a match | Ho (kernel), Hs (rates) | 9.26 | 7.53 | 7.28 |
| P3s | the same rules on the fixed set | Hs; DeepSeek's share of matched passes assumed | 8.88 | 8.36 | 6.91 |
| **P4** | **+ campaign 2's two-stage head** | Hs numerics, bytes *estimated* | **9.40** | **7.55** | **7.44** |

- Against R75-result's lane rung 6 (10.66 / 10.48 / 10.07, `linear`): the lane was worth 1.3 / 2.9 / 2.6 on paper.
  Most of Llama-2's and DeepSeek's loss is not the lane: it is the official prompt's guess rates (1.333 and 1.185
  against the set's 1.448 and an assumed 1.385).
- What-ifs, not rungs: a DeepSeek head at 1.30 tokens per pass gives 7.52 (no better than the selective lookup,
  because the head makes every pass width 2); at 1.45 it gives 8.32. C5 × 0.93 on the body (never screened, r13)
  would add 0.6 / 0.5 / 0.5. A confidence gate adds under 1 % at this pass ratio (r27).
- By clock state (P4): 9.37 / 7.52 / 7.44 at C 46 (the 2,150 cap); 8.92 / 7.15 / 7.35 at C 40; 7.98 / 6.39 / 7.15
  at C 33 (the 1,555 cap).
- Pass ratio P2 ÷ P1: 1.09 at C 50, 1.15 at C 40, 1.30 at C 33 (Llama-2).
- **What stands between the phone and this column:** the packed body is 3.4 to 3.8 GB of anonymous memory (B:
  the phone held 3.4 GB and not 4.2, R72; the app's rule refuses it) or a side file of the same size (C: about
  180 lines, unbuilt, +3.4 to 3.8 GB of storage per model). The owner chose A.

### 2.2 Plain mapped rows (choice A, the owner's path)

| rung | what | class | Yi | Llama-2 | DeepSeek |
|---|---|---|---|---|---|
| A0 | stock: plain dot products, plain Q6_K head (15 / 7.5 / 24 ms, r17), O 8 | G (count and laptop; no phone record) | 5.07 | 4.59 | 4.52 |
| **A1** | **Y1b in place** | Ho | **6.31** | **5.78** | **5.57** |
| | band: no zeroing moves … `down` cold at 2.1 | | 6.44 … 6.20 | 5.91 … 5.68 | 5.69 … 5.49 |
| A2 | head only copied into the repack type (0.22 / 0.11 / 0.34 GB anonymous) | D on another model; the app's rule counts the whole file today | 6.63 | 5.92 | 6.00 |
| A3 | O as r15 | G | 6.77 | 5.96 | 6.04 |
| **A4** | **width 2 with an in-place two-position kernel (c 2.1); same guess rules as P3** | G (kernel not written) | **7.70** | **6.22** | **6.71** |

- A0 is compute-bound, so it moves with the clock: 4.82 / 4.34 / 4.30 at C 46, 4.41 / 3.96 / 3.95 at C 40,
  3.90 / 3.48 / 3.50 at C 33. A1 at the same states: 6.28 / 5.76 / 5.55, 5.99 / 5.48 / 5.31, 5.37 / 4.88 / 4.78.
- Width 2 in place has a pass ratio of 1.26 at C 50 and 1.33 at C 46. It pays only at full clocks, and on
  Llama-2 only just (+0.26; a loss from C 46 down). With r31's self-timing gate it cannot lose.
- **Y1b gives path A most of the packed stock speed (0.91 / 0.93 / 0.90 of P0) with no copy.** It also makes the
  phone's numerics the freeze's: Y1b equals the packed gemv bit for bit (*host*, four shapes), while the plain
  stock path differs from it by about ±0.02 % (*host*, memo-05).
- Not priced as a rung, one line for the lead: the app's rule admits a copy of 2.75 GB by its own arithmetic. A
  runtime that packs tensors up to that size (long-row `down` first) and runs the rest on Y1b would sit within
  1 to 3 % of the packed column. It is 2.75 GB of anonymous memory, between what campaign 1 survived (2.3 GB) and
  what it did not (3.6 GB). Whether that is still "choice A" is the lead's and the owner's reading.

## 3. Heat at claim length, with R76's reading

| run | load from a passed gate | first cap | P(cap before the decode ends) |
|---|---|---|---|
| any lane rung (off the list) | CPU set with the GPU | **14.0 s** (*device*, R76 H1; second step at 24.0 s) | the cap falls inside an 11 to 18 s run |
| CPU only, prefix reuse on, runs 2 to 10 | decode only, 3.5 to 7 s | about 30 s (28 to 41; n = 3, R72) | 0.02 or less (judgement) |
| CPU only, packed, no reuse | prefill 7.6 / 9.8 / 8.0 s + decode 4 to 5 s (r33) | same | 0.06 / 0.14 / 0.08 (r35) |
| CPU only, path A stock, no reuse | prefill 26 / 33 / 27 s (plain rows cost 2.2 against the packed prefill's 0.65 per position) + decode 6 to 7 s | same | about 0.6 (judgement): the decode starts at the first step |
| CPU only, path A with Y1b, no reuse | prefill 17 / 22 / 18 s + decode 5 to 6 s | same | about 0.25 |

- The packed stock decode does not care (bus-bound: 0 to 3 %, r32). What a cap costs is the second position:
  P4 loses 0.03 at 2,150, 0.5 / 0.4 / 0.1 at 1,785 and 1.4 / 1.2 / 0.3 at 1,555.
- Path A cares at width 1 already: A0 loses 5 % at 2,150 and 13 % at 1,785.
- **So prefix reuse is a condition of the path-A column, not a refinement.** Without it, path A's stock claim run
  crosses the 30 s line before its decode ends.
- Two R76 facts that qualify "full clocks": the slow stretch began 2.8 s before the printed cap, and cpu 7's
  hardware limit read 2,592 under a printed 2,841 in one arm. A printed cap at maximum is not proof of C 52 to 59.
  Both were seen with the GPU on; neither is shown for CPU-only load.
- Limit: the CPU-only 30 s rests on three probe blocks with no time stamps (r35's caveat). R76 does not add to it.

## 4. The lever with the largest host-settleable unknown: Y1b as the decode kernel on path A

**What it is.** Four of the file's own Q4_0 rows at a time, read where they are mapped: per row `EOR 0x88`,
`SHL 4` / `AND 0xF0`, two `SDOT`; three `ADDP` put four row sums in one vector; then the packed kernel's float
step. Probe code: `y1b()` in patch 0002. Not wired into any decode path.

**Why it is the largest.** On the owner's path nothing else is worth 1 tok/s, and nothing about it has been
measured where it matters:

| unknown | range | effect on A1 (*estimated*) | who settles it |
|---|---|---|---|
| per-chunk identity against the freeze build on whole models, at `-b 512`, `-ub 32`, `-ub 1` | identical or not | not identical → a declared numerics change (it would then be compared against the budget; the plain path's own shift is ±0.02 %) | **host, fully** |
| hot cost per weight | 1.24 (count, zeroing moves removed) to 1.51 (M4) | ±0.13 tok/s at C 50; 5.99 to 6.40 (Yi) at C 40 | **host**: count with `tools/c3_a78_pipes.py`, M4 timing |
| cold cost on long rows (`down`, 6.2 KB rows, 22 to 26 % of the body) | 1.5 to 2.2 | −0.1 at C 50, −0.4 at C 40 | **host in kind** (the M4 shows the effect and would show a cure); the phone in size |
| prefill | per-position loop at 1.45, against 2.2 stock | 17 / 22 / 18 s against 26 / 33 / 27 s | host count |
| does the mapped body stay resident | r17: P 0.85 / 0.65 / 0.55 | a precondition of every path-A number | phone only (r18's `x5s` arm) |

**Price.**
- Bytes: 0 in the file, 0 anonymous, 0 storage. (Choice C: +3.4 / 3.8 / 3.8 GB storage. Choice B: 3.4 / 3.8 / 3.8 GB
  anonymous.)
- Milliseconds per token, A0 → A1: 187.4 → 150.6, 207.0 → 164.4, 210.2 → 170.6 at C 50
  (**−36.8 / −42.6 / −39.6**); 215.4 → 158.6, 239.9 → 173.4, 240.5 → 178.9 at C 40 (−56.8 / −66.5 / −61.6).
- tok/s: **+1.24 / +1.19 / +1.05** at C 50; +1.58 / +1.52 / +1.36 at C 40. All *estimated*.
- P(it delivers +0.8 tok/s or more over the plain stock path on all three, on the phone): 0.6 (judgement). The
  risks are the cold long rows and that A0 itself has no phone number to stand on.
- It cannot reach 12 and does not claim to: it moves the owner's path from about 5 to about 6.

**Second, on the packed column only:** the W-position kernel inside llama.cpp (identity against batch 1, on the
KleidiAI path too) and a DeepSeek head. Together P2 → P3: +2.1 / +1.3 / +0.9 tok/s. Both are host-settleable
(memo-10 T10; T11). DeepSeek's head is worth building only above about 1.35 tokens per pass on the official
prompt: at 1.30 it ties the lookup, which costs no training.

## 5. The cost model itself: which term bounds, and what a change of scheme must deliver

### 5.1 The terms (*device* inputs; C 50 and C 40 *estimated*)

| term | Yi | Llama-2 | DeepSeek | note |
|---|---|---|---|---|
| body bytes per token | 3.11 GB | 3.64 GB | 3.42 GB | *device*, R72 |
| **bus term**, bytes ÷ 29.9 GB/s | **104.0 ms** | **121.7 ms** | **114.4 ms** | **above 79.2 alone** |
| compute term at the stock `sdot` cost, C 50 / C 40 | 62.2 / 77.8 ms | 72.8 / 91.0 ms | 68.4 / 85.5 ms | above the body budget at C 40 on all three, at C 50 on two |
| body as measured | 120.6 ms | 140.6 ms | 130.8 ms | bus + about a quarter of compute |
| head | 9.0 ms | 4.4 ms | 13.9 ms | DeepSeek: 18 % of 79.2 |
| attention, KV, graph (O) | 5 ms | 7 ms | 7 ms | grows with context |
| passes per token | 1 | 1 | 1 | 0.69 / 0.75 / 0.84 with the measured guess rules |
| body budget, 79.2 − head − O | 65.1 ms | 67.8 ms | 58.3 ms | |

**Which term bounds.**
- Width 1, stock bytes: the **bus**, on all three, at every clock state seen (r31).
- Any second position, or any byte cut beyond about 40 %: **compute** takes over (crossover at 1.75 positions at
  C 40, memo-10; at a byte share under 0.6 × k at C 50).
- The head is third and matters on DeepSeek only. Attention is fourth; it doubles at width 2.
- Passes divide everything except the head.

**Two limits that frame every scheme** (width 1, KleidiAI head, O as r15):

| | Yi | Llama-2 | DeepSeek |
|---|---|---|---|
| zero arithmetic, stock bytes | 7.99 tok/s | 7.10 | 7.05 |
| zero bytes, stock arithmetic, C 50 (C 40) | 12.37 (10.56) | 11.24 (9.50) | 10.68 (9.20) |

Neither side alone reaches 12 on all three, even at its impossible end. memo-11 said this for bytes; it holds for
arithmetic too, and more strongly.

### 5.2 What a scheme must deliver

Largest share of the body, counted in bytes **and** in arithmetic together (b = k), that meets 79.2 ms:

| form | Yi | Llama-2 | DeepSeek |
|---|---|---|---|
| width 1 | 0.54 | 0.48 | 0.45 |
| width 1, two-stage head | 0.56 | 0.49 | 0.48 |
| width 2 every pass at the measured guess rates, two-stage head, C 50 | 0.75 | 0.58 | 0.50 |

Or, on stock bytes, tokens per pass needed (two-stage head, C 50):

| width | extra position at 0.45 of a pass (counted) | extra positions free (not reachable: exact floor 0.26 to 0.32) | have |
|---|---|---|---|
| 2 | 1.89 / 2.15 / 2.17 | 1.79 / 2.05 / 2.07 | 1.455 / 1.333 / 1.185 (ceiling 2) |
| 3 | 2.22 / 2.54 / 2.56 | 1.88 / 2.16 / 2.18 | about 1.5 to 1.7 (memo-10, *estimated*) |
| 4 | 2.69 / 3.08 / 3.11 | 1.96 / 2.26 / 2.29 | about 1.5 to 1.8 |

So the requirement in one line: **about half the weight-touches per committed token, in bytes and arithmetic at
once, or a guesser right about 0.7 to 0.85 of the time in a chain.** Everything on the books is a constant
(10 to 30 %) on one side.

### 5.3 The four schemes, priced

| scheme | what it would have to deliver | what the books say | bytes | ms | state |
|---|---|---|---|---|---|
| **Delta across tokens**: y_t = y_(t−1) + W (x_t − x_(t−1)), touching only columns whose input changed | changed share d ≤ 0.45 / 0.40 / 0.36 on the row-major file (0.50 / 0.45 / 0.42 with a column-major copy), with the exact column kernel at 1.55 per touched weight; d ≤ 0.93 only to equal stock | **d has never been measured.** Nearest: active-column sets are not stable between tokens (h 0.002 / 0.005, *host*, memo-05 T5). The column kernel at keep 0.58 to 0.65 ran at 0.95 to 0.99 of stock on the phone (*device*, R72), so the lines here are optimistic. Not exact: the activation scale changes per token, so the carried sum is a float running total that drifts and needs refresh passes | 0, or +3.1 to 3.6 GB for a column-major copy | at d 0.9: slower than stock | **open by lack of data; one host count (T-D1). P(d under the line on all three) 0.01** |
| **One read for several tokens**, beyond width 2 | tokens per pass as in §5.2 | The only tokens that can share a read are guessed ones: token t+1's first input is token t's output. So this is memo-10's tree; its bound is the guesser, not the kernel. Even with free extra positions width 2 needs 1.79 / 2.05 / 2.07 | 0 | – | closed (memo-10); the free-position row is new and closes the kernel side |
| **Another arithmetic domain for the product** on the same bytes (tables, bit planes, narrower activations, 16-bit sums) | nothing it can deliver is enough: zero arithmetic gives 7.99 / 7.10 / 7.05 | Differs from U2: U2 was tables on re-encoded 2- and 3-bit weights, a byte cut with a kernel that turned out dearer than `sdot`. Here the bytes stay, so the bus term stands. The exact `sdot` domain is already at 25 slots per 128 weights with a floor of 8; expanding weights at load to skip the unpack costs 1.78 × the bytes | 0 (or more) | at most −16 to −19 ms at width 1 with zero arithmetic | **closed by arithmetic**; no test. Its only use is a cheaper extra position: 0.45 → 0.30 is +0.1 tok/s at C 50 |
| **Runtime-derived low rank or factored paths** | parameter share kept ≤ 0.54 / 0.48 / 0.45 with factors at 4.5 bits, ≤ 0.30 / 0.27 / 0.25 at 8.5 bits, inside +5 % | SVD-LLM V2 at 20 % fewer parameters: +25 % and +30 % perplexity (*published*, memo-11 bytes §6). The budget buys under 5 %. Deriving it on the phone is hours of SVD per model; a host-made sidecar is allowed, the loss is not | – | under −6 ms inside the budget | closed (memo-11); the requirement row is new |

**What this leaves.** Three families can cut weight-touches per token: fewer bits (the budget stops at 73 to 86 %
of the bytes, memo-11), fewer weights (columns: 40 / 20 / 20 % inside the budget, and the kernel that skips them
costs what it saves, R69 and R72), more tokens per touch (1.19 to 1.46 of a ceiling of 2). They do not multiply:
each needs its own kernel and the first two share one budget. The one priced cell near 12 is still memo-11 §7's
2-bit-class served file (11.6 / 11.3 / 10.9), which changes the served file and is the owner's question. A new
"computation scheme" inside the present rules has to be a fourth family. Delta across tokens is the only
candidate I can name that has no number against it yet, and my prior on it is low because the residual stream
changes with every token.

## 6. For Tom (through the lead)

### 6.1 Test list (all host; each closes a question)

| # | test | cost | closes when |
|---|---|---|---|
| B-T1 | **Y1b in the decode path, identity.** Behind a knob, route every Q4_0 `mul_mat` on mapped rows (extra buffers off) through `y1b()`, looping positions for batches. Per-chunk PPL at `-b 512`, `-ub 32`, `-ub 1`, 40 chunks, Yi and Llama-2 (DeepSeek if time), against the repack build's logs | one small patch; three short runs per model | any chunk differs → class "declared numerics", and the lever needs a budget check instead |
| B-T2 | **Y1b cost at our shapes.** 4096 × 4096, 11008 × 4096, 4096 × 11008; one thread and four pinned (Docker arm64, as `tom_t1.cpp`); hot and cold on a file-backed arena; against `--no-repack` rows and the packed gemv. Arms: as is; zeroing moves removed; a linear prefetch of the next four-row group | bench only | cold `down` stays above 1.8 with every arm → price A1 at its low band; whole-model cost above 1.7 at one thread → Y1b is worth under +0.6 tok/s, stop |
| B-T3 | **In-place two-position kernel, count only.** Compile for `-mcpu=cortex-a78`, count with `tools/c3_a78_pipes.py` | minutes | above 2.3 → width 2 on path A is closed (ratio over 1.33 at full clocks) |
| T-D1 | **Delta count.** With an eval callback on the fixed set (Yi, Llama-2): at every `mul_mat` input, the share of Q8_0 codes equal to the previous token's, and the share of 32-blocks whose scale is equal; per layer and per matrix kind | a few lines in `tools/c3_head_dump.cpp` | body-weighted unchanged share under 0.5 on either subject → delta across tokens closed |
| T10 (memo-10) | W-position kernel in the tree, identity against batch 1 on the repack and KleidiAI paths | as listed there | only if the lead keeps the packed column |

### 6.2 Try list (after its test; env-gated, off by default, model-generic, two subjects)

| # | lever | after | label | price (Yi / Llama-2 / DeepSeek) | P |
|---|---|---|---|---|---|
| B-Y1 | Y1b as the decode kernel on mapped rows | B-T1, B-T2 | exact against the freeze build (if B-T1 passes) | 0 bytes; −37 / −43 / −40 ms; +1.2 / +1.2 / +1.1 tok/s over path-A stock | 0.6 |
| B-Y2 | head-only copy into the repack type | the lead's reading of the app rule | exact against the freeze build | +0.22 / 0.11 / 0.34 GB anonymous; −7.3 / −3.8 / −12.2 ms; +0.3 / +0.1 / +0.4 tok/s | 0.5 |
| B-Y3 | prefix reuse on in the claim protocol | none (owner allowed) | protocol | 0 bytes; removes the cap risk of §3 | 0.9 |
| B-Y4 | width 2 on path A: in-place two-position kernel, rule T, DeepSeek on lookup, self-timing gate | B-T3, heads | exact output | 36 MB sidecar per head; +0.9 / +0.3 / +0.7 tok/s at full clocks, 0 under a cap | 0.25 |

Nothing here is a device request. A phone number for A0 and A1 would need r18's `x5p` cell plus a Y1b row, its
own pre-registration and the owner's permission.

## 7. Ranked next questions (the lead decides; I do not continue)

1. Is choice A the base of campaign 3's reported result? If yes, the K0 quoted to the owner (6.81 / 5.88 / 5.84,
   6.90 / 6.21 / 6.22) should carry a note that it is the packed layout, and the path-A stock (about 4.4 to 5.1 /
   4.0 to 4.6 / 3.9 to 4.5, *estimated*, unmeasured) should stand beside it.
2. B-T1 and B-T2: is Y1b exact on whole models and what does it cost at 4096 × 11008 cold?
3. T-D1: how much of an input changes between consecutive tokens? One count closes the only unpriced scheme.
4. Does a head-only copy, or the largest copy the app's own rule admits, count as choice A?
5. DeepSeek head: only if a quick probe (ridge on E-T23's dump) suggests more than 1.35 tokens per pass.

*Added 2026-11-27 (Bonnie):* Ebony's bus items (up-first rows, `down` columns, UD, scale index) are re-priced on
this ladder in [bonnie-ud-check.md](bonnie-ud-check.md). §5.3's "the kernel that skips them costs what it saves"
holds for the uniform column rule and not for `down` alone with an exact mask; see that file's §3.

## 8. Not done, limits, sources

- No host or device run. Every path-A number rests on a count and on laptop ratios; the phone has never run Q4_0
  on plain rows, Y1b, or any two-position kernel.
- The in-place two-position cost (2.1) is my count by hand. DeepSeek's share of matched passes on the fixed set is
  assumed (0.30).
- Official-prompt guess rates are 31 guesses per model; one hit moves a rate by 0.03. The phone's reply can differ
  from the dump's (Yi and DeepSeek differ between builds, tom-et23.md).
- Plain-head times (15 / 7.5 / 24 ms) and prefill times on plain rows are *estimated* from ratios, not measured.
- Web: one search for upstream work on an in-place or mmap-compatible Q4_0 repack on Arm. It found Arm's
  description of the KleidiAI repack step and nothing that reads the file's rows in place; not a full survey.
  - <https://learn.arm.com/learning-paths/mobile-graphics-and-gaming/kai_sme2_matmul_ukernel_explained/explain_with_an_example_p1/>
- Repository: `reports/R72-result.md`, `R75-result.md`, `R76-result.md`; `memo-10-tree.md`; `memo-11-bytes.md`,
  `memo-11-exotic.md`; `memo-05-work/tom-host.md` §2; `memo-12-work/bonnie-joins-r6.md`, `r12`, `r13`, `r14`,
  `r15`, `r17`, `r24`, `r27`, `r31`, `r32`, `r33`, `r35`; `ebony-joins-r17.md`; `tom-et23.md`;
  `patches/llama.cpp-c3/0002-d-rows.patch`; `CAMPAIGN.md` (owner decisions 2026-11-24, 2026-11-26).
