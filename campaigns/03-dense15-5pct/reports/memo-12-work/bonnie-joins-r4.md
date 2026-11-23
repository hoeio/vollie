# c3 memo-12 work, round 4 (Bonnie): the width-2 together cell, head on the lane, clock feedback

- **Date:** 2026-11-23. **Author:** bonnie, `claude-fable-5-2`. Follows [bonnie-joins-r3.md](bonnie-joins-r3.md).
  Not committed. No verdict here. **Ebony has not reviewed this file.** Her `ebony-joins-r3.md` appeared after §0 to §6 were
  written; §7 reviews it and gives the merged R75 table. Where §3 and §7 differ on clocks, §7 is my position.
- **Labels.** *device*, *derived*, *estimated*, *source* as in r3. Triples are Yi / Llama-2 / DeepSeek.
- **Script:** [bonnie_tree_r2.py](bonnie_tree_r2.py) (one guard added: it refuses C below 31 GB/s, see §3).

## 0. Answer first

1. **Together cell at two columns:** sum 37.0 GB/s of weight bytes per pass (80 % band 31.5 to 41.5). My model
   says 33.6 to 36.7, Ebony's pair model 40.4. **Line: sum ≥ 38.5 → hers; ≤ 35.5 → mine.** The cleaner
   discriminator is the CPU alone: `both_cpu_gbs` ÷ `cpu_alone_gbs` at two columns, 0.83 in mine, 0.98 in hers.
2. **The discriminator fails if ggml's two-row kernel costs as much as two gemvs.** Then the CPU is compute-bound
   in both models (16.4 against 16.9 GB/s) and the cell cannot tell them apart; the sum then reads 32 to 35.
3. **Head on the lane:** upstream's Q6_K gemv should run on the Adreno 660 (same extensions as the Q4_0 kernel,
   all three vocabularies are multiples of 128, *source* and *device*). Worth 1.0 / 0.4 / 1.7 ms per token after C7 at K 300
   (+0.06 to +0.22 tok/s). Small; it is a rider on a lane, not a reason for one.
4. **Clock feedback: my r3 question 6 was wrong.** The lane halves CPU work per token, not per second. The four
   threads stay busy and the GPU's power is added, so heat rises. R74's own split block shows it: the A78 cap fell
   from 2,150 to 1,785 MHz inside the block (*device*). Price: C 40 → 35 → 30 costs the lane 0.3 and 0.6 tok/s at
   K 300 in the pair model; the lane absorbs part of it by moving rows to the GPU.

## 1. Bands for the together cell (x4m; *estimated*; centre and 80 % band; GB/s of weight bytes per pass)

Inputs: R74 `x3 togsum` 25.64 / 19.77 / 21.53 (*device*); memo-10's c(2) = 1.45 for the CPU; cg 1.2 to 1.26;
launch gap 125 to 140 µs per kernel in the `tog` state (*derived*). KleidiAI takes the two positions as two rows
of one matmul call, so the weights are read once (*source*, `kleidiai.cpp`: m = ne11); what the second row costs
in compute is not measured (band c 1.3 to 2.0).

| field | centre | 80 % band | mine (`bonnie`) | hers (`pair`) |
|---|---|---|---|---|
| tog1sum `cpu_alone_gbs` | 25.6 | 24.5–27.0 | R74 25.64 | same |
| tog1sum `both_cpu_gbs` / `both_gpu_gbs` | 19.8 / 21.5 | 18.5–21.0 / 19.5–24.5 | R74 19.77 / 21.53 | same |
| tog1sum `sum_vs_cpu_alone` | 1.61 | 1.50–1.72 | R74 1.611 | same |
| tog2sum CPU alone, 2 columns | 22.0 | 17.0–24.5 | 22.5 at c 1.45; 17.2 at c 2.0 | same |
| tog2sum GPU alone, 2 columns (wall) | 21.1 | 15.0–23.5 | 25.36 MB ÷ (896 × cg + 125 µs) | same |
| tog2sum `both_cpu_gbs` | **19.5** | 16.0–22.5 | **18.7** (16.4 at c 2.0) | **22.1** (16.9 at c 2.0) |
| tog2sum `both_gpu_gbs` | 17.5 | 13.0–20.5 | 17.9 (cg 1.2); 14.9 (cg 1.45) | 18.4 (cg 1.26) |
| tog2sum sum | **37.0** | 31.5–41.5 | 36.7 (cg 1.2); 33.6 (cg 1.45) | 40.4 |
| sum ÷ `cpu_alone_gbs` at one column | 1.44 | 1.23–1.62 | 1.43; 1.31 | 1.57 |
| sum at 2 columns ÷ sum at 1 column | 0.90 | 0.77–0.99 | 0.89; 0.81 | 0.98 |
| `both_cpu_gbs` ÷ CPU alone, both at 2 columns | 0.89 | 0.80–1.00 | 0.83 | 0.98 |

If `sum_vs_cpu_alone` in tog2sum divides by the CPU alone at two columns instead, the centre is 1.68 (1.45 to 1.90).

### 1.1 Which value says who is right, and the stack it implies

| reading | model | stack at K 0 | stack at K 300 |
|---|---|---|---|
| sum ≤ 35.5, or CPU ratio ≤ 0.87 with CPU alone ≥ 21 | mine, cg near 1.45 | 12.60 / 12.24 / 11.77 | 10.80 / 10.66 / 10.33 |
| sum 35.5 to 38.5 | mine at cg 1.2, or between | 13.58 / 13.24 / 12.66 | 11.28 / 11.20 / 10.81 |
| sum ≥ 38.5, and CPU ratio ≥ 0.93 with CPU alone ≥ 21 | hers | 14.89 / 14.57 / 13.84 | 11.89 / 11.88 / 11.43 |
| CPU alone at 2 columns < 19 (the two-row kernel is dear) | neither is tested; the stack needs the W-position CPU kernel, or both priced at c 2.0 | not priced | not priced |
| sum < 31.5 | below both; width 2 on the lane is worth under +0.4 over `old` | about 12.2 / 11.9 / 11.4 | about 10.5 / 10.4 / 10.1 |

- Once the cell is run the models are not needed: the body factor at width 2 is `cpu_alone_gbs` (one column) ÷
  the two-column sum, wall rates, with the `tog` state's gaps inside. `bonnie_tree_r2.rates()` can be fed the three
  measured rates directly.
- Stacks need every unbuilt lever at its centre (trained head, C7, C5, O 5.2).

## 2. Head on the lane (upstream Q6_K gemv; *source* unless marked)

Read: `kernels/gemv_noshuffle_q6_k_f32.cl` (kernel signatures and pragmas; the loop bodies were skimmed, not
traced), `ggml-opencl.cpp` near lines 5218, 8767 to 8977, 22603.

- **Will it run on the 660?** Likely. P 0.8 (*estimated*).
  - Extensions used: `cl_khr_fp16`, `cl_khr_subgroups`, `cl_qcom_reqd_sub_group_size`. All three are in the
    phone's list (*device*, R72 `x2info`). These are the Q4_0 kernel's, which built and ran in R74.
  - Upstream adds `-DGGML_CL_A6X_CONSTFOLD_FIX` for A6X on this program too, and names A6X in its routing.
  - The no-shuffle path is correct only when the row count is a multiple of 128 (upstream's guard). Vocabularies:
    64,000 / 32,000 / 102,400. All pass.
  - Not checked: whether the 660's compiler builds this file (R74 built only the Q4_0 files).
- **Layout it needs:** two weight images (`src0_ql` low 4 bits, `src0_qh` high 2 bits), sub-block scales and
  super-block scales as `half2` buffers. A second GPU copy of the head in that layout: 0.21 / 0.11 / 0.34 GB whole,
  about 40 % of it on a lane.
- **What a numerics emulation needs (for Tom):**
  1. Q6_K's super-block scale d is fp16 in the file, so it is exact; the 8-bit sub-scales are exact in half.
  2. The risk is the product: (q − 32) × sub-scale reaches 32 × 127 = 4,064, one bit more than half's 11-bit
     mantissa holds. If the kernel forms it in half before the multiply by d, the weight is rounded by up to
     2⁻¹¹ (5e-4 relative), the same order as the Q4_0 kernel's measured 6e-4 to 8e-4 (*device*, R74 identity).
  3. So the emulation is the R74 §7 item 1 pattern with one more arm: the Q6_K dequantise step in the kernel's
     order and types (read from the loop body, lines 200 to 290), accumulation in float as the kernel does. Check
     per-chunk PPL at `-b 512` and `-ub 1` with the body arm on and off; the budget is shared.
  4. An identity cell on the phone (one 4,096-wide head of 32,000 rows against the float64 reference) is the same
     code as x3b with the other kernel.
- **Price** (*estimated*). Head on the CPU: 9.04 / 4.40 / 13.88 ms (*device*, R72 L7). One stage, one join per
  token. GPU Q6_K rate unknown: taken as 0.7 × its Q4_0 byte rate (band 0.4 to 1.0), lane factor 0.74 (0.62 to 0.90).
  - After C7 (head × 0.5): saves 1.2 / 0.6 / 1.8 ms, less 0.5 × K once per token: **1.0 / 0.4 / 1.7 ms at K 300**.
    That is +0.15 / +0.06 / +0.22 tok/s near 11.5.
  - Without C7: 2.2 / 1.0 / 3.5 ms at K 300.
  - C7 screens rows; a screened head has a tenth of the rows and may not be worth a launch (a stage pays only when
    its CPU time exceeds K). On Llama-2 after C7 the head is 2.2 ms: it pays at any K under 2 ms.
  - A two-column variant exists upstream (`_q6_K_f32_mc3`, opt-in by `GGML_OPENCL_Q6K_MC3`), not read.

## 3. Clock feedback

- **Correction to r3 §4 item 6.** I wrote that the lane halves the CPU's work and should ease the caps. Per token
  yes; per second no. The CPU threads work through every pass and passes come faster. Power per second stays at
  four busy cores, plus the GPU.
- **Device records on caps against load** (all *device*; none is a controlled load sweep):

  | record | load | A78 cap read |
  |---|---|---|
  | R72 L4, Yi, 3 threads on 4,5,6 | 3 big cores | 2,419 MHz |
  | R72 L4, Yi, 4 threads on 4-7 | 4 big cores | 1,555 to 1,785 |
  | R72 L4, Llama-2 and DeepSeek | 3 or 4 big cores | 1,555 throughout |
  | R74 `x3 tog` (first timed block, CPU + GPU) | 4 cores + GPU, about 10 s | 2,419 held |
  | R74 `x3 split` (later block, CPU + GPU) | 4 cores + GPU, about 16 s | 2,150, then 1,785 from the fourth arm |
  | campaign 2 knowledge.md (R27) | decode | "on other days they sat at 1,555 MHz" |

  Reading: the cap follows accumulated heat, and it has been seen at 1,555 with the CPU alone. Nothing here shows
  the GPU making it worse, and nothing shows a lane easing it. R74's GPU read `thermal_pwrlevel=0` (not throttled).
- **Where C sits in a real decode** (*estimated*): C 40 was set as the centre between full clocks (50) and the L4
  caps (35 with the X1 on-core). For a sustained decode with the GPU on, I put C at 35 (30 to 40); P(cap at or
  below 1,785 MHz after a minute) 0.7.
- **Price** (`bonnie_tree_r2.py`, K 300; pair at cg 1.26 / mine at cg 1.2):

  | C GB/s | `pair` | `bonnie` | balanced GPU share (`pair`, Llama-2) |
  |---|---|---|---|
  | 40 | 11.89 / 11.88 / 11.43 | 11.28 / 11.20 / 10.81 | 0.35 |
  | 35 | 11.57 / 11.57 / 11.15 | 11.14 / 11.08 / 10.71 | 0.38 |
  | 30 | 11.27 / 11.27 / 10.88 | 10.96 / 10.91 / 10.56 | 0.42 |

  - At K 0 the pair model loses 0.26 and 0.5 tok/s for the same steps (14.89 → 14.63 → 14.36 on Yi).
  - The lane softens a cap: as the CPU slows, the balanced share moves rows to the GPU, which the CPU cap does not
    touch. The same cap costs the CPU-only width-2 pass more (memo-10: kB(2) 1.12 → 1.23 from C 40 to 35).
  - **Limit of my script:** it takes the measured body as fixed, so it is valid only while the CPU is bus-bound at
    width 1 (C above 30.5). Below that the stock body itself slows; the script now refuses such input. Ebony's
    C 24 column (no line for 12) is not reproduced here.
- **What R75 adds:** every x4s and x4m line prints `max_mhz`. A block of 18 arms with CPU + GPU load gives the
  first cap-against-time record under lane load, after a cool start.

## 4. Next questions

1. **The merged R75 band table** with Ebony (brief item 2): done, §7.
2. **Cost of ggml's two-row matmul against one row** on the A78 (c at width 2). It decides whether the together
   cell can discriminate (§0 item 2). Host-readable: count the KleidiAI kernel's work per row for m = 2; device:
   the cell's own CPU-alone arm at two columns against one.
3. **Lane beside a capped CPU:** a share that follows the cap needs the overlap rows in both places (Ebony A.5).
   Price the memory for a share range of 0.35 to 0.45: about 0.1 × the body, 0.3 to 0.4 GB.
4. **GPU heat:** no record of the Adreno's own throttle under minutes of load. `thermal_pwrlevel` over a 60 s
   GPU arm would give it.
5. **Width 3 on the lane** if x4m reads `dev_ratio(3)` ≤ 1.3 (r3 §4 item 5); `bonnie_tree_r2.py` would need W 3.

## 5. Limits

- §1's split between the two models is a judgement; the cell replaces both.
- §2 rests on kernel signatures, pragmas and upstream's comments, not on a traced loop or a build on the phone.
- §3 uses six records, none a controlled sweep; C in a real decode is a judgement until R73 or a lane round runs.

## 6. Sources

Repository: R74 JSON (`x3 tog`, `x3 split` `max_mhz` fields), [R72-result.md](../R72-result.md) §4 and L4 notes,
`campaigns/02-25toks-5pct/knowledge.md` (R27 line), [ebony-joins-r2.md](ebony-joins-r2.md) A.6.
Upstream at the base (*source*, `lab/llama.cpp-c3`): `ggml/src/ggml-opencl/kernels/gemv_noshuffle_q6_k_f32.cl`,
`ggml/src/ggml-opencl/ggml-opencl.cpp`, `ggml/src/ggml-cpu/kleidiai/kleidiai.cpp`. No web source this round.

## 7. Review of ebony-joins-r3.md and the merged R75 band table

### 7.1 Review

- **Her x4m centre (1.3 against my 1.2):** her reason holds. R72's "read" kernel has ALU work of its own, so 31.14
  GB/s is not a pure bus ceiling and my fit of the gemv's compute share is a lower bound. Merged centre 1.25.
- **Her `gpu_dev_ms` at shares 0.55 and 0.60 (1.13 / 1.23 against my 1.08 / 1.18):** a contention factor of 1.15
  against my 1.10. R74 supports 1.10 (`tog`); the split cell's own `gpu_ms` fits either. Merged with both centres.
- **x4e with the C11 variant:** I had no band for it. SVM fine-grain with atomics is the one defined way for a
  running kernel to see a host write, so P rises. I take her 0.55 for "any mode sees the flag" but keep 0.10 for
  "all three lines pass" until the looped gemv rate is seen (her 0.15).
- **Hand-off cell: this is a term my K never held.** A join also moves the stage's input to the GPU and its
  output back. Her 90 µs (40 to 200) by write and read calls would add to every K in r2 §2.5 and r3 §2; her 5 µs
  for buffers kept mapped would not. **K in a lane = launch dead time + hand-off.** My P(a no-call hand-off returns
  correct rows) 0.55, close to her 0.60; R74's kernels already read a host-written input buffer, but through an
  explicit write.
- **Clocks (her §2): her records are better than mine in §3, and I change my position.** Campaign 2's memo-15
  (225 runs: cap against headroom at the gate) and R68 (thermal status sets the cap) are the load-linked records I
  said did not exist. Her reading stands: within one gated run the clocks hold (R74 kept full caps for about 15 s
  under CPU + GPU load); the start state decides; and the K0 bodies were measured at 1,555 to 1,785 MHz, so C 35
  is the state of the inputs, not a pessimistic case. **Merged: C 40 to 50 for a gated claim run (her P 0.6), C 35
  for sustained decoding (mine).** The stack tables at C 40 are then not optimistic for a claim-format run.
  My §3 point stays: a lane does not lower heat per second.
- **Her §1.3 (x4s `cpu_ms` at shares 0.55 and 0.60 against 0.20 shows how the CPU's rate moves with the GPU's
  duty):** agreed, and it needs no new cell. My split model's fitted value is 1 + 0.45 × duty; a slope near 0.30
  would favour `pair`.
- **Not reviewed:** her §3 (other runtimes); nothing there changes a price by her own account.

### 7.2 Merged table (centre; 80 % band; *estimated*). B = Bonnie, E = Ebony, "both" = agreed

Rows not listed here stay as in [bonnie-joins-r2.md](bonnie-joins-r2.md) §2.1 and §2.2 (x4a, x4b, x4f, x4c, x4d,
x4r, x4i: **mine only; Ebony has given no bands for them and has not reviewed them**).

| cell, field | merged | whose | disagreement |
|---|---|---|---|
| x4b best `ovh_us` | 215; 140–280 | B | – |
| x4b P(best ≤ 250 / ≤ 150 / ≤ 100) | 0.80 / 0.12 / 0.02 | B | – |
| x4a `gap_p50_us`, launcher 0-3 / cpu 4 | 200; 145–270 / 210; 140–285 | B | – |
| x4f `set_to_start_p50_us` | 190 to 210; 110–280 | B | – |
| x4c `gate_up_vs_11008`; identity on both shapes | 1.14; 1.04–1.20; P 0.90 | B | – |
| x4d runs / passes the 0.5 × line | P 0.55 / 0.25 | B (E's P 0.25 in her r1) | – |
| x4r held | P 0.35 | B | – |
| x4m kernel created / runs / identity given runs / all | P 0.92 / 0.85 / 0.88 to 0.90 / 0.75 | both | – |
| x4m `dev_ratio` 2 columns | **1.25; 1.05–1.80** | B 1.2, E 1.3 | centre; P(≤ 1.30) 0.58 against 0.50 |
| x4m `dev_ratio` 3 / 4 columns | 1.5 to 1.65; 1.15–2.6 / 1.9 to 2.0; 1.35–3.4 | B / E | centres |
| x4m `wall_ratio` 2 columns | 1.15 to 1.22; 1.04–1.65 | B / E | centres |
| x4m together, 2 columns: sum GB/s | **37.0; 31.5–41.5** | B | `bonnie` 33.6 to 36.7, `pair` 40.4; line at 35.5 and 38.5 (§1.1) |
| x4m together: `both_cpu_gbs` / `both_gpu_gbs` | 19.5; 16.0–22.5 / 17.5; 13.0–20.5 | B | CPU: 18.7 against 22.1 |
| x4m together: `both_cpu_gbs` ÷ CPU alone, both at 2 columns | 0.89; 0.80–1.00 | B | 0.83 against 0.98; no test if CPU alone at 2 columns < 19 |
| x4m together, 1 column: `sum_vs_cpu_alone` | 1.61; 1.50–1.72 | B (R74 1.611) | – |
| x4s lead: best ratio over 0.20 to 0.60 | 0.72 to 0.73; 0.67–0.79 | both | – |
| x4s lead: best share | 0.50 (P 0.60 to 0.78) | both | weight on 0.55: 0.09 against 0.20 |
| x4s share 0.55: ratio; `gpu_ms`; `wait_ms` | 0.78; 1.38; 0.21 | both | – |
| x4s share 0.55 / 0.60 `gpu_dev_ms` | 1.08 to 1.13 / 1.18 to 1.23 | B / E | contention 1.10 against 1.15 |
| x4s in-situ K (`gpu_ms` − `gpu_dev_ms`, GPU-bound share) | 300; 200–380 | B | – |
| x4s P(optimum interior) | 0.85 to 0.90 | both | – |
| x4e flag seen within 1 ms, any mode incl. C11 | P 0.55 | E (B 0.35 without C11) | accepted |
| x4e C11: builds and runs / flag seen given runs | P 0.55 / 0.75 | E | not reviewed by me beyond §7.1 |
| x4e C11 `rtt_p50_us`; counter `iv_p50_us` (given seen) | 20; 3–200 / 5; 1–60 | E | – |
| x4e all three lines pass in some mode | P 0.10 to 0.15 | B / E | – |
| hand-off extra µs per join: write + read / map + unmap / no call | 90; 40–200 / 60; 25–150 / 5; 0–30 | E | – |
| hand-off: no-call variant correct / best ≤ 20 µs and correct | P 0.55 to 0.60 / 0.50 | B / E | – |

Lines that go with the table: cg lines in r3 §1.2 (Ebony: print the `pair` column with her 60 µs slack; agreed);
model line in §1.1 here; design kill and build values in r2 §2.3; K to stack in r3 §2, with **K = launch dead time
+ hand-off**.
