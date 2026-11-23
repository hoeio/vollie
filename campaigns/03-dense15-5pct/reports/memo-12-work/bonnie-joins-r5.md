# c3 memo-12 work, round 5 (Bonnie): corrected merged R75 table, ggml's two-column kernel, cap-following share, GPU throttle, width 3, file-backed arm

- **Date:** 2026-11-23. **Author:** bonnie, `claude-fable-5-2`. Follows [bonnie-joins-r4.md](bonnie-joins-r4.md);
  reviews [ebony-joins-r4.md](ebony-joins-r4.md) §1 (read in full). Not committed. No verdict here.
  **Ebony has not reviewed this file.**
- **Labels.** *device* = printed by the phone. *derived* = arithmetic on *device* fields. *host* = laptop.
  *estimated* = model, count or judgement. *external* = another group's number or text. *source* = read in the
  local tree. Triples are Yi / Llama-2 / DeepSeek.
- **Script:** [bonnie_joins_r5.py](bonnie_joins_r5.py) (`python3 -I -B`, self-check passes; loads
  `bonnie_tree_r2.py` unchanged).
- **State of the kit.** §1 and §3 are keyed on `patches/llama.cpp-c3/0007-x4-join-probe.patch` as saved at 19:15
  (93 KB) and checked against `tom-r35-build.md` as saved at 19:19, which now describes x4m alone (`x4 mc`,
  `x4 mcid`, `x4 mcsum`), the together cell (`x4 tog`, `x4 tog2sum`, `x4 tog1sum`), the C11 mode (`mode=svm_c11`),
  x4s to 0.60 (22 arms) and the hand-off cell (`x4 handoff`). Source and note agree on every field I use.

## 0. Answer first

1. **Ebony's kernel reading is right, and I withdraw my centre.** At ne1 = 2 the CPU arm runs KleidiAI's four-row
   tile, at **2.60 × a gemv per weight by issue count** (65 slots against 25), 2.3 to 2.6 on the device scale. Her
   line reference is the F32 function; the Q4_0 call goes through `compute_forward_qx`, with the same result.
   Two, three and four columns cost the same. CPU alone at two columns: **14.1 GB/s** (12.2 to 16.5); P(under 19)
   0.90.
2. **My sum line (38.5 / 35.5) cannot be read on the cell as built. Her GPU-ratio line can, with one change and
   one caveat.** Change: when the CPU alone at two columns reads under 19, the `bonnie` reading is the GPU ratio
   ≤ 0.90 alone (the sum-ratio half of her rule cannot be met: my own model gives 0.905 there). Caveat: the tile
   makes the CPU draw 14 GB/s of bus, not the 20 to 22 a lane would draw, so the cell leans towards `pair`.
3. **Cheapest change to the cell: one mixed arm, CPU at one column beside the GPU's `_mc3` at four columns.** No
   new kernel. `both_cpu_gbs` there against the one-column cell: 0.98 in my model, 1.28 in hers.
4. **The W-position CPU kernel is now a build item of any width-2 lane**, not an option (agreed with Ebony). Stock
   ggml at batch 2 costs the CPU more than two separate gemvs.
5. **A share that follows the clock cap is worth +0.24 to +0.45 tok/s per 5 GB/s of cap mismatch** and costs 0.16
   to 0.25 GB of rows held twice (r4 said 0.3 to 0.4). It also makes numerics depend on the phone's heat.
6. **Adreno throttle: no record shows it, in about 80 s of GPU load.** R74's `gpu_max_mhz` stayed 840 in all 29
   lines that print it while the A78 cap fell twice. Public reviews of this phone show the GPU dropping after about one
   minute of a 3D load. R75 already prints what is needed in three blocks; one field is missing in the others.
7. **Width 3 on the lane is closed at the merged centre** (`dev_ratio(3)` 1.5 to 1.65 loses on all three). At 1.3
   it wins only on Llama-2 in my model and nowhere in hers.
8. **The file-backed residency arm** is designed in §7 (a later kit; needs owner questions (a) and (b)).

## 1. Item 0: the corrected merged R75 table (*estimated*; centre; 80 % band)

Correction to r4 §7.2: I wrote that Ebony had given no bands for x4a, x4b, x4f, x4c, x4d, x4r and x4i. Her r2
part B.3 gives them. This table replaces r4 §7.2 and r2 §2.1 / §2.2. Where two centres are printed, the band is
the union. "field" is the printed field in the patch as of 19:15; **none** = no such line exists.

### 1.1 Events

| event | B | E | merged | field |
|---|---|---|---|---|
| x4i recordable queues in the extension string | 0.99 | 0.99 | 0.99 | `x4 ext text=` |
| x4i `present=1`, `max_size` > 0 | 0.80 | 0.80 | 0.80 | `x4 recq` |
| x4i `svm_coarse` / `svm_fine_buffer` / `svm_fine_system` | 0.85 / 0.30 / 0.05 | 0.90 / 0.70 / 0.05 | 0.90 / **0.60 to 0.70** / 0.05. I move to 0.60: her vendor text is first-hand | `x4 dev` |
| x4i `svm_atomics` | – | 0.65 | 0.65 (hers; accepted) | `x4 dev` |
| x4d runs (`err=0`) / passes (replay ≤ 0.5 × sync1) | 0.55 / 0.25 | 0.55 / 0.20 | 0.55 / 0.20 to 0.25 | `x4 rec setup ok=`, `x4 recsum replay_vs_sync1=` |
| x4c identity ok on both stacked shapes | 0.90 | 0.90 | 0.90 | `x3 identity` (m 22016, m 12288) |
| x4e flag seen within 1 ms, modes as built (1.x atomics) | 0.35 | 0.40 | 0.35 to 0.40 | `x4 pk test=flag seen= rtt_p50_us=` |
| x4e flag seen, any mode with the C11 variant | – | 0.55 | 0.55 | `x4 pk test=flag mode=svm_c11 seen=` |
| x4e C11: builds and runs / flag seen given runs | – | 0.55 / 0.75 | hers | `x4 pk c11 err= svm_atomics=`; failure `reason=build_c11` |
| x4e all three lines pass in some mode | 0.10 | 0.12 to 0.15 | 0.10 to 0.15 | `x4 pk` flag, counter, `x4 pksum` |
| x4e a child hangs or times out | 0.35 | 0.35 | 0.35 | `x4 child rc=-1001 timeout=1` |
| x4a the driver serialises | 0.60 | 0.50 | 0.50 to 0.60 | `x4 tl submit_us=`, `end_us=` |
| x4b best `ovh_us` ≤ 250 / ≤ 150 / ≤ 100 | 0.80 / 0.12 / 0.02 | 0.75 / 0.20 / 0.03 | 0.75 to 0.80 / 0.12 to 0.20 / 0.02 to 0.03 | `x4 rt ovh_us=` |
| x4b 0-3 spin `submit_to_start_us` ≥ 30 below cpu 4 spin | 0.50 | 0.45 | 0.45 to 0.50 | `x4 rt submit_to_start_us=` |
| x4f the driver prepares at enqueue | 0.40 | 0.30 | 0.30 to 0.40 | `x4 la set_to_start_p50_us=` against `submit_to_start_p50_us=` |
| x4m kernel created / runs / identity given runs / all | 0.92 / 0.85 / 0.88 / 0.75 | = / = / 0.90 / 0.76 | 0.92 / 0.85 / 0.88 to 0.90 / 0.75 | `x4 mc ok=0 reason=`, `x4 mc err=`, `x4 mcid ok=` |
| x4s the optimum is interior | 0.90 | 0.85 | 0.85 to 0.90 | `x4 splitsum ratio=` over shares |
| x4r held / killed | 0.35 / 0.10 | 0.25 / 0.15 | 0.25 to 0.35 / 0.10 to 0.15 | `x4 res sum`, `x4 res done`; killed = no `x4r end` |
| hand-off: no-call variant correct / best ≤ 20 µs and correct | 0.55 | 0.60 / 0.50 | 0.55 to 0.60 / 0.50 | `x4 handoff mode=ahp_persist match=`; `extra_us=` |
| together: CPU alone at 2 columns reads under 19 GB/s | **0.90** | 0.65 | 0.65 to 0.90 | `x4 tog2sum cpu_alone_gbs=` |
| together: the line reads `pair` / undecided / `bonnie` | 0.60 / 0.20 / 0.20 | 0.50 / 0.25 / 0.25 | 0.50 to 0.60 / 0.20 to 0.25 / 0.20 to 0.25 | derived from `x4 tog2sum` (§2.3) |

### 1.2 Cells

| cell, field | B | E | merged | field |
|---|---|---|---|---|
| x4a `gap_p50_us`, launcher 0-3 | 200; 145–270 | 200; 110–270 | 200; 110–270 | `x4 tlsum case=batch8 lcpu=0-3` |
| x4a `gap_p50_us`, launcher cpu 4 | 210; 140–285 | 185; 100–285 | 185 / 210; 100–285 | `x4 tlsum lcpu=4` |
| x4a flush after each enqueue: change of gap | −10; −60 to +30 | = | = | `case=batch8flush` − `batch8` |
| x4a `s2b_first_us` (0-3) | 190; 130–220 | = | = | `x4 tlsum` |
| x4a `enq_us` per launch, 0-3 / cpu 4 | 125 / 62; 85–185 / 35–115 | = | = | `x4 tlsum enq_us=` |
| x4a layer of 7, (wall − Σ dev) ÷ 7, 0-3 / cpu 4 | 235 / 215; 150–300 / 130–290 | 235 / 200 | 235 / 200 to 215 | `case=layer7`: (`wall_us` − `dev_sum_us`) ÷ 7 |
| x4a gap, layer launches − 2816-row launches | – | +40; 0 to +120 | hers; accepted (R74: 199 → 320 µs from 2.4 to 25 MB, *device*) | `gap_p50_us` layer7 − batch8 |
| x4b `ovh_us` 0-3: finish / spin / wfe | 400 / 265 / 395; 320–500 / 190–360 / 310–500 | = | = | `x4 rt wait=` |
| x4b `ovh_us` cpu 4: finish / spin / wfe | 310 / 255 / 305; 225–410 / 150–345 / 220–410 | = / 235 (130–345) / = | spin 235 / 255; 130–345 | `x4 rt` |
| x4b `ovh_us` cpu 7: finish / spin / wfe | 295 / 250 / 290; 215–390 / 145–335 / 210–390 | = / 235 (130–345) / = | spin 235 / 250; 130–345 | `x4 rt` |
| x4b best `ovh_us` of nine | 215; 140–280 | 205; 120–290 | 205 / 215; 120–290 | min of `ovh_us`, `rep=0` |
| x4b `queued_to_start_us` 0-3 finish / 0-3 spin / cpu 4 spin | 340 / 240 / 240; 275–425 / 175–330 / 140–320 | = | = | `x4 rt` |
| x4b `submit_to_start_us`, same three | 162 / 120 / 177; 125–200 / 75–195 / 75–220 | = | = (bimodal) | `x4 rt` |
| x4b p90 − p50, finish / spin (scored: the jitter term) | 150 / 90; 60–300 / 30–250 | = | = | `p90_us` − `p50_us` |
| x4b launcher CPU per call, finish on 0-3; `dev_us` | 300; 200–450. 240; 225–265 | = | = | tail `lcpu_us=`; `dev_us=` |
| x4f `set_to_start_p50_us`, cpu 4 | 210; 110–280 | 215 | 210 to 215; 110–280 | `x4 la mode=gate`. **The kit runs cpu 4 only; the 0-3 value (190) has no cell** |
| x4f `rtargs` | −8 on `ovh_us`; −30 to +5 | = | **row changed:** `enq_us` at `args=2` − at `args=15`: −10; −22 to −3 (mine, new) | `x4 rtargs enq_us=`. The kit has 2 args, not 3, and its note says not to read `ovh_us` |
| x4c gate‖up cold GB/s; `gate_up_vs_11008` | 24.3; 1.14 (1.04–1.20) | 23.5; 1.10 (1.00–1.20) | 23.5 / 24.3; 1.10 / 1.14 (1.00–1.20) | `x4 stacksum` |
| x4c q‖k‖v GB/s; `qkv_vs_4096`; against 11008 | 21.6; 1.42; 1.02 (19.0–25.5; 1.25–1.60; 0.93–1.10) | = | = | `x4 stacksum`; last = `qkv_gbs` ÷ `m11008_gbs` |
| x4d `ovh_us_per_launch_replay`; `replay_vs_sync1` | 138; 0.87 (26–230; 0.74–0.97) | 170; 0.80 (40–250; 0.55–0.97) | 138 / 170; 0.80 / 0.87 (26–250; 0.55–0.97) | `x4 recsum` |
| x4e flag `rtt_p50_us`; counter `iv_p50_us` (given seen) | 30; 5–300. 10; 1–100 | = | = | `x4 pk` |
| x4e C11 `rtt_p50_us`; `iv_p50_us` | – | 20; 3–200. 5; 1–60 | hers | `x4 pk mode=svm_c11` |
| x4e `q4p48_vs_q4n` | 0.85; 0.50–1.02 | = | = | `x4 pksum` |
| x4m `dev_ratio` 2 / 3 / 4 columns | 1.2 / 1.5 / 1.9 | 1.3 / 1.65 / 2.0 | 1.25; 1.05–1.80 / 1.5 to 1.65; 1.15–2.6 / 1.9 to 2.0; 1.35–3.4 | `x4 mcsum dev_ratio=` |
| x4m `wall_ratio` 2 columns; max_rel_err | 1.15. 6e-4 to 8e-4 | 1.22. = | 1.15 / 1.22; 1.04–1.65. = | `x4 mcsum wall_ratio=`, `x4 mcid max_rel_err=` |
| x4s `t_cpu_alone_ms` | 1.90; 1.80–2.05 | = | = | `x4 splitsum` |
| x4s feeder, share 0.25: ratio; `gpu_ms`; `wait_ms` | 0.86; 1.00; 0.02 (0.83–0.90; 0.85–1.15; 0–0.06) | = | = | `x4 splitsum mode=feeder` |
| x4s lead: best ratio over 0.20 to 0.60; best share | 0.73; 0.70–0.79. 0.50 (0.78) | 0.72; 0.67–0.79. 0.50 (0.60) | 0.72 to 0.73; 0.67–0.79. 0.50 | `x4 splitsum mode=lead` |
| x4s lead, share 0.20 / 0.35 ratio | 0.92 / 0.82; 0.88–0.96 / 0.79–0.86 | = | = | same |
| x4s share 0.50: `gpu_ms`; `gpu_dev_ms`; `wait_ms`; `enq_us` | 1.29; 0.985; 0.04; 65 | =; 1.03 (0.92–1.10); 0.08 (0.02–0.25); = | 1.29 (1.15–1.42); 0.985 / 1.03 (0.90–1.10); 0.04 / 0.08 (0–0.25); 65 (35–125) | `x4 split` |
| x4s share 0.55: ratio; `gpu_ms`; `wait_ms`; `gpu_dev_ms` | 0.78; 1.38; 0.21; 1.08 | =; =; =; 1.13 | 0.78 (0.72–0.84); 1.38 (1.28–1.46); 0.21 (0.11–0.29); 1.08 / 1.13 (0.98–1.22) | `x4 split share=0.55` (the 18:55 note stops at 0.50) |
| x4s share 0.60: the same four | 0.83; 1.48; 0.39; 1.18 | =; =; =; 1.23 | 0.83 (0.77–0.90); 1.48 (1.38–1.56); 0.39 (0.29–0.47); 1.18 / 1.23 (1.07–1.33) | `x4 split share=0.60` |
| x4s in-situ K = `gpu_ms` − `gpu_dev_ms`, GPU-bound share | 300; 200–380 | 290; 170–400 | 290 / 300; 170–400 | derived |
| x4s GPU device-time rate beside the CPU, best share | 25.7 | 24.6; 22.5–27.5 | 24.6 / 25.7; 22.5–27.5 | `gpu_rows` × 4096 × 0.5625 B ÷ `gpu_dev_ms` |
| x4s `cpu_ms` ÷ (CPU rows' share × `t_cpu_alone_ms`), share 0.25 / 0.50 | 1.14 / 1.32 (my fit, 1 + 0.45 × duty) | 1.14 / 1.30; 1.08–1.22 / 1.20–1.40 | hers; agreed | derived |
| x4r (if held) `cpu_gbs_median`; majflt median | 18.5; 0 (17.5–19.5; 0–5) | = | = | `x4 res sum` |
| hand-off extra µs per join: write + read / map + unmap / no call | – | 90; 40–200 / 60; 25–150 / 5; 0–30 | hers | `x4 handoff extra_us=`, modes `write` / `map`, `ahp_map` / `ahp_persist` |
| together, 1 column: `cpu_alone_gbs`; both CPU / GPU; `sum_vs_cpu_alone` | 25.6; 19.8 / 21.5; 1.61 (1.50–1.72) | = | = | `x4 tog1sum` |
| together, 2 columns: CPU alone | **14.1; 12.2–16.5** (was 22.0) | 15.5; 13.0–22.5 | 14.1 / 15.5; 12.2–22.5 | `x4 tog2sum cpu_alone_gbs=` |
| together: CPU alone at 2 columns ÷ at 1 column | 0.55; 0.47–0.65 | 0.60; 0.50–0.88 | 0.55 / 0.60; 0.47–0.88 | `cpu_alone_gbs` tog2sum ÷ tog1sum |
| together, 2 columns: GPU alone | 20.4; 15.0–23.5 | 19.8; 14.5–23.5 | 20; 14.5–23.5 | `x4 tog2sum gpu_alone_gbs=` |
| together, 2 columns: `both_cpu_gbs` | **13.7; 11.7–16.0** (was 19.5) | 15.0; 12.5–22.0 | 13.7 / 15.0; 11.7–22.0 | `x4 tog2sum both_cpu_gbs=` |
| together, 2 columns: `both_gpu_gbs` | 18.5; 13.5–21.5 | 18.6; 13.5–22.0 | 18.6; 13.5–22.0 | `x4 tog2sum both_gpu_gbs=` |
| together, 2 columns: sum | **32.5; 27.5–36.5** (was 37.0) | 33.5; 28.0–41.0 | 32.5 / 33.5; 27.5–41.0 | `x4 tog2sum sum_gbs=` |
| together, 2 columns: GPU both ÷ GPU alone (**the model line**) | 0.95; 0.85–1.00 | 0.94; 0.85–1.00 | 0.95; 0.85–1.00. Models: `bonnie` 0.87, `pair` 1.00 | `both_gpu_gbs` ÷ `gpu_alone_gbs`; cross-check `gpu_alone_dev_us` ÷ `both_gpu_dev_us` |
| together, 2 columns: sum ratio (both ÷ alone) | 0.95; 0.88–1.00 | 0.95; 0.85–1.00 | 0.95; 0.85–1.00. Models at c 2.5: 0.905, 0.988 | `sum_gbs` ÷ (`cpu_alone_gbs` + `gpu_alone_gbs`) |
| together, 2 columns: `sum_vs_cpu_alone` as printed (÷ CPU alone at **two** columns) | 2.30; 1.9–2.8 | not given | mine only. It is not r4's 1.44 row, which divided by the one-column CPU | `x4 tog2sum sum_vs_cpu_alone=` |
| mixed arm, if built (§2.4): `both_cpu_gbs` at (CPU 1, GPU 4 columns) ÷ at (1, 1) | 1.15; 0.97–1.30 | not seen by her | mine only. Models: 0.98, 1.28 | **none** (arm not in the kit) |

Lines that go with the table: cg lines in r3 §1.2 (print the `pair` column with 60 µs slack); the model line in
§2.3 here (it replaces r4 §1.1); design kill and build values in r2 §2.3 with Ebony's B.4 changes (mailbox line
100 µs for flag and counter together; "not seen" kills only if the C11 variant ran); K = launch dead time +
hand-off.

## 2. Item 1: what ggml runs at ne1 = 2 on the A78, and what the together cell can show

### 2.1 The kernel (*source*)

- **Path.** The probe's CPU arm is a ggml graph of `ggml_mul_mat` on a Q4_0 tensor in the KleidiAI buffer type
  (`cpu_buft=CPU_KLEIDIAI`, *device*, R74). `compute_forward` sends Q4_0 to `compute_forward_qx`
  (`kleidiai.cpp` line 822 in `lab/llama.cpp-c3r32`, whose ggml sources are the kit's). There
  `is_gemv = src1->ne[1] == 1` (line 1148) and the kernel is `is_gemv ? gemv : gemm` (line 1197).
  Ebony cites lines 849 to 964, which are `compute_forward_f32`; the F32 function has the same test, so her
  conclusion holds, but the Q4_0 lines are 1137 to 1491.
- **Which kernels.** On Android the table's dotprod entry (kernels.cpp lines 685 to 727) gives
  `…qsi8d32p4x4_qsi4c32p4x4_16x4_neon_dotprod` as gemm and `…qsi8d32p1x4_qsi4c32p4x4_1x4_neon_dotprod` as gemv.
  The entries before it need SVE or i8mm. The A78 and X1 are Armv8.2 cores with dotprod and without i8mm; this
  was my reading in memo-08 and a web search today did not find a source that states it either way. The phone's
  log line `kleidiai: primary q4 kernel feature` would settle it; no report of ours prints it.
- **What the gemm kernel does with two rows.** KleidiAI v1.31.0 (`ggml-cpu/CMakeLists.txt` line 653), assembly
  read in `lab/llama.cpp-c1h6/build-kai/_deps/kleidiai-src` (same version). With m < 16 the kernel leaves its
  16-row loop and runs a four-row tile (labels 14 to 18): per 64 bytes of weights it loads 4 weight vectors and 8
  activation vectors and issues **32 SDOT** (8 per activation row, for 4 rows), then 4 FMUL, 4 SCVTF and 4 FMLA.
  With two rows the other two rows of the packed tile are padding: computed and not stored. The weights are read
  once.
- **Count** (A78 rules of `tools/c3_a78_pipes.py`; per 128 weights):

  | kernel | V0 / V1 / either | slots | floor, cycles | against the gemv |
  |---|---|---|---|---|
  | gemv (memo-10 E2) | 6 / 4 / 15 | 25 | 12.5 | 1.00 |
  | tile at ne1 = 2, 3 or 4 | 12 / 4 / 49 | 65 | 32.5 | **2.60** |
  | W-position kernel, 2 positions (memo-10 E2, `ebony_gw.c`) | 6 / 4 / 26 | 36 | 18.0 | 1.44 |
  | two gemv calls | – | 50 | 25.0 | 2.00, and the weights are read twice |

  The gemv measured 15.1 cycles per block against its floor of 12.5 (*device*, R72). If the 2.6 extra cycles add,
  the tile is 2.32 ×; if they scale, 2.60 ×. **c for ggml at two columns: 2.5 (2.3 to 2.9).** Ebony's 2.3 to 2.6
  is confirmed.
- **Consequences.**
  1. Stock ggml at batch 2 costs the CPU more than two separate gemvs in compute. A width-2 pass through
     unmodified ggml is compute-bound: 14.1 GB/s alone against 25.6 at one column (*estimated*).
  2. Three and four columns cost the same as two. A prefill batch pays 0.65 × a gemv per position
     (Ebony r4 §3 uses this).
  3. Every stack since memo-10 assumes c(2) = 1.45. That is the purpose-built kernel, exact on random data only
     (*host*). It must be built and checked inside llama.cpp before any width-2 price is a device prediction.

### 2.2 What the cell as built reads (`bonnie_joins_r5.py` part 1; wall GB/s; *estimated*)

| CPU cost c | model | CPU alone | GPU alone | both CPU | both GPU | sum | GPU ratio | sum ratio |
|---|---|---|---|---|---|---|---|---|
| 1.45 (not this kit) | `bonnie` cg 1.2 | 22.5 | 21.1 | 18.4 | 18.4 | 36.8 | 0.869 | 0.843 |
| 1.45 (not this kit) | `pair` cg 1.26 | 22.5 | 20.2 | 19.8 | 20.2 | 40.0 | 1.000 | 0.936 |
| 2.5 | `bonnie` cg 1.2 | 14.1 | 21.1 | 13.6 | 18.4 | 31.9 | 0.869 | 0.904 |
| 2.5 | `bonnie` cg 1.45 | 14.1 | 17.8 | 13.5 | 15.5 | 29.0 | 0.869 | 0.908 |
| 2.5 | `pair` cg 1.26 | 14.1 | 20.2 | 13.7 | 20.2 | 34.0 | 1.000 | 0.988 |
| 2.3 / 2.9 | either | 15.2 / 12.4 | – | 14.5–14.7 / 11.9–12.1 | – | 30.0–35.0 / 27.4–32.3 | as above | 0.905–0.990 |

- The CPU columns do not separate the models (13.6 against 13.7). The sum separates them by 2 GB/s, less than
  the spread from cg. **My r4 line on the sum is withdrawn** unless the CPU alone at two columns reads ≥ 21
  (P 0.10).
- Here cg is the alone `dev_ratio`. In this table `pair` lets the GPU run at min(24.6, 28.32 ÷ cg).

### 2.3 The reading line (Ebony's, with my two changes)

| reading at two columns, four rates from the same block | says |
|---|---|
| GPU both ÷ GPU alone ≥ 0.95, or sum ratio ≥ 0.94 | `pair` |
| GPU ratio ≤ 0.90; and sum ratio ≤ 0.89 **only if the CPU alone at two columns reads ≥ 19** | `bonnie` |
| between | undecided; print both stack columns |

- **Change 1.** With the tile, my own model gives a sum ratio of 0.905, so her "and" could never read `bonnie`.
  Below 19 GB/s the GPU ratio decides alone.
- **Change 2, a caveat on `pair`.** My `bonnie` variant carries a measured factor (0.869) with no mechanism. With
  the tile the CPU draws 14 GB/s of bus instead of 19 to 22. Any physical contention model then lets the GPU run
  freer. So GPU ratio ≥ 0.95 in this cell shows "no contention at a CPU draw of 14", not at the lane's 20 to 22.
  Written as: **a `pair` reading from the tile cell is `pair` at light CPU draw**; the stack prints both columns
  unless the mixed arm (§2.4) agrees.
- I cannot put this to Ebony inside this run. Her line is accepted as the primary reading; the two changes are my
  proposal to her and to the lead.

### 2.4 The cheapest change to the cell

**One mixed arm: CPU through ggml at ne1 = 1 (the gemv, bus-bound, drawing all it can) beside the GPU's `_mc3`
at n_cols 4.** Both kernels are already in the kit; the arm is the one-column `tog` arm with the GPU kernel
swapped. About 3 s of phone time for an ABBA pair.

| arm (CPU columns, GPU columns) | `both_cpu_gbs` `bonnie` / `pair` | ÷ the (1, 1) cell | GPU both ÷ GPU alone |
|---|---|---|---|
| (1, 1), R74 | 19.77 (*device*) | 1.00 | 0.865 (*device*) |
| (1, 2), cg 1.25 | 19.6 / 21.0 | 0.99 / 1.06 | 0.87 / 1.00 |
| (1, 3), cg 1.6 | 19.4 / 24.2 | 0.99 / 1.23 | 0.87 / 1.00 |
| **(1, 4), cg 1.95** | **19.4 / 25.2** | **0.98 / 1.28** | 0.87 / 1.00 |

- **Line:** CPU ratio ≥ 1.15 → the bus is conserved (`pair`); ≤ 1.05 → the CPU keeps its one-column share
  (`bonnie`). R74's two `both` arms differed by 2.7 % (*device*), so 1.05 is outside noise.
- It reads nothing if x4m shows `dev_ratio(4)` under about 1.3 (the GPU is then not slowed enough to free bus).
  P of that: 0.10.
- The arm brackets the lane from the heavy side (CPU draw 25), the tile cell from the light side (14). The lane
  with the W-position kernel sits between (20 to 22). If both read `pair`, write `pair`. If they differ, the
  truth is a contention that grows with total draw, and the stack is between the columns.
- **Not cheapest, but the real cell:** the W-position kernel in the probe (`ebony_gw.c`, compiled into libllama
  with a dotprod target attribute, since the kit ships r33's CPU module byte-identical). It needs the kernel's
  packed layout exposed to the probe. I do not ask for it in r35.
- **Free in the cell as built:** the CPU-alone arm at two columns is the first *device* value of ggml's batch-2
  cost. If Tom's arm takes ne1 as a parameter, one CPU-alone arm at ne1 = 4 checks the reading (same rate as at
  2, within 5 %).

## 3. Item 2: do the rows map to printed fields? (patch of 19:15, build note of 19:19)

**Every row of §1 now maps to a printed field or to arithmetic on printed fields, except the rows below.**

| row | state |
|---|---|
| mixed arm (§2.4) | no arm. In `tog()` the CPU and GPU column counts are one variable `n`; the arm needs them apart, an input image of 4 columns and an output buffer of 4 × 11008 floats |
| x4f `set_to_start_p50_us` on launcher 0-3 (my r2 centre 190) | no cell: x4f runs on cpu 4 only. Dropped from §1.2 |
| x4f `rtargs` "3 args against 15, on `ovh_us`" | built as 2 args against 15 and to be read on `enq_us`. Row rewritten in §1.2 |
| r4 §1's "sum ÷ `cpu_alone_gbs` at one column" (1.44) and "sum at 2 columns ÷ sum at 1 column" (0.90) | no single field; derived across `tog2sum` and `tog1sum`. `tog2sum sum_vs_cpu_alone` divides by the two-column CPU (new row in §1.2). Both r4 rows are superseded by the tile reading anyway |
| thermal state after each block (Ebony r3 §2.5) | `thermal_pwrlevel` is printed before each parent block (`x3 cond`), not after. Android thermal status is not printed (only headroom, `hr=`) |
| GPU thermal cap in launcher cells and x4r | `gpu_max_mhz` is in the `astr` tail only (x4c, x4d, x4m together, x4s, hand-off arms). The launcher tail (`tl`, `rt`, `la`, `rtargs`) and the `x4 res pass` lines do not print it (§5) |

Notes on rows that do map, for whoever writes the pre-registration:
- **Together cell as built:** two columns first, then one. Six arms each (cpu, gpu, both, both, gpu, cpu), 0.5 s
  warm + 1.5 s measured. CPU: 64 copies of 4096×4096 through ggml with x of n columns (so the tile at n = 2). GPU:
  11008×4096 cold ring, **8 gemvs then one `clFinish`** per call. The CPU-alone and GPU-alone arms at two columns
  are in the same block, as Ebony's line needs. If `_mc3` cannot be created, `x4 tog2 ok=0 … then=n_cols_1_only`.
- **Hand-off cell as built:** 6,656 GPU rows, lead thread on cpu 4 alone, no CPU rows. Modes `none`, `write`,
  `map`, `ahp_map`, `ahp_persist`. `extra_us` = wall − the `none` wall, so it includes any change in the launch
  itself. `match` is a bitwise compare with the write-mode reference. Ebony's three rows map to `write`; `map`
  and `ahp_map`; `ahp_persist`. "Correct rows" = `match=1`.
- **C11 mode:** `mode=svm_c11`, with `x4 pk c11 … err= svm_atomics=` saying whether its program built.
- **x4s to 0.60:** 22 arms (note, arm order). The note's phone-time table still says "18 arms" for x4s and has
  no time for the together or hand-off cells unless Tom has added them; by the arm lengths they are about 24 s
  (together), 15 s (cg cell) and 13 s (hand-off) of measured load (*estimated*).
- **One difference from R74's `tog`:** the together cell's GPU feeder uses the profiling queue (x3c used the plain
  queue). The one-column band (1.50 to 1.72 around R74's 1.611) stands, but a miss low by a few percent would
  have this as a candidate cause.

## 4. A share that follows the clock cap (`bonnie_joins_r5.py` part 2; *estimated*)

Stage model of `bonnie_tree_r2.py` with the share fixed at the optimum for a design C and the clocks at another.
K 300, cg 1.25, width 2 with the W-position kernel. tok/s, Yi / Llama-2 / DeepSeek.

| design C → real C | `bonnie` | `pair` | GPU share (Llama-2) `bonnie` / `pair` |
|---|---|---|---|
| 40 → 40 (following) | 11.16 / 11.07 / 10.69 | 11.88 / 11.88 / 11.43 | 0.347 / 0.355 |
| 35 → 35 (following) | 10.84 / 10.75 / 10.41 | 11.41 / 11.40 / 11.00 | 0.371 / 0.384 |
| 31 → 31 (following) | 10.48 / 10.41 / 10.10 | 10.94 / 10.91 / 10.55 | 0.394 / 0.417 |
| 40 → 35 | 10.60 / 10.54 / 10.23 | 10.99 / 10.97 / 10.62 | fixed |
| 40 → 31 | 9.92 / 9.87 / 9.61 | 10.10 / 10.06 / 9.78 | fixed |
| 35 → 40 | 10.87 / 10.81 / 10.44 | 11.43 / 11.42 / 11.00 | fixed |
| 35 → 31 | 10.19 / 10.11 / 9.83 | 10.50 / 10.46 / 10.15 | fixed |

- **Following is worth +0.24 to +0.45 tok/s for one 5 GB/s step of mismatch, +0.56 to +0.84 for two.** A share
  tuned for C 40 loses 0.24 to 0.42 at C 35. A share tuned for C 35 gets no faster at C 40 (10.87 against 10.84:
  the GPU side binds), so it gives up 0.29 to 0.45 there.
- **Bytes.** The share moves by 0.047 (`bonnie`) to 0.062 (`pair`) between C 40 and C 31. Those rows must exist
  in both layouts: **0.16 to 0.21 GB (Yi), 0.19 to 0.25 GB (Llama-2, DeepSeek).** This replaces r4's 0.3 to
  0.4 GB, which was for a range of 0.10.
- **Where the copy lives.** As GPU buffers it is unreclaimable. As a second anonymous CPU copy it takes Llama-2 to
  4.02 + 0.25 = 4.27 GB against the 4.2 that held file-backed and the 3.4 that held anonymous (*device*, R69,
  R72). The cheap place is the served file itself: the overlap rows read from the mapped GGUF by the stock
  (not repacked) kernel, 5 to 6 % of the body, zero new bytes. Its speed on the phone is not measured.
- **Cheaper still: one fixed share tuned for C 35 to 37.** With the clock odds of ebony-joins-r4 §3 (C ≥ 40: 0.55;
  33 to 40: 0.35; lower: 0.10) the expected loss against following is about 0.2 to 0.3 tok/s.
- **Numerics.** A row computed on the GPU and on the CPU does not give the same bits (half-precision dequantise
  on the GPU; quantised activations on the CPU). A share that follows the cap makes the output depend on the
  phone's heat. By the rule for numerics that differ by mode, the budget check must hold at both ends of the
  share range, and a claim run would have to log the share used. This is a cost of the design, for the lead.
- A share chosen once per run, at its start, from the cap then read, needs the same bytes and the same check. It
  avoids changing layout inside a token.

## 5. Adreno throttle over minutes

**Campaign 1 and 2 records:** only whole-model OpenCL decode rates, 0.7 to 2.9 tok/s (*device*,
`docs/negative-results.md`, `reports/04_campaign.md`). No clock, no time series. Nothing to use.

**R74, read again for this** (all *device*; JSON `384bc4b5…`):

| block | GPU load | `gpu_max_mhz` (kgsl `max_gpuclk`) | GPU rate | A78 cap at the same time |
|---|---|---|---|---|
| x3g, 10 arms, about 32 s, GPU alone | busy 31 to 76 % | 840 in all 10 | hot ÷ cold 1.04 | 2,419 throughout |
| x3c `tog`, 6 arms, about 10 s | busy 86 to 89 % | 840 | 24.60 and 25.16 alone, first and last | 2,419 |
| x3c `split`, 8 arms, about 16 s | busy 34 to 62 % | 840 in all 8 | – | 2,150, then 1,785 |
| x3r, 100 passes, 15 s | GPU ring beside the CPU | not printed per pass | median 21.02 GB/s; first 30 passes mean 20.3, last 30 mean 21.1 (one pass at 6.5) | – |
| before each of the four blocks | – | `thermal_pwrlevel=0`, `devfreq_max` 840 MHz, four times | – | – |

- **About 80 s of GPU load inside 10 minutes; no sign of a GPU cap, while the CPU cap fell twice.** Correction to
  r4 §3: `thermal_pwrlevel=0` was read four times (after each gate), not once.
- **What `max_gpuclk` means** (*external*, kernel source mirrors found by search today; versions differ from the
  phone's kernel): the node reports the frequency of the entry at the thermal power level, so it shows the
  thermal cap. A cooling request can also arrive through devfreq, which the same driver folds in.
- **Public record for this phone** (*external*, 3DMark Wild Life stress test, 20 one-minute loops): stability
  56.4 % (HotHardware), 64.1 % with the drop "after the first run" (sogi.com.tw: best loop 5,419, lowest 3,474),
  71 % (UL's median of user runs). That is a 3D load at full GPU power; the Adreno 660 is reported at up to 11 W
  against 7.2 W for its predecessor (*external*, Notebookcheck citing XDA). No source gives a clock trace.
- **Reading** (*estimated*). A game drops the GPU after about one minute. Our lane load is a 32-token run: 3 to
  4 s of GPU work at 75 to 90 % busy after a CPU-only prefill, then a gate of 60 to 180 s. P(the GPU's cap falls
  inside one claim run from a gated start) 0.05. P(it falls somewhere in a 10-run block) 0.2. For sustained
  decoding over minutes, P 0.6 within 2 minutes.
- **Price** (*estimated*). Ebony r3: GPU compute × 0.8 costs 0.45 tok/s at K 290 in `pair`. One kgsl step down
  from 840 MHz is about 7 % of clock (the 660's table as I recall it: 840, 778, 738, 676, 608 …; *recalled*, not
  read today), so about −0.15 tok/s per step, −0.6 at four steps.
- **What R75 gives without change:** `gpu_max_mhz` and `gpu_busy` on every `astr` line: x4c (8 arms), x4m's
  together cell (12 arms, 24 s), x4s (18 to 22 arms, 45 to 80 s of CPU + GPU), the hand-off arms (10), x4d. That
  is the longest GPU-loaded record so far.
  Band: P(`gpu_max_mhz` < 840 on any R75 line) 0.15; P(`thermal_pwrlevel` > 0 on any `x3 cond` line) 0.10.
- **Cheapest addition:** print `gpu_max_mhz` in the launcher tail and on the x4r pass lines (one sysfs read
  each). x4r is the last and hottest block and today prints only `gpuclk`, which the phone refuses (EACCES).
  r4's question 4 asked for a 60 s GPU arm; with this field it is not needed in r35.

## 6. Width 3 on the lane (`bonnie_joins_r5.py` part 3; *estimated*)

Width 2 at cg 1.25 against width 3 at a given `dev_ratio(3)`; CPU with the W-position kernel (c(3) = 1.9); tree
of two candidates, best shape; trained head, C7, C5, O 5.2 at centre.

| K µs | model | width 2 | width 3 at cg(3) 1.3 | at 1.6 | cg(3) below which width 3 wins |
|---|---|---|---|---|---|
| 0 | `bonnie` | 13.36 / 13.01 / 12.46 | 13.30 / 13.59 / 12.63 | 12.27 / 12.50 / 11.69 | 1.29 / 1.45 / 1.35 |
| 300 | `bonnie` | 11.17 / 11.07 / 10.71 | 10.97 / 11.44 / 10.74 | 10.42 / 10.81 / 10.18 | 1.21 / 1.46 / 1.31 |
| 0 | `pair` | 14.91 / 14.59 / 13.86 | 14.24 / 14.59 / 13.49 | 13.21 / 13.49 / 12.55 | 1.16 / 1.30 / 1.21 |
| 300 | `pair` | 11.89 / 11.89 / 11.44 | 11.44 / 11.98 / 11.22 | 10.92 / 11.38 / 10.69 | 1.10 / 1.34 / 1.20 |

- **At the merged centre (1.5 to 1.65) width 3 loses on all three in both models,** by 0.3 to 1.1 tok/s.
- **At 1.3 it is a Llama-2 lever only:** +0.4 to +0.6 in my model, +0.0 to +0.1 in hers; Yi loses in both.
  Llama-2 has the highest head top-1 (0.497, *host*); width 3 is paid in accuracy at depth 2, which is not
  measured (`TOPK`, `C2` in `bonnie_tree.py` are assumptions).
- A `dev_ratio(3)` of 1.3 would come with a `dev_ratio(2)` near 1.15, which lifts width 2 as well. So the
  comparison above flatters width 3.
- **Line for the pre-registration:** width 3 on the lane is worth a host count only if `dev_ratio(3)` ≤ 1.20;
  my P of that 0.12. r4's threshold of 1.3 is withdrawn. `bonnie_tree_r2.py` needs no change: `toks_w` in the r5
  script carries W 3.

## 7. A file-backed residency arm (design for a later kit; nothing measured)

**Why.** Only Yi holds an anonymous repacked body (3.36 GB against 3.4 held; Llama-2 4.02 and DeepSeek 4.01 do
not; *device*, R72). File-backed capacity read 4.2 GB (*device*, R69 P1), with no GPU buffers present. A lane
moves 1.0 to 1.5 GB into GPU buffers, which cannot be dropped. No arm has measured file-backed pages beside GPU
buffers.

**Arm (working name x5r).**
1. Write a scratch file of F bytes into the app's files (R69's P1 wrote 4.6 GB the same way), map it read-only
   and shared, touch it once.
2. Allocate G bytes of GPU buffers as x4r does (copies of 11008×4096) and write them.
3. 100 passes: four threads on 4-7 run the R72 read kernel over the map while the GPU ring runs the gemv. Per
   pass: `cpu_gbs`, `gpu_gbs`, majflt, MemAvailable, PSI memory, as x3r.
4. **Hold step (new):** stop for 60 s with campaign 2's page hold (R60) on, then one more pass. Then the same
   with the hold off. The gate between claim runs is 60 to 180 s (ebony-joins-r4 §4); eviction there is the risk
   a 15 s arm cannot see.
5. Control: the same sizes with an anonymous arena instead of the map (this is x4r).

| step | file F + GPU G = total | stands for | P(held) |
|---|---|---|---|
| 1 | 2.6 + 1.0 = 3.6 GB | Yi's CPU part with margin; anchor (x3r held 3.5 anonymous + GPU) | 0.85 |
| 2 | 2.6 + 1.4 = 4.0 GB | Llama-2 or DeepSeek at GPU share 0.35 | 0.55 |
| 3 | 3.0 + 1.0 = 4.0 GB | the same at share 0.25 | 0.60 |
| 4 | 2.9 + 1.5 = 4.4 GB | the same with the 0.25 GB overlap of §4 and the head on the lane | 0.25 |

- **Criteria,** as R74 §5: median CPU read ≥ 0.80 × (alone − 0.36 × GPU rate); median majflt per pass ≤ 50;
  `done err=0`. Added: the first pass after each 60 s hold reads ≥ 0.8 × the median.
- **Why file-backed may hold where anonymous does not** (*estimated*): a dropped file page costs one read from
  storage, a swapped anonymous page costs zram work and memory. 50 major faults per pass at an assumed 0.1 to
  0.2 ms each is 5 to 10 ms, about 4 to 8 % of a 120 to 140 ms pass; that is what the ≤ 50 line allows.
- **What the lane then needs** (not part of the arm): the CPU's rows in a file. Two routes:
  1. A phone-built layout side file (the repacked rows written once, sha-keyed). Storage twice per model. This
     is owner questions (a) and (b), still open; the lead's provisional answer is yes.
  2. The served GGUF itself through the stock kernel (no repack). No storage. A declared numerics change, and
     its phone speed is not measured (R72 names it as `--no-repack`).
- **Cost of the arm:** about 60 to 90 s of blocks plus the two holds (2 min) and gates; one 4 GB write
  (*estimated* 10 to 30 s on this storage); peak memory as the step says. Step 4 may kill the process: run it
  last, save the report first, pre-register "killed" as "not held".

## 8. Next questions, ranked

1. **Score-sheet check:** when the lead writes the pre-registration, re-read §1's field column against the
   committed kit (done here against the 19:15 patch and the 19:19 note; a later rebuild would need it again).
2. **The W-position kernel inside llama.cpp** (Tom, host): exact per chunk against stock at `-ub 1` and batch 2?
   And its real A78 cost against the 1.44 count. Every width-2 price hangs on it (§2.1 item 3).
3. **Does the phone really pick the dotprod entry?** One log line (`kleidiai: primary q4 kernel feature`) in the
   next kit's header. If it were the i8mm entry, the gemm is a different kernel and §2.1's count does not apply.
4. **The mixed arm's physics, from data we have:** R72 and R69 read sums (GPU read beside CPU read at different
   CPU thread counts) may already show whether the GPU's rate recovers when the CPU draws less. Host, an hour.
5. **Stock-kernel speed for overlap rows** (§4, §7 route 2): R72's type or stock rows may hold a not-repacked
   Q4_0 rate; else one probe cell.
6. **Depth-2 head accuracy** (`C2`), which decides width 3 on Llama-2 (§6): host, on Tom's existing dumps.
7. **GPU prefill to keep the big cores cool** (Ebony r4 question 6): price it once R75 gives the hand-off cost.

## 9. Limits

- §2.1 is an issue count from assembly, not a timing. The kernel version is read from the build file and the
  assembly from another lab tree of the same version; the kit's binary was not disassembled.
- The Cortex-A78 pipe rules are those I tabled in memo-08 from Arm's guide. Today's fetch of the guide returned
  an unreadable PDF, so they were not re-read.
- §2.3's caveat and §2.4's line rest on two rigid models; the mixed arm exists because neither has a mechanism
  for a partly loaded bus.
- §4 holds the body fixed at its R72 value and is valid for C ≥ 31 only (the script's guard).
- §5's probabilities are judgement; no clock trace of this GPU under minutes of compute load exists anywhere I
  found.
- §6 inherits the unmeasured depth-2 and top-k assumptions of memo-10.
- §7's probabilities are judgement between two measured points (3.5 GB held, 4.2 GB anonymous not held).

## 10. Sources (2026-11-23)

Repository: `lab/llama.cpp-c3r32/ggml/src/ggml-cpu/kleidiai/kleidiai.cpp` (lines 818 to 830, 846 to 858, 1137 to
1491) and `kernels.cpp` (lines 392 to 732, 1129 to 1141); `ggml/src/ggml-cpu/CMakeLists.txt` line 653;
`lab/llama.cpp-c1h6/build-kai/_deps/kleidiai-src/kai/ukernels/matmul/matmul_clamp_f32_qsi8d32p_qsi4c32p/`
(`…16x4_neon_dotprod_asm.S`, `…1x4_neon_dotprod_asm.S`, read); `tools/c3_a78_pipes.py`; `memo-10-tree.md` E2;
`memo-08-work/bonnie-path.md` §2; `patches/llama.cpp-c3/0005-x3-gpu-probe.patch` (lines 983 to 1102, 1229 to
1233) and `0007-x4-join-probe.patch` (as of 19:15); `tom-r35-build.md` (19:19); R74 JSON; `R72-result.md` lines
194 to 195; `docs/negative-results.md`, `reports/04_campaign.md`; [ebony-joins-r2.md](ebony-joins-r2.md) B.2 to
B.4, [ebony-joins-r3.md](ebony-joins-r3.md), [ebony-joins-r4.md](ebony-joins-r4.md).

External (web search today; summaries, pages not opened unless marked):
- Arm Cortex-A78 Software Optimization Guide, <https://documentation-service.arm.com/static/5fa2bd1eb1a7c5445f2906fd>
  (fetched; PDF not readable by the tool).
- KleidiAI micro-kernel guide, <https://learn.arm.com/learning-paths/cross-platform/kleidiai-explainer/page3/>
  (i8mm kernels need an Armv8.6 i8mm target).
- kgsl power control source, <https://android.googlesource.com/kernel/msm/+/05fe60fb845b2d3d5514355b2c2c52cc78a4a182/drivers/gpu/msm/kgsl_pwrctrl.c>
  and <https://gitlab.e.foundation/e/devices/android_kernel_fairphone_fp5/-/commit/27344ce0ec2b524630c4948b7a0470f20ce2b029>.
- Galaxy Z Flip3 stress tests: <https://hothardware.com/reviews/samsung-galaxy-z-flip3-review?page=2>,
  <https://www.sogi.com.tw/articles/samsung_galaxy_z_flip3_5g/6256711>,
  <https://benchmarks.ul.com/hardware/phone/Samsung+Galaxy+Z+Flip3+5G+review>.
- Adreno 660 power: <https://www.notebookcheck.net/Overclocked-Adreno-650-outperforms-the-Snapdragon-888-s-Adreno-660-in-Genshin-Impact-and-other-benchmarks.591774.0.html>.
