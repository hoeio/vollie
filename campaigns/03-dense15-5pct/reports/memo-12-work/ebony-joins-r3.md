# c3 memo-12 work (Ebony, round 3): bands for the added R75 cells, merged table, clocks in a real decode, runtimes read

Date: 2026-11-23. Author: Ebony (`claude-fable-5-2`). Not committed; the lead commits. No verdict here.
Follows [ebony-joins-r2.md](ebony-joins-r2.md) and reviews [bonnie-joins-r3.md](bonnie-joins-r3.md) (read in full,
`python3 -I -B bonnie_joins_r3.py` run; its self-checks pass, and its `pair` variant reproduces my three checks).

Labels: *device* = printed by the phone; *device-derived* = my arithmetic on *device* fields; *estimated* = a model
or my judgement; *external* = another group's number; *source* = read in the local tree. Triples are Yi / Llama-2 /
DeepSeek. Nothing here is a new phone number.

**Limit of part 1:** `patches/llama.cpp-c3/0007-x4-join-probe.patch` held x4m when I read it (source read), but
**not yet** the C11-atomics variant of x4e or the hand-off cell. My bands for those two are written against the
lead's description (flag / counter with C11 atomics on SVM; extra µs per join for writing the input and reading
the output), not against field names. If Tom's variants differ, the rows need renaming, not re-pricing.

## 1. Bands for the added cells, and the merged table (for the R75 pre-registration)

### 1.1 Review of Bonnie's r3 bands

- **x4m:** her method (a two-term reader fitted to the GPU's read-only and gemv rates) is the right first model.
  My centre is a little higher, for one reason: R72's "read-only" kernel at 31.1 GB/s is itself a kernel with ALU
  work, so it is not a pure bus ceiling, and the gemv's compute share may be larger than 28.3 against 31.1 says.
  `_mc3` repeats the activation read, both dequantise macros and the accumulate per column (*source*); only the 8
  weight image reads are shared.
- **Her reading rule "the lane's cg is at most the alone `dev_ratio`": agreed.**
- **Her cg lines (§1.2): agreed as lines.** They are computed in both models, which is what the pre-registration
  needs. I would print the `pair` columns with the 60 µs slack, not without.
- **x4s to 0.60:** agreed within noise; I keep a slightly lower centre and a little more weight on 0.55.
- **`bonnie_tree_r2.py`:** her stage form (CPU at its alone rate only while the GPU is dead) is better than the
  one in `ebony_pair.py` (a flat κ 0.45). Her `pair` row at K 452 (11.12 / 11.08 / 10.71) should replace my
  11.65 / 11.66 / 11.23. I accept that correction to my r2 A.6 table.
- **Her §3 (residency falls with K: 1.0 to 1.5 GB at K 300): agreed,** and it withdraws the worry in my r2 that
  every design sits outside what x3r covered.
- She gives no band for the C11 variant or the hand-off cell; those rows below are mine alone.

### 1.2 Merged table

"=" means I would write her band. **Bold** marks a real disagreement (centres differ by more than a quarter of the
band); for those the pre-registration should print both centres and the union of the bands. All *estimated*.

| cell, field | Bonnie r3 (centre; 80 %) | Ebony | merged |
|---|---|---|---|
| x4m `clCreateKernel(_mc3)` succeeds | P 0.92 | = | 0.92 |
| x4m runs at n_cols 2, `err=0` | P 0.85 | = | 0.85 |
| x4m identity ok on every column, given it runs | P 0.88 | 0.90 | 0.88 to 0.90 |
| x4m all of the above | P 0.75 | 0.76 | 0.75 |
| x4m `dev_ratio` n_cols 2 | 1.2; 1.05–1.75. P(≤ 1.30) 0.58 | **1.3; 1.08–1.80. P(≤ 1.30) 0.50; P(> 1.70) 0.15** | centres 1.2 / 1.3; band 1.05–1.80 |
| x4m `dev_ratio` n_cols 3 | 1.5; 1.15–2.5 | 1.65; 1.2–2.6 | centres 1.5 / 1.65; band 1.15–2.6 |
| x4m `dev_ratio` n_cols 4 | 1.9; 1.35–3.3 | 2.0; 1.35–3.4 | 1.9 to 2.0; 1.35–3.4 |
| x4m `wall_ratio` n_cols 2 | 1.15; 1.04–1.60 | 1.22; 1.05–1.65 | centres 1.15 / 1.22; 1.04–1.65 |
| x4m max_rel_err per column | 6e-4 to 8e-4 | = | = |
| x4s lead, best ratio over 0.20 to 0.60 | 0.73; 0.70–0.79 | 0.72; 0.67–0.79 | 0.72 to 0.73; 0.67–0.79 |
| x4s lead, best share | 0.50 (0.78), 0.45 (0.13), 0.55 (0.09) | 0.50 (0.60), 0.45 (0.20), 0.55 (0.20) | 0.50 (0.60 to 0.78) |
| x4s share 0.55: ratio; `gpu_ms`; `wait_ms`; `gpu_dev_ms` | 0.78; 1.38; 0.21; 1.08 | =; =; =; **1.13 (1.02–1.22)** | `gpu_dev_ms` centres 1.08 / 1.13 |
| x4s share 0.60: the same four | 0.83; 1.48; 0.39; 1.18 | =; =; =; **1.23 (1.10–1.33)** | `gpu_dev_ms` centres 1.18 / 1.23 |
| x4s P(the optimum is interior) | 0.90 | 0.85 | 0.85 to 0.90 |
| x4e C11 variant: SVM fine-grain buffer **with** atomics offered, and the kernel builds | – | P 0.55 (0.70 × 0.80) | mine only |
| x4e C11 variant: flag seen within 1 ms, given it runs | – | P 0.75 | mine only |
| x4e C11 variant: flag seen within 1 ms, unconditional | – | P 0.42 | mine only |
| x4e any mode (1.x atomics on hostptr or SVM, or C11): flag seen within 1 ms | r2: 0.35 (without the C11 variant) | 0.55 | 0.35 to 0.55; **the added variant is why mine is higher** |
| x4e C11 flag `rtt_p50_us`, given seen | – | 20; 3–200 | mine only (her r2 band for the other modes: 30; 5–300) |
| x4e C11 counter `iv_p50_us`, given seen | – | 5; 1–60 | mine only (hers, other modes: 10; 1–100) |
| x4e all three of the lines pass in at least one mode | r2: 0.10 | 0.15 | 0.10 to 0.15 |
| hand-off, enqueue write of the input + blocking read of the output: extra µs per join | – | 90; 40–200 | mine only |
| hand-off, map / unmap of both per join | – | 60; 25–150 | mine only |
| hand-off, buffers kept mapped or io-coherent or SVM (no call per join): extra µs | – | 5; 0–30 | mine only |
| hand-off, the no-call variant returns correct rows (identity) | – | P 0.60 | mine only |
| hand-off, best variant ≤ 20 µs with correct rows | – | P 0.50 | mine only |

My r2 B.3 rows stand for every cell that did not change.

Reading rules I ask for (new):
- **x4m:** `dev_ratio(2)` is read against Bonnie's lines (her §1.2 table): ≤ 1.30 "worth a build"; above 1.70
  "12 through the lane at width 2 has no line with slack". Agreed by both.
- **Hand-off:** the extra µs of the best variant **with correct rows** is added to every in-situ K before a price
  is written. A no-call variant with wrong rows is not a variant.
- **x4e:** "flag not seen" closes the persistent kernel only if the C11 variant ran (built, SVM atomics offered).

### 1.3 What x4m does not decide

x4m is the GPU alone. It gives cg; it does not give the CPU's rate beside a slowed GPU, which is the term Bonnie's
`bonnie` variant and my `pair` variant differ on (0.6 to 1.1 tok/s at K 300, her §2). That still needs the
width-2 `tog` cell with the CPU's W-position kernel (my r2 E-J7, her r3 §4 item 3). A partial answer is free in
r35 if Tom logs it: **x4s at share 0.55 and 0.60 is GPU-bound, so the CPU's rows finish early; `cpu_ms` there
against share 0.20 shows how the CPU's rate moves with the GPU's duty.** That is the same physics at width 1.

## 2. Do the clocks hold in a real decode? (my question 6)

### 2.1 What a "decode" is here

The metric is llamadart's `decodeTokensPerSec` on the official prompt, **32 tokens**, greedy, with the r16
cool-down gate before each run (CAMPAIGN.md terms table). At 8 to 12 tok/s that is **2.7 to 4 s of decode** after
the prompt's prefill (§2.4: about 12 to 15 s on a dense 7B, *estimated*). The caps react over tens of seconds. So the question is not "does the phone sustain it" but
"what state does a gated start give, and does it last 4 s".

### 2.2 Device records

| record | what it shows | label |
|---|---|---|
| R74 `x3c` (this phone, CPU on 4-7 **and** GPU loaded, gated start at headroom 0.647) | caps 2,419 / 2,841 (full) through all six `tog` arms, about 15 s of load, headroom 0.70 to 0.74; then 2,150 at headroom 0.84 (about 17 s in), then 1,785 at 0.92 (about 25 to 35 s in) | *device* |
| R74 `x3g` (GPU alone, 32 s) | caps full throughout; headroom 0.673 → 0.783 | *device* |
| R74 `x3r` gate | started at headroom 1.17 after `x3c`; fell to 0.72 in 35 s, then needed the full 180 s to reach 0.66 (timed out) | *device* |
| R72 L4 (4 threads, long sweep cells, no gate between arms) | caps 1,555 to 1,785 in 54 of 63 cells; Llama-2 and DeepSeek at 1,555 throughout | *device* |
| campaign 2, memo-15 §1.3 (cap against headroom at the gate, 225 runs) | headroom < 0.70: median 2,112 MHz, 88 to 94 % of runs ≥ 2,000; 0.70 to 0.75: 73 %; 0.75 to 0.80: 42 %; above 0.80: 1,555 | *device* |
| campaign 2, R68 cap probes | thermal status ≥ 1 caps policies 4 and 7 at 1,420 to 1,785 whatever else; status 0 with power saving off allows 2,419 / 2,841. Status 1 appeared at headroom 0.77 to 1.0 | *device* |
| campaign 2 claim-format blocks (R68 A and B, R70, R71; 3 threads, about 1.3 s of decode per run) | 8 to 10 of 10 runs uncapped per block | *device* |
| campaign 2, R60 and R62 | every gate wait timed out; no cell near the goal | *device* |

No record exists of a dense 7B decode on this phone (R73 is held), and none of the GPU's own thermal level under a
long load (`thermal_pwrlevel=0` was read once, before R74's blocks).

### 2.3 Reading

- **Within one run the clocks hold.** With CPU and GPU both loaded from a gated start, R74 kept full caps for about
  15 s; a 32-token decode needs 3 to 4 s. P(a run that starts at headroom ≤ 0.70 and thermal status 0 ends
  uncapped) about 0.9 (*estimated* from the first two rows).
- **The risk is the start state, not the run.** One lane run puts in more heat than a campaign-2 run (4 threads
  and the GPU for 3 to 4 s plus the prefill, against 3 threads for about 1.3 s). After 35 s of that load the gate
  took the full 180 s and timed out at 0.66. For a claim of 10 runs: P(at least 9 of 10 start at ≤ 0.70) about 0.6,
  my judgement, with campaign 2's spread (10 of 10 on good days, every gate timed out on bad days).
- **The measured bodies are already at the bad state.** The K0 bodies (120.6 / 140.6 / 130.8 ms) come from R72's
  L4 arms at 1,555 to 1,785 MHz. So C 35 is the clock state *of the numbers every stack is built on*, and a
  gated claim run is likely better than that, not worse. Memo-09's "clocks" row (0 to −16 / −19 / −19 ms) is that
  difference, and it is not in any stack.
- **Compute rate by state** (three A78 hot 34.6 GB/s at 2,419 MHz, *device* R72 L3; the X1 at 20.5 is a prediction):
  gated start, X1 on-core: C 50 to 55. Caps 2,150: about 45. Caps 1,785: about 38. Caps 1,555 (thermal status 1):
  33 to 35. X1 thread on a small core at 1,555: 24. Status 3 (1,420): about 30.

### 2.4 The pair model by clock state (*estimated*; `ebony_pair.py`, GC 28.3, 60 µs of jitter per split stage where K > 0; trained head, C7, C5, O 5.2 at centre)

| state | C | pair kB(2) | K 0 | K 150 | K 290 | K 452 |
|---|---|---|---|---|---|---|
| gated claim run, X1 on-core | 45 to 50 | 1.05 to 1.06 | 15.0 / 14.7 / 13.9 | 12.7 / 12.6–12.8 / 12.1 | 11.9 / 11.9–12.1 / 11.45 | 11.4 / 11.3–11.5 / 10.9 |
| memo-10's centre | 40 | 1.06 | 14.89 / 14.57 / 13.84 | 12.62 / 12.55 / 12.04 | 11.77 / 11.78 / 11.34 | 11.20 / 11.15 / 10.77 |
| thermal status 1 (the measured bodies' state) | 35 | 1.11 | 14.38 / 14.05 / 13.38 | 12.25 / 12.16 / 11.69 | 11.42 / 11.41 / 11.00 | 10.74 / 10.78 / 10.43 |
| status 2 to 3 | 30 | 1.18 | 13.66 / 13.32 / 12.74 | 11.73 / 11.61 / 11.19 | 10.97 / 10.92 / 10.56 | 10.25 / 10.27 / 9.97 |
| X1 thread off-core at 1,555 | 24 | 1.28 | 12.73 / 12.38 / 11.89 | 11.04 / 10.89 / 10.54 | 10.36 / 10.28 / 9.97 | 9.75 / 9.73 / 9.47 |

- **Above C 40 the pair model is flat** (both readers are bus-bound at width 2), so a cool start buys almost
  nothing over the centre: +0.1 to +0.3 tok/s. The loss is all below 40: −0.4 at C 35, −0.8 at C 30, −1.4 at C 24.
- These rows hold the body fixed at its R72 value. A cool start also shortens the body itself by up to memo-09's
  clocks row; not added here because nothing measured separates it from the X1's placement.
- **The state a claim run is most likely to see is C 40 to 50 with P about 0.6, C 33 to 38 with P about 0.3, and
  C 24 to 30 with P about 0.1** (the last needs the X1 thread off-core, which the pin guard patch 0004 exists to
  prevent, or thermal status ≥ 2 at the start). Weighted centre at K 290: about 11.6 / 11.6 / 11.2.
- **The prefill comes first, and it is long** (added after the table). The official prompt is 702 characters
  (`memo-10-work/clyde-r73-plan.md`), about 170 to 200 tokens with a chat template (*estimated*; not counted with
  the subjects' tokenisers). A dense 7B prefill on four threads is compute-bound; at C 50 and 3.1 to 3.6 GB of
  weights per token that is about 14 to 16 tok/s, so **12 to 15 s of full four-thread load right before the
  decode** (*estimated*; no dense prefill was ever timed on this phone; campaign 1's small model read 15 to 22
  tok/s, *device*, `docs/round5-comparison.md`). R74's timeline puts the first cap step (2,150) at about 17 s of
  load. So the decode most likely starts at 2,419 or 2,150 and ends at 2,150 or 1,785: C about 38 to 45, not 50.
  Revised odds: C ≥ 40 with P 0.5, C 33 to 40 with P 0.4, C 24 to 30 with P 0.1. The weighted centre at K 290
  moves by −0.1: about **11.5 / 11.5 / 11.1**. A prefill on the GPU (upstream's batched kernels) would move the
  heat off the big cores; not priced.
- GPU-side heat: perf hint HIGH forces the GPU's top frequency but "thermal controls ... can override" (Qualcomm
  guide §4.5.5, *external*, read). If the GPU's level drops, GC falls; at GC 22.6 (0.8 ×) the C 40 row loses about
  0.45 tok/s at K 290 (11.33 / 11.30 / 10.91). No record says whether it drops in 4 s; R74's `x3g` kept its rate
  for 32 s with `gpu_busy` steady.
- Bonnie's point (r3 §4 item 6) that the lane halves the CPU's work and so should cool it: at width 1, yes (CPU at
  18.9 of 25.6 GB/s). At width 2 the CPU does 21.4 GB/s × 1.45 of arithmetic, about 1.2 × stock. So the lane with
  width 2 heats the CPU **more** than stock, and the GPU on top. The feedback goes the wrong way; it only fails to
  matter because the run is 4 s long.

### 2.5 What would settle it

Not a probe cell: R73 as planned (first real decode, with caps, placement and thermal status logged per run) is
the measurement. A rider that costs nothing in any kit: print Android `thermal_status` and the kgsl
`thermal_pwrlevel` before and after every block (the cap probe already reads the first).

## 3. Runtimes and papers not read before

| source | read status | what it says for us |
|---|---|---|
| **Lever** (arXiv 2605.16786v1, May 2026) | HTML read through the fetch tool; figure values missing | Flash-resident targets (Llama-3.1-8B, Qwen3 4B to 14B, Q4_0) on 8 Gen 3 and 8 Elite phones; **1.5 to 2.9 tok/s**: an I/O-bound regime (91 % of a verify pass is I/O), not ours. Draft is EAGLE3 (a trained draft head, the memo-10 E5 question). Verify runs all layers as one batch on the NPU, the head on the CPU for the accepted path only. Useful number: verify latency 828 ms at a tree of 16 nodes against 1,255 ms at 32 (*external*; about 1.5 × for twice the width, my arithmetic). Early-exit pruning keeps 48 % of upper-layer rows for −12 % latency and 95 % of accepted tokens |
| **AHASD** (arXiv 2604.25326v3, DAC 2026) | HTML read through the fetch tool | **A simulation** (cycle-accurate NPU + LPDDR5-PIM simulators); the "GPU-only" baseline is a laptop RTX 4090. No phone measurement, no dispatch figure. Not usable as evidence. One qualitative point: asynchronous draft / verify lowered acceptance by 25 % before their fix |
| **HeteroMosaic** (arXiv 2607.12839v4, MICRO 2026) | first 100,000 characters read through the fetch tool; results sections not reached | AMD Ryzen AI (iGPU + NPU), CPU excluded as a compute reader. Its roofline for accelerators sharing memory is the same shape as `ebony_pair.py`: each reader min(bus share × intensity, compute), with a shared-bus cap; and it names two power regimes ("turbo" and sustained) that shift the best split (0.5 against 0.6 in their example). No per-hand-off figure, no GB/s measured in what I read |
| **HELIX** (arXiv 2611.06513, Nov 2026) | abstract only (no HTML tried) | Server GPUs (H100, B200) with emerging accelerators; operator-level partition. Not our hardware |
| **PowerInfer-2** (arXiv 2406.06282) | search summary only; no 2026 update found | For 7B models it claims memory savings at llama.cpp's speed, not more speed; its gains are on sparse (ReLU-fied) models and models larger than memory. Nothing for a dense 7B that fits |
| **MLC-LLM** on Adreno | search summary of Qualcomm's IWOCL slides and blog | Llama-2 decode 5.3 → 11.2 tok/s (8 Gen 2) and 6.4 → 14.2 (8 Gen 3) after Qualcomm's optimisations (*external*); Qualcomm's listed Android targets start at 8 Gen 1. **No Adreno 660 or Snapdragon 888 number found anywhere** |
| **ExecuTorch**, Qualcomm backend | search summary of the docs | The SoC must be in the backend's chipset enum; the lists seen name 8 Gen 1 and later. SM8350 not found. Agrees with memo-09 (QNN LLM support starts at SM8550) |
| **mllm** | search summary of the README and a 2026 note | QNN ahead-of-time full-graph execution on the NPU (Feb 2026), linked models for SM8650. No CPU + GPU split, nothing for the 888 |

Reading:
- **Nobody found splits one matrix's rows between the CPU and a GPU, and nobody reports a phone-side verify batch
  that shares the weight read between two processors.** Our lane with width 2 has no published twin; the nearest
  shapes are HeteroInfer (GPU + NPU, CPU as control plane) and HeteroMosaic's roofline.
- Every phone number found is for a 2023 or later SoC with a runtime that the 888 is not on the support list of.
- Lever's 1.5 × for twice the tree width is in line with memo-10's kB(W) on a compute-bound reader; it says
  nothing about a bus-bound pair.

## 4. Next questions, ranked, and what I did

| # | question | state |
|---|---|---|
| 1 | Bands for the added cells; merged table | **done** (§1) |
| 2 | Clocks in a real decode | **done as far as records go** (§2): within a 32-token run they hold; the start state decides; centre C 40 to 50 with P 0.6. Open: R73 |
| 3 | The unread runtimes | **done** (§3); nothing changes a price |
| 4 | Does the CPU's rate move with the GPU's duty (the `bonnie` against `pair` term)? | A reading of x4s's shares 0.55 and 0.60 against 0.20 is proposed (§1.3); needs no new cell |
| 5 | How many joins does one accepted token really cost at width 2? τ divides the body, but a rejected second position still paid its share of every join | priced implicitly (K is per pass, τ per pass); **checked: consistent** in both scripts |
| 6 | The gate after a lane run: how long, and does it time out? | no record; R74's 180 s time-out after 35 s of load is the only hint. A kit rider: the gate series is already printed; read it after x4s and x4r |
| 7 | The prefill: does it heat the start of the decode? | **done by arithmetic** (§2.4, last bullet but one): 12 to 15 s of four-thread load before the decode, so C 38 to 45 at the start. Open: the prompt's token count per subject (host, minutes) and a timed dense prefill (R73) |
| 8 | Head top-1 on the device at width 2 (0.379 / 0.497 are *host*, DeepSeek's is assumed) | not mine to measure; the stack's τ is the second-largest unknown after cg |
| 9 | A file-backed residency arm (Bonnie r3 §3.4) | hers; agreed it is the arm that would settle Llama-2 and DeepSeek |

## 5. Sources (2026-11-23)

Repository: `campaigns/03-dense15-5pct/CAMPAIGN.md` (metric row); R74 JSON (gate series, `max_mhz`, `hr` per cell);
`R72-result.md` §2 and §4; `campaigns/02-25toks-5pct/reports/memo-15-cost-model-refit.md` §1.3;
`campaigns/02-25toks-5pct/reports/README.md` (R68 cap probes, R68 to R71 claim blocks, R60, R62);
`campaigns/02-25toks-5pct/knowledge.md` §1; `patches/llama.cpp-c3/0007-x4-join-probe.patch` (x4m, x4s, x4e as of
19:00); [bonnie-joins-r3.md](bonnie-joins-r3.md), `bonnie_tree_r2.py`, `bonnie_joins_r3.py`.

External: <https://arxiv.org/html/2605.16786v1> (read); <https://arxiv.org/html/2604.25326v3> (read);
<https://arxiv.org/html/2607.12839v4> (first 100,000 characters); <https://arxiv.org/abs/2611.06513> (abstract);
PowerInfer-2 <https://arxiv.org/abs/2406.06282> (search summary); Qualcomm IWOCL slides on MLC
<https://www.iwocl.org/wp-content/uploads/1494-siva-reddy-qualcomm.pdf> and blog
<https://www.qualcomm.com/developer/blog/2025/02/harnessing-qualcomm-adreno-gpu-generative-ai-open-source-approach>
(search summaries); ExecuTorch Qualcomm backend docs <https://docs.pytorch.org/executorch/stable/backends-qualcomm.html>
(search summary); mllm <https://raw.githubusercontent.com/UbiquitousLearning/mllm/main/README.md> (search summary).
