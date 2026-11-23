# c3 memo-12 work, round 8 (Bonnie): the KleidiAI buffer's count and size; the file-backed residency arm as a kit specification

- **Date:** 2026-11-23. **Author:** bonnie, `claude-fable-5-2`. Follows [bonnie-joins-r7.md](bonnie-joins-r7.md). Not
  committed. No verdict here. **Ebony has not reviewed this file.**
- **Labels.** *device* = printed by the phone. *derived* = arithmetic on *device* fields. *host* = laptop.
  *estimated* = arithmetic, model or judgement. *source* = read in `lab/llama.cpp-c3r32` (its ggml sources are the
  kit's) or in the KleidiAI v1.31.0 sources in `lab/llama.cpp-c1h6/build-kai/_deps/kleidiai-src`. Triples are Yi /
  Llama-2 / DeepSeek.
- Nothing ran this round: no model, no load, no phone. §1 is source reading and arithmetic; §2 is a design.
  No web source: nothing here states the state of a model or of upstream beyond the local tree.

## 0. Answer first

1. **One KleidiAI buffer per model** (*source*): the loader keeps one context per buffer type, and the type sets
   no size limit, so every Q4_0 matmul weight and the Q6_K head land in a single allocation.
2. **The head grows by 22 % in it** (*source*, *estimated*): KleidiAI dequantises each Q6_K row and re-quantises it
   to int8 with one scale per row, 4,108 bytes per row against 3,360. Head: 263 / 131 / 421 MB against 215 / 108 /
   344 MB in the served file.
3. **Buffer size, which is the side file's size for a CPU-only body:** **3,221 / 3,599 / 3,658 MiB**
   (3.38 / 3.77 / 3.84 GB; *estimated*, exact tensor arithmetic). A load log line would confirm it to the MiB.
4. **A finding outside the brief: the phone's stock path is not the path the freeze measured.** The S3 freeze ran
   ggml's repack path; the phone runs KleidiAI, which re-quantises the head to int8 per row. I expect the
   difference in perplexity to be small (under 0.05 %, *estimated*), but the budget's reference is "the served
   file's stock perplexity" and nobody has measured it on the path the phone serves. It is question 1 of §3.
5. **R69's own data narrow the side file's design** (*device*, two runs): a 4.2 GB read-only map held in one run
   of two; the same size mapped executable held in both. So the side file should be mapped executable, as
   campaign 2's patch 0011 does. R69 also gives the phone's write rate: 4.6 GB in 10.3 s and 7.6 s, so the first
   run writes a side file in **4 to 9 s** (replaces r7's wider guess for the write part).
6. **The residency arm is specified in §2** as kit blocks `x5w`, `x5s1` to `x5s5`, with printed fields, order,
   kill rule and bands. About 9 minutes of blocks. It needs no loader change and no model.

## 1. Buffer count and head growth (next question 1)

### 1.1 Count (*source*)

- `llama-model-loader.h` lines 148 to 163: tensors are grouped into contexts keyed by (buffer type, lazy flag).
  Lazy reading is off unless asked for. So all tensors that the KleidiAI type takes share one context.
- `llama-model.cpp` line 1888 on: one buffer set per context. A non-default type gets
  `ggml_backend_alloc_ctx_tensors_from_buft`, which splits only when a type reports a maximum size.
  `kleidiai.cpp` leaves `get_max_size` null (unlimited). **One buffer.**
- Which tensors it takes (`supports_op`, lines 1788 to 1860): 2-D weights of `MUL_MAT` with float activations, of
  type Q4_0, Q4_K, Q8_0, Q6_K or F32, when a kernel exists. The token embedding is read by `GET_ROWS` and stays
  mapped from the served file. Norm vectors are not matmuls.
- The KleidiAI type is listed before ggml's repack type (`ggml-cpu.cpp` lines 59 to 66), so it wins every tensor it
  supports.
- `VOLLIE_HEAD2_REPACK=1` (campaign 2, in this tree) sends the Q6_K head to the repack type instead. It is unset
  in campaign 3's kits as far as I can see; with it set there would be two buffers and a smaller KleidiAI one.

### 1.2 What happens to the head (*source*)

`repack`, lines 1529 to 1596: for a Q6_K tensor every row is dequantised to float, its largest magnitude is
found, the scale is that ÷ 127, and each value is rounded to int8. The packed size is
`nr × (k + 4 + 4 + 4)` per group of `nr` rows (`kai_rhs_pack_nxk_qsi8cxp_qsi8cx_neon.c` line 37): one byte per
weight plus a multiplier, a row sum and a bias per row.

| | rows × 4,096 | served Q6_K (3,360 B per row) | in the KleidiAI buffer (4,108 B per row) |
|---|---|---|---|
| Yi | 64,000 | 215.0 MB | 262.9 MB |
| Llama-2 | 32,000 | 107.5 MB | 131.5 MB |
| DeepSeek | 102,400 | 344.1 MB | 420.7 MB |

This is upstream's behaviour, not ours. It is a re-quantisation: the head the phone multiplies by is not the
served head bit for bit (§0 item 4).

### 1.3 Size (*estimated*; exact arithmetic on the shapes)

Q4_0 packs to 72 bytes per 128 weights. Per layer: Yi 173,015,040 weights (k and v are 512 × 4,096); Llama-2 and
DeepSeek 202,375,168.

| | layers | Q4_0 body, bytes | head, bytes | **buffer** | check against the served file |
|---|---|---|---|---|---|
| Yi | 32 | 3,114,270,720 | 262,912,000 | 3,377,182,720 = **3,220.7 MiB** | body + Q6_K head + embedding = 3.11 + 0.22 + 0.15 = 3.48 GB; file 3.48 GB |
| Llama-2 | 32 | 3,642,753,024 | 131,456,000 | 3,774,209,024 = **3,599.4 MiB** | 3.64 + 0.11 + 0.07 = 3.82; file 3.83 |
| DeepSeek | 30 | 3,415,080,960 | 420,659,200 | 3,835,740,160 = **3,658.0 MiB** | 3.42 + 0.34 + 0.24 = 4.00; file 4.00 |

- Plus a 64-byte-aligned header per tensor (under 0.1 MiB in all).
- The last column checks the arithmetic: the same shapes reproduce the served files' sizes to 0.01 GB (*host*).
- **Correction to r7 §0:** the CPU-only side file is 3.38 / 3.77 / 3.84 GB (r7 wrote 3.4 / 3.8 / 3.8). Beside a
  lane at GPU share 0.37: 2.22 / 2.43 / 2.57 GB (body rows × 0.63, plus the whole head).
- **What stays anonymous beside it** (*estimated*): the KV at n_ctx 512 in F16, 0.03 / 0.27 / 0.25 GB (Yi has 512
  KV dimensions per layer, the others 4,096), plus compute scratch and the app. R72's "resident 3.36 / 4.02 /
  4.01 GB" were such totals.
- **Reading for residency.** Today's anonymous need is the buffer plus the KV: about 3.41 / 4.04 / 4.09 GB,
  against 3.4 held and 4.2 not held (*device*, R72). With the side file the anonymous part falls to 0.03 / 0.27 /
  0.25 GB and the rest is file-backed.

### 1.4 What a load log would confirm

One load of each served file on the laptop's KleidiAI build with the dotprod entry forced
(`lab/c3/head/build-kai`, `GGML_KLEIDIAI_SME=0`), reading three lines:

| line (as the loader prints it) | expected |
|---|---|
| `load_tensors: CPU_KLEIDIAI model buffer size = … MiB` | one such line; 3,220.7 / 3,599.4 / 3,658.0 MiB (± 0.1) |
| `load_tensors: CPU_Mapped model buffer size = … MiB` | the embedding and small tensors: about 141 / 70 / 220 MiB |
| `kleidiai: primary q4 kernel feature …` | `DOTPROD` (the phone's entry; the M4 would otherwise pick SME or i8mm, with other pack sizes) |

- A second `CPU_KLEIDIAI` line, or a `CPU_REPACK` line, refutes §1.1.
- A KleidiAI size that differs by the head's growth (48 / 24 / 77 MiB) says the head went elsewhere.
- Minutes, no perplexity run. Tom or the lead; I do not run loads.

## 2. The file-backed residency arm as a kit specification (next question 3)

**Purpose.** Show whether a file-backed body stays resident on the phone beside a lane's GPU buffers and the
anonymous KV, at the sizes of each subject, through a pass loop and through a 60 s wait. No model, no loader
change, no numerics. For Tom after R75; the round needs its own pre-registration and the owner's permission.

### 2.1 What R69 already says (*device*, kit r29, block `p1`, two runs; medians of passes 1 to 19)

| map size | read-only map: GB/s, majflt per pass (run 1 / run 2) | executable map (run 1 / run 2) |
|---|---|---|
| 2.7 GB | 28.9, 0 / 27.5, 0 | 28.8, 0 / 26.2, 0 |
| 3.0 GB | 28.7, 0 / 28.0, 0 | 29.6, 0 / 29.2, 0 |
| 3.4 GB | 25.1, 0 / 29.3, 0 | 22.5, 0 / 29.9, 0 |
| **4.2 GB** | **2.6, 325** / 28.1, 6 | 27.2, 4 / 28.6, 7 |
| 4.6 GB | 1.8, 332 / 3.0, 134 | 1.4, 486 / 1.9, 568 |

- Write of the 4.6 GB scratch file with `fsync`: 10.3 s and 7.6 s, so 0.45 to 0.61 GB/s.
- The first pass over a map is slow even when it holds (1.3 to 20 GB/s).
- These arms had no GPU buffers and no large anonymous memory beside them. That is the gap.

### 2.2 Blocks and order

Button: "Run c3 probe r36 (file-backed residency). Cool start." Order: `x5w`, `x5s1`, `x5s2`, `x5s3`, `x5s4`,
`x5s5`. The r16 gate runs before every block; the report is saved before and after every block; each block starts
with the `hdr` lines of r32 (zram, PSI, meminfo) and x3's `x3 cond` lines. Smallest risk first, so a kill loses
the least.

| block | file F (map) | GPU G | anonymous A | total | stands for | phases |
|---|---|---|---|---|---|---|
| `x5w` | write the 3.9 GB scratch file once into the app's files directory, `fsync` | – | – | – | first-run write rate | – |
| `x5s1` | 2.2 GB (x) | 1.15 GB | 0.05 GB | 3.4 | Yi with a lane; anchor (x3r held 3.5 GB anonymous + GPU, R74) | 0, a |
| `x5s2` | 2.4 GB (x) | 1.35 GB | 0.30 GB | 4.05 | Llama-2 with a lane | 0, a, hold on, b, hold off, c |
| `x5s3` | 2.6 GB (x) | 1.25 GB | 0.25 GB | 4.1 | DeepSeek with a lane | 0, a, hold on, b, hold off, c |
| `x5s4` | 3.85 GB (x, then r) | 0 | 0.30 GB | 4.15 | Llama-2 or DeepSeek on the CPU alone, whole body file-backed | 0, a, hold on, b; then the same with map r: a only |
| `x5s5` | 2.6 GB (x) | 1.50 GB | 0.30 GB | 4.4 | the lane with an overlap copy and margin | 0, a |

- Every step maps a prefix of the one scratch file. Sizes are the tensor arithmetic of §1.3 at GPU share 0.37.
- GPU buffers: copies of 11008 × 4096 on the general kernel, as x4r builds them.
- Anonymous memory: filled once, then read once per pass by the threads (it stands for the KV, which a pass
  reads).

**Phases inside a step:**

| phase | what | length |
|---|---|---|
| map, touch | map the prefix (x: `PROT_READ \| PROT_EXEC`, shared; r: `PROT_READ`), read it once on one thread | as long as it takes |
| 0 | CPU alone: four threads on 4-7 run R72's read kernel over the file map, a quarter each, and the anonymous memory | 5 passes |
| a | the same while a feeder on 0-3 runs the GPU ring (8 launches, then `clFinish`), as x3r | 15 s, at most 200 passes |
| hold on | everything stops for 60 s with campaign 2's page hold on the file map (§2.7): every 5 s, and at the end, the pages lost since the start are asked back with `WILLNEED`; at the end it waits until 98 % are back or 5 s have passed | 60 s + at most 5 s |
| b | as phase a | 10 passes |
| hold off | everything stops for 60 s, no hold | 60 s |
| c | as phase a | 10 passes |

Four threads on 4-7 because that is the lane's set; x3r used three on 4-6, so rates are not compared with x3r's
directly.

### 2.3 Printed lines (tag `x5`; fields named as x3r's where they are x3r's)

| line | fields |
|---|---|
| `x5 write` | `gb= s= gbs= fsync_s= free_gb_before= err= errno=` |
| `x5 res stage=start` | `step=<s1..s5> file_gb= gpu_gb= anon_gb= map=<x\|r> memavail_kb= cached_kb= swapfree_kb= vmrss_kb= rssanon_kb= rssfile_kb= vmswap_kb= kgsl_page_alloc=` |
| `x5 res stage=gpu_alloc` | `step= kern=general copies= gpu_mb= alloc_s= err=` |
| `x5 res stage=anon_fill` | `step= anon_gb= fill_s= majflt=` |
| `x5 res stage=file_map` | `step= file_gb= map= err= errno=` (errno 13 on an executable map = the directory refuses it; the step then runs with map r and says `map=r fallback=1`) |
| `x5 res stage=file_touch` | `step= s= gbs= majflt= resident_pct=` |
| `x5 res pass` | `step= phase=<0\|a\|b\|c> pass= s= gbs= gpu_gbs= majflt= minflt= memavail= cached= swapfree= vmswap= vmrss= rssfile= mhz= max_mhz= gpu_mhz= gpu_max_mhz= resident_pct=` |
| `x5 res hold` | `step= mode=<on\|off> s= polls= lost_pages_max= readback_mb= back_pct= back_wait_ms= resident_pct_before= resident_pct_after= majflt= memavail_before= memavail_after= hr_before= hr_after= thermal_pwrlevel=` (`polls`, `readback_mb`, `back_pct`, `back_wait_ms` are -1 with the hold off) |
| `x5 res sum` | `step= phase= map= passes= secs= cpu_gbs_median= cpu_gbs_min= cpu_gbs_first= gpu_gbs= majflt_median= majflt_sum= resident_pct_min=` |
| `x5 res abort` | `step= phase= pass= reason=<memavail\|alloc\|map\|cl> memavail_kb= err=` |
| `x5 res done` | `step= err=` |
| `x5s<n> end` | `s=` |

Definitions:
- `gbs` is file-map bytes read per second by the four threads over the pass (the anonymous read is timed
  inside the pass but not counted in the bytes). `gpu_gbs` is weight bytes per second on the GPU ring, -1 with
  no GPU.
- `resident_pct` is `mincore` over the file map, as a percentage of its pages; computed on every tenth pass and
  at the points named, -1 otherwise.
- `cpu_gbs_first` is the first pass of the phase. `rssfile` is `RssFile` from `/proc/self/status`.
- `gpu_max_mhz` is kgsl `max_gpuclk` (the rider the lead accepted for r35).
- The kit aborts a step itself when MemAvailable falls under 150 MB at the end of a pass (`reason=memavail`),
  frees everything and goes on to the next block. That keeps the process alive for the later steps.

### 2.4 Reading rules (for the pre-registration; the verdicts are the lead's)

Reference rate R0 = `cpu_gbs_median` of phase 0 in `x5s1` (the smallest map, CPU alone).

| reading | rule |
|---|---|
| a step **held** | phase a: `cpu_gbs_median` ≥ 0.80 × (R0 − 0.36 × `gpu_gbs`), `majflt_median` ≤ 50, and `x5 res done step=… err=0`. This is R74 §5's rule with the step's own rates. For `x5s4` (no GPU): ≥ 0.80 × R0 |
| **survives the wait with the hold** | held, and phase b's `cpu_gbs_first` ≥ 0.80 × phase a's median |
| **survives the wait without help** | the same for phase c |
| **not held** | any criterion of "held" missed, or `x5 res abort` in phase 0 or a |
| **killed** | no `x5s<n> end` line for a started block. That step is "not held (killed)"; later steps are "not measured". Not a void: the report up to the kill is scored |
| anchor failed | `x5s1` not held: the arm is not comparable with x3r and nothing is read from the later steps beyond description |
| executable map refused | `fallback=1`: the step is scored as a read-only map and labelled so |

What each outcome means:
- `x5s2` and `x5s3` held and surviving the wait with the hold: a packed side file (r7 §1) gives Llama-2 and
  DeepSeek a resident body beside a lane. Build it.
- Held but not surviving the wait even with the hold: a claim block of gated runs reloads pages at every run;
  price the first pass (`cpu_gbs_first`) into the claim, or hold harder.
- `x5s4` held and the lane steps not: the GPU buffers are what breaks residency. Then a file-backed body serves a
  CPU-only decode for these two subjects and the lane is a Yi-only lever.
- Nothing above `x5s1` held: Llama-2 and DeepSeek have no resident configuration with a repacked body on this
  phone. That would be said with these numbers.

### 2.5 Bands (*estimated*; centre; 80 % band)

| row | field | prediction |
|---|---|---|
| write rate | `x5 write gbs=` | 0.52; 0.40–0.65 (R69: 0.45 and 0.61, *device*) |
| R0, four threads alone | `x5s1` phase 0 `cpu_gbs_median` | 30; 26–33 (R69 map at 2.7 GB 26 to 29 with its thread set; R72 read 29.9) |
| `x5s1` held | – | P 0.85 |
| `x5s1` phase a | `cpu_gbs_median`; `gpu_gbs`; `majflt_median` | 21; 18–24. 21; 19–23. 0; 0–5 |
| `x5s2` held / survives with hold / without | – | P 0.55 / 0.45 / 0.25 |
| `x5s3` held / survives with hold / without | – | P 0.50 / 0.40 / 0.20 |
| `x5s4` held, map x / map r | – | P 0.65 / 0.35 (R69 at 4.2 GB: x 2 of 2, r 1 of 2, with no anonymous memory beside) |
| `x5s5` held / killed or aborted | – | P 0.20 / 0.30 |
| a held step, phase a | `majflt_median`; `resident_pct_min` | 0; 0–10. 99; 95–100 |
| after 60 s with the hold | `resident_pct_after` | 98; 90–100 |
| after 60 s without | `resident_pct_after` | 85; 55–100 |
| first pass after a wait that lost 10 % of the pages | `cpu_gbs_first` ÷ median | 0.5; 0.2–0.9 |
| GPU cap moves in any step | `gpu_max_mhz` < 840 | P 0.10 |
| executable map refused in the files directory | `fallback=1` | P 0.05 (campaign 2 R67: internal storage allows it, *device*) |

Per model, the step that decides:

| subject | step | P(a resident repacked body beside a lane, through the wait) |
|---|---|---|
| Yi | `x5s1` (and it already fits anonymously) | 0.85 |
| Llama-2 | `x5s2` | 0.45 |
| DeepSeek | `x5s3` | 0.40 |

### 2.6 Cost

- **Phone time** (*estimated*): `x5w` 8 to 10 s; `x5s1` 30 s; `x5s2` and `x5s3` about 170 s each (two waits of
  60 s); `x5s4` about 130 s; `x5s5` 30 s. About 9 minutes of blocks, plus six gates of up to 180 s.
- **Storage:** 3.9 GB of scratch, deleted at the end (R69 had 42.6 GB free, *device*).
- **Peak memory:** the step's total plus the app; `x5s5` is last because it is the likeliest to be killed.
- **Code:** x4r's GPU ring and pass loop, R69 `p1`'s scratch file and map, campaign 2's hold. New: the phases,
  `mincore`, the abort rule. No change to the CPU module, so the kit can ship r33's library again.
- Not in scope: the side file itself, any model, a hold stronger than campaign 2's.

### 2.7 Campaign 2's hold, read for this arm (answers next question 3)

*Source:* `patches/app-c2/0006-kit-r22.patch` (kit r22, memo-20 §2); *device:* R60.

- **What it is.** App-side, no environment variable. At the end of a run it records which pages of the model
  file are resident. During the gate's wait, at every poll and once at the end, it compares with `mincore` and
  asks the kernel to read back each run of lost pages (`WILLNEED`). After the gate it waits until 98 % of the
  wanted pages are back, at most 5 s.
- **What it did** (*device*, R60, medians of 4 runs each): Granite 16.61 with the hold against 11.58 without (arm
  off), OLMoE 17.48 against 13.86.
- **It works on a mapped file only.** An anonymous repacked body has no file to read back from; a page the
  wait took goes to zram and comes back by a fault during the pass. **So today's dense path (anonymous KleidiAI
  buffer) gets nothing from the hold, and a side file is what would make the hold apply.** This is a second
  reason for the side file beside the capacity one.
- **Cost at dense sizes** (*estimated*): `mincore` over 2.2 to 3.85 GB is 0.5 to 1 million pages per poll,
  milliseconds. The read-back is the lost bytes at the storage rate: 10 % of 2.6 GB is 0.26 GB, well under a
  second at the read rates R69's first passes show (1.3 to 20 GB/s through the page cache; a cold read is not
  measured). The 5 s cap of the kit was set for 1 GB files; `back_wait_ms` shows whether it binds here.
- The arm implements the same rule natively in the probe (the app's hold needs a loaded model).

## 3. Next questions, ranked

1. **Stock perplexity on the phone's path.** One host run per subject on the KleidiAI build with the dotprod
   entry (40 chunks first; full set if the 40 differ from the repack path by more than 0.02 %), at `-b 512` and
   `-ub 1`. It answers what the head's int8 re-quantisation costs and which number the +5 % budget is taken
   from. Tom; under an hour for the screens. The lead decides which path defines "stock".
2. **The load log of §1.4** (minutes). Fixes the side file's size and confirms one buffer.
3. ~~Does campaign 2's hold touch file pages or lock them, and at what cost?~~ **Done, §2.7:** it reads lost
   pages back with `WILLNEED`, works on mapped files only, and is cheap at these sizes (*estimated*).
4. **Pack rate of the scalar routine** (r7 question 2): with R69's write rate known, the pack is now the larger
   unknown of the first run.
5. **Fixed share: which C to tune for**, after R75.
6. **Ebony's `H-max`** (largest values into and out of each matmul), for the lane's half-precision dequantise.

## 4. Limits

- §1 is read from source. A build flag or a later patch in the kit's tree could change which type takes the head;
  §1.4's log is the check.
- §1.3's KV figures assume F16 KV at n_ctx 512 and full-width KV for Llama-2 and DeepSeek.
- §2.1 re-reads R69's `p1` passes with my own short code; medians exclude pass 0.
- §2.5's probabilities are judgement between measured points; the arm exists because the points do not cover the
  case.
- The 0.05 % in §0 item 4 is a resolution argument (int8 per row against Q6_K's sub-block scales), not a
  measurement.
- §2.7 reads the kit's Dart code for the hold; the native version in the probe is a re-implementation to be
  checked against it.

## 5. Sources (2026-11-23)

Repository: `lab/llama.cpp-c3r32/src/llama-model-loader.h` (lines 148 to 163), `src/llama-model-loader.cpp` (lines
1121 to 1148), `src/llama-model.cpp` (lines 1123 to 1170, 1888 to 1945), `ggml/src/ggml-cpu/ggml-cpu.cpp` (lines 48
to 79), `ggml/src/ggml-cpu/kleidiai/kleidiai.cpp` (lines 1494 to 1672, 1700 to 1921), `kernels.cpp` (lines 935 to
980); KleidiAI `kai/ukernels/matmul/pack/kai_rhs_pack_nxk_qsi8cxp_qsi8cx_neon.c`; `lab/c3/models/*-q4_0.gguf`
(sizes, *host*); R69 JSON, both runs (`p1` lines); R69-result.md lines 100 to 112; R72-result.md lines 186 to 195;
R74-result.md §5; `patches/llama.cpp-c3/0001-probe.patch` (`p1`), `0005-x3-gpu-probe.patch` (x3r, lines 1324 to
1376); `campaigns/02-25toks-5pct/STATUS.md` lines 364 to 367 (exec map, R67); `patches/app-c2/0006-kit-r22.patch` (hold) and campaign 2 `reports/README.md` R60; `tools/c3_freeze.sh` line 10 (the
freeze's build); [ebony-joins-r5.md](ebony-joins-r5.md) §3.1.
