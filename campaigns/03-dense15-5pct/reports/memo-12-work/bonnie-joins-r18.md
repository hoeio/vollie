# c3 memo-12 work, round 18 (Bonnie): kit specification draft, "which body can the phone hold, and how fast is the one it ships"

- **Date:** 2026-11-24. **Author:** bonnie, `claude-fable-5-2`. Follows [bonnie-joins-r17.md](bonnie-joins-r17.md);
  replaces the arm of [bonnie-joins-r8.md](bonnie-joins-r8.md) §2 (same lines, two steps and one block added). Not
  committed. No verdict here. **Ebony has not reviewed this file.**
- **Labels.** *device*, *host*, *estimated* as before. Triples are Yi / Llama-2 / DeepSeek. Every prediction here
  is *estimated*. Nothing was measured this round; no web source.
- **Status.** A draft for a pre-registration after R75. It needs the owner's permission for a device round and
  the lead's rules; the reading rules below are proposals. Working name: kit r36, round R76. No model is loaded,
  no numerics change, no change to the CPU module (the kit can ship r33's library again).

## 0. What the round answers

| question | block | decides |
|---|---|---|
| How fast is the body the shipped app would run (choice A: plain mapped rows, stock kernels)? | `x5p` | A's stock speed, today an estimate (4.3 / 3.9 / 3.9 tok/s, r17) |
| How fast is the head in each of its three places (KleidiAI, repack type, plain)? | `x5p` | the head knob's time (r12 task H3) and A's head |
| How fast does the phone pack and write a side file? | `x5p`, `x5w` | choice C's first run (12 to 18 s, r9) |
| Does a file-backed body stay resident at each subject's size, alone and beside a lane's GPU buffers, through a pass loop and a 60 s wait? | `x5s0` … `x5s5` | whether choices A and C exist for Llama-2 and DeepSeek (r17) |

Choice B (anonymous copies) is not re-tested here: R72's `anon34` / `anon42` and R75's `x4r` are its record.

## 1. Blocks and order

Button: "Run c3 probe r36 (body residency and stock kernel). Cool start." Order: **`x5p`, `x5w`, `x5s0`, `x5s1`,
`x5s2`, `x5s3`, `x5s4`, `x5sA`, `x5s5`.** Timing cells first, while the phone is cool; then memory steps from
the smallest risk to the largest. The r16 gate (up to 180 s) runs before every block. The report is saved before
and after every block. Every block starts with r32's `hdr` lines (zram, PSI, meminfo), x3's `x3 cond` lines and
r35's `x4 therm` rider (Android thermal status and kgsl `thermal_pwrlevel`, before and after).

### 1.1 `x5p`: the stock kernels and the three head placements (model-free)

All cells through ggml graphs, as R72's `l4`, `l5` and `l7` cells were. "plain" = weights allocated in the
default CPU buffer with no extra buffer type (what the app's rule gives a file over 2.7 GB).

| cell | what | arms |
|---|---|---|
| layer | one layer of each subject's shapes (Yi: k and v 512 rows; the others 4,096), 11 layers in a cold arena, as `l4 arm=ggml_stock` | buffer `CPU_KLEIDIAI` against `CPU` (plain), ABBA; thread sets `4,5,6` and `4,5,6,7`; 0.5 s + 1.5 s per arm |
| hot | Q4_0 gemv, one thread on cpu 4, 4,096 and 11,008 inputs, as `l5` | `CPU_KLEIDIAI` against `CPU`, ABBA, 1 s per arm |
| head | Q6_K head of 64,000 / 32,000 / 102,400 rows × 4,096, as `l7` | `CPU_KLEIDIAI`, `CPU_REPACK`, `CPU`; order K R P P R K; thread sets `7,4,5` and `4,5,6,7`; 1.5 s per arm |
| build | while each arena is built: bytes and seconds of the pack | every arena |

Arenas are built one at a time and freed (largest: 11 layers of Llama-2, 1.25 GB). Peak memory about 1.4 GB.

### 1.2 `x5w` and the residency steps

`x5w` writes one scratch file of **4.0 GB** into the app's files directory with `fsync`. Every step maps a prefix
of it. GPU buffers are copies of 11008 × 4096 on the general kernel, as `x4r`. Anonymous memory stands for the
KV cache and scratch; it is filled once and read once per pass.

| block | file F (map) | GPU G | anonymous A | total | stands for | phases |
|---|---|---|---|---|---|---|
| `x5s0` | 3.5 GB (x) | 0 | 0.05 GB | 3.55 | **Yi under A** (3.48 GB file) and **under C, CPU only** (3.38 GB); anchor against R69 (3.4 GB held twice) | 0, a |
| `x5s1` | 2.2 GB (x) | 1.15 GB | 0.05 GB | 3.4 | Yi under C with a lane; anchor against R74's x3r | 0, a |
| `x5s2` | 2.4 GB (x) | 1.35 GB | 0.15 GB | 3.9 | Llama-2 under C with a lane | 0, a, hold on, b, hold off, c |
| `x5s3` | 2.6 GB (x) | 1.25 GB | 0.15 GB | 4.0 | DeepSeek under C with a lane | 0, a, hold on, b, hold off, c |
| `x5s4` | 3.85 GB (x, then r) | 0 | 0.15 GB | 4.0 | **Llama-2 under A** (3.83 GB) and **Llama-2 or DeepSeek under C, CPU only** (3.77 / 3.84 GB) | 0, a, hold on, b, hold off, c; then map r: a only |
| `x5sA` | 4.0 GB (x) | 0 | 0.15 GB | 4.15 | **DeepSeek under A** (4.00 GB file) | 0, a, hold on, b, hold off, c |
| `x5s5` | 2.6 GB (x) | 1.50 GB | 0.30 GB | 4.4 | a lane with margin (an overlap copy, a head beside it) | 0, a |

Changes against r8 §2: the scratch file grows from 3.9 to 4.0 GB; `x5s0` and `x5sA` are new; the anonymous part of
`x5s2` to `x5s4` falls from 0.25–0.30 to 0.15 GB, because the cache is Q8_0 at n_ctx 512 (r15: 17 / 136 / 128 MiB);
`x5s4` gains the wait without the hold.

Phases inside a step (unchanged from r8 §2.2):

| phase | what | length |
|---|---|---|
| map, touch | map the prefix (x: `PROT_READ \| PROT_EXEC`, shared; r: `PROT_READ`), read it once on one thread | as long as it takes |
| 0 | CPU alone: four threads on 4-7 run R72's read kernel over the map, a quarter each, and read the anonymous memory | 5 passes |
| a | the same while a feeder on 0-3 runs the GPU ring (8 launches, then `clFinish`); with G = 0 it is phase 0 continued | 15 s, at most 200 passes |
| hold on | everything stops for 60 s; every 5 s, and at the end, the pages of the map lost since the phase began are asked back with `WILLNEED`; at the end wait until 98 % are back or 5 s have passed (campaign 2's hold, r8 §2.7) | 60 s + at most 5 s |
| b | as phase a | 10 passes |
| hold off | everything stops for 60 s, nothing is asked back | 60 s |
| c | as phase a | 10 passes |

## 2. Printed lines

### 2.1 `x5p`

| line | fields |
|---|---|
| `x5 pl build` | `subj=<yi15-6b\|llama2-7b\|deepseek7b\|head64000\|head32000\|head102400> buft=<CPU_KLEIDIAI\|CPU_REPACK\|CPU> src_mb= dst_mb= s= gbs= err=` (`gbs` = source bytes ÷ seconds of the buffer's `set_tensor` calls; for `CPU` it is a copy) |
| `x5 pl layer` | `subj= arm=<kai\|plain> buft= pos=<0..3> set= threads= layers_in_graph=11 layer_mb= ms_per_layer= body_ms= n_layer= gbs= calls= secs= proc_cpu_over_wall= proc_nivcsw= cpu0_ok= cpu0_off_calls=` + R72's tail (`mhz= mhz_range= ramp_s= ramp_ok= max_mhz=`) |
| `x5 pl layersum` | `subj= set= threads= ms_layer_kai= ms_layer_plain= plain_vs_kai= body_ms_kai= body_ms_plain=` (each the mean of its two arms) |
| `x5 pl hot` | `type=q4_0 buft= pos= k_in=<4096\|11008> rows= set=4 threads=1 ns_per_w= gbs= calls= secs=` + tail |
| `x5 pl hotsum` | `k_in= ns_per_w_kai= ns_per_w_plain= plain_vs_kai=` |
| `x5 pl head` | `type=q6_K vocab= k_in=4096 buft= pos=<0..5> mb= ms_per_call= gbs= set= threads= calls= secs= proc_cpu_over_wall= cpu0_ok= cpu0_off_calls=` + tail (`mb` and `gbs` on the bytes the buffer really holds, not the served size) |
| `x5 pl headsum` | `vocab= set= threads= ms_kai= ms_repack= ms_plain= repack_vs_kai= plain_vs_kai=` |
| failures | `x5 pl <cell> ok=0 reason=<alloc\|buft\|graph> buft= err=` (`reason=buft`: the tensor did not land in the buffer type asked for; the line prints the type it got) |
| `x5p end` | `s=` |

### 2.2 `x5w` and the steps (as r8 §2.3, with the pack line of r9 dropped: `x5 pl build` carries it)

| line | fields |
|---|---|
| `x5 write` | `gb= s= gbs= fsync_s= free_gb_before= err= errno=` |
| `x5 res stage=start` | `step=<s0\|s1\|s2\|s3\|s4\|sA\|s5> file_gb= gpu_gb= anon_gb= map=<x\|r> memavail_kb= cached_kb= swapfree_kb= vmrss_kb= rssanon_kb= rssfile_kb= vmswap_kb= kgsl_page_alloc=` |
| `x5 res stage=gpu_alloc` | `step= kern=general copies= gpu_mb= alloc_s= err=` (not printed when G = 0) |
| `x5 res stage=anon_fill` | `step= anon_gb= fill_s= majflt=` |
| `x5 res stage=file_map` | `step= file_gb= map= fallback=<0\|1> err= errno=` (`fallback=1`: the executable map was refused and the step runs read-only) |
| `x5 res stage=file_touch` | `step= s= gbs= majflt= resident_pct=` |
| `x5 res pass` | `step= phase=<0\|a\|b\|c> pass= s= gbs= gpu_gbs= majflt= minflt= memavail= cached= swapfree= vmswap= vmrss= rssfile= mhz= max_mhz= gpu_mhz= gpu_max_mhz= resident_pct=` |
| `x5 res hold` | `step= mode=<on\|off> s= polls= lost_pages_max= readback_mb= back_pct= back_wait_ms= resident_pct_before= resident_pct_after= majflt= memavail_before= memavail_after= hr_before= hr_after= thermal_pwrlevel=` |
| `x5 res sum` | `step= phase= map= passes= secs= cpu_gbs_median= cpu_gbs_min= cpu_gbs_first= gpu_gbs= majflt_median= majflt_sum= resident_pct_min=` |
| `x5 res abort` | `step= phase= pass= reason=<memavail\|alloc\|map\|cl> memavail_kb= err=` |
| `x5 res done` | `step= err=` |
| `x5w end`, `x5s<n> end` | `s=` |

Definitions as r8 §2.3: `gbs` counts map bytes read by the four threads; `gpu_gbs` is −1 with no GPU;
`resident_pct` is `mincore` over the map on every tenth pass and at the named points, −1 otherwise;
`cpu_gbs_first` is the phase's first pass; the hold's counters are −1 with the hold off.

## 3. Kill, abort and void

| event | reading |
|---|---|
| the kit aborts a step itself (`x5 res abort`: MemAvailable under 150 MB at the end of a pass, or an allocation or map error) | that step is **not held**; the kit frees everything and goes on |
| the process is killed (a started block has no `end` line) | that step is **not held (killed)**; later blocks are **not measured**. Not a void: everything before it is scored |
| `x5s0` not held | the arm is not comparable with R69; later steps are reported as description only |
| `x5 res stage=file_map fallback=1` | the step is scored as a read-only map and labelled so |
| a timing cell of `x5p` with `ramp_ok=0` or `cpu0_ok=0` | reported, not used for a ratio; the ratio is taken from the arms that are clean, and "not measured" if fewer than one clean arm per side |
| `x5 pl … ok=0 reason=buft` | that buffer's cells are "not measured" (the kit could not place the tensor) |
| voids | as R75: APK hash differs, power saving on, charger plugged. Cool start not met is a condition, not a void |

## 4. Reading rules (proposals for the pre-registration)

R0 = `cpu_gbs_median` of phase 0 in `x5s0`.

| reading | rule |
|---|---|
| a step **held** | phase a: `cpu_gbs_median` ≥ 0.80 × (R0 − 0.36 × `gpu_gbs`) (with G = 0: ≥ 0.80 × R0), `majflt_median` ≤ 50, and `x5 res done … err=0` (R74 §5's rule) |
| **survives the wait with the hold** | held, and phase b's `cpu_gbs_first` ≥ 0.80 × phase a's median |
| **survives the wait without help** | the same for phase c |
| **choice A exists for a subject** | its step (Yi `x5s0`, Llama-2 `x5s4`, DeepSeek `x5sA`) is held and survives the wait with the hold |
| **choice C exists, CPU only** | Yi `x5s0`; Llama-2 and DeepSeek `x5s4`: held and surviving with the hold |
| **choice C exists with a lane** | Yi `x5s1`; Llama-2 `x5s2`; DeepSeek `x5s3`: held and surviving with the hold |
| **A's stock speed** | 950 ÷ (`body_ms_plain` on 4-7 + O + `ms_plain` of the head), with O as the round's pre-registration fixes it (8 ms so far) |
| **the head knob's time** | `repack_vs_kai` per vocabulary size; the speed claim of r12 is dropped for a subject where it is ≥ 1.00 |
| **side-file first run** | body bytes ÷ `x5 pl build gbs` (KleidiAI) + file bytes ÷ `x5 write gbs` |

What the outcomes mean:
- **A exists and C does not** (the same memory, so this can only come from the lane steps): the fast kernels
  serve a CPU-only decode from a side file; the lane is a Yi-only lever.
- **Neither `x5s4` nor `x5sA` held:** Llama-2 and DeepSeek have no resident configuration on this phone at all,
  under any of the three choices. That closes the three-subject goal on residency alone, and would be said with
  these numbers.
- **`plain_vs_kai` at or under 1.25 on all three:** choice A is much better than estimated and deserves its own
  stack (width 1 only).

## 5. Bands (*estimated*; centre; 80 % band)

### 5.1 `x5p`

| row | field | Yi | Llama-2 | DeepSeek |
|---|---|---|---|---|
| layer, KleidiAI, 4-7 | `ms_layer_kai` | 3.9; 3.6–4.7 | 4.9; 4.4–5.8 | 4.7; 4.2–5.6 |
| layer, plain ÷ KleidiAI, 4-7 | `plain_vs_kai` | 1.62; 1.40–1.85 | the same | the same |
| layer, plain ÷ KleidiAI, 4,5,6 | `plain_vs_kai` | 1.75; 1.45–2.05 | the same | the same |
| body, plain, 4-7 | `body_ms_plain` | 195; 172–219 | 228; 201–255 | 212; 187–237 |
| head, KleidiAI, 7,4,5 | `ms_kai` | 9.0; 8.4–9.8 | 4.4; 4.1–4.9 | 13.9; 12.9–15.2 |
| head, repack type | `repack_vs_kai`; `ms_repack` | 0.85; 0.79–0.95. 7.7 | the same. 3.7 | the same. 11.8 |
| head, plain | `plain_vs_kai`; `ms_plain` | 1.7; 1.4–2.1. 15 | the same. 7.5 | the same. 24 |

| row | field | prediction |
|---|---|---|
| hot gemv, one A78, KleidiAI | `ns_per_w_kai` | 0.081; 0.076–0.088 (R72: 0.0806, *device*) |
| hot gemv, plain ÷ KleidiAI | `plain_vs_kai` | 2.2; 1.9–2.6 (count 2.4, laptop 2.0) |
| pack rate, KleidiAI, Q4_0 | `x5 pl build gbs` | 1.0; 0.7–1.5 GB/s (laptop 4.1 to 4.4, r9) |
| build rate of the int8 head | `x5 pl build gbs` (head, `CPU_KLEIDIAI`) | 0.15; 0.08–0.3 GB/s of served bytes (a scalar float pass per weight) |
| P(the repack head is slower than the KleidiAI head on at least one size) | – | 0.2 |
| P(`plain_vs_kai` for the layer ≤ 1.25 on any subject) | – | 0.07 |

The KleidiAI layer row repeats R72's `l4 arm=ggml_stock` on 4-7 (3.94 / 4.87 / 4.70 ms, *device*, taken at capped
clocks); a cooler phone reads lower.

### 5.2 `x5w` and the steps

| row | field | prediction |
|---|---|---|
| write rate | `x5 write gbs` | 0.52; 0.40–0.65 (R69: 0.45 and 0.61, *device*) |
| R0 | `x5s0` phase 0 `cpu_gbs_median` | 30; 26–33 |
| a held step with a GPU, phase a | `cpu_gbs_median`; `gpu_gbs`; `majflt_median` | 21; 18–24. 21; 19–23. 0; 0–10 |
| after 60 s with the hold | `resident_pct_after` | 98; 90–100 |
| after 60 s without | `resident_pct_after` | 85; 55–100 |
| first pass after a wait that lost 10 % of the pages | `cpu_gbs_first` ÷ median | 0.5; 0.2–0.9 |
| GPU cap moves in any step | `gpu_max_mhz` < 840 | P 0.10 |
| executable map refused | `fallback=1` | P 0.05 |

| step | held | survives with the hold | survives without | killed or aborted |
|---|---|---|---|---|
| `x5s0` (3.5 GB mapped) | 0.90 | – | – | 0.02 |
| `x5s1` (2.2 + 1.15 GPU) | 0.85 | – | – | 0.03 |
| `x5s2` (2.4 + 1.35 GPU) | 0.60 | 0.50 | 0.25 | 0.07 |
| `x5s3` (2.6 + 1.25 GPU) | 0.55 | 0.45 | 0.22 | 0.08 |
| `x5s4` (3.85 GB mapped), map x / map r | 0.70 / 0.40 | 0.60 | 0.30 | 0.08 |
| `x5sA` (4.0 GB mapped) | 0.60 | 0.50 | 0.25 | 0.10 |
| `x5s5` (2.6 + 1.5 GPU + 0.3) | 0.20 | – | – | 0.30 |

Anchors: R69 `p1` held 3.4 GB twice under both maps, held 4.2 GB twice executable and once of twice read-only,
never 4.6 GB (*device*, no GPU buffers, no anonymous memory beside); R74 x3r held 3.5 GB (0.85 GPU + 2.6
anonymous). `x5s2` and `x5s3` are 0.05 above r8's values because their anonymous part is smaller.

### 5.3 Per subject, what the round is expected to say

| | A exists | C exists, CPU only | C exists with a lane | A's stock speed if it exists |
|---|---|---|---|---|
| Yi | 0.88 (`x5s0`; no wait phase, so "held") | 0.88 | 0.85 (`x5s1`) | 4.3; 3.9–4.9 tok/s |
| Llama-2 | 0.60 (`x5s4`) | 0.60 | 0.50 (`x5s2`) | 3.9; 3.5–4.4 |
| DeepSeek | 0.50 (`x5sA`) | 0.60 (`x5s4`) | 0.45 (`x5s3`) | 3.9; 3.5–4.4 |
| all three | 0.35 | 0.45 | 0.30 | – |

## 6. Cost

- **Phone time** (*estimated*): `x5p` about 2.5 min (layer cells 50 s, head cells 55 s, hot cells 8 s, arena
  builds 30 to 40 s); `x5w` 8 to 10 s; `x5s0`, `x5s1`, `x5s5` 30 s each; `x5s2`, `x5s3`, `x5sA` about 170 s each;
  `x5s4` about 190 s. **About 14 minutes of blocks, plus nine gates** of up to 180 s.
- **Storage:** 4.0 GB of scratch, deleted at the end (R69 had 42.6 GB free, *device*).
- **Peak memory:** each step's total plus the app; `x5s5` is last because it is the likeliest to be killed.
- **Code:** x4r's GPU ring and pass loop; R69 `p1`'s scratch file and maps; R72's `l4`, `l5`, `l7` cells with a
  choice of buffer type; campaign 2's hold, re-implemented natively. New: placing a probe tensor in a named
  buffer type (the plain one needs the extra types switched off for that tensor; the repack one can use the
  campaign-2 knob with a tensor named `output.weight`), the phases, `mincore`, the abort rule.
- **Not in scope:** the side file itself; any model; choice B; the lane; a hold stronger than campaign 2's.

## 7. Next questions, ranked

1. **After R75: fold its results into this draft** (`x4r` for choice B at 4.1 GB; `gpu_max_mhz`; the gate series
   after lane-like load) and hand the lead a version with final numbers.
2. **The same choice for the head and the cache at claim time:** with choice C the head (0.1 to 0.4 GB) and the
   KV stay anonymous. Write the exact anonymous budget per subject under C with the head in each place, against
   the steps above. Mine; short.
3. **What a dense decode on the host can settle before any phone round:** per-op profile of a token (O and the
   attention's share), the three KV types, flash attention's "auto". A specification for Tom that uses the
   existing profiler knob. Mine.
4. **The ladder under A and under C** once `x5p` or R75 gives numbers.
5. **DeepSeek's head; cold read rate; fixed share.**

## 8. Limits

- Every probability is a judgement between a few measured points; the round exists because they do not cover
  the case.
- The steps use a read kernel over a scratch file, not the real kernels over real weights; they test memory, not
  the decode. A step that holds says the pages stay; it does not say the app's own memory (Dart, the UI, the
  runtime's scratch) leaves the same room. The anonymous part stands in for that only roughly.
- `x5p`'s plain cells assume the probe can place a tensor in the default buffer; if the runtime refuses, the
  cell prints `reason=buft` and the question stays open.
- The layer bands are scaled from R72's cells, which ran at capped clocks.
- Field names are proposals; Tom's build note fixes them.

## 9. Sources (2026-11-24)

Repository: [bonnie-joins-r8.md](bonnie-joins-r8.md) §2 (the arm this replaces), r9 §4, r12 §1.3 and §3, r15 §1.3,
r16 §1, r17 §1; R69 JSON (`p1`) and `R69-result.md`; R72 JSON (`l4`, `l5`, `l7` line formats and values) and
`R72-result.md` (lines 118, 186 to 195); `R74-result.md` §5; `patches/llama.cpp-c3/0001-probe.patch` (`p1`),
`0003-r72-rows.patch` and `0005-x3-gpu-probe.patch` (cells and x3r); `tom-r35-build.md` (tails, riders);
`patches/app-c2/0006-kit-r22.patch` (the hold); `lab/app-c3/lib/device.dart:591`.
