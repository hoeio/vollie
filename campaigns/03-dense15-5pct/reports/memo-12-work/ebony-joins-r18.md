# c3 memo-12 work (Ebony, round 18): why the x4s split saved 7.7 %, and what a persistent fast kernel is worth

Date: 2026-11-25. Author: Ebony (`claude-fable-5-2`). Not committed; the lead commits. No verdict here.
Follows [ebony-joins-r17.md](ebony-joins-r17.md). Read for it: [R75-result.md](../R75-result.md) (both sessions), the
session-2 JSON `reports/device/r75a1-c3probe-r36-2026-11-24T1345.json` line by line for x4s, x4m, x4e, and the probe
source `patches/llama.cpp-c3/0007-x4-join-probe.patch` (functions `x4s`, `launch`, `spin_ev`, kernels `q4n`, `q4p`).
Written before reading Bonnie r34 (§6 compares).

Labels: *device* = printed by the phone; arithmetic on *device* values is written beside its inputs; *estimated* =
model output; *external* = a published source, with how much of it I read. Triples are Yi / Llama-2 / DeepSeek.
Script: [ebony_split_r18.py](ebony_split_r18.py) (`python3 -I -B ebony_split_r18.py <repo root>`); it loads the
committed stage model unchanged. Nothing ran on a phone.

## 0. Answer first

1. **0.923 is the mean of two different phones.** At share 0.55 the first arm took 1.880 ms and the second 1.508 ms
   (*device*). The first arm lies in a stretch of six consecutive arms (pos 5 to 10, about 15 s, starting with the
   first cap step) in which everything was slow, the feeder arm included: its twin arms at share 0.25 read 2.022
   and 1.516 ms (*device*). The second arm alone is 0.80 to 0.85 of CPU alone, depending on which CPU-alone arm it
   is divided by. **My reading of the cell without the bad stretch: 0.82, not 0.92; the model said 0.66.**
2. **The rest of the gap (0.66 to 0.82) is the join, not the coupling.** `K_meas` 149 µs is QUEUED to START only.
   In the three arms where the GPU was on the critical path, the cycle time that is not kernel time was **384, 400
   and 549 µs** (*device* arithmetic: `enq_us` + `cpu_ms` + `wait_ms` − `gpu_dev_ms`). Two things the rule did not
   count: (a) launch plus `clFlush` takes 124 to 245 µs on the lead thread **before** `ggml_backend_graph_compute`
   starts, so all four CPU threads idle through it, where the stage model lets the CPU work during K; (b) the spin
   sees the event 150 µs or more after the GPU's END stamp. The in-situ join is about 400 µs, close to R74's 452.
3. **The coupling term in the cool arms is the committed one.** Fitted on the four arms before the first cap step:
   the CPU rows run **1.36 times slower while the kernel runs** (plus 47 µs fixed per graph). The committed `pair`
   model has 1.353. S10 at share 0.50 was 1.878 in the bad-stretch arm and 1.374 in the other; the priced band was
   1.20 to 1.40. **My band: 1.28 to 1.45 at full clocks (centre 1.36); 1.5 to 2.1 once the big cores are capped at
   1,420 MHz; 2.4 to 3.2 in the bad stretch, which I cannot explain from the file.** §1.4 names four candidates
   and §4 the checks.
4. **F survives, and by more than the R75 price.** The phone paid about 400 µs per split, not 162. A persistent
   kernel removes all of it. Against the lane as probed, in the same heat state: **+2.0 / +2.1 / +1.9 tok/s at the
   general kernel's rate, +1.2 / +1.3 / +1.2 at 0.8 of it, +0.3 / +0.4 / +0.4 at 0.6** (width 1, built pieces
   only, *estimated*). The +0.5 line holds down to a rate of about 0.65.
5. **But most of F's gain is available without a persistent kernel.** Design G (§3.3): the launch moves inside the
   pool's job so three threads compute while one launches, and a one-line trailing kernel writes a done flag in
   shared memory that the workers poll. Uses the upstream kernel as it is. **+0.8 / +1.0 / +0.9 over as probed;
   F over G is +1.2 / +1.1 / +1.0 at full rate and +0.35 / +0.25 / +0.23 at 0.8.** So F is worth its build over G
   only if the persistent kernel keeps about 0.9 of the general kernel's rate. Build G first.
6. **No prior art for a persistent kernel on Adreno, and none for a faster gemv.** The pattern is known on NVIDIA
   (Pagoda's MasterKernel, persistent threads). On mobile the published answer to the same problem is HeteroLLM's:
   keep ordinary kernels, replace `clFinish` (about 400 µs on Adreno, their number) with a flag in mapped memory.
   That is design G. The upstream general kernel already draws 24 to 28.5 GB/s (*device*), as much as four CPU
   threads; published GPU-alone figures on a newer Snapdragon are 65 to 73 % of practical bus peak, the same class.
7. **Nothing here reaches 12.** F in the rung-7 form at full rate: 11.86 / 11.66 / 11.24 cool (*estimated*).

## 1. Question 1: the split gap

### 1.1 What one x4s cycle is

From the source (`x4s`, lead mode), one matrix is: `launch()` (15 `clSetKernelArg`, one enqueue on the profiling
queue), `clFlush`, then `ggml_backend_graph_compute` on the CPU rows (pool on 4-7, the lead is worker 0), then
`spin_ev`. So the cycle is `enq_us` + `cpu_ms` + `wait_ms`, in series on the lead thread; it matches `ms_per_mat`
within 4 % in 16 of 18 arms (4.3 % and 6.3 % in the other two) (the means include the warm-up, the window does not). `gpu_ms` − `gpu_dev_ms` is
QUEUED to START. It equals `enq_us` within 21 µs in the GPU-critical arms (179 against 200; 114 against 126; 112
against 124): **the kernel starts when the flush returns, and the CPU rows start at the same moment, not before.**

The stage model's line is `t_split = K + (B − ca·K) ÷ (cb + g)`: the CPU works at its alone rate through K. In the
probe it does not. That alone is worth `enq × ca ÷ (cb + g)`, about 0.2 × 0.6 = 0.12 ms on this cell.

### 1.2 Arm by arm (*device*; ratios arithmetic)

`base` is the nearer CPU-alone arm: 1.776 ms (pos 0, full clocks) for pos 1 to 10, 1.895 ms (pos 21, big cores
capped at 1,420 MHz, GPU idle at a 491 cap) for the rest. Note the pair: **CPU alone lost 6.7 % between full clocks
and a 1,420 cap.** The matrix is bus-bound; the cap alone does little.

| pos | share | cap p4 / GPU MHz | `ms_per_mat` | ÷ base | `cpu_ms` | S10 | excess µs | overlap ms | excess ÷ overlap | non-kernel µs | GPU GB/s | `enq_us` |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.20 | 2419 / 840 | 1.728 | 0.973 | 1.584 | 1.115 | 163 | 0.391 | 0.42 | – | 25.6 | 216 |
| 2 | 0.25 | 2419 / 840 | 1.683 | 0.948 | 1.500 | 1.126 | 168 | 0.523 | 0.32 | – | 24.2 | 214 |
| 3 | 0.30 | 2419 / 840 | 1.622 | 0.913 | 1.447 | 1.164 | 204 | 0.609 | 0.33 | – | 25.2 | 213 |
| 4 | 0.35 | 2419 / 840 | 1.610 | 0.907 | 1.396 | 1.209 | 242 | 0.680 | 0.36 | – | 26.0 | 214 |
| 5 | 0.40 | 2419 / 840 | 1.853 | 1.043 | 1.565 | 1.469 | 499 | 0.861 | 0.58 | – | 23.6 | 230 |
| 6 | 0.45 | 2150 / 840 | 1.830 | 1.030 | 1.591 | 1.629 | 614 | 0.966 | 0.64 | – | 23.5 | 234 |
| 7 | 0.50 | 2150 / 840 | 2.017 | 1.136 | 1.668 | 1.878 | 780 | 1.133 | 0.69 | – | 22.4 | 203 |
| 8 | 0.55 | 2150 / 840 | 1.880 | 1.059 | 1.619 | 2.026 | 820 | 1.212 | 0.68 | – | 23.1 | 210 |
| 9 | 0.60 | 2150 / 840 | 1.964 | 1.106 | 1.574 | 2.216 | 864 | 1.393 | 0.62 | 549 | 21.8 | 200 |
| 10 | 0.25 feeder | 2150 / 840 | 2.022 | 1.139 | 1.988 | – | – | – | – | – | – | – |
| 11 | 0.25 feeder | 1785 / 676 | 1.516 | 0.800 | 1.606 | – | – | – | – | – | – | – |
| 12 | 0.60 | 1785 / 676 | 1.659 | 0.875 | 1.198 | 1.580 | 440 | 1.198 | 0.37 | 400 | 23.9 | 126 |
| 13 | 0.55 | 1785 / 676 | 1.508 | 0.796 | 1.245 | 1.460 | 392 | 1.137 | 0.34 | 384 | 24.6 | 124 |
| 14 | 0.50 | 1785 / 676 | 1.518 | 0.801 | 1.302 | 1.374 | 355 | 1.075 | 0.33 | – | 23.6 | 147 |
| 15 | 0.45 | 1420 / 676 | 1.593 | 0.841 | 1.350 | 1.295 | 308 | 0.945 | 0.33 | – | 24.0 | 233 |
| 16 | 0.40 | 1420 / 676 | 1.819 | 0.960 | 1.577 | 1.387 | 440 | 0.901 | 0.49 | – | 22.6 | 239 |
| 17 | 0.35 | 1420 / 676 | 1.900 | 1.003 | 1.672 | 1.357 | 440 | 0.754 | 0.58 | – | 23.5 | 238 |
| 18 | 0.30 | 1420 / 676 | 1.974 | 1.042 | 1.737 | 1.309 | 411 | 0.682 | 0.60 | – | 22.5 | 245 |
| 19 | 0.25 | 1420 / 676 | 1.985 | 1.047 | 1.759 | 1.238 | 338 | 0.599 | 0.56 | – | 21.2 | 239 |
| 20 | 0.20 | 1420 / 491 | 2.065 | 1.090 | 1.854 | 1.223 | 338 | 0.570 | 0.59 | – | 17.6 | 238 |

S10 = `cpu_ms` ÷ ((1 − share) × base). Excess = `cpu_ms` − (1 − share) × base. Overlap = min(`gpu_dev_ms`,
`cpu_ms`). Non-kernel = cycle − `gpu_dev_ms`, given where `wait_ms` > 0.1 (the GPU is critical). GPU GB/s = weight
bytes of the GPU rows ÷ `gpu_dev_ms`. The cap column is the policy-4 maximum on the arm's tail; inside pos 5 the
cpu-7 clock already read 2,150.

Four groups:

- **Clean (pos 1 to 4):** full clocks, GPU at 840. Excess = 47 µs + 0.267 × overlap (least squares, four points):
  the CPU rows run at 0.733 of their alone rate while the kernel runs, **slowdown 1.36**, plus 47 µs fixed per
  graph. Bus sum while both run: 28.57 ÷ 1.36 + 25.3 = **46.3 GB/s**; R74's bus sum is 46.1. The committed `pair`
  model has slowdown 1.353. In these arms the split still loses its gain: 0.907 at best, because 214 µs of `enq_us`
  sit in front of every graph.
- **Bad stretch (pos 5 to 10):** nominal clocks nearly full (2,150 to 2,419), GPU at 840, and yet the CPU loses 0.58
  to 0.69 of every overlapped millisecond (slowdown 2.4 to 3.2), the GPU kernel is 10 % slower at the same GPU clock
  (21.8 to 23.6 against 24.2 to 26.0), and the feeder arm inside it is 33 % slower than its twin outside. The lead
  thread shows 116 to 320 voluntary switches per arm here against 1 to 2 in the clean arms (third field of `thr`),
  so it blocks inside the graph compute. Both sides slower at once: the bus sum while both run fell to about 33 to
  35 GB/s.
- **Capped at 1,785, GPU 676 (pos 11 to 14):** the best arms of the block. 0.796 to 0.875 of the last CPU arm.
  `enq_us` 124 to 147 against 200 to 245 everywhere else; these are the arms where the lead ends in the spin, so
  the next launch follows the last kernel with no idle gap on the GPU. Reading: a launch into an idle GPU costs
  about 100 µs more than a launch into a busy one. Not shown by any other cell.
- **Capped at 1,420 (pos 15 to 20):** excess = 151 µs + 0.345 × overlap: slowdown 1.53 with the fixed term three
  times the clean one; if the fixed term were still 47 µs the slowdown would be 1.77 to 2.14. The GPU kernel is 5 %
  slower than at the 1,785 cap with the same 676 MHz GPU clock. The split is slower than CPU alone at every share
  below 0.40.

### 1.3 The gap in parts (ratio units of the cell; arithmetic and *estimated*)

| part | size | basis |
|---|---|---|
| model at K 149, coupling 1.30, no slack | 0.644 (the lead's 0.663 has the slack) | committed line |
| the join is about 400 µs and serial in front of the CPU rows, not 149 µs beside them | +0.14 to +0.16 | non-kernel 384 / 400 µs in the two clean GPU-critical arms; my line `0.40 + B ÷ (cb + g)` gives 0.825 |
| coupling 1.36 to 1.40 instead of 1.30 | +0.015 | clean fit |
| **cell without the bad stretch** | **0.82 (0.80 to 0.85)** | pos 13 ÷ pos 21 = 0.796; ÷ probe mean = 0.822; ÷ pos 0 = 0.849 |
| the first arm of every share from 0.40 up fell in the bad stretch | +0.10 | (1.880 + 1.508) ÷ 2 ÷ 1.8355 = 0.923 |

`K_r` 1,064 µs in the result is therefore not one cost. About 400 µs of it is a real join (K_meas missed the
flush-side idle and the END-to-seen tail), and the rest is a 15 s state of the phone.

**The END-to-seen tail.** In the GPU-critical arms, `cpu_ms` + `wait_ms` − `gpu_ms` = 170, 160, 148 µs (pos 9,
12, 13). That is the time from the END stamp to the spin's exit if QUEUED was stamped at the end of the launch,
and up to `enq_us` more if it was stamped at the start. It is from means, so it is a floor with an uncertain
ceiling: **150 to 350 µs**. Session 2's cpu-4 spin on an idle phone had a whole `ovh_us` of 131 to 140 µs, so the
tail is longer in situ. My reading (not shown): the driver's completion thread has to be scheduled, cores 4-7 are
full of pinned workers, and cores 0-3 sit at 1,094 MHz. The x4e counter was visible to the CPU in 3.75 µs
(*device*), with no driver thread in the path.

### 1.4 Candidates for the bad stretch and for the cap sensitivity (none shown by the file)

| # | candidate | for | against | my P (contributes) |
|---|---|---|---|---|
| C1 | a transient at the thermal step (Samsung's thermal service doing something beyond the frequency cap for some seconds) | starts with the first cap step and the headroom jump 0.79 → 0.90; ends by itself while the caps deepen; hits lead and feeder modes alike | no line names an action | 0.40 |
| C2 | the DDR and cache-bus clock follows the capped CPU clock. Qualcomm documents a static map from CPU frequency to L3 / LLCC / DDR frequency plus a traffic monitor on the CPU side (*external*, vendor documentation for a newer kernel, read as a search summary) | CPU alone hardly notices the caps (6.7 %), but CPU and GPU together both slow at an unchanged GPU clock: the pattern of a smaller bus sum (46 → 36 to 40 at the 1,420 cap) | the 1,785 arms are the best of the block; `devfreq` of the bus is unreadable (errno 13) | 0.30 for the 1,420 arms, 0.10 for the bad stretch |
| C3 | background memory work: each arm allocates and frees its rings (203 MB in all; the GPU share of it grows from 41 MB at share 0.20 to 122 MB at 0.60) with swap 71 % full; reclaim and zram draw bus | bigger GPU rings from pos 5 on; x4r saw other processes' pages go | allocation is outside the timed window; MemAvailable 2.2 GB | 0.15 |
| C4 | pool parking: the workers idle through `enq_us`, sleep (hundreds of voluntary switches on 5 and 6 in every lead arm, 0 to 5 in CPU alone) and wake late; slower at low clocks | the fixed term 47 → 151 µs; the lead's own sleeps in the bad stretch | the feeder arm at pos 10 has no `enq` stall and is as slow | 0.30 for the fixed term, 0.10 for the bad stretch |

Published measurements on shared-DRAM contention between CPU cores and an integrated GPU exist for Jetson boards,
not for Snapdragon: BWLOCK++ (Ali and Yun, ECRTS 2018, arXiv 1712.08738, *external*, search summary) reports GPU
kernels slowed up to 3× by memory-heavy CPU co-runners on a Jetson TX2; Cavicchioli et al. (ETFA 2017) characterise
the same path qualitatively. HeteroLLM (arXiv 2501.14794 v2, 2025-10-04, *external*, read through a page summary)
gives the one Snapdragon figure I found: on an 8 Gen 3 a single processor draws 40 to 45 GB/s in decode against a
practical peak of 61.9 and a theoretical 68; GPU and NPU together reach 59.5. It does not measure one processor
slowing another. Snapdragon 888: LPDDR5 at 3,200 MHz, 51.2 GB/s theoretical (*external*, AnandTech's table); our
46 GB/s sum is 90 % of that. **I found no published CPU-beside-GPU slowdown for Snapdragon 888 or Adreno 660.**

### 1.5 My band for the coupling term

CPU slowdown while a GPU kernel drawing about 24 GB/s runs beside four CPU threads (σ; the committed `pair` value
is 1.353):

| phone state | centre | band | basis |
|---|---|---|---|
| full clocks, GPU 840 | **1.36** | 1.28 to 1.45 | x4s pos 1 to 4 fit 1.36; x4m one-column cell 1.27 at GPU duty 0.875, so 1.32 while the kernel runs; R74 1.297 at duty 0.875, so 1.355 |
| big cores capped at 1,785, GPU 676 | 1.40 | 1.33 to 1.58 | S10 of pos 12 to 14 (they include the fixed term) |
| big cores capped at 1,420 | 1.8 | 1.5 to 2.1 | pos 16 to 20, both treatments of the fixed term |
| the bad stretch | – | 2.4 to 3.2 | pos 5 to 9; a state, not a term; I do not price it |

As S10 at share 0.50 (the prereg's measure): 1.35 to 1.45 cool; the measured 1.62 is the mean of 1.878 and 1.374.
The term to refit is not σ. It is the dead time: **400 µs serial per split (340 to 570) for the design as probed.**

## 2. Question 2: prior art for F

### 2.1 Persistent kernels

- **NVIDIA, well known:** Pagoda (Yeh et al., PPoPP 2017; *external*, search summary) keeps a MasterKernel alive
  on the GPU; the CPU posts tasks to a table in shared memory and GPU threads poll it. Persistent-thread kernels
  spin-wait for work; the stated costs are power while spinning and loss of the native block synchronisation.
  Mirage's persistent kernel (arXiv 2512.22219) is the LLM-serving form. All CUDA, all desktop or server.
- **Adreno, OpenCL, Vulkan on phones: I found none.** Searches for a persistent kernel, for SVM atomics used as a
  CPU-to-kernel channel, and for a kernel that outlives one launch on Adreno returned nothing. R75's x4e is the
  only measurement I know of: flag seen by the running kernel in 0.9 µs, counter seen by the CPU in 3.75 µs, only
  with OpenCL C 2.0 atomics on fine-grain SVM; nothing visible in the two 1.x modes (*device*).
- **The mobile answer to the same cost is not a persistent kernel.** HeteroLLM (SOSP 2025 work, *external*, page
  summary): `clFinish` costs "approximately 400 microseconds" on their Adreno; they map the buffers into both
  address spaces, put a flag beside the output tensor, sleep for the predicted kernel time and then poll the flag
  for a few microseconds. Reported effect on decode: 2.2× to 4× over their own baseline. That is a done flag on
  ordinary kernels: design G below.
- **A second 400 to 550 µs figure:** a Qualcomm fork note on llama.cpp's split-K decode gemv says an extra reduce
  dispatch costs "about 550 µs of host round-trip against 2.7 µs of GPU work" (*external*, as summarised from the
  PR 26477 page, merged 2026-09-03; I did not read the commit itself). Same size as our in-situ join.

### 2.2 A gemv that reaches a good fraction of bandwidth

- **Upstream's general Q4_0 kernel is already there.** 28.5 GB/s of weight bytes alone at 840 MHz, 24.5 at 676
  (*device*, x4m), 24.2 to 26.0 beside four CPU threads (*device*, x4s). Four CPU threads alone: 28.6. Theoretical
  bus 51.2 (*external*): 56 % alone, and the pair reaches 90 %.
- **Published GPU-alone decode on a newer chip is the same class:** 40 to 45 of a practical 61.9 GB/s, 65 to 73 %
  (*external*, HeteroLLM). ML Drift (arXiv 2505.00232, *external*, search summary) reports Llama-3.1-8B at 12.7
  tok/s on an Adreno 750; I do not convert it because I did not read its weight sizes.
- **Newest upstream work does not touch this kernel.** PR 26477 (opened 2026-08-02, merged 2026-09-03; *external*,
  page summary): split-K decode gemv and a tiled lm_head for q4_K, q6_K and q8_0, gated to X2E and A8x after
  measurement; split-K ungated was −1.3 % on an Adreno 840 and −20 % on an 850; the tiled layout regresses on A7x.
  Nothing for Q4_0 and nothing for 6xx. No faster gemv to borrow.
- **Why the probe's kernel ran at 1.83 GB/s:** `q4n` reads 18-byte blocks byte by byte from a global buffer, with
  64 work-items and a barrier tree per row. Upstream reads the quants through an image, keeps scales apart, and
  takes two rows per subgroup. The 13× is layout and access path, not the loop. (iv) passed because both arms were
  the slow kernel; it says a row claim costs under 1.5 % of a 60 µs row, which is no bound for a 4 µs row.

### 2.3 What a persistent version of the general kernel has to solve

| # | problem | size | how it is decided |
|---|---|---|---|
| P1 | **weights of many matrices in one launch.** A kernel's image arguments are fixed at launch. `max_alloc_mb` 923, `img_max_texels` 134,217,728 (*device*). One layer's gate‖up is 50.7 MB, so one argument holds at most 18 layers of it | design: a kernel per layer (all the layer's lane matrices as fixed arguments) needs no switch; a kernel per token needs 2 to 4 image arguments per matrix kind and a branch | source read, host build |
| P2 | **row claim granularity.** One atomic per row is too many at 4 µs a row | claim 64 to 128 rows per atomic; cost under 1 % by arithmetic | device rate |
| P3 | **rate inside the loop:** image reads, local reduction and barriers inside `for (;;)`, a compiler that may not unroll | unknown; this is the precondition. Pass ≥ 0.90 of the general kernel, kill < 0.80 (§3.2) | device only |
| P4 | **lifetime.** kgsl's dispatcher has a 2,000 ms command timeout and a 200 ms timer that compares progress registers (*external*, an older msm kernel's `adreno_dispatch.c`, page summary; the Flip 3's values not read). x4e ran one kernel for 188 ms without a fault (*device*). A token is 85 to 145 ms | per layer (3 to 5 ms) is safe; per token is near the 200 ms timer | kernel source read, then device |
| P5 | **the screen.** The GPU also draws the app. HeteroLLM saw a game drop to 0 fps under back-to-back OpenCL kernels (*external*). A kernel that holds the GPU for a token delays every frame by up to a token | per layer bounds it at one layer | device: frame times |
| P6 | **heat.** A spinning kernel keeps the GPU at 100 % between stages; x4s's `gpu_busy` was 23 to 77 %. The GPU cap fell to 676 and 491 after about 60 s of probe load, and the kernel's rate with it (28.5 → 24.5 → 17.6 GB/s, *device*) | at the 491 cap F runs at 0.62 of its rate and most of its gain is gone | device: 60 s spin against 60 s idle |
| P7 | **the CPU side.** The pool must not stall at the hand-over (§1.1) and must see the done counter without the driver | same change as design G | host |

## 3. Prices (*estimated*; committed stage model, `pair`, coupling and dead time set per row)

### 3.1 On the x4s cell (ratio to CPU alone)

| design | cool (σ 1.40) | big cores capped (σ 1.8) |
|---|---|---|
| committed line (K 149, σ 1.30) | 0.644 | – |
| as probed: 400 µs dead, serial | 0.825 (phone, clean arm: 0.80 to 0.85) | 0.920 |
| F at the general kernel's rate: 18 µs + 60 µs slack | 0.649 | 0.745 |
| F at 0.8 of the rate | 0.727 | – |

### 3.2 On the subjects (tok/s; stock 6.90 / 6.21 / 6.22)

"As probed" has no slack added: the 400 µs is what the cell paid, imbalance included. F and G carry my 60 µs.

| design | width 1, O 8, measured head (built pieces only) | Δ against as probed | rung-7 form (width 2 by two launches, C5 0.93, head × 0.5, O 5.2) | Δ |
|---|---|---|---|---|
| committed: K 162.3 beside the CPU, slack 60 | 9.11 / 8.53 / 8.33 | – | 11.33 / 11.19 / 10.81 | – |
| **as probed, cool: 400 µs dead, σ 1.40** | **7.59 / 6.82 / 6.78** | 0 | 9.62 / 9.18 / 8.96 | 0 |
| as probed, 340 µs / 570 µs | 7.82 / 7.10 / 7.04; 7.28 / 6.51 / 6.49 | +0.2 to +0.3; −0.3 | 9.84 / 9.48 / 9.24; 9.23 / 8.82 / 8.63 | |
| as probed, capped: σ 1.8 | 7.36 / 6.57 / 6.56 | (own base) | 9.55 / 9.09 / 8.89 | |
| **G: launch inside the job, done flag** (K 200 beside the CPU, 127 µs dead) | 8.41 / 7.85 / 7.72 | **+0.82 / +1.03 / +0.94** | 10.59 / 10.46 / 10.14 | +0.97 / +1.29 / +1.19 |
| G, capped | 8.05 / 7.43 / 7.34 | +0.69 / +0.85 / +0.78 | 10.49 / 10.34 / 10.04 | |
| **F, rate 1.0** (78 µs dead) | 9.59 / 8.93 / 8.68 | **+2.00 / +2.11 / +1.90** | 11.86 / 11.66 / 11.24 | +2.24 / +2.49 / +2.28 |
| F, rate 0.8 | 8.76 / 8.10 / 7.95 | +1.17 / +1.28 / +1.17 | 11.12 / 10.91 / 10.55 | |
| F, rate 0.6 | 7.88 / 7.23 / 7.16 | +0.29 / +0.41 / +0.38 | 10.35 / 10.11 / 9.83 | |
| F, capped, rate 1.0 / 0.8 | 8.91 / 8.25 / 8.08; 8.04 / 7.38 / 7.30 | +1.55 / +1.67 / +1.52; +0.68 / +0.81 / +0.74 | 11.68 / 11.48 / 11.07; 10.63 / 10.40 / 10.09 | |
| F per layer (one launch of about 400 µs per layer, +100 µs per split) | 8.57 / 8.04 / 7.89 | +0.98 / +1.22 / +1.11 | 10.71 / 10.62 / 10.29 | |

Readings:

- **The lane as the phone ran it is a +0.6 to +0.7 tok/s lever** (7.59 / 6.82 / 6.78 against stock). The lead's crude
  figure from the 0.923 ratio was 7.40 / 6.68 / 6.66. Mine is a little higher because I take the clean arm.
- **F's "worth a build" survives my slack and the refit**, by a wide margin against the lane as probed, down to a
  persistent-kernel rate of about 0.65. The R75 price (+0.36 to +0.45 with my slack) understated it because it took
  the join for 162 µs.
- **Against G, F needs the rate.** F − G at width 1: +1.18 / +1.08 / +0.96 at rate 1.0; about +0.77 / +0.66 / +0.60
  at 0.9 (interpolated); +0.35 / +0.25 / +0.23 at 0.8. **My line: F is worth its build over G only at ≥ 0.9.**
- **F per layer is G-sized.** If P4 or P5 forces one launch per layer, F gives +0.2 over G and is not worth it.
  F pays only as a per-token kernel, which is exactly where the 200 ms timer and the screen are in the way.
- G's row is the softer one: it assumes the done flag removes the whole END-to-seen tail and that the launch costs
  one thread 200 µs while three work. Both are checks in §4.

### 3.3 Design G in one paragraph (new; Y19 if the lead numbers it)

Y13 (launch from the lead, flush, CPU rows, event spin) is what x4s built. G changes two things. (1) The launch is
the first task of worker 0 **inside** the graph's job, and CPU rows are claimed in chunks, so workers 1 to 3 start
at once and worker 0 joins after its flush. (2) A trailing kernel of one line, enqueued behind the gemv in the same
flush (kernels run back to back, gap 1 µs, *device*, A1), stores a sequence number into a fine-grain SVM word with
a C 2.0 atomic; whichever worker runs out of CPU rows polls that word. No `clGetEventInfo`, no driver thread, no
`clFinish`. The gemv kernel, its arguments and its arithmetic are upstream's, so the lane's outputs do not change:
**bit-identical to the lane with `clFinish`**. It is HeteroLLM's fast synchronisation without the predicted sleep.

## 4. Test list (for Tom through the lead; device items wait for the owner and their own pre-registration)

| # | check | where, cost | what closes it |
|---|---|---|---|
| T1 | **Pool stall.** CPU-alone ring of the 22016 × 4096 graph with a 200 µs busy stall on the lead before each `graph_compute`, against none; print `cpu_ms` and voluntary switches per worker | host, 1 h; the same arm in any later probe, 4 s | excess ≤ 10 µs and no sleeps: C4 is not the pool's spin budget (it stays a phone wake cost). ≥ 30 µs or sleeps: the pool parks inside 200 µs; fix the spin budget (L4) |
| T2 | **Done flag on ordinary kernels.** A trailing one-line kernel storing to an SVM atomic word behind the general gemv; identity of the gemv output unchanged; on the host shim, order and visibility | host (shim or PoCL), 0.5 day; device cell 3 s | device: flag seen ≤ 20 µs after the END stamp → G's tail is gone. > 100 µs → G loses its second half; price falls to about +0.4 |
| T3 | **END to seen, directly.** Per cycle: time from the END stamp to the spin's exit, and from the launch's first call to QUEUED; p10 / p50 / p90, not means | parser plus two fields; device cell reuses x4s, 0 s extra | p50 < 60 µs → my 150 to 350 µs tail is wrong and the 400 µs is something else; I withdraw §1.3 row 2's second half |
| T4 | **x4s again, cool, in both orders** (0.60 → 0.20 first), per-cycle percentiles, a 5 s rest between arms, rings allocated once before the block | device, about 70 s | best clean ratio outside 0.76 to 0.88 → my 0.82 reading is wrong. No stretch of slow arms at the first cap step → C1 was a one-off |
| T5 | **Bus under a cap.** GPU kernel alone (a) CPUs idle, (b) 4-7 spinning on registers at full clock, (c) 4-7 reading memory; and the pair at full clocks against the pair after the caps have fallen, same GPU clock | device, 30 s plus a heat-up | (b) faster than (a) by ≥ 5 %, or the pair's sum under caps ≤ 0.85 of the cool sum at the same GPU clock → C2 holds: the bus shrinks with the CPU clock |
| T6 | **Persistent general kernel: compile and identity.** The general kernel's row arithmetic inside a claim loop (64-row chunks), per-layer form; output equal to the general kernel's bit for bit on 11008 × 4096 and 22016 × 4096 | host, 1 to 2 days | not bit-equal per row → F becomes a numerics change and needs the budget; does not build with images inside the loop under `-cl-std=CL2.0` → F as designed is closed |
| T7 | **Persistent general kernel: rate.** Device rate against the general kernel, same buffers, 840 and 676 MHz | device, 10 s | ≥ 0.90: F goes ahead of G. 0.80 to 0.90: F is G-sized, build G only. < 0.80: F closed |
| T8 | **Spin heat.** A kernel that only polls its flag for 60 s against 60 s idle: `gpu_max_mhz`, CPU caps and headroom after each | device, 2 min plus gates | caps lower after the spin than after idle → F must leave the GPU between tokens (per-token at most), and a per-token F is re-priced at the capped rows |
| T9 | **Lifetime and the screen.** One kernel alive 50, 150, 400, 1,000 ms (child process): faults, `x4 therm`, and the app's frame times while it runs | device, 20 s; kgsl source read 1 h on the host | a fault at or under 400 ms, or frames delayed by the kernel's length → per-layer only, and by §3.2 F is then not worth it over G |
| T10 | **Launch into an idle GPU.** `enq_us` against the idle gap since the last kernel (0, 0.2, 0.5, 1, 5 ms) | device, 5 s | a step of ≥ 60 µs → L5 (keep-awake) is worth a line; none → the 124 against 215 µs was the clocks |

My bands for T4 if it runs cool: best lead ratio 0.82 (0.76 to 0.88) at share 0.50 to 0.55; S10 at 0.50: 1.40
(1.30 to 1.50); END to seen p50 200 µs (120 to 350); stall arm excess 80 µs (30 to 160).

## 5. Try list (levers; generic across models; none reaches 12)

| # | lever | label | price over the lane as probed (*estimated*, width 1) | needs |
|---|---|---|---|---|
| L1 | **Launch inside the pool's job, CPU rows claimed in chunks** (G, first half) | bit-identical | about +0.4 (150 of the 400 µs) | T1 |
| L2 | **Done flag by a trailing kernel, polled by the workers** (G, second half) | bit-identical | about +0.4 to +0.5 | T2, T3 |
| G = L1 + L2 | | bit-identical | **+0.82 / +1.03 / +0.94** | both |
| L3 | **F: persistent general kernel**, per token | bit-identical per row if T6 passes; otherwise numerics change | +2.0 / +2.1 / +1.9 at rate 1.0; **over G +1.2 / +1.1 / +1.0, and only at rate ≥ 0.9** | T6 to T9; G's CPU side |
| L4 | **Pool does not park across a hand-over** (spin budget covers the launch) | bit-identical | inside L1; alone about +0.1 | T1 |
| L5 | **Keep the GPU awake between stages** (a no-op kernel or the F loop) | bit-identical | about +0.2 if T10 shows the step | T10 |
| L6 | **Self-timed split** (Bonnie r31 item 8): a stage that measures slower split than alone falls back for the rest of the run | bit-identical per path; the path taken varies | does no harm: in 10 of 18 lead arms the split was not 4 % faster than the nearer CPU-alone arm, and in 9 it was slower | none |

Order I would build: T1 and T2 on the host, then L1 + L2 (G), then T6. F's device questions (T7 to T9) decide
whether L3 is built at all.

## 6. Comparison with Bonnie r34

*Dated note 2026-11-25: [bonnie-joins-r34.md](bonnie-joins-r34.md) appeared after §0 to §5 and §7 were written; I
read it then. Her §7 says mine was not there when she finished. Nothing above was changed after reading hers.*

**Where two independent readings agree** (all *estimated* from the same *device* lines):

| item | Bonnie r34 | mine |
|---|---|---|
| 0.923 is a mix of phone states; ABBA did not cancel | four states A to D by the printed caps | the same four groups, found from the excess per overlap |
| the cell at cool clocks, as built | 0.83 (0.78 to 0.86) | 0.82 (0.76 to 0.88) |
| CPU slowdown beside the kernel, cool | 1.38 (1.28 to 1.54) | 1.36 (1.28 to 1.45) |
| the launch is serial and K_meas is that same interval | yes, within 20 µs | yes, within 21 µs |
| F at slack 60, width 1, measured head | 9.57 / 8.89 / 8.65 | 9.59 / 8.93 / 8.68 |
| the mechanism of the throttled states is not in the file | yes | yes |

**Where we differ; both positions stand.**

1. **The size of the join as built.** Bonnie: serial 207 + 13.3 µs plus an end slack of 115 µs (fit on five arms
   near balance), 335 µs in all; lane as built 7.76 / 7.04 / 6.98. Mine: 400 µs (340 to 570), read directly from
   the GPU-critical arms as cycle − kernel time (384, 400, 549); lane 7.59 / 6.82 / 6.78. The 0.2 tok/s between us
   is the END-to-seen tail: in my arithmetic it is 150 to 350 µs, in her fit 80 to 270 with centre 115. My arms are
   states B and C only; hers include cool arms where the GPU is not critical. **T3 decides** (per-cycle END to
   seen, p50). Under 100 µs she is right; over 200 µs I am.
2. **What a flag removes.** She keeps the 115 µs slack when the launch is hidden (8.17 / 7.55 / 7.44) and gives
   0.4 to "slack near zero with a shared-memory flag", because CPU-side jitter stays unless both sides claim rows.
   My G removes the tail and keeps 60 µs (8.41 / 7.85 / 7.72). Her caution is fair: my G row is the soft one
   (§3.2). **T2 decides.**
3. **Capped states.** She: C 1.57, B and D 2.2 to 2.8, with CPU-alone rates interpolated for B and C. I: C 1.40
   (1.33 to 1.58) against the last CPU arm, D 1.5 to 2.1 after separating a fixed term of 151 µs, B 2.4 to 3.2.
   The difference in C is the denominator (no CPU arm ran in B or C; either choice is a choice). In D I think part
   of what she books as coupling is a fixed cost per graph that grew three times; her form cannot tell and neither
   can mine without T1.
4. **Whether F is the next build.** She prices F against the lane as built (+2.4 to +2.7 over stock at slack 60)
   and names it as what the 12 cells need. I add that most of that is reachable by G with upstream's kernel, that
   F beats G by +0.5 only at ≥ 0.9 of the kernel rate, and that the form of F that pays (per token) is the one
   kgsl's timers and the compositor stand against (§8). She gives P 0.3 to a fast persistent kernel; my chance
   that F beats G by +0.5 on the phone is about 0.12.

**Her four points to attack:** (1) the state split: I reached the same one without it, so I do not attack it;
(2) interpolated CPU-alone rates for B and C: they move C's ratio between 0.80 and 0.85 and nothing else, both
of us should print both; (3) the slack fit on five arms: see difference 1, I think 115 µs is low for a
GPU-critical stage; (4) "launch hidden" at a quarter of the launch: I made the same assumption in G (50 of
200 µs) and it rests on claimed rows; her next question 1 is the right check and I leave it to her.

**Not in hers, in mine:** prior art (§2), F's seven problems and lifetime (§2.3, §8), G as a named design, T5 to
T10. **Not in mine, in hers:** P(12) term by term (0.002), cool in-situ K 220 µs, the hot-and-cool canary cell
(her §6; my T4 is the same cell with percentiles; one cell should serve both), thread placement.

## 7. Next questions (ranked)

1. **kgsl on the Flip 3's kernel (msm-5.4, lahaina):** the progress registers of the 200 ms fault timer, whether
   long-IB detection is on, and whether a compute command can be pre-empted by the compositor's ringbuffer. It
   decides per token against per layer, and by §3.2 that decides F. Source read; no device.
2. **The static map from CPU frequency to DDR frequency on lahaina** (device tree `cpu-llcc-ddr-bw` tables): which
   DDR step a 1,420 MHz gold cap votes for. Decides C2 on paper before T5.
3. **What MNN's and MLC's OpenCL back ends do at the CPU / GPU boundary** (event, flush, mapped flag): a second
   independent design for G's flag, and their published sync costs.
4. **Whether a done flag can also carry the output** (the kernel writes y into the same SVM page): removes H's
   13 µs and the read.
5. **A GPU-side prefill** to keep the big cores cold before decode (my r17 item 4 from the other side).

I continue with 1.

## 8. Next question 1, first pass: kgsl and a kernel that lives for a token (*external*; search and page summaries of public kernel sources, not the Flip 3's own tree)

- **Timeout.** `adreno_drawobj_timeout = 2000` ms ("Command batch timeout"); a command still running after it, with
  nothing retired on its ringbuffer, raises "gpu timeout" when long-IB detection is on. An msm `adreno.c` device
  definition sets `.long_ib_detect = 1`. Source: android.googlesource.com `kernel/msm`,
  `drivers/gpu/msm/adreno_dispatch.c` at `a80b2e0` (an older branch than msm-5.4; read 2026-11-25 through a page
  summary).
- **Soft fault.** `_fault_timer_interval = 200` ms: every 200 ms while work is in flight the driver compares a list
  of registers (the list includes `CP_IB1_BASE` by a search summary; I did not read the full list) and the retired
  timestamp; no change in any of them sets a soft fault. A kernel that spins on a flag retires nothing and, as far
  as I can tell, moves no command-stream pointer. x4e's 188 ms kernel passed (*device*), which is under one
  interval. Whether soft-fault detection is enabled on the Flip 3 is not known to me.
- **Pre-emption.** The downstream driver's default is `preempt_level = 1`, and the upstream description of level 1
  is "preemption at either the draw call level or the bin boundary level". One OpenCL kernel launch is one
  dispatch. Reading: the compositor's ringbuffer cannot take the GPU inside a kernel; it waits for the kernel to
  end. This agrees with HeteroLLM's game at 0 fps.

What it means for F (my reading, no verdict):

- A **per-token** kernel holds the screen for a token (85 to 145 ms at 7 to 12 tok/s) and sits near the 200 ms
  soft-fault compare on slow tokens. A **per-layer** kernel (3 to 5 ms) is safe on both counts.
- By §3.2 a per-layer F is worth about +0.2 over G. So the form of F that pays is the form these sources warn
  against. **My P that a per-token F passes T8 and T9 on the phone: 0.25.** With P(rate ≥ 0.9) about 0.5, the
  chance that F beats G by +0.5 is about 0.12.
- This does not close F: T9 is cheap and the sources are not the phone's tree. It does set the order: **G first;
  F only after T9**, and T9 before T6's two days of kernel work.

Ranked next questions after this pass: (1) the lahaina static map from CPU clock to DDR clock (C2 on paper);
(2) MNN's and MLC's CPU / GPU boundary; (3) the phone's own kgsl tree (Samsung's open-source release for the
Flip 3) for `ft` settings and `preempt_level`; (4) output in the flag's SVM page; (5) a GPU-side prefill.
