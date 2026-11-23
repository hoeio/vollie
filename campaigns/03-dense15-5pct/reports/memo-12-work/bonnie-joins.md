# c3 memo-12 work (Bonnie): what a GPU join costs, and lane designs that cut or hide it

- **Date:** 2026-11-23. **Author:** bonnie, `claude-fable-5-2`. Part of joint round memo-12 with Ebony (her part: new
  ideas; she reviews this pricing). Not committed. No verdict here; the lead decides.
- **Labels.** *device* = printed by the phone in R74 (`device/r74-c3probe-r34-2026-11-23T1743.json`). *derived* =
  arithmetic on two *device* fields of the same line or cell (stated each time). *estimated* = model output.
  *external* = a public source, not measured by us. *recalled* = my memory, no source found today.
- **Script:** [bonnie_joins.py](bonnie_joins.py) (`python3 -I bonnie_joins.py`; loads `memo-10-work/bonnie_tree.py`
  by path, unchanged). Its self-check reproduces R74-result §4 (P_G 7.86 / 7.39 / 7.30, stack 9.55 / 9.50 / 9.26,
  zero-dispatch 10.34 / 9.53 / 9.22 and 12.23 / 11.87 / 11.44) and the x3 split time (0.865 ms).

## 1. Answer first

1. **No design reaches 12 on all three. No design reaches 12 on Llama-2 or DeepSeek at any join cost:** their centre
   stack at zero join cost is 11.87 and 11.44 (*estimated*, R74-result §4). A join design can only approach that
   ceiling.
2. **Yi reaches 12.00 only if a join costs ≤ 33 µs** on the GPU side, at 4 joins per layer (128 per token), with
   every unbuilt lever of the stack at its centre (*estimated*). The measured join is 452 µs (*device*): 14 times
   too much. Only the most exotic design (F, a persistent kernel) has a centre there: 12.02, P about 0.08.
3. **R74's price charged the join in series; the x3 split cell says the CPU keeps working through it.** With the
   GPU's row share re-balanced, a join of K µs costs about 0.52 × K of wall time (*derived* from the split cell, one
   cell). That is a re-pricing, not a new lever: centre stack 10.01 / 9.72 / 9.47 at the join as measured, against
   9.55 / 9.50 / 9.26 as R74 wrote it (*estimated*).
4. **Realistic designs (poll, fast-core dispatch, pre-enqueue) land at about 10.4 to 10.6 / 10.2 to 10.4 / 9.9 to
   10.1** (*estimated* centres). The step to about 11.5 / 11.2 / 10.8 needs the 199 µs between submit and start to
   be removable, which R74 cannot tell.
5. **New from the raw lines: every kernel launch costs 200 to 320 µs of non-device time even with no wait**
   (*derived*, §2.2). The GPU is idle for that time. So "enqueue everything, wait once" does not make launches free
   on this driver as the probe drives it.
6. **Memory moves with the share.** The balanced lane puts 1.6 to 1.9 GB in GPU buffers (*estimated*); R74's
   residency arm held 0.85 GB (*device*). Every design above E is outside what x3r covered.

## 2. Decomposing the 452 µs (R74 fields only)

### 2.1 One isolated round trip (`x3 rt`: launch + `clFinish` per call, caller pinned to cpus 0-3)

The probe prints `queued_to_start` (START − QUEUED) and `submit_to_start` (START − SUBMIT) from 100 calls on the
profiling queue, and `wall_us` from a separate loop of 715 to 1,256 calls on the plain queue.

| rows | wall | dev (START→END) | ovh = wall − dev | QUEUED→SUBMIT (*derived*: q2s − s2s) | SUBMIT→START | rest (*derived*: wall − q2s − dev) |
|---|---|---|---|---|---|---|
| 512 | 477.1 | 93.6 | 383.4 | 143.1 (37 %) | 183.7 (48 %) | 56.7 (15 %) |
| 1024 | 651.8 | 120.8 | 530.9 | 164.5 | 194.4 | 172.1 (see note) |
| **2816** | 691.4 | 239.4 | **452.0** | **195.9 (43 %)** | **199.0 (44 %)** | **57.1 (13 %)** |
| 4096 | 838.3 | 357.4 | 480.9 | 208.1 (43 %) | 196.7 (41 %) | 76.1 (16 %) |

All µs, *device* unless marked. Note on 1024 rows: p50 (691.7) is above the mean (651.8), so that cell is bimodal
and its "rest" is not trusted.

What each part is:
- **QUEUED→SUBMIT, 143 to 208 µs: the host side.** The command is enqueued but the GPU does not have it yet. It is
  driver work in user space and the submit ioctl, on an A55.
- **SUBMIT→START, 184 to 199 µs: kernel and GPU side.** Flat across sizes. It holds the kgsl dispatch path and any
  GPU idle exit.
- **Rest, about 57 µs:** the 15 `SetKernelArg` calls before QUEUED, plus END → `clFinish` returning (completion
  interrupt, waking the caller).

**Not separable from R74's fields:**
- inside SUBMIT→START: GPU idle exit against plain dispatch latency against a GPU clock ramp (`gpu_mhz` is
  unreadable, EACCES);
- inside the rest: pre-enqueue host time against the completion notice. So the completion notice is **at most**
  57 µs; R74 does not give it alone;
- the caller's CPU clock in the rt cell (the cell prints no `mhz`). In the neighbouring `x3 alone` cells the little
  cluster read 691 to 1,094 MHz of 1,804 (*device*);
- whether the profiling queue changes the timing (wall and the stamps come from different loops and queues).

### 2.2 The launch itself costs time, with no wait (not in R74-result §4)

| cell (*device*) | wall per launch | dev per launch | gap (*derived*) | feeder CPU per launch (*derived*: `feeder` share × wall) | `gpu_busy` | dev ÷ wall |
|---|---|---|---|---|---|---|
| alone 512×4096 cold (8 launches, 1 finish) | 297.0 | 97.6 | 199.4 | 119 | 34 % | 33 % |
| alone 4096×4096 cold | 626.5 | 365.0 | 261.5 | 157 | 60 % | 58 % |
| alone 4096×11008 cold | 1,290.0 | 970.4 | 319.6 | 184 | 76 % | 75 % |
| alone 11008×4096 cold | 1,207.0 | 895.6 | 311.4 | 179 | 74 % | 74 % |
| layer sync1 (7 launches, 1 finish) Yi / Llama-2 / DeepSeek | 815.6 / 851.6 / 861.7 | 522.6 / 599.7 / 600.9 | 293.0 / 251.9 / 260.8 | 178 / 141 / 164 | 65 / 70 / 70 % | |
| layer sync7 (finish after each) | 1,249 / 1,305 / 1,111 | same | 727 / 705 / 510 | 509 / 482 / 270 | 56 / 66 / 56 % | |
| tog, gpu arm (same work as alone 11008×4096 cold) | 1,031 / 1,008 (*derived* from 24.60 / 25.16 GB/s) | not measured there | 135 / 112 against 895.6 | 117 / 94 | 86 / 89 % | |

Reading:
- One finish per 8 launches would add 452 ÷ 8 = 56 µs per launch. The gap is 200 to 320 µs. So **145 to 265 µs per
  launch is spent with no wait at all** (*derived*).
- `gpu_busy` equals dev ÷ wall in every row. The GPU is idle during the gap: the kernels of a batch are **not** run
  back to back.
- The feeder thread's own CPU time per launch (94 to 184 µs) is the same size as QUEUED→SUBMIT (143 to 208 µs).
  Two independent fields agree: **the enqueue-to-submit part is host CPU work in the driver on a little core, in
  series with the GPU.**
- Per token, GPU alone at one finish per layer loses 65.6 / 56.4 / 54.8 ms to this (*device*: `body_ms_sync1` −
  `body_ms_dev`).
- State dependence is large. Llama-2 and DeepSeek have the same seven shapes, yet sync7 costs 705 against 510 µs per
  dispatch in consecutive cells. The same 11008×4096 batch costs 1,207 µs in x3g and 1,008 to 1,031 µs in x3c. The visible
  difference is the little-cluster clock range (691–1,094 against 806–1,804 MHz); it is not a controlled comparison.
  **Any band on the join is at least ±100 µs from state alone.**

### 2.3 The join inside a real split (`x3 split`, feeder spinning on 0-3, CPU rows on 4-7)

| share | gpu rows | `gpu_ms` (launch + finish) | dev for those rows, alone (*derived*, interpolated from x3 rt) | join K in situ (*derived*) | `cpu_ms` | `wait_ms` | ratio |
|---|---|---|---|---|---|---|---|
| 0.25 | 2,816 | 0.719 / 0.796 | 0.239 | 480 / 557 µs | 0.820 / 0.823 | 0.035 / 0.068 | 0.912 |
| 0.35 | 3,840 | 0.839 / 0.832 | 0.334 | 505 / 498 µs | 0.748 / 0.783 | 0.136 / 0.119 | 0.951 |
| 0.50 | 5,504 | 1.010 / 0.857 | 0.487 | 523 / 370 µs | 0.623 / 0.664 | 0.409 / 0.211 | 1.020 |

- The join costs the same inside a split as alone: 370 to 557 µs, centre about 500 (*derived*). A spinning feeder
  with the little cluster at 1,094 to 1,804 MHz did **not** make it cheaper. That is evidence against "the little
  core's clock explains the host part".
- **The CPU does not wait for the join; it computes its rows through it.** At share 0.25 the CPU side takes
  0.82 ms, the GPU side 0.72 to 0.80 ms, and the main thread waits only 35 to 68 µs.
- So the wall cost of a join is not K. With the share tuned so both sides end together:

  **t(stage) = lane × A + κ × K**, A = the stage's CPU-alone time, lane = 0.621, **κ = 0.52**

  κ comes from the `x3 tog` rates (CPU 25.64 alone, 19.77 beside the GPU; GPU 21.53): κ = ρc ÷ (ρc + ρg) = 0.521.
  Check against the split cell: 0.621 × 0.948 + 0.521 × 0.530 = 0.865 ms; measured 0.865 (*device*), with K
  inside the in-situ range. One cell, one matrix shape. κ band 0.38 (CPU at full rate during the join) to 0.58
  (K at the low end of its range).
- R74-result §4 used κ = 1 (join in series after a full lane). That overstates the cost of each join by about
  two, and understates the number of joins (§3).
- A stage is worth splitting only if (1 − lane) × A > κ × K, which is **A > 1.37 × K**.
- The split arm's worker threads read 0.90 to 0.93 of a core (0.96 to 0.99 in the CPU-only arm) with about one
  voluntary switch per matrix: the pool sleeps while the main thread waits. A real lane must keep that wait at
  zero by its share, not by spinning.

## 3. Joins forced by data dependence

A decoder layer is a chain of four matmul stages. The output of each is the input of the next, so nothing can be
deferred across a stage:

| stage | input | Yi A (ms/layer) | Llama-2 | DeepSeek | break-even K (µs, κ 0.52) |
|---|---|---|---|---|---|
| q, k, v (same input, parallel) | normed residual | 0.457 | 1.093 | 1.084 | 332 / 795 / 789 |
| o | attention output (attention itself on the CPU) | 0.365 | 0.364 | 0.361 | 266 / 265 / 263 |
| gate, up (same input, parallel) | normed residual after attention | 1.964 | 1.958 | 1.943 | 1,429 / 1,424 / 1,413 |
| down | silu(gate) × up | 0.982 | 0.979 | 0.971 | 714 / 712 / 707 |

A = body ms × the stage's share of the layer's weights ÷ n_layer (*estimated*; the body is not purely matmul).

- **Every stage with GPU rows ends in one forced join. A fully split lane has 4 joins per layer: 128 / 128 / 120
  per token, not 64 / 64 / 60.**
- A join is removed only by giving the whole stage to the CPU, which loses 0.379 × A of gain there.
- **Whole matrices do not hide a join across stages.** While the GPU computes gate and up, the CPU has nothing it
  is allowed to start: o is finished, down needs the result. The CPU can work "on other matrices meanwhile" only
  inside a stage: gate on the GPU beside up on the CPU, q on the GPU beside k and v on the CPU.
- One launch per stage is enough if same-input matrices are stored as one GPU buffer ([q; k; v] and [gate; up] are
  just taller matrices for the general kernel, which matched the specialised one: G5 0.997, *device*). Otherwise
  each extra launch adds the 145 to 265 µs of §2.2 on the GPU side.

## 4. Designs and prices

**The calls.** For each design, `bonnie_joins.body(subject, W, K, mode)` sums the four stage times per layer and
picks, per stage, the cheaper of CPU-alone and split. Then:
- P_G = 950 ÷ (body′ + 8 + head) (K0 with the lane alone, as R74-result §4);
- centre stack = `bonnie_tree.best(s, 2, C=40, g=24.82, disp=D2, hs=0.5, c5=0.93, O=5.2)`, with D2 = body′(W 2)
  − body × kB × c5 × lane(W 2). So `disp` holds the join cost **and** the gain lost on stages left to the CPU.
  `bonnie_tree.py` is used as committed.
- "ms per token" below = body′ − body × lane at W 1: what the design costs against a free lane.

K is the GPU-side overhead per join. Bands: worst = K high with κ 0.58, best = K low with κ 0.38. All *estimated*.

| design | K µs (band) | joins/layer Yi / L2 / DS | ms per token Yi / L2 / DS (best to worst) | P_G | centre stack (worst to best) | GPU buffers GB |
|---|---|---|---|---|---|---|
| R74 as priced (series, κ 1) | 452 | 2 / 2 / 2 | 28.9 / 28.9 / 27.1 | 7.86 / 7.39 / 7.30 | 9.55 / 9.50 / 9.26 | not stated |
| **E** shares re-balanced, join as measured | 500 (452–560) | 2 / 3 / 3 | 26.7 (20.9–30.8) / 29.4 (20.9–35.6) / 27.6 (19.6–33.3) | 8.01 / 7.36 / 7.27 | **10.01** (9.59–10.31) / **9.72** (9.17–10.12) / **9.47** (8.96–9.83) | 0.94 / 1.24 / 1.16 |
| **B** E + poll, no `clFinish` | 470 (400–560) | 2 / 3 / 3 | 25.7 / 27.9 / 26.2 | 8.08 / 7.44 / 7.35 | 10.08 (9.59–10.42) / 9.82 (9.17–10.26) / 9.55 (8.96–9.96) | 0.96 / 1.27 / 1.19 |
| **B+** B + dispatch from a fast core, or a recorded queue | 350 (250–500) | 2 / 3 / 3 | 21.7 (12.2–28.5) / 21.9 (12.2–32.3) / 20.5 (11.4–30.2) | 8.37 / 7.81 / 7.69 | 10.38 (9.71–10.94) / 10.20 (9.34–10.74) / 9.91 (9.11–10.41) | 1.04 / 1.39 / 1.30 |
| **A** B+ + pre-enqueue behind a gate | 290 (205–500) | 3 / 3 / 3 | 19.0 (10.0–28.5) / 18.9 (10.0–32.3) / 17.7 (9.3–30.2) | 8.57 / 8.01 / 7.86 | 10.55 (9.71–11.15) / 10.40 (9.34–10.93) / 10.09 (9.11–10.58) | 1.18 / 1.45 / 1.36 |
| **C** A + GPU kept out of idle | 120 (35–500) | 4 / 4 / 4 | 8.0 (1.7–28.5) / 8.0 (1.7–32.3) / 7.5 (1.6–30.2) | 9.51 / 8.82 / 8.59 | 11.45 (9.71–**12.03**) / 11.19 (9.34–11.70) / 10.82 (9.11–11.28) | 1.46 / 1.74 / 1.63 |
| **F** persistent kernel, shared-memory mailbox | 30 (5–100) | 4 / 4 / 4 | 2.0 (0.2–7.4) / 2.0 (0.2–7.4) / 1.9 (0.2–7.0) | 10.12 / 9.34 / 9.05 | **12.02** (11.41–12.20) / 11.69 (11.16–11.85) / 11.27 (10.78–11.41) | 1.58 / 1.86 / 1.74 |
| **D1** whole matrices only (gate; q on L2, DS), K as E | 500 | 1 / 2 / 2 | 36.3 / 39.2 / 36.6 | 7.41 / 6.84 / 6.80 | 8.97 / 8.73 / 8.54 | 0.81 / 1.11 / 1.04 |
| **D2** whole matrices + down rows, K as A | 290 | 2 / 3 / 3 | 22.5 / 25.5 / 23.8 | 8.30 / 7.59 / 7.49 | 9.77 / 9.61 / 9.36 | 1.14 / 1.44 / 1.35 |
| **D3** the same, K as C | 120 | 2 / 3 / 3 | 14.2 / 17.2 / 16.0 | 8.95 / 8.13 / 7.98 | 10.41 / 10.17 / 9.88 | 1.19 / 1.50 / 1.40 |
| Z zero join cost (ceiling) | 0 | 4 / 4 / 4 | 0 | 10.34 / 9.53 / 9.22 | 12.23 / 11.87 / 11.44 | 1.62 / 1.90 / 1.78 |

Stack against K, row shares free (P_G / stack, *estimated*):

| K µs | Yi | Llama-2 | DeepSeek |
|---|---|---|---|
| 0 | 10.34 / 12.23 | 9.53 / 11.87 | 9.22 / 11.44 |
| 20 | 10.19 / 12.09 | 9.41 / 11.75 | 9.11 / 11.33 |
| 50 | 9.98 / 11.89 | 9.22 / 11.58 | 8.95 / 11.17 |
| 100 | 9.64 / 11.57 | 8.93 / 11.30 | 8.69 / 10.92 |
| 200 | 9.03 / 10.98 | 8.41 / 10.78 | 8.22 / 10.44 |
| 300 | 8.53 / 10.51 | 7.98 / 10.37 | 7.83 / 10.06 |
| 452 | 8.12 / 10.13 | 7.50 / 9.87 | 7.40 / 9.60 |
| 560 | 7.88 / 9.87 | 7.19 / 9.55 | 7.12 / 9.30 |

Stack = 12.00: **Yi at K ≤ 33 µs; Llama-2 and DeepSeek at no K.**

### 4.1 Each design: what it removes, the evidence, P

Judgement P = the design delivers its centre K on this phone inside the rules.

**E. Re-balanced shares (task item e).** Nothing is built beyond the lane itself: a static row share per stage,
chosen so that the GPU side (K + its rows) ends with the CPU side. Evidence: §2.3, *device*, one shape.
- K 500 (452 to 560): the in-situ values of §2.3.
- P 0.6. The risk is κ at other shapes and at batch 2, and the per-launch gap if a stage needs two launches.

**B. Poll instead of `clFinish` (task item b).** `clFlush`, then spin on `clGetEventInfo` or on the output buffer.
- Removes at most the "rest" of §2.1: ≤ 57 µs, and that figure also holds the pre-enqueue arg setting. Centre
  −30 µs.
- Side evidence: sync7 burns 270 to 509 µs of feeder CPU per dispatch against 141 to 178 µs at sync1 (*derived*), so
  `clFinish` is not a cheap sleep on this driver. Whether a poll avoids that work is not known.
- Polling the output buffer for a sentinel is not sound by itself: work-group order is not defined, and OpenCL
  gives no visibility guarantee before the event completes.
- P 0.7 for −30 µs. **Polling alone is worth about 0.07 to 0.10 tok/s on the stack.** It is not the lever.

**B+. Dispatch from a fast core, or a recorded queue.** Targets QUEUED→SUBMIT (196 µs at 2,816 rows).
- Fast core: an A78 at 2,419 MHz against an A55 at 691 to 1,094 MHz is 2.2 to 3.5 times the clock (*device*
  clocks) and about twice the work per cycle (*recalled*). Priced: 196 → 80 µs (40 to 196). It costs one of the
  four workers that time, about 10 to 25 µs of stage time, not charged above.
- Against it: §2.3. A spinning feeder with the little cluster at up to 1,804 MHz still showed K ≈ 500.
- Recorded queue: `cl_qcom_recordable_queues` records a sequence of enqueues once and replays it with one call.
  *External:* Qualcomm's Adreno SDK FAQ says the OpenCL SDK 1.3 added a document and an example for it; one
  community listing of an Adreno device's extensions includes it (device not named). I did not find the spec text
  (the Khronos registry's qcom directory does not carry it) or any overhead figure. **Whether the phone's driver
  (E031.38, dated 05/11/25) exposes it is not known: kit r34 does not print `CL_DEVICE_EXTENSIONS`.**
- `cl_qcom_perf_hint`: *external* (Khronos registry) says HIGH is the default for an OpenCL context. So setting it
  should change nothing; printing it is free.
- P 0.35.

**A. Pre-enqueue (task item a).** Enqueue a stage's kernel before its input exists, gated by a user event (or
enqueue and flush early from a helper thread), so that the 196 µs of host work overlaps the CPU's previous stage.
- "All of a token's kernels at once" is not possible in a split lane: each kernel's input is produced by the stage
  before it, half of it on the CPU. What can be moved early is the host work, not the start.
- Forced joins stay at one per split stage (§3): 2 to 4 per layer.
- Unknown: whether the driver builds and submits the command at enqueue, or does the same work when the gate
  opens. In the second case A = B. §2.2 leans that way: 8 enqueues before one finish did not run back to back.
- K 290 (205 to 500). P 0.3.
- *External, for contrast:* upstream's OpenCL backend never joins inside a graph. `graph_compute` only enqueues;
  `synchronize` enqueues a barrier and waits once (read from `ggml-opencl.cpp` at upstream commit `1f8baae`,
  2026-11-23). Upstream has no clFinish-per-node path outside profiling, no perf hint and no recorded queue. A CPU
  + GPU lane with joins inside a layer is not something upstream has tuned.

**C. Keep the GPU out of idle (task item c).** Targets SUBMIT→START (199 µs).
- What the 199 µs is, R74 cannot say (§2.1). *External:* kgsl's idle timer defaults to 80 (a Fairphone 5 kernel
  Kconfig, `QCOM_KGSL_IDLE_TIMEOUT`; unit read as ms). Our gaps are under 1 ms, so this is **not** a slumber exit.
  It is either a short power-collapse exit handled by the GMU (*recalled*, no source found today) or plain dispatch
  latency (kernel dispatcher thread, GMU message), which user space cannot remove.
- A "busy kernel" cannot be the tool as the driver behaves in §2.2: a filler on the same in-order queue delays the
  real kernel, a filler on a second queue shares the GPU, and short fillers each pay the launch cost.
- So C is priced as "the 199 µs turns out removable": K 120 (35 to 500). P 0.15.

**F. Persistent kernel with a shared-memory mailbox (added; exotic).** One launch per token (or per layer). The
kernel loops: spin on a flag word in a zero-copy buffer, compute the stage's rows, write the result and a done
flag; the CPU does the mirror image. No enqueue, no submit, no completion interrupt inside the token.
- K 30 µs (5 to 100) if it works: cache-line hand-off plus the GPU's poll period.
- Against it: reading a buffer the host writes while the kernel runs is undefined in OpenCL; the GPU's own cache may
  never show the update; a kernel that runs for a whole token may be killed by the driver's watchdog (*external*:
  an older kgsl tree shows a 2,000 ms command timeout) and holds the GPU against the screen; a spinning GPU adds
  heat, and heat is what caps the CPU clocks (R72).
- It is the only design whose centre reaches 12 on Yi (12.02), and it leaves Llama-2 at 11.69 and DeepSeek at 11.27.
- P 0.08. Not host-testable.

**D. Whole matrices (task item d).** gate on the GPU beside up on the CPU; on Llama-2 and DeepSeek also q beside
k and v. D2 and D3 add row-split down.
- Join count: 1 to 3 per layer. The join is still forced (§3); only its overlap changes.
- Load balance: gate against up is 0.50 where the balanced share is 0.52 at batch 1, so the pin costs little there
  (0.648 × A against 0.621 × A, and it hides a K of up to about 100 µs). q of q, k, v is 0.33: the CPU side binds.
  At batch 2 the GPU slows by 1.45 (memo-10's rule), the balanced share falls to about 0.44, and a pinned 0.50 is
  GPU-bound.
- **Priced 0.3 to 1.0 tok/s below free row shares at every K.** Its real advantages are not speed: the CPU never
  touches gate, and the GPU copy replaces a CPU copy one for one.
- P 0.5 for D1's centre (it needs nothing new), as for E.

## 5. What a test would have to show

**Host (laptop).** The laptop cannot measure an Adreno join. It can check the model that turns K into ms:
1. **Join count and stage sizes.** A CPU-only stand-in for the lane in the campaign's llama.cpp: a helper thread
   takes a fixed row share of a stage, sleeps K, then computes; the main pool computes the rest. Print joins per
   token for the three subjects (expect 128 / 128 / 120 fully split) and the measured A per stage against §3's
   weight-share estimate. Pass: A within ±15 % of §3, else this memo's stage table is redone.
2. **κ.** In the same stand-in, sweep K ∈ {0, 100, 300, 500} µs and the share; fit t = lane × A + κ × K. This
   checks the scheduling arithmetic and the pool's sleep per stage (§2.3, last bullet), not the phone's κ.
3. **Row-split identity.** CPU rows computed as two ranges give per-chunk PPL equal to stock at `-b 512`, `-ub 32`
   and `-ub 1` (rule 4). The GPU rows are a numerics change and belong to Tom's fp16-emulation check (R74 §7 item 1).

**Device cells (only with a pre-registration and the owner's permission; device rounds are held).** Each cell
answers one of the "not separable" items of §2:
1. `CL_DEVICE_EXTENSIONS` printed. Decides whether B+ has a recorded queue at all.
2. **Per-kernel stamps for a batch:** 8 launches, then flush; print QUEUED, SUBMIT, START, END of each against the
   first QUEUED. If START(j) − END(j−1) is under 20 µs for j ≥ 2, the GPU runs back to back and the 199 µs is a
   first-dispatch cost (C is alive). If each kernel shows its own 150 to 200 µs, it is dispatch latency (C is dead).
3. **`clFlush` after every enqueue** against one finish per 8: does the per-launch gap (200 to 320 µs) fall to
   under 60 µs? This is A's premise.
4. **The rt cell from cpu 4 and cpu 7 at full clock**, and from 0-3 with the clock printed. B+ needs
   QUEUED→SUBMIT ≤ 100 µs.
5. **Poll against finish:** the same rt cell with `clFlush` + `clGetEventInfo` spin. B needs ovh down by ≥ 30 µs.
6. **User-event gate:** enqueue behind a user event, set it, stamp set → START. A needs ≤ 250 µs where the plain
   path shows 395.
7. **The split cell again** with a fused [gate; up] buffer, share swept 0.20 to 0.50 in steps of 0.05, at batch 1
   and with two input columns. This gives κ and the in-situ K at the lane's real shape.
8. **Mailbox kernel (F):** does a kernel see a host write to a mapped zero-copy buffer while it runs, and how long
   after? Any answer above 100 µs, or "never", closes F.
9. **Residency at the design's GPU share:** x3r held 0.85 GB; A needs 1.2 to 1.5 GB, C and F 1.5 to 1.9 GB.

**Lines the cells must meet for the stack to move (rows free, *estimated*):**

| in-situ K per join | centre stack Yi / Llama-2 / DeepSeek | what it means |
|---|---|---|
| ≤ 300 µs | ≥ 10.51 / 10.37 / 10.06 | +0.5 over E on all three |
| ≤ 100 µs | ≥ 11.57 / 11.30 / 10.92 | within 1.1 of 12 on all three |
| ≤ 33 µs | ≥ 12.00 / 11.68 / 11.26 | Yi at 12; the other two not |

## 6. Limits of this pricing

- κ is fitted to one split cell (one matrix shape, share grid 0.25 / 0.35 / 0.50) in a round without the cool start (R74-result §2), with
  the A78 policies capped at 1,785 to 2,150 of 2,419 MHz in that block (*device*, `max_mhz`).
- lane 0.621 comes from GPU wall rates that already hold one launch gap per 25 MB. Charging K on top double counts
  a little: at most about 5 ms per token in the lane's favour (*estimated*; it needs the GPU's device rate beside
  a busy CPU, which R74 did not print).
- At batch 2 the GPU needs a two-column kernel or two launches per stage; the second launch would add §2.2's gap.
  The stack rows assume one launch.
- Stage times are weight shares of the R72 body; attention, norms and activations are inside the body but not in
  the table.
- The centre stack still needs every non-GPU lever at its centre (trained head at width 2, C7, C5, O 5.2); none is
  built. The stack is a price, not a plan (prereg R74 §5).
- State alone moves a join by ±100 µs (§2.2).

## 7. Lists for Tom (through the lead; Ebony's part to be merged)

**Test list**
1. Host stand-in lane: joins per token, A per stage, κ sweep (§5 host 1 and 2).
2. Row-split identity on the CPU side at the three batch settings (§5 host 3).
3. Read the probe's `launch()`: can the 15 `SetKernelArg` calls be cut to the 3 that change (weights image, scale
   buffer, row count)? Host-checkable by reading; its gain is inside the 57 µs "rest".
4. Device cells 1 to 9 of §5, as one probe block, if the owner allows a round. Cells 1 to 3 decide most of the band.

**Try list (after its test)**
1. E: static per-stage shares from the κ fit, fused same-input buffers, one launch per stage.
2. B+ and A together if cells 3, 4 and 6 pass: helper thread on a fast core, early enqueue behind a gate.
3. F only if cell 8 shows a hand-off under 100 µs.

## 8. For Ebony's review

1. κ = 0.52 from one cell: is the t = lane × A + κ × K form right, or does the CPU side lose more than ρc when the
   feeder and the driver are active?
2. Four forced joins per layer (§3). Is there a schedule I missed that overlaps two stages?
3. B+: the split cell argues against the clock explanation. Is 350 µs too kind?
4. F: any published case of a GPU kernel polling host-written memory on a mobile driver?
5. The 199 µs: any source that says what an Adreno 6xx does between submit and start after under 1 ms of idle?

## 9. Sources (read 2026-11-23)

Repository: [R74-result.md](../R74-result.md); the R74 JSON (blocks x3g, x3c); prereg
[R74 §5](../../prereg/R74-x3-gpu-probe-device.md); `patches/llama.cpp-c3/0005-x3-gpu-probe.patch` (`roundtrip`,
`layer`, `batch`, `split`, `run_arm`: what each field measures); [memo-09-gap.md](../memo-09-gap.md) §4, §5.2,
§10.7; `memo-10-work/bonnie_tree.py`. Campaign 1 has one OpenCL line (`docs/negative-results.md`: 0.7 to 2.9 tok/s,
always below the CPU, *device*) and no join or dispatch figure.

External (labelled where used):
- Khronos registry, `cl_qcom_perf_hint`: https://registry.khronos.org/OpenCL/extensions/qcom/cl_qcom_perf_hint.html
- Khronos registry, qcom extension directory (no recordable-queues text): https://registry.khronos.org/OpenCL/extensions/qcom/
- Qualcomm Adreno GPU SDK FAQ (search summary; mentions the recordable-queues document in OpenCL SDK 1.3):
  https://developer.qualcomm.com/software/adreno-gpu-sdk/faq
- An Adreno extension listing (device not named): https://gist.github.com/woachk/9a0c542a946e4d2838564f104e0833d7
- llama.cpp `ggml/src/ggml-opencl/ggml-opencl.cpp` at `1f8baae` (read first-hand):
  https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-opencl/ggml-opencl.cpp
- kgsl idle timeout Kconfig (search summary):
  https://gitlab.e.foundation/e/devices/android_kernel_fairphone_fp5/-/commit/e915fc5948702ab461a14d3e0974f8e72b40a5cf
- kgsl dispatcher, older tree (search summary; inflight limit 15, command timeout 2,000 ms):
  https://android.googlesource.com/kernel/msm/+/a80b2e0035e75c0460b54e2217ca0104631d547d/drivers/gpu/msm/adreno_dispatch.c
- Qualcomm's OpenCL optimisation guide 80-NB295-11: the link now redirects to a login page; **not read**.

Searched and not found: a measured Adreno enqueue or round-trip cost in µs; what an Adreno 6xx does between submit
and start; `cl_qcom_recordable_queues` support by GPU model.
