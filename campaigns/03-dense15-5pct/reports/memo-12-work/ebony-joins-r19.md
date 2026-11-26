# c3 memo-12 work (Ebony, round 19): the lahaina bus votes on paper, the Flip 3's kgsl settings, and what MNN and MLC do at the boundary

Date: 2026-11-25. Author: Ebony (`claude-fable-5-2`). Not committed; the lead commits. No verdict here.
Follows [ebony-joins-r18.md](ebony-joins-r18.md) (its ranked items 1 to 3). Bonnie's [r34](bonnie-joins-r34.md) was
read before writing; §6 compares.

Labels: *device* = printed by the phone (R75 session 2); *estimated* = arithmetic or model output; *external* =
a public source, read as stated; *source* = this repository's code. Triples are Yi / Llama-2 / DeepSeek.
Script: [ebony_bus_r19.py](ebony_bus_r19.py) (`python3 -I -B ebony_bus_r19.py <repo root>`): the tables of §1 as
data with the kernel's lookup rule and a self-check, and the prices of §4 through `ebony_split_r18.py` unchanged
(it asserts that r18's three rows reproduce). Nothing ran on a phone.

**What was read (all *external*, files fetched and read directly on 2026-11-25, not search summaries, unless a
line says otherwise):**

- A public mirror of Samsung's open-source kernel for the Flip 3, variant SM-F7110 (Android 15 release):
  `github.com/xiliuya/android_kernel_samsung_b2q` at commit `31574e5bd1` (2026-11-07). Files: `lahaina.dtsi`,
  `lahaina-v2.dtsi`, `lahaina-gpu.dtsi`, `lahaina-v2-gpu.dtsi`, `lahaina-thermal.dtsi`,
  `lahaina-thermal-overlay.dtsi`, Samsung's overlay `samsung/b2/b2q/b2q_chn_hkx_w00_r12.dts`, the defconfig,
  `drivers/devfreq/{governor_memlat,governor_bw_hwmon,devfreq_icc}.c`, `drivers/interconnect/qcom/{icc-rpmh,bcm-voter}.c`,
  `drivers/gpu/msm/{adreno.c,adreno.h,adreno_dispatch.c,adreno_a6xx.c,adreno_a6xx_preempt.c,adreno_a6xx_gmu.c,adreno-gpulist.h,kgsl_bus.c,governor_gpubw_mon.c}`,
  `drivers/cpufreq/qcom-cpufreq-hw.c`, `drivers/thermal/qcom/ddr_cdev.c`.
  **Limits:** it is a third party's mirror of the SM-F7110 release, not the owner's phone's own build; I do not
  know the owner's variant or firmware. The SoC, the base device tree and the driver are the same family.
- Qualcomm's post-boot script for lahaina as carried in a Xiaomi SM8350 tree (`LineageOS/android_device_xiaomi_sm8350-common`,
  branch `lineage-21`, `rootdir/bin/init.kernel.post_boot-lahaina.sh`). **Samsung's own vendor script was not
  found and not read.** Every statement below that depends on it says so.
- MNN `master` at `28b8cfac4c` (2026-11-24): `source/backend/opencl/core/{OpenCLBackend.cpp,OpenCLRunningUtils.cpp,runtime/OpenCLRuntime.cpp}`.
- TVM tag `v0.18.0`: `src/runtime/opencl/{opencl_device_api.cc,opencl_common.h,opencl_module.cc}`. TVM `main`
  (`1091f2fc97`, 2026-11-24; latest release v0.27.0, 2026-09-25) has no OpenCL runtime file in its tree listing; I
  did not find where it went, so the MLC statement is about the older tag.

## 0. Answer first

1. **"DDR follows the capped CPU clock" is closed for the split arms, on paper and by the phone's own numbers.**
   The DDR clock is the largest of several votes. One of them is the GPU's own, a fixed table by GPU power level:
   3,196 MHz at 840, 778 and 738 MHz, 2,736 at 676, at most 1,555 at 491 (*external*). So in the bad stretch
   (GPU at 840) the bus was voted to its top step whatever the CPU caps were, and at the 1,420 cap with the GPU
   at 676 it was voted to 2,736 or 3,196. The sums the phone drew there (28 to 34 and 33 to 36 GB/s, *estimated*
   from *device* lines by Bonnie) are far under either ceiling (43.8 and 51.1 GB/s theoretical). **The loss in
   states B and D is not a bus that shrank; the CPU side drew less from a bus that was there.**
   My P that the DDR clock contributes: bad stretch 0.10 → **0.02**; 1,420 arms 0.30 → **0.05**.
2. **The static map is real and steep, but it binds only where no GPU vote stands.** Gold cores above 1,881.6 MHz
   vote 2,092 MHz, at or below it 1,555; only a prime core above 2,592 MHz votes 3,196. The traffic monitor's
   vote is capped at 1,708 MHz in Qualcomm's script. On paper a CPU-alone run under a 1,420 cap has a 27.3 GB/s
   bus. **The phone drew 26.77 GB/s there (*device*, pos 21), and 30.1 with the GPU at 491 (*estimated*, pos 20),
   so on this phone some vote held at least 2,092 MHz**: Samsung's settings differ from the public script, or
   another voter exists. Either way the cap did not take the bus down as far as the paper allows.
3. **One paper effect survives, small:** with the GPU capped to 676 MHz its vote is 2,736 MHz, a 43.8 GB/s bus.
   State C's sum of 41.0 to 42.3 GB/s would be 94 to 97 % of that. If the bus was at that step, C's CPU slowdown
   is a bus ceiling and Bonnie's 1.57 is the better number than my 1.40. **I move my state-C centre to 1.50
   (1.38 to 1.60).** Cool band unchanged: 1.36 (1.28 to 1.45).
4. **Two candidates from the same tree that I had not listed.** (a) The cpufreq driver reads back the *requested*
   step; the hardware's own limit is a separate node, `cpuN/dcvsh_freq_limit`. Bonnie's "clock below the printed
   one" is therefore possible on this kernel and there is a file that shows it. (b) Battery-current protection
   isolates cpu 4 and 5 and caps the GPU at level 2: **closed by the phone**, because the GPU cap stayed at
   level 0 through the bad stretch (*device*).
5. **kgsl on the Flip 3 tree: the 200 ms soft-fault timer is not armed on the A660.** The feature bit is absent
   from the A660's list, so `fast_hang_detect` stays 0 and the timer returns at once. What is armed: the 2,000 ms
   command timeout (`long_ib_detect = true`) and a hardware hang interrupt. Pre-emption is level 1 with a 10 s
   timeout. **I withdraw r18 §8's "a per-token kernel sits near the 200 ms compare".** A token-long kernel has a
   14-fold margin to the only software timer. The screen cost stands: nothing in the driver lets the
   compositor in before a kernel ends.
6. **A middle form of F follows from that: one kernel per block of layers, sized to a display frame.** Per 4
   layers (about 15 ms): **+1.70 / +1.85 / +1.67 over the lane as probed, +0.88 / +0.82 / +0.73 over G at the
   general kernel's rate, +0.50 / +0.43 / +0.39 at 0.9, nothing at 0.8** (*estimated*, width 1). Per token stays
   +1.18 / +1.08 / +0.97 over G. My chance that some form of F beats G by 0.5 tok/s: 0.12 → **0.18**.
7. **MNN and MLC give no second design for G's flag.** Both run the whole model on the GPU and cross the
   boundary once per inference with a blocking call (`finish()` or a blocking read). MNN flushes every 10
   kernels on Adreno against every 2 elsewhere, and replays recordable queues (the mechanism R75's x4d measured).
   Neither splits a matrix or polls shared memory. HeteroLLM stays the only published flag design.
8. **A power-collapse mechanism for the "launch into an idle GPU" step.** The A660 has inter-frame power collapse
   on, with a hysteresis register written at start-up. It fits r18's 124 to 147 µs against 200 to 245 µs. G with
   a warm launch: +0.16 / +0.19 / +0.17 over G (*estimated*).
9. **At a GPU cap of 491 MHz the lane as probed is worth nothing** (6.92 / 6.22 / 6.23 against stock 6.90 /
   6.21 / 6.22, *estimated* from pos 20's rates); G keeps +0.46, F +0.85. Nothing here reaches 12.

## 1. The lahaina bus votes (item 1)

### 1.1 How the DDR clock is chosen (*external*, kernel tree)

- Every voter holds its own interconnect path to the memory controller node. `qcom_icc_aggregate` **sums the
  average votes and takes the largest peak vote** (`icc-rpmh.c`); the bus-control module takes the larger of
  the two results (`bcm-voter.c`). `devfreq_icc.c` passes the governor's frequency as the **peak** vote. So for
  the governors below the DDR clock is the **maximum** of the votes, not their sum.
- The DDR steps (LPDDR5 rows of `ddr_bw_opp_table`): 200, 451, 547, 681, 768, 1,017, 1,555, 1,708, 2,092, 2,736,
  3,196 MHz. The table's "MB/s" are nominal (MHz × 4 bytes); the real bus at 3,196 MHz is 51.1 GB/s
  (*estimated*: 64 data pins, both edges; AnandTech's 51.2).

The voters:

| voter | rule | what it gives under our load |
|---|---|---|
| **`cpu4_llcc_ddr_latmon`** (static map, gold and prime cores; governor `mem_latency`) | takes the fastest core whose instructions per cache miss are at or under `ratio_ceil` (400 in Qualcomm's script) and looks its **measured** frequency up: first row at or above it, else the last row (`core_to_dev_freq`) | ≤ 1,075.2 MHz → 768; **≤ 1,881.6 → 1,555; ≤ 2,592 → 2,092; above → 3,196** (LPDDR5 map of `lahaina-v2.dtsi`) |
| **`cpu7_llcc_ddr_latmon`** (prime floor, `ratio_ceil` 25,000) | same lookup | ≤ 2,592 → 200; above → **3,196** |
| **`cpu_llcc_ddr_bw`** (traffic monitor, governor `bw_hwmon`) | measured MB/s through the cache-to-DDR port ÷ 0.8, rounded up to a zone | saturates at our traffic; **`max_freq` 6,515 MB/s = 1,708 MHz in Qualcomm's script** (Samsung's not read) |
| **kgsl** (GPU, `kgsl_bus.c`, table `qcom,bus-table-ddr8`) | a bus level per GPU power level, moved inside [min, max] by the GPU's own bus governor | 840, 778, 738 MHz: level 11 = **3,196** fixed; **676: level 10 = 2,736** (max 11); 608: 1,555 (max 3,196); 540: 1,555 (max 2,092); **491 and 443: 1,555 (min 768, max 1,555)**; 315: 451 |
| `ddr_cdev` (thermal) | a *floor* of 1,017 MHz at 90 °C package temperature (`ddr_cdev.c` sets a peak vote) | not a cap |
| cpu 0-3 maps, compute monitors, UFS, display | lower rows | below the above |

Cross-check with the phone: R75 printed `thermal_pwrlevel=3` with `gpu_max_mhz=676` (*device*); level 3 of the
speed-bin-0 table is 676 MHz. The table is the phone's.

Two more clocks follow the gold cores by static maps with a loose condition (`ratio_ceil` 4,000): the shared L3
(1,593.6 MHz at full clocks, 1,516.8 at a 2,150 cap, 1,324.8 at 1,785, **1,056 at 1,420**) and the CPU-to-cache
bus (1,000 / 933 / 806 / 806 MHz).

### 1.2 The states of x4s on paper, against what the phone drew

From the script. "Paper DDR" is the range the votes allow; sums are Bonnie r34's (*estimated* from *device*).

| state (pos) | gold / prime / GPU cap, MHz | CPU map vote | GPU vote | paper DDR, MHz | theoretical GB/s | drew, GB/s | reading |
|---|---|---|---|---|---|---|---|
| A cool (1-4) | 2,419 / 2,841 / 840 | 3,196 (prime) | 3,196 | 3,196 | 51.1 | 44.6 to 47.9 | 87 to 94 % of the bus: bus-bound, as priced |
| B bad stretch (5-9) | 2,150 / 2,150 / 840 | 2,092 | **3,196** | **3,196** | 51.1 | 28.2 to 34.3 | 55 to 67 %: **not the bus** |
| C (12-14) | 1,785 / 1,785 / 676 | 1,555 | 2,736 to 3,196 | 2,736 to 3,196 | 43.8 to 51.1 | 41.0 to 42.3 | 94 to 97 % of 43.8, or 81 % of 51.1 |
| D (16-19) | 1,420 / 1,420 / 676 | 1,555 | 2,736 to 3,196 | 2,736 to 3,196 | 43.8 to 51.1 | 33.4 to 36.3 | 76 to 83 % of the lower step: **not the bus** |
| D, GPU 491 (20) | 1,420 / 1,420 / 491 | 1,555 | ≤ 1,555 | 1,708 (monitor cap) | 27.3 | 30.1 | **above the paper ceiling** |
| CPU alone, capped (21) | 1,420 / 1,420 / asleep | 1,555 | – | 1,708 | 27.3 | 26.77 (*device*) | **98 % of the paper ceiling** |
| CPU alone, cool (0) | 2,419 / 2,841 / asleep | 3,196 | – | 3,196 | 51.1 | 28.57 (*device*) | 56 %: not bus-bound alone |

Readings:

- **B and D are not explained by any DDR step.** In B the GPU's vote alone puts the bus at the top. In D the GPU's
  vote is the same as in C, and D drew 6 to 8 GB/s less than C.
- **Pos 20 and 21 contradict a 1,708 MHz bus.** The best the phone showed elsewhere is 94 % of theoretical (one
  arm of A). 98 % with four CPU threads and 110 % with the pair are not credible. So at the 1,420 cap the bus
  was at 2,092 MHz or more. The paper offers no CPU vote above 1,708 there. **Either Samsung's script does not cap
  the traffic monitor at 6,515, or a voter I did not find holds the bus.** My P that the bus sat at 1,708 in
  pos 21: 0.08. It follows that the 6.7 % that CPU alone lost under the cap may be a lower DDR step (2,092:
  33.5 GB/s, 80 % used) as easily as Bonnie's compute coupling (α 0.17); the file cannot separate them and the
  paper cannot either. T5b below does.
- **C is the one state where a paper step may bind.** 2,736 MHz is the GPU's vote at 676; the GPU's bus governor
  may raise it one level. P(the bus was at 2,736 in C): 0.5.
- The GPU kernel's own rate follows the GPU clock, not the bus: 17.6 against 23.9 GB/s at 491 against 676 MHz is
  0.74, the clock ratio is 0.73 (*device* rates, arithmetic).

### 1.3 Candidates for the bad stretch and the capped states, revised

| # | candidate | r18 P | now | why |
|---|---|---|---|---|
| C1 | a transient of Samsung's thermal service at the cap step (user space, not in the kernel tree) | 0.40 | **0.40** | nothing new; still starts with the first step and ends by itself |
| C2 | DDR clock follows the capped CPU | 0.30 (D), 0.10 (B) | **0.05, 0.02** | §1.2 |
| C3 | reclaim and zram during the block | 0.15 | 0.15 | nothing new |
| C4 | pool parking across the serial launch | 0.30 fixed term, 0.10 B | same | Bonnie r34 §9 found the mechanism for the fixed term (one barrier before the claim loop) |
| C5 | **hardware clock limit below the printed step** (Bonnie r34 §3.4, first candidate) | not listed | **0.25 (B), 0.15 (D)** | `qcom_cpufreq_hw_get` returns the requested step from the performance-state register; the hardware limit is kept apart in `dcvsh_freq_limit` and exported per policy as `/sys/devices/system/cpu/cpu{0,4,7}/dcvsh_freq_limit`. So the node the probe reads cannot show it, and one more file can |
| C6 | L3 and CPU-to-cache clocks follow the cap (static, §1.1) | not listed | **0.20 (D only)** | 1,056 MHz at the 1,420 cap against 1,325 at 1,785: the order C good, D bad. Against: CPU alone lost only 6.7 % |
| C7 | battery-current protection isolates cpu 4 and 5 (level 0) or 6 and 7 (level 1) | not listed | **0.03** | the maps are in Samsung's overlay but marked `disable-thermal-zone`, and level 0 also sets the GPU to level 2 (738 MHz); the phone printed level 0 and 840 through the stretch (*device*) |

The sum over B is under 1: the bad stretch is still unexplained, and no further source read will explain it. It
needs T4 with T11's two extra lines.

## 2. MNN's and MLC's boundary (item 2; *external*)

| runtime | where the CPU waits | flush | shared memory | split of one matrix |
|---|---|---|---|---|
| **MNN** OpenCL (master, 2026-11-24) | `onSync(toCpu)` → `commandQueue().finish()`; `copyFromDevice` → blocking `enqueueReadBuffer`; with fine-grain SVM: `finish()` and then the SVM pointer is handed out | **every 10th kernel on Adreno, every 2nd on other GPUs** (`runKernel2D`, `run3DKernelDefault`) | `CL_MEM_ALLOC_HOST_PTR` pools; SVM behind a build flag; no flag word | none; an operator is on one back end |
| MNN, recordable queues (`cl_qcom_recordable_queues`) | one `EnqueueRecordingQCOM` per recording with updated arguments; a source comment says the per-frame `finish()` was removed because every host read does its own blocking transfer | – | – | none |
| **MLC / TVM** OpenCL (tag v0.18.0) | `StreamSync` → `clFinish`; device-to-host copy → `clFinish`, then a blocking `clEnqueueReadBuffer`; kernels are enqueued without a wait outside profiling | none explicit | a mapped `host_ptr` per buffer (`GetNativePtr`), unmapped after a `clFinish` | none |
| HeteroLLM (r18 §2.1) | predicted sleep, then a flag in mapped memory | – | yes | by rows, with the NPU |
| ours as probed (x4s) | event spin after `clFlush` | every launch | fine-grain SVM | by rows |

What it means:

- **No independent second design for the flag, and no published sync cost** from either. Both avoid the boundary
  instead of making it cheap: one crossing per token (the logits), where a `clFinish` of about 400 µs is 0.4 %
  of a token. Our lane crosses it 100 to 200 times per token. That is why the cost is ours to remove.
- **MNN's flush rule is a second witness that a flush is dear on Adreno** (one in ten against one in two). It
  agrees with R75's 92 to 211 µs with a flush against 12 to 38 without (*device*).
- **Recordable queues are MNN's answer to launch cost** and the one R75 already measured (x4d, 44.3 µs per
  replayed launch, *device*; design D is closed by the lead). Nothing to add to the try list from MNN.
- Not examined: MNN's LLM front end above the back end, and `cl_qcom_perf_hint`, which MNN sets (Bonnie's r34
  item 5 asks about it).

## 3. kgsl on the Flip 3 tree (item 3; *external*, the SM-F7110 mirror)

| setting | value in the tree | where |
|---|---|---|
| soft fault detect (the 200 ms register compare) | **not armed on the A660.** `adreno_fault_detect_init` returns unless the core has `ADRENO_SOFT_FAULT_DETECT`; the A660 and A660v2 feature lists are `RPMH, GPMU, APRIV, IOCOHERENT, CONTENT_PROTECTION, IFPC, PREEMPTION` (+ `ACD, BCL` on v2). `fast_hang_detect` stays 0 and `adreno_dispatcher_fault_timer` returns on its first line | `adreno.c`, `adreno-gpulist.h`, `adreno_dispatch.c` |
| `_fault_timer_interval` | 200 ms (inert, above) | `adreno_dispatch.c` |
| command timeout | `adreno_drawobj_timeout = 2000` ms, `long_ib_detect = true` by default; on expiry "gpu timeout" and a fault, unless the context has `NO_FAULT_TOLERANCE` (the OpenCL driver makes the context; we cannot set it) | `adreno_dispatch.c`, `adreno.c` |
| fault policy | replay, skip the command, throttle (3 faults in 3,000 ms) | `adreno.c`, `adreno_dispatch.c` |
| hardware hang interrupt | on: `RBBM_INTERFACE_HANG_INT_CNTL = (1 << 30) | 0xcfffff`; the interrupt is in the mask and raises a hard fault ("GPU hang detected", and Samsung adds an error report) | `adreno_a6xx.c`, `adreno.c` |
| pre-emption | on for the A660; `preempt_level = 1`, `skipsaverestore = true`, `usesgmem = true`; a stuck pre-emption faults after `ADRENO_PREEMPT_TIMEOUT` = 10,000 ms | `adreno_a6xx.c`, `adreno_a6xx_preempt.c`, `adreno.h` |
| context priority | medium unless the creator sets it; the ringbuffer is chosen by priority | `adreno_drawctxt.c`, `adreno.h` |
| inter-frame power collapse | on (`ADRENO_IFPC`); hysteresis register written `0x000A1680` at start | `adreno_a6xx_gmu.c` |
| idle timeout to slumber | 80 ms (`CONFIG_QCOM_KGSL_IDLE_TIMEOUT`) | defconfig |

What I could not read from the tree: what the hardware hang counter counts (the register's unit and which
activity resets it are not in the driver), and the layout of the hysteresis word. x4e's 188 ms kernel raised no
fault (*device*), so a kernel that computes or spins for 188 ms is not a "hang" to that counter at 840 MHz. If
the low field of the hysteresis word is 5,760 ticks of the 19.2 MHz clock it is 300 µs; that is a guess.

What it means for F (no verdict):

- **Faults: the per-token form is safe on paper.** 85 to 145 ms against 2,000 ms. r18 §8's 200 ms worry was read
  from an older tree where the feature is on; on this tree it is off. P(a fault at or under 400 ms in T9): r18
  implied about 0.4; **now 0.08** (the hardware counter is the remainder).
- **The screen: unchanged, and now the only thing against a long kernel.** Level-1 pre-emption switches at
  boundaries inside a command stream; one compute dispatch has none. A higher-priority ringbuffer waits for the
  kernel to end, up to 10 s before the driver calls it stuck. So frames wait for the kernel's length, with no
  fault. P 0.85. A token-long kernel means the screen updates once per token, 7 to 12 times a second, while the
  model writes. Whether that is acceptable is the owner's to say, not a driver limit.
- **Hence the block form (§4.2):** a kernel that lives for 2 layers (about 7 ms, inside one 120 Hz frame) or 4
  layers (about 15 ms, inside one 60 Hz frame). T9 gets the lifetimes 7, 15 and 30 ms.
- **Power collapse explains r18's cheaper launches.** When the lead ended in the spin the next launch followed
  the last kernel at once and `enq_us` was 124 to 147 µs; elsewhere the GPU had idled for 0.3 to 1 ms and it was
  200 to 245 µs (*device*). A GPU that the management unit has collapsed has to be woken by the launch. This is a
  mechanism for T10's step, and for Bonnie's r34 item 5 ("why 124 to 147 in state C").

## 4. Prices (*estimated*; committed stage model through r18's script; width 1, O 8, measured head)

### 4.1 Coupling bands after this round

CPU slowdown while the GPU kernel runs beside four CPU threads:

| phone state | r18 | r19 | why it moved |
|---|---|---|---|
| full clocks, GPU 840 | 1.36 (1.28 to 1.45) | **unchanged** | on paper the bus is at the top step by two votes (prime floor, GPU level 11); nothing in the tree argues against the cool fit |
| caps at 1,785, GPU 676 (state C) | 1.40 (1.33 to 1.58) | **1.50 (1.38 to 1.60)** | the GPU's vote at 676 is 2,736 MHz; C's sum would be 94 to 97 % of that bus. Half-way to Bonnie's 1.57 because P(2,736) is 0.5 |
| caps at 1,420, GPU 676 (state D) | 1.8 (1.5 to 2.1) | **unchanged, mechanism open** | not the bus (§1.2); C5 and C6 are the paper candidates |
| GPU cap 491 | – | **1.8 with the GPU kernel at 0.74 of its rate** | new row from pos 20 (one arm) |
| bad stretch | 2.4 to 3.2, not priced | **same, not priced** | not the bus; unexplained |

### 4.2 Designs

| design | tok/s | over as probed, same state | over G cool |
|---|---|---|---|
| as probed, cool (r18) | 7.59 / 6.82 / 6.78 | 0 | – |
| **G**, cool (r18) | 8.41 / 7.85 / 7.72 | +0.82 / +1.03 / +0.94 | 0 |
| G with a warm launch (125 µs, §3) | 8.57 / 8.04 / 7.89 | +0.98 / +1.22 / +1.11 | +0.16 / +0.19 / +0.17 |
| **F per token**, rate 1.0 (r18) | 9.59 / 8.93 / 8.68 | +2.00 / +2.11 / +1.90 | +1.18 / +1.08 / +0.97 |
| F per token, rate 0.9 | 9.18 / 8.52 / 8.32 | +1.59 / +1.70 / +1.54 | +0.77 / +0.67 / +0.60 |
| F per 8 layers (about 30 ms) | 9.44 / 8.80 / 8.57 | +1.85 / +1.98 / +1.79 | +1.03 / +0.95 / +0.85 |
| **F per 4 layers** (about 15 ms) | 9.29 / 8.67 / 8.45 | +1.70 / +1.85 / +1.67 | **+0.88 / +0.82 / +0.73** |
| F per 4 layers, rate 0.9 | 8.91 / 8.28 / 8.11 | +1.32 / +1.46 / +1.33 | +0.50 / +0.43 / +0.39 |
| F per 4 layers, rate 0.8 | 8.51 / 7.89 / 7.75 | +0.92 / +1.07 / +0.97 | +0.10 / +0.04 / +0.04 |
| F per 2 layers (about 7 ms) | 9.01 / 8.42 / 8.23 | +1.41 / +1.60 / +1.45 | +0.60 / +0.57 / +0.51 |
| F per layer (r18) | 8.57 / 8.04 / 7.89 | +0.98 / +1.22 / +1.11 | +0.16 / +0.19 / +0.17 |
| as probed, state C, σ 1.50 | 7.53 / 6.74 / 6.70 | 0 | – |
| G, state C | 8.30 / 7.73 / 7.61 | +0.78 / +0.99 / +0.91 | −0.10 / −0.12 / −0.11 |
| F per token, state C | 9.39 / 8.73 / 8.51 | +1.86 / +1.99 / +1.80 | |
| as probed, state C at σ 1.60 | 7.47 / 6.66 / 6.63 | −0.06 / −0.07 / −0.07 | |
| as probed, GPU cap 491 | 6.92 / 6.22 / 6.23 | 0 (stock is 6.90 / 6.21 / 6.22) | |
| G, GPU cap 491 | 7.38 / 6.68 / 6.66 | +0.46 / +0.46 / +0.42 | |
| F per token, GPU cap 491 | 7.77 / 7.12 / 7.06 | +0.85 / +0.90 / +0.83 | |

The block rows spread one 400 µs launch over the block's split stages (four per layer, as r18's per-layer row
did). Rung-7 form: G 10.59 / 10.46 / 10.14; F per token 11.86 / 11.66 / 11.24; F per 4 layers 11.51 / 11.35 /
10.96; at rate 0.9 11.17 / 11.00 / 10.64. **None reaches 12.**

What each finding does:

| finding | to G | to F |
|---|---|---|
| DDR votes (§1) | nothing cool; state C 0.1 lower than cool; **no bus reason to fear the caps** beyond that. The GPU's vote lasts while kgsl is awake (80 ms idle timeout), so it is the same for G and F | the same |
| GPU cap 491 | G is worth +0.46 there; as probed is worth 0. L6 (self-timed fall-back) matters only for the lane as probed | +0.85 |
| power collapse (§3) | +0.16 to +0.19 if the launch can be kept warm (L5); P(T10 shows the step) 0.45 → **0.60** | F never lets the GPU idle; nothing to gain, and it is the heat question T8 asks |
| kgsl timers (§3) | none | per-token: fault risk gone. Screen cost stays. The block form makes F's price a choice between +0.5 to +0.9 (frame-sized) and +1.0 to +1.2 (token-sized) over G |
| general kernel's source (§8) | none: G uses it as it is | two more changes in the hot loop than r18 counted (claimed row index, x from a global ring instead of an image). **P(rate ≥ 0.9): 0.5 → 0.4** |

**My chance that F beats G by 0.5 tok/s on the phone, redone.** Block form (4 layers): needs a rate of about
0.93 (P 0.33), no heat penalty from a GPU that never idles (T8, P 0.55), frames in order at 15 ms (P 0.9):
0.16. Per-token form: rate ≥ 0.85 (P 0.5), T8 (0.55), and the owner accepting a screen that updates per token
(I put no number on the owner; say 0.3): 0.08. The two overlap on the rate and on T8. **About 0.18 (0.10 to
0.25)**, up from 0.12. The order of work does not change: **G first.**

## 5. Test list and try list: changes to r18's (device items wait for the owner and their own pre-registration)

| # | check | where, cost | what closes it | my band |
|---|---|---|---|---|
| **T5** (changed) | **drop arms (a) and (b)**: on paper the GPU's vote does not depend on the CPU clock, so "GPU alone, CPUs idle against CPUs spinning" cannot show a difference. Keep the pair at full clocks against the pair under caps at the same GPU cap | device, 10 s plus a heat-up | pair sum under caps ≥ 0.92 of cool at the same GPU cap → no state-D effect in a clean cell | 0.80 (0.70 to 0.95) |
| **T5b** (new) | **Is CPU alone under caps limited by a DDR step?** The CPU ring under caps (i) GPU asleep, (ii) GPU held busy at its cap by a kernel that touches no memory (register spin, duty ≥ 0.75) | device, 8 s, after the heat-up of T5 | (ii) faster by ≥ 4 % → the CPU's own vote sits under the GPU's; the 6.7 % is the bus and Bonnie's α is refit without those cells. (ii) within 2 % or slower → the bus is not the limit; the paper cap at 1,708 does not hold on this phone | P(≥ 4 %) 0.15 |
| **T11** (new) | **Three more lines per arm**: `cpu{0,4,7}/dcvsh_freq_limit`, `/sys/devices/system/cpu/isolated`, and a 20 ms register-only spin on cpu 7 and cpu 4 timed against the monotonic clock (an effective-clock canary; the same as Bonnie's §6 compute canary) | probe, 20 lines; 40 ms per arm | `dcvsh_freq_limit` under the printed cap, or the canary off by > 5 % from the printed step, in a slow arm → C5 holds. A core listed in `isolated` → C7. Neither in a slow arm → both closed. An unreadable node is printed as such | P(C5 seen in a slow arm) 0.2 |
| **T9** (changed) | lifetimes **7, 15, 30,** 150, 400, 1,000 ms; frame times of the app while each runs | device, 25 s | a fault at ≤ 400 ms (my P 0.08) → blocks only. Frames delayed by about the kernel's length (P 0.85) → the block size is the frame budget; the owner is asked about per-token | |
| **T10** (band changed) | as r18; gaps 0, 0.1, 0.2, 0.5, 1, 5 ms | device, 5 s | a step of ≥ 60 µs → L5 | P 0.60; the step between 0.2 and 1 ms |
| **T6** (form changed) | persistent general kernel **in the block form**: rows claimed 64 to 128 at a time, x read from a global SVM ring (one fresh slot per stage), the block's weights behind one image; identity against the general kernel bit for bit | host, 2 to 3 days (was 1 to 2) | as r18; plus: `CL_DEVICE_MAX_READ_IMAGE_ARGS` and the texel limit printed | |
| **T12** (new, optional, owner, read-only) | the phone's own values: `grep -n "max_freq\|io_percent\|ratio_ceil" /vendor/bin/init.kernel.post_boot*.sh` over adb | 1 minute; changes nothing on the phone | says whether the traffic monitor is capped at 6,515 on this phone (§1.2's contradiction) | P(not capped at 6,515) 0.6 |

Try list, changes only (all generic across models):

| # | lever | label | price (*estimated*, width 1, cool) | needs |
|---|---|---|---|---|
| L1 | launch inside the pool's job | bit-identical | as r18, **with Bonnie's two parts**: the hook after the Q4_0 barrier and `GGML_KLEIDIAI_CHUNK_MULTIPLIER=16` for lane stages (her §9); at multiplier 4 it gains nothing | T1 with her addition |
| L3 | F, **now in two forms**: per token, and per block of 2 to 8 layers | bit-identical per row if T6 passes | over G: per token +1.18 / +1.08 / +0.97; per 4 layers +0.88 / +0.82 / +0.73; both only at rate ≥ 0.9 | T6 to T9 |
| L5 | keep the launch warm: the next launch follows the last kernel within the collapse hysteresis, or a no-op kernel fills the gap | bit-identical | +0.16 / +0.19 / +0.17 over G | T10 |
| L6 | self-timed fall-back | bit-identical per path | needed for the lane as probed at a GPU cap of 491 (worth 0 there); not needed with G | none |

No lever for the bus: an app cannot vote, and holding the GPU busy to keep its vote costs more heat than T5b
could repay (at most 6.7 % of CPU-alone stages under caps).

## 6. Bonnie r34: where I agree and where I differ

**Agree, with new support from the tree:**

- Her §3.4 second candidate ("DRAM frequency voted down ... does not fit B"): the tables confirm B cannot be it.
- Her first candidate ("clock below the printed one"): the Flip 3's cpufreq driver does read back the request,
  and keeps the hardware limit in another node. I had not listed it; it is C5 with P 0.25 for B. T11 reads it.
- Her state-C slowdown 1.57 against my 1.40: the GPU's 2,736 MHz vote at 676 gives her number a mechanism. I
  moved to 1.50.
- Her §9 (16 chunks, one barrier before the claim loop): accepted into L1 as written.
- Her §3.5 and item 5 (the launch costs 214 µs in A and 124 to 147 in C): power collapse is my reading (§3).
- "Battery at 15 % is not needed for the caps": the battery-current maps are disabled in Samsung's overlay and
  the GPU cap did not move, so C7 is closed too.

**Differ; both positions stand:**

1. **"DRAM frequency voted down fits D against C."** On paper it does not: the GPU's vote is the same in C and D
   (same 676 cap), and D's sum is under either step. Where a low step *is* on paper (pos 20 and 21) the phone
   drew more than that bus can carry. I put 0.05 on it for D. She may answer that the GPU's bus governor could
   have left C at level 11 and D at level 10; that makes D's ceiling 43.8 GB/s, still 7 above what D drew.
2. **CPU alone under caps (her α 0.17 out-of-sample cells, pos 0 and 21).** She reads the 6 % as compute
   coupling. I say the paper allows a DDR step under it (2,092 MHz, 80 % used) and that the two cannot be told
   apart from the file. T5b decides. If (ii) is faster, her two cells are not evidence for α.
3. **The size of the join and what a flag removes** (r18 §6 items 1 and 2): unchanged, T2 and T3 decide.
4. **F.** She: F is what the 12 cells need, P(fast persistent kernel) 0.3. I: P(rate ≥ 0.9) 0.4 after reading the
   kernel (§8), F over G by 0.5 at 0.18, and the block form is the one I would build because it does not need
   the owner to accept a per-token screen.

**Her pricing I checked this round:** "one microsecond of serial launch costs what 2.4 µs of K cost" (1 ÷ (1 −
0.589)): right in her model. Her cool K of 207 + 13.3 µs "serial as built": I agree for cool arms and add that
it is a property of a GPU that had idled (power collapse), so it is the right K for the lane as probed and too
high for G with a warm launch. Her 44.3 µs replay figure for a recordable queue is the launch without the wake;
whether replay also shortens the flush is her item 5 and is open.

## 7. Next questions (ranked)

1. **What the general Q4_0 kernel needs to live across stages** (source read of the kernel the probe used):
   decides T6's form and cost. First pass in §8.
2. **Whether the done flag can carry y** (dst as fine-grain SVM, read by the CPU after the flag): removes H's
   13 µs and the read; G only. One paragraph in §8; the rest is T2.
3. **Samsung's thermal service at a cap step** (what else it changes: cpusets, boosts, task placement): the only
   reading left for the bad stretch that a source could inform. Public sources are thin; P(useful) 0.2.
4. **A GPU-side prefill** to keep the big cores cold before decode (r17 item 4, r18 item 5).
5. **`cl_qcom_perf_hint` and the flush** (Bonnie's item 5; MNN sets the hint): whether a hint keeps the GPU out
   of power collapse without a filler kernel.

I continue with 1.

## 8. Next question 1, first pass: the general kernel across stages (*source*: `lab/llama.cpp-c3/ggml/src/ggml-opencl/kernels/gemv_noshuffle_q4_0_f32.cl`)

*Source* facts about `kernel_gemv_noshuffle_q4_0_f32`:

- Arguments: quants through an image over a buffer (`image1d_buffer_t src0_q`, one 32-bit channel), scales
  through a global `half2` buffer, **x through a second image** (`src1`), y to a global `float` buffer, then
  sizes. Fifteen arguments, matching the probe's 15 `clSetKernelArg`.
- A work-group is four waves of a half-size subgroup; a work-item owns **two rows fixed by `get_global_id(0)`**.
  Per step along K the first four items of a wave read 8 x values each and the wave shares them with
  `sub_group_broadcast`; the weights are eight image reads per item. One `barrier` for the reduction over the
  four waves, then two stores.

What a version that lives for several stages has to change, and what each change risks:

| # | change | why | risk |
|---|---|---|---|
| 1 | **rows claimed, not given by the global id** | work-groups that have not started cannot start while the first ones wait for the next stage; only as many groups run as the hardware runs at once. Each running group claims 64 or 128 row pairs per atomic and loops (r18 P2; the probe's `q4p` did this with the slow kernel) | one atomic and one extra barrier per claim: under 1 % by arithmetic at 128 rows (*estimated*); the index is no longer a built-in, which may cost the compiler something |
| 2 | **x from a global SVM ring, not from an image** | an image's content is fixed at launch as far as the kernel can rely on; R75's x4e saw CPU writes inside a running kernel only through C 2.0 atomics on fine-grain SVM (*device*). With a ring of one fresh 16 KB slot per stage (16 slots for a 4-layer block, 256 KB) no address is read twice in a kernel's life, so plain loads after the atomic sequence number are enough | x is 6 % of the bytes a group reads (16 KB against 262 KB, arithmetic); the path changes from the texture unit to global loads. Device only |
| 3 | **the block's weights behind fixed arguments** | arguments are fixed at launch. One image per block with a base offset per stage: a 4-layer block of fused gate and up is 45 M texels of the 134 M limit (*device* limit, arithmetic), so 11 layers fit one image. Needs the lane's matrices of a block allocated as one buffer at load (layout only; the served file is not touched) | load-time code; residency unchanged in bytes |
| 4 | **y and a done counter in SVM** | the CPU must see the stage end without the driver (r18 G); each group adds its claimed rows to an atomic counter | x4e: 3.75 µs to the CPU (*device*) |
| 5 | a wait loop between stages | the kernel spins on the sequence number | heat: T8. At most a block long, so no timer is near (§3) |

Bit-identity: the row arithmetic (the two dequantise macros, the reduction order over four waves) is untouched
by 1 to 5; rows move between work-items, not between formulas. I expect per-row identity to the general kernel
and therefore to the lane with `clFinish`; T6 shows it or F becomes a numerics change.

**For question 2 (y in the flag's page):** y is already a plain global buffer in upstream's kernel. Passing an SVM
pointer for it needs no kernel change, so G can have it: the CPU reads y from shared memory once the trailing
kernel's flag is seen. One thing to show on the phone in T2: y read through SVM right after the flag equals y
after a `clFinish`, in every cycle of 1,000 (the GPU's write cache must be drained when the trailing kernel
runs; I expect so, in-order queue, but it is not written anywhere I read).

Effect on the plan: T6 grows from 1 to 2 days to 2 to 3 (changes 2 and 3 are new), and P(rate ≥ 0.9) goes from
0.5 to 0.4. Both are in §4.2 and §5 already.

Ranked next questions after this pass: (1) the done flag carrying y and the exact trailing-kernel form for T2,
with Tom's host shim; (2) Samsung's thermal service at a cap step; (3) a GPU-side prefill; (4)
`cl_qcom_perf_hint` and the flush; (5) how many work-groups an Adreno 660 runs at once (sets the NDRange of
change 1; device only, one cell).

## Sources (*external*; all fetched 2026-11-25)

- Samsung Flip 3 kernel mirror (SM-F7110, Android 15 release): https://github.com/xiliuya/android_kernel_samsung_b2q/tree/31574e5bd1
  (device tree under `arch/arm64/boot/dts/vendor/qcom/` and `arch/arm64/boot/dts/samsung/b2/b2q/`; drivers as listed at the top).
- Qualcomm lahaina post-boot script in a Xiaomi SM8350 tree: https://github.com/LineageOS/android_device_xiaomi_sm8350-common/blob/lineage-21/rootdir/bin/init.kernel.post_boot-lahaina.sh
- MNN OpenCL back end (master `28b8cfac4c`, 2026-11-24): https://github.com/alibaba/MNN/tree/master/source/backend/opencl/core
- TVM OpenCL runtime (tag v0.18.0): https://github.com/apache/tvm/blob/v0.18.0/src/runtime/opencl/opencl_device_api.cc
- Carried from r18 and Bonnie r34, not re-read: HeteroLLM arXiv 2501.14794; AnandTech's Snapdragon 888 table;
  Qualcomm's DVFS governor pages (Bonnie's sources).
