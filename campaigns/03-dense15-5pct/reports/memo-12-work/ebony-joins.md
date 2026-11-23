# c3 memo-12 work (Ebony): the join after R74. What others do, what the raw lines say, structures that avoid it

Date: 2026-11-23. Author: Ebony (`claude-fable-5-2`). Not committed. **Bonnie has not reviewed this**, and her
`bonnie-joins.md` did not exist when I finished (checked twice, last at the end of this work). Her script
`bonnie_joins.py` appeared at the end; §6.1 reviews its bands from the code alone. §6 was written before I saw it.

Labels: *device* = a value the phone printed in R74 (or R72 where said); *device-derived* = my arithmetic on two or
more *device* fields of the same line (still *estimated*, but with no model in between); *estimated* = a model;
*external* = another group's number on other hardware; *source* = read in the upstream tree at base `620d2225c`.
Nothing here is a new phone number. Work file (untracked): [ebony_joins.py](ebony_joins.py) (self-checked).

Triples are Yi / Llama-2 / DeepSeek. Goal 12 tok/s = 79.2 ms.

## 1. Verdict

- **The 452 µs is a launch cost, not a wait cost.** 72 to 87 % of it lies before the kernel starts
  (`queued_to_start_us` 327 to 405 of `wall − dev` 383 to 531, *device*); the part after the kernel ends is 57 to
  172 µs (*device-derived*). Launches that are already queued behind each other still cost 199 to 320 µs each
  (`x3 alone`, 8 per `clFinish`) and 252 to 293 µs each (`x3 layer` sync1, 7 per `clFinish`) (*device-derived*).
  So "wait less" (polling) and "enqueue earlier" (pre-enqueue on the stock queue) attack the smaller part.
- **A dead GPU costs half of its dead time, not all of it,** as long as the CPU has rows of the same level to do:
  per level T = (B + r_g·λ) ÷ (r_c + r_g), so λ is weighted by r_g ÷ (r_c + r_g) = 0.52 (*estimated*; it reproduces
  the `x3 split` cell: 0.850 ms against 0.865 measured, *device*). R74's additive `64 × 452 µs` overstates the
  dispatch by about 2 ×, and its `lane = 0.621` on every matrix understates the body, because a row split needs one
  launch per dependency level (4 per layer, not 2). The two errors nearly cancel: my launch-level price of the same
  lane is **8.13 / 7.51 / 7.42** tok/s (*estimated*) against the result's 7.86 / 7.39 / 7.30.
- **The number to cut is λ, GPU dead time per launch, and the count of launches.** Price by λ (*estimated*, K0 +
  lane only, levels stacked so that one launch covers a level):

  | λ per launch | tok/s | body × | what would give it |
  |---|---|---|---|
  | 0.452 ms (R74 `x3 rt`) | 8.13 / 7.51 / 7.42 | 0.83 / 0.81 / 0.81 | stacking alone (Y12) |
  | 0.30 ms | 8.54 / 7.99 / 7.85 | 0.78 / 0.76 / 0.76 | the queued-launch level R74 already shows |
  | 0.12 ms | 9.51 / 8.83 / 8.60 | 0.69 / 0.68 / 0.68 | the best queued cell (`x3 tog` gpu arm, §2 F5), if it is real |
  | 0.05 ms | 9.98 / 9.22 / 8.95 | 0.65 / 0.64 / 0.64 | recordable replay or a fast launcher at its best |
  | 0.02 ms + 0.45 ms per token | 10.14 / 9.36 / 9.07 | 0.63 | persistent GPU worker kernel (Y15) |
  | 0 | 10.34 / 9.53 / 9.22 | 0.621 | the result's zero-dispatch column |

- **No structure found here reaches 12**; the ceiling of every one is the zero-dispatch column (10.3 / 9.5 / 9.2
  alone, 12.2 / 11.9 / 11.4 in the centre stack, R74 §4). The two structures the lead suggested (GPU runs a whole
  speculative chain; GPU-only layers in a pipeline) are **closed by price** (§4.1, §4.2). The campaign verdict does
  not move.
- **What is worth a device cell,** if the owner ever allows one: a launch timeline (X4-a) and a launcher-core and
  wait-mode matrix (X4-b). They cost a few seconds of phone time and decide between every design in Bonnie's lane
  and mine. Host work before that: level stacking and the numerics of one reorder (§7).

## 2. What the R74 raw lines already say (my re-read; `python3 -I`, not committed)

| # | fact | cells | label |
|---|---|---|---|
| F1 | Isolated launch + `clFinish`: `ovh_us` 383 / 531 / 452 / 481 at 512 / 1024 / 2816 / 4096 rows. `queued_to_start_us` 327 / 359 / 395 / 405, of which `submit_to_start_us` 184 / 194 / 199 / 197 | `x3 rt` | *device* |
| F2 | So queued→submit (driver, host side) is 143 to 208 µs, submit→start (kernel driver and GPU) 184 to 199 µs, and end→return (`wall − dev − queued_to_start`) 57 / 172 / 57 / 76 µs | `x3 rt` | *device-derived*; wall is from the plain queue, the rest from the profiling queue |
| F3 | Eight launches, then one `clFinish`: `us_per_gemv − dev_us` = 199 (512 rows), 261 (4096²), 320 (4096×11008), 311 (11008×4096) µs per launch, cold. It grows with the matrix | `x3 alone` | *device-derived* |
| F4 | Seven launches of a layer, one `clFinish`: `ovh_us_per_dispatch` 293 / 252 / 261. A `clFinish` after each adds `rt_us_per_dispatch` 434 / 453 / 249 | `x3 layersum` | *device* |
| F5 | The same shape, ring and batch code as F3's last cell, run in `x3 tog` (gpu arm): 24.60 and 25.16 GB/s against 21.01. That is 1,019 against 1,207 µs per gemv, an overhead of about **124 µs** if `dev_us` is the same 896 (not measured in `tog`) | `x3 tog` pos 1, 4; `x3 alone` | *device-derived*, one assumption |
| F6 | Feeder thread CPU time per queued launch (`feeder` × wall): 119 µs (512 rows), 157, 179 (11008×4096) in `x3 alone`; 141 to 178 in sync1; **270 to 480** in sync7; about 95 to 115 in the `tog` gpu arms | all GPU-only cells | *device-derived* |
| F7 | Policy 0 (the A55s, where the feeder is pinned) read 691 to 1,094 MHz of 1,804 in the `x3 alone` and `x3 layer` cells and 806 to 1,804 in the `tog` gpu arms. `ramp_ok=1` says nothing here: the GPU-only arms have no CPU set to ramp, and the GPU clock is unreadable (`e13`) | `mhz_range` | *device* |
| F8 | Llama-2 and DeepSeek have identical layer shapes, yet sync7 reads 9.134 against 7.774 ms per layer (17 % apart), with feeder CPU 0.369 against 0.243 | `x3 layer` | *device* |
| F9 | Row split of one 11008×4096 with a `clFinish` per matrix: the CPU waited 35 and 68 µs at share 0.25 (119 to 136 at 0.35, 211 to 409 at 0.50). Ratio 0.912 at 0.25 | `x3 split` | *device* |

Reading:
- F3 and F4: **the stock queue does not make queued launches cheap.** Whatever the Adreno driver does at enqueue
  and submit, 200 to 320 µs per launch remain when the host never waits in between.
- F6 against F3: 55 to 60 % of the queued overhead is CPU time of the launching thread, and it is serial with the
  GPU (if it overlapped, sync1 would read near `dev_ms` + 0.5 ms; it reads `dev_ms` + 1.8 to 2.1 ms).
- F5, F7, F8: the overhead moves by a factor of 2.5 between cells that differ only in the state of the launching
  core's clock (and possibly the GPU's, which is unreadable). **My leading hypothesis: a large part of λ is driver
  code running on an A55 at 0.7 to 1.1 GHz.** P 0.5. Against it: in `x3 split` the feeder spins (so its clock
  should be up) and `gpu_ms − dev` is still about 440 to 520 µs (F9, with `dev` taken from `x3 rt`; contention
  from four busy A78s is the confound). X4-b decides it.
- F9: with the CPU busy on its own rows, the join itself is 35 to 68 µs of CPU wall time. The join is not the cost;
  the GPU's late start is, and it is paid as lost GPU rows (the 0.52 weight of §1).
- F3's growth with size (199 → 320 µs) is not explained by anything above. Candidates: per-launch work that scales
  with the buffer (mapping, cache maintenance), or device time outside START→END. X4-a prints it.

## 3. What others do (scouted 2026-11-23)

| runtime / source | how it submits and joins | join or launch cost they report | hardware | label, read status |
|---|---|---|---|---|
| **llama.cpp OpenCL backend**, base `620d2225c` (2026-10-15) | `graph_compute` enqueues every node and returns; no `clFinish`, no `clFlush`, no wait per op in a non-profiling build. The join is the blocking `clEnqueueReadBuffer` in `get_tensor` (logits): **one join per token**. In-order queue, properties 0. No recordable queue, no perf or priority hint, no SVM. Weights: device allocation, `ALLOC_HOST_PTR` + map as a fallback; a comment says `USE_HOST_PTR` is not zero-copy on Adreno | none stated | – | *source*, read (`ggml-opencl.cpp` lines 7029-7037, 7775-7812, 8611 on, 11771 on) |
| Qualcomm, IWOCL 2026 talk on that backend (May 2026) | lists "Recordable Command Queue/Buffers" under future work: `cl_qcom_recordable_queue` "enables replay of per-layer launches with minimal CPU overhead"; `cl_khr_command_buffer` to replace it; pre-recorded command buffers marked "partial" for OpenCL | "up to ~10 % performance uplift observed on selected large models (MoE)" | Snapdragon X2 class; not a phone number | *external*; slide text extracted from the PDF, layout lost |
| **HeteroInfer / HeteroLLM** (SOSP 2025) | GPU + NPU, tensor-level split. Finds `clFinish` too slow and replaces it: buffers mapped into both address spaces, the waiting thread **sleeps for the predicted kernel time, then polls a flag** from a small or middle core | "`clFinish` … a fixed latency of approximately **400 microseconds**"; `usleep` granularity 80 to 100 µs; polling "a few microseconds"; submission "approximately tens of microseconds" (prefill); fast sync gives decode 4.01 × for the tensor-level split on Llama-8B | Snapdragon 8 Gen 3 (Adreno 750) | *external*; HTML v2 read through the fetch tool's summary with quotes |
| **MNN** OpenCL runtime (master) | creates a `CL_QUEUE_RECORDABLE_QCOM` queue when the device reports `cl_qcom_recordable_queues` and `CL_DEVICE_RECORDABLE_QUEUE_MAX_SIZE` > 0; commit batch `encorderNumForCommit`, default 10; sets `CL_CONTEXT_PERF_HINT_QCOM = HIGH` and priority hint LOW on Adreno; `finish()` only when switching to the profiling queue | none stated. arXiv 2607.05475 (read for memo-11): Qwen2.5-7B decode on an 8 Gen 3, MNN GPU 11.7 tok/s against llama.cpp GPU 5.4 | – / Snapdragon 8 Gen 3 | *external*; `OpenCLRuntime.cpp` read through the fetch tool. **Whether MNN's LLM path replays recordings was not checked** |
| `cl_qcom_perf_hint` (Khronos registry) | context property HIGH / NORMAL / LOW; **HIGH is the default**, NORMAL lets the governor choose | no latency text | – | *external*, spec read |
| `cl_qcom_recordable_queues`, `cl_qcom_priority_hint` | not in the Khronos registry (directory listing read); documented only in the Adreno OpenCL SDK, which I could not open | – | – | *external*, search summary |
| kgsl (downstream kernel trees) | CPU latency bound (pm QoS) default 501 µs in older trees; GPU idle timeout 80 ms in one Snapdragon 7-class config; dispatch on kthreads | no submit→start figure | not the 888's tree | *external*, **search summary only** |
| MLC / TVM, ExecuTorch / QNN, PowerInfer-2 | not read this round | – | – | nothing stated |
| "Is Your NPU Ready" (2607.05475), "When NPUs Are Not Always Faster" (2605.27435), AHASD (2604.25326), HeteroMosaic (2607.12839) | sync and dispatch named as the decode-stage cost on NPUs (backend sync 3,449 ms against 71 ms for the CPU path in one breakdown) | no per-join µs | Snapdragon 8 Gen 3 | *external*, **search summary only** this round |

This phone reports `cl_qcom_recordable_queues`, `cl_qcom_perf_hint`, `cl_qcom_priority_hint`,
`cl_qcom_ext_host_ptr_iocoherent`, `cl_qcom_ion_host_ptr`, `cl_qcom_android_ahardwarebuffer_host_ptr` and the global
int32 atomics (*device*, R72 `x2info`).

Reading:
- **Our 452 µs is the published number.** HeteroInfer's 400 µs for `clFinish` on an Adreno 750 is the same cost two
  GPU generations later. It is a property of the driver path, not of this phone's state.
- Their cure has two parts, and they map onto F2 differently. Sleep-then-poll removes the end→return part (our 57 to
  172 µs). Their "tens of microseconds" submission is far below our queued→start (327 to 405 µs); either the 8 Gen 3
  driver is faster there, or their launcher sat on a faster core at full clock. Not known.
- **Upstream already has one join per token** and is still the slowest GPU runtime in the one published comparison.
  With R74's 252 to 293 µs per queued launch (F4) and upstream's count of launches per token (I did not count; at 7
  gemvs alone it is 210 to 224), that is ≥ 53 to 66 ms per token of launch cost on this phone (*estimated*).
  **Cutting joins without cutting launch cost reproduces upstream's problem.**
- Recordable replay is what the vendor itself names for this, and MNN has it. The only published size of the gain is
  "up to ~10 %" on a desktop-class SoC.
- The perf hint is already HIGH by default, so "ask for high performance" is not a lever. NORMAL would be worse.

## 4. Structures that change the shape, not the price

### 4.1 GPU runs the guessed next token's whole pass beside the CPU (one join per token): closed by price

- Shape: upstream's backend as a second instance computes the head's guess for token t + 1 through every layer while
  the CPU computes token t. One join per token.
- Both read the whole body at once, so the CPU drops to 19.77 of 25.64 GB/s (body × 1.30) and the GPU pass takes the
  `x3 layer` sync1 time under contention.
- Price (*estimated*, measured head top-1): pass 228 / 233 / 231 ms for 1.38 / 1.50 / 1.44 tokens:
  **5.74 / 6.11 / 5.91** tok/s, below K0 (6.90 / 6.21 / 6.22).
- Why: the phone is bus-bound, and a second position that shares the weight read (width 2 on the CPU, kB 1.12,
  memo-10) costs far less than a second reader of the same bytes. Any design that gives the GPU *its own* copy of the
  work loses to one that gives it *a share of the same* work.

### 4.2 Layer pipeline: CPU lower half, GPU upper half, 2 joins per token: closed by price

- At width 1 the chain is serial, so the two halves never overlap: no gain (GPU alone 24.9 ≈ CPU alone 25.6 GB/s).
- It overlaps only speculatively: an exit head at mid depth guesses token t + 1 and the CPU starts its lower half
  while the GPU finishes token t. Price with **free launches** and the GPU's device time (*estimated*): 8.22 / 7.50 /
  7.39 at exit top-1 0.5; 8.84 / 8.09 / 7.93 at 0.7; 9.57 / 8.78 / 8.55 at 0.9.
- The row-split lane at λ 0.12 ms gives 9.51 / 8.83 / 8.60 with no guesser. The pipeline also needs the whole
  attention path and the KV cache of 16 layers on the GPU, and a mid-depth head nobody has trained. Dominated.

### 4.3 Level stacking: one launch per dependency level (cheap, real)

- Matrices that read the same input can be one GPU buffer and one launch: `gate‖up` as 22,016 × 4,096 and `q‖k‖v` as
  12,288 × 4,096 (Yi: 5,120 × 4,096), built at load in the lane's own copy. The upstream general kernel takes M as
  an argument (general ÷ pick 0.997, *device*), so no kernel change and the same per-row arithmetic.
- A row split then needs 4 launches per layer for all seven matrices, not 7. Whether a level is split at all follows
  from λ: at 0.452 ms only the FFN levels pay on Yi and `o` never pays; at 0.12 ms every level pays (script).
- Price: the first row of §1's table. Against R74's per-matrix mechanism at the same λ (7.64 / 6.80 / 6.77 in my
  model) stacking is worth +0.5 to +0.7 tok/s. Largest buffer 50.7 MB against `max_alloc_mb` 923 (*device*).

### 4.4 FFN column partition: one join for the whole FFN

- Give the GPU a set J of FFN hidden units: rows J of `gate` and `up`, and **columns J of `down`**. A column range in
  multiples of 32 is a clean Q4_0 sub-matrix (blocks run along the input). Each side then computes
  h_J = silu(gate_J·x) × (up_J·x) and its partial `down[:, J]·h_J` without the other; the layer needs one join,
  y = y_cpu + y_gpu.
- GPU chain per layer: 2 to 3 launches (`gate‖up` rows, the GLU, the `down` column block), no host step inside.
- By the launch-level model it does **not** beat §4.3: the cost is per launch, and this has as many launches. Its
  value appears only if X4-a shows the cost is per host round trip after all. It changes the summation order of
  `down` (numerics change beyond the lane's own; host-testable as a pure reorder).
- Kept as a variant, not proposed first.

### 4.5 Persistent GPU worker kernel: one launch per token, joins through shared memory (exotic)

- One NDRange per token (or per few layers) whose work items behave like ggml's thread pool: loop over a job list;
  wait for the job's "input ready" word in a buffer shared with the CPU; claim rows with `atomic_add` on a shared
  counter (the phone has the global int32 atomics); write outputs and a "done" count. The CPU, when its own rows are
  finished, reads the count. No driver call between layers: λ becomes memory latency.
- What must be true, none of it known for Adreno:
  1. a running kernel sees a CPU write to an io-coherent shared buffer (`cl_qcom_ext_host_ptr_iocoherent`). The
     GPU's own cache may return a stale word forever; the work-around is to publish the flag as a stripe and poll a
     fresh address each time;
  2. the CPU sees the GPU's "done" write while the kernel is still running;
  3. the upstream sub-group kernel body keeps its rate inside a loop with claimed rows, and the kernel is not
     treated as hung (each lives about one token, 60 to 150 ms; upstream prefill kernels run that long).
- Prior art: persistent threads and in-kernel CPU notification exist on CUDA (search summary); I found nothing for
  OpenCL on a mobile GPU, and a 2007 vendor forum answer says a kernel cannot poll for new input (old, CUDA).
- Price if all three hold: 10.14 / 9.36 / 9.07 (§1). P(all three hold on the 660) 0.10, my judgement.

### 4.6 Smaller items

- **Launch from the pool's own lead thread on a big core,** then `clFlush` and compute; check completion by event
  status after the CPU rows are done. No feeder thread, no `clFinish`. If F7's hypothesis holds this is most of λ.
- **Launch-ahead on a user event:** enqueue a level's launch with a wait list on a user event while the previous
  level is still running, and complete the event when x is written. Hides queued→submit (143 to 208 µs) only if the
  driver prepares at enqueue; X4-f.
- **Recordable replay** of the per-layer launch set (§3). Cuts launcher CPU time (F6), not submit→start.
- Not proposed: asking for a perf hint (already HIGH); a third position on the GPU (§4.1's reason).

## 5. Bottom line for the stack

Every design here is bounded by R74's zero-dispatch column, and that column's centre stack is 12.23 / 11.87 / 11.44
(*estimated*, lead). So even λ = 0 leaves Llama-2 and DeepSeek under 12 with every unbuilt lever at its centre.
What this round can change is the lane's price inside the stack: from the result's 9.55 / 9.50 / 9.26 towards about
11 if λ falls to 0.05 to 0.12 ms. I have not re-run `bonnie_tree.best` with a launch-level dispatch term; that is
Bonnie's model. Her `bonnie_joins.py` already has the 0.52 weight and the four levels (§6.1).

## 6. For Bonnie (her file was not there; challenges to the four designs as named)

1. **Polling instead of `clFinish`.** The reachable saving per join is end→return, 57 to 172 µs (F2), not the 452.
   And F9 says the CPU already waits only 35 to 68 µs at a balanced share. A band centred above about 100 µs saved
   per join needs a reason.
2. **Pre-enqueue.** R74's sync1 and batch-of-8 cells *are* pre-enqueued launches, and they read 199 to 320 µs each
   (F3, F4). A band below about 200 µs per launch for pre-enqueue on the stock queue needs a mechanism R74 did not
   already run (a faster launcher core, a user event, a recording).
3. **Keep the GPU awake.** The `x3 alone` cells ran at `gpu_busy` 60 to 76 % with launches back to back and still
   show 261 to 320 µs each; submit→start is 184 to 199 µs at every size. That does not look like a sleep state
   (downstream trees put the idle timeout at 80 ms; search summary, not the 888's tree). P it is a power state: 0.2.
   The GPU clock was unreadable in every cell, so a low-clock GPU is not excluded.
4. **Whole-matrix assignment.** It cuts joins, but F3 says the cost is per launch and grows with the matrix; and a
   level with one matrix (`o`, `down`) leaves one side idle, so the body factor is not 0.621. Please price it per
   level with the dead time weighted by r_g ÷ (r_c + r_g), and compare with §4.3 at the same λ.
5. **The result's formula.** `body × lane + n_join × ovh` is additive. The split cell (F9) supports the weighted
   form. If she keeps the additive form, the two positions should stand side by side in the memo.
6. Attack mine: the 0.52 weight assumes the CPU always has rows of the same level; F5's 124 µs rests on one
   unmeasured `dev_us`; P 0.5 for the slow-A55 hypothesis.

### 6.1 Review of `bonnie_joins.py` (script only, read after §1 to §6 were written; I did not run it)

Agreed, and reached independently: four dependency stages per layer, a stage split only where it beats the CPU
alone, and the dead time weighted by κ = 0.521. Her calibration of the split cell uses K 530 µs (0.865 ms); mine uses
452 (0.850 ms). Same model, so E-J5 is already done on her side, and my §1 table should be read as K0 + lane only.

Her K bands (µs per join, centre and range), with my position:

| design | hers | my position |
|---|---|---|
| E, as measured | 500 (452 to 560) | agree |
| B, poll | 470 (400 to 560) | agree. The reachable part is 57 to 172 µs (F2) and the CPU waits 35 to 68 µs already (F9) |
| B+, fast core or recorded queue | 350 (250 to 500) | centre fine; **low end too high**. F5 reads about 124 µs in a cell whose only known difference is the launcher's clock. Low end 150 |
| A, B+ and pre-enqueue behind a gate | 290 (205 to 500) | **not additive with B+.** Both remove launcher work in queued→submit (143 to 208 µs, F2). Her two steps take 180 µs out of that one segment, its whole centre. I would put A at 320 |
| C, A and GPU kept out of idle | 120 (35 to 500) | **centre too low, and it double-counts.** It needs submit→start (184 to 199 µs) to vanish. The only support is F5, and F5 cannot tell a hot launcher (B+) from a hot GPU (C). If the launcher is the cause, C adds nothing to B+; if the GPU is, B+ gives little. One of the two steps is empty. Centre 220, range 100 to 500; the low end 35 needs both causes at once |
| F, persistent kernel | 30 (5 to 100) | agree on K (mine 20). The band is conditional on three unknowns (§4.5); P 0.10 belongs beside it |
| D, whole matrices | K as E / A / C | follows the rows above |

Disagreement to keep in the memo: **her ladder B+ → A → C reaches 120 µs at the centre; mine stops near 220** until
X4-a and X4-b say which of launcher clock or GPU state is behind F5.

## 7. Lists for Tom (through the lead)

### 7.1 Test list

Host (no phone):

| # | test | cost | kill line / closes when |
|---|---|---|---|
| E-J1 | Re-read of the R74 raw lines for the launch decomposition | **done** (§2) | – |
| E-J2 | Does upstream wait per op? | **done** (§3, *source*): no; one blocking read per token | – |
| E-J3 | **Level stacking on the host emulation.** In Tom's half-precision emulation of the kernel (R74 §7 item 1), run `gate‖up` and `q‖k‖v` as stacked buffers and compare per-row outputs with the per-matrix path | a few lines on top of the numerics check | any row differs → stacking is not arithmetic-neutral against the lane; price it as its own numerics change or drop it |
| E-J4 | **Column-partition reorder.** `down` computed as two partial sums over a column split at a multiple of 32, added; per-chunk PPL at `-b 512` and `-ub 1` on two subjects, with the lane's emulation on | one knob, two PPL runs per subject (owner's laptop or host with permission) | outside the campaign budget at either batch size → §4.4 closed. Run only if X4-a makes §4.4 worth it |
| E-J5 | **Launch-level dispatch term in the stack model** (Bonnie): exists in `bonnie_joins.py` (§6.1); its stack column at K 452 / 300 / 120 / 50 goes into the joint memo | **done on her side**, not run by me | centre stack at K 50 under 11.0 on all three → nothing in this memo is worth a device cell |
| E-J6 | Count upstream's kernel launches per token for one subject (a counter in `ggml_cl_compute_forward`, host build with any OpenCL device, or by reading the graph) | under an hour | – (a fact for §3's last reading) |

Device cells (each needs the owner's permission and its own pre-registration; listed so that a later kit can carry
them; a few seconds each, no model loaded):

| # | cell | kill line |
|---|---|---|
| X4-a | **Launch timeline.** On the profiling queue, a 7-launch layer and an 8-launch batch: print QUEUED, SUBMIT, START, END of every event relative to the first, and the launcher's CPU time | gaps END(i)→START(i+1) ≥ 150 µs with the launcher on a big core at full clock → the cost is inside the driver or GPU per launch: pre-enqueue and polling are closed, only fewer launches (Y12) and Y15 remain |
| X4-b | **Launcher core and wait mode.** `x3 rt` at 2,816 rows from cpu 0-3 (as R74), cpu 4 and cpu 7, each after a 0.5 s spin to full clock; wait by `clFinish`, by `clFlush` + spin on the event status, by `clFlush` + callback | best `ovh_us` > 150 µs → λ stays in 250 to 450 µs and §1's lower rows are out of reach by host-side means |
| X4-c | **Stacked launch.** `gate‖up` as one 22,016 × 4,096 buffer on the general kernel: identity line and cold rate | identity `ok=0`, alloc error, or rate < 0.9 × the 11008×4096 cold rate → Y12 closed |
| X4-d | **Recordable queue.** Print `CL_DEVICE_RECORDABLE_QUEUE_MAX_SIZE`; record one layer's launches, replay, wall per layer against sync1 | queue cannot be created, or overhead per launch not under 0.5 × sync1's (293 / 252 / 261 µs) → Y14 closed |
| X4-e | **Persistent kernel** (child process, 2 s timeout): (1) time for a running kernel to see a CPU-written flag, plain and striped; (2) time for the CPU to see a kernel-written count; (3) rate of the looped, row-claiming gemv against `x3 alone` | flag not seen within 1 ms or the child hangs; or (2) median > 50 µs; or (3) < 0.8 × g → Y15 closed |
| X4-f | **Launch-ahead.** Enqueue behind a user event 1 ms early, then complete it; print completion→START | ≥ 300 µs → launch-ahead hides nothing |
| – | Riders with any of the above: launcher thread CPU time in every `rt` cell; GPU clock by any readable node; `x3 split` with levels stacked | facts |

X4-a and X4-b come first; the others depend on their answer.

### 7.2 Try list

All: model-generic (shapes and counts read from the GGUF), env-gated and off by default, at least two subjects.
The lane itself is a **declared numerics change** (half-precision scales, R74 prereg §0); the labels below are
relative to that lane.

| # | lever | after | label | price (K0 + lane, tok/s, *estimated*) | P |
|---|---|---|---|---|---|
| Y12 | **Level-stacked lane buffers**: `gate‖up` and `q‖k‖v` as one buffer and one launch each; a level is split only where the script's rule says it pays at the measured λ | E-J3; X4-c | same arithmetic as the lane (to be shown by E-J3) | 8.13 / 7.51 / 7.42 at λ 0.452; +0.5 to +0.7 over per-matrix launches | 0.6 |
| Y13 | **No feeder, no `clFinish`**: launch from the pool's lead thread on a big core, `clFlush`, compute the CPU rows, then check the event | X4-b under 150 µs | bit-identical to the lane with `clFinish` | towards 9.5 / 8.8 / 8.6 (λ 0.12) | 0.3 |
| Y14 | **Recordable replay** of each layer's launch set | X4-d | bit-identical to the lane | between λ 0.30 and 0.12 rows; the vendor's own figure is "up to ~10 %" (*external*) | 0.25 |
| Y15 | **Persistent GPU worker kernel** with shared-memory handshake | X4-e, all three parts | same kernel arithmetic; row order within a matrix may differ, outputs per row do not | 10.14 / 9.36 / 9.07 | 0.10 |
| Y16 | FFN column partition (one join per FFN) | X4-a says the cost is per host round trip; E-J4 | numerics change (summation order of `down`) | no gain over Y12 by the launch-level model | 0.1 |

Not proposed for a build: the GPU speculative chain (§4.1) and the layer pipeline (§4.2), both closed by price.

## 8. Sources (2026-11-23)

- Repository: `R74-result.md`; `device/r74-c3probe-r34-2026-11-23T1743.json` (raw `x3` lines);
  `prereg/R74-x3-gpu-probe-device.md` §5; `patches/llama.cpp-c3/0005-x3-gpu-probe.patch` (`roundtrip`, `batch`,
  `layer`, `split`, `run_arm`); R72 device JSON (`x2info` extension list); `memo-09-gap.md` §10.7; `memo-11-exotic.md`.
- Upstream tree at base `620d2225c` (2026-10-15), `ggml/src/ggml-opencl/ggml-opencl.cpp`, read locally.
- HeteroInfer, arXiv 2501.14794v2, SOSP 2025 (read through the fetch tool, with quotes): <https://arxiv.org/html/2501.14794v2>
- H. Wang et al., "Advancing OpenCL-Based LLM Inference in Llama.cpp on Qualcomm Adreno GPUs", IWOCL 2026 slides
  (text extracted from the PDF): <https://www.iwocl.org/wp-content/uploads/IWOCL-2026-Wang-Llamacpp.pdf>
- MNN `OpenCLRuntime.cpp`, master (read through the fetch tool):
  <https://raw.githubusercontent.com/alibaba/MNN/master/source/backend/opencl/core/runtime/OpenCLRuntime.cpp>
- `cl_qcom_perf_hint` (read): <https://registry.khronos.org/OpenCL/extensions/qcom/cl_qcom_perf_hint.html>;
  registry directory (read): <https://registry.khronos.org/OpenCL/extensions/qcom/>
- Qualcomm OpenCL ML SDK post, recordable queues named (search summary):
  <https://www.qualcomm.com/developer/blog/2021/05/accelerate-your-models-our-opencl-ml-sdk>
- kgsl dispatcher and power settings in downstream trees (search summary only):
  <https://android.googlesource.com/kernel/msm/+/a80b2e0035e75c0460b54e2217ca0104631d547d/drivers/gpu/msm/adreno_dispatch.c>
- Persistent kernels and in-kernel notification on CUDA (search summary only): <https://arxiv.org/abs/2204.02064v2>,
  <https://mediatum.ub.tum.de/doc/1743007/1743007.pdf>,
  <https://forums.developer.nvidia.com/t/using-gpu-cpu-polling-to-reduce-overhead/1870>
- arXiv 2607.05475, 2605.27435, 2604.25326, 2607.12839 (search summary this round; the first two were read for memo-11).

Not done: no device run; MLC / TVM, ExecuTorch / QNN and PowerInfer-2 not read; the Adreno OpenCL SDK guide and the
recordable-queue specification not opened; upstream's launches per token not counted; `bonnie_tree.best` not re-run
with a launch-level term; Bonnie's `bonnie-joins.md` not reviewed because it did not exist (her script was read, not run).
