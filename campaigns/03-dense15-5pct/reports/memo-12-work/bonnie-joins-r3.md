# c3 memo-12 work, round 3 (Bonnie): x4m and x4s bands, the corrected stack model, residency above 1.5 GB

- **Date:** 2026-11-23. **Author:** bonnie, `claude-fable-5-2`. Follows [bonnie-joins-r2.md](bonnie-joins-r2.md)
  (committed) and [ebony-joins-r2.md](ebony-joins-r2.md) part A. Not committed. No verdict here.
  **Ebony has not reviewed this file.**
- **Labels.** *device* = printed by the phone (R74, or R72 / R69 where named). *derived* = arithmetic on *device*
  fields. *estimated* = model or judgement. *source* = read in the local upstream tree at the campaign base.
- **Scripts** (`python3 -I`, self-checks pass): [bonnie_tree_r2.py](bonnie_tree_r2.py) (item 2),
  [bonnie_joins_r3.py](bonnie_joins_r3.py) (items 1 and 3).
- Triples are Yi / Llama-2 / DeepSeek. 12 tok/s = 79.17 ms. K = GPU dead time per split stage, µs.
  cg = GPU time per byte at two columns ÷ at one column.

## 0. Answer first

1. **x4m:** `dev_ratio` at 2 / 3 / 4 columns: **1.2 / 1.5 / 1.9** (80 % bands 1.05–1.75 / 1.15–2.5 / 1.35–3.3).
   P(the kernel is created and runs with `err=0`) 0.85; P(that and identity on every column) 0.75.
2. **cg lines:** `dev_ratio(2)` ≤ 1.30 makes 12 on all three priceable in both stack models; above 1.7 only the
   most favourable model has a line, and only at K near 0; above 2.17 none has. Width 2 stays better than width 1
   on the lane up to cg 1.67 at worst, so cg does not kill width 2; it kills the route to 12.
3. **x4s over 0.20 to 0.60:** best ratio **0.73** (0.70 to 0.79) at share 0.50 (P 0.78; 0.45: 0.13; 0.55: 0.09).
   Shares 0.55 and 0.60 are GPU-bound in 9 worlds of 10, so they read K directly.
4. **`bonnie_tree_r2.py`** carries three named variants: `old` (as committed, reproduces R74-result §4), `bonnie`
   (reproduces r2 §3.1: 12.60 / 12.24 / 11.77) and `pair` (reproduces Ebony's 14.89 / 14.57 / 13.84). It prices K
   inside each variant, which r2 did not: the r2 lines move a little (cg 1.2: 12 at K ≤ 194 / 170 / 97, was 209 /
   183 / 104).
5. **Residency:** the balanced GPU share falls as K rises. At K 300 the lane wants 1.0 to 1.5 GB of GPU buffers
   on every subject; x4r's 1.5 GB covers that. Only a lane with K near 0 and cg near 1.1 wants 1.7 to 2.0 GB on
   Llama-2 and DeepSeek, which nothing measured covers. A cap at 1.5 GB costs nothing at K 300; a cap at 0.85 GB
   (R74's arm) costs 0.9 to 1.7 tok/s.

## 1. Bands for the pre-registration (all *estimated*; centre and 80 % band)

### 1.1 x4m: upstream `_mc3` on 11008×4096 cold, GPU alone

Model: the GPU as a two-term reader like the CPU (time per byte = max(bus, compute) + 0.25 × min). Inputs: GPU
read-only 31.14 GB/s alone and 25.81 beside the CPU (*device*, R72); gemv device rate 28.32 (*device*, R74). The
kernel shares the weight loads and repeats dequantise and accumulate per column (*source*): compute per byte
× 1.9 / 2.8 / 3.7 at 2 / 3 / 4 columns.

| world | `dev_ratio` alone, 2 / 3 / 4 columns | cg in the lane (beside the CPU), 2 / 3 columns |
|---|---|---|
| compute as the two-term fit says (78 GB/s) | 1.08 / 1.24 / 1.57 | 1.07 / 1.14 |
| compute twice as dear | 1.47 / 2.07 / 2.67 | 1.29 / 1.80 |
| compute three times as dear, or the kernel compute-bound at 28.3 already | 1.73 / 2.47 / 3.20 | 1.61 to 1.71 / 2.28 to 2.42 |

| event or field | prediction | note |
|---|---|---|
| `clCreateKernel("kernel_gemv_noshuffle_q4_0_f32_mc3")` succeeds | P 0.92 | the program holding it built on the phone in R74 (`x3 build rc=0`, *device*); the kernel was never created |
| runs at n_cols 2 with `err=0` | P 0.85 | risks: sub-group size and local size on this driver, compiler E031.38 |
| identity `ok=1` on every column, given it runs | P 0.88 | upstream's comment claims per-column identity with the one-column kernel |
| all of the above | P 0.75 | – |
| `dev_ratio` n_cols 2 | **1.2** (1.05 to 1.75) | P(≤ 1.20) 0.50; P(≤ 1.30) 0.58; P(> 1.70) 0.12 |
| `dev_ratio` n_cols 3 | 1.5 (1.15 to 2.5) | – |
| `dev_ratio` n_cols 4 | 1.9 (1.35 to 3.3) | – |
| `wall_ratio` n_cols 2 | 1.15 (1.04 to 1.60) | (896 × ratio + gap) ÷ (896 + gap), gap 125 to 311 µs |
| max_rel_err per column, given identity | 6e-4 to 8e-4 | R74 identity lines 6.2e-4 to 7.6e-4 (*device*) |

Reading rule: **the lane's cg is at most the alone `dev_ratio`** (the lane's bus share is smaller, so compute
hides better). Use cg = `dev_ratio(2)` as the conservative value; the value beside the CPU needs a `both` arm.

### 1.2 cg lines (width-2 pass on the lane; trained head, C7, C5, O 5.2 at their centres)

cg needed for 12.00 on Yi / Llama-2 / DeepSeek, from `bonnie_tree_r2.py`:

| K µs | `bonnie` (floor) | `pair` (Ebony) | `pair` with her 60 µs slack | `pair` at C 35 |
|---|---|---|---|---|
| 0 | 1.66 / 1.53 / 1.38 | 2.93 / 2.56 / 2.17 | 2.11 / 1.95 / 1.67 | 2.37 / 2.13 / 1.86 |
| 150 | 1.31 / 1.24 / 1.10 | 2.00 / 1.85 / 1.55 | 1.55 / 1.50 / 1.18 | 1.67 / 1.58 / 1.36 |
| 300 | none | 1.12 / 1.13 / none | none | none / 1.02 / none |
| 452 | none | none | none | none |

Lines for the pre-registration:

| `dev_ratio(2)` | meaning |
|---|---|
| ≤ 1.10 | 12 on all three has a join line of 150 µs in the floor model and 239 / 164 µs in `pair` without / with slack |
| ≤ 1.30 | **worth a build:** both models have a line for all three (floor: K ≤ 43 µs, DeepSeek binds; `pair`: K ≤ 213, or 130 with slack). The lane's GPU side moves to `_mc3` for the width-2 pass |
| 1.30 to 1.70 | a line exists only in `pair` (K ≤ about 150 at 1.55; K near 0 at 1.67 with slack). Not priceable without the `tog` cell at width 2 that decides between the models |
| > 1.70 | no line with slack in any model; > 2.17 none at all. **12 through the lane at width 2 is closed**; the lane stays a 10 to 11 tok/s lever |
| ≥ 2.0 | `_mc3` is no better than two launches; drop it |

- Width 2 against width 1 on the lane: width 2 wins up to cg 1.67 / 2.10 / 1.84 at K 0 in the floor model, and
  further at higher K and in `pair` (2.4 and above). **No cg inside my band kills width 2 itself.**
- At K 300, `dev_ratio` 1.0 to 2.0 moves the stack from 11.79 / 11.78 / 11.34 to 10.10 / 9.90 / 9.63 (floor) and
  from 12.10 / 12.13 / 11.65 to 11.05 / 10.95 / 10.59 (`pair`). So cg is worth about 1.1 to 1.7 tok/s at the K
  R75 will most likely show.

### 1.3 x4s lead mode, shares 0.20 to 0.60 (replaces r2 §2.2's "best ratio in 0.20 to 0.50" row)

Duty-aware split model of r2 (fitted to R74's three shares), 0.03 ms of join slack added, K and launch time drawn
from the x4b Monte Carlo (K 300, band 200 to 380).

| field | centre | 80 % band | note |
|---|---|---|---|
| best ratio over 0.20 to 0.60 | **0.73** | 0.70 to 0.79 | – |
| best share | 0.50 (P 0.78) | 0.45 (P 0.13), 0.55 (P 0.09) | 0.55 is best only if K ≤ about 150 |
| share 0.55: ratio; `gpu_ms`; `wait_ms`; `gpu_dev_ms` | 0.78; 1.38; 0.21; 1.08 | 0.72–0.84; 1.28–1.46; 0.11–0.29; 0.98–1.18 | GPU side binds in 9 of 10 worlds |
| share 0.60: ratio; `gpu_ms`; `wait_ms`; `gpu_dev_ms` | 0.83; 1.48; 0.39; 1.18 | 0.77–0.90; 1.38–1.56; 0.29–0.47; 1.07–1.29 | GPU side binds |
| P(the optimum is interior: the ratio rises on both sides of the best share) | 0.90 | – | was about 0.35 with the sweep ending at 0.50 |

By K (ratio at the best share): K 500 → 0.78 at 0.45; K 300 → 0.73 at 0.50; K 200 → 0.72 at 0.50; K 120 →
0.68 at 0.55; K 30 → 0.66 at 0.55. Reading rule unchanged: K = `gpu_ms` − `gpu_dev_ms` at a share where the GPU
side binds (`wait_ms` > 0.1).

## 2. `bonnie_tree_r2.py`

- Loads `bonnie_tree.py` and `bonnie_joins.py` by path; neither is changed.
- One stage model for every variant: the CPU runs at its alone rate while the GPU is dead and at its contended
  rate while the kernel runs; t_split = K + (B − ca × K) ÷ (cb + g) + slack; a stage is split only if that beats
  the CPU alone. Slack = J0 + J1 × K (mine 0 + 0.107 K; Ebony's 60 µs + 0).
- Variants: `old` (through `bonnie_joins.price`); `bonnie` (CPU bus share fixed at its width-1 value, GPU wall
  rate ÷ cg; option `gdev=True` for the device-time rate); `pair` (bus sum 46.1 GB/s, GPU device-time rate ÷ cg).
- Self-check: R74-result §4 (12.23 / 11.87 / 11.44; 9.55 / 9.50 / 9.26); r2 §3.1 (12.60 / 12.24 / 11.77 at cg
  1.45; 13.58 / 13.24 / 12.66 at cg 1.2); Ebony A.6 (rates 21.4 + 19.5 = 40.9 at width 2, 43.5 at width 1;
  14.89 / 14.57 / 13.84 at K 0).

| K µs | `old` | `bonnie` cg 1.45 | `bonnie` cg 1.2 | `pair` cg 1.26 | `pair`, slack 60 | `pair`, C 35 |
|---|---|---|---|---|---|---|
| 0 | 12.23 / 11.87 / 11.44 | 12.60 / 12.24 / 11.77 | 13.58 / 13.24 / 12.66 | 14.89 / 14.57 / 13.84 | 13.65 / 13.48 / 12.87 | 14.63 / 14.30 / 13.61 |
| 100 | 11.57 / 11.30 / 10.92 | 11.93 / 11.66 / 11.25 | 12.72 / 12.48 / 11.98 | 13.73 / 13.55 / 12.93 | 12.85 / 12.76 / 12.22 | 13.45 / 13.26 / 12.67 |
| 200 | 10.98 / 10.78 / 10.44 | 11.33 / 11.14 / 10.76 | 11.95 / 11.80 / 11.37 | 12.74 / 12.66 / 12.13 | 12.14 / 12.11 / 11.64 | 12.44 / 12.36 / 11.86 |
| 300 | 10.51 / 10.37 / 10.06 | 10.80 / 10.66 / 10.33 | 11.28 / 11.20 / 10.81 | 11.89 / 11.88 / 11.43 | 11.57 / 11.59 / 11.17 | 11.57 / 11.57 / 11.15 |
| 452 | 10.13 / 9.87 / 9.60 | 10.34 / 10.16 / 9.87 | 10.69 / 10.57 / 10.25 | 11.12 / 11.08 / 10.71 | 11.05 / 10.98 / 10.62 | 10.67 / 10.71 / 10.37 |
| K for 12.00 | 33 / none / none | 89 / 41 / none | 194 / 170 / 97 | 286 / 284 / 218 | 220 / 218 / 137 | 249 / 244 / 182 |

- Difference from Ebony's own table: at K 452 her file reads 11.65 / 11.66 / 11.23; this script reads 11.12 /
  11.08 / 10.71 for the same variant because it adds my slack (0.107 K) and lets the CPU run at its alone rate
  only during the dead time. With her slack instead: 11.05 / 10.98 / 10.62, close to her "with jitter" column.
- Width 1 on the lane (no trained head): `bonnie` 11.97 / 10.73 / 10.84 at K 0, no K reaches 12; `pair` 12.54 /
  11.26 / 11.34 at K 0, Yi at K ≤ 52.
- **Most likely R75 outcome (K 300) priced in the three forms: 10.5 / 10.4 / 10.1 (`old`), 10.8 to 11.3 / 10.7 to
  11.2 / 10.3 to 10.8 (`bonnie`, cg 1.45 to 1.2), 11.6 to 11.9 / 11.6 to 11.9 / 11.2 to 11.4 (`pair`).**
  No variant reaches 12 on DeepSeek at K 300.

## 3. Residency above 1.5 GB

### 3.1 What a lane holds

- A static row share moves bytes, it does not add them: the GPU copy (the kernel's own layout, built on the phone
  at load) holds share f of the body; the CPU reads the other 1 − f from the file mapping and never touches the
  GPU's rows. Total resident = the stock decode's bytes: body 3.11 / 3.64 / 3.42 GB per token (*device*, R72), plus
  the Q6_K head and embeddings; R74-result §5 uses 3.36 / 4.0 / 4.0 GB.
- What changes is the kind: f of the body becomes driver memory that the kernel cannot drop; the rest stays clean
  file pages that it can. Under pressure the file half takes all the reclaim.
- Building the GPU copy needs each matrix's file pages and its GPU buffer at once, one matrix at a time (at most
  51 MB for a stacked level); the file pages of GPU rows can then be dropped (`madvise`), inside the runtime.
- A share that is balanced at run time needs the overlap rows in both places (Ebony A.5 item 2): not priced here.

### 3.2 How much the GPU wants (balanced share, width 2; *estimated*, `bonnie_joins_r3.py`)

| model, cg | K 0: share; GB | K 300: share; GB |
|---|---|---|
| `bonnie`, 1.45 | 0.44; 1.38 / 1.61 / 1.51 | 0.30–0.32; 0.93 / 1.17 / 1.09 |
| `bonnie`, 1.2 | 0.49; 1.52 / 1.78 / 1.67 | 0.34–0.36; 1.06 / 1.32 / 1.21 |
| `pair`, 1.26 | 0.48; 1.48 / 1.74 / 1.63 | 0.33–0.35; 1.04 / 1.29 / 1.21 |
| `pair`, 1.08 | 0.54; 1.69 / 1.97 / 1.85 | 0.38–0.40; 1.18 / 1.46 / 1.37 |

- **The 1.6 to 1.9 GB of my r1 table is the K 0 case.** At the K R75 will most likely show, every subject is
  under 1.5 GB. The need rises only as the join gets cheap.
- Cost of a cap on GPU buffers (`pair`, cg 1.26, K 300; stack tok/s): cap 1.5 GB: no loss; cap 1.2 GB: 11.89 /
  11.50 / 11.40 against 11.89 / 11.88 / 11.43; cap 0.85 GB: 11.03 / 10.19 / 10.07.

### 3.3 What the capacity data allows

| arm | held bytes | result | label |
|---|---|---|---|
| R69 P1 | 4.2 GB file-backed | held (capacity C) | *device* |
| R72 anon34 / anon42 | 3.4 / 4.2 GB anonymous | held / not held | *device* |
| R74 x3r | 0.85 GB GPU + 2.6 GB anonymous = 3.45 GB | held 15 s; passes 91 to 99 slowed, memavail down to 0.59 GB, majflt 0 | *device* |
| R75 x4r | 1.5 GB GPU + 2.6 GB anonymous = 4.1 GB | pending; my P(held) 0.35 | – |

- Yi (3.36 GB, GPU 0.9 to 1.5 GB): inside what x3r held. P(holds in a decode) about 0.8 (*estimated*).
- Llama-2 and DeepSeek (4.0 GB, GPU 1.1 to 1.5 GB at K 300): between x3r (held, with pressure showing) and anon42
  (not held). The nearest evidence for them is R69's 4.2 GB file-backed. A lane's mix (1.1 to 1.5 GB pinned + 2.5
  to 2.9 GB file-backed) has not been run. P(holds) about 0.4 (*estimated*).

### 3.4 What x4r can and cannot settle

- **If it holds:** 4.1 GB with 1.5 GB pinned stays resident for 15 s. That covers Llama-2's and DeepSeek's totals
  and their GPU share up to K about 300 at any cg, with anonymous memory standing in for the file half. It is
  evidence for, not proof: the app's own memory, a decode's duration and file-backed reclaim are outside the arm.
- **If it does not hold:** it does not close the two subjects. Anonymous pages cost a compress and a fault where a
  clean file page costs a drop and a read; R69 held 4.2 GB file-backed where R72 lost 4.2 GB anonymous. It would
  say a lane needs its CPU half file-backed (no repack of the CPU rows on those two subjects).
- **Cannot settle either way:** GPU buffers of 1.7 to 2.0 GB (the K near 0, cg near 1.1 case); kgsl limits above
  1.5 GB (`global_mb` 3,693 and `max_alloc_mb` 923, *device*, say the allocation itself should pass); behaviour over
  minutes; the mix with file pages.
- **The arm that would settle it** (not in R75): GPU buffers at 1.5 and 1.9 GB beside a **file-backed** read of
  2.5 GB of a scratch file, 60 s. It is R69 P1's file plus x3r's GPU ring.

## 4. Next questions, and what I did

1. **Head on the lane** (r2 §3.3): worth 1.7 / 0.8 / 2.6 ms after C7, one join per token, so K costs it under
   0.3 ms. Upstream has `gemv_noshuffle_q6_k_f32.cl` and a `q6_K ... _mc3` pair for two columns (*source*, names
   only; bodies not read). Open, for Tom on the host: its numerics. The Q4_0 kernel forms (q − 8) × scale in half
   precision; if the Q6_K kernel forms its 6-bit weight × two-level scale in half too, the head's error is larger
   than the body's (64 levels × 16 sub-scales). The emulation check of R74 §7 item 1 should carry a Q6_K arm
   before the head goes on the lane.
2. **O** is still the 8 → 5.2 ms allowance. Near 12, 1 ms of O is 0.11 to 0.15 tok/s; O 5.2 → 4.0 moves DeepSeek
   by +0.12. No host route to measure it; R73 is held. Unchanged.
3. **Which model, `bonnie` or `pair`?** They differ by 0.6 to 1.1 tok/s at K 300. The deciding cell is Ebony's
   `tog` at width 2 (A.6): the CPU's two-position kernel beside `_mc3`. x4m gives cg alone; it does not give the
   CPU's rate beside a slowed GPU. New question for the lead: whether that cell rides on a later kit.
4. **Slack: scales with K or not?** x4b's p90 − p50 in the spin cells (my band 90, 30 to 250) and x4s `wait_ms` at
   the best share (0.04, 0 to 0.20). Under 50 µs of spread supports mine.
5. **New: three columns.** If `dev_ratio(3)` ≤ 1.3 the lane makes a third position cheap on the GPU side, and
   memo-10's "width 2 is best" was priced with the CPU alone. Not priced here; `bonnie_tree_r2.py` covers widths
   1 and 2 only. Worth doing only if x4m shows it.
6. **New: clocks.** Both variants lose their line at C 24 (L4 caps, X1 thread on a small core). The lane halves
   the CPU's work per token, which should lower heat and lift the caps; nothing prices that feedback.

## 5. Limits

- cg bands rest on a two-term model of the GPU fitted to two rates; the worlds in §1.1 are my judgement.
- `pair` is my re-implementation of Ebony's model from her text; it matches her three published checks, not her
  script line by line.
- Every stack number still needs the trained head at width 2, C7, C5 and O 5.2 at their centres; none is built.
- §3's probabilities are judgements on four capacity arms, none of which is a decode.

## 6. Sources

Repository: R74 and R72 JSON and results, R69-result (P1), [bonnie-joins-r2.md](bonnie-joins-r2.md),
[ebony-joins-r2.md](ebony-joins-r2.md) A.2 to A.7, [tom-r35-build.md](tom-r35-build.md) (x4s arm order, scope).
Upstream at the base (*source*, `lab/llama.cpp-c3/ggml/src/ggml-opencl`): `kernels/gemv_noshuffle_q4_0_f32.cl`
(`_mc3`, lines 290 to 400), `ggml-opencl.cpp` lines 1225 to 1278 (kernel names). No new web source this round.
