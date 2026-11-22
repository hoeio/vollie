# Pre-registration R74: x3 GPU lane probe (upstream Adreno Q4_0 gemv; c3 probe kit r34)

This file is committed and pushed **before** the APK is uploaded, and it is not edited afterwards. Any change goes
into `R74-amendment-1.md`. The tag `prereg-R74` is the owner's to create. The commit hash and the push time show the
order.

- **Date / author:** 2026-11-22, clyde (lead).
- **Why this round exists:** the owner holds device rounds until a real breakthrough, but allowed this one GPU
  measurement as an exception (owner, 2026-11-22: "GPU 측정은 예외로 허용합니다. x3 프로브 준비하세요."; CAMPAIGN.md, end,
  commit `cf059df`). R73 stays reserved for the held first real decode, so this round is **R74**. Scope: the GPU lane
  only. No model is loaded.
- **Agent models** (full IDs):
  - lead (clyde): `claude-opus-5-5`, this file;
  - bonnie: `claude-fable-5-2`, G lane price and band (`memo-09-gap.md` §4, §5.2; `memo-10-tree.md` §5.3);
  - ebony: `claude-fable-5-2`, G review, dispatch and residency costs (`memo-09-gap.md` §10.7, §10.10 T3);
    R72 joint reading (`R72-result.md` §11);
  - tom: `claude-opus-5-5`, x3 design (`memo-09-work/tom-host.md` §1), patch 0005, app 0004, kit r34 and parser
    (`memo-09-work/tom-r34-build.md`, commit `6732767`);
  - annie: `claude-sonnet-5-5`, books.
- **Subjects** (memo-06 §6): Yi-1.5-6B-Chat, Llama-2-7b-chat, DeepSeek-LLM-7B-chat. They enter only as matrix shapes
  (4096×4096, 512×4096 for Yi's k and v, 11008×4096, 4096×11008) and layer counts (32 / 32 / 30).
- **Goal context:** 12 tok/s printed on each subject = 79.2 ms per token (CAMPAIGN.md Amendment 1). R74 is not a
  claim round.
- **Labels:** every number in §4 and §5 is *estimated* until the result marks it *device*.
- **Starts from:** kit r33 (R72) and campaign 2's kit protocol (r16 cool-down gate before each block, heat, clock,
  power and charger records); campaign 1's OpenCL compiler helper `tool/clc.c` (shipped again as `libvollie_clc.so`).

## 0. Build under test

Hashes re-computed by the lead on 2026-11-22 from the files in `lab/dist-c3/r34/` and `patches/` (`shasum -a 256`);
all match `tom-r34-build.md`.

| part | file | sha256 |
|---|---|---|
| APK | `vollie-c3-probe-r34.apk`, 16,564,106 B, package `com.insaneutils.vollie.c2`, versionCode 34, versionName 1.34.0, same debug certificate as r33 | `2b0a292978913c3d546d798aac895b414f1845b655133edc5d0eb2a5fcbe81de` |
| BUILD file | `BUILD-r34.txt`, 5,555 B | `caf3ab6614d78f13e9452a5d6b8bc97afb9d01e3d5149b311db6fcd08cbe2c5f` |
| sums | `SHA256SUMS` (lists the APK and BUILD file above) | `d4897c69739bb1879ce309c2e26248816418b783e0dec552d38619e2f0c035f8` |
| runtime probe | `patches/llama.cpp-c3/0005-x3-gpu-probe.patch` (only `src/llama-vollie-probe.cpp`) | `6fbb4ec83ea53a6fd39cde94d24b6df3102784cd3c4a7820dbb63bfc6f0e753c` |
| app | `patches/app-c3/0004-kit-r34-probe.patch` | `632a589de7c7a3084bfdaa656d2ac9edb8c290f2a40ada27b490257e2ba7b1ec` |
| libllama.so | in the APK (c3 0001, 0002, 0003, 0005) | `bbb3ee244bf3801e46cef904ac41b300f1e3b9db804f0fb657e2c65f110841bc` |
| libggml-cpu-android_armv8.2_2.so | in the APK, byte-identical to r33 | `46e0735d2824ba6f3a156e219e058a701f6024fdbecbb2ef937b3127115a9694` |
| upstream kernels, verbatim at base `620d2225c` | `gemv_noshuffle_q4_0_f32.cl` (22,301 B), `gemv_noshuffle_q4_0_f32_spec.cl` (16,486 B) | `636126fd…cdf65`, `3f0b490a…00711` (printed by `x3 src` on the phone) |

- **Identity of the CPU module** (*host*): the CPU module ships byte-identical from r33, so 0003's knob-unset
  per-chunk PPL identity at `-b 512`, `-ub 32` and `-ub 1` carries over. The probe runs no model.
- **Numerics:** none changed in R74. A GPU lane built later is a **declared numerics change** (the kernel forms
  `(q − 8) × d` in half precision against f32 activations; `tom-host.md` §1.4) and would need the +5 % budget at
  `-b 512` and `-ub 1`.
- **Parser:** `tools/c3_probe_parse.py` (`x3report`, `--selftest` passes, *host*).

## 1. Device state

- The owner's Galaxy Z Flip 3, power saving **off**, charger **out**; both are recorded.
- The deep-sleeping list label is chosen in the app and recorded.
- **Cool start:** the phone rests at least 30 min, screen off and not in use, before the run. The r16 gate (cap
  180 s) runs before each block.
- No model is loaded.
- Kit r34 is installed over kit r33 (versionCode 34 > 33, same certificate; no uninstall).

## 2. What is measured (blocks as built; run order x3b, x3g, x3c, x3r)

| block | content |
|---|---|
| `x3b` | `x3 cond`, `x3 info`, `x3 src`; `x3 build` for four kernels (general; spec 4096×4096, 4096×11008, 11008×4096), each in a child process via `libvollie_clc.so`, try 1 with upstream's options, try 2 with `-cl-opt-disable`; `x3 buildlog`; `x3 load`; six `x3 identity` lines against float64 (512×4096, 2816×4096, 11008×4096 on the general kernel; the three spec shapes on upstream's pick) |
| `x3g` | `x3 alone` per shape (512×4096, 4096×4096, 4096×11008, 11008×4096) hot and cold (cold = ring of about 512 MB), plus the general kernel on 11008×4096 cold; `x3 rt` round trip at 512 / 1024 / 2816 / 4096 rows; `x3 layer` and `x3 layersum` per subject (4-layer ring, sync1 and sync7, device time) |
| `x3c` | `x3 tog` ABBA cpu, gpu, both, both, gpu, cpu (ggml Q4_0 on 4-7, about 604 MB cold set; GPU 11008×4096 cold; feeder on 0-3) and `x3 togsum`; `x3 split` at GPU share 0.25 / 0.35 / 0.50 of one 11008×4096 matrix and `x3 splitsum` |
| `x3r` | 0.9 GB of GPU buffers held and read by the GPU while 3 A78 (4-6) read a 2.6 GB anonymous arena for 15 s (or 200 passes); per-pass counters; `x3 res sum` |

Recorded with every timed arm: CPU policy clock ranges against scaling and hardware max, `ramp_ok`, kgsl GPU MHz,
thermal headroom, per-thread `thr` (the parser's M / U / P / C labels).

**Phone time** (*estimated*, `tom-r34-build.md`): about 2 to 2.5 min of blocks plus four gates of up to 180 s each,
so about 3 to 15 min.

## 3. Budget accounting

No numerics change and no model run. Nothing is spent.

## 4. Predictions (*estimated*; centre with 80 % band)

**Sources.** g band 8 (4 to 15) GB/s merged in `memo-09-gap.md` §10.7 (Bonnie and Ebony agreed); dispatch 0.1 to
0.2 ms per launch (Ebony, §10.7); CPU loss 0.36 GB/s per GPU GB/s and the CPU + GPU read sum 46.5 GB/s (R72 x2,
*device*); GPU read alone 31.1 GB/s and the probe's own q4 kernel 1.88 GB/s (R69 and R72, *device*); ggml Q4_0 cold on
4-7 26.67 GB/s (R72 L2, *device*); compile risk from `tom-host.md` §1.2 and §1.3 (R72 `x2r` err −6, *device*;
campaign 1 E031.38 compiler segfaults on some newer kernels; campaign 1 ran the OpenCL backend on this phone at 0.7
to 2.9 tok/s, *device*, with the gemv's own status not recorded). Byte sizes are Q4_0 (18 B per 32 weights).

### 4.1 Build and identity (probabilities, lead's judgement on the sources above)

| event | P |
|---|---|
| the general kernel reaches `x3 load ok=1` (try 1 or try 2) | 0.60 |
| at least one of the four kernels reaches `x3 load ok=1` (**P(build ok)**) | 0.70 |
| all four reach `x3 load ok=1` | 0.45 |
| every built kernel's identity line is `ok=1`, given it built | 0.85 |
| all four decision rules of §5 pass (lane **open**) | 0.20 |

### 4.2 Measured quantities

| row | quantity (cell) | prediction |
|---|---|---|
| B1 | compile seconds per kernel, the successful try (`x3 build ... how=`) | 4 (1 to 30) s |
| B2 | `x3 load` `wg_max` | ≥ 256 (P 0.9) |
| B3 | identity `max_rel_err`, 11008×4096 on upstream's pick | 1e-3 (1e-4 to 5e-3) |
| G1 | **g: GPU alone, 11008×4096, cold, upstream's pick (`x3 alone` gbs)** | **8 (4 to 15) GB/s** |
| G2 | GPU alone, 11008×4096, hot ÷ cold | 1.05 (0.95 to 1.40) |
| G3 | GPU alone cold, 4096×4096 / 4096×11008 | 7.5 (3.5 to 14) / 7.5 (3.5 to 14) GB/s |
| G4 | GPU alone cold, 512×4096 (general) | 4 (1.5 to 9) GB/s |
| G5 | general ÷ pick on 11008×4096 cold | 0.95 (0.70 to 1.05) |
| G6 | `dev_gbs` ÷ `gbs`, 11008×4096 cold | 1.05 (1.00 to 1.30) |
| G7 | g ÷ the probe's own q4 kernel (1.88 GB/s, R72) | 4.3 (2.1 to 8.0) |
| D1 | **dispatch: `x3 rt` `ovh_us` at 2816 rows** | **150 (50 to 400) µs** |
| D2 | `x3 rt` `dev_us` at 2816 rows (6.5 MB) | 810 (430 to 1,600) µs |
| D3 | `x3 layersum` `rt_us_per_dispatch` (any subject) | 150 (50 to 400) µs |
| L1 | `x3 layersum` `body_ms_sync1`: Yi / Llama-2 / DeepSeek (layer 97.3 / 113.8 / 113.8 MB × 32 / 32 / 30 at g) | 390 (210 to 780) / 455 (245 to 910) / 427 (230 to 855) ms |
| C1 | `x3 togsum` `cpu_alone_gbs` (ggml Q4_0, 4-7) | 26 (21 to 29.5) GB/s |
| C2 | `x3 togsum` `both_cpu_gbs` / `both_gpu_gbs` | 23 (18 to 28) / 7.5 (3.5 to 14) GB/s |
| C3 | **`x3 togsum` `sum_vs_cpu_alone`** | **1.17 (1.04 to 1.35)** |
| S1 | `x3 splitsum` best `ratio` (min over shares) | 0.97 (0.82 to 1.20), at share 0.25 |
| R1 | `x3 res sum` `cpu_gbs_median` (3 A78 read beside the GPU) | 25 (19 to 28.5) GB/s |
| R2 | x3r median majflt per pass | 0 (0 to 5) |

How each row is scored:
- A value inside its band is "inside"; outside is a **miss**, high or low.
- A row whose cell did not run (no kernel, alloc error, killed block) is "not measured", not a miss.
- The §4.1 probabilities are scored by the event's outcome and reported; they are not bands.

## 5. Decision rules (scored as written, in this order)

**Definitions** (from the phone's own lines):
- *Built kernel*: a kernel with an `x3 load ... ok=1` line.
- *Pick*: the kernel the probe uses for a shape (`kern=` in the cell; the spec kernel if it built, else general).
- *g_cold(s)*: for subject s, the layer-equivalent GPU-alone cold rate over its seven matrices,
  g_cold(s) = layer bytes ÷ Σ (bytes_i ÷ gbs_i), with gbs_i the `x3 alone` cold `gbs` of matrix i's shape on its
  pick (Yi: q, o 4096×4096; k, v 512×4096; gate, up 11008×4096; down 4096×11008. Llama-2 and DeepSeek: q, k, v, o
  4096×4096; the same FFN). Llama-2 and DeepSeek have the same g_cold.

1. **Build.** Kill the GPU lane if no kernel is built (all four `x3 build` lines end `how=fail` after both tries, or
   no `x3 load` reads `ok=1`). Rules 2 to 4 are then "not measured".
2. **Identity.** A built kernel whose `x3 identity` line reads `ok=0` is closed, whatever its speed, and its cells
   are not used. Kill the GPU lane if the pick for 11008×4096 (the shape of x3c and x3r) fails identity, or if no
   built kernel passes.
3. **GPU alone, cold, subject shapes.** Kill the lane for subject s if g_cold(s) < 7 GB/s. Kill the GPU lane if
   g_cold < 7 GB/s on every subject. A lane killed on some subjects and not others is reported as "did not
   generalise", open only where it passed. A shape whose cold cell did not run makes g_cold(s) "not measured" for
   every subject that uses it; the 11008×4096 cell (G1, `tom-host.md` §1.5 item 9) is reported beside it.
4. **CPU + GPU together.** Kill the GPU lane if `x3 togsum` `sum_vs_cpu_alone` < 1.15 (R72 §11, `tom-host.md` §1.5
   item 9).
5. **Open.** If rules 1 to 4 all pass (for a subject: rules 1, 2, 4 and its rule 3), the lane is **open**. Open
   means only: worth a host numerics check (a C emulation of the kernel's half-precision `(q − 8) × d` inside the
   CPU backend behind a knob, per-chunk PPL at `-b 512` and `-ub 1` on two subjects) and a lane design (static row
   share, joins per layer, the CPU never touching the lane's rows). It is not a device go for the lane.

**What feeds the K0 stack price per subject** (computed in the result for every outcome where rules 1 and 2 pass;
*estimated* from *device* inputs):

- K0 inputs, R72-result §5 (ggml stock body on 4-7, the descriptive line that `memo-09-work/bonnie_gap.py`
  reproduces as 6.90 / 6.21 / 6.22 tok/s; K1 is slower than ggml on every subject and parked): body 120.6 / 140.6 /
  130.8 ms, head (Q6_K, 7,4-5) 9.04 / 4.40 / 13.88 ms, O = 8 ms.
- **lane = 1 ÷ `sum_vs_cpu_alone`** (x3 togsum). The equivalent g for the memo-09 / memo-10 models is
  g_eff = (`sum_vs_cpu_alone` − 1) × 26 ÷ 0.64.
- **dispatch(s) = n_join(s) × `x3 rt` `ovh_us` at 2816 rows**, with n_join = 2 joins per layer (memo-09's 64 at 32
  layers): 64 / 64 / 60. Descriptive beside it: one join per matrix, 7 × n_layer = 224 / 224 / 210 joins, and the
  `x3 layersum` per-subject overhead n_layer × (ms_sync7 − dev_ms).
- **P_G(s) = 950 ÷ (body(s) × lane + dispatch(s) + 8 + head(s))** tok/s printed: K0 with the GPU lane alone.
- **Centre stack (memo-10 §5.3):** width 2 with the trained head (Tom's measured top-1), GPU lane, C7 head × 0.5,
  C5 body × 0.93, O 5.2 ms, C 40, computed with `bonnie_tree.best(s, 2, C=40, g=g_eff, disp=dispatch(s), hs=0.5,
  c5=0.93, O=5.2)`. Its non-GPU levers are unbuilt; the stack is a price, not a plan.
- At the §4 centres (g_eff 8, 150 µs, 9.6 / 9.6 / 9.0 ms): P_G = **7.5 / 6.8 / 6.8** tok/s; centre stack
  **9.3 / 9.1 / 8.9** tok/s (*estimated*).

**The line at which the GPU lane makes the centre stack ≥ 12 on all three** (*estimated*, lead's arithmetic with
`bonnie_tree.py` as committed; DeepSeek binds):

| dispatch per token | g_eff needed (Yi / Llama-2 / DeepSeek) | `sum_vs_cpu_alone` needed (all three) |
|---|---|---|
| 0 ms | 23.1 / 25.9 / 29.9 GB/s | ≥ 1.74 |
| 3 ms | 25.9 / 28.5 / 32.9 GB/s | ≥ 1.81 |
| 9.6 ms (the centre, 64 × 150 µs) | 32.8 / 34.7 / 40.3 GB/s | ≥ 1.99 |

- The ceilings measured so far (*device*): the GPU reads 31.1 GB/s alone, so g cannot exceed about 31; the CPU + GPU
  read sum 46.5 GB/s is 1.79 × the CPU's 26.
- **So, written now: there is no line inside the measured ceilings at a realistic dispatch cost.** The centre stack
  reaches 12 on all three only if dispatch is ≤ 1.3 ms per token (≤ 20 µs per join at 64 joins) **and** the GPU
  Q4_0 kernel computes at ≥ 0.96 of the GPU's own read rate, **and** every unbuilt lever of the stack lands at its
  centre. With the GPU lane alone on K0, no g up to 31 reaches 12 on any subject (best: Yi 10.8 at g 31.6 and
  3 ms, *estimated*).
- Yi alone: the line is g_eff ≥ 25.9 at 3 ms (`sum_vs_cpu_alone` ≥ 1.64), inside the ceiling; at 9.6 ms it is 32.8,
  outside.
- The result states P_G(s) and the centre stack at the measured inputs, and the measured gap to 79.2 ms per subject.
  An open lane does not change the campaign verdict unless these prices move it; it narrows the gap.

**Residency (x3r): descriptive, with this reading.**
- *Held* means all three: the median CPU read over passes index 5 to the end ≥ 0.80 × (29.90 − 0.36 × the arm's
  `gpu_gbs`) (R72 block `a` read on 4-6, less the R72 contention per GPU GB/s); the median majflt per pass ≤ 50; and
  the `x3 res done` line.
- Footprint: 0.9 GB GPU + 2.6 GB anonymous = 3.5 GB, against R72's anonymous capacity: anon34 (3.4 GB) held, anon42
  (4.2 GB) not.
  - Held: a lane holding about 0.9 GB of GPU buffers beside a 2.6 GB CPU-side body is resident at 3.5 GB. That covers
    Yi's split body (resident estimate 3.36 GB). Llama-2 and DeepSeek (4.0 GB) are not covered by this arm; they still
    need a file-backed CPU share.
  - Not held: the GPU buffers press on the same memory as the anonymous body; a lane then needs a smaller GPU share
    or a file-backed CPU share on every subject, and Yi's 0.04 GB margin (R72) is gone.
  - Limits: the app's own memory is outside the arm; the arm runs 15 s, not a decode.

**Void (kit):**
- the APK hash differs (the owner checks the downloaded APK against `r34/SHA256SUMS`): the whole round is void;
- power saving reads on: every block run in that state is void;
- the charger reads plugged: that block is void;
- the process is killed during a block: that block only is void (in x3r also reported as "killed at pass n"; the
  earlier blocks are saved).

**Not voids (conditions).** Reported beside every number they touch: CPU clocks below max (`ramp_ok=0`,
`mhz_range`), GPU clock below `max_gpuclk`, thermal headroom, off-core or misplaced CPU threads (M / U / P / C
labels), gate timeouts, the deep-sleeping list label, and `x3 cond` (THP, devfreq). The rules use the cells as built;
a `ramp_ok=1` subset is given descriptively beside each rule input.

A kernel that fails to build is a result for rule 1, not a void. Misses are written as misses.

## 6. Claim rule

Not a goal claim. No model is loaded; no number from R74 is a phone token rate. P_G and the stack are *estimated*
from *device* inputs.

## 7. What a result means

- **Killed at rule 1 or 2:** the upstream Adreno kernel does not work on this phone's E031.38 compiler as built. With
  it goes the last measured route above the CPU read rate (R72 §5). The campaign verdict (12 tok/s out of reach on all
  three by priced levers) is restated with that fact.
- **Killed at rule 3 or 4:** the kernel works but the lane does not pay. Same consequence, with g and the sum in
  numbers.
- **Open:** Tom's host numerics check and a lane design (§5 rule 5) go on the host list. The prices above say how
  far the lane narrows the gaps; a device test of a lane needs its own pre-registration and the owner's permission.

## 8. The run, for the owner

1. After this file is pushed, download `r34/vollie-c3-probe-r34.apk`, `r34/BUILD-r34.txt` and `r34/SHA256SUMS` from
   `HoeioUser/vollie-kits`. Check that the APK's sha256 is
   `2b0a292978913c3d546d798aac895b414f1845b655133edc5d0eb2a5fcbe81de`.
2. Install it over kit r33. No uninstall is needed (34 > 33, same certificate).
3. Phone state:
   - charger out;
   - power saving off;
   - choose the deep-sleeping list session label in the app (on or off, as you prefer; it is recorded);
   - let the phone rest at least 30 min, screen off, not in use.
4. Research kit page: **Run c3 probe r34 (GPU)**. Do not use the phone until it finishes (about 2 to 3 min of blocks
   plus up to four cool-down gates of 3 min each; about 3 to 15 min in total).
5. **Save c3 probe to Downloads.** Send the JSON. It goes into `campaigns/03-dense15-5pct/reports/device/`.
6. If the app is killed (most likely in x3r, the last block), the earlier blocks are already saved. Send the file as
   it is.
