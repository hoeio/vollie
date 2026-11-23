# Pre-registration R75: x4 GPU join probe (launch, join and residency cost of a GPU lane; c3 probe kit r35)

This file is committed and pushed **before** the APK is uploaded, and it is not edited afterwards. Any change goes
into `R75-amendment-1.md`. The tag `prereg-R75` is the owner's to create. The commit hash and the push time show the
order.

- **Date / author:** 2026-11-23, clyde (lead).
- **Why this round exists:** device rounds are held until a real breakthrough (owner, 2026-11-22). The owner allowed
  the X4 join probe as a second exception (owner, 2026-11-23: "측정 시도 허락합니다", in answer to the lead's report of
  memo-12; CAMPAIGN.md, end, commit `2517042`). R73 stays reserved for the held first real decode, so this round is
  **R75**. Scope: the GPU lane's launch and join cost and its residency at a larger GPU share. No model is loaded.
- **x4m is in scope (owner, 2026-11-23, in chat: "x4m도 넣어도 됩니다").** The x4m block (upstream's two- to
  four-column gemv `_mc3`, the CPU + GPU together cell at one and two columns, and the mixed arm) was added after the
  first permission, as part of the GPU-lane measurement; the owner then confirmed it explicitly.
- **Agent models** (full IDs):
  - lead (clyde): `claude-opus-5-5`, this file;
  - bonnie: `claude-fable-5-2`, two instances this round (context filled, then a usage-limit stop): join pricing and
    bands, `bonnie-joins.md`, `-r2.md` to `-r5.md`, `bonnie_joins*.py`, `bonnie_tree_r2.py`; the merged table of
    §4 is her r5 §1;
  - ebony: `claude-fable-5-2`, two instances this round (same reason): review and bands, `ebony-joins.md`, `-r2.md`
    to `-r5.md`, `ebony_joins.py`, `ebony_pair.py`; her r5 §5 (commit `83b90b2`) reviews Bonnie's r5 and agrees
    every merged row, and her r5 §4 re-keys her rows to the r35 fields;
  - tom: `claude-opus-5-5`, two instances: kit r35 and its final rebuild (patch 0007, app 0005,
    `tools/c3_x4child.c`, parser `x4report`, `tom-r35-build.md`, kit commit `b4b9168`) and, separately, the
    numerics emulation (0006, not in this kit);
  - annie: `claude-sonnet-5-5`, books.
- **Subjects** (memo-06 §6): Yi-1.5-6B-Chat, Llama-2-7b-chat, DeepSeek-LLM-7B-chat. They enter only as matrix shapes
  and layer counts (32 / 32 / 30). Triples below are always Yi / Llama-2 / DeepSeek.
- **Goal context:** 12 tok/s printed on each subject = 79.2 ms per token (CAMPAIGN.md Amendment 1). R75 is not a
  claim round.
- **Labels:** every number in §4 and §5 is *estimated* until the result marks it *device*. Prices in §5.3 are
  *estimated* from *device* inputs.
- **Starts from:** kit r34 (R74: x3 blocks, `run_arm`, the r16 cool-down gate before each block, heat, clock, power
  and charger records), and through it campaign 2's kit protocol and campaign 1's `libvollie_clc.so`. Device anchors
  from R74 (*device*): `x3 rt` `ovh_us` 452 at 2,816 rows (launcher 0-3); `x3 togsum` CPU alone 25.64, both CPU 19.77,
  both GPU 21.53 GB/s, `sum_vs_cpu_alone` 1.611; x3r held 0.85 GB GPU + 2.6 GB anonymous for 15 s with MemAvailable
  down to 0.59 GB.

## 0. Build under test

The final r35 build (kit commit `b4b9168`: mixed arm in x4m, `gpu_max_mhz` in the launcher tails and on the x4r
pass lines, `x4 therm` before and after every block, the `x4 kai` line). Hashes re-computed by the lead on
2026-11-23 with `shasum -a 256` from `lab/dist-c3/r35/` and `patches/`; all match `tom-r35-build.md`. The first r35
build (kit commit `e961ecf`; APK `089ab530…4c62`) is superseded and is not the build under test.

| part | file | sha256 |
|---|---|---|
| APK | `vollie-c3-probe-r35.apk`, 16,621,998 B, package `com.insaneutils.vollie.c2`, versionCode 35, versionName 1.35.0, same debug certificate as r34 (`c49dde85…1716`) | `465d97bb7932fc4c9b7c675e1b3e2b78536466f9dec3442d4b8c59405f558af5` |
| BUILD file | `BUILD-r35.txt` | `5211ba23acf957ee42d288e466a98ebd35583c2c9b7d7a1455704dc56b29e91f` |
| sums | `SHA256SUMS` (lists the APK and BUILD file above) | `1eb6cdffa9e1ce5a84f703b740f7709dca14cc726f8ef8d6a2c4d52f3a01d321` |
| runtime probe | `patches/llama.cpp-c3/0007-x4-join-probe.patch` (only `src/llama-vollie-probe.cpp`, on 0005) | `fcd6c1e8c7a71f57224239e248c380ea189e4ca346412a17c20d44f4bcc60a8a` |
| app | `patches/app-c3/0005-kit-r35-probe.patch` (on 0004; unchanged by the rebuild) | `14c96c38faad7e01959beb793b6087d64b8580920ae157e975943b17c654fad0` |
| child process source | `tools/c3_x4child.c` | `d75cf8ca58252f85fdc55667395e297fd738d8fbb4f92656d339eb2edae38469` |
| libllama.so | in the APK (c3 0001, 0002, 0003, 0005, 0007) | `eb67f4bc0d5a1af77070d929f6562bfb602bf59ac264ba8f47a06882e87a2e96` |
| libvollie_x4c.so | in the APK (child loader, 5,184 B, after the gradle strip; unchanged by the rebuild) | `fb61545acea9990c3bff418682f46970eba24fd92b56af2b256368302779315f` |
| libggml-cpu-android_armv8.2_2.so | in the APK, byte-identical to r33 and r34 | `46e0735d2824ba6f3a156e219e058a701f6024fdbecbb2ef937b3127115a9694` |
| kit commit | the commit that adds the rebuilt r35 sources | `b4b9168` |

- **Identity of the CPU module** (*host*): it ships byte-identical from r33, so 0003's knob-unset per-chunk PPL
  identity at `-b 512`, `-ub 32` and `-ub 1` carries over. (A fresh rebuild of it from the same sources is not
  byte-reproducible, 29,096 bytes differ, cause not traced; it does not touch R75 because the shipped bytes are
  r33's.)
- **Numerics:** none changed in R75. The probe runs only with `VOLLIE_PROBE` set by the app's probe button; unset,
  the entry returns -1. A GPU lane built later is a declared numerics change (R74 §0).
- **Parser:** `tools/c3_probe_parse.py` (`x4report`; `--selftest` prints `selftest x4 ok` and `selftest ok`, *host*;
  its 61 synthetic x4 lines include `x4 therm`, `x4 kai`, `x4 togmix` and `x4 togmixsum`).

## 1. Device state

- The owner's Galaxy Z Flip 3, power saving **off**, charger **out**; both are recorded.
- The deep-sleeping list label is chosen in the app and recorded.
- **True cool start** (R74's was not met): the phone rests at least 30 min, screen off and not in use, and **no other
  probe runs before this one** in that rest or after it, including campaign 2's cap probe. The kit cannot check it;
  the first block's `x3 cond`, its clocks and the gate series show the state, and the result reports them.
- The r16 gate (cap 180 s) runs before each of the ten blocks.
- No model is loaded.
- Kit r35 is installed over kit r34 (versionCode 35 > 34, same certificate; no uninstall).

## 2. What is measured (blocks as built; run order x4i, x4a, x4b, x4f, x4c, x4d, x4e, x4m, x4s, x4r)

Field names are those of `tom-r35-build.md` (final build); the rules in §5 cite them. The block order is
unchanged by the rebuild.

| block | content (lines) |
|---|---|
| `x4i` | `x4 ext`, `x4 dev`, `x4 recq`, `x4 entry`; fresh compile `x3 src`, `x3 build`, `x3 buildlog`, `x3 load`, `x3 loadlog`; `x3 identity` (general, 2816 × 4096); child `x4k_c` before the GPU context opens: `x4 kai kernel=<F\|unseen> features= q4_buft= backend= cpu_ok=`, `x4 kai log text=`, `x4 child blk=x4k_c` |
| `x4a` | launch timeline on the profiling queue: `x4 tl setup`, `x4 tl` per event (`queued_us`, `submit_us`, `start_us`, `end_us`, `gap_us`), `x4 tlsum` for cases `batch8`, `layer7`, `batch8flush`, launchers 0-3 then cpu 4 |
| `x4b` | `x4 rt` for launcher 0-3, 4, 7 × wait `finish`, `spin`, `wfe`, plus 0-3 `finish` `rep=1` (drift anchor): 10 cells, 2816 × 4096 general |
| `x4f` | `x4 la mode=gate` and `mode=plain` (launch-ahead behind a user event), `x4 rtargs` ABBA 15 / 2 / 2 / 15 args, launcher cpu 4 |
| `x4c` | `x3 identity` m 22016 (gate‖up) and m 12288 (q‖k‖v); `x3 alone` cold at 11008, 22016, 4096, 12288 rows (ABBA); `x4 stacksum` |
| `x4d` | child `x4d_c`: `x4d skip` or `x4 rec setup`, `x4 rec` (ABBA sync1 / replay), `x4 recsum`; `x4 child`, `x4 childlog` |
| `x4e` | children `x4e1_c` (flag), `x4e2_c` (counter), `x4e3_c` (gemv): `x4 pk c11`, `x4 pk test=flag`, `x4 pk test=counter` in modes `hostptr`, `svm`, `svm_c11`; `x4 pk test=gemv`, `x4 pksum`; `x4 child`, `x4 childlog` |
| `x4m` | cg cell: `x4 mc setup`, `x4 mcid` (n 2, 3, 4), `x4 mc` (ABBA n 1, 2, 3, 4, 4, 3, 2, 1), `x4 mcsum`; together cell: `x4 tog setup`, `x4 tog` (ABBA cpu, gpu, both, both, gpu, cpu) at two columns then one, `x4 tog2sum`, `x4 tog1sum` (or `x4 tog2 ok=0 ... then=n_cols_1_only`); mixed arm: `x4 togmix choice gpu_n_cols=<4\|3\|0> n4_ok= n3_ok=` (`ok=0 reason=` when skipped), `x4 togmix setup`, `x4 togmix pos= cpu_n_cols=1 gpu_n_cols= arm=`, `x4 togmixsum cpu_n_cols=1 gpu_n_cols= cpu_alone_gbs= gpu_alone_gbs= both_cpu_gbs= both_gpu_gbs= both_cpu_vs_tog1_both_cpu= sum_gbs= sum_vs_cpu_alone= cpu_alone_ms= both_cpu_ms= gpu_alone_dev_us= both_gpu_dev_us= err=` (CPU at ne1 = 1 beside the GPU's `_mc3` at 4 columns, 3 if the n = 4 identity failed; the ratio divides by the same block's `x4 tog1sum both_cpu_gbs`) |
| `x4s` | `x4 split setup`, 22 `x4 split` arms (cpu, lead 0.20 to 0.60 step 0.05, feeder 0.25 twice, lead 0.60 to 0.20, cpu), `x4 splitsum`; hand-off cell: `x4 handoff setup`, 10 `x4 handoffarm`, `x4 handoff` per mode (`none`, `write`, `map`, `ahp_map`, `ahp_persist`) |
| `x4r` | 59 copies of 11008 × 4096 (1.5 GB) of GPU buffers read by the GPU while 3 A78 (4-6) read a 2.6 GB anonymous arena: `x4 res stage=`, `x4 res pass=`, `x4 res sum`, `x4 res done err=` |

Recorded with every timed cell: the launcher `<tail>` (`lcpu_us`, `lcpu_over_wall`, `lplace`, `thr`, `mhz_range`,
`max_mhz`, `gpu_mhz`, `gpu_max_mhz`, `hr`, `ramp_s`, `ramp_ok`) or x3's `astr` tail (`thr`,
`feeder`, `mhz`, `mhz_range`, `max_mhz`, `gpu_mhz`, `gpu_max_mhz`, `gpu_busy`, `hr`, `ramp_s`, `ramp_ok`); `x3 cond`
before each parent block; `x4 therm blk= when=<before|after> status= thermal_pwrlevel= gpu_max_mhz= gpu_mhz=` at the
start and end of every parent block; `gpu_max_mhz=` at the end of every `x4 res pass` line; `x4<blk> end s=`.

**Phone time** (*estimated*, `tom-r35-build.md`, final build): about 3.6 to 7 min of blocks plus ten gates of up to
180 s each, so about 4 to 37 min.

**Memory peak** (*estimated*, `tom-r35-build.md`): x4m about 1.2 GB, x4s about 0.6 GB, **x4r about 4.1 GB**. Tom's
note: at 4.1 GB a low-memory kill or swap is likely. x4r is last and the report is saved before it, so a kill loses
only x4r.

## 3. Budget accounting

No numerics change and no model run. Nothing is spent.

## 4. Predictions (*estimated*; centre with 80 % band)

**Source.** The merged table of `reports/memo-12-work/bonnie-joins-r5.md` §1 (Bonnie r5, which folds in Ebony's
bands from `ebony-joins-r2.md` B.3, `ebony-joins-r3.md` §1.2 and `ebony-joins-r4.md` §1.3 / §1.4), with Ebony's
r5 changes folded in (`ebony-joins-r5.md` §4, §5: her rows re-keyed to r35 fields, her centre for the CPU alone at
two columns moved, the hand-off map row split, a band for the mixed arm). B = Bonnie, E = Ebony. Where two centres are printed, both are reported against the result and the row is
scored on the **union band**. "=" means the two agree. The together-cell reading line of §5.2 is **joint**: Ebony's
base with Bonnie's two changes, accepted by Ebony in her r5 §5.

How each row is scored:
- A value inside its band is "inside"; outside is a **miss**, high or low.
- A row whose cell did not run (no kernel, alloc error, skip line, killed block, child killed) is
  "not measured", not a miss.
- Event probabilities are scored by the event's outcome and reported (a Brier score over §4.1 is given); they are
  not bands.

### 4.1 Events

| # | event | B | E | merged | field |
|---|---|---|---|---|---|
| V1 | recordable queues in the extension string | 0.99 | 0.99 | 0.99 | `x4 ext text=` contains `cl_qcom_recordable_queues` |
| V2 | recordable queue present, `max_size` > 0 | 0.80 | 0.80 | 0.80 | `x4 recq present=1 max_size=` |
| V3 | `svm_coarse` / `svm_fine_buffer` / `svm_fine_system` = 1 | 0.85 / 0.30 / 0.05 | 0.90 / 0.70 / 0.05 | 0.90 / 0.60 to 0.70 / 0.05 | `x4 dev` |
| V4 | `svm_atomics` = 1 | – | 0.65 | 0.65 (E) | `x4 dev svm_atomics=` |
| V5 | x4d runs / passes (replay overhead ≤ 0.5 × sync1's) | 0.55 / 0.25 | 0.55 / 0.20 | 0.55 / 0.20 to 0.25 | `x4 rec setup ok=1`; `x4 recsum` `ovh_us_per_launch_replay` ≤ 0.5 × `ovh_us_per_launch_sync1` |
| V6 | identity ok on both stacked shapes | 0.90 | 0.90 | 0.90 | `x3 identity` m 22016 and m 12288 `ok=1` |
| V7 | flag seen within 1 ms, 1.x modes (`hostptr` or `svm`) | 0.35 | 0.40 | 0.35 to 0.40 | `x4 pk test=flag` `ok=1` and `rtt_p50_us` ≤ 1000 |
| V8 | flag seen within 1 ms, any mode with `svm_c11` | – | 0.55 | 0.55 (E) | same, any of the three modes |
| V9 | C11 builds and runs / flag seen given it runs | – | 0.55 / 0.75 | E | `x4 pk c11 err=0` and a `mode=svm_c11` flag line without `ok=0`; then V7's test on it |
| V10 | all three F lines pass in some mode (§5.1 rule F "worth") | 0.10 | 0.12 to 0.15 | 0.10 to 0.15 | `x4 pk` flag, counter, `x4 pksum` |
| V11 | a child hangs or times out | 0.35 | 0.35 | 0.35 | any `x4 child rc=-1001 timeout=1` |
| V12 | the driver serialises (most i ≥ 1 of `batch8` have `submit_us(i)` ≥ `end_us(i−1)`) | 0.60 | 0.50 | 0.50 to 0.60 | `x4 tl` |
| V13 | best `ovh_us` ≤ 250 / ≤ 150 / ≤ 100 | 0.80 / 0.12 / 0.02 | 0.75 / 0.20 / 0.03 | 0.75 to 0.80 / 0.12 to 0.20 / 0.02 to 0.03 | min `x4 rt ovh_us=` over the nine `rep=0` cells |
| V14 | 0-3 `spin` `submit_to_start_us` ≥ 30 below cpu 4 `spin` | 0.50 | 0.45 | 0.45 to 0.50 | `x4 rt submit_to_start_us=` |
| V15 | the driver prepares at enqueue | 0.40 | 0.30 | 0.30 to 0.40 | `x4 la mode=gate set_to_start_p50_us` ≤ `x4 la mode=plain submit_to_start_p50_us` + 30 |
| V16 | `_mc3` created / runs / identity given runs / all | 0.92 / 0.85 / 0.88 / 0.75 | 0.92 / 0.85 / 0.90 / 0.76 | 0.92 / 0.85 / 0.88 to 0.90 / 0.75 | no `x4 mc ok=0 reason=`; `x4 mc err=0`; `x4 mcid ok=1` every column |
| V17 | x4s optimum is interior | 0.90 | 0.85 | 0.85 to 0.90 | `x4 splitsum mode=lead ratio=` rises on both sides of the best share |
| V18 | x4r held (§5.4) / killed | 0.35 / 0.10 | 0.25 / 0.15 | 0.25 to 0.35 / 0.10 to 0.15 | `x4 res sum`, `x4 res done`; killed = no `x4r end` |
| V19 | hand-off: no-call variant correct / best ≤ 20 µs and correct | – | 0.60 / 0.50 | 0.55 to 0.60 / 0.50 | `x4 handoff mode=ahp_persist match=1`; best `extra_us` ≤ 20 with `match=1` |
| V20 | together: CPU alone at two columns under 19 GB/s | 0.90 | 0.85 | 0.85 to 0.90 | `x4 tog2sum cpu_alone_gbs=` |
| V21 | together line (§5.2) reads `pair` / undecided / carried-over | 0.50 / 0.30 / 0.20 | 0.50 / 0.25 / 0.25 | 0.50 / 0.25 to 0.30 / 0.20 to 0.25 | derived from `x4 tog2sum` |
| V22 | `gpu_max_mhz` < 840 on any R75 line / `thermal_pwrlevel` > 0 on any `x3 cond` | 0.15 / 0.10 | – | B | `astr` and launcher tails; `x3 cond` |
| V23 | the two-column together part is skipped | – | 0.25 | 0.25 (E) | `x4 tog2 ok=0 reason= then=n_cols_1_only` |
| V24 | mixed arm `both_cpu_vs_tog1_both_cpu` ≥ 1.15 / ≤ 1.05 (scored only at `gpu_n_cols=4`) | – | 0.50 / 0.25 | E | `x4 togmixsum` |

Note on V18: the researchers' pre-data band is P(held) 0.25 to 0.35 and P(killed) 0.10 to 0.15 (kept as ranges;
Ebony r5 §5), scored as written. Beside it, not scored: Tom's build note expects a low-memory kill or swap in x4r to
be likely at 4.1 GB.

### 4.2 Cells

| # | cell, field | B (centre; band) | E | merged (scored on) | field |
|---|---|---|---|---|---|
| A1 | `gap_p50_us`, `batch8`, launcher 0-3 | 200; 145–270 | 200; 110–270 | 200; 110–270 | `x4 tlsum case=batch8 lcpu=0-3` |
| A2 | `gap_p50_us`, `batch8`, launcher cpu 4 | 210; 140–285 | 185; 100–285 | 185 / 210; 100–285 | `x4 tlsum case=batch8 lcpu=4` |
| A3 | `batch8flush` − `batch8` `gap_p50_us` (same launcher) | −10; −60 to +30 | = | = | `x4 tlsum` |
| A4 | `s2b_first_us`, 0-3 | 190; 130–220 | = | = | `x4 tlsum` |
| A5 | `enq_us` per launch, 0-3 / cpu 4 | 125 / 62; 85–185 / 35–115 | = | = | `x4 tlsum enq_us=` |
| A6 | `layer7` (`wall_us` − `dev_sum_us`) ÷ 7, 0-3 / cpu 4 | 235 / 215; 150–300 / 130–290 | 235 / 200 | 235 / 200 to 215; 150–300 / 130–290 | `x4 tlsum case=layer7` |
| A7 | `gap_p50_us` `layer7` − `batch8` | – | +40; 0 to +120 | E | `x4 tlsum` |
| B1 | `ovh_us` 0-3: finish / spin / wfe | 400 / 265 / 395; 320–500 / 190–360 / 310–500 | = | = | `x4 rt lcpu=0-3 rep=0` |
| B2 | `ovh_us` cpu 4: finish / spin / wfe | 310 / 255 / 305; 225–410 / 150–345 / 220–410 | spin 235; 130–345 | spin 235 / 255; 130–345; others = | `x4 rt lcpu=4` |
| B3 | `ovh_us` cpu 7: finish / spin / wfe | 295 / 250 / 290; 215–390 / 145–335 / 210–390 | spin 235; 130–345 | spin 235 / 250; 130–345; others = | `x4 rt lcpu=7` |
| B4 | **best `ovh_us` of nine** | **215; 140–280** | **205; 120–290** | **205 / 215; 120–290** | min `x4 rt ovh_us=`, `rep=0` |
| B5 | `queued_to_start_us` 0-3 finish / 0-3 spin / cpu 4 spin | 340 / 240 / 240; 275–425 / 175–330 / 140–320 | = | = | `x4 rt` |
| B6 | `submit_to_start_us`, same three | 162 / 120 / 177; 125–200 / 75–195 / 75–220 | = | = | `x4 rt` |
| B7 | `p90_us` − `p50_us`, finish / spin cells (median over launchers) | 150 / 90; 60–300 / 30–250 | = | = | `x4 rt` |
| B8 | `lcpu_us`, finish on 0-3; `dev_us` | 300; 200–450. 240; 225–265 | = | = | tail `lcpu_us=`; `x4 rt dev_us=` |
| F1 | `set_to_start_p50_us`, gate, cpu 4 | 210; 110–280 | 215 | 210 / 215; 110–280 | `x4 la mode=gate` |
| F2 | `rtargs`: `enq_us` at `args=2` − at `args=15` (mean of the ABBA pairs) | −10; −22 to −3 | – | B (r5) | `x4 rtargs enq_us=` |
| C1 | gate‖up cold GB/s; `gate_up_vs_11008` | 24.3; 1.14 (21.5–27.0; 1.04–1.20) | 23.5; 1.10 (1.00–1.20) | 23.5 / 24.3; 1.10 / 1.14; 21.5–27.0; 1.00–1.20 | `x4 stacksum` |
| C2 | q‖k‖v GB/s; `qkv_vs_4096`; `qkv_gbs` ÷ `m11008_gbs` | 21.6; 1.42; 1.02 (19.0–25.5; 1.25–1.60; 0.93–1.10) | = | = | `x4 stacksum` |
| D1 | `ovh_us_per_launch_replay`; `replay_vs_sync1` | 138; 0.87 | 170; 0.80 | 138 / 170; 0.80 / 0.87; 26–250; 0.55–0.97 | `x4 recsum` |
| E1 | flag `rtt_p50_us`; counter `iv_p50_us`, 1.x modes, given seen | 30; 5–300. 10; 1–100 | = | = | `x4 pk mode=hostptr`/`svm` |
| E2 | the same, `svm_c11`, given seen | – | 20; 3–200. 5; 1–60 | E | `x4 pk mode=svm_c11` |
| E3 | `q4p48_vs_q4n` | 0.85; 0.50–1.02 | = | = | `x4 pksum` |
| M1 | **`dev_ratio` 2 / 3 / 4 columns** | 1.2 / 1.5 / 1.9 | 1.3 / 1.65 / 2.0 | **2: 1.2 / 1.3; 1.05–1.80**. 3: 1.5 / 1.65; 1.15–2.6. 4: 1.9 / 2.0; 1.35–3.4 | `x4 mcsum n_cols= dev_ratio=` |
| M2 | `wall_ratio` 2 columns; `max_rel_err` per column | 1.15; 6e-4 to 8e-4 | 1.22; = | 1.15 / 1.22; 1.04–1.65. 6e-4 to 8e-4 (scored as a band) | `x4 mcsum wall_ratio=`; `x4 mcid max_rel_err=` |
| T1 | one column: `cpu_alone_gbs`; `both_cpu_gbs` / `both_gpu_gbs`; `sum_vs_cpu_alone` | 25.6 (24.5–27.0); 19.8 / 21.5 (18.5–21.0 / 19.5–24.5); 1.61 (1.50–1.72) | = | = | `x4 tog1sum` |
| T2 | two columns: `cpu_alone_gbs` | 14.1; 12.2–16.5 | 14.5; 12.2–18.0 (r5; was 15.5) | 14.1 / 14.5; 12.2–18.0 | `x4 tog2sum` |
| T3 | `cpu_alone_gbs` two ÷ one column | 0.55; 0.47–0.65 | 0.57; 0.47–0.72 (r5) | 0.55 / 0.57; 0.47–0.72 | tog2sum ÷ tog1sum |
| T4 | two columns: `gpu_alone_gbs` | 20.4; 15.0–23.5 | 19.8; 14.5–23.5 | 20; 14.5–23.5 | `x4 tog2sum` |
| T5 | two columns: `both_cpu_gbs` | 13.7; 11.7–16.0 | 14.0; 11.7–17.5 (r5) | 13.7 / 14.0; 11.7–17.5 | `x4 tog2sum` |
| T6 | two columns: `both_gpu_gbs` | 18.5; 13.5–21.5 | 18.6; 13.5–22.0 | 18.6; 13.5–22.0 | `x4 tog2sum` |
| T7 | two columns: `sum_gbs` | 32.5; 27.5–36.5 | 32.6; 27.5–38.5 (r5) | 32.5 / 32.6; 27.5–38.5 | `x4 tog2sum` |
| T8 | **GPU ratio** = `both_gpu_gbs` ÷ `gpu_alone_gbs`, two columns | 0.92; 0.85–0.96 | 0.94; 0.85–1.00 | **0.92 / 0.94; 0.85–1.00** | `x4 tog2sum`; cross-check `gpu_alone_dev_us` ÷ `both_gpu_dev_us` |
| T9 | **sum ratio** = `sum_gbs` ÷ (`cpu_alone_gbs` + `gpu_alone_gbs`), two columns | 0.95; 0.88–1.00 | 0.95; 0.85–1.00 | 0.95; 0.85–1.00 | `x4 tog2sum` |
| T10 | two columns: `sum_vs_cpu_alone` as printed (÷ the two-column CPU) | 2.30; 1.9–2.8 | 2.25; 1.9–2.8 (r5) | 2.25 / 2.30; 1.9–2.8 | `x4 tog2sum` |
| T11 | **mixed arm** `both_cpu_vs_tog1_both_cpu` (CPU 1 column beside GPU 4 columns, ÷ the one-column cell; scored only at `gpu_n_cols=4`) | 1.15; 1.02–1.30 | 1.20; 1.00–1.30 (r5 §5) | 1.15 / 1.20; 1.00–1.30 | `x4 togmixsum both_cpu_vs_tog1_both_cpu=` |
| S1 | `t_cpu_alone_ms` | 1.90; 1.80–2.05 | = | = | `x4 splitsum` |
| S2 | feeder share 0.25: `ratio`; `gpu_ms`; `wait_ms` | 0.86; 1.00; 0.02 (0.83–0.90; 0.85–1.15; 0–0.06) | = | = | `x4 splitsum mode=feeder` |
| S3 | lead: best `ratio` over 0.20 to 0.60; best share | 0.73; 0.70–0.79. 0.50 | 0.72; 0.67–0.79. 0.50 | 0.72 / 0.73; 0.67–0.79. 0.50 (P 0.60 to 0.78) | `x4 splitsum mode=lead` |
| S4 | lead share 0.20 / 0.35 `ratio` | 0.92 / 0.82; 0.88–0.96 / 0.79–0.86 | = | = | `x4 splitsum` |
| S5 | lead 0.50: `gpu_ms`; `gpu_dev_ms`; `wait_ms`; `enq_us` | 1.29; 0.985; 0.04; 65 | =; 1.03; 0.08; = | 1.29 (1.15–1.42); 0.985 / 1.03 (0.90–1.10); 0.04 / 0.08 (0–0.25); 65 (35–125) | `x4 split share=0.50 mode=lead` (mean of 2 arms) |
| S6 | lead 0.55: `ratio`; `gpu_ms`; `wait_ms`; `gpu_dev_ms` | 0.78; 1.38; 0.21; 1.08 | =; =; =; 1.13 | 0.78 (0.72–0.84); 1.38 (1.28–1.46); 0.21 (0.11–0.29); 1.08 / 1.13 (0.98–1.22) | `x4 split share=0.55` |
| S7 | lead 0.60: the same four | 0.83; 1.48; 0.39; 1.18 | =; =; =; 1.23 | 0.83 (0.77–0.90); 1.48 (1.38–1.56); 0.39 (0.29–0.47); 1.18 / 1.23 (1.07–1.33) | `x4 split share=0.60` |
| S8 | **in-situ K_meas** (§5.3) | 300; 200–380 | 290; 170–400 | **290 / 300; 170–400 µs** | derived from `x4 split` |
| S9 | GPU device-time rate beside the CPU, best share | 25.7 | 24.6; 22.5–27.5 | 24.6 / 25.7; 22.5–27.5 GB/s | `gpu_rows` × 4096 × 0.5625 B ÷ `gpu_dev_ms` |
| S10 | `cpu_ms` ÷ ((1 − share) × `t_cpu_alone_ms`), share 0.25 / 0.50 | 1.14 / 1.32 | 1.14 / 1.30; 1.08–1.22 / 1.20–1.40 | E (agreed) | `x4 split` |
| H1 | hand-off `extra_us`: `write` / `map` / `ahp_map` / `ahp_persist` | – | 90; 40–200 / 70; 30–170 / 50; 20–130 / 5; 0–30 (r5 split her r3 map row 60; 25–150) | E | `x4 handoff mode= extra_us=` (`write_us` + `read_us` beside it) |
| R1 | x4r, if held: `cpu_gbs_median`; majflt median | 18.5; 0 (17.5–19.5; 0–5) | = | = | `x4 res sum` |

### 4.3 Bands with no printed field (not scored)

- Bonnie r2's x4f `set_to_start_p50_us` on launcher 0-3 (190): x4f runs on cpu 4 only. Not scored.
- Bonnie r2's x4f "`ovh_us` at 3 args − at 15 args" (−8): the kit cuts to 2 args and is read on `enq_us`; replaced
  by F2. Not scored.
- Bonnie r4's sum line (sum ≥ 38.5 / ≤ 35.5), "sum ÷ one-column `cpu_alone_gbs`" (1.44) and "two-column sum ÷
  one-column sum" (0.90): superseded by r5 §2 (the KleidiAI four-row tile at ne1 = 2); no single field. Not scored.
- Bonnie r5 §5 per-minute GPU throttle reading (P 0.6 within 2 min of sustained decode): no cell runs that long.
  Not scored.
- Ebony r3 §2.5 "thermal state after each block": now printed (`x4 therm when=after`), but no band was written for
  it. Descriptive only.
- T11 and V24 at `gpu_n_cols=3`: Bonnie's and Ebony's bands are for four GPU columns. At three the mixed arm is
  read by §5.2 item 2, but T11 and V24 are not scored. With `x4 togmix choice gpu_n_cols=0` (skipped) they are not
  measured.

## 5. Decision rules (scored as written)

**Definitions** (from the phone's own lines):
- *Built kernel*: a kernel with an `x3 load ... ok=1` line in the block that uses it. A kernel whose `x3 identity`
  (or `x4 mcid`) reads `ok=0` is closed and its cells are not used (R74 rule 2).
- *Mean of two arms*: every ABBA quantity is the mean of its two arms, as the summary lines print it.
- *Best mode* (x4e): among `hostptr`, `svm`, `svm_c11`, the mode with flag `ok=1` and counter
  `visible_during_run=1` that has the least `rtt_p50_us` + `iv_p50_us`.
- *C11 ran*: `x4 pk c11 test=flag err=0` and a `x4 pk test=flag mode=svm_c11` line without `ok=0`.
- *Hand-off H*: the least `x4 handoff extra_us` among modes `write`, `map`, `ahp_map`, `ahp_persist` with
  `match=1`, floored at 0. A mode with `match=0` or `-1` is not a variant. If none has `match=1`, H = the `write`
  mode's `write_us` + `read_us` and the result says so.

### 5.1 Per design (Bonnie r2 §2.3 with Ebony's r2 B.4 and r3 §1.2 changes)

"Worth a build" means: the design's implied K (§5.3) gives at least +0.5 tok/s over the measured in-situ K on all
three subjects; it goes on the host list for a lane design. It is not a device go. "Kill" closes the design for this
lane. Between the two: "open, not worth a build now", reported with its number.

| design | cells and fields | kill | worth a build |
|---|---|---|---|
| **E / Y12** level stacking | x4c `x3 identity` m 22016, m 12288; `x4 stacksum gate_up_vs_11008`; x4s best lead `ratio` | identity `ok=0` on m 22016, an alloc error, `gate_up_vs_11008` < 0.90, or the best lead `ratio` over 0.20 to 0.60 ≥ 0.95 | identity ok, `gate_up_vs_11008` ≥ 1.00, and best lead `ratio` ≤ 0.85. q‖k‖v is judged apart: identity `ok=0` on m 12288 or `qkv_gbs` ÷ `m11008_gbs` < 0.90 closes q‖k‖v stacking only |
| **B** event spin | x4b, same launcher: `ovh_us` `wait=finish` − `wait=spin` | the saving is < 30 µs on every launcher (0-3, 4, 7) | the saving is ≥ 80 µs on any launcher |
| **B+ / Y13** fast or hot launcher | x4b best `ovh_us` of the nine `rep=0` cells | > 350 µs | ≤ 250 µs (Ebony: ≤ 150 opens her lower rows; > 150 means λ stays in 250 to 450, reported) |
| **A** launch-ahead | x4f `x4 la mode=gate set_to_start_p50_us`; x4b `x4 rt lcpu=4 wait=spin queued_to_start_us`; `x4 la mode=plain submit_to_start_p50_us` | gate `set_to_start_p50_us` ≥ 300, or it is less than 60 µs below the cpu 4 `spin` `queued_to_start_us` | gate `set_to_start_p50_us` ≤ plain `submit_to_start_p50_us` + 30 |
| **C** submit→start removable | x4a `x4 tl` per event, i ≥ 2, every case and launcher | median `gap_us` over i ≥ 2 ≥ 150 in every case **and** `submit_us(i)` < `end_us(i−1)` for most i ≥ 2 (submitted early, still late: a per-command device cost) | median `gap_us` over i ≥ 2 < 50 in any case. If `submit_us(i)` ≥ `end_us(i−1)` for most i (the driver serialises): **undecided, Y14 decides** |
| **Y14** recordable replay | x4i `x4 recq`; x4d `x4 rec setup`, `x4 recsum` | `x4 recq present=0`, `x4d skip reason=no_extension` or `queue err=`, or `ovh_us_per_launch_replay` > 0.5 × `ovh_us_per_launch_sync1` | `ovh_us_per_launch_replay` ≤ 100 |
| **F / Y15** persistent kernel | x4e `x4 pk`, `x4 pksum`, `x4 child` | any of: (i) no mode has flag `ok=1` with `rtt_p50_us` ≤ 1000, **counted only if C11 ran**; (ii) no mode has `visible_during_run=1`, counted only if C11 ran; (iii) the best mode's `rtt_p50_us` + `iv_p50_us` > 100 µs; (iv) max(`q4p48_vs_q4n`, `q4p192_vs_q4n`) < 0.80 or `same_rows=0` | a best mode exists, its `rtt_p50_us` + `iv_p50_us` ≤ 60 µs, and (iv) passes |
| **Y16** FFN column partition | x4a `x4 tlsum case=batch8 lcpu=4 gap_p50_us`; x4b `x4 rt lcpu=4 wait=finish ovh_us` | gap ≥ 0.5 × that `ovh_us` | gap ≤ 0.25 × that `ovh_us` |
| **D** whole matrices | x4s `x4 splitsum mode=lead` | `ratio` at share 0.50 is more than 0.03 above the best lead `ratio` | share 0.50 has the best lead `ratio`: no stacked buffer is needed for the FFN |
| **cg** (width 2 on the lane) | x4m `x4 mcsum n_cols=2 dev_ratio=` | **> 1.70: the width-2 route to 12 is closed** (no line with slack in any variant); the lane stays a 10 to 11 tok/s lever. ≥ 2.0: `_mc3` is no better than two launches; drop it | **≤ 1.30: worth a build** (every variant has a line for all three, §5.3); the lane's GPU side moves to `_mc3` for the width-2 pass. ≤ 1.10 is reported as the strong case. 1.30 to 1.70: a line exists only in `pair`; priced only if §5.2 reads `pair` |
| **Hand-off** | x4s `x4 handoff` | – (a cost, not a design) | – . H is added to every in-situ K before any price (§5.3) |
| **any design at Llama-2 or DeepSeek size** | x4r (§5.4) | not held | held |

Further rules:
- **Not measured, not closed.** Y14 with `x4 child blk=x4d_c rc` ≠ 0 before `x4 rec setup` printed (a crash from
  the unverified qcom signatures, `tom-r35-build.md`) is "not measured": the call path failed, the mechanism is not
  tested. F with C11 not run (build failed, no fine-grain SVM, no SVM atomics) and no 1.x mode passing is "not
  closed, not measured on C11", reported with `x4 pk c11` and the build log. A child timeout (`rc=-1001`) makes the
  modes it did not print "not measured"; a mode that hung after printing `ok=0` counts as failed for (i) and (ii).
- **Width 3** (r5 §6): `dev_ratio(3)` is descriptive. Closed at the merged centre (1.5 to 1.65); the result reports
  whether `dev_ratio(3)` ≤ 1.3, the only value where it wins anywhere (Llama-2, Bonnie's model).
- **x4 kai**: descriptive. If `kernel=` names an i8mm (or SVE / SME) kernel rather than DOTPROD, or `q4_buft=` is not
  the KleidiAI buffer type, r5 §2.1's tile cost (c 2.5) does not hold and T2 to T5 are read with that note.
- **Profiling queue in x4m.** The together cell and the mixed arm feed the GPU on the profiling queue (one event per
  launch); R74's `x3 tog` used the plain queue. Comparisons inside R75 (two columns against one, mixed arm against
  the one-column cell, every ratio of §5.2) use the same queue on both sides and are not affected. The only
  comparison with R74 cells is T1, whose band is centred on R74's `x3 togsum` values: T1 is scored as written, and a
  miss of a few percent low names the queue as a candidate cause beside the clocks. K_meas (§5.3) comes from x4s,
  which uses the profiling queue in lead mode by design; x4s's feeder mode uses the plain queue as R74 did.

### 5.2 The together cell: which contention model (x4m; Bonnie r5 §2.3 to §2.5 on Ebony's r4 §1.2)

The line is **joint**. Ebony's base (r4 §1.2: GPU ratio ≥ 0.95 or sum ratio ≥ 0.94 → `pair`; GPU ratio ≤ 0.90 and
sum ratio ≤ 0.89 → carried-over) with Bonnie's two changes (r5 §2.3, §2.5): the thresholds move to ≥ 0.93 and
≤ 0.89, because three *device* `x1gpu` cells (R69, R72) put the GPU ratio's ceiling at 0.952; and below 19 GB/s of
CPU alone at two columns the GPU ratio decides alone. Ebony accepted both (r5 §5), and also the caveat that a tile
`pair` reading is `pair` at light CPU draw. The mixed-arm line is Bonnie's (r5 §2.5 item 4); Ebony agrees it is the
right arm and gives its band (T11).

With T8 (GPU ratio), T9 (sum ratio), T2 (`cpu_alone_gbs` at two columns) and T11 (mixed arm):

1. **Tile reading** (two-column cell):
   - `pair` if T8 ≥ 0.93 or T9 ≥ 0.94;
   - carried-over if T8 ≤ 0.89 and, only when T2 ≥ 19, also T9 ≤ 0.89 (below 19 the GPU ratio decides alone);
   - otherwise undecided.
   - A `pair` tile reading with T2 < 19 is "`pair` at light CPU draw" (the tile makes the CPU draw about 14 GB/s,
     not the lane's 20 to 22).
2. **Mixed-arm reading** (T11, `x4 togmixsum both_cpu_vs_tog1_both_cpu`, at the printed `gpu_n_cols` of 4 or 3;
   informative only if `x4 mcsum dev_ratio` at that `n_cols` ≥ 1.3; skipped if `gpu_n_cols=0`): ≥ 1.20 `pair`; 1.05 to 1.20 `linear`;
   < 1.05 contradicts the `x1gpu` cells, read as carried-over (`bonnie`, rigid) and flagged for explanation.
3. **Model for the stack** (§5.3):
   - mixed `pair` and tile `pair` → **`pair`**;
   - mixed `linear` → **`linear`**, whatever the tile says;
   - mixed < 1.05 → **`bonnie`** (rigid), with `linear` printed beside it;
   - mixed `pair`, tile undecided or carried-over → **undecided**: `linear` and `pair` both printed;
   - mixed not measured or not informative → the tile alone: tile `pair` with T2 ≥ 19 → `pair`; tile `pair` with
     T2 < 19 or tile undecided → undecided; tile carried-over → `linear`.
4. **The CPU at width 2** (Bonnie r5 §0 item 4, Ebony r4 §1.3): every width-2 stack assumes the W-position CPU
   kernel (c(2) = 1.45, exact on random data only, *host*). Stock ggml at ne1 = 2 is predicted at T3 ≈ 0.55. If T3
   < 0.80 the width-2 stacks are written "conditional on building the W-position kernel"; that kernel is a build
   item of any width-2 lane in every outcome.

### 5.3 From the outcome to K and the prices (*estimated* from *device* inputs)

**In-situ K.**
- *K_meas* = the median, over the lead-mode shares where the mean `wait_ms` of the two `x4 split` arms is > 0.1 ms
  (the GPU side binds), of the two-arm mean of (`gpu_ms` − `gpu_dev_ms`) × 1000 µs. That is START − QUEUED of the
  GPU event with the launcher on cpu 4 (flush, spin). If no share has `wait_ms` > 0.1, K_meas is the same median
  over all lead shares, and the result says that K is bounded from above only (r2 §2.4).
- Cross-checks (descriptive): K_x4b = best `ovh_us` + 40 µs (r2 §2.4); K_r = (r − 0.621) × `t_cpu_alone_ms` ÷ 0.521
  at an interior best ratio r.
- **K_eff = K_meas + H** (hand-off added to the in-situ K; Ebony r3 §1.2, agreed by Bonnie r4 §7.1).
- *Design K* for each design of §5.1 that passes "worth a build", replacing the launch part of K_meas:
  Y14: `ovh_us_per_launch_replay` + H; F: best mode's `rtt_p50_us` + `iv_p50_us` + H; A: gate `set_to_start_p50_us`
  + H; B / B+: best `ovh_us` + 40 + H. K_best = the least design K. Design K is a price of an unbuilt mechanism,
  not a measurement.
- If E is killed for gate‖up, the four-stage model does not hold; the lead prices the stack descriptively at
  K_eff × 7 / 4 per stage and says so.

**cg.** cg = `x4 mcsum n_cols=2 dev_ratio=` (the conservative reading, r3 §1.1). If `_mc3` fails,
cg is "not measured": the stack is printed at the merged centre cg 1.25 marked "cg not measured", and 12 is not
priceable through width 2 (§5.3 line).

**Prices, per subject** (scripts as committed, unchanged; Python `-I`):
- `old` (R74's form, kept for comparison with R74): `bonnie_tree_r2.old(s, K)`.
- `pair`: `bonnie_tree_r2.toks(s, 2, K, "pair", cg=cg)` with Bonnie's slack (J0 0, J1 0.107), and with Ebony's
  slack (`J0=60, J1=0`) beside it.
- `linear`: `bonnie_joins_r5.toks_linear(s, K, cg)`.
- `bonnie` (only if §5.2 reads it): `bonnie_tree_r2.toks(s, 2, K, "bonnie", cg=cg)`.
- All at C 40, trained head, C7 head × 0.5, C5 body × 0.93, O 5.2, every unbuilt lever at its centre: the stack is
  a price, not a plan.
- Printed at K_eff and at K_best, in the model(s) §5.2 names, and `old` at K_eff, with the gap to 79.2 ms per
  subject.

**At the merged centres** (K_eff ≈ 300 + 5 to 90 µs of hand-off, cg 1.25; *estimated*, lead's run of the scripts
on 2026-11-23): `old` 10.51 / 10.37 / 10.06; `linear` 11.61 / 11.57 / 11.15; `pair` 11.89 / 11.89 / 11.44 tok/s
at K 300. **No variant prices 12 on DeepSeek at the most likely K.**

**The line at which 12 on all three becomes priceable** (written now; K for 12.00 tok/s, *estimated*, scripts as
committed; µs of K_eff, Yi / Llama-2 / DeepSeek; DeepSeek binds):

| cg | `linear` | `pair` (Bonnie slack) | `pair` (Ebony slack 60 µs) |
|---|---|---|---|
| 1.00 | 312 / 317 / 253 | 312 / 317 / 253 | 256 / 261 / 183 |
| 1.10 | 286 / 284 / 218 | 302 / 303 / 239 | 242 / 244 / 164 |
| 1.20 | 260 / 252 / 184 | 292 / 291 / 226 | 228 / 227 / 147 |
| 1.25 | 248 / 236 / 168 | 287 / 285 / 219 | 222 / 220 / 138 |
| 1.30 | 235 / 221 / 151 | 282 / 279 / 213 | 216 / 212 / 130 |
| 1.45 | 199 / 177 / 104 | 255 / 245 / 177 | 178 / 166 / 81 |
| 1.70 | 143 / 107 / 30 | 205 / 184 / 112 | 109 / 82 / none |
| 2.00 | 80 / 29 / none | 150 / 116 / 39 | 28 / none / none |

`old` (R74's form): Yi at K ≤ 33, Llama-2 and DeepSeek at no K. Width 1 on the lane: no K reaches 12 on Llama-2 or
DeepSeek in any variant (r3 §2).

- **Rule (scored):** 12 on all three is **priceable** after R75 if and only if K_eff (or K_best for a design that
  passed "worth a build", written as conditional on that build) is ≤ the DeepSeek value of the table at the
  measured cg, in the model §5.2 names; for `pair` the Ebony-slack column is used (the stricter); for an undecided
  reading the `linear` column; for `bonnie` its own line (at cg 1.30: K ≤ 43). Interpolate linearly in cg.
- In plain numbers at the centre cg 1.25: K_eff ≤ **168 µs** (`linear`) or ≤ **138 µs** (`pair`, Ebony slack). The
  merged band for K_meas is 170 to 400 µs, so a priceable line needs K at the bottom of the band and H near 0,
  or a design K that passes.
- "Priceable" means a price at the centres of unbuilt levers (trained head, C7, C5, the W-position CPU kernel, the
  lane itself), not a plan and not a device claim.

### 5.4 Residency (x4r): descriptive, with this reading

- *Held* means all three (R74 §5 criteria): the median CPU read over passes index 5 to the end ≥ 0.80 × (29.90 −
  0.36 × the arm's `gpu_gbs`); the median majflt per pass ≤ 50; and the `x4 res done` line.
- Footprint: 1.5 GB GPU + 2.6 GB anonymous = 4.1 GB (+ the app), against R74's x3r 3.45 GB (held, pressure showing)
  and R72's anon42 (4.2 GB anonymous, not held).
- **Held:** 4.1 GB with 1.5 GB pinned stays resident for 15 s. That covers Llama-2's and DeepSeek's totals and
  their GPU share up to K about 300 at any cg (Bonnie r3 §3.4), with anonymous memory standing in for the file half.
  Evidence for, not proof (the app's memory, a decode's duration and file-backed reclaim are outside the arm).
- **Not held, the block completes:** as R74 §5's "not held" reading; the per-pass lines say where it fell.
- **Killed in x4r** (no `x4 res done`, no `x4r end`; the app restarted or the JSON stops in x4r): **not a void.**
  It is scored as "not held (killed at pass n, or before pass 0 if no pass line printed)". It means: 1.5 GB of GPU
  buffers beside a 2.6 GB anonymous arena is not survivable in this app state. It does **not** close Llama-2 or
  DeepSeek: a clean file page costs a drop and a read where an anonymous page costs a compress and a fault, and
  R69 held 4.2 GB file-backed. It says a lane on those two needs its CPU half file-backed (no repack of the CPU
  rows), and that its GPU share above R74's 0.85 GB is untested until a file-backed arm runs (Bonnie r3 §3.4,
  r5 §7; a later kit). Yi stays covered by R74's x3r. The `x4 res stage=` lines and the last pass line are
  reported with the MemAvailable they show.

### 5.5 Void (kit)

- The APK hash differs (the owner checks the downloaded APK against `r35/SHA256SUMS`): the whole round is void.
- Power saving reads on: every block run in that state is void.
- The charger reads plugged: that block is void.
- The process is killed during a block other than x4r: that block only is void (the earlier blocks are saved). In
  x4r a kill is the outcome of §5.4, not a void.
- A child process killed at its timeout is not a void of the parent block (§5.1 "not measured" rules).

**Not voids (conditions).** Reported beside every number they touch: CPU clocks below max (`ramp_ok=0`,
`mhz_range`, `max_mhz`), GPU clock below 840 MHz (`gpu_mhz`, `gpu_max_mhz`), thermal headroom (`hr`), the thermal
lines before and after each block, `x3 cond` (THP, devfreq, `thermal_pwrlevel`), off-core or misplaced threads
(M / U / P / C labels, `lplace`), gate timeouts, the deep-sleeping list label, and **whether the cool start was
met** (an earlier probe in the session, or a hot first gate). The rules use the cells as built; a `ramp_ok=1` subset
is given descriptively beside each rule input. A kernel that fails to build or a skip line is a result, not a void.
Misses are written as misses.

## 6. Claim rule

Not a goal claim. No model is loaded; no number from R75 is a phone token rate. K, cg and the stacks are *estimated*
from *device* inputs.

## 7. What a result means

- **K_eff above the line (the most likely case, §5.3):** the GPU lane's join cost is measured, and with it the last
  open term of the lane. The campaign verdict (12 tok/s out of reach on all three by priced levers) is restated with
  K, cg and the contention model in numbers, per subject. The lane remains a 10 to 11.5 tok/s lever at its centres.
- **A design passes "worth a build" and its K_best is under the line:** that design goes on the host list (lane
  design, numerics emulation with the lane's kernel, W-position CPU kernel). A device test of a lane needs its own
  pre-registration and the owner's permission, and R73 stays held.
- **cg > 1.70:** width 2 through the lane is closed as a route to 12, whatever K is.
- **x4r killed or not held:** the residency reading of §5.4; it constrains a lane on Llama-2 and DeepSeek, it does
  not close them.
- **`_mc3` fails:** cg and the contention model stay unmeasured; the result prints the stacks at cg 1.25 as "cg not
  measured" and writes 12 as not priceable through width 2.

## 8. The run, for the owner

1. After this file is pushed, download `r35/vollie-c3-probe-r35.apk`, `r35/BUILD-r35.txt` and `r35/SHA256SUMS` from
   `HoeioUser/vollie-kits`. Check that the APK's sha256 is
   `465d97bb7932fc4c9b7c675e1b3e2b78536466f9dec3442d4b8c59405f558af5`.
2. Install it over kit r34. No uninstall is needed (35 > 34, same certificate).
3. Phone state:
   - charger out;
   - power saving off;
   - choose the deep-sleeping list session label in the app (on or off, as you prefer; it is recorded);
   - let the phone rest at least 30 min, screen off, not in use;
   - **run no other probe before this one** (not campaign 2's cap probe either).
4. Research kit page: **Run c3 probe r35 (GPU joins)**. Do not use the phone until it finishes (about 3.6 to 7 min
   of blocks plus up to ten cool-down gates of 3 min each; about 4 to 37 min in total).
5. **Save c3 probe to Downloads.** Send the JSON. It goes into `campaigns/03-dense15-5pct/reports/device/`.
6. If the app is killed (most likely in x4r, the last block), the earlier blocks are already saved. Send the file as
   it is.
