# Pre-registration R72: K1 family cost, subject-shaped layers, 3 and 4 cores, per-type cost, barriers, head, anonymous capacity (c3 probe kit r32)

This file is committed and pushed **before** the APK is uploaded, and it is not edited afterwards. Any change goes
into `R72-amendment-1.md`. The tag `prereg-R72` is the owner's to create. The commit hash and the push time show the
order.

- **Date / author:** 2026-11-22, clyde (lead). This is the second device round of campaign 3. R70 and R71 are
  campaign 2's numbers, so R72 is the next free one.
- **Agent models** (full IDs):
  - lead (clyde): `claude-opus-5-5`, plan `reports/memo-08-work/clyde-r72-plan.md`;
  - bonnie: `claude-fable-5-2`, bands in `bonnie-path.md` §10.4;
  - ebony: `claude-fable-5-2`, confirmation and the L3 reading in `ebony-paths.md` §12;
  - tom: `claude-opus-5-5`, patch 0003, app 0002, kit r32 and parser (`tom-r32-build.md`);
  - annie: `claude-sonnet-5-5`, books.
- **Subjects** (memo-06 §6, `e582606`): Yi-1.5-6B-Chat, Llama-2-7b-chat and DeepSeek-LLM-7B-chat. Reserves are
  Mistral-7B-Instruct-v0.3 and Qwen2.5-7B-Instruct. No model is loaded in R72. The subjects enter only as matrix
  shapes, kept fractions and vocabulary sizes.
- **Goal context:** 12 tok/s on each subject (CAMPAIGN.md Amendment 1). R72 is not a claim round.
- **Labels:** every number in §4 is *estimated* until the result marks it *device*.
- **Starts from:** kit r29's probe (R69) and campaign 2's kit protocol: the r16 cool-down gate before each block, and
  the heat, clock, power and charger records.

## 0. Build under test

| part | file | sha256 |
|---|---|---|
| APK | `vollie-c3-probe-r32.apk`, 16,506,426 B, package `com.insaneutils.vollie.c2`, versionCode 32, versionName 1.32.0 | `08a37bd5896816fef88b6b0bd21fc8d753208be0af1313462e94c25c59f3a50c` |
| BUILD file | `BUILD-r32.txt` (the APK's `kit_build.json`) | `c8e8b9ea2460d0f14971315a7abe08b5af1d24111f1563f777811d680778188b` |
| runtime probe rows | `patches/llama.cpp-c3/0003-r72-rows.patch` (on 0001 and 0002; changes only `src/llama-vollie-probe.cpp`) | `a8799a36ba581e35fb5a1f0663f3a0abc645c64d27617317235be6d1c4ffb934` |
| app | `patches/app-c3/0002-kit-r32-probe.patch` (on app-c2 0001 to 0012 and app-c3 0001) | `a54d2cc8cac009377651f88c11e8ac9483c7941fc51b1a19da9766c5a36f0343` |
| libllama.so | in the APK | `350b57c3fd3cfd2b4018b5a5bc56dbb8695415ba2bac599740ad64f571134e63` |
| libggml-cpu-android_armv8.2_2.so | in the APK | `46e0735d2824ba6f3a156e219e058a701f6024fdbecbb2ef937b3127115a9694` |

- **Identity** (*host*, `tom-r32-build.md` §4). With `VOLLIE_PROBE` unset, the per-chunk running values on Llama-2-7B
  Q4_0 are byte-identical to stock `620d2225c` at `-b 512`, `-ub 32` and `-ub 1`.
- **Inside the probe:** K1, K1m and k1m4 are checked against the stock 4x4 gemv with `memcmp`, with adversarial
  scales. The phone prints these as `identity` lines.
- **Numerics:** none are changed and no model is run.
- **Parser:** `tools/c3_probe_parse.py` (`r32report`, `--selftest`).

## 1. Device state

- The owner's Galaxy Z Flip 3, with power saving **off** and the charger **out**. Both are recorded.
- The deep-sleeping list state is recorded.
- **Cool start:** the phone rests at least 30 min, screen off and not in use, before the run starts. The r16 gate,
  now capped at 180 s, runs before each block.
- No model is loaded (`model_unloaded` recorded).
- The install goes over kit r29 (versionCode 29 < 32).

## 2. What is measured (blocks as built; run order a, l1, l4, l23, l5, l6, l7, x2, anon34, anon42)

| block | content |
|---|---|
| `a` | R69 anchors with R69's timing: p2 read and q4x4 on `4-6`, `7,4-5`, `4-7` |
| `l1` (L1, L1m, L1m4) | K1, K1m and k1m4 at keep 1.0, 0.666, 0.65, 0.60, 0.58 and 0.50; inputs 4096 and 11008; hot and cold; on cpu 4 (A78) and cpu 7 (X1). Stock cells before and after each group (`drift`). Identity lines. 168 cells |
| `l4` (L4) | one layer of each subject's shapes over a cold arena: Yi with kv 512 at keep 0.58, Llama-2 and DeepSeek with kv 4096 at keep 0.60. Stock against the rule: k and v stock; q, o, gate, up and down on K1 or K1m. Core sets 3 A78, X1 + 2 A78 and X1 + 3 A78, in ABBA order. Then `l4sum`, and a ggml stock layer cell |
| `l23` (L2, L3) | 3 and 4 big cores. Read and q4x4 cold; q4x4 hot with a private slice per thread (`l3`, per-thread G weights/s); ggml Q4_0 gemv cold (`l2 kern=ggml_q4_0`) |
| `l5` (L5) | ggml gemv for Q4_0, IQ4_NL, Q4_K, IQ4_XS, Q3_K, IQ3_S, IQ3_XXS, Q2_K, Q6_K and Q8_0, with Q4_0 run again as pass 2. Inputs 4096 and 11008, hot and cold, on `4`, `7`, `7,4-5` and `4-7` |
| `l6` (L6) | graph barriers: chains of 617 (Yi and Llama-2), 579 (DeepSeek) and 225 nodes on `7`, `4-6`, `7,4-5` and `4-7`, plus a spin-barrier arm |
| `l7` (L7) | Q6_K head (with Q8_0 for reference) at vocab 64,000 / 32,000 / 102,400 × 4096, cold, on 3 core sets |
| `x2` | R69's X2 rows, a retry of the same kernels built with `-cl-opt-disable` (`x2r`), and `x2info` |
| `anon34`, `anon42` | anonymous-memory capacity: 3.4 and 4.2 GB anonymous mmap, filled by 3 pinned threads and read for 20 passes or 45 s. Majflt, minflt, Cached, SwapFree, VmSwap and VmRSS are recorded |

**Recorded conditions:**
- every r32 cell's clock range (`mhz_range`), `ramp_s`, `ramp_ok` and `max_mhz`;
- the per-thread `thr` field (cpu, CPU ÷ wall, nivcsw, nvcsw, off-core checks);
- `cpu0_off_calls` in the ggml cells.

**Phone time** (*estimated*): about 20 to 40 min, and about 65 min in the worst case.

## 3. Budget accounting

No numerics change and no model run. Nothing is spent.

## 4. Predictions (*estimated*; centre with 80 % band)

**Source of the bands.** They are the merged bands of `bonnie-path.md` §10.4, confirmed row by row in
`ebony-paths.md` §12, with Ebony's amended L3 reading.

**R69 anchors used.** R69 run 1 (16:46, the scored run) is the anchor. R69 run 2 (17:05) is descriptive only and is
not used.
- The `a` row below is new: it is anchored on R69 run 1 (lead).
- L2 is checked against R69 run 1. Its merged band contains both of R69 run 1's values (28.56 GB/s on 4-7, and
  r4 ÷ r3 = 1.00 for 4-7 ÷ 7,4-5), so it is kept as written.
- L4 note: R69 run 1's r of 28.56 lies near the r 30 case that sets the band's upper end. The band is kept as
  merged, so this is stated, not used to move it.
- No other row has an R69 anchor. L1 is a ratio in the same cell, L3 was never measured all-threads-hot, and
  L5 to L7 are new.

| row | quantity | prediction |
|---|---|---|
| a | each anchor cell (read and q4x4 on 4-6, 7,4-5, 4-7) ÷ the same R69 run 1 cell | 1.05 (0.85 to 1.20) |
| L1 | K1 k, A78 (cpu 4), hot, keep 0.58 to 0.60, inputs 4096 / 11008 | **1.85 (1.65 to 2.2)**; cold 0.1 lower (1.75, 1.55 to 2.1) |
| L1 | K1 k, X1 (cpu 7), hot | 1.5 (1.15 to 2.0) |
| L1m | K1m k, A78 / X1 | 1.55 (1.45 to 1.75) / 1.25 (1.0 to 1.6) |
| L1m4 | k1m4 k, A78 / X1 | 1.45 (1.35 to 1.65) / 1.25 (1.0 to 1.6) |
| L1x | k1m4 ÷ K1 on the A78, same group | **0.79 (0.72 to 0.90)** |
| L2 | r4, cold, ggml threads, X1 + 3 A78 (`l2 kern=ggml_q4_0` on 4-7) | 26 (20 to 33) GB/s; r4 ÷ r3 1.06 (0.95 to 1.22) |
| L3 | c4 ÷ c3 = `l3` gw_per_s on 4-7 ÷ on 7,4-5 | **1.08 (0.90 to 1.24)** |
| L4 | t_rule ÷ t_stock, cold, best set (`l4sum`, the better of K1 and K1m) | Yi 0.68 (0.58 to 0.92); Llama-2 0.74 (0.65 to 0.93); DeepSeek 0.75 (0.65 to 0.93) |
| L5 | Q6_K ÷ Q4_0 per weight, hot, 1 thread (cpu 4) | 1.5 (1.15 to 2.0) |
| L6 | empty barriers per token (`l6 kind=ggml`, 617 nodes), 3 / 4 threads | 2.5 (1.0 to 6) / 3.0 (1.2 to 7) ms |
| L7 | head cold, 3 threads (7,4-5): Yi / Llama-2 / DeepSeek | 10 (7 to 16) / 5 (3.5 to 8) / 16 (11 to 26) ms |

How each row is scored:
- A value inside its band is "inside". Outside it is a **miss**, high or low.
- Where a row names two widths, both cells are scored.
- **Where a row lists several cells, the row is scored on its first-named cell**, and the others are reported:
  - L1 and L1x are scored at keep 0.60, input 4096;
  - L3 and L2 use 7,4-5 as the 3-core set;
  - for `a`, every cell is scored separately.

## 5. Decision rules (scored as written)

**Definitions** (from the phone's own lines):
- t_stock(s, set) and t_rule(s, set) are the body ms per token in `l4sum`, for subject s and core set.
- t_head(s) is L7's Q6_K ms on 7,4-5.
- O = 8 ms; L6 is reported beside it.
- The 12 tok/s steady time is 79.2 ms, printed = 0.95 × steady.

- **K1 family (kernel choice, A78).**
  - Keep K1m or k1m4 for the build if L1x ≤ 0.90 on the scored A78 cell and its identity line is 1.
  - Close both K1m and k1m4 if L1x ≥ 0.97. In between, they are parked, and K1 stays the baseline kernel.
  - A variant whose identity line is not 1 is closed, whatever its speed.
- **K1 lever per subject (L4).**
  - **Kill** if t_rule ÷ t_stock ≥ 1.00 on every core set.
  - **Open** if ≤ 0.90 on the best set.
  - Otherwise **parked**.
  - Reported per subject with what generalised: Yi is the grouped-attention subject, Llama-2 and DeepSeek the MHA
    class. One subject open and another killed is "did not generalise", and the lever stays open where it passed.
- **Fourth core (bit-identical).**
  - **Adopted** for the next round if t_stock(X1 + 3 A78) ≤ 0.95 × t_stock(X1 + 2 A78) on at least two subjects.
  - **Closed** if ≥ 1.00 on all three. Otherwise parked.
  - Independently, closed on compute if L3 c4 ÷ c3 < 1.10.
  - **Ebony's L3 reading:** a 4th-core miss (c4 ÷ c3 < 1.10, or a closed L4 line) is labelled "pre-empted" when
    the 4th thread's CPU ÷ wall in the `thr` field is < 0.9, and "compute" otherwise.
  - Both labels are misses as written. The label only says which lever is left: pinning or priority against kernel
    work.
- **K0 per subject (12 tok/s).**
  - P(s) = 0.95 × 1000 ÷ (min over sets and the kept kernels of t_rule + t_head + O).
  - Gap(s) = (that sum) − 79.2 ms.
  - If P(s) < 12, the subject is written as "out of reach with the K1 family and the 4th core", with t_rule, the
    set, t_head, O and the gap in ms. Only the unbuilt levers remain for it (GPU lane, C5 up rule, C6 units of eight,
    C7 head screen), and each must be priced against that gap.
  - It is also given with O = L6 (3 or 4 threads) + 5 ms, as descriptive only.
- **Type filter (L5).**
  - A type replaces Q4_0 as a freeze candidate only if it meets both conditions:
    - its cost per weight ÷ Q4_0 (pass 2 counts, as the mean of passes 1 and 2) is ≤ 1.00 on 4-7, both hot and cold;
    - it has fewer bits per weight than Q4_0 (4.5).
  - A type above 1.30 on 4-7 hot is closed as the "fits-C" path (memo-07 test 8).
  - Otherwise Q4_0 stays the working type at the freeze (R69: C 4.2 GB file-backed holds all three Q4_0 files).
- **Anonymous capacity** (Bonnie's P1 limit, R69 result §3).
  - **Held** means three things together:
    - the median read rate over passes 6 to 20 (pass index 5 to 19, or those that exist under the 45 s cap) is
      ≥ 0.80 × block `a`'s plain read on 4-6;
    - the median majflt per pass is ≤ 50;
    - the arm prints its `done` line.
  - **Reading** against the subjects' resident estimates (file + head + KV at n_ctx 512, *estimated*,
    `bonnie_path.py`): Yi 3.36, Llama-2 4.02 and DeepSeek 4.01 GB.
    - anon42 held: all three subjects can hold a repacked (anonymous) body.
    - anon34 held but anon42 not: only Yi can, with a 0.04 GB margin. Llama-2 and DeepSeek then need a file-backed
      body (Y1, or `--no-repack` as a declared numerics change).
    - anon34 not held: no subject's repacked body is shown resident. Every subject then needs a file-backed body.
  - The app's own memory is not inside the arm, so a held arm is an upper bound on the repacked body, and is
    stated so.
- **GPU lane (x2).** Descriptive only: no new GPU kernel. The `x2r` retry and `x2info` are reported against R69
  Rule D, which already left rank 1 open on reads. No decision is attached.
- **Void (kit):**
  - the APK hash differs (the owner checks the downloaded APK against `r32/SHA256SUMS`);
  - power saving reads on, which voids every block run in that state;
  - the charger reads plugged, which voids that block;
  - the process is killed during a block, which voids that block. In `anon34` and `anon42` a kill is also reported
    descriptively ("killed at pass n"). That size's capacity is then "not measured".
- **Not voids (conditions).** These are reported beside every number they touch:
  - clocks below max (`ramp_ok=0`, `mhz_range`);
  - off-core workers (`cpu_ok`, `thr`, `cpu0_off_calls`);
  - gate timeouts;
  - the deep-sleeping list.
  The rules use the cells as built. A subset with only `ramp_ok=1` is given descriptively beside each rule input.
- Misses are written as misses.

## 6. Claim rule

Not a goal claim. No model is loaded, so no tok/s number from R72 is a phone token rate. K0's P(s) is *estimated*
from *device* inputs.

## 7. What a result means

- **L1x ≤ 0.90:** the next K1 build uses k1m4 (or K1m).
- **L4 open on a subject:** K1 is built into a runtime patch for that subject's matrices. It needs per-chunk PPL at
  `-b 512` and `-ub 1` at its frontier before any device use.
- **K0 below 12 on all three:** 12 tok/s is written as out of reach with the CPU levers measured so far, with the
  three gaps in ms. The research goes to the GPU lane and to the levers that cut compute.
- **anon34 not held:** the repacked Q4_0 path is closed for residency on this phone. The next residency work is a
  file-backed body.

## 8. The run, for the owner

1. After this file is pushed, download `r32/vollie-c3-probe-r32.apk`, `r32/BUILD-r32.txt` and `r32/SHA256SUMS` from
   `HoeioUser/vollie-kits`. Check that the APK's sha256 is `08a37bd5896816fef88b6b0bd21fc8d753208be0af1313462e94c25c59f3a50c`.
2. Install it over kit r29. No uninstall is needed, since 32 > 29.
3. Check the phone state:
   - charger out;
   - power saving off;
   - choose the deep-sleeping list session label in the app (on or off, as you prefer; it is recorded);
   - let the phone rest at least 30 min, screen off.
4. Research kit page: **Run c3 probe r32**. Do not use the phone until it finishes (about 20 to 40 min, at most
   about 65 min).
5. **Save c3 probe to Downloads.** Send the JSON. It goes into `campaigns/03-dense15-5pct/reports/device/`.
6. If the app is killed in an anon block, the earlier blocks are already saved. Send the file as it is.
