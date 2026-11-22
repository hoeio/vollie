# R72 result: K1 family cost, subject-shaped layers, 3 and 4 cores, per-type cost, barriers, head, anonymous capacity (c3 probe kit r33)

### R72 Probe: K1 / K1m / k1m4, subject layers, fourth core, types, barriers, head, anonymous capacity (kit r33, no model)

- **Date:** 2026-11-22, 17:59:55 to 18:40:11 local. **Label:** *device* for every value the phone printed. Ratios,
  P(s), gaps and byte budgets derived from them are arithmetic on *device* inputs and are marked *estimated* where
  they predict a token rate.
- **Sources:**
  - [device/r72-c3probe-r33-2026-11-22T1759.json](device/r72-c3probe-r33-2026-11-22T1759.json), sha256
    `cbc332ddc4879062623a537373ea0eae5e0cac3517198590b520115ced8b0cb2`, committed `37129fc`.
  - Parser `tools/c3_probe_parse.py` (`--selftest` ok) for the tables; every rule input re-read from the raw lines
    with the lead's own short code (`python3 -I`, not committed).
  - Pre-registration [R72](../prereg/R72-probe-k1-cores-types-anon-device.md) (`364c438`) and
    [amendment 1](../prereg/R72-amendment-1.md) (`dc19d32`, committed 17:57:50, before the run started at 17:59:55).
    Attempt 1 on kit r32 ([device/r72-attempt1-c3probe-r32-2026-11-22T1749.json](device/r72-attempt1-c3probe-r32-2026-11-22T1749.json))
    ran no block and is not scored (amendment 1).
- **Subjects** (memo-06 §6): Yi-1.5-6B-Chat, Llama-2-7b-chat, DeepSeek-LLM-7B-chat. No model was loaded; they enter as
  shapes. Goal 12 tok/s on each.
- **Scored by:** clyde (lead), `claude-opus-5-5`.

## 1. Void checks: none met

| void | state | source |
|---|---|---|
| APK hash differs | **not triggered (lead's check of the build; the owner's download check is assumed, see note)** | The JSON reads `kit=r33`, `app=1.33.0+33`. The probe prints no APK or library hash line; the only identity lines are the kernel `memcmp` lines (§2). The kit files in `lab/dist-c3/r33/` hash to the amendment's values: APK `1917aa4c…7db7c`, BUILD `d2272777…084df` (lead, `shasum -a 256`, 2026-11-22). |
| power saving on | not triggered | `powerSave: false` at start, at end and before each of the 10 blocks |
| charger plugged | not triggered | `plugged_start` false; `plugged_before` false in all 10 blocks |
| process killed | not triggered | every block has its end line and rc > 0, no `error` field: a 19.3 s, l1 523.2, l4 248.6, l23 58.8, l5 579.0, l6 26.8, l7 112.7, x2 14.0, anon34 5.9, anon42 6.2 s; `stopped_early` false |

Note: the void reads "the owner checks the downloaded APK against `r32/SHA256SUMS`" (r33 after amendment 1). The
owner's statement of that check is not in the JSON. The lead has checked the files that were uploaded; if the
owner's installed APK differed, every block is void. *Owner: please confirm the r33 sha256 you checked.*

Recorded, not voids: deep-sleeping list **on**; `model_unloaded=olmoe-Mi.gguf`; free storage 42.5 GB at start; battery
77 % to 63 %, 25.5 to 31.3 °C; at block `a` MemAvailable 2.60 GB, zram swap free 1.39 of 4.19 GB.

## 2. Conditions beside every number

- **Gates:** no timeout. Each of the 10 gates passed (headroom below 0.65) after 30 to 160 s.
- **Identity:** every `l1 identity` line (inputs 4096 and 11008, keep 1.00, 0.60, 0.58) reads k1=1 k1m=1 k1m4=1; every
  `l4 identity` line (three subjects) reads k1=1 k1m=1. No variant is closed on identity.
- **The X1 was mostly not where the threads were pinned.** The `thr` field gives, per thread, off-core samples ÷
  samples and the last cpu seen. The thread pinned to cpu 7 was off cpu 7 in every sample, last seen on cpu 0 to 3,
  with the X1 policy at 844 MHz (its floor), in:
  - all 18 L4 arms on `7,4,5`, and the first stock arm (pos 0) on `4,5,6,7` for all three subjects;
  - all five L3 cells that include cpu 7 (`7`, `7,4,5`, `4,5,6,7`), so the cpu-7 thread gives 3.95 G weights/s, which
    is a small-core rate (A78 threads: 20.5);
  - L2 `read` and `q4x4` on `7,4,5` and `4,5,6,7` (cpu_ok 2/3 and 3/4);
  - 65 of 84 L1 cells on cpu 7 (all `4096` cells and the `11008 hot` group). The 17 `11008 cold` cells on cpu 7 that
    did run on the X1 (cpu_ok 1/1, X1 at 2,841) are the only clean X1 L1 cells.
  - In L4 the X1 thread came onto cpu 7 from arm pos 1 of `4,5,6,7` onward and stayed for that set.
  - Also off-core: one A78 thread in the first stock arm (pos 0) on `4,5,6` for Llama-2 and DeepSeek, and the
    `stock_before` cell of the L1 `11008 hot` group on cpu 4 (its drift reads 0.199, so that group's k values are not
    usable as ratios).
  - CPU ÷ wall stayed 0.96 to 1.00 for these threads, so CPU ÷ wall alone does not detect this.
  - Why the X1 refused its pinned thread is not measured. A hypothesis for the next round: the vendor's core control
    isolates the prime core at low load and moves pinned tasks off it.
- **Clocks.** `ramp_ok=1` means each used policy reached its *current* `scaling_max_freq`, not the hardware maximum.
  - 191 of 408 cells are `ramp_ok=1` with `scaling_max_freq` at the hardware maximum (A78 2,419, X1 2,841 MHz).
  - The L4 arms ran with the cap lowered: 54 of 63 L4 cells read `max_mhz` p4 at 1,555 to 2,150 of 2,419 (Llama-2 and
    DeepSeek at 1,555 throughout; Yi at 2,419 on `4,5,6`, 2,150 on `7,4,5`, 1,555 to 1,785 on `4,5,6,7`).
  - All L1 cells on cpu 4 ran with the A78 at 2,419 (`ramp_ok=1`, 83 of 84).
  - L7, L6 (A78 sets) and L5 4-thread cells are mostly at full caps.
- **ggml cells:** `cpu0_off_calls` 0 in every L4 and L2 ggml cell; non-zero in some L5 single-thread cells (for example
  Q4_0 at 11008 hot on cpu 4: 3,822; IQ4_XS 4096 hot cpu 4: 17,748) and in L6 cells on `7` and `7,4,5`.
- **ABBA spread:** in L4 the first stock arm (pos 0) is slower than the last (pos 5) on every A78 set (Yi `4,5,6`
  4.657 vs 3.376 ms per layer). Pos 0 is also where an off-core thread appears. The rule uses ABBA means as built.

## 3. Bands (§4), scored as written

Scoring choices where the text leaves a choice, stated before reading the outcome of each:
- L1, L1m, L1m4, L1x: keep 0.60, input 4096 (§4 text). "Two widths" is read as the slash-separated cells of a row
  (A78 / X1; 3 / 4 threads; subjects). L1m and L1m4 are scored hot, the regime of the L1 row they extend.
- L6 3 threads: `4,5,6`, the first-named 3-thread set in §2's set list (`7`, `4-6`, `7,4-5`, `4-7`); `7,4,5` is reported.
- L5: Q4_0 as the mean of passes 1 and 2 (the type filter's own basis); pass 1 alone gives the same verdict.

| row | cell | band (*estimated*) | *device* | verdict | conditions |
|---|---|---|---|---|---|
| a | read 4-6 ÷ R69 (28.79) | 0.85 to 1.20 | 29.90 → 1.039 | inside | cpu_ok 3/3 |
| a | read 7,4-5 ÷ R69 (30.92) | 0.85 to 1.20 | 29.74 → 0.962 | inside | 2/3 |
| a | read 4-7 ÷ R69 (29.91) | 0.85 to 1.20 | 29.66 → 0.992 | inside | 4/4 |
| a | q4x4 4-6 ÷ R69 (30.09) | 0.85 to 1.20 | 29.10 → 0.967 | inside | 3/3 |
| a | q4x4 7,4-5 ÷ R69 (28.56) | 0.85 to 1.20 | 30.80 → 1.078 | inside | 3/3 |
| a | q4x4 4-7 ÷ R69 (28.56) | 0.85 to 1.20 | 30.01 → 1.051 | inside | 4/4 |
| L1 | K1 k, A78, hot | 1.65 to 2.2 | 1.992 | inside | on-core, 2,419 MHz, ramp_ok 1 |
| L1 | K1 k, A78, cold | 1.55 to 2.1 | 2.239 | **MISS** high | on-core, 2,419, ramp_ok 1 |
| L1 | K1 k, X1, hot | 1.15 to 2.0 | 1.343 | inside | **not an X1 reading**: off cpu 7 in 581/581 samples, X1 844 MHz, ramp_ok 0 |
| L1m | K1m k, A78 | 1.45 to 1.75 | 1.921 | **MISS** high | on-core, 2,419 |
| L1m | K1m k, X1 | 1.0 to 1.6 | 1.332 | inside | not an X1 reading (586/586 off) |
| L1m4 | k1m4 k, A78 | 1.35 to 1.65 | 1.959 | **MISS** high | on-core, 2,419 |
| L1m4 | k1m4 k, X1 | 1.0 to 1.6 | 1.337 | inside | not an X1 reading (583/583 off) |
| L1x | k1m4 ÷ K1, A78 | 0.72 to 0.90 | 0.983 | **MISS** high | same cells as above |
| L2 | r4, ggml Q4_0 cold, 4-7 | 20 to 33 GB/s | 26.67 | inside | ramp_ok 1, caps 2,150 |
| L2 | r4 ÷ r3 (7,4-5: 28.44) | 0.95 to 1.22 | 0.938 | **MISS** low | X1 placement in ggml cells not recorded |
| L3 | c4 ÷ c3, `l3` gw/s 4-7 ÷ 7,4-5 | 0.90 to 1.24 | 65.54 ÷ 45.01 = 1.456 | **MISS** high | cpu-7 thread off-core in both (3.95 G/s each): the ratio measures a third A78, not the X1 |
| L4 | Yi, best set (4-7), better of K1/K1m | 0.58 to 0.92 | 0.960 (K1m) | **MISS** high | stock pos 0 arm X1 off-core; caps 1,555 to 1,785 |
| L4 | Llama-2 | 0.65 to 0.93 | 0.948 (K1m) | **MISS** high | same; caps 1,555 |
| L4 | DeepSeek | 0.65 to 0.93 | 0.987 (K1m) | **MISS** high | same; caps 1,555 |
| L5 | Q6_K ÷ Q4_0, hot, cpu 4, 4096 | 1.15 to 2.0 | 0.0781 ÷ 0.0807 = 0.968 | **MISS** low | ramp_ok 1, 2,419 |
| L6 | 617 nodes, 3 threads (4,5,6) | 1.0 to 6 ms | 0.183 | **MISS** low | ramp_ok 1; on `7,4,5`: 1.359 (inside; ramp_ok 0, cpu0_off 200) |
| L6 | 617 nodes, 4 threads | 1.2 to 7 ms | 0.211 | **MISS** low | ramp_ok 1 |
| L7 | Yi head Q6_K, 7,4-5 | 7 to 16 ms | 9.039 | inside | ramp_ok 1 |
| L7 | Llama-2 | 3.5 to 8 ms | 4.396 | inside | ramp_ok 1 |
| L7 | DeepSeek | 11 to 26 ms | 13.881 | inside | ramp_ok 1, caps 2,150 |

**14 of 26 inside, 12 misses.** Three of the 14 inside (the X1 L1 cells) are not X1 measurements. The `a` anchors
reproduce R69 run 1 within ±8 % on all six cells.

## 4. Decision rules (§5), scored as written

| rule | inputs (*device*) | verdict |
|---|---|---|
| **K1 family** (A78) | L1x = 0.983 (≥ 0.97); K1m ÷ K1 = 0.964; identity 1 for all three kernels | **K1m and k1m4 closed.** K1 stays the baseline kernel. |
| **K1 lever, Yi** (L4) | t_rule ÷ t_stock (better of K1/K1m): 4-6 1.098, 7,4,5 1.222, 4-7 **0.960** | **parked** (not ≥ 1.00 on every set; not ≤ 0.90) |
| **K1 lever, Llama-2** | 4-6 1.042, 7,4,5 1.191, 4-7 **0.948** | **parked** |
| **K1 lever, DeepSeek** | 4-6 1.011, 7,4,5 1.166, 4-7 **0.987** | **parked** |
| K1 lever, generalisation | the grouped-attention subject (Yi) and the two MHA subjects give the same verdict | generalised (parked on all three). K1 alone (K1m closed): 0.972 / 0.956 / 0.998, still parked |
| **Fourth core** | t_stock(4-7) ÷ t_stock(7,4,5): Yi 3.938 ÷ 4.562 = **0.863**, Llama-2 4.872 ÷ 7.024 = **0.694**, DeepSeek 4.699 ÷ 7.040 = **0.667**; L3 c4 ÷ c3 = 1.456 (≥ 1.10) | **Adopted** (≤ 0.95 on three subjects; not closed on compute). See the caveat below |
| Ebony's L3 label | no fourth-core miss | not applied. The 4th thread's CPU ÷ wall was 0.994 while it ran off cpu 7 in 1,415 of 1,415 samples, so the label's proxy would have read "compute" here |
| **K0 per subject** | §5 below | **P(s) < 12 on all three**: out of reach with the K1 family and the 4th core |
| **Type filter** (L5) | no L5 cell runs cold on 4-7 (cold cells are on `4` and `7` only), so "≤ 1.00 on 4-7 both hot and cold" cannot be met by any type | **No type replaces Q4_0.** Q4_0 stays the working type at the freeze |
| type filter, close clause | 4-7 hot ÷ Q4_0 (4096): IQ4_XS 1.459, Q3_K 1.731, IQ3_S 2.934, IQ3_XXS 1.862 (all > 1.30); Q2_K 1.170, IQ4_NL 1.103, Q4_K 0.948 | **closed as the fits-C path:** IQ4_XS, Q3_K, IQ3_S, IQ3_XXS. Q2_K not closed (parked). IQ4_NL and Q4_K have 4.5 bpw, so they are not candidates. The rule names no input; at 11008 IQ4_XS reads 1.238 (would not close) and Q2_K 0.883, IQ4_NL 0.900, Q4_K 0.847 |
| **Anonymous capacity** | §6 below | **anon34 held, anon42 not held** |
| **GPU lane** | §7 below | descriptive |

**Caveat on the fourth core (stated, not used to change the verdict).** The `7,4,5` arms never had their cpu-7
thread on the X1, so the scored ratio is "X1 + 3 A78" against "2 A78 + one thread on a small core", not against
"X1 + 2 A78". Descriptive evidence that does not depend on this: the ggml stock layer cells (cpu0_off_calls 0,
CPU ÷ wall ≥ 0.997, ramp_ok 1) give 4-7 ÷ 4-6 = 0.823 / 0.824 / 0.821 and 4-7 ÷ 7,4,5 = 0.846 / 0.940 / 0.918 (Yi /
Llama-2 / DeepSeek). The fourth core helps by 6 to 18 % on every subject in the runtime's own path.

**Caveat on the K1 lever (descriptive).** The only set where the rule's ratio dips below 1.00 is 4-7, and there the
stock mean includes the pos 0 arm whose X1 thread was off-core.
- With only arms where every thread stayed on its cpu, the rule is slower than stock on every set and every subject:
  4-7 K1 ÷ stock = 1.084 / 1.050 / 1.133, and 4-6 = 1.31 / 1.24 / 1.20 (one stock arm each).
- Against the ggml stock layer (KleidiAI, the path the app runs), K1 on 4-7 is 1.016 / 1.060 / 1.076.
- K1 saves 36 / 29 / 30 % of the bytes but runs at 16.2 / 17.3 / 17.1 GB/s effective, against ggml's 25.8 to 26.1 GB/s.
  It is compute-bound at the capped clocks of this run.

## 5. K0 per subject (12 tok/s: 79.2 ms printed)

P(s) = 0.95 × 1000 ÷ (t_rule + t_head + O), with t_rule the minimum over sets of the kept kernel's body (K1 only:
K1m is closed), t_head the L7 Q6_K value on 7,4-5, O = 8 ms. *Estimated* from *device* inputs.

| subject | t_rule (set) | t_head | O | sum | P(s) | gap to 79.2 ms |
|---|---|---|---|---|---|---|
| Yi-1.5-6B | 122.53 ms (K1, 4-7) | 9.04 | 8 | 139.57 | **6.81** | **+60.4 ms** |
| Llama-2-7B | 149.06 ms (K1, 4-7) | 4.40 | 8 | 161.46 | **5.88** | **+82.3 ms** |
| DeepSeek-7B | 140.66 ms (K1, 4-7) | 13.88 | 8 | 162.54 | **5.84** | **+83.3 ms** |

- **All three are written as out of reach with the K1 family and the 4th core.** What is left for each is the unbuilt
  levers: GPU lane, C5 up rule, C6 units of eight, C7 head screen. Each must be priced against these gaps.
- Descriptive, O = L6 (4 threads) + 5 ms = 5.21 / 5.21 / 5.20 ms: P = 6.95 / 5.99 / 5.95, gaps +57.6 / +79.5 / +80.5 ms.
- Descriptive, the ggml stock body (4-7) in place of t_rule: P = 6.90 / 6.21 / 6.22. Stock ggml is as fast as or faster
  than the K1 rule on every subject.
- The `ramp_ok=1` subset does not change K0: the K1 4-7 arms that set t_rule are all `ramp_ok=1` and on-core. They
  ran under a lowered cap (A78 1,555 to 1,785 MHz).
- Cross-check: R69's best CPU price at c_b as written was 6.8 / 5.9 / 5.9 (*estimated*). R72's layer-shaped *device*
  inputs give 6.81 / 5.88 / 5.84.

### What 12 tok/s now needs, per subject (*estimated*, arithmetic on *device* inputs)

Body budget = 79.2 − t_head − 8 ms. Read rate r = 29.90 GB/s (block `a`, read on 4-6). Bytes are the probe's own
per-layer sizes (MiB) × layers.

| subject | body budget | body now (K1 / ggml) | factor needed | stock body bytes | max bytes at r | as share of stock | K1 rule bytes at the read floor |
|---|---|---|---|---|---|---|---|
| Yi | 62.2 ms | 122.5 / 120.6 | 0.51× | 3.11 GB | 1.86 GB | 0.60 | 1.98 GB → 66.2 ms → 11.4 tok/s |
| Llama-2 | 66.8 ms | 149.1 / 140.6 | 0.45× | 3.64 GB | 2.00 GB | 0.55 | 2.57 GB → 86.1 ms → 9.6 tok/s |
| DeepSeek | 57.3 ms | 140.7 / 130.8 | 0.41× | 3.42 GB | 1.71 GB | 0.50 | 2.41 GB → 80.5 ms → 9.3 tok/s |

In plain words:
- **Yi** needs its body to take half its present time. That means reading at most 60 % of the stock bytes at the full
  read rate. The K1 rule at keep 0.58 touches 64 % of them, so even a K1 that ran at the read rate (it now runs at
  0.54 of it) stops at about 11.4 tok/s.
- **Llama-2** needs at most 55 % of its stock bytes at the full read rate. The K1 rule at keep 0.60 (k,v dense) stops at
  9.6 tok/s even at the read floor.
- **DeepSeek** needs at most 50 % of its stock bytes; its 102,400-word head costs 13.9 ms. K1 at the read floor stops at
  9.3 tok/s.
- On the CPU alone, no priced lever closes these gaps. The only measured way above r is the CPU + GPU read sum: 46.5
  GB/s (§7). It needs a GPU Q4_0 kernel that is not yet built; the probe's runs at 1.88 GB/s.

## 6. Anonymous capacity

Held = median read over passes 6 to 20 (index 5 to 19) ≥ 0.80 × block `a` read on 4-6 (0.80 × 29.90 = 23.92 GB/s),
median majflt per pass ≤ 50, and the `done` line.

| arm | passes | median GB/s, idx 5 to 19 | median majflt | done | verdict | also recorded |
|---|---|---|---|---|---|---|
| anon34 (3.4 GB) | 20 | **28.02** | 0 | yes | **held** | fill 3.0 s; VmRSS 3.37 GB; VmSwap 64 to 70 MB; MemAvailable before 2.37 GB |
| anon42 (4.2 GB) | 20 | **23.00** (0.769 × a) | 1 | yes | **not held** (read rate) | passes 13 to 17 fell to 16.9 to 19.8 GB/s; one pass with 110 majflt; VmSwap 83 to 90 MB; MemAvailable before 3.61 GB |

**Reading (as written):** anon34 held, anon42 not.
- Only Yi can hold a repacked (anonymous) body: estimated resident 3.36 GB against 3.4 GB, a 0.04 GB margin.
- Llama-2 (4.02 GB) and DeepSeek (4.01 GB) need a file-backed body: Y1, or `--no-repack` as a declared numerics change.
- Limits:
  - The app's own memory is not inside the arm, so 3.4 GB held is an upper bound for Yi's repacked body.
  - The arms ran about 3 s of passes each, far shorter than a decode run.
  - anon42 missed by 0.92 GB/s. Its slow passes coincide with zram swap-free rising by 0.4 GB, as the system moved
    other memory into zram.

## 7. GPU lane (descriptive; no decision)

- `x2info`: Adreno 660, OpenCL 2.0, 3 compute units, `max_mhz=1` (driver report), 3,693 MB global memory.
  `cl_qcom_dot_product8` and `cl_khr_subgroups` are present.
- **`x2r` (`-cl-opt-disable` retry): build error, err = −6.** −6 is the OpenCL code for out of host memory. No
  retry number exists.
- GPU alone: read 31.14 GB/s (R69 30.86), q4 kernel 1.88 GB/s (R69 1.88).
- Together on 4-6:
  - read + CPU q4x4: CPU 20.70 + GPU 25.81 = 46.51 GB/s (cpu_ok 2/3; R69 47.40);
  - GPU q4 + CPU q4x4: 30.35 + 1.79 = 32.14.
- Against R69 Rule D: rank 1 (CPU + GPU split) stays open on reads, and is reproduced. The GPU's Q4_0 compute is
  still the blocker.

## 8. `ramp_ok=1` subset (descriptive)

- L1 (A78): every scored cell is on-core, `ramp_ok=1`, at 2,419 MHz. The L1, L1m, L1m4 and L1x values above are full-clock values.
- L4: no `7,4,5` arm is `ramp_ok=1`. On 4-6 and 4-7, the `ramp_ok=1` means are:

  | subject | 4-6 stock / K1 / K1m | 4-7 stock / K1 / K1m |
  |---|---|---|
  | Yi | 4.016 / 4.409 / 4.476 | 3.532 / 3.829 / 3.782 |
  | Llama-2 | 6.212 / 6.545 / 6.472 | 4.436 / 4.659 / 4.620 |
  | DeepSeek | 6.217 / 6.297 / 6.285 | 4.138 / 4.689 / 4.639 |

  In that subset, t_rule ÷ t_stock on 4-7 is 1.07 to 1.13 on all three subjects. Under the L4 rule's thresholds this
  pattern would read "kill" on every set. That is descriptive and not the verdict.
- L3, L2 on sets with cpu 7: `ramp_ok=0`. The ggml L2 cells are `ramp_ok=1`.
- L5, L6 and L7 scored cells: `ramp_ok=1`, except L6 on `7,4,5` (`ramp_ok=0`).

## 9. What the result means (§7 of the pre-registration)

- L1x > 0.90: K1m and k1m4 are closed; the next K1 build, if any, uses K1.
- L4 open on no subject: no K1 runtime patch is built from R72.
- K0 below 12 on all three: **12 tok/s is out of reach with the CPU levers measured so far.** Gaps +60.4 / +82.3 /
  +83.3 ms (Yi / Llama-2 / DeepSeek). The research goes to the GPU lane and to levers that cut compute and bytes.
- anon34 held but anon42 not: the repacked Q4_0 path is shown resident only for Yi. The next residency work for
  Llama-2 and DeepSeek is a file-backed body.
- New and not pre-registered:
  - the pinned X1 thread ran on small cores in most cells;
  - the A78 cap was lowered to 1,555 to 1,785 MHz in the L4 arms.

  Both bear on every core-set number from R69 and R72 and are carried into the next round.

## 10. Next round proposal (lead; to be pre-registered before any run)

1. **R73 (model-free, short): X1 placement and clock caps.**
   - Record per cell: cpu 7's online, isolated and core-control state, `scaling_max_freq` per policy, and the thermal
     zones.
   - Test runtime-side fixes, each off unless switched on:
     - re-pin on a `sched_getcpu` mismatch;
     - a short all-big-core warm-up before pinning;
     - an unpinned arm for comparison.
   - Re-run the L4 `7,4,5` and `4-7` stock cells with ggml.
   - Pass rule (draft): the cpu-7 thread on-core in ≥ 90 % of samples on `7,4,5`.
   - Why: every core-set result so far carries this confound, including the fourth-core adoption.
2. **GPU Q4_0 kernel probe.** A real Adreno Q4_0 gemv (sources: llama.cpp's OpenCL backend kernels for Adreno, to be
   checked by Ebony), run alone and beside the CPU at subject shapes.
   - It is the only measured route above r: the read sum is 46.5 GB/s *device*.
   - Kill line (draft): GPU Q4_0 under 10 GB/s alone, or a CPU + GPU sum on Q4_0 under 1.15 × the CPU alone.
3. **S3 freeze and a first real decode on one subject at stock** (served files on the laptop; then one phone run of Yi
   at 4-7). It calibrates the K0 arithmetic against a real token rate before more levers are priced.
4. **Owner:** the goal stays at 12 tok/s on all three. On present numbers, P(all three) is far below the 0.05 to 0.15
   prior (lead's judgement; Bonnie and Ebony's reading is below).
