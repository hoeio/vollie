# R74 result: x3 GPU lane probe (upstream Adreno Q4_0 gemv, c3 probe kit r34)

- **Date:** 2026-11-23, 17:43:06 to 17:53:01 local. **Label:** *device* for every value the phone printed. P_G,
  stack prices and gaps are arithmetic on *device* inputs and are *estimated*.
- **Sources:**
  - [device/r74-c3probe-r34-2026-11-23T1743.json](device/r74-c3probe-r34-2026-11-23T1743.json), sha256
    `384bc4b508a8098590d9ec923fbe78142c49492c59a65ab3d7e8df9dcba75c57`.
  - Sent with it (descriptive, not part of R74):
    [device/r74-capprobe-r34-2026-11-23T1736.json](device/r74-capprobe-r34-2026-11-23T1736.json), sha256
    `4d0d699a71e5a70d085005a13b43f79e5942e46bb4863f09ace079bead9d3743`, campaign 2's T-A cap probe (memo-31), run
    17:36:48 to 17:42:48.
  - Pre-registration [R74](../prereg/R74-x3-gpu-probe-device.md) (`3c1cc55`, tag `prereg-R74`).
  - Every rule input re-read from the raw lines with the lead's own short code (`python3 -I`, not committed); the
    stack prices with `memo-10-work/bonnie_tree.py` as committed.
- **Scored by:** clyde (lead), `claude-opus-5-5`.

## 1. Void checks: none met

| void | state | source |
|---|---|---|
| APK hash differs | not triggered (lead's check of the build; owner's download check assumed) | JSON `kit=r34`, `app=1.34.0+34`. `lab/dist-c3/r34/` hashes to the prereg values: APK `2b0a2929…81de`, BUILD `caf3ab66…2c5f`, SUMS `d4897c69…35f8` (lead, `shasum -a 256`, 2026-11-23). *Owner: please confirm the sha256 you checked.* |
| power saving on | not triggered | `powerSave: false` at start, end and before all 4 blocks |
| charger plugged | not triggered | `plugged_start` false; `plugged_before` false in all 4 blocks |
| process killed | not triggered | all 4 blocks have their end line; `x3 res done err=0`; `stopped_early` false |

## 2. Conditions beside every number

- **Cool start not met.** The prereg asks for 30 min of rest. The T-A cap probe ran until 17:42:48, with three
  threads spinning on cpus 4 to 7 from 17:39:48. R74 started 18 s later. This is a condition, not a void (§5 voids).
  - The x3b gate timed out at 180 s (headroom 0.706). The x3r gate also timed out (0.66). The x3g and x3c gates
    passed.
  - Direction: heat lowers clocks and rates, so the rates below are, if anything, low.
- **CPU clocks:** every timed cell reads `ramp_ok=1`.
- **GPU clock:** unreadable (`gpu_mhz=e13`, EACCES). The `x3 cond` line read `devfreq_cur=315 MHz`, max 840,
  `thermal_pwrlevel=0`, before the blocks.
- **Load log:** every `x3 loadlog` reads `Error: Program not built!`, while `x3 load` reads `ok=1`. The identity
  lines pass, so the kernels ran correctly. The text is read as a log query on a binary-loaded program, not as a
  build failure. Tom checks this.
- Deep-sleeping list **on**. `model_unloaded=olmoe-Mi.gguf`. Battery 69 to 66 %, 26.1 to 27.7 °C.

## 3. Decision rules (as written, in order)

| rule | input (*device*) | verdict |
|---|---|---|
| 1 build | all four kernels: `x3 build rc=0 how=opt` on try 1, `x3 load ok=1` | **pass** |
| 2 identity | six lines, all `ok=1`; max_rel_err 6.2e-4 to 7.6e-4, rows_bad 0, nonfinite 0; the 11008×4096 pick (`s11008_1_4096`) passes | **pass** |
| 3 GPU alone cold | g_cold Yi **17.53** GB/s; Llama-2 / DeepSeek **18.33** GB/s (cold gbs: 4096×4096 15.06, 512×4096 3.97, 11008×4096 21.01, 4096×11008 19.66) | **pass on all three** (≥ 7) |
| 4 CPU + GPU | `x3 togsum sum_vs_cpu_alone` **1.611** (cpu alone 25.64, both 19.77 + 21.53 = 41.30 GB/s) | **pass** (≥ 1.15) |
| 5 | rules 1 to 4 pass on every subject | **GPU lane OPEN on all three** |

As §5 rule 5 says, "open" means only two things: a host numerics check, and a lane design. It is not a device go.

## 4. Prices (*estimated* from *device* inputs, §5 as written)

Inputs:
- lane = 1 ÷ 1.611 = 0.621.
- g_eff = 0.611 × 26 ÷ 0.64 = 24.8 GB/s.
- `x3 rt` `ovh_us` at 2816 rows = **452.0 µs**. The prereg centre was 150; this is 3.0 times higher.

| subject | dispatch (2 joins/layer) | P_G (K0 + lane) | gap to 79.2 ms | P_G at 0 dispatch | centre stack (W 2, `bonnie_tree.best`) | stack at 0 dispatch |
|---|---|---|---|---|---|---|
| Yi | 28.9 ms | **7.86** tok/s | +41.7 ms | 10.34 | **9.55** | 12.23 |
| Llama-2 | 28.9 ms | **7.39** | +49.4 ms | 9.53 | **9.50** | 11.87 |
| DeepSeek | 27.1 ms | **7.30** | +51.0 ms | 9.22 | **9.26** | 11.44 |

Descriptive, beside the prices:
- At one join per matrix (7 per layer), dispatch costs 95 to 101 ms per token, and P_G falls to 4.7 to 4.9.
- The `x3 layersum` overhead, n_layer × (ms_sync7 − dev_ms), is 163 / 158 / 107 ms.
- GPU alone, the whole body at sync1 takes 183 / 191 / 181 ms, against a device time of 117 / 134 / 126 ms. The CPU
  body is 121 / 141 / 131 ms.

Reading:
- **The kernel and the bus deliver.** g is about 2.2 times the prereg centre, and the CPU+GPU sum is 41.3 GB/s.
  That is 0.89 of R72's 46.5 read ceiling.
- **Dispatch is the binding cost.** At the measured 452 µs per round trip, the lane lifts K0 by about 1.2 tok/s per
  subject: 6.81 → 7.86, 5.88 → 7.39, 5.84 → 7.30.
- At zero dispatch, the centre stack would reach 12.23 on Yi and fall short on Llama-2 (11.87) and DeepSeek (11.44).
  It still needs every unbuilt lever of the stack at its centre. §5's line held as written: no realistic dispatch
  reaches 12 on all three.
- **The campaign verdict is unchanged:** 12 tok/s on all three is out of reach by priced levers. The gap narrows by
  19 to 33 ms per subject (K0 gaps +60.4 / +82.3 / +83.3 → +41.7 / +49.4 / +51.0 ms).

What now carries the lane is the join cost, not g. The prereg's line (≤ 20 µs per join at 64 joins) is 23 times
below the measured 452 µs. A lane design must cut joins or overlap them, for example:
- an asynchronous queue with one host wait per token;
- event polling instead of `clFinish`;
- the CPU never waiting on the GPU inside a layer.

Its price is a host-design question first.

## 5. Residency (x3r, descriptive with the prereg's reading)

| criterion | value | line | result |
|---|---|---|---|
| median CPU read, passes 5 to 99 | 18.72 GB/s | 0.80 × (29.90 − 0.36 × 21.05) = 17.86 | met |
| median majflt per pass | 0 | ≤ 50 | met |
| `x3 res done` | `err=0`, 100 passes, 15.2 s | present | met |

**Held.** 0.85 GB of GPU buffers beside a 2.6 GB anonymous arena (3.5 GB) stayed resident.
- That covers Yi's split body (3.36 GB).
- Llama-2 and DeepSeek (4.0 GB) are not covered by this arm.
- Limits as written: the app's own memory is outside the arm, and the arm ran 15 s, not a decode.
- Passes 91 to 99 slowed (CPU 5.7 to 15.0 GB/s, with majflt 0 and memavail falling to 0.59 GB). This is read as
  memory pressure starting near the end, and it is recorded.

## 6. Bands (§4.2): 10 of 28 cells inside

| row | measured (*device*) | band | |
|---|---|---|---|
| B1 compile s | 0.29 (general), 0.14 (spec) | 1 to 30 | miss low |
| B2 wg_max | 512 | ≥ 256 | inside |
| B3 max_rel_err 11008×4096 pick | 6.88e-4 | 1e-4 to 5e-3 | inside |
| G1 g, 11008×4096 cold | 21.01 | 4 to 15 | miss high |
| G2 hot ÷ cold | 1.040 | 0.95 to 1.40 | inside |
| G3 4096×4096 / 4096×11008 cold | 15.06 / 19.66 | 3.5 to 14 | miss high / miss high |
| G4 512×4096 cold | 3.97 | 1.5 to 9 | inside |
| G5 general ÷ pick | 0.997 | 0.70 to 1.05 | inside |
| G6 dev_gbs ÷ gbs | 1.348 | 1.00 to 1.30 | miss high |
| G7 g ÷ 1.88 | 11.18 | 2.1 to 8.0 | miss high |
| D1 ovh_us 2816 rows | 452.0 | 50 to 400 | miss high |
| D2 dev_us 2816 rows | 239.4 | 430 to 1,600 | miss low |
| D3 rt_us_per_dispatch Yi / Llama-2 / DeepSeek | 433.8 / 453.3 / 248.9 | 50 to 400 | miss / miss / inside |
| L1 body_ms_sync1 | 182.7 / 190.8 / 181.0 | 210–780 / 245–910 / 230–855 | miss low ×3 |
| C1 cpu_alone_gbs | 25.64 | 21 to 29.5 | inside |
| C2 both_cpu / both_gpu | 19.77 / 21.53 | 18–28 / 3.5–14 | inside / miss high |
| C3 sum_vs_cpu_alone | 1.611 | 1.04 to 1.35 | miss high |
| S1 best split ratio | 0.912 at share 0.25 | 0.82 to 1.20, at 0.25 | inside |
| R1 x3r cpu_gbs_median | 18.74 | 19 to 28.5 | miss low |
| R2 majflt median per pass | 0 | 0 to 5 | inside |

§4.1 events: all five happened.
- general kernel loads (P 0.60);
- at least one kernel loads (0.70);
- all four load (0.45);
- identity all ok given built (0.85);
- lane open (**0.20**).

The prediction underrated the GPU kernel twice over: in compile risk and in rate. It underrated dispatch the other
way: the overhead was 3 times the centre.

## 7. Next (host only; R73 stays held)

1. **Tom, host numerics check (§5 rule 5).** A C emulation of the kernel's half-precision `(q − 8) × d` inside the
   CPU backend, behind a knob. Per-chunk PPL at `-b 512` and `-ub 1` on two subjects, against the +5 % budget.
   Also: the `loadlog` text.
2. **Lane design with the join cost as the target.**
   - Measure a host-side model of joins per token. Price the async-queue and event-poll variants with Bonnie and
     Ebony.
   - The prereg line to beat is about 20 µs per join at 64 joins, or about 1.3 ms per token.
   - A device test of a lane needs its own pre-registration and the owner's permission.
3. Annie: books (results.csv R74 row, STATUS, memo index).

*Note 2026-11-23 (lead): the owner confirmed that the downloaded r34 APK's sha256 matched `2b0a2929…81de`. The APK void
is not triggered on the owner's check either.*

*Note 2026-11-23 (Annie, number audit; appended, the text above is not changed).* Every number in §1 to §6 was
re-derived from the raw JSON (`384bc4b5…`) and the committed scripts (`python3 -I`); all reproduce. One comparison
needs its basis stated. The K0 figures in §4 "Reading" and §7 (6.81 / 5.88 / 5.84 tok/s, gaps +60.4 / +82.3 / +83.3 ms)
come from R72-result §5, which uses the K1 body (122.53 / 149.06 / 140.66 ms). P_G and the prices above use the ggml
stock body (120.6 / 140.6 / 130.8 ms), as the prereg §5 inputs say. On the same ggml base K0 is 6.90 / 6.21 / 6.22 tok/s
(gaps +58.5 / +73.8 / +73.5 ms, *estimated*), so the lane lifts K0 by +0.96 / +1.18 / +1.08 tok/s and narrows the gap
by 16.8 / 24.4 / 22.5 ms. Against R72's K0 the lifts are +1.05 / +1.51 / +1.46 tok/s and the narrowing 19 to 33 ms, as
written above. The sentence "about 1.2 tok/s per subject" fits neither exactly (about 1.0 to 1.5). P_G, the stack
prices, the gaps to 79.2 ms (79.17) and the verdict are unaffected. The lead decides whether to reword.
