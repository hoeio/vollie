# Pre-registration R76: x4s re-cell, gap sweep and heat pass (is design G worth building; c3 probe kit r37)

This file is committed and pushed **before** the APK is uploaded, and it is not edited afterwards. Any change goes
into `R76-amendment-1.md`. The tag `prereg-R76` is the owner's to create. The commit hash and the push time show the
order.

- **Date / author:** 2026-11-26, clyde (lead).
- **Why this round exists:** device rounds are held until a real breakthrough (owner, 2026-11-22). After R75 the lead
  reported that no priced design reaches 12 tok/s on all three subjects and that no R76 outcome can change that
  (`reports/memo-12-work/clyde-r76-plan.md` §2, §5). The owner chose option (a) (2026-11-26, "a로 가죠"; recorded in
  CAMPAIGN.md, commit `bb37b27`): run R76 to decide whether design G is worth building as a **below-goal** result.
- **Research method:** the owner returned to the pre-2026-11-23 method on 2026-11-26 (cost). This round used no new
  researcher instance; the bands are copied from work already committed.
- **Agent models** (full IDs):
  - lead (clyde): `claude-opus-5-5`, this file and `clyde-r76-plan.md`;
  - bonnie: `claude-fable-5-2`, `bonnie-joins-r35.md` (bands §4.2, outcomes §4.3), committed `1b006b4`;
  - ebony: `claude-fable-5-2`, `ebony-joins-r19.md` (T10, T11, §5), committed `1b006b4`;
  - tom: `claude-opus-5-5`, kit r37 (`tom-r37-build.md`);
  - annie: `claude-sonnet-5-5`, books (`bb37b27`).
- **Subjects:** Yi-1.5-6B-Chat, Llama-2-7b-chat, DeepSeek-LLM-7B-chat. They enter only as matrix shapes. No model is
  loaded. Triples are Yi / Llama-2 / DeepSeek.
- **Goal context:** 12 tok/s on each subject = 79.2 ms per token (CAMPAIGN.md Amendment 1). R76 is not a claim
  round, and its best outcome leaves the campaign verdict (12 not reachable by priced levers) unchanged.

## 0. Build under test

Kit r37, built by Tom (`reports/memo-12-work/tom-r37-build.md`), files in `lab/dist-c3/r37/`. sha256 values checked
by the lead with `shasum -a 256` before this commit:

| item | sha256 |
|---|---|
| `vollie-c3-probe-r37.apk` (16,646,106 B, versionCode 37, same certificate as r36) | `72541fcc4db0dde9d8d85446be8c33c92d175aec4fc9bf5deed5f29e2683f96c` |
| `BUILD-r37.txt` | `8e27a4bed234b0e7755922f4e2c27e600d1110472d2f50ab32faac1aa21a0210` |
| `SHA256SUMS` | `c91225f784bca125368421b464d4ae469601aa903eae921763712738cc67ad3b` |
| `patches/llama.cpp-c3/0009-r37-cells.patch` (on 0008) | `039cafd7b1563e10d1cc1e0535338049af30e9e4ed922ea4bf3c1e09b52d9d93` |
| `patches/app-c3/0007-kit-r37-probe.patch` (on 0006) | `b382e6b4f0de441f33725e609b05407eadfa513b1d2d72f7f5c30bae866929b7` |
| `tools/c3_probe_parse.py` (R76 section) | `2cce500135fca1fc49639821fc46df25cf8e24d72e131bc4ea1a74fe3fe364bb` |
| `libllama.so` in the APK | `4202674776dd6f6b8571a3a42e6a2a342c571b8484c8f043add6e3d746497c2d` |
| `libapp.so` in the APK | `5d06700a2f78f554a6767ba38eff8f586b398ffaf5f9247651ff3c30ae257b88` |
| `libggml-cpu-android_armv8.2_2.so` (byte-identical to r33 to r36) | `46e0735d2824ba6f3a156e219e058a701f6024fdbecbb2ef937b3127115a9694` |

Numerics: none. The CPU module is r36's file; 0009 changes only the probe file, off unless `VOLLIE_PROBE` is set.
Host checks (*host*, PoCL, functional only) are in `tom-r37-build.md`: all blocks ran with err 0, the heat-pass stop
worked under fake caps, parser self-tests pass and its output on the seven committed c3 JSONs is unchanged.

## 1. Device state

- The owner's Galaxy Z Flip 3, power saving **off**, charger **out**; both recorded.
- Deep-sleeping list label chosen in the app and recorded.
- **True cool start:** at least 30 min rest, screen off, not in use, and no other probe before this one.
- The r16 gate (cap 180 s) runs before each of the four blocks.
- No model is loaded. Kit r37 is installed over kit r36 (37 > 36, same certificate).

## 2. What is measured (run order x4i, x4sc, x4sg, x4sh)

- **x4i:** compiles the kernel binaries the other blocks load (about 1 s). Not scored.
- **x4sc (C, cool pass):** ten triplets cpu 0.5 s | split 1.0 s | cpu 0.5 s, 6 s rest between them: lead at 0.40,
  0.45, 0.50, 0.55; feeder at 0.25; stall triplet; lead at 0.55, 0.50, 0.45, 0.40. Lines `x4 sc trip`, `x4 sc sum`,
  `x4 sc best`, per-arm `x4 sc arm`, canary `x4 sc can`.
- **x4sg (G, gap sweep):** launch then flush from cpu 4 after a host-timed idle gap of 0, 0.1, 0.2, 0.3, 0.4, 0.5, 1,
  5 ms, round robin, 5 s measured. Lines `x4 sg gap`, `x4 sg arm` (`g1_step_us`, `g4_step_us`).
- **x4sh (H, heat pass):** 1 s split arms at 0.45 without rests, every 8th arm CPU alone, until 10 s after the second
  CPU cap step or 45 s; 1 Hz sampler (`x4 sh samp`); summary `x4 sh sum`.

Dropped from r36: x4f, x4c, x4d, x4e, x4m, x4s (with its hand-off cell), x4r.

## 3. Budget accounting

No numerics change; the +5 % budget is not touched. No model, 0 bytes of weights.

## 4. Predictions (*estimated*; Bonnie r35 §4.2 merged with Ebony r18 / r19, copied without change; P = probability inside)

Field keys are the parser's R76 section (`tools/c3_probe_parse.py`, sha above). "Lead arms" are mode `lead`.

### 4.1 Cool pass (x4sc)

| # | quantity | field | band | P |
|---|---|---|---|---|
| C1 | best lead ratio, shares 0.40 to 0.55 | `x4 sc best` `ratio_med` | **0.83 (0.76 to 0.88)** | 0.75 |
| C2 | share of the best ratio | `x4 sc best` share | 0.40 to 0.55 | – |
| C2b | the best share is 0.50 or 0.55 | same | yes | 0.45 |
| C4 | launch plus flush p50, lead arms at 0.40 and 0.45 | `lf_us` p50 | 190 to 235 µs | 0.7 |
| C5 | END to seen p50, lead arms with \|`slack_us`\| p50 ≤ 200 µs | `e2s_us` p50 | 80 to 350 µs | 0.75 |
| C6 | END to seen p50, lead arms with `slack_us` p50 ≥ 400 µs | `e2s_us` p50 | under 30 µs | 0.75 |
| C7 | \|ascending − descending\| at each share | `x4 sc sum` `d_ratio` | 0.04 or less at 3 of 4 shares | 0.6 |
| C8 | feeder triplet ratio at 0.25 | `x4 sc trip` `ratio`, feeder | 0.81 to 0.89 | 0.7 |
| C9 | stall triplet excess | `excess_us` | 80 (30 to 160) µs | 0.8 (Ebony r18's band) |
| C10 | spread of the cpu-5 canary over the pass (max ÷ min − 1, pinned readings only) | `x4 sc can` `c5_ns_per_w` | 2 % or less | 0.85 |
| C11 | no cap and `thermal_pwrlevel` 0 through the pass | arm tails, canary lines | yes | 0.8 |

C5 and C6 use `e2s_us` on `clk=mono` arms only; arms with `clk=queued` are reported beside, not scored. If no arm
falls in a C5 or C6 class, that row is "not measured".

CPU slowdown beside the GPU kernel (Bonnie's C3, 1.38, 1.28 to 1.54) has no single printed field; it is computed in
the result from `cpu_ms` and the cpu arms and reported, not scored.

### 4.2 Gap sweep (x4sg)

| # | statement | field | P |
|---|---|---|---|
| G1 | `lf_us` p50 is 60 µs or more higher at a 0.5 ms gap than at 0.2 ms | `g1_step_us` ≥ 60 | 0.7 |
| G2 | the step (largest rise between neighbouring gaps) lies between 0.25 and 0.40 ms, i.e. between the 0.2 / 0.3 or 0.3 / 0.4 rows | `x4 sg gap` `lf_us` | 0.55 |
| G3 | gaps 0 to 0.2 ms all in 110 to 160 µs and gaps 0.5 to 5 ms all in 195 to 250 µs | same | 0.55 |
| G4 | no further step between 1 and 5 ms | \|`g4_step_us`\| ≤ 30 | 0.7 |

The host-timed gap is the cell's label; `dgap_us` (device) is reported beside each row.

### 4.3 Heat pass (x4sh)

| # | quantity or statement | field | band | P |
|---|---|---|---|---|
| H1 | seconds to the first cap of either kind | `first_cap_s` | 17 (12 to 26) | 0.65 |
| H2 | the CPU's cap comes before the GPU's | `first_cpu_step_s` < `first_gpu_step_s` (or no GPU step) | yes | 0.7 |
| H3 | headroom at the first cap | `x4 sh samp` hr at `first_cap_s` | 0.76 to 0.92 | 0.75 |
| H4 | second CPU step within 12 s of the first | `second_cpu_step_s` − `first_cpu_step_s` | ≤ 12 | 0.7 |
| H5 | a slow stretch (split `ms_per_mat` ≥ 1.10 × mean of the split arms in the first 8 s) starts within 3 s of the first cap | `x4 sh arm` | yes | 0.6 |
| H6 | the cpu-alone arm inside the slow stretch is at most 8 % slower than the first cpu-alone arm | `x4 sh arm` mode cpu | yes | 0.55 |
| H7 | the canary falls 10 % or more in the slow stretch | `x4 sh can` `c5_ns_per_w` | yes | 0.35 |
| H8 | split `lf_us` p50 stays within 15 % from the first to the last split arm | `x4 sh arm` | yes | 0.6 |

H5 to H7 are "not measured" if no slow stretch appears. If no cap appears in x4sh, H1 reads "over 26 s or none"
(outside the band) and H2 to H7 are not measured.

## 5. Decision rules (scored as written)

The question is whether design G is built (host, then its own device round).

- **Build G:** C1 ≤ 0.86 and x4sc not void. G goes to Tom's host list (launch inside the pool job, trailing done
  flag, `GGML_KLEIDIAI_CHUNK_MULTIPLIER=16` on lane stages; Ebony r18 / r19 L1).
- **Kill line, lane as built:** C1 ≥ 0.90. The as-built lane rungs fall to about +0.5 tok/s (Bonnie r35 §4.3) and
  **G is not built**; GPU-lane work stops and the result says so.
- **0.86 < C1 < 0.90:** G is not built by default; the lead reports the price at the measured C1 and the owner
  decides.
- **L5 (keep the launch warm):** priced only if G1 holds (step ≥ 60 µs); it is an addition to G, not a separate
  build.
- **Heat margin:** if H1 < 12 s, any later lane claim on Llama-2 carries Bonnie's hot-end margin (about 0.4 tok/s);
  if H1 > 26 s or no cap, heat is not a price of the lane at claim length. Descriptive for G's decision.
- **Canary (T11):** `dcvsh_freq_limit` below the printed cap, or a spin more than 5 % off the printed step, in a
  slow arm → a hidden clock limit (Ebony C5) holds. A core in `isolated` → C7 holds. Neither → both closed.

### 5.1 Void (kit)

- APK hash differs from §0: the whole round is void.
- Power saving on, or charger plugged: every block run in that state is void.
- Process killed during a block: that block only is void.
- Any cap or `thermal_pwrlevel` above 0 during x4sc or x4sg: that block is void for C1 to C10 / G1 to G4 (C11 is then
  a miss, not a void).
- A triplet whose two cpu arms differ by more than 3 % (`cpu_diff` > 0.03): that triplet is void; C1 uses the
  remaining triplets of that share, and a share with no valid triplet is not measured.
- The gate before x4sh timed out: x4sh is void.

**Not voids:** cool start not met (reported), clocks below max, refused canary pins (`-1`), `clk=queued` arms.
Misses are written as misses.

## 6. Claim rule

Not a goal claim. No model is loaded; no number from R76 is a phone token rate. Prices from it are *estimated* from
*device* inputs.

## 7. What a result means

- **C1 ≤ 0.86:** the lane's cool split stands; G is worth about +0.8 to +1.0 tok/s over the lane as probed (cool,
  *estimated*), with rung-7 forms about 10.6 / 10.5 / 10.1. Below the goal; reported as such.
- **C1 ≥ 0.90:** the CPU term is wrong at high share even cool; the lane is worth about +0.5 tok/s; GPU-lane work
  stops.
- **Whatever the outcome:** 12 tok/s on all three stays out of reach by priced levers; the campaign verdict is
  restated with R76's numbers.

## 8. The run, for the owner

1. After this file is pushed, create tag `prereg-R76` on its commit. Give the lead an HF write token for the `r37/`
   upload, or upload `lab/dist-c3/r37/` yourself (APK, `BUILD-r37.txt`, `SHA256SUMS`) to `HoeioUser/vollie-kits` as
   `r37/`.
2. Download `r37/vollie-c3-probe-r37.apk` and check its sha256 is
   `72541fcc4db0dde9d8d85446be8c33c92d175aec4fc9bf5deed5f29e2683f96c`. Install it over kit r36 (no uninstall).
3. Phone state: charger out, power saving off, deep-sleeping label chosen, at least 30 min rest with the screen off,
   no other probe before this one.
4. Research kit page: **Run c3 probe r37 (R76: cool pass, gap sweep, heat pass)**. Do not use the phone until it
   finishes (about 2 to 3.5 min of blocks plus up to four gates of 3 min; about 2 to 15 min in total).
5. **Save c3 probe to Downloads.** Send the JSON; it goes into `campaigns/03-dense15-5pct/reports/device/`.
6. If the app is killed, the earlier blocks are already saved. Send the file as it is.
