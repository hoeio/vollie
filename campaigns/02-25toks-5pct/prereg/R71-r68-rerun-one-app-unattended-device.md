# Pre-registration R71: R68 re-run in one app, unattended (kit r30: c2, c1 and stock runtimes in one APK)

Committed and pushed **before** the kit is uploaded and before any r30 run on the phone. Do not edit afterwards;
changes go into `R71-amendment-1.md`. The owner creates `prereg-R71` on this commit if wanted.

- **Date / author:** 2026-11-20, clyde (lead, main session).
- **Why:** the owner asked for one app that runs R68 again, installs as an update over the c1 APK that is on the
  phone now, and runs unattended (owner, 2026-11-20). R68's session R was void (power saving on) and its stock APK
  would not install (amendment 5). In R71, all three runtimes run in one sitting under one phone state.
- **Owner decisions this round rests on:** power saving off (2026-11-18); the deep-sleeping list may be on
  (2026-11-19, R70); one unattended app (2026-11-20).
- **Agent models** (full IDs): lead `claude-opus-5-5` (this file); tom `claude-opus-5-5` (app patch 0013, kit r30,
  parser and scorer reading of r30); bonnie `claude-fable-5-2` and ebony `claude-fable-5-2` (R68 scorer and audit,
  reused unchanged); annie `claude-sonnet-5-5` (books).
- **Models and files:** `granite-Mi.gguf` `8df94de7…0bc0f45`, `olmoe-Mi.gguf` `5abf710b…8e444abb`, internal copies
  (kept from the r28 install), sha256 checked by the kit.

## 0. Build under test

| field | value |
|---|---|
| APK | `vollie-c2-1.30.0+30-research-r30.apk` (HF `r30/vollie-r30.apk`), 20,795,890 bytes |
| sha256 | `5459fc67d8e0073883e28d87ed02769f4f1f8eac7d8783ae0a7092e48de3221b` |
| package | `com.insaneutils.vollie.c2`, versionCode 30 (installs as an update over r28-c1, 28), cert `c49dde85…` |
| app patches | `patches/app-c2/0001` to `0013`; `0013-kit-r30.patch` `fd0d6b781980457d99cd35dcb1b02c5cfb9a10c1ed7c823791216e04394cec0e` |
| build record | `lab/dist-c2/BUILD-r30.txt` (sha256 `c3dbeb1a…5469a`) |

Runtime libraries. All are **byte-identical** to the r28 APKs of R68 §0, unpacked, not rebuilt (lead re-checked,
*host*):

| runtime | libllama (file in APK) | CPU module (file in APK) |
|---|---|---|
| c2 | `libllama.so` `126bfb4f…` | `libggml-cpu-android_armv8.2_2.so` `6f1d859a…` |
| c1 | `libllama_rt_c1.so` `f2f4f88a…` | `libggml_cpu_rt_c1.so` `53bfee45…` |
| stock | `libllama_rt_stock.so` `de2ccec7…` | `libggml_cpu_rt_stock.so` `604e0f69…` |

The other native libraries are byte-identical to r28's. `libapp.so` (the kit) is new.

How a runtime is chosen (0013): before Flutter starts, the app loads the chosen libllama (SONAME `libllama.so`)
and sets `VOLLIE_CPU_MODULE`. With that set, llamadart loads only that CPU module and skips load-all. c2 sets
nothing, which is r28's path. A change of runtime is a process restart.

## 1. What is run

The phone rests 30 min or more with the screen off, not charging and not in use (R70's rest rule). Then the owner
taps **Auto** once. The kit runs, unattended and in this order:

| segment | runtime | plan |
|---|---|---|
| S1 | c2 | R68 §1 c2 plan: b1 to b5, two warm-ups each, unchanged (b5 = OLMoE num + touch + nice, the R70 configuration) |
| cool | – | 30 min idle on the static dark screen |
| S2 | c1 | R68 §1 reference plan: Granite then OLMoE, two warm-ups + 10 claim-format runs each, R68's c1 env |
| S3 | stock | the same, with no env |

Within each segment, the r16 gate and hold run as in r28. The cap probes are not run (power saving cannot be set
from the app). R68 session B (list off) is not part of R71.

- **Phone state:** power saving off, the deep-sleeping list **on** in all three segments, charger out, battery
  60 % or more at start. The start label is "list on". This differs from R68 §1, which had session R with the list
  off. That is a stated difference, so R71's S2 and S3 are not compared with a list-off state.
- **Chain safety:** these are kit rules, not lever rules. The chain stops, and records why, if:
  - the battery is below 15 %;
  - the mapped libllama or CPU module is not the active runtime's (§0);
  - power saving still reads on after 3 re-reads 10 min apart.

  A charger pauses the kit at the gate (R63 rule), and plugged runs are not scored. A cut-off segment runs again
  from its start, at most twice; its partial report is kept and reported, not scored.

## 2. Budget accounting

As in R68 §2. No numerics change is new:

- Granite num 9.5056 / 9.5120 and OLMoE num 12.0927 / 12.0976 (*host*, R61), both inside their budgets.
- c1 and stock have no budget line claimed.
- The runtime selection does not touch numerics. It is a library choice, checked per load by hash.

## 3. Predictions (*estimated*, written after R68's results)

**Mechanisms.**

| prediction | probability |
|---|---|
| The chain finishes all three segments with no stop | 0.75 (main risks: battery, power-saving re-enable, a restart failing on the phone; none was seen on the phone yet) |
| Every load's runtime check passes | 0.95 |
| Power saving reads off in every run | 0.95 |
| b1 uncapped (cpu 7 max ≥ 2,419) in ≥ 9 of 10 runs | 0.75 |
| nice void again (threads already at −10, R68 amendment 4) | 0.9 |
| touch passes on both models (R68 §4 rule; passed in R68 A and B) | 0.7 per model |

**Levels** (medians of 10 claim-format runs, *device* when measured):

| block | prediction | R68 reference (*device*) |
|---|---|---|
| S1 b1 OLMoE off | 24.8 (23.0 to 25.8) | A 25.20, B 23.03 |
| S1 b2 Granite num | 23.0 (22.0 to 24.0) | A 23.27, B 22.59 |
| S1 b5 OLMoE num + touch | 26.5 (25.0 to 27.8) | A 27.25, B 25.47 |
| S1 b5 runs below 25.0 (of 10) | 0 to 3 | A 0, B 3 |
| S2 c1 Granite (R49 env) | 21.0 (19.5 to 23.0) | 20.08 (void: power saving on); R49 20.51 |
| S2 c1 OLMoE (`c1_ship`) | 23.0 (21.0 to 25.0) | 22.66 (void: power saving on) |
| S3 stock Granite | 13 (8 to 18) | none (R68 bands kept) |
| S3 stock OLMoE | 12 (6 to 18) | none |

Order: c2 b5 > c1 OLMoE > stock OLMoE, and c2 b2 > c1 Granite > stock Granite, on both models: probability 0.8.
S2 and S3 run after S1 and the cool-down, so heat can favour S1. That is a known confound of the fixed order and
is reported, not corrected.

## 4. Pass, kill and void

- touch, nice, phone identity (greedy text) and kit void rules: R68 §4 and amendments 1 to 4, as written, scored
  with `score_r68.py` (r30 reading, no rule changed) on S1.
- **Runtime identity:** every row's `runtime_check` passes; a failed row voids that segment.
- **Go-rule reading (rehearsal, not a claim):** S1 b5 median ≥ 25.7 and at most 2 of 10 below 25.0, with all 10
  uncapped. R71 does not replace R70: a goal claim needs R70 (or its amendment) as written.
- **Reference segments:** descriptive per model. A segment with fewer than 8 valid runs per model is void.
- Misses are written as misses.

## 5. What a result means

- S1 as R68 A: the list-on rehearsal state repeats, and R70's band is supported. S1 well below it: the phone state
  of the day decides, as in R60 and R62.
- S2 and S3 valid: they replace R49 and R52 as the comparison for later claims under power saving off and list on,
  per model, with what generalised.
- Chain finished: the unattended kit can carry R70 and campaign 3's probe kit (owner's choice).

## 6. Hygiene

As R68 §6. In addition:

- `r30-chain.json` and `r30-chain-log.txt`: segments, restarts, mapped libraries with sha256, and battery and
  charger per run.
- One report per segment.
- The owner sends every `vollie-r30-*` file saved to Downloads.
