# Pre-registration R62: skip + head with hold in every run, and threads 3 against 4, on Granite and OLMoE (kit r23)

Committed and pushed **before** the APK is handed to the owner. Do not edit afterwards; changes go into
`R62-amendment-1.md`. No tag (owner's wish); the commit hash and push time show the order.

- **Date / author:** 2026-11-16, clyde (lead).
- **Campaign / models / files:** campaign 2. `granite-Mi.gguf` `8df94de7…0bc0f45`, `olmoe-Mi.gguf`
  `5abf710b…8e444abb`, already on the phone, hash checked by the kit.
- **Agent models** (full IDs): clyde `claude-fable-5-2`; tom `claude-opus-5-5` (patch 0006, kit r23, memo-22);
  bonnie `claude-opus-5-5` (memo-21, the source of the threads test); annie `claude-sonnet-5-5` (not used).
- **Build under test:** `vollie-c2-1.23.0+23-research-r23.apk`, 17,212,039 bytes, sha256
  `2cf9d2ad193b77458f003d6ec9c9de5b6d0d6464c4597a2dd5e32512bcefd90c`. Runtime: llama.cpp `620d2225c` + campaign
  1's `patches/llama.cpp/0001` to `0003` + `patches/llama.cpp-c1-olmoe/0001` to `0006`, built with KleidiAI on
  as every kit. CPU module `19f36b6e…00cbd8f1`, `libllama.so` `edcfadf1…1768a534`. App patches
  `patches/app-c2/0001` to `0007`. Kit description: [memo-22](../reports/memo-22-kit-r23.md).
- **Device:** the owner's Galaxy Z Flip 3. **Labels:** every number in §3 is *estimated* (from R60's *device*
  runs and R61's *host* values) until a result says *device*.
- **Starts from campaign 1:** its runtime, kit r16's gate, prompt, 32 greedy tokens and metric. R60 is the
  comparison round; its hold quads are the prior.
- **Phone state asked of the owner:** background apps closed (R60 ran with a banking app in the background),
  not charging, phone left idle a few minutes before the start. The owner says afterwards what was the case.

## 1. What is tested

Blocks Granite, OLMoE, OLMoE, Granite; two ABBA quads (off num num off) per block; 16 scored runs per model.

- **num** (declared numerics change), as R60: OLMoE `VOLLIE_SKIP_MASS=900 VOLLIE_HEAD2=1024`; Granite
  `GGML_EXPERT_SKIP=130 GGML_HEAD2=2048 GGML_HEAD2_Q=q4_0`.
- **threads** 3 against 4: one quad each per block, order alternated across blocks. Both counts run inside the
  same affinity set (cores 4 to 7), so this tests a fourth worker on the same four big cores, with the
  runtime's barriers (666 per token Granite, 305 OLMoE). It is not memo-21's barrier-free helper.
- **Changed against R60, in both arms:** (a) OLMoE is loaded with `VOLLIE_HEAD2_REPACK=1`, which keeps its Q6_K
  head out of KleidiAI: **a declared change of OLMoE's off arm** (R61, *host*: 11.7132 / 11.7159 against
  11.7165 / 11.7198 with the head in KleidiAI). (b) The page hold runs before every gated run, in a cheaper
  form (memo-22: at most one pass in the wait, up to four after the gate, 8 s limit).

## 2. Budget accounting (R61, *host*, KleidiAI build; the budgets are unchanged)

| model | stock (`-b 512` / `-ub 1`) | num (`-b 512` / `-ub 1`) | budget | ceiling |
|---|---|---|---|---|
| OLMoE-Mi | 11.7132 / 11.7159 | 12.0927 (+3.24 %) / 12.0976 (+3.26 %) | 12.3003 | 13.1304 |
| Granite-Mi | 9.2795 / 9.2732 | 9.5056 (+2.44 %) / 9.5120 (+2.58 %) | 9.6879 | 10.5379 |

Limit: the host's KleidiAI uses i8mm kernels, the phone's dotprod kernels. Threads, hold and the kit's lines
change no arithmetic; one reply text per arm and model is expected across both thread counts.

## 3. Predictions

State: the level bands hold if the clock is 2,112 MHz in the profiled runs and `Cached` is at least 2.4 GB
before at least 14 of a model's 16 scored runs. Otherwise the level bands are not scored for that model and the
effects still are.

| quantity | Granite | OLMoE |
|---|---|---|
| off arm, 3 threads, median of 4 (tok/s) | 16.6 (15.0 to 18.5) | 17.8 (16.0 to 21.0) |
| num arm, 3 threads, median of 4 | 19.0 (17.5 to 20.5) | 20.3 (18.0 to 23.0) |
| num minus off, mean of the 4 quads | **+2.4 tok/s (+1.4 to +3.4)**, +14 % (+8 to +21) | **+2.8 tok/s (+1.2 to +4.5)**, +15 % (+7 to +25) |
| expert bytes skipped in decode | 39.6 % (38 to 44) | 13.0 % (11.5 to 14.5) |
| head op in the num profiled runs, median | 3.9 ms (3.5 to 4.3), as R60 | **2.6 ms (2.1 to 3.1)**, from 3.9 |
| OLMoE head line | | `c2_calls=33`, `c2_declined=0`, active copy, in every num run |
| threads 4 minus 3, both arms together | −3 % (−12 to +6) | +4 % (−6 to +14) |

Hold (all 32 scored runs): resident MB before the run at least 95 % of the wanted MB in at least 12 of 16 runs
per model; CPU per wait, median, 0.9 s (0.6 to 1.3); never more than 8.5 s between gate and run.
Generalisation prediction: num helps both models by a similar per cent; a fourth thread helps neither
resolvably; if it helps one, it is OLMoE (fewer barriers, more expert work).

Arithmetic: R60's hold quads gave Granite +2.29 and +2.47 tok/s, OLMoE (skip alone) +3.67 and +1.21; the
repaired head is priced at −1.3 to −1.5 ms of about 50 (memo-21). Levels are R60's hold medians (16.61, 18.96,
17.48, 19.25), OLMoE raised for the head.

## 4. Pass, kill and void

- **num, per model.** Pass: the mean of the four quad effects is positive and larger than 2 × SE, and the decode
  skip counter is inside its band; for OLMoE also the head line as predicted. Band hit or miss is scored
  separately. Unresolved or negative: "not shown on this model".
- **threads, per model.** Statistic: mean of the 8 scored runs at 4 threads minus mean of the 8 at 3, Welch SE.
  Pass: positive and larger than 2 × SE. **Kill:** negative and larger than 2 × SE: plain four-thread decode is
  closed on that model. Otherwise "no resolved difference"; three threads stay. Neither outcome closes memo-21's
  barrier-free helper, which is a different mechanism.
- **hold (protocol check, not a factor).** Fails if on OLMoE fewer than 8 of 16 scored runs start with 90 % of
  the wanted MB resident; then r22's per-poll passes return at a lower rate in the next kit. The CPU and time
  predictions are scored hit or miss.
- **Void (kit, not lever):** a model or library hash differs; fewer than 6 scored runs per arm and model; texts
  differ within an arm; a num run's skip counter is zero; Granite's head line shows no active copy; **OLMoE's
  num runs show `c2_declined` above zero or no active copy, or the load setting is missing** (voids num on
  OLMoE only); the thread count applied differs from the spec (voids threads).
- Misses are written as misses.

## 5. Claim rule

Not a goal claim. A 25 tok/s claim needs campaign 1's r16 run list and claim positions with a frozen
configuration, pre-registered on its own.

## 6. Hygiene

Per scored run: clock, gate result, `Cached`, resident MB before and after, wanted MB, major faults, threads
applied, thread placement samples. No run is excluded from the means; runs after a gate timeout are listed.

## 7. What a result means

- num passes on both and OLMoE's head runs: freeze the two settings; the next round is about the remaining ms
  (memo-21: helper threads without barriers, the routing prior, the LUT kernel), not about these levers.
- OLMoE's head still declines: read `c2_why`; the fix is in the runtime, no rerun of this kit for it.
- threads killed or unresolved: memo-21 item 1 goes on only as the barrier-free helper, with a microbench first.
- gates time out again with background apps closed: the gate threshold itself is examined against campaign
  1's r16 data before the next kit.
