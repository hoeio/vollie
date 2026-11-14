# Pre-registration R60: skip + two-stage head, and page hold through the gate wait, on Granite and OLMoE (kit r22)

Committed and pushed **before** the APK is handed to the owner. Do not edit afterwards; changes go into
`R60-amendment-1.md`. No tag (owner's wish); the commit hash and push time show the order.

- **Date / author:** 2026-11-15, clyde (lead).
- **Campaign / models / files:** campaign 2. `granite-Mi.gguf` `8df94de7…0bc0f45`, `olmoe-Mi.gguf`
  `5abf710b…8e444abb` (full hashes in the model cards), already on the phone from kit r21, hash checked by the kit.
- **Agent models** (full IDs): clyde `claude-fable-5-2`; tom `claude-opus-5-5` (patches, kit, memo-20);
  bonnie `claude-opus-5-5` (not used this round; the bands are the lead's, from R58's device data);
  annie `claude-sonnet-5-5` (not used this round).
- **Build under test:** `vollie-c2-1.22.0+22-research-r22.apk`, 17,208,611 bytes, sha256
  `e2e59adab40096983130573928a5f069b67eda985a6acb18bee36fb828f86914`. Runtime: llama.cpp `620d2225c` + campaign
  1's `patches/llama.cpp/0001` to `0003` + `patches/llama.cpp-c1-olmoe/0001` to `0005`. CPU module
  `5d7c550c…0e43de46d7`, `libllama.so` `815f3453…e70bf1c02`. App patches `patches/app-c2/0001` to `0006`.
  Kit description: [memo-20](../reports/memo-20-kit-r22.md).
- **Device:** the owner's Galaxy Z Flip 3. **Labels:** every number in §3 is *estimated* (from R58's *device*
  token times and R59's *host* skip fractions) until a result says *device*.
- **Starts from campaign 1:** its runtime, kit r16's gate, prompt, 32 greedy tokens and metric; R58's off arm is
  the baseline.

## 1. What is tested

Two factors, each on both models in one session (blocks Granite, OLMoE, OLMoE, Granite; 8 scored runs per block,
two ABBA quads off num num off; 16 scored runs per model, 8 per arm).

- **num** (a numerics change, declared): OLMoE `VOLLIE_SKIP_MASS=900 VOLLIE_HEAD2=1024`; Granite
  `GGML_EXPERT_SKIP=130 GGML_HEAD2=2048 GGML_HEAD2_Q=q4_0`. **off** is R58's knob-off arm.
- **hold** (no numerics; app side): at each gate poll the app asks the kernel (`posix_fadvise(WILLNEED)`) to
  read back the model pages that were resident after the previous run and are gone. On for one quad per block,
  order alternated, so each model has it first once and last once.

## 2. Budget accounting (R59, *host*)

| model | stock | config (`-b 512` / `-ub 1`) | budget | ceiling |
|---|---|---|---|---|
| OLMoE-Mi | 11.7146 | 12.1053 (+3.34 %) / 12.0946 (+3.26 %) | 12.3003 | 13.1304 |
| Granite-Mi | 9.2266 | 9.5173 (+3.15 %) / 9.5076 (+3.04 %) | 9.6879 | 10.5379 |

Hold and the kit's counting lines change no arithmetic (identity on the host, memo-20). Texts: one reply text
per arm and model; off and num texts differ by design.

## 3. Predictions

State: the bands hold for a cool phone (2,112 MHz at the end of the runs) with `Cached` at least 2.4 GB before
the runs, as in R58. If the session is throttled or `Cached` is lower, the level bands are not scored and the
effects are still scored.

| quantity | Granite | OLMoE |
|---|---|---|
| off arm, median of 8 (tok/s) | 17.2 (15.5 to 18.5) | 19.7 (16.0 to 21.5) |
| num minus off, mean of the 4 quads | **+22 % (+12 to +32 %)**, about +3.8 tok/s | **+15 % (+8 to +22 %)**, about +2.9 tok/s |
| num arm, median of 8 | 21.0 (18.5 to 23.0) | 22.7 (19.0 to 25.0) |
| num arm, fault-free token in the profiled runs | 44 ms (41 to 48) | 36.5 ms (35 to 39) |
| expert bytes skipped in decode | 41 % (38 to 44) | 13 % (11.5 to 14.5) |

Arithmetic: OLMoE 42 ms per token in R58 (expert op 27.8 ms, head 3.5 ms): 13 % of the expert op is 3.6 ms,
the head screen about 2 ms. Granite 55 ms in R58 without campaign 1's levers; campaign 1's skip 117 and head
gave about 46 ms tokens, skip 130 a little more.

Hold, scored on the runs that follow a gate wait of at least 30 s, both arms together:
(1) resident MB before the run is higher with hold: OLMoE by at least 300 MB, Granite by at least 150 MB;
(2) tok/s of those runs higher: OLMoE +3 (+1 to +6), Granite +0.7 (0 to +2);
(3) cost: under 1 s of CPU per wait, and at most 5 s added after the gate.
Generalisation prediction: num helps both models, Granite more in per cent; hold helps both, OLMoE more.

## 4. Pass, kill and void

- **num, per model.** Pass: the ABBA effect is positive and larger than 2 × SE, and the decode skip counter is
  inside its band. Inside or outside the per-cent band is scored hit or miss separately. Unresolved or negative:
  written as "not shown on this model"; no second kit with the same setting without a new reason.
- **hold.** Pass per model: (1) holds and the tok/s difference is positive and larger than 2 × SE. **Kill:**
  on OLMoE's wait-preceded runs the resident MB before the run is not higher by at least 200 MB with hold, or
  less than half of the advised MB came back. Then the fadvise hold is closed and the next option (bulk
  re-read before token 1, memo-18 option c) is taken. If (1) holds but tok/s is unresolved: "pages kept, speed
  not shown", not a pass.
- **Void (kit, not lever):** a model hash or library hash differs; fewer than 6 scored runs per arm and model;
  texts differ within an arm; the num arm's skip counter is zero or its head line shows no active copy; the
  hold calls fail (errors on every pass), which voids the hold factor only.
- Misses are written as misses. Nothing here is scored against R58's closed lever.

## 5. Claim rule

Not a goal claim. A 25 tok/s claim needs campaign 1's r16 run list and claim positions with the frozen
configuration, pre-registered on its own. This round says only whether the num arm reaches the neighbourhood.

## 6. Hygiene

Per scored run: clock at the end, gate result, `Cached`, resident MB before and after, major faults. No run is
excluded from the ABBA means; runs after a gate timeout are listed. `/proc/self/io` was unreadable in r21; this
kit adds `ru_inblock`, which may read zero on the phone, in which case major faults stand in.

## 7. What a miss means

- num resolved on both, OLMoE num median at 22 or more: freeze the two configurations and pre-register the
  r16-protocol claim round for both models; LFM2.5 is proposed to the owner as third subject.
- num effect below band: read the profiled runs' op times first (is the skip acting in the fused path, is the
  head copy used), before any new lever.
- hold killed: memo-18 option c; the residency loss in the wait remains the main limit on both models.
