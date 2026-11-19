# Pre-registration R70: campaign 2 claim round, OLMoE num + touch, deep-sleeping list on, power saving off (kit r28 c2, unchanged)

Committed and pushed **before** the claim session runs. Do not edit afterwards; changes go into
`R70-amendment-1.md`. No tag from the session; the owner creates `prereg-R70` on this commit.

- **Date / author:** 2026-11-19, clyde (lead, main session). R69 is campaign 3's probe (pre-registered, not run), so
  this claim round takes R70.
- **Owner decisions this round rests on:**
  - 2026-11-18: the claim runs with power saving off.
  - 2026-11-19: a claim round may run with the deep-sleeping list on. R68's go rule had chosen list off as the
    stricter state; the owner allowed list on after R68 sessions A and B.
- **Agent models** (full IDs): lead `claude-opus-5-5`; bonnie `claude-fable-5-2` and ebony `claude-fable-5-2`
  (memo-31, R68 audit and scorer); tom `claude-opus-5-5` (kit r28); annie `claude-sonnet-5-5` (books).
- **Build:** the r28 c2 APK, **unchanged** (no rebuild):
  - `vollie-r28-c2.apk`: 16,399,218 bytes, sha256 `71297cd838507b1a6cc8c3e9c6ea9831a2c029346b0ec7251aee2c931fce6350`.
  - Same as R68's c2 build (runtime and library hashes in R68 §0).
- **Model and file:** `olmoe-Mi.gguf`, sha256 `5abf710b…8e444abb`, internal copy, sha checked by the kit.
- **Configuration claimed (the kit's b5):**
  - Arm num: `VOLLIE_SKIP_MASS=900 VOLLIE_HEAD2=1024` plus OLMoE's load env (`VOLLIE_HEAD2_REPACK=1`).
  - Touch on: `VOLLIE_TOUCH=1`, `VOLLIE_TOUCH_S=12`.
  - Also pin pool 0xf0, exec map, kv clear, 3 threads, r16 gate with hold.
  - `VOLLIE_NICE=-10` is set, but the threads are already at −10 as the app's baseline (amendment 4 of R68).
  - It is recorded as a condition, not as a lever.
- **Budget (numerics):** R61 *host* PPL 12.0927 at `-b 512` and 12.0976 at `-ub 1`, against the budget 12.3003 =
  stock served 11.7146 × 1.05. Inside at both settings. Hard ceiling F16 10.9420 × 1.20 = 13.130, inside. Touch, pin,
  exec and kv clear are bit-identical (R66 to R68). The greedy text is checked on the phone (§4).
- **Phone state:**
  - Galaxy Z Flip 3, Android 15.
  - Power saving **off**, the deep-sleeping list **on**, charger out.
  - Before starting, the phone rests 30 minutes or more with the screen off, not charging and not in use (R68
    cap probes: thermal status ≥ 1 caps the big cores whatever power saving is).
  - The start screen's label is "deep-sleeping list on".
- **Labels:** every number in §3 is *estimated* until the result says *device*.

## 1. What is run

One full kit session of the r28 c2 APK, exactly as R68 session A: blocks b1 to b5 in the kit's fixed order. No knob
or order is changed, so the claim block runs in the state that was rehearsed.

- **b5 (OLMoE num + touch, 10 claim-format runs after two warm-ups) is the claim block.**
- b1 to b4 are reported, not claimed. b2 is Granite num, 10 claim-format runs: the second model is reported per model,
  as campaign 2's two purposes require.

## 2. Claim rule (CAMPAIGN.md, draft, applied as written)

- **Pass:** the median `decodeTokensPerSec` of b5's 10 scored runs is **≥ 25.0 tok/s**.
- **Counted runs:** every scored run counts. No run is dropped for being slow, capped or contended; the warm-ups are
  not scored.
- **Void**, which means no claim either way and the round is repeated under a new number:
  - the APK, library or model hash differs from this file;
  - power saving reads on in any b5 run;
  - the charger is plugged before or after any b5 run;
  - the process is killed during b5;
  - the deep-sleeping label reads off;
  - fewer than 10 scored b5 runs exist.
- **Identity:** the 32 greedy tokens of all b5 runs are one text, and that text equals R68's OLMoE num text (md5
  `40f6ce50…`). A difference voids the claim until it is explained.
- Caps are **reported** per run (cap before and after, C1 strict) and do not exclude a run.

## 3. Predictions (*estimated*)

- b5 median 26.5 tok/s (24.8 to 28.0). P(pass) = 0.80, from R68 session A (27.25, 0 of 10 below 25.0, list on) and
  session B (25.47, list off). Session-to-session drift is the main risk.
- b5 runs below 25.0: 0 to 3.
- Uncapped (C1 strict) in at least 9 of 10 b5 runs: P 0.8.
- b1 OLMoE off median 24.8 (23.0 to 26.0); b2 Granite num 23.2 (22.0 to 24.5).

## 4. What a result means

- **Pass:** campaign 2's goal is met on OLMoE-1B-7B (`olmoe-Mi`, Q4-class mix, budget +5 %): "median X tok/s,
  *device*, R70". Granite's b2 median is reported beside it as not reaching 25. The claim states its conditions:
  power saving off, deep-sleeping list on, cool start, internal model copy.
- **Fail:** the miss is reported with these numbers. Campaign 2 closes as "25 tok/s not reached under the claim
  rule", with R68's rehearsal values beside it, unless the owner asks for another round.
- **Void:** a new claim round under a new number. This file stays as it is.
- Session R's reference values (campaign 1 runtime and stock llama.cpp, power saving off) are reported beside the claim
  when they are in.
