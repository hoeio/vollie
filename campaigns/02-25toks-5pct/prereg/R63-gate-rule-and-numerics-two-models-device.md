# Pre-registration R63: plateau gate rule against the r16 gate, and skip + head, on Granite and OLMoE (kit r24)

Committed and pushed **before** the APK is handed to the owner. Do not edit afterwards; changes go into
`R63-amendment-1.md`. No tag (owner's wish); the commit hash and push time show the order.

- **Date / author:** 2026-11-16, clyde (lead).
- **Campaign / models / files:** campaign 2. `granite-Mi.gguf` `8df94de7…0bc0f45`, `olmoe-Mi.gguf`
  `5abf710b…8e444abb`, on the phone, hash checked by the kit.
- **Agent models** (full IDs): clyde `claude-fable-5-2`; tom `claude-opus-5-5` (kit r24, memo-23, the fault
  analysis); bonnie `claude-opus-5-5` and annie `claude-sonnet-5-5` not used this round.
- **Build under test:** `vollie-c2-1.24.0+24-research-r24.apk`, 17,221,503 bytes, sha256
  `6a655ca7e16a9f4ee99c86ef456463b640b967bb7dfb4e6e421e3a9affa5beba`. Runtime as R62 (llama.cpp `620d2225c` +
  `patches/llama.cpp/0001` to `0003` + `patches/llama.cpp-c1-olmoe/0001` to `0006`, KleidiAI on); the libraries
  were rebuilt and differ from r23's only in the ELF build ID (memo-23, *host*): CPU module `e6ef857b…dc3a84ef`,
  `libllama.so` `aeab4dad…1cb4b9df`. App patches `patches/app-c2/0001` to `0008`. Kit:
  [memo-23](../reports/memo-23-kit-r24.md).
- **Device:** the owner's Galaxy Z Flip 3. **Labels:** every number in §3 is *estimated* (from R60 and R62,
  *device*, and their replay on the host) until a result says *device*.
- **Starts from campaign 1:** its runtime, prompt, 32 greedy tokens, metric and kit r16's gate, which is one of
  the two gate rules compared here. R62 is the comparison round.
- **Phone state asked of the owner:** background apps closed, charger out (the kit now pauses while a charger
  is connected), phone on the desk as before. Any time of day.

## 1. What is tested

Blocks Granite, OLMoE, OLMoE, Granite; two ABBA quads (off num num off) per block; 16 scored runs per model;
three threads throughout; hold as in r23 in every gated run; OLMoE loaded with `VOLLIE_HEAD2_REPACK=1`.

- **gate rule** (protocol, no numerics), one quad each per block, order alternated: **r16** (headroom below
  0.65, or 90 s) against **plateau** (below 0.65; or, from 30 s on, the lowest of the last two 5 s readings is
  no more than 0.01 below the lowest of the two before and at least 0.05 below the wait's highest reading; or
  90 s). Reason: in R60 and R62 the headroom stopped near 0.72 after about 40 s and every wait ran to 90 s.
- **num** (declared numerics change), unchanged: OLMoE `VOLLIE_SKIP_MASS=900 VOLLIE_HEAD2=1024`; Granite
  `GGML_EXPERT_SKIP=130 GGML_HEAD2=2048 GGML_HEAD2_Q=q4_0` (frozen after R62).
- **Recorded to test a reading, no setting changed:** system swap-in and fault counters per run
  (`/proc/vmstat`), the process's `VmSwap`, `RssAnon`, `RssFile`; idle baseline; thermal zones if readable.

## 2. Budget accounting

As R62 (R61, *host*, KleidiAI build): OLMoE num 12.0927 / 12.0976 against 12.3003; Granite num 9.5056 / 9.5120
against 9.6879. The gate rule and the new counters change no arithmetic.

## 3. Predictions

**Exercised or not.** The gate-rule factor is *exercised* for a model if at least 6 of its 8 scored r16-rule
runs wait 80 s or longer. If not (a day on which the 0.65 gate is reached), the factor is "not exercised" for
that model and only its descriptive rows are reported.

If exercised (plateau minus r16, 8 against 8 scored runs per model, both arms, Welch SE):

| quantity | Granite | OLMoE |
|---|---|---|
| wait length | −38 s (−30 to −45) | −38 s (−30 to −45) |
| headroom at run start | +0.00 (−0.02 to +0.03) | the same |
| file MB lost in the wait | −250 (−100 to −450) | −250 (−100 to −450) |
| `VmSwap` at run start | −60 MB (0 to −150) | −60 MB (0 to −150) |
| tok/s | **+1.2 (−0.3 to +3.0)** | **+1.5 (−0.5 to +3.5)** |

num minus off, mean of the four quads: Granite **+2.4 tok/s (+1.4 to +3.4)**; OLMoE **+1.0 (−1.0 to +3.0)**
(R60 skip alone +1.71 ± 0.70, R62 +0.15 ± 1.00; the head is priced at 0). Skip counters 39.6 % (38 to 44) and
13.0 % (11.5 to 14.5).

Levels, r16-rule quads, median of 4, scored if the clock is 2,112 MHz in the profiled runs and `Cached` is at
least 2.4 GB before 14 of a model's 16 scored runs: Granite off 16.0 (14.5 to 18.5), num 18.0 (16.5 to 20.5);
OLMoE off 17.0 (15.0 to 21.0), num 17.5 (15.5 to 22.0).

**Swap reading (memo-23 §1, inferred so far):** most major faults of a gated run are swap-ins of the app's own
memory. Prediction: over the gated runs of each model, the system's `pswpin` during the run is at least half
of the run's major faults in the median run, and the two correlate with r of at least 0.6. If `/proc/vmstat`
is not readable in the app the prediction is not scorable.

Generalisation prediction: the plateau rule helps both models by a similar amount; num is resolved on Granite
and not on OLMoE.

## 4. Pass, kill and void

- **gate rule, per model, if exercised.** Pass: tok/s difference positive and larger than 2 × SE, with the wait
  shorter and the headroom at run start not higher by more than 0.03. **Kill:** tok/s difference negative and
  larger than 2 × SE, or the clock capped below 2,112 MHz in more plateau runs than r16 runs by 3 or more: the
  plateau rule is closed. Otherwise "shorter session, speed not shown": the rule may still be used in research
  kits to save time, not as a speed lever.
- **num, per model.** As R62: positive and larger than 2 × SE, skip counter in its band, head line active with
  no declines. **OLMoE, third round:** if not shown again, `MASS=900` is not carried into further device rounds
  until the memory state before the run (swap, pages) is under control; a host-side reason is needed to bring
  it back.
- **Charger.** A scored run flagged `plugged_run` is dropped together with its quad from every statistic and
  listed. Void for a model if fewer than 6 scored runs per arm remain.
- **Void (kit, not lever):** a model or library hash differs from this document; texts differ within an arm;
  a num run's skip counter is zero or its head line shows a decline or no active copy; OLMoE's load setting is
  missing; the recorded gate rule differs from the spec (voids the gate factor).
- Misses are written as misses.

## 5. Claim rule

Not a goal claim. A 25 tok/s claim uses campaign 1's r16 gate and claim positions unless the owner decides
otherwise beforehand; a pass of the plateau rule here is the material for that decision, not the decision.

## 6. Hygiene

Per scored run: gate rule, why the wait ended, wait length, clock and cap, `Cached`, resident and wanted MB,
`VmSwap`, major faults, swap-ins, charger flags. Idle baseline and thermal zones once per session.

## 7. What a result means

- Plateau passes: research kits use it; the owner is asked whether a claim round may.
- Swap reading confirmed: the next lever is keeping the app's anonymous memory (about 0.5 to 0.7 GB: repack
  copies, KV, head copies) out of zram during the wait, or making it smaller; pre-registered on its own.
- Swap reading missed: the fault source is looked for again before any residency lever.
- Not exercised on both models: the day is recorded with its baseline; R58's levels are the comparison.
