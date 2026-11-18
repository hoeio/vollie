# Pre-registration R68: claim-format rehearsal, page touch and worker priority, and two reference runtimes, power saving off (kit r28)

Committed and pushed **before** the APKs are uploaded. Do not edit afterwards; changes go into
`R68-amendment-1.md`. No tag (owner's wish); the commit hash and push time show the order.

- **Date / author:** 2026-11-18, clyde (lead, main session).
- **Campaign / models / files:** campaign 2. `granite-Mi.gguf` `8df94de7…0bc0f45`, `olmoe-Mi.gguf`
  `5abf710b…8e444abb`, run from the internal copies (as R67), sha256 checked by the kit.
- **Agent models** (full IDs): lead `claude-opus-5-5` (this file, memo-31 decisions); bonnie `claude-fable-5-2`
  and ebony `claude-fable-5-2` (memo-31, joint); tom `claude-opus-5-5` (patch 0012, app patch 0012, parser,
  three APKs, memo-32); annie `claude-sonnet-5-5` (books).
- **Builds under test** (all `1.28.0+28`, package `com.insaneutils.vollie.c2`, debug certificate `c49dde85…`, as
  r27; kit: [memo-32](../reports/memo-32-kit-r28.md)):

  | APK | runtime | bytes | sha256 |
  |---|---|---|---|
  | `vollie-c2-1.28.0+28-research-r28.apk` | c2: 620d2225c + `patches/llama.cpp/0001-0003` + `patches/llama.cpp-c1-olmoe/0001-0012`, KleidiAI on; CPU module `6f1d859a…3d28e`, `libllama.so` `126bfb4f…ede8` | 16,399,218 | `71297cd838507b1a6cc8c3e9c6ea9831a2c029346b0ec7251aee2c931fce6350` |
  | `vollie-c2-1.28.0+28-research-r28-c1.apk` | c1: campaign 1's shipped series (`patches/llama.cpp/0001-0003`, unchanged since `686f17d`), KleidiAI on; CPU module `53bfee45…d11c`, `libllama.so` `f2f4f88a…3861` | 16,368,862 | `766ff392006a5d066f86b7573f2ead74e4b1e42bdc3b8f72f0104c79fb79f3dc` |
  | `vollie-c2-1.28.0+28-research-r28-stock.apk` | stock: 620d2225c unpatched, KleidiAI on (as kit r17 / R52); CPU module `604e0f69…cd0d`, `libllama.so` `de2ccec7…cb8c` | 16,319,578 | `c9b156a10ed7e255ec8db2b01a39d69774df6024bb66955cb2218f49a93a6857` |

  App patches `0001` to `0012` in all three. Runtime patch `0012-c2-touch-nice.patch` `81c458f2…a93da`; app patch
  `0012-kit-r28.patch` `792b2866…ac83`.
- **Device and phone state:** the owner's Galaxy Z Flip 3, Android 15. **Power saving off** in every session
  (owner, 2026-11-18: the claim state); the kit records it at session start, end and per run. Charger out. The
  deep-sleeping list is on in session A and off in session B and R (background apps closed by hand there).
- **Labels:** every number in §3 is *estimated* until a result says *device*.
- **Starts from R67 / memo-31 / memo-32:** kit r27's protocol (r16 gate, hold, kv clear, 3 threads, pin pool 0xf0,
  exec mapping, internal copies, static screen, no OpenCL), unchanged in the c2 APK.
- **Lead's decisions on memo-32:** the reference runtimes are separate APKs (one process cannot swap
  `libllama.so`); sessions are split as below. Campaign 1's Granite setting (skip 117) has no PPL on granite-Mi:
  its block is a speed reference only, never a quality-checked configuration. ADPF is not in R68 (not built).

## 1. What is tested

**Sessions** (order fixed): A (c2 APK, list on), B (c2 APK, list off), R (c1 APK, then stock APK, then the c2 APK
reinstalled). A and B may be on the same day with at least 30 min between them; R on another day part or after a
cool-down. Before A, the owner runs the T-A cap probe once with power saving off and once with it on (power
saving switched back off afterwards), labelled by state; more probes over two days are welcome and are reported,
not scored.

**c2 plan, per session** (blocks O G O G O, two warm-ups each, not scored):

| block | model, arm | scored runs | configuration |
|---|---|---|---|
| b1 | OLMoE off | 10, claim format | pin + exec |
| b2 | Granite num (skip 130 + head2 2048 q4_0) | 10, claim format | pin + exec |
| b3 | OLMoE off | 16 = 4 quads, touch × nice, Williams | pin + exec |
| b4 | Granite num | 16, same | pin + exec |
| b5 | OLMoE num (`VOLLIE_SKIP_MASS=900 VOLLIE_HEAD2=1024`, R61's setting) | 10, claim format | pin + exec + touch + nice |

- **touch (T)** `VOLLIE_TOUCH=1` (0012): a helper on cpus 0-3 reads one byte per page of the hold's wanted set
  through the model mapping every 12 s of the gate wait and once at its end (memo-31 TRY 1, Y1 b). Bit-identical.
- **nice (N)** `VOLLIE_NICE=-10` (0012): the pool's graph threads set their nice to −10 and read it back (memo-31
  TRY 2). Bit-identical. Whether the phone allows it is a mechanism question (§3).
- Claim-format blocks: 10 consecutive gated runs with no factor varied, scored as a claim would be (§5), as a
  rehearsal.

**Reference plan (session R)**, each APK: Granite then OLMoE, two warm-ups and 10 claim-format runs per model,
r16 gate, hold, 3 threads, the app's DartWorker pin (the runtime pin, kv clear, exec and touch are absent in these
runtimes; their calls fail and are recorded). c1: Granite with R49 / kit r16's shipped env (skip 117 + head2
q4_0 2048 + sampler bypass + tiny-op 262144 + expert I/O 1), OLMoE with kit r19's `c1_ship`. stock: no env.

## 2. Budget accounting

- Granite num as R62 (R61 *host*: 9.5056 / 9.5120 against 9.6879). OLMoE num (b5): R61 *host* 12.0927 /
  12.0976 against 12.3003, inside at `-b 512` and `-ub 1`; it spends 0.38 of the 0.586 left. OLMoE off is stock.
- Identity of 0012 (*host*, memo-32 §2): macOS per-chunk PPL equal to off for touch, nice and all at `-b 512`,
  `-ub 32` and `-ub 1` on both models (nice is not compiled on macOS); Linux aarch64 container: `all` (pool + nice
  set successfully + exec + touch) equal to off at all three settings on both models. The single-knob Linux runs
  were stopped unfinished; the combined arm covers both Linux code paths. Greedy text: one text per process on
  macOS (OLMoE, Granite, Granite num, 7 prompts each) and on Linux without `CAP_SYS_NICE` (OLMoE 7/7, Granite 6/6),
  and with it on OLMoE; Granite num with a successful nice set was not run on the host, so the phone's
  identity check (§4) covers it.
- c1 and stock: no budget line is claimed for their Granite / OLMoE settings in this round.

## 3. Predictions (*estimated*)

**Mechanisms.**
- Power saving reads off in every run of A, B and R: probability 0.95 (owner's setting).
- Cap probe: with power saving on, the cpu 7 `scaling_max_freq` is at or below 2,150 MHz for most of the probe;
  with it off, at or above 2,419: probability 0.65 (memo-31's leading explanation, not yet shown).
- Uncapped (cpu 7 max ≥ 2,419) in at least 9 of 10 runs of b1: probability 0.75 per session.
- touch: in every T run the end pass touches ≥ 90 % of the wanted set; after waits of 60 s or more, the
  wanted MB missing at run start is lower by at least half with T than without.
- nice: the set succeeds on the phone (`c2-nice` read-back −10): probability 0.5. If it fails (EACCES), the nice
  factor is void, recorded as "not permitted".

**Levels and effects, per session** (medians of claim-format blocks; factor effect = paired within quad):

| quantity | OLMoE | Granite |
|---|---|---|
| b1 / b2 claim-block median, uncapped | **24.9 tok/s (23.0 to 25.4)** | **23.5 (22.5 to 24.5)** |
| same, capped | 22.0 (21 to 23) | 21.0 (20 to 22) |
| runs below 25.0 in b1 (of 10) | 4 to 7 | n/a |
| touch, tok/s | +0.3 (−0.5 to +1.5) | +0.1 (−0.5 to +1.0) |
| touch, wanted MB missing at run start (median of runs after waits ≥ 60 s) | −50 % (−30 to −90 %) | −50 % (−30 to −90 %) |
| nice (if permitted), tok/s | +0.3 (−0.5 to +2.0) | +0.2 (−0.5 to +1.5) |
| b5 median minus b1 median (OLMoE num + touch + nice) | **+1.0 (+0.4 to +1.6)** | n/a |

Reference runtimes (session R, uncapped, power saving off; there is no comparable earlier phone state, so the bands
are wide): c1 Granite 22.5 tok/s (19 to 25), against R49's 20.51 (*device*, power saving most likely on); c1 OLMoE
17 (12 to 22); stock Granite 13 (8 to 18); stock OLMoE 10 (5 to 18).

Generalisation: touch moves missing pages on both models, tok/s more on OLMoE; nice acts only on runs with run-queue
wait and has the same sign on both.

## 4. Pass, kill and void

- **touch, per model.** Pass: wanted MB missing at run start (runs after waits ≥ 60 s) lower by at least 30 % with
  T, and tok/s not negative beyond 2 × SE. Kill: missing MB not lower by 30 %, or tok/s negative beyond 2 × SE.
  If fewer than 3 pairs have waits ≥ 60 s, the loss endpoint is "not testable this session".
- **nice, per model (if permitted).** Pass: tok/s positive beyond 2 × SE, or, in runs where the off cell has
  run-queue wait above 100 ms per decode, that wait lower by at least half. Kill: tok/s negative beyond 2 × SE.
  Otherwise "permitted, speed not shown". Void if the set fails on the phone.
- **Claim-format blocks:** descriptive; scored against memo-31 §4's go rule (session B, all 10 runs uncapped,
  median ≥ 25.7 tok/s and at most 2 of 10 below 25.0; session A median ≥ 25.0) for the configuration that would
  be claimed. Meeting it opens a claim round (R69, its own pre-registration); missing it is reported with these
  numbers.
- **Identity on the phone:** the 32 greedy tokens are equal within a model and arm across every c2 cell (and b5
  has one text across its runs). A difference kills the knob(s) involved until explained.
- **Void (kit, not lever):** a model or library hash differs from this document; an internal copy fails its
  sha256; the charger rule of R63 §4; power saving reads on in a run (that run is reported but not scored, and a
  block with fewer than 8 valid runs is void).
- Misses are written as misses.

## 5. Claim rule

Not a goal claim. A rehearsal: no run is dropped for being slow, capped or contended; the claim-format blocks are
scored as a claim would be so that the go rule can be read.

## 6. Hygiene

Per run: block, cell, power-save state, cap and clocks, worker affinity and nice, `c2-touch` and `c2-nice` lines,
`c2-exec`, `c2-kvclear`, MB lost in the wait, wanted MB missing at run start (smaps by inode, fallback
`RssFile`), `/proc/stat` and `/proc/loadavg` deltas or their errno, TTFT, `Cached`, charger flags, library hashes.
Per session: label (list on / off), power state at start and end, the exec check rows, copy records. Per probe: the
T-A report, labelled by power-save state.

## 7. What a result means

- Go rule met in B: R69 is a claim round on that configuration.
- Cap probe follows power saving: the cap is explained; the power-saving state stays a recorded claim condition.
  It does not follow: the cap is "not predictable from the app", and P(pass) is multiplied by the share of
  uncapped sessions seen.
- touch pass: it stays on; the claim configuration includes it. Kill: Y1 b is closed on this kernel.
- nice void: ADPF is priced again as the remaining priority route. Pass: nice stays on.
- Reference blocks: campaign 1's and stock's speed under the claim state replace R49 and R52 as the comparison
  for later claims; both are reported per model with what generalised.
