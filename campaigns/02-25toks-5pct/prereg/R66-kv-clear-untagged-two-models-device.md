# Pre-registration R66: KV and recurrent clear with MADV_DONTNEED on the untagged address, on Granite and OLMoE (kit r26)

Committed and pushed **before** the APK is uploaded. Do not edit afterwards; changes go into
`R66-amendment-1.md`. No tag (owner's wish); the commit hash and push time show the order.

- **Date / author:** 2026-11-18, clyde (lead, main session).
- **Campaign / models / files:** campaign 2. `granite-Mi.gguf` `8df94de7…0bc0f45`, `olmoe-Mi.gguf`
  `5abf710b…8e444abb`, hash checked by the kit.
- **Agent models** (full IDs): lead `claude-opus-5-5` (this file); tom `claude-opus-5-5` (tree, identity checks,
  app patch 0010, parser, build, memo-27); bonnie `claude-fable-5-2` and annie `claude-sonnet-5-5` not used.
- **Build under test:** `vollie-c2-1.26.0+26-research-r26.apk`, 17,228,499 bytes, sha256
  `e0adfb8f1ae4e08c99e4effecd6e912e8462a26a2eadfc6a22a5060278c83854`, built with `tools/c2_build_kit.sh` from
  the app copy (2.3.0+45 + `patches/app-c2/0001` to `0010`) and the tree `lab/llama.cpp-c1r26` = llama.cpp
  `620d2225c` + `patches/llama.cpp/0001` to `0003` + `patches/llama.cpp-c1-olmoe/0001` to `0009`, KleidiAI on.
  CPU module `b93c720d…b7055085` (same sources as r25; differs only in string addresses from the tree path,
  *host*), `libllama.so` `497c6d28…cc811ceaf6f`. Kit: [memo-27](../reports/memo-27-kit-r26.md). Same debug
  certificate as r25; versionCode 26, installs over r25 / r25b.
- **Device:** the owner's Galaxy Z Flip 3. **Labels:** every number in §3 is *estimated* until a result says
  *device*.
- **Relation to R65:** R65 (kit r25b, an r25 APK with libllama.so swapped while `~/Documents` was unreadable) is
  superseded for the phone by this round now that the app source is reachable. If an r25b report exists, it is
  still scored under R65 as written; R65 is not edited.
- **Starts from R64:** kit r25's protocol (memo-26) with two simplifications from R64's results: the gate is
  campaign 1's r16 rule only (plateau not shown on either model), and each model runs one arm (Granite num,
  frozen after R62; OLMoE off, since R63 §4 parks `MASS=900`). OLMoE keeps its load env `VOLLIE_HEAD2_REPACK=1`.
- **Phone state asked of the owner:** background apps closed, charger out, phone on the desk.

## 1. What is tested

**kvclear**, declared bit-identical: `VOLLIE_KV_DONTNEED=1` set against unset. Patch 0008 releases the page-
aligned interior of the KV (and Granite's recurrent) buffer with `madvise(MADV_DONTNEED)` before every prompt
instead of a memset; 0009 does the range, the `/proc/self/maps` check and `madvise` on the address with its top
byte cleared (aarch64) and prints `tag=0x..`. R64 printed `why=map` in every kv run; inferred cause: Android's
tagged heap pointers.

Design (memo-27): blocks Granite, OLMoE, OLMoE, Granite; per block two warm-ups (model's arm, no kv), two ABBA
quads of the kv factor with the r16 gate (headroom below 0.65, or 90 s), two profiled runs (no kv); hold, charger
guard, three threads. Kv order: b1 G `K p p K p K K p`, b2 O `p K K p K p p K`, b3 O `K p p K p K K p`, b4 G
`p K K p K p p K`. 8 kv and 8 plain scored runs per model; 48 runs, 40 gated. Run ids `granite_num_r16[_kv]`,
`olmoe_off_r16[_kv]` (warm-ups `_wu`, profiled `_prof`).

Env: Granite every run `GGML_TINY_OP=262144 GGML_EXPERT_IO=1 VOLLIE_SAMPLER_BYPASS=1 GGML_C2_IO=1
GGML_EXPERT_SKIP=130 GGML_HEAD2=2048 GGML_HEAD2_Q=q4_0 VOLLIE_SKIP_STATS=1`; OLMoE every run `GGML_TINY_OP=262144
GGML_EXPERT_IO=1 VOLLIE_SAMPLER_BYPASS=1 GGML_C2_IO=1`; kv runs add `VOLLIE_KV_DONTNEED=1`.

## 2. Budget accounting

No numerics change. Granite num as R62 (R61 *host*: 9.5056 / 9.5120 against 9.6879); OLMoE off is stock.
Identity of 0008 + 0009, knob on against off (*host*, memo-27): per-chunk PPL equal on both models at
`-b 512` and `-b 512 -ub 32` (20 chunks) and `-b 512 -ub 1` (8 chunks), on macOS (memset fallback, `why=os`) and
in a Linux aarch64 container (`why=ok`, OLMoE 67,104,768 B released per clear at ctx 512); greedy text equal over
three clears per process at the app's settings (ctx 4096, q8_0), on and off, both models, both systems (Linux:
285,208,576 B released for OLMoE, 17,821,696 + 58,056,704 B for Granite); the tagged-pointer test passes. glibc
does not tag pointers, so the untagging itself is tested only by `tools/c2_kvclear_tag_test.cpp`.

## 3. Predictions (*estimated*, kv minus plain, per model, paired per ABBA quad)

| quantity | OLMoE | Granite |
|---|---|---|
| `c2-kvclear` in kv runs | `why=ok`, `kv` released ≥ 250 MB, `tag` non-zero | `why=ok`, `kv` + `rs` released ≥ 50 MB, `tag` non-zero |
| pre-prompt major faults, mean | −3,000 (0 to −15,000) | −1,000 (0 to −10,000) |
| TTFT | −40 ms (0 to −250) | −10 ms (+50 to −120) |
| tok/s | 0 (−1.0 to +1.0) | 0 (−1.0 to +1.0) |

The fault and TTFT bands are wide on purpose: in R64 the plain medians of pre-prompt faults were 6 and 26, with
single runs up to 40k–53k, so the mean effect depends on how many runs fault.

Mechanism prediction (the one this round is for): `why=ok` in every kv run's clears on both models, with a
non-zero `tag`.

## 4. Pass, kill and void

- **Mechanism, per model.** Pass: `why=ok` in every clear line of at least 7 of the 8 kv scored runs. (R65 wrote "15 of 16"; r25 and r26 have 8 kv scored runs per model, so R65's count was a slip, noted there.)
  Miss: otherwise; the printed `why` and `tag` say which step failed. If `tag=0x00` with `why=map`, the tagged-
  pointer reading is wrong.
- **Effect, per model.** Pass: mean pre-prompt faults lower with kv and TTFT lower by more than 2 × SE (paired over the 4 quads; Welch over runs reported too).
  **Kill:** any text differs between kv and plain within an arm (the identity claim is wrong; the knob is closed
  until explained), or TTFT higher by more than 2 × SE. Otherwise "mechanism works, effect not shown".
- **Void (kit):** a model or library hash differs from this document; released > 0 in a run without kv; the charger rule
  of R63 §4 (a flagged run is dropped with its quad; void for a model below 3 quads).
- Misses are written as misses.

## 5. Claim rule

Not a goal claim.

## 6. Hygiene

Per run: kv arm, every `c2-kvclear` line (why, released, tag), pre-prompt faults, TTFT, wait length, `Cached`,
clock cap, charger flags.

## 7. What a result means

- Mechanism and effect pass on both: the knob is on in every later kit.
- Mechanism passes, effect not shown: the knob is kept on (it costs nothing on the metric and frees memory and
  swap slots); its value is memory headroom, not speed; the next memory lever is memo-24 item 3.
- Mechanism misses: the clear path on Android is looked at from the printed `why` and `tag`; no further memory
  lever is pre-registered until it is understood.
