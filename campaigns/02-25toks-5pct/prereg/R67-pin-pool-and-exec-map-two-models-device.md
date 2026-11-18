# Pre-registration R67: runtime thread pin with a persistent pool, and the model mapped PROT_EXEC, on Granite and OLMoE (kit r27)

Committed and pushed **before** the APK is uploaded. Do not edit afterwards; changes go into
`R67-amendment-1.md`. No tag (owner's wish); the commit hash and push time show the order.

- **Date / author:** 2026-11-18, clyde (lead, main session).
- **Campaign / models / files:** campaign 2. `granite-Mi.gguf` `8df94de7…0bc0f45`, `olmoe-Mi.gguf`
  `5abf710b…8e444abb`. Both are now also copied to the app's internal storage (same bytes, sha256-checked); the
  file is unchanged.
- **Agent models** (full IDs): lead `claude-opus-5-5` (this file, decisions); tom `claude-opus-5-5` (patches 0010,
  0011, app patch 0011, parser, build, memo-28, memo-30); bonnie `claude-fable-5-2` and ebony `claude-fable-5-2`
  (memo-29, which proposed Y1 and T1-T3; ebony ran as a stand-in with `ebony.md`'s text); annie
  `claude-sonnet-5-5` (books).
- **Build under test:** `vollie-c2-1.27.0+27-research-r27.apk`, 16,352,258 bytes, sha256
  `c05ebfd1a1454f726ab3632bdd16638d41547874dc0ce03057c4e540da3986d4`; tree `lab/llama.cpp-c1r27` = llama.cpp
  `620d2225c` + `patches/llama.cpp/0001` to `0003` + `patches/llama.cpp-c1-olmoe/0001` to `0011`, KleidiAI on;
  CPU module sha256 `edc68be1e4f3840e185955ca0c5a59c48b7a6c46852a4f8fe64917f4b9077398`,
  `libllama.so` `1394a5fe1b01022df47eaf6cc49e76405ecd121cd93b6e7dbd847fcc6efa12f2`; app patches `0001` to `0011`;
  no OpenCL libraries. Same debug certificate as r26. Kit: [memo-30](../reports/memo-30-kit-r27.md).
- **Device:** the owner's Galaxy Z Flip 3, Android 15. **Bench condition (owner, 2026-11-18):** the apps that
  restarted themselves in R66 are in Samsung's deep-sleeping list. Recorded; a goal claim must also hold without
  it. Charger out.
- **Labels:** every number in §3 is *estimated* until a result says *device*.
- **Starts from R66 / memo-28 / memo-29:** kit r26's protocol (r16 gate, Granite num, OLMoE off, blocks G O O G,
  warm-ups, profiled runs, hold, charger guard, three threads), kv clear on in every run (R66 §7).

## 1. What is tested

Two runtime levers, both declared bit-identical and off unless set, as a balanced 2 × 2 inside every quad of four
scored runs (cells Px, pX, PX, px; Williams order per memo-30 §4), so each factor has 8 against 8 scored runs per
model:

- **pin (P)** `VOLLIE_POOL=1 VOLLIE_CPUMASK=0xf0` (patch 0010): a persistent threadpool whose workers the runtime
  pins to cpus 4-7 at graph start, so the pin no longer depends on which Dart thread calls decode. Reason
  (memo-28 #1): in r23-r26 the workers ran on the A55 cores in 39 of 60 OLMoE gated and profiled runs and in 0 of
  12 warm-ups (*device*); campaign 1's pin is lost after the gate wait (inferred cause: Dart retires the pinned
  thread). p = today's behaviour.
- **exec (X)** `VOLLIE_EXEC_MAP=1` (patch 0011): the model mapping is `mprotect`-ed to `PROT_READ|PROT_EXEC`
  before the gate wait, so its pages are on Linux 5.4's executable-file reclaim path (referenced exec file pages
  are re-activated rather than demoted; Ebony's reading of upstream `mm/vmscan.c`, memo-29 Y1). No read-back
  through the mapping in this kit. x = plain `PROT_READ`.

Session start (owner's order): the exec check runs first on internal storage (mountinfo row; a 1 MiB probe file:
open, mmap, `mprotect` to exec, fresh exec mmap, errno each). Only if both exec calls succeed are the models
copied from the external files (or downloaded if absent), sha256-checked, and the session runs from the internal
copies; otherwise the session runs from the external files and the exec factor is void. The external folder and
the first 1 MiB of each external model file are probed the same way (T2).

Declared instrument changes in every run (memo-28 (a), #2, #5, #6; memo-30): bookkeeping cleanups, static kit
screen without the spinner, no OpenCL, `n_ctx` fixed at 4096, `VOLLIE_SCHED=1` (T1 state counters). The profiled
`_profb` runs add T3's block-mode profile. These change the instrument, not the comparison: every arm of R67 has
them, so levels are not compared with R66.

## 2. Budget accounting

No numerics change. Granite num as R62 (R61 *host*: 9.5056 / 9.5120 against 9.6879); OLMoE off is stock.
Identity (*host*, memo-30 §2): per-chunk PPL equal to off for pool, exec and both, at `-b 512` and `-b 512 -ub
32` (20 chunks) and `-b 512 -ub 1` (8 chunks), both models, macOS and Linux aarch64 (where the pin and
`mprotect` paths run: one cpu per worker, `prot=r-xs`, errno 0); greedy text equal across the four cells on both
systems, Granite also in its num arm.

## 3. Predictions (*estimated*; factor effect = mean of its on runs minus its off runs, paired within quad)

**Mechanisms.**
- Internal exec check passes (both exec calls errno 0): probability 0.7. External (FUSE) path refuses exec:
  probability 0.8. Recorded either way.
- pin: in every P run every worker's affinity reads `0xf0` and at least 95 % of thread samples are on cpus 4-7;
  in p runs, A55 samples (cpus 0-3) appear in at least 2 of the 8 p scored runs per model.
- exec (if not void): `prot=r-x` on the model range in every X run, `where=api` (applied before the gate) in at
  least 7 of the 8 X scored runs per model.

**Effects, per model.**

| quantity | pin, Granite | pin, OLMoE | exec, Granite | exec, OLMoE |
|---|---|---|---|---|
| tok/s | +1.0 (−0.5 to +3.0) | +1.0 (−0.5 to +3.0) | 0 (−1.0 to +1.5) | +0.5 (−1.0 to +3.0) |
| decode ms/token | −2 (+1 to −6) | −2 (+1 to −6) | 0 (+2 to −3) | −1 (+2 to −6) |
| file MB lost in the gate wait, median | 0 (±100) | 0 (±100) | −30 % (0 to −60 %) | −30 % (0 to −60 %) |
| wanted MB missing at run start, median | 0 (±100) | 0 (±100) | −30 % (0 to −60 %) | −30 % (0 to −60 %) |

Interaction pin × exec: none expected (0 ± 1 tok/s); reported descriptively.

Generalisation: pin helps both models by a similar amount; exec, if it works, helps OLMoE more than Granite
(R66: OLMoE's decode tracks missing pages, Granite's weakly).

## 4. Pass, kill and void

- **pin, per model.** Pass: tok/s difference positive and beyond 2 × SE (paired over quads), with the pin
  mechanism met. **Kill:** negative beyond 2 × SE: the pool + pin is closed in this form. Otherwise "pin restored,
  speed not shown": the pin stays on in later kits anyway, because it restores campaign 1's intended placement
  (a protocol correction, not a lever claim).
- **exec, per model (if not void).** Pass: median file MB lost in the wait lower by at least 25 % with X, and
  tok/s difference not negative beyond 2 × SE. **Kill:** the loss is not lower by 25 % (Ebony's rule, memo-29), or
  tok/s negative beyond 2 × SE: Y1 is closed on this kernel. Pass on loss with tok/s within 2 × SE: "residency
  shown, speed not shown"; it stays on and is re-tested with the read-back (Y1 b).
- **Identity on the phone:** the 32 greedy tokens are equal in all four cells of a model. A difference kills the
  knob(s) involved until explained.
- **Void (kit, not lever):** a model or library hash differs from this document; an internal copy fails its
  sha256; the charger rule of R63 §4 (a flagged run is dropped with its quad; a factor is void for a model below
  3 quads). The exec factor alone is void if the internal exec check fails; a P run whose workers are not
  pinned voids the pin factor for that model.
- Misses are written as misses.

## 5. Claim rule

Not a goal claim. The deep-sleeping bench condition and internal storage are recorded.

## 6. Hygiene

Per run: cell, worker affinity and sampled cores, `c2-exec` line (prot, errno, where), `c2-kvclear`, file MB
lost in the wait, wanted MB missing at run start, T1 counters (and whether they were readable), TTFT, clock cap,
`Cached`, charger flags. Per session: the exec check rows (internal, external, model files), copy or download
records, free space.

## 7. What a result means

- pin pass on both: the pool + 0xf0 pin is the new base; strict one-core-per-thread placement (Y4) is the next
  placement arm. Pin not shown: the pin stays on as a protocol correction; Y4 is priced again from T1.
- exec pass or "residency shown": exec stays on; the read-back through the mapping (Y1 b) is built next, and
  `MASS=900` may be re-tested once T1 shows the memory state under control (R63 §4).
- exec kill: Y1 is closed on this kernel; the internal copies may be removed at the owner's word.
- exec void: the probe rows say which call failed; no further exec-mapping work.
