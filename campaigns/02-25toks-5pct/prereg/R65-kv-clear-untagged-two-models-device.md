# Pre-registration R65: KV and recurrent clear with MADV_DONTNEED on the untagged address, on Granite and OLMoE (kit r25b)

Committed and pushed **before** the APK is uploaded. Do not edit afterwards; changes go into
`R65-amendment-1.md`. No tag (owner's wish); the commit hash and push time show the order.

- **Date / author:** 2026-11-17, clyde (lead, main session).
- **Campaign / models / files:** campaign 2. `granite-Mi.gguf` `8df94de7…0bc0f45`, `olmoe-Mi.gguf`
  `5abf710b…8e444abb`, hash checked by the kit.
- **Agent models** (full IDs): lead `claude-opus-5-5` only (patch 0009, kit r25b, this file); no subagents.
- **Build under test:** `vollie-c2-1.25.0+25-research-r25b.apk`, 17,249,029 bytes, sha256
  `ea61e0ad46196f636202f3430af987de0749a815659b6f589853e96babfcc921`, signed with the same debug key as r25
  (certificate `c49dde85…69fbb1716`), zipaligned for 16 KB pages.
  - **What r25b is:** the r25 APK (`1bbbec0c…c4180d4`) with two entries replaced: `lib/arm64-v8a/libllama.so`
    (rebuilt from llama.cpp `620d2225c` + `patches/llama.cpp/0001` to `0003` + `patches/llama.cpp-c1-olmoe/0001`
    to `0009`, `android/build_cpu.sh`, NDK 28.2, sha256 `4c651c4b…9ea78c81`) and `assets/kit_build.json` (0009
    added to the series, the new libllama hash, `kit: r25b`). Every other entry, the CPU module
    (`d6b7b3c9…c1f731c`) and the app code included, is r25's byte for byte.
  - **Why this way:** macOS denies this session and the owner's remote shell access to `~/Documents` since
    2026-11-17, where the app source and the build trees are; the APK could be rebuilt only from its parts.
  - **Rebuild check:** the CPU module rebuilt from the same series at a path of the same length differs from
    r25's only in string addresses (the disassembly differs in `adr`/`add`/`adrp` operands and nowhere else), so
    the rebuilt libllama is taken to be r25's plus 0009 (*host*, inferred for libllama, which was not compared
    instruction by instruction).
  - The app's own kit name stays `r25` in the report (it is compiled into the Dart code); the build record says
    `r25b`.
- **Device:** the owner's Galaxy Z Flip 3. **Labels:** every number in §3 is *estimated* until a result says
  *device*.
- **Phone state asked of the owner:** as R64. Install over r25 (same package, key and version).

## 1. What is tested

**kvclear**, declared bit-identical: `VOLLIE_KV_DONTNEED=1` set against unset. 0009 computes the page range, the
`/proc/self/maps` check and `madvise` on the buffer address with its top byte cleared (aarch64), memsets the edges
through the original pointer, and adds `tag=0x..` (the address's top byte) to the `c2-kvclear` line. R64 printed
`why=map` in every kv run; inferred cause: Android's tagged heap pointers.

Design: kit r25's protocol unchanged (memo-26): blocks G O O G, two warm-ups, two ABBA quads of num/off per block
with plateau and r16 gates, two profiled runs, three threads, hold, charger guard; kv alternating per scored run,
8 kv and 8 plain scored runs per model, paired within arm × gate cell.

**Owner's decision (2026-11-17):** the protocol cannot be changed without the app source, so OLMoE's num runs
(`VOLLIE_SKIP_MASS=900 VOLLIE_HEAD2=1024`) run again although R63 §4 parks `MASS=900`. In this round they are
**recorded, not scored** as a num lever: no num pass or kill for OLMoE is read from R65. Granite num and both
gate rules are reported descriptively only (their questions were answered in R64).

## 2. Budget accounting

No numerics change. Host identity of 0008 (R64 §2) stands. 0009 changes only the address arithmetic inside the
Linux branch and the log line; on macOS it compiles to the same path as 0008 (the `os` fallback), so a macOS
per-chunk run cannot test it. Tested instead (*host*, Ubuntu 24.04 aarch64 container):
`tools/c2_kvclear_tag_test.cpp` tags a 64 MB anonymous mapping with 0xb4; the maps check fails on the tagged range
and passes on the untagged one, `madvise` succeeds, every byte reads zero, 67,104,768 B released. Not repeated:
the per-chunk PPL of 0008 on Linux with 0009 applied (the models are in `~/Documents`).

## 3. Predictions (*estimated*, kv minus plain, per model, paired)

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

- **Mechanism, per model.** Pass: `why=ok` in at least 15 of 16 kv runs (in every clear line of those runs).
  Miss: otherwise; the printed `why` and `tag` say which step failed. If `tag=0x00` with `why=map`, the tagged-
  pointer reading is wrong.
- **Effect, per model.** Pass: mean pre-prompt faults lower with kv and TTFT lower by more than 2 × SE (paired).
  **Kill:** any text differs between kv and plain within an arm (the identity claim is wrong; the knob is closed
  until explained), or TTFT higher by more than 2 × SE. Otherwise "mechanism works, effect not shown".
- **Void (kit):** a model or library hash differs from this document; released > 0 in a run without kv; R63 §4's
  charger rule.
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
