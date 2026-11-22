# R72 amendment 1: build under test is kit r33 (app-only fix of r32)

Committed and pushed **before** the r33 APK is uploaded and before attempt 2 runs. `R72-probe-k1-cores-types-anon-device.md`
(`364c438`) is not edited; this note replaces only its build-under-test lines. Date / author: 2026-11-22, lead (main session,
`claude-opus-5-5`); fix by tom (`claude-opus-5-5`).

## Why

Attempt 1 (kit r32, `reports/device/r72-attempt1-c3probe-r32-2026-11-22T1749.json`) finished in 5 s with no block. Cause
(*host* repro, `reports/memo-08-work/tom-r33-build.md`): the closure passed to `Isolate.run` captured the app state through
r32's new report closures, which is unsendable; the first block threw before it was recorded. No probe ran, so attempt 1
holds no data and is not scored. It is not one of the four voids; it is a failed kit run, reported here.

## Build under test (replaces the r32 lines)

| item | value | sha256 |
|---|---|---|
| APK | `vollie-c3-probe-r33.apk`, 16,506,302 B, package `com.insaneutils.vollie.c2`, versionCode 33, versionName 1.33.0 | `1917aa4cc2b28918f21b36f74a80fb2a927f711051c501ce5550bbd38a97db7c` |
| BUILD file | `BUILD-r33.txt` | `d227277778a28e5b36a2f765403bae2f374e0ab8faabd1536b1db044189084df` |
| app patch | `patches/app-c3/0003-kit-r33-probe.patch` (after 0002) | `0ee03f5aa15e5a4934f09b52c1c20a4aefb4bb905e7ae601e5e063aa6c93aa1c` |

- The runtime is r32's unchanged: 12 of 13 native libraries in the APK are byte-identical to r32's, including `libllama.so`
  and the CPU backends; only `libapp.so` (Dart) differs. r32's knob-unset identity therefore carries over.
- App changes: isolate calls go through static helpers; an exception in one block is recorded as that block's `error`
  and the loop continues. Blocks, order, gate, cells and timing are r32's.

## Unchanged

Predictions, decision rules, voids (APK hash now against r33's), claim rule and run steps of the R72 file, with r33 in
place of r32 (install over r32 or r29; no uninstall; button "Run c3 probe r33"; file `vollie-c3probe-r33-<time>.json`).
A block carrying `error` is "not measured" and reported as such; it is not a fifth void.
