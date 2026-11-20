# R70 amendment 1: the claim session runs on kit r30's c2 runtime (r28 cannot be reinstalled over r30)

Date: 2026-11-20. Author: clyde (lead). Written and pushed before any R70 data exists. Owner, 2026-11-20: "추천대로 하죠"
(go with the recommendation).

## Why

R70 §0 names the r28 c2 APK (versionCode 28). Kit r30 (R71, versionCode 30) is installed over r28-c1 for the R71
re-run. After that, r28 is a downgrade, and the phone refuses it. Uninstalling to reinstall r28 would delete the
internal model copies (about 6 GB), which R70 runs from.

## Change

The build under test is kit r30, in its **c2 runtime**:

| field | value |
|---|---|
| APK | `vollie-c2-1.30.0+30-research-r30.apk` (HF `r30/vollie-r30.apk`), 20,795,890 bytes |
| sha256 | `5459fc67d8e0073883e28d87ed02769f4f1f8eac7d8783ae0a7092e48de3221b` |
| runtime | c2: `libllama.so` `126bfb4f…ede8`, CPU module `6f1d859a…3d28e` |
| other native libraries | byte-identical to r28's |

- The c2 libraries are those of R70 §0, byte-identical, unpacked from the r28 APK and not rebuilt.
- **What differs:** the kit code (`libapp.so`, app patch `0013-kit-r30.patch`). It adds the runtime selector, a
  check of the mapped libraries per load, and the unattended chain. The c2 run list and its rows keep proto `r28`,
  so the plan b1 to b5 is R70 §1's, with the same order, knobs and env.
- **How it is run:** Settings shows "Runtime: c2". The owner taps **"Run research kit r30 once, this runtime"**.
  R70 does **not** use the Auto chain, and **no R71 segment counts as R70**: R70 is its own session. It runs after
  R71 has finished, on another day part or after the 30 min rest of R70's phone-state rules.
- **Void additions:**
  - `runtime_info.active` is not `c2`;
  - any load row's `runtime_check` fails;
  - the APK hash differs from the table above.

  These replace the "APK hash" part of R70 §2's void list. The library and model hash rules stand.

Nothing else changes: no prediction, band, claim rule or reading.
