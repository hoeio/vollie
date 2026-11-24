# R75 amendment 1: re-run of the unmeasured blocks on kit r36 (x4f harness fix; parent-hang rule)

This note is committed and pushed **before** the r36 APK is uploaded and before the re-run. The pre-registration
`R75-x4-gpu-join-probe-device.md` (commit `2588f40`, tag `prereg-R75`) is not edited. This note replaces only what it
names below; everything else of the R75 file stands as written.

- **Date / author:** 2026-11-24, clyde (lead). Harness fix and kit r36: tom.
- **Agent models** (full IDs): clyde `claude-opus-5-5` (this note, the R75 result); tom `claude-opus-5-5` (kit r36,
  the x4f fix); annie `claude-sonnet-5-5` (books); bonnie `claude-fable-5-2`, ebony `claude-fable-5-2` (no new
  bands in this note; their R75 bands stand).

## 1. Why

Session 1 (kit r35, 2026-11-24 10:35:52, `reports/device/vollie-c3probe-r35-2026-11-24T1035.json`) completed x4i,
x4a and x4b, then hung in x4f until the owner force-stopped the app at about 13:14. Scored in
[R75-result.md](../reports/R75-result.md):
- x4i, x4a, x4b are scored;
- x4f is void (§5.5, process killed during the block);
- x4c, x4d, x4e, x4m, x4s, x4r are not measured.

Probable cause (lead's reading of `0007-x4-join-probe.patch`, `x4f`; Tom confirms or corrects it in his r36 build
note):
- In the `gate` iteration, the kernel is enqueued behind a user event and flushed.
- Then `clSetUserEventStatus` releases it, and `spin_ev()` polls the kernel's event with no further `clFlush` and no
  deadline.
- If the driver does not submit a command released after its flush, the poll never ends. x4f runs in the parent
  process, so no child timeout applies.

No line of the R75 file predicted a hang in a parent block. This note adds a rule for one (§5) before the re-run.

## 2. Build under test: kit r36 (replaces §0's table for session 2)

Placeholders are filled by Tom's final build note before this file is committed. The lead re-computes every hash
with `shasum -a 256` before the push. **Do not commit this file with a placeholder left.**

| part | file | sha256 |
|---|---|---|
| APK | `vollie-c3-probe-r36.apk`, 16,624,482 B, package `com.insaneutils.vollie.c2`, versionCode 36, same debug certificate as r35 (`c49dde85…1716`) | `5d852dea39e691ea10850d3104ae232fbde56e5439eb929a3c895778fbc36bda` |
| BUILD file | `BUILD-r36.txt` | `36b7b34fb1d294a2755969748baac31c4a3d549c419d0bfa91625ad632813887` |
| sums | `SHA256SUMS` | `275521a6a0ab9eba958e3bc57dd9ade3f649bd3debdfffe36507eb9d11ca537a` |
| runtime probe | `patches/llama.cpp-c3/0008-x4f-fix.patch` (on 0007) | `0ac3645957506060cb82e017a4e35068e64144ee846ab3b3b63c09100dd56ba8` |
| app | `patches/app-c3/0006-kit-r36-probe.patch` (on 0005) | `8397bcedf6153596bdfcb28950e59acc8c9f61986e488a1d2dc3e650a26a87fd` |
| child process source | `tools/c3_x4child.c` (unchanged since r35, commit e961ecf) | `d75cf8ca58252f85fdc55667395e297fd738d8fbb4f92656d339eb2edae38469` |
| libllama.so | in the APK | `610c0610281a4aac017e5894c742af0682adbec557071b14f6bf3e98d43718ac` |
| libggml-cpu-android_armv8.2_2.so | in the APK; must be byte-identical to r33 to r35 (`46e0735d…9694`) | `46e0735d2824ba6f3a156e219e058a701f6024fdbecbb2ef937b3127115a9694` (byte-identical) |
| kit commit | the commit that adds the r36 sources | `4b4afb0` |
| build note | `campaigns/03-dense15-5pct/reports/memo-12-work/tom-r36-build.md` | – |

- **Scope of r36 under this note.** Only the R75 blocks below. If kit r36 also carries other blocks (for example the
  x5 cells of Bonnie r18's kit r36 / R76 draft), they are **not** run under this note. They need their own
  pre-registration. The R75 button must run only the blocks of §3. *Tom: state the button name and the block list
  it runs.*
- **Numerics:** none changed. The CPU module ships byte-identical, so 0003's knob-unset identity carries over, as in
  R75 §0.

## 3. Blocks of session 2 (replaces §2's run order for the re-run)

Run order: **x4i, x4f, x4c, x4d, x4e, x4m, x4s, x4r** (R75's order without x4a and x4b), the r16 gate before each.

| block | in session 2 | scored |
|---|---|---|
| `x4i` | **runs**: the fresh kernel compile and load that every later block uses, the identity at 2816 rows, and the `x4 ext` / `x4 dev` / `x4 recq` / `x4 kai` lines | V1 to V4 and the x4i inputs are **not re-scored**; session 1 holds them. Session 2's x4i is a build step. Its lines are compared with session 1's and any difference is reported (descriptive). An identity `ok=0` here closes the general kernel for session 2 (R74 rule 2) |
| `x4a` | **not repeated** (complete in session 1) | session 1's A1 to A7 stand |
| `x4b` | **not repeated** as a block. Kit r36 does not carry it (Tom, 2026-11-24): the drift reading is instead x4f's plain arm and its four `rtargs` cells (cpu 4, spin) against session 1's x4b cpu 4 spin cell; the 25 % flag below applies to that pair. (Not built:) one 0-3 `spin` anchor cell (`x4 rt rep=2 wait=spin lcpu=0-3`) at the start of x4f | B1 to B8 and V13, V14 are not re-scored. The anchor is descriptive: it ties session 2 to session 1's 188.6 µs. If it differs from 188.6 by more than 25 %, every rule that mixes the two sessions (§4) carries a "session drift" flag. The flag is descriptive; the rules still apply as written |
| `x4f` | **runs, with the harness fix of §4** | V15, F1, F2, design A |
| `x4c`, `x4d`, `x4e`, `x4m`, `x4s`, `x4r` | run as built in r35 (Tom states any change; none is intended beyond the deadline of §4) | every cell and rule of R75 §4 and §5 that reads them |

**Phone time** (*estimated*): R75 §2 gives 3.6 to 7 min of blocks for all ten. Without x4a and x4b (21.4 s
*device* in session 1) that is about 3.2 to 6.6 min, plus eight gates of up to 180 s: about 3.5 to 31 min. The
deadline adds at most about 30 s to x4f and about 65 s to x4s (*estimated*, Tom). The app's per-block watchdog (600 s, below) bounds the run at 8 × (180 s gate + 600 s block) = 104 min.

## 4. The x4f fix: a probe-harness change, not a change of what is predicted

- After `clSetUserEventStatus` releases the gate, the harness calls `clFlush` on the same queue before the wait.
- Every wait loop in the parent (`spin_ev` and any other poll Tom names) has a **2 s deadline**. A wait that reaches
  it stops. It is counted on the cell's line (Tom's field name, called `deadline` here), and the cell is flagged.
- What the harness does after a deadline (end the cell, or go on with the next iteration; release and re-create the
  queue or not) is Tom's choice and is written in `tom-r36-build.md`. The rules of §5 cover both.
- **Effect on the measurement.** In `gate` mode, `set_to_start` now includes the cost of that `clFlush`. A
  launch-ahead on this driver needs the flush to start at all, so the flush is part of the mechanism being priced.
  The bands of F1 and V15 were written without it in mind. They are **scored as written**, and the result names the
  flush beside them.
- `plain` mode and `rtargs` are unchanged except for the deadline.

**Unchanged:** every prediction of R75 §4 (events, cells, centres, bands), every decision rule of §5.1 to §5.4,
the claim rule (§6), the meanings (§7), the definitions and the voids of §5.5 (APK hash now against
`r36/SHA256SUMS`). No band, budget or threshold moves.

**Inputs from two sessions.** Some rules read one input from session 1 and one from session 2:
- design A's kill line (x4f gate against x4b's cpu 4 `spin` `queued_to_start_us`, 96.4 µs in session 1);
- §5.3's cross-check K_x4b (x4b);
- design B / B+ design K against K_meas (x4s).

They are applied as written, with each input's session named beside it.

## 5. New pre-written rules: deadlines and hangs in a parent block

These rules were written after session 1 and before session 2. They cover a case that R75 did not cover. They do not
change any band or threshold.

1. **Deadline in a cell.** A wait stopped at 2 s enters that cell's percentiles as 2,000,000 µs (censored). It is
   left out of every mean, and the line prints the count.
   - **Fewer than 10 %** of the cell's waits reached the deadline: the cell is scored as written. The flag and
     the count stand beside it.
   - **10 % or more, but under half:** percentile fields are still exact at p50 and are scored as written. Means and
     p90 fields are "not measured (deadline)".
   - **Half or more:** p50 is a lower bound (≥ 2,000,000 µs). A rule that compares p50 with a threshold below 2 s is
     scored on that bound. A band row whose upper end is below 2 s is a **miss, high**.
2. **x4f in particular.**
   - Gate waits reach the deadline and plain waits do not: the released command does not start within 2 s on this
     driver. F1 and V15 are scored by item 1. Design A is read by its own kill line, with set→start ≥ 300 µs
     satisfied by the bound.
   - Plain waits reach the deadline too (the queue itself is stuck): x4f is **void** as a harness or driver failure,
     not a design result. A is "not measured".
3. **A parent block that hangs anyway** (no `x4<blk> end`, the JSON's `running_block` names it):
   - the owner force-stops the app **when the run has not finished 110 min after its start** (above the 104 min bound set by the app's 600 s per-block watchdog, which records a stuck block as `hang: true` with `x4 hang blk=` and moves on; such a block is void and the later blocks are scored as they ran, each with an "after-hang" flag because the stuck native call stays in its thread), or earlier if the app
     plainly shows no progress, and sends the JSON as it is;
   - that block is void by §5.5 ("process killed during a block"). If it is x4r, §5.4's "killed in x4r" reading
     applies instead, as R75 wrote;
   - the blocks after it are **not measured**. They are not misses;
   - the round is reported partial again. Another re-run needs another amendment. The lead does not ask for one
     before Tom has a cause.
4. **A deadline or hang in a block that ran clean in session 1** (x4i) is reported as a fact of session 2. Session
   1's scores stand.
5. **Lead's event probabilities for this note** (*estimated*, lead only; scored by outcome, kept out of R75's §4.1
   Brier score):
   - V25: any wait in any parent block reaches the 2 s deadline, P 0.15;
   - V26: a parent block hangs despite the fix and the owner stops the app, P 0.05.

## 6. Device state and the run, for the owner (replaces §1 and §8 for session 2)

State as R75 §1:
- charger out, power saving off;
- the deep-sleeping list label chosen in the app;
- at least 30 min of rest, screen off, not in use;
- **no other probe before this one**;
- no model loaded.

Kit r36 installs over r35 (versionCode 36 > 35, same certificate; no uninstall).

Steps:
1. After this note is pushed, download `r36/vollie-c3-probe-r36.apk`, `r36/BUILD-r36.txt` and `r36/SHA256SUMS`
   from `HoeioUser/vollie-kits`. Check that the APK's sha256 is `5d852dea39e691ea10850d3104ae232fbde56e5439eb929a3c895778fbc36bda`.
2. Install it over kit r35.
3. Set the phone state above.
4. Research kit page: "Run c3 probe r36 (GPU joins, resume)". Do not use the phone until it finishes (about 3.5 to 31 min,
   *estimated*). **If it has not finished 110 min after the start, force-stop the app** (§5 item 3). A single stuck block is cut by the app after 600 s.
5. **Save c3 probe to Downloads** and send the JSON, finished or not. It goes into
   `campaigns/03-dense15-5pct/reports/device/`.
6. If the app is killed in x4r (the last block, about 4.1 GB), the earlier blocks are already saved. Send the file
   as it is.

Memory note (*device*, session 1 `hdr mem`): MemAvailable was 2.81 GB with 3.2 GB of 4.19 GB swap in use before the
run. x4r needs about 4.1 GB (*estimated*). The phone is not changed for the run (the rule is "fix it in the runtime,
never in the user's phone"). The state is recorded and reported beside x4r.
