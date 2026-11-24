# Kit r36: R75 resume after the x4f hang, build notes

Tom, 2026-11-24. R75's kit r35 hung in block x4f on the phone. The device JSON
(`reports/device/vollie-c3probe-r35-2026-11-24T1035.json`) has `running_block=x4f`, `finished=None` and x4i, x4a, x4b
complete, and the run went 2.5 h until the owner force-stopped it. Kit r36 fixes the hang (patch 0008) and runs the rest of R75.
The pre-registration is the lead's (R75 amendment 1). Not committed, not uploaded.

**Naming.** Bonnie r18 already uses "kit r36" for the R76 x5 draft. This build keeps r36 for the R75 resume kit, as briefed.
Its button runs only R75 blocks and no x5 block is in it, so the x5 kit takes the next number (r37).

## Cause

The cause is found in the code. The mechanism on Adreno is inferred, not observed.

Each `gate` iteration of x4f (r35, `patches/llama.cpp-c3/0007`) ran these steps:

1. `clCreateUserEvent`.
2. Enqueue the kernel on the profiling queue with that event as its wait list.
3. `clFlush`.
4. Spin to 1 ms.
5. `clSetUserEventStatus(CL_COMPLETE)`.
6. `spin_ev(ev)`: poll `CL_EVENT_COMMAND_EXECUTION_STATUS` until complete.

Two things together hang this loop:

- **No flush after the release.** Step 3 flushed while the command was still blocked, and nothing flushed after step 5.
  On a driver that submits a released command only at the next flush or finish, the command stays queued.
- **No deadline.** `spin_ev` in r35 had no deadline at all:
  `do { GetEventInfo(...) } while (!e && st > 0)`.

x4f runs in the parent process, so no child timeout applied, and the app had no per-block timeout. The JSON holds no x4f line
because r35 returned its lines only when the block returned.

Evidence:

- **Code reading:** the lead's reading checks out line by line.
- **Device:** the report was saved before x4f started, and x4f is the last block named.
- **Host emulation** (*host*, PoCL, arm64 Docker): an OpenCL shim (`libOpenCL.so`, ahead of the ICD loader on `LD_LIBRARY_PATH`)
  holds back every user-event release until the next `clFlush`, `clFinish` or `clWaitForEvents`. It does this by gating the
  real command on a shadow user event. Everything else passes straight through.
  - r35's libllama under the shim: x4i completes, then x4f never returns (killed by `timeout 90`, rc 124). The hang reproduces.
  - On plain PoCL, which submits on release, r35's x4f completed (r35 build notes). That fits a driver-dependent cause.
  - r36's libllama under the same shim: x4f completes, gate n=50, err=0, dl=0.
  - r36's no-flush check under the shim gave `done_without_flush=0 done_after_flush=1` on all three tries.

The device line `x4 la nf` will show whether Adreno behaves like the shim. If it prints `done_without_flush=1`, the cause on
the phone was something else, and the deadline is what keeps r36 from hanging.

## Fix: `patches/llama.cpp-c3/0008-x4f-fix.patch` (on 0007; only `src/llama-vollie-probe.cpp`)

- **x4f gate:** `clFlush(g.qp)` right after `clSetUserEventStatus`. `set_to_done` now includes that flush. It is the real cost of
  release-then-submit.
- **Deadline on every event spin:** `spin_ev` gives up after 2 s (`EV_DEADLINE`) and returns `-2001` (`EV_DL`). It reads the
  clock only every 256 polls, so the spin itself is unchanged. Its callers:
  - `wait1` (x4b and x4f `rt` / `rtargs`, spin mode);
  - the x4f gate and plain arms;
  - x4s lead joins;
  - x4s hand-off joins.
- **Other wait paths in x4a..x4r**, checked:
  - x4s feeder mode's `while (done != q) {}` gets the same 2 s deadline (err -2001). x3's copy of that loop (R74) is not
    touched.
  - rtcell's 100 profiled calls stop at the first deadline, so a stuck queue costs 2 s, not 200 s.
  - The x4e host waits already had `PK_DEADLINE` (2 s), and the kernels have iteration caps.
  - The x4d and x4e children have parent timeouts.
  - The 1 ms gate spin and every timed loop are bounded by the clock.
  - `clFinish`, `clWaitForEvents` (x4b `wfe` only, not in r36's order) and blocking read, write and map calls cannot be given a
    deadline from inside the call. They flush implicitly, so the no-submit cause does not apply to them. The app's per-block
    timeout covers them (below).
- **What a deadline in x4f does:**
  - In the gate arm it stops that arm and keeps what the arm has. It then issues one more `clFlush` and waits 2 s again; the
    result is `flush_recovered=1/0`.
  - In the plain arm it ends the loop.
  - `x4 la` gains `dl= stopped_at= flush_recovered=` after `err=`, before the tail (`stopped_at=-1` means the mode did not stop).
- **Cause check `x4 la nf`** (new, after the gate/plain arm, before `rtargs`): three gated launches released without a flush,
  each waited on for 2 s, then flushed and waited on for 2 s more. Fields:
  `try= done_without_flush= wait_us= done_after_flush= after_flush_us= err=`. Its own waits are not counted in `x4 dl`.
- **`x4 dl blk= n=`:** the number of deadlines in the block. It is printed in every parent x4 block, before `x4 therm ... when=after`.
- **Live file:** every parent x4 block also writes each line to `<dir>/x4-live-<blk>.txt` as it is printed (`fwrite` +
  `fflush` per line, outside the timed loops; children already tee'd).
- **`x4 pin`** (x4i, on the lead's request): `x4 pin cpu4=ok@4 cpu5=.. cpu6=.. cpu7=<ok@<cpu seen>|e<errno>>`, from pinning the
  x4i thread to each big cpu in turn at idle. The thread's affinity is restored afterwards.

The x3 blocks and every x4 cell's order, sizes and timing windows are unchanged.

## Per-block watchdog (app patch `patches/app-c3/0006-kit-r36-probe.patch`)

- **Timeout:** `_c3ProbeIso(blk, dir).timeout(600 s)` (`Research.c3BlockLimit`).
- **On expiry**, the block entry gets:
  - `hang: true`;
  - `lines` = the lines of `<dir>/x4-live-<blk>.txt`, plus `c3-probe: x4 hang blk=<blk> s=600 lines=<n>`.
- **Then** the report is saved as after any block, and the run goes on to the next block. A hang therefore loses that block's
  last unprinted cell, not the run.
- **Ceiling:** the stuck native call stays in its isolate thread, because Dart cannot interrupt FFI.
  - If it is stuck in a sleeping driver call, later blocks run beside it.
  - If it were spinning, it would load one CPU for the rest of the run. The deadlines above make that unlikely.
  - Blocks after a `hang` block should be read with that in mind. The parser lists the hang as an error.
- **Other app changes:**
  - version 1.36.0+36, kit r36;
  - button "Run c3 probe r36 (GPU joins, resume)";
  - order x4i, x4f, x4c, x4d, x4e, x4m, x4s, x4r;
  - the design string names this file and the 2 s / 600 s limits;
  - the same r16 gate before every block (up to 180 s), and the report is saved before and after every block.

**x4b anchor: not added.** x4f's `plain` arm and its four `rtargs` cells are cpu-4 spin-wait round trips at 2816 rows on the
same kernel, measured in the r36 session itself. They are the within-session anchor for drift against R75's x4b
`lcpu=4 wait=spin` cell. A separate x4b rep would repeat them.

## The lead's second question: R75's x4b "cpu 7" cells ran on cpu 4

R75 (*device*): the three `lcpu=7` cells read `lplace=n/n@4`, `ramp_ok=0`, `ramp_s=2.50`, `thr=none`. Policy 7 also sampled
its idle clock in those cells (third `mhz_range` value, for example `..2419/844` against 2841 max).

- **Reading (inferred, not verified):** `sched_setaffinity({7})` failed and the failure went unseen. `prep()` ignores
  `pin_cpus`'s return value.
- **Why the thread stayed on cpu 4:** after a failed pin it kept its restored mask, and the scheduler left the spinning thread
  on cpu 4, where the previous cell had pinned it.
- **Likely reason the pin fails:** cpu 7 paused by core_ctl while the system is idle, or cpu 7 missing from the app's cpuset at
  that moment.
- **Against a plain ban:** in R74 the ggml pool workers did run pinned on cpu 7 under load (*device*, `thr` labels 4 to 7).
- **No pin code is changed:** r36 runs no cpu-7 launcher (L7 is used only by x4b). x4f launches from cpu 4. x4m and x4s use
  the pool on 4-7, whose per-thread `thr` labels show any misplacement (M). x4r uses the arena threads as in R74.
- **x4i's `x4 pin` line answers it on the phone:** `cpu7=e22` means the pin is refused at idle.

## Contents and hashes

| item | file | sha256 |
|---|---|---|
| runtime patch | `patches/llama.cpp-c3/0008-x4f-fix.patch` (132 lines) | `0ac3645957506060cb82e017a4e35068e64144ee846ab3b3b63c09100dd56ba8` |
| app patch | `patches/app-c3/0006-kit-r36-probe.patch` (112 lines, on 0005) | `8397bcedf6153596bdfcb28950e59acc8c9f61986e488a1d2dc3e650a26a87fd` |
| parser | `tools/c3_probe_parse.py` (dl, nf, hang, pin; `x4r36selftest`) | (tracked file) |
| APK | `lab/dist-c3/r36/vollie-c3-probe-r36.apk`, 16,624,482 B, versionCode 36, versionName 1.36.0, package `com.insaneutils.vollie.c2`, certificate SHA-256 `c49dde85…1716` (same as r35) | `5d852dea39e691ea10850d3104ae232fbde56e5439eb929a3c895778fbc36bda` |
| build record | `lab/dist-c3/r36/BUILD-r36.txt` | `36b7b34fb1d294a2755969748baac31c4a3d549c419d0bfa91625ad632813887` |
| sums | `lab/dist-c3/r36/SHA256SUMS` | `275521a6a0ab9eba958e3bc57dd9ade3f649bd3debdfffe36507eb9d11ca537a` |

Libraries in the APK, against r35:
- `libllama.so` (c3 0001, 0002, 0003, 0005, 0007, 0008) changed: `610c0610281a4aac017e5894c742af0682adbec557071b14f6bf3e98d43718ac`.
- `libapp.so` (Dart) changed.
- Every other library is byte-identical to r35:
  - `libggml-cpu-android_armv8.2_2.so` `46e0735d…9694` (= r33 = r34 = r35);
  - `libvollie_x4c.so` `fb61545a…315f`;
  - `libvollie_clc.so`.

Numerics: none. The CPU module is byte-identical, and the probe is outside every inference path and off unless `VOLLIE_PROBE`
is set. A perplexity check therefore has nothing to compare; per-chunk identity follows from the identical module.

Build: tree `lab/llama.cpp-c3r36`, a clone of 620d2225c with patches/llama.cpp, patches/llama.cpp-c1-olmoe and c3 0001, 0002,
0003, 0005, 0007 applied, then 0008. After 0007 it equals the r35 tree file for file (`cmp`). The commands:

```
sh android/build_cpu.sh lab/llama.cpp-c3r36 <libs>        # then the r35 APK's CPU module copied over <libs>
LIBS_FROM=<libs> APP_SERIES="<app-c2 0001-0012> patches/app-c3" sh tools/c2_build_kit.sh lab/llama.cpp-c3r36 lab/app-c3 patches/llama.cpp patches/llama.cpp-c1-olmoe <dir with c3 0001 0002 0003 0005 0007 0008>
```

The APK was then copied to a plain name.

## Checks (all *host*; host timings mean nothing for the phone)

- 0001, 0002, 0003, 0005, 0007, 0008 applied to an empty tree reproduce the r36 probe file exactly (`cmp`).
- **Warnings:** in new code, none.
  - macOS `clang++ -fsyntax-only -Wall -Wextra -Wcast-qual` gives only 0001's two.
  - The Android build (NDK 28.2) gives only those two.
  - Linux gcc gives only old-code warnings (pragma, indentation at line 224).
- **Functional run, PoCL, `--cpus=4`, all ten blocks** (x4i, x4f, x4c, x4d, x4e, x4m, x4s, x4r, x4a, x4b), 3 min 10 s:
  - every cell err=0; every `x4 dl` n=0;
  - children rc=0;
  - the knob unset gives -1.
  - x4f's live file equals the block's x4 lines.
  - `x4 pin cpu4=ok@4 cpu5=ok@5 cpu6=ok@6 cpu7=ok@7`.
  - The no-flush check on plain PoCL gave `done_without_flush=1` (PoCL submits on release).
- **Shim, no-submit-on-release** (x4i, x4f, x4s):
  - x4f gate n=50, err=0, dl=0; nf `done_without_flush=0 done_after_flush=1` ×3;
  - x4s lead, feeder and hand-off: all err=0, `x4 dl` n=0 (they flush before they spin);
  - r35 under the same shim: x4f hangs (above).
  - A first shim version matched recycled event handles to old entries. Fixed (newest first) before these runs.
- **Shim with `SHIM_NEVER=1`** (a gated command never runs, even after a flush; the worst case):
  - x4f returned in 29.9 s;
  - gate `dl=1 err=-2001`; plain `dl=1` (the in-order queue was blocked behind it);
  - nf `0/0` ×3; `rtargs` err=-2001 with the profiled loop cut short;
  - `x4 dl blk=x4f n=7`; queue teardown returned.
- `python3 -I tools/c3_probe_parse.py --selftest` prints `selftest r72 ok`, `x3 ok`, `x4 ok`, `r36 ok`, `selftest ok`. The r35
  device JSON gives output byte-identical to the committed parser's.
- **Not checked:**
  - on any phone;
  - the app's timeout path, which is Dart and was not run on a device or emulator. It compiled in the release build.

Host files (scratch, not kept): the shim source and the runner were in the session scratchpad.

## Phone time and memory (*estimated*)

| block | estimate |
|---|---|
| x4i | about 1 s (R75 *device*: 1.0 s) + `x4 pin` 4 ms |
| x4f | 12 to 20 s: r35's 6 to 10 s + the nf check (about 6 s if Adreno holds the release: 3 × (2 s + flush); about 0 if it does not). Worst case with a stuck queue: about 30 s more (gate 2 + 2, plain 2, nf 12, rtargs 4 × 2, *host* analogue 29.9 s) |
| x4c | 25 to 40 s |
| x4d | 8 to 16 s (timeout 30) |
| x4e | 5 to 12 s (worst 30) |
| x4m | 57 to 113 s |
| x4s | 70 to 145 s; worst with deadlines on every join about 2 s per arm more (32 arms, about +65 s) |
| x4r | 25 to 35 s |

- **Normal case:** about 3.5 to 6.5 min of blocks, plus eight r16 gates of up to 180 s each: total about 4 to 30 min.
- **Hard ceiling:** 8 × (180 s gate + 600 s block limit) = 104 min, reached only if every block hung.

Memory peaks, as r35 (nothing in 0008 allocates):
- x4c about 0.8 GB;
- x4m about 1.2 GB;
- x4s about 0.6 GB;
- x4r about 4.1 GB. A low-memory kill is likely there. It is last, and the report is saved before it.

## Values for R75 amendment 1 (`<TOM: …>` placeholders)

- Runtime patch `patches/llama.cpp-c3/0008-x4f-fix.patch`, sha256 `0ac3645957506060cb82e017a4e35068e64144ee846ab3b3b63c09100dd56ba8`.
- App patch `patches/app-c3/0006-kit-r36-probe.patch`, sha256 `8397bcedf6153596bdfcb28950e59acc8c9f61986e488a1d2dc3e650a26a87fd`.
- APK `vollie-c3-probe-r36.apk` (1.36.0+36), sha256 `5d852dea39e691ea10850d3104ae232fbde56e5439eb929a3c895778fbc36bda`; libllama `610c0610…18ac`.
- Button: "Run c3 probe r36 (GPU joins, resume)".
- Deadlines:
  - 2 s per event wait (err -2001);
  - 600 s per block (app);
  - worst case x4f about +30 s, x4s about +65 s;
  - absolute ceiling 104 min for the whole run.
