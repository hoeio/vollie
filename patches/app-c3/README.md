# Campaign 3 app patches (research kit)

Apply after `../app-c2/0001` to `0012` (app 2.3.0+45 + kit r28), in order.

## 0001-kit-r29-probe.patch

Kit r29 (version 1.29.0+29, versionCode 29): same package `com.insaneutils.vollie.c2` and debug key, so it installs over
r28 and keeps the downloaded models. The r28 run list is unchanged. New: the button "Run c3 probe r29" (no model; the
loaded model is unloaded first; requires the session label and the charger out). It runs the runtime's
`llama_vollie_probe` (`patches/llama.cpp-c3/0001-probe.patch`) for the blocks p2, x1, x2, p1 in that order, each after the
r16 cool-down gate (headroom below 0.65, read every 5 s, at most 90 s), from an isolate, with `VOLLIE_PROBE=1` set only
around the call. P1's scratch file goes to the internal kit folder (`filesDir/kit-models`) and is deleted by the probe.
Report (`kind: c3_probe`): power state at start, before each block and at the end, battery temperatures, the gate
series, and every `c3-probe:` line; saved in the app and next to the models, and "Save c3 probe to Downloads"
(`vollie-c3probe-r29-<time>.json`). Parser: `python3 -I tools/c3_probe_parse.py <json>`.

Build: `APP_SERIES="patches/app-c2 patches/app-c3" tools/c2_build_kit.sh lab/llama.cpp-c3k lab/app-c3 patches/llama.cpp
patches/llama.cpp-c1-olmoe patches/llama.cpp-c3`, then the APK is copied to a plain name without '+'.

## 0002-kit-r32-probe.patch

Kit r32 (version 1.32.0+32, versionCode 32), applied after 0001; installs over r29. The button "Run c3 probe r32" (no
model needed) runs the blocks a, l1, l4, l23, l5, l6, l7, x2, anon34, anon42 in that order with `patches/llama.cpp-c3/0003`.
The cool-down gate before each block waits at most 180 s (r29: 90 s) and records `max_s` and whether it timed out. The
report is saved before each block (`running_block`) and after it, so a killed run (anon arms) keeps the finished blocks.
The probe's output buffer is 1 MB. File name `vollie-c3probe-r32-<time>.json`.

Build: `APP_SERIES="$(cd patches/app-c2 && ls 00[01][0-9]-kit-r[12][0-9].patch | sed 's#^#patches/app-c2/#' | tr '\n' ' ')patches/app-c3"
tools/c2_build_kit.sh lab/llama.cpp-c3r32 lab/app-c3 patches/llama.cpp patches/llama.cpp-c1-olmoe patches/llama.cpp-c3`
(app-c2 0001 to 0012 by name: 0013 and 0014 are not in the c3 app copy), then the APK is copied to a plain name without '+'.

## D1 harness (memo-05 §4): design only, no patch yet

D1 loads a real Q4_0 7B body with no repack, maps it executable and touches it, and records residency and kills
over 20 runs per model. It needs no runtime change:

- **No repack:** `--no-repack` is `llama_model_params.use_extra_bufts = false`. The kit already sets it per model
  (`KitModel.repack: false` -> `useExtraBuffers: false`). This is Y1c, a declared numerics change (memo-05 §3.4).
  Its full `-b 512` and `-ub 1` runs go into the round's budget before any device use.
- **Exec map and touch:** use `VOLLIE_EXEC_MAP=1` (c1-olmoe 0011) with the files in internal storage
  (`kitStorageStep`), and `VOLLIE_TOUCH=1`, `VOLLIE_TOUCH_S=12` (0012). These are the existing `execOn` and `touchOn`.
- **Residency:** use the kit's existing records in every run. These are the file's mincore before and after
  (`res_after`), smaps by inode, and the `c2-io` read bytes and major faults per prompt. Add a
  `llama_vollie_probe("hdr")` call (0002) before and after each run. Its PSI totals and zram `mm_stat` give the
  pressure each run caused.
- **Hold off:** the hold refetches lost pages, so it would hide the loss D1 measures.

What a `patches/app-c3/0002-kit-d1.patch` would add. It is off by default: a new button, and the r29 run list is
unchanged.

1. `kitModels` entries for Mistral-7B-Instruct-v0.3, Qwen2.5-7B-Instruct and Yi-1.5-6B, each with `repack: false`.
2. One block per model:
   - load;
   - 2 unscored warm-ups;
   - 20 gated `official32` runs (r16 gate, 3 threads, pin + exec + touch, kv clear, no hold);
   - `hdr` lines before and after every run.
3. A kill record:
   - Before each run, the kit stores `{run id, time}` under an in-progress key. The report is already saved after
     every row.
   - At the next kit start, a stored report without `finished` gets `DeviceProfile.lastExit()` (ApplicationExitInfo:
     reason, importance, pss, rss, time) and `killed_in`.

It is not written yet, for three reasons:

- The campaign-3 served files are not chosen. None of them is hosted with a sha256 in `HoeioUser/vollie2`, and the
  kit needs a URL and a sha256 per file.
- The lab's Qwen2.5 Q4_0 is a two-part split. `KitModel` takes one file.
- D1 runs only if R69 holds 3.4 GB.
