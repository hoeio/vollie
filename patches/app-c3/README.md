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
