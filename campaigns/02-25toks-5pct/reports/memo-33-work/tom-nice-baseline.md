# Why every DartWorker read nice -10 in R68 session B (Tom, 2026-11-19)

Data: `campaigns/02-25toks-5pct/reports/device/r28_2026-11-18.json` (*device*, kit r28, 72 runs, one app process).
Code read: `patches/llama.cpp-c1-olmoe/0010-c2-pool.patch`, `0012-c2-touch-nice.patch`, `patches/app-c2/0012-kit-r28.patch`
(and `lab/app-c2` for the Flutter setup). No fix made, nothing built, nothing run on the phone.

## Answer in one paragraph

Nothing in the runtime or the kit set -10. The value was already there before any set: every DartWorker thread in the app
process, in every one of the 72 runs (warm-ups and VOLLIE_NICE-unset runs included), reads -10, and threads created fresh
in a run read -10 too. On Linux a new thread takes its nice value from the thread that creates it. The Dart VM creates
its DartWorker threads from Dart threads, and the first of those is the Android main thread, which runs the Flutter UI
isolate (Flutter merges the platform and UI threads on Android by default since 3.29; the host has Flutter 3.41.2 and
`lab/app-c2` does not opt out). For the top (foreground) app, Android's ActivityManager sets the main thread to
`THREAD_PRIORITY_TOP_APP_BOOST` = -10. So -10 is the baseline every DartWorker inherits, the pool's workers included,
because they are created by a DartWorker. When VOLLIE_NICE=-10 was set, 0012 compared and found no difference, made no
set (`s0`), and saved nothing to restore (`restored=0/0`). That is correct behaviour for the code as written. The nice
factor in session B therefore has no contrast: the N and n cells ran at the same priority.

## The four questions

1. **Is "off restores" only for a set made in the same process?** Yes, by design: 0012 restores only a value it changed
   itself (`saved` is set only after a successful `setpriority`). That is not the cause here. No set ever happened
   (`s0:f0` on every thread of every c2-nice line), so there was nothing to restore. The off runs were not left at -10
   by an earlier on run. They started at -10 (see pos 1 and 2 below).
2. **Is the pool persistent, so threads keep -10?** The pool is persistent (0010: one static `ggml_c2_pool`, freed only
   when a larger thread count is asked). Its two workers are tids 21270 and 21271 in all 72 runs, both models. But
   persistence is not the cause. The workers read -10 already in the first warm-up (pos 1), where no run had set
   VOLLIE_NICE, and the per-run callers, which are new threads in almost every run, read -10 as well.
3. **Does the app start at -10?** Yes, and this is the cause. In detail: Android gives the top app's main thread (tid =
   pid) and its RenderThread nice -10 (`ProcessList.THREAD_PRIORITY_TOP_APP_BOOST = -10`, applied in `OomAdjuster` when
   the process's sched group becomes TOP_APP and the FIFO UI option is off; source: AOSP
   `services/core/java/com/android/server/am/ProcessList.java` and `OomAdjuster.java`, from memory, not checked online in
   this session). The value is not `THREAD_PRIORITY_URGENT_DISPLAY` (-8) or `THREAD_PRIORITY_DISPLAY` (-4). Flutter's own
   thread setter (android_shell_holder.cc) sets the engine's separate ui/raster/io threads to -1 / -5 / +10 (from
   memory). If the UI thread were a separate `1.ui` thread, DartWorkers would most likely inherit -1, not -10. -10
   matches the main thread, which fits the merged thread. The kit's Kotlin and Dart code makes no priority call (grep of
   `lab/app-c2/lib` and `android/app/src` for setThreadPriority / setpriority / THREAD_PRIORITY: none). llamadart
   starts its inference isolate with `Isolate.spawn` (`llama_cpp_backend.dart:340`), which puts it on a DartWorker.
4. **Is the read-back reading something else?** No. Two independent readings agree. 0012's `getpriority(PRIO_PROCESS, tid)`
   on Linux returns the one thread's nice (bionic converts the kernel's 20-nice back). The kit's `workerThreads()` reads
   `/proc/self/task/TID/stat` field 19 (nice) and field 18 (priority = 10 = 20 + nice for CFS, so this agrees). The tids
   in c2-nice (w0 caller, w1/w2 workers) are the same tids that `workers_after` lists at -10 in the same run.

## Evidence from the JSON (*device*)

- Pool workers: `c2-pool` w1 = 21270, w2 = 21271 in all 72 runs (both models, all blocks). Created in pos 1
  (olmoe_off_wu.0) by caller 20209. Their tids are above 20209 and they exist from the first pool graph on.
- Caller w0: a new tid in almost every run (20209 in pos 1-2, then 23113, 24019, ... 13359 in pos 72; it repeats only
  inside a warm-up pair and the run after it). Each new caller reads -10 at its first graph without a set (`s0`).
- `workers_after` lists 5 to 7 DartWorker threads per run. **Every** entry in all 72 runs is nice -10, priority 10,
  policy 0 (SCHED_OTHER), cgroup cpuset/cpu `top-app`. That includes pos 1 and 2 (warm-ups, VOLLIE_NICE unset) and the
  46 runs without VOLLIE_NICE.
- The 0012 `c2-nice` line is printed only when VOLLIE_NICE is set or a restore happened, so it appears in the 26 N runs
  (pos 27, 29, 33, 34, 36, 38, 39, 40, 46, 48, 49, 50, 53, 55, 59, 60, 63-72). Every one reads
  `restored=0/0 w0=<caller>:-10:s0:f0:e0 w1=21270:-10:s0:f0:e0 w2=21271:-10:s0:f0:e0`.
- No thread in any run reads a nice other than -10, so no run of session B measured priority 0.

What the JSON cannot show: the main thread's own nice (the kit lists DartWorker threads only). The TOP_APP_BOOST
explanation is the only one that fits all of the above, but it is an inference, not a reading.

## Consequences for scoring (for the lead; no verdict here)

- Session B's nice factor is void for lack of contrast (N = n = -10). Under ebony's proposed B3 reading ("drop a quad when
  an n run has a `workers_after` nice of -10 on a pool tid") every quad would be dropped.
- Session A runs the same APK in the same foreground app. Expect the same: -10 everywhere, `s0`, `restored=0/0`, so again
  no contrast (*estimated*, by the same mechanism). Unless the prereg or an amendment already says how a no-contrast
  result is scored, the lead should settle that before session A is scored.
- The finding generalises across models: Granite and OLMoE both ran at -10, in one process with one pool.
- Side note: after the app leaves the top (screen off, background), Android puts the main thread back to 0. DartWorkers
  created before that keep -10, because inheritance is a snapshot taken when the thread is created. A thread's nice
  can therefore depend on when in the app's life it was created.

## Fix for a later kit (described only, not made)

The runtime already accepts any VOLLIE_NICE value and sets it when the thread's value differs. The smallest fix is in the
design, not the code:

1. **Make "off" an explicit value.** The n arm sets `VOLLIE_NICE=0` and the N arm sets `VOLLIE_NICE=-10`. Then both arms are
   sets from the same -10 baseline: n raises the nice to 0, which is always allowed, and N leaves it at -10. The
   restore-by-tid path is no longer used by the design. Read it as "Android's default for this app (-10) versus a plain
   priority (0)": it measures what -10 is worth, not whether going above the baseline helps.
2. **If the question is "does a priority above the baseline help",** the on arm must be below -10 (for example -15 or
   -19). Whether an app may lower its own thread's nice that far depends on RLIMIT_NICE and the SELinux/cgroup policy.
   Not known for this phone. 0012 already records the failure (`f`, `e`) and stops asking, so one kit run would answer
   it. Note that THREAD_PRIORITY_URGENT_AUDIO (-19) is a value apps can ask for through the Android API on their own
   threads, so -19 is worth a try (*not verified*).
3. **Record the baseline.** The kit should log, once per run before the first graph, the main thread's nice
   (`/proc/self/task/<pid>/stat`, field 19) and one DartWorker's nice, and extend `workerThreads()` (or a sibling call)
   to the main thread. The baseline is then a reading, not an inference. The c2-nice line could also print `orig`
   (the value seen before any set) so that "already at want" shows as such and is not mistaken for a successful set.
4. The prereg for any later nice round must state the baseline it expects (-10 from TOP_APP_BOOST) and the pass rule
   must compare against that baseline.
