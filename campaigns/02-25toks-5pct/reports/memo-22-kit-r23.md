# Memo 22: research kit r23, OLMoE's head out of KleidiAI, hold in every gated run, threads 3 against 4

Agent model: claude-opus-5-5

Date 2026-11-16. Author: Tom (code). **Nothing here ran on a phone, and no phone speed is stated.** The APK was built
and inspected on the laptop. Labels: *host* = Apple M4 Pro, relative only; *estimated* = arithmetic; *device* = quoted
from an existing report, with the report named. Kit r23 starts from kit r22 ([memo-20](memo-20-kit-r22.md)); this
memo lists what changed. R60 = kit r22's day (`reports/device/r22_2026-11-15.json`); R61 = the host PPL on a KleidiAI
build.

## 1. Runtime: campaign 1's runtime + `patches/llama.cpp-c1-olmoe/0001` to `0006`

r22's runtime plus one patch:

| patch | what | env |
|---|---|---|
| `0006-c2-head2-kleidiai.patch` | at model load, KleidiAI refuses a Q6_K `output.weight`, so the head goes to `CPU_REPACK` (q6_K_8x4 / 8x8), the layout 0004's `VOLLIE_HEAD2` runs on; the `c2-head-run` line gains `c2_declined=` and `c2_why=` (the last decline reason: `<code>:<buffer of output.weight>`) | `VOLLIE_HEAD2_REPACK` (load time) |

Why: in R60 the `armv8.2_2` module is built with KleidiAI on (`android/build_cpu.sh`, as in campaign 1), KleidiAI took
OLMoE's Q6_K head into its own int8 layout, and 0004 declined every head op, so no `c2-head-run` line was printed.
Granite's head is tied and is not taken. With the knob unset 0006 is identical (per-chunk, 4 chunks, `-b 512`, `-ub 32`,
`-ub 1`, builds with and without KleidiAI, *host*, 2026-11-15). With the knob set, the off arm's OLMoE head
goes back to Q6_K numerics. That is a declared change of the off arm, and R61 scored it (*host*):
OLMoE off + `VOLLIE_HEAD2_REPACK=1` 11.7132 / 11.7159 (`-b 512` / `-ub 1`), num 12.0927 / 12.0976.

The knob is read inside KleidiAI's `supports_op`. That function is consulted only for tensors in KleidiAI's buffer, and
the weight buffer is chosen at load. After the load the head sits in `CPU_REPACK`, whose own `supports_op` answers, so
the app sets the knob for the load only and unsets it afterwards. The host runs of R61 kept it set for the whole
process; the two are equivalent by this reading of the code, not by a run.

### 1.1 Android build

`tools/c2_build_kit.sh lab/llama.cpp-c1r23 lab/app-c2 patches/llama.cpp patches/llama.cpp-c1-olmoe`. The tree is a fresh
clone at 620d2225c with `patches/llama.cpp/0001` to `0003` and `patches/llama.cpp-c1-olmoe/0001` to `0006` applied, and
the script checks it file by file against those nine patches. Campaign 1's `android/build_cpu.sh` is unchanged and the
flags are r22's (memo-20 §1.1), KleidiAI on. The build exits 0. The only warnings come from vendored code, plus the
`ccache` notice.

| | bytes | sha256 |
|---|---|---|
| `vollie-c2-1.23.0+23-research-r23.apk` | 17,212,039 | `2cf9d2ad193b77458f003d6ec9c9de5b6d0d6464c4597a2dd5e32512bcefd90c` |
| `libggml-cpu-android_armv8.2_2.so` | 1,761,328 | `19f36b6e8c2dcaa523b6d2bdfbc00f757e68bebd4ef2ec72e54a49de00cbd8f1` |
| `libllama.so` | 4,285,208 | `edcfadf16df981c55db44e6dee9b4b0cbe8b39a941a9ce4abec365f31768a534` |

- The script checked that the APK carries exactly these two libraries and the build record: `assets/kit_build.json`,
  which lists the nine series patches and app patches `0001` to `0007` with their sha256. The full text is in
  `lab/dist-c2/BUILD-r23.txt`.
- The CPU module carries the string `VOLLIE_HEAD2_REPACK` and the extended `c2-head-run` format.
- `libllama.so` has no source change against r22: `src/` and `include/` are identical between the r22 and r23 trees,
  and 0006 touches only `ggml-cpu.c` and `kleidiai.cpp`. It has the same size and the same section sizes, and its
  `strings` are equal once the tree name (`c1r22` / `c1r23`) is masked. Its bytes still differ: 147,950 bytes in 4,648
  ranges spread over the code. So the hash is new. This build is not byte-reproducible across tree paths; the cause
  was not traced further.

### 1.2 Host checks (*host*)

- 0006 identity with the knob unset and the R61 values: see §1. The full identity sets were not repeated for this kit,
  as the brief asked.
- App: `flutter analyze lib test` reports no issues; `flutter test` reports all 35 tests passed. Two of them are new:
  - the extended `c2-head-run` line is tagged;
  - the r23 run list.
- Parser: `--selftest` passes, with a new synthetic r23 case (two models, threads effect, declines, a missing load
  env). The R60 report `reports/device/r22_2026-11-15.json` still parses with r22's tables.

## 2. Hold in every gated run, cheaper

### 2.1 What R60 showed (*device*)

- Every gate wait ran the full 90 s.
- Without hold, a wait took 714 to 1,032 MB of the file (median per model and arm).
- Hold raised the MB resident before the run by +842 (OLMoE) and +1,165 (Granite), and tok/s by +4.72 ± 1.70 and
  +5.28 ± 0.98.
- The cost was 5.1 s CPU per wait (median of 24 gated runs with hold, range 3.9 to 8.0 s). Each wait ran 20 passes,
  and 3,374 MB were advised in total, against about 21 MB of blocks actually read.
- A fit over those 24 runs (*estimated* from *device* values): CPU ms ≈ 180 × passes + 0.45 × advised MB, or
  205 × passes + 0.26 × advised MB + 1.3 × blocks-read MB, with a residual of about 0.5 s. So most of the cost was
  the passes: one `mincore` of a 3.1 GB mapping is about 0.2 s of kernel CPU.
- The Dart loop over the vectors is about 5 ms per pass (AOT build on the host, 750k pages), so the cost is the
  kernel's, not the app's.
- r22 also polled `mincore` every 200 ms for up to 5 s after the gate, outside `cpu_us`. So r22's real cost was
  higher than reported.
- Advice was repeated: the same lost pages were advised again at every 5 s poll. The read-back counter stayed far
  below the MB advised, so the readahead was apparently refused or deferred under memory pressure, and pages were
  advised again.

### 2.2 The r23 rule

"Wanted" is unchanged: the pages resident at the end of the previous run of that model. A pass is one `mincore`, then
`posix_fadvise(WILLNEED)` on each run of wanted pages that are gone (r22's code).

1. **In the wait:** at most one pass, at the first gate poll at least 45 s after the wait began (at about 45 s in a
   90 s wait). There is none if the gate passes earlier. It reads back what the first half of the wait took, so
   that the final read has less left to do.
2. **After the gate:** a pass, then one pass every 1 s, until one of these:
   - at least 95 % of the wanted pages are resident at a pass (`back`);
   - a pass gained less than 1 % of them over the one before (`stall`);
   - 4 passes ran (`rounds`);
   - 8 s passed (`time`).
3. There are no other `mincore` calls: the 200 ms polls are gone, so every `mincore` of the hold is in `cpu_us`.

Hold is in every gated run of both arms: the 32 scored runs and the 8 profiled runs. The warm-ups do not hold. This
makes hold part of the protocol, not a factor.

### 2.3 Expected cost (*estimated*, from R60's fit)

- **Passes per wait:** 1 in the wait plus 2 to 3 after the gate is typical. The first final pass advises and the
  second sees the pages back or stalled. The maximum is 1 + 4 = 5.
- **Advised MB per wait:** the first half's loss (about 350 to 500 MB if R60's 700 to 1,000 MB per 90 s is linear),
  then the second half's, then what is still missing at the second final pass. About 700 to 1,500 MB.
- **CPU:**
  - typical, 3 passes: 3 × 0.18 to 0.205 s + 0.26 to 0.45 ms × 700 to 1,000 MB ≈ 0.55 to 0.62 + 0.18 to 0.45 ≈ **0.75 to 1.05 s**;
  - worst, 5 passes and 1,500 MB: 5 × 0.205 + 0.45 × 1,500 / 1,000 ≈ **1.7 s**;
  - r22: 5.1 s.
- **Against the 1 s target:** the target is met only in the typical case and not guaranteed. The pass cost (about
  0.2 s per `mincore` of 3.1 GB) sets a floor of about 0.6 s at 3 passes. Going lower needs fewer passes (for example,
  no in-wait pass) or a cheaper residency check than a whole-file `mincore`.
- **`cpu_us`** is `getrusage(RUSAGE_SELF)` around each pass, the whole process. It is an upper bound on the hold's own
  CPU and includes the UI.
- **Wall time after the gate:** 1 to 3 s typical, at most 8 s. Over 40 runs that is about 1 to 2 min, at most 5.3 min.

### 2.4 What the report shows per gated run

`hold_stats` keeps r22's fields: `passes`, `lost_mb_max`, `advised_mb`, `advise_calls`, `advise_fails`, `pass_ms`,
`cpu_us`, `inblock`, `final_ms`, `back_mb`. It adds:

- `wait_passes`, `final_passes`, `back_permille` (wanted pages resident at the last pass, ‰), `stop`, and `mincore_ms`
  (the `mincore` part of `pass_ms`).
- **`advised_mb` is now readable two ways:** it is the sum over passes (a page advised twice counts twice), and the new
  `advised_distinct_mb` counts each page once per wait. Their ratio is the repetition that cost R60 its 20 passes.
- `adv_before_mb` / `adv_back_mb`: of the pages advised before the last pass, how many were resident at it. This tells
  whether the advice was honoured.
- `per_pass`: per pass, `t_ms`, `lost_mb`, `have_mb`, `advised_mb`, `calls`, `adv_before_mb`, `adv_back_mb`,
  `mincore_ms`, `advise_ms`, `cpu_ms`, `inblock_mb`.

`wait_lost_mb` is measured at the first final pass, before its advice. It is r22's definition, but in r23 it is net of
the in-wait pass.

## 3. Threads 3 against 4

### 3.1 How the app applies the thread count (code reading, `lib/assistant.dart`, `lib/research.dart`)

- The kit's load (`ensureLoad` → `_loadOn`) and each run without a reload (`_benchOn` → `_engine.setThreads(threads,
  read)`, that is `llama_set_n_threads`) give llama.cpp the spec's count for both prompt and decode. The row records
  what was applied (`threads_loaded = [threads, read threads]`).
- **The threads are restricted, as a set, to cores 4 to 7.** Before every completion, `_pin()` calls
  `pinInferenceThreads(_pinnedCores(n))`, which sets `sched_setaffinity` on the app's DartWorker threads.
  `_pinnedCores` returns the phone's fast cores, {4, 5, 6, 7} on the SD888 (cores 4 to 6 are A78, core 7 is the X1),
  whenever `n` ≤ 4, so the mask is the same at 3 and at 4 threads.
- There is no per-thread pinning, so memo-21's "not pinned" is right about single threads, but the set is restricted.
- llamadart attaches no threadpool, so ggml makes a disposable one per graph from the calling DartWorker thread. The
  workers inherit its affinity mask and its thread name.
- Default threadpool parameters apply: poll 50, no cpumask, so ggml sets no affinity of its own.
- **Does a 4th thread land on the 4th big core?** It can: at 4 threads, 4 workers share 4 allowed cores. Where each
  one runs is the kernel's choice, and an unrestricted app thread (UI, raster, the kit's own isolate) or another
  app's thread can take one of the four. At 3 threads, the 3 workers float over the same 4 cores, so the X1 is used
  by whichever worker the scheduler puts there.
- The app's own default is `autoThreads` = the saved bench pick, else min(3, performance threads).
- Campaign 1 found 3 better than 4 (R18, *device*), with 1,909 barriers per token at the time. Today's runtime has
  666 per token (R60, *device*). That is why the question is asked again (memo-21 item 1).

### 3.2 Design

- Each block has two quads (off num num off), one at 4 threads and one at 3.
- The quad at 4 threads comes first in odd blocks and second in even ones. In the block order G O O G, each model
  therefore gets 4 threads first once and last once, as r22 did with hold.
- Warm-ups and profiled runs are at 3 threads.
- No new runtime knob, no pinning, no `mlock`, no LUT.
- The thread count should not change the numerics: the ops split rows, not reductions. The parser checks this: one
  text per arm and thread count, and the same text at 3 and 4.

### 3.3 Placement record

Each run carries `pinned_cpus` (the set the app applied; `null` means every core) and `thread_samples`. These are up to
3 samples, at least 500 ms apart, taken during the decode after the first chunk. Each sample is `[ms into the run,
number of DartWorker threads in state R, the CPU of each]`, read from `/proc/self/task/*/stat` (state = field 3,
processor = field 39). The sampler runs in the app's main isolate at a chunk callback, outside the inference threads.
It is a snapshot, not a schedule trace. Spinning workers in ggml's poll show as R.

### 3.4 Confounds

- Each model has 2 blocks, so the threads effect per model rests on 2 paired block differences per arm (4 quads in
  all). The alternation cancels a linear drift across a block, not a step.
- Hold is now on in every gated run, so thread count and hold are not crossed.

## 4. Arms and load env: the exact env per spec

| model | load env (during the load only) | arm | run env |
|---|---|---|---|
| both | | off | `GGML_TINY_OP=262144 GGML_EXPERT_IO=1 VOLLIE_SAMPLER_BYPASS=1 GGML_C2_IO=1` (r22's off arm) |
| OLMoE (`olmoe-Mi`) | `VOLLIE_HEAD2_REPACK=1` (both arms) | num | off + `VOLLIE_SKIP_MASS=900 VOLLIE_HEAD2=1024 VOLLIE_SKIP_STATS=1` |
| Granite (`granite-Mi`) | none | num | off + `GGML_EXPERT_SKIP=130 GGML_HEAD2=2048 GGML_HEAD2_Q=q4_0 VOLLIE_SKIP_STATS=1` |
| both | | profiled runs | the arm's env + `GGML_CPU_PROF=1 GGML_CPU_AB_TOKEN=1 GGML_CPU_AB=knob:MMID_FUSE+CHUNK2:1:1` |

- Repack is on at every load, as in r22.
- `GGML_PREFILL_COLD` and `GGML_KEEP_WARM` are set nowhere. `VOLLIE_HEAD2_REPACK` is never in a run env, and it is in
  `_knownKnobs`, so `clearKnobs` removes it if it is ever left set.
- Hold and threads are not envs: they are the row's `hold` flag and `threads`.

## 5. Run list (as built; 48 runs, 40 gated, 40 with hold)

Blocks **Granite, OLMoE, OLMoE, Granite**. Each block:

1. Load its model (forced) with its load env.
2. Two unscored warm-ups: arm off, 3 threads, no gate, no hold.
3. Quad 1 and quad 2: off num num off, scored, gated, hold.
4. One profiled run per arm at 3 threads, gated, hold: off then num in blocks 1 and 3, num then off in blocks 2 and 4.

| block | model | quad 1 | quad 2 | profiled |
|---|---|---|---|---|
| b1 | granite | 4 threads | 3 threads | off, num |
| b2 | olmoe | 3 threads | 4 threads | num, off |
| b3 | olmoe | 4 threads | 3 threads | off, num |
| b4 | granite | 3 threads | 4 threads | num, off |

- Per model: 16 scored runs (8 off, 8 num; 8 at each thread count), 4 profiled, 4 warm-ups.
- Run ids: `<model>_<arm>_t<threads>`, `_prof`, `_wu`.
- Kept from r22: r16's gate (below 0.65, read every 5 s, at most 90 s), the prompt, 32 greedy tokens, mincore before
  and after, meminfo, clocks, faults, the I/O lines, the save-to-file button.
- `test/kit_lines_test.dart` checks this list from the model table.

## 6. Report schema (additions over r22)

**Top level:**
- `kit: r23`, `app: 1.23.0+23`.
- `kernel_release` (`uname(2)`).
- `owner_note`: the text on the start screen.
- `baseline_note`.
- `hold_rule`: back 0.95, stall 0.01, rounds 4, every 45,000 ms, settle 1,000 ms, max 8,000 ms, and the advice.
- `thread_rule`: base 3, alt 4, and a note on the affinity set.
- `load_env` is now per model.
- `models[<id>].load`.

**Per row:**
- `pinned_cpus`, `threads_loaded`, `thread_samples`.
- `load_env_set` on the row of each load: the env native code saw during the load.
- the new `hold_stats` fields (§2.4).
- the `c2-head-run` line with `c2_declined` / `c2_why`.

**Parser** (`tools/c2_research_parse.py`, kit r23):

- **Checks:** all of r22's, plus per model:
  - the load env present at every load of that model, absent from the other model's loads, and never in a run env;
  - one text per arm and thread count, and the same text at 3 and 4;
  - head declines in any row (with the reason);
  - num rows without an active head copy;
  - gated rows without `hold_stats`;
  - rows whose applied thread count is not the spec's;
  - the kernel release and the owner note.
- **Per model × arm × threads:** tok/s median [min..max], MB resident before, MB lost in the wait, back %, gate s, decode
  faults and inblock MB, prefill, TTFT, skip %, hold CPU ms, ms after the gate, `Cached`.
- **num − off per model:** mean over ABBA quads ± SE, all quads and per thread count.
- **Threads 4 − 3 per model and arm** (off, num, both):
  - per block, mean of the arm's runs at 4 minus at 3, then the mean over blocks ± SE;
  - all runs, Welch.
- **Placement:** per model and thread count, samples, running threads (median), and the share of running samples on
  cpus 0 to 3, 4 to 6 and 7; the pinned sets seen.
- **Hold per gated run:** gate s, want and lost MB, passes (wait + final), CPU ms (`mincore` ms), ms after the gate,
  back %, advised MB sum / distinct, how much of the earlier advice was resident at the last pass, the stop reason, and
  MB resident before.
- **Hold summary:** median [min..max] of CPU ms, `mincore` ms, passes, ms after the gate, back %, the number of runs at
  95 % or more, advised distinct MB, blocks read MB, fadvise fails.

## 7. Owner steps

1. Before the session, close background apps (R60: a banking app was running). The start screen shows the same line,
   in bold, under the kit button.
2. Install `lab/dist-c2/vollie-c2-1.23.0+23-research-r23.apk`. It is an update over r22 (same package and key), and both
   downloaded files stay. No new download.
3. Open the app, go to Settings, and tap **"Run research kit r23 (~35-80 min)"** on a cool phone, with the same charger
   state as r22. Keep the app in front; the screen stays on.
4. When it ends, tap **"Save research results to Downloads"** and send `Download/vollie-research-r23-<time>.json`.

## 8. Time (*estimated*)

- **Every gate times out (R60's day):** r22 took about 70 minutes (*device*, R60, 09:48 to 10:58) with the same 48 runs.
  r23 drops r22's up to 5 s after the gate in 24 runs and adds 1 to 3 s (at most 8 s) in 40 runs. **About 70 minutes,
  at most about 75.**
- **Gates pass (r21's day, 29.6 minutes, *device*):** plus 1 to 2 minutes of hold. **About 32 to 40 minutes.**
- **What to cut, in order:** the profiled runs of blocks 3 and 4, then all profiled runs, then the second warm-up. The
  quads and the fourth block are the comparison and are not cut.

## 9. Each model as a full subject

| | Granite (`granite-Mi`) | OLMoE (`olmoe-Mi`) | generalised? |
|---|---|---|---|
| head | campaign 1's tied head, not taken by KleidiAI | untied Q6_K head, needs 0006 at load | no: the fix applies only to an untied Q6_K head |
| off arm vs r22 | unchanged | head numerics back to Q6_K (R61: 11.7132 vs 11.7165, *host*) | per model |
| hold | the same code and rule | the same code and rule | answered per model by r23 |
| threads 3 vs 4 | the same factor | the same factor | answered per model by r23 |

## 10. Not verified

- **Anything on Android.** Never run on a phone:
  - whether 0006 keeps OLMoE's head out of KleidiAI on the phone (its effect was reproduced on a host KleidiAI build
    only) and whether the head line then shows `c2_copy=1`;
  - the r23 hold rule, its cost and whether 1 s is enough for the readahead;
  - `uname` and the `/proc/self/task` sampler;
  - where Android puts 4 workers on 4 allowed cores.
- The cost estimates in §2.3 are a fit to R60's 24 runs, extrapolated to 3 to 5 passes. The fit's residual (about
  0.5 s) is as large as the target.
- That unsetting `VOLLIE_HEAD2_REPACK` after the load is equivalent to keeping it set (§1) rests on reading the code.
- The run time is an estimate from R60's and r21's days.
- `libllama.so` is not byte-identical to r22's, though its sources are (§1.1).
