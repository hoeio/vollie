# Memo 23: research kit r24, the plateau gate, a charger guard, and why OLMoE's skip arm faulted more in R62

Agent model: claude-opus-5-5

Date 2026-11-16. Author: Tom (code). **Nothing here ran on a phone, and no phone speed is stated.** The APK was built
and inspected on the laptop. Labels: *host* = Apple M4 Pro, relative only; *estimated* = arithmetic; *device* = quoted
from an existing report or computed from one, with the report named. R60 = kit r22's day
(`reports/device/r22_2026-11-15.json`); R62 = kit r23's day (`reports/device/r23_2026-11-16.json`). Kit r24 starts from
kit r23 ([memo-22](memo-22-kit-r23.md)); this memo lists what changed.

## 1. Diagnosis: OLMoE's num arm and the major faults (no phone; *device* data from R60 and R62)

The numbers below are computed from the two raw reports, from these fields:
- the `c2-io` line (major and minor faults at the prompt's start, first decode graph and end);
- `c2-io-blk` (`ru_inblock`);
- `mem_before` / `mem_after`: `/proc/meminfo` plus this process's `RssAnon`, `RssFile` and `VmSwap` from
  `/proc/self/status`;
- `res_before`, `wait_lost_mb`, `hold_stats`;
- the profiled runs' per-token fault counts.

The kernel is `5.4.274-qgki` (R62's `kernel_release`).

### 1.1 The major faults are mostly swap-ins of the app's own anonymous memory, not file reads

- **Gated runs start with part of the process swapped out; warm-ups do not.** `VmSwap` at the start of gated runs:
  - R62: median 263 MB (Granite) and 216 MB (OLMoE);
  - R60: 267 and 294 MB;
  - at the warm-ups (no wait): 50 to 70 MB.
  - System `SwapFree` was often near 0 (zram nearly full) at run start or end.
- **The waits also unmap the file.** `RssFile` (file pages mapped by the process) at gated run start was a median of
  186 MB (Granite) and 539 MB (OLMoE) in R62, against about 2 GB at the warm-ups. At the same time `mincore` showed
  1.9 to 2.4 GB of the file in the page cache.
- **Majors follow swap.** Across the gated runs, a run's major faults correlate with the `VmSwap` at its start:
  - Granite: r = 0.58 (R62), 0.85 (R60);
  - OLMoE: r = 0.38 (R62), 0.75 (R60).
  - Majors do not correlate with the blocks read (r = 0.26, −0.20, −0.56, −0.28). Warm-ups take 338 to 1,208 majors
    per run; gated runs take 12,700 to 38,500 (medians).
- **Decode majors can exceed the pages read.** In 6 of 20 OLMoE gated runs (R62), the decode took more major faults
  than it read pages. Examples (decode majors / decode pages read via `ru_inblock`):

  | run | decode majors | decode pages read |
  |---|---|---|
  | pos 16 | 5,149 | 1,866 |
  | pos 17 | 7,770 | 707 |
  | pos 33 | 5,471 | 1,744 |

  A fault that reads a file page always submits that read from the faulting process, so it is counted in
  `ru_inblock`. The excess must come from somewhere else.
- **Why these are swap-ins (inferred from Linux 5.4 source, not verified on this kernel's build):**
  - A file fault is major only when the page is not in the page cache (`filemap_fault`). It then reads the page, and
    the read is counted in `ru_inblock`.
  - A fault on a file page that is in the cache but not mapped is minor. These are the pages the hold read back
    without mapping them; they show up in `minflt`.
  - A zram swap-in is a major fault (`do_swap_page`). zram reads synchronously, without block-read accounting, so it
    is not counted in `ru_inblock`.
  - So the excess majors are zram swap-ins: the repacked weights (Q4_0 attention, OLMoE's Q6_K head in `CPU_REPACK`),
    the KV cache and the compute buffers, swapped out during the 90 s wait.
- **Some swap-outs happen during the decode itself.** In 3 of the 5 high-fault OLMoE runs, `VmSwap` rose across the
  run (pos 21: 157 → 268 MB).
- **Answer to "why does a run that starts with 93 % of the file resident still take thousands of majors?"**
  "Resident" is the model file's page cache. The majors are mostly the process's anon memory coming back from zram
  (100 to 435 MB per run), plus the missing 5 to 7 % of the file: about 150 MB, read in the prefill, where the median
  `ru_inblock` is 100 to 300 MB.
  - The prefill takes most of them: median prefill majors 20,000 (OLMoE) and 35,000 to 70,000 (Granite), against
    about 1,000 in the decode.
  - Pages lost during the prefill are not the main source: the decode's `ru_inblock` is small (median 7 to 14 MB) in
    most runs.

### 1.2 The num arm's extra decode faults are not shown to be an arm effect

- **The decode fault counts are bimodal in both arms.** OLMoE scored runs (R62) split into about 1,000 and
  5,000 to 11,000 faults:

  | arm | 3 threads | 4 threads | high runs |
  |---|---|---|---|
  | num | 987, 5,149, 5,471, 7,770 | 908, 960, 8,264, 11,026 | 5 of 8 |
  | off | 380, 1,138, 1,140, 9,181 | 883, 931, 1,499, 11,017 | 2 of 8 |

  5 of 8 against 2 of 8 gives Fisher's exact p = 0.32 (two-sided).
- **Profiled runs show no excess:** num 1,093 and 938, off 577 and 1,107.
- **R60 showed no excess either.** Its num arm (skip alone, the head declined) had 572 against 532 (hold quads).
- **The high runs cluster in time, across arms:** pos 16, 17, 19, 20, 21 (19 is off), then 33 and 34 (34 is off).
  ABBA balances a linear drift. An episode two runs long, though, falls on the adjacent num, num pair in the middle of
  a quad more often than on off. The medians 5,310 against 1,139 rest on 4 runs each.
- **Total majors per run:** num 13,280 against off 9,591 (medians over the 10 gated runs per arm; prefill 12,332
  against 8,400, decode 3,121 against 1,123).

### 1.3 The candidates

1. **"The prefetch requests a different set than the kernel computes": refuted by the code.**
   - `ggml_cpu_expert_prefetch` (campaign 1, `GGML_EXPERT_IO=1`) builds the skip mask first, with
     `ggml_cpu_skip_prepare`, the same decision the op makes next. It advises `need[e]` = routed and not skipped in
     every slot.
   - The c2 mass rule (0003) is part of that mask.
   - The next-layer helper (bit 4) is off in the kit.
   - In any case, file pages that miss the prefetch would arrive with a read counted in `ru_inblock`, and the excess
     majors have none.
2. **"Pages in the cache but unmapped counted as major": refuted by the kernel code (5.4 `filemap_fault`).**
   - Those faults are minor.
   - They are what the hold produces (`RssFile` 25 to 1,400 MB at run start, while the page cache holds about 2 GB),
     and they show as tens of thousands of minor faults per run.
3. **"The num arm's 58 MB anon head copy pushes file pages out": not supported.**
   - The copy is built only in the first num run of a quad. Five builds are logged (`c2-head-build`, 147 to 199 ms).
   - High-fault runs include a build (16) and non-builds (17, 21, 33).
   - Low-fault runs include builds (23, 28, 32, 36).
   - The 58 MB allocation can still add reclaim pressure. That is not excluded, only not visible.
4. **Remaining arm-specific hypothesis (inferred only, not supported by the totals):** the num head reads about 7,000
   of OLMoE's 50,304 head rows per token from the repacked Q6_K head (84.5 MB anon).
   - Swapped-out head pages would then come back token by token in the decode. In the off arm the whole head is read
     at the prefill's logits.
   - That would move swap-ins from the prefill into the decode, at most 20,600 pages. The prefill/decode split above
     does not show such a shift clearly.
   - Granite's exact rows come from the file-backed tied embedding, which matches its having no excess.
   - R60's declined head matches it too.

### 1.4 What it costs, and what r24 records to decide it

- **A rough fit (*estimated* from *device* data; 20 unplugged OLMoE scored and profiled runs, no standard errors):**
  decode ms per token ≈ 49.5 + 0.104 × decode-inblock MB + 2.8 × (thousands of decode majors) − 5.3 (num) + 26.3
  (4 threads); residual 13.7 ms.
  - With faults and I/O held equal, the num arm is about 5 ms per token faster. At about 60 ms per token that is
    roughly +1.5 tok/s, R60's size.
  - The faults ate it. This is a reading, not a test.
- **What r24 records to test §1.1 and §1.2 directly** (a recording only; no setting changes):
  - `/proc/vmstat` `pswpin`, `pswpout`, `pgmajfault`, `pgpgin`, `pgpgout` at the run's start, at its first chunk and at
    its end (system-wide);
  - `mem_first`: `VmSwap` / `RssFile` / `RssAnon` after the prefill.
  - Decode swap-ins then show directly as `pswpin` between the first chunk and the end. If the app's SELinux domain
    cannot read `/proc/vmstat`, the field is null.

The lever this points to is keeping the app's anon memory (about 0.5 to 0.7 GB, the repacked weights in it) out of
zram during the wait. It is not part of r24; it would need its own pre-registration. Candidates: `mlock` of the repack
buffers (the limit is 64 MiB, memo-21), a periodic touch of the anon buffers only (sized about 0.6 GB against the
file's 3 GB), or less repack.

## 2. The plateau gate

### 2.1 What the heat series show (*device*, R60 and R62)

- R60 and R62 contain 80 gated waits. Every one ran the full 90 s and none reached 0.65.
- The headroom (Android's forecast) often rises for 10 to 25 s right after a run, then falls to about 0.72 by 40 to
  50 s. After that it moves by ±0.02 around that level.
- On 2026-11-14 (r16 control, r20, r21 summaries) the 0.65 gate was reached in a median of 35 s, or at once. Those
  summaries carry no heat series, so the plateau rule cannot be replayed on them.

### 2.2 The rule

At every reading of the wait, in this order:
1. **gate:** headroom < 0.65 (r16, unchanged);
2. **plateau:** at least **30 s** into the wait, and both of:
   - the lowest of the **last 2** readings is not more than **0.01** below the lowest of the 2 readings before them
     (it has stopped falling);
   - it is at least **0.05** below the highest reading of this wait. It did fall: the forecast often rises first, and
     without this condition the rule fires on the rise.
3. **timeout:** 90 s.

Null readings are skipped. The constants are `Research.plateauFrom`, `plateauWindow`, `plateauEps` and `plateauFall`.
The same rule is in the parser as `plateau_end` (`--plateau` replays it on any report with heat series). Both the app's
unit test and the parser's selftest check it on two R62 series (pos 4: ends at 50 s at 0.725; pos 6: does not fire
while the forecast rises, ends at 60 s).

### 2.3 Replay on every gated run of R60 and R62 (*device* series, replayed on the host)

`tools/c2_research_parse.py --plateau <report>`:

| report | model | runs | wait as run (s) | plateau wait (s), median [range] | headroom at plateau end | headroom at the 90 s end | lowest of the 90 s, median | s saved, sum |
|---|---|---|---|---|---|---|---|---|
| R60 | Granite | 20 | 90 to 95 | 47 [30..90] | 0.738 [0.708..0.769] | 0.742 [0.707..0.784] | 0.732 | 806 |
| R60 | OLMoE | 20 | 90 to 95 | 50 [42..81] | 0.722 [0.708..0.761] | 0.722 [0.704..0.766] | 0.713 | 773 |
| R62 | Granite | 20 | 90 to 93 | 51 [45..91] | 0.774 [0.665..0.893] | 0.770 [0.665..0.966] | 0.748 | 659 |
| R62 | OLMoE | 20 | 90 to 93 | 51 [45..71] | 0.742 [0.707..0.777] | 0.734 [0.694..0.815] | 0.730 | 765 |

- **All 80 waits end by plateau.** The wait is about 40 s shorter (median), and the headroom at the end is within
  about 0.01 of where the 90 s wait ended.
- **The headroom fell by more than 0.03 after the rule's end in 5 of 80 waits** (2 in R60, 3 in R62). This is the
  rule's error: an early end on a pause.
- **Variants tried on the same data:**
  - 3-reading windows end later and fire on the post-run rise more often (> 0.03 further fall in 6 to 17 of 40);
  - eps 0.005 ends about 5 s later, with the same error;
  - starting at 20 s ends earlier and errs more (3 and 7).
  - The chosen rule had the fewest errors at about 50 s.

### 2.4 On a day when the gate is reached

- On a steady fall of 0.02 to 0.03 per 5 s (the series above before the plateau), two windows 10 s apart differ by
  more than 0.01, so the rule does not fire; the gate at 0.65 ends the wait as in r16.
- It can end a wait earlier only if the headroom pauses for 10 s above 0.65 after 30 s.
- Not replayable: no heat series exist for 2026-11-14 (*not verified*). The report records `wait_end` per run, so r24
  shows how often each rule ended a wait.

### 2.5 The factor

- Per block, one ABBA quad with r16's gate and one with the plateau gate. The plateau quad comes first in odd blocks
  and second in even ones, so in G O O G each model gets it first once and last once (as threads in r23).
- Profiled runs use r16's gate.
- **Confound:** hold is unchanged, but its in-wait pass runs at the first poll at or after 45 s. A plateau wait that
  ends before 45 s has no in-wait pass. Under the replay's median of about 50 s most plateau waits still get one. The
  report shows `wait_passes` per run.

## 3. Charger guard

- At every gate poll the app reads `plugged` (`BatteryManager`).
- If a charger is connected, the kit shows "Charger connected: unplug it to continue (paused N s)" and re-reads every
  5 s until it is unplugged.
- Then the wait starts again: clock, readings and heat series restart.
- Each pause is in `plug_pauses` ([ms into the wait, pause ms]), and `wait_ms` is the wait after the last pause.
- A run with a charger at its start or its end (`plugged` / `plugged_after`) gets `plugged_run: true`.
- The parser lists these runs and counts num − off also without quads that contain one ("flag-free"). It drops none:
  the pre-registration decides that.
- Not covered: a charger connected and removed during a 3 to 6 s decode. The rows record only the start and the end.
- **Hold after a pause:** the in-wait pass is not repeated after a charger pause; the final passes still read back
  what was lost.

## 4. Runtime and build

- The runtime is unchanged: `patches/llama.cpp/0001` to `0003` and `patches/llama.cpp-c1-olmoe/0001` to `0006`, built
  from the r23 tree `lab/llama.cpp-c1r23`, checked file by file by `tools/c2_build_kit.sh` and built with
  `android/build_cpu.sh`, flags unchanged.
- **The libraries were rebuilt and their hashes changed, but only in the ELF build ID.** Against r23's libraries, both
  differ in exactly 20 bytes, at file offsets 736 to 755. That range is the `.note.gnu.build-id` descriptor (section at
  0x2d0, 0x24 bytes). Code, data and strings are byte-identical.

| | bytes | sha256 |
|---|---|---|
| `vollie-c2-1.24.0+24-research-r24.apk` | 17,221,503 | `6a655ca7e16a9f4ee99c86ef456463b640b967bb7dfb4e6e421e3a9affa5beba` |
| `libggml-cpu-android_armv8.2_2.so` | 1,761,328 | `e6ef857b101f92289ec8c949091379dff98398baa295ababaf92d640dc3a84ef` (r23: `19f36b6e…`) |
| `libllama.so` | 4,285,208 | `aeab4dad19065ca6e8affb83cd0f085ff24adf1e90c16d8805a913931cb4b9df` (r23: `edcfadf1…`) |

The build record (`assets/kit_build.json`, `lab/dist-c2/BUILD-r24.txt`) lists the nine series patches and app patches
`0001` to `0008`.

## 5. Arms, load env, run list

- The arms and the load env are r23's (memo-22 §4):
  - OLMoE: `VOLLIE_HEAD2_REPACK=1` at load, both arms; num = `VOLLIE_SKIP_MASS=900 VOLLIE_HEAD2=1024 VOLLIE_SKIP_STATS=1`;
  - Granite: num = `GGML_EXPERT_SKIP=130 GGML_HEAD2=2048 GGML_HEAD2_Q=q4_0 VOLLIE_SKIP_STATS=1`.
- 3 threads in every run.
- Hold is r23's rule in every gated run.

**48 runs, 40 gated.** Blocks Granite, OLMoE, OLMoE, Granite; each block:
1. Load (forced).
2. 2 warm-ups: no gate, no hold.
3. Two scored quads (off num num off): one with the plateau gate, one with r16's.
4. 2 profiled runs with r16's gate.

| block | model | quad 1 | quad 2 | profiled |
|---|---|---|---|---|
| b1 | granite | plateau | r16 | off, num |
| b2 | olmoe | r16 | plateau | num, off |
| b3 | olmoe | plateau | r16 | off, num |
| b4 | granite | r16 | plateau | num, off |

- Per model: 16 scored runs (8 per arm, 8 per gate rule), 4 profiled, 4 warm-ups.
- Run ids: `<model>_<arm>_pl`, `_r16`, `_prof`, `_wu`.
- `test/kit_lines_test.dart` checks this list.

## 6. Report schema (additions over r23)

**Top level:**
- `kit: r24`, `app: 1.24.0+24`.
- `gate.plateau` (from_ms, window, eps, fall, the test) and `charger_rule`.
- `thread_rule.threads` = 3.
- `idle_baseline`: before the first run, 7 samples 5 s apart over 30 s of [ms, headroom, battery °C, plugged, thermal
  status].
- `thermal_zones`: every `/sys/class/thermal/thermal_zone*` with type, temp or the error. The app's SELinux domain may
  deny these; errors are recorded and the kit continues.

**Per row:**
- `gate_rule`, `wait_end` (gate / plateau / timeout / stopped), `wait_ms`, `plug_pauses`;
- `plugged_after`, `plugged_run`;
- `vmstat_before` / `vmstat_first` / `vmstat_after`, `mem_first`;
- `zones_before`: the zones that were readable at the start.

**Parser** (`tools/c2_research_parse.py`, kit r24):
- **Checks:** r23's, plus:
  - each row's gate rule against its spec;
  - wait ends per rule;
  - charger pauses and flagged runs;
  - how many rows have vmstat;
  - readable thermal zones;
  - idle baseline samples.
- **Per model × arm × gate rule:** tok/s, wait s and ends, headroom at the start, MB resident before and lost in the
  wait, decode faults and inblock, `pswpin` prefill / decode, `VmSwap` before, hold CPU.
- **num − off:** over ABBA quads, all quads, per rule, and flag-free.
- **Plateau − r16 per model:** for tok/s, wait s, MB resident before, MB lost in the wait, headroom at the start,
  big-core MHz and cap MHz; per block paired ± SE, and all runs, Welch.
- **Lists and tables:** the flagged runs; a per-run swap and fault table; r23's hold table.
- `--plateau` replays the rule on a report's heat series.
- `--selftest` passes with synthetic r21 to r24 cases and the rule's two R62 series. The R60 and R62 reports parse as
  before.

## 7. Owner steps

1. Before the session, close background apps and unplug the charger. The start screen says so. If a charger is
   plugged in, the kit pauses and shows it.
2. Install `lab/dist-c2/vollie-c2-1.24.0+24-research-r24.apk`. It is an update over r23 and keeps both models.
3. In Settings, tap **"Run research kit r24 (~35-75 min)"**. The first 30 s are an idle baseline.
4. When it ends, tap **"Save research results to Downloads"** and send `Download/vollie-research-r24-<time>.json`.

## 8. Time (*estimated*)

- **Gates never reached (R60 / R62 days):** R62 took 72 minutes (10:31 to 11:43, *device*). r24 drops the slower
  four-thread runs. The replay saves about 36 to 40 s per plateau wait, × 16 waits ≈ 10 minutes. Add 0.5 minutes of
  baseline. **About 60 to 63 minutes.**
- **Gates reached (2026-11-14 days, median wait 35 s):** like r21's 29.6 minutes (*device*), plus hold. **About 32 to
  40 minutes.**
- Charger pauses add their own length.
- What to cut, in order, as in memo-22.

## 9. Never run on a phone

- The plateau rule in the app, and its interaction with the hold's in-wait pass.
- The charger guard. `plugged` comes from the battery broadcast, which is presumably sticky (it has been read since
  r16).
- Reading `/proc/vmstat` and `/sys/class/thermal` from the app's SELinux domain.
- The idle baseline.
- §1's mechanism. The zram reading is inferred from the 5.4 kernel source and the reports' counters; this kernel's
  build was not read. r24's `pswpin` counters are the test.
- The libraries were rebuilt and differ from r23's only in the build ID (§4). They were not run.
