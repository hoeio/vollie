# Memo 31 working draft (Ebony): slow tail, clock cap, levers, claim-round design

Agent model: claude-fable-5-2 (Ebony). Date 2026-11-18. Not committed. Repository HEAD when written: `d59e7d9`.

**Nothing here ran on a phone and no host benchmark ran.** It is short scripts over the stored raw reports
(`reports/device/r25_2026-11-16.json`, `r26_2026-11-18.json`, `r27_2026-11-18.json`, and a cap count over every
stored report of both campaigns), a reading of `lab/app-c2/lib/research.dart` and
`lab/llama.cpp-c1r27/ggml/src/ggml-cpu/ggml-cpu.c`, web sources, and arithmetic. Labels: *device* = read or
computed from a stored report; *estimated* = arithmetic, a resampling, or a judgement; "source" = a published
claim not reproduced here. **Every split below is post hoc, not pre-registered, and on 16 scored runs per model;
R67's verdicts stand as written.** Scripts (same folder): `ebony_runs.py`, `ebony_tail.py`, `ebony_capzones.py`,
`ebony_claimsim.py`. Verdicts and pre-registrations are the lead's.

## 0. Summary

1. **R67's 25 tok/s is first of all a clock result.** In the stored reports r13 to r26 of both campaigns, cpu 7
   never read above 2,150 MHz and the resting cap was 2,035 (cpus 4-6 then run at 2,112); r27 is the first
   session at the hardware maxima 2,419 / 2,841 (*device*, §2.1; before r13 there are two isolated readings). Clean OLMoE off went from 44.9 ms per token
   (R64/R66, memo-29) to about 38.0 to 38.6 ms (R67), with no kernel change. A claim round under the 2,035 cap
   would score about 21 to 22 tok/s (*estimated* from memo-29's floor), whatever else we do.
2. **The cap is not explained by any temperature the kit records.** r26 sat at 2,035 with cooler readings
   (headroom 0.71 to 0.74, xo-therm 36.1 °C, battery 25.5 to 26.7 °C) than r27 had while uncapped (headroom 0.74
   to 0.80, xo-therm 36.8 to 38.7 °C) (*device*, §2.2). Something that differs by session decides it. Five
   candidates, none shown; a five-minute probe separates most of them (§2.4, T-A).
3. **The slow tail has two mechanisms** (*device*, §1):
   - **A, CPU taken from the workers** (run-queue wait 315 to 461 ms summed over the three workers in one decode,
     700 to 1,100 involuntary switches, almost no faults): OLMoE pos 15, 17, 22 (17.6 to 19.1 tok/s). The stolen
     time is not the app's own other threads. It appeared in the first runs after a model switch in three of
     three switches. Pin does not prevent it (pos 15 is a pin run).
   - **B, pages** (91 s gate waits, 150 to 540 MB of the wanted set missing at run start, thousands of faults,
     system time 130 to 410 ms): pos 16 and 34 (18.9, 19.6; unpinned) and, milder, pos 18 and 33 (23.2, 23.7;
     pinned). The exec mapping did not keep idle pages mapped or resident through a 90 s wait.
4. **For the claim the tail rate comes first, and a margin second.** Clean runs are 24.98 to 25.56 (8 runs, 7
   of them at or above 25.0). Resampling: with today's clean level and no tail, P(median of 10 ≥ 25.0) is about
   1.0; with R67's pin-on runs (4 of 8 below 25.0) it is 0.57 (*estimated*, §4.1). One run in four in the tail
   gives P ≈ 0.9. *Changed after reading Bonnie's draft:* my first version said a margin hardly matters; that
   came from a model that put every tail run at 20 tok/s. With the real tail values the mild runs (23.2, 23.7)
   count through the mean of the 5th and 6th run, and −1.5 ms per token lifts the same resampling to 0.87
   (her table, which I reproduce). Both axes count.
5. **Ranked for the claim** (§3): (1) know and record the cap state; (2) touches through the executable mapping
   during the wait, from a little core (Y1 b, restated); (3) worker scheduling priority (new), with a name for
   who preempts; (4) a clean-run margin from `MASS=900` (numerics, built) and the outside-graph crumbs. The 2025
   to 2026 literature and llama.cpp gave nothing new that is cheap for a CPU-only phone (§3.3).
6. **My view of the next round: not the claim.** One rehearsal session in the claim's exact layout, with the cap
   and thermal-status fields added, Y1 b and the priority arm as factors, on both models; then a second without
   the deep-sleeping list. Claim only if both rehearsals show at most 2 of 10 runs below 25.0 at an uncapped
   clock (§4.3).

## 1. The slow tail (R67, *device*)

### 1.1 OLMoE scored runs, one row each

Columns: wait = gate wait in s; miss = wanted MB not resident at run start; mapped = the process's `RssFile` at
run start (MB; see §1.4); dmaj / dmin = major / minor faults in the decode; sys = process system CPU in the
decode (ms); rq = run-queue wait summed over the three workers in the decode (ms, `c2-sched`); ivcs = their
involuntary switches; excess = decode wall minus 1,215 ms (the best run, pos 21). Clocks read 2,419 / 2,841 as
cap before and after in every row.

| pos | cell | tok/s | wait | miss | mapped | dmaj | dmin | sys | rq | ivcs | excess ms | type |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 15 | PX | 17.63 | 51 | 27 | 2,704 | 61 | 3,694 | 98 | 461 | 1,084 | +541 | A |
| 16 | px | 18.89 | 80 | 188 | 2,150 | 675 | 19,057 | 406 | 36 | 215 | +424 | B (a worker on cpus 0-3 in 2 of 8 samples) |
| 17 | pX | 19.05 | 0 | 18 | 2,493 | 3 | 4,010 | 89 | 315 | 708 | +410 | A |
| 18 | Px | 23.18 | 91 | 150 | 429 | 4,246 | 6,961 | 134 | 37 | 189 | +121 | B, mild |
| 19 | px | 25.02 | 45 | 0 | 2,562 | 10 | 4,367 | 72 | 0 | 6 | +23 | clean |
| 20 | Px | 25.37 | 55 | 0 | 2,576 | 5 | 3,323 | 70 | 0 | 2 | +6 | clean |
| 21 | PX | 25.48 | 0 | 0 | 2,579 | 1 | 3,302 | 89 | 0 | 0 | 0 | clean |
| 22 | pX | 18.24 | 66 | 151 | 1,335 | 59 | 6,587 | 92 | 329 | 1,028 | +483 | A |
| 27 | Px | 25.07 | 92 | 251 | 1,346 | 51 | 5,500 | 116 | 0 | 13 | +20 | clean (despite miss) |
| 28 | pX | 24.32 | 91 | 251 | 793 | 105 | 6,966 | 114 | 3 | 16 | +58 | B, mild |
| 29 | px | 25.21 | 35 | 19 | 2,571 | 7 | 4,227 | 58 | 0 | 4 | +14 | clean |
| 30 | PX | 25.56 | 0 | 12 | 2,571 | 1 | 3,479 | 76 | 0 | 0 | −3 | clean |
| 31 | pX | 25.08 | 35 | 0 | 2,575 | 0 | 4,363 | 72 | 0 | 6 | +20 | clean |
| 32 | PX | 24.98 | 0 | 0 | 2,575 | 0 | 3,312 | 57 | 2 | 14 | +25 | clean |
| 33 | Px | 23.67 | 91 | 232 | 23 | 189 | 6,726 | 163 | 2 | 33 | +94 | B, mild |
| 34 | px | 19.58 | 91 | 538 | 81 | 614 | 14,104 | 408 | 13 | 359 | +366 | B |

Granite (same script): type A in pos 3 (Px, 9.42 tok/s, rq 948 ms; no graph start on cpu 7 in the whole run and
the cap fell to 2,150 during it), pos 5 (px, rq 160, also faults), pos 39 (PX, rq 162) and pos 40 (px, rq 155);
the other twelve scored runs have rq ≤ 11 ms.

### 1.2 Type A: what the fields say, causal against coincident

- **Causal for the lost time (arithmetic on the run itself):** in the three OLMoE type-A runs the workers' summed
  run-queue wait is 461 / 315 / 329 ms and the wall excess 541 / 410 / 483 ms; their on-CPU time rose by 1,046 /
  828 / 1,003 ms over a clean run's 3,385 ms. That is the signature of one worker descheduled while the other two
  spin at the next barrier (wall ≈ wait, extra CPU ≈ 2 × wait). A token has 305 barriers, so an interval is about
  0.12 ms, and the mean wait per involuntary switch is about 0.4 ms: every preemption outlasts the op it hits.
  The expert op already claims chunks dynamically (`ggml_cpu_mmid_claim`), so finer chunks cannot help.
- **Not the app's other threads:** process CPU minus the workers' CPU in the decode is +139 to +255 ms in every
  run, slow or clean (it is +230 to +250 in pin runs and +140 to +185 without the pool, a pool cost of about
  80 ms per decode that is worth a look by Tom, but it does not move with the tail).
- **Not the exec mapping** (I checked because all three OLMoE cases are X runs): Granite's four cases are Px, px,
  PX, px, and the two profiled OLMoE runs with rq 161 and 222 ms (pos 36, 23) are exec off. Pooled over both
  models, 4 of 16 X scored runs and 3 of 16 x runs have rq above 100 ms.
- **Not the pin, either way:** pos 15 and Granite pos 3 and 39 are P runs with every sample on cpus 4-7.
- **Coincident, possibly causal: a model switch shortly before.** Type A sits in runs 3 to 5 after a load that
  replaced the other model (OLMoE pos 15, 17; Granite 3, 5 and 39, 40), plus pos 22 and two profiled runs. Block
  b3, which reloaded the same model (2.6 s, page cache kept), has none in eight scored runs. A switch leaves 2 to
  2.7 GB of the other model's pages to be reclaimed and the swap device full (`swapfree` 0 to 8 MB before all
  three OLMoE cases, but also before four clean runs, so a full swap is not sufficient).
- **Not thermal:** headroom 0.64 before all three, clocks at the maxima before and after.
- **Unknown:** who runs instead. `c2-sched` counts the wait, not the thread that caused it. Candidates: kernel
  reclaim threads (kswapd, zram compression), per-CPU kernel workers for I/O completion (which a floating mask
  cannot dodge), other processes after the switch. A vendor kernel option to keep kswapd off chosen cores exists
  in Qualcomm-derived trees (`CONFIG_KSWAPD_CPU_AFFINITY_MASK`, source: a Sony msm8994 commit, search summary);
  whether this phone sets it is unknown.

### 1.3 Type B: what the fields say

- All five B rows follow waits of 80 to 91 s (gate timed out). After waits of 0 to 55 s the miss is 0 to 27 MB
  in every run. Within the 91 s waits the loss runs through the whole wait (pos 33: 655 MB gone at 45 s, 1,324 MB
  at 91 s after the first read-back, `hold_stats.per_pass`), so there is no onset time to dodge.
- The hold stops by its round limit before the set is back (pos 33: 81.5 % back, stop "rounds"; pos 34: miss
  538 MB). What it reads back is unmapped (advice only) and is the first to go again.
- **Pin changes how much a missing page costs** (n small, descriptive): pinned runs with 150 to 251 MB missing
  lost +121, +94 and +20 ms; unpinned runs with 188 and 538 MB missing lost +424 and +366 ms. Pos 16 had a worker
  on the little cores in 2 of 8 samples. So part of pin's +1.94 tok/s is robustness to faults, and pin plus a
  complete read-back should leave little of type B.
- **The exec mapping alone does not hold idle pages**: after 90 s waits the X runs were as unmapped as the x
  runs (pos 28: 793 MB mapped; pos 22 after 65 s: 1,335 MB; x runs 23 to 1,346 MB) and lost 418 and 636 MB. That
  is what the 5.4 code predicts: the executable class protects pages whose reference bit is set, and nothing
  references them in an idle wait. R67's X column is therefore mostly the wait-length confound the lead named; the
  test of Y1 is its second half (touches through the mapping during the wait), not yet built.

### 1.4 T1's smaps field: the answer is already in the report

`smaps_before.model` reads 0 because the model's mapping is summed into `file_rest` (size 3,652 MB in every run,
which contains the 2,986 MB mapping; inferred cause: the kit compares the `/data/user/0/...` path with the
`/data/data/...` name that `/proc/self/smaps` prints). `file_rest` Rss equals `mem_before`'s `file=` (`RssFile`)
within 10 to 200 MB in all 16 runs (*device*), so **mapped MB at run start = `RssFile`**, and T1's question is
answered: after a long wait resident ≫ mapped (2,550 resident, 23 mapped at pos 33), but that alone is cheap in
the decode (pos 27: 1,224 MB resident-but-unmapped, +20 ms) and costs TTFT instead (+0.4 to +1.5 s; the prefill
takes 43,000 to 62,000 minor faults against about 3,000). The decode cost follows the *missing* MB and the
majors. Fix for Tom: match the mapping by device and inode, or by file name suffix.

## 2. The clock cap

### 2.1 History (*device*, every stored raw report)

`cap=` is `scaling_max_freq` of cpufreq policies 4 and 7 (`research.dart:282`). Counts of the value over all
rows of a report: r22 2,035 ×90, 1,785 ×3, 1,555 ×3; r23 2,035 ×75, 1,555 ×18, 1,785 ×3; r25 2,035 ×70,
1,555 ×23; r26 2,035 ×79, 1,555 ×13, 2,150 ×3; **r27 2,419/2,841 ×83, 2,150 ×8, 1,555 ×4, 1,785 ×1**. Campaign 1's
reports r13 to r16 have no cap field, but cpu 7 never reads above 2,150 in them (2,035 and 2,112 dominate).
Earlier there are two isolated uncapped readings (found by Bonnie, I looked them up): r5 (2026-11-04) one
`mhz_after` of 2,419 / 2,380, and r12 (2026-11-07) the first baseline run's `mhz_before` with cpus 4-6 at 2,419.
So the uncapped state is not new to the phone; a whole session in it is. So
**campaign 1's 20.51 tok/s and every campaign-2 number before R67 were measured under the 2,035 cap**, and R67 is
the only uncapped session we have.

One value is written for both clusters and it is always a step of the X1's table (2,150, 2,035, 1,785, 1,670,
1,555); the A78s then run at the next step of their own table above it (2,112 under 2,035, 2,227 under 2,150).
That looks like one limiter that takes a single kHz value for all big cores (Samsung kernels have such a
max-limit interface used by the system's resource manager; I could not find this phone's kernel source, so this
is an inference from the pattern, not a source).

### 2.2 It is not a function of the recorded temperatures

| | cap before | headroom | xo-therm °C | battery °C | cpu-1-7 °C |
|---|---|---|---|---|---|
| r26 pos 7, 9, 11 (gated, 90 s waits) | 2,035 (2,150 once) | 0.71 to 0.74 | 36.1 | 25.5 to 26.7 | 40 to 42 |
| r27 pos 3 to 6 (gated, 90 s waits) | 2,419 / 2,841 | 0.74 to 0.80 | 36.8 to 38.7 | 26.4 to 26.8 | 48 to 53 |
| r27 pos 1, 2 (warm-ups, after the 6 GB copy) | 1,555 | 0.75 to 0.82 | 38.9 to 41.1 | 26.7 | 48 to 52 |
| r27 pos 37, 38 → 39 | 2,419 → 2,150 → 1,785, then 2,419 again after 92 s | 0.77 → 1.19 → 0.74 | | | |

Pooled over r25, r26 and r27, no zone separates "uncapped" from "capped" (best: a modem PA sensor, AUC 0.98, with
overlapping ranges; headroom 0.94; `ebony_capzones.py`). Inside r27 the cap behaves like a fast thermal ladder
(steps down above headroom about 0.95, back to the maximum within one 92 s wait). In r22 to r26 the same ladder
appears with **2,035 as its resting step instead of the maximum**. A simple hysteresis on headroom does not fit
both sessions (r27 returns to the maximum at 0.74 to 0.80; r26 stays at 2,035 at 0.71).

### 2.3 Candidates for the session-level cause (probabilities are my judgement, *estimated*)

| candidate | for | against | P |
|---|---|---|---|
| a resource-manager level driven by a temperature we do not record, with slow hysteresis (the session's peak sets a level that needs a deeper cool-down to clear) | r26 began at thermal status 2 and headroom 0.91 to 0.99, r25 at status 3; r27 began at status 1 and its skin-side zones are 2 to 4 °C lower | r27 itself went to headroom 1.19 at pos 38 and was uncapped again 92 s later | 0.30 |
| a power setting that was on before and off in r27 (Samsung's power saving has a "limit CPU speed to 70 %" option; 70 % of 2,841 is 1,989 and the next X1 step is 2,035) | the number; one value for both clusters; constant over two weeks | 2,150 appears a few times in r23 and r26, which a standing 70 % limit should hide; nobody changed a setting that we know of | 0.25 |
| the kit's own changes in r27: static screen instead of a spinner at the panel's refresh rate, no OpenCL driver in the process | both are new in r27; less heat also fits the gate being met (headroom floor 0.64 against 0.72 in R60 to R66) | no known mechanism from an animation to a clock cap except heat, and §2.2 argues against heat | 0.20 |
| the deep-sleeping list (owner's bench condition) | new before r27 | no known mechanism | 0.10 |
| something else (battery level, time since boot, a system update, charging history) | not recorded | | 0.15 |

Sources, all weak for the mechanism: Samsung community posts confirm the 70 % option exists on Galaxy phones,
including a post about a Snapdragon 888 Galaxy Z (search summaries, no technical detail); the "Light"
performance profile is reported as about 20 % slower in benchmarks on other models (9to5Google 2022-08-15,
PhoneArena; search summaries). No source found for the kernel interface or for a 2,035,200 value.

### 2.4 Can the runtime influence it legitimately?

- **Directly: no.** `scaling_max_freq` is not writable by an app. The Performance Hint API (ADPF; Java API 31,
  NDK API 33) sends work-duration hints for a set of thread ids; Android's documentation says an app "cannot
  directly assign CPU frequencies" and that the system decides from the hint and the thermal state (source:
  developer.android.com, performance-hint-api; source.android.com). A hint cannot raise a policy maximum. Game
  Mode and vendor game services are for packages declared as games; declaring an LLM app a game to change
  throttling is not something I would put before the owner. A Samsung performance SDK with a "CPU max" parameter
  is mentioned in a Samsung developer forum thread (search summary only; status and availability unknown): a
  desk check for Tom, low odds.
- **Indirectly: yes, if the cause is the app's own heat or behaviour.** The static screen and the missing OpenCL
  driver are app-side and stay; a shorter or cooler warm-up, touches from a little core, and no read-back CPU on
  the big cores before a run are all runtime choices. If T-A shows that the spinner or the GPU driver moves the
  cap, the fix is ours and rule 5 is untouched.
- **If it is a phone setting** (power saving on), nothing in the runtime changes it and rule 5 forbids asking
  users to. Then the question "which phone state does the claim describe" is the owner's, as it was for the
  cool-down gate: the claim's pre-registration would name the state it holds in (for example "power saving off,
  the default") and report the capped number beside it.
- **Worth one line in the kit regardless:** `PowerManager.isPowerSaveMode()`, `getCurrentThermalStatus()` at
  every gate poll (the kit has it only in the idle baseline), `getThermalHeadroomThresholds()` (API 35), battery
  per cent, the display refresh rate, and `scaling_max_freq` of policy 0 as well.

## 3. Levers

All ms per token, *estimated* unless marked, negative = faster. B = bit-identical (identity check at `-ub 1`,
`-ub 32`, `-b 512` required), N = declared numerics change. "q" = share of claim runs below 25.0 tok/s; R67's
pin-on runs had q = 4 of 8.

### 3.1 TEST list

| # | test | who, cost | decides | closing result |
|---|---|---|---|---|
| T-A | **Cap probe**, no model: a kit screen that logs once per second for six minutes `scaling_max_freq` (policies 0, 4, 7), `scaling_cur_freq`, headroom, thermal status, power-save flag, battery per cent and temperature, while it alternates every 60 s spinner on / static, and in the second half adds a three-thread spin load on cpus 4-7. The owner runs it a few times over two days (morning, evening, after use, deep-sleeping list on and off) and answers one question: was power saving or a performance profile changed between r26 and r27. | Tom S; owner 6 min per probe | §2.3; whether an uncapped session can be expected, and whether the app controls it | cap follows the spinner → app-side, keep static, price the official screen; cap follows the power-save flag → owner decision on the claim's state; cap differs between probes with every recorded field equal → say "not predictable from the app", and the claim's P is multiplied by the observed share of uncapped probes |
| T-B | **Who preempts.** Per run: worker `nice`, policy and cgroup (`/proc/self/task/<tid>/cgroup`, `sched_getattr`), and around the decode `/proc/stat` per-cpu rows and `/proc/loadavg` if readable (unverified), plus `c2-sched` per token in the profiled runs so that bursts show. | Tom S; rides | whether type A is steady competition (priority can help) or short high-priority bursts (it cannot) | per-cpu `irq`/`softirq`/`system` time on cpus 4-7 explains the wait → kernel work, L2 closed, look at I/O we cause; `user` time of others → L2 first |
| T-C | **Model-switch settling.** In the rehearsal, one block per model twice (O G O G), ten gated runs per block; score type A by position after a load. | rides | whether type A is a property of the first minutes after a switch | no type A after position 5 in any block and at least 3 before → state it as a position effect; the claim layout then decides q, and the lead decides whether a claim protocol may differ from r16's warm-up count |
| T-D | **smaps fix** (§1.4): match the mapping by inode; until then use `RssFile`. | Tom, minutes | T1's split | done when `model` Rss equals `RssFile` within 50 MB |
| T-E | **Does the official benchmark screen animate?** Read the app's benchmark UI code. | Tom, desk | whether campaign 1's "plain run" carries the spinner cost the kit just removed | static → nothing; animated → propose the static screen for the shipped benchmark as an app change (lead, owner) |

### 3.2 TRY list (ranked by expected effect on a claim)

| rank | lever | who, work | OLMoE | Granite | label | generic? | needs |
|---|---|---|---|---|---|---|---|
| L1 | **Touches through the executable mapping during the gate wait** (Y1 b restated with R67's data): a helper pinned to a little core reads one byte per page of the wanted set every 10 to 15 s of the wait and once at its end; no advice calls, no `mincore` per pass. Each touch maps the page and sets its reference bit, which is the case the 5.4 executable class protects. Endpoints: MB missing and `RssFile` at run start, loss per second of wait, then tok/s in runs after waits of 60 s or more. | Tom S to M | clean 0; after a 90 s wait 0 to −3 (pos 18, 33: 2.9 to 3.8 above clean); q −0.1 to −0.3; TTFT −0.4 to −1.5 s after long waits | same sign, smaller (its decode tracks missing MB weakly) | B | yes: kernel behaviour | nothing; exec on in both arms, touch as the factor, waits of fixed length or loss scored per second |
| L2 | **Worker scheduling priority** (new): the pool's workers call `setpriority(PRIO_PROCESS, tid, −10)` (try −16 as a second level) at pool start and restore on exit; kit line with errno and the value read back. AOSP's `init.rc` sets `setrlimit nice 40 40` (android-15.0.0_r1; fetched 2026-11-18 through a summarising tool, so check on the phone), which lets an app thread go below 0; and Flutter's own engine sets its raster thread to −5 by `setpriority` in every Android app, with a fallback to −2 "depending on the OEM" (flutter engine `android_shell_holder.cc`, mirror, search summary). Second arm if cheap: an ADPF hint session for the three worker tids (`APerformanceHint_createSession`, report each token's duration); `getPreferredUpdateRateNanos` = −1 means unsupported and closes it. | Tom S (nice), S to M (ADPF) | clean 0; type-A runs 0 to −13 (they are 13 to 17 above clean); q 0 to −0.15 | same | B | yes | T-B in the same kit; kill: rq wait in the decode not lower by half in runs where it exceeds 100 ms without the lever (needs enough such runs; otherwise "not testable this session") |
| L3 | **`MASS=900` re-test on OLMoE** (memo-29 Y3; built, inside the budget at both settings). The cap changes its price: the expert op is compute-bound, so its gain shrinks by about the clock ratio. | lead; no build | −1.0 to −1.7 at the 2,419/2,841 clocks (memo-29's −1.1 to −2.0 net × about 0.85); turns a 24.98 into about 25.7 | n/a (Granite has its own setting) | N (12.0927 / 12.0976 against 12.3003) | rule yes | lead's reading of R63 §4; R62's extra decode faults in the skip arm must be re-read with exec + pin on |
| L4 | **Outside-graph pipelining and graph crumbs** (memo-29 Y6, Y5): 0.65 to 0.85 ms per token is outside the graph (*device*, memo-29 §1.1). | Tom S to M | −0.3 to −0.9 together | −0.5 to −1.5 | B | yes | r27's block-mode profile (T3) is in the report (`_profb` runs); Bonnie's pricing |
| L5 | **Quiet gate** (new, unconventional): before a run the pinned workers spin for 100 ms and read their own run-queue wait; if it is above a threshold the run waits (bounded, like the cool-down gate). | Tom S | q −0.1 to −0.2 if type A is a state that passes | same | B for numerics; **changes the protocol** | yes | T-B, T-C; and the owner's word, because a claim run would then select its state by one more gate than r16's |
| L6 | **Hold to completion** as a stop-gap if L1 is not ready: let the read-back run until 99 % is back instead of four rounds. | Tom, a constant | after 90 s waits 0 to −2; costs 1 to 3 s of CPU before the run (heat) | same | B | yes | superseded by L1 if L1 works |
| – | **Hedged chunks** (idea only): when a worker is descheduled, another recomputes its claimed chunk into scratch and commits it; bit-identical because the computation is deterministic. Bounds a type-A stall by a chunk, but a late writer needs a commit protocol that is easy to get wrong. | Tom L | type-A runs up to −10 | same | B | yes | only if L2 is closed by T-B and type A stays |

Not levers but conditions: pin (0010) and the exec mapping stay on; the static screen and no OpenCL stay.

### 3.3 Frontier check, 2025 to 2026 (searched 2026-11-18)

- **llama.cpp**: the MoE work of 2026 is GPU-side expert caching. PR #26563 (opened 2026-08, closed; CUDA only,
  hot experts copied to VRAM, cold on the CPU, output not bit-identical; fetched), successors #26824 (closed) and
  #30059; #29887 "GPU cache for host-resident MoE experts" merged 2026-10-03 and #30966 (CPU share of cache
  misses) merged 2026-11-07 (search summaries, not opened). PR #21067 overlaps the next layer's weight copy to
  the GPU with compute (blog of 2026-04-28, fetched). None applies to a CPU-only phone; the idea that survives
  the translation is memo-21's "warm the next op's weights from an idle core", already behind T7.
- **MobileMoE** (arXiv 2605.27358, 2026-05-26; HTML fetched, first 100,000 characters): an ExecuTorch kernel
  that fuses top-k, dispatch, gate + up projection as one GEMM per expert, activation, down projection and the
  weighted sum into one op; int4 weights with int8 dynamic activations; 53 tok/s at 256 tokens of context for a
  0.9 B-active / 5.3 B-total model on a Snapdragon 8 Elite with four threads. It says nothing on thread
  placement, paging or preemption. Our runtime already fuses the expert pair (`MMID_FUSE`); its remaining idea,
  the weighted sum inside the op, is memo-29's Y5. Int8 activations are a numerics change with a new kernel:
  not cheap.
- **ZipMoE** (2601.21198), **FlashMoE** (2601.17063), **SliceMoE** (2512.12990v3): expert caching against SSD
  I/O (abstracts and search summaries). They address the regime R55 to R58 were in, not a resident model.
- **Android 17's memory daemon** (source.android.com, mmd) moves zram maintenance to idle time; not on this
  phone (Android 15).

Nothing in this search offers a cheap, bit-identical millisecond on a resident OLMoE that memo-29 does not
already list. The new items of this memo come from R67's own counters (L1's restatement, L2, L5).

## 4. The claim round

### 4.1 What R67 says about the draft rule (*estimated* by resampling *device* runs; `ebony_claimsim.py`)

| pool resampled (10 draws, 200,000 times) | P(median ≥ 25.0) |
|---|---|
| the 8 pin-on scored runs | 0.57 |
| the 4 pin + exec runs | 0.61 |
| all cells, positions 19 to 22 and 27 to 34 (no model switch just before) | 0.74 |
| the 8 clean runs only | 1.00 |

Two-part model (a run is in the tail with probability q, otherwise drawn from the clean runs made faster by d
ms per token):

| q | d = 0 | d = 0.5 | d = 1.0 |
|---|---|---|---|
| 0.50 | 0.34 | 0.38 | 0.37 |
| 0.375 | 0.64 | 0.69 | 0.69 |
| 0.25 | 0.88 | 0.92 | 0.92 |
| 0.125 | 0.98 | 1.00 | 1.00 |

These are 8 to 16 runs of one evening, at an uncapped clock, with the bench condition; the true uncertainty is
wider than the table. **Limit of this model (seen after reading Bonnie's draft):** it puts every tail run at
20 tok/s, so a margin d shows no effect. With the real values, resampling the 8 pin-on runs shifted by d gives
0.56 / 0.70 / 0.74 / 0.87 at d = 0 / 0.5 / 1.0 / 1.5 ms (her `bonnie_ppass.py`; I reran it), because a mild tail
run at 24.3 to 24.9 as the 5th value still passes when the 6th is a clean 25.9. The reading that survives: the
cap decides first (a 2,035 session fails whatever else holds), then q and the margin together: fewer tail runs,
and the remaining mild ones close enough to the line.

### 4.2 Design points

1. **Positions fixed in the pre-registration, no discards.** Type A sat in runs 3 to 5 after a model switch
   (§1.2). A rule that names the claim runs after seeing them, or that drops "disturbed" runs, would be a best-
   run claim. The layout (which model first, how many warm-ups, which ten runs) is written before the session and
   is r16's unless the owner agrees to a change; if a change is wanted (more warm-ups, a quiet gate), it is
   declared as a protocol change against campaign 1 and measured in the rehearsal first.
2. **N = 10 as in campaign 1**, so the two claims are the same kind of statement. A larger N does not help while
   q is near 0.5; a smaller q does.
3. **Heat inside the block.** In b3 the gate was met for runs 3 to 6 (waits 0 to 35 s) and timed out for runs 7
   and 8 (headroom 0.71, 0.77), which are 23.67 and 19.58. Ten runs in a row will have more timed-out waits at the
   end than R67's eight. L1 is the lever for exactly those runs; without it, expect the last third of a block in
   the tail.
4. **Both models in the session** (owner's rule), Granite reported as a full subject under the same rule even
   though nobody expects 25 from it (best clean run 23.55, *device*). Order fixed beforehand. If the claim block
   comes first after the session's first load it meets the settling tail; if it comes second it meets a switch.
   T-C tells which is worse; do not choose after the fact.
5. **With and without the bench condition: two pre-registered sessions**, the claim being that both medians are
   at least 25.0, order fixed (I would run the one without the deep-sleeping list first: it is the harder one and
   it is the one a user's phone resembles).
6. **The cap is recorded, not a void condition**, unless the owner chooses a state-conditioned claim; then the
   state is something the kit reads before the first run and the capped result is still reported. A rule that
   voids capped sessions and keeps uncapped ones would select the favourable phone state after the fact.
7. **A stated P(pass) in the pre-registration**, from the rehearsal, with the go rule written now: claim only if
   two rehearsal sessions in the claim's layout (one without the bench condition) each have at most 2 of 10 runs
   below 25.0 and an uncapped clock in at least 9 of 10 runs. With R67 alone I put P(pass) of a claim session at
   **0.35 to 0.6 if the clock is uncapped, and near 0 if it is not**; I cannot put a number on the clock
   (one uncapped session in about twelve, with unknown cause).
8. **Identity on the phone** as in R67: the 32 tokens equal to the off arm's in every claim run.

### 4.3 Proposed next round (for the lead)

A rehearsal, not a claim: kit with T-A (as a separate button), T-B, T-D riding; layout O G O G, two warm-ups
and ten gated runs per block; pin and exec on throughout; factor 1 = L1 (touches during the wait), factor 2 =
L2 (priority), 2 × 2 inside quads as in R67, the last two runs of each block plain. Then the same kit once more
without the deep-sleeping list. `MASS=900` (L3) only after that, as the margin, if the lead reopens it.

## 5. What I could not verify

- The cause of the cap change, the limiter's identity, and whether any app API moves it (§2).
- Who preempts the workers in type A; whether negative nice is granted on this phone (AOSP says the limit allows
  it; the phone's errno decides); whether ADPF is supported here.
- `init.rc`'s line was read through a summarising fetch of base64 text; the same fetch quoted a `memlock` line
  that does not match the 64 MiB the kit read on the phone, so treat the quote as unconfirmed.
- The path-mismatch explanation of the smaps field (inferred from the sizes; the code was not traced).
- llama.cpp PR states other than #26563 are search summaries. MobileMoE was read to 100,000 of 163,000
  characters.
- That type A follows model switches: three switches, one session.

## 6. Sources (fetched or searched 2026-11-18)

- Android Performance Hint API: https://developer.android.com/games/optimize/adpf/performance-hint-api ;
  https://source.android.com/docs/core/perf/performance-hint-api (search summaries)
- AOSP `init.rc`, android-15.0.0_r1:
  https://android.googlesource.com/platform/system/core/+/refs/tags/android-15.0.0_r1/rootdir/init.rc
- Flutter engine, Android thread priorities (`setpriority` −5 for the raster thread, fallback −2):
  https://flutter.googlesource.com/mirrors/engine/+/2dfcaaa7e2c4aa762f2a7f5228127d64f2266e1a/shell/platform/android/android_shell_holder.cc
  (search summary of a mirror snapshot; the current upstream file was not read)
- Samsung power saving "limit CPU speed to 70 %": Samsung community threads (search summaries), for example
  https://r2.community.samsung.com/t5/Galaxy-S/One-of-the-best-features-added-for-S21FE-on-OneUI-7-and-it-s-not/td-p/19538517
- Samsung performance profile "Light": https://9to5google.com/2022/08/15/galaxy-z-flip-4-fold-4-battery-profile/ ;
  https://www.phonearena.com/news/samsung-galaxy-z-fold-4-light-performance-profile-battery-life_id141952
  (search summaries)
- Samsung performance SDK thread:
  https://forum.developer.samsung.com/t/samsung-performance-sdk-only-has-effect-on-the-most-powerful-cores/12621
  (search summary)
- kswapd affinity option in a Qualcomm-derived tree:
  https://gitlab.e.foundation/e/devices/android_kernel_sony_msm8994/-/commit/57f848a9a900e1a3bc549e6d3760b135f4d0277e
  (search summary)
- llama.cpp PR #26563: https://github.com/ggml-org/llama.cpp/pull/26563 (fetched); release notes for #29887,
  #30966: https://newreleases.io/project/github/ggml-org/llama.cpp/release/b12191 (search summary); PR #21067:
  https://am17an.bearblog.dev/prefetching-weights-in-llamacpp/ (fetched)
- MobileMoE: https://arxiv.org/html/2605.27358 ; ZipMoE: https://arxiv.org/pdf/2601.21198 ; FlashMoE:
  https://arxiv.org/abs/2601.17063 ; SliceMoE: https://arxiv.org/html/2512.12990v3
- Android memory daemon: https://source.android.com/docs/core/perf/mmd (search summary)
- Repository: `reports/device/r22` to `r27` raw reports, campaign 1's `reports/device/r13` to `r16`; R67's
  R-log entry; memos 23, 28, 29, 30; `lab/app-c2/lib/research.dart:282`;
  `lab/llama.cpp-c1r27/ggml/src/ggml-cpu/ggml-cpu.c:2519` (`ggml_cpu_mmid_claim`).

## 7. Review of Bonnie's draft (`bonnie-draft.md`, read 2026-11-18 after §1 to §6 were written)

We worked from the same report without seeing each other's text. Where the two drafts agree independently I say
so briefly; the rest is where I think her prices are off, and where mine were.

### 7.1 Agreed independently (two readings, same result)

- The three classes of slow run, the contention signature (run-queue wait, involuntary switches, the other two
  workers spinning), that the app's own threads are not the competitor, that pin does not prevent it.
- `smaps`: the model's numbers are in `file_rest`; mapped ≪ resident after a timeout.
- The clock: first uncapped session; not a function of headroom or battery temperature; P(pass | capped) ≈ 0.
- P(pass | uncapped, today's build) about 0.55 (hers) against 0.35 to 0.6 (mine).
- Next round: a rehearsal in claim format, twice (with and without the deep-sleeping list), not the claim.
- Her §2.3 (a zero wait is the gate's first poll right after a run that ended below the gate, 10 of 10) is new
  to me and I checked it against the table in §1.1: correct. It also closes the lead's caveat on the exec
  column from the other side, and it means a claim block alternates waited and back-to-back runs.

### 7.2 Where I was wrong and she is right

- **Margin.** My §4.1 model made a margin look worthless; her mixture and shifted resampling are the better
  model (corrected in §0 item 4 and §4.1 above).
- **History.** I wrote "never above 2,150 since 2026-11-07"; she found two isolated uncapped readings in r5 and
  r12 (confirmed, §2.1).

### 7.3 Where I price differently

1. **P(uncapped next session) = 0.7 with the bench condition, 0.5 without (her §3.2).** Too high, in my view,
   and the argument has a gap. Her case is "23 capped sessions, then an uncapped one on the day two idle-power
   causes were removed; a cooler idle phone fits all of these". But her own next bullet, and my §2.2, show the
   cap is **not** a function of how cool the phone reads: r26 stayed at 2,035 at lower headroom and lower
   board temperatures than r27 had while uncapped, and r27 returned to the maximum 92 s after headroom 1.19. If
   heat is not the path, the spinner and the OpenCL driver have no known path to the cap, and "the day we
   removed them" is one coincidence among at least four changes of that day (also the deep-sleeping list,
   internal storage, an evening session). I would write 0.45 (0.2 to 0.7) with the bench condition and the same
   without it, since no mechanism ties the list to the clock either. Both numbers are judgements; the lead
   should carry the range, and T-A (my §3.1) or her block 4 is what replaces them. Consequence: her overall 0.4
   becomes about 0.25.
2. **"× 0.9 for a clock state seen in three sessions out of three" (her §5.3).** Three of three with an
   unknown cause supports about 0.8 at most, and less if the three sessions are on consecutive days in the same
   room. Small, but it is the factor that decides whether the go rule delivers her 0.8.
3. **L2, the 1.28 ms per token per GB cached-but-unmapped (r 0.89, n 11).** The fit is two clusters: eight runs
   at 0.1 to 0.2 GB unmapped and three after a timeout. In those three, unmapped GB, missing MB (150, 251, 232)
   and swapped-out app memory (165, 125, 320 MB) all move together, so the slope cannot be assigned to
   "unmapped". Pos 27 is the counter-example inside her own table: 1.1 GB unmapped, 251 MB missing, +0.2 ms.
   The decode's minor faults differ by only 2,000 to 3,500 between timeout and clean runs (fault-around maps 16
   pages at a time and the prefill does most of it: 43,000 to 62,000 prefill minors), which at any plausible
   cost per fault is under 0.1 ms per token. I think the penalty is the missing MB and the swap-ins (pos 18:
   4,246 decode majors), not the mapping. **This matters for the design of the lever:** an end-of-wait touch
   through the mapping repairs the mapping and, as a side effect, reads the missing pages; if the cost is the
   missing pages, then what matters is that the touch pass is complete and last, and that the pages stop
   leaving during the wait. So the factor should have three levels, not two: today's hold; touch pass at the
   wait's end (hers); touches every 10 to 15 s from a little core plus the end pass (mine, L1). Her price for
   the end pass (−1.4 on a timeout run, 0 to −3.3) I accept as a band; the "0.1 to 0.4 s CPU" cost is right only
   when the hold has already read everything back, otherwise the pass reads from storage and costs what the
   hold costs today (1.4 to 2.6 s, *device*).
4. **"Touch the app's own hot anonymous ranges" inside L2.** Memo-29 §1.3 found that swap before the run does
   not predict decode time and parked the anon hold as a TTFT lever; her draft brings it back without saying
   what changed. R67 does offer a reason (pos 18's swap-in majors inside the decode), on one run. It should be
   its own arm or be left out; bundled into L2 it cannot be attributed.
5. **L1, `MASS=900`: −1.3 (−0.7 to −2.0).** Close to mine (−1.0 to −1.7). Three reservations. (a) The clock
   scaling is applied to the net; the dense excess of the skip arm (+0.5 ms on OLMoE, memo-29 §1.2) is on
   byte-bound ops and may not shrink with the clock, which moves the centre to about −1.1. (b) "Also reads
   fewer pages after a timeout" is asserted; R62 measured the opposite sign in faults (5,310 decode faults in
   the skip arm against 1,139, median), and that open item was the reason R63 §4 parked the lever. Until it is
   explained, the lever's effect on tail runs has an unknown sign. (c) Its tok/s record is +1.71 ± 0.70,
   +0.15 ± 1.00, +2.02 ± 1.09: one of three rounds is a null, so P(the lever delivers ≥ 0.7 ms in the claim
   session) is perhaps 0.7, and "L1 alone gives 0.79" should be read as 0.7 × 0.79 + 0.3 × 0.53 ≈ 0.71.
6. **L3, strict pin −0.3 (0 to −0.7).** The basis is a slope over seven clean runs whose two fastest are the
   back-to-back zero-wait runs (pos 21, 30); stay share and zero wait are confounded there. And R67 now gives
   the downside a shape: in the contention runs the workers did move among cpus 4-7 (pos 15: thread 0 started
   graphs on cpus 4, 5, 6 and 7), which is the escape a strict pin removes. I price it 0 (+0.3 to −0.5) clean
   and negative in a contention episode, and would not put it into a claim build before a contention run has
   been seen with it. As a rehearsal arm on Granite, fine.
7. **L4, priority: "not verified that an app may lower nice".** I can add two sources: AOSP's `init.rc` sets the
   nice limit to 40 for everything init starts, zygote's children included, and Flutter's engine calls
   `setpriority(…, −5)` on its raster thread in every Android app, with a fallback for OEMs that refuse (§3.2,
   §6). So the call is likely allowed; the phone's errno decides. Her "converts perhaps a third" has no basis
   either way, like my 0 to −0.15 on q; what would give one is knowing whether the competitor is a normal-
   priority thread (T-B).
8. **Rehearsal layout (her §5.1): four blocks of ten per model, levers by block.** Two problems. Levers
   compared between blocks an hour apart is what rule 8 warns against, and R67 itself shows why: the
   contention episodes cluster in time and by distance from a model switch, so "block 2 against block 1" would
   hand a lever whatever the phone did in those ten minutes. And about 92 runs is 1.5 to 2 hours on a phone
   whose heat decides the waits. I would keep **one** claim-format base block per model (that is the rehearsal
   proper and gives the class mix) and put the levers into quads as in R67. Her block 4 (spinner back on) is a
   good idea and cheaper than my separate probe for the first answer; its weakness is that six runs at the end
   of a session cannot show a slow limiter, and a null there does not clear the spinner. I would run it in
   session A at the end and in session B at the start.
9. **Go rule (her §5.3) against mine (§4.2 item 7).** Hers asks for a median of 25.7 in the session without the
   bench condition; with a clean level of 25.24 that can only be met with `MASS=900` in the claimed
   configuration, so the rule quietly makes the claim a numerics claim. That may be right, but the lead should
   decide it in the open: a bit-identical claim (pin + exec + touches) and a numerics claim are different
   statements about the campaign's question, and the budget spent (0.38 of 0.586) should be in the sentence
   that reports it.
10. **Claim session = the one without the deep-sleeping list (her §5.2).** Agreed, and stronger than my "both
    must pass, harder one first".

### 7.4 What both drafts leave open

The clock state's cause; who preempts; whether the contention belongs to r27's first session (6.2 GB copied and
hashed shortly before, her §6) or to model switches (my §1.2): the two readings predict differently for the
rehearsal (hers: no contention in a session without a copy; mine: contention in runs 3 to 5 after each switch),
so the rehearsal's layout O G O G decides between them at no extra cost.
