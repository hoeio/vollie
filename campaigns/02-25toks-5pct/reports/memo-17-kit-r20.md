# Memo 17: research kit r20, campaign 1's kit r16 protocol on OLMoE, and a keep-warm lever

Agent model: claude-opus-5-5

Date 2026-11-14. Author: Tom (code). **Nothing here ran on a phone. No speed is stated for the phone.** The APK was
built and inspected on the laptop. Labels: *device* = read from campaign 1's r16 report
(`reports/device/r16_2026-11-08.json`), *host* = Apple M4 Pro beside another perplexity run, relative only,
*estimated* = arithmetic. Owner's standing instruction (2026-11-14): campaign 2 starts from Vollie 1; this kit is kit
r16 with OLMoE in it, and every addition is a delta that is reported apart.

## 1. Kit r16's protocol, reconstructed (the reference)

**Sources.** Campaign 1's app source is not under version control. `../vollie/vollie/lib/research.dart` (production
2.3.0+45) is the r16 kit: it says `kit = 'r16'` and differs from the shipped r16 build (2.2.21+44, R48 amendment 2)
only in the `app` string and in where three defaults come from (R49's production note: the fallbacks of tiny-op,
expert I/O and threads moved from saved preferences into code; the r16 report's `saved` block,
`tiny 262144, expert_io 1, write_threads 3, read_threads 3`, is the owner's saved preferences, so the values are the
same). `../vollie/lab/logs/r48/research.dart.pre-r48` (r15) shows the r16 changes. Gate: campaign 1's
`Assistant._coolDown` (`assistant.dart`). Prompt reuse: llamadart 0.15.6 in `third_party/llamadart`. Every r16 timing
quoted below is *device*.

**Run list (21 runs, in this order).** Every run goes through the same `measure`: set threads in place (3 write,
3 read; no reload), reset every knob to the served default and set the run's env, the gate, the optional pause, then
the reply.

| pos | phase | what | env on top of the served default |
|---|---|---|---|
| 1 | baseline | official 32 (`official 32 · 3t`) | none |
| 2 | warmup | story 200 (`story 200 · profiled`, "Write a short story about a lighthouse.", 200 tokens; cpu4/cpu7 clock every 8 received chunks) | profiled |
| 3 | ref | story 200, the same | profiled |
| 4-7 | group 1 | plain, profiled, profiled, plain (official 32) | profiled = `GGML_CPU_PROF=1 GGML_CPU_AB_TOKEN=1 GGML_CPU_AB=knob:MMID_FUSE+CHUNK2:1:1` (an A/A label, no change to the arithmetic) |
| 8-11 | group 2 | the same; **20 s idle pause after the gate** before pos 8 | |
| 12-15 | group 3 | the same | |
| 16-19 | group 4 | the same; **20 s pause** before pos 16 | |
| 20-21 | — | plain ×2 | |

**The gate (how r16 got its mix of waits).** `Research.coolOfficial = true`, and `measure` defaults to `cool: true`,
so **every** run, the baseline and the stories included, went through `_coolDown(0.65)`:

```
until = now + 90 s
loop: h = getThermalHeadroom()            (one platform call)
      if h == null or h < 0.65 or now > until: return h
      wait 5 s
```

So it is "below 0.65, or 90 s", read every 5 s, the deadline tested after each read. There was **no end-of-run
headroom read**, so consecutive reads were always ≥ 5 s apart and Android never returned NaN. The recorded `cool_ms`
values are multiples of the 5 s poll plus a few ms per read: 0-1 ms (first read already below 0.65), 40,046 /
40,049 (8 polls), 60,071 / 60,079 (12), 65,064 (13), 80,094 (16), 90,087-90,106 (timeout). In r16:

- **9 timeouts:** pos 1-8 while the phone cooled from 0.870 to 0.661, and pos 17 (0.652).
- **6 passes after 40-80 s:** pos 9, 12, 13, 15, 18, 20.
- **6 passes at the first read:** pos 10, 11, 14, 16, 19, 21. Their idle before the run was 15 to 86 ms (pos 16:
  20 s, its pause).

Once the forecast sat at 0.636-0.652, a 5-s Granite run either left it just under 0.65 (immediate pass) or just over
(a wait until it dipped). The headroom series inside a wait was not recorded.

**Settings.**
- Served default compiled into the app: `GGML_EXPERT_SKIP=117`, `GGML_HEAD2=2048`, `GGML_HEAD2_Q=q4_0`,
  `VOLLIE_SAMPLER_BYPASS=1`.
- Saved: `GGML_TINY_OP=262144`, `GGML_EXPERT_IO=1`, 3 threads.
- Every other knob was set to the empty string. 0001 reads `""` as unset, which means its compiled defaults:
  `MMID=3`, `Q3K_X4`, `Q3K_VEC`, `Q3K_ILP`, `Q3K_KERN`, `MMID_FUSE`, `MMID_CHUNK2`, `SSM_INPLACE`, `CONV_RING`.
- Repack on: `canCopyWeights` gives 7.7 GB − 0.7 GB ≥ 5 GB.
- **`GGML_REPACK_RELEASE` was never set.** The app does not set it anywhere; the kit only clears it at the end.
- KV q8_0 with flash attention, context 4096.

**Prompt reuse.** llamadart's default is `reusePromptPrefix = true`. For an identical prompt the shared prefix is the
whole prompt, and llamadart then clears the context memory and decodes the full prompt
(`reusedPrefix >= nTokens && !exactStateLoadMatch` → `_decodeAndCacheFullPrompt` → `_clearContextMemory`). That is why
every r16 run shows `cached_prompt_tokens 0` and a full prefill. Kits r17-r19 set `reusePromptPrefix: false`, which
clears the memory and decodes the full prompt by the other branch. **For the benchmark prompt the two are the same
work.**

**Between runs.** The model stays loaded for the whole kit. There is no reload at the start: the model is as the app
loaded it, with the load's one-token "Hi" warm-up. Threads are set in place. The head-copy warm-up runs only when the
anonymous copies change, which never happened in r16. The context is cleared by each prefill. Nothing else touches
the phone. At the end the knobs are reset, the model is unloaded and reloaded, and the JSON is written.

**R49's claim method.** Pre-registered in R48: the median of the 10 plain runs (`phase on`, "official 32 ·
default"), at pos 4, 7, 8, 11, 12, 15, 16, 19, 20 and 21. The baseline (pos 1), the stories and the profiled runs are
excluded. Result: 20.51 tok/s.

## 2. Kit r20 = that protocol on OLMoE, then two clearly separated additions

APK `lab/dist-c2/vollie-c2-1.20.0+20-research-r20.apk`:
- 17,164,163 B, sha256 `7981aaa682a703752c4b82bd63a269c453368c490bc619da1ce6009599883b93`;
- package `com.insaneutils.vollie.c2`, versionCode 20, debug key: installs as an update over r19 and keeps the model.

Build record `lab/dist-c2/BUILD-r20.txt`:
- native `620d2225c` + `patches/llama.cpp/0001-0003` + `patches/llama.cpp-c1-olmoe/0001-keep-warm.patch`;
- CPU module sha256 `f60f0fcc…`, libllama `454f88dd…`, both byte-checked against the APK;
- app patches 0001-0004.

The Dart snapshot carries `1.20.0+20`, `GGML_KEEP_WARM` and `story200`, and the CPU module carries the
`c2-keepwarm` line.

**Baseline configuration** (r19's `c1_ship`): `GGML_TINY_OP=262144 GGML_EXPERT_IO=1 VOLLIE_SAMPLER_BYPASS=1`, compiled
defaults, repack on by the app's rule, no `GGML_REPACK_RELEASE` (as r16). It is r16's shipping env minus Granite's
numerics: no skip 117, no head2, hence no head copy and no copy warm-up. "Profiled" adds r16's A/A label.

| part | pos | runs |
|---|---|---|
| **1, proto `r16`** | 1-21 | exactly §1's list with the baseline configuration: `ship` (baseline), `ship_story` ×2, groups g1-g4 (`ship`, `ship_prof` ×2, `ship`; 20 s pause before g2 and g4), `ship` ×2. r16's gate before every run |
| **2, proto `kw`** | 22-25 | r16's group 2 again (20 s pause, plain, profiled ×2, plain), gate as r16, **`GGML_KEEP_WARM=10`** in every run. The helper acts in the idle time after a run that had the knob on, so the waits before pos 23-25 are covered and the wait before pos 22 is not (an in-group control) |
| **3, proto `warm`** (a different protocol) | 26-48 | no gate, runs back to back. `ship` warm-up (unscored) + 10 `ship`; `io` warm-up + 5 with `GGML_EXPERT_IO=5`; `off` warm-up + 5 with every knob off (`c1_off`, R25's vanilla set), last because it is the slowest and hottest |

**What differs from r16, and why.**
- **Gate.**
  - The same rule, poll and limit, also before pos 1. r19's 10-minute first wait is gone.
  - Kept from r19: thermal reads spaced ≥ 1.1 s. A null reading is read again instead of meaning "cool" (r16 never
    saw one).
  - The headroom series is recorded.
  - Gated runs have no end-of-run read, as in r16.
- **Warm block headroom.** One read after each run (`heat_after`), which also stands for the next run's "before"
  (`heat_from`), so back-to-back runs get no 1.1 s spacing delay.
- **Residency.**
  - `mincore` of the model before and after each run (r19's counter), plus clock caps and the battery.
  - These add idle time between runs that r16 did not have: two counts of about 0.75 M pages each, *estimated* 0.2
    to 1 s together.
  - In the warm block the count after a run stands for the next run's "before".
- **Streaming.** r16's (llamadart's default batching) for every run; r19 used one chunk per token for profiled runs.
- **Prompt reuse.** `reusePrefix = false`; for the benchmark prompt this is the same full prefill as r16 (§1).
- **Report rows** carry `pos`, `proto` and `phase`. Untagged runtime lines printed during a run's wait (the keep-warm
  passes) are filed into that run's `c2` list.
- **Unchanged from r19:** the template (still llamadart's `<s>` fallback for `{{ bos_token }}`, memo-13 §6), the
  build and lib hashes, and the model hash after the last run.

## 3. Parser

`python3 -I tools/c2_research_parse.py <r20.json>` prints, on top of the earlier sections:

- **Part 1, position by position**, campaign 1's r16 run against r20's run at the same `pos`: tok/s, headroom at the
  gate (T = timed out), gate wait, idle before, prompt time, A78 clock at the end (+ caps), r16's `RssFile` against
  r20's model residency before → after, `Cached`, major faults. Then R49's ten plain runs, r16 median against r20
  median.
- **Every wait of a gated run:** model MB lost from the page cache between the previous run's end and this run's
  start, with keep-warm on or off in that wait, and the passes seen (count, CPU ms, longest pass, aborted). Then the
  medians with and without keep-warm.
- **The warm block apart:** per run tok/s, headroom before → after, A78 clock before → after, cap, residency.
  Warm-ups form their own group and are not scored.

`--r16` defaults to `reports/device/r16_2026-11-08.json`. `--selftest` covers the new parts. The output on the r16
and r17 reports is unchanged. Stories are left out of the text identity check (another prompt); the line
"r20: every 32-token run … gives one text" covers `ship`, `ship_prof`, `ship_kw`, `ship_kw_prof`, `io`, `off` and
the warm-ups.

## 4. Keep-warm (`patches/llama.cpp-c1-olmoe/0001-keep-warm.patch`, `GGML_KEEP_WARM=<s>`, off unless set)

**Built on campaign 1's parked code.**
- `GGML_MMID_TOUCH` (R39-R41) touched page heads inside the expert op. It was parked because the device page-latency
  model did not hold, so it never helped residency between replies.
- `EXPERT_IO` bits 1 and 4 use `MADV_WILLNEED` and a helper thread, and only act during a token.
- On a resident page `MADV_WILLNEED` does nothing (the kernel's readahead skips cached pages), so it cannot mark a page
  as used.
- The new patch therefore uses MMID_TOUCH's idea, one demand read per page, moved to idle time. It runs on a helper
  thread in the style of EXPERT_IO bit 4, with one change: the read goes through `process_vm_readv` (Linux/Android) or
  `vm_read_overwrite` (macOS), which return an error instead of a signal if the model has been unmapped. That way the
  helper never needs to know when the app frees the model.

**Mechanism.**
- Each graph start reads `GGML_KEEP_WARM` (the period) and registers the `*_exps` `MUL_MAT_ID` weights. OLMoE: 48
  ranges, 2,769 MB.
- A detached helper wakes every period. If no graph is running and the last one ended ≥ 1 s ago, it walks the ranges
  in steps of 1,024 pages:
  - `mincore` first, so pages already reclaimed are skipped (no flash reads, no fight for memory);
  - then one byte of every resident page through the safe read, which sets the page's referenced state so reclaim
    passes it over;
  - before each step it checks the busy flag and the graph counter, and stops the pass when a generation starts.
- Scheduling: SCHED_BATCH, nice 19, pinned to the CPUs with the lowest `cpuinfo_max_freq` (A55 cores 0-3 on the
  SD888), never the big cores it would otherwise inherit.
- One `c2-keepwarm` line per pass reports the thread's CPU µs.

**Duty cycle (period 10 s in the kit).**
- *Host*: 0.30 to 0.52 s of CPU per pass over 169,008 resident 16-KB pages.
- Phone: about 676 k 4-KB pages per full pass, one system call per 1,024 pages. *Estimated* 0.2 to 0.7 s per pass on an
  A55: **2 to 7 % of one little core**, and only while idle.
- The heat risk is that small A55 load. The kit measures it: CPU ms per wait, from the lines.

**Host checks** (*host*, `olmoe-Mi`, 4 chunks, `-t 4`, build options as memo-13 §7, KleidiAI and BLAS off):

| check | result |
|---|---|
| `llama-perplexity` per chunk, knob unset vs `GGML_KEEP_WARM=1` | identical at `-b 512` (8.9646 / 10.4244 / 9.2074 / 9.4783), `-ub 32` (8.9131 / 10.3435 / 9.0932 / 9.3987), `-ub 1` (8.8302 / 10.3747 / 9.1120 / 9.4183) |
| `-ub 1`, unpatched base build (`lab/llama.cpp-base`) run today | 8.8302 / 10.3747 / 9.1120 / 9.4183: equal. Memo-07's `-ub 1` row (8.9550 / …) is **not** what the base build gives today; `-b 512` and `-ub 32` match memo-07. Flagged for the lead; it does not affect this patch |
| official prompt, 32 greedy tokens (`-t 3 -ctk q8_0 -ctv q8_0 -fa on --repeat-penalty 1.1 --repeat-last-n 64`) | one text (md5 of the output file `29165b94…`, reply starts "The ocean is salty primarily due to a process known as "evaporative concentration."", as r19) for: unset, keep-warm, `c1_ship`, `c1_ship` + keep-warm, `EXPERT_IO=5`, `c1_off`, profiled `c1_ship` and profiled `c1_ship` + keep-warm (31 per-token lines each) |
| idle behaviour (`llama-server`, `GGML_KEEP_WARM=2`, two requests 9 s apart, `cache_prompt false`) | passes during idle only, each `mincore`-filtered: `resident_mb=2769 touched=169008`, 0.30-0.52 s CPU; the reply after the idle wait equals the one before |
| stop when a generation starts (14 requests at 3.0-3.5 s intervals) | 3 passes ended `aborted=1` at a request (at 858, 1,154 and 2,762 MB walked) |
| `flutter analyze lib test`, `flutter test test/kit_lines_test.dart`, `--selftest` | clean, 4 tests pass, OK |

On macOS nothing was purged (no sudo), so the host shows that the calls are made and filtered, not that a page was
saved from reclaim. **Not verified:** anything on Android. That includes whether the app may call `process_vm_readv`
(bionic lists it; same-process access needs no ptrace permission), the pass time on an A55, and whether the touched
pages survive a 90 s wait. If the call is refused, the lines show `touched=0` and nothing else changes.

## 5. Run time on the phone (*estimated*)

Inputs: OLMoE official runs took 12 to 21 s in r19 (cold); a 200-token story at 4 to 8 tok/s takes about 30 to 50 s.

- **Hot phone, every gate times out (r19's day):**
  - part 1: 21 × 90 s + 40 s of pauses + 19 × 21 s + 2 × 45 s ≈ 40 min;
  - part 2: 4 × 90 s + 20 s + 4 × 21 s ≈ 8 min;
  - part 3: 17 × 15 s + 6 × 25 s ≈ 7 min;
  - start and end (hashes, the 3 GB model hash, reload) ≈ 2 min;
  - **total ≈ 57 min.**
- **Phone that cools during the kit (r16's day):** about 25 to 40 min.
- **At the 60-minute limit in the worst case.** What to drop first, in order:
  1. the `c1_off` warm block (about 3 min);
  2. the `EXPERT_IO=5` warm block (about 2 min);
  3. then part 2.

  Part 1 is the protocol and is not shortened.

## 6. What could not be reconstructed from campaign 1's sources

1. **The exact 2.2.21+44 source.** It is not kept. The 2.3.0+45 file is used with the difference stated in §1. This is
   inferred, not checked byte for byte.
2. **pos 4 (an official run right after a story) shows 0 cached tokens in r16.** Under llamadart's reuse default the
   two prompts' shared template head could have been reused. Why it was not is not established. r20 avoids the
   question: reuse off, a full prefill by construction.
3. **The headroom inside r16's waits, and the page cache during them.** Neither was recorded. r16 has only the
   passing reading and process `RssFile` / `Cached` before and after each run, so the r16 side of the wait table has no
   residency numbers.
4. **What else the phone ran on 2026-11-08.** Not recorded.

## 7. What the owner does

1. Install `vollie-c2-1.20.0+20-research-r20.apk` over r19 ("Vollie C2"). The model stays.
2. Open the app, wait for the model to load, start the research kit, and leave the phone on the table with the screen
   on. Up to about an hour (§5).
3. Send back the report: the clipboard, or `research-r20-….json` next to the model.
