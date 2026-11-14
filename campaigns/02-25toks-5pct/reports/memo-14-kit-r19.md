# Memo 14: research kit r19, campaign 1's runtime on OLMoE

Agent model: claude-opus-5-5

Date 2026-11-14. Author: Tom (code). **Nothing here ran on a phone. No speed is stated.** The APK was built and inspected
on the laptop. Host checks (*host*: Apple M4 Pro, 3 threads, while a perplexity sweep ran) covered texts and line
formats only. Run times are *estimated*.

## 1. What was built

| item | value |
|---|---|
| APK | `lab/dist-c2/vollie-c2-1.19.0+19-research-r19.apk` (size and sha256 in `lab/dist-c2/BUILD-r19.txt`) |
| package, key | `com.insaneutils.vollie.c2`, Android debug key: installs as an update over r17/r18 and keeps the model |
| version | 1.19.0+19, kit `r19` |
| runtime | llama.cpp `620d2225c` + `patches/llama.cpp/0001-cpu-runtime.patch` (byte-identical to campaign 1's tiny-op patch, the r16 build) + `0002-granite-rollback.patch` (inert on `olmoe`) + `0003-sampler-bypass.patch`. No campaign-2 patch. Android build by campaign 1's `android/build_cpu.sh` (KleidiAI on, as r16 and r17/r18) through `tools/c2_build_kit.sh <tree> lab/app-c2 patches/llama.cpp` |
| OLMoE-specific runtime change | none needed (memo-13 §7: loads and runs, bit-identical with Granite's numerics off). `patches/llama.cpp-c1-olmoe/` is not created |
| app | `patches/app-c2/0001` + `0002` + **`0003-kit-r19.patch`** |
| parser | `tools/c2_research_parse.py` (r19 support, `--selftest` extended) |

## 2. Rows (`Research.runs` in `lib/research.dart`), all on the same CPU module, repack on

Knob names are campaign 1's (`docs/runtime-knobs.md`). Every run starts from every known knob unset; the row's env is
then set. "Unset" on this module means 0001's compiled defaults: `MMID=3`, `Q3K_X4=1`, `Q3K_VEC=1`, `Q3K_ILP=1`,
`Q3K_KERN=1`, `MMID_FUSE=1`, `MMID_CHUNK2=1`, `SSM_INPLACE=1`, `CONV_RING=1` (the last two are inert on OLMoE: no SSM
ops). `GGML_EXPERT_SKIP` and `GGML_HEAD2` are off unless set, and no row sets them.

| row | exact env |
|---|---|
| (a) `c1_off` | `GGML_TINY_OP=0 GGML_MMID=0 GGML_Q3K_X4=0 GGML_Q3K_VEC=0 GGML_Q3K_ILP=0 GGML_Q3K_KERN=0 GGML_MMID_FUSE=0 GGML_MMID_CHUNK2=0 GGML_SSM_INPLACE=0 GGML_CONV_RING=0 GGML_EXPERT_SKIP=0 GGML_EXPERT_IO=0` (R25's "vanilla" set: stock arithmetic and stock op paths) |
| (b) `c1_ship` | `GGML_TINY_OP=262144 GGML_EXPERT_IO=1 VOLLIE_SAMPLER_BYPASS=1`: r16's shipping settings without skip 117 and without the tied head2 |
| (c) `c1_ship_io` | `GGML_TINY_OP=262144 GGML_EXPERT_IO=5 VOLLIE_SAMPLER_BYPASS=1`: bit 1 (routed read-ahead) + bit 4 (next-layer guess on a helper thread, any core), campaign 1's best measured setting (5 → 12.0 against 1 → 10.6 tok/s, `docs/runtime-knobs.md`). Bit 2 (`MADV_RANDOM`) lost in R12 and is not used |
| profiled variants | the row's env + `GGML_CPU_PROF=1 GGML_CPU_AB_TOKEN=1` + `GGML_CPU_AB=knob:MMID_FUSE+CHUNK2:0:0` for (a), `…:1:1` for (b) and (c) (r16's A/A label; it changes no path) |
| all rows, at load | `GGML_REPACK_RELEASE=1`, set only around the kit's model load (see below) |

**Repack release is kit-wide, not a row.** It acts once, when the model loads. Making it a row would need a reload at
each switch (two per pass, each followed by a run on a disturbed cache). Memo-13 §3 prices its steady-state value at
0 MB. The kit therefore reloads the model once at its start with the knob set and unsets it after the load. The
`res_*.rest_mb` field shows whether the 235 MB of repacked sources left the cache.

Order (one spec per line; ids repeat, so `rep` is numbered across the list):

| # | spec | kind | reps |
|---|---|---|---|
| 1 | `cold` (env of (a), unscored; first timed run, after the kit's reload) | official32 | 1 |
| 2-4 | `c1_off`, `c1_ship`, `c1_ship_io` | official32 | 3 each |
| 5-6 | `c1_off`, `c1_off_prof` | official32, profiled32 | 2, 1 |
| 7-8 | `c1_ship`, `c1_ship_prof` | official32, profiled32 | 2, 1 |
| 9-10 | `c1_ship_io`, `c1_ship_io_prof` | official32, profiled32 | 2, 1 |

That is 19 timed runs: 5 plain official runs and 1 profiled run per row, plus `cold`. Greedy texts are kept whole.
(a), (b) and (c) must give one text (host: they do, §5).

## 3. The cool-down gate

- **Spacing.** `DeviceProfile.thermal()` waits until at least 1.1 s have passed since its previous call. Android's
  `getThermalHeadroom` returns NaN when polled more than once a second. That NaN was the r17/r18 bug (memo-13 §4).
- **Rule.** Before every timed run the gate reads the headroom until it is below 0.65 (r15/r16's threshold), at most
  **10 minutes before the kit's first timed run** (`cold`) and **90 s before every other run**. A null reading means
  "unknown, read again" (the next read comes 1.1 s later), never "cool". A real reading above the gate waits 5 s.
- **Recorded per run:**
  - `cool_ms`;
  - `heat` (the last real reading; null if none);
  - `heat_series` (`[ms since gate start, headroom or null]` per read);
  - `gate_timeout` (the wait ran out);
  - `gate_unknown` (no real reading at all);
  - `gate_first`.

  The end-of-run read `heat_after` is kept; the spacing makes it safe.
- **Model hashing.** The model is not hashed before the timed runs. With the `.verified` marker present the app does
  not hash at start (`model_store.dart`, unchanged), and the kit hashes the file after its last run, as in r17/r18.
- **The phone.** The kit changes no phone setting and asks nothing of the user.

## 4. Report schema additions (on top of r18's, memo-12 §5)

| where | key | meaning |
|---|---|---|
| report | `tok_fields_c1` | names of campaign 1's compact per-token tuple: `ab tok us head router majflt minflt mhz4 mhz7 nodes barriers kern fuse chunk2` |
| report | `gate` | `{below, wait_ms, first_wait_ms}` |
| report | `load_env` | `{"GGML_REPACK_RELEASE": "1"}` |
| report | `read_ahead_kb` | `/sys/block/sd{a,b,c}/queue/read_ahead_kb` ('' if unreadable) |
| run | `tok_format` | `c1` when the per-token tuples are campaign 1's format |
| run | `heat_series`, `gate_timeout`, `gate_unknown`, `gate_first` | §3 |
| run | `res_before`, `res_after` | page-cache residency of the model file: `{experts_mb, rest_mb, total_mb, ms}`. `mincore` over the app's own read-only `MAP_SHARED` mapping of the file (no data read). Experts = the 48 routed-expert tensors of `olmoe-Mi.gguf` (32 byte ranges, 2,768,240,640 B, from the frozen file's tensor table, constant `Research.expertRanges`); rest = everything else (dense, repacked sources, head, embeddings). Counted after the gate and right before the run, and right after the run |
| run | `cap_before`, `cap_after` | `cap=<policy4>/<policy7>` from `scaling_max_freq`, MHz: the thermal cap |
| run | `mem_before`, `mem_after` | unchanged: MemAvailable, Cached, SwapFree, RssAnon, RssFile, VmSwap |

The `rep` of a spec id counts on across the list (for example `c1_off` reps 0-2, then 3-4).

## 5. Host check (*host*)

`llama-completion` from the scratch build of 0001-0003 (memo-13 §7). Settings as the app uses them: the kit's rendered
prompt, `-n 32 --temp 0 --ignore-eos -t 3 -ctk q8_0 -ctv q8_0 -fa on --repeat-penalty 1.1 --repeat-last-n 64`,
`GGML_REPACK_RELEASE=1`.

| row | text md5 | per-token lines |
|---|---|---|
| stock (the c2dev build, all knobs unset) | `0527d38b…` | — |
| (a), (b), (c) | `0527d38b…` (all three) | — |
| (a) profiled | `0527d38b…` | 31, parse in Dart and Python |
| (b) profiled | `0527d38b…` | 31, parse |
| (c) profiled | `0527d38b…` | 31, parse |

- The host text starts "The ocean is salty primarily due to a process known as "evaporative concentration."", the
  same opening as the r17 device text. The host and device numerics still differ, and a full-text match is not
  claimed.
- The lines report `kern=0 fuse=0 chunk2=0` for (a) and `kern=1 fuse=1 chunk2=1` for (b) and (c). Host clocks print
  `mhz=-1/-1`; both regexes accept that.
- `flutter analyze` is clean, `flutter test test/kit_lines_test.dart` passes (3 tests), and
  `python3 -I tools/c2_research_parse.py --selftest` prints OK. The parser output on the r17 report is unchanged
  except for the flag wording.
- The residency counter's code was run on macOS (16 KB pages) against the model file: 187,525 pages, totals and range
  split consistent, 84 to 107 ms. It has not run on Android.

## 6. Parser (`tools/c2_research_parse.py`)

- Reads campaign 1's per-token lines, as compact tuples (`tok_format` `c1`) or full lines, and adds the op table
  (`mmid`, `mm`, `glu`, summed `wait` in µs). Steady ms/token comes from them, as for c2 lines.
- Text classes also key on `GGML_EXPERT_SKIP` and `GGML_HEAD2`. A new line, "r19 rows … give one text: yes/NO",
  covers (a), (b), (c) and their profiled runs.
- Per group: median `res_before` / `res_after` expert and rest MB, and the clock caps after the runs.
- Hygiene flags per run:
  - `throttled: clock X<1800`: a big core below `--min-mhz` at run end;
  - `fault-bound: decode majflt N`: more than 100 major faults in the decode tokens of a profiled run;
  - `fault-bound: run majflt N`: a plain run with more than `--fault-run` (default 1,000) major faults, prefill
    included;
  - `gate timed out (heat H)`;
  - `gate unknown`.

  R49's fast runs had 3 to 42 decode faults, its slow ones 361 to 1,131.

## 7. What the owner does

1. Install `vollie-c2-1.19.0+19-research-r19.apk` over the installed r17/r18 (same app, "Vollie C2"). The model stays.
2. Open the app, wait for the model to load, start the research kit (as for r17), and leave the phone alone on the
   table, screen on (the kit keeps it on). If the phone is warm, the first run waits up to 10 minutes; the status line
   shows the headroom.
3. When it ends, send the report (clipboard, or the `research-r19-….json` file next to the model).

## 8. Run time (*estimated*)

- Per official run on r17's day: wall 17 to 25 s, plus a residency count of well under 1 s (a 750 k-page loop) twice.
- Gates:
  - 0 to 10 min before `cold`;
  - then 0 to 90 s per run, 18 times.
- Kit start: one reload, about 8 s (r17's reload was 8.2 s), plus the template and lib hashes.
- End: unload, then a 3 GB hash, under 1 min.
- Total:
  - **cool phone:** about 19 × 22 s + 1 min ≈ **8 to 10 min**;
  - **hot phone like r17's day:** 10 min + 18 × 90 s + 19 × 25 s + 1 min ≈ **46 min**;
  - a phone that cools after the first wait lies in between.

## 9. Not verified

- Anything on the phone: the gate's behaviour, the residency counter on Android, whether `mincore` reports file
  pages for the app's own file on this kernel (expected: the app owns the file). Without that, `res_*` comes back
  null and the run still proceeds.
- That `read_ahead_kb` is readable under the app's SELinux domain.
- Speed of any row, and whether (b) or (c) beats (a) on a thrashing page cache.
- KleidiAI is on in the Android build and off in the host checks (as for every kit). The device text equality across
  rows is checked by the parser, not here.
- The bit-4 helper thread's next-layer guess hit rate on OLMoE (unknown; it is advice only).
- **Correction to memo-13 (dated 2026-11-14, found while preparing the residency ranges).** The working-set script
  added the GGUF data offset to tensor offsets that the reader already returns as absolute, which shifted every page
  boundary by 1,781,952 B. Re-run with the correct offsets: every OLMoE figure in memo-13 §1 and §3 is unchanged.
  Granite moves by at most 0.1 MB (distinct decode 1,353.6 → 1,353.7 MB; LRU 1.5 GB early decode 55.2 → 55.1
  MB/token). No conclusion changes.
