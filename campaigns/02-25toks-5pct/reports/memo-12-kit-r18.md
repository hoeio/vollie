# Memo 12: research kit r18, the lever kit

Agent model: claude-opus-5-5

Date 2026-11-14. Author: Tom (code). **Nothing here ran on a phone. No speed is stated.** The APK was built and
inspected on the laptop. Host runs (*host*: Apple M4 Pro, `-t 3`, while a perplexity sweep ran) checked only texts and
line formats. Run times and sigmas are *estimated*: arithmetic on campaign 1's device figures, cited where used.

## 1. What was built

| item | value |
|---|---|
| APK | `lab/dist-c2/vollie-c2-1.18.0+18-research-r18.apk` (private), **17,128,219 B**, sha256 `066e80455ce626be8fe3385a05973d28c8f9aa9baf72bb49259fd3a144774cfa` |
| package / version | `com.insaneutils.vollie.c2`, "Vollie C2", versionCode **18** (r17: 17), versionName 1.18.0, minSdk 24, `arm64-v8a` |
| mode / signer | release (AOT), v2 signature, `CN=Android Debug`, certificate SHA-256 `c49dde85…fbb1716`, **the same as r17** |
| runtime | llama.cpp `620d2225c` + `patches/llama.cpp-c2/0001` to `0009` in the private scratch tree `lab/llama.cpp-c2-android`. Its diff is byte-identical to the other Tom's `lab/llama.cpp-c2dev`. `tools/c2_build_kit.sh` also applied the series to a clean export of `620d2225c` and compared every touched file |
| `libggml-cpu-android_armv8.2_2.so` | 1,684,496 B, sha256 `3782b02a1967a4743098e5bb24f622e5c042f49ab7ab4b55e552bbf8c837ab1d`, byte-identical in the APK |
| `libllama.so` | 4,287,008 B, sha256 `c012a035ad14ff66762758d35a2e883360b72a454ee23b4f9f389400ebc29e89`, byte-identical in the APK |
| build record in the APK | `assets/kit_build.json`: llama.cpp commit; series sha256 `0001 62b02dc0…`, `0002 b9bc9ada…`, `0003 ae95a7b9…`, `0004 49a16a6d…`, `0005 a61c857b…`, `0006 46c5dbc7…`, `0007 4315cddb…`, `0008 ed422bf8…`, `0009 4762f779…`; app patches `0001-kit-r17 7fd0dafb…`, `0002-kit-r18 0b8da795…`; both library hashes. The kit copies it into the report as `build`, and the parser compares the on-phone library hashes with it |
| app change | `patches/app-c2/0002-kit-r18.patch` on top of `0001` (README there). It applies cleanly to a pristine 2.3.0+45 copy after 0001 (checked) |
| tools | `tools/c2_build_kit.sh` (series check, build record); `tools/c2_build_android.sh` (now records each patch's sha256 when given the series directory); `tools/c2_research_parse.py` (A/B, identity, counters; `--selftest` passes) |

**Update over r17, model kept.** Same package, same debug key, higher versionCode: Android installs r18 as an update. The
model is in the app's external files directory (`Android/data/com.insaneutils.vollie.c2/files/olmoe-Mi.gguf` with its
`.verified` marker). Android keeps that directory across updates and deletes it only on uninstall. The download
constants are unchanged, so the app finds the verified file and does not download again. Not verified on the phone.

## 2. Run list (`Research.runs` in `lib/research.dart`, one line per spec)

Stack = `VOLLIE_TINY_OP=65536 VOLLIE_Q3K_X4=2 VOLLIE_MMID_NB=1 VOLLIE_MMID_FUSE=1 VOLLIE_EXPERT_IO=1
VOLLIE_SAMPLER_BYPASS=1` (memo-07's combination). A/B runs add `VOLLIE_PROF=1 VOLLIE_GRAPH_STATS=1 VOLLIE_AB_WARM=16
VOLLIE_AB_BLOCK=8`. Skip = `VOLLIE_SKIP_MASS=900 VOLLIE_SKIP_STATS=1 VOLLIE_SKIP_WIRE=1`. `head2K` = `'1024'` (one
constant). Repack is on everywhere except `norepack`. Every run starts from all knobs unset, plus `VOLLIE_LINES=stderr` and
`VOLLIE_RUN=<id>.<rep>`.

| # | id | kind | reps | env | notes |
|---|---|---|---|---|---|
| 1 | `cold` | official32 | 1 | none | unscored |
| 2 | `stock` | official32 | 3 | none | anchor to R52 |
| 3 | `stack` | official32 | 5 | stack | |
| 4 | `stack_prof` | profiled32 | 2 | stack + `VOLLIE_PROF=1 VOLLIE_GRAPH_STATS=1` | per-token lines with the class breakdown, full lines kept |
| 5 | `ab_tiny` | abba | 3 | stack + `VOLLIE_AB=knob:TINY_OP:0:65536` | |
| 5 | `ab_x4` | abba | 3 | stack + `knob:Q3K_X4:0:2` | |
| 5 | `ab_x4_kernel` | abba | 3 | stack + `knob:Q3K_X4:1:2` | |
| 5 | `ab_nb` | abba | 3 | stack + `knob:MMID_NB:0:1` | FUSE needs NB, so side A is also unfused: this A/B measures NB and FUSE together against neither |
| 5 | `ab_fuse` | abba | 3 | stack + `knob:MMID_FUSE:0:1` | |
| 5 | `ab_io` | abba | 3 | stack + `knob:EXPERT_IO:0:1` | read-ahead acts on faults: also read the parser's "all blocks" figure |
| 5 | `ab_sampler` | abba | 3 | stack + `knob:SAMPLER_BYPASS:0:1` | sampling is outside the graph: score it on the **wall** figure, not the graph figure |
| 6 | `skip` | official32 | 5 | stack + skip | |
| 6 | `ab_skip` | abba | 3 | stack + `VOLLIE_SKIP_STATS=1 VOLLIE_SKIP_WIRE=1` + `knob:SKIP_MASS:0:900` | |
| 7 | `ab_head2` | abba | 3 | stack + skip + `knob:HEAD2:0:1024` | |
| 8 | `package` | official32 | 5 | stack + skip + `VOLLIE_HEAD2=1024` | |
| 9 | `norepack` | official32 | 2 | stack, repack off | one model reload |
| 10 | text | — | — | — | every reply is kept whole. The parser checks the texts within each numerics class (repack, skip, head2) and stock == stack |

- `official32`: the benchmark prompt, 32 tokens, greedy, the app's streaming, `decodeTokensPerSec`.
- `profiled32`: the same with one chunk per token.
- `abba`: the story prompt (`Write a short story about a lighthouse.`), **400 tokens**, one chunk per token. The
  hot-only cool-down gate applies (headroom < 0.85, r15's rule for A/B stories, since both sides share the heat).
- Prompt-prefix reuse stays off (r17).

**Host check of the exact env combinations** (series 0001 to 0009, `llama-completion`, official prompt, `-n 32`,
greedy, repeat penalty 1.1 over 64, `VOLLIE_LINES=stderr`, *host*):

| result | runs |
|---|---|
| text md5 `4ff6b707…` | stock = stack = stack_prof = every exact-lever A/B (tiny, x4, x4_kernel, nb, fuse, io, sampler) |
| text md5 `41af6815…` (repack off) | stock = stack |
| text md5 `e16fc82d…` | skip = package = `ab_head2` |
| text md5 `fe646d4b…` | `ab_skip` (switches the rule every 4 tokens, as expected) |

Every A/B run printed one `c2-ab:` line and 16 side-A plus 15 side-B per-token lines (`WARM=4 BLOCK=4` on the host,
32 tokens). All 326 runtime lines matched memo-11's frozen regexes. Skip runs printed `c2-skip-run`; head2 runs printed
`c2-head-build` and `c2-head-run`. On the host the story prompt ran the full 400 greedy tokens without an end token
(399 decode graphs).

## 3. A/B size and expected sigma (*estimated*)

One generation of 400 tokens gives 399 decode graphs. The first 16 are warm-up; the remaining 383 make 47 full blocks
of 8 plus one block of 7, in the order A B B A. That is about 24 blocks per side per generation, and **three generations
per lever give about 72 blocks per side.**

The per-block standard deviation is taken from campaign 1, where the per-block mean ms/token was measured on the phone:
1.6 to 2.0 ms on clean 8-token blocks (R45), and 2.3 to 4.5 ms in the wider range quoted in knowledge.md §6. With
σ(B−A) = sd × √(2/n):

| clean blocks per side | sd 1.6 | sd 2.0 | sd 3.0 | sd 4.5 |
|---|---|---|---|---|
| 72 (all clean) | 0.27 | 0.33 | 0.50 | 0.75 |
| 50 (30 % dropped) | 0.32 | 0.40 | 0.60 | 0.90 |
| 48 (two generations, all clean) | 0.33 | 0.41 | 0.61 | 0.92 |

So with three generations a 1 ms effect is resolvable (σ ≤ 0.4) if OLMoE's clean-block sd is at or below campaign 1's
R45 value and no more than about a third of the blocks are dropped. At sd 3 or more no sub-1 ms effect is resolvable in
this kit. Campaign 1 found 30 or more counted blocks per side necessary, and its 3-ABBA kits counted only 7 to 13. The
r18 count is larger because a generation is 400 tokens long and the blocks are 8 tokens.

**Dropped blocks** (parser): a block with any major fault, or with a logged cpu4/cpu7 clock below `--min-mhz` (default
1,800). Warm-up (`blk=0`) never counts. The parser also prints B−A over all blocks; for `ab_io`, whose effect is on
faults, that is the figure to read.

## 4. Run time (*estimated*)

| part | runs | per run | expected | worst case (every gate 90 s) |
|---|---|---|---|---|
| official32 / profiled32 | 23 | gate (r16 mean ≈ 60 s, maximum 90) + about 8 s of prompt and reply | ≈ 26 min | ≈ 38 min |
| abba | 27 (9 × 3) | hot-only gate (r15: mostly short; assumed 15 s) + about 18 s (400 tokens at about 40 ms, memo-08's band, plus the prompt) | ≈ 15 min | ≈ 49 min |
| reload (norepack), final unload, model hash (3.07 GB), end reload | — | — | ≈ 2 min | ≈ 3 min |
| **total** | 50 | | **≈ 43 min** | **≈ 90 min** |

The expected total is inside the 75-minute limit. The worst case is over it only if every gate times out, and r16's gates
averaged about 60 s, not 90. If the lead wants a guarantee under 75 min, the first cut I would make is the A/B
generations from 3 to 2. That saves about 5 min expected and 16 min in the worst case, and σ rises to the 48-block row.
**Nothing was cut.** The button says "~45-75 min".

Report size (*estimated*): about 10,800 per-token tuples of about 90 characters plus chunk times, about 1.1 MB of JSON.
The copy button therefore copies `c2gz:` + base64(gzip), about 0.3 MB. The file next to the model is plain JSON.

## 5. Report schema changes from r17

```
build {kit, app, llama_cpp, series [{file, sha256}], app_patches [{file, sha256}], libs {name: sha256}}
story, tok_fields [names of a per-token tuple]
unmatched_lines [str]    runtime lines whose run= id matched no row (should be empty)
idle_lines               count of run=- lines (warm-ups, loads; knobs cleared)
runs[] gains:
  run_tag                VOLLIE_RUN of this run (<id>.<rep>)
  env_set {VAR: value}   every known knob as native code saw it after the run's env was set (getenv)
  gate                   the headroom the gate waited for (0.65 official, 0.85 abba)
  tok [[ab, tok, us, majflt, minflt, mhz4, mhz7, prompt, blk, threads, ops, barriers, syncs, skipped, pairs,
        head, router, mm, mmid, attn, other, wait]]   per-token lines of this run (memo-11 §3), compact
  prof [str]             profiled32 only: the full per-token lines, op table included
  c2 [str]               this run's per-prompt lines (c2-ab, c2-skip-run, c2-graph-run, c2-head-run, c2-head-build)
  chunk_ms               now also for abba runs
```

**How lines reach the right row** (memo-11 §8, notes 1 and 2): the kit sets `VOLLIE_LINES=stderr` for every run and load,
so the lines go to the app's stderr log file whatever llamadart's logger does. After each run it reads the new log
lines, finds `cpu-prof ab=` or `c2-` anywhere in a line (a prefix does no harm), and files each line by its `run=` id
into that run's row. A prompt's `c2-*-run` lines are printed only when the next prompt starts, so they land in the row
that owns them one run later. At the end the kit frees the context (`unloadModel`), which prints the last prompt's
lines, and files them **before** the end reload truncates the log. The kit's own repack reload uses `_loadOn`, which does
not truncate the log. This is the robust way of the two: it needs no dummy prompt, and a line can never be counted for
the wrong run. Lines without `run=` (`c2-head-build`) go to the run being measured.

## 6. Parser (`tools/c2_research_parse.py`)

- Reads plain JSON or the clipboard form `c2gz:…`.
- Identity: model size, verification marker and final hash; library hashes against `build.libs`; the series and app
  patch hashes; the chat template; the prompt.
- Greedy texts: identical within each class (repack, `VOLLIE_SKIP_MASS`, `VOLLIE_HEAD2`); `stock == stack`; whether
  `skip == package`.
- Per group: the median, minimum and maximum of printed tok/s; steady ms/token; clocks; faults; counters (expert bytes
  skipped, head2 exact rows per column, graph ops, tiny ops, barriers, syncs); hygiene flags.
- Per A/B id: clean and total block counts per side; graph B−A ± σ with the verdict (σ ≤ `--sigma` 0.4 → resolved; else
  resolved only if |B−A| ≥ 1 ms and ≥ 2σ; else UNRESOLVED); the same over all blocks; wall B−A ± σ from the chunk
  arrival times (the sampler's figure); B−A per class; mean counts per side (does the lever act: barriers, ops, syncs,
  skipped pairs).

`--selftest` covers an A/B with 2 generations (a known −1.0 ms effect, one faulted and one throttled block dropped), the
identity classes, the build check, counters and the `c2gz:` form. Run on the host A/B lines of §2, the parser read every
run (4-token blocks there, so those numbers mean nothing as speed).

## 7. What the owner does

1. Download the r18 APK where the lead puts it (17,128,219 B, sha256 `066e8045…774cfa`).
2. Open it and install. Android offers **Update** for "Vollie C2", because r18 replaces r17. The owner's Vollie is
   not touched, and the 3 GB model stays.
3. Open Vollie C2 and wait for the model to load (no download). Settings → **"Run research kit r18 (~45-75 min)"**.
   Leave the phone alone, screen on, not charging if possible.
4. At the end: **"Copy research results"** and paste the text (it starts with `c2gz:`). Or send the file
   `research-r18-<time>.json` from `Android/data/com.insaneutils.vollie.c2/files/`.

## 8. Not verified

- **Nothing ran on a phone**, including that the update keeps the model, that `VOLLIE_LINES=stderr` lines reach the log
  file through the app's `dup2`, the profile's overhead, and the memory of the head2 copy (+58 MB) on top of repack.
- The A/B sigma rests on campaign 1's Granite block sd; OLMoE's is unknown until r18 reports it.
- The host texts used llama.cpp's sampler and renderer at `-t 3`. The phone uses llamadart's (r17 checks that the
  template is the model's own).
- If llamadart ends a story early on the phone (an end token), that generation has fewer blocks. On the host it did not.
- The Dart kit passed `dart analyze` and the line-parsing test (`flutter test test/kit_lines_test.dart`); the kit
  itself did not run.
