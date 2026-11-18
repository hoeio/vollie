# Memo 32: research kit r28 (R68's rehearsal): touch and nice (0012), T-A cap probe, T-B fields, T-D smaps fix, two reference APKs

Agent model: claude-opus-5-5

Date 2026-11-18. Author: Tom (code). **Nothing here ran on a phone, and no phone speed is stated.** The APKs were built
and inspected on the laptop. Labels: *host* = Apple M4 Pro (macOS, or a Ubuntu 24.04 aarch64 container on it), relative
only; *estimated* = arithmetic; *device* = quoted from an existing report. Kit r28 is kit r27 ([memo-30](memo-30-kit-r27.md))
with memo-31's TEST list (T-A, T-B, T-D, T-E) and TRY ranks 1 to 3, and the lead's three additions of 2026-11-18
(power-save state; the two reference runtimes). Nothing is committed or uploaded by this memo's author.

## 1. What was built

| item | file |
|---|---|
| runtime patch | `patches/llama.cpp-c1-olmoe/0012-c2-touch-nice.patch` (README entry added) |
| c2 tree | `lab/llama.cpp-c1r28` = 620d2225c + `patches/llama.cpp/0001-0003` + `patches/llama.cpp-c1-olmoe/0001-0012` (the series applies in order on a clean export and equals the tree, checked file by file) |
| reference trees | `lab/llama.cpp-r28c1` = 620d2225c + `patches/llama.cpp/0001-0003` (campaign 1's shipped series; the three patch files and `android/build_cpu.sh` are unchanged since `686f17d`, checked with `git diff 686f17d HEAD`); `lab/llama.cpp-r28stock` = 620d2225c, no patch |
| app patch | `patches/app-c2/0012-kit-r28.patch` (README entry added) |
| parser | `tools/c2_research_parse.py`: kit r28 (checks, tables, selftest) |
| build script | `tools/c2_build_kit.sh`: `RUNTIME=c2|c1|stock` and the series argument `none` |
| APKs | `lab/dist-c2/vollie-c2-1.28.0+28-research-r28.apk`, `...-r28-c1.apk`, `...-r28-stock.apk`, with `BUILD-r28.txt`, `BUILD-r28-c1.txt`, `BUILD-r28-stock.txt` |

### 1.1 Runtime patch 0012 (bit-identical, both knobs off unless set)

- **`VOLLIE_TOUCH=1`** (`libllama.so`, on 0011's mapping registry). Exported `llama_vollie_touch_begin(want, n)` starts
  one helper thread restricted to cpus 0-3. Every `VOLLIE_TOUCH_S` seconds (default 12) it reads one byte per page of
  the page set through the model mapping. `llama_vollie_touch_end()` asks for one more pass, waits until that pass is
  done, ends the helper and returns the pages it touched.
  - Page set: the caller's vector. The kit passes the hold's wanted set, that is the pages resident at the end of
    this model's previous run. Without a vector it uses the pages resident at `begin` (mincore), and if that fails,
    every page. The `src` field says which.
  - Without the knob, both calls return -1 and do nothing.
  - Each pass holds the registry mutex, so an unload waits for it.
  - Kit line, one per pass: `c2-touch run= pass= final= src= at_ms= set= pages= ms= cpu= majflt= minflt= sum=`.
    `majflt` and `minflt` are the helper thread's own faults during the pass, so the line shows how many pages had
    left the cache (majors) and how many were only unmapped (minors).
- **`VOLLIE_NICE=<n>`** (CPU module, on 0010's pool; Linux; pool graphs only, so it needs `VOLLIE_POOL=1`, which every
  r28 run has).
  - At each graph start, every thread of the graph (thread 0 = the caller) compares `getpriority(tid)` with n and
    calls `setpriority` when they differ. The pool's workers therefore set it once, after the pool starts.
  - A thread whose set fails is not asked again in that run, and its errno is kept.
  - With the knob unset, every thread whose value 0012 changed gets its first value back, by tid, at the next graph
    start. An "n" run after an "N" run therefore runs at the old priority. If the Dart caller thread changes, the old
    caller gets its value back as well.
  - Kit line, per prompt when the knob was set or a restore ran:
    `c2-nice run= want= pool= restored=<ok>/<failed> w<i>=<tid>:<value read back>:s<sets>:f<failed>:e<errno>`.
- **Host-only knob `VOLLIE_TOUCH_HOST=1`** (`common/common.cpp`; not in any APK library). The tools start the touch
  helper after the model load and leave it running, so `llama-perplexity` runs with touches beside its graphs.

### 1.2 App patch 0012 (kit r28, version 1.28.0+28)

- **R68 run list** (`Research.blockPlan`, c2 APK). Blocks O G O G O, each with two warm-ups (no gate, no hold):

  | block | model, arm | runs | configuration |
  |---|---|---|---|
  | b1 | OLMoE off | 10 claim-format | P X (pin + exec), no touch, no nice |
  | b2 | Granite num | 10 claim-format | P X |
  | b3 | OLMoE off | 16 = 4 quads | P X, touch x nice, Williams rows `(k + 2 x model index) % 4` |
  | b4 | Granite num | 16 = 4 quads | same |
  | b5 | OLMoE num (MASS=900 + head 1024) | 10 claim-format | P X + touch + nice |

  - Per model and factor that gives 8 on against 8 off.
  - Every scored run has kv clear, `VOLLIE_SCHED=1`, r16's gate, the hold and 3 threads. There are no profiled runs.
    That saves time and leaves out T-B's per-token `c2-sched` (see §6).
  - In b5, "num" is the setting R61 measured: 12.0927 / 12.0976 *host*, against the budget 12.3003. That setting
    includes `VOLLIE_HEAD2=1024`. Using `MASS=900` alone would be a configuration with no measured PPL. The lead should
    confirm b5's touch and nice in the pre-registration: memo-31 §7 says "whichever the host checks allow", and §2
    allows both.
  - Run ids: `<m>_<arm>_wu`, `<m>_<arm>_claim`, `<m>_<arm>_q[_touch][_nice]`.
- **Touch calls.** In every gated run, `touch_begin` is called right after `exec_sync` and before the gate wait, with
  the hold's wanted vector. `touch_end` is called after the hold's last pass and before `smaps_before` and every other
  pre-run read. Rows record `touch_begin` (rc, pages, ms) and `touch_end` (pages, ms). Runs with touch off record rc -1.
- **T-B, per run:**
  - `power`: `isPowerSaveMode`, `Settings.Global low_power`, thermal status, battery %. It is read before the run,
    outside the timed window.
  - `workers_after`: every DartWorker thread's tid, nice, priority, policy, cpu and cgroup, after the run. The pool's
    threads are among them, and the c2-pool line gives their tids.
  - `proc_before` / `proc_after`: the cpu rows of `/proc/stat` and `/proc/loadavg`, or their errno, around the run.
- **T-D.** `DeviceProfile.smaps` matches the model's rows by inode (`stat(2)`) as well as by path. The parser falls back
  to `RssFile` from `mem_before` when the model's Rss reads 0, as in R67.
- **Session label.** A required choice on the start screen, "Session: deep-sleeping list on / off". The kit and the
  probe buttons stay disabled until it is chosen. It is recorded as `session_label` and `deep_sleeping_list`.
- **Power saving (lead, addition 1).** At the start, the kit reads `isPowerSaveMode` and `low_power` into
  `power_start`, and at the end into `power_end`.
  - If either reads on, the session is recorded as `session_power: "power-save on"`.
  - The kit's status line then starts with "POWER SAVING IS ON (recorded as power-save on)".
  - The start screen shows the state, in red when it is on.
  - The kit never changes the setting.
- **T-A cap probe.** The button "Run cap probe r28 (6 min, no model)" starts it; no model runs during it.
  - One sample per second for 360 s, with these fields: `scaling_max_freq` and `scaling_cur_freq` of policies 0, 4
    and 7, the headroom, thermal status, power saving, `low_power`, battery % and °C, plugged, and the spinner and load
    flags.
  - The headroom is read only on every second sample, because the API returns NaN when it is polled more than once a
    second.
  - A spinner is drawn in minutes 1, 3 and 5 and not in minutes 2, 4 and 6.
  - From 180 s to the end, three isolates spin, each pinned to cpus 4-7 (their `sched_setaffinity` rc is recorded).
  - The probe is labelled with the session label and the power-save state read at its start.
  - Its report is separate from the kit report: shared preferences key `cap_probe_report`, a file
    `capprobe-r28-<time>.json` next to the models, and "Save cap probe to Downloads", which saves
    `Download/vollie-capprobe-r28-<time>.json`.
- **Reference runtimes (lead, addition 2):** see §3.

## 2. Host checks (*host*, 2026-11-18)

### 2.1 Setup

The builds are those of memo-30 §2: macOS AppleClang without OpenMP, and the Ubuntu 24.04 aarch64 container with
gcc 13.3 and `GGML_OPENMP=OFF`. Both use `armv8.2-a+dotprod+fp16`, no KleidiAI, shared libraries.

- Common env for every configuration: `GGML_TINY_OP=262144 GGML_EXPERT_IO=1 VOLLIE_SAMPLER_BYPASS=1 GGML_C2_IO=1`,
  `-c 512`, wikitext-2-raw test.
- Configurations:

  | name | env on top of the common env |
  |---|---|
  | `attach` | none (the tool as shipped) |
  | `off` | `VOLLIE_NO_ATTACH=1` |
  | `touch` | off + `VOLLIE_TOUCH=1 VOLLIE_TOUCH_HOST=1 VOLLIE_TOUCH_S=2` (passes beside the graphs for the whole run) |
  | `nice` | off + `VOLLIE_POOL=1 VOLLIE_CPUMASK=<mask> VOLLIE_SCHED=1 VOLLIE_NICE=-10` |
  | `all` | nice + `VOLLIE_EXEC_MAP=1` + touch |

- Threads and masks: macOS `-t 8`, mask `0xf0`. Linux `-t 6`, mask `0x1,...,0x20`.
- The Linux PPL ran with `--cap-add SYS_NICE`, so that the set succeeds there.

### 2.2 Per-chunk PPL

20 chunks at `-b 512` and at `-ub 32`, 8 chunks at `-ub 1`. Each configuration is compared value by value with `off`.

| platform | model | `-b 512` | `-ub 32` | `-ub 1` |
|---|---|---|---|---|
| macOS | olmoe-Mi | attach, touch, nice, all equal, 20 of 20 (off 8.9646 10.4244 9.2074 9.4783 … 12.5367) | touch, nice, all equal, 20 (8.9131 … 12.5029) | touch, nice, all equal, 8 (8.9467 … 10.2291) |
| macOS | granite-Mi | attach, touch, nice, all equal, 20 (4.9584 6.6520 6.1119 6.6024 … 9.7318) | touch, nice, all equal, 20 (4.9398 … 9.7486) | touch, nice, all equal, 8 (4.9839 … 8.5192) |
| Linux | olmoe-Mi | all equal, 20 (8.9607 10.4468 9.2244 9.5063 … 12.5459) | all equal, 20 (8.8685 … 12.4817) | all equal, 8 (8.9975 … 10.2878) |
| Linux | granite-Mi | all equal, 20 (4.9811 6.6764 6.1185 6.6045 … 9.7685) | all equal, 20 (4.9737 … 9.7453) | all equal, 8 (4.9805 … 8.5720) |

- Every `off` value printed equals memo-30's for the same build system and setting, so 0012 changed nothing with its
  knobs off.
- Linux ran `off` and `all` only. `all` is pool + nice + exec + touch, with the nice set succeeding under
  `--cap-add SYS_NICE`. The separate Linux `attach`, `touch` and `nice` runs were stopped at the owner's 20:50 deadline,
  unfinished. On Linux, `all` covers both knobs; on macOS each knob was also run on its own.
- In the `touch` and `all` logs, the touch helper printed 15 to 28 passes per run, during the graphs.

### 2.3 Greedy text

The program is memo-30's greedy test, extended (`greedy2.cpp`): app settings, `n_ctx` 4096, q8_0 KV, flash attention,
3 threads, 32 greedy tokens. Seven prompts run in one process, with the knobs re-set between prompts:

1. off
2. touch "pre" (begin, 2.5 s with `VOLLIE_TOUCH_S=1`, end, then the prompt)
3. touch "during" (begin before the prefill, end after the 32 tokens, so passes run during the decode)
4. pool + nice -10
5. pool only (the restore)
6. pool + exec + nice + touch "pre"
7. off again

These run on OLMoE, Granite, and Granite with its num arm.

| platform | OLMoE | Granite | Granite num |
|---|---|---|---|
| macOS (mask `0xf0`) | 7 prompts, 1 text (`ef13174b`) | 7, 1 text (`ac2378ae`) | 7, 1 text (`ac2378ae`) |
| Linux, no `CAP_SYS_NICE` (mask `0x10,0x20,0x40`) | 7, 1 text (`648d90ac`) | 6 of 7 (the 7th was stopped), 1 text (`ac2378ae`) | not run (stopped) |
| Linux, `--cap-add SYS_NICE` | off, nice, pool, off: 1 text (`648d90ac`) | not run (time) | not run (time) |

The md5 is over this program's 32-token text. It is not comparable with memo-30's md5, which came from another program.

### 2.4 What the lines showed

- **Linux, `src=resident`:** 729,031 of 729,031 pages, 8 to 29 ms per pass, `cpu=0` or `1`, 0 majors and 0 minors.
  The model was resident and mapped, which is the host's case. `end` returned 729,031.
- **macOS, `src=all`:** 182,259 pages of 16 KB per pass, `cpu=-1` (no `sched_getcpu`). The first pass after a run took
  524 ms and the later passes 1 to 11 ms.
- **Nice, Linux without `CAP_SYS_NICE`.** Docker's default capability set lacks it, root included. Each thread printed
  `f1:e13`, that is one failed set with `EACCES`, and the value read back was 0. The next "off" prompt printed no
  restore, because nothing had been changed. This is the case the phone may be in.
- **Nice, Linux with `--cap-add SYS_NICE`:** `c2-nice run=g2 want=-10 pool=1 restored=0/0 w0=8:-10:s1:f0:e0 w1=75:-10:s1:f0:e0 w2=76:-10:s1:f0:e0`: each
  thread set once and read back -10. The next "off" prompt printed `restored=3/0` with each value back at 0. The text was
  unchanged.
- On macOS the nice code is not compiled in (Linux only), and the line shows zeros.

## 3. Reference runtimes (lead's addition 2)

### 3.1 What I matched

- **Campaign 1's shipped runtime (`686f17d`).** The series is `patches/llama.cpp/0001-cpu-runtime.patch` (sha256
  `08c1b796…`, unchanged since `686f17d`), `0002-granite-rollback`, `0003-sampler-bypass`, on 620d2225c, built with
  `android/build_cpu.sh` with KleidiAI on, as campaign 1 and kit r19 built it.
  - **It runs OLMoE:** kit r19 (R53) and kit r20 ran this exact runtime on olmoe-Mi.
  - Env, Granite: R49 / kit r16's shipped settings, that is `Research.served` (skip 117 + head2 q4_0 / 2048 + sampler
    bypass) plus the phone's saved tiny-op 262,144 and expert I/O 1.
  - Env, OLMoE: kit r19's `c1_ship` (`GGML_TINY_OP=262144 GGML_EXPERT_IO=1 VOLLIE_SAMPLER_BYPASS=1`, no numerics).
  - **Caveat for the lead:** skip 117 was budgeted on campaign 1's abliterated file. Its PPL on the campaign-2 stock
    file granite-Mi was never measured (R59 screened 130). As a speed reference it is "the shipped config" as asked.
    It is not a quality-checked configuration on this file.
- **Stock llama.cpp.** The earlier campaign-2 stock baseline on the phone is R52 with kit r17, built with
  `tools/c2_build_kit.sh` on the unpatched 620d2225c tree, which calls `android/build_cpu.sh` with its default
  `KLEIDI=ON` (memo-10 §1). I matched that: `RUNTIME=stock tools/c2_build_kit.sh lab/llama.cpp-r28stock lab/app-c2 none`,
  same script, KleidiAI on, the variant module `android_armv8.2_2`, and llamadart's bundle for the other libraries.
  - Off settings: no env at all, and no load env.
  - The host stock PPLs of R50 and R51 were built without KleidiAI (`tools/c2_build_android.sh` and the host trees).
    Those were quality numbers, not the phone runtime.

### 3.2 How they are selectable: a deviation from the brief, for the lead to decide

The brief asks for the reference libraries in the same APK, selectable per block. I did not build that, because it
cannot be done safely in one process:

- llamadart binds `libllama.so` by soname through `libllamadart.so`'s native assets.
- ggml then picks the CPU module from the APK's library folder by file prefix and score.
- A second `libllama.so` cannot be loaded beside the first, and a loaded one cannot be swapped. Per-block selection would
  need a process restart between blocks, with the kit's state saved and resumed. It would also need a soname-preload
  trick for `libllama.so` and a way to keep ggml's scan from picking the c2 module. None of this can be tested without
  a phone, and a failure would only show on the phone in the middle of a session.

Instead I built **three APKs from the same Dart code and the same app patches**:

- They have the same package, signing key and versionCode 28, so each installs as an update over the others and keeps
  the internal model copies.
- Each one's `assets/kit_build.json` carries `runtime` (`c2`, `c1` or `stock`), and the kit's run list follows it.
  - The c2 APK runs R68's plan.
  - The c1 and stock APKs run the reference plan: Granite then OLMoE, two warm-ups and 10 claim-format runs per model,
    with r16's gate, the hold and 3 threads. Pin there is the app's DartWorker pin: the runtime pin, kv clear, exec and
    touch need c1-olmoe patches that these runtimes do not have. Their calls fail the symbol lookup and record the error.
  - Power saving is recorded as in every r28 run.
- Every load row already hashes the mapped native libraries (`libs`), and the report holds `build.runtime` and
  `build.libs`, so the library that ran is identified per load.

If the lead needs one APK, the restart design above is the way. I estimate it at about a day of work, and it would need
a first phone test of its own.

### 3.3 Session time (*estimated*) and a proposed split

The basis is the stored sessions (*device*):

- R67 / kit r27: 47 minutes for 40 gated runs and 8 warm-ups. The cycle per gated run (wait + hold + run) had a
  median of 78 s and a maximum of 107 s.
- r26: 102 s median, when every wait timed out.
- Warm-up about 6 s, reload 3 to 13 s.
- The storage step with the copies present: about 13 s of hashing.

| session | gated runs | time |
|---|---|---|
| c2 (R68 plan) | 62, plus 10 warm-ups and 5 loads | 62 × 78 to 102 s + about 3 min ≈ **84 to 108 min** |
| c1 reference | 20 | ≈ **30 to 38 min** |
| stock reference | 20 | ≈ **35 to 45 min**; OLMoE on stock ran 3.3 tok/s with faults in R52, so its runs are longer (about 25 s each against about 8 s) |
| everything in one session | 102 | ≈ 2.5 to 3.2 h, above the 2.5 h line |

The split is needed anyway, because the reference runtimes are separate APKs. No reference block is cut. Proposed order,
for the lead and owner:

1. **Session A** (c2 APK, deep-sleeping list on, power saving off): R68 plan, about 1.4 to 1.8 h.
2. **Session B** (c2 APK, deep-sleeping list off, power saving off, background apps closed): the same plan. Memo-31
   §4's go rule reads this session.
3. **Session R** (power saving off, list off, on a later day or after a cool-down):
   - install `r28-c1`, run the kit (about 35 min);
   - install `r28-stock`, run the kit (about 40 min);
   - reinstall the c2 APK.

   That is about 1.3 h including the installs. I would put c1 first, so that the reference to campaign 1 does not depend
   on the stock block that follows. A stricter alternative is c1, stock, c1 on two days, which would show a drift in
   heat or time of day.

## 4. T-E: does the official benchmark screen animate? (desk check)

Yes. The official benchmark is "Optimize for this phone" (`Assistant.optimize`: `_coolDown`, then
`_complete(_benchPrompt, _benchTokens)` per candidate).

- While it runs, `optimizing` is non-null. The settings page then replaces the button's icon with an indeterminate
  `CircularProgressIndicator(strokeWidth: 2)` (18 px), and that indicator repaints every frame for the whole run, timed
  prompts included.
- The same holds for campaign 1's kit r16, read in the sibling app at 2.3.0+45 (`settings_screen.dart` lines 292-297):
  its runs set `optimizing` through `_researchStatus` and drew the same spinner unless the owner left the page.
- Campaign 1's plain runs (R49, 20.51 tok/s *device*) and kits r17 to r26 therefore carried the spinner's cost. Kit r27
  and r28 do not, because they show a static screen. Whether that cost moves the clock cap is T-A's question.
- If T-A shows it does, the fix belongs in the app: a static benchmark screen. That is for the lead and the owner to
  propose; this memo does not change the shipped app.

## 5. Build record

`tools/c2_build_kit.sh`, `android/build_cpu.sh` flags unchanged. `flutter analyze lib test`: no issues. `flutter test`:
all passed; `kit_lines_test.dart` has 15 tests, including the r28 run list, the reference plans and the line tags.
Signing: the debug key, certificate SHA-256 `c49dde85…69fbb1716`, the same as r27. Package
`com.insaneutils.vollie.c2`, versionCode 28, versionName 1.28.0, in all three APKs.

| file | bytes | sha256 |
|---|---|---|
| `vollie-c2-1.28.0+28-research-r28.apk` (runtime c2) | 16,399,218 | `71297cd838507b1a6cc8c3e9c6ea9831a2c029346b0ec7251aee2c931fce6350` |
| its `libggml-cpu-android_armv8.2_2.so` / `libllama.so` | | `6f1d859a962f473c7f7b5d1547a2530157a3ca98bbe046524bce3d90b4e3d28e` / `126bfb4f04ec7fff1135cf6e7c1c65e191556aad695c52f718e0e3a645d9ede8` |
| `vollie-c2-1.28.0+28-research-r28-c1.apk` (runtime c1) | 16,368,862 | `766ff392006a5d066f86b7573f2ead74e4b1e42bdc3b8f72f0104c79fb79f3dc` |
| its CPU module / `libllama.so` | | `53bfee45511d14177fa4165d897a1d44b51310c51bdc83aef8f543e8e7d9d11c` / `f2f4f88af78d601ae1ba14dc895a06aa23d9320661a55271a8fa5cf3c9c23861` |
| `vollie-c2-1.28.0+28-research-r28-stock.apk` (runtime stock) | 16,319,578 | `c9b156a10ed7e255ec8db2b01a39d69774df6024bb66955cb2218f49a93a6857` |
| its CPU module / `libllama.so` | | `604e0f69c12b5972dd2ba46f1b24fd462f6cf9e4fefe6041afcff45b4d68cd0d` / `de2ccec771c869b15f237e66112830fe43c2391eae807522ee58c0620c45cb8c` |
| `0012-c2-touch-nice.patch` | | `81c458f24e7b35683620b17336c632b0dea65c775c968c2db209f35a06ea93da` |
| app `0012-kit-r28.patch` | | `792b2866387ecd65ac0d38d9419c1abec25cbc153ca947b1f2719c9dbc6cac83` |

- `libllama.so` in the c2 APK exports `llama_vollie_exec_sync`, `llama_vollie_touch_begin` and
  `llama_vollie_touch_end`, imports `mincore` and `sched_setaffinity`, and holds the `c2-touch` format.
- The c2 CPU module imports `getpriority` and `setpriority` and holds the `c2-nice` format.
- The c1 and stock libraries export none of the vollie symbols.
- The c2 APK was built twice in this session (stock, c2, c1, c2). The second build's hashes are the ones above and in
  `BUILD-r28.txt`. The two builds' hashes differ, as memo-10 found builds do.
- The c1 libraries cannot be byte-compared with campaign 1's shipped ones, because builds are not reproducible. The
  identity is the series hashes in `BUILD-r28-c1.txt`, which equal the files at `686f17d`.

## 6. Not done, and never run on a phone

- **Per-token `c2-sched` (T-B) is not built.** It needs a runtime line per token. The per-phase line, the per-run
  `/proc` deltas and `workers_after` are what r28 has.
- **The ADPF hint session (memo-31 TRY 2) is not built (time, owner's 20:50 deadline).** I had also judged it not cheap:
  - The code is cheap. It would be `APerformanceHint_createSession` on the three worker tids through `dlopen` of
    `libandroid.so` (API 33 and later), with a token duration reported per graph, roughly 80 lines in 0012.
  - The design is not cheap. As a second level of the priority factor it turns R68's 2 x 2 quads into 2 x 3 cells, so
    the Williams layout, the 8-against-8 count per factor and the session time (§3.3) all change.
  - It adds no clock headroom: a hint cannot raise `scaling_max_freq` (memo-31 §2).
  - If the lead wants it, it fits best as its own claim-format block after R68 shows whether nice is granted, because
    if `setpriority` is refused on the phone, ADPF is the remaining priority route.
- **Not verified on the phone:**
  - Whether an app thread may lower its nice. The container refused it without `CAP_SYS_NICE`, and the phone may
    refuse it too; the `c2-nice` line's `f`/`e` fields will say.
  - Whether `/proc/stat` and `/proc/loadavg` are readable under the app's SELinux domain. They are probably denied on
    Android 8 and later; the errno is recorded.
  - Whether `stat(2)`'s inode matches smaps' inode field. It should for an f2fs file.
  - The touch pass time on the phone. On the host it was 8 to 29 ms for a resident and mapped 3 GB mapping. After a
    90 s wait with 0.5 to 1 GB unmapped it will be minor faults, and majors for pages that left the cache.
  - Whether a little core keeps up with that.
  - Whether `Settings.Global low_power` is readable. It returns -2 when not.
- **The reference APKs replace each other on install, with the same versionCode.** I have not checked that the phone's
  installer accepts a same-version update. If it refuses, uninstalling is not an option, because it deletes the
  internal copies. The fix would be to rebuild the references with versionCode 29 and 30.
- R67 shows that the DartWorker that calls ggml can change after an idle wait. With nice, the old caller gets its value
  back at the next pool graph, by tid.
