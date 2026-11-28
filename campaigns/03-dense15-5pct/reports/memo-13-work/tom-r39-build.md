# Kit r39 for R73: the first real decode of campaign 3 (choice A stock, one arm), build notes

Tom (`claude-opus-5-5`), 2026-11-28. The owner approved R73 on 2026-11-28 ("1로 가죠", after R77). Scope:
`memo-10-work/clyde-r73-plan.md` §2, §3 and §7, with the lead's changes of 2026-11-28 (stock = choice A, one arm, no pin-guard
A/B, no repack or KleidiAI arm, nCtx 512, downloads from `HoeioUser/vollie3`). Reuses r28's research run (gate, hold, kv clear
env, per-run records) and r38's native libraries. Not committed, not uploaded.

**Labels.** §5 numbers are *host* (owner's Apple M4 Pro). §4 durations and memory are *estimated*. Nothing ran on the phone.

## 1. Choice A's load path in the code

- `lab/app-c3/lib/device.dart:591`: `canCopyWeights(copyBytes) => totalRam - copyBytes >= 5000000000`.
- `lab/app-c3/lib/assistant.dart:456-467`: `_loadedRepack = _repack ?? device.canCopyWeights(<file bytes>)`, then
  `useExtraBuffers: _loadedRepack`.
  - The `ModelStore.copyBytes` substitute applies only to olmoe-Mi's exact size, so for the c3 files it is the file's length.
  - `_repack == null` means the app's rule.
- `assistant.dart:471-472`: KV cache `q8_0` for K and V on the CPU (llamadart turns on flash attention for it).
- The shipped Vollie 2 app has the same rule: `lab/app-prod3/lib/device.dart:107` and `assistant.dart:448`. Its unknown-file
  branch also uses the file's length.
- **Result on the phone (*estimated* from R72's 7.75 GB header):**

  | subject | file bytes | 7.75 GB − bytes | copy allowed |
  |---|---|---|---|
  | Yi | 3.48 GB | 4.27 GB | no |
  | Llama-2 | 3.83 GB | 3.92 GB | no |
  | DeepSeek | 4.00 GB | 3.75 GB | no |

  Every subject gets `useExtraBuffers: false`: no repack, no KleidiAI, plain rows read from the file mapping (`CPU_Mapped`).
- **What r39 does:**
  - `KitModel.repack` and `RunSpec.repack` are now `bool?`, and the three models carry `repack: null`.
  - `ensureLoad` (`research.dart:1397`) sets `_repack = on` = null, so `_params` applies the app's own rule unchanged.
  - The block's first row records `load_rule` (`total_ram`, `file_bytes`, `can_copy_weights`, `use_extra_buffers`) and
    `load_lines`: every llama.cpp log line of that load with `model buffer size`, `CPU_Mapped`, `CPU_REPACK`, `KLEIDIAI`, `repack`,
    `n_ctx`, `type_k`/`type_v`, `flash_attn`, `special tokens cache` or `vollie-vocab`.
  - Those lines are the buffer type actually used. The parser prints it per model ("buffer types").
- **Not built:** a repack or KleidiAI arm. That is choice B, which was not adopted.

## 2. What is in it

### Runtime

- **New patch `patches/llama.cpp-c3/0013-vocab-control.patch`** (30 lines). It changes only `src/llama-vocab.cpp` (libllama), and
  is off unless `VOLLIE_VOCAB_CONTROL=<id>[,<id>...]` is set.
  - With the knob set, those token ids get the control attribute before the special-token cache is built.
  - So a prompt tokenized with special-token parsing (llamadart does: `llama_cpp_service.dart` 6358-6369) matches their text as
    one token.
  - It prints one info line per id: `vollie-vocab-control id= text=`.
  - Text tokenized without special parsing (`llama-perplexity`) is unchanged.
  - The kit sets `VOLLIE_VOCAB_CONTROL=6` only while Yi loads (`KitModel.load`). This is the owner's 2026-11-24 decision that
    `<|im_start|>` is one token.
- **Series in the APK:** `patches/llama.cpp` 0001-0003, `llama.cpp-c1-olmoe` 0001-0012, and c3 0001, 0002, 0003, 0005, 0007, 0008,
  0009, 0012, 0013.
  - Tree: `lab/llama.cpp-c3r39` (a copy of the r38 tree plus 0013). `tools/c2_build_kit.sh`'s tree check passed.
- **0004 pin guard: not included.** The placement logging it would add is already in r28's lines:
  - `c2-pool` gives each worker's affinity read-back at the last graph start and a per-graph `sched_getcpu` histogram.
  - `c2-sched` gives decode-phase time on cpu, run-queue time and context switches.
- **CPU module:** `libggml-cpu-android_armv8.2_2.so`, byte-identical to r33-r38 (`46e0735d…9694`). The only rebuilt native
  library is `libllama.so`.

### App: `patches/app-c3/0009-kit-r39-decode.patch` (613 lines, on 0008)

It changes `lib/research.dart`, `lib/settings_screen.dart`, `pubspec.yaml` and `test/kit_lines_test.dart`.

- **Version and buttons**
  - Version 1.39.0+39 (kit r39, versionCode 39). Same package, `com.insaneutils.vollie.c2`, and the same certificate.
  - Run button: "Run research kit r39 (R73: stock decode, Yi, Llama-2, DeepSeek; ~50-80 min)".
  - The c3 probe button is removed. Its code stays.
- **Model table:** only the three frozen files.

  | id | file | bytes | sha256 | load env |
  |---|---|---|---|---|
  | `yi15-6b` | `yi15-6b-q4_0.gguf` | 3,479,326,848 | `702cc2ebaa3bd4933598e25b556961b95ac260ed15df74fa09df2b2b7b48a299` | `VOLLIE_VOCAB_CONTROL=6` |
  | `llama2-7b` | `llama2-7b-q4_0.gguf` | 3,825,808,512 | `8eb4952f2592fdfc1288695f6b85aaf48d5628ec5d2c75f9387d3144652ad4bf` | none |
  | `deepseek7b` | `deepseek7b-q4_0.gguf` (unmodified; see §6) | 4,000,061,152 | `f403ddc298020af0fe8dfbbfec9a74cd3f8951938d6c21540054d85f09315118` | none |

  - URLs: `https://huggingface.co/HoeioUser/vollie3/resolve/main/<folder>/<file>`, with folders `yi15-6b/`, `llama2-7b/` and
    `deepseek7b/`.
  - `main` is not pinned; every file is sha256-checked before it is renamed into place (§3).
  - Template `model` (the file's own). Settings `_c1Ship`, no numerics. Granite and OLMoE are out of the table.
- **Run list** (`blockPlan`, block `claimp`): Yi, then Llama-2, then DeepSeek. Per model:
  1. load (forced at block start);
  2. 2 warm-ups (no gate, no hold);
  3. 10 `official32` runs (r16 gate: below 0.65 or 90 s; then the hold);
  4. 1 `profiled32` run (gated, unscored).

  39 runs in all.
- **Env of every run:**
  - `GGML_TINY_OP=262144`, `GGML_EXPERT_IO=1`, `VOLLIE_SAMPLER_BYPASS=1` (`_c1Ship`);
  - `GGML_C2_IO=1`, `VOLLIE_KV_DONTNEED=1`, `VOLLIE_SCHED=1`;
  - `VOLLIE_POOL=1` with `VOLLIE_CPUMASK=0x10,0x20,0x40,0x80` (thread *i* on cpu 4+*i*, R72's 4-7 cell);
  - `VOLLIE_EXEC_MAP=1`.
  - The profiled run also gets `GGML_CPU_PROF=1`, `GGML_CPU_AB_TOKEN=1` and `GGML_CPU_AB=knob:MMID_FUSE+CHUNK2:1:1`. These are r16
    and r26's per-token lines under an A/A label.
  - Not set: touch, nice and any numerics knob.
  - `VOLLIE_EXEC_MAP` is Vollie 2's shipped setting for every model (`app-prod3/lib/engine.dart:82`), so I kept it as part of the
    app's own path. It is bit-identical. Drop it in the prereg if the lead does not count it as choice A.
- **Threads and context:** 4 threads, pinned by the app to the fast cores and by the pool to one cpu each. `nCtx` 512.
- **Prefix reuse: off, because the app cannot do it for this prompt.**
  - The vendored llamadart re-ingests an exact full-prompt replay even with `reusePromptPrefix: true` (`llama_cpp_service.dart`:
    `reusedPrefix == nTokens` goes to `_decodeAndCacheFullPrompt`).
  - The official prompt is identical in every run, so nothing would be reused.
  - The report says so (`r73.reuse_prefix`).
- **DeepSeek prompt:** candidate a, the app's render as is (literal `<s>`, no BOS token, 156 tokens on host per ebony-joins-r14).
  Nothing changed.
- **Downloads land where the kit loads from.**
  - "Download kit r39 models into internal storage" now writes to `filesDir/kit-models` (`Research.kitIntDir`, `research.dart:558`).
    This is the folder `kitStorageStep` loads from.
  - The free-space check uses that folder's free bytes: 11.3 GB plus 200 MB.
  - `kitStorageStep` then finds the internal files, hashes them and uses them. Nothing is in the external folder, so there is no
    copy. Phone storage is 11.3 GB, not 22.6.
  - If the internal folder ever refused an executable mapping, r39 still runs from the internal files, with `exec_factor` void
    (`research.dart:752`). Before r39 such a session stopped.
- **Per-run logging added (K-d to K-i), all outside the timed call except the 1 Hz timer:**
  - **K-d, K-g:** `clk_1hz`, a 1 Hz timer during the run. Each sample is
    `[ms, cur0, cur4, cur7, max0, max4, max7 (MHz), cur_freq of every /sys/class/devfreq node]`.
    - Once per session, `r73.clk` records `cpuinfo_max_mhz` per policy and, per devfreq node, `governor`, `min_freq`, `max_freq` and
      `available_frequencies` (`''` = unreadable).
    - THP `enabled` was already in the header.
  - **K-e:** `zones_gate_start`, `zones_before` (existing) and `zones_after`.
  - **K-f:** `core_before` and `core_after`: `online`, `isolated`, `cpu7_online`, and `cpu{0,4,7}/core_ctl/*` (`e: …` if the folder
    is not readable).
  - **K-h:** `text_ids`, the reply re-tokenized with `addSpecial: false`. Equal ids mean equal text. They are not the sampled ids,
    which llamadart does not expose. The text was already recorded.
  - **K-i:** `mem_sys_before` and `mem_sys_after` (zram0 `mm_stat`, PSI memory and cpu). r28 already recorded majflt, minflt, VmRSS
    anon/file/swap, MemAvailable, vmstat, smaps and residency per run.
  - **K-b, K-c:** in the runtime's existing `c2-pool` and `c2-sched` lines (above), plus `thread_samples` and `workers_after`.
- **Unit test:** `test/kit_lines_test.dart` replaces r28's run-list tests with an R73 run-list test. `flutter test` passes (13 tests).
  `flutter analyze lib` shows no errors: 5 unused-field warnings (old r28 constants) and 2 old doc-comment infos.

### Parser: `tools/c3_decode_parse.py` (new, 207 lines)

- **Usage:** `python3 -I tools/c3_decode_parse.py <vollie-research-r39-*.json>`; `--selftest` runs the self-test.
- **Sections:**
  1. **Conditions:** power saving, deep-sleeping label, storage used, exec factor and hardware maximum. Void echoes: charger, and
     power saving per run.
  2. **Files and load path:** bytes, `sha256_end` against expected (a mismatch voids that subject), `canCopyWeights`, extra
     buffers, buffer types from `load_lines`, rendered prompt tokens.
  3. **Runs table:** decode tok/s, prompt tokens, ttft, gate end and timeout, heat, majflt, wait-lost MB, resident MB, the lowest
     cpu-7 cap in the run, and the per-worker labels.
  4. **"R73 inputs" per model:**
     - over the scored runs: their number, median, minimum and maximum decode tok/s;
     - the all-C subset (count and median);
     - runs with an off-core worker;
     - distinct reply texts, prompt-token counts, gate timeouts;
     - the profiled run's median us/token, head and rest, and the lowest mhz4 and mhz7.
- **Labels:** R72's four-way M / U / P / C, with the same default thresholds (`ON_MIN` 0.9, `CLK_MIN` 0.9, `CW_MIN` 0.9, `NIV_HI`
  50), applied to the decode's own lines:
  - M: `c2-pool`'s histogram, the share of the worker's graph starts on its own cpu;
  - U: the lowest 1 Hz `scaling_cur_freq` of its policy against `cpuinfo_max_freq`;
  - P: decode-phase `c2-sched` involuntary switches, and time on cpu ÷ (on cpu + run queue).

  This is descriptive. The off-core pattern is measured, not voided. Whether a label voids a run is the prereg's call.
- **Self-test:** a synthetic report with six runs. It exercises the M, U and P paths, a second reply text, the CPU_Mapped buffer
  line and the profiled split. It prints `selftest r73 ok`.
- **Not done:** a run on a real r39 report (none exists). No numbers are pre-registered in the parser; the K0 windows belong in
  the prereg.

## 3. Hashes for the pre-registration

| item | file | sha256 |
|---|---|---|
| runtime patch | `patches/llama.cpp-c3/0013-vocab-control.patch` (30 lines) | `7fef37cdb28ee8d1996c2efe33577dfa95d92d46f20a069a42d363289c257659` |
| vocab source after the series | `src/llama-vocab.cpp` (tree `lab/llama.cpp-c3r39`) | `be4328806be314fdf5b963c701af19198d07ac798e6b0b852f0a31d0a12d75b5` |
| app patch | `patches/app-c3/0009-kit-r39-decode.patch` (613 lines, on 0008) | `7f7c9c3f034aa599c28e269a860000c47235d4bc8985e82558e67693cd83b80a` |
| app source | `lab/app-c3/lib/research.dart` after 0009 | `5601ae51c133629e6576137644605bc243c2c30e1f444af5def5eb90dc81120e` |
| parser | `tools/c3_decode_parse.py` (working copy) | `9b3a4c78f0b382da18811fcc6f7ff56e3ac409844f25cf813ed2cc29804624b9` |
| APK | `lab/dist-c3/r39/vollie-c3-decode-r39.apk`, 16,706,066 B, versionCode 39, versionName 1.39.0, package `com.insaneutils.vollie.c2`, certificate SHA-256 `c49dde856ca5bb044caf878a39a03077a8f958d841c47510672f7d269fbb1716` (same as r38; `apksigner verify`) | `ee6777653a2b07e7667e8065ba495231f8222bb90a6b677a3d1e49c114979470` |
| build record | `lab/dist-c3/r39/BUILD-r39.txt` | `b88911a6cfd219e4a067785ca96c37f08aff6c47135ba3df66fc48c29434198d` |
| sums | `lab/dist-c3/r39/SHA256SUMS` (APK and BUILD; `shasum -c` OK) | `2977171c7f08668ab5836691f203539e07ca264c6c94fc8155b7ad18f013816d` |
| kit_build.json in the APK | `lab/app-c3/assets/kit_build.json` | `d9bc79bed917d4330323db071a6b343998ebe7e3086a0f701d159690e5837ef8` |
| served Yi | `yi15-6b-q4_0.gguf`, 3,479,326,848 B | `702cc2ebaa3bd4933598e25b556961b95ac260ed15df74fa09df2b2b7b48a299` |
| served Llama-2 | `llama2-7b-q4_0.gguf`, 3,825,808,512 B | `8eb4952f2592fdfc1288695f6b85aaf48d5628ec5d2c75f9387d3144652ad4bf` |
| served DeepSeek | `deepseek7b-q4_0.gguf`, 4,000,061,152 B | `f403ddc298020af0fe8dfbbfec9a74cd3f8951938d6c21540054d85f09315118` |

The three served-file hashes were re-checked on the laptop today and equal the freeze logs (`lab/c3/logs/freeze-*.log`, line 2).

**Libraries in the APK, against r38** (checked file by file):

| library | sha256 | against r38 |
|---|---|---|
| `libllama.so` (+ c3 0013) | `db62e5cf233babfc19ae9cd4bd38f0dde21e90d7374d85ae51b92a1e4815cbe4` | changed |
| `libapp.so` (Dart) | `a8ba9a19db3b89aaffec2d5af58f8457ed2da65ba07959d0d6f3cbd23e15c487` | changed |
| `libggml-cpu-android_armv8.2_2.so` | `46e0735d2824ba6f3a156e219e058a701f6024fdbecbb2ef937b3127115a9694` | byte-identical (= r33 to r38) |

Every other library (libggml-base, libggml, the other CPU variants, libllama-common, libllamadart, libmtmd, libflutter,
libdatastore_shared_counter, libvollie_clc, libvollie_x4c) is also byte-identical to r38.

**Build commands:**

```
cp -R lab/llama.cpp-c3r38 lab/llama.cpp-c3r39            # then 0013 applied (src/llama-vocab.cpp)
sh android/build_cpu.sh lab/llama.cpp-c3r39 <libs>        # libllama.so from here; the CPU module replaced by r38's file
LIBS_FROM=<libs> APP_SERIES="<app-c2 0001-0012> patches/app-c3" sh tools/c2_build_kit.sh lab/llama.cpp-c3r39 lab/app-c3 \
  patches/llama.cpp patches/llama.cpp-c1-olmoe <dir with c3 0001 0002 0003 0005 0007 0008 0009 0012 0013>
```

After the build, the APK was copied to `lab/dist-c3/r39/vollie-c3-decode-r39.apk`, and BUILD-r39.txt got r38's clc / x4c notes
plus a series note and a run-list note. The build-time copies are in `lab/dist-c2/`.

**Report field names (new in r39):**
- per run: `load_rule`, `load_lines` (block's first row), `zones_gate_start`, `zones_after`, `core_before`, `core_after`,
  `mem_sys_before`, `mem_sys_after`, `clk_1hz`, `text_ids`;
- report: `r73` (`plan`, `choice_a`, `kv_cache`, `threads`, `reuse_prefix`, `yi_vocab`, `deepseek_prompt`, `pin_guard`, and `clk`
  with `cpuinfo_max_mhz`, `devfreq` and `clk_fields`).

Unchanged from r28 and used by the parser: `decode_tps`, `prompt_tokens`, `ttft_ms`, `gate_timeout`, `wait_end`, `heat`, `majflt`,
`wait_lost_mb`, `res_after`, `text`, `c2` (the `c2-pool` and `c2-sched` lines), `tok` and `tok_format`, `models.*.sha256_end`,
`templates.*.rendered_tokens`, `storage`, `session_power`. Run ids: `<model>_off_wu`, `<model>_off_claim`, `<model>_off_prof`.

## 4. Phone time and memory (*estimated*)

| step | estimate | basis |
|---|---|---|
| download, one time, before the session (button), Wi-Fi | 11.3 GB: about 4 to 10 min at 20 to 50 MB/s, plus 3 sha256 passes of 10 to 30 s each | the r27 `_fetch` path |
| session storage step | 3 sha256 passes, 30 to 90 s in all | |
| idle baseline | 30 s | |
| per model: load + 2 warm-ups | 40 to 90 s; the first warm-up reads the whole file from flash (3.5 to 4.0 GB) | |
| per gated run | 20 to 130 s: gate 0 to 90 s, hold up to 8 s, prefill 153 to 186 tokens at about 8 to 20 tok/s on plain rows (8 to 23 s), decode 32 tokens at 4 to 7 tok/s (5 to 8 s) | the prefill rate on choice A has never been measured on the phone |
| per model, 11 gated runs | 4 to 24 min | |
| **all three** | **about 20 to 80 min; about 55 to 75 min if most gates run to 90 s** (the button says ~50-80) | |
| end hashes | 30 to 90 s | |

**Memory per model** (choice A; nothing is copied):

| subject | file-backed (the mapped file) | KV q8_0 at n_ctx 512 | compute and logits buffers | app |
|---|---|---|---|---|
| Yi | 3.48 GB | about 18 MB (4 KV heads) | about 0.1 to 0.3 GB | about 0.2 GB |
| Llama-2 | 3.83 GB | about 143 MB | as Yi | as Yi |
| DeepSeek | 4.00 GB | about 134 MB (30 layers) | as Yi | as Yi |

- Anonymous memory is under 0.6 GB on all three, against R72's 3.4 GB held.
- Residency risk is the file-backed body: R69 held 3.4 GB mapped in both runs, and 4.2 GB 1 of 2 read-only, 2 of 2 executable.
  The executable mapping and the hold are on.
- The app's own olmoe-Mi (3.1 GB) must be on the phone and loaded to start the kit. It is unloaded at Yi's load.
- Storage: 11.3 GB internal plus 200 MB margin.

## 5. Host checks (*host*; host timings say nothing about the phone)

**Per-chunk PPL identity against the freeze build.**
- Freeze build: `lab/llama.cpp-c3/build`, `llama-perplexity` sha256 `e5264b7c…`, re-hashed today.
- r39 build: a macOS build of `lab/llama.cpp-c3r39` with the freeze build's CMake flags (CPU, armv8.2-a+dotprod+fp16, repack on,
  KleidiAI off, llamafile on).
- Command: `-c 512 -b 512 -t 8 -ngl 0` on Yi's served file and the freeze text.
- Arms: freeze, r39 with every knob unset, and r39 with the kit's env. The kit's env is the run env above plus
  `VOLLIE_VOCAB_CONTROL=6` and `VOLLIE_NO_ATTACH=1`, so that the pool runs: `c2-pool ... pool_graphs=5/0`.

| setting | freeze | r39 knobs unset | r39 kit env + vocab knob |
|---|---|---|---|
| `-b 512`, 4 chunks | 5.0145, 6.9742, 7.4150, 8.4733 | identical | identical |
| `-ub 32`, 4 chunks | 5.0084, 6.9175, 7.3651, 8.4080 | identical | identical |
| `-ub 1`, 2 chunks | 5.0373, 6.9391 | identical | identical |

The freeze's own full-set logs (`lab/logs/p/ppl-c3f-yi15-6b-q4_0-{b512,ub1}-wiki.log`) have the same first chunks.

**This identity is on the freeze's settings: F16 KV and repack on.**
- The phone's choice-A configuration is different: plain rows, and q8_0 KV with flash attention, which is the app's own setting.
  Bonnie r17 §1.2 already noted that it needs its own reference: `llama-perplexity --no-repack -ctk q8_0 -ctv q8_0 -fa on`.
- That is not a change made by r39, and no lever is scored in R73.
- Not run on the full set.

**0013 does what it is for (Yi, `llama-tokenize`, the official prompt in Yi's ChatML).**
- Without the knob: 167 tokens, no id 6 (`▁<`, `|`, `im`, …), as ebony-joins-r14 found.
- With `VOLLIE_VOCAB_CONTROL=6`: 156 tokens, id 6 twice, id 7 once.
- Against the publisher's `tokenizer.json` (`lab/c3/dl/yi15-tokenizer/`): the same count, 156, but 3 positions differ.
  - llama.cpp's SPM path puts a space-prefix piece after each special token: `▁user` instead of `user`, `▁` plus `\n` after
    `<|im_end|>`, and `▁assistant` instead of `ass` + `istant`.
  - That is `add_space_prefix` in `llama-vocab.cpp` (line 3616), not the marker.
  - The owner's decision covers the marker only. The space prefix is not changed. If it matters, it is a new owner question.

**Builds and checks.**
- **Kit:** `tools/c2_build_kit.sh` passed its tree check, the library check and the kit_build.json check. `apksigner` and `aapt2`
  confirm the package, version and certificate.
- **App patch:** applied to r38's four files, it reproduces r39's (`cmp`).
- **Tests:** `flutter test test/kit_lines_test.dart` passes. The parser self-test passes.
- **HF:** `https://huggingface.co/api/models/HoeioUser/vollie3` returns 401 today (the repo does not exist, or is not public).

## 6. Files the lead must upload to `HoeioUser/vollie3` (public; nothing was uploaded)

| local path | repo path | bytes | sha256 |
|---|---|---|---|
| `lab/c3/models/yi15-6b-q4_0.gguf` | `yi15-6b/yi15-6b-q4_0.gguf` | 3,479,326,848 | `702cc2ebaa3bd4933598e25b556961b95ac260ed15df74fa09df2b2b7b48a299` |
| `lab/c3/models/llama2-7b-q4_0.gguf` | `llama2-7b/llama2-7b-q4_0.gguf` | 3,825,808,512 | `8eb4952f2592fdfc1288695f6b85aaf48d5628ec5d2c75f9387d3144652ad4bf` |
| `lab/c3/models/deepseek7b-q4_0.gguf` | `deepseek7b/deepseek7b-q4_0.gguf` | 4,000,061,152 | `f403ddc298020af0fe8dfbbfec9a74cd3f8951938d6c21540054d85f09315118` |

- **DeepSeek is the unmodified file.** clyde-freeze.md §1 and clyde-r73-plan §7 name `deepseek7b-bos-q4_0.gguf` (`f195d24f…`).
  That was superseded on 2026-11-22: the owner refused the BOS change, and the freeze log, line 2, served `deepseek7b-q4_0.gguf`.
  Do not upload the `-bos` file.
- The licence files (Llama 2 Community License plus its NOTICE line, the DeepSeek model licence, apache-2.0 for Yi) and the card
  go in the same commit series, as plan §7 says.
- **The kit's URL uses `resolve/main/`.**
  - If the lead wants a pinned commit in the URL, that is a one-line change of `Research._vollie3` and a rebuild of `libapp.so`
    only.
  - Integrity does not depend on the pin: every file is sha256-checked before use, and a wrong file is deleted.

## 7. Not done, limits

- **Not run anywhere:**
  - the kit on a phone;
  - the r39 Dart code against real llamadart output: `load_lines` matching, the devfreq readability and `text_ids` are untested
    off the phone;
  - the 1 Hz timer's own cost. One sysfs pass per second on the UI isolate is expected to be negligible, but it is not measured.
- **K-h** gives re-tokenized text ids, not the sampled ids.
- **c2-pool's histogram** covers prefill and decode graphs together. There is no decode-only split without a runtime change.
- **No pin guard (0004),** as the lead decided.
- **Housekeeping (Annie or the lead):**
  - the series and app READMEs need their 0013 and 0009 rows;
  - CLAUDE.md and STATUS should note that DeepSeek's served file is the unmodified one;
  - the plan's K0 windows and labels go into the prereg.

## Summary

1. Kit r39 is built: `lab/dist-c3/r39/vollie-c3-decode-r39.apk` (1.39.0+39, same certificate, sha256 `ee677765…9470`), with
   BUILD-r39.txt and SHA256SUMS. Not uploaded, not committed.
2. Choice A is the app's own rule (`device.dart:591`, `assistant.dart:456-467`). For all three files it gives no copy, so plain
   `CPU_Mapped` rows and q8_0 KV. The kit records the rule and the buffer lines per load.
3. Run list: Yi, Llama-2, DeepSeek, each with 2 warm-ups, 10 gated official32 runs and 1 profiled run, on 4 threads on cpus 4-7,
   n_ctx 512.
   - Prefix reuse is off, because llamadart re-ingests identical prompts.
   - The files download straight into the internal folder the kit loads from (11.3 GB, no second copy).
4. Runtime: one new libllama patch, 0013 (`VOLLIE_VOCAB_CONTROL`, Yi's `<|im_start|>` as one token: 167 → 156 tokens, *host*). The
   CPU module is byte-identical to r38. Per-chunk PPL equals the freeze build at `-b 512`, `-ub 32` and `-ub 1` on Yi, with the
   knobs unset and with the kit env (*host*).
5. The lead must upload the three files listed in §6 to the public `HoeioUser/vollie3`. DeepSeek is the unmodified
   `f403ddc2…5118`, not the `-bos` file. The parser `tools/c3_decode_parse.py` (self-test passes) reports labels and medians per
   model for the prereg.
