# Memo 10: research kit r17 (app side), stock-only build

Agent model: claude-opus-5-5

Date 2026-11-14. Author: Tom (code). **Nothing here ran on a phone. No speed is stated.** The APK was built and
inspected on the laptop (*host*). Run-time figures are *estimated* from campaign 1's r16 report.

## 1. What the kit needs from the runtime (for the patch series in `patches/llama.cpp-c2/`)

The environment-variable names are the patch author's. The app sets variables with `setenv` and clears them with
`unsetenv` between runs, in one process, without a reload unless stated. These are the requirements:

1. **Knobs re-read per graph, not per process.** Every lever's variables are read at the start of every
   `ggml_graph_compute` in the CPU module. libllama must not latch them either: today `c2/0001` holds
   `VOLLIE_SKIP_*` in a `static const` in `build_moe_ffn`, and the app's one-token warm-up at load fixes it before
   any kit run. If a knob changes the graph, either wire the graph for it under a separate load-time switch (the kit
   reloads for those) or make llama.cpp's graph reuse rebuild when the knob changes. Unset and empty both mean off,
   and off must be bit-identical to stock at `-ub 1`, `-ub 32`, `-b 512`.
2. **Load-time-only switches are declared as such** in the patch README, with their names. The kit reloads the
   model for them, as it does for repack.
3. **Counters per run, without process exit.** At the first graph of every prompt (n_tokens > 1), print one line
   with the counters gathered since the previous prompt, then reset them. The same applies to
   `VOLLIE_SKIP_STATS` (tokens, pairs, % pairs skipped, % expert bytes skipped, MB per token). The kit keeps every
   stderr line that starts with `c2-`, `cpu-prof` or `cpu-bench`.
4. **Per-token profile lines** in campaign 1's shape, when the profiler variable is set: one line per one-token
   graph, `cpu-prof ab=<A|B> tok=<n> us/token=<µs> majflt=<n> minflt=<n> mhz=<cpu4>/<cpu7> … | <OP> <count>:<µs
   compute>+<µs wait> …`. Keys are `key=value` before the `|`, and `tok` counts from 1 at each prompt.
   `tools/c2_research_parse.py` reads exactly this; it already parses r16's lines.
5. **In-run A/B hook** like campaign 1's `GGML_CPU_AB=knob:<NAME>:<A>:<B>` with `_WARM`, `_BLOCK`, `_TOKEN`,
   `_BAAB`: the side sets the knob's value per graph; `knob:<NAME>:0:0` is the A/A null; counting restarts at each
   prompt; every c2 lever has a `NAME`. The `ab=` side goes into the per-token line.

## 2. What was built

| item | value |
|---|---|
| APK | `lab/dist-c2/vollie-c2-1.17.0+17-research-r17.apk` (private), **17,098,002 B**, sha256 `5b7ae90614dee32b0b2eb5ae5a7a7db0923b0b16aae8bae36135fb3393994f27` |
| package / version | `com.insaneutils.vollie.c2`, label "Vollie C2", versionCode 17, versionName 1.17.0, minSdk 24, target 36 |
| mode / signer | release (AOT: `libapp.so` present), APK Signature Scheme v2, `CN=Android Debug` (certificate SHA-256 `c49dde85…fbb1716`) |
| ABI | `arm64-v8a` only |
| `libllama.so` in the APK | 4,286,400 B, sha256 `fadab87e6d9790948430396fa6ee2ad9f9112dec32dd571df444631275af2e45`, byte-identical to the build |
| `libggml-cpu-android_armv8.2_2.so` in the APK | 1,644,688 B, sha256 `80e614a729a2d478a4666ab10504d98e6cf28085c9a39defa2ea1dd2a20ba26d`, byte-identical to the build |
| runtime source | llama.cpp `620d2225c` + `patches/llama.cpp-c2/0001-olmoe-expert-skip.patch` (sha256 `62b02dc0…e9e42a`) only, in the private scratch tree `lab/llama.cpp-c2-android`. Both libraries contain the `VOLLIE_SKIP_*` names and none of campaign 1's knob names (`GGML_TINY_OP`, `GGML_HEAD2`, `GGML_CPU_PROF`, `VOLLIE_SAMPLER_BYPASS`) |
| other native libraries | llamadart's `v0.7.0-2` bundle (`libggml-base.so`, `libllama-common.so`, `libllamadart.so`, …), unchanged |
| app changes | `patches/app-c2/0001-kit-r17.patch` against app 2.3.0+45 (README there); applies cleanly to a pristine copy (checked) |
| build script | `tools/c2_build_kit.sh <llama.cpp tree> <app copy>` |
| parser | `tools/c2_research_parse.py` (`--selftest` passes) |

The CPU module built earlier the same day from the same tree had a different sha256 (`97b85b39…`). The build is not
bit-reproducible across runs, so a kit is identified by the hashes the kit itself reports (`libs`), not by a rebuild.

## 3. What the kit does

One button, **Settings → "Run research kit r17 (~5-20 min)"**. The kit runs the spec list `Research.runs` in
`lib/research.dart`, top to bottom. Each spec is one line:

```dart
(id: 'stock', group: 'stock', kind: 'official32', env: {}, repack: true, threads: 0, reps: 5, pause: 0),
```

- `kind`:
  - `official32`: the benchmark. The prompt is `_benchPrompt` (byte-identical to `tools/c2_official_prompt.txt`;
    the report carries it). 32 tokens, `temp: 0` with llamadart's default repeat penalty 1.1, the app's normal
    streaming, `decode_tps` = llamadart's `decodeTokensPerSec`.
  - `profiled32`: the same with one stream chunk per token and every chunk's arrival time kept.
  - `abba`: story 200 with the spec's env and chunk times (for an in-run A/B once the runtime has the hook).
- `env`: set before the run, after every known knob is unset (campaign 1's names, campaign 2's names, and every name
  in the list).
- `repack`: llama.cpp's weight repack at load. A spec whose value differs from the loaded model reloads it
  (`reload_ms`).
- `threads`: 0 means the app's pick (3 on the SD888 without a saved speed test).
- `reps`: repetitions in a row. `pause`: idle seconds after the cool-down gate.

The list as built (stock only, all `env` empty):

| id | group | kind | repack | reps |
|---|---|---|---|---|
| `cold` | baseline | official32 | on | 1 |
| `stock` | stock | official32 | on | 5 |
| `stock-prof` | profiled | profiled32 | on | 2 |
| `stock-nr` | norepack | official32 | off (reload) | 2 |

Greedy identity comes from the `text` of every run, kept whole.

Fixed for every run:
- the cool-down gate (Android thermal headroom < 0.65, at most 90 s, as r15/r16);
- **prompt-prefix reuse off** (`Research.reusePrefix = false`). OLMoE is attention-only, so llamadart would
  otherwise reuse the previous identical prompt's KV cache and skip the prefill. Campaign 1's Granite always
  re-read the prompt. Lead: flip the constant if the benchmark should include the reuse;
  `cached_prompt_tokens` shows which happened.

Before the runs, the kit records:
- the sha256 of every mapped `ggml`/`llama` library (from `/proc/self/maps`);
- the model's metadata and the chat template: architecture, the GGUF's `tokenizer.chat_template`, whether a custom
  template is set (it is not), the rendered benchmark prompt and its token count;
- the model's size and the download's verification marker.

After the runs it hashes the whole model file again (last, so the hash does not warm the page cache before the
timed runs). At the end every knob is unset and the model is reloaded with the app's own repack rule.

No Granite default is left. The served skip 117, the two-stage head, the sampler bypass and the saved or fallback
`GGML_TINY_OP` / `GGML_EXPERT_IO` are gone, so the stock group runs the unmodified runtime. Outside the kit the app's
repack rule now gives **on** for olmoe-Mi (`copyBytes` 235,505,664 B, memo-05; 7.3 GB − 0.24 GB ≥ 5 GB), the kernel
path of the R50/R51 host perplexities.

**What this build cannot do yet:** any lever run. With `c2/0001`, `VOLLIE_SKIP_*` set by the kit does nothing (item
1 above), the `c2-skip:` line never prints, and there are no `cpu-prof` lines, so `profiled32` per-token times come
only from the Dart side (`chunk_ms`, which includes the time outside the graph). When the patches of §1 land, the
lead adds spec lines (for example `env: {'VOLLIE_SKIP_MASS': '900', …}`) and rebuilds; no other app change is needed
unless a lever is load-time only.

## 4. Report schema (JSON, one object)

```
kit, app, started, finished, stopped_early
header [str]            chip, RAM, context, CPU parts, max clocks, fast cores, features, GPU, setting (r16's)
saved                   {write_threads, context, reuse_prefix}
kernel, thp, mem_start, mem_end
specs [spec]            the run list as built: {id, group, kind, env, repack, threads, reps, pause_s}
prompt                  the benchmark prompt text
libs {name: sha256}, lib_paths [str]
template                {architecture, model_name, custom_template, model_template, rendered, rendered_tokens, format}
model                   {file, bytes, expected_bytes, verified_marker, sha256_end}
notes [str]
runs [run]:
  id, group, kind, rep, env, repack, repack_loaded, threads, reload_ms (only when it reloaded)
  cool_ms, heat (headroom at start), gate_timeout, pause_s, idle_ms (since the previous run's end)
  batt_c_before, batt_c_after, plugged, heat_after
  mhz_before, mhz_after   "mhz=<cpu0>,…,<cpu7>" MHz
  mem_before, mem_after   "avail= cached= swapfree= anon= file= swap=" MB (anon/file = this process's RssAnon/RssFile)
  majflt, minflt, io_read_bytes   process-wide, over the run (prefill included)
  wall_ms, tokens, prompt_tokens, cached_prompt_tokens, ttft_ms, duration_ms, decode_tps, prefill_tps
  chunk_ms [ms]           profiled32/abba: arrival time of each token's chunk from the request
  prof [str]              runtime lines (c2-*, cpu-prof*, cpu-bench*) printed during the run
  text                    the whole reply
  sample_ms, decode_ms    llamadart's sampling and llama_decode time
  error                   exception text if the run failed
```

Parse: `python3 -I tools/c2_research_parse.py <report.json>`. It prints:
- the identity checks: model size, marker and final hash against `5abf710b…`; the library hashes; whether the
  rendered prompt equals OLMoE's own template applied to the prompt (`<|endoftext|><|user|>\n…\n<|assistant|>\n`);
  whether the prompt equals `tools/c2_official_prompt.txt`; the prompt token counts seen;
- per group: n; median, min and max printed tok/s; steady ms/token (median over tokens 2 and later, from runtime
  lines if present, else from `chunk_ms`); big-core MHz after the run; the median majflt and minflt;
- the runs that break hygiene: big-core clock below `--min-mhz` (default 1800), major faults in decode tokens (only
  with runtime per-token lines), cool-down gate timeout, cached prompt tokens, error, or a stock run's text that
  differs from the first stock run.

The rendered token count is reported, not forced. The host renderer gave 151 tokens; Granite's was 155.

## 5. What the owner does

1. The lead puts the APK where the owner can download it on the phone. Size 17,098,002 B; check sha256
   `5b7ae906…94f27`.
2. Open the file on the phone and install. Android asks once to allow installs from that app (browser or Files),
   as with campaign 1's APKs. **"Vollie C2" appears as a second app; the owner's Vollie and its model are not
   touched.** Nothing is uninstalled.
3. Open Vollie C2 and download the model, about 3.07 GB, on Wi-Fi. The download needs about 3.2 GB free storage,
   resumes after a break, and checks the sha256 before the model is used.
4. When the chat screen is ready: Settings → **"Run research kit r17"**. Leave the phone alone with the screen on.
   The progress shows as `Research r17 · m:ss · <group> · <id> n/N`.
5. When it ends: **"Copy research results"** and paste the JSON into the reply. The same JSON is saved as
   `research-r17-<time>.json` next to the model, in the app's files folder (`Android/data/com.insaneutils.vollie.c2/files/`).

**Expected run time (*estimated*):** 10 timed runs. Each spends up to 90 s at the cool-down gate plus about 5 to 12
s of prompt and reply (r16: 5 to 12 s per official run on Granite). Add one reload for the repack-off group, a
final full hash of 3.07 GB (tens of seconds), and the end reload. Total about **5 to 20 min**, mostly cool-down.
The first model load after the download is not part of the kit.

## 6. Not verified

- **Nothing ran on a phone.** It is not shown that the app loads olmoe-Mi through llamadart, that llamadart renders
  OLMoE's template as llama.cpp does, that the c2 `libllama.so` works with the bundle's `libggml-base.so` on the
  device (same commit, same build script as campaign 1's), that repack on fits in memory beside the 3.07 GB mmap
  (*estimated* yes: 225 MiB), or that the download URL serves the pinned bytes to the app.
- The Dart code passed `dart analyze` (lib and test); it did not run.
- `chunk_ms` steadiness depends on llamadart emitting one chunk per token with both stream thresholds at 1; a chunk
  holding a partial UTF-8 character shifts one sample.
- Hosting the APK and handing it to the owner is the lead's step.
