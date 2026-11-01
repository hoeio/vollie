# 01 — How Vollie runs inference vs. stock llama.cpp

Scope: the Flutter app in `vollie/` (paths below are relative to it; `LD/` = `third_party/llamadart`,
the vendored llamadart 0.15.6). Chosen model: Granite-4.0-H-Tiny abliterated, Vollie's Q4_0 file
(see `00_model_choice.md`). Baseline: upstream llama.cpp **v0.8.0 @ 8642541b** (2026-10-28).
Everything below is read from the code; anything not read there is marked *inferred* or *assumption*.
No phone was available: every speed number here is **measured on host** (Apple M4 Pro, CPU only,
Metal off, armv8.2-a+dotprod+fp16 build to match the SD888's ISA) and is a relative number only.

## 1. Summary table

| Area | Vollie (as shipped) | Stock llama.cpp (llama-cli / llama-bench defaults) | Matters on SD888? |
|---|---|---|---|
| llama.cpp version | llamadart-native `v0.7.0-2` = llama.cpp **620d2225c** (2026-10-15) (`LD/lib/src/hook/native_release_pins.dart:8`, `tool/build_cpu.sh:6-7`) | v0.8.0 @ 8642541b | small, see §9 |
| Backend | CPU by default (`lib/assistant.dart:244-249`); OpenCL (self-built for Adreno 6xx) opt-in | CPU (Android) | CPU is the right default (OpenCL writes ~35% slower, user's phone test 2026-10-25) |
| CPU kernels | runtime-selected variant `android_armv8.2_2` (dotprod+fp16) of llamadart's bundle, **built with KleidiAI** (404 `kleidi` strings in the shipped .so), no OpenMP (`pubspec.yaml:47-59`) | same ISA if built for it; KleidiAI off by default | KleidiAI only acts through extra buffers, which Vollie disables on 8 GB phones (next row) |
| Weight repack (Q4_0 → 4x4 interleaved) | **OFF on 8 GB phones for Granite**: `useExtraBuffers: device.canCopyWeights(fileSize)` (`lib/assistant.dart:448`), `totalRam - model >= 5 GB` (`lib/device.dart:107`) → 7.3 GB − 3.98 GB < 5 GB | ON (`--repack 1`, `use_extra_bufts=true`) | **yes — biggest single difference**, §3 |
| Weights in memory | mmap (default `useMmap=true`, `useMlock=false`, `LD/.../model_params.dart:514-515`) | mmap | page cache can be evicted under pressure → re-read stalls (*inferred*) |
| Context | 8192 if total RAM ≥ 7 GB and ≥ 3.5 GB free, else 4096 (`lib/device.dart:97,118`) | `n_ctx=0` = trained context, shrunk by `--fit` (`common/common.h:459,487`) | Granite: 4 attention layers only → small KV |
| Batch | n_batch = min(n_ctx, 2048), n_ubatch = 512 (`LD/.../model_params.dart:785-827`) | 2048 / 512 | prefill only |
| KV cache | q8_0 K and V on CPU (`lib/assistant.dart:449-453`) → flash attention forced on (`LD/.../llama_cpp_param_values.dart:27-37`, applied after the CPU path disables it, `LD/.../llama_cpp_service.dart:4315-4332`; *inferred* final state FA=on) | f16, FA auto | negligible for Granite (4/40 layers have KV) |
| Threads | `performanceThreads` = big cores via MIDR part / `cpu_capacity` / max freq (`lib/device.dart:326-378`); same count for batch unless `optimize()` picked another (`lib/assistant.dart:266-274, 446-447`) | `-t` = hardware concurrency heuristics (4 on most phones) | yes |
| Core pinning | `sched_setaffinity` on every `DartWorker` thread to the big-core mask before each generation (`lib/device.dart:385-416`, called `lib/assistant.dart:503,506,1478`) | none unless `-C/--cpu-mask` | yes: keeps A55s out of barriers |
| Thermal | `heatPlan`: at thermal status ≥ 4 halve threads and **reload the model** (`lib/assistant.dart:1121-1202`) | none | yes for sustained runs |
| Sampling | top_k 20 → top_p 0.8 → temp 0.7 → dist, repeat penalty 1.1 over last 64 (`lib/assistant.dart:1484-1489`, `LD/.../generation_params.dart:962-966`, `LD/.../llama_cpp_service.dart:4826-4833, 6677-6805`) | penalties off, top_k 40, top_p 0.95, min_p 0.05, temp 0.8 | ≤ 1% of a token's time (*inferred*: 100k-vocab sort after top_k) |
| Decode loop | one `llama_decode` per token in a Dart `async*` loop on a worker isolate (`LD/.../llama_cpp_service.dart:6845-6911`) | C++ loop | per-token FFI overhead, §8 |
| Speculative / MTP | none: no `SpeculativeDecodingConfig` in `lib/`, `loadMtp` false (`LD/.../model_params.dart:516`) | off unless `--spec-type` | Granite has no MTP head |
| Prompt reuse | Vollie-level state save/restore of whole turns (`lib/assistant.dart:1223-1467`), system prompt primed to `turn_sys.state` (`:1316-1357`), prefill while typing (`:1017-1076`), pre-reply (`:976-1010`) | prefix cache in server only | perceived latency, not decode tok/s |
| UI streaming | tokens batched to the UI isolate every 8 tokens / 512 bytes (`LD/.../native_token_stream_batcher.dart:27-45`), `notifyListeners` per chunk (`lib/assistant.dart:1492-1502`) | stdout | negligible (§8) |
| Env knobs | no `GGML_*`/`LLAMA_*` env vars set anywhere | — | — |

## 2. Compute backends and kernels
- Vollie bundles only llamadart's CPU backend plus its own OpenCL module (`pubspec.yaml:41-59`,
  `tool/build_opencl.sh`). Default is CPU; OpenCL is only offered on Qualcomm SoCs and must pass a
  "2+2" self-test (`lib/assistant.dart:217-220, 480-494`). With `ComputeDevice.cpu`, llamadart sets
  `n_gpu_layers=0` and turns off KQV/op offload (`LD/.../llama_cpp_service.dart:836-842, 4315-4322`).
- The CPU module is chosen at runtime among four ISA variants; on the SD888 that is `android_armv8.2_2`
  (dotprod+fp16). The shipped .so contains KleidiAI (404 `kleidi` strings; verified on
  `build/native_assets/android/jniLibs/lib/arm64-v8a/libggml-cpu-android_armv8.2_2.so`) and no OpenMP
  symbols (ggml's own threadpool). Stock llama.cpp for Android builds the same ISA when configured so;
  KleidiAI is off by default.
- **Both** repack and KleidiAI are *extra buffer types*: when Vollie passes `use_extra_bufts=false`
  (every 8 GB phone with the Granite file, `lib/device.dart:107`), neither runs, and the Q4_0 mat-vec uses
  the generic `ggml_vec_dot_q4_0_q8_0` dotprod path straight from the mmap. That is the main kernel
  difference from stock llama.cpp, which repacks by default.
- Hexagon: not used (ggml-hexagon targets HTP v73+, SD888 is v68). Vulkan: removed from the bundle.

## 3. Quantization handling
- Same GGUF and same quant types as stock (Q4_0 i1 mix: Q4_0 + 5× Q4_1 `ffn_down_exps` + Q6_K tied
  embedding/lm_head + F32 router/norms/SSM params). No runtime requantization.
- `expert_used_count` is patched to **4** in the file (ModelDownloadService.kt:35-38); stock llama.cpp
  with the original file uses the model's **6**. This is a file edit, not an app setting, and it is the
  single largest quality cost (see `00_model_choice.md`: +12.4% PPL vs +4.2% at 6 experts).
- Repack (Q4_0 → 4x4 interleaved, `ggml-cpu/repack.cpp`) is off on 8 GB phones as above.

## 4. Memory layout / mmap
- Weights are mmapped (llamadart default `useMmap=true`); no mlock. With repack off, the only anonymous
  memory is the compute buffer (~211 MiB at ubatch 512), KV (4 attention layers × q8_0) and the 58 MB
  recurrent state. Host peak footprint: **128 MB** (repack off) vs **3,931 MB** (stock repack on) for the
  same file (`/usr/bin/time -l`, "peak memory footprint").
- Consequence on the phone (*inferred*): the 3.98 GB file lives in page cache; under memory pressure
  Android can evict expert pages, which are then re-read from flash on the next token that routes to
  them. Stock llama.cpp with repack would instead hold 3.6 GB anonymous (and was OOM-killed on this phone).

## 5. Threading and core affinity
- Threads = number of big cores detected from MIDR part numbers, `cpu_capacity` and max frequency
  (`lib/device.dart:326-378`); on the SD888 that should be 4 (X1 + 3×A78) — *to be verified on device*.
  Stock llama-cli uses `cpu_get_num_math()`, typically also 4 on 4+4 phones.
- Vollie pins every `DartWorker` thread with `sched_setaffinity` to the big-core mask before each
  generation (`lib/device.dart:385-416`). llama.cpp's own threadpool threads are created by ggml inside
  the worker, so they are pinned only if they inherit the mask at creation (*inferred*: they are created
  at context init, before `pinInferenceThreads` runs for the first generation — whether they end up
  pinned depends on whether the ggml threadpool is re-created per decode; **to be verified on device**).
  Stock llama.cpp offers `--cpu-mask/--cpu-strict` for exactly this, which llamadart does not expose.
- Thermal: at Android thermal status ≥ 4 Vollie halves threads and reloads the model
  (`lib/assistant.dart:1121-1202`); stock has nothing comparable.

## 6. KV cache and recurrent state
- q8_0 K/V with flash attention (forced on by llamadart for quantized KV). Granite has KV in only 4 of
  40 layers (34 MB f16 at ctx 4096), so this saves ~17 MB and costs nothing measurable. The 36 Mamba2
  layers carry a 58 MB f32 state that is read and written every token in both Vollie and stock.

## 7. Sampling
- Vollie: penalties(1.1, last 64) → top-k 20 → top-p 0.8 → temp 0.7 → dist, built natively via FFI
  (`LD/.../llama_cpp_service.dart:6677-6805`). Stock defaults: top-k 40, top-p 0.95, min-p 0.05, temp 0.8,
  no repeat penalty (`common/common.h:235-245`). Both are a few hundred µs per token at a 100k vocab
  (*estimate*), < 1% of a ~70-100 ms phone token.

## 8. Scheduling, speculative decoding, Flutter/FFI layer
- One `llama_decode` per token from a Dart `async*` loop in a worker isolate; tokens reach the UI isolate
  in batches of 8 tokens / 512 bytes (`LD/.../native_token_stream_batcher.dart:27-45`). Per-token Dart work
  is one FFI decode call, one sampler call and one `token_to_piece` (*estimate*: tens of µs).
- No speculative decoding or MTP (Granite has no MTP head; speculation lost on host, see `02`).
- Vollie-only latency features (not decode speed): turn state save/restore, system-prompt priming,
  prefill while typing, pre-reply (`lib/assistant.dart:976-1076, 1223-1467`).

## 9. Host comparison: Vollie's llama.cpp vs stock v0.8.0 (measured on host, relative only)

Same file (A, 4 experts), `tools/host_bench.sh` = llama-bench with Vollie's config (FA on, q8_0 KV,
mmap), pp64/tg64, 5 reps, armv8.2-a+dotprod+fp16 builds of each commit (`lab/logs/p/bench.csv`).

| threads | repack | v0.8.0 pp / tg (tok/s) | Vollie 620d2225 pp / tg | Δ tg |
|---|---|---|---|---|
| 1 | off | 38.6 / 27.8 | 36.9 / 27.6 | −0.8% |
| 1 | on | 82.8 / 46.1 | 82.3 / 46.2 | +0.1% |
| 4 | off (Vollie on 8 GB, until 2026-11-01) | 165.8 / 80.1 | 164.2 / 78.7 | −1.8% |
| 4 | on (stock default) | 303.4 / 114.0 | 296.1 / 111.0 | −2.6% |

- **Perplexity is identical** on the two commits: same file, 100 chunks of wikitext-2 10.6243 on both,
  en.txt 12.2009 on both; the final D1 file 11.2156 en.txt on both.
- **Threadpool:** llamadart never attaches a ggml threadpool, so ggml creates and joins one per decode
  call. A llama-bench switch added in the lab (`LLAMA_BENCH_DISPOSABLE_TP`) measured no difference on the
  host: 4 threads tg 79.8 (per-call) vs 80.1 (persistent), pp 165.8 vs 165.8.
- **RSS** (host peak footprint, non-reclaimable): repack off 128 MB, repack on 3,931 MB, D1 dense-only
  repack 659 MB.
- **Biggest difference on the phone at the time:** repack off (−30% decode on host at 4 threads,
  −40% at 1 thread). The commit gap (≤ 2.6%) and the threadpool are noise in comparison.

## 10. Likely bottleneck on SD888, and the evidence

*As written before any phone measurement (2026-11-01 morning):* memory bandwidth plus the missing
repack, i.e. ~0.9 GB/token at 10-16 GB/s → 8-12 tok/s.

*Revised with device evidence (same day, user-run Settings speed tests, see `03_final.md`):* the
bottleneck is **per-op fixed cost**, mostly the thread sync after each of ~2,000 compute ops per token:
- 7.4 tok/s = 135 ms/token, while 0.93 GB at the ~15 GB/s this phone sustained with Qwen3.5-4B would
  take 56-72 ms.
- Running small ops on one thread without the sync (GGML_TINY_OP) doubled writing at fixed threads:
  2 threads 3.2 → 7.0 tok/s, 4 threads 3.4 → 5.6 tok/s at matched heat. Output stays bit-identical.
- 3 threads beat 4 in every warm run, which fits a sync-bound workload: one slowed core holds every sync.
- On the M4 host the same 1,870 syncs cost ~0.7 ms/token in total (GGML_CPU_PROF), which is why host
  measurements could not show this.

## Changes to Vollie since this comparison (2026-11-01)

Model file → `vollie-granite-4.0-h-tiny-d1-e5.gguf` (5 experts, PPL +6.4%), repack back on for 8 GB
phones (0.7 GB copy), heat-throttle thread change in place instead of a model reload, the tiny-op CPU
module, and a speed test that sweeps it. Details and versions in `03_final.md`.

