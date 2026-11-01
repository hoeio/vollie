# Vollie — serving a MoE LLM fast on a phone CPU

Research and runtime code for running an LLM well on an 8 GB Android phone with llama.cpp. It
comes from [Vollie](https://huggingface.co/HoeioUser/vollie-qwen3.5-4b), an on-device assistant;
this repository holds only the serving side: the CPU-module patches, the app-side tuning code, the
evaluation tools, and the full research log. The app itself is not part of it.

Test device: Samsung Galaxy Z Flip 3 (Snapdragon 888: 1x Cortex-X1 + 3x A78 + 4x A55, 8 GB LPDDR5).
Model: Granite-4.0-H-Tiny abliterated (7B mixture-of-experts, 64 experts, Mamba-2 hybrid), one fixed
GGUF ([recipe](models/granite-4.0-h-tiny-d6-e6.md)). The constraint throughout: perplexity at most
10% above the same model's F16 weights.

## Results so far

Every number is labelled: *device* = measured on the phone by its owner through the app's speed
test; *host* = Apple M4 Pro, CPU only, relative only.

| | writing (decode) | reading (prefill) | how |
|---|---|---|---|
| stock llama.cpp settings, warm phone | 3.4 tok/s | 16 tok/s | device |
| + sync skipping around small ops | 7.5 | 18 | device |
| + expert prefetch (page-cache aware) | 10.7 | 19 | device |
| + next-layer expert guessing | **12.2** | **25** | device |
| quality of the served file | wikitext-2 PPL +6.2% vs F16, en +7.0% | | host |

What limited the phone, in the order it was found (details in [docs/findings.md](docs/findings.md)):

1. **Thread syncs, not bandwidth.** ~2,000 compute ops per token, each followed by a sync of all
   threads; on the SD888 that was about half of every token. Small ops now run on one thread
   without syncing: 3.2 → 7.0 tok/s at fixed threads, output bit-identical.
2. **The experts don't fit in the page cache.** 3 GB of MoE experts are read through mmap; with
   ~2 GB of cache on a busy 8 GB phone, every token re-read hundreds of pages from flash, one
   synchronous fault at a time. Prefetching exactly the routed experts per layer, and guessing the
   next layer's experts a layer early (85% right on the device), took writing to 10.7-12.2 tok/s.
3. **Now: expert compute and measurement noise.** The 3-bit expert kernel is compute-bound on the
   A78, and single runs on a warm phone vary ±20%. A faster Q3_K kernel and an in-generation A/B
   (the setting alternates every token) are the current work.

The goal is 20 tok/s. On this phone the dense weights alone take ~25 ms per token (near the
SoC's bandwidth), so 50 ms/token is the realistic ceiling; it is not reached yet.

## Layout

| path | what |
|---|---|
| [patches/llama.cpp/](patches/llama.cpp/) | the CPU-module patch (against a pinned llama.cpp commit): small-op sync skipping, expert I/O, next-layer expert guessing, Q3_K kernel, per-op profiler, in-generation A/B |
| [patches/llamadart/](patches/llamadart/) | small patch to the Dart bindings: change threads without reloading, extra-buffer switch |
| [android/](android/) | build the patched CPU module for Android and drop it into an app |
| [runtime/dart/](runtime/dart/) | app-side helpers: set the knobs, pin threads to big cores, read page faults / memory / clocks, pick the best setting |
| [docs/](docs/) | findings, runtime knobs, the on-device tuning procedure |
| [reports/](reports/) | the research log: model choice, Vollie vs llama.cpp, methods survey, final report, `results.csv` |
| [tools/](tools/) | perplexity gate, host benchmarks, quantization recipes, on-device benchmark kit |
| [models/](models/) | how the served GGUF was made, with checksums (weights are on Hugging Face) |

## Quick start

```sh
git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp
git checkout $(cat ../vollie/patches/llama.cpp/BASE)
git apply ../vollie/patches/llama.cpp/0001-cpu-runtime.patch
../vollie/android/build_cpu.sh . out/        # libggml-cpu-android_armv8.2_2.so
```

Every knob is off unless set, so the module behaves like stock until the app turns things on
([docs/runtime-knobs.md](docs/runtime-knobs.md)), for example:

```sh
GGML_TINY_OP=65536 GGML_EXPERT_IO=5 llama-cli -m model.gguf -t 3 ...
```

Each change in the patch was checked bit-identical against stock kernels with
`llama-perplexity` (one-token, 32-token and 512-token batches) before it shipped.

## How the research is recorded

[reports/03_final.md](reports/03_final.md) is the living log: one row per experiment (R1, R2, …)
with what changed, what was measured where, and the verdict, including what failed and a race
bug found on the way. [reports/results.csv](reports/results.csv) has one line per measurement.
New results are appended there, never rewritten.

## License

MIT (see [LICENSE](LICENSE)), like llama.cpp and llamadart, which the patches apply to.
