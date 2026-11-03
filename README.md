# Vollie — serving a MoE LLM fast on a phone CPU

Research and runtime code for running an LLM well on an 8 GB Android phone with llama.cpp. It
comes from [Vollie](https://huggingface.co/HoeioUser/vollie-qwen3.5-4b), an on-device assistant;
this repository holds only the serving side: the CPU-module patches, the app-side tuning code, the
evaluation tools, and the full research log. The app itself is not part of it.

Test device: Samsung Galaxy Z Flip 3 (Snapdragon 888: 1x Cortex-X1 + 3x A78 + 4x A55, 8 GB LPDDR5).
Model: Granite-4.0-H-Tiny abliterated (7B mixture-of-experts, 64 experts, Mamba-2 hybrid), one fixed
GGUF ([recipe](models/granite-4.0-h-tiny-d6-e6.md)). Goal: 20 tok/s printed. The one hard limit:
wikitext-2 perplexity at most 20 % above the same model's F16 weights; the working budget for the
speed rule is +2 % over the served file. All conditions are in [RULES.md](RULES.md).

## Vanilla llama.cpp vs Vollie, same phone

| | vanilla llama.cpp | Vollie | label | where logged |
|---|---|---|---|---|
| Fits in 8 GB | no: default settings (repack on) were OOM-killed, 3,931 MB footprint | yes: 659 MB footprint (D1 file); served D6 file at 674 MB anon + ~2 GB file cache | device | `results.csv` lines 4, 9, 27; `03_final.md` |
| Writing speed (decode) | 3.4 tok/s (stock settings, warm phone, 32 tokens) | **14.2 tok/s** over a 200-token story with expert skip 115; **10.2 tok/s** with the bit-identical runtime alone; 11.3 tok/s on the official 32-token bench (swapping day, page-cache-bound) | device | vanilla: `results.csv` line 16 (app 2.1.7+19); Vollie: `03_final.md` "R24 — round-4 device results" |
| Reading speed (prefill) | 16 tok/s | 20-25 tok/s (20.1 in round 4; 25 in round 3) | device | `results.csv` lines 16, 29, 84; `03_final.md` R24 |
| Output identical to stock | (reference) | yes for every default knob (bit-identical at `-ub 1/32/512`); not for expert skip | host | `RULES.md` §4; `results.csv` lines 17, 49, 82 |
| Quality, wikitext-2 PPL | F16 9.0438; served file 9.6051 (+6.21 %) | 9.6051 (+6.21 %) bit-identical runtime; with expert skip 115: 9.7771 (+1.79 % over stock, budget +2 %, limit 10.853) | host | `results.csv` lines 21, 71; `RULES.md` §1 |

*Footnote.* The vanilla row is from an earlier app build (2.1.7+19, D1 file, 4 threads) on a
different day; the OOM row is the earlier i1-Q4_0 file with repack on. Round 5 will re-measure vanilla
and Vollie in the same run on the same day (kit r5 has a "vanilla" row), and this table will be
updated then. *Device* = measured on the phone by its owner through the app; *host* = Apple M4 Pro,
CPU only, relative or machine-independent only. Decode numbers vary by day with page-cache state
(round 4: swapping day, 18-1,390 major faults per token).

## What Vollie's runtime does differently

Sync-skipping around small ops: tiny ops run on one thread without a sync of all threads (R5-R6,
R20). Page-cache-aware expert prefetch of exactly the routed experts, plus next-layer expert guessing
a layer early (R10-R13; 85 % right on the device). Dynamic 32-row expert chunks and a 4-row Q3_K
kernel (R17-R18). SSM state updated in place instead of copied (R24, bit-identical). Router-weight
expert skip: experts with a small normalised router weight are not computed, under the +2 % PPL budget
(R21-R24; threshold 115). What was tried and closed (4 compute threads, helper on an A55, n-gram and
draft-model speculation, early exit, hot-expert cache) is in the log with numbers. Findings in the
order they were made: [docs/findings.md](docs/findings.md).

## How it is measured

On the real phone, by its owner, through the app's own speed test and a research kit that returns one
JSON per round. Every number is labelled *device*, *host* or *estimated*. Predictions are written down
before each round (`reports/03_final.md`, "pre-registered") and scored against the result, misses
included. Small differences are scored by in-generation A/B (the setting alternates inside one
generation); the parser is `tools/research_parse.py`. Work goes round by round; every result is one
row in `reports/results.csv` and one entry in the R-log, appended, never rewritten.

## Where to look

| path | what |
|---|---|
| [RULES.md](RULES.md) | the conditions every change must meet |
| [reports/03_final.md](reports/03_final.md) | the living log, R1, R2, ... with verdicts |
| [reports/04_campaign.md](reports/04_campaign.md) | the 20 tok/s campaign plan and round results |
| [reports/05_compute_less.md](reports/05_compute_less.md) | the compute-less-per-token memo and predictions |
| [reports/results.csv](reports/results.csv), [reports/device/](reports/device/) | one row per measurement; raw device JSON |
| [patches/llama.cpp/](patches/llama.cpp/) | CPU-module patches against a pinned llama.cpp commit (`BASE`) |
| [patches/llamadart/](patches/llamadart/) | Dart-binding patch: change threads without reload, extra-buffer switch |
| [tools/](tools/) | perplexity gate, host benchmarks, on-device research kit and parser |
| [android/](android/), [runtime/dart/](runtime/dart/) | build the module for Android; app-side helpers |
| [docs/](docs/), [models/](models/) | findings, runtime knobs, tuning; how the served GGUF was made (weights on Hugging Face) |

## Quick start

```sh
git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp
git checkout $(cat ../vollie/patches/llama.cpp/BASE)
git apply ../vollie/patches/llama.cpp/0001-cpu-runtime.patch
../vollie/android/build_cpu.sh . out/        # libggml-cpu-android_armv8.2_2.so
```

Knobs are environment variables ([docs/runtime-knobs.md](docs/runtime-knobs.md)); the app sets them
on the phone, and `GGML_EXPERT_SKIP` stays off unless set. Example:

```sh
GGML_TINY_OP=262144 GGML_EXPERT_IO=5 llama-cli -m model.gguf -t 3 ...
```

## License

MIT (see [LICENSE](LICENSE)), like llama.cpp and llamadart, which the patches apply to.
