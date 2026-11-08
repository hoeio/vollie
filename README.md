# Vollie — serving a MoE LLM fast on a phone CPU

**What / why / result:** Runtime patches for llama.cpp that make a 7B mixture-of-experts model (Granite-4.0-H-Tiny) write at **20.51 tok/s median on the official 32-token bench** (R49, round 16, research build 2.2.21+44; 10 runs, 8 of them 20.2-20.95; device, resident model, fast cores at 2,112 MHz) on a 2021 Android phone (Snapdragon 888, 8 GB). A cold start under memory pressure still prints 12.3-18.5 tok/s (the cold baseline and the two runs that followed heavy memory use): the remaining loss is page residency at the start of a run, not arithmetic. For scale: stock llama.cpp printed 8.5 the same day as R31; the bit-identical runtime alone gave 12.0 vs 9.4 stock (×1.28, R25). Same model file throughout. Output is bit-identical to the served file except one expert-skip rule that costs +1.91 % perplexity (budget +2 %); the served config is skip 117 + two-stage head q4_0/2048 + the sampler fast path (PPL 9.7886). What got it there: R31 (skip 117 + two-stage head) 16.4; a bit-identical in-register Q3_K expert kernel (default since R35; its device line falsified under the pooled rule in R38, then passed on new data in R47, measured on throttled clocks); a bit-identical fused expert op (passed R47, −0.80 ± 0.10 ms/token, default since R47). Every prediction was written down before its phone run, and every number is measured on the phone and logged round by round, wins and losses alike (table below).

**Reproduce (no phone needed):** `git clone https://github.com/hoeio/vollie && cd vollie && tools/reproduce.sh` builds llama.cpp at the pinned base with the patches, downloads the model (3.16 GB, sha256-checked) and wikitext-2, and checks the two host claims: bit-identity of the runtime knobs (per-chunk PPL to the last digit) and the perplexity gate (stock vs `GGML_EXPERT_SKIP=118`, PASS/FAIL against the +2 % budget; ~1-1.5 h on a laptop, `--quick` screens it in ~15 min). The Android CPU module needs the NDK: `android/build_cpu.sh` (see `android/README.md`). Every phone number comes from the device JSONs, which need the phone. Verified 2026-11-05 from a clean clone on macOS arm64 (Apple M4, 8 threads): identity equal, stock 9.6051, skip 118 9.7952; not yet run on Linux (log `lab/logs/r27/reproduce_macos-arm64.log`, kept in the maintainers' workspace; summary row in `reports/results.csv`).

**Here / not here:** Here: CPU-module patches, build script, on-device research kit and parser, perplexity gate, the full log with raw device JSON (`reports/device/`). Not here: the Flutter app, CI, releases. The evidence is the device JSONs and the pre-registered predictions scored against them.

**Store builds lag.** The store version of the app can be behind the builds measured here, because store review takes time (on 2026-11-04 the store version is 2.1.10, per the owner; its numbers are the last column of the table below). The other numbers come from builds the owner measured: the app's own speed test before 2.2.6+29, research-kit APKs from 2.2.6+29 on (latest: 2.2.21+44, kit r16).

Research and runtime code for running an LLM well on an 8 GB Android phone with llama.cpp. It
comes from Vollie, an on-device assistant. Its model files are on Hugging Face in
[HoeioUser/vollie-qwen3.5-4b](https://huggingface.co/HoeioUser/vollie-qwen3.5-4b); that repository name
predates the switch to Granite (the app ran Qwen3.5-4B in October): the repository holds one Qwen3.5-4B file and
three Granite-4.0-H-Tiny files, and the Granite file used here is identified by its sha256
([models/](models/granite-4.0-h-tiny-d6-e6.md)).
This repository holds only the serving side: the CPU-module patches, the app-side tuning code, the
evaluation tools, and the full research log. The app itself is not part of it.

Test device: Samsung Galaxy Z Flip 3 (Snapdragon 888: 1x Cortex-X1 + 3x A78 + 4x A55, 8 GB LPDDR5).
Model: Granite-4.0-H-Tiny abliterated (7B mixture-of-experts, 64 experts, Mamba-2 hybrid), one fixed
GGUF ([recipe](models/granite-4.0-h-tiny-d6-e6.md)). Goal: 20 tok/s printed. The one hard limit:
wikitext-2 perplexity at most 20 % above the same model's F16 weights; the working budget for the
speed rule is +2 % over the served file. All conditions are in [RULES.md](RULES.md).

## Vanilla llama.cpp vs Vollie, same phone

| | vanilla llama.cpp | Vollie, bit-identical runtime | Vollie + expert skip 118 | Store build 2.1.10 (earlier test, D1 file, not same-day; see *Note on the store build*) | label | where logged |
|---|---|---|---|---|---|---|
| Private memory (anon) after the run | 679-690 MB; not a difference between columns: mmap weights with repack off are shared by all three (see *Footnote*). The stock-repack comparison is a different test, see *Note on stock repack* | 692-707 MB (the two baseline runs; 645-707 MB over all 13 runs of the day) | 660-689 MB | not measured on the phone (anon not logged); estimated need 3.8 GB resident = 3.1 GB mmapped experts + 0.66 GB repack copy | device | `reports/device/r5_2026-11-04.json` (`mem_after`, app 2.2.10+33) |
| Writing speed, official 32-token bench (decode) | 9.4 tok/s | 12.0 tok/s (better of two same-session baseline runs; the first read 8.2 with 97 MB of mapped file pages resident at start, the second 12.0 with 251 MB) | 13.9 tok/s | 6.1 tok/s best run (2 threads, 32 written tokens, heat 76-83 %); the same session's knob sweep read 4.3-4.4 at 2 threads | device | `03_final.md` "R25 — round-5 device results"; `results.csv` row "R25 VANILLA VS VOLLIE"; `reports/device/r5_2026-11-04.json` |
| Reading speed (prefill, same runs) | 15.1 tok/s | 16.2-18.0 tok/s | 21.6 tok/s | not logged for this build | device | same as above |
| 200-token story, printed | 10.2 tok/s | 14.5 tok/s | 12.5 tok/s (see note) | no such test in this build | device | same as above |
| 200-token story, steady profile (ms per token, last windows) | 82-97 | 62-66 | 55.4 (≈ 18 tok/s steady) | 137-163 (2 threads; average over a whole profiled reply, not a steady window; 354-1,181 major faults per token) | device | same as above |
| Output identical to stock | (reference) | yes for every default knob (bit-identical at `-ub 1/32/512`) | no (expert skip) | yes (tiny-op v2 only, host-checked, plus stock dense repack) | host | `RULES.md` §4; `results.csv` identity rows (tiny-op v2, Q3_K x4, R23/R24 probes) |
| Quality, wikitext-2 PPL | F16 9.0438; served file 9.6051 (+6.21 %) | 9.6051 (+6.21 %) | 9.7952 (+1.98 % over stock; budget +2 %, limit 10.853) | D1 file: 9.6230 (+6.40 % over F16, +0.19 % over the served file) | host | `results.csv` rows "D6" and "R25 SKIP THRESHOLD EDGE"; `RULES.md` §1 |

*Note on the store build.* The last column is app **2.1.10+22** (R9), the version the owner states is in the store
on 2026-11-04. It was measured with the app's own speed test and profile, before the research kit existed: there is
no raw JSON, the numbers are as transcribed in `results.csv` (row "app 2.1.10+22") and `03_final.md` R9. The owner
recalls it was before about 18:00 on 2026-11-01; the repo logs the build number, not the date. It ran the **D1-e5
file** (`vollie-granite-4.0-h-tiny-d1-e5.gguf`, Q4_0 dense + IQ4_XS experts, 5 experts, 3.77 GB), not the served D6
file: dense weights repacked, experts read through mmap, tiny-op v2 only, no expert prefetch, 2 threads. File,
knobs, free memory, heat and test protocol all differ from the same-day columns, so no speedup should be computed
between this column and the others; it records where the store version stands.

*Note on stock repack (not same-day, not the served file).* The 3,931 MB figure that stood in the memory row
earlier is a **host** number, not a phone number: Apple M4 Pro, `/usr/bin/time -l` peak memory footprint,
llama.cpp v0.8.0@8642541b, the first Q4_0 file (4 experts, PPL 10.1619, +12.4 %: it failed the 10 % gate in force
then; the limit has since been raised to 20 %, see RULES.md §1) with repack on, against
128 MB with repack off. On the phone that configuration was killed by Android (`CPU_REPACK` 3105 + KleidiAI
541 MiB), which is the only device observation behind it. It was not repeated for the served file (its Q3_K
experts have no repack path) or on 2026-11-04; the date of the original host run is not recorded in this repo.
Logged in `results.csv` (row "A file with repack ON"), `reports/00_model_choice.md`, `reports/02_methods.md`. The
older 674 MB anon figure (app 2.2.2+25, R12) is superseded by the same-day values in the table.

*Note on the 12.5.* That run was in a worse page-cache state (77 vs 34 major faults per token). The in-run A/B
measured the skip rule at −7.6 ± 6.0 ms/token (−8.3 ± 2.9 at threshold 115), i.e. ≈ ×1.12-1.15 over the
bit-identical runtime.

*Round 6, same-day row (device).* 2026-11-05, fast cores capped at 1,555 MHz (r5: 2,112): official 32 vanilla 6.1 → Vollie 9.9 (bit-identical) → 14.0 (skip 118); story 200 printed 8.9 → 12.05 → 12.7; steady 100.9 → 74.6 → 63.7 ms/token; ratio 1.35.

*Footnote.* All speed numbers in the table above, except the store-build column and the round-6 row, are from one day (2026-11-04, a swapping day: swap free 0-386 MB,
6-110 major faults per token), same app (2.2.10+33, kit r5), same file, 3 threads, measured by the owner
through the app. Each cell is a single run, except the bit-identical official-32 decode figure: it is the better of
two baseline runs (8.2 and 12.0), since RULES.md §5 scores a setting by its best run; the vanilla figure is one run.
Printed tok/s over 200 tokens varies with page-cache state more than with the runtime (the
same build printed 14.5 and 12.5 in consecutive runs). "Vanilla" = every Vollie CPU-module knob at its stock
value; what env knobs cannot switch, and so is shared by all columns: q8_0 KV cache + flash attention, mmap
weights with repack off, big-core pinning + 3 threads, the Granite rollback patch, context/template/greedy
decoding. The repack row is the first Q4_0 file (the served file's Q3_K experts have no repack path, so stock
llama.cpp may well load it). *Device* = measured on the phone by its owner through the app; *host* = Apple M4 Pro,
CPU only, relative or machine-independent only.

## What Vollie's runtime does differently

Sync-skipping around small ops: tiny ops run on one thread without a sync of all threads (R3, R7;
barrier latency measured at 0.3 µs in R18 — the waits are imbalance, not sync cost). Page-cache-aware
expert prefetch of the routed experts that are not skipped (skip-aware since R26), plus next-layer expert guessing a layer early (R12-R13;
85 % right on the device). Faster NEON Q3_K dot, dynamic 32-row expert chunks and a 4-row Q3_K kernel
(R15, R17-R20). SSM state and conv state updated in place instead of copied (R24-R27, bit-identical). Router-weight
expert skip: experts with a small normalised router weight are not computed, under the +2 % PPL budget
(R21-R27; threshold 118 since round 5, 115 before). What was tried and closed (4 compute threads, helper on an A55, n-gram and
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
for p in 0001-cpu-runtime 0002-granite-rollback 0003-sampler-bypass; do git apply ../vollie/patches/llama.cpp/$p.patch; done
../vollie/android/build_cpu.sh . out/        # libggml-cpu-android_armv8.2_2.so
```

Knobs are environment variables ([docs/runtime-knobs.md](docs/runtime-knobs.md)); the app sets them
on the phone, and `GGML_EXPERT_SKIP` stays off unless set. Example:

```sh
GGML_TINY_OP=262144 GGML_EXPERT_IO=5 llama-cli -m model.gguf -t 3 ...
```

## License

MIT (see [LICENSE](LICENSE)), like llama.cpp and llamadart, which the patches apply to.
