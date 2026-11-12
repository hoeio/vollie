# Tools

Scripts that produced every number in `reports/`. They expect the lab workspace in
`docs/workspace.md` (models, text sets and llama.cpp builds under `lab/`, which git ignores). Where a script's
usage line says `lab/tools/...`, that is its path in the private workspace; here it is `tools/...`. Scripts
keep their names and places because the logs in `reports/` cite them by path.

## Reproduce and measure on the host

| script | does |
|---|---|
| `ppl.sh <tag> <model.gguf>` | **the quality gate**: wikitext-2-raw test + `lab/text/en.txt`, ctx 512, `llama-perplexity`; resumable |
| `reproduce.sh [--quick] [workdir]` | host reproduction of the no-phone claims: patched build, model sha256, knob bit-identity, wikitext-2 gate stock vs skip 118; resumable. Defaults are campaign 1's Granite file; `REPRO_URL`, `REPRO_SHA`, `REPRO_STOCK_REF`, `REPRO_SKIP_REF`, `REPRO_BUDGET`, `REPRO_HARD` select another model's file and values |
| `ppl_table.py` | perplexity → bits per byte, to compare models with different tokenizers |
| `quant_variant.sh <tag> <type> [args]` | requantize from the F16 reference with the same imatrix (`EXPERTS=` sets experts per token) |
| `host_bench.sh <tag> <model.gguf>` | host CPU-only `llama-bench` with the app's settings (FA, q8_0 KV), appends to `bench.csv`; relative numbers only |
| `cost.py <model.gguf>` | bytes read per token, active parameters, state size, from the GGUF |
| `sd888_estimate.py` | the bandwidth model used for estimates (superseded by device measurements, kept for the record) |
| `layer_k_sweep.sh` | per-layer expert-count sensitivity on the wikitext-2 *validation* split (needs `patches/llama.cpp/experiments`) |

## Measure on the phone

| script | does |
|---|---|
| `research_kit.md` | the card of the research kit (the "Run research kit" button in the app): what it runs and the result file |
| `research_parse.py <report.json>` | turns a research-kit JSON report (`reports/device/`) into tables, `results.csv` rows and R-log text; `--official` scores round 16, `--gate` the block-gated A/B |
| `build_android_bench.sh`, `bench_device.sh`, `BENCH_DEVICE.md` | cross-built `llama-bench`/`llama-perplexity` and a kit to run them on a phone via adb or Termux, with a CSV result format |

On a phone, Vollie's in-app speed test (`docs/tuning.md`) replaced the adb kit, since no device
was attached to the lab machine.

## Checks (`checks/`)

Host checks (bit-identity and kit-bug regressions). They expect the lab layout (`lab/checks/`).

| script | does |
|---|---|
| `kit_runs.sh`, `kit_runs.cpp` | cpu-prof/A-B window carry-over between kit runs |
| `ab_skip_id.sh` | a no-op skip A/B equals off |
| `q3k_x4_check.sh`, `q3k_x4_check.c` | the Q3_K x4 repack is bit-identical |
| `gate_r12.sh` | `research_parse.py --gate` on the r12 device JSON reproduces the worked example of `reports/memos/round13_plan.md` §B |
| `official_check.py` | runnable check of `research_parse.py --official` on a synthetic kit-r16 report with known answers |

## One-off analysis scripts (host)

Each answered one question in the research log and is kept so the number can be reproduced. They are not part
of the serving code. The last column lists the files in this repository that cite the script by name (found by
searching for its file name; the citation may use a shortened form).

| script | question it answered | cited in |
|---|---|---|
| `head2stage_fetch.py` | pull per-token output-head inputs from a running `llama-server` | `reports/memos/head2stage_fix.md` |
| `head2stage.py` | replay the two-stage output head on true head inputs: low-rank and low-bit screens | `reports/results.csv`, `reports/memos/head2stage_fix.md` |
| `head2stage_check.py` | reconcile the two head-evaluation routes against `llama-perplexity` | not cited by name |
| `head2stage_q8check.py` | does the Q8_K activation quantisation explain the gap between the two routes? | not cited by name |
| `head2stage_requant.py` | copy the served file with the tied head requantised Q6_K to Q4_0 (lab only) | not cited by name |
| `exps_requant.py` | PPL of a runtime Q4_0 copy of the routed experts (a host artefact, never served) | `reports/results.csv`, `reports/memos/last_3ms.md`, patch `0001` |
| `dense_acts.py` | per-token input statistics of every dense matrix product (column-skip candidates) | used by `dense_colskip.py` |
| `dense_colskip.py` | perplexity cost of skipping near-zero input columns of dense products | `reports/results.csv`, `reports/05_compute_less.md`, `reports/memos/dense_bytes_device.md`, `reports/memos/round14_plan.md` |
| `dense_headthr_check.py` | independent numpy check of the output-head case of `dense_colskip.py` | `reports/memos/round14_plan.md` |
| `draft_gate.py` | confidence-gated emission from saved logits of a target and a draft model | `reports/results.csv`, `reports/05_compute_less.md` |
| `expert_overlap.py` | expert overlap across consecutive decoded tokens, from `GGML_EXPERT_LOG` lines | `reports/04_campaign.md`, `reports/results.csv` |
| `ngram_sim.py` | acceptance of prompt-lookup (n-gram) speculation, simulated on a greedy sequence | `reports/04_campaign.md`, `reports/results.csv`, `reports/03_final.md` |

## `q3k_kernel/`

The round-10 audit of the Q3_K expert kernel on the A78: prototypes, a host bench and the NDK disassembly
mix. Its own [README](q3k_kernel/README.md) explains it; `bench.out` is the output of one run.
