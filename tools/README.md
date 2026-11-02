# Tools

Scripts that produced every number in `reports/`. They expect the lab workspace in
`docs/workspace.md` (models, text sets and llama.cpp builds under `lab/`, which git ignores).

| script | does |
|---|---|
| `ppl.sh <tag> <model.gguf>` | **the quality gate**: wikitext-2-raw test + `lab/text/en.txt`, ctx 512, `llama-perplexity`; resumable |
| `ppl_table.py` | perplexity → bits per byte, to compare models with different tokenizers |
| `quant_variant.sh <tag> <type> [args]` | requantize from the F16 reference with the same imatrix (`EXPERTS=` sets experts per token) |
| `host_bench.sh <tag> <model.gguf>` | host CPU-only `llama-bench` with the app's settings (FA, q8_0 KV), appends to `bench.csv`; relative numbers only |
| `cost.py <model.gguf>` | bytes read per token, active parameters, state size, from the GGUF |
| `sd888_estimate.py` | the bandwidth model used for estimates (superseded by device measurements, kept for the record) |
| `layer_k_sweep.sh` | per-layer expert-count sensitivity on the wikitext-2 *validation* split (needs `patches/llama.cpp/experiments`) |
| `build_android_bench.sh`, `bench_device.sh`, `BENCH_DEVICE.md` | cross-built `llama-bench`/`llama-perplexity` and a kit to run them on a phone via adb or Termux, with a CSV result format |

On a phone, Vollie's in-app speed test (`docs/tuning.md`) replaced the adb kit, since no device
was attached to the lab machine.

Host checks (bit-identity and kit-bug regressions) are in `tools/checks/`: `kit_runs.sh`/`.cpp` (cpu-prof/A-B window carry-over between kit runs), `ab_skip_id.sh` (no-op skip A/B equals off), `q3k_x4_check.sh`/`.c` (Q3_K x4 repack bit-identical). They expect the lab layout (`lab/checks/`).
