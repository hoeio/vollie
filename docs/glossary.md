# Glossary

Terms that appear throughout the logs. Definitions are taken from [RULES.md](../RULES.md), the
[README](../README.md), [runtime-knobs.md](runtime-knobs.md) and [reports/03_final.md](../reports/03_final.md);
where a term is used more precisely in a log entry, the entry wins.

## Labels

| term | meaning |
|---|---|
| *device* | measured on the phone (Galaxy Z Flip 3, Snapdragon 888) by its owner through the app |
| *host* | measured on a laptop (Apple M4); relative or machine-independent numbers only (for example perplexity) |
| *estimated* | arithmetic on other numbers, not a measurement |

## Process

| term | meaning |
|---|---|
| R-log, R-number | the numbered entries (R1 to R49) in [reports/03_final.md](../reports/03_final.md), with pre-registrations and verdicts |
| round, kit `rN` | one session of the app's research kit on the phone; its raw JSON is in [reports/device/](../reports/device/) (`r1` to `r16`) |
| pre-registration | the prediction and the pass rule of a round, written into the log before the run and scored as written |
| claim rule | the pass rule for the headline result: "20 reached iff the median of the 10 plain runs is ≥ 20.0" |
| plain run / profiled run | a plain run is only timed; a profiled run also writes per-token `cpu-prof` lines (`GGML_CPU_PROF`) |
| cool-down gate | before each official run the kit waits (up to 90 s) until Android's thermal headroom forecast is under 0.65 |
| official 32 | the app's benchmark: official prompt, 32 generated tokens, greedy, printed as llamadart's `decodeTokensPerSec` |
| ttft | time to first token, in the per-run tables |

## Quality

| term | meaning |
|---|---|
| PPL gate | wikitext-2 perplexity with `llama-perplexity` (test set, context 512, all chunks, `tools/ppl.sh`); hard limit +20 % over F16 (9.0438, so ≤ 10.853) |
| budget | the +2 % over the served file's stock PPL (≤ 9.797) allowed for the speed rule; final for the campaign |
| bit-identical | per-chunk `llama-perplexity` values equal at `-ub 1`, `-ub 32` and `-b 512`, to the last digit |
| served file | the frozen GGUF `vollie-granite-4.0-h-tiny-d6-e6.gguf` ([models/](../models/granite-4.0-h-tiny-d6-e6.md)) |

## Runtime

| term | meaning |
|---|---|
| knob | an environment variable of the patched CPU module, read before each graph; off unless set, except the ones marked default on ([runtime-knobs.md](runtime-knobs.md)) |
| in-run A/B, ABBA | `GGML_CPU_AB`: alternates a setting inside one generation (after a warm-up, blocks in A B B A order) and reports B − A ± 1σ, so both sides share heat, clock and cache |
| tiny-op | `GGML_TINY_OP`: ops below a size threshold run on one thread with no sync between them |
| `MUL_MAT_ID` / `MUL_MAT` | the ggml ops for the routed-expert matrix products / for the dense matrix products |
| expert skip | `GGML_EXPERT_SKIP`: experts with a small normalised router weight are not computed; the threshold is in permille (117 in the served config; 118 is the edge of the +2 % budget) |
| two-stage head (head2) | the output head is screened with a runtime Q4_0 copy over all rows, and the top 2,048 rows are recomputed exactly |
| sampler bypass | `VOLLIE_SAMPLER_BYPASS=1`: the repetition penalty visits only the penalised tokens; greedy text identical |
| major fault (majflt) | a page fault that reads the page from storage; on this phone, the signature of experts being re-read from flash |
| page residency | whether the model's pages are already in memory when a run starts; low residency is the cause of the slow cold start |
| cold start | the first run after memory pressure or idle time evicted the model's pages (12.3-18.5 tok/s in R49) |
| SD888 | Qualcomm Snapdragon 888 (1x Cortex-X1 + 3x A78 + 4x A55) |
