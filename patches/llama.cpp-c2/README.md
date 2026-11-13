# Campaign 2 patch series for llama.cpp

Base: `620d2225cc436ab0b001e3200e590e3b286ef9ab` (the same as `../llama.cpp/BASE`). Each patch applies to a clean
tree of the base, holds one lever and is independent of campaign 1's `../llama.cpp/0001-cpu-runtime.patch` (not
checked for co-application with it). **Every lever is off unless its environment variable is set; with all knobs
unset the build gives the stock graph and stock results.**

```sh
git -C <llama.cpp at 620d2225c> apply patches/llama.cpp-c2/0001-olmoe-expert-skip.patch
```

## 0001-olmoe-expert-skip.patch: expert skip by router weight (numerics change when on)

Touches `src/llama-graph.cpp` (`build_moe_ffn`) and `ggml/src/ggml-cpu/ggml-cpu.c` (`MUL_MAT_ID`). With a skip knob
set, libllama gives each routed-expert `MUL_MAT_ID` the mixture weights as `src[3]`. The CPU module then drops the
low-weight (slot, token) pairs from the row map, so an expert left with no rows is not read at all (in decode
that is every skipped expert), and zeroes their output rows. The rules are scale-free, so they mean the same with or
without top-k renormalisation; written for OLMoE (`norm_w = false`), checked only on OLMoE. Design and checks:
`campaigns/02-25toks-5pct/reports/memo-04-olmoe-expert-skip.md`.

| knob | value | meaning (per token; v = weight of a selected expert, S = sum of the k selected weights) |
|---|---|---|
| `VOLLIE_SKIP_SUM` | permille, > 0 = on | skip if v < value/1000 × S. On renormalised weights (S = 1) this is campaign 1's `GGML_EXPERT_SKIP` |
| `VOLLIE_SKIP_MASS` | permille, > 0 = on | rank by v; skip each expert once the higher-ranked ones already hold value/1000 of S |
| `VOLLIE_SKIP_MAX` | permille, > 0 = on | skip if v < value/1000 × (largest selected v) |
| `VOLLIE_SKIP_MINK` | integer, default 1 | the top MINK experts are never skipped (at least 1) |
| `VOLLIE_SKIP_STATS` | 1 = on | at process exit, one `c2-skip:` line on stderr: tokens, pairs, % pairs skipped, % expert bytes skipped, MB per token selected and skipped, % of the bytes the batch would have read that it did not read |

If several rules are set, an expert is skipped when any of them says so. The kept weights are not renormalised.
Values are read once per process. Only the generic CPU `MUL_MAT_ID` path skips. Experts in a repacked
(extra-buffer) type or on another backend ignore `src[3]` and run in full. OLMoE's Q3_K experts are not repacked.
