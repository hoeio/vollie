# Lab workspace layout

The tools and the paths inside `reports/` refer to this layout (none of it is in git):

| path | contents |
|---|---|
| `lab/models/ref/` | `Huihui-granite-4.0-h-tiny-abliterated.f16.gguf` (13.9 GB, mradermacher) and its `imatrix.gguf` |
| `lab/models/var/` | requantized variants (`tools/quant_variant.sh`), e.g. `d6-q3k-e6.gguf` = the served file |
| `lab/text/wikitext-2-raw/` | `wiki.test.raw` (the gate), `wiki.valid.raw` (tuning only) |
| `lab/text/en.txt` | 57.7 KB public-domain English prose (sanity set) |
| `lab/llama.cpp-v080/` | upstream v0.8.0 @ 8642541b, `build-888` = CPU-only armv8.2-a+dotprod+fp16, `build-native` for F16 references |
| `lab/llama.cpp-tiny/` | llama.cpp 620d2225c (the app's commit) + `patches/llama.cpp` |
| `lab/logs/p/` | every perplexity log (`ppl-<tag>-<set>.log`), quantize log and `bench.csv` |

In the reports, `vollie/lib/...`, `vollie/tool/...` and `vollie/third_party/llamadart` are files of
the Vollie app, which is not in this repository; their serving parts are in `runtime/dart`,
`android/` and `patches/llamadart`. `lab/tiny-op.patch` is `patches/llama.cpp/0001-cpu-runtime.patch`, `lab/granite-rollback.patch` is `0002-granite-rollback.patch`, `lab/sampler-bypass.patch` is `0003-sampler-bypass.patch`.
