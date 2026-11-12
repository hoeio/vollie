# Served file: `<file name>.gguf` (frozen)

Copy this card for every model of a campaign. One card, one file, one hash.

| | |
|---|---|
| campaign | |
| download | <Hugging Face repo @ pinned commit> |
| size | bytes |
| sha256 | |
| model | maker, name, total / active parameters, experts, layers |
| source weights | where the F16 comes from, imatrix if any |
| types | dense tensors, experts, output head, experts used per token |
| llama.cpp support | `BASE` commit it was verified on, architecture name |
| wikitext-2 test PPL, F16 (ctx 512, all chunks) | |
| wikitext-2 test PPL, served file | and % over F16 |
| **budget** (served PPL × the campaign's factor) | absolute value |
| **hard ceiling** (F16 × 1.20) | absolute value |
| en.txt PPL (sanity only) | F16 and served |

Recipe: the exact command (`tools/quant_variant.sh` or other), so the file can be rebuilt.

Why these types: the measurements behind each choice, with R-numbers once they exist.
