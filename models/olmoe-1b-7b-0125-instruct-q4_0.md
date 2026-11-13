# Served file: `OLMoE-1B-7B-0125-Instruct-Q4_0.gguf` (frozen for campaign 2, second OLMoE file)

The publisher's own Q4_0, unchanged. Added by the owner on 2026-11-13, **after** the host screen R50, as a second
served file beside the rule's choice ([olmoe-Mi](olmoe-1b-7b-0125-instruct-mi.md)); see CAMPAIGN.md, owner
decision 9. It has its own budget.

| | |
|---|---|
| campaign | 2 (25 tok/s, +5 %) |
| download | `allenai/OLMoE-1B-7B-0125-Instruct-GGUF` @ `2ac1d27317927518c9ef7bd99f91f3f1e4ee288d` |
| size | 3,928,038,048 bytes |
| sha256 | `df43bd738dfdcbe889c5f854120c271e9e53e2c624f212ac2523f8bc11ae9bf3` |
| model | AI2 OLMoE-1B-7B-0125-Instruct, stock weights, Apache-2.0 (structure as in the olmoe-Mi card) |
| source weights | the publisher's quantization; whether an imatrix was used is not documented |
| types | all weights Q4_0 (experts included), `output.weight` Q6_K; 8 experts per token |
| llama.cpp support | `BASE` `620d2225c`, architecture `olmoe`, unpatched build |
| wikitext-2 test PPL, F16 (ctx 512, all 560 chunks) | 10.9420 ± 0.0896 (*host*, R50) |
| wikitext-2 test PPL, served file | 11.3784 ± 0.0931 (*host*, R50), **+3.99 %** over F16 |
| **budget** (served PPL × 1.05) | **11.9473** |
| **hard ceiling** (F16 × 1.20) | 13.1304 |
| en.txt PPL (sanity only) | not measured yet |
| weight bytes per token (`tools/cost.py`) | 697 MB = dense 152 + experts 453 + router 8 + head 85 |

Why it is here: memo-03 estimates (*estimated*, nothing measured) that the Q3_K expert kernel is compute-bound
at about twice the dense byte rate, so Q4_0 experts might cost fewer milliseconds although they are 107 MB more
per token. Unknowns: the phone rate of a Q4_0 expert matmul that is not repacked has never been measured, and
the file is 0.86 GB larger than olmoe-Mi, which matters for the 2.1 GB page cache. Its absolute budget (11.9473)
is tighter than olmoe-Mi's (12.3003).
