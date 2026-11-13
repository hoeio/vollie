# Served file: `granite-Mi.gguf` (frozen for campaign 2)

Chosen by the host screen R50 (rule: memo-01 part C). Stock weights; campaign 1's file
([granite-4.0-h-tiny-d6-e6.md](granite-4.0-h-tiny-d6-e6.md)) was built from an abliterated variant and is a
different file. Not yet uploaded anywhere: it exists on the owner's laptop and can be rebuilt from the recipe.

| | |
|---|---|
| campaign | 2 (25 tok/s, +5 %) |
| download | not uploaded yet (owner); sources: `ibm-granite/granite-4.0-h-tiny-GGUF` @ `08d5a8a9741dd5c1a95d2d39e25253226aa1464e` (bf16), `mradermacher/granite-4.0-h-tiny-i1-GGUF` @ `4a7953840f6dfb0d0f6ff3a2fe6528c414328e3f` (`granite-4.0-h-tiny.imatrix.gguf`) |
| size | 3,161,269,344 bytes |
| sha256 | `8df94de731a08a377150e4fcc5dcbd16212a6d102bc0b628cdc905cba0bc0f45` |
| model | IBM Granite 4.0 H Tiny, stock weights, Apache-2.0: 6.94 B total, 1.47 B active per token (header count), 36 Mamba-2 + 4 attention layers, 64 experts, 6 used per token, shared expert, vocabulary 100,352, tied head |
| source weights | publisher bf16 (sha256 `f71ef0b7…4b4535ff`; the publisher has no F16 file), mradermacher's imatrix (calibration text not documented) |
| types | dense Q4_0 (shared expert included), experts Q3_K, tied head `token_embd` Q6_K; 6 experts per token (the model's own count) |
| llama.cpp support | `BASE` `620d2225c`, architecture `granitehybrid`, unpatched build |
| wikitext-2 test PPL, BF16 reference (ctx 512, all 565 chunks) | 8.7816 ± 0.0629 (*host*) |
| wikitext-2 test PPL, served file | 9.2266 ± 0.0664 (*host*), **+5.07 %** over BF16 |
| **budget** (served PPL × 1.05) | **9.6879** |
| **hard ceiling** (BF16 × 1.20) | 10.5379 |
| en.txt PPL (sanity only) | not measured yet |
| weight bytes per token (`tools/cost.py`, header inventory) | 806 MB = dense 420 + experts 243 + router 16 + head 126; recurrent state 58 MB read and written per token. Identical to campaign 1's d6-e6 file |

Recipe (unpatched `llama-quantize` at `BASE`):

```sh
$LLAMA/llama-quantize --imatrix granite-4.0-h-tiny.imatrix.gguf \
  --tensor-type ffn_gate_exps=q3_k --tensor-type ffn_up_exps=q3_k --tensor-type ffn_down_exps=q3_k \
  --output-tensor-type q6_k granite-4.0-h-tiny-bf16.gguf granite-Mi.gguf Q4_0
```

Why these types (R50, *host*): publisher Q4_0 (3,962,938,208 bytes) 9.1045 (+3.68 %); the mix without an imatrix
9.2758 (+5.63 %); with the imatrix 9.2266 (+5.07 %). Both mixes pass the 1.08 rule; the lower one is chosen. The
tensor types and bytes per token equal campaign 1's served file, so campaign 1's millisecond costing is expected
to carry over (*estimated*); its perplexity table (skip thresholds) does not, because the weights differ.
