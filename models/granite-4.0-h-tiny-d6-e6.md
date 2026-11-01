# Served file: `vollie-granite-4.0-h-tiny-d6-e6.gguf`

| | |
|---|---|
| download | [HoeioUser/vollie-qwen3.5-4b @ 8ab9bf2d](https://huggingface.co/HoeioUser/vollie-qwen3.5-4b/blob/8ab9bf2d710eb7f1bcb80ef2817b06643ce2e6c1/vollie-granite-4.0-h-tiny-d6-e6.gguf) |
| size | 3,161,269,888 bytes |
| sha256 | `5dc1daa0cd3e5099bdaa79beeb9e744bf7c65e0993ea62e4671ddd63b2b5cb25` |
| model | huihui-ai `Huihui-granite-4.0-h-tiny-abliterated` (IBM Granite 4.0 H Tiny, Apache-2.0: 6.9B MoE, 64 experts, 36 Mamba-2 + 4 attention layers) |
| source weights | `mradermacher/Huihui-granite-4.0-h-tiny-abliterated-GGUF` F16, with mradermacher's imatrix |
| types | Q4_0 dense tensors, **Q3_K experts**, Q6_K output head; **6 experts per token** (the model's own count) |
| wikitext-2 test PPL (ctx 512) | 9.6051 vs F16 9.0438: **+6.21%** |
| en.txt PPL | 11.1001 vs F16 10.3768: **+6.97%** |

Recipe (`tools/quant_variant.sh`, llama.cpp v0.8.0):

```sh
EXPERTS=6 tools/quant_variant.sh d6-q3k-e6 Q4_0 \
  --tensor-type ffn_up_exps=q3_k --tensor-type ffn_gate_exps=q3_k --tensor-type ffn_down_exps=q3_k
```

Why these types (all PPL measured on host, details in `reports/03_final.md` §4):
- Q4_0 dense: the only part that gets llama.cpp's repacked copy on ARM (~0.7 GB), which fits an
  8 GB phone; repacking all-Q4_0 weights (3.6 GB) got the app killed.
- Experts in a type without a repack path, so they stay mmapped. IQ4_XS experts at 5 per token
  (+6.4%) were the previous file; 3-bit experts at 6 per token are better (+6.2%) and 0.6 GB
  smaller, which matters because the experts are re-read from flash when they don't fit in the
  page cache. IQ3_XXS experts failed the gate on en.txt (+10.05%).
- Q6_K head: a Q4_0 or Q4_K head failed the gate.
