# Served file: `olmoe-Mi.gguf` (frozen for campaign 2)

Chosen by the host screen R50 (rule: memo-01 part C). It can also be rebuilt from the recipe below; the sha256 decides.

| | |
|---|---|
| campaign | 2 (25 tok/s, +5 %) |
| download | [HoeioUser/vollie2 @ e893e8a6](https://huggingface.co/HoeioUser/vollie2/blob/e893e8a65cff2f09f3de45e4840537c625806e0b/olmoe-Mi.gguf) (uploaded 2026-11-13; the server-side sha256 equals the value below); sources: `allenai/OLMoE-1B-7B-0125-Instruct-GGUF` @ `2ac1d27317927518c9ef7bd99f91f3f1e4ee288d` (F16), `mradermacher/OLMoE-1B-7B-0125-Instruct-i1-GGUF` @ `8eec686da67fd0d16b6f8f37cce694f45fb9f906` (`imatrix.dat`) |
| size | 3,072,399,552 bytes |
| sha256 | `5abf710b775cd64fd89d74a783c8a7056a409ba795cd401b9dd448ef8e444abb` |
| model | AI2 OLMoE-1B-7B-0125-Instruct, stock weights, Apache-2.0: 6.92 B total, 1.18 B active per token (header count), 16 layers, 64 experts, 8 used per token, expert FFN 1024, no shared expert, vocabulary 50,304, untied head |
| source weights | publisher F16 (sha256 `b9258178…bc8da20e`), mradermacher's imatrix (calibration text not documented) |
| types | dense Q4_0 (including `token_embd`), experts Q3_K, `output.weight` Q6_K; 8 experts per token (the model's own count) |
| llama.cpp support | `BASE` `620d2225c`, architecture `olmoe`, unpatched build |
| wikitext-2 test PPL, F16 (ctx 512, all 560 chunks) | 10.9420 ± 0.0896 (*host*) |
| wikitext-2 test PPL, served file | 11.7146 ± 0.0960 (*host*), **+7.06 %** over F16 |
| **budget** (served PPL × 1.05) | **12.3003** |
| **hard ceiling** (F16 × 1.20) | 13.1304 |
| en.txt PPL (sanity only) | not measured yet |
| weight bytes per token (`tools/cost.py`, header inventory) | 590 MB = dense 152 + experts 346 + router 8 + head 85; KV 128 KiB per position; no recurrent state |

Recipe (unpatched `llama-quantize` at `BASE`):

```sh
$LLAMA/llama-quantize --imatrix imatrix.dat \
  --tensor-type ffn_gate_exps=q3_k --tensor-type ffn_up_exps=q3_k --tensor-type ffn_down_exps=q3_k \
  --output-tensor-type q6_k OLMoE-1B-7B-0125-Instruct-F16.gguf olmoe-Mi.gguf Q4_0
```

Why these types (R50, *host*): the publisher's Q4_0 (3,928,038,048 bytes) is 11.3784 (+3.99 %); the same mix
without an imatrix is 11.9321 (+9.05 %) and fails the 1.08 rule; with the imatrix it passes at +7.06 %. The small
experts lose more at 3 bits than Granite's (Granite: +1.3 % from Q4_0 to this mix, OLMoE: +3.0 %). The mix is
0.86 GB smaller than the Q4_0 file and reads 107 MB less per token.

Known limits for the speed work ([memo-02](../campaigns/02-25toks-5pct/reports/memo-02-lever-applicability.md)):
the two-stage head does nothing on this model as the patch stands (untied, repacked head), and expert skip has
a different meaning (top-8 softmax weights do not sum to 1), so no Granite threshold carries over.
