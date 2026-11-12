# Memo 01: candidate re-check and host screen plan

Date: 2026-11-12. Agent models: researched by a general-purpose stand-in given `.claude/agents/bonnie.md`, model
`claude-opus-5-5`, because the pinned agents had not loaded yet in that session.

Labels: *device* (phone), *host* (laptop), *estimated* (arithmetic). Nothing in this memo is a measurement.
Every "current state" claim carries a URL, read on 2026-11-12.

Variants (owner and lead, 2026-11-12): Instruct, stock weights. Priority: OLMoE, Granite, LFM2.5.

Base: `620d2225c` = `620d2225cc436ab0b001e3200e590e3b286ef9ab`, committed 2026-10-15 ("llama.cpp : bump version
to 0.7.0 (#30603)"), <https://api.github.com/repos/ggml-org/llama.cpp/commits/620d2225c>. At that commit:
- `src/llama-arch.cpp` contains `"olmoe"`, `"granitehybrid"`, `"lfm2"` and `"lfm2moe"`;
- `src/llama-vocab.cpp` accepts the pre-tokenizers `"olmo"` and `"lfm2"`;
- `src/models/` holds `olmoe.cpp`, `granite-hybrid.cpp`, `lfm2.cpp` and `lfm2moe.cpp`.

Source: <https://raw.githubusercontent.com/ggml-org/llama.cpp/620d2225c/src/llama-arch.cpp> and the sibling paths.

---

## Part A. Candidate re-check 2026-11-12

Sizes come from the HF tree API, `https://huggingface.co/api/models/<repo>/tree/main`. The GGUF facts
(architecture, tensor names and types) were read from the file headers: an HTTP range request fetched the
metadata only, no weights.

### A1. Granite-4.0-H-Tiny

- **Stock weights:** `ibm-granite/granite-4.0-h-tiny`, Apache-2.0. The HF repo was created 2025-09-16
  (<https://huggingface.co/api/models/ibm-granite/granite-4.0-h-tiny>).
- **Public release:** 2025-10-02 according to a third-party page (<https://www.llmreference.com/model/granite-4.0-h-tiny>).
  I did not find the date in IBM's own text
  (<https://www.ibm.com/new/announcements/ibm-granite-4-0-hyper-efficient-high-performance-hybrid-models>).
  This model is already the instruct model; IBM publishes no separate "-instruct" repo for H-Tiny.
- **llama.cpp support:**
  - architecture `granitehybrid`;
  - added in PR #13550 "Granite Four", merged 2025-07-11 (<https://github.com/ggml-org/llama.cpp/pull/13550>);
  - present at the base;
  - no open llama.cpp issue matches "granitehybrid" (GitHub issue search, 2026-11-12).
- **GGUF files:**
  - Publisher, `ibm-granite/granite-4.0-h-tiny-GGUF`: bf16 13.89, Q4_0 3.96, Q4_K_M 4.23, Q3_K_S/M/L 3.06/3.35/3.60,
    Q6_K 5.71 GB. There is no F16 file, only bf16.
  - `bartowski/ibm-granite_granite-4.0-h-tiny-GGUF`: Q4_0 4.09 GB, plus an imatrix file.
  - `mradermacher/granite-4.0-h-tiny-i1-GGUF`: i1-Q4_0 3.98 GB, plus `granite-4.0-h-tiny.imatrix.gguf`.
  - `unsloth/granite-4.0-h-tiny-GGUF`: BF16 and UD-* files.
- **Structure** (config.json, <https://huggingface.co/ibm-granite/granite-4.0-h-tiny/blob/main/config.json>):
  - 40 layers: 36 Mamba-2 and 4 attention layers, NoPE;
  - hidden size 1536;
  - every layer has 64 experts with 6 active (the stock count), expert FFN 512, and a shared expert of 1024;
  - GQA with 12 heads and 4 KV heads;
  - vocabulary 100,352, embeddings tied (the head is `token_embd`);
  - total 6.94 B parameters in the GGUF header.
- **Quantization in the publisher's Q4_0 (header):** every 2-D weight is Q4_0, `token_embd` is Q6_K, and norms,
  router and SSM scalars are F32.

### A2. OLMoE-1B-7B

- **Variants** (all Apache-2.0, `allenai/`):
  - `OLMoE-1B-7B-0924` (base, repo created 2024-07-20) and `-0924-Instruct`;
  - `OLMoE-1B-7B-0125` (base, created 2025-01-21) and **`OLMoE-1B-7B-0125-Instruct`** (created 2025-01-27; the
    chosen variant).

  Sources: <https://huggingface.co/api/models/allenai/OLMoE-1B-7B-0125-Instruct> and the sibling repos. According
  to the GGUF header metadata, 0125-Instruct derives from `OLMoE-1B-7B-0125-DPO`, with an RLVR-GSM stage.
- **llama.cpp support:**
  - architecture `olmoe`, pre-tokenizer `olmo`;
  - added in PR #9462, merged 2024-09-16 (<https://github.com/ggml-org/llama.cpp/pull/9462>);
  - present at the base;
  - the open issues that mention OLMoE are not CPU/ARM bugs (for example #31599, HIP).
- **GGUF files:**
  - Publisher, `allenai/OLMoE-1B-7B-0125-Instruct-GGUF`: F16 13.84, Q4_0 3.93, Q4_K_M 4.21,
    Q3_K_S/M/L 3.02/3.34/3.61, Q6_K 5.68 GB.
  - `mradermacher/OLMoE-1B-7B-0125-Instruct-i1-GGUF`: i1-Q4_0 3.94 GB and IQ3_* 2.69 to 3.08 GB, plus `imatrix.dat`.
  - No bartowski repo was found for 0125 (HF search, 2026-11-12).
- **Structure** (config.json, <https://huggingface.co/allenai/OLMoE-1B-7B-0125-Instruct/blob/main/config.json>;
  paper <https://arxiv.org/abs/2409.02060>):
  - 16 layers, hidden size 2048;
  - 64 experts with 8 active, expert FFN 1024, no shared expert;
  - full multi-head attention (16 heads, 16 KV heads) with QK-norm, context 4096;
  - vocabulary 50,304, head untied (`output.weight`);
  - total 6.92 B parameters (GGUF), 1.3 B active (paper).
- **Quantization in the publisher's Q4_0 (header):** all weights Q4_0 (including `token_embd`), `output.weight` Q6_K.
- **Bytes per token** (*estimated* from tensor shapes): about 690 MB at Q4_0 and about 580 MB with Q3_K experts.
  Granite at 4 experts is 776 MB (campaign 1). The byte advantage is small; the layer count (16 against 40) is
  the larger advantage.
- **Published quantization quality:** none found for llama.cpp Q4/Q3 quants. EAQuant
  (<https://arxiv.org/pdf/2506.13329>) reports OLMoE wikitext-2 PPL for GPTQ-family methods, not GGUF: 3-bit GPTQ
  11.65, 3-bit EAQuant 9.60, 4-bit about 6.7 to 6.9. That is the paper's setup, and I did not check it. The
  3-bit numbers point to a large loss at 3 bits, which supports the small-expert risk.

### A3. LFM2.5-8B-A1B

- **Weights:** `LiquidAI/LFM2.5-8B-A1B`, the reasoning-tuned instruct model with a chain of thought.
  - The repo was created 2026-05-28 (<https://huggingface.co/api/models/LiquidAI/LFM2.5-8B-A1B>), and the blog
    gives the release date as May 28, 2026 (<https://www.liquid.ai/blog/lfm2-5-8b-a1b>).
  - The card says 8.3 B total and 1.5 B active; the GGUF header counts 8.47 B
    (<https://huggingface.co/LiquidAI/LFM2.5-8B-A1B>). **The model exists as described.**
  - The card also lists `new_version: LiquidAI/LFM3-1.7B`.
- **Licence `lfm1.0`, not open source** (<https://huggingface.co/LiquidAI/LFM2.5-8B-A1B/blob/main/LICENSE>):
  - commercial use is licensed only to entities below US$10M annual revenue (§5);
  - non-commercial and research use is allowed.

  Owner decision: research use only, third priority.
- **llama.cpp support:**
  - architecture `lfm2moe`, pre-tokenizer `lfm2`;
  - LFM2 was added in PR #14620 (2025-07-11, <https://github.com/ggml-org/llama.cpp/pull/14620>) and LFM2-MoE in
    PR #16464 (2025-10-07, <https://github.com/ggml-org/llama.cpp/pull/16464>);
  - present at the base.
  - **Open issue #29002** (2026-09-16, CPU, <https://github.com/ggml-org/llama.cpp/issues/29002>): co-batching
    several sequences in one ubatch corrupts the short-conv state of LFM2/LFM2.5. Single-sequence decode is
    byte-identical. How this bears on the screen is in B4.
- **GGUF files:**
  - Publisher, `LiquidAI/LFM2.5-8B-A1B-GGUF`: F16 and BF16 16.95 each, Q4_0 4.84, Q4_K_M 5.16, Q5_K_M 6.03,
    Q6_K 6.96, Q8_0 9.01 GB. **The publisher has no Q3 file.**
  - `unsloth/LFM2.5-8B-A1B-GGUF`: UD-Q3_K_M 3.94, UD-Q3_K_XL 4.00, UD-IQ3_XXS 3.27, UD-IQ4_XS 4.26 GB, plus
    `imatrix_unsloth.gguf_file`.
  - `RemySkye/LFM2.5-8B-A1B-i1-GGUF` is a third-party repo.
  - No bartowski or plain mradermacher stock repo was found.
- **Structure** (config.json, <https://huggingface.co/LiquidAI/LFM2.5-8B-A1B/blob/main/config.json>, and the
  GGUF header):
  - 24 layers: 18 short-conv layers (L_cache 3) and 6 GQA layers (32 heads, 8 KV heads), hidden size 2048;
  - the first 2 layers are dense FFN 7168;
  - the other 22 have 32 experts with 4 active, expert FFN 1792, sigmoid gating with an expert bias, and no
    shared expert;
  - vocabulary 128,000 (up from 65,536 in LFM2, per the blog), embeddings tied (the head is `token_embd`).
- **Quantization in the publisher's Q4_0 (header):** all weights Q4_0, `token_embd` Q6_K.
- **Bytes per token** (*estimated*): about 1.0 GB at Q4_0, of which the head is about 215 MB. That is about 1.3
  times Granite's 776 MB.
- **Phone speed evidence:**
  - The blog says "~30 tokens/s on a phone" without naming a device or a quantization. It also gives 253 tok/s on
    an M5 Max and 146 on a Ryzen AI Max+ 395.
  - An iPhone 17 Pro figure of about 32 tok/s uses CoreAI, not llama.cpp
    (<https://huggingface.co/mlboydaisuke/LFM2.5-8B-A1B-CoreAI>).
  - No Snapdragon llama.cpp number was found.
  - Quality at 4 bits: not found.
- **Likely reasons for the earlier rejection** (my *inference*; the original notes are private):
  - the heavy 128K head;
  - the most active bytes of the three models;
  - the licence;
  - long chain-of-thought output.

  The conv op has been in-tree since 2025-07, and I found no evidence that it is slow on CPU.

---

## Part B. Host screen plan (owner runs, Apple M4)

Placeholders:
- `$LLAMA`: the directory with `llama-quantize` and `llama-perplexity` built at base `620d2225c`;
- `$W`: the work directory;
- `$TEXT`: `wiki.test.raw`.

**Use the same build for all three rows of a model.** In campaign 1, two builds of the same tag differed by
0.12 % (F16) and 0.66 % (Q4_0, 20 chunks) (`reports/00_model_choice.md`). That is large next to an 8 % rule.

Rows per model:

| row | file | what it is |
|---|---|---|
| R | the 16-bit reference | bf16 for Granite (no F16 exists), F16 for OLMoE (the only 16-bit file), **BF16** for LFM2.5 (the weights' native dtype; F16 also exists) |
| P | the publisher's Q4_0 | as downloaded |
| M | the mix, built from R | `Q4_0` base; `*_exps` (gate, up, down) Q3_K; head Q6_K; **no imatrix** |
| Mi | M with a stock imatrix | optional; only where a stock imatrix with a verifiable URL exists |

In the selection rule (part C), "F16" means row R.

### B1. What was verified in the quantize source at the base

Source: <https://raw.githubusercontent.com/ggml-org/llama.cpp/620d2225c/tools/quantize/quantize.cpp> and
`src/llama-quant.cpp`.

- **Flags:** `--tensor-type NAME=TYPE`, `--output-tensor-type`, `--token-embedding-type`, `--imatrix`, `--pure` and
  `--dry-run` all exist.
- **How `--tensor-type` matches:** the name is lower-cased and used as a regular expression with `regex_search`
  on the full tensor name. `ffn_up_exps` therefore matches `blk.N.ffn_up_exps.weight` and not `ffn_up_shexp`.
- **Tied heads:** quantize assumes the embeddings are tied until it sees `output.weight`.
  - For a tied model, `token_embd` gets the output type: `--output-tensor-type` if given, otherwise Q6_K for
    Q4_0 (llama-quant.cpp, around line 495).
  - **Do not pass `--token-embedding-type` for Granite or LFM2.5.** In the outer function it returns early and
    would override the head type.
  - For OLMoE (untied), `--output-tensor-type q6_k` sets `output.weight`; `token_embd` stays Q4_0, as in P.
- **Q3_K block fit:** Q3_K needs rows that are a multiple of 256. Every expert row length is one: 1536 and 512
  (Granite), 2048 and 1024 (OLMoE), 2048 and 1792 (LFM2.5). So no fallback type is expected (*estimated* from
  the shapes).
- **Expert tensor names** (from the headers):

| arch | expert tensors (→ Q3_K) | shared / dense FFN (stay Q4_0) | head |
|---|---|---|---|
| granitehybrid | `blk.N.ffn_{gate,up,down}_exps.weight`, 40 layers | `ffn_{gate,up,down}_shexp` | tied `token_embd.weight` |
| olmoe | `blk.N.ffn_{gate,up,down}_exps.weight`, 16 layers | none | `output.weight` |
| lfm2moe | `blk.N.ffn_{gate,up,down}_exps.weight`, layers 2 to 23 | `ffn_{gate,up,down}` in layers 0 and 1 | tied `token_embd.weight` |

  This matches campaign 1's d6 recipe, where only `*_exps` were Q3_K and the shared expert stayed Q4_0
  (`models/granite-4.0-h-tiny-d6-e6.md`).
- **Imatrix loading:** `common_imatrix_load` tries GGUF and falls back to the legacy `.dat` format by content, so
  file extensions do not matter.

### B2. Downloads (pinned to the repo commit on 2026-11-12)

Expected sha256 values are the HF LFS object ids from the tree API.

```sh
mkdir -p $W/olmoe $W/granite $W/lfm25

# OLMoE-1B-7B-0125-Instruct  (repo commit 2ac1d27317927518c9ef7bd99f91f3f1e4ee288d)
huggingface-cli download allenai/OLMoE-1B-7B-0125-Instruct-GGUF OLMoE-1B-7B-0125-Instruct-F16.gguf \
  --revision 2ac1d27317927518c9ef7bd99f91f3f1e4ee288d --local-dir $W/olmoe          # 13,844,566,016 B
huggingface-cli download allenai/OLMoE-1B-7B-0125-Instruct-GGUF OLMoE-1B-7B-0125-Instruct-Q4_0.gguf \
  --revision 2ac1d27317927518c9ef7bd99f91f3f1e4ee288d --local-dir $W/olmoe          #  3,928,038,048 B
# optional Mi: mradermacher imatrix (repo commit 8eec686da67fd0d16b6f8f37cce694f45fb9f906)
huggingface-cli download mradermacher/OLMoE-1B-7B-0125-Instruct-i1-GGUF imatrix.dat \
  --revision 8eec686da67fd0d16b6f8f37cce694f45fb9f906 --local-dir $W/olmoe          #     21,631,411 B

# Granite-4.0-H-Tiny  (repo commit 08d5a8a9741dd5c1a95d2d39e25253226aa1464e)
huggingface-cli download ibm-granite/granite-4.0-h-tiny-GGUF granite-4.0-h-tiny-bf16.gguf \
  --revision 08d5a8a9741dd5c1a95d2d39e25253226aa1464e --local-dir $W/granite        # 13,891,206,272 B
huggingface-cli download ibm-granite/granite-4.0-h-tiny-GGUF granite-4.0-h-tiny-Q4_0.gguf \
  --revision 08d5a8a9741dd5c1a95d2d39e25253226aa1464e --local-dir $W/granite        #  3,962,938,208 B
# optional Mi: mradermacher imatrix of the stock model (repo commit 4a7953840f6dfb0d0f6ff3a2fe6528c414328e3f)
huggingface-cli download mradermacher/granite-4.0-h-tiny-i1-GGUF granite-4.0-h-tiny.imatrix.gguf \
  --revision 4a7953840f6dfb0d0f6ff3a2fe6528c414328e3f --local-dir $W/granite        #     38,451,424 B

# LFM2.5-8B-A1B  (repo commit 49c14831707011e64d70b2ebd8462ba08d608434)
huggingface-cli download LiquidAI/LFM2.5-8B-A1B-GGUF LFM2.5-8B-A1B-BF16.gguf \
  --revision 49c14831707011e64d70b2ebd8462ba08d608434 --local-dir $W/lfm25          # 16,947,260,640 B
huggingface-cli download LiquidAI/LFM2.5-8B-A1B-GGUF LFM2.5-8B-A1B-Q4_0.gguf \
  --revision 49c14831707011e64d70b2ebd8462ba08d608434 --local-dir $W/lfm25          #  4,844,678,368 B
# optional Mi: unsloth imatrix (repo commit 37563684c4a625e9958906fd924e15095cf9c896)
huggingface-cli download unsloth/LFM2.5-8B-A1B-GGUF imatrix_unsloth.gguf_file \
  --revision 37563684c4a625e9958906fd924e15095cf9c896 --local-dir $W/lfm25          #     17,375,232 B

# check every download against the HF LFS sha256
cd $W && shasum -a 256 -c <<'EOF'
b9258178b1bbab5726901d4d25ef54be8e68bf93ad26af4bbab50400bc8da20e  olmoe/OLMoE-1B-7B-0125-Instruct-F16.gguf
df43bd738dfdcbe889c5f854120c271e9e53e2c624f212ac2523f8bc11ae9bf3  olmoe/OLMoE-1B-7B-0125-Instruct-Q4_0.gguf
3e1de50f624cae516992d2fccb9df1ce9b8f034e64e466e99e8aad1dd94f8bce  olmoe/imatrix.dat
f71ef0b7b6259d2077a5ddd10da519d3e078e3133c9160f9c93d5f488b4535ff  granite/granite-4.0-h-tiny-bf16.gguf
85c1f5484c7974a06d33642779ed633b144d7cfa0b498a64e35d08a7116882ef  granite/granite-4.0-h-tiny-Q4_0.gguf
05020e50cf6074449d8b2a66f4cb2501802c06f009948dd8f6f9f2ccf50732c1  granite/granite-4.0-h-tiny.imatrix.gguf
2b471037695ef5908cb88065c7ebb23fe285a07950d7433fabf0c865273728ab  lfm25/LFM2.5-8B-A1B-BF16.gguf
48ed1465d761311b2fd57b7fb46cf969a20b3a8281945b04e10f52bd1609e715  lfm25/LFM2.5-8B-A1B-Q4_0.gguf
09cf3e3f38c9906dc0239750f524d4e728b8fe985ad04bcfc07dd55bbb6e4474  lfm25/imatrix_unsloth.gguf_file
EOF
```

The `curl` equivalent of any line is
`curl -L -o <file> https://huggingface.co/<repo>/resolve/<commit>/<file>`.

### B3. Building M (and Mi)

Run `--dry-run` first: it prints each tensor's chosen type without writing a file. Check that exactly the
`*_exps` tensors show `q3_K` and that the head shows `q6_K`.

```sh
EXPS="--tensor-type ffn_gate_exps=q3_k --tensor-type ffn_up_exps=q3_k --tensor-type ffn_down_exps=q3_k"

# OLMoE (untied head: output.weight)
$LLAMA/llama-quantize --dry-run $EXPS --output-tensor-type q6_k \
  $W/olmoe/OLMoE-1B-7B-0125-Instruct-F16.gguf Q4_0
$LLAMA/llama-quantize $EXPS --output-tensor-type q6_k \
  $W/olmoe/OLMoE-1B-7B-0125-Instruct-F16.gguf $W/olmoe/olmoe-M.gguf Q4_0
# optional Mi
$LLAMA/llama-quantize --imatrix $W/olmoe/imatrix.dat $EXPS --output-tensor-type q6_k \
  $W/olmoe/OLMoE-1B-7B-0125-Instruct-F16.gguf $W/olmoe/olmoe-Mi.gguf Q4_0

# Granite (tied head = token_embd; do NOT pass --token-embedding-type)
$LLAMA/llama-quantize --dry-run $EXPS --output-tensor-type q6_k \
  $W/granite/granite-4.0-h-tiny-bf16.gguf Q4_0
$LLAMA/llama-quantize $EXPS --output-tensor-type q6_k \
  $W/granite/granite-4.0-h-tiny-bf16.gguf $W/granite/granite-M.gguf Q4_0
# optional Mi
$LLAMA/llama-quantize --imatrix $W/granite/granite-4.0-h-tiny.imatrix.gguf $EXPS --output-tensor-type q6_k \
  $W/granite/granite-4.0-h-tiny-bf16.gguf $W/granite/granite-Mi.gguf Q4_0

# LFM2.5 (tied head = token_embd; do NOT pass --token-embedding-type)
$LLAMA/llama-quantize --dry-run $EXPS --output-tensor-type q6_k \
  $W/lfm25/LFM2.5-8B-A1B-BF16.gguf Q4_0
$LLAMA/llama-quantize $EXPS --output-tensor-type q6_k \
  $W/lfm25/LFM2.5-8B-A1B-BF16.gguf $W/lfm25/lfm25-M.gguf Q4_0
# optional Mi
$LLAMA/llama-quantize --imatrix $W/lfm25/imatrix_unsloth.gguf_file $EXPS --output-tensor-type q6_k \
  $W/lfm25/LFM2.5-8B-A1B-BF16.gguf $W/lfm25/lfm25-Mi.gguf Q4_0
```

The form `--dry-run <input> Q4_0`, without an output path, is accepted at the base: `quantize.cpp` parses
`<input> <ftype>` and skips the output name when `dry_run` is set (around line 570).

**Size of M** (*estimated* from the tensor shapes at Q3_K 3.4375 and Q4_0 4.5 bits per weight):
- Granite about 3.2 GB (campaign 1's d6 file with the same shapes was 3,161,269,888 B);
- OLMoE about 3.1 GB;
- LFM2.5 about 3.8 GB.

### B4. Perplexity and hashes

```sh
# one model at a time; same $LLAMA build and same thread count for R, P, M (and Mi)
for f in $W/olmoe/OLMoE-1B-7B-0125-Instruct-F16.gguf $W/olmoe/OLMoE-1B-7B-0125-Instruct-Q4_0.gguf $W/olmoe/olmoe-M.gguf; do
  $LLAMA/llama-perplexity -m "$f" -f $TEXT -c 512 -b 512 -ngl 0 2>&1 | tee "$f.ppl.log"
done
for f in $W/granite/granite-4.0-h-tiny-bf16.gguf $W/granite/granite-4.0-h-tiny-Q4_0.gguf $W/granite/granite-M.gguf; do
  $LLAMA/llama-perplexity -m "$f" -f $TEXT -c 512 -b 512 -ngl 0 2>&1 | tee "$f.ppl.log"
done
for f in $W/lfm25/LFM2.5-8B-A1B-BF16.gguf $W/lfm25/LFM2.5-8B-A1B-Q4_0.gguf $W/lfm25/lfm25-M.gguf; do
  $LLAMA/llama-perplexity -m "$f" -f $TEXT -c 512 -b 512 -ngl 0 2>&1 | tee "$f.ppl.log"
done
# (add the *-Mi.gguf files to the loops if built)

shasum -a 256 $W/*/*-Q4_0.gguf $W/*/*-M.gguf $W/*/*-Mi.gguf
ls -l $W/*/*.gguf
grep -H "Final estimate" $W/*/*.ppl.log
```

Without `--chunks`, `llama-perplexity` runs every chunk of `$TEXT`. Send back the `Final estimate` lines, the
`n_seq=` line of each log, the byte sizes and the sha256 values.

**LFM2.5 and issue #29002.** At the base, `tools/perplexity/perplexity.cpp` computes
`n_seq = max(1, n_batch / n_ctx)` (line 503) and sets `n_parallel = max(1, n_batch / n_ctx)` (line 2012). So at
`-b 512 -c 512`, **`n_seq` is 1: one sequence per batch, and no co-batching.** The issue's trigger (several
sequences in one ubatch) is therefore absent from this run. I verified this from the source; it was not run.
Two conditions follow:
- Check that each LFM2.5 log prints `n_seq=1`.
- Do not use `-b` larger than `-c` for LFM2.5, because `-b 1024 -c 512` would give `n_seq=2`.

No cross-check is needed for the screen. The campaign's later `-ub 1 / -ub 32 / -b 512` bit-identity checks all
keep `n_seq=1` for the same reason.

### B5. Results table (owner fills in; *host*)

| model | row | file | bytes | sha256 | PPL ± | % over R |
|---|---|---|---|---|---|---|
| OLMoE-0125-Instruct | R | OLMoE-1B-7B-0125-Instruct-F16.gguf | 13,844,566,016 | b9258178… | | 0 |
| OLMoE-0125-Instruct | P | OLMoE-1B-7B-0125-Instruct-Q4_0.gguf | 3,928,038,048 | df43bd73… | | |
| OLMoE-0125-Instruct | M | olmoe-M.gguf | | | | |
| OLMoE-0125-Instruct | Mi | olmoe-Mi.gguf (optional) | | | | |
| Granite-4.0-H-Tiny | R | granite-4.0-h-tiny-bf16.gguf | 13,891,206,272 | f71ef0b7… | | 0 |
| Granite-4.0-H-Tiny | P | granite-4.0-h-tiny-Q4_0.gguf | 3,962,938,208 | 85c1f548… | | |
| Granite-4.0-H-Tiny | M | granite-M.gguf | | | | |
| Granite-4.0-H-Tiny | Mi | granite-Mi.gguf (optional) | | | | |
| LFM2.5-8B-A1B | R | LFM2.5-8B-A1B-BF16.gguf | 16,947,260,640 | 2b471037… | | 0 |
| LFM2.5-8B-A1B | P | LFM2.5-8B-A1B-Q4_0.gguf | 4,844,678,368 | 48ed1465… | | |
| LFM2.5-8B-A1B | M | lfm25-M.gguf | | | | |
| LFM2.5-8B-A1B | Mi | lfm25-Mi.gguf (optional) | | | | |

### B6. Download size and wall time (*estimated*)

| model | download (R + P, + imatrix) | wall time |
|---|---|---|
| OLMoE | 17.8 GB | unknown |
| Granite | 17.9 GB | unknown |
| LFM2.5 | 21.8 GB | unknown |

The only campaign-1 timing note is a full 16-bit pass of Qwen3.5-4B (dense) on the host, which was stopped after
6 h at chunk 152 of 580 (`reports/00_model_choice.md`). The 16-bit row R is expected to be the slowest row by a
wide margin. Plan it as an overnight-class run per model (*estimated*, not timed).

---

## Part C. Selection rule (written before the screen, lead's rule)

"For each model: choose M if PPL(M) ≤ PPL(F16) × 1.08; otherwise choose P if PPL(P) ≤ PPL(F16) × 1.08;
otherwise no file is chosen and the model goes back to the owner. The chosen file's PPL is the model's 'stock
served PPL'; budget = that × 1.05, capped by the hard ceiling F16 × 1.20. The rule is not changed after the first
screen result."

Note: campaign 1's served Granite file (d6-e6: Q4_0 dense, Q3_K experts, Q6_K head) sat at **+6.21 % over F16**
(*host*: 9.6051 against 9.0438, `models/granite-4.0-h-tiny-d6-e6.md`). That is why 1.08 was chosen. That file was
built **with** an imatrix and from abliterated weights, so a stock M built without an imatrix may land higher.

Interpretation for this screen: "F16" in the rule is row R (bf16 for Granite and LFM2.5, F16 for OLMoE). **Lead decision on Mi (clyde, 2026-11-12, before any screen result):** Mi is eligible, because an imatrix is an
ordinary part of a quantization recipe and campaign 1's served file used one. Where the rule says "M", read: of
M and Mi (if built), the files with PPL ≤ PPL(R) × 1.08; if both qualify, the one with the lower PPL is chosen.
A lower served PPL gives a tighter absolute budget; that is accepted. The owner is asked to build and measure Mi
for all three models. The undocumented calibration text of the stock imatrices stays listed as a risk in part D.

---

## Part D. Not verified / risks

- **Q3 files:** none of the three models has a publisher file of the campaign-1 mix. M is a new file built by
  the owner, so its sha256 is only known after the build. Freezing it means a new served file, recorded in
  `models/<name>.md`.
- **No imatrix in M:** for OLMoE's small experts (FFN 1024), 3-bit quantization without an imatrix may exceed
  +8 %. EAQuant reports large 3-bit losses for OLMoE under GPTQ-family methods (A2). Not measured.
- **Imatrix calibration text:** the calibration text of the stock imatrices (mradermacher, unsloth) is not
  documented on the cards I read. If it included wikitext-2 test, Mi would be optimistic.
- **Not run:** that llama-quantize at the base reads the unsloth imatrix (`.gguf_file` extension) correctly.
  Format detection is by content, so it is expected to work.
- **Reference row:** Granite and LFM2.5 use a BF16 reference and OLMoE uses F16, so the rule's "F16" is row R.
  BF16 matmul speed on the host CPU build is unknown.
- **LFM2.5:**
  - issue #29002 is open upstream (B4);
  - the licence limits commercial use (A3);
  - no device speed figure names a Snapdragon device;
  - quality at 4 bits: not found.
- **Granite release date:** IBM's own release date was not found in IBM text (third-party source only).
- **Cost figures:** all bytes-per-token and file-size figures for M are *estimated* from tensor shapes.
- **Wall time:** unknown (B6).
- **Blocked:** nothing was blocked by the network allowlist during this research.
