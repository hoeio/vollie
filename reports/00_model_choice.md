# 00 — Model choice (Phase 0) — **LOCKED: Granite-4.0-H-Tiny abliterated (Huihui)**

Date: 2026-10-31/11-01. Host: Apple M4 Pro (8P+4E, 24 GB), macOS 15.3, CPU only.
Tool: upstream llama.cpp **v0.8.0 @ 8642541b** (latest stable tag, 2026-10-28), built CPU-only with
`GGML_NATIVE=OFF GGML_CPU_ARM_ARCH=armv8.2-a+dotprod+fp16`, Metal/BLAS/KleidiAI off (the SD888's ISA:
dotprod+fp16, no i8mm — see `02_methods.md` §1). F16/BF16 reference runs used the same tag built
natively (`build-native`) because F16 matmul on the armv8.2 build is 2.2x slower; on the first 4
chunks the two builds agree to 0.12% (6.913 vs 6.921), and on a Q4_0 model to 0.66% (20 chunks).

## Fixed evaluation (used for the whole project, `tools/ppl.sh`)
- **Primary (the gate):** wikitext-2-raw `wiki.test.raw`, `llama-perplexity -c 512 -b 512`, all 580/565 chunks.
- **Sanity:** `lab/text/en.txt` (57.7 KB of public-domain English prose, Pride and Prejudice excerpt), same settings.
- Cross-model comparison: **bits per byte** = log2(PPL) × tokens/bytes of the file (`tools/ppl_table.py`;
  assumes the scored second half of each chunk is representative). English only; multilingual ability
  is explicitly not a criterion.
- 10% budget = against the **same abliterated model's F16 weights with the model's own routing
  (6 experts/token)**. Vollie's 4-experts/token setting is an inference-time change and counts against
  the budget.

## Candidates and provenance

| | file (HF) | base | abliteration | reference used |
|---|---|---|---|---|
| **A** | `HoeioUser/vollie-qwen3.5-4b` / `vollie-granite-4.0-h-tiny-q4_0.gguf` (3,979,159,200 B, sha256 a454ea66…; = local `lab/models/granite-fold.gguf`) | ibm-granite/granite-4.0-h-tiny (7B MoE, 64 experts, Mamba2 hybrid) | huihui-ai `Huihui-granite-4.0-h-tiny-abliterated`. The huihui repo now returns HTTP 404, so the method can't be read from its card; mradermacher's GGUF cards only say "abliterated". huihui's usual method is refusal-direction ablation (*unverified for this model*). File = mradermacher i1-Q4_0 (imatrix) with `expert_used_count` 6→4 and residual_scale folded into weights (earlier sessions) | `mradermacher/Huihui-granite-4.0-h-tiny-abliterated-GGUF` `…f16.gguf` (13.9 GB) |
| **B** | same repo, `vollie-qwen3.5-4b-q4_0-mtp.gguf` (2,300,469,312 B, sha256 afcb45e9…) | Qwen/Qwen3.5-4B (dense hybrid Gated-DeltaNet) | huihui-ai `Huihui-Qwen3.5-4B-abliterated`; its card says it was made by **fine-tuning with TRL** on a non-thinking dataset (not pure direction ablation). File = pure Q4_0 + vocab trim 248k→150k + official MTP head grafted | local `lab/models/abl-mtp-bf16.gguf` (BF16 of the same weights + MTP head) |
| C (extra) | `mradermacher/Ling-3.0-tiny-Uncensored-Abliterated-i1-GGUF` `i1-Q4_0` (4.54 GB) | inclusionAI/Ling-3.0-tiny (7.9B/1.3B active MoE, KDA+MLA hybrid, MIT) | Securelayer7 "Heretic/Optuna" abliteration (card: refusals 35→8/100, KL 0.046) | not obtained (rejected on Q4_0 quality, below) |

Extra candidates searched (≤30 min, HF API; see research notes): only **Ling-3.0-tiny** was a plausible win
(fewer active bytes than Granite in principle, MIT, upstream support since PR #26608) → measured, rejected.
Rejected without download: dense 3B Granite 4.1/4.2 abliterated (≈2 GB/token → ≤ ~8 tok/s estimated on SD888),
Huihui-gemma-4-E2B QAT abliterated (≈1.2-1.3 GB/token, 262k vocab head; cannot reach 20 tok/s), Huihui-MoE-1.3B-A0.6B
(too weak per its card), earlier-session rejects (LFM2.5-8B-A1B, Qwen3.5-2B/0.8B, SmallThinker, Trinity Nano —
see memory notes). "English-only" does not help Granite: its 100k vocab is already 97.5% Latin-script (earlier
measurement), so a vocab trim saves <3% of lm_head bytes.

## Results (wikitext-2 test = gate; en.txt = sanity; all PPL measured on host)

| | PPL wiki (q4_0 file) | bpb wiki | PPL en | bpb en | own reference PPL wiki | Δ vs own reference |
|---|---|---|---|---|---|---|
| **A Granite-H-Tiny** (as shipped, 4 experts) | 10.1619 ±0.074 | **0.750** | 12.2009 | **0.841** | F16, 6 experts: **9.0438** (en 10.3768) | **+12.4%** (en +17.6%) |
| A, same file, model's own 6 experts | 9.4210 | 0.725 | 10.7896 | 0.799 | 9.0438 | +4.2% (en +4.0%) |
| A, 5 experts | 9.6176 | 0.732 | 11.1614 | 0.811 | 9.0438 | +6.3% (en +7.6%) |
| B Qwen3.5-4B (MTP, vocab-trimmed) | 15.3745 ±0.139 | 0.909 | 15.8789 | 0.938 | BF16: run stopped at chunk 152/580 (13.7518; Q4_0 at the same chunk 14.5754) | **+6.0%** on the matched 152 chunks (*partial*: the full BF16 pass was stopped after 6 h to free the host once B was rejected) |
| C Ling-3.0-tiny (top-8) | 14.3216 ±0.119 | 0.906 | 13.3105 | 0.962 | not obtained | — |

bpb = bits per UTF-8 byte (lower is better), comparable across tokenizers. Granite is ahead of B and C by
0.16 bpb on wikitext-2 and 0.10-0.12 bpb on en.txt, i.e. ~20% fewer bits per byte.

## Cost per generated token (`tools/cost.py`, machine-independent)

| | total params | active params/token | weight bytes/token | file | KV f16 @ ctx 4096 | recurrent state (r+w per token) | graph nodes/token | arch notes (llama.cpp ARM CPU) |
|---|---|---|---|---|---|---|---|---|
| **A** (4 experts) | 6.94 B | 1.28 B (2.6 GFLOP) | **776 MB** (dense 420, experts 213, router 16 F32, head 126 Q6_K) | 3.98 GB | 34 MB (4 attn layers) | 58 MB f32 Mamba2 | 3253 | `granitehybrid`: 36 Mamba2 + 4 GQA layers, 64 experts; all ops supported on CPU; MUL_MAT_ID repack supported |
| A (6 experts) | | 1.46 B | 882 MB | | | | | |
| **B** | 4.07 B (incl. MTP) | 3.95 B (7.9 GFLOP) | 2227 MB | 2.30 GB | 134 MB (8 attn layers) | 53 MB (Gated DeltaNet) | 1905 | `qwen35`: 24 GDN + 8 attn; MTP head usable via `--spec-type draft-mtp` (PR #22673) but **measured slower on this phone** (5.0 vs 5.9 tok/s, user's device test 2026-10-25) |
| **C** | 7.89 B | 1.14 B (2.3 GFLOP) | 722 MB (untied 157k head 198 MB) | 4.54 GB | 57 MB (6 MLA layers) | ~19 MB KDA (*estimate*) | 3164 | `bailingmoe3`: KDA linear attention + MLA, 128 experts top-8, needs llama.cpp ≥ PR #26608 (Aug 2026); present in Vollie's 620d2225c (`src/llama-arch.cpp`); thinking on by default |

Peak RAM (host "peak memory footprint" = non-reclaimable; mmap'd weights come on top as page cache):
A repack off 128 MB, A repack on 3,931 MB, B repack on 2,269 MB. All fit 8 GB with mmap; only A with full
repack does not (OOM on the user's phone).

## Estimated SD888 decode (`tools/sd888_estimate.py`; **estimates, to be verified on device**)

Anchored on the user's own Vollie phone test of B (5.9 hot / 6.6 cold tok/s, repack on) and scaled by
(a) traffic per token and (b) the host's 4-thread and 1-thread tg ratio on the same build:

| configuration | estimated SD888 decode |
|---|---|
| shipped A, 4 experts, repack off | 8.8 – 12.1 tok/s |
| A, 4 experts, repack on (OOM on 8 GB) | 12.5 – 17.3 tok/s |
| A, 5 experts, repack off | 8.2 – 11.5 tok/s |
| A, 6 experts, repack off | 7.8 – 10.6 tok/s |
| D1, 5 experts, dense repack (shipped 2026-11-01) | 10.1 – 14.3 tok/s |
| D1, 4 experts, dense repack (fails the gate) | 10.9 – 15.3 tok/s |

**Measured afterwards on the user's phone (2026-11-01):** D1 5 experts, 4 threads, stock CPU module:
**7.4 tok/s** writing, below the estimated range. The estimate assumed decode is bandwidth-bound; the
device showed ~60-80 ms/token of fixed per-op cost instead (see `03_final.md` §5). The model choice
does not change: B measured 6.6 cold / 5.9 hot tok/s on the same phone (2026-10-25) with worse quality
(0.909 vs 0.750 bpb), and A has since reached 8.2 tok/s there with runtime changes only.

The MTP head (B) was measured on this phone in an earlier session to *lose* (5.0 vs 5.9 tok/s): decode on
the 888 is compute/thermal-bound and a 2-token verify costs nearly a full step. Granite has no MTP head.

## Decision

**A, Granite-4.0-H-Tiny abliterated**, is locked. It has the best English quality per byte (0.750 vs ~0.91 bpb)
and moves ~1/3 of B's bytes per token. B's only advantage is a lower non-reclaimable footprint than A with
full repack, and that is addressed in Phase 3 without changing the model. C is no faster per byte and
clearly worse in quality.

Consequence for the budget: the *shipped* A configuration (4 experts) is **+12.4% vs its F16 reference and
fails the 10% gate**. Phase 3 has to restore quality (5 experts: +6.3%) before trading anything for speed.

