# c3 memo-12 work, round 15 (Bonnie): the KV cache type is already Q8_0 on the phone; what is in O for a dense token

- **Date:** 2026-11-23. **Author:** bonnie, `claude-fable-5-2`. Follows [bonnie-joins-r14.md](bonnie-joins-r14.md). Not
  committed. No verdict here. **Ebony has not reviewed this file.**
- **Labels.** *device*, *host*, *estimated*, *source* as before. Triples are Yi / Llama-2 / DeepSeek.
- Nothing was measured this round. §1 is source reading (the kit's app `lab/app-c3` and runtime
  `lab/llama.cpp-c3r32`); §2 is arithmetic on *device* inputs from R72 and campaign 2. No web source.

## 0. Answer first

1. **KV in Q8_0 is not a lever. The app has set it since campaign 1** (*source*, `lab/app-c3/lib/assistant.dart`
   lines 468 to 472: `cacheTypeK` and `cacheTypeV` are `q8_0` on the CPU, and the library turns flash attention on
   for it). I listed it as a lever in memo-08 and as a candidate in r14 §3; both are wrong. **The +0.2 tok/s of
   r14 §3 is withdrawn**: it is already in whatever the phone runs.
2. **It is a third way in which the phone's stock path differs from the freeze** (after KleidiAI and the head,
   r9 to r12). The gate script runs `llama-perplexity` with the default F16 cache. Campaign 1's and 2's models
   had few or narrow attention layers; ours have 32 or 30 full ones. **Nobody has measured the three subjects'
   perplexity with the cache type the phone uses.** The second reference Tom is building should carry it.
   Specification in §1.4.
3. **Corrections that follow** (§1.3): the KV buffer at n_ctx 512 is 17 / 136 / 128 MiB, not the F16 figures in
   r8 and in the R73 plan (32 / 256 / 240 MiB). At the kit's fixed n_ctx 4096 it is 0.13 / 1.06 / 1.00 GiB of
   anonymous memory, which matters for residency if the kit's context is not lowered.
4. **O for a dense token is about 5 / 7 / 7 ms, band 3.5–8 / 5–10 / 5–10** (*estimated*; §2). Attention over the
   KV is most of it on the two full-attention subjects: **3 to 6.5 ms at 200 positions, growing with every
   token of context.** The stacks' 5.2 ms is about right for Yi and about 2 ms low for Llama-2 and DeepSeek.
5. **A width-2 pass pays the attention twice.** The stacks charge 0.5 / 1.5 / 1.5 ms for the second position; by
   §2 it is nearer 3.2 / 5.3 / 5.0 ms. That is another −0.25 to −0.3 tok/s on rungs 5 to 7 of r14
   (*estimated*).

## 1. The KV cache type

### 1.1 What the kit does (*source*)

- `assistant.dart` `_params`: `cacheTypeK` / `cacheTypeV` = `KvCacheType.q8_0` when the device is the CPU. The
  comment: "q8_0 halves the KV cache with no visible quality loss; llamadart turns on flash attention for it."
  The campaign-3 kit's app is this file (`lab/app-c3`, version 1.35.0).
- The library (`third_party/llamadart`, 0.15.6) passes them to `llama_context_params.type_k` / `type_v` and
  enables flash attention.
- The runtime (`llama-context.cpp` lines 4064 to 4070, 557 to 560): a quantised V cache requires flash attention;
  with the default "auto" it switches it on.
- On the CPU flash attention is `ggml_compute_forward_flash_attn_ext` (`ops.cpp`): per head, the query is
  converted once to the K type's dot-product format, one dot product per cached position, then each V row is
  accumulated with its weight.
- Campaign 1's own bench scripts say the same: `tools/host_bench.sh` and `tools/bench_device.sh` run
  `-fa on -ctk q8_0 -ctv q8_0`; `reports/02_methods.md` §2 is headed "Vollie config: FA on, q8_0 KV".
- **The gate does not:** `tools/ppl.sh` passes no cache type, so the freeze and every screen ran F16.

### 1.2 Why it did not matter before and may now

| model | attention layers | K and V width per layer | KV elements per position |
|---|---|---|---|
| Granite-4.0-H-Tiny (campaigns 1, 2) | 4 of 40 | 512 | 4,096 |
| OLMoE (campaign 2) | 16 | 2,048 | 65,536 |
| Yi-1.5-6B | 32 | 512 (4 KV heads of 128) | 32,768 |
| Llama-2-7B | 32 | 4,096 | 262,144 |
| DeepSeek-7B | 30 | 4,096 | 245,760 |

Every attention output on Llama-2 and DeepSeek is a sum over 8-bit keys and values. The expected effect is
small (8 bits with a scale per 32 values is finer than the Q4_0 weights around it); my band is +0.0 to +0.2 % of
perplexity (*estimated*, no measurement on any 7B model in this repository). It still has to be a number, for
the same reason as the KleidiAI shift: it is part of the unmodified runtime on the phone.

### 1.3 Bytes (*estimated*; Q8_0 is 34 bytes per 32 values)

| | per position, Q8_0 | at n_ctx 512 | at n_ctx 4096 (the kit's fixed value since r27) | F16 at 512, for comparison |
|---|---|---|---|---|
| Yi | 34,816 B | 17 MiB | 136 MiB | 32 MiB |
| Llama-2 | 278,528 B | 136 MiB | 1.06 GiB | 256 MiB |
| DeepSeek | 261,120 B | 128 MiB | 1.00 GiB | 240 MiB |

- **Correction to r8 §1.3:** "what stays anonymous beside the side file" is 0.02 / 0.14 / 0.13 GB at n_ctx 512,
  not 0.03 / 0.27 / 0.25. The residency steps of r8 §2 carry 0.25 to 0.30 GB of anonymous memory and so are on
  the safe side.
- **For the R73 plan** (`memo-10-work/clyde-r73-plan.md` line 48, which gives F16 sizes): with the app as it is
  the cache is Q8_0, half those sizes. Its choice of n_ctx 512 still stands: 1.0 GiB of cache at 4096 is not
  something Llama-2 or DeepSeek can hold beside a 3.8 GB body. Campaign 2's `VOLLIE_KV_DONTNEED=1` keeps the
  untouched part of the cache out of memory and is in the kit's tree.
- A plan or kit that states the KV type should say which it is. If a round wants F16 (to match the freeze), the
  app has to be changed for it.

### 1.4 Host screen for Tom (not run by me)

On the KleidiAI build of r9 §3 (or H1's build of the kit's tree, if the head knob is being tested), the three
served files, 40 chunks:

```
for m in yi15-6b llama2-7b deepseek7b; do
  for kv in f16 q8_0; do
    GGML_KLEIDIAI_SME=0 sh tools/ppl.sh --llama-dir lab/llama.cpp-c3/build-kai/bin --text lab/c3/text/wiki.test.raw \
      c3k40-$m-q4_0-b512-fa-$kv lab/c3/models/$m-q4_0.gguf --chunks 40 -fa on -ctk $kv -ctv $kv
    GGML_KLEIDIAI_SME=0 sh tools/ppl.sh --llama-dir lab/llama.cpp-c3/build-kai/bin --text lab/c3/text/wiki.test.raw \
      c3k40-$m-q4_0-ub1-fa-$kv lab/c3/models/$m-q4_0.gguf --chunks 40 -fa on -ctk $kv -ctv $kv -ub 1
  done
done
```

| read | expected | says |
|---|---|---|
| `-fa on -ctk f16` against the run without these flags (Tom's stock run) | equal per chunk, or equal to rounding | whether "auto" already meant flash attention on the laptop; if not equal, flash attention itself is a fourth difference |
| `q8_0` against `f16`, same build | +0.0 to +0.2 % at chunk 40 on each subject; P(any subject above +0.3 %) 0.1 | the cost of the phone's cache type |
| the log's context lines (`flash_attn`, `type_k`, `type_v`, the KV buffer size) | `q8_0`; 17 / 136 / 128 MiB at n_ctx 512 | that the flags took |

- Twelve short runs, about an hour by the freeze logs' seconds per chunk (r9).
- **If the shift is above 0.1 %, the full-set second reference should be run with the phone's settings:**
  KleidiAI, `-fa on -ctk q8_0 -ctv q8_0`, and whatever the head knob decision is. That is one reference, not
  three; it should be defined once, in the pre-registration of the first round that loads a model.
- The same flags belong in every later host check of a lever (Tom's 0006 emulation included).

## 2. What is in O for a dense token

**Definition.** The body (120.6 / 140.6 / 130.8 ms, *device*, R72) is the seven weight matmuls per layer run
through ggml on the phone, with their activation quantisation and their thread barriers. The head is separate.
**O is everything else in a token.** No dense decode has run on this phone, so every row below is *estimated*
unless marked.

Basis: campaign 2's device profile of Granite and its split for OLMoE (`memo-24` §B, *device* totals 4.1 to 5.1
ms, split *estimated*); R72 L6 (*device*); the CPU flash-attention loop (*source*). Decode at about 200
positions (the official prompt is 150 to 190 tokens with its wrapper, Ebony r4 §3, *host*), four threads.

| part | work per token | Yi | Llama-2 | DeepSeek | basis |
|---|---|---|---|---|---|
| **attention over the KV** (32 or 30 flash-attention ops) | K: one dot product of 128 values per head and cached position; V: one weighted add of 128 values per head and position. At 200 positions: 52 / 52 / 49 million element-steps; KV bytes read 7 / 56 / 52 MB | **2.5; 1.5–4.5** | **4.5; 3–6.5** | **4.2; 3–6.3** | OLMoE: 0.8 to 1.6 ms for 14 MB and 16 layers (campaign 2); scaled by bytes for the full-attention subjects and by element-steps for Yi, whose small cache is re-read by 32 heads |
| graph dispatch | 617 / 617 / 579 nodes at about 1 µs | 0.6; 0.4–0.9 | 0.6; 0.4–0.9 | 0.6; 0.4–0.9 | campaign 2: 0.96 to 1.4 µs per node with the profiler on (*device*) |
| thread sync of the non-matmul nodes | – | 0.21 | 0.21 | 0.20 | **R72 L6, *device***, 4 threads |
| straggler waits on the small ops | – | 0.5; 0.2–1.0 | 0.5; 0.2–1.0 | 0.5; 0.2–1.0 | campaign 2: OLMoE 0.8 to 1.1 ms in all, part of it on matmuls |
| RMS norm (65 or 61 ops of 4,096) | 0.27 million values | 0.2; 0.1–0.4 | 0.2 | 0.2 | OLMoE 0.3 ms |
| RoPE on q and k | 0.15 to 0.26 million values | 0.15; 0.1–0.3 | 0.2 | 0.2 | OLMoE 0.1 to 0.2 ms |
| K and V written to the cache as Q8_0 | 33 / 262 / 246 thousand values | 0.05 | 0.15; 0.1–0.3 | 0.15 | OLMoE 0.1 ms |
| SiLU × up, residual adds | 0.35 + 0.26 million values | 0.3; 0.15–0.5 | 0.3 | 0.3 | arithmetic |
| sampler and token hand-back to the app | argmax over 64,000 / 32,000 / 102,400 logits, the app's sampler chain, one callback | 0.3; 0.1–1.5 | 0.2; 0.1–1.0 | 0.4; 0.1–2.5 | with campaign 2's `VOLLIE_SAMPLER_BYPASS=1`; without it the chain walks the vocabulary and the upper end applies |
| **O** | | **4.8; 3.5–8** | **6.9; 5–10** | **6.8; 5–10** | sum of centres; bands by judgement, not by adding extremes |

Reading:
- **Attention is the term that scales.** It is linear in the position: at 500 positions it is 6 / 11 / 10 ms. The
  claim run decodes 32 tokens after 150 to 190, so 200 is the right centre for it and wrong for long chats.
- **Yi's attention is compute, not bytes.** Its cache is 8 times smaller, but 32 query heads each walk it. So its
  KV type hardly changes its speed; on Llama-2 and DeepSeek the Q8_0 cache halves 105 MB of reads per token.
- **The 8 ms allowance was fair as an allowance; the 5.2 ms the stacks use is low for two subjects.** Using
  §2's centres instead of 5.2: Yi +0.4 ms the other way (4.8), Llama-2 and DeepSeek −1.7 and −1.6 ms. Per committed
  token at width 2 that is about +0.03 / −0.15 / −0.15 tok/s in r14's rung 6.
- **Width 2.** A second position repeats the attention and the small ops for itself. Its cost is O minus the
  per-pass parts (dispatch, sync, waits, sampler): about **3.2 / 5.3 / 5.0 ms**, against the stacks' `xpos` of 0.5
  / 1.5 / 1.5. Per committed token (÷ 1.33 to 1.45) that is 2.0 / 2.6 / 2.5 ms more: about −0.25 / −0.3 / −0.3
  tok/s.
- Together on rung 6 (*estimated*): roughly −0.2 / −0.45 / −0.45 tok/s. I have not re-run the tables: these
  centres are estimates of estimates, and R73 or a dense decode on the host with the profiler replaces them.

### 2.1 Bands for R73 (first dense decode), if it prints campaign 1's per-op profile

| printed quantity | Yi | Llama-2 | DeepSeek |
|---|---|---|---|
| steady token time minus the weight matmuls minus the head (O) | 4.8; 3.5–8 ms | 6.9; 5–10 | 6.8; 5–10 |
| flash-attention ops, summed, at 180 to 220 positions | 2.5; 1.5–4.5 ms | 4.5; 3–6.5 | 4.2; 3–6.3 |
| its slope per 100 positions | 1.2; 0.7–2.2 ms | 2.2; 1.5–3.2 | 2.1; 1.5–3.1 |
| everything else in O | 2.3; 1.6–4 ms | 2.4; 1.7–4 | 2.6; 1.8–4.5 |
| KV buffer in the load log, n_ctx 512 | 17 MiB | 136 MiB | 128 MiB |

A miss high on the attention row with the others inside would point at the V accumulation on a quantised cache
(it converts each row before adding); the F16 cache is then the comparison arm.

## 3. Next questions, ranked

1. **Define "the phone's stock configuration" once, as a list:** build option (KleidiAI), head placement,
   KV type, flash attention, context size, threads, sampler bypass, the kit's environment. Each with where it is
   set and whether the freeze had it. The lead needs it for the pre-registration of the first model-loading
   round; I have found three mismatches one at a time and there may be more. Mine, from the app and kit source.
2. **Re-run the ladder with §2's O and second-position costs** once Tom's stock runs and R75 are in, so the
   rungs move once.
3. **Can the second position's attention be made cheaper?** The two positions of a width-2 pass share every
   cached K and V; only their queries differ. The CPU loop walks the cache once per query. A two-query walk
   would read each row once. Source reading; a small exact kernel change if it holds.
4. **DeepSeek's head** (Tom's H2 dump, then training).
5. **Cold read rate of the served file; fixed share: which C**, after R75.

## 4. Limits

- §1 reads the app's and the library's source; I did not see a load log from the phone that prints the cache
  type. Campaign 2's memo-24 measured OLMoE's KV buffer at the Q8_0 size on the phone, which agrees.
- I did not check what "auto" flash attention resolves to in `llama-perplexity` on the laptop; §1.4's first row
  is there for that.
- §2 scales a campaign-2 estimate for another model. Its attention row could be off by a factor of two either
  way; the band says so.
- The sampler row assumes the bypass is on in the kit; I did not read the kit's environment for campaign 3.
- §2's stack corrections are not re-run through the scripts.

## 5. Sources (2026-11-23)

Repository: `lab/app-c3/lib/assistant.dart` (lines 454 to 478), `lab/app-c3/pubspec.yaml`,
`lab/app-c3/third_party/llamadart/lib/src/core/models/config/kv_cache_type.dart`, `…/inference/model_params.dart`
(lines 424 to 431, 584 to 587), `…/backends/llama_cpp/load_param_helpers.dart` (lines 104 to 117);
`lab/llama.cpp-c3r32/src/llama-context.cpp` (lines 316 to 317, 557 to 560, 3992 to 4004, 4064 to 4070),
`ggml/src/ggml-cpu/ops.cpp` (lines 8996 to 9075); `tools/ppl.sh`, `tools/host_bench.sh` line 10,
`tools/bench_device.sh` line 45; `reports/02_methods.md` (§2, row 16), `docs/round5-comparison.md` line 51,
`docs/runtime-knobs.md` line 62; campaign 2 `reports/memo-24-anon-memory-and-rest.md` (lines 21, 97, 355 to 376);
`patches/llama.cpp-c1-olmoe/README.md` (0008, 0009); campaign 3 `reports/memo-10-work/clyde-r73-plan.md` line 48,
`reports/memo-08-work/bonnie-path.md` line 140, `reports/R72-result.md` (L6); [ebony-joins-r4.md](ebony-joins-r4.md) §3.
