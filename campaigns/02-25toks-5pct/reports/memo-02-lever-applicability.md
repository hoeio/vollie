# Memo 02: which campaign-1 levers take effect on `olmoe` and `lfm2moe`

Agent model: claude-opus-5-5

Date 2026-11-13. Author: Tom (code). Read-only analysis of `patches/llama.cpp/0001-cpu-runtime.patch` (base
`620d2225c`), `0003-sampler-bypass.patch`, the base sources in `lab/llama.cpp-base`, and the GGUF headers of the three
chosen files (header-only reader, my own code; tensor types and shapes below come from it). **Nothing was run except
`git apply --check` (0001 and 0003 apply cleanly to the base) and the header read. No speed is stated anywhere.**
Numbers are *estimated* (arithmetic from shapes) unless labelled *device* (campaign 1).

Patch references are line numbers in `0001-cpu-runtime.patch` ("P:n"); base references are paths under
`lab/llama.cpp-base`.

## 0. The short version

- **`0001` contains no architecture gate.** It lives in `ggml-cpu` and never sees the model's architecture. What it
  matches is op patterns and tensor names: `ffn_moe_weighted` (P:4381), `ffn_gate_inp` (P:2910, P:4673), `_exps`
  (P:3247, P:4438), `output.weight` / `token_embd.weight` (P:2616, P:5373), `blk.%d.` (P:3097). All three
  architectures build their MoE through the same `build_moe_ffn` (`src/llama-graph.cpp:2121`), so most levers
  match without change. The only architecture-name gate in the series is `0002` (`libllama`, rollback), which is
  not needed: `LLM_ARCH_LFM2MOE` is already in upstream's list (`src/llama-arch.cpp:1244`), and OLMoE has no
  recurrent state.
- **Six levers are on by default and will run on the new models the moment they are loaded:** `GGML_MMID=3`,
  `GGML_Q3K_X4`, `GGML_Q3K_ILP`, `GGML_Q3K_KERN`, `GGML_MMID_FUSE` + `CHUNK2`, and `GGML_CONV_RING` (P:5206-5244).
  None of them has been checked bit-identical on `olmoe` or `lfm2moe`. For LFM2.5, `CONV_RING` takes a code
  path (the generic `d_conv != 4` loop, P:5521-5537) that never ran in campaign 1, on device or host.

## 1. File facts that matter (header read, *estimated* MB = bytes / 10^6)

| | granitehybrid (`granite-Mi`) | olmoe (`olmoe-Mi`) | lfm2moe (`lfm25-Mi`) |
|---|---|---|---|
| layers | 40 (36 Mamba-2 + 4 attn) | 16 attn | 24 (18 short-conv + 6 GQA); 0-1 dense FFN |
| experts / used | 64 / 6 | 64 / 8 | 32 / 4 (layers 2-23) |
| expert tensor shape (gate, up; down) | [1536, 512, 64]; [512, 1536, 64] | [2048, 1024, 64]; [1024, 2048, 64] | [2048, 1792, 32]; [1792, 2048, 32] |
| expert type, total | Q3_K, 2,595 MB | Q3_K, 2,768 MB | Q3_K, 3,331 MB |
| one expert (gate+up+down) | 1.01 MB | 2.70 MB | 4.73 MB |
| routed expert bytes per token, no skip | 243 MB | 346 MB | 416 MB |
| gating, weight normalisation | softmax, top-k renormalised (`norm_w=true`, `granite-hybrid.cpp:266`) | softmax over 64, **not** renormalised (`norm_w=false`, `olmoe.cpp:143`) | sigmoid (`expert_gating_func=2`), selection on sigmoid + `exp_probs_b`, weights renormalised over top-4 (`lfm2.cpp:174`, `llama-graph.cpp:2192-2196, 2262-2276`) |
| head | tied, `token_embd` Q6_K [1536, 100,352], 126.4 MB | **untied**, `output.weight` Q6_K [2048, 50,304], 84.5 MB; `token_embd` Q4_0, 58.0 MB | tied, `token_embd` Q6_K [2048, 128,000], 215.0 MB |
| dense Q4_0 | 412.0 MB | 151.0 MB (attn q/k/v/o) + `token_embd` 58.0 | 238.3 MB |
| dense not repackable on ARM | Q4_1 `ffn_down_shexp` ×5, 4.9 MB | none | Q4_1 `ffn_down` (dense layers 0-1), 18.4 MB |
| router `ffn_gate_inp` F32 | 15.7 MB | 8.4 MB | 5.8 MB |

## 2. The table

Legend: **as is** = takes effect with the patch unchanged (by reading; identity on that model still unchecked);
**small** = takes effect after a small change; **new** = needs new code; **n/a** = not applicable.

| lever (knob) | granitehybrid | olmoe | lfm2moe |
|---|---|---|---|
| small ops on thread 0, syncs skipped (`GGML_TINY_OP`, app-set) | as is, *device* R25 (P:4953-5012, P:5047, P:5122) | as is: element-count rule, no op or name pattern (P:4982-4988) | as is (same) |
| `MUL_MAT_ID` without internal barrier + dynamic chunks (`GGML_MMID=3`, default) | as is, *device* R18 | as is for gate/up: `fast_ok` needs `src1->ne[1]==1`, `n_as ≤ 512`, `n_as·k ≤ 4096` (64·8 = 512) (P:3487-3497). The down projection (`src1->ne[1]` = 8) takes the stock path on every arch, Granite included | as is (32·4 = 128) |
| expert read-ahead (`GGML_EXPERT_IO` bit 1) | as is, *device* R18 | as is: siblings found by shared `ids` within 32 nodes (P:4739-4799); up, gate and down share `selected_experts` (`llama-graph.cpp:2321, 2334, 2435`) | as is |
| `MADV_RANDOM` (bit 2) | as is (lost on device) | as is | as is |
| next-layer guess (bit 4) | as is, *device* 82-85 % hit | as is: the guess ranks raw router logits (P:4552-4564), the same order as softmax | **small**: the real selection is `sigmoid(logit) + exp_probs_b` (`llama-graph.cpp:2175, 2194`), so ranking raw logits ignores the bias and guesses a different set. Fix: find the next layer's `exp_probs_b` ADD and rank by `sigmoid(l) + b`. About 15 lines; advice only, so no risk to the numbers |
| 4-row Q3_K dot (`GGML_Q3K_X4`, default) + ILP chains | as is, *device* R18 | as is: any `n % 256 == 0`, rows in fours inside the `MUL_MAT_ID` chunk (P:2997-3001, P:867-889). Serves all three expert `MUL_MAT_ID`s; K = 2048 (gate, up) and 1024 (down) | as is: K = 2048 and 1792 (7 blocks) |
| in-register Q3_K kernel (`GGML_Q3K_KERN`, default) | as is, *device* R47 | as is: the asm loops over `nb` blocks, no shape constant (P:380-386, P:881-884) | as is |
| fused gate+up+SwiGLU (`GGML_MMID_FUSE`, default) + `CHUNK2` | as is, *device* R47 | as is: same builder (up, gate, `swiglu_split`, `llama-graph.cpp:2321-2377`); needs `ne01 % 16 == 0` (1024 ✓) and 32-row chunks (P:3557-3578). `MUL_MAT_ID` 48 → 32, `GLU` 16 → 0 (*estimated*) | as is: 1792 % 16 = 0 ✓. `MUL_MAT_ID` 66 → 44; `GLU` 24 → 2 (the 2 dense layers keep theirs) (*estimated*). No swiglu clamp for this arch (`llama-graph.cpp:2356`; key absent) |
| in-place SSM state (`GGML_SSM_INPLACE`, default) | as is, *device* (36 layers) | n/a (no `SSM_SCAN`; the matcher finds 0 layers, P:4210-4272) | n/a (short conv, no scan) |
| conv state in place (`GGML_CONV_RING`, default) | as is, *device* (36 layers, unrolled `d_conv=4` path) | n/a | **as is, by reading, but unverified and on by default**: LFM2's `concat(conv_state, transpose(b·x))` + one `CPY` + `SSM_CONV` (`lfm2.cpp:240-274`) meets every matcher condition I checked (P:4302-4348: state width 2 ≤ 8, CPY view offset 4 bytes, the state comes from `build_rs`'s `GET_ROWS`, not the cache). It then runs the **generic** loop (P:5521-5537, `ns1 = 2`), which never ran in campaign 1. The sum order matches stock `ssm_conv`, but this has to be shown at three batch sizes before any LFM2 identity claim. Expected: 18 `CONCAT` + 18 `CPY` dropped per token |
| expert skip by router weight (`GGML_EXPERT_SKIP`) | as is, *host* PPL and *device* | **small, semantics differ**: the threshold is applied to the `ffn_moe_weighted` MUL's `src[1]` as it stands (P:3357-3401, P:4378-4386). For OLMoE that is the raw softmax probability over 64 experts, not renormalised, so the 8 weights sum to S < 1 and a permille value means a different, per-token-varying cut. `SKIP_RENORM`'s 1/(1−dropped) (P:3396) and `TOPP` (cumulative mass) also assume S = 1; `GAP` is scale-free. Granite-like semantics: compare `p/S` behind a new env flag, about 10 lines, bit-identical when off. Either way the threshold needs a new host PPL screen | as is (weights are renormalised to sum 1, as Granite's). The bias affects selection only, and the weights are unbiased sigmoid values. With k = 4 the mean weight is 0.25 (Granite 1/6), so the threshold needs a new host screen |
| two-stage head (`GGML_HEAD2`, Q4_0 screen + exact rows) | as is, *device* R31: tied, the repacked head's plain rows come from the `token_embd` `GET_ROWS` source (P:2625-2635) | **new**: the head is untied and repacked (Q6_K 8x4 on dotprod, `repack.cpp:5061-5071`; 50,304 % 8 = 0). The head2 code finds no plain `output.weight` rows, returns false and silently runs the stock head (P:2627, "an untied head that is repacked has no plain rows here"). See §4 | as is (tied; same path as Granite). Screen copy 147.5 MB anon (Granite 86.7, *device* figure). M = 2,048 of 128,000 rows has to be re-screened |
| sampler fast path (`VOLLIE_SAMPLER_BYPASS`, `0003`) | as is, *device* R33 | as is: keyed on `ctx->n_vocab` and the penalty map, nothing model-specific (`0003`: lines 25-40). Gains scale with vocabulary: 50,304 (*estimated* half of Granite's loop) | as is: 128,000 entries (*estimated* 1.28× Granite's loop) |
| Q4_0 repack interaction (stock extra buffers + `GGML_REPACK_RELEASE`) | stock repack: 412.0 MB Q4_0 + 126.4 MB Q6_K head copy = 538.5 MB anon (*estimated*) | 151.0 MB Q4_0 (attn) + 84.5 MB Q6_K `output` = 235.5 MB anon; `token_embd` Q4_0 stays mmapped (`GET_ROWS` only); experts Q3_K are never repacked (no Q3_K entry in `repack.cpp:4974-5071`) | 238.3 MB Q4_0 + 215.0 MB Q6_K head copy = 453.3 MB anon; Q4_1 18.4 MB plain |
| router F16 cast (`GGML_ROUTER_CAST`, closed numerics experiment) | as is | as is (name `ffn_gate_inp`, F32; 8.4 → 4.2 MB, *estimated*) | as is (5.8 → 2.9 MB) |
| Q4_0 expert copies (`GGML_EXPERT_Q4_LAYERS`, dead) | as is | **small**: its rank list is Granite's 40 layers (P:3219); usable only with `GGML_EXPERT_Q4_LIST`. Closed in campaign 1 | same |
| shared-expert pruning (`GGML_SHEXP_PRUNE`, closed) | as is | n/a (no shared expert; also hard-coded 1536 / 1024, P:3824-3834) | n/a |
| kernel bench (`GGML_CPU_BENCH`) | as is | **small**: the shapes are Granite's (K 1,536 × 512 rows, K 512 × 1,536 rows, P:2458-2459); OLMoE needs K 2,048 and 1,024. About 5 lines | **small**: K 2,048 and 1,792 |
| profiler / A/B (`GGML_CPU_PROF`, `GGML_CPU_AB*`) | as is | as is (decode detection from node 0, the embedding `GET_ROWS`, P:5180) | as is |

## 3. Per-token graph facts (what the builder tells, what it does not)

Granite reference: 1,775 nodes and 666 barriers per token after the fused op (*device*, R47/R48).

Counts by op type that follow directly from the builders (*estimated*, one decode token, fused op on):

| op | granitehybrid | olmoe | lfm2moe |
|---|---|---|---|
| `MUL_MAT_ID` (fused pair counted once) | 80 (*device*) | 32 (16 fused + 16 down) | 44 (22 + 22) |
| `GLU` | 76 (*device*) | 0 | 2 |
| `MUL_MAT` (weights) | 249 from the builder (72 ssm + 16 attn + 120 shexp + 40 router + head) | 81 (64 attn + 16 router + head) | 89 (36 conv + 24 attn + 6 dense FFN + 22 router + head) |
| attention ops (`FLASH_ATTN_EXT` if FA is on) | 4 | 16 (full MHA, 16 KV heads × 128) | 6 (GQA, 8 KV heads) |
| `SSM_SCAN` / `SSM_CONV` | 36 / 36 | 0 / 0 | 0 / 18 |

**What I cannot infer:** total nodes and barriers for OLMoE and LFM2.5. They depend on graph-time fusions,
whether flash attention is on, which ops are empty views, and the tiny-op lookahead, so they have to be counted, not
derived. A caution on method: my builder-derived Granite `MUL_MAT` count (249) does not reproduce the 265 in the
docs' sample `cpu-prof` line (`docs/runtime-knobs.md`, an older state). So treat derived counts as approximate until
one profiled host run gives `nodes=` and `barriers=` per model (`GGML_CPU_PROF=1`, a one-token graph; the owner runs
it). What the structure does say: OLMoE has 16 layers against Granite's 40, and per layer one attention block (about a
dozen small ops: norms, QK-norms, RoPE, KV writes) plus one MoE block (router, softmax, argsort, `GET_ROWS`, three
expert ops, weighting, eight views and seven `ADD`s). Its node count should be well under Granite's. That is
*estimated* and not a number.

## 4. Sizes for the "small" and "new" cells (*estimated*)

| item | lines | bit-identity risk |
|---|---|---|
| head2 for an untied repacked head (OLMoE). Option A: record the mmap source pointer of each repacked tensor in `ggml_backend_cpu_repack_buffer_set_tensor` (the hook `0001` already has for `GGML_REPACK_RELEASE`, P:5612-5616) and use it as head2's plain rows. Needs `use_mmap`, and `GGML_REPACK_RELEASE` must skip the head, or the exact rows fault in from flash. Option B: un-interleave Q6_K 8x4 rows from the repacked copy | A: 30-40; B: 80-120 | none when off. When on it is a declared numerics change as on Granite; the exact rows (plain Q6_K `vec_dot`) match the stock repacked gemv to float rounding, not to the bit, as already on Granite (P:2497-2499). Option A's risk is a dangling pointer when mmap is off, so guard it. B's risk is layout bugs (round-trip test) |
| expert skip on normalised weight for OLMoE (`p/S`), new flag | ~10 | none when off; a numerics change when on (budgeted) |
| bit-4 guess with sigmoid + bias (LFM2.5) | ~15 | none (advice only) |
| `GGML_CPU_BENCH` shapes per model | ~5 | none (measurement) |
| new: in-place KV / attention work for OLMoE's 16 full-MHA layers | not sized: no campaign-1 lever covers attention, and Granite had only 4 attention layers | not applicable yet |

New work goes into a new patch series (`0004-…` and up, one lever per patch); `0001` is not edited.

## 5. Resident memory (*estimated*, from the headers)

| | anon (repacked dense + head copy) | + head2 Q4_0 screen | mmapped experts | page cache on the phone |
|---|---|---|---|---|
| granitehybrid | 538.5 MB | +86.7 MB | 2,595 MB | about 2.1 GB (*device*, campaign 1) |
| olmoe | 235.5 MB | +58.0 MB (once §4 exists) | 2,768 MB | |
| lfm2moe | 453.3 MB | +147.5 MB | 3,331 MB | |

OLMoE frees about 300 MB of anon memory compared with Granite (*estimated*), and that memory can hold page cache.
Its experts are 7 % larger, though, and each routed expert is 2.7 times Granite's. LFM2.5's experts exceed Granite's by
about 0.74 GB.

## 6. Not verified

1. Bit-identity of the six default-on levers on `olmoe` and `lfm2moe` (`-ub 1`, `-ub 32`, `-b 512` per-chunk PPL).
   For LFM2.5, `CONV_RING`'s generic path above all. Until this is done, a stock-PPL measurement on these files
   with the patched build is not a stock measurement. Either set `GGML_MMID=0 GGML_Q3K_X4=0 GGML_Q3K_KERN=0
   GGML_MMID_FUSE=0 GGML_CONV_RING=0` for the stock reference, or use an unpatched build.
2. That the `CONV_RING` matcher actually fires on LFM2.5's graph (expect a `conv-ring layers=18` line once per process)
   and that nothing between the `CONCAT` and `SSM_CONV` writes over the old state in the allocator's real layout.
3. That llama.cpp at `BASE` repacks the Q6_K head on the SD888 A78 (dotprod, no i8mm), so that OLMoE's head2 really
   falls back. On a host or device with `use_extra_bufts` off the head is plain and head2 would work as is.
4. Whether the campaign build enables KleidiAI. If it does, some Q4_0 tensors may go to its buffer instead of
   `CPU_REPACK`, and the memory column changes.
5. Node, barrier and op counts per token for OLMoE and LFM2.5 (§3). Flash attention on or off on the phone build.
6. The fused-op adjacency (up, gate and `GLU` consecutive in `cgraph->nodes`) on the two new graphs. It follows from
   the DFS order of `build_moe_ffn`, but was not printed.
7. The sampler bypass with OLMoE's vocabulary (`ctx->n_vocab` against the candidate count, 50,304 tensor rows).
8. Bit-4 guess hit rates on either new model, and the value of read-ahead on OLMoE's 2.7 MB experts.
9. Expert-skip PPL curves for either model. Nothing about the threshold transfers from Granite.
