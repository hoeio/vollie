# Campaign 2 patch series for llama.cpp

Base: `620d2225cc436ab0b001e3200e590e3b286ef9ab` (the same as `../llama.cpp/BASE`). The patches form one series:
apply them in order to a clean tree of the base (`0002` and later expect the ones before them; each was checked to
apply and build in sequence). One lever per patch. The series replaces campaign 1's
`../llama.cpp/0001-cpu-runtime.patch` for campaign 2 and is not meant to be applied together with it. **Every lever is
off unless its environment variable is set; with all knobs unset the build gives the stock graph and stock results.**
Campaign 1 switched several of these levers on by default (`GGML_MMID=3`, `GGML_Q3K_X4`, `GGML_Q3K_KERN`,
`GGML_MMID_FUSE`); here each has a new `VOLLIE_` name and is off when unset, so a campaign-1 knob setting does nothing
on this build.

```sh
for p in patches/llama.cpp-c2/0*.patch; do git -C <llama.cpp at 620d2225c> apply "$PWD/$p"; done
```

Identity checks for `0002` to `0007` (OLMoE `olmoe-Mi`, *host*):
`campaigns/02-25toks-5pct/reports/memo-07-c2-exact-levers.md`.

## 0001-olmoe-expert-skip.patch: expert skip by router weight (numerics change when on)

Touches `src/llama-graph.cpp` (`build_moe_ffn`) and `ggml/src/ggml-cpu/ggml-cpu.c` (`MUL_MAT_ID`). With a skip knob
set, libllama gives each routed-expert `MUL_MAT_ID` the mixture weights as `src[3]`. The CPU module then drops the
low-weight (slot, token) pairs from the row map, so an expert left with no rows is not read at all (in decode
that is every skipped expert), and zeroes their output rows. The rules are scale-free, so they mean the same with or
without top-k renormalisation; written for OLMoE (`norm_w = false`), checked only on OLMoE. Design and checks:
`campaigns/02-25toks-5pct/reports/memo-04-olmoe-expert-skip.md`.

| knob | value | meaning (per token; v = weight of a selected expert, S = sum of the k selected weights) |
|---|---|---|
| `VOLLIE_SKIP_SUM` | permille, > 0 = on | skip if v < value/1000 × S. On renormalised weights (S = 1) this is campaign 1's `GGML_EXPERT_SKIP` |
| `VOLLIE_SKIP_MASS` | permille, > 0 = on | rank by v; skip each expert once the higher-ranked ones already hold value/1000 of S |
| `VOLLIE_SKIP_MAX` | permille, > 0 = on | skip if v < value/1000 × (largest selected v) |
| `VOLLIE_SKIP_MINK` | integer, default 1 | the top MINK experts are never skipped (at least 1) |
| `VOLLIE_SKIP_STATS` | 1 = on | at process exit, one `c2-skip:` line on stderr: tokens, pairs, % pairs skipped, % expert bytes skipped, MB per token selected and skipped, % of the bytes the batch would have read that it did not read |

If several rules are set, an expert is skipped when any of them says so. The kept weights are not renormalised.
Values are read once per process. Only the generic CPU `MUL_MAT_ID` path skips. Experts in a repacked
(extra-buffer) type or on another backend ignore `src[3]` and run in full. OLMoE's Q3_K experts are not repacked.

## 0002-tiny-ops.patch: tiny ops on thread 0, barriers between them skipped (bit-identical)

`ggml/src/ggml-cpu/ggml-cpu.c` (compute loop). Campaign 1's `GGML_TINY_OP` (device: the thread syncs were half of each
token on the SD888, R25). An op whose output and inputs have at most N elements (`GET_ROWS`: output and index list)
runs on thread 0 alone, and the barrier between two consecutive such ops is skipped; the next op that runs on all
threads is still preceded by a barrier. Mat-muls, flash attention and every op that calls `ggml_barrier` or the shared
chunk counter itself never count as tiny; a fused `RMS_NORM` + `MUL` is tiny only if both nodes are. The fusion test was
split out of `ggml_cpu_try_fuse_ops` into `ggml_cpu_fuse_count` so the lookahead makes the loop's own decision.

| knob | value | meaning |
|---|---|---|
| `VOLLIE_TINY_OP` | elements, unset / 0 = off | the threshold N; campaign 1 ran 65536 on the phone. Read per graph |
| `VOLLIE_GRAPH_STATS` | 1 = on (counter only) | at exit one `c2-graph:` line on stderr: one-token graphs seen and, per such graph, the mean graph nodes, ops run, ops run tiny, barriers between ops (the closing one included) and syncs (every `ggml_barrier` call, those inside `MUL_MAT` / `MUL_MAT_ID` included). Adds one untaken branch to `ggml_barrier` when off |

## 0003-q3k-x4.patch: 4-row Q3_K dot on the un-repacked expert path (bit-identical)

`arch/arm/quants.c`, `quants.h`, `ggml-cpu.c` (`ggml_compute_forward_mul_mat_id_one_chunk`). Q3_K experts are never
repacked, so every expert matmul of `olmoe-Mi` takes the generic `MUL_MAT_ID`, on the phone and on the host. There
Q3_K rows are computed four at a time against the shared Q8_K activation; each row's integer block sum is exact and its
float step is the stock one, so every output equals the stock `ggml_vec_dot_q3_K_q8_K` to the bit. arm64 only.

| knob | value | meaning |
|---|---|---|
| `VOLLIE_Q3K_X4` | unset / 0 = off, 1, 2 | 1 = the intrinsics 4-row dot (campaign 1 `GGML_Q3K_X4` + `GGML_Q3K_ILP`); 2 = the hand-scheduled in-register asm kernel (campaign 1 `GGML_Q3K_KERN`, device R47; needs dotprod, else mode 1). Read per graph |

## 0004-mmid-no-barrier.patch: one-token `MUL_MAT_ID` without its internal barrier (bit-identical)

`ggml-cpu.c`. Campaign 1's `GGML_MMID=3` (device R18). In a one-token graph the gate and up expert ops (activation
`[n_embd, 1]`) skip the stock op's internal barrier: every thread quantizes the whole activation row into its own slot
(the stock op quantizes 256-blocks of the same row split across threads; Q8_K is per block, so the bytes are equal),
builds the same row map itself, and claims 32-row chunks from a counter per expert that thread 0 resets before the
barrier in front of the op (two banks by op parity). The down projection (one row per slot) keeps the stock path. 0001's
skip runs inside this path too; its env read is now under a lock because every thread reaches it.

| knob | value | meaning |
|---|---|---|
| `VOLLIE_MMID_NB` | 1 = on, else off | the no-barrier one-token path. Read per graph |

## 0005-mmid-fuse.patch: fused gate + up + SwiGLU expert op (bit-identical)

`ggml-cpu.c`. Campaign 1's `GGML_MMID_FUSE` (device R47/R48). The one-token up and gate `MUL_MAT_ID`s of a layer and
the `GLU` (`SWIGLU`, split form) after them run as one op on the 0004 path: one activation quantize, one map, one counter
per expert; per claimed 32-row chunk the gate and up rows are written as the two ops write them and `ggml_vec_swiglu_f32`
runs over that chunk (boundaries are multiples of 16 floats and `ne01 % 16 == 0`, so each element takes the same SIMD
path as on the whole row). With 0001's skip on, thread 0 zeroes the skipped gate and up rows as the two ops would and
runs the same swiglu on them, so skip and fuse can be on together. OLMoE: 2 ops (up, gate) and the `GLU` become 1 per
layer.

| knob | value | meaning |
|---|---|---|
| `VOLLIE_MMID_FUSE` | 1 = on, else off | the fused op; needs `VOLLIE_MMID_NB=1` (without it nothing fuses). Read per graph |

## 0006-expert-prefetch.patch: expert read-ahead (bit-identical, advice only)

`ggml-cpu.c`. Campaign 1's `GGML_EXPERT_IO` bit 1 (device R18, R26). Before the first `MUL_MAT_ID` of a layer, thread 0
calls `madvise(MADV_WILLNEED)` on the routed experts' slices of every sibling expert op (same `ids`, next 32 nodes), so
mmapped experts are requested together instead of faulting in one page at a time on the compute threads. In a
one-token graph the experts that 0001's skip leaves without a row are not advised. Campaign 1's other `EXPERT_IO` bits
(`MADV_RANDOM`, next-layer guess) are not ported.

| knob | value | meaning |
|---|---|---|
| `VOLLIE_EXPERT_IO` | 1 = on, else off | the read-ahead. Read per graph |

## 0007-sampler-bypass.patch: repetition-penalty fast path (bit-identical)

`src/llama-sampler.cpp` (libllama, not the CPU module and not the app). Campaign 1's `../llama.cpp/0003-sampler-bypass.patch`
with the same code: when the candidates are the whole vocabulary in token order, only the penalised tokens (at most
`last_n`) are visited instead of one map lookup per candidate (OLMoE: 50,304 candidates per token). Any other layout takes
the stock loop. The top-k / top-p / temperature bypass at `temp <= 0` of campaign 1 lives in the app's bindings
(`../llamadart/`), not here.

| knob | value | meaning |
|---|---|---|
| `VOLLIE_SAMPLER_BYPASS` | non-empty and not `0` = on | the fast path. Read per sampler call |

## Not in the series

The two-stage output head (campaign 1 `GGML_HEAD2`) is a numerics change and is not ported yet; its design for OLMoE's
untied head is in memo-07. Campaign 1's other knobs (profiler, A/B, anon cache, Q4 expert copies, touch, chunk2, router
cast, SSM/conv in place) are not ported.
