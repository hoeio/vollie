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
`campaigns/02-25toks-5pct/reports/memo-07-c2-exact-levers.md`; for `0008` and the `0009` smoke check:
`campaigns/02-25toks-5pct/reports/memo-11-kit-runtime-and-head.md`.

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
Values are read once per process (from 0008 on: per graph). Only the generic CPU `MUL_MAT_ID` path skips. Experts in a repacked
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
skip runs inside this path too; its env read is now under a lock because every thread reaches it. (0008 moves that read to the
calling thread before the workers start; the lock is gone.)

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

## 0008-kit-interface.patch: knobs per graph, per-prompt lines, per-token profile, in-run A/B (bit-identical)

`ggml-cpu.c`, `ggml-cpu.cpp` (one proc address), `src/llama-context.cpp`, `src/llama-graph.cpp`, `src/llama-sampler.cpp`.
Nothing here changes a result; the knobs below only print lines or choose, per graph, between values whose results are
already known to be equal (exact levers) or that are being compared (skip, head2). Line formats and frozen regexes:
`campaigns/02-25toks-5pct/reports/memo-11-kit-runtime-and-head.md`.

- **Knobs per graph.** Every `VOLLIE_*` knob of 0001 to 0009 is read again at the start of every graph (the CPU module:
  `ggml_graph_compute`, on the calling thread before the workers start; libllama: every graph build and every sampler
  call), so an app can change them between prompts. 0001's once-per-process latch and 0004's lock are gone.
- **Prompts.** A prompt starts at the first graph, at a graph of more than one token after a one-token graph, or when
  `VOLLIE_RUN` changes. When a prompt ends (the next one starts, or the context is freed), its counters are printed and
  reset: `c2-skip-run:` (with `VOLLIE_SKIP_STATS=1`), `c2-graph-run:` (with `VOLLIE_GRAPH_STATS=1`), `c2-head-run:`
  (with `VOLLIE_HEAD2`). The process-exit lines `c2-skip:`, `c2-graph:` and `c2-head:` stay on stderr for the host tools.
- **Lines** go through ggml's log callback at INFO level (the one `llama_log_set` installs; the app's native log),
  not `atexit`. `VOLLIE_LINES=stderr` writes them to stderr instead.
- **Profile.** `VOLLIE_PROF=1`: one `cpu-prof` line per one-token graph (campaign 1's per-token style): side, token,
  µs, page faults, cpu4/cpu7 clock, ops, barriers, syncs, skipped pairs, time per class (head, router, other
  `MUL_MAT`, experts `MUL_MAT_ID`, attention `FLASH_ATTN_EXT` + `SOFT_MAX`, other, wait at barriers) and the op table.
- **A/B.** `VOLLIE_AB=knob:<NAME>:<A>:<B>` (campaign 1's `GGML_CPU_AB=knob:...`): one-token graph t of a prompt runs
  side A while t <= `VOLLIE_AB_WARM` (default 16), then blocks of `VOLLIE_AB_BLOCK` (default 8) graphs in the order
  A B B A (`VOLLIE_AB_BAAB=1`: B A A B); prompt graphs run side A. The side's value replaces the knob's env value for that
  graph. Names: `TINY_OP`, `Q3K_X4`, `MMID_NB`, `MMID_FUSE`, `EXPERT_IO`, `SAMPLER_BYPASS` (libllama's sampler applies
  the same rule to its call count), `SKIP_SUM`, `SKIP_MAX`, `SKIP_MASS`, `HEAD2`. An unknown name prints one
  `c2-ab: error=...` line and runs without A/B.

| knob | read | meaning |
|---|---|---|
| `VOLLIE_SKIP_SUM`, `_MAX`, `_MASS`, `_MINK`, `_STATS` (0001) | per graph | as above |
| `VOLLIE_SKIP_WIRE` (0008) | **load time** (graph build) | 1 = libllama wires the router weights into the expert ops even with every skip threshold unset, so a threshold set later (or by A/B) acts. llama.cpp reuses a built graph while the shapes match, so the wiring of the first build can persist: set it before the context's first decode. Wiring is automatic when a threshold is set or `VOLLIE_AB` names a `SKIP_` knob at the build. Results unchanged with the thresholds unset |
| `VOLLIE_TINY_OP`, `VOLLIE_GRAPH_STATS` (0002), `VOLLIE_Q3K_X4` (0003), `VOLLIE_MMID_NB` (0004), `VOLLIE_MMID_FUSE` (0005), `VOLLIE_EXPERT_IO` (0006) | per graph | as above |
| `VOLLIE_SAMPLER_BYPASS` (0007) | per sampler call | as above |
| `VOLLIE_PROF`, `VOLLIE_AB`, `VOLLIE_AB_WARM`, `VOLLIE_AB_BLOCK`, `VOLLIE_AB_BAAB`, `VOLLIE_RUN`, `VOLLIE_LINES` (0008) | per graph | as above; `VOLLIE_RUN`: up to 32 characters, others become `_` |
| `VOLLIE_HEAD2`, `VOLLIE_HEAD2_BATCH`, `VOLLIE_HEAD2_CHECK` (0009) | per graph | the screen copy is made at the first head op that needs it and kept until exit (57.9 MB anon), also after the knob is unset |

## 0009-head2-untied.patch: two-stage output head for an untied Q6_K head (numerics change when on)

`ggml-cpu.c` (`ggml_c2_head2`, called from `ggml_compute_forward` before the extra-buffer dispatch) and `repack.cpp`
(helpers). Campaign 1's `GGML_HEAD2` with `GGML_HEAD2_Q=q4_0`, for the untied head `output.weight` (Q6_K) and for both
layouts the kit can load it in: plain rows (`-nr`) and this module's `q6_K_8x4` / `q6_K_8x8` repack (repack on, how the
kit runs OLMoE). At one token:

1. **Screen.** Every row from a Q4_0 copy of the head against a Q8_0 activation. The copy is the head's own rows
   (repack on: un-interleaved from the `q6_K_8x*` copy in the `CPU_REPACK` buffer) dequantised and requantised to Q4_0
   without an imatrix, interleaved four rows at a time into `block_q4_0x4` (4x4 with dotprod, the SD888; 4x8 with i8mm;
   plain Q4_0 rows on CPUs with neither), in anonymous memory (malloc) owned by the CPU module: 50,304 × 2,048 → 57.9 MB
   (*estimated*). Made once, threads split, at the first head op; repack on and off give the same copy.
2. **Select** the K rows with the largest screen logits (ties to the lower id, the same set for any thread count),
   campaign 1's histogram select.
3. **Exact rows** with the stock kernel on the stock Q8_K activation: plain rows with `vec_dot_q6_K_q8_K` row by row (the
   stock `-nr` decode head); repacked rows with the stock `q6_K_8x*` gemv on every touched 8-row group, so all 8 rows of
   such a group come out exact. These rows equal the stock head bit for bit at one token. The other rows keep the screen
   logit.

| knob | value | meaning |
|---|---|---|
| `VOLLIE_HEAD2` | K rows, unset / 0 = off | the two-stage head with K exact rows (K < rows). Numerics change |
| `VOLLIE_HEAD2_BATCH` | 1 = on (test only) | also at more than one column (every column of a batch), so `llama-perplexity -b 512` runs it. There the exact rows are the decode gemv's values, not the stock `-b 512` gemm's |
| `VOLLIE_HEAD2_CHECK` | 1 = on (host test) | thread 0 also computes the whole head with the stock kernel per column and counts exact rows that differ (`mismatches`, must be 0) and columns whose true argmax is in the screened set (`top1_in_set`); also a fingerprint of the copy in `c2-head-build:` |

Full-set PPL: `tools/c2_head2_sweep.sh`. Not handled (stock path): a tied head (`token_embd`), another head type, other
extra buffer types, rows not a multiple of 8.

## Not in the series

Campaign 1's head2 low-rank screen and tied-head path, and its other knobs (anon cache, Q4 expert copies, touch, chunk2,
router cast, SSM/conv in place) are not ported. The profiler and A/B are 0008, the untied head2 is 0009.
