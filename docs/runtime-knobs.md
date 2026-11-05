# Runtime knobs of the patched CPU module

Environment variables read by `ggml_graph_compute` before every graph, so an app can change them
between generations (`setenv`; `runtime/dart` has setters) without reloading the model. Unset or 0
means stock behaviour, except for the knobs marked default on below (`GGML_MMID`, `GGML_Q3K_X4`, `GGML_Q3K_ILP`, `GGML_SSM_INPLACE`, `GGML_CONV_RING`), where `0` switches them off. None changes results; each was checked bit-identical with
`llama-perplexity` at `-ub 1`, `-ub 32` and `-b 512`.

| variable | values | effect | measured |
|---|---|---|---|
| `GGML_TINY_OP` | elements, e.g. `65536` | ops with at most this many elements (in output and inputs; GET_ROWS by output) run on thread 0 alone; syncs between consecutive such ops are skipped. Mat-muls, SSM scan, flash attention and ops with internal syncs never count as small | SD888 2 threads: 0 → 3.2, 65536 → 7.0 tok/s |
| `GGML_EXPERT_IO` | bit mask | MoE expert reads through mmap: **1** `MADV_WILLNEED` the routed experts of a layer (gate/up/down together) when its first `MUL_MAT_ID` runs; **2** `MADV_RANDOM` on expert tensors (no read-around); **4** helper thread guesses the next layer's experts (its router on this layer's FFN input) and prefetches them; **8** guess 2 more than the model routes to | SD888 3 threads: 0 → 6.1, 1 → 11.8 tok/s; 5 → 12.0 vs 1 → 10.6 in another run; 2 lost; 8 inconclusive |
| `GGML_Q3K_VEC` | `1` (default) / `0`; empty value = on since round 5 (was read as 0; harmless, the x4 kernel bypasses it) | NEON Q3_K dot with vectorized sub-block sums / upstream loop | M4 A/B: expert mat-muls −4..−8% |
| `GGML_CPU_PROF` | `1` / `2` | **1**: thread 0 times every op of one-token graphs (**2**: of 2-8-token graphs, i.e. speculative verify batches; `us/token` is then µs per graph and `tokens=` their total): per op type, count, µs computing, µs waiting at the sync after it; one `cpu-prof` line per 32 graphs on stderr, with `pred=` (share of real experts that were guessed) | |
| `GGML_CPU_AB` | `q3k`, `tiny:<n>` | alternates the setting every one-token graph (A = normal, B = upstream Q3_K loop / threshold `<n>`); with `GGML_CPU_PROF` the stats are split into `ab=A` and `ab=B` lines over the same 64 tokens. Not for `GGML_EXPERT_IO`: one token's prefetch warms the cache for the next | |

Reading a `cpu-prof` line:

```
cpu-prof graphs=32 threads=3 tiny=65536 eio=5 q3k=1 pred=85% us/token=98176 compute=74768 wait=23016 nodes=1963 barriers=818 | MUL_MAT_ID 120:34214+16861 MUL_MAT 265:28518+3364 ...
```

`us/token` is the whole graph; `compute` + `wait` is thread 0's time in ops and at syncs. After `|`,
per op type, sorted by cost: `count per token : µs computing + µs waiting`. A page fault inside a
kernel shows as computing time on the thread that took it; pair the line with major-fault counts
(`/proc/self/stat`) to tell flash reads from arithmetic.

## Round-1 knobs (defaults as stated per knob)

Introduced in app 2.2.6+29 (research kit `r1`; see the campaign report). Off unless marked default on; checked bit-identical at `-ub 1`, `-ub 32`, `-b 512` unless marked otherwise.

| variable | values | effect | status |
|---|---|---|---|
| `GGML_MMID` | bit mask, **default 3** (`''` = unset) | one-token `MUL_MAT_ID`: **1** no barrier inside the op (each thread quantizes the activation row and builds the expert map itself); **2** 32-row chunks claimed dynamically through per-expert counters that thread 0 resets before the preceding barrier (needs 1) | bit-identical; device: with `GGML_Q3K_X4`, −8.3 %/token (R18) |
| `GGML_Q3K_X4` | `1` (**default**), `0` | Q3_K expert rows four at a time sharing the activation loads (`ggml_vec_dot_q3_K_q8_K_x4`) | bit-identical; device: with `GGML_MMID=3`, −8.3 %/token (R18) |
| `GGML_EXPERT_SKIP` | permille | an expert whose normalized routing weight is below this is not computed (its rows are zero); `GGML_EXPERT_SKIP_RENORM=1` rescales the remaining weights | **not bit-exact**; host wikitext-2 60 chunks: stock 9.870, 30 → 9.874, 60 → 9.845, 100 → 9.916, 150 (renorm) → 13.63; only ~1% of routed experts fall under 0.06 |
| `GGML_HELPER_CPUS` | `0-3`, `0,1,2,3` | cores for the next-layer expert-guess helper thread (re-read per job; `''` = every core) | device R18: 4 threads + helper on an A55 no better than 3 threads; closed |
| `GGML_REPACK_RELEASE` | `1` (at load) | after a tensor is repacked, `MADV_PAGEOUT` (fallback `MADV_COLD`/`DONTNEED`) its memory-mapped source pages so the page cache can hold experts instead | device R18: inconclusive (noise) |
| `GGML_BARRIER_BENCH` | n | once per distinct value: n empty barriers before the first op of the next one-token graph; prints `cpu-bench barriers=n threads=t us/barrier=x` | measurement |
| `GGML_CPU_AB` | `knob:NAME:A:B[,NAME:A:B…]` | generic in-generation A/B over TINY_OP, MMID, Q3K_VEC, Q3K_X4, Q3K_ILP, SSM_INPLACE, CONV_RING, EXPERT_SKIP. **ABBA since round 5** (`knob:` mode only): warm-up tokens, then blocks in A B B A order, one `cpu-prof ab= blk=` line per block; the parser reports B − A ± 1σ over post-warm-up blocks | measurement |
| `GGML_CPU_AB_WARM` | tokens, default 16 | warm-up tokens before the first ABBA block | measurement |
| `GGML_CPU_AB_BLOCK` | tokens, default 8 | tokens per ABBA block | measurement |
| `GGML_SSM_INPLACE` | **default on** since round 5 (empty = unset = on), `0` = off | the one-token `SSM_SCAN` writes its new state straight into the recurrent-cache slot; the state `CPY` node is dropped | bit-identical |
| `GGML_CONV_RING` | **default on** since round 6 (empty = unset = on), `0` = off | the one-token `SSM_CONV` reads the old conv state and x in place and shifts the state into the cache itself; the 36 `CONCAT` and 36 conv `CPY` ops are dropped (nodes 1911 → 1839); single sequence, K = 1, d_conv = 4 only, else stock path | bit-identical; host −1.2 ms/token on its own ops (noisy) |
| `GGML_HEAD2` | rows `M` (0 / unset = off) | two-stage output head: a cheap copy (type `GGML_HEAD2_Q`) screens all rows, the top-`M` rows are recomputed exactly from the stock head; default off, bit-identical when off | **not bit-exact**; host wikitext-2 PPL +0.19 % at `GGML_HEAD2_Q=q2_k`, `M`=8,192 (9.6237 vs 9.6051), argmax-identical; `reports/memos/head2stage_fix.md` |
| `GGML_HEAD2_Q` | `q2_k` (default) / `q3_k` / `q4_0` | type of the screening copy made at the first head op | only with `GGML_HEAD2`; `reports/memos/head2stage_fix.md` |
| `GGML_HEAD2_BATCH` | `1` | test only (`llama-perplexity`): the same two-stage head per column at batch > 1, so the full-set gate can be run | only with `GGML_HEAD2`; `reports/memos/head2stage_fix.md` |
| `GGML_EXPERT_LOG` | `1` | lab: prints the routed expert ids per layer per token (`eid …`) | measurement |

`cpu-prof` lines also carry `head2= head= majflt= minflt=` (head2: ops on the two-stage head; head: head time; majflt / minflt: `getrusage` page faults per profiling window, and per side of an ABBA block in `ab=` lines).

`cpu-prof` lines now carry `x4= mmid= skip=<t>/<share>% mmid_thr=<µs per thread inside MUL_MAT_ID>`.

llamadart (`patches/llamadart`): `ModelParams.speculativeRollbackTokenMax` is now accepted for recurrent/hybrid
models (llama.cpp 620d2225c keeps `n_rs_seq` per-token recurrent snapshots and `llama_memory_seq_rm` rolls back within
them), which lets `SpeculativeDecodingConfig.draftSimple` run on Granite 4.0 H. Each slot costs ~58 MB on Granite 4.0 H Tiny.
