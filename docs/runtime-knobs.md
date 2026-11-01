# Runtime knobs of the patched CPU module

Environment variables read by `ggml_graph_compute` before every graph, so an app can change them
between generations (`setenv`; `runtime/dart` has setters) without reloading the model. Unset or 0
means stock behaviour. None changes results; each was checked bit-identical with
`llama-perplexity` at `-ub 1`, `-ub 32` and `-b 512`.

| variable | values | effect | measured |
|---|---|---|---|
| `GGML_TINY_OP` | elements, e.g. `65536` | ops with at most this many elements (in output and inputs; GET_ROWS by output) run on thread 0 alone; syncs between consecutive such ops are skipped. Mat-muls, SSM scan, flash attention and ops with internal syncs never count as small | SD888 2 threads: 0 → 3.2, 65536 → 7.0 tok/s |
| `GGML_EXPERT_IO` | bit mask | MoE expert reads through mmap: **1** `MADV_WILLNEED` the routed experts of a layer (gate/up/down together) when its first `MUL_MAT_ID` runs; **2** `MADV_RANDOM` on expert tensors (no read-around); **4** helper thread guesses the next layer's experts (its router on this layer's FFN input) and prefetches them; **8** guess 2 more than the model routes to | SD888 3 threads: 0 → 6.1, 1 → 11.8 tok/s; 5 → 12.0 vs 1 → 10.6 in another run; 2 lost; 8 inconclusive |
| `GGML_Q3K_VEC` | `1` (default) / `0` | NEON Q3_K dot with vectorized sub-block sums / upstream loop | M4 A/B: expert mat-muls −4..−8% |
| `GGML_CPU_PROF` | `1` | thread 0 times every op of one-token graphs: per op type, count, µs computing, µs waiting at the sync after it; one `cpu-prof` line per 32 graphs on stderr, with `pred=` (share of real experts that were guessed) | |
| `GGML_CPU_AB` | `q3k`, `tiny:<n>` | alternates the setting every one-token graph (A = normal, B = upstream Q3_K loop / threshold `<n>`); with `GGML_CPU_PROF` the stats are split into `ab=A` and `ab=B` lines over the same 64 tokens. Not for `GGML_EXPERT_IO`: one token's prefetch warms the cache for the next | |

Reading a `cpu-prof` line:

```
cpu-prof graphs=32 threads=3 tiny=65536 eio=5 q3k=1 pred=85% us/token=98176 compute=74768 wait=23016 nodes=1963 barriers=818 | MUL_MAT_ID 120:34214+16861 MUL_MAT 265:28518+3364 ...
```

`us/token` is the whole graph; `compute` + `wait` is thread 0's time in ops and at syncs. After `|`,
per op type, sorted by cost: `count per token : µs computing + µs waiting`. A page fault inside a
kernel shows as computing time on the thread that took it; pair the line with major-fault counts
(`/proc/self/stat`) to tell flash reads from arithmetic.
