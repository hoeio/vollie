# Patches

Kept as patches against pinned upstream versions, not as forks, so they stay reviewable and can
be rebased when upstream moves.

## llama.cpp — `llama.cpp/0001-cpu-runtime.patch` (base: `llama.cpp/BASE`)

Touches only the CPU backend (`ggml/src/ggml-cpu/ggml-cpu.c`, `arch/arm/quants.c`). Every feature
is off unless its environment variable is set (read before each graph), and none changes
results: each was checked bit-identical with `llama-perplexity` at one-token, 32-token and
512-token batches. See `docs/runtime-knobs.md` for the knobs.

| feature | knob | why |
|---|---|---|
| small ops on one thread, syncs between them skipped | `GGML_TINY_OP` | ~2,000 ops/token each followed by a sync of all threads; on an SD888 the syncs were half of each token |
| expert prefetch | `GGML_EXPERT_IO` bit 1 | MoE experts read through mmap fault in one page at a time on compute threads; the routed experts' slices are requested together instead |
| no read-around on expert tensors | `GGML_EXPERT_IO` bit 2 | measured worse on the SD888 (more, smaller faults); kept for other devices |
| next-layer expert guess | `GGML_EXPERT_IO` bit 4 (+8: two more) | a helper thread applies the next layer's router to this layer's input and prefetches a layer early; 82-85% right |
| vectorized Q3_K dot (NEON) | `GGML_Q3K_VEC` (default on) | sub-block sums kept in vectors instead of 16 serial horizontal adds per block |
| per-op profiler | `GGML_CPU_PROF=1` | µs computing and waiting per op type, one stderr line per 32 decode tokens |
| in-generation A/B | `GGML_CPU_AB=q3k` or `tiny:<n>` | alternates a compute setting every token so heat and clocks hit both sides alike |

A race in an earlier version (a skipped sync before a fused RMS_NORM+MUL) produced garbage for
prompt batches under ~42 tokens; it is fixed here, and `-ub 32` is now part of every check.

## llamadart — `llamadart/0001-setthreads-extra-buffers.patch` (base: `llamadart/BASE`)

For apps using the [llamadart](https://pub.dev/packages/llamadart) Flutter bindings:
- `ModelParams.useExtraBuffers` → `llama_model_params.use_extra_bufts`: lets an app turn llama.cpp's
  weight repacking off on phones where the repacked copy doesn't fit in RAM.
- `LlamaEngine.setThreads(threads, threadsBatch)` → `llama_set_n_threads`: change threads on a loaded
  model instead of reloading it (used for heat throttling and fast thread sweeps).
