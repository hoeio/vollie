# Patches

Kept as patches against pinned upstream versions, not as forks, so they stay reviewable and can
be rebased when upstream moves.

## llama.cpp — `llama.cpp/0001-cpu-runtime.patch` (base: `llama.cpp/BASE`)

Touches only the CPU backend (`ggml/src/ggml-cpu/ggml-cpu.c`, `arch/arm/quants.c`, `repack.cpp`,
`kleidiai.cpp`, `quants.h`, and the new header `vollie-release.h`). Every feature is off unless its environment variable is set (read before each graph) —
except `GGML_MMID` (default 3) and `GGML_Q3K_X4` (default 1) since round 2, both measured −8.3 %/token
on the device by in-generation A/B (`reports/03_final.md` R18) — and none changes
results: each was checked bit-identical with `llama-perplexity` at one-token, 32-token and
512-token batches. See `docs/runtime-knobs.md` for the knobs.

| feature | knob | why |
|---|---|---|
| small ops on one thread, syncs between them skipped | `GGML_TINY_OP` | ~2,000 ops/token each followed by a sync of all threads; on an SD888 the syncs were half of each token |
| expert prefetch | `GGML_EXPERT_IO` bit 1 | MoE experts read through mmap fault in one page at a time on compute threads; the routed experts' slices are requested together instead |
| no read-around on expert tensors | `GGML_EXPERT_IO` bit 2 | measured worse on the SD888 (more, smaller faults); kept for other devices |
| next-layer expert guess | `GGML_EXPERT_IO` bit 4 (+8: two more) | a helper thread applies the next layer's router to this layer's input and prefetches a layer early; 82-85% right |
| vectorized Q3_K dot (NEON) | `GGML_Q3K_VEC` (default on; empty value = on since round 5, was read as 0; harmless, the x4 kernel bypasses it) | sub-block sums kept in vectors instead of 16 serial horizontal adds per block |
| per-op profiler | `GGML_CPU_PROF=1` | µs computing and waiting per op type, one stderr line per 32 decode tokens; the profile window now resets at every prompt (> 8 tokens), so per-run windows no longer inherit the previous run's graphs |
| in-generation A/B | `GGML_CPU_AB=q3k` or `tiny:<n>` | alternates a compute setting every token so heat and clocks hit both sides alike |
| expert-budget rules (R22) | `GGML_EXPERT_TOPP` / `GGML_EXPERT_GAP` / `GGML_EXPERT_MINK` | router-weight adaptive expert count (cumulative-mass, gap, floor); a `k_avg` exit counter reports the mean experts used |
| independent-chain Q3_K x4 dot | `GGML_Q3K_ILP` (default on since round 4: empty = unset = on, `0` = off) | several accumulator chains in the 4-row Q3_K dot; bit-identical |
| prefetch in the Q3_K dot | `GGML_Q3K_PF` | software prefetch of the next block; bit-identical (measured on device R23: no gain; kept in the patch, closed) |
| anon-memory hot-expert cache | `GGML_EXPERT_ANON=<N>` | window-LFU copy of the hottest experts into anonymous memory; same bytes (measured on device R23: no gain; kept in the patch, closed) |
| in-place SSM state | `GGML_SSM_INPLACE` (default **on** since round 5: empty = unset = on, `0` = off; bit-identical) | for single-sequence decode the SSM_SCAN writes its new state straight into the recurrent-cache slot and the state CPY node is dropped from compute; matched per graph only when sizes/offsets match exactly and nothing else touches the cache between scan and copy; 36 Mamba layers |
| conv state in place | `GGML_CONV_RING` (default off; bit-identical) | the one-token `SSM_CONV` reads the old conv state and x where they already are and shifts the state into the cache itself; the 36 `CONCAT` and 36 conv `CPY` ops are dropped (nodes 1911 → 1839); single sequence, K = 1, d_conv = 4 only, else stock path | host −1.2 ms/token on its own ops (noisy) |
| more A/B sides | `GGML_CPU_AB=` `EXPERT_SKIP` / `Q3K_ILP` / `Q3K_PF` / `EXPERT_ANON` / `SSM_INPLACE` / `CONV_RING` | the in-generation A/B now alternates these too. **ABBA since round 5**: `GGML_CPU_AB_WARM` (default 16) warm-up tokens, then `GGML_CPU_AB_BLOCK` (default 8)-token blocks in A B B A order, one `cpu-prof ab= blk=` line per block; `knob:` mode only; the parser reports B − A ± 1σ over post-warm-up blocks |
| release helper | `ggml/src/ggml-cpu/vollie-release.h` (new header) | the `GGML_REPACK_RELEASE=1` code: after a tensor is repacked, `madvise(MADV_PAGEOUT)` (fallbacks `MADV_COLD` / `MADV_DONTNEED`) on the mmap source pages so the page cache can hold experts instead; included by `ggml-cpu.c` and `repack.cpp`; data and results unchanged |
| no barrier + dynamic chunks in `MUL_MAT_ID`, 4-row Q3_K, expert skip, helper affinity, source-page release, barrier bench, generic A/B | see `docs/runtime-knobs.md` (round 1, R17/R18) | the round-1 campaign knobs; `GGML_CPU_PROF=2` profiles 2-8-token graphs (speculative verify batches), now main-context only |

Expert-skip threshold edge under the +2 % budget: 118 permille (full-set wikitext-2 9.7952, +1.98 %; 119 fails) — round 5.

`0001` is produced with `git -C lab/llama.cpp-tiny diff HEAD -- ggml/src/ggml-cpu` against base `620d2225c`.

A race in an earlier version (a skipped sync before a fused RMS_NORM+MUL) produced garbage for
prompt batches under ~42 tokens; it is fixed here, and `-ub 32` is now part of every check.

## llama.cpp — `llama.cpp/0002-granite-rollback.patch` (base: `llama.cpp/BASE`)

One line in `src/llama-arch.cpp`: `LLM_ARCH_GRANITE_HYBRID` joins `llm_arch_supports_rs_rollback()`
(it has the same Mamba-2 layer as Nemotron-H, which is listed). Without it `llama_context` clamps
`n_rs_seq` to 0, no recurrent-state snapshots are reserved, and a hybrid target cannot roll a rejected
draft back — draft-model speculation fails on the first rejection (round 1, R18). This patch lives in
`libllama.so`, not the CPU module, so an app on a prebuilt llama.cpp bundle has to rebuild that library
(`android/README.md`).

## llamadart — `llamadart/0001-setthreads-extra-buffers.patch` (base: `llamadart/BASE`)

For apps using the [llamadart](https://pub.dev/packages/llamadart) Flutter bindings:
- `ModelParams.useExtraBuffers` → `llama_model_params.use_extra_bufts`: lets an app turn llama.cpp's
  weight repacking off on phones where the repacked copy doesn't fit in RAM.
- `LlamaEngine.setThreads(threads, threadsBatch)` → `llama_set_n_threads`: change threads on a loaded
  model instead of reloading it (used for heat throttling and fast thread sweeps).
- `ModelParams.speculativeRollbackTokenMax` accepted on recurrent/hybrid models (→ `n_rs_seq`), and a
  draft-model run on such a model throws before the first draft when `llama_n_rs_seq` is 0 (the
  library lacks `0002-granite-rollback.patch`) instead of failing on the first rejected draft.
- `hook/build.dart`: a present `third_party/bin/<bundle>/` directory is used as the native bundle, and
  it is checked before the cached-bundle return (upstream only honours it behind an env var that does
  not reach a hook run under the Gradle daemon). That is how the rebuilt `libllama.so` ships.

## Experiments only — `llama.cpp/experiments/lab-v0.8.0.patch` (base: `experiments/BASE`, v0.8.0)

Lab probes behind results in `reports/`, never shipped: per-layer expert counts for Granite (used
by `tools/layer_k_sweep.sh`), `LLAMA_NO_REPACK` (keep chosen tensors out of the repacked copy; it
showed 60% of the repack gain comes from the dense weights alone, which led to the D-series
files), and a llama-bench switch for llama.cpp's per-call threadpool (no difference on host).
