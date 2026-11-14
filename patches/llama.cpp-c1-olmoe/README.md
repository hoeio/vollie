# Campaign 2 patches on top of campaign 1's runtime (OLMoE)

Owner's instruction of 2026-11-14: campaign 2 starts from campaign 1's shipped state. These patches apply **after**
`patches/llama.cpp/0001-cpu-runtime.patch`, `0002-granite-rollback.patch` and `0003-sampler-bypass.patch` (base
`620d2225c`, see `patches/llama.cpp/BASE`). Those three files are never edited. Each patch here is one lever, off unless
its environment variable is set.

```sh
git -C <tree> checkout 620d2225c
for p in patches/llama.cpp/000[123]-*.patch patches/llama.cpp-c1-olmoe/0*.patch; do git -C <tree> apply "$PWD/$p"; done
git -C <tree> add -N ggml/src/ggml-cpu/vollie-release.h   # 0001's new file, so the build script's file check sees it
tools/c2_build_kit.sh <tree> lab/app-c2 patches/llama.cpp patches/llama.cpp-c1-olmoe
```

## 0001-keep-warm.patch

`GGML_KEEP_WARM=<period in seconds>` (unset, empty or 0 = off). File: `ggml/src/ggml-cpu/ggml-cpu.c`, +196 lines.
Design and host checks: `campaigns/02-25toks-5pct/reports/memo-17-kit-r20.md` §4.

- **What it does.** While no graph has run for at least 1 s, a helper thread wakes every `<period>` s and walks the
  routed-expert weights (every `MUL_MAT_ID` weight whose name contains `_exps`, registered from the graphs the knob saw;
  OLMoE: 48 tensors, 2,769 MB). Per step of 1,024 pages it calls `mincore`, then reads one byte of every **resident**
  page through `process_vm_readv` on Linux/Android (`vm_read_overwrite` on macOS). The kernel then sees those page-cache
  pages as referenced and keeps them through an idle wait. Pages that are already gone are not read back: no I/O.
- **Safety.** Both reads return an error instead of a signal on unmapped memory, so a model freed during a pass costs
  nothing; `mincore` fails on an unmapped range and the pass skips it. The helper never calls `getenv` (the app sets
  environment variables from another thread); the period is copied from the environment at each graph start.
- **Stops when a generation starts.** Every graph start bumps a counter and sets a busy flag; the helper checks both
  before every 1,024-page step and abandons the pass (`aborted=1`).
- **Cost.** SCHED_BATCH, nice 19, pinned to the CPUs with the lowest `cpuinfo_max_freq` (the SD888's A55 cores 0-3),
  never to the graph thread's big cores. One line per pass on stderr:
  `c2-keepwarm pass=<n> period_s=<p> ranges=<r> mb=<walked> resident_mb=<resident> touched=<pages read> wall_us=<µs> cpu_us=<thread CPU µs> aborted=<0|1>`.
  Duty cycle = `cpu_us` / (`period_s` × 10⁶). *Host* (M4 Pro, 16 KB pages, OLMoE fully resident): 0.30 to 0.52 s CPU
  per pass. On the phone (4 KB pages, about 4× the pages, one system call per 1,024 pages, A55): unknown, *estimated*
  0.2 to 0.7 s per pass, so 2 to 7 % of one A55 at a 10 s period. Kit r20 measures it.
- **Off = nothing.** With the variable unset and never set, the only added work is one `getenv` per graph; no thread
  exists.
- **Identity** (*host*, `olmoe-Mi`, 4 chunks, `-t 4`): per-chunk PPL identical with the knob unset and with
  `GGML_KEEP_WARM=1` at `-b 512`, `-ub 32` and `-ub 1`, and equal to the unpatched base build. Greedy official prompt
  (32 tokens) identical across unset, keep-warm, shipping settings, shipping + keep-warm, `EXPERT_IO=5`, every knob off,
  and the profiled variants (31 per-token lines each). It is advice only: the helper reads weights and writes nothing
  the graph reads.
- **Not verified:** anything on Android (whether `process_vm_readv` is allowed for the app, how fast a pass runs on the
  A55, whether the kernel keeps the touched pages through a 90 s wait).

## 0002-prefill-cold.patch

`GGML_PREFILL_COLD=<tau>` (unset, empty or 0 = off) and `GGML_C2_IO=1` (unset = off), plus `GGML_PREFILL_COLD_LOG=1` for
host checks. Files: `ggml/src/ggml-cpu/ggml-cpu.c` (+218 lines), `ggml/src/ggml-cpu/ggml-cpu.cpp` (+3, the CPU
backend's free flushes the open `c2-io` line). Design, tau and host checks: `campaigns/02-25toks-5pct/reports/memo-19-kit-r21.md` §1.

- **What it does.** In a batch graph (`n_tokens > 1`: a prompt, a perplexity chunk) thread 0 counts, at each layer's
  first `MUL_MAT_ID` with more than one row, the rows the routing sends to each expert. After that layer has run (at the
  next layer's first `MUL_MAT_ID`, past the barrier before it, and at the graph's end) every slice `e` of the
  `MUL_MAT_ID` weights sharing those ids that the prompt routed 1 to tau-1 times is advised, page-aligned inward. Only
  weights in a `CPU_Mapped` buffer (the model file's mapping) are advised, never anonymous memory such as a repacked
  copy. Slices routed 0 times are not touched. No architecture names: `granitehybrid` and `olmoe` take the same path.
- **Advice**, probed once on the first slice: `MADV_COLD` (20, Linux 5.4+), then `MADV_PAGEOUT` (21), then none. The
  probe prints `c2-pcold-advice advice=<cold|pageout|none> errno_cold=<e> errno_pageout=<e> page=<bytes>`; macOS has
  neither advice, so the host always shows `none` (the slices are counted, nothing is issued).
- **Lines.** One per batch graph: `c2-pcold tau=<t> advice=<a> tokens=<n> layers=<l> routed_slices=<r> marked_slices=<m>
  mb=<MB advised> fails=<madvise errors> us=<thread-0 µs in the advice>`. With `GGML_C2_IO=1`, one per prompt (printed
  when the next prompt starts or the backend is freed): `c2-io run=<VOLLIE_RUN> prompt_graphs=<p> decode_graphs=<d>
  us=a/b/c read_bytes=a/b/c rchar=a/b/c majflt=a/b/c minflt=a/b/c`, sampled at the prompt's first batch graph (a), the
  first one-token graph after it (b) and after the last graph (c). Prefill = b - a, decode = c - b; -1 where
  `/proc/self/io` is unreadable (macOS). A `-ub 1` perplexity run has no batch graph, so it prints neither line.
- **Identity** (*host*, both models, 4 chunks, `-t 4`): per-chunk PPL and the md5 of the `--kl-divergence-base` file
  equal with the knobs unset and with `GGML_PREFILL_COLD=8 GGML_C2_IO=1` at `-b 512`, `-ub 32` and `-ub 1`; greedy
  official text (32 tokens) equal across unset, shipping, shipping + `C2_IO`, + `PREFILL_COLD` at tau 4, 8, 16, + the
  slice log and the profiled variants (memo-19 §1.4).
- **Known limit (not met by kit r21):** a batch of 2 to 7 tokens routes every slice fewer than 8 times, so a short
  follow-up message would mark its whole routed set. A shipping version needs a floor on the batch size.
- **Not verified:** anything on Android: whether the phone's kernel has `MADV_COLD` (the reports do not give its version;
  kit r21 records `Platform.operatingSystemVersion`), whether the advice changes what the phone reclaims in a wait.
