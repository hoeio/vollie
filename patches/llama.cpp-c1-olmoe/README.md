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
