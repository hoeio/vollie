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

## 0003-c2-expert-skip.patch (numerics change when on)

`VOLLIE_SKIP_SUM`, `VOLLIE_SKIP_MAX`, `VOLLIE_SKIP_MASS` (permille, unset / 0 = off), `VOLLIE_SKIP_MINK` (default 1),
`VOLLIE_SKIP_STATS=1`. File: `ggml/src/ggml-cpu/ggml-cpu.c`. The scale-free skip rules of
`patches/llama.cpp-c2/0001` (same meaning, same float arithmetic; table in `patches/llama.cpp-c2/README.md`), written
into campaign 1's own skip mask (`ggml_cpu_skip_prepare`) instead of a second mechanism. So they act wherever campaign
1's `GGML_EXPERT_SKIP` acts: the stock `MUL_MAT_ID` (prefill, `-b 512`), the one-token no-barrier path (`GGML_MMID`), the
fused gate + up + SwiGLU op (`GGML_MMID_FUSE`, on by default) and the expert read-ahead (`GGML_EXPERT_IO` does not advise
skipped slices). The mixture weights are found as campaign 1 finds them (`src[1]` of the `ffn_moe_weighted` MUL after the
op), so libllama is not patched and no architecture name is used. Experts in a repacked (extra-buffer) type still run in
full; both served files have Q3_K experts, which this module does not repack.

- **Beside campaign 1's knobs.** `GGML_EXPERT_SKIP`, `GGML_EXPERT_TOPP`, `GGML_EXPERT_GAP`, `GGML_EXPERT_MINK` and
  `GGML_EXPERT_SKIP_RENORM` are unchanged. If knobs of both sets are set, a slot is skipped when **any** rule says so (the
  rule campaign 1 already used between its own knobs); each set keeps its own MINK. Campaign 1's absolute thresholds and
  the c2 relative ones agree only where the weights are renormalised (Granite: `GGML_EXPERT_SKIP=t` is
  `VOLLIE_SKIP_SUM=t` up to float rounding of S); on OLMoE (`norm_w = false`) only the c2 rules are scale-free.
- **Knobs per graph** (campaign 1's place, the calling thread). A/B: `GGML_CPU_AB=knob:SKIP_MASS:0:900` (also
  `SKIP_SUM`, `SKIP_MAX`); the per-side skip counts are campaign 1's `cpu-prof` fields.
- **`VOLLIE_SKIP_STATS=1`:** one `c2-skip:` line at exit on stderr with 0001's fields (plus `c1_skip`, `c1_topp`,
  `c1_gap`), counting the skips of every rule, campaign 1's included: tokens, pairs, % pairs skipped, % expert bytes
  skipped, MB per token selected and skipped, % of the bytes a batch would have read that it did not read.

## 0004-c2-head2-untied.patch (numerics change when on)

`VOLLIE_HEAD2=<K>` (unset / 0 = off), `VOLLIE_HEAD2_BATCH=1` (test only), `VOLLIE_HEAD2_CHECK=1` (host test). Files:
`ggml/src/ggml-cpu/ggml-cpu.c`, `ggml/src/ggml-cpu/repack.cpp`. `patches/llama.cpp-c2/0009`'s two-stage head for an untied
Q6_K head (`output.weight`, plain rows or this module's `q6_K_8x4` / `q6_K_8x8` repack), unchanged in its arithmetic; the
Q4_0 screen helpers and the top-K select are campaign 1's (`ggml_cpu_head2_q4_0_*`, `ggml_cpu_head2_topm`), which are the
same code. Only the three Q6_K helpers are new in `repack.cpp`. Method and knob table: `patches/llama.cpp-c2/README.md`,
section 0009.

- **Beside campaign 1's `GGML_HEAD2`.** Campaign 1's head keeps the stock path on a repacked untied head (OLMoE with repack
  on, the kit's setting), so it does nothing for OLMoE there; this patch fills that case. The hook runs before
  `GGML_HEAD2`'s and takes `output.weight` only: if both knobs are set, `VOLLIE_HEAD2` wins on an untied head and
  `GGML_HEAD2` keeps a tied head (`token_embd.weight`, Granite), which `VOLLIE_HEAD2` never takes.
- **Knobs per graph.** A/B: `GGML_CPU_AB=knob:C2_HEAD2:0:1024`; the copy is then made on the first graph. With the knob
  and the A/B both off, the copy (`copy_mb=58.0` on OLMoE, *host* `c2-head-build:` line) is freed at the next graph (campaign 1's
  rule for its own copy); the small select scratch stays.
- **Lines** on stderr: `c2-head-build:` once per copy, `c2-head:` at exit (0009's fields). 0008's per-prompt
  `c2-head-run:` line is not ported (campaign 1's runtime has no 0008).

## Host checks of 0003 and 0004 (2026-11-14/15, *host*, M4 Pro, `-t 8`, build `armv8.2-a+dotprod+fp16`, repack on)

- **Identity, knobs unset:** per-chunk PPL of 8 chunks equal between the tree with 0001-0002 and the tree with
  0001-0004 at `-b 512`, `-ub 32` and `-ub 1`, on `olmoe-Mi` and `granite-Mi`; at `-b 512` both also equal the first 8
  chunks of R50's unpatched-base logs.
- **Same decisions as the c2 series:** `VOLLIE_SKIP_MASS=900` gives the c2 build's per-chunk values and `c2-skip:`
  counters (4 chunks `-b 512`, 2 chunks `-ub 1`), and on the full set at `-b 512` R51's 12.1031 ± 0.10045 with 13.39 %
  of expert bytes skipped. `VOLLIE_HEAD2=1024` (+ `_CHECK`) gives the c2 build's values, the same copy fingerprint and
  `mismatches=0` (2 chunks, `-b 512` with `_BATCH` and `-ub 1`); with `MASS=900` on the full set at `-b 512` R53's 12.1053.
- The `-ub 1` runs go through campaign 1's fused one-token expert op, the phone's path; the counters show the skip there
  (`batch_bytes_unread` equals the bytes skipped, so every skipped expert slice is left unread).

## 0005-kit-lines.patch (counters only, bit-identical)

File: `ggml/src/ggml-cpu/ggml-cpu.c`. Kit r22's per-prompt lines (`campaigns/02-25toks-5pct/reports/memo-20-kit-r22.md`).
Everything here runs only with `GGML_C2_IO=1` (0002's counter) and only prints; no knob of its own.

- `c2-io-blk run=<VOLLIE_RUN> inblock=a/b/c`: `getrusage` `ru_inblock` (512-byte blocks read by the process) at 0002's
  three points (prompt start, first decode graph, end). Printed right after the `c2-io` line, which keeps its format.
  It is the same kernel counter as `/proc/self/io` `read_bytes`, read by a system call instead of a file the app could
  not open in r21.
- `c2-skip-run: run= sum= max= mass= c1_skip= c1_topp= c1_gap= prefill_… decode_…` with, per part, tokens, % pairs
  skipped, % expert bytes skipped, MB per token selected and skipped, % of read bytes left unread: 0003's
  `VOLLIE_SKIP_STATS` counters (every rule, campaign 1's included) split at the prompt's first decode graph. Printed
  only if the prompt counted pairs, so it needs `VOLLIE_SKIP_STATS=1` and a skip rule.
- `c2-head-run: run= c2_k= c2_calls= c2_cols= c2_exact_rows_per_col= c2_copies= c2_copy= c1_m= c1_calls= c1_copies= c1_copy=`:
  0004's head and campaign 1's `GGML_HEAD2` in this prompt (ops run, copies made, whether a copy exists at the end).
  Printed only if a two-stage head ran. Campaign 1's head gets three counters (ops, columns, copies) for this, its
  arithmetic untouched.
- **Identity** (*host*, 4 chunks, `-t 8`): per-chunk PPL equal to the 0001-0004 tree at `-b 512`, `-ub 32` and `-ub 1`
  for both models in both kit r22 arms (off: `GGML_TINY_OP=262144 GGML_EXPERT_IO=1 VOLLIE_SAMPLER_BYPASS=1 GGML_C2_IO=1`;
  num: + the model's num env, memo-20 §3).
