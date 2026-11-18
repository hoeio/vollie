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

## 0006-c2-head2-kleidiai.patch (load-time placement knob; numerics change when on in a KleidiAI build)

`VOLLIE_HEAD2_REPACK=1`, read **at model load** (unset = off). Files: `ggml/src/ggml-cpu/kleidiai/kleidiai.cpp`,
`ggml/src/ggml-cpu/ggml-cpu.c`.

- **Why (R60).** The Android module is built with KleidiAI on (`android/build_cpu.sh`); its extra buffer comes before
  `CPU_REPACK` and takes Q6_K weights (it requantizes them to int8 per row, `qsi8cx`). So OLMoE's `output.weight` sits in
  `CPU_KLEIDIAI`, `ggml_c2_head2_q6k_il` returns -1 and 0004 declines every head op. Host checks ran without KleidiAI,
  where the head is `q6_K_8x4` in `CPU_REPACK`. Reproduced on the host with a KleidiAI build (`GGML_KLEIDIAI_SME=0`).
- **What it does.** With the knob set at load, KleidiAI's `supports_op` declines a MUL_MAT whose weight is a Q6_K
  `output.weight`, so the loader puts it in `CPU_REPACK` (`q6_K_8x4` on a dotprod module), the layout 0004 handles. Only
  that tensor moves. Without KleidiAI in the build the knob does nothing. The app must set it before the model loads (both
  arms, since the kit loads once per block); it moves the stock head too, so it is a numerics change of the off arm on
  the phone (from KleidiAI's int8 head back to the Q6_K head the host's budgets were measured with).
- **Kit line.** `c2-head-run:` is also printed when `VOLLIE_HEAD2` declined in the prompt, with two fields appended:
  `c2_declined=<head ops declined> c2_why=<code>:<buffer of output.weight>`; codes `type`, `layout`, `ref`, `batch`, `k`,
  `shape` (the conditions 0004 already had, unchanged). R60's phone runs would have printed `c2_why=layout:CPU_KLEIDIAI`.
- **Host checks (2026-11-15, *host*, M4 Pro, `-t 8`, 4 chunks, `armv8.2-a+dotprod+fp16`, OpenMP on, Metal/BLAS off).**
  Knobs unset and kit off env: per-chunk PPL equal to the 0001-0005 tree at `-b 512`, `-ub 32`, `-ub 1`, both without and
  with KleidiAI. `VOLLIE_HEAD2=1024` (+ `_BATCH`, `_CHECK`, and with `MASS=900`): equal to 0001-0005 at `-b 512` and `-ub 1`
  (2 chunks). KleidiAI build + `VOLLIE_HEAD2_REPACK=1` + `VOLLIE_HEAD2=1024`: the head runs (`exact=q6_K_8x4`), copy
  fingerprint `0cb3ff34cab00989` equal to the non-KleidiAI build, `mismatches=0`. Per-chunk values of a KleidiAI build
  differ from a non-KleidiAI build in any case (its Q4_0 kernels), with or without this patch.

## 0007-c2-routing-prior.patch (numerics change when on)

`VOLLIE_PRIOR=<lambda x 1000>` (unset = off; `0` = counters only, no change), `VOLLIE_PRIOR_N=<tokens>` (default 4),
`VOLLIE_PRIOR_J=<j>` (default 0), with `VOLLIE_SKIP_STATS=1` a `c2-prior:` line at exit; host checks only:
`VOLLIE_PRIOR_DUMP=<file>`, `VOLLIE_PRIOR_DUMP_P=<file>`, `VOLLIE_PRIOR_POS=mask`. File: `ggml/src/ggml-cpu/ggml-cpu.c`
(+316 lines, 3 changed). memo-21 item N2 (cache-conditional routing) in the deterministic-history form. Host study: R64 queue in
`lab/logs/r64-prior/` (untracked), analysis `tools/c2_prior_trace.py`.

- **Rule.** Per MoE layer `l` and token at KV position `p`: `H` = union of the experts **selected** at layer `l` at
  positions `p-1 .. p-N` that this instance evaluated; the top k is taken over `probs_e * exp(lambda)` for `e` in `H`
  (plus the top `J` experts by unbiased probs, which are thereby always kept) and `probs_e` otherwise. For softmax gating
  (OLMoE, Granite) that is `+lambda` on the router logit. The mixing weights stay the unbiased probs: only the
  `ffn_moe_argsort-<l>` op is replaced, and the `get_rows` that reads the weights reads its source, not the score. So
  0003's skip rules (`VOLLIE_SKIP_MASS`, `GGML_EXPERT_SKIP`) still rank by the true weight and compose unchanged; the
  history is the selected set, before skip (the skip decision is made later, in the expert op).
- **Same decisions in a batch and one token at a time.** The history is a per-layer ring of 4,096 entries keyed by
  position with a position tag; an entry counts only if its tag is the position wanted. Thread 0 decides the rows of
  one graph in row order and each row writes its entry before the next row reads. A 512-token batch therefore takes
  the decisions 512 one-token graphs take **given the same probs**. Positions come from the graph's rope positions
  (`inp_pos`, OLMoE); a graph without rope (Granite-4.0-H's attention is NoPE: no `ROPE` node, no positions in the
  graph) uses, per token row, the number of KV cells its attention-mask row admits minus 1, which is the position in a
  single causal sequence. A pruned last layer (only the output rows, `out_ids`) maps its rows through `out_ids`. Both
  inputs are copied at graph start (the allocator reuses an input's memory after its last reader).
- **Sequence start, chunk start, KV reuse.** Position 0 has no history (`p-d < 0`). A new sequence overwrites the
  entries position by position before any later position reads them, so stale entries of an earlier sequence never
  count. A rollback or a KV prefix reuse finds the entries written when those positions were last evaluated, which
  are the ones the reused KV cells belong to. Not covered: a KV state loaded from a file (entries absent: the prior
  starts empty at the loaded positions) and interleaved multi-sequence batches (sequences would share entries; a
  contiguous sequence after another one in the same batch is fine). In a perplexity chunk the last layer routes only
  positions 256 to 511 (outputs), in both settings alike; in an app prompt it routes only the last prompt token, so
  the last layer's history at the first decode token is that token alone.
- **Counters** (`VOLLIE_SKIP_STATS=1` and `VOLLIE_PRIOR` set), on the slots the expert op really kept (after skip),
  per token and layer: `forced_per_lt` (selected experts outside the unbiased top k), `hist_overlap` (selected and in
  `H`), `prev_overlap_kept` (kept and kept at `p-1`), `new_vs_prev_per_lt` / `_per_tok` (kept slices not kept at
  `p-1`, the per-layer means on the `c2-prior-layer:` line), `ws32_slices` (distinct kept slices over aligned
  32-position windows, all layers). `VOLLIE_PRIOR_DUMP` writes 32-byte records (pos, layer, selected, kept, unbiased
  masks) per token and layer; `tools/c2_prior_trace.py stats|cmp|sim` reads them.
- **Cost.** With the knob unset: one `getenv` per graph. On: a graph scan per graph (positions, `out_ids`, k), the
  MoE argsort on thread 0 alone (two 64-element insertion sorts per row), about 2.1 MB of ring for 16 layers (5.2 MB for
  40). Not plumbed into `GGML_CPU_AB`.
- **Host checks (2026-11-16, *host*, M4 Pro, `-t 8`, build `armv8.2-a+dotprod+fp16`, KleidiAI, Metal and BLAS off (the budgets' build kind),
  OpenMP on; env `GGML_TINY_OP=262144 GGML_EXPERT_IO=1 VOLLIE_SAMPLER_BYPASS=1 GGML_C2_IO=1 VOLLIE_SKIP_STATS=1`):**
  - *Identity.* Knob unset and `VOLLIE_PRIOR=0` (counters on): per-chunk PPL of 3 chunks equal to the 0001-0006 tree at
    `-b 512`, `-ub 32` and `-ub 1`, on `olmoe-Mi` and `granite-Mi`, in the off arm and the num arm (OLMoE `MASS=900` +
    `VOLLIE_HEAD2=1024`, Granite `GGML_EXPERT_SKIP=130` + `GGML_HEAD2=2048 GGML_HEAD2_Q=q4_0`, `_BATCH` at `-b 512`).
    The 60-chunk stock values equal R50's and R59's at chunk 60 (OLMoE 12.6795, Granite 9.3419 / 9.3420, Granite num
    9.6526 / 9.6464).
  - *The op computes the defined rule in every batch shape.* `VOLLIE_PRIOR=1000 VOLLIE_PRIOR_N=4 VOLLIE_PRIOR_J=1`,
    2 chunks: `tools/c2_prior_trace.py replay` (the rule written out sequentially, its own history, on the probs the op
    saw) differs from the op's selection in 0 of 15,903 rows (OLMoE) and 0 of 40,527 (Granite), at `-b 512`, `-ub 32`
    and `-ub 1` each.
  - *Positions.* On OLMoE, `VOLLIE_PRIOR_POS=mask` gives dumps byte-identical to the rope positions at `-b 512` and
    `-ub 1` (2 chunks), so Granite's mask positions are the same quantity.
  - *What differs between settings is the probs, and the prior amplifies it.* The router probs of a batch and of one
    token differ in the last bits; in stock that changes 7.5 % of the (token, layer) expert sets between `-b 512` and
    `-ub 1` (OLMoE, 3 chunks), each change confined to its token. With the prior a change also moves the next N tokens'
    bias: 29 to 31 % of the sets differ at `VOLLIE_PRIOR=1000 N=4`. Hence the PPL gate at both settings is needed, not
    a formality.
  - *Full sets* (same env; `VOLLIE_PRIOR=500 VOLLIE_PRIOR_N=16`, the same setting on both models; queue and exact
    commands in `lab/logs/r64-prior/queue.txt`): OLMoE num + prior 12.2454 (`-b 512`) / 12.2435 (`-ub 1`) against num
    12.1053 / 12.0946 and the budget 12.3003; OLMoE prior alone 11.8587 (`-b 512` only) against stock 11.7146. Granite
    num + prior 9.5487 / 9.5478 against num 9.5173 / 9.5076 and the budget 9.6879; Granite prior alone 9.3172 (`-b 512`
    only) against 9.2266. All inside the budgets; the prior-alone rows lack their `-ub 1` run.
  - *What it buys* is file-page residency only: the expert bytes a token computes do not change (OLMoE 347.6 MB, num
    300 MB; Granite 246.3 MB, num 139 to 141 MB per token), so a fully resident run gains nothing. The 32-token
    working set falls 7 to 22 % and memo-18's simulator puts the decode re-read at 1.8 GB of cache 33 to 90 % lower
    (*estimated*); numbers and the reading against R62 / memo-23 are in the R64 hand-off, not here.
- **Not verified:** anything on Android; the cost of the thread-0 argsort on the A78 (*estimated* microseconds per
  layer); KV reuse and rollback (argued above, not exercised by `llama-perplexity`); the app's prompt shape (the
  simulator windows are wikitext positions 300 to 487 of each chunk, not the official prompt).

## 0008-c2-kv-dontneed.patch (declared bit-identical)

`VOLLIE_KV_DONTNEED=1` (unset or 0 = off), read at every clear. Files: `src/llama-impl.cpp`, `src/llama-impl.h`,
`src/llama-kv-cache.cpp`, `src/llama-memory-recurrent.cpp` (libllama only; the CPU module is untouched). memo-24 §A3 (a);
kit and checks: `campaigns/02-25toks-5pct/reports/memo-26-kit-r25.md`.

- **Path.** `llama_memory_clear(mem, true)` (llamadart before every prompt when `reusePrefix = false`) reaches
  `llama_kv_cache::clear(true)` and `llama_memory_recurrent::clear(true)` (a hybrid memory such as Granite's calls both),
  which called `ggml_backend_buffer_clear(buf, 0)`, a `memset` of the whole buffer in ggml-base. Both now call
  `llama_c2_buffer_clear0`. The clears at cache construction are unchanged.
- **What it does with the knob.** On Linux/Android, for a host buffer of type `CPU`: the page-aligned interior gets
  `madvise(MADV_DONTNEED)` and the unaligned edges a `memset`. Declined (whole `memset`, reason in the line) if the
  buffer is not `CPU` (`notcpu`), has no whole page (`small`), lies in a mapping that is not private anonymous per
  `/proc/self/maps` (`map`, checked once per base address), or `madvise` fails (`madvise`). Elsewhere (macOS): `memset`
  (`os`). The CPU buffer comes from `ggml_aligned_malloc` → `posix_memalign` (bionic scudo / glibc: private anonymous
  memory), so released pages read back as zeros.
- **Kit line** (with the knob or `GGML_C2_IO=1`, one per buffer cleared): `c2-kvclear run=<VOLLIE_RUN> kind=kv|rs on=0|1
  bytes=<buffer> released=<bytes given back by DONTNEED> why=ok|off|os|notcpu|small|map|madvise us=<clear time>`.
- **Host checks (2026-11-16, *host*, 4 chunks at `-b 512` and `-b 512 -ub 32`, 2 chunks at `-b 512 -ub 1`, `-c 512`,
  kit off env + `GGML_C2_IO=1`).** Knob on equals knob off per chunk for `olmoe-Mi` and `granite-Mi` at all three
  settings, (a) on macOS (M4 Pro, clang, `armv8.2-a+dotprod+fp16`; the knob's path is `memset` there, `why=os`) and
  (b) on Linux aarch64 (Ubuntu 24.04 container on the same laptop, gcc, same arch flags), where every clear printed
  `why=ok` and released the buffer less one edge page (OLMoE KV 67,104,768 of 67,108,864 B at `-c 512`; Granite KV
  4,190,208 and recurrent 58,056,704 of 58,060,800 B). Greedy: a test program on the app's settings (`n_ctx` 4096, q8_0
  K/V, flash attention) runs three prompts in one process, each after `llama_memory_clear(mem, true)`, 32 tokens: the
  three texts are equal to each other and between knob off and on, both models, on both platforms (Linux: OLMoE KV
  285,208,576 of 285,212,672 B released per clear). A Linux unit test of the function itself (buffers of 285 MB + 123 B,
  12,295 B, 100 B, 61 MB, filled non-zero first): every byte zero after the clear, RSS down by 291,960 KB for the 285 MB
  buffer, reuse after a release works.
- **Not verified:** anything on Android (bionic, the phone's kernel, zram, THP mode); the `map` decline path (no public
  API makes a `CPU` buffer over shared memory; a `CPU_Mapped` buffer over a shared mapping is declined as `notcpu` and
  zeroed by the `memset`).

## 0009-c2-kv-untag (2026-11-17, fix of 0008; bit-identical, same knob `VOLLIE_KV_DONTNEED=1`)

R64: on the phone every clear printed `why=map` (released 0): Android arm64 heap pointers carry a tag in the top
byte (scudo, top-byte ignore), and the tagged address matched no line of `/proc/self/maps`. 0009 computes the
page range, the maps check and the `madvise` on the untagged address (top byte cleared on aarch64; no change
elsewhere) and memsets the edges through the original pointer. The `c2-kvclear` line gains `tag=0x..` (the top
byte of the buffer address) so the phone shows the cause directly. Host check (*host*, Ubuntu 24.04 aarch64
container): `tools/c2_kvclear_tag_test.cpp` tags a 64 MB anonymous mapping with 0xb4: the maps check fails on
the tagged range and passes on the untagged one, `madvise` succeeds, every byte reads zero, 67,104,768 B
released; `llama-impl.cpp` compiles on macOS and Linux. Not yet run on the phone; per-chunk identity on the
full build still to be repeated with the kit tree.

Note 2026-11-18 (Tom, kit r26 tree `lab/llama.cpp-c1r26` = 620d2225c + `patches/llama.cpp/0001-0003` + `0001-0009`;
0009 applies on 0008): per-chunk identity of 0008 + 0009 repeated, `VOLLIE_KV_DONTNEED=1` against unset, kit off env,
`-c 512`, 20 chunks at `-b 512` and `-b 512 -ub 32`, 8 at `-b 512 -ub 1`, both models: equal on macOS (clear path `os`)
and in the Linux aarch64 container (`why=ok` in every clear); greedy texts (n_ctx 4096, q8_0, three clears per process)
equal off and on on both systems; the tagged-pointer test passes in the container (*host*; memo-27 §2).
