# Memo 11: kit interface in the runtime (0008) and the untied two-stage head (0009)

Agent model: claude-opus-5-5

Date 2026-11-14. Author: Tom (code). Patches `patches/llama.cpp-c2/0008-kit-interface.patch` and
`0009-head2-untied.patch` on top of `0001` to `0007` (base `620d2225c`); knobs and design in
[the series README](../../../patches/llama.cpp-c2/README.md). **No speed is stated anywhere.** Every number below is
*host* (Apple M4 Pro, `-DGGML_NATIVE=OFF -DGGML_CPU_ARM_ARCH=armv8.2-a+dotprod+fp16`, Metal and BLAS off, `-t 4`, while
an 8-thread perplexity sweep ran on the same machine) or *estimated* (arithmetic). Perplexities cover **4 chunks only**
(`-c 512 --chunks 4`): identity checks and a smoke test, not quality results, never compared with a budget.

## 1. What exists

- **0008 (bit-identical).** Every `VOLLIE_*` knob is read per graph; per-prompt counter lines through the log callback;
  a per-token `cpu-prof` line (`VOLLIE_PROF=1`) with a per-class breakdown; in-run ABBA A/B
  (`VOLLIE_AB=knob:<NAME>:<A>:<B>`) for the six exact levers, the three skip rules and `HEAD2`.
- **0009 (declared numerics change, off unless `VOLLIE_HEAD2=<K>`).** memo-07 §5's design, extended to the repacked
  head because the kit runs OLMoE with repack on (memo-08 §9). Screen: a Q4_0 copy, `block_q4_0x4` (4x4 on dotprod,
  the SD888), 57.95 MB (*estimated*: 50,304 rows × 2,048 / 32 × 18 B) of anonymous memory owned by the CPU module,
  made once at the first head op from the head's own rows (repack on: un-interleaved from the `q6_K_8x4` copy in the
  `CPU_REPACK` buffer; repack off: the mmapped rows). Exact rows: the K best screen rows recomputed with the stock kernel;
  with repack on that is the stock `q6_K_8x4` gemv on every touched 8-row group, so all 8 rows of a touched group are
  exact.

## 2. Knobs: run time and load time

"Per graph" = read again at the start of every `ggml_graph_compute` (CPU module) or every graph build (libllama), so
an app may change it between prompts (and between tokens). The table in the README has the meanings.

| knob | patch | read |
|---|---|---|
| `VOLLIE_SKIP_SUM`, `VOLLIE_SKIP_MAX`, `VOLLIE_SKIP_MASS`, `VOLLIE_SKIP_MINK`, `VOLLIE_SKIP_STATS` | 0001 | per graph (was once per process) |
| `VOLLIE_SKIP_WIRE` | 0008 | **load time**: at graph build; llama.cpp reuses a built graph while shapes match, so set it before the context's first decode if a skip threshold may be switched on later |
| `VOLLIE_TINY_OP`, `VOLLIE_GRAPH_STATS` | 0002 | per graph |
| `VOLLIE_Q3K_X4` | 0003 | per graph |
| `VOLLIE_MMID_NB` | 0004 | per graph |
| `VOLLIE_MMID_FUSE` | 0005 | per graph |
| `VOLLIE_EXPERT_IO` | 0006 | per graph |
| `VOLLIE_SAMPLER_BYPASS` | 0007 | per sampler call |
| `VOLLIE_PROF`, `VOLLIE_AB`, `VOLLIE_AB_WARM`, `VOLLIE_AB_BLOCK`, `VOLLIE_AB_BAAB`, `VOLLIE_RUN`, `VOLLIE_LINES` | 0008 | per graph (`VOLLIE_AB` for `SAMPLER_BYPASS`: per sampler call) |
| `VOLLIE_HEAD2`, `VOLLIE_HEAD2_BATCH`, `VOLLIE_HEAD2_CHECK` | 0009 | per graph; the screen copy is made at first use and kept until exit |

Exit-only registrations (`atexit` for `c2-skip:`, `c2-graph:`, `c2-head:`) happen once, at the first graph that has the
knob set. Repack on/off is not a `VOLLIE_` knob (it is the model load parameter `use_extra_bufts`, `-nr` on the host).

## 3. Line formats and frozen regexes

All lines are single lines. By default they go through ggml's log callback at INFO level (`llama_log_set`); with
`VOLLIE_LINES=stderr` they go to stderr. **The regexes are not anchored at the line start**: a logger may prefix lines
(llama.cpp's own host tools print `0.00.276.155 I ` before a line, and at their default verbosity they drop library
INFO lines altogether, so host runs need `VOLLIE_LINES=stderr`). Fields are separated by one space; `run` is
`[A-Za-z0-9._-]{1,32}` (`-` when `VOLLIE_RUN` is unset). Times are µs on the calling thread's monotonic clock
(`ggml_time_us`).

**Per token** (`VOLLIE_PROF=1`, one per one-token graph, printed after the graph):

```
cpu-prof ab=B tok=6 us/token=23172 majflt=0 minflt=6945 mhz=0/0 run=- prompt=1 blk=2 threads=4 ops=501 barriers=274 syncs=414 skipped=0 pairs=0 head=0 router=94 mm=1470 mmid=14340 attn=76 other=1265 wait=5866 | ADD 144:50+6 MUL 16:855+3 RMS_NORM 65:180+11 MUL_MAT 81:1564+213 MUL_MAT_ID 48:14340+5595 GET_ROWS 19:7+2 SET_ROWS 32:38+11 SOFT_MAX 16:7+2 ROPE 32:31+6 ARGSORT 16:27+0 FLASH_ATTN_EXT 16:69+12 GLU 16:77+5
```

```
cpu-prof ab=([AB-]) tok=(\d+) us/token=(\d+) majflt=(\d+) minflt=(\d+) mhz=(\d+)/(\d+) run=([A-Za-z0-9._-]+) prompt=(\d+) blk=(\d+) threads=(\d+) ops=(\d+) barriers=(\d+) syncs=(\d+) skipped=(\d+) pairs=(\d+) head=(\d+) router=(\d+) mm=(\d+) mmid=(\d+) attn=(\d+) other=(\d+) wait=(\d+) \|((?: [A-Z0-9_]+ \d+:\d+\+\d+)*)
op table entry:  ([A-Z0-9_]+) (\d+):(\d+)\+(\d+)      (op name, count, compute µs, barrier wait µs; thread 0)
```

The example is *host* (the `-ub 1` perplexity A/B rerun of §4; `head=0` because llama-perplexity asks for no logits in
the first half of a chunk). All 2,047 `cpu-prof` lines of that run matched the regex, op table included; every
`c2-` line of the multi-prompt run in §5 matched its regex below.

- `ab`: side of this graph (`-` = no A/B). `tok`: one-token graph number within the prompt (1 = the first decode
  token). `blk`: A/B block (0 = warm-up). `mhz`: cpu4/cpu7 `scaling_cur_freq` at the line (0 where sysfs has none).
  `majflt`/`minflt`: process page faults during the graph (`getrusage`). `ops`, `barriers`: ops run and barriers
  between them, the closing one included; `syncs`: every `ggml_barrier` call, per thread. `skipped`/`pairs`: 0001's
  (expert, token) pairs skipped / routed in this graph.
- Classes (thread 0 compute µs, the same node in exactly one class): `head` = `MUL_MAT` on `output.weight` or
  `token_embd.weight`; `router` = `MUL_MAT` on `*ffn_gate_inp*`; `mm` = other `MUL_MAT` (attention projections);
  `mmid` = `MUL_MAT_ID` (experts; a 0005 fused group counts as its first node); `attn` = `FLASH_ATTN_EXT` + `SOFT_MAX`;
  `other` = every other op; `wait` = thread 0's wait at the barriers after the ops. `us/token` is the whole graph on
  the calling thread, so it also holds the graph's start and end work outside the ops.
- `tools/c2_research_parse.py` `token_lines` reads `tok`, `us/token`, `majflt`, `minflt` and the class fields from this
  line as it is (numeric `key=value` before `|`); it does not read `ab` (a letter) or `run`. See §8.

**Per prompt** (printed when the prompt has ended: at the next prompt's first graph, or when the context is freed):

```
c2-skip-run: run=(\S+) prompt=(\d+) tokens=(\d+) pairs=(\d+) pairs_skipped=([\d.]+)% expert_bytes_skipped=([\d.]+)% MB_per_token_selected=([\d.]+) MB_per_token_skipped=([\d.]+) batch_bytes_unread=([\d.]+)%
c2-graph-run: run=(\S+) prompt=(\d+) one_token_graphs=(\d+) nodes=([\d.]+) ops_run=([\d.]+) ops_tiny=([\d.]+) barriers=([\d.]+) syncs=([\d.]+)
c2-head-run: run=(\S+) prompt=(\d+) k=(\d+) calls=(\d+) cols=(\d+) exact_rows_per_col=([\d.]+) checked=(\d+) top1_in_set=([\d.]+)% mismatches=(\d+)
```

`c2-skip-run` needs `VOLLIE_SKIP_STATS=1` (tokens include the prompt's tokens); `c2-graph-run` needs
`VOLLIE_GRAPH_STATS=1` (or `VOLLIE_PROF=1`) and counts one-token graphs only; `c2-head-run` appears when the head2
path ran in the prompt (`k` = the last K used).

**Per prompt, at its start, with A/B:**

```
c2-ab: run=(\S+) prompt=(\d+) knob=([A-Z0-9_]+) a=(-?\d+) b=(-?\d+) warm=(\d+) block=(\d+) order=(ABBA|BAAB)
c2-ab: error=unknown_or_malformed value=(.*)          (once per process)
```

**Once, when the head2 copy is made:**

```
c2-head-build: screen=(q4_0x4x4|q4_0x4x8|q4_0) exact=(q6_K_8x4|q6_K_8x8|q6_K) rows=(\d+) copy_mb=([\d.]+) ms=([\d.]+) threads=(\d+) sum=([0-9a-f]{16})
```

(`sum` is a fingerprint of the copy, 0 unless `VOLLIE_HEAD2_CHECK=1`.)

**At process exit, on stderr (host tools; unchanged from 0001 / 0002, new for 0009):**

```
^c2-skip: sum=… max=… mass=… mink=… tokens=… pairs=… pairs_skipped=…% …        (0001, VOLLIE_SKIP_STATS=1)
^c2-graph: tiny_op=… one_token_graphs=… nodes=… ops_run=… ops_tiny=… barriers=… syncs=…   (0002)
^c2-head: k=(\d+) calls=(\d+) cols=(\d+) exact_rows_per_col=([\d.]+) checked=(\d+) top1_in_set=([\d.]+)% mismatches=(\d+) copies=(\d+) build_ms=([\d.]+)
```

The exit lines print the knob values current at exit (under A/B: the env value, not the side's).

**Overhead of the profile (not measured).** Per profiled graph: three clock reads per op on thread 0 (501 ops), two
`getrusage` calls, two sysfs opens, one formatted line through the log callback, and one atomic add per `ggml_barrier`
call per thread for `syncs`. *Estimated* 0.1 to 0.3 ms per token on the SD888, about 0.3 to 1 % of a 30 to 40 ms token;
campaign 1 measured about +2.5 % for its profiler (*device*). The kit should treat `VOLLIE_PROF=1` runs as their own
condition and compare speed only between runs with the same `VOLLIE_PROF`.

## 4. Identity (0008 knobs, *host*)

Reference: the base build (no patches), memo-07 §3. Every configuration below reproduced every per-chunk value and the
md5 of the `--kl-divergence-base` file (all logits, as 16-bit, plus the exact per-token log-sum-exp) at all six settings,
and the 32-token greedy text (`301bf855…` for both `rp` and `-nr`). Build: the series 0001 to 0008.

| setting | rp: per-chunk PPL | rp: logits md5 | -nr: per-chunk PPL | -nr: logits md5 |
|---|---|---|---|---|
| `-ub 512` | 8.9646 10.4244 9.2074 9.4783 | `3c8948f1…` | 8.9278 10.4339 9.2003 9.4521 | `8dd05b9e…` |
| `-ub 32` | 8.9131 10.3435 9.0932 9.3987 | `ba2b6002…` | 8.8914 10.3906 9.1549 9.4018 | `32adfbc6…` |
| `-ub 1` | 8.9550 10.4710 9.1889 9.4713 | `69f9df67…` | 8.9619 10.5154 9.2265 9.4738 | `851f77bf…` |

Configurations, each identical to this table and to the base text: knobs unset; `VOLLIE_PROF=1 VOLLIE_GRAPH_STATS=1`;
`VOLLIE_AB=knob:TINY_OP:0:65536` (`WARM=2 BLOCK=3`, `VOLLIE_PROF=1`); `VOLLIE_MMID_NB=1 VOLLIE_AB=knob:MMID_FUSE:0:1`
(`WARM=2 BLOCK=3`); `VOLLIE_SKIP_WIRE=1` (thresholds unset). A/B acts in perplexity only at `-ub 1` (every graph is one
token there): a rerun of the `TINY_OP` A/B at `-ub 1` with `VOLLIE_LINES=stderr` printed 1,024 side-A and 1,023 side-B
`cpu-prof` lines, B with `barriers=274` against A's 501, logits md5 `69f9df67…` as stock.

Series 0001 to 0009, the same six settings and the greedy text, identical to the table: knobs unset; and
`VOLLIE_AB=knob:HEAD2:0:0` (`WARM=2 BLOCK=3`: the screen copy is made, both sides run the stock head).

## 5. A/B greedy text identity (*host*, 64 tokens, `WARM=4 BLOCK=4`, `VOLLIE_PROF=1`, series 0001 to 0009)

`llama-completion -p "The capital of France is" -n 64 --temp 0 -t 4`. Each A/B run printed 32 side-A and 31 side-B
`cpu-prof` lines (63 decode graphs). The "B side" column is the last B line's counts, which show the lever acting.

| A/B | text md5, rp and -nr | reference | B side (A side: ops 501, barriers 501, syncs 646) |
|---|---|---|---|
| none | `264380bf…` both | — | — |
| `TINY_OP:0:65536` | `264380bf…` both | = none | barriers 274, syncs 419 |
| `Q3K_X4:0:2` | `264380bf…` both | = none | counts unchanged (kernel swap) |
| `MMID_NB:0:1` | `264380bf…` both | = none | syncs 614 |
| `MMID_FUSE:0:1` (with `MMID_NB=1`) | `264380bf…` both | = none | ops 469, barriers 469, syncs 582 (A: syncs 614) |
| `EXPERT_IO:0:1` | `264380bf…` both | = none | counts unchanged (advice only) |
| none, `--repeat-penalty 1.1` | rp `038008eb…`, -nr `46a52a0c…` | — | — |
| `SAMPLER_BYPASS:0:1`, `--repeat-penalty 1.1` | rp `038008eb…`, -nr `46a52a0c…` | = none with penalty | — |
| `HEAD2:0:1024` (numerics change) | `264380bf…` both | = none | syncs 649 |
| `SKIP_MASS:0:900` (numerics change) | rp `af7e4967…`, -nr `e7a1041d…` | differs, as expected | — |
| `VOLLIE_SKIP_MASS=900`, no A/B | rp `e199e657…`, -nr `17b2679d…` | — | — |
| `knob:NOPE:0:1` | `264380bf…` both | = none | one `c2-ab: error=…` line, no A/B |

The skip A/B text differs from both pure runs, as it must: it switches the rule every four tokens.

Multi-prompt check: a small libllama program (scratch, not in the repository) loaded the model once and ran two
prompts with `VOLLIE_RUN` set to `runA` and `runB`; the output had one `c2-skip-run`, `c2-graph-run` (and, on the full
series with `VOLLIE_HEAD2=1024`, `c2-head-run`) line per prompt with `prompt=1` / `prompt=2` and the right run ids, the
second prompt's lines printed at `llama_free`, then the exit lines. The per-prompt skip counts summed to the exit line's.

## 6. Two-stage head (0009), smoke (*host*, 4 chunks)

**Exactness.** (a) `VOLLIE_HEAD2=50303` (every 8-row group touched, so every row exact) at `-ub 1`, repack on: per-chunk
8.9550 10.4710 9.1889 9.4713, logits md5 `69f9df67…`, the stock `-ub 1` values bit for bit (1,025 head calls). (b)
`VOLLIE_HEAD2_CHECK=1` compared the exact rows with the whole head from the stock kernel in every checked column:
**0 mismatches** in all 12 runs below and in greedy decoding (33 calls, rp and -nr, K = 1,024 and 50,303). (c) The screen
copies made from the repacked rows and from the plain rows have the same fingerprint (`0cb3ff34cab00989`). (d) Greedy
text with `VOLLIE_HEAD2=1024`: `301bf855…`, as stock, rp and -nr.

**PPL smoke** (`-c 512 -b 512 --chunks 4 -t 4`, `VOLLIE_HEAD2_BATCH=1 VOLLIE_HEAD2_CHECK=1`; 1,025 scored columns per
run). "Top-1 in set": columns whose true argmax (whole stock head) is among the K screened rows. In batch mode the exact
rows are the decode gemv's values while the stock `-b 512` head is the gemm, so the K rows themselves differ slightly
from the stock rows here; that and 4 chunks make the sign of small differences meaningless.

| path | setting | per-chunk PPL | final | vs its stock | exact rows / column | top-1 in set |
|---|---|---|---|---|---|---|
| rp | stock | 8.9646 10.4244 9.2074 9.4783 | 9.4783 | — | all | — |
| rp | K=1024 | 8.9725 10.4190 9.2020 9.4719 | 9.4719 | −0.07 % | 6,981.9 | 100 % |
| rp | K=2048 | 8.9735 10.4195 9.2023 9.4722 | 9.4722 | −0.06 % | 12,908.1 | 100 % |
| rp | K=4096 | 8.9740 10.4282 9.2095 9.4799 | 9.4799 | +0.02 % | 22,559.0 | 100 % |
| rp | MASS=900 | 9.1525 10.7417 9.8247 10.1338 | 10.1338 | +6.92 % | all | — |
| rp | K=1024 + MASS=900 | 9.1651 10.7361 9.8189 10.1274 | 10.1274 | +6.85 % | 6,964.4 | 100 % |
| rp | K=2048 + MASS=900 | 9.1591 10.7326 9.8168 10.1257 | 10.1257 | +6.83 % | 12,868.9 | 100 % |
| rp | K=4096 + MASS=900 | 9.1614 10.7445 9.8264 10.1350 | 10.1350 | +6.93 % | 22,495.0 | 100 % |
| -nr | stock | 8.9278 10.4339 9.2003 9.4521 | 9.4521 | — | all | — |
| -nr | K=1024 | 8.9369 10.4230 9.1884 9.4424 | 9.4424 | −0.10 % | 1,024 | 100 % |
| -nr | K=2048 | 8.9382 10.4212 9.1928 9.4440 | 9.4440 | −0.09 % | 2,048 | 100 % |
| -nr | K=4096 | 8.9278 10.4272 9.1964 9.4491 | 9.4491 | −0.03 % | 4,096 | 100 % |
| -nr | MASS=900 | 9.0953 10.7432 9.8350 10.1206 | 10.1206 | +7.07 % | all | — |
| -nr | K=1024 + MASS=900 | 9.0992 10.7253 9.8170 10.1036 | 10.1036 | +6.89 % | 1,024 | 100 % |
| -nr | K=2048 + MASS=900 | 9.1013 10.7275 9.8206 10.1064 | 10.1064 | +6.92 % | 2,048 | 100 % |
| -nr | K=4096 + MASS=900 | 9.1031 10.7384 9.8320 10.1183 | 10.1183 | +7.05 % | 4,096 | 100 % |

The MASS=900 rows are the 4-chunk smoke values (pairs skipped 13.77 %); R51 has the full-set skip frontier. The
budget question is only answered by the full set (§7).

**Bytes per token with repack on (*estimated*, from the host exact-row counts above).** With the repacked head every
touched 8-row group is read whole: 6,982 exact rows per column at K = 1,024 is about 873 groups × 13,440 B = 11.7 MB.
Head bytes per token: stock 84.5 MB; K = 1,024: 57.95 + 11.7 = 69.7 MB; K = 2,048: 57.95 + 21.7 = 79.6 MB; K = 4,096:
57.95 + 37.9 = 95.9 MB, **more than stock**. With repack off: 57.95 + K × 1,680 B = 59.7 / 61.4 / 64.8 MB. On the kit's
repack-on path only K around 1,024 or below saves bytes; memo-07 §5's 25 MB saving holds only for the plain head.
A group-aware select (pick the best 8-row groups, not rows) would bound the exact bytes at K/8 groups; not built.

**Host build time of the copy:** 51 to 69 ms (repack on, 4 threads), 28 to 42 ms (repack off), *host*; once per process.

## 7. Full-set commands (owner's laptop)

New script `tools/c2_head2_sweep.sh` (resumable, prints a table with "% stock" against its own stock row). With the
series 0001 to 0009 applied and built in `<tree>`:

```sh
# repack on (the kit's OLMoE path): stock, HEAD2 = 1024 / 2048 / 4096 alone and with MASS=900
tools/c2_head2_sweep.sh <tree> lab/olmoe/olmoe-Mi.gguf <wiki.test.raw> lab/c2-head2-rp \
  stock K=1024 K=2048 K=4096 MASS=900 K=1024,MASS=900 K=2048,MASS=900 K=4096,MASS=900
# repack off (the phone's -nr path, for comparison)
PPL_ARGS=-nr tools/c2_head2_sweep.sh <tree> lab/olmoe/olmoe-Mi.gguf <wiki.test.raw> lab/c2-head2-nr \
  stock K=1024 K=2048 K=4096 MASS=900 K=1024,MASS=900 K=2048,MASS=900 K=4096,MASS=900
```

Each setting runs `env VOLLIE_SKIP_STATS=1 [VOLLIE_HEAD2=<K> VOLLIE_HEAD2_BATCH=1] [VOLLIE_SKIP_MASS=900]
llama-perplexity -m <model> -f <text> -c 512 -b 512 -t 8 -ngl 0 [-nr]`. No `VOLLIE_HEAD2_CHECK` (it computes the
whole head a second time on one thread). 16 full runs; the stock rows can be checked against R50.

## 8. Notes for the lead (kit integration)

1. **Per-prompt lines arrive late.** A prompt's `c2-*-run` lines are printed when the next prompt starts or the context
   is freed. The kit reads its log right after each reply, so prompt N's lines land in row N+1 and the last prompt's
   only at context free. Associate by the `run=` / `prompt=` fields (set `VOLLIE_RUN` per run), or free the context (or
   send one short prompt) before reading the last run. The `cpu-prof` lines are printed per token, without delay.
2. **Prefix.** The kit filters with `startsWith('cpu-prof')` / `startsWith('c2-')`. If llamadart's native logger adds a
   prefix, those lines are lost; `contains` (or the unanchored regexes of §3) is safe. Not checked on the device.
3. **Parser.** `tools/c2_research_parse.py` `token_lines` drops `ab=` (non-numeric); A/B scoring needs `ab` and `blk`
   parsed (e.g. `ab=([AB-])`). Not changed here.
4. **`tok=`** counts one-token graphs per prompt (1 = the first decode token); campaign 1's per-token line used the
   A/B block number there.

## 9. Android cross-build

`tools/c2_build_android.sh` (unchanged) on a tree with 0001 to 0009 applied: exit 0 with NDK 28.2.13676358 (static,
no OpenMP, KleidiAI off), llama-bench, llama-perplexity and llama-completion built; the only warnings are 0001's three
float-to-double promotions. Compiled only, not run. The script records one optional patch file's sha256
(`[patch-file]`), not a series: for this run its `COMMIT` file says "patch none" plus the tree's diff stat (8 files,
2,034 insertions). The nine patch files applied in order to a clean `620d2225c` tree give, file by file, the tree
that was tested here. Suggested change (not made): accept the series directory and list each patch's sha256.

## 10. Not verified

1. Anything on the device: speed, the profile's overhead, the A/B's effect on timing, whether the lines reach
   `vollie_load.log` through llamadart's logger and with what prefix.
2. Identity on the full set and at thread counts other than 4; identity on any model but `olmoe-Mi`.
3. The head2 PPL on the full set (§7) and therefore whether any K fits the budget; the 4-chunk smoke says nothing on that.
4. The `q6_K_8x8` / `q4_0x4x8` (i8mm) layouts: the host build has no i8mm, so only the 4x4 / 8x4 paths ran. The SD888 has
   no i8mm either.
5. That llama.cpp never reuses a graph built without skip wiring after a threshold is set mid-context without
   `VOLLIE_SKIP_WIRE=1` (documented as the user's job, not tested).
6. The head2 copy's memory on the phone (+57.95 MB anon, *estimated*) against the low-memory killer.
7. Concurrency: the kit state is process-global; two contexts computing at once in one process would mix counters (the
   kit runs one).
