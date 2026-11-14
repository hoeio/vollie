# Memo 19: research kit r21, prefill COLD on two models in one session

Agent model: claude-opus-5-5

Date 2026-11-14. Author: Tom (code). **Nothing here ran on a phone, and no phone speed is stated.** The APK was built
and inspected on the laptop. Labels: *host* = Apple M4 Pro, relative only; *estimated* = arithmetic; *device* = quoted
from an existing report, with the report named. Lever and tau come from memo-18 §3 (d) and §5.

## 1. The patch: `patches/llama.cpp-c1-olmoe/0002-prefill-cold.patch`

It applies on top of campaign 1's `0001` to `0003` and `0001-keep-warm`. Files: `ggml-cpu.c` +218 lines,
`ggml-cpu.cpp` +3. The full description is in `patches/llama.cpp-c1-olmoe/README.md`.

### 1.1 What it does

- **`GGML_PREFILL_COLD=<tau>`** is off when unset. It is advice only.
- It acts only in batch graphs (`n_tokens > 1`).
- At each layer's first `MUL_MAT_ID` with more than one row, thread 0 counts how many rows the routing sends to each
  expert.
- After that layer has run, it advises every slice the prompt routed 1 to tau-1 times. "After the layer has run" means
  at the next layer's first `MUL_MAT_ID`, behind the barrier that precedes it, or at the end of the graph.
- The advice covers every `MUL_MAT_ID` weight that shares those ids (gate, up and down), page-aligned inward.
- Slices routed 0 times are left alone.
- Only `CPU_Mapped` buffers are advised, which means the file mapping. Anonymous copies are never advised, because on
  anonymous memory the advice would mean swap.
- The pruned last layer has a one-row `ids`, so it is skipped.
- No architecture name appears anywhere: `granitehybrid` and `olmoe` take the same code path.

### 1.2 Advice order, probed at run time on the first slice

1. `MADV_COLD` (20, Linux 5.4 and later): the pages are deactivated and stay cached until reclaim wants the room.
2. `MADV_PAGEOUT` (21, same kernel version): the pages are reclaimed at once.
3. None: the slices are counted, but no call is made.

The probe prints one line:
`c2-pcold-advice advice=<cold|pageout|none> errno_cold=<e> errno_pageout=<e> page=<bytes>`.

Then each batch graph prints one line:
`c2-pcold tau= advice= tokens= layers= routed_slices= marked_slices= mb= fails= us=`.

The phone's kernel version is not in any report; `/proc/version` reads empty in the app (r17 shows `"kernel":""`). Kit
r21 also records Dart's `Platform.operatingSystemVersion`. Whether COLD exists on the phone is therefore settled by
the probe line, not assumed.

### 1.3 The counter: `GGML_C2_IO=1`, separate and off by default

- It samples `/proc/self/io` `read_bytes` and `rchar` plus `getrusage` major and minor faults at three points:
  - the prompt's first batch graph;
  - the first one-token graph after it;
  - the end of every graph.
- It prints one line per prompt:
  `c2-io run=<VOLLIE_RUN> prompt_graphs= decode_graphs= us=a/b/c read_bytes=a/b/c rchar=a/b/c majflt=a/b/c minflt=a/b/c`.
- Prefill is b − a and decode is c − b.
- The line is printed when the next prompt starts or when the CPU backend is freed. It carries its `run=` tag, so the
  kit files it to the right row even though it arrives late.

### 1.4 tau = 8

- Memo-18's replay is flat for tau from 4 to 16 for both models; 8 is the middle of that range.
- Marking more slices (16: about 1.2 GB) risks pushing out slices that the decode also needs. Marking fewer (4: about
  0.37 GB) leaves most of the prompt-only bytes active.

MB marked on the official prompt (*host*, 16 KB pages, inward-aligned; `-t 3`, KV q8_0, flash attention, ctx 4096):

| model | prompt as the phone renders it | marked slices (layer × expert pairs) | MB marked | tau 4 / 16 (llama-completion `-cnv` render) |
|---|---|---|---|---|
| OLMoE `olmoe-Mi` | 156 tokens, 15 counted layers | 729 (243) | 645 | 358 / 1,205 (151 tokens; 666 at tau 8) |
| Granite `granite-Mi`, no skip | 155 tokens, 39 counted layers | 2,301 (767) | 740 | 386 / 1,183 |

- Memo-18 estimated about 690 and 870 MB. Its Granite figure was for d6-e6 with skip 117.
- On the phone's 4 KB pages, inward alignment wastes less, so the phone marks slightly more: at most about +17 MB for
  OLMoE and +55 MB for Granite (*estimated*, at most 12 KB more per slice).
- The thread-0 cost on the host is 9 to 27 µs per prompt. That is with no system call, because macOS has neither
  advice. The phone's cost of about 700 MB of `madvise` is unknown; it shows in `us=`.

### 1.5 Host checks (*host*)

**Identity.** Per-chunk PPL (4 chunks, `-c 512 -t 4`) and the md5 of the `--kl-divergence-base` file. "Unset" here
means `GGML_TINY_OP=262144 GGML_EXPERT_IO=1 VOLLIE_SAMPLER_BYPASS=1`; "set" means the same plus
`GGML_PREFILL_COLD=8 GGML_C2_IO=1`.

| model | `-b 512` | `-ub 32` | `-ub 1` |
|---|---|---|---|
| OLMoE, unset = set | 8.9646 / 10.4244 / 9.2074 / 9.4783, md5 `3c8948f1` | 8.9131 / 10.3435 / 9.0932 / 9.3987, `ba2b6002` | 8.9550 / 10.4710 / 9.1889 / 9.4713, `69f9df67` |
| Granite, unset = set | 4.9584 / 6.6520 / 6.1119 / 6.6024, `559d9d5c` | 4.9398 / 6.6449 / 6.0940 / 6.5732, `772221a7` | 4.9610 / 6.6458 / 6.0918 / 6.5559, `8b9171bb` |

- The knob acted in the `-b 512` and `-ub 32` runs: 2 and 62 `c2-pcold` lines.
- A `-ub 1` run has no batch graph, so neither line is printed there. Its row shows only that the knobs are inert at
  one token.
- **For the lead:** with this scratch build, OLMoE at `-ub 1` gives memo-07's values (8.9550 …). Memo-17 had 8.8302
  from the c1kw build. Both are the same source with different build options (this build: BLAS off,
  `GGML_CPU_ARM_ARCH=armv8.2-a+dotprod+fp16`, default Accelerate and OpenMP). Identity is judged within one build, so
  this does not affect the result.

**Texts.** Official prompt, 32 greedy tokens, each model's own template, `--repeat-penalty 1.1 --repeat-last-n 64`.
One text per model across all of the following env sets: stock, shipping, shipping + `C2_IO` (the off arm),
+ `PREFILL_COLD=8` (the on arm), + the slice log, both profiled arms (31 per-token lines each), and tau 4 and 16.

| model | output md5 |
|---|---|
| OLMoE | `4ff6b707…` |
| Granite | `b97e8c05…` |

**The marked set is the tracer's rarely routed set.**
- `GGML_PREFILL_COLD_LOG=1` prints every marked (layer, expert) with its count.
- The id tracer (`tools/c2_ws_trace_ids.cpp`, the memo-18 fix) logged the prefill routing of the same rendered
  prompt.
- The two sets are equal, counts included: OLMoE 243 pairs, Granite 767 pairs.
- On macOS no advice is issued, so `mincore` cannot show pages leaving. What the host shows is that exactly the ranges
  the tracer calls rarely routed are passed to the advice.

**Parsing.** The host `c2-io` lines parse with the parser's `io_of`. On macOS `read_bytes` and `rchar` read as -1 and
come out as empty; the faults and µs parse.

**App.** `flutter analyze` is clean; `kit_lines_test.dart` passes 6 tests (2 new: the r21 lines, and the run list as
generated from the model table); `tools/c2_research_parse.py --selftest` passes, with an r21 case on three models.

## 2. The model table (owner's addition of 2026-11-14)

`Research.kitModels` is one list of entries with these fields: id, display name, architecture, file, URL, bytes,
sha256, source (`app` or `kit`), template source, repack rule and baseline settings, plus each file's expert byte
ranges.

Generated from the list:
- the download button and its storage check;
- the blocks;
- the mincore split;
- the report's `models` block;
- the parser's N-column table.

| id | file | bytes | sha256 | arch | source | template | repack | settings |
|---|---|---|---|---|---|---|---|---|
| granite | `granite-Mi.gguf` (commit `e893e8a6`) | 3,161,269,344 | `8df94de7…` | granitehybrid | kit download | the file's own | on | campaign 1 shipping minus numerics |
| olmoe | `olmoe-Mi.gguf` (commit `e893e8a6`) | 3,072,399,552 | `5abf710b…` | olmoe | the app's download | the file's own | on | the same |

**Both models run without skip 117 and without head2:** the baseline is `GGML_TINY_OP=262144 GGML_EXPERT_IO=1
VOLLIE_SAMPLER_BYPASS=1`. The report says so in `baseline_note`. Repack is set explicitly at every load (`_repack =
true`), so the app's size rule never applies. That rule uses the hard-coded `copyBytes` for the exact file size, and
granite-Mi does not match it. Every row records `repack_loaded`.

Granite's template is the one embedded in `granite-Mi.gguf` (6,418 characters); the kit leaves `_template` null.

## 3. Run list

Blocks run in table order forward, then backward: **Granite, OLMoE, OLMoE, Granite**. Each block:

1. Load its model, forced even when the same model is already loaded, repack on. Before the load, mincore of every
   file and meminfo are recorded.
2. Two unscored warm-ups (arm off, no gate).
3. Eight scored official runs, knob off/on in ABBA: off, on, on, off, off, on, on, off.
4. One profiled run per arm (campaign 1's A/A per-token label): off then on in blocks 1 and 3, on then off in blocks 2
   and 4.

r16's gate runs before every run except the warm-ups: go below 0.65, read every 5 s, give up after 90 s, reads at
least 1.1 s apart, and a null reading means "read again". That is 48 runs in total, 40 of them gated.

- **On arm:** baseline + `GGML_C2_IO=1` + `GGML_PREFILL_COLD=8`.
- **Off arm:** baseline + `GGML_C2_IO=1`.

The arms differ in one knob only.

## 4. Report schema (additions over r20)

**Top level:**
- `models`: per id, the table fields, the `.verified` marker and the sha256 at the end of the kit (both files are
  hashed last);
- `templates`: per id, the render of the official prompt and its token count;
- `block_order`, `baseline_note`, `os_version`;
- `res_start` and `res_end`: mincore of every file.

**Per row:**
- `model`, `arm`, `loaded_file`, `repack_loaded`;
- `reload_ms` and `libs` at each block's load;
- `res_both_before_load` and `mem_before_load` on a block's first row;
- the gate series, `heat`, `gate_timeout`, `cap_before` and `cap_after`;
- `res_before` and `res_after`: mincore split into experts and the rest, using each file's own ranges;
- `mem_before` and `mem_after`: MemAvailable, Cached, SwapFree, RssAnon, RssFile and VmSwap;
- `majflt`, `minflt`, `io_read_bytes` (the whole run), and `io_read_bytes_to_first_chunk` (the app's own prefill
  split);
- `ttft_ms`, `first_chunk_ms`, `decode_tps`;
- the `c2` lines: `c2-io`, `c2-pcold`, `c2-pcold-advice`;
- `text`; and `tok`/`prof` for profiled runs.

**Parser** (`tools/c2_research_parse.py`):
- **Identity per model:** hash, libraries at every load against the build record, template, one text across both
  arms, repack on everywhere, no skip or head2 set, a `c2-io` line in every row.
- **Advice probe.**
- **Per model and arm:** the medians of tok/s, decode read MB and major faults, prefill read MB and ms, TTFT, mincore
  experts+rest before and after, MB marked, and Cached.
- **ABBA pairs, on minus off,** per block and per model, as median [min..max] n: tok/s, decode read MB, and the
  decode `read_bytes` change in %.
- **A side-by-side table** with one column per model.

## 5. Owner steps

1. Install `lab/dist-c2/vollie-c2-1.21.0+21-research-r21.apk`. It is an update over r20 (same package and key); the
   downloaded olmoe-Mi stays.
2. On Wi-Fi, open the app, go to Settings and tap **"Download kit r21 models: granite-Mi.gguf (3.2 GB, Wi-Fi)"**. Keep
   the app open; the screen stays on.
   - It needs about 3.4 GB free: 3.16 GB plus a 0.2 GB margin. Both kit files together take 6.2 GB.
   - If space is short, the app says how much is needed and how much is free.
   - A pause or a dropped connection resumes on the next tap.
   - After the download the file is hashed (sha256) before use.
3. Tap **"Run research kit r21"** on a cool phone, with the same charger state and the same habits as r20.
4. When it ends, tap **"Save research results to Downloads"** and send the file `Download/vollie-research-r21-<time>.json`
   (Files app, share). "Copy research results" still works as a fallback.

## 6. Time and download (*estimated*)

- **Download:** 3.16 GB at 10 to 40 MB/s on Wi-Fi is 1.5 to 5 minutes, plus about 0.5 minutes of hashing.
- **On r20's day** (*device*, `r20_2026-11-14_summary.json`): gate waits were mostly 30 to 45 s and an official run
  took about 10 s. Per block: load 0.5 to 1 min, warm-ups 0.4 min, and 10 gated runs × about 52 s ≈ 9 min.
  **4 blocks ≈ 40 minutes**, plus about 1 minute for the two end hashes.
- **Worst case, every gate times out:** 10 × 105 s per block ≈ 18 min, so **about 75 minutes. That is over 60.**
- **What to cut, in order:**
  1. the profiled pair of blocks 3 and 4: about −7 min at worst;
  2. all profiled runs: about −14 min at worst in total, which gives about 61 min;
  3. the second warm-up of every block.

  The ABBA scored runs and the fourth block are not cut. They are the comparison, and the G O O G order spreads the
  day's drift over both models.

## 7. Each model as a full subject: what generalised and what did not

| | Granite (`granite-Mi`) | OLMoE (`olmoe-Mi`) | generalised? |
|---|---|---|---|
| campaign 1's runtime loads and runs it | yes (campaign 1) | yes (memo-13 §7) | yes |
| campaign 1's default-on knobs bit-identical | yes (campaign 1) | yes (memo-13 §7, *host*) | yes |
| keep-warm (R56) | not run on Granite | closed: the phone does not keep the touched pages | not known for Granite |
| 0002 identity at three batch sizes | yes (*host*, §1.5) | yes (*host*, §1.5) | yes: one code path, no architecture names |
| marked set = the tracer's rarely routed set | yes, 767 pairs | yes, 243 pairs | yes |
| MB marked, official prompt | 740 | 645 | the same order; Granite's counted layers are 2.6× OLMoE's |
| speed effect on the phone | open (r21) | open (r21) | r21 answers it per model |

Each model gets its own verdict in R-r21, scored as pre-registered. A result that holds for one model only is reported
as that, not averaged.

## 8. What a third model needs (LFM2.5, `lfm2moe`; not built now)

**File.**
- The mix `lfm25-M.gguf` is 3,815,238,880 B (on the laptop); the publisher's Q4_0 is about 4.84 GB.
- With the mix, the three kit files take about 10.0 GB.
- For the table entry: upload with a pinned commit and sha256, extract the expert ranges with the gguf-py snippet used
  for memo-19's ranges (`_exps` tensors, data offset, adjacent runs merged), architecture `lfm2moe`, the file's own
  template, repack on, and the same baseline.

**Host checks before it goes into a kit:**
1. Campaign 1's runtime loads `lfm2moe`. `LLM_ARCH_LFM2MOE` is upstream (memo-02).
2. Identity of campaign 1's default-on knobs at `-ub 1`, `-ub 32` and `-b 512` against an unpatched build and against
   `GGML_MMID=0 GGML_Q3K_X4=0 GGML_Q3K_KERN=0 GGML_MMID_FUSE=0 GGML_CONV_RING=0`. Above all **`CONV_RING`**: memo-02 §2
   found that LFM2's graph meets the matcher but then runs the generic loop (`ns1 = 2`), which never ran in campaign 1.
   Expect a `conv-ring layers=18` line and per-chunk equality. Repeat with and without repack.
3. 0002's identity at the same three sizes, and the greedy text with the knob on and off.
4. The tracer slice check (§1.5) on LFM2.5's sigmoid + bias routing. The counts come from `ids`, so the gating
   function does not matter, but this has not been run.
5. The template render of the official prompt, and the prompt's token count.
6. Run time: two more blocks. Forward plus backward gives 6 blocks, about 60 minutes on r20's day and about 113 minutes
   in the worst case. A three-model kit would then need fewer runs per block, or two sessions.

**Effort** (*estimated*):
- about half a day of host work if `CONV_RING`'s generic path is identical;
- one more day if it is not (fix it in a new series patch, or gate it off for `lfm2moe`, and run the identity again);
- plus the upload and one table entry.

## 9. Not verified

- **Anything on Android:** whether `MADV_COLD` exists on the phone's kernel and is allowed on the app's own file
  mapping, what the advice costs there, and whether it changes what the phone reclaims during a 90 s wait. Note that
  5.4's COLD acts only on pages mapped in this process (mapcount 1); the prefill has just mapped them.
- That the decode does not need many of the marked slices on the phone. On the host, the marked set is the prompt's
  rarely routed set; memo-18's replay is the only estimate of overlap.
- A prompt longer than `n_ubatch` (512) counts per micro-batch. The kit's prompts are 155 and 156 tokens.
- **Known limit:** a 2- to 7-token batch marks its whole routed set (the host warm-up shows 360 slices). The kit never
  runs one with the knob on. A shipping version needs a floor on the batch size.
- The download code (HTTP Range, MediaStore save) was compiled and analysed, not run on a phone.
- The run time is an estimate from r20's gate waits.
