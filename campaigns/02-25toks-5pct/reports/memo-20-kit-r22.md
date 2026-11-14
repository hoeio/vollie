# Memo 20: research kit r22, skip and head (arm "num") and holding the pages through the gate wait, two models

Agent model: claude-opus-5-5

Date 2026-11-15. Author: Tom (code). **Nothing here ran on a phone, and no phone speed is stated.** The APK was built
and inspected on the laptop. Labels: *host* = Apple M4 Pro, relative only; *estimated* = arithmetic; *device* = quoted
from an existing report, with the report named. The levers and their settings come from R59 (*host*); the residency
problem from R56 and R58 (*device*).

## 1. Runtime: campaign 1's runtime + `patches/llama.cpp-c1-olmoe/0001` to `0005`

Kit r21's runtime (`patches/llama.cpp/0001` to `0003`, `patches/llama.cpp-c1-olmoe/0001-keep-warm`, `0002-prefill-cold`)
plus R59's two patches and one counters-only patch:

| patch | what | env |
|---|---|---|
| `0003-c2-expert-skip.patch` | the c2 skip rule on campaign 1's skip mask (OR with campaign 1's rules) | `VOLLIE_SKIP_MASS`, `VOLLIE_SKIP_STATS` |
| `0004-c2-head2-untied.patch` | the two-stage head for an untied `output.weight` (OLMoE; repacked q6_K) | `VOLLIE_HEAD2` |
| `0005-kit-lines.patch` | per-prompt lines for the kit; counters only | none of its own (prints under `GGML_C2_IO=1`) |

**0005 is an addition beyond "r21 + 0003, 0004".** 0003's and 0004's statistics are printed once per process, at
the end; the kit needs them per run and split into prefill and decode, and it needs an I/O source that the app can read
(r21's `/proc/self/io` read -1 in every row, R58). 0005 adds three lines, each tagged `run=<VOLLIE_RUN>` and printed
with 0002's `c2-io` line (at the next prompt or when the backend is freed); 0002's line keeps its format:

- `c2-io-blk run= inblock=a/b/c`: `getrusage` `ru_inblock` (512-byte blocks) at the prompt's start, its first decode
  graph and its end.
- `c2-skip-run: run= sum= max= mass= c1_skip= c1_topp= c1_gap= prefill_tokens= prefill_pairs_skipped=% … decode_tokens= …`:
  0003's counters (every rule, campaign 1's included) split at the first decode graph. Only when pairs were counted.
- `c2-head-run: run= c2_k= c2_calls= c2_cols= c2_exact_rows_per_col= c2_copies= c2_copy= c1_m= c1_calls= c1_copies= c1_copy=`:
  both two-stage heads in this prompt. Only when one ran. Campaign 1's head gets three counters, no arithmetic change.

The library contains no architecture name in any of these paths; Granite and OLMoE take the same code.

### 1.1 Android build

`tools/c2_build_kit.sh lab/llama.cpp-c1r22 lab/app-c2 patches/llama.cpp patches/llama.cpp-c1-olmoe`, which checks the
tree file by file against 620d2225c + the eight series patches and then runs campaign 1's `android/build_cpu.sh`
unchanged: NDK 28.2.13676358, `android-28`, arm64-v8a, Release, flexible page sizes, `GGML_NATIVE=OFF`,
`GGML_CPU_ALL_VARIANTS=ON` (the variant `android_armv8.2_2` = armv8.2-a+dotprod+fp16 is the one shipped), KleidiAI on,
`GGML_BACKEND_DL=ON`, `GGML_OPENMP=OFF`, `GGML_LLAMAFILE=OFF`, stripped. These are r21's flags. The build exits 0;
the only warnings come from vendored code (KleidiAI, cpp-httplib).

| | bytes | sha256 |
|---|---|---|
| `vollie-c2-1.22.0+22-research-r22.apk` | 17,208,611 | `e2e59adab40096983130573928a5f069b67eda985a6acb18bee36fb828f86914` |
| `libggml-cpu-android_armv8.2_2.so` | 1,760,256 | `5d7c550cf37597c58eebd148e6763582ec1b1e47198dd603e5634b0e43de46d7` |
| `libllama.so` | 4,285,208 | `815f34534777edd2a080a357529dc57ad9f65d4f1644138218b1b07e70bf1c02` |

- The script checked that the APK carries exactly these two libraries and the build record (`assets/kit_build.json`,
  which lists the eight series patches and app patches `0001` to `0006` with their sha256; full text in
  `lab/dist-c2/BUILD-r22.txt`).
- `libllama.so` has no source change against r21 (no new patch touches `src/`). It is 192 B larger because the
  embedded `__FILE__` paths name the build tree `llama.cpp-c1r22` instead of `llama.cpp-c1pc` (`strings` diff).
- The CPU module carries the strings of all three new lines and of both knobs.

### 1.2 Host checks (*host*)

**Off = r21's arithmetic.** R59 showed 0003 and 0004 identical with their knobs unset (per-chunk, three sizes, both
models). 0005 against the 0001-0004 tree, 4 chunks, `-t 8`, per-chunk PPL equal at `-b 512`, `-ub 32` and `-ub 1` in both
arms of both models. Arm envs as in §3. Examples:

| model | arm | setting | chunks 1 to 4 |
|---|---|---|---|
| OLMoE | off | `-b 512` | 8.9646 / 10.4244 / 9.2074 / 9.4783 |
| OLMoE | num | `-b 512` | 9.1525 / 10.7417 / 9.8247 / 10.1338 |
| OLMoE | num | `-ub 1` | 9.0820 / 10.7699 / 9.8473 / 10.1310 |
| Granite | off | `-b 512` | 4.9584 / 6.6520 / 6.1119 / 6.6024 |
| Granite | num | `-b 512` | 5.1565 / 7.0463 / 6.5935 / 6.9756 |
| Granite | num | `-ub 1` | 5.1143 / 7.0451 / 6.5571 / 6.9776 |

The off values are memo-19's (§1.5), so the off arm is r21's arithmetic on the host build. The Android library was
built from the same tree with the flags above; it was not run.

**Budgets** (R59, full wikitext-2, *host*): OLMoE num 12.1053 at `-b 512`, 12.0946 at `-ub 1`; Granite num 9.5173 and
9.5076, budget 9.6879. Both inside at both sizes.

**Texts.** Official prompt, 32 greedy tokens, `llama-completion -cnv -st -t 3 -ctk q8_0 -ctv q8_0 -fa on -c 4096
--repeat-penalty 1.1 --repeat-last-n 64`, the c1r22 host build:

| model | off md5 | num md5 | num decode: expert bytes skipped, MB/token read | head |
|---|---|---|---|---|
| OLMoE | `2af52b9b…` | `3604009d…` | 13.03 %, 45.1 | `c2_calls=33 c2_copy=1` |
| Granite | `ce8fa3d4…` | `4826d783…` | 41.03 %, 99.8 | `c1_calls=33 c1_copy=1` |

The two arms give different texts on both models (as a numerics change should). The parser checks one text per arm
per model on the phone. The host `llama-completion` render differs from the app's render, so these hashes are not the
phone's.

## 2. The hold factor: keep the model's pages through the gate wait

### 2.1 What exists, and why it is not reused

- **Keep-warm (`0001-keep-warm`, R56, *device*):** a helper thread touches the file's pages. It ran (18 passes,
  0.26 to 0.65 s CPU each), but the system still took pages back between passes (1,915 to 1,540 MB in one wait), and
  the runs were no faster (11.69 against 11.72 tok/s). A touch costs CPU on every resident page every pass, and it
  cannot bring back a page that is already gone; it only delays reclaim of the ones still there.
- **`mlock`:** pins pages, but on Android an app's `RLIMIT_MEMLOCK` is normally 64 KB (*not checked on this phone*),
  far below 2 to 2.7 GB. Pinning 2.7 GB of an 8 GB phone would also push other apps out. Kit r22 records the limit
  (`memlock_rlimit`, `getrlimit(RLIMIT_MEMLOCK)`); if the phone allows it, a later kit can test it.
- **A slow periodic touch:** keep-warm at a lower rate. It has the same flaw as keep-warm (no read-back).

### 2.2 The mechanism (app side, no runtime change, no env)

1. At the end of every run, the app stores the file's `mincore` vector (one byte per page): "wanted" = resident at the
   end of the previous run of this model.
2. During the gate wait, at each gate poll (every 5 s, r16's rule unchanged), it takes `mincore` again, finds the
   wanted pages that are gone and issues `posix_fadvise(fd, off, len, POSIX_FADV_WILLNEED)` on each run of them.
   WILLNEED starts asynchronous read-ahead into the page cache and returns.
3. After the gate passes: one more pass, then it polls `mincore` every 200 ms until 98 % of the wanted pages are back
   or 5 s have passed.

The phone's reclaim can still take the pages again; what this buys is that the pages lost during the wait are
read back before the run starts instead of faulting in during the first tokens. It touches only the file that the
next run uses. It reads, on the phone's storage, only the MB that were lost.

### 2.3 Cost (*estimated*; measured per run in `hold_stats`)

- **CPU per pass:** one `mincore` on 3.07 to 3.16 GB (about 770,000 pages at 4 KB), plus a Dart loop over the two
  vectors: some ms per pass. At one pass per 5 s that is well under 1 % of one core.
- **Read-back:** kernel read-ahead of the lost MB; R58's waits lost a few hundred MB (*device*, R58). The CPU is the
  kernel's page allocation and I/O completion, order 0.1 s per few hundred MB. The phone does not compute during it.
- **After the gate:** at most 5 s of mostly idle waiting (25 `mincore` polls).
- No helper thread; no CPU between passes; nothing in the runs themselves.

Each hold row reports `passes`, `lost_mb_max`, `advised_mb`, `advise_calls`, `advise_fails`, `pass_ms`, `cpu_us`
(the process's `getrusage` user+system time across the passes), `inblock` (blocks read during the passes),
`final_ms` and `back_mb`, plus `want_mb` and `wait_lost_mb` (measured before the final pass, so it shows what the wait
took in both hold states).

## 3. Arms: the exact env per arm and model

Every gated and warm-up run sets the baseline, then the arm. `GGML_PREFILL_COLD` and `GGML_KEEP_WARM` are set nowhere.

| model | arm | env |
|---|---|---|
| both | off | `GGML_TINY_OP=262144 GGML_EXPERT_IO=1 VOLLIE_SAMPLER_BYPASS=1 GGML_C2_IO=1` (r21's knob-off arm) |
| OLMoE (`olmoe-Mi`) | num | off + `VOLLIE_SKIP_MASS=900 VOLLIE_HEAD2=1024 VOLLIE_SKIP_STATS=1` |
| Granite (`granite-Mi`) | num | off + `GGML_EXPERT_SKIP=130 GGML_HEAD2=2048 GGML_HEAD2_Q=q4_0 VOLLIE_SKIP_STATS=1` |
| both | profiled runs | the arm's env + `GGML_CPU_PROF=1 GGML_CPU_AB_TOKEN=1 GGML_CPU_AB=knob:MMID_FUSE+CHUNK2:1:1` |

`VOLLIE_SKIP_STATS=1` only counts (it prints the `c2-skip-run` line). Repack is on at every load, as in r21. Hold is not
an env; it is the row's `hold` flag.

## 4. Run list (as built; 48 runs, 40 gated)

Blocks **Granite, OLMoE, OLMoE, Granite**. Each block:

1. Load its model (forced), repack on; mincore of both files and meminfo before the load.
2. Two unscored warm-ups, arm off, no gate, no hold.
3. Quad 1: off, num, num, off (scored, gated).
4. Quad 2: off, num, num, off (scored, gated).
5. One profiled run per arm, gated, hold on: off then num in blocks 1 and 3, num then off in blocks 2 and 4.

Hold is on for quad 1 in blocks 1 and 3, and for quad 2 in blocks 2 and 4. So Granite gets hold first in block 1 and
hold last in block 4; OLMoE gets hold last in block 2 and hold first in block 3.

| block | model | quad 1 | quad 2 | profiled |
|---|---|---|---|---|
| b1 | granite | hold on | hold off | off, num |
| b2 | olmoe | hold off | hold on | num, off |
| b3 | olmoe | hold on | hold off | off, num |
| b4 | granite | hold off | hold on | num, off |

Per model: 16 scored runs (8 off, 8 num; 8 with hold), 4 profiled, 4 warm-ups. Hold on in 24 runs (16 scored + 8
profiled). The first gated run of a block has a "wanted" vector from the block's second warm-up. Run ids:
`<model>_<arm>`, `_h` with hold, `_prof`, `_wu`. Kept from r21: r16's gate (below 0.65, read every 5 s, at most 90 s),
the prompt, 32 greedy tokens, mincore before and after, meminfo (`Cached`), clock, faults, the save-to-file button.
`test/kit_lines_test.dart` checks this list from the model table.

**Note on the head copies.** Both head knobs build their copy on first use and free it when the knob is unset, so
every num run rebuilds it in its prompt (OLMoE about 58 MB, Granite about 87 MB, *host*). This is in TTFT, not in the
decode tok/s; the `c2-head-run` line shows `copies=1` per num run.

## 5. Report schema (additions over r21)

**Top level:** `kit: r22`, `baseline_note` (the arms, no PREFILL_COLD, no KEEP_WARM), `hold_rule` (0.98, 5,000 ms,
200 ms, the advice), `memlock_rlimit` (soft, hard), `models[<id>].num`.

**Per row:** `arm`, `hold`, `hold_stats` (§2.3), `want_mb`, `wait_lost_mb`, `inblock_read_bytes` and
`inblock_read_bytes_to_first_chunk` (the app's own `getrusage(RUSAGE_SELF)` `ru_inblock` × 512 over the run),
`run_cpu_ms`, `io_wait_ticks` (`/proc/self/stat` field 42, delay-accounting block-I/O ticks; may read 0 if the kernel
does not account it), and the runtime's `c2-io-blk`, `c2-skip-run`, `c2-head-run` lines.

**I/O sources other than `/proc/self/io`:** `ru_inblock` from `getrusage`, read by the runtime (0005, per phase) and by
the app (per run). It counts the same kernel I/O accounting as `read_bytes` but through a system call. If the phone's
kernel builds without task I/O accounting it reads 0; major faults per phase remain the fallback (as in r21).

**Parser** (`tools/c2_research_parse.py`, kit r22):
- **Checks per model:** r21's checks, plus one text per arm, off text ≠ num text, the num env present in every num row
  and absent from every off row, no PREFILL_COLD/KEEP_WARM, a `c2-skip-run` line and an active head copy in every num
  row; the I/O source rows and the memlock limit.
- **Per model × arm × hold:** tok/s median [min..max], MB resident before, MB lost in the wait, gate seconds, decode
  major faults and inblock MB, prefill ms, TTFT, skip %, MB/token skipped, hold read-back MB, CPU ms, ms after the gate,
  `Cached`.
- **num − off per model:** mean over ABBA quads of mean(num) − mean(off), ± SE; all quads, hold-on quads, hold-off
  quads.
- **Hold effect, off arm:** hold minus no hold, tok/s and MB resident before the run, Welch SE; all runs and the runs
  after a gate wait of at least 30 s (`--wait-s`). The num arm is shown for reference.
- **Hold cost per run:** median [min..max] of read-back MB, CPU ms, passes, ms after the gate, blocks read, fadvise fails.
- `--selftest` passes, with a synthetic r22 case on two models.

## 6. Owner steps

1. Install `lab/dist-c2/vollie-c2-1.22.0+22-research-r22.apk`. It is an update over r21 (same package and key); both
   downloaded files (`olmoe-Mi.gguf`, `granite-Mi.gguf`) stay. No new download.
2. Open the app, go to Settings, tap **"Run research kit r22 (~35-80 min)"** on a cool phone, with the same charger
   state and the same habits as r21. Keep the app in front; the screen stays on.
3. When it ends, tap **"Save research results to Downloads"** and send `Download/vollie-research-r22-<time>.json`.

## 7. Time and storage (*estimated*)

- **Storage:** the two kit files, 6.2 GB, already on the phone after r21; the report is a few MB.
- **Typical:** r21 had the same run count and gate (48 runs, 40 gated) and took 29.6 minutes (*device*,
  `r21_2026-11-14_summary.json`, 21:40:42 to 22:10:19). Hold adds at most 5 s after the gate in 24 runs, at most
  2 minutes. **About 32 to 35 minutes.**
- **Worst case, every gate times out:** memo-19's 75 minutes plus 2 minutes of hold: **about 80 minutes.**
- **What to cut, in order:** the profiled runs of blocks 3 and 4, then all profiled runs, then the second warm-up. The
  quads and the fourth block are the comparison and are not cut.

## 8. Each model as a full subject

| | Granite (`granite-Mi`) | OLMoE (`olmoe-Mi`) | generalised? |
|---|---|---|---|
| skip rule | campaign 1's `GGML_EXPERT_SKIP=130` | c2's `VOLLIE_SKIP_MASS=900` | no: different rules, one per model, both screened inside budget at two sizes (R59) |
| two-stage head | campaign 1's tied head, 2,048 rows, Q4_0 | 0004's untied head, 1,024 rows | the idea is shared, the code path differs (tied vs untied) |
| host decode bytes skipped | 41.03 % | 13.03 % | not the same size of lever |
| hold | the same code | the same code | answered per model by r22 |
| 0005 identity, three sizes, two arms | yes (*host*) | yes (*host*) | yes |

## 9. Not verified

- **Anything on Android:** whether `posix_fadvise(WILLNEED)` on the app's file reads pages back on this kernel, how
  fast, and whether reclaim takes them again before the run; whether `ru_inblock` and `/proc/self/stat` field 42 are
  non-zero for the app; the `RLIMIT_MEMLOCK` value. The hold code (FFI to `mincore`, `posix_fadvise`, `getrusage`,
  `getrlimit`) compiles, passes `flutter analyze` and is in the release APK, but it was never executed: the app runs
  only on Android, and the unit tests cover the run list and the line parsing, not the hold loop.
- On macOS `ru_inblock` read 0 in the host runs, so the `c2-io-blk` values were parsed but never seen non-zero.
- Whether the num arm's speed gain on the phone matches the host's skipped bytes; the head copies are rebuilt in
  each num run's prompt (§4).
- The Android libraries were built, not run; the identity is the host build's.
- The run time is an estimate from r21's day.
