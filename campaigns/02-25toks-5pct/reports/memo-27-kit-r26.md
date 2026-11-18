# Memo 27: research kit r26, the kv clear on the untagged address (patch 0009) with r25's protocol simplified by R64

Agent model: claude-opus-5-5

Date 2026-11-18. Author: Tom (code). **Nothing here ran on a phone, and no phone speed is stated.** The APK was built
and inspected on the laptop. Labels: *host* = Apple M4 Pro (macOS, or a Linux aarch64 container on it), relative only;
*estimated* = arithmetic; *device* = quoted from an existing report. Kit r26 is kit r25 ([memo-26](memo-26-kit-r25.md))
rebuilt from the app source with patch 0009 in the runtime and the protocol cut down to the questions R64 left open.
Kit r25b (R65) was the r25 APK with only `libllama.so` swapped, made while `~/Documents` was unreadable; r26 is the
proper build of the same runtime.

## 1. The runtime change (0009)

R64 (*device*): every `c2-kvclear` line of every kv run said `released=0 why=map`. Inferred cause: Android arm64 heap
pointers carry a tag in the top byte (scudo, top-byte ignore), so the buffer address matched no line of
`/proc/self/maps`. `patches/llama.cpp-c1-olmoe/0009-c2-kv-untag.patch` (written by the lead, 2026-11-17) clears the top
byte on aarch64 before the page range, the maps check and `madvise`, memsets the edges through the original pointer,
and appends `tag=0x..` (the address's top byte) to the `c2-kvclear` line. Same knob, `VOLLIE_KV_DONTNEED=1`; declared
bit-identical. It changes `libllama.so` only.

Kit line now: `c2-kvclear run=<VOLLIE_RUN> kind=kv|rs on=0|1 bytes=<buffer> released=<bytes> why=ok|off|os|notcpu|small|map|madvise us=<µs> tag=0x<top byte>`.
On the phone with the knob the expected line is `why=ok`, `released` about 285 MB (OLMoE `kv`) or about 17.8 + 58.1 MB
(Granite `kv` + `rs`), and a non-zero `tag` if the tagged-pointer reading is right. `tag` is computed only when the knob
is on, so plain runs print `tag=0x00`.

## 2. Host checks (*host*, 2026-11-18)

Tree `lab/llama.cpp-c1r26` = 620d2225c + `patches/llama.cpp/0001-0003` + `patches/llama.cpp-c1-olmoe/0001-0009`; every
patch applies in order with `git apply --check` (0009 on 0008). Host builds without KleidiAI, `armv8.2-a+dotprod+fp16`,
shared libraries: macOS (AppleClang 16, no OpenMP) and an Ubuntu 24.04 aarch64 container (gcc 13.3, `GGML_OPENMP=OFF`,
kernel 6.10.14-linuxkit). Kit off env `GGML_TINY_OP=262144 GGML_EXPERT_IO=1 VOLLIE_SAMPLER_BYPASS=1 GGML_C2_IO=1`,
`-c 512`, wikitext-2-raw test, knob unset against `VOLLIE_KV_DONTNEED=1`.

**Per-chunk PPL, knob off against on** (20 chunks at `-b 512` and `-b 512 -ub 32`, 8 at `-b 512 -ub 1`; first four
values and the last shown):

| platform | model | `-b 512` | `-ub 32` | `-ub 1` | clears with the knob |
|---|---|---|---|---|---|
| macOS (`-t 8`) | olmoe-Mi | equal, 20 of 20 (8.9646 10.4244 9.2074 9.4783 … 12.5367) | equal, 20 (… 12.5029) | equal, 8 (… 10.2291) | `memset` (`why=os`), 21 / 21 / 9 lines |
| macOS | granite-Mi | equal, 20 (4.9584 6.6520 6.1119 6.6024 … 9.7318) | equal, 20 (… 9.7486) | equal, 8 (… 8.5192) | `why=os` |
| Linux container (`-t 6`) | olmoe-Mi | equal, 20 (8.9607 10.4468 9.2244 9.5063 … 12.5459) | equal, 20 (… 12.4817) | equal, 8 (… 10.2878) | `why=ok`, 67,104,768 of 67,108,864 B released, every clear (21 / 21 / 9) |
| Linux | granite-Mi | equal, 20 (4.9811 6.6764 6.1185 6.6045 … 9.7685) | equal, 20 (… 9.7453) | equal, 8 (… 8.5720) | `why=ok` every clear: KV 4,194,304 (`-b 512`) or 4,190,208 B, recurrent 58,056,704 B |

- The perplexity tool clears with `data = true` before every chunk, so each Linux on-run went through the `madvise` path
  21 or 9 times.
- The first four chunks equal memo-26's values on both systems (0001-0008 build, same flags): adding 0009 changed no
  value at the chunks both runs have.
- macOS and Linux values differ from each other (compiler); the comparison is within one build.

**Greedy text on the app's settings** (memo-26's test program: `n_ctx` 4096, q8_0 K and V, flash attention, three
prompts in one process, each after `llama_memory_clear(mem, true)`, 32 greedy tokens), knob off and on:

| platform | OLMoE | Granite | clears with the knob |
|---|---|---|---|
| macOS | three texts equal, off = on (md5 `6716c00a…`) | equal, off = on (`041ac181…`) | `why=os` |
| Linux | three texts equal, off = on (`e128edc3…`) | equal, off = on (`041ac181…`) | OLMoE `kv` 285,208,576 of 285,212,672 B, Granite `kv` 17,821,696 + `rs` 58,056,704 B, `why=ok` each of the 3 clears |

On glibc the heap is untagged, so these lines print `tag=0x00`; the tagged case is the next check.

**Tagged-pointer test** (`tools/c2_kvclear_tag_test.cpp`, container): `maps tagged=0 untagged=1 madvise=0 nonzero=0
tag=0xb4 released=67104768`, exit 0: the maps check fails on the tagged range and passes on the untagged one, `madvise`
succeeds, every byte reads zero.

**Not checked:** Android itself (bionic and scudo's tag values, kernel 5.4.274); whether the phone's tag is in fact
non-zero. The kit line shows both.

## 3. Protocol: r25 simplified by R64

R64 answered the gate question (plateau not shown on either model) and Granite's num question (+2.68 ± 0.71 tok/s,
pass), and R63 §4 parks OLMoE's `MASS=900` after three rounds without a shown gain. The kv factor was void. So r26 keeps
only the kv question and one arm per model:

- **One arm per model** (`Research.armOf`): Granite `num` (`GGML_EXPERT_SKIP=130 GGML_HEAD2=2048 GGML_HEAD2_Q=q4_0
  VOLLIE_SKIP_STATS=1`), OLMoE `off`. Every run of a model's blocks is in that arm: warm-ups, scored and profiled runs
  (the warm-ups now warm the arm that is scored; in r25 they were always `off`).
- **OLMoE load env kept:** `VOLLIE_HEAD2_REPACK=1` at every OLMoE load, as r23-r25. Reason: OLMoE's off arm stays the
  same code path as in R62-R64 (head in CPU_REPACK, R61 found it inside the budget and without effect on PPL), so its
  numbers compare with those rounds; dropping it would be a second change in a round about one factor. Without
  `VOLLIE_HEAD2` the head copy is not used.
- **r16's gate only** (below 0.65 or 90 s) in every gated run. The plateau code and its report fields stay (unused).
- **Kept from r25:** blocks G O O G, per block a load, 2 warm-ups (no gate, no hold), 8 scored runs, 2 profiled runs
  (r16 gate, the model's arm, profiling env, kv off), 3 threads, hold in every gated run, charger guard, idle baseline,
  every counter and field of r25.
- **kv factor in two ABBA quads per block:**

| block | model | arm | scored runs 1-4 | scored runs 5-8 |
|---|---|---|---|---|
| b1 | granite | num | on off off on | off on on off |
| b2 | olmoe | off | off on on off | on off off on |
| b3 | olmoe | off | on off off on | off on on off |
| b4 | granite | num | off on on off | on off off on |

  Each model starts one block with kv and one without; kv sits on positions 1 4 6 7 or 2 3 5 8 (mean 4.5 either way).
  Per model 8 kv and 8 plain scored runs in 4 quads. The lead's task text gave "on off off on / off on on off"; the
  reversal in even blocks is this memo's choice (balances the first scored run after the warm-ups per model; one line
  to undo in `_block`).

48 runs (44 specs, warm-ups have 2 reps), 40 gated, as r25.

**Names for the pre-registration (taken from the built app code):**

| | value |
|---|---|
| kit / app / proto | `r26` / `1.26.0+26` / `r26` |
| always set | `VOLLIE_LINES=stderr`, `VOLLIE_RUN=<id>.<rep>` |
| Granite env, every run | `GGML_TINY_OP=262144 GGML_EXPERT_IO=1 VOLLIE_SAMPLER_BYPASS=1 GGML_C2_IO=1 GGML_EXPERT_SKIP=130 GGML_HEAD2=2048 GGML_HEAD2_Q=q4_0 VOLLIE_SKIP_STATS=1` |
| OLMoE env, every run | `GGML_TINY_OP=262144 GGML_EXPERT_IO=1 VOLLIE_SAMPLER_BYPASS=1 GGML_C2_IO=1` |
| OLMoE load env | `VOLLIE_HEAD2_REPACK=1` (during loads only); Granite none |
| kv runs add | `VOLLIE_KV_DONTNEED=1` |
| profiled runs add | `GGML_CPU_PROF=1 GGML_CPU_AB_TOKEN=1 GGML_CPU_AB=knob:MMID_FUSE+CHUNK2:1:1` |
| run ids | `granite_num_wu`, `granite_num_r16`, `granite_num_r16_kv`, `granite_num_prof`; `olmoe_off_wu`, `olmoe_off_r16`, `olmoe_off_r16_kv`, `olmoe_off_prof` |
| report | `kv_rule`, `arm_rule`, `models.<m>.arm`, spec `kv`, `gate_rule` `r16` / `none` |
| kit line | `c2-kvclear` with `kind`, `on`, `bytes`, `released`, `why`, `us`, `tag` |

## 4. Parser

`tools/c2_research_parse.py` reads proto r26: r24's checks; arms per model, gate rules, kv id suffix and placement, lines
without `tag=`; void check (released > 0 without kv) and the kv runs releasing 0; texts kv against plain per model.
Tables: kv minus plain per model, paired per ABBA quad (mean(kv) − mean(plain), ± SE over the 4 quads) and Welch, for
r25's metrics (pre-prompt majors, pswpin start → first chunk, TTFT, decode graph ms/token, tok/s, VmSwap at end,
released MB), with an effect readout in R65 §4's wording; the mechanism lines (kv runs with `why=ok` in every clear
line out of all kv runs, counts per (kind, why) and (kind, tag) for kv and plain runs, released MB per kv run); r24's
stats line per model, arm and kv; charger runs, swap per gated run, hold table. A kit r25 report whose lines carry
`tag=` (kit r25b, R65) gets the mechanism lines after r25's tables. Output on the stored r22, r23 and r25 reports is
byte-identical to the version before this change; the selftest covers r26 (pass, a `why=map` fallback, a releasing plain
run, a line without `tag=`) and the r25b case.

## 5. Build

`tools/c2_build_kit.sh lab/llama.cpp-c1r26 lab/app-c2 patches/llama.cpp patches/llama.cpp-c1-olmoe`, app patches
`0001` to `0010`, `android/build_cpu.sh` flags unchanged. App unit tests and `flutter analyze` pass.

| | bytes | sha256 |
|---|---|---|
| `vollie-c2-1.26.0+26-research-r26.apk` | 17,228,499 | `e0adfb8f1ae4e08c99e4effecd6e912e8462a26a2eadfc6a22a5060278c83854` |
| `libggml-cpu-android_armv8.2_2.so` | 1,767,336 | `b93c720d43ec3cb5ca3d25bd722b50048b77a30f41e00d5c62545640b7055085` (r25: `d6b7b3c9…`) |
| `libllama.so` | 4,287,504 | `497c6d2822f211e74afc736f999a41c5d9adffcfce991f789a376cc811ceaf6f` (r25: `17a197d4…`; r25b: `4c651c4b…`) |

- Signed with the same debug key as r25 (certificate SHA-256 `c49dde85…69fbb1716`); package
  `com.insaneutils.vollie.c2`, versionCode 26, versionName 1.26.0.
- **The CPU module's hash differs from r25's although its sources are the same** (0009 touches `src/llama-impl.cpp`
  only): the build tree's path (`llama.cpp-c1r26` instead of `-c1r25`) is in the module's `__FILE__` strings, which moves
  string addresses. The disassembly of the two modules has the same length (297,948 lines) and differs only in `adrp`,
  `adr` and `add` lines (string addresses), as the lead found for r25b (*host*, inferred: no instruction other than an
  address computation differs).
- `libllama.so` contains the `c2-kvclear … tag=0x%02x` format and imports `madvise`.
- The build record (`assets/kit_build.json`, `lab/dist-c2/BUILD-r26.txt`) lists the twelve series patches (0009 sha256
  `73426173…6244dfc`) and app patches `0001` to `0010` (0010 sha256 `2b0d5395…bb944a4`).
- The tree's new file `ggml/src/ggml-cpu/vollie-release.h` (from 0001) has to be marked with `git add -N` before
  `c2_build_kit.sh` runs, as in the r25 tree; otherwise the script stops with "the tree's changed files differ".

## 6. Owner steps

1. Before the session, close background apps and unplug the charger (the start screen says so; the kit pauses on a
   charger).
2. Install `vollie-c2-1.26.0+26-research-r26.apk` (from the HF repo `r26/`). It is an update over r25 / r25b and keeps
   both models.
3. In Settings, tap **"Run research kit r26 (~35-75 min)"**. The first 30 s are an idle baseline.
4. When it ends, tap **"Save research results to Downloads"** and send `Download/vollie-research-r26-<time>.json`.

## 7. Time (*estimated*)

Same run count and waits as r25 with every gate on r16's rule: in R64 every r16 wait ran to 91 s (*device*), so about
60 to 65 minutes on such a day, 32 to 40 when the gates are reached (memo-23 §8's arithmetic).

## 8. Never run on a phone

0009's untagging on bionic and scudo; whether the phone's buffers are tagged at all (the `tag` field answers it); the
fault and TTFT effect of the clear; the warm-ups in Granite's num arm (r25 warmed with off).
