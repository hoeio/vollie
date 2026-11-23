# c3 memo-12 work, round 16 (Bonnie): the phone's stock configuration as one table; can a width-2 pass share the KV walk

- **Date:** 2026-11-23. **Author:** bonnie, `claude-fable-5-2`. Follows [bonnie-joins-r15.md](bonnie-joins-r15.md). Not
  committed. No verdict here. **Ebony has not reviewed this file.**
- **Labels.** *device*, *host*, *estimated*, *source* as before. Triples are Yi / Llama-2 / DeepSeek.
- Nothing was measured this round. It is source reading: the kit's app (`lab/app-c3`, version 1.35.0+35, with
  `third_party/llamadart` 0.15.6), the kit's runtime tree (`lab/llama.cpp-c3r32`), and the freeze's build
  (`lab/llama.cpp-c3/build`). Line numbers are as read today. No web source.
- **Scope note.** No campaign-3 kit has loaded a model yet (R73 is held). "The phone's configuration" below is
  what the app and its research code would do today, with campaign 2's kit settings as they stand in the same
  source. What R73's kit sets is the lead's to fix; the table is the list to fix it from.

## 0. Answer first

1. **A fourth mismatch, and the largest: by the app's own rule a dense 7B file loads without KleidiAI or any
   repack.** `canCopyWeights` allows the faster layouts only if total RAM minus the copy is at least 5 GB
   (`device.dart` line 591). On this phone (7.75 GB total, R72's header) that admits copies up to about 2.7 GB; the app counts the whole
   file, 3.5 to 4.0 GB.
   Campaign 1 wrote the rule after the phone was killed copying a 3.6 GB model. **So every campaign-3 speed
   number (R72's body, K0 6.90 / 6.21 / 6.22) describes a configuration the app refuses unless a kit forces it**
   (`_repack`, `assistant.dart` lines 455 to 456). With the rule as it is, the phone runs the stock kernel:
   about 4.5 / 4.0 / 4.1 tok/s (r6 §1.3, *estimated*).
2. **The table (§1) has 27 rows; 8 differ between the phone and the freeze in a way that can touch numerics.**
   Four are already known (KleidiAI, the head, the KV type, flash attention). New: the repack rule above, in
   substance; the llamafile option, Apple's Accelerate on the laptop and the compiler, by rounding only. Context
   size and the prompt's batch size differ too; they change memory and timing, not logits.
3. **What the lead needs to decide once, before the first model-loading round** (§1.3): repack forced on or not,
   the head knob, the KV type, n_ctx, threads. Then one host reference with exactly those, and the same flags in
   every later check.
4. **Sharing the KV walk at width 2:** the two positions already read the same cache rows, and the second read
   comes from the CPU's own caches, so there are no bytes to save. What can be shared is the conversion of each
   Q8_0 value row to float, done once per query today (*source*). **Saving: 1 to 3 ms per pass on Llama-2 and
   DeepSeek, less on Yi** (*estimated*, wide because nothing has profiled this loop on the phone). It is exact:
   each query keeps its own operations in the same order. An F16 value cache removes the conversion altogether
   for 26 MB more bytes per token.

## 1. The phone's stock configuration

"Freeze" = `tools/c3_freeze.sh` → `tools/ppl.sh` → `lab/llama.cpp-c3/build/bin/llama-perplexity -c 512 -b 512
-t 8 -ngl 0` (and `-ub 1`). "Numerics" = can the row change a logit: **yes**, **round** (rounding order only),
**no**.

### 1.1 Build

| # | item | phone (kit) | where set | freeze | numerics |
|---|---|---|---|---|---|
| 1 | KleidiAI | **on** | `lab/app-c3/tool/build_cpu.sh:19` (`-DGGML_CPU_KLEIDIAI=${KLEIDI:-ON}`) | **off** (`CMakeCache.txt`) | **yes** (r9 to r11) |
| 2 | CPU instruction set | variant `android_armv8.2_2` = dotprod + fp16 arithmetic | `build_cpu.sh:19, 26`; `ggml/src/CMakeLists.txt:528` | `armv8.2-a+dotprod+fp16` | no (same set) |
| 3 | repack type compiled in | yes (default) | – | yes | – |
| 4 | llamafile kernels | **off** | `build_cpu.sh:20` | **on** | round: they take matmuls no extra buffer type owns (float weights); a dense model has none of size |
| 5 | Apple Accelerate | none | – | **on** | round: vector helpers in norms and sums |
| 6 | OpenMP | off | `build_cpu.sh:20` | requested, not found (`GGML_OPENMP_ENABLED=OFF`) | no |
| 7 | compiler | NDK clang 19 (r28) | kit build notes | Apple clang | round at most |
| 8 | source tree | `620d2225c` + `patches/llama.cpp` + `llama.cpp-c1-olmoe` + c3 0001, 0002, 0003, 0005, 0007 | `tom-r35-build.md` | `620d2225c` + the campaign-2 series (`26f78f925`) | no with every knob unset, by each patch's identity check; **those checks ran on the repack path only** (r10) |

### 1.2 Load and context

| # | item | phone (app as it is) | where set | freeze | numerics |
|---|---|---|---|---|---|
| 9 | **faster weight layouts (KleidiAI / repack copies)** | **refused for a file over about 2.7 GB**: `totalRam − copyBytes ≥ 5 GB`, with the file's length as the copy size. A kit can force it per run | `lib/device.dart:591`; `lib/assistant.dart:455–456, 466, 481`; `lib/model_store.dart:19` | repack on (the tool's default) | **yes**, and speed: the stock kernel is about 2.2 × a gemv (r6) |
| 10 | head placement | KleidiAI when row 9 is on; else plain Q6_K | `kleidiai.cpp:1793–1800`; knob set for OLMoE only, `lib/research.dart:108` | repack type | **yes** (r11, r12) |
| 11 | KV cache type | **Q8_0** for K and V on the CPU | `lib/assistant.dart:471–472` | F16 (no flag) | **yes** (r15) |
| 12 | flash attention | on (required by row 11) | `llamadart/…/load_param_helpers.dart:104–117`; `llama-context.cpp:4064–4067` | "auto"; what it resolved to is not in the logs | round if both are on; **yes** if the freeze ran without it |
| 13 | context size | 4,096 (8,192 with ample free RAM); campaign 2's kit fixed 4,096 from r27; the R73 plan says 512 | `lib/device.dart:599–602`; `lib/assistant.dart:164, 421, 460` | 512 | no for a short prompt; it sets the KV buffer: 17 / 136 / 128 MiB at 512, 0.13 / 1.06 / 1.00 GiB at 4,096 (r15) |
| 14 | batch | logical 2,048, micro 512: a prompt of 150 to 190 tokens is one batch; decode is batch 1 | `llamadart/…/model_params.dart:466, 471` | `-b 512` and `-ub 1` | covered: the gate's two settings bracket it (prompt batch 150 to 190 is not 512, same kernels) |
| 15 | threads | chosen by the app's own timing sweep; campaign 2 claimed with 3; the R73 plan says 4 on cpus 4-7 | `lib/assistant.dart:522–523, 623, 723` | 8 | no (an output depends on its own row) |
| 16 | load mode | mapped file | `load_param_helpers.dart:84` | mapped | no |
| 17 | chat template | the model's own unless the app has one | `lib/assistant.dart:476, 507–509` | none (raw text) | not the gate's concern; it sets the benchmark's token count |
| 18 | served file | the frozen Q4_0, by sha256 | freeze | the same file | – |

### 1.3 Environment (campaign 2's kit lists, as they stand in `lib/research.dart`)

| # | variable | value in the kit | where set | freeze | numerics |
|---|---|---|---|---|---|
| 19 | `VOLLIE_HEAD2_REPACK` | `1` for OLMoE only | `research.dart:108` | n/a | **yes**; the head knob (r12) |
| 20 | `GGML_TINY_OP` | `262144`: small ops run on one thread | `research.dart:113`; `lib/device.dart:1071` | unset | no (campaign 1: identical per chunk) |
| 21 | `GGML_EXPERT_IO` | `1` | `research.dart:113`; `device.dart:1079` | unset | no; expert read-ahead, nothing to do in a dense model |
| 22 | `VOLLIE_SAMPLER_BYPASS` | `1` | `research.dart:113` | n/a | no: exact fast path; greedy either way (`assistant.dart:560`, `temp: 0`) |
| 23 | `GGML_C2_IO`, `VOLLIE_SCHED`, `VOLLIE_LINES`, `VOLLIE_SKIP_STATS` | counters and line routing | `research.dart:116, 135, 357` | unset | no |
| 24 | `VOLLIE_KV_DONTNEED` | `1` (a factor from r25) | `research.dart:125` | unset | no (identity shown, campaign 2) |
| 25 | `VOLLIE_POOL`, `VOLLIE_CPUMASK` | `1`, `0xf0` (factor "pin") | `research.dart:129` | unset | no |
| 26 | `VOLLIE_EXEC_MAP`, `VOLLIE_TOUCH`, `VOLLIE_TOUCH_S`, `VOLLIE_NICE` | residency and scheduling factors | `research.dart:133, 139, 142` | unset | no |
| 27 | campaign 2's numerics knobs (`VOLLIE_SKIP_MASS`, `VOLLIE_HEAD2`, `GGML_EXPERT_SKIP`, `GGML_HEAD2`, `GGML_HEAD2_Q`) | set in the "num" arms only | `research.dart:120–121` | unset | **yes**; MoE levers, off for campaign 3's stock |

The kit clears every name of its own list before each run (`research.dart:362–376`), so nothing leaks from one
run to the next. Campaign 3's knobs (`VOLLIE_PROBE`, `VOLLIE_X3_EMU…`) are off unless a block sets them.

### 1.4 What follows

- **Rows that can change a logit and differ from the freeze:** 1, 9, 10, 11, 12 (if the freeze ran without
  flash attention), and by rounding 4, 5, 7. Rows 13 and 14 do not change logits for the gate's text but decide
  memory and which kernels run.
- **Row 9 is the one to decide first.** Three coherent choices exist:

  | choice | body path | speed (*estimated* K0) | memory | numerics against the freeze |
  |---|---|---|---|---|
  | the app's rule as shipped | plain Q4_0 rows from the mapped file, stock kernel | 4.5 / 4.0 / 4.1 tok/s | file-backed, nothing copied | not bit-equal to repack (memo-05: ±0.02 % on two other models, *host*) |
  | kit forces the copies (what R72 measured) | KleidiAI | 6.90 / 6.21 / 6.22 | 3.4 to 3.8 GB anonymous: shown to fit for Yi only | r9 to r11 |
  | KleidiAI layout from a mapped side file (r7, r8) | KleidiAI | the same | file-backed | the same as the row above |

  The claim rule says "fix it in the runtime, never in the user's phone": a configuration that the shipped app
  would refuse is a research setting until the rule in `device.dart` is changed for it, and that rule exists
  because the phone was killed.
- **One reference, defined once.** I propose the pre-registration of the first model-loading round carries a
  block "stock configuration" with rows 1, 9, 10, 11, 12, 13, 15 filled in, and that the host reference is run
  with the matching flags: KleidiAI build, `GGML_KLEIDIAI_SME=0`, `-fa on -ctk q8_0 -ctv q8_0`, the head knob as
  decided. Tom's runs under way give its pieces; the full-set run comes after the choices.
- **Not found in this pass:** any setting of the KleidiAI thread hint or chunk multiplier
  (`GGML_TOTAL_THREADS`, `GGML_KLEIDIAI_CHUNK_MULTIPLIER`); they are unset in the kit as far as I can see and do
  not change bits.

## 2. Can a width-2 pass share the KV walk? (next question 3)

### 2.1 The loop as it is (*source*, `ops.cpp` lines 9000 to 9110)

Rows of work are (query position, head) pairs; a thread takes a range of them. For each pair:
1. the query vector (128 values) is converted once to the key type's format;
2. for every cached position: one dot product with the key row, a scale, the mask, an `expf` for the running
   softmax;
3. the value row is added with its weight. **For a quantised cache the row is first converted to float**
   (`v_to_float(v_data, V32, DV)`), then `ggml_vec_mad_f32`.

With two positions in the pass, the two queries of a head are neighbours in that order, so the same thread
walks the same keys and values twice in a row.

### 2.2 What sharing can and cannot save

- **Bytes: nothing.** One head's keys and values at 200 positions are 54 KB in Q8_0; the second walk finds them
  in the core's own cache. The first walk brings them from memory either way.
- **Compute: the value conversion.** Step 3's conversion does not depend on the query. Converting each value row
  once and adding it to both accumulators halves that part. Steps 1, 2 and the add itself are per query and
  stay.
- **How large is the conversion?** Per cached row of 128 values (*estimated* from the operations, not timed):
  key dot product about 10 ns, `expf` and bookkeeping about 25 ns, conversion 20 to 110 ns (20 if the compiler
  vectorises the Q8_0 dequantise loop, 110 if it runs scalar), add about 30 ns. So the conversion is 25 to 65 %
  of a row.
- **Saving for the second position:** 25 to 65 % of its attention: **0.6–1.6 / 1.1–2.9 / 1.0–2.7 ms per pass**
  on §2 of r15's centres (2.5 / 4.5 / 4.2 ms). Per committed token, divide by 1.33 to 1.45.
- **Exactness:** each query's dot products, `expf` calls and adds are the same operations in the same order;
  only where the float copy of the value row comes from changes, and it is the same floats. Bit-identical by
  construction; to be shown per chunk at `-ub 2` against stock, as any runtime change.
- **Size of the change:** the row loop takes a group of consecutive queries of one head instead of one;
  two accumulators instead of one. About 40 lines in one function, off unless a knob is set.

### 2.3 Two cheaper cousins

| option | what | gain | cost |
|---|---|---|---|
| a vectorised Q8_0-to-float for the value rows | check the compiled loop on the phone's build; hand-write it if scalar | up to the same 25 to 65 %, **at width 1 as well** | exact (same floats); a disassembly check first, as `tools/c3_a78_pipes.py` does |
| F16 values, Q8_0 keys (`type_v` = F16) | no conversion at all; the add reads halves directly | the conversion, at every width | 26 / 26 / 25 MB more bytes per token on the full-attention subjects (about 1 ms at the phone's read rate; less on Yi), twice the value half of the KV buffer; an app setting (row 11); a numerics change towards the freeze |

- The first is the one to look at first: it helps every token, not only the second position, and may already be
  true (then the whole of §2.2 shrinks to its low end).
- The second trades compute for bytes; on a phone whose cores are the scarce thing during a width-2 pass it may
  pay. It belongs with the KV screen Tom has queued (r15 §1.4) as a third arm: K Q8_0 with V F16.

## 3. Next questions, ranked

1. **Row 9, priced properly:** what the phone can hold and run today for each subject under each of the three
   choices of §1.4, with R69's and R72's residency records, and what change to `canCopyWeights` each would
   need. This decides what "stock" means for campaign 3's speed numbers. Mine.
2. **Is the Q8_0 value conversion vectorised in the phone's module?** Disassemble `dequantize_row_q8_0` (or
   whatever `v_to_float` resolves to) from the shipped `libggml-cpu-android_armv8.2_2.so`. Minutes; mine if the
   lead allows reading the binary, else Tom's.
3. **Re-run the ladder** with r15's O and second-position costs and with whatever §2 yields, after R75 and Tom's
   stock runs.
4. **What "auto" flash attention resolved to in the freeze** (row 12): one load log line from a `llama-perplexity`
   run on the lab build. Tom's KV screen answers it.
5. **DeepSeek's head; cold read rate of the served file; fixed share: which C.**

## 4. Limits

- The table is read from source at today's line numbers; I did not run the app or a load. A row could be
  overridden somewhere I did not read (the app's settings screen, a kit patch not yet applied).
- Row 9's "about 2.7 GB" uses the phone's total RAM as R72's header printed it (7,565,184 kB); the app reads its
  own value at run time.
- "Numerics: no" for campaign 1's and 2's knobs repeats their identity checks, which ran on the repack path and
  on other models.
- §2's times are operation counts, not measurements; nothing has profiled flash attention on a dense model on
  this phone.

## 5. Sources (2026-11-23)

Repository: `lab/app-c3/tool/build_cpu.sh` (lines 13 to 36); `lab/app-c3/lib/device.dart` (lines 583 to 602, 1029 to
1085), `lib/assistant.dart` (lines 158 to 165, 421, 454 to 481, 518 to 560, 623, 723, 745 to 789), `lib/research.dart`
(lines 100 to 146, 207 to 208, 357 to 376), `lib/model_store.dart` (lines 13 to 19), `pubspec.yaml`;
`lab/app-c3/third_party/llamadart/lib/src/backends/llama_cpp/load_param_helpers.dart` (lines 84, 104 to 117, 136 to
165) and `…/core/models/inference/model_params.dart` (lines 331 to 378, 424 to 434, 462 to 471);
`lab/llama.cpp-c3r32/ggml/src/CMakeLists.txt` (lines 527 to 532), `src/llama-context.cpp` (lines 3992 to 4004, 4064
to 4070), `ggml/src/ggml-cpu/kleidiai/kleidiai.cpp` (lines 315 to 317, 1793 to 1800), `ggml/src/ggml-cpu/ops.cpp`
(lines 9000 to 9110); `lab/llama.cpp-c3/build/CMakeCache.txt`; `tools/ppl.sh`, `tools/c3_freeze.sh`;
`reports/02_methods.md` line 18; `docs/runtime-knobs.md` lines 10 and 62; R72 JSON (`hdr mem`);
`memo-10-work/clyde-r73-plan.md` line 48; `tom-r35-build.md`.
