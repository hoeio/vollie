# c3 memo-12 work, round 19 (Bonnie): the anonymous budget under choice C; a host dense-decode specification for Tom

- **Date:** 2026-11-24. **Author:** bonnie, `claude-fable-5-2`. Follows [bonnie-joins-r18.md](bonnie-joins-r18.md). Not
  committed. No verdict here. **Ebony has not reviewed this file.**
- **Labels.** *device*, *host*, *estimated*, *source* as before. Triples are Yi / Llama-2 / DeepSeek.
- **New *host* facts, read from load logs that already exist** (`lab/c3/logs/h3-<m>-b512-40.log.err`, Tom's
  screens on the lab build, no KleidiAI): the weight buffer, the mapped buffer, the KV buffer, the compute
  buffer, and what flash attention's "auto" resolved to. I ran nothing. No web source.

## 0. Answer first

1. **The load logs confirm r8's size arithmetic to the megabyte** (*host*): the repack buffer is 3,175.08 /
   3,576.54 / 3,585.00 MiB, which is my Q4_0 body plus the served Q6_K head (2,969.9 + 205.1; 3,474.0 + 102.5;
   3,256.9 + 328.1). The mapped rest is 140.64 / 70.33 / 225.02 MiB. So r8 §1.3's KleidiAI sizes (body + int8
   head) stand on a checked body.
2. **Flash attention's "auto" was "enabled" in the freeze's build** (*host*: `flash_attn = auto` followed by
   `resolve_fused_ops: Flash Attention enabled` in all three logs). Row 12 of r16's table is settled: the phone
   and the freeze both run flash attention. The KV **type** is still the difference (F16 against Q8_0).
3. **Anonymous memory under choice C** (*estimated* from the *host* buffer sizes and campaign 2's *device*
   remainder):

   | MiB | Yi | Llama-2 | DeepSeek |
   |---|---|---|---|
   | read every token ("hot"): KV in use, app and runtime | about 160 | about 200 | about 200 |
   | touched once at the prompt, then idle: the rest of the compute buffer and of the KV | up to 165 | up to 180 | up to 310 |
   | the head, if it goes to the repack type and stays anonymous (hot) | + 205 | + 103 | + 328 |

4. **Against r18's steps:** the hot part (0.15 GB) was right. The idle part was missing; it can be pushed to
   zram, where it still takes room. **I add an idle anonymous part to each step: 0.15 / 0.20 / 0.30 GB**, filled
   once and not read again (§1.3).
5. **A repack-type head should not stay anonymous under C.** On DeepSeek it adds 0.33 GB that is read every
   token and cannot be dropped: more than the whole hot budget. It is a re-ordering of served bytes like the
   body, so it can be a second mapped file with the same key (§1.4).
6. **The host decode specification for Tom is §2:** nine `llama-bench` runs per subject on a KleidiAI build of
   the kit's tree with campaign 1's per-op profiler, at two context depths and three cache types. Laptop
   numbers are relative only; what carries over is the split of a token and the ratios between arms.

## 1. The anonymous budget under choice C (next question 2)

### 1.1 What the laptop's load logs give (*host*; lab build, `-c 512 -b 512`, existing logs)

| line in the log | Yi | Llama-2 | DeepSeek | my arithmetic before seeing it |
|---|---|---|---|---|
| `CPU_REPACK model buffer size` | 3,175.08 MiB | 3,576.54 | 3,585.00 | 3,175.0 / 3,576.5 / 3,585.0 (r8 body + served head) |
| `CPU_Mapped model buffer size` (largest) | 140.64 MiB | 70.33 | 225.02 | 141 / 70 / 220 (r8 §1.4) |
| `CPU KV buffer size`, F16, n_ctx 512 | 32.00 MiB | 256.00 | 240.00 | 32 / 256 / 240 (r15) |
| `CPU compute buffer size`, micro-batch 512 | 165.52 MiB | 105.02 | 240.52 | not estimated before |
| the same at micro-batch 1 (`-ub 1` logs) | – | 0.21 | – | – |
| `CPU output buffer size` | – | 0.12 | – | – |
| `flash_attn` | auto → enabled | auto → enabled | auto → enabled | unknown until now |

- The compute buffer follows the vocabulary: 512 positions of logits are 125 / 63 / 200 MiB of it.
- These are the lab build's numbers. On a KleidiAI build the weight buffer becomes 3,220.7 / 3,599.4 / 3,658.0 MiB
  (int8 head, r8); the other lines should not change. Tom's KleidiAI runs print them.

### 1.2 The budget on the phone under C (*estimated*)

The app asks for micro-batch 512 and Q8_0 keys and values; the R73 plan sets n_ctx 512.

| item, MiB | Yi | Llama-2 | DeepSeek | kind | basis |
|---|---|---|---|---|---|
| KV buffer, Q8_0, n_ctx 512: allocated | 17 | 136 | 128 | anonymous | r15 |
| … of it in use at 220 positions | 7 | 58 | 55 | hot | × 220 ÷ 512; the rest is never touched if `VOLLIE_KV_DONTNEED=1` |
| compute buffer: reserved | 166 | 105 | 241 | anonymous | §1.1 |
| … touched by a prompt of 150 to 190 tokens | 50–165 | 35–105 | 75–241 | idle after the prompt | a third of it by position count, all of it if the allocator spreads; not known |
| … used by a one-token decode graph | under 1 | under 1 | under 1 | hot | §1.1, micro-batch 1 |
| app, Dart heap, runtime scratch | 131–147 | 131–147 | 131–147 | hot, mostly | campaign 2's remainder on this phone (*device*, memo-24) |
| trained guess head and its shortlist (lane stacks only) | about 80 | about 60 | about 110 | hot | two 4,096-wide blocks in F16 plus 8,192 head rows; *estimated* |
| **hot, stock decode** | **about 160** | **about 200** | **about 200** | | |
| **idle after the prompt** | **up to 165** | **up to 180** | **up to 310** | | compute buffer plus the untouched KV if the knob is off |
| head in the repack type, anonymous | + 205 | + 103 | + 328 | hot | r12 §1.4 |
| two-stage head's screening copy (if used) | + 141 | + 70 | + 225 | hot | r12 §2 |

- "Hot" pages are read every token; losing one costs a fault inside the decode. "Idle" pages can go to zram
  without slowing the decode, but zram is RAM: at the phone's own ratio (R72 header: 2.75 GB stored in 0.53 GB)
  it takes about a fifth of their size.
- File-backed beside it: the side file (3.38 / 3.77 / 3.84 GB CPU-only with the int8 head inside; 2.2 / 2.4 /
  2.6 GB beside a lane) and the embedding rows the prompt touches (a few MiB of 141 / 70 / 225).

### 1.3 Against the steps of r18

| step | r18: hot anonymous | needed, hot | needed, idle | change to the step |
|---|---|---|---|---|
| `x5s0` (Yi, CPU only) | 0.05 GB | 0.17 GB | 0.15 GB | hot 0.15; add idle 0.15 |
| `x5s1` (Yi, lane) | 0.05 GB | 0.17 + 0.08 (guess head) | 0.15 | hot 0.25; add idle 0.15 |
| `x5s2` (Llama-2, lane) | 0.15 GB | 0.21 + 0.06 | 0.20 | hot 0.25; add idle 0.20 |
| `x5s3` (DeepSeek, lane) | 0.15 GB | 0.21 + 0.11 | 0.30 | hot 0.30; add idle 0.30 |
| `x5s4` (Llama-2 / DeepSeek, CPU only) | 0.15 GB | 0.21 | 0.20 to 0.30 | hot 0.20; add idle 0.30 |
| `x5sA` (DeepSeek under A) | 0.15 GB | 0.21 | 0.30 | hot 0.20; add idle 0.30 |
| `x5s5` (margin) | 0.30 GB | – | – | hot 0.30; add idle 0.30 |

- **New field and stage** for the specification: `x5 res stage=anon_idle step= idle_gb= fill_s=` (filled once
  and never read again, with the same pattern the hot part uses; how well the real buffers compress is not
  known, so the step reports where the pages went instead of assuming a ratio). The pass lines need nothing new: `vmswap`, `swapfree`
  and `rssanon` show where it went.
- **Effect on r18's probabilities:** each step's total rises by 0.2 to 0.45 GB, part of it compressible.
  I lower "held" by 0.05 on the lane steps and by 0.05 to 0.10 on the 3.85 and 4.0 GB steps:

  | step | held, r18 → now | survives with the hold, r18 → now |
  |---|---|---|
  | `x5s0` | 0.90 → 0.88 | – |
  | `x5s1` | 0.85 → 0.82 | – |
  | `x5s2` | 0.60 → 0.55 | 0.50 → 0.45 |
  | `x5s3` | 0.55 → 0.48 | 0.45 → 0.40 |
  | `x5s4` (map x) | 0.70 → 0.62 | 0.60 → 0.52 |
  | `x5sA` | 0.60 → 0.50 | 0.50 → 0.42 |

- A cheap runtime help that the budget suggests: **release the compute buffer's prompt-sized part after the
  prefill** (it is not needed again until the next prompt), as campaign 2 did for the KV with
  `MADV_DONTNEED`. It turns "idle" into "gone": up to 165 / 105 / 241 MiB. Bit-identical by construction if the
  buffer is not read before it is rewritten; that has to be shown, not assumed.

### 1.4 The head under C

| head placement | file-backed | anonymous, hot | budget share (r12) | note |
|---|---|---|---|---|
| in KleidiAI, inside the side file | + 263 / 131 / 421 MB of file | 0 | up to a fifth of Yi's | simplest |
| repack type, anonymous | 0 | 205 / 103 / 328 MiB | 0 | doubles DeepSeek's hot anonymous memory |
| repack type, from a second mapped file | + 215 / 108 / 344 MB of file | 0 | 0 | the same design as the body's side file (a re-ordering by upstream's own routine); about 60 more lines |
| plain Q6_K from the served file (no copy at all) | the served file's own pages | 0 | 0 against the freeze's kernel family | 1.5 to 2 × the head's time (r17, *estimated*); r18's `x5p` measures it |

If the head knob is adopted with choice C, the third row is the one that keeps both the budget and the memory.

## 2. A dense decode on the laptop: specification for Tom (next question 3)

**Why.** No dense decode has been profiled anywhere in this campaign. O, the attention's share, the cost of a
second position and the three cache types are all estimates (r15, r16, r17). The laptop cannot give phone
milliseconds; it can give the split of a token by op type and the ratios between arms, which is what those
estimates are made of.

**Build.** The kit's tree, because campaign 1's profiler (`GGML_CPU_PROF`) is in it and not in the freeze's tree:
`lab/llama.cpp-c3r32`, host build with KleidiAI on, options as r9 §3 (this is task H1's build; add the target
`llama-bench`).

**Runs** (per subject; `-t 4` as on the phone; three repetitions):

```
B=lab/llama.cpp-c3r32/build-kai/bin; M=lab/c3/models
for m in yi15-6b llama2-7b deepseek7b; do
 for kv in "f16 f16" "q8_0 q8_0" "q8_0 f16"; do set -- $kv
  for p in 64 192 448; do
   GGML_KLEIDIAI_SME=0 GGML_CPU_PROF=1 GGML_TINY_OP=262144 \
   $B/llama-bench -m $M/$m-q4_0.gguf -t 4 -ngl 0 -fa on -ctk $1 -ctv $2 -p $p -n 64 -r 3 -o csv \
     > lab/c3/logs/dd-$m-k$1-v$2-p$p.csv 2> lab/c3/logs/dd-$m-k$1-v$2-p$p.err
  done
 done
done
```

- `-p 192 -n 64` decodes at positions 192 to 256, the claim run's range. `-p 64` and `-p 448` give the slope
  with position.
- `GGML_CPU_PROF=1` prints one `cpu-prof` line per 32 one-token graphs on stderr: per op type the count, the
  microseconds computing and the microseconds waiting (`docs/runtime-knobs.md`). `GGML_TINY_OP=262144` is the
  kit's setting.
- Nine runs per subject, 27 in all; each is a load plus a few seconds. About 20 minutes with the loads.
- One more arm for the head knob, at `-p 192`, Q8_0 cache: the same with `VOLLIE_HEAD2_REPACK=1` (3 runs).

**What to report** (all *host*, relative only):

| quantity | from | my expectation |
|---|---|---|
| the log's context lines per arm (`flash_attn`, `type_k`, `type_v`, KV and compute buffer sizes) | the `.err` files | Q8_0 KV at 34,816 / 278,528 / 261,120 bytes per position × the bench's n_ctx; flash attention enabled |
| share of a token's time in the weight matmuls, the head, flash attention, everything else | `cpu-prof` at `-p 192`, Q8_0 | the laptop is compute-richer and bus-richer than the phone, so I give no band for the shares; they are the first data |
| flash attention at `-p 448` ÷ at `-p 64` | `cpu-prof` | 5.0; 4.0–6.0 (positions 480 against 96) |
| flash attention, Q8_0 values ÷ F16 values (keys Q8_0 in both) | arms 2 and 3 at `-p 192` | 1.4; 1.15–1.8 (the value conversion; r17's count is for an A78) |
| flash attention, F16 keys and values ÷ Q8_0 keys with F16 values | arms 1 and 3 | 0.9 to 1.2 |
| the non-matmul, non-head, non-attention rest, per token | `cpu-prof` | flat in position; the same for the three cache types within 5 % |
| head op time, repack type ÷ KleidiAI | the extra arm | under 1 on the laptop says little; the M4 picks an i8mm kernel for the int8 head (r9) |
| tokens per second per arm | the `.csv` | for the record only |

**Readings I would attach:**
- If the attention's slope is linear and its Q8_0 ÷ F16-value ratio is above 1.15, r17's per-row split is
  right in kind, and the two attention changes (sharing the conversion at width 2; F16 values) are worth a
  per-chunk identity check and a phone cell.
- If "everything else" is more than a third of the non-matmul time on the laptop, r15's O is light on the small
  ops and its bands for R73 should widen upward.
- If the profiler shows an op type r15's table does not list with more than 3 % of a token, that is a new row.

**What it cannot say:** phone milliseconds, the phone's bus-bound matmul share, thread waits on four unequal
cores. Those stay with R73.

**A width-2 arm, if Tom's batch-2 tool is at hand** (`tools/c3_head_b2check.cpp` feeds two positions per pass):
the same profile with `GGML_CPU_PROF=2`, to read the second position's attention directly. Optional; it needs
the tool built against this build.

## 3. Next questions, ranked

1. **Fold R75 into r18's draft** when its report arrives (today, by the lead's note): choice B's record at
   4.1 GB, the GPU cap fields, the gates after lane-like load; then the idle-anonymous change of §1.3.
2. **Is the compute buffer safe to release after the prefill?** Read the scheduler's allocator: is any part of
   the buffer read across graphs. If not, a `MADV_DONTNEED` after the prompt is a bit-identical runtime change
   worth up to 0.24 GB on DeepSeek. Source reading, mine.
3. **A second side file for a repack-type head** (§1.4): what the repack type's allocation needs, by analogy
   with r7. Source reading, mine.
4. **Re-read r15 to r17 against Tom's host decode** when it exists.
5. **DeepSeek's head; cold read rate; fixed share.**

## 4. Limits

- §1.1 reads logs of the lab build at `-c 512 -b 512`. The phone's context is built by the app's library with
  its own batch settings; the compute buffer there may differ, and no phone load log of a dense model exists.
- "Touched by the prompt" in §1.2 is a range because I did not read how the graph allocator places a
  190-token graph inside a 512-token reservation.
- The app remainder is campaign 2's, measured with other models and an app one version older.
- §1.3's probability changes are judgement on judgement.
- §2's expectations for the laptop are weak by design: the laptop has four vector pipes and three times the
  memory bandwidth.

## 5. Sources (2026-11-24)

Repository: `lab/c3/logs/h3-yi15-6b-b512-40.log.err`, `h3-llama2-7b-b512-40.log.err`, `h3-deepseek7b-b512-40.log.err`
and the `-ub 1` logs beside them (load lines); `lab/llama.cpp-c3r32/ggml/src/ggml-cpu/ggml-cpu.c` (`GGML_CPU_PROF`
present; absent in `lab/llama.cpp-c3`); `lab/llama.cpp-c3/tools/llama-bench/llama-bench.cpp` (options),
`common/arg.cpp` (lines 1758, 2441); `tools/host_bench.sh`; `docs/runtime-knobs.md` lines 10 and 13; campaign 2
`reports/memo-24-anon-memory-and-rest.md` (lines 19, 105); R72 JSON (`hdr zram`); my r8 §1.3, r12 §1.4 and §2, r15,
r16 §1, r17, r18.
