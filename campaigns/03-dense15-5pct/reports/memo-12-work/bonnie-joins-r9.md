# c3 memo-12 work, round 9 (Bonnie): which path defines "stock served perplexity"; the pack rate

- **Date:** 2026-11-23. **Author:** bonnie, `claude-fable-5-2`. Follows [bonnie-joins-r8.md](bonnie-joins-r8.md). Not
  committed. No verdict here. **Ebony has not reviewed this file.**
- **Labels.** *device*, *derived*, *host*, *estimated*, *source* as before. Triples are Yi / Llama-2 / DeepSeek.
- **New measurement, one, *host*:** a one-thread timing of KleidiAI's Q4_0 weight pack routine on the laptop
  ([bonnie_pack_time.c](bonnie_pack_time.c)), relative only, taken at a load average of 23. No model ran, no
  perplexity ran, nothing ran on the phone. No web source: everything is read in the repository and the local trees.

## 0. Answer first

1. **Campaign 2 met this exact question on 2026-11-15 and the owner decided it (option (a), R61).** The rule that
   came out: the budgets stay the absolute values from the host build without KleidiAI; stock is measured a
   second time on a host build with KleidiAI; both references are printed side by side; a numerics setting must be
   inside both. Campaign 3's terms say "as campaign 2" for the budget. **So the terms already tell us what to do;
   what is missing is the campaign-3 measurement.**
2. **My r8 guess (under 0.05 %) is too low against that record.** R61 (*host*, full set): KleidiAI moved
   Granite's stock by +0.57 % at `-b 512` and +0.50 % at `-ub 1` with only 541 MiB of its weights in KleidiAI;
   OLMoE by −0.01 and +0.02 %. Our subjects put 3.2 to 3.7 GiB through it. **My band for the three: −0.3 to
   +1.0 %, centre +0.3 %** (*estimated*). At +0.5 % the phone's unmodified runtime already uses a tenth of the
   +5 % budget.
3. **The laptop can run the phone's Q4_0 kernels.** With `GGML_KLEIDIAI_SME=0` the M4 selects the dotprod entry
   for Q4_0, the same two kernels as the phone, and the i8mm entry for the 8-bit head (*host*, R61's and Tom's
   load logs). So the host run is the phone's path for the body and differs only in the head's kernel.
4. **The run:** 40 chunks per subject at `-b 512` and `-ub 1`, about 25 to 30 minutes for all three plus one
   build (*estimated* from the freeze logs' seconds per chunk). Commands in §3. The freeze logs already hold the
   repack path's value at chunk 40, so no second arm is needed.
5. **Pack rate:** the routine packs **4.1 to 4.4 GB/s on one laptop thread** (*host*, contended). On an A78 I put
   it at 0.7 to 1.5 GB/s (*estimated*). The pack is therefore not what a first run waits for; the write-back is
   (0.45 to 0.61 GB/s, *device*, R69). First run with a side file: **8 to 18 s** (*estimated*).

## 1. What the written terms say

| text | where | says |
|---|---|---|
| "perplexity at most +5 % over the served file's stock perplexity" | campaign 3 `CAMPAIGN.md` line 37 | the reference is the served file, stock |
| "PPL ≤ stock served PPL × 1.05, one budget for everything that changes numerics, at both `-b 512` and `-ub 1`", source column "as campaign 2" | same, line 48 | campaign 2's reading applies |
| "'stock served PPL' in the budget means the served GGUF run by the unmodified runtime (no skip, no two-stage head)" | campaign 2 `CAMPAIGN.md` line 104 | the reference is a runtime, not a build |
| "`llama-perplexity`, wikitext-2-raw test set, context 512, batch 512, all chunks (`tools/ppl.sh`)" | `COMMON_RULES.md` rule 2 | the test is fixed; **the build is not named** |
| "Budgets are written before results and do not move after them… A budget is stated as an absolute PPL per model" | `COMMON_RULES.md` rule 3 | the freeze values stand whatever a later run shows |
| the S3 freeze | `tools/c3_freeze.sh` line 10; `lab/llama.cpp-c3/build/CMakeCache.txt` | built **without KleidiAI**: `GGML_CPU_KLEIDIAI=OFF`, `GGML_CPU_REPACK=ON`, `GGML_CPU_ARM_ARCH=armv8.2-a+dotprod+fp16`, Metal off, BLAS off |
| the phone's module | campaign 2 `STATUS.md` line 141; R74 `cpu_buft=CPU_KLEIDIAI` (*device*) | built **with KleidiAI** |

Campaign 2's record of the same gap (`STATUS.md` lines 141 to 150, `prereg/R61-kleidiai-host-ppl.md`,
`reports/README.md` R61):
- Found 2026-11-15: "the phone's stock numerics have never been measured on the host".
- Owner, option (a): "measure with KleidiAI on the host and state both references side by side; the budget rule is
  unchanged."
- How results were read, fixed before the runs: the existing budgets are not changed; a second column is added
  (stock with KleidiAI, and that × 1.05); a setting is "inside on the phone's build" if it is at or below the
  same-build stock × 1.05 at both batch settings **and** at or below the existing absolute budget.
- Result (*host*, full set): OLMoE 11.7132 / 11.7159 against 11.7146; Granite 9.2795 / 9.2732 against 9.2266.

## 2. What is ambiguous, and what is new for campaign 3

1. **No rule names the build.** Rule 2 fixes the test and the script; the script takes any binary directory. The
   only text that settles the build is campaign 2's owner decision, which campaign 3 inherits through "as
   campaign 2". The lead may want that written into campaign 3's own terms as a dated note.
2. **The freeze did not carry the second reference.** `results.csv` has one stock value per subject and setting,
   all from the build without KleidiAI. Until the KleidiAI run exists, "inside on the phone's build" cannot be
   read for any lever.
3. **Which way the two readings bind.** If KleidiAI raises stock (as on Granite), the absolute budget is the
   tighter one: a lever has less than 5 % of room on the phone. If it lowers stock, the same-build reading is
   tighter. Either way the room is at most 5 % and usually less.
4. **Dense subjects are more exposed than campaign 2's.** In R61 KleidiAI held 541 MiB of Granite and 243 MiB of
   OLMoE. Here it holds the whole body and the head: 3,221 / 3,599 / 3,658 MiB (r8 §1.3, *estimated*).
5. **The head is re-quantised on the phone's path** to int8 with one scale per row (r8 §1.2, *source*). In R61
   that alone moved OLMoE by +0.02 to +0.05 %. Campaign 3 has no knob decision for it: `VOLLIE_HEAD2_REPACK`
   exists in the kit's tree (`lab/llama.cpp-c3r32`) and not in the freeze's tree (`lab/llama.cpp-c3`).
6. **R61's stated limit is narrower than written.** It says the M4 picks KleidiAI's i8mm kernels. The load logs
   say: `primary q4 kernel feature DOTPROD` and `primary q8 kernel feature I8MM` (*host*, `lab/logs/r61/`, and
   Tom's batch-2 logs under `lab/c3/head/`). The source agrees: on Apple the dotprod Q4_0 entry is listed first
   (`kernels.cpp` lines 490 to 536). So only the 8-bit head runs a different kernel on the laptop. Integer dot
   products are exact in either; the float steps were not compared.
7. **Every "bit-identical" check of campaign 3 so far ran the repack path** (Ebony r5 §3.1). That is the same gap
   seen from the identity side; her condition 7 covers it for the W-position kernel.

## 3. The 40-chunk run, as commands (not run by me)

**Build** (one new directory beside the freeze's; the same tree and commit as the freeze, `lab/llama.cpp-c3` at
`26f78f925`, so that KleidiAI is the only variable):

```
cd <repo root>
cmake -S lab/llama.cpp-c3 -B lab/llama.cpp-c3/build-kai -DCMAKE_BUILD_TYPE=Release \
  -DGGML_CPU_KLEIDIAI=ON -DGGML_NATIVE=OFF -DGGML_CPU_ARM_ARCH=armv8.2-a+dotprod+fp16 \
  -DGGML_METAL=OFF -DGGML_BLAS=OFF \
  -DFETCHCONTENT_SOURCE_DIR_KLEIDIAI=$PWD/lab/llama.cpp-c1h6/build-kai/_deps/kleidiai-src
cmake --build lab/llama.cpp-c3/build-kai -j 8 --target llama-perplexity
```

- These are the freeze build's options with KleidiAI switched on; R61's build (`lab/llama.cpp-c1h6/build-kai`) has
  the same eight options. Any other option should be copied from `lab/llama.cpp-c3/build/CMakeCache.txt`.
- The last option reuses the KleidiAI v1.31.0 sources already on the laptop, so the build needs no download. I
  have not tried it; without it CMake fetches the pinned archive and checks its MD5.
- Do not use `lab/c3/head/build-kai` for this: it has `GGML_BLAS=ON`, which the freeze build does not.

**Runs** (six, through the project's fixed script; `-c 512 -b 512 -t 8 -ngl 0` come from the script):

```
for m in yi15-6b llama2-7b deepseek7b; do
  GGML_KLEIDIAI_SME=0 sh tools/ppl.sh --llama-dir lab/llama.cpp-c3/build-kai/bin \
    --text lab/c3/text/wiki.test.raw c3k40-$m-q4_0-b512 lab/c3/models/$m-q4_0.gguf --chunks 40
  GGML_KLEIDIAI_SME=0 sh tools/ppl.sh --llama-dir lab/llama.cpp-c3/build-kai/bin \
    --text lab/c3/text/wiki.test.raw c3k40-$m-q4_0-ub1 lab/c3/models/$m-q4_0.gguf --chunks 40 -ub 1
done
```

- Logs land in `lab/logs/p/ppl-c3k40-<m>-q4_0-<b512|ub1>-wiki.log`. The script also runs its English sanity text
  when that file is present.
- No other environment: campaign 3's stock is the runtime with every knob unset.

**Lines to read in each log:**

| line | expected | why |
|---|---|---|
| `kleidiai: primary q4 kernel feature` | `DOTPROD` | the phone's Q4_0 kernels; anything else voids the comparison |
| `kleidiai: primary q8 kernel feature` | `I8MM` | the stated limit (head only) |
| `load_tensors: CPU_KLEIDIAI model buffer size` | one line; 3,220.7 / 3,599.4 / 3,658.0 MiB | r8 §1.4's check, free with this run |
| `[40]` and `Final estimate` | compare with the table below | the answer |

**The repack path's values at chunk 40, already in the freeze logs** (*host*, `lab/logs/p/ppl-c3f-<m>-q4_0-*-wiki.log`):

| | `-b 512`, `[40]` | `-ub 1`, `[40]` | full set (for scale) |
|---|---|---|---|
| Yi | 7.6966 | 7.6817 | 8.3875 / 8.3855 |
| Llama-2 | 7.4071 | 7.4075 | 7.7435 / 7.7442 |
| DeepSeek | 8.2231 | 8.2192 | 8.7311 / 8.7306 |

**Time** (*estimated* from the freeze logs: 2.5 to 2.8 s per chunk at `-b 512`, 9.1 to 11.0 s at `-ub 1`): about
8 to 9 minutes per subject, 25 to 30 minutes for the six runs. KleidiAI was 5 to 15 % slower than the repack path
on the M4 in campaign 1's notes, so allow 35.

**Reading I would pre-state** (the lead's to write; this follows R61's):
- Budgets unchanged (8.8069 / 8.1307 / 9.1677 at `-b 512`; 8.8048 / 8.1314 / 9.1671 at `-ub 1`).
- Report the shift at chunk 40 per subject and setting. My prediction: within −0.3 to +1.0 % on each; P(at least
  one subject outside ±0.3 %) 0.5.
- If any shift exceeds ±0.1 %, run the full set on that build for all three (six runs, about 7.5 h by the freeze
  logs), because the second reference must be a full-set value before any lever is read against it.
- If all are inside ±0.1 %, note it and keep the freeze values as both references until a lever round needs the
  full-set value.
- A shift is not a budget event: no lever is on. It changes how much room the levers have.

**What it does not answer:** the phone's dotprod kernel for the 8-bit head (the M4 uses i8mm), and the head's
re-quantisation alone. The second needs `VOLLIE_HEAD2_REPACK=1`, which is only in the kit's tree; a third build
from `lab/llama.cpp-c3r32` would isolate it. I would ask for that only if the first run shows a shift.

## 4. The pack rate (next question 4)

### 4.1 Source

- Correction to r7 §1.1: the routine moves the nibbles **four bytes at a time**, not two. The gemm kernel's `kr`
  is 8 (`…16x4_neon_dotprod.c` line 41), so a segment is 4 bytes and the routine takes its 32-bit branch: one
  4-byte load, one XOR with 0x88888888, one 4-byte store, per row and segment. Per block of four rows: 4 two-byte
  scale copies and 16 such moves, for 72 bytes.
- It reads four source rows at once, 2,304 bytes apart for a 4,096-wide matrix and 6,192 apart for an 11,008-wide
  one, and writes one sequential stream.
- The loader calls it tensor by tensor on one thread (`set_tensor` inside the load loop).
- The head takes a different, slower road: each Q6_K row is dequantised to float, scanned for its maximum, and
  rounded to int8 (r8 §1.2). Not timed.

### 4.2 Host timing (*host*, one thread, relative only)

`bonnie_pack_time.c`: the library's own routine on random Q4_0 bytes, a ring of 24 matrices (0.6 GB) so that
source and destination are out of cache, best of three, destination touched first. A self-check re-derives one
packed block.

| shape (rows × inputs) | GB/s of source | check |
|---|---|---|
| 11,008 × 4,096 | 4.43 | passed |
| 4,096 × 11,008 | 4.14 | passed |

Apple M4 Pro, `cc -O3`, load average 23 at the time. So this is a floor for the laptop, and says nothing direct
about the phone.

### 4.3 Price for the phone (*estimated*)

- A78 against the M4 on scalar copy-like code: a third to a sixth. **0.7 to 1.5 GB/s on one thread.**
- Head re-quantisation: 131 to 421 million weights of scalar float work; 1 to 4 s.

| first load | body bytes | read served file | pack | write side file (R69: 0.45 to 0.61 GB/s, *device*) | total |
|---|---|---|---|---|---|
| today (anonymous, every load) | 3.1 to 3.6 GB | not measured cold | 2 to 5 s | – | 4 to 10 s with the head, plus the read |
| side file, CPU-only | 3.4 to 3.8 GB written | the same | the same | 5.5 to 8.5 s | **12 to 18 s** |
| side file beside a lane (share 0.37) | 2.2 to 2.6 GB written | the same | 1.5 to 4 s | 3.6 to 5.8 s | **8 to 13 s** |

- The pack and the write can overlap (the kernel writes back while the next tensor is packed), so the totals are
  upper ends of a sum.
- Every later load skips the read of the body, the pack and the head's re-quantisation.
- r7's "5 to 30 s" narrows to 8 to 18 s. The largest term left unmeasured is the cold read of the served file.
- A phone number costs one line in a kit: pack one 25 MB matrix 8 times, print GB/s. Worth adding to the
  residency kit (r8 §2) as `x5 pack gbs= src_mb= reps= check=`.

## 5. Next questions, ranked

1. **Has any campaign-3 host screen to date been read against the wrong reference?** The k,v-dense rule's +3.40 %
   and +3.55 % (*host*) and Tom's 0006 emulation are on the repack path. With a second reference they need a
   second reading. A list of every campaign-3 perplexity figure and the build it came from; mine, from the logs.
2. **What in KleidiAI's path differs numerically from the repack path for Q4_0?** The activation quantiser
   (Ebony r5 notes a rounding difference), the float order, or both. Source reading; it says whether the shift
   is random or one-sided, and so whether Granite's +0.57 % is the likely sign for us.
3. **The head knob for campaign 3:** keep the head in KleidiAI (int8) or send it to the repack type. It changes
   the buffer and the side file by 131 to 421 MB and the head's kernel. Needs item 2 and the 40-chunk run.
4. **Cold read rate of the served file on the phone** (§4.3's missing term); a kit line.
5. **Fixed share: which C to tune for**, after R75.
6. **Ebony's `H-max`.**

## 6. Limits

- §1 quotes the written terms and campaign 2's record; I did not ask whether the owner's option (a) was meant
  for later campaigns. The lead should confirm it with the owner rather than take my reading.
- §0 item 2's band is an analogy from two sparse models with a small KleidiAI share to three dense ones.
- §3's build line was not run; the reuse of the fetched sources is untested.
- §4.2 is one laptop under load; §4.3's ratio to the A78 is a judgement.
- I did not open R61's per-chunk logs; the values are those of campaign 2's report.

## 7. Sources (2026-11-23)

Repository: `campaigns/03-dense15-5pct/CAMPAIGN.md` (lines 14, 37, 48); `campaigns/COMMON_RULES.md` (rules 2, 3);
`campaigns/02-25toks-5pct/CAMPAIGN.md` (lines 100 to 104), `STATUS.md` (lines 70 to 72, 141 to 150),
`prereg/R61-kleidiai-host-ppl.md`, `reports/README.md` (R61); `tools/ppl.sh`, `tools/c3_freeze.sh`;
`lab/llama.cpp-c3/build/CMakeCache.txt`, `lab/llama.cpp-c1h6/build-kai/CMakeCache.txt`,
`lab/c3/head/build-kai/CMakeCache.txt`; `lab/logs/r61/*.load.log` and `queue.txt` (kernel lines, command form);
`lab/logs/p/ppl-c3f-*-q4_0-*-wiki.log` (seconds per chunk, `[40]`, final values);
`lab/llama.cpp-c3r32/ggml/src/ggml-cpu/kleidiai/kleidiai.cpp` (lines 315 to 421) and `kernels.cpp` (lines 490 to
536); `lab/llama.cpp-c3/ggml/src/ggml-cpu/CMakeLists.txt` (lines 650 to 690); KleidiAI
`kai_rhs_pack_nxk_qsi4c32pscalef16_qsu4c32s16s0.c` and `…16x4_neon_dotprod.c` (lines 40 to 42); `results.csv`
(budgets); `reports/02_methods.md` line 62 (KleidiAI 5 to 15 % slower than repack on the M4, campaign 1).
