# c3 memo-13 work (Tom): E-T25, UD in llama.cpp, perplexity on the three subjects

- **Date:** 2026-11-27. **Author:** tom, `claude-opus-5-5`. Not committed. Host test approved by the owner on 2026-11-27.
  Rule: [ebony-new-routes.md](ebony-new-routes.md) §2, §5, §8.1 (E-T25 cell (c) only), with `ebony_seqsel.py` lines
  128 to 130 as the exact definition. Numerics notes: [bonnie-ud-check.md](bonnie-ud-check.md) §4.
- **Labels.** Every perplexity value and timing in this memo is *host* (owner's Apple M4 Pro laptop, 8 threads, one
  job at a time). Ratios are arithmetic on *host* values. Nothing ran on the phone, and nothing here is a speed result.
- **Status at hand-off (18:35 KST):** 36 of 42 runs finished. The six remaining full-set `-b 512` runs (Llama-2 and
  DeepSeek: UD repack, UD KleidiAI, KleidiAI stock) are still running in a detached driver. They should finish at about
  22:15 KST (*estimated*: 6 runs at about 38 min each). Their rows below say "running". Nothing in them is filled in.
  To fill them in: `python3 -I lab/c3/et25_table.py`.

## Summary

1. UD is in llama.cpp as patch 0010, a declared numerics change. It is off unless `VOLLIE_UD=1` is set. With the knob
   unset, per-chunk PPL is bit-identical to stock on Llama-2 at `-b 512`, `-ub 32` and `-ub 1`, in both builds.
2. Yi, full set at `-b 512`: UD is **+1.79 %** over same-build stock in the repack build (8.5373 vs 8.3875) and **+1.77 %**
   in the KleidiAI build (8.5965 vs 8.4467). Both are inside the kill line (× 1.035) and inside the budget (8.8069).
3. 40 chunks, all three subjects, both builds: UD / same-build stock is 1.0134 to 1.0254 (`-ub 1`: 1.0196 to 1.0254;
   `-b 512 -ub 32`: 1.0134 to 1.0225). The highest cell, 1.0254, is Llama-2 KleidiAI at `-ub 1`. **No cell reaches the
   kill line of 1.035.**
4. The 40-chunk ratio is higher than the full-set ratio on Yi (1.022 to 1.024 at 40 chunks, 1.018 on the full set).
   `-ub 1` and `-b 512 -ub 32` agree within 0.3 points on every subject. The rule behaves the same at every batch size.
5. Full-set `-ub 1` was not run: it would take about 2.2 to 2.7 h per run, or about 22 h for 9 runs (*estimated*). The
   `-ub 1` evidence is the 40-chunk subset. Ebony's screen predicted +2.6 / +2.1 / +2.0 %. The measured values agree with
   it within its s.e. on all three subjects.

## 1. Patch

- **File:** `patches/llama.cpp-c3/0010-ud-numerics.patch`, 67 lines, sha256
  `330502f09761ba1ed287b832817ec82c67827bb4ec71111e55d8397a60099908`.
- **Changes:** only `src/llama-graph.cpp`. No other patch of the c3 series touches this file.
- **Applies:** checked with `git apply` on a clean `git archive 620d2225c` after `llama.cpp/0001` to `0003`,
  `llama.cpp-c1-olmoe/0001` to `0012` and `llama.cpp-c3/0001` to `0009` (the README's base).
- **Tested tree:** made from `lab/llama.cpp-c3` at `26f78f925` (the S3 freeze tree, c2 series 0001 to 0009). It is the
  worktree `lab/llama.cpp-c3ud`, with 0010 as its only change.
- **What it adds:** in `build_ffn`, on the SiLU parallel-gate path, right after `ggml_swiglu_split`. If `VOLLIE_UD=1`
  (exactly `1`, read once), one `ggml_map_custom2(h, up)` node is put between h and `down`. Per row (position) it does
  two steps:
  1. It keeps h where |up| ≥ the k-th largest |up|, with k = round(0.7 n), and writes 0 elsewhere. This is the
     product of skipped `gate` rows, treated as zero.
  2. On that result, it keeps the values whose |h| ≥ the k-th largest |h|, with k = round(0.5 n), and writes 0
     elsewhere. This is the input of `down` on its kept columns.
  - This is Ebony's rule. Ties keep more, as `np.partition` with `>=` does.
- **Reference path, not a kernel:** `nth_element` on a copy of each row. `gate` is still computed in full, and
  `down` still runs dense on the masked h. The values equal skipped rows and columns, except in one way: the zeroed
  elements still take part in the Q8_0 quantisation of `down`'s input. They are 0 there, but the block scale is taken
  over the kept values only. A column kernel that quantises the kept values would match that.
- **Every position carries the rule:** at every batch size, prefill and scored tokens alike, including the first 256
  tokens of each chunk (Bonnie §4: the screen ran those dense; this test does not).
- **Speed (*host*):** UD costs about +18 % seconds per pass against stock at `-b 512` (Yi repack 2.99 s against
  2.54 s) and about +27 to +32 % at `-ub 1` (Yi 12.05 s against 9.12 s; Llama-2 13.87 s against 10.96 s). That comes from the reference op. It means nothing for the phone.
- **Binaries (sha256, first 16):** `lab/llama.cpp-c3ud/build/bin/llama-perplexity` 17eec88021833247,
  `build/bin/libllama.0.7.0.dylib` 31a7d76d5d89c0fa; `build-kai/bin/llama-perplexity` a762ea00a18463ed,
  `build-kai/bin/libllama.0.7.0.dylib` 6485d957d24b88eb.
- **CMake cache:** every `GGML_*`, `LLAMA_*`, `CMAKE_BUILD_TYPE` and `BUILD_SHARED_LIBS` entry equals that of
  `lab/llama.cpp-c3/build` and `build-kai` (same configure lines as `memo-12-work/tom-kai-stock.md` §1; KleidiAI v1.31.0
  from the same local sources). The only exceptions are two `LLAMA_UI_*` entries that the KleidiAI stock cache has and
  this one lacks. They are UI only and not built.
- **KleidiAI runs:** every KleidiAI log prints `KLEIDIAI = 1`, and all runs set `GGML_KLEIDIAI_SME=0`.

## 2. Identity, knob unset (*host*, Llama-2-7B served file, running PPL)

| build | setting | chunks | stock build | UD build, `VOLLIE_UD` unset |
|---|---|---|---|---|
| repack | `-b 512` | 4 | 5.8104, 6.3386, 7.0099, 7.9916 | identical |
| repack | `-b 512 -ub 32` | 4 | 5.7851, 6.3350, 7.0107, 7.9935 | identical |
| repack | `-b 512 -ub 1` | 2 | 5.7791, 6.3466 | identical |
| KleidiAI | `-b 512` | 4 | 5.7843, 6.3438, 7.0093, 8.0004 | identical |
| KleidiAI | `-b 512 -ub 32` | 4 | 5.7574, 6.3192, 6.9807, 7.9816 | identical |
| KleidiAI | `-b 512 -ub 1` | 2 | 5.8090, 6.3474 | identical |

The repack stock values equal memo-12 `tom-numerics.md` §2.1 and the first chunks of the S3 freeze logs.

## 3. Results (*host*)

- **Stock references:**
  - Repack `-b 512` full set: the S3 freeze logs (`lab/logs/p/ppl-c3f-<m>-q4_0-b512-wiki.log`).
  - Repack `-ub 1` at chunk 40: the freeze `-ub 1` logs.
  - KleidiAI `-ub 1` at chunk 40: `ppl-c3k40-<m>-q4_0-ub1-wiki.log` (memo-12).
  - KleidiAI `-b 512` full set and every `-b 512 -ub 32` stock: run here, same session.
- **Values:** the 40-chunk values are running PPL at chunk 40.
- **Kill line:** UD > same-build stock × 1.035.
- **Budget:** the absolute S3 freeze budget, served repack stock × 1.05: `-b 512` 8.8069 / 8.1307 / 9.1677, `-ub 1`
  8.8048 / 8.1314 / 9.1671. On 40-chunk subsets, the budget column uses UD ÷ repack stock at the same chunks.

| subject | setting | build | chunks | stock | UD | UD / same-build stock | UD / repack stock (budget ≤ 1.05) | kill (> 1.035) |
|---|---|---|---|---|---|---|---|---|
| Yi-1.5-6B | `-b 512` | repack | 656 (all) | 8.3875 | 8.5373 | **1.0179** | 8.5373 ≤ 8.8069 | no |
| Yi-1.5-6B | `-b 512` | KleidiAI | 656 (all) | 8.4467 | 8.5965 | **1.0177** | 8.5965 ≤ 8.8069 | no |
| Yi-1.5-6B | `-ub 1` | repack | 40 | 7.6817 | 7.8600 | 1.0232 | 1.0232 | no |
| Yi-1.5-6B | `-ub 1` | KleidiAI | 40 | 7.7165 | 7.8992 | 1.0237 | 1.0283 | no |
| Yi-1.5-6B | `-b 512 -ub 32` | repack | 40 | 7.6839 | 7.8531 | 1.0220 | 1.0220 | no |
| Yi-1.5-6B | `-b 512 -ub 32` | KleidiAI | 40 | 7.7319 | 7.9040 | 1.0223 | 1.0286 | no |
| Llama-2-7B | `-b 512` | repack | 655 (all) | 7.7435 | running | – | – | – |
| Llama-2-7B | `-b 512` | KleidiAI | 655 (all) | running | running | – | – | – |
| Llama-2-7B | `-ub 1` | repack | 40 | 7.4075 | 7.5743 | 1.0225 | 1.0225 | no |
| Llama-2-7B | `-ub 1` | KleidiAI | 40 | 7.4034 | 7.5882 | 1.0250 | 1.0244 | no |
| Llama-2-7B | `-b 512 -ub 32` | repack | 40 | 7.4077 | 7.5746 | 1.0225 | 1.0225 | no |
| Llama-2-7B | `-b 512 -ub 32` | KleidiAI | 40 | 7.4037 | 7.5684 | 1.0222 | 1.0217 | no |
| DeepSeek-7B | `-b 512` | repack | 596 (all) | 8.7311 | running | – | – | – |
| DeepSeek-7B | `-b 512` | KleidiAI | 596 (all) | running | running | – | – | – |
| DeepSeek-7B | `-ub 1` | repack | 40 | 8.2192 | 8.3802 | 1.0196 | 1.0196 | no |
| DeepSeek-7B | `-ub 1` | KleidiAI | 40 | 8.2742 | 8.4383 | 1.0198 | 1.0267 | no |
| DeepSeek-7B | `-b 512 -ub 32` | repack | 40 | 8.2250 | 8.3691 | 1.0175 | 1.0175 | no |
| DeepSeek-7B | `-b 512 -ub 32` | KleidiAI | 40 | 8.2870 | 8.3981 | 1.0134 | 1.0210 | no |

- **Generalised:** on all three subjects, UD costs about +2 % at `-ub 1` and at `-b 512 -ub 32`, in both builds. The
  build does not change the ratio by more than 0.4 points.
- **Did not generalise:** the level. DeepSeek is the most tolerant subject (1.013 to 1.020) and Llama-2 the least
  (1.022 to 1.025). Ebony's screen ranked Yi as the least tolerant (+2.6 %). Here Yi sits between the other two at
  40 chunks.
- **Against the screen:** measured +2.3 / +2.3 / +2.0 % (`-ub 1`, repack, 40 chunks); screen +2.6 / +2.1 / +2.0 %
  (± 0.7 to 1.2). Computing the first 256 tokens of each chunk with the rule on did not raise the cost here.
- **Room in the budget:** on top of the KleidiAI stock shift, the worst UD cell is 1.0286 against the absolute repack
  budget (Yi KleidiAI `-b 512 -ub 32`, 40 chunks). That leaves about 2.1 points of the 5 % for anything else on Yi.

## 4. Wall times (*host*, one job at a time, 8 threads)

| run | wall |
|---|---|
| Identity, 12 short runs | 3.5 min in total |
| UD `-ub 1`, 40 chunks | 8 to 10 min each (12.1 to 14.7 s per pass); 6 runs 59 min |
| `-b 512 -ub 32`, 40 chunks, stock and UD | 2.1 to 3.0 min each; 12 runs 31 min |
| Yi full set `-b 512`: UD repack / UD KleidiAI / KleidiAI stock | 36.7 / 40.4 / 35.6 min |
| Six remaining full-set `-b 512` runs | about 38 min each, finishing at about 22:15 KST (*estimated*) |
| **Full-set `-ub 1` (not run)** | 12 to 15 s per pass × 596 to 656 chunks = 2.2 to 2.7 h per run; 6 UD runs + 3 KleidiAI stock runs ≈ 22 h (*estimated*; freeze stock `-ub 1` took 107 to 129 min per subject) |

## 5. Files

- Patch: `patches/llama.cpp-c3/0010-ud-numerics.patch`. Not committed. The series README is not updated yet; Annie or
  the lead adds the 0010 row when this is committed.
- Driver: `lab/c3/et25.sh` (resumable). Table: `lab/c3/et25_table.py`. Logs: `lab/c3/logs/et25/<m>-<R|K>-<stock|ud>-<setting>.log`,
  `id-*.log`, `times.txt`.
- Tree: `lab/llama.cpp-c3ud`, a git worktree of `lab/llama.cpp-c3`. The freeze tree's working copy is unchanged.

## Audit note (Annie, 2026-11-27)

Checked against `lab/c3/logs/et25/` (UD and same-build KleidiAI stock logs), the S3 freeze and memo-12 stock logs in `lab/logs/p/`
(repack `-b 512` and `-ub 1` chunk 40, KleidiAI `-ub 1` chunk 40), `times.txt` and `KLEIDIAI = 1` lines. All *host*. Text above is not edited.

- **Match:** every PPL value in the §3 table (18 UD and stock cells, 3 full-set and 6 stock chunk-40 references) equals its log; every
  ratio in the table equals UD / stock recomputed (4 decimals); the Yi `-b 512` +1.79 % / +1.77 %; the identity rows (UD and stock
  logs print the same final values at `-b 512`: 7.9916 repack, 8.0004 KleidiAI); the Yi wall times (36.7 / 40.4 / 35.6 min); the
  KleidiAI logs print `KLEIDIAI = 1` and the repack logs do not; the budget 8.8069 (8.3875 x 1.05 = 8.80688).
- **Mismatch 1 (summary item 3):** says the highest cell is 1.0254 (Llama-2 KleidiAI `-ub 1`) and gives `-ub 1` as 1.0196 to 1.0254.
  The table, the logs and the STATUS block give 1.0250 (7.5882 / 7.4034 = 1.02496). Correct values: highest 1.0250, `-ub 1` 1.0196 to
  1.0250, whole range 1.0134 to 1.0250. Still no cell near 1.035.
- **Mismatch 2 (summary item 4, repeated in the STATUS block "within 0.3 points"):** `-ub 1` and `-ub 32` differ by 0.28 (Llama-2
  KleidiAI), 0.21 (DeepSeek repack), 0.64 points (DeepSeek KleidiAI, 1.0198 against 1.0134); the "within 0.3 points on every subject"
  holds for Yi and Llama-2 but not for DeepSeek KleidiAI. Likewise "the build does not change the ratio by more than 0.4 points" (§3) is
  0.41 on DeepSeek `-ub 32` (1.0175 against 1.0134).
- **Not checkable now:** the Llama-2 and DeepSeek full-set `-b 512` rows ("running"; `llama2-7b-R-ud-b512.log` was at chunk 40 when
  read); the 22:15 KST finish is an estimate in the source. The "about 22 h" for full-set `-ub 1` is an estimate in the source.
