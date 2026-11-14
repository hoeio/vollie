# Memo 07: campaign 1's exact levers ported to the campaign-2 series, checked on OLMoE

Agent model: claude-opus-5-5

Date 2026-11-14. Author: Tom (code). Patches `patches/llama.cpp-c2/0002` to `0007` on top of `0001-olmoe-expert-skip`
(base `620d2225c`), knobs in [the series README](../../../patches/llama.cpp-c2/README.md). Source material: campaign 1's
`patches/llama.cpp/0001-cpu-runtime.patch` and `0003-sampler-bypass.patch` (read only, not edited). **No speed is stated
anywhere.** Every number below is *host* (Apple M4 Pro, build flags of `tools/reproduce.sh`:
`-DGGML_NATIVE=OFF -DGGML_CPU_ARM_ARCH=armv8.2-a+dotprod+fp16`, Metal and BLAS off, `-t 4`, while an 8-thread
perplexity sweep ran on the same machine). Perplexities cover **4 chunks only**: they are identity checks, not quality
results, and are never compared with a budget. Node, barrier and sync counts are *host* counts of this build's graph,
not phone counts.

## 1. What exists

The series is stacked: apply `0001` to `0007` in order to a clean base. Each patch was applied in sequence to a fresh
worktree and built after each step (macOS host build after every patch; Android arm64 cross-build of `ggml-cpu` and
`llama` with NDK 28.2, `armv8.2-a+dotprod+fp16`, for the whole series). The applied series reproduces the development
tree file for file. All knobs are new `VOLLIE_` names and are **off when unset**; campaign 1's `GGML_` names do nothing
on this build (campaign 1 had `GGML_MMID=3`, `GGML_Q3K_X4`, `GGML_Q3K_KERN` and `GGML_MMID_FUSE` on by default, memo-02).

| patch | lever (campaign 1 name) | knob (off = unset) | files | lines +/− | identity, 4 chunks, `-b 512` / `-ub 32` / `-ub 1`, repack on and `-nr` | greedy text, 32 tokens, repack on and `-nr` |
|---|---|---|---|---|---|---|
| 0002-tiny-ops | small ops on thread 0, barriers between them skipped (`GGML_TINY_OP`) + `c2-graph` counter | `VOLLIE_TINY_OP=<elements>`, `VOLLIE_GRAPH_STATS=1` | ggml-cpu.c | +144 / −9 | identical at all six (65536 and 10^9) | identical |
| 0003-q3k-x4 | 4-row Q3_K dot (`GGML_Q3K_X4` + `ILP`) and in-register asm kernel (`GGML_Q3K_KERN`) | `VOLLIE_Q3K_X4=1` (intrinsics) / `=2` (asm) | arch/arm/quants.c, quants.h, ggml-cpu.c | +641 / −1 | identical at all six (mode 1 and mode 2) | identical |
| 0004-mmid-no-barrier | one-token `MUL_MAT_ID` without its internal barrier, dynamic chunks (`GGML_MMID=3`) | `VOLLIE_MMID_NB=1` | ggml-cpu.c | +138 / −13 | identical at all six (acts at `-ub 1` only) | identical |
| 0005-mmid-fuse | fused gate + up + SwiGLU (`GGML_MMID_FUSE`), works with 0001's skip | `VOLLIE_MMID_FUSE=1` (needs `VOLLIE_MMID_NB=1`) | ggml-cpu.c | +102 / −3 | identical at all six (acts at `-ub 1` only) | identical |
| 0006-expert-prefetch | expert read-ahead (`GGML_EXPERT_IO` bit 1) | `VOLLIE_EXPERT_IO=1` | ggml-cpu.c | +64 / 0 | identical at all six | identical |
| 0007-sampler-bypass | repetition-penalty fast path (campaign 1 `0003`) | `VOLLIE_SAMPLER_BYPASS=1` | src/llama-sampler.cpp | +27 / 0 | not applicable (perplexity does not sample); in the all-on run | identical with penalties (§3.3) |
| — | two-stage output head (`GGML_HEAD2`) | not implemented | | | declared numerics change; design in §5 | |

"Identical" means: the four per-chunk values printed by `llama-perplexity` are equal to the unpatched base's **and** the
md5 of the `--kl-divergence-base` file is equal. That file holds, for each of the 1,024 scored tokens, two exact floats
(the scale and the minimum log-probability, which contains the float log-sum-exp) and all 50,304 logits as 16-bit
steps of the token's own logit range, so the hash is a much finer test than four printed digits.

Where a lever acts: 0004 and 0005 change only one-token expert ops, so at `-b 512` and `-ub 32` they leave the
computation unchanged by design; `-ub 1` and the greedy generation are their real tests. 0002, 0003 and 0006 act at
every batch size. That the levers actually ran was confirmed in a throwaway debug build (counters, then reverted):
with `VOLLIE_Q3K_X4=1` or `=2`, 1,007,616 four-row calls in a 2-token generation; with `VOLLIE_MMID_NB=1` the new
path ran (144 calls); with `VOLLIE_EXPERT_IO=1` 1,980 `madvise` calls; with `VOLLIE_SAMPLER_BYPASS=1` the bypass fired
with 50,304 candidates. 0002 and 0005 show it in their own counter (§4).

## 2. Port notes (what differs from campaign 1)

- **0002.** Same rule as campaign 1 (exclusion list; `GET_ROWS` sized by output and indices; fused `RMS_NORM` + `MUL` tiny
  only if both nodes are). I re-derived the exclusion list from the base: every op that calls `ggml_barrier` or the
  shared chunk counter (`acc`, `set`, `count_equal`, `out_prod`, `diag_mask`, the convolutions, `flash_attn_ext`,
  `flash_attn_back`, `add_rel_pos`, `rwkv_wkv6`, `gla`, `gated_delta_net`, `cross_entropy_loss`, `mul_mat`,
  `mul_mat_id`) is excluded; `flash_attn_back` was missing in campaign 1's list (training only, harmless there). The
  fusion test was split out of `ggml_cpu_try_fuse_ops` into `ggml_cpu_fuse_count` so the lookahead makes exactly the
  loop's decision; with the knob unset the loop runs the stock code. Campaign 1's profiler, A/B and other knobs that
  shared this loop are not ported.
- **0003.** The intrinsics path (ILP) and the asm kernel are copied unchanged; the slow non-ILP fallback, the
  prefetch variant (`Q3K_PF`, closed in R23) and the kernel bench are left out. Campaign 1's single-row `Q3K_VEC`
  change is not ported: with 1,024- and 2,048-row experts every row goes through the 4-row path, so the stock
  single-row dot is never used for experts.
- **0004.** Ported without campaign 1's `MMID` bit 1 / bit 2 split (here one knob = both), without the anon cache,
  Q4 expert copies, page touch, `CHUNK2` and `GGML_MM_CHUNK`; chunks are 32 rows. 0001's skip runs inside the new
  path: `ggml_c2_skip_map` got a `primary` argument (only thread 0 zeroes rows and counts) and its one-time env read
  is now under ggml's critical section, because every thread reaches it at once on this path. Its results are
  unchanged (the skip runs below).
- **0005.** Ported as in campaign 1, plus a check that `src[3]` (the skip weights) is the same on both ops. The skipped
  rows get the same swiglu on zeroed inputs as the stock `GLU` gives them, so skip and fuse together equal skip
  alone (§3.2).
- **0006.** Bit 1 only; `MADV_RANDOM` (bit 2, lost on the device) and the next-layer guess (bit 4) are not ported. In a
  one-token graph the advised set is the op's own row map with 0001's skip applied.
- **0007.** Campaign 1's `0003-sampler-bypass.patch`, code unchanged, comment updated. **It lives in llama.cpp**
  (`libllama`, `src/llama-sampler.cpp`), not in the app. Campaign 1's second sampler bypass (skipping top-k / top-p /
  temperature at `temp <= 0`) lives in the app's bindings (`patches/llamadart/0001-…`), not in llama.cpp, and is not part
  of this series.

## 3. Check outputs (*host*, `olmoe-Mi.gguf`, `-c 512 -b 512 --chunks 4 -t 4`)

### 3.1 Knob off and each lever on

Base = clean `620d2225c` build. Every row below was produced identically by: the base; the full series with all
knobs unset; `VOLLIE_TINY_OP=65536`; `VOLLIE_TINY_OP=1000000000` (every non-excluded op tiny); `VOLLIE_Q3K_X4=1`;
`VOLLIE_Q3K_X4=2`; `VOLLIE_MMID_NB=1`; `VOLLIE_MMID_NB=1 VOLLIE_MMID_FUSE=1`; `VOLLIE_EXPERT_IO=1`; and all of them
together (`TINY_OP=65536 Q3K_X4=2 MMID_NB=1 MMID_FUSE=1 EXPERT_IO=1 SAMPLER_BYPASS=1`).

| batch | repack | [1] | [2] | [3] | [4] | md5 of the log-prob file |
|---|---|---|---|---|---|---|
| `-b 512` | on | 8.9646 | 10.4244 | 9.2074 | 9.4783 | `3c8948f1f71e32d43f01129a5e2feab8` |
| `-ub 32` | on | 8.9131 | 10.3435 | 9.0932 | 9.3987 | `ba2b600296cc2ab361b9b02aab535002` |
| `-ub 1` | on | 8.9550 | 10.4710 | 9.1889 | 9.4713 | `69f9df67b848ff8c1265cb8409a743f9` |
| `-b 512` | `-nr` | 8.9278 | 10.4339 | 9.2003 | 9.4521 | `8dd05b9e5b0fabb153b5516f4a4d20cf` |
| `-ub 32` | `-nr` | 8.8914 | 10.3906 | 9.1549 | 9.4018 | `32adfbc659ee3375902f859449ed0e00` |
| `-ub 1` | `-nr` | 8.9619 | 10.5154 | 9.2265 | 9.4738 | `851f77bf28e23041de3d38f63674a79c` |

The repack-on rows equal memo-04's base values. Greedy text (`llama-completion -p "The capital of France is" -n 32
--temp 0 -no-cnv`), the same for every configuration above, with and without `-nr`: "Paris. The Louvre Museum is one
of the world's largest and most famous art museums, located in Paris. The Eiffel Tower".

The suite was run twice: once on the series before the `syncs` counter was added to 0002, and once on the final
series (the one in the repository). Both runs gave the table above in every configuration.

### 3.2 Expert skip together with the exact levers

`VOLLIE_SKIP_MASS=900` alone (full series, other knobs unset) against `VOLLIE_SKIP_MASS=900` with `TINY_OP=65536 Q3K_X4=2 MMID_NB=1 MMID_FUSE=1
EXPERT_IO=1`: identical at all six settings (a numerics change against the base, as expected; these values are
not quality results):

| batch | repack | [1] | [2] | [3] | [4] | md5 |
|---|---|---|---|---|---|---|
| `-b 512` | on | 9.1525 | 10.7417 | 9.8247 | 10.1338 | `c2fc922bd730e4dbcf8d7a9306940b3d` |
| `-ub 32` | on | 9.1744 | 10.8070 | 9.8857 | 10.1427 | `c7d8b3538cbb8efe4bc77d0b9a1d4641` |
| `-ub 1` | on | 9.0107 | 10.6077 | 9.7258 | 10.0374 | `533cdbc997de192e0c170511979b01b4` |
| `-b 512` | `-nr` | 9.0953 | 10.7432 | 9.8350 | 10.1206 | `919e70d821f37e93de401f0c82e76d4a` |
| `-ub 32` | `-nr` | 9.0324 | 10.6648 | 9.7529 | 10.0752 | `02d5098b4c846e0417f97a97c7fdf852` |
| `-ub 1` | `-nr` | 9.1695 | 10.7052 | 9.8040 | 10.1124 | `0b7cfa94bace65f0c9f50e7aca2f60b1` |

A build with `0001` alone gives the same values and hashes at `-b 512` (repack on), `-ub 1` (repack on) and `-ub 1 -nr`
(the three settings rerun on it), so the refactor of `ggml_c2_skip_map` in 0004 did not change 0001's behaviour; the
`-b 512` row also equals memo-04's MASS=900 smoke value (10.1338). Greedy text with skip is the same with and without the levers, and so is the `c2-skip:` counter line (32-token
generation: 15.5887 % of pairs skipped with `-nr`, 15.7546 % with repack on, in both). So skip and fuse can be on
together.

### 3.3 Sampler bypass

`llama-completion ... -n 32 --ignore-eos --temp 0 --repeat-penalty 1.3 --frequency-penalty 0.2 --presence-penalty 0.1`:
`VOLLIE_SAMPLER_BYPASS=0` and `=1` give the same text (md5 `33d75a3090f10558eb38c679efd630f5` with repack on,
`1ec2c48dd233edab15cee081d75d3ef2` with `-nr`), and the `-nr` text equals the unpatched base's.

## 4. Per-token node, barrier and sync counts (*host*, not phone counts)

`VOLLIE_GRAPH_STATS=1`, `llama-completion`, 32 generated tokens, `-t 4`, mean over the 31 one-token graphs; the same with
repack on and with `-nr`. "ops run" = nodes that are not views and are computed (a fused group counts once); "barriers"
= barriers between ops (the closing one included); "syncs" = every `ggml_barrier` call, those inside `MUL_MAT` and
`MUL_MAT_ID` included.

| setting | graph nodes | ops run | ops run tiny | barriers between ops | syncs (all `ggml_barrier` calls) |
|---|---|---|---|---|---|
| stock (all knobs unset) | 937 | 501 | 0 | 501 | 646 |
| `VOLLIE_TINY_OP=65536` (0002) | 937 | 501 | 324 | 274 | 419 |
| `VOLLIE_MMID_NB=1` (0004) | 937 | 501 | 0 | 501 | 614 |
| `VOLLIE_MMID_NB=1 VOLLIE_MMID_FUSE=1` (0004 + 0005) | 937 | 469 | 0 | 469 | 582 |
| `TINY_OP=65536` + `MMID_NB` + `MMID_FUSE` | 937 | 469 | 308 | 242 | 355 |
| all levers (+ `Q3K_X4=2`, `EXPERT_IO`, `SAMPLER_BYPASS`) | 937 | 469 | 308 | 242 | 355 |

Reading: OLMoE's decode graph has 937 nodes, of which 501 run (the rest are views); stock, each is followed by a
barrier. With `VOLLIE_TINY_OP=65536` 324 of the 501 run on thread 0 alone and the barriers between ops fall from 501 to
274, the syncs from 646 to 419. Stock has 145 syncs inside ops per token (646 − 501: the `MUL_MAT`s, the
`MUL_MAT_ID`s and others); the no-barrier path removes the 32 of the one-token gate and up ops (646 → 614), and the
fused op removes 32 more ops and their barriers (16 layers: up, gate and `GLU` become one op). All exact levers
together: 355 syncs per token against 646 stock (−45 %, a *host* graph count). How much
time a barrier costs on the SD888 is a device question (campaign 1: on the SD888 the syncs were about half of each token, `patches/README.md`;
`docs/runtime-knobs.md`: 2 threads, 3.2 → 7.0 tok/s with `GGML_TINY_OP=65536`, *device*); nothing here measures it.

## 5. Two-stage output head for the untied head (design only, not implemented)

Left out because it is a declared numerics change and the exact levers came first. Design for `olmoe-Mi` on the phone
path (repack off):

- **What changes.** The head `MUL_MAT` (`output.weight`, Q6_K, [2048, 50,304], 84.5 MB, untied) at one token: a
  cheap copy scores all 50,304 rows, the M best rows are recomputed exactly with the stock Q6_K `vec_dot` on the same
  Q8_K activation, the other rows keep their screen logit. Greedy and top-M sampling see exact logits; the rest of the
  distribution is approximate, so perplexity changes and the result needs its own budget line (one budget for
  everything that changes numerics, CAMPAIGN.md).
- **Why repack off makes it simpler than memo-02 feared.** With `use_extra_bufts` off the head is a plain mmapped
  tensor (`CPU_Mapped`), so its plain rows are `src0` itself: no un-interleaving of `q6_K_8x4` and no saved mmap
  pointer is needed (memo-02 §4 options A and B were for the repacked head). The exact rows are then the stock
  computation of those rows, bit for bit, because the stock `-nr` head is the same `vec_dot_q6_K_q8_K` per row. With
  repack on (host default) the stock head is the repacked gemv and the exact rows match it only to float rounding.
- **Artefact.** A screen copy derived at runtime from the served file (allowed by COMMON_RULES rule 1, R42 amendment 2):
  `output.weight` dequantised and re-quantised to Q4_0, interleaved 4 rows at a time into `block_q4_0x4` for the
  dotprod gemv (campaign 1's `GGML_HEAD2_Q=q4_0`, the best device variant in R31). Size 50,304 × 2,048 / 32 × 18 B =
  57.9 MB (*estimated*); a Q2_K screen (campaign 1's first default) would be 33.8 MB (*estimated*).
- **Where the copy lives with repack off.** Not in a `CPU_REPACK` buffer (there is none with repack off): an anonymous
  mmap owned by the CPU module, made once at the first one-token head op (campaign 1's code path), or cached in the
  app's files and mmapped back (rule 1's derived-artefact clause) so that it can be evicted like the experts. Memory
  +57.9 MB anon (*estimated*) on top of memo-02 §5's figures, which drop the 235.5 MB repack copy when repack is off.
- **How many exact rows.** Campaign 1 used M = 2,048 of 100,352 rows (2.0 %; PPL +0.01 % on Granite, *host*). For
  50,304 rows the same share is about 1,024; the sweep should screen M ∈ {512, 1,024, 2,048, 4,096} for PPL with a
  batch-capable test mode (campaign 1's `GGML_HEAD2_BATCH`), because the head2 path only acts at one token and
  `-b 512` would otherwise never run it.
- **Bytes per token (*estimated*).** Stock 84.5 MB; two-stage 57.9 MB screen + M × 1,680 B (2,048 / 256 × 210 B per
  Q6_K row) = 59.6 MB at M = 1,024, about 25 MB less per token. Whether that is worth the numerics is Bonnie's pricing
  and the lead's call; at campaign 1's byte rate it would be about 1 ms (*estimated*).
- **Code.** By reading (*estimated*), most of campaign 1's head2 hunk carries over (the Q4_0 4x4 screen, the exact-row recompute, the
  top-M selection); the head detection needs `output.weight` instead of the tied `token_embd`, and the repacked-head
  branch can be dropped for the phone path. Knob proposal: `VOLLIE_HEAD2=<M>` (unset = off), `VOLLIE_HEAD2_BATCH=1`
  (test only).

## 6. Left out, and why

- The two-stage head (§5): numerics change, design only.
- Campaign 1's profiler (`GGML_CPU_PROF`), in-generation A/B (`GGML_CPU_AB*`), kernel bench (`GGML_CPU_BENCH`): not
  levers; the new `VOLLIE_GRAPH_STATS` counter covers node and barrier counts. A device round that needs per-op timing
  or ABBA blocks needs them ported first (the research kit relies on `GGML_CPU_AB`).
- `GGML_MMID_CHUNK2`, `GGML_MMID_TOUCH`, `GGML_EXPERT_ANON`, `GGML_Q3K_PF`, `GGML_EXPERT_IO` bits 2 and 4: measured no
  gain or lost on the device in campaign 1, or tied to the guess machinery.
- `GGML_SSM_INPLACE`, `GGML_CONV_RING`: not applicable to OLMoE (memo-02).
- Nothing was left out because it failed the identity check: every ported lever passed.

## 7. Not verified

1. Anything on the device: speed, the SD888's barrier cost, and whether each lever helps on this model there.
2. Identity on the full wikitext-2 set (4 chunks were run) and on any model but `olmoe-Mi` (Granite, LFM2.5 not run).
3. Identity at thread counts other than 4 (the phone runs 2 to 4) and with OpenMP builds (this host build has none).
4. That the Android build behaves as the host build: only compiled (NDK 28.2), not run.
5. The asm kernel on a core without dotprod (falls back to mode 1 by compile-time test only).
6. 0004 and 0005 on prompt batches: they act only at one token, so a prompt is the stock path by construction.
7. Whether OLMoE's graph keeps up, gate and `GLU` adjacent under every build option (it does on this build: the fused
   op fired in every layer, 32 ops fewer per token).
8. The node and barrier counts on the phone build (flash attention and fusions may differ by build options).
