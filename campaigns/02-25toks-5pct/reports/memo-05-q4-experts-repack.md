# Memo 05: why expert skip was inert on the publisher's Q4_0 OLMoE file

Agent model: claude-opus-5-5

Date 2026-11-14. Author: Tom (code). Source read at base `620d2225c` (`lab/llama.cpp-base`, clean) and in the
patched tree `lab/llama.cpp-c2` (0001 applied). **No speed is stated anywhere.** Every PPL here is *host*
(Apple M4 Pro, `-t 4`, `--chunks 2` or 1, wikitext-2 test, `-c 512 -b 512`), a smoke check only, never compared
with a budget. Byte counts are read from the GGUF headers (*host*, exact arithmetic) or from llama.cpp's load log
(*host*). Statements about the phone are from reading the code (*estimated*), not measured.

Short answer: **the lead's hypothesis is right.** On this build every Q4_0 tensor, including the three 3-D
`*_exps` per layer, is repacked at load into a `CPU_REPACK` buffer. The CPU module then hands each expert
`MUL_MAT_ID` to the repack class's own forward function before the generic code is reached. That function ignores
`src[3]`, so the 0001 skip code never runs, and the counter is registered only inside that code. The graph side
works as intended. The 0001 README already says so ("Experts in a repacked (extra-buffer) type ... ignore
`src[3]` and run in full"); R51 amendment 1 picked a file that the patch documents as unsupported.

## 1. The code path for a Q4_0 expert matmul

Dispatch, in order:

1. `ggml_compute_forward` asks the extra buffer types first: `ggml/src/ggml-cpu/ggml-cpu.c:1748`
   (`ggml_cpu_extra_compute_forward`, `traits.cpp:12`). If the weight's buffer is `CPU_REPACK`,
   `extra_buffer_type::get_tensor_traits` (`repack.cpp:5227`) returns the tensor's traits, and
   `tensor_traits::compute_forward` (`repack.cpp:4586`, case `MUL_MAT_ID` at 4591) runs
   **`forward_mul_mat_id`, `repack.cpp:4783`**. It builds its own row map on thread 0 (4848 to 4864), then calls
   `gemv<block_q4_0, 4, 4, Q8_0>` once per (expert, row) (4908), which is
   `ggml_gemv_q4_0_4x4_q8_0`, `ggml/src/ggml-cpu/arch/arm/repack.cpp:230`. The generic
   `ggml_compute_forward_mul_mat_id` (`ggml-cpu.c:1554`, where 0001 hooks in at the end of its row-map block,
   1646 to 1658) is never entered.
2. Only without repack does the switch reach `ggml-cpu.c:1871`, the generic path. For Q4_0 the tiled path
   (`tiled/tiled.cpp:1227`) declines (it takes Q4_K/Q5_K/Q6_K only), so the work is `vec_dot` at `ggml-cpu.c:1537`,
   that is `ggml_vec_dot_q4_0_q8_0`, `arch/arm/quants.c:297` (NEON dotprod; `nrows = 1` without i8mm,
   `ggml-cpu.c:245`).

Who decides to repack:

- libllama offers the extra buffer types when `llama_model_params.use_extra_bufts` is true (default true,
  `src/llama-model.cpp:3347`; list built in `make_cpu_buft_list`, `llama-model.cpp:1155`). Per tensor it then
  takes the first buffer type whose `supports_op` accepts the tensor's op.
- `extra_buffer_type::supports_op`, `repack.cpp:5191`: `MUL_MAT` needs a 2-D weight (5194), **`MUL_MAT_ID`
  needs a 3-D weight (5208 to 5212)**. So yes, it applies to 3-D `*_exps` tensors. `token_embd` (used by
  `GET_ROWS`) is not repacked.
- `ggml_repack_get_optimal_repack_type`, `repack.cpp:4925`; the Q4_0 branch is 4974 to 4989:
  - `q4_0_8x8`: AVX2, or SVE + i8mm with a 256-bit SVE vector, and `ne[1] % 8 == 0`;
  - `q4_0_4x8`: NEON + i8mm (`__ARM_FEATURE_MATMUL_INT8`), `ne[1] % 4 == 0`;
  - `q4_0_4x4`: NEON + dotprod, `ne[1] % 4 == 0`.
  The ARM feature tests are compile-time macros (`ggml_cpu_has_dotprod`, `ggml-cpu.c:3817`;
  `ggml_cpu_has_matmul_int8`, 3833). This host build reports `DOTPROD = 1`, no `MATMUL_INT8`, so it picks
  **`q4_0_4x4`**. The verbose load log (*host*) shows 112 tensors repacked `q4_0_4x4` (16 layers × q, k, v, o, up,
  gate, down) plus `output.weight` (Q6_K) repacked `q6_K_8x4` (Q6_K branch, `repack.cpp:5061` to 5070).
- An Android arm64 build for the SD888 (`armv8.2-a+dotprod+fp16`, no i8mm, no SVE; the app's
  `android_armv8.2_2` variant, `reports/01_vollie_vs_llamacpp.md` §2) satisfies exactly the same tests, so it would
  pick the same `q4_0_4x4` and `q6_K_8x4` (*estimated*, from the code).

## 2. Memory cost of the repack

From the header of `OLMoE-1B-7B-0125-Instruct-Q4_0.gguf` (3,928,038,048 bytes on disk; *host*):

| repacked tensors | count | bytes |
|---|---|---|
| `ffn_{up,gate,down}_exps` Q4_0 (3-D) | 48 | 3,623,878,656 |
| `attn_{q,k,v,output}` Q4_0 | 64 | 150,994,944 |
| `output.weight` Q6_K | 1 | 84,510,720 |
| **total** | 113 | **3,859,384,320 (3,680.6 MiB)** |

The load log agrees: `CPU_REPACK model buffer size = 3680.60 MiB`; `CPU_Mapped` keeps only 55.3 MiB
(`token_embd`, norms, router) (*host*). With `-nr` the log shows `CPU_Mapped 3744.37 MiB` and no `CPU_REPACK`
buffer (*host*).

The repack buffer is a plain allocation (`ggml_backend_cpu_repack_buffer_type_alloc_buffer`, `repack.cpp:5168`
allocates through the CPU buffer type): **anonymous, unevictable memory** that the kernel cannot drop and re-read
the way it can drop mmapped file pages. The Android build would allocate the same 3.86 GB (*estimated*, same
selection and same block sizes; the repacked blocks are the same size as the originals).

Campaign 1's precedent (`models/granite-4.0-h-tiny-d6-e6.md`; R-log 2026-10-31): repacking an all-Q4_0 file
(3.6 GB) got the app killed on this 8 GB phone. **The same applies here:** 3.86 GB anonymous plus the KV cache
and compute buffers is a larger copy than the one that was killed. The mix `olmoe-Mi` repacks only
its Q4_0 attention and Q6_K head: 235,505,664 bytes (224.6 MiB, header arithmetic, *host*), and its Q3_K experts
stay mmapped.

Switches at the base:

- **Global:** `-nr` / `--no-repack` (`common/arg.cpp:2427`) sets `llama_model_params.use_extra_bufts = false`
  (`include/llama.h:353`). At build time, `GGML_CPU_REPACK=OFF` (`ggml/CMakeLists.txt:152`).
- **Per tensor: none.** `-ot <pattern>=CPU` does not help: an override to the CPU buffer type re-runs
  `select_weight_buft` over the CPU list *including* the extra types (`src/llama-model-loader.cpp:1241` to 1243).
  Checked (*host*): with `-ot _exps=CPU` all 48 expert tensors were still repacked and the buffer was still
  3,680.60 MiB.
- **The app:** it passes `useExtraBuffers: device.canCopyWeights(fileSize)`, true only if
  total RAM − file size ≥ 5 GB (`reports/01_vollie_vs_llamacpp.md` §1, citing `lib/device.dart:107`). On this
  phone (about 7.3 GB) that is false for both OLMoE files (3.93 GB and 3.07 GB), so the app loads them without any
  repack (*estimated*, from the cited rule).
- **Campaign 1's 0001** has no switch to turn repack off. Its repack-related knobs (`docs/runtime-knobs.md`):
  `GGML_REPACK_RELEASE` (pages out the mmapped source pages after a repack; it does not shrink the anonymous
  copy), `GGML_HEAD2_Q=q4_0` (repacks its own screening copy), `GGML_EXPERT_Q4_LAYERS` (makes repacked expert
  copies; dead, R33). Turning repack off was left to the app (`use_extra_bufts`, `patches/llamadart/0001-…`;
  research kit `KIT_NOREPACK`, `tools/checks/kit_runs.cpp:40`).

## 3. Why no `c2-skip:` line

**The skip logic was bypassed entirely, not applied in a path that does not count.**

- The graph side ran: `build_moe_ffn` does not depend on the buffer type, and with a knob set it puts the mixture
  weights into `src[3]` of the expert ops. The repack `supports_op` does not look at `src[3]`, so the op stays on
  the repack path.
- The CPU side never ran. The repacked `forward_mul_mat_id` (§1) neither reads `src[3]` nor zeroes any row. The
  0001 hook sits only in the generic `ggml_compute_forward_mul_mat_id`.
- The counter is registered (`atexit`) and the env vars are read on the first call of `ggml_c2_skip_map`. With no
  call there is no `atexit`, so there is no line. Every output row was computed in full, so the PPL is bit-identical
  to stock.

Confirmation runs (*host*, `--chunks 2 -t 4`, patched build; per-chunk values `[1]`, `[2]`):

| setting | repack | `[1]` | `[2]` | `c2-skip:` line |
|---|---|---|---|---|
| no knob | on (default) | 8.0895 | 10.2068 | none |
| `VOLLIE_SKIP_SUM=45 VOLLIE_SKIP_STATS=1` | on (default) | 8.0895 | 10.2068 | **none** (identical to stock) |
| no knob | off (`-nr`) | 8.0525 | 10.1423 | none |
| `VOLLIE_SKIP_SUM=45 VOLLIE_SKIP_STATS=1` | off (`-nr`) | 7.9087 | 10.0088 | printed: 2.5876 % of pairs, 2.5876 % of expert bytes skipped, 3.1401 % of batch bytes unread |

So with repack off, the existing 0001 skips Q4_0 experts with no change (the generic row map is type-independent).
The single-chunk verbose runs showed the same `[1]` values.

## 4. What making skip work for Q4_0 experts would take

**(a) In the repacked forward function**, `repack.cpp:4848` to 4866: call a skip routine on thread 0 after the
row map is built and before the barrier at 4866. Experts left with no rows are skipped by the existing
`cne1 == 0` test (4872). It looks like about ten lines, but there are three snags:
- `ggml_c2_skip_map` is `static` in `ggml-cpu.c`. Its row-map stride is wrong for this function: the generic map
  is indexed `row_id * n_ids * n_tokens` (`ggml-cpu.c:1476`), the repack map `row_id * ne12` (`repack.cpp:4846`). Reusing it
  as it is would write past each expert's slice. It needs a stride argument and a C export into C++.
- It would change only a host research path. The phone does not take this path, because the app turns repack off
  for this file and stock repack does not fit in memory (§2).
- With the knob on, the repacked kernel (`q4_0_4x4` gemv) and the generic one (`vec_dot`) are different numerics
  (see the table: stock `[2]` 10.2068 against 10.1423), so a frontier measured this way is not the phone's
  frontier.
Risk: low for correctness if the stride is handled. **Value: none for the campaign.** Not implemented.

**(b) Keep the experts un-repacked**, which today means `-nr` for the whole model, because no per-tensor switch
exists. The kernel is then the generic `MUL_MAT_ID` with `ggml_vec_dot_q4_0_q8_0` (dotprod), the same one the app
runs on the phone. Per-expert skipping **already works there with 0001 unchanged** (§3, last row). **Nothing to
implement:** rerun the R51 Q4_0 sweep with `-nr`.

No 0002 patch was written.

## 5. Consequences to raise with the lead (not decided here)

1. **The host references were not taken on the phone's kernel path.** The host screens (`lab/screen.sh`,
   R50) and the R51 sweeps ran with repack on (the default). On the phone the app runs both OLMoE files with repack
   off (§2). The kernels differ in numerics. Two-chunk stock values (*host*): Q4_0 file 10.2068 (repack) against
   10.1423 (`-nr`); `olmoe-Mi` 10.4244 against 10.4339 (only its attention and head are repacked). The R50
   reference PPLs and the R51 frontier are therefore numbers for the repacked host path. Whether the served
   reference has to be re-measured with `-nr` is a rule question for the lead. A full-set `-nr` run belongs on the
   owner's laptop.
2. The counter's `MB_per_token_selected` is slightly low by construction: it counts tokens on the first expert
   op, but in the last layer llama-perplexity passes only the output rows (`inp_out_ids`) into the FFN, so fewer
   pairs per token are seen. The percentages are ratios of pairs or bytes and are unaffected. In the `-nr` run,
   pairs/(tokens × 384) = 0.969 (*host*).
