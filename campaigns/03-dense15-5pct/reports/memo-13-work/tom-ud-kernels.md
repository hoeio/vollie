# c3 memo-13 work (Tom): UD's speed side on the host: E-T28, E-T27 + B-T2, B-T1, and the price at those costs

- **Date:** 2026-11-28. **Author:** tom, `claude-opus-5-5`. Not committed. The owner chose this host test on 2026-11-28.
  Tests as written in [ebony-new-routes.md](ebony-new-routes.md) §8.1 (E-T27, E-T28) and [bonnie-cpu-ladder.md](bonnie-cpu-ladder.md)
  §7 (B-T1, B-T2), folded as [bonnie-ud-check.md](bonnie-ud-check.md) §6 asks. UD as in [tom-et25.md](tom-et25.md) (patch 0010).
- **Labels.** Every timing and ratio here is *host* (owner's Apple M4 Pro, one thread, macOS, not pinned). **Host ratios
  are relative only. The phone sets the size**, and memo-05 already showed that the M4 and the A78 disagree on the cold
  cost of these kernels (K1 1.53 to 1.64 here against 1.99 to 2.24 on the phone, R72). Byte counts are exact counts.
  Prices are *estimated* (arithmetic on Bonnie's model). Perplexity values are *host*. Nothing ran on the phone.

## Summary

1. **E-T28:** K1 on `down` (4096 × 11008, all three subjects) at keep 0.5 with real UD masks costs **1.66 to 1.70 per
   touched weight hot and 1.64 to 1.71 cold** at TR 128 (cold ÷ hot about 1.0). That is 0.83 to 0.86 of the dense packed gemv's time.
   K1's layout reads **0.5556** of `down`'s bytes, exactly Bonnie's (4 + 32 × 0.5) ÷ 36. TR 512 is dead cold (3.70 to 3.89).
2. **E-T27:** in-place Y1b on the real 70 % of `gate` rows (by |up|) costs **1.48 to 1.52 hot and 2.57 to 2.60 cold** per
   touched weight. Cold, 70 % of the rows take 1.80 to 1.82 × the dense packed time, which is more than dense Y1b on all rows
   (1.53 to 1.78). **Ebony's line (over 1.3 cold, so R1 is about −4 ms) fires on the host.** No arm helps: the prefetch arms are worse.
3. **B-T2:** dense Y1b costs 1.46 to 1.49 hot on all three shapes. Cold it costs 1.50 (4096²), 1.53 to 1.78 (11008 × 4096) and
   **2.41 to 2.46 on `down`** (4096 × 11008). The plain `--no-repack` vec_dot costs 2.0 there. Removing the zeroing moves costs
   +0.05 to +0.1, and the next-group prefetch takes cold to 2.8. **Both B-T2 kill lines fire on the host when cold** (`down`
   is over 1.8 with every arm, and the whole model is about 1.8 at one thread). Hot, neither fires.
4. **B-T1:** Y1b in the decode path (new patch 0011, `VOLLIE_Y1B=1` with `--no-repack`) gives **per-chunk PPL identical to
   the repack build** on Llama-2 at `-b 512`, `-ub 32` and `-ub 1`. With the knob unset it is identical too. The plain
   `--no-repack` path differs in every chunk, so the test can see a difference.
5. **Price (*estimated*, C 50, Y1b base):** at the host costs, A′ UD is **−23.5 / −23.4 / −21.7 ms hot and −18.7 / −20.3 / −18.8 ms
   cold**. Bonnie's inputs give −22.8 / −22.7 / −21.0. Path A (R1 only) is −9.4 / −9.3 / −8.6 hot, −6.1 / −6.2 / −5.7 cold, and
   +2.0 ms if the skipped rows save no bus time. R3 is not dead: break-even K1 is 4.5 to 4.7 at C 50 and 3.15 at C 40, against
   1.64 to 1.71 measured cold. But Ebony's E-T28 hot line (1.6) fires, just barely, so B2's price needs a device cell before any build.

## 0. Setup

- **Masks from a real decode (E-T25's path):** `lab/c3/bench/ud_dump.cpp` links `lab/llama.cpp-c3ud/build` (patch 0010).
  It runs with `VOLLIE_UD=1` on `wiki.test.raw`: it prefills 128 tokens, then decodes 8 tokens one at a time
  (teacher-forced, ubatch 1). It reads `ffn_up-<il>` and `ffn_vollie_ud-<il>` through `cb_eval`.
  - `gate` row mask: |up| among the 70 % largest (0010's rule).
  - `down` input mask: `ffn_vollie_ud != 0`.
  - 256 / 256 / 240 masks (8 tokens × 32 / 32 / 30 layers). Realised keep: 0.7000 rows and 0.5000 inputs on every
    subject.
  - Every `down`-kept input is also `up`-kept (1.0000).
  - Kept inputs per 32-block: min 4, 5th percentile 11, mean 16, 95th percentile 21, max 28 on all three subjects. No
    block is empty.
  - Mask overlap between consecutive tokens: 0.54 / 0.57 / 0.55.
- **Bench:** `lab/c3/bench/ud_bench.c` includes `tools/c3_colkern_bench.c` unchanged (memo-05's K1, Y1b, stock 4x4 copy)
  and links the freeze tree's `libggml-cpu`, the same way as memo-05.
  - Weights are random. The kernels' time does not depend on the values.
  - Single thread, best of 3 trials of 90 calls.
  - **hot** = the first 1024 rows repeated (2.4 to 6.3 MB). **cold** = at least 1 GiB of copies used round-robin (41 to 43
    copies, far past the SLC).
  - Masks rotate per call, so every call has a different mask and a different copy.
  - **k** = time ÷ (touched share × dense time of the stock packed 4x4 gemv in the same regime).
  ```sh
  B=$PWD/lab/llama.cpp-c3/build/bin
  clang -O3 -march=armv8.2-a+dotprod+fp16 -DGGML_LINK -Wno-unused-function -o lab/c3/bench/ud_bench lab/c3/bench/ud_bench.c -L$B -lggml-cpu -lggml-base -Wl,-rpath,$B
  VOLLIE_UD=1 lab/c3/bench/ud_dump lab/c3/models/<m>-q4_0.gguf lab/c3/text/wiki.test.raw lab/c3/bench/udmask-<m>.bin 128 8
  lab/c3/bench/ud_bench lab/c3/bench/udmask-<m>.bin 4096 90; lab/c3/bench/ud_bench x 4096 90 dense   # 4096 x 4096
  ```
  Each was run twice (r1, r2). Logs: `lab/c3/logs/udk/bench-<m>-r<n>.log`, `dump-<m>.log`.
- **Identity in the bench (`memcmp`):**
  - K1 TR 128 / TR 512 on real masks 0 to 3 equals the stock 4x4 gemv on the masked x.
  - Every Y1b arm, dense and on the real row list, equals the stock 4x4 gemv (on the same rows).
  - The bench's stock copy equals the library's `ggml_gemv_q4_0_4x4_q8_0`.
  - All YES, on all three subjects.

## 1. E-T28: K1 on `down` at keep 0.5, real masks (*host*)

| `down` 4096 × 11008 | Yi (r1, r2) | Llama-2 | DeepSeek | time ÷ dense stock (all) |
|---|---|---|---|---|
| K1 TR 128, hot, k | 1.67, 1.67 | 1.70, 1.67 | 1.66, 1.67 | 0.83 to 0.85 |
| K1 TR 128, cold, k | 1.71, 1.66 | 1.66, 1.66 | 1.64, 1.66 | 0.82 to 0.86 |
| K1 TR 512, hot, k | 1.70, 1.71 | 1.74, 1.70 | 1.70, 1.70 | 0.85 to 0.87 |
| K1 TR 512, cold, k | 3.89, 3.74 | 3.74, 3.70 | 3.76, 3.77 | 1.85 to 1.94 |
| K1 layout lines touched ÷ all (TR 128 and 512; min = max over every mask) | 0.5556 | 0.5556 | 0.5556 | (4 + 32 × 0.5) ÷ 36 = 0.5556 |

- **Byte count:** exact. Every (tile, block) has a kept input, so its scales are always read. The `down` bytes at keep 0.5
  are **0.5556** of the file's `down`, as Bonnie expected.
- **Against memo-05 (random masks, keep 0.60):** TR 128 was 1.53 to 1.64. Keep 0.5 costs about 0.05 more per touched
  weight. That is the per-block float step spread over fewer columns (0.043 ÷ d in memo-05's count). Real masks cost the
  same as random ones: kept counts per block range 4 to 28 and the cost does not move.
- **E-T28 kill lines, as written:**
  - "Over 1.6 per touched weight hot on the M4": 1.66 to 1.70 hot, so **it fires, by 0.06 to 0.10**. B2's price is cut
    and needs a device cell before any build.
  - "Cold ÷ hot over 1.3": 0.98 to 1.02 at TR 128, so **it does not fire**. At TR 512 it is 2.2, so TR 512 is out.

## 2. E-T27 + B-T2: in-place Y1b, `gate` row skip and dense, with arms (*host*)

### 2.1 `gate` (11008 × 4096), kept rows 0.700 by real |up|

Kept rows hot 0.698 (rows < 1024). Row lists are trimmed to a multiple of 4.

| arm | hot k (6 runs) | cold k (6 runs) | cold time ÷ dense stock |
|---|---|---|---|
| Y1b-rows (as is) | 1.48 to 1.52 | **2.57 to 2.60** | 1.80 to 1.82 |
| zeroing moves removed | 1.57 to 1.59 | 2.68 to 2.86 | 1.87 to 2.0 |
| next-group prefetch (one line per row and block) | 1.46 to 1.49 | 2.79 to 2.85 | about 2.0 |
| both | 1.51 to 1.55 | 2.78 to 3.21 | 2.0 to 2.2 |
| burst prefetch of the next group's rows (my arm) | 1.73 to 1.79 | 3.40 to 3.61 | 2.4 to 2.5 |
| for reference: dense Y1b on the same shape, all rows | 1.47 to 1.49 | 1.53 to 1.78 | 1.53 to 1.78 |

- **Cold:** skipping 30 % of the rows in place is **slower than computing all of them with dense Y1b** on this host.
  - This is memo-05 E6's reading (1.92 to 3.36 on random rows) confirmed with real masks.
  - Rows are 2,304 B (36 lines). A kept-row list breaks the stream at every skipped row.
  - The M4's prefetcher handles four dense streams well and a jumping list badly.
  - Software prefetch does not cure it here. Both forms make cold worse.
- **E-T27 kill line, as written:** over 1.3 per touched weight cold → R1's price falls to about −4 ms and R1 is not worth a
  build. Measured 2.57 to 2.60, so **it fires**. The phone's prefetchers are different, and only a device cell (memo-05's
  D2 `y1b_rows` cold row) can overturn this.

### 2.2 B-T2: dense Y1b at three shapes (k against the packed 4x4 gemv)

| shape | regime | lib 4x4 gemv | `--no-repack` vec_dot | Y1b | zeroing removed | prefetch | both | burst prefetch |
|---|---|---|---|---|---|---|---|---|
| 4096 × 4096 | hot | 0.99 to 1.01 | 2.00 to 2.01 | 1.46 | 1.57 to 1.58 | 1.46 | 1.51 to 1.53 | 1.74 |
| | cold | 1.00 to 1.01 | 2.00 to 2.01 | **1.50 to 1.53** | 1.61 to 1.62 | 2.80 to 2.81 | 2.79 to 2.80 | 3.42 to 3.43 |
| 11008 × 4096 (`gate`, `up`) | hot | 0.99 to 1.01 | 2.01 to 2.04 | 1.47 to 1.49 | 1.55 to 1.59 | 1.46 to 1.48 | 1.52 to 1.55 | 1.70 to 1.80 |
| | cold | 0.99 to 1.02 | 1.97 to 2.00 | **1.53 to 1.78** | 1.63 to 1.94 | 2.63 to 2.82 | 2.61 to 2.81 | 3.21 to 3.46 |
| 4096 × 11008 (`down`) | hot | 0.99 to 1.01 | 1.99 to 2.01 | 1.45 to 1.49 | 1.52 to 1.58 | 1.45 to 1.47 | 1.51 to 1.54 | 1.80 to 1.84 |
| | cold | 0.99 to 1.02 | 1.99 to 2.09 | **2.41 to 2.46** | 2.49 to 2.54 | 2.78 to 2.84 | 2.80 to 2.86 | 3.52 to 3.67 |

- **Zeroing moves removed** (the `sdot` chains run on and each block's sum is a difference of running sums, which is
  exact): this is slower on the M4, by 0.05 to 0.1. The four `MOVI` are free here, and the loop-carried chain is not.
  The A78 may differ (Bonnie's count gives 1.24 without them). This arm is cheap to put in a device cell.
- **Prefetch:** neutral hot, and much worse cold. Do not build it.
- **Long rows cold:** on `down`, Y1b (2.41 to 2.46) is slower than the plain one-row `vec_dot` (2.0). That is four
  concurrent 6.2 KB streams against one. memo-05 had 2.07 to 2.21 at 14,336 and 18,944 inputs.
- **Whole model, one thread** (weights: attention 1 − F at the 4096² cost, `gate` + `up` 2F/3, `down` F/3; F 0.782 / 0.668 /
  0.668): hot about 1.47 on all three. Cold **1.82 / 1.78 / 1.78** (*estimated* from *host* cells; Yi's 4096 × 512 `k`/`v`
  were not benched and are taken at the 4096² cost).
- **B-T2 kill lines, as written:**
  - "Cold `down` above 1.8 with every arm → price A1 at its low band": **fires** (best arm 2.41).
  - "Whole-model cost above 1.7 at one thread → Y1b is worth under +0.6 tok/s, stop": **fires cold** (1.78 to 1.82) and
    **does not fire hot** (1.47). The rule does not say which regime counts. A decode streams the body from DRAM on the
    phone, so cold is the relevant regime there. But the M4's cold behaviour is exactly what memo-05 said the phone may
    not share. This is a device question.
- Four pinned threads (B-T2's second arm) were not run. macOS cannot pin threads, and the host's multi-thread timing
  is bus-shared and says nothing about the phone.

## 3. B-T1: Y1b in the decode path, identity (*host*, Llama-2-7B served file, running PPL)

**Patch:** `patches/llama.cpp-c3/0011-y1b-decode.patch`, 85 lines, sha256
`89bc4874d90e1436527c8e4b3c45984a45e84125f93b3a1fd36f3287725f3b19`. Not committed.
- **Scope:** it changes only `ggml/src/ggml-cpu/ggml-cpu.c`. It applies on the freeze tree (`26f78f925`) and after 0010
  (checked with `git apply` on a clean archive).
- **Knob:** `VOLLIE_Y1B=1` (exactly `1`, read once). It is compiled only with NEON + dotprod + fp16 vector arithmetic;
  otherwise the knob reads 0.
- **What it changes:** on the generic `mul_mat` path, a Q4_0 `src0` (plain rows, so extra buffers are off, for example
  `--no-repack`) skips both llamafile sgemm calls. In each 16-row tile, every position runs Y1b on groups of four rows.
  A tail group repeats the last row pointer and stores only the valid rows. Only `num_rows_per_vec_dot == 1` takes this path.
- **Quantisation of the input is unchanged:** `quantize_row_q8_0` on the generic path. memo-05 T4 showed it equals
  `quantize_mat_q8_0_4x4` per row, and that the 4x4 gemm equals the gemv per row.

Build: worktree `lab/llama.cpp-c3y1b` of `lab/llama.cpp-c3` at `26f78f925`, with 0011 as its only change. The
configuration is the same as `lab/llama.cpp-c3/build`: every `GGML_*`, `LLAMA_*`, `CMAKE_BUILD_TYPE` and
`BUILD_SHARED_LIBS` entry of the cache was diffed, with no difference. Binaries (sha256, first 16 hex digits):
`build/bin/libggml-cpu.0.27.0.dylib` aa3342b631400c51, `build/bin/llama-perplexity` e8bf3742b7fc1a22.
Command: `llama-perplexity -m lab/c3/models/llama2-7b-q4_0.gguf -f wiki.test.raw -c 512 -b 512 -t 8 -ngl 0 [--chunks n] [-ub u]`.

| setting | A: repack build (freeze), default | B: 0011, `VOLLIE_Y1B=1 --no-repack` | C: 0011, knob unset, default | D: 0011, knob unset, `--no-repack` (control) |
|---|---|---|---|---|
| `-b 512`, 4 chunks | 5.8104, 6.3386, 7.0099, 7.9916 | **identical** | identical | 5.7871, 6.3335, 6.9987, 7.9833 |
| `-b 512 -ub 32`, 4 chunks | 5.7851, 6.3350, 7.0107, 7.9935 | **identical** | identical | 5.7840, 6.3427, 7.0095, 7.9921 |
| `-b 512 -ub 1`, 2 chunks | 5.7791, 6.3466 | **identical** | identical | 5.7924, 6.3426 |

- A equals tom-et25 §2's repack stock values.
- So Y1b on mapped rows gives the freeze build's numbers. It is class B (bit-identical), not a declared numerics change.
- One subject only, as the brief allows.
- Host time per pass (B 7.5 / 8.7 / 16.6 s against A 2.8 / 3.2 / 10.1 s) comes from the per-position loop in prefill and
  from 8 host threads. It means nothing for the phone.
- Logs: `lab/c3/logs/udk/bt1-<A|B|C|D>-<b512|ub32|ub1>.log`.

## 4. Price at these costs (*estimated*; `lab/c3/ud_price.py` on `bonnie_ud_check.py`, unchanged)

**Model.** Bonnie's `bk` / `ms1`, with the base kernel at 1.45 (Y1b body), `down` bytes 0.5556, 0.5 ms per top-k selection
and O as r15.
- **Columns:** path A has no copy and the plain head, and only R1 is available. A′ has the head copy and `down` in K1's
  layout, so R1, R3 and UD are available. The ms differences do not depend on the head or O. Only tok/s does.
- **Self-check:** with Bonnie's inputs the script reproduces her path-A UD at C 50, −22.8 / −22.7 / −21.0.
- **Inputs:** the host k values are put in directly as her kernel inputs: kr = row kernel per touched weight, KC = K1
  per touched weight, both against the packed gemv.
- **Caveat on the cold numbers:** a cold host k already contains the M4's stalls, which her model also charges through
  bytes. The rows-save-no-bytes line brackets that.

ms per token against the Y1b base, Yi / Llama-2 / DeepSeek, C 50:

| inputs | path A: R1 | path A: R1, rows save no bytes | A′: R3 | **A′: UD** | A′ UD tok/s (base 6.77 / 5.96 / 6.04) |
|---|---|---|---|---|---|
| Bonnie (kr 1.45, KC 2.1) | −9.5 / −9.4 / −8.8 | −1.3 / −1.3 / −1.1 | −13.3 / −13.2 / −12.3 | **−22.8 / −22.7 / −21.0** | 8.08 / 6.94 / 6.97 |
| host hot (kr 1.49, KC 1.67) | −9.4 / −9.3 / −8.6 | −1.2 / −1.2 / −1.0 | −14.2 / −14.1 / −13.1 | **−23.5 / −23.4 / −21.7** | 8.13 / 6.98 / 7.01 |
| host cold (kr 2.59, KC 1.66) | −6.1 / −6.2 / −5.7 | **+2.0 / +2.0 / +1.9** | −14.2 / −14.1 / −13.1 | **−18.7 / −20.3 / −18.8** | 7.81 / 6.82 / 6.86 |

- **Path A tok/s with R1** (base 6.43 / 5.82 / 5.61): 6.87 / 6.17 / 5.91 hot, and 6.71 / 6.05 / 5.80 cold.
- **At C 40** (the more compute-bound clock state; Bonnie uses C 33 for the 1,555 cap): with host cold inputs, R1 is **+5.7 / +5.7 / +5.3 ms** (a loss)
  and A′ UD is −9.0 / −9.0 / −8.3. With host hot inputs, A′ UD is −24.2 / −24.1 / −22.3.
- **Break-even** (price 0, Y1b base):

  | | C 50 | C 40 |
  |---|---|---|
  | R1 row kernel kr | 3.12 / 3.28 / 3.27 (1.90 if the rows save no bytes) | 2.18 |
  | R3 K1 cost KC | 4.49 / 4.71 / 4.70 | 3.15 |

**Kill readings asked in the brief:**
- **R1 over 1.3 per touched weight cold → R1 about −4 ms:** host cold is 2.57 to 2.60, so the line **fires**. On Bonnie's
  model R1 is −6 ms if the skipped rows save bus time, +2 ms if they do not (host cold), and a loss at C 40. On path A,
  where R1 is the only UD part available, UD's speed side is worth 0 to about −6 ms (*estimated*). That is under
  +0.3 tok/s.
- **K1 on `down` cold over the value that makes R3 ≤ 0 → R3 dead:** the break-even is 4.5 to 4.7 (C 50) or 3.15 (C 40),
  against 1.64 to 1.71 measured cold at TR 128. **R3 is not dead.** It carries most of A′'s UD price (−13 to −14 ms of
  the −19 to −24).
- If the phone's K1 is the R72 value (2.1 to 2.24), R3 is still far from its break-even. TR 512 cold (3.7 to 3.9) would be
  dead at C 40.

## 5. Files (none committed)

- `patches/llama.cpp-c3/0011-y1b-decode.patch` (new; sha256 above). The series README needs its row (Annie or the lead).
- Tree: `lab/llama.cpp-c3y1b` (worktree, 0011 only) and its `build/`.
- Mask dump: `lab/c3/bench/ud_dump.cpp` (sha256 first 16 hex digits c4a9a1681c84e770), masks `lab/c3/bench/udmask-<m>.bin`.
- Bench: `lab/c3/bench/ud_bench.c` (42951c990d216ab6), includes `tools/c3_colkern_bench.c` unchanged.
- Price: `lab/c3/ud_price.py` (9be2ae2b143b9ba8; asserts Bonnie's −22.8 / −22.7 / −21.0).
- Logs: `lab/c3/logs/udk/`.
- Nothing in campaign 1 or in the served files was touched. `tools/` is unchanged.

## 6. Not done, limits

- No phone cell. All three decisive numbers are host cold costs: R1's row list, Y1b on `down`, and K1 at TR 128. The
  M4 showed in memo-05 that it can mislead in both directions. The device cells that settle them already exist in form:
  0002's `p2d` `y1b_rows` row and 0003's `l1` K1 cells. They would need the 11,008-input and keep-0.5 variants.
- B-T1 used one subject (Llama-2) and short runs (4 / 4 / 2 chunks), as the brief allows. No KleidiAI build was tested:
  Y1b matches the repack gemv, not KleidiAI's kernel.
- B-T2's four-thread arm was not run (see §2.2).
- UD's own rule cost (two top-k selections per layer) is priced at Bonnie's 0.5 ms each and was not measured.
