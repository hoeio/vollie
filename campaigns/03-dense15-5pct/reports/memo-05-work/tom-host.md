# c3 memo-05 work file, Tom: host results for the TEST lists of Bonnie and Ebony

Agent model: `claude-opus-5-5` (Tom). Date 2026-11-19. Repository HEAD when run: `c4273ee`. Machine: the owner's
laptop (Apple M4 Pro). **Every measured number here is *host* (relative only) unless it says otherwise. Nothing was
run on the phone.** Predictions are quoted from `bonnie-draft.md` (B) and `ebony-draft.md` (E). They are *estimated*
values and were written before these runs.

## 0. Setup and what changed

- **Kernel bench:** `tools/c3_colkern_bench.c` (extended). It is a bench-only tool with no runtime knob. The old
  mode (memo-02 H6/H6b) is unchanged and is still the default. New modes are `k1` (T1, T2, T4, E6) and `k1lines`
  (T7). With `-DGGML_LINK`, the bench also links the lab tree's compiled `libggml-cpu`. That build uses
  `armv8.2-a+dotprod+fp16`, which is the phone's path. The bench then checks identity against the real
  `ggml_gemv_q4_0_4x4_q8_0`, `ggml_gemm_q4_0_4x4_q8_0`, `ggml_quantize_mat_q8_0_4x4`, `quantize_row_q8_0` and
  `ggml_vec_dot_q4_0_q8_0`, not only against the bench's own copy.
  ```sh
  B=lab/llama.cpp-c3/build/bin
  clang -O3 -march=armv8.2-a+dotprod+fp16 -DGGML_LINK -DTR=1024 -o lab/c3/bench/k1lib1024 tools/c3_colkern_bench.c -L$B -lggml-cpu -lggml-base -Wl,-rpath,$PWD/$B
  # the same with -DTR=512 -> k1lib512 (TR only sets the memo-02 kernel "h6col"; K1's tile size is a runtime argument)
  lab/c3/bench/k1lib1024 14336 4096 30 k1; lab/c3/bench/k1lib1024 4096 14336 30 k1      # Mistral gate/up, down
  lab/c3/bench/k1lib512 18944 3584 30 k1;  lab/c3/bench/k1lib512 3584 18944 30 k1       # Qwen2.5 gate/up, down
  lab/c3/bench/k1lib512 <N> <K> 1 k1lines                                                # T7
  # logs: lab/c3/logs/t1-k1-<N>x<K>.log, lab/c3/logs/t7-lines.log
  ```
- **Regimes, all single thread.** Random weights and activations, as in memo-02.
  - **hot:** the first 1024 rows, repeated. That is 2.1 to 10.9 MB, which fits the M4 Pro's L2.
  - **warm:** the whole matrix, repeated. This is memo-02's method.
  - **cold:** 29 to 33 copies of the matrix, 1.1 GB in all, used round-robin.
  - **Timing:** best of 3 trials of 30 calls each (hot: 30 × N/1024 calls).
  - **k:** time per touched weight divided by the stock 4x4 gemv's time per weight, in the same regime.
- **Masks:** each input is kept independently with probability d. Realised d is in the logs; it is within ±0.015
  of the target.
- **PPL tool:** `tools/c3_colskip.py` (extended).
  - E2's gate rule is a new group, `gate`.
  - T5's statistics are written when `C3_KEEPSTAT=<npz>` is set.
  - `QS` gains the quantiles 0.7 and 0.8.
  - Old tables carry their own `qs` list, so they read as before.
- **Models:** the same two Q4_0 files as memo-02, in `lab/c3/models`. Nothing was downloaded. Disk free stayed at
  15 GB or more.
- **No runtime patch was written.** `patches/` is untouched, and so is every campaign 1 file.

## 1. T1: K1, strip-register axpy with bias-form nibbles

**Identity: bit-identical** (`memcmp` of all outputs) to ggml's NEON 4x4 gemv fed the same zeroed q8 inputs.
- **Where:** at d 1.0, 0.65 and 0.60, at TR 128 and TR 512, on all four shapes.
- **Against the linked library too:** the bench copy equals the library's `ggml_gemv_q4_0_4x4_q8_0` to the bit.
- **The scalar C reference is not the same path.** `ggml_gemv_q4_0_4x4_q8_0_generic` differs from the NEON kernel
  in 93 to 97 % of outputs (max rel 5e-4 to 1.6e-2), because it uses another float order. So "identical to ggml's
  path" means the NEON path the phone runs. K1 matches it.

**k per touched weight (*host*), d 0.60** (realised 0.595 to 0.605), hot / warm / cold:

| shape (rows x inputs) | K1 TR 128 | K1 TR 512 | memo-02 kernel (h6col, same session) |
|---|---|---|---|
| 14336 x 4096 (Mistral gate/up) | 1.62 / 1.64 / 1.57 | 1.57 / 1.63 / **2.22** | TR 1024: 2.58 / 2.66 / 3.65 |
| 4096 x 14336 (Mistral down) | 1.58 / 1.54 / 1.58 | 1.64 / 1.59 / **2.43** | TR 1024: 2.61 / 2.58 / 3.66 |
| 18944 x 3584 (Qwen2.5 gate/up) | 1.57 / 1.59 / 1.53 | 1.57 / 1.67 / **2.20** | TR 512: 2.64 / 2.72 / 3.08 |
| 3584 x 18944 (Qwen2.5 down) | 1.55 / 1.55 / 1.56 | 1.61 / 1.65 / **2.54** | TR 512: 2.61 / 2.58 / 2.67 |

d 0.65 and d 1.0, hot / warm / cold:

| shape | K1 TR 128, d 0.65 | K1 TR 512, d 0.65 | K1 TR 128, d 1.0 | K1 TR 512, d 1.0 |
|---|---|---|---|---|
| 14336 x 4096 | 1.58 / 1.53 / 1.54 | 1.58 / 1.61 / 1.87 | 1.42 / 1.43 / 1.39 | 1.42 / 1.66 / 2.56 |
| 4096 x 14336 | 1.59 / 1.54 / 1.55 | 1.62 / 1.59 / 2.10 | 1.44 / 1.40 / 1.40 | 1.40 / 1.59 / 2.55 |
| 18944 x 3584 | 1.54 / 1.54 / 1.52 | 1.51 / 1.66 / 2.17 | 1.40 / 1.39 / 1.36 | 1.40 / 1.64 / 3.40 |
| 3584 x 18944 | 1.53 / 1.53 / 1.53 | 1.58 / 1.62 / 2.24 | 1.40 / 1.37 / 1.39 | 1.39 / 1.59 / 2.49 |

**Time against dense stock (*host*), TR 128, cold, d 0.60:** 0.936 / 0.944 / 0.925 / 0.941. So at 60 % kept, K1 is
only 6 to 8 % faster than the dense stock kernel on this host. The host reads about 52 GB/s on one core (memo-02),
so it is compute-bound. The phone's byte-bound regime is where K1 can gain (B §2.2 prices it there).

**Against the predictions:**

| | prediction | measured (*host*) | verdict |
|---|---|---|---|
| B T1, host k at d 0.60 | **1.6 (1.35 to 1.95)** | TR 128: **1.53 to 1.64** in every regime and shape; TR 512: 1.57 to 1.67 hot/warm, **2.20 to 2.54 cold** | TR 128 inside the band at the centre; TR 512 cold outside it (high) |
| Kill: drop K1 if k ≥ 2.3 at d 0.60 | – | TR 128: max 1.64, **not killed**. TR 512 cold: 2.43 and 2.54 on the two `down` shapes, 2.20 and 2.22 on gate/up | **K1 survives at TR 128.** At TR 512 it is killed cold on 2 of 4 shapes on this host |
| B, A78 k | 1.8 (1.55 to 2.2) | not measurable here | device (D2) |
| E, A78 k (SMLAL on V0 only) | **2.1 (1.5 to 2.6)** | not measurable here; instruction mix below | device (D2) |

**Why TR 512 is worse cold (*host* reading, not proven):**
- Within one (tile, block), K1 walks the four 128-row strips of each kept column at a 256-byte stride. So the
  first strip pulls one line from each of about 19 columns at 256 B spacing.
- At TR 128 a column is one line, and the kept columns form a dense forward stream with 1-line holes.
- The M4's prefetcher handles the second pattern well and the first one poorly. The phone's prefetchers are
  different, so the TR choice still waits for D2 and R69's `p2col` cold row.

**Software prefetch variant (not in B's spec, tried and dropped).** `pf` prefetches the next block's scales and
kept columns. It was **worse** in every cell: TR 128 cold d 0.60 k 2.09 to 2.19 against 1.53 to 1.58 without it.
It stays in the bench as `K1-tr*-pf`; it is not proposed.

**Instruction mix of K1 per touched weight**, for E's device estimate. Counted from the `clang -O3` assembly of
`gemv_k1`. The source is the same for an Android build; the A78's compiler output may differ in scheduling.

| part | per kept column and 128-row strip (128 weights) | per touched weight |
|---|---|---|
| `SMLAL`/`SMLAL2` (8h) | 16 | **0.125** |
| `SDOT` | 0 | **0** |
| `AND` + `USHR` (16b) | 4 + 4 | 0.0625 |
| `LD1R` (splat of x_j) | 1 | 0.0078 |
| vector loads (`LDP q`) | 2 (64 B) | 0.0156 |
| scalar (index load, `MADD`, 2 `ADD`, loop) | about 6 | 0.047 |
| **float step, per strip and block (128 rows x 32 inputs)**: 16 `SUB`, 32 `SSHLL`, 32 `SCVTF`, 32 `FCVTL`, 32 `FMUL`, 32 `FMLA` = 176 vector ALU; plus 16 `LDR q` (scales), 32 `LDR q` + 32 `STR q` (float accumulators in L1) | – | 176 ÷ (4096 d) = **0.043 ÷ d** ALU (0.072 at d 0.60, 0.066 at d 0.65); loads+stores 80 ÷ (4096 d) |
| per block: Σ x_j over kept inputs (for the bias term) | scalar, nk loads and adds per (tile, block) | 1 ÷ TR scalar ops per weight (0.008 at TR 128) |

- **Totals.** Vector ALU per touched weight: 0.1875 + 0.043 ÷ d, which is **0.259 at d 0.60** and 0.254 at d 0.65.
  This matches B's count. Of that, 0.125 is `SMLAL`, and the rest is 0.134 at d 0.60. `LD1R` and the loads come on
  top.
- **Stock 4x4 gemv for comparison** (library assembly): per 4 rows x 32 inputs (128 weights) it uses 8 `SDOT`
  (0.0625 per weight), 4 `SSHL` + 4 `AND`, and the float step. That is **0.164 vector ALU per weight** (B's 21 per
  128), with 0 `SMLAL`.
- **Floor on two vector pipes with `SMLAL` on V0 only** (*estimated*). Against the stock floor of 0.082 cycles per
  weight:
  - Perfect pairing, with every non-`SMLAL` µop on V1: max(0.125, 0.259 ÷ 2) = **0.130 cycles per weight, 1.58 ×**.
  - E's reading, with the float step also on the bottleneck: 0.125 + 0.036 = **0.161, 1.96 ×**.
  - The host cannot separate the two. The host ratio k = 1.55 to 1.64 against a host V-count ratio of 1.58 says only
    that on the M4 the kernel runs at its µop count.

## 2. T2: Y1b, in-place 4-row gemv on the file's Q4_0 rows

The rows are the file's 18-byte blocks, row-major. Per row the kernel does `EOR 0x88`, `SHL 4` / `AND 0xF0`, and two
`SDOT` against x[0..15] / x[16..31]. Then 3 `ADDP` combine the four rows, and the float step is stock's.

**Identity: bit-identical** to the library's 4x4 gemv on all four shapes (and per row against the gemm, T4).

**k (*host*), hot / warm / cold:**

| shape | Y1b | Y1c kernel (`ggml_vec_dot_q4_0_q8_0` per row, the `--no-repack` path) | library 4x4 gemv (check) |
|---|---|---|---|
| 14336 x 4096 | 1.49 / 1.51 / 1.53 | 2.01 / 2.03 / 2.01 | 1.01 / 1.03 / 1.00 |
| 4096 x 14336 | 1.51 / 1.62 / **2.21** | 2.06 / 1.98 / 2.02 | 1.01 / 1.00 / 1.01 |
| 18944 x 3584 | 1.47 / 1.48 / 1.45 | 2.02 / 2.02 / 1.99 | 0.98 / 1.00 / 0.99 |
| 3584 x 18944 | 1.47 / 1.65 / **2.07** | 1.99 / 1.97 / 1.98 | 0.99 / 1.00 / 1.00 |

**Instruction mix per 128 weights** (assembly): 8 `SDOT`, 4 `EOR`, 4 `SHL`, 4 `AND`, 3 `ADDP`, 2 `FCVTL`, 1 `SCVTF`,
1 `FMUL`, 1 `FMLA`, and 3 `LD1` lane + 1 `LD1R` for the scales. That is B's 31. On top of that the compiler adds
**4 `MOVI` zeroing the `SDOT` accumulators**. They are avoidable: start each row's `SDOT` chain from the previous
block's partials and reduce once per row. That change was not tried.

| | prediction | measured (*host*) | verdict |
|---|---|---|---|
| B T2, hot | **1.4 (1.2 to 1.7)** | **1.47 to 1.51** | inside the band, above the centre |
| Kill: k > 1.7 hot → Y1b loses to Y1a | – | max 1.51 hot | **not killed** |
| Identity fails → class N | – | identical | class B |
| (not predicted) cold | – | 1.45 to 1.53 on short rows (K 3584/4096); **2.07 to 2.21 on long rows** (K 14336/18944, `down`) | B §5 warned that four unaligned streams may cost more cold; on this host they do, on the `down` shapes only |

## 3. T4: does the 4x4 gemm equal the gemv per element?

**Yes, bit for bit, on all four shapes** (*host*, the lab library). Setup: 8 activation rows, with outliers added
(×20 on every 97th element).
- `ggml_quantize_mat_q8_0_4x4` (the gemm's quantiser) gives the same scales and q8 values per row as
  `quantize_row_q8_0` (the gemv's quantiser).
- `ggml_gemm_q4_0_4x4_q8_0` over 8 rows equals `ggml_gemv_q4_0_4x4_q8_0` per row. 0 differences in 114,688 /
  32,768 / 151,552 / 28,672 outputs.
- K1 (TR 512, all inputs kept) and Y1b per row equal the gemv on the same rows. So they also equal the gemm.

**Consequence:** a K1 or Y1b build can run prefill as a gemv per token, or keep the stock gemm for prefill, without
a declared change. This holds for the Q4_0 4x4 path, the phone's non-i8mm path. It does not say that `-ub 32` and
`-b 512` give equal per-chunk values: they do not, at stock either (§4). Other operations depend on the batch.

## 4. T3: Y1c, `--no-repack`

Logs: `lab/c3/logs/t3-<model>-<b512|ub32|ub1>-<stock|nr>.log` (4 chunks) and `t3-<model>-b512-40-<stock|nr>.log`.
All runs used `llama-perplexity -c 512 -t 8` with the lab build, with `-nr` for no repack. memo-02 ran its identity
checks on 4 chunks (2 at `-ub 1`). Here every setting has 4 chunks, and the declared-change delta has 40 chunks at
`-b 512`.

| model, setting | stock [1] [2] [3] [4] | `--no-repack` [1] [2] [3] [4] |
|---|---|---|
| Mistral `-b 512` | 5.2873 5.4066 6.3280 7.0083 | 5.2846 5.4051 6.3240 7.0106 |
| Mistral `-ub 32` | 5.3060 5.4174 6.3351 7.0187 | 5.2967 5.4185 6.3362 7.0179 |
| Mistral `-ub 1` | 5.2980 5.4042 6.3238 7.0042 | 5.3030 5.4097 6.3291 7.0110 |
| Qwen2.5 `-b 512` | 5.6829 7.8820 6.9811 6.6410 | 5.7097 7.9176 7.0063 6.6714 |
| Qwen2.5 `-ub 32` | 5.7143 7.9410 7.0256 6.6752 | 5.7073 7.9349 7.0093 6.6733 |
| Qwen2.5 `-ub 1` | 5.7086 7.9282 7.0254 6.6805 | 5.7101 7.9333 7.0382 6.6888 |
| **40 chunks, `-b 512`, final** | Mistral **5.8844**, Qwen2.5 **6.9575** (= memo-02) | Mistral **5.8836 (−0.014 %)**, Qwen2.5 **6.9587 (+0.017 %)** |

- **Y1c is not bit-identical; it is a declared numerics change.** Every chunk differs at every setting on both
  models, from the 3rd decimal of the running PPL. On the 40 chunks the per-chunk NLL moves by −0.0046 to +0.0055
  (Mistral) and −0.0067 to +0.0079 (Qwen2.5). The 40-chunk totals move by −0.014 % and +0.017 %. Either way that
  is far inside the +5 % budget, and of the size of the batch-setting differences that stock itself shows.
- **Kernel cost:** the `vec_dot` row kernel costs **k = 1.97 to 2.06** in every regime and shape (§2 table,
  *host*, single thread).
- **Whole-model ns per weight (llama-bench):** see §4.1.

| | prediction | measured (*host*) | verdict |
|---|---|---|---|
| B T3: per-chunk values differ in the 4th decimal or later | 4th decimal | **3rd decimal** of the running PPL (for example Qwen2.5 `-b 512` chunk 1: 5.6829 → 5.7097); 40-chunk total ±0.02 % | missed (the differences are larger than predicted, the total is smaller than any budget concern) |
| B T3: k 1.6 to 2.2 on the host | 1.6 to 2.2 | kernel 1.97 to 2.06 | inside |
| B: identical → class B, else declared with a 40-chunk delta | – | not identical; delta −0.014 % / +0.017 % | **declared change**, cost about 0 |
| B Y1c A78 k | 1.9 (1.5 to 2.4) | – | device |

### 4.1 llama-bench, `--repack 1` against `--repack 0` (*host*)

`llama-bench -p 0 -n 64 -t 1,4,8 --repack 1,0 -r 3` gives decode (tg64) in tok/s. ns per weight = 1e9 ÷ (tok/s ×
weights read per token). The weights per token are the parameters minus the lookup embedding: Mistral 7.11 B,
Qwen2.5 7.07 B. Logs: `lab/c3/logs/t3-bench-<model>.log`.

| model, threads | repack on: tok/s, ns/w | `--no-repack`: tok/s, ns/w | ratio (Y1c ÷ stock) |
|---|---|---|---|
| Mistral, 1 | 12.09, 0.0116 | 6.27, 0.0224 | **1.93** |
| Mistral, 4 | 39.03, 0.0036 | 21.83, 0.0064 | 1.79 |
| Mistral, 8 | 44.45, 0.0032 | 29.61, 0.0047 | 1.50 |
| Qwen2.5, 1 | 11.62, 0.0122 | 6.15, 0.0230 | **1.89** |
| Qwen2.5, 4 | 40.42, 0.0035 | 22.25, 0.0064 | 1.82 |
| Qwen2.5, 8 | 46.69, 0.0030 | 33.49, 0.0042 | 1.39 |

On one thread the whole model costs 1.9 × with repack off, close to the kernel's 2.0. That covers the Q6_K head too,
which also loses its repack. At 8 threads the host turns byte-bound and the ratio falls to 1.4 to 1.5. This is the
mechanism B §1.2 relies on for the phone (R = 1.68), seen here in the host's own regime. The phone's numbers are D1's.

## 5. T7: lines touched by K1's layouts (exact count, random masks)

A (tile, block) region is 18 × TR bytes: TR/32 lines of scales, then 32 columns of TR/128 lines each. A block with at
least one kept input touches its scales. So the touched share is exactly **(4 + 32 d) ÷ 36 = 1 − 8/9 × (1 − d)** for
every TR, which is B's b. TR changes only the size of the holes. A dropped column is TR/128 lines, and runs of
dropped columns join into longer holes.

Shares of the **skipped** lines by hole length, at d 0.60 on 14336 x 4096 (realised d 0.612, touched 0.6554 = b):

| TR | 1 line | 2 to 3 | 4 to 7 | 8 to 15 | 16 or more |
|---|---|---|---|---|---|
| 128 | 0.387 | 0.481 | 0.132 | 0 | 0 |
| 512 | 0 | 0 | 0.387 | 0.481 | 0.132 |
| 1024 | 0 | 0 | 0 | 0.387 | 0.613 |

- The other shapes and d 0.65 give the same picture (`lab/c3/logs/t7-lines.log`). At TR 128, 37 to 43 % of the
  skipped lines sit in 1-line holes and 46 to 48 % in 2 to 3-line holes. TR 1024 does not divide Qwen2.5's 3,584
  and 18,944 rows.
- Reading against b_eff: at TR 128, 83 to 90 % of the skipped lines are in holes of 1 to 3 lines. A stream
  prefetcher that runs ahead by 2 or more lines would fetch most of them. At TR 512, every hole is 4 lines or
  more. The host's cold timing (§1) shows the other side of the trade: TR 128 had the better compute pattern on
  the M4. The phone decides both (D2, `p2col`).

## 6. E6: Y1b on four arbitrary row pointers (Ebony)

The same kernel, with the four rows taken from a sorted random kept-row list. **Identity: bit-identical** to the 4x4
gemv on the same rows, at kept fractions 1.0, 0.5 and 0.3 on all four shapes.

k per touched weight (*host*), hot / warm / cold:

| shape (row length) | kept 1.0 | 0.5 | 0.3 |
|---|---|---|---|
| 14336 x 4096 (2.3 KB rows) | 1.49 / 1.53 / 1.53 | 1.49 / 1.66 / **3.36** | 1.50 / 1.57 / **3.28** |
| 18944 x 3584 (2.0 KB) | 1.46 / 1.51 / 1.47 | 1.47 / 1.55 / **3.22** | 1.47 / 1.56 / **3.20** |
| 4096 x 14336 (8.1 KB) | 1.52 / 1.62 / 2.18 | 1.49 / 1.50 / 2.09 | 1.48 / 1.48 / 2.11 |
| 3584 x 18944 (10.7 KB) | 1.46 / 1.68 / 2.05 | 1.46 / 1.46 / 1.96 | 1.46 / 1.43 / 1.92 |

| | prediction / rule | measured (*host*) | verdict |
|---|---|---|---|
| E6 kill: k over 1.7 | – | hot 1.46 to 1.52, warm 1.43 to 1.68: pass. **Cold: 1.92 to 3.36, fails on every shape** | passes in cache; fails cold on this host |
| E: Y1b's 1.5 holds (§5 item 4) | 1.5 | 1.46 to 1.52 hot | holds |

**The cold row-skip case is the one Ebony's §3 needs, and on this host it saves almost no time.**
- On 2 KB rows (the `gate`/`up` shapes, which are the rows her §3 skips), half the rows take **1.05 ms against 0.96
  ms for all of them**. At 0.3 kept they take 0.61 ms, that is 64 % of the dense time for 30 % of the bytes.
- Random 2 KB rows break the stream, and each row starts with a demand miss.
- Long rows (8 to 11 KB) keep k near 2. That is the same as dense Y1b cold on those shapes.
- This is *host* only. It is the host analogue of b_eff for rows, and it argues for a prefetch-ahead of the next
  kept rows if §3 is ever built. D2 or E5 would measure it on the phone.

## 7. E1: `down`-input-only sparsity at 60 / 70 / 80 %

Method: memo-02 §3, the same tool and the same 40 test chunks. The table is new, `lab/c3/thr5-<m>-q4_0.npz`
(wikitext-2 train, 16 chunks). It adds quantiles 0.7 and 0.8. Its random sample stream differs from thr2's,
because the gate samples are drawn as well, so its 0.6 thresholds are not exactly thr2's. Logs:
`lab/c3/logs/e1e2-<m>-b512-40.log` and `e1e2-<m>-ub1-40.log`. Stock, 40 chunks: Mistral 5.8844 (`-ub 1` 5.8810),
Qwen2.5 6.9575 (`-ub 1` 6.9607). Pass line +4.5 %.

| cell (`down` input only) | Mistral `-b 512` (realised) | Mistral `-ub 1` | Qwen2.5 `-b 512` (realised) | Qwen2.5 `-ub 1` |
|---|---|---|---|---|
| 60 % | 6.0850, **+3.41 %** (0.600) | 6.0779, **+3.35 %** | 7.0911, **+1.92 %** (0.588) | 7.1222, **+2.32 %** |
| 70 % | 6.3334, +7.63 % (0.695) | not run | 7.3419, +5.53 % (0.682) | not run |
| 80 % | 6.9588, +18.26 % (0.790) | not run | 8.0597, +15.84 % (0.775) | not run |

`-ub 1` was run only for the cell that passes at `-b 512`. The 70 and 80 % cells already fail at `-b 512`. In
memo-02, `-ub 1` stayed within 0.5 points of `-b 512` on every cell run at both settings.

| | rule / prediction (E) | measured (*host*) | verdict |
|---|---|---|---|
| E1 closes if: under 60 % inside +4.5 % on either model | – | 60 % is inside on both at both settings (+3.41 / +1.92 % at `-b 512`, +3.35 / +2.32 % at `-ub 1`) | **not closed** |
| E1 opens if: 70 % on both | – | 70 % fails on both (+7.63 / +5.53 %) | **not opened** |
| E §3 price assumed s = 0.7 inside the budget (variant B, 7.5 tok/s; with `down` columns 9.3) | s 0.7 | frontier about 60 to 65 % on Mistral, about 65 % on Qwen2.5 (between the cells) | the s = 0.7 prices are above the frontier; reprice at s 0.6 |

- **Generalised:** `down` alone holds 60 % on both models. The cost curve is steep on both between 60 and 80 %.
- **Did not generalise:** the level. Qwen2.5 is the more tolerant model here, by about 1.5 to 2 points at each
  cell. That is the reverse of the all-groups ranking in memo-02 §3.
- This is the exact-product ranking, an upper bound for any proxy (E3). E3 was not run; it is gated on E1 opening.

## 8. E2: gate rule, |SiLU(g)| threshold

Rule: zero neuron i of layer L (the `ffn_down` input, which means its `up` row and its `down` column) when
|SiLU(g_i)| of the full `gate` output is under a per-layer train threshold. The thresholds are from thr5, gate
samples. 40 chunks, `-b 512`.

| cell | Mistral (realised) | Qwen2.5 (realised) |
|---|---|---|
| 30 % | 6.2577, **+6.34 %** (0.308) | 7.2173, **+3.73 %** (0.305) |
| 50 % | 7.8989, +34.2 % (0.507) | 8.3758, +20.4 % (0.498) |
| 70 % | 33.085, × 5.62 (0.694) | 13.2826, +90.9 % (0.687) |

| | rule | measured (*host*) | verdict |
|---|---|---|---|
| E2 closes if no cell under 40 % fits on both models | – | Mistral fails at 30 % (+6.34 %); Qwen2.5 fits at 30 % (+3.73 %) | **closed** (no cell fits on both); **did not generalise** |

- The gate alone is a much worse ranking than the product. At the same 50 %, `down`-input zeroing by |SiLU(g)·u|
  cost +1.21 / +0.70 % in memo-02 (16 chunks), against +34 / +20 % here. This agrees with memo-01's CATS reading.
- `-ub 1` was not run: the rule is closed at `-b 512`.
- What Qwen2.5's 30 % would buy: 0.3 × `up` rows, about 8 % of the token's bytes (*estimated*). That alone is not
  a lever.

## 9. T5: H5, per-input keep frequency and the 32-token union

Thresholds: each model's frontier cell from memo-02 §3 (thr2 table, element rule, all four groups). Mistral 40 %
(PPL 6.0277, +2.44 %) and Qwen2.5 35 % (7.1732, +3.10 %), 40 chunks, `-b 512`. Both equal memo-02's values.
Per input and column, the tool counts the share of tokens on which the column is kept. h = the share of columns kept
on at least 98 % of tokens. The 32-token union = the share of columns kept by at least one token of a window of 32
consecutive tokens, averaged over windows. Files: `lab/c3/t5-keep-<m>.npz`, logs `lab/c3/logs/t5-<m>-b512-40.log`.

| group (byte share of the body) | Mistral: mean kept, h (max over layers), 32-token union | Qwen2.5 |
|---|---|---|
| `qkv` (0.115 / 0.071) | 0.608, **0.003** (0.052), 0.988 | 0.662, **0.001** (0.004), 0.991 |
| `o` (0.077 / 0.055) | 0.589, **0.006** (0.114), 0.981 | 0.652, **0.002** (0.012), 0.993 |
| `gate`/`up` input (0.538 / 0.583) | 0.607, **0.000** (0.001), 1.000 | 0.663, **0.001** (0.005), 0.999 |
| `down` input (0.269 / 0.291) | 0.587, **0.004** (0.026), 1.000 | 0.650, **0.013** (0.313, one layer), 0.977 |
| **body-weighted** | 0.600, **h 0.002**, union **0.997** | 0.658, **h 0.005**, union **0.992** |

Histogram of keep frequency per column, as the share of columns in bins 0–0.2 / 0.2–0.4 / 0.4–0.6 / 0.6–0.8 /
0.8–0.9 / 0.9–0.98 / 0.98–1:

| | Mistral | Qwen2.5 |
|---|---|---|
| `qkv` | .015 .011 .513 .380 .055 .022 .003 | .010 .006 .190 .710 .071 .012 .001 |
| `o` | .031 .128 .332 .373 .096 .033 .006 | .009 .060 .273 .475 .146 .035 .002 |
| `gate`/`up` input | .001 .001 .529 .457 .010 .002 .000 | .001 .001 .058 .919 .016 .003 .001 |
| `down` input | .000 .007 .628 .338 .013 .011 .004 | .027 .006 .219 .647 .058 .030 .013 |

| | prediction (B §2.3) | measured (*host*) | verdict |
|---|---|---|---|
| h, residual-stream inputs (`qkv`, `gate`/`up`) | 0.02 to 0.10 | 0.000 to 0.003 | **missed low** |
| h, `down` input | 0.10 to 0.35 | 0.004 / 0.013 | **missed low** |
| h, body-weighted | 0.05 to 0.15 | **0.002 / 0.005** | **missed low** |
| K2 gate: build if h ≥ 0.2 on both | – | h ≤ 0.005 on both | **K2 closed on both models** |

- **The 32-token union is 0.98 to 1.00 of all columns on every group of both models.** So the threshold rule spreads
  its kept set evenly: almost every column is kept on 40 to 70 % of tokens, and almost none on every token.
- Two consequences, both *host*:
  - There is no hot core for K2 to serve with `sdot`.
  - Column sparsity does not shrink the resident set over a 32-token window. Every column of every matrix is
    still read within 32 tokens. Residency gains from sparsity are closed on both models.
- **Generalised:** all of the above, on both models.
- **Did not generalise:** only the shape of the histogram. Qwen2.5's `gate`/`up` input is narrowly centred at 0.6 to
  0.8 (92 %); Mistral's is split across 0.4 to 0.8.

## 10. T6: Y1a dry run

**Not run** (skipped, as the task allowed). It needs a loader path that writes and maps the repacked buffers, that
is, real runtime code in a new patch series. That is more than a host check. The repack is a pure function of the
file bytes (memo-05 B §1.1: a byte permutation plus a constant XOR), so determinism is not in doubt. The check that
matters is per-chunk identity with the side file mapped, and it comes with the patch.

## 11. Summary per model, and what generalised

All *host*.

| test | Mistral-7B-v0.3 | Qwen2.5-7B | generalised? |
|---|---|---|---|
| T1 K1 identity | bit-identical to ggml's NEON 4x4 gemv | same | yes (model-generic kernel) |
| T1 K1 k at d 0.60, TR 128 | 1.54 to 1.64 | 1.53 to 1.59 | yes: hot, warm and cold alike, inside B's host band; kill line 2.3 not reached |
| T1 K1 k at d 0.60, TR 512, cold | 2.22 / 2.43 | 2.20 / 2.54 | yes: worse cold on both (host prefetch pattern); killed on the `down` shapes |
| T2 Y1b identity, k hot | identical, 1.49 / 1.51 | identical, 1.47 / 1.47 | yes; cold 2.1 to 2.2 on the long-row `down` shapes of both |
| T4 gemm = gemv | yes, bit for bit | yes | yes (the 4x4 path is shape-generic) |
| T3 Y1c | declared change; 40-chunk −0.014 %; whole-model 1.93 × at 1 thread | declared; +0.017 %; 1.89 × | yes: tiny PPL effect, about 2 × compute on both |
| T7 lines | touched = (4 + 32 d)/36 exactly; at TR 128, 83 to 90 % of skipped lines in 1 to 3-line holes | same | yes (arithmetic) |
| E6 Y1b on row lists | identity; k 1.5 hot; cold 3.3 on 2 KB rows | identity; 1.5 hot; cold 3.2 on 2 KB rows | yes: in cache it passes, cold row skips on short rows save little on the host |
| E1 `down` only | 60 % +3.41 % (`-ub 1` +3.35 %); 70 % fails | 60 % +1.92 % (`-ub 1` +2.32 %); 70 % fails (+5.53 %) | direction yes; Qwen2.5 more tolerant here |
| E2 gate rule | 30 % fails (+6.34 %) | 30 % fits (+3.73 %), 50 % fails | **no**; closed by the both-models rule |
| T5 hot core h | 0.002 | 0.005 | yes: no hot core; K2 closed; 32-token union 0.99 to 1.00 |
| T6 Y1a | not run | not run | – |

**Decisions these numbers feed (the lead's):**
- K1 passes T1 at TR 128 on the host.
  - Its device value is the open question: B 1.8, E 2.1. The `SMLAL` floor is 1.58 to 1.96 × by count (§1).
  - TR 512 should not be the first device candidate, unless the phone's cold row says otherwise.
- Y1b passes T2 and T4. Y1c is a declared change with a negligible PPL delta, so it needs one 40-chunk-style check
  per model in the device pre-registration. D1 can use it as is.
- K2 is closed (T5).
- The gate rule (E2) is closed.
- E1 is open in the middle at 60 to 65 %. With E6's cold result, Ebony's row-skip variant needs a reprice at
  s 0.6, and a prefetch-ahead before any device claim.
- K1-pf is dropped.
