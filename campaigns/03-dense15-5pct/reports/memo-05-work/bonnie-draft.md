# c3 memo-05 work file, Bonnie: prices for Y1 (kernel layout in a mapped file) and a cheaper exact column kernel

Agent model: `claude-fable-5-2` (Bonnie). Date 2026-11-19. Repository HEAD when written: `458102c`. Draft for the
joint round with Ebony; verdicts are the lead's. **R69 has not been run. No phone number in this file is new.**

Labels: *device* = a stored phone measurement, named by R-number; *host* = laptop (memo-02) or a read of the lab
tree `lab/llama.cpp-c3` (base `620d2225c` + c2 series); *published*; *recalled* = from memory, not confirmed today;
*estimated* = my arithmetic. **Every price and band below is *estimated*.** Bands are my 80 % intervals.

Inputs used throughout (R69 pre-registration centres, *estimated*): r = 24 GB/s, c_b = 40.4 GB/s, R = 1.683.
Bodies (*host*, memo-02 H1): Mistral 4.037 GB per token (blocks 3.926, Q6_K head 0.110), Qwen2.5 4.119 GB
(blocks 3.671, head 0.447). Kept fraction d: Mistral 0.60, Qwen2.5 0.65; b = 1 − 8/9 × (1 − d) = 0.644 / 0.689.

## 0. Summary

1. **Y1 does not add capacity; it moves the body to the ledger R69 measures.** At stock the phone build copies
   the whole body into anonymous memory (4.036 / 4.118 GB, *host*); the app was killed at 3.93 GB of such a copy
   (R9, *device*). With Y1 the usable body is C − 0.3 GB (band 0.1 to 0.6). Both lead bodies fit only if the 4.6
   step is held, and then with 0.26 / 0.18 GB to spare at the centre.
2. **There are three ways to have no anonymous copy, and the cheapest is not a side file.** Y1c: repack off
   (`use_extra_bufts = false`, zero code, declared numerics change, about 1.9 × `sdot` per weight). Y1b: a 4-row
   kernel that reads the GGUF rows in place (about 1.5 ×, no file, no owner question, bit-identical in decode).
   Y1a: the repacked layout written once to a side file (1.0 ×, +4.0 / +4.1 GB of storage, owner question).
   At r = 24 the phone is byte-bound by 1.68 ×, so Y1b's 1.5 × costs 0 to 15 ms per token; from r ≈ 27 it costs
   more and Y1a is the one to build.
3. **The side file is the same category as a calibration table and is already covered by COMMON_RULES rule 1**
   (R42 amendment 2), if the phone builds it. It is argued in §1.3 and kept as an owner question for after the
   R68 JSON. **Every column layout (item 2) needs that answer**: a column-major copy cannot be anonymous.
4. **Column kernel K1 (strip-register axpy, bias-form nibbles): 1.8 × `sdot` per touched weight (1.55 to 2.2)**
   against the host kernel's 2.6. That is under k\* (2.81 / 2.59), under the 1.15-gain line (2.44 / 2.25) across
   the band, and at the full-gain line (1.81 / 1.78) at the centre. Token: Mistral 176 → 118 ms, Qwen2.5 180 → 133
   ms at r = 24 (O = 8 ms), 5.7 → 8.5 and 5.6 → 7.5 tok/s. **K2 (hot-core split)** lowers k by 0.1 to 0.4 more,
   but only if H5 finds always-kept inputs; my prior is the low end.
5. Nothing here reaches 15 tok/s. Best case of this memo on a 4.0 GB body is about 8.5 tok/s.

## 1. Y1: kernel layout in a mapped file

### 1.1 What the base does (*host*, read of the lab tree)

- `src/llama-model.cpp:1123` (`make_cpu_buft_list`) puts the CPU's extra buffer type `CPU_REPACK` ahead of the
  plain CPU type when `use_extra_bufts` is true (default, `:3347`; CLI `-nr/--no-repack` turns it off).
- Only the default buffer type is built on the mapping (`:1916`, `buffer_from_host_ptr`). A tensor that
  `CPU_REPACK` accepts gets a buffer from `ggml_backend_cpu_repack_buffer_type_alloc_buffer`
  (`ggml/src/ggml-cpu/repack.cpp:5168`), which is `ggml_backend_buft_alloc_buffer(cpu)`: an aligned malloc,
  that is, **anonymous memory**.
- `load_all_data` (`src/llama-model-loader.cpp:1736`) then calls `ggml_backend_tensor_set` with the pointer into
  the mapping; `ggml_backend_cpu_repack_buffer_set_tensor` (`repack.cpp:5150`) runs
  `repack_q4_0_to_q4_0_4_bl(t, 4, …)` (`:3541`): for each group of 4 rows and each 32-weight block,
  `make_block_q4_0x4` (`:3083`) writes the 4 fp16 scales, then the 4 rows' nibble bytes interleaved in 4-byte
  pieces, each XORed with `0x88` (bias form to sign form).
- So the repack is **a permutation of bytes plus a constant XOR**: 4 × 18 B in, 72 B out, the same 0.5625 B per
  weight, exactly invertible. Q6_K heads go through `q6_K_8x4` the same way.
- After the load the mapping is trimmed only before the first and after the last tensor that stayed on it
  (`:1846`); the source pages of repacked tensors stay mapped, unreferenced, and are the first to be reclaimed.

### 1.2 Three builds with no anonymous copy

V = vector µops per weight on the A78's two ASIMD pipes (*recalled*: A78 has V0/V1, X1 four; not confirmed
today, the Arm guide's tables did not load). Stock 4x4 gemv per 4 rows × 32 inputs: 4 `shl`, 4 `and`, 8 `sdot`,
`scvtf`, 2 `fcvt`, `fmul`, `fmla` = 21 V per 128 weights = **0.164 V/w**; floor 0.082 cycles per weight, measured
0.108 (R41, *device*), so the V count explains 76 % and is the unit I price in.

| | Y1c repack off | Y1b in-place 4-row kernel | Y1a side file (layout cache) |
|---|---|---|---|
| what | `use_extra_bufts = false`; weights stay on the model mapping; plain `ggml_vec_dot_q4_0_q8_0` | new gemv that loads 4 GGUF rows' blocks, XORs `0x88`, then the stock math | per-tensor repack into a scratch buffer, `write_raw` to `<model>.<layout>` in the app's internal files, then mapped with `llama_mmap` |
| V per 128 w | about 40 (per row: `and`, `ushr`, 2 `sub`, 2 `sdot`, convert, non-fused multiply and add) | 4 `eor` + 4 `shl` + 4 `and` + 8 `sdot` + 3 `addp` + 3 `ins` (scales) + 5 = 31 | 21 (stock) |
| k per weight | **1.9 (1.5 to 2.4)** | **1.5 (1.35 to 1.8)** | **1.00**; 0.93 if block-major (*host*, memo-02 §8.1) |
| numerics | declared change (lane-wise float sums; another order than 4x4) | bit-identical to 4x4 gemv (same integer sums, same fma order); prefill needs a 4-row gemm or runs gemv per token | bit-identical |
| storage | 0 | 0 | **+4.036 / +4.118 GB**; 8.1 / 8.5 GB per model with its GGUF; 9.4 GB free in R67 (*device*) with both c2 models in, so one model at a time unless the c2 models go |
| anonymous peak | 0 | 0 | largest tensor once: 0.447 GB (Qwen2.5 head) unless written in row groups; 38 MB otherwise |
| first-load time | as stock minus repack | as stock minus repack | read 4.0 GB (2.8 to 11 s at 0.7 to 2.7 ms/MB, *device* fits) + write 4.0 GB (4 to 8 s at 0.5 to 1 GB/s) = **7 to 19 s once**; later loads map only |
| token cost at r = 24 (max model: byte-bound while k ≤ R = 1.68) | 1.9 > 1.68: +13 %, about **+22 ms** | 0 in the max model; **0 to +15 ms** on the rounder real curve | **0 ms** |
| token cost at r = 30 (R = 1.35) | +41 %, +55 ms | +11 %, +15 ms | 0 |
| code | none | about 150 lines, one kernel + dispatch | about 150 to 250 lines: loader path, header check, one ggml-cpu export to wrap a pointer as a `CPU_REPACK` buffer |
| owner | none | none | question (§1.3), storage |

**What carries over from `patches/llama.cpp-c1-olmoe/0011-c2-exec-map.patch`:**
- The registry is filled in `llama_mmap::impl`'s constructor (`c2_exec_register`). Any mapping made through
  `llama_mmap` is covered with no new line: Y1b and Y1c use the model mapping itself; Y1a is covered if the side
  file is opened as a `llama_mmap`. `VOLLIE_EXEC_MAP`, `llama_vollie_exec_sync()` and the `c2-exec` line carry as is.
- `0012` (touch, `mincore`) walks the same registry (`c2_exec_maps`), so `VOLLIE_TOUCH` covers the side file too.
  `0008/0009` (kv clear) and `0010` (pool, pin) do not depend on the layout.
- R67 (*device*): internal storage accepted `PROT_EXEC` (`r-xs`, errno 0); external storage refused (EACCES).
  The side file therefore lives in internal storage, which is also where rule 1 puts a cached artefact.
- What does not carry: 0011 never covered the anonymous repack copy (that is why Y1 exists), and R67's effect
  was on idle loss (1.8 against 8.7 and 0.7 against 3.5 MB/s), not on capacity under load. P1's `map=x` arm is
  the first capacity evidence.

### 1.3 Is the side file a "weight-derived artifact"? (owner question; ask only after the R68 JSON)

For "same category":
- It is computed from the served file by a committed, deterministic procedure, sits beside the model, and the
  served file is untouched. That is the 2026-11-19 definition.
- It is a **stronger** case than a calibration table: no outside text enters, and it is a bijection (byte
  permutation and XOR), so it holds exactly the model's information and can be checked by inverting it.
- The runtime already makes this exact copy in RAM on every load. Y1a changes where the copy lives, not what
  the kernel reads.
- COMMON_RULES rule 1 already says a derived artefact that the runtime computes once on the phone from the
  served file and caches in the app's files counts as runtime (R42 amendment 2).

Against:
- It is 4 GB, a second full copy of the weights; every byte read per token then comes from a file that is not
  the served file. "File not changed" holds in letter; a sceptic may say not in spirit.
- It doubles storage, a product cost the user pays.
- Shipped from the host it would be a converted model file. It must be built on the phone.

My position: same category, with a stricter handling than a table. Proposed terms for the lead's reading:
built on the phone by the runtime only, never shipped; header = model sha256, layout id and version, tensor-table
hash; the runtime refuses a mismatch and rebuilds; deletable at any time (a cache); its sha256 is deterministic,
so the host computes the expected value and the pre-registration lists it next to the model's; per-chunk identity
at `-b 512`, `-ub 32`, `-ub 1` (class B, spends no budget). Open detail: a full sha256 of 4 GB on every load
costs 3 to 11 s cold; I would check size, mtime and the header hash per load and the full sha at build time.

Questions for the owner (collected, not asked now): (1) is a 4 GB layout cache, built on the phone and
deletable, runtime under rule 1 and managed like the 2026-11-19 artifacts? (2) is 2 × storage per model
acceptable? (3) does the answer change if the layout is not ggml's 4x4 but block-major or column-major (§2)?
Y1b and Y1c need no answer.

### 1.4 Page cache against anonymous memory on Android

- `lmkd` (Android 11 and later, *published*): triggers on PSI stall (`psi_partial_stall_ms` 70,
  `psi_complete_stall_ms` 700 on high-end devices), on free swap under `swap_free_low_percentage`, and on
  **thrashing = workingset refaults as a percentage of the file-backed page cache** (`thrashing_limit` 100 on
  high-end devices, decay 10). Victims are taken by `oom_score_adj` from the top down; the foreground app is
  last. Samsung's property values on this phone are **not known**; the kit can print `getprop ro.lmk.*` if
  readable (not verified).
- Anonymous pages can leave only to zram. Free swap is 0 to 144 MB in most sessions (*device*), so the app's
  anonymous memory is in effect pinned, and too much of it ends in a kill (R9: 3.93 GB, *device*).
- Clean file pages are dropped by reclaim without any kill and come back from flash at 0.7 to 2.7 ms per MB
  (*device*). So Y1 changes the failure from "killed" to "slow", and thrashing on our file is itself a kill
  signal that removes background apps first, which is the route to more than 3.4 GB.
- One ledger: both kinds draw on the same RAM. *Device* brackets on the total (file held + app anonymous):
  3.3 GB held (R58: 2.67 + about 0.6), 4.4 GB killed as anonymous (R9: 3.93 + about 0.5), 5.4 GB not held
  (R9: 2.27 anonymous + 3.1 mapped). A lead body needs 4.04 to 4.12 + 0.3 to 0.6. P1 measures that total as file.

### 1.5 Predicted gain per step of Rule C (*estimated*; usable body = C − 0.3 GB, band 0.1 to 0.6)

| R69 outcome | stock (anonymous copy) | with Y1 (any of a, b, c) | what Y1 buys |
|---|---|---|---|
| 3.4 not held (C < 3.4; my P 0.60) | leads killed at load (4.04 > 3.93, R9); P(runs) ≤ 0.05 | usable ≤ 3.1 GB: no lead, no Yi (3.4) | nothing for the subjects; Y1 is not built. Only a body under 3.1 GB would use it |
| 3.4 held, 4.2 not | as above; Yi's 3.4 GB copy sits inside the 2.27 to 3.93 bracket, P 0.4 | usable 3.1 to 3.9: Yi fits with P 0.5; leads do not | +0.8 GB (0 to +1.6) over the 2.27 GB proven as anonymous |
| 4.2 held, 4.6 not | leads killed | usable 3.9 to 4.3: Mistral P(fits) 0.5, Qwen2.5 0.4; Rule C scores "do not fit at Q4_0" | the leads become a coin flip; a two-stage head (c2 0004) trims Qwen2.5's touched head by about 0.3 GB |
| 4.6 held (my P 0.06) | leads killed | usable ≥ 4.3 at the centre: both fit, margins +0.26 / +0.18 GB; at 0.6 GB overhead −0.04 / −0.12 | fits against killed: +1.8 to +2.0 GB over 2.27 |

Which Y1 on the day: R ≥ 1.8 (r ≤ 22) → Y1b alone is free; 1.35 < R < 1.8 → Y1b first, Y1a if the owner
agrees; R ≤ 1.35 (r ≥ 30) → Y1a. Y1c is the zero-code arm that can ride in the next kit whatever R is.

## 2. A cheaper exact column kernel

Lines at r = 24, c_b = 40.4 (R = 1.683): kill k\* = R ÷ d = **2.81 / 2.59**; 1.15-gain line k ≤ R ÷ (1.15 d) =
**2.44 / 2.25**; full byte gain (1 ÷ b = 1.55 / 1.45) needs k ≤ b R ÷ d = **1.81 / 1.78** (Mistral / Qwen2.5).

### 2.1 Why the host kernel costs 2.6 (my count of `tools/c3_colkern_bench.c`)

Per kept column and 16 B (32 weights): `and`, `ushr`, 2 `sub`, 4 `smlal` = 8 V = 0.250 V/w, plus **4 loads and 4
stores of the int16 sums in L1**. Per (row, block), whatever d is: 10 V per 8 rows for the float step =
0.039 ÷ d V/w. Total at d = 1.0 / 0.6: 0.289 / 0.315 V/w = 1.76 / 1.92 × stock by V count. The host measured
2.19 / 2.63, so 1.24 to 1.37 × of the count is memory traffic and fixed work per block (the `memset`, the reload).
On the A78 the store side is narrower than on the M4, so I expect the same kernel at or above 2.6 there.

### 2.2 K1: strip-register axpy with bias-form nibbles (exact)

- **Layout:** as the host kernel, column-major in tiles of TR rows, scales first, 18 × TR bytes per (tile,
  block), the same 0.5625 B/w. Nibbles stay in the GGUF's bias form (no XOR). **TR = 512 or 128**: TR 1024 does
  not divide Qwen2.5's 3,584 and 18,944 rows (*host*, header shapes); 512 and 128 divide every block matrix of both.
- **Kernel:** rows are taken in strips of 128 (64 B = one cache line of each column). The 128 int16 sums live in
  16 registers; nothing is stored until the strip's float step. Per kept column and line: 4 loads; `and 0x0F`
  and `ushr 4` per load = 8 V; 16 `smlal`/`smlal2` with the splatted x_j = 16 V. **24 V per 128 w = 0.1875 V/w**,
  0 stores. With TR 512 the kept columns of a (tile, block) are first touched in column order (one load per
  line; the DRAM read, sequential runs of 256 B), then the four strips run out of L1 (unit 9.2 KB).
- **Bias trick:** Σ (u − 8) x = Σ u x − 8 Σ x, with u the unsigned nibble. Σ x over the block's kept inputs is
  one integer per block; 8 times it is subtracted from every row sum (16 V per strip and block). Intermediate sums wrap
  in int16 (15 × 127 × 32 = 60,960), but `smlal` is modular and the true sum fits (8 × 127 × 32 = 32,512 <
  2^15), so the result is exact. Then `acc = fma(acc, (float) isum, dx × dw)` in block order, as stock.
- **V per touched weight:** 0.1875 + (160 + 16) ÷ (128 × 32 d) = **0.259 (d 0.60) / 0.254 (d 0.65)** = 1.58 /
  1.55 × stock. With the host's excess factor applied at 1.0 to 1.35 (no stores, so nearer 1.1):
  **k = 1.8 (1.55 to 2.2)** on the A78.
- **Exactness:** same integer sum and float step as memo-02's kernel, which was `memcmp`-identical to the 4x4
  gemv on the host. Class B against ggml's kernel. Checked on the host at threshold 0 per chunk at three settings.
  The thresholds themselves are the declared numerics change (H3), with their own table and check.
- **Price at r = 24 (blocks only; heads stay stock; O = 8 ms):**

| k | Mistral: max(bR, kd), gain on blocks, token ms, tok/s | Qwen2.5 |
|---|---|---|
| stock | –, 1.00, 176, 5.7 | –, 1.00, 180, 5.6 |
| 1.55 | 1.085, 1.55, 118, 8.5 | 1.159, 1.45, 132, 7.6 |
| **1.8** | 1.085, **1.55**, **118**, **8.5** | 1.170, **1.44**, **133**, **7.5** |
| 2.2 | 1.32, 1.275, 141, 7.1 | 1.43, 1.18, 157, 6.4 |
| 2.6 (host kernel) | 1.56, 1.08, 164, 6.1 | 1.69, 1.00, 180, 5.6 |

  At r = 30 (R = 1.347) and k = 1.8: gains 1.25 / 1.15, so K1 still passes Rule A's gain line on Mistral and is
  at it on Qwen2.5. The max model overstates gains near its corner by up to 10 % (bands memo §8).
- **What I cannot price: b_eff.** The model assumes dropped columns are not fetched. A stream prefetcher that
  runs into a dropped column fetches it anyway; with TR 128 every hole is one line and I expect most holes to be
  fetched (b_eff near 1), with TR 512 a hole is 4 lines. R69's cold `p2col` row (TR 1024, 8-line holes) gives
  the first *device* value of b_eff; K1's TR is chosen after it. If b_eff ≥ 0.9 at TR 1024, column sparsity is
  closed as a byte lever whatever k is.

### 2.3 K2: hot-core split on top of K1 (exact; needs a table)

- Inputs that the thresholds keep on nearly every token gain nothing from a column layout. Store a fraction h of
  each matrix's inputs (the most often kept, from a hot-first table) in the stock 4x4 form and run them with
  `sdot` at k = 1; a hot input that a token's threshold drops gets x_j = 0 in the q8 vector, which is exactly
  what skipping computes. The other 1 − h go through K1.
- k_eff = (h + (d − h) × k1) ÷ d; bytes are unchanged (both parts read all scales). With k1 = 1.8:

| h (share of inputs always kept) | 0 | 0.1 | 0.2 | 0.3 |
|---|---|---|---|---|
| k_eff, Mistral (d 0.60) | 1.80 | 1.67 | 1.53 | 1.40 |
| k_eff, Qwen2.5 (d 0.65) | 1.80 | 1.68 | 1.55 | 1.43 |

- **h is not measured** (H5 was not run). My prior: residual-stream inputs (`qkv`, `gate`/`up`) 0.02 to 0.10,
  `down` inputs 0.10 to 0.35, body-weighted 0.05 to 0.15, so k_eff 1.6 to 1.75: about 5 to 10 % of compute, and
  only where K1 is compute-bound (Qwen2.5 at r = 24, both from r ≈ 26). It costs a second layout in the same
  file and a table (a weight-derived artifact with its own sha and check). Build only if H5 shows h ≥ 0.2.

### 2.4 Priced and dropped (one line each)

- `tbl` product tables (two 16-entry tables per x_j, low and high byte): 2 `tbl` + 2 `zip` + 2 `add` + 1 extract
  per 16 w = 0.44 V/w, twice K1.
- Zipping four kept columns into row lanes for `sdot`: 8 expand + 16 `zip` + 8 `sdot` per 128 w = 0.25 V/w, worse
  than K1's 0.1875 and needs strides across columns.
- Four-input (quad) granularity so that `sdot` applies natively: a quad is kept unless all four inputs drop,
  1 − 0.4⁴ = 97 % at d 0.60; no bytes saved. Block sparsity already closed this (memo-02 §8).
- `usdot`/i8mm, SVE: not on the A78 or X1 of the Snapdragon 888 (*recalled*).

## 3. TEST list

Host, Tom can run now (all single thread, `tools/c3_colkern_bench.c` extended, `-march=armv8.2-a+dotprod+fp16`):

| # | test | decides |
|---|---|---|
| T1 | K1 at TR 512 and 128, both shapes, d 1.0 / 0.65 / 0.60: `memcmp` identity against the 4x4 gemv, and k. My host prediction: **k = 1.6 (1.35 to 1.95) at d 0.60** | k ≥ 2.3 on the host at d 0.60 → K1 is not better than what R69 already measures; drop it |
| T2 | Y1b in-place 4-row gemv: `memcmp` identity against the 4x4 gemv, k hot and on a cold arena. Prediction: **1.4 (1.2 to 1.7)** hot | k > 1.7 hot → Y1b loses to Y1a at the centre r; identity fails → class N or dropped |
| T3 | Y1c: `llama-perplexity --no-repack` on both models, 4 chunks at `-b 512`, `-ub 32`, `-ub 1` against stock, and `llama-bench`-style ns per weight. Prediction: per-chunk values differ in the 4th decimal or later; k 1.6 to 2.2 on the host | identical → Y1c is class B and needs nothing; else declared, with a 40-chunk delta |
| T4 | Does 4x4 gemm equal 4x4 gemv per element (one matrix, 8 input rows, `memcmp`)? | whether Y1b and K1 can run prefill as gemv per token without a declared change |
| T5 | H5, now needed twice: per-input keep frequency at each model's frontier thresholds (40 chunks), histogram per group, h at 0.98; and the union of kept inputs over 32 tokens | K2 (h ≥ 0.2 body-weighted on both → build; else closed); resident-set shrink |
| T6 | Y1a dry run on the host: write the repacked buffers to a file, map it, per-chunk identity at three settings, sha256 stable over two builds, load time first and second | that the side file is deterministic and bit-identical before any owner question |
| T7 | Lines touched by K1's layouts at TR 128 / 512 / 1024 for random masks at d 0.60 and 0.65, counted exactly, with holes of 1, 4 and 8 lines listed | the best case of b_eff per TR, to read against R69's cold row |

Device, next kit (after R69; each needs its own pre-registration):

| # | test | decides |
|---|---|---|
| D1 | A real 4 GB body with no anonymous copy: Y1c (`--no-repack`), Mistral and Qwen2.5 in one session, `map=x` on, touch on: token ms, major faults per token, `Cached`, `VmSwap`, `mincore` of the model mapping at run start and end | Rule C on a real decode (C − overhead), with zero kernel work. Run only if R69 holds 4.2 |
| D2 | `p2col`-style rows for K1 (TR 128, 512) and Y1b on a pinned A78, hot and cold, d 1.0 / 0.65 / 0.60 | k and b_eff on the phone; Rule A applied to K1 |
| D3 | Kit header: `getprop ro.lmk.*`, `MemTotal`, zram size, `/proc/pressure/memory` if readable | which `lmkd` regime the phone is in |
| D4 | Y1a against Y1b, ABBA in-run, once both exist | the real cost of the in-place kernel at the measured R |

## 4. TRY list (in order; each gated)

1. **Y1b, the in-place kernel** (gate: T2, T4; R69 C ≥ 4.2 and R > 1.35). No file, no owner question.
2. **Y1a, the layout cache** (gate: T6, the owner's answer, R69 C ≥ 4.2; first choice if R ≤ 1.35). Block-major
   variant only if the owner's answer covers non-stock layouts.
3. **K1 in the Y1a file** (gate: T1, T7, R69 Rule A not killed and cold `p2col` b_eff < 0.9, the owner's
   answer to question 3). Then thresholds (Y4) as the declared change.
4. **K2** (gate: T5 h ≥ 0.2 on both models, K1 compute-bound on the phone).
5. **A fourth decode thread with K1** (rider, no build): K1 moves the token from byte-bound to compute-bound, so
   the 1.15 × that four threads gave in campaign 1 (R13/R14, *device*) should count for more; ABBA arm only.

## 5. Where I may be wrong

- The V-pipe model explains 76 % of the stock kernel on the A78 and 73 to 81 % of the host column kernel. K1's
  band rests on it. The pipe counts are recalled, not read today.
- b_eff (prefetch into holes) can erase the byte gain of every column layout; nothing in the repository
  measures it yet.
- The one-ledger view of memory is my reading of three sessions with different phone states. P1 replaces it.
- Y1b's unaligned 18-byte blocks on four streams per thread may cost more cold than hot; T2's cold row is a
  host hint only.
- `lmkd` defaults are AOSP's; Samsung's values are unknown.

## 6. Sources

- Lab tree (base `620d2225c`): `ggml/src/ggml-cpu/repack.cpp` (3083, 3541, 5150, 5168), `src/llama-model.cpp`
  (1123, 1916, 3347), `src/llama-model-loader.cpp` (1736, 1846), `ggml/src/ggml-cpu/arch/arm/quants.c` (297),
  `common/arg.cpp` (2427); `patches/llama.cpp-c1-olmoe/0011`, `0012` and README; `tools/c3_colkern_bench.c`.
- Repository: memo-01, memo-02 (§1, §4, §8), memo-03, `memo-04-work/bonnie-bands.md`, the R69 pre-registration,
  campaign 2 `reports/README.md` (R58, R67), `reports/03_final.md` (R9, R41), COMMON_RULES rule 1.
- Web, 2026-11-19: lmkd README, <https://android.googlesource.com/platform/system/memory/lmkd/+/refs/heads/main/README.md>
  (property definitions and defaults); <https://source.android.com/docs/core/perf/lmkd> (PSI strategy, thrashing
  as refaults against the file cache); Arm Cortex-A78 Software Optimization Guide,
  <https://documentation-service.arm.com/static/5fa2bd1eb1a7c5445f2906fd> (found, tables not read; pipe counts
  stay *recalled*).
