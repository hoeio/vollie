# Kit r38 for R77: the CPU kernel costs of UD's A′ half and Y1b on the phone, build notes

Tom (`claude-opus-5-5`), 2026-11-28. The owner approved the probe on 2026-11-28 ("1로 가죠"). Scope: [tom-ud-kernels.md](tom-ud-kernels.md) §6
(the phone cells for `y1b_rows`, the K1 `down` cell at 11,008 inputs and keep 0.5, and dense Y1b), reusing r37's app, gates, JSON,
canary and parser. CPU only: no GPU, no model, 0 bytes of model weights. Not committed, not uploaded.

**Labels.** Every number in §5 is *host* (owner's Apple M4 Pro) or *device* (copied from R72's committed JSON, cited as such).
Durations and memory in §4 are *estimated*. Nothing ran on the phone.

## 1. What is in it

**Runtime patch:** `patches/llama.cpp-c3/0012-r38-cells.patch` (717 lines; about 400 of them are the embedded masks). It sits on 0009
and changes only `src/llama-vollie-probe.cpp`. It adds three blocks, `udk1`, `udr1` and `y1bd`, in a new `namespace r38`. The
probe is off unless `VOLLIE_PROBE` is set, and nothing outside the file calls it. No kernel of the CPU module changes. The kit ships
r37's CPU module byte for byte.

**App patch:** `patches/app-c3/0008-kit-r38-probe.patch` (92 lines, on 0007).
- Version 1.38.0+38, kit r38, versionCode 38.
- Block list `udk1, udr1, y1bd`.
- Button "Run c3 probe r38 (R77: UD and Y1b kernel costs)". The cap-probe, download and research-run labels say r38.
- The design string names this file.
- r37's r16 gate (up to 180 s) still runs before every block, so there is a gate between blocks. The 600 s per-block limit and the
  live-file rescue are also unchanged.

**Common to all three blocks:**
- **Weights:** random Q4_0 bytes of the real shapes. The fp16 scales are made finite in all three layouts (4x4 groups, K1 chunks,
  plain blocks). The kernels' time does not depend on the values.
- **Cold reads:** the weights live in a 256 MB anonymous arena of whole-matrix copies, and call k uses copy k mod copies. Copies per
  shape: 10 at 4096 × 11008 and at 11008 × 4096, 28 at 4096². Each matrix is 9.4 to 25.4 MB, far past the A78 L2 (512 KB), the
  L3 (4 MB) and the SLC (3 MB), so every read comes from DRAM. That is the phone's decode case.
- **x:** random Q8_0, the activation as ggml quantizes it.
- **Thread sets:**
  - `4,5,6,7`: one pinned thread per cpu. Work is handed out as 128-row units from a shared counter, with a spin barrier after
    each matrix, as R72 l4.
  - `4`: one A78, for comparison with R72 L1.
  - The calling thread sits on 0-3, as in R72.
- **Chains (triplet ordering):** each set runs stock | lever | stock | lever | … | stock. Every lever has a stock arm right before
  it and right after it, which gives one `ud trip` line per lever. `stock_diff` = |st1 − st2| ÷ mean, the same formula as r37's
  `cpu_diff`. The parser marks a triplet `v` when `stock_diff` is above 0.03; the prereg decides what that means.
- **Arm timing:**
  - 0.3 s of warm-up, then a wait for full clock. Probe arms wait up to 1 s. The ggml arms use r32's `gtime`, which waits up to 4 s.
    Either way, `ramp_ok` says whether full clock was reached.
  - Then 1.0 s measured. Thread 0 times every matrix (from prep to the second barrier), which gives `ms_p` = p10/p50/p90.
  - r37's canary runs after every arm, about 0.14 s.
- **Stock:** the probe's copy of `ggml_gemv_q4_0_4x4_q8_0` (the repack path). It is bit-identical to the library's, as memo-05 and
  R72 showed.
- **Identity on the phone:** one line per block, from `memcmp` on 256 rows.
- **Live file:** the lines are also written to `<dir>/x4-live-<blk>.txt`, so the app keeps them if a block hangs.
- **Therm lines:** `ud therm` (status, headroom, `thermal_pwrlevel`, caps) is printed before and after each block.

### 1.1 Real masks

Twelve masks are embedded, four per subject, taken from `lab/c3/bench/udmask-<subj>.bin` (E-T25's decode dump, [tom-ud-kernels.md](tom-ud-kernels.md) §0).
- **Which masks:** token / layer (2, 4), (3, 12), (5, 20), (6, 28) for every subject.
- **Content of each mask:**
  - the `gate` row bits (|up| among the top 70 %);
  - the `down` input bits (`ffn_vollie_ud != 0`).
- **Encoding:** packed 1 bit per entry, base64.
- **Generator:** `lab/c3/bench/r38_mkmask.py`, sha256 `d52deeefa2b5680644540bd890d2a780f7f092f3651bd9c20d1ddbe0a85d6072`.
- **Source files (sha256):**
  - Yi `8c24c3f415b585af0fea14f6f294169f5ac1a1b792e0881bac8bba30c0e1cd93`;
  - Llama-2 `9d46246944f7249b7c3ebca558d47fbf4a4359d64cc798bb7f767fc29e93f983`;
  - DeepSeek `3b7244a04a926cdce3b0c05523dd43299819d8687d23b60b0260219619937a54`.
- **Properties of every mask:**
  - row keep 0.7000, which gives a row list of 7,704 rows after the cut to a multiple of 4;
  - input keep 0.5000;
  - kept inputs per 32-block: minimum 8 to 10, mean 16, maximum 23 to 27;
  - K1 bytes 0.5556 of `down`;
  - every kept input is also a kept row.
- **On the phone:** each block prints these properties in one `ud mask` line per mask.
- **Use:** masks rotate per call within a subject. K1's kept lists are built on thread 0 for every call, inside the timed window,
  as a runtime would have to.

### 1.2 Blocks and cells

| block | shape | per set (4-7, then 4) | arms |
|---|---|---|---|
| `udk1` (ud-k1) | `down` 4096 rows × 11008 inputs | stock \| K1 Yi \| stock \| K1 Llama-2 \| stock \| K1 DeepSeek \| stock | 7 × 2 = 14 |
| `udr1` (ud-r1) | `gate` 11008 × 4096 | stock \| Y1b dense \| stock \| rows Yi \| stock \| rows Llama-2 \| stock \| rows DeepSeek \| stock | 9 × 2 = 18 |
| `y1bd` (y1b-dense) | 4096², 11008 × 4096, 4096 × 11008 | per shape: stock \| Y1b \| stock \| ggml_plain \| stock \| ggml_repack \| stock \| ggml_kai \| stock | 9 × 3 × 2 = 54 |

- **K1:** `k1x<0>`, R72's K1, at TR 128.
- **rows:** `y1b_rows`, in place on the file's rows. Its units are 128-entry slices of the kept-row list.
- **ggml arms:**
  - A graph of 5 to 14 `mul_mat` nodes over distinct copies, about 128 MB in total, run through r32's `gtime` (ggml's threads;
    `VOLLIE_POOL` and `VOLLIE_CPUMASK` pins, as l4 / l5).
  - The weights are forced into one buffer type per arm:
    - `CPU` is `--no-repack`. The kit builds with `GGML_LLAMAFILE=OFF`, so this arm is `ggml_vec_dot_q4_0_q8_0` through the
      generic `mul_mat`.
    - `CPU_REPACK` is the q4_0_4x4 gemv with ggml's threading.
    - `CPU_KLEIDIAI` is what the app runs; R72 showed `buft=CPU_KLEIDIAI` as the picked type.
  - If a type is missing or declined, the block prints `ud skip ... got=` and runs no arm for it.
  - These arms include ggml's per-node quantisation of x. The probe arms do not.
  - `bytes_per_mat` is the plain file's byte count, even for KleidiAI's packed layout.
- **K1 against KleidiAI on `down`:** there is no single triplet for this, because the ggml arms are only in `y1bd`. The parser
  forms it across the two blocks through their stock arms of the same shape and set. It is labelled as such.

## 2. Line formats (field names per cell)

Every line starts with `c3-probe: `.

- **`ud therm blk= when=before|after`:** `status hr thermal_pwrlevel max_mhz`.
- **`ud mask`:** `subj t l file16 ok rows_kept rows_list in_kept in_blk_min in_blk_mean in_blk_max k1_bytes_share`. Printed in
  `udk1` and `udr1`.
- **`ud identity blk=`:** 1 means bit-identical.
  - `udk1`: `rows k_in k1_real=<equal>/<masks>` (K1 on each real mask against stock fed the masked x).
  - `udr1`: `y1b=` and `y1b_rows_real=<equal>/<masks>` (each mask's rows below 256 against stock's rows).
  - `y1bd`: `y1b=<equal>/2` (4096 and 11008 inputs).
- **`ud setup blk=`:** `shape(s) arena_mb copies warm_s meas_s sets setup_s`.
- **`y1bd` only:**
  - `ud backend blk=y1bd`: the CPU device and its buffer types.
  - `ud ggml blk=y1bd rows= k_in= copies= probe_copies= make_s=`: one per shape.
  - `ud skip` when a buffer type is missing.
- **`ud arm`** (one per arm):
  - Common fields: `blk pos t_ms cores threads kern subj rows k_in d mats ms_per_mat ms_p ns_per_touched_w bytes_per_mat gbs buft hr status`.
    - `t_ms` is ms since the block's setup ended.
    - `kern` is one of `stock k1 rows y1b ggml_plain ggml_repack ggml_kai`.
    - `d` is the touched share: the mean kept inputs for K1, the mean list rows for `rows`, otherwise 1.
    - `ms_p` is p10/p50/p90 of the per-matrix ms; ggml arms print `-1/-1/-1`.
    - `ns_per_touched_w` = ms × 10⁶ ÷ (rows × K × d).
    - `buft` is `probe` for the probe's own kernels.
  - Probe arms then add r32's tail: `cpu_ok thr mhz mhz_range ramp_s ramp_ok max_mhz`, with `thr` in the l1-l4 format, so the
    parser's M / U / P / C labels apply.
  - ggml arms then add r32's `gline`: `set threads calls secs proc_cpu_over_wall proc_nivcsw cpu0_ok cpu0_off_calls mhz mhz_range ramp_s ramp_ok max_mhz`.
- **`x4 <blk> can`** (after every arm; r37's canary, unchanged):
  `pos t_ms c5_ns_per_w c5_pin c5_at c5_cur spin7_mhz spin7_pin spin7_at spin7_cur spin4_mhz spin4_pin spin4_at spin4_cur dcvsh isolated thermal_pwrlevel`.
- **`ud trip`** (one per lever, printed after each chain):
  `blk pos cores kern subj rows k_in d st1_ms lever_ms st2_ms stock_diff t_vs_stock k`.
  - `t_vs_stock` = lever ms per matrix ÷ the mean of st1 and st2.
  - `k` = `t_vs_stock` ÷ `d` (per touched weight). This is the same definition as the host bench and R72 L1.
- **`<blk> end s=`:** the block's seconds.

**Parser:** `tools/c3_probe_parse.py`, new section "R77" (`r38report`, `r38selftest`).
- **Tables:** masks, triplets (with the `v` mark), arms (with labels), canaries.
- **"R77 inputs" per core set:**
  - K1 `k` per subject;
  - dense Y1b `k` on `gate`, `rows` `k` per subject, and `rows` ms ÷ dense Y1b ms;
  - per shape: `t_vs_stock` of Y1b, ggml_plain, ggml_repack and ggml_kai, and Y1b ÷ ggml_kai;
  - K1 `down` time ÷ ggml_kai `down` time, across the two blocks through their stock arms.
- **Void echoes:** triplets over 3 %, arms with `ramp_ok=0`, off-core arms, canaries with `thermal_pwrlevel` above 0, and identity
  lines that are not all equal.

## 3. Hashes for the pre-registration

| item | file | sha256 |
|---|---|---|
| runtime patch | `patches/llama.cpp-c3/0012-r38-cells.patch` (717 lines, on 0009) | `a5945fc292af4c8e15506ecdc716a5de1fd51ab5e93cd06bb2534190f430bc13` |
| app patch | `patches/app-c3/0008-kit-r38-probe.patch` (92 lines, on 0007) | `f6ea7c6a9126330451868b1f078a72d253aa2853c48941e6faa4c4061bb287fa` |
| parser | `tools/c3_probe_parse.py` (`r38report`, `r38selftest`; working copy) | `3fb2be126b9f6e2033ac64eab7e57111c67d79d1c8f74aa1e0cdb44b614713ee` |
| probe source after the series | `src/llama-vollie-probe.cpp` (tree `lab/llama.cpp-c3r38`) | `0f9095d098518c6d99d318e8210dff71a088bf97f1f15a639bf4c5138651ef67` |
| APK | `lab/dist-c3/r38/vollie-c3-probe-r38.apk`, 16,711,466 B, versionCode 38, versionName 1.38.0, package `com.insaneutils.vollie.c2`, certificate SHA-256 `c49dde856ca5bb044caf878a39a03077a8f958d841c47510672f7d269fbb1716` (same as r37; `apksigner verify`) | `c28673904cfa2be817ed9607c174bda9f36849f06164919d8dfc292456544111` |
| build record | `lab/dist-c3/r38/BUILD-r38.txt` | `6b922b213f744fd89c73c8e92141a57f24978c8187726eac318f3593b0a44614` |
| sums | `lab/dist-c3/r38/SHA256SUMS` (APK and BUILD; `shasum -c` OK) | `15958f1de88e067221f4b4459d12d569709b65db57b97e6c520a1324b1e98e22` |
| mask generator | `lab/c3/bench/r38_mkmask.py` | `d52deeefa2b5680644540bd890d2a780f7f092f3651bd9c20d1ddbe0a85d6072` |

Libraries in the APK, against r37:

| library | sha256 | against r37 |
|---|---|---|
| `libllama.so` (c3 0001, 0002, 0003, 0005, 0007, 0008, 0009, 0012) | `71b660429cff6433efafa1b2931d6391406afe5bbd3838162ad4b9fea4d42622` | changed |
| `libapp.so` (Dart) | `2e733d01d32f476752d86828c551c5b0f3812a25cf9bb9c13784904ce666fff4` | changed |
| `libggml-cpu-android_armv8.2_2.so` | `46e0735d2824ba6f3a156e219e058a701f6024fdbecbb2ef937b3127115a9694` | byte-identical (= r33 to r37) |
| `libvollie_x4c.so`, `libvollie_clc.so` | as in BUILD-r37 | byte-identical |

Every other library is also byte-identical to r37 (checked file by file): libggml-base, libggml, the other CPU variants,
libllama-common, libllamadart, libmtmd, libflutter and libdatastore_shared_counter.

**Build:**
- **Tree:** `lab/llama.cpp-c3r38`, a clone at 620d2225c with `patches/llama.cpp` 0001-0003, `patches/llama.cpp-c1-olmoe`
  0001-0012 and c3 0001, 0002, 0003, 0005, 0007, 0008, 0009 applied. It equals the r37 tree file for file (`cmp`). Then 0012 was
  applied, and the new files were marked with `git add -N`.
- **Commands:**

```
sh android/build_cpu.sh lab/llama.cpp-c3r38 <libs>          # libllama.so taken from here; the CPU module from the r37 APK
LIBS_FROM=<libs> APP_SERIES="<app-c2 0001-0012> patches/app-c3" sh tools/c2_build_kit.sh lab/llama.cpp-c3r38 lab/app-c3 \
  patches/llama.cpp patches/llama.cpp-c1-olmoe <dir with c3 0001 0002 0003 0005 0007 0008 0009 0012>
```

- **After the build:**
  - The APK was copied to the plain name.
  - The BUILD file adds r37's notes for `libvollie_clc.so` and `libvollie_x4c.so`, plus a series note and a blocks note.
  - The build-time copies remain in `lab/dist-c2/`, as for r37.

**Numerics: none, and checked.**
- 0012 touches only the probe file. No inference path calls that file, and the CPU module is r37's file byte for byte.
- Host check (*host*): Llama-2-7B served file, `llama-perplexity -c 512 -b 512 -t 8 -ngl 0` on macOS CPU builds of the r37 tree and
  the r38 tree. Per-chunk values are **identical**:
  - `-b 512`, 4 chunks: 5.8104, 6.3386, 7.0099, 7.9916;
  - `-ub 32`, 4 chunks: 5.7851, 6.3350, 7.0107, 7.9935;
  - `-ub 1`, 2 chunks: 5.7791, 6.3466.
- r38 with `VOLLIE_PROBE=1` at `-b 512` gives the same values.
- Logs are in the session scratchpad and are not kept.

## 4. Phone time and memory (*estimated*)

| block | arms | estimate | basis |
|---|---|---|---|
| `udk1` | 14 | 25 to 40 s | arm 0.3 + 0 to 1.0 clock wait + 1.0 + 0.14 canary; arena fill and identity about 1 to 2 s |
| `udr1` | 18 | 30 to 50 s | as above |
| `y1bd` | 54 (36 probe, 18 ggml) | 85 to 200 s | probe arms as above; ggml arms wait up to 4 s for clock (`gtime`); nine 128 MB ggml weight sets (repack / KleidiAI conversion), about 5 to 15 s |

- **Normal case:** about 2.3 to 3 min of blocks with a cool start. r37's arms waited at most 0.5 s for clock on the phone (R76
  *device*). The longest case is about 5 min, if every arm waits its full cap; R72 hit the 4 s cap in 42 % of its cells (*device*).
- **Gates:** three r16 gates of up to 180 s each come on top.
- **Host functional run** (*host*, Docker arm64 Linux, 8 vCPUs; no cpufreq, so no clock wait): 21 s, 27 s and 81 s, 2 min 9 s in
  total.
- **Load:** about 88 s of measured kernel time; the rest is warm-up, canaries and setup.
- **Memory peak:** `y1bd` holds the 256 MB arena plus three ggml weight sets of about 128 MB each, so about 0.65 GB. `udk1` and
  `udr1` hold the 256 MB arena only. That is under r37's 1.2 GB.
- **Ceiling:** 3 × (180 s gate + 600 s block limit) = 39 min, reached only if every block hung.

## 5. Host ratios and device priors mapped to each cell (for the bands)

**Sources.**
- **H** = [tom-ud-kernels.md](tom-ud-kernels.md), *host*: M4, one thread, cold = at least 1 GiB of copies.
- **D** = this kit's Docker functional run, *host*: M4 Pro under Docker, 4 threads unpinned to physical cores. It is a functional
  run, not a measurement design. It is shown only because H has no 4-thread values. The ranges span two runs (before and after the
  final 1 s clock-wait cap; no cpufreq in Docker, so the cap has no effect there).
- **R72** = *device*, R72 L1 / L2 / L5 (committed JSON). These are the closest phone cells already measured.
- The prereg should take bands from R72 where a cell exists. memo-05 and R72 showed the M4 misleads.

| cell (parser field) | H (1 thread, cold) | D (Docker, *host*): cpu 4 / 4-7 | R72 *device* prior |
|---|---|---|---|
| `udk1` K1 `k`, real masks, keep 0.5 | 1.64 to 1.71 (TR 128) | 1.69 to 1.70 / 2.14 to 2.23 | **2.621** (cpu 4, cold, 11008 inputs, 1024 rows, random mask keep 0.50, ramp_ok 1; t_vs_stock 1.310) |
| `udk1` K1 `t_vs_stock` | 0.82 to 0.86 | 0.85 / 1.07 to 1.12 | 1.310 (as above). R72 L4 4-7 whole layer at keep 0.58 to 0.60: K1 ÷ stock 1.05 to 1.13 |
| `udr1` dense Y1b `k` (11008 × 4096) | 1.53 to 1.78 | 1.53 to 1.58 / 1.80 to 1.92 | none (D2's `p2d` never ran on the phone) |
| `udr1` rows `k` (70 %, real) | 2.57 to 2.60 (t 1.80 to 1.82) | 2.97 to 3.20 / 2.54 to 2.80 | none |
| `udr1` rows ms ÷ dense Y1b ms | 1.02 to 1.19 (1.80 to 1.82 ÷ dense 1.53 to 1.78) | 1.33 to 1.43 / 0.98 to 1.01 | none |
| `y1bd` Y1b `t_vs_stock` 4096² / 11008 × 4096 / 4096 × 11008 | 1.50 to 1.53 / 1.53 to 1.78 / 2.41 to 2.46 | 1.57, 1.54 to 1.58, 2.52 to 2.63 / 1.83 to 1.85, 1.80 to 1.92, 2.01 to 2.30 | none |
| `y1bd` ggml_plain (`--no-repack`) | 2.00 to 2.01 / 1.97 to 2.00 / 1.99 to 2.09 (direct `vec_dot`) | 2.17 to 2.23 / 1.79 to 2.00 | none |
| `y1bd` ggml_repack | 1.00 to 1.02 (lib gemv) | 0.99 to 1.02 / 0.96 to 1.01 | none for repack. Probe stock on cpu 4 cold: 0.047 to 0.051 ns/w (11.1 to 11.9 GB/s) |
| `y1bd` ggml_kai | not run (H had no KleidiAI build) | 1.01 to 1.04 / 0.99 to 1.04 | L5 Q4_0 KleidiAI, cpu 4 cold: 0.0504 to 0.0522 ns/w (10.8 to 11.2 GB/s), so about 1.0 to 1.05 × the probe stock above; L2 on 4-7 cold: 0.0211 ns/w (26.7 GB/s) |

**Price lines from [tom-ud-kernels.md](tom-ud-kernels.md) §4** (*estimated*, so the lead can put kill lines on these fields):
- **R3** (K1 on `down`, the `udk1` `k`) is dead above a break-even KC of 4.5 to 4.7 at C 50, or 3.15 at C 40. If the phone repeats
  R72's 2.62, R3 survives at both.
- **R1** (the `udr1` rows `k`) has break-even kr 3.12 to 3.28 at C 50 (1.90 if the skipped rows save no bus time), and 2.18 at C 40.
  Ebony's E-T27 line is rows `k` over 1.3, which means R1 is worth about −4 ms.
- **B-T2's kill line** (Y1b on `down`) is `y1bd` Y1b 4096 × 11008 `t_vs_stock` over 1.8 with every arm. This kit has one Y1b arm.
  The arm without the zeroing moves and the prefetch arm were not built (§6).
- **Whole model:** H's whole-model weighting (attention 1 − F at the 4096² cost, gate + up 2F/3, down F/3) can be applied directly to
  the three `y1bd` Y1b `t_vs_stock` values.

**Reading the D column** (*host*, with care): at 4 threads the stock arm becomes bus-bound, and the ratios move toward the byte ratio.
- K1 `t_vs_stock` rises to about 1.1.
- Y1b on `down` falls from 2.6 to about 2.1.
- `rows` ÷ dense Y1b falls to about 1.0.

The phone's bus differs, so these values are not a band. They only say that the 4-7 and cpu 4 cells can differ in direction, as R72
L1 against L4 already did.

## 6. Checks (all *host*; host timings say nothing about the phone)

- **Source:**
  - The series 0001, 0002, 0003, 0005, 0007, 0008, 0009 and 0012, applied to a clean archive, reproduces the r38 probe file (`cmp`).
  - The app patch applied to r37's three files reproduces the r38 files (`cmp`).
  - `tools/c2_build_kit.sh`'s own tree check passed.
- **Warnings:** macOS `clang++ -std=c++17 -fsyntax-only -Wall -Wextra -Wcast-qual` gives only 0001's two old warnings. The Android
  NDK 28.2 build gives the same two. Linux gcc gives old-code warnings only: the pragma, the cast and the indentation.
- **Functional run:** arm64 Linux in Docker (`vollie-linux-build`, `--cpus=8`), with libllama built with KleidiAI. That build needed
  `-include cstdio` for gcc, because of a missing include in old `vollie-release.h` that NDK clang does not need. The three blocks
  were run through `tools/c3_x4child.c` with `VOLLIE_PROBE=1`:
  - All arms produced lines, and every `cpu_ok` read 4/4 or 1/1.
  - Every identity line was 1, or 12/12, or 2/2.
  - The ggml arms found `CPU_KLEIDIAI`, `CPU_REPACK` and `CPU`, and each repacked as expected (`repack: ... q4_0_4x4`).
  - The block times were 21 s, 27 s and 81 s.
  - Without the tee, `<dir>/x4-live-udk1.txt` was written (50 lines).
  - With the knob unset, the entry returns -1. With an unknown block, it returns -2.
- **Parser:**
  - `python3 -I tools/c3_probe_parse.py --selftest` prints `r72 ok`, `x3 ok`, `x4 ok`, `r36 ok`, `r37 ok`, `r38 ok` and `selftest ok`.
  - The host run, wrapped as a report JSON, parses into all tables.
  - All nine committed c3 device JSONs give output byte-identical to the committed parser's.
- **Not checked:**
  - Anything on a phone.
  - Whether the phone keeps cpu 7 pinned in the 4-7 set. R72 lost cpu 7 in many cells; the `thr` labels and `cpu_ok` show it per arm.

## 7. Not done, limits

- **Not built:**
  - Y1b without the zeroing moves, and the next-group prefetch arm (B-T2's arms). The host had the first at +0.05 to +0.1 and the
    second worse when cold. Each would add 2 arms per shape and set.
  - Hot cells.
  - X1-alone cells.
  - The 4-thread Y1b in the real decode path (0011). The kit measures kernels, not 0011's `mul_mat` loop.
- **UD's own selection:** the top-k selection is not timed. It is still priced at 0.5 ms each (Bonnie).
- **Thread 0's serial prep:** K1's lists are built on thread 0 while the other three wait at the barrier. This is in the 4-7 time,
  as in R72 l4. A runtime could build them in parallel.
- **x quantisation:** the ggml arms quantize x per node and the probe arms do not, so ggml arms carry a small extra cost (11,008 or
  4,096 floats per matrix).
- **Housekeeping:** the series and app READMEs need their 0012 and 0008 rows (Annie or the lead).

## Summary

1. Kit r38 is built: `lab/dist-c3/r38/vollie-c3-probe-r38.apk` (1.38.0+38, versionCode 38, same certificate, sha256 `c2867390…4111`)
   with `BUILD-r38.txt` and `SHA256SUMS`. It is not uploaded and not committed.
2. New runtime patch `0012-r38-cells.patch` with blocks `udk1`, `udr1` and `y1bd` (CPU only, 12 real UD masks embedded, off unless
   `VOLLIE_PROBE`), and app patch `0008-kit-r38-probe.patch`. Every arm is in a stock | lever | stock chain on 4-7 and on cpu 4, with
   r37's canary after each.
3. No numerics change: the CPU module is byte-identical to r37's (`46e0735d…9694`). The host per-chunk PPL of the r37 and r38 trees
   is identical at `-b 512`, `-ub 32` and `-ub 1` (Llama-2, 4 / 4 / 2 chunks).
4. Host: identities pass, all three blocks ran in Docker (2 min 9 s), and the parser's `R77` section and selftest pass with old
   reports unchanged. Nothing was run on the phone.
5. Phone time is estimated at 2.3 to 3 min of blocks (up to about 5 min if every clock wait hits its cap) plus three gates, with a
   memory peak of about 0.65 GB. R72's *device* K1 k at 11,008 inputs and keep 0.50 is 2.62 against the host's 1.70: use the device
   prior for the bands.
