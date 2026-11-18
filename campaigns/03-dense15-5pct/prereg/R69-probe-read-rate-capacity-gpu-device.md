# Pre-registration R69: dense read rate, compute ceiling, resident capacity and GPU headroom on the phone (c3 probe kit r29)

Committed and pushed **before** the APK is uploaded. Do not edit afterwards; changes go into `R69-amendment-1.md`.
No tag (owner's wish); the commit hash and push time show the order.

- **Date / author:** 2026-11-19, clyde (lead, main session). First device round of campaign 3. The number follows
  R68 (campaign 2). A campaign-2 claim round, if one comes, takes the next free number.
- **Agent models** (full IDs): lead `claude-opus-5-5`; bonnie `claude-fable-5-2` (bands, `reports/memo-04-work/bonnie-bands.md`);
  ebony `claude-fable-5-2` (memo-03, block-sparsity review); tom `claude-opus-5-5` (probe patch, kit, parser,
  memo-02); annie `claude-sonnet-5-5` (books).
- **Build under test:** `vollie-c3-probe-r29.apk` (uploaded under that plain name; the BUILD file's first line names
  the build-time file), 16,430,946 bytes, sha256 `02736b568503624fa036f86c994c5407516605522f1db901ed2f0d58b866eaf9`,
  versionCode 29, the same package and debug certificate as r28. Runtime: kit r28's c2 runtime plus
  `patches/llama.cpp-c3/0001-probe.patch` (sha256 `ac5c7057…`, knob `VOLLIE_PROBE`, off when unset). App: r28's
  patches plus `patches/app-c3/` (sha256 `a1c51b22…`). Record: `lab/dist-c3/BUILD-r29.txt`. Parser:
  `tools/c3_probe_parse.py`.
- **Identity (*host*, memo-02):** the knob unset gives stock per chunk at `-b 512`, `-ub 32` and `-ub 1`. The probe runs
  before any graph and changes no numerics.
- **Device and state:** the owner's Galaxy Z Flip 3. Power saving **off** and the charger out, both recorded. The
  deep-sleeping list state is recorded. No model is loaded during the probe.
- **Labels:** every number in §3 is *estimated* until the result says *device*.
- **Starts from campaign 2:** kit r28's protocol (r16 cool-down gate before each block, heat and clock series,
  power and charger records).

## 1. What is measured (memo-02 §6, with the six fixes of memo-04-work §7)

The order is P2 and X1 (cool phone), then X2, then P1 last.

| block | content |
|---|---|
| P2 | cold 1 GB arena: plain read loop and the stock `q4_0_4x4` gemv; core sets 1, 2 and 3 A78, X1 alone, X1 + 2 A78, X1 + 3 A78, 3 A78 + 4 A55; 1 s warm-up and 2 s measurement per cell; clocks of every policy before and after |
| P2hot | the same gemv on a slice that fits each core class's L2, one pinned thread per class |
| P2col | the exact column kernel (memo-02 H6) and the block-major skip-list kernel (H6b) on a pinned A78 at kept fraction 1.0, 0.65 and 0.60, hot and cold: the per-touched-weight cost k on the phone |
| X1 | 1, 2 and 4 streams per thread on 3 A78s |
| X2 | GPU `float4` read and a Q4_0 dot kernel (OpenCL, `dlopen`); then CPU and GPU reading together; the GPU kernel is named on every line |
| P1 | scratch files of 2.7, 3.0, 3.4, 4.2 and 4.6 GB on internal storage, `map=r` and `map=x`, 20 passes each; the files are deleted at the end |

## 2. Budget accounting

No numerics change and no model run. Nothing is spent.

## 3. Predictions (*estimated*; centres with 80 % bands, from memo-04-work/bonnie-bands.md)

| quantity | prediction |
|---|---|
| r = P2 `q4x4`, 3 threads (X1 + 2 A78, else 3 A78), GB/s | **24 (19 to 30)** |
| P2 plain read, 3 A78 | 27 (20 to 36) |
| q4x4 ÷ read, 3 A78 | 0.90 (0.75 to 1.00) |
| (X1 + 3 A78) ÷ 3 A78, q4x4 | 1.13 (1.00 to 1.30) |
| P2hot cycles per weight: A78 / X1 / A55 | 0.105 (0.095 to 0.118) / 0.078 (0.070 to 0.090) / 0.33 (0.22 to 0.48) |
| c_b = 0.87 × 0.5625 × Σ 1/ns_per_w over X1 + 2 A78, GB/s | 40.4 (35 to 45) |
| P2col k on the A78, column kernel at 0.60 to 0.65 kept | 2.6 (2.0 to 3.3) |
| P2col k, block-major skip list | 0.95 (0.85 to 1.10) |
| X1 gain at 2 / 4 streams | 1.05 (0.95 to 1.25) / 1.05 (0.88 to 1.35) |
| X2 block prints all lines | P 0.80 |
| X2 GPU read | 30 GB/s (15 to 42) |
| X2 GPU q4 kernel | 10 GB/s (3 to 22) |
| S = CPU + GPU sum ÷ CPU-alone q4x4 (read kernel on the GPU) | 1.25 (0.95 to 1.60) |
| P1 capacity C (largest step held, all smaller steps held) | **3.1 GB (2.6 to 3.8)**; P(3.4 held) 0.35, P(4.2 held) 0.12, P(4.6 held) 0.06 |
| clocks: 2,419 on cpus 4 to 6 and 2,841 on cpu 7 in ≥ 80 % of cells | P 0.8 |

A step is **held** when its median rate over passes 6 to 20 is at least 0.80 × the same arm's median at 2.7 GB, and
its median major faults per pass are at most 50. The better of the two map arms counts.

## 4. Decision rules (scored as written)

Definitions: r as in §3; c_b as in §3; R = c_b ÷ r; d = kept fraction (Mistral 0.60, Qwen2.5 0.65, memo-02 H3);
k\*(r) = max(R, 1) ÷ d.

- **A, column kernel (Y3/Y4).** k is P2col's column-kernel k at the model's d.
  - Per model: **pass** if k ≤ k\*(r) and the token gain max(R, 1) ÷ max(b × R, k × d) ≥ 1.15, with
    b = 1 − 8/9 × (1 − d). Action: build Y3/Y4 for that model.
  - **Kill** if k > k\*(r). Otherwise "open, gain under 1.15".
  - If one model passes and the other does not, the result is reported as "did not generalise", and the lever
    stays open on the model that passed.
- **B, R20 (draft speculation).**
  - **Kill** if r ≥ 0.71 × c_b.
  - **Reopen as a host item** if r ≤ 0.61 × c_b. The host item covers the greedy-match rate on the fixed prompt
    set and greedy identity at batch 2. It is killed inside if the match rate is under 0.4 on either model.
  - In between, it is re-priced on paper only.
- **C, residency.**
  - **Kill for the two lead candidates** if C < 3.4 GB, that is, the 3.4 step is not held in either arm. Strict
    dense then continues only as host research.
  - **Pass for the lead candidates** if the 4.6 step is held. Y1 (kernel layout in a mapped file, no anonymous
    copy) then becomes the first build.
  - If 3.4 is held and 4.6 is not, the result is "the lead candidates do not fit at Q4_0". Reserves Yi-1.5-6B,
    a lower-bit served file (H2) and a smaller resident set (H5) are then priced against C.
- **D, CPU + GPU split (rank 1) and multi-stream GEMV (rank 2).**
  - **Kill the GPU lane** if S < 1.15, or if X2 read is under 10 GB/s after one retry with `-cl-opt-disable`, or if
    the block prints no number.
  - **Rank 1 stays open** if S ≥ 1.30 and S ≥ 1.15 × the best X1 gain.
  - **In between,** rank 1 is parked, and rank 2 is built only if the X1 gain is at least 1.15.
  - **Bus limit:** if (3 A78 + 4 A55) ÷ 3 A78 ≤ 1.10 and S < 1.15, ranks 1, 2 and 4 close together.
- **Void (kit):**
  - the APK hash differs;
  - power saving reads on, which voids every block run in that state;
  - the charger reads plugged, which voids that block;
  - the process is killed during a block, which voids that block.
- Misses are written as misses.

## 5. Claim rule

Not a goal claim. Even the best outcome of every rule prices under 10 tok/s on a 4.0 GB body (*estimated*,
memo-04-work §6). The probe decides which levers are worth a build, and whether a 7B body fits at all.

## 6. What a result means

- C < 3.4 GB: on this phone and at Q4_0, the campaign's goal is out of reach on residency alone. This is stated
  with the number. Research continues on host-side levers that shrink the resident set.
- Rule A kill and rule D kill together: the remaining speed path is the bit-identical work carried over from
  campaign 2 plus rank 2. 15 tok/s is reported as not reachable on this hardware, with the measured r, c_b and C.
