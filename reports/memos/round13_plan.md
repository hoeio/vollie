# Round-13 plan: the non-kernel half of the expert slot, a block-gated ABBA rule, kit fixes (Bonnie, 2026-11-07, after R38)

Labels: *device* = `reports/device/r8_2026-11-06.json`, `r10`-`r12_2026-11-07.json` (per-block tables via `tools/research_parse.py` helpers, scratch
scripts, nothing in `lab/` touched); *host* = `lab/tiny-op.patch` read (MMID=3 code, `Q3K_PF` code); *estimated* = arithmetic; *arch note* = a
Cortex-A78 property I state from memory and flag for Tom to confirm in the TRM. The kernel speed line is closed (R38); nothing here reopens it. The
R38 numbers appear only as calibration for the gate design (§B) and as the measurement of how much a major fault costs.

## A. The non-kernel half of the expert slot: what it is made of

**Inputs (device).** Slot = gate + up + down of one Q3_K expert = 9,216 row-blocks (256 weights, 110 B) = **1,013,760 B**; 3 threads → 3,072 row-blocks
per thread per slot. Clocked bench on an A78 at 2,112 (R37): old kernel 117.49 ns/row at K 1,536 = **19.6 ns per row-block**, new **14.5**, cache-hot
(338 KB slice, in L2). In situ (r8 clean day, 2,112): `MUL_MAT_ID` compute 19.3 ms / 160 slots = 120 µs; skip ABBA 116 µs → **37.8-39 ns per row-block**.
So per thread per slot: kernel **60 µs** (old) / **44.5** (new); **non-kernel 56-60 µs** (old kernel in situ) — I use 50-60 for the new kernel
because part of the memory stall was hidden under the longer compute and is exposed when compute shrinks (that is also why the kernel's in-situ Δ was
always going to be less than its bench ratio). R27's clock experiment (×0.736 clock → `MUL_MAT_ID` ×1.24-1.30): time = C/f + M gives a clock-invariant
M = **16-33 % of the slot = 19-40 µs** (*estimated*; a lower bound on memory-side time, since SD888 bus/DDR DCVS follows the CPU clock).

**What MMID=3 does per op (host, patch lines 3095-3165).** Every thread: `from_float` of the 1,536-float activation to Q8_K (≈ 1 µs, redundant ×3), builds
the row map (~0.2 µs), then for each routed expert claims **32-row chunks** with `atomic_fetch_add` on a per-expert 64-B-padded counter (gate/up: 16
chunks of 21,120 B; down: 48 chunks of 7,040 B). Per slot across the three ops: **80 successful + 9 failing RMWs**, ≈ 30 per thread. No barrier inside the
op; the op-end barrier is in the `wait` line, not in `compute` (compute = thread 0's time).

| term (per thread per slot, new kernel, 2,112 MHz) | arithmetic | µs | label |
|---|---|---|---|
| kernel issue | 14.5 ns × 3,072 | **44.5** | device (bench) |
| chunk atomics | ≈ 30 contended RMWs/thread × 80-200 ns (cross-core line transfer + the dependent-address pipeline drain) | **2-6** | estimated |
| per-op fixed (Q8_K quantize, map, dispatch, skipped-row memset) | 3 ops × ≈ 1.5 µs / 4.1 slots per layer | **1-2** | estimated |
| x4 wrapper call + dst stores per 4-row group | 768 groups × ≈ 24 ns (gate/up) … 3 ns/row-block (down) | **2-4** | estimated |
| thread arrival / chunk tail | measured `MUL_MAT_ID` wait 0.75 ms/token clean ÷ 164 slots = 4.6 µs — **in the wait line, not in compute** | (4.6) | device |
| **page-granular stream**: 1,013,760 B / 4,096 = **247 pages per slot, 82 per thread**; per new 4-KB page: TLB miss + page walk whose PTE is not cached (5 MB of PTEs for the 2.5 GB expert set > L3), ≈ 100-150 ns; the first demand line after the walk ≈ 100-150 ns, not overlapped with it; hardware-prefetcher restart at the page boundary (*arch note*: A-series prefetchers do not cross 4-KB pages), first 2-3 lines at latency, partly hidden by OoO ≈ 100-200 ns | 82 × 300-500 ns | **25-41** | estimated |
| residual (L2→L1 refill contention, DVFS-coupled memory latency, profiler) | by subtraction | 5-15 | estimated |
| **non-kernel total** | | **50-60** | |

**Which is the large term.** Bandwidth is not: 1.01 MB / 100 µs = **10 GB/s = 42 % of the 23.9 GB/s ridge** (device-shown on dense `MUL_MAT`). Atomics and
arrival together are ≤ 10 µs/slot by count (atomics ≤ 89 RMWs at ≤ 200 ns spread over 3 threads; arrival sits in the 0.75 ms/token wait line). The large term
is **latency per 4-KB page**: ~250 page starts per slot, each a serialised TLB walk + cold first line + prefetcher restart. Two device facts fit it and nothing
else does: (1) `Q3K_PF` (R23, `__builtin_prefetch` = `prfm` of the next 4 rows) gave +0.6 / −3.8 = noise — *arch note*: Cortex-A cores drop a software
prefetch that misses the TLB, so `prfm` cannot pull a page walk forward, and the next rows are usually in the same page anyway; (2) `EXPERT_ANON=16` (R24)
cut minor faults 60 % with ms flat — the anon copy still had 4-KB pages (file THP is off on this kernel: `thp:` empty), so a page-fault term fell and a
page-walk term did not. **Where I disagree with Clyde's framing:** "stream / atomics / arrival" puts three names on one half; by the arithmetic two of
them are ≤ 10 % of it, and the stream term is *per-page latency*, not bytes — so "any proposal must cut bytes" does not apply to the expert op the way it
does to dense `MUL_MAT`. A second disagreement: the host cannot falsify anything here (M4: 16-KB pages, 4-wide ASIMD, fast walker); the instrument is the
device `cpu-bench`, which the kit already carries.

### A.1 Levers (bit-exact, bytes/token unchanged, numerics unchanged)

| rank | lever | mechanism | predicted `MUL_MAT_ID` Δ at 2,112, fault-free (parser line = compute + wait) | falsifier | exact? |
|---|---|---|---|---|---|
| **1** | **`GGML_MMID_TOUCH=1` — page-head touch at chunk claim.** After the `atomic_fetch_add` returns chunk c, before the kernel: one *demand load* (`volatile` byte read into a sink, not `prfm`) of the first byte of each 4-KB page the chunk spans (gate/up 5-6 pages, down 2-3). 6 independent loads start 6 page walks + 6 line fills in parallel; the kernel then meets warm TLB entries and primed lines | converts 5-6 serial 300-500 ns page starts per chunk into one overlapped ≈ 400 ns → recovers 50-80 % of the page term: −12 to −32 µs/slot × 164 slots = **−2.0 to −5.2 ms; central −3.0; band −1.5..−4.5** | (a) the bench below shows cold-vs-hot ≈ 0 (then the stall is not the stream and the lever is dead before a kit); (b) device ABBA Δ > −0.5 under the §B gate; (c) `MUL_MAT`/head move (they must not: 0 ± 0.3) | yes — same loads the kernel issues µs later; fault count unchanged (the touch takes the minor fault the kernel would have taken) |
| 2 | `GGML_MMID_FUSE` (gate + up of one expert in one chunk loop, GLU folded) — Tom's > 1-day item from R34 | −40 barriers/token (746 → 706), −40 ops' fixed costs, one Q8_K quantize, GLU op gone | **−0.4 to −0.8** (R34's estimate, unchanged) | Δ > −0.2; `GLU` not → 0 | yes (same vec_dot per row, same float order if the GLU is applied per element as today) |
| 3 | fewer chunk RMWs: claim 2 chunks per `fetch_add` while ≥ 2·nth chunks remain, 1 at the tail | halves the 2-6 µs/slot atomics term, tail imbalance unchanged | **−0.15 to −0.5** | Δ > 0 | yes |

Not proposed: `MM_CHUNK` ≠ 64 / MMID chunk ≠ 32 (closed, and larger chunks raise the tail); `prfm` prefetch (closed, and the wrong instruction for a TLB-bound
stall); `MADV_HUGEPAGE` on the file mapping (file THP off, `thp:` empty); expert-major thread ownership (a 338-KB stream has the same page starts as 16 chunks:
no gain by this model); guided/finer tails (the whole imbalance budget is 0.75 ms/token).

**Cheapest falsifier, before any kit round (Tom, bench-only, ≈ 30 lines in the existing `cpu-bench`):** three extra fields on the K 1,536 and K 512 lines,
1 thread, same slice sizes: `hot=` (today's number), `cold=` (the slice placed in a 64-MB arena, each trial on a slice ≥ 8 MB from the last so L2/L3 hold none of
it, same in-situ access order: 32-row chunks, 4-row groups), `cold_touch=` (cold + the page-head demand loads at each 32-row chunk start). Predictions
(*estimated*, A78 at 2,112, ns per row-block): hot 14.5; **cold 28-36** (the in-situ 32-33 should land inside — if cold ≈ hot, the non-kernel half is not the
stream and §A is wrong); **cold_touch 19-26** (≥ 40 % of the gap recovered opens lever 1 for a kit ABBA; < 20 % closes it). A fourth field `cold_thp=` on a
2-MB-aligned anon arena with `madvise(MADV_HUGEPAGE)` + `AnonHugePages` from `smaps_rollup` tells whether anon THP exists on this kernel (diagnostic only: the
served file is mmap'd, so THP is not a lever for the frozen file — but it is the generalisation datum for ≥ 12 GB phones with anon expert copies).

**What I would pre-register for round 13: the bench first (same APK rebuild as the knob), then lever 1 as the one ABBA knob**, gated by the bench result read
before the ABBA is scored (the kit runs both; the pre-registration says the ABBA counts only if `cold/hot ≥ 1.5`). Lever 2 waits: it is a day of Tom's and
0.6 ms. Steady arithmetic if lever 1 lands at its central: R31 clean steady 52.4-53.0 − kernel (−2.6..−3.6 *estimated*, device-unshown) − 3.0 → **46-47.5
ms** → printed fault-free (1.13 outside the graph) **20.6-21.2** (*estimated*). **Not pre-registered as reached**: two device-unshown terms are stacked.

## B. A block-gated ABBA scorability clause for round 13

**Why block gating, sized from r12 (device).** Regressing the 46 kern-ABBA blocks of r12 (2,112 on every block) on their own majflt/token: token
**+0.38 ± 0.14 ms per major fault per token**, `MUL_MAT_ID` (compute + wait) **+0.22 ± 0.10**, `MUL_MAT` +0.11 ± 0.03, head +0.041 ± 0.005, `SSM_SCAN`
+0.005 ± 0.003 (r² 0.07-0.56); the 12 always-on windows give 0.92 ± 0.15 with window 1 (warm-up) in, 0.36 ± 0.40 without; 58 pooled **0.49 ± 0.11**.
**Cost per major fault ≈ 0.35-0.5 ms/token at 2,112 (*estimated*).** Side B drew +15.6 majflt/token in the R38 pool; × 0.38 = +5.9 on the token (observed
+5.63), × 0.22 = +3.4 on `MUL_MAT_ID` (observed +1.84). The fault bias is the size of the effect being measured. **Position:** across the five kern ABBAs
of r10-r12, the first block of each ABBA group drew 74 majflt/token against 105 / 92 / 92 for positions 2-4 — side A owns position 1, so **A is favoured
by the kit's group start** (B/A 1.18-1.23 in every pool). That is a kit artefact, fixed in §C by alternating ABBA / BAAB.

**Choosing T.** r8 (clean day) ABBA blocks: medians 1.5 / 1.9 / 8.1, q75 2.8-11.9, max 10.8-18.6; r12 run 1: q25 5.2, median 9.1, q75 16.5; r12 run 2: min
10.8; r10 / r11: min 36 / 90. T = 5 leaves r8's third ABBA with 9 of 23 blocks and r12 run 1 with 4; T = 10 keeps 21-22 of 23 on r8 and 13 of 23 on r12
run 1 and is the convention R33/R35 already used. At T = 10 with the side-balance clause below the residual bias is ≤ 3 × 0.22 = **0.7 ms** on the
`MUL_MAT_ID` line, under the 1.0-ms pass-to-falsify gap proposed for lever 1.

**Clause (paste-ready).**
> *Block gate.* The ABBA knob `K` is scored on its pre-named line over **counted blocks only**. A block counts iff (a) its `majflt` per token ≤ **10**, (b) its
> `mhz_blocks` entry equals the modal clock of the run (2,112 / 2,035 expected; blocks at any other clock do not count), (c) it is not `blk=0`. A side needs
> **≥ 8 counted blocks**; the counted sides' mean majflt/token must differ by **≤ 3**; the counted Δ must have **σ ≤ 1.0 ms** (else the line is *unresolved*,
> not scored). The kit runs the ABBA **three times per session (ABBA, BAAB, ABBA; 69 blocks)**; all counted blocks of the session pool. If after one session
> any condition fails, **one** re-run session is allowed and its counted blocks pool with the first; if it still fails, the line is **parked** (not passed,
> not falsified; re-proposed only with a new kit, never scored "whatever σ"). Untouched lines (`MUL_MAT`, head, `SSM_SCAN`) over the same counted blocks must
> sit at 0 ± 0.3; a move beyond it is reported as a bias flag, and the knob's line is then scored only if |Δ| > 2 × the untouched move. Secondary,
> pre-registered, not a score: ANCOVA over all same-clock blocks (Δ = side coefficient with majflt/token as covariate), reported with its σ.
>
> *Legitimacy test.* The block gate applies only to **class K knobs**: both sides print identical greedy text and identical per-block `kavg`, `adv`, `skip`,
> `head2`, `expq4`, `anon_mb`, `nodes`, `barriers`, **and** a host fresh-process pages-touched test passes — run each side once in a new process on the same
> 200-token story with `GGML_CPU_PROF=1`, compare the process's total minor faults (`getrusage ru_minflt`, = pages first-touched); equal within 2 % → class K.
> Knobs that fail either part (skip thresholds, `EXPERT_IO`, anon / LRU caches, head2 sizes, expq4) are **class F**: for them, gating on faults conditions on
> the outcome and is illegitimate; the R33/R35 side-level rule (both sides' mean ≤ 10 majflt/token, fault ratio reported) stays.

**Worked example on r12 run 1 (illustrative, not a score; the pre-registered rule for that run was the pooled one).** Counted blocks at T = 10: A {1, 8, 9, 12, 13,
16, 17} (7), B {6, 7, 10, 11, 14, 15} (6); majflt 5.4 / 6.4 (balance ✓, |Δ| = 1.0 ≤ 3); clock ✓. Parser `MUL_MAT_ID` **−6.27 ± 3.10**, `MUL_MAT` −0.83 ± 0.91,
head −0.01 ± 0.10, `SSM_SCAN` −0.06 ± 0.11. **Under the clause this would be unresolved on two counts: 7 / 6 < 8 per side and σ 3.1 > 1.0.** The estimate is
also fragile at this n: T = 8 gives −1.53 ± 1.61 on 5 / 4 blocks, T = 12 gives −7.56 ± 3.07 on 8 / 6, T = 15 gives −3.63 ± 4.45 on 9 / 7 — one block (B 15 at
15.7 ms) swings it by 3 ms. That fragility is the argument for ≥ 8 per side and three ABBA runs, not for a softer gate. The clause would have made r10 and r11
unresolved at zero counted blocks (min 36 / 90 majflt/token) rather than letting them into a pool.

## C. Kit fixes to request from Tom (via Clyde)

1. **Pin `cpu-bench` to an A78** (`sched_setaffinity` to {4, 5, 6} for the bench thread, restore after; if the pin fails, print `cpu=-1 pinned=0` and the parser
   marks the line "unpinned"). r12's bench ran on **cpu 2, an A55, 1,804 → 1,094 MHz during the bench** — absolute times are unusable (0.58 × the A78's rate and a
   falling clock); yet new/old **0.780 / 0.723 vs the A78's 0.742 / 0.724**. Implication: the kernel's gain is issue-count-shaped and survives a 2 × 64-bit in-order
   core — a **generalisation datum** for other devices, and no revival of the A55 helper (ns/row 1.7 × the A78's).
2. **Bench fields** `hot= cold= cold_touch= [cold_thp=]` per shape as in §A.1, with `exact_*` for each; the parser prints `cold/hot` and `(cold − touch)/(cold − hot)`.
3. **ABBA order alternation**: the kit runs the knob's ABBA three times as **ABBA, BAAB, ABBA**; the per-block line already carries `ab=`, so the parser needs
   nothing new. Cancels the position-1 fault advantage (§B).
4. **Drop from the kit**: the skip control (nine confirmations), the vanilla official 32, one baseline, and both always-on stories (one-order, closed as a scoring
   method). Keep: baseline ×1, bench, ABBA ×3 (≈ 3 × 80 s), official 32 default ×2. ≈ 10-11 min *estimated* → a second session fits the same afternoon.
5. **Parser**: implement the §B gate as `--gate T` (counted blocks, side balance, σ, untouched-line flags, ANCOVA secondary) so the score is mechanical.
6. **Warm-up**: keep `blk=0` at 16 tokens; do not add page-cache warming (it would change the fault count on the run and is a class-F action).

## What Clyde should pre-register for round 13 (10 lines)

1. Build: `GGML_MMID_TOUCH` (env, default off; demand-load page heads at chunk claim; bit-identical: random row-quads 0 differ, PPL chunks 1-3 equal to all digits, greedy 32 identical) + bench fields `cold`/`cold_touch`/`cold_thp` + kit order ABBA/BAAB/ABBA + parser `--gate 10`.
2. Bench predictions (A78 pinned, 2,112, K 1,536, ns/row-block): hot 14.5 ± 0.5; **cold 28-36**; **cold_touch 19-26**. Gate: ABBA scored only if cold/hot ≥ 1.5.
3. ABBA `MMID_TOUCH:0:1`, line = parser `MUL_MAT_ID` (compute + wait) over counted blocks: **band −1.5..−4.5, central −3.0; pass ≤ −1.5; falsified > −0.5**.
4. Untouched lines over counted blocks: `MUL_MAT`, head, `SSM_SCAN` each 0 ± 0.3.
5. Scorability: §B clause verbatim (T = 10, ≥ 8 counted blocks/side, same clock, side means within 3 majflt/token, σ ≤ 1.0; one re-run session; else parked).
6. Class test for `MMID_TOUCH`: identical text and fields on both sides; host fresh-process `ru_minflt` equal within 2 % — recorded before the phone run.
7. Bench falsifier for §A: cold/hot < 1.2 → the non-kernel half is not the page stream; lever 1 dead, §A's model withdrawn; levers 2-3 remain.
8. If cold_touch recovers < 20 % of (cold − hot) while cold/hot ≥ 1.5: the stall is the stream but not TLB-shaped; lever 1 dead; next audit = per-page prefetcher ramp (hardware), outside the runtime.
9. Printed: official 32 default ×2 reported with clock and majflt, **not a scored line**; the steady arithmetic 46-47.5 ms / 20.6-21.2 printed is *estimated* and stacks a device-unshown kernel term — not pre-registered as reached.
10. Session plan: one session; second session only if §B says unresolved; after that the line is parked.
