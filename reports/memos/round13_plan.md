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

## B.1 Ruling on the class-K test for `MMID_TOUCH` (Bonnie, same day, after Tom's host numbers)

Tom (host, M4, 16-KB pages, fresh process, 200-token story, `ru_minflt`): 3 threads off 205,434 vs on 212,643 = **+3.5 %**; 2 threads **−4.7 %**; 6 threads
**+2.0 %**; spread 0.3 %; load + prompt ≈ 116,800 so decode-only moves ≈ ±8 %. **The sign flips with thread count.** A changed page *set* is a property of the
chunk's rows and cannot depend on who reads them; a fault-event count can — when the touch's 5-6 page-head loads are in flight together on several threads,
neighbouring faults race the fault-around that would have mapped them from one event, so the same pages produce more (or, with another interleaving, fewer)
*events*. My §B instrument was wrong: `ru_minflt` counts fault events, not pages touched. Nothing in Tom's data says the page set moved; by code reading it is
the chunk's own rows on both sides.

**Ruling (1): `MMID_TOUCH` is class K for the majflt ≤ 10 block gate.** The gate's quantity is the major fault = a page-cache miss; residency is decided by
reclaim and by which pages are first read in which order at ms scale, and the touch reads the same pages 0.1-10 µs before the kernel would, inside the same
chunk, so it cannot change whether a page is resident. The minor-fault movement is part of the lever's real cost (≈ 1-2 µs per extra event, in `MUL_MAT_ID`
honestly) and is reported, not gated. Clyde's argument stands; the ±8 % is a reason to *measure* the class on the device, not to drop the gate.

**(2) Pre-registered checks that decide the class on data (paste-ready):**
> *Class check for `MMID_TOUCH` (device, over ALL same-clock blocks of the three alternated ABBA runs, not only counted ones).* (a) majflt/token side ratio
> B/A within **0.80-1.25** → the knob does not move major faults; outside it → **disqualifier**: the block gate is not applied and the knob falls to the
> side-level rule (both sides' mean ≤ 10, fault ratio reported) for this round. (b) `adv` equal on both sides to the unit (readahead path undisturbed);
> unequal → disqualifier. (c) minflt/token side ratio B/A **reported next to Δ**: ≤ 1.30 = expected fault-splitting, a finding, not a flag; > 1.30 = a
> finding flagged "touch splits fault-around" for Tom (the lever is unchanged this round). (d) Host, before the phone run: the fresh-process `ru_minflt`
> test **at 1 thread** (faults serialise; no race) must agree within **0.5 %** — agreement confirms the timing explanation on data; disagreement means the
> page set differs, my reading of the code is wrong, and the knob is class F before the round starts.

**(3)** Not "no": falling to the side-level rule now would discard the gate on the strength of an instrument that measures the wrong thing. If (d) fails on the
host or (a)/(b) fail on the device, the fall-back is automatic by the clause above. Thresholds and the lever are unchanged.

## D. r13 read (Bonnie, 2026-11-07 evening; `reports/device/r13_2026-11-07.json`; all numbers *device* unless labelled)

**(i) The bench result is a verdict on the bench, not on §A.** Two instrument facts: (1) the r13 line `old=201.57 new=157.01 q4x4=78.68` equals r12's **A55** line
(201.57 / 157.22 / 78.68) to 4-5 significant digits while `cpu=5 pinned=1 mhz=2112/2112` is printed — five numbers do not coincide across a different core at a
different clock; the thread ran where r12's ran (an A55 at 1,804) and `cpu=` was read before a migration, or the pin landed on another thread. 1.7-1.8 × R37's
verified A78 line is exactly the A55/A78 ratio r12 showed. (2) **K 512: cold/hot 1.012.** Same bytes (338 KB), same 4-KB page count (83) as K 1,536, so under *any*
memory model cold ≥ hot by the same absolute amount; cold ≈ hot says the "cold" slice was not cold — best-of-7 trials on a reused slice picks the hot trial (64 MB /
8 MB stride = 8 fresh placements < 21 trials). Consequently the K 1,536 **1.248 is a lower bound** (cold − hot ≥ 38.9 ns/row = 6.5 ns/row-block = **≥ 20 µs/slot**
absolute at 3,072 row-blocks, against the model's 25-41) and the 46 % recovery is read as "≥ 46 % of a partially-cold gap" — inside my 50-80 % band's floor,
unverified. `cold_thp` with `anon_huge_kb=0` is the same memory as `cold` (187.6 vs 195.9 = the trial noise, 4 %); it says anon THP is off on this kernel (fact).
Model values (*estimated*): a verified A78 at 2,112 should show cold/hot **1.55-1.9** (hot 14.5 + page term 8-13 ns/row-block); an A55 at 1,804 (hot 26.2
ns/row-block, same ns of latency, less OoO overlap) **1.3-1.5**; a core time-shared with a spinning worker scales hot and cold together → ratio unchanged but
absolutes ≈ 2 ×. r13's 1.248 on the A55 with a partly-warm slice sits under the A55 band, consistent with the slice-reuse reading. The §A gate (≥ 1.5) is **not
evaluated**, not failed; §A is not withdrawn and not confirmed.

**Bench fixes before any ABBA counts (Tom):** (a) pause the threadpool during the bench and pin to an A78 the workers do not hold; read `sched_getcpu()` *after each
trial* and discard trials off the pinned core, print `cpu_ok=n/21`; (b) **one fresh slice per trial, no best-of**: write a 16-MB stream between trials, report the
median of 7 cold slices and `hot` = the same slice re-run immediately; (c) drop `cold_thp` unless `anon_huge_kb > 0`; (d) built-in falsifier: K 512 cold/hot < 1.05
with fresh slices = the instrument is wrong, line marked invalid. A bench-only kit (baseline + bench, ≈ 2 min of the owner's time) is a kit check, not an ABBA session.

**(ii) ABBA: unresolved under §B — zero counted blocks (every block 11-124 majflt/token; file cache 42-293 MB before runs, swap → 290 MB).** The kit changes worked:
BAAB cancelled the position term (adv B/A 1.005; per run 1.017 / 0.984; majflt B/A 0.929), the class checks (a)(b)(c) all pass, untouched lines sit at zero
(`MUL_MAT` −0.39 ± 0.84, head −0.12 ± 0.24, `SSM_SCAN` +0.11 ± 0.12). The ungated pooled `MUL_MAT_ID` −0.07 ± 3.28 (34 A / 35 B) is a day number, not evidence either
way (σ 3.3 on a −3 central). **Recommendation: do not spend the one re-run session yet.** Order: Tom fixes the bench → bench-only kit → if a verified A78 shows
cold/hot ≥ 1.5 **and** K 512 ≥ 1.3, the re-run session is the ABBA ×3 (same clause, T = 10); if cold/hot < 1.2 on a verified core, **park lever 1** and withdraw
§A's page-latency reading (then the non-kernel half is in-core-coupled and the next audit is the chunk loop, not memory). A re-run on a thrashing day cannot count
and would burn the session.

**(iii) Revision to §A — none from r13 (instrument), one hedge added.** Old numbers kept: page term 25-41 µs/slot, cold/hot 1.55-1.9 at the A78. Hedge: if a verified
bench shows K 512 cold ≈ hot *with* K 1,536 cold/hot > 1.2, the term is **stream-shape, not page**: the x4 kernel at K 512 sweeps an 880-B region (≈ 1 stream) while at
K 1,536 it interleaves four 660-B streams over 2,640 B (4 trained streams, 10 lines each) — a prefetcher-ramp term that a page-head touch only partly fixes and a
row-group reorder (gate/up rows visited so the four streams become one) would fix — noted, not proposed. The 46 % recovery, if it survives a verified bench, favours
the page reading over the stream-shape one.

## E. r13b read: the bench is verified, §A was built on an X1 number (Bonnie, 2026-11-07 night; `reports/device/r13b_2026-11-07.json`; *device* unless labelled)

**(1) The X1 reading is the one that survives; the "A55" story in R40/§D was wrong (mine too).** Verified line (cpu 6-6, migr 0, cpu_ok 21/21, 2,112/2,112):
old 201.52 / new 156.86 / int8 100.30 / q4x4 78.58 / scalar 131.38 ns/row at K 1,536 — equal to r12 and r13 to three digits, so **r12, r13, r13b all ran on A78s**
and R37's r11 line (117.49 / 87.13, `cpu=4` read once) is the outlier. Clyde's arithmetic checks: 201.52 / 117.49 = **1.715** = (2,841 / 2,112 = 1.345) × **1.275**.
The decisive test is that the r13b / r11 ratio is **kernel-dependent**: old 1.715, new 1.800, q4x4 1.818, scalar 1.624, int8 1.599 (K 512: 1.737 / 1.838 / 1.811 /
1.612). A clock difference or a time-shared core gives one factor for every kernel; a different microarchitecture does not — the issue-bound kernels (q4x4, new)
gain most (4 NEON pipes), the load-bound ones (int8, scalar) least. That is the X1 (cpu 7) at 2,841 with 1.19-1.35 × the A78's IPC. No other reading fits all nine
ratios. Cross-check with the count model: old = 201.52 / 6 = **33.6 ns = 70.9 cycles per row-block at 2,112 — exactly the 2-pipe NEON issue floor (70-71.5)** from
`expert_kernel.md`; new = 26.1 ns = **55.2 cycles** vs its 377 / 4 / 2 = 47 floor (85 %), and equal to the "ideal in-place unpack floor 54-55 cycles" of §2 of that
memo. The kernel sits on the floor cache-hot; r10/r11's 0.724-0.742 ratios were X1 ratios and still agree (0.778 / 0.766 on the A78).

**(2) §A is withdrawn — by the slot share, not by the flag.** In situ 37.8-39 ns/row-block vs cache-hot 33.6 → the old kernel is **86-89 % of the slot**; non-kernel
**4.2-5.4 ns/row-block = 13-17 µs/slot = 2.1-2.8 ms/token** at 164 slots (*estimated*), not 50-60 µs/slot: the "non-kernel half" was an X1 bench divided into an A78
slot. The page-latency model (25-41 µs/slot) cannot fit in 13-17 µs and is withdrawn; R27's 16-33 % clock-invariant share is consistent with the new 11-15 % at its low
edge. cold/hot 1.043 and recovery 0.36 on a verified core bound the touch lever at 0.36 × 7.3 ns/row = 0.44 ns/row-block = **≤ 1.4 µs/slot = ≤ −0.2 ms/token** —
not worth an ABBA. **The K 512 flag:** its premise ("cold ≥ hot under any memory model") was mine and is wrong for a kernel at its issue floor streaming 2.9 GB/s per
core behind a hardware prefetcher — cold ≈ hot (0.989, 1 % noise) is the expected measurement, not an invalid one. Ruling: the flag fired on a false premise; the
instrument is valid; the decision rule "< 1.2 on a verified A78 → park lever 1, withdraw the §A page reading" governs and is met on both shapes. **I agree with
Clyde's call on all three points** (§A withdrawn, `MMID_TOUCH` parked, the ABBA session not spent). One note, not a dissent: the flag did its job — it forced the
premise check that found the X1 — so flag-style falsifiers stay in the kit. Still standing from this memo: the §B block gate and class test, §B.1, and the
ABBA/BAAB alternation (r13: adv 1.005, majflt B/A 0.93 — the position term is gone).

**(3) What is left with a compute-bound expert op at its issue floor (all 2,112, *estimated* from verified bench numbers; R38 stands as scored).**
| rank | lever | arithmetic | ms/token | status |
|---|---|---|---|---|
| — | new kernel (compiled default, exact) | 33.6 → 26.1 ns/row-block = −7.45 × 3,072 × 164 | **−3.75** (−3.5..−4.1 by k̄) | device-unshown as scored (R38); the clean subset −6.27 ± 3.10 is 1σ-consistent with it |
| 1 | `MMID_FUSE` (gate+up+GLU one pass) + two chunks per `fetch_add` | 40 barriers + 40 ops' fixed costs + half the atomics, out of the 2.1-2.8 ms non-kernel | **−0.6..−1.3**, exact | Tom > 1 day; the last exact crumb of the expert op |
| 2 | kernel | at 85 % of its own floor and on the in-place-unpack floor | ≤ −0.5 | closed as a line: 3-bit extraction costs the issue slots |
| 3 | exact 4-bit container (q4x4 13.1 ns/row-block = 40 µs compute under a 59 µs byte floor) | slot 59-65 vs 80 + 15 → −30..−36 µs × 164 | **−4.9..−5.9**, exact | needs the anon copy: 3.45 GB all, 86 MB/layer; ≥ 8 layers = R24's 618-MB page-cache loss → negative on 8 GB; **the generalisation lever for ≥ 12 GB phones**, not this one |
| 4 | int8 reference (16.7 ns/row-block = 0.50 old / 0.64 new) | 2.7 MB/slot → 118 µs byte floor | 0 | diagnostic only: unpack is 36 % of the new kernel |
| 5 | fewer MACs: skip | capped at 118 (+2 % final) | 0 | closed |
| 6 | dense `MUL_MAT` 22-24 ms = 525 MB at the byte rate | bytes only fall by (a) a byte-rate container below 4 bits — none exists on dotprod cores (R29: Q2_K/Q3_K dense screens were issue-bound) — or (b) structural numerics: vocabulary subset of the tied head, channel/neuron pruning of `ssm_in`/`ssm_out` (296 MB = 12.5 ms; 05_compute_less §1135 table has host PPL per zeroed fraction) | −2..−5 for a 10-20 % byte cut | **numerics; outside the +2 % speed-rule budget, inside the +20 % hard limit (served 9.7886 → ≤ 10.853 = +10.9 % headroom)** — the owner's call with host PPL numbers first (`tools/ppl.sh`, an afternoon); Clyde's round-14 dissent, priced |

Where 20 stands on this arithmetic: clean steady 52.4-53.0 (R31, old kernel) − 3.75 (kernel, unshown) = **48.7-49.3 → printed fault-free ≈ 19.9-20.1** at 1.13 ms
outside; + lever 1 → 47.4-48.7 → **20.2-20.7**. Every path to 20 on this phone under the current budget runs through the kernel claim the rules say is unshown, plus
one exact crumb; the only ≥ 3 ms lever left is numerics under the hard limit (rank 6), which is a decision, not a measurement.
