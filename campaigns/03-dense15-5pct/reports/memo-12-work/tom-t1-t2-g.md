# c3 memo-12 work: Ebony r18 T1 and T2, Bonnie r34 §9 (chunk multiplier), and a sketch of design G

- **Date:** 2026-11-25. **Author:** tom, `claude-opus-5-5`. Not committed. No shipped patch is changed.
- **Brief:** [ebony-joins-r18.md](ebony-joins-r18.md) §3.3 (G), §4 (T1, T2) and §5 (L1, L2, L4), with the addition from
  [bonnie-joins-r34.md](bonnie-joins-r34.md) §9: the stall on thread 0 after the KleidiAI Q4_0 barrier, at
  multipliers 4 and 16, and an identity check of multiplier 16.
- **Labels.** Every timing here is *host*: arm64 Linux in Docker on the owner's M4 Pro, 4 threads pinned to the
  container's cpus 0 to 3, or PoCL. Timings are relative only. The perplexity values are *host*. The arithmetic is
  *estimated*. Nothing ran on the phone.

## 1. Multiplier 16 is bit-identical (*host*, Yi-1.5-6B served Q4_0, 40 chunks)

The build is `lab/llama.cpp-c3/build-kai` (KleidiAI, the stock-shift build of [tom-kai-stock.md](tom-kai-stock.md)),
run with `GGML_KLEIDIAI_SME=0`, `-c 512 -t 8 --chunks 40`, on `lab/c3/text/wiki.test.raw`. The multiplier is read from
`GGML_KLEIDIAI_CHUNK_MULTIPLIER`; unset means 4.

| setting | multiplier 4 | multiplier 16 | per-chunk values, all 40 |
|---|---|---|---|
| `-b 512` | 7.7347 (`ppl-c3k40-yi15-6b-q4_0-b512-wiki.log`, 2026-11-24) | 7.7347 | **identical** |
| `-b 512 -ub 32` | 7.7319 (new run) | 7.7319 | **identical** |
| `-ub 1` | 7.7165 (`ppl-c3k40-yi15-6b-q4_0-ub1-wiki.log`) | 7.7165 | **identical** |

New logs: `lab/logs/p/ppl-c3km16-yi15-6b-q4_0-{b512,ub32,ub1}-wiki.log` and `ppl-c3km4-yi15-6b-q4_0-ub32-wiki.log`.
This is expected from the source: the multiplier moves the column ranges of the chunks, not the arithmetic of a
column. It was checked on one model, as briefed. Limit: the host's KleidiAI kernel choice for batched rows
(`I8MM` on the M4) is not the phone's. Identity across multipliers within one kernel is what was shown.

## 2. T1, pool stall (`tom_t1.cpp`)

**Build.** A scratch copy of `lab/llama.cpp-c3r36` (the phone's full series), compiled for arm64 Linux with KleidiAI
(`armv8.2-a+dotprod+fp16`, OpenMP off). It carries a test-only hack, never shipped: in `compute_forward_qx`, right
after the barrier commented "Publishes both LHS packing and the initialized dynamic chunk queue", thread 0 busy-waits
`T1_STALL_US` µs. The harness runs ggml's CPU backend with `VOLLIE_POOL=1` (the app's persistent pool) on 4 threads.

**Cells.**
- **Cold:** 22016 × 4096 Q4_0 in the `CPU_KLEIDIAI` buffer, 8 copies (406 MB ring), one gemv per graph, 3 s per arm.
- **Arms:** `none`; `pre` (a 200 µs busy stall on the calling thread before each `graph_compute`, outside the timed
  part: Ebony's T1); `in` (the 200 µs stall inside the op: Bonnie's addition).
- **Order:** each arm at multipliers 4 and 16, ABBA over two passes. Output hash equal in every arm.

| multiplier | none, p50 µs | `pre` 200 µs | excess | `in` 200 µs | excess |
|---|---|---|---|---|---|
| 4 | 361.8 | 369.6 | +7.8 | 369.0 | **+7.2** |
| 16 | 361.1 | 370.1 | +9.0 | 368.5 | **+7.4** |

- **Voluntary switches:** about 1.0 per graph on each of the three workers, in every arm including `none`. The
  calling thread has 0. The persistent pool runs at poll 0, so the workers sleep between every two graphs whether
  the lead stalls or not.
- **Ebony's T1 reading.** The 200 µs gap before a graph costs +8 to +9 µs here, which is at her 10 µs line. The "no
  sleeps" condition fails, and it fails the same way with no stall at all. The workers park between graphs by
  design. So the question for the phone is the wake cost after each graph, not a spin budget that runs out inside
  200 µs. A within-graph hand-over (G's L1, launch inside the op) does not park: `ggml_barrier` spins.
- **Bonnie's rule as written:** "within 60 µs of no-stall at 16" passes (+7.4). "About 200 µs late at 4" fails
  (+7.2). Her line "if 16 is not better than 4 on the host, my chunk reading is wrong" is literally triggered. But
  the cause is the host, not her reading. This cell is bandwidth-bound, with headroom on the M4. While thread 0
  stalls, the other three run their chunks faster, so the lost thread costs almost nothing at either multiplier.
  (A 5,000 µs stall gives 5,082 µs: the other three finish the whole op, then thread 0 runs its one reserved chunk.)

**Hot cell, to separate the chunk logic from the bus.** 4096 × 4096, one copy (cache-resident, graph 63 µs). The
in-op stall is 12 µs, 0.19 of the graph, near the phone case's 214 ÷ 1,269 = 0.17. ABBA, 4 runs per cell:

| multiplier | none, p50 µs | `in` 12 µs | excess |
|---|---|---|---|
| 4 | 62.4 | 73.9 | **+11.5** (the whole stall) |
| 16 | 63.0 | 66.0 | **+3.0** (a quarter of it) |

When a chunk takes the same time whether 3 or 4 threads run, **Bonnie's reading holds**:
- At 4, the late thread still ends one stall late.
- At 16, the loss is about stall ÷ threads.
- A simulation of the source's claim rule (each thread takes its reserved chunk i, then claims the next free one;
  equal chunk times) gives her table's numbers for the phone case: 1,269 µs of rows, lead 214 µs late, excess 214 /
  214 / 159 / 79 µs at multipliers 4 / 8 / 16 / 32 (*estimated*).
- Scaling the hot cell's ratio instead gives about +53 µs at 16 (*estimated*).

Which of the two the phone is in, bandwidth-bound like the cold host cell or equal-chunk like the hot one, is a phone
question. Campaign 2's thread-scaling lines put four A78s near the bus, so the truth is probably between the two.
That is a reading, not a measurement.

## 3. T2, done flag on ordinary kernels (`tom_t2.cpp`, PoCL 5.0, `vollie-ocl` image, *host*)

**Setup:**
- Upstream's general q4_0 gemv (as in [tom-mc3-check.md](tom-mc3-check.md) §4, the fp16 host shim), writing into a
  fine-grain SVM buffer (PoCL offers SVM atomics).
- Behind it, in the same in-order queue and the same flush, a one-line kernel
  `atomic_store_explicit(f, seq, memory_order_release, memory_scope_all_svm_devices)` (built `-cl-std=CL3.0`).
- The host polls the word with an acquire load. No event, no `clFinish` in the wait.

| cell | cycles | flag seen | rows not final when the flag is seen | rows differing from the gemv alone after `clFinish` |
|---|---|---|---|---|
| 11008 × 4096 | 100 | 100 | **0** | **0** |
| 4096 × 4096 | 200 | 200 | **0** | **0** |
| control: flag kernel enqueued **before** the gemv | 50 | 50 | 204,800 (every row, every cycle) | 0 |

- Identity holds. The output with the trailing kernel is bit-identical to the gemv alone, as it must be, since the
  gemv's kernel and arguments are unchanged.
- Order and visibility hold on PoCL. The control shows the check can fail.
- Flush-to-seen times (PoCL p50 2.7 to 7.7 ms) say nothing about Adreno. The device question stays Ebony's: flag
  seen within 20 µs of the END stamp.
- **Limit, a spec point for the device test.** The release store orders the flag kernel's own writes. That the
  gemv's plain writes, from an earlier kernel in an in-order queue, are visible to the host once the flag is seen is
  what PoCL does. I did not find that the OpenCL 2.0 memory model guarantees it without an event or a synchronisation
  point. On Adreno it must be checked the way this file does (rows final at the flag), not assumed. If it fails, the
  gemv itself can end with a release fence on the same scope (one line in a copy of the kernel; the arithmetic is
  unchanged).

## 4. Design G, a sketch (not built)

What G is (Ebony §3.3): L1, the launch inside the op, plus L2, a trailing done flag. Where it goes in our tree, behind
one knob, `VOLLIE_LANE_G=1`, unset = today's code path untouched:

1. **Hook (L1).** In `kleidiai.cpp` `compute_forward_qx`, after the "Publishes both LHS packing…" barrier (the line
   T1 stalls on):
   - If the op's weight carries a lane tag (set at load for lane stages), thread 0 enqueues the GPU part: upstream's
     gemv on the GPU's rows, then the one-line flag kernel with sequence number s, then `clFlush`. Then it joins the
     claim loop.
   - The claim loop runs over the CPU's columns only: `n_cpu` in place of `n` in the loop bound. Nothing else in the
     loop changes.
   - Placed after the barrier, the other three threads are not held (Bonnie §9). About 30 lines, *estimated*.
2. **Chunking.** Lane stages run at multiplier 16. Its identity is §1. It would be set per op, not process-wide, so
   that stock stages keep 4 and their outputs are unchanged anyway.
3. **Join (L2).** After the claim loop, thread 0 polls the SVM word (acquire) until it reads s, with a 2 s deadline
   as in r36's `spin_ev` that falls back to `clFinish`. Then it copies the GPU's rows from the SVM buffer into `dst`.
   That copy is the `write` hand-off, 13.3 µs on the phone at share 0.30, *device*, R75 H1. The other threads wait at
   the next node's `ggml_barrier`, which spins. No `clGetEventInfo`, no driver thread.
4. **Numerics.** G is bit-identical **to the lane with `clFinish`**, not to stock. The lane itself is a numerics
   change (GPU rows in the upstream kernel's arithmetic). It is already declared as one and needs the budget, per the
   R74 next action, the half-precision `(q − 8) × d` host check. G adds no further change.
   - Checks: per-row equality with the `clFinish` lane on 11008 × 4096 and 22016 × 4096 (as T2 did).
   - With the knob unset: per-chunk perplexity at `-ub 1`, `-ub 32` and `-b 512` equal to the runtime without G.
5. **Host test before any device cell:** T1's harness plus T2's OpenCL side, in one process on PoCL. Rows final at
   the join, G against the `clFinish` lane bit for bit, deadline path exercised by a shim that drops the flag kernel.
   About a day, *estimated*.
6. **What only the phone answers:** T2's 20 µs line, T3 (END to seen), and whether the phone's chunks behave like the
   hot or the cold cell of §2.

## 5. Not done

- **F sketch:** not started. The lead put it after G, and Ebony orders it after the device test T9.
- **Ebony's L4 (spin budget):** the pool parks between graphs at poll 0 in every arm (§2). Whether a spin across
  graphs is worth a line depends on the phone's wake cost. The host does not give it.

## 6. Files

- `campaigns/03-dense15-5pct/reports/memo-12-work/tom_t1.cpp`, `tom_t2.cpp` (new, host harnesses).
- The stall hack lives only in the scratch tree (session scratchpad). It is a three-line block after the barrier
  named in §2, and it is never to be shipped.
- Logs: `lab/logs/p/ppl-c3km{4,16}-yi15-6b-q4_0-*-wiki.log`.
