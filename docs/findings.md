# What limits LLM decode on an 8 GB phone, and what helped

A short account of the research in `reports/03_final.md`, in the order things were found.
Device numbers come from the owner's Galaxy Z Flip 3 (Snapdragon 888) through Vollie's in-app
speed test; host numbers are an Apple M4 Pro, CPU only, and only ever used as relative numbers.

## 1. The estimate was wrong: it isn't bandwidth

The plan assumed decode is memory-bandwidth bound: ~0.9 GB read per token at the 13-16 GB/s this
phone reached with a dense 4B model → 10-14 tok/s. The first device run gave **7.4 tok/s**. Half of
the 135 ms per token was not explained by bytes.

## 2. Thread syncs around small ops

The model (Granite-4.0-H-Tiny: 36 Mamba-2 + 4 attention layers, 64 experts) runs ~2,000 compute
ops per token (`llama-eval-callback`), and ggml syncs all threads after every one. On the M4 a sync
costs ~0.4 µs (measured with the profiler: 1,870 syncs ≈ 0.7 ms/token); on the SD888 the phone's
own A/B showed them as roughly half of each token.

Fix (`GGML_TINY_OP`): ops below a size threshold run on one thread, with no sync between
consecutive ones. Syncs per token: 1,909 → 874. Device: **3.2 → 7.0 tok/s** at 2 threads, 3.4 → 5.6
at 4 threads. Output bit-identical.

A race in the first version: the look-ahead treated a following RMS_NORM as small, but it ran
fused with its MUL on all threads, so threads read data before it was written. Decode was safe by
accident (one row), prompt batches under ~42 tokens gave garbage (PPL = vocab size). Caught on the
host with `-ub 32`, fixed by sharing one rule between loop and look-ahead.

## 3. The experts don't fit in the page cache

Per-op profiling on the phone (`GGML_CPU_PROF`) then showed the MoE experts taking 71 of 121 ms per
token, at ~3.5 GB/s against ~19 GB/s for the dense weights. Page-fault counters settled why: **354-
1,181 major faults per token**. The experts (3.1 GB) are the only large weights still read through
mmap, the phone kept ~2.1 GB in its whole page cache, and every token re-read part of them from
flash, one synchronous fault at a time on whichever compute thread touched the page.

A smaller file (3-bit experts, 2.5 GB) did not fix it: on the next run the cache was 0.3 GB smaller
too. Asking the user to free memory isn't a fix either, so the work moved into the runtime:

- **Prefetch the routed experts** (`GGML_EXPERT_IO` bit 1): when a layer's routing is known, request
  exactly the 6 chosen experts' slices of its gate, up and down matrices together. Device: **6.1 →
  11.8 tok/s**, major faults 708 → 253 per token.
- **No read-around** (bit 2): lost on the device (more, smaller faults).
- **Guess the next layer** (bit 4): a helper thread applies layer L+1's router to layer L's input and
  prefetches a layer early. Right for 82-83.5% of experts on host text, **85% on the device**; best
  writing **12.2 tok/s**.

## 4. Now: compute and noise

With the reads hidden, expert time scales with thread count (24 ms at 4 threads, 34 ms at 3): the
3-bit Q3_K kernel is compute-bound on the A78 (on the M4 it is 1.43x slower per weight than Q4_0).
A vectorized NEON Q3_K dot is bit-identical and 4-8% faster on the M4.

Single runs on a warm phone vary about ±20% (the same setting measured 12.0 and 8.4 tok/s a minute
apart), which hides any change smaller than that. Two answers: score each thread count by its best
run, and an **in-generation A/B** (`GGML_CPU_AB`) that alternates a compute setting every token so
heat and clocks fall on both sides alike.

## Things that did not help (measured)

- Speculative decoding (draft model, n-gram, EAGLE-3) and MTP: slower on this MoE (host).
- Adreno 660 OpenCL: 0.7-2.9 tok/s, always below the CPU (device).
- More threads than big cores: an A55 in every sync (device).
- 4 threads for writing once the expert-guess helper runs: it takes a core from the compute threads
  (device: writing best at 3, reading best at 4).

## Where the ceiling was (2026-11-01 view)

Dense weights ~25 ms per token, already near this SoC's bandwidth. 20 tok/s = 50 ms/token needs the
expert compute and the remaining waits down to ~25 ms together. That is the open problem.

## 5. Rounds 3-6 (R21-R27, 2026-11-02 to 2026-11-05): measure the knob, not the day

Everything above was measured with one-order A/Bs and single runs on a phone whose page cache and clock
changed between runs. From round 5 the kit runs every knob as an **ABBA** block inside one generation
(16 warm-up tokens on A, then 8-token blocks A B B A …), so both sides share the same heat, clock and
cache. Three rounds of that settled the numbers:

| knob (bit-identical unless noted) | worth, ms/token, in-run | confirmations |
|---|---|---|
| SSM scan in place (`GGML_SSM_INPLACE`) | −4.0 to −4.6 | r4, r5, r6 |
| conv state ring (`GGML_CONV_RING`) | −0.7 at 2,112 MHz, −1.25 at 1,555 | r5, r6 → default on |
| Q3_K ILP (`GGML_Q3K_ILP`) | −1 to −2 (one-order rounds said −18 %: order bias) | r5 |
| **skip rule at 118** (`GGML_EXPERT_SKIP`, PPL +1.98 %, inside the +2 % budget) | **−8 at 2,112 MHz, −10.5 at 1,555**; 29.8 % of expert slots skipped, k̄ 4.2 | r3, r4, r5 ×2, r6 ×2 |

**Vanilla vs Vollie, same app, same file, same day (device).** Stock llama.cpp kernels and settings vs the exact
stack vs the stack plus skip 118: official 32-token benchmark **9.4 → 12.0 → 13.9** tok/s; 200-token story printed
**10.2 → 14.5 → 12.5**, steady **82-97 → 62-66 → 55.4** ms/token (2026-11-04). The exact stack is **−27 ms/token
(−30 %)**, almost all of it `MUL_MAT_ID` read waits and small-op barriers (1,926 → 746 per token); it never touched
the dense products (25.9 → 23.1). The skip rule adds ≈ ×1.12-1.15. An earlier "×1.40 printed" for the skip rule
was a control run with a bad page cache and is withdrawn.

**What "printed" measures on this phone.** Steady at 118 is 55 ms = 18 tok/s, but the 200-token printed rate is set
as much by the page-cache term — a 22-62 ms penalty in the first 32 tokens and 34-365 major faults per token — as by
the steady runtime: runs at 91 / 63 / 55 ms steady printed 10.2 / 14.5 / 12.5. Bytes read per token, not multiply-adds,
is what moves the printed number from here.

**A free clock experiment (round 6).** On 2026-11-05 the phone held its fast cores at 1,555 MHz instead of 2,112.
Same build, same knobs: the expert products went ×1.24 and the skip rule's saving ×1.3 (compute-bound); the dense
`MUL_MAT` went 23.1 → 25.0 ms (×1.08, bandwidth-bound); the SSM scan did not move. Every steady number now carries its
clock. The dense products are the only block large enough for the last 4-6 ms, and only fewer bytes per token can
move them.

**Closed with device numbers:** speculation of every kind (verify of k tokens costs ≈ k tokens: the CPU is
compute-bound at any batch > 1), early exit, 4 threads, an A55 helper, anonymous expert caches, Q3_K prefetch,
thresholds above 118 (the +2 % budget is final), `EXPERT_IO=0` (+0.85 ms, ×9.6 faults), moving `madvise` off the
compute threads (thread 0 was never slower).

## Where the ceiling is (2026-11-05)

Steady at 118: **55 ms = 18 tok/s** at full clock (device), 63.7 ms at 1,555 MHz; printed 12.5-14.5 on a swapping
day. 20 tok/s printed = 48.6 ms steady plus a quiet warm-up. Exact crumbs left ≈ 1-1.5 ms (barrier waits). The rest
has to come from the dense `MUL_MAT` reading fewer bytes per token inside the frozen file — the open problem of round 7
(`reports/04_campaign.md` §13). First byte-cutting lever found inside the rules (R28, host): a two-stage lm_head, Q2_K screen plus the top-8,192 exact rows, 9.6237 (+0.19 %), argmax-identical, estimated −2.0 to −2.5 ms/token; it does not fit beside skip 118 under the +2 % budget, so the next step is the owner's.
