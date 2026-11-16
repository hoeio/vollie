# Memo 24: the app's anonymous memory through a wait, and the "rest" of a token, on both models

Agent model: claude-opus-5-5

Date 2026-11-16. Author: Bonnie (research). **Nothing here ran on a phone, and no build or benchmark ran on the host**
(a perplexity queue was using the laptop's CPU). This memo is reading, short scripts over the two raw device reports,
web sources and arithmetic. It proposes levers and experiments; verdicts are Clyde's.

Labels: *device* = read or computed from `device/r22_2026-11-15.json` (R60) or `device/r23_2026-11-16.json` (R62), or
from campaign 1's `reports/device/r15_2026-11-08.json` where named; *host* = R61's load logs (`lab/logs/r61/*.load.log`,
Apple M4 Pro, KleidiAI build) or GGUF headers read on the laptop; *estimated* = arithmetic or a regression over *device*
numbers (in-sample, no standard errors unless given); "source: <url>" = a published claim not reproduced here. "Gated
run" = a run after a 90 s gate wait; "warm-up" = no wait.

## 0. Summary

1. **The anon account closes (table §A1).** RssAnon + VmSwap is constant per model and arm within a session
   (*device*): Granite off 785 to 792 MB, num 871 to 877; OLMoE off 655 (R62) / 671 (R60), num 710 (R62). Inside it:
   KleidiAI weights, a head copy, the KV cache, Granite's recurrent state, and a remainder of 131 to 147 MB (compute
   buffers, Dart/Flutter, libraries; *estimated* as the remainder). Two items were not on anyone's list:
   - **OLMoE's KV cache is 285 MB** (q8_0, `n_ctx` 4096, 16 layers), of which the benchmark uses about 15 MB. **Every
     kit run rewrites all 285 MB** with a `memset` before the prompt: llamadart calls `llama_memory_clear(mem, true)`,
     which calls `ggml_backend_buffer_clear(buf, 0)`.
   - **Granite's 541 MiB KleidiAI buffer includes a 148 MiB int8 copy of the tied head.** In the num arm (the frozen
     Granite setting), head2 never reads that copy after load.
2. **The swap hypothesis survives as a description of the faults, and gains a new main source. Its cost is much
   smaller than memo-23's fit says.**
   - Granite's in-graph prefill majors track the net swap-in, page for page (r = 0.86 to 0.96; slope ≈ 1.0 page per
     major in R62).
   - OLMoE's largest block of majors comes *before* the prompt graph: median 27,800 (R62) and 47,400 (R60) per gated
     run, against 1 to 266 at warm-ups. The four highest runs are 70,316 to 70,727, just above the KV cache's
     69,632 pages. Granite's pre-prompt maximum is 19,688, against 18,530 pages of KV + recurrent state.
   - That time is in TTFT: TTFT minus prefill-graph time = 41.6 ms + 3.47 µs × pre-prompt majors (OLMoE, r = 0.99,
     n = 48; Granite 6.1 µs, r = 0.92).
   - Swap-ins cost microseconds each. At 3 threads, 1,000 decode majors cost **0.12 to 0.20 ms per token**
     (pooled fits, *estimated*), not 2.8. Memo-23's 2.8 is reproduced (2.71) only when the 4-thread runs are pooled
     in; 4-thread runs alone give 1.33.
   - The slow runs are explained by **file pages re-read in the decode** (0.085 to 0.125 ms per token per MB read in
     the run), not by swap-ins.
3. **The swap-in time is mostly in TTFT, which the metric does not count.** Anon matters for tok/s in two ways:
   - its RAM competes with the model's page cache;
   - its swap slots fill zram (SwapFree < 50 MB at 28 of 48 run ends in R62), and with no free slots the kernel can only
     drop file pages.
   OLMoE's 270 MB of zero KV pages count against SwapFree although zram stores them without memory (same-filled pages,
   source: kernel zram doc).
4. **Ranked shortlist, Part A (§A4):**
   1. Clear the KV and recurrent buffers with `MADV_DONTNEED` instead of `memset`. Bit-identical by construction (zeros
      either way), generic.
      - TTFT: OLMoE −0.10 to −0.25 s, Granite −0.02 to −0.12 s.
      - Memory: frees about 270 MB of OLMoE's RAM and swap slots per run.
      - Decode: OLMoE 0 to −2 ms/token through residency (unmeasured), Granite ≈ 0.
   2. Granite num: keep the tied head out of KleidiAI at load: −155 MB of anon. Bit-identical in the num arm by code
      reading (to be checked). Decode 0 to −1 ms.
   3. Anon hold: `madvise(MADV_WILLNEED)` on the hot anon ranges after the gate. Linux 5.4 swaps anon in on this
      advice (source). TTFT −0.1 to −0.5 s, decode ≈ 0.
   4. The plateau gate (R63, already pre-registered).
   5. `mlock` within 64 MiB, or a periodic touch: crumbs.
5. **Part B: the per-token lines cannot give an op-by-op account.**
   - Kits r22/r23 print only `MUL_MAT_ID`, `MUL_MAT`, `GLU` and `SSM_SCAN` (OLMoE: the first two), by design (R46
     amendment 1). Everything else is a residual.
   - Fastest-half fault-free tokens at 3 threads (*device*, additive means):
     - Granite num: 45.5 to 45.7 ms graph + 0.83 to 0.95 ms outside the graph; rest 4.9 to 5.1; waits in the four ops
       2.5 to 2.7.
     - OLMoE num: 40.8 to 41.2 + 0.66 to 0.78; rest 4.1 to 4.8; waits 0.9 to 1.2.
   - Campaign 1's r15 (same Granite architecture and types, full 14-op table, *device*) splits Granite's rest: 3.3 ms of
     small ops and 1.7 to 2.5 ms of per-node dispatch.
   - **Biggest surprises:**
     - Granite's non-head dense ops are 1.1 to 1.5 ms *slower* in the num arm than in the off arm, at equal clocks.
     - `SSM_SCAN` moves about 42 GB/s of state reads plus writes, against the dense GEMV's 24 to 27 GB/s of reads. The
       dense rate may not be the DRAM ceiling.
6. **Combined top 5 (§C):**
   1. KV/RS `DONTNEED` clear.
   2. Block-mode profile plus a read-bandwidth microbench: zero or near-zero code; it opens the dense-rate pool
      (Granite −1.1 to −3.5 ms, OLMoE −0.5 to −1.3 if real).
   3. Granite tied head out of KleidiAI in num.
   4. Per-thread pinning (Granite −0.3 to −1.0, OLMoE −0.1 to −0.5).
   5. Graph crumbs: the unexplained 1.1 ms of `MUL` and the dispatch (Granite −0.5 to −1.5).
   - **Granite still does not reach 25 tok/s on these levers alone.** Fault-free 46.4 ms minus at most about 4.5 ms is
     about 42 ms (≈ 23 tok/s, *estimated*); memo-21's expert-op pool is still needed.
   - OLMoE fault-free (41.5 to 41.9 ms incl. outside) needs −2.7 to −3.2 ms.

## A. The app's anonymous memory

### A1. The account

Device totals are RssAnon + VmSwap of the process (`mem_before` / `mem_after`, `anon=` and `swap=`). They are constant
across a session, so a page moves between RAM and zram and is not freed. The items below are sized from the R61 host
logs and the GGUF headers (*host*), from the app's load parameters and llama.cpp's allocation code (read here), and
from arithmetic (*estimated*).

| item | OLMoE-Mi MB | Granite-Mi MB | source | touched |
|---|---|---|---|---|
| KleidiAI Q4_0 dense weights | 151 (attention q/k/v/o; `CPU_KLEIDIAI` 144.00 MiB) | 413 (`ssm_in` 200.6, `ssm_out` 95.6, shexp 101.8, attention 14.2; ≈ 393.7 MiB) | R61 load logs; GGUF header | every decode token |
| head copy in KleidiAI (int8, one scale per row) | 103.6 in R60 (98.8 MiB = 242.83 − 144.00); 0 since R62 (`VOLLIE_HEAD2_REPACK=1`) | **≈ 155** (≈ 147.9 MiB; 393.7 + 147.9 = 541.6 against the logged 541.08 MiB) | R61 logs; llama.cpp loads a tied head as `TENSOR_DUPLICATED` into the output buffer type (`src/models/granite-hybrid.cpp:65`, `llama-model-loader.cpp:1167`); KleidiAI accepts Q6_K through its q8 kernels (`kleidiai.cpp:1811`); 0006's exclusion tests only the name `output.weight` | off: every token. **Granite num: never after load** (head2 reads the mmapped `token_embd`, `0001` lines 2622-2632) |
| `CPU_REPACK` Q6_K head (8x4) | 84.5 (80.60 MiB), R62 onwards | 0 | R61 rows 2-3 | off: every token; num: about 7,000 of 50,304 rows per token |
| head2 screen copy (anon) | 58 (num) | 86.7 (num, Q4_0x4) | memo-23 (`c2-head-build`); `docs/runtime-knobs.md` | num: every token |
| KV cache, q8_0, `n_ctx` 4096 | **285.2** (16 layers × K+V × 4,096 × 2,176 B); used ≈ 15 MB at about 210 positions | 17.8 (4 attention layers, 512-wide K/V); used ≈ 0.9 | app `assistant.dart:459-471` (q8_0 K/V, `contextSize`); report header "context 4096" | used part: every token. **The whole buffer is `memset` before every prompt** (§A2.2) |
| recurrent state (F32, one sequence) | – | 58.1 (36 × (1.573 MB ssm + 0.04 MB conv)) | arithmetic from the header (state 128, inner 3,072, 48 heads); runtime-knobs "≈ 58 MB per slot" | every token (read and written); `memset` before every prompt |
| compute buffers, Dart/Flutter heap, libraries, tokenizer | ≈ 131 to 134 | ≈ 147 | remainder against the device totals | prefill: wide; decode: a few MB (*estimated*) |
| **device total RssAnon + VmSwap, median** | off 655 (R62), 671 (R60); num 710 (R62), 671 (R60, head declined) | off 785 to 792; num 871 to 877 | *device* | |

Checks that the table is right (*device*):
- **num − off:** +85 MB on Granite (the 86.7 MB Q4_0 copy) and +55 MB on OLMoE in R62 (the 58 MB screen). In R60, where
  OLMoE's head declined and no copy was made, the difference is 0.
- **Remainder:** after the named items it is 131 to 147 MB on both models, which is credible for compute buffers plus
  a Flutter app.
- **The worst-case compute buffer** (for example, the logits for 512 outputs: OLMoE 103 MB, Granite 205 MB) is
  allocated but never touched by the app's one-output prompts, so it is not in RSS.
- **VmSwap after a run, num minus off:** Granite +107 to +122 MB (medians 200 to 209 against 87 to 94). That is the
  unread int8 head sitting in zram. OLMoE R62: +80 MB (the repacked head's unread rows). OLMoE R60: −9 MB (its head was
  read in full every token). The three arms come out as the table predicts.

**Not verified:**
- that the phone's KleidiAI (dotprod kernels) packs to the same sizes as the host's (i8mm). The phone totals agree with
  it to within about 15 MB;
- the compute-buffer sizes (no load log from the phone).

### A2. Testing the hypothesis with the existing reports

**A2.1 Majors follow swap on Granite** (*device*, gated runs, n = 20 per session):
- correlation of run majors with the net swap-in (VmSwap before − after, in pages): r = 0.88 (R60) and 0.87 (R62);
- with the prefill majors alone: r = 0.96 and 0.86;
- ratio of majors to net pages swapped in: 1.15 and 1.20;
- majors against blocks read: r = −0.56 and +0.26.
- The net swap-in is a lower bound on the gross: pages can be swapped out during the run.

On OLMoE the same correlation is weak (r = 0.68 in R60, 0.14 in R62). §A2.2 explains why.

**A2.2 The largest block of OLMoE's majors is the KV clear, before the prompt graph.**
- Pre-prompt majors are the run's `majflt` minus the `c2-io` window (prompt start to end). Gated runs: medians R60
  47,374 and R62 27,802 on OLMoE, 4,949 and 3,084 on Granite. Warm-ups: 1 to 266.
- **Ceiling.** OLMoE's four highest are 70,316 to 70,727; the KV buffer is 69,632 pages. Granite's highest is 19,688;
  KV + recurrent state is 18,530 pages.
- **Mechanism (code read here):**
  - The kit runs with `reusePrefix = false` (`research.dart:70`).
  - llamadart therefore calls `_clearContextMemory`, which calls `llama_memory_clear(memory, true)`
    (`llama_cpp_service.dart:5944`).
  - That reaches `llama_kv_cache::clear(data=true)` and the recurrent clear, which call `ggml_backend_buffer_clear(buf,
    0)`, a `memset` of the whole buffer (`ggml-backend.cpp:2519`).
  - A write to a swapped-out page swaps it in first: `do_swap_page` returns `VM_FAULT_MAJOR` and counts
    `PGMAJFAULT`, and zram is `SWP_SYNCHRONOUS_IO` (source: Linux 5.4.274 `mm/memory.c`).
- **Cost.**
  - TTFT − prefill-graph ms = 41.6 + 0.00347 × pre-prompt majors (OLMoE, r = 0.99, n = 48); Granite 19.8 + 0.00609 ×
    majors (r = 0.92). So **3.5 µs per zero-page swap-in** (*estimated*) and up to about 0.25 s of TTFT.
  - The intercepts differ by 22 ms, which fits a `memset` of 210 MB more at about 10 GB/s.
- **What the memset does to memory.** About 95 % of OLMoE's KV holds zeros the benchmark never reads.
  - Between runs those pages go to zram as same-filled pages. These take no zram memory but do take swap slots
    (source: kernel zram doc, `same_pages`).
  - Each run then writes all 285 MB back into RAM just before the prefill, when the file cache needs the room.
  - At the worst it was about 276 MB of R60 pos 23's 533 MB VmSwap.

**A2.3 Is zram full?**
- SwapFree below 50 MB: at run start in 13 of 48 (R60) and 12 of 48 (R62) runs; at run end in 3 and **28 of 48**.
- `mem_end` of R62: `swapfree=0`.
- SwapTotal is not recorded, so "full" means no free slots. It does not mean that compressed memory is exhausted.
- With no free slots, anon cannot be swapped out and reclaim must drop file pages. So swapping one page in can push
  out a page of the model's file, not necessarily another anon page.

**A2.4 What the faults cost.** Least-squares fits over unplugged, uncapped runs (warm-ups included), *estimated*:

| model, threads | n | decode ms/token per 1,000 decode majors | ms/token per MB read in the decode | prefill µs per major | prefill ms per MB read |
|---|---|---|---|---|---|
| OLMoE, 3 threads (R60 + R62) | 40 | **0.20** (R60 −0.16, R62 0.38) | 0.085 | 1.7 | 1.26 |
| Granite, 3 threads (R60 + R62) | 30 | **0.12** (R60 0.19; R62, n = 8: 1.24) | 0.125 | 7.4 | 2.30 |
| OLMoE, 4 threads (R62) | 8 | 1.33 | 0.284 | – | – |
| memo-23's fit (R62 OLMoE, 3 + 4 threads pooled, a 4-thread dummy) | 24 | 2.71 here (memo-23: 2.8) | 0.101 | – | – |

- **Reading.** At 3 threads a decode swap-in costs about 4 to 6 µs (0.12 to 0.20 × 32 / 1,000 ms). The R62 OLMoE runs
  with 5,000 to 9,000 decode majors therefore lost about 1 to 2 ms per token to them. Their slowness lies elsewhere,
  in a few runs with 250 to 630 MB of decode reads (+21 to +54 ms per token).
- The 2.8 coefficient belongs to the 4-thread runs, where high fault counts and slow tokens coincide (heat, pages and
  barriers together, R62 caveat).
- **Swap's direct cost per gated run (*estimated*):**
  - OLMoE: TTFT +0.1 to +0.25 s (KV clear) + 20 to 50 ms (in-graph); decode 0.2 to 2 ms per token.
  - Granite: TTFT +0.1 to +0.5 s (in-graph swap-ins of the KleidiAI weights, 35,000 to 80,000 × 7.4 µs); decode
    < 0.1 ms per token.

**A2.5 What would falsify it in R63.** r24 records `/proc/vmstat` (system-wide) at the start, at the first chunk and at
the end, and `mem_first`.
- **Hypothesis falsified** if, in most gated runs (say ≥ 12 of 20 per model),
  Δpswpin(start → first chunk) < 0.5 × (app majors in that window − app inblock pages in that window).
  - `inblock` / 4 KB is an upper bound on file majors.
  - pswpin is system-wide, so it can only be too high from other processes. The test therefore cuts one way.
- **Supported** if Δpswpin covers at least the app's non-file majors and Δpswpout between runs is at least the app's
  VmSwap growth.
- **Predictions to log with it (*estimated*):**
  - OLMoE pre-prompt majors (`majflt` minus the `c2-io` window) ≤ 72,000 in every run, median 20,000 to 50,000 in gated
    runs;
  - Granite ≤ 21,000;
  - warm-ups < 500;
  - Δpswpin over the same window ≥ the pre-prompt majors.
  - If OLMoE's pre-prompt majors exceed 72,000, the KV-clear reading is wrong.
- If `/proc/vmstat` is null (SELinux), only the VmSwap deltas remain, and they are net.

### A3. Options, priced on both models

What an app may do here, checked:
- The app is the top activity with `FLAG_KEEP_SCREEN_ON` during the kit (`research.dart:373`), so its oom score is the
  foreground one.
- kswapd moves anonymous dirty pages to zram regardless of the owner's importance. The LMK's `oom_adj_score` decides
  only which process is killed (source: Android memory-management doc).
- A foreground service therefore changes nothing for an app already on top.
- Android's compaction of cached apps applies to cached processes, which the kit is not.
- targetSdk is 36 (the host's merged manifest, `flutter.targetSdkVersion`).
- The phone's `os_version` `AP3A.240905.015.A2` is the Android 15 base (android-15.0.0_r1, source: android-building
  list), not Android 14 (inferred: Samsung builds on that base).
- Kernel `5.4.274-qgki` (*device*).
- `RLIMIT_MEMLOCK` 64 MiB (*device*).
- `thp` is empty in both reports, so the THP mode is unknown.

**(a) Make it smaller.**
- **KV and recurrent buffers cleared without touching them** (new; generic to every model with a KV cache).
  - Change the CPU buffer's `clear(…, 0)` so that the page-aligned interior gets `madvise(MADV_DONTNEED)` and only the
    edges get a `memset`. An anonymous private range reads as zeros after `DONTNEED`, so the contents equal those after
    the `memset`.
  - **Numerics:** bit-identical by construction. Check with per-chunk PPL at `-ub 1`, `-ub 32` and `-b 512`
    (`llama-perplexity` clears with `data=true` per chunk, so it exercises the path) and with greedy text over three
    consecutive app runs.
  - Alternative: `memset` only the cells used since the last clear (same result, more code).
  - **Gain:**
    - OLMoE TTFT −0.10 s (R62 median) to −0.25 s (R60 maximum) (*estimated*, §A2.2).
    - The 270 MB of zeros are no longer re-inflated before each prefill, so about 270 MB of RAM and 270 MB of swap slots
      stay free.
    - The decode's first writes to new KV rows become zero-fill minor faults instead of swap-ins: OLMoE's floor of
      about 1,000 decode majors per run, ≈ −0.2 ms per token.
    - Decode through residency: **0 to −2 ms per token** on R62-like days (unmeasured; it depends on whether reclaim
      hands the room to the model's file).
    - Granite: TTFT −0.02 to −0.12 s; RAM 18 MB; decode ≈ 0. Its recurrent state (58 MB) is read every token anyway.
  - **Cost:** about 15 ms of `madvise` per run. Each run then takes about 3,700 (OLMoE) or 15,000 (Granite, state
    included) zero-fill minor faults at about 1 µs, *estimated*.
  - **Cheapest decisive test:** the host identity checks above (Tom, under a day). Then one kit A/B knob, read from
    the pre-prompt majors and RssAnon + VmSwap after the run. Predicted: OLMoE pre-prompt majors < 2,000; total anon
    655 → about 385 MB.
- **Granite num: the tied head kept out of KleidiAI and `CPU_REPACK` at load** (load knob; the num arm only).
  - The output tensor then stays the mmapped `token_embd` (the loader reuses the same tensor when the buffer type
    matches, `llama-model-loader.cpp:1372`).
  - **Gain:** −155 MB of anon (*estimated*). Today this sits mostly in zram (+107 to +122 MB of VmSwap in num runs,
    *device*), and it uses swap slots. Decode 0 to −1 ms through residency; TTFT ≈ 0.
  - **Numerics:** head2's screen and exact rows read the Q4_0 copy and the mmapped file rows, never the KleidiAI copy
    (`0001` 2622-2632), so the num arm should be bit-identical. Check at three settings with `GGML_HEAD2_BATCH=1`. In
    the off arm it is a declared change (R61: KleidiAI moves Granite's stock by +0.5 %).
  - **Generality:** tied-head models on a KleidiAI build with head2. OLMoE's head is untied and needed in num, so this
    does not apply to it.
  - **Test:** host load log `CPU_KLEIDIAI` 541.08 → about 394 MiB, plus the identity runs.
- **KleidiAI off for Granite:**
  - no anon saving: `CPU_REPACK` takes the Q4_0 tensors (412 MB) and the head (Q6_K 8x4, 126 MB);
  - the numerics move (R61);
  - repack off altogether is a net loss (+3.7 ms Granite, +2.7 ms OLMoE plain kernels, not bit-identical; memo-18 (e)).
  - **Dead.**
- **Selective repack:** no candidate. Every repacked tensor is a decode-hot dense weight, except Granite's num head
  above.

**(b) Keep it warm.**
- **Touch during the wait:** one load per 4 KB page of the hot anon ranges every 5 to 10 s.
  - Hot ranges: OLMoE about 290 MB (attention, head, screen, KV prefix); Granite about 560 to 640 MB.
  - Cost: 75,000 to 160,000 pages at about 50 to 100 ns, so **4 to 16 ms of CPU per pass** and ≤ 0.3 s per 90 s wait
    (*estimated*). That is cheaper than the hold's 2 s.
  - Effect: the reclaim moves to file pages, our model's included, or to other apps. Inside our process that is close
    to zero-sum. It helps only because the file hold restores file pages *before* the run, while swapped anon comes back
    on demand *inside* the run.
  - Gain: TTFT −0.1 to −0.5 s; decode 0 to −0.4 ms. Risk: more pressure on the rest of the phone. Bit-identical,
    generic.
- **Process state:** no lever (above). Samsung's own memory manager could not be checked (§E).

**(c) Pin or advise.**
- **`mlock` ≤ 64 MiB** (`mlock2(MLOCK_ONFAULT)` exists since Linux 4.4).
  - Best target: OLMoE's 58 MB screen copy in num, or Granite's 58 MB recurrent state.
  - Gain: ≤ 16,000 swap-ins per run, so TTFT ≤ −0.03 to −0.12 s; decode ≈ 0. Costs 64 MiB the kernel cannot reclaim.
    Crumb.
- **`madvise` on 5.4:** `MADV_COLD` and `MADV_PAGEOUT` exist (source) but work the other way. None protects.
  - **`MADV_WILLNEED` on an anonymous range walks it and reads swapped pages back** (`swapin_walk_ops`, source: Linux 5.4
    `mm/madvise.c`). That is the anon analogue of the file hold.
  - Issue it from a helper thread at the gate's end over the hot anon ranges: 50,000 to 110,000 swap-ins at 2 to 7 µs,
    so 0.1 to 0.7 s of CPU before the run (*estimated*).
  - Gain: TTFT −0.1 to −0.5 s (Granite more); decode ≈ 0, since decode swap-ins come from swap-outs during the run.
- **memfd / ashmem:** shmem pages swap to zram as well; ashmem pinning concerns purging, not swap. No protection.
- **THP:** unknown mode; THP swap on arm64 5.4 is not expected. No lever.

**(d) Let the file side take the pressure.**
- Already done: prefill-cold (R58), and the repacked sources age out by themselves (memo-13 §3).
- Swappiness is system-wide and not ours.
- New: 270 MB of OLMoE's pressure is self-made (the KV `memset`); (a) removes it.

**(e) Shorten the wait.** No in-wait anon series exists. From the endpoints (*device*):
- VmSwap: 35 to 90 MB at warm-ups, 216 to 294 MB (medians) after 90 s.
- File loss, R62 hold passes: 444 to 487 MB lost by 45 s, 602 to 657 MB by about 90 s (medians, after the 45 s
  read-back).
- A 45 to 50 s plateau wait would lose about 25 % less file and an unknown share of the anon (linear: 120 to 190 MB of
  VmSwap, *estimated*).
- r24 records `wait_ms` and VmSwap at the start, which gives the slope over the plateau waits' spread (30 to 90 s).

**(f) Other.**
- Android's own advice: the system reclaims clean file pages and compresses dirty anon into zram (source: Android
  memory-management doc).
- llama.cpp's `--mlock` fails on hosts without a raised `RLIMIT_MEMLOCK` (source: jacar.es note); on-device apps
  generally rely on mmap.
- RAM Plus on Samsung is a zram size setting with writeback (source: SamMobile; Android Authority as summarised in
  search). Its value is the user's, so it is not ours to change.

### A4. Ranked shortlist, Part A (*estimated*)

| # | lever | numerics | OLMoE | Granite | cheapest decisive experiment |
|---|---|---|---|---|---|
| 1 | KV / recurrent clear by `MADV_DONTNEED` | bit-identical by construction | TTFT −0.10 to −0.25 s; −270 MB RAM and swap slots; decode 0 to −2 ms (−0.2 certain-ish) | TTFT −0.02 to −0.12 s; decode ≈ 0 | host identity at 3 settings + greedy ×3; then a kit knob A/B read from the pre-prompt majors |
| 2 | Granite num: tied head not in KleidiAI | identical in num (check); declared in off | n/a | −155 MB anon / zram; decode 0 to −1 ms | host load log + identity |
| 3 | anon hold: `MADV_WILLNEED` on hot anon after the gate | bit-identical | TTFT −0.05 to −0.15 s | TTFT −0.1 to −0.5 s | kit knob; prefill majors and TTFT |
| 4 | plateau gate (R63) | none | file loss −25 %; anon unknown | same | R63 as registered |
| 5 | touch-keep-warm of anon / `mlock` 64 MiB | bit-identical | ≤ −0.4 ms decode | ≈ 0 | after #1 to #3 only |

## B. The "rest" of a token

### B1. What the per-token numbers mean (code, not guessed)

From `patches/llama.cpp/0001-cpu-runtime.patch`:
- `NAME count:us+wait_us` is per token: the op count; then thread 0's µs inside `ggml_compute_forward` (`t1 − t0`);
  then µs from the end of that compute to the end of the barrier after the op (the other threads finishing, plus the
  sync). There is no barrier after tiny ops (lines 5051-5077, 5144-5145).
- `head=` and `router=` are compute + wait of those `MUL_MAT`s and are *included* in `MUL_MAT`. `us/token` is the whole
  `ggml_graph_compute` (line 5434).
- **The per-token line prints only `MUL_MAT_ID`, `MUL_MAT`, `GLU` and `SSM_SCAN`** (lines 4872-4876, "R46 am. 1:
  scored ops only"). OLMoE has no `GLU` (fused into `MUL_MAT_ID`) and no `SSM_SCAN`.
- So "rest" = `us/token` − Σ(4 ops) = small ops' compute + wait, plus per-node loop overhead, the profiler's clock reads,
  and graph start and end.

Medians are not additive (the median-of-difference rest, 5.0 ms, differs from the difference of medians, 6.2 ms).
The table therefore uses **means over the fastest half of the tokens with ≤ 5 majors at 2,112 / 2,035 MHz**. R62's
Granite profiled runs at pos 47/48 were capped at 1,555 MHz (charger) and are excluded.

### B2. The account at 3 threads (*device*, ms per token)

| | Granite num (R60 pos 12, 47; R62 pos 12) | Granite off (R60 11, 48; R62 11) | OLMoE num (R60 23; R62 23) | OLMoE off (R60 24; R62 24) |
|---|---|---|---|---|
| graph total | **45.5 to 45.7** | 52.5 to 53.6 | **40.8 to 41.2** | 43.3 to 46.4 |
| `MUL_MAT_ID` (experts) | 13.2 to 13.4 + 0.6 to 0.7 | 20.9 to 21.0 + 0.5 to 0.65 | 25.6 to 26.2 + 0.35 to 0.5 | 28.8 to 30.2 + 0.3 to 0.7 |
| `MUL_MAT` (all dense) | 21.8 + 1.7 | 21.6 to 22.2 + 1.8 to 2.0 | 9.4 to 9.8 + 0.6 to 0.7 | 9.3 to 10.2 + 0.7 to 0.9 |
| of it: head / router | 3.8 to 3.85 / 0.8 | 5.4 to 5.5 / 0.8 to 0.9 | 2.9 to 3.7 / 0.4 | 3.2 to 3.8 / 0.4 |
| of it: other dense (attention; Granite `ssm_in`, `ssm_out`, shexp) | **18.8 to 18.9** | **17.3 to 17.8** | 6.4 to 6.7 | 6.3 to 6.9 |
| `SSM_SCAN` | 2.6 to 2.7 + 0.2 to 0.3 | 2.3 to 2.4 + 0.2 | – | – |
| `GLU` (shexp SwiGLU) | 0.21 | 0.21 | – | – |
| rest (residual) | 4.9 to 5.1 | 4.8 to 5.1 | 4.1 to 4.8 | 4.2 to 4.4 |
| waits in the four ops | 2.5 to 2.7 | 2.6 to 2.8 | 0.9 to 1.2 | 1.0 to 1.6 |
| nodes / barriers | 1,775 / 666 | same | 533 / 305 | same |
| outside the graph (decode wall / 32 − graph) | 0.83 to 0.95 | 0.74 to 1.05 | 0.66 to 0.78 | 0.66 to 0.71 |

**Granite's rest, split with campaign 1's full table** (`reports/device/r15_2026-11-08.json`, *device*):
- Conditions: d6-e6 file, same architecture and tensor types, `fuse=1` blocks with 1,775 nodes and ≤ 2 majors,
  skip 117 + head2.
- Small-op compute ≈ 3.3 ms:
  - `MUL` 76 ops 1.08 to 1.12 (14 µs per op);
  - `ADD` 428 ops 0.62 to 0.64;
  - `RMS_NORM` 117 ops 0.40 to 0.41;
  - `GLU` 0.25, `SSM_CONV` 0.24, `FLASH_ATTN_EXT` 0.14 to 0.34, `UNARY` 0.17, `GET_ROWS` 0.16, `ARGSORT` 0.11, `SCALE`
    0.07, `CONT` 0.06.
- Small-op waits 0.15 to 0.6 ms (`RMS_NORM` 0.05 to 0.3, `ADD` 0.04 to 0.14).
- Per-node dispatch (`us/token` − compute − wait) 1.7 to 2.5 ms, i.e. 0.96 to 1.4 µs per node. That includes the
  profiler's two to three clock reads per node: ≈ 0.15 to 0.2 ms, *estimated*.
- This matches today's 4.9 to 5.1 ms.

**Barrier waits (Granite):**
- 666 barriers × 0.30 µs (R18, *device*, 3 threads) ≈ 0.2 ms of pure sync.
- The other 2.3 to 2.5 ms of the four-op waits are imbalance or stragglers. `MUL_MAT` alone is 6.4 µs per op, the same
  as in campaign 1 (6.8, `last_3ms.md`).
- OLMoE: 305 × 0.30 ≈ 0.1 ms sync, 0.8 to 1.1 ms imbalance.

**OLMoE's rest (4.1 to 4.8 ms) has no device split anywhere.** An *estimated* split from Granite's per-op rates and
OLMoE's shapes:
- dispatch ≈ 0.5 (533 nodes × 1 µs);
- flash attention 16 ops ≈ 0.8 to 1.6 (q8_0 KV ≈ 14 MB per token at about 210 positions, plus 16 heads over 3 threads);
- `RMS_NORM` incl. q/k norms ≈ 0.3;
- `ROPE` 32 ≈ 0.1 to 0.2;
- the K/V `CPY` (quantise to q8_0) ≈ 0.1;
- expert weighting and sum (`MUL`, `ADD`) ≈ 0.3 to 0.5;
- router softmax / argsort / get_rows ≈ 0.15;
- waits ≈ 0.3.

### B3. What campaign 1 already did to each item

| item | campaign 1 | status |
|---|---|---|
| barriers around small ops | `GGML_TINY_OP` (sync skipping), R25 | shipped; 666 barriers left |
| `MUL_MAT_ID` | no-barrier op, x4 kernel, in-register Q3_K kernel, fuse + chunk2 (R18, R35, R47) | shipped; compute-bound |
| dense `MUL_MAT` | repack 4x4 sdot at about the byte rate; `MM_CHUNK=16` lost on the device (R33) | "kernels cannot help" (memo-21) |
| `SSM_SCAN` / `SSM_CONV` | state in place, conv ring (rounds 5 and 6) | shipped |
| `RMS_NORM` + `MUL` | upstream fusion kept (`ggml_cpu_can_fuse_rms_norm_mul`) | shipped; the 76 `MUL`s are other ops |
| expert-output `ADD` chains, other `MUL`s | priced (−0.5 to −1.0, `last_3ms.md` (f)), never built | open |
| thread placement | 4 threads lost (R18, R62); the app restricts the 3 workers to cpus 4 to 7 as a set, no per-thread pin (`research.dart:559`, *device* `pinned_cpus`) | per-thread pin open |
| outside the graph | sampler bypass (R33) | 0.7 to 1.0 ms left |

### B4. Remaining milliseconds with a plausible bit-identical lever

1. **The dense rate.**
   - Facts (*device*):
     - Granite's other dense ops take **1.1 to 1.5 ms longer in the num arm** than in the off arm, at the same clocks
       (18.8 to 18.9 against 17.3 to 17.8 ms for the same 417 MB: 22.1 against 23.4 to 24.1 GB/s). This is memo-21's
       unexplained "1.4 ms by which `MUL_MAT` does not fall".
     - Granite's `SSM_SCAN` reads and writes 36 × 1.57 MB of state per token in 2.6 to 2.7 ms ≈ **42 GB/s combined**
       (*estimated* from the header shapes), while the dense GEMV reads 22 to 27 GB/s.
   - Readings (not tested):
     - the dense read rate is limited by per-core miss parallelism or by the bus clock governor, not by the DRAM
       ceiling;
     - Qualcomm's BWMON / memlat governors vote DDR and LLCC frequency from sampled CPU traffic (source: Qualcomm DCVS
       doc, Linux memlat patch), so a token whose compute-bound expert phase got shorter could see a different bus
       clock in its dense phase.
   - **Gain:** Granite −1.1 to −1.5 ms if num got the off arm's rate; −2.5 to −3.5 ms if the dense ops reached about
     30 GB/s. OLMoE −0.5 to −1.3.
   - **Cheapest experiment:**
     - (i) one profiled run per model in block mode (`GGML_CPU_AB_TOKEN` unset: the 14-op table, zero code);
     - (ii) a read-bandwidth microbench in the `GGML_CPU_BENCH` harness: 1 to 4 threads, A78-only against A78 + X1, a
       64 MB buffer, with and without a software prefetch distance;
     - (iii) `/sys/class/devfreq/*/cur_freq` read by the kit before and after a run, if SELinux allows (small app
       change).
   - **Kill** if 3 A78 threads cannot read above 27 GB/s.
2. **Waits: imbalance, 2.3 to 2.5 ms on Granite and 0.8 to 1.1 on OLMoE.**
   - The 3 workers float over 2,112 MHz A78s and a 2,035 MHz X1 with about 1.2 to 1.3 × the A78's IPC, so static
     splits (attention k/v, router: below 12 chunks) end unevenly.
   - **Lever:** per-thread pinning, e.g. three A78s, or two A78s + X1 with the X1 taking the static remainder.
     Bit-identical (placement only).
   - **Gain:** Granite −0.3 to −1.0, OLMoE −0.1 to −0.5 (*estimated*).
   - **Test:** a kit spec field per run (`pinned_cpus` already exists as a set; per-thread affinity is a small Dart and
     llamadart change), ABBA, reading the wait columns.
3. **Granite graph crumbs:**
   - `MUL` 1.1 ms (14 µs per op on vectors of a few thousand floats, unexplained since campaign 1);
   - the 200 expert-output `ADD`s;
   - dispatch 1.7 to 2.5 ms.
   - **Lever:** fuse the weighted expert sum, in the same order, into the `MUL_MAT_ID` epilogue, and find the slow
     `MUL` shape. Bit-identical if the order is kept.
   - **Gain:** −0.5 to −1.5 (Granite), −0.2 to −0.6 (OLMoE, 8 experts × 16 layers).
   - **Test:** a host print of the 76 `MUL` shapes and their host times (Tom, an hour), then (i) above on the phone.
4. **OLMoE attention** (estimated 0.8 to 1.6 ms):
   - 16 heads over 3 threads end 6 / 5 / 5, so a few per cent is idle;
   - the q8_0 dequantise in the flash-attention kernel.
   - First (i) to see the real number; then either split heads × KV blocks across threads (exact only if the reduction
     order is kept: per-head partial sums merged in fixed order), or nothing.
5. **Outside the graph** 0.66 to 1.05 ms: pipeline detokenisation and the Dart callback (memo-21 N5), 0 to −0.5.
6. **Not a lever:** `SSM_SCAN` (it moves its state at about 42 GB/s, near the bus; F16 state would be a numerics change)
   and the head (Granite num 3.8 ms, at its screen floor, `last_3ms.md` (a)).

**Where 25 tok/s stands (*estimated*).** 968.75 / 25 = 38.75 ms per token, outside the graph included.
- **Granite num, fault-free best half:** 45.6 + 0.9 = 46.5 ms (20.8 tok/s).
  - With levers 1 to 3 at their top (−1.5, −1.0, −1.5) and 0.5 outside: ≈ 42 ms (23 tok/s).
  - With lever 1's 30 GB/s case: ≈ 40 ms (24 tok/s).
  - It still needs memo-21's expert pool (13.3 ms against a 6.0 to 7.7 ms floor) to reach 38.75.
- **OLMoE num:** 41.0 + 0.7 = 41.7 ms (23.2 tok/s). It needs −3 ms: levers 1 to 4 together give 0.8 to 4 ms.
- Both need **fault-free decodes**, which no gated run of R60 or R62 had.

## C. Combined shortlist (top 5 across A and B)

| # | lever | numerics | OLMoE | Granite | cheapest experiment |
|---|---|---|---|---|---|
| 1 | KV / recurrent clear by `MADV_DONTNEED` (A3 a) | bit-identical by construction | TTFT −0.10 to −0.25 s; −270 MB RAM and swap slots; decode 0 to −2 ms | TTFT −0.02 to −0.12 s; decode ≈ 0 | host identity at 3 settings + greedy ×3; then a kit knob read from the pre-prompt majors and the anon total |
| 2 | open the dense-rate pool: block-mode profile + read-bandwidth microbench (B4.1) | measurement | −0.5 to −1.3 ms if real | −1.1 to −3.5 ms if real | one block-mode profiled run per model (env only) + a bench line (small code) |
| 3 | Granite num: tied head out of KleidiAI (A3 a) | identical in num (check) | n/a | −155 MB anon; decode 0 to −1 ms | host load log + identity |
| 4 | per-thread pinning (B4.2) | bit-identical | −0.1 to −0.5 ms | −0.3 to −1.0 ms | kit per-run cpu assignment, ABBA, wait columns |
| 5 | graph crumbs: expert-sum fusion, the slow `MUL` (B4.3) | bit-identical if ordered | −0.2 to −0.6 ms | −0.5 to −1.5 ms | host shape print, then the block-mode profile |

The anon hold (`MADV_WILLNEED`, A3 c) is sixth. It is cheap but moves only TTFT.

## D. Where the brief, the repository or earlier memos are contradicted

1. **"Make the op-by-op account from the cpu-prof lines in r22/r23":** those lines carry four op classes (two on
   OLMoE) by design (R46 amendment 1). The account above uses residuals, with campaign 1's r15 for Granite's split.
   OLMoE's rest has no device split.
2. **"KleidiAI off for Granite's 541 MiB":**
   - 541 MiB is the Q4_0 dense weights plus a ~148 MiB int8 copy of the tied head.
   - STATUS (2026-11-15, Tom) says Granite's tied head is not taken by KleidiAI. The R61 log arithmetic and the loader
     code say it is: as a duplicated output tensor named `token_embd.weight`, which 0006's `output.weight` test does
     not catch.
   - KleidiAI off would hand the same tensors to `CPU_REPACK` and save no anon.
3. **Memo-23's "≈ 2.8 ms per token per thousand decode majors":** reproduced only by pooling 3- and 4-thread runs. At 3
   threads the fit is 0.12 to 0.20 (4 to 6 µs per swap-in).
4. **Memo-23's prefill / decode split leaves out the largest group on OLMoE:** the pre-prompt majors of the KV
   `memset` (median 27,800 to 47,400 per gated run).
5. **"The fadvise hold cannot fix it":** true of `posix_fadvise`, but `madvise(MADV_WILLNEED)` on anon swaps in on Linux
   5.4 (source). An anon hold exists.
6. **Memo-21's Granite "rest ≈ 6.2 ms, waits ≈ 2.5 ms":** 6.2 is a difference of medians. The additive account gives
   4.9 to 5.1 ms of rest plus 0.21 of `GLU`, and 2.5 to 2.7 of waits in the four ops plus 0.15 to 0.6 of small-op waits.
7. **"Android 14":** the phone's `os_version` is the Android 15 base build (source; inferred for Samsung's build).
8. **"0.2 to 0.45 GB pushed to zram":** gated-run VmSwap medians are 216 to 294 MB with a range of 96 to 613. On OLMoE
   up to about 276 MB of it is zeros of its own KV cache.

## E. What could not be verified

- The phone's KleidiAI packing sizes and compute-buffer sizes (no phone load log); the int8-head reading rests on host
  logs and on the device totals agreeing to about 15 MB.
- SwapTotal, the zram algorithm, `page-cluster`, writeback (RAM Plus) on this phone; the THP mode; whether Samsung's
  memory manager treats a top app's anon differently.
- That this Samsung 5.4 build matches upstream 5.4.274 in `do_swap_page`, `swap_readpage` and `madvise_willneed` (read
  upstream only).
- The 3.5 to 7 µs per swap-in and 0.085 to 0.125 ms per token per decode MB are in-sample regressions with no standard
  errors, and they mix sessions.
- That `MADV_DONTNEED` on llama.cpp's CPU buffer is safe on bionic: the buffer comes from `ggml_aligned_malloc`; private
  anonymous memory is expected, not checked on the phone.
- Whether the 270 MB freed by lever 1 ends up as the model's page cache: unmeasured; the decode range is a guess with a
  floor of zero.
- The cause of Granite's slower dense ops in num; any measured SD888 CPU read-bandwidth figure (only the 51.2 GB/s
  nominal was found).
- OLMoE's small-op split; flash attention's real cost.

## F. Sources

- Linux 5.4 `madvise_willneed` (anon: `swapin_walk_ops`; `MADV_COLD`, `MADV_PAGEOUT` accepted):
  https://raw.githubusercontent.com/torvalds/linux/v5.4/mm/madvise.c
- Linux 5.4.274 `swap_readpage` (`PSWPIN` on the `bdev_read_page` path, no `submit_bio`):
  https://raw.githubusercontent.com/gregkh/linux/v5.4.274/mm/page_io.c
- Linux 5.4.274 `do_swap_page` (`SWP_SYNCHRONOUS_IO`, `VM_FAULT_MAJOR`, `PGMAJFAULT`):
  https://raw.githubusercontent.com/gregkh/linux/v5.4.274/mm/memory.c
- zram same-filled pages, writeback: https://www.kernel.org/doc/html/v5.4/admin-guide/blockdev/zram.html
- Android memory management (kswapd, zram, LMK and `oom_adj_score`):
  https://developer.android.com/topic/performance/memory-management
- Android 15 base build `AP3A.240905.015.A2`: https://groups.google.com/g/android-building/c/22IjxTHrLj4
- Samsung RAM Plus: https://www.sammobile.com/news/samsung-ram-plus-does-not-work-the-way-you-think/ ;
  https://www.androidauthority.com/p-3565667 (search summary only)
- llama.cpp `--mlock` and `RLIMIT_MEMLOCK`: https://jacar.es/?p=7984
- Qualcomm DDR / LLCC voting (BWMON, memlat):
  https://docs.qualcomm.com/bundle/publicresource/80-80021-3/topics/dynamic-clock-and-voltage-scaling-dcvs.md ;
  https://lists.infradead.org/pipermail/linux-arm-kernel/2024-July/940397.html
- SD888 memory (nominal 51.2 GB/s): https://www.anandtech.com/show/16271
- Repository:
  - `patches/llama.cpp/0001-cpu-runtime.patch` (profile lines 4060-4110, 4840-4920, 5051-5150; head2 2490-2640);
  - `patches/llama.cpp-c1-olmoe/0006`;
  - `lab/llama.cpp-c1r23` (`src/llama-kv-cache.cpp:391-401`, `src/llama-memory-recurrent.cpp:150-162`,
    `ggml/src/ggml-backend.cpp:2517-2520`, `src/models/granite-hybrid.cpp:57-65`, `src/llama-model-loader.cpp:1155-1185`
    and `1365-1395`, `ggml/src/ggml-cpu/kleidiai/kleidiai.cpp:1785-1830`);
  - `lab/app-c2` (`lib/assistant.dart:453-476`, `lib/research.dart:70`, `:373`, `:559`;
    `third_party/llamadart/.../llama_cpp_service.dart:5040-5056`, `5915-5945`);
  - `lab/logs/r61/*.load.log`;
  - `reports/device/r15_2026-11-08.json`, `reports/memos/last_3ms.md`, `docs/runtime-knobs.md`;
  - memo-13, memo-18, memo-21, memo-23; R60 to R62.
