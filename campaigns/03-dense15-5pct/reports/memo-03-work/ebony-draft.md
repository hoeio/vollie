# Campaign 3 memo-03, Ebony's draft: the unpriced levers (GPU, NPU, read rate, fewer bytes)

Agent model: claude-fable-5-2 (Ebony). Date 2026-11-18. Repository HEAD when written: `6a8b7fb`.
Round 2 of the joint work; scope is [memo-01](../memo-01-feasibility.md) §7.

**Nothing here ran on a phone or on the host.** Labels: *device* = quoted from a stored campaign 1 or 2 phone
result (named); *host* = a read of the source tree on the laptop; *estimated* = my arithmetic; *published* = a
number from a cited source, measured by others on other hardware. Web pages were read on 2026-11-18 through a
summarising fetch tool or a search summary; each source line says which. Pricing in bytes and milliseconds is
Bonnie's lane: every *estimated* figure is a first guess for her to replace. This draft proposes; verdicts and
pre-registrations are the lead's.

Baseline used throughout (memo-01 §1): a dense 7B body reads 4.0 GB per token at Q4_0; the CPU alone prices at
4.5 to 5.7 tok/s at 19 to 24 GB/s (*estimated* from *device* rates), with an `sdot` compute cap of 7.6 to 9.3 tok/s
on three cores (*estimated* from *device* benches) and an unsolved residency wall.

## 0. Summary

1. **One arithmetic fact first.** 15 tok/s × 4.0 GB = 60 GB/s, and the Snapdragon 888's LPDDR5 bus is 51.2 GB/s
   nominal (*published*, AnandTech, search summary). **No engine and no combination of engines reads a 4.0 GB body
   15 times per second on this phone.** GPU and NPU levers can raise the rate; they cannot reach the goal on a
   7B body without a byte cut as well. They can matter for the provisional 4.5 B subject (2.53 GB per token needs
   38 GB/s).
   *Correction, same day: 2.61 GB and 46 to 51 GB/s by Bonnie's header inventory; see §9.*
2. **Campaign 1's sentence "the GPU shares the same DRAM the CPU already saturates" was an assumption, and the
   evidence is against it.** The CPU reads 19 to 24 GB/s of a nominal 51.2 (*device*). On a Snapdragon 8+ Gen 1,
   which has the same nominal bus, a published study measured 39 GB/s through the Adreno 730 and 7.33 tok/s on a
   7B model on the GPU against 4.11 on the CPU. On the 8 Gen 3, one engine reads 40 to 45 GB/s and two engines
   together 59 to 60 of a 68 GB/s ceiling (*published*, HeteroLLM). Whether this phone's bus has that headroom is
   **unmeasured**, and one short device bench decides it (§5 X1, X2).
3. **GPU alone does not beat the CPU here.** The stock OpenCL backend is declared unsupported on phone Adreno 6xx
   drivers by its own documentation (master, read today). Campaign 1 ran it with workarounds that switch off its
   fast kernels. I estimate 1 to 2.5 tok/s on a 7B body with the stock backend and 3.5 to 6 with a purpose-written
   kernel (§1).
4. **A concurrent CPU + GPU split of each mat-vec is the only lever in this round that could move a dense body by
   more than 1.3 times**: 6.5 to 9 tok/s central on 4.0 GB (*estimated*, conditional on X1 and X2), against 4.5
   to 5.7 for the CPU alone. llama.cpp's own layer split (`-ngl`) is sequential and gains nothing (§1.4). It is a
   large build and a declared numerics change.
5. **The Hexagon 780 is closed with numbers** (§2): upstream llama.cpp has had v68 support since 2025-11-24, HVX
   only, and its author measured 0.27 to 0.49 of the CPU rate on a 1B model. QNN and ExecuTorch need a converted,
   activation-quantised model, which is a different file and costs +4.7 to +20.7 % perplexity where published.
6. **Read rate from the CPU side** (§3): the bus governor cannot be set by an app; what an app can change is how
   many independent miss streams each core runs. A multi-stream GEMV is bit-identical and is the cheapest thing to
   test. Huge pages, A55 helper readers and a bus warm-up are small or blocked.
7. **Fewer bytes without a new kernel** (§4): a per-token FFN-sublayer skip (FFN-SkipLLM) skips whole tensors, so
   it saves DRAM traffic in the stock layout. It extends memo-01's H7 on the host. Entropy-coded weights are
   bit-identical but lose on decode compute by 5 to 10 times.
8. **Correction to memo-01 §7.** The stored records do not show that the 0.7 to 2.9 tok/s was on MoE models; the
   one attributed OpenCL measurement was on the dense hybrid Qwen3.5-4B (§1.1).

Ranking by expected tok/s on a dense 7B body is in §6.

## 1. Dense Q4_0 on the Adreno 660

### 1.1 What campaign 1 actually ran (*host*: read of the sibling `vollie` tree and campaign 1 reports)

| item | finding | where |
|---|---|---|
| backend | llama.cpp `620d2225c` (2026-10-15), the campaign base. It is **not** an older backend. Built for OpenCL 2.0 (`-DGGML_OPENCL_TARGET_VERSION=200`) | `vollie/tool/build_opencl.sh` |
| driver | Adreno 660, compiler **E031.38**. It segfaults on some kernels and then stays broken for the process | `vollie/tool/clc.c`, `opencl-compile-retry.patch` |
| workarounds | multi-row decode GEMV (`mul_mv_f16_f32_mrow`) not built; `l4` and RMS-norm kernels base variant only, fusion off; Qualcomm subgroup shuffles off; flash attention declined to the CPU; five ops fall back to the CPU; a failed program is retried with `-cl-opt-disable`, which is an unoptimised build | same patch |
| the number | "0.7 to 2.9 tok/s, always below the CPU (*device*)". **No R-number, no stored JSON, no model named beside it** (`reports/memos/readme_r49_sources.md`: "03 'Didn't' list (no R-number in the log)") | `reports/03_final.md` |
| the one attributed run | Qwen3.5-4B Q4_0 (a **dense** hybrid, 2.23 GB per token): OpenCL "about 35 % slower than the CPU", owner's phone test of 2026-10-25, when the CPU gave 6.6 cold and 5.9 hot tok/s | `reports/02_methods.md` rows 13 and the anchor table; `01_vollie_vs_llamacpp.md` |
| heat | "back-to-back speed tests halved every number (GPU 2.9 → 0.7 tok/s)" | comment in `vollie/lib/assistant.dart` |
| contradiction | another comment in the same file says "OpenCL failed on the 888" | `assistant.dart`, `heatPlan` |

Reading (*estimated*): 35 % below 5.9 to 6.6 is 3.8 to 4.3 tok/s on 2.23 GB, an effective 8.6 to 9.6 GB/s, with
a crippled kernel set. The 2.9 figure may be the same test when warm, or Granite; the records cannot tell.
**So the campaign 1 number is the ceiling of the stock backend on this phone's driver. It is not a measurement of
what the Adreno 660 can read.** Memo-01 §7's "on MoE models and an older backend" should be corrected on both
counts.

### 1.2 The backend today (*published*)

- **Support.** `docs/backend/OPENCL.md` on master (fetched today) lists Adreno 750, 810, 830, 840, X1-85 and X2-90
  as verified. Its note and its known issues say that A6x GPUs work "with recent drivers and compilers (usually
  found in IoT platforms)" and do not work "on A6xx GPUs found in phones with old drivers and compilers". The text
  at the campaign base is the same (*host*).
- Qualcomm's IWOCL 2025 slides (PDF read in full): "Support selected Adreno 600 devices (like the Qualcomm
  Robotics RB5)"; "Ported to old devices: performance being optimized (Adreno 600 serials)".
- **Measured decode, 7B dense, Q4_0** (same slides, slide 14): Llama-2 7B gives 12 tok/s on the Adreno 750,
  15 on the Adreno 830 and 18 on the Snapdragon X Elite; Llama-3 8B gives 10, 11 and 13. Slide 13 states the
  model directly: decode is memory bound, "3.5 GB is required for every token, e.g. for a device with memory BW
  of 60 GB, 17 tokens/sec is the peak".
- **No measured tok/s for a 7B model on any Adreno 6xx was found** (three searches). A 2024 study (arXiv
  2410.03613v5) had a Snapdragon 870 tablet (Adreno 650) in its set and reports that MLC LLM crashed on it.
- A newer measurement study (arXiv 2607.05475v2, Xiaomi 17, Snapdragon 8 Elite Gen 5) has llama.cpp at 12.6 tok/s
  on the CPU and 10.2 on the GPU for Qwen2.5-7B: on current phones the OpenCL path is now **below** the CPU path
  (table read through the fetch tool; the tool warned the extraction was noisy).

### 1.3 The GPU's achievable DRAM read rate (*published* for other chips; **unmeasured** for the Adreno 660)

| chip, bus | GPU read rate | CPU beside it | source |
|---|---|---|---|
| Snapdragon 8 Gen 3, Adreno 750, LPDDR5X (77 GB/s theoretical) | 63 GB/s by clpeak; 43.3 GB/s in decode | one engine 40 to 45 GB/s; GPU + NPU 59.1; ceiling about 68 | arXiv 2410.03613v5 Table IV; HeteroLLM arXiv 2501.14794 §3.3, §5.3 |
| Snapdragon 8+ Gen 1, Adreno 730, LPDDR5-6400 (**51.2 nominal, as this phone**) | 39 GB/s by clpeak; 7.33 tok/s on Llama-2 7B, about 26 to 28 GB/s effective (*estimated* at 3.5 to 3.8 GB per token) | CPU 4.11 tok/s at 4, 6 and 8 threads, about 14 to 16 GB/s effective (*estimated*) | arXiv 2410.03613v5 Tables IV and V, read twice. The GPU row is labelled "Xiaomi Pad (GPU)", llama.cpp, 6 threads; the paper states a 1.78 times decode speed-up over the CPU and does not explain it |
| Snapdragon 8cx Gen 3, Adreno 690 (an A6xx-generation GPU, compiler E031.45) | 51 to 58 GB/s by clpeak; 1.9 to 2.6 TFLOPS single precision | not given | a public gist, 2024-03-17; one run, some fields implausible |
| Snapdragon 8 Elite Gen 5 | CPU and GPU 57 to 60 GB/s; NPU 39.7 | | arXiv 2607.05475v2 |
| **Snapdragon 888, Adreno 660** | **not found anywhere** | 18.9 to 24.3 GB/s (*device*, R25, R29; power saving most likely on) | |

What this supports:

- An A6xx-generation GPU is not limited by its architecture to a low read rate (the Adreno 690 row).
- On the one published chip with this phone's nominal bus, the GPU read 39 GB/s while the CPU's decode implied
  about 15. **The CPU did not saturate that bus.**
- By graphics-benchmark scaling the Adreno 660 is roughly 0.5 of the 690 and 0.75 of the 730 (recalled, not
  checked today). I put its clpeak read rate at **20 to 35 GB/s** (*estimated*; wide on purpose) and a real Q4_0
  GEMV at 0.7 of clpeak, as the 750 shows (43 to 46 of 63): **14 to 24 GB/s**, which is 3.5 to 6 tok/s on 4.0 GB.
- So the GPU alone is at best level with the CPU. **GPU-only is not a lever. The question is the sum.**

### 1.4 A CPU + GPU split of the weight bytes

- **llama.cpp's `-ngl` split is sequential.** The GPU runs its layers, then the CPU runs the rest. The two never
  read at the same time, so the token time is the sum of two partial times and the bus sees one reader. With a GPU
  no faster than the CPU it gains nothing. (*host*: scheduler behaviour, as campaign 1's CPU fallbacks showed.)
- **The gain needs a concurrent split inside each mat-vec**: the GPU computes rows 0 to k and the CPU threads rows
  k to n of the same matrix, joined before the next op. This is HeteroLLM's "tensor-level" row cut. On the 8 Gen
  3 it raised used bandwidth from 43.3 to 59.1 GB/s (1.36 times) and decode by 8.5 to 23.4 % over GPU-only
  (*published*; GPU + NPU, W4A16 weights computed in float).
- **Price on this phone** (*estimated*; every input unmeasured here):
  - Bus ceiling by HeteroLLM's ratio (68 of 76.8): about 45 GB/s. Two engines by its ratio (59.1 of 76.8): about
    39 GB/s.
  - Sum of the parts with a 0.7 contention factor on the GPU: 19 to 24 plus 10 to 17 gives 29 to 41, capped near
    40. **Combined 29 to 40 GB/s, or 7.2 to 10 tok/s on 4.0 GB by bytes alone.**
  - Compute stops binding: the CPU computes only its share (0.6 of the rows lifts its cap from 7.6 to 9.3 up to
    12.7 to 15.5 tok/s).
  - Less the join cost (one GPU enqueue and wait per mat-vec, about 230 mat-vecs per token on a 32-layer model,
    20 to 100 µs each: 5 to 23 ms per token) and the heat of two engines: **6.5 to 9 tok/s central, 10 to 11 in
    the best corner**. That is 1.3 to 1.7 times the CPU alone.
  - On the provisional 2.53 GB body the same rates give 11.5 to 15.8 tok/s by bytes. **This is the lever that could
    carry the 4.5 B subject across 15, and nothing else in this round is.**
- **Residency side effect (unpriced, could be decisive either way).** The GPU's share sits in driver memory
  (kgsl), which the kernel cannot evict the way it evicts page cache. A 0.4 share moves 1.6 GB out of the page
  cache and leaves 2.4 GB file-backed, inside the 2.67 GB this phone has held (*device*, R58). But 1.6 GB of
  unreclaimable memory is in the range where the app has run (2.27 GB anonymous, *device*) and below where it was
  killed (3.93 GB, *device*, R9); and pinned memory pushes reclaim onto the remaining file pages (the reason
  memo-29 withdrew the dma-buf pin). Bonnie should price this, not me.
- **Costs and rules.**
  - Numerics: the GPU kernel computes in float and the CPU in int8 `sdot`. A declared numerics change, checked at
    `-b 512` and `-ub 1` (the split applies only to decode-shaped ops, so the two differ: the batched path stays
    on the CPU). Expected perplexity change far inside the budget (Qualcomm reports CPU and GPU "closely
    matching"; *published*, no number for our gate).
  - Driver: the kernel must compile on E031.38. Campaign 1 found the simple base kernels compile and the newer
    ones crash; a single hand-written Q4_0 GEMV is in the simple class (*estimated*).
  - Build: large. A CPU-backend GEMV that owns an OpenCL queue, a zero-copy or copied weight share, a row-split
    tuner. Three to five days of Tom's time before the first device number (*estimated*).
  - Heat: campaign 1 saw GPU runs halve when hot. The claim format's 32-token runs with a cool-down gate limit
    this; a sustained chat would not hold the rate.
  - "Fix it in the runtime": respected; no setting on the phone is touched. The kit must load `libOpenCL.so`
    again, which r27 removed on purpose (memo-28 §2.5): load it only in the arm that uses it.

## 2. The Hexagon 780 (v68)

| path | state on v68 (*published* unless marked) | verdict |
|---|---|---|
| llama.cpp `ggml-hexagon` | PR #17394 "Initial Hexagon v68/v69 support", merged 2025-11-24. HVX only (no HMX kernels on v68). The author's numbers on a v68 device ("Makena", the 8cx Gen 3 by the PR text), Llama-3.2-1B Q4_0: NPU 7.21 tok/s against CPU 26.92; with flash attention 8.88 against 17.99. The author: "Runs models, but really slow"; the maintainer: performance "is not going to be that good". Crashes worked around with `GGML_HEXAGON_NHVX=1` and by disabling the DMA bypass. At the campaign base the build list is v73, v75, v79 and v81 only, and no v68 string is in the backend (*host*) | **0.27 to 0.49 of the CPU.** On a 7B body: 1.2 to 2.8 tok/s (*estimated* by ratio), before the session limit below |
| session size | one NPU session maps about 3.5 GB; larger models are mapped and unmapped during the graph (base tree docs, *host*). A 4.0 GB body does not fit one session | further loss |
| newer chips, for scale | 8 Gen 3 (v75), llama.cpp b7588, Q4_0, Llama-3.1-8B and Qwen3-8B: NPU `MUL_MAT` 1.55 to 1.67 times the CPU in decode but only 1.05 to 1.20 times end to end, and +22 to +51 % battery drain (arXiv 2605.27435). 8 Elite Gen 5 (v81): NPU DDR read 39.7 GB/s against 57 to 60 for CPU and GPU; 20 to 30 µs per host-to-NPU call, about 533 ops per token (arXiv 2607.05475v2) | even with HMX and three more generations the NPU is a 1.05 to 1.2 times lever in llama.cpp |
| QNN / Genie | static-shape graphs, activations quantised (W4A16). Genie reaches 14.6 tok/s on Qwen2.5-7B on the 8 Elite Gen 5, with perplexity +4.7 to +20.7 % on its Llama models and W4A16 10 to 42 % above the CPU W4 baseline (arXiv 2607.05475v2). No statement of LLM support on SM8350 was found: two searches returned Linux IoT v68 packages (QCS6490) and nothing for the 888 | **breaks two rules**: the model is a converted context binary, not the served GGUF; and the published perplexity cost is at or over the +5 % budget |
| ExecuTorch Qualcomm backend | supported SoC list starts at Snapdragon 8 Gen 1 in the copy found (search summary; the list in `qc_schema.py` was not read) | no SM8350 |
| MLC LLM | no Hexagon path; it uses OpenCL on Adreno | see §1 |
| mllm, PowerInfer-2, HeteroLLM | NPU work on v73 and later only, all through QNN | no v68 |

**Can an app use it without root?** Unsigned protection domains exist for this purpose and were reported working
on a non-rooted Snapdragon 855 (Qualcomm forum, search summary), and failing on other devices for firmware or
policy reasons. For this phone it is **unknown**. An app also needs `libcdsprpc.so` declared with
`uses-native-library` and exported by the vendor's public library list. A 40-line probe would answer it (§5 X5).

**Verdict I propose: closed for campaign 3, with these numbers.** It corrects campaign 1's reason ("no supported
path in this llama.cpp"): a path has existed upstream for a year, and it is slower than the CPU. As a third
concurrent reader beside CPU and GPU it would add at most the gap between two engines and the bus ceiling, about
5 GB/s (*estimated*), for a FastRPC build nobody has done on v68. I would not spend the probe unless X2 shows
the two-engine sum already at the ceiling and the campaign still needs 10 % more.

## 3. More read rate from the CPU side

### 3.1 The bus and its governors (*published*, partly search summaries; nothing read from this phone)

- Peak: 4 × 16-bit LPDDR5 at 3,200 MHz, 51.2 GB/s (AnandTech, search summary). This confirms the "recalled, not
  checked" figure in CAMPAIGN.md.
- Qualcomm's kernels scale the DDR and cache-controller clocks with devfreq governors: `bw_hwmon` votes from the
  measured CPU-to-DDR traffic; `memlat` votes from per-core cache-miss counters, including the L3 clock; a
  "compute" governor maps the CPU frequency to a DDR floor (Qualcomm DVFS documentation and a third-party
  article, search summaries; a sibling-SoC device tree with a `cpu4-computemon` table of CPU frequency to LPDDR5
  frequency up to 3,196 MHz, search summary only, the page itself returned HTTP 429 twice).
- Two consequences:
  - **A CPU clock cap can lower the DDR floor.** If Samsung's power saving held the big cores near 2.1 GHz in
    campaign 1, the 19 to 24 GB/s may have been read on a bus below 3,196 MHz. Memo-01's P2 (dense rate with
    power saving off) is therefore also a bus test. This is a reading, not a finding.
  - **An app cannot set any of this**: the nodes are root-owned and the votes come from hardware monitors. ADPF
    cannot either (campaign 2 memo-31 and memo-32). The big-core L3 clock is voted the same way and needs no action.
- What an app controls is the traffic the monitors see. A decode that starts after the cool-down wait starts on a
  low bus vote. The ramp is one or a few sampling windows; over 32 tokens that is under 1 % (*estimated*). A
  bus warm-up thread is not worth a lever. It is free to observe in X1 (rate per 10 ms slice).

### 3.2 Is the CPU's 19 to 24 GB/s a bus limit or a core limit? (the question everything in §1.4 and §3 hangs on)

Evidence for a **core** limit (headroom exists):

- Granite's `SSM_SCAN` moves about 42 GB/s read plus write while the dense GEMV reads 22 to 27 (*estimated* from
  *device* profiles, memo-24).
- The published rows of §1.3: a GPU reading 39 GB/s on the same nominal bus.

Evidence for a **bus** limit (no headroom):

- Four threads read 22.7 GB/s (*device*, R13, R14), no more than three threads' 18.9 to 24.3.
- On the 8+ Gen 1 the CPU's 7B decode was flat at 4.11 tok/s from 4 to 8 threads (*published*).

Both sets fit a third reading: each big core is limited by how many misses it keeps in flight, and extra threads
land on shared limits (the 4th core is taken by Android's threads here, R18 and R62). Campaign 2's T7 microbench
(memo-29) would have separated these and **was never run**. It is the first item of §5.

### 3.3 Levers

| lever | mechanism | numerics | price on a dense 7B body (*estimated*) | state |
|---|---|---|---|---|
| **Multi-stream GEMV** | each thread walks 2 to 4 row ranges at once instead of one, so the core keeps more independent prefetch streams and misses in flight. The per-row arithmetic and order do not change | bit-identical | read rate × 1.0 to 1.3, so 4.5 to 5.7 becomes up to 5.9 to 7.4, then the `sdot` cap (7.6 to 9.3) takes over | new; gated by X1 |
| 4th big-core thread for dense ops only | more cores | bit-identical | × 1.0 to 1.1; R18 and R62 lost on MoE, a dense body is a different regime (Bonnie, memo-01 draft §1) | rides on X1 |
| A55 helper readers | A55 threads stream ahead of a big core's range into the shared 4 MB L3 and never join a barrier (campaign 2's N3 / Y11, priced 0 to −3 ms there) | bit-identical | × 1.0 to 1.15. An A55 reads slower than the core it serves, so it can cover only part of the stream; the lead window is about 1 MB per stream | rides on X1 (total rate with 8 readers) |
| 2 MB pages | 4.0 GB is 1.02 M pages per token: one TLB walk per page, and stride prefetchers restart at each 4 KB boundary | bit-identical | × 1.02 to 1.08. **Blocked twice**: THP mode on this phone is unknown (memo-24, memo-26), and only anonymous memory can be huge on kernel 5.4, which is the copy the residency plan removes | one arm of X1; no build |
| bus warm-up | keep traffic up before the first token | bit-identical | < 1 % | not proposed |
| non-temporal loads | weights bypass the caches | bit-identical | about 0: the A78 treats `LDNP` as a plain load (recalled, not checked) | not proposed |

## 4. Fewer bytes per token without changing the file

| idea | bytes saved in the stock layout? | budget | price on a 7B body (*estimated*) | state |
|---|---|---|---|---|
| **Per-token FFN-sublayer skip** (FFN-SkipLLM, arXiv 2404.03865, EMNLP 2024): skip a whole FFN for a token when the FFN's input and output are nearly parallel in the layers before it | **yes**: whole tensors are not read; no new kernel, no derived copy | declared numerics. The rule is per token position, so it is the same at `-b 512` and `-ub 1`. The paper reports 25 to 30 % of FFN blocks skipped with "marginal change" on task scores and **no perplexity** (abstract and search summary only) | if 5 to 15 % of FFN blocks fit inside +5 %: 4 to 12 % of bytes, × 1.04 to 1.13 | extends memo-01 H7 (static sublayer ablation); campaign 1's R21 (+2.5 % for the cheapest whole layer, *host*, Granite) is the caution |
| Entropy-coded derived copy (EntroLLM, arXiv 2505.02380: Huffman-coded integer weights, lossless, "31.9 to 146.6 % faster" on a Jetson; search summary) | yes, 15 to 30 % if the nibble entropy is 3.2 to 3.8 bits (*estimated*) | bit-identical | **loses.** A table-driven decoder needs about 0.4 ns per symbol against 0.05 ns per weight for `sdot`: 7.1 B symbols on three cores is about 950 ms per token | closed with a number, as memo-21 closed it for experts |
| Bit-plane copy, coarse pass then refinement of the rows that matter | half the bytes of `gate` and `up` on unrefined rows | declared numerics | × 1.0 to 1.1: compute is per weight, not per bit, and compute already binds at 7.6 to 9.3 | not proposed |
| Self-drafted second column (n-gram or prompt lookup, greedy output identical) | reads the body once for two candidate tokens | bit-identical output | × 1.08 to 1.25 only while the token is byte-bound; it turns into a loss as soon as §1.4 or sparsity removes the byte bound (my memo-01 draft §3.5) | stays closed; the free probe column of memo-01 settles it |
| A small draft model run on the idle GPU | – | a second model file | – | needs the owner; not proposed |

## 5. Proposed TEST list additions (device items ride on memo-01's P1/P2 kit; none needs a model round)

| # | test | cost | closes if | opens if |
|---|---|---|---|---|
| **X1** | **Bus-headroom bench** (campaign 2's T7, extended). In the `GGML_CPU_BENCH` harness: sum-reduce and Q4_0x4 GEMV over 512 MB, with 1, 2, 3, 4 big cores; 3 big cores plus 1 to 4 A55 readers on separate buffers; 1, 2, 4 streams per thread; 4 KB pages against `MADV_HUGEPAGE`; GB/s per 10 ms slice for the ramp. Power saving off, clocks recorded | Tom S to M (about 150 lines); device 3 minutes | total with 8 readers ≤ 1.10 × the 3-core rate: the bus is the limit; multi-stream, helpers **and the CPU + GPU split** close (subject to X2) | 3 threads with multi-stream ≥ 1.15 × single-stream: build the multi-stream GEMV |
| **X2** | **GPU read-rate probe.** One OpenCL program on the Adreno 660: a clpeak-style global-memory sum over 512 MB and a plain Q4_0 GEMV; compile result on E031.38 with and without `-cl-opt-disable`; `CL_DEVICE_MAX_MEM_ALLOC_SIZE`; the extension list (int8 dot product, host-pointer and hardware-buffer imports); GPU alone, then **concurrent with the 3-thread CPU stream of X1** | Tom M (about 200 lines, own process as `clc.c` does); device 3 minutes. Zero-build first look: the public clpeak Android app, GPU alone only (its store listing was not checked) | GPU alone < 10 GB/s, or CPU + GPU together < 1.15 × CPU alone: the GPU lane closes for campaign 3 with a device number | together ≥ 1.30 ×: Bonnie prices the split kernel with measured rates |
| X3 | sysfs readability from the app: `/sys/class/devfreq/*/cur_freq`, `/sys/kernel/mm/transparent_hugepage/enabled`, kgsl clock and busy nodes | 10 lines in the kit header | – | readable: every later run records the bus clock |
| X4 | Host: per-token FFN-skip screen added to H7. Rule from FFN-SkipLLM (cosine of FFN input and output, first layers always run), threshold from wikitext-2 train, 40 chunks at `-b 512` and `-ub 1`, both models | Tom S on top of H7 | fewer than 5 % of FFN blocks skipped inside +5 % with a 0.5-point margin on both models | ≥ 10 % on both: a try item with no kernel work |
| X5 | Hexagon unsigned-PD probe | 40 lines and a manifest entry | **not proposed for now** (§2) | – |

## 6. Proposed TRY list additions, ranked by expected tok/s on a dense 7B body (4.0 GB per token)

All *estimated*. "If it works" is the conditional range; P is my probability that the gate passes and the lever
gives at least the low end. The CPU-alone reference is 4.5 to 5.7.

| rank | lever | tok/s if it works | P | label | generic across models | gate |
|---|---|---|---|---|---|---|
| 1 | Concurrent CPU + GPU row split of decode mat-vecs | 6.5 to 9 (best corner 10 to 11) | 0.3 | numerics change | yes: any Q4_0 mat-vec; MoE expert ops as well | X1, X2 |
| 2 | Multi-stream GEMV on the CPU | 5 to 7.4 | 0.3 | bit-identical | yes | X1 |
| 3 | Per-token FFN-sublayer skip | 4.7 to 6.4 | 0.35 | numerics change | yes, threshold per model | X4 |
| 4 | A55 helper readers; 4th dense thread; 2 MB pages | 4.6 to 6.2 | 0.2 | bit-identical | yes | X1 |
| 5 | GPU alone, purpose-written kernel | 3.5 to 6 | 0.2 | numerics change | yes | X2 |
| 6 | GPU alone, stock OpenCL backend | 1 to 2.5 | – | numerics change | – | closed by §1.1 and §1.2 |
| 7 | Hexagon 780 through `ggml-hexagon` | 1.2 to 2.8 | – | numerics change | – | closed by §2 |
| – | Hexagon through QNN or ExecuTorch | not applicable | – | different file, over budget | – | closed by rule |

- Ranks 1 and 2 draw on the same bus headroom and do not add: together about 7 to 9.5.
- Rank 3 multiplies with either.
- **Nothing here changes memo-01's verdict for a 7B body**: the best stack of this round is 7 to 10 tok/s, and
  60 GB/s is above the bus. For the provisional 2.53 GB body, rank 1 prices at 11.5 to 15.8 by bytes, which is
  the first lever in either round whose range contains 15.
  *Withdrawn, same day: re-priced at 8.3 to 12.9 printed; see §9.*

## 7. Not checked, and where I may be wrong

- The Adreno 660's read rate, its compile behaviour on a new kernel, its allocation limit, and what kgsl memory
  does to lmkd on this phone. Every number in §1.3 and §1.4 for this phone is scaled from other chips.
- The 8+ Gen 1 GPU row: I read the tables through a summarising tool, twice, with the same values (39 GB/s;
  7.33 against 4.11 tok/s; llama.cpp). The row names the device only as "Xiaomi Pad". The paper gives no CPU
  read rate, so "the CPU did not saturate that bus" is my inference from its token rates, not its statement.
- The governor description is from Qualcomm's public documents and sibling-SoC trees. Samsung's SM-F711 kernel
  source was not read; it is published and would settle the DDR table and the power-saving question on the desk.
- The Adreno 660 to 690 and 730 ratios are recalled.
- FFN-SkipLLM: abstract only; no perplexity number was seen.
- Whether a hand-written OpenCL GEMV would trip the same compiler crashes as llama.cpp's newer kernels.

## 8. Sources (all read 2026-11-18)

- llama.cpp `docs/backend/OPENCL.md`, master: https://raw.githubusercontent.com/ggml-org/llama.cpp/master/docs/backend/OPENCL.md (fetch tool)
- H. Wang, Qualcomm, IWOCL 2025 slides: https://www.iwocl.org/wp-content/uploads/iwocl-2025-hongqiang-wang-lamacpp-backend-update.pdf (PDF text extracted locally, read in full)
- llama.cpp PR #17394, Hexagon v68/v69, merged 2025-11-24: https://github.com/ggml-org/llama.cpp/pull/17394 (fetch tool)
- HeteroLLM, arXiv 2501.14794v1 (2025-01): https://arxiv.org/html/2501.14794v1 (fetch tool, full text)
- "Understanding LLMs in Your Pockets", arXiv 2410.03613v5: https://arxiv.org/html/2410.03613v5 (fetch tool, first 100,000 characters)
- "Is Your NPU Ready for LLMs?", arXiv 2607.05475v2 (2026-10-16): https://www.alphaxiv.org/abs/2607.05475.md (fetch tool; table extraction noisy)
- "When NPUs Are Not Always Faster", arXiv 2605.27435v1 (2026-05-22): https://www.alphaxiv.org/abs/2605.27435.md (fetch tool)
- clpeak on the Adreno 690, gist of 2024-03-17: https://gist.github.com/woachk/3fa9916bb12667bb11320ad62bc6c81f (fetch tool)
- AnandTech, Snapdragon 888 deep dive (51.2 GB/s): https://www.anandtech.com/show/16271 (search summary)
- Qualcomm DVFS governors documentation: https://dragonwingdocs.qualcomm.com/System/Kernel/configure-the-dynamic-voltage-and-frequency-scaling-dvfs-governors (search summary)
- Bus DCVS overview (third party): https://besthub.dev/articles/bus-dcvs-dynamic-clock-and-voltage-scaling-for-ddr-bandwidth-management-in-modern-socs-7c6856ccd62e (fetch tool)
- Fairphone 5 kernel commit with `computemon` and bwmon tables: https://gitlab.e.foundation/e/devices/android_kernel_fairphone_fp5/-/commit/3a2d0a2e08966a94c7219677ba0644fca4c95200 (search summary only; HTTP 429 on fetch)
- ExecuTorch Qualcomm backend README (AOSP copy): https://android.googlesource.com/platform/external/executorch/+show/HEAD/backends/qualcomm/README.md (search summary)
- Unsigned PD reports: https://developer.qualcomm.com/comment/13372 and https://research.checkpoint.com/2021/pwn2own-qualcomm-dsp/ (search summary)
- FFN-SkipLLM, arXiv 2404.03865: https://arxiv.org/abs/2404.03865 (search summary)
- EntroLLM, arXiv 2505.02380: https://arxiv.org/html/2505.02380 (search summary)
- llama.cpp issue #20603 (Adreno 740, Vulkan slower than CPU, 2026-03-15): https://github.com/ggml-org/llama.cpp/issues/20603 (fetch tool; not used for a number)
- Repository: `reports/02_methods.md`, `reports/03_final.md`, `reports/04_campaign.md`, `reports/01_vollie_vs_llamacpp.md`, campaign 2 memo-21, memo-24, memo-29, memo-31; sibling tree `vollie/tool/build_opencl.sh`, `opencl-compile-retry.patch`, `clc.c`, `lib/assistant.dart`; llama.cpp at `620d2225c` (`docs/backend/OPENCL.md`, `docs/backend/snapdragon/`, `ggml/src/ggml-hexagon/CMakeLists.txt`).

## 9. Corrections to §0 to §6 after reading Bonnie's draft (same day; the text above is left as written)

Bonnie's review (her §11) changes four of my numbers. I accept all four.

1. **The provisional subject is 4,644 M weights and 2.61 GB per token at best** (*host*, her header inventory),
   not "4.5 B, 2.53 GB". Read §0 item 1, §1.4 and §6 with that figure. With her non-product time O and the
   printed factor 0.95, 15 printed needs 46 to 51 GB/s on that body, not 38.
2. **CPU + GPU split on Gemma 4 E4B: 8.3 to 12.9 tok/s printed** (*estimated*, her re-price with my inputs), not
   "11.5 to 15.8 by bytes". **The range does not contain 15. My sentence "the first lever whose range contains
   15" is withdrawn.**
3. **CPU + GPU split on a 7B body, same treatment** (*estimated*): bytes 100 to 138 ms at 29 to 40 GB/s, joins 5
   to 23 ms, O 5 to 10 ms: 110 to 171 ms, **5.6 to 8.6 tok/s printed** (I wrote 6.5 to 9 without the printed
   factor). Against her stock row at the same clocks (4.3 to 5.5 printed at 19 to 24 GB/s) the lever is still
   × 1.3 to 1.6.
4. **The drafted second column moves from "closed" to "priced, gated by P2"** (§10 point 3 below has the one
   reservation).

One design change that her join-cost point leads to, for whoever prices the split next: the graph already has
independent sibling products on the same input (`gate` and `up`; `q`, `k` and `v`). Giving one sibling to each
engine needs **one join per sibling group and no row split**: four joins per layer instead of seven to nine. On
Gemma's small matrices (14.7 MB each) that halves the overhead she counted. `down`, `o` and the head still need
a row split or stay on one engine.

## 10. Review of Bonnie's draft (`bonnie-draft.md`, read 2026-11-18 after §0 to §8 were written)

Her header inventory is the first exact byte table of the campaign, and her §9 formula turns P2 and H6 into a
read-off. I checked a sample of cells by hand (Mistral stock at 51 GB/s; E1 at 19 GB/s; 40 % at k = 2.2) and
they reproduce. Points where I would change a number or a rule:

1. **The H6 kill line: I disagree with moving it to 1.6 ×, and the disagreement should stay written.**
   - Her position: at k = 2.2 the column path gives 7.0 to 7.3 and falls under the stock path from 30 GB/s, so
     kill above 1.6 ×.
   - My position: that kills the lever at the only read rate this phone has shown. At 19 GB/s her own formula
     gives 6.8 to 7.0 for 40 % at k = 2.2 against 4.5 to 4.6 stock, which is × 1.5. The break-even k depends on
     r (*estimated* from her formula, Mistral, 40 %, S2): **3.5 at 19 GB/s, 2.7 at 24, 2.0 at 33, 1.7 from 39**.
   - Proposal: the kill line is not a constant. H6 reports k; the lever is closed only if k is above the
     break-even at the read rate P2 measures. If H6 must be scored before P2, keep memo-01's 2.2.
2. **Her 51 GB/s column is not a reachable rate.** Two engines used 59.1 of a nominal 76.8 on the 8 Gen 3 and the
   authors put the ceiling at about 68 (*published*, HeteroLLM). The same ratios here are about 39 and 45 GB/s
   (*estimated*). So "four threads at 46 GB/s or more" and the "15.6 to 17.2" and "14.7 to 15.9" cells sit at or
   above the bus ceiling for any reader, and three CPU threads have shown 24. I would cut the tables at 45.
3. **The drafted second column.** Her arithmetic is right and matches my round-1 §3.5. My reservation is the
   acceptance rate, not the mechanism:
   - For Gemma she assumes a = 0.5 to 0.7 with the publisher's MTP drafter, and says she found no number. That
     drafter is a second file (owner).
   - For Mistral and Qwen2.5 there is no publisher drafter. The free drafters (n-gram, prompt lookup) accept far
     less on plain generation; campaign 1's host runs lost with them. On a 7B body at a = 0.2 the same formula
     gives × 1.2 at 19 GB/s and × 0.98 at 24 (*estimated*).
   - It is exclusive with every lever that raises r (my ranks 1 and 2). I agree with "priced, gated by P2", and
     would add a second gate: a host count of a on the fixed prompt set, which costs nothing on the phone.
4. **The Gemma 4 quantisation scare (her §5) may be one tensor group, and her campaign mix makes the cheap
   choice the wrong way round.** Q3_K_M at perplexity 919 and Q4_K_M at × 3.24 on the base model, with E2B at
   × 1.10, look like a broken tensor or a conversion fault more than a model that is fragile everywhere
   (my reading; the issue closed without a cause). The lookup tables are the first suspects: `token_embd` and
   the 2,819 M-weight PLE table are read one row per token and carry a whole token's identity in a few thousand
   values.
   - **The PLE table costs zero bytes per token at any precision.** Her mix stores it at Q4_0 (1,585 MB). At
     Q8_0 it is about 3.0 GB of storage and the same 2 to 3 pages per token. Storage is not a budget here.
   - Proposal for G1: add arms that move one group at a time from Q4_0 to Q8_0 (PLE table; head; gates and
     projection; FFN; attention), 40 chunks each. Five short runs say which group carries the loss, before
     anyone concludes that the subject needs a QAT ruling or 5.5 bits everywhere.
5. **Gemma 3n's exact 95 % cut is the first bit-identical activation-sparsity lever in either round**, and it is
   under-ranked as a footnote to a same-family model. Two checks for G7: the cut is computed from the mean and
   deviation of all 16,384 gate outputs of the token, so `gate` is read whole and only `up` rows and `down`
   columns are skipped (her price already assumes this); and at `-b 512` the kept set differs per position, so
   the batched path must keep the stock code. If G7 confirms 95 %, it is also a free test bed for the column
   kernel (H6) at a density where the kernel cost cannot decide the result.
6. **O = 6 to 12 ms** is the one input of her bands with no measurement behind it on a dense model, as she says.
   R68's stock-llama.cpp reference block on the phone will give the stock-runtime node cost under power saving
   off; the first dense baseline should print the 14-op table so that O is read, not assumed.
7. Small: her §6 uses four threads as a row; R18 and R62 lost because Android's own threads take the fourth big
   core, and that does not depend on the model. I would label the row "needs X1's four-core arm" and not carry it
   into a best corner.

**Joint position, as I would sign it** (*estimated*): hers, with two edits. For a 7.1 B body nothing in either
draft changes memo-01's verdict: 60 GB/s is above the bus, and the best stack of this round is 6 to 9.5 tok/s
printed at the read rates that are physically available. For Gemma 4 E4B the exact band is 6.7 to 13.8 by read
rate; the CPU + GPU split and the column kernel are the two levers that reach the top of that band, and neither
contains 15. The first Gemma test is the per-group quantisation ladder.
