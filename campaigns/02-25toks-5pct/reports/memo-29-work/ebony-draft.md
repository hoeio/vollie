# Ebony draft: frontier candidates for the joint round (2026-11-18)

Agent model: claude-fable-5-2 (stand-in for `ebony`). Draft for Bonnie's review; not committed, no tracked file
edited. **Nothing here was run on the phone or on the host.** Labels: *device* = read from a kit report or memo
that quotes one (named each time), *host*, *estimated* = my arithmetic, "source" = a published claim not
reproduced here. Where I only saw an abstract or a search summary, I say so. Verdicts and pre-registrations are
the lead's.

Baselines used for pricing (all quoted, none new): fault excess per token in hold runs about 10 to 11 ms (OLMoE)
and 4 to 5 ms (Granite) (*device* arithmetic, memo-21 §0.3, R60); pages lost in a gate wait 506 to 954 MB at the
median, hold restores 84.5 % at the median (*device*, R66); file re-read in the decode costs 0.085 to 0.125 ms
per token per MB (*estimated* fit, memo-24); expert op OLMoE 27 to 31 ms, Granite num 13.5 to 13.9 ms, both
compute-bound; dense ops byte-bound (*device*, memo-21 §0); head 3.8 to 4.2 ms on both (*device*, R60);
fault-free floors 46.5 ms Granite num, 41.7 ms OLMoE num (*estimated*, memo-24); target 38.75 ms.

## 0. Summary

Five things I think are new for this repository and worth a decision; everything else in §2 is a "checked, stays
closed" note so nobody has to search it again.

1. **A1. Map the model `PROT_EXEC`.** Linux 5.4's reclaim treats referenced file pages of an executable mapping
   as a protected class (source read: `mm/vmscan.c` v5.4, both code sites quoted in §1). Bit-identical, one flag,
   generic. It changes what campaign 2's closed keep-warm (R56) and the hold can achieve. Needs the model file in
   internal app storage (the external files folder is a `noexec` mount). Price: 0 to −5 ms (OLMoE), 0 to −2.5 ms
   (Granite) on gated runs, 0 in fault-free runs (*estimated*).
2. **A2. Pin the dense anonymous weights the app already holds (KleidiAI/repack buffers, 0.4 to 0.55 GB) in a
   dma-buf (`AHardwareBuffer` BLOB).** A narrow reopening of memo-15's "priced, not proposed" full-expert pin: no
   new memory is taken, because quantised weights do not compress in zram anyway. Bit-identical. Price: TTFT −0.1
   to −0.5 s, decode 0 to −2 ms (*estimated*).
3. **B1. Layer- and token-aware expert count** (Ban&Pick, EAC-MoE) instead of one mass threshold for all layers.
   Declared numerics, same rule at `-b 512` and `-ub 1`. Price: OLMoE 0 to −4 ms beyond `MASS=900`, Granite 0 to
   −1.5 ms (*estimated*); host frontier first.
4. **B2. A cheap-precision tier for low-weight experts using Q3_K's own `hmask` plane** (SliceMoE's bit-sliced
   idea with no new file). Declared numerics, new kernel. Price: OLMoE 0 to −3 ms, Granite 0 to −1.5 ms
   (*estimated*); host PPL emulation first.
5. **C1. Certified sub-vocabulary head** (CSV-Decode): exact greedy/top-k token with a bound, no full projection.
   My prior is that the bound is too loose in 1,536 to 2,048 dimensions (P(pass) about 0.2); one host hour closes
   it either way. Price if it passes: 0 to −2.5 ms on both.

Speculative decoding (five 2025-2026 MoE papers), the Hexagon NPU backend, `MADV_POPULATE_READ`, userfaultfd,
MGLRU, THP and ADPF hints: checked, each stays closed or is a zero-cost check (§2.D to §2.G).

## 1. Facts I verified in sources

| fact | value | source |
|---|---|---|
| 5.4 reclaim, inactive list | a mapped file page with a young PTE is `PAGEREF_KEEP` on first sight and activated on the second; **if the mapping is `VM_EXEC` it is activated on first sight** ("Activate file-backed executable pages after first usage") | `page_check_references()`, [torvalds/linux v5.4 mm/vmscan.c](https://raw.githubusercontent.com/torvalds/linux/v5.4/mm/vmscan.c), fetched 2026-11-18 |
| 5.4 reclaim, active list | referenced active file pages are deactivated like unreferenced ones, **except `VM_EXEC` file pages, which go back on the active list** ("so that executable code get better chances to stay in memory under moderate memory pressure") | `shrink_active_list()`, same file |
| origin and caveat | Wu Fengguang 2009; the changelog itself calls `PROT_EXEC` on data an abuse a home-made app may choose; reviewers: not a guarantee, only extra rounds | [lkml 2009-05](https://lkml.iu.edu/hypermail/linux/kernel/0905.2/01048.html) (search summary only) |
| SELinux, Android 15 | `allow untrusted_app_all app_data_file:file { r_file_perms execute };` followed by an `auditallow` (it logs) | [sepolicy android-15.0.0_r1, untrusted_app_all.te](https://android.googlesource.com/platform/system/sepolicy/+/refs/tags/android-15.0.0_r1/private/untrusted_app_all.te), fetched |
| where the model is | the app's **external** files folder (`Android/data/<pkg>/files/`) | memo-09 line 38, memo-10 line 164 (repository) |
| dma-buf pin, unprivileged | `AHardwareBuffer` BLOB pages are pinned, bypass `RLIMIT_MEMLOCK`, 2,047 MiB per buffer (lock fails at 2^31), read bandwidth 1.00x anon; A/B "dense weights pinned" +17.9 % decode (2.588 to 3.053 tok/s), order effect not excluded, one device | [BigMoeOnEdge android-memory.md](https://raw.githubusercontent.com/Helldez/BigMoeOnEdge/master/docs/android-memory.md), dated 2026-07-21, **Android 16, kernel 6.12, 12 GB phone, gpt-oss-120b**; one informal source, already cited in memo-15 |
| same source, restore | 1.76 GB restored with `MADV_WILLNEED` in 6.4 s was reclaimed again within 8 s: "restored pages land on the inactive list" | same |
| phone | kernel `5.4.274-qgki`, `RLIMIT_MEMLOCK` 64 MiB, THP field empty | *device*, memo-24 §A3 |

Not verified: that Qualcomm's `qgki` 5.4 tree keeps these two `vmscan.c` sites unchanged (vendor hooks, a
vendor zram/reclaim module: BigMoeOnEdge reports one on its phone); the mount options of the internal and
external app folders on this phone; ION/gralloc BLOB limits on a 5.4 kernel (the 2,047 MiB figure is from 6.12).

## 2. Candidates

### A. Residency and kernel memory

**A1. `PROT_EXEC` mapping of the model (new; I found no write-up of it for model weights).**
- Source: kernel code above (2009 heuristic, present in v5.4). Idea is mine; the search for prior use on model
  files returned nothing.
- What it means here: add `PROT_EXEC` to the model's `mmap` behind an environment variable. Then (i) a page the
  hold has read back and that one compute or touch pass has referenced is activated at the first reclaim scan
  instead of being kept one round; (ii) once active and referenced it stays active, where a plain file page is
  deactivated regardless of use. Campaign 2's keep-warm (R56: 11.69 against 11.72 tok/s, pages still lost between
  10 s passes, closed) touched plain file pages, for which 5.4 ignores the reference bit on the active list. So
  the reopening is not "try keep-warm again" but "the same touch on a different page class". By my reading a
  touched plain page survives one scan round, a touched exec page three stages (inactive, active, inactive).
- Conditions: the file must sit on a mount without `noexec` with the `app_data_file` label, that is the
  internal files folder, not `Android/data/...` (expected `EPERM`/`EACCES` there). So the kit needs the download
  target moved to internal storage; the bytes and hash of the served file do not change. The `auditallow` writes
  log lines. arm64 flushes the instruction cache once per newly read exec page (*estimated* below 1 µs per page,
  about 0.7 s over 680 k pages on a first full load; to be measured as TTFT).
- Different file: no. Numerics: bit-identical (protection bits only). Generic: any model, any mmapped runtime.
- Price (*estimated*): it can only remove fault excess. If the wait loss halves, OLMoE 0 to −5 ms and Granite 0
  to −2.5 ms per token on gated runs; nothing on a fault-free run. A second gain if it works: the 2 s of hold
  CPU before each run (R62) may no longer be needed (heat).
- Risk: under a burst (an app restart allocating 0.5 to 1 GB in a second) several scan rounds fit between two
  touches and the protection is used up; vendor reclaim changes. It also shifts pressure to other apps' file
  pages, which is what the kernel does for every shared library; I do not see a rule-5 problem, but it is the
  lead's call.

**A2. dma-buf pin for the dense anonymous weights (narrow reopening of memo-15 §4).**
- Source: BigMoeOnEdge (above), its "dense weights pinned" mode.
- Why it is not the proposal memo-15 declined: that was 2.73 GB of experts, new unreclaimable memory, "changes
  the victim". This is the 0.4 to 0.55 GB of repacked dense weights and head copy the app holds as anon today
  (memo-24 §A1: Granite 541 MiB KleidiAI buffer, OLMoE less). Those bytes are 4-bit-packed and close to
  incompressible, so when kswapd pushes them to zram they still occupy about the same RAM and additionally a
  swap slot each; memo-24 found SwapFree below 50 MB at 28 of 48 run ends and, with no slot free, reclaim can
  only drop file pages, ours included. Pinning them costs no RAM the phone had usefully recovered and returns
  0.2 to 0.3 GB of slots (*estimated* from the VmSwap medians 216 to 294 MB, memo-24).
- What it means here: allocate the CPU backend's repack/KleidiAI weight buffer from `AHardwareBuffer_allocate`
  (BLOB, CPU read/write usage), lock once, keep for the model's life. On 5.4 this goes through ION, not dma-buf
  heaps (not verified on this phone).
- Different file: no. Numerics: bit-identical (same bytes, other backing). Generic: every model with repacked
  dense tensors.
- Price (*estimated*): TTFT −0.1 to −0.5 s (the same swap-ins memo-24 prices for the anon hold, removed instead
  of pre-paid); decode 0 to −2 ms through slots and residency, unmeasured. The +17.9 % of the source is a
  larger-than-RAM model on another kernel and does not transfer.
- Risk: lmkd sees less reclaimable memory and kills cached apps sooner (the app itself is foreground). Bonnie
  should price whether that is a rule-5 question; my position is that it is not new memory, hers in memo-15 was
  about the full copy.

**A3. Alternative to A2 if a pin is refused: write the repacked dense weights once to a derived file and mmap
it.** Clean file pages need no swap slot and are dropped for free; with A1 they get the exec class too. Storage
+0.4 to 0.55 GB (owner question, far below the 2.6 to 2.8 GB of memo-21's LUT file); COMMON_RULES §1 counts a
derived artefact as runtime. Bit-identical. Also removes the repack from every cold load. Price: as A2 for the
slots, minus the TTFT part (file re-reads replace swap-ins). I rank it below A2.

**A4. Crumb: `splice` of page-cache pages into pipes holds a page reference that reclaim cannot break.** Capped by
`pipe-user-pages-soft` (64 MiB by default, not verified on the phone), the same order as `mlock`'s 64 MiB, and
additive to it. Bit-identical. 0 to −0.5 ms. Only after A1.

**Checked and dead here:** `MADV_POPULATE_READ` (Linux 5.14; the phone is 5.4.274); userfaultfd (5.4: anonymous
and shmem only, not file pages); `process_madvise` on other apps (needs privileges); MGLRU (not in 5.4; vendor
disabled even on 6.12 per BigMoeOnEdge); THP/`MADV_HUGEPAGE` (reported `[never]` on the 6.12 phone, field empty
here); cgroup `memory.min/low` (memcg v1); `oom_score_adj`/foreground service (kill order only, memo-24 agrees).
`MADV_COLD/PAGEOUT` on own pages is the lever BigMoeOnEdge calls the only working one; campaign 2 already
tried its form (`GGML_PREFILL_COLD`, R58, unresolved and closed), I do not reopen it.

### B. Experts (the only compute pool)

**B1. Layer- and token-aware expert count.**
- Sources: Ban&Pick, [arXiv 2509.06346v2](https://arxiv.org/html/2509.06346v2), 2025-09-30 (page read): per
  token and layer K = round(K_min + (K_base − K_min)·S), S from a per-layer sensitivity (KL of the logits when
  that layer alone goes from top-8 to top-3, on a calibration set) and a per-token top-3/top-8 mass ratio;
  Qwen3-30B-A3B runs at 4.82 of 8 experts on average, 1.25x under vLLM. EAC-MoE,
  [ACL 2025](https://aclanthology.org/2025.acl-long.633) (abstract only): prunes rarely selected experts at
  inference.
- Weak points of the sources: task accuracy only, no perplexity; OLMoE is cited but **not tested**; one result
  drops 2.44 points (HumanEval+). GPU serving, not a phone.
- What it means here: the c2 skip rule is one mass threshold for all layers (`MASS=900`: 13.39 % of expert bytes
  for +3.32 %, *host*, R51). Replace it by a per-layer threshold table from a sensitivity sweep on text that is
  not the test set (wikitext-2 train split), stored as a small derived table or as constants per model card.
  The token part of Ban's rule is what `MASS` already does.
- Different file: no. Numerics: declared; a per-token rule, so identical logic at `-b 512` and `-ub 1` (the gate
  at both still applies). Generic: any top-k MoE; the table is per model, the code is not.
- Price (*estimated*): the expert op is compute-bound, so bytes skipped are milliseconds. If the same +3.3 %
  buys 20 to 27 % instead of 13.4 %, OLMoE gains a further 1.9 to 4.3 ms on a 27 to 31 ms op: **0 to −4 ms**.
  Granite already skips more (21.2 to 13.7 ms at skip 130): **0 to −1.5 ms**.
- Note for Bonnie: OLMoE's `MASS=900` is "not shown" in three device rounds (R64: +2.02 ± 1.09 tok/s). At 20
  tok/s that central value is about −4.5 ms, the size memo-03 predicted; the problem looks like σ, not absence.
  B1 does not fix σ; it needs the fault-free state as much as the existing rule does.

**B2. Bit-sliced precision by router weight, on Q3_K's own planes (mine, after SliceMoE).**
- Source: SliceMoE, [arXiv 2512.12990v3](https://arxiv.org/html/2512.12990v3), 2026-03-12 (page read): experts
  stored as MSB and LSB slices; "critical" experts use both, the others the MSB slice; perplexity 7.11 against
  7.06 at its 4-bit setting; simulated accelerator with LPDDR4 and UFS, not a phone CPU; it uses its own nested
  quantisation (AMAT), which would be a different file.
- What it means here without a new file: a Q3_K weight is a 2-bit field plus one `hmask` bit that selects the
  −4 offset, so `hmask` is already an MSB plane inside the served file. A three-tier rule per expert and token:
  full kernel above a weight threshold, **MSB-only kernel** (sign plane and block scales; the 2-bit field
  replaced by its mean) in a middle band, skip below. The middle kernel reads 46 of 110 bytes per block and
  needs roughly half the NEON operations (*estimated* from op counts: masked sum against decode plus `sdot`).
- Different file: no. Numerics: declared. Generic: every Q3_K expert tensor (both models).
- Price (*estimated*): only useful if an expert at MSB precision hurts PPL much less than a skipped one. If 20 %
  of OLMoE's remaining expert mass moves to the half-cost tier at no PPL cost, −2.7 ms; range **0 to −3 ms**
  OLMoE, **0 to −1.5 ms** Granite. Pages touched do not change (the planes share pages), so no residency gain.
- This competes with B1 for the same PPL budget; one frontier run should carry both (test T5).

**Checked, not proposed:** llama.cpp's 2026 MoE work is for GPU offload (expert cache in VRAM, PR #29887;
CPU-run cache misses, PR #30966; a `mul_mat_vec_q_moe` kernel for batch above one, PR #20905; next-layer weight
prefetch to the GPU, [blog 2026-04-28](https://am17an.bearblog.dev/prefetching-weights-in-llamacpp/)): **search
summaries only, PR pages not opened**. None changes single-token CPU decode; the batch kernel could matter for
TTFT, which the metric does not count. Cache-conditional routing (arXiv 2412.00099) is memo-25, done.
ik_llama.cpp's row-interleaved `_R4` types: I found R4 forms for the IQ*K types and no evidence of a Q3_K_R4
NEON kernel (search summary); Vollie's 4-row in-register kernel already is that idea. No action.

### C. Head

**C1. Certified sub-vocabulary head.**
- Source: CSV-Decode, [arXiv 2511.21702v1](https://arxiv.org/html/2511.21702v1), 2025-11-16 (page read):
  k-means over the output rows (2,000 clusters), Cauchy-Schwarz bound per cluster (centroid·h + radius·|h|),
  open only clusters whose bound exceeds the current k-th logit; exact top-k when certified, full projection
  otherwise. Reported: 18 to 19 % of the vocabulary opened, fallback 1.2 %, 2.7x and more **on A100 GPUs**; the
  paper's own numbers are not consistent between tables and it reports no CPU result.
- What it means here: the benchmark is greedy and the app's sampler fast path already consumes an argmax. A
  load-time k-means over the served head rows (derived, in memory; rows of the existing repacked copy permuted
  by cluster, so no extra bytes) and a head op that returns the certified argmax or top-k logits. The bound is
  computed on the same quantised activations, so certified means the token equals the full head's.
- Different file: no. Numerics: token-identical when certified, fallback otherwise; the logits vector is not
  produced, so `llama-perplexity` cannot run through it. **Rules question for the lead:** identity would be shown
  by argmax equality at every position of the test set and the PPL tool keeps the full head. Top-p/min-p chains
  need the denominator and cannot use it.
- My doubt, written down so Bonnie can attack it: in 1,536 to 2,048 dimensions k-means over 50 k rows removes
  little of the row norm, so radius·|h| is of the order of the largest logit and the bound rarely certifies. The
  repository's own evidence points the same way (low-rank head screen: recall 0.87, closed, R43). I expect it to
  fail on these models and would not build before the host test.
- Price if it passes (*estimated*): centroids 4 % plus 19 % of the rows against a 3.8 to 4.2 ms head: −2.5 ms at
  best on both; 0 if the opened share is above about 60 %.

### D. Speculative decoding: re-checked, stays closed

2025-2026 MoE papers (all **abstracts or summaries only**): MoESD
([arXiv 2505.19645](https://arxiv.org/abs/2505.19645)): MoE gains more than dense at medium batch sizes on GPU
servers. SpecMoE ([2604.10152](https://www.alphaxiv.org/abs/2604.10152)) and S2-MoE
([2608.15018](https://arxiv.org/pdf/2608.15018)): self-drafting from parts of the target. DraftExpert
([2607.24434](https://www.alphaxiv.org/abs/2607.24434v1), 2026-07-27, page read): 1.45x, but with **trained**
draft experts (an extra file) and only where experts are loaded on demand; it states that verifying a block
touches the union of the target experts. ELMoE-3D (search snippet): verification loads experts even for rejected
tokens. Every reported gain comes from avoided expert transfers or idle batch capacity. R20 measured the
opposite regime on this phone (0.48 to 0.76x, compute-bound above batch one), and a block of k tokens costs k
times the expert op here. No new evidence against R20; a draft with fewer experts and no verification is just
the skip rule.

### E. Clock and heat

- ADPF `PerformanceHintManager` was campaign 1's C6 (reports/04_campaign.md line 181) and I find no measurement
  of it in the R-log. The kits log `scaling_max_freq` as the cap (2,112 / 2,035 MHz against hardware maxima
  2,419 / 2,841): the limit is thermal mitigation, which a hint session does not lift. Zero-cost check T6 below;
  otherwise price 0.
- Sustained-load papers ([arXiv 2603.23640](https://arxiv.org/pdf/2603.23640), 2026-03, abstract only): thermal
  and power limits, not peak compute, set sustained phone throughput. Consistent with the R-log, no lever in it.

### F. Other processors

- llama.cpp's Hexagon NPU backend (PR #16547, merged, experimental) lists Hexagon v73, v75, v79, v81 (search
  summary of the docs). The SD888's Hexagon 780 is older (v68, from my memory, not checked). Dead as shipped.
- Long shot, not priced: HVX through an unsigned FastRPC domain as a "fourth worker" that is not one of the big
  cores (four plain threads lost on both models, R62). Needs the Hexagon SDK, shared ION buffers and a Q3_K
  kernel for HVX; nothing known about its rate here. I list it only so it is on record as unexamined.

### G. Other runtimes

MNN-LLM ([arXiv 2506.10443](https://arxiv.org/pdf/2506.10443), 2025; search summary): per-core load shares set
at start-up from measured capability, then rebalanced. That is Bonnie's per-thread pinning item (memo-24 §B4.2)
with an unequal static split for the X1; no separate candidate. I found no 2026 release note of MNN, ExecuTorch
or mllm with a memory technique beyond mmap.

## 3. Draft TEST list (decides a question; cost; what closes it)

| # | test | where | cost | closes when |
|---|---|---|---|---|
| T1 | Does the phone accept a `PROT_EXEC` file mapping, and where? Kit line: `mmap` 1 MB of the model with and without `PROT_EXEC` from the external and the internal files folder, print errno; print the two `/proc/self/mountinfo` rows | device, rides on any kit | about 30 lines of app code, no runtime change | internal folder refuses too: A1 dead on this phone |
| T2 | A1 itself: model in internal storage, `VOLLIE_MMAP_EXEC=1` against 0 in ABBA over gate waits, hold on in both arms, plus one arm with the R56 touch helper at a 2 s period; read MB lost per wait (mincore before and after), decode `ru_inblock`, TTFT | device | one mmap flag behind an env var; download path change; reuse `0001-keep-warm` | median wait loss not down by 25 % in any exec arm: closed. TTFT up by more than 0.5 s: closed |
| T3 | dma-buf feasibility: allocate one 512 MiB `AHardwareBuffer` BLOB, lock, time a 3-thread read pass against anon, hold it through one gate wait, report RssAnon/VmSwap and whether the app survives | device, kit line | small JNI/NDK test, no runtime change | allocation or lock fails, or read rate below 0.95x anon: A2 dead, A3 becomes the candidate |
| T4 | Is the dense anon really incompressible in zram? Host: compress the repacked buffers page by page with LZ4 (the usual zram algorithm; the phone's is not verified) | host | under an hour | ratio above about 1.5: A2's "costs no RAM" argument fails, re-price |
| T5 | Expert numerics frontier, both models, 60 chunks then full set at `-b 512` and `-ub 1`: (a) per-layer sensitivity sweep and a per-layer threshold table fitted on wikitext-2 train; (b) MSB-only tier emulated in the expert op (scalar, slow is fine) | host | about a day of CPU; emulation patch off by default | (a) skipped bytes at the budget below 1.3x today's (OLMoE 17.4 %): B1 closed. (b) at equal PPL, op-count cost above skip-only minus 1 ms: B2 closed |
| T6 | Read from stored reports (r23 to r26): is `scaling_cur_freq` below `scaling_max_freq` during decode? | desk | minutes | never below: ADPF hint priced 0 and closed |
| T7 | Certified head on real hidden states: dump the final hidden state at 2,000 positions per model, k-means the dequantised head rows (C = 500, 2,000, 8,000), count certified positions and opened rows | host, numpy | an hour or two | median opened share above 60 % or fallback above 20 %: C1 closed |
| T8 | Is the model path double-cached? From stored reports: growth of `Cached` against model MB resident at first load | desk | minutes | growth about 1x (what the R60 figures suggest): no issue. About 2x: moving the file to internal storage becomes the first lever |

Order I would run them: T1, T6, T8 (free), T4 and T7 (an afternoon on the host), T3 with the next kit, T5 when
the host CPU is free, T2 once T1 says yes.

## 4. Draft TRY list (levers to build; all generic across both models)

| # | lever | label | first price, per token unless said (*estimated*) | needs |
|---|---|---|---|---|
| Y1 | A1 `PROT_EXEC` model mapping (+ the existing touch helper) | bit-identical | OLMoE 0 to −5 ms, Granite 0 to −2.5 ms on gated runs; 0 fault-free | T1 yes; model in internal storage |
| Y2 | A2 dense anon in a pinned dma-buf | bit-identical | TTFT −0.1 to −0.5 s; decode 0 to −2 ms | T3, T4; a lead/owner view on rule 5 |
| Y3 | A3 derived file for repacked dense weights (only if Y2 is refused) | bit-identical | as Y2 without the TTFT part; +0.4 to 0.55 GB storage | owner: storage |
| Y4 | B1 per-layer skip table | numerics, declared; gate at `-b 512` and `-ub 1` | OLMoE 0 to −4 ms beyond `MASS=900`; Granite 0 to −1.5 | T5 (a) |
| Y5 | B2 MSB-only expert tier | numerics, declared; gate at both | OLMoE 0 to −3 ms; Granite 0 to −1.5; shares the budget with Y4 | T5 (b); a new NEON kernel |
| Y6 | C1 certified head | token-identical, no logits; rules question | 0 to −2.5 ms on both | T7; lead's ruling on how identity is shown |
| Y7 | A4 pipe pin, 64 MiB | bit-identical | 0 to −0.5 ms | after Y1 |

What I expect to generalise: Y1 to Y3 and Y7 are properties of the phone and the kernel, not of a model. Y4 and
Y5 share code but their tables and gains are per model. Y6 depends on each head's geometry.

Honest position on the goal: none of these changes memo-24's arithmetic for Granite (46.5 ms fault-free, 38.75
needed). Y1 to Y3 move runs toward the fault-free floor; only Y4, Y5 and Y6 lower the floor, by at most about
−4 (Granite) and −8 ms (OLMoE) if everything passes at its top, which it will not. OLMoE (41.7 ms) is the model
for which 25 tok/s is arithmetically open.

## 5. Open points and likely disagreements for Bonnie

1. A2 against memo-15: is pinning existing incompressible anon "taking memory by force"? I say no (T4 decides
   the premise); memo-15 priced the full-expert copy and declined to propose it.
2. A1 against R56: I claim the keep-warm failure was structural for plain file pages on 5.4 and that the exec
   class changes it; the counter-argument is burst reclaim (several rounds between touches). Only T2 settles it.
3. B1's price assumes the PPL-per-byte frontier is uneven across layers on OLMoE. The sources show that on
   Qwen3 and DeepSeek for task accuracy, not for OLMoE or perplexity.
4. C1: I am proposing a test I expect to fail; if Bonnie can show from R43's data that it must fail, drop T7.
5. Not verified by me: all llama.cpp PR numbers in §2.B (search summaries); the vendor kernel's `vmscan.c`; the
   Hexagon version of the SD888; `pipe-user-pages-soft` and mount options on the phone; whether `AHardwareBuffer`
   BLOB is CPU-cached on this phone's gralloc.
