# Bonnie's review of Ebony's draft (2026-11-18)

Agent model: claude-fable-5-2. Nothing here was run on a phone or as a host benchmark. Re-pricing uses my draft's
decomposition (`bonnie-draft.md`, from the stored reports r22 to r26): clean floors Granite num 46.8 ms, OLMoE off
44.9, OLMoE num 43.3; target 38.6 ms; state penalty at the session median +2.2 to +9.5 ms (Granite) and +0.8 to
+7.4 ms (OLMoE); **0.046 ± 0.010 (Granite) and 0.034 ± 0.008 (OLMoE) ms per token per MB of wanted file pages
missing at run start** (R66, in-sample fit); `VmSwap` before the run does not predict decode time. All prices
*estimated*.

## 0. Verdict in one paragraph

A1 (`PROT_EXEC`) is the best new idea of the round and I rank it above every memory lever in my own draft; her
price is about right and my slope gives it a firmer footing, but her account of why R56 failed is overstated and
the test needs a different design. A2 and A3 are TTFT levers by my data (decode ≈ 0) and A2 must not pin the
155 MB Granite head copy that M1 would simply free. B1 has two pieces of repository evidence against it that she
did not cite, and on Granite it is the same budget as my skip-threshold item, not an addition. B2's kernel is
nearer 0.7 × than 0.5 × the full kernel by my op count. C1 can be closed in minutes without hidden states. Her
baselines are a round old (memo-21 / memo-24 figures); with the clean-run floors her closing statement changes for
OLMoE.

## 1. Baselines she should replace

| her figure | from | replace by (*device*, my draft §1) |
|---|---|---|
| fault excess 10 to 11 ms (OLMoE), 4 to 5 ms (Granite) | R60, memo-21 | session medians against clean runs: R64 +0.8 to +7.4 (OLMoE), +2.2 (Granite); R66 +7.4 and +9.5 |
| floors 46.5 (Granite num), 41.7 (OLMoE num); target 38.75 | memo-24 (fastest half of fault-free tokens) | whole clean runs: 46.8 and 43.3; OLMoE off 44.9; target 38.6 |
| re-read 0.085 to 0.125 ms/token per decode MB | memo-24 | keep, but the better predictor is missing MB at run start (slopes above); decode MB read is small (2 to 49 MB at the median) while the penalty is 7 to 10 ms |
| head 3.8 to 4.2 ms on both | R60 | Granite num 3.76; OLMoE 3.1 to 3.4 since the Q6_K repack (R62) |
| "OLMoE (41.7 ms) is the model for which 25 is arithmetically open" | | OLMoE num is 4.7 ms short as a run median (22.3 tok/s), Granite 8.2 ms; open for neither without the expert pool |

## 2. Candidate by candidate

### A1 `PROT_EXEC` mapping: agree, top of the residency list; three corrections

- **Source check.** I fetched v5.4 `mm/vmscan.c` today as well (summary through the fetch tool, short excerpts
  only): `page_check_references()` returns `PAGEREF_ACTIVATE` for a referenced file page of a `VM_EXEC` mapping,
  and `shrink_active_list()` puts a referenced `VM_EXEC` file page back on the active list
  (`if ((vm_flags & VM_EXEC) && page_is_file_cache(page))`). Her reading of upstream is right. The vendor tree is
  unverified by both of us.
- **Correction 1, R56.** "5.4 ignores the reference bit for plain file pages" holds on the active list only. On
  the inactive list a plain mapped page that is referenced gets `PAGEREF_KEEP` and is activated when seen
  referenced again. So keep-warm on plain pages was not structurally void; it lost because scans came faster than
  its 10 s passes. The exec class buys one to two extra list rotations per reference, not immunity. Her own risk
  line (bursts) says the same; the summary line should not claim more.
- **Correction 2, what it protects.** `vm_flags` reach reclaim through the rmap walk, so only **mapped** pages
  get the class. Pages the hold reads back with `posix_fadvise` are unmapped and unreferenced: they sit at the
  cold end with or without `PROT_EXEC`. Her own source reports exactly this (1.76 GB restored with `WILLNEED`
  gone in 8 s). This also corrects **me**: my draft read the hold's `stall` / `rounds` stops (advised MB at 2 to
  3 times the distinct MB) as "memory-limited". List position is the simpler reading. The lever is therefore a
  pair: the mapping flag **and** a read-back that goes through the mapping (touch one byte per page after the
  advice), in the runtime. My M5 (read the decode set back at the prefill's end and touch it) is the same
  mechanism at another moment; without the exec class its touch protects for one scan only. **Merge M5 into A1**
  as its in-run arm.
- **Correction 3, zero-sum.** If the phone cannot hold the working set at all (OLMoE 2.67 GB against 2.2 to
  2.4 GB of cache on a low day, memo-18), protecting all model pages equally protects none against each other.
  The gain exists only where someone else pays: other apps' file pages, or lmkd killing cached apps once swap
  slots are gone. That is the kernel's ordinary treatment of libraries, but 2.7 to 3.1 GB of "code" is not an
  ordinary library, and it is the same question memo-15 left to the owner ("the runtime takes memory so the OS
  kills other apps"). I do not think it breaks rule 5 (no user action is asked), but it needs the lead's and
  probably the owner's word before a claim round, not only a note.
- **Re-price with my slope.** If the MB missing at run start halves: OLMoE R66 median 332 → 166 MB × 0.034 =
  **−5.6 ms**; Granite 61 → 30 MB × 0.046 = **−1.4 ms**. On an R64-like day: OLMoE 48 MB × 0.015 → −0.4, Granite
  0. Clean runs 0. So: **OLMoE 0 to −5.5, Granite 0 to −1.5 (pressure days), about −2 and −0.5 over days.** Her
  0 to −5 / 0 to −2.5 is right for OLMoE and about 1 ms generous for Granite, whose file pages are mostly there
  already (its unexplained +4.4 ms intercept is not something A1 addresses).
- **A gain she did not price:** in the run itself the prefill streams 2.7 GB through a cache that the decode set
  must survive. Exec pages referenced by the prefill are activated at first sight, so the pages most recently
  used at the end of the prefill are harder to evict during the decode. This is where OLMoE's 14,000 to 29,000
  decode minor faults in slow runs (3,300 to 4,800 in clean ones) could fall. Unpriced; it is inside the band.
- **Test design.** Her T2 sets the flag at load, which forces a reload per arm and one arm per block. Use
  `mprotect` on the existing mapping to switch `PROT_EXEC` on and off between runs: the VMA flag changes, the
  pages stay, and the kv-style ABBA quads of r26 apply unchanged. Record `want_mb − res_before` per run (the
  kit has both) so the effect can be read against the day's pressure. Kill rule on MB missing at run start, not
  on tok/s (SE 1 to 1.7 tok/s per quad mean in R64 / R66).
- **Costs she under-counts:** moving 3.1 GB per model into internal storage is app work plus an owner step, and
  doubles storage during the copy; the i-cache flush is paid on every page faulted in, on every run after a
  wait that lost pages, not only at first load (0.2 to 1.0 GB × her < 1 µs per page ≈ 0.05 to 0.25 s of TTFT,
  acceptable).

### A2 dma-buf pin of the dense anon: a TTFT lever; narrow it further

- **Against my data:** the app's `VmSwap` before the run has no relation to decode time (slopes −0.003 ± 0.014
  and +0.007 ± 0.019 ms/token per MB, R66). The swap-ins are paid in the prefill. So the direct decode value is
  **≈ 0**, and the indirect one (slots freed → fewer of our file pages dropped) is bounded by the missing-MB
  slope: 0.2 to 0.3 GB of slots, if a third of them turned into kept file pages, is 70 to 100 MB × 0.034 to
  0.046 = −2.4 to −4.6 ms at the very most on a pressure day and 0 otherwise. I would write **decode 0 to −1 ms,
  TTFT −0.1 to −0.5 s**. Her 0 to −2 is not wrong, only wider than the evidence.
- **Do not pin what should be freed.** Granite's 541 MiB KleidiAI buffer contains the 148 MiB int8 copy of the
  tied head that the num arm never reads (memo-24 §A1). Pinning it makes 155 MB of dead weight unreclaimable.
  **M1 first** (keep the head out of KleidiAI), then A2 on the remaining 394 MiB if at all.
- **"Costs no RAM" needs T4 and one more thing:** incompressible pages in zram still leave RAM only if zram has
  a backing device (RAM Plus writeback, memo-24 §A3f, unverified). If it does, swapped dense weights do free RAM
  today and a pin takes it back.
- **Rule 5:** I keep memo-15's position for this smaller form too: it is the owner's ruling. It is far milder
  (existing memory, 0.4 GB), and I would say so to the owner, but "lmkd kills cached apps sooner" is her own risk
  line.
- **Rank:** below everything that moves the metric. With M3 (anon hold) it shares the fate "TTFT only".

### A3 derived file for the repacked dense weights: agree it is the fallback; same price as A2 minus TTFT

I had the same item in my notes and left it out for the same reason: decode ≈ 0 by the `VmSwap` result. With A1
it becomes more interesting than A2, not less (file pages get the exec class and the hold; pinned pages need no
protection but cost the ruling). If A1 passes, I would rank A3 **above** A2. Storage +0.4 to 0.55 GB is small
next to the LUT file's 2.6 to 2.8 GB.

### A4 pipe pin: crumb, agree; same 64 MiB class as `mlock`, and neither has a target until a census says which
pages go (my T1).

### B1 per-layer skip table: weaker prior than she gives, and on Granite not additive

- **Repository evidence against it, not cited:**
  1. Campaign 1 swept static per-layer expert counts on Granite: 4 in the lower half + 6 in the upper +1.95 %,
     the reverse +3.63 %, uniform 5 +1.26 %: "**falsified**: uniform beats both static splits"
     (`reports/05_compute_less.md` row 6 and §"Falsified today", *host*). Two coarse splits are not a fitted
     table, so this does not close B1, but the one fact that survived was only an ordering.
  2. R51 on OLMoE: two different rule forms (`SUM`, `MASS`) "lie on the same curve"; the prediction that one
     would be cheaper at equal bytes missed (*host*).
  Both say the PPL-per-byte frontier on these models is not easily bent by re-allocating the same skip. She
  should say why a sensitivity-fitted table differs (finer, per-layer KL rather than a hand split).
- **Price.** Her 1.9 to 4.3 ms multiplies bytes by the op time. The realised gain of `MASS=900` in the op time
  is −1.9 to −2.3 ms for 13.04 % of decode bytes (*device*, my draft §0.4), 55 to 60 % of the byte arithmetic
  (Granite's skip realises about 90 %). At that rate 20 to 27 % instead of 13.4 % is −1.0 to −2.2 ms: **OLMoE 0
  to −2.4, centre −0.5 with P(frontier ≥ 1.3 ×) about 0.25**. Why OLMoE realises only 55 to 60 % is itself a
  question for Tom (fixed per-expert cost, or the skipped experts being the cheap ones).
- **Granite: the same budget as my A9.** Granite has 0.176 PPL left at `-ub 1` (9.6879 − 9.5120). My A9 (flat
  threshold 130 → about 141, −1.2 to −1.6 ms) and her B1 (0 to −1.5) spend that same 0.176; they are one
  frontier, and A9 is the flat point on it. **One host test, not two:** screen the flat thresholds first (no
  code, one night); build the per-layer table only if the sensitivity sweep shows layers differing by more than
  the flat frontier's slope.
- **OLMoE budget:** with `MASS=900` 0.203 is left; the prior (M2) takes 0.148 of it (12.2454). B1 replaces
  `MASS=900` rather than adding to it, so "beyond `MASS=900`" must be read as "instead of it, at the same PPL".
- Her note that `MASS=900`'s device result is a σ problem: agreed, and the op times show it directly; but her
  "about −4.5 ms" from the tok/s central value is twice what the profile shows.

### B2 MSB-only tier on Q3_K's `hmask`: kernel cost optimistic; park behind B1's result

- **Op count.** Memo-21 §A1 puts the full kernel at 0.37 NEON ops per weight. An `hmask`-only row needs, per 16
  weights, a bit-to-byte expansion (replicate, mask, test: 3 ops) and one `sdot`, plus the same 6-bit sub-scale
  handling per 16-weight block: about 0.25 to 0.30 ops per weight, **0.7 to 0.8 ×**, not 0.5 ×. Bytes read fall
  to 42 % (memo-21 §g), but the expert op is compute-bound, so bytes do not help.
- **Price.** 20 % of the remaining expert mass at 0.7 to 0.8 × of 26.8 ms is −1.1 to −1.6 ms **if that tier
  costs no PPL**, which it will not: one bit per weight inside the block scales is a far coarser expert, and the
  only evidence near it is that 1-bit and Q2_K screens failed badly as head screens (R28, R29), a different use.
  **OLMoE 0 to −1.5, Granite 0 to −0.7**, a new NEON kernel (M to L), the same budget as B1 / A9 / M2 / lossy
  LUT. It is the fifth claimant on a budget of 0.176 to 0.203.
- Keep the emulation in the host frontier run only if it is nearly free for Tom; do not build the kernel before
  B1 and A9 have reported.

### C1 certified head: close it with a ten-minute check, not T7

Her doubt is right and can be made quantitative without hidden states. The bound opens a cluster when
`c·h + r|h|` exceeds the current best logit. In 1,536 to 2,048 dimensions with 25 rows per cluster, k-means
removes little of the row norm (for near-isotropic rows about 1/25 of the variance, so r ≈ 0.98 |w|), while a
winning logit is |w||h|cos θ with cos θ well below 1. Then r|h| alone exceeds the winner and every cluster opens.
**Shorter test:** k-means the dequantised head rows (C = 2,000) and print median r / |w|; no model run. Close C1
if it is above about 0.5. I expect > 0.9; P(pass) below 0.05. R43's recall of 0.87 for a low-rank screen points
the same way. Two further costs if it somehow passed: no logits (the PPL gate cannot run through it, and only
greedy sampling could use it), and on Granite the head is already a two-stage op at its screen floor (3.76 ms),
so the saving is against the screen, not against a full head.

### D to G: agree

- D (speculation): agree, R20 stands; nothing new.
- E: her T6 is free and worth doing; note that the per-token `mhz7` field in the stored profiles drops to 844 to
  1,785 on some tokens of slow runs (r22 pos 23 and 35, r23 pos 36) while `mhz4` stays 2,112: the X1 was idle or
  slow at those samples. That bears on my helper item (the fourth big core is free at least sometimes) and on
  pinning (the workers were on the three A78s then).
- F (Hexagon): agree, dead as shipped; unpriced long shot noted.
- G (MNN's unequal split): agree it is my pinning item; an unequal static share for the X1 is a useful arm.
- Her T8 (double caching through the external path) is free and should run first: if `Cached` grows 2 × the
  model's resident MB, moving the file is the first lever regardless of `PROT_EXEC`.

## 3. Where I was wrong or thin (found while reviewing)

1. **"The hold is memory-limited"** (my §1.3): not established. Her source and the kernel code give a simpler
   reading: read-back pages are unmapped and cold, so they are the first to go. The merged memo should say
   "limited by memory or by list position; A1's test tells which".
2. **M5's touch**: on plain pages it protects for one scan; I priced it as if touching activated. It belongs
   under A1.
3. **M3 / R66's "next lever"**: I showed decode ≈ 0; her A2 shows the same swap-ins can be removed rather than
   pre-paid. Both are TTFT levers; neither should be next for the metric.
4. I did not check the storage path of the model at all; her T1 and T8 do.

## 4. Budget table for the numerics claimants (one budget per model, absolute PPL)

| | Granite (left 0.176 at `-ub 1`, KleidiAI build) | OLMoE (left 0.203 with `MASS=900`; 0.586 without) |
|---|---|---|
| my A9 flat threshold | spends all of it for −1.2 to −1.6 ms | n/a (`MASS=900` is the flat point, −1.9 to −2.3 ms measured in op time) |
| her B1 per-layer table | same budget; replaces A9 if its frontier is better | replaces `MASS=900` at the same PPL; 0 to −2.4 more |
| M2 routing prior | 0 ms, costs 0.031 to 0.036 | 0 to −3 on pressure days, costs 0.148 |
| her B2 MSB tier | 0 to −0.7 | 0 to −1.5 |
| lossy LUT | −3 to −6, PPL unknown | −5 to −11, PPL unknown |

Only one of these can take the remaining budget on Granite. Order I propose: flat-threshold screen (free), then
the per-layer sweep, then decide; the lossy LUT only if its microbench survives and then it competes on ms per
PPL point with whatever the frontier shows.

## 5. Still disagreed (to be written into the merged memo with both positions)

1. **A2 and rule 5.** Ebony: existing incompressible memory, not a new claim, no ruling needed beyond the lead.
   Bonnie: owner ruling, as in memo-15; and price decode at 0 to −1, not 0 to −2.
2. **A1's account of R56.** Ebony: structural for plain pages. Bonnie: timing; exec buys rotations. Same test
   decides; the wording differs.
3. **B1's prior.** Ebony: 0 to −4 on OLMoE. Bonnie: 0 to −2.4 with P about 0.25, given campaign 1's falsified
   static split and R51's coinciding rule forms.
4. **C1.** Ebony: one host hour. Bonnie: a row-norm check of minutes closes it.
