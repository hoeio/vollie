# Ebony's review of Bonnie's draft (2026-11-18)

Agent model: claude-fable-5-2 (stand-in for `ebony`). Not committed. I re-ran two things with my own short
scripts over the stored reports (`joint/ebony_check.py`, `joint/ebony_cap.py`); everything I computed is a
*device*-data, in-sample, post hoc reading, exactly like the fits it checks. Nothing ran on the phone or as a
host benchmark. Where I disagree, both positions are written down.

## 0. Verdict in six lines

1. The decomposition (§1.1, §1.2) is sound and is the best account of a clean run the repository has. Keep it.
2. The missing-MB finding (§0.3) reproduces. **For OLMoE it is robust; for Granite it rests on two runs.**
3. The TRY ranking prices A8 (`MASS=900`) by the expert op alone; by the draft's own table the net is about half.
4. A5 (strict per-thread pinning) is ranked second on a clean-state argument; on a pressure day it can lose.
5. A9 and my B1 are one question and should be one host run; A9 alone spends Granite's whole remaining budget.
6. A pool is missing from the gap table: **the clock cap reads 2,035 at every heat from 0.67 to 1.08.** If that
   ceiling is not thermal, it is worth as much as the whole TRY list on OLMoE. My own draft priced it 0; I
   withdraw that (§4.4).

## 1. Decomposition and floors

- Agreed: clean whole-run floors (Granite num 46.8 ms, OLMoE off 44.9, OLMoE num 43.3) and the correction that
  memo-24's OLMoE figure was a fast-half-of-tokens number. My draft used 41.7 for OLMoE num; the merged memo
  should use 43.3 and I change my "arithmetically open" sentence accordingly: OLMoE num needs −4.7 ms clean.
- Caveat to state in the memo: the clean runs are selected on the outcome's neighbours (few faults, < 5 MB
  read, cap 2,035). They are a floor **conditional on the 2,035 cap**, not a floor of the phone (§4.4).
- "OLMoE num 21.4 to 23.5 (7 runs, R64 only)": one session. R66 ran OLMoE off only, so the num floor has no
  second day. Say so next to the number.
- The dense excess in the skip arm now shows on **both** models in the draft's own table: Granite +0.8 to
  +1.3 ms, OLMoE other dense 7.34 / 7.38 (num) against 6.86 (off), +0.5 ms. The draft treats it as a Granite
  puzzle. Two models with the same sign make it a property of "less expert work before the dense op", which
  is what the trust purpose asks us to notice. It also means every skip lever (A8, A9, my B1, B2) pays back
  roughly 15 to 25 % of its expert-op gain in the dense ops until the cause is found. T4 should include one
  in-run A/B (`GGML_CPU_AB`, skip on against off per token) reading the other-dense time.

## 2. The fits

Re-run on r26 (R66), uncapped gated runs at 3 threads, decode wall per token against `want_mb −
res_before.total_mb`:

| model | n | slope ms/token per MB | r | Spearman | leave-one-out slope | without the 2 largest-missing runs |
|---|---|---|---|---|---|---|
| Granite num | 17 | 0.046 | 0.79 | 0.62 | 0.037 to 0.056 | **slope 0.020, r 0.41** |
| OLMoE off | 19 | 0.034 | 0.71 | 0.78 | 0.030 to 0.044 | slope 0.046, r 0.78 |

- OLMoE: reproduced and stable under every cut I tried. I accept "decode cost tracks missing file pages".
- Granite: the slope halves and r falls to 0.41 when the two runs with the most missing MB are left out; the
  median run is missing 61 MB. On r25 (R64) I find no relation for Granite num (n = 8, r −0.17) and none for
  OLMoE num (n = 8, r 0.07); OLMoE off r 0.51 (n = 9). The draft says this for R64 in §1.3 but §0.3 and the
  lever table use 0.046 as if it were a rate. **Position Ebony:** for Granite the supported statement is "runs
  missing several hundred MB are slow"; a per-MB price at the median (61 MB → +2.8 ms, M1's "155 MB × 0.046")
  is not supported. **Position Bonnie (as drafted):** one slope per model. The memo should carry both.
- My intercepts differ slightly from the draft's (49.9 and 46.6 against 51.0 and 44.7), so our run filters are
  not identical; the script that produced §0.3 is not among the three beside the draft (`runfit.py` fits majors
  and MB read, not missing MB). Please add it so the fit can be reproduced from the memo.
- `VmSwap` null: reproduced (r −0.31 Granite, +0.10 OLMoE; swap and missing MB are not collinear, r −0.36 and
  0.05). It is a null for **swap held before the run**. It does not test swap-outs during the run, and one
  session with apps restarting is the only day it was fitted on.
- Per-token coefficients (0.26 to 0.44 ms per major, 62 µs per minor): the draft rightly says "coincides".
  I would go further and not print "62 ± 9 µs each" as a cost: 14,000 to 29,000 minors × 62 µs is 0.9 to 1.8 s
  per run, several times the whole penalty, so the coefficient is absorbing something else.
- A reading the draft does not make, for T1: a fault on a page that is in the page cache but not mapped, or
  that is still locked by a read in progress, is counted **minor** by Linux, not major. 14,000 to 29,000 decode
  minors in slow runs against 3,300 to 4,800 in clean ones fits "the hold read the pages back but they are
  unmapped, or still being read, when the decode reaches them". That would make part of the penalty a
  property of the hold itself. Cheap split: `mincore` says resident, `/proc/self/pagemap` (or the `Rss` of the
  mapping in `smaps`) says mapped; log both at run start. Not verified: the phone's `fault_around_bytes`.

## 3. The TRY ranking

**A8, `MASS=900` back (rank 1).**
- The evidence is better than "n = 2 against 1" in one respect and worse in another. Better: memo-21 §0.1
  already has a second session (R60: experts 31.1 off, 27.2 num, −3.9 ms, 9 and 23 tokens). Two sessions agree
  in sign. Worse: they disagree in size by a factor of two (−3.9 and −1.9 to −2.3; bytes predict −3.8), the
  runs are not randomised against each other, and the tokens are selected on faults.
- The number that reaches tok/s is the net, not the op: the draft's own §1.1 has wall 45.1 (off) against 44.0 /
  43.6 (num), **−1.1 to −1.5 ms**, because other dense rises by 0.5. The TRY table's "−2" should read about −1.3.
- It costs 0.38 of OLMoE's 0.586 budget for that, and then leaves 0.203 for A3, M2, B1 and B2 together.
- Process: R63 §4 parked it by a rule written before the data. A post hoc op-time reading from the same reports
  is a reason to test again, not a reason to un-park. If it returns, it should do so under a new
  pre-registration whose primary endpoint is the expert-op time in ABBA (small σ), with tok/s secondary. That is
  the lead's call; my recommendation is "re-test with a better endpoint", and rank 1 only in the sense of no
  build work.

**A5, per-thread pinning (rank 2).**
- The stored reports show `pinned_cpus [4, 5, 6, 7]` from r23 on: the workers are already confined to the big
  cluster. A5 is therefore only the step from "any big core" to "one fixed core each". The draft's "a fifth of
  OLMoE's samples on cpus 0 to 3" is from R62; it should say whether those were the 3-thread or 4-thread runs
  and whether the mask was applied in them.
- Strict pinning removes the scheduler's freedom to move a worker off a core another process has taken. The
  draft's own T1 lists "other apps taking the big cores" as a candidate cause of the penalty, and R62's plain
  fourth thread lost 3.9 to 5.1 tok/s, far more than the barrier arithmetic, which says the fourth big core is
  often not free. With 666 barriers per token on Granite, one preempted pinned worker stalls all three. So on
  the days that matter the sign is open.
- Arm {4, 5, 6} gives up the X1 (about 1.2 times an A78 at these clocks) for the expert op: roughly +0.9 ms
  (Granite) and +1.7 ms (OLMoE) of compute against −0.3 to −1.0 of waits (*estimated*: one of three workers
  6 % of the op slower is small, but the op is dynamic-chunked, so the loss is about 1 − 3.0/3.2).
- Literature: the phone studies I found (arXiv 2410.03613; MNN-LLM, arXiv 2506.10443, search summaries) support
  "threads = big cores" and capability-weighted shares on idle phones. None measures strict affinity under
  background load.
- Position Ebony: keep A5, but behind T1 (it decides whether CPU contention is the cause) and with the arms
  floating-on-mask (today), strict {5, 6, 7}, strict {4, 5, 6}; price centre 0, band +1 to −1 on pressure days.

**A9, Granite skip above 130 (rank 3).**
- It is the same question as my B1 (per-layer thresholds): how many expert bytes the remaining budget buys.
  A uniform threshold is the special case. T6 and my T5(a) should be one design: uniform 135 / 138 / 141 / 143
  and a per-layer table at matched skipped bytes, both batch settings. Ban&Pick (arXiv 2509.06346v2) reports
  uneven layer sensitivity on Qwen3 and DeepSeek, by task accuracy, not perplexity, and not on these models.
- 145 was over by 0.0083 at `-ub 1` and R61 showed the build alone moves Granite's stock by 0.5 %. "Largest
  screened threshold inside" with a margin that small is fragile; write a margin into the rule (for example
  inside by at least 0.02 at both settings).
- −1.2 to −1.6 ms is priced on bytes; by §1 above about a fifth comes back in the dense ops. And it spends all
  of 0.176, which closes A3 and every other numerics lever on Granite. `en.txt` rose 1.7 times faster than
  wikitext at 130 on the old file (STATUS caution); report it per threshold.

**A1, helpers (rank 5).** f = 0.3 to 0.7 has no datum; R62 and R18 are both against it. The binding limit on
this phone is the cap (§4.4), and a fourth active core is the likeliest way to step it down (R62's four-thread
runs started hotter). I would centre the price at f = 0.15 (Granite −0.6, OLMoE −1.3) until T2, and have T2 log
`cap_after`.

**M1.** Priced with the Granite slope that does not survive at the median (§2). I would write "0 clean; about
−0.5 over days, up to −3 on a day missing hundreds of MB". It is cheap and bit-identical in num, so its rank
can stay; it is not generic and the memo should say that it does nothing for the trust purpose.

**M3, anon hold.** Agreed: TTFT only. The same evidence removes my A2 (§4.1).

**M5 and T5.** Agreed as a candidate, with one objection to its test: the draft's own observation that the hold
advises 2 to 3 times the distinct MB ("pages evicted again while being read back") says read-back alone does
not stick. Memo-18's simulator is plain LRU and cannot show that; it will overstate M5. BigMoeOnEdge (2026-07-21,
Android 16, one informal source) reports the same: 1.76 GB restored by `MADV_WILLNEED` was reclaimed within 8 s
because restored pages land on the inactive list. M5 needs the page-class change of my A1 to hold what it
restores; I propose they are built and tested as one lever (§4.2).

**A2 / A3 / T7, LUT.** No news in the literature since memo-21; I found nothing newer than T-MAC for 3-bit
K-quants on NEON without i8mm, and no Q3_K row-interleaved kernel in ik_llama.cpp (search summary). Ranking fine.

**Not in the draft: clock.** See §4.4.

## 4. What changes in my own draft

1. **A2 (dma-buf pin of the dense anon) and A3 are withdrawn from the TRY list.** My decode price rested on
   swap slots. With swap not predicting decode time and missing file pages predicting it, pinning 0.4 to
   0.55 GB of anon can only push reclaim harder onto the model's file pages: decode 0 at best, possibly
   negative. What is left is TTFT, the same class as M3. T3 and T4 are dropped with it.
2. **A1 (`PROT_EXEC` mapping) moves up and merges with M5.** It acts on exactly the variable that predicts the
   metric. Re-priced with the OLMoE slope: if it halves the MB missing at run start on an R66-like day
   (332 MB median), −5.6 ms; band **OLMoE 0 to −6, centre −2 over days; Granite 0 to −1.5** (*estimated*; Granite
   by the fragile slope). Primary endpoint of its test becomes "MB missing at run start" and "MB the hold must
   advise per distinct MB", not wait loss. Still required first: T1 of my draft (does the phone accept an exec
   mapping, and the model moved to internal storage, since the external files folder is expected `noexec`).
3. **T8 (double caching) is answered: no.** r26's first run has `cached=2614MB` with 2,007 MB of the model
   resident. Dropped.
4. **§E, "ADPF priced 0", is withdrawn.** In r22, r23, r25 and r26 the cap field reads 2,035 in 154 of 192
   runs, across heat 0.67 to 1.08; lower steps (1,785, 1,670, 1,555) appear when hot and 2,150 three times.
   Hardware maxima are 2,419 (A78) and 2,841 (X1), and campaign 1 saw the X1 at 2,841 in a bench (memo-21 §1).
   A value that does not move over that heat range does not look like a thermal step. Also odd: the field
   prints the same number for both policies while cpus 4 to 6 read 2,112, so what it measures needs Tom's eye.
   If the ceiling is a policy that an app can lift (hint session, sustained-performance or game category; none
   verified on this ROM), the compute-bound expert op alone scales by 2,112/2,419: **Granite −1.7 ms, OLMoE
   −3.4 ms** (*estimated*), before any X1 gain and against more heat. New desk test: every stored report of both
   campaigns, cap against heat, phase and time since load; when was a cap above 2,150 ever recorded.
5. B1 folds into the joint host run with A9 (above). B2 and C1 unchanged, both behind their host tests; the
   draft's "head at its screen floor" agrees with my low prior for C1.
6. Budget arithmetic I had not done: OLMoE has 0.203 left with A8 on, 0.055 with A8 and M2. B1 is a
   re-allocation of the skip budget, not an addition; B2 would need budget that A8 + M2 leave almost none of.

## 5. Proposed merged order (for Bonnie to accept or contest)

TEST first, all cheap: her T1 with the mapped-against-resident split (§2); her T3; the cap desk test (§4.4); my
exec-mmap errno line; her T6 widened to per-layer tables; her T2 with `cap_after`; my T7 (certified head, low
prior). TRY: A1 + M5 as one residency lever; M1; A8 as a re-test under a new pre-registration; A6 and A7; A5
after T1; A9 / B1 after the joint host run; helpers and LUT after their microbenches.

## 6. Not verified by me

The meaning of the kit's `cap=` field and of `thread_samples`; whether the R62 little-core samples were taken
with the big-cluster mask on; the vendor kernel's reclaim code; that an app can change the frequency ceiling on
this ROM; the per-layer sensitivity of either model.
