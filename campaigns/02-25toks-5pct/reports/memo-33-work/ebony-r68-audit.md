# R68 pre-registration audit: where scoring would need a judgement call (Ebony)

Agent model: claude-fable-5-2. Date: 2026-11-18. Written **before any R68 device data exists**; nothing here is a
result. Not an R-entry, not a verdict: proposals for the lead to fix in `R68-amendment-2.md` before the data is read.

Read: `prereg/R68-rehearsal-touch-nice-references-device.md` (called "prereg" below), `prereg/R68-amendment-1.md`,
memo-31 §4, memo-32, `tools/c2_research_parse.py` (r28 part, lines 1651 to 1839 and the selftest),
`patches/app-c2/0012-kit-r28.patch`, and R67's report `reports/device/r27_2026-11-18.json` to see what the fields hold
(numbers quoted from it are *device*, R67). The parser was read, not run.

"Field" means a key of the kit report or a parser function. "Exists" means the parser prints the scored quantity
today; "raw only" means the report has the data but the parser does not compute the scored quantity.

## 0. The four that can change a verdict

1. **"Wanted MB missing at run start" has two definitions that differ by a factor of about ten** (item A1).
2. **The touch loss endpoint is measured after touch's own forced final pass** (item A2), so with the prereg's
   definition it passes by construction.
3. **The nice rule's second route selects on the off cell** (item B4): regression to the mean passes it without any
   effect.
4. **"Uncapped" is written as "cpu 7 max ≥ 2,419", but 2,419 is policy 4's ceiling; cpu 7's is 2,841** (item C1).

## A. Touch

**A1. What "wanted MB missing at run start" is.**
- Prereg §6 says "(smaps by inode, fallback `RssFile`)", that is want minus *mapped*. Memo-31's "150 to 540 MB
  missing" was `want_mb − res_before.total_mb`, that is want minus *resident* (mincore) (`memo-31-work/ebony_runs.py`
  line 24). On R67's timeout runs the first gives 929 to 2,759 MB and the second 2 to 538 MB (*device*, R67,
  recomputed today from the stored report).
- Neither is "the wanted set": both subtract a whole-file number from the wanted set's size, so pages outside the set
  hide missing ones.
- Units differ: `want_mb` is bytes / 10^6, `smaps_model` is kB / 1000 (1.024 times larger unit), `RssFile` is a
  /proc/status value that also counts the libraries. On R67's clean runs want − `RssFile` reads 71 to 250 MB with
  nothing missing (*device*, R67).
- **Proposed reading:** missing(x) = max(0, `want_mb` − `smaps_before.model[1]` × 1024 / 10^6), the prereg's
  definition; one source per session and model (smaps by inode when it is above 0 in every scored quad run, else
  `RssFile` for all of them, never mixed); the resident version max(0, `want_mb − res_before.total_mb`) is reported
  beside it and not scored.
- Parser: raw only. `R28_METRICS` has "model Rss before MB" and "res before MB" but no want-minus metric.

**A2. The endpoint is read after the final touch pass.** `touch_end` blocks until its last pass is done, then
`smaps_before` is read (app patch 0012, line order `touch_end`, `power`, `smaps_before`, `res_before`). A T run has
therefore just mapped its wanted set; a t run has not. The 30 % pass rule then tests "does reading a page map it", not
"did touching during the wait keep pages".
- **Proposed reading:** score the rule as written (A1), and state in the verdict that it is a mechanism check of the
  final pass; the answer to "did the wait cost less" is read, unscored, from `wait_lost_mb` (measured before the hold's
  read-back), the final pass's `majflt` / `minflt` / `ms` (`touch_final(x, 8 / 9 / 6)`), `ttft_ms` and decode `majflt`.
- Parser: `wait_lost_mb` and the touch fields exist as paired metrics.

**A3. Which runs are "after waits ≥ 60 s".**
- **Proposed reading:** a quad-block scored run (`block == 'quad'`, `phase == 'on'`) with `wait_ms ≥ 60000`, whatever
  `wait_end` says (gate or timeout); `cool_ms` is not used. R67 had waits of 65 to 81 s that ended by the gate.
- Claim blocks are not in the touch comparison (b1 has no touch, b5 has it with another arm).
- Parser: raw only (`wait_ms` exists; no filter. `--wait-s` is r22's and defaults to 30).

**A4. How "lower by at least 30 %" is computed, and what a "pair" is.**
- **Proposed reading:** effect = 1 − median(missing, T runs of A3) / median(missing, t runs of A3), per model and
  session; pass at ≥ 0.30.
- **Pair:** a quad that holds at least one T run and one t run with `wait_ms ≥ 60000`; fewer than 3 such quads (of 4),
  or a t median of 0, gives "not testable this session".
- The wait length is itself after the treatment (the helper adds heat on cpus 0-3); the share of waits ≥ 60 s is
  reported per level.
- Parser: missing.

**A5. Touch tok/s and "2 × SE".**
- **Proposed reading:** effect = mean over quads of (mean of the 2 T runs − mean of the 2 t runs) of `decode_tps`;
  SE = sample standard deviation of the per-quad values / √(quads); "negative beyond 2 × SE" is effect < −2 × SE; per
  model, per session, not pooled. The Welch column is not used.
- Parser: exists (`r27_factor`, line "tok/s: paired"), but it does not drop void runs (D2).

**A6. "In every T run the end pass touches ≥ 90 % of the wanted set."**
- **Proposed reading:** in every gated run with `VOLLIE_TOUCH` in `env` (quad T cells and b5), the last `c2-touch`
  line with `final=1` has `src=want`, `pages / set ≥ 0.90`, and `set` × page size within 1 % of `want_mb`.
- Parser: raw only (medians of `touch_final(x, 5)` and `(x, 4)` per level; no per-run check, no `src` check).

**A7. Loss endpoint not testable.**
- **Proposed reading:** when A4 says "not testable", touch is neither pass nor kill on the loss endpoint; the tok/s
  kill still applies; the verdict reads "loss not testable, speed not negative" and §7's "it stays on" does not follow.

**A8. Two sessions, one verdict.** The prereg scores "per session", §7 speaks of one outcome.
- **Proposed reading:** each model and session is scored on its own; a lever passes for a model only when it passes in
  every session where it is testable; a kill in either session is a kill; anything else is "mixed", reported per
  session.

## B. Nice

**B1. "Permitted".**
- **Proposed reading:** nice is permitted in a session when, in every quad N run of both models, the last `c2-nice`
  line shows all pool threads (the caller `w0` included) with value read back = −10 and `f0`; `workers_after` must show
  −10 for the same tids.
- Parser: raw only (`nice_of`; the table sums values and fails over runs, not per run).

**B2. Nice on some threads only** (for example the pool's workers but not the Dart caller, or the reverse).
- **Proposed reading:** if any N run has a thread not at −10, the factor is "partly permitted": no pass or kill, the
  tok/s effect is reported with the per-thread pattern, and §7 treats it as void (ADPF priced again).
- If no thread in any N run reads −10: void, "not permitted".

**B3. Restore.** An n run after an N run must be back at the first value.
- **Proposed reading:** a quad is dropped from the nice factor when an n run in it has `restored` failures > 0 or a
  `workers_after` nice of −10 on a pool tid.
- Parser: raw only.

**B4. "Run-queue wait above 100 ms per decode ... lower by at least half."**
- Unit: memo-31's 315 to 461 ms is the whole decode phase, all three workers summed. The parser's `wq_ms_tok` is per
  token (÷ `decode_graphs`); 100 ms per decode is about 3.2 ms per token.
- **Proposed reading:** rq(x) = Σ over workers of `wq_us` / 1000 from the `c2-sched` line with `phase=decode`; pairs
  are the two same-touch pairs in each quad (tn with tN, Tn with TN); a pair counts when rq(n run) > 100; the route
  passes when at least 3 pairs count and the median of rq(N run) / rq(n run) over them is ≤ 0.5.
- **Flaw, not removable by a reading:** selecting pairs on a high off cell makes the on cell lower by regression to
  the mean (type A was 3 of 16 runs in R67, sporadic). Proposed control, reported with the result: the mirror statistic
  (pairs where rq(N run) > 100, ratio n / N) and the counts of runs above 100 ms per level. If the mirror also reads
  ≤ 0.5, the verdict is "permitted, speed not shown".
- Parser: missing (per-token value only; no pairing).

**B5. Nice tok/s:** as A5 with N and n.

## C. Clock cap and the go rule

**C1. "Uncapped".** `cap_before` is `cap=<policy 4>/<policy 7>` in MHz (app patch 0003, `_caps`). Stored states:
2419/2841, 2150/2150, 2035/2035, 1785/1785, 1555/1555 (*device*, kit r26 and kit r27 reports in `reports/device/`). "cpu 7 max ≥ 2,419" would call a
state such as 2419/2419 uncapped although the prime core is limited.
- **Proposed reading:** a run is uncapped when `cap_before` **and** `cap_after` both read policy 4 ≥ 2419 and
  policy 7 ≥ 2841. In R67 two runs started at 2419/2841 and ended at 2150/2150 (9.4 and 15.9 tok/s, *device*); they
  count as capped.
- If the lead keeps the written threshold instead, it is read on policy 7 of `cap_before`, and every run where the two
  readings disagree is listed.
- Parser: partly (`cap7(x)` gives policy 7 of `cap_before` as min..max per block; no count, no `cap_after`, no
  policy 4).

**C2. "All 10" against "9 of 10" against "block 1 is capped".** Prereg §3 predicts on ≥ 9 of 10 in b1; the go rule
wants all 10; memo-31's third bullet says "if block 1 is capped".
- **Proposed reading:** (a) the §3 prediction is scored on b1, ≥ 9 of 10, per session; (b) the go rule needs 10 of 10
  uncapped in the claimed block of session B; (c) a block "is capped" when 2 or more of its 10 runs are capped, and the
  third bullet fires when b1 or the claimed block is capped in either session.

**C3. Which level band applies (§3 table, "uncapped" and "capped" rows).**
- **Proposed reading:** ≥ 9 of 10 runs uncapped: the uncapped band; ≥ 9 of 10 capped (any state below C1): the capped
  band; otherwise "mixed", both bands shown, neither scored. The same for session R's bands, which are written for
  "uncapped".

**C4. The configuration to be claimed: b1 or b5.**
- **Proposed reading:** both are evaluated. A configuration meets the go rule when the **same block** has, in B,
  10 valid runs, 10 uncapped, median `decode_tps` ≥ 25.7 and at most 2 runs < 25.0, and in A a median ≥ 25.0. If both
  meet it, b1 is the claim configuration (no budget spent). If only b5 does, b5 is, exactly as run.
- b5 is eligible only when: its texts are one (identity), touch is not killed on OLMoE in either session, and the
  pre-run memory state is readable (smaps by inode > 0 in its runs; memo-31 §4, numerics parked under R63 §4).
- A configuration that was not run as a block (b5 without a killed knob, or b1 with touch) cannot meet the go rule in
  R68.
- Comparisons are on unrounded `decode_tps`; the median of 10 is the mean of the 5th and 6th values.
- Parser: per-block median and count below 25.0 exist; the rule itself is missing.

**C5. b5's touch or nice void.**
- **Proposed reading:** nice not permitted: b5 is "num + touch (nice asked, refused)", still eligible, and the
  b5 − b1 prediction (+1.0, +0.4 to +1.6) is scored as written. Nice partly permitted, or touch failing A6 in any b5
  run: b5's level is reported, the b5 − b1 prediction is scored as written, and b5 is not eligible for the go rule.
- b5 − b1 = median(b5) − median(b1) of the same session; position (b5 last, b1 first) is confounded and said so.
- Check that b5 ran its num arm: `VOLLIE_SKIP_MASS=900` and `VOLLIE_HEAD2=1024` in `env_set`, and a non-zero skip
  counter (R63's void rule; not repeated in R68's prereg).

**C6. Which report is session B.**
- **Proposed reading:** B is the first report with `build.runtime == 'c2'`, `session_label == 'deep-sleeping list
  off'`, `stopped_early == false` and all 62 scored runs, started on the night of 2026-11-18 (amendment 1); A is the
  first such report with "list on". A stopped or repeated session is reported and not scored. The label is the owner's
  tap; nothing in the report can confirm it.
- Parser: `session_label` exists; `runtime` is read as `r['runtime']` or `build.runtime`.

## D. Void rules

**D1. "Power saving reads on in a run".**
- **Proposed reading:** `power_label(x['power']) == 'on'`, that is `powerSave` is true or `lowPower` > 0. A label of
  "?" (no `powerSave`) is not "on": the run is scored if the session's `power_start` and `power_end` both read off,
  otherwise unscored. The state is read once before the run; a change during a run is not seen.
- Parser: exists as a count; **the block medians and the factor tables still include those runs.**

**D2. Unscored runs inside a block.**
- **Proposed reading:** claim block: the median and the count below 25.0 are taken over valid runs; fewer than 8 valid
  runs voids the block; the go rule needs all 10 valid. Quad block: a quad with any invalid run (power on, or
  `plugged_run`, R63 §4) is dropped whole; fewer than 3 whole quads voids that factor for that model and session.
- Parser: missing (no filter by power; `plugged_run` is handled by older kits' code, not in `r28_tables`).

**D3. Hashes.** The prereg prints shortened hashes.
- **Proposed reading:** the full values are memo-32 §5's; compared with `build.libs` and each load row's `libs`
  (libraries), `models[m].sha256_end` and `storage.models[m].sha256` (files).
- Parser: prints them; does not compare with constants.

**D4. Identity.**
- **Proposed reading:** per model and arm, one md5 of `text` over all rows of a report, warm-ups included, and the same
  md5 in A and B; a row with no `text` is listed. A difference inside quad cells suspends the knob(s) whose level
  separates the texts; inside b5, all of num, touch and nice.
- Parser: exists within a report (`r28_checks`), not across reports; rows without text are skipped silently.

## E. Cap probe (T-A)

- No parser reads `vollie-capprobe-r28-*.json`. **Missing.**
- **Proposed reading:** scored probes are the first complete one (360 samples) of each power state by its `label`;
  "most of the probe" is more than half of all samples; power saving on predicts `max7` (sample index 3) ≤ 2150, off
  predicts `max7` ≥ 2841 (or ≥ 2419 if C1 keeps the written number); the prediction holds only if both do. The halves
  (no load, load) are reported apart.

## F. Parser work this implies (for Tom, through the lead)

| need | state |
|---|---|
| missing MB (A1), both versions, unit-corrected | add |
| filter `wait_ms ≥ 60000`, per-quad pair count, relative effect (A3, A4) | add |
| per-run touch check: `src`, pages / set (A6) | add |
| per-run nice status: all / some / none at −10, restore (B1 to B3) | add |
| rq per decode, paired by touch level, with mirror (B4) | add |
| uncapped count per block from `cap_before` and `cap_after`, both policies (C1, C2) | add |
| go-rule line per claim block; b5 − b1 (C4, C5) | add |
| drop power-on and plugged runs and their quads before every statistic (D1, D2) | add |
| cap-probe reader (E) | add |
| cross-report identity and session picking (C6, D4) | by hand, or add |

## G. Not judgement calls, but worth a line in the amendment

- Session B runs unattended overnight: the charger must be out all night (D2 drops plugged quads), and 62 gated runs
  need the battery to last about 1.4 to 1.8 h of load (*estimated*, memo-32 §3.3).
- With 4 quads, "beyond 2 × SE" is about an 86 % two-sided level (*estimated*) (t, 3 d.f.), not 95 %. Scored as written.
