# R68 amendment 2: scoring readings fixed before any data is seen

Date: 2026-11-18. Author: clyde (lead). Written and pushed **before the lead or any agent has seen R68 data.**
Session B may already have run on the phone overnight, but no report has reached the repository or this session.
The source is Ebony's audit, `reports/memo-33-work/ebony-r68-audit.md`. The item numbers below refer to that audit,
which gives the exact field reading for each item. Each reading is adopted **as proposed there**, except where this
file states otherwise. Nothing here changes a prediction, a band, a pass/kill threshold or the go rule's numbers.

## Adopted as proposed

- **Touch**
  - A1 (missing MB: want minus mapped, unit-corrected, one source per session and model; the resident version is
    reported, not scored).
  - A3 (waits of 60 s or more means quad runs with `wait_ms ≥ 60000`).
  - A4 (relative effect of medians; a pair is a quad with a T and a t run over 60 s; fewer than 3 such quads means
    "not testable").
  - A5 (paired per quad, SE over quads, per model and session).
  - A6 (per-run end-pass check, `src=want`, ≥ 90 %).
  - A7 and A8 (two sessions: pass needs a pass wherever testable; a kill in either session is a kill; otherwise
    "mixed").
- **Nice**
  - B1, B2 and B3 (permitted, partly permitted, not permitted; restore failures drop the quad).
  - B5 (tok/s as A5).
- **Clock cap and go rule**
  - C2 and C3 (§3 prediction on ≥ 9 of 10; the go rule needs 10 of 10; a block counts as capped at ≥ 2 capped runs;
    the band is chosen by ≥ 9 of 10, otherwise "mixed").
  - C4 and C5 (b1 and b5 are both evaluated; b1 wins if both pass; b5's eligibility conditions; the num-arm check).
  - C6 (which report is A and which is B).
- **Validity and cap probe**
  - D1 to D4 (power-on runs and their quads are dropped before every statistic; hashes are memo-32 §5's full
    values; identity across reports).
  - E (cap-probe reading).

## Lead's choices where the audit offered one

- **A2 (touch endpoint read after the final pass).** The loss rule is scored as written and reported as a
  **mechanism check of the final pass**. §7's "it stays on" needs, in addition, the tok/s condition of §4. It also
  needs `wait_lost_mb` (unscored, read before the hold) not to be higher with T. That second condition is a
  reporting condition and adds no new pass route.
- **B4 (nice, the run-queue route).**
  - rq is read per decode, as the sum over workers of `wq_us` / 1000, with pairs within the same touch level.
  - The route needs at least 3 qualifying pairs and a median ratio ≤ 0.5.
  - The **mirror control** is part of the rule: if the mirror statistic (pairs selected on the N run) also reads
    ≤ 0.5, the route does not pass, and the verdict is "permitted, speed not shown".
- **C1 (uncapped).** The stricter reading is adopted. A run is uncapped when `cap_before` **and** `cap_after` both
  read policy 4 ≥ 2,419 **and** policy 7 ≥ 2,841. This is stricter than the written "cpu 7 ≥ 2,419", so it can only
  make the go rule harder to meet. The written reading is listed beside it for every run where the two disagree.

## Noted (not judgement calls)

- With 4 quads, "beyond 2 × SE" is about an 86 % two-sided level (t, 3 d.f.). It is scored as written.
- An unattended session needs the charger out all night. Plugged quads are dropped (D2).
