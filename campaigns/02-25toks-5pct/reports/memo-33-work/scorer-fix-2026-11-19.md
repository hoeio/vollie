# R68 scorer: field-reading fix, 2026-11-19

Author: Ebony. Scope: `score_r68.py` only. Trigger: the first real kit r28 report (session B, started
2026-11-18T21:02, sha256 `c0b5af60c548aff2a5e222979b1e494653c78f7ae96031b47df59f179a433d63`), on which the scorer ran
but misread the layout. The synthetic reports the scorer was tested on had fields the kit does not write.

**No rule changed.** No threshold, band, pass or kill rule, and no reading fixed in `prereg/R68-amendment-2.md` or
`prereg/R68-amendment-3.md` was touched. Amendment 3 foresees this ("If either differs, the scorer is corrected to
the real layout. That correction is a code fix, not a change of rule."). The diff is: one helper that fills a null
field, one fallback field name, one check line, one selftest case, and the dated note in the docstring.

## The three bugs

| # | symptom on the real file | cause | fix |
|---|---|---|---|
| 1 | The claim-block section printed nothing; the go rule read "MISSED" on no data. | Real runs carry `group` (b1..b5) and `phase` (warmup / on); `block` is null. The scorer selected blocks by `x['block']`. | `fill_blocks()`, called once in `Session.__init__`: per group, the run's own `block` if set, else `design.blocks[group number − 1].block`, else the run id suffix (`_claim` → claim, `_q…` → quad). Warm-ups take their group's kind. Every later reader is unchanged. |
| 2 | "B1/B2 nice permission: N runs 0"; no factor lines for either model. | The same null `block`: no quad was found, so there were no N runs to count. The cell reading itself was already right: it is taken from `env` (`VOLLIE_TOUCH`, `VOLLIE_NICE`), and the null spec fields `touch` / `nice` of the runs are not read. | Fixed by 1. Added one output line, "layout check", which counts scored runs whose id suffix (`_q_touch`, `_q_nice`, `_q_touch_nice`) disagrees with the block kind used or with the env cell, and lists them. On session B: 0 of 62. |
| 3 | D3 printed "VOID (internal copy fails its sha256 or has none)" for both models. | For a copy already in place the kit writes `storage.models.<m>.present_sha256` with `action: present`; the scorer read only `sha256`. | `sha256`, else `present_sha256` (as `tools/c2_research_parse.py` does); the line names the field it read. A wrong or missing value still voids. |

## Confirmed on the real file (amendment 3, last paragraph)

- `workers_after` rows are `[tid, nice, priority, policy, cpu, cgroup]`, as the scorer assumed (*device*: for example
  `[21270, -10, 10, 0, 5, …]`).
- The `c2-nice` line is `c2-nice run=<tag> want=-10 pool=1 restored=0/0 w0=<tid>:<value>:s<sets>:f<fails>:e<errno> …`;
  the scorer's pattern matches it unchanged. The same holds for `c2-touch`, `c2-sched`, `c2-pool`, `c2-io` and
  `c2-skip-run`.
- The models' `expected_sha256` equal the scorer's full hashes.

## Selftest

`python3 -I score_r68.py --selftest` passes. Case 9 is new: the synthetic session B rewritten into the real layout
(`block`, `touch`, `nice` null in the runs; `design.blocks`; `storage.models.<m>.present_sha256`; six threads in
`workers_after`), scored once with `design.blocks` and once with the id suffix alone; it must give case 1's numbers
and no VOID. It also checks that a wrong `present_sha256` still voids and that an id contradicting its env is listed.

## Not a field-reading bug, and not changed: the nice factor on session B

The fixed scorer prints "VERDICT nice: VOID factor (0 whole quads)" for both models. That is rule B3 applied as
written (amendment 2: a quad is dropped when an n run's pool thread reads −10 in `workers_after`), on correctly read
fields. What the file shows (*device*): every thread in `workers_after` reads nice −10 in all 72 runs, the first
warm-up included, before any N run; and all 78 worker entries of the 26 `c2-nice` lines read `−10:s0:f0:e0`, that is,
zero sets. The pool was at −10 before the knob was ever switched on, so N and n runs were in the same state and the
factor had no contrast. "Permitted: hit" is therefore true only in the letter (nothing was set). Whether B3's −10 test
or the audit's sentence "back at the first value" is the reading to keep is the lead's decision; the scorer was left
as it was. For reference only, not a verdict: without the B3 drop the paired N − n effect is −0.45 ± 0.58 tok/s
(OLMoE) and −0.51 ± 0.13 tok/s (Granite), 4 quads each (*device*).

## Output

`score-r68-sessionB.txt` in this folder is the fixed scorer's full output on the session B file. Session A, session R
and the cap probes have not arrived; A8 and the go rule's session A part are provisional, as the output says. The
three items it lists as "still open after amendment 2" are decided by amendment 3 (items 1, 6 and 3) with the reading
the scorer already uses.
