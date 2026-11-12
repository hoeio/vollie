# Campaigns

Vollie is run as a series of **campaigns**. A campaign is one goal, with its own rules, budget, claim rule and
research log. Campaigns never rewrite each other: a finished campaign is frozen (tagged) and the next one is
added beside it.

| # | goal | status | where |
|---|---|---|---|
| 1 | 20 tok/s on a Snapdragon 888 phone, one frozen 7B MoE file (Granite-4.0-H-Tiny), runtime only, quality budget +2 % | **done**: median 20.51 tok/s (*device*, R49). Frozen at tag `goal1-final` | the repository root as it is: [README](../README.md), [RULES.md](../RULES.md), [reports/](../reports/README.md), [patches/](../patches/README.md), [models/](../models/) |
| 2 | 25 tok/s on the same phone, a 7B-class model, runtime only, quality budget +5 % | **planning**, nothing measured yet | [02-25toks-5pct/](02-25toks-5pct/CAMPAIGN.md) |

## Principles shared by every campaign

[COMMON_RULES.md](COMMON_RULES.md). Campaign 1's full rules stay in [../RULES.md](../RULES.md), unchanged.

## How a campaign is laid out

```
campaigns/<nn>-<name>/
  CAMPAIGN.md      goal, budget accounting, claim rule, status, open decisions
  candidates.md    which models are in or out, and why (with sources)
  feasibility.md   the arithmetic behind the goal (labelled estimates)
  prereg/          one file per pre-registered round (TEMPLATE.md)
  reports/         the campaign's R-log (one "### R<n>" heading per entry, numeric order)
  results.csv      one row per result (schema in the campaign folder)
```

## Pre-registration is proven by git

A round's prediction, pass rule and kill rule are written into `prereg/` and **committed before the device run**,
then tagged `prereg-R<n>`. The commit time is the evidence that it was written first. A result is scored against
that commit as written, misses included. If a rule has to change, a new file and a new tag are made; the old
ones stay.

## What does not move

Files of finished campaigns are not moved or renamed: the R-log and the docs link to them by path. New work goes
into new files.
