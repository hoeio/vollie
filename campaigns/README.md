# Campaigns

Vollie is run as a series of **campaigns**. A campaign is one goal, with its own rules, budget, claim rule and
research log. Campaigns never rewrite each other: a finished campaign is frozen (tagged) and the next one is
added beside it.

| # | goal | status | where |
|---|---|---|---|
| 1 | 20 tok/s on a Snapdragon 888 phone, one frozen 7B MoE file (Granite-4.0-H-Tiny), runtime only, quality budget +2 % | **done**: median 20.51 tok/s (*device*, R49). Frozen at commit `686f17d` | the repository root as it is: [README](../README.md), [RULES.md](../RULES.md), [reports/](../reports/README.md), [patches/](../patches/README.md), [models/](../models/) |
| 2 | 25 tok/s on the same phone, a 7B-class model, runtime only, quality budget +5 % | **closed (owner, 2026-11-21)**: goal met on OLMoE-1B-7B, not on Granite-4.0-H-Tiny. OLMoE b5 num + touch median **27.21 tok/s**, 1 of 10 runs below 25.0 (*device*, R70; power saving off, deep-sleeping list on, warm start); Granite num 23.21 tok/s, 10 of 10 below 25.0 (*device*). Result: [CAMPAIGN.md](02-25toks-5pct/CAMPAIGN.md), state: [STATUS.md](02-25toks-5pct/STATUS.md) *Correction 2026-11-22: this row said "in progress" (device rounds R52 to R68, 25 tok/s not claimed) until the owner closed campaign 2 on 2026-11-21.* | [02-25toks-5pct/](02-25toks-5pct/CAMPAIGN.md) |
| 3 | 15 tok/s (first goal) on the same phone, a **dense** 7B-class model, file not changed (used freely), quality budget +5 % | **research, under Amendment 1 (owner, 2026-11-21)**: 12 tok/s on **each** of three dense 7B-class models of different families, all three must pass, budget +5 %, file not changed. Subjects: Yi-1.5-6B-Chat, Llama-2-7b-chat, DeepSeek-LLM-7B-chat (reserves Mistral-7B-Instruct-v0.3, Qwen2.5-7B-Instruct). R69 (probe, no model) is scored: no void; r 28.56 GB/s and c_b 24.46 GB/s at reduced clocks (*device*); no subject reaches 12 with the priced CPU levers (*estimated*). R72 (probe, kit r33) ran all ten blocks on 2026-11-22; scoring pending. State: [STATUS.md](03-dense15-5pct/STATUS.md), goal: [campaigns/03-dense15-5pct/](03-dense15-5pct/CAMPAIGN.md) *Correction 2026-11-22: this row said "R69 device probe pre-registered 2026-11-19, not run" and a goal of 15 tok/s only; true until R69 ran on 2026-11-22 and Amendment 1 (2026-11-21) set 12 tok/s on three models.* | [03-dense15-5pct/](03-dense15-5pct/CAMPAIGN.md) |

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
