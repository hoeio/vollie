# Vollie: instructions for a new session

You are continuing a long research project. **Read this file, then
[campaigns/02-25toks-5pct/STATUS.md](campaigns/02-25toks-5pct/STATUS.md)**; that file says where the work stands
and what to do next. Do not start from the 1,074-line R-log: [knowledge.md](campaigns/02-25toks-5pct/knowledge.md)
is the digest, and the originals are the reference when the two disagree.

## What this is

Vollie runs a sparse (mixture-of-experts) LLM on a 2021 phone CPU (Samsung Galaxy Z Flip 3, Snapdragon 888,
8 GB) by changing only the runtime: patches to llama.cpp's CPU module, measured on the phone.

- **Campaign 1 (finished, frozen at commit `686f17d`):** 20 tok/s on Granite-4.0-H-Tiny, quality budget +2 %.
  Result: median 20.51 tok/s (*device*, R49). Do not edit its files; the R-log links to them by path.
- **Campaign 2 (current, planning):** 25 tok/s on a 7B-class model, file not changed, quality budget +5 % over the
  served file's stock perplexity. Goal, terms and open decisions: [campaigns/02-25toks-5pct/CAMPAIGN.md](campaigns/02-25toks-5pct/CAMPAIGN.md).
  Nothing in campaign 2 has been measured yet.

## Reading order (about 30 minutes)

1. [campaigns/02-25toks-5pct/STATUS.md](campaigns/02-25toks-5pct/STATUS.md): state, decisions, next actions.
2. [campaigns/COMMON_RULES.md](campaigns/COMMON_RULES.md): the rules that bind every campaign.
3. [campaigns/02-25toks-5pct/CAMPAIGN.md](campaigns/02-25toks-5pct/CAMPAIGN.md), [feasibility.md](campaigns/02-25toks-5pct/feasibility.md),
   [candidates.md](campaigns/02-25toks-5pct/candidates.md), [knowledge.md](campaigns/02-25toks-5pct/knowledge.md).
4. When you need detail: [docs/overview.md](docs/overview.md), [docs/levers.md](docs/levers.md),
   [docs/negative-results.md](docs/negative-results.md), [docs/runtime-knobs.md](docs/runtime-knobs.md),
   [reports/README.md](reports/README.md) (index of R1 to R49), [RULES.md](RULES.md) (campaign 1's full rules).

## Rules you must keep (short form; the full text is in the files above)

- **The served model file is frozen.** Speed comes from the runtime. A different file or model is proposed to the
  owner first.
- **Budgets are written before results and never move afterwards.** Campaign 2: PPL ≤ stock served PPL × 1.05,
  one budget for everything that changes numerics; hard ceiling F16 × 1.20.
- **Label every number:** *device* (measured on the phone), *host* (laptop, relative only), *estimated*
  (arithmetic). Never present a host number or an estimate as a phone number, and never invent a measurement.
- **Predict first.** Prediction, pass rule and kill rule go into `campaigns/02-25toks-5pct/prereg/` and are
  committed (and tagged `prereg-R<n>`) before any device run. Misses are scored as written.
- **Every runtime change is bit-identical unless declared a numerics change**, checked with `llama-perplexity`
  per-chunk values at `-ub 1`, `-ub 32` and `-b 512`, and is off unless switched on by an environment variable.
- **Fix it in the runtime, never in the user's phone.**
- If the goal looks out of reach, say so with numbers. Do not bend a rule to make a number pass.
- Results are appended; history is not rewritten. Corrections are dated notes.
- R-numbers continue from campaign 1: the first entry of campaign 2 is **R50**.

## Who does what

- **The owner** runs everything that needs hardware: phone benchmarks (the app's research kit, device JSON back into
  `reports/device/`) and full-set perplexity runs on the owner's laptop (Apple M4). Ask for them with exact
  commands and the pre-registration commit to run against.
- **You** do research, plans, pre-registrations, patch and kernel code, scripts, parsers, analysis, and keep the
  repository current. You cannot measure speed yourself.
- The owner writes to you in Korean; reply in Korean. Repository documents are in English.

## Working in the repository

- Work on a branch, push it, open a PR; the owner merges. Do not push to `main`.
- Commit messages end with the attribution lines the session's system reminder gives you.
- **End every session by updating [STATUS.md](campaigns/02-25toks-5pct/STATUS.md)**: what was done, what was decided,
  what is open, the next actions, and the session log line. That is how the next session continues.
- Do not move or rename files of campaign 1. Add new files.
- `patches/llama.cpp/0001-cpu-runtime.patch` (5,699 lines, base `620d2225c`) is not edited; new work goes into a
  new series (see CAMPAIGN.md).
- Do not put the owner's e-mail address or private paths into tracked files.

## Environment notes (observed 2026-11-12; they may differ in your session)

- Reading a public repository needs no attachment; pushing needs the repository attached with push access.
- Pushing **tags** was refused (HTTP 403) from the cloud session, and the GitHub GraphQL API is blocked (use the REST
  API). The owner creates tags (`goal1-final` at `686f17d` is still to be created; `prereg-R<n>` tags likewise).
- Running repository scripts (for example `tools/research_parse.py`) was blocked by the session's safety classifier
  as external code. Reading the data with your own short code worked; ask the owner to run repository tools.
- The cloud workspace reaches only allowlisted hosts. Large model downloads (GGUF, F16) and full wikitext-2
  perplexity runs belong on the owner's laptop. Do not claim to have run them.
- Check with a web search before stating anything about the current state of models or llama.cpp support; the
  candidate list in `candidates.md` is dated.
