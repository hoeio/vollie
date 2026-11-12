# Docs

Start with the [overview](overview.md). Everything here is either original documentation or a page that was
moved or derived from the original files, and each page says which.

| page | what | origin |
|---|---|---|
| [run-on-your-phone.md](run-on-your-phone.md) | run the benchmark or the research kit in the app, read the result, and share it | derived from the kit card and tuning page |
| [overview.md](overview.md) | how the project got to 20 tok/s, in plain language | derived from findings, README, R-log |
| [findings.md](findings.md) | what limits decode on an 8 GB phone, in the order things were found | original |
| [levers.md](levers.md) | the levers that survived, with outputs and device effects | moved from the README, unchanged |
| [negative-results.md](negative-results.md) | what was tried and dropped, with R-numbers | moved from the README, unchanged |
| [round5-comparison.md](round5-comparison.md) | vanilla llama.cpp vs Vollie on 2026-11-04, with its notes and footnote | moved from the README, unchanged |
| [runtime-knobs.md](runtime-knobs.md) | every environment variable of the patched CPU module | original |
| [tuning.md](tuning.md) | how the app's on-device speed test picks settings | original |
| [workspace.md](workspace.md) | the private `lab/` layout the logs and tools refer to (not in git) | original |
| [glossary.md](glossary.md) | terms used in the logs | derived |
| [img/](img/make_charts.py) | the two README charts and the script that draws them | new |

These pages describe **campaign 1** (20 tok/s). Campaign 2's documents are in
[../campaigns/](../campaigns/README.md).

Related, outside this folder: the research log is indexed in [../reports/README.md](../reports/README.md), the
rules in [../RULES.md](../RULES.md) (campaign 1) and [../campaigns/COMMON_RULES.md](../campaigns/COMMON_RULES.md)
(every campaign), the scripts in [../tools/README.md](../tools/README.md).
