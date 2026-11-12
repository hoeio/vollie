# Reports: a readable index of the research log

> **This page is derived from the original files in this folder.** It was written for readability: the
> one-line outcomes below are condensed from [03_final.md](03_final.md) (the R-log), its device JSONs and the
> memos next to it. It adds no measurements and is not a substitute for the originals. Every row names the
> entry it comes from; when a row and its entry differ, the entry wins. The original files in this folder
> (the numbered reports, `memos/`, `device/`, `results.csv`, the audit) were **not modified** by the change
> that added this page.
>
> Labels follow [../RULES.md](../RULES.md): *device* = measured on the phone, *host* = laptop, *estimated* =
> arithmetic. Terms are in the [glossary](../docs/glossary.md).
>
> This folder is **campaign 1's** log (R1 to R49) and is closed. Campaign 2's log starts at R50 in
> [../campaigns/02-25toks-5pct/reports/](../campaigns/02-25toks-5pct/reports/README.md).

## What is in this folder

| file | what it is |
|---|---|
| [00_model_choice.md](00_model_choice.md) | Phase 0: why Granite-4.0-H-Tiny was chosen (locked); the previously shipped 4-expert file failed the gate (+12.4 %) |
| [01_vollie_vs_llamacpp.md](01_vollie_vs_llamacpp.md) | Phase 1: how the app runs inference compared with stock llama.cpp |
| [02_methods.md](02_methods.md) | Phase 2: a survey of methods (17 ranked) for this model on the SD888 |
| [03_final.md](03_final.md) | **the R-log**: status table, setup, and every round R1 to R49 with pre-registrations and verdicts |
| [04_campaign.md](04_campaign.md) | the campaign plan to 20 tok/s: system model, roofline, idea list (written 2026-11-01) |
| [05_compute_less.md](05_compute_less.md) | the research after the owner's restatement of 2026-11-02: compute less per printed token; holds several pre-registrations |
| [memos/](memos/) | Bonnie's research memos (round plans and reads), cited by the R-log entries below |
| [device/](device/) | raw JSON of every research-kit round, `r1` to `r16`, plus `r13b` and `r14b` |
| [results.csv](results.csv) | one row per result, wins and losses |
| [mirror_audit_2026-11-05.md](mirror_audit_2026-11-05.md) | an audit of this repository's tracked files, 2026-11-05 |

## Reading `03_final.md`

- It is chronological, appended to as work went on, and is **not** a clean top-to-bottom document.
- The header line "Last updated" (2026-11-07, R43) is older than the file's last entry (R49, 2026-11-08).
- §6 "What worked, what didn't" and §7 "Claims that still need on-device verification" are a snapshot of
  2026-11-02 at R22 (the header says so); they sit in the middle of the file, between R43 and R44.
- For the current picture read §1 (rewritten 2026-11-05) and then R47-R49 at the end.
- The log has its own index in §5; it lists R1 to R28 only. The table below covers R1 to R49.

## R-number index

### R1-R16: app builds, profiling and the file (2026-10-31 to 2026-11-02)

The log's own summary table in §5 has a row for each of R1 to R23; the rows below condense them.

| R | what | outcome |
|---|---|---|
| R1 | first Granite build (D1-e5 file, repack on for 8 GB) | first Granite number on the phone: 7.4 tok/s writing, 14 reading (4 threads); below the 10-14 estimate |
| R2 | thread count changed in place on heat (`setThreads`) | not verified on the device at the time |
| R3-R7 | own CPU module with tiny-op, then the race fix and tiny-op v2 | tiny-op roughly doubled speed at 2 threads (3.2 to 7.0 tok/s); R5 found and fixed a race that broke small prompt batches; v2 cut syncs per token from 1,909 to 874 |
| R8-R9 | per-op profiler, then page-fault counters | experts do not fit the page cache: 354-1,181 major faults per token |
| R10 | file D6-e6 (Q3_K experts, 6 experts per token) | the served file; 632 major faults per token at the time |
| R11 | memory fields added to the profile line | superseded before a run |
| R12-R14 | expert I/O: prefetch the routed experts, guess the next layer, guess two more | [docs/findings.md](../docs/findings.md) §3 reports 6.1 to 11.8 tok/s from prefetch and 85 % correct guesses on the device |
| R15 | faster NEON Q3_K dot and the in-generation A/B (`GGML_CPU_AB`) | the A/B mechanism that later rounds depend on |
| R16 | campaign Phase A/B, host only | roofline estimate: the file's DRAM roofline is 46.8 ms per token, so raw 20 tok/s sits at the roofline (*estimated*) |

### R17-R49: rounds on the device (2026-11-02 to 2026-11-08)

"Kit" is the device JSON in [device/](device/). A round has a build/pre-registration entry and a results entry.

| R | round, kit | what | outcome |
|---|---|---|---|
| R17 | round 1 | build and pre-registered predictions: no-barrier expert op, 4-row Q3_K, expert skip knobs | host results; predictions fixed before the run |
| R18 | round 1, `r1` | device results | falsified: barrier latency (0.17-2.0 µs, the wait is imbalance); falsified on compute: 4-row Q3_K (-2.4 %); falsified: expert-guess helper on an A55 (-13 %) |
| R19 | round 2 | build: `MMID` and `Q3K_X4` become defaults; `libllama.so` rebuilt for speculation tests | pre-registered |
| R20 | round 2, `r2` | device results | draft-model speculation loses at every setting (0.48-0.76x); tiny-op 262,144 made the default |
| R21 | host | research reset: compute less per printed token | router-weight adaptive expert count is about 3x more PPL-efficient than a uniform expert count; early exit dead |
| R22 | host | which expert-budget rule | absolute router-weight threshold wins at equal expert count |
| R23 | round 3, `r3` | place the threshold; measure expert-path terms | Q3_K software prefetch and the anonymous hot-expert cache gave no device gain and were closed (see [docs/negative-results.md](../docs/negative-results.md)); the independent-chain Q3_K kernel became a default in round 4 |
| R24 | round 4, `r4` | pre-registration, build, results | Bonnie's verdict: exact plus approved levers fall short of 20 on a bad-cache day; a "x1.40" skip claim was later withdrawn (R25) |
| R25 | round 5, `r5` | vanilla vs Vollie, same phone | official 32: 9.4 to 12.0 to 13.9 tok/s; skip threshold edge 118 (+1.98 %; 119 fails); the ABBA in-run A/B starts here. Table: [docs/round5-comparison.md](../docs/round5-comparison.md) |
| R26-R27 | round 6, `r6` | fast cores capped at 1,555 MHz | same three rows 6.1 to 9.9 to 14.0; experts compute-bound, dense products bandwidth-bound. Host reproduce check verified 2026-11-05 |
| R28 | host | two-stage output head (Q2_K screen) | the first lever that cuts dense bytes inside the rules: PPL +0.19 %, argmax-identical; does not stack with skip 118 under the +2 % budget |
| R29 | round 7, `r7` | device results for the Q2_K/Q3_K head screens | head screens are compute-bound and slower than the stock head; fault instrument worked |
| R30 | round 8 | build (host): repacked Q4_0 screen | host head op roughly halves (1.73 to 0.86 ms) |
| R31 | round 8, `r8` | device results | repacked Q4_0 screen works: head op -1.70 ± 0.13 ms; skip 117 plus head: PPL 9.7886 (+1.91 %) |
| R32 | round 9 | build (host): sampler fast path, static Q4_0 expert copy, finer chunks | pre-registered |
| R33 | round 9, `r9` | device results | sampler fast path hit (-0.77 ms/token outside the graph); finer dense chunks falsified; static Q4_0 copy dead by perplexity |
| R34 | round 10 | build (host): in-register Q3_K kernel (562 to 377 NEON ops per 4-row block) | pre-registered |
| R35 | round 10, `r10` | device results | kernel exact on the device and faster in isolation; speed line unscorable (thrashing day); compiled default from here |
| R36 | round 11 | build (host) | scoring rule from R35 reused |
| R37 | round 11, `r11` | device results | speed line unscorable a second time; kernel exact on the device again |
| R38 | round 12, `r12` | device results, kernel line scored | falsified under the pooled pre-registered rule; the kernel stays the default as an exact change |
| R39 | round 13 | pre-registration (page-head touch, block-gated A/B rule) | from [memos/round13_plan.md](memos/round13_plan.md) |
| R40 | round 13, `r13` | device results | unresolved: no fault-free blocks; the bench had run on an A55 (amendment) |
| R41 | round 13b, `r13b` | bench-only re-check | page-latency reading withdrawn; `MMID_TOUCH` parked; the kernel is about 87 % of the expert slot (*estimated*) |
| R42 | round 14 | owner's decisions and pre-registration (fused gate+up op, two chunks per claim) | pre-registered band -0.5 to -1.05 ms |
| R43 | host | dense-numerics screen | router F16 cast passes perplexity (9.7864, text identical on host); Q8_0 cast and low-rank head screen closed |
| R44 | round 14, `r14` | device results | scored sum unresolved (page faults dominated the noise) |
| R45 | round 14, `r14b` | the one allowed re-run | unresolved again; fused op parked as a device claim, kept as an exact default-off knob |
| R46 | round 15 | pre-registration: a measurement round (per-token interleaved A/B, sequential rule) | from [memos/round15_plan.md](memos/round15_plan.md); owner approved a fresh kernel pre-registration |
| R47 | round 15, `r15` | device results | fused op shown on the device and joins the compiled default; kernel experiment passes (-3.530 ± 0.117 ms at throttled clocks, about -2.67 at 2,112 MHz by scaling, *estimated*); router F16 cast changed the device's greedy text |
| R48 | round 16 | pre-registration: the official 32 under the shipping build, decomposed | claim rule fixed before the run; the APK was frozen with byte checks |
| R49 | round 16, `r16` | device results | **20.51 tok/s median over 10 plain runs: 20 reached**; the prediction (18.0-20.0) missed high; the decomposition line was unscorable. A correction note after the entry explains why slow runs are slow |

The log also records a production build (2.3.0+45) in a paragraph after R49.

## How to check a row

1. Open the entry named in the row in [03_final.md](03_final.md) (search for `### R<number>`).
2. For device numbers, the raw data is in [device/](device/) and can be re-read with
   `python3 -I tools/research_parse.py <json>` (see [../tools/research_kit.md](../tools/research_kit.md)).
3. For one-line results across all rounds, see [results.csv](results.csv).
