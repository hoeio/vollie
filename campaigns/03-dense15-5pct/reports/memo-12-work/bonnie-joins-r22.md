# c3 memo-12 work, round 22 (Bonnie): the claim-run configuration sheet for campaign 3; does a pack file shorten the load

- **Date:** 2026-11-24. **Author:** bonnie, `claude-fable-5-2`. Follows [bonnie-joins-r21.md](bonnie-joins-r21.md). Not
  committed. No verdict here. **Ebony has not reviewed this file.**
- **Labels.** *device*, *host*, *estimated*, *source* as before. Triples are Yi / Llama-2 / DeepSeek.
- **New this round, one reading of existing *device* data:** the `reload_ms` values in campaign 1's and 2's kit
  reports (46 reloads), read with my own short code. Everything else is source and the written terms. Nothing was
  run. No web source.
- **What the sheet is.** A list for the lead to cite or correct when R73 (the first round that loads a dense
  model) is pre-registered. It records what the terms fix, what the R73 plan sets, what the app does by default,
  and what is still open. It decides nothing.

## 0. Answer first

1. **The sheet is §1: 38 rows.** 12 are fixed by the campaign's terms, 17 are set by the R73 plan or campaign 2's
   kit, 3 are the app's defaults that nobody chose for a dense model, and **6 are open**. §1.6 lists eight
   decisions: those six and two settings (context size, gate length) that are set but worth confirming.
2. **Three rows of the R73 plan need a correction or a sentence** (§1.5): it calls the forced copies "stock" (the
   shipped app refuses them for these files); it gives the cache sizes for F16 (the app sets Q8_0); and its host
   reference is the freeze, which differs from the phone in four ways.
3. **The open decisions, in the order they block each other:** the body's load path (A / B / C); the head's
   placement; the reference the budget is read against; the cache type to claim with; then four small ones.
4. **A pack file would shorten every load after the first, but campaign 2's records cannot say by how much.**
   Its 46 reloads took 2.6 to 12.9 s (medians 8.8 s Granite, 7.0 s OLMoE, *device*) with copies of only 0.24 to
   0.54 GB, so most of a load is not the copy. For a dense subject I put today's load at **11 to 31 s** and a
   mapped load at **7 to 9 s** (*estimated*). Load time is outside the claim's metric.

## 1. The sheet

Status: **T** = fixed by the campaign's terms or common rules; **P** = set in the R73 plan
(`reports/memo-10-work/clyde-r73-plan.md` §2); **K** = campaign 2's kit as it stands in `lab/app-c3/lib/research.dart`;
**A** = the app's default, not chosen for a dense model; **O** = open. "Freeze" says whether the S3 freeze (the
budget's reference) ran the same thing.

### 1.1 What is claimed and how it is counted

| # | item | value | source | status |
|---|---|---|---|---|
| 1 | goal | median of 10 plain runs ≥ 12.0 tok/s, on each of three subjects | `CAMPAIGN.md` terms table (claim rule; Amendment 1) | T |
| 2 | metric | llamadart `decodeTokensPerSec`, 32 tokens, greedy | `CAMPAIGN.md` (metric); `lab/app-c3/lib/assistant.dart:807` (`_benchTokens = 32`), `:1163`, `:560` (`temp: 0`) | T |
| 3 | prompt | the official prompt, 702 bytes; 147 / 179 / 147 tokens before the chat wrapper (*host*, Ebony r4) | `tools/c2_official_prompt.txt`; `assistant.dart:1147` (`_benchPrompt`) | T |
| 4 | path through the app | the plain completion call (`_complete`), run kind `official32` | `assistant.dart:555, 630`; `research.dart:16, 217` | K |
| 5 | positions and arms fixed before the run; never a best run | – | `CAMPAIGN.md`; `COMMON_RULES.md` rule 11 | T |
| 6 | multi-token methods | count committed tokens, greedy output identical, scored on a fixed prompt set as well | `CAMPAIGN.md` (metric) | T |
| 7 | budget | PPL ≤ stock served PPL × 1.05 at `-b 512` and `-ub 1`: 8.8069 / 8.1307 / 9.1677 and 8.8048 / 8.1314 / 9.1671 | `CAMPAIGN.md`; `results.csv` (S3 freeze, *host*) | T |
| 8 | ceiling | F16 × 1.20: 9.8128 / 8.9918 / 10.0938 | the same | T |
| 9 | served files | the frozen Q4_0 files, sha256 `702cc2eb…`, `8eb4952f…`, `f403ddc2…` (DeepSeek unmodified) | `tools/c3_freeze.sh`; `results.csv` | T |

### 1.2 Phone state and protocol

| # | item | value | source | status |
|---|---|---|---|---|
| 10 | phone | Galaxy Z Flip 3, Snapdragon 888, 8 GB | `CAMPAIGN.md` | T |
| 11 | power saving | off | `CAMPAIGN.md` (claim rule) | T |
| 12 | charger | out | the same | T |
| 13 | deep-sleeping list | state recorded | the same | T |
| 14 | start | cool start (30 min of rest in the probes' pre-registrations); campaign 2's claim was accepted with a warm start, stated as a deviation | R74 and R75 pre-registrations; campaign 2 `CAMPAIGN.md` line 144 | P |
| 15 | gate before each timed run | thermal headroom under 0.65, polled every 5 s, at most 90 s | `assistant.dart:1337`; `research.dart:74–77` | K (the probe kits waited up to 180 s) |
| 16 | page hold through the gate | on for scored runs | `research.dart:158–161, 242` | K; it works on mapped files only (r8 §2.7) |
| 17 | block per model | load; 2 unscored warm-ups; gated `official32` runs; profiled runs after | R73 plan §2 | P |
| 18 | storage | the model on internal storage (an executable map is refused elsewhere) | campaign 2 R67; `research.dart:133` | K |

### 1.3 Build and load

| # | item | value | source | status | freeze |
|---|---|---|---|---|---|
| 19 | CPU module | built with KleidiAI, variant `android_armv8.2_2` | `lab/app-c3/tool/build_cpu.sh:19, 26` | K | **no** (KleidiAI off) |
| 20 | **the body's load path** | A: the app's rule refuses the faster layouts for a file over 2.7 GB and reads plain rows. B: the kit forces the copies (`repack: true`). C: the same layout from a mapped side file | `lib/device.dart:591`; `assistant.dart:455–456`; `research.dart:168`; r17, r21 | **O**; the R73 plan sets B | repack type, not KleidiAI |
| 21 | **the head** | in KleidiAI as int8 (default under B and C); or the repack type with `VOLLIE_HEAD2_REPACK=1`; plain Q6_K under A | `kleidiai.cpp:1793–1800`; `research.dart:108`; r12 | **O** | repack type |
| 22 | load mode | mapped file | `llamadart/…/load_param_helpers.dart:84` | A | same |
| 23 | load-time environment | none in the kit except the head knob for OLMoE | `research.dart:104–108` | K | – |

### 1.4 Context and run-time settings

| # | item | value | source | status | freeze |
|---|---|---|---|---|---|
| 24 | context size | 512 in the R73 plan; the kit's constant is 4,096; the app's rule gives 4,096 or 8,192 | plan §2; `research.dart:193`; `device.dart:599–602` | P | 512 |
| 25 | **KV cache type** | Q8_0 for keys and values (the app, on the CPU) | `assistant.dart:471–472` | **O** (A today) | **F16** |
| 26 | flash attention | on (required by row 25) | `load_param_helpers.dart:104–117`; `llama-context.cpp:4064–4067` | A | on ("auto" resolved to enabled, r19) |
| 27 | batch | logical 2,048, micro 512; the prompt is one batch, the decode is batch 1 | `llamadart/…/model_params.dart:466, 471` | A | `-b 512` and `-ub 1` |
| 28 | threads | 4, one per cpu 4 to 7 (`VOLLIE_POOL=1`, a per-thread mask) | plan §2; `research.dart:129` | P | 8 (no effect on values) |
| 29 | chat template | the model's own | plan §2; `assistant.dart:476` | P | none (raw text) |
| 30 | `GGML_TINY_OP=262144`, `GGML_EXPERT_IO=1`, `VOLLIE_SAMPLER_BYPASS=1` | campaign 1's shipping settings without numerics | `research.dart:113`; plan §2 ("baseline env") | P | unset; identical by earlier checks on the repack path |
| 31 | `VOLLIE_KV_DONTNEED=1` | the cache is cleared by releasing pages before every prompt | `research.dart:125`; plan ("kv clear every run") | P | unset; identical by campaign 2's check |
| 32 | `VOLLIE_EXEC_MAP`, `VOLLIE_TOUCH`, `VOLLIE_NICE`, `VOLLIE_SCHED` | campaign 2's residency and scheduling factors | `research.dart:133–142` | **O** for a dense claim: the plan names pool, pin and hold only | unset |
| 33 | pin guard | arm B of R73: `VOLLIE_PIN_GUARD=500`, `VOLLIE_PIN_WARM=0` (patch 0004) | plan §2 | P (a factor) | – |
| 34 | counters and lines | `GGML_C2_IO=1`, `VOLLIE_LINES=stderr`, profiler in the profiled runs only | `research.dart:116, 357`; `assistant.dart:745` | K | – |
| 35 | speculative session, draft model, rollback slots | none (0) | `assistant.dart:473–475` | K | – |
| 36 | **compute buffer release** | not built (`VOLLIE_CB_DONTNEED`, r20) | – | **O** | – |
| 37 | **GPU lane, width 2, guess head** | off: R73 is the stock decode | plan §2 | P | – |
| 38 | **the host reference the claim's budget is read against** | campaign 2's rule: the freeze's absolute budget stands, and a second reference is measured on the phone's path; a setting must be inside both | campaign 2 R61; r9 §1 | **O**: the second reference does not exist yet | – |

### 1.5 Three corrections to the R73 plan's text

| plan text (`clyde-r73-plan.md` §2) | what the source says | suggested wording |
|---|---|---|
| "repack on (`KitModel.repack: true`), i.e. llama.cpp's default: this is stock" | the shipped app would not make these copies for a 3.5 to 4.0 GB file (`device.dart:591`); the kit forces them. It is upstream's default and R72's body, not the app's behaviour | "repack forced on by the kit; the app's own rule refuses it for these files (choice B of memo-12 r17)" |
| "F16 KV at 4096 is 2.0 GiB … at 512 it is 256 / 240 / 32 MiB" | the app sets Q8_0 on the CPU: 136 / 128 / 17 MiB at 512, 1.06 / 1.00 / 0.13 GiB at 4,096 | state the type; give the Q8_0 sizes |
| the K0 check against 6.90 / 6.21 / 6.22 and budgets from the freeze | the phone's path differs from the freeze in KleidiAI, the head, the cache type (rows 19, 21, 25); the app's rule is a fourth (row 20) | cite the second reference of row 38 once it exists |

### 1.6 The open decisions, in blocking order

| # | decision | options | what it waits on | who |
|---|---|---|---|---|
| 1 | body load path (row 20) | A / B / C | the owner (it has gone to the owner); for C: storage, and the r18 residency round | owner |
| 2 | head placement (row 21) | KleidiAI int8 / repack type | Tom's stock run and H1; r12 | lead |
| 3 | second reference (row 38) | which flags define it: KleidiAI, `-fa on -ctk q8_0 -ctv q8_0`, the head as decided; 40 chunks or full set | decisions 2 and 4; Tom's runs | lead |
| 4 | cache type to claim with (row 25) | Q8_0 as the app has it / F16 to match the freeze / Q8_0 keys with F16 values | Tom's KV screen and host decode | lead |
| 5 | residency factors for a dense claim (row 32) | which of exec map, touch, nice are part of the protocol | choice A or C makes exec map and the hold apply; under B they do nothing for the body | lead |
| 6 | context size in a claim (row 24) | 512 as planned | settled unless the fixed prompt set needs 1,024 (the plan uses 1,024 there) | lead |
| 7 | compute buffer release (row 36) | build it or not | Tom; the poison check | lead |
| 8 | gate length (row 15) | 90 s as the kit / 180 s as the probes | campaign 2 claimed with 90 s; Ebony r4 §4 expects time-outs later in a block | lead |

## 2. Does a pack file shorten the load on the phone? (next question 3)

### 2.1 What campaign 1's and 2's kits recorded (*device*; `reload_ms`, all reports under `reports/device/` of
both campaigns)

| model | copies made at load | reloads | median | range |
|---|---|---|---|---|
| Granite-Mi (3.16 GB file) | 541 MiB in KleidiAI | 21 | 8.8 s | 5.6 to 12.9 s |
| OLMoE-Mi (3.07 GB file) | 243 MiB in KleidiAI (or split with the repack type) | 25 | 7.0 s | 2.6 to 11.0 s |
| one reload with repack off | none | 1 | 8.2 s | – |

- **Most of a load is not the copy.** The one load without any copy took as long as the median with one. What
  the 7 to 9 s are made of is not split in the reports (the context, a 285 MB cache clear on OLMoE, the
  tokenizer, the file's first pages, the unload before it).
- The two models differ by 1.8 s for 298 MiB more copy, about 6 s per GB. That is confounded by everything else
  that differs between them; I read it as an upper bound.

### 2.2 For a dense subject (*estimated*)

| | today, choice B | choice C, first load | choice C, later loads |
|---|---|---|---|
| fixed part | 7 to 9 s | 7 to 9 s | 7 to 9 s |
| reading and packing 3.4 to 3.8 GB | 4 to 22 s (lower end: r9's 0.7 to 1.5 GB/s pack and a cached file; upper end: 6 s per GB) | the same | none: the file is mapped, and its pages come in during the prompt |
| writing the side file | – | 5.5 to 8.5 s (R69's write rate, *device*) | – |
| **load** | **11 to 31 s** | **17 to 40 s** | **7 to 9 s** |
| first prompt | the body is already in memory | the same | reads the body from storage as it goes: 3.4 to 3.8 GB, a few seconds more on the first prompt after a cold start |

- So C moves 4 to 22 s per load into the first run, once per model, and later loads cost the fixed part.
- **None of it is in the claim's metric** (decode tokens per second after the prompt). It matters for the user,
  for a kit's phone time, and for how much is resident when the first gate starts.
- A direct number costs nothing extra in r18's round: `x5 pl build gbs=` is the pack rate on the phone, and the
  kit can print the load's wall time when a model is first loaded in R73.

## 3. Next questions, ranked

1. **Fold R75 into r18's draft** when its report is in the repository (not there at 04:38 today), with r19's
   idle part and r20's two changes.
2. **What the 7 to 9 s of a load are:** read the app's `_loadOn` and the library's load call for the steps
   (unload, map, context, cache clear, template, warm-up) and attach campaign 2's per-step fields if the reports
   have them. Useful for kit time and for the first gate. Mine.
3. **The fixed prompt set for multi-token methods** (row 6): the terms require it beside the official prompt.
   Where it is defined, its lengths per subject, and what n_ctx and gate it implies. Mine.
4. **Re-read r15 to r17 against Tom's host decode and KV screen** when they exist.
5. **DeepSeek's head; cold read rate; fixed share.**

## 4. Limits

- The sheet reads the app, the kit code and the plan at today's lines. The R73 kit does not exist; a row marked
  P or K is what the plan or campaign 2's code says, not what a built kit does.
- I did not re-read campaign 2's memo-31 for the go rule's own text; row 1 cites the terms table, which points
  to it.
- §2.1 groups reloads by model name only; I did not separate first loads from reloads or check which of them
  followed a different repack setting.
- §2.2's ranges are wide because the copy's share of a load was never measured on the phone.

## 5. Sources (2026-11-24)

Repository: `campaigns/03-dense15-5pct/CAMPAIGN.md` (terms table, lines 40 to 54), `campaigns/COMMON_RULES.md`,
`campaigns/03-dense15-5pct/results.csv`; `reports/memo-10-work/clyde-r73-plan.md` (§2, lines 36 to 60);
`campaigns/02-25toks-5pct/CAMPAIGN.md` (lines 26 to 28, 141 to 144); `lab/app-c3/lib/assistant.dart` (lines 421, 455 to
476, 555 to 560, 630, 745, 807, 1147, 1163, 1337), `lib/research.dart` (lines 16 to 21, 74 to 77, 100 to 142, 158 to
193, 217, 242, 357), `lib/device.dart` (lines 591, 599 to 602), `tool/build_cpu.sh`; `lab/app-c3/third_party/llamadart`
(`load_param_helpers.dart` lines 84, 104 to 117; `model_params.dart` lines 466, 471); `tools/c2_official_prompt.txt`,
`tools/c3_freeze.sh`; campaign 1 and 2 device reports (`reports/device/*.json`, `campaigns/02-25toks-5pct/reports/device/*.json`:
`reload_ms`, `repack_loaded`); `models/granite-4.0-h-tiny-stock-mi.md`, `models/olmoe-1b-7b-0125-instruct-mi.md`;
my r8, r9, r12, r15 to r21; [ebony-joins-r4.md](ebony-joins-r4.md) §3 and §4.
