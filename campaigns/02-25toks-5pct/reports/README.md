# Campaign 2 research log

Entries are appended, never rewritten. Format: one `### R<n> <title>` heading per entry, in numeric order, so a
link to an R-number can be anchored; the first entry is **R50**. Each entry quotes its pre-registration file
and says whether it passed, was falsified or stayed unresolved, misses included. Labels: *device*, *host*,
*estimated*.

### R50 Host screen: one served file per model (OLMoE, Granite; LFM2.5 pending)

- **Date:** 2026-11-12 to 2026-11-13. **Label:** every number in this entry is *host* (Apple M4 Pro, 24 GB).
- **Rule written before the run:** [memo-01](memo-01-candidates-and-host-screen.md) part C, committed at `67d2ce2`
  before any screen result (no tag: the owner creates tags). Not a device round, so there is no `prereg/` file;
  the first device pre-registration is R51.
- **Who ran it:** the lead session, on the owner's laptop, with the owner's permission (2026-11-12). The earlier
  assumption that the session was a cloud container was wrong.
- **Build:** clean worktree of llama.cpp `620d2225c`, **unpatched**, CPU only, `armv8.2-a+dotprod+fp16`,
  `-DGGML_NATIVE=OFF` (the flags of `tools/reproduce.sh`); `llama-perplexity -c 512 -b 512 -t 8 -ngl 0`, all
  chunks, `n_seq=1` in every log. Text: `wiki.test.raw`, sha256 `173c87a5…b9e7dd08`. The same build made the
  mixed files with `llama-quantize` (recipes in the model cards). All downloads matched the sha256 values in
  memo-01 B2.
- **Rows:** R = 16-bit reference, P = publisher Q4_0, M = mix (dense Q4_0, experts Q3_K, head Q6_K) without an
  imatrix, Mi = the same with a stock imatrix.

| model | row | bytes | sha256 | PPL | % over R | rule (≤ R × 1.08) |
|---|---|---|---|---|---|---|
| OLMoE-1B-7B-0125-Instruct (560 chunks) | R (F16) | 13,844,566,016 | `b9258178…` | 10.9420 ± 0.0896 | 0 | limit 11.8174 |
| | P | 3,928,038,048 | `df43bd73…` | 11.3784 ± 0.0931 | +3.99 | pass |
| | M | 3,072,399,360 | `847e3ce5…` | 11.9321 ± 0.0979 | +9.05 | **fail** |
| | Mi | 3,072,399,552 | `5abf710b…` | 11.7146 ± 0.0960 | +7.06 | pass, **chosen** |
| Granite-4.0-H-Tiny, stock (565 chunks) | R (BF16) | 13,891,206,272 | `f71ef0b7…` | 8.7816 ± 0.0629 | 0 | limit 9.4841 |
| | P | 3,962,938,208 | `85c1f548…` | 9.1045 ± 0.0651 | +3.68 | pass |
| | M | 3,160,777,568 | `7aaeddf6…` | 9.2758 ± 0.0663 | +5.63 | pass |
| | Mi | 3,161,269,344 | `8df94de7…` | 9.2266 ± 0.0664 | +5.07 | pass, **chosen** (lower of M, Mi) |

- **Verdict, as the rule is written:** OLMoE: M fails, Mi passes, so Mi is the served file. Granite: M and Mi
  pass, Mi is lower, so Mi is the served file. Budgets (served × 1.05) and hard ceilings (R × 1.20):
  OLMoE **12.3003** / 13.1304; Granite **9.6879** / 10.5379. Cards:
  [olmoe](../../../models/olmoe-1b-7b-0125-instruct-mi.md),
  [granite](../../../models/granite-4.0-h-tiny-stock-mi.md).
- **What it says:** OLMoE's small experts lose more at 3 bits (P to Mi +3.0 %, Granite +1.3 %), as feared, but
  the file still passes with an imatrix. Stock Granite sits lower than campaign 1's abliterated file on both
  rows (campaign 1: F16 9.0438, served 9.6051).
- **Not part of the gate, noted:** host wall time per pass was 1.2 s (OLMoE P) to 46 s (Granite R); the first
  hours ran under memory pressure on the laptop (swap nearly full), which changes time, not perplexity. One
  OLMoE Mi run was cut at chunk 481 by a session time limit and re-run from the start; one Granite M run was cut
  at chunk 25 and re-run. No result was taken from a cut run.
- **Open:** LFM2.5-8B-A1B rows P, M and Mi are finished and R (BF16) is running; they are appended as a dated
  addendum when R ends. `en.txt` sanity values for the chosen files are not measured yet. M files and the
  16-bit references of OLMoE and Granite were deleted for disk space (rebuildable from pinned sources).

Tool note: `tools/cost.py` on the chosen files gives weight bytes per token of 590 MB (OLMoE Mi) and 806 MB
(Granite Mi, identical to campaign 1's d6-e6).

#### R50 addendum, 2026-11-13: LFM2.5-8B-A1B

Same build, flags and text as R50; 575 chunks, `n_seq=1` in every log; all numbers *host*. P is the publisher's
file already on the laptop (sha256 equal to memo-01 B2); the imatrix for Mi is unsloth's.

| row | bytes | sha256 | PPL | % over R | rule (≤ R × 1.08 = 35.6021) |
|---|---|---|---|---|---|
| R (BF16) | 16,947,260,640 | `2b471037…` | 32.9649 ± 0.3211 | 0 | **suspect, see below** |
| P | 4,844,678,368 | `48ed1465…` | 28.4864 ± 0.2672 | -13.59 | pass |
| M | 3,815,238,880 | `046da7eb…` | 28.9005 ± 0.2684 | -12.33 | pass, **chosen by the rule as written** (lower of M, Mi) |
| Mi | 3,817,074,144 | `37ec82bf…` | 30.5696 ± 0.2895 | -7.27 | pass |

- **The reference row is not credible.** A 16-bit model that is 14 to 16 % worse than its own 4-bit quantizations
  is not a quantization effect. M was built from this same BF16 file and scores 28.9005, so the weights are
  sound; the likely cause is the BF16 inference path for `lfm2moe` on this CPU build (not investigated, not
  confirmed). Granite's reference was BF16 too and behaved normally (8.7816 against 9.1045 for its Q4_0).
- **Verdict as written:** M and Mi are both below the limit, M is lower, so the rule names `lfm25-M.gguf`. The
  choice between M and Mi does not depend on R. What does depend on R, and is therefore **not established**:
  "% over 16-bit" and the hard ceiling (R × 1.20 = 39.5579). The +5 % budget would be 30.3455
  (28.9005 × 1.05).
- **The stock imatrix made this model worse** (Mi +5.78 % against M), the opposite of OLMoE and Granite.
- **Absolute level:** about 28 on wikitext-2 for a chain-of-thought chat model; only ratios are used here.
- **No card is frozen and nothing is hosted for LFM2.5.** Memo-03 puts it outside the stop rule (*estimated*:
  about 47 ms even with 30 % of expert bytes skipped), so no phone round is planned. Owner, 2026-11-13: record as
  written, flag the reference, leave a re-run with the publisher's F16 file (16.95 GB, about 8.5 h, *estimated*)
  as optional. BF16 and Mi were deleted for disk space; M is kept in `lab/`.

### R51 OLMoE expert-skip frontier on the host

- **Date:** 2026-11-13 to 2026-11-14. **Label:** every number is *host* unless marked *estimated*.
- **Pre-registration:** [R51](../prereg/R51-olmoe-expert-skip-frontier-host.md) (commit `ab7875e`) and
  [amendment 1](../prereg/R51-amendment-1.md) (commit `e16630b`), both pushed before the first sweep log existed
  (the sweep started 2026-11-13 20:43 local). No tags, at the owner's wish.
- **Build and run:** as pre-registered (llama.cpp `620d2225c` + `patches/llama.cpp-c2/0001`, 8 threads, 560 chunks,
  `n_seq=1` in every log). File `olmoe-Mi.gguf` (`5abf710b…`), stock 11.7146, budget 12.3003.
- **Identity:** the `stock` row printed PPL = 11.7146 ± 0.09601, R50's digits. **Hygiene deviation:** the rule
  asked for a counter line with 0 pairs skipped in the stock row; the patch prints its counter only when a skip
  knob is set, so the stock log has none. The identical PPL covers the purpose; the rule as written was not met.
- **Added settings:** `SUM=70` and `SUM=75`, between `SUM=65` (inside) and `SUM=80` (over), which were 13 points of
  bytes apart; this is the addition §1 allows.

| setting | PPL | % over stock | expert bytes skipped | MB per token skipped | budget |
|---|---|---|---|---|---|
| stock | 11.7146 ± 0.0960 | 0 | 0 | 0 | |
| SUM=45 | 11.7390 ± 0.0963 | +0.21 | 1.52 % | 5.1 | inside |
| MASS=950 | 11.7480 ± 0.0964 | +0.29 | 1.60 % | 5.4 | inside |
| SUM=55 | 11.8002 ± 0.0969 | +0.73 | 4.03 % | 13.5 | inside |
| MASS=925 | 11.9213 ± 0.0984 | +1.76 | 8.64 % | 29.0 | inside |
| SUM=65 | 11.9615 ± 0.0986 | +2.11 | 9.27 % | 31.1 | inside |
| SUM=70 (added) | 12.1250 ± 0.1004 | +3.50 | 13.07 % | 43.8 | inside |
| **MASS=900** | **12.1031 ± 0.1005** | **+3.32** | **13.39 %** | 44.9 | inside, **f\*** |
| SUM=75 (added) | 12.3170 ± 0.1025 | +5.14 | 17.48 % | 58.6 | over by 0.0167 |
| MASS=850 | 12.4142 ± 0.1038 | +5.97 | 19.96 % | 66.9 | over |
| SUM=80 | 12.5456 ± 0.1049 | +7.09 | 22.26 % | 74.6 | over |

- **f\* = 13.39 %** (`MASS=900`, +3.32 %). The budget crossing lies between 13.39 % and 17.48 %, 4.1 points
  apart, so the round is resolved by the 5-point rule.
- **Prediction 1 (central 10 %, band 6 to 15 %): hit**, in the upper part of the band.
- **Prediction 2 (SUM lower than MASS at equal bytes, by 0.5 to 3 % of stock PPL): miss.** The two rules lie on
  one curve within noise, and where they can be compared MASS is the slightly better one (`MASS=900`: 13.39 % at
  +3.32 %; `SUM=70`: 13.07 % at +3.50 %).
- **Lever status by §4:** not "sufficient at the central estimate" (f\* < 16 %) and not "closed" (f\* ≥ 5 %).
  By §7 the next round is the device baseline of `olmoe-Mi`.
- **In milliseconds (*estimated*, memo-03, not part of the scoring):** 13.39 % of expert bytes is about 3.9 ms at
  0.29 ms per 1 %; memo-03's central need is 16 %, so the central estimate ends about 0.8 ms short of 38.0 ms,
  far inside that model's own uncertainty (it needs 0 to 41 %). The host cannot decide this; the phone can.
- **The 4-chunk smoke values were pessimistic** (`SUM=65`: +4.4 % on 4 chunks, +2.11 % on the full set).
- **Not done:** `en.txt` for `MASS=900`; no setting is served on this round alone.

#### R51 amendment 1 (publisher Q4_0 file): void

All nine rows of the sweep on `OLMoE-1B-7B-0125-Instruct-Q4_0.gguf` printed the stock value, 11.3784 ± 0.09306,
and no row has a counter line: **the skip knobs had no effect on this file.** The identity check passes
trivially and there is no frontier, so the amendment's prediction is neither hit nor missed; the part is void.
Cause not yet established. The likely one: on ARM, Q4_0 expert tensors take llama.cpp's repacked path, which the
c2 patch does not touch. If so it also revives a campaign-1 finding for this file: repacking all-Q4_0 weights
(3.6 GB then) got the app killed on the 8 GB phone (`models/granite-4.0-h-tiny-d6-e6.md`), which is why campaign 1
put its experts in a type without a repack path. Tom is asked to establish the cause.

### R52 OLMoE on the stock runtime, first device run (kit r17)

- **Date:** 2026-11-14, 11:07 to 11:12 local. **Label:** *device* unless marked. Raw report:
  [device/r17_2026-11-14.json](device/r17_2026-11-14.json); parse with `tools/c2_research_parse.py`.
- **Pre-registration:** [R52](../prereg/R52-olmoe-stock-baseline-device.md), commit `7776632`, pushed before the
  APK was handed over. The owner ran kit r17 itself, so amendment 1 (scoring from r18) does not apply.
- **Checks that passed:** model sha256 and both library hashes as pre-registered; `repack_loaded` true in `stock`,
  false in `norepack`; reply text identical in all eight repack-on runs and within the two repack-off runs;
  prompt 156 tokens; 3 threads.

| group | runs | printed tok/s | ms per token (decode_ms / 32) | major faults per run | big-core MHz at run end |
|---|---|---|---|---|---|
| `cold` | 1 | 3.84 | 251 | 16,794 | 1,440 / 1,420 |
| `stock` | 5 | 3.61, 3.34, 2.70, 3.38, 3.08: **median 3.34** | 267 to 358 | 19,583 to 35,447 | 1,440 falling to 1,075 |
| per-token runs | 2 | 3.76, 3.30 | median token gap 247 and 258; fastest single tokens 116 and 124 | 20,818 and 27,633 | 1,075 / 960 to 1,075 |
| `norepack` | 2 | 2.93, 2.33 | 330, 414 | 19,826 and 29,623 | 1,075 to 1,440 / 844 to 960 |

- **Prediction (13.8 tok/s, band 12.1 to 15.0): the measured median is 3.34, a factor of four below the band.**
- **Verdict as the rule is written: void.** §4 voids the round if fewer than three `stock` runs meet the hygiene
  of §6, and none does: every run has major faults throughout the decode tokens, and the one run for which the
  gate waited timed out. The void is recorded together with the plain fact that the prediction would have
  been missed by a wide margin: the cost model priced bytes at memory speed and assumed a resident model.
- **Second prediction** (repack off slower by 0 to 5 ms): unresolved; both groups are fault-bound.
- **Cause ([memo-13](memo-13-r52-diagnosis.md)):** page cache and heat, not arithmetic.
  - The benchmark touches almost the whole expert file: the 156-token prefill alone reads 968 of 1,024 expert
    slices, 2,617 MB; with the 32 decode tokens 2,725 MB (*host* trace). The phone held 1.5 to 1.7 GB of the file
    resident, with at most 2.35 GB of page cache and no free swap. Campaign 1's fast Granite runs (R49) had 2.7
    to 3.0 GB of page cache; Granite's own working set on this prompt is 2,569 MB (*host* trace), so campaign 1
    never made the model resident from the runtime: the phone had room on that day.
  - Thermal headroom was 0.97 to 1.10 for the whole session (the kit's "moderate" threshold is 0.93); the A78
    clock cap stepped down 1,440, 1,209, 1,075 MHz and did not come back. In r16, runs with 35k to 74k faults
    still ended at 2,112 MHz, so the low clocks here are thermal.
  - **Kit bug:** the cool-down gate did not wait. Each run ends by reading the thermal headroom and the next
    gate reads it again within about 50 ms; Android returns NaN when polled more than once a second, and the kit
    took that as "cool". Kit r18 has the same bug. The kit had also just downloaded and hashed 3 GB before the
    first run.
  - Template: the GGUF carries only the BOS id; llamadart falls back to the text `<s>` (three tokens). No effect
    on the expert set or on speed (*host*).
- **What follows (R52 §7, "optimistic miss / find the term first"):** kit r18 is withdrawn before being run
  ([R54 amendment 1](../prereg/R54-amendment-1.md)). The next kit fixes the gate, reports residency, and runs
  campaign 1's runtime beside the c2 series at the owner's instruction.

Memos (not R-entries): [memo-01, candidate re-check and host screen plan](memo-01-candidates-and-host-screen.md);
[memo-02, which campaign-1 levers apply to `olmoe` and `lfm2moe`](memo-02-lever-applicability.md);
[memo-05, why skip was inert on Q4_0 experts: repack](memo-05-q4-experts-repack.md);
[memo-08, predictions for the device baselines of OLMoE; Granite verdict proposed](memo-08-device-baseline-predictions.md);
[memo-06, adb CLI kit (abandoned: the phone cannot be attached)](memo-06-device-cli-kit.md);
[memo-09, how a research APK is built](memo-09-research-apk-path.md);
[memo-10, kit r17](memo-10-kit-r17.md);
[memo-12, kit r18 (lever kit)](memo-12-kit-r18.md);
[memo-13, diagnosis of R52: page cache, heat, gate bug, campaign 1's runtime on OLMoE](memo-13-r52-diagnosis.md);
[memo-03, lever pricing](memo-03-lever-pricing.md);
[memo-04, OLMoE expert-skip](memo-04-olmoe-expert-skip.md);
[memo-07, campaign-1 exact levers in the c2 series, identity on OLMoE](memo-07-c2-exact-levers.md);
[memo-11, kit interface in the runtime (0008) and the untied two-stage head (0009)](memo-11-kit-runtime-and-head.md).
