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

#### R51 amendment 2, 2026-11-14: the frontier without repack (the `-nr` path)

Pre-registration: [amendment 2](../prereg/R51-amendment-2.md), commit `93612e8`, pushed before the first `-nr`
full-set log existed. Same build, text and flags as R51 with `PPL_ARGS=-nr`; 560 chunks and `n_seq=1` in all 13
logs; all numbers *host*.

| file | setting | PPL | % over its own `-nr` stock | expert bytes skipped | R51 value with repack |
|---|---|---|---|---|---|
| `olmoe-Mi` | stock | 11.7135 ± 0.0960 | 0 | 0 | 11.7146 |
| | MASS=900 | 12.1032 ± 0.1005 | +3.33 | 13.39 % | +3.32, 13.39 % |
| | SUM=70 | 12.1315 ± 0.1005 | +3.57 | 13.07 % | +3.50, 13.07 % |
| | SUM=75 | 12.3212 ± 0.1025 | +5.19 | 17.48 % | +5.14, 17.48 % |
| publisher Q4_0 | stock | 11.3793 ± 0.0931 | 0 | 0 | 11.3784 (R50) |
| | SUM=45 | 11.3935 ± 0.0932 | +0.12 | 1.56 % | |
| | MASS=950 | 11.3973 ± 0.0933 | +0.16 | 1.64 % | |
| | SUM=55 | 11.4560 ± 0.0938 | +0.67 | 4.13 % | |
| | MASS=925 | 11.5570 ± 0.0951 | +1.56 | 8.72 % | |
| | SUM=65 | 11.5955 ± 0.0953 | +1.90 | 9.43 % | |
| | **MASS=900** | **11.7442 ± 0.0972** | **+3.21** | **13.42 %** | |
| | MASS=850 | 12.0710 ± 0.1006 | +6.08 | 20.05 % | |
| | SUM=80 | 12.1834 ± 0.1016 | +7.07 | 22.41 % | |

- **Prediction 1** (`olmoe-Mi` `-nr` stock within ± 0.5 % of 11.7146): **hit** (−0.01 %).
- **Prediction 2** (each ratio within ± 0.4 points of R51, bytes within ± 0.3 points): **hit** for all three
  settings (largest shift +0.07 points; bytes equal to two decimals).
- **Prediction 3** (Q4_0 `-nr` stock within ± 0.5 % of 11.3784; f\* central 11 %, band 6 to 16 %): **hit**
  (+0.01 %; f\* = 13.42 % at `MASS=900`, budget 11.3793 × 1.05 = 11.9483). The crossing lies between 13.42 % and
  20.05 %, 6.6 points apart, so by R51's 5-point rule the position of the crossing is **unresolved**; f\* is
  at least 13.42 %. No settings were added: the file is the second candidate and residency (R52) now decides
  first.
- **The `MASS=900` rule:** it stays inside +5.00 % of the `-nr` stock (+3.33 %), so R51's f\* = 13.39 % stands
  on the no-repack path as well.
- **Amendment 1 is thereby re-run:** with repack off the skip acts on the Q4_0 file, and its curve is the same
  as `olmoe-Mi`'s within about 0.2 points at each setting.
- **Hygiene deviation:** the rule asked for "no `CPU_REPACK` buffer line" in every log. At this log level
  `llama-perplexity` prints no buffer lines at all, with or without repack, so the check cannot be made from
  the logs. Indirect evidence that `-nr` took effect: the stock value differs from the repack value in the way
  the 4-chunk comparison predicted, and the skip acted on the Q4_0 file, which it cannot do with repack on.

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

### R53 Untied two-stage head on OLMoE (host)

- **Date:** 2026-11-14. **Label:** *host*. Entered after R55 because the sweep finished later; the numbering
  follows the pre-registrations.
- **Pre-registration:** [R53](../prereg/R53-olmoe-head2-host.md), commit `45ce55d`, pushed before the sweep started.
- **Build and run:** as pre-registered: llama.cpp `620d2225c` + `patches/llama.cpp-c2/0001` to `0009`, repack on,
  `tools/c2_head2_sweep.sh`, 8 threads, 560 chunks and `n_seq=1` in all six logs. File `olmoe-Mi.gguf`.
- **Identity rows, both passed:** `stock` 11.7146 ± 0.09601 and `MASS=900` 12.1031 ± 0.10045, the digits of R50
  and R51. The nine-patch series with every knob unset is therefore identical to the base on the full set, and
  its skip equals the one-patch build's.

| setting | PPL | % over stock | expert bytes skipped | exact head rows per token |
|---|---|---|---|---|
| stock | 11.7146 ± 0.0960 | 0 | 0 | all 50,304 |
| K=1024 | 11.7165 ± 0.0961 | +0.016 | 0 | 6,953 |
| K=2048 | 11.7157 ± 0.0960 | +0.009 | 0 | 12,818 |
| MASS=900 | 12.1031 ± 0.1005 | +3.316 | 13.39 % | all |
| K=1024 + MASS=900 | 12.1053 ± 0.1005 | +3.335 | 13.39 % | 6,936 |
| K=2048 + MASS=900 | 12.1041 ± 0.1005 | +3.325 | 13.39 % | 12,785 |

- **Predictions:** alone within ± 0.10 % of stock: **hit** (+0.016 %, +0.009 %). With skip +3.32 % ± 0.12 points:
  **hit** (+3.335 %, +3.325 %); the two changes add without interaction.
- **Pass rule:** both K pass (alone ≤ +0.15 %, with `MASS=900` ≤ 12.3003). The smaller one, **K=1024**, is the
  head setting to use; with skip it leaves 0.195 of PPL (1.66 points) of the budget.
- **What it is worth:** little. With repack on, K=1024 still computes about 6,950 rows exactly per token; memo-11
  put the saving at about 15 MB of head bytes, about 0.6 ms on the phone (*estimated*).
- **Hygiene:** head rows carry the `c2-head:` counter line and skip rows the `c2-skip:` line. Deviation, as in
  R51 amendment 2: the rule asked for a `CPU_REPACK` buffer line and this log level prints no buffer lines;
  repack on is shown indirectly by the stock value 11.7146 (the no-repack value is 11.7135).
- **Standing:** these patches sit on a stock base. Under the owner's rule of 2026-11-14 ("start from Vollie 1")
  the head and the skip rule are to be carried onto campaign 1's runtime before they are used on the phone.

R54: withdrawn before any run, see prereg/R54-amendment-1.md

### R55 OLMoE on campaign 1's runtime, residency measured (kit r19)

- **Date:** 2026-11-14, 12:39 to 13:22 local. **Label:** *device* unless marked.
- **Pre-registration:** [R55](../prereg/R55-olmoe-c1-runtime-residency-device.md), commit `ec9927b`, pushed before
  the APK was handed over.
- **Data:** [device/r19_2026-11-14_summary.json](device/r19_2026-11-14_summary.json), a summary transcribed by the
  lead from the report the owner pasted (per-run fields and the per-token times of the three profiled runs).
  **The raw report is not stored as a file**; the repository parser was therefore not run on it, and the
  statistics below come from the transcribed fields.
- **Checks:** model sha256 as pre-registered; the same reply text in all 19 runs; `repack_loaded` true; 156 prompt
  tokens; 3 threads. Not checked beforehand: the R55 text names the APK hash, not the library hashes (the report
  gives CPU module `252bfab2…`, `libllama.so` `7de4c7f1…`, and the three campaign-1 patch hashes).
- **The gate now works and it timed out every time.** Before the first run it waited the full 10 minutes:
  headroom 0.87, lowest 0.774, back to 0.866; the 0.65 target was never reached. Before every other run it
  waited 90 s and ended at 0.81 to 0.86. The clock cap was 1,555 MHz on all big cores for the whole session
  (one reading of 2,150 before one run). Battery 33.3 to 34.1 °C.

| row | printed tok/s, five runs | median | ms per token (median) | major faults per run | mincore before run (MB) |
|---|---|---|---|---|---|
| `c1_off` | 3.68, 4.43, 4.54, 4.37, 5.02 | **4.43** | 218 | 16,029 to 59,659 | 609 to 1,225 |
| `c1_ship` | 6.88, 8.03, 6.95, 8.60, 7.20 | **7.20** | 134 | 2,186 to 16,004 | 375 to 1,304 |
| `c1_ship_io` | 7.36, 7.51, 7.64, 7.85, 7.02 | **7.51** | 129 | 990 to 20,836 | 631 to 1,391 |

Profiled runs (one each): `c1_off` 5.59, `c1_ship` 7.44, `c1_ship_io` 10.64 tok/s. `cold`: 3.07.

- **Residency (the reason for the round):** no run was resident. Before a run 0.4 to 1.4 GB of the file was in
  the page cache; right after a run 1.7 to 2.2 GB (maximum 2,180 MB), against the 2,725 MB the benchmark touches
  (*host*, memo-13) and the 2.6 GB line of the pre-registration. Two facts follow: the phone could not hold the
  working set on this day, and it dropped 0.5 to 1.4 GB of the model's pages again during each 90-second wait.
  `Cached` was 1.9 to 2.5 GB before runs; free swap was 0 to 142 MB.
- **State and scoring.** All 18 scored runs are "not resident": the expected-state prediction is a **hit**
  (≥ 12 of 18). The clock, 1,555 MHz, lies between the pre-registered "full" (≥ 2,000) and "throttled" (1,075 to
  1,450) columns, where §3 calls for memo-15's per-run formula with ± 25 %; that scoring needs the raw report
  and Bonnie's formula and is **not done here**. Against the table as printed: each median lies inside both
  "not resident" cells of its row (`c1_off` 4.43 in 2.8 to 5.8 and 2.4 to 5.2; `c1_ship` 7.20 in 5.1 to 12.1 and
  4.2 to 10.5; `c1_ship_io` 7.51 in 5.1 to 13.5 and 4.1 to 11.5) and inside the "expected medians" (3.5 to 4.5,
  7 to 10, 7.5 to 10.5). The bands are wide; this is agreement, not a sharp test.
- **Ratio prediction (steady ms of `c1_ship` / `c1_off` = 0.45 to 0.55 if not resident): miss.** Measured 0.616:
  campaign 1's levers and read-ahead helped less than predicted.
- **What campaign 1's runtime does on OLMoE:** 4.43 to 7.20 tok/s (× 1.63) with its shipping settings, 7.51
  with the routed read-ahead plus next-layer guess (× 1.70). Major faults per run fall from 16k to 60k to
  1k to 21k.
- **Nearest thing to a compute-bound token:** in the `c1_ship_io` profiled run 26 of 32 tokens had at most 10
  major faults; their median is 73 ms, the fastest token 55 ms, at a 1,555 MHz cap (barriers 305 per token,
  against 564 with the knobs off). Per token the expert op took about 38 to 43 ms, the dense matmuls about 11 to
  12 ms, the head about 4 ms in the cleanest tokens. That is consistent with memo-08's estimate of about 47 to 48
  ms for this stack at full clock (*estimated*), and it is far from 38 ms.
- **Kernel-level estimates (§4):** not applicable; no `c1_ship` run was resident at full clock.
- **Not a verdict on the goal by the rule, but the numbers are plain:** on this phone on this day the best
  configuration printed a median of 7.5 tok/s; the model was never resident and the clock never left its
  thermal cap, and neither is something the runtime controls under the campaign's rules.

### R56 OLMoE under campaign 1's kit protocol (kit r20)

- **Date:** 2026-11-14, 16:28 to 16:57 local. **Label:** *device* unless marked.
- **Pre-registration:** [R56](../prereg/R56-olmoe-r16-protocol-device.md), commit `a524ce9`, pushed before the APK
  was handed over.
- **Data:** [device/r20_2026-11-14_summary.json](device/r20_2026-11-14_summary.json), transcribed by the lead from
  the pasted report (per-run fields; per-token arrays of two profiled runs). The raw report is not stored as a
  file and the repository parser was not run on it.
- **Checks:** model sha256 as pre-registered; one reply text in all 46 official-prompt runs; 156 prompt tokens;
  all four runtime patch hashes and the library hashes are in the report.
- **State: this time the phone was cool, as in campaign 1's fast runs.** Battery 23 to 26 °C; from run 2 the
  gate reached 0.62 to 0.65 (as in r16) and the clock was 2,112 / 2,035 MHz for all of part A and B. Free swap 0
  to 144 MB, `Cached` 2.1 to 2.8 GB before runs.

**Part A, r16's protocol (21 runs).** The ten claim positions: 9.89, 11.94, 7.52, 13.33, 10.79, 15.02, 8.71,
11.72, 13.24, 9.72: **median 11.25 tok/s** (campaign 1, Granite, same positions: 20.51). All 19 official runs:
median 12.08. Prefill 5.6 to 7.9 s (r16: about 3.6 s). Model pages in the page cache before a run: 0.2 to 2.2 GB;
after a run 2.0 to 2.39 GB, never the 2,725 MB the benchmark touches (*host*).

**Part B, keep-warm on (r16's group 2 repeated).** 9.41, 11.13, 14.62, 11.61 (mean 11.69) against 7.52, 12.58,
13.46, 13.33 (mean 11.72) without it. Pages before the run 445, 1,703, 1,935, 1,540 MB against 1,127, 1,136,
1,352, 1,937 MB. The helper ran (18 passes, 0.26 to 0.65 s CPU each), but the system still took pages back
between passes (1,915 MB to 1,540 MB in one wait).

**Part C, warm block, no gate wait.**

| configuration | runs | tok/s | median | state |
|---|---|---|---|---|
| `c1_ship` | 10 | 11.70, 16.11, 16.71, 11.01, 12.03, 12.98, 9.95, 11.08, 12.74, 11.87 | **11.95** | first four at 2,112 MHz (median 13.9), then the cap fell to 1,785, 1,670, 1,555 as headroom rose to 0.97 |
| `EXPERT_IO=5` | 5 | 13.78, 10.44, 11.34, 12.98, 12.74 | 12.74 | cap 1,555 MHz |
| `c1_off` | 5 | 4.96, 7.01, 7.51, 6.47, 6.10 | 6.47 | cap 1,555 MHz |

In the warm block 2.06 to 2.36 GB of the model was in the page cache before each run and each `c1_ship` run still
had 1,001 to 2,579 major faults.

**Scoring.**

1. State (hot for all of part A, probability 0.6): **miss**; the phone cooled from run 2.
2. Part A hot: not applicable (no hot runs).
3. Part A cool (claim-position median 16 to 21, central 19): **miss**, 11.25. The condition "at least 1.9 GB
   resident before the run" held for 2 of the 19 runs only.
4. Part B was written for the hot state; in the cool state it showed **no gain** (11.69 against 11.72). By §5
   keep-warm is closed as a lever.
5. Part C `c1_ship`: the hot band (11 to 15, central 13) contains the median 11.95, and the residency and
   fault conditions held for all ten runs: **hit** as far as the hot cell goes. The four runs at full clock
   (median 13.9) are below the cool band (17 to 21): **miss** for the cool cell.
6. Ratios: `EXPERT_IO=5` within ± 10 % of `c1_ship` (+6.6 %): **hit**. `c1_off` at 55 to 80 % of `c1_ship`: 54 %,
   **miss** by one point.

- **The comparison with Vollie 1 at equal state (§3):** cool, 2,112 MHz, campaign 1's protocol and runtime:
  OLMoE 11.25 tok/s at the claim positions, Granite 20.51. **OLMoE does not match Vollie 1.** It is also not
  "slower although resident": it was never resident.
- **Where the difference is (per-token profile, runs 13 and 14):** the tokens with at most 5 major faults have
  a median of 43.4 ms (run 14) and 50.8 ms (run 13), fastest 42.3 ms, at 2,112 MHz; campaign 1's Granite token
  was about 46 ms. So OLMoE's fault-free token is as fast as Granite's or faster (expert op about 28 to 30 ms,
  dense matmuls about 9.6 ms, head about 3.5 ms). But the first ten tokens of every run fault: their mean is 84
  ms (run 14, started 0.3 s after the previous run with 2.24 GB resident) and 117 ms (run 13). With only 32
  tokens per run that sets the average.
- **Reading:** in the same cool state, with the same runtime and protocol, the gap to Vollie 1 is residency
  after the prompt, not arithmetic. The prompt's prefill touches 2.6 GB of experts; the phone gave the model at
  most 2.39 GB; the tokens generated next find part of their experts evicted. Granite ran fault-free with
  about 2.0 GB of file pages resident in r16. The shortfall for OLMoE is about 0.35 to 0.5 GB.

### R57 Same-day control: campaign 1's own kit on Granite (2026-11-14)

- **Date:** 2026-11-14, 18:47 to 19:02 local, the same day as R56 (16:28). **Label:** *device*.
- **What it is:** the owner ran campaign 1's unchanged app and kit (app 2.3.0+45, kit r16, Granite d6-e6, the
  configuration of R49). No pre-registration file: it is a control the lead asked for, and its reading was
  written beforehand in [memo-16](memo-16-why-far-from-campaign-1.md) §4: "If Granite also stays near 12 tok/s, the
  day's state explains the gap; if it reaches 20, reason 5 carries more weight."
- **Data:** [device/r16-control_2026-11-14_summary.json](device/r16-control_2026-11-14_summary.json), transcribed by
  the lead from the pasted report; the raw report is not stored as a file.
- **State:** cool, as on 2026-11-08: headroom 0.62 to 0.65 at the gate, 2,112 / 2,035 MHz. But less memory:
  `Cached` 1.81 to 2.26 GB before runs (2026-11-08: 2.24 to 2.98 GB), the app's resident file pages at most
  1.63 GB (2026-11-08: 2.02 GB), free swap 0 for most runs.

| | Granite, campaign 1's kit, 2026-11-08 (R49) | Granite, the same kit, 2026-11-14 | OLMoE, kit r20 part A, 2026-11-14 (R56) |
|---|---|---|---|
| median of the ten claim positions | 20.51 | **11.46** | 11.25 |
| the ten values | 14.09 to 20.95 | 7.80 to 15.04 | 7.52 to 15.02 |
| prefill of the prompt | about 3.6 s | 5.2 to 7.2 s | 5.6 to 7.9 s |
| major faults per official run | 4 to 73,524; fast runs 4 to 40 | 1,947 to 12,199 | 1,001 to 33,288 |
| clock | 2,112 MHz | 2,112 MHz | 2,112 MHz |
| `Cached` before runs | 2.24 to 2.98 GB | 1.81 to 2.26 GB | 2.1 to 2.8 GB |

- **Reading, as written beforehand:** Granite stayed near 12 tok/s, so the day's state explains the gap. On this
  day **campaign 1's own configuration does not reproduce 20.51; it prints 11.46**, and OLMoE under the same
  protocol prints 11.25. The two models are equal within the spread of the runs.
- **Dated correction to R56 (2026-11-14):** R56 scored "OLMoE does not match Vollie 1" against the 2026-11-08
  report, as its rule said. With this control the comparison at equal state is 11.25 against 11.46. The scoring
  of R56 stands as written; its reading "the shortfall is specific to OLMoE" does not. Granite is not resident
  on this phone today either.
- **Not equal in every respect:** the two runs were 2.3 hours apart and the Granite run had less page cache
  (1.8 to 2.3 GB) than the OLMoE run (2.1 to 2.8 GB); Granite's app also holds about 0.7 GB of anonymous memory
  for its repacked copy against about 0.23 GB for OLMoE. The data do not rank the two models more finely than
  "about the same".
- **Fault-free tokens:** Granite's late tokens in the profiled runs take about 46 to 53 ms (dense matmuls about
  22 ms, experts about 14 to 19 ms; 1,775 nodes, 666 barriers); OLMoE's about 43 to 51 ms (dense about 9.6 ms,
  experts about 28 to 30 ms; 533 nodes, 305 barriers). The arithmetic of both is where campaign 1 left it.
- **What it means for the campaign:** the number campaign 1 claimed depends on how much page cache the phone
  gives the app, and that changed between 2026-11-08 and 2026-11-14 by about 0.5 to 0.7 GB for reasons the
  reports do not show. Residency under about 2 GB of page cache is the problem to solve for both models, and it
  comes before any kernel work.

**Dated correction, 2026-11-14 (memo-18):** the working-set figures quoted in R52, R55, R56 and R57 from memo-13
(OLMoE 2,725 MB, Granite 2,569 MB) came from a tracer with a bug in how it read the expert ids of batched
tokens. The corrected page trace gives **Granite 1,966 to 1,986 MB** (prefill 1,908, decode 1,111) and **OLMoE
2,673 MB** (prefill 2,524, decode 2,155), *host*. Granite's figure now agrees with the 1,990 to 2,015 MB of file
pages at which R49's fault-free runs sat (*device*). Consequence for R57's reading: OLMoE and Granite printed
the same on 2026-11-14 because Granite's run had less page cache; at equal page cache Granite needs about
0.7 GB less in total and about 1.0 GB less for the decode tokens alone, so it becomes fault-free at a cache
size (about 2.0 GB) at which OLMoE cannot (its decode alone needs 2.15 GB).

### R58 `GGML_PREFILL_COLD` on Granite and OLMoE in one session (kit r21)

- **Date:** 2026-11-14, 21:40 to 22:10 local. **Label:** *device*. Pre-registration:
  [prereg/R58-prefill-cold-two-models-device.md](../prereg/R58-prefill-cold-two-models-device.md) (commit `f942602`).
- **Source:** [device/r21_2026-11-14_summary.json](device/r21_2026-11-14_summary.json), a summary the lead
  transcribed from the report the owner pasted (32 scored runs and 8 profiled runs); the raw file is not in the
  repository.
- **Build and files:** kit r21, app 1.21.0+21, campaign 1's runtime (`patches/llama.cpp/0001` to `0003`) plus
  `patches/llama.cpp-c1-olmoe/0001`, `0002`. `granite-Mi.gguf` and `olmoe-Mi.gguf`, hashes as in the cards. **No
  numerics change on either model** (no skip, no two-stage head); one reply text per model across both arms.
- **Phone state:** cool, 2,112 / 2,035 MHz at the end of every scored run; three gate timeouts (headroom 0.65 to
  0.71). `Cached` before the runs 2,493 to 2,886 MB (Granite) and 2,425 to 3,200 MB (OLMoE): the pre-registered
  **"high" state throughout; the "low" state (1.8 to 2.3 GB) the lever was designed for did not occur.**
- **Void checks:** `MADV_COLD` accepted (errno 0); texts equal; hashes equal; 8 scored runs per arm and model.
  Not void.

| model | arm | median tok/s (8 runs) | range | TTFT median |
|---|---|---|---|---|
| Granite-Mi | knob off | **17.19** | 14.4 to 17.9 | 5,294 ms |
| Granite-Mi | `GGML_PREFILL_COLD=8` | 17.27 | 15.7 to 17.6 | 5,196 ms |
| OLMoE-Mi | knob off | **19.70** | 11.1 to 22.3 | 4,844 ms |
| OLMoE-Mi | `GGML_PREFILL_COLD=8` | 20.06 | 11.5 to 21.4 | 4,661 ms |

- **Effect (mean of the within-block on minus off, four ABBA quads per model, with SE):** Granite
  **+0.09 ± 0.50 tok/s (+0.5 %)**; OLMoE **+0.93 ± 2.74 tok/s (+5.1 %)**. Both are below 2 × SE: **unresolved**.
- **Scoring, as written:**
  - Knob-off level, high state: Granite predicted 17 to 21, printed 17.19: **hit**. OLMoE predicted 11 to 15,
    printed 19.70: **miss** (the model was too slow by a third; memo-18's OLMoE floor and re-read terms).
  - Effect bands: Granite +0 to +10 % in the high state: +0.5 %, inside the band but unresolved. OLMoE +5 to
    +25 %: +5.1 %, unresolved. Neither counts as a pass (the pass rule needs a resolved gain).
  - Prediction 1 (decode read bytes lower): **not scorable**; `/proc/self/io` is not readable in the app
    (`read_bytes=-1`).
  - Prediction 2 (TTFT shorter by 0.6 to 2.0 s): 0.10 s and 0.18 s: **miss**.
  - Prediction 3 (marked bytes): OLMoE 653.9 MB against 645: **hit**; Granite 677.0 MB against 740: about 9 %
    low (the prompt rendered to 177 tokens).
  - Prediction 4 (helps both, Granite more; at equal cache Granite faster with the knob off): **miss** on both
    parts. With no numerics change OLMoE is the faster model when resident (19.70 against 17.19).
  - **Kill rule: "no resolved tok/s gain on either model" is met. The lever is closed as written.** What the
    data do and do not show: in the high-cache state the lever does nothing measurable, as its own high-state
    band said; the low-cache state was not tested and the read-byte rule could not be tested at all. Reopening
    it would need a new pre-registration and a kit that can produce the low state.
- **What decides a run's speed (not pre-registered, a reading):** the pages resident before the run. OLMoE runs
  that started with at least 2,500 MB of the file resident: median **21.03** (n = 8); the others 16.27 (n = 8;
  783 MB resident gave 11.15, 1,359 MB gave 11.53). Granite: at least 1,950 MB resident 17.36 (n = 11), others
  16.94 (n = 5; 1,232 MB gave 14.42). The slow runs are those that follow a gate wait of 45 to 91 s, during
  which the phone took file pages away although `Cached` stayed above 2.4 GB. After a run OLMoE reached
  2,672 MB resident, which is its whole working set (memo-18: 2,673 MB).
- **Fault-free tokens (profiled runs):** OLMoE all 32 tokens 41.6 to 48.2 ms in the best run (expert op
  27.8 ms, dense 9.6 ms, head 3.5 ms; 22.16 tok/s printed); Granite-Mi 53 to 57 ms. Granite here runs without
  campaign 1's skip and two-stage head, so its 17.2 is not comparable with R49's 20.51.
- **What generalised and what did not:** on both models the knob did nothing resolvable in a high-cache state,
  and on both the resident pages before the run decide the speed. Not general: memo-18's ranking of the two
  models; OLMoE's exposure to a wait is much larger (range 11.1 to 22.3 against 14.4 to 17.9) because its
  working set is 0.7 GB bigger.
- **What it means for the goal (*estimated*, not a claim):** OLMoE resident and cool is at about 42 ms per
  token with no numerics spent. The skip rule `MASS=900` (R51: 13.4 % of expert bytes, +3.32 %) and the untied
  head (R53: +0.016 %) exist only in the stock-base series; carried onto campaign 1's runtime they would take
  roughly 3.7 ms and 2 ms off, giving about 36 to 38 ms, which is the neighbourhood of 25 tok/s. That is the
  next round to pre-register, on both models, together with keeping the pages through the gate wait.

### R59 Skip rule and untied head carried onto campaign 1's runtime; two-setting gate on both models (host)

- **Date:** 2026-11-14 to 2026-11-15. **Label:** *host* (Apple M4 Pro, `llama-perplexity -c 512 -t 8 -ngl 0`,
  full wikitext-2 test: OLMoE 560 chunks, Granite 565). Logs: `lab/logs/r59/` (not tracked).
- **No pre-registration of its own.** The budgets (OLMoE 12.3003, Granite 9.6879) and the two-setting gate
  (`-b 512` and `-ub 1`, owner 2026-11-14) were written before; the OLMoE setting is R51's and R53's. The
  Granite thresholds were screened here, so the Granite choice below is a selection made after seeing these
  numbers, under the rule "the largest screened threshold inside the budget at both settings".
- **What changed:** new patches [`0003-c2-expert-skip`](../../../patches/llama.cpp-c1-olmoe/0003-c2-expert-skip.patch)
  and [`0004-c2-head2-untied`](../../../patches/llama.cpp-c1-olmoe/0004-c2-head2-untied.patch) on 620d2225c +
  campaign 1's `0001` to `0003` + `0001-keep-warm`, `0002-prefill-cold`. The c2 rules write into campaign 1's
  own skip mask, so they act in the prefill, the one-token path, the fused op and the read-ahead. Off unless
  `VOLLIE_SKIP_*` / `VOLLIE_HEAD2` is set; campaign 1's `GGML_EXPERT_SKIP` and `GGML_HEAD2` are unchanged.
- **Identity, knobs unset:** per-chunk values equal to the tree without the two patches at `-b 512`, `-ub 32`
  and `-ub 1`, 8 chunks, both models. With `MASS=900` and with `VOLLIE_HEAD2=1024` the values equal the
  stock-base c2 builds (2 to 4 chunks). The skip acts on the phone's path: experts are Q3_K (no repack), and at
  `-ub 1` the bytes left unread equal the bytes skipped.

| model | setting | `-b 512` | `-ub 1` | budget | expert bytes skipped |
|---|---|---|---|---|---|
| OLMoE-Mi | stock | 11.7146 (R50) | 11.7133 | 12.3003 | 0 |
| OLMoE-Mi | `MASS=900` | 12.1031 (+3.32 %) | 12.0925 (+3.24 %) | inside at both | 13.39 % |
| OLMoE-Mi | `MASS=900` + `VOLLIE_HEAD2=1024` | **12.1053 (+3.34 %)** | **12.0946 (+3.26 %)** | inside at both | 13.39 % |
| Granite-Mi | stock | 9.2266 (R50) | 9.2273 | 9.6879 | 0 |
| Granite-Mi | campaign 1's setting: skip 117 + `GGML_HEAD2=2048` | 9.4010 (+1.89 %) | not run | inside at `-b 512` | 33.33 % |
| Granite-Mi | skip 130 + head2 | **9.5173 (+3.15 %)** | **9.5076 (+3.04 %)** | inside at both | 42.39 % |
| Granite-Mi | skip 145 + head2 | 9.6850 (+4.97 %) | 9.6962 (+5.08 %) | **over at `-ub 1` by 0.0083** | 51.74 % |
| Granite-Mi | skip 160 + head2 | 9.9617 (+7.97 %) | not run | over | 59.50 % |

  Percentages are against the stock value at the same setting. The `-b 512` stock references are R50's runs on
  the unpatched base; the first 8 chunks on campaign 1's runtime equal them.
- **Reading:** R51 (12.1031) and R53 (12.1053) reproduce to the digit on campaign 1's runtime. The two-setting
  gate matters: Granite 145 would have passed on `-b 512` alone (by 0.0029) and fails at `-ub 1`. Settings
  for the device round: **OLMoE `MASS=900` + head K=1024; Granite skip 130 + head2 2048.** The skip buys three
  times more bytes on Granite (42 %) than on OLMoE (13 %) inside the same +5 %: OLMoE's routing weights are
  flatter. Thresholds 131 to 144 were not screened.
- **Not run:** Granite 117 at `-ub 1`; skip without the head (full set); the Android build of the two patches.

### R60 Skip + two-stage head, and page hold through the gate wait, on Granite and OLMoE (kit r22)

- **Date:** 2026-11-15, 09:48 to 10:58 local. **Label:** *device*. Pre-registration:
  [prereg/R60-numerics-and-hold-two-models-device.md](../prereg/R60-numerics-and-hold-two-models-device.md) (commit `da1db1e`).
- **Source:** [device/r22_2026-11-15_summary.json](device/r22_2026-11-15_summary.json), a summary the lead
  transcribed from the report the owner pasted (32 scored runs, 8 profiled runs with per-token times); the raw
  file is not in the repository. The statistics below are the lead's arithmetic on that summary (same
  definitions as `tools/c2_research_parse.py`'s r22 tables); the repository parser was not run on the raw file.
- **Build and files:** kit r22, app 1.22.0+22; CPU module `5d7c550c…`, `libllama.so` `815f3453…`, model hashes
  as pre-registered. One reply text per arm and model; off and num texts differ on both models.
- **Phone state:** clock 2,112 / 2,035 MHz in the profiled runs; **the gate timed out after 90 s before every
  scored run** (headroom 0.70 to 0.78; R58 mostly reached 0.65). `Cached` before the runs: OLMoE 2,576 to
  3,144 MB; **Granite 1,824 to 2,848 MB, five of 16 runs below 2.4 GB**, and one Granite run (pos 39) with the
  cap at 1,555. By §3 the level bands are scored for OLMoE and **not scored for Granite**.
- **Void checks:** hashes equal; 8 scored runs per arm and model; texts equal within each arm; skip counters
  non-zero; fadvise fails 0. Granite head line `c1_m=2048 c1_calls=33 c1_copy=1` in every num run. **OLMoE: no
  `c2-head-run` line in any num run and the head op took 3.8 to 4.2 ms in both arms (medians of the profiled
  runs): `VOLLIE_HEAD2=1024` was set but the untied head did not run on the phone** (it ran on the host,
  memo-20: `c2_calls=33`). By the void rule ("its head line shows no active copy") **the num factor is void on
  OLMoE** (kit, not lever). What OLMoE's num arm measured is the skip rule alone.

| model | arm | median tok/s, 8 runs | with hold (4) | without hold (4) | range |
|---|---|---|---|---|---|
| Granite-Mi | off | 14.50 | 16.61 | 11.58 | 10.5 to 17.0 |
| Granite-Mi | num (skip 130 + head2) | 17.20 | **18.96** | 11.30 | 10.1 to 19.4 |
| OLMoE-Mi | off | 15.74 | 17.48 | 13.86 | 7.9 to 19.5 |
| OLMoE-Mi | num (skip only ran) | 17.93 | **19.25** | 13.28 | 9.6 to 22.7 |

- **num minus off (mean of the four ABBA quads ± SE):**
  - **Granite: +1.44 ± 1.28 tok/s (+10.4 ± 10.2 %): unresolved, "not shown on this model" as written.** Quads:
    hold +2.29 and +2.47 (+14.3 %, +14.9 %), no hold +3.34 and −2.33. The negative quad's num runs started with
    492 and 633 MB resident. Decode skip counter 39.62 % (band 38 to 44): **hit**. Per-cent band (+12 to +32):
    **miss**.
  - **OLMoE: void (head did not run).** For the record, skip alone: +1.71 ± 0.70 tok/s (+10.6 ± 4.0 %), 2.4 × SE;
    decode skip counter 13.04 % (band 11.5 to 14.5): **hit**. Not counted as a pass.
- **Level bands:** OLMoE off 15.74 against 16.0 to 21.5: **miss**; OLMoE num 17.93 against 19.0 to 25.0:
  **miss** (and void). Granite not scored (state); for the record 14.50 and 17.20 would both be below their
  bands.
- **Fault-free tokens, num arm, profiled runs:** Granite 48.2 ms (20 tokens) and 46.4 ms (12 tokens), pooled
  47.3 ms, against 44 (41 to 48): **inside the band at its upper edge** (off: 52.4 and 54.3 ms). OLMoE (skip
  only) 40.4 ms (5 tokens) and one token at 47.5; fastest tokens 38.7 to 39.4 ms, against 36.5 (35 to 39):
  **miss**; the 2 ms assigned to the head are missing.
- **Hold (all 32 scored runs follow a 90 s wait; both arms together; Welch SE, 8 against 8 per model):**
  1. Resident MB before the run: **OLMoE +842 ± 173** (medians 2,436 against 1,704; predicted at least 300):
     **hit**. **Granite +1,165 ± 94** (2,018 against 974; at least 150): **hit**. Lost in the wait, medians:
     OLMoE 120 MB with hold, 714 without; Granite 49 against 1,025.
  2. tok/s: **OLMoE +4.72 ± 1.70** (band +1 to +6): **hit**, resolved. **Granite +5.28 ± 0.98** (band 0 to +2):
     resolved, **miss on the high side**. TTFT: OLMoE −1.4 ± 0.3 s, Granite −3.0 ± 0.5 s.
  3. Cost: CPU per wait median **5.2 s** (3.9 to 7.2) against "under 1 s": **miss**. Time after the gate: 10 of
     16 waits 5.05 to 5.34 s against "at most 5 s": **miss** (the 5 s limit plus one poll). The read-back was
     repeated: 3.4 GB advised per wait (median, summed over about 20 passes) to keep a few hundred MB.
  - **Pass on both models** ((1) holds, tok/s positive and above 2 × SE). **Kill rule not met:** OLMoE +842 MB;
    "half of the advised MB came back" cannot be read literally because `advised_mb` counts the same pages on
    every pass; the hold runs started with 86 to 98 % of the wanted MB resident (OLMoE).
- **Generalisation, as predicted and as found:** "num helps both, Granite more in per cent": **not shown**
  (Granite unresolved, OLMoE void). "hold helps both, OLMoE more": **helps both: yes; OLMoE more: no**
  (Granite gained as much or more; without hold Granite lost about 1 GB per wait in this session).
- **Readings (not pre-registered):**
  - Without hold this session was far slower than R58 (OLMoE 13.86 against 19.70, Granite 11.58 against
    17.19): every wait ran the full 90 s and took 0.7 to 1.0 GB of the file away. Hold restores R58's level, not
    more: with hold, off prints 17.48 (OLMoE) and 16.61 (Granite).
  - With hold, Granite's num effect is +2.38 tok/s in both quads (two quads, so no SE worth the name); the
    token time falls from 52 to 54 ms to 46 to 48 ms (expert op about 21 to about 13 ms, head 5.5 to 3.9 ms).
  - Even hold runs still take 2,000 to 140,000 major faults per run: pages are back in the cache but have to
    be mapped again; no run of this session was fault-free.
- **Neighbourhood (§5, not a claim):** best state of this round, num with hold: Granite 18.96, OLMoE 19.25
  (skip only). Neither is near 25. §7's condition "num resolved on both, OLMoE num median at 22 or more" is
  not met; no claim round is pre-registered.
- **Next:** (1) find why the untied head does not run in the phone's `armv8.2_2` CPU module (candidate: the
  Q6_K head's layout there is none of the three the patch accepts) and fix it in the runtime; (2) make the hold
  cheap (one read-back near the end of the wait instead of twenty) and keep it on in both arms; (3) rerun num on
  both models with hold on throughout, pre-registered as a new round.

- **Note, 2026-11-15 (later the same day):** the owner supplied the raw report; it is stored as
  [device/r22_2026-11-15.json](device/r22_2026-11-15.json) (sha256 `883e7288…409b4524`). The transcribed
  summary equals it in every field of the 32 scored and 8 profiled runs, and `tools/c2_research_parse.py`
  reproduces the medians, the quad effects and all void checks above. The parser's own hold table uses the off
  arm alone: Granite +4.34 ± 0.91 tok/s (resolved), **OLMoE +3.99 ± 2.27 (1.8 × SE, not resolved)**. The
  pre-registration scores hold on both arms together (+4.72 ± 1.70), so OLMoE's pass stands as written, but it
  leans on the num-arm runs.

- **Note, 2026-11-15 (owner):** the Toss app was running in the background during this session. That is a
  candidate cause for the gate never reaching 0.65 and for the larger page loss per wait than in R58; it is not
  shown by this data. No rerun of kit r22: OLMoE's head would not run in it either.

### R61 Stock and num perplexity on a host build with KleidiAI, the phone's build option (host)

- **Date:** 2026-11-15, 18:59 to about 23:30 local. **Label:** *host* (Apple M4 Pro, `llama-perplexity -c 512
  -t 8 -ngl 0`, full wikitext-2 test: OLMoE 560 chunks, Granite 565). Pre-statement:
  [prereg/R61-kleidiai-host-ppl.md](../prereg/R61-kleidiai-host-ppl.md) (commit `017c452`). Owner's option (a).
- **Build:** `lab/llama.cpp-c1h6/build-kai`: `620d2225c` + `patches/llama.cpp/0001` to `0003` +
  `patches/llama.cpp-c1-olmoe/0001` to `0006`, KleidiAI on, `GGML_KLEIDIAI_SME=0`. Logs and the exact command
  of every job: `lab/logs/r61/` (`queue.txt`; not tracked). All rows ran with the kit's off env
  (`GGML_TINY_OP=262144 GGML_EXPERT_IO=1 VOLLIE_SAMPLER_BYPASS=1 GGML_C2_IO=1`) as the pre-statement says; R59
  ran without it (identical per chunk on 4 chunks at three batch settings, not proven on the full set).
- **Limit:** on the M4 KleidiAI uses its i8mm kernels, the phone its dotprod kernels.

| row | setting | `-b 512` | `-ub 1` | without KleidiAI (R50 / R59) |
|---|---|---|---|---|
| 1 | OLMoE-Mi, off env (head in KleidiAI, int8) | 11.7165 | 11.7198 | 11.7146 / 11.7133 |
| 2 | OLMoE-Mi, off + `VOLLIE_HEAD2_REPACK=1` | 11.7132 | 11.7159 | 11.7146 / 11.7133 |
| 3 | OLMoE-Mi, row 2 + `MASS=900` + head 1024 | 12.0927 (+3.24 %) | 12.0976 (+3.26 %) | 12.1053 (+3.34 %) / 12.0946 (+3.26 %) |
| 4 | Granite-Mi, off env | 9.2795 | 9.2732 | 9.2266 / 9.2273 |
| 5 | Granite-Mi, row 4 + skip 130 + head2 2048 | 9.5056 (+2.44 %) | 9.5120 (+2.58 %) | 9.5173 (+3.15 %) / 9.5076 (+3.04 %) |

  Per cent in rows 3 and 5 is against the same build's stock at the same setting (rows 2 and 4).

- **Two references side by side (the existing budgets are unchanged):**

| model | stock, no KleidiAI | budget (stands) | stock with KleidiAI (`-b 512` / `-ub 1`) | × 1.05 |
|---|---|---|---|---|
| OLMoE-Mi | 11.7146 | **12.3003** | 11.7132 / 11.7159 | 12.2989 / 12.3017 |
| Granite-Mi | 9.2266 | **9.6879** | 9.2795 / 9.2732 | 9.7435 / 9.7369 |

- **Reading as fixed beforehand:** both num settings are **inside on the phone's build** by both readings, at
  both batch settings: OLMoE 12.0927 / 12.0976 (below 12.2989 / 12.3017 and below 12.3003); Granite 9.5056 /
  9.5120 (below 9.7435 / 9.7369 and below 9.6879). Row 1: 11.7165 / 11.7198, far below 11.7146 × 1.05; R60's
  OLMoE off arm with the head in KleidiAI was +0.02 to +0.05 % from the file's own head.
- **Predictions:**
  - Stock with KleidiAI within ±0.5 %: OLMoE −0.01 / +0.02 %: **hit**. Granite **+0.57 %** at `-b 512`:
    **miss**; +0.50 % at `-ub 1`: at the edge, inside.
  - Row 1 within ±0.7 %: +0.02 / +0.06 %: **hit**.
  - Both num settings inside: **hit**.
  - Per-cent increase within ±0.3 points of R59: OLMoE 0.10 and 0.00 points: **hit**. Granite 0.71 and 0.46
    points lower: **miss** (on the favourable side).
- **Log checks:** OLMoE's head is in `CPU_KLEIDIAI` in row 1 (242.83 MiB, no `CPU_REPACK`) and in `CPU_REPACK`
  in rows 2 and 3 (144.00 + 80.60 MiB). Row 3: head ran, `c2_declined=0`, `mismatches=0`, skip 13.386 % of
  expert bytes at both settings. Row 5 at `-b 512`: `c1_calls=566 c1_copy=1`, skip 42.39 %. **Row 5 at `-ub 1`
  has no head line in the log** (campaign 1's head prints none without a batch graph; R59's log has the same
  gap), so that the head ran there is taken from the value differing from skip alone, not from a counter.
- **What it means:** KleidiAI moves Granite's stock numerics by about half a per cent (it handles 541 MiB of
  Granite's weights against 243 MiB of OLMoE's) and OLMoE's hardly at all. The phone's Granite therefore serves
  about 9.28, not 9.23; the skip + head setting ends at nearly the same absolute value on either build, so the
  absolute budget 9.6879 holds with the same margin (0.176 at `-ub 1`). `VOLLIE_HEAD2_REPACK=1` is safe for
  OLMoE's off arm (−0.003 to −0.004). Not measured: the phone's dotprod KleidiAI kernels; campaign 1's Granite
  file (d6-e6) on a KleidiAI build.

### R62 Skip + head with hold in every run, and threads 3 against 4, on Granite and OLMoE (kit r23)

- **Date:** 2026-11-16, 10:31 to 11:43 local. **Label:** *device*. Pre-registration:
  [prereg/R62-numerics-threads-hold-two-models-device.md](../prereg/R62-numerics-threads-hold-two-models-device.md)
  (commit `586491c`). **Source:** the raw report [device/r23_2026-11-16.json](device/r23_2026-11-16.json)
  (sha256 `21bde80c…20aeb7bf`), read with `tools/c2_research_parse.py`; a few figures below (per-cent effects,
  resident / wanted, profiled-token medians) are the lead's arithmetic on the same file.
- **Void checks:** hashes equal the build; 8 scored runs per arm and model; one text per arm and model, the
  same at 3 and 4 threads; OLMoE loaded with `VOLLIE_HEAD2_REPACK=1` at both loads; threads applied equal the
  spec; skip counters non-zero; no head declines. **OLMoE's head ran on the phone:** `c2_calls=33
  c2_declined=0 c2_copy=1` in every num run (patch 0006 works). Not void.
- **Phone state:** **all 32 gates timed out again** (headroom at the end of the wait 0.67 to 0.97 Granite, 0.70
  to 0.82 OLMoE). `Cached` at least 2.4 GB before 12 of 16 Granite runs and 16 of 16 OLMoE runs; Granite's
  last block ran with the cap at 1,555 MHz. By §3 **Granite's level bands are not scored**, OLMoE's are. Whether
  background apps were closed: not yet stated by the owner.

| model | arm | 3 threads, median of 4 (runs) | 4 threads, median of 4 (runs) |
|---|---|---|---|
| Granite-Mi | off | 15.95 (15.95, 16.01, 15.94, 12.52) | 9.90 (7.76, 14.24, 12.03, 7.58) |
| Granite-Mi | num | **17.65** (17.79, 17.50, 17.48, 18.82) | 10.99 (10.42, 17.69, 11.55, 9.74) |
| OLMoE-Mi | off | 16.85 (17.86, 9.99, 18.10, 15.84) | 12.45 (7.97, 13.51, 11.49, 13.41) |
| OLMoE-Mi | num | 16.40 (15.39, 17.40, 11.68, 17.99) | 13.39 (6.26, 13.85, 12.93, 13.84) |

- **num minus off (mean of the four quads ± SE):**
  - **Granite: +2.37 ± 0.69 tok/s (+18.6 ± 5.3 %): pass** (3.4 × SE; quads +3.06, +1.67, +3.92, +0.84). Bands
    (+1.4 to +3.4; +8 to +21 %): **hit**. Skip counter 39.62 %: **hit**. Head op 3.7 ms in the clean profiled
    run (off 5.5): **hit**.
  - **OLMoE: +0.15 ± 1.00 tok/s (+1.6 ± 6.9 %): not shown on this model** (quads +2.48, −0.69, +0.94, −2.14).
    Skip counter 13.04 %: **hit**. Head line as predicted: **hit**. Head op time: 3.4 and 3.1 ms in the num
    profiled runs against 3.3 and 3.5 ms off; predicted 2.6 (2.1 to 3.1): **miss**. The head screens all
    50,304 rows and then recomputes about 7,000 rows exactly per token (`c2_exact_rows_per_col=7023.5`; the
    host shows 6,936 to 7,057, so this is the lever as built, not a phone fault): on the phone it saves nothing
    measurable. Against R60, the off arm's head op itself fell from 3.8 to 4.2 ms (KleidiAI's int8) to 3.3 to
    3.5 ms (Q6_K repack).
- **Levels, 3 threads:** OLMoE off 16.85 (16.0 to 21.0): **hit**; OLMoE num 16.40 (18.0 to 23.0): **miss**.
  Granite not scored (state); for the record 15.95 and 17.65 are inside their bands.
- **threads 4 minus 3 (8 against 8 runs, both arms, Welch):** **Granite −5.12 ± 1.37 tok/s, OLMoE −3.87 ±
  1.51 tok/s: negative and beyond 2 × SE on both. Kill rule met: plain four-thread decode is closed on both
  models.** Predicted −3 % and +4 %: both **miss** (about −30 % and −23 %). Caveat, a reading: the four-thread
  runs also started hotter and, on Granite, with less of the file resident (medians 1,669 and 1,703 MB against
  2,001 and 2,034), so part of the loss may be heat and pages rather than barriers. The thread samples show
  the workers inside cores 4 to 7 with about a fifth of OLMoE's samples on cores 0 to 3.
- **Hold (protocol check):** resident before the run at least 95 % of the wanted MB: **6 of 16 runs on each
  model** (predicted at least 12): **miss**; medians 92.6 % (Granite) and 94.2 % (OLMoE). Fail rule (fewer
  than 8 of 16 OLMoE runs at 90 %): 12 of 16, **not failed**. CPU per wait, median: 2.1 s (Granite) and 1.9 s
  (OLMoE) against 0.9 (0.6 to 1.3): **miss**. Longest gate-to-run time 6.9 s (limit 8.5): **hit**. The cheap
  hold is cheaper than R60's (5.2 s) and keeps less (R60's hold runs started with 86 to 98 % on OLMoE; here the
  waits still lost 0.4 to 1.0 GB at the median before the read-back).
- **Generalisation:** "num helps both by a similar per cent": **miss** (Granite +18.6 %, OLMoE not shown, for
  the second round). "A fourth thread helps neither": **hit** (it hurts both).
- **Readings (not pre-registered):**
  - Fault-free tokens, clean profiled runs: Granite 53.9 ms off, 45.2 ms num; OLMoE 43.5 ms off (2 tokens)
    and 51.5 ms num (1 token). No OLMoE run was near fault-free: its num runs took a median of 5,310 major
    faults in the decode against 1,139 off (3 threads), which is where its skip gain went. Why the skip arm
    faults more is not known.
  - Two sessions in a row never reached the gate in 90 s, on a phone that reached it in R58. Each run then
    waits the full 90 s and loses pages; the levels of R58 (19.70 and 17.19 without numerics) were not
    reproduced in R60 or R62.
- **Neighbourhood (§5, not a claim):** best cells of the round are Granite num 17.65 and OLMoE off 16.85 at
  three threads. **Nothing here is near 25 tok/s.**
- **What it means (§7):** Granite's setting (skip 130 + head2 2048) is confirmed twice with hold and is frozen.
  OLMoE: the untied head is built correctly and is worth about nothing on the phone; the skip rule's gain is
  not shown in two rounds. Four plain threads are closed; memo-21's barrier-free helper stays open as a separate
  mechanism. The limit on both models is the state before the run (heat gate never reached, pages lost in the
  wait, faults during the decode), not the kernels.

- **Note, 2026-11-16 (owner, after the scoring):** a charger may have been plugged in about 45 minutes into the
  session. The report agrees: `plugged` is true from pos 40 to pos 48 (Granite's last block: 7 of its 8 scored
  runs and both profiled runs), the battery rose from 28.8 to 31.0 °C and the cap fell to 1,555 MHz from pos 42
  on. Blocks 1 to 3 and pos 39 ran unplugged, and all their gates timed out too, so the charger does not
  explain the timeouts. The scoring above stands as written (no run is excluded); what leans on the charged
  block:
  - **Granite num:** unaffected in size. The two unplugged quads (block 1) give +3.06 and +1.67, mean +2.37,
    the same mean as all four; with two quads the SE is 0.70.
  - **Granite threads:** block 4's four-thread quad ran entirely charged and capped, so the kill on Granite
    leans on confounded runs. Block 1 alone (unplugged, 4 against 4 runs): -4.29 ± 2.23 tok/s, negative
    but only 1.9 × SE, and its four-thread quad came first after the warm-up with less of the file
    resident. **Read Granite's four-thread result as "lost, not cleanly measured"; OLMoE's (−3.87 ± 1.51, all
    unplugged) is clean.** The kill is not reopened by this note; reopening would need a new pre-registration.
  - **OLMoE:** nothing was charged.
  - Next kit: stop and wait (or mark the run void) when `plugged` turns true.

- **Correction to the R62 entry, 2026-11-16 (memo-23 §1):** the reading "OLMoE's num runs took more major
  faults, which is where its skip gain went" is **not shown as an arm effect**: the high-fault runs are 5 of 8
  num against 2 of 8 off (Fisher p = 0.32), cluster in time across both arms, and are absent in the profiled
  runs and in R60. The faults themselves appear to be mostly swap-ins of the app's own memory from zram, not
  file reads (inferred from the counters and the Linux 5.4 source; R63 records `pswpin` to test it). "Resident
  MB" measures the file only and does not see this.

### R64 KV and recurrent clear with MADV_DONTNEED, with R63's gate rule and num, on Granite and OLMoE (kit r25)

- **Date:** 2026-11-16, from 21:00 local. **Label:** *device*. Pre-registration:
  [prereg/R64-kv-clear-dontneed-two-models-device.md](../prereg/R64-kv-clear-dontneed-two-models-device.md)
  (commit `00cda7a`). **Source:** [device/r25_2026-11-16.json](device/r25_2026-11-16.json) (sha256
  `5b31f7a2…5f85`), read with `tools/c2_research_parse.py` (commit `fb781dd`); the Cached and clock checks are
  the lead's reading of the same file. R63 (kit r24) was not run; its rules are applied here as R64 §1 says.
- **Checks:** model hashes OK; one library hash per library, equal to the build; 0001-0008 and app 0001-0009 as
  built; one text per model and arm, **the same with kv on and off** (Granite `afb8490b` / `13584df0`, OLMoE
  `f8a9fdc4` / `40f6ce50`); no charger; 8 kv and 8 plain scored runs per model, all paired.
- **kv factor: VOID (kit).** Every kv run printed `c2-kvclear ... released=0 why=map` (32 `kv` and 16 `rs`
  lines): the runtime check of `/proc/self/maps` declined every buffer and fell back to the full memset, so the
  lever never ran. Cause, inferred and not yet verified: Android's heap pointers on arm64 carry a tag in the top
  byte (scudo with top-byte ignore), so the tagged buffer address matches no line of `/proc/self/maps`. Fix
  (memo to follow): untag the address before the maps check and before `madvise`. The kv-minus-plain figures the
  parser prints (Granite TTFT −294 ± 528 ms, OLMoE +51 ± 286; decode −4.2 ± 3.6 and −2.2 ± 3.3 ms) compare two
  identical code paths and are a measure of noise only.
- **Premise of the lever on this day:** pre-prompt major faults, median of the plain scored runs, were 6
  (Granite) and 26 (OLMoE); predicted for OLMoE off 20k to 50k: **miss**. Large pre-prompt fault counts still
  occurred in single runs (OLMoE up to 40,307, Granite up to 53,437), mostly after 90 s waits. `/proc/vmstat`
  was not readable in any of 48 rows, so the swap reading (R63 §3) is **not scorable**.
- **Phone state:** `Cached` at least 2.4 GB before all 16 scored runs of each model (2,637 to 3,106 MB); cap
  2,035 MHz in the profiled runs, below 2,112, so **levels are not scored**. Every r16 wait ran to 91 s (8 of 8
  per model): the gate factor is **exercised** on both models.
- **Gate rule, plateau minus r16 (kv pooled, both arms):**
  - Granite: tok/s −1.18 ± 1.44 (blocks) / ± 1.14 (runs); OLMoE +0.21 ± 2.32 / ± 1.28. Neither beyond 2 × SE:
    **"shorter session, speed not shown"** on both. Predicted +1.2 (−0.3 to +3.0) and +1.5 (−0.5 to +3.5):
    Granite **miss** (low), OLMoE **hit**.
  - Wait −43.6 and −42.8 s (−30 to −45): **hit** on both. File MB lost in the wait −232 and −153 (−100 to
    −450): **hit**. Headroom at start +0.01 (−0.02 to +0.03): **hit**. `VmSwap` at start, difference of the
    arm medians about −20 and −10 MB (0 to −150): **hit**. Clock cap no worse with plateau.
- **num minus off (mean of the four ABBA quads, kv pooled):**
  - **Granite +2.68 ± 0.71 tok/s: pass** (3.8 × SE); band +1.4 to +3.4: **hit**. Plain runs only +2.79 ± 1.86.
  - **OLMoE +2.02 ± 1.09 tok/s: not shown** (1.9 × SE); band −1.0 to +3.0: **hit**. Plain only +1.86 ± 2.48.
    This is the third round without a shown gain (R60 +1.71 ± 0.70 was skip alone; R62 +0.15 ± 1.00). By R63 §4
    `MASS=900` is not carried into further device rounds until the pre-run memory state is under control, or a
    host-side reason brings it back.
  - Under r16 waits num minus off was larger on both (Granite +3.86 ± 0.34, OLMoE +3.45 ± 0.72, 2 quads each)
    than under plateau waits (+1.49, +0.59): a reading, not pre-registered.
- **Generalisation:** "plateau helps both by a similar amount": **miss** (neither shown). "num resolved on
  Granite, not on OLMoE": **hit**. "kv effect on both, scaled by buffer size": **not testable** (void).
- **Scored runs, tok/s medians (kv pooled; not levels):** Granite off 16.17 / 16.04, num 19.72 / 16.91 (r16 /
  plateau); OLMoE off 18.47 / 20.91, num 21.88 / 21.06.

### R66 KV and recurrent clear with MADV_DONTNEED on the untagged address, on Granite and OLMoE (kit r26)

- **Date:** 2026-11-18, from 12:38 local. **Label:** *device*. Pre-registration:
  [prereg/R66-kv-clear-untagged-two-models-device.md](../prereg/R66-kv-clear-untagged-two-models-device.md)
  (commit `b06ed48`). **Source:** [device/r26_2026-11-18.json](device/r26_2026-11-18.json) (sha256
  `45769b42…7fcee561bd`), read with `tools/c2_research_parse.py`; `Cached` and cap are the lead's reading of the
  same file.
- **Checks:** model and library hashes equal the build (series 0001-0009, app 0001-0010); one arm per model
  (Granite num, OLMoE off), r16 gate in every gated run; texts equal kv against plain (Granite `13584df0`, OLMoE
  `f8a9fdc4`); no charger; released 0 in every run without kv. **Not void.**
- **Owner's note (2026-11-18):** background apps kept starting by themselves during the session, so memory
  pressure from other apps may be in these runs. The counters agree: every wait ran to 91 s; the file lost in
  the wait was 506 to 954 MB at the median (R64: 0 to 330); the hold got back 84.5 % at the median (95 % or more
  in 10 of 40 runs); the app's `VmSwap` before a run reached 800 MB. `Cached` at least 2.4 GB before 13 of 16
  Granite and 16 of 16 OLMoE runs; cap 2,035 MHz in the profiled runs. Levels were not pre-registered here.
- **Mechanism: pass on both models.** `why=ok` in every clear line of 8 of 8 kv runs per model; `tag=0xb4` in
  every kv line (32 Granite, 16 OLMoE): **the tagged-pointer reading of R64 is confirmed on the phone.**
  Released per kv run 151.8 MB (Granite, `kv` + `rs`) and 570.4 MB (OLMoE; two clears per run of 285.2 MB):
  bands (≥ 50, ≥ 250): **hit**.
- **Effect (kv minus plain, paired over 4 ABBA quads ± SE):**
  - **OLMoE: pass.** Pre-prompt major faults −15,860 ± 9,128 (lower; band 0 to −15,000: **miss**, larger);
    TTFT **−704 ± 149 ms** (4.7 × SE; band 0 to −250: **miss**, larger); decode −4.5 ± 6.7 ms/token;
    tok/s +1.44 ± 1.74 (band −1.0 to +1.0: **miss**, high, inside its SE); `VmSwap` at run end −18 ± 15 MB.
  - **Granite: mechanism works, effect not shown.** Pre-prompt faults −8,231 ± 3,867 (lower; band 0 to
    −10,000: **hit**); TTFT +6 ± 263 ms (band +50 to −120: **hit**, not beyond 2 × SE); decode −5.1 ± 4.0;
    tok/s +0.85 ± 0.92 (band: **hit**).
  - Texts identical: **no kill.**
- **Generalisation:** the mechanism and the fault reduction appear on both models; the TTFT gain is shown on
  OLMoE only (its KV buffer is about four times Granite's, as predicted, but the gain is about three times the
  predicted band; the heavy memory pressure of this session is the likely reason, a reading).
- **What it means (R66 §7):** the knob is kept on in every later kit on both models (OLMoE pass; Granite
  "mechanism works, effect not shown", which also keeps it on). Its value is TTFT and memory, not the 32-token
  metric. Next memory lever: memo-24 item 3.

- **Dated correction, 2026-11-18 (Annie, books audit against `r26_2026-11-18.json` and the parser):** every figure above matches the parser output. One labelling slip in `results.csv`: the two R66 rows carry `ppl` and `ppl_vs_stock_pct` (9.5120 / +2.58 for Granite, 11.7159 / 0.00 for OLMoE) under the label *device*. Those two columns are **host** values copied from R61 (KleidiAI build, `-ub 1`; R62's rows say so in their notes), not measured in R66; the *device* figures of R66 are the effect numbers in the notes column. The rows are not edited.

### R67 Runtime thread pin with a persistent pool, and the model mapped PROT_EXEC, 2 × 2 on Granite and OLMoE (kit r27)

- **Date:** 2026-11-18, from 17:56 local. **Label:** *device*. Pre-registration:
  [prereg/R67-pin-pool-and-exec-map-two-models-device.md](../prereg/R67-pin-pool-and-exec-map-two-models-device.md)
  (commit `847acfd`). **Source:** [device/r27_2026-11-18.json](device/r27_2026-11-18.json) (sha256
  `0ee086ce…ae82333`), read with `tools/c2_research_parse.py`; the wait-length split and `Cached` count below
  are the lead's arithmetic on the same file.
- **Checks:** model and library hashes equal the build (series 0001-0011, app 0001-0011, no OpenCL); every
  quad holds the four cells; kv on in every run; texts equal in all four cells (Granite `13584df0`, OLMoE
  `f8a9fdc4`); no charger. **Not void.** Bench condition: deep-sleeping list (owner).
- **Storage check (session start):** internal `/data/data/<pkg>` f2fs `rw,nosuid,nodev,noatime`, all four probe
  calls errno 0: **pass** (predicted p = 0.7). External `/storage/emulated/0/Android/data`: not mounted noexec,
  but `mprotect` and `mmap` with exec return errno 13 (EACCES): **refused**, as predicted (p = 0.8). Both models
  copied in (10.2 s and 52.6 s, sha256 OK); 6.2 GB of the 15.6 GB free used.
- **Phone state:** cap `2419/2841` in the profiled runs (R64 and R66 read 2,035 there; the cause of the change
  is not known: instrument changes of this kit, a cooler phone, or the deep-sleeping list); gates were reached
  in many runs (R66: none). `Cached` at least 2.4 GB before 15 of 16 Granite runs. Levels are not compared with
  R66 (instrument changed, §1).
- **Mechanisms.** pin: every P run's workers had mask `0xf0`, ran on the pool (528 of 528 graphs per model), 0
  runs with a worker on cpus 0-3; in p runs a worker was on cpus 0-3 in 6 of 8 (Granite) and 4 of 8 (OLMoE)
  runs: **hit** (≥ 2 of 8). exec: `r-xs`, errno 0, `where=api` in 8 of 8 X runs per model: **hit**; `mprotect`
  took 80-164 ms, before the gate.
- **pin (P − p), paired over 4 quads:**
  - **OLMoE: +1.94 ± 0.69 tok/s, pass** (2.8 × SE); decode −3.9 ± 1.5 ms/token; bands **hit**.
  - **Granite: −0.28 ± 0.78 tok/s, "pin restored, speed not shown"**; tok/s band **hit**, decode +4.2 ± 5.1
    (band +1 to −6): **miss**.
- **exec (X − x), paired over 4 quads:**
  - Pre-registered criterion (median file MB lost in the wait lower by at least 25 %): Granite 160 against 762
    (−79 %), OLMoE 6 against 308 (−98 %): **met on both** → by §4 **"residency shown, speed not shown"** on both
    (tok/s Granite +2.46 ± 2.20, OLMoE −0.71 ± 1.53, neither beyond 2 × SE). Bands for the loss (−30 %, 0 to
    −60 %): **miss** on both (larger). tok/s bands: Granite **miss** (high), OLMoE **hit**.
  - **Reading that weakens the pass (lead, not pre-registered):** X runs had more zero-length waits (gate
    already met at the start: Granite 3 of 8 against 1 of 8, OLMoE 4 of 8 against 0 of 8), and a zero wait
    loses nothing. Over waits of 60 s or more only, the median loss is 713 against 783 MB (Granite, n 5 / 7,
    −9 %) and 527 against 836 MB (OLMoE, n 2 / 5); the median loss per second of wait is 1.8 against 8.7 MB/s
    (Granite) and 0.7 against 3.5 MB/s (OLMoE). The criterion as written is met; how much of it is the mapping
    and how much the wait length is not resolved by this session. The next exec test must score the loss per
    second of wait, or over fixed-length waits.
  - TTFT: OLMoE −419 ± 298, Granite −124 ± 236 ms (descriptive).
- **Interaction (cell means, tok/s):** Granite px 19.53, Px 19.70, pX 22.44, PX 21.72; OLMoE px 22.17, Px 24.32,
  pX 21.67, PX 23.41. Descriptive: on Granite the exec column carries the gain, on OLMoE the pin column.
- **T1:** `c2-sched` and `smaps` readable in 16 of 16 runs per model; but smaps' model Rss reads 0 MB in every
  run while mincore reads 1.9-2.7 GB resident, so the mapped-against-resident split is **not usable** as built
  (a parser or kit field error; to fix before it is used).
- **Generalisation:** pin helped OLMoE and not Granite; exec's loss reduction appears on both, with the wait-
  length caveat. "pin helps both by a similar amount": **miss**. "exec helps OLMoE more than Granite": **miss**
  (the speed signal, unresolved, is on Granite).
- **What it means (§7):** pin stays on (pass on OLMoE; protocol correction on Granite). exec stays on
  ("residency shown"), with the caveat above; Y1 b (read-back through the mapping) is next for it, and its test
  scores loss per second of wait. Not a goal claim: OLMoE's P runs had a median of 25.03 tok/s and the PX cell
  25.23 (n 4), with the bench condition and a 2,419/2,841 cap; that is material for a claim round under the
  claim rule, not a claim.

### R68 Claim-format rehearsal: touch x nice, campaign-1 and stock references, power saving off (kit r28)

- **Status:** pre-registered; not yet run. **Label:** *device* (when run). Pre-registration:
  [prereg/R68-rehearsal-touch-nice-references-device.md](../prereg/R68-rehearsal-touch-nice-references-device.md).
  Kit r28 is three APKs (c2, c1, stock); their hashes are in the pre-registration. Sessions A, B and R and the T-A
  probes are described there; no result exists yet. Memo: [memo-32](memo-32-kit-r28.md).

### R68 (session B of A/B/R; partial)

- **Date:** session started 2026-11-18 21:02 local; scored 2026-11-19. **Label:** *device*. Pre-registration:
  [prereg/R68-rehearsal-touch-nice-references-device.md](../prereg/R68-rehearsal-touch-nice-references-device.md)
  with [amendment 1](../prereg/R68-amendment-1.md), [2](../prereg/R68-amendment-2.md) and
  [3](../prereg/R68-amendment-3.md). **Source:** [device/r28_2026-11-18.json](device/r28_2026-11-18.json) (sha256
  `c0b5af60c548aff2a5e222979b1e494653c78f7ae96031b47df59f179a433d63`), scored with
  [memo-33-work/score_r68.py](memo-33-work/score_r68.py); output in
  [memo-33-work/score-r68-sessionB.txt](memo-33-work/score-r68-sessionB.txt). The scorer needed a field-reading fix
  against the real report layout, no rule changed:
  [memo-33-work/scorer-fix-2026-11-19.md](memo-33-work/scorer-fix-2026-11-19.md). This is session B only (c2 APK,
  deep-sleeping list off). **Still to come:** the T-A cap probes (power saving on and off), session A (list on),
  session R (c1, then stock).
- **Validity (D1 to D4):** no void; power saving off in 72 of 72 runs; texts identical per model and arm.
- **Claim-format blocks (10 runs each; all uncapped by the strict reading unless noted):**
  - **b1 OLMoE off:** median 23.03 tok/s, in the band 23.0 to 25.4; 9 of 10 uncapped (run 1 capped, 2150 to 1785);
    all 10 runs below 25.0, prediction 4 to 7 below 25.0: **MISS**.
  - **b2 Granite num:** median 22.59 tok/s, in the band (22.5 to 24.5); 10 of 10 uncapped.
  - **b5 OLMoE num (MASS=900 + HEAD2=1024) + touch (+ nice, see below):** median **25.47 tok/s**, 10 of 10 uncapped,
    3 runs below 25.0 (23.22, 23.52, 24.20). b5 minus b1 = +2.44 tok/s (band +0.4 to +1.6): **MISS, high**.
    Position is confounded (b5 last, b1 first).
- **Go rule: MISSED** (needs median >= 25.7 and at most 2 runs below 25.0; here 25.47 and 3). Session A is still
  to come, so this is provisional for A8. Rehearsal, **not a claim**.
- **Touch (T minus t, paired over 4 quads):**
  - **OLMoE: PASS**, +1.43 ± 0.94 tok/s. Wanted MB missing at run start: median T 0 against t 1756 (-100 %, beyond
    the -90 % band edge: band **miss**, rule pass).
  - **Granite: PASS**, +0.76 ± 0.71 tok/s; missing T 0 against t 867 (same band miss).
  - A6 (end pass) holds in 18 of 18 (OLMoE) and 8 of 8 (Granite) gated T runs.
- **Nice:** the set is permitted (read-back -10; prediction **hit**), but the factor is **VOID on both models**
  under B3 as adopted. Every pool thread already read -10 in all 72 runs, including the first warm-up, and all
  `c2-nice` lines show zero sets, so N and n had no contrast (0 whole quads, fewer than 3).
  - **Lead's decision, 2026-11-19:** B3's adopted -10 test stands. The alternative "first value" reading is
    reported, not scored: N minus n would read -0.45 ± 0.58 (OLMoE) and -0.51 ± 0.13 (Granite) tok/s on a factor
    with no physical contrast.
  - **Consequence:** every run of this session ran with pool threads at -10. b5 measures num + touch at nice -10,
    and so do b1 and b2.
  - **Open question for Tom:** why restore did not return the threads to their old value. Is the pool persistent
    across runs, or was the baseline set before the first run?
- **First warm-ups** were capped at 1555/1555 with power saving off (likely heat or prior use; not established).

### R68 (cap probes)

- **Date:** 2026-11-19, 14:00 and 14:06 local. **Label:** *device*. **Sources:**
  [device/capprobe-r28_2026-11-19T1400.json](device/capprobe-r28_2026-11-19T1400.json) (recorded power saving off) and
  [device/capprobe-r28_2026-11-19T1406.json](device/capprobe-r28_2026-11-19T1406.json) (power saving on); scorer
  output [memo-33-work/score-r68-capprobes.txt](memo-33-work/score-r68-capprobes.txt).
- *Owner, 2026-11-19:* **both probes ran with the deep-sleeping list on** (the Toss app was in deep sleep). The 14:00
  probe's recorded label "deep-sleeping list off" is a mis-selection on the start screen; the raw file is left as it
  is. The list state is not a factor of the cap-probe prediction, so the scored outcome is unchanged.
- **Scored (E, P 0.65, "the cap follows power saving"): MISS.** Power saving on: max7 <= 2150 in 89 % of samples
  (hit). Power saving off: max7 >= 2841 in 3 % of samples (miss).
- **Lead's cross-tab of the cap against Android `thermal_status` (descriptive, unscored):**

  | power saving | thermal_status | cpu 7 cap (samples) |
  |---|---|---|
  | off | 0 | 2841: 10, 2150: 10 |
  | off | 1 | 1555: 223, 1670: 40, 1785: 10 |
  | off | 2 | 1555: 50 |
  | off | 3 | 1420: 17 |
  | on | 0 | 2035: 244, 2841: 38, 2150: 10 |
  | on | 1 | 1555: 50, 1670: 10, 1785: 7, 2150: 1 |

  Two mechanisms: thermal status >= 1 caps policies 4 and 7 at 1420 to 1785 whatever power saving is; at status 0,
  power saving on holds 2035/2035 (R64's value) and off allows 2419/2841. The off probe ran first and started at
  status 3 (battery 31.2 °C, phone warm from use). Under the spin load the on probe stayed at status 0 (2035) and the
  off probe at status 1 (1555). The kit's gate (headroom <= 0.65) does not ensure status 0: status 1 appeared at
  headroom 0.77 to 1.0.
- **Consequence:** memo-31's leading explanation is half right. A claim needs power saving off **and** thermal status
  0; C1's strict uncapped rule (per-run cap before and after) already enforces it. For a later kit: record
  `thermal_status` per run and gate on status 0 as well as headroom.

### R68 (session A; sessions A and B together)

- **Date:** session A started 2026-11-19 15:22 local. **Label:** *device*. **Sources:**
  [device/r28_2026-11-19.json](device/r28_2026-11-19.json) (sha256 `54af112b...f23fba4`, checked on copy) and session B
  [device/r28_2026-11-18.json](device/r28_2026-11-18.json); scorer output
  [memo-33-work/score-r68-sessionsAB.txt](memo-33-work/score-r68-sessionsAB.txt) (`score_r68.py`, run with `python3 -I`).
- **Session A:** deep-sleeping list on; power saving off in 72 of 72 runs; no void; greedy texts identical (olmoe
  num, olmoe off, granite num: one md5 each, same in A and B).
- **b1 OLMoE off:** median 25.20 tok/s, 4 of 10 below 25.0 (prediction 4 to 7: hit), 10 of 10 uncapped.
- **b2 Granite num:** median 23.27 tok/s (inside the band 22.5 to 24.5).
- **b5 OLMoE num + touch** (threads at -10, the app baseline, amendment 4): median **27.25 tok/s**, 0 of 10 below
  25.0, 10 of 10 uncapped. b5 minus b1 = +2.05 tok/s (band +0.4 to +1.6: **MISS, high**; position confounded, b5 last,
  b1 first).
- **Touch:** OLMoE +0.14 ± 0.28 tok/s, PASS; Granite +0.54 ± 0.57 tok/s, PASS. A6 failed in 2 of 18 OLMoE T runs (set
  2894 MB vs want 2980; 2841 vs 2916), which makes b5 ineligible under C5 (the scorer lists olmoe_num_claim.0).
- **A8 over both sessions:** touch PASS on both models. Nice is VOID in both sessions (threads already at -10 in all
  runs, no contrast; amendment 4).
- **Go rule: MISSED.** It reads session B's median (b5 B 25.47 < 25.7, 3 of 10 below 25.0; b1 B 23.03). b5 is also
  ineligible by A6. No claim round is opened by R68's rule.
- *Lead's observation (unscored):* with the list on, session A ran faster than session B on every block (b1 +2.17,
  b2 +0.68, b5 +1.78 tok/s). The sessions are on different days and phone states, so this is not a controlled list
  effect.
- Rehearsal, not a claim. Session R (campaign 1 and stock references) is still to come.

### R68 (session R, c1, 2026-11-19 19:09: void, power saving on)

- **Date:** started 2026-11-19 19:09 local. **Label:** *device*. **Sources:**
  [device/r28-c1_2026-11-19T1909.json](device/r28-c1_2026-11-19T1909.json) (sha256 `5facdf8a...1085855`, checked on
  copy); scorer output [memo-33-work/score-r68-sessionR-c1-1909.txt](memo-33-work/score-r68-sessionR-c1-1909.txt)
  (`score_r68.py`, run with `python3 -I`). Kit r28, runtime c1 (campaign 1's APK).
- **VOID by D1/D2:** power saving read on at start, at end and in 24 of 24 runs (battery level 57 to 53 %, not
  charging). Hashes are ok (D3). Texts: Granite `a93b223e`, OLMoE `f8a9fdc4`, the latter equal to c2's OLMoE off text.
  Both claim blocks have 0 valid runs of 10.
- **Descriptive only, not scored:** cap 2035/2035 in every run. c1 Granite (R49's shipped env) median 20.08 tok/s;
  c1 OLMoE (c1_ship) median 22.66 tok/s (10 claim runs each, *device*). The Granite value is close to R49's 20.51,
  which supports the reading that campaign 1's claim ran with power saving on, but this is not shown.
- **Next:** the owner reruns c1 with power saving off. The stock APK install failed on the phone ("package
  invalid"). The Hugging Face copy of `r28/vollie-r28-stock.apk` was re-downloaded and checked: sha256 `c9b156a1...`
  matches, signature `c49dde85...`, versionCode 28 as c1, zip alignment ok. The file is intact; the likely cause is
  the phone's download.

### R71 R68 re-run in one unattended app (kit r30 S1 and cool, kit r31 S2 and S3)

- **Date:** S1 2026-11-20 20:59 to 21:58, cool-down to 22:29; S2/S3 2026-11-21 10:01 to 11:18 (Resume chain).
  **Label:** *device*.
- **Sources:**
  - [device/r30-chain-0-c2-c2-2026-11-20T2059.json](device/r30-chain-0-c2-c2-2026-11-20T2059.json),
    [device/r30-chain-2-c1-c1-2026-11-21T1001.json](device/r30-chain-2-c1-c1-2026-11-21T1001.json),
    [device/r30-chain-3-stock-stock-2026-11-21T1031.json](device/r30-chain-3-stock-stock-2026-11-21T1031.json);
  - chain state and logs `device/r30-chain*_2026-11-2{0,1}.*`;
  - scorer output [score-r71-S1.txt](memo-33-work/score-r71-S1.txt) and [score-r71-S2S3.txt](memo-33-work/score-r71-S2S3.txt).
  - Pre-registration [R71](../prereg/R71-r68-rerun-one-app-unattended-device.md), [amendment 1](../prereg/R71-amendment-1.md).
- **Valid:** power saving off in 120 of 120 runs. Every load's runtime check is ok (the c2, c1 and stock libraries
  are byte-identical to R68 §0). Texts are identical within each arm.
- **S1, c2 (list on):** b1 OLMoE off 25.53 tok/s (0 of 10 below 25.0); b2 Granite num 23.81; b5 OLMoE num + touch
  28.48 (0 of 10 below 25.0). Every claim-format run was uncapped. The R71 §4 go-rule reading is met (rehearsal).
  - Touch: neither pass nor kill on either model (loss endpoint not testable).
  - Nice: void (no contrast).
- **S2/S3 (a day later):** c1 Granite 21.68, c1 OLMoE 24.28, stock Granite 16.61, stock OLMoE 23.68 tok/s.
- **Predictions:**

  | prediction | result |
  |---|---|
  | b1, b2, below-25 count | inside |
  | b5 (band 25.0 to 27.8) | MISS high |
  | c1 both models, stock Granite | inside |
  | stock OLMoE (band 6 to 18) | MISS high |
  | order c2 > c1 > stock on both models | hit |
  | chain finishes without a stop | MISS |
  | nice void | hit |
  | touch pass | MISS on both (not testable) |
- **Chain:** r30 stopped at the restart into c1. Diagnosis (inferred, emulator-reproduced): r30 cleared
  keep-screen-on at the end of S1; the 10 min screen timeout, confirmed by the owner, put the phone to sleep; Android
  15 then blocked the restart. r31's restart worked on the phone on the first try.
- **Lead's reading (cross-day, not scored):**
  - OLMoE: c2's lead over stock is mostly the num arm (28.48 / 25.53 / 23.68).
  - Granite: it is mostly the runtime (23.81 against 16.61).

### R70 Claim round: OLMoE num + touch, list on, power saving off (kit r31 c2 runtime, amendments 1 and 2)

- **Date:** 2026-11-21 11:22 to 12:23 local. **Label:** *device*.
- **Sources:** [device/r31_2026-11-21T1122.json](device/r31_2026-11-21T1122.json) (sha256 `4cae5203...131d8a6`);
  scorer output [memo-33-work/score-r70.txt](memo-33-work/score-r70.txt); pre-registration
  [R70](../prereg/R70-claim-olmoe-num-touch-list-on-device.md),
  [amendment 1](../prereg/R70-amendment-1.md), [amendment 2](../prereg/R70-amendment-2.md).
- **Void checks (§2 and amendments):** none met.
  - Model and library hashes ok; runtime c2, `runtime_check` ok in every load.
  - Power saving off in 72 of 72 runs; charger out; not killed; label "list on"; 10 of 10 scored b5 runs.
  - b5 text `40f6ce50` in all 12 rows, equal to R68's.
- **Claim rule (§2): PASS.** The b5 median is **27.21 tok/s**, at least 25.0 (1 of 10 below 25.0; 2 of 10 capped at
  2150 at the start, counted as the rule says).
- **Deviation, stated with the claim:**
  - R70's phone state asks for a 30 min rest before the start; amendment 1 asks for it after R71.
  - The session started 3 min 52 s after R71's S3 ended (11:18:20), at battery 47 %, so the start was **warm,
    not cool**.
  - The rest rule is not in §2's void list, so the verdict stands as written.
  - The claim is therefore stated with "warm start" in place of §4's "cool start".
  - The deviation works against the claim: a warm phone, and two capped runs, none of them in S1 of R71.
- **Other blocks:** b1 OLMoE off 25.14 (3 of 10 below 25.0); b2 Granite num 23.21 (10 of 10 below 25.0; Granite
  does not reach 25).
- **Predictions (§3):**

  | prediction | result |
  |---|---|
  | b5 (band 24.8 to 28.0) | inside |
  | below 25.0: 1 (band 0 to 3) | inside |
  | P(pass) 0.80 | pass |
  | uncapped in at least 9 of 10 b5 runs | MISS (8) |
  | b1 (band 23.0 to 26.0) | inside |
  | b2 (band 22.0 to 24.5) | inside |
- **Result (§4):** campaign 2's goal is met on OLMoE-1B-7B (`olmoe-Mi`, budget +5 %): **median 27.21 tok/s,
  *device*, R70**.
  - Conditions: power saving off, deep-sleeping list on, warm start, internal model copy.
  - Granite-4.0-H-Tiny reaches 23.21 tok/s (num) under the same conditions; it does not reach 25.
  - References under power saving off and list on (R71): campaign 1's runtime OLMoE 24.28 / Granite 21.68, stock
    OLMoE 23.68 / Granite 16.61.

Memos (not R-entries): [memo-01, candidate re-check and host screen plan](memo-01-candidates-and-host-screen.md);
[memo-28, code audit of the kit app and runtime: thread pin lost in gated runs, per-run waste](memo-28-code-audit.md);
[memo-29, joint round (Bonnie and Ebony): TEST list T1-T11 and TRY list Y1-Y12 for Tom; working files in `memo-29-work/`](memo-29-joint-test-and-try-lists.md);
[memo-32, kit r28 (three APKs c2, c1, stock; touch x nice; T-A probes; sessions A, B, R)](memo-32-kit-r28.md);
[memo-31, joint round (Bonnie and Ebony): claim confidence not sufficient; tail types A and B; go rule; TEST list T-A..T-E and TRY list for kit r28; working files in `memo-31-work/`](memo-31-claim-confidence.md);
[memo-30, kit r27 (persistent pool + per-thread mask, patch 0010; PROT_EXEC model mapping, patch 0011; pin x exec 2x2 per quad; internal-storage exec check and copy; T1, T3 riders)](memo-30-kit-r27.md);
[memo-27, kit r26 (kv clear on the untagged address; one arm per model, r16 gate, kv in ABBA quads)](memo-27-kit-r26.md);
[memo-26, kit r25 (kit r24 plus the kv clear by MADV_DONTNEED, patch 0008; kv in ABBA pairs)](memo-26-kit-r25.md);
[memo-25, routing prior toward recently used experts (patch 0007), host study, not an R-entry](memo-25-routing-prior-host.md);
[memo-24, the app's anonymous memory through a wait and the "rest" of a token, on both models](memo-24-anon-memory-and-rest.md);
[memo-21, unconventional levers priced on both models (owner's list and new directions)](memo-21-unconventional-levers.md);
[memo-22, kit r23 (head knob at load, cheap hold, threads 3 against 4)](memo-22-kit-r23.md);
[memo-23, kit r24 (plateau gate rule, charger guard, swap counters) and why runs fault: zram swap-ins](memo-23-kit-r24.md);
[memo-02, which campaign-1 levers apply to `olmoe` and `lfm2moe`](memo-02-lever-applicability.md);
[memo-05, why skip was inert on Q4_0 experts: repack](memo-05-q4-experts-repack.md);
[memo-08, predictions for the device baselines of OLMoE; Granite verdict proposed](memo-08-device-baseline-predictions.md);
[memo-06, adb CLI kit (abandoned: the phone cannot be attached)](memo-06-device-cli-kit.md);
[memo-09, how a research APK is built](memo-09-research-apk-path.md);
[memo-10, kit r17](memo-10-kit-r17.md);
[memo-12, kit r18 (lever kit)](memo-12-kit-r18.md);
[memo-14, kit r19 (campaign 1's runtime, fixed gate, residency)](memo-14-kit-r19.md);
[memo-13, diagnosis of R52: page cache, heat, gate bug, campaign 1's runtime on OLMoE](memo-13-r52-diagnosis.md);
[memo-15, cost model refit with I/O and clock terms; residency odds](memo-15-cost-model-refit.md);
[memo-16, why R55 is far from campaign 1: r16 against r19](memo-16-why-far-from-campaign-1.md);
[memo-17, kit r20: campaign 1's r16 protocol on OLMoE, warm block, keep-warm](memo-17-kit-r20.md);
[memo-18, residency options priced with a page-cache simulator; corrected working sets](memo-18-residency-options.md);
[memo-19, kit r21: two models in one session, prefill-cold](memo-19-kit-r21.md);
[memo-20, kit r22: numerics arm and page hold on two models](memo-20-kit-r22.md);
[memo-03, lever pricing](memo-03-lever-pricing.md);
[memo-04, OLMoE expert-skip](memo-04-olmoe-expert-skip.md);
[memo-07, campaign-1 exact levers in the c2 series, identity on OLMoE](memo-07-c2-exact-levers.md);
[memo-11, kit interface in the runtime (0008) and the untied two-stage head (0009)](memo-11-kit-runtime-and-head.md).
