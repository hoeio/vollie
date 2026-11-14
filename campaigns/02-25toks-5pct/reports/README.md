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

Memos (not R-entries): [memo-01, candidate re-check and host screen plan](memo-01-candidates-and-host-screen.md);
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
[memo-03, lever pricing](memo-03-lever-pricing.md);
[memo-04, OLMoE expert-skip](memo-04-olmoe-expert-skip.md);
[memo-07, campaign-1 exact levers in the c2 series, identity on OLMoE](memo-07-c2-exact-levers.md);
[memo-11, kit interface in the runtime (0008) and the untied two-stage head (0009)](memo-11-kit-runtime-and-head.md).
