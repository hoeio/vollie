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

Memos (not R-entries): [memo-01, candidate re-check and host screen plan](memo-01-candidates-and-host-screen.md);
[memo-02, which campaign-1 levers apply to `olmoe` and `lfm2moe`](memo-02-lever-applicability.md);
[memo-03, lever pricing](memo-03-lever-pricing.md);
[memo-04, OLMoE expert-skip](memo-04-olmoe-expert-skip.md).
