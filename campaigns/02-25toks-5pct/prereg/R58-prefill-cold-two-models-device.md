# Pre-registration R58: prefill-cold on two models in one session (kit r21)

Committed and pushed **before** the APK is handed to the owner. Changes go into `R58-amendment-1.md`.

- **Date / author:** 2026-11-14, clyde (lead).
- **Starting point:** campaign 1's runtime, shipping settings and kit protocol (r16's gate). Two purposes of the
  campaign: the lever is measured on two models in the same session and reported per model.
- **Models / files:** stock Granite-4.0-H-Tiny, `granite-Mi.gguf`,
  `8df94de731a08a377150e4fcc5dcbd16212a6d102bc0b628cdc905cba0bc0f45`; OLMoE-1B-7B-0125-Instruct, `olmoe-Mi.gguf`,
  `5abf710b775cd64fd89d74a783c8a7056a409ba795cd401b9dd448ef8e444abb`; both from `HoeioUser/vollie2` @ `e893e8a6`.
- **Agent models** (full IDs): clyde `claude-fable-5-2`; tom `claude-opus-5-5` (memo-18, memo-19, patch, kit);
  bonnie `claude-opus-5-5` (memo-15, used by memo-18); annie `claude-sonnet-5-5`.
- **Build under test:** APK `vollie-c2-1.21.0+21-research-r21.apk`, 17,189,039 bytes, sha256
  `86dd8d34f9150afcb825d29c417cd36efbd85468aec46ca70bd4cf4e6a15e380`, package `com.insaneutils.vollie.c2`. Runtime:
  llama.cpp `620d2225c` + campaign 1's `patches/llama.cpp/0001` to `0003` + `patches/llama.cpp-c1-olmoe/0001`,
  `0002-prefill-cold.patch`; app patches `patches/app-c2/0001` to `0005`. Hashes of all are in the report.
- **Labels:** *device* where a value names R56 or R57, *host* for traces and identity checks, *estimated* for
  everything predicted (memo-18's simulator, fitted on R56 and R57 themselves).

## 1. What is tested

`GGML_PREFILL_COLD=8` ([memo-19](../reports/memo-19-kit-r21.md)): during the prefill, expert slices that the
prompt routed fewer than 8 times are marked cold (`MADV_COLD`, else `MADV_PAGEOUT`), so the page cache keeps what
generation will use. Advice only: per-chunk perplexity and the greedy text are identical with the knob on and
off on both models (*host*, 4 chunks, `-b 512`, `-ub 32`, `-ub 1`). No numerics change; budgets untouched
(Granite 9.6879, OLMoE 12.3003). Both models run campaign 1's shipping settings without skip and without the
two-stage head; repack on.

Run list: blocks Granite, OLMoE, OLMoE, Granite; each block: load, 2 unscored warm-ups, 8 scored official runs
with the knob in the order off, on, on, off, off, on, on, off, one profiled run per knob state; r16's gate
before each run. 48 runs.

## 2. Prediction

The phone's page cache is not ours to set; "low" = `Cached` before runs 1.8 to 2.3 GB (as on 2026-11-14),
"high" = 2.5 GB or more (as on 2026-11-08). Medians of the scored runs per model and knob state:

| | knob off, low | knob off, high | effect of the knob (on against off, same block) |
|---|---|---|---|
| Granite | 10 to 14 tok/s | 17 to 21 | **+25 %**, band +10 to +40 % (low); +0 to +10 % (high, already resident) |
| OLMoE | 8 to 12.5 | 11 to 15 | **+14 %**, band +5 to +25 % (low and high) |

Further predictions:
1. Decode read bytes per run (from the `c2-io` line) fall with the knob on: Granite by at least 15 %, OLMoE by
   at least 10 %.
2. Time to the first token is shorter with the knob on by 0.6 to 2.0 s on both models.
3. Marked per official prompt: about 740 MB (Granite) and 645 MB (OLMoE), as on the host.
4. Generality: the lever helps both models, Granite more than OLMoE; at equal page cache Granite is faster than
   OLMoE with the knob off (memo-18: Granite's working set is 1.97 to 1.99 GB, OLMoE's 2.67 GB).

Resolution: single runs spread by about ± 30 % on this phone in this state (R56, R57); each model has 16 scored
runs, 8 per knob state, in ABBA order. The effect is taken as the mean of the within-block on-minus-off
differences with its standard error; an effect smaller than twice its standard error is "unresolved".

## 3. Pass rule and kill rule

- **Pass, per model:** on-minus-off positive and resolved, **and** decode read bytes lower as in prediction 1.
- **Kill (written with the lever, memo-18):** Granite's decode read bytes not lower by at least 15 %, or no
  resolved tok/s gain on either model. Then the lever is closed and the next residency option of memo-18 is taken.
- **Generalised / did not generalise** is reported explicitly: pass on both, on one (which), or on none.
- Each numbered prediction is scored hit or miss on its own.
- **Void:** the advice line reports "none" (the phone's kernel offers neither advice; then the lever cannot be
  tested on this phone and that is the result); reply texts differ between knob states of a model; a model or
  library hash mismatch; fewer than 6 scored runs per knob state for a model.
- **Nothing here passes or fails 25 tok/s.**

## 4. Hygiene counts

Per run: gate readings and timeout flag, clock caps, mincore of the model file (experts, rest) before and
after, `Cached`, free swap, RssFile, RssAnon, major and minor faults, the `c2-io` line (read bytes: prefill,
decode), MB marked, reply text. State flags do not remove runs; the block order (Granite, OLMoE, OLMoE,
Granite) is what makes the two models comparable on the day.

## 5. What a miss means

- Knob helps Granite and not OLMoE: expected direction of memo-18 taken to its end (OLMoE's decode set alone
  exceeds the cache); the lever is kept for Granite and reported as not general.
- Knob helps neither although decode bytes fall: the I/O is not what limits the first tokens; look at the
  per-token profile (clock at the first tokens, read-ahead) before the next option.
- Knob-off medians outside both state rows: the simulator's fit does not hold on a new day; refit before
  using it again.
