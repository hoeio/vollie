# Pre-statement R61: stock and num perplexity on a host build with KleidiAI (the phone's build option)

Written and pushed before the runs. A host measurement, not a device round; changes go into an amendment.

- **Date / author:** 2026-11-15, clyde (lead). **Owner decision (2026-11-15):** option (a): measure with
  KleidiAI on the host and state both references side by side; the budget rule is unchanged.
- **Agent models:** clyde `claude-fable-5-2`; tom `claude-opus-5-5` (runs the queue on the owner's laptop).
- **Why:** the phone's CPU module is built with `GGML_CPU_KLEIDIAI=ON`; R50, R51, R53 and R59 and both budgets
  come from host builds without it. KleidiAI computes the Q4_0 attention tensors and OLMoE's Q6_K head
  differently (R60 follow-up, *host*, 4 chunks: per-chunk values move by up to about 0.5 %).
- **Build:** `lab/llama.cpp-c1h6/build-kai`: llama.cpp `620d2225c` + `patches/llama.cpp/0001` to `0003` +
  `patches/llama.cpp-c1-olmoe/0001` to `0006`, KleidiAI on, SME off. Limit: on the M4 KleidiAI picks its i8mm
  kernels, the phone its dotprod kernels; the two may differ in the last digits. **Label:** *host*.
- **Runs (full wikitext-2 test, `-c 512 -t 8 -ngl 0`, each at `-b 512` and at `-ub 1`), 10 in all:**
  1. OLMoE-Mi, kit off env (the phone's off arm in R60: head in KleidiAI's int8).
  2. OLMoE-Mi, kit off env + `VOLLIE_HEAD2_REPACK=1` (the off arm of the next kit).
  3. OLMoE-Mi, 2 + `VOLLIE_SKIP_MASS=900 VOLLIE_HEAD2=1024` (num).
  4. Granite-Mi, kit off env.
  5. Granite-Mi, 4 + `GGML_EXPERT_SKIP=130 GGML_HEAD2=2048 GGML_HEAD2_Q=q4_0` (num).
  Batch-mode test switches the head needs under `llama-perplexity` are set as in R59.

## How the results are read (fixed now)

- The existing budgets (OLMoE 12.3003, Granite 9.6879; stock 11.7146 and 9.2266 without KleidiAI) are **not
  changed**. A second column is added: stock with KleidiAI (rows 2 and 4, the next kit's off arm) and that value
  × 1.05.
- A num setting is **inside on the phone's build** if rows 3 and 5 are at or below (same-build stock × 1.05) at
  both `-b 512` and `-ub 1`, each against the stock value at the same setting, **and** at or below the existing
  absolute budget. If a setting is outside either, it is not used in a device round as it stands; the threshold
  is re-screened and the result goes to the owner. The F16 × 1.20 ceilings (13.1304, 10.5379) stand.
- Row 1 against row 2 says what R60's OLMoE off arm ran against the file's own head; if row 1 is above
  11.7146 × 1.05 that is reported as a fact about R60, not a budget event (no lever was on).

## Predictions

- Stock with KleidiAI is within ±0.5 % of the value without it on both models and both settings (rows 2, 4);
  row 1 within ±0.7 %.
- Both num settings stay inside by both readings; the per-cent increase over same-build stock is within
  ±0.3 points of R59's (+3.34 / +3.26 % OLMoE, +3.15 / +3.04 % Granite).
- Miss on either: reported as a miss; a num setting outside is handled as written above.
