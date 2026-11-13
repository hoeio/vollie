# R51 amendment 2: the frontier on the phone's kernel path (no repack)

- **Date / author:** 2026-11-14, clyde. Committed before any `-nr` full-set log exists.
- **Agent models:** as in R51; the finding is tom's ([memo-05](../reports/memo-05-q4-experts-repack.md)).
- **Labels:** *host* for R50/R51 values and 4-chunk checks, *estimated* otherwise.

## Why

Memo-05 established two things. (1) llama.cpp repacks every Q4_0 tensor at load by default, and the repacked
`MUL_MAT_ID` path ignores the c2 skip, which is why amendment 1 was void. (2) The app loads files of this size
**without** repack on the 8 GB phone (`use_extra_bufts = false`: total RAM minus file size is below 5 GB), so the
phone runs the un-repacked kernels. R50 and R51 were measured with the host default, repack **on**. The two
paths are not bit-identical: on the first 4 chunks `olmoe-Mi` gives 8.9646, 10.4244, 9.2074, 9.4783 with repack
and 8.9278, 10.4339, 9.2003, 9.4521 without (*host*); Granite-Mi differs in the same way.

The budgets of R50 stay as written (stock × 1.05 on the R50 measurement). This amendment checks that the R51
verdict also holds on the path the phone runs, by measuring reference and skip settings **both** with `-nr`.

## What is run

`tools/c2_skip_sweep.sh` with `PPL_ARGS=-nr`, same build, text, flags, 560 chunks:

1. `olmoe-Mi.gguf`: `stock MASS=900 SUM=70 SUM=75`.
2. `OLMoE-1B-7B-0125-Instruct-Q4_0.gguf`: the nine settings of amendment 1 (its re-run; with `-nr` the skip code
   is reached).

## Predictions

1. `olmoe-Mi`, `-nr` stock PPL within ± 0.5 % of 11.7146 (band 11.656 to 11.773).
2. `olmoe-Mi`, each setting's "% over its own `-nr` stock" within ± 0.4 points of its R51 value (`MASS=900`
   +3.32, `SUM=70` +3.50, `SUM=75` +5.14), and bytes skipped within ± 0.3 points (13.39, 13.07, 17.48 %).
3. Q4_0 file: `-nr` stock within ± 0.5 % of 11.3784; f\* against its own `-nr` stock × 1.05: central 11 %, band 6
   to 16 % (amendment 1's prediction, unchanged).

## Rules

- If prediction 2 holds for `MASS=900` (ratio ≤ +5.00 % in particular), R51's f\* = 13.39 % stands for the phone
  path. If `MASS=900` exceeds +5.00 % of the `-nr` stock, R51 gets a dated correction: f\* on the phone path is
  the largest setting that stays inside, and that value is used from then on.
- A served skip setting is judged, when it is fixed in a device pre-registration, by its `-nr` full-set PPL
  against the model's budget **and** by its ratio to the `-nr` stock; the stricter of the two decides. That is
  written here so the looser one cannot be picked later.
- Hygiene: every log 560 chunks, `n_seq=1`, no `CPU_REPACK` buffer line; skip rows have a counter line.
- No speed is read from this round. Host wall time without repack is not a phone number.
