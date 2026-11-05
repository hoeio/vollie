# Mirror audit, 2026-11-05 (Annie)

Scope: every tracked file of github.com/hoeio/vollie (head 133c16a at the start, 9e318a6 at the end).
Mechanical checks that passed: `results.csv` 108 rows, all 9 columns, header as documented, LF only, no CRLF;
all six `reports/device/r*.json` parse and `python3 -I tools/research_parse.py` runs on each; `bash -n` clean on
every `*.sh`; every relative link in every `*.md` resolves; `patches/llama.cpp/0001` and `0002` apply cleanly, in
order, to a clean worktree at BASE 620d2225c (`BASE` file states it; both patches are byte-equal to
`lab/tiny-op.patch` and `lab/granite-rollback.patch`); the llamadart patch is content-identical to a fresh diff of
the vendored `lib`/`hook` against pub-cache llamadart 0.15.6 (only the `a/`,`b/` prefix style differs) and applies
with `patch -p0`; every `GGML_*` knob in `reproduce.sh` exists in the patch; no `/Users/...` path and no e-mail in any
tracked file; `.gitignore`, `LICENSE` (MIT) present; no stray files. README round-6 row (6.1 -> 9.9 -> 14.0, 8.9 ->
12.05 -> 12.7, 100.9 -> 74.6 -> 63.7, ratio 1.35) and the round-5 table match the R25/R27 blocks and `results.csv`.
No withdrawn claim is stated as live (the x1.40 appears only in R24-era history, now marked withdrawn).

## (A) Flaws fixed

- 33a36ce README: "latest: 2.2.10+33, kit r5" -> 2.2.11+34, kit r6; prefetch described as skip-aware (R26); in-place conv state listed; skip-rule range R21-R25 -> R21-R27.
- 33a36ce `docs/runtime-knobs.md`, `patches/README.md`: `GGML_CONV_RING` was documented "default off"; the patch says default on since r6 (empty = unset = on). Intro line "unset or 0 means stock" now names the default-on knobs.
- 33a36ce `docs/runtime-knobs.md`: four "device pending" cells replaced by logged results (MMID / Q3K_X4: -8.3 %/token, R18; helper on an A55: no better than 3 threads, closed; REPACK_RELEASE: inconclusive).
- 33a36ce `patches/README.md`: skip-aware read-ahead and the `adv=` field described.
- 9e318a6 `03_final.md` header "Last updated 2026-11-02 / R22" -> 2026-11-05 / R27, with a note that section 1 and sections 6-7 are the 2026-11-02 snapshot.
- 9e318a6 `03_final.md` section 1: note that its gate column (<= 9.948, +10 %) predates the owner's +20 % (<= 10.853) and +2 % (<= 9.797) rulings.
- 9e318a6 `03_final.md`: "phone run pending" in the R24 / R25 / R26 headers (and "build pending" in the R24 pre-registration) now point to the block that scored them.
- 9e318a6 `03_final.md` section 8: pointer to `tools/reproduce.sh`, `android/build_cpu.sh`, `patches/` for a reader without the private tree.
- 9e318a6 `03_final.md` line 160 and `05_compute_less.md` lines 778-779: "+3.60 % / +5.20 %" for skip 130/140 annotated "as first written; +3.67 % / +5.25 % over 9.6051, corrected in R25" (the correction already stood in `04_campaign.md` section 12 and the `results.csv` R25 threshold row); original text kept.
- 9e318a6 `04_campaign.md` section 11 and `05_compute_less.md` section 7.3: "x1.40" marked "withdrawn in R25".
- 9e318a6 `04_campaign.md` section 12 and `05_compute_less.md` (EXPERT_IO thread-0 madvise paragraph): madvise lever marked closed by R27.
- 9e318a6 `05_compute_less.md` header: "Nothing here was run on the phone" corrected (true for sections 0-4 only).
- 9e318a6 `02_methods.md`: the "+10 %" gate marked as the limit in force then.

## (B) Requests

1. **`docs/findings.md` stops around R20 and calls the open problem "expert compute and waits down to ~25 ms".** Fix: a new section "5. The compute-less rounds (R21-R27)" covering skip rule, ABBA, vanilla vs Vollie, clock test, what is closed, where 20 stands. Who: Bonnie (memo text), numbers from R25/R27 only. Why: README links it as "findings in the order they were made"; a stranger reads it first and finds the story ends before the headline result.
2. **`03_final.md` structure is not in R-number order and section 5 is a flat run of bold paragraphs.** R23 blocks sit after R26 (lines ~228-296 after 219), R20 and R18 after that, R27 inside the same section; no per-R headings, so links like "R25" cannot be anchored. Fix: a mechanical re-sort into `### R<n>` headings in numeric order (Annie can do it; it moves text but changes none, so Clyde should approve a content-preserving move) plus a 10-line index table at the top. Who: Annie after Clyde's yes. Why: a reader cannot find "R23" by scrolling.
3. **`03_final.md` section 1 and sections 6-7 are a 2026-11-02 snapshot** (best = 12.2 tok/s, "20 is the plausible ceiling", open-items lists "after R20"). Annotated, not rewritten. Fix: Clyde rewrites section 1 to the R25/R27 same-day table (the README table is the source). Who: Clyde (verdicts are interpretation). Why: the first table of the log disagrees with the README.
4. **`04_campaign.md` section 13 and `03_final.md` R27 cite "the dense `MUL_MAT` byte memo (section 8)" of `05_compute_less.md`; that section does not exist** (memo ends at 7.8). Fix: Bonnie's memo, then either number it section 8 or reword the references. Who: Bonnie. Why: dangling cross-reference in the paragraph that names the only path to 20.
5. **No raw JSON for the store build 2.1.10+22 (README last column).** The column is transcribed from `results.csv` and R9 only. Fix: if the owner still has the speed-test text export of that build, add it as `reports/device/store_2.1.10+22.txt`; otherwise nothing, the README note already says so. Who: Clyde asks the owner. Why: it is the only table column without an evidence file.
6. **Date of the original host repack run (3,931 MB) is not recorded.** Fix: if `lab/logs` has the file mtime, state it in `results.csv` row "A file with repack ON"; a log lookup, no new run. Who: Tom. Why: README note says "not recorded in this repo".
7. **Host-only claims are reproducible but device claims are not, and `reproduce.sh` was last run on the M4 only.** Fix: one `tools/reproduce.sh --quick` run on a clean Linux x86 container or second machine, and report PASS/FAIL, so a stranger's first run is known to work. Who: Tom. Why: the README tells strangers to run it; an untested second platform is the likeliest first failure.
8. **Several tools and docs depend on the private `lab/` layout** (`ppl.sh`, `host_bench.sh`, `tools/checks/*`, `docs/workspace.md`, `03_final.md` section 8). They state it, but a stranger cannot run them. Fix: either make `ppl.sh` take `--llama-dir`/`--text` arguments (Tom, small), or add the sentence "needs the maintainers' lab layout; `reproduce.sh` is the portable entry point" to each script header. Who: Tom. Why: broken-looking scripts read as a flaw.
9. **The research kit card (`tools/research_kit.md`) is for the build already scored.** Fix: when kit r7 is built, replace the card; until then no action. Who: Tom at the round-7 build.

## (C) Unsure / for Clyde

- `runtime/dart/pubspec.lock` is tracked although `.gitignore` lists `pubspec.lock` (the ignore does not untrack it). A library package normally does not commit it. I left it; `git rm --cached` is one line if you say so.
- The annotations I added to `03_final.md` / `04` / `05` are in brackets or after the original wording, original text intact. If you consider even that an edit of findings, `git revert 9e318a6` removes them in one step (not history rewriting).
- I read "R24 one-order A/Bs are superseded by R25/R27 ABBA" as already stated in `04_campaign.md` section 11-12 and `03_final.md` R25 ("Anything one-order in r3/r4 ... is superseded"); I did not add a banner to the R23/R24 blocks. Say if you want one.
- `docs/runtime-knobs.md` section heading "Round-1 knobs ... All off unless set" still sits above knobs that are now default on; the intro now carries the exception, the heading is unchanged.
- The `03_final.md` R-blocks mix `R24 pre-registered` / `R24 build` / `R24 results` headers under one number, and R26/R27 are numbered by the round and the scoring, not in sequence with R23; I did not renumber.
