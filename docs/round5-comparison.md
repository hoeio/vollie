# Round-5 comparison: vanilla llama.cpp vs Vollie, same phone (2026-11-04)

This page holds, word for word, the comparison table and its notes that used to sit at the bottom of the
README. They were kept unchanged from earlier versions of the README and are moved here only so the README
can be short. The numbers are the round-5 numbers (2026-11-04, kit r5); the current result is in the
[README](../README.md) and in [R49](../reports/03_final.md).

## Vanilla llama.cpp vs Vollie, same phone

| | vanilla llama.cpp | Vollie, bit-identical runtime | Vollie + expert skip 118 | Store build 2.1.10 (earlier test, D1 file, not same-day; see *Note on the store build*) | label | where logged |
|---|---|---|---|---|---|---|
| Private memory (anon) after the run | 679-690 MB; not a difference between columns: mmap weights with repack off are shared by all three (see *Footnote*). The stock-repack comparison is a different test, see *Note on stock repack* | 692-707 MB (the two baseline runs; 645-707 MB over all 13 runs of the day) | 660-689 MB | not measured on the phone (anon not logged); estimated need 3.8 GB resident = 3.1 GB mmapped experts + 0.66 GB repack copy | device | `reports/device/r5_2026-11-04.json` (`mem_after`, app 2.2.10+33) |
| Writing speed, official 32-token bench (decode) | 9.4 tok/s | 12.0 tok/s (better of two same-session baseline runs; the first read 8.2 with 97 MB of mapped file pages resident at start, the second 12.0 with 251 MB) | 13.9 tok/s | 6.1 tok/s best run (2 threads, 32 written tokens, heat 76-83 %); the same session's knob sweep read 4.3-4.4 at 2 threads | device | `03_final.md` "R25 — round-5 device results"; `results.csv` row "R25 VANILLA VS VOLLIE"; `reports/device/r5_2026-11-04.json` |
| Reading speed (prefill, same runs) | 15.1 tok/s | 16.2-18.0 tok/s | 21.6 tok/s | not logged for this build | device | same as above |
| 200-token story, printed | 10.2 tok/s | 14.5 tok/s | 12.5 tok/s (see note) | no such test in this build | device | same as above |
| 200-token story, steady profile (ms per token, last windows) | 82-97 | 62-66 | 55.4 (≈ 18 tok/s steady) | 137-163 (2 threads; average over a whole profiled reply, not a steady window; 354-1,181 major faults per token) | device | same as above |
| Output identical to stock | (reference) | yes for every default knob (bit-identical at `-ub 1/32/512`) | no (expert skip) | yes (tiny-op v2 only, host-checked, plus stock dense repack) | host | `RULES.md` §4; `results.csv` identity rows (tiny-op v2, Q3_K x4, R23/R24 probes) |
| Quality, wikitext-2 PPL | F16 9.0438; served file 9.6051 (+6.21 %) | 9.6051 (+6.21 %) | 9.7952 (+1.98 % over stock; budget +2 %, limit 10.853) | D1 file: 9.6230 (+6.40 % over F16, +0.19 % over the served file) | host | `results.csv` rows "D6" and "R25 SKIP THRESHOLD EDGE"; `RULES.md` §1 |

*Note on the store build.* The last column is app **2.1.10+22** (R9), the version the owner states is in the store
on 2026-11-04. It was measured with the app's own speed test and profile, before the research kit existed: there is
no raw JSON, the numbers are as transcribed in `results.csv` (row "app 2.1.10+22") and `03_final.md` R9. The owner
recalls it was before about 18:00 on 2026-11-01; the repo logs the build number, not the date. It ran the **D1-e5
file** (`vollie-granite-4.0-h-tiny-d1-e5.gguf`, Q4_0 dense + IQ4_XS experts, 5 experts, 3.77 GB), not the served D6
file: dense weights repacked, experts read through mmap, tiny-op v2 only, no expert prefetch, 2 threads. File,
knobs, free memory, heat and test protocol all differ from the same-day columns, so no speedup should be computed
between this column and the others; it records where the store version stands.

*Note on stock repack (not same-day, not the served file).* The 3,931 MB figure that stood in the memory row
earlier is a **host** number, not a phone number: Apple M4 Pro, `/usr/bin/time -l` peak memory footprint,
llama.cpp v0.8.0@8642541b, the first Q4_0 file (4 experts, PPL 10.1619, +12.4 %: it failed the 10 % gate in force
then; the limit has since been raised to 20 %, see RULES.md §1) with repack on, against
128 MB with repack off. On the phone that configuration was killed by Android (`CPU_REPACK` 3105 + KleidiAI
541 MiB), which is the only device observation behind it. It was not repeated for the served file (its Q3_K
experts have no repack path) or on 2026-11-04; the date of the original host run is not recorded in this repo.
Logged in `results.csv` (row "A file with repack ON"), `reports/00_model_choice.md`, `reports/02_methods.md`. The
older 674 MB anon figure (app 2.2.2+25, R12) is superseded by the same-day values in the table.

*Note on the 12.5.* That run was in a worse page-cache state (77 vs 34 major faults per token). The in-run A/B
measured the skip rule at −7.6 ± 6.0 ms/token (−8.3 ± 2.9 at threshold 115), i.e. ≈ ×1.12-1.15 over the
bit-identical runtime.

*Round 6, same-day row (device).* 2026-11-05, fast cores capped at 1,555 MHz (r5: 2,112): official 32 vanilla 6.1 → Vollie 9.9 (bit-identical) → 14.0 (skip 118); story 200 printed 8.9 → 12.05 → 12.7; steady 100.9 → 74.6 → 63.7 ms/token; ratio 1.35.

*Footnote.* All speed numbers in the table above, except the store-build column and the round-6 row, are from one day (2026-11-04, a swapping day: swap free 0-386 MB,
6-110 major faults per token), same app (2.2.10+33, kit r5), same file, 3 threads, measured by the owner
through the app. Each cell is a single run, except the bit-identical official-32 decode figure: it is the better of
two baseline runs (8.2 and 12.0), since RULES.md §5 scores a setting by its best run; the vanilla figure is one run.
Printed tok/s over 200 tokens varies with page-cache state more than with the runtime (the
same build printed 14.5 and 12.5 in consecutive runs). "Vanilla" = every Vollie CPU-module knob at its stock
value; what env knobs cannot switch, and so is shared by all columns: q8_0 KV cache + flash attention, mmap
weights with repack off, big-core pinning + 3 threads, the Granite rollback patch, context/template/greedy
decoding. The repack row is the first Q4_0 file (the served file's Q3_K experts have no repack path, so stock
llama.cpp may well load it). *Device* = measured on the phone by its owner through the app; *host* = Apple M4 Pro,
CPU only, relative or machine-independent only.
