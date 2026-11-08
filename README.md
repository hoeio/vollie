# Vollie: a mixture-of-experts LLM at 20 tok/s on a 2021 phone CPU

## What Vollie is

Vollie is an on-device assistant app for Android. This repository is its serving research: runtime
patches for llama.cpp, the tools that check them, and the full log of every measurement. The app itself
is not here.

- Device: Samsung Galaxy Z Flip 3 (Snapdragon 888: 1x Cortex-X1 + 3x A78 + 4x A55, 8 GB).
- Model: Granite-4.0-H-Tiny abliterated (7B mixture-of-experts, 64 experts, Mamba-2 hybrid), one frozen
  GGUF ([recipe and sha256](models/granite-4.0-h-tiny-d6-e6.md); weights on Hugging Face in
  [HoeioUser/vollie-qwen3.5-4b](https://huggingface.co/HoeioUser/vollie-qwen3.5-4b), a repository name
  from before the switch to Granite).

## The result

Vollie's own benchmark (official prompt, 32 tokens, greedy, llamadart's `decodeTokensPerSec`) printed a
**median of 20.51 tok/s** over 10 runs (*device*, R49, round 16, research build 2.2.21+44, kit r16).

- 8 of 10 runs printed 20.2-20.95. Best 20.95. The 10 runs: 14.09, 18.47, 20.63, 20.62, 20.23, 20.83,
  20.95, 20.40, 20.84, 20.25.
- Every one of them ended with the fast cores at 2,112 / 2,035 MHz. Each run waited first at the kit's
  cool-down gate, as the app's speed test does.
- The claim rule was written before the run: "20 reached iff the median of the 10 plain runs is
  ≥ 20.0", never from a best run. The prediction was median 18.0-20.0, central 19.0. It missed high.
- Greedy text was identical across all 19 official runs of the session.

**Cold start is slower: 12.3-18.5 tok/s** (*device*, R49). That is the kit's cold baseline (12.3, its first
run) and the two runs after the heavy memory use of the reference stories (14.09, 18.47). The cause is
page residency, not arithmetic: those runs started after the stories or idle time had evicted the
model's pages, which then faulted back in during the prefill and the first decode tokens. Every plain run
held the 2,112 MHz clock. Once the pages are resident, the steady official token is ≈ 45.5 ms in the
graph + ≈ 1.05 ms outside it (*device*, read from the per-token lines of three resident runs, not a
scored line).

For scale, two earlier days, each measured against Vollie in the same session (*device*, same phone,
same model file):

| | official 32, printed tok/s | where logged |
|---|---|---|
| stock llama.cpp settings (every Vollie knob off; see the footnote of the table at the end) | 9.4 (2026-11-04) / 8.50 (2026-11-06) | R25 / R31 |
| Vollie, bit-identical runtime only | 12.0 (2026-11-04, better of 8.2 and 12.0) | R25 |
| Vollie, served config, round 16 | 20.51 median (2026-11-08) | R49 |

## The rules it was done under

The full text is [RULES.md](RULES.md). In short:

- **Quality gate:** wikitext-2 perplexity (`llama-perplexity`, test set, context 512, all chunks,
  `tools/ppl.sh`). Hard limit +20 % over the model's F16 weights (9.0438, so ≤ 10.853). The served file's
  stock PPL is 9.6051.
- **One approved numerics budget for speed:** router-weight expert skip, +2 % over the served file
  (≤ 9.797), final for the campaign. The served config spends +1.91 % (9.7886, *host*; perplexity is
  machine-independent). That covers skip 117 and the two-stage head together.
- **Everything else is bit-identical:** `llama-perplexity` per-chunk values equal at `-ub 1`, `-ub 32`
  and `-b 512`, and every new knob is off-switchable.
- **The served file is frozen.** Speed comes from the runtime only. Nothing is asked of the user's phone.
- **Measured on the phone, predicted in advance.** Every speed number here is measured on the device by
  its owner through the app. The prediction and the pass rule of each round are written into
  [reports/03_final.md](reports/03_final.md) before the run, and scored as written, misses included.
- Labels: *device* = measured on the phone; *host* = measured on a laptop (Apple M4), relative or
  machine-independent only; *estimated* = arithmetic, not a measurement.

## What got there

The levers that survived, in the served config. Effects are *device* in-run A/Bs in ms per token unless
labelled; minus is faster.

| lever | output | device effect | log |
|---|---|---|---|
| bit-identical runtime stack: sync-skipping around small ops, no-barrier expert op, 4-row Q3_K kernel with independent chains, SSM state in place, expert prefetch | identical | steady token 91.0 → 63.9 ms vs stock settings (ratio 1.42); official 32 9.4 → 12.0 | R25 |
| expert skip at threshold 117 (experts with a small normalised router weight are not computed) | PPL +1.89 % alone (9.7869, *host*) | at threshold 118: token −7.90 ± 2.12, `MUL_MAT_ID` −8.28 ± 1.14 at 2,112 MHz | R25, R31 |
| two-stage output head: a runtime Q4_0 copy screens all rows, the top 2,048 are recomputed exactly | PPL +0.01 % alone (9.6061, *host*); with skip 117: 9.7886 | head op −1.70 ± 0.13 (on skip 118); −1.60 ± 0.14 in the 118 → 117 + head swap | R31 |
| sampler fast path: the repetition penalty visits only the ≤ 64 penalised tokens | identical | outside-graph time 1.90 → 1.13 ms per token (at 1,555 MHz) | R33 |
| in-register Q3_K expert kernel | identical (greedy text identical on the device, four times) | compiled default since R35. **Falsified** under the pre-registered pooled rule in R38: `MUL_MAT_ID` +1.84 ± 4.22 on fault-heavy days; the verdict stands. A new pre-registered experiment **passed** in R47: −3.530 ± 0.117, measured on throttled clocks (1,670 and 1,555 MHz); ≈ −2.67 at 2,112 MHz by clock scaling (*estimated*) | R35, R38, R47 |
| fused expert op: gate, up and SwiGLU in one op, two chunks per claim | identical | `MUL_MAT_ID` + `GLU` −0.802 ± 0.100; compiled default from the r16 build | R47, R48 |

## What did not work

Each was measured and closed; numbers are in the log.

- Draft-model speculation and batched verification: 0.48-0.76x the plain rate on the device; the SD888
  CPU is compute-bound at any batch above one (R20). n-gram, EAGLE-3 and MTP drafting lost on the host.
- Early exit: the cheapest layer to drop already costs +2.5 % perplexity (*host*, R21).
- 4 compute threads, and the expert-guess helper on an A55 (R18).
- Expert skip above threshold 118: 119 fails the +2 % budget (9.8087, *host*, R25).
- Anonymous-memory hot-expert cache and Q3_K software prefetch (R23).
- Finer dense mat-mul chunks (R33).
- Static Q4_0 copy of the 4 most-routed layers' experts: over budget, no device effect (R32, R33).
- Q2_K / Q3_K screens for the two-stage head: compute-bound, slower than the stock head (R29).
- Touching expert pages ahead of the kernel (`MMID_TOUCH`): parked (R41).
- Dense structural numerics under a separate +5 % budget: router cast to F16 changed the device's greedy
  text (R47), low-rank head screen and shared-expert pruning failed perplexity (R43).
- OpenCL on the Adreno 660: 0.7-2.9 tok/s, always below the CPU (*device*, see "Didn't" in
  [reports/03_final.md](reports/03_final.md)).

## How to reproduce, and where things are

**Host checks, no phone needed.** `tools/reproduce.sh` builds llama.cpp at the pinned base with the
patches, downloads the model (sha256-checked) and wikitext-2, and checks two claims: bit-identity of the
runtime knobs (per-chunk PPL to the last digit) and the perplexity gate for stock vs
`GGML_EXPERT_SKIP=118`. `--quick` screens it. Verified 2026-11-05 from a clean clone on an Apple M4:
stock 9.6051, skip 118 9.7952 (`reports/results.csv`, R27 reproduce row). It checks skip 118, not the
served 117 + two-stage head pair.

**Android CPU module.** Apply the patches to the commit in `patches/llama.cpp/BASE`, then build with the
NDK:

```sh
git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp
git checkout $(cat ../vollie/patches/llama.cpp/BASE)
for p in 0001-cpu-runtime 0002-granite-rollback 0003-sampler-bypass; do git apply ../vollie/patches/llama.cpp/$p.patch; done
../vollie/android/build_cpu.sh . out/        # libggml-cpu-android_armv8.2_2.so
```

Knobs are environment variables ([docs/runtime-knobs.md](docs/runtime-knobs.md)); the app sets them
on the phone, and `GGML_EXPERT_SKIP` stays off unless set. Example:

```sh
GGML_TINY_OP=262144 GGML_EXPERT_IO=5 llama-cli -m model.gguf -t 3 ...
```

| path | what |
|---|---|
| [RULES.md](RULES.md) | the conditions every change must meet |
| [reports/03_final.md](reports/03_final.md) | the R-log, R1 to R49, with pre-registrations and verdicts |
| [reports/results.csv](reports/results.csv) | one row per result, wins and losses |
| [reports/device/](reports/device/) | raw device JSON of every research-kit round (r1 to r16) |
| [reports/04_campaign.md](reports/04_campaign.md), [reports/05_compute_less.md](reports/05_compute_less.md), [reports/memos/](reports/memos/) | campaign plan, compute-less memo, research memos |
| [patches/llama.cpp/](patches/llama.cpp/) | CPU-module patch, Granite rollback patch, sampler patch, pinned `BASE` |
| [patches/llamadart/](patches/llamadart/) | Dart-binding patch: change threads without reload, extra-buffer switch |
| [tools/](tools/) | `reproduce.sh`, `ppl.sh` (quality gate), `research_parse.py` (device JSON parser; `--official` for round 16), `research_kit.md` (the kit card), host benchmarks, `checks/` |
| [android/](android/), [runtime/dart/](runtime/dart/) | build the module for Android; app-side helpers |
| [docs/](docs/), [models/](models/) | findings, runtime knobs, tuning; how the served GGUF was made |

## Honest limits

- **Cold start.** The first run after memory pressure printed 12.3-18.5, not 20 (R49). The loss is page
  residency at the start of a run. Fixing it in the runtime is open.
- **One device, one model.** Everything here is one Snapdragon 888 phone and one Granite file. The goal
  includes that the method generalizes to other models and devices; that is the next step and is not
  shown yet.
- **The metric.** The printed number is llamadart's `decodeTokensPerSec`. Its time includes one final
  decode whose logits are never used, ≈ 3 % of the official number (*estimated*, R48). It was left in, so
  the benchmark's definition did not change.
- **Not scored.** Round 16's second line, which was to split the gap between ramp and faults, was
  unscorable (R49): its reference story ran throttled.
- **The app in the store lags.** The 20.51 is the research build 2.2.21+44. The production app has not
  been updated with it; that is the owner's call.

---

The table below is the round-5 comparison (2026-11-04), kept unchanged from earlier versions of this page.

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

## License

MIT (see [LICENSE](LICENSE)), like llama.cpp and llamadart, which the patches apply to.
