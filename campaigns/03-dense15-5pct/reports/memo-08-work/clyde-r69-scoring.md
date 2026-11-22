# memo-08 work file, Clyde: R69 scoring sheet (written 2026-11-22, before any R69 data)

The rules are those of `prereg/R69-probe-read-rate-capacity-gpu-device.md` (commit `458102c`) and
`prereg/R69-amendment-1.md`. This sheet only says how each parser output feeds them. It changes no rule, band or
threshold. Where the two prereg files leave a mechanical choice open, the choice is written here, before the data,
and marked "reading".
- Subjects: Yi-1.5-6B-Chat, Llama-2-7b-chat, DeepSeek-LLM-7B-chat (memo-06 §6, `e582606`).
- Goal: 12 tok/s on each.

**Tools.** Run both, on the owner's JSON in `reports/device/`:

```sh
python3 -I tools/c3_probe_parse.py <json>                                                 # tables (*device*)
python3 -I campaigns/03-dense15-5pct/reports/memo-08-work/clyde_r69_score.py <json>       # bands, rules A to D, lines, voids
```

The scorer has a self-test (`--selftest`, passes). It imports the parser's `parse()`. If the sheet and the scorer
disagree, the sheet wins, and the scorer is corrected with a dated note.

## 1. Void rules (R69 §4), applied first

| void | parser / JSON input | action |
|---|---|---|
| APK hash differs | not in the JSON; the owner checks the downloaded APK against `r29/SHA256SUMS` (`02736b56…`) | the whole round is void |
| power saving on | `power_start`, `power_end` and each block's `power_before` (raw values printed by the scorer) | every block run in that state is void |
| charger plugged | `plugged_start`; each block's `plugged_before` | that block is void |
| process killed during a block | the block has no `c3-probe: <block> end` line (the parser's `meta.<block>_s` is missing) | that block is void. A block that prints `error=` and its end line is "no number", not killed |
| a worker left its cpu | the cell is marked `*` (cpu_ok < n) | reading: the cell is excluded from every rule input. If the r cell is `*`, r falls back to the 3 A78 cell (R69 §3's own fallback), and this is noted |

Recorded, not voids: `deep_sleeping_list`, `gate.timeout` and the hdr lines (r29 has no hdr; 0002 is not in r29).

## 2. Inputs to the rules

| symbol | parser output | note |
|---|---|---|
| r | P2 table, `q4x4 gemv` at cores `7,4-5` (X1 + 2 A78) | the cell exists in r29 (`sets` in 0001). Fallback `4-6` |
| read, 3 A78 | P2 table, `read` at `4-6` | |
| (X1 + 3 A78) ÷ 3 A78 | q4x4 at `4-7` ÷ q4x4 at `4-6` | |
| bus ratio (Rule D) | q4x4 at `0-6` ÷ q4x4 at `4-6` | reading: q4x4, the kernel r uses (§4 D does not name one) |
| c_b | `0.87 × 0.5625 × (1/ns_x1 + 2/ns_a78)`, from P2hot `ns_per_w` | GB/s |
| P2hot bands | R69 §3 states them in **cycles** per weight; the line prints ns | reading: cycles = ns × the cell's own clock in GHz, the mean of the before and after values of `mhz=` (cpu0 for the A55, cpu4 for the A78, cpu7 for the X1). Check: 0.105 cycles at 2.419 GHz with 0.078 at 2.841 GHz gives c_b 40.4, the pre-registered centre |
| k (Rule A) | P2 line `cores 4: col / stock per touched weight` | the pinned-A78 row (§1 says "on a pinned A78"). The `4-6` row is reported, not scored |
| k, skip list | the same line, `skip / stock` | prediction only |
| X1 gain at 2 / 4 streams | `streams 2` ÷ `streams 1`, `streams 4` ÷ `streams 1` | |
| X2 read / q4 | `GPU alone, kern read` / `kern q4` | |
| S | `together, gpu_kern read … sum` ÷ P2 q4x4 at `4-6` | the CPU lane of `x1gpu` is `4-6` |
| P1 held | not printed by the parser (it prints all-pass and last-5 medians) | reading: "passes 6 to 20" = `pass=` index 5 to 19 (the probe counts from 0). With the 45 s cap per arm, use the passes from index 5 that exist. A step with none, or an `error=mmap` line, is **not measured**, not "not held". The scorer computes this |
| C | the largest step held with every smaller step held | if a larger step is not measured, C is a lower bound and is said to be one |

## 3. Build gap found before the data (not a change to R69)

R69 §1 lists P2col "at kept fraction 1.0, 0.65 and 0.60, hot and cold". Kit r29 does not run all of these.
`0001-probe.patch`, block `p2` (d), runs only:
- keep 0.60;
- cold;
- cores `4` and `4-6`;
- stock as the 1.0 comparator.

There is no 0.65 row and no hot row. Consequences, fixed now:
- **Rule A at d 0.60** (amendment item 1, reference density) is scored.
- **Rule A at d 0.65** is "not scored from R69: no row in the build". It is a gap between the build and the
  pre-registration, and is written as such in the result.
- **P2col prediction** (k 2.6, 2.0 to 3.3, "at 0.60 to 0.65 kept") is scored on the 0.60 row.

## 4. Rules as they apply to the three subjects

| rule | inputs | outcome lines (scorer) |
|---|---|---|
| A, reference d 0.60 | R, k, d 0.60, b 0.644 | pass if k ≤ k\*(r) and gain ≥ 1.15; kill if k > k\*(r); otherwise open |
| A, **Yi** (amendment item 2, provisional until its freeze) | Yi's *host* frontier is 42 % uniform (d 0.58, row 0.60) or 40 % (d 0.60) | both verdicts are printed. The one at the full-set level chosen at the freeze is written into Yi's freeze record before its first device round |
| A, **Llama-2, DeepSeek** | the k,v-dense rule: 40 % on every matrix but `attn_k`, `attn_v`. Body kept fraction d_s = 0.166 + 0.834 × 0.60 = 0.666 | under amendment item 2, d_s > 0.625 needs the 0.65 row. There is none (§3), so: **not scored from R69; needs a device row** (R72 L1/L4). The scorer prints a mixed-rule price on the 0.60 row as *estimated*, with no verdict |
| B | r ÷ c_b | kill ≥ 0.71; reopen as host item ≤ 0.61. The host item's inner kill is "match rate under 0.4 on **any** of Yi, Llama-2, DeepSeek" |
| C | steps, C | 4.6 held: pass (Y1 first). 3.4 not held in either arm: kill for Q4_0 bodies ≥ 6 B, and each subject is frozen at a file with resident estimate ≤ C. Otherwise, in between: served files priced against C, reserves Mistral and Qwen2.5 |
| D | S, X2 read, X1 gains, bus ratio | as R69 §4 D. The `-cl-opt-disable` retry is not in the r29 build (`BuildProgram` with empty options), so "after one retry" is scored as no retry made: a read under 10 GB/s kills the lane |
| goal (amendment item 5) | | "12 tok/s is not reachable" for every subject whose priced token time at the measured r and c_b exceeds 83.3 ms, with r, c_b and C |

**Rule C reading for the subjects (descriptive, from amendment item 4).** The 3.4 GB step is below Yi's Q4_0 file
(3.48 GB, *host*) plus KV and buffers (about 0.15 GB, *estimated*). The 3.4 step held therefore does not show that
any subject fits. On the memo-06 §5 freeze rule, Yi needs C ≥ about 3.63 GB, which is the 4.2 step; Llama-2 (3.83 GB)
and DeepSeek (4.00 GB) need 4.2 as well. This is the rule's own arithmetic, reported beside the verdict. It is not a
new kill line.

**Known limit of P1** (Bonnie, memo-08-work/bonnie-path.md §C; not a change to R69). P1 measures how much
*file-backed mapped* memory stays resident: scratch files with `map=r` and `map=x`. With repack on, llama.cpp puts
the body into the CPU buffer type (`repack.cpp`, `ggml_backend_cpu_buffer_type()`), which is anonymous memory. Under
Android's low-memory killer and zram, the capacity for anonymous memory may differ from the file-backed figure in
either direction.
- R69's C is scored as written. In the result it is stated as "file-backed capacity".
- It is used as the resident limit only for a body that is mapped from a file (Y1, the `--no-repack` residency test
  of D1).
- An anonymous-arena arm at 3.4 and 4.2 GB, with the same held rule, is a candidate rider for R72 (bonnie-path D3).

## 5. Byte and compute lines (amendment item 6; *estimated* from *device* inputs)

B12 = r × (83.3 − O) ms and W12 = c_b ÷ 0.5625 × (83.3 − O) ms, at O 6, 8 and 12 ms, printed by the scorer.
The memo-07 convention (printed = 0.95 × steady, so 79.2 ms) is reported beside it as descriptive only. The
amendment's 83.3 ms is the one written before the data.

## 6. Predictions

Every §3 row is marked hit or miss against its 80 % band by the scorer:
- r, read, the ratios;
- P2hot in cycles;
- c_b, k col, k skip;
- the X1 gains, X2 read and q4, S;
- C.

The probabilities in §3 (P(3.4 held) and so on) and the X2 "prints all lines" row are scored by hand from the P1
table and the X2 lines. Clocks: cells with cpu4 ≥ 2,419 and cpu7 ≥ 2,841 MHz at both readings, counted over every
line with `mhz=`; a hit if at least 80 %.
