# Memo 08: prediction bands for the first OLMoE device baselines (R52, R53), instrument mapping, Granite reading

Agent model: claude-opus-5-5

Date 2026-11-14. Author: Bonnie (research). **No measurement was made for this memo.** Every number is
*estimated* unless it carries *device* (campaign 1, with its R-number), *host*, or *source* (read from code at a
named path, which is not a measurement). I read device JSON (`reports/device/r8_2026-11-06.json`,
`r16_2026-11-08.json`) with my own short code; those numbers are labelled *device, my reading*. No job was run apart
from that and a small arithmetic script. The laptop was under load (load average 41) during this session, so I
deliberately ran nothing that could disturb a host sweep.

Inputs: STATUS.md, knowledge.md, memo-03, memo-05, R51 amendment 2, the model cards, `reports/01_vollie_vs_llamacpp.md`,
`reports/04_campaign.md`, `models/granite-4.0-h-tiny-d6-e6.md`, `docs/runtime-knobs.md`, `docs/levers.md`,
`docs/round5-comparison.md`, `tools/research_kit.md`, R1-R49 in `reports/03_final.md`, the app source of the
campaign-1 app (the owner's local checkout, version 2.3.0+45), and `lab/llama.cpp-base` (`620d2225c`) for
`llama-bench` and `repack.cpp`.

## 0. Short version

| | ms/token, fault-free, 2,112 MHz, 3 threads (low / central / high) | tok/s | basis |
|---|---|---|---|
| **R52** stock `620d2225c`, repack off, CLI greedy at positions 155-187 | **65.6 / 71.6 / 81.5** (corner sum 59.5-91.4) | 15.3 / 14.0 / 12.3 | §2 |
| R52 at 4 threads | 61 / 77 / 97 (weak) | 16.4 / 13.0 / 10.3 | §2.3 |
| **R53** bit-identical stack (six levers, §3) | **45.8 / 49.7 / 54.0** (corner 41.4-60.1) | 21.8 / 20.1 / 18.5 | §3 |
| R53 + skip 13.39 % + untied head2 + wait crumbs, official-equivalent | 43.8 (37.5-50.0) | printed about 21.8 (19.1-25.4) | §5 |

- **The repack finding changes memo-03.** In campaign 1 the served Granite's dense Q4_0 and Q6_K head ran **repacked**
  on the phone (§1). Memo-03 took those repacked rates as the OLMoE dense and head rates. On the phone OLMoE runs them
  un-repacked. That adds about **+2.7 ms** (+0.4 to +6.6) to every OLMoE band.
- **A second correction that is not about repack.** Memo-03 took the expert rate after the stack from one R49 fit
  (0.083 ms/MB). I rebuilt it from the measured lever deltas starting at the clean R31 vanilla row. The result is
  0.091 (0.080-0.098), which adds about **+3 ms**. Attention at 16 full-MHA layers, scaled from Granite's measured
  FA slope, adds +0.6.
- **Net result.** Memo-03's E0 of 43.5 ms becomes about 50 ms. With the skip R51 found inside the budget, OLMoE
  ends at about 44 ms against the 38.0 line. **At the central value, 25 tok/s is not reached with the priced levers.**
  It is reachable only if R53 comes in at 42 ms or less (§5).
- **Granite: proposed "no"** (§7). No single unpriced lever is worth about 5 ms inside 9.6879. The crumbs add up to
  about 3 ms, and two of them need rulings.

## 1. What campaign 1 actually ran: dense and head repacked

**Finding.** From R1 (app 2.1.1+13) through R49 (2.2.21+44), the phone loaded the served Granite file
`d6-e6` **with `use_extra_bufts = true`**. So its 420 MB of dense Q4_0 ran from a repacked `q4_0_4x4` anon copy,
and its tied Q6_K head ran from a `q6_K_8x4` copy. That is about 0.55-0.7 GB of anon memory. Only the Q3_K experts
were read from the mmap (they have no repack path).

Evidence, originals first:

1. *source*, app `lib/assistant.dart:461-464`: `useExtraBuffers: device.canCopyWeights(fileLength ==
   ModelStore.expectedBytes ? ModelStore.copyBytes : fileLength)`. `lib/model_store.dart:12,19`: expectedBytes
   = 3,161,269,888 (the d6-e6 file) and copyBytes = 700,000,000 ("only the Q4_0 dense tensors and Q6_K head").
   `lib/device.dart:107`: `totalRam − copyBytes ≥ 5,000,000,000`. On this phone (about 7.3 GB reported) that is
   7.3 − 0.7 ≥ 5, so **true**.
2. R-log R1 row: "D1-e5 file; repack on for 8 GB (`ModelStore.copyBytes` → `canCopyWeights`)". `reports/04_campaign.md:29`
   (R16): "dense weights repacked (`useExtraBuffers` is true: 8 GB − 0.7 GB copy ≥ 5 GB)". Line 46: dense read from
   the "repacked copy in app memory (anon, 0.7 GB)". The model card: Q4_0 dense is "the only part that gets llama.cpp's
   repacked copy on ARM (~0.7 GB), which fits an 8 GB phone".
3. *device* R12: the app's own RssAnon was 674 MB (the R11 field is "app memory incl. repacked copy"). Without the copy,
   anon would be compute buffer, about 18 MB of KV, 58 MB of state and the Dart heap. That is well under 674 MB
   (*estimated*).

**Against it (campaign-1 documents that I read as wrong, not edited):**

- the R25 vanilla footnote (`reports/03_final.md:369`) and `tools/research_kit.md:52-53`: "weights run from the
  mmap because `canCopyWeights` turns repacked copies off";
- `tools/checks/kit_runs.cpp:6,40`: "the phone's path (R9: no repack)";
- `reports/05_compute_less.md:1173`: "no-repack (the phone's path)";
- `docs/round5-comparison.md:12,52`.

`reports/01_vollie_vs_llamacpp.md` §1-§4 was right **for the all-Q4_0 file before R1** (the rule then used the file
size); its own last section says "repack back on for 8 GB phones (0.7 GB copy)". The production host replay
(`lab/logs/prod/prod_text.sh`) matched the device's 32-token text with `KIT_NOREPACK=1`. No repack-on replay was
run, so that match does not discriminate.

**Caveat.** I read the app tree at 2.3.0+45, not the r16 APK. `model_store.dart` is dated 2026-11-01 (R1), so the
rule predates every kit round. One logcat line at load (`CPU_REPACK model buffer size`) on the phone would settle it,
and so would the CLI pair in §2.4.

**Consequences.**

- (a) Campaign 1's 0.0398-0.0411 ms/MB "dense byte rate" and its 18.9-22.8 GB/s head rates are **repacked**
  rates.
- (b) Campaign 1's host identity references (`-nr`, `KIT_NOREPACK`) ran a different dense kernel from the phone.
- (c) **`granite-Mi` (3,161,269,344 bytes) is 544 bytes smaller than `expectedBytes`.** The unmodified app therefore
  loads it with the file-size rule: 7.3 − 3.16 < 5, so **repack off**. It would not run campaign 1's dense path.
  The app rule must be made per file before any Granite app round. In the CLI, use `--repack 1`.
- (d) Memo-05's conclusion for the two OLMoE files holds under the current rule. Neither file equals
  `expectedBytes`, so the rule uses the file size and gives false for both. Memo-05 cited the pre-R1 form of the rule.
- (e) For `olmoe-Mi` a per-file rule would give copyBytes = 235.5 MB (memo-05 §2), which fits easily. Repack-on for its
  attention and head is therefore an available app-side lever (§5.3). On the phone the repack-off path is a
  consequence of a hard-coded size, not of memory.

Base `620d2225c` repack types for arm64 dotprod (*source*, `repack.cpp:4974-5070`): Q4_0 4x4, Q4_K 8x4, Q5_K 8x4,
Q6_K 8x4, IQ4_NL. There is **no Q3_K repack**. A web search found the upstream Q6_K dotprod repack work (#19360) and
none for Q3_K.

## 2. R52: stock `620d2225c`, repack off, 3 threads on the big cores

### 2.1 Anchors (campaign 1, *device*, d6-e6, stock knobs, dense and head repacked)

- **R31 vanilla story, last windows** (r8 JSON, 2,112 MHz, 11.6-16 majflt/token, positions about 150-217, my
  reading): **79.0 / 84.9 / 80.2 ms/token**. The last window splits as:
  - `MUL_MAT_ID` 31.9 + 11.1 wait (243 MB, k = 6: **0.131 ms/MB stock Q3_K**, wait about 92 µs per op over 120 ops);
  - `MUL_MAT` 24.3 + 1.8 (562.5 MB);
  - `SSM_SCAN` 3.6;
  - `CPY` 2.7;
  - the other 1,398 nodes 4.8 ms, i.e. **3.45 µs per small node**;
  - nodes 1,963, barriers 1,926.
- **R25 vanilla, 91.0 ms**: the "before the first fix" number. It was a swapping day (109 majflt/token), so I use it
  as the fault-laden upper reference, not the anchor.
- **Bare barrier**: 0.17 / 0.30 / 2.0 µs at 2 / 3 / 4 threads (R18, `GGML_BARRIER_BENCH`). The per-node cost is op
  work and skew, not the barrier itself.
- **Repacked Q4_0x4, cache-hot, 1 thread on an A78**: 43.23 ns/row at K 1,536, 2,112 MHz (R37 `cpu-bench`). That is
  864 B/row, so 20 GB/s per thread.
- **Plain/repacked Q4_0 on the host**: 2.36× per row cache-hot (40.8 vs 17.3 ns, R30); 1.6-1.8× at op level (R30).
  Plain/repacked Q6_K head 1.73 vs 1.17 = 1.48× (R27 host amendment, `05_compute_less.md:1173`).
- **Granite FA on the device** (r8 vanilla, my reading): 127 / 172 / 215 / 261 / 307 µs per token at mean positions
  of about 65 / 97 / 129 / 161 / 193. That is **1.41 µs per position for 4 layers × 12 query heads** (29 ns per
  query-head position), plus about 35 µs fixed. The cost grows with position, so masked cells are not paid for.

### 2.2 Terms for `olmoe-Mi` (590 MB/token; 16 layers, 64 experts, k = 8, MHA 16 × 128)

| term | MB | rate or rule | low / central / high (ms) |
|---|---|---|---|
| experts, stock Q3_K `vec_dot`, generic `MUL_MAT_ID` | 346 | 0.120 / 0.132 / 0.150 ms/MB (R31 0.131 with faults; longer K 2,048 rows slightly favour the low side) | 41.5 / 45.7 / 51.9 |
| expert waits (48 ops) | | 92 µs/op fixed gives 4.4; proportional to op work (0.33 of compute) gives 15 | 4 / 8 / 15 |
| attention q/k/v/o, **plain** `vec_dot_q4_0_q8_0` | 151 | 0.041 / 0.050 / 0.065 ms/MB: per-thread plain is about 43 × (1.8 to 2.36) = 78-102 ns/row, i.e. 25-33 GB/s at 3 threads, against a 22-25 GB/s stream, so it sits at the ridge | 6.2 / 7.55 / 9.8 |
| dense waits (81 `MUL_MAT`) | | 6.8 µs/op (R31: 1.8 ms / 265) | 0.4 / 0.6 / 1.0 |
| head `output.weight`, **plain** Q6_K | 84.5 | 0.045 / 0.057 / 0.077 ms/MB. Floor: R31's repacked 0.044. Compute is at least 1.48× repacked, and repacked was clock-flat at 1,555 (R29), so plain is at most about 4.9 ms at 2,112 under max(); partial overlap goes higher | 3.8 / 4.8 / 6.5 |
| router F32 (16 × 524 KB) | 8.4 | about 25 µs/op (Granite 20 µs at 393 KB, r16) | 0.35 / 0.40 / 0.50 |
| flash attention, q8_0 KV, mean position 171 | 11.9 of KV | 256 query-head positions per position (5.33× Granite) × 29 ns × 1.125 (16 heads over 3 threads: 6/6/4) = 8.4 µs/position, plus 16 × 9 µs | 1.0 / 1.5 / 2.3 |
| small nodes | | about 725 nodes (about 870 − 145 big ops, counted from the `olmoe` graph structure, *estimated*) × 3.45 µs | 2.0 / 2.6 / 3.5 |
| outside the graph, CLI greedy (no app, no Dart) | | sampler over 50,304, logits copy, graph setup | 0.2 / 0.5 / 0.9 |
| **total** | | corner sums / root-sum-square | **59.5 / 71.6 / 91.4**; RSS **65.6-81.5** |

**R52 prediction (pre-registrable).** Fault-free steady ms/token of the 155-prompt / 32-token greedy CLI run:

- **central 71.6 ms (14.0 tok/s)**;
- likely band 65.6-81.5 (15.3-12.3 tok/s);
- outer band 59.5-91.4.

With an f16 KV cache (the CLI default unless `-ctk/-ctv q8_0` is given), attention is 0.9 / 1.4 / 2.3 and the total
moves by −0.1. **Pin `-fa on -ctk q8_0 -ctv q8_0` to match the app.** In `llama-bench` at this base, FA defaults to
`auto` and the cache to f16 (*source*, `tools/llama-bench/llama-bench.cpp:406`).

Memo-03's stock estimate was 77.3 (65.6-93.8). It moves down, because the expert waits and small nodes are now
anchored on the clean R31 row instead of R25, and up by +2.7 for the plain dense and head path.

### 2.3 Four threads (weak)

Campaign 1 found 4 threads slower than 3 on every build:

- R18 official 32: 3t 8.4 against 4t 7.7 tok/s, so ms × 1.09 (*device*);
- R8: 4t 3.9 against 3t 6.3 in the sync-bound era;
- the bare barrier at 4 threads is 2.0 µs against 0.30 (R18).

On the other side, the 4th big core would cut the compute-bound expert term by up to a quarter, if it were free of
Android's threads. **Band: T4/T3 = 0.85 / 1.08 / 1.35, i.e. 61 / 77 / 97 ms.** For R53 at 4 threads: 0.90 / 1.08 /
1.30 × R53. Campaign 1 closed 4 threads (knowledge §4). I would run it only as a falsifier of the sync model, not as
a lever.

### 2.4 One cheap companion run that calibrates the plain rate (recommended)

The plain-kernel rates in §2.2 rest on host ratios, because no device number exists for plain Q4_0 or Q6_K.
`llama-bench` on **`granite-Mi` with `--repack 0,1`** in the same adb session measures both rates directly on 420 MB
of Q4_0 plus a 126 MB Q6_K head. The repacked copy fits, as campaign 1 showed.

Prediction: T(repack 0) − T(repack 1) = +3.8 (dense) + 1.4 (head) = **+5.2 ms (RSS +1.5 to +9; corner 0 to +14)**.
The OLMoE dense-and-head penalty is then about 0.45 × the measured Granite difference (151/420 of the Q4_0 part and
84.5/126.4 of the head). This run also checks §1: repack-on Granite should reproduce R31's `MUL_MAT` rate.

## 3. R53: the bit-identical stack, lever by lever, in switch-on order

These are the c2 series ports of campaign 1's exact levers. Each one is off unless switched on and is identity-checked
on `olmoe` first. Each row is one ABBA, B = the lever on, on top of the rows above it. Δ is in ms at 2,112 MHz on
fault-free tokens; "low / central / high" give the smallest / central / largest saving. Repack stays off throughout.

| # | lever (c1 knob) | what it removes on OLMoE | Δ ms | steady after | basis |
|---|---|---|---|---|---|
| 0 | stock (R52) | | | **71.6** | §2 |
| 1 | tiny ops on thread 0 without syncs (`TINY_OP=262144`) | small-node cost 2.6 → 1.5 | −0.4 / **−1.1** / −2.3 | 70.5 | R18 −4.3 on Granite (one run, σ 3-4) × 725/1,398 nodes = −2.2 at most; thread-0 serial work remains |
| 2 | no barrier inside `MUL_MAT_ID`, dynamic 32-row chunks (`MMID=3`) | expert waits 8 → 1 | −3.0 / **−7.0** / −13.0 | 63.5 | R18: wait 16.5 → 6.0, −7.5 over 120 ops; R31 stack wait 0.66-1.6 ms / 120 ops (*device*); the band follows the stock-wait band |
| 3 | 4-row Q3_K kernel with shared activation loads, ILP and the faster dot (`Q3K_X4`, `Q3K_ILP`, `Q3K_VEC`) | expert compute 0.132 → 0.114 ms/MB | −3.0 / **−6.3** / −9.0 | 57.2 | R31 vanilla → control `MUL_MAT_ID` compute −4.2 to −4.7 on 243 MB (*device*, my reading), × 346/243 |
| 4 | in-register Q3_K kernel (`Q3K_KERN`) | 0.114 → 0.094 | −4.5 / **−6.8** / −9.0 | 50.4 | R35: in-situ slot 116-120 → 96-100 µs per 1.0138 MB (−0.0197 ms/MB); R47: −2.67 at about 168 slots (−0.0157 ms/MB) gives −5.4 |
| 5 | fused gate+up+SwiGLU and chunk2 (`MMID_FUSE`, `CHUNK2`) | 16 `GLU` nodes and barriers; activation reuse | −0.3 / **−1.0** / −1.8 | 49.4 | R47 −0.80 ± 0.10 on Granite; per op × 16/40 = −0.32; per MB × 346/170 = −1.6 |
| 6 | expert prefetch (`EXPERT_IO=1`) | nothing on resident tokens; thread-0 `madvise` costs time | **+0.3** (−1.0 to +1.0) | **49.7** | R25 priced "madvise off the compute threads" at −0.4 to −1.0, so the call costs that much; it gains only when tokens fault. Score it with majflt matched |

Not in the CLI: the sampler bypass (app-only, about −0.1 to −0.3 on 50,304 entries). Not applicable: `SSM_INPLACE`,
`CONV_RING`, the shared-expert items (memo-02).

**R53 prediction: 49.7 ms (RSS 45.8-54.0; corner 41.4-60.1), 20.1 tok/s CLI.** After the stack the expert term is
31.6 ms (27.7-33.9).

- The ladder from R31 gives 0.094 ms/MB.
- The R49 in-situ fit gives 0.082-0.087. It is the only direct measurement of the final state, but on Granite's shapes
  and at k̄ 4.2.
- About 8 µs per slot between the two is unattributed. The low end of the band is the R49 fit.

### 3.1 Where the +6.5 ms against memo-03's E0 (43.5, app terms) comes from

| item | Δ ms |
|---|---|
| plain attention Q4_0 instead of repacked (repack finding) | +1.45 |
| plain Q6_K head instead of repacked (repack finding) | +1.27 |
| expert rate after the stack: ladder from R31 instead of the R49 fit alone | +3.0 |
| expert waits (1.0 against 0.4), FA (1.5 against 0.9), dense waits (0.60 against 0.53) | +1.27 |
| small nodes (1.5 against 2.12) and app outside (0.8 against 0.9) | −0.72 |
| prefetch cost | +0.3 |

**The repack finding alone accounts for +2.7 ms (+0.4 to +6.6).**

## 4. Instrument mapping

Let T_g be the in-graph ms per decode token at the official context. Each instrument adds its own outside time.

| instrument | per-token time | mapping |
|---|---|---|
| `llama-bench -p 0 -n 64` (tg64, depth 0) | positions 0-63 (mean 32); random tokens, no sampler; mean of 5 reps after a warm-up | T_bench0 = T_g − 1.3 (0.8-1.9, FA at mean position 32 against 171) + 0.2 (0.1-0.4, outside) + 0.15 (first-token ramp spread over 64). **Use `-d 155`:** then T_bench155 = T_g + 0.13 (mean position 187) + 0.35 ≈ T_g + 0.5 (±0.3) |
| fixed prompt, 155 → 32 greedy CLI (Tom's binary) | positions 155-186; the CLI sampler | T_cli = T_g + 0.5 (0.2-0.9). Take eval ms / number of decode calls, whether or not the binary decodes the 32nd token |
| app `decodeTokensPerSec` | (completion − 1) × 1000 / (duration − TTFT); `decode_ms` holds **32** decode graphs (R48 am. 1) | **printed = 0.995 × 31,000 / (32 · (T_g + t_app) + R)**, with t_app = 0.8 (0.6-1.05, Granite's 1.05 with bypass, sampler share halved for 50,304 entries) and the ramp R = 0 / 12 / 25 ms (r16, memo-03 §1.4) |

**CLI to official, before the official run:**

> printed_pred = 0.995 × 31,000 / (32 · (T_cli + 0.3) + 12)

- Uncertainty: ±0.4 ms/token on t_app − t_cli and ±12 ms on R. Add 0 to +0.15 ms/token for llamadart's per-call
  threadpool (it never attaches one; no difference on the host, R-log §9; device unmeasured).
- Together that is **±1.5 % in tok/s** on a resident, fault-free day (*estimated*).
- It does **not** cover the app's memory: the app holds 0.5-0.7 GB more anon than a CLI process, which leaves less page
  cache for OLMoE's 2.77 GB of experts. A CLI run that is fault-free does not prove the app run is. Log majflt
  per token on both.
- For `llama-bench` replace 0.3 with −0.2 (at `-d 155`) or with +1.0 (at `-d 0`).

Worked values (central R = 12): T_cli 71.6 → printed 13.3; 49.7 → 19.1; 40 → 23.7; 37.7 → 25.1.

## 5. Falsification and the decision table

### 5.1 What R52 would falsify

- **The whole model** is falsified if the fault-free CLI steady is **below 59 or above 92 ms** (outside the corner
  band). Inside 65.6-81.5 it is a hit. Between those limits it counts as a miss with the direction recorded.
- **R52 > 81.5.** The experts (stock rate above 0.15 ms/MB) or their waits (above 15 ms) must be larger. They are
  about 75 % of the token. The plain dense and head rates can add at most +3.9 ms beyond central, so they cannot explain
  it alone. Which one it is decides R53:
  - excess in the waits: `MMID=3` removes it, and R53 moves by about 0.15 × the excess;
  - excess in the kernel: the kernel levers scale it by about 0.69, and R53 moves by about 0.7 × the excess.
- **R52 < 65.6.** The Q3_K stock rate is below 0.12 on K 2,048 rows, or the stock waits are below 4 ms. Levers 2-4 then
  have less to remove: R53 moves by about 0.4-0.7 × the shortfall, not one for one.
- **Splitting the terms without a profiler** (the base build has none):
  1. `llama-bench -d 0` against `-d 155`: the attention slope predicts **+1.3 ms (0.8-1.9)**. Outside that range the
     FA term is wrong.
  2. 4 against 3 threads: the sync and wait share. Outside 0.85-1.35 the wait model is wrong.
  3. The §2.4 Granite repack pair: plain against repacked.
  4. If the phone falls to 1,555 MHz on some run, the change separates compute-bound terms (experts, ×1.36) from
     byte-bound ones (×1.0-1.09, R27).
  5. The c2 build with every knob off and `GGML_CPU_PROF=1`, if Tom ports the profiler: the full op split. That is
     the best single R52 add-on.

### 5.2 Package after R53 and the table

Every package item is *estimated*:

- **Expert skip `MASS=900`, f = 13.39 %** (R51, *host*, repack-on path; R51 amendment 2 re-measures it on `-nr`).
  Δ = 0.925 (0.89-0.96) × f × T_exp(R53). With T_exp = 31.6 (27.7-33.9) that is **0.292 ms per 1 %** (0.247-0.325),
  so **−3.9 (−3.3 to −4.4)**.
  - Use the measured `MUL_MAT_ID` of R53 when it exists.
  - f on the official prompt can differ from wikitext's: log k̄.
- **Untied two-stage head.** With repack off the head is plain, so head2 works as it stands (memo-02 §6, line 128),
  without option A. The screen is a runtime Q4_0x4 copy of 58.0 MB plus 2,048 exact Q6_K rows (3.4 MB), about
  2.7-2.9 ms against 4.8: **−2.0 (−0.9 to −3.8)**. PPL charge 0 to +0.2 % (central +0.05), unmeasured. It fits beside
  `MASS=900` only if the sum stays ≤ 12.3003; that margin is 0.197 on R51's number.
- **Wait crumbs: −0.25 (−0.1 to −0.4).**
- **Package total: −6.2 (−4.3 to −8.6).**

Official-equivalent T_off = T_R53,cli + 0.3. **25 printed needs T_off ≤ 37.97 / 38.38 / 38.75 at R = 25 / 12 / 0.**
I use 38.0.

| R53 steady (CLI, ms) | after the package: central (weak to strong package) | 25 reachable? |
|---|---|---|
| ≤ 40 | ≤ 34.1 (≤ 36.0 to ≤ 31.7) | **yes**, even with the weak package |
| 40 to 42 | 34.1-36.1 (36.0-38.0 to 31.7-33.7) | **yes**; at 42 the weak package lands exactly on the line |
| 42 to 44 | 36.1-38.1 (38.0-40.0 to 33.7-35.7) | **probable at the central package** (head2 ≥ −2.0 and f on the official prompt ≥ 13 %); no with the weak package |
| > 44 | > 38.1 | **no with this package.** Unpriced extras, all needing rulings (§5.3), could add up to −2.9 ms-equivalent. Above about 46.8 even those do not reach 38. **Stop-rule line (short ≥ 6 ms after the package): R53 ≥ 49.9**, or ≥ 52.8 with every extra |

**My R53 central (49.7) sits at the stop-rule line.** Read plainly: at the central estimate, OLMoE-Mi reaches about
**43.8 ms ≈ 21.8 printed tok/s** with every priced lever inside the budget. Reaching 25 needs R53 at 42 ms or less,
which is the fast end of the band (corner low 41.4). The device baseline decides it; this memo does not.

### 5.3 Extras outside the package (each needs a ruling; priced, not proposed)

- **Repack-on for `olmoe-Mi`'s attention** (per-file app rule, 235.5 MB anon, or a runtime Q4_0x4 copy like head2's
  screen).
  - Value: −1.4 (−0.2 to −3.6).
  - It is a numerics change against the phone's `-nr` path. However, the R50 stock reference itself was measured on the
    repack-on path, so for the lead it is a rule question, not a quality question. Memo-05 §5 raised it.
- **Skipping the unused 32nd decode** (R48 am. 1): printed = 31,000/(31 · T + R). The 25 line moves to T ≤ 39.6
  (R = 12), worth about **−1.2 ms-equivalent**. It is bit-exact, but it changes what the benchmark's time contains.
  Owner decision 3 ("metric unchanged") has to be read on it.
- **Router F32 → F16**: −0.26 (closed in campaign 1, R47).

## 6. The publisher Q4_0 file with repack off

- **Kernel.** Experts in Q4_0 run the generic `MUL_MAT_ID` with `vec_dot_q4_0_q8_0` (memo-05 §1). They read 453 MB per
  token.
- **Price.** In situ:
  - the byte floor is 0.041;
  - R35's repacked Q4_0x4 slot was 62-68 µs per 1.35 MB, i.e. 0.046-0.050 ms/MB (*device*-derived);
  - plain adds a compute share at the ridge (§2.2).
  Rate **0.050 / 0.058 / 0.075 ms/MB → 22.7 / 26.3 / 34.0 ms**, against the mix's Q3_K on the full stack at
  27.7 / 31.6 / 33.9.
- **Compute only, the Q4_0 file is about 5 ms faster: Δ −5.0 / −5.3 / +0.1.** Levers 3-4 do not apply to it; levers 1,
  2 and 5 do, if the fused path is type-generic.
- **Residency decides it, not the kernel.**
  - Its experts are 3.62 GB against about 2.1-2.4 GB of page cache (less inside the app).
  - About 1.2-1.5 GB cannot be resident. Each token reads 1/8 of the experts, so roughly 35-40 % of each token's
    expert slices would come from flash (*estimated*, uniform-routing assumption).
  - Each 1 % of a token's expert bytes that faults costs about 2-4 ms at 1-2 GB/s from flash.
  - Campaign 1's 3.1 GB D1 experts gave 354-1,181 majflt/token and 137-163 ms tokens (R9, *device*).
- **Verdict.** **Worse than the Q3_K mix on this phone unless it proves resident.** That is unlikely. The mix's
  2.77 GB is itself borderline.
- **The settling measurement.** In the same session, right after the `olmoe-Mi` run on the same c2 build with levers
  1, 2 and 5 on: a warm, gated `llama-bench -p 0 -n 64 -d 155 --repack 0 -t 3` of the Q4_0 file, with majflt per token
  read from `/proc/<pid>/stat` before and after.
  - Better only if steady ≤ `olmoe-Mi` (same levers) − 3 ms **and** majflt ≤ 10 per token on the second and later
    reps.
  - Its own frontier and its budget (11.3784 × 1.05 = 11.947) come after.

## 7. Granite: is there an unpriced lever worth about 5 ms inside 9.6879?

**Proposed answer: no.** Memo-03's Granite package is 43.2 ms (40.6-46.4): +5.2 short of 38.0.

- **That number stays valid only if `granite-Mi` runs with repack on.** The unmodified app loads it with repack off
  (§1c). That would cost about +5.2 ms more (§2.4). The per-file copy rule is a precondition, not a gain.
- **What is left that is unpriced or unused (all *estimated*):**

| item | Δ (ms or equivalent) | status |
|---|---|---|
| skip the unused 32nd decode | −1.35 equivalent (3.1 % of 43.2) | needs the metric ruling (§5.3); applies to every model |
| skip selection by cumulative mass instead of a weight threshold | −0.5 to −1.0 (+3 to +5 % share at equal PPL, a guess; R51's SUM/MASS edge on OLMoE was not a clean win) | needs a host frontier on `granite-Mi` |
| router F32 → F16 | −0.49 (*device* R47) | closed in campaign 1 on text divergence; reopening is the lead's call |
| outside-graph 1.05 ms (graph setup, detokenisation, Dart) | −0.3 to −0.5 (ceiling, split unknown) | unpriced; needs a profile of the outside time |
| dense requant at load | −4.3 for 24 % fewer bytes, but it must share the budget with skip. Near the budget edge skip buys about 1.0 ms per 1 % PPL (118 → about 138: −2.9 ms for +2.9 %), so the net is 4.3 − 1.0 × PPL(%). At the base there is no sub-4-bit byte-rate repacked container: only ≥ 4-bit types and 1-bit Q1_0 (*source*) | needs a new kernel; likely PPL +3 % or more; **not credible as 5 ms** |

- **Sum of the first four: −2.6 to −3.3. Remaining shortfall +1.9 to +2.6 at the central value** (−1.4 to +6.9 over
  the band).
- **Reading.** No lever of about 5 ms exists. A phone round for Granite at 25 would need both rulings, a better skip
  frontier, and the fast third of the band.
- **Proposed "no" for the lead's R-log, with these numbers:** central 43.2 − 3.0 = 40.2 ms-equivalent against the
  38.0 line, about 23.8 printed, short +1.9 to +2.6 ms. Revisit only if a `granite-Mi` skip frontier gives ≥ 50 % of slots inside
  9.6879.

## 8. Not verified

1. That the r16 APK contains the same `canCopyWeights(copyBytes)` rule as the 2.3.0+45 source I read (§1 caveat).
2. Every plain-kernel rate on the A78. Host ratios stand in until the §2.4 run.
3. The OLMoE node count (about 870) and the FA per-op cost.
4. The split of stock `MUL_MAT_ID` waits into fixed and proportional parts (the widest band, ±5 ms).
5. Whether `olmoe-Mi` is resident in the CLI and in the app. Every band is fault-free by definition.
6. f on the official prompt, and head2's PPL on OLMoE on the `-nr` path.
7. Whether Tom's 32-token binary decodes the 32nd token. The per-token mapping does not depend on it; a
   whole-run comparison does.

## 9. Addendum (same day): `olmoe-Mi` with weight repack ON on the phone

This is what a campaign-2 kit or app gets with a per-file copy size. For `olmoe-Mi` the copy is 235,505,664 B
(224.6 MiB, memo-05 §2, *host* header arithmetic), so 7.3 − 0.24 ≥ 5 and the rule turns repack on. Only the 64
attention Q4_0 tensors (`q4_0_4x4`) and the untied Q6_K head (`q6_K_8x4`) are copied. The Q3_K experts stay
mmapped, because base `620d2225c` has no Q3_K repack (*source*). Every number below is *estimated*. The two columns
use the same method: RSS bands, with prefetch's band included, which is why the OFF R53 band differs slightly from
§3.

### 9.1 Bands side by side

Rates with repack on:

- attention 151 MB at the repacked rate 0.040 / 0.041 / 0.044 ms/MB (R49 0.0398, R25 0.0411, *device*-derived),
  giving 6.0 / 6.2 / 6.6 ms;
- head 84.5 MB at the repacked Q6_K rate 0.044 / 0.046 / 0.055 (R31 5.58-5.85 ms and R29 6.5-7.0 ms for 126.4 MB,
  *device*), giving 3.7 / 3.9 / 4.6 ms.

Package with repack on:

- skip 13.39 %: −3.9 (−3.3 to −4.4), unchanged;
- untied head2 on a **repacked** head: −1.1 (−0.8 to −1.9), from 3.9 to a 2.7-2.9 ms screen. This needs memo-02's
  option A (30-40 lines of new code), because head2 refuses a repacked untied head as built;
- wait crumbs −0.25.
- Total −5.25 (−4.2 to −6.7), against −6.15 (−4.3 to −8.6) with repack off.

T_off = T_cli + 0.3. Printed = 0.995 × 31,000 / (32 · T_off + 12). All values are low / central / high.

| | repack OFF: ms | OFF: printed tok/s | repack ON: ms | ON: printed tok/s |
|---|---|---|---|---|
| R52 stock, CLI | 65.6 / 71.6 / 81.5 | 14.5 / 13.3 / 11.7 | **63.5 / 69.4 / 78.9** | 15.0 / 13.8 / 12.1 |
| R53 stack, CLI | 45.2 / 49.8 / 53.8 | 21.0 / 19.1 / 17.7 | **43.3 / 47.5 / 50.5** | 21.9 / 20.0 / 18.8 |
| final, T_off (stack + skip + head2 + crumbs) | 36.9 / 43.9 / 49.8 | 25.9 / 21.8 / 19.2 | **36.9 / 42.5 / 46.6** | 25.9 / 22.5 / 20.5 |

### 9.2 Does the +2.7 ms disappear?

**Mostly, about −2.25 of it.** Attention drops 7.55 → 6.2 and the stock head 4.8 → 3.9.

- The residual +0.45 ms against memo-03 is not a kernel cost. I priced the repacked head at R31's measured 0.046
  ms/MB, where memo-03 used 0.042.
- **At the end of the package the gain is smaller: only −1.4 ms.** Head2 brings the head to about 2.8 ms on either
  path, so repack's head gain and head2's gain overlap. What repack adds on top of head2 is the attention term alone,
  −1.35 (−0.2 to −3.2).
- The price of that −1.4 is option A's new code. With repack off, head2 works as built.

### 9.3 Memory and residency

**Anon cost: +224.6 MiB, unevictable.**

**Total demand is unchanged.**

- With repack off, the same 235 MB sit in the page cache as file pages.
- With repack on, the copy is anon, and its mmapped source pages are never touched again after load, so they age out
  (`GGML_REPACK_RELEASE` in c1 forces that).
- Either way the demand is about 2.77 GB of experts plus 0.24 GB.

Campaign-1 device comparison:

- R49 had fault-free tokens with about 0.55-0.7 GB repacked anon plus 2.6 GB of experts, i.e. about 3.3 GB of demand.
- R12 had 674 MB app anon, 2.07 GB of the model resident and 5 majflt/token (*device*).
- R10 showed that the page cache tracks free memory, not the file: it shrank about 0.3 GB when a 0.6 GB smaller file
  arrived (*device*).
- OLMoE with repack on needs about 3.0 GB, about 0.3 GB less than the Granite configuration that was resident in
  R49 (*estimated*).

Risks:

- **The anon copy does not shrink the expert cache enough to matter.** At most it competes for the 0.24 GB it
  replaces.
- An LMK kill is unlikely. The 0.7 GB copy never caused one (R-log open item); 3.6 GB did.
- The real residency risk is the 2.77 GB of experts against about 2.1-2.4 GB of cache. That risk is the same on both
  paths.

### 9.4 Which perplexity reference governs

The kernel class (*source* + memo-05, *host* load log) is as follows:

- **Repack ON on the phone** = `q4_0_4x4` attention + `q6_K_8x4` head + plain Q3_K experts. That is **the same path as
  R50/R51** (host default repack on; the host build has dotprod and no i8mm, so it picks the same `4x4`/`8x4` types).
  R50's 11.7146 and its budget 12.3003, and R51's f* = 13.39 %, apply to this configuration as measured.
- **Repack OFF on the phone** (the unmodified app, and the planned CLI R52) = all generic `vec_dot`. That is **the
  amendment-2 `-nr` path.** Its references are amendment 2's.

Bit identity across M4 and A78 is not verified on either path. Thread count can also change float order in FA and in
reductions.

### 9.5 Decision table, repack ON

25 needs the final T_off ≤ 38.0. On the R53 CLI steady that is **≤ 42.95 at the central package** (≤ 41.9 weak, ≤ 44.4
strong). **Stop line (short ≥ 6 ms after the package): R53 ≥ 49.0.**

| R53 steady (CLI, ms) | final T_off, central (weak to strong package) | 25 reachable? |
|---|---|---|
| ≤ 40 | ≤ 35.1 (≤ 36.1 to ≤ 33.6) | yes |
| 40 to 42 | 35.1-37.1 (36.1-38.1 to 33.6-35.6) | yes; the weak package misses by 0.1 at 42 |
| 42 to 44 | 37.1-39.1 (38.1-40.1 to 35.6-37.6) | at the central package only up to 42.95; a coin flip |
| > 44 | > 39.1 | no with this package. The extras left are the 32nd decode (−1.2 equivalent) and router F16 (−0.26), both rulings. Stop rule at R53 ≥ 49.0 |

**Central with repack on:**

- R53 47.5 gives a final 42.5 ms, **4.5 ms above 38.0** and **1.5 ms inside the stop line** (22.5 printed).
- With repack off it is 43.9 (5.9 above 38.0, 0.1 inside the stop line).
- Repack on is the better phone configuration by about 1.4 ms, and it is the one that R50/R51 already describe.

## Sources

- llama.cpp PR #17494 (arm64 Q4_K dotprod gemm/gemv), <https://github.com/ggml-org/llama.cpp/pull/17494>
- llama.cpp PR #19360 (arm64 Q6_K repack gemm/gemv with dotprod), <https://app.semanticdiff.com/gh/ggml-org/llama.cpp/pull/19360/overview>
- The web search (2026-11-14) found no Q3_K repack kernel upstream. Base `620d2225c` has none (`repack.cpp`, read
  locally).
