# Knowledge digest from campaign 1

A working summary so a new session does not have to read the 1,074-line R-log (`reports/03_final.md`). **It is a
digest: when it disagrees with an original (R-log entry, `results.csv`, the docs), the original wins**, and the
discrepancy should be fixed here. R-numbers point to the entries. Labels: *device*, *host* (Apple M4), *estimated*.

## 1. Hardware and measurement facts

- Phone: Galaxy Z Flip 3, Snapdragon 888 (1× Cortex-X1, 3× A78, 4× A55), 8 GB. Compute threads run on the big
  cores (3 threads were best; 4 threads and an A55 helper lost, R18). Fast cores ended R49 runs at 2,112 / 2,035
  MHz; on other days they sat at 1,555 MHz, and compute-bound ops scale with the clock (R27).
- Page cache: only about 2.1 GB of the phone's memory stays as cache, smaller than the model's mmapped experts.
- Dense-weight read rate on the device: about 19 GB/s; the byte rate used for costing is **0.0411 ms/MB**
  (R25, *estimated*).
- Host numbers (M4) are relative only: a thread sync costs about 0.4 µs there versus about half of every token on
  the phone (R4); the page-cache thrash does not appear on a 24 GB host (R9).

## 2. What one Granite token is made of

Granite-4.0-H-Tiny: 36 Mamba-2 + 4 attention layers, 64 experts (6 used per token), served file Q4_0 dense,
Q3_K experts (3.1 GB, read through mmap), Q6_K head. Steady official token at the R49 state: **about 45.5 ms in
the graph + 1.05 ms outside** (*device*).

Dense bytes read per token (host inventory, R42; ms at 0.0411 ms/MB, *estimated*; device `MUL_MAT` measured 22 to
24 ms):

| tensor group | MB | ms |
|---|---|---|
| output head (Q4_0x4 screen 86.7 + 2,048 exact rows) | 89.3 | 3.7 |
| `ssm_in` | 200.6 | 8.2 |
| `ssm_out` | 95.6 | 3.9 |
| shared expert | 106.7 | 4.4 |
| attention | 14.2 | 0.6 |
| router `ffn_gate_inp` (F32) | 15.7 | 0.65 |
| norms / conv | 3.4 | |
| **total** | **525.5** | **21.6** |

After the fused expert op a device graph has 1,775 nodes and 666 barriers per token. For comparison, stock
settings had about 1,926 barriers per token on the device (findings.md); the host graph counts differ (stock
1,947 nodes and 1,910 barriers on the host, R25 vanilla check).

## 3. Levers that survived (details: [../../docs/levers.md](../../docs/levers.md), knobs: [../../docs/runtime-knobs.md](../../docs/runtime-knobs.md))

Bit-identical stack (small ops on one thread without syncs, expert prefetch, 4-row Q3_K kernel, in-place SSM
state, in-register kernel, fused gate+up+SwiGLU, sampler fast path): steady token 91.0 → 63.9 ms at that stage
(R25). Expert skip by router weight (threshold 117): about −8 to −10 ms (*device*, R25/R27). Two-stage head
(Q4_0 screen, top 2,048 exact rows): about −1.6 to −1.7 ms. Fused expert op: −0.80 ± 0.10 ms (R47).
Served numerics: skip 117 + head 2048, PPL 9.7886 (+1.91 %).

## 4. Closed, with numbers (do not re-propose without new evidence)

- Speculative decoding of every kind: 0.48 to 0.76× the plain rate; the CPU is compute-bound at any batch above one (R20).
- Early exit: the cheapest layer already costs +2.5 % PPL (R21).
- 4 compute threads, A55 expert-guess helper (R18); anonymous-memory expert cache and Q3_K software prefetch (R23);
  finer dense chunks (R33); static Q4_0 copy of the 4 most-routed layers' experts (R32, R33); Q2_K/Q3_K head
  screens (R29); `MMID_TOUCH` (parked, R41); OpenCL on the Adreno 660, 0.7 to 2.9 tok/s (*device*).
- Dense structural numerics under the +5 % cap (R42, R43): router F32→F16 passes but is worth about −0.32 ms; the
  low-rank head screen failed on recall (0.87) and on the denominator, closed at every rank and clamp;
  shared-expert pruning at 12.5 % measured ratio 1.068 against a kill limit of 1.012 (pruning itself about
  +6.7 %); Mamba-2 head pruning was not tried (about 12 % PPL per ms); a hard vocabulary subset (out-of-subset
  targets give infinite PPL, and choosing the subset on the test text is not allowed); dynamic channel skipping (R29);
  sub-4-bit dense containers (no byte-rate kernel). The cap was left unspent.

## 5. Perplexity data (Granite, wikitext-2 test, ctx 512, all 565 chunks, *host*)

F16 9.0438; stock served file 9.6051 (+6.21 % over F16); `en.txt` F16 10.3768, stock 11.1001.

| expert skip threshold | PPL | % over stock | note |
|---|---|---|---|
| 115 | 9.7771 | +1.79 | R23 |
| 118 | 9.7952 | +1.98 | campaign 1's edge, R25 |
| 119 | 9.8087 | +2.12 | fails the +2 % budget |
| 130 | 9.9574 | +3.67 | first written as +3.60, corrected in R25; `en.txt` 11.7796 |
| 140 | 10.1089 | +5.25 | first written as +5.20; over the +5 % cap; `en.txt` 11.9853 |

Served pair skip 117 + two-stage head (Q4_0 screen, 2,048 rows): 9.7886 (+1.91 %); the head alone adds +0.01 %.
Head screens used alone fail the gate (Q2_K +52 %, Q3_K +12.8 %, Q4_0 +4.25 %).

**The one unpriced lever:** requantising the dense Q4_0 tensors to fewer bits at load, as a runtime-derived
artefact (the owner ruled such an artefact is "runtime", R42 amendment 2). The Q3_K x4 kernel for experts exists
(`tools/q3k_kernel/`). Hint of the cost: re-quantising an already-Q4 tensor changed 10 to 14 % of its bytes
(relative weight error 1.6 to 2.6 %) for +0.15 % PPL (R43). A quarter fewer dense bytes would be about −5 ms
(*estimated*) but its PPL is expected to exceed the budget; only a host screen settles it.

## 6. Instruments and how a claim is made

- **ABBA in-run A/B** (`GGML_CPU_AB=knob:NAME:A:B`): warm-up tokens, then blocks A B B A, so heat, clock and
  cache hit both sides; report B − A ± σ. Per-block σ on clean blocks is 2.3 to 4.5 ms, so a small effect needs
  about 30 counted blocks per side; a verdict on a sub-1 ms effect needs σ ≤ 0.4 or it is "unresolved" (R42).
- **Research kit** in the app: one JSON report; parse with `tools/research_parse.py <report.json> --official`
  (the owner runs it). Raw reports are in `reports/device/r1…r16`.
- **Official benchmark:** 155-token prompt, 32 generated tokens, greedy, llamadart `decodeTokensPerSec`; each run
  waits at the kit's cool-down gate. Claim rule: median of 10 plain runs. R49: 14.09, 18.47, 20.63, 20.62,
  20.23, 20.83, 20.95, 20.40, 20.84, 20.25 → median 20.51. Its time includes one final decode with unused logits
  (about 3 %, *estimated*, R48); it was left in.
- Cold start (after memory pressure) prints 12.3 to 18.5 tok/s: page residency, not arithmetic. Fixing it in the
  runtime is open.
- Always read page faults, clocks and memory next to a timing: they explained most surprises (R9).

## 7. Where things are

| need | file |
|---|---|
| a specific R-number's one-line verdict | [../../reports/README.md](../../reports/README.md) |
| the full R-log, pre-registrations and verdicts | `reports/03_final.md` |
| every result as a row | `reports/results.csv` |
| raw device reports | `reports/device/` |
| dense-numerics plan and kill rules | `reports/memos/round14_plan.md` |
| expert-skip design and costing | `reports/05_compute_less.md` |
| the patches and what each feature is for | [../../patches/README.md](../../patches/README.md) |
| Q3_K kernel generator and benches | `tools/q3k_kernel/` |
| host reproduction | `tools/reproduce.sh` (accepts `REPRO_*` variables for other models) |
| the app-side helper library | `runtime/dart/lib/vollie_runtime.dart` |
