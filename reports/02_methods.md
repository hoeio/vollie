# 02 — Methods survey for Granite-4.0-H-Tiny (abliterated, Q4_0) on Snapdragon 888

Model is locked (see `00_model_choice.md`). Sources are linked inline; the full research notes with
a URL for every claim are summarised here. "Measured on host" = Apple M4 Pro, CPU only; "estimated"
= the SD888 model in §1; nothing here was measured on a phone in this project.

## 1. What limits decode on the SD888 (the model behind every estimate)

| Fact | Value | Source / how |
|---|---|---|
| ISA (X1, A78, A55) | Armv8.2-A + dotprod + fp16. **No i8mm, no SVE, no bf16** | LLVM `AArch64Processors.td`, Arm X1 TRM |
| DRAM | LPDDR5-6400, 4×16-bit, 51.2 GB/s theoretical | AnandTech S888 deep dive (16271) |
| Best published llama.cpp effective bandwidth on SD888 | 16.6 GB/s (qwen2-0.5B Q4_0, 39.4 tok/s), 24 GB/s (1.5B F16, 8.05 tok/s), 8 threads | [llama.cpp PR #13079](https://github.com/ggml-org/llama.cpp/pull/13079) |
| User's own phone (Vollie phone test, 2026-10-25, Qwen3.5-4B Q4_0, 2.23 GB/token, repack on) | 6.6 tok/s cold → 5.9 hot; prefill 17-21 tok/s | earlier sessions' device logs (user-run, not reproduced here) → **13-15 GB/s effective** |
| Granite bytes per token (weights) | 776 MB = dense 420 + experts 213 (4 of 64) + router 16 + lm_head 126 (Q6_K) | `tools/cost.py` (measured from the GGUF) |
| Granite recurrent state traffic | 58 MB f32 Mamba2 state + conv, read **and** written each token → ~116 MB | `tools/cost.py` |
| Granite graph | 3253 nodes per decode graph (≈2000 compute ops; Qwen3.5-4B: 1905 nodes) | llama.cpp `sched_reserve` log (measured) |
| Repack on Vollie's 8 GB SD888 | **off** (OOM-killed with it on: CPU_REPACK 3105 MiB + KleidiAI 541 MiB) | `lib/device.dart:107`, user's crash log 2026-10-31 |

Estimate used throughout (explicit assumptions):
`t_token ≈ bytes_per_token / BW_eff + overhead`, with `BW_eff` = 13-16.6 GB/s for repacked Q4_0 at
4 big threads (the two calibration points above). Non-repacked Q4_0 decodes 22% slower on the host
(memory note, re-measured in `01`), so `BW_eff` ≈ 10-13 GB/s for the shipped no-repack config.
Overhead (barriers on 3253 nodes, ssm_scan, routing) is folded into BW_eff by the calibration, which
used dense models with ~4x fewer graph nodes per byte read (Qwen3.5-4B: 1905 nodes / 2.3 GB;
Granite: 3253 / 0.9 GB), so the Granite estimate is optimistic by an unknown amount
(assumption: 0-25%; the 0.5B calibration model shows ≤ 25 ms/token of fixed cost at 39 tok/s).

## 2. Measured facts that drive the ranking (all host, Granite, Vollie config: FA on, q8_0 KV, ctx default)

PPL = wikitext-2 test, ctx 512; reference F16 with the model's 6 experts = **9.0438**; gate ≤ **9.948** (+10%, the limit in force when this memo was written; since 2026-11-02 the hard limit is +20 % = ≤ 10.853, `RULES.md` §1).

| config | wiki PPL | Δ vs F16 | weight MB/token | host tg64 4 thr (1 thr) | non-reclaimable RAM (host peak footprint) |
|---|---|---|---|---|---|
| F16, 6 experts (reference) | 9.0438 | 0 | 2,500+ | — | — |
| **shipped**: Q4_0 i1, 4 experts, repack **off** (what the 8 GB phone runs) | 10.1619 | **+12.4% FAIL** | 776 | 80.1 (27.8) | 128 MB |
| same file, repack on (OOM on the 8 GB phone) | 10.1619 | +12.4% FAIL | 776 | 114.0 (46.1) | 3,931 MB |
| Q4_0 i1, **5** experts, repack off | 9.6176 | +6.3% pass | 829 | 75.5 (26.8) | 128 MB |
| Q4_0 i1, 6 experts, repack off | 9.4210 | +4.2% pass | 882 | 71.2 | 128 MB |
| IQ4_NL (same bytes), 4 experts, repack off | 9.9999 | +10.6% FAIL | 774 | 73.1 (24.8) | — |
| repack dense only (lab knob `LLAMA_NO_REPACK=exps`), 4 experts | 10.1619 | FAIL | 776 | 101.0 (38.7) | ~650 MB |

Take-aways: (1) the shipped 4-expert setting is the single largest quality loss (~6 points of the 12);
(2) repack is worth +42% (4 thr) to +66% (1 thr) decode on this ISA, and ~60% of that comes from the
dense 0.5 GB alone; (3) IQ4_NL buys ~2 PPL points at equal bytes but costs ~9% speed unrepacked.

## 3. Ranked methods

Effort: S < 1 day, M = days, L = weeks / needs training. "Verify w/o device": can the claim be checked on
the host? Speedups are for SD888 decode and are **estimates** unless marked host-measured.

| # | method | expected SD888 decode effect | PPL cost (measured unless noted) | effort | SD888 feasible | verify w/o device |
|---|---|---|---|---|---|---|
| 1 | **Restore repack within RAM: experts in a non-repacked type (IQ4_XS), dense Q4_0 repacked**, `use_extra_bufts=true` on 8 GB phones (GGUF + 1-line app change, stock llama.cpp) | host +15% at 5 experts vs shipped (92.1 vs 80.1, 4 thr); phone est. +15-35% vs no-repack | see `results.csv` (D1 rows) | S | yes: 0.66 GB copy vs 2.3 GB Qwen copy that ran | speed: host only; RAM: host footprint |
| 2 | **Expert count 4 → 5** (GGUF key) — required by the gate, not a speedup | −6% host | −6 points of PPL (12.4% → 6.3%) | S | yes | yes (PPL) |
| 3 | Per-layer expert count (k=4 on insensitive layers, 5-6 on sensitive), GGUF array key + 1-line libllama patch (`granite-hybrid.cpp`) | +0-5% vs uniform 5 (fewer expert bytes) | tune on wiki **valid**, gate on test | M (patched libllama.so must ship at llamadart's commit) | yes | yes |
| 4 | Lower-bit experts (IQ3_S/Q3_K, 3.4-3.5 bpw) at 5 experts | −5% bytes/token → +3-5% est. | est. +2-4 points (not run) | S | yes, but Q3_K/IQ3 dequant is compute-heavier on A78 | PPL yes, speed partly |
| 5 | lm_head Q6_K → Q4_0 | −5% bytes → +3-5% est. | +2.7% on en.txt (earlier session) | S | yes | yes |
| 6 | Thread count / big-core pinning (`-C 0xF0 --cpu-strict 1`, 3 vs 4 threads, `--poll`) | 0 to +30% (A55 in a barrier = slowest-core stall; PR #13079 shows 3-5x build/barrier swings) | none | S (Vollie already pins DartWorker threads) | yes | **no** — needs device (kit: `tools/bench_device.sh`) |
| 7 | Tiny-op barrier skip (`GGML_TINY_OP`, earlier lab patch; 3253 nodes/token) | 0-15% est. on SD888; −3..−15% on M4 | none (bit-identical) | S (needs custom CPU .so) | yes | **no** |
| 8 | f16 Mamba2 state (58 MB r/w per token → 29) | +3-6% est. | unknown, small | M (ssm_scan kernel) | yes | PPL yes |
| 9 | KleidiAI dotprod kernels for the repacked dense part (stock bundle has them) | −15..+10% (5-15% slower on M4) | none | S | yes (stock) | no |
| 10 | Speculative: n-gram / draft 350M / EAGLE-3 | **negative** on this MoE: measured 0.53-0.6x on M4; verify cost grows with union of experts (Cascade, arXiv 2506.20675) | none | M-L (EAGLE needs training) | runs, but loses | yes (lost on host) |
| 11 | MTP | n/a — Granite has no MTP head (training needed) | — | L | — | — |
| 12 | Activation sparsity (TEAL/CATS/R-Sparse) | 1.2-1.5x claimed with custom kernels; Q4_0 block-32 limits column skipping | +1-3% claimed at 40% | L (kernels) | uncertain | PPL yes |
| 13 | Adreno 660 OpenCL | measured on this phone (Qwen): writes ~35% slower than CPU | none | — | runs with workarounds | no |
| 14 | Hexagon HTP | ggml-hexagon builds v73+ only; SD888 is v68 | — | L | **no** | — |
| 15 | QAT / SpinQuant / 2-bit (ParetoQ, QuIP#, AQLM) | up to −40% bytes | needs training or no ggml kernels | L | — | — |
| 16 | KV-cache quant beyond q8_0 | ~0: only 4/40 layers have KV (34 MB f16 at 4k) | — | S | yes | yes |
| 17 | Weights in RAM vs mmap (`--no-mmap`, mlock) | removes page-cache refault stalls under memory pressure; no steady-state gain | none | S | risky on 8 GB | partly |

## 3b. Update 2026-11-01: first phone measurement of Granite

User-run Settings speed test, Z Flip 3, app 2.1.1+13, `vollie-granite-4.0-h-tiny-d1-e5.gguf`, CPU 4 threads:
**7.4 tok/s writing, 14 tok/s reading (measured on device)**, below the 10-14 estimate above.

- 7.4 tok/s = 135 ms/token. At the 13-16.6 GB/s this phone reached with Qwen3.5-4B (2.23 GB/token at
  6.6 tok/s ≈ 15 GB/s), Granite's ~0.93 GB/token (weights + Mamba state) needs only 56-72 ms. So
  **~60-80 ms per token is fixed cost, not bandwidth**: Qwen ran 1905 graph nodes per token, Granite 3253
  (≈2000 compute ops, each followed by a thread sync; 108 of them are empty). The §1 caveat that the
  estimate was "optimistic by an unknown amount" turned out to be ~2x.
- So method 7 (tiny-op barrier skip) moves up to the top: it is the only shipped-without-libllama lever
  on that fixed cost. App 2.1.3+15 ships it in the armv8.2_2 CPU module (bit-identical PPL on host) and the
  speed test sweeps its threshold on the device. Method 6 (thread count) is already swept by the speed test.
- File-level node cuts (fold residual_scale: −80 SCALE nodes/token) wait for the sweep: if syncs are
  the cost, they help; with tiny-op on, they mostly don't.
- **Result (measured on device, same day):** tiny-op doubled writing at fixed threads (2 threads 3.2 →
  7.0 tok/s; 4 threads 3.4 → 5.6 at matched heat), and v2 (874 syncs/token instead of 1,909) reached
  8.2 tok/s at 3 threads. Method 7 is now the best-performing method in this survey; full log in
  `03_final.md` §5.

## 4. Is 20 tok/s at ≤ 10% plausible?

Updated with the phone measurement (§3b): even with zero fixed cost, 0.93 GB/token at ≤ 16.6 GB/s is
≥ 56 ms → **≤ 18 tok/s ceiling** for the gate-passing file; 20 tok/s needs ≤ 0.83 GB/token at the best
bandwidth ever seen on this SoC *and* no per-token overhead. Still not plausible; the realistic target is
cutting the ~70 ms fixed cost (→ ~10-13 tok/s, estimate).

Original estimate (before the phone number):

Estimated SD888 decode (method in §1 and `03_final.md`): shipped config ≈ 8-12 tok/s (to be verified on
device); best gate-passing config found so far (method 1 + 2) ≈ 10-15 tok/s. Reaching 20 needs about
50 ms/token: at the 13-16.6 GB/s effective bandwidth seen on this SoC that is ≤ 0.65-0.8 GB of total
traffic with *perfect* kernels, while the gate-passing configurations move 0.93-1.0 GB/token
(weights + Mamba state). **Under the ≤ 10% PPL constraint, 20 tok/s with this model is not plausible on
the SD888 CPU by these estimates**; the closest trade-off is listed in `03_final.md`.

