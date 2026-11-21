# memo-06 work file, Clyde: re-plan at 12 tok/s (arithmetic, rounds, kill lines, owner items)

2026-11-21, clyde (`claude-opus-5-5`). Nothing here is measured. Every number is *estimated*, or quoted with its
label. Inputs come from memo-01, memo-02 (H1 bytes, *host* headers), memo-03, memo-05 and R69's bands
(memo-04-work/bonnie-bands.md). Merged into memo-06 §5.

## 1. What 12 tok/s requires

**Token model** (memo-04-work §6). t = max(B_eff ÷ r, W_eff ÷ c_w) + O, with:

| symbol | meaning |
|---|---|
| B_eff | bytes read per token |
| W_eff | touched weights × the kernel's per-weight cost k, in units of stock Q4_0 `sdot` weights |
| c_w | c_b ÷ 0.5625 bytes per weight: the Q4_0 weight rate |
| O | fixed per-token cost, 6 to 12 ms (8 central) |

12 tok/s means t ≤ 83.3 ms. Both lines must hold at once.

**Byte line.** B12 = r × (83.3 − O):

| r, GB/s | 19 | 24 | 30 | 35 | 40 | 45 |
|---|---|---|---|---|---|---|
| B12 at O 8, GB | 1.43 | 1.81 | 2.26 | 2.64 | 3.01 | 3.39 |
| B12 at O 12, GB | 1.36 | 1.71 | 2.14 | 2.50 | 2.85 | 3.21 |

A dense Q4_0 body of 7.0 B multiplied weights (4.0 GB, memo-02 H1) needs r ≥ 54 GB/s, which is above the
nominal 51.2 GB/s bus. A 6.0 B body (Yi-1.5-6B class, about 3.4 GB) needs r ≥ 45 GB/s, which is Ebony's reachable
ceiling.

**Weight line.** W12 = c_w × (83.3 − O):

| c_b, GB/s (R69 band 35 to 45, centre 40.4) | 35 | 40.4 | 45 |
|---|---|---|---|
| c_w, G weights/s | 62 | 72 | 80 |
| W12 at O 8, G weights | 4.7 | 5.4 | 6.0 |

This is the finding of this file. **Every model in the 6.0 to 8.5 B class multiplies at least about 5.5 B weights
per token**, so at the stock Q4_0 per-weight cost it is compute-bound below 12 tok/s on X1 + 2 A78, at any read
rate.
- At c_b 40.4: 7.0 B multiplied gives at most 9.5 tok/s; 6.0 B gives 10.9.
- A 4th big core adds about × 1.13 (R69 band; W12 becomes about 6.1).
- Lowering the bits does not move this line unless the low-bit kernel costs no more per weight than Q4_0 does.
  Q3_K runs at about half the `sdot` rate (*device*, memo-01).

**Per model size and served bits.** P is multiplied weights, bpw the average served bits per multiplied weight,
d the kept fraction under column sparsity, and b = 1 − 8/9 × (1 − d) the share of bytes still read. 12 tok/s needs
both of these:
- bytes: P × bpw/8 × b ≤ B12(r);
- weights: P × d × k ≤ W12(c_b), or more compute from the GPU lane (R69 Rule D) or a 4th core.

The highest bpw that the byte line allows (O 8):

| P, B | dense, r 24 | dense, r 35 | d 0.60 (b 0.644), r 24 | d 0.60, r 35 |
|---|---|---|---|---|
| 6.0 | 2.41 | 3.52 | 3.75 | 5.47 |
| 6.5 | 2.23 | 3.25 | 3.46 | 5.04 |
| 7.0 | 2.07 | 3.01 | 3.21 | 4.68 |
| 7.5 | 1.93 | 2.81 | 3.00 | 4.37 |

**Residency caps bpw from above.** The cap is (C − 0.15 GB) × 8 ÷ P. The 0.15 GB is KV at ctx 512 plus compute
buffers, *estimated*.
- At R69's central C 3.1 GB: 3.9 bpw at P 6.0, 3.4 at P 7.0.
- At C 3.4: 4.3 at P 6.0, 3.7 at P 7.0.
- Column sparsity does not shrink the resident set (H5: the 32-token union is about 1.0, *host*).

**Quality caps bpw from below.** On Llama-3-8B, every quantisation below about 3.05 bpw fails F16 × 1.20 (memo-01,
*published*). This is model-specific and has to be measured per subject (H2).

**The corner, read together (estimated).** Take a 6.0 to 7.0 B body served at about 3.1 to 3.7 bpw, with column
sparsity at d about 0.60.
- The byte line holds from r about 20 to 28 GB/s.
- The weight line needs k ≤ 1.29 (P 7.0) to 1.50 (P 6.0) per touched weight, at d 0.60 and c_b 40.4, for a kernel
  that reads a sub-4-bit file.
- The best measured exact column kernel is K1 on Q4_0 at k 1.53 to 1.64 (*host*). No arm64 repack or fast kernel
  exists at `620d2225c` for IQ3 or Q3_K.

So 12 needs three things at once, on each of the three subjects:
- a sub-4-bit file inside the ceiling;
- a new low-bit exact column kernel at k ≤ 1.3 to 1.5;
- r of about 24 GB/s or more, or a GPU lane that adds compute.

**Lead's prior** (a judgement, not a price; memo-06 §3 replaces it): P(all three pass) is about 0.05 to 0.15. The
owner is told this now, with these lines, and nothing is bent to change it.

## 2. Round plan (from now to a claim per model)

R-numbers are given as "next free". R70 and R71 are campaign 2's, so the next free number after R69 is R72 at the
time of writing. Campaign 3's next kit is r32.

| stage | where | content | exit / kill line |
|---|---|---|---|
| S0 | paper + host | memo-06 §2 to §4: pool, pricing at r 19 / 35, host screens (H1 headers, H3 column frontier, H2 ladder where disk allows) | three subjects + two reserves chosen by §1 (memo-06 §6) |
| S1 = R69 | device, no model | probe kit r29 as pre-registered + R69-amendment-1. Can run now, in parallel with S0 | Rules A to D as amended. **K0** per subject: if no priced lever set meets both 12 lines at R69's r, c_b, C, the subject is "out of reach by arithmetic", written with numbers |
| S2 = R72 | device, no model | kit r32: D2 kernel rows (K1 TR 128, Y1b, short-row skip) on a pinned A78, hot and cold; **per-type gemv rows** on synthetic arenas of the subjects' shapes (Q4_0, Q4_K, IQ4_NL, Q3_K, IQ3_S, Q2_K at least); D3 header; X1+3 A78 row | k per type and per kernel, *device*. **K-type**: a type whose per-weight cost exceeds W12's room for every subject at the R69 r is dropped from the freeze choice |
| S3 | host, owner's laptop | **served-file freeze per subject**: quantise from the publisher's BF16 with stock `llama-quantize` at the base (imatrix from wikitext-2 train allowed, declared and sha-recorded); full-set F16 PPL and stock served PPL (`tools/ppl.sh`, all chunks); ceiling check stock ≤ F16 × 1.20; resident estimate ≤ C(R69) − 0.15 GB; choose, among the types that pass, the lowest estimated token time at R69/R72 numbers (ties to more bits); write the budget (stock × 1.05, absolute PPL) in the model card; sha256 of the BF16 shards and the GGUF | **K-F**: no type passes ceiling and residency → the subject "does not fit inside the ceiling", reported; the lead proposes a reserve, the owner replaces |
| S4 = baseline | device | all three subjects in one session: load, residency over 20 decode passes (D1), median of 10 plain runs at campaign 2's best bit-identical configuration (pin, exec map, touch, kv clear) | **K1** per subject: baseline (*device*) × the product of the upper-band gains of the open levers < 12.0 → "12 out of reach on this subject with priced levers", reported; research continues only on new levers |
| S5 = lever rounds | host, then device | one pre-registration per round; numerics levers pass full-set PPL at `-b 512` and `-ub 1` per subject first (*host*); on the device every lever on **at least two of the three subjects in the same session** (all three when the kit holds them), in-run ABBA, reported per model with what generalised and what did not | each lever's own kill line; K1 re-applied after every round |
| S6 = go + claim | device | go block: median ≥ 12.15 tok/s (a 1 ms margin) and at most 2 of 10 below 12.0, uncapped clocks; then one claim per subject: N = 10 plain runs in a pre-named block, positions fixed, no run dropped, not right after a model switch, power saving off, deep-sleeping list recorded, charger out; the claim configuration's full-set PPL ≤ budget at `-b 512` and `-ub 1` and ≤ F16 × 1.20 | **pass** per subject: median ≥ 12.0. **Campaign goal**: all three pass. A failed claim stays in the log; a new attempt needs its own pre-registration and a changed configuration |

Notes:
- **Start state.** Everything starts from campaign 2's shipped state: the c2 series 0001 to 0009 and kit r28's
  protocol, r16 gate, heat and clock series. New numbers are compared with campaign 2's under the same phone state.
- **Before S1.** The memo-06 §1 rule says no candidate is measured on the phone before the three are chosen and
  frozen. S1 and S2 load no model, so they may run before the choice.
- **Choice timing.** The choice (S0) uses R69's r if R69 has reported by then, else r 19 and 35 both, as §1
  says. It is not delayed to wait for R69.
- **Shared rounds.** Every lever is model-generic code, so a lever round is one kit for all three subjects. The
  generality report is a table per lever × subject.

## 3. Pre-amendment owner questions (a) to (d): lead's provisional answers

The owner delegated ("proceed without asking", 2026-11-21). These are **lead decisions, to be confirmed by the
owner**. Each one is reversed if the owner says no.

- **(a) Is Y1a, a phone-built kernel-layout side file keyed by sha, a weight-derived artifact? Provisional yes.**
  COMMON_RULES 1 already says "A derived artefact that the runtime computes once on the phone from the served file
  and caches in the app's files counts as runtime" (owner's answer, R42 amendment 2). The 2026-11-19 sidecar
  regime applies:
  - committed generator;
  - model and table sha256s, with the runtime refusing a mismatch;
  - the sha256s listed in every pre-registration.
  Y1a is an exact re-layout, so its identity check is the bit-identity check, not a budget item.
- **(b) Is twice the storage per model acceptable? Provisional yes, for the research kit.**
  - Each round records free storage before and after in the kit header.
  - Three subjects at about 3 to 4 GB each means about 18 to 24 GB with copies (*estimated*).
  - At 12 tok/s, Y1a matters only if C ≥ 4.2 GB (memo-05 D4), and the sub-4-bit files of §1 make it less likely
    to be built at all.
- **(c) Does the answer cover non-stock layouts (a column-major copy, a 1-bit proxy file)? Provisional yes for
  both, under the same regime.**
  - A column-major copy is an exact re-layout (bit-identity check).
  - A 1-bit proxy is lossy data used only to choose what to skip. It is counted inside the +5 % budget at
    `-b 512` and `-ub 1`, with its own PPL check, like a calibration table.
  - This is the least certain of the four answers. If the owner says no, the lever that uses it closes.
- **(d) The served Q4_0 file per subject: superseded by Amendment 1.**
  - The subjects change, and §1 shows that a Q4_0 body of 6 B or more cannot meet the byte line under 45 to
    54 GB/s, so the served file is not assumed to be Q4_0.
  - Provisional rule (S3): we quantise each subject ourselves from the publisher's BF16 safetensors with stock
    `llama-quantize` at the campaign base, into one file (no split). We record repo@commit, the sha256 of every
    input shard and of the output, and the imatrix, if any.
  - No third-party GGUF is used. This also removes the second-state stand-in and the publisher's two-part Qwen
    split.
  - Disk: about 15 to 17 GB of BF16 per model on the owner's laptop, one model at a time, keeping 10 GB free.
