# c3 memo-05: no-copy layouts (Y1), a cheaper exact column kernel (K1), multi-stream GEMV and row skip (joint round 3)

Date: 2026-11-19. Bonnie (`claude-fable-5-2`) and Ebony (`claude-fable-5-2`) drafted; each reviewed the other.
Tom (`claude-opus-5-5`) ran the host tests. The lead (`claude-opus-5-5`) merged the round and made the decisions
in §3. Sources: [memo-05-work/](memo-05-work/) contains `bonnie-draft.md`, `ebony-draft.md` (the review is in its
§5) and `tom-host.md` (every measurement beside both drafts' predictions). R69 has not run, so there is no phone
number in this memo. Measurements are *host* (M4 Pro, relative only); prices are *estimated*.

## 1. What the host showed (*host*, tom-host.md)

| test | result | against the prediction |
|---|---|---|
| T1 K1 (strip-register axpy, bias-form nibbles), TR 128, d 0.60 | bit-identical to ggml's NEON 4x4 gemv; k 1.53 to 1.64, hot and cold, all four shapes | Bonnie 1.6 (1.35 to 1.95): hit |
| T1 K1, TR 512, cold | k 2.20 to 2.54; killed on the two `down` shapes | – |
| K1 instruction mix per touched weight | 0.125 SMLAL, 0.0625 AND/USHR, float 0.043/d; 0.259 V/w against stock 0.164 | device floor 1.58× (paired pipes) to 1.96× (SMLAL on V0 only, Ebony), *estimated*; the M4 cannot tell them apart |
| T2 Y1b (in-place 4-row kernel) | bit-identical; hot k 1.47 to 1.51; cold 2.07 to 2.21 on the long-row `down` shapes | Bonnie 1.4 (1.2 to 1.7) hot: hit |
| T4 gemm vs gemv | equal per element on all shapes; K1 and Y1b also equal the gemm | – |
| T3 `--no-repack` (Y1c) | **not** bit-identical, differences from the 3rd decimal; 40-chunk `-b 512` delta −0.014 % Mistral, +0.017 % Qwen2.5; 1.9× ns per weight at 1 thread, 1.4 to 1.5× at 8 | Bonnie: 4th decimal, a miss |
| T7 lines touched | exactly (4 + 32d)/36 for every TR | Bonnie's b: hit |
| E6 Y1b on arbitrary row pointers | bit-identical; hot k 1.46 to 1.52; cold 1.92 to 3.36, failing on every shape | Ebony kill line 1.7: killed cold |
| E1 `down` input only | 60 % inside on both models at both settings (Mistral +3.41 / +3.35 %, Qwen2.5 +1.92 / +2.32 %); 70 % outside on both | Ebony's s = 0.7 prices are past the frontier |
| E2 gate rule (abs SiLU(g)) | Mistral fails at 30 % (+6.34 %), Qwen2.5 fits (+3.73 %); 50 % fails on both | closed; did not generalise |
| T5 H5 active-set stability | body-weighted h is 0.002 / 0.005; the 32-token union covers 0.98 to 1.00 of all columns | Bonnie's prior 0.05 to 0.15: a miss, low |
| T6 Y1a dry run | not run (it needs a loader patch) | – |

## 2. Where the two drafts disagree

- **K1's device cost.**
  - Bonnie: 1.8 (1.55 to 2.2).
  - Ebony: 2.1 (1.5 to 2.6), because SMLAL issues on V0 only.
  - Tom's instruction count puts the floor between 1.58 and 1.96.
  - Only a phone row decides it (D2).
- **Multi-stream GEMV.** The pre-registered band for R69 X1 (Bonnie: 1.05, P(≥ 1.15) about 0.25) stands. Ebony's
  lower band is recorded here, before the data, to be scored beside it:
  - 2 streams: 1.02 (0.94 to 1.12).
  - 4 streams: 1.00 (0.88 to 1.12).
  - P(≥ 1.15) 0.10 at 2 streams and 0.08 at 4.
- **Ebony's 15 tok/s challenge.** It found no corner that falls on both models; one paper corner reaches 15.0 on
  Mistral alone, with a joint chance under 10⁻⁴ (*estimated*). memo-01 §1 stands.

## 3. Lead's decisions

1. **Closed:**
   - **K2** (hot-core split): h is 0.002 / 0.005.
   - **H5 as a residency lever:** column sparsity does not shrink the resident set; the 32-token union is about 1.0.
   - **The gate rule (E2).**
   - **K1 at TR 512.**
   - **Software prefetch in the column kernel:** worse in every cell, *host*.
   - **LDNP and DC ZVA:** priced at zero.
   - **T-MAC tables on Q4_0.**
   - **SpAx, NorSA, KV and embedding work.**
2. **Open, gated by the phone:**
   - **K1 at TR 128** (only TR 128 survives). It passes Rule A on the host. Its device k must be read on the A78,
     cold. If D2 reads k ≥ 2.3 at d 0.60, K1 is killed.
   - **Y1b.** It is bit-identical and the hot cost is close to Bonnie's price, but the cold cost on `down` is about
     2.1. At the phone's byte-bound r about 24 the cold row is the one that matters. D2 decides.
3. **Row skip (Ebony's proxy-ranked FFN rows) is parked.**
   - On the host, half the short gate/up rows cost as much as all of them, cold (E6).
   - The `down` input frontier sits between 60 and 70 %.
   - It returns only if D2's cold row on the phone shows that skipped short rows save time.
   - The proxy side file joins the owner questions below.
4. **Y1c (`--no-repack`) is a declared numerics change** with a negligible PPL delta (*host*). It is the cheapest
   way to put a real 4 GB body on the phone with no anonymous copy, so D1 uses it as a **residency test**, not as a
   speed configuration. Before any device use, its full -b 512 and -ub 1 runs go into that round's budget
   accounting.
5. **Owner questions,** to be asked only after the R68 report arrives (owner's standing instruction):
   - (a) Is a phone-built, sha-keyed kernel-layout side file (Y1a) a weight-derived artifact, like the calibration
     tables?
   - (b) Is twice the storage acceptable?
   - (c) Does the answer also cover non-stock layouts (a column-major copy, a 1-bit proxy file)?
6. **No goal change.** The best measured-and-priced path in this round is about 7.4 to 8.5 tok/s on a 4.0 GB body
   (*estimated*). 15 tok/s stays out of reach by the arithmetic, and the goal is not raised.

## 4. Next device items (for the kit after R69; each gets its own pre-registration and depends on R69's outcome)

| # | item | condition |
|---|---|---|
| D1 | a real Q4_0 7B body with `--no-repack` + exec map + touch, both models (and Yi-1.5-6B): load, residency over 20 decode passes, kill or not | R69 holds 3.4 GB (Ebony: run it from 3.4 on) |
| D2 | `p2col`-style rows for K1 (TR 128) and Y1b, hot and cold, on a pinned A78, plus short-row skipping | always |
| D3 | `getprop ro.lmk.*`, zram size, PSI lines in the kit header | always |
| D4 | Y1a against Y1b, ABBA within one run | owner's answer (a), and R69 holds 4.2 |
| E5 | real gemv with 2 and 4 streams and PRFM distances | R69's X1 gain ≥ 1.10, or Rule D sends it |
