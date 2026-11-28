# Pre-registration R77: UD and Y1b kernel cells on the phone (c3 probe kit r38)

This file is committed and pushed **before** the APK is uploaded, and it is not edited afterwards. Any change goes
into `R77-amendment-1.md`. The tag `prereg-R77` is the owner's to create.

- **Date / author:** 2026-11-28, clyde (lead).
- **Why this round exists:** device rounds are held until a real breakthrough (owner, 2026-11-22). After E-T25 (UD inside
  the budget on all three subjects, *host*) and Tom's host kernel tests (`reports/memo-13-work/tom-ud-kernels.md`), the lead
  offered a small device probe of the deciding kernel costs; the owner chose it (2026-11-28, "1로 가죠"). R73 stays reserved for
  the first real decode, so this round is **R77**.
- **Research method:** pre-2026-11-23 (owner, 2026-11-26). No new researcher instance; bands are the lead's, from committed
  work (R72 L1 *device*, tom-ud-kernels *host*, bonnie-ud-check break-even lines).
- **Agent models:** lead (clyde) `claude-opus-5-5`, this file; bonnie `claude-fable-5-2` (`bonnie-cpu-ladder.md`,
  `bonnie-ud-check.md`); ebony `claude-fable-5-2` (`ebony-new-routes.md`); tom `claude-opus-5-5` (patches 0010, 0011, kit r38,
  `tom-et25.md`, `tom-ud-kernels.md`, `tom-r38-build.md`); annie `claude-sonnet-5-5` (books).
- **Subjects:** Yi-1.5-6B-Chat, Llama-2-7b-chat, DeepSeek-LLM-7B-chat, as shapes and as real UD masks (four per subject). No
  model is loaded. Triples are Yi / Llama-2 / DeepSeek.
- **Arms of the campaign:** R3 (`down` through K1) is an **A′** lever (needs the `down` copy, CAMPAIGN.md 2026-11-27); R1
  (`gate` row skip) and Y1b are **choice A** levers. Not a claim round.

## 0. Build under test

Kit r38 (`reports/memo-13-work/tom-r38-build.md`), files in `lab/dist-c3/r38/`; sha256 checked by the lead before this commit:

| item | sha256 |
|---|---|
| `vollie-c3-probe-r38.apk` (16,711,466 B, versionCode 38, same certificate as r37) | `c28673904cfa2be817ed9607c174bda9f36849f06164919d8dfc292456544111` |
| `BUILD-r38.txt` | `6b922b213f744fd89c73c8e92141a57f24978c8187726eac318f3593b0a44614` |
| `SHA256SUMS` | `15958f1de88e067221f4b4459d12d569709b65db57b97e6c520a1324b1e98e22` |
| `patches/llama.cpp-c3/0012-r38-cells.patch` (on 0009) | `a5945fc292af4c8e15506ecdc716a5de1fd51ab5e93cd06bb2534190f430bc13` |
| `patches/app-c3/0008-kit-r38-probe.patch` (on 0007) | `f6ea7c6a9126330451868b1f078a72d253aa2853c48941e6faa4c4061bb287fa` |
| `tools/c3_probe_parse.py` (R77 section) | `3fb2be126b9f6e2033ac64eab7e57111c67d79d1c8f74aa1e0cdb44b614713ee` |
| `lab/c3/bench/r38_mkmask.py` (mask generator; copied into the memo-13 work folder at commit) | `d52deeefa2b5680644540bd890d2a780f7f092f3651bd9c20d1ddbe0a85d6072` |
| `libllama.so` / `libapp.so` in the APK | `71b66042…2622` / `2e733d01…fff4` (full values in tom-r38-build §3) |
| CPU module (byte-identical to r37) | `46e0735d2824ba6f3a156e219e058a701f6024fdbecbb2ef937b3127115a9694` |

No numerics change (host per-chunk PPL r37 vs r38 identical at `-b 512`, `-ub 32`, `-ub 1` on Llama-2, short runs).

## 1. Device state

As R76: power saving off, charger out, deep-sleeping label recorded, at least 30 min rest with the screen off, no other probe
first; r16 gate before each block; kit r38 installed over r37.

## 2. What is measured

Blocks `udk1` (K1 on `down` 4096 × 11008 at the real UD masks, keep 0.5), `udr1` (dense Y1b and Y1b on the real 70 % `gate` row
lists, 11008 × 4096), `y1bd` (Y1b and ggml's plain / repack / KleidiAI mat-vec on 4096², 11008 × 4096, 4096 × 11008), each on
cpus 4-7 (four pinned threads) and on cpu 4 alone, in chains stock | lever | stock. Fields: `ud trip` (`t_vs_stock`, `k`,
`stock_diff`), `ud arm`, canary, and the parser's "R77 inputs".

## 3. Predictions (*estimated*; centre with band; P = lead's probability that the result is inside)

"Median" = median over the three subjects' triplet values (or over shapes where stated) in the named core set. Scored on the
4-7 set; the cpu-4 set is reported beside and scored only for D1.

| # | quantity | field | band | P |
|---|---|---|---|---|
| D1 | K1 `k` on `down`, cpu 4 alone, median | `udk1` trip `k` | 2.6 (2.2 to 3.1); R72 L1 *device* 2.621 with a random mask | 0.7 |
| D2 | K1 `k` on `down`, 4-7, median | same | 2.4 (1.8 to 3.2) | 0.6 |
| D3 | K1 `t_vs_stock` on `down`, 4-7, median | `udk1` trip `t_vs_stock` | 1.30 (1.00 to 1.75) | 0.6 |
| D4 | `rows` `k` (Y1b on the 70 % `gate` rows), 4-7, median | `udr1` trip `k` | 2.6 (2.0 to 3.4); host cold 2.57 to 2.60 | 0.55 |
| D5 | dense Y1b `t_vs_stock` on `gate` 11008 × 4096, 4-7 | `udr1` | 1.65 (1.3 to 2.1) | 0.55 |
| D6 | ggml plain (`--no-repack`) `t_vs_stock`, 4-7, median over the three shapes | `y1bd` | 2.0 (1.6 to 2.5) | 0.6 |
| D7 | Y1b ÷ ggml plain, 4-7, median over the three shapes | parser "R77 inputs" | 0.85 (0.65 to 1.05) | 0.55 |
| D8 | ggml KleidiAI ÷ ggml repack, 4-7, median over the three shapes | `y1bd` | 1.0 (0.90 to 1.10) | 0.7 |
| D9 | identity lines all equal | `ud identity` | yes | 0.95 |

## 4. Decision rules (scored as written; break-even lines from `tom-ud-kernels.md` §4, *estimated*)

On the 4-7 set (the decode's thread set), median over subjects:

- **R3 (A′):** K1 `k` ≤ 3.15 → R3 alive at every clock priced (break-even 3.15 at C 40, 4.5 to 4.7 at C 50); A′ UD goes to the
  build list. 3.15 < `k` ≤ 4.5 → alive at C 50 only; the lead reports the price and the owner decides. `k` > 4.5 → **R3 dead**.
- **R1 (choice A):** `rows` `k` ≤ 1.90 → R1 alive even if skipped rows save no bus time; 1.90 < `k` ≤ 2.18 → alive only if skipped
  rows save bus time, reported; `k` > 2.18 → **R1 dead** (a loss at C 40).
- **Y1b (choice A):** Y1b ÷ ggml plain ≤ 0.90 on the median of the three shapes **and** ≤ 1.00 on each shape → Y1b goes to the
  build list for path A. Otherwise Y1b is parked.
- Prices after the round are recomputed by the lead with `lab/c3/ud_price.py` at the measured `k` values (*estimated* from
  *device* inputs).

### 4.1 Void (kit)

- APK hash differs from §0: round void. Power saving on or charger plugged: that block void. Process killed: that block void.
- A triplet with `stock_diff` > 0.03: that triplet void; the median uses the rest, and a cell with no valid triplet is "not
  measured".
- An arm with `ramp_ok=0`, an off-core label, or a canary with `thermal_pwrlevel` > 0 in its triplet: that triplet void.
- Any identity line not all equal: the cells of that block are void (a kernel bug, not a cost).

**Not voids:** cool start not met (reported), CPU caps where all three arms of a triplet share the same cap (reported beside;
R76 voided a whole block for a cap, which this rule avoids by matching within each triplet). Misses are written as misses.

## 5. Claim rule and meaning

Not a goal claim; no token rate is measured. Whatever the outcome, 12 tok/s on all three stays out of reach by priced levers
(STATUS 2026-11-27: choice A best 6.73 / 6.13 / 5.88, A′ best 8.08 / 6.94 / 7.51, *estimated*). R77 decides which of R3, R1 and
Y1b are worth building toward the first decode round (R73), and at what price.

## 6. The run, for the owner

1. After this file is pushed, create tag `prereg-R77` on its commit; give the lead an HF write token for `r38/` (or upload
   `lab/dist-c3/r38/` yourself to `HoeioUser/vollie-kits` as `r38/`).
2. Download `r38/vollie-c3-probe-r38.apk`; check sha256
   `c28673904cfa2be817ed9607c174bda9f36849f06164919d8dfc292456544111`; install over r37.
3. Phone state as §1 (30 min rest, screen off, charger out, power saving off, no other probe first).
4. Research kit page: run the r38 probe (R77). About 2.3 to 5 min of blocks plus three gates (*estimated*).
5. Save the c3 probe to Downloads and send the JSON.
