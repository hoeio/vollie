# c3 memo-13 work (Bonnie): price check of Ebony's bus items (R1, R3, UD, R2)

- **Date:** 2026-11-27. **Author:** bonnie, `claude-fable-5-2`. Not committed. No verdict here. Checks
  [ebony-new-routes.md](ebony-new-routes.md) §2, §3, §6, §7 against [bonnie-cpu-ladder.md](bonnie-cpu-ladder.md).
- **Labels.** *device*, *host*, *estimated*, judgement. Triples are Yi / Llama-2 / DeepSeek.
- **Script:** [bonnie_ud_check.py](bonnie_ud_check.py) (`python3 -I -B`; loads `bonnie_cpu_ladder.py`). Its
  self-check reproduces her §3 rows (−8.9 / −8.9 / −8.3, −13.0 / −13.0 / −12.0, −21.9 / −21.8 / −20.3, −2.2 /
  −2.6 / −2.5, −23.5 / −23.9 / −22.2) from her inputs, so every difference below is an input, not the model.
- Nothing was measured. No web search. Every price is *estimated*.

## Answer first

1. **Her arithmetic is right and her model is mine.** Her prices fall by about 12 % on my inputs, for three
   stated reasons (§3). UD re-priced at full clocks: **−19.1 / −19.0 / −17.6 ms on the packed column,
   −22.8 / −22.7 / −21.0 ms on path A** (hers −21.9 / −21.8 / −20.3).
2. **New top stacks, claim decode at C 50** (*estimated*):

   | | Yi | Llama-2 | DeepSeek | made of |
   |---|---|---|---|---|
   | path A, was A4 | 7.70 | 6.22 | 6.71 | |
   | **path A with UD** | **8.08** | **6.94** | **7.51** | Yi and Llama-2: UD at width 1; DeepSeek: UD with width 2 on lookup matches |
   | packed, was P4 | 9.40 | 7.55 | 7.44 | |
   | **packed with UD** | **9.40** | **7.55** | **8.17** | Yi and Llama-2: P4 unchanged (UD under width 2 loses 0.2 and 0.06); DeepSeek: UD with width 2 |

   UD and width 2 mostly exclude each other, as she said. UD adds +0.4 / +0.7 / +0.8 on path A and 0 / 0 / +0.7
   on the packed column. Still 3.9 / 5.1 / 4.5 and 2.6 / 4.4 / 3.8 tok/s under 12.
3. **Only the row rule (R1) is choice A as adopted.** The `down` copy (0.76 to 0.81 GB) passes the app
   rule's own arithmetic but the app refuses it today, so it needs the same reading as my next-question 4. The
   scale index needs the whole body re-laid (3.0 to 3.5 GB): it fails the rule as anonymous memory and exists
   only under choice C (§2).
4. **Two of her three compute inputs do not hold as written.** The row kernel's 1.15 is for a layout that does
   not exist; in place it is 1.45 hot, and memo-05's E6 measured skipped 2 KB rows at 1.9 to 3.4 cold (*host*),
   which is why row skip was parked. The scale table's +3 % is +4 to +16 % by my count. K1's 2.1 stands, but its
   layout reads every block's scales, so `down` at half its columns is 0.556 of its bytes, not 0.5 (§3).
5. **Her screen does not cover the owner's rule.** UD is a numerics change that must be inside +5 % at `-b 512`
   and `-ub 1` with `llama-perplexity`. The screen is a float pass on 6 chunks with the context half computed
   dense. Her E-T25 is the right test and is not run (§4). P(UD inside the budget at both settings on all
   three): 0.55 (judgement).

## 1. Re-priced on the ladder (C 50; base and inputs as bonnie-cpu-ladder §1)

Bases: packed P2 (7.13 / 6.28 / 6.35), path A3 (6.77 / 5.96 / 6.04; Y1b body, head-only copy). ms per token
against the base, then tok/s. Gate rows read in place by a four-row kernel on a row list at 1.45; `down` through
K1 at 2.1 on its own layout; 0.5 ms per token per top-k selection (*estimated*, not in her prices).

| item | packed: ms | packed: tok/s | path A: ms | path A: tok/s |
|---|---|---|---|---|
| R1, up-first, 30 % of `gate` rows skipped | −7.6 / −7.6 / −7.0 | 7.56 / 6.61 / 6.66 | −9.5 / −9.4 / −8.8 | 7.26 / 6.33 / 6.40 |
| R1 if skipped rows save no bytes (memo-05 E6's host reading) | +0.6 / +0.6 / +0.6 | 7.10 / 6.26 / 6.33 | −1.3 / −1.3 / −1.1 | 6.83 / 6.00 / 6.09 |
| R3, `down` at half its columns | −11.4 / −11.4 / −10.6 | 7.80 / 6.79 / 6.83 | −13.3 / −13.2 / −12.3 | 7.48 / 6.49 / 6.55 |
| **UD** | **−19.1 / −19.0 / −17.6** | **8.32 / 7.18 / 7.20** | **−22.8 / −22.7 / −21.0** | **8.08 / 6.94 / 6.97** |
| R2, scale index, arithmetic × 1.08 (× 1.03 hers; × 1.16) | −1.7 / −1.9 / −1.8 (−2.4 / −2.8 / −2.7; −0.4 / −0.5 / −0.4) | 7.22 / 6.36 / 6.43 | not available | – |
| UD + two-stage head, width 1 | | 8.43 / 7.20 / 7.36 | | – |
| UD + R2 + two-stage head, width 1 | | 8.51 / 7.27 / 7.43 | | – |
| UD + width 2 (rule T; DeepSeek lookup at 2 or more) | | 9.20 / 7.49 / 8.17 | | 8.04 / 6.46 / 7.51 |

- On strict path A (no copy at all, plain head): R1 alone on A1 gives 6.73 / 6.13 / 5.88 (from 6.31 / 5.78 / 5.57).
- By clock state, UD at width 1: packed 8.27 / 7.14 / 7.16 at C 40 and 7.65 / 6.74 / 6.78 at C 33; path A
  7.43 / 6.40 / 6.47 and 6.63 / 5.69 / 5.79. At the 1,555 cap the packed UD keeps half its price (−9.2 / −10.3 /
  −9.5 ms); she has −15.4 / −16.5 / −15.4 with the 1.15 row kernel.
- Why UD under width 2 is small: the two positions' masks are independent (her S1 count), so the shared read
  keeps 91 % of `gate` and 75 % of `down`, and K1 runs once per position at 2.1 where the W-position kernel
  costs 1.45 for both. DeepSeek gains because only 22 % of its passes are width 2.
- R2 at width 2 or at the 1,555 cap costs what it saves or more (her own note; at × 1.16 it is +7 ms at C 33).

## 2. The two copies against choice A, the app's rule and residency

| item | what it needs on the phone | app rule (`canCopyWeights`: RAM − copy ≥ 5 GB, so up to 2.75 GB) | choice A as adopted | residency |
|---|---|---|---|---|
| R1 | nothing: the file's `gate` rows in place | not involved | **yes** | as A (0.85 / 0.65 / 0.55, r17) |
| R3 | `down` in K1's strip layout, 0.81 / 0.81 / 0.76 GB | passes by arithmetic; **refused today**, because the app passes the whole file's length and then loads with no extra buffers at all | **no, as written** ("plain mapped rows"). It is the same question as a head-only copy: may the runtime make a small copy the rule's own arithmetic admits? One line in the app, one runtime knob | mapped pages fall to 2.67 / 3.02 / 3.24 GB (the file's `down` is never read); anonymous rises by 0.8 GB. Total unchanged. 0.8 GB anonymous is under what the phone held (3.4 GB, R72) and under campaign 1's surviving 2.3 GB copy. My P: as A, less 0.05 |
| R3 as a side file | the same bytes file-backed, +0.8 GB storage | not the gate | needs the side-file mechanism (r7, about 180 lines) and owner questions (a) to (c), which were never answered | as A |
| R2 | every body tensor re-laid, 3.0 to 3.5 GB | **fails** as anonymous memory | **no**; choice C only | – |
| R3 + head-only copy | 1.0 / 0.9 / 1.1 GB anonymous | passes by arithmetic | the same reading | as above |

Her sentence "the original `down` pages are then never read, so residency does not grow" holds for the total. It
does move 0.8 GB from memory the kernel can drop to memory it must swap, which is the direction the rule was
written against, at a quarter of the size that killed campaign 1's load.

## 3. Her compute inputs against the cost model

| input | hers | what the books say | mine | effect |
|---|---|---|---|---|
| K1 per touched weight | 2.1 (*device*, R72 L1: 1.992 hot, 2.239 cold) | confirmed in R72-result §3. One A78 at 2,419 MHz, keep 0.60, **4,096 inputs**; the 11,008-input hot group was flagged for drift. R72's four-thread layer cells agree with it through the reader (model 1.04 against 0.95 to 1.13) | **2.1 stands** | none; at keep 0.5 the per-block float step adds about 2 % |
| `down` bytes at keep 0.5 | × 0.5 | K1's layout holds each block's scales in every strip; a block with any kept input reads them: touched share (4 + 32 d) ÷ 36 (*host*, memo-05 T7, exact) | **× 0.556** | R3 −1.5 ms; b for UD 0.81 / 0.83 / 0.83, not 0.79 / 0.82 / 0.82 |
| row kernel on any four rows | 1.15 (judgement; "a per-row packed side layout") | that layout is a second copy of `gate` (0.8 GB) and is not built. In place: Y1b on row lists 1.46 to 1.52 hot, **1.92 to 3.36 cold, 3.2 to 3.3 on 2 KB rows** (*host*, memo-05 E6). `gate` rows are 2,304 bytes. E6 is why the lead parked row skip | **1.45 hot; cold open** | R1 −0.9 ms hot. If the phone behaves as the M4 did cold, R1 is worth 0 on the packed column |
| scale table | +3 % arithmetic | per 128 weights the packed kernel loads four scales as one vector. An index needs four extracts (scalar) and four lane loads: +1 to +4 vector slots on 25 | **+4 to +16 %, centre +8 %** (hand count, not compiled) | R2 −1.7 / −1.9 / −1.8, not −2.2 / −2.6 / −2.5 |
| selection | not charged | two top-k selections over 11,008 values per layer per token | 1 ms per token for UD (*estimated*) | −1 ms |
| "stays bus-bound, k 0.96 to 0.97" | at C 40, KR 1.15 | with 1.45 rows k is 1.01 to 1.02; still bus-bound at C 40 and above, compute-bound at C 33 | agree at full clocks | UD halves at the 1,555 cap |
| FFN share of body bytes | 0.782 / 0.668 / 0.668 | agrees with my shape count (`down` 22 to 26 % of the body) | agree | – |

Her §6 challenge ("R72 parked K1 as a uniform rule; `down` alone at keep 0.5 was never priced") is correct. My
§5.3 sentence "the kernel that skips them costs what it saves" is true of the uniform rule and wrong for this
cell: on one matrix with a free exact mask the body's arithmetic rises 1 to 2 % while its bytes fall 10 to 12 %.

## 4. Numerics and the owner's rule

- UD changes numerics. It spends the one +5 % budget, per subject, against the absolute budgets of the S3
  freeze and against same-build stock × 1.05 (option (a), 2026-11-23).
- **Is it batch-dependent?** As a rule, no: each position picks its own channels from its own `up` and h, at
  any batch size. As built it becomes batch-dependent the moment prefill takes another path (dense prefill,
  or a batched kernel that applies one mask per tile). Either way the check is the same: full set,
  `llama-perplexity`, `-b 512` and `-ub 1` (and `-ub 32` if tiles exist), both inside.
- **What her screen covers:** a float32 numpy pass, 5 to 8 chunks, standard error 0.7 to 1.2 % on UD, and the
  **first 256 tokens of each chunk computed dense**. So the scored tokens attend to an exact context. Under the
  gate every position carries the rule, at both settings. The screen is therefore not `-b 512`, not `-ub 1`,
  and biased low by an amount nobody has measured.
- What supports it anyway: memo-05 E1, `llama-perplexity`, 40 chunks, both settings, `down` input only at 60 %
  dropped: +3.41 / +3.35 % (Mistral), +1.92 / +2.32 % (Qwen2.5) (*host*). Her D65 on the subjects is +5.1 /
  +5.0 / +3.7 %, so the subjects are less tolerant than those two, and D50 at +1.1 to +2.0 % is plausible.
- Room: the KleidiAI build's stock is already +0.50 / +0.07 / +0.69 % over the freeze (*host*, 40 chunks), so
  the absolute budget leaves about 4.5 / 4.9 / 4.3 %. UD at +2.6 / +2.1 / +2.0 % (± 1) fits with room; UD plus
  anything else on Yi does not have much.
- P (judgement): U30 alone inside at both settings on all three 0.75; D50 alone 0.75; UD 0.55. P(UD inside
  **and** its kernels deliver 70 % of the price above on all three on the phone): about 0.15.

## 5. Agree / differ

| her number or claim | mine | verdict |
|---|---|---|
| prices are differences on memo-11's model | reproduced to 0.1 ms from her inputs | **agree** (arithmetic) |
| R1 −8.9 / −8.9 / −8.3 ms, in place on path A | −7.6 / −7.6 / −7.0 packed; −9.5 / −9.4 / −8.8 on path A; 0 to −1.3 if cold rows behave as in E6 | **differ in size** (row kernel 1.45, selection); **the cold cell is open** and already has a host reading against it |
| R3 −13.0 / −13.0 / −12.0 ms | −11.4 / −11.4 / −10.6 packed; −13.3 / −13.2 / −12.3 on path A | **differ by 12 %** on the packed column (scales always read); agree on path A, where K1 also replaces a dearer kernel |
| UD −21.9 / −21.8 / −20.3 ms; 8.21 / 7.24 / 7.18 alone | −19.1 / −19.0 / −17.6; 8.32 / 7.18 / 7.20 from P2 | **differ by 12 % in ms**; tok/s close because my base has O 5 / 7 / 7 |
| UD at C 33: −15.4 / −16.5 / −15.4 | −9.2 / −10.3 / −9.5 | **differ**: compute-bound with in-place rows |
| R2 −2.2 / −2.6 / −2.5 ms; a full-clock item; needs a copy | −1.7 / −1.9 / −1.8; agree on both conditions | **differ in size**, agree in kind |
| UD + R2 −23.5 / −23.9 / −22.2 | −20.2 / −20.4 / −18.9 | **differ**, same reasons |
| K1 at 2.1 per touched weight | 2.1 | **agree** |
| arbitrary-row kernel 1.15 | 1.45 hot in place; 1.15 needs a second copy | **differ** |
| scale table +3 % | +4 to +16 % | **differ** |
| R1 needs no copy; R3 needs 0.76 to 0.81 GB; R2 is not on path A | the same; R3 is not choice A as written either | **agree**, with the app-rule note |
| "do not multiply with width 2" | UD under width 2: −0.2 / −0.06 / +0.7 against P4 | **agree**, with DeepSeek the exception |
| UD +2.6 / +2.1 / +2.0 % | not re-measured; biased low by the dense context half | **not checked**; E-T25 decides |
| P(12 on all three) under 0.002 | under 0.0005 CPU only | **agree** in substance |

## 6. For the lead (one pass; I stop here)

1. E-T25 first (her list): it is the only item that can stop UD outright. Ask for `-b 512` and `-ub 1` on all
   three and both builds, as she wrote it.
2. Fold her E-T27 into my B-T2: one bench, plain rows, four-row kernel, dense and with 30 % of rows skipped by a
   real mask, cold. Its closing line should be hers (over 1.3 per touched weight cold → R1 about −4 ms).
3. E-T28 as she wrote it, plus the byte count of K1's layout at keep 0.5 with a real mask (expect 0.556).
4. The reading asked in my next-question 4 now covers three copies: head only (0.1 to 0.34 GB), `down` in K1's
   layout (0.8 GB), both (about 1 GB).
5. R2 only if choice C is ever adopted; compile the indexed scale load once to replace my hand count.
