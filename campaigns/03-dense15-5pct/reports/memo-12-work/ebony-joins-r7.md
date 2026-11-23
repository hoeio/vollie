# c3 memo-12 work (Ebony, round 7): review of Bonnie r7 to r11, the head on the phone's path checked independently, the guess head against the int8 head, the head-on-lane rider relabelled

Date: 2026-11-23. Author: Ebony (`claude-fable-5-2`). Not committed; the lead commits. No verdict here.
Follows [ebony-joins-r6.md](ebony-joins-r6.md). Reviews [bonnie-joins-r7.md](bonnie-joins-r7.md) to
[bonnie-joins-r11.md](bonnie-joins-r11.md): r11 and both of its scripts read in full; r7 to r10 read in their
"answer first" sections and the parts quoted below. I have not seen Tom's KleidiAI stock result; nothing here uses it.

Labels as before. Triples are Yi / Llama-2 / DeepSeek. New *host* numbers: `ebony_head_int8.py` (new) on Tom's
32-prompt dumps, and `ebony_hmax.py` extended by one line and re-run on the three served files (two wikitext
chunks, two threads, low priority). No Metal, no phone.

## 0. Answer first

1. **Bonnie's head proxy is right; my own implementation, written from the source, gives her numbers to the last
   digit.** Change of −log p(dumped token), KleidiAI's head path against ggml's: **+0.00536 on Yi (standard error
   0.00098), +0.00008 on Llama-2 (0.00022)**. Hers: +0.00536 and +0.00008.
2. **It does not explain my Llama-2 divergence of r6, and it should not.** The head alone gives a mean KL
   divergence of 0.00013 on Llama-2. I measured 0.0009 to 0.0012 between the paths on the whole model. So on
   Llama-2 most of the token-by-token difference comes from the body. It has no direction (PPL ratio 0.9999),
   which is what she predicts for Llama-2; but it is not "a few rounding ties, negligible" per token.
3. **The missing prediction, DeepSeek: it is in Yi's class.** The head's input has a range of **47.7 × its rms** on
   DeepSeek, 51.6 on Yi, 15.8 on Llama-2 (*host*, median over 1,024 wikitext positions). One activation scale per
   vector costs DeepSeek what it costs Yi. **My band for the stock shift on the phone's path: Yi +0.4 to +1.5 %,
   DeepSeek +0.3 to +1.2 %, Llama-2 within ±0.05 %.** Hers for Yi is +0.3 to +1.0 %.
4. **I withdraw my r6 estimate "about ±0.05 % in the mean" for Yi and DeepSeek.** It was read off Llama-2, the one
   subject the head rule does not hurt. Her r9 also shows the rule I asked for already exists (campaign 2, R61:
   two references side by side, a setting must be inside both).
5. **A guess head scores the same against the phone's head** (my question 4): Llama-2 0.4759 → 0.4759, Yi
   0.3554 → 0.3534 (*host*, proxy). No change to any head price.
6. **The Q6_K head on the lane is a numerics change against the phone's stock, and it should gain budget, not
   spend it** (my question 7): it replaces int8-per-row weights and one-scale activations by the file's Q6_K
   weights and float activations. Relabelled in §4. The same holds for any head knob; Bonnie prices that in r12.
7. **Kit r35 already carries the q8 line:** `x4 kai log text=` holds every log line containing "kleidiai", so
   `primary q8 kernel feature` arrives with R75 if the library prints it. No rider is needed.

## 1. Review of Bonnie r7 to r11

| round | her claim | my position |
|---|---|---|
| r7 | the packed side file is the served bytes re-ordered and XOR-ed, no numerics; one new allocation path and a skip in `set_tensor` | agree. It is what I meant in r6 §5 ("the runtime's own derived layout, written once") |
| r7 | owner question (a) is answered by rule 1 of `COMMON_RULES.md`; (b) is the real choice | agree; I read the rule's sentence again and it says so. **One addition:** the head in that file is the int8 re-make, a lossy derived artefact. If the head knob of r11 moves the head out of KleidiAI, the side file is body only and 0.13 to 0.42 GB smaller |
| r7 | half-precision accumulation follows 2.0e-4 × √(K ÷ 2): 0.0091 and 0.0148 against my simulated 0.0091 and 0.0144 | agree; it is the random walk I used for my estimate before simulating |
| r7 | two additions to my eight conditions: run the shadow check on the lane's own row range; name the KleidiAI version in the self-check's message | agree, both |
| r8 | one KleidiAI buffer per model, 3,221 / 3,599 / 3,658 MiB | agree; Llama-2's 3,599.38 MiB is in my r6 log (*host*) |
| r8 | the head grows 22 % as int8 | agree (*source*, the lines I read for r6) |
| r8 | map the side file executable: R69's 4.2 GB read-only map held in one run of two, the executable map in both | agree with the reading; two runs is thin, and her arm tests both |
| r8 | the residency arm `x5w`, `x5s1` to `x5s5` | agree with the design. Not checked: her bands. One request: `x5s4` and one lane step should also run with the head's bytes left out, since §4 below may move the head |
| r9 | the terms already say what to do: budgets stay absolute from the build without KleidiAI; stock is measured again with it; inside both | **agree; this replaces my r6 §3.4, first row** |
| r9 | the M4 picks dotprod for Q4_0 and i8mm for the 8-bit head | agree; my r6 log shows both lines |
| r9 | pack rate 4.1 to 4.4 GB/s on a laptop thread; first run 8 to 18 s | not checked |
| r10 | every campaign-3 perplexity figure is from the repack build | agree |
| r10 | 0006 does nothing on a KleidiAI build as written | agree with the reasoning (*source* not re-read by me) |
| r10 | for Q4_0 the two paths differ only in rounding of exact half-way activations | **agree on the arithmetic, disagree on the size it implies.** See §2.2 |
| r11 | Granite's head was in KleidiAI in R61 (541.08 MiB = 392.9 + 148.1) | agree with the arithmetic; not re-derived |
| r11 | the head path differs in weights **and** in activations (one scale and zero point per vector) | **agree; I had named only the weights in r6.** Checked in `kai_lhs_quant_pack_qai8dxp_f32.c`: minimum and maximum over the whole vector, 255 steps, a rounded zero point, round half away from zero |
| r11 | Yi +0.3 to +1.0 %, Llama-2 ±0.05 %, DeepSeek unknown | Yi and Llama-2 agreed (mine a little higher on Yi, §2.3); DeepSeek filled in §2.3 |
| r11 | per-chunk spread should be far smaller in a dense model than OLMoE's 1 to 2 % | **not on the three chunks I have** (§2.2): −0.45 %, +0.62 %, −0.19 % on Llama-2 |
| r11 §2 | the all-GPU emulation arm needs only a two-line decline in KleidiAI; the mixed arm 60 to 80 lines | agree with the design; her trap (head on different paths in the two builds) is the one that would bite |

## 2. The head on the phone's path, checked independently

### 2.1 Same numbers from a second implementation (*host*, proxy)

`ebony_head_int8.py`: Tom's CPU dumps of the 32-prompt set (the state the head reads, per decoded row), the served
Q6_K head dequantised with gguf-py, quantisers written from `kleidiai.cpp` lines 1515 to 1580 and
`kai_lhs_quant_pack_qai8dxp_f32.c`. Float matmul; the kernel's integer arithmetic is not simulated.

| | Llama-2 | Yi |
|---|---|---|
| rows; base check (float head picks the dumped token) | 3,638; 0.9995 | 3,563; 0.9921 |
| hidden state range ÷ rms, median | 10.9 | 41.3 |
| change of −log p(dumped token), KleidiAI − ggml (standard error) | **+0.00008** (0.00022) | **+0.00536** (0.00098) |
| head-only KL divergence between the paths: mean; 99.9 %; max | 0.000127; 0.0130; 0.0148 | 0.003581; 0.0593; 0.0932 |
| the two heads pick the same token | 99.73 % | 97.90 % |

- Her values for the first three rows: 3,638 and 3,563; 0.9995 and 0.9921; 10.9 and 41.3; +0.00008 and +0.00536.
  Identical. One difference in method that did not matter: her activation rounding is to even, the source rounds
  half away from zero.
- Yi's shift is 5.5 standard errors from zero. Llama-2's is zero.

### 2.2 Does it explain the Llama-2 divergence of r6? No

| | mean KLD | same top-1 |
|---|---|---|
| head only, proxy on chat rows (§2.1) | 0.00013 | 99.73 % |
| whole model, repack against KleidiAI, `-b 512`, 3 wikitext chunks (r6) | 0.00091 | 98.82 % |
| whole model, repack against KleidiAI, `-ub 1`, 1 chunk (r6) | 0.00116 | 98.04 % |
| whole model, repack `-ub 1` against repack `-b 512`: **same head path on both sides** (r6) | 0.00072 | 98.82 % |

- The head accounts for about a tenth of the measured divergence on Llama-2. Caution: the proxy is on peaked
  chat rows and the measurement on wikitext, where the same logit change gives a larger divergence; with that the
  head's share could be a third. It is not most of it.
- The last row has no head difference at all and shows 0.0007. So differences in the body that have no direction
  still move single tokens: per chunk the two paths differ by **−0.45 %, +0.62 %, −0.19 %** on Llama-2 (*host*,
  from the cumulative values of r6), while the three-chunk mean differs by −0.01 %.
- Consequences:
  1. Her mean prediction for Llama-2 stands (no direction).
  2. Her expectation that a dense model's per-chunk spread is far below OLMoE's 1 to 2 % is not what three
     chunks show. A 40-chunk mean then carries about ±0.1 % of noise (0.5 % ÷ √40, *estimated*). **A shift on Yi
     or DeepSeek under about 0.2 % cannot be read from 40 chunks**; her +0.3 to +1.0 % can.
  3. "Identical per chunk" between the paths is not available even for the Q4_0 body, whatever the arithmetic
     says about ties. What produces 0.0007 inside one build across batch settings is not explained by either of
     us. It is next question 3.

### 2.3 DeepSeek: the missing prediction (*host*)

No hidden-state dump of DeepSeek exists. The driver of the effect is one number, the range of the head's input
against its rms, and that can be read from a forward pass. `ebony_hmax.py`, one line added, two wikitext chunks:

| head input, range ÷ rms per position | Yi | Llama-2 | DeepSeek |
|---|---|---|---|
| wikitext, 1,024 positions: median; 90th percentile; max | 51.6; 61.2; 70.2 | 15.8; 27.1; 89.0 | **47.7; 52.2; 59.0** |
| chat dumps (§2.1), median | 41.3 | 10.9 | no dump |
| largest \|input\| | 191.1 | 81.3 | 18.4 |

- **DeepSeek's final state is as wide as Yi's on every position.** 255 steps over 48 × the rms leave an ordinary
  component about five levels, as on Yi.
- On wikitext the ratio is 1.25 to 1.45 × what it is on the chat dumps, for both models that have both.
- Bands (*estimated*): the proxy's shift on Yi is +0.54 % of mean log-probability on chat rows. Wikitext rows are
  flatter and wider, which pushes the shift up; the proxy ignores the kernel's zero-point arithmetic, which I
  take as neutral. **Yi +0.4 to +1.5 % (centre +0.8); DeepSeek +0.3 to +1.2 % (centre +0.6; its head has 102,400
  rows, which I do not credit either way); Llama-2 within ±0.05 %.**
- If that holds, two of three subjects spend a tenth to a quarter of the +5 % budget in the unmodified phone
  runtime, by the R61 rule's second reference. That makes the head path the first numerics question of the
  campaign, before any lane.

### 2.4 Where Yi's wide final state comes from (next question 4, done at once; *host*, the same dumps)

| | Yi | Llama-2 |
|---|---|---|
| largest dimensions, mean \|x\| ÷ rms | 3889: 18.8; 2292: 14.4; 3017: 10.2; 2326: 8.8; 3254: 8.5 | 3241: 3.9; 1360: 3.2; 1927: 2.9 |
| the dimension holding the row's maximum | 3889 on 50 % of rows, 2292 on 13 % | 3241 on 17 % |
| range ÷ rms without the top 0 / 4 / 8 / 32 dimensions | 41.3 / 27.1 / 22.5 / 9.7 | 10.9 / 10.2 / 9.7 / 8.3 |
| one-scale activation error ÷ rms; the same with the top 4 dimensions carried exactly | 4.67 % → **2.50 %** | 1.24 % → 1.15 % |

- **On Yi it is a small fixed set: about thirty dimensions, four of which do half the damage.** With those four
  kept exact, one scale per vector is as good as ggml's blocks of 256 (2.39 %, Bonnie r11).
- This is why a block scale works (the outliers spoil only their own blocks), and it says a cheaper repair than
  moving the head exists in principle: the same int8 kernel with a handful of columns added in float. It would be
  a numerics change towards the file, model-generic if the columns are found at load from the head's own
  statistics or a calibration pass. Not priced; it competes with Bonnie's head knob, which needs no new code.
- DeepSeek has no dump; E-T10 gives its list.

## 3. A guess head against the phone's head (my question 4; *host*, proxy)

Medusa-1 heads, full rank (`head3000-full.pt`, `head2117-full.pt`). Target at row j + 1 and guess at row j,
each through the same head:

| | Llama-2 | Yi |
|---|---|---|
| float Q6_K head for guess and target (Tom's measure) | 0.4759 (Tom: 0.476) | 0.3554 (Tom: 0.355) |
| phone's head for guess and target | **0.4759** | **0.3534** |
| float head for the guess, phone's head for the target | 0.4756 | 0.3543 |
| the two heads give the same guess | 98.35 % of rows | 96.41 % |
| the phone's head picks the dumped token | 99.73 % | 97.87 % |

- **No change on Llama-2, −0.002 on Yi.** Inside my r6 band (−0.01 to 0.00). The guesses differ on 2 to 4 % of
  rows, but the targets move with them.
- So the head tables of r4 §2.1 and Tom's top-1 values stand on the phone's path. The `tok` heads were not
  re-scored (they need the embedding rows); I expect the same.
- Limit: the targets here are "what the phone's head would pick from the host's hidden state". The phone's own
  hidden states differ too (§2.2); that cannot be simulated from dumps.

## 4. The Q6_K head on the lane, relabelled (my question 7)

Bonnie r4 §2 priced a head on the GPU (upstream's Q6_K gemv) at +0.15 / +0.06 / +0.22 tok/s after C7 and said
what its emulation needs. Its label has to change, because the reference changed:

| | weights | activations | against the phone's stock head |
|---|---|---|---|
| phone's stock (KleidiAI) | int8, one scale per row | int8, one scale and zero point per vector | – |
| host reference so far (repack) | Q6_K as stored | int8, one scale per 256 values | closer to the file |
| head on the lane (upstream Q6_K gemv) | Q6_K as stored, products in half | float | **a numerics change; closer to the file than either** |

- **Label: numerics change, declared, not bit-identical.** It needs the `-b 512` and `-ub 1` checks like any
  other, on the KleidiAI build (her r11 §2.5, second row).
- **Expected sign: it gives budget back.** Against the phone's stock: Yi −0.4 to −1.5 %, DeepSeek −0.3 to −1.2 %,
  Llama-2 ±0.05 % (*estimated*: the shift of §2.3 undone, less the kernel's half-product error of about 5e-4,
  which is small beside it).
- That changes its rank. As a speed rider it was worth 0.06 to 0.22 tok/s. As a numerics item it may be worth
  0.5 to 1 % of perplexity on two subjects, which is room for C5 or C7.
- **The same is true of every way of taking the head out of KleidiAI's 8-bit chain** (the repack head knob
  `VOLLIE_HEAD2_REPACK`, a head on the lane, a two-stage head whose second stage is exact). Each is "a numerics
  change against the phone's stock that moves towards the file". Under the R61 rule each must be inside both
  references; none can fail the first because of the head. Bonnie prices the knob in r12; I do not.
- One rule question for the lead, not for me to answer: the campaign's stock is "the served GGUF run by the
  unmodified runtime". If the unmodified runtime's head costs Yi 0.8 %, a knob that removes it lowers the
  phone-path PPL **below stock**. The budget is a ceiling, so nothing forbids it; but the pre-registration should
  say which head the "stock" column was run with.

## 5. Kit r35 and the q8 line

[tom-r35-build.md](tom-r35-build.md) lines 100 to 101: `x4 kai kernel= features= q4_buft=` gives the Q4_0 entry
and the HWCAP bits (dotprod, fp16, sve, i8mm, sme); `x4 kai log text=` copies every ggml log line containing
"kleidiai" during the backend's load, to 1,500 characters. The library prints `primary q4 kernel feature` and
`primary q8 kernel feature` on adjacent lines (*host*, my r6 log), so the q8 entry is in that text.
- Band: P(`kernel=DOTPROD`) 0.95; P(the log text shows the q8 line reading DOTPROD) 0.85; P(`features` lists
  i8mm) 0.03.
- If `kernel=unseen` (the log line is at a level the phone's build drops), `features=` still decides it.

## 6. Lists (additions only)

| # | test | where | cost | closes when |
|---|---|---|---|---|
| E-T10 | DeepSeek's hidden states on the 32 prompts with Tom's dump tool, then both head scripts | host, CPU, minutes | – | proxy shift under +0.001 → my DeepSeek band is wrong and the range ratio is not the driver |
| E-T11 | In Tom's 40-chunk KleidiAI stock runs: per-chunk spread on each subject beside the mean | host, free | 0 | spread ≥ 0.3 % per chunk on Llama-2 → the body's undirected noise is real and 40 chunks resolve only shifts above 0.2 % |
| E-T12 | KL mode, repack `-ub 1` against repack `-b 512`, with the head forced to float on both sides (or the head excluded by comparing the final hidden states) | host, minutes | one small tool change | divergence stays near 0.0007 → it is the body; then find which op differs between batch settings |

No new try item. E-Y1 (the W-position kernel) keeps its precondition from r6: a KleidiAI host build with the
8-bit kernels on the dotprod entry.

## 7. Next questions, ranked

| # | question | state |
|---|---|---|
| 1 | Tom's KleidiAI stock result against the three bands of §2.3, with the per-chunk spread | waiting for his file; mine to read next |
| 2 | DeepSeek's hidden-state dump and the proxy (E-T10) | host, minutes, CPU; Tom's tool |
| 3 | What differs inside one build between `-ub 1` and `-b 512` to give a divergence of 0.0007 (E-T12)? | *source* first: the gemv and the tile in ggml's repack path, the head's Q6_K gemv against its gemm, the attention's float matmuls |
| 4 | Where do Yi's and DeepSeek's wide final states come from: a few fixed dimensions (then a per-block scale fixes it and a lane's float input is exact), or many? | host, minutes on the dumps: the top dimensions by magnitude and how stable they are across rows |
| 5 | Does the same one-scale rule touch anything else on the phone's path (Q8_0 tensors, the F32 chain)? | *source*; the served files have no Q8_0 body tensor, so probably not |
| 6 | R75 free reads (r5 §7 item 2, r6 §6 item 6) | R75 |
| 7 | H-max on the official prompt with chat templates; greedy dump rate at 32 sequences | idle laptop |

## 8. Sources (2026-11-23)

Repository: `lab/llama.cpp-base/ggml/src/ggml-cpu/kleidiai/kleidiai.cpp` (lines 1515 to 1580);
`lab/llama.cpp-c1h6/build-kai/_deps/kleidiai-src/kai/ukernels/matmul/pack/kai_lhs_quant_pack_qai8dxp_f32.c`
(the function body); `lab/c3/head/{llama2,yi}/eval.{rows,h16}`, `head3000-full.pt`, `head2117-full.pt`;
`tools/c3_head_train.py` (the block's form and the pair rule); [tom-r35-build.md](tom-r35-build.md) lines 100 to
101; `campaigns/COMMON_RULES.md` rule 1; Bonnie's r7 to r11 and `bonnie_head_requant.py`, `bonnie_head_paths.py`.
No external source this round.
