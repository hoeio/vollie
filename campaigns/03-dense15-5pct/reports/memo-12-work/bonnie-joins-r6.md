# c3 memo-12 work, round 6 (Bonnie): the stock Q4_0 kernel, depth-2 guesses, the numerics of a cap-following share

- **Date:** 2026-11-23. **Author:** bonnie, `claude-fable-5-2`. Follows [bonnie-joins-r5.md](bonnie-joins-r5.md);
  reviews [ebony-joins-r5.md](ebony-joins-r5.md) §0, §1 and §5 (read; §2 to §4 skimmed). Not committed. No verdict
  here. **Ebony has not reviewed this file.**
- **Labels.** *device*, *derived*, *host*, *estimated*, *external*, *source* as in r5. Triples are Yi / Llama-2 /
  DeepSeek.
- **Scripts:** [bonnie_depth2.py](bonnie_depth2.py) (`lab/venv-c3/bin/python3 -I -B bonnie_depth2.py <repo root>`;
  numpy only, reads Tom's dumps, trains nothing); prices from `bonnie_joins_r5.py` functions, run inline.
- **New measurements, all *host*:** hit clustering and a model-free depth-2 guess on Tom's saved head guesses
  (§2). Nothing ran on the phone and no model ran.

## 0. Answer first

1. **No phone record of Q4_0 on the stock (not repacked) path exists.** R72's type cells ran Q4_0 in the KleidiAI
   buffer only. By issue count the stock dot product is **2.4 × the gemv per weight**; Tom's laptop read 2.0 × for
   the kernel and 1.9 × for a whole model on one thread. I use **c 2.2 (1.9 to 2.5)**.
2. **At that cost the served file is not a usable source of rows for a width-2 lane.**
   - Overlap rows read from the served file (my r5 §4 idea): **withdrawn.** At width 2 the stock layout costs
     about 4.0 × a gemv per weight against 1.44 for the W-position kernel, and that eats the gain of following.
   - The whole CPU part from the served file (r5 §7 route 2): 4.0 to 5.0 tok/s on the CPU alone, and 8.0 to 8.7
     on a width-1 lane at K 300 against 9.2 to 10.0 with the gemv. No width 2.
   - **So for Llama-2 and DeepSeek the only file-backed route at width 2 is the packed side file,** which is
     owner questions (a) and (b).
3. **Depth 2 on Tom's dumps (*host*):** head hits cluster (Llama-2: 0.606 after a hit, 0.386 after a miss; Yi
   0.509 and 0.303). A model-free depth-2 guess (most frequent successor) is right 0.274 / 0.250 of the time
   given depth 1 was right. So memo-10's `C2` lies between **0.55 and 1.22 (Llama-2), 0.66 and 1.34 (Yi)**; it
   assumed 1.0.
4. **Width 3 on Llama-2 needs both `dev_ratio(3)` ≤ 1.3 and a trained second head with `C2` ≥ 0.8 to 1.0.** With
   the model-free guess it loses 0.4 to 0.7 tok/s. P(both) about 0.04. Width 3 stays closed.
5. **A cap-following share costs no budget to speak of, but it triples the host check and breaks
   reproducibility.** Expected spread of PPL across shares: under 0.05 % (*estimated*) against a 5 % budget.
   The check grows from 6 full-set runs (about 8 to 12 h) to 12 to 18 (16 to 36 h). I move to Ebony's position:
   **do not build it; use one fixed share tuned for C 35 to 37** (expected loss 0.2 to 0.3 tok/s).
6. **Ebony's GPU prefill (+0.03 to +0.10 tok/s, not worth a build): agreed.** Her 0.65 × a gemv per prefill
   position holds on the kernel's 16-row path too (my count 0.63).

## 1. The stock Q4_0 kernel (next question 5)

### 1.1 What the phone has measured

R72 L5 ran ten types through ggml (*device*). The buffer type decides the kernel:

| type | buffer | ns per weight, cpu 4 hot, 4096 inputs, `ramp_ok=1` | ÷ Q4_0 |
|---|---|---|---|
| Q4_0 | `CPU_KLEIDIAI` | 0.0806 | 1.00 |
| Q4_K | `CPU_REPACK` | 0.0713 | 0.88 |
| IQ4_NL | `CPU_REPACK` | 0.0750 | 0.93 |
| Q2_K | `CPU` (stock dot products) | 0.1098 | 1.36 |
| IQ4_XS | `CPU` | 0.1228 | 1.52 |
| Q3_K | `CPU` | 0.1657 | 2.06 |

- **Q4_0 never ran in the plain `CPU` buffer on the phone.** The three plain-buffer types are other formats; they
  show only that row-at-a-time stock kernels sit at 1.4 to 2.1 × on an A78.
- *Host* (memo-05, Tom, `--no-repack` against repack on the M4): kernel 2.0 ×; whole model 1.9 × on one thread,
  1.4 to 1.5 × on eight threads, where the laptop turns byte-bound. Not bit-identical to the repacked path; PPL
  delta −0.014 % and +0.017 % on 40 chunks.

### 1.2 Count (*source*, *estimated*)

`ggml_vec_dot_q4_0_q8_0`, plain NEON branch (`arch/arm/quants.c`; the A78 has no i8mm, so one row per call). Per
64 weights: 2 AND, 2 USHR, 4 SUB, 4 SDOT, 2 SCVTF, 4 scalar half-to-float converts, 2 scalar FMUL, 2 DUP, 2
multiply-adds that are not fused (4 slots), 2 zeroing moves.

| kernel | slots per 128 weights | floor, cycles | against the KleidiAI gemv |
|---|---|---|---|
| KleidiAI gemv | 25 | 12.5 | 1.00 |
| stock dot product, one position | 60 (V0 8, V1 4, either 48) | 30 | **2.4** |
| stock layout, two positions, unpack shared (a kernel nobody has written) | 100 | 50 | 4.0 |
| W-position kernel on the packed layout, two positions | 36 | 18 | 1.44 |

The repacked layout wins because it puts four rows in one vector: one SDOT serves four rows. The stock layout has
one row per vector and pays the float step per block per row.

**c for the stock kernel: 2.2 (1.9 to 2.5)**, between the laptop's 1.9 to 2.0 and the count's 2.4.

### 1.3 Prices (*estimated*; C 40; two-term reader as in `bonnie_tree_r2.py`)

| use | value |
|---|---|
| four threads alone, GB/s | 15.8 (17.9 to 14.1) against the gemv's 25.64 (*device*) |
| body on the CPU alone | × 1.62 (1.43 to 1.81) |
| K0 on the CPU alone, tok/s (O 8, stock head) | 4.47 / 3.96 / 4.06 (c 1.9: 5.02 / 4.46 / 4.55; c 2.5: 4.03 / 3.56 / 3.67), against 6.90 / 6.21 / 6.22 |
| width-1 lane, K 0 | 11.54 / 10.32 / 10.45 against 12.54 / 11.26 / 11.34 with the gemv (`pair` rates) |
| width-1 lane, K 300 | **8.68 / 7.97 / 8.16** against 9.98 / 9.16 / 9.32 |
| width 2 | 4.0 × per weight on the CPU's rows; about 9 GB/s for four threads. Not a lane |

- **Overlap rows from the served file (r5 §4): withdrawn.** The overlap is about 10 % of the CPU's rows in the
  cool state. At 4.0 against 1.44 per weight those rows make the CPU side 10 to 17 % slower, which costs 0.35 to
  0.6 tok/s after re-balancing. Following the cap gains 0.24 to 0.45. The overlap must be real bytes in a fast
  layout: 0.16 to 0.25 GB (r5 §4).
- **File-backed route 2 (r5 §7): a width-1 fallback only.** It gives Llama-2 and DeepSeek a body that stays
  resident at 8 tok/s, 1.2 to 1.3 below the anonymous gemv they cannot hold.
- **File-backed route 1 is therefore the route:** the CPU's rows in the packed layout, written once to the app's
  files and mapped. Bytes: 2.4 to 2.9 GB of storage per model beside the 4.0 GB served file (the GPU's rows need
  no side file; they are built from the served file at load). First-run cost: one pack and write of that size
  (*estimated* 10 to 30 s). It needs the buffer type to accept a mapped file as tensor data, and the owner's
  answer to (a) and (b).
- A probe cell would replace the count with a number in 3 s: Q4_0 in the plain buffer, hot on cpu 4 and cold on
  4-7 (the L5 cell with repack off). Only worth it if route 2 is wanted as a fallback.

## 2. Depth 2 from Tom's head dumps (next question 6; *host*)

Data: the 32-prompt set (CPU dumps, 128 tokens each) and the saved guesses of the best head per model
(`head3000-tok` for Llama-2, `head2117-tok` for Yi). The script reproduces Tom's top-1: 0.497 and 0.379.

| | Llama-2 | Yi |
|---|---|---|
| pairs | 3,606 | 3,531 |
| top-1 (a1) | 0.497 | 0.379 |
| P(hit at j + 1 given hit at j) | **0.606** (n 1,768) | **0.509** (n 1,324) |
| P(hit at j + 1 given miss at j) | 0.386 (n 1,806) | 0.303 (n 2,175) |
| model-free depth 2: most frequent successor of the depth-1 guess, P(right given depth 1 right) | **0.274** | **0.250** |
| `C2` from the model-free guess (that ÷ a1) | 0.55 | 0.66 |
| `C2` ceiling (the head with the true next state ÷ a1) | 1.22 | 1.34 |
| sum of path probabilities, chain of two: model-free / assumed `C2` 1.0 / ceiling | 0.632 / 0.744 / 0.798 | 0.474 / 0.523 / 0.572 |

- **Hits cluster.** After a hit the next guess is right 1.2 to 1.3 × as often as on average. A second head
  reading row j cannot know more than the first head reading row j + 1 knows, so 1.22 and 1.34 are ceilings for
  `C2`.
- **The floor is a real guess source:** a successor table from the training rows (470,907 rows for Llama-2).
  For Yi the three training files overlap, so its table counts some rows twice; the guess is unaffected in kind.
- A trained second head (Medusa's head 2) lands between. No number for it exists here; training one is the same
  tool with the label shifted by one more row (Tom, minutes on the existing dumps).
- **Not measured:** the head's own second-best guess (the "top-k" shape). The saved files hold only the top-1
  guess. In memo-10's assumption the chain beats top-2 on Llama-2 whenever `C2` ≥ 0.36, so the chain is the shape
  that matters.

### 2.1 What it does to width 3 (*estimated*; `dev_ratio(3)` as cg(3); width 2 at cg 1.25; tok/s gained by width 3)

| model, K µs | subject | cg(3) 1.3: `C2` 0.55 / 0.8 / 1.0 / 1.22 | cg(3) 1.5: same |
|---|---|---|---|
| `linear`, 300 | Llama-2 | −0.36 / +0.03 / +0.33 / +0.67 | −0.78 / −0.41 / −0.12 / +0.21 |
| `linear`, 300 | Yi | −0.31 / −0.31 / −0.24 / −0.04 | −0.71 / −0.71 / −0.63 / −0.43 |
| `linear`, 300 | DeepSeek | −0.37 / −0.23 / 0.00 / +0.24 | −0.76 / −0.61 / −0.40 / −0.16 |
| `pair`, 300 | Llama-2 | −0.62 / −0.23 / +0.08 / +0.42 | −1.02 / −0.64 / −0.34 / −0.01 |
| `pair`, 300 | Yi | −0.54 / −0.54 / −0.46 / −0.26 | −0.91 / −0.91 / −0.83 / −0.63 |
| `pair`, 150 | Llama-2 | −0.73 / −0.30 / +0.03 / +0.41 | −1.27 / −0.86 / −0.53 / −0.18 |

- **Llama-2 gains only with `C2` ≥ 0.8 (`linear`) or ≥ 1.0 (`pair`) and cg(3) ≤ 1.3.** That is a second head
  right 0.40 to 0.50 of the time after a depth-1 hit, against a ceiling of 0.606 and a model-free 0.274.
- Yi loses at every `C2` up to the ceiling. DeepSeek's head does not exist; with the assumed 0.438 it needs the
  ceiling.
- P(`dev_ratio(3)` ≤ 1.3) 0.12 to 0.2 (r5); P(a trained second head reaches `C2` 0.9 on Llama-2) 0.3, my
  judgement. **P(width 3 pays on Llama-2) about 0.04. Closed; not worth Tom's time before R75 reads
  `dev_ratio(3)`.**
- One use of the clustering that costs nothing: a **confidence gate by history** (run width 2 only after a hit).
  In `pair` a wrong guess costs little, so it is worth under 0.1 tok/s (Ebony r4 §2.2 says the same of a
  confidence gate). Not pursued.

## 3. The numerics of a share that follows the cap: what the budget check must cover

**What changes with the share.** A row on the GPU is computed with float activations and half-precision
dequantised weights; a row on the CPU with int8 activations per block and exact integer products. Both are
inside the declared numerics of a lane. The share decides which rows get which.

**Expected size** (*estimated*). Two anchors: swapping the CPU's activation layout on every row moved PPL by
−0.014 % and +0.017 % (*host*, memo-05); the GPU kernel's rows differ from float64 by 6e-4 to 8e-4 (*device*,
R74 identity). Tom's emulation result for the lane (patch 0006) is not reported yet. I expect PPL to move by
under 0.05 % between any two shares, against budgets of +5 % (0.42 / 0.39 / 0.44 PPL). **The budget is not at
risk; the procedure is.**

**What a check has to cover if the share can move:**

| item | why | cost |
|---|---|---|
| full set at the smallest and the largest GPU share of the pre-registered range, at `-b 512` and `-ub 1` | the rule for numerics that differ by mode: inside the budget in every mode the phone can be in | 2 shares × 2 batch modes × 3 subjects = **12 full-set runs** (a fixed share needs 6) |
| the row map of the runtime, per stage | the share is set per stage (q‖k‖v, o, gate‖up, down); the emulation must move the same rows the lane moves | code in the emulation knob, no runs |
| one arm with the share alternating between its ends, token by token, at `-ub 1` | the KV cache then holds states made under one share and read under another; the ends do not show that | 3 more runs; 40 chunks each is enough if the ends agree to 0.05 % |
| one interior share, 40 chunks | PPL need not be monotone in the share | 3 short runs |
| the claim run logs the share per run, and a run outside the range is void | otherwise no host run corresponds to a phone run | kit and pre-registration text |

- **Host time** (*derived* from the freeze logs: 4.0 h for Yi's three full-set runs, 6.2 h for Llama-2's): 1.3
  to 2 h per full-set run. Fixed share: 6 runs, 8 to 12 h. Following: 12 full runs plus 6 short ones, **16 to 26 h**;
  18 full runs if the alternating arm is run in full, 24 to 36 h.
- **What no check gives back:** the same prompt can give different text on a warm and a cool phone. The speed
  claim does not need equal text, but campaign rule 4 ("bit-identical unless declared") would need the
  declaration to say "numerics depend on the thermal state inside a tested range".
- A share fixed once per run, at its start, drops the alternating arm (12 full runs + 3 short). It keeps the two
  ends and the log rule.

**Price of each choice** (*estimated*; r5 §4 and the clock odds of ebony-joins-r4 §3):

| choice | tok/s against the best following share | bytes | host check | reproducible |
|---|---|---|---|---|
| fixed share tuned for C 35 to 37 | −0.2 to −0.3 in expectation; −0.29 to −0.45 in a cool run | 0 | 6 runs, 8 to 12 h | yes |
| share chosen once per run | −0.0 to −0.1 (a cap that moves inside the run) | 0.16 to 0.25 GB in a fast layout | 12 full + 3 short runs, 16 to 25 h | per run, with the log |
| share following inside a run | 0 | the same | 12 full + 6 short runs, 16 to 26 h (18 full: 24 to 36 h) | no |

**My position moves to Ebony's (her r5 §5): do not build the following share.** Three reasons, in order:
1. The bytes fall on Llama-2 and DeepSeek, which already do not fit (4.02 + 0.25 GB against 4.2 file-backed and
   3.4 anonymous, *device*).
2. The zero-byte source I offered in r5 does not work at width 2 (§1.3).
3. 0.2 to 0.3 tok/s does not decide whether 12 is reached in any stack either of us holds.
A fixed share tuned for the capped state is also the safer side: it gets no slower when the phone is cool.

## 4. Review of ebony-joins-r5.md

- **§1 GPU prefill, +0.03 to +0.10 tok/s, not a try-list item: agreed.**
  - Her CPU row rests on my "0.65 × a gemv per position". I checked it on the path a prefill takes: with 16 or
    more rows KleidiAI's kernel runs its 16-row loop, four tiles per weight block, 63 slots per four rows: 0.63
    × a gemv per position (*source*, *estimated*). Her 18 to 26 tok/s stands.
  - Her point that the GPU saves heat only by finishing sooner is the right frame, and her comparator (a pause
    between prefill and decode) is cheaper than any kernel. It is an owner question, as she says.
  - One addition: a GPU prefill changes which rows are hot in which cache before the decode, not only the
    clocks. Not priced; small.
- **§1.4 half-precision accumulation in upstream's batched kernel (0.9 to 1.4 % of rms, *host*, random data):**
  not checked by me. If it holds, that kernel is outside any budget argument made for the gemv, as she says.
- **§5 review of my r5: accepted.** Her mixed-arm band (1.20; 1.00 to 1.30) against mine (1.15; 1.02 to 1.30):
  print both. Her review may predate my §2.5 (the `x1gpu` cells, the `linear` variant, and `pair` read at ≥ 0.93
  instead of ≥ 0.95); she accepts "two changes" and §2.5 holds a third. That one still needs her answer.
- **Her §3 (eight conditions for the W-position kernel to be exact against KleidiAI):** skimmed, not reviewed.
  It is the work both stacks wait on.

## 5. Next questions, ranked

1. **`pair` line at 0.93 or 0.95** (r5 §2.5): needs Ebony's answer before the pre-registration is frozen.
2. **A second head on the existing dumps** (Tom, minutes): gives `C2` a number. Only worth it if R75 reads
   `dev_ratio(3)` ≤ 1.3.
3. **The packed side file as runtime:** what the KleidiAI buffer type needs in order to take a mapped file
   (allocation hook, header, sha key), and the first-run pack time on the phone. Design, host.
4. **Fixed share: which C to tune for.** Needs the cap record of a real decode (R73) or R75's `max_mhz` under
   lane-like load; until then C 35 to 37.
5. **The one-thread hot rate of the stock kernel on the phone** (one probe cell), only if route 2 is kept as a
   fallback.
6. **Half-precision accumulation in the batched GPU kernel** (Ebony r5 §1.4): re-derive from the kernel source.

## 6. Limits

- §1.2 is a count from C source with intrinsics, not from compiled assembly; the compiler may fuse or hoist some
  of it. The band 1.9 to 2.5 is meant to hold that.
- §1.3 uses the two-term reader at C 40 and the `pair` rates at width 1.
- §2 is one head per model on 32 prompts; DeepSeek has no head. The ceiling argument assumes a second head
  cannot beat the first head given the true state.
- §2.1 inherits `bonnie_tree.py`'s tree model and the r5 `linear` variant.
- §3's "under 0.05 %" is a judgement from two anchors; Tom's emulation is the measurement.
- Run times in §3 are subject totals from log time stamps, which include download and conversion.

## 7. Sources (2026-11-23)

Repository: R72 JSON (`l5` lines; sha `cbc332dd…`); `lab/llama.cpp-c3r32/ggml/src/ggml-cpu/arch/arm/quants.c`
(`ggml_vec_dot_q4_0_q8_0`, NEON branch); KleidiAI `…16x4_neon_dotprod_asm.S` (16-row loop) as in r5;
`memo-05-y1-column-kernels.md` (T3) and `memo-05-work/tom-host.md` lines 196 to 210; `memo-09-work/tom-head.md`
§4; `lab/c3/head/{llama2,yi}/eval.rows`, `train*.rows`, `head*-tok.pt.evalset.pred.npy`; `tools/c3_head_train.py`
(file formats); `lab/c3/logs/freeze-*.log` (start times); `results.csv` (budgets);
[ebony-joins-r5.md](ebony-joins-r5.md). No web source this round: nothing here states the current state of a
model or of llama.cpp beyond the local tree.
