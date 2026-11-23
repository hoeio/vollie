# c3 memo-12 work, round 12 (Bonnie): the head knob for campaign 3; does the head screen (C7) inherit the one-scale activations

- **Date:** 2026-11-23. **Author:** bonnie, `claude-fable-5-2`. Follows [bonnie-joins-r11.md](bonnie-joins-r11.md). Not
  committed. No verdict here. **Ebony has not reviewed this file.** I have not seen `tom-kai-stock.md`.
- **Labels.** *device*, *host*, *estimated*, *source* as before. Triples are Yi / Llama-2 / DeepSeek.
- **One new *host* number:** a blocks-of-32 arm added to [bonnie_head_paths.py](bonnie_head_paths.py) and the script
  re-run on Tom's existing dumps (no dump made, no llama.cpp run). Everything else is existing *device* data
  (R72 block `l7`; campaign 2 R60 and R62), source, or arithmetic. No web source.

## 0. Answer first

1. **On what exists, sending the head to the repack type (`VOLLIE_HEAD2_REPACK=1`) wins on all three counts.**

   | per subject (Yi / Llama-2 / DeepSeek) | head in KleidiAI (today's default) | head in the repack type |
   |---|---|---|
   | share of the +5 % budget used by the unmodified runtime | 6 to 20 % / under 1 % / 0 to 20 % (*estimated*, r11's prediction) | 0: it is the path the freeze measured |
   | head time per token | 9.04 / 4.40 / 13.88 ms (*device*, R72) | 7.7 / 3.7 / 11.8 ms (*estimated*; band 7.1–8.6 / 3.5–4.2 / 11.0–13.2) |
   | head bytes in memory | 263 / 131 / 421 MB | 215 / 108 / 344 MB |

   The time is the weak cell: the repack Q6_K kernel was never timed on this phone at 4,096 inputs. Campaign 2
   timed both paths on OLMoE's 2,048-wide head: 3.8 to 4.2 ms in KleidiAI, 3.3 to 3.5 ms in the repack type
   (*device*, R60 and R62).
2. **The gain is small in speed and real in budget:** 1.3 / 0.7 / 2.1 ms per token (*estimated*), which is +0.18 /
   +0.10 / +0.29 tok/s near 11.5; and up to a fifth of Yi's budget returned.
3. **C7 does not inherit the one-scale activations. It cannot run with the head in KleidiAI at all** (*source*):
   the two-stage head declines any head that is not plain or in the repack type's 8-row layout. So choosing C7
   already forces the knob. Its screening stage multiplies by activations in blocks of 32, its exact stage in
   blocks of 256.
4. **A side finding on C7's price.** Every stack holds C7 at "head × 0.5". The two-stage head as campaign 2
   built it reads a Q4_0 copy of every row plus the exact rows: by bytes that is 0.80 / 0.91 / 0.76 of the Q6_K
   head at K 1024 (*estimated*), and on OLMoE it saved nothing measurable on the phone (*device*, R62). If C7 is
   that design, the stacks are 1.2 to 2.0 ms per token optimistic. Question 1 of §4.
5. **Tom tasks** are in §3: one host run that would confirm the budget cell, two dumps, and one probe cell for
   the time.

## 1. The head knob

### 1.1 What each setting is (*source*)

| | head in KleidiAI | head in the repack type (`VOLLIE_HEAD2_REPACK=1` at load) |
|---|---|---|
| where it is decided | default: KleidiAI's `supports_op` takes a Q6_K matmul weight | the knob makes it decline `output.weight`; the repack type takes it (`kleidiai.cpp` lines 1793 to 1800) |
| weights | dequantised and re-quantised to int8, one scale per row; 4,108 bytes per row | the served Q6_K blocks interleaved 8 rows at a time (`q6_K_8x4`); 3,360 bytes per row; a re-ordering, no new rounding |
| activations | one scale and zero point per vector (r11 §1.3) | Q8_K: one scale per 256 values |
| kernel on the phone | KleidiAI int8 dot-product gemv | ggml's `ggml_gemv_q6_K_8x4_q8_K`, which has a NEON dot-product body in `arch/arm/repack.cpp` (line 1327 on) |
| the freeze's path | no | **yes**: the lab build (KleidiAI off) puts the head in the repack type |
| in which tree | every tree | the kit's tree (`lab/llama.cpp-c3r32`) and campaign 2's; not `lab/llama.cpp-c3` |
| load work | scalar float re-quantisation of 131 to 421 million weights (*estimated* 1 to 4 s, r9) | an interleave copy |

The knob matches the tensor name `output.weight`. All three subjects have an untied head of that name. (Granite's
tied head has another name; that is why campaign 2's knob never applied to it.)

### 1.2 Budget share (*estimated*)

- Head in KleidiAI: r11's prediction, Yi +0.3 to +1.0 %, Llama-2 within ±0.05 %, DeepSeek +0.0 to +1.0 %
  (no dump). As a share of +5 %: 6 to 20 % / under 1 % / 0 to 20 %.
- Head in the repack type: the head's arithmetic is the freeze's. What remains between the phone's path and the
  freeze is the Q4_0 body, which r10 reads as rounding ties: noise around zero.
- The proxy's noise, for scale (*host*, re-run today with one more arm): on Yi the float head gives 0.24615,
  ggml's blocks of 256 give 0.24484, blocks of 32 give 0.24593, KleidiAI's one scale gives 0.25020. The three
  block schemes sit within 0.0013 of each other; KleidiAI is 0.004 to 0.005 outside them. On Llama-2 all four
  are within 0.0002.
- Tom's run in progress scores the "head in KleidiAI" column directly. The "repack" column needs task H1 (§3).

### 1.3 Head time on the phone

**Measured, head in KleidiAI** (*device*, R72 `l7`, hot, `ramp_ok=1`):

| | rows | ms, 3 threads (7,4,5) | ms, 4 threads (4-7) | bytes read as int8 | effective rate, 3 threads |
|---|---|---|---|---|---|
| Yi | 64,000 | 9.039 | 8.944 | 262.9 MB | 29.1 GB/s |
| Llama-2 | 32,000 | 4.396 | 4.524 | 131.5 MB | 29.9 GB/s |
| DeepSeek | 102,400 | 13.881 (cap 2,150 MHz) | 14.271 | 420.7 MB | 30.3 GB/s |

- The probe prints `gbs` on the served Q6_K bytes (23.8 to 24.8). Counted on the bytes KleidiAI really reads,
  the rate is 29 to 30 GB/s, the phone's read rate (29.9, R72). **The int8 head is bus-bound**, and the fourth
  thread adds nothing. So its time is its bytes.

**Not measured: the repack Q6_K kernel on this phone at these sizes.** Two estimates:

| basis | Yi / Llama-2 / DeepSeek |
|---|---|
| bytes, if it is bus-bound too (215 / 108 / 344 MB at the same rate) | 7.4 / 3.6 / 11.4 ms |
| campaign 2's device ratio on OLMoE's head (50,304 × 2,048): 3.3–3.5 ms repack against 3.8–4.2 ms KleidiAI, ratio 0.85 (0.79 to 0.92) | 7.7 / 3.7 / 11.8 ms |
| **used** | **7.7 / 3.7 / 11.8 ms; band 7.1–8.6 / 3.5–4.2 / 11.0–13.2** |

- OLMoE's byte ratio is 0.82 and its time ratio 0.85, so the repack kernel was near bus-bound there as well.
- Risk: Q6_K needs a 6-bit unpack per weight, the int8 head needs none. At 4,096 inputs and with a capped clock
  the unpack could bind before the bus. Then the knob costs time instead of saving it. P(the repack head is
  slower than the KleidiAI head on at least one subject) 0.2. Task H3 measures it.
- Saving at the centre: 1.3 / 0.7 / 2.1 ms per token; with C7 at × 0.5 in the stack, half of that.

### 1.4 Bytes

| | KleidiAI buffer | repack buffer | anonymous total | side file (r8) |
|---|---|---|---|---|
| head in KleidiAI | 3,221 / 3,599 / 3,658 MiB | none | the same | holds the head: 3.38 / 3.77 / 3.84 GB CPU-only |
| head in the repack type | 2,970 / 3,474 / 3,257 MiB | 205 / 103 / 328 MiB | 46 / 23 / 73 MiB less | Q4_0 body only: 3.11 / 3.64 / 3.42 GB; the head needs a second mapped file or stays anonymous |

- The repack head is also a pure re-ordering of served bytes, so the side-file design of r7 extends to it with
  the same key. Not needed for a first build: 0.2 to 0.4 GB anonymous beside a file-backed body fits what the
  residency arm tests (r8 §2 steps carry 0.25 to 0.30 GB anonymous).
- A head on the GPU lane (r4 §2) uses the served Q6_K layout in GPU buffers and is not touched by the knob.

### 1.5 What the knob costs to adopt

- **A kit change:** the app sets `VOLLIE_HEAD2_REPACK=1` before the model loads (campaign 2's kits r23 on did).
- **A statement in the pre-registration:** on the phone's build this changes the head's numerics from KleidiAI's
  to ggml's. Campaign 2 declared it as "a change of the off arm" (R61 row 2 against row 1). Here it moves the
  phone towards the freeze, not away; the lead decides how it is declared.
- **An identity check:** KleidiAI build with the knob against the lab build, per chunk. Expected: equal up to
  the body's rounding ties, so close but not bit-equal (r10 §2).

## 2. Does C7 inherit the one-scale activations? (next question 3)

*Source:* `patches/llama.cpp-c1-olmoe/0004-c2-head2-untied.patch` (the two-stage head for an untied Q6_K head,
knob `VOLLIE_HEAD2=<K>`) and `0006-c2-head2-kleidiai.patch`.

- **Stage 1, screen:** every row from a Q4_0 copy of the head (the served Q6_K dequantised and re-quantised
  without an imatrix), interleaved 4 rows at a time like any Q4_0 weight, against a **Q8_0 activation (blocks of
  32)**. The copy is anonymous memory made at the first head op.
- **Stage 2, exact:** the K rows with the largest screen logits are recomputed with the stock Q6_K kernel on the
  stock **Q8_K activation (blocks of 256)**; whole 8-row groups come out exact. The other rows keep their screen
  logit.
- **Layout it accepts:** plain rows, or the repack type's `q6_K_8x4` / `q6_K_8x8`. "Anything else (another extra
  buffer type): the caller keeps the stock path." A head in KleidiAI is "anything else". Campaign 2 hit exactly
  this on the phone (R60: the head did not run) and wrote the knob for it.

**So:** C7 never sees the one-scale rule, because C7 and a KleidiAI head exclude each other. With C7 on, the
head's exact rows are on the freeze's path and the screen is on blocks of 32.

On Yi's spiky hidden state (*host*, proxy, today's added arm): activation error ÷ rms is 0.94 % in blocks of 32,
2.39 % in blocks of 256 and 4.67 % under one scale. The screen's activation side is therefore the mildest of the
three; its weight side (a Q4_0 copy) is the coarse part and is what the top-K recompute exists to repair.

**What C7 costs in bytes for our heads** (*estimated*, K 1024, exact rows in groups of 8, as campaign 2 measured
about 7,000 exact rows per token on OLMoE):

| | Q4_0 screening copy (anonymous, extra) | bytes read per token: copy + exact rows | ÷ the Q6_K head's bytes |
|---|---|---|---|
| Yi | 147 MB | 147 + 24 = 171 MB | 0.80 |
| Llama-2 | 74 MB | 74 + 24 = 98 MB | 0.91 |
| DeepSeek | 236 MB | 236 + 24 = 260 MB | 0.76 |

- The head is bus-bound (§1.3), so time follows bytes: C7 as built would be × 0.76 to 0.91, not × 0.5.
- Campaign 2's device record agrees in kind: OLMoE's two-stage head 3.1 to 3.4 ms against 3.3 to 3.5 ms off,
  "saves nothing measurable" (R62). Campaign 1's Granite head did save (5.5 → 3.7 ms), with a tied head and its
  own design.
- memo-09 lists C7 as "head screen (head × 0.5)", P 0.35. I did not find which design that row means. If it is
  this one, the centre should be about × 0.8, and each stack loses 1.2 to 2.0 ms per token (about 0.15 to 0.3
  tok/s near 11.5). If it means a screen on a smaller proxy (a 1-bit or low-rank head), it is unbuilt.

## 3. Tasks for Tom (through the lead; none run by me)

| # | task | cost | pass or reading |
|---|---|---|---|
| H1 | Build the kit's tree (`lab/llama.cpp-c3r32`) for the host with KleidiAI on, options as r9 §3. Run the three served files, 40 chunks, `-b 512` and `-ub 1`, with `GGML_KLEIDIAI_SME=0 VOLLIE_HEAD2_REPACK=1`. Check the load log: head in `CPU_REPACK` (205 / 103 / 328 MiB), `CPU_KLEIDIAI` at 2,970 / 3,474 / 3,257 MiB | one build, about 30 minutes of runs | each value within ±0.05 % of the freeze's `[40]` (r9 §3 table). That confirms the head is the whole shift and gives the knob's stock reference |
| H2 | Dump the final hidden states on the 32 prompts for **DeepSeek** and for **Granite** (campaign 2's file) with `tools/c3_head_dump.cpp`; add the two files to `bonnie_head_paths.py` (Granite's tensor is `token_embd.weight`) and run it | minutes each, no build | range ÷ rms and the proxy shift. DeepSeek: fills r11's missing prediction. Granite: a large value closes R61's open point |
| H3 | A probe cell for the next kit: R72's `l7` with the head in the repack type. Lines as `l7`, plus `buft=CPU_REPACK`, for 64,000 / 32,000 / 102,400 rows at 3 and 4 threads, ABBA against the KleidiAI head | about 20 s of phone time | band 7.7 / 3.7 / 11.8 ms (7.1–8.6 / 3.5–4.2 / 11.0–13.2). **Kill line for the speed claim:** repack ≥ KleidiAI on a subject. The budget argument for the knob does not depend on it |
| H4 | If C7 stays in the stacks: the two-stage head on the three served files, `VOLLIE_HEAD2=1024` with the knob, 40 chunks at both settings (with `VOLLIE_HEAD2_BATCH=1` for `-b 512`), and its exact-row count per token | minutes each on the build of H1 | perplexity against that build's stock; exact rows per token decide the byte factor of §2 |

## 4. Next questions, ranked

1. **Which design is the stacks' C7, and what is its honest factor?** Read memo-08 and memo-09 for its origin,
   price the built two-stage head on our three heads (bytes, exact rows, K), and re-run the stack tables with
   the result. Mine; source and arithmetic.
2. **C5 (gate-row rule, body × 0.93):** the same audit. It is worth four times C7 in milliseconds and has no
   perplexity figure on the three subjects that I found (r10).
3. **DeepSeek's and Granite's hidden states** (task H2), then the per-subject band for DeepSeek.
4. **Why is Yi's final hidden state so spiky, and does the same spike hurt elsewhere?** The lane's GPU kernel
   takes float activations, so it is immune; the CPU's Q4_0 path uses blocks of 32. Ebony's `H-max` asks the
   neighbouring question for half-precision range. A short joint note.
5. **Cold read rate of the served file; fixed share: which C**, after R75.

## 5. Limits

- §0's budget row for Yi rests on r11's proxy, which Tom's run has not yet confirmed or refuted.
- §1.3's repack times are scaled from one other model on the same phone (different width, a different session);
  nothing times the Q6_K repack kernel at 4,096 inputs here.
- §1.4's buffer sizes are tensor arithmetic (r8), not a load log.
- §2's byte factors assume K 1024 and campaign 2's exact-row count; our heads may select differently.
- I read the two-stage head's description in the patch, not every line of its code.

## 6. Sources (2026-11-23)

Repository: R72 JSON (`l7` lines) and `R72-result.md` lines 102 to 104; campaign 2 `reports/README.md` (R60 lines
527; R61; R62 lines 672 to 680), `STATUS.md` lines 141 to 150; `patches/llama.cpp-c1-olmoe/0004-c2-head2-untied.patch`
(header comment, lines 8 to 27, 397 to 440) and `0006-c2-head2-kleidiai.patch`; `docs/runtime-knobs.md` line 56;
`campaigns/03-dense15-5pct/reports/memo-09-gap.md` lines 82 to 83; `lab/c3/head/{llama2,yi}/eval.*` through
`bonnie_head_paths.py`.
Source trees: `lab/llama.cpp-c3r32/ggml/src/ggml-cpu/kleidiai/kleidiai.cpp` (lines 1788 to 1800),
`arch/arm/repack.cpp` (lines 1327 to 1513), `repack.cpp` (lines 1115, 4393, 5062).
