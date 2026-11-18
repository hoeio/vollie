# c3 memo-01: is 15 tok/s on a dense 7B-class model within reach? (joint round, Bonnie and Ebony)

Date: 2026-11-18. Authors: Bonnie (`claude-fable-5-2`) and Ebony (`claude-fable-5-2`), each with a review of the
other's draft. Merged and decided by the lead (`claude-opus-5-5`). The drafts are in
[memo-01-work/](memo-01-work/): `bonnie-draft.md` has its review in §8 and `ebony-draft.md` in §6. Nothing was
measured for this memo. Every number is published, quoted from campaigns 1 and 2 with its label, or *estimated*.

## 1. Verdict

**15 tok/s on a body that reads 7.1 B or more weights per token is out of reach with every priced method.** Both
researchers put the chance below 0.05 (*estimated*). The project rule is to say this with numbers, and the numbers
follow. The campaign continues as research: it measures how far the ceiling really is (§4), and it prices the
one opening (§3).

| | Bonnie | Ebony |
|---|---|---|
| central | 4 to 8 tok/s | – |
| best priced combination | 5.1 to 9.3 | 8 to 9 (40 % column sparsity) |
| physical corner | 12 to 13.5 | – |

Three limits bind at the same time (*device* inputs from campaigns 1 and 2, *estimated* outputs):

- **Resident bytes.** The largest model set held resident so far is 2.67 GB (R58). The app was killed at
  3.93 GB of repack copy (R9). A Q4_0 7B body is 4.0 to 4.2 GB, and the repack adds an anonymous second copy.
- **Bandwidth.** Q4_0 reads give 4.5 to 5.7 tok/s at 19 to 24 GB/s, and 12.1 tok/s at the nominal 51 GB/s.
- **Compute.** The repacked 4-bit `sdot` kernel caps an 8B model at 7.6 to 9.3 tok/s on three cores (R41 benches).

## 2. What does not work (*estimated* from published sources and campaign data)

- **Fewer bits do not buy speed.**
  - Q3_K runs at half the `sdot` rate (*device*).
  - IQ2, IQ3 and Q3_K have no arm64 repack path at `620d2225c`.
  - On Llama-3-8B, every quantisation below about 3.05 bpw fails F16 × 1.20.
- **Output-side and dense-to-MoE methods are far outside +5 %.** CATS costs +15 % or more; Deja Vu, CMoE
  (+33 to +65 % at 25 % FFN sparsity) and ExpertWeaver are worse.
- **What +5 % can buy is input-side (column) sparsity of 25 to 40 % on stock SiLU models** (TEAL, DIP, SpQt).
  - In the file's layout, skipped columns save no DRAM traffic.
  - It needs an exact column-major Q4_0 copy and a new kernel at 1.2 to 2.6 × `sdot` per kept weight (unbenched;
    Bonnie 2.6 ×, Ebony 1.7 to 2 ×).
  - The net is 1.05 to 1.5 × on the token.
- **Draft speculation stays closed** (R20).
- **The two-stage head saves nothing with a Q4_0 head.**

## 3. The opening: Gemma 4 E4B

Gemma 4 E4B has 8 B parameters in total but reads about 4.5 B per token, because its per-layer embeddings are
looked up, not multiplied.
- Bonnie prices it at 7.5 to 11 tok/s, and at 15 only if the read rate with power saving off is at least 33 GB/s.
- Ebony prices it at 12 to 14 tok/s (Bonnie notes 13 to 14 by Ebony's own inputs).
- Whether it counts as "dense 7B-class" is the owner's decision. It is held until the owner answers, and until
  then it is worked on as a **provisional** subject, labelled as such.

## 4. Lead's decisions (provisional where marked; open questions are kept for the owner's return)

- **Candidates:**
  - Mistral-7B-Instruct-v0.3 and Qwen2.5-7B-Instruct (Ebony §1); reserves Llama-3.1-8B and Yi-1.5-6B.
  - Gemma 4 E4B provisional, held for the owner.
  - RWKV-7 G1 7.2B (ReLU² FFN, not a transformer) is held for the owner. H4 runs first, so that the question
    carries a number.
- **Calibration-derived side tables** (thresholds, hot-first order), computed from the served file and a fixed text,
  are runtime data. *Provisional*; the owner confirms.
- **Stock numerics:**
  - The budget's stock PPL is measured on the host as in campaign 2.
  - A kernel that matches ggml's reference path but not KleidiAI's is a declared numerics change, checked at
    `-b 512` and `-ub 1`.
- **Goal:** stays at 15 tok/s as written. Raising it to 20 is not on the table: confidence is the opposite of
  sufficient.
- **Disk:** the laptop has about 29 GB free. Tom downloads quantised GGUFs (Q8_0 as the F16 proxy, labelled) rather
  than F16 safetensors, and keeps at least 10 GB free.

## 5. TEST list for Tom (host now; device items are written up for the next phone window)

| # | test | decides |
|---|---|---|
| H1 | Byte inventory from GGUF headers per candidate, Gemma 4 E4B included: bytes read per token by tensor group, head type, what the phone build repacks | the byte inputs of every price |
| H3 | Input-sparsity tolerance, no new kernel: generalise `tools/dense_colskip.py` (or add an env knob) to all seven block matrices. Thresholds from wikitext-2 **train**. Grid 20 / 30 / 40 / 50 % model-wide and per group; first N tokens dense, N = 0, 1, 8; 40 to 60 chunks at `-b 512` and `-ub 1` | whether the column lever is worth a kernel. Below 25 % inside +5 % with a 0.5-point margin on both models closes it |
| H6 | Column-kernel microbench: exact int4 column kernel against Q4_0x4 on 14336 × 4096 at density 100 / 60 / 50 %, host first. Identity at threshold 0 against the non-KleidiAI kernel | above 2.2 × `sdot` per touched weight on the A78 closes Y3/Y4 |
| H7 | Sublayer ablation: skip one attention or FFN sublayer at a time, 20 chunks, both models | the cheapest sublayer costs more than +1.0 % on both → closed |
| H5 | Stability of the active set (rides on H3) | whether sparsity can shrink the resident set |
| H2 | Served-quantisation ladder (Q8_0 proxy, Q4_0, Q4_K_M, Q3_K_M, IQ3_XXS, with and without imatrix) | the lowest file inside F16 × 1.20, and the budget's stock PPL |
| H4 | ReLU² exact-zero count on RWKV-7 G1 (minutes) | whether the owner question is worth asking |
| P1, P2 | Device: resident-capacity probe; dense rate and `sdot` ceiling with power saving off. Written up as a kit addition and a pre-registration draft; not run until the owner gives a phone window | open items 2 and the residency limit |

## 6. TRY list (after the tests that gate them)

1. A kernel layout written to a file and mapped `PROT_EXEC`, with no anonymous repack copy (Y1; gated by H1 and
   P1).
2. An exact column-major Q4_0 kernel (Y3), then input thresholds (Y4); gated by H6 and H3.
3. A cache-aware mask and hot-first file order (Y5); gated by H5.
4. A lower-bit served file (Y6); gated by H2, P1 and P2.

## 7. Next joint round (Bonnie + Ebony, round 2)

Questions left open by this round, priced so that no exotic lever is dropped unexamined:

- Dense Q4_0 on the Adreno 660 through Qualcomm's OpenCL backend. Campaign 1 measured 0.7 to 2.9 tok/s on MoE
  models (*device*); the dense case was never priced. Also: a GPU read rate beside the CPU (a CPU + GPU split of
  bytes).
- The Hexagon 780 (v68) NPU through QNN or ggml-hexagon.
- Gemma 4 E4B in detail: llama.cpp support, PLE bytes, residency.
- Anything else that reads fewer bytes per token without changing the file.
