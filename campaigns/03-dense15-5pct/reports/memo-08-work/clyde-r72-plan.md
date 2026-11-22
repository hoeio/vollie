# memo-08 work file, Clyde: plan for R72 (device, no model) and the S3 freeze list

2026-11-22, clyde (`claude-opus-5-5`). **Draft, not a pre-registration.** The bands and the exact pass and kill
values come from Bonnie and Ebony's memo-08 bands. The pre-registration `prereg/R72-*.md` is written from this
file once those bands exist, and before any R72 data exists. Nothing here is measured. Numbers are *estimated*, or
quoted with their label.

- Subjects (memo-06 §6, correction of 2026-11-22, `e582606`): Yi-1.5-6B-Chat, Llama-2-7b-chat, DeepSeek-LLM-7B-chat.
- Reserves: Mistral-7B-Instruct-v0.3, Qwen2.5-7B-Instruct.

## 1. Why R72, and what it must deliver

The gap ledger prices each subject as t = max(B ÷ r, W_eff ÷ c) + O.
- Its inputs today are R69's pre-registered centres plus *host* kernel ratios: r 24, r4 27, c 40.4 GB/s, the 4th
  core at × 1.15, and K1 at k 1.55 to 2.2.
- R69 measures r, c_b and C, and k at the old Mistral shapes on one A78.
- R69 does not measure four things the ledger needs, and R72 measures them:

| ledger row | quantity | why R69 does not give it |
|---|---|---|
| L1 | **k of K1** per touched weight, on one **A78** and on the **X1**, hot and cold, at the **subjects' shapes** (inputs 4096 and 11008) and **frontiers** (keep 0.58 Yi; 0.60 Llama-2 and DeepSeek; 0.65 Llama-2 fallback) | R69's P2col and 0002's D2 rows run 4096 and 14336 at keep 0.60 / 0.65, on the A78 only |
| L2 | **four-core read rate r4**: cold Q4_0 gemv through ggml's own threadpool on X1 + 3 A78, beside the three-core r3 | R69 has the X1 + 3 A78 cell, but with the probe's own loop, not ggml's threads and barriers |
| L3 | **compute ceiling with 3 and 4 cores, measured together**: hot gemv with every thread running at once, each in its own L2 slice, clocks logged | R69's c_b is a sum of single-thread hot cells (× 0.87); the 4-core value is assumed (× 1.15 to 1.245) |
| L4 | **body time per subject**: one layer's real mat-vec sequence (q, k, v, o, gate, up, down at the subject's shape) over a cold arena, at 3 and 4 big cores, stock against the subject's rule (k and v stock, other matrices K1 at the frontier) | nothing in R69 runs a subject-shaped layer or the rule |
| L5 | **per-type gemv cost** against Q4_0x4: the types a served file may use | R69 has Q4_0 only |
| L6 | **sync cost**: an empty barrier sequence of one token's mat-vec count at 3 and 4 threads | O is assumed at 6 to 12 ms |
| L7 | **head time**: Q6_K head gemv (llama-quantize's default head type) at each subject's vocabulary (64,000; 32,000; 102,400 rows × 4096) | not measured |

L4 is the row that matters most. It turns k, r and c into a per-subject body time with the lever on, measured
under one heat state with ABBA. The other rows explain L4 and feed the per-type and core-count choices.

## 2. Blocks (kit r32; runtime = 0001 + 0002 + a new 0003 probe patch, `VOLLIE_PROBE`, no model loaded)

Order: anchors first (cool phone), then D2s, then T, then G, then B, then H. The r16 cool-down gate runs before
each block.

| block | content | cells (*estimated* time) |
|---|---|---|
| A, anchors | R69's `p2` q4x4 and plain read on 3 A78, X1 + 2 A78, X1 + 3 A78. They are the same code, so R72 can be compared with R69 under its own heat state | 6 cells × 3 s, ~0.5 min |
| D2s, K1 at subject shapes (L1) | 0002's `p2d` kernels with inputs 4096 and 11008 (14336 kept as the reserve Mistral's shape). Keep 1.0 / 0.65 / 0.60 / 0.58 / 0.50 (0.50 is a curve point). Hot and cold; one pinned A78 (cpu 4) and the X1 (cpu 7); stock cells before and after each group (drift). Y1b rows as in 0002. Identity lines (`memcmp` against the stock 4x4 with the dropped x zeroed) at both new shapes | ~2.5 min |
| T, token-shaped sweep (L4, L2) | per subject, one layer's 7 mat-vecs at its shapes (Yi: k, v 512 rows; Llama-2 and DeepSeek: 4096 rows) through a 1 GB cold arena (about 10 layers, past the system cache). Two arms: stock (4x4 on every matrix) and rule (k, v stock; q, o, gate, up, down K1 at the frontier keep). Core sets 3 A78, X1 + 2 A78, X1 + 3 A78. Same thread split and spin barrier for both arms, ABBA in each cell. Prints ms per layer, × the layer count (32 / 32 / 30) = body ms per token, and GB/s. One stock cell runs through a ggml graph (CPU backend, repack buffer, `n_threads`, the same affinity) to tie the probe's threads to ggml's (L2) | 3 subjects × 3 sets × ABBA, plus Llama-2 at keep 0.65: ~3 min |
| G, per-type gemv (L5, L3) | ggml's own `vec_dot` / repack gemv for Q4_0 (4x4), IQ4_NL (4x4), Q4_K, IQ4_XS, Q3_K, IQ3_S, IQ3_XXS, Q2_K, Q6_K, Q8_0. Shapes 4096 → 4096 and 11008 → 4096. Hot and cold; 1 thread on the A78 and on the X1; then **all threads at once**, hot, on X1 + 2 A78 and X1 + 3 A78 (L3), with clocks logged during the cell. ns per weight and the ratio to Q4_0x4 in the same regime | ~4 min |
| B, barriers (L6) | empty spin barriers, as many per "token" as the subject's decode graph has nodes (Tom counts them on the host from a `-ub 1` graph of each served file; about 225 mat-vecs plus norms, rope, attention and adds), 3 and 4 threads, 200 tokens | ~0.2 min |
| H, head (L7) | Q6_K gemv, rows 64,000 / 32,000 / 102,400 × 4096, cold, 3 and 4 threads; Q8_0 beside it for the two-stage head | ~0.5 min |
| hdr | D3 header as 0002 | – |

About 12 to 15 phone minutes plus the cool gates (*estimated*). Riders only if their host gates pass before the
build:
- C6 units-of-eight kernel (memo-07 test 5: u ≥ 0.12 on two models and host k ≤ 1.15). If it rides, it adds a
  third arm to T.
- The A55 weighted chunks (C2). Not in R72: no host design exists.

## 3. Derived quantities and draft decision rules (values to be fixed in the pre-registration)

Every derived number is *estimated* from *device* inputs, printed = 0.95 × steady, O = 8 ms (6 and 12 beside it)
until L6 replaces the sync part of it.

- **Body time.** t_rule(s, set) and t_stock(s, set) from T, per subject s.
- **Token price.** P(s) = 0.95 × 1000 ÷ (min over sets of t_rule + t_head(s) + O), per subject.
- **K1 lever, per subject.**
  - Kill if t_rule ÷ t_stock ≥ 1.00 cold on every core set.
  - Open if ≤ 0.90 on the best set.
  - In between, parked.
  - Reported per subject, with what generalised: Yi is the GQA subject, Llama-2 and DeepSeek the MHA class.
- **4th core (bit-identical).**
  - Adopted for the next round if t_stock(X1 + 3 A78) ≤ 0.95 × t_stock(X1 + 2 A78) on at least two subjects.
  - Closed if it is ≥ 1.00 on all three.
  - Read with L3: if c4 ÷ c3 < 1.10, the lever is closed on compute, whatever T says.
- **K0 update.** If P(s) < 12 for a subject at its best set, that subject is "out of reach with K1 and the 4th
  core". The statement is written with t_rule, r4, c4 and t_head. Only the unbuilt levers remain for it (C3 GPU
  lane, C5 up rule, C6, C7), each with a price that must cover the gap 1000 × 0.95 ÷ 12 − (t_rule + t_head + O) ms.
- **K-type (per-type).**
  - A type whose 4-thread hot cost per weight leaves no room under the weight line for any subject at R69's C is
    dropped from the S3 choice.
  - If R69 holds the 3.4 GB step and no type below Q4_0 survives, the freeze is Q4_0 for all three subjects.
- **Consistency checks (void a block, not the round).**
  - Identity lines not all 1: void D2s and the K1 arm of T.
  - The probe stock cell differs from the ggml-graph stock cell by more than 10 % at the same set: report it, and
    price T with the ggml cell's ratio.
  - Clocks under the r16 floor in more than 20 % of a block's cells: void that block.
- **Void rules (kit):** as R69 (APK hash, power saving on, charger plugged, process killed).

What R72 cannot decide, and must not be read as deciding:
- any real-model number. No model is loaded, so the claim on PPL and the real decode overhead come from S4 and S5;
- residency (that is R69's P1 and D1).

## 4. Dependencies and order

- R69 runs first, as pre-registered. R72 can run in the same phone window right after it (kit r32 installs over
  r29: same package and certificate).
- If R69 voids, R72 still runs; its anchors give r and the four-core pair under its own state.
- Code: Tom writes `0003` (T, G, B, H blocks; ggml-graph stock cell). The kit app patch adds the new blocks to
  the probe page. Parser rows go into `tools/c3_probe_parse.py`. Identity (*host*): knob unset gives per-chunk PPL
  equal to stock at `-b 512`, `-ub 32`, `-ub 1` on one subject file.
- Bands: Bonnie and Ebony's memo-08 give centres and 80 % bands for L1 to L7 and P(s), per subject. The lead
  writes the pre-registration from them, and Annie commits it before the APK is uploaded to `HoeioUser/vollie-kits` as `r32/`.

## 5. S3 freeze: what is needed from the owner's laptop (one subject at a time; keep 10 GB free)

Current state (*host*, memo-06-work/tom-screens.md §5.1):
- The Q4_0 files below are kept.
- The F16 files were deleted after 40-chunk runs.
- No full-set PPL of any subject exists.
- The lab tree is `lab/llama.cpp-c3` (`620d2225c` + c2 series 0001 to 0009, commit `26f78f925`, CPU build of
  memo-02 §0); the text is `lab/c3/text/wiki.test.raw`.
- Free disk is about 50 GB today.

| subject | source (repo @ commit) | candidate served file (not yet frozen) | sha256 (*host*) |
|---|---|---|---|
| Yi-1.5-6B-Chat | 01-ai/Yi-1.5-6B-Chat @ `771924d1` | `yi15-6b-q4_0.gguf` (Q4_0 body, Q6_K head), 3,479,326,848 B | `702cc2ebaa3bd4933598e25b556961b95ac260ed15df74fa09df2b2b7b48a299` |
| Llama-2-7b-chat | meta-llama/Llama-2-7b-chat-hf @ `f5db02db` (gated) | `llama2-7b-q4_0.gguf`, 3,825,808,512 B | `8eb4952f2592fdfc1288695f6b85aaf48d5628ec5d2c75f9387d3144652ad4bf` |
| DeepSeek-LLM-7B-chat | deepseek-ai/deepseek-llm-7b-chat @ `afbda8b3` | `deepseek7b-bos-q4_0.gguf`, 4,000,061,152 B, **BOS on** | `f195d24f5e29884bdfe3a6586bff0115c1988956fabd2cf2d07a3505d35da9ec` (checked 2026-11-22) |

**DeepSeek BOS (lead decision, to be confirmed by the owner).**
- The served file is the converter's output with one metadata field set, `tokenizer.ggml.add_bos_token` false →
  true, using `gguf_set_metadata.py` (`lab/c3/logs/bos.log`: one byte differs).
- This follows the publisher's `tokenizer_config.json` (`add_bos_token: true`). The converter's choice of false
  followed the tokenizer post-processor.
- It is a metadata change made before the freeze, not a change to a frozen file. The weights are byte-identical.
- The F16 reference gets the same one-field change, so that F16 PPL, stock PPL, the budget and the phone all
  score with BOS.
- The 40-chunk cells with BOS are 8.3095 at stock and 8.6276 at 40 % with k and v dense, `-b 512` (*host*): +3.83 %.
- The `-ub 1` cells with BOS are in `lab/c3/logs/bos-deepseek7b-ub1-40.log` (*host*). Tom reports both cells in
  memo-08 before the slot-3 rule is final (memo-06 §6 correction).

**Commands** (owner's laptop, repository root; `B=lab/llama.cpp-c3/build/bin`, `PY=lab/venv-c3/bin/python`). The
F16 conversion is the one memo-06-work/tom-screens.md §5.1 used; `<hf-dir>` is the publisher's snapshot at the
pinned commit, with every shard's sha256 checked against its LFS id.

```sh
# 1. F16 reference (re-made; recorded, then deleted after its PPL runs)
$PY lab/llama.cpp-c3/convert_hf_to_gguf.py <hf-dir> --outtype f16 --outfile lab/c3/models/<m>-f16.gguf
#    DeepSeek only: same BOS field as the served file
$PY lab/llama.cpp-c3/gguf-py/gguf/scripts/gguf_set_metadata.py lab/c3/models/deepseek7b-f16.gguf tokenizer.ggml.add_bos_token true --force
shasum -a 256 lab/c3/models/<m>-f16.gguf
#    Check: re-quantising this F16 to Q4_0 gives the candidate file's sha256 above (proves the pair belongs together)
$B/llama-quantize lab/c3/models/<m>-f16.gguf lab/c3/models/<m>-requant-q4_0.gguf Q4_0 && shasum -a 256 lab/c3/models/<m>-requant-q4_0.gguf

# 2. Full-set gate, wikitext-2 test, ctx 512, -b 512, all chunks (tools/ppl.sh; adds en.txt if present)
tools/ppl.sh --llama-dir $B --text lab/c3/text/wiki.test.raw c3-<m>-f16  lab/c3/models/<m>-f16.gguf
tools/ppl.sh --llama-dir $B --text lab/c3/text/wiki.test.raw c3-<m>-q4_0 lab/c3/models/<served file>

# 3. Stock at -ub 1, all chunks: the -ub 1 reference for every numerics lever (not needed for the freeze itself;
#    about 2 to 4 h per model on 8 threads, estimated)
$B/llama-perplexity -m lab/c3/models/<served file> -f lab/c3/text/wiki.test.raw -c 512 -b 512 -ub 1 -t 8 -ngl 0 \
  > lab/logs/p/ppl-c3-<m>-q4_0-ub1-wiki.log 2>&1
```

**Freeze record per subject** (a model card from `models/_TEMPLATE.md`; Annie commits):
- repo @ commit and the sha256 of every input shard;
- the F16 sha256 and, for DeepSeek, the BOS note;
- the served sha256 and byte size;
- F16 PPL and served PPL, full set;
- budget = served × 1.05 (absolute) and ceiling = F16 × 1.20 (absolute), with served ≤ ceiling checked;
- the resident estimate (file + KV at ctx 512 + buffers) against R69's C;
- the -ub 1 stock PPL once run;
- the K1 rule level at the full set:
  - Llama-2 at 40 % with k and v dense, or 35 % if the full set fails;
  - DeepSeek with BOS;
  - Yi at 42 % (memo-07) or its last passing level.
  This is checked with the same rule at `-b 512` and `-ub 1` before any lever round, not at the freeze.

**Type choice at the freeze.**
- Q4_0 is the working type: K1, the k,v-dense rule and every *host* frontier are Q4_0.
- A different type is considered only if R69 fails the 3.4 GB step (C < 3.4 makes Yi, at 3.48 GB, non-resident)
  or if R72's G block shows a type that is no slower than Q4_0 per weight.
- Then the S3 rule of memo-06 §5.3 applies (the lowest priced token time among types inside the ceiling and C), and
  that type's F16-relative PPL is run as in step 2.
- The freeze can therefore be recorded for Q4_0 now. It is final for each subject when R69's C is known; if C
  forces a different type, the Q4_0 record is superseded by a dated note.
