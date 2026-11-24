# c3 memo-12 work S-T1: llama.cpp's own n-gram speculation against our lookup simulation

- **Date:** 2026-11-24. **Author:** tom, `claude-opus-5-5`. Not committed. Test S-T1 of
  [ebony-joins-r16.md](ebony-joins-r16.md) §5.
- **Labels.** Every run is *host*: the owner's Apple M4 Pro laptop, CPU, `-t 4`, one run at a time. The other Tom's
  KleidiAI stock perplexity was running beside these runs. Pass counts and texts are exact counts from those runs.
  Timings are not reported as speeds. Nothing ran on the phone.
- **Subjects:** Yi-1.5-6B and Llama-2-7B, served Q4_0 files. DeepSeek is left out until the owner decides on its
  prompt.

## 0. Answer first

1. **At the settings asked for, upstream drafts nothing.** `ngram-simple` with M = 1 and N = 2 or 3 made 0 drafts
   in all 36 runs, so its output and pass count equal a plain run.
   - The cause is in `common_ngram_simple_draft` (`common/ngram-map.cpp`, *source*):
     `if (copy_max < n_draft_min) return {}`. A draft shorter than the n-gram is dropped, so M < N never drafts.
   - The smallest working settings are therefore N = M = 2 and N = M = 3. Both were run as well.
2. **Upstream equals a 40-line mirror of its own rule exactly.** `tom_st1.py` (`mirror()`) mirrors upstream's lookup
   and its verify loop on the plain greedy tokens. It gives the same pass count as upstream in all 36 runs of
   `ngram-simple` (N = M = 2 and N = M = 3; 9 prompts, 2 subjects). Upstream's behaviour can therefore be priced
   offline from dumps.
3. **Against Bonnie r24's simulation** (same tokens, first 32 per prompt):
   - **Pass count, closest arm.** The closest arm is her lookup restricted to a match length of at least 2 (rule T's
     lookup half, without the head). Upstream N = M = 2 is within one pass of it on 17 of 18 prompts. It is never
     worse: it is one pass *better* on 8 prompts, because it copies two tokens on a match, and 2 passes better on
     Llama-2 eval7 (21 against 23).
   - **Totals over 9 prompts:** Yi 1.059 tokens per pass against 1.047; Llama-2 1.138 against 1.108.
   - **Her "lookup only" arm** (3, else 2, else 1 tokens) also uses one-token matches, which upstream cannot do. It is
     higher on Yi (1.099; eval2 25 passes against upstream's 32) and about equal on Llama-2 (1.143).
   - **Official prompts:** Yi 31 / 31 / 31 passes (upstream N = M = 2 / Bonnie lookup / Bonnie lookup ≥ 2).
     Llama-2 25 / 25 / 26. Ebony's event ("N = 2 reproduces our lookup passes on the official prompt within one
     pass") holds at N = M = 2. At the literal M = 1 it fails on Llama-2 (32 against 25).
   - **Rule T** needs the trained head: Yi 1.346 and Llama-2 1.548 over these 9 prompts. No upstream type supplies a
     head guess. Upstream can stand in only for the lookup half.
4. **Text.** 106 of 108 speculative runs give exactly the plain run's tokens over the whole run (33 or 34 tokens).
   - **Two runs differ:** Yi eval7 with `ngram-simple` N = M = 2 and with `ngram-mod` n-match 2. Both first differ
     at generated token 30, where 593 becomes 5962. Both are N = 2 runs, which verify drafts of width 2 to 5 at that
     point.
   - The loop keeps only tokens the model itself picks, so this is not a logic error. It is numerics: the verify
     batch (ne1 > 1) runs ggml's batched matmul, and its sums differ slightly from the one-token path's. A near-tie
     at token 30 flips.
   - **So upstream's speculation is "exact" only up to batch numerics,** on this host build (repack Q4_0 4x4, no
     KleidiAI). It is not bit-identical to plain greedy decode on every prompt.
   - This is the same question the W-position kernel and the KleidiAI-build checks face. A phone build would need
     its own text check.
5. **The verdict does not move.** On these 18 prompts the lookup alone gives 1.04 to 1.15 tokens per pass. Upstream
   removes the lookup build item; it does not raise the price. The width-W verify still runs ggml's stock batch
   path.

## 1. Build and tool

- **Tree:** `lab/llama.cpp-base` at `620d2225c`, with no source changes, built in a new directory `build-spec`.
  - Options: `GGML_NATIVE=OFF`, `GGML_CPU_ARM_ARCH=armv8.2-a+dotprod+fp16`, `GGML_METAL=OFF`, `GGML_BLAS=OFF`, KleidiAI
    off, repack on. These are the same as the freeze build.
  - Target `llama-speculative-simple` (sha256 first 16 `c07cab801c28851a`); `libggml-cpu` `1c9f5acef1398429`.
  - The tree's `git status` shows `ggml/src/ggml-cpu/vollie-release.h` as deleted. No file in that tree includes it.
- **Tool:** `tools/c3_spec_ids.cpp`, built by `tools/c3_spec_ids.sh` into `build-spec/bin/c3-spec-ids` (`2ceaae85530f337b`).
  - It is upstream's `examples/speculative-simple/speculative-simple.cpp`, changed in only three ways:
    1. `--ids FILE` takes the prompt as token ids, so nothing is re-tokenised.
    2. `--spec-type none` runs plain through the same loop. Upstream returns an error there.
    3. It prints one `c3spec` line: generated ids, and per pass "drafted : accepted".
  - The draft, verify and roll-back code is upstream's.
- **Prompts:** Tom's dump ids (`lab/c3/head/<m>/{official,eval}.ptok`), i.e. the official prompt and the first 8 of
  the 32 set prompts (eval index 0 to 7).
  - These are the ids the dump tool rendered with each file's own chat template. Yi has no BOS; Llama-2 starts with
    BOS 1.
  - Prompt lengths: Yi 167 / 32 / 31 / 35 / 30 / 233 / 26 / 52 / 287; Llama-2 186 / 21 / 20 / 23 / 18 / 253 / 15 / 41 / 280.
- **Settings:** greedy (`--temp 0 --top-k 1`), `-n 32`, `-c 1024`, `-t 4`, `-ngl 0`.
  - Upstream's loop stops after the first pass that takes it past 32, so runs have 33 or 34 tokens. Pass counts are
    taken over the first 32 tokens.
  - Configurations:
    - `ngram-simple`: N = 2, M = 1; N = 3, M = 1; N = 2, M = 2; N = 3, M = 3;
    - `ngram-mod`: n-match 2 and 3, n-min 1, n-max 4 (upstream warns that n-match < 16 is "too small").
- **Run:** `lab/c3/queue-st1.sh`, 126 runs, 09:32 to 09:42, results in `lab/c3/spec/st1.txt`. Analysis:
  `python3 -I -B tom_st1.py <repo root>` (this folder).

## 2. Plain run against the dump

- The plain run equals the dump's greedy tokens over the first 32 on 17 of 18 prompts.
- **Yi eval7 differs at token 7.** Likely cause, not checked: the dump tool decodes all prompts in parallel (`n_par`
  sequences in one batch, unified KV), so its decode runs the batched path, as in §0 item 4.
- The mirror and Bonnie's arms below therefore use this build's plain tokens as the truth.
- For Yi eval7 only, the head's guesses used by "B rule T" were made on the dump's context. Past token 7 that one
  value is approximate.

## 3. Tokens per pass over the first 32 tokens (passes in brackets; `*k` = text differs from plain, first at token k)

Columns:
- "mir" = the mirror of upstream's `ngram-simple`.
- "B" = Bonnie r24's arms, as copied from `bonnie_claim_guess.py`. Her script on its own reproduces the official
  values shown here: Yi 1.032 / 1.455, Llama-2 1.280 / 1.333.

**Yi-1.5-6B**

| prompt | simple n2 m1 | simple n3 m1 | simple n2 m2 | simple n3 m3 | mod n2 d4 | mod n3 d4 | mir n2m2 | mir n3m3 | B lookup | B lookup ≥ 2 | B rule T |
|---|---|---|---|---|---|---|---|---|---|---|---|
| official | 1.000 (32) | 1.000 (32) | 1.032 (31) | 1.000 (32) | 1.032 (31) | 1.000 (32) | 1.032 (31) | 1.000 (32) | 1.032 (31) | 1.032 (31) | 1.455 (22) |
| eval0 | 1.000 (32) | 1.000 (32) | 1.000 (32) | 1.000 (32) | 1.000 (32) | 1.000 (32) | 1.000 (32) | 1.000 (32) | 1.032 (31) | 1.000 (32) | 1.333 (24) |
| eval1 | 1.000 (32) | 1.000 (32) | 1.103 (29) | 1.103 (29) | 1.143 (28) | 1.103 (29) | 1.103 (29) | 1.103 (29) | 1.103 (29) | 1.067 (30) | 1.280 (25) |
| eval2 | 1.000 (32) | 1.000 (32) | 1.000 (32) | 1.000 (32) | 1.000 (32) | 1.000 (32) | 1.000 (32) | 1.000 (32) | 1.280 (25) | 1.000 (32) | 1.333 (24) |
| eval3 | 1.000 (32) | 1.000 (32) | 1.000 (32) | 1.000 (32) | 1.000 (32) | 1.000 (32) | 1.000 (32) | 1.000 (32) | 1.000 (32) | 1.000 (32) | 1.333 (24) |
| eval4 | 1.000 (32) | 1.000 (32) | 1.032 (31) | 1.000 (32) | 1.032 (31) | 1.000 (32) | 1.032 (31) | 1.000 (32) | 1.067 (30) | 1.032 (31) | 1.455 (22) |
| eval5 | 1.000 (32) | 1.000 (32) | 1.000 (32) | 1.000 (32) | 1.000 (32) | 1.000 (32) | 1.000 (32) | 1.000 (32) | 1.032 (31) | 1.000 (32) | 1.333 (24) |
| eval6 | 1.000 (32) | 1.000 (32) | 1.333 (24) | 1.231 (26) | 1.455 (22) | 1.280 (25) | 1.333 (24) | 1.231 (26) | 1.391 (23) | 1.280 (25) | 1.455 (22) |
| eval7 | 1.000 (32) | 1.000 (32) | 1.103 (29) *30 | 1.032 (31) | 1.103 (29) *30 | 1.032 (31) | 1.103 (29) | 1.032 (31) | 1.067 (30) | 1.067 (30) | 1.185 (27) |
| **all 9** | 1.000 (288) | 1.000 (288) | **1.059 (272)** | 1.036 (278) | **1.071 (269)** | 1.040 (277) | 1.059 (272) | 1.036 (278) | 1.099 (262) | 1.047 (275) | 1.346 (214) |

Drafted / accepted over whole runs: simple n2 m2 57 / 17, n3 m3 30 / 10; mod n2 70 / 20, n3 36 / 11.

**Llama-2-7B**

| prompt | simple n2 m1 | simple n3 m1 | simple n2 m2 | simple n3 m3 | mod n2 d4 | mod n3 d4 | mir n2m2 | mir n3m3 | B lookup | B lookup ≥ 2 | B rule T |
|---|---|---|---|---|---|---|---|---|---|---|---|
| official | 1.000 (32) | 1.000 (32) | 1.280 (25) | 1.143 (28) | 1.280 (25) | 1.185 (27) | 1.280 (25) | 1.143 (28) | 1.280 (25) | 1.231 (26) | 1.333 (24) |
| eval0 | 1.000 (32) | 1.000 (32) | 1.000 (32) | 1.000 (32) | 1.000 (32) | 1.000 (32) | 1.000 (32) | 1.000 (32) | 1.032 (31) | 1.000 (32) | 1.391 (23) |
| eval1 | 1.000 (32) | 1.000 (32) | 1.143 (28) | 1.103 (29) | 1.143 (28) | 1.143 (28) | 1.143 (28) | 1.103 (29) | 1.103 (29) | 1.103 (29) | 1.524 (21) |
| eval2 | 1.000 (32) | 1.000 (32) | 1.143 (28) | 1.103 (29) | 1.143 (28) | 1.143 (28) | 1.143 (28) | 1.103 (29) | 1.231 (26) | 1.103 (29) | 1.524 (21) |
| eval3 | 1.000 (32) | 1.000 (32) | 1.067 (30) | 1.032 (31) | 1.067 (30) | 1.032 (31) | 1.067 (30) | 1.032 (31) | 1.067 (30) | 1.032 (31) | 1.524 (21) |
| eval4 | 1.000 (32) | 1.000 (32) | 1.185 (27) | 1.185 (27) | 1.185 (27) | 1.185 (27) | 1.185 (27) | 1.185 (27) | 1.143 (28) | 1.143 (28) | 1.778 (18) |
| eval5 | 1.000 (32) | 1.000 (32) | 1.032 (31) | 1.000 (32) | 1.032 (31) | 1.000 (32) | 1.032 (31) | 1.000 (32) | 1.032 (31) | 1.032 (31) | 1.684 (19) |
| eval6 | 1.000 (32) | 1.000 (32) | 1.032 (31) | 1.000 (32) | 1.032 (31) | 1.000 (32) | 1.032 (31) | 1.000 (32) | 1.103 (29) | 1.032 (31) | 1.600 (20) |
| eval7 | 1.000 (32) | 1.000 (32) | 1.524 (21) | 1.684 (19) | 1.778 (18) | 1.600 (20) | 1.524 (21) | 1.684 (19) | 1.391 (23) | 1.391 (23) | 1.684 (19) |
| **all 9** | 1.000 (288) | 1.000 (288) | **1.138 (253)** | 1.112 (259) | **1.152 (250)** | 1.121 (257) | 1.138 (253) | 1.112 (259) | 1.143 (252) | 1.108 (260) | 1.548 (186) |

Drafted / accepted over whole runs: simple n2 m2 91 / 37, n3 m3 60 / 29; mod n2 158 / 40, n3 72 / 31.

## 4. Reading

- **Arms.**
  - `ngram-mod` with n-match 2 and a draft of up to 4 is the best upstream arm on both subjects (Yi 1.071, Llama-2
    1.152). It drafts long: 70 and 158 tokens drafted for 20 and 40 accepted.
  - On a bus-bound phone, a wasted draft position costs one more column in the verify batch. The tokens-per-pass
    gain over N = M = 2 (+0.012 / +0.014) is unlikely to pay for 13 / 67 more drafted positions (*estimated*,
    not priced here; the researchers' batch-cost model should price it).
  - N = M = 2 is the arm to price: its pass counts are exactly the mirror's, and the mirror can run on any dump.
- **Upstream as the lookup source of rule T.** Rule T uses the lookup only at match length ≥ 2, then guesses one
  token. Upstream N = M = 2 is the same lookup with a 2-token copy. Over these prompts it is within one pass of rule
  T's lookup half on 17 of 18 prompts, and never worse.
  - It cannot supply rule T's head fallback, which carries most of rule T's gain (1.346 / 1.548 against about 1.05 /
    1.11 for the lookup alone).
  - Using upstream would need either a custom drafter (a new `common_speculative_impl`) or the head as a draft model
    type.
- **Exactness.** Upstream keeps a draft only when the verify batch picks it. The verify batch is a batched matmul,
  so the text equals plain greedy decode only up to batch numerics.
  - Here, 2 of 108 runs changed text at token 30 of one prompt.
  - Any multi-token method on the phone (lookup, copy-run, head guesses) carries the same caveat. Its budget is a
    text or PPL check per build, not bit identity.
  - A per-column-identical verify kernel (the W-position kernel) would remove the caveat.

## 5. Limits

- 9 prompts per subject, 32 tokens each. These are per-prompt counts, not a distribution.
- One host build (repack Q4_0, no KleidiAI). The text-flip count depends on the batch kernels, so the phone's KleidiAI
  path can differ.
- `ngram-mod` uses a hash table, and I did not mirror it. Its counts are upstream's own.
- llamadart's speculative settings (S-T2) were not read here.
