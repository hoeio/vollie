# memo-10 work file, Clyde: plan for R73 (first real stock decode), and the written line for lookahead

Date: 2026-11-22. Author: Clyde (`claude-opus-5-5`). **This is a plan, not the pre-registration.** Not committed.
Inputs:
- R72-result.md §5, §9 and §11 (the revised order);
- memo-09-gap.md, including Ebony's review §10;
- memo-09-work/tom-host.md (0004 guard, four-way labels, GPU build);
- memo-09-work/tom-head.md (T1, T1b, T5);
- clyde-freeze.md (this folder).

Labels: *device* = phone (R69, R72); *host* = laptop; *estimated* = arithmetic. Nothing here is a new measurement.

## 1. Where R73 sits

1. **S3 freeze** on the laptop (clyde-freeze.md). It is required before R73, because R73 downloads the frozen files
   by sha256.
2. **R73, device: first real stock decode** on Yi-1.5-6B and Llama-2-7B, plus DeepSeek-LLM-7B as the third block.
   R73 is a calibration round. It tests whether the K0 arithmetic (built from probe cells) predicts a real
   `decodeTokensPerSec`. It also carries the placement and clock logging that every core-set number since R69
   lacked.
3. **R74, device: GPU lane probe x3** (tom-host §1.5), with T2 (batch-2 cost) and T4 (Hexagon reachability) as riders.
   It is planned after R73. Its kill lines are R72 §11 item 2 with Ebony's additions (memo-09 §10.10 T3).

The goal stays at 12 tok/s on each of the three. On present prices it is out of reach. The best priced stack is
9.9 / 9.0 / 8.9 tok/s (Ebony) or 11.7 / 10.6 / 10.6 (Bonnie, with lever M at p 0.60, which T5 has since closed). In
both cases P(all three) is about 0.003 to 0.005 (*estimated*, judgement). R73 does not change that, and it is not
meant to.

## 2. Design (draft)

**Kit r34** (next number after r33). It is campaign 1 and 2's research app with one new block, "c3 decode R73",
started by its own button.

**Runtime.** The kit r33 series (`patches/llama.cpp/0*`, `llama.cpp-c1-olmoe/0001` to `0012`, `llama.cpp-c3/0001` to
`0003`) plus `llama.cpp-c3/0004-pin-guard.patch` (sha256 `d6c665995b686c4c26cac0971e3434ab80507b37d9aa644b80bc632385ef008b`).
Every knob is off except the ones named below. This is the start-from-Vollie-1 rule: the c2 pool, pin, hold and the
r16 gate come from campaigns 1 and 2. The only new runtime code is 0004.

**Models, in this fixed order:** Yi, Llama-2, DeepSeek. If the phone or the time stops the block early, the first two
still meet the two-model rule.

| item | value | why |
|---|---|---|
| files | the three frozen served files, downloaded by the kit (§7); sha256 checked before load | frozen |
| repack | on (`KitModel.repack: true`), i.e. llama.cpp's default: this is stock | stock = the upstream default path |
| threads, cores | 4; `VOLLIE_POOL=1`, `VOLLIE_CPUMASK=0x10,0x20,0x40,0x80` (thread *i* on cpu 4+*i*; Tom confirms it is the order of R72's adopted `4-7` cell) | R72 adopted the fourth core; K0's stock body is the `4-7` ggml cell |
| baseline env | `_c1Ship` (campaign 1's shipping settings without numerics) | start from Vollie 1 |
| context | **nCtx 512** for this block (the kit's fixed 4096 since r27 is not kept) | F16 KV at 4096 is 2.0 GiB on Llama-2 and 1.9 GiB on DeepSeek (MHA), against R72's anon42 not held; at 512 it is 256 / 240 / 32 MiB (Yi). The official prompt (702 characters) and 32 tokens fit |
| prompt | the official prompt (`tools/c2_official_prompt.txt` = the app's `_benchPrompt`), 32 tokens, greedy, the model's own chat template | the metric |
| block per model | load; 2 unscored warm-ups; **12 gated `official32` runs, ABBA by arm (A B B A A B B A A B B A)**, 6 per arm; then 2 unscored `profiled32` runs (one A, one B) | r28 claim format, with r16's gate (90 s) and hold, kv clear every run |
| arm A | guard off | stock |
| arm B | `VOLLIE_PIN_GUARD=500`, `VOLLIE_PIN_WARM=0` | tom-host §2.1: start at a small budget and read the counters |
| phone state | cool start, power saving off, charger out, deep-sleeping list recorded, no other model loaded, the app in front | as R69 and R72 |

Time (*estimated*):
- Per model: a load with repack of 3.5 to 4.0 GB, then 14 timed runs at up to 90 s of gate plus about 15 s of
  prefill and decode each, then 2 profiled runs. That is about 25 to 30 minutes.
- About 80 to 90 minutes for all three, plus the downloads (11.3 GB).

## 3. Kit additions (what Tom builds; all logging only, no numerics)

| # | addition | where | feeds |
|---|---|---|---|
| K-a | Placement guard 0004 as arm B; its `c2-pin` line per prompt (`checked:off at start:recovered:failed:us:off at end` per worker) | runtime 0004 (exists); kit sets the env per arm | guard rule |
| K-b | **Affinity read-back**: each worker's `sched_getaffinity` mask, printed in the `c2-pin` line (`aff=<hex>`) at the prompt's first and last graph | small addition to 0004; still bit-identical, re-checked at `-b 512`, `-ub 32`, `-ub 1` | label M (the half r33 could not give) |
| K-c | **Per-token cpu**: the existing `c2-pool` histogram per prompt (one `sched_getcpu` per worker per graph, so per decoded token) | exists (0010) | label M |
| K-d | **Clock caps against the hardware maximum**: per policy (0, 4, 7) `scaling_cur_freq`, `scaling_max_freq` and `cpuinfo_max_freq`, sampled at 1 Hz during each run and once in the gate wait | kit (DeviceProfile sampler) | label U; R72's 1,555 to 1,785 MHz A78 cap |
| K-e | **Thermal zones**: every `/sys/class/thermal/thermal_zone*/{type,temp}` at gate start, gate end and run end | kit | heat beside every timing (COMMON_RULES 8) |
| K-f | **core_ctl and isolation** (R72 §10.1): `cpu7/online`, `/sys/devices/system/cpu/isolated`, `cpu*/core_ctl/*` if readable (else `e<errno>`), at run start and end | kit | Ebony's isolation hypothesis |
| K-g | **T6 riders**: every `/sys/class/devfreq/*/{cur_freq,max_freq,min_freq,governor,available_frequencies}` readable without root (bus and DDR nodes), 1 Hz during runs; THP `enabled` and `defrag` at block start | kit | T6 line (§5) |
| K-h | **Greedy token ids** of every run (32 ids) | kit report | identity A vs B; the device greedy text for later multi-token work |
| K-i | Residency per run, as r28: majflt, minflt, VmRSS, VmSwap, MemAvailable, zram `mm_stat`, PSI (`hdr` probe line) | exists | anon42 lesson |
| K-j | **Parser**: Tom's four-way label (M / U / P / C) applied to the decode workers from K-a to K-d, a per-run label table, and the K0 check of §5 | `tools/c3_probe_parse.py` or a sibling | every rule below |

Before the pre-registration, Tom does three host checks and reports them:
- the runtime with 0004 plus K-b, identical per-chunk PPL to the series without 0004 at `-b 512`, `-ub 32` and `-ub 1`;
- the new kit block builds;
- the parser self-test passes on a synthetic decode report.

Tom also checks that the kit downloads straight into the folder it loads from. If it keeps a second, internal copy,
the phone needs about 2 × 11.3 GB.

## 4. Prediction (draft centres; the bands are Bonnie and Ebony's job before the pre-registration)

K0 stock (R72 §5, *estimated* from *device* cells): ggml stock body on `4-7` + Q6_K head + O 8 ms, printed 0.95 ×.

| subject | K0 stock, printed tok/s | ±10 % window |
|---|---|---|
| Yi-1.5-6B | 6.90 | 6.21 to 7.59 |
| Llama-2-7B | 6.21 | 5.59 to 6.83 |
| DeepSeek-7B | 6.22 | 5.60 to 6.84 |

Things that can pull a real decode below K0:
- The L4 cells were 1 to 4 s long. A sustained decode heats the phone, and the A78 cap fell to 1,555 MHz in L4.
- On Llama-2 and DeepSeek the repacked body (4.02 / 4.01 GB) is above the anonymous capacity R72 held (anon34 held,
  anon42 not), so swap or zram traffic is possible.
- Prefill heat carries into decode.

One thing can push a real decode above K0: O = 8 ms is an allowance, while graph sync measured 0.2 ms (L6).

The lead's prior is that Llama-2 and DeepSeek land below their windows more often than Yi. That is a judgement, to
be priced by Bonnie and Ebony.

## 5. Draft rules (to be written into `prereg/R73-…md` before the run)

1. **K0 check, per subject (the pass rule).** Take the median `decodeTokensPerSec` of the 6 arm-A runs.
   - **Pass** if it is inside the ±10 % window of §4.
   - Otherwise **K0 is rebuilt for that subject before any lever is priced on it.** The rebuild uses the profiled
     runs' per-token split (body, head, other) and the labels. Every price in memo-09 and R72 §5 for that subject is
     then marked superseded by a dated note.
   - Scored per subject and reported per subject: what generalised and what did not.
2. **Guard (arm B against arm A), on Yi and Llama-2** (DeepSeek descriptive).
   - **Adopt** if, on both subjects:
     - the cpu-7 worker is on-core in ≥ 0.90 of B's graph starts;
     - the median B ÷ median A is ≥ 1.00.
   - **Kill** if any of these holds:
     - B ÷ A < 0.97 on either subject;
     - `failed` > 0 and `recovered` = 0 in ≥ 0.90 of B's prompts. That means affinity cannot move the thread
       (core_ctl isolation), which no runtime call fixes.
     - the greedy ids of any B run differ from A's (the guard is not bit-identical on the device).
   - **Otherwise open.**
   - Also, as written in R72 §10.1 as a draft: the cpu-7 worker on-core in ≥ 90 % of samples under B.
3. **T6 (Ebony's rider).** **memo-08 §2.4 opens** if the DDR or bus devfreq node is below its top step in ≥ 20 % of
   1 Hz samples during A runs on any subject. The lever is "the DDR clock follows the CPU down": r in a real decode
   may be 10 to 25 % below 29.9 GB/s. If no such node is readable, the rider is "not measured", not a pass.
4. **Labels (descriptive, no verdict).**
   - Each worker in each run gets M / U / P / C. Thresholds are Tom's parser defaults (`ON_MIN` 0.9, `CLK_MIN` 0.9,
     `CW_MIN` 0.9, `NIV_HI` 50). They are written into the pre-registration so that they are fixed.
   - The K0 check is also reported on the subset of runs whose four workers are all C. That subset is not the
     verdict.
5. **Identity (descriptive within A).** All 6 A runs of a subject give the same 32 greedy ids. A run that differs is
   reported (it would mean run-to-run nondeterminism on the device).
6. **Voids (as R69 and R72):**
   - APK sha256 not as pre-registered;
   - power saving on, or the charger in;
   - a block killed before 6 A and 6 B runs.

   New for R73:
   - a served file whose downloaded sha256 differs voids that subject's block.

What R73 does not do:
- no lever, no numerics change, no K1;
- no GPU (R74);
- no fixed-prompt-set runs. No multi-token method is scored, and nCtx 512 is too short for the set's longest
  prompt plus 128 tokens.

## 6. The fixed prompt set: adopt Tom's `eval.txt`

**Decision (lead): adopt** `lab/c3/head/prompts/eval.txt`, sha256
`ff1f32e39eed0713daad5561278fbb9b292892bcb3c3efba3161b2feaa0572c0` (32 prompts, 11,125 bytes), as campaign 3's
fixed prompt set. It is used for every multi-token method, beside the official prompt (CAMPAIGN.md metric row).

Why:
- It was drawn by a rule written into `tools/c3_head_prompts.py` before any number was computed on it:
  - databricks-dolly-15k at HF commit `bdd27f4d`, file sha256 `2df90833…94ec`;
  - record index ≡ 7 (mod 100), at most 2,500 characters, the first 32 in file order.
  No prompt was picked by hand or by a result.
- It is disjoint from the head-training prompts by construction (asserted in the script).
- Its held-out and set top-1 agree within 0.01 to 0.02 (tom-head §4.1), so it is not an easy or a hard corner of
  Dolly.
- None existed, and T1, T5 and Y1 all need one.

Conditions, to go into the pre-registration that first scores on it:
- From adoption on, nothing is tuned on it. Guess sources, thresholds and heads are tuned on Dolly train and
  held-out only.
- The host values already computed on it (T1, T5) stay *host* and are not re-used as device predictions without
  their label.
- It is committed with its generator and attribution: `campaigns/03-dense15-5pct/prompts/eval.txt` plus a README
  that names the source, the licence (CC BY-SA 3.0, Databricks) and the sha256s. A public repository may carry it
  under that licence.
- Runs on the set use nCtx 1024. The longest prompt is 1,478 characters; with the template and 128 new tokens it
  needs more than 512.

## 7. How the served files reach the phone

- None of the three is hosted. The kit needs a public URL with a pinned commit and a sha256. The app downloads
  without a token, as `granite-Mi.gguf` and `olmoe-Mi.gguf` come from `HoeioUser/vollie2` at commit `e893e8a6…`.
- **Proposal: a new public model repository, `HoeioUser/vollie3`, for campaign 3's served files.** Do not add them
  to `vollie2`. Its card is tagged apache-2.0 and OLMoE-based, and two of the three new files carry other
  licences:
  - Yi-1.5-6B-Chat: apache-2.0.
  - Llama-2-7b-chat: Llama 2 Community License. Redistribution needs a copy of the agreement and a NOTICE file with
    "Llama 2 is licensed under the LLAMA 2 Community License, Copyright (c) Meta Platforms, Inc. All Rights
    Reserved."
  - DeepSeek-LLM-7B-chat: the DeepSeek model licence, whose use restrictions must be passed on.

  Each file goes into its own folder (`yi15-6b/`, `llama2-7b/`, `deepseek7b/`) with the publisher's licence text
  taken from the pinned commit. The card lists for each file: source repo @ commit, recipe (`tools/c3_freeze.sh`),
  bytes, sha256 and the DeepSeek BOS note.
- **Upload** happens after the freeze is committed. Either the owner runs it:

  ```sh
  hf repo create HoeioUser/vollie3 --repo-type model     # public
  hf upload HoeioUser/vollie3 lab/c3/models/yi15-6b-q4_0.gguf yi15-6b/yi15-6b-q4_0.gguf
  hf upload HoeioUser/vollie3 lab/c3/models/llama2-7b-q4_0.gguf llama2-7b/llama2-7b-q4_0.gguf
  hf upload HoeioUser/vollie3 lab/c3/models/deepseek7b-bos-q4_0.gguf deepseek7b/deepseek7b-bos-q4_0.gguf
  ```

  or the lead runs it with a write token the owner gives for that session (never written into a file).
- The licence files and the card go up in the same commit series.
- Then the lead reads the final commit sha from the HF API and verifies each file's LFS sha256 against §1 of
  clyde-freeze.md. Tom writes `kitModels` entries with that commit in the URL: `source: 'kit'`, `template: 'model'`,
  `repack: true`, and bytes and sha256 as frozen.
- **Phone storage:** 11.3 GB for the three files, more if the kit keeps an internal copy (§3). The kit checks free
  storage before each download and records "not measured: storage" instead of failing a block. The owner frees
  space if needed; that is the research phone, not a user's.
- Owner decisions needed: F3 (below).

## 8. Rule B reopening: the written line for lookahead (lever L)

**History, unchanged.** R69 Rule B scored KILL: R20, draft speculation, at r ÷ c_b = 1.17 against kill ≥ 0.71. Its
reopened host item, never opened, had the line "killed inside if the match rate is under 0.4 on any of the three
subjects" (R69 amendment 1 item 3). Both stand as scored.

**Why a new line, and why 0.4 no longer binds.**
- The lever is different. There is no draft model, one extra position, and a dedicated two-vector kernel at about
  1.45 × the per-weight issue count (memo-09 §5.1; Ebony 1.40 to 1.60).
- The regime is different. R72 measured the stock body as bus-bound: ggml 25.8 to 26.7 GB/s against a 29.9 GB/s read
  (*device*). R69's line was set when c_b (3 clock-depressed cores) put the second position's cost near 2 ×.
- So the break-even match rate is no longer a fixed number. It is p ≈ kB − 1, with kB the measured cost of a batch-2
  pass. A line in the measured pair replaces a line that assumed the cost.
- This is a reopening by the lead, with these reasons, written before T2 (the only device input) has run. It is
  not a re-score of R69.

**The two candidates:**
- **Bonnie:** close L if p < 0.20, or if kB > 1.20, each alone (on the official prompt or the set, on either
  model).
- **Ebony:** close L if (1 + p) ÷ kB < 1.05 on any subject.

**Choice: Ebony's joint line**, written with coverage so that a confidence-gated source is priced as it would run.

> **L is closed if G = (1 + p) ÷ (c × kB_eff + 1 − c) < 1.05 on any subject, on the official prompt or on the fixed
> prompt set.** p = share of decode steps whose single guess is right; c = share of steps that carry a guess (run at
> batch 2; the rest run at batch 1); kB_eff = the device batch-2 pass time ÷ the batch-1 pass time at subject shapes,
> sustained ≥ 60 s on `4-7` (T2), plus any per-pass guess cost ÷ the body time. With c = 1 this is Ebony's line
> exactly. p and c come from a replay of the device greedy text (R73, official prompt) where it exists, and the host
> replay otherwise (labelled). The guess source is fixed before T2 runs (below). The line is scored per subject and
> reported per subject.

**Reasons:**
1. G is the quantity the lever is worth: the predicted body gain per committed token. Separate lines can close a
   lever that pays and keep one that does not:
   - p 0.19 at kB 1.00 is G 1.19, and Bonnie's line closes it;
   - p 0.21 at kB 1.19 is G 1.02, and Bonnie's line keeps it.
2. A p cut on the official prompt is one hit wide. There are 31 guesses per model, so 6/31 = 0.194 and 7/31 = 0.226.
   A line on G spreads the decision onto kB, the half that is not yet measured.
3. 1.05 is the "under 5 %" both reviewers used. It is also about the size of effect that in-run ABBA has scored
   before. That second point is the lead's recollection of the R-log; it was not re-checked for this file.
4. "Any subject" follows R69's Rule B wording and both reviewers. On present prices no subject reaches 12 without
   every lever, so a lever that cannot work on one subject cannot carry the three-model goal. It is still reported
   for the subjects where it works.

**What is already known, stated so that nobody reads the choice as made blind.** The host T1 values (tom-head §6)
were known when this line was chosen. Bonnie's line closes L today on Yi's official prompt: the best model-free arm
is f at p 0.194. The chosen line instead hands the decision to kB. With the host p and c, the largest kB_eff each
model-free source can have and still pass (*estimated*, G ≥ 1.05):

| source | Yi set | Llama-2 set | Yi official | Llama-2 official | worst cell |
|---|---|---|---|---|---|
| a, prompt lookup | 1.344 | 1.334 | 1.032 | 1.519 | 1.032 |
| e, last-logit rescoring | 1.341 | 1.388 | 1.028 | 1.495 | 1.028 |
| **f, counted 3-gram table** | 1.182 | 1.223 | 1.142 | 1.075 | **1.075** |
| b, prompt bigram | 1.223 | 1.257 | 0.956 | 1.498 | 0.956 |
| c, self-successor | 1.173 | 1.169 | 0.631 | 0.267 | 0.267 |

- **Guess source, fixed now:** the model-free source with the highest worst cell, **f**. The rule is the
  max-min over the measured arms, so it is chosen by the rule, not by preference.
  - f is a calibration-derived table under the owner's 2026-11-19 decision: a sidecar, sha-keyed, generated by a
    committed script.
  - It is counted from the model's own greedy outputs on Dolly train prompts, which tom-head used. That is the
    "fixed calibration text" for this table, named in its header.
- **So L lives only if the sustained device kB_eff ≤ 1.075** (Bonnie's centre is 1.08, Ebony's 1.15).
  - Even at kB 1.00, f's G on the official prompt is 1.19 / 1.13 (Yi / Llama-2), a body cut of about 16 % / 11 %.
    That is about +1 tok/s on Yi (*estimated*).
  - L is a small lever, not a route to 12.
- DeepSeek has no T1 yet. Tom runs T1 on DeepSeek (host, free) before T2 is scored, because the line is "any
  subject".

**Lever M (trained head), scored as written.** T5's line was "head top-1 < 0.50 → M closed" (memo-09 §7.1). It was
missed on both models: Yi 0.379, Llama-2 0.497 (*host*). **M is closed.**
- Using the head as L's guess source (tom-head §8 item 1) would be M under another name. It is not taken.
- A re-test of Llama-2 with doubled data would need its own line written first. The lead does not order it: Yi
  would need 5 to 7 more doublings (tom-head §4.3), so M cannot carry the three-model goal.
- The owner's 2026-11-22 permission for heads stays on record. It is simply not used.

**Consequence for R74.** T2 rides R74 as planned (about 6 phone-minutes; Ebony's three arms, each ≥ 60 s). It is
the only remaining input L needs. Nothing for L is built (Y1) before T2 passes the line on Yi and Llama-2.

## 9. Owner commands (summary; the exact ones go into the pre-registration)

1. **Now, laptop:** the freeze (clyde-freeze.md §4). About 12 to 21 h serial (*estimated*), about 28 GB peak disk.
2. **After the freeze, laptop or lead with a token:** create `HoeioUser/vollie3` and upload the three files (§7), if
   the owner agrees to F3.
3. **After the R73 pre-registration is committed** (the owner tags `prereg-R73`): install kit r34 from
   `HoeioUser/vollie-kits` `r34/` (APK sha256 in the pre-registration). Set the phone state of §2, press
   "c3 decode R73", and save the report JSON into `campaigns/03-dense15-5pct/reports/device/`.

## 10. Owner questions

- **F1.** DeepSeek BOS on, as frozen (clyde-freeze.md §7).
- **F2.** F16 at `-ub 1`: not needed (clyde-freeze.md §3).
- **F3.** Host the three served files in a new public HF repo `HoeioUser/vollie3`, with each publisher's licence
  (Llama 2 Community License with its NOTICE line; DeepSeek model licence; apache-2.0)? Or name another place. The
  kit cannot carry a token, so the files must be public or side-loaded.

## 11. Before the pre-registration (checklist)

- Freeze done; three model cards committed (Annie).
- Files hosted, LFS sha256 verified, commit pinned (lead).
- Kit r34 built, host identity of 0004 + K-b, parser self-test (Tom).
- Bands for §4 from Bonnie and Ebony (centres and 80 % bands per subject, with the residency and heat risks
  priced).
- The §8 line and §6 set copied into the pre-registration verbatim.
- "Agent models" field: clyde and tom `claude-opus-5-5`, bonnie and ebony `claude-fable-5-2`, annie
  `claude-sonnet-5-5`.
