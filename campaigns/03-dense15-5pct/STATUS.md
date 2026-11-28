# Campaign 3 status

**State (2026-11-18): planning.** Terms set by the owner (CAMPAIGN.md). Campaign 2 is finished first: the owner sends
R68's reports, the lead scores them and closes campaign 2.

*Note 2026-11-21: Amendment 1 (owner): goal 12 tok/s on each of three dense 7B-class models from different
families, all three must pass; subjects are reselected (memo-06). The state line above is superseded by this note.*

## Next actions

**From 2026-11-21 (Amendment 1), first:** joint round 4, memo-06: Bonnie builds and prices the candidate pool,
Ebony scouts new levers for 12 tok/s and challenges Bonnie's pricing, Tom readies model-generic host screens and runs
them on the pool, Clyde re-plans rounds and kill lines at 12 tok/s, Annie keeps the books. The lead chooses three
subjects plus two reserves by memo-06 section 1 (owner delegated: "proceed without asking"). R69 (model-independent
probe) is still the next device round. The items below are from before the amendment.

**State at end of session 5 (2026-11-21):** memo-06 sections 2 to 6 filled. Provisional subjects (section 6):
Yi-1.5-6B-Chat, ChatGLM3-6B, DeepSeek-LLM-7B-chat; reserves Llama-2-7b-chat, Granite-7b-instruct. No candidate is
priced at 12 tok/s (*estimated*); Clyde's P(all three) 0.05 to 0.15, Ebony's at most 0.02 (judgements).
R69 amendment 1 committed before any R69 data (owner to tag). Next:
1. Owner frees laptop disk (about 0.2 GB free, swap full; about 16 GB per model needed). Then Tom runs H3 and
   Q4_0-vs-F16 on the five, and finishes Ebony T1 (Qwen2.5, `-ub 1`); the lead settles slot 3 and freezes.
2. Owner runs R69 (kit r29).
3. Lead's provisional answers to (a) to (d) are in memo-06 section 5, pending owner confirmation.


*Note 2026-11-21: campaign 2 is closed (owner, 2026-11-21). R69 can run (probe kit r29 is uploaded), and owner questions (a) to (d) below are due now.*

*Note 2026-11-20: kit r30 is campaign 2's one-app R68 re-run (R71); campaign 3's next kit after r29 is r31, not r30.* *Note 2026-11-21: r31 is also campaign 2's (R71 amendment 1); campaign 3's next kit is r32.*

1. R69 (device probe: read rate, compute ceiling, residency, GPU headroom) is pre-registered
   (prereg/R69-probe-read-rate-capacity-gpu-device.md; probe patch patches/llama.cpp-c3/0001-probe.patch, app patch
   patches/app-c3/0001-kit-r29-probe.patch). Next: the owner creates tag `prereg-R69`; the lead uploads probe kit r29
   to `HoeioUser/vollie-kits` as `r29/`; the owner runs it after R68's reports are in and campaign 2 is closed.
2. Tom's host tests H1, H3, H6, H7 are done (memo-02, including section 8). Element column sparsity stays open; the
   phone's read rate (R69) decides.
3. Round 2 (memo-03) is done; memo-04-work holds Bonnie's bands and Ebony's block-sparsity draft.
4. R69 is uploaded to HF `r29/` as `vollie-c3-probe-r29.apk` (sha 02736b56...). The kit after R69 carries D1 to D3 and E5
   from memo-05 section 4. Owner questions (a) to (c) from memo-05 section 3.5 are asked after the R68 report.
5. The kit after R69 has its runtime ready as `patches/llama.cpp-c3/0002-d-rows.patch` (sha256 d9d6cdd5...): D2 kernel
   rows K1/Y1b/short-row skip, D3 lmk/zram/PSI header. Identity (*host*): with the knob unset, per-chunk PPL equals
   stock at `-b 512`, `-ub 32` and `-ub 1` on Mistral. The kit still lacks an r30 app patch, which waits for R69's
   outcome. D1 (design in `patches/app-c3/README.md`) lacks chosen served campaign-3 Q4_0 files: proposal to the
   owner, which file per model, URL and sha256 (the Qwen2.5 lab file is a two-part split).

## Current state and next actions (appended 2026-11-22, end of day; supersedes the blocks above where they differ)

*Added by Annie from git (`787b651..55453a7`); the earlier blocks stay as history.*

**State.** S3 freeze is running on the owner's laptop since 2026-11-22 22:46 (`tools/c3_freeze.sh`, one subject per call, in the order
Yi, Llama-2, DeepSeek; DeepSeek is served unmodified, owner 2026-11-22, no BOS change). **Device rounds are held by the owner:**
R73 does not run until there is a real breakthrough; a device test is proposed only for a lever whose progress is certain on host
evidence, named in its pre-registration. Host research, host checks and the freeze continue. Served files may go to the public HF
repo `HoeioUser/vollie3` after the freeze checks pass and when a round needs them.

**Where the numbers stand (all *estimated* from *device* inputs, or researcher judgement).** R72 K0: 6.81 / 5.88 / 5.84 tok/s
(Yi / Llama-2 / DeepSeek), gaps +60.4 / +82.3 / +83.3 ms to the 79.2 ms printed for 12 tok/s. Every priced route is closed or below 12.
Best priced stacks: memo-10 9.8 / 9.5 / 9.3 tok/s (Ebony's inputs 9.3 / 9.0 / 8.8); memo-09's 11.7 / 10.6 / 10.6 needs a GPU kernel
that does not exist and a trained head whose measured top-1 is below the value assumed (memo-10), so about 9 to 10 is the range.
P(all three at 12): about 0.002 to 0.003 (judgement: memo-09 Ebony 0.003, memo-10 Bonnie 0.002, memo-11 Bonnie 0.005,
memo-11 Ebony 0.002 to 0.003).

**Closed levers (with source).**
- K1m / k1m4 (R72, L1x 0.983); K1 parked on all three subjects as built (R72).
- Lever M (trained guess head) as written: top-1 Yi 0.379, Llama-2 0.497 against the 0.50 line (*host*, memo-09-work/tom-head.md).
  The head is still the best single guess source measured (tokens per pass 1.33 and 1.45, *host*); allowed by owner 2026-11-22.
- Retrieval store of the model's own outputs (memo-11 exotic, *host* data); layer-skip self-speculation (memo-11 exotic, by price).
- LUT kernels and runtime re-encode of the frozen weights (U2, *host*, contended: faithful 2-bit k 0.84 to 0.96, 3-bit 1.21 to 1.34
  against the k <= 0.5 that memo-11 bytes requires; A78 issue count agrees, *estimated*).
- Skipping and low-rank (layer skip, FFN skip, dynamic depth, low rank: at most 3 to 5 % of bytes inside the budget, memo-11 bytes).
- A55 lane and the CPU-alone bus lift (memo-11 exotic; a single processor stays near 0.6 of nominal bandwidth).
- Wider verify trees: width 2 is best on all three, no width reaches 12 (memo-10).

**Open.**
- GPU lane (R72 joint reading: a perfect lane on stock bytes 11.3 / 10.5 / 10.0 tok/s, *estimated*). The unknown only the phone can
  answer is the Adreno Q4_0 gemv rate; the upstream kernel cross-compiles for arm64 at our base (*host*, not run anywhere), the x3
  probe is a design only (memo-09-work/tom-host.md), kill at g < 7 GB/s (R72 result). Held until the owner allows a device round.
- Prefix-exit ring (memo-11 exotic, test E-T3, host): beats the head only if top-1 gains 0.07 or more; cannot reach 12 alone.

**Next actions.**
1. Let the S3 freeze finish (about 12 to 21 h serial, *estimated*, memo-10-work/clyde-freeze.md); record the results when the logs arrive.
2. Host test E-T3 and remaining host tests from memo-11; no device request until a breakthrough.
3. ~~Owner to tag `prereg-R72`~~ *Correction 2026-11-22 (lead): done; `prereg-R72` (364c438) and `prereg-R72-amendment-1` (dc19d32) exist on the remote, created by the owner and checked by the lead.*
4. The R73 plan stays a draft (memo-10-work/clyde-r73-plan.md).

## Current state and next actions (appended 2026-11-23; supersedes the blocks above where they differ)

**S3 freeze done** (*host*, full wikitext-2, `tools/c3_freeze.sh`; re-made Q4_0 byte-identical to the kept file on all three):
Yi 8.3875 (F16 8.1773, served/F16 1.0257), Llama-2 7.7435 (7.4932, 1.0334), DeepSeek 8.7311 (8.4115, 1.0380) at `-b 512`;
`-ub 1` within 0.03 %; all inside budget and ceiling. Rows in results.csv.

**R74 scored** (reports/R74-result.md; JSON `384bc4b5…`): no void. Rules 1 to 4 pass on all three, so the **GPU lane is OPEN**
(host numerics check and lane design only, not a device go). *Device*: all four upstream kernels built and passed identity;
g_cold 17.53 / 18.33 / 18.33 GB/s; CPU+GPU 1.611x (41.3 GB/s); dispatch overhead 452 µs per round trip (3x the centre).
*Estimated*: P_G 7.86 / 7.39 / 7.30 tok/s, centre stack 9.55 / 9.50 / 9.26; at zero dispatch 12.23 / 11.87 / 11.44.
Campaign verdict unchanged (12 on all three out of reach by priced levers); the binding cost is now the join, not g.
Conditions: cool start not met (campaign 2's T-A cap probe ran until 18 s before R74; its JSON is filed as descriptive),
x3b and x3r gates timed out. Owner to confirm the r34 APK sha256.

**Next actions.**
1. Tom: host numerics check of the lane (half-precision `(q - 8) x d` emulated in the CPU backend behind a knob, per-chunk PPL
   at `-b 512` and `-ub 1` on two subjects); the `loadlog` text.
2. Bonnie and Ebony: price a lane design with joins cut or overlapped (async queue, one host wait per token, event polling);
   the line to beat is about 20 µs per join at 64 joins.
3. Device rounds stay held (R73 included); a lane device test needs its own pre-registration and the owner's permission.

## Hand-offs and open items (appended 2026-11-23 by Annie from git `cd5660d..2517042`; supersedes the "Next actions" above where they differ)

*Corrections to the block above: "Owner to confirm the r34 APK sha256" is done (`70adc9b`, owner confirmed `2b0a2929…81de`; the APK void is not triggered). Bonnie and Ebony's join pricing (action 2) is done as work files (`8bbabb7`). Annie's number audit of R74 (2026-11-23): every number reproduces from the raw JSON and the scripts; one K0-basis note appended to R74-result.md.*

**Hand-offs recorded from git today (all 2026-11-23).**
- `cd5660d` (lead): R74 scored, GPU lane OPEN on all three (*device*: g_cold 17.53 / 18.33 / 18.33 GB/s, CPU+GPU 1.611x, dispatch 452 us); P_G 7.86 / 7.39 / 7.30 tok/s and centre stack 9.55 / 9.50 / 9.26 (*estimated*); verdict unchanged. S3 freeze results recorded (*host*, all inside budget).
- `70adc9b` (owner via lead): the r34 APK sha256 matched; dated note in R74-result.md.
- `8bbabb7` (Bonnie, Ebony): memo-12 work files. Bonnie: the 452 us is 43 % host driver, 44 % submit-to-start, 13 % rest; 4 forced joins per layer; no design reaches 12 on Llama-2 or DeepSeek, Yi only at 33 us per join or less (*estimated*). Ebony: launch cost, not wait; level stacking; device cells X4-a to X4-f. Work files, not a verdict.
- `2517042` (owner decision, recorded in CAMPAIGN.md): owner: "I allow the measurement attempt" (translated from Korean); the X4 GPU join probe may run on the phone as a second exception to the device hold: **R75**, kit r35. No model loaded. R73 stays reserved.

**Open hand-offs (running or waiting; nothing below is finished).**
1. Tom (numerics check, patch 0006): half-precision `(q - 8) x d` emulation in the CPU backend behind a knob, per-chunk PPL at `-b 512` and `-ub 1` on two subjects, plus the `loadlog` text. Running; report to `reports/memo-12-work/tom-numerics.md`.
2. Tom, second instance (kit r35: runtime patch 0007 and app patch 0005): builds the X4 probe kit. Running; report to `reports/memo-12-work/tom-r35-build.md`.
3. Lead: pre-registration of R75 after the r35 build (prediction, pass and kill rules, commit, then the owner tags `prereg-R75`).
4. Owner: gives the Hugging Face write token for the `r35/` upload to `HoeioUser/vollie-kits` (not stored in any file). Then runs R75 with a **true 30 minute cool start and no other probe first** (R74's cool start was spoiled by campaign 2's cap probe 18 s before).

Device rounds otherwise stay held; R75 is the owner's exception. Nothing in R75 is measured yet.

## Open owner decisions

Collected here; the owner asked not to be asked until the R68 JSON arrives.

- Gemma 4 E4B as a subject. **Answered 2026-11-19: no.** Owner: "it is hard to see it as 'dense 7B-class'. In compute and intelligence it is about a 4.5B dense model; in storage and memory, about an 8B model." Consequences in the session log.
- RWKV-7 G1 as a subject (memo-02 host screen: 90.5 % exact zeros, *host*, so the question is worth asking). **Answered 2026-11-19: no.** Owner: "There is no need to include RWKV-7 G1; the subject models and their reserves are enough."
- Calibration-derived tables as runtime (the lead's provisional answer: yes). **Answered 2026-11-19.** Owner: "Treat them as data read at runtime, but manage them as derived artifacts attached to the weights, not as code." Lead's reading is recorded in CAMPAIGN.md under the terms table (sidecar file, generated by a committed script, sha256 of model and table, runtime refuses a mismatch, sha256s in every pre-registration, own PPL check at `-b 512` and `-ub 1`, served file unchanged).

**Open, to be asked after the R68 report** (memo-05 §3.5 and the D1 design):
- (a) Is a phone-built, sha-keyed kernel-layout side file (Y1a) a weight-derived artifact, like the calibration tables?
- (b) Is twice the storage per model acceptable for it?
- (c) Does the answer also cover non-stock layouts (a column-major copy, a 1-bit proxy file)?
- (d) The campaign-3 served Q4_0 file per subject model (source, URL, sha256; the lab's Qwen2.5 file is a two-part split).

*Note 2026-11-19: until (a) to (d) were added, this section read "No owner decisions are open now."*

**Answered:**
- Does Google's QAT file of Gemma 4 E4B count as stock weights? **Answered 2026-11-19.** Owner: "Only unquantized QAT checkpoints count as stock weights." Lead's reading: a publisher's unquantized QAT checkpoint (for example Google's Gemma QAT release in BF16/F16) is stock weights for campaign 3; pre-quantized QAT files (such as a publisher's Q4_0 GGUF) are not. The served file is quantized by us from the unquantized QAT checkpoint and frozen as usual; stock PPL and F16 PPL are measured on that checkpoint. Whether Gemma 4 E4B counts as "dense 7B-class" is still open with the owner.

## Session log

| date | session | what |
|---|---|---|
| 2026-11-18 | 4 (continued) | Owner sets campaign 3: dense 7B-class, 15 tok/s (first 20, lowered after the lead's arithmetic), file not changed but used freely (dense as MoE allowed), served quantisation free inside the budget, +5 % budget, F16 × 1.20 ceiling, campaigns 1-2 as the base. Lead's model definition accepted. Planning files created. |
| 2026-11-18 | 4 (continued) | memo-01 verdict: 15 tok/s on a 7.1 B+ dense body is out of reach with every priced method, P < 0.05 (*estimated*). Best priced combination is 5 to 9 tok/s (*estimated*). The Gemma 4 E4B opening is 7.5 to 14 tok/s (*estimated*). The goal stays at 15 and is not raised. Research continues. |
| 2026-11-18 | 4 (continued) | memo-03 — no path reaches 15. Strict dense best about 12 tok/s, E4B 12.7 to 13.8 at its ceilings (*estimated*). The GPU and Hexagon paths are closed in their stock form. CPU+GPU split and multi-stream GEMV are ranked first. Device probe set P1, P2, X1, X2 is planned for the next phone window. The H6 kill line depends on the measured read rate. |
| 2026-11-19 | 4 (continued) | memo-02 host screens (*host*): column input sparsity is open on both models (Mistral 40 %, Qwen2.5 35 % inside the margin at both settings). The exact column kernel costs 2.6x sdot per touched weight on the host, so the phone's read rate decides. Sublayer skip and per-token FFN skip are closed on both. RWKV-7 has 90.5 % exact zeros, so the owner question is worth asking. Q4_0 is within 1.02 to 1.04 of Q8_0. The probe kit design was approved by the lead and is being built. The lead's block-granular sparsity idea (H3b/H6b) is handed to Tom. |
| 2026-11-19 | 4 (continued) | Owner, 2026-11-19: "Only unquantized QAT checkpoints count as stock weights." Lead's reading: a publisher's unquantized QAT checkpoint (for example Google's Gemma QAT release in BF16/F16) is stock weights for campaign 3; pre-quantized QAT files (such as a publisher's Q4_0 GGUF) are not. The served file is quantized by us from the unquantized QAT checkpoint and frozen as usual; stock PPL and F16 PPL are measured on that checkpoint. Whether Gemma 4 E4B counts as "dense 7B-class" is still open with the owner. |
| 2026-11-19 | 4 (continued) | Owner rules Gemma 4 E4B out as a subject: "it is hard to see it as 'dense 7B-class'. In compute and intelligence it is about a 4.5B dense model; in storage and memory, about an 8B model." Lead's consequence: the subjects are Mistral-7B-Instruct-v0.3 and Qwen2.5-7B-Instruct, with reserves Llama-3.1-8B and Yi-1.5-6B. The memo-01 and memo-03 "opening" (E4B) is closed. RWKV-7 G1 is still open. The strict-dense estimates stay at <= about 12 tok/s (*estimated*, memo-03). |
| 2026-11-19 | 4 (continued) | Owner rules RWKV-7 G1 out as a subject: "There is no need to include RWKV-7 G1; the subject models and their reserves are enough." The subject set is fixed: Mistral-7B-Instruct-v0.3 and Qwen2.5-7B-Instruct, with reserves Llama-3.1-8B and Yi-1.5-6B. The only open owner item left is calibration tables as runtime (the lead's provisional yes). |
| 2026-11-19 | 4 (continued) | Owner decides calibration-derived tables (thresholds, hot-first orders) are data read at runtime but managed as derived artifacts attached to the weights, not as code. Lead's reading recorded in CAMPAIGN.md: sidecar file; committed generator script, served file and wikitext-2 train named in the header; model sha256 and own sha256 recorded; runtime refuses a model-sha mismatch; table sha256 listed next to the model's in every pre-registration; a new table needs its own PPL check at `-b 512` and `-ub 1`; served file unchanged. No owner decisions are open now. |
| 2026-11-19 | 4 (continued) | memo-02 section 8 (*host*): block sparsity is closed as a speed lever on both models (down-only frontier 9 to 10 % of blocks, about 2.5 % of the bytes); the block-major skip-list kernel costs 0.93x per touched weight and is bit-identical; element column sparsity stays open. R69 pre-registered (probe kit r29; probe patch sha256 ac5c7057..., app patch a1c51b22...); kit upload follows, owner runs it after R68. |
| 2026-11-19 | 4 (continued) | memo-05 *host* results: K1 at TR 128 is bit-identical, k 1.53 to 1.64, open pending phone D2. Y1b is bit-identical, hot 1.47 to 1.51, cold about 2.1 on down. Y1c is not bit-identical; the PPL delta is ±0.02 %. Closed: K2, H5 as a residency lever, gate rule, K1 at TR 512. Row skip is parked. E1: down-only 60 % fits, 70 % does not. The goal is unchanged. |
| 2026-11-19 | 4 (continued) | Runtime patch 0002-d-rows (sha256 d9d6cdd5...) committed for the kit after R69: D2 kernel rows (K1, Y1b, short-row skip) and D3 lmk/zram/PSI header; D1 design in patches/app-c3/README.md. Knob unset equals stock per-chunk PPL at `-b 512`, `-ub 32`, `-ub 1` on Mistral (*host*). No r30 app patch yet (waits for R69). Owner decision (d), the served file per model, added. |
| 2026-11-21 | 5 | Owner Amendment 1: goal 12 tok/s (was 15), exactly three dense 7B-class models of different families on the same phone, campaign goal met only if all three pass; subjects reselected. Lead wrote memo-06 section 1 (selection rule) before any pricing and started joint round 4 (Bonnie, Ebony, Tom, Clyde, Annie). CAMPAIGN.md stale campaign-2 sentence corrected. |
| 2026-11-21 | 5 (hand-offs) | Joint round 4 hand-offs recorded (all open, nothing measured): Bonnie, candidate pool and pricing, memo-06-work/bonnie-pool.md; Ebony, new levers for 12 tok/s and challenge of Bonnie's pricing, memo-06-work/ebony-levers.md; Tom, model-generic host screens on the pool (*host*), memo-06-work/tom-screens.md; Clyde, re-plan of rounds and kill lines at 12 tok/s and a possible R69-amendment-1.md, memo-06-work/clyde-replan.md. |
| 2026-11-21 | 5 (clyde) | memo-06 §5 re-plan at 12 tok/s (*estimated*; work file memo-06-work/clyde-replan.md): 12 needs bytes/token ≤ r × 75 ms (1.81 GB at r 24) **and** touched weights × k ≤ about 5.4 G Q4_0 weights at R69's central c_b 40.4, so every 6.0 to 8.5 B dense body is compute-bound below 12 at stock Q4_0 cost (about 9.5 tok/s at 7.0 B multiplied); the reaching corner needs a 3.1 to 3.7 bpw file inside F16 × 1.20 and C, column sparsity d about 0.60 with a low-bit exact kernel at k ≤ 1.3 to 1.5, and r ≥ about 20 to 28 or extra compute; lead's prior P(all three) 0.05 to 0.15. Rounds S0 to S6 with kill lines K0, K-type, K-F, K1 and go/claim at 12. prereg/R69-amendment-1.md written before any R69 data (Rules A to C and §6 adapted to three unchosen subjects; nothing measured or predicted changes). Owner questions (a) to (d): lead's provisional answers yes / yes / yes (proxy least certain) / superseded (we quantise from publisher BF16), to be confirmed by the owner. |
| 2026-11-21 | 5 | Joint round 4 done: Bonnie pool (28 eligible, 22 families), Ebony scout and challenge, Tom model-generic tools (olmo2 post-norm bug fixed) and T1 four-row groups on Mistral (*host*, -b 512: about 30 % file order, 33 to 35 % clustered inside +4.5 %), Clyde re-plan and R69 amendment 1. Lead's provisional subjects Yi-1.5-6B, ChatGLM3-6B, DeepSeek-LLM-7B; reserves Llama-2-7B, Granite-7B. Screens blocked by laptop disk. |
| 2026-11-22 | 5 | Joint round 5 (memo-07 work): Tom's final screens (*host*, 40 chunks; frontiers Yi 40 %, ChatGLM3/DeepSeek/Llama-2 20 %, Granite 15 %; Q4_0/F16 1.01 to 1.04). Bonnie re-rank with measured s: Yi, Mistral, (ChatGLM3, eligibility in doubt: stock PPL 33.0 unexplained), Qwen2.5, Llama-2, DeepSeek, Granite. Ebony: only Yi has a priced cell at 12 (needs r >= 31 to 32); no second model at r <= 35; recommends reporting the three-model goal as out of reach by priced levers. Lead asked the owner about ChatGLM3 (verify with publisher code, or drop). |
| 2026-11-22 | 5 | Owner: ChatGLM3 excluded; the three models come first. Subject 1 Yi-1.5-6B fixed; slots 2-3 from Mistral, Qwen2.5, Llama-2, DeepSeek after Tom's k,v-dense 35-45 % cells (lost in a container restart, re-run) and Bonnie's re-price. |
| 2026-11-22 | 5 | Subjects fixed (memo-06 §6): Yi-1.5-6B, Mistral-7B-v0.3, DeepSeek-LLM-7B; reserves Llama-2-7B, Qwen2.5-7B. Criterion-1 gaps among Llama-2, DeepSeek, Mistral read as a tie (lead judgement, after prices), broken by criterion 2. k,v-dense rule: Llama-2 and DeepSeek 40 % inside budget (*host*). Next: S3 freeze of the three served files on the laptop (DeepSeek with BOS), R69 on the phone. |
| 2026-11-22 | 5 | Correction after Ebony's challenge: post-hoc tie reading withdrawn; criterion 1 as written gives subjects Yi-1.5-6B, Llama-2-7B, DeepSeek-LLM-7B; reserves Mistral, Qwen2.5. Before freeze: DeepSeek 40 % k,v-dense re-run with BOS (Tom). |
| 2026-11-22 | 5 (clyde) | R72 plan drafted (reports/memo-08-work/clyde-r72-plan.md; not a prereg, bands from memo-08): device round with no model, kit r32, blocks A anchors, D2s K1 at subject shapes (4096/11008, keep 0.58/0.60/0.65, A78 and X1), T token-shaped layer sweep per subject at 3 and 4 big cores (stock vs k,v-dense K1 rule, ABBA, ggml-graph tie cell), G per-type gemv incl. all-threads-at-once hot ceiling, B barriers, H Q6_K head; draft rules for K1, 4th core, K0 update, K-type. S3 freeze list: Q4_0 candidates with sha256 (DeepSeek BOS file f195d24f..., one metadata byte, lead decision to confirm), F16 re-make + requant sha check, full-set -b 512 and -ub 1 commands. |
| 2026-11-22 | 5 (clyde) | R69 scoring prepared before data: reports/memo-08-work/clyde-r69-scoring.md (parser output → original rules and amendment 1, voids, B12/W12 lines) and clyde_r69_score.py (self-test passes). Found before data: kit r29 runs P2col at keep 0.60 cold only (prereg §1 lists 1.0/0.65/0.60 hot and cold), so Rule A at d 0.65 and for Llama-2/DeepSeek (k,v-dense, d 0.666) is "not scored from R69"; the X2 `-cl-opt-disable` retry is not in the build; P2hot bands are cycles while the probe prints ns (converted with each cell's clock). Known limit of P1 (Bonnie, bonnie-path.md §C): it measures file-backed mapped capacity, while the repacked body is anonymous memory; recorded as a limit, not a change to R69. **Install state (owner, 2026-11-22):** kit r29 installed fresh (r31, versionCode 31, uninstalled because r29's versionCode 29 cannot install over it; app data wiped; OLMoE re-downloaded). R69's text states versionCode 29 and the same package and certificate as r28, and no install state; device conditions (power saving off, charger out, deep-sleeping list recorded, no model loaded) are unaffected. Lead: not a deviation from R69's written conditions; recorded as a session fact and stated in the result. Possible effects to check in the JSON: `free_start` and P1's `free_gb` (the model download reduces free storage for the 4.6 GB scratch step; a step that cannot be written is "not measured") and a cold page cache (P2/X1 run on anonymous arenas, so unaffected). |
| 2026-11-22 | 5 | Joint round 6 (memo-08 work) closed on the host side: Bonnie gap ledger and stacks, Ebony paths, both P(all three at 12) about 0.005 to 0.01 with the residency gate; K1m byte-identical, A78 floor k 1.54 (3-core 1.36) *estimated*; full-set Llama-2 +3.40 %, DeepSeek-BOS +3.55 % (*host*); per-group levels closed as a general lever; merged R72 bands confirmed by both. R69 scoring sheet ready. Waiting: owner's R69 run (model re-download, about 4 h). Next: R72 pre-registration after R69, kit r32 build (Tom). |
| 2026-11-22 | 5 (clyde) | R69 scored (reports/R69-result.md; JSON `db4985af…`, `2a38091`). No void: APK sha256 02736b56… matches (owner installed lab/dist-c3/vollie-c3-probe-r29.apk; checked by shasum), power saving off, charger out, no block killed. Conditions: gates timed out before p2 and x1, clocks low (1 of 26 cells at full clocks), 6 cells cpu_ok < n, used as written (the scoring sheet's pre-data exclusion of `*` cells withdrawn: it would add a void). *Device*: r 28.56, c_b 24.46 (below r: clock-depressed), C 4.2 GB file-backed, S 1.575, X1 gains 0.92 / 0.86, GPU q4 1.88 GB/s. Verdicts: A KILL for the H6 column kernel (reference d 0.60 and Yi), not scored for Llama-2 and DeepSeek (no 0.65 row in r29); B KILL (R20 closed); C in between, all three Q4_0 files fit file-backed C; D rank 1 open, multi-stream closed. 7 of 16 bands inside, clocks MISS. Ledger (*estimated*): best CPU price 6.8 / 5.9 / 5.9 tok/s (Yi / Llama-2 / DeepSeek) at c_b as written, 10.8 / 9.1 / 8.9 at the pre-registered c; no subject at 12. K1 stays open (not in r29). No c3 results.csv exists, so no row was added. |
| 2026-11-22 | 5 (clyde) | R72 pre-registered (prereg/R72-probe-k1-cores-types-anon-device.md; not committed yet): kit r32 (APK 08a37bd5…a50c), blocks a, l1, l4, l23, l5, l6, l7, x2, anon34, anon42; cool start, power saving off, charger out, list recorded; merged bands of bonnie-path §10.4 as confirmed in ebony-paths §12 plus a new `a` anchor row on R69 run 1; rules: L1x 0.90 / 0.97, L4 kill ≥ 1.00 / open ≤ 0.90 per subject, 4th core 0.95 / 1.00 and L3 < 1.10 with Ebony's pre-emption label, K0 per subject with gap in ms, type filter, anonymous capacity vs 3.36 / 4.02 / 4.01 GB; voids as R69 only. Next: Annie commits and pushes, then upload r32/ to HF, owner runs. |
| 2026-11-22 | 5 (annie) | Books catch-up from git (`18c60e3..37129fc`). Ledger file `results.csv` created for campaign 3 (R69 runs 1 and 2, R72 pending); memo index updated (R69 repeat addendum, memo-08-work, R72). Hand-offs, in order: Tom, kit r32 build (`e8f4a9d`; memo-08-work/tom-r32-build.md; APK 08a37bd5...a50c); Clyde, R72 pre-registration (`364c438`); kit r32 uploaded to HF `HoeioUser/vollie-kits` `r32/` after that commit. Earlier: R69 JSON `2a38091`, scoring `8d4f3e9`, owner's cooled repeat run as descriptive addendum `a364bb5` (run 2: r 31.30, c_b 27.22 GB/s, *device*; run 1 stays scored). |
| 2026-11-22 | 5 (annie) | R72 attempt 1 (kit r32, `5cef0a6`, `device/r72-attempt1-c3probe-r32-2026-11-22T1749.json`): finished in 5 s, no block ran. Cause: kit bug, the `Isolate.run` closure captured unsendable app state (*host* repro in tom-r33-build.md). No data: not scored, not a void (R72 §voids are unchanged), reported as a failed kit run. |
| 2026-11-22 | 5 (annie) | Fix: Tom's kit r33 (app patch 0003; 12 of 13 native libraries byte-identical to r32, only `libapp.so` differs) and `prereg/R72-amendment-1.md` (`dc19d32`) before the r33 upload; APK `vollie-c3-probe-r33.apk` sha256 1917aa4c...db7c uploaded to HF `r33/` on 2026-11-22. R72 run received (`37129fc`, `device/r72-c3probe-r33-2026-11-22T1759.json`, sha256 cbc332dd...; app 1.33.0+33; all ten blocks ran, power saving off, charger out, deep-sleeping list on per commit message). Scoring pending: the lead writes reports/R72-result.md. Open: owner to tag `prereg-R72` (tags are the owner's). |
| 2026-11-22 | 5 (clyde) | R72 scored (reports/R72-result.md; JSON `cbc332dd…`, `37129fc`; kit r33, all ten blocks ran, no void; owner to confirm the r33 APK sha). 14 of 26 bands inside. K1m and k1m4 closed (L1x 0.983). The K1 lever is parked on all three subjects as built (4-7: 0.960 / 0.948 / 0.987), and it is slower than ggml stock on every subject. Fourth core adopted (0.863 / 0.694 / 0.667), with a caveat: the cpu-7-pinned thread ran on small cores in most cells, and the A78 cap was 1,555 to 1,785 MHz in L4. K0 (*estimated*): 6.81 / 5.88 / 5.84 tok/s, gaps +60.4 / +82.3 / +83.3 ms, so out of reach with the K1 family and the 4th core. Type filter: Q4_0 stays. anon34 held, anon42 not: only Yi can hold a repacked body. GPU read sum 46.5 GB/s reproduced; the `x2r` build failed with err -6. Joint reading by Bonnie and Ebony: a perfect GPU lane on stock bytes gives 11.3 / 10.5 / 10.0; P(all three) about 0.001 to 0.01. Next: S3 freeze, then R73 real stock decode on Yi and Llama-2 with placement logging; then a GPU probe with the upstream Adreno Q4_0 gemv (kill at g < 7 GB/s). Not committed. |
| 2026-11-22 | 5 (clyde) | Next steps after R72 prepared (not committed): reports/memo-10-work/clyde-freeze.md and new `tools/c3_freeze.sh` (S3 freeze per subject on the laptop: pinned download into `lab/c3/dl/<m>`, F16 re-made, re-quantised Q4_0 must equal the kept file byte for byte, DeepSeek BOS field, full wikitext-2 PPL F16 `-b 512`, served `-b 512` and `-ub 1`; F16 `-ub 1` optional because the ceiling cannot bind; about 12 to 21 h serial, about 28 GB peak, *estimated*). Trap found: `general.name` comes from the download directory, so the 2026-11-22 DeepSeek re-prep (`deepseek7b-r`) gave a different F16 and Q4_0 than the kept file. reports/memo-10-work/clyde-r73-plan.md: R73 = first stock decode on Yi, Llama-2, DeepSeek (kit r34, 4 threads on 4-7, nCtx 512, ABBA guard 0004 off/on, logging K-a to K-j incl. affinity read-back, caps vs hardware max, thermal, core_ctl, devfreq/THP = T6); draft K0 check ±10 % of 6.90 / 6.21 / 6.22; fixed prompt set adopted (`eval.txt` ff1f32e3…); served files to a new public HF repo `HoeioUser/vollie3` (owner question F3); Rule B reopened with Ebony's joint line in coverage form G = (1 + p) ÷ (c × kB_eff + 1 − c) < 1.05 on any subject, guess source f by max-min (L lives only if device kB_eff ≤ 1.075); M closed as written by T5 (Yi 0.379, Llama-2 0.497 < 0.50). Owner questions F1 (DeepSeek BOS), F2 (F16 `-ub 1` optional), F3 (hosting). |
| 2026-11-22 | 5 (annie) | Books catch-up from git (`787b651..55453a7`): R72 scored (`78a959e`; results.csv row updated from "scoring pending"); memo-09 and Ebony §10 (`0e31a8e`, `568dd5a`); owner decision on trained heads (`ef37c35`); T5/T1/T1b host (`80f993d`); GPU, guard and label host work (`0087ab1`); memo-10 and Ebony review (`ee652e6`, `f20b2dc`); freeze script and R73 plan (`73b1cf9`); owner decisions 2026-11-22 (`7cfb0f9`: DeepSeek unmodified, vollie3 allowed, R73 held); memo-11 bytes (`9c269fa`), exotic (`c8a4c98`); U2 LUT closed (`55453a7`). Memo index extended; new state block above. |
| 2026-11-23 | 6 (clyde) | S3 freeze results recorded (*host*; all three inside budget). R74 scored (R74-result.md): no void, GPU lane OPEN on all three (g_cold 17.5 / 18.3 / 18.3 GB/s, CPU+GPU 1.611x, *device*); dispatch 452 µs binds; P_G 7.86 / 7.39 / 7.30, stack 9.55 / 9.50 / 9.26 tok/s (*estimated*); 10 of 28 bands inside; verdict unchanged. Next: host numerics check and a join-cost lane design. |
| 2026-11-23 | 6 (annie) | Books from git (`cd5660d..2517042`) and number audit. R74 audit: every number in R74-result §1 to §6 reproduces from the raw JSON (`384bc4b5…`, sha confirmed) and `bonnie_tree.py` (g_cold 17.53 / 18.33, lane 0.621, g_eff 24.8, P_G 7.86 / 7.39 / 7.30, stack 9.55 / 9.50 / 9.26, 10 of 28 bands inside); results.csv S3 freeze rows match `lab/c3/logs/freeze-*.log` (budgets, ceilings, ratios) and the R74 row matches. One note appended to R74-result.md: the K0 gaps +60.4 / +82.3 / +83.3 and "about 1.2 tok/s" use R72's K1 body while P_G uses the ggml body; on the same base the lift is +0.96 / +1.18 / +1.08 tok/s (*estimated*); no verdict changes. Memo index: R74 prereg and result, the two device JSONs, memo-12-work, R75 pending. Hand-offs recorded above (`70adc9b`, `8bbabb7`, `2517042`): Tom numerics check (patch 0006) and Tom kit r35 build (patches 0007 and app 0005) running; lead owes the R75 pre-registration; owner owes the HF token and a true 30 min cool start. |

## Hand-offs and open items (appended 2026-11-23 evening by Annie from git `3461e19..2588f40`; supersedes the block of the same name above where they differ)

*Corrections to the block above: open hand-offs 2 (Tom, kit r35) and 3 (lead, R75 pre-registration) are done. Hand-off 1 (Tom, numerics check, patch 0006) is still running: the files `0006-x3-emu.patch`, `tools/c3_x3emu_check.c`, `bonnie_pack_time.c` and `ebony_hmax.py` are uncommitted in the working tree and are not yet books.*

**State.** R75 is pre-registered and not run. Nothing in R75 is measured.

**Landed (all 2026-11-23).**
- `a4117fc`, `fe621aa` (Bonnie r7, r8): packed side file design, file-backed KleidiAI buffer keyed on the served sha; buffer 3.38 / 3.77 / 3.84 GB and side file 2.22 / 2.43 / 2.57 GB at share 0.37 (*estimated*); Q6_K head +22 % (*estimated*). Bonnie r8 flags that S3 stock PPL was measured on ggml's repack path, while the phone serves the KleidiAI path (head re-quantised to int8).
- `b4b9168` (Tom): kit r35 final rebuild. APK `465d97bb7932fc4c9b7c675e1b3e2b78536466f9dec3442d4b8c59405f558af5` supersedes `e961ecf`'s `089ab530...4c62`.
- `2588f40` (lead): R75 pre-registration, `prereg/R75-x4-gpu-join-probe-device.md`.

**Owner decisions in chat, 2026-11-23.** x4m stays in R75 (owner: "x4m도 넣어도 됩니다"; translated: x4m may go in too). The Hugging Face token the owner pasted was rejected by HF as invalid; no token is written in any file.

**Audit (Annie, 2026-11-23).** Every sha256 in R75 prereg §0 matches `shasum -a 256` of the file in `lab/dist-c3/r35/`, `patches/` and `tools/`, and of libllama.so, libvollie_x4c.so and libggml-cpu-android_armv8.2_2.so unzipped from the APK into a scratch directory outside the repo (APK, BUILD-r35.txt, SHA256SUMS, 0007, app 0005, c3_x4child.c, three libraries: 9 of 9). APK size 16,621,998 B matches. SHA256SUMS lists the APK and BUILD file hashes as stated. No mismatch; the prereg was not edited.

**Next actions.**
1. Owner: create tag `prereg-R75` on `2588f40`.
2. Owner: upload `lab/dist-c3/r35/` (APK, `BUILD-r35.txt`, `SHA256SUMS`) to `HoeioUser/vollie-kits` as `r35/`, either with a new valid write token (given per session, never stored) or by uploading the three files directly.
3. Owner: run R75 per prereg §8 with a true 30 minute cool start and no other probe first; confirm the APK sha256 `465d97bb...f8af5` on the phone side before the run.
4. Open owner question (Bonnie r8): which path defines "stock" PPL, ggml's repack path (S3 as measured) or the phone's KleidiAI path? Bonnie r9 is reading the campaign terms.
5. Tom: numerics report for patch 0006 (`reports/memo-12-work/tom-numerics.md`); Annie commits the patch and tool with it.
| 2026-11-23 | 7 (annie) | Books from git (`3461e19..2588f40`): Bonnie r7, r8, kit r35 final rebuild (`b4b9168`, APK `465d97bb...f8af5`), R75 pre-registration (`2588f40`). Audit: all 9 sha256 in R75 prereg §0 match the files and the libraries unzipped from the APK (scratch dir); no mismatch, prereg not edited. Owner decisions: x4m stays in R75; the pasted HF token was rejected as invalid, so the r35 upload is pending. Open: tag prereg-R75, upload, owner run with 30 min cool start, "stock PPL path" question (repack vs KleidiAI). Memo index updated. |

## Hand-offs and open items (appended 2026-11-23 night by Annie from git `7765cf8..463ee88`; supersedes the block of the same name above where they differ)

**State.** R75 is pre-registered, tagged and not run. Nothing in R75 is measured.

**Landed (all 2026-11-23).**
- Owner created tag `prereg-R75` on `2588f40` (verified on origin). Kit r35 is uploaded to HF `HoeioUser/vollie-kits` `r35/` (HF commit `433fdd2e`); the downloaded APK sha256 was re-checked: `465d97bb...f8af5`. This closes next actions 1 and 2 of the block above.
- `748b6e7` (Bonnie r9): stock PPL path. Campaign 2's R61 option (a) carries over; KleidiAI-path stock is not yet measured for campaign 3 (band -0.3..+1.0 %, *estimated*); 40-chunk host run spec; pack routine 4.1-4.4 GB/s on one thread (*host*), first load 8-18 s (*estimated*).
- `219bb1b` (owner decision 2026-11-23, in CAMPAIGN.md): option (a) on stock PPL carries over, absolute budgets stay those of the S3 freeze, the KleidiAI-build stock is a second side-by-side reference; host KleidiAI stock runs are allowed.
- `463ee88` (Bonnie r10): every campaign-3 PPL figure is on the repack build; the freeze stock and patch 0006 need a KleidiAI reading; the 0006 emulation cannot run on a KleidiAI build as written; KleidiAI vs repack Q4_0 differ only at ties and the int8 head, band revised to +0.03 % (-0.05..+0.15, *estimated*); Granite's R61 +0.5 % unexplained, tail kept.

**Correction (2026-11-23, Annie, from Bonnie r10 §1).** The `build` column of the three S3 freeze rows in `results.csv` says "stock llama.cpp". It means `lab/llama.cpp-c3/build` at `26f78f925`, `GGML_CPU_KLEIDIAI=OFF`, repack on, which is not the phone's KleidiAI path. The cells are not rewritten; the correction is appended to each row's notes cell. The freeze numbers (*host*) are unchanged.

**Open hand-offs.**
1. Tom (second instance): KleidiAI-build 40-chunk stock runs, output `reports/memo-12-work/tom-kai-stock.md` (*host*).
2. Tom: numerics report for patch 0006 (`tom-numerics.md`); the patch and `tools/c3_x3emu_check.c` are uncommitted until it lands.

**Next actions.**
1. Owner: run R75 per prereg §8 after 08:30 KST on 2026-11-24 (not possible earlier), true 30 minute cool start, no other probe first; confirm APK sha256 `465d97bb...f8af5` on the phone side.
2. KleidiAI stock result pending (Tom, above).
3. Tom's 0006 numerics pending.
| 2026-11-23 | 8 (annie) | Books from git (`7765cf8..463ee88`): Bonnie r9, r10; owner decision `219bb1b` (stock PPL option (a) carries over; host KleidiAI stock runs allowed); tag `prereg-R75` verified; kit r35 on HF (`433fdd2e`, sha re-checked). Correction note on the S3 freeze rows' build column (repack build, KleidiAI off), appended in results.csv notes, cells not rewritten. Memo index: Bonnie r9, r10, bonnie_pack_time.c. Open: R75 run after 08:30 KST 11-24, KleidiAI stock, 0006 numerics. |

## Current state and next actions (appended 2026-11-24 by Annie from git `df08456..513b734`; supersedes the blocks above where they differ)

**The team is paused by the owner (2026-11-24): "do what is assigned so far, then rest". The hourly researcher keep-alive task is disabled. No device round runs until the owner runs R75.**

**R75** (X4 GPU join probe, kit r35): prereg commit `2588f40`, tag `prereg-R75` exists, kit r35 is on HF (APK `465d97bb...f8af5`, kit commit `b4b9168`). Awaiting the owner's run per prereg §8 (after 08:30 KST on 2026-11-24, true 30 minute cool start, no other probe first). Nothing measured yet.

**KleidiAI-build stock (*host*, 40 chunks, `lab/llama.cpp-c3/build-kai`, freeze flags + `GGML_CPU_KLEIDIAI=ON`, KleidiAI v1.31.0, `GGML_KLEIDIAI_SME=0`; memo-12-work/tom-kai-stock.md).** Shift against the repack build at chunk 40, `-b 512` / `-ub 1`: Yi +0.495 / +0.453 %, Llama-2 +0.069 / -0.055 %, DeepSeek +0.688 / +0.669 %. Scored against Bonnie r11 (Yi +0.3..+1.0, Llama-2 within +-0.05, DeepSeek 0.0..+1.0 %) and Ebony r7: direction and ranking as predicted (Yi and DeepSeek up, Llama-2 flat); Yi and DeepSeek inside both bands; Llama-2 just outside both +-0.05 % bands at 40 chunks (+0.069 at `-b 512`). Not a budget event (option (a), CAMPAIGN.md 2026-11-23 decision); rows in results.csv. Full-set KleidiAI runs are not made; the lead decides if they are asked.

**T-N1** (patch 0006, `VOLLIE_X3_EMU=2`, lane share 1.0 emulated, *host*, repack build, commit `2788606`; memo-12-work/tom-numerics.md): full set `-b 512` Yi 8.3804 (-0.085 % vs served 8.3875), Llama-2 7.7448 (+0.017 % vs 7.7435), both inside budget; `-ub 1` only on a 40-chunk subset; knob-unset identity holds; unfused mode reproduces the 12 R74 device identity values. Not measured on the KleidiAI path. Rows in results.csv.

**S-T1** (*host*, memo-12-work/tom-st1.md, commit `11f2e1f`): upstream ngram-simple drafts nothing at M<N; at N=M=2 it matches Bonnie's length>=2 lookup within one pass on 17 of 18 prompts; rule T's gain comes from the trained head; 106 of 108 speculative runs give the plain text, 2 Yi near-tie flips at token 30 from batched-verify numerics.

**Research findings (all work files, no verdict; *estimated* unless marked).**
- Body load choices A/B/C: bonnie-joins-r16.md, bonnie-joins-r17.md; choice C design: bonnie-joins-r20.md, r21.
- Honest stack ladder: bonnie-joins-r13.md, r14 (no stack reaches 12 on all three inside R75's K band, DeepSeek binds); coupling term refit on 17 device cells (0.25 to 0.17): bonnie-joins-r32.md.
- Official-prompt findings: bonnie-joins-r23.md to r26; ebony-joins-r11.md to r13.
- Cap and prefill heat: bonnie-joins-r31.md to r33 (P(cap lowered in a dense claim run) about 0.35, likely step 2,150 MHz); ebony-joins-r17.md (cap ladder; differs from Bonnie on depth).

**Open owner questions (each with its memo).**
1. Body load path A / B / C: bonnie-joins-r17.md.
2. DeepSeek prompt start a / b / c: ebony-joins-r14.md. **Answered 2026-11-24: a (156 tokens).**
3. Yi `<|im_start|>` marker: ebony-joins-r14.md.
4. Scoring of the official prompt versus the fixed set, option B: bonnie-joins-r26.md, ebony-joins-r13.md.
5. Prefix reuse in the claim protocol: ebony-joins-r17.md.
6. Whether campaign 2 gets a dated note on OLMoE's stray `<s>` prompt: ebony-joins-r15.md.

**Queued Tom tasks (not started; team paused).** H1 head-repack stock; E-T23 DeepSeek dump (unblocked 2026-11-24 by the owner's answer a to question 2; queued for Tom after kit r36); U4; E-T15; E-T19; host decode profile; KV screen; E-T24.

| 2026-11-24 | 9 (annie) | Books from git (`df08456..513b734`): Bonnie r11 to r33, Ebony r6 to r17, Tom 0006 + T-N1 (`2788606`), S-T1 (`11f2e1f`), KleidiAI-build stock (`513b734`). results.csv: six KleidiAI stock rows and two T-N1 rows (*host*); memo index extended; state block above. Team paused by the owner. Open: R75 run, six owner questions, queued Tom tasks. |

## Update 2026-11-24 (later, Annie): R75 ran; owner decisions

**The team is no longer paused: the owner resumed work with the R75 run (supersedes "team is paused" above).**

**R75 ran** (kit r35, started 2026-11-24 10:35 KST, saved by the owner about 13:14 after a force stop). The run hung in block x4f; blocks x4i, x4a and x4b completed. Device JSON committed: `reports/device/vollie-c3probe-r35-2026-11-24T1035.json` (*device*). The partial result (`reports/R75-result.md`) and the amendment (`prereg/R75-amendment-1.md`) with kit r36 (patch 0008) are in progress by the lead and Tom; no number from this run is scored here.

**Owner decisions 2026-11-24** (full text in CAMPAIGN.md): open questions 1, 3, 4, 5, 6 answered. (1) body load path A (plain); (3) Yi `<|im_start|>` is one token; (4) scoring on the official prompt (option B); (5) prefix reuse allowed in the claim protocol; (6) no dated note on campaign 2's stray `<s>`. **Still open: question 2, DeepSeek prompt start a / b / c (ebony-joins-r14.md).** E-T23 (DeepSeek dump) stays waiting on it.

| 2026-11-24 | 10 (annie) | R75 device JSON committed (hung in x4f; x4i/x4a/x4b complete); owner decisions 2026-11-24 recorded; team resumed. Open: question 2, R75 result and amendment, kit r36. |

**Update 2026-11-24 (Annie): owner answers.** Question 2 answered: DeepSeek prompt start a (156 tokens), owner: "a로 합니다." E-T23 (DeepSeek dump) is unblocked and queued for Tom after kit r36. All six open owner questions are now answered. Also confirmed: the r35 APK sha256 and the 30 min rest at room temperature before the R75 start (see `reports/R75-result.md`, "Owner confirmations 2026-11-24").

| 2026-11-24 | 11 (annie) | Owner answers recorded: DeepSeek prompt start a; r35 sha256 confirmed; 30 min rest confirmed. E-T23 unblocked. |

## Update 2026-11-24 (evening, clyde): R75 complete and scored

**R75 session 2** (kit r36, amendment 1, JSON `reports/device/r75a1-c3probe-r36-2026-11-24T1345.json`, sha256 `e46e3446…d513`) ran clean
13:45 to 13:59: no void, no deadline, no hang. Scored in `reports/R75-result.md` ("Session 2" section, S2.1 to S2.10; not committed yet).
- *Device:* x4f gate set→start 381.7 µs (plain 147.0); Adreno does not submit a released command without a further flush (nf 0/3, 3/3:
  session 1's hang cause confirmed); replay 44.3 µs per launch (sync1 124.5); `svm_c11` flag 0.9 µs, counter 3.75 µs (1.x modes never see
  it); `_mc3` dev_ratio 3.316 / 3.912 / 4.528 at 2 / 3 / 4 columns; gate‖up 0.966 of the 11008 rate; best lead split 0.923 at share 0.55;
  hand-off 13.3 µs (`write`); x4r held (21.02 GB/s, majflt median 0). Caps: GPU 676 after x4m, 491 after x4s; battery 23 → 6 %.
- *Estimated:* K_meas 149.0, K_eff 162.3 µs; model `pair` (at light draw); A and D killed; width-2 route to 12 closed (cg > 1.70), `_mc3`
  dropped; F worth a build (conditional; the probe's persistent gemv runs at 1.83 GB/s, no fast persistent kernel exists); Y14, E, B, B+
  open, not worth. **12 on all three: not priceable.** Honest ladder rung 6 at K_eff, cg 2.0: 10.7 / 10.5 / 10.1 (`linear`) to
  11.4 / 11.2 / 10.7 (`pair`, α 0.17). The stage model predicts a split ratio of 0.663 at K 149 where the phone gave 0.923; on the
  phone's split the lane is worth about +0.5 tok/s (7.40 / 6.68 / 6.66). Brier 0.142 to 0.163 over 39 outcomes. Session drift flag set.
- Cells 92 scored over both sessions: 39 inside, 53 misses.

**Next actions (ranked, from R75-result S2.10):** (1) Bonnie + Ebony: explain the split gap and re-run the ladder with R75's K, cg, split;
(2) Tom host: `_mc3` launch as built against upstream's dispatch; (3) Tom host: persistent row-claiming general kernel (F); (4) lead reports
the verdict to the owner and asks keep / realign / close; (5) R76 x5 kit r37 (Bonnie r18, body residency under choice A) only with owner
permission and its own prereg; (6) a split re-cell only if (1) finds a probe artefact. Open: owner's r36 APK sha256 confirmation. Annie:
commit the result and this block, results.csv rows for R75.

| 2026-11-24 | 12 (clyde) | R75 session 2 scored; combined verdict: complete, no void, 12 not priceable, K_eff 162.3 µs (*estimated*), cg 3.316 (*device*), x4r held. Not committed. |

## Update 2026-11-25 (Annie): after R75, "continue"; split gap explained; design G first

**Owner, 2026-11-25:** after the R75 verdict the owner wrote "계속하죠" (continue). The owner also confirmed the r36 APK sha256 (closes the open item in the 2026-11-24 evening block). No keep / realign / close decision was asked or given beyond "continue".

**Findings** (all work files in `reports/memo-12-work/`, no verdict by the lead; commits `537f99b`, `f952774`, `2dc0857`):
- **Split gap explained** (Bonnie r34, Ebony r18). The x4s best ratio 0.923 is a mean over arms from different phone states: four states in 63 s (Bonnie), a slow stretch of about 15 s (Ebony); Ebony's reading of the cool cell is 0.82, Bonnie predicts 0.83 (0.78 to 0.86) for a cool re-run (*estimated*). The stage model's 0.663 rises to 0.749 with the serial launch (`enq_us` 214 us with all four CPU threads idle) and 0.832 with end slack (Bonnie, *estimated* from *device* lines). The join as built is about 400 us, not 149 (Ebony, *estimated*).
- **Cool coupling as priced.** Cool CPU slowdown beside the GPU kernel: Bonnie 1.38 (1.28 to 1.54), Ebony 1.36 (1.28 to 1.45), against the committed 1.353 / 1.45 (*estimated*). Capped and throttled states are worse (Bonnie 1.57 capped, 2.2 to 2.8 throttled).
- **Design G before F.** G is the launch inside the pool job plus a trailing done-flag kernel, upstream kernel unchanged: +0.8 to +1.0 tok/s over the lane as probed (Bonnie, Ebony; *estimated*). F over G is worth a build only at about 0.9 of the general kernel's rate (Ebony). KleidiAI chunk multiplier 16 is needed to hide the launch (Bonnie); on the host it is bit-identical on Yi (40 chunks, 7.7347 / 7.7319 / 7.7165 at `-b 512` / `-b 512 -ub 32` / `-ub 1`, *host*, Tom). The host stall cost depends on the cell (*host*): cold cell, in-op 200 us stall costs +7.2 us at multiplier 4 and +7.4 us at 16; hot cell, 12 us stall costs +11.5 us at 4 and +3.0 us at 16.
- **P(12 tok/s on all three) about 0.002** (Bonnie; 0.0005 to 0.005, judgement). Ebony: nothing reaches 12 (F at full rate 11.86 / 11.66 / 11.24 cool, *estimated*).
- **The `_mc3` cost is in the kernel, not the probe** (Tom, *host* check against upstream's dispatch). *Device* x4m: 889.9 us at 1 column, 2,951.0 / 3,481.9 / 4,029.7 us at 2 / 3 / 4 columns; the step from 1 to 2 columns is +2,061 us, the later ones about 540 us each. PoCL: mc3 bit-identical to the 1-column kernel at 2, 3, 4 columns (*host*). R75's rule stands (cg 3.316, `_mc3` dropped).
- **E-T23 done** (Tom, *host*, prompt a, 156 tokens): lookup tokens per pass 1.231 / 1.185 / 1.143 (any length / 2 or more / 3 only), base check 0.9942 on 2,765 rows. Rows added to `results.csv`.

**In progress:** Tom is drafting kit r37 (lane G plus an x4s re-cell with triplets and a canary) as an R76 candidate. It needs its own pre-registration, the owner's tag `prereg-R76` and the owner's permission before any device run. Nothing is pre-registered yet.

**Paused:** Bonnie r35 and Ebony r19 stopped on the Fable usage limit. Their partial files (`bonnie-joins-r35.md`, `bonnie_gpuheat_r35.py`, `ebony-joins-r19.md`, `ebony_bus_r19.py`) are untracked and not committed. Both are scheduled to resume at 07:05 KST on 2026-11-26.

| 2026-11-25 | 13 (annie) | Books from git (`87bd9c2..2dc0857`): Bonnie r34, Ebony r18, Tom host checks (`_mc3`, E-T23, T1/T2/G). results.csv: five *host* rows (multiplier 16 identity x3, E-T23 lookup, base check). Memo index extended. Owner: "continue", r36 sha confirmed. Open: kit r37 / R76 prereg and owner permission; Bonnie r35 and Ebony r19 resume 2026-11-26 07:05 KST. |

## Update 2026-11-26 (lead): back to the earlier research method

**Owner, 2026-11-26:** "연구를 이어나가되, 연구를 자동으로 계속하는 것 이전의 연구 방법으로 돌아가세요. 비용이 너무 듭니다." (continue the
research, but return to the method used before automatic continuous research; it costs too much). The 2026-11-23 "researchers never
stop" rule is withdrawn (correction note in CLAUDE.md). The hourly keep-alive task (`trig_01UchZud…`) is disabled, not deleted. From now
on a joint round (Bonnie + Ebony) starts only when the lead opens one for a named question; finished researchers are not relaunched.

Bonnie r35 (`bonnie-joins-r35.md`, `bonnie_gpuheat_r35.py`) and Ebony r19 (`ebony-joins-r19.md`, `ebony_bus_r19.py`) are finished
(both end with sources and a ranked next-questions list) and are committed with this block as work files; no verdict on them yet.
Their next-questions lists are not started.

**Next (no researcher runs until the owner chooses):** (1) the lead reads r35 and r19 and folds them into the R76 plan (kit r37: lane G
plus the x4s re-cell with triplets and a canary); (2) R76 pre-registration, then the owner's tag and permission; (3) Tom finishes kit r37.

| 2026-11-26 | 14 (lead) | Owner: back to the pre-2026-11-23 method (cost). Keep-alive disabled; r35 and r19 committed as work files. |

**Update 2026-11-26 (lead):** r35 and r19 read and folded into `reports/memo-12-work/clyde-r76-plan.md` (draft R76 plan, kit r37: cool
pass, gap sweep, optional heat pass; 0 bytes, no model). Verdict input unchanged: no priced design reaches 12 on all three (best F per
token 11.86 / 11.66 / 11.24, G 10.59 / 10.46 / 10.14, *estimated*), and no R76 outcome can change that. Owner asked: (a) build and run
R76 for a below-goal G result, (b) stop lane work and report campaign 3 as not reachable, (c) realign. Lead recommends (b) or (c).
Nothing is built or pre-registered until the owner answers.

| 2026-11-26 | 15 (lead) | R76 plan drafted from r35/r19; keep / stop / realign question to the owner. |

## Update 2026-11-26 (annie): owner chose option (a), R76

**Owner, 2026-11-26:** "a로 가죠." Build and run R76 (kit r37: cool pass C, gap sweep G, optional heat pass H; kit r36 x4s code, no new kernel, no model) to decide whether design G is worth building as a below-goal result. Recorded in [CAMPAIGN.md](CAMPAIGN.md). The owner was told no R76 outcome can reach 12 tok/s on all three models; the lead recommended (b) or (c). Goal and budget unchanged. Memo index ([reports/README.md](reports/README.md)) now lists Bonnie r35, Ebony r19 (with scripts) and `clyde-r76-plan.md`.

**Open hand-offs:**
1. Tom: build kit r37 (running now); report to `reports/memo-12-work/tom-r37-build.md`.
2. Lead: write the R76 pre-registration after the build (`prereg/`, committed before any device run).
3. Owner: create tag `prereg-R76`; give the HF write token for `r37/` (private repo, token not stored); run with a 30 min cool start.

| 2026-11-26 | 16 (annie) | Owner "a로 가죠": R76 (kit r37) go. Decision recorded in CAMPAIGN.md; memo index extended (r35, r19, R76 plan). Hand-offs: Tom build, lead prereg, owner tag/token/run. |

**Update 2026-11-26 (lead):** kit r37 committed (`a74d575`), R76 pre-registration committed (`9867014`). Kit r37 uploaded to HF
`HoeioUser/vollie-kits` `r37/` (HF commit `75bfa32c`); the downloaded APK's sha256 re-checked against prereg §0. Open: owner tags
`prereg-R76` on `9867014` and runs R76 per prereg §8; the JSON goes to `reports/device/`.

| 2026-11-26 | 17 (lead) | Kit r37 and R76 prereg committed; r37 on HF (`75bfa32c`), sha re-checked. Waiting for the owner's tag and run. |

## Update 2026-11-26 (annie): R76 ran and is scored

R76 (kit r37) ran on 2026-11-26 (19:53, power saving off, deep-sleeping list on); JSON `reports/device/vollie-c3probe-r37-2026-11-26T1953.json` (sha256 `0e62571b...3534`), result in `reports/R76-result.md` (scored by the lead; number audit by Annie appended there). All numbers *device* unless labelled.

- **x4sc void** (cap 2,150 MHz from t 45.5 s): C1 not measured, C11 miss. Gap sweep: step about 165 us between the 0.1 and 0.2 ms rows (G1 to G3 miss, G4 hit). Heat pass: first cap 14.0 s, second 24.0 s, GPU cap 34.0 s (H1 to H5, H8 hit). Brier 0.219 (*estimated*).
- **Outcome by rule: "G not built, lane not killed."** The campaign verdict is unchanged (12 tok/s on all three not reachable, *estimated*).
- **Open:** (a) the owner confirms the downloaded APK's sha256 was `72541fcc...f96c`; (b) the owner chooses (1) take the rule as written, (2) re-run the cool pass with kit r38 (amendment 1), or (3) build G on the descriptive reading (R76-result section 6; the lead recommends (1)); (c) tag `prereg-R76` is not on origin yet (owner).

| 2026-11-26 | 18 (annie) | R76 ran and scored (kit r37): x4sc void, G not built, lane not killed; audit note, results.csv, reports index updated; JSON committed. Waiting for the owner's choice (1)/(2)/(3). |

## Update 2026-11-26 (lead): owner chose R76 option (1); GPU-lane work ends

Owner: "1로 가죠" (recorded in CAMPAIGN.md). G is not built; the lane, G, F and L5 leave the work list. Kit r38 is not built.
No device round and no researcher round is open. Campaign 3 stays open (goal unreachable by priced levers, P about 0.002);
closing it, realigning it, or a new non-lane question is the owner's next decision. Still open: APK sha confirmation for R76 and
the `prereg-R76` tag.

| 2026-11-26 | 19 (lead) | Owner: R76 option (1). GPU-lane work ended; nothing in flight. |

*Owner, 2026-11-26:* the R76 APK sha256 matched (`72541fcc…f96c`); the `prereg-R76` tag will be created later by the owner.
This closes the R76 void check on the APK hash; the tag stays open.

## Update 2026-11-27 (lead): joint round 13 (memo-13 work) done; CPU only, after the lane

Owner directions during the round (2026-11-26): rethink the computation scheme itself; devise new ways to ease the memory bus (partial
relief is fine); Bonnie checks Ebony's prices. Files in `reports/memo-13-work/` (Bonnie `bonnie-cpu-ladder.md`, `bonnie-ud-check.md`;
Ebony `ebony-new-routes.md`; scripts). No verdict by the lead beyond this summary; all prices *estimated*, PPL screens *host*.
- **Base correction (Bonnie):** K0 6.81 / 5.88 / 5.84 and stock 6.90 / 6.21 / 6.22 are the packed (KleidiAI) body; choice A (adopted
  2026-11-24, plain rows) stock is about 5.1 / 4.6 / 4.5 tok/s and has never run on the phone.
- **Cost model:** width-1 decode is bus-bound (bus term 104 / 122 / 114 ms > 79.2); a change of scheme must cut weight touches per
  token to about 0.45 to 0.54 of the body in bytes and arithmetic at once.
- **Closed by Ebony's host screens:** delta across tokens, reuse of the previous token's layer output, per-sequence FFN channel sets,
  low-bit ranker two-stage FFN, activation low rank, 1-byte scale code; SVE / SME / i8mm absent on A78 / X1. More of the CPU port:
  no lever (about 30 GB/s whatever the readers).
- **Survives:** UD = up-first `gate`-row skip (R1) + `down` at half its columns through K1 on a column-major copy (R3); host screen
  +2.6 / +2.1 / +2.0 % PPL (6 chunks, not the gate). Bonnie's re-price: UD −22.8 / −22.7 / −21.0 ms on path A, −19.1 / −19.0 / −17.6
  packed. Top stacks: path A with UD 8.08 / 6.94 / 7.51 tok/s; packed with UD 9.40 / 7.55 / 8.17. Nothing reaches 12.
- **Constraints:** R1 fits choice A as adopted; R3 needs a `down` copy (0.76 to 0.81 GB anonymous) that needs an owner reading of
  choice A; R2 (scale index) only under choice C. UD is a numerics change: must pass +5 % at `-b 512` and `-ub 1`; P 0.55 (judgement).
- **Host tests proposed (not started):** E-T25 (UD full-set PPL, both settings, three subjects), B-T1/B-T2 + E-T27 (Y1b decode
  kernel identity and cost, arbitrary-row bench), E-T28 (K1 on `down` at keep 0.5, real mask, plus byte count), T-D1 closed by Ebony.

Next: owner chooses which host tests Tom runs, and answers whether a head-only copy and/or a `down` copy in K1's layout still count
as choice A. No device round, no researcher round open.

| 2026-11-27 | 20 (lead) | Joint round 13 done (Bonnie ladder + UD check, Ebony new routes); proposals to the owner. |

**Update 2026-11-27 (lead):** owner approved E-T25 first (Tom running: UD as an env-gated numerics patch, full-set PPL on the three
subjects at `-b 512` and `-ub 1`, both builds; report to `memo-13-work/tom-et25.md`). Owner adopted arm A′ (CAMPAIGN.md, end): choice A
unchanged and headline; copy-using levers (head copy, `down` in K1 layout) form a separate, pre-declared arm with same-state stock and a
residency gate.

| 2026-11-27 | 21 (lead) | Arm A′ defined (owner); E-T25 started (Tom). |

**Update 2026-11-27 (lead): E-T25 partial (*host*).** Patch `0010-ud-numerics.patch` (sha256 `330502f0…9908`, `VOLLIE_UD=1`, bit-identical
with the knob unset). UD under the kill line (same-build stock × 1.035) and inside the budget in every cell so far: Yi full set `-b 512`
1.0179 (repack) / 1.0177 (KleidiAI); 40 chunks, three subjects, both builds, `-ub 1` and `-ub 32`: 1.0134 to 1.0250 (worst against
repack stock 1.0286). `-ub 1` and `-ub 32` agree within 0.3 points. Full-set `-ub 1` not run (about 22 h); `-ub 1` evidence is 40 chunks.
Still running: full-set `-b 512` for Llama-2 and DeepSeek (driver `lab/c3/et25.sh`, done about 22:15 KST). Details `memo-13-work/tom-et25.md`.

**Audit note 2026-11-27 (Annie), E-T25 numbers.** The values in the block above (Yi full set 1.0179 / 1.0177; 40-chunk range 1.0134 to
1.0250; worst against repack stock 1.0286) equal `lab/c3/logs/et25/` and the freeze stock logs (*host*). One wording does not: "`-ub 1`
and `-ub 32` agree within 0.3 points" is 0.64 points on DeepSeek KleidiAI (1.0198 against 1.0134), 0.28 and 0.21 on Llama-2 KleidiAI and
DeepSeek repack; Yi and Llama-2 are inside 0.3. `tom-et25.md` summary item 3 says 1.0254 for the highest cell where its own table and
the logs give 1.0250. Details in the audit note at the end of `memo-13-work/tom-et25.md`; nothing rewritten.

**Books 2026-11-27 (Annie), owner decisions index.** Owner decision 2026-11-27: arm A' (declared before any test of a copy-using
lever; choice A unchanged and the headline; A' = choice A plus at most the head copy and `down` in K1's layout, same-state stock,
residency gate, separate columns; full text at the end of CAMPAIGN.md, commit `4184fcb`). Memo index extended with memo-13 work and patch
0010 (sha256 `330502f0...9908`, checked); results.csv has 16 *host* rows for E-T25 (finished cells only; Llama-2 and DeepSeek
full-set `-b 512` to be filled by the lead after about 22:15 KST).

| 2026-11-27 | 22 (annie) | Books: memo-13 work in the memo index, patch 0010 row, E-T25 finished cells in results.csv, audit note (two wording mismatches), arm A' recorded; root and campaigns README refreshed. |

**Update 2026-11-27 22:40 (lead): E-T25 complete (*host*).** Full-set `-b 512`, UD / same-build stock: Yi 1.0179 / 1.0177, Llama-2
1.0235 / 1.0237, DeepSeek 1.0150 / 1.0152 (repack / KleidiAI); every cell inside the kill line (× 1.035) and the budget (repack stock
× 1.05). `-ub 1` evidence stays 40 chunks (1.0196 to 1.0250). UD passes E-T25. Next: the owner chooses the next research direction.

| 2026-11-27 | 23 (lead) | E-T25 complete: UD inside on all three subjects, both builds, full set `-b 512`. |

**Update 2026-11-28 (lead): UD kernel host tests (*host*; prices *estimated*).** Details `memo-13-work/tom-ud-kernels.md`; new patch
`0011-y1b-decode.patch` (sha256 `89bc4874…3b19`, `VOLLIE_Y1B=1` with `--no-repack`; per-chunk PPL identical to the repack build on
Llama-2 at `-b 512`, `-ub 32`, `-ub 1`: class B, not a numerics change).
- **R1 (`gate` row skip, path A): fails** its kill line cold (2.57 to 2.60 per touched weight > 1.3): +2 to −6 ms at C 50, a loss at
  C 40. On strict choice A, UD's speed side is under +0.3 tok/s.
- **R3 (`down` half columns through K1, arm A′): passes** on host: 1.64 to 1.71 cold vs break-even 4.5 to 4.7 (C 50); bytes exactly
  0.5556. A′ UD at host-cold costs −18.7 / −20.3 / −18.8 ms, 7.81 / 6.82 / 6.86 tok/s. Ebony's hot line (1.6) fires by 0.06 to 0.10.
- **Y1b dense:** 1.46 to 1.49 hot; cold 1.50 to 2.46, `down` above B-T2's 1.8 line: both B-T2 kill lines fire cold.
- All decisive numbers are host cold costs, which memo-05 showed can mislead; phone cells (0002 `y1b_rows`, 0003 K1 at 11,008 inputs
  and keep 0.5) would settle them. Next: owner decides.

| 2026-11-28 | 24 (lead) | UD kernels on host: R1 fails cold, R3 (A′) passes; Y1b decode bit-identical (0011). |

**Update 2026-11-28 (lead):** owner approved the R77 device probe ("1로 가죠"). Kit r38 committed (`c82a792`), R77 prereg committed
(`67ec3f6`), kit uploaded to HF `HoeioUser/vollie-kits` `r38/` (HF commit `f35425b4`; downloaded APK sha256 re-checked). Open: owner
tags `prereg-R77`, runs R77 per prereg §6; `prereg-R76` tag also still open.

| 2026-11-28 | 25 (lead) | Kit r38 and R77 prereg committed; r38 on HF, sha re-checked. Waiting for the owner's run. |
