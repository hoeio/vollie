# c3 memo-12 work (Ebony, round 4): the width-2 together cell, the guess head, the prompt's length, the gate after a lane run

Date: 2026-11-23. Author: Ebony (`claude-fable-5-2`). Not committed; the lead commits. No verdict here.
Follows [ebony-joins-r3.md](ebony-joins-r3.md); reviews [bonnie-joins-r4.md](bonnie-joins-r4.md) (read in full).

Labels: *device*, *device-derived*, *host*, *estimated*, *external*, *source* as in round 3. Triples are Yi /
Llama-2 / DeepSeek. The only new measurements are three *host* token counts (§3).

**Re-keying (item 1 of the brief): not done, because the names do not exist yet.** At 19:06
`patches/llama.cpp-c3/0007-x4-join-probe.patch` (939 lines) and `tom-r35-build.md` (18:55) held x4m and x4s to
0.60, and no `tog2`, no C11 flag mode and no hand-off line. My rows below and in r3 are keyed by description. When
the names land, the mapping is one to one: nothing in a band depends on a name.

## 1. The width-2 together cell (`x4 tog2sum` beside `x4 tog1sum`)

### 1.1 One fact that changes what the cell can show

The CPU arm is "ggml ne1 = 2 on 4-7". In the probe that is the KleidiAI buffer type (`cpu_buft=CPU_KLEIDIAI`,
*device*, R74). *Source* (`ggml/src/ggml-cpu/kleidiai/kleidiai.cpp` lines 849 to 964, base tree): with two rows
the gemv path is refused (`use_gemv = m == 1 ...`) and the call goes to the **gemm** kernel with a packed
left-hand side. That kernel works four activation rows at a time. I read its cost in memo-10 E2 as 2.3 to 2.6 × a
gemv per weight (*estimated* from assembly), and with two rows it still walks the four-row kernel.

- So the CPU's cost at two columns in this cell is most likely **c about 2.3 to 2.6, not memo-10's 1.45** (the
  1.45 is for a purpose-built W-position kernel such as `memo-10-work/ebony_gw.c`). P(the CPU alone at two
  columns reads under 19 GB/s) 0.65, my judgement.
- Bonnie saw the same risk (her r4 §0 item 2) and put it as the less likely case; her centre for the CPU alone at
  two columns is 22.0. This is our one real disagreement on this cell.
- If I am right, `both_cpu_gbs` is compute-bound in both models (13.7 against 14.1 GB/s) and **her line on the
  sum (≥ 38.5 / ≤ 35.5) and on the CPU ratio cannot separate the models.** The cell still can, on the GPU side.

### 1.2 The reading line I propose: contention ratios at two columns

The two models differ in one thing: does the bus loss that the readers inflict on each other at one column
(*device*, R74: CPU 19.77 ÷ 25.64 = 0.771; GPU 21.53 ÷ 24.88 = 0.865; sum 41.30 ÷ 50.52 = **0.817**) stay when
both become compute-heavier at two columns?

- `bonnie`: it stays. GPU both ÷ GPU alone at two columns = 0.865 by construction; sum ratio 0.85 (CPU c 1.45) to
  0.90 (c 2.4).
- `pair`: it goes, because compute-bound readers do not fill the bus. GPU ratio 0.97 to 1.00; sum ratio 0.975 to
  0.99, at either CPU cost.

| reading at two columns (all four rates from the same block) | says |
|---|---|
| `both_gpu_gbs` ÷ GPU alone ≥ 0.95, **or** (both CPU + both GPU) ÷ (CPU alone + GPU alone) ≥ 0.94 | `pair`: the bus is no longer the shared limit at width 2 |
| GPU ratio ≤ 0.90 **and** sum ratio ≤ 0.89 | `bonnie`: the one-column contention carries over |
| between | undecided; print both stack columns |

- This line does not depend on what ggml's two-row kernel costs, which is why I prefer it to a line on the sum.
- It needs the GPU-alone arm at two columns in the same block (the `tog` arms have it; x4m prints it as well).
- Her line on the sum stays valid **only if** the CPU alone at two columns reads ≥ 21 GB/s (her own guard).

### 1.3 Bands (*estimated*; GB/s of weight bytes per pass, wall rates)

| field | Bonnie r4 (centre; 80 %) | Ebony | note |
|---|---|---|---|
| tog1sum: `cpu_alone_gbs`; both CPU / GPU; `sum_vs_cpu_alone` | 25.6; 19.8 / 21.5; 1.61 (1.50–1.72) | = | R74's values |
| tog2: CPU alone, 2 columns | 22.0; 17.0–24.5 | **15.5; 13.0–22.5** | **disagree** (§1.1). P(< 19) mine 0.65 |
| tog2: CPU alone at 2 columns ÷ at 1 column | 0.86 | **0.60; 0.50–0.88** | this is memo-10's T2 for the stock path |
| tog2: GPU alone, 2 columns | 21.1; 15.0–23.5 | 19.8; 14.5–23.5 | follows the x4m centres (1.2 / 1.3) |
| tog2: `both_cpu_gbs` | 19.5; 16.0–22.5 | **15.0; 12.5–22.0** | disagree, same cause |
| tog2: `both_gpu_gbs` | 17.5; 13.0–20.5 | 18.6; 13.5–22.0 | mine is higher because a compute-bound CPU frees the bus |
| tog2: sum | 37.0; 31.5–41.5 | **33.5; 28.0–41.0** | disagree, same cause |
| tog2: GPU both ÷ GPU alone | 0.83 (from her centres) | **0.94; 0.85–1.00** | **the model line (§1.2)** |
| tog2: sum ratio (both ÷ alone, two columns) | 0.86 (from her centres) | **0.95; 0.85–1.00** | the model line |
| P(the line reads `pair` / undecided / `bonnie`) | – | 0.50 / 0.25 / 0.25 | mine |

What each outcome means for the stack:
- **`pair` and CPU ratio (2 columns ÷ 1) ≥ 0.80:** the pair stack as printed (K 300: 11.9 / 11.9 / 11.4 at C 40).
- **`pair` and CPU ratio near 0.6:** the bus physics is as I say, but stock ggml cannot deliver width 2 cheaply on
  the CPU side. The lane then needs the W-position CPU kernel (host-exact on random data only, memo-10 E2); the
  pair stack is conditional on building it. Until then width 2 on the lane prices near the committed form.
- **`bonnie`:** her r3 table (K 300: 10.8 to 11.3 / 10.7 to 11.2 / 10.3 to 10.8).

### 1.4 Review of Bonnie's r4

- **Her §7.2 merged table says of x4a, x4b, x4f, x4c, x4d, x4r, x4i: "mine only; Ebony has given no bands for
  them and has not reviewed them". That is not so.** My r2 part B.3 (committed in `7db2f41`) reviews every one of
  those rows and gives my own value where I differ. For the pre-registration these should be merged too:

  | cell, field | Bonnie | Ebony (r2 B.3) |
  |---|---|---|
  | x4i `svm_fine_buffer` present | P 0.30 | **P 0.70** (vendor guide §7.5) |
  | x4b best `ovh_us`; P(≤ 150) | 215 (140–280); 0.12 | 205 (120–290); 0.20 |
  | x4b spin cells on cpu 4 / cpu 7 | 255 / 250 | 235 (130–345) |
  | x4a `gap_p50_us`, launcher 0-3, low end | 145 | 110 |
  | x4a `gap_p50_us`, launcher cpu 4 | 210 (140–285) | 185 (100–285) |
  | x4a the driver serialises | P 0.60 | P 0.50 |
  | x4f the driver prepares at enqueue | P 0.40 | P 0.30 |
  | x4c `gate_up_vs_11008` | 1.14 (1.04–1.20) | 1.10 (1.00–1.20) |
  | x4d replay overhead per launch; ratio to sync1; passes | 138; 0.87; P 0.25 | 170; 0.80; P 0.20 |
  | x4s `wait_ms` at share 0.50 | 0.04 | 0.08 (0.02–0.25) |
  | x4r held / killed | P 0.35 / 0.10 | P 0.25 / 0.15 |

  and the three rows I asked to add (GPU device-time rate in x4s; `cpu_ms` ratio at shares 0.25 and 0.50; gap
  against buffer size in x4a).
- **Her merged x4m, x4s, x4e and hand-off rows: agreed as merged.** `dev_ratio(2)` 1.25 (1.05–1.80) is fine.
- **Her "K in a lane = launch dead time + hand-off": agreed;** it is the rule I asked for in r3.
- **Her clock merge (C 40 to 50 for a gated claim run, C 35 sustained): agreed,** with §3 and §4 below as the
  conditions.
- **Her head on the lane (§2): agreed as priced** (+0.06 to +0.22 tok/s; a rider).
- **Her x4s slope reading (CPU slowed by 1 + 0.45 × GPU duty in her fit; near 0.30 would favour `pair`):** agreed
  as a second, free reading of the same question.

## 2. The guess head: what raises top-1, priced against the line

### 2.1 Where the line is now

The 0.50 line was memo-09's kill line for lever M in the committed stack. In the pair model the head top-1
needed for 12.00 depends on K (*estimated*, `ebony_pair.py`, 60 µs of jitter per split stage, C7, C5, O 5.2):

| in-situ K (with hand-off) | C 40 | C 35 | have (*host*, `memo-09-work/tom-head.md`) |
|---|---|---|---|
| 150 | 0.30 / 0.43 / 0.43 | 0.35 / 0.48 / 0.49 | 0.379 / 0.497 / not trained (0.438 assumed) |
| 220 | 0.36 / 0.48 / 0.49 | 0.40 / 0.53 / 0.54 | |
| 290 | 0.41 / 0.53 / 0.54 | 0.46 / 0.58 / 0.60 | |
| 452 | 0.50 / 0.62 / 0.64 | 0.57 / 0.69 / 0.70 | |

- +0.05 of top-1 is worth about +0.35 tok/s on each subject near K 290.
- So at R75's likely K the heads are short by 0.03 to 0.10 (Yi), 0.03 to 0.09 (Llama-2) and about 0.10 to 0.16
  (DeepSeek, if its head lands where the other two average). **DeepSeek is the binding subject on this term.**

### 2.2 What is known to move a head (sources, with numbers where they give any)

| lever | evidence | expected on our subjects | cost |
|---|---|---|---|
| More self-distilled data | *host* (Tom §4.3): Llama-2 +0.04 per doubling (0.394 → 0.434 → 0.476), Yi +0.02 to +0.03. Overfitting is the limit (Yi train 0.78, held-out 0.33) | Llama-2 0.52 to 0.54 at 0.9 M pairs; Yi 0.40 at 0.6 M, 0.50 needs 5 to 7 doublings | 2 h of Metal dumps and 7 GB per doubling at Tom's 120 rows/s |
| **Teacher-forced dumps at `-b 512`** (memo-10 T12, not run) | the label is the model's own argmax at every position of any text, so no greedy generation is needed; rows per second are the prefill rate, not the decode rate | 10 to 30 × more rows per hour, so 2 to 4 doublings in the time of one. If the slopes hold: Llama-2 0.58 to 0.64, Yi 0.44 to 0.50 (*estimated*; slopes usually flatten, and teacher-forced text is not the model's own continuation, so a domain gap is possible) | one script change; hours; disk 8 KB per row is the limit (1 M rows = 8 GB) |
| Soft targets (distil on the stored top-64 logits, not the hard token) | *external* (Hydra, arXiv 2402.05109, read through the fetch tool): a teacher loss alone gave 1.04 × throughput over the label loss on Vicuna-7B; all their later heads use it. No accuracy figure in the text | +0.01 to +0.03, mostly through less overfitting (my judgement) | none: Tom's dumps already hold each row's top-64 logits |
| Conditioning on the token just produced (`tok`, EAGLE's and Hydra's idea) | *host*: +0.02 on both models. *External* (Hydra): sequential dependence gives 1.11 × Medusa's throughput on Vicuna-7B | already in the numbers above | – |
| Deeper block (Hydra++: 4-layer MLP; "5+ layers gave no added benefit") | *external* (Hydra): part of 1.27 × over Medusa together with the teacher loss and a decoder layer | small or negative here: capacity is not our limit, data is (rank 512 and weight decay did not help, *host*) | + guess ms per head evaluation |
| A draft decoder layer (EAGLE; Hydra++'s extra layer: 1.12 × acceptance length) | *external* (EAGLE, memo-10 E5): 0.76 first-token on Llama-2-chat-7B with a 0.24 B layer trained on 68,000 dialogues | 0.65 to 0.75 on Llama-2 with the published weights (measurable today, memo-10 T13); unknown on Yi and DeepSeek | **an owner question** (a draft model, not a guess head; memo-10 E5); 5.7 to 6.2 ms per draft step; not a laptop job to train |
| Features from an earlier layer beside the last | *external* (Future Lens, arXiv 2311.04897, **abstract only**): a single hidden state of GPT-J-6B predicts later tokens "with more than 48 % accuracy" at some layers | unknown; +0.00 to +0.04. The last state is shaped for token t + 1; a mid-layer state may carry more of t + 2 | host test on existing prompts needs the dump tool to store one more layer (2 × disk) |
| A per-step chooser between prompt lookup and the head | *host* (Tom §4.2): lookup-then-head gives Llama-2 0.548 on the official prompt (it copies its notes) but hurts on the set; "a rule that picks per step was not tried" | +0.02 to +0.05 on the set, more on the official prompt | host, minutes, on existing dumps |
| A confidence gate (run width 2 only when the head is sure) | memo-10 Y6: −2 to −6 ms in the committed model | worth **less** in the pair model: a wrong guess costs kB 1.06, not 1.12 to 1.27 | host, minutes |

Reading:
- Medusa's 0.60 and EAGLE's 0.76 are for heads trained on 68,000 ShareGPT dialogues with a decoder layer or on
  in-domain chat; ours are trained on 1,500 to 3,000 Dolly prompts with one residual block. **The cheapest
  untried step is more rows per hour (teacher-forced dumps) with soft targets**; both are host work on tools that
  exist.
- Against the table in 2.1: Llama-2 at 0.55 to 0.60 would meet the K 290 line at C 40 and miss it at C 35. Yi at
  0.44 to 0.48 meets it at C 40. DeepSeek needs 0.54 to 0.60, which only the data route could give and nothing
  measured supports yet.
- P(all three heads at or above their K 290, C 40 lines after the data route): 0.15, my judgement
  (Yi 0.6 × Llama-2 0.6 × DeepSeek 0.4).

### 2.3 What a host test would show, in order of cost

1. **Soft targets and the per-step chooser on the existing dumps** (minutes each). Pass: +0.02 on the prompt set
   for either; else drop it.
2. **Teacher-forced dump rate** (one short run): rows per second at `-b 512` on the laptop. Under 5 × the greedy
   rate closes the data route as a laptop job.
3. **One more doubling on Llama-2 with teacher-forced rows**, scored on the prompt set (greedy, CPU, as now).
   Pass: ≥ 0.53. A value under 0.50 means teacher-forced rows do not transfer to greedy decode, and the route is
   back to greedy dumps at 2 h per doubling.
4. **DeepSeek's head** (below).
5. Optional, memo-10 T13: the published EAGLE layer on Llama-2. It bounds what a draft layer could give and costs
   a 0.5 GB download (the lead's call); it cannot ship without the owner.

### 2.4 What Tom needs to train DeepSeek's head

- **Nothing new in code.** `tools/c3_head_dump.cpp`, `c3_head_prompts.py` and `c3_head_train.py` are
  model-generic by their description (hidden size and head read from the GGUF; the head is the file's own Q6_K
  output matrix, dequantised exactly).
- Inputs: the served file `lab/c3/models/deepseek7b-q4_0.gguf` (the unmodified one, owner 2026-11-22; **not**
  `deepseek7b-bos-q4_0.gguf`), its sha256 in the head's metadata; the same Dolly split; the model's own chat
  template as in the file.
- Work: 3,000 training prompts → about 450,000 pairs: about 1 to 1.5 h of Metal dumps and about 7 GB of fp16
  hidden states at Tom's rates (*host*, his §4.3), 2 to 3 minutes of training, then the 32-prompt CPU dump (one
  sequence, run alone) and the official prompt.
- Two checks specific to this subject:
  1. Vocabulary 102,400 (against 64,000 and 32,000). The shortlist result (8,192 most frequent training targets
     within ±0.01 of the full head) was shown on the other two; re-check it here, and the head's phone cost
     (13.88 ms full, *device*, R72) makes the shortlist matter most on this subject.
  2. Tom's self-check that the dequantised head reproduces llama.cpp's greedy token (0.9999 / 1.0000 on the
     others) must pass on the file **without** the BOS change, with the template the phone will use.
- Licence: not checked by me today. The lead should read the DeepSeek-LLM-7B-chat licence for a clause on
  training on outputs before a head ships (Tom did this for Llama-2).
- Disk: the laptop was at 0.2 GB free on 2026-11-21 and the freeze used about 28 GB; 7 GB must be free first.
- My band for its prompt-set top-1 at 450,000 pairs: **0.43 (0.36 to 0.50)**. It is a Llama-architecture model
  with a large bilingual vocabulary, like Yi; I put it between Yi and Llama-2, nearer Yi.

## 3. The official prompt's length per subject, and the prefill's heat

*Host*, 2026-11-23: `lab/llama.cpp-c3/build/bin/llama-tokenize --show-count` on the served files in
`lab/c3/models/`, input `tools/c2_official_prompt.txt` (702 bytes).

| subject | prompt text alone | with a chat wrapper I wrote by hand |
|---|---|---|
| Yi (`yi15-6b-q4_0.gguf`) | **147** | 167 (ChatML: `<|im_start|>user … <|im_end|>`, assistant header) |
| Llama-2 (`llama2-7b-q4_0.gguf`) | **179** | 186 (`[INST] … [/INST]`) |
| DeepSeek (`deepseek7b-q4_0.gguf`) | **147** | 153 (`User: … Assistant:`) |

- The first column is the tokenizer's own count (the tool's default handling of BOS). The second column uses my
  hand-written wrappers, **not** the templates stored in the files or the one the app applies; it is an upper
  estimate of the wrapper's size (6 to 20 tokens), not a measured prompt length. The app's real count is in any
  run's log (`n_prompt` / prompt-eval tokens); R73 would print it.
- So the prefill is **about 150 to 190 tokens**, inside my r3 estimate (170 to 200).

Prefill time and heat (*estimated*):
- No dense 7B prefill was ever timed on this phone. Prefill is compute-bound (the weights are read once per
  batch). With the four-row kernel at 2.3 to 2.6 × a gemv for four rows (0.6 × per row) and C 40 to 50, the
  compute rate is 65 to 80 GB/s of weight bytes per position: about 21 to 26 / 18 to 22 / 19 to 23 tok/s.
- That gives **6 to 8 s (Yi), 8.5 to 10.5 s (Llama-2), 6.5 to 8 s (DeepSeek)** at full clocks, up to 1.5 × that if
  the clocks or the X1's placement are bad: a band of **6 to 15 s**. (My r3 figure of 12 to 15 s assumed the gemv
  rate; this is the corrected one.)
- R74's timeline under CPU + GPU load from a gated start: full caps to about 15 s, 2,150 MHz at about 17 s,
  1,785 at 25 to 35 s (*device*). Prefill is CPU-only load, which R72 L4 shows is enough by itself to cap
  (1,555 to 1,785 in long cells).
- **Prefill plus a 3 to 4 s decode is 9 to 19 s of load.** In the centre case the whole run ends before the first
  cap step; in the slow case the decode starts at full caps and may end at 2,150. So r3's odds stand or improve a
  little: C ≥ 40 with P 0.55, C 33 to 40 with P 0.35, lower with P 0.10. Llama-2 has the longest prompt and is the
  most exposed.
- Running the prefill on the GPU (upstream's batched Q4_0 kernel is in the tree) would take the heat off the big
  cores before the decode. Not priced: it needs the whole graph on the GPU for the prefill, which is not the lane.

## 4. What a lane run does to the next gate (R74's gate series)

The gate waits for thermal headroom ≤ 0.65, at most 180 s. R74's four gates (*device*, 5 s samples):

| gate before | load that came before it | headroom at start → peak | time to 0.72 | time to 0.65 | result |
|---|---|---|---|---|---|
| `x3b` | the T-A cap probe (three spinning threads for 3 min, ended 18 s earlier) | 0.863 → 0.881 | about 40 s, then it drifted back up to 0.81 twice | not reached (0.67 to 0.71 at 150 to 180 s) | **timed out** |
| `x3g` | `x3b` (1.3 s of GPU work) | 0.709 → 0.737 | – | 85 s | passed |
| `x3c` | `x3g` (32 s, GPU alone) | 0.897 → 0.912 | 30 s | 60 s | passed |
| `x3r` | `x3c` (35.5 s, four big cores + GPU) | 1.172 → 1.21 | 35 s | not reached (0.66 at 175 to 180 s) | **timed out** |

Reading (*device-derived*):
- **Two phases.** Headroom falls to about 0.72 within 30 to 40 s whatever the load was. From 0.72 to 0.65 it
  takes another 30 to 140 s or more, and that tail depends on the heat already in the phone (both time-outs came
  after minutes of earlier load).
- **Load to peak:** 32 s of GPU alone left 0.90; 35.5 s of CPU + GPU left 1.17. A lane run is 9 to 19 s of load
  (§3), mostly CPU prefill: I expect a peak of **0.85 to 1.0** (*estimated*, between the two rows).
- **The next gate after a lane run:** about 30 to 40 s to 0.72, then 0.65 in 60 to 120 s early in a session and a
  time-out at 0.66 to 0.70 later in it. P(time-out) per gate: about 0.2 for the first three runs, 0.5 by the
  tenth (*estimated* from two time-outs in four gates, both after long load).
- **A time-out is not a capped run.** At 0.66 to 0.70 the cap is ≥ 2,000 MHz in 88 % of campaign 2's runs
  (memo-15 §1.3, *device*). The damage of a time-out is 180 s of waiting, not the clock: **a 10-run claim block
  takes about 15 to 35 minutes** (10 × (9 to 19 s + 60 to 180 s)).
- What would cap a run is thermal status ≥ 1 at its start (R68: seen at headroom 0.77 to 1.0), which the gate at
  0.65 to 0.70 avoids.
- Campaign 2's page hold through the gate wait (R60: +4.7 / +5.3 tok/s, 5 s of CPU per wait) matters more here:
  a 3 to 4 GB body must stay resident for 60 to 180 s between runs, and a lane holds 1.0 to 1.5 GB of it as GPU
  buffers that cannot be dropped. That part is in the lane's favour.

For the R75 result: the kit prints the gate series before every block. The gates after x4s (CPU + GPU, 18 arms)
and x4m's together arms are the first after lane-like load from a cool start; read their time to 0.72 and to 0.65
against this table.

## 5. Next questions, ranked, and what I did

| # | question | state |
|---|---|---|
| 1 | What does ggml's two-row path cost on the A78 (c at width 2 for the stock CPU kernel)? | Read in the source (§1.1): it is the four-row gemm; my estimate 2.3 to 2.6. The together cell's CPU-alone arm measures it. If ≥ 2, **the W-position CPU kernel becomes a build item of the lane**, not an option |
| 2 | Is the W-position kernel exact inside llama.cpp, not only on random data? | not done; memo-10 T10's added arm. Host, Tom |
| 3 | Teacher-forced dump rate and transfer to greedy decode | not done; host tests 2 and 3 of §2.3 |
| 4 | DeepSeek's head | specified (§2.4); Tom's to run |
| 5 | The app's real prompt length and prefill time per subject | host counts done (§3); the device values need R73 |
| 6 | Does a GPU prefill leave the big cores cool for the decode? | not priced; needs upstream's whole-graph OpenCL path to run on the 660 (campaign 1: it ran, 0.7 to 2.9 tok/s decode, *device*) and the decode to switch to the lane afterwards. A design question first |
| 7 | Gate time-outs across a 10-run block with lane load | estimated (§4); R75's gate series is the first record |
| 8 | Licence of DeepSeek-LLM-7B-chat on training with outputs | not checked; the lead's |

## 6. Sources (2026-11-23)

Repository: `lab/llama.cpp-c3` (`ggml/src/ggml-cpu/kleidiai/kleidiai.cpp` lines 685 to 964, read);
`lab/llama.cpp-c3/build/bin/llama-tokenize` run on `lab/c3/models/{yi15-6b,llama2-7b,deepseek7b}-q4_0.gguf` with
`tools/c2_official_prompt.txt` (*host*); R74 JSON (gate series, block times); `memo-09-work/tom-head.md` §1, §4;
`memo-10-tree.md` E2, E5, E8; `campaigns/02-25toks-5pct/reports/memo-15-cost-model-refit.md` §1.3;
[bonnie-joins-r4.md](bonnie-joins-r4.md).

External: Hydra, arXiv 2402.05109v2 (read through the fetch tool; no per-head accuracy in the text, the values are
in figures): <https://arxiv.org/html/2402.05109v2>. Future Lens, arXiv 2311.04897 (**abstract only**):
<https://arxiv.org/abs/2311.04897>. EAGLE, arXiv 2401.15077, and EAGLE-2, 2406.16858 (**search summary only**
this round; the 0.76 first-token figure for Llama-2-chat-7B is the one read for memo-10 E5).
Not found: any published top-1 for a one-block next-next head on Yi-1.5 or DeepSeek-LLM; any 2026 paper giving a
head's accuracy against training-set size.
