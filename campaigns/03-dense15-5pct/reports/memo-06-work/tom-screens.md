# memo-06 work: Tom's host screens for the reselection (joint round 4)

Date: 2026-11-21. Author: Tom (`claude-opus-5-5`), on the owner's laptop (Apple M4 Pro). Every number is ***host***
(file headers or `llama-perplexity`-style runs on the laptop; relative only) or *estimated* (arithmetic), as labelled.
Nothing here was measured on the phone. No runtime patch was changed.

## 0. Summary

- **Tools are now model-generic** (§1). `tools/c3_colskip.py` had two silent errors on non-pre-norm blocks:
  - it zeroed the residual stream in place on post-norm blocks (OLMo-2 and Olmo-3);
  - it never calibrated the `gate`/`up` input when that input is `ffn_inp`.
  Both are fixed. The fixes are checked on 7 architectures, and the old results on Mistral are reproduced to the
  last bit.
- **No new column-sparsity or quantisation-ratio screen was run.** None of Bonnie's top five (Yi-1.5-6B-Chat,
  ChatGLM3-6B, Llama-2-7b-chat, Granite-7b-instruct, DeepSeek-LLM-7B-chat) and no other pool candidate is on the
  laptop. Free disk was 6.1 GB at the start and 2.2 to 3.1 GB later, against the lead's floor of 5 GB left after
  a download. Every candidate is therefore **"not screened: disk"** for H3 and for Q4_0 vs Q8_0/F16 (§3).
- **Header inventory (Ebony's T7) of 24 candidates** came from HTTP range reads, with no download (§2). It gives
  per-token bytes, head type, tied or untied, and architecture. Every architecture is in llama.cpp at `620d2225c`.
- **Ebony T1, partial (§4):** Mistral at `-b 512` holds about 30 % of four-neuron groups in file order and about
  33 to 35 % clustered, inside +4.5 %. Qwen2.5 and `-ub 1` were not run: memory pressure filled swap and the disk.
- The H3 results that exist are still memo-02's (40 chunks):
  - Mistral-7B-v0.3 holds 40 % and Qwen2.5-7B 35 % uniform input sparsity inside the margin at both `-b 512`
    and `-ub 1`;
  - Q4_0 = Q8_0 x 1.0195 (Mistral) and x 1.0357 (Qwen2.5).

## 1. Genericity audit and fixes (*host*)

Tools audited:
- `tools/c3_colskip.py` (calib, eval, ablate, ffnskip) and `tools/c3_h3_grid.sh`: the memo-02 and memo-05 screens.
- `tools/c3_gguf_inventory.py`: H1 and T7.
- `tools/dense_acts.py`, `tools/dense_colskip.py`, `tools/dense_headthr_check.py`.

The three `dense_*` files are campaign-1 Granite tools and are not edited. `dense_colskip.py`'s Granite group map and
`dense_headthr_check.py`'s fixed paths are superseded by `c3_colskip.py`. `c3_colskip.py` only borrows the ctypes
structs from `dense_acts.py`.

### Problems found in `c3_colskip.py`

1. **In-place zeroing is wrong on post-norm and parallel blocks** (silent). The tool zeroes the input tensor of a
   mat-mul in place. That equals skipping weight columns only if nothing else reads that tensor.
   - Pre-norm blocks are safe: the inputs are the norm outputs, `kqv_out` and the SwiGLU output.
   - OLMo-2 and Olmo-3 (arch `olmo2`, post-norm) are not: there the Q/K/V input and the gate/up input are the
     residual stream itself, so the old tool also zeroed the residual.
   - Measured on OLMo-2-0425-1B-Instruct Q4_0, 4 chunks, uniform 30 % on qkv, o and down: old tool PPL **16.31**,
     fixed tool **13.57** (stock 13.63).
2. **gate/up never calibrated when its input is `ffn_inp`** (silent). The X4 ratio branch returned before sampling.
   On `olmo2` the gate/up group got no thresholds, and eval skipped it without a word.
3. A mat-mul input that is not contiguous F32 was skipped without a word.
4. The fused `attn_qkv` weight (ChatGLM, Phi-3 style) was not in the group map.
5. A llama.h struct mismatch, which would come with a later llama.cpp base, was not caught in `c3_colskip.py`.
6. `c3_h3_grid.sh` sent stderr to /dev/null. A failing cell left an empty log, not an error.

### Fixes (in `tools/c3_colskip.py` and `tools/c3_h3_grid.sh`, uncommitted)

- **Consumer plan.**
  - The discovery ubatch now visits every graph node in compute order. For each targeted input it collects every
    reader, through views.
  - If only the target mat-muls read the input, it is zeroed in place (`inplace`), as before.
  - If other nodes read it, but only after the last target mat-mul, the tool keeps a copy, zeroes the input, and
    restores it after that mat-mul (`restore`). This is exact.
  - Otherwise the tool refuses the model with an assertion: an input that feeds two groups (parallel blocks, e.g.
    Cohere `cohere2`, Phi-2), or a reader before the last target.
  - The plan is printed as `consumer plan: {...}`. After every chunk the tool asserts that every saved input was
    restored.
- **Calibration** now also samples `ffn_inp` when it is a gate/up input.
- **Unedited inputs** are counted. They show in `RESULT` and in calib output as `UNEDITED inputs {...}`.
- **Group map:** `attn_qkv` is mapped to `qkv`.
- **Struct check:** `assert` on the default `n_batch == 2048` and `n_ubatch == 512` (dense_acts' check).
- **Grid script:** stderr goes to `<log>.err`, and the script exits with the cell name if a cell gives no `RESULT`.

### Checks (*host*)

Identity: the tool's per-chunk values with no edit, against `llama-perplexity` running values, at `-b 512` (4 chunks)
and `-b 512 -ub 1` (1 or 2 chunks):

| model (file) | arch | plan | `-b 512` | `-ub 1` |
|---|---|---|---|---|
| Llama-3.2-1B-Instruct Q8_0 | llama | all inplace | equal ([4]11.8082) | – |
| Gemma-3-1B-it Q8_0 | gemma3 | all inplace | equal ([4]24.8621) | equal ([1]18.3025) |
| EXAONE-3.5-2.4B-Instruct Q8_0 | exaone | all inplace | equal ([4]10.8292) | equal ([1]7.2049) |
| StableLM-3B-4E1T Q8_0 | stablelm | all inplace | equal ([4]7.9480) | equal ([1]5.8461) |
| Qwen2.5-7B-Instruct Q4_0 (lab file) | qwen2 | all inplace | equal ([4]6.6410) | equal ([1]5.7086) |
| OLMo-2-0425-1B-Instruct Q4_0 (allenai, sha256 `c5efa891…e222c`) | olmo2 | **restore** qkv and gu; inplace o and down | equal ([4]13.6339) | equal ([2]12.3847) |

The OLMo-2 1B file was downloaded for this test only (0.89 GB) and deleted afterwards. The small models are
pre-existing files on the laptop.

Regressions:

| check | result |
|---|---|
| Mistral-7B-v0.3 Q4_0, uniform 40 %, first table, 4 chunks | per-chunk NLL bit-equal to memo-02's `h3-mistral-b512-16.log` |
| Llama-3.2-1B, uniform 40 %: `inplace` against forced `restore` (`C3_FORCE_RESTORE=1`) | bit-equal per chunk |
| calib table on Llama-3.2-1B before and after the calib fix | identical (225 arrays) |
| ablate and ffnskip modes | run |

Not tested on a real graph:
- ChatGLM (fused `attn_qkv` and fused gate+up): covered by the map, but no file was on disk.
- The refusal path (`cohere2` parallel block): an assertion; no file on disk.

Runtime patches: not touched, so no runtime identity run was needed.

## 2. Header inventory of the pool (Ebony T7; *host*, headers only)

Method: `python3 -I tools/c3_gguf_inventory.py <gguf-py> <url>`. Only the header is fetched, by HTTP range requests.

- The file read is a stand-in with the candidate's shapes: the publisher's GGUF where one exists, otherwise a common
  re-quantiser's.
- Per-token bytes (*estimated*) are given at the campaign's form, Q4_0 body plus Q6_K head (a tied head at Q4_0),
  from the header's parameter counts.
- "arch" is the GGUF architecture. Every one of them is registered in `src/llama-arch.cpp` at `620d2225c`.
- "tool plan" is what §1 predicts for `c3_colskip.py`.

| candidate | arch | total B (header) | head | vocab | FFN share of body | Q4_0 GB/token (*est.*) | tool plan |
|---|---|---|---|---|---|---|---|
| **Yi-1.5-6B-Chat** | llama | 6.061 | untied | 64,000 | 0.78 | **3.33** | inplace (tested arch) |
| **ChatGLM3-6B** | chatglm | 6.244 | untied | 65,024 | 0.83 | **3.43** | fused qkv and gate+up; untested graph |
| **Llama-2-7b-chat** | llama | 6.738 | untied | 32,000 | 0.67 (MHA) | 3.75 | inplace |
| **Granite-7b-instruct** (QuantFactory GGUF) | llama | 6.738 | untied | 32,008 | 0.67 (MHA) | 3.75 | inplace |
| **DeepSeek-LLM-7B-chat** | llama | 6.910 | untied | 102,400 | 0.67 (MHA) | 3.76 | inplace |
| A.X-3.1-Light | llama | 7.265 | untied | 102,400 | 0.67 (MHA) | 3.96 | inplace |
| Olmo-3-7B-Instruct | olmo2 | 7.298 | untied | 100,278 | 0.67 (MHA) | 3.98 | restore (tested on OLMo-2 1B) |
| OLMo-2-1124-7B-Instruct | olmo2 | 7.299 | untied | 100,352 | 0.67 | 3.98 | restore |
| salamandra-7b-instruct | llama | 7.768 | untied | 256,000 | 0.76 | 4.05 | inplace |
| Tri-7B | llama | 7.527 | untied | 128,256 | 0.67 | 4.07 | inplace |
| Falcon3-7B-Instruct | llama | 7.456 | untied | 131,072 | 0.89 | 4.07 | inplace |
| Minerva-7B-instruct-v1.0 | llama | 7.400 | untied | 51,264 | 0.81 | 4.10 | inplace |
| Teuken-7B-instruct (research v0.4 file) | llama | 7.453 | **tied** | 250,680 | 0.82 | 4.19 | inplace |
| internlm2.5-7b-chat | internlm2 | 7.738 | untied | 92,544 | 0.81 | 4.24 | not run |
| EXAONE-3.5-7.8B-Instruct | exaone | 7.818 | untied | 102,400 | 0.81 | 4.27 | inplace (tested arch) |
| openPangu-Embedded-7B | pangu-embedded | 8.031 | untied | 153,376 | 0.79 | 4.33 | not run |
| Ministral-8B-Instruct-2410 | llama | 8.020 | untied | 131,072 | 0.78 | 4.35 | inplace |
| Llama-3.1-8B-Instruct | llama | 8.030 | untied | 128,256 | 0.81 | 4.36 | inplace |
| kanana-1.5-8b-instruct | llama | 8.030 | untied | 128,259 | 0.81 | 4.36 | inplace |
| Apertus-8B-Instruct-2509 | apertus | 8.053 | untied | 131,072 | 0.81 (2 matrices, xIELU) | 4.37 | not run |
| Qwen3-8B | qwen3 | 8.191 | untied | 151,936 | 0.78 | 4.42 | inplace (same pattern as qwen2) |
| MiniCPM4-8B | minicpm | 8.185 | untied | 73,448 | 0.85 | 4.51 | not run |
| c4ai-command-r7b | cohere2 | 8.028 | **tied** | 256,000 | 0.81 | 4.52 | **refused** (parallel block; the H3 tool cannot screen it as written) |
| granite-3.3-8b-instruct | granite | 8.171 | **tied** | 49,159 | 0.79 | 4.60 | inplace |

For reference, memo-02 H1 (*host*): Mistral-7B-v0.3 4.04 GB per token, Qwen2.5-7B 4.12.

Reading (*estimated*, bytes only):
- The two smallest bodies are Bonnie's first two, Yi-1.5-6B (3.33 GB) and ChatGLM3-6B (3.43 GB). Both are about
  17 % under Mistral.
- The 6.7 to 6.9 B MHA Llama-shaped models (Llama-2, Granite-7b, DeepSeek-7B) read 3.75 GB. Their attention is a
  third of the body, and memo-02 found attention inputs more tolerant than gate/up, which may help H3.
- ChatGLM3 is the only top-five candidate whose graph the tool has not run.
- The Granite-7b file read is a re-quantiser's (general.name "Models"). The publisher's own GGUF repository was not
  reachable without a login.

## 3. Screens not run: disk

Disk: 6.1 GB free at session start, 2.2 to 3.1 GB later. The drop was not this session's: the scratchpad held at
most 0.9 GB, now deleted. The lead's rule is at least 5 GB left after a download.

| candidate | H3 column frontier (`-b 512`, `-ub 1`) | Q4_0 vs Q8_0/F16 |
|---|---|---|
| Yi-1.5-6B-Chat | not screened: disk | not screened: disk |
| ChatGLM3-6B | not screened: disk | not screened: disk |
| Llama-2-7b-chat | not screened: disk | not screened: disk |
| Granite-7b-instruct | not screened: disk | not screened: disk |
| DeepSeek-LLM-7B-chat | not screened: disk | not screened: disk |
| A.X-3.1-Light, Olmo-3-7B-Instruct | not screened: disk | not screened: disk |
| Llama-3.1-8B (old reserve) | not screened: disk | not screened: disk |
| Mistral-7B-v0.3, Qwen2.5-7B | memo-02 §3: 40 % / 35 % inside the margin at both settings | memo-02 §0: x 1.0195 / x 1.0357 |

What a screen needs, one model at a time (*estimated*):
- Q4_0 file, 3.6 to 4.3 GB;
- Q8_0 file as the F16 proxy, 6.5 to 7.5 GB (or F16, 12 to 14 GB);
- the 5 GB floor.

So about **16 GB free** for H3 plus the Q8_0 ratio, or about **10 GB** for H3 alone. Per model this is about 1.5 h
on this laptop (*host* experience from memo-02: calib 16 train chunks, 3 cells x 2 settings x 40 chunks).

Commands, once the disk allows (the same method as memo-02 §3; files deleted after use):

```sh
B=lab/llama.cpp-c3/build/bin; T=lab/c3/text
python3 -I tools/c3_colskip.py calib $B <q4_0.gguf> $T/wiki.train.raw 16 lab/c3/thr-<m>-q4_0.npz      # check 'consumer plan' line
tools/c3_h3_grid.sh $B <q4_0.gguf> lab/c3/thr-<m>-q4_0.npz $T/wiki.test.raw 40 512 lab/c3/logs/h3-<m>-b512-40.log 0:all:0 0.3:all:0 0.35:all:0 0.4:all:0
tools/c3_h3_grid.sh $B <q4_0.gguf> lab/c3/thr-<m>-q4_0.npz $T/wiki.test.raw 40 1   lab/c3/logs/h3-<m>-ub1-40.log  0:all:0 <frontier cell>
$B/llama-perplexity -m <q8_0.gguf> -f $T/wiki.test.raw -c 512 -b 512 --chunks 40      # ratio Q4_0 / Q8_0
```

Order if disk frees up (Bonnie's top five, smallest body first): Yi-1.5-6B, ChatGLM3-6B, DeepSeek-LLM-7B, Llama-2-7b,
Granite-7b. ChatGLM3 first gets a 1-chunk identity check, because its graph has not been run.

## 4. Ebony T1: whole-neuron FFN skip in groups of four (*host*, partial: Mistral at `-b 512` only)

**Rule** (Ebony, memo-06-work/ebony-levers.md §2.3 and §5):
- Per token and layer, look at the four neurons of a group at the `ffn_down` input, a = SiLU(g) x u.
- All four are set to 0 when the group's largest |a| is under the layer's train threshold.
- That is exactly what skipping the group's four `gate`/`up` rows and four `down` columns computes. Kept weights
  are untouched (k = 1.0); with an empty skip list the result is stock.
- The score is the exact product, so this is the rule's upper bound. A proxy score (T2) is not tested here.

**Orders:**
- **file:** rows 4i to 4i+3, which are the 4x4 repack groups of the phone build.
- **clustered:** greedy grouping. Each seed neuron (file order) takes the 3 unassigned neurons whose |a| correlates
  most with it over the train tokens. This is a row permutation, exact on gate/up; `down` then needs the same
  column order.

**Calibration:**
- Text: wikitext-2 **train**, first 4 chunks (2,048 tokens). The thresholds are the per-layer quantiles of the
  group maximum over those tokens.
- 2,048 tokens is fewer than memo-02's 16 chunks; this is a screen.

**Tools:**
- `tools/c3_colskip.py`: `calib` with `C3_ACTDUMP=<prefix>` writes `<prefix>-file.npz` and
  `<prefix>-clustered.npz`; `eval` with `C3_GRP4=<table>` and group `grp4`.
- `tools/c3_grp4_table.py`: the order and threshold builder; its self-check passes.

**Test:** 40 chunks of wikitext-2 test. The 1B dry run was on Llama-3.2-1B.

Results. Mistral-7B-v0.3 Q4_0, `-b 512`, 40 chunks:

| target share of groups | file order: realised, ΔPPL | clustered: realised, ΔPPL |
|---|---|---|
| 0.20 | 0.204, +1.57 % (5.9767) | 0.193, +1.06 % (5.9469) |
| 0.30 | 0.300, **+3.65 %** (6.0990) | 0.287, **+2.60 %** (6.0376) |
| 0.40 | 0.393, +6.95 % (6.2934) | 0.379, +4.78 % (6.1657) |

- The stock reference is 5.8844 (memo-02, same chunks).
- The pass line is +4.5 % (memo-02 convention, budget ×1.05 with 0.5 points of margin).
- Log: `lab/c3/logs/t1-mistral-b512-40.log`, with RESULT and PERCHUNK lines.

**Reading (Mistral only, `-b 512` only; not yet a two-model or two-setting result):**
- The frontier inside +4.5 % is about **30 % of groups in file order** and about **33 to 35 % clustered**.
  Clustered 0.40 realised 0.38 at +4.78 %: inside the budget, outside the margin. The 0.35 cell was not run.
- Clustering lowers the cost at equal targets, by 0.5 to 2.2 points. Part of that comes from a lower realised
  share: 0.29 against 0.30 at target 0.30.
- Against Ebony's pre-data band: file order 20 to 35 %, a **hit** (about 30 %); clustered 30 to 45 %, a **hit,
  low end** (about 33 to 35 %); P(≥ 40 % on two models) 0.3, **not reached on Mistral**.

Not run, and why:
- **Qwen2.5:** its calibration made macOS grow swap until the disk had about 250 MB left. The watchdog killed it,
  and free disk stayed at about 230 MB with swap full (15.1 GB).
  - The file-order table had already been written: `lab/c3/t1-qwen25-file.npz`. The clustered table was not.
  - No 7B eval was started after that: each one holds a 4.4 GB anonymous repack copy.
- **`-ub 1` on Mistral:** not run, for the same reason and for time. About 3.5 min per cell at `-b 512`; `-ub 1` is
  several times slower.

To finish T1 once the disk has about 5 GB free:

```sh
B=lab/llama.cpp-c3/build/bin; T=lab/c3/text; Q=lab/c3/models/qwen2.5-7b-q4_0-00001-of-00002.gguf
C3_ACTDUMP=lab/c3/t1-qwen25 python3 -I tools/c3_colskip.py calib $B $Q $T/wiki.train.raw 4 lab/c3/t1-thr-qwen25-scratch.npz 8   # rebuilds both tables
for o in file clustered; do for q in 0.2 0.3 0.4; do C3_GRP4=lab/c3/t1-qwen25-$o.npz python3 -I tools/c3_colskip.py eval $B $Q $T/wiki.test.raw 40 512 - $q grp4 0 8; done; done
C3_GRP4=lab/c3/t1-mistral-clustered.npz python3 -I tools/c3_colskip.py eval $B lab/c3/models/mistral-7b-v0.3-q4_0.gguf $T/wiki.test.raw 40 512 - 0.35 grp4 0 8
# then -ub 1 (argument 512 -> 1) on each model's frontier cell, file and clustered
```
