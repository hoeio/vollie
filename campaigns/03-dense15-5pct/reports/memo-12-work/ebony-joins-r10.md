# c3 memo-12 work (Ebony, round 10): what a width-2 pass can share in the Q8_0 attention, a repair of the head inside KleidiAI's chain, review of Bonnie r19 and r20

Date: 2026-11-24. Author: Ebony (`claude-fable-5-2`). Not committed; the lead commits. No verdict here.
Follows [ebony-joins-r9.md](ebony-joins-r9.md). Reviews [bonnie-joins-r19.md](bonnie-joins-r19.md) and
[bonnie-joins-r20.md](bonnie-joins-r20.md) (their "answer first" sections; the bodies not read).

Labels as before. Triples are Yi / Llama-2 / DeepSeek. New *host* numbers: `ebony_head_outliers.py` (new) and one
two-fold run on Yi's head dumps. No model ran, no Metal, no phone.

## 0. Answer first

1. **At width 2 the exact sharing in the attention is the value conversion and almost nothing else.** By count,
   the conversion is about half of the vector operations of a key row; the key's scales, its byte loads and the
   mask add 2 ns. The integer score, the exponential and the weighted add are per query and cannot be shared.
   Bonnie's 0.9 / 1.7 / 1.6 ms per pass stands; my additions are under 0.1 ms.
2. **Two exact changes at width 1 came out of the same reading, and neither is in any stack:**
   - **Yi only:** its 32 query heads read 4 key-value heads, so the op converts each value row **eight times**
     per query. One conversion per group is exact. Saving about 0.3 of Yi's attention (*estimated*: 0.5 to 1.0 ms
     per token at 200 positions).
   - **All three:** the op walks one head at a time over all cached rows, so every step lands 4,352 bytes from
     the last one. A key-major order (all of a thread's heads for one cached row, then the next row) reads the
     cache in contiguous kilobyte pieces and is exact. Its value is the cache-miss time of today's walk, which
     nothing has measured on the phone: 0 to 5 ms per token at 200 positions (*estimated*, wide on purpose).
3. **The head can be repaired inside KleidiAI's chain with sixteen float columns.** On Yi (*host*, proxy, against
   the float head): stock chain +0.0041 in −log p; with 8 fixed dimensions carried in float +0.0010; with 16
   +0.0007; with 32 +0.0005, which is the int8 weights' own share. The dimensions must come from data: the final
   norm's gain does not name them. Cost on the phone about 0.1 ms per token (*estimated*). It is a fallback for
   the case where the repack head is slower than the int8 head on the phone (Bonnie's knob needs no new code).
4. **A correction to the head proxy, hers and mine:** the dumped token was chosen by ggml's path, so scoring it
   favours that path. ggml's path reads −0.0013 **below** the float head on Yi (3 standard errors). The
   path-neutral size of Yi's head loss is +0.0041, not +0.0054. My r7 band for Yi becomes **+0.3 to +1.2 %**.
5. **Bonnie r19 and r20: agreed.** They answer two of my three r9 requests better than I put them (an idle
   anonymous part instead of my flat 0.25 GB; the head as a second mapped file). One piece of evidence for her
   poison arm in §3.

## 1. Width 2 in the phone's attention (my question 4)

### 1.1 What one key row costs (*source*: `ops.cpp` lines 8985 to 9156; counts for DK = DV = 128)

On the phone the cache is Q8_0, so every query takes the per-query path (r8 §1.1). For one query, one head and
one cached row `ic`:

| step | operations | time (*estimated*; Bonnie r17 §2.3 where marked B) | function of |
|---|---|---|---|
| mask | 1 half-to-float, 1 multiply, 1 compare | 1 ns | the query's row of the mask |
| score: `vec_dot` of the Q8_0 key with the Q8_0 query, 4 blocks | 8 loads of key bytes, 8 of query bytes, 8 SDOT, 4 SCVTF, 8 half-to-float (two scales per block), 4 multiplies, 4 FMLA | 10 ns (B) | key row **and** query |
| online softmax | scale, mask add, compare with the running maximum, one `expf`, the running sum; on a new maximum 32 vector multiplies over the accumulator | 25 ns (B) | the query's own state |
| **value conversion**: `dequantize_row_q8_0`, 4 blocks | 32 loads, 64 widenings, 32 SCVTF, 32 multiplies, 32 stores, 4 scale conversions | **36 to 45 ns (B)** | **the value row only** |
| weighted add: `vec_mad_f32` | 32 loads of the value, 32 of the accumulator, 32 FMLA, 32 stores | 30 ns (B) | value row and the query's weight |
| **sum** | about 330 vector operations, 170 of them the conversion | about 105 ns | |

Per token that is rows × heads × layers: 200 × 32 × 32 = 205,000 row steps on Llama-2, 21 ms of one thread, 5 to
6 ms on four (her r15: 3 to 6.5 ms).

### 1.2 What two queries can share, exactly

| shared | saving per second-query row | why it is exact |
|---|---|---|
| the value conversion | 36 to 45 ns | the converted row is a function of the cached bytes alone; both queries then read the same 128 floats in the same order of rows into their own accumulators |
| the key's four scale conversions and its 8 byte loads | 1 to 2 ns | half-to-float is exact; the product of the two scales is one multiply either way |
| the mask's conversion | under 1 ns | exact conversion; the two queries keep their own mask rows (the second sees one more key) |
| cache misses on the key and value rows | all of them | already free: the second query reads what the first just read (her r16) |
| **not shareable:** the integer score, `expf`, the running maximum and sum, the weighted add | – | each is a function of the query |

- So a second position costs about 60 of 105 ns per row: **0.57 to 0.62 of the first** (*estimated*).
  Her saving per pass (0.9 / 1.7 / 1.6 ms) is this, and it is the whole of what is exact at width 2.
- Things I looked at and dropped because they are not exact: skipping keys whose weight is too small to change
  the accumulator (true for most elements, not provably for all); a vector `expf` over heads (another
  implementation, other last bits).
- The second query's key set is the first's plus the first position's own row. That row is written to the cache
  by a `SET_ROWS` node before the attention node in the same graph (*host*, my node listing of r8), so nothing
  special is needed.
- Check when built: E-T17 of r9 (`EBONY_B=2 EBONY_Q8KV=1`, zero differing nodes).

### 1.3 Two exact changes at width 1

**(a) Yi: one conversion per key-value head, not per query head.** *Host*, GGUF headers: Yi has 32 heads and 4
key-value heads; Llama-2 and DeepSeek have 32 and 32. In the op, `iv2 = iq2 / rv2` (lines 9032 to 9034): eight query heads
read the same value row and each converts it again.
- Change: loop the eight heads of a group inside the row loop and convert once.
- Exact: each head keeps its own maximum, sum and accumulator and sees the rows in the same ascending order; only
  the interleaving between heads changes.
- Saving: 7 of 8 conversions, 0.30 to 0.37 of Yi's attention. If Yi's attention is 1.5 to 3 ms per token at 200
  positions (her r15 band for Yi's whole O is 3.5 to 8), that is **0.5 to 1.0 ms per token**, at every width.
- It does not help the binding subjects. It is model-generic code (any grouped-attention model gains).

**(b) Key-major order.** The cache view is [128 values, 32 heads, rows] (*host*, r8's listing: `cache_k_l0 (view)`
ne [128, 32, 256]). The op takes one head and walks `ic` over all rows: 136 bytes of key, then 136 of value, and
the next step is a whole cached row further on (4,352 bytes for K, the same for V).
- Per layer at 200 positions the cache is 1.7 MB; over 32 layers 56 MB. One token later the body's 3.6 GB have
  passed through the caches, so each of these reads comes from memory again.
- Today: each (head, row) step touches about 5 new cache lines, more than a page apart. A stride that crosses a
  page each step is the case hardware prefetchers handle worst, and one loop iteration is about 300 operations,
  so the core cannot run far ahead to overlap misses. **If each step waits for memory, a row costs nearer 200 ns
  than 105.** Whether it does is not known: no record times this op on the phone, and I found no statement of
  the A78's prefetcher behaviour across pages (not searched today either).
- Change: for each thread's set of heads, make the row loop the outer one. Then a thread reads 8 × 136 bytes of
  K and of V contiguously per row, 17 lines in sequence, and its 8 accumulators (4 KB) stay in the first-level
  cache.
- Exact: the same argument as (a).
- Value: the miss time of today's order. **0 to 5 ms per token on Llama-2 and DeepSeek at 200 positions**
  (*estimated*: nothing if the misses already overlap, about the attention's own compute again if they do not);
  it grows with context length.
- (a) falls out of (b) for free: with the row loop outside, the eight heads of a Yi group are adjacent.
- A laptop cannot price it (large caches, another prefetcher). Bonnie's r19 §2 host profile gives the op's share
  of a token on the laptop, not this. It needs the cell of §4.

## 2. A repair of the head inside KleidiAI's chain (my question 3, on Yi's dumps)

### 2.1 Design

The loss on Yi comes from a few dimensions of the final state that are 10 to 20 times its rms on every row
(r7 §2.4). One scale over the whole vector then gives every other dimension a few levels.

1. A fixed list D of k dimensions per model.
2. Before the head's activation quantiser: copy the k values aside and set them to zero in the vector. The
   asymmetric quantiser represents zero exactly (its range always includes 0 and the zero point is an integer;
   checked in `ebony_head_outliers.py`), so the stock int8 kernel then adds exactly nothing for those columns.
3. After the kernel: add W[:, D] · x[D] in float, with the columns taken once at load from the file's Q6_K head.
4. Cost per token: vocabulary × k multiply-adds on a contiguous block. At k = 16: 1.0 / 0.5 / 1.6 M of them,
   about **0.1 / 0.05 / 0.16 ms** on one thread (*estimated*), against a head of 9.04 / 4.40 / 13.88 ms
   (*device*, R72). Memory: 4 / 2 / 6.5 MB.
5. About 40 lines around the head's call in `compute_forward_qx`, behind a knob. A numerics change, towards the
   file.

Where D comes from:
- **Not from the weights.** I tested the obvious load-time rule, the largest final-norm gains
  (`output_norm.weight`): on Yi the gain's range is small (median 3.6, largest 7.8) and its top 8 hold only 5 of
  the 8 dimensions that are largest in the data; the second and third largest (2292, 3017) are missing. With 32
  dimensions by gain the loss only falls to +0.0017 against ggml's path.
- **From data:** the k dimensions with the largest |x| ÷ rms seen over a calibration text. That makes D a
  calibration-derived table, which the owner already ruled on (2026-11-19: a sidecar with the model's sha256, a
  committed generator, its own check at both batch settings).

### 2.2 Result on Yi (*host*, proxy; 3,563 rows of the 32-prompt set; dimensions chosen on one half of the prompts and applied to the other, both ways)

−log p(dumped token) minus the same through the float head (the file's Q6_K weights, float activations); mean
and standard error:

| head path | change | s.e. |
|---|---|---|
| ggml's (Q8_K activations in blocks of 256, Q6_K weights) | −0.0013 | 0.0004 |
| KleidiAI's chain as it is | **+0.0041** | 0.0009 |
| with 4 dimensions in float | +0.0027 | 0.0006 |
| with 8 | **+0.0010** | 0.0005 |
| with 16 | +0.0007 | 0.0003 |
| with 32 | +0.0005 | 0.0002 |

- **Eight dimensions remove three quarters of the loss; sixteen leave what the int8 weights cost by themselves**
  (Bonnie r11: +0.0006 for the weights alone). The activation error falls from 4.8 % of rms to 1.4 % at 16 and
  0.8 % at 32, below ggml's 2.4 %.
- **Row 1 is negative, and that is a flaw of the proxy, not a property of ggml.** The dumps were made on ggml's
  path, so the scored token is the one that path's logits put first. Scoring it favours that path by about
  0.0013 on Yi. Every "against ggml" figure of r7, r11 and r12 carries it. Against the float head, which did not
  choose the token either, Yi's loss is +0.0041. I lower my r7 band for Yi's stock shift to **+0.3 to +1.2 %**
  and leave DeepSeek's (+0.3 to +1.2 %) and Llama-2's.
- Llama-2 (first run of the script, gain-ranked dimensions): no effect at any k, as expected (+0.0001 to 0.0000).
- DeepSeek: no dump. Its final norm's gains are flat as well (median 0.32, largest 0.6), so its list also needs
  data. With Tom's E-T10 dump the script gives its table in a minute.

### 2.3 Where it stands beside the head knob

| | head knob (repack type), Bonnie r12 | float columns in KleidiAI's chain |
|---|---|---|
| shift against the freeze's head | 0 by construction | about +0.0007 in −log p on Yi (the int8 weights) |
| new code | none (the knob exists) | about 40 lines and a sidecar table |
| head time | 0.85 × the int8 head (*estimated*; my band 0.90, 0.78 to 1.05) | the int8 head + 0.05 to 0.16 ms |
| needs calibration | no | yes |

**Use the knob.** The float columns are the answer only if R76's `x5p` shows the repack head slower than the
int8 head on the phone by more than the budget is worth, or if a later lever needs the head to stay in
KleidiAI's buffer.

## 3. Review of Bonnie r19 and r20

| her claim | my position |
|---|---|
| r19: the load logs confirm the buffer sizes to the megabyte | agree; Llama-2's 3,576.54 MiB is also in my r6 log ("CPU_REPACK … 3576") |
| r19: flash attention "auto" resolved to enabled in the freeze | agree, independently: my node listing shows a `FLASH_ATTN_EXT` node per layer (*host*, r8) |
| r19: hot anonymous 0.16 / 0.20 / 0.20 GB; an idle part of 0.15 / 0.20 / 0.30 GB added to each step | **agree, and better than my r9 request** (a flat 0.25 GB). The compute buffer I pointed at is in her idle part, where it belongs |
| r19: a repack head should not stay anonymous under C; a second mapped file | agree; it is my r9 request 2 in its cleaner form |
| r19 §2: nine `llama-bench` runs per subject with the per-op profiler | not reviewed. One caution from §1.3 here: the attention's share on the laptop understates the phone's if today's walk waits for memory there |
| r20: the compute buffer holds nothing that one graph leaves for the next, so its pages can be released between calls | agree with the reading for the nodes that feed the result. **Evidence for her poison arm:** in my node comparisons one node differs between two identical runs every time, `node_1`, with values near 1e30. It is the token lookup of the "mixed" input branch (`llama-graph.cpp` lines 2642 to 2661); with a token-only batch that branch's inputs are never written and `ggml_build_forward_select` discards its result. So the graph does contain nodes whose buffers hold stale memory. A poison arm will light up there (`node_1`, `node_2`, the `SET_ROWS` view). That is expected and harmless as long as the selected branch is the token one; the arm should compare the logits and the nodes downstream of the select, not every buffer |
| r20: the cost is paid at the next prompt, 0.05 to 0.2 s | agree it is outside the metric |
| r20: the repack head maps from a file by the same mechanism, about 40 lines more | agree with the reasoning; not checked in the source |

## 4. Lists (additions only)

**Tests**

| # | test | where | cost | closes when |
|---|---|---|---|---|
| E-T18 | a model-free cell for the attention walk: the flash-attention op alone on a Q8_0 cache of Llama-2's shape at 200 and 500 rows, (i) as it is, cache evicted between calls by a 64 MB read; (ii) the same, cache hot; (iii) key-major order | device, a rider in a kit that already loads ggml graphs (R76's `x5p` is one) | about 10 s | (i) ÷ (ii) ≤ 1.15 → the walk does not wait for memory and (b) is closed; otherwise (iii) ÷ (i) is (b)'s price |
| E-T19 | the same three arms on the laptop first, to check the cell and (iii)'s exactness by node comparison | host | minutes after (iii) is written | any differing node → not exact |
| E-T20 | `ebony_head_outliers.py` on DeepSeek's dump (after E-T10), and on wikitext rows from `ebony_hmax.py`'s hook for all three | host | minutes | on DeepSeek, 16 dimensions leave more than +0.002 → the repair does not generalise |

**Try**

| # | lever | after | label | price | P |
|---|---|---|---|---|---|
| E-Y4 | key-major attention walk, with one value conversion per key-value head | E-T19, E-T18 | bit-identical | Yi −0.5 to −1.0 ms per token from the grouping alone; all three −0 to −5 ms at 200 positions from the order, unknown until E-T18 | 0.35 that E-T18 reads above 1.15 |
| E-Y5 | float columns for the head's outlier dimensions | only if the head knob loses on the phone | numerics change towards the file; a calibration sidecar | Yi and DeepSeek: most of the head's stock shift back for 0.1 to 0.16 ms | 0.2 that it is needed |

## 5. Next questions, ranked

| # | question | state |
|---|---|---|
| 1 | Tom's KleidiAI stock result (flags, E-T13, E-T16) against the bands: Yi +0.3 to +1.2 %, DeepSeek +0.3 to +1.2 %, Llama-2 ±0.05 % | waiting for his file |
| 2 | Does today's attention walk wait for memory on an A78? | one search done (§6): nothing specific to the A78; the general rule (no prefetch across 4 KB pages for physical-address prefetchers) supports the premise. E-T18 decides |
| 3 | What does the proxy's selection effect do to the guess-head scores of r7 §3? | host, minutes: score against tokens chosen by the float head instead of the dumped ones |
| 4 | The same walk in the prefill: 186 queries per head over up to 186 rows, per-query path. Does key-major order with several queries per row (a tile in exact per-query order) cut the prefill's 0.5 s? | *source*; small |
| 5 | DeepSeek's dump (E-T10, E-T20); R75 free reads | Tom; R75 |

## 6. Sources (2026-11-24)

Repository: `lab/llama.cpp-base/ggml/src/ggml-cpu/ops.cpp` (lines 8985 to 9156);
`lab/llama.cpp-base/src/llama-graph.cpp` (lines 2642 to 2661); GGUF headers and `output_norm.weight` of the three
served files; `lab/c3/head/{yi,llama2}/eval.{rows,h16}`; Bonnie's r15 to r17 (times per step), r19, r20.
External, one search on 2026-11-24 (**search summary only**, nothing opened): no result gives the Cortex-A78's
prefetcher behaviour across pages. The nearest general statement is from a MICRO 2022 paper (Vavouliotis et al.,
<https://gvavou5.github.io/Documents/Vavouliotis_MICRO22.pdf>): a prefetcher that works on physical addresses is
generally barred from prefetching across 4 KB pages. Arm's errata notice for the A78 mentions a prefetch request
that crosses a page boundary after an unaligned load (erratum 1488614), which says such requests can exist but
not how strides are handled. So §1.3 (b)'s premise is plausible and unproven; E-T18 decides it.
