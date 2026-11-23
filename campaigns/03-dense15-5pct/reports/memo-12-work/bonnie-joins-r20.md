# c3 memo-12 work, round 20 (Bonnie): can the compute buffer be released after the prefill; a mapped file for a repack-type head

- **Date:** 2026-11-24. **Author:** bonnie, `claude-fable-5-2`. Follows [bonnie-joins-r19.md](bonnie-joins-r19.md). Not
  committed. No verdict here. **Ebony has not reviewed this file.**
- **Labels.** *device*, *host*, *estimated*, *source* as before. Triples are Yi / Llama-2 / DeepSeek.
- Nothing was measured or built this round. It is source reading in the kit's tree (`lab/llama.cpp-c3r32`):
  `src/llama-context.cpp`, `src/llama-graph.cpp`, `ggml/src/ggml-alloc.c`, `ggml/src/ggml-cpu/repack.cpp`. Sizes are
  r19's (*host* load logs). No web source.

## 0. Answer first

1. **Yes: by the code's structure the compute buffer holds nothing that one graph leaves for the next.** Every
   decode call builds or reuses a graph, writes all its inputs again, and recomputes every node; results leave
   the buffer by copy before the call returns. So giving its pages back to the system between calls changes no
   value. **Up to 166 / 105 / 241 MiB of idle anonymous memory go away after the prompt** (*host* sizes; the
   part a 150-to-190-token prompt really touches is smaller and not known).
2. **It is a declared-identical runtime change of about 60 lines, off unless a knob is set.** The check that
   matters is not only per-chunk perplexity: released pages come back as zeros, so an op that read memory it
   had not written would now read zeros instead of stale values. A poison arm (fill the range with a pattern
   instead of releasing it) finds such an op; §1.4.
3. **Its cost is paid at the next prompt, not in the decode:** the buffer's pages fault in again, about 0.05 to
   0.2 s per prompt (*estimated* from campaign 2's fault cost). The benchmark's metric is the decode rate.
4. **A repack-type head can be mapped from a file by the same mechanism as the body.** The repack buffer type
   allocates and fills exactly as the KleidiAI type does: one allocation from the plain CPU buffer type, then
   `set_tensor` re-orders the served blocks into it. The Q6_K re-ordering changes no byte's value and no size
   (205 / 103 / 328 MiB, the served head's size). **About 40 more lines if the mapped allocation is written once
   and shared by both types** (*estimated*).
5. Together these take choice C's anonymous memory during a decode to the hot 0.16 / 0.20 / 0.20 GB of r19, with
   nothing large left idle and the head off the anonymous side whichever kernel it uses.

## 1. Releasing the compute buffer (next question 2)

### 1.1 What lives in it (*source*)

- `llama_context::process_ubatch` (`llama-context.cpp` lines 1508 to 1575): for each micro-batch it either reuses
  the previous graph (same parameters: `can_reuse`) or resets the scheduler, builds a new graph and calls
  `ggml_backend_sched_alloc_graph`. **In both cases it then calls `res->set_inputs(&ubatch)`**, then computes.
- `llm_graph_result::set_inputs` (`llama-graph.cpp` line 1414) calls every input object's `set_input`. For a
  dense decoder those are the token ids, positions, output indices and the attention's indices and mask. I read
  `out_ids` and the KV attention's: each writes its whole tensor on every call; there is no "unchanged, skip"
  path. `can_reuse` only decides whether the graph's shape still fits.
- `ggml_gallocr_alloc_graph` (`ggml-alloc.c` line 1052) assigns each leaf and node its address inside the
  buffer from the stored plan. It writes no tensor data and clears nothing.
- A graph's nodes are all recomputed by a compute; a node reads only its sources, which are earlier nodes,
  inputs just written, or tensors in other buffers (weights, the KV cache).
- After the compute, `decode` copies the logits (and embeddings, if asked) out into the context's own output
  buffer (0.12 MiB in the load logs). Nothing later reads them from the compute buffer.
- State that does persist is elsewhere: the KV cache has its own buffer; campaign 2's head copies are their own
  allocations.

**So no tensor in the compute buffer is read by a later graph before that graph writes it.** Between two decode
calls its contents are dead.

### 1.2 Where the idle memory is

The buffer is reserved once for the largest graph (micro-batch 512: 165.52 / 105.02 / 240.52 MiB, *host*). The
prompt's graph (150 to 190 positions) touches part of it; a one-token decode graph is planned into 0.21 MiB
(*host*, Llama-2 at `-ub 1`). After the prompt the rest is never written again until the next prompt, but it
stays resident or goes to zram.

### 1.3 The change

| piece | what | where | size |
|---|---|---|---|
| accessor | walk the graph allocator's buffers and return each host buffer's base and size (the struct is private to `ggml-alloc.c`) | `ggml-alloc.c`, `ggml-backend.cpp` (pass-through for the scheduler) | about 25 lines |
| release | `madvise(base, size, MADV_DONTNEED)` on the page-aligned interior of each CPU compute buffer; the address untagged first, as campaign 2's 0009 does for the KV cache | a small helper beside campaign 2's KV code | about 15 lines |
| trigger | at the end of a `decode` call whose batch had more than a few tokens (a prompt), when no compute is running; also drop the remembered graph (`gf_res_prev_active = nullptr`) so the next call rebuilds, which removes any dependence on input-reuse details | `llama-context.cpp` | about 20 lines |
| knob | `VOLLIE_CB_DONTNEED=1`; unset = today's behaviour. `=2`: poison instead of release (§1.4) | read per call | – |

- It applies to every model and both choices A and C; it has nothing dense-specific in it.
- It releases the whole buffer, not only the tail: the decode graph's 0.21 MiB faults back in at the first
  token (about 54 pages, once).
- Forcing a rebuild after a prompt costs nothing new: the first decode token after a prompt has a different
  shape and rebuilds today.

### 1.4 What has to be shown before it counts as bit-identical

| check | why | cost |
|---|---|---|
| per-chunk perplexity, knob on against unset, at `-ub 1`, `-ub 32`, `-b 512`, two subjects, on the KleidiAI build | the rule for every runtime change. With `-b 512` every chunk is a "prompt", so the buffer is released between chunks: the hardest setting | minutes per setting at 4 chunks; `-ub 1` releases only after batches above the threshold, so add a run with the threshold at 1 |
| the same with **poison** (`=2`): instead of releasing, fill the buffer with a NaN pattern after each call | released pages read as zeros. If some op read memory it had not written (padding, a tail of a packed block), stale values and zeros could both look harmless; a NaN pattern cannot. Equal perplexity under poison is the real proof | the same runs again |
| greedy text on the 32-prompt set, knob on against unset | the decode path at batch 1 after a real prompt | minutes |
| on the phone, when a kit carries it: `rssanon`, `vmswap` and major faults per run with the knob on and off | that the pages really go and the decode does not fault | a kit factor, as campaign 2's `kv` factor was |

- My P that the plain check passes: 0.9. That the poison check passes at once: 0.75. KleidiAI's packing of
  activations and the padded tiles are where I would look first if it fails; a failure there would be a finding
  about stock, not about this change.

### 1.5 Price

| | Yi | Llama-2 | DeepSeek | basis |
|---|---|---|---|---|
| idle anonymous memory removed | up to 166 MiB | up to 105 MiB | up to 241 MiB | r19 §1.1 (*host*); the touched share of a 190-token prompt is unknown: a third to all of it |
| extra time at the next prompt | 0.05–0.15 s | 0.03–0.10 s | 0.07–0.2 s | 13 to 62 thousand page faults at about 3.5 µs (campaign 2 memo-24, *device*, for zero-filled pages of the KV) |
| effect on the decode rate | none direct | none | none | the decode graph's pages are touched once |
| effect on residency | the r18 steps' idle part (0.15 / 0.20 / 0.30 GB, r19 §1.3) shrinks to the untouched KV, which campaign 2's knob already drops | | | – |

It is worth most on DeepSeek, the subject with the least room.

## 2. A mapped file for a repack-type head (next question 3)

### 2.1 How the repack type allocates (*source*, `repack.cpp`)

- `ggml_backend_cpu_repack_buffer_type_alloc_buffer` (line 5170): `ggml_backend_buft_alloc_buffer(ggml_backend_cpu_buffer_type(), size)`,
  then it replaces `init_tensor` and `set_tensor`. The KleidiAI type does the same (`kleidiai.cpp` line 1705).
- `get_alloc_size` is null (line 5248), so a tensor takes `ggml_nbytes`: the served size.
- `set_tensor` (line 5151) calls the tensor's repack routine. For a Q6_K head that is `repack_q6_K_to_q6_K_8_bl`
  (line 3762): eight rows at a time, block by block, into `block_q6_Kx8`. It needs the row count to be a
  multiple of 8 (64,000, 32,000 and 102,400 all are) and copies every field; nothing is re-quantised.
- One context per buffer type (r8 §1.1). With the head knob on, the repack buffer of a dense subject holds the
  head and nothing else: KleidiAI still takes every Q4_0 matmul weight, and the embedding is not a matmul.

### 2.2 What the second file needs

The body's design (r7 §1.2) has four parts: a mapped allocation, a skip in `set_tensor`, a keyed header, and a
safe write. None is specific to KleidiAI.

| part | for the repack-type head | new code |
|---|---|---|
| mapped allocation | the same helper, called from the repack type's `alloc_buffer` when the knob is on: map `<files>/<model sha>.<buffer type>.pack` instead of allocating. `ggml_backend_cpu_buffer_from_ptr` already wraps a host pointer in a CPU buffer; the helper adds a free hook that unmaps | shared with the body; about 10 lines at the call site |
| skip | `set_tensor` returns at once when the file was found valid | about 5 lines |
| key | the served file's sha256, the buffer type's name, the interleave chosen (`q6_K_8x4` on the phone; `8x8` needs i8mm), the buffer size, and per tensor name, offset, size | the same header code; the kernel-variant field differs |
| safe write | temporary name, `msync`, checksum, rename | shared |
| executable map | as campaign 2's 0011 for the model file | shared |

- **About 40 lines beyond the body's change, if the helper is written once for both types** (*estimated*; r7 put
  the body's at about 180).
- The file is 215 / 108 / 344 MB, written once in under a second at R69's write rate.
- Bit-identity: the mapped bytes are the bytes `set_tensor` would have produced. The check is the body's:
  per-chunk values equal with the knob on and off, and equal between a first and a second run.
- The key must include the interleave, because a library built or selected differently (a phone with i8mm)
  packs differently. A mismatch rebuilds; it cannot mis-read.

### 2.3 What it gives

| | head in KleidiAI, in the body's side file | head in the repack type, anonymous | head in the repack type, mapped |
|---|---|---|---|
| anonymous, read every token | 0 | 205 / 103 / 328 MiB | **0** |
| file-backed | + 263 / 131 / 421 MB | 0 | + 215 / 108 / 344 MB |
| budget used by the unmodified runtime (r11, r12; *estimated*) | up to a fifth on Yi | 0 | **0** |
| head time (r12; *estimated*) | 9.04 / 4.40 / 13.88 ms (*device*) | 7.7 / 3.7 / 11.8 | 7.7 / 3.7 / 11.8 |
| code | r7 | r7 + the knob at load | r7 + about 40 lines |

The last column is the configuration I would now write down as "choice C": body in KleidiAI's layout from one
mapped file, head in the repack type's layout from another, cache Q8_0 at n_ctx 512, compute buffer released
after the prompt. Its anonymous memory during a decode is the app, the cache in use and a megabyte of graph:
about 0.16 / 0.20 / 0.20 GB.

## 3. Next questions, ranked

1. **Fold R75 into r18's draft** when its report arrives, with r19's idle part and this round's two changes to
   what the steps stand for.
2. **Write choice C as one design note for Tom** (not a patch): the shared mapped-buffer helper, both call
   sites, the header, the knobs, the identity checks, in the order to build them. r7, r12, r19 and this round
   hold the pieces. Mine.
3. **Does anything else in a context stay anonymous and large?** The context's logits buffer when the app asks
   for all positions, the sampler's candidate array over the vocabulary (102,400 entries on DeepSeek), the
   tokenizer's tables. A pass through the app library's decode loop for allocations that scale with vocabulary
   or context. Mine.
4. **Re-read r15 to r17 against Tom's host decode** when it exists.
5. **DeepSeek's head; cold read rate; fixed share.**

## 4. Limits

- §1.1 is a reading of the dense decode path. I read the input classes a Llama-style decoder uses and not every
  class in the file; a model family with other inputs would need its own reading.
- I did not read the campaign-1 and campaign-2 patches in the kit's tree line by line for code that keeps a
  pointer into the compute buffer across calls. The poison check of §1.4 is what would catch one.
- "Touched share of the prompt's graph" is unknown; the saving could be a third of the table's upper values.
- §2's line counts are estimates for code that does not exist.
- Nothing here was built, run or measured.

## 5. Sources (2026-11-24)

Repository, all in `lab/llama.cpp-c3r32`: `src/llama-context.cpp` (lines 715 to 740, 1508 to 1575),
`src/llama-graph.cpp` (lines 255 to 290, 525 to 563, 1414 to 1418), `ggml/src/ggml-alloc.c` (lines 482 to 496, 1009
to 1100), `ggml/src/ggml-cpu/repack.cpp` (lines 3434, 3762 to 3788, 5144 to 5250), `ggml/src/ggml-cpu/kleidiai/kleidiai.cpp`
(lines 1700 to 1718); `patches/llama.cpp-c1-olmoe/README.md` (0008, 0009, 0011); campaign 2
`reports/memo-24-anon-memory-and-rest.md`; `lab/c3/logs/h3-*-b512-40.log.err` (buffer sizes, through r19); my r7,
r8, r12, r18, r19.
