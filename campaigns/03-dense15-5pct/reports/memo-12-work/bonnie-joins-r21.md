# c3 memo-12 work, round 21 (Bonnie): choice C as one design note for Tom; what else in a context is anonymous and large

- **Date:** 2026-11-24. **Author:** bonnie, `claude-fable-5-2`. Follows [bonnie-joins-r20.md](bonnie-joins-r20.md). Not
  committed. No verdict here. **Ebony has not reviewed this file.**
- **Labels.** *device*, *host*, *estimated*, *source* as before. Triples are Yi / Llama-2 / DeepSeek.
- Nothing was measured or built. §1 is a design from source read in rounds 7, 8, 12, 19 and 20; §2 is a new
  source pass through the app library's decode loop (`lab/app-c3/third_party/llamadart`, 0.15.6) and the
  runtime's context (`lab/llama.cpp-c3r32`). No web source.
- **Status of §1.** A design note, not a patch and not a decision. It needs the owner's answer on storage
  (questions (a), (b)) and on the A / B / C choice before anyone builds it. Names of knobs, files and lines are
  proposals.

## 0. Answer first

1. **Choice C is four small pieces** (§1): one shared helper that gives a buffer type a mapped file instead of
   anonymous memory; two call sites (the KleidiAI type for the body, the repack type for the head); the release
   of the compute buffer after a prompt. About 300 lines in all (*estimated*), every piece off unless its knob is
   set, none changing a value.
2. **Build order:** the helper and the KleidiAI call site first (that alone is choice C for a CPU-only decode);
   the repack call site second (needs the head knob decision); the compute-buffer release third (independent,
   helps every choice); the app's rule last.
3. **The identity checks are listed as a matrix** (§1.6): knob off, first run (file built), second run (file
   mapped), per chunk at the three batch settings on two subjects on the KleidiAI build; a byte comparison of two
   independently built files; a kill during the build; a wrong key; and the poison arm for the compute buffer.
4. **Nothing else in a context is anonymous and large** (§2). The rest adds up to about 20 / 15 / 40 MiB
   (*estimated*): tokenizer tables, graph bookkeeping, one row of logits, the sampler's candidate array. All of
   it sits inside campaign 2's measured app remainder (131 to 147 MB, *device*). Two things to keep out of a
   claim run are named: all-position logits (only with a speculative session) and the chat path's state save.

## 1. Choice C: design note

### 1.1 What it is

| part | today | under C |
|---|---|---|
| Q4_0 body | KleidiAI's layout in anonymous memory, packed at every load; the app refuses it for a file over 2.7 GB | the same layout in a file under the app's files directory, built at the first load, mapped afterwards |
| Q6_K head | int8 inside the same anonymous buffer | with `VOLLIE_HEAD2_REPACK=1`: the repack type's 8-row layout, in a second mapped file |
| compute buffer | reserved for 512 positions, touched by the prompt, then idle | pages given back after the prompt |
| KV cache | Q8_0; campaign 2's knob drops the untouched part | unchanged |
| numerics | – | **none of the four changes a value**; the head knob does (towards the freeze), and is a separate decision (r12) |

Sizes (*estimated*; r8, r12, r19): body file 2,970 / 3,474 / 3,257 MiB; head file 205 / 103 / 328 MiB; anonymous
during a decode about 0.16 / 0.20 / 0.20 GB. With the head left in KleidiAI there is one file of 3,221 / 3,599 /
3,658 MiB.

### 1.2 Piece 1: the shared helper (new file in `ggml/src/ggml-cpu/`, about 150 lines)

**Interface** (C, called by a buffer type's `alloc_buffer` and `set_tensor`):

| function | does |
|---|---|
| `vollie_pack_alloc(buft, buft_name, variant, size)` → buffer or null | knob off or any failure: null, and the caller allocates as today. Else opens or creates `<dir>/<key>.<buft_name>.pack`, maps it, and returns a CPU buffer on the mapped payload with a free hook that unmaps |
| `vollie_pack_is_mapped(buffer)` → bool | true when the file was found complete and valid: `set_tensor` must do nothing |
| `vollie_pack_note(buffer, tensor)` | first run only: records the tensor's name, offset and size in the table |
| `vollie_pack_finalize()` | first run only, called once after the loader has set every tensor: `msync`, checksum, mark complete, rename, re-map read-only (and executable if asked) |

**Knobs** (read at load only):

| variable | meaning |
|---|---|
| `VOLLIE_PACK_DIR` | directory for the files; unset = everything off |
| `VOLLIE_PACK_KEY` | the served file's sha256 in hex, given by the app or the tool (the runtime does not hash 4 GB itself). Missing with the directory set = off, with a line saying why |
| `VOLLIE_PACK_EXEC` | `1`: map `PROT_READ \| PROT_EXEC` (campaign 2's 0011 for the model file); a refusal falls back to read-only and says so |
| `VOLLIE_PACK_VERIFY` | `1`: recompute the payload checksum at every load (tests only; it reads the whole file) |

**File layout:**

| region | content |
|---|---|
| header, 64 KiB | magic and version; the key; the buffer type's name; the **variant** string; payload size; tensor count; payload checksum; a "complete" flag written last |
| tensor table (inside the 64 KiB; 291 tensors need about 25 KiB) | per tensor: name, offset in the payload, size |
| payload | exactly the bytes the buffer type would have held, starting on a 64 KiB boundary (so it is page-aligned for 4 and 16 KiB pages) |

**The variant string** is what makes a stale file impossible to mis-read. For the KleidiAI type: the library
version (`v1.31.0`), the selected Q4_0 and 8-bit kernel names, `nr`, `kr`, block length, and the header struct's
version. For the repack type: the layout chosen (`q6_K_8x4` on this phone). A different phone, build or
selection gives a different string, and the file is rebuilt.

**Validity at a later load:** magic, version, complete flag, key, buffer type, variant and payload size must all
match what the allocator asks for now; and in `set_tensor` the tensor's name, offset and size must match its
table entry. Any mismatch: print the reason, unlink the file, and run this load as a first run.

**First run:** create under a temporary name, size it, map read-write shared, let `set_tensor` pack into the
map as it packs into memory today, `finalize`. After each tensor, release the served file's source pages
(campaign 1's `GGML_REPACK_RELEASE` does this) so that two copies of the body are never resident together.

**A killed first run** leaves only the temporary file, which the next load deletes.

**One line per buffer at load** (stderr, as the kit's other lines):
`c3-pack buft= state=<off|built|mapped|rebuilt> why= file_mb= tensors= pack_s= pack_gbs= write_s= map=<r|x> verify=<0|1>`.

### 1.3 Pieces 2 and 3: the two call sites

| | KleidiAI type (body) | repack type (head) |
|---|---|---|
| `alloc_buffer` | `kleidiai.cpp:1705`: try `vollie_pack_alloc` first; else the present line | `repack.cpp:5170`: the same |
| `set_tensor` | `kleidiai.cpp:1687`: if mapped, check the table entry and return; else pack as now, then `vollie_pack_note` | `repack.cpp:5151`: the same |
| variant | KleidiAI version, kernel names, `nr`, `kr`, block length | the repack layout's name |
| what lands in the buffer | every Q4_0 matmul weight; the head too unless the head knob is on | with `VOLLIE_HEAD2_REPACK=1`: the head only |
| lines | about 25 | about 25 |
| check specific to it | the packed header inside each tensor (`kleidiai_weight_header`) is read from a read-only map: confirm no code writes to it after load | the row count must be a multiple of 8 (64,000 / 32,000 / 102,400 are) or the repack type declines and nothing is mapped |

Plus one call to `vollie_pack_finalize()` in the loader after all tensors are set (`llama-model.cpp`, after the
buffers are filled; one guarded line).

### 1.4 Piece 4: the compute buffer (r20 §1.3, about 60 lines)

`VOLLIE_CB_DONTNEED=1`: at the end of a decode call that carried a prompt, give the compute buffers' pages back
and drop the remembered graph. `=2`: poison instead (tests). Independent of the pack files; useful under choices
A and B as well.

### 1.5 The app (last; a few lines in `lib/device.dart` and `lib/assistant.dart`)

- Set `VOLLIE_PACK_DIR` (the files directory, which Android keeps until uninstall), `VOLLIE_PACK_KEY` (the sha
  the app already checks at download), `VOLLIE_PACK_EXEC=1`, and the head knob if decided, before the load.
- Replace the memory test for the faster layouts by a storage test when the pack is on: free storage ≥ the
  file sizes plus a margin; the directory accepts an executable map (campaign 2's start-of-session check).
  `canCopyWeights` keeps its present meaning for the anonymous path.
- The first load is longer by 8 to 18 s (*estimated*, r9); the app shows its normal loading state.
- In a kit: a factor "pack" with the `c3-pack` line, `RssFile` / `RssAnon`, major faults and the load time in
  the report, as campaign 2's `exec` and `kv` factors were reported.

### 1.6 Identity checks (host; KleidiAI build of the kit's tree, `GGML_KLEIDIAI_SME=0`)

| # | check | arms | pass |
|---|---|---|---|
| 1 | per-chunk perplexity, 4 chunks, at `-ub 1`, `-ub 32`, `-b 512`; Yi and Llama-2 | (a) knobs unset; (b) pack on, first run (file built); (c) pack on, second run (file mapped); (d) as (c) with the executable map | every per-chunk value of (b), (c), (d) equals (a) |
| 2 | the same with the head knob and the second file | (a') head knob on, pack off; (b'), (c') pack on | (b'), (c') equal (a') |
| 3 | the file's bytes | build the body file twice from scratch; `cmp` | identical; and the payload equals a dump of the anonymous buffer of arm (a) |
| 4 | stale or wrong file | a key of another model; a variant string edited; a truncated file; one payload byte flipped (with `VOLLIE_PACK_VERIFY=1`) | each is refused with its reason, the file is rebuilt, values equal (a) |
| 5 | a killed build | `SIGKILL` during the first run | no file under the final name; the next run builds and equals (a) |
| 6 | read-only map | run (c) under a debugger watch or with the map `PROT_READ` only (it is) | no fault: nothing writes to the weights after load |
| 7 | compute buffer release | `VOLLIE_CB_DONTNEED=1` against unset, at the three batch settings, and with the trigger threshold at one token | equal per chunk |
| 8 | compute buffer **poison** | `VOLLIE_CB_DONTNEED=2` (NaN pattern instead of release), the same settings | equal per chunk. A difference means some op reads memory it did not write: a finding about stock, to be reported before anything else |
| 9 | greedy text, 32-prompt set, all knobs on against unset | – | identical text |
| 10 | load time and resident memory on the laptop (relative only) | first run, second run, unset | reported; the second run must not read the served file's body (check with the file system's read counters) |

Checks 1, 3 and 8 are the ones I would not ship without.

### 1.7 Order and size

| step | pieces | lines (*estimated*) | gives |
|---|---|---|---|
| 1 | helper + KleidiAI call site + `finalize` call | about 190 | choice C for a CPU-only decode, head in KleidiAI |
| 2 | checks 1, 3, 4, 5, 6 | tests | – |
| 3 | repack call site | about 25 | the head in its own file (with the knob) |
| 4 | check 2 | tests | – |
| 5 | compute buffer release + checks 7, 8 | about 60 | less idle anonymous memory under every choice |
| 6 | the app's rule and the kit factor | about 30 in Dart | a device round |

Before step 6: the residency round of r18 should have said that a mapped body of this size stays resident;
otherwise the code works and the phone still cannot hold it.

### 1.8 What the lead or the owner has to settle first

| question | options | my reading |
|---|---|---|
| storage of 2.2 to 3.8 GB more per model | yes / no | without it the fast path cannot ship for a 4 GB file (r17) |
| where the files live | files directory (kept) / cache directory (Android may clear it; then a rebuild) | files |
| verification at each load | header only / full checksum | header only; the rename makes a complete file atomic, and a full read costs 2 to 4 s per load |
| the head | in KleidiAI (one file) / repack type (two files, the head knob) | waits on Tom's H1 and the stock run |
| who gives the key | the app's stored sha / the runtime hashes the file | the app; hashing 4 GB at every load would cost more than the pack saves |

## 2. What else in a context is anonymous and large? (next question 3)

*Source:* `llamadart/lib/src/backends/llama_cpp/llama_cpp_service.dart` (the generate loop and its helpers),
`lab/app-c3/lib/assistant.dart`, `lab/llama.cpp-c3r32/src/llama-context.cpp` and `llama-sampler.cpp`.

| item | size (Yi / Llama-2 / DeepSeek) | when | basis |
|---|---|---|---|
| logits the context keeps | one row: 0.26 / 0.13 / 0.41 MB | every token | the library marks only the prompt's last token for output (`service.dart:6605`); the load logs print 0.12 MiB for Llama-2 (*host*) |
| … if every position were output | 190 rows: 49 / 24 / 78 MB | only with a speculative session (`outputAllLogits: speculativeSession != nullptr`, lines 6385 to 6468) | **keep speculative sessions out of a claim run**; the kit's rollback setting is 0 |
| sampler candidate array | 12 bytes per vocabulary entry: 0.8 / 0.4 / 1.2 MB, allocated per sampled token | every token unless bypassed | `llama-sampler.cpp:915–941`; `VOLLIE_SAMPLER_BYPASS=1` is in campaign 2's kit list |
| repetition-penalty history, grammar, stop buffers | kilobytes | – | `service.dart:6727–6739` |
| token and piece buffers | `nCtx` × 4 bytes and 256 bytes | per call | `service.dart:5060, 5274` |
| the batch | arrays for 2,048 tokens: under 0.1 MB | once | `llama_batch_init` with the logical batch |
| tokenizer tables (token text, scores, the text-to-id map; merge ranks for a BPE vocabulary) | about 10 / 5 / 25 MB | at load, then read per token for one entry | *estimated*: 100 to 250 bytes per entry; 64,000 / 32,000 / 102,400 entries |
| graph bookkeeping (tensor headers for up to 8 × the tensor count, the scheduler's tables) | about 5 MB | once | *estimated* from the context's graph size rule |
| thread-pool scratch for KleidiAI's activation packing | under 3 MB at a 190-token prompt, kilobytes at one token | grows to the largest batch | *estimated*: 190 rows × (11,008 + scales) bytes |
| chat state save after a reply | the cache in use, written to a file: 7 / 58 / 55 MB at 220 positions, through a temporary buffer | the chat path only (`assistant.dart:1599, 1630–1650`); not the benchmark's `_complete` path (line 555) | **a claim run through the chat path would pay this after every reply**, in memory and in time during the gate wait |
| a second context | none | – | no decision or embedding context is created by the app (no use in `lib/`) |
| **sum of the standing items** | **about 20 / 15 / 40 MiB** | | all inside campaign 2's app remainder of 131 to 147 MB (*device*) |

Reading:
- **Nothing here needs a change.** r19's hot budget (about 160 / 200 / 200 MiB) already contains it through the
  app remainder; DeepSeek's larger vocabulary may add 10 to 20 MiB to that remainder, which was measured with a
  50,000-entry vocabulary.
- Two conditions for a claim run follow, both already true of campaign 2's kit as far as I can read it: the
  benchmark goes through the plain completion path (no state save), and no speculative session is open.
- The sampler bypass matters a little more here than in campaign 2: without it each token allocates and fills
  1.2 MB on DeepSeek.

## 3. Next questions, ranked

1. **Fold R75 into r18's draft** when its report is in the repository (I have not seen it yet), with r19's idle
   part and r20's two changes.
2. **A "claim-run configuration" sheet for campaign 3:** one page that fixes path (plain completion), prompt,
   context size, cache type, threads, knobs and the three decisions still open (A / B / C, the head, the KV
   type), each with its source line, so that R73's pre-registration can cite it. r16's table is the raw
   material. Mine.
3. **Does the pack file help the load itself on the phone?** Today a load reads and packs 3.5 to 4.0 GB every
   time; under C it maps. Campaign 2's reports have load times for OLMoE with and without the copies; read them
   for a per-GB figure. Mine.
4. **Re-read r15 to r17 against Tom's host decode** when it exists.
5. **DeepSeek's head; cold read rate; fixed share.**

## 4. Limits

- §1 is a design. Line counts are estimates; the helper's interface is a proposal; I have not checked how the
  loader orders `alloc_buffer`, `set_tensor` and any later write for every tensor kind (check 6 is there for
  that).
- The variant string assumes KleidiAI's packed layout depends only on the things listed; a future library
  version could add another. The version is in the string for that reason.
- §2's sizes for tokenizer tables and graph bookkeeping are estimates, not readings of a heap profile.
- I read the generate loop for allocations that scale with vocabulary or context; I did not read every helper
  in a 6,000-line file.

## 5. Sources (2026-11-24)

Repository: `lab/llama.cpp-c3r32/ggml/src/ggml-cpu/kleidiai/kleidiai.cpp` (lines 518 to 570, 1687 to 1718),
`repack.cpp` (lines 5144 to 5250), `src/llama-model.cpp` (lines 1888 to 1945), `src/llama-context.cpp` (lines 1672,
2060, 2234 to 2235), `src/llama-sampler.cpp` (lines 915 to 941); `lab/app-c3/third_party/llamadart/lib/src/backends/llama_cpp/llama_cpp_service.dart`
(lines 5060, 5274 to 5350, 6385 to 6468, 6605, 6727 to 6739); `lab/app-c3/lib/assistant.dart` (lines 555 to 560, 1565
to 1650, 1710), `lib/device.dart` (line 591); `patches/llama.cpp-c1-olmoe/README.md` (0008, 0009, 0011);
`docs/runtime-knobs.md` (lines 38, 62); campaign 2 `reports/memo-24-anon-memory-and-rest.md`; my r7, r8, r9, r12,
r17, r19, r20.
