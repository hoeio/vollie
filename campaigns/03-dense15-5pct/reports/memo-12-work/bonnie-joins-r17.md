# c3 memo-12 work, round 17 (Bonnie): the three ways to load a dense body, priced per subject; the Q8_0 value conversion as the phone ships it

- **Date:** 2026-11-24. **Author:** bonnie, `claude-fable-5-2`. Follows [bonnie-joins-r16.md](bonnie-joins-r16.md). Not
  committed. No verdict here. **Ebony has not reviewed this file.**
- **Labels.** *device*, *host*, *estimated*, *source* as before. Triples are Yi / Llama-2 / DeepSeek.
- **New this round, one reading of a shipped binary:** the kit r35 APK's native libraries, unzipped into the
  session's scratch directory outside the repository and disassembled with the NDK's `llvm-objdump` (28.2). The
  CPU module's sha256 is `46e0735d…9694`, the value in Tom's build note. Nothing was run. §1 is arithmetic on
  earlier *device* and *host* numbers. No web source.

## 0. Answer first

1. **Only one of the three choices is what the phone can do today, and it is the slow one.**

   | per subject (Yi / Llama-2 / DeepSeek) | A: the app's rule (plain mapped rows) | B: kit forces the copies (anonymous KleidiAI) | C: KleidiAI layout from a mapped side file |
   |---|---|---|---|
   | stock speed, tok/s (*estimated*) | 4.3 / 3.9 / 3.9 | 6.90 / 6.21 / 6.22 | 6.90 / 6.21 / 6.22 |
   | memory the body needs | 3.5 / 3.8 / 4.0 GB, file-backed | 3.4 / 3.8 / 3.8 GB, anonymous | 3.4 / 3.8 / 3.8 GB, file-backed |
   | P(stays resident through a claim block) (*estimated*) | 0.85 / 0.65 / 0.55 | 0.45 / 0.15 / 0.15 | 0.85 / 0.65 / 0.60 |
   | width-2 lane possible | no (r6) | yes | yes |
   | change needed | none | the app's memory rule loosened to where campaign 1 was killed | a runtime change (r7) and a new rule keyed on storage |
   | "fix it in the runtime, never in the user's phone" | consistent | **not consistent as it stands** | consistent |

2. **B is the configuration every campaign-3 speed number assumes, and it is the one the evidence is against for
   two subjects.** The phone held 3.4 GB of anonymous memory and not 4.2 (*device*, R72); Llama-2 and DeepSeek
   need about 4.0 to 4.1 GB with the cache and buffers. To admit them the app's bound ("5 GB left") would have to
   fall to 3.7 GB left, which also re-admits the 3.6 GB copy that got the phone killed in campaign 1.
3. **C is the only choice that is both fast and inside the rules.** It costs 3.4 to 3.8 GB more storage per
   model, a first run of 12 to 18 s, about 180 lines of runtime, and it is not yet shown to stay resident at
   3.8 GB (the r8 arm).
4. **The Q8_0 value conversion is vectorised in the shipped library** (*source*, disassembly). It lives in
   `libggml-base.so`, not in the CPU module; per 32 values it does 8 four-wide converts and one software
   half-to-float for the scale. By the A78's issue rules it costs about 86 cycles per 128-value row, **36 to
   45 ns**. That puts the conversion at **30 to 45 % of the attention's per-row work**, the lower half of r16's
   band.
5. **So sharing it at width 2 saves about 0.9 / 1.7 / 1.6 ms per pass** (band 0.75–1.1 / 1.35–2.0 / 1.25–1.9;
   *estimated*). A hand-written version cannot gain much at width 1: the convert instruction is bound to one
   pipe at half rate. An F16 value cache is the way to remove it.

## 1. Row 9, priced

### 1.1 The rule (*source*)

`lib/device.dart:591`: `canCopyWeights(copyBytes) => totalRam − copyBytes >= 5,000,000,000`. `lib/assistant.dart:455`
passes the file's length. With 7.75 GB of RAM (R72's header) the limit is a 2.75 GB file. The comment above the
rule: "an 8 GB SD888 was killed for low memory copying the all-Q4_0 Granite (3.6 GB), not Qwen3.5-4B (2.3 GB)",
and "5 GB left over is fitted to that one phone".

When the rule says no, the library loads with `useExtraBuffers: false`: no KleidiAI, no repack, the weights are
read from the mapped file by ggml's row-at-a-time kernels.

### 1.2 Choice A: the app as shipped

| | Yi | Llama-2 | DeepSeek | basis |
|---|---|---|---|---|
| body, ms per token | 195 (172–219) | 228 (201–255) | 212 (187–237) | R72 body × 1.62 (1.43–1.81): the stock dot product at 2.2 × a gemv (r6; count 2.4, laptop 2.0) |
| head, ms | 15 (13–18) | 7.5 (6.6–8.8) | 24 (21–28) | plain Q6_K rows, 1.5 to 2 × the KleidiAI head; no phone number for this path (R72's plain-buffer types ran 1.4 to 2.1 × Q4_0) |
| **stock tok/s** | **4.3 (3.9–4.9)** | **3.9 (3.5–4.4)** | **3.9 (3.5–4.4)** | 950 ÷ (body + 8 + head) |
| mapped bytes | 3.48 GB | 3.83 GB | 4.00 GB | the served file |
| anonymous bytes | KV 17 / 136 / 128 MiB at n_ctx 512 (1 GiB at 4,096 on the two full-attention subjects), scratch | | | r15 |
| residency | 3.4 GB mapped held in both R69 runs; 4.2 GB held 1 of 2 read-only, 2 of 2 executable (*device*) | | | r8 §2.1 |
| P(resident through a claim block) | 0.85 | 0.65 | 0.55 | judgement between those points; assumes campaign 2's exec map and hold, which work on a mapped file |
| numerics against the freeze | the freeze ran repack; this path was ±0.02 % of it on two other models (*host*, memo-05) | | | needs its own reference: `llama-perplexity -nr` with the KV flags |
| lane | width 1 only: 8.7 / 8.0 / 8.2 at K 300 (r6); no width 2 | | | the stock layout costs 4 × per weight at two positions |

Nothing has to change. It is consistent with every rule. It is also 1.5 to 2 tok/s under the K0 the campaign
has been quoting, and it cannot carry the lever the goal needs.

### 1.3 Choice B: the kit forces the copies

| | Yi | Llama-2 | DeepSeek | basis |
|---|---|---|---|---|
| stock tok/s | 6.90 | 6.21 | 6.22 | R72 (*estimated* from *device*) |
| anonymous: KleidiAI buffer | 3.38 GB | 3.77 GB | 3.84 GB | r8 §1.3 |
| anonymous in all, n_ctx 512 | about 3.45 GB | about 4.0 GB | about 4.05 GB | + KV + scratch |
| against the phone's record | at the edge: 3.4 GB held (*device*, R72 `anon34`) | between 3.4 held and 4.2 not held | the same | a probe with nothing else loaded |
| the load itself | the file (3.5 to 4.0 GB) is read and the copy is built beside it; this is the moment campaign 1's phone was killed at 3.6 GB | | | `device.dart` comment; campaign 1 |
| P(resident through a claim block) | 0.45 | 0.15 | 0.15 | judgement; the hold cannot help, it works on mapped files only (r8 §2.7) |
| what the rule would have to become | "≥ 4.27 GB left" | "≥ 3.92 GB left" | "≥ 3.75 GB left" | 7.75 GB − file length |

- A bound of 3.7 GB admits all three. It also admits a 3.6 GB copy, the case the rule was written to stop.
- Forcing it in a kit (`_repack`) is what campaign 2 did for its small copies and what R72's model-free probe
  stood for. As a claim configuration for a 4 GB file it asks the phone to do what it was killed doing.
- **Not consistent with "fix it in the runtime, never in the user's phone"** unless a device round shows the
  load and ten gated runs survive on each subject, and the app's rule is then changed on that evidence. Even
  then it would hold for this phone state only.

### 1.4 Choice C: the packed side file

| | Yi | Llama-2 | DeepSeek | basis |
|---|---|---|---|---|
| stock tok/s | 6.90 | 6.21 | 6.22 | the same kernels on the same bytes as B |
| mapped: side file | 3.38 GB | 3.77 GB | 3.84 GB | r8; 3.11 / 3.64 / 3.42 GB with the head in the repack type |
| anonymous | KV and scratch; the head if it goes to the repack type (0.2 / 0.1 / 0.3 GB) | | | r12 |
| storage | + 3.4 / 3.8 / 3.8 GB beside the served file | | | owner question (b) |
| first run | 12 to 18 s more, once | | | r9 §4.3 |
| P(resident through a claim block), CPU only | 0.85 | 0.65 | 0.60 | as A: the same kind of memory at nearly the same size; exec map and hold apply |
| with a lane (side file 2.2 / 2.4 / 2.6 GB + 1.15 / 1.35 / 1.25 GB of GPU buffers) | 0.85 | 0.45 | 0.40 | r8 §2.5 |
| runtime change | about 180 lines behind a knob in the KleidiAI buffer type (r7 §1.2) | | | not built |
| app rule | `canCopyWeights` stops being the gate for this path: the copy is no longer app memory. New test: free storage ≥ the side file, and the directory accepts an executable map | | | `device.dart`; campaign 2 R67 |

- Consistent with the rules: the file is derived on the phone from the served file by upstream's own packer
  (COMMON_RULES rule 1), with no user action.
- Not shown: residency at 3.8 GB with four threads reading at 20 GB/s, and beside GPU buffers. That is the arm
  of r8 §2, steps `x5s4` and `x5s2` / `x5s3`.

### 1.5 What I would put to the owner

1. **Campaign 3's stock is not what the shipped app would run.** Under the app's rule the three subjects start
   near 4 tok/s, not 6 to 7.
2. **Which stock the campaign means is a choice of rule, not of measurement:**
   - keep the rule → choice A, and the goal is out of reach by a wide margin on the levers held;
   - keep the rule's purpose and change its mechanism → choice C, at the price of storage and a build;
   - loosen the rule → choice B, against campaign 1's own evidence.
3. The side file was owner questions (a) and (b) already. This round adds a reason they are not optional: without
   it the fast path is not a configuration the app can ship for a 4 GB file.

## 2. The Q8_0 value conversion as shipped (next question 2)

### 2.1 Where it is (*source*)

`ops.cpp` line 8998: `v_to_float = ggml_get_type_traits(v->type)->to_float`. For Q8_0 that is
`dequantize_row_q8_0`, exported by **`libggml-base.so`** (1,092,120 bytes in the r35 APK), not by the CPU module.
So it is compiled once for baseline armv8, without the half-precision instructions.

### 2.2 The code (disassembly of `dequantize_row_q8_0`, offset 0xbd794, 81 instructions)

One loop iteration per block of 32 values:
- the block's half scale is turned into a float **in software** (about 14 scalar instructions: shifts, an OR, two
  float operations and a select), because the baseline build has no half-to-float instruction;
- then eight times: load 4 bytes, `sshll` to 16 bits, `sshll` to 32 bits, `scvtf`, `fmul` by the scale, store.

So it is vectorised, four values at a time, with no loop inside the block.

### 2.3 Cost on an A78 (*estimated*, by the issue rules of `tools/c3_a78_pipes.py`)

| per block of 32 values | slots |
|---|---|
| `scvtf` × 8 (one pipe only, half rate) | 16 on V0 |
| `sshll` × 16 (the other pipe only) | 16 on V1 |
| `fmul` × 8, `dup`, the scalar float steps | about 11 on either |
| floor | max(16, 16, 43 ÷ 2) = **21.5 cycles** |

- A value row of one head is 128 values: 4 blocks, **86 cycles, 36 ns at 2,419 MHz**; with the 1.2 that the
  gemv showed between floor and measurement, 43 ns. Band 36 to 45 ns.
- Beside it per cached row (r16 §2.2, *estimated*): key dot product about 10 ns, `expf` and bookkeeping about
  25 ns, the weighted add about 30 ns. **The conversion is 30 to 45 % of the row.**

### 2.4 What follows

| option | saving | note |
|---|---|---|
| share the converted row between the two queries of a width-2 pass (r16 §2.2) | 30 to 45 % of the second position's attention: **0.9 / 1.7 / 1.6 ms per pass** (0.75–1.1 / 1.35–2.0 / 1.25–1.9) | exact; about 40 lines |
| a hand-written conversion at width 1 | at most 25 %: 8 bytes per load would halve the `sshll` count, but the eight `scvtf` stay on one pipe at half rate, 16 cycles of the 21.5 | not worth a patch by itself |
| build `libggml-base.so` with the half-precision instructions | removes about 14 scalar instructions per block, none of them on the binding pipes | no gain by the count |
| F16 values, Q8_0 keys | the whole conversion, at every width: about 1.0 / 1.8 / 1.7 ms per position | 26 / 26 / 25 MB more bytes per token on the full-attention subjects; a numerics change towards the freeze; the third arm I asked for in Tom's KV screen |

- r16 item 4's "1 to 3 ms per pass" narrows to **0.75 to 2.0 ms**.
- The larger terms of the attention (the add and the `expf` per cached row) are per query and stay. A width-2
  pass on Llama-2 or DeepSeek still pays about 3.5 ms for its second position after sharing.
- None of this is measured. R73, or a dense decode on the host with campaign 1's per-op profile, gives the
  attention's real share; §2.3 says how to split it once it is known.

## 3. Next questions, ranked

1. **The residency arm of r8 is now the campaign's gating measurement,** ahead of everything about the lane: it
   decides whether choice C exists for Llama-2 and DeepSeek. Re-read its steps against §1 (the CPU-only step
   `x5s4` at 3.85 GB matters most) and add a step for choice A's size (4.0 GB mapped, executable, no GPU). Mine,
   as an addendum to the specification.
2. **Choice A measured instead of estimated:** one probe cell (Q4_0 and Q6_K in the plain buffer, hot and cold,
   3 and 4 threads) gives the stock kernel's rate on the phone; it replaces the 2.2 and the 1.5-to-2 guesses of
   §1.2. A kit line; specification mine.
3. **The ladder under choice A and under C,** once the above and R75 are in: the same rungs with the body each
   choice implies.
4. **What "auto" flash attention resolved to in the freeze; DeepSeek's head; cold read rate; fixed share.**

## 4. Limits

- §1's probabilities are judgement between a handful of residency points measured by probes with no model
  loaded. No dense model has been loaded on this phone in campaign 3.
- Choice A's speed rests on a kernel that was never timed on the phone for Q4_0 or for a plain Q6_K head.
- "Killed at 3.6 GB" is campaign 1's record as the app's comment gives it; I did not re-read that round.
- §2.3 is an issue count on disassembly, not a timing; the per-row shares beside it are estimates from operation
  counts.
- I read the disassembly of one function. I did not check which `vec_mad` and `vec_dot` variants the CPU module
  uses in the same loop.

## 5. Sources (2026-11-24)

Repository: `lab/dist-c3/r35/vollie-c3-probe-r35.apk` (`lib/arm64-v8a/libggml-base.so`, `libggml-cpu-android_armv8.2_2.so`;
read in the scratch directory, not copied into the repository); `lab/app-c3/lib/device.dart` (lines 583 to 591),
`lib/assistant.dart` (lines 455 to 466); `lab/llama.cpp-c3r32/ggml/src/ggml-cpu/ops.cpp` (lines 8996 to 9110);
`tools/c3_a78_pipes.py`; `reports/R69-result.md` and R69 JSON (`p1`), `reports/R72-result.md` (lines 186 to 195);
`reports/memo-05-y1-column-kernels.md` (T3); campaign 2 `STATUS.md` line 367 (exec map), `reports/README.md` (R60);
my r6 §1, r7 §1, r8 §1 to §2, r9 §4, r12, r15, r16.
