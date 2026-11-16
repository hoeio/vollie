# Pre-registration R64: clear the KV and recurrent buffers with MADV_DONTNEED instead of memset, on Granite and OLMoE (kit r25)

Committed and pushed **before** the APK is handed to the owner. Do not edit afterwards; changes go into
`R64-amendment-1.md`. No tag (owner's wish); the commit hash and push time show the order.

- **Date / author:** 2026-11-16, clyde (lead, main session).
- **Campaign / models / files:** campaign 2. `granite-Mi.gguf` `8df94de7…0bc0f45`, `olmoe-Mi.gguf`
  `5abf710b…8e444abb`, on the phone, hash checked by the kit.
- **Agent models** (full IDs): clyde `claude-opus-5-5` (lead, from this round on, owner 2026-11-16); tom
  `claude-opus-5-5` (patch 0008, kit r25); bonnie wrote memo-24, on which this round rests, in a session
  whose agent files were loaded before the switch to `claude-fable-5-2` (so most likely `claude-opus-5-5`, not
  verified); annie `claude-sonnet-5-5` not used.
- **Build under test:** `vollie-c2-1.25.0+25-research-r25.apk`, 17,229,319 bytes, sha256
  `1bbbec0ca533a0529305d99711bd1e1ac9664af1727586e62b56fed8e4c180d4` (build record `lab/dist-c2/BUILD-r25.txt`).
  Runtime: llama.cpp `620d2225c` + `patches/llama.cpp/0001` to `0003` + `patches/llama.cpp-c1-olmoe/0001` to
  `0008`, KleidiAI on. `libllama.so` `17a197d4…6cebb`, CPU module `d6b7b3c9…c1f731c`. The CPU module differs
  from r24's because 0007 (routing prior) is compiled in; it is never switched on, and the kit clears
  `VOLLIE_PRIOR`, `VOLLIE_PRIOR_N`, `VOLLIE_PRIOR_J` before every run. App patches `patches/app-c2/0001` to
  `0009`. Kit: [memo-26](../reports/memo-26-kit-r25.md).
- **Device:** the owner's Galaxy Z Flip 3. **Labels:** every number in §3 is *estimated* (memo-24's fits over
  R60 and R62, *device*) until a result says *device*.
- **Starts from campaign 1 / R63:** kit r24's protocol (memo-23) unchanged, with one factor added. R62 and, if
  run before this, R63 are the comparison rounds.
- **Phone state asked of the owner:** as R63: background apps closed, charger out, phone on the desk.

## 1. What is tested

**kvclear** (declared bit-identical; protocol-free runtime change): `VOLLIE_KV_DONTNEED=1` set against unset. When the runtime
clears a CPU buffer to zero (the KV cache, and Granite's recurrent state, before every prompt because kit runs
set `reusePrefix = false`), the page-aligned interior is released with `madvise(MADV_DONTNEED)` and only the
edges are memset. Released pages read back as zeros, so every value the model sees is unchanged.

Reason (memo-24 §2): OLMoE's KV buffer is 285 MB (69,632 pages); its zero pages are swapped out during the gate
wait and swapped back in by the memset, up to about 70k faults per run; TTFT minus prefill = 41.6 ms + 3.47 µs
per fault (*device*, r = 0.99, n = 48).

Arms and order: kit r24's design unchanged (blocks G O O G, two warm-ups, two ABBA
quads per block with plateau and r16 gates, two profiled runs, three threads, hold, charger guard; 48 runs, 40
gated). The kv factor alternates per scored run: odd blocks on off on off in quad 1 and off on off on in quad
2, even blocks the reverse; one kv and one plain run per arm × gate cell; 8 kv and 8 plain scored runs per
model. Warm-ups and profiled runs never set it. Run ids carry `_kv`; report field `kv_rule`. Arms: `off`;
OLMoE `num` = `VOLLIE_SKIP_MASS=900 VOLLIE_HEAD2=1024`, load `VOLLIE_HEAD2_REPACK=1`; Granite `num` =
`GGML_EXPERT_SKIP=130 GGML_HEAD2=2048 GGML_HEAD2_Q=q4_0`. The kv comparison is paired within arm × gate cell
(kv minus plain); R63's gate and num factors are scored as written in R63 on the same runs, with kv on and off
pooled, and reported also on plain runs only.

## 2. Budget accounting

No numerics change. Host identity (patch 0008, knob on against off): per-chunk PPL equal for olmoe-Mi and granite-Mi at `-b 512` and
`-b 512 -ub 32` (4 chunks) and `-b 512 -ub 1` (2 chunks), on macOS (fallback memset path) and in an Ubuntu
24.04 aarch64 container (real `madvise` path, `why=ok` on every clear); greedy text equal over three
clear-and-generate runs per process at the app's settings (ctx 4096, q8_0 K/V), knob on and off, both models,
both systems; a Linux unit test of the exported clear zeroes every byte (*host*, memo-26). Not verified on
Android (scudo, kernel 5.4.274, zram).. The num settings are R63's
and their budget lines stand (R61, *host*).

## 3. Predictions (*estimated*, kvclear on minus off, per model, Welch SE)

| quantity | OLMoE | Granite |
|---|---|---|
| bytes released per clear (kit line) | about 285 MB (≥ 250) | KV 17.8 + recurrent 58.1, about 76 MB (≥ 50) |
| major faults before the prompt graph, median | off 20k–50k, on ≤ 2k | off ≤ 21k, on ≤ 2k |
| TTFT | **−0.15 s (−0.10 to −0.25)** | **−0.05 s (−0.02 to −0.12)** |
| decode ms/token | −0.2 (0 to −2) | 0 (+0.5 to −0.5) |
| tok/s (32-token metric) | +0.1 (−0.5 to +1.0) | 0 (−0.5 to +0.5) |
| `VmSwap` at run end | −100 MB (0 to −270) | −20 MB (0 to −70) |

Generalisation prediction: the fault and TTFT effect appears on both models, scaled by buffer size (OLMoE
about four times Granite's); the tok/s metric moves on neither within its SE.

## 4. Pass, kill and void

- **kvclear, per model.** Pass: pre-prompt faults lower with on (median on below half of median off) and TTFT
  lower by more than 2 × SE. **Kill:** TTFT higher by more than 2 × SE, or any text differs between arms (then
  the identity claim is wrong; the knob is closed until explained). Otherwise "unresolved". The tok/s metric is
  reported but is not the pass criterion: the lever is priced at about 0 on it.
- **Identity on the phone:** the 32 greedy tokens are the same in both arms of the same num setting, every run.
- **Void (kit, not lever):** a model or library hash differs from this document; the released-bytes counter is
  zero in any on run or nonzero in any off run; the rules of R63 §4 for charger and hashes.
- Misses are written as misses.

## 5. Claim rule

Not a goal claim.

## 6. Hygiene

As R63, plus per run: kvclear arm, released bytes, pre-prompt faults, TTFT.

## 7. What a result means

- Pass on both: the knob is kept on in every later kit (it costs nothing on the metric and frees about 270 MB
  of swap slots on OLMoE); memo-24 item 3 (Granite tied head out of KleidiAI) is next on memory.
- Faults drop but TTFT does not: the TTFT fit of memo-24 is wrong; the fault cost is looked for again.
- Faults do not drop: the clear path or the allocator on Android is not what patch 0008 assumes; Tom checks the
  buffer's mapping on the phone before any further memory lever.
