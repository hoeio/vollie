# Memo 13: why R52 ran at 3.3 tok/s (page cache, heat, a broken gate) and campaign 1's runtime on OLMoE

Agent model: claude-opus-5-5

Date 2026-11-14. Author: Tom (code). Input: the first device report of campaign 2,
`reports/device/r17_2026-11-14.json` (kit r17, `olmoe-Mi.gguf`, stock runtime), read against campaign 1's
`../../../reports/device/r16_2026-11-08.json` (Granite d6-e6, kit r16, the R49 session). Labels: *device* (read from the
JSONs), *host* (Apple M4 Pro, 4 threads, beside a perplexity sweep at 8 threads; relative only), *estimated*
(arithmetic). **Nothing here is a phone speed measurement made by me.** Scratch tools (an expert-trace program linked to
libllama, a working-set and cache simulator, a scratch llama.cpp tree with campaign 1's series) live outside the
repository; they can be added under `tools/` if the lead wants them kept.

## 0. Summary

1. **The official benchmark touches almost the whole expert file in its prefill.** 156 prompt tokens × 8 of 64
   experts × 16 layers select 968 of the 1,024 (layer, expert) slices = **2,617 MB** (*host* trace, page-aligned).
   With the 32 decode tokens the union is 1,008 slices = **2,725 MB**, plus 9 MB of dense pages read through mmap per
   token with repack on. Granite d6-e6 does the same: 2,502 of 2,560 slices in the prefill, union **2,569 MB** + 24 MB.
   The two models need about the same file cache for a warm benchmark run (2.73 versus 2.59 GB, +0.14 GB for OLMoE).
2. **What differs is the phone, not the runtime.** On 2026-11-08 (r16) the phone had 2.24 to 2.98 GB `Cached` and
   swap free; on 2026-11-14 (r17) it had 2.18 to 2.35 GB `Cached`, `SwapFree` 0, and the app held 1.45 to 1.71 GB of
   file pages (*device*). Below about 2.7 GB the official run re-reads 1.5 to 2.1 GB in the prefill and 20 to 130 MB
   per decode token (LRU, *estimated*). The fault counts of r17 fit that with a 256 KB read-around per fault.
   Campaign 1 never made the experts resident by a runtime mechanism. `GGML_EXPERT_IO=1` made the re-reads cheaper
   (R12: 6.1 → 11.8 tok/s; R27/R29: +66 ms/token without it on a thrash day). The fast R49 runs were days with enough
   cache. Kit r17 ran with expert read-ahead off.
3. **OLMoE cannot be made resident on the phone as it was on 2026-11-14.** About 2.73 GB is needed and at most 2.35 GB
   of cache existed for every file on the system. No advice, release or skip rule closes 0.4 to 1.2 GB. Expert
   skip `MASS=900` cuts decode bytes per token by 13 % but leaves the prefill union unchanged (967 slices). What a
   runtime can do is make the re-reads cheaper (read-ahead, larger requests) and protect the decode-hot set from the
   prefill's sweep (§3). On a day like 2026-11-08 the margin is about 0 to 0.25 GB: borderline.
4. **The clocks are thermal.** Headroom was 0.97 to 1.10 for the whole session; the A78 cap stepped down
   1,440 → 1,209 → 1,075 MHz as the headroom rose and never recovered. In r16 the same counters showed 2,112 MHz in runs
   with 35 k to 74 k major faults, so I/O alone did not lower the clocks there.
5. **The cool-down gate is broken in kits r17 and r18** (it reads 0 and null). `heat_after` calls
   `getThermalHeadroom` at the end of each run. The next run's gate calls it again within a second, Android returns
   NaN, the app maps NaN to `null`, and `_coolDown` treats `null` as "cool". The one run that waited had a model reload
   (8.2 s) in between. Kit r16 had no `heat_after` read, so its gate worked.
6. **Template:** llamadart rendered `{{ bos_token }}` as `<s>`, its fallback when the GGUF has no BOS string. That is
   three text tokens and not OLMoE's token. The effect on this benchmark is small: +1 prompt token, and the same expert
   union (§5).
7. **Campaign 1's runtime builds and runs OLMoE as it stands.** With Granite's numerics knobs unset it is bit-identical
   to stock at all six settings (4 chunks), and the official greedy text equals stock with repack on and off (*host*,
   §7).

## 1. Expert working set of the official benchmark (OLMoE `olmoe-Mi`)

**Method.** A 120-line C++ program links libllama (the c2dev build, all knobs unset) and sets `cb_eval` on
`ffn_moe_topk-<il>` and `ffn_moe_weights-<il>`. It logs every selected expert id and its mixture weight per layer and
token. The prompt is the exact string the kit rendered (`template.rendered` in the JSON, 156 tokens, checked). The
prefill is one 156-token batch. The 32 decode graphs were then fed the **device's own reply** (forced tokens) because
the host's greedy text diverges from the phone's at about token 6 (host and device numerics differ, as noted in R49).
Byte positions come from the GGUF's tensor table. One expert's gate, up and down slices are 3 × 901,120 B =
**2.7034 MB** per layer (the Q3_K slices are exact 4 KB multiples). The file's experts total 2,768.2 MB.

| quantity (*host* trace; MB *estimated* from the file layout) | slices | MB |
|---|---|---|
| prefill (156 tokens) | 968 of 1,024 | 2,617 |
| decode, per token | 128 | 346.0 |
| decode, distinct over 32 tokens | 790 | 2,138 |
| union prefill + decode | 1,008 | 2,725 |
| decode slices never seen in the prefill | 40 | 108 |
| dense bytes read through mmap per token, repack on (router F32 + norms; `token_embd` is one row) | | 8.9 |
| repacked sources (attention Q4_0 150.8 + head Q6_K 84.5), read once at load, repack on | | 235.5 |
| dense bytes read through mmap per token, repack off | | 244.4 |

New bytes per decode token that the previous N decode tokens did not touch: N = 1: 80.7 slices (218 MB); 2: 69.3
(187); 4: 49.0 (133); 8: 29.2 (79); 16: 12.3 (33). Slices never touched before in the run (not in the prefill or an
earlier token): 1.25 per token on average, 6/8/3/8 in the first four tokens, 0 to 1 at the end.

**LRU page-cache simulation** (4 KB pages; per run: prefill per layer, then 32 decode tokens; the dense pages are
touched every token; with repack on the 235.5 MB of sources are read once at load and then age out). The cache is the
memory available to *this file's* pages. MB re-read from flash, *estimated*:

| repack | cache | run 1 prefill | runs ≥ 2 prefill | decode tokens 1-8, per token | decode tokens 9-32, per token |
|---|---|---|---|---|---|
| on | 1.2 GB | 2,626 | 2,341 | 161.5 | 110.5 |
| on | 1.5 GB | 2,626 | 2,144 | 126.9 | 76.5 |
| on | 1.8 GB | 2,626 | 1,884 | 100.2 | 38.7 |
| on | 2.1 GB | 2,626 | 1,527 | 68.4 | 20.7 |
| on | 2.4 GB | 2,626 | 1,056 | 42.0 | 12.0 |
| on | 2.8 GB | 2,626 | 0 | 0 | 0 |
| off | 1.5 GB | 2,862 | 2,509 | 156.1 | 100.7 |
| off | 1.8 GB | 2,861 | 2,289 | 121.8 | 68.3 |
| off | 2.1 GB | 2,861 | 2,008 | 93.8 | 36.1 |
| off | 2.4 GB | 2,861 | 1,629 | 64.6 | 18.3 |

Repeated runs of the same prompt reach their steady state at run 2. Each later run is a cyclic sweep over a set larger
than the cache, which is LRU's worst case. Below about 2.73 GB (on) or 2.97 GB (off), **every run re-reads most of the
prefill**. Repack off costs about the same as 0.24 GB less cache.

**Reconciling the device fault counts.** Bytes per steady run (prefill + 8 × early + 24 × late), *estimated*:
cache 1.5 GB: 4,995 MB; 1.8 GB: 3,615 MB; 2.1 GB: 2,571 MB. The device's `stock` runs had 19.6 k to 35.4 k major faults
(median 22.6 k). That is 114 to 221 KB per fault at the median, or 73 to 294 KB over the range. **A 256 KB read-around
fits.** The fault read-around is `ra_pages`, doubled because llama.cpp opens the model with `POSIX_FADV_SEQUENTIAL`
(`src/llama-mmap.cpp:488`, base 620d2225c), so 2 × 128 KB at the common `read_ahead_kb` of 128. Part of each window is
already cached. The app's resident file pages (1.45 to 1.71 GB) point to an effective cache of about 1.6 to 2.0 GB.
`read_ahead_kb` and `/proc/self/io` were not in the report, so 128 KB is an assumption.

**To ms per token** (*estimated*). Faults block the three compute threads (queue depth ≤ 3, 128 to 256 KB requests).
For UFS 3.1 I assume 0.3 to 0.9 GB/s for that pattern and 1.2 to 2.0 GB/s sequential; both are assumptions, not
measurements. Decode I/O at an effective 1.5 to 2.1 GB of cache is then 21 to 77 MB/token late (23 to 257 ms) and 68
to 127 MB/token early (76 to 423 ms). Compute at the session's clocks: memo-08's stock centre is 72 ms/token at normal
clocks; at 1,075 to 1,440 MHz against 2,112 to 2,419 that is about 110 to 160 ms. The sum, about 130 to 420 ms, covers
the *device* gaps (tokens 1-4: 367 to 671 ms; later tokens: 116 to 505 ms, typically 150 to 300 ms). The prefill re-read
of 1.5 to 2.1 GB per run fits a TTFT of 9.1 to 12.1 s together with throttled compute; the two cannot be separated with
r17's counters. **The per-token pattern repeats across runs because the routing is deterministic.** The same 40 to 80
slices miss at the same token in every run, so the gap list repeats.

## 2. The control: Granite d6-e6 (campaign 1's file, R49)

Same method, the official prompt with Granite's template (155 tokens, equal to r16's `prompt_tokens`), decode forced
to the r16 device text. One expert slice is 1.0138 MB (3 × 337,920 B). Experts total 2,595 MB; 24.0 MB of dense
pages are read per token with repack on (412.0 MB of repacked sources).

| quantity (*host* trace) | stock (6 of 64) | skip 117 (R49's rule) |
|---|---|---|
| prefill slices / MB | 2,502 of 2,560 / 2,537 | 2,500 / 2,535 |
| decode per token | 240.0 slices / 243.3 MB | 163.4 / 165.6 MB (k̄ 4.09, matches R25's 4.1 to 4.3) |
| decode distinct over 32 | 1,328 / 1,354 MB | 1,025 / 1,046 MB |
| union | 2,533 / **2,569 MB** | 2,529 / 2,565 MB |
| new per token vs previous 8 decode tokens | 47.9 MB | 37.9 MB |
| LRU repack on, cache 2.1 GB: prefill (runs ≥ 2) / late decode per token | 1,640 MB / 9.2 MB | 1,825 / 6.9 MB |
| LRU repack on, cache 1.5 GB | 1,991 / 16.2 MB | 2,112 / 10.0 MB |
| LRU repack on, cache 2.8 GB | 0 / 0 | 0 / 0 |

**Comparison.** The union is nearly the same: OLMoE needs +0.14 GB. OLMoE's decode re-reads under the same
cache are 2 to 8 times Granite's. It has 16 layers with 2.7 MB slices against 40 layers with 1.0 MB slices, and more
of the decode working set is new per token (OLMoE 2,138 MB distinct over 32 tokens against Granite's 1,354). Read-ahead
and skip help Granite more for the same reason.

**What made Granite resident in R49.** It was the memory state, not a residency mechanism:

- r16's fast runs had `Cached` 2.7 to 2.97 GB and `MemAvailable` 2.8 to 2.96 GB, with `SwapFree` 113 to 1,268 MB.
  The runs with 35 k to 74 k faults followed the reference stories or idle time that evicted pages (R49 correction
  note). r17 never had more than 2.35 GB `Cached`.
- Campaign 1 met the same wall on bad days: R9 (2.1 GB cached, 354 to 1,181 faults/token), R10 (1.85 GB, 632/token),
  R24/R25 ("the bench reads the page cache more than the runtime").
- The runtime made faulting cheaper, not rarer. `GGML_EXPERT_IO` bit 1 (one `MADV_WILLNEED` per routed slice when a
  layer's routing is known) is in `patches/llama.cpp/0001-cpu-runtime.patch` at lines 4115-4126 (description),
  4739-4797 (`ggml_cpu_expert_prefetch`, skip-aware since R26) and 5057 (call). R12 measured 6.1 → 11.8 tok/s; R27 and
  R29 measured +66 ms/token without it on a thrash day. The read-ahead turns blocking faults into async reads, so it
  also cuts the *major* fault count (R27: `eio 0` ×9.6 majflt).
- c2's `0006-expert-prefetch.patch` (`VOLLIE_EXPERT_IO=1`) reproduces bit 1 only. Bits 2, 4 and 8 are not ported.
  **Kit r17 ran with it off**; kit r18's `stack` has it on.

**Open discrepancy.** In r16's fault-free runs the app's `RssFile` stayed at 2,007 to 2,015 MB, with 4 to 33 major
and about 3.6 k to 3.9 k minor faults per run. That allows at most about 2.0 + 0.25 GB of file pages touched, against
2.57 GB in my trace. I could not resolve this from the report. Candidates: an `RssFile` accounting effect on that
kernel, or fewer slices touched on the device than on the host. A `mincore()` count of the model mapping before and
after each run would measure residency directly. It is cheap (one call over 3 GB = 750 k pages) and belongs in the next
kit. If the device does touch less, every "needed" figure here is high by up to about 0.3 GB, and OLMoE on a
2026-11-08 day would be resident, not borderline.

## 3. What could make OLMoE resident, runtime only (file frozen)

Needed for a warm official run: **2.73 GB** of this file's pages (repack on), or 2.97 GB with repack off. Available on
2026-11-14: about 1.6 to 2.0 GB effective, 2.35 GB `Cached` at most for all files.

| lever | MB saved from the needed set | closes the gap? |
|---|---|---|
| `GGML_REPACK_RELEASE` (page out repacked sources) | 0 in steady state: the 235.5 MB are read once at load and age out first (R17/R18: "the kernel had already evicted the source pages → 0 gain") | no; it helps only the first run after load |
| repack off | −235 MB anon, +235 MB hot file pages | no: a wash for residency, and slower compute |
| expert skip `MASS=900` | prefill union 968 → 967; decode distinct 790 → 747 (−116 MB); decode bytes per token 346 → 301.5 MB (−13 %); LRU late decode at 1.5 GB 76.5 → 53.9 MB/token, at 2.1 GB 20.7 → 18.6 | no: it does not shrink the working set, only the per-token reads |
| prefill-only top-k cap (numerics, not budgeted) | top-1 in the prefill still touches 872 slices (2,357 MB); top-2 956 (2,584 MB) | no |
| `GGML_DENSE_WILLNEED` | dense through mmap is only 9 MB/token with repack on | no |
| read-ahead (`EXPERT_IO` bit 1, c2 0006) | 0 MB, but faults become async multi-MB reads | no; it is the measured ×1.9 (R12) on thrash days |
| next-layer guess (`EXPERT_IO` bit 4, not in c2) | 0 MB; overlaps the reads with compute | no; R18/R23 were inconclusive on Granite |
| `MADV_RANDOM` (bit 2) | trims read-around waste (a 256 KB window crossing an 880 KB slice edge, *estimated* up to 15 to 29 %) | no; lost on the SD888 (R12: 9.3 versus 11.8 tok/s) |
| hot set kept in anonymous memory (campaign 1's `EXPERT_ANON`, R23) | moves bytes from page cache to anon; the memory is the same | no. It can only protect a decode-hot set from the prefill sweep (below) |
| `mlock` of hot slices | Android's app `RLIMIT_MEMLOCK` is typically 64 KB (*not checked on this phone*) | not available |

**The one structural point: the prefill evicts the decode set.** LRU loses because the 2.6 GB prefill sweep evicts the
pages the decode needs next. Decode re-read lower bounds at 32 tokens, *estimated*:

| cache for this file | LRU late decode | Belady (optimal, prefill-warmed) | static set by prefill frequency | static set by this reply's decode frequency (oracle, not realisable) |
|---|---|---|---|---|
| 1.5 GB | 76.5 MB/token | 39.5 | 152.8 | 29.6 |
| 1.8 GB | 38.7 | 26.1 | 131.4 | 10.8 |
| 2.1 GB | 20.7 | 17.1 | 96.2 | 1.2 |

A runtime policy that keeps decode-hot slices (for example, the slices the last N decode tokens used, held in anon
memory or re-advised with `MADV_WILLNEED` after the prefill) could approach Belady. That is 26 to 40 MB/token at 1.5 to
1.8 GB, against LRU's 39 to 77. The prefill still re-reads (2.73 GB − cache) every run. **None of this makes OLMoE
resident on 2026-11-14's phone.** It lowers the per-token flash bytes from about 40 to 125 to about 25 to 40 MB, which
is still about 30 to 130 ms of I/O per token at 0.3 to 0.9 GB/s (*estimated*). The 25 tok/s goal needs ≤ 40 ms per
token in total.

**Conclusion for §3.** The page cache on that day cannot hold OLMoE's benchmark working set (2.73 GB against 1.6 to
2.35 GB), and no runtime-only lever closes 0.4 to 1.1 GB. On a day like 2026-11-08 (≈ 2.9 GB `Cached`) it fits with
little or no margin. Granite, which was resident only on such days, needs 0.14 GB less. Whether 25 tok/s is reachable
therefore depends on the phone's memory state as much as on the runtime. R52's 3.3 tok/s is a thrash-and-heat number,
not a compute number.

## 4. Clocks and the cool-down gate

**Clocks: thermal, not I/O.** *Device* data:
- Thermal headroom was 0.97 to 1.10 from the first run on. The OEM's "moderate" is 0.933; ≥ 1.0 forecasts severe
  throttling. The battery was 31.6 to 33.2 °C.
- The A78 policy (cpu 4-6) read 1,440 after runs 1-2, 1,209 after run 3 and 1,075 from then on (once 710 before a
  run). That is a ratchet that follows the headroom (heat_after 0.97 → 1.05 → 1.07 → 1.08 to 1.10). Schedutil gives
  an iowait boost on I/O wake-ups; blocked threads lower the utilisation, which gives a noisy level, not a one-way
  ratchet.
- r16 used the same counters (`mhz_after`) and read the A78 at 2,112 after runs with 35,450, 65,038 and 73,524 major
  faults. Heavy faulting did not lower the clocks on that day.
- The phone was already hot when r17 began: the first run's heat_after was 0.97. Why is not in the report.

What would settle it: log `scaling_max_freq` of policies 4 and 7 (the thermal cap) next to `scaling_cur_freq`, and
the thermal status, before and after each run.

**Gate: a kit bug, introduced in r17 and still in r18.**
- `lab/app-c2/lib/research.dart:317` reads `row['heat_after'] = (await DeviceProfile.thermal()).headroom` when a run
  ends.
- The next run's gate (`research.dart:280` → `assistant.dart:808` `_coolDown`) calls `thermal()` again within tens of
  milliseconds (`idle_ms` 40 to 62).
- `PowerManager.getThermalHeadroom` returns NaN when polled more than once a second (the app's own comment,
  `MainActivity.kt:142-143`). The Kotlin side turns NaN into `null`.
- `_coolDown` returns at once on `h == null`, so the result is `cool_ms` 0 and `heat` null.
- The exception, `stock-nr` 0, had an 8.2 s model reload before the gate. Its read was real (0.975), and it waited the
  full 90 s and timed out.
- The first run (`cold`) followed the header's `thermal()` call.

Kit r16 had no `heat_after` read (its rows have no such field), so its gate saw real values (0.64 to 0.87,
`cool_ms` up to 90 s).

Fix, Dart only, no runtime change:
1. In `_coolDown`, treat `null` as "unknown". Wait 1.1 s and re-read, up to 3 times, before giving up, and record
   `heat_null=true` if it stays null.
2. Read `heat_after` and then wait at least 1.1 s before the next gate.

Also, on a phone at 0.97 or more, the 0.65 threshold is not reachable in 90 s. With the gate fixed, every run would wait
90 s and be flagged `gate_timeout` (as `stock-nr` 0 was). The flag works as designed; the pre-registration has to say
how such runs are scored.

## 5. The template

- The GGUF has `tokenizer.ggml.bos_token_id` = `eos_token_id` = 50279. In this vocabulary that is the text
  `|||IP_ADDRESS|||`, and id 0 is `<|endoftext|>`. `add_bos_token` is false.
- `<|user|>` and `<|assistant|>` are **not** special tokens here, only plain text: `<|user|>\n` tokenizes as `<`,
  `|`, `user`, `|>`, newline.
- The template begins with `{{ bos_token }}`.
- llamadart fills `bos_token` from the metadata key `tokenizer.ggml.bos_token`. A GGUF has only the id, so the key is
  absent and the handler falls back to `'<s>'`
  (`third_party/llamadart/lib/src/core/template/handlers/generic_handler.dart:81`; the report's `format` is 0). The
  rendered prompt therefore starts with the literal text `<s>`, three tokens (`<`, `s`, `><`).
- The parser's expected render uses `<|endoftext|>` (id 0), the upstream tokenizer's BOS text. I did not re-check that
  against the publisher's files in this session (no web access).

Does it matter for this benchmark?
- **Speed: no.** The prompt is 156 tokens rather than 155. A trace with `<|endoftext|>` gives the same prefill union
  (968 slices) and the same total union (1,008); the decode distinct set is 771 rather than 790.
- **Text:** the greedy reply differs on the host but stays on topic. The benchmark definition is the app's render
  (owner decision 3), so the render stays as it is for comparability with R52. If it is ever fixed, that is a new
  benchmark text and has to be declared.

## 6. Everything campaign 1 did about residency and cold start

Line numbers are in `patches/llama.cpp/0001-cpu-runtime.patch` (identical to campaign 1's `lab/tiny-op.patch`, the r16
build). App paths are relative to campaign 1's app directory `vollie/lib/`.

| item | where | what it does | measured | in the c2 series? |
|---|---|---|---|---|
| `GGML_EXPERT_IO` bit 1 | 4115-4126, 4739-4797, call 5057, env 5164 | `MADV_WILLNEED` on the routed slices of gate, up and down when a layer's routing is known; skip-aware since r6 | R12 6.1 → 11.8 tok/s; R27 `eio 0` ×9.6 majflt; R29 +66 ms/token on a thrash day | yes, 0006 (`VOLLIE_EXPERT_IO=1`); off in r17, on in r18 `stack` |
| `GGML_EXPERT_IO` bit 2 | 4466-4476, reset 5168-5172 | `MADV_RANDOM` on expert tensors (no read-around) | R12: 9.3 versus 11.8 tok/s, lost | no |
| `GGML_EXPERT_IO` bits 4/8 | 4478-4570 (helper thread, next-layer guess with layer L+1's router) | prefetches the next layer's guessed experts off the compute threads | R18/R19 85 % guess hit, speed inconclusive | no |
| `GGML_DENSE_WILLNEED` | 4424-4458, call 5367, env 5255 | `MADV_WILLNEED` on all non-expert weights at each prompt graph | R32/R33: run together with the sampler bypass (outside-graph 1.90 → 1.13 ms); `MUL_MAT` window-1 excess unchanged (+6.6 → +6.6) | no (OLMoE: 9 MB/token, moot) |
| `GGML_REPACK_RELEASE` | `vollie-release.h` 5651-5699; calls 5466 (KleidiAI), 5616 (repack) | `MADV_PAGEOUT` (else `COLD`/`DONTNEED`) on a tensor's mmapped source after its repack | R17/R18 inconclusive ("the kernel had already evicted the source pages → 0 gain") | no |
| `GGML_EXPERT_ANON` | 3035-3131, used 3527/3709 | the N most-used experts copied to anon memory (`MADV_HUGEPAGE`) | R23: hit 66 to 78 %, Δ ≈ −0.7 ms (noise), +648 MB anon; closed | no |
| `GGML_MMID_TOUCH` | 4142-4153, 3542, 3630, env 5212 | demand-touch one byte per 4 KB page of a claimed chunk | R39-R41: parked (cold ≈ hot on a verified A78) | no |
| `GGML_EXPERT_Q4_LAYERS` | env 5255 | Q4_0x4 anon copy of the hottest layers' experts | R32/R33 numerics fail, dead | no |
| anon `MADV_NOHUGEPAGE` (bench arena) | 2455 | — | — | no |
| `mlock`, `mincore`, `readahead()`, `posix_fadvise` | none in 0001; llama.cpp's own `POSIX_FADV_SEQUENTIAL` at load | — | — | — |
| app: one-token warm-up after load | `assistant.dart:423-425` (`_complete('Hi', 1)`); 4-token warm-up after a speed-test reload, `assistant.dart:712` | builds the anon copies (head2) before the first reply; touches the model once | — | the c2 app has the same code |
| app: repack (CPU extra buffers) rule | `device.dart:100-107` `canCopyWeights`: total RAM − 700 MB copy ≥ 5 GB; `model_store.dart:19` | repack on on an 8 GB phone | R9: 0.66 GB repack copy counted against the cache | c2 kit sets repack per spec |
| app: served defaults | `research.dart` `applyServed`; production fallback after R49: tiny 262,144, expert I/O 1, 3 threads | r16 ran `EXPERT_IO=1` from the owner's saved preferences (`saved.expert_io` = 1 in r16) | R49 production note | r17 ran everything off |
| kit between runs | r16 `research.dart:156-195`: cool-down gate (headroom < 0.65, ≤ 90 s), 20 s idle pauses in groups 2 and 4, reference stories before the official groups | stories and idle evicted pages, so the next official run faulted 35 k to 74 k (R49 correction) | R48/R49 | the c2 gate is broken (§4) |
| file shrink D1 → D6 | R10 | 3.77 → 3.16 GB file, 6 experts per token | R10: not fixed by itself (cache shrank that day) | n/a (file frozen) |

R-log entries: R9 (fault counters, the diagnosis), R10, R11 (memory fields), R12 (expert I/O), R17/R18 (release,
helper, A55), R23 (anon cache, Q3K prefetch), R25/R26 (skip-aware read-ahead), R27/R29 (eio 0 falsifier, +66 ms on a
thrash day), R32/R33 (Q4 copies, dense willneed), R39-R41 (`MMID_TOUCH`), R48/R49 (residency at start decides a run;
cold start 12.3 to 18.5).

**In 0001 but missing from the c2 series:** `EXPERT_IO` bits 2, 4 and 8, `DENSE_WILLNEED`, `REPACK_RELEASE`,
`EXPERT_ANON`, `MMID_TOUCH`, `EXPERT_Q4_LAYERS`, and among the exact compute levers the in-register Q3_K kernel
(`Q3K_KERN`), `Q3K_ILP`, `MMID_CHUNK2`, `MM_CHUNK` and the router cast (c2 README, "Not in the series"). Of the
residency items, only bits 4/8 and `REPACK_RELEASE` are untested on a thrash day with OLMoE-sized slices, and §3 prices
both at about 0 MB of residency.

## 7. Campaign 1's full runtime on OLMoE (host part done)

**Build.** Scratch clone of `620d2225c` with `patches/llama.cpp/0001`, `0002` (Granite rollback, inert on `olmoe`) and
`0003` (sampler bypass) applied cleanly. CMake used the dev tree's options: Release, `GGML_NATIVE=OFF`,
`GGML_CPU_ARM_ARCH=armv8.2-a+dotprod+fp16`, no Metal, KleidiAI off, **`GGML_BLAS=OFF`**. A first build with the default
BLAS backend gave different `-b 512` and `-ub 32` values because Accelerate runs the batch mat-muls. That was a build
mismatch, not the patch. It loads `olmoe-Mi` with no change.

**Identity** (*host*, `-c 512 --chunks 4 -t 4`, per-chunk values and the md5 of the `--kl-divergence-base` log-prob
file): every configuration below equals memo-07's unpatched base at every setting run, both values and md5:

| configuration | `-b 512` | `-ub 32` | `-ub 1` | `-b 512 -nr` | `-ub 32 -nr` | `-ub 1 -nr` |
|---|---|---|---|---|---|---|
| all env unset (0001's compiled defaults: `MMID=3`, `Q3K_X4`, `Q3K_VEC`, `Q3K_ILP`, `Q3K_KERN`, `MMID_FUSE`, `MMID_CHUNK2`, `SSM_INPLACE`, `CONV_RING` on) | = | = | = | = | not run | = |
| r16 shipping minus numerics: + `GGML_TINY_OP=262144 GGML_EXPERT_IO=1` | = | = | = | = | = | = |
| + residency: `GGML_EXPERT_IO=5 GGML_REPACK_RELEASE=1 GGML_DENSE_WILLNEED=1` | = | = | = | = | not run | = |
| vanilla (R25's off-set: every default-on knob = 0, `TINY_OP=0`, `EXPERT_IO=0`, `EXPERT_SKIP=0`) | = | = | = | not run | not run | not run |

The base row is 8.9646 / 10.4244 / 9.2074 / 9.4783 (`3c8948f1…`) at `-b 512`, and 8.9550 / 10.4710 / 9.1889 / 9.4713
(`69f9df67…`) at `-ub 1`. **Official prompt, 32 greedy tokens** (`llama-completion -f <kit render> -n 32 --temp 0
-no-cnv --ignore-eos`): the campaign-1 runtime with `TINY_OP=262144 EXPERT_IO=1` gives the same text as the stock build
(the c2dev build with all knobs unset, which memo-07 showed equals the base). Repack on gives md5 `bde0011c…` for both,
repack off `88bae85d…` for both. A `GGML_CPU_PROF=1` line confirms that the levers engage on `olmoe`: `mmid=3 x4=1 ilp=1
kern=1 fuse=1 chunk2=1 eio=1 adv=384` (16 layers × 8 experts × 3 tensors advised per token), `nodes=469 barriers=241`
per token at 3 threads. Host µs/token is not a phone number and is not used.

**Switched off or guarded for OLMoE:** `GGML_EXPERT_SKIP` (its rule assumes renormalised top-k weights; OLMoE's are
not), `GGML_HEAD2` (the tied-head path keys on Granite's `token_embd`; OLMoE has an untied Q6_K `output`, use c2's 0009
if wanted), and `GGML_ROUTER_CAST`. All three are off unless set. `SSM_INPLACE` and `CONV_RING` are inert (no SSM ops).
The `EXPERT_IO` bit 4 guess reads the next layer's `ffn_gate_inp` and norm; it ran bit-identical here, but its hit
rate on OLMoE is unknown.

**Plan for a same-session comparison (no APK built).**
- One CPU module built from 0001, not two: Android's backend loader picks one `libggml-cpu-*` variant per process, so
  the c2 and c1 modules cannot both be loaded in one session without llamadart changes. Every comparison row then
  becomes an env set on that module:
  - (a) "stock" = R25's vanilla set (all default-on knobs 0), bit-identical above;
  - (b) campaign 1's runtime = env unset + `GGML_TINY_OP=262144 GGML_EXPERT_IO=1`;
  - (c) = (b) + `GGML_EXPERT_IO=5 GGML_REPACK_RELEASE=1` (the release needs a reload, so it goes last; `DENSE_WILLNEED`
    is moot with repack on).
- The c2 runtime's own rows stay comparable only across sessions, through R52/R54. If the lead needs (a) to be the c2
  build itself, the cost is a second APK, or porting 0001's exact levers into the c2 series. The second is the larger
  job.
- libllama from the same tree (0002 inert, 0003 sampler bypass under its old name `VOLLIE_SAMPLER_BYPASS`, as in c2's
  0007).
- App (`lab/app-c2`, kit r19):
  1. add the `GGML_*` names to the knobs the kit clears before each run (otherwise a value leaks into the next spec);
  2. table rows with `GGML_*` env;
  3. the gate fix of §4;
  4. log `mincore` residency (needs a small native hook; or read `RssFile` as now), `scaling_max_freq`,
     `read_ahead_kb` and `/proc/self/io` if readable.
- Parser: 0001's per-token lines are campaign 1's `cpu-prof ab=… tok=…` format (`GGML_CPU_AB_TOKEN=1`), not 0008's
  `c2-` lines. Port the reader from campaign 1's `tools/research_parse.py` into `tools/c2_research_parse.py`, as new
  code. Plain official runs need nothing new.
- Effort, *estimated*: Android build of the 0001 tree with `tools/c2_build_kit.sh`'s flow, 1 to 2 h; kit table, knob
  list and gate fix, 2 to 3 h; parser reader for the c1 line format, 2 h; host checks (text identity per row, line
  regexes), 1 to 2 h. **About one working day for Tom**, plus Clyde's pre-registration.
- It cannot be tested here whether campaign 1's runtime beats c2's `stack` on OLMoE. The exact levers it adds
  (`Q3K_KERN`, `ILP`, `CHUNK2`) were worth a few ms on Granite (R35-R38, R47). §3 says the page cache, not compute,
  sets R52's number.

## 8. Not verified

- Device page residency itself (no `mincore` or `/proc/self/io` in the report); `read_ahead_kb` on this phone; UFS
  rates (assumed).
- The Granite `RssFile` discrepancy (§2).
- The publisher's BOS text (§5) against the upstream files.
- Host traces used the c2dev libllama with the device text forced. The prefill routing on the device may differ
  slightly in its numerics. The decode slice sets of the stock and proper-template traces differ by 19 slices; the
  prefill unions do not.
- The identity table covers 4 chunks, not the full set.
