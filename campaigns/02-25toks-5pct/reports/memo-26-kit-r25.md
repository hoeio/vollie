# Memo 26: research kit r25, kit r24 plus the KV clear by MADV_DONTNEED (patch 0008), on Granite and OLMoE

Agent model: claude-opus-5-5

Date 2026-11-16. Author: Tom (code). **Nothing here ran on a phone, and no phone speed is stated.** The APK was built
and inspected on the laptop. Labels: *host* = Apple M4 Pro (macOS, or a Linux aarch64 container on it), relative only;
*estimated* = arithmetic; *device* = quoted from an existing report. Kit r25 is kit r24 ([memo-23](memo-23-kit-r24.md))
with one factor added; this memo lists only what changed. The lever is memo-24 §A3 (a), shortlist item 1
([memo-24](memo-24-anon-memory-and-rest.md)).

## 1. The lever

The kit runs with `reusePrefix = false`, so llamadart calls `llama_memory_clear(mem, true)` before every prompt. That
reaches `llama_kv_cache::clear(true)` and, on Granite's hybrid memory, also `llama_memory_recurrent::clear(true)`; both
called `ggml_backend_buffer_clear(buf, 0)`, a `memset` of the whole buffer. On the phone that rewrites OLMoE's 285 MB
KV buffer (q8_0, `n_ctx` 4096) and Granite's 17.8 MB KV + 58.1 MB recurrent state before every prompt, swapping their
zero pages back in after a gate wait (memo-24 §A2.2, *device* fit: TTFT − prefill = 41.6 ms + 3.47 µs per pre-prompt
major fault on OLMoE).

Patch `patches/llama.cpp-c1-olmoe/0008-c2-kv-dontneed.patch` (libllama only; README entry there):
- `VOLLIE_KV_DONTNEED=1`, read at every clear, unset = off (the old `memset`).
- On Linux/Android, for a host buffer of type `CPU` in a private anonymous mapping (checked in `/proc/self/maps` once
  per base address): `madvise(MADV_DONTNEED)` on the page-aligned interior, `memset` on the edges. Any other case, or a
  failing `madvise`, falls back to the whole `memset`; macOS always does.
- **Why the pages read back as zeros on the phone:** the KV and recurrent buffers come from
  `ggml_backend_alloc_ctx_tensors_from_buft` → the CPU buffer type → `ggml_aligned_malloc` → `posix_memalign`. On
  Android (bionic, scudo) and glibc a block of this size is a private anonymous `mmap`; `MADV_DONTNEED` on private
  anonymous memory gives zero-fill-on-demand pages. The `/proc/self/maps` check makes the claim checked at run time
  rather than assumed; if it does not hold the line says `why=map` and the `memset` runs.
- The clears at cache construction are unchanged, so the load is the same in both settings.
- **Kit line**, one per buffer cleared, printed when the knob or `GGML_C2_IO=1` is set (every kit run):
  `c2-kvclear run=<VOLLIE_RUN> kind=kv|rs on=0|1 bytes=<buffer bytes> released=<bytes released by DONTNEED>
  why=ok|off|os|notcpu|small|map|madvise us=<clear µs>`.
  Expected on the phone with the knob: OLMoE one `kind=kv` line with `released` about 285,208,576; Granite a `kv` line
  (about 17.8 MB) and an `rs` line (about 58.1 MB). Without the knob `released=0 why=off`.

## 2. Host checks (*host*, 2026-11-16)

All with the kit's off env (`GGML_TINY_OP=262144 GGML_EXPERT_IO=1 VOLLIE_SAMPLER_BYPASS=1 GGML_C2_IO=1`), `-c 512`,
wikitext-2-raw test; tree `lab/llama.cpp-c1r25` = 620d2225c + `patches/llama.cpp/0001-0003` +
`patches/llama.cpp-c1-olmoe/0001-0008`; builds without KleidiAI, `armv8.2-a+dotprod+fp16`.

**Per-chunk PPL, knob off against on** (4 chunks at `-b 512` and `-b 512 -ub 32`, 2 chunks at `-b 512 -ub 1`; time
allowed no more):

| platform | model | `-b 512` | `-ub 32` | `-ub 1` | clear path with the knob |
|---|---|---|---|---|---|
| macOS (clang, `-t 8`) | olmoe-Mi | equal (8.9646 10.4244 9.2074 9.4783) | equal | equal | `memset` (`why=os`) |
| macOS | granite-Mi | equal (4.9584 6.6520 6.1119 6.6024) | equal | equal | `memset` (`why=os`) |
| Linux aarch64 container (gcc, `-t 10`) | olmoe-Mi | equal (8.9607 10.4468 9.2244 9.5063) | equal | equal | `why=ok`, 67,104,768 of 67,108,864 B released per clear |
| Linux | granite-Mi | equal (4.9811 6.6764 6.1185 6.6045) | equal | equal | `why=ok`, KV 4,190,208 of 4,194,304 B, recurrent 58,056,704 of 58,060,800 B |

- macOS and Linux values differ from each other (compiler); the comparison is within one build.
- macOS, knob unset: equal at `-b 512` (4 chunks, both models) to the existing 0001-0007 build (`lab/llama.cpp-c1pr/build`).
- The perplexity tool clears with `data = true` before every chunk, so the on runs went through the `madvise` path
  4 or 2 times each on Linux. Caveat: KV cells outside the used range are masked, so PPL alone would not see a
  non-zero cell; the next two checks look at the bytes and at the app's settings.

**Greedy text on the app's settings** (test program: `n_ctx` 4096, q8_0 K and V, flash attention, the benchmark's
prompt shape, three prompts in one process, each after `llama_memory_clear(mem, true)`, 32 greedy tokens): the three
texts are equal to each other and between knob off and on, for both models, on macOS and on Linux. On Linux the OLMoE
clear released 285,208,576 of 285,212,672 B each time (the phone's KV size), Granite 17,821,696 + 58,056,704 B.

**Unit test of the function on Linux** (buffers of 299,040,891 B, 12,295 B, 100 B and 61,001,728 B from the CPU buffer
type, filled with 0xAB first): every byte is zero after the clear with and without the knob; RSS fell by 291,960 KB for
the 299 MB buffer and 59,512 KB for the 61 MB one; writing and clearing again after a release works; the 100-byte buffer
is declined as `small`; a `CPU_Mapped` buffer over a shared mapping is declined (`notcpu`) and zeroed by the `memset`.

Test sources (not tracked): the scratch programs `kvclear_test.cpp` and `greedy.cpp` of this session. They link the
built `libllama`; the container was Ubuntu 24.04 with g++ and cmake.

**Not checked:** Android itself (bionic and scudo, kernel 5.4.274, zram, the unknown THP mode); the `map` decline path
(no public API puts a `CPU` buffer over shared memory). The phone run shows both in the `c2-kvclear` lines.

## 3. Runtime and build

- Series: `patches/llama.cpp/0001-0003` + `patches/llama.cpp-c1-olmoe/0001-0008`, checked file by file and built by
  `tools/c2_build_kit.sh` with `android/build_cpu.sh`, flags unchanged.
- **The CPU module differs from r24's because of 0007** (routing prior, committed after r24 was built; off without
  `VOLLIE_PRIOR`, its identity with the knob unset was checked on the host in memo-25). The kit never sets `VOLLIE_PRIOR`
  and now clears it before every run. 0008 changes `libllama.so` only.

| | bytes | sha256 |
|---|---|---|
| `vollie-c2-1.25.0+25-research-r25.apk` | 17,229,319 | `1bbbec0ca533a0529305d99711bd1e1ac9664af1727586e62b56fed8e4c180d4` |
| `libggml-cpu-android_armv8.2_2.so` | 1,767,336 | `d6b7b3c98fa2ba66f75019d756b22b79a72dd2638e2e1a1e7f647689cc1f731c` (r24: `e6ef857b…`) |
| `libllama.so` | 4,287,440 | `17a197d48d779e8dc1780d9486e80e75782f9f2053a4c16a2d3ea451bc56cebb` (r24: `aeab4dad…`) |

The build record (`assets/kit_build.json`, `lab/dist-c2/BUILD-r25.txt`) lists the eleven series patches and app
patches `0001` to `0009`. The built `libllama.so` imports `madvise` and contains the `c2-kvclear` format.

## 4. Arms, factor, run list

Everything of r24 stays: blocks G O O G, per block 2 warm-ups, two scored ABBA quads (off num num off) with the plateau
gate first in odd blocks, 2 profiled runs with r16's gate, 3 threads, hold in every gated run, charger guard, the same
arms and load env, the same counters. **48 runs, 40 gated.**

**One factor added, kv**: `VOLLIE_KV_DONTNEED=1` set in half of the scored runs, alternated per run:

| block | model | quad 1 (gate) | kv in quad 1 | quad 2 (gate) | kv in quad 2 |
|---|---|---|---|---|---|
| b1 | granite | plateau | on off on off | r16 | off on off on |
| b2 | olmoe | r16 | off on off on | plateau | on off on off |
| b3 | olmoe | plateau | on off on off | r16 | off on off on |
| b4 | granite | r16 | off on off on | plateau | on off on off |

- Each arm × gate-rule cell of a block has one kv run and one plain run; kv sits on scored positions 1 3 6 8 or
  2 4 5 7 of a block (mean position equal). Per model: 8 kv and 8 plain scored runs (4 per arm, 4 per gate rule).
- Warm-ups and profiled runs never set it (as in r24).
- Pairs for scoring: within each quad, kv on and off of the same arm and gate rule (2 pairs per quad, 8 per model).
- Run ids: r24's `<model>_<arm>_pl` / `_r16` plus suffix `_kv` when on (e.g. `olmoe_num_pl_kv`); spec field `kv`
  (true / false); report `kv_rule`; `proto: r25`.

**Names for the pre-registration:**
- env: `VOLLIE_KV_DONTNEED=1` (on), unset (off);
- arms unchanged: `off`, `num` (OLMoE num `VOLLIE_SKIP_MASS=900 VOLLIE_HEAD2=1024 VOLLIE_SKIP_STATS=1`; Granite num
  `GGML_EXPERT_SKIP=130 GGML_HEAD2=2048 GGML_HEAD2_Q=q4_0 VOLLIE_SKIP_STATS=1`; OLMoE load env `VOLLIE_HEAD2_REPACK=1`);
- factor `kv`: spec `kv: true|false`, id suffix `_kv`;
- line `c2-kvclear` with `kind`, `on`, `bytes`, `released`, `why`, `us`.

## 5. Report schema (additions over r24)

- Top level: `kit: r25`, `app: 1.25.0+25`, `kv_rule`, `baseline_note` mentions the factor; each spec has `kv`.
- Per row: the runtime's `c2-kvclear` lines among the row's runtime lines (tagged with `run=`, filed like `c2-io-blk`).
  Already recorded since r24 and used for this factor: `ttft_ms`, `majflt` (with the `c2-io` line's window, giving the
  pre-prompt majors as memo-24 defines them), `vmstat_before` / `vmstat_first` / `vmstat_after` (`pswpin`), `mem_before`
  / `mem_first` / `mem_after` (`VmSwap`).
- **Parser:** `tools/c2_research_parse.py` is not changed in this hand-off. It parses r25 as r24 and pools kv on and off
  inside each arm; a kv split (pre-prompt majors, Δpswpin start → first chunk, TTFT, tok/s, `released`, kv − plain per
  model with paired ± SE) is still to be written before scoring.

## 6. Owner steps

1. Before the session, close background apps and unplug the charger (the start screen says so; the kit pauses on a
   charger).
2. Install `lab/dist-c2/vollie-c2-1.25.0+25-research-r25.apk`. It is an update over r24 and keeps both models.
3. In Settings, tap **"Run research kit r25 (~35-75 min)"**. The first 30 s are an idle baseline.
4. When it ends, tap **"Save research results to Downloads"** and send `Download/vollie-research-r25-<time>.json`.

## 7. Time (*estimated*)

The run list has r24's length and waits; the factor adds about 0 (a `madvise` takes milliseconds). So as memo-23 §8:
about 60 to 63 minutes on a day when the gates are not reached, 32 to 40 minutes when they are.

## 8. Never run on a phone

- `MADV_DONTNEED` on bionic's allocation, the `/proc/self/maps` check from native code (the app already reads that
  file for the library hashes in every kit since r21; if it could not be read, every clear would say `why=map` and the
  factor would be void, not a lever result), the swap and fault effect itself.
- Everything memo-23 §9 lists for r24.
