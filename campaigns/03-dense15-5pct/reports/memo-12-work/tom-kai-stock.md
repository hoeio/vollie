# c3 memo-12 work: stock perplexity on a host build with KleidiAI (40 chunks), partial

- **Date:** 2026-11-24 (started 2026-11-23). **Author:** tom, `claude-opus-5-5`. Not committed. Plan:
  [bonnie-joins-r9.md](bonnie-joins-r9.md) §1 and §3; host run allowed by the owner on 2026-11-23 (end of
  [../../CAMPAIGN.md](../../CAMPAIGN.md)).
- **Labels.** Every perplexity value and timing here is *host* (owner's Apple M4 Pro laptop). Shifts are arithmetic
  on *host* values. Nothing ran on the phone.
- **Status: 1 of 6 runs finished** (Yi `-b 512`). The host was shared with another agent's perplexity chain
  (`lab/llama.cpp-c3x6`, two 8-thread jobs on 12 cores, load average 24 to 45). Yi `-ub 1` ran at 166 to 320 s per
  chunk against the freeze's 9.1 to 11.0 s, about 3.3 h per `-ub 1` run, which is longer than an agent's 2 h
  background-task limit. It was killed once by that limit (2026-11-24 01:2x, at chunk 18) and stopped by me
  the second time (05:13, at chunk 9). The five missing runs need the resume command in §5.

## 1. Build

Same tree and commit as the S3 freeze (`lab/llama.cpp-c3` at `26f78f9251ff`, `git status` clean before and after;
the source tree was not modified). Separate build directory `lab/llama.cpp-c3/build-kai`:

```
cmake -S lab/llama.cpp-c3 -B lab/llama.cpp-c3/build-kai -DCMAKE_BUILD_TYPE=Release \
  -DGGML_CPU_KLEIDIAI=ON -DGGML_NATIVE=OFF -DGGML_CPU_ARM_ARCH=armv8.2-a+dotprod+fp16 \
  -DGGML_METAL=OFF -DGGML_BLAS=OFF \
  -DFETCHCONTENT_SOURCE_DIR_KLEIDIAI=$PWD/lab/llama.cpp-c1h6/build-kai/_deps/kleidiai-src
cmake -S lab/llama.cpp-c3 -B lab/llama.cpp-c3/build-kai -DLLAMA_BUILD_SERVER=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_OPENSSL=OFF
cmake --build lab/llama.cpp-c3/build-kai -j 8 --target llama-perplexity
```

- **KleidiAI v1.31.0** (the version the tree pins, `KLEIDIAI_COMMIT_TAG "v1.31.0"`), from the sources R61's build
  had already fetched (`lab/llama.cpp-c1h6/build-kai/_deps/kleidiai-src`, project `VERSION 1.31.0`). No download.
- **Flags against the freeze build** (`lab/llama.cpp-c3/build/CMakeCache.txt`), every user-visible cache entry
  diffed: the only option that differs is `GGML_CPU_KLEIDIAI` (OFF in `build`, ON in `build-kai`). Equal in both:
  `CMAKE_BUILD_TYPE=Release`, `GGML_NATIVE=OFF`, `GGML_CPU_ARM_ARCH=armv8.2-a+dotprod+fp16`, **`GGML_BLAS=OFF`**
  (no `libggml-blas` in `build-kai/bin`), `GGML_METAL=OFF`, `GGML_ACCELERATE=ON`, `GGML_CPU_REPACK=ON`,
  `GGML_LLAMAFILE=ON`, `GGML_OPENMP=ON` (not found by Apple clang in either, `GGML_OPENMP_ENABLED=OFF`),
  `BUILD_SHARED_LIBS=ON`, `LLAMA_BUILD_SERVER=OFF`, `LLAMA_BUILD_TESTS=OFF`, `LLAMA_OPENSSL=OFF`. The remaining
  diff lines are KleidiAI's own cache entries, CTest defaults, and OpenSSL find results that the first configure
  cached (unused once `LLAMA_OPENSSL=OFF`).
- Binaries (sha256, first 16): `build-kai/bin/llama-perplexity` 2a15f95f93d555a1, `build-kai/bin/libggml-cpu.0.27.0.dylib`
  c14fe2e41c177c09.
- **Library check (Ebony r6 §3, `nice` and `DYLD_LIBRARY_PATH`):** the runs used neither `nice` nor
  `DYLD_LIBRARY_PATH`. The binary links `@rpath/libggml*.dylib` with `LC_RPATH` = `lab/llama.cpp-c3/build-kai/bin`;
  `lsof` on the running Yi process listed only `build-kai/bin` ggml and llama libraries. Every gated log prints
  `KLEIDIAI = 1` in `system_info`, which the freeze logs do not.

## 2. Runs

Invocation as in r9 §3 (`tools/ppl.sh`, so `-c 512 -b 512 -t 8 -ngl 0`, wikitext text `lab/c3/text/wiki.test.raw`,
the same as `tools/c3_freeze.sh`), with `GGML_KLEIDIAI_SME=0`, `--chunks 40`, plus `-ub 1` for the ub1 tag. The
served file's sha256 was checked before each run against `results.csv` (Yi `702cc2ebaa3b…` ok, 3 times).

| run | tag | log | state |
|---|---|---|---|
| Yi `-b 512` | `c3k40-yi15-6b-q4_0-b512` | `lab/logs/p/ppl-c3k40-yi15-6b-q4_0-b512-wiki.log` (+ `-en.log`) | done |
| Yi `-ub 1` | `c3k40-yi15-6b-q4_0-ub1` | `lab/logs/p/ppl-c3k40-yi15-6b-q4_0-ub1-wiki.log` | **partial, chunk 9; no `Final estimate`, so `tools/ppl.sh` re-runs it** |
| Llama-2, DeepSeek, both settings | `c3k40-<m>-q4_0-<b512\|ub1>` | not yet | not started |

## 3. Kernel path and buffers

The gated logs do not carry the `kleidiai:` and `load_tensors:` lines: in this tree `llama-perplexity` does not
print them at its default verbosity (the freeze logs have no buffer lines either). So one extra one-chunk run per
subject, same binary, same environment, `-lv 4 --chunks 1`, logs `lab/logs/p/diag-c3k40-<m>-q4_0-load.log`:

| subject | `CPU_KLEIDIAI model buffer size` | Bonnie r8 prediction | `CPU_REPACK` line | `[1]` |
|---|---|---|---|---|
| Yi | 3,220.75 MiB | 3,220.7 | none | 5.0354 (equal to the gated `-b 512` run's `[1]`) |
| Llama-2 | 3,599.38 MiB | 3,599.4 | none | 5.7843 |
| DeepSeek | 3,658.06 MiB | 3,658.0 | none | 5.1557 |

The remaining weights (embedding table and norms) sit in `CPU_Mapped` (140.64 / 70.33 / 225.02 MiB plus 0.02 MiB
pieces). The kernel-feature lines do not appear even at `-lv 4`. They do appear when the same `build-kai` libraries
are loaded by the E-T13 script below (ggml's default stderr logger, `GGML_KLEIDIAI_SME=0`):
`kleidiai: primary q4 kernel feature DOTPROD`, `kleidiai: primary q8 kernel feature I8MM`, `no compatible q4_k kernels`,
`no compatible f32 kernels`, `SME disabled`. Source agrees (`kleidiai/kernels.cpp`: with SME off, the first Q4_0
entry whose features match is the `__APPLE__` DOTPROD entry; the first Q8_0 match is I8MM|DOTPROD).

## 4. Result so far

Chunk-40 values of the repack path, re-read from the freeze logs (`lab/logs/p/ppl-c3f-<m>-q4_0-<b512|ub1>-wiki.log`):
Yi 7.6966 / 7.6817, Llama-2 7.4071 / 7.4075, DeepSeek 8.2231 / 8.2192 (`-b 512` / `-ub 1`), equal to r9's table.

| subject | setting | repack `[40]` (freeze) | KleidiAI `[40]` | shift |
|---|---|---|---|---|
| Yi | `-b 512` | 7.6966 | **7.7347** | **+0.50 %** |
| Yi | `-ub 1` | 7.6817 | not finished (chunk 9: 8.7217 against 8.6734, +0.56 %) | (partial) |
| Llama-2 | both | 7.4071 / 7.4075 | not run | |
| DeepSeek | both | 8.2231 / 8.2192 | not run | |

- **Pre-stated reading (r9): Yi `-b 512` is outside ±0.1 %.** The KleidiAI path raises Yi's 40-chunk stock by
  +0.50 % (*host*). That is inside Bonnie's band (−0.3 to +1.0 %) and close to R61's Granite (+0.57 %). Not a
  budget event; the lead decides about full-set runs.
- The shift is not a late jump: at `-b 512` the KleidiAI running estimate is above the repack path's at all 40
  `[n]` values (chunk 1: 5.0354 against 5.0145). Yi English sanity at `-b 512`, 40 chunks: 8.1805 (no freeze value
  at 40 chunks to compare with).

## 5. To finish (resume command; run when the laptop is not shared)

The script below verifies each sha256 and skips finished logs, so it re-runs only the five missing runs. Time
(*estimated* from the freeze logs, unshared laptop): about 30 min for all five. Shared with one other 8-thread
job, one `-ub 1` run took about 3.3 h.

```
for e in yi15-6b:702cc2ebaa3bd4933598e25b556961b95ac260ed15df74fa09df2b2b7b48a299 \
         llama2-7b:8eb4952f2592fdfc1288695f6b85aaf48d5628ec5d2c75f9387d3144652ad4bf \
         deepseek7b:f403ddc298020af0fe8dfbbfec9a74cd3f8951938d6c21540054d85f09315118; do
  m=${e%%:*} s=${e#*:} f=lab/c3/models/$m-q4_0.gguf
  for b in b512 ub1; do
    [ "$(shasum -a 256 $f | cut -d' ' -f1)" = "$s" ] || { echo "sha mismatch $f"; break 2; }
    x=; [ $b = ub1 ] && x="-ub 1"
    GGML_KLEIDIAI_SME=0 sh tools/ppl.sh --llama-dir lab/llama.cpp-c3/build-kai/bin \
      --text lab/c3/text/wiki.test.raw c3k40-$m-q4_0-$b $f --chunks 40 $x
  done
done
```

## 6. E-T13 cross-build node check (Ebony's `ebony_batchdiff.py`, cross-build mode)

Yi, N = 16 tokens of `wiki.test.raw`, 2 threads, `GGML_KLEIDIAI_SME=0`, `EBONY_SAVE` once with `build/bin`
(loaded `CPU_REPACK` 3,175.08 MiB) and once with `build-kai/bin` (loaded `CPU_KLEIDIAI` 3,220.75 MiB). The
script records the last token's column of every 2-D float node of one 16-token batch. Comparison (*host*):

- 553 nodes in both, same names and order.
- **Layer 0: every node bit-equal**, including `Qcur-0`, `Kcur-0`, `Vcur-0`, `kqv_out-0`, `attn_out-0`, the FFN
  and `l_out-0`. Layer 1's `Qcur-1`, `Kcur-1` and `Vcur-1` are bit-equal too.
- **First difference: `kqv_out-1`** (the attention output of layer 1, not a Q4_0 weight product): max |d| 5.5e-5,
  relative 6e-4, all 4,096 entries. From there on 525 of 553 nodes differ; `result_output` max |d| 0.49, same
  argmax.
- Limit: only the last column is recorded. Layer 1's attention reads K and V of all 16 positions, so a difference
  in earlier columns of `Kcur-1` or `Vcur-1`, which this script does not see, would also show first at
  `kqv_out-1`. This check does not show which kernel causes the +0.50 %.
