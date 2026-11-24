# c3 memo-12 work: stock perplexity on a host build with KleidiAI (40 chunks)

- **Date:** 2026-11-24 (started 2026-11-23). **Author:** tom, `claude-opus-5-5`. Not committed. Plan:
  [bonnie-joins-r9.md](bonnie-joins-r9.md) §1 and §3; host run allowed by the owner on 2026-11-23 (end of
  [../../CAMPAIGN.md](../../CAMPAIGN.md)).
- **Labels.** Every perplexity value and timing here is *host* (owner's Apple M4 Pro laptop). Shifts are arithmetic
  on *host* values. Nothing ran on the phone.
- **Status: 6 of 6 runs finished.** Yi `-b 512` ran on 2026-11-23 on a shared laptop. The other five ran on
  2026-11-24, 09:27 to 10:32, one job at a time, after the laptop was free (load average about 2; Yi `-ub 1` at
  8.4 s per chunk). Two earlier Yi `-ub 1` attempts on the shared laptop (another agent's two 8-thread perplexity
  jobs, load average 24 to 45, 166 to 320 s per chunk) did not finish: one was killed by the agent's 2 h
  background-task limit at chunk 18, and I stopped the other at chunk 9. `tools/ppl.sh` overwrote that partial
  log; the log kept is the complete 2026-11-24 run.

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
served file's sha256 was checked before each run against `results.csv` (Yi `702cc2ebaa3b…`, Llama-2
`8eb4952f2592…`, DeepSeek `f403ddc29802…`: all ok). The five 2026-11-24 runs used the same command, without `nice` or
`DYLD_LIBRARY_PATH`. All six wiki logs print `KLEIDIAI = 1`.

Logs: `lab/logs/p/ppl-c3k40-<m>-q4_0-<b512|ub1>-wiki.log` and `-en.log` (the script's English sanity text, also
40 chunks), for `<m>` = `yi15-6b`, `llama2-7b`, `deepseek7b`.

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

## 4. Result

Chunk-40 values of the repack path, re-read from the freeze logs (`lab/logs/p/ppl-c3f-<m>-q4_0-<b512|ub1>-wiki.log`):
Yi 7.6966 / 7.6817, Llama-2 7.4071 / 7.4075, DeepSeek 8.2231 / 8.2192 (`-b 512` / `-ub 1`), equal to r9's table.

| subject | setting | repack `[40]` (freeze) | KleidiAI `[40]` | shift | `[n]` where KleidiAI > repack | en.txt, 40 chunks, KleidiAI |
|---|---|---|---|---|---|---|
| Yi | `-b 512` | 7.6966 | 7.7347 | **+0.50 %** | 40 of 40 | 8.1805 |
| Yi | `-ub 1` | 7.6817 | 7.7165 | **+0.45 %** | 39 of 40 | 8.1857 |
| Llama-2 | `-b 512` | 7.4071 | 7.4122 | +0.07 % | 38 of 40 | 4.9294 |
| Llama-2 | `-ub 1` | 7.4075 | 7.4034 | −0.06 % | 8 of 40 | 4.9324 |
| DeepSeek | `-b 512` | 8.2231 | 8.2797 | **+0.69 %** | 36 of 40 | 7.1759 |
| DeepSeek | `-ub 1` | 8.2192 | 8.2742 | **+0.67 %** | 39 of 40 | 7.1697 |

All values *host*. Shifts to three decimals: +0.495, +0.453, +0.069, −0.055, +0.688, +0.669 %. The `[n]` column
compares the running estimates of the two logs at each of the 40 chunks.

- **Pre-stated reading (r9): four of six shifts exceed ±0.1 %** (Yi and DeepSeek at both settings). Llama-2 is
  inside ±0.1 % at both. The lead decides whether to ask for full-set runs. A shift is not a budget event.
- Against Bonnie's band (−0.3 to +1.0 %, centre +0.3 %): all six are inside. Her P(at least one subject outside
  ±0.3 %) = 0.5: two of three subjects are outside ±0.3 %.
- The direction holds at both batch settings for Yi and DeepSeek (KleidiAI higher, `-b 512` and `-ub 1` within
  0.05 percentage points of each other). Llama-2 has no consistent direction: +0.07 % at `-b 512`, −0.06 % at `-ub 1`.
- Size only: R61's Granite moved +0.57 / +0.50 % with 541 MiB in KleidiAI. Here, with 3.2 to 3.7 GiB, the
  movement ranges from near zero (Llama-2) to +0.69 % (DeepSeek). The amount of weight in KleidiAI does not predict
  the shift by itself.
- Not tested: which KleidiAI step causes it (the int8 head re-quantisation of r8 §1.2, the per-row activation
  quantisation of the Q4_0 kernels, or float order). §6 only places the first differing recorded node.

## 5. Reproduce

The loop of r9 §3, with a sha256 check before each run; finished logs are skipped. Unshared laptop: about 4 min per
`-b 512` run and 13 to 18 min per `-ub 1` run, each including the English text (*host*, 2026-11-24).

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

