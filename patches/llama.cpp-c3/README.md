# Campaign 3 patch series for llama.cpp

Base: the campaign 2 phone runtime of kit r28, i.e. `620d2225c` + `../llama.cpp/0001` to `0003` + `../llama.cpp-c1-olmoe/0001`
to `0012`, applied in that order (campaign 3 starts from campaign 2). Every lever is off unless its environment variable
is set.

```sh
for p in patches/llama.cpp/0*.patch patches/llama.cpp-c1-olmoe/0*.patch patches/llama.cpp-c3/0*.patch; do git -C <llama.cpp at 620d2225c> apply "$PWD/$p"; done
```

## 0001-probe.patch: device probes P1, P2, X1, X2 (no numerics change)

New file `src/llama-vollie-probe.cpp` in libllama, built with `-march=armv8.2-a+dotprod+fp16` on arm64 (a stub that returns
-1 elsewhere); `src/CMakeLists.txt` lists it. Design: `campaigns/03-dense15-5pct/reports/memo-02-host-screens.md` §6.

| knob / call | meaning |
|---|---|
| `VOLLIE_PROBE` | unset, empty or `0` = off: `llama_vollie_probe` returns -1 and does nothing |
| `int llama_vollie_probe(const char * block, const char * dir, char * out, size_t cap)` | runs one block (`p2`, `x1`, `x2`, `p1`; `dir` = folder for P1's scratch file) and writes one `c3-probe:` line per cell into `out`; returns the bytes written, -1 off, -2 unknown block |

Nothing in the graph, the kernels or the model path calls the probe, so with the knob unset or set the per-chunk PPL is
the r28 runtime's. Checked (*host*, Mistral-7B-v0.3 Q4_0, `llama-perplexity` running values): identical to the r28
runtime and to stock `620d2225c` at `-b 512` and `-ub 32` (4 chunks) and `-ub 1` (2 chunks), and with `VOLLIE_PROBE=1`
at `-b 512`. The probe itself was run on the host (functional test only; host numbers mean nothing for the phone).
Parser: `tools/c3_probe_parse.py` (`--selftest`).

## 0002-d-rows.patch: D2 rows and D3 header of memo-05 §4 (no numerics change)

Changes only `src/llama-vollie-probe.cpp` (on 0001). Still off unless `VOLLIE_PROBE` is set, and nothing outside the file
calls it. Design: `campaigns/03-dense15-5pct/reports/memo-05-y1-column-kernels.md` §3, §4; kernels as
`tools/c3_colkern_bench.c` (memo-05 work file `tom-host.md` T1, T2, E6).

| call / block | what it adds |
|---|---|
| every block (and the new block `hdr`, header only) | D3 `hdr` lines first: `props` (the readable `ro.lmk.*`, `persist.device_config.lmkd_native.*`, `ro.config.low_ram`, via `__system_property_foreach`; `n=` count), `zram` (`/sys/block/zram0/disksize`, `mm_stat`, legacy `mem_used_total`, `/proc/swaps` line 2), `psi` (`/proc/pressure/memory`, `/proc/pressure/cpu`, one line per `some`/`full`), `mem` (`/proc/meminfo`), `vm` (swappiness, min_free_kbytes, watermark_scale_factor, extra_free_kbytes, the old in-kernel LMK minfree). A source that cannot be read prints `e<errno>` (PSI: `errno=<n>`) |
| `p2` (at its end) and the new block `p2d` (D2 alone) | an identity line on the phone (K1 at keep 1.0 and 0.6, Y1b, Y1b on a row list, each `memcmp` against the stock 4x4 gemv), then one `p2d` line per cell on cpu 4 (one pinned A78), single thread, 1 s warm-up + 2 s measured: inputs 4096 (gate/up) and 14336 (down); hot (one matrix of about 260 to 290 KB, inside the A78's 512 KB L2) and cold (1024-row matrices walked through a 512 MB arena); kernels stock, K1 at TR 128 at keep 1.0 / 0.65 / 0.60 (not hot on 14336: one 128-row tile is 1 MB), Y1b (dense), and Y1b on a sorted half-kept row list (4096 inputs, 2304 B rows; Qwen2.5's are 2016 B) |

Each `p2d` line prints `k` = ns per touched weight ÷ the stock ns per weight of the same shape and regime (the mean of a
stock cell run before and one run after the group, `drift` = after ÷ before), and `t_vs_stock` = k × d, the time against
dense stock. D2 adds about 70 s to `p2` (*host* functional run: 69.5 s).

Checked (*host*, Mistral-7B-v0.3 Q4_0, `llama-perplexity -c 512 -t 8`, knob unset): running values identical to stock
`620d2225c` at `-b 512` and `-ub 32` (4 chunks) and `-ub 1` (2 chunks). The new code compiles with the NDK (API 28,
`-march=armv8.2-a+dotprod+fp16`). Blocks `hdr` and `p2d` ran on the host as a functional test (identity lines all 1;
host timings mean nothing for the phone). Parser: `tools/c3_probe_parse.py` (`--selftest`) reads `hdr` and `p2d`.
