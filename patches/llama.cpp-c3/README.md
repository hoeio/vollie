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
