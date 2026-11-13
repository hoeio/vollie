# Memo 06: device CLI kit over adb (stopped)

Agent model: claude-opus-5-5

Date 2026-11-14. Author: Tom (code). **Nothing here was run on a device. No speed is stated.**

The work stopped after the owner's message of 2026-11-14: the phone can never be attached to the laptop by
USB, so the adb route is dead. The research-APK route replaces it ([memo-09](memo-09-research-apk-path.md)).
What exists, for the record:

| item | where | state |
|---|---|---|
| static arm64 `llama-bench`, `llama-perplexity`, `llama-completion`, clean base `620d2225c` | `lab/device-bin/c2/base/` (private) | built; `COMMIT`, `SHA256SUMS` beside them |
| the same, base + `patches/llama.cpp-c2/0001-olmoe-expert-skip.patch` (sha256 `62b02dc0…e9e42a`) | `lab/device-bin/c2/skip/` (private) | built |
| build script (campaign-1 flags: `armv8.2-a+dotprod+fp16`, no i8mm, no OpenMP, KleidiAI off) | `tools/c2_build_android.sh` | works |
| official prompt text, `_benchPrompt` as the app builds it (one user turn, no trailing newline; checked against the app source) | `tools/c2_official_prompt.txt` (702 bytes, sha256 `e62d90f7…390d35`) | usable by any later kit |
| adb driver `tools/c2_bench_device.sh` | deleted 2026-11-14 (lead's decision) | it was unfinished: `--dry` never ran end to end, and its `tg` row parser split on the commas inside llama-bench's quoted `cpu_info` field (found in a host smoke test) |

sha256 of the binaries: base `llama-bench d6a17c83…`, `llama-perplexity 22245441…`, `llama-completion 36d9b2fc…`;
skip `llama-bench 6c32a8c5…`, `llama-perplexity a320bbe4…`, `llama-completion 66654954…` (full values in the
`SHA256SUMS` files).

One finding carries over to any kit. `llama-completion -cnv -st -f c2_official_prompt.txt -n 32 --temp 0` reproduces
the official run's prompt and length on OLMoE. A host smoke run gave 151 prompt tokens with OLMoE's chat
template (Granite's gave 155) and `eval time … / 31 runs`: the CLI times 31 decodes and leaves out the 32nd, which
llamadart runs and the app counts. Its number also leaves out llamadart's time outside the graph (about 1.05 ms
per token in campaign 1, *device*), so a CLI rate would read higher than the app's `decodeTokensPerSec`.
