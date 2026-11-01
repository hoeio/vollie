# On-device benchmark kit (Snapdragon 888 / Galaxy Z Flip 3)

Nothing in `reports/` was measured on a phone in this project. This kit lets anyone with an
SD888-class phone produce the missing numbers and send back one CSV.

## What it measures
`llama-bench` decode (tg64) and prefill (pp64) tok/s for one GGUF, with Vollie's runtime
config: CPU only, flash attention on, q8_0 KV cache, mmap. Default matrix (`threads:cpu_mask:repack`):

| spec | meaning |
|---|---|
| `1:0x80:0` | X1 only |
| `2:0xC0:0` | X1 + one A78 |
| `3:0xE0:0` | X1 + two A78 |
| `4:0xF0:0` | all big cores, pinned (closest to Vollie on an 8 GB phone: repack off) |
| `4:0x0:0` | 4 threads, unpinned (scheduler may use A55s) |
| `4:0xF0:1` | weight repack on (needs ~3.6 GB extra RAM for Granite; may be OOM-killed on 8 GB) |

Override with `MATRIX="4:0xF0:0 3:0xE0:0"`, `REPS=5`. Before each run it waits until the
hottest readable thermal zone is below `COOL_C` (default 42 °C, max 5 min).

## Build the binaries (on a Mac/Linux host with the Android NDK)
```sh
tools/build_android_bench.sh lab/llama.cpp-v080      # -> tools/device-bin/{llama-bench,llama-perplexity,COMMIT}
```
armv8.2-a+dotprod+fp16 (no i8mm: the X1/A78 lack it), static, no OpenMP, KleidiAI off (`KLEIDI=ON` to compare).

## Run
With adb (USB debugging on):
```sh
PPL_FILE=lab/text/wikitext-2-raw/wiki.test.raw tools/bench_device.sh adb lab/models/granite-fold.gguf
```
In Termux (no PC): copy `device-bin/*`, `bench_device.sh`, the model and optionally `ppl.txt`
(wikitext-2 test) into one directory, then `sh bench_device.sh run granite-fold.gguf .`

Close other apps, keep the phone unplugged from fast chargers, screen on, on a table (not in a case).
Takes ~10-20 min for the default matrix.

## Send back
`out/results_<device>_<time>.csv`, `out/device_<time>.txt`, and `out/ppl_<time>.log` if run.

## CSV format
```
device,soc,build_commit,model,model_sha16,threads,cpu_mask,repack,fa,kv_type,test,n_prompt,n_gen,avg_tok_s,stddev_tok_s,temp_start_c,temp_end_c
SM-F711N,SM8350,8642541b...,granite-fold.gguf,a454ea660db7f70c,4,0xF0,0,on,q8_0,tg,0,64,<tok/s>,<sd>,38,47
```
- `test`: `tg` = decode (the number the 20 tok/s goal refers to), `pp` = prefill.
- `temp_*`: hottest readable thermal zone before/after, °C (0 if unreadable).
- The `ppl` log's "Final estimate" over 20 chunks should match the host value for the same
  chunks to ~±0.5%; a larger gap means a kernel difference worth reporting.
