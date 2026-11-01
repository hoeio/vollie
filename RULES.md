# Rules

The conditions every change to Vollie's serving has to meet, and why. They come from the
project's brief and from what measurements on the phone taught along the way; the experiment
behind each lesson is in `reports/03_final.md` (R-numbers).

## 1. Goal and the one hard limit

- **Goal:** ≥ 20 tok/s writing (decode) on a Snapdragon 888 phone (Galaxy Z Flip 3, 8 GB).
- **Hard limit:** wikitext-2 perplexity at most **10% above the same model's F16 weights**
  (F16 = 9.0438, so ≤ 9.948). It is never traded for speed, and never relaxed quietly.
- **The quality test is fixed:** `llama-perplexity`, wikitext-2-raw test set, context 512, batch 512,
  all chunks (`tools/ppl.sh`). It does not change mid-project to make numbers look better.
- `lab/text/en.txt` (English prose) is measured and reported with every result as a **sanity check**;
  it is not part of the pass/fail gate. Only English is evaluated.
- If the goal looks out of reach, say so with numbers instead of bending a rule.

## 2. The model

- **Model: Granite-4.0-H-Tiny abliterated (Huihui)**, locked: no other model, no architecture
  change, no distillation.
- **The served file is frozen:** `vollie-granite-4.0-h-tiny-d6-e6.gguf` (sha256 `5dc1daa0…`,
  `models/`). Speed has to come from the runtime: kernels, threading, scheduling, how weights are
  read. A different file, or a different model, is proposed to the owner first, never shipped silently.

## 3. Real phones, as they are

- **Fix it in the runtime, never in the user's phone.** No asking users to reboot, close apps, free
  memory or change system settings. A crowded 8 GB phone with ~2 GB of page cache, warm, with
  Android's own threads running, is the condition to serve in. (R10/R11)
- **Measure on the device.** The host hid both large effects found so far: thread-sync cost
  (~0.4 µs per sync on an M4 vs about half of every token on the SD888, R4) and page-cache thrash
  (24 GB of RAM vs ~2 GB of cache, R9). Host speed numbers are relative only.

## 4. Correctness

- **Every runtime change gives identical output to the stock kernels:** `llama-perplexity` per-chunk
  values equal at `-ub 1` (decode), `-ub 32` (small prompt batches) and `-b 512`. The `-ub 32` case
  caught a race the other two missed (R5).
- New behaviour is **off unless switched on** (an environment variable), so stock behaviour stays one
  setting away and every feature can be measured against it.

## 5. Measurement and honesty

- **Label every speed number:** *measured on device*, *measured on host*, or *estimated*. Never present
  a host number or an estimate as a phone number.
- **Compare under the same heat.** Wait for the thermal headroom to drop before each timed run and
  record it (back-to-back tests had halved every number, R5→R6); score a setting by its best run
  (single runs on a warm phone vary ±20%, R14); for small differences use the in-generation A/B,
  which alternates the setting every token (R15).
- **Count what the kernel counts:** major/minor page faults, memory and core clocks next to every
  timing. Fault counts turned "the experts are slow" into "the experts are re-read from flash" (R9).

## 6. How work is recorded

- Keep a change only if it helps and stays within the limit; otherwise revert it and write down why.
- Every result goes into `reports/results.csv` (one row) and `reports/03_final.md` (the R-log),
  failures included. Results are appended; history is not rewritten.
- Small commits with clear messages.
- Anything destructive, costing money, or needing hardware or access not at hand is asked first.

## 7. Shipping

Builds that go to users (APK, Play bundle) are release-signed, keep every native library 16 KB page
aligned (Android 15+), and download the frozen model file pinned to a Hugging Face commit and
verified by SHA-256.
