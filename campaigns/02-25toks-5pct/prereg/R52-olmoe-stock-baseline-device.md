# Pre-registration R52: OLMoE on the stock runtime, device baseline (kit r17)

Committed and pushed **before** the APK is handed to the owner. Do not edit afterwards; changes go into
`R52-amendment-1.md`. No tag (owner's wish); the commit hash and push time show the order.

- **Date / author:** 2026-11-14, clyde (lead).
- **Campaign / model / file sha256:** campaign 2; OLMoE-1B-7B-0125-Instruct; `olmoe-Mi.gguf`,
  `5abf710b775cd64fd89d74a783c8a7056a409ba795cd401b9dd448ef8e444abb` (downloaded by the kit from
  `HoeioUser/vollie2` @ `e893e8a6`, hash checked on the phone).
- **Agent models** (full IDs): clyde `claude-fable-5-2`; bonnie `claude-opus-5-5` (memo-08, the bands);
  tom `claude-opus-5-5` (memo-09, memo-10, the kit); annie `claude-sonnet-5-5`.
- **Build under test:** APK `vollie-c2-1.17.0+17-research-r17.apk`, 17,098,002 bytes, sha256
  `5b7ae90614dee32b0b2eb5ae5a7a7db0923b0b16aae8bae36135fb3393994f27`, package `com.insaneutils.vollie.c2`
  (installs beside the owner's app), release mode, debug-signed. Runtime: llama.cpp `620d2225c` +
  `patches/llama.cpp-c2/0001` with **every knob unset**, which R51's identity row showed to be the stock
  arithmetic on the host. `libllama.so` `fadab87e…275af2e45`, CPU module `80e614a7…a20ba26d`. App patch:
  `patches/app-c2/0001-kit-r17.patch`.
- **Device:** the owner's Galaxy Z Flip 3 (Snapdragon 888, 8 GB), the phone of campaign 1.
- **Labels:** every number here is *estimated* (memo-08 §1, §3, §9) until a result block says *device*.

## 1. What is tested

No lever. The served file on the unmodified runtime, to (a) show that OLMoE loads and generates through the
app, (b) give the first *device* number of campaign 2, (c) test the cost model on a second architecture.

Kit run list as built ([memo-10](../reports/memo-10-kit-r17.md)): 1 cold official run (reported, not scored);
group `stock`: official 32 × 5, repack **on**; 2 runs with per-token times; group `norepack`: official 32 × 2
with repack **off** (model reloaded). App thread pick (3 threads expected), the kit's cool-down gate before
each timed run, prompt-prefix reuse off (every run prefills the whole prompt; the decode rate is what is
scored). Official prompt text as in campaign 1, 32 generated tokens, greedy. The rendered prompt length in
OLMoE tokens is unknown until the report (Granite's was 155).

## 2. Budget accounting

| item | PPL | % over stock |
|---|---|---|
| stock served file (R50, *host*, repack on) | 11.7146 | 0 |
| config under test | 11.7146 (the stock arithmetic; no numerics change) | 0 |
| budget (stock × 1.05) | 12.3003 | |
| hard ceiling (F16 × 1.20) | 13.1304 | |

Identity: the five `stock` runs and the two per-token runs must return the same reply text (greedy, same
kernel path). The `norepack` runs may differ from them in text (another kernel path, R51 amendment 2); they must
equal each other.

## 3. Prediction

Quantity: the kit's printed `decodeTokensPerSec` (the campaign's metric), median of the five `stock` runs.

| group | steady ms per token (central, likely band) | printed tok/s (central, band) |
|---|---|---|
| `stock`, repack on | 69.4 (63.5 to 78.9) | **13.8 (12.1 to 15.0)** |
| `norepack` | 71.6 (65.6 to 81.5) | 13.3 (11.7 to 14.5) |

Conversion (memo-08 §3): printed ≈ 0.995 × 31,000 / (32 × (T + 0.3) + 12), fault-free tokens only. Second
prediction: `norepack` is slower than `stock` by 0 to 5 ms per token (central 2.2). With two runs this is
resolvable only if the difference exceeds the run-to-run spread; otherwise "unresolved".

Resolution: single phone runs vary by about ± 20 % (campaign 1); the median of five warm, gated runs is the
scored value. Where per-token times exist, the steady token (median of tokens 4 to 32 of the two per-token
runs) is reported against the ms band as well.

## 4. Pass rule and kill rule

- **Hit:** the median of the five `stock` runs lies in 12.1 to 15.0 tok/s.
- **Miss, cost model too pessimistic:** above 15.0. **Miss, too optimistic:** below 12.1. Either is written as
  a miss; Bonnie refits the model before the bands of the next round are written.
- **Void (kit, not model):** the model hash in the report differs; a library hash differs from this file; the
  report lacks the five `stock` runs; the reply texts within `stock` differ; fewer than three `stock` runs meet
  hygiene (§6). A void round is repeated after the kit is fixed.
- **Not a verdict on the goal:** nothing in this round can pass or fail 25 tok/s.

## 5. Claim rule

Not a goal claim.

## 6. Hygiene counts

Per timed run, read with `tools/c2_research_parse.py`: big-core clock at the end of the run recorded; no major
page faults continuing into the decode tokens beyond the first three; the cool-down gate passed (not timed
out). Runs that fail are listed and excluded from the median; the cold run is always excluded. `repack_loaded`
must be true in `stock` and false in `norepack`.

## 7. What a miss means

- Hit or pessimistic miss: the lever kit (exact stack, per-lever A/B, skip `MASS=900`) is the next round; its
  bands are written from the measured baseline, not from memo-08's centre.
- Optimistic miss (slower than 12.1): find the term first (clock, faults, attention at the real prompt length,
  thread count) from the report's side data; the lever kit follows only when the baseline is understood.
- Load failure or out-of-memory with repack on: the next kit runs repack off and the governing reference
  becomes R51 amendment 2's (`-nr`) path.
