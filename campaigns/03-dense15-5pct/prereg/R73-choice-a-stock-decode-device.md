# Pre-registration R73: first real decode on choice A (stock, three subjects; c3 decode kit r39)

This file is committed and pushed **before** the APK is uploaded, and it is not edited afterwards. Any change goes
into `R73-amendment-1.md`. The tag `prereg-R73` is the owner's to create.

- **Date / author:** 2026-11-28, clyde (lead).
- **Why this round exists:** R77 (`reports/R77-result.md` §5) left the deciding question as choice A's real decode rate, which
  no probe cell can give. The owner chose option 1 (2026-11-28, "1로 가죠"): move to R73, the round reserved since 2026-11-22
  for the first real decode. The device hold (owner, 2026-11-22) is lifted for this round by that decision.
- **Research method:** pre-2026-11-23 (owner, 2026-11-26). Bands are the lead's from committed work; no new researcher instance.
- **Agent models:** lead (clyde) `claude-opus-5-5`; bonnie `claude-fable-5-2` (`memo-13-work/bonnie-cpu-ladder.md`, the path-A
  stock estimate used below); ebony `claude-fable-5-2` (no input this round); tom `claude-opus-5-5` (kit r39,
  `memo-13-work/tom-r39-build.md`); annie `claude-sonnet-5-5` (books).
- **Subjects:** Yi-1.5-6B-Chat, Llama-2-7b-chat, DeepSeek-LLM-7B-chat, the frozen served files (S3 freeze). Triples are
  Yi / Llama-2 / DeepSeek.
- **Arm:** choice A only (CAMPAIGN.md 2026-11-24): the app's own load rule (`lab/app-c3/lib/device.dart:591`,
  `assistant.dart:456-467`; shipped app the same), which for these files on 7.75 GB gives no copy: plain mapped rows, q8_0 KV.
  `VOLLIE_EXEC_MAP=1` is kept because the shipped Vollie 2 sets it for every model (bit-identical). No lever is tested.

## 0. Build under test

Kit r39 (`memo-13-work/tom-r39-build.md`), `lab/dist-c3/r39/`; sha256 checked by the lead before this commit:

| item | sha256 |
|---|---|
| `vollie-c3-decode-r39.apk` (16,706,066 B, versionCode 39, same certificate as r38) | `ee6777653a2b07e7667e8065ba495231f8222bb90a6b677a3d1e49c114979470` |
| `BUILD-r39.txt` | `b88911a6cfd219e4a067785ca96c37f08aff6c47135ba3df66fc48c29434198d` |
| `SHA256SUMS` | `2977171c7f08668ab5836691f203539e07ca264c6c94fc8155b7ad18f013816d` |
| `patches/llama.cpp-c3/0013-vocab-control.patch` | `7fef37cdb28ee8d1996c2efe33577dfa95d92d46f20a069a42d363289c257659` |
| `patches/app-c3/0009-kit-r39-decode.patch` (on 0008) | `7f7c9c3f034aa599c28e269a860000c47235d4bc8985e82558e67693cd83b80a` |
| `tools/c3_decode_parse.py` | `9b3a4c78f0b382da18811fcc6f7ff56e3ac409844f25cf813ed2cc29804624b9` |
| `libllama.so` / `libapp.so` | `db62e5cf…cbe4` / `a8ba9a19…c487` (full values in tom-r39-build) |
| CPU module (unchanged since r33) | `46e0735d2824ba6f3a156e219e058a701f6024fdbecbb2ef937b3127115a9694` |

Served files (downloaded by the kit from public `HoeioUser/vollie3`, sha256-checked before load):

| file | bytes | sha256 |
|---|---|---|
| `yi15-6b/yi15-6b-q4_0.gguf` | 3,479,326,848 | `702cc2ebaa3bd4933598e25b556961b95ac260ed15df74fa09df2b2b7b48a299` |
| `llama2-7b/llama2-7b-q4_0.gguf` | 3,825,808,512 | `8eb4952f2592fdfc1288695f6b85aaf48d5628ec5d2c75f9387d3144652ad4bf` |
| `deepseek7b/deepseek7b-q4_0.gguf` (unmodified, no BOS change, owner 2026-11-22) | 4,000,061,152 | `f403ddc298020af0fe8dfbbfec9a74cd3f8951938d6c21540054d85f09315118` |

No numerics change: per-chunk PPL equals the freeze build at `-b 512`, `-ub 32`, `-ub 1` on Yi (*host*). Patch 0013 makes Yi's
`<|im_start|>` one token only while Yi loads (owner, 2026-11-24); the space piece after special tokens is left as llama.cpp does it
(not covered by the owner's decision; noted, not changed). Prefix reuse is off (the vendored llamadart re-ingests an exact replay).

## 1. Device state and run list

- Power saving off, charger out, deep-sleeping label recorded, at least 30 min rest, screen off, no other probe first; the three
  files downloaded before the session (r39 download button).
- Per model, in the order Yi, Llama-2, DeepSeek: load; 2 unscored warm-ups; **10 gated `official32` runs** (r16 gate, hold, kv
  clear); 1 unscored `profiled32` run. 4 threads on cpus 4-7 (`VOLLIE_POOL=1`, `VOLLIE_CPUMASK=0x10,0x20,0x40,0x80`), nCtx 512,
  official prompt, 32 tokens, greedy, the model's chat template, DeepSeek prompt start a.
- Metric: the app's `decodeTokensPerSec` per run; per model the **median of the valid claim runs**.

## 2. Predictions (*estimated*; P = lead's probability the result is inside)

Bonnie's path-A stock (memo-13 `bonnie-cpu-ladder.md`, from one-thread ratios): 5.07 / 4.59 / 4.52 tok/s at full clocks. R77's
descriptive four-thread cells (`R77-result.md` §4, void) put ggml plain at 0.89 to 1.02 of the repack-type stock, which would
move choice A toward the packed stock (6.90 / 6.21 / 6.22 ggml, *device*-derived). The lead's centre sits between the two.

| # | quantity | band | P |
|---|---|---|---|
| S1 | Yi median decode tok/s | 5.8 (4.6 to 7.0) | 0.75 |
| S2 | Llama-2 median | 5.2 (4.1 to 6.3) | 0.75 |
| S3 | DeepSeek median | 5.2 (4.1 to 6.3) | 0.75 |
| S4 | the three medians sit nearer the packed stock than Bonnie's path-A estimate (each median ≥ the midpoint 5.99 / 5.40 / 5.37) | yes on all three | 0.4 |
| S5 | spread of the valid claim runs per model (max ÷ min − 1) | ≤ 10 % on each | 0.6 |
| S6 | Llama-2 and DeepSeek slower than Yi (both medians below Yi's) | yes | 0.8 |
| S7 | at least 6 valid claim runs per model | yes on all three | 0.8 |

## 3. Rules (scored as written)

- **Re-base:** the measured medians replace every choice-A stock estimate in the campaign's ladders (Bonnie's path-A column is
  re-based by ratio). This is the round's purpose; no lever is adopted or killed.
- **Path-A model check:** each median within ±12 % of 5.07 / 4.59 / 4.52 → Bonnie's one-thread-ratio model of path A holds. A
  median above +12 % on two or more subjects → the model is wrong in the direction R77 hinted (four-thread plain ≈ packed); the
  lead re-prices Y1b (parked) and R1 (dead) only if their cost reading depended on the one-thread base.
- **Placement:** worker labels (M / U / P / C, parser thresholds) are **reported per run, not used as a void** (R77 voided every
  four-thread cell on this pattern). A run is reported both in the all-valid median and in a C-only median (runs where every
  worker is C), if at least 3 C-only runs exist.

### 3.1 Void

- APK hash differs from §0, or a served file's sha256 check fails: that model (or the round) void.
- Power saving on, or charger plugged: runs in that state void.
- A claim run whose gate timed out, or that was killed: that run void. A model with fewer than 6 valid claim runs: "not measured".
- Not voids: cool start not met (reported), clock caps (reported per run with the 1 Hz samples), placement labels (above).

## 4. Claim rule and meaning

Not a goal claim. R73 measures choice A's stock decode, the base every campaign-3 lever is priced on. The goal (12 tok/s on each
of the three) is out of reach on present prices whatever R73 reads: even the packed stock is 6.90 / 6.21 / 6.22.

## 5. The run, for the owner

1. After this file is pushed: create tag `prereg-R73` on its commit. The lead uploads the three served files to public
   `HoeioUser/vollie3` (with licence files and a card) and kit r39 to `HoeioUser/vollie-kits` `r39/` (write token per session).
2. Download `r39/vollie-c3-decode-r39.apk`; check sha256 `ee6777653a2b07e7667e8065ba495231f8222bb90a6b677a3d1e49c114979470`;
   install over r38.
3. In the kit, use the r39 download button for the three models (about 11.3 GB on the phone, Wi-Fi). olmoe-Mi must be present.
4. Phone state as §1; then run **c3 decode R73**. About 55 to 75 min (*estimated*); do not use the phone meanwhile.
5. Save the report and send the JSON.
