# memo-10 work file, Clyde: S3 freeze of the three served files (owner's laptop)

Date: 2026-11-22. Author: Clyde (`claude-opus-5-5`). Not committed. Nothing here was run: every time and size is
*estimated* from earlier *host* logs named beside it. The commands are for the owner's laptop (Apple M4 Pro, 24 GB,
about 40 GB free on 2026-11-22). Inputs: memo-06-work/tom-screens.md §5.1 (files, shas, recipe),
memo-08-work/clyde-r72-plan.md §5 (freeze list, DeepSeek BOS), memo-08-work/tom-k1.md §6 (full-set stock runs).

## 1. What is frozen

| subject | publisher repo @ commit | served file (`lab/c3/models/`) | bytes | sha256 | F16 sha256 (screen, deleted) |
|---|---|---|---|---|---|
| Yi-1.5-6B-Chat | `01-ai/Yi-1.5-6B-Chat` @ `771924d1c83d67527d665913415d7086f11ea9c0` | `yi15-6b-q4_0.gguf` | 3,479,326,848 | `702cc2ebaa3bd4933598e25b556961b95ac260ed15df74fa09df2b2b7b48a299` | `7ecff8aaf72278b573d53eb62b7102e28a9f72d3d61960a4d532a3298823b114` |
| Llama-2-7b-chat | `meta-llama/Llama-2-7b-chat-hf` @ `f5db02db724555f92da89c216ac04704f23d4590` (gated) | `llama2-7b-q4_0.gguf` | 3,825,808,512 | `8eb4952f2592fdfc1288695f6b85aaf48d5628ec5d2c75f9387d3144652ad4bf` | `175c701f53317f0f209479a3ec3706de06714ea1115d01143ace93fe26bfefdb` |
| DeepSeek-LLM-7B-chat | `deepseek-ai/deepseek-llm-7b-chat` @ `afbda8b347ec881666061fa67447046fc5164ec8` | `deepseek7b-bos-q4_0.gguf` (BOS on) | 4,000,061,152 | `f195d24f5e29884bdfe3a6586bff0115c1988956fabd2cf2d07a3505d35da9ec` | `29b971949796eec79583562e740d861b66680d8f8782aeeaa88c5e04b45bd5b2` (before BOS) |

- Type: Q4_0 body, Q6_K output head (`llama-quantize` default), as screened. R69 C (4.2 GB file-backed) holds all
  three files, and R72 kept Q4_0 by its type filter, so the freeze is Q4_0 for all three. That is the
  clyde-r72-plan §5 rule, now final.
- DeepSeek's served file is the kept `deepseek7b-q4_0.gguf` (`f403ddc2…5118`) with one metadata field set,
  `tokenizer.ggml.add_bos_token` false → true (one byte differs; `lab/c3/logs/bos.log`). It follows the publisher's
  `tokenizer_config.json`. This is the lead's decision. The owner has not confirmed it yet (owner question F1 below).
- Tool tree: `lab/llama.cpp-c3` at `26f78f925` (`620d2225c` + c2 series 0001 to 0009, CPU build of 2026-11-18,
  unchanged since). Binary sha256s: `llama-perplexity` `e5264b7c…470d`, `libggml-cpu` `04ff5998…e7fc1`,
  `libllama` `1c2aba8e…8849`, `llama-quantize` `ca14cc64…894a`. Conversion venv `lab/venv-c3`.
- Text: `lab/c3/text/wiki.test.raw`, sha256 `173c87a5…b9e7dd08`. This is the file campaigns 1 and 2 used (checked
  against the sibling `vollie/lab/text/wikitext-2-raw/wiki.test.raw`). Sanity text: `en.txt`, sha256 `101b2525…96d8`.

## 2. A trap found while preparing this (why the commands are a script)

The F16 sha256 depends on the download directory's name. `convert_hf_to_gguf.py` writes `general.name` from the
directory's basename. The kept DeepSeek file says `'Deepseek7B'`, from `c3/dl/deepseek7b`.

The 2026-11-22 DeepSeek re-download went to `c3/dl/deepseek7b-r`. It produced a different F16 (`9ff3cf82…a008`) and
a different Q4_0 (`404e4b0f…baa6`), not the kept file. So the 40-chunk F16-with-BOS value 8.0449 (*host*) came from
an F16 whose metadata differs from the served file's parent. The tensors are very likely the same, but that was
never shown.

The script therefore downloads into `lab/c3/dl/<m>` under the screen's names. Its binding check is that the Q4_0
re-made from the new F16 equals the kept Q4_0 byte for byte. That check proves the F16 reference belongs to the
served file. An F16 sha different from the screen's is only a warning, because the re-made Q4_0 is the real check.

## 3. What is measured, and the lead's decisions on it

| run | setting | needed for | decision |
|---|---|---|---|
| served Q4_0 | `-b 512`, all chunks | **budget** at `-b 512` = stock × 1.05; quality gate | required |
| served Q4_0 | `-b 512 -ub 1`, all chunks | **budget** at `-ub 1` = stock(`-ub 1`) × 1.05; every numerics lever is checked at both | required |
| F16 | `-b 512`, all chunks | **ceiling** = F16 × 1.20 | required |
| F16 | `-b 512 -ub 1`, all chunks | descriptive only | **optional** (`F16_UB1=1`), see below |

- **Ceiling at `-ub 1`.** The lead states the ceiling from F16 at `-b 512` (the gate's setting, COMMON_RULES 2) and
  applies it at both settings. It cannot bind. Served/F16 is 1.024 to 1.037 on 40 chunks (*host*), so stock × 1.05
  is about F16 × 1.08 to 1.09, far below F16 × 1.20. An F16 `-ub 1` run would cost about 6 to 15 hours per model
  (*estimated*) and decide nothing. It is offered, not asked.
- **Stock `-b 512` values that already exist.** Llama-2 7.7435 and DeepSeek-BOS 8.8342 (*host*,
  `lab/c3/logs/ppl-*-q4_0-b512-full.log`) were made by the same binary and text. The script re-runs them under
  the freeze's names. The re-run must reproduce them to all printed digits (CPU, same build). If it does not, stop
  and tell the lead.
- **Lever cells that predate the freeze.** The K1 k,v-dense 40 % full-set cells (Llama-2 +3.40 %, DeepSeek +3.55 %,
  *host*) were measured before this freeze. The budget's rule (stock × 1.05 at each setting) has been written since
  2026-11-18 (CAMPAIGN.md), so the absolute values below are arithmetic on that rule, not a choice. Those cells stay
  host screens and are scored against the frozen absolute budget only in a pre-registered round. K1 is parked
  (R72), so none is due.

## 4. Owner commands (laptop, repository root)

The script is `tools/c3_freeze.sh` (new, uncommitted). Annie commits it with this file before the owner runs it.
Record the commit with `git rev-parse HEAD`.

Per subject it does the following:
1. Checks the served file's sha256.
2. Downloads the publisher's weights at the pinned commit and checks every LFS file's sha256.
3. Converts them to F16 with the screen's recipe.
4. Re-quantises to Q4_0 and requires byte equality with the kept file.
5. DeepSeek only: sets the same BOS field on the F16.
6. Runs the PPL set through `tools/ppl.sh` (ctx 512, `-t 8`, `-ngl 0`, wikitext-2 test plus `en.txt`).
7. Deletes the F16 and prints the budget and ceiling.

It can be resumed: finished logs are skipped. It refuses to download with less than 30 GB free.

```sh
cd ~/Documents/project/activate/vollie2
git pull && git rev-parse HEAD                           # the commit the freeze ran against; send it with the results
mkdir -p lab/text && cp ../vollie/lab/text/en.txt lab/text/en.txt   # en.txt sanity text (copy only; sibling untouched)
hf auth whoami                                           # Llama-2 is gated: the laptop's HF login must have access
# close other heavy apps (F16 is 12 to 14 GB of a 24 GB laptop); then, one subject after another, in the background:
nohup caffeinate -i sh -c 'sh tools/c3_freeze.sh yi15-6b && sh tools/c3_freeze.sh llama2-7b && sh tools/c3_freeze.sh deepseek7b' \
  > lab/c3/logs/freeze-all.out 2>&1 &
# progress:
tail -f lab/c3/logs/freeze-all.out
# optional, after the above, only if the owner wants F16 at -ub 1 too (6 to 15 h per model, estimated; re-downloads):
# F16_UB1=1 sh tools/c3_freeze.sh yi15-6b    (and the other two)
```

If `hf` is not on the PATH, use `lab/venv-c3/bin/hf auth whoami`. The download itself uses the same login.

## 5. Where the outputs go

| output | path (untracked) |
|---|---|
| step log: LFS shas, F16 sha, re-made-Q4_0 check, BOS byte count, RESULT and budget lines | `lab/c3/logs/freeze-<m>.log` (`<m>` = `yi15-6b`, `llama2-7b`, `deepseek7b`) |
| PPL logs (per-chunk running values inside) | `lab/logs/p/ppl-c3f-<m>-f16-b512-{wiki,en}.log`, `ppl-c3f-<m>-q4_0-b512-{wiki,en}.log`, `ppl-c3f-<m>-q4_0-ub1-{wiki,en}.log` (and `-f16-ub1-` if run) |
| driver output | `lab/c3/logs/freeze-all.out` |

What the owner sends back: the commit from the first command, and the last two lines of each `freeze-<m>.log`
(RESULT and budget). The lead can also read the logs directly when the session runs on the laptop.

Then Annie writes one model card per subject from `models/_TEMPLATE.md` (`models/c3-<m>-q4_0.md`), with:
- repo @ commit and the LFS sha256s;
- F16 sha256 (and the BOS note for DeepSeek);
- served sha256 and bytes;
- F16 and served PPL, full set (*host*);
- budget at `-b 512` and at `-ub 1`, and the ceiling, as absolute values;
- served ≤ ceiling;
- en.txt PPL as the sanity check;
- the binary shas of §1.

The freeze is complete when all three cards are committed. Until then no lever round scores PPL against any of the
three.

## 6. Time and disk (*estimated*, from the named *host* logs)

| step | Yi | Llama-2 | DeepSeek | basis |
|---|---|---|---|---|
| download + LFS check + convert + re-quantise | about 20 min | about 25 min | about 20 min | `prep-*.log`: 17, 23, 18 min on 2026-11-21 |
| F16 `-b 512`, full set | 0.7 to 1.5 h | 1 to 3 h | 1.5 to 2.5 h | 40-chunk ETAs 2.6 / 31.5 / 9.4 min (Llama-2's under swap) × 15 to 16 |
| served `-b 512`, full set | about 0.5 h | 0.6 h | 0.55 h | `ppl-*-b512-full.log`: 36 and 32 min |
| served `-ub 1`, full set | 2 to 4 h | 2 to 4 h | 2 to 4 h | `bos-deepseek7b-ub1-40.log`: 459 s per 40 chunks (another harness) → 1.9 h |
| en.txt runs | minutes | minutes | minutes | 57,706 bytes |
| **total, required** | **about 3.5 to 6 h** | **4 to 8 h** | **4.5 to 7.5 h** | **about 12 to 21 h for the three, serial** |
| optional F16 `-ub 1` | 6 to 15 h each | | | 3 to 4 × the served `-ub 1` |

Disk: the peak per subject is the publisher snapshot plus the F16 (Yi 12.1 + 12.1, Llama-2 13.5 + 13.5, DeepSeek
13.8 + 13.8 GB), about 28 GB. The snapshot is deleted right after conversion, then a 3.5 to 4.0 GB re-quantised copy
exists for a minute. With about 40 GB free that leaves about 12 GB. One subject at a time is required; the driver
command above is serial. Nothing in `lab/c3/models/` needs deleting. If space runs short,
`chatglm3-6b-q4_0.gguf` (3.6 GB, excluded by the owner) and `granite7b-q4_0.gguf` (3.8 GB, not in the set) are the
candidates. Deleting them is the owner's call.

## 7. Owner questions this step raises

- **F1. DeepSeek BOS.** Confirm the served DeepSeek file is the converter's output with `add_bos_token` set to true
  (the publisher's `tokenizer_config.json`), one metadata byte changed before the freeze. The alternative is the
  unmodified converter output (`f403ddc2…`, BOS off), whose stock PPL is lower on 40 chunks (8.2231 against 8.3095,
  *host*) but which does not tokenise as the publisher does.
- **F2. F16 at `-ub 1`:** not needed by the lead (§3). Run it only if the owner wants it.
