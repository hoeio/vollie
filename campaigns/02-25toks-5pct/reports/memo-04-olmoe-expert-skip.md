# Memo 04: expert skip for OLMoE, a host instrument for the PPL frontier

Agent model: claude-opus-5-5

Date 2026-11-13. Author: Tom (code). New patch series `patches/llama.cpp-c2/`, first patch
`0001-olmoe-expert-skip.patch` (base `620d2225c`), knobs in [its README](../../../patches/llama.cpp-c2/README.md), sweep
script `tools/c2_skip_sweep.sh`. **No speed is stated anywhere.** Every PPL here is *host* (Apple M4 Pro, 4 threads,
while another long run used the machine) and covers **4 chunks only. That makes it a smoke check, not a result.**
It shows that the knob works and roughly where the settings fall. Nothing here is measured against the budget.

## 1. The rule

On OLMoE the 8 selected weights are raw softmax probabilities over 64 experts (`norm_w = false`), so their sum S
is below 1 and changes from token to token. Campaign 1's absolute permille cut (`v < t`) therefore means a
different cut on every token here (memo-02). The new rules compare each weight only with the token's own weights,
so a value means the same on any token and on any architecture:

- **SUM t:** skip if v < t/1000 × S. On top-k-renormalised weights (Granite, S = 1) this is exactly campaign 1's rule,
  so the Granite thresholds keep their meaning.
- **MASS p:** rank by v and skip each expert once the higher-ranked ones already hold p/1000 of S (cumulative mass).
- **MAX t:** skip if v < t/1000 × v_max (relative to the largest weight). Added because it costs one term.
- **MINK** (default 1): the top expert is never skipped, so no token loses its whole MoE output.

**No renormalisation.** The kept weights keep their values, and a skipped expert adds nothing. The model was trained
with an unnormalised mixture, so dividing by the kept mass would change what the remaining weights mean. A
renormalising variant (scale by S / S_kept) is not built. It is worth adding only if the frontier shows a large
loss of output scale.

## 2. Where the bytes are saved, and why 0001 is not needed

The stock CPU `MUL_MAT_ID` (`ggml-cpu.c`, `ggml_compute_forward_mul_mat_id`) already works per expert:
- thread 0 builds `matrix_row_counts[n_as]` and a (slot, token) row map from `ids`;
- every thread then loops over the experts and skips those with `cne1 == 0` (`continue`), so their weight slice
  `src0->data + cur_a*nb02` is never touched.

The patch leaves the stock map in place and rebuilds it without the skipped pairs, in the same order, on thread 0
before the stock barrier. Each skipped pair's output row is set to zero, so the `ffn_moe_weighted` MUL and the
`ADD` chain see 0 × w. The same mask is computed from the same weights for up, gate and down, so all three skip the
same experts. In decode (one token) a skipped expert ends up with a count of 0, so none of its 2.70 MB (gate, up
and down, Q3_K, *estimated* from shapes) is read or computed. In a 512-token batch an expert is still read if any
token keeps it. The counter reports both cases.

The weights reach the op through `src[3]` of each expert `MUL_MAT_ID`. libllama sets it in `build_moe_ffn` only when a
skip knob is > 0, and a stock graph never has a `src[3]` there. With the knobs unset the graph is the stock graph
and the CPU code takes the stock path. Nothing from campaign 1's `0001` is needed: its graph lookahead by tensor
name is replaced by the explicit `src[3]`. Limits: only the generic CPU path skips. Repacked expert types and other
backends ignore `src[3]` and compute everything, which gives correct results with no skip. OLMoE's Q3_K experts are
not repacked.

## 3. Counter

`VOLLIE_SKIP_STATS=1` prints one line at exit, for example (SUM=65, 4 chunks, *host*):

```
c2-skip: sum=65 max=0 mass=0 mink=1 tokens=2050 pairs=762600 pairs_skipped=10.5271% expert_bytes_skipped=10.5271%
  MB_per_token_selected=335.217 MB_per_token_skipped=35.289 batch_bytes_unread=2.4596%
```

`expert_bytes_skipped` is the per-token fraction, which is what decode saves. `batch_bytes_unread` is what the
running batch did not read: at `-ub 1` it equals the skipped fraction (13.77 % = 13.77 %, MASS=900), and at
`-b 512` it is much smaller (2.29 %). `MB_per_token_selected` is 335 rather than the decode figure of 346 MB because
`llama-perplexity` runs the last layer only on the scored half of each chunk (seen in a debug print: 45 expert ops
at 512 tokens and 3 at 256). The `tokens` count includes 2 warm-up tokens.

## 4. What was checked (all *host*, `-t 4`, `--chunks 4`, `-c 512`, served file `olmoe-Mi.gguf`)

- `git apply --check` on the clean base tree: applies. Build with the flags of `tools/reproduce.sh`
  (`-DGGML_NATIVE=OFF -DGGML_CPU_ARM_ARCH=armv8.2-a+dotprod+fp16`, Metal off): no errors.
- **Knob unset, patched against unpatched (`lab/llama.cpp-base`), per-chunk PPL:**

| batch | unpatched | patched | |
|---|---|---|---|
| `-b 512` | 8.9646, 10.4244, 9.2074, 9.4783 | 8.9646, 10.4244, 9.2074, 9.4783 | identical |
| `-b 512 -ub 32` | 8.9131, 10.3435, 9.0932, 9.3987 | 8.9131, 10.3435, 9.0932, 9.3987 | identical |
| `-b 512 -ub 1` | 8.9550, 10.4710, 9.1889, 9.4713 | 8.9550, 10.4710, 9.1889, 9.4713 | identical |

  The `-b 512` values also equal the first four chunks of R50's served-file log (8 threads). A knob set to `0`
  behaves as unset (`VOLLIE_SKIP_SUM=0`: same four values).
- **Knob set (smoke, `-b 512`; 4-chunk stock 9.4783):**

| setting | expert bytes skipped | PPL (4 chunks) | vs 4-chunk stock |
|---|---|---|---|
| MASS=950 | 1.99 % | 9.4891 | +0.1 % |
| SUM=50 | 3.25 % | 9.6129 | +1.4 % |
| MAX=200 | 5.52 % | 9.6632 | +1.9 % |
| SUM=65 | 10.53 % | 9.8939 | +4.4 % |
| MASS=900 | 13.77 % | 10.1338 | +6.9 % |
| MAX=300 | 17.55 % | 10.7903 | +13.8 % |
| MASS=850 | 20.82 % | 10.4613 | +10.4 % |
| SUM=80 | 23.76 % | 10.7731 | +13.7 % |
| MASS=800 | 28.26 % | 11.2311 | +18.5 % |
| MAX=400 | 33.94 % | 13.3489 | +40.8 % |
| MASS=700 | 40.90 % | 12.9672 | +36.8 % |
| SUM=100 | 41.91 % | 12.8539 | +35.6 % |

  MASS=900 at `-ub 32` and `-ub 1` also runs (13.78 % and 13.77 % skipped; per-chunk values differ from `-b 512`,
  as stock values already do between batch sizes). Reading of the smoke values, not a result: SUM and MASS lie
  close together and MAX is worse. On these 4 chunks the +5 % line falls near 10 % of expert bytes. OLMoE's
  experts look much less skippable than Granite's (campaign 1: threshold 130 gave +3.67 % over the full set).
  The full sweep decides this.

## 5. Not checked

- Full-set PPL of any setting, and full-set identity with the knobs unset (the sweep's `stock` row is that check:
  it must reproduce 11.7146).
- `en.txt` with skip (campaign 1: skip raised `en.txt` PPL much more than wikitext-2).
- Co-application with campaign 1's `0001`, which also rewrites `MUL_MAT_ID`. The c2 series is meant to stand alone.
- Granite and LFM2.5 (the rules are generic, but they were run only on OLMoE). Any phone build or speed.

## 6. The sweep (to be started by the lead after the LFM2.5 reference run ends)

```sh
git -C lab/llama.cpp-base worktree add --detach ../llama.cpp-c2 620d2225cc436ab0b001e3200e590e3b286ef9ab  # if absent
git -C lab/llama.cpp-c2 apply "$PWD/patches/llama.cpp-c2/0001-olmoe-expert-skip.patch"                   # if not applied
# build: the cmake flags of tools/reproduce.sh, target llama-perplexity, build dir lab/llama.cpp-c2/build
nohup caffeinate -i sh tools/c2_skip_sweep.sh lab/llama.cpp-c2 lab/olmoe/olmoe-Mi.gguf <wiki.test.raw> lab/logs/c2skip \
  > lab/logs/c2skip.out 2>&1 &
```

Default settings (about 5 % to 40 % of expert bytes, by the smoke values): `stock SUM=55 SUM=65 SUM=80 SUM=100
MASS=925 MASS=900 MASS=850 MASS=800`. The list is dense at the low end because the budget is likely to sit there. The
`stock` row (knobs unset, patched build) is the full-set identity check and the reference for "% over stock". The
script is resumable: rerunning it skips finished logs and prints the table (setting, PPL ±, % over stock, % pairs
and % bytes skipped, MB per token). R50 took about 66 min per OLMoE pass at 8 threads (*host*), so 9 settings take
about 10 h (*estimated*). If the frontier crosses the budget between two points, add settings by naming them on the
command line.
