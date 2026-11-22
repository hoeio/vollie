# c3 memo-07 work file, Bonnie: re-price with measured column frontiers, re-rank, and what 12 needs

Agent model: `claude-fable-5-2` (Bonnie). Date 2026-11-22. Draft for the lead; verdicts and the choice are the
lead's. **Nothing here is measured on the phone, and I ran no model on the host.** R69 has not reported.

Labels: *host* = Tom's final screens, `memo-06-work/tom-screens.md` sections 5 and 6 (commit `a6f6b36`), and his logs
in `lab/c3/logs/` (read-only, 40 chunks of wikitext-2 test, ctx 512; relative only); *device* = a stored phone number, named by R-number;
*published* = a publisher's or paper's number; *estimated* = my arithmetic (script `bonnie_reprice.py` beside this
file, which imports the shapes of `memo-06-work/bonnie_price.py`). **Every tok/s is *estimated* and printed
(0.95 x steady).**

## 0. Summary

1. **With measured frontiers the screened order is Yi-1.5-6B, Mistral-7B-v0.3, ChatGLM3-6B, Qwen2.5-7B,
   Llama-2-7B, DeepSeek-LLM-7B, Granite-7B.** Sparsity now outweighs bytes: Mistral (4.04 GB, s 0.40) prices above
   DeepSeek (3.76 GB, s 0.20) at r 19 and r 24 at both ends of the band, and at r 35 whenever the kernel costs
   k under about 1.8.
2. **The provisional trio no longer follows the rule. Mistral-7B-v0.3 takes rank 2; DeepSeek falls to rank 6.**
   ChatGLM3-6B's eligibility is in doubt (lead, 2026-11-22: stock PPL 33.0, unexplained), so the trio is given
   both ways. **Without ChatGLM3: Yi-1.5-6B, Mistral-7B-v0.3, Qwen2.5-7B; reserves Llama-2-7b, DeepSeek-LLM-7B.**
   With it: Yi-1.5-6B, Mistral-7B-v0.3, ChatGLM3-6B; reserves Qwen2.5-7B, Llama-2-7b.
3. **12 tok/s is priced on one subject only, and only in a corner.** Yi-1.5-6B prints 12 when r is at least
   31.2 GB/s, the 3.4 GB residency step holds (P 0.35, pre-registered), and either the kernel costs k ≤ 1.46 or
   a fourth big core is used. Its top is 11.7 on three cores and 13.7 on four at r 35.
4. **No second subject reaches 12 with a priced lever.** Mistral tops at 9.7 (three cores) and 11.7 (four) at
   r 35 and needs r ≥ 37.1; every other screened model needs r ≥ 40 and a kernel under 1.4 even on four cores.
5. **Ebony and I agree on the cell and on the order** (her `ebony-compute.md` section 3.3, written separately):
   one cell reaches 12 (Yi, fourth core, r ≥ 31 to 32); Mistral and Qwen2.5 are worth more than the Llama-2 class.
6. **At R69's central capacity (C 3.1 GB) no Q4_0 candidate is resident.** The type that fits prices at 3.1 to
   5.6 tok/s. Residency, not the kernel, is the first wall.
7. My memo-06 assumption (s 0.30 to 0.40 for unscreened models) missed low on four of five new screens.

## 1. Inputs and what changed since memo-06

| input | value | label | change from memo-06 section 3.1 |
|---|---|---|---|
| column frontier s, inside stock x 1.045 at both `-b 512` and `-ub 1` | Yi 0.40, Mistral 0.40, Qwen2.5 0.35, ChatGLM3 0.20 (0.35 at `-b 512` alone), DeepSeek 0.20, Llama-2 0.20, Granite-7B 0.15 | *host* | was assumed 0.30 to 0.40 for all but Mistral and Qwen2.5 |
| s for the 21 unscreened models | 0.15 to 0.40 (the range of the seven); rank at the median 0.20 | **assumed** | was 0.30 to 0.40 |
| compute c | 40.4 GB/s of Q4_0 on X1 + 2 A78 = 71.8 G weights/s; 50.3 GB/s = 89.4 G on four cores | *estimated* (R69 pre-registered centres) | was 69.9 G; Ebony's and the lead's value, one convention for the three files |
| fixed cost O | 8 ms (6 to 12) | *estimated* | was 5 to 10 |
| read rate r | 19, 24 (R69's centre), 35 GB/s | *estimated* | r 24 added |
| kernel cost k (K1, per touched weight) | 1.55 to 2.2 | *estimated* device price; *host* 1.53 to 1.64 | none |
| capacity C | 3.1 GB (2.6 to 3.8); P(3.4 step held) 0.35, P(4.2) 0.12, P(4.6) 0.06 | *estimated* (R69 pre-registration) | new in the price |
| Q4_0 / F16 PPL | Yi 1.024, ChatGLM3 1.012, DeepSeek 1.037, Llama-2 1.030, Granite-7B 1.026 | *host* | was unmeasured; all far inside F16 x 1.20, so Q4_0 is allowed for all five |

Formula (memo-03 section 9, unchanged): steady ms = max((B_H + B_L (1 − 0.889 s)) / r, (W_H + W_L (1 − s) k) / c)
+ O; printed = 950 / steady ms; a model's price is the better of stock and the column kernel. Band: low end
(k 2.2, O 12), high end (k 1.55, O 6). Check: with memo-06's inputs put back the script reproduces memo-06's Yi
row (11.5).

The grid is coarse where it matters: DeepSeek and Llama-2 pass at 20 % and fail at 30 %, with no 25 % cell;
Granite passes at 15 % and fails at 20 %. A straight line between the two cells puts the +4.5 % crossing near
27 % (DeepSeek), 22 % (Llama-2) and 16 % (Granite-7B) (*estimated* from *host* cells, not a measurement).

Because c is now above r 35, **every stock row is byte-bound at all three read rates**, so the stock order is the
order of bytes; in memo-06 the r 35 column was compute-bound.

## 2. Criterion 1, whole pool, re-ranked

Order: midpoint of the lever band at r 19 plus that at r 35 (memo-06 used the top; the midpoint is used now
because the low end depends on s). Unscreened models are placed at s 0.20 and their band runs from s 0.15 to
0.40; the last column is where each would stand if it screened at 0.40.

| # | model | family | Q4_0 GB/token | s | stock r 19 / 24 / 35 | levers r 19 | levers r 24 | levers r 35 | r 35, 4 cores | unscreened: at s 0.40, r 19 / r 35 |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | Yi-1.5-6B-Chat | 01.AI | 3.33 | **0.40** host | 5.2 / 6.5 / 9.2 | 7.4 to 7.7 | 8.1 to 9.6 | 8.9 to 11.7 | 9.8 to 13.7 |  |
| 2 | Mistral-7B-Instruct-v0.3 | Mistral | 4.04 | **0.40** host | 4.3 / 5.4 / 7.7 | 6.3 to 6.6 | 6.7 to 8.2 | 7.5 to 9.7 | 8.2 to 11.7 |  |
| 3 | chatglm3-6b | Zhipu | 3.43 | **0.20** host | 5.0 / 6.3 / 9.0 | 5.8 to 6.1 | 6.1 to 7.6 | 8.6 to 9.1 | 8.6 to 10.8 |  |
| 4 | Qwen2.5-7B-Instruct | Alibaba Qwen | 4.12 | **0.35** host | 4.2 / 5.3 / 7.6 | 5.6 to 5.8 | 6.4 to 7.3 | 7.3 to 9.0 | 7.8 to 10.4 |  |
| 5 | Llama-2-7b-chat | Meta | 3.75 | **0.20** host | 4.6 / 5.8 / 8.3 | 5.4 to 5.6 | 5.6 to 7.0 | 8.0 to 8.4 | 8.0 to 9.8 |  |
| 6 | deepseek-llm-7b-chat | DeepSeek | 3.76 | **0.20** host | 4.6 / 5.8 / 8.2 | 5.3 to 5.5 | 5.7 to 6.9 | 8.0 to 8.4 | 8.0 to 9.9 |  |
| 7 | OLMo-7B-0724-Instruct | AI2 | 3.81 | 0.15 to 0.40 assumed | 4.6 / 5.7 / 8.1 | 5.1 to 6.9 | 5.6 to 8.6 | 7.9 to 10.2 | 7.9 to 12.2 | 6.9 / 10.2 |
| 8 | granite-7b-instruct | IBM | 3.75 | **0.15** host | 4.6 / 5.8 / 8.3 | 5.2 to 5.3 | 5.6 to 6.7 | 8.0 to 8.4 | 8.0 to 9.2 |  |
| 9 | calm2-7b-chat | CyberAgent | 3.86 | 0.15 to 0.40 assumed | 4.5 / 5.6 / 8.0 | 5.0 to 6.7 | 5.5 to 8.4 | 7.8 to 10.2 | 7.8 to 12.0 | 6.7 / 10.2 |
| 10 | A.X-3.1-Light | SKT | 3.96 | 0.15 to 0.40 assumed | 4.4 / 5.5 / 7.8 | 4.9 to 6.5 | 5.4 to 8.1 | 7.6 to 10.0 | 7.6 to 11.5 | 6.5 / 10.0 |
| 11 | Olmo-3-7B-Instruct | AI2 | 3.98 | 0.15 to 0.40 assumed | 4.4 / 5.5 / 7.8 | 4.8 to 6.5 | 5.3 to 8.1 | 7.6 to 9.9 | 7.6 to 11.5 | 6.5 / 9.9 |
| 12 | OLMo-2-1124-7B-Instruct | AI2 | 3.98 | 0.15 to 0.40 assumed | 4.4 / 5.5 / 7.8 | 4.8 to 6.4 | 5.3 to 8.1 | 7.6 to 9.9 | 7.6 to 11.5 | 6.4 / 9.9 |
| 13 | salamandra-7b-instruct | BSC | 4.05 | 0.15 to 0.40 assumed | 4.3 / 5.4 / 7.7 | 4.7 to 6.0 | 5.5 to 7.5 | 7.4 to 10.1 | 7.4 to 10.6 | 6.0 / 10.1 |
| 14 | Falcon3-7B-Instruct | TII | 4.07 | 0.15 to 0.40 assumed | 4.3 / 5.3 / 7.6 | 4.7 to 6.3 | 5.2 to 7.9 | 7.4 to 9.7 | 7.4 to 11.3 | 6.3 / 9.7 |
| 15 | Tri-7B | Trillion Labs | 4.07 | 0.15 to 0.40 assumed | 4.3 / 5.3 / 7.6 | 4.7 to 6.2 | 5.2 to 7.8 | 7.4 to 9.8 | 7.4 to 11.1 | 6.2 / 9.8 |
| 16 | Minerva-7B-instruct-v1.0 | Sapienza | 4.10 | 0.15 to 0.40 assumed | 4.2 / 5.3 / 7.6 | 4.7 to 6.4 | 5.2 to 8.0 | 7.4 to 9.6 | 7.4 to 11.4 | 6.4 / 9.6 |
| 17 | internlm2_5-7b-chat | Shanghai AI Lab | 4.24 | 0.15 to 0.40 assumed | 4.1 / 5.1 / 7.4 | 4.6 to 6.1 | 5.0 to 7.6 | 7.1 to 9.3 | 7.1 to 10.9 | 6.1 / 9.3 |
| 18 | EXAONE-3.5-7.8B-Instruct | LG | 4.27 | 0.15 to 0.40 assumed | 4.1 / 5.1 / 7.3 | 4.5 to 6.0 | 5.0 to 7.6 | 7.1 to 9.3 | 7.1 to 10.8 | 6.0 / 9.3 |
| 19 | openPangu-Embedded-7B | Huawei | 4.33 | 0.15 to 0.40 assumed | 4.0 / 5.0 / 7.2 | 4.5 to 5.9 | 4.9 to 7.3 | 7.0 to 9.3 | 7.0 to 10.5 | 5.9 / 9.3 |
| 20 | Ministral-8B-Instruct-2410 | Mistral | 4.35 | 0.15 to 0.40 assumed | 4.0 / 5.0 / 7.2 | 4.5 to 5.9 | 4.9 to 7.3 | 7.0 to 9.2 | 7.0 to 10.5 | 5.9 / 9.2 |
| 21 | Llama-3.1-8B-Instruct | Meta | 4.36 | 0.15 to 0.40 assumed | 4.0 / 5.0 / 7.2 | 4.4 to 5.9 | 4.9 to 7.3 | 7.0 to 9.2 | 7.0 to 10.5 | 5.9 / 9.2 |
| 22 | kanana-1.5-8b-instruct | Kakao | 4.36 | 0.15 to 0.40 assumed | 4.0 / 5.0 / 7.2 | 4.4 to 5.9 | 4.9 to 7.3 | 7.0 to 9.2 | 7.0 to 10.5 | 5.9 / 9.2 |
| 23 | Apertus-8B-Instruct-2509 | Swiss AI | 4.37 | 0.15 to 0.40 assumed | 4.0 / 5.0 / 7.2 | 4.4 to 5.9 | 4.9 to 7.3 | 6.9 to 9.1 | 6.9 to 10.5 | 5.9 / 9.1 |
| 24 | Qwen3-8B | Alibaba Qwen | 4.42 | 0.15 to 0.40 assumed | 4.0 / 4.9 / 7.1 | 4.4 to 5.7 | 4.8 to 7.2 | 6.9 to 9.1 | 6.9 to 10.3 | 5.7 / 9.1 |
| 25 | Teuken-7B-instruct-v0.6 | OpenGPT-X | 4.46 | 0.15 to 0.40 assumed | 3.9 / 4.9 / 7.0 | 4.3 to 5.5 | 4.9 to 6.9 | 6.8 to 9.2 | 6.8 to 9.8 | 5.5 / 9.2 |
| 26 | MiniCPM4-8B | OpenBMB | 4.51 | 0.15 to 0.40 assumed | 3.9 / 4.8 / 6.9 | 4.3 to 5.8 | 4.7 to 7.3 | 6.7 to 8.8 | 6.7 to 10.4 | 5.8 / 8.8 |
| 27 | granite-3.3-8b-instruct | IBM | 4.65 | 0.15 to 0.40 assumed | 3.8 / 4.7 / 6.7 | 4.2 to 5.7 | 4.6 to 7.1 | 6.6 to 8.5 | 6.6 to 10.2 | 5.7 / 8.5 |
| 28 | c4ai-command-r7b-12-2024 | Cohere | 4.79 | 0.15 to 0.40 assumed | 3.7 / 4.6 / 6.6 | 4.0 to 5.2 | 4.6 to 6.5 | 6.4 to 8.6 | 6.4 to 9.2 | 5.2 / 8.6 |

Reading:
- **Screened models (rule-firm):** Yi-1.5-6B > Mistral-7B-v0.3 > ChatGLM3-6B > Qwen2.5-7B > Llama-2-7b >
  DeepSeek-LLM-7B > Granite-7b. Llama-2 and DeepSeek are tied within 0.1 everywhere.
- **Where the order turns.** At s 0.20 the column kernel costs more compute than it saves (it breaks even at
  k = 1 / (1 − s) = 1.25; at s 0.40 the line is 1.67). So the 20 % models gain only while byte-bound (r 19: +0.7
  to +1.0 tok/s) and nothing at r 35. Mistral and Yi gain at every r when k is near 1.55, and at r 19 and r 24 at any k in the band.
- **Mistral against DeepSeek:** ahead by 1.0 to 1.3 at r 19 and r 24 at both ends of the band; at r 35 ahead by
  1.3 at k 1.55 and behind by 0.5 at k 2.2 (there both fall back to stock and bytes decide). The crossover is
  near k 1.8.
- **ChatGLM3-6B: eligibility in doubt (lead, 2026-11-22).** Its stock PPL is 33.0 (*host*), about 4.5 times
  its peers, unexplained and not cross-checked against the publisher's code. Tom's prefix check made it worse
  (8 chunks: 48.26 without `[gMASK]sop`, 53.36 with; *host*), so the prefix is not the cause. Its curve is flat
  and not monotone (`-ub 1`: 30 % +5.05 %, 35 % +4.74 %), and it passes 35 % at `-b 512` alone. It is priced at
  0.20. At that value it leads Qwen2.5 by 0.2 to 0.3 at r 19 and by 0.1 to 1.3 at r 35, and trails it at the low
  end of r 24. At 0.35 it would price 6.8 to 7.1 at r 19 and 8.6 to 10.6 at r 35.
- **The two rankings of the screened models** (rule-firm; the table's numbering includes ChatGLM3):

  | rank | with ChatGLM3 | without ChatGLM3 |
  |---|---|---|
  | 1 | Yi-1.5-6B-Chat | Yi-1.5-6B-Chat |
  | 2 | Mistral-7B-Instruct-v0.3 | Mistral-7B-Instruct-v0.3 |
  | 3 | ChatGLM3-6B (in doubt) | Qwen2.5-7B-Instruct |
  | 4 | Qwen2.5-7B-Instruct | Llama-2-7b-chat |
  | 5 | Llama-2-7b-chat | DeepSeek-LLM-7B-chat |
  | 6 | DeepSeek-LLM-7B-chat | Granite-7b-instruct |
  | 7 | Granite-7b-instruct | |

  Ranks 1 and 2 do not depend on ChatGLM3. Unscreened models follow from rank 7 (OLMo-7B-0724 at the assumed
  median) in both.
- **Unscreened models cannot be ranked above a screened one by the rule.** Four would reach rank 2 or 3 if
  they screened at 0.40 (OLMo-7B-0724, calm2-7b, A.X-3.1-Light, Olmo-3). All four are full-attention (32/32)
  models with an 11,008-wide FFN, the shape of the three that screened at 0.15 to 0.20.
- **A pattern, n = 7, confounded with age and training:** the three models with grouped attention to 4 or 8
  key-value heads screen at 0.35 to 0.40 (Yi, Mistral, Qwen2.5); the three with full attention at 0.15 to 0.20
  (Llama-2, DeepSeek, Granite-7B); ChatGLM3 (2 key-value heads) sits between or is mis-screened. It is a
  hypothesis for the test list, not a criterion.

**Does the provisional trio (memo-06 section 6) still follow the rule?**

| slot | memo-06 section 6 | by the rule with measured s | note |
|---|---|---|---|
| subject 1 | Yi-1.5-6B-Chat | Yi-1.5-6B-Chat | unchanged; first on criteria 1 and 2 |
| subject 2 | ChatGLM3-6B | **Mistral-7B-Instruct-v0.3** | rank 2 on criterion 1; s 0.40 at both settings; Q4_0 / Q8_0 x 1.02 |
| subject 3 | DeepSeek-LLM-7B-chat | **Qwen2.5-7B-Instruct** without ChatGLM3; **ChatGLM3-6B** with it | DeepSeek is rank 6 (5 without ChatGLM3) and no longer follows |
| reserves | Llama-2-7b-chat, Granite-7b-instruct | without ChatGLM3: Llama-2-7b-chat, DeepSeek-LLM-7B-chat; with it: Qwen2.5-7B, Llama-2-7b-chat | Llama-2 and DeepSeek tie; criterion 3 (head 3 % against 9 %) puts Llama-2 first |

Families: 01.AI, Mistral, Alibaba Qwen, Meta, DeepSeek (or Zhipu) are five. Without ChatGLM3 the rule returns the
earlier set exactly (Mistral, Qwen2.5, with the former reserve Yi-1.5-6B first). The rule returns two of the earlier subjects
(Mistral, Qwen2.5), which have the most host evidence (E1, T1, K1 shapes). The choice is the lead's.

## 3. What 12 printed needs, per screened model

12.0 printed is 79.2 ms steady; with O = 8 ms, 71.2 ms are left. That allows 5.11 G weight-equivalents of compute
on three cores (6.37 G on four) and 71.2 ms x r of bytes. All columns *estimated*; s is *host*.

- "k max": the highest cost per touched weight the column kernel may have at the measured s. **Under 1.0 means
  no exact kernel can do it**: even a sparse kernel as cheap as dense `sdot` is too slow.
- "s needed at k 1.55": the frontier the model would need at the bottom of K1's device price.
- "r min": bytes at the measured s over 71.2 ms. R69's pre-registered 80 % band for r is 19 to 30.
- Residency: every byte read per token must be held (memo-05: column sparsity does not shrink the resident set).
  "Step" is the smallest R69 P1 file (2.7, 3.0, 3.4, 4.2, 4.6 GB) that covers the Q4_0 token. Yi's margin under
  the 3.4 step is 0.07 GB, less than the 0.15 GB the lead's freeze rule (memo-06 section 5.3, S3) asks for.
- Last column: at the central C of 3.1 GB, the largest type whose token fits C − 0.15 = 2.95 GB, priced
  compute-bound and dense at that type's kernel cost: Q3_K at 2.0 x Q4_0 per weight (*device*, R41), IQ3 at 3 x
  (*published*, Ebony's source in memo-06 section 3.2). IQ4_XS (x 0.954) does not fit for any model.

| model | s (host) | k max, 3 cores | k max, 4 cores | s needed at k 1.55, 3 / 4 cores | r min GB/s at s | Q4_0 GB | R69 step that holds it | P(step held) | at C 3.1: type that fits, printed tok/s |
|---|---|---|---|---|---|---|---|---|---|
| Yi-1.5-6B-Chat | 0.40 | 1.46 | 1.84 | 0.43 / 0.29 | 31.2 | 3.33 | 3.4 | 0.35 | Q3_K_M (2.87 GB): 5.6 |
| Mistral-7B-Instruct-v0.3 | 0.40 | 1.19 | 1.49 | 0.54 / 0.42 | 37.1 | 4.04 | 4.2 | 0.12 | IQ3_XXS (2.84 GB): 3.1 |
| chatglm3-6b | 0.20 | 1.06 | 1.33 | 0.45 / 0.31 | 40.2 | 3.43 | 4.2 | 0.12 | IQ3_S (2.70 GB): 3.7 |
| Qwen2.5-7B-Instruct | 0.35 | 1.08 | 1.37 | 0.55 / 0.42 | 41.8 | 4.12 | 4.2 | 0.12 | IQ3_XXS (2.89 GB): 3.1 |
| Llama-2-7b-chat | 0.20 | 0.96 | 1.20 | 0.50 / 0.38 | 43.6 | 3.75 | 4.2 | 0.12 | IQ3_XXS (2.64 GB): 3.3 |
| deepseek-llm-7b-chat | 0.20 | 0.97 | 1.22 | 0.50 / 0.37 | 44.3 | 3.76 | 4.2 | 0.12 | IQ3_XXS (2.64 GB): 3.4 |
| granite-7b-instruct | 0.15 | 0.90 | 1.13 | 0.50 / 0.38 | 45.9 | 3.75 | 4.2 | 0.12 | IQ3_XXS (2.64 GB): 3.3 |

Reading:
- **Yi-1.5-6B** is the only model with a row inside reach: k ≤ 1.46 on three cores is under K1's host floor
  (1.53); on four cores k ≤ 1.84 is inside the device price. It still needs r ≥ 31.2 (above the 80 % band) and
  the 3.4 step. It has no slack on s: it needs 0.43 at k 1.55 on three cores and has 0.40, with +5.11 % at 0.45.
- **Mistral** needs four cores and k ≤ 1.49, r ≥ 37.1 and the 4.2 step (P 0.12).
- **The 20 % models are out on three cores by arithmetic** (k max under 1.1) and need r ≥ 40 on any core count.
- **All three at 12** therefore needs r ≥ 37 to 42 GB/s, a 4.2 GB resident set and a kernel at or under the host
  floor on four cores. I do not put a probability on it; each of the three conditions is outside R69's
  pre-registered 80 % band or at P 0.12. If R69 reads inside its bands, the lead's kill line K0 (memo-06 section
  5.3) is met for every subject but Yi, and for Yi unless r ≥ 31 and the 3.4 step holds.
- **At C 3.1 the best resident price is 5.6 tok/s (Yi, Q3_K_M).** A 3-bit kernel at the best *published* cost
  (1.5 x, ik_llama.cpp on an M2) would make that 7.4. Neither is near 12.

## 4. Challenge to Ebony's numbers, and where we agree

Read: her `ebony-compute.md` sections 1 to 3 and 5 (her section 4, the challenge to this file, was still a
placeholder), and memo-06 section 3.2. Prices here are *estimated*; frontiers are *host*.

**Agreed, reached separately.**
- One cell reaches 12: Yi-1.5-6B with K1 columns at s 0.40 and a fourth core. Hers: 12.3 at r ≥ 32, k ≤ 1.6,
  core factor 1.15. Mine: r ≥ 31.2, and k ≤ 1.84 at x 1.245 or k ≤ 1.69 at x 1.15 (top 13.7 or 13.3 at r 35).
- No second model reaches 12 at r ≤ 35. Best second: Mistral, 10.2 (hers) and 9.7 to 11.7 (mine, three to four
  cores at the top of the band; 11.0 at x 1.15).
- K1 columns are compute-neutral at s 0.40 and compute-negative at s 0.20 (break-even k = 1 / (1 − s)).
- The order: Mistral and Qwen2.5 above the Llama-2 class. Criterion 2 has overtaken the byte order.
- Her re-price of her own Y8 downwards (C4: +0.4 to +0.8). Mine below is lower still.

**Two conventions still differ; the merged memo needs one of each.**
- c: she now uses 69.9 G weights/s (my memo-06 value); I moved to 71.8 G (40.4 GB/s, R69's pre-registered
  centre, her memo-06 value). We crossed. Her weight line is 4.98 G, mine 5.11 G; compute-bound cells differ by
  2.7 %. I propose 40.4 GB/s, because R69 scores that number.
- Fourth core: she uses x 1.15 (1.00 to 1.25); my columns use 50.3 / 40.4 = 1.245, the perfect-scaling end. Hers
  is the better centre. At x 1.15 my section 3 "k max, 4 cores" reads 1.69 (Yi) and 1.37 (Mistral).

**Challenges.**

1. **C6 (units of eight channels): her u looks two to four times too high.** Her own section 2.1 finds the inputs
   Gaussian up to the median in all seven models. For independent half-normal channels, a unit is skipped only
   when all eight are under the threshold, so u = q^8, and the skipped elements reach up to the q-quantile.
   Matching the input energy that H3 removes at each model's measured frontier (3.5 % at s 0.40, 0.4 % at
   s 0.20) gives **u about 0.10 for Yi and Mistral, 0.07 for Qwen2.5 and 0.02 for the Llama-2 class**
   (*estimated*; equal energy as the measure of damage is my assumption). Her u (0.18, 0.15, 0.08) removes 8.1 %
   of the energy at 0.18, past Yi's cliff (5.1 % at s 0.45, which costs +5.11 %, *host*). Her U2 kill line is
   0.12; this predicts C6 closes unless per-channel scale structure (sorting inside a block) correlates the
   eight. Prediction before U2: Yi and Mistral 0.10 (0.06 to 0.15), sorted order at most +0.04.
2. **Her C1 + C6 and C1 + C6 + C3 + C7 columns need r of 44 to 49 GB/s** by her own table and print 10.7 to
   10.9 on Yi at r 35. They are not routes to 12 at any r inside R69's band, and she says so; I would drop them
   from the merged memo's headline table and keep them as section 3.2's second table only.
3. **C4 / Y8 (groups of four) prices under E1 at r 35.** With her proxy costs and no extra `down` sparsity
   (one budget: a skipped neuron is already a zeroed `down` input; T1 alone costs +3.5 to +4.8 % at 33 to 38 %,
   E1 alone +3.4 % at 60 %), on the two models where both are screened:

   | model | lever | r 24 | r 35 | r 35, 4 cores | r 45, 4 cores |
   |---|---|---|---|---|---|
   | Mistral-7B-Instruct-v0.3 | E1 `down` columns, s 0.60, K1 k 2.2 to 1.55 | 6.1 to 6.3 | 8.5 to 9.0 | 8.5 to 9.0 | 10.7 to 11.4 |
   | Mistral-7B-Instruct-v0.3 | T1 groups, g 0.33, with proxy to proxy-free | 5.5 to 6.5 | 7.8 to 9.4 | 7.8 to 9.4 | 9.7 to 11.8 |
   | Qwen2.5-7B-Instruct | E1 `down` columns, s 0.60, K1 k 2.2 to 1.55 | 5.9 to 6.2 | 8.4 to 8.8 | 8.4 to 8.8 | 10.5 to 11.2 |
   | Qwen2.5-7B-Instruct | T1 groups, g 0.36, with proxy to proxy-free | 5.5 to 6.5 | 7.7 to 9.3 | 7.7 to 9.3 | 9.7 to 11.8 |

   Groups are −0.7 to +0.5 against `down` columns at r 35, and both are under uniform columns on Mistral (9.7
   at the top). Both are byte-bound at r 35, so the fourth core adds nothing to them. E1 cuts about 14 % of the
   token's bytes, groups 4 to 17 %, uniform columns at 0.40 about 35 %. T1 used the exact product, so it is an
   upper bound for any proxy rule; memo-06's "groups at 40 % plus `down` at s 0.6" spent the budget twice.
4. **C5 (up rule) computes all of `up`**, so it saves at most (g/3 of gate rows + `down` columns at s = g): at
   g 0.33 and k 1.6 the `down` part costs 1.07 of dense, and the whole lever is the `gate` third only. Her FFN
   factor 0.91 is right; on bytes it is 0.79 only if `down` goes through K1, which then costs compute. Worth U4
   only after U1 says the FFN is where the tolerant models have room.
5. **Hypothesis A (k and v dense on full-attention models) can change slot 3, and only that.** If it lifts
   Llama-2 to 0.35 on the rest, k and v (16.6 % of the body) stay dense and Llama-2 prices 6.0 to 6.2 at r 19,
   7.0 to 7.7 at r 24 and 8.0 to 9.6 at r 35: above ChatGLM3 and Qwen2.5, still under Mistral at r 19 and r 24.
   That makes U1 (b) the most valuable cell on her list.
6. **Her P(resident) 0.3 "at 3.3 GB plus the column copy"** needs the layout stated: if K1 reads a column-major
   copy and the row layout is not also held, the resident set is 3.33 GB and the pre-registered P(3.4 held) is
   0.35; if both are held it is 6.7 GB and the cell is closed. Tom should say which K1 needs.

Where her memo-06 challenge to me was right: residency was missing from my prices (now section 3); one c and no
four-core row (now both); Llama-2's rank leaned on TEAL's +3.0 % at 40 %, and the *host* screen gives +4.2 % at
20 % and +5.7 % at 30 %, so LaRoSA's re-run was the better guide. The served-quantisation table of
`memo-06-work/bonnie-pool.md` section 4 has no header row; its columns are model, Q4_0, Q4_K_M, IQ4_XS, Q3_K_M,
IQ3_S, IQ3_XXS, file + KV + app. Not adopted: the two-stage head as a model property (at most 0.3 tok/s).

## 5. Score of my memo-06 statements

| statement (memo-06 section 3.1 and bonnie-pool.md) | result |
|---|---|
| s 0.30 to 0.40 assumed for unscreened models | **miss, low** on 4 of 5 (0.15, 0.20, 0.20, 0.20 at both settings); hit on Yi (0.40) |
| "older, less trained models may be more tolerant (TEAL's Llama-2 row)" | **miss**: the three oldest full-attention models are the least tolerant |
| top five by bytes: Yi, ChatGLM3, Llama-2, Granite-7B, DeepSeek | three of the five fall under Mistral and Qwen2.5 once s is measured |
| Q4_0 inside F16 x 1.20 for each (unmeasured then) | holds: x 1.012 to 1.037 (*host*) |
| Q4_0 bytes per token from `config.json` | hold: Tom's header inventory gives the same GB to two decimals (*host*) |

## 6. Joint test list (host, Tom) and try list (device)

Merged from this file and Ebony's `ebony-compute.md` section 5 (her numbers U and Z kept). She has not yet
reviewed the order. Tom's rule stands: one 7B job at a time.

**Test list (host).** Ordered by how much each can move the choice of subjects or the price of 12.

| # | test | cells | decides | closes when |
|---|---|---|---|---|
| 1 | **U6** ChatGLM3 eligibility: the publisher's own code (transformers, its tokenizer) on the same 8 chunks against the GGUF at F16, then token ids and per-chunk values compared | 2 runs of 8 chunks; running the publisher's custom code (`trust_remote_code`) needs the owner's or lead's consent first (Tom, section 5.3) | whether ChatGLM3 is a subject at all | values agree → 33 is the model's level on this text and the lead rules on the gate; they differ → a conversion or graph fault, and its screen is void |
| 2 | **U1** where the split sits, on Llama-2 and Yi, then DeepSeek: (b) uniform 30 / 35 % with k and v inputs dense; (a) four per-group cells at 30 %; (c), (d) first tokens and layers 0 to 3 dense | about 8 per model | slot 3 (Llama-2 with k, v dense prices above ChatGLM3 and Qwen2.5, section 4 item 5); the E1 price for Yi | nothing moves the full-attention frontier by 10 points → the class is priced at 0.20 and leaves the reserve list |
| 3 | **B1** the missing grid cells: 25 % on DeepSeek and Llama-2, 42 % on Yi, both settings | 6 | ranks 5 and 6; Yi's slack to its cliff (it needs 0.43 at k 1.55 on three cores) | no kill; replaces my interpolation |
| 4 | **B2** E1 (`down` input 40 / 50 / 60 %) on Yi; **U5** T1 on Yi and Llama-2 | 6 + 12 | whether subject 1 has an FFN-only lever; scores Ebony's section 1 | Yi under 50 % on `down` → Yi's only lever is uniform columns |
| 5 | **B3** H3 on one newer full-attention model (A.X-3.1-Light or Olmo-3-7B) and one unscreened grouped-attention model (Minerva-7B or salamandra-7b) | 2 models | the attention-type pattern; whether any unscreened model can displace rank 2 or 3 | both land in their class's range → stop screening the pool |
| 6 | **U2**, then **U3**: C6 unit frontier (10 / 15 / 20 / 25 %), then its kernel | 8 per model; microbench | whether a compute lever exists that K1 is not | under 0.12 on Yi and Mistral (my prediction: 0.10) → closed |
| 7 | **B4** per-type gemv cost (Q4_0x4, IQ4_NL, Q3_K, IQ3_S) and Q3_K_M / IQ4_XS PPL against F16 on Yi and Mistral | 4 types, 4 PPL rows | the price of the type that fits C (section 3, last column) | 3-bit above 1.3 x in the best kernel → the fits-C path stays under 8 tok/s |
| 8 | **U4** up rule; **B5** joint cell (T1 groups at 20 to 30 % plus `down` element thresholds), Mistral and Qwen2.5 | 6 + 8 | whether any FFN-only rule beats E1 in bytes saved inside one budget | none does → C4, C5 and Y8 close behind E1 |
| – | T6 (1-bit proxy kernel), T2 (proxy ranking) | | only if item 8 leaves a proxy lever standing | |

Tom to state, with no run: which layout K1 at TR 128 reads, and whether the row layout must also be resident.

**Try list (device; each needs its own pre-registration).**

| # | lever or measurement | gate | subjects | *estimated* gain (printed) |
|---|---|---|---|---|
| 1 | **R69** as pre-registered (r, c_b, C, k on the A78, core-set cells) | none; next device round | none loaded | sets the r column, the k range and the step held; K0 per subject follows from section 3 |
| 2 | residency over 20 passes of the real Q4_0 files through the no-copy path (memo-05 D1, Y1c as the test) | R69: 3.4 step held | Yi first (3.33 GB), then Mistral (4.04) | none by itself; without it no Q4_0 price is real |
| 3 | **Z2** K1 columns at TR 128, thresholds per subject at the *host* frontier | R69 k ≤ 1 / (1 − s), and under 2.3 at 0.60 kept; full-set PPL at both settings | Yi (0.40), Mistral (0.40), third subject | r 24: +1.8 to +3.0 (Yi), +1.4 to +2.8 (Mistral); r 35: 0 to +2.3, 0 to +1.9; at s 0.20: 0 to +1.2 at r 24, 0 at r 35 |
| 4 | **Z1** X1 + 3 A78, then weighted A55 chunks | R69's core-set cells | all | only when compute-bound: Yi and Mistral at r 35 up to +2.0 at x 1.245, +1.6 and +1.3 at x 1.15; 0 while byte-bound |
| 5 | the type that fits C (Q3_K_M or a mixed file), if the 3.4 step is not held | B4 | Yi | 5.6 (7.4 with a 1.5 x kernel); a floor, not a route to 12 |
| 6 | **Z3** C6 unit skip; **Z4** GPU compute lane; **Z5** up rule; **Z6** head screen | U2 and U3; R69 Rule D; U4; none | as Ebony's section 5 | as she priced; Z3 at my u (0.10): about half of hers |

## 7. Addendum 2026-11-22: slots 2 and 3 after the owner's exclusion of ChatGLM3

Trigger: the owner excluded ChatGLM3-6B (2026-11-22); Yi-1.5-6B is subject 1 (lead). New *host* input:
`tom-split.md` section 5 (`6e90fb7`). With the attention k and v inputs dense, Llama-2-7b and DeepSeek-LLM-7B hold
40 % on every other input at both `-b 512` and `-ub 1`: Llama-2 +3.97 / +3.95 % (0.5 point under the line),
DeepSeek +2.95 / +3.54 %. That skips 33.4 % of the block's weights (k and v are 16.6 % of the body). Mistral
(0.40) and Qwen2.5 (0.35) are plain uniform frontiers (memo-02).

Conventions (agreed, `ebony-compute.md` section 8): c 40.4 GB/s = 71.8 G weights/s on X1 + 2 A78; fourth core
x 1.15 on compute, priced at R69's own four-core read rate (pre-registered 27 GB/s, 21 to 35), not the column's
r; O 8 ms; printed = 0.95 x steady. Band as in section 1 (k 2.2 with O 12, to k 1.55 with O 6). k and v keep the
stock 4x4 layout at k = 1; every other matrix uses the K1 layout, one layout per tensor, so the resident set is the
stock size (Tom, `tom-split.md` section 4). All tok/s *estimated*; frontiers *host*. Script:
`bonnie_reprice.py`, last block.

| rank | model | rule (host frontier) | share of body skipped | stock r 19 / 24 / 35 | levers r 19 | levers r 24 | levers r 35 | 4 cores at r4 27 / 35 |
|---|---|---|---|---|---|---|---|---|
| 1 | Llama-2-7b-chat | 0.40, k and v dense | 33.4 % | 4.6 / 5.8 / 8.3 | 6.2 to 6.5 | 7.4 to 8.1 | 8.0 to 10.2 | 8.4 to 9.1 / 8.4 to 11.5 |
| 2 | deepseek-llm-7b-chat | 0.40, k and v dense | 33.4 % | 4.6 / 5.8 / 8.2 | 6.1 to 6.3 | 7.5 to 7.9 | 8.0 to 10.4 | 8.4 to 8.8 / 8.6 to 11.2 |
| 3 | Mistral-7B-Instruct-v0.3 | 0.40 uniform | 40.0 % | 4.3 / 5.4 / 7.7 | 6.3 to 6.6 | 6.7 to 8.2 | 7.5 to 9.7 | 7.6 to 9.2 / 7.6 to 11.0 |
| 4 | Qwen2.5-7B-Instruct | 0.35 uniform | 35.0 % | 4.2 / 5.3 / 7.6 | 5.6 to 5.8 | 6.4 to 7.3 | 7.3 to 9.0 | 7.2 to 8.2 / 7.3 to 10.3 |

Ranking key: sum of both band ends at r 19, 24 and 35.

- **Order by criterion 1: Llama-2-7b, DeepSeek-LLM-7B, Mistral-7B-v0.3, Qwen2.5-7B.** Keeping k and v dense
  lowers compute at a high kernel cost (they run at k = 1). That is why the two full-attention models lead at r 24
  and r 35 with fewer skipped weights than Mistral.
- **The margins are small.**
  - Llama-2 against Mistral: −0.1 at r 19; +0.7 / −0.1 at r 24 (low / high end); +0.5 / +0.5 at r 35.
  - Llama-2 against DeepSeek: within 0.3 everywhere.
  - Both gaps are smaller than the error of these estimates (c, r, k are all R69 centres). Whether that counts as
    a tie, which passes to criterion 2, is the lead's call.
  - If it is a tie, criterion 2 favours Mistral: 40 % of its weights skipped at +2.44 / +2.92 %, against 33.4 % at
    +3.97 / +3.95 % for Llama-2 (thin margin) and +2.95 / +3.54 % for DeepSeek.
- **Slots by the strict order:** subject 2 Llama-2-7b-chat (Meta), subject 3 DeepSeek-LLM-7B-chat (DeepSeek);
  reserves Mistral-7B-Instruct-v0.3, then Qwen2.5-7B-Instruct. With Yi (01.AI) that is three families, and all
  four candidates are of different families from each other and from Yi.
- **If the lead reads Llama-2 against Mistral as a tie:** criterion 2 puts Mistral in slot 2 or 3, and the choice
  between Llama-2 and DeepSeek for the other slot goes to DeepSeek on criterion 2 (more margin at the same share).
  Criterion 3 (head 3 % against 9 %) would favour Llama-2.
- **Flags.**
  - DeepSeek was screened without BOS (the converter wrote `add_bos_token = false`, Tom section 5.1 of
    tom-screens.md). The served file must decide this, and its budget is measured that way.
  - Llama-2 is gated (manual approval; the laptop's login already has it).
  - Llama-2's 40 % cell is 0.5 point under the line, so a full-set run may move it to 35 % (29.2 % skipped).
    Priced at 35 %, Llama-2 prints 6.0 to 6.2 / 7.0 to 7.7 / 8.0 to 9.6 at r 19 / 24 / 35: under Mistral at r 19,
    level with it at r 35 (*estimated*, one grid step).

**What 12 printed needs** (71.2 ms after O 8; s and the k, v rule as above):

| model | k max, 3 cores | k max, 4 cores (x 1.15) | r min GB/s at s | Q4_0 GB | R69 step | P(step held) |
|---|---|---|---|---|---|---|
| Llama-2-7b-chat | 1.21 | 1.44 | 37.5 | 3.75 | 4.2 | 0.12 |
| deepseek-llm-7b-chat | 1.21 | 1.47 | 38.6 | 3.76 | 4.2 | 0.12 |
| Mistral-7B-Instruct-v0.3 | 1.19 | 1.37 | 37.1 | 4.04 | 4.2 | 0.12 |
| Qwen2.5-7B-Instruct | 1.08 | 1.26 | 41.8 | 4.12 | 4.2 | 0.12 |

- **No candidate for slots 2 and 3 reaches 12 with a priced lever.** Each needs r ≥ 37 on three cores, outside
  R69's pre-registered 80 % band (19 to 30), and a kernel at or under k 1.2 (three cores) or 1.26 to 1.47 (four
  cores at x 1.15). K1's host floor is 1.53.
- **On four cores the read rate is R69's four-core cell, so the top is byte-bound:** at r4 27 the four print
  7.2 to 9.2, and at r4 35 the tops of their bands are 10.3 to 11.5. Llama-2 at r4 35 and k 1.55 (11.5) is the nearest cell to 12.
- **Residency is as before:** every one of the four needs the 4.2 GB step (P 0.12, pre-registered).
- **For Yi (subject 1), the section 3 row stands with the new four-core convention.** At x 1.15 k may be up to 1.69
  on four cores, and at r4 35 the top of its band is 13.3. Its frontier moved to 42 % (plain; +2.63 / +3.95 % at `-b 512` / `-ub 1`, *host*,
  `tom-split.md` section 2), which nearly closes its "s needed at k 1.55" gap on three cores: it needs 0.43 and has
  0.42.

Requested: Ebony checks this section (a short note in her file).

## 8. Not done, and where I may be wrong

- No run of mine: every *host* number is Tom's. I first read his logs while he was still running; the values
  used here equal his final file (`a6f6b36`).
- Frontiers are 40-chunk screens at a +4.5 % margin, on a coarse grid. A full-set run can move a model by one
  grid step; one step on DeepSeek or Llama-2 (to 0.30) would not lift them above Mistral at r 19 or r 24.
- c, r, C and k are R69's pre-registered centres and my device price, not measurements. The whole of section 3
  moves with them: at c_b 45 the three-core compute budget is 5.70 G and Yi's k max is 1.64.
- The sparse kernel's bytes factor (0.889 s) and k are taken as model-independent.
- The four-core columns use 50.3 / 40.4 = x 1.245, the perfect-scaling end; Ebony uses x 1.15 (section 4).
- No web search was made in this round: nothing here states the current state of a model or of llama.cpp; the
  pool, licences and support are memo-06's (read 2026-11-21).
- Ebony's challenge to this file (her section 4) was a placeholder when I finished; she has not reviewed
  sections 4 and 6.

Sources: `lab/c3/logs/h3-frontiers.log`, `h3-<model>-{b512,ub1}-40.log`, `ppl-<model>-{f16,q4_0}-b512-40.log`,
`t1-{mistral,qwen25}-{b512,ub1}-40.log` (*host*, Tom); `memo-06-work/tom-screens.md` sections 5 and 6
(`a6f6b36`); memo-02 (H3), memo-03 section 9, memo-05 (K1, E1, T5), memo-06 sections 1, 3.2, 5, 6;
`prereg/R69-probe-read-rate-capacity-gpu-device.md` section 3; `prereg/R69-amendment-1.md`.
