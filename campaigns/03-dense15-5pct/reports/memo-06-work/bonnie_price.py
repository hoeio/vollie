# estimated: memo-03 §9 formula applied to published configs (HF config.json, read 2026-11-21). Nothing here is measured.
# usage: python3 -I bonnie_price.py   (prints the markdown tables of bonnie-pool.md)
C = 69.9   # G weights/s, 3 threads, S2 clocks, in situ (memo-01 §1, estimated from device benches R37/R41)
Q4, Q6 = 4.5 / 8, 6.5625 / 8   # bytes per weight: Q4_0 body, Q6_K head (what llama-quantize writes; memo-02 H1)
# name, family, total params (HF safetensors), layers, d, ffn, vocab, heads, kv heads, head_dim, tied
M = [
 ("Yi-1.5-6B-Chat", "01.AI", 6061035520, 32, 4096, 11008, 64000, 32, 4, 128, 0),
 ("chatglm3-6b", "Zhipu", 6243584032, 28, 4096, 13696, 65024, 32, 2, 128, 0),
 ("deepseek-llm-7b-chat", "DeepSeek", None, 30, 4096, 11008, 102400, 32, 32, 128, 0),
 ("Llama-2-7b-chat", "Meta", 6738417664, 32, 4096, 11008, 32000, 32, 32, 128, 0),
 ("granite-7b-instruct", "IBM", 6738481152, 32, 4096, 11008, 32008, 32, 32, 128, 0),
 ("OLMo-7B-0724-Instruct", "AI2", 6888095744, 32, 4096, 11008, 50304, 32, 32, 128, 0),
 ("calm2-7b-chat", "CyberAgent", 7008948224, 32, 4096, 11008, 65024, 32, 32, 128, 0),
 ("salamandra-7b-instruct", "BSC", 7768117248, 32, 4096, 11008, 256000, 32, 8, 128, 0),
 ("A.X-3.1-Light", "SKT", 7264800768, 32, 4096, 10880, 102400, 32, 32, 128, 0),
 ("Olmo-3-7B-Instruct", "AI2", 7298011136, 32, 4096, 11008, 100278, 32, 32, 128, 0),
 ("OLMo-2-1124-7B-Instruct", "AI2", 7298617344, 32, 4096, 11008, 100352, 32, 32, 128, 0),
 ("Tri-7B", "Trillion Labs", 7526944768, 32, 4096, 11008, 128256, 32, 32, 128, 0),
 ("Falcon3-7B-Instruct", "TII", 7455550464, 28, 3072, 23040, 131072, 12, 4, 256, 0),
 ("Qwen2.5-7B-Instruct", "Alibaba Qwen", 7615616512, 28, 3584, 18944, 152064, 28, 4, 128, 0),
 ("Mistral-7B-Instruct-v0.3", "Mistral", 7248023552, 32, 4096, 14336, 32768, 32, 8, 128, 0),
 ("Minerva-7B-instruct-v1.0", "Sapienza", 7399542784, 32, 4096, 14336, 51264, 32, 8, 128, 0),
 ("internlm2_5-7b-chat", "Shanghai AI Lab", 7737708544, 32, 4096, 14336, 92544, 32, 8, 128, 0),
 ("EXAONE-3.5-7.8B-Instruct", "LG", 7818448896, 32, 4096, 14336, 102400, 32, 8, 128, 0),
 ("openPangu-Embedded-7B", "Huawei", 8030890112, 34, 4096, 12800, 153376, 32, 8, 128, 0),
 ("Teuken-7B-instruct-v0.6", "OpenGPT-X", 7452725248, 32, 4096, 13440, 250680, 32, 2, 128, 1),
 ("Ministral-8B-Instruct-2410", "Mistral", 8019808256, 36, 4096, 12288, 131072, 32, 8, 128, 0),
 ("Llama-3.1-8B-Instruct", "Meta", 8030261248, 32, 4096, 14336, 128256, 32, 8, 128, 0),
 ("kanana-1.5-8b-instruct", "Kakao", 8030285824, 32, 4096, 14336, 128259, 32, 8, 128, 0),
 ("Apertus-8B-Instruct-2509", "Swiss AI", 8053338176, 32, 4096, 21504, 131072, 32, 8, 128, 0),   # ffn: see note
 ("Qwen3-8B", "Alibaba Qwen", 8190735360, 36, 4096, 12288, 151936, 32, 8, 128, 0),
 ("MiniCPM4-8B", "OpenBMB", 8185253888, 32, 4096, 16384, 73448, 32, 2, 128, 0),
 ("c4ai-command-r7b-12-2024", "Cohere", 8028033024, 32, 4096, 14336, 256000, 32, 8, 128, 1),
 ("granite-3.3-8b-instruct", "IBM", 8170864640, 40, 4096, 12800, 49159, 32, 8, 128, 1),
]
def shape(m):
    name, fam, tot, L, d, f, V, h, kv, hd, tied = m
    nmat = 2 if name.startswith("Apertus") else 3   # Apertus: plain MLP with xIELU (up, down), not a gated FFN
    attn = 2 * d * h * hd + 2 * d * kv * hd
    WL = L * (attn + nmat * d * f); WH = V * d
    calc = WL + WH * (1 if tied else 2)
    if tot: assert abs(calc / tot - 1) < 0.006, (name, calc, tot)   # the one check: shapes reproduce the published count
    return WL / 1e9, WH / 1e9, (tot or calc) / 1e9
def toks(WL, WH, r, s=0.0, k=1.0, O=7.5):
    BL, BH = WL * Q4, WH * Q6
    ms = max((BH + BL * (1 - 0.889 * s)) / r, (WH + WL * (1 - s) * k) / C) * 1000 + O
    return 0.95 * 1000 / ms
def band(WL, WH, r):
    st = (toks(WL, WH, r, O=10), toks(WL, WH, r, O=5))
    lo = max(st[0], toks(WL, WH, r, 0.30, 2.2, 10)); hi = max(st[1], toks(WL, WH, r, 0.40, 1.55, 5))
    return st, (lo, hi), toks(WL, WH, r, 0.40, 1.2, 5)
rows = []
for m in M:
    WL, WH, tot = shape(m); B = WL * Q4 + WH * Q6
    rows.append((m, WL, WH, tot, B, band(WL, WH, 19), band(WL, WH, 35)))
rows.sort(key=lambda x: -(x[5][1][1] + x[6][1][1]))   # rank: lever-band top, r = 19 and r = 35 summed
f2 = lambda t: "%.1f to %.1f" % t
print("| # | model | family | total B | W/token G | Q4_0 GB/token | head share | stock r=19 | levers r=19 | stock r=35 | levers r=35 | k=1.2 corner r=35 |\n|" + "---|" * 12)
for i, (m, WL, WH, tot, B, b19, b35) in enumerate(rows, 1):
    print("| %d | %s | %s | %.3f | %.2f | %.2f | %.0f %% | %s | %s | %s | %s | %.1f |" % (i, m[0], m[1], tot, WL + WH, B, 100 * WH * Q6 / B, f2(b19[0]), f2(b19[1]), f2(b35[0]), f2(b35[1]), b35[2]))
print("\n| model | Q4_0 (4x4 sdot) | Q4_K_M x1.055 | IQ4_XS x0.954 | Q3_K_M x0.862 | IQ3_S x0.788 | IQ3_XXS x0.703 | file+KV+app GB |\n|" + "---|" * 8)
for m, WL, WH, tot, B, b19, b35 in rows:
    emb = 0 if m[10] else WH * Q4; kvb = m[3] * 512 * 2 * m[8] * m[9] * 2 / 1e9   # F16 KV at ctx 512
    print("| %s | %.2f | %.2f | %.2f | %.2f | %.2f | %.2f | %.2f |" % (m[0], B, B * 1.055, B * 0.954, B * 0.862, B * 0.788, B * 0.703, B + emb + kvb + 0.4))
