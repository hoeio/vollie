# estimated: memo-03 section 9 formula with host-screened column frontiers (Tom, lab/c3/logs/h3-frontiers.log, 2026-11-22).
# Nothing here is measured on the phone. usage: python3 -I bonnie_reprice.py
import contextlib, io, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'memo-06-work'))
with contextlib.redirect_stdout(io.StringIO()):
    from bonnie_price import M, shape, Q4, Q6          # shapes and the published-count assert are reused, not copied
CB = {3: 40.4, 4: 50.3}                                # GB/s of Q4_0, R69 pre-registered centres (X1+2 A78; X1+3 A78)
C = {n: v / Q4 for n, v in CB.items()}                 # G weights/s: 71.8 and 89.4
S = {"Yi-1.5-6B-Chat": .40, "chatglm3-6b": .20, "deepseek-llm-7b-chat": .20, "Llama-2-7b-chat": .20,
     "granite-7b-instruct": .15, "Mistral-7B-Instruct-v0.3": .40, "Qwen2.5-7B-Instruct": .35}   # host, both batch settings
SLO, SHI, SMED = .15, .40, .20                         # unscreened: range and median of the seven screened models (assumed)
def ms(WL, WH, r, s, k, O, n=3):                       # steady ms; stock is s=0, k=1
    return max((WH * Q6 + WL * Q4 * (1 - 0.889 * s)) / r, (WH + WL * (1 - s) * k) / C[n]) * 1000 + O
def p(WL, WH, r, s, k, O, n=3):                        # printed tok/s, better of stock and the column kernel
    return 950 / min(ms(WL, WH, r, 0, 1, O, n), ms(WL, WH, r, s, k, O, n))
def band(WL, WH, r, slo, shi, n=3): return p(WL, WH, r, slo, 2.2, 12, n), p(WL, WH, r, shi, 1.55, 6, n)
rows = []
for m in M:
    WL, WH, tot = shape(m); s = S.get(m[0])
    key = sum(sum(band(WL, WH, r, s or SMED, s or SMED)) / 2 for r in (19, 35))   # rank: band midpoints at r 19 and r 35
    rows.append((key, m, WL, WH, s))
rows.sort(key=lambda x: -x[0])
f = lambda t: "%.1f to %.1f" % t
print("| # | model | family | Q4_0 GB/token | s | stock r 19 / 24 / 35 | levers r 19 | levers r 24 | levers r 35 | r 35, 4 cores | unscreened: at s 0.40, r 19 / r 35 |\n|" + "---|" * 11)
for i, (key, m, WL, WH, s) in enumerate(rows, 1):
    lo, hi = (s, s) if s else (SLO, SHI)
    st = " / ".join("%.1f" % p(WL, WH, r, 0, 1, 8) for r in (19, 24, 35))
    up = "" if s else "%.1f / %.1f" % (p(WL, WH, 19, .4, 1.55, 6), p(WL, WH, 35, .4, 1.55, 6))
    print("| %d | %s | %s | %.2f | %s | %s | %s | %s | %s | %s | %s |" % (i, m[0], m[1], WL * Q4 + WH * Q6,
          "**%.2f** host" % s if s else "0.15 to 0.40 assumed", st, f(band(WL, WH, 19, lo, hi)), f(band(WL, WH, 24, lo, hi)),
          f(band(WL, WH, 35, lo, hi)), f(band(WL, WH, 35, lo, hi, 4)), up))
T = 950 / 12 - 8                                        # ms for bytes and compute at printed 12, O = 8
step = lambda B: next(x for x in (2.7, 3.0, 3.4, 4.2, 4.6, 9) if x >= B)
PH = {2.7: "–", 3.0: "–", 3.4: "0.35", 4.2: "0.12", 4.6: "0.06"}
print("\n| model | s (host) | k max, 3 cores | k max, 4 cores | s needed at k 1.55, 3 / 4 cores | r min GB/s at s | Q4_0 GB | R69 step that holds it | P(step held) | at C 3.1: type that fits, printed tok/s |\n|" + "---|" * 10)
for key, m, WL, WH, s in rows:
    if m[0] not in S: continue
    B = WL * Q4 + WH * Q6; km = [(C[n] * T / 1000 - WH) / (WL * (1 - s)) for n in (3, 4)]
    sn = [1 - (C[n] * T / 1000 - WH) / (WL * 1.55) for n in (3, 4)]
    rmin = (WH * Q6 + WL * Q4 * (1 - 0.889 * s)) / T * 1000
    # largest type whose bytes fit C - 0.15 = 2.95 GB; (bytes ratio, kernel cost per weight vs Q4_0 sdot: device R41 / published)
    typ = next(((t, q, kk) for t, q, kk in (("Q4_0", 1, 1), ("Q3_K_M", .862, 2.0), ("IQ3_S", .788, 3.0), ("IQ3_XXS", .703, 3.0)) if B * q <= 2.95))
    fit = "%s (%.2f GB): %.1f" % (typ[0], B * typ[1], 950 / ((WL + WH) * typ[2] / C[3] * 1000 + 8))
    print("| %s | %.2f | %.2f | %.2f | %.2f / %.2f | %.1f | %.2f | %.1f | %s | %s |" % (m[0], s, km[0], km[1], sn[0], sn[1], rmin, B, step(B), PH[step(B)], fit))
# the one check: memo-06's Yi row is reproduced when its inputs are put back (c 69.9, O 5, s 0.40, k 1.55, r 35 -> 11.5)
WL, WH, _ = shape(M[0]); C[9] = 69.9; assert abs(950 / ms(WL, WH, 35, .40, 1.55, 5, 9) - 11.5) < 0.05
# FFN-only levers screened on Mistral and Qwen2.5 only (host: memo-05 E1; Tom's T1, tom-screens.md section 6).
# E1: `down` input columns at s 0.60 through K1. T1: whole neurons in groups of four rows of gate+up at k 1.0, share g
# (clustered order, exact score); proxy priced as Ebony does (0.25 of gate+up bytes, 0.2 of its compute) or free (bound).
def ffn(name, r, n, O, e1k=None, g=None, proxy=True):
    m = next(x for x in M if x[0] == name); WL, WH, _ = shape(m); F = 3 * m[3] * m[4] * m[5] / 1e9; A = WL - F
    by, co = [F * 2 / 3, F / 3], [F * 2 / 3, F / 3]                                # [gate+up, down], G weights
    if e1k: by[1] *= 1 - 0.889 * 0.6; co[1] *= 0.4 * e1k
    if g: by[0] *= (1 - g) + 0.25 * proxy; co[0] *= (1 - g) + 0.2 * proxy
    return 950 / (max((WH * Q6 + (A + sum(by)) * Q4) / r, (WH + A + sum(co)) / C[n]) * 1000 + O)
print("\n| model | lever | r 24 | r 35 | r 35, 4 cores | r 45, 4 cores |\n|" + "---|" * 6)
for name, g in (("Mistral-7B-Instruct-v0.3", .33), ("Qwen2.5-7B-Instruct", .36)):
    for lab, kw_lo, kw_hi in (("E1 `down` columns, s 0.60, K1 k 2.2 to 1.55", dict(e1k=2.2), dict(e1k=1.55)),
                              ("T1 groups, g %.2f, with proxy to proxy-free" % g, dict(g=g), dict(g=g, proxy=False))):
        cells = ["%.1f to %.1f" % (max(ffn(name, r, n, 12), ffn(name, r, n, 12, **kw_lo)), max(ffn(name, r, n, 6), ffn(name, r, n, 6, **kw_hi)))
                 for r, n in ((24, 3), (35, 3), (35, 4), (45, 4))]
        print("| %s | %s | %s |" % (name, lab, " | ".join(cells)))
