#!/usr/bin/env python3
"""memo-09 pricing (Bonnie). Every output is *estimated*: arithmetic on R72 *device* inputs.

ms per committed token = (body * lane * kB * c5 + disp + O) / (1 + p) + head * hs + xh
  lane = cpu / (cpu + 0.64 * g_eff): a second bus master computing g GB/s costs the CPU 0.36 GB/s per GB/s
         (R72 block a: CPU read 29.90 -> 20.70 while the GPU read 25.81); g_eff = g / 1.45 at batch 2.
  kB   = pass-time factor of a batch-2 weight pass on the CPU (1.0 = bus-bound, as hoped).
  p    = P(the guessed next token is right) -> 1 + p committed tokens per weight pass.
printed tok/s = 950 / ms.
"""
SUBJ = {  # ggml stock body ms (4-7), head ms (Q6_K, 7,4-5): R72 section 5, *device*
    "Yi": (120.6, 9.04), "Llama-2": (140.6, 4.40), "DeepSeek": (130.8, 13.88)}
CPU = 26.0  # GB/s, ggml Q4_0 cold on 4-7 (25.8 to 26.7, *device*)

def ms(body, head, g=0.0, p=0.0, kB=1.0, disp=0.0, O=8.0, hs=1.0, c5=1.0, xh=0.0):
    g_eff = g / 1.45 if p > 0 else g
    lane = CPU / (CPU + 0.64 * g_eff)
    return (body * lane * (kB if p > 0 else 1.0) * c5 + disp + O) / (1 + p) + head * hs + xh

STACKS = [
    ("stock (K0, ggml)", {}),
    ("H: Hexagon lane g 5, dispatch 6.4", dict(g=5, disp=6.4)),
    ("G: GPU lane g 8, dispatch 3", dict(g=8, disp=3)),
    ("G at g 15", dict(g=15, disp=3)),
    ("L: lookahead p 0.15, kB 1.20 (low)", dict(p=0.15, kB=1.20)),
    ("L: lookahead p 0.25, kB 1.08 (centre)", dict(p=0.25, kB=1.08)),
    ("L: lookahead p 0.40, kB 1.00 (high)", dict(p=0.40, kB=1.00)),
    ("M: trained next-next head p 0.60, kB 1.08, +5 ms", dict(p=0.60, kB=1.08, xh=5)),
    ("L + G(8)", dict(p=0.25, kB=1.08, g=8, disp=3)),
    ("L + G(8) + head screen 0.5 + O 5.2 + C5 0.93", dict(p=0.25, kB=1.08, g=8, disp=3, hs=0.5, O=5.2, c5=0.93)),
    ("L + G(15) + head screen + O 5.2 + C5", dict(p=0.25, kB=1.08, g=15, disp=3, hs=0.5, O=5.2, c5=0.93)),
    ("M + G(8)", dict(p=0.60, kB=1.08, xh=5, g=8, disp=3)),
    ("M + G(8) + head screen + O 5.2 + C5", dict(p=0.60, kB=1.08, xh=5, g=8, disp=3, hs=0.5, O=5.2, c5=0.93)),
    ("M + G(8) + H(5) + head screen + O 5.2 + C5", dict(p=0.60, kB=1.08, xh=5, g=13, disp=9.4, hs=0.5, O=5.2, c5=0.93)),
]

def table():
    rows = []
    for name, kw in STACKS:
        cells = []
        for s, (b, h) in SUBJ.items():
            t = ms(b, h, **kw)
            cells.append((t, t - ms(b, h), 950 / t))
        rows.append((name, cells))
    return rows

if __name__ == "__main__":
    # self-check against R72 section 5's descriptive ggml line: 6.90 / 6.21 / 6.22
    got = [round(950 / ms(b, h), 2) for b, h in SUBJ.values()]
    assert got == [6.90, 6.21, 6.22], got
    print("| stack | " + " | ".join(SUBJ) + " |\n|---|---|---|---|")
    for name, cells in table():
        print(f"| {name} | " + " | ".join(f"{t:.1f} ms ({d:+.1f}) → {r:.1f}" for t, d, r in cells) + " |")
    # p needed for 12 printed (79.2 ms) with the centre GPU lane and the small levers
    for s, (b, h) in SUBJ.items():
        for g in (0, 8, 15):
            kw = dict(kB=1.08, g=g, disp=3 if g else 0, hs=0.5, O=5.2, c5=0.93)
            p = next((q / 100 for q in range(1, 300) if ms(b, h, p=q / 100, **kw) <= 79.17), None)
            print(f"p needed for 12, {s}, g {g}: {p}")
