#!/usr/bin/env python3
"""memo-10 Ebony review: re-price widths with an unpack-once cost split. Every output is *estimated*.
Uses Bonnie's model (bonnie_tree.py) unchanged; only c(W), the compute of a W-position pass in stock gemvs, varies.
Split from the A78 issue count of ebony_gw.c (NDK r28 clang -O3 -mcpu=cortex-a78, tools/c3_a78_pipes.py):
  stock 25 slots per 128 weights; W-vector kernel 10 shared (8 unpack, FCVTL of the weight scales) + 13 per position
  (8 SDOT, MOVI, SCVTF x2, FMUL, FMLA)."""
import bonnie_tree as bt

CW = {
    "Bonnie 1+0.45(W-1)":            lambda W: 1 + 0.45 * (W - 1),
    "count (10+13W)/25":             lambda W: 1.0 if W == 1 else (10 + 13 * W) / 25,
    "floor: W=2 at 18/15.1, +0.43":  lambda W: 1.0 if W == 1 else 18 / 15.1 + 0.43 * (W - 2),   # stock not issue-bound
    "what-if flat 0.30":             lambda W: 1 + 0.30 * (W - 1),
    "what-if flat 0.15":             lambda W: 1 + 0.15 * (W - 1),
}
ALONE = dict(C=40)
CENTRE_B = dict(C=40, g=8, disp=3, hs=0.5, c5=0.93, O=5.2)
CENTRE_E = dict(C=38.6, g=8, disp=8, hs=0.5, c5=0.93, O=5.2)
TOP = dict(C=50, g=15, disp=3, hs=0.5, c5=0.93, O=5.2)
# EAGLE-size draft, ms per draft step: FC 2h->h (33.5 M weights) + one body layer + 8,192-row shortlist of the head
DRAFT = {"Yi": 120.6 / 32 + 0.73 + 1.16, "Llama-2": 140.6 / 32 + 0.73 + 1.13, "DeepSeek": 130.8 / 30 + 0.73 + 1.11}

def row(kw, subj=None, widths=bt.WIDTHS):
    out = []
    for n, s in (subj or bt.SUBJ).items():
        cand = [(bt.ms(s, 1, 0, 0, **kw)[0], 1)] + [(bt.best(s, W, **kw)[0], W) for W in widths[1:]]
        t, W = min(cand); out.append(f"W={W} {950 / t:.2f}")
    return " | ".join(out)

def chain(s, W, c2, kw):   # chain of W-1 guesses, depth-n conditional accuracy = a1 * c2 for n >= 2
    acc, p = 0.0, 1.0
    for i in range(W - 1):
        p *= s["a1"] * (1 if i == 0 else c2); acc += p
    return bt.ms(s, W, acc, W - 1, **kw)[0]

if __name__ == "__main__":
    bt.selfcheck()
    for name, f in CW.items():
        bt.c = lambda W, e=None, f=f: f(W)
        if name.startswith("Bonnie"):   # her printed cells
            assert [round(950 / bt.best(s, 2, **ALONE)[0], 2) for s in bt.SUBJ.values()] == [7.94, 7.79, 7.38]
            assert [round(950 / bt.best(s, 2, **CENTRE_B)[0], 2) for s in bt.SUBJ.values()] == [9.82, 9.49, 9.25]
        print(f"\n## {name}:  c(W) " + " ".join(f"{f(W):.2f}" for W in bt.WIDTHS))
        for C in (50, 40, 35):
            print(f"  kB at C {C}: " + " ".join(f"W={W} {bt.kB(W, C):.2f}" for W in bt.WIDTHS[1:]))
        print("  per-width alone C 40 (Yi/L/D): " + "; ".join(
            f"W={W} " + "/".join(f"{950 / bt.best(s, W, **ALONE)[0]:.2f}" for s in bt.SUBJ.values()) for W in bt.WIDTHS[1:]))
        for lab, kw in (("alone C 40", ALONE), ("centre stack, Bonnie inputs", CENTRE_B), ("centre stack, Ebony inputs", CENTRE_E), ("top of bands", TOP)):
            print(f"  best width, {lab}: {row(kw)}")
        # EAGLE-size draft what-if: published-like accuracy, chain, c2 0.92 (EAGLE 1-alpha / 0-alpha)
        for a1 in (0.60, 0.75):
            subj = {n: dict(s, a1=a1, d=0.90, guess=DRAFT[n]) for n, s in bt.SUBJ.items()}
            for lab, kw in (("centre B", CENTRE_B), ("centre E", CENTRE_E), ("top", TOP)):
                cells = []
                for n, s in subj.items():
                    t, W = min((chain(s, W, 0.92, kw), W) for W in (2, 3, 4, 6))
                    cells.append(f"W={W} {950 / t:.2f}")
                print(f"  draft layer top-1 {a1}, {lab}: " + " | ".join(cells))
