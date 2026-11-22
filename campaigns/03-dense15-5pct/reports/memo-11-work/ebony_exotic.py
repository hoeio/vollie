#!/usr/bin/env python3
"""memo-11 (Ebony): prices of the exotic routes. Every output is *estimated* (arithmetic on R72 *device* inputs,
tom-head.md *host* accuracies and *published* acceptance). Uses Bonnie's model (memo-10-work/bonnie_tree.py) unchanged."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "memo-10-work"))
import bonnie_tree as bt

GOAL = 950 / 12                      # 79.17 ms printed
LAYERS = {"Yi": 32, "Llama-2": 32, "DeepSeek": 30}
S = bt.SUBJ

def tok(ms): return 950 / ms

def need_b(s, a, W=2, C=40.0, hs=1.0, O=5.2, guess=None, g=0.0, disp=0.0, c2=0.92):
    """Largest body-time factor b (bytes x kernel cost, applied as c5) at which the subject prints 12, chain of W-1."""
    acc, p = 0.0, 1.0
    for i in range(W - 1):
        p *= a if i == 0 else a * c2; acc += p
    sub = dict(s, a1=a, guess=s["guess"] if guess is None else guess)
    lo, hi = 0.0, 2.0
    for _ in range(50):
        mid = (lo + hi) / 2
        if bt.ms(sub, W, acc, W - 1, C=C, O=O, hs=hs, c5=mid, g=g, disp=disp)[0] <= GOAL: lo = mid
        else: hi = mid
    return lo

def ring(n, s, a, W=2, l=2, C=40.0, adapter=2.8, c2=0.9, **kw):
    """Prefix-exit self-speculation (Kangaroo mapped to a bus-bound phone): each drafted token re-reads the first l
    layers alone (bus rate, width 1), then rides the deep layers at width W. adapter = one attention block in Q8_0
    (67 M weights at 24 GB/s) per draft step; plus the 8,192-row shortlist already in s['guess'] (minus Tom's block)."""
    f = l / LAYERS[n]
    acc, p = 0.0, 1.0
    for i in range(W - 1):
        p *= a if i == 0 else a * c2; acc += p
    tau = 1 + s["d"] * acc
    hs, O, c5 = kw.get("hs", 1.0), kw.get("O", 8.0), kw.get("c5", 1.0)
    g, disp = kw.get("g", 0.0), kw.get("disp", 0.0)
    lane = 26.0 / (26.0 + 0.64 * g / bt.c(W))
    body = s["body"] * c5 * lane * ((1 - f) * bt.kB(W, C) + f * W)          # prefix read W times at width 1
    guess = (adapter + s["guess"] - 1.48) * (W - 1)                           # 1.48 ms = Tom's two blocks
    return (body + disp + O + guess + s["xpos"] * (W - 1)) / tau + s["head"] * hs, tau

def skipdraft(s, alpha=0.9, skip=0.45, n=1, C=40.0, O=8.0):
    """SWIFT / Draft & Verify class: n draft tokens through (1 - skip) of the layers (their bytes cross the bus),
    then one verify pass at width n + 1. No head; each draft step still needs the output head (full)."""
    tau = sum(alpha ** i for i in range(n + 1))
    body = s["body"] * (n * (1 - skip) + bt.kB(n + 1, C))
    return (body + O + n * s["head"]) / tau + s["head"], tau

def selfcheck():
    bt.selfcheck()
    # ring with l = 0 and no adapter must equal Bonnie's width-2 chain
    for n, s in S.items():
        a, b = ring(n, s, s["a1"], l=0, adapter=1.48)[0], bt.ms(s, 2, s["a1"], 1)[0]
        assert abs(a - b) < 1e-9, (a, b)
    # need_b at the measured head must be > 1 exactly when Bonnie's stack prints >= 12
    assert need_b(S["Yi"], 0.379) < 1.0
    y = S["Yi"]; b = need_b(y, 0.6)
    assert abs(bt.ms(dict(y, a1=0.6), 2, 0.6, 1, C=40, O=5.2, c5=b)[0] - GOAL) < 1e-6

if __name__ == "__main__":
    selfcheck()
    print("## 1. body-time factor b needed for 12, CPU only, exact, O 5.2 (W=2 | best of W=2,3)")
    for lab, kw in (("full head", dict(hs=1.0)), ("+ C7 head x 0.5", dict(hs=0.5))):
        for C in (40, 35):
            print(f"  {lab}, C {C}")
            for a in ("meas", 0.50, 0.60, 0.75, 0.85):
                row = []
                for n, s in S.items():
                    av = s["a1"] if a == "meas" else a
                    gs = None if (a == "meas" or av <= 0.5) else {"Yi": 5.7, "Llama-2": 6.2, "DeepSeek": 6.2}[n]  # draft layer
                    b2 = need_b(s, av, 2, C=C, guess=gs, **kw); b3 = need_b(s, av, 3, C=C, guess=gs, **kw)
                    row.append(f"{b2:.2f} | {max(b2, b3):.2f}")
                print(f"    top-1 {a}: " + "   ".join(row))
    print("  same with the GPU lane g 8, dispatch 8 (Ebony), C 38.6, C7:")
    for a in ("meas", 0.60, 0.75):
        row = []
        for n, s in S.items():
            av = s["a1"] if a == "meas" else a
            gs = None if a == "meas" else {"Yi": 5.7, "Llama-2": 6.2, "DeepSeek": 6.2}[n]
            row.append(f"{max(need_b(s, av, W, C=38.6, hs=0.5, guess=gs, g=8, disp=8) for W in (2, 3)):.2f}")
        print(f"    top-1 {a}: " + "   ".join(row))

    print("\n## 2. prefix-exit ring, alone (O 8), tok/s: Yi / Llama-2 / DeepSeek   [head W=2 alone: 7.94 / 7.79 / 7.38]")
    for C in (40, 35):
        for l in (2, 4):
            for a in (0.50, 0.60, 0.70, 0.80):
                best = []
                for n, s in S.items():
                    t, W = min((ring(n, s, a, W=W, l=l, C=C)[0], W) for W in (2, 3, 4))
                    best.append(f"W={W} {tok(t):.2f}")
                print(f"  C {C}, exit layer {l}, top-1 {a}: " + " / ".join(best))
    print("  ring in the centre stack (G 8 disp 8, C7, C5 0.93, O 5.2, C 38.6):")
    for a in (0.60, 0.70, 0.80):
        best = []
        for n, s in S.items():
            t, W = min((ring(n, s, a, W=W, l=2, C=38.6, g=8, disp=8, hs=0.5, c5=0.93, O=5.2)[0], W) for W in (2, 3, 4))
            best.append(f"W={W} {tok(t):.2f}")
        print(f"    top-1 {a}: " + " / ".join(best))

    print("\n## 3. layer-skip self-draft (no head), alone, tok/s  [stock 6.90 / 6.21 / 6.22]")
    for C in (50, 40):
        for alpha, skip in ((0.80, 0.45), (0.90, 0.45), (0.97, 0.50)):
            row = []
            for n, s in S.items():
                t, k = min((skipdraft(s, alpha, skip, k, C)[0], k) for k in (1, 2, 3))
                row.append(f"n={k} {tok(t):.2f}")
            print(f"  C {C}, alpha {alpha}, skip {skip}: " + " / ".join(row))

    print("\n## 4. GPU lanes (lane alone, O 8): body factor and tok/s")
    # lane = 26 / (26 + g * (1 - 0.36 * rho)); rho = bytes the GPU reads per Q4_0 byte it replaces (R72: the CPU
    # loses 0.36 GB/s per GB/s the GPU reads). Q4_0 kernel rho 1 (Bonnie's 0.64); exact int8 copy rho 34/18 = 1.89.
    lanef = lambda g, rho: 26.0 / (26.0 + g * (1 - 0.36 * rho))
    assert abs(lanef(8, 1) - 26.0 / (26.0 + 0.64 * 8)) < 1e-12
    for lab, lane, disp in (("upstream Q4_0 gemv, g 8, dispatch 8", lanef(8, 1), 8),
                            ("exact int8 copy at the GPU's whole bus share (25.8 / 1.89 = 13.7 GB/s Q4_0-equivalent)", lanef(13.7, 34 / 18), 8),
                            ("perfect Q4_0 lane: read sum 46.5", 26.0 / 46.5, 0)):
        print(f"  {lab}: lane {lane:.3f}: " + " / ".join(f"{tok(s['body'] * lane + disp + 8 + s['head']):.2f}" for s in S.values()))
    print("  perfect Q4_0 lane + measured head W=2 (kB 1.12) + O 5.2, exact apart from the lane:")
    print("    " + " / ".join(f"{tok((s['body'] * 26.0 / 46.5 * 1.12 + 5.2 + s['guess']) / (1 + s['d'] * s['a1']) + s['head']):.2f}" for s in S.values()))

    print("\n## 5. what-if: CPU port at r instead of 29.9 (DDR step hypothesis), stock W=1, O 8")
    for r in (29.9, 38.0, 46.0):
        row = []
        for n, s in S.items():
            t = lambda R, C=40.0: max(1 / R, 1 / C) + 0.25 * min(1 / R, 1 / C)
            row.append(f"{tok(s['body'] * t(r) / t(29.9) + 8 + s['head']):.2f}")
        print(f"  r {r}: " + " / ".join(row))
