#!/usr/bin/env python3
"""memo-10 pricing (Bonnie): verify passes wider than 2. Every output is *estimated*.

Inputs: R72 *device* (body, head, read 29.9 GB/s, A78 0.118 cycles per weight), tom-head.md *host* (top-1, tokens
per pass), Ebony's two-point fit (memo-09 section 10.3). Top-k and depth-2 acceptance are NOT measured: see TOPK, C2.

  pass time factor  kB(W) = t(c(W)) / t(1),  t(c) = max(1/R, c/C) + 0.25 * min(1/R, c/C)
     R = 29.9 GB/s read, C = hot compute rate of the 4 threads in GB/s, c(W) = 1 + E * (W - 1), E = 0.45
  ms per committed token = (body*lane*kB*c5 + disp + O + guess + xpos*(W-1)) / tau + head*hs
     lane = 26 / (26 + 0.64 * g / c(W));   tau = 1 + d * sum(path probabilities of the W-1 candidate nodes)
  printed tok/s = 950 / ms
"""
import heapq, random

R, E = 29.9, 0.45
# body ms, head ms, guess ms per head evaluation (tok block x2 + 8,192-row shortlist), top-1, d, xpos ms
#   d = (tau - 1) / top-1 as measured by Tom (pass-weighted against row-weighted accuracy)
#   DeepSeek: head NOT trained or measured; top-1 and d are the mean of the other two (*estimated*)
SUBJ = {
    "Yi":       dict(body=120.6, head=9.04,  guess=2.64, a1=0.379, d=(1.326 - 1) / 0.379, xpos=0.5),
    "Llama-2":  dict(body=140.6, head=4.40,  guess=2.61, a1=0.497, d=(1.448 - 1) / 0.497, xpos=1.5),
    "DeepSeek": dict(body=130.8, head=13.88, guess=2.59, a1=0.438, d=0.88,                xpos=1.5),
}
TOPK = 0.98   # top-5 = top-1 shifted by +0.98 in log-odds (Together's Medusa post: 0.60 -> 0.80); ranks by 1/r
C2 = 1.0      # P(depth n+1 top-1 right | depth n right) = C2 * top-1  (judgement; band 0.7 to 1.15)
WIDTHS = (1, 2, 3, 4, 6, 8)

def c(W, e=E): return 1 + e * (W - 1)

def kB(W, C, e=E):
    t = lambda x: max(1 / R, x / C) + 0.25 * min(1 / R, x / C)
    return t(c(W, e)) / t(1.0)

def rank_probs(a1, n=8):
    """P(the rank-r candidate is the right one), r = 1..n."""
    lo = lambda p: __import__("math").log(p / (1 - p))
    a5 = 1 / (1 + __import__("math").exp(-(lo(a1) + TOPK)))
    unit = (a5 - a1) / sum(1 / r for r in range(2, 6))
    return [a1] + [unit / r for r in range(2, n + 1)]

def tree(a1, n_nodes, shape="greedy", c2=C2):
    """Sum of path probabilities of the candidate nodes; returns (sum, max depth, description)."""
    q1 = rank_probs(a1)
    qn = [x * c2 for x in q1]                      # conditional distribution at depth >= 2
    if n_nodes == 0: return 0.0, 0, "-"
    if shape == "topk":                             # all candidates at depth 1
        return sum(q1[:n_nodes]), 1, f"top-{n_nodes} at depth 1"
    if shape == "chain":
        s, p = 0.0, 1.0
        for i in range(n_nodes):
            p *= a1 if i == 0 else a1 * c2
            s += p
        return s, n_nodes, f"chain of {n_nodes}"
    if shape == "two":                              # k1 at depth 1, the rest under the best depth-1 node
        best = max(((sum(q1[:k1]) + q1[0] * sum(qn[:n_nodes - k1]), k1) for k1 in range(1, n_nodes + 1)))
        return best[0], 2 if best[1] < n_nodes else 1, f"{best[1]} at depth 1 + {n_nodes - best[1]} at depth 2"
    # greedy (Medusa's construction): take the n nodes with the highest path probability
    heap, s, depth, per_depth = [(-q1[0], 1, 0, 1.0)], 0.0, 0, {}
    for _ in range(n_nodes):
        negp, dep, r, parent = heapq.heappop(heap)
        s += -negp; depth = max(depth, dep); per_depth[dep] = per_depth.get(dep, 0) + 1
        dist = q1 if dep == 1 else qn
        if r + 1 < len(dist): heapq.heappush(heap, (-parent * dist[r + 1], dep, r + 1, parent))   # next sibling
        heapq.heappush(heap, (negp * qn[0], dep + 1, 0, -negp))                                   # first child
    return s, depth, "+".join(str(per_depth[k]) for k in sorted(per_depth))

def ms(s, W, acc_sum, depth, C=40.0, g=0.0, disp=0.0, O=8.0, hs=1.0, c5=1.0, e=E):
    lane = 26.0 / (26.0 + 0.64 * g / c(W, e))
    tau = 1 + s["d"] * acc_sum
    guess = s["guess"] * depth                      # one head evaluation per candidate depth
    return (s["body"] * lane * kB(W, C, e) * c5 + disp + O + guess + s["xpos"] * (W - 1)) / tau + s["head"] * hs, tau

def best(s, W, **kw):
    """Best static shape at width W: (ms, tau, description)."""
    out = []
    for shape in ("topk", "chain", "two", "greedy"):
        a, dep, desc = tree(s["a1"], W - 1, shape)
        t, tau = ms(s, W, a, dep, **kw)
        out.append((t, tau, shape + ": " + desc))
    return min(out)

def gated(s, C=40.0, conc=1.5, n=20000, **kw):
    """Width 1 or 2 chosen per step from the head's own confidence q (assumed calibrated, q ~ Beta with mean
    top-1 and concentration conc: an ASSUMPTION until Tom prints the real histogram). Dinkelbach on tokens/time."""
    rnd = random.Random(0)
    qs = [rnd.betavariate(s["a1"] * conc, (1 - s["a1"]) * conc) for _ in range(n)]
    hs = kw.get("hs", 1.0)
    t = {W: ms(dict(s, d=0.0), W, 0, min(W - 1, 1), C=C, **kw)[0] - s["head"] * hs for W in (1, 2)}
    t[1] += s["guess"]                              # the head still runs to produce the confidence
    lam = 1 / t[1]
    for _ in range(30):
        tok = tim = 0.0
        for q in qs:
            w = 2 if (1 + s["d"] * q) - lam * t[2] > 1 - lam * t[1] else 1
            tok += 1 + (s["d"] * q if w == 2 else 0); tim += t[w]
        lam = tok / tim
    return tim / tok + s["head"] * hs, tok / n

def selfcheck():
    stock = [round(950 / ms(s, 1, 0, 0)[0], 2) for s in SUBJ.values()]
    assert stock == [6.90, 6.21, 6.22], stock                       # R72 section 5, memo-09
    y = SUBJ["Yi"]                                                  # tom-head.md section 5, Yi tok, kB 1.08: 115.3
    t = (y["body"] * 1.08 + 8 + y["guess"]) / 1.326 + y["head"]
    assert abs(t - 115.3) < 0.05, t
    assert abs(kB(2, 50) - 1.06) < 0.005 and abs(kB(2, 35) - 1.22) < 0.01   # Ebony 10.3
    assert abs(tree(0.5, 3, "chain")[0] - (0.5 + 0.25 + 0.125)) < 1e-9
    assert tree(0.4, 5, "greedy")[0] >= max(tree(0.4, 5, x)[0] for x in ("topk", "chain", "two")) - 1e-9

STACKS = [
    ("alone, full clocks (C 50)", dict(C=50)),
    ("alone, centre (C 40)", dict(C=40)),
    ("alone, L4 caps (C 35)", dict(C=35)),
    ("+ O 5.2 (exact only), C 40", dict(C=40, O=5.2)),
    ("+ G(8, disp 3) + C7 + C5 + O 5.2, C 40 [Bonnie]", dict(C=40, g=8, disp=3, hs=0.5, c5=0.93, O=5.2)),
    ("same, disp 8, C 38.6 [Ebony inputs]", dict(C=38.6, g=8, disp=8, hs=0.5, c5=0.93, O=5.2)),
    ("top of bands: G(15) + C7 + C5 + O 5.2, C 50", dict(C=50, g=15, disp=3, hs=0.5, c5=0.93, O=5.2)),
]

if __name__ == "__main__":
    selfcheck()
    print("kB(W):  C  " + "  ".join(f"W={W}" for W in WIDTHS))
    for C in (55, 50, 40, 35, 24):
        print(f"       {C:>3}  " + "  ".join(f"{kB(W, C):.2f}" for W in WIDTHS), f"  crossover W = {1 + (C / R - 1) / E:.2f}")
    for n, s in SUBJ.items():
        print(n, "cumulative top-k 1..7:", " ".join(f"{sum(rank_probs(s['a1'])[:k]):.3f}" for k in range(1, 8)))
    for name, kw in STACKS:
        print(f"\n## {name}\n| W | " + " | ".join(SUBJ) + " |\n|---|---|---|---|")
        for W in WIDTHS:
            cells = []
            for s in SUBJ.values():
                t, tau, desc = best(s, W, **kw) if W > 1 else (*ms(s, 1, 0, 0, **kw), "-")
                cells.append(f"{desc}; tau {tau:.2f}; {t:.1f} ms -> {950 / t:.2f}")
            print(f"| {W} | " + " | ".join(cells) + " |")
        print("| 1 or 2, gated | " + " | ".join(f"tau {tau:.2f}; {t:.1f} ms -> {950 / t:.2f}"
              for t, tau in (gated(s, **kw) for s in SUBJ.values())) + " |")
    # shapes side by side at C 40, alone
    print("\n## shapes, alone, C 40: tau / tok/s")
    for W in WIDTHS[1:]:
        for shape in ("topk", "chain", "two", "greedy"):
            row = []
            for s in SUBJ.values():
                a, dep, desc = tree(s["a1"], W - 1, shape); t, tau = ms(s, W, a, dep, C=40)
                row.append(f"{desc}: {tau:.2f} / {950 / t:.2f}")
            print(f"| {W} | {shape} | " + " | ".join(row) + " |")
    # tau needed for 12 printed (79.17 ms) per width, and the oracle (tau = W)
    for name, kw in STACKS[3:]:
        print(f"\n## tau needed for 12: {name}")
        for W in WIDTHS[1:]:
            row = []
            for s in SUBJ.values():
                num = ms(dict(s, d=0.0), W, 0, 1, **kw)[0] - s["head"] * kw.get("hs", 1.0)
                row.append(f"{num / (79.17 - s['head'] * kw.get('hs', 1.0)):.2f}")
            print(f"| {W} | " + " | ".join(row) + f" | (ceiling {W}) |")
    # top-1 at which width W first beats width W-1 (best static shape), alone
    print("\n## top-1 needed for width W to beat W-1 (Yi body, d 0.88)")
    for C in (50, 40, 35):
        row = []
        for W in (2, 3, 4):
            need = next((a / 100 for a in range(5, 100)
                         if best(dict(SUBJ["Yi"], a1=a / 100, d=0.88), W, C=C)[0]
                         < (best(dict(SUBJ["Yi"], a1=a / 100, d=0.88), W - 1, C=C)[0] if W > 2 else ms(SUBJ["Yi"], 1, 0, 0)[0])), None)
            row.append(f"W={W}: {need}")
        print(f"C {C}: " + ", ".join(row))
    # sensitivity: cheaper extra positions (E) and better deep heads (C2)
    print("\n## sensitivity, alone C 40: best width and tok/s")
    for e in (0.30, 0.45, 0.60):
        row = []
        for s in SUBJ.values():
            cand = [(ms(s, 1, 0, 0, C=40)[0], 1)] + [(min(ms(s, W, *tree(s["a1"], W - 1, sh)[:2], C=40, e=e)[0]
                    for sh in ("topk", "chain", "two", "greedy")), W) for W in WIDTHS[1:]]
            t, W = min(cand); row.append(f"W={W} {950 / t:.2f}")
        print(f"E {e}: " + " | ".join(row))
