#!/usr/bin/env python3
"""memo-12 round 28 (Bonnie): the drivers behind bonnie-joins-r27.md that were not stored: (a) section 1.1's half against
float sum, (b) section 1.4's spread of head hits per 31 steps, (c) section 2.5's sweep of P2 / P1 and the stack model's
ratios. Uses bonnie_chooser_r27.py's functions unchanged. *host* counts, *estimated* rates; held-out training prompts only.
Run: lab/venv-c3/bin/python3 -I -B bonnie_r27_tables.py <repo root>      (about 2 minutes)"""
import sys, os, importlib.util
import numpy as np
root = sys.argv[1]; here = os.path.dirname(os.path.abspath(__file__))
_s = importlib.util.spec_from_file_location("ch", os.path.join(here, "bonnie_chooser_r27.py")); ch = importlib.util.module_from_spec(_s); _s.loader.exec_module(ch)
t2, bt, r5 = ch.t2, ch.bt, ch.r5

# (a) half against float accumulation of a weighted value sum (Ebony r8 section 1.2 row 1 measured 1.4e-3)
rng = np.random.default_rng(0); out = []
for _ in range(200):
    w = np.exp(rng.normal(0, 2, 64)).astype(np.float32); v = rng.normal(0, 1, (64, 128)).astype(np.float32)
    a16 = np.zeros(128, np.float16); a32 = np.zeros(128, np.float32)
    for i in range(64): a16 = (a16 + np.float16(w[i]) * v[i].astype(np.float16)).astype(np.float16); a32 += w[i] * v[i]
    out.append(np.abs(a16.astype(np.float32) - a32).max() / np.abs(a32).max())
print("(a) half vs float value sum, 64 rows: median %.1e, 10-90 %% %.1e to %.1e" % (np.median(out), np.quantile(out, .1), np.quantile(out, .9)))

# (c, first half) the stack model's P2 / P1
print("(c) P2 / P1 from the stack model:")
for subj in ch.M:
    for K in (300, 452):
        for cg in (ch.r13.CG, 2.0):
            cells = []
            for model in ("pair", "linear"):
                keep = t2.rates
                if model == "linear": t2.rates = r5.rates_linear
                try: b1 = t2.body(subj, 1, K, "pair", c5=1.0)[0]; b2 = t2.body(subj, 2, K, "pair", cg=cg, c5=1.0)[0]
                finally: t2.rates = keep
                cells.append(f"{model} {(b2 + 5.2 + bt.SUBJ[subj]['xpos'] + ch.ATT2[subj]) / (b1 + 5.2):.3f}")
            print(f"    {subj} K {K} cg {cg:.2f}: " + "; ".join(cells))

for name in ch.M:
    stp, top1, tot = ch.held_steps(name); prm = sorted(stp); folds = [set(prm[i::5]) for i in range(5)]
    if name == "Llama-2":                                      # (b)
        first = np.array([sum(x[3] == x[0] for x in st[:31]) for st, n in stp.values() if len(st) >= 31])
        allw = np.array([sum(x[3] == x[0] for x in st[i:i + 31]) for st, n in stp.values() for i in range(0, len(st) - 30, 31)])
        for lab, v in (("first 31 guess steps of each held-out prompt", first), ("all disjoint 31-step windows", allw)):
            print(f"(b) Llama-2 head3000-tok, {lab}: n {len(v)}, mean {v.mean():.2f}, sd {v.std(ddof=1):.2f} (binomial {np.sqrt(31 * top1 * (1 - top1)):.2f}), share with <= 6 hits {np.mean(v <= 6):.3f}")
    p1 = np.array([x[4] for st, _ in stp.values() for x in st]); grid = [0.0] + np.quantile(p1, np.arange(0.1, 1.0, 0.1)).round(3).tolist() + [1e9]
    P1, _, G, head = ch.costs(name, "pair")
    for ratio in (1.05, 1.10, 1.15, 1.20, 1.30, 1.40, 1.50):   # (c, second half)
        c = (P1, P1 * ratio, G, head); T = ch.sim(stp, ch.pol_T, *c)[0]; gT = gF = 0.0
        for f in folds:
            tr = set(prm) - f; rT = ch.sim(stp, ch.pol_T, *c, only=f)[0]
            v = max(grid, key=lambda v: ch.sim(stp, ch.make((v, v, 0, 0, 2)), *c, only=tr)[0])      # rule T + gate: one fitted number
            gT += ch.sim(stp, ch.make((v, v, 0, 0, 2)), *c, only=f)[0] / rT / 5
            gF += ch.sim(stp, ch.make(ch.fit(stp, tr, grid, c)), *c, only=f)[0] / rT / 5
        print(f"(c) {name} P2/P1 {ratio:.2f}: p* {(c[1] - P1) / (950 / T - head):.2f}; width 1 {950 / (P1 + head):.2f}; head always {ch.sim(stp, ch.pol_head, *c)[0]:.2f}; "
              f"rule T {T:.2f}; rule T + gate {100 * (gT - 1):+.2f} %; full F {100 * (gF - 1):+.2f} % (cross-fitted, over rule T)")
