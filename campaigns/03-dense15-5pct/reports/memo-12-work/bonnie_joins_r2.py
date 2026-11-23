#!/usr/bin/env python3
"""memo-12 round 2 (Bonnie): revised join bands, R75 (x4) prediction bands, and what a perfect lane still needs.
Every output is *estimated*. Loads bonnie_joins.py (and through it memo-10's bonnie_tree.py) by path, unchanged.
Run: python3 -I bonnie_joins_r2.py
"""
import importlib.util, os, random
_d = os.path.dirname(os.path.abspath(__file__))
_s = importlib.util.spec_from_file_location("bonnie_joins", os.path.join(_d, "bonnie_joins.py"))
bj = importlib.util.module_from_spec(_s); _s.loader.exec_module(bj); bt = bj.bt
SUBJ = list(bt.SUBJ)

# ---- 1. duty-aware split model, fitted to R74 x3 split (three shares, one shape) ---------------------------------
DEV0, DEVROW, GCONT, X = 14.0, 0.0801, 1.10, 0.45   # GPU START->END us = 14 + 0.0801 * rows (x3 rt 2816, alone 11008);
                                                    # x 1.10 beside a busy CPU (x3 tog); CPU slowed 1 + 0.45 * GPU duty
def split(A_ms, rows_total, f, K_us, Lh_us=0.0):
    """(t_cpu, t_gpu, t) ms for one stage of CPU-alone time A; K = GPU dead time; Lh = launch time taken from the CPU side."""
    gdev = GCONT * (DEV0 + DEVROW * f * rows_total) / 1e3
    base = (1 - f) * A_ms
    tc = (base + (base * base + 4 * X * base * gdev) ** 0.5) / 2          # t = base * (1 + X * gdev / t)
    tg = K_us / 1e3 + gdev
    return tc, tg, max(tc, tg) + Lh_us / 1e3

def check_split():
    # R74 x3 split, 11008 x 4096, A 0.948 ms, K in situ about 500 us: cpu_ms 0.820/0.823, 0.748/0.783, 0.623/0.664;
    # gpu_ms 0.719/0.796, 0.839/0.832, 1.010/0.857 (device)
    for f, cpu, gpu in ((0.25, 0.82, 0.76), (0.35, 0.765, 0.835), (0.50, 0.643, 0.93)):
        tc, tg, _ = split(0.948, 11008, f, 500)
        assert abs(tc - cpu) < 0.03 and abs(tg - gpu) < 0.08, (f, tc, tg)

# ---- 2. Monte Carlo for the x4 cells (2816 x 4096, R74 anchors: Q->S 196, S->St 199, args 20, end->return 37) -----
def ln(rnd, c, lo, hi):   # lognormal with median c and 80 % band lo..hi (asymmetric bands: use the wider side)
    import math
    s = max(math.log(hi / c), math.log(c / lo)) / 1.2816
    return c * math.exp(rnd.gauss(0, s))

def draw(rnd):
    """One world: speed-ups of the host part per launcher cell and of submit->start per hypothesis."""
    u = rnd.random()
    hyp = "k" if u < 0.50 else ("c" if u < 0.80 else "g")   # S->St follows: little-cluster state / launcher core / GPU (fixed)
    host = {("L", "fin"): ln(rnd, 1.1, 1.0, 1.45),   # the 0.5 s pre-spin decays once the thread sleeps in clFinish
             ("L", "spin"): ln(rnd, 1.7, 1.25, 2.3),
            ("M", "fin"): ln(rnd, 2.2, 1.3, 3.8), ("M", "spin"): ln(rnd, 3.0, 1.6, 5.0)}
    host[("X", "fin")], host[("X", "spin")] = host[("M", "fin")] * ln(rnd, 1.15, 1.0, 1.35), host[("M", "spin")] * ln(rnd, 1.15, 1.0, 1.35)
    big = ln(rnd, 2.2, 1.4, 3.5); lit = ln(rnd, 2.0, 1.4, 2.6); one = lambda: ln(rnd, 1.05, 1.0, 1.15)
    s2s = {}
    for c in "LMX":
        for w in ("fin", "spin"):
            if hyp == "g": d = one()
            elif hyp == "k": d = (lit if w == "spin" else 1 + (lit - 1) * 0.3) if c == "L" else one()
            else: d = (big if w == "spin" else 1 + (big - 1) * 0.5) if c != "L" else (1 + (big - 1) * (0.5 if w == "spin" else 0.2))
            s2s[(c, w)] = d
    fin, spin = ln(rnd, 37, 20, 60), ln(rnd, 8, 3, 20)
    cell = {}
    for k in host:
        h = 196 / host[k]; s = 199 / s2s[k]; a = 20 / host[k]; e = spin if k[1] == "spin" else fin / (1.2 if k[0] != "L" else 1.0)
        st = ln(rnd, 1.0, 0.87, 1.15)                           # state: R74 moved the same cell by +-100 us between blocks
        cell[k] = dict(ovh=(h + s + a + e) * st, q2s=(h + s) * st, s2s=s * st, host=h * st, end=e)
    prep = rnd.random() < 0.40                                  # the driver builds the command at enqueue (user-event gate hides Q->S)
    ser = rnd.random() < 0.60                                   # the driver submits kernel j only after kernel j-1 has ended
    return dict(hyp=hyp, cell=cell, prep=prep, ser=ser, rnd_gap=ln(rnd, 1.0, 0.8, 1.25))

def bands(xs):
    xs = sorted(xs); n = len(xs); return xs[n // 2], xs[n // 10], xs[n * 9 // 10]

def mc(n=20000):
    rnd = random.Random(75); W = [draw(rnd) for _ in range(n)]; out = {}
    for k in W[0]["cell"]:
        for fld in ("ovh", "q2s", "s2s"): out[f"x4b {k[0]} {k[1]} {fld}"] = bands([w["cell"][k][fld] for w in W])
    best = [min(c["ovh"] for c in w["cell"].values()) for w in W]
    out["x4b best ovh"] = bands(best)
    out["P(best ovh <= 150)"] = sum(b <= 150 for b in best) / n
    out["P(best ovh <= 250)"] = sum(b <= 250 for b in best) / n
    out["P(best ovh <= 100)"] = sum(b <= 100 for b in best) / n
    out["P(L spin s2s < M spin s2s - 30)"] = sum(w["cell"][("L", "spin")]["s2s"] < w["cell"][("M", "spin")]["s2s"] - 30 for w in W) / n
    # x4a: gap per queued launch END(j-1) -> START(j) = S->St + end handling (R74: 235 at this size by interpolation of 199..311)
    for c, lab in (("L", "0-3"), ("M", "cpu 4")):
        out[f"x4a gap batch {lab}"] = bands([(w["cell"][(c, "fin")]["s2s"] + 37) * w["rnd_gap"] for w in W])
        out[f"x4a QUEUED(8)-QUEUED(1) {lab}"] = bands([7 * 140 * w["cell"][(c, "fin")]["host"] / 196 * w["rnd_gap"] for w in W])   # 140 us of enqueue CPU per launch at the R74 state
    # x4f: user event set -> START; plain QUEUED -> START on the same core (cpu 4, spin) beside it
    out["x4f set->START cpu 4"] = bands([(w["cell"][("M", "spin")]["s2s"] + (15 if w["prep"] else w["cell"][("M", "spin")]["host"] * 0.8)) for w in W])
    out["x4f set->START 0-3"] = bands([(w["cell"][("L", "spin")]["s2s"] + (20 if w["prep"] else w["cell"][("L", "spin")]["host"] * 0.8)) for w in W])
    # x4d: replay gap per launch (given the recording works)
    out["x4d replay ovh per launch"] = bands([(ln(rnd, 60, 20, 150) if w["ser"] else w["cell"][("L", "fin")]["s2s"] + 37) for w in W])
    # in-situ K for x4s (launcher cpu 4, flush, event spin): GPU dead time = host + S->St + end; +50 for the in-situ excess seen in R74
    ks = [w["cell"][("M", "spin")] for w in W]
    out["x4s K in situ"] = bands([c["host"] + c["s2s"] + c["end"] + 20 / 3 + 40 for c in ks])
    out["x4s Lh (launch time taken from the CPU side)"] = bands([c["host"] for c in ks])
    return out, W

# ---- 3. stack prices ----------------------------------------------------------------------------------------------
def stack(subj, K, **kw): return bj.price(subj, K, **kw)["stack"]
def pg(subj, K): return bj.price(subj, K)["pg"]

DESIGNS_R2 = [  # name, K centre, low, high (us of GPU dead time per forced join), P
    ("E   shares re-balanced, join as measured (= Y12 level stacking)", 500, 452, 560, 0.6),
    ("B   E + event spin instead of clFinish", 470, 400, 560, 0.7),
    ("B+  launcher off the cold A55: fast core, hot little cluster or recorded queue (= Y13 / Y14)", 300, 150, 500, 0.35),
    ("A   B+ + launch-ahead behind a user event", 270, 130, 500, 0.3),
    ("C   A + submit->start removed (GPU or kernel side kept hot)", 220, 100, 500, 0.2),
    ("F   persistent kernel, shared-memory mailbox (= Y15)", 30, 5, 100, 0.10),
    ("Z   zero dead time", 0, 0, 0, None),
]

def best_generic(s, W, **kw):
    return min((bt.ms(s, W, *bt.tree(s["a1"], W - 1, sh)[:2], **kw)[0] for sh in ("topk", "chain", "two", "greedy")))

def perfect(subj, lane_scale=1.0, hs=0.5, c5=0.93, O=5.2, a1=None, C=40.0, head_gpu=1.0, W=2, e=bt.E):
    """Centre stack with a zero-dead-time lane; lane_scale multiplies the lane factor (0.621 at W 1)."""
    s = dict(bt.SUBJ[subj]);
    if a1 is not None: s["a1"] = a1
    # bonnie_tree's lane = 26 / (26 + 0.64 g / c(W)); scale it by moving g
    g = bj.G_EFF
    if lane_scale != 1.0:
        l1 = bj.lane(1) * lane_scale; g = 26 * (1 / l1 - 1) / 0.64
    t = best_generic(s, W, C=C, g=g, disp=0.0, hs=hs * head_gpu, c5=c5, O=O, e=e) if W > 1 else bt.ms(s, 1, 0, 0, C=C, g=g, disp=0.0, hs=hs * head_gpu, c5=c5, O=O)[0]
    return t

def solve(f, lo, hi, target=950 / 12.0, n=60):
    """bisection for f(x) = target on [lo, hi]; None when the target is not bracketed."""
    a, b = f(lo) - target, f(hi) - target
    if a * b > 0: return None
    for _ in range(n):
        mid = (lo + hi) / 2; m = f(mid) - target
        if a * m <= 0: hi = mid
        else: lo, a = mid, m
    return (lo + hi) / 2

if __name__ == "__main__":
    bj.selfcheck(); check_split()
    for subj, z in zip(SUBJ, (12.23, 11.87, 11.44)): assert abs(950 / perfect(subj) - z) < 0.02, (subj, 950 / perfect(subj))

    print("## 1. revised design bands: centre stack (worst to best), tok/s")
    for name, kc, kl, kh, p in DESIGNS_R2:
        row = []
        for subj in SUBJ:
            row.append(f"{stack(subj, kc):.2f} ({stack(subj, kh, kap=0.58):.2f}-{stack(subj, kl, kap=0.38):.2f}) P_G {pg(subj, kc):.2f}")
        print(f"  {name}: K {kc} ({kl}-{kh}) P {p} | " + " | ".join(row))
    print("\n## stack and P_G against in-situ K")
    for k in (0, 33, 50, 100, 150, 200, 250, 300, 350, 400, 452, 500, 560):
        print(f"  K {k:3}: " + "  ".join(f"{pg(s, k):.2f} / {stack(s, k):.2f}" for s in SUBJ))
    print("  launcher on a worker: K_eff = K + (1 - kappa) / kappa * Lh = K + 0.92 * Lh")

    print("\n## 2. R75 cells, Monte Carlo (median, 10 %, 90 %)")
    out, W = mc()
    for k, v in out.items():
        print(f"  {k}: " + (f"{v[0]:.0f} ({v[1]:.0f} to {v[2]:.0f})" if isinstance(v, tuple) else f"{v:.2f}"))
    print("  hypothesis shares: " + ", ".join(f"{h} {sum(w['hyp'] == h for w in W) / len(W):.2f}" for h in "kcg"))

    print("\n## x4c stacked cold rates (GB/s): bytes / (dev + gap); gap 311 (R74 alone state) and 125 (R74 tog state)")
    for lab, rows in (("gate||up 22016x4096", 22016), ("q||k||v 12288x4096", 12288), ("11008x4096 (R74 21.01)", 11008)):
        b = rows * 4096 * 0.5625 / 1e9; dev = (DEV0 + DEVROW * rows) / 1e6
        print(f"  {lab}: {b / (dev + 311e-6):.1f} / {b / (dev + 125e-6):.1f}  dev {dev * 1e6:.0f} us")

    print("\n## x4s fused [gate;up] 22016 x 4096, A = 1.896 ms: ratio t_split / t_cpu_alone by share (join slack not added)")
    for lab, K, Lh in (("feeder 0-3 + clFinish (R74 anchor)", 500, 0), ("cpu 4 + flush + spin, K 300", 300, 65), ("K 200", 200, 65), ("K 120", 120, 50), ("K 30", 30, 0)):
        cells = []
        for f in (0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50):
            tc, tg, t = split(1.896, 22016, f, K, Lh); cells.append(f"{t / 1.896:.3f}{'g' if tg > tc else 'c'}")
        print(f"  {lab:36}: " + "  ".join(cells))
    print("  wait_ms = max(0, t_gpu - t_cpu) + slack; gpu_ms = K + 1.10 * dev")

    print("\n## 3. perfect lane (K = 0): what else must move for 12.00 (79.17 ms), one lever at a time, the rest at centre")
    print("  centre: " + " / ".join(f"{perfect(s):.2f} ms = {950 / perfect(s):.2f}" for s in SUBJ))
    levers = [("lane factor x (0.621 -> 0.621 x)", lambda s: (lambda x: perfect(s, lane_scale=x)), 0.80, 1.0),
              ("head scale hs (0.5)", lambda s: (lambda x: perfect(s, hs=x)), 0.0, 0.5),
              ("c5 (0.93)", lambda s: (lambda x: perfect(s, c5=x)), 0.70, 0.93),
              ("O ms (5.2)", lambda s: (lambda x: perfect(s, O=x)), 0.0, 5.2),
              ("head top-1 a1", lambda s: (lambda x: perfect(s, a1=x)), bt.SUBJ[SUBJ[0]]["a1"] * 0 + 0.30, 0.95),
              ("C GB/s (40)", lambda s: (lambda x: perfect(s, C=x)), 40.0, 80.0),
              ("E, cost of the 2nd position (0.45)", lambda s: (lambda x: perfect(s, e=x)), 0.0, 0.45)]
    for lab, mk, lo, hi in levers:
        print(f"  {lab}: " + " / ".join((lambda v: "none in range" if v is None else f"{v:.3f}")(solve(mk(s), lo, hi)) for s in SUBJ))
    print("  sensitivities at the centre, ms per token: " )
    for lab, kw in (("lane x 0.95 (0.590)", dict(lane_scale=0.95)), ("lane x 0.92 (0.571)", dict(lane_scale=0.92)), ("lane at the R72 read ceiling 25.64 / 46.5 = 0.551", dict(lane_scale=0.551 / 0.6207)),
                    ("hs 0.3", dict(hs=0.3)), ("head on the lane too (hs x 0.62)", dict(head_gpu=0.62)), ("c5 0.90", dict(c5=0.90)), ("O 4.0", dict(O=4.0)), ("O 3.0", dict(O=3.0)),
                    ("C 50", dict(C=50.0)), ("W 3 at the centre", dict(W=3)), ("W 3 at C 50", dict(W=3, C=50.0)), ("W 1 (no head)", dict(W=1))):
        print(f"    {lab:50}: " + " / ".join(f"{perfect(s, **kw) - perfect(s):+.2f} -> {950 / perfect(s, **kw):.2f}" for s in SUBJ))
    print("  combos:")
    for lab, kw in (("lane 0.590 + O 4.0", dict(lane_scale=0.95, O=4.0)), ("lane 0.590 + hs 0.3", dict(lane_scale=0.95, hs=0.3)),
                    ("hs 0.3 + O 4.0", dict(hs=0.3, O=4.0)), ("lane 0.590 + hs 0.3 + O 4.0", dict(lane_scale=0.95, hs=0.3, O=4.0)),
                    ("lane 0.571 + head on the lane", dict(lane_scale=0.92, head_gpu=0.62))):
        print(f"    {lab:50}: " + " / ".join(f"{950 / perfect(s, **kw):.2f}" for s in SUBJ))
    print("  the same combos at K = 100 us are lower by about " + " / ".join(f"{stack(s, 0) - stack(s, 100):.2f}" for s in SUBJ) + " tok/s; at K = 30 by " + " / ".join(f"{stack(s, 0) - stack(s, 30):.2f}" for s in SUBJ))

    # ---- 4. width 2 beside the lane: the CPU's second position hides under its smaller bus share; the GPU's cost cg is unmeasured
    CA, CB, GB = 25.64, 19.77, 21.53                                 # x3 togsum (device)
    def rate_cpu(W, both, C=40.0):                                   # memo-10's t(c) with the bus share fitted to the tog rates at W 1
        inv = lambda r: 1 / r - 0.25 / C                             # 1/R from the measured rate (bus-bound at W 1)
        Rb = 1 / inv(CB if both else CA); x = bt.c(W) / C
        return 1 / (max(1 / Rb, x) + 0.25 * min(1 / Rb, x))
    def perfect2(subj, cg=1.45, gscale=1.0, K=0.0, C=40.0, **kw):
        """W 2 centre stack, zero dead time unless K: body term = body * c5 * 25.64 / (CPU rate beside the GPU at W 2 + GPU rate / cg)."""
        s = bt.SUBJ[subj]; c5 = kw.get("c5", 0.93); hs = kw.get("hs", 0.5); O = kw.get("O", 5.2)
        best = None
        for sh in ("topk", "chain", "two", "greedy"):
            a, dep, _ = bt.tree(s["a1"], 1, sh); t0, tau = bt.ms(s, 2, a, dep, C=C, g=0.0, disp=0.0, hs=hs, c5=c5, O=O)
            F = CA / (rate_cpu(2, True, C) + GB * gscale / cg)
            t = t0 - s["body"] * bt.kB(2, C) * c5 / tau + s["body"] * c5 * F / tau
            best = t if best is None else min(best, t)
        return best + (950 / stack(subj, K) - 950 / stack(subj, 0) if K else 0.0)
    assert abs(rate_cpu(1, False) - CA) < 1e-6 and abs(rate_cpu(1, True) - CB) < 1e-6
    print("\n## 4. width 2 beside the lane (K = 0): CPU W 2 rate beside the GPU %.2f GB/s (x %.3f; alone x %.3f, memo-10 kB 1.12)" % (rate_cpu(2, True), CB / rate_cpu(2, True), CA / rate_cpu(2, False)))
    print("  memo-10 lane at W 2 as used so far: body x %.4f; this model at cg 1.45: body x %.4f" % (bj.lane(2) * bt.kB(2, 40), CA / (rate_cpu(2, True) + GB / 1.45)))
    for cg in (1.0, 1.1, 1.2, 1.45, 1.7, 2.0):
        print(f"  cg {cg:4}: " + " / ".join(f"{perfect2(s, cg):.2f} ms = {950 / perfect2(s, cg):.2f}" for s in SUBJ))
    print("  cg needed for 12.00 at K = 0: " + " / ".join((lambda v: "none" if v is None else f"{v:.2f}")(solve(lambda x, s=s: perfect2(s, x), 1.0, 3.0)) for s in SUBJ))
    for K in (30, 100, 200, 300):
        print(f"  cg needed at K = {K}: " + " / ".join((lambda v: "none" if v is None else f"{v:.2f}")(solve(lambda x, s=s: perfect2(s, x, K=K), 1.0, 3.0)) for s in SUBJ))
    print("  cg 1.2 at K 0 / 100 / 300: " + " ; ".join(" / ".join(f"{950 / perfect2(s, 1.2, K=K):.2f}" for s in SUBJ) for K in (0, 100, 300)))
    print("  cg 1.45 at K 0 / 100 / 300: " + " ; ".join(" / ".join(f"{950 / perfect2(s, 1.45, K=K):.2f}" for s in SUBJ) for K in (0, 100, 300)))
    print("  cg 2.0 (two launches) at K 0 / 100 / 300: " + " ; ".join(" / ".join(f"{950 / perfect2(s, 2.0, K=K):.2f}" for s in SUBJ) for K in (0, 100, 300)))
    print("  cg 1.2 + hs 0.3 + O 4.0 at K 0 / 100 / 300: " + " ; ".join(" / ".join(f"{950 / perfect2(s, 1.2, K=K, hs=0.3, O=4.0):.2f}" for s in SUBJ) for K in (0, 100, 300)))
    print("  C 35 (L4 caps): cg 1.45 " + " / ".join(f"{950 / perfect2(s, 1.45, C=35.0):.2f}" for s in SUBJ) + "; cg 1.2 " + " / ".join(f"{950 / perfect2(s, 1.2, C=35.0):.2f}" for s in SUBJ))
    print("  r2 stack against K (cg 1.45 | cg 1.2 | cg 2.0):")
    for K in (0, 30, 100, 150, 200, 220, 270, 300, 360, 500):
        print(f"    K {K:3}: " + " | ".join(" / ".join(f"{950 / perfect2(s, cg, K=K):.2f}" for s in SUBJ) for cg in (1.45, 1.2, 2.0)))
    print("  K at which r2 reaches 12.00 (cg 1.45 | cg 1.2): " + " | ".join(" / ".join((lambda v: "none" if v is None else f"{v:.0f}")(solve(lambda k, s=s: perfect2(s, cg, K=max(k, 1e-9)), 0.0, 560.0)) for s in SUBJ) for cg in (1.45, 1.2)))
    for cg in (1.08, 1.1, 1.3, 1.7):
        print(f"  cg {cg}: K for 12.00 " + " / ".join((lambda v: "none" if v is None else f"{v:.0f}")(solve(lambda k, s=s: perfect2(s, cg, K=max(k, 1e-9)), 0.0, 560.0)) for s in SUBJ)
              + "; stack at K 0 / 100 / 220 / 300 / 500: " + " ; ".join(" / ".join(f"{950 / perfect2(s, cg, K=K):.2f}" for s in SUBJ) for K in (0, 100, 220, 300, 500)))
    # GPU compute slack from device rates: gemv dev 28.32 GB/s cold against GPU read-only 31.14 (R72); beside the CPU dev 25.7 against read 25.81
    comp = 0.25 / (1 / 28.32 - 1 / 31.14); rb = 1 / (1 / 25.7 - 0.25 / comp)
    print(f"  two-term model on the GPU: compute-alone rate {comp:.0f} GB/s; cg at 1.9 x compute per byte = {(1 / rb + 0.25 * 1.9 / comp) * 25.7:.3f}; compute-bound worst case {25.7 * 1.9 / 28.32:.2f}")
