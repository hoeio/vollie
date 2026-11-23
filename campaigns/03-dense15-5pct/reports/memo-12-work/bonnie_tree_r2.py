#!/usr/bin/env python3
"""Stack model with the width-2 lane correction (memo-12 r2/r3). Every output is *estimated*.
memo-10's bonnie_tree.py and memo-12's bonnie_joins.py are loaded by path and stay untouched.

Three named variants of "ms per committed token with a CPU + GPU lane":
  old     memo-10 form as committed: body * lane(W) * kB(W) (what R74-result section 4 used). Kept for the self-check.
  bonnie  CPU keeps its W 1 bus share beside the GPU (22.6 GB/s); GPU rate = g1 / cg.
  pair    Ebony's shared bus (ebony-joins-r2 A.6): bus sum 46.1 GB/s, CPU bus share = 46.1 - GPU rate.
Common stage model (four stages per layer; rates in GB/s of weight bytes per pass; K = GPU dead time per split stage):
  the CPU runs at its alone rate ca(W) while the GPU is dead, at cb(W) while the GPU kernel runs:
     t_split = K + (B - ca * K) / (cb + g) + slack,   t_alone = B / ca,   a stage is split only if that is faster
  slack = J0 + J1 * K per split stage (Bonnie: J0 0, J1 0.107; Ebony: J0 60 us, J1 0).
Inputs (*device*): x3 togsum 25.64 / 19.77 / 21.53 GB/s (R74); GPU device rate beside the CPU 24.6 (derived from gpu_busy).
Run: python3 -I bonnie_tree_r2.py
"""
import importlib.util, os
_d = os.path.dirname(os.path.abspath(__file__))
def _load(n, p):
    s = importlib.util.spec_from_file_location(n, p); m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
bj = _load("bonnie_joins", os.path.join(_d, "bonnie_joins.py")); bt = bj.bt
SUBJ = list(bt.SUBJ)
CA, CB, GW, GD, BUS = 25.64, 19.77, 21.53, 24.6, 46.1     # CPU alone, CPU beside GPU, GPU wall, GPU device time (both beside the CPU), bus sum

def cpu_rate(W, Rb, C=40.0):
    """memo-10's two-term pass model: time per byte = max(1/Rb, c(W)/C) + 0.25 * min(...)."""
    x = bt.c(W) / C
    return 1 / (max(1 / Rb, x) + 0.25 * min(1 / Rb, x))
def bus_of(rate, C=40.0):                                  # bus share that gives `rate` at W 1 (bus-bound)
    return 1 / (1 / rate - 0.25 / C)

def rates(W, variant, cg=1.45, C=40.0, gdev=False):
    """(ca, cb, g): CPU alone, CPU beside the GPU, GPU. cg = GPU time per byte at width W / at width 1 (1 at W 1)."""
    cgw = 1.0 if W == 1 else cg
    assert C >= 31, "valid only while the CPU is bus-bound at width 1 (C above 30.5 GB/s): the measured body is taken as fixed"
    ca = cpu_rate(W, bus_of(CA), C)
    if variant == "bonnie":
        g = (GD if gdev else GW) / cgw
        cb = cpu_rate(W, bus_of(CB if not gdev else 19.3), C)   # 19.3: Ebony's centre for the CPU beside a gap-free GPU
    elif variant == "pair":
        g = GD / cgw; cb = cpu_rate(W, BUS - g, C)
    else: raise ValueError(variant)
    return ca, cb, g

def body(subj, W, K_us, variant, cg=1.45, C=40.0, c5=0.93, J0=0.0, J1=0.107, gdev=False):
    """(body ms per pass with the lane, split stages per layer, GPU share of the body bytes)."""
    s = bt.SUBJ[subj]; ca1 = cpu_rate(1, bus_of(CA), C); ca, cb, g = rates(W, variant, cg, C, gdev)
    tot = sum(w for _, w in bj.STAGES[subj]); K = K_us / 1e3; t = n = gb = 0.0
    for _, w in bj.STAGES[subj]:
        B = s["body"] * c5 * w / tot / bj.NL[subj] * ca1          # stage "bytes" in ms * GB/s
        alone = B / ca
        sp = K + (B - ca * K) / (cb + g) + J0 / 1e3 + J1 * K if B > ca * K else 1e9
        if sp < alone: t += sp; n += 1; gb += g * (sp - K - J0 / 1e3 - J1 * K) / B * w
        else: t += alone
    return t * bj.NL[subj], int(n), gb / tot

def ms(subj, W=2, K=0.0, variant="bonnie", hs=0.5, O=5.2, a1=None, **kw):
    """ms per committed token; W in (1, 2). Head, guess, xpos and tau as in bonnie_tree.ms."""
    s = bt.SUBJ[subj]; a = s["a1"] if a1 is None else a1
    b = body(subj, W, K, variant, **kw)[0]
    if W == 1: return b + O + s["head"] * hs
    tau = 1 + s["d"] * a
    return (b + O + s["guess"] + s["xpos"]) / tau + s["head"] * hs
def toks(*a, **k): return 950 / ms(*a, **k)
def old(subj, K_us):                                           # the committed form, through bonnie_joins
    return bj.price(subj, K_us)["stack"]

def solve(f, lo, hi, target, n=60):
    a = f(lo) - target
    if a * (f(hi) - target) > 0: return None
    for _ in range(n):
        m = (lo + hi) / 2
        if (f(m) - target) * a > 0: lo = m
        else: hi = m
    return (lo + hi) / 2

def selfcheck():
    bj.selfcheck()                                             # R74-result section 4: 12.23 / 11.87 / 11.44, 9.55 / 9.50 / 9.26
    for subj, z0 in zip(SUBJ, (12.23, 11.87, 11.44)): assert abs(old(subj, 0) - z0) < 0.02
    assert abs(rates(1, "bonnie")[1] - CB) < 1e-6 and abs(rates(1, "bonnie")[0] - CA) < 1e-6
    for subj, z in zip(SUBJ, (12.60, 12.24, 11.77)):           # bonnie-joins-r2 section 3.1 (cg 1.45, wall rate, K 0)
        assert abs(toks(subj, 2, 0, "bonnie") - z) < 0.02, (subj, toks(subj, 2, 0, "bonnie"))
    for subj, z in zip(SUBJ, (13.58, 13.24, 12.66)): assert abs(toks(subj, 2, 0, "bonnie", cg=1.2) - z) < 0.02
    ca, cb, g = rates(2, "pair", cg=GD / 19.5)                 # ebony-joins-r2 A.6: W 2 CPU 21.4, GPU 19.5, sum 40.9
    assert abs(cb - 21.4) < 0.1 and abs(cb + g - 40.9) < 0.15, (cb, g)
    assert abs(sum(rates(1, "pair")[1:]) - 43.5) < 0.1
    for subj, z in zip(SUBJ, (14.89, 14.57, 13.84)):           # her pair model at K 0, C 40, GC 28.3
        assert abs(toks(subj, 2, 0, "pair", cg=GD / 19.5) - z) < 0.06, (subj, toks(subj, 2, 0, "pair", cg=GD / 19.5))

if __name__ == "__main__":
    selfcheck()
    CGP = GD / 19.5
    print("body factor at W 2 (K 0): old %.3f | bonnie cg 1.45 %.3f, cg 1.2 %.3f | bonnie device-rate cg 1.26 %.3f | pair cg 1.26 %.3f" % (
        bj.lane(2) * bt.kB(2, 40), CA / sum(rates(2, "bonnie")[1:]), CA / sum(rates(2, "bonnie", 1.2)[1:]), CA / sum(rates(2, "bonnie", CGP, gdev=True)[1:]), CA / sum(rates(2, "pair", CGP)[1:])))
    cols = [("old", lambda s, K: old(s, K)), ("bonnie cg 1.45", lambda s, K: toks(s, 2, K, "bonnie")), ("bonnie cg 1.2", lambda s, K: toks(s, 2, K, "bonnie", cg=1.2)),
            ("pair cg 1.26", lambda s, K: toks(s, 2, K, "pair", cg=CGP)), ("pair cg 1.26, slack 60", lambda s, K: toks(s, 2, K, "pair", cg=CGP, J0=60, J1=0)),
            ("pair cg 1.26, C 35", lambda s, K: toks(s, 2, K, "pair", cg=CGP, C=35.0)), ("W 1 bonnie", lambda s, K: toks(s, 1, K, "bonnie")), ("W 1 pair", lambda s, K: toks(s, 1, K, "pair"))]
    for name, f in cols:
        print(f"\n## {name}: stack tok/s against K")
        for K in (0, 30, 100, 150, 200, 300, 452, 500):
            print(f"  K {K:3}: " + " / ".join(f"{f(s, K):.2f}" for s in SUBJ))
        print("  K for 12.00: " + " / ".join((lambda v: "none" if v is None else f"{v:.0f}")(solve(lambda k, s=s: f(s, k), 0, 600, 12.0)) for s in SUBJ))
