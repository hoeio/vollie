#!/usr/bin/env python3
"""memo-12 round 30 (Bonnie): the bus-against-compute rule ("two-term reader") bracketed by a second functional form
through the same two phone points (memo-09 section 10.3: read 29.9 GB/s; stock 26.0 GB/s at full clocks with compute
taken as 50, and 24.7 GB/s at the L4 caps with compute taken as 35), and r29's P2 / P1 table under each. *estimated*.
  form A (as committed)   t = max(1/R, x/C) + 0.25 * min(1/R, x/C)
  form B (p-norm)         t = ((1/R)^p + (x/C)^p)^(1/p);   p 2.10 passes the first point, 2.65 the second
Both are anchored on the same one-column device rates (CPU alone 25.64, CPU beside the GPU as form A gives it at W 1).
Run: python3 -I -B bonnie_reader_r30.py"""
import importlib.util, os
_d = os.path.dirname(os.path.abspath(__file__))
_s = importlib.util.spec_from_file_location("r29", os.path.join(_d, "bonnie_ratio_r29.py")); r29 = importlib.util.module_from_spec(_s); _s.loader.exec_module(r29)
t2, bt, r5, r13 = r29.t2, r29.bt, r29.r5, r29.r13
A = (t2.cpu_rate, t2.bus_of, t2.BUS, r5.BUS)

def form_b(p):
    rate = lambda W, Rb, C=40.0: 1 / ((1 / Rb) ** p + (bt.c(W) / C) ** p) ** (1 / p)
    bus = lambda r, C=40.0: 1 / ((1 / r) ** p - (1 / C) ** p) ** (1 / p)
    return rate, bus
def use(p, C):
    """p None = form A. The bus sum is re-anchored so that the CPU beside the GPU at W 1 is what form A gives."""
    t2.cpu_rate, t2.bus_of, t2.BUS, r5.BUS = A
    if p is None: return
    cb1 = A[0](1, A[2] - t2.GD, C); rate, bus = form_b(p)
    t2.cpu_rate, t2.bus_of = rate, bus; t2.BUS = r5.BUS = t2.GD + bus(cb1, C)

def points(p):                                  # the two phone points, stock at W 1 (x = 1)
    f = (lambda C: 1 / (1 / 29.9 + 0.25 / C)) if p is None else (lambda C: 1 / ((1 / 29.9) ** p + (1 / C) ** p) ** (1 / p))
    return f(50.0), f(35.0)
assert abs(points(None)[0] - 26.0) < 0.05 and abs(points(None)[1] - 24.7) < 0.15
assert abs(points(2.10)[0] - 26.0) < 0.1 and abs(points(2.65)[1] - 24.7) < 0.1

if __name__ == "__main__":
    k0 = bt.c(2); forms = (("A", None), ("B p 2.10", 2.10), ("B p 2.35", 2.35), ("B p 2.65", 2.65))
    for lab, p in forms: print(f"form {lab}: the two phone points (26.0, 24.7 GB/s) -> {points(p)[0]:.1f}, {points(p)[1]:.1f}")
    rows = [("as priced: W kernel, cg %.2f" % r13.CG, k0, r13.CG, True), ("W kernel, two GPU launches", k0, 2.0, True), ("one-row kernel twice, batched GPU kernel", 2.0, r13.CG, True),
            ("nothing new: one-row twice, two launches", 2.0, 2.0, False), ("stock batch 2 (tile 2.60), two launches", 2.6, 2.0, False)]
    for C in (40.0, 36.9, 46.0):
        print(f"\n## lane, K 300, C {C}: P2 / P1, Llama-2, pair / linear, per form  (Yi about 0.02 lower)")
        for lab, ka, cg, sh in rows:
            cells = []
            for fl, p in forms:
                use(p, C); cells.append(f"{fl}: {r29.ratio('Llama-2', ka, cg, C, 300, 'pair', sh):.2f} / {r29.ratio('Llama-2', ka, cg, C, 300, 'linear', sh):.2f}")
            print(f"  {lab}: " + " | ".join(cells))
        print(f"## no lane, C {C}: P2 / P1, Llama-2")
        for lab, ka, sh in (("W kernel", k0, True), ("one-row kernel twice", 2.0, False), ("stock batch 2 (tile)", 2.6, False)):
            cells = []
            for fl, p in forms: use(p, C); cells.append(f"{fl}: {r29.ratio('Llama-2', ka, 1.0, C, 0, 'pair', sh, lane=False):.2f}")
            print(f"  {lab}: " + " | ".join(cells))
    use(None, 40.0)
