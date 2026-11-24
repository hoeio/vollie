#!/usr/bin/env python3
"""memo-12 round 32 (Bonnie): the coupling term of the bus-against-compute rule refitted on every cold ggml Q4_0
(KleidiAI) cell of the committed probe reports, each with its own clocks and thread set. Inputs are *device*; the fit
and everything derived from it are *estimated*.
  time per byte = max(1/R, 1/C) + alpha * min(1/R, 1/C)
  C = h * (sum of A78 MHz + x * X1 MHz) / 2227     h = hot GB/s of one A78 at 2,227 MHz (R72 L3: 11.54 for the probe kernel)
  R = 29.9 GB/s for three or four threads (R72 block a), 21.4 for one A78 (R69, R69b: 22.15, 20.59)
Cells: R72 L4 arm=ggml_stock (9), R72 L2 kern=ggml_q4_0 (3), R74 x3 tog arm=cpu (2), R72 L5 q4_0 cold on cpu 4 (3).
R69 has no ggml cell. A thread is taken at the top of its policy's mhz_range; the X1 thread is taken as on-core.
Run: python3 -I -B bonnie_coupling_r32.py <repo root>"""
import sys, os, json, re
root = sys.argv[1]; dev = os.path.join(root, "campaigns/03-dense15-5pct/reports/device")
g = lambda l, k: (re.search(r"\b" + k + r"=(\S+)", l) or [None, None])[1]
def lines(f):
    for b in json.load(open(os.path.join(dev, f)))["blocks"]:
        for l in b.get("lines") or []:
            if isinstance(l, str): yield l
def clocks(l):                               # (A78 MHz, X1 MHz) at the top of mhz_range
    a = g(l, "mhz_range").split("..")[1].split(",")[0].split("/"); return int(a[1]), int(a[2])
cells = []
for l in lines("r72-c3probe-r33-2026-11-22T1759.json"):
    if "arm=ggml_stock" in l or " l2 kern=ggml_q4_0" in l or (" l5 " in l and "type=q4_0" in l and "regime=cold" in l and g(l, "set") == "4" and g(l, "cpu0_ok") == "1"):
        tag = "R72 L4 " + g(l, "subj") if "arm=ggml_stock" in l else ("R72 L2 arena" if " l2 " in l else "R72 L5 k_in " + g(l, "k_in") + " pass " + g(l, "pass"))
        cells.append((tag, g(l, "set"), clocks(l), float(g(l, "gbs"))))
for l in lines("r74-c3probe-r34-2026-11-23T1743.json"):
    if "x3 tog " in l and "arm=cpu" in l: cells.append(("R74 tog pos " + g(l, "pos"), g(l, "set"), clocks(l), float(g(l, "cpu_gbs"))))
assert len(cells) == 17, len(cells)

def C_of(cset, mhz, h, x):
    cpus = [int(c) for c in cset.split(",")]; return h * (sum(mhz[0] for c in cpus if c != 7) + x * mhz[1] * (7 in cpus)) / 2227
def rate(cset, mhz, h, x, al):
    R = 21.4 if "," not in cset else 29.9; C = C_of(cset, mhz, h, x); return 1 / (max(1 / R, 1 / C) + al * min(1 / R, 1 / C))
def err(h, x, al, sel=cells): return (sum((rate(s, m, h, x, al) / v - 1) ** 2 for _, s, m, v in sel) / len(sel)) ** 0.5
A = [i / 200 for i in range(0, 101)]; H = [9 + i * 0.1 for i in range(0, 61)]; X = [0.8 + i * 0.05 for i in range(0, 25)]
fits = {"alpha only (h 11.54, x 1.44)": min(((err(11.54, 1.44, a), 11.54, 1.44, a) for a in A)),
        "alpha and h (x 1.44)": min(((err(h, 1.44, a), h, 1.44, a) for a in A for h in H)),
        "alpha, h and x": min(((err(h, x, a), h, x, a) for a in A for h in H for x in X)),
        "committed: alpha 0.25, h 11.54, x 1.44": (err(11.54, 1.44, 0.25), 11.54, 1.44, 0.25)}
assert abs(rate("4,5,6,7", (2419, 2841), 11.54, 1.44, 0.25) - 1 / (1 / 29.9 + 0.25 / C_of("4,5,6,7", (2419, 2841), 11.54, 1.44))) < 1e-9

if __name__ == "__main__":
    for lab, (e, h, x, a) in fits.items(): print(f"{lab}: rms relative error {100 * e:.1f} %  ->  alpha {a:.3f}, h {h:.2f}, x {x:.2f}")
    e, h, x, a = fits["alpha and h (x 1.44)"]
    print("\ncell | set | A78 / X1 MHz | measured GB/s | fit (alpha and h) | committed | C in the fit")
    for tag, s, m, v in cells: print(f"{tag} | {s} | {m[0]} / {m[1]} | {v:.2f} | {rate(s, m, h, x, a):.2f} | {rate(s, m, 11.54, 1.44, 0.25):.2f} | {C_of(s, m, h, x):.1f}")
    print("\nleave one group out (alpha and h refitted without the group, error on the group):")
    for grp in ("R72 L4", "R72 L2", "R74", "R72 L5"):
        tr = [c for c in cells if not c[0].startswith(grp)]; te = [c for c in cells if c[0].startswith(grp)]
        _, hh, aa = min(((err(q, 1.44, w, tr), q, w) for w in A for q in H)); print(f"  without {grp}: alpha {aa:.3f}, h {hh:.2f}; error on it {100 * err(hh, 1.44, aa, te):.1f} %")
    print("\nbody (bytes = R72 L4 four-thread body_ms x GB/s), four threads on their cores, ms:")
    byt = {c[0][7:]: c[3] for c in cells if c[0].startswith("R72 L4") and c[1] == "4,5,6,7"}; ms0 = {"yi15-6b": 120.55, "llama2-7b": 140.61, "deepseek7b": 130.77}
    for lab, (hh, xx, aa) in (("fit alpha and h", (h, x, a)), ("committed", (11.54, 1.44, 0.25))):
        for st, m in (("cap 1,555", (1555, 1555)), ("cap 2,035", (2035, 2035)), ("full 2,419 / 2,841", (2419, 2841)), ("cap 1,555, three A78 only", None)):
            r = rate("4,5,6,7", m, hh, xx, aa) if m else rate("4,5,6", (1555, 844), hh, xx, aa)
            print(f"  {lab}, {st}: rate {r:.1f} GB/s, C {C_of('4,5,6,7', m, hh, xx) if m else C_of('4,5,6', (1555, 844), hh, xx):.1f}; body " + " / ".join(f"{ms0[k] * byt[k] / r:.1f}" for k in ms0))
    # the width-2 ratio of r29 / r31 with the fitted coupling (form A with alpha as a parameter), loaded from the r30 script
    import importlib.util
    _s = importlib.util.spec_from_file_location("r30", os.path.join(os.path.dirname(os.path.abspath(__file__)), "bonnie_reader_r30.py")); m = importlib.util.module_from_spec(_s); _s.loader.exec_module(m)
    r29, t2, bt, r13 = m.r29, m.t2, m.bt, m.r13; k0 = bt.c(2); A0 = m.A
    def use_alpha(al, C):
        t2.cpu_rate, t2.bus_of, t2.BUS, m.r5.BUS = A0
        cb1 = A0[0](1, A0[2] - t2.GD, C)
        t2.cpu_rate = lambda W, Rb, C=40.0: 1 / (max(1 / Rb, bt.c(W) / C) + al * min(1 / Rb, bt.c(W) / C))
        t2.bus_of = lambda r, C=40.0: 1 / (1 / r - al / C)
        t2.BUS = m.r5.BUS = t2.GD + t2.bus_of(cb1, C)
    print("\nP2 / P1, Llama-2, K 300 (pair / linear): lane as priced | lane, W kernel, two launches | no lane, W kernel | W kernel alone")
    for C in (32.2, 35.0, 40.0, 46.0, 57.6):
        for al in (0.25, a, 0.14):
            use_alpha(al, C); k = t2.cpu_rate(1, t2.bus_of(t2.CA, C), C) / t2.cpu_rate(2, t2.bus_of(t2.CA, C), C)
            print(f"  C {C}, alpha {al:.2f}: {r29.ratio('Llama-2', k0, r13.CG, C, 300, 'pair'):.2f} / {r29.ratio('Llama-2', k0, r13.CG, C, 300, 'linear'):.2f} | "
                  f"{r29.ratio('Llama-2', k0, 2.0, C, 300, 'pair'):.2f} / {r29.ratio('Llama-2', k0, 2.0, C, 300, 'linear'):.2f} | {r29.ratio('Llama-2', k0, 1.0, C, 0, 'pair', lane=False):.2f} | {k:.2f}")
    t2.cpu_rate, t2.bus_of, t2.BUS, m.r5.BUS = A0
