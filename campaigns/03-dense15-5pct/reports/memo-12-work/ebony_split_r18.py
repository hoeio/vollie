#!/usr/bin/env python3
"""memo-12 round 18 (Ebony): R75 x4s arm by arm, and the lane re-priced with the dead time and coupling the phone paid.
Inputs are *device* lines of the R75 session-2 JSON; every derived value is arithmetic on them; every tok/s is *estimated*.
Loads bonnie_floor_r14.py (and through it the committed stage model) unchanged; only t2.BUS and t2.GD are set per row.
Run: python3 -I -B ebony_split_r18.py <repo root>"""
import sys, os, json, re, importlib.util
root = sys.argv[1]; here = os.path.dirname(os.path.abspath(__file__))
J = os.path.join(root, "campaigns/03-dense15-5pct/reports/device/r75a1-c3probe-r36-2026-11-24T1345.json")
g = lambda l, k: (re.search(r"\b" + k + r"=(\S+)", l) or [None, None])[1]
arms = []
for b in json.load(open(J))["blocks"]:
    if b["block"] != "x4s": continue
    for l in b["lines"]:
        if "x4 split pos=" in l:
            a = {k: float(g(l, k)) for k in ("pos", "share", "gpu_rows", "cpu_rows", "ms_per_mat", "cpu_ms", "wait_ms", "gpu_ms", "gpu_dev_ms", "enq_us")}
            a["mode"] = g(l, "mode"); a["cap"] = g(l, "max_mhz").split(",")[1].split(":")[1].split("/")[0]; a["gmax"] = g(l, "gpu_max_mhz")
            a["thr"] = g(l, "thr"); arms.append(a)
assert len(arms) == 22 and arms[0]["mode"] == "cpu" and arms[21]["mode"] == "cpu"
A0, A1 = arms[0]["ms_per_mat"], arms[21]["ms_per_mat"]; TM = (A0 + A1) / 2      # CPU alone: first arm, last arm, the probe's mean
assert abs(A0 - 1.776) < 1e-9 and abs(A1 - 1.895) < 1e-9
MB = 22016 * 4096 * 0.5625 / 1e6                                              # weight MB of the fused matrix
def base(a): return A0 if a["pos"] <= 10 else A1                              # nearest CPU-alone arm
lead = [a for a in arms if a["mode"] == "lead"]
for a in lead:
    a["T"] = a["enq_us"] / 1e3 + a["cpu_ms"] + a["wait_ms"]                   # the lead thread's cycle: launch + flush, CPU rows, spin
    a["exp"] = (1 - a["share"]) * base(a); a["s10"] = a["cpu_ms"] / a["exp"]
    a["ov"] = min(a["gpu_dev_ms"], a["cpu_ms"]); a["exc"] = a["cpu_ms"] - a["exp"]
    a["nk"] = a["T"] - a["gpu_dev_ms"]                                        # cycle time that is not GPU kernel time
    a["gdev"] = a["gpu_rows"] * 4096 * 0.5625 / 1e6 / a["gpu_dev_ms"]         # GPU device GB/s

def fit(pts):                                                                 # least squares y = a + b x
    n = len(pts); sx = sum(x for x, _ in pts); sy = sum(y for _, y in pts); sxx = sum(x * x for x, _ in pts); sxy = sum(x * y for x, y in pts)
    b = (n * sxy - sx * sy) / (n * sxx - sx * sx); return (sy - b * sx) / n, b
clean = [a for a in lead if a["pos"] <= 4]                                    # full clocks, GPU 840, before the first cap step
F0, sl = fit([(a["ov"], a["exc"]) for a in clean]); SIG = 1 / (1 - sl)
cap14 = [a for a in lead if 16 <= a["pos"] <= 20]
F0c, slc = fit([(a["ov"], a["exc"]) for a in cap14])

# ---- prices: the committed stage model, pair, with BUS set for a chosen CPU slowdown beside the GPU and GD for the GPU rate
s = importlib.util.spec_from_file_location("fl", os.path.join(here, "bonnie_floor_r14.py")); fl = importlib.util.module_from_spec(s); s.loader.exec_module(fl)
t2 = fl.t2; S = t2.SUBJ; BUS0, GD0 = t2.BUS, t2.GD
def setup(sig, rho=1.0):
    t2.GD = GD0 * rho; t2.BUS = (t2.GD + t2.bus_of(t2.CA / sig)) if sig else BUS0 - GD0 + t2.GD
    return t2.rates(1, "pair")
def price(form, K, J0, sig, rho=1.0):
    setup(sig, rho)
    try:
        if form == "w1":   f = lambda x: t2.toks(x, 1, K, "pair", c5=1.0, hs=fl.KAI[x] / fl.bt.SUBJ[x]["head"], O=8, J0=J0, J1=0)      # built pieces only
        elif form == "r7": f = lambda x: t2.toks(x, 2, K, "pair", cg=2.0, c5=0.93, hs=0.5, O=5.2, J0=J0, J1=0)                         # rung 7 form
        return [f(x) for x in S]
    finally: t2.BUS, t2.GD = BUS0, GD0
f3 = lambda v: " / ".join(f"{x:.2f}" for x in v)
SIG0 = (lambda r: r[0] / r[1])(setup(None)); t2.BUS, t2.GD = BUS0, GD0
STOCK = [950 / (fl.bt.SUBJ[x]["body"] + 8 + fl.KAI[x]) for x in S]
assert all(abs(a - b) < 0.01 for a, b in zip(STOCK, (6.90, 6.21, 6.22)))

if __name__ == "__main__":
    print(f"CPU alone: first arm {A0:.3f} ms, last arm {A1:.3f} ms (caps p4 {arms[21]['cap']}, GPU {arms[21]['gmax']}); probe mean {TM:.4f}; matrix {MB:.2f} MB")
    print("\npos share cap/gpu  T=enq+cpu+wait  ms_per_mat  T/base  cpu_ms  S10   excess_us overlap_ms exc/ov  nonkernel_us  gpu_dev_GB/s  enq_us  wait")
    for a in lead:
        print(f"{a['pos']:3.0f} {a['share']:.2f} {a['cap']:>4}/{a['gmax']:>3}  {a['T']:.3f}  {a['ms_per_mat']:.3f}  {a['ms_per_mat'] / base(a):.3f}  {a['cpu_ms']:.3f}  {a['s10']:.3f}  "
              f"{1e3 * a['exc']:6.0f}  {a['ov']:.3f}  {a['exc'] / a['ov']:.2f}  {1e3 * a['nk']:6.0f}  {a['gdev']:5.1f}  {a['enq_us']:.0f}  {a['wait_ms']:.3f}")
    fe = [a for a in arms if a["mode"] == "feeder"]
    print("feeder 0.25 twins: ms_per_mat " + " / ".join(f"{a['ms_per_mat']:.3f} (pos {a['pos']:.0f}, /base {a['ms_per_mat'] / base(a):.3f})" for a in fe))
    print(f"\nclean arms (pos 1-4): excess = {1e3 * F0:.0f} us + {sl:.3f} x overlap  ->  CPU slowdown while the kernel runs {SIG:.2f}, fixed {1e3 * F0:.0f} us per graph")
    print(f"cap-1420 arms (pos 16-20): excess = {1e3 * F0c:.0f} us + {slc:.3f} x overlap -> slowdown {1 / (1 - slc):.2f}, fixed {1e3 * F0c:.0f} us")
    for lab, fx in (("fixed term forced to the clean arms' value", F0),):
        v = [(a["exc"] - fx) / a["ov"] for a in cap14]; print(f"cap-1420 arms, {lab}: slowdown {1 / (1 - min(v)):.2f} to {1 / (1 - max(v)):.2f}")
    bad = [a for a in lead if 5 <= a["pos"] <= 9]
    print("bad stretch (pos 5-9): exc/overlap " + ", ".join(f"{a['exc'] / a['ov']:.2f}" for a in bad) + "; S10 " + ", ".join(f"{a['s10']:.2f}" for a in bad))
    assert 1.30 < SIG < 1.42 and abs(SIG0 - 1.353) < 0.005 and abs((lead[7]["ms_per_mat"] + lead[10]["ms_per_mat"]) / 2 / TM - 0.923) < 0.001
    crit = [a for a in lead if a["wait_ms"] > 0.1]
    print("GPU-critical arms (wait > 0.1 ms): non-kernel time per cycle " + ", ".join(f"pos {a['pos']:.0f}: {1e3 * a['nk']:.0f} us (enq {a['enq_us']:.0f}, QUEUED->START {1e3 * (a['gpu_ms'] - a['gpu_dev_ms']):.0f})" for a in crit))
    print("  END -> seen by the spin, lower bound (cpu + wait - gpu_ms): " + ", ".join(f"{1e3 * (a['cpu_ms'] + a['wait_ms'] - a['gpu_ms']):.0f}" for a in crit) + " us; upper bound adds enq")
    # the 0.923 cell
    p8, p13 = lead[7], lead[10]; assert p8["pos"] == 8 and p13["pos"] == 13
    print(f"\nshare 0.55: arm pos 8 {p8['ms_per_mat']:.3f} (first half), arm pos 13 {p13['ms_per_mat']:.3f}; mean / probe mean = {(p8['ms_per_mat'] + p13['ms_per_mat']) / 2 / TM:.3f}")
    print(f"  pos 13 alone: / last CPU arm {p13['ms_per_mat'] / A1:.3f}, / probe mean {p13['ms_per_mat'] / TM:.3f}, / first CPU arm {p13['ms_per_mat'] / A0:.3f}")
    sec = {a["share"]: a["ms_per_mat"] for a in lead if a["pos"] >= 12}; b2 = min(sec, key=sec.get)
    print(f"  second-half best: share {b2:.2f}, {sec[b2]:.3f} ms; first-half best (pos 1-4): {min(a['ms_per_mat'] for a in clean):.3f} ms = {min(a['ms_per_mat'] for a in clean) / A0:.3f} of the first CPU arm")
    # stage model on this cell: t = D + (B - ca K) / (cb + g)
    def cell(ca_ms, sig, gg, K, D): ca = MB / ca_ms; return (K + (MB - ca * K) / (ca / sig + gg) + D) / ca_ms
    print(f"\nstage model on the cell: committed (K 0.149, sigma 1.30, g 23.85, no slack) {cell(TM, 1.30, 23.85, 0.149, 0):.3f}")
    for lab, sig, gg, D in (("as probed, cool: dead 0.40 serial, sigma 1.40, g 25.8", 1.40, 25.8, 0.40), ("as probed, capped: dead 0.40, sigma 1.8, g 24.0", 1.8, 24.0, 0.40),
                            ("F, cool: dead 0.078, sigma 1.40, g 25.8", 1.40, 25.8, 0.078), ("F, capped: dead 0.078, sigma 1.8, g 24.0", 1.8, 24.0, 0.078),
                            ("F at 0.8 of the kernel rate, cool", 1.40, 25.8 * 0.8, 0.078)):
        print(f"  {lab}: {cell(TM, sig, gg, 0, D):.3f}")
    print(f"\ncommitted pair: CPU slowdown beside the GPU {SIG0:.3f}; stock {f3(STOCK)}")
    rows = [("committed: K 162.3 (CPU works during K), slack 60, sigma committed", 162.3, 60, None, 1.0),
            ("as probed, cool: dead 400 serial, sigma 1.40", 0, 400, 1.40, 1.0), ("as probed, cool, dead 340", 0, 340, 1.40, 1.0), ("as probed, cool, dead 570", 0, 570, 1.40, 1.0),
            ("as probed, capped: dead 400, sigma 1.8", 0, 400, 1.8, 1.0),
            ("G: launch inside the pool job + done flag: K 200 (CPU works), dead 127, sigma 1.40", 200, 127, 1.40, 1.0), ("G capped: sigma 1.8", 200, 127, 1.8, 1.0),
            ("F: dead 18 + slack 60, sigma 1.40, kernel rate 1.0", 0, 78, 1.40, 1.0), ("F, kernel rate 0.8", 0, 78, 1.40, 0.8), ("F, kernel rate 0.6", 0, 78, 1.40, 0.6),
            ("F capped: sigma 1.8, rate 1.0", 0, 78, 1.8, 1.0), ("F capped: sigma 1.8, rate 0.8", 0, 78, 1.8, 0.8),
            ("F per layer (launch 400 us over the layer's split stages ~ +100): dead 178, sigma 1.40", 0, 178, 1.40, 1.0)]
    for form, lab in (("w1", "width 1, O 8, measured head (built pieces only)"), ("r7", "rung-7 form: width 2 by two launches (cg 2.0), C5 0.93, head x 0.5, O 5.2")):
        print(f"\n## {lab}: tok/s Yi / Llama-2 / DeepSeek, and the difference from 'as probed' in the same heat state")
        ref = {}
        for lab2, K, J0, sig, rho in rows:
            v = price(form, K, J0, sig, rho)
            if lab2.startswith("as probed, cool: dead 400"): ref[1.40] = v
            if lab2.startswith("as probed, capped"): ref[1.8] = v
            d = " | vs as probed: " + " / ".join(f"{a - b:+.2f}" for a, b in zip(v, ref[sig])) if sig in ref else ""
            print(f"  {lab2:86} {f3(v)}{d}")
