#!/usr/bin/env python3
"""memo-12 round 35 (Bonnie): heat with the GPU on. For every block of the campaign-3 probe JSONs: the gate series
before it, and per arm the load class, the headroom reading (hr), the CPU cap (policy 4 / 7 max) and the GPU cap,
on a clock rebuilt from the arms' own lengths. Lines carry no time stamp: an arm's length is ramp_s + its measured
window (count x mean), and an arm without counts takes the design's nominal length; the block's remaining native time (setup, ring fills) is unplaced, so each event has a
band [t, t + unplaced]. *device* fields; the clock, the classes and everything derived are *estimated*.
Run: python3 -I -B bonnie_gpuheat_r35.py <repo root> [-v]"""
import sys, os, json, glob, re
root = sys.argv[1]; V = "-v" in sys.argv
kv = lambda s: dict(re.findall(r"(\w+)=(\S+)", s))
F = lambda k, a, d=None: float(k[a]) if a in k and re.fullmatch(r"-?[\d.]+(e[-+]?\d+)?", k[a]) else d

def window(k):
    """measured seconds of an arm, from its own counts"""
    for n, t, u in (("mats", "ms_per_mat", 1e-3), ("gemvs", "us_per_gemv", 1e-6), ("joins", "wall_us", 1e-6), ("layers", "ms_per_layer", 1e-3)):
        if F(k, n) and F(k, t) and F(k, t) > 0: return F(k, n) * F(k, t) * u
    if F(k, "secs"): return F(k, "secs")
    c, g = F(k, "cpu_calls", 0), F(k, "gpu_batches", 0)
    if c and F(k, "cpu_ms_per_graph", -1) > 0: return c * F(k, "cpu_ms_per_graph") * 1e-3
    return None  # unknown: the design's nominal length is used and the arm is flagged

def klass(k, s):
    """load class: C4 = cpus 4-7 loaded, G = GPU busy >= 20 %, both = lane-like"""
    busy = F({"b": k.get("gpu_busy", "0").split(",")[0]}, "b", 0)
    t = k.get("thr", "none")                                       # pinned worker threads: thr entries, else threads=, else set=
    n = t.count(",") + 1 if t != "none" else int(F(k, "threads", 0) or (len(k["set"].split(",")) if "set" in k else 0))
    if k.get("arm") == "gpu": n = 0
    return ("C4" if n >= 4 else "C3" if n == 3 else "c1") + ("+G" if busy >= 20 else "")

def caps(k):
    m = dict(re.findall(r"p([47]):(\d+)/", k.get("max_mhz", "")))
    return (int(m.get("4", 2419)), int(m.get("7", 2841))) if m else None

def blocks():
    for f in sorted(glob.glob(os.path.join(root, "campaigns/03-dense15-5pct/reports/device/*c3probe*.json"))):
        d = json.load(open(f))
        for b in d.get("blocks", []):
            yield os.path.basename(f)[:14], d, b

rows = []
for fn, d, b in blocks():
    g = b.get("gate") or {}; ser = g.get("series") or [[0, float("nan")]]
    arms = []; t = 0.0
    for l in b["lines"]:
        s = str(l); k = kv(s)
        if "max_mhz" not in k or "ramp_s" not in k: continue
        w = window(k); kl = klass(k, s)
        dur = (F(k, "ramp_s", 0) or 0) + (w if w is not None else 1.5 if kl[0] == "C" else 1.0)  # nominal: CPU and together cells 1.5 s, GPU cells 1.0 s (the kits' design lines)
        t += dur
        arms.append(dict(t=t, kl=kl, hr=F(k, "hr"), cap=caps(k), gmax=F(k, "gpu_max_mhz"), busy=k.get("gpu_busy", "").split(",")[0], what=" ".join(s.split()[2:5]), nowin=w is None))
    if not arms: continue
    native = (b.get("native_ms") or b["ms"]) / 1000; slack = max(0.0, native - t)
    full = arms[0]["cap"]; ev = {}
    for a in arms:
        if a["cap"] and "cpu" not in ev and (a["cap"][0] < 2419 or a["cap"][1] < 2841): ev["cpu"] = a
        if a["cap"] and "cpu2" not in ev and a["cap"][0] < 2150: ev["cpu2"] = a
        if a["gmax"] and "gpu" not in ev and a["gmax"] < 840: ev["gpu"] = a
    lane = sum(1 for a in arms if a["kl"] in ("C4+G", "C3+G")); c4 = sum(1 for a in arms if a["kl"] in ("C4", "C3")); gonly = sum(1 for a in arms if a["kl"] == "c1+G")
    r = dict(fn=fn, blk=b["block"], native=native, slack=slack, gate0=ser[0][1], gatepk=max(v for _, v in ser), gate1=ser[-1][1], wait=ser[-1][0] / 1000, to=g.get("timeout"),
             hr0=arms[0]["hr"], hr1=arms[-1]["hr"], armt=t, ev=ev, mix=(lane, c4, gonly, len(arms)), lvl=(b.get("power_before") or {}).get("level"), bc=f"{b.get('battery_c_before')}->{b.get('battery_c_after')}", start_capped=bool(arms[0]["cap"] and arms[0]["cap"] != (2419, 2841)), arms=arms)
    rows.append(r)

print("block table (hr0, hr1: first and last arm; arms: CPU set + GPU / CPU set alone (3 or 4 threads) / GPU alone / all; t in s of arm time, band = + unplaced)")
for r in rows:
    e = "; ".join(f"first {n} cap at {a['t']:.0f} to {a['t'] + r['slack']:.0f} s (hr {a['hr']}, {'gpu max ' + str(int(a['gmax'])) if n == 'gpu' else 'p4/p7 ' + str(a['cap'])}, arm '{a['what']}')" for n, a in r["ev"].items()) or "no cap"
    print(f"{r['fn']} {r['blk']:7s} batt {r['lvl']}% {r['bc']} C gate {r['gate0']:.3f} pk {r['gatepk']:.3f} -> {r['gate1']:.3f} in {r['wait']:.0f} s{' TIMEOUT' if r['to'] else ''}; block {r['native']:.1f} s (arms {r['armt']:.1f}, unplaced {r['slack']:.1f}); arms {r['mix']}; hr {r['hr0']} -> {r['hr1']}{' START CAPPED' if r['start_capped'] else ''}; {e}")
    if V:
        for a in r["arms"]: print(f"      t={a['t']:5.1f} {a['kl']:5s} hr={a['hr']} cap={a['cap']} gmax={a['gmax']} busy={a['busy']} {a['what']}{' (no window)' if a['nowin'] else ''}")

print("\nheadroom per second of arm time, by block (hr is refreshed about every 10 s, so only block-long slopes are read)")
for r in rows:
    if r["armt"] >= 8 and r["hr0"] and r["hr1"] and not r["start_capped"]:
        lo, hi = (r["hr1"] - r["hr0"]) / (r["armt"] + r["slack"]), (r["hr1"] - r["hr0"]) / r["armt"]
        print(f"  {r['fn']} {r['blk']:7s} arms {r['mix']}: {lo:+.4f} to {hi:+.4f} per s; start {r['hr0']}")

print("\ngate after a block: load seconds of the block before, against the wait (the next block's gate in the same file)")
bl = list(blocks()); mix = {(r["fn"], r["blk"]): r["mix"] for r in rows}
for (fa, _, a), (fb, _, b) in zip(bl, bl[1:]):
    g = b.get("gate") or {}; ser = g.get("series")
    if fa == fb and ser:
        print(f"  {fa} after {a['block']:7s} ({(a.get('native_ms') or a['ms']) / 1000:6.1f} s, arms {mix.get((fa, a['block']), '-')}): next gate starts {ser[0][1]:.3f}, peak {max(v for _, v in ser):.3f} at {max(ser, key=lambda x: x[1])[0] / 1000:.0f} s, {'TIMEOUT at ' + format(ser[-1][1], '.3f') if g.get('timeout') else 'passes in ' + format(ser[-1][0] / 1000, '.0f') + ' s'}")

# self-check: the x4s block is rebuilt to the events r34 read off by position (cap at pos 6, GPU cap at pos 11)
x = [r for r in rows if r["blk"] == "x4s"][-1]
assert "pos=6" in x["ev"]["cpu"]["what"] and "pos=11" in x["ev"]["gpu"]["what"], x["ev"]
assert abs(x["native"] - 63.0) < 0.2 and 50 < x["armt"] < 63
print("\nself-check ok (x4s: CPU cap first at pos 6, GPU cap first at pos 11, block 63.0 s)")

# ---- the flush against the GPU's idle gap (x4s lead arms; printed means, *device*; the gap is *estimated*) ----
print("\nx4s lead arms: launch + flush (enq_us) against the GPU's idle time before it (ms_per_mat - gpu_ms)")
d = json.load(open(glob.glob(os.path.join(root, "campaigns/03-dense15-5pct/reports/device/r75a1-*.json"))[0]))
gk = []
for l in [b for b in d["blocks"] if b["block"] == "x4s"][0]["lines"]:
    k = kv(str(l))
    if k.get("mode") == "lead" and "pos" in k and "enq_us" in k: gk.append((F(k, "ms_per_mat") - F(k, "gpu_ms"), F(k, "enq_us"), int(k["pos"]), caps(k), F(k, "gpu_max_mhz")))
for gap, enq, pos, cp, gm in sorted(gk): print(f"  gap {gap:5.2f} ms  enq {enq:5.1f} us  pos {pos:2d}  cap {cp}  gpu max {gm:.0f}")
short = [e for g_, e, *_ in gk if g_ < 0.33]; long_ = [e for g_, e, *_ in gk if g_ >= 0.33]
print(f"  gap < 0.33 ms: n {len(short)}, enq {min(short):.0f} to {max(short):.0f}; gap >= 0.33 ms: n {len(long_)}, enq {min(long_):.0f} to {max(long_):.0f}")
assert max(short) < min(long_), "the knee reading fails"

# ---- what the first cap costs a claim run (all *estimated*; inputs are named in bonnie-joins-r35.md section 2) ----
import math, random
Phi = lambda z: 0.5 * (1 + math.erf(z / math.sqrt(2)))
SUBJ = ("Yi", "Llama-2", "DeepSeek"); PRE = (7.6, 9.8, 8.0); STOCK = (6.90, 6.21, 6.22)   # r33 section 1.4; R75 rung 0
G1, G7, AB = (8.41, 7.85, 7.72), (10.59, 10.46, 10.14), (7.76, 7.04, 6.98)             # Ebony r18 section 3.2 (G); Bonnie r34 section 4.1 (as built)
def price(tps, gain, Tp, Td, sig, after, n=32, runs=10, trials=20000, seed=35):
    """heat clock: a run uses t / T of the time to the first cap (T per kind of load); the cap comes when the used
    fraction passes D, D lognormal with median 1 and spread sig. gain = what the lane multiplies the speed by
    (None: no lane). Tokens after the cap: no lane 0.97 x the speed (r32); lane: speed / gain x after (state B)."""
    out = []
    for i, s in enumerate(SUBJ):
        td = n / tps[i]; fp = PRE[i] / Tp; fe = fp + td / Td
        slow = 0.97 * tps[i] if gain is None else min(tps[i], tps[i] / gain[i] * after)
        def run(D):
            if D >= fe: return tps[i]
            x = 0.0 if D <= fp else (D - fp) / (fe - fp)          # share of the decode before the cap
            return n / (x * n / tps[i] + (1 - x) * n / slow)
        rnd = random.Random(seed); med = []
        for _ in range(trials):
            v = sorted(run(math.exp(rnd.gauss(0, sig))) for _ in range(runs)); med.append((v[runs // 2 - 1] + v[runs // 2]) / 2)
        med.sort(); pend = Phi(math.log(fe) / sig)
        out.append(dict(load=PRE[i] + td, fe=fe, pin=Phi(math.log(fp) / sig), pend=pend, half=run((fp + fe) / 2), mean=sum(med) / trials, p10=med[trials // 10],
                        p5=sum(math.comb(runs, j) * pend ** j * (1 - pend) ** (runs - j) for j in range(runs // 2, runs + 1))))
    return out
gG = tuple(a / b for a, b in zip(G1, STOCK)); gA = tuple(a / b for a, b in zip(AB, STOCK))
CASES = [("no lane, stock", STOCK, None), ("no lane, a method at 12.0", (12.0,) * 3, None), ("lane as built, width 1", AB, gA), ("G, width 1", G1, gG), ("G, rung-7 form", G7, gG), ("G in a method at 12.0", (12.0,) * 3, gG)]
tri = lambda r, k, f: " / ".join(format(x[k], f) for x in r)
for lab, (Tp, Tc, Tl, sig, after) in (("centre: time to the first cap 22 s (prefill), 30 s (CPU decode), 17 s (lane); spread 0.45; lane after the cap 0.95 x no lane", (22, 30, 17, 0.45, 0.95)),
                                      ("cool end: 30 / 36 / 24 s; spread 0.30; 1.00", (30, 36, 24, 0.30, 1.00)), ("hot end: 16 / 28 / 13 s; spread 0.60; 0.88", (16, 28, 13, 0.60, 0.88))):
    print(f"\nclaim run, {lab}  (Yi / Llama-2 / DeepSeek)")
    print("  case                       | load s             | used at the end    | P(cap in prefill)  | P(cap by the end)  | run capped at mid-decode | median of 10: mean    | its 10th percentile   | P(5 or more of 10 capped)")
    for name, tps, gain in CASES:
        r = price(tps, gain, Tp, Tc if gain is None else Tl, sig, after)
        print(f"  {name:26s} | {tri(r, 'load', '.1f'):18s} | {tri(r, 'fe', '.2f'):18s} | {tri(r, 'pin', '.2f'):18s} | {tri(r, 'pend', '.2f'):18s} | {tri(r, 'half', '.2f'):24s} | {tri(r, 'mean', '.2f'):21s} | {tri(r, 'p10', '.2f'):21s} | {tri(r, 'p5', '.3f')}")
# self-check of the pricing: no heat -> the method's own speed; a cap before the decode -> the after-cap speed
c = price((12.0,) * 3, gG, 1e9, 1e9, 0.45, 0.95, trials=50); assert all(abs(x["mean"] - 12.0) < 1e-9 and x["pend"] < 1e-9 for x in c)
c = price(G1, gG, 1e-9, 17, 0.45, 0.95, trials=50); assert all(abs(x["mean"] - 0.95 * s_) < 1e-6 for x, s_ in zip(c, STOCK))
print("\npricing self-check ok")
