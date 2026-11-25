#!/usr/bin/env python3
"""memo-12 round 34 (Bonnie): why R75's x4s split read 0.923 where the stage model at K 149 gave 0.663.
Inputs are *device* lines of the R75 session-2 JSON; every derived rate, fit and price is *estimated*.
  1  per arm: the measured window split into launch (enq), CPU rows, wait; the CPU's rate while the GPU kernel runs (rb)
  2  a cell model on those terms; what the cell would read in each phone state and with the launch hidden
  3  the ladder of bonnie_floor_r14.py (loaded unchanged) with a serial launch term, the measured end slack and refitted rates
Run: python3 -I -B bonnie_split_r34.py <repo root>"""
import sys, os, json, re, importlib.util
root = sys.argv[1]; _d = os.path.dirname(os.path.abspath(__file__))
def _load(n):
    s = importlib.util.spec_from_file_location(n, os.path.join(_d, n + ".py")); m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
g_ = lambda l, k: (re.search(r"\b" + k + r"=(\S+)", l) or [None, None])[1]
J = json.load(open(os.path.join(root, "campaigns/03-dense15-5pct/reports/device/r75a1-c3probe-r36-2026-11-24T1345.json")))
L = [l for b in J["blocks"] if b["block"] == "x4s" for l in b["lines"] if "x4 split pos=" in l]
assert len(L) == 22
MB = 22016 * 4096 * 0.5625 / 1e6                       # fused [gate;up], MB of weight bytes
PH = lambda p: "A" if p <= 4 else "B" if p <= 10 else "C" if p <= 14 else "D"   # by the printed caps (R75 S2.2): none / 2,150 / 1,785 / 1,420
RA = {"A": 28.57, "B": 28.3, "C": 27.7, "D": 26.77}   # CPU alone GB/s: A and D are the cpu arms (pos 0, pos 21); B and C interpolated on the cap (estimated)
arms = []
for l in L:
    f = lambda k: float(g_(l, k)); p = int(g_(l, "pos")); mode = g_(l, "mode"); a = dict(pos=p, mode=mode, sh=f("share"), ms=f("ms_per_mat"), ph=PH(p), line=l)
    if mode == "lead":
        a.update(enq=f("enq_us") / 1e3, wait=f("wait_ms"), gms=f("gpu_ms"), dev=f("gpu_dev_ms"), grows=int(g_(l, "gpu_rows")))
        a["cpu"] = a["ms"] - a["enq"] - a["wait"]                       # CPU rows in the measured window (the printed cpu_ms includes the warm-up)
        gb, cb = a["grows"] / 22016 * MB, (1 - a["grows"] / 22016) * MB; ov = min(a["dev"], a["cpu"])
        a["g"] = gb / a["dev"]; a["rb"] = (cb - RA[a["ph"]] * (a["cpu"] - ov)) / ov   # GPU starts as the flush returns (gpu_ms - dev = enq within 20 us)
        a["rb_lo"] = (cb - 28.57 * (a["cpu"] - ov)) / ov; a["rb_hi"] = (cb - 26.77 * (a["cpu"] - ov)) / ov
        a["K"] = a["gms"] - a["dev"]; a["s10"] = a["cpu"] / ((1 - a["grows"] / 22016) * MB / RA["A"])
        a["lead_gpu"] = a["enq"] + a["cpu"] - (0.02 + a["gms"])        # how long before the CPU rows end the GPU kernel ended (QUEUED taken 20 us after a0)
    arms.append(a)
lead = [a for a in arms if a["mode"] == "lead"]

# ---- 2. cell model: t = Lh (all CPU threads idle) + max(tc, tg) + slack; CPU at ra while the GPU is dead, rb while it runs
def cell(f, ra, rb, g, Lh, Kp=0.0, J0=0.2, Jw=0.4, dev0=0.014):
    dev = dev0 + f * MB / g; tg = Kp + dev; w = (1 - f) * MB
    if w <= ra * Kp: tc = w / ra
    else:
        tc = Kp + (w - ra * Kp) / rb
        if tc > tg: tc = tg + (w - ra * Kp - rb * dev) / ra
    d = tc - tg                                                          # > 0: the GPU ended first
    return Lh + max(tc, tg) + (J0 if d <= 0 else J0 * max(0.0, 1 - d / Jw))   # slack: 5-point descriptive fit (pos 7 to 9, 12 to 14)
def best(**kw): return min((cell(i / 200, **kw), i / 200) for i in range(20, 141))
STATE = {"A": dict(ra=28.57, rb=20.6, g=25.3, Lh=0.214), "B": dict(ra=28.3, rb=10.5, g=23.0, Lh=0.215),
         "C": dict(ra=27.7, rb=17.6, g=24.0, Lh=0.132), "D": dict(ra=26.77, rb=12.1, g=22.6, Lh=0.239)}

# ---- 3. ladder: bonnie_floor_r14 unchanged; t2.body replaced by the same stage loop with a serial launch and a fixed slack
fl = _load("bonnie_floor_r14"); t2, r5, r13, bt, bj = fl.t2, fl.r5, fl.r13, fl.bt, fl.t2.bj
A0 = (t2.cpu_rate, t2.bus_of, t2.GD, t2.BUS, t2.body)
P = dict(Lh=0.0, J=None)                                # serial launch us per split stage; slack us (None: as committed, J1 0.107 * K)
def body_r34(subj, W, K_us, variant, cg=1.45, C=40.0, c5=0.93, J0=0.0, J1=0.107, gdev=False):
    s = bt.SUBJ[subj]; ca1 = t2.cpu_rate(1, t2.bus_of(t2.CA, C), C); ca, cb, g = t2.rates(W, variant, cg, C, gdev)
    tot = sum(w for _, w in bj.STAGES[subj]); K = K_us / 1e3; t = n = 0.0
    ser = P["Lh"] / 1e3 * (1 if W == 1 else 1 + 40 / max(P["Lh"], 1e-9) * (P["Lh"] > 0))   # width 2 by two launches: + 40 us for the second enqueue (estimated)
    sl = (J0 / 1e3 + J1 * K) if P["J"] is None else P["J"] / 1e3
    for _, w in bj.STAGES[subj]:
        B = s["body"] * c5 * w / tot / bj.NL[subj] * ca1; alone = B / ca
        sp = ser + K + (B - ca * K) / (cb + g) + sl if B > ca * K else 1e9
        if sp < alone: t += sp; n += 1
        else: t += alone
    return t * bj.NL[subj], int(n), 0.0
def state(al=0.25, cb1=None, g1=None, C=40.0):
    """alpha, CPU rate beside the GPU at width 1 and GPU device rate (None: as committed, 18.95 and 24.6)."""
    t2.cpu_rate, t2.bus_of, t2.GD, t2.BUS = A0[:4]; r5.GD, r5.BUS = A0[2], A0[3]
    cb0 = A0[0](1, A0[3] - A0[2], C) if cb1 is None else cb1
    if al != 0.25:
        t2.cpu_rate = lambda W, Rb, C=40.0: 1 / (max(1 / Rb, bt.c(W) / C) + al * min(1 / Rb, bt.c(W) / C))
        t2.bus_of = lambda r, C=40.0: 1 / (1 / r - al / C)
    if g1 is not None: t2.GD = r5.GD = g1
    t2.BUS = r5.BUS = t2.GD + t2.bus_of(cb0, C)
KAI, REP, C7B, S = fl.KAI, fl.REP, fl.C7B, t2.SUBJ
f3 = lambda xs: " / ".join(f"{x:.2f}" for x in xs)
H = 13.3
JOIN = [  # label, parallel GPU dead time K, serial launch Lh, slack J (us)
    ("lead's stage form, K_eff 162.3 (R75 S2.10)", 162.3, 0.0, None),
    ("as built in x4s, cool: launch serial 207 + H, slack 115", 0.0, 207 + H, 115),
    ("as built, phase C values: launch serial 124 + H, slack 115", 0.0, 124 + H, 115),
    ("launch hidden behind the rows (unbuilt): serial 52 + H, GPU dead 155, slack 115", 155.0, 52 + H, 115),
    ("F persistent kernel (unbuilt fast version): K 17.95, slack 60", 17.95, 0.0, 60),
    ("F, slack 0 (upper bound)", 17.95, 0.0, 0)]
RUNG = [("2 width 1, O 8, head as measured", lambda x, m, K: fl.w1(x, K, 8, KAI[x], m)),
        ("4 width 1, repack head, O 5.2", lambda x, m, K: fl.w1(x, K, 5.2, REP[x], m)),
        ("6 width 2 two launches cg 2.0, two-stage head", lambda x, m, K: fl.w2(x, K, 5.2, C7B[x], m, 2.0)),
        ("7 = 6 with C5 0.93 and head x 0.5", lambda x, m, K: fl.w2(x, K, 5.2, KAI[x] * 0.5, m, 2.0, 0.93))]

def demo():
    assert abs(MB - 50.72) < 0.01
    r2 = _load("bonnie_joins_r2")                     # the prediction's own model: best ratio at K 149, no launch term, A 1.835
    b2 = min(r2.split(1.835, 22016, i / 200, 149)[2] / 1.835 for i in range(20, 141)); assert abs(b2 - 0.654) < 0.003, b2   # r2's duty-aware split, fine share grid
    b = 0.621 + 0.521 * 0.149 / 1.835; assert abs(b - 0.663) < 0.001      # R75 S2.7's stage form (lane factor 0.621, K weight 0.521), the 0.663 of the brief
    state(); P.update(Lh=0.0, J=None); t2.body = body_r34
    for x, z in zip(S, (9.61, 8.97, 8.72)): assert abs(fl.w1(x, 162.3, 8, KAI[x], "linear") - z) < 0.01       # R75 S2.10 rung 2
    for x, z in zip(S, (11.91, 11.71, 11.29)): assert abs(fl.w2(x, 162.3, 5.2, KAI[x] * 0.5, "pair", 2.0, 0.93) - z) < 0.01   # rung 7 pair
    state(0.17)
    for x, z in zip(S, (12.26, 12.07, 11.61)): assert abs(fl.w2(x, 162.3, 5.2, KAI[x] * 0.5, "pair", 2.0, 0.93) - z) < 0.01   # rung 7 pair, alpha 0.17
    state()
    for a in lead:                                     # the cell model reproduces every lead arm it was fitted on within 8 %
        if a["pos"] in (7, 15, 20): continue           # arms in which the cap or the GPU clock changed
        assert abs(cell(a["sh"], **STATE[a["ph"]]) / a["ms"] - 1) < 0.08, (a["pos"], cell(a["sh"], **STATE[a["ph"]]), a["ms"])
    return b

if __name__ == "__main__":
    b0 = demo()
    print("== 1. x4s lead arms (device inputs; cpu = ms_per_mat - enq - wait; rb = CPU GB/s while the GPU kernel runs, band over ra 28.57..26.77)")
    print("pos ph share | ms_per_mat enq cpu wait | K=gpu_ms-dev | GPU GB/s | rb (band) | rb+g | S10 window | GPU ended before CPU rows by (ms) | thr cw 5/6/7, niv 5/6/7/4, nv4/mats")
    for a in lead:
        th = [t.split(":") for t in g_(a["line"], "thr").split(",")]
        print(f"{a['pos']:2} {a['ph']} {a['sh']:.2f} | {a['ms']:.3f} {a['enq']:.3f} {a['cpu']:.3f} {a['wait']:.3f} | {a['K']:.3f} | {a['g']:.1f} | {a['rb']:.1f} ({a['rb_lo']:.1f}..{a['rb_hi']:.1f}) | {a['rb'] + a['g']:.1f} | "
              f"{a['s10']:.3f} | {a['lead_gpu']:+.3f} | {'/'.join(t[1] for t in th[:3])}, {'/'.join(t[2] for t in th)}, {th[3][3]}/{g_(a['line'], 'mats')}")
    for a in arms:
        if a["mode"] != "lead": print(f"   pos {a['pos']} {a['mode']} share {a['sh']:.2f} phase {a['ph']}: ms_per_mat {a['ms']:.3f} = {MB / a['ms']:.2f} GB/s over all rows; ratio to pos 0 {a['ms'] / arms[0]['ms']:.3f}")
    for ph in "ABCD":
        v = [a for a in lead if a["ph"] == ph and a["pos"] not in (7, 15, 20)]
        print(f"   phase {ph}: rb {min(a['rb'] for a in v):.1f}..{max(a['rb'] for a in v):.1f}, GPU {min(a['g'] for a in v):.1f}..{max(a['g'] for a in v):.1f}, enq {min(a['enq'] for a in v) * 1e3:.0f}..{max(a['enq'] for a in v) * 1e3:.0f} us, "
              f"K {min(a['K'] for a in v) * 1e3:.0f}..{max(a['K'] for a in v) * 1e3:.0f} us, split factor ra / (rb + g) {RA[ph] / (sum(a['rb'] + a['g'] for a in v) / len(v)):.3f}, CPU slowdown ra / rb {RA[ph] / (sum(a['rb'] for a in v) / len(v)):.2f}")

    print("\n== 2. the cell in each state (ratio to the cpu arm of the same state; share in brackets)")
    print(f"   stage model as priced (K 149, no launch term, no slack): {b0:.3f}")
    for ph, alone in (("A", 1.776), ("B", MB / 28.3), ("C", MB / 27.7), ("D", 1.895)):
        st = STATE[ph]; t, s = best(**st); th, sh = best(**dict(st, Lh=st["Lh"] / 4), Kp=st["Lh"] * 0.97); tf, sf = best(**dict(st, Lh=0.0), Kp=0.018, J0=0.06)
        ti, si = best(**dict(st, Lh=0.0), J0=0.0)
        print(f"   state {ph}: as built {t / alone:.3f} ({s:.2f}) | launch hidden behind the rows {th / alone:.3f} ({sh:.2f}) | F, K 18 us, slack 60 {tf / alone:.3f} ({sf:.2f}) | no launch, no slack {ti / alone:.3f} ({si:.2f})")
    A = STATE["A"]; alone = 1.776
    steps = [("priced", b0), ("cool rates of phase A, K 149 beside the rows, no slack", best(**dict(A, Lh=0.0), Kp=0.149, J0=0.0)[0] / alone),
             ("+ launch serial (enq 214 us, all four threads idle)", best(**A, J0=0.0)[0] / alone), ("+ end slack as fitted", best(**A)[0] / alone),
             ("state C as built (cap 1,785, GPU 676)", best(**STATE["C"])[0] / (MB / 27.7)), ("state B as built (throttle onset)", best(**STATE["B"])[0] / (MB / 28.3)),
             ("the printed cell: mean of a B arm and a C arm over the mean of the A and D cpu arms", 0.923)]
    print("   path from 0.663 to 0.923: " + "; ".join(f"{lab} {v:.3f}" for lab, v in steps))
    print("   measured best arm of each phase over the cpu arm of its state: " + "; ".join(
        f"{ph} pos {a['pos']} share {a['sh']:.2f} {a['ms'] / al:.3f}" for ph, al in (("A", 1.776), ("B", MB / 28.3), ("C", MB / 27.7), ("D", 1.895)) for a in [min((x for x in lead if x["ph"] == ph), key=lambda x: x["ms"])]))

    print("\n== 3. ladder, tok/s Yi / Llama-2 / DeepSeek (all estimated)")
    RATES = [("rates as committed (cb 18.95, g 24.6, factor 0.589)", None, None), ("cool refit centre (cb 18.9, g 23.1, factor 0.610)", 18.9, 23.1), ("cool refit bad end (cb 17.9, g 22.2, factor 0.640)", 17.9, 22.2)]
    top = {}
    for rl, cb1, g1 in RATES:
        for al in (0.25, 0.17):
            state(al, cb1, g1); print(f"\n-- {rl}, alpha {al}")
            for jl, K, Lh, Jx in JOIN:
                P.update(Lh=Lh, J=Jx); print(f"   {jl}")
                for lab, f in RUNG:
                    v = {m: [f(x, m, K) for x in S] for m in ("linear", "pair")}; top[(rl, al, jl, lab)] = v
                    print(f"      {lab:48} linear {f3(v['linear'])} | pair {f3(v['pair'])}")
    hit = [(k, m) for k, v in top.items() for m in v if min(v[m]) >= 12.0]
    print("\n   rungs at or above 12.00 on all three: " + ("none" if not hit else ""))
    for k, m in hit: print(f"      {m} | {k[0]} | alpha {k[1]} | {k[2]} | {k[3]}: {f3(top[k][m])}")
    state(); P.update(Lh=0.0, J=None); t2.body = A0[4]
    print("\n   crude, as R75 S2.10: the cell's ratio applied to the whole body (O 8, measured head): " + "; ".join(
        f"{lab} {f3(950 / (bt.SUBJ[x]['body'] * fac + 8 + KAI[x]) for x in S)}" for lab, fac in (("stock 1.000", 1.0), ("printed 0.923", 0.923), ("state A as built 0.832", 0.832), ("state C as built 0.816", 0.816), ("state D as built 0.951", 0.951))))

# ---- 4. next question 1: the KleidiAI Q4_0 path claims row chunks (nth x GGML_KLEIDIAI_CHUNK_MULTIPLIER, default 4; thread i
# starts on chunk i, then takes the next free one). When does the CPU side end if the lead starts `late` ms after the others?
def claim_end(work_ms, late, mult, nth=4):
    """work_ms: thread-ms of CPU rows in all. Returns the time the last chunk ends (ms after the workers start)."""
    n = nth * mult; c = work_ms / n; free = [late] + [0.0] * (nth - 1); nxt = nth; end = [f + c for f in free]   # each thread's first chunk is its own
    while nxt < n:
        i = min(range(nth), key=lambda j: end[j]); end[i] += c; nxt += 1
    return max(end)
if __name__ == "__main__":
    assert abs(claim_end(4.0, 0.0, 4) - 1.0) < 1e-9
    W = 4 * 1.269                                       # state A at share 0.42: CPU rows 1.269 ms on four threads (section 2 model)
    print("\n== 4. launch hidden behind claimed chunks, state A, share 0.42: CPU rows end (ms after the workers start); serial launch was 0.214 + 1.269 = 1.483")
    for mult in (4, 8, 16, 32):
        print(f"   chunk multiplier {mult:2}: lead on time {claim_end(W, 0.0, mult):.3f}; lead 0.214 late {claim_end(W, 0.214, mult):.3f}; ideal {W / 4 + 0.214 / 4:.3f}")
