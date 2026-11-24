# Lead's R75 price run: committed memo-12 scripts loaded unchanged; outputs *estimated*.
import importlib.util, sys
D = sys.argv[1]
s = importlib.util.spec_from_file_location("fl", D + "/bonnie_floor_r14.py"); fl = importlib.util.module_from_spec(s); s.loader.exec_module(fl)
t2, r5, r13 = fl.t2, fl.r5, fl.r13
S = t2.SUBJ
KEFF, CG, KF, KY = 162.3, 3.316, 4.65 + 13.3, 44.3 + 13.3
f3 = lambda xs: " / ".join(f"{x:.2f}" for x in xs)
A0 = (t2.cpu_rate, t2.bus_of, t2.BUS, r5.BUS)
def alpha(al, C=40.0):
    t2.cpu_rate, t2.bus_of, t2.BUS, r5.BUS = A0
    if al == 0.25: return
    cb1 = A0[0](1, A0[2] - t2.GD, C)
    t2.cpu_rate = lambda W, Rb, C=40.0: 1 / (max(1 / Rb, t2.bt.c(W) / C) + al * min(1 / Rb, t2.bt.c(W) / C))
    t2.bus_of = lambda r, C=40.0: 1 / (1 / r - al / C)
    t2.BUS = r5.BUS = t2.GD + t2.bus_of(cb1, C)
print("== 5.3 prices at measured cg", CG)
for lab, K in (("K_eff", KEFF), ("K_best F (cond.)", KF), ("Y14 design K", KY), ("K 0", 0.0)):
    print(lab, K, "pair B:", f3(t2.toks(x, 2, K, "pair", cg=CG) for x in S),
          "| pair E:", f3(t2.toks(x, 2, K, "pair", cg=CG, J0=60, J1=0) for x in S),
          "| linear:", f3(r5.toks_linear(x, K, CG) for x in S))
print("old at K_eff:", f3(t2.old(x, KEFF) for x in S))
print("== same at cg 2.0 (two launches of the general kernel, descriptive)")
for lab, K in (("K_eff", KEFF), ("K_best F", KF), ("K 0", 0.0)):
    print(lab, "pair B:", f3(t2.toks(x, 2, K, "pair", cg=2.0) for x in S), "| pair E:", f3(t2.toks(x, 2, K, "pair", cg=2.0, J0=60, J1=0) for x in S), "| linear:", f3(r5.toks_linear(x, K, 2.0) for x in S))
KAI, REP, C7B = fl.KAI, fl.REP, fl.C7B
for al in (0.25, 0.17):
    alpha(al)
    print(f"\n== ladder alpha {al}")
    for m in ("linear", "pair"):
        rows = [("0 stock", lambda x: 950 / (fl.bt.SUBJ[x]["body"] + 8 + KAI[x])),
          ("1 w1 K452 O8 KAI head", lambda x: fl.w1(x, 452, 8, KAI[x], m)),
          ("2 w1 K_eff 162.3", lambda x: fl.w1(x, KEFF, 8, KAI[x], m)),
          ("2F w1 K 17.95 (F, unbuilt)", lambda x: fl.w1(x, KF, 8, KAI[x], m)),
          ("3 +repack head", lambda x: fl.w1(x, KEFF, 8, REP[x], m)),
          ("4 +O 5.2", lambda x: fl.w1(x, KEFF, 5.2, REP[x], m)),
          ("5a w2 cg2.0 (two launches)", lambda x: fl.w2(x, KEFF, 5.2, REP[x], m, 2.0)),
          ("5b w2 _mc3 cg3.316 measured", lambda x: fl.w2(x, KEFF, 5.2, REP[x], m, CG)),
          ("6 5a + two-stage head", lambda x: fl.w2(x, KEFF, 5.2, C7B[x], m, 2.0)),
          ("7 5a + C5 0.93 + head x0.5", lambda x: fl.w2(x, KEFF, 5.2, KAI[x] * 0.5, m, 2.0, 0.93)),
          ("7F = 7 at K 17.95", lambda x: fl.w2(x, KF, 5.2, KAI[x] * 0.5, m, 2.0, 0.93)),
          ("7c = 7 at cg 1.26 (old centre, not measured)", lambda x: fl.w2(x, KEFF, 5.2, KAI[x] * 0.5, m, r13.CG, 0.93)),
          ("w1 K0 O5.2 repack", lambda x: fl.w1(x, 0, 5.2, REP[x], m)),
          ("7 at K0 cg2.0", lambda x: fl.w2(x, 0, 5.2, KAI[x] * 0.5, m, 2.0, 0.93))]
        print(" ", m)
        for lab, f in rows: print(f"   {lab:44} {f3(f(x) for x in S)}")
alpha(0.25)
print("\n== worth-a-build deltas over K_eff at cg", CG)
for lab, K in (("F", KF), ("Y14", KY), ("B/B+ s1", 188.6 + 40 + 13.3)):
    for nm, f in (("pair B", lambda x, K: t2.toks(x, 2, K, "pair", cg=CG)), ("pair E", lambda x, K: t2.toks(x, 2, K, "pair", cg=CG, J0=60, J1=0)), ("linear", lambda x, K: r5.toks_linear(x, K, CG))):
        print(lab, nm, " / ".join(f"{f(x, K) - f(x, KEFF):+.3f}" for x in S))
print("gap to 79.2 ms at K_eff, pair B / pair E / linear / old (ms):")
for nm, f in (("pair B", lambda x: t2.toks(x, 2, KEFF, "pair", cg=CG)), ("pair E", lambda x: t2.toks(x, 2, KEFF, "pair", cg=CG, J0=60, J1=0)), ("linear", lambda x: r5.toks_linear(x, KEFF, CG)), ("old", lambda x: t2.old(x, KEFF))):
    print(nm, " / ".join(f"{950 / f(x) / 0.95 * 0.95 - 79.2:+.1f}" for x in S))
print("K0 at cg", CG, "pair E / linear:", f3(t2.toks(x, 2, 0, "pair", cg=CG, J0=60, J1=0) for x in S), f3(r5.toks_linear(x, 0, CG) for x in S))
