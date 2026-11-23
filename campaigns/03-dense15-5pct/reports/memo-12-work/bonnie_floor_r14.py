#!/usr/bin/env python3
"""memo-12 round 14 (Bonnie): the lane stack as a ladder, one evidence class at a time. Every output is *estimated*
arithmetic; the label of a rung is the weakest factor it needs. Loads bonnie_stack_r13.py (and through it the earlier
scripts) unchanged. Run: python3 -I -B bonnie_floor_r14.py"""
import importlib.util, os
_d = os.path.dirname(os.path.abspath(__file__))
_s = importlib.util.spec_from_file_location("r13", os.path.join(_d, "bonnie_stack_r13.py")); r13 = importlib.util.module_from_spec(_s); _s.loader.exec_module(r13)
r5, t2, bt = r13.r5, r13.t2, r13.bt
def lin(f):
    keep = t2.rates; t2.rates = r5.rates_linear
    try: return f()
    finally: t2.rates = keep
def w1(subj, K, O, head_ms, model):          # width 1 on the lane, no guess head
    hs = head_ms / bt.SUBJ[subj]["head"]
    f = lambda: t2.toks(subj, 1, K, "pair" if model != "bonnie" else "bonnie", c5=1.0, hs=hs, O=O)
    return lin(f) if model == "linear" else f()
def w2(subj, K, O, head_ms, model, cg, c5=1.0):
    hs = head_ms / bt.SUBJ[subj]["head"]
    f = lambda: t2.toks(subj, 2, K, "pair" if model != "bonnie" else "bonnie", cg=cg, c5=c5, hs=hs, O=O)
    return lin(f) if model == "linear" else f()
KAI, REP, C7B = r13.KAI, r13.REP, r13.HEAD["c7built"]
RUNGS = [
    ("0 stock, CPU alone", lambda s, m: 950 / (bt.SUBJ[s]["body"] + 8 + KAI[s])),
    ("1 lane, width 1, K 452 as measured, O 8, head as measured", lambda s, m: w1(s, 452, 8, KAI[s], m)),
    ("2 = 1 with K 300 (R75's centre; design unbuilt)", lambda s, m: w1(s, 300, 8, KAI[s], m)),
    ("3 = 2 with the head in the repack type", lambda s, m: w1(s, 300, 8, REP[s], m)),
    ("4 = 3 with O 5.2", lambda s, m: w1(s, 300, 5.2, REP[s], m)),
    ("5a = 4 at width 2, GPU by two launches (cg 2.0), W-position CPU kernel, trained head", lambda s, m: w2(s, 300, 5.2, REP[s], m, 2.0)),
    ("5b = 5a with _mc3 at cg 1.26", lambda s, m: w2(s, 300, 5.2, REP[s], m, r13.CG)),
    ("6 = 5b with campaign 2's two-stage head", lambda s, m: w2(s, 300, 5.2, C7B[s], m, r13.CG)),
    ("7 committed: 5b with C5 0.93 and head x 0.5 of the KleidiAI head", lambda s, m: w2(s, 300, 5.2, KAI[s] * 0.5, m, r13.CG, 0.93)),
]
if __name__ == "__main__":
    r13.r5.selfcheck()
    for s, z in zip(t2.SUBJ, (6.90, 6.21, 6.22)): assert abs(RUNGS[0][1](s, "pair") - z) < 0.01
    for s, z in zip(t2.SUBJ, (11.89, 11.88, 11.43)): assert abs(RUNGS[-1][1](s, "pair") - z) < 0.02
    for m in ("bonnie", "linear", "pair"):
        print(f"\n## {m}: tok/s Yi / Llama-2 / DeepSeek")
        for lab, f in RUNGS: print(f"  {lab:88} " + " / ".join(f"{f(s, m):.2f}" for s in t2.SUBJ))
    print("\nfloor at other K (rung 1 form, width 1, O 8, measured head), pair: " + "; ".join(f"K {K}: " + " / ".join(f"{w1(s, K, 8, KAI[s], 'pair'):.2f}" for s in t2.SUBJ) for K in (0, 150, 300, 452)))
