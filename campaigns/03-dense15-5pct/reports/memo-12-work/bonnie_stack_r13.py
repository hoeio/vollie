#!/usr/bin/env python3
"""memo-12 round 13 (Bonnie): the lane stacks with C5 and C7 at what the evidence supports. Every output is *estimated*.
Loads bonnie_joins_r5.py (and through it bonnie_tree_r2.py) unchanged. Run: python3 -I -B bonnie_stack_r13.py
Head ms per token (Yi / Llama-2 / DeepSeek):
  kai      9.04 / 4.40 / 13.88   head in KleidiAI (*device*, R72 l7)
  as_was   kai x 0.5             the committed stacks' C7
  repack   7.7 / 3.7 / 11.8      head in the repack type (r12, *estimated*)
  c7built  repack x 0.80 / 0.91 / 0.76   campaign 2's two-stage head by bytes at K 1024 (r12)
Body: c5 0.93 (as committed) or 1.00 (C5 out)."""
import importlib.util, os
_d = os.path.dirname(os.path.abspath(__file__))
_s = importlib.util.spec_from_file_location("r5", os.path.join(_d, "bonnie_joins_r5.py")); r5 = importlib.util.module_from_spec(_s); _s.loader.exec_module(r5)
t2, bt = r5.t2, r5.bt
KAI = dict(zip(t2.SUBJ, (9.04, 4.40, 13.88))); REP = dict(zip(t2.SUBJ, (7.7, 3.7, 11.8))); F = dict(zip(t2.SUBJ, (0.80, 0.91, 0.76)))
HEAD = {"as_was": {s: KAI[s] * 0.5 for s in t2.SUBJ}, "kai": KAI, "repack": REP, "c7built": {s: REP[s] * F[s] for s in t2.SUBJ}}
CG = t2.GD / 19.5
def toks(subj, K, model, head, c5):
    hs = HEAD[head][subj] / bt.SUBJ[subj]["head"]
    if model == "linear":
        keep = t2.rates; t2.rates = r5.rates_linear
        try: return t2.toks(subj, 2, K, "pair", cg=CG, c5=c5, hs=hs)
        finally: t2.rates = keep
    return t2.toks(subj, 2, K, model, cg=CG, c5=c5, hs=hs)
ROWS = [("as committed: C5 0.93, C7 x 0.5", "as_was", 0.93), ("C5 out, C7 x 0.5", "as_was", 1.0), ("C5 in, C7 as built", "c7built", 0.93),
        ("C5 out, C7 as built (knob on)", "c7built", 1.0), ("C5 out, no C7, head in repack", "repack", 1.0), ("C5 out, no C7, head in KleidiAI", "kai", 1.0)]
if __name__ == "__main__":
    r5.selfcheck()
    for s, z in zip(t2.SUBJ, (11.59, 11.54, 11.13)): assert abs(toks(s, 300, "linear", "as_was", 0.93) - z) < 0.02      # r5 section 2.5
    for s, z in zip(t2.SUBJ, (11.89, 11.88, 11.43)): assert abs(toks(s, 300, "pair", "as_was", 0.93) - z) < 0.02
    for model in ("linear", "pair"):
        print(f"\n## {model} (W 2, GPU 19.5 GB/s beside the CPU, C 40): tok/s Yi / Llama-2 / DeepSeek at K 0, 150, 300; K (us) for 12.00")
        for lab, head, c5 in ROWS:
            cells = ["  ".join(" / ".join(f"{toks(s, K, model, head, c5):.2f}" for s in t2.SUBJ) for K in (0, 150, 300))]
            ks = " / ".join((lambda v: "none" if v is None else f"{v:.0f}")(t2.solve(lambda k, s=s: toks(s, k, model, head, c5), 0, 600, 12.0)) for s in t2.SUBJ)
            print(f"  {lab:34} {cells[0]}   | {ks}")
