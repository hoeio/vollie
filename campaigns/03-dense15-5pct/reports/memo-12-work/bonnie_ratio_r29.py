#!/usr/bin/env python3
"""memo-12 round 29 (Bonnie): what the existing device records allow for P2 / P1 (time of a width-2 pass over a width-1
pass on the lane). Every output is *estimated*: the r14 stack model with its four non-device width-2 factors moved to
their corners. kappa = CPU cost of two positions over one (2.0 = the one-row kernel run twice, needs nothing new;
1.43 = the W-position kernel, host); cg = GPU time at two columns over one (2.0 = two launches of the measured kernel);
C = hot CPU compute rate (34.6 = three A78, device, R72 L3); A2 = second position's attention (r15, with / without r17's
shared conversion). Run: python3 -I -B bonnie_ratio_r29.py"""
import importlib.util, os
_d = os.path.dirname(os.path.abspath(__file__))
_s = importlib.util.spec_from_file_location("r13", os.path.join(_d, "bonnie_stack_r13.py")); r13 = importlib.util.module_from_spec(_s); _s.loader.exec_module(r13)
r5, t2, bt = r13.r5, r13.t2, r13.bt
A2 = {"Yi": (3.2, 0.9), "Llama-2": (5.3, 1.7), "DeepSeek": (5.0, 1.6)}

def ratio(subj, kappa, cg, C, K, model, share=True, O=5.2, lane=True):
    keep_c, keep_r = bt.c, t2.rates
    bt.c = lambda W, e=None: 1 + (kappa - 1) * (W - 1)
    if model == "linear": t2.rates = r5.rates_linear
    try:
        if lane: b1 = t2.body(subj, 1, K, "pair", C=C, c5=1.0)[0]; b2 = t2.body(subj, 2, K, "pair", cg=cg, C=C, c5=1.0)[0]
        else: b1 = bt.SUBJ[subj]["body"]; b2 = b1 * t2.cpu_rate(1, t2.bus_of(t2.CA, C), C) / t2.cpu_rate(2, t2.bus_of(t2.CA, C), C)
    finally: bt.c, t2.rates = keep_c, keep_r
    a, sv = A2[subj]
    return (b2 + O + bt.SUBJ[subj]["xpos"] + a - (sv if share else 0)) / (b1 + O)

assert abs(ratio("Yi", 1 + (bt.c(2) - 1), r13.CG, 40.0, 300, "pair") - 1.094) < 0.002          # r27 section 2.5
assert abs(ratio("Llama-2", 1 + (bt.c(2) - 1), 2.0, 40.0, 300, "linear") - 1.253) < 0.002
if __name__ == "__main__":
    k0 = bt.c(2)
    corners = [("low: W kernel, batched GPU kernel at no extra cost, C 46, shared conversion", (k0, 1.0, 46.0, True)),
               ("centre as priced: W kernel, cg %.2f, C 40" % r13.CG, (k0, r13.CG, 40.0, True)),
               ("W kernel, two GPU launches, C 40", (k0, 2.0, 40.0, True)),
               ("one-row CPU kernel twice, batched GPU kernel, C 40", (2.0, r13.CG, 40.0, True)),
               ("nothing new built: one-row kernel twice, two launches, C 40, no shared conversion", (2.0, 2.0, 40.0, False)),
               ("the same at C 34.6 (three A78 hot, device)", (2.0, 2.0, 34.6, False)),
               ("stock batch 2 as it is: KleidiAI four-row tile (2.60 x a gemv by issue count, r5), two launches, C 40", (2.6, 2.0, 40.0, False))]
    print(f"kappa of the W-position kernel in the model: {k0:.2f}")
    for K in (300, 452):
        print(f"\n## lane, K {K} us: P2 / P1, pair / linear, Yi | Llama-2 | DeepSeek")
        for lab, (ka, cg, C, sh) in corners:
            print(f"  {lab}: " + " | ".join(f"{ratio(s, ka, cg, C, K, 'pair', sh):.2f} / {ratio(s, ka, cg, C, K, 'linear', sh):.2f}" for s in A2))
    print("\n## no lane (CPU alone, bus 25.64): P2 / P1, Yi | Llama-2 | DeepSeek")
    for lab, ka, C, sh in (("W kernel, C 46", k0, 46.0, True), ("W kernel, C 40", k0, 40.0, True), ("W kernel, C 34.6", k0, 34.6, True), ("one-row kernel twice, C 40", 2.0, 40.0, False), ("one-row kernel twice, C 34.6", 2.0, 34.6, False), ("stock batch 2 (tile, 2.60), C 40", 2.6, 40.0, False)):
        print(f"  {lab}: " + " | ".join(f"{ratio(s, ka, 1.0, C, 0, 'pair', sh, lane=False):.2f}" for s in A2))
