#!/usr/bin/env python3
"""memo-12 pricing (Bonnie): what a GPU join costs per token under each lane design. Every output is *estimated*.

Inputs (*device*, R74): x3 togsum cpu alone 25.64, both 19.77 (CPU) + 21.53 (GPU) GB/s; x3 rt 2816 rows; x3 split.
Stage model, per layer: four dependency stages {q,k,v} -> {o} -> {gate,up} -> {down}; the output of one is the
input of the next, so every stage that has GPU rows ends in one join. A = the stage's CPU-alone time.
  split at GPU share f, K = GPU-side overhead per dispatch (not device time):
     t(f) = max((1 - f) * A * RC, f * A * RG + K)          RC = 25.64 / 19.77, RG = lane * RC / (RC - lane)
     best f:  t = lane * A + KAPPA * K,   KAPPA = RC / (RC + RG)        (W 1: lane 0.621, KAPPA 0.521)
  a stage is split only if that beats A (CPU alone).
The stack goes through memo-10's bonnie_tree.best() unchanged: disp = this model's body minus body * lane.
Run: python3 -I bonnie_joins.py   (bonnie_tree.py is loaded by path)
"""
import importlib.util, os
_p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "memo-10-work", "bonnie_tree.py")
_s = importlib.util.spec_from_file_location("bonnie_tree", _p); bt = importlib.util.module_from_spec(_s); _s.loader.exec_module(bt)

SUM, G_EFF, RC = 1.611, 0.611 * 26 / 0.64, 25.64 / 19.77
NL = {"Yi": 32, "Llama-2": 32, "DeepSeek": 30}
M44, M5, M11 = 4096 * 4096, 512 * 4096, 11008 * 4096                      # weights per matrix
STAGES = {s: [("qkv", 3 * M44), ("o", M44), ("gu", 2 * M11), ("down", M11)] for s in ("Llama-2", "DeepSeek")}
STAGES["Yi"] = [("qkv", M44 + 2 * M5), ("o", M44), ("gu", 2 * M11), ("down", M11)]
PIN = {"qkv": 1 / 3, "gu": 0.5}                                           # whole-matrix shares: q of q,k,v; gate of gate,up
C = 40.0

def lane(W): return 26.0 / (26.0 + 0.64 * G_EFF / bt.c(W))               # bonnie_tree's own lane
def rg(W): return lane(W) * RC / (RC - lane(W))
def kappa(W): return RC / (RC + rg(W))

def body(subj, W, K_us, mode="rows", c5=1.0, kap=None):
    """(body ms with the lane, joins per layer, GPU share of the body bytes). mode: rows | whole | whole+down"""
    s, K = bt.SUBJ[subj], K_us / 1e3
    tot = sum(w for _, w in STAGES[subj]); kb = bt.kB(W, C) if W > 1 else 1.0
    t = joins = gbytes = 0.0
    for name, w in STAGES[subj]:
        A = s["body"] * kb * c5 * w / tot / NL[subj]
        k = kappa(W) if kap is None else kap
        cand = [(A, 0, 0.0)]
        if mode == "rows" or (mode == "whole+down" and name == "down"):
            f = (A * RC - K) / (A * (RC + rg(W)))
            if f > 0: cand.append((lane(W) * A + k * K, 1, f))
        if mode != "rows" and name in PIN and not (subj == "Yi" and name == "qkv"):
            f = PIN[name]; cand.append((max((1 - f) * A * RC, f * A * rg(W) + K), 1, f))
        ts, j, f = min(cand); t += ts; joins += j; gbytes += f * w
    return t * NL[subj], int(joins), gbytes / tot

def price(subj, K_us, mode="rows", kap=None):
    s = bt.SUBJ[subj]
    b1, j1, sh = body(subj, 1, K_us, mode, kap=kap)
    pg = 950 / (b1 + 8 + s["head"])
    b2, j2, _ = body(subj, 2, K_us, mode, c5=0.93, kap=kap)
    d2 = b2 - s["body"] * bt.kB(2, C) * 0.93 * lane(2)
    t, tau, _ = bt.best(s, 2, C=C, g=G_EFF, disp=d2, hs=0.5, c5=0.93, O=5.2)
    return dict(joins=j1, disp1=b1 - s["body"] / SUM, pg=pg, joins2=j2, disp2=d2, stack=950 / t, share=sh,
                gpu_gb=sh * sum(w for _, w in STAGES[subj]) * 0.5625 * NL[subj] / 1e9)

def selfcheck():
    bt.selfcheck()
    assert abs(lane(1) - 1 / SUM) < 2e-3 and abs(kappa(1) - 0.521) < 2e-3, (lane(1), kappa(1))
    for subj, n, pg, st, pg0, st0 in (("Yi", 64, 7.86, 9.55, 10.34, 12.23), ("Llama-2", 64, 7.39, 9.50, 9.53, 11.87),
                                      ("DeepSeek", 60, 7.30, 9.26, 9.22, 11.44)):          # R74-result section 4
        s = bt.SUBJ[subj]
        for d, a, b in ((n * 0.452, pg, st), (0.0, pg0, st0)):
            assert abs(950 / (s["body"] / SUM + d + 8 + s["head"]) - a) < 0.006
            assert abs(950 / bt.best(s, 2, C=C, g=G_EFF, disp=d, hs=0.5, c5=0.93, O=5.2)[0] - b) < 0.011
        # K = 0 with rows: the o stage and every other stage split, so this model equals R74's zero-dispatch column
        assert abs(price(subj, 0)["pg"] - pg0) < 0.02 and abs(price(subj, 0)["stack"] - st0) < 0.02, price(subj, 0)
    # x3 split, one 11008 x 4096 matrix, CPU alone 0.948 ms: measured best 0.865 ms at share 0.25
    A = 0.948; assert abs(lane(1) * A + kappa(1) * 0.530 - 0.865) < 0.005

DESIGNS = [  # name, mode, K centre, K low, K high (us per join, GPU side)
    ("R74 as priced (2 joins/layer in series)", None, 452, 452, 452),
    ("E  shares re-balanced, join as measured", "rows", 500, 452, 560),
    ("B  E + poll instead of clFinish", "rows", 470, 400, 560),
    ("B+ B + dispatch from a fast core / recorded queue", "rows", 350, 250, 500),
    ("A  B+ + pre-enqueue behind a gate", "rows", 290, 205, 500),
    ("C  A + GPU kept out of idle", "rows", 120, 35, 500),
    ("F  persistent kernel, shared-memory mailbox", "rows", 30, 5, 100),
    ("D1 whole matrices: gate (+ q) on GPU, K as E", "whole", 500, 452, 560),
    ("D2 whole matrices + down rows, K as A", "whole+down", 290, 205, 500),
    ("D3 whole matrices + down rows, K as C", "whole+down", 120, 35, 500),
    ("Z  zero join cost", "rows", 0, 0, 0),
]

if __name__ == "__main__":
    selfcheck()
    print(f"lane W1 {lane(1):.3f} W2 {lane(2):.3f}; kappa W1 {kappa(1):.3f} W2 {kappa(2):.3f}; RG W1 {rg(1):.3f} W2 {rg(2):.3f}")
    for subj in bt.SUBJ:
        tot = sum(w for _, w in STAGES[subj])
        print(subj, "stage CPU-alone ms per layer:", ", ".join(f"{n} {bt.SUBJ[subj]['body'] * w / tot / NL[subj]:.3f}" for n, w in STAGES[subj]),
              "| break-even K us:", ", ".join(f"{n} {bt.SUBJ[subj]['body'] * w / tot / NL[subj] * (1 - lane(1)) / kappa(1) * 1e3:.0f}" for n, w in STAGES[subj]))
    for name, mode, kc, kl, kh in DESIGNS:
        print(f"\n## {name}: K {kc} ({kl} to {kh}) us")
        for subj in bt.SUBJ:
            s = bt.SUBJ[subj]
            if mode is None:
                d = 2 * NL[subj] * kc / 1e3
                print(f"  {subj:9} joins/layer 2  disp {d:5.1f} ms  P_G {950 / (s['body'] / SUM + d + 8 + s['head']):.2f}  "
                      f"stack {950 / bt.best(s, 2, C=C, g=G_EFF, disp=d, hs=0.5, c5=0.93, O=5.2)[0]:.2f}")
                continue
            # band: worst = K high with kappa 0.58, best = K low with kappa 0.38 (kappa band only where the share is free)
            c, worst, best_ = price(subj, kc, mode), price(subj, kh, mode, kap=0.58 if mode == "rows" else None), price(subj, kl, mode, kap=0.38 if mode == "rows" else None)
            print(f"  {subj:9} joins/layer {c['joins']} (W2 {c['joins2']})  disp {c['disp1']:5.1f} ms ({best_['disp1']:.1f} to {worst['disp1']:.1f})  "
                  f"P_G {c['pg']:.2f} ({worst['pg']:.2f} to {best_['pg']:.2f})  stack disp {c['disp2']:5.1f}  stack {c['stack']:.2f} ({worst['stack']:.2f} to {best_['stack']:.2f})  "
                  f"GPU share {c['share']:.2f} = {c['gpu_gb']:.2f} GB")
    print("\n## K (us) at which the centre stack is 12.00, rows mode, kappa as modelled")
    for subj in bt.SUBJ:
        ks = [k for k in range(0, 600) if price(subj, k)["stack"] >= 12.0]
        print(f"  {subj}: {'K <= %d us' % max(ks) if ks else 'none (%.2f at K = 0)' % price(subj, 0)['stack']}")
    print("\n## stack against K (rows mode): " + ", ".join(bt.SUBJ))
    for k in (0, 20, 50, 100, 150, 200, 300, 452, 560):
        print(f"  K {k:3}: " + "  ".join(f"{price(s, k)['pg']:.2f} / {price(s, k)['stack']:.2f}" for s in bt.SUBJ))
