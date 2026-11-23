#!/usr/bin/env python3
"""memo-12 round 5 (Bonnie). Every output is *estimated*. Run: python3 -I -B bonnie_joins_r5.py
Loads bonnie_tree_r2.py by path (unchanged). Four parts:
  1  the together cell at two columns when the CPU arm is ggml's four-row tile (c = 2.3 .. 2.9 instead of 1.45),
     and the mixed arm (CPU one column, GPU two columns)
  2  a fixed row share against a share that follows the clock cap
  3  width 3 on the lane
"""
import importlib.util, os
_d = os.path.dirname(os.path.abspath(__file__))
_s = importlib.util.spec_from_file_location("t2", os.path.join(_d, "bonnie_tree_r2.py")); t2 = importlib.util.module_from_spec(_s); _s.loader.exec_module(t2)
bt, bj = t2.bt, t2.bj
CA, CB, GW, GD, BUS = t2.CA, t2.CB, t2.GW, t2.GD, t2.BUS
DEV_US, GAP_US = 896.0, 125.0          # 11008x4096 device time and launch gap in the `tog` state (R74, derived)
CG_GAPLESS = 18.9                      # CPU rate while the GPU kernel runs (Ebony r2 A.2, derived from gpu_busy 0.875)

# ---- 1. issue count of the tile ggml runs at ne1 = 2..4 (kai ..._qsi8d32p4x4_qsi4c32p4x4_16x4_neon_dotprod, m < 16 path, label_17/18)
TILE = dict(v0=2 * 2 + 4 * 2, v1=4, any=4 + 1 + 32 + 4 + 4 + 4)   # fcvtl x2, scvtf x4 (2 cycles each) | shl x4 | and x4, movi 0xf0, sdot x32, fmul x4, fmla x4, movi #0 x4
GEMV = dict(v0=6, v1=4, any=15)                                   # memo-10 E2 (Tom's count)
def floor(k): return max(k["v0"], k["v1"], sum(k.values()) / 2)

def cpu(c, Rb, C=40.0):                 # two-term reader with an explicit compute factor c
    x = c / C; return 1 / (max(1 / Rb, x) + 0.25 * min(1 / Rb, x))

def tog2(c, cg, variant, C=40.0):
    """(CPU alone, GPU alone, both CPU, both GPU), wall GB/s, CPU at compute factor c, GPU at time factor cg.
    GPU dead GAP_US per launch in every arm (the `tog` state). GPU device rate alone 28.32 / cg (R74).
    bonnie: the one-column contention factor (24.6 / 28.32) carries over, and the CPU keeps its one-column bus share.
    pair:   the GPU runs at min(24.6, 28.32 / cg); the CPU's bus share is BUS minus that."""
    busy = DEV_US * cg / (DEV_US * cg + GAP_US)
    ca = cpu(c, t2.bus_of(CA, C), C); ga = 28.32 / cg
    if variant == "bonnie": g = GD / cg; cb_run = cpu(c, t2.bus_of(CG_GAPLESS, C), C)
    else: g = min(GD, ga); cb_run = cpu(c, min(bt.R, BUS - g), C)
    return ca, ga * busy, busy * cb_run + (1 - busy) * ca, g * busy

# ---- 2. fixed share against a following share (one stage; W 2 with the W-position CPU kernel)
def stage_t(B, s, K, ca, cb, g, J0=0.0, J1=0.107):
    if s <= 0: return B / ca
    tg = K + s * B / g; w = (1 - s) * B
    if w <= ca * K: tc = w / ca
    else:
        tc = K + (w - ca * K) / cb
        if tc > tg: tc = tg + (w - ca * K - cb * (tg - K)) / ca
    return max(tg, tc) + J0 / 1e3 + J1 * K

def body_fixed(subj, K_us, variant, C_design, C_real, cg=1.25, c5=0.93, W=2):
    """Body ms when each stage's share is the best one at C_design but the clocks are at C_real. Returns (ms, GPU share)."""
    s = bt.SUBJ[subj]; K = K_us / 1e3; tot = sum(w for _, w in bj.STAGES[subj]); t = gb = 0.0
    ca1 = t2.cpu_rate(1, t2.bus_of(CA, 40.0), 40.0)                        # body bytes are fixed by the R72 body at W 1
    rd, rr = t2.rates(W, variant, cg, C_design), t2.rates(W, variant, cg, C_real)
    for _, w in bj.STAGES[subj]:
        B = s["body"] * c5 * w / tot / bj.NL[subj] * ca1
        sh = min((stage_t(B, i / 200, K, *rd), i / 200) for i in range(0, 161))[1]
        t += stage_t(B, sh, K, *rr); gb += sh * w
    return t * bj.NL[subj], gb / tot

def toks_fixed(subj, K, variant, Cd, Cr, W=2, **kw):
    s = bt.SUBJ[subj]; b = body_fixed(subj, K, variant, Cd, Cr, W=W, **kw)[0]
    return 950 / ((b + 5.2 + s["guess"] + s["xpos"]) / (1 + s["d"] * s["a1"]) + s["head"] * 0.5)

# ---- 3. width 3 on the lane (tree of two candidate nodes, best shape; the CPU needs the W-position kernel: c(3) = 1.9)
def toks_w(subj, W, K, variant, cg, C=40.0):
    s = bt.SUBJ[subj]; b = t2.body(subj, W, K, variant, cg=cg, C=C)[0]; best = 1e9
    for shape in ("topk", "chain", "two", "greedy"):
        a, dep, _ = bt.tree(s["a1"], W - 1, shape)
        best = min(best, (b + 5.2 + s["guess"] * dep + s["xpos"] * (W - 1)) / (1 + s["d"] * a) + s["head"] * 0.5)
    return 950 / best

# ---- 4. a third variant between the two: the CPU's bus share falls linearly with the GPU's draw.
# Anchors (*device*): R69 / R72 x1gpu: CPU q4x4 on 4-6 reads 30.0 to 30.35 beside a GPU drawing 1.8 GB/s (alone 30.09) and
# 20.7 to 21.2 beside a GPU reading 25.8 to 26.2; R74 tog: the gapless CPU rate 18.9 beside the gemv at 24.6.
def rates_linear(W, variant, cg=1.45, C=40.0, gdev=False):
    cgw = 1.0 if W == 1 else cg; g = GD / cgw; ra = t2.bus_of(CA, C)
    return t2.cpu_rate(W, t2.bus_of(CA), C), t2.cpu_rate(W, ra - (ra - (BUS - GD)) * g / GD, C), g
def toks_linear(subj, K, cg, C=40.0):
    keep = t2.rates; t2.rates = rates_linear
    try: return t2.toks(subj, 2, K, "pair", cg=cg, C=C)
    finally: t2.rates = keep

def selfcheck():
    t2.selfcheck()
    assert floor(GEMV) == 12.5 and floor(TILE) == 32.5
    ca, ga, cb, g = tog2(1.0, 1.0, "bonnie")                                # one column reproduces R74's togsum
    assert abs(ca - 25.64) < 0.01 and abs(cb - 19.77) < 0.15 and abs(g - 21.53) < 0.1, (ca, cb, g)
    for subj in t2.SUBJ:                                                     # a following share reproduces bonnie_tree_r2 within the grid
        for v in ("bonnie", "pair"):
            assert abs(toks_fixed(subj, 300, v, 40, 40) - t2.toks(subj, 2, 300, v, cg=1.25)) < 0.06, (subj, v)
            assert abs(toks_w(subj, 2, 300, v, 1.25) - t2.toks(subj, 2, 300, v, cg=1.25)) < 0.02

if __name__ == "__main__":
    selfcheck()
    print("1. tile slots %d, floor %.1f cycles per 128 weights; gemv %d, %.1f; ratio %.2f; device-scaled (gemv 15.1 measured): additive %.2f, proportional %.2f" % (
        sum(TILE.values()), floor(TILE), sum(GEMV.values()), floor(GEMV), floor(TILE) / floor(GEMV), (floor(TILE) + 2.6) / 15.1, floor(TILE) / floor(GEMV)))
    print("   together cell at two columns, wall GB/s: CPU alone / GPU alone / both CPU / both GPU / sum | GPU ratio | sum ratio | CPU ratio")
    for c in (1.45, 2.3, 2.5, 2.9):
        for v, cg in (("bonnie", 1.2), ("bonnie", 1.45), ("pair", 1.26)):
            ca, ga, cb, g = tog2(c, cg, v)
            print(f"   c {c:4} {v:6} cg {cg:4}: {ca:5.1f} / {ga:5.1f} / {cb:5.1f} / {g:5.1f} / {cb + g:5.1f} | {g / ga:.3f} | {(cb + g) / (ca + ga):.3f} | {cb / ca:.3f}")
    print("   mixed arm, CPU one column beside the GPU at n_cols: both CPU wall GB/s (bonnie / pair), ratio to the one-column cell")
    b1 = {v: tog2(1.0, 1.0, v)[2] for v in ("bonnie", "pair")}
    for n, cg in ((2, 1.25), (3, 1.6), (4, 1.95)):
        r = {v: tog2(1.0, cg, v)[2] for v in ("bonnie", "pair")}
        print(f"   n_cols {n} cg {cg}: {r['bonnie']:.1f} / {r['pair']:.1f}   ratio {r['bonnie'] / b1['bonnie']:.3f} / {r['pair'] / b1['pair']:.3f}")
    print("\n2. share fixed at its C_design optimum, clocks at C_real (K 300, cg 1.25, W 2): tok/s Yi / Llama-2 / DeepSeek, GPU share (Llama-2)")
    for v in ("bonnie", "pair"):
        for Cd in (40.0, 35.0, 31.0):
            for Cr in (40.0, 35.0, 31.0):
                print(f"   {v:6} design C {Cd:.0f} real C {Cr:.0f}: " + " / ".join(f"{toks_fixed(s, 300, v, Cd, Cr):.2f}" for s in t2.SUBJ) + f"   share {body_fixed('Llama-2', 300, v, Cd, Cr)[1]:.3f}")
    print("\n3. width 3 against width 2 on the lane (W 2 at cg 1.25): tok/s Yi / Llama-2 / DeepSeek")
    for v in ("bonnie", "pair"):
        for K in (0, 150, 300):
            print(f"   {v:6} K {K:3}: W 2 " + " / ".join(f"{toks_w(s, 2, K, v, 1.25):.2f}" for s in t2.SUBJ)
                  + " | W 3 at cg(3) " + "; ".join(f"{cg}: " + " / ".join(f"{toks_w(s, 3, K, v, cg):.2f}" for s in t2.SUBJ) for cg in (1.3, 1.6, 2.0)))
        for K in (0, 150, 300):
            print(f"   {v:6} K {K:3}: cg(3) below which W 3 beats W 2: " + " / ".join(
                (lambda x: "none" if x is None else f"{x:.2f}")(t2.solve(lambda cg, s=s: toks_w(s, 3, K, v, cg) - toks_w(s, 2, K, v, 1.25), 1.0, 4.0, 0.0)) for s in t2.SUBJ))
    print("\n4. stack tok/s at W 2, alone cg 1.45 (g 19.5 beside the CPU), C 40: bonnie (wall, cg 1.26) | linear | pair")
    for K in (0, 150, 300, 452):
        print(f"   K {K:3}: " + " / ".join(f"{t2.toks(s, 2, K, 'bonnie', cg=1.26):.2f}" for s in t2.SUBJ) + " | " + " / ".join(f"{toks_linear(s, K, GD / 19.5):.2f}" for s in t2.SUBJ)
              + " | " + " / ".join(f"{t2.toks(s, 2, K, 'pair', cg=GD / 19.5):.2f}" for s in t2.SUBJ))
    print("   CPU rate beside the GPU at W 2 (c 1.45): linear %.1f, pair %.1f, bonnie %.1f GB/s" % (rates_linear(2, "", GD / 19.5)[1], t2.rates(2, "pair", GD / 19.5)[1], t2.rates(2, "bonnie", 1.26)[1]))
    print("   K for 12.00, linear: " + " / ".join((lambda v: "none" if v is None else f"{v:.0f}")(t2.solve(lambda k, s=s: toks_linear(s, k, GD / 19.5), 0, 600, 12.0)) for s in t2.SUBJ))
