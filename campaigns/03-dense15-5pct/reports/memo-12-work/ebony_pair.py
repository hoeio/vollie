#!/usr/bin/env python3
"""memo-12 round 2 (Ebony): how width (W positions per weight read) composes with the CPU + GPU lane.
Every output is *estimated*. Run: python3 -I ebony_pair.py   (loads memo-10-work/bonnie_tree.py by path, unchanged)

memo-10's stack uses  body * lane(W) * kB(W):  kB(W) is the CPU-ALONE slow-down (bus R = 29.9 GB/s) and
lane(W) slows the GPU by c(W) on top. Here the two processors share one bus instead:
    CPU gemv rate   r_c = 1 / (max(1/Rc, c/C) + 0.25 * min(1/Rc, c/C))      memo-09 section 10.3's fit, Rc = its bus share
    GPU rate        r_g = min(G_BUS, GC / cg(W))                              device-time rate; GC = its compute ceiling
    bus             Rc  = min(R, S - r_g)                                     S calibrated on x3 tog (W = 1)
A stage of B bytes split with GPU dead time K:  T = B / (r_c + r_g) + kappa * K (+ jitter);  left on the CPU if cheaper.
Inputs (*device*): R 29.9 (R72), x3 tog 25.64 / 19.77 / 21.53 GB/s and gpu_busy 0.875 (R74), dev_gbs alone 28.3 (R74).
"""
import importlib.util, os
_p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "memo-10-work", "bonnie_tree.py")
_s = importlib.util.spec_from_file_location("bonnie_tree", _p); bt = importlib.util.module_from_spec(_s); _s.loader.exec_module(bt)

R, BUSY = 29.9, 0.875
G_BUS = 21.53 / BUSY                                   # 24.6: GPU device-time rate beside the CPU
C_GAPLESS = (19.77 - (1 - BUSY) * 25.64) / BUSY        # 18.9: CPU rate while the GPU kernel runs
LANE1 = 25.64 / (C_GAPLESS + G_BUS)                   # 0.589: W = 1 lane factor from measured rates only
NL = {"Yi": 32, "Llama-2": 32, "DeepSeek": 30}
M44, M5, M11 = 4096 * 4096, 512 * 4096, 11008 * 4096
STAGES = {s: [3 * M44, M44, 2 * M11, M11] for s in ("Llama-2", "DeepSeek")}; STAGES["Yi"] = [M44 + 2 * M5, M44, 2 * M11, M11]

def rate(Rc, c, C): return 1 / (max(1 / Rc, c / C) + 0.25 * min(1 / Rc, c / C))
def S(C): return 1 / (1 / C_GAPLESS - 0.25 / C) + G_BUS            # bus sum that reproduces the W = 1 cell (46.1 at C 40)

def pair(W, C=40.0, GC=28.3, cg=None):
    """(CPU alone, CPU beside GPU, GPU) byte rates at width W. cg: GPU cost factor (default c(W); W = repeated launches)."""
    c = bt.c(W); cg = c if cg is None else cg
    rg = min(G_BUS, GC / cg); rc = rate(min(R, S(C) - rg), c, C)
    return rate(R, c, C), rc, rg

def body(subj, W, K_us, C=40.0, GC=28.3, cg=None, kappa=0.45, jit_us=0.0, hers=False):
    """Body ms per pass with the lane, and joins per layer. Split stage: A1 * LANE1 * pair kB(W) + kappa * K + jitter."""
    s = bt.SUBJ[subj]; tot = sum(STAGES[subj]); s1 = sum(pair(1, C, GC)[1:]); sw = sum(pair(W, C, GC, cg)[1:])
    t = j = 0.0
    for w in STAGES[subj]:
        A1 = s["body"] * w / tot / NL[subj]                        # CPU alone, W = 1 (R72 body by weight share)
        alone = A1 * bt.kB(W, C)
        split = (A1 * LANE1 * s1 / sw if not hers else A1 * bt.kB(W, C) * 26 / (26 + 0.64 * 24.82 / bt.c(W))) + (kappa * K_us + jit_us) / 1e3
        if split < alone: t += split; j += 1
        else: t += alone
    return t * NL[subj], int(j)

def stack(subj, W, K_us, **kw):
    s = bt.SUBJ[subj]; b, j = body(subj, W, K_us, **kw); out = []
    for shape in ("topk", "chain", "two", "greedy"):
        a, dep, _ = bt.tree(s["a1"], W - 1, shape); tau = 1 + s["d"] * a
        out.append((b * 0.93 + 5.2 + s["guess"] * dep + s["xpos"] * (W - 1)) / tau + s["head"] * 0.5)
    return 950 / min(out), j

def selfcheck():
    bt.selfcheck()
    ra, rc, rg = pair(1)
    assert abs(rc - C_GAPLESS) < 1e-6 and abs(rg - 24.6) < 0.05 and abs(S(40) - 46.1) < 0.1, (rc, rg, S(40))   # W = 1 cell, by construction
    assert abs(ra - 25.64) < 0.5                                                   # the fit's CPU-alone rate: 25.2 against 25.64 (device)
    assert abs(rc * BUSY + 25.64 * (1 - BUSY) - 19.77) < 1e-6
    for subj, st0 in (("Yi", 12.23), ("Llama-2", 11.87), ("DeepSeek", 11.44)):     # her composition, K 0: R74-result section 4
        assert abs(stack(subj, 2, 0, hers=True)[0] - st0) < 0.03, (subj, stack(subj, 2, 0, hers=True))

if __name__ == "__main__":
    selfcheck()
    print(f"G_BUS {G_BUS:.2f}  CPU beside a gapless GPU {C_GAPLESS:.2f}  S(40) {S(40):.1f} GB/s (R72 read sum 46.5, R69 47.4, device)")
    print("\npair rates (CPU alone / CPU beside / GPU / sum) and pair kB = sum(1) / sum(W):")
    for lab, kw in (("C 40, GC 28.3, W-column GPU kernel", {}), ("C 40, GC 24.6 (GPU compute-bound already)", dict(GC=24.6)),
                    ("C 40, GC 40", dict(GC=40.0)), ("C 35, GC 28.3", dict(C=35.0)), ("C 24, GC 28.3", dict(C=24.0))):
        s1 = sum(pair(1, **kw)[1:])
        print(f"  {lab}")
        for W in (1, 2, 3, 4):
            ra, rc, rg = pair(W, **kw); print(f"     W {W}: {ra:5.1f} / {rc:5.1f} / {rg:5.1f} / {rc + rg:5.1f}   pair kB {s1 / (rc + rg):.3f}   CPU-alone kB {bt.kB(W, kw.get('C', 40.0)):.3f}")
    ra, rc, rg = pair(2, cg=2.0); print(f"  C 40, GPU = two launches of the one-column kernel: W 2 sum {rc + rg:.1f}, pair kB {sum(pair(1)[1:]) / (rc + rg):.3f}")
    print("\ncentre stack tok/s (trained head, C7, C5, O 5.2), Yi / Llama-2 / DeepSeek; [joins per layer]")
    rows = [("memo-10 composition (bonnie_joins Z row)", dict(hers=True, kappa=0.521)),
            ("pair model, C 40, GC 28.3", {}), ("pair model, C 40, GC 24.6", dict(GC=24.6)), ("pair model, C 35", dict(C=35.0)),
            ("pair model, C 24 (caps, X1 thread on a small core)", dict(C=24.0)), ("pair model, C 40, GC 40", dict(GC=40.0)),
            ("pair model, C 40, GPU two launches (cg 2)", dict(cg=2.0)), ("pair model, C 40, jitter 60 us per split stage", dict(jit_us=60.0))]
    for lab, kw in rows:
        print(f"  {lab}")
        for K in (0, 50, 120, 220, 300, 452):
            best = []
            for subj in bt.SUBJ:
                c = max((stack(subj, W, K, **kw) + (W,)) for W in ((2,) if kw.get("hers") else (1, 2, 3, 4)))
                best.append(f"{c[0]:.2f} (W{c[2]}) [{c[1]}]")
            print(f"     K {K:3}: " + "  /  ".join(best))
    print("\nK (us) at which the stack is 12.00 on the slowest subject:")
    for lab, kw in rows[1:]:
        ks = [K for K in range(0, 700, 2) if min(max(stack(s, W, K, **kw)[0] for W in (1, 2, 3, 4)) for s in bt.SUBJ) >= 12.0]
        print(f"  {lab}: {'K <= %d' % max(ks) if ks else 'none'}")
