#!/usr/bin/env python3
"""memo-12 r3 (Bonnie): x4m and x4s (0.20..0.60) bands, cg lines. *estimated*. Run: python3 -I bonnie_joins_r3.py"""
import importlib.util, os
_d = os.path.dirname(os.path.abspath(__file__))
def _load(n):
    s = importlib.util.spec_from_file_location(n, os.path.join(_d, n + ".py")); m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
t2 = _load("bonnie_tree_r2"); r2 = _load("bonnie_joins_r2"); SUBJ = t2.SUBJ

# GPU two-term model: time per byte = max(b, c*u) + 0.25*min(b, c*u); b = 1/bus share, u = compute time per byte at one column
def gpu_t(b, u, c): return max(b, c * u) + 0.25 * min(b, c * u)
B_ALONE, B_LANE = 1 / 31.14, 1 / 25.81                      # GPU read-only rate alone / beside the CPU (R72, device)
def u_from_alone(dev_rate=28.32): return (1 / dev_rate - B_ALONE) / 0.25      # valid while bus-bound alone
def colcost(n, per=0.9): return 1 + per * (n - 1)            # _mc3: weight loads shared, dequantise + accumulate per column
def alone_ratio(n, u, per=0.9): return gpu_t(B_ALONE, u, colcost(n, per)) / gpu_t(B_ALONE, u, 1)
def lane_cg(n, u, per=0.9): return gpu_t(B_LANE, u, colcost(n, per)) / gpu_t(B_LANE, u, 1)

if __name__ == "__main__":
    t2.selfcheck(); r2.check_split()
    print("## x4m: dev_ratio alone (n_cols 2 / 3 / 4) and the lane's cg at 2 columns, by GPU compute time per byte u")
    for lab, u in (("u from 28.32 against 31.14 (two-term centre)", u_from_alone()), ("u x 2", 2 * u_from_alone()), ("u x 3", 3 * u_from_alone()), ("compute-bound at 28.3 alone", 1 / 28.32 / 1.0)):
        print(f"  {lab:46}: alone {alone_ratio(2, u):.2f} / {alone_ratio(3, u):.2f} / {alone_ratio(4, u):.2f}   lane cg(2) {lane_cg(2, u):.2f}  cg(3) {lane_cg(3, u):.2f}")
    print("  wall_ratio at dev 896 us, gap 311 / 125: " + ", ".join(f"dev_ratio {r}: {(896 * r + 311) / 1207:.2f} / {(896 * r + 125) / 1021:.2f}" for r in (1.08, 1.2, 1.45, 1.9)))

    print("\n## cg lines (width-2 pass on the lane; trained head, C7, C5, O 5.2 at centre)")
    for var, kw in (("bonnie", {}), ("pair", {}), ("pair, slack 60", dict(J0=60, J1=0)), ("pair, C 35", dict(C=35.0))):
        v = var.split(",")[0]
        for K in (0, 150, 300, 452):
            need = [t2.solve(lambda cg, s=s: t2.toks(s, 2, K, v, cg=cg, **kw), 1.0, 4.0, 12.0) for s in SUBJ]
            w1 = [t2.solve(lambda cg, s=s: t2.toks(s, 2, K, v, cg=cg, **kw) - t2.toks(s, 1, K, v, **{k: x for k, x in kw.items()}), 1.0, 6.0, 0.0) for s in SUBJ]
            f = lambda xs: " / ".join("none" if x is None else f"{x:.2f}" for x in xs)
            print(f"  {var:15} K {K:3}: cg for 12.00 <= {f(need)};  width 2 = width 1 at cg {f(w1)}")
    print("  stack at cg 1.0 / 1.1 / 1.2 / 1.3 / 1.45 / 1.7 / 2.0, K 300:")
    for v in ("bonnie", "pair"):
        print(f"    {v}: " + " | ".join(" / ".join(f"{t2.toks(s, 2, 300, v, cg=cg):.2f}" for s in SUBJ) for cg in (1.0, 1.1, 1.2, 1.3, 1.45, 1.7, 2.0)))

    print("\n## x4s lead mode, shares 0.20..0.60: ratio (slack 0.03 ms added), c = CPU side binds, g = GPU side binds")
    best = {}
    for K, Lh in ((500, 65), (380, 65), (300, 65), (200, 50), (120, 50), (30, 20)):
        cells = []
        for f in (0.20, 0.30, 0.40, 0.45, 0.50, 0.55, 0.60):
            tc, tg, t = r2.split(1.896, 22016, f, K, Lh); cells.append((round((t + 0.03) / 1.896, 3), f, "g" if tg > tc else "c"))
        best[K] = min(cells)
        print(f"  K {K:3} Lh {Lh}: " + "  ".join(f"{f:.2f}:{r:.3f}{w}" for r, f, w in cells) + f"   best {best[K][1]:.2f} -> {best[K][0]:.3f}")
    out, W = r2.mc()
    import collections
    res = []
    for w in W[:4000]:
        c = w["cell"][("M", "spin")]; K = c["host"] + c["s2s"] + c["end"] + 47; Lh = c["host"]
        res.append(min(((r2.split(1.896, 22016, f / 100, K, Lh)[2] + 0.03) / 1.896, f / 100) for f in range(20, 61, 5)))
    rs = sorted(r for r, _ in res); n = len(rs)
    print(f"  Monte Carlo (x4b worlds): best ratio {rs[n // 2]:.3f} ({rs[n // 10]:.3f} to {rs[n * 9 // 10]:.3f}); best share: " +
          ", ".join(f"{k:.2f}: {v / n:.2f}" for k, v in sorted(collections.Counter(f for _, f in res).items())))
    for f in (0.55, 0.60):
        xs = []
        for w in W[:4000]:
            c = w["cell"][("M", "spin")]; K = c["host"] + c["s2s"] + c["end"] + 47
            tc, tg, t = r2.split(1.896, 22016, f, K, c["host"]); xs.append(((t + 0.03) / 1.896, tg, max(0, tg - tc) + 0.03, tc))
        col = lambda i: sorted(x[i] for x in xs)
        print(f"  share {f}: ratio {col(0)[n // 2]:.3f} ({col(0)[n // 10]:.3f} to {col(0)[n * 9 // 10]:.3f}); gpu_ms {col(1)[n // 2]:.2f} ({col(1)[n // 10]:.2f} to {col(1)[n * 9 // 10]:.2f}); "
              f"wait_ms {col(2)[n // 2]:.2f} ({col(2)[n // 10]:.2f} to {col(2)[n * 9 // 10]:.2f}); cpu_ms {col(3)[n // 2]:.2f}; gpu_dev_ms {1.1 * (14 + 0.0801 * f * 22016) / 1e3:.2f}")

    print("\n## GPU share of the body bytes at balance (GB of GPU buffers), W 2")
    for v, cg in (("bonnie", 1.45), ("bonnie", 1.2), ("pair", 1.26), ("pair", 1.08)):
        for K in (0, 300):
            sh = [t2.body(s, 2, K, v, cg=cg)[2] for s in SUBJ]
            print(f"  {v} cg {cg} K {K}: share " + " / ".join(f"{x:.2f}" for x in sh) + "  GB " + " / ".join(f"{x * b:.2f}" for x, b in zip(sh, (3.11, 3.64, 3.42))))
    print("  cost of a GPU-buffer cap (pair, cg 1.26, K 300; CPU-bound stage time ~ (1 - f) / (1 - f*)):")
    for cap in (1.5, 1.2, 0.85):
        row = []
        for s, b in zip(SUBJ, (3.11, 3.64, 3.42)):
            fs = t2.body(s, 2, 300, "pair", cg=t2.GD / 19.5)[2]; f = min(fs, cap / b); bd = t2.body(s, 2, 300, "pair", cg=t2.GD / 19.5)[0]
            S = t2.bt.SUBJ[s]; tau = 1 + S["d"] * S["a1"]; m = (bd * (1 - f) / (1 - fs) + 5.2 + S["guess"] + S["xpos"]) / tau + S["head"] * 0.5
            row.append(f"{950 / m:.2f}")
        print(f"    cap {cap} GB: " + " / ".join(row) + "   (no cap: " + " / ".join(f"{t2.toks(s, 2, 300, 'pair', cg=t2.GD / 19.5):.2f}" for s in SUBJ) + ")")
