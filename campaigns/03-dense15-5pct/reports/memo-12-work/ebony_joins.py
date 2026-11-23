"""memo-12 (Ebony): launch-level price of the GPU lane. All outputs are *estimated* from R74 *device* inputs.
Model per dependency level of a layer (B bytes, CPU and GPU both reading):
    T_split = (B + r_g * lam) / (r_c + r_g)      lam = GPU dead time per level (launch latency + gaps + wake)
    T_level = min(B / c_alone, T_split)          a level is left on the CPU when the split does not pay
Run: python3 -I ebony_joins.py
"""
C_ALONE, C_BOTH, G_BOTH = 25.64, 19.77, 21.53          # GB/s, x3 togsum (device)
MB = lambda rows, k: rows * k * 18 / 32 / 1e6            # Q4_0 bytes
SUBJ = {  # name: (kv rows, layers, K0 body ms, head ms, measured head top-1)   R72/R74 (device), tom-head (host)
    "Yi": (512, 32, 120.6, 9.04, 0.379), "Llama-2": (4096, 32, 140.6, 4.40, 0.497), "DeepSeek": (4096, 30, 130.8, 13.88, 0.438)}
O = 8.0

def levels(kv, stacked=True):
    q, kvb, o, f = MB(4096, 4096), MB(kv, 4096), MB(4096, 4096), MB(11008, 4096)
    if stacked: return [q + 2 * kvb, o, 2 * f, f]            # {q,k,v} {o} {gate,up} {down}: one launch each
    return [q, kvb, kvb, o, f, f, f]                          # one launch per matrix (R74 "7 joins" row)

def t_split(B, lam): return (B / 1e3 + G_BOTH * lam * 1e-3) / (C_BOTH + G_BOTH) * 1e3   # ms; B in MB, lam in ms
def t_alone(B): return B / C_ALONE                                                        # ms

def factor(kv, lam, stacked=True):
    lv = levels(kv, stacked); n = 0; t = 0.0
    for B in lv:
        a, s = t_alone(B), t_split(B, lam)
        if s < a: t += s; n += 1
        else: t += a
    return t / sum(t_alone(B) for B in levels(kv)), n

def toks(name, lam, stacked=True, per_token_ms=0.0):
    kv, L, body, head, _ = SUBJ[name]; f, n = factor(kv, lam, stacked)
    return 950 / (body * f + head + O + per_token_ms), f, n * L

if __name__ == "__main__":
    # self-checks
    assert abs(factor(4096, 0.0)[0] - C_ALONE / (C_BOTH + G_BOTH)) < 1e-9          # lam 0, all levels split = 1/1.611
    s = t_split(MB(11008, 4096), 0.452); assert 0.80 < s < 0.90, s                  # x3 split measured 0.865 ms (share 0.25)
    print(f"check: one 11008x4096 matrix split at lam 0.452 ms -> {s:.3f} ms (x3 split measured 0.865, device)")
    print(f"dead-time weight r_g/(r_c+r_g) = {G_BOTH / (C_BOTH + G_BOTH):.3f}")
    for lab, lam, st, pt in [("R74 per-matrix launch, lam 0.452", 0.452, False, 0), ("stacked levels, lam 0.452", 0.452, True, 0),
                             ("stacked, lam 0.30 (queued-launch level)", 0.30, True, 0), ("stacked, lam 0.12 (best queued cell, tog)", 0.12, True, 0),
                             ("stacked, lam 0.05", 0.05, True, 0), ("persistent kernel, lam 0.02 + 0.45 ms per token", 0.02, True, 0.45),
                             ("lam 0", 0.0, True, 0)]:
        r = [toks(n, lam, st, pt) for n in SUBJ]
        print(f"{lab:48s} " + " / ".join(f"{t:.2f}" for t, _, _ in r) + "   body x " + " / ".join(f"{f:.3f}" for _, f, _ in r)
              + "   launches/token " + " / ".join(str(n) for _, _, n in r))
    # structural alternatives (section 4)
    print("GPU runs the guessed token's whole pass beside the CPU (one join per token):")
    for n, (kv, L, body, head, p) in SUBJ.items():
        gpu = {"Yi": 182.7, "Llama-2": 190.8, "DeepSeek": 181.0}[n] * 24.88 / G_BOTH       # x3 layer sync1 (device), under contention
        t = max(body * C_ALONE / C_BOTH, gpu) + head + O
        print(f"  {n}: pass {t:.0f} ms, {950 * (1 + p) / t:.2f} tok/s at top-1 {p}   (K0 {950 / (body + head + O):.2f})")
    print("speculative layer pipeline (CPU lower half, GPU upper half, exit-head guess at mid depth, launches free):")
    for n, (kv, L, body, head, p) in SUBJ.items():
        dev = {"Yi": 117.05, "Llama-2": 134.34, "DeepSeek": 126.18}[n]                      # x3 layersum body_ms_dev (device)
        c, g = body / 2, dev / 2
        for a in (0.5, 0.7, 0.9):
            t = a * max(c * C_ALONE / C_BOTH, g * 24.88 / G_BOTH) + (1 - a) * (c + g) + head + O
            print(f"  {n} a={a}: {950 / t:.2f}", end="")
        print()
