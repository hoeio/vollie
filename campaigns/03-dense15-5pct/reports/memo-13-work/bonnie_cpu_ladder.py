#!/usr/bin/env python3
"""memo-13 (Bonnie): the CPU-only ladder, no lane rung. Every output is *estimated* arithmetic on quoted inputs.
Loads memo-10-work/bonnie_tree.py unchanged (R, the two-term reader, SUBJ). Run: python3 -I -B bonnie_cpu_ladder.py

  body(c, C) = body_R72 * t(c, C) / t(1, C),  t(c, C) = max(1/R, c/C) + A * min(1/R, c/C)   (A 0.25 committed, 0.17 r32 refit)
  c = compute per weight against the KleidiAI gemv at one position:
      1.00 packed gemv (device base)      1.45 packed W-position kernel at two positions (memo-10 E2: count 1.44, M4 1.42)
      2.20 plain rows, stock dot (r6)     1.45 Y1b in place, one position (memo-05: M4 1.47 to 1.51, count 35/25 = 1.40)
      2.10 Y1b-style in-place kernel at two positions (count, this memo: 18 shared + 17 per position, over 25)
  ms per token = [(1 - s2) * (body1 + O) + s2 * (body2 + O + xpos) + sh * guess] / tau + head;  printed = 950 / ms
"""
import importlib.util, os
_d = os.path.dirname(os.path.abspath(__file__))
_s = importlib.util.spec_from_file_location("bt", os.path.join(_d, "..", "memo-10-work", "bonnie_tree.py")); bt = importlib.util.module_from_spec(_s); _s.loader.exec_module(bt)
R, S = bt.R, list(bt.SUBJ)
BODY = {s: bt.SUBJ[s]["body"] for s in S}
HEAD = {"kai": (9.04, 4.40, 13.88), "repack": (7.7, 3.7, 11.8), "two": (7.7 * .80, 3.7 * .91, 11.8 * .76), "plain": (15.0, 7.5, 24.0)}
O15, XPOS, GUESS = (5.0, 7.0, 7.0), (3.2, 5.3, 5.0), (2.64, 2.61, 2.59)          # r15; r15; tom-head.md section 5
DS = "-0 L1 L2 x3 -0 -0 x1 -0 -0 -0 x1 x1 -0 x1 -0 L1 L2 L3 L3 L3 L3 L3 L3 L3 x3 -0 x1 -0 -0 -0 x1".split()  # tom-et23.md, R reply

def t(c, C, A=0.25): return max(1 / R, c / C) + A * min(1 / R, c / C)
def f(c, C, A=0.25): return t(c, C, A) / t(1.0, C, A)

def lookup(steps, minlen, n=32):
    """Passes, width-2 passes and hits of a one-guess lookup over the step string (guess i is for token i + 1)."""
    tkn = passes = w2 = hits = 0
    while tkn < n:
        passes += 1
        e = steps[tkn] if tkn < len(steps) else "-0"
        if e[0] in "Lx" and int(e[1]) >= minlen:
            w2 += 1
            if e[0] == "L": hits += 1; tkn += 1
        tkn += 1
    return passes, w2, hits

def ms(i, c1, c2, C, head, O, tau=1.0, s2=0.0, sh=0.0, A=0.25, bs=1.0):
    s = S[i]; b1, b2 = BODY[s] * bs * f(c1, C, A), BODY[s] * bs * f(c2, C, A)
    return ((1 - s2) * (b1 + O) + s2 * (b2 + O + XPOS[i]) + sh * GUESS[i]) / tau + head
def row(lab, fn): print(f"  {lab:74} " + " / ".join(f"{950 / fn(i):5.2f}" for i in range(3)))

# guess sources on the official prompt (*host*): Yi rule T 1.455, Llama-2 rule T 1.333 (r24); DeepSeek lookup alone (E-T23)
P_any, P_2, P_3 = lookup(DS, 1), lookup(DS, 2), lookup(DS, 3)
TAU = lambda src: [(1.455, 1.0, 0.85), (1.333, 1.0, 0.85), src]      # (tau, share of width-2 passes, share with a head evaluation)
DS2 = (32 / P_2[0], P_2[1] / P_2[0], 0.0); DSA = (32 / P_any[0], P_any[1] / P_any[0], 0.0)
SET = [(1.391, 1.0, 0.85), (1.486, 1.0, 0.85), (1.127, 0.30, 0.0)]  # fixed set: rule T (r24); DeepSeek lookup >= 2 (E-T23; share assumed)

def ladder(C, A=0.25):
    print(f"\n## C {C} GB/s, coupling {A}: tok/s Yi / Llama-2 / DeepSeek")
    print(" packed layout (choice B or C; the layout R72 priced)")
    row("P0 stock: KleidiAI body, head, O 8", lambda i: ms(i, 1, 1, C, HEAD["kai"][i], 8, A=A))
    row("P1 + head in the repack type", lambda i: ms(i, 1, 1, C, HEAD["repack"][i], 8, A=A))
    row("P2 + O as r15 (5 / 7 / 7)", lambda i: ms(i, 1, 1, C, HEAD["repack"][i], O15[i], A=A))
    row("P3 + width 2, W-position kernel; rule T, DeepSeek lookup >= 2 only", lambda i: ms(i, 1, 1.45, C, HEAD["repack"][i], O15[i], *TAU(DS2)[i], A=A))
    row("P3s  same guess rules scored on the fixed set", lambda i: ms(i, 1, 1.45, C, HEAD["repack"][i], O15[i], *SET[i], A=A))
    row("P4 + campaign 2's two-stage head", lambda i: ms(i, 1, 1.45, C, HEAD["two"][i], O15[i], *TAU(DS2)[i], A=A))
    row("P4d what-if: DeepSeek head at tau 1.30 / 1.45 (cols 1-2 = P4)", lambda i: ms(i, 1, 1.45, C, HEAD["two"][i], O15[i], *(TAU((1.30, 1, .85))[i] if i < 2 else (1.30, 1, .85)), A=A))
    row("P4e   ... at 1.45", lambda i: ms(i, 1, 1.45, C, HEAD["two"][i], O15[i], *(TAU((1.45, 1, .85))[i]), A=A))
    row("P5 what-if: + C5 x 0.93 on the body (never screened)", lambda i: ms(i, 1, 1.45, C, HEAD["two"][i], O15[i], *TAU(DS2)[i], A=A, bs=.93))
    print(" plain mapped rows (choice A, the owner's path)")
    row("A0 stock: plain dot products (c 2.2), plain head, O 8", lambda i: ms(i, 2.2, 2.2, C, HEAD["plain"][i], 8, A=A))
    row("A1 + Y1b in place (c 1.45)", lambda i: ms(i, 1.45, 1.45, C, HEAD["plain"][i], 8, A=A))
    row("A1lo  Y1b at c 1.25 (no MOVI)", lambda i: ms(i, 1.25, 1.25, C, HEAD["plain"][i], 8, A=A))
    row("A1hi  Y1b at c 1.45 and down (24 % of body) cold at 2.1", lambda i: ms(i, 1.45 + .24 * .65, 0, C, HEAD["plain"][i], 8, A=A))
    row("A2 + head-only copy in the repack type", lambda i: ms(i, 1.45, 1.45, C, HEAD["repack"][i], 8, A=A))
    row("A3 + O as r15", lambda i: ms(i, 1.45, 1.45, C, HEAD["repack"][i], O15[i], A=A))
    row("A4 + width 2 in place (c 2.1); rule T, DeepSeek lookup >= 2", lambda i: ms(i, 1.45, 2.1, C, HEAD["repack"][i], O15[i], *TAU(DS2)[i], A=A))
    print(f"   pass ratio P2/P1 (Llama-2): packed {(BODY['Llama-2'] * f(1.45, C, A) + 7 + 5.3) / (BODY['Llama-2'] + 7):.2f}, in place {(BODY['Llama-2'] * f(2.1, C, A) + 7 + 5.3) / (BODY['Llama-2'] * f(1.45, C, A) + 7):.2f}")

# ---- the cost model itself (memo section 5): which term bounds, and what a change of scheme must deliver
GB = (3.11, 3.64, 3.42)                                   # body bytes per token, *device* R72 section 5
def g(b, k, C, A=0.25):
    """Body factor at byte share b and compute share k (both against stock, one position)."""
    x, y = b / R, k / C
    return (max(x, y) + A * min(x, y)) / t(1.0, C, A)
def solve(fn, lo, hi, target=79.17):
    """Largest x in [lo, hi] with fn(x) <= target (fn rising); None if fn(lo) > target."""
    if fn(lo) > target: return None
    for _ in range(50):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if fn(mid) <= target else (lo, mid)
    return lo
def model(C=50):
    fm = lambda v: "none" if v is None else f"{v:.2f}"
    print(f"\n## cost model, C {C}: Yi / Llama-2 / DeepSeek")
    print("  bus term bytes / R, ms:            " + " / ".join(f"{1000 * GB[i] / R:6.1f}" for i in range(3)))
    print(f"  compute term bytes / C at k 1, ms: " + " / ".join(f"{1000 * GB[i] / C:6.1f}" for i in range(3)))
    print("  body (device), head (device), O (r15): " + " / ".join(f"{BODY[S[i]]:.1f}+{HEAD['kai'][i]}+{O15[i]}" for i in range(3)))
    print("  body budget at 79.17 (KleidiAI head, O r15): " + " / ".join(f"{79.17 - HEAD['kai'][i] - O15[i]:.1f}" for i in range(3)))
    w1 = lambda i, b, k, head="kai": BODY[S[i]] * g(b, k, C) + O15[i] + HEAD[head][i]
    print("  tok/s with zero bytes read, k 1:   " + " / ".join(f"{950 / w1(i, 0, 1):.2f}" for i in range(3)))
    print("  tok/s with zero compute, b 1:      " + " / ".join(f"{950 / w1(i, 1, 0):.2f}" for i in range(3)))
    print("  width 1: largest b = k for 12:     " + " / ".join(fm(solve(lambda x: w1(i, x, x), 0, 1)) for i in range(3)))
    print("  width 1, two-stage head: same:     " + " / ".join(fm(solve(lambda x: w1(i, x, x, 'two'), 0, 1)) for i in range(3)))
    print("  width 1: largest b at k 1 / k 0.5: " + " / ".join(fm(solve(lambda x: w1(i, x, 1.0), 0, 1)) + " ; " + fm(solve(lambda x: w1(i, x, 0.5), 0, 1)) for i in range(3)))
    T = TAU(DS2)
    w2 = lambda i, x, e=0.45, tau=None, sh=0.85: ((BODY[S[i]] * g(x, x * (1 + e), C) + O15[i] + XPOS[i] + sh * GUESS[i]) / (tau or T[i][0]) + HEAD["two"][i])
    print("  width 2 every pass, measured tau (DeepSeek 1.185), two-stage head: largest b = k: " + " / ".join(fm(solve(lambda x: w2(i, x), 0, 1)) for i in range(3)))
    for e in (0.45, 0.30, 0.0):
        print(f"  width 2, stock bytes, e {e}: tau needed for 12: " + " / ".join(f"{(BODY[S[i]] * g(1, 1 + e, C) + O15[i] + XPOS[i] + .85 * GUESS[i]) / (79.17 - HEAD['two'][i]):.2f}" for i in range(3)))
    for W in (3, 4, 8):
        print(f"  width {W}, e 0.45 / e 0: tokens per pass needed: " + " / ".join(f"{(BODY[S[i]] * g(1, 1 + .45 * (W - 1), C) + O15[i] + (XPOS[i] + GUESS[i]) * (W - 1)) / (79.17 - HEAD['two'][i]):.2f} ; {(BODY[S[i]] * g(1, 1, C) + O15[i] + (XPOS[i] + GUESS[i]) * (W - 1)) / (79.17 - HEAD['two'][i]):.2f}" for i in range(3)))
    # delta across tokens: share d of input columns touched, exact column kernel at kc per touched weight (memo-05 K1: 1.55)
    for kc, lay in ((1.55, "row-major file"), (1.55, "column-major copy"), (1.0, "column-major copy")):
        bsh = (lambda d: (4 + 32 * d) / 36) if lay.startswith("row") else (lambda d: d)
        print(f"  delta, {lay}, k {kc} per touched weight: largest touched share d for 12 / to equal stock: "
              + " / ".join(fm(solve(lambda d: w1(i, bsh(d), kc * d), 0, 1)) + " ; " + fm(solve(lambda d: w1(i, bsh(d), kc * d), 0, 1, w1(i, 1, 1))) for i in range(3)))
    # low rank: W ~ U V at rank share q of n (params 2 q), factors at bits bf against 4.5
    for bf in (4.5, 8.5):
        print(f"  low rank, factors at {bf} bpw: largest parameter share kept for 12 (width 1, k 1 per param): " + " / ".join(fm(solve(lambda p: w1(i, p * bf / 4.5, p), 0, 1)) for i in range(3)))

if __name__ == "__main__":
    bt.selfcheck()
    assert [round(950 / ms(i, 1, 1, 40, HEAD["kai"][i], 8), 2) for i in range(3)] == [6.90, 6.21, 6.22]          # R72 section 5
    assert (P_any[0], P_2[0], P_3[0]) == (26, 27, 28), (P_any, P_2, P_3)                                         # tom-et23.md: 1.231 / 1.185 / 1.143
    assert abs(f(2.2, 40) - 1.62) < 0.03                                                                         # r17 section 1.2: body x 1.62
    a0 = [950 / ms(i, 2.2, 2.2, 40, HEAD["plain"][i], 8) for i in range(3)]
    assert all(abs(x - y) < 0.15 for x, y in zip(a0, (4.3, 3.9, 3.9))), a0                                        # r17: 4.3 / 3.9 / 3.9
    print("DeepSeek lookup (passes, width-2 passes, hits): any", P_any, ">=2", P_2, "3", P_3)
    for C in (50, 46, 40, 33): ladder(C)
    ladder(50, 0.17); ladder(40, 0.17)
    print("\nfactors f(c, C): " + "; ".join(f"C {C}: " + " ".join(f"{c}:{f(c, C):.3f}" for c in (1.25, 1.45, 1.6, 2.1, 2.2)) for C in (50, 46, 40, 33)))
    model(50); model(40)
    # tau needed for 12 at P4 form, C 50, width 2 every pass
    print("tau needed for 12.0 (P4 form, C 50): " + " / ".join(f"{(BODY[S[i]] * f(1.45, 50) + O15[i] + XPOS[i] + .85 * GUESS[i]) / (79.17 - HEAD['two'][i]):.2f}" for i in range(3)), "(ceiling 2)")
