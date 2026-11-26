#!/usr/bin/env python3
"""memo-13 (Bonnie): Ebony's bus items (R1 up-first gate rows, R3 down columns through K1, UD, R2 scale index) re-priced
on bonnie_cpu_ladder.py's inputs. Every output is *estimated*. Run: python3 -I -B bonnie_ud_check.py
Self-check: with her inputs (C 40, O 8, KleidiAI head, row kernel 1.15, down bytes x 0.5) this reproduces her section 3."""
import importlib.util, os
_d = os.path.dirname(os.path.abspath(__file__))
_s = importlib.util.spec_from_file_location("L", os.path.join(_d, "bonnie_cpu_ladder.py")); L = importlib.util.module_from_spec(_s); _s.loader.exec_module(L)
S, BODY, HEAD, O15, XPOS, GUESS, g = L.S, L.BODY, L.HEAD, L.O15, L.XPOS, L.GUESS, L.g
F = (0.7818, 0.6684, 0.6684)          # FFN share of body bytes (*host*, Ebony's tensor table)
KC = 2.1                              # K1 per touched weight (*device*, R72 L1: 1.992 hot, 2.239 cold, one A78, keep 0.60, 4096 inputs)
DB_E, DB_B = 0.5, (4 + 32 * 0.5) / 36 # down bytes at keep 0.5: hers; K1's layout reads every block's scales (memo-05 T7): 0.556
SEL = 0.5                             # ms per token per top-k selection over 11,008 channels x layers (*estimated*, not measured)

def bk(i, base, rows, down, kr, db, scale=False, kscale=1.0):
    """Body byte share and compute share (in packed-gemv units). base = kernel of everything untouched (1 packed, 1.45 Y1b)."""
    f = F[i] / 3
    b = (1 - 3 * f) + f + f * (0.7 if rows else 1.0) + f * (db if down else 1.0)
    k = (1 - 3 * f) * base + f * base + f * (0.7 * kr if rows else base) + f * (0.5 * KC if down else base)
    if scale: b, k = b * 17.5 / 18, k * kscale
    return b, k
def ms1(i, b, k, C, head, O, sel=0.0): return BODY[S[i]] * g(b, k, C) + O + sel + head
def tri(v): return " / ".join(f"{x:6.2f}" for x in v)
def dms(fn, ref): return " / ".join(f"{fn(i) - ref(i):+5.1f}" for i in range(3))

if __name__ == "__main__":
    # --- her numbers, her inputs
    her = lambda rows, down, scale=False: [ms1(i, *bk(i, 1, rows, down, 1.15, DB_E, scale, 1.03), 40, HEAD["kai"][i], 8) - ms1(i, 1, 1, 40, HEAD["kai"][i], 8) for i in range(3)]
    for got, want in ((her(1, 0), (-8.9, -8.9, -8.3)), (her(0, 1), (-13.0, -13.0, -12.0)), (her(1, 1), (-21.9, -21.8, -20.3)), (her(0, 0, True), (-2.2, -2.6, -2.5)), (her(1, 1, True), (-23.5, -23.9, -22.2))):
        assert all(abs(a - b) < 0.15 for a, b in zip(got, want)), (got, want)
    print("her section 3 reproduced with her inputs (C 40, O 8, row kernel 1.15, down bytes 0.5, scale k 1.03)")
    for C in (50, 40, 33):
        print(f"\n## C {C}")
        for col, base, head, kr_set in (("packed column (base P2: repack head, O r15)", 1.0, "repack", (1.45, 1.15)), ("path A (base A3: Y1b body, head-only copy, O r15)", 1.45, "repack", (1.45,))):
            ref = lambda i: ms1(i, 1, base, C, HEAD[head][i], O15[i])
            print(f" {col}: base tok/s " + tri([950 / ref(i) for i in range(3)]))
            for kr in kr_set:
                for lab, rows, down, db, sel in ((f"R1 gate rows, row kernel {kr}", 1, 0, DB_B, SEL), ("R1, cold rows as memo-05 E6 (no bytes saved)", 1, 0, DB_B, SEL),
                                                 ("R3 down via K1, bytes 0.556", 0, 1, DB_B, SEL), (f"UD, row kernel {kr}", 1, 1, DB_B, 2 * SEL)):
                    if "E6" in lab:
                        if kr != 1.45: continue
                        fn = lambda i: ms1(i, 1.0, bk(i, base, 1, 0, kr, db)[1], C, HEAD[head][i], O15[i], sel)
                    else: fn = lambda i: ms1(i, *bk(i, base, rows, down, kr, db), C, HEAD[head][i], O15[i], sel)
                    print(f"   {lab:46} ms {dms(fn, ref)}   tok/s {tri([950 / fn(i) for i in range(3)])}")
            if base == 1.0:
                for ks in (1.03, 1.08, 1.16):
                    fn = lambda i: ms1(i, *bk(i, base, 0, 0, 1, 1, True, ks), C, HEAD[head][i], O15[i])
                    print(f"   R2 scale index, arithmetic x {ks:<24} ms {dms(fn, ref)}   tok/s {tri([950 / fn(i) for i in range(3)])}")
                fn = lambda i: ms1(i, *bk(i, base, 1, 1, 1.45, DB_B, True, 1.08), C, HEAD["two"][i], O15[i], 2 * SEL)
                print(f"   UD + R2 (x 1.08) + two-stage head, width 1         tok/s {tri([950 / fn(i) for i in range(3)])}")
            fn = lambda i: ms1(i, *bk(i, base, 1, 1, 1.45, DB_B), C, HEAD["two" if base == 1 else head][i], O15[i], 2 * SEL)
            print(f"   UD{' + two-stage head' if base == 1 else ''}, width 1 (top UD stack)          tok/s {tri([950 / fn(i) for i in range(3)])}")
            # UD with width 2: masks of the two positions taken as independent; K1 once per position; gate rows: both / one / none
            w2 = 1.45 if base == 1 else 2.1
            T = L.TAU(L.DS2)
            def ud2(i):
                f = F[i] / 3
                b1, k1 = bk(i, base, 1, 1, 1.45, DB_B)
                b2 = (1 - 3 * f) + f + f * 0.91 + f * (4 + 32 * 0.75) / 36
                rk2 = 0.49 * 2.1 + 0.42 * 1.45                             # in-place rows: two positions 2.1, one position 1.45
                k2 = (1 - 3 * f) * w2 + f * w2 + f * rk2 + f * 2 * 0.5 * KC
                tau, s2, sh = T[i]
                p1 = BODY[S[i]] * g(b1, k1, C) + O15[i] + 2 * SEL
                p2 = BODY[S[i]] * g(b2, k2, C) + O15[i] + XPOS[i] + 4 * SEL
                return ((1 - s2) * p1 + s2 * p2 + sh * GUESS[i]) / tau + HEAD["two" if base == 1 else head][i]
            print(f"   UD + width 2 (same guess rules)                    tok/s {tri([950 / ud2(i) for i in range(3)])}")
