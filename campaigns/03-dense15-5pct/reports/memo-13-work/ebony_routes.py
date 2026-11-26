#!/usr/bin/env python3
"""memo-13 (Ebony): prices of the two-stage FFN route. Every output is *estimated*.

Model: memo-11-work/bonnie_bytes.py loaded unchanged (body = s * [max(x, y) + 0.25 * min(x, y)], x bus, y compute;
R 29.9 GB/s *device*; printed tok/s = 950 / ms). Run: python3 -I -B ebony_routes.py

Two-stage FFN with a share `skip` of the channels dropped per token:
  bytes(FFN)   = 2/3 * PB + (1 - skip)                       proxy of gate and up, then kept rows / columns
  compute(FFN) = 2/3 * kp + 2/3 * (1 - skip) * KR + 1/3 * (1 - skip) * KC
PB  proxy bytes per gate/up weight against Q4_0's 4.5 bits: 1-bit copy with a row scale, 0.23
kp  proxy kernel cost per weight against the packed Q4_0 gemv: 0.35 (sign x three bit-planes of the input, popcount;
    a count, not built) to 1.1 (sdot on a +-1 expansion; a count, not built)
KR  exact row kernel on any four rows, one stream per row: 1.15 (judgement; memo-05's in-place kernel was 1.47 hot on
    plain Q4_0 rows, *host*; a per-row packed side layout removes the unpack that cost it)
KC  column kernel K1 on `down`: 2.1 per touched weight (*device*, R72 L1: 1.992 hot, 2.239 cold, A78)
"""
import importlib.util, os
here = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("bb", os.path.join(here, "..", "memo-11-work", "bonnie_bytes.py"))
bb = importlib.util.module_from_spec(spec); spec.loader.exec_module(bb)
F = {"Yi": 0.7818, "Llama-2": 0.6684, "DeepSeek": 0.6684}     # FFN share of the body's bytes (*host*, GGUF tensor table)
PB, KR, KC, E = 0.23, 1.15, 2.1, 0.45

def bk(n, skip, kp):
    f = F[n]
    return (1 - f) + f * (2 / 3 * PB + 1 - skip), (1 - f) + f * (2 / 3 * kp + 2 / 3 * (1 - skip) * KR + (1 - skip) * KC / 3)

def ms1(n, skip, kp, C=40.0, O=8.0, hs=1.0):
    b, k = bk(n, skip, kp); return bb.ms(bb.S[n], b=b, k=k, C=C, O=O, hs=hs)

def ms2(n, skip, kp, C=40.0, O=5.2, hs=0.5):
    """Width 2 on top: the two positions have their own masks (taken as independent: the worst case for sharing)."""
    s, f = bb.S[n], F[n]
    keep = 1 - skip
    b = (1 - f) + f * (2 / 3 * PB + 1 - skip * skip)
    k = (1 - f) * (1 + E) + f * (2 / 3 * kp * (1 + E) + 2 / 3 * KR * (keep * keep * (1 + E) + 2 * skip * keep) + 2 * keep * KC / 3)
    scale = s["body"] / bb.fit(1e3 * s["B"] / bb.R, 1e3 * s["B"] / 40.0)
    body = scale * bb.fit(1e3 * b * s["B"] / bb.R, 1e3 * k * s["B"] / C)
    return (body + O + s["guess"] + s["xpos"]) / s["tau2"] + s["head"] * hs

def line(label, fn):
    v = [fn(n) for n in bb.S]
    print(f"{label:66s}" + "  ".join(f"{x:6.1f} {950 / x:5.2f}" for x in v))

if __name__ == "__main__":
    assert [round(950 / bb.ms(bb.S[n]), 2) for n in bb.S] == [6.90, 6.21, 6.22]
    assert all(abs(a - b) < 1e-9 for n in bb.S for a, b in zip(bk(n, 0.0, 0.0), ((1 - F[n]) + F[n] * (2 / 3 * PB + 1), 1 - F[n] + F[n] * (2 / 3 * KR + KC / 3))))
    print(f"{'ms per token, printed tok/s':66s}" + "  ".join(f"{n:>12s}" for n in bb.S))
    line("stock (memo-11 model, C 40, O 8)", lambda n: bb.ms(bb.S[n]))
    for skip in (0.35, 0.5, 0.65):
        for kp in (0.35, 1.1):
            for C in (40.0, 50.0):
                line(f"two-stage FFN alone, skip {skip:.2f}, kp {kp:.2f}, C {C:.0f}", lambda n: ms1(n, skip, kp, C))
    for skip in (0.5, 0.65):
        for kp in (0.35, 1.1):
            line(f"  + C7 + O 5.2, skip {skip:.2f}, kp {kp:.2f}, C 40", lambda n: ms1(n, skip, kp, 40.0, 5.2, 0.5))
            for C in (40.0, 50.0):
                line(f"  + C7 + O 5.2 + width 2 (measured heads), skip {skip:.2f}, kp {kp:.2f}, C {C:.0f}", lambda n: ms2(n, skip, kp, C))
    line("what-if: a free perfect predictor at skip 0.65, rows only, + C7 + O 5.2", lambda n: bb.ms(bb.S[n], b=1 - F[n] * 0.65, k=1 - F[n] * 0.65, O=5.2, hs=0.5))
    # (an 8-bit scale code was screened and closed: single blocks are off by up to 0.7 to 2.0 x, PPL x 15 on Yi, *host*)
    # exact-row rules (no proxy): compute `up` (or `gate`) fully, skip a share of the other matrix's rows; optionally
    # the same channels' `down` columns through K1. Then the scale code and the three together.
    def rows(n, skip, down, extra_b=1.0, extra_k=1.0):
        f = F[n]
        fb = 1 - skip / 3 - (skip / 3 if down else 0)
        fk = 1 / 3 + (1 - skip) * KR / 3 + ((1 - skip) * KC / 3 if down else 1 / 3)
        return ((1 - f) + f * fb) * extra_b, ((1 - f) + f * fk) * extra_k
    for skip in (0.3, 0.4):
        for down in (False, True):
            for C in (40.0, 50.0):
                line(f"row rule, skip {skip:.2f}, down columns {'K1' if down else 'dense'}, C {C:.0f}",
                     lambda n: bb.ms(bb.S[n], *rows(n, skip, down), C=C))
    for C in (40.0, 50.0):
        line(f"scale code 12 bits lossless (bytes x 17.5/18), C {C:.0f}", lambda n: bb.ms(bb.S[n], b=17.5 / 18, k=1.03, C=C))
        line(f"row rule 0.30 + K1 down + 12-bit scales, C {C:.0f}", lambda n: bb.ms(bb.S[n], *rows(n, 0.3, True, 17.5 / 18, 1.03), C=C))
        line(f"  the same + C7 + O 5.2, C {C:.0f}", lambda n: bb.ms(bb.S[n], *rows(n, 0.3, True, 17.5 / 18, 1.03), C=C, O=5.2, hs=0.5))
        line(f"two-stage, 2-bit proxy (PB 0.45, kp 0.9), skip 0.50, C {C:.0f}", lambda n: bb.ms(bb.S[n],
             (1 - F[n]) + F[n] * (2 / 3 * 0.45 + 0.5), (1 - F[n]) + F[n] * (2 / 3 * 0.9 + 2 / 3 * 0.5 * KR + 0.5 * KC / 3), C=C))
        line(f"bound: no arithmetic at all, stock bytes (bus term + head + O 8), C {C:.0f}", lambda n: 1e3 * bb.S[n]["B"] / bb.R + 8 + bb.S[n]["head"])
    # exact mask after gate and up (D50): `down` at half its columns through K1; and U30 rows + D50 columns (UD)
    def d50(n, kc=KC): return 1 - F[n] / 6, 1 - F[n] / 3 + F[n] / 3 * 0.5 * kc
    def ud(n, eb=1.0, ek=1.0): return (1 - F[n] * 0.8 / 3) * eb, (1 - F[n] * 2 / 3 + F[n] * (0.7 * KR + 0.5 * KC) / 3) * ek
    for C in (40.0, 33.0):
        line(f"D50: down columns by |h|, K1 at 2.1, C {C:.0f}", lambda n: bb.ms(bb.S[n], *d50(n), C=C))
        line(f"UD: U30 gate rows + D50 down columns, C {C:.0f}", lambda n: bb.ms(bb.S[n], *ud(n), C=C))
        line(f"UD + 12-bit scales, C {C:.0f}", lambda n: bb.ms(bb.S[n], *ud(n, 17.5 / 18, 1.03), C=C))
        line(f"stock at C {C:.0f}", lambda n: bb.ms(bb.S[n], C=C))
    line("UD + 12-bit scales + C7 + O 5.2, C 40", lambda n: bb.ms(bb.S[n], *ud(n, 17.5 / 18, 1.03), O=5.2, hs=0.5))
    print("D50 (b, k): " + " ".join(f"{d50(n)[0]:.3f}/{d50(n)[1]:.3f}" for n in bb.S) + "   UD (b, k): " + " ".join(f"{ud(n)[0]:.3f}/{ud(n)[1]:.3f}" for n in bb.S))
    print("row rule shares (b, k): " + "  ".join(f"{sk:.1f}{'d' if dn else ''}: " + " ".join(f"{rows(n, sk, dn)[0]:.3f}/{rows(n, sk, dn)[1]:.3f}" for n in bb.S) for sk in (0.3, 0.4) for dn in (False, True)))
    print("bytes and compute shares of the body (b, k):")
    for skip in (0.5, 0.65):
        for kp in (0.35, 1.1):
            print(f"  skip {skip:.2f} kp {kp:.2f}: " + "  ".join(f"{bk(n, skip, kp)[0]:.3f} {bk(n, skip, kp)[1]:.3f}" for n in bb.S))
