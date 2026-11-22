#!/usr/bin/env python3
"""memo-11 pricing (Bonnie): fewer bytes and cheaper compute per token. Every output is *estimated*.

Inputs: R72 *device* (stock body ms, body GB, head ms, read 29.9 GB/s), memo-10's pass model (Ebony's two-point
fit), tom-head.md *host* (tokens per pass at width 2).

  body(b, k, W) = s * [max(x, y) + 0.25 * min(x, y)]
     x = b * B / R                      bus time; b = bytes as a share of the stock Q4_0 body, B = stock body GB
     y = k * (1 + e * (W - 1)) * B / C  compute time; k = cost per weight against the stock Q4_0 sdot kernel
     s = measured stock body / model stock body (per subject, 0.96 to 0.98)
  ms per committed token = (body + O + guess + xpos * (W - 1)) / tau + head * hs;   printed tok/s = 950 / ms
"""
R = 29.9
S = {  # body ms, body GB, head ms, tokens per pass at width 2 (DeepSeek: mean, not measured), guess ms, xpos ms
    "Yi":       dict(body=120.6, B=3.11, head=9.04,  tau2=1.326, guess=2.64, xpos=0.5),
    "Llama-2":  dict(body=140.6, B=3.64, head=4.40,  tau2=1.448, guess=2.61, xpos=1.5),
    "DeepSeek": dict(body=130.8, B=3.42, head=13.88, tau2=1.385, guess=2.59, xpos=1.5),
}

def fit(x, y): return max(x, y) + 0.25 * min(x, y)

def body(s, b=1.0, k=1.0, W=1, C=40.0, e=0.45):
    scale = s["body"] / fit(1e3 * s["B"] / R, 1e3 * s["B"] / C)
    return scale * fit(1e3 * b * s["B"] / R, 1e3 * k * (1 + e * (W - 1)) * s["B"] / C)

def ms(s, b=1.0, k=1.0, W=1, C=40.0, O=8.0, hs=1.0, e=0.45):
    if W == 1: return body(s, b, k, 1, C) + O + s["head"] * hs
    return (body(s, b, k, W, C, e) + O + s["guess"] + s["xpos"]) / s["tau2"] + s["head"] * hs

def b_needed(s, k, C=40.0, O=8.0, hs=1.0, goal=79.2):
    """Largest byte share at which a one-position token meets the goal; None if compute alone is too slow."""
    lo, hi = 0.0, 1.5
    if ms(s, 0.0, k, 1, C, O, hs) > goal: return None
    for _ in range(50):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if ms(s, mid, k, 1, C, O, hs) <= goal else (lo, mid)
    return lo

def row(label, **kw):
    print(f"{label:58s}" + "  ".join(f"{ms(S[n], **kw):6.1f} {950 / ms(S[n], **kw):5.2f}" for n in S))

if __name__ == "__main__":
    # self-check: stock reproduces R72 section 5 (6.90 / 6.21 / 6.22)
    got = [round(950 / ms(S[n]), 2) for n in S]
    assert got == [6.90, 6.21, 6.22], got
    print(f"{'lever (ms, printed tok/s)':58s}" + "  ".join(f"{n:>12s}" for n in S))
    row("stock Q4_0")
    row("3-bit 3.44 bpw, ggml Q3_K kernel (k 1.73, device)", b=0.764, k=1.73)
    row("3-bit 3.44 bpw, sdot at Q4_0 cost (k 1.0)", b=0.764, k=1.0)
    row("3-bit 3.44 bpw, LUT kernel k 0.75", b=0.764, k=0.75)
    row("3-bit 3.44 bpw, LUT k 0.60 (top of band)", b=0.764, k=0.60)
    row("mixed 3/4-bit 3.9 bpw (b 0.86), LUT k 0.85", b=0.86, k=0.85)
    row("Q2_K-mix served file (b 0.66), ggml Q2_K (k 1.11, device)", b=0.66, k=1.11)
    row("Q2_K-mix served file (b 0.66), LUT k 0.55", b=0.66, k=0.55)
    row("pure 2.625 bpw (b 0.583), LUT k 0.50", b=0.583, k=0.50)
    row("3-bit LUT 0.75 + C7 + O 5.2", b=0.764, k=0.75, O=5.2, hs=0.5)
    row("Q2_K-mix LUT 0.55 + C7 + O 5.2", b=0.66, k=0.55, O=5.2, hs=0.5)
    row("pure 2.625 LUT 0.50 + C7 + O 5.2", b=0.583, k=0.50, O=5.2, hs=0.5)
    row("3-bit LUT 0.75 + C7 + O 5.2 + width 2 (e 0.85)", b=0.764, k=0.75, O=5.2, hs=0.5, W=2, e=0.85)
    row("Q2_K-mix LUT 0.55 + C7 + O 5.2 + width 2 (e 0.85)", b=0.66, k=0.55, O=5.2, hs=0.5, W=2, e=0.85)
    row("pure 2.625 LUT 0.50 + C7 + O 5.2 + width 2 (e 0.85)", b=0.583, k=0.50, O=5.2, hs=0.5, W=2, e=0.85)
    row("Q2_K-mix LUT 0.55 + C7 + O 5.2 + width 2, C 50", b=0.66, k=0.55, O=5.2, hs=0.5, W=2, e=0.85, C=50)
    row("3-bit LUT 0.75 x 20 % fewer bytes+compute (skip) + C7 + O", b=0.764 * 0.8, k=0.75 * 0.8, O=5.2, hs=0.5)
    row("Q2_K-mix LUT 0.55 x 15 % skip + C7 + O 5.2", b=0.66 * 0.85, k=0.55 * 0.85, O=5.2, hs=0.5)
    print("\nbyte share needed for 12 tok/s at one position (bpw = share x 4.5)")
    for lab, kw in (("stock head, O 8", {}), ("C7 + O 5.2", dict(O=5.2, hs=0.5))):
        for k in (1.0, 0.75, 0.5, 0.35):
            for C in (35.0, 40.0, 50.0):
                v = [b_needed(S[n], k, C, **kw) for n in S]
                print(f"  {lab:16s} k {k:4.2f} C {C:2.0f}: " +
                      "  ".join("  none      " if x is None else f"{x:5.3f} ({x * 4.5:4.2f})" for x in v))
