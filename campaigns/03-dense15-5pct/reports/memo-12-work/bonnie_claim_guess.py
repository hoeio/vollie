#!/usr/bin/env python3
"""memo-12 round 24 (Bonnie): the claim prompt's 31 guesses for every guess source that exists, and a chooser fixed on
the TRAINING split only. *host*, numpy, reads Tom's dumps (lab/c3/head/<m>/{official,eval,train1500}.{rows,ptok}) and
his saved head guesses; no model run. Lookup = tools/c3_head_t1.py arm a (newest earlier match of the last 3, else 2,
else 1 tokens; propose its successor), re-implemented here so that the match length is known.
Run: lab/venv-c3/bin/python3 -I -B bonnie_claim_guess.py <repo root>
Rule "T" (fixed on training data only): use the lookup's guess when its match length n has, on the training prompts,
a precision above the head's held-out top-1 (tom-head.md section 4.1); else the head.
"""
import sys, os
import numpy as np
root = sys.argv[1]
M = {"Yi": ("yi", "head2117-tok", 0.377), "Llama-2": ("llama2", "head3000-tok", 0.471)}     # best head, its held-out top-1 (Tom)
ALT = {"Yi": ["head-tok"], "Llama-2": ["head-tok", "head3000-full"]}
def load(prefix):
    rows = np.fromfile(prefix + ".rows", dtype=np.int32).reshape(-1, 4); raw = np.fromfile(prefix + ".ptok", dtype=np.int32)
    ptok, i = {}, 0
    while i < len(raw): p, n = int(raw[i]), int(raw[i + 1]); ptok[p] = raw[i + 2:i + 2 + n].tolist(); i += 2 + n
    seqs = {}
    for p in np.unique(rows[:, 0]):
        idx = np.where(rows[:, 0] == p)[0]; idx = idx[np.argsort(rows[idx, 1])]
        seqs[int(p)] = (ptok[int(p)], rows[idx, 2].tolist(), idx)
    return seqs
def lookup(ctx, nmax=3):
    for n in range(nmax, 0, -1):
        if len(ctx) <= n: continue
        suf = ctx[-n:]
        for s in range(len(ctx) - n - 1, -1, -1):
            if ctx[s:s + n] == suf: return ctx[s + n], n
    return None, 0
def steps(seqs, pred=None):
    """per step: (truth, lookup guess, match length, head guess)"""
    out = {}
    for p, (prompt, y, idx) in seqs.items():
        st = []
        for j in range(len(y) - 1):
            g, n = lookup(prompt + y[:j + 1]); h = None if pred is None or pred[idx[j]] < 0 else int(pred[idx[j]])
            st.append((y[j + 1], g, n, h))
        out[p] = (st, len(y))
    return out
def tau(stp, choose):
    tok = pas = hit = gss = 0
    for st, n in stp.values():
        j = 0
        while j < n:
            pas += 1
            if j < n - 1:
                g = choose(st[j])
                if g is not None:
                    gss += 1
                    if g == st[j][0]: hit += 1; j += 2; continue
            j += 1
        tok += n
    return tok / pas, hit, gss
assert lookup([1, 2, 3, 9, 1, 2, 3]) == (9, 3) and lookup([5, 6]) == (None, 0) and lookup([4, 7, 4]) == (7, 1)
for name, (d, head, held) in M.items():
    base = os.path.join(root, "lab/c3/head", d)
    tr = steps(load(os.path.join(base, "train1500")))
    cnt = {n: [0, 0] for n in (1, 2, 3)}
    for st, _ in tr.values():
        for y, g, n, _h in st:
            if n: cnt[n][1] += 1; cnt[n][0] += (g == y)
    prec = {n: cnt[n][0] / max(1, cnt[n][1]) for n in cnt}; use = {n for n in prec if prec[n] > held}
    allsteps = sum(len(st) for st, _ in tr.values())
    print(f"{name}: training prompts {len(tr)}, steps {allsteps}; lookup precision by match length 3 / 2 / 1: "
          + " / ".join(f"{prec[n]:.3f} (share of steps {cnt[n][1] / allsteps:.3f})" for n in (3, 2, 1)) + f"; head held-out top-1 {held}; rule T uses lookup at lengths {sorted(use)}")
    for tag in ("official", "eval"):
        seqs = load(os.path.join(base, tag))
        print(f"  {tag}:")
        for hd in [head] + ALT[name]:
            pred = np.load(os.path.join(base, f"{hd}.pt.{'official' if tag == 'official' else 'evalset'}.pred.npy")); stp = steps(seqs, pred)
            rules = [("head", lambda s: s[3]), ("lookup only", lambda s: s[1]), ("lookup, else head (Tom's ah)", lambda s: s[1] if s[1] is not None else s[3]),
                     ("lookup at length 3, else head", lambda s: s[1] if s[2] == 3 else s[3]), ("lookup at length >= 2, else head", lambda s: s[1] if s[2] >= 2 else s[3]),
                     ("rule T", lambda s: s[1] if s[2] in use else s[3]),
                     ("oracle: either right (not a rule)", lambda s: s[0] if (s[1] == s[0] or s[3] == s[0]) else s[3])]
            line = "; ".join(f"{lab}: {t:.3f} ({h}/{g})" for lab, (t, h, g) in ((lab, tau(stp, f)) for lab, f in rules))
            print(f"    {hd}: {line}")
        if tag == "official":
            stp = steps(seqs, np.load(os.path.join(base, f"{head}.pt.official.pred.npy"))); st = list(stp.values())[0][0]
            print("    per step (head hit, lookup hit, match length): " + " ".join(("H" if h == y else "-") + ("L" if g == y else ("x" if g is not None else "-")) + str(n) for y, g, n, h in st))
