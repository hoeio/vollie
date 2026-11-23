#!/usr/bin/env python3
"""memo-12 round 25 (Bonnie): copy-run passes. When the newest earlier match of the last n tokens exists, the tokens that
followed it are proposed as a chain of W-1 guesses and verified in one pass of width W (exact: a guess is committed only
if the model's own greedy token equals it). *host* for the counts (Tom's dumps), *estimated* for the time model.
Run: lab/venv-c3/bin/python3 -I -B bonnie_copyrun.py <repo root>
Policy selection uses TRAINING prompts only (train1500) plus the head's held-out top-1; the set and the official prompt
are then read with the selected policy (the set: a second look, see the memo).
Time model per pass (ms), from the lane stack of r14 rung 6 (K 300, C5 out, head knob, two-stage head):
  t(W) = body(W) + O + xpos * (W - 1) [+ 2.6 if the head makes the guess];   + head_ms per committed token
  body(W): bonnie_tree_r2.body with cg(2,3,4) = 1.26, 1.61, 1.96 and the W-position CPU kernel (c = 1.45, 1.9, 2.35)
"""
import sys, os, importlib.util
import numpy as np
root = sys.argv[1]; here = os.path.dirname(os.path.abspath(__file__))
_s = importlib.util.spec_from_file_location("r14", os.path.join(here, "bonnie_floor_r14.py")); r14 = importlib.util.module_from_spec(_s); _s.loader.exec_module(r14)
t2, bt, r13 = r14.t2, r14.bt, r14.r13
M = {"Yi": ("yi", "head2117-tok", 0.377), "Llama-2": ("llama2", "head3000-tok", 0.471)}
CG = {1: 1.0, 2: r13.CG, 3: 1.6 * r13.CG / 1.25, 4: 1.95 * r13.CG / 1.25}; NMAX = 8; WMAX = 4
def load(prefix):
    rows = np.fromfile(prefix + ".rows", dtype=np.int32).reshape(-1, 4); raw = np.fromfile(prefix + ".ptok", dtype=np.int32)
    ptok, i = {}, 0
    while i < len(raw): p, n = int(raw[i]), int(raw[i + 1]); ptok[p] = raw[i + 2:i + 2 + n].tolist(); i += 2 + n
    out = {}
    for p in np.unique(rows[:, 0]):
        idx = np.where(rows[:, 0] == p)[0]; idx = idx[np.argsort(rows[idx, 1])]; out[int(p)] = (ptok[int(p)], rows[idx, 2].tolist(), idx)
    return out
def match(ctx):
    """(match length capped at NMAX, start of the continuation) for the newest earlier match of the longest suffix; (0, -1) if none."""
    L = len(ctx); best = (0, -1)
    last = ctx[-1]
    for s in range(L - 2, -1, -1):                      # s = position of the candidate's last token
        if ctx[s] != last: continue
        n = 1
        while n < NMAX and s - n >= 0 and ctx[s - n] == ctx[L - 1 - n]: n += 1
        if n > best[0]: best = (n, s + 1)
        if n == NMAX: break
    return best
def times(subj, model, O, xpos):
    def body(W):
        f = lambda: t2.body(subj, W, 300, "pair", cg=CG[W], c5=1.0)[0]
        return r14.lin(f) if model == "linear" else f()
    return {W: body(W) + O + xpos * (W - 1) for W in range(1, WMAX + 1)}
def run_hist(seqs):
    """hist[n][r] = count of steps with match length n whose continuation is right for exactly r leading tokens (r capped at WMAX-1)."""
    h = {n: [0] * WMAX for n in range(1, NMAX + 1)}; steps = 0
    for prompt, y, _ in seqs.values():
        for j in range(len(y) - 1):
            ctx = prompt + y[:j + 1]; n, c = match(ctx); steps += 1
            if n == 0: continue
            r = 0
            while r < WMAX - 1 and j + 1 + r < len(y) and c + r < len(ctx) + r and (ctx + y[j + 1:j + 1 + r])[c + r] == y[j + 1 + r]: r += 1
            h[n][r] += 1
    return h, steps
def pick_policy(hist, T, p_head, head_ms, lam):
    """per match length: the action with the best (tokens - lam * ms); actions: 'h' (head at width 2) or copy width 2..WMAX."""
    pol = {}
    for n in range(0, NMAX + 1):
        best = ("h", (1 + p_head) - lam * (T[2] + 2.6 + head_ms * (1 + p_head)))
        if n:
            tot = sum(hist[n])
            if tot:
                for W in range(2, WMAX + 1):
                    etok = sum(hist[n][r] * (1 + min(r, W - 1)) for r in range(WMAX)) / tot
                    v = etok - lam * (T[W] + head_ms * etok)
                    if v > best[1]: best = (W, v)
        pol[n] = best[0]
    return pol
def simulate(seqs, pred, pol, T, head_ms, p_head=None, rng=None):
    tok = ms = passes = 0.0; used = {}
    for prompt, y, idx in seqs.values():
        j = -1; ctx = list(prompt)                       # j = index of the last committed row; row 0 comes from the prompt's own pass
        # the first token of a reply is produced by the prefill; decode starts with y[0] known
        j = 0; tok += 0
        while j < len(y) - 1:
            c_ctx = prompt + y[:j + 1]; n, c = match(c_ctx); a = pol[n]; passes += 1
            if a == "h":
                if pred is not None: hit = pred[idx[j]] >= 0 and int(pred[idx[j]]) == y[j + 1]; guess = pred[idx[j]] >= 0
                else: hit = rng.random() < p_head; guess = True
                got = 2 if (hit and j + 2 <= len(y) - 1 + 1) else 1
                got = min(got, len(y) - 1 - j); ms += (T[2] + 2.6 if guess else T[1]) + head_ms * got
            else:
                W = a; r = 0; seq = c_ctx
                while r < W - 1 and j + 1 + r < len(y) and (seq + y[j + 1:j + 1 + r])[c + r] == y[j + 1 + r]: r += 1
                got = min(1 + r, len(y) - 1 - j); ms += T[W] + head_ms * got
            used[a] = used.get(a, 0) + 1; tok += got; j += got
    return tok, ms, passes, used
if __name__ == "__main__":
    r14.r13.r5.selfcheck()
    assert match([1, 2, 3, 9, 1, 2, 3]) == (3, 3) and match([5, 6]) == (0, -1) and match([4, 7, 4]) == (1, 1)
    import random
    for name, (d, head, held) in M.items():
        base = os.path.join(root, "lab/c3/head", d); S = bt.SUBJ[name]; head_ms = r13.HEAD["c7built"][name]
        tr = load(os.path.join(base, "train1500")); keys = sorted(tr)[::3]; tr = {k: tr[k] for k in keys}      # every third training prompt (500)
        hist, steps = run_hist(tr)
        print(f"\n{name}: training prompts {len(tr)}, steps {steps}; head held-out top-1 {held}")
        print("  match length: share of steps | P(first r continuation tokens right), r = 1, 2, 3")
        for n in range(1, NMAX + 1):
            tot = sum(hist[n])
            if tot: print(f"   n {n}{'+' if n == NMAX else ' '}: {tot / steps:.3f} | " + " ".join(f"{sum(hist[n][r:]) / tot:.3f}" for r in (1, 2, 3)))
        for model, O, xpos, lab in (("pair", 5.2, S["xpos"], "stack costs (O 5.2, xpos as memo-10)"), ("linear", 5.2, S["xpos"], "stack costs, linear"),
                                    ("pair", {"Yi": 4.8, "Llama-2": 6.9}[name], {"Yi": 3.2, "Llama-2": 5.3}[name], "r15 costs (O and second-position attention)")):
            T = times(name, model, O, xpos); lam = 1 / (T[2] / (1 + held)); rng = random.Random(0)
            for _ in range(12):                                                    # Dinkelbach on the training prompts
                pol = pick_policy(hist, T, held, head_ms, lam); tk, ms, _, _ = simulate(tr, None, pol, T, head_ms, held, rng); lam = tk / ms
            base_pol = {n: ("h" if n < 2 else 2) for n in range(NMAX + 1)}          # rule T of r24
            print(f"  {lab}, {model}: pass ms at width 1..4 = " + " / ".join(f"{T[W]:.1f}" for W in range(1, 5)) + f"; policy by match length 0..{NMAX}: " + " ".join(str(pol[n]) for n in range(NMAX + 1)))
            for tag in ("official", "eval"):
                seqs = load(os.path.join(base, tag)); pred = np.load(os.path.join(base, f"{head}.pt.{'official' if tag == 'official' else 'evalset'}.pred.npy"))
                res = []
                for plab, P in (("head only", {n: "h" for n in range(NMAX + 1)}), ("rule T", base_pol), ("copy-run", pol)):
                    tk, ms, ps, used = simulate(seqs, pred, P, T, head_ms); res.append(f"{plab}: {950 * tk / ms:.2f} tok/s, {tk / ps:.3f} tokens/pass")
                print(f"    {tag}: " + "; ".join(res))
