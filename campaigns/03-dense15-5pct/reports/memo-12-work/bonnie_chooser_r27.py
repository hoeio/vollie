#!/usr/bin/env python3
"""memo-12 round 27 (Bonnie): E-T21, the per-step chooser (lookup, head, or no guess), fitted and scored on the HELD-OUT
TRAINING prompts only (the 5 % that tools/c3_head_train.py never trained on; same rng). The fixed prompt set and the
official prompt are NOT read here. *host* arithmetic on Tom's dumps and saved heads; no model run. Every tok/s it prints
is *estimated* (pass costs from the r14 stack model, rung "priced lane": K 300, O 5.2, _mc3 at two columns).
Run: lab/venv-c3/bin/python3 -I -B bonnie_chooser_r27.py <repo root>

Pass costs (ms): width 1 = P1; width 2 = P2 (second position: lane at two columns, xpos, a second attention walk);
G = guess head, paid on every pass that consults it; the exact head is paid per committed token (constant, left out of the
choice, kept in the rate). A guess with hit probability p is worth making when p > p* = (P2 - P1) / (ms per token - head).
"""
import sys, os, importlib.util
import numpy as np
root = sys.argv[1]; here = os.path.dirname(os.path.abspath(__file__))
_s = importlib.util.spec_from_file_location("r13", os.path.join(here, "bonnie_stack_r13.py")); r13 = importlib.util.module_from_spec(_s); _s.loader.exec_module(r13)
r5, t2, bt = r13.r5, r13.t2, r13.bt
sys.path.insert(0, os.path.join(root, "lab/llama.cpp-c3/gguf-py"))
import gguf, torch
torch.set_num_threads(2)
M = {"Yi": ("yi", "train2117", "head2117-tok", "yi15-6b-q4_0.gguf", 0.3770), "Llama-2": ("llama2", "train3000", "head3000-tok", "llama2-7b-q4_0.gguf", 0.4710)}
ATT2 = {"Yi": 3.2 - 0.9, "Llama-2": 5.3 - 1.7}     # r15 second-position attention minus r17's shared value conversion (*estimated*)

def costs(subj, model):
    keep = t2.rates
    if model == "linear": t2.rates = r5.rates_linear
    try:
        b1 = t2.body(subj, 1, 300, "pair", c5=1.0)[0]; b2 = t2.body(subj, 2, 300, "pair", cg=r13.CG, c5=1.0)[0]
    finally: t2.rates = keep
    s = bt.SUBJ[subj]
    return b1 + 5.2, b2 + 5.2 + s["xpos"] + ATT2[subj], s["guess"], r13.REP[subj]

def ptoks(prefix):
    raw = np.fromfile(prefix + ".ptok", dtype=np.int32); out, i = {}, 0
    while i < len(raw): p, n = int(raw[i]), int(raw[i + 1]); out[p] = raw[i + 2:i + 2 + n].tolist(); i += 2 + n
    return out
def lookup(ctx, nmax=3):                     # tools/c3_head_t1.py arm a, with the match length (as bonnie_claim_guess.py)
    for n in range(nmax, 0, -1):
        if len(ctx) <= n: continue
        suf = ctx[-n:]
        for s in range(len(ctx) - n - 1, -1, -1):
            if ctx[s:s + n] == suf: return ctx[s + n], n
    return None, 0
assert lookup([1, 2, 3, 9, 1, 2, 3]) == (9, 3) and lookup([5, 6]) == (None, 0) and lookup([4, 7, 4]) == (7, 1)

def held_steps(name):
    d, dump, head, model, want = M[name]; base = os.path.join(root, "lab/c3/head", d)
    rd = gguf.GGUFReader(os.path.join(root, "lab/c3/models", model))
    deq = lambda nm: torch.from_numpy(gguf.quants.dequantize(*(lambda t: (t.data, t.tensor_type))(next(t for t in rd.tensors if t.name == nm))).astype(np.float32))
    Wh, E = deq("output.weight"), deq("token_embd.weight"); n_embd = Wh.shape[1]
    rows = np.fromfile(os.path.join(base, dump + ".rows"), dtype=np.int32).reshape(-1, 4)
    H = np.memmap(os.path.join(base, dump + ".h16"), dtype=np.float16, mode="r", shape=(len(rows), n_embd))
    order = np.lexsort((rows[:, 1], rows[:, 0])); r = rows[order]; ok = (r[1:, 0] == r[:-1, 0]) & (r[1:, 1] == r[:-1, 1] + 1)
    ja = order[:-1][ok]; prompts = np.unique(rows[ja, 0])
    held = sorted(np.random.default_rng(0).choice(prompts, size=max(1, int(len(prompts) * 0.05)), replace=False).tolist())   # the trainer's held-out set
    pt = ptoks(os.path.join(base, "train1500")); pb = ptoks(os.path.join(base, "train1500b"))
    sd = torch.load(os.path.join(base, head + ".pt"), map_location="cpu", weights_only=True); W, b, Wt = sd["W"].float(), sd["b"].float(), sd["Wt"].float()
    out, hit, tot = {}, 0, 0
    for p in held:
        idx = np.where(rows[:, 0] == p)[0]; idx = idx[np.argsort(rows[idx, 1])]; y = rows[idx, 2].tolist()
        prompt = pt[p] if p < 10000 else pb[p - 10000]
        with torch.no_grad():
            x = torch.from_numpy(np.asarray(H[idx[:-1]], dtype=np.float32)); yt = torch.as_tensor(y[:-1])
            lg = (x + torch.nn.functional.silu(x @ W.T + E[yt] @ Wt.T + b)) @ Wh.T
            t2v, t2i = lg.topk(2, dim=1); p1 = torch.softmax(lg, 1).max(1).values
        st = []
        for j in range(len(y) - 1):
            g, n = lookup(prompt + y[:j + 1]); h = int(t2i[j, 0])
            st.append((y[j + 1], g, n, h, float(p1[j]), float(t2v[j, 0] - t2v[j, 1])))
            hit += h == y[j + 1]; tot += 1
        out[p] = (st, len(y))
    assert abs(hit / tot - want) < 0.002, (name, hit / tot, want)      # reproduces the trainer's held-out top-1
    return out, hit / tot, tot

# a policy maps a step to (guess or None, head consulted?)
def pol_head(s): return s[3], True
def pol_T(s): return (s[1], False) if s[2] >= 2 else (s[3], True)
def make(theta, conf=4):
    c0, c1, c2, c3, skip = theta          # c_n: at match length n use the head when its confidence >= c_n, else the lookup; c0: no lookup -> head if >= c0 else no guess
    cn = (c0, c1, c2, c3)                 # skip = n: at match length >= n take the lookup without running the guess head (0 = never)
    def f(s):
        y, g, n, h, *_ = s; c = s[conf]
        if n == 0: return (h if c >= c0 else None), True
        if skip and n >= skip: return g, False
        if g == h: return h, True
        return (h if c >= cn[n] else g), True
    return f
def sim(stp, pol, P1, P2, G, head, only=None):
    ms = tok = pas = 0.0
    for p, (st, n) in stp.items():
        if only is not None and p not in only: continue
        j = 0
        while j < n:
            pas += 1
            if j < n - 1:
                g, used = pol(st[j]); ms += G * used
                if g is not None:
                    ms += P2
                    j += 2 if g == st[j][0] else 1
                    continue
            ms += P1; j += 1
        tok += n
    return 950 * tok / (ms + head * tok), tok / pas
def fit(stp, only, grid, c, conf=4):
    best = (0.0, 0.0, 0.0, 0.0, 0); f = lambda th: sim(stp, make(th, conf), *c, only=only)[0]; fb = f(best)
    for _ in range(2):                    # coordinate ascent; ponytail: not a global search, 5 small coordinates, add a full grid if the two rounds disagree
        for k in range(5):
            for v in ((0, 2, 3) if k == 4 else grid):
                th = best[:k] + (v,) + best[k + 1:]; r = f(th)
                if r > fb + 1e-9: best, fb = th, r
    return best

if __name__ == "__main__":
    for name in M:
        stp, top1, tot = held_steps(name); prm = sorted(stp); folds = [set(prm[i::5]) for i in range(5)]
        S = [s for st, _ in stp.values() for s in st]; y = np.array([s[3] == s[0] for s in S]); p1 = np.array([s[4] for s in S]); mg = np.array([s[5] for s in S])
        auc = lambda v: (np.argsort(np.argsort(v))[y].mean() - (y.sum() - 1) / 2) / (len(y) - y.sum())
        print(f"\n## {name}: held-out prompts {len(stp)}, guess steps {tot}, head top-1 {top1:.4f}; AUC of head-hit by top-1 probability {auc(p1):.3f}, by top-2 logit gap {auc(mg):.3f}")
        qs = np.quantile(p1, np.arange(0, 1.01, 0.1)); line = []
        for a, b in zip(qs[:-1], qs[1:]):
            m = (p1 >= a) & (p1 <= b); line.append(f"{a:.2f}-{b:.2f}: {y[m].mean():.2f}")
        print("  head precision by decile of its top-1 probability: " + "; ".join(line))
        for n in (0, 1, 2, 3):
            m = [s for s in S if s[2] == n]
            if n == 0: print(f"  no lookup candidate: share {len(m) / len(S):.3f}, head precision {np.mean([s[3] == s[0] for s in m]):.3f}"); continue
            ag = [s for s in m if s[1] == s[3]]; di = [s for s in m if s[1] != s[3]]
            print(f"  match length {n}: share {len(m) / len(S):.3f}; lookup precision {np.mean([s[1] == s[0] for s in m]):.3f}, head {np.mean([s[3] == s[0] for s in m]):.3f}; "
                  f"agree {len(ag) / len(m):.2f} of them (precision {np.mean([s[0] == s[3] for s in ag]):.3f}); disagree: lookup right {np.mean([s[1] == s[0] for s in di]):.3f}, head right {np.mean([s[3] == s[0] for s in di]):.3f}")
        for model in ("pair", "linear"):
            c = costs(name, model); P1, P2, G, head = c
            base, tpp = sim(stp, pol_head, *c); pstar = (P2 - P1) / (950 / base - head)
            print(f"  [{model}] P1 {P1:.1f} ms, P2 {P2:.1f} ms (ratio {P2 / P1:.3f}), guess {G:.2f}, head {head:.1f}; break-even hit probability p* {pstar:.3f}")
            print(f"    width 1, no guess head: {950 / (P1 + head):.2f} tok/s;  head always: {base:.2f} (tokens per pass {tpp:.3f});  rule T: {sim(stp, pol_T, *c)[0]:.2f} ({sim(stp, pol_T, *c)[1]:.3f})")
            for lab, conf, vals in (("top-1 probability", 4, p1), ("top-2 logit gap", 5, mg)):
                grid = [0.0] + np.quantile(vals, np.arange(0.1, 1.0, 0.1)).round(3).tolist() + [1e9]
                num = den = tp = 0.0; per = []
                for f in folds:           # cross-fit: thresholds from four fifths of the held-out prompts, scored on the fifth
                    th = fit(stp, set(prm) - f, grid, c, conf); r, t = sim(stp, make(th, conf), *c, only=f); r0 = sim(stp, pol_head, *c, only=f)[0]; rT = sim(stp, pol_T, *c, only=f)[0]
                    per.append((r / r0 - 1, r / rT - 1)); tp += t / 5
                per = np.array(per) * 100; th = fit(stp, None, grid, c, conf); rall, tall = sim(stp, make(th, conf), *c)
                print(f"    chooser on {lab}: cross-fitted gain over head-always {per[:, 0].mean():+.2f} % (folds {per[:, 0].min():+.2f} to {per[:, 0].max():+.2f}), over rule T {per[:, 1].mean():+.2f} % "
                      f"(folds {per[:, 1].min():+.2f} to {per[:, 1].max():+.2f}); tokens per pass {tp:.3f}")
                print(f"      fitted on all held-out prompts (in-sample {rall:.2f} tok/s, tokens per pass {tall:.3f}): c0 {th[0]}, c1 {th[1]}, c2 {th[2]}, c3 {th[3]}, skip-at {th[4]}")
