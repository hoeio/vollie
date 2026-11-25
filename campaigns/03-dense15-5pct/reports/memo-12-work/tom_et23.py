#!/usr/bin/env python3
"""E-T23 (Tom, 2026-11-25): DeepSeek dump results 1 to 4 of ebony-joins-r12 §2.4, *host*. Reads lab/c3/head/<m>/
{official,official-p,eval}.{ptok,rows,gap}. Lookup as ebony_deepseek_prompt.py (longest match 3, 2, 1, latest
occurrence; one guess per pass). Run: python3 -I tom_et23.py <repo root>"""
import sys, os
import numpy as np
root = sys.argv[1]; H = os.path.join(root, "lab/c3/head")
sys.path.insert(0, os.path.join(root, "lab/llama.cpp-c3/gguf-py"))
def load(d, n):
    p = os.path.join(H, d, n); pt = np.fromfile(p + ".ptok", np.int32); rows = np.fromfile(p + ".rows", np.int32).reshape(-1, 4)
    gap = np.fromfile(p + ".gap", np.float32) if os.path.exists(p + ".gap") else None
    prompts, i = {}, 0
    while i < len(pt): prompts[int(pt[i])] = pt[i + 2:i + 2 + pt[i + 1]].tolist(); i += 2 + pt[i + 1]
    return prompts, rows, gap
def lookup(ctx):
    for n in (3, 2, 1):
        if len(ctx) <= n: continue
        tail = ctx[-n:]
        for q in range(len(ctx) - n - 1, -1, -1):
            if ctx[q:q + n] == tail: return ctx[q + n], n
    return None, 0
def steps(pr, out):   # per step j (guess of out[j+1]): (guess made, match length, right)
    r = []
    for j in range(len(out) - 1):
        g, ln = lookup(pr + out[:j + 1]); r.append((g is not None, ln, g is not None and g == out[j + 1]))
    return r
def passes(out, st, minlen):
    h = [ok and ln >= minlen for _, ln, ok in st] + [False]; j = p = 0
    while j < len(out): p += 1; j += 2 if (j + 1 < len(out) and h[j]) else 1
    return p
def vocab(m):
    from gguf import GGUFReader
    f = GGUFReader(os.path.join(root, f"lab/c3/models/{m}-q4_0.gguf")).fields["tokenizer.ggml.tokens"]
    return [bytes(f.parts[i]).decode("utf-8", "replace") for i in f.data]
def text(v, ids): return "".join(v[t] for t in ids).replace("Ġ", " ").replace("Ċ", "\n").replace("▁", " ")
V = {"deepseek": vocab("deepseek7b")}
# results 1 to 3: the official prompt
reps = {}
for d in ("deepseek", "llama2", "yi"):
    for n in ("official", "official-p"):
        if not os.path.exists(os.path.join(H, d, n + ".rows")): continue
        pr, rows, gap = load(d, n); o = np.argsort(rows[:, 1]); out = rows[o, 2].tolist(); reps[(d, n)] = out
        line = f"{d} {n}: prompt tokens {len(pr[0])}, first ids {pr[0][:4]}, reply {len(out)} tokens"
        if gap is not None: g = gap[o]; s = np.argsort(g)[:3]; line += "; smallest top-2 gaps " + ", ".join(f"{g[i]:.3f} at token {i}" for i in s)
        print(line)
        if d == "deepseek":
            st = steps(pr[0], out); print("   reply:", repr(text(V[d], out)))
            print("   lookup steps:", " ".join(("L" if ok else ("x" if gm else "-")) + str(ln) for gm, ln, ok in st))
            print("   tokens per pass, lookup alone: " + "; ".join(f"length >= {k}: {passes(out, st, k)} passes, {len(out) / passes(out, st, k):.3f}" for k in (1, 2, 3)))
    if (d, "official-p") in reps and (d, "official") in reps:
        a, b = reps[(d, "official")], reps[(d, "official-p")]; same = [x == y for x, y in zip(a, b)]
        print(f"   {d} R/F16/as-today against P/Q8_0/one-batch: equal tokens {sum(same)} of {len(a)}, first difference at {same.index(False) if False in same else -1}")
# result 4: the set, lookup precision by match length
pr, rows, gap = load("deepseek", "eval"); tot = {1: [0, 0], 2: [0, 0], 3: [0, 0]}; ntok = npass = {1: 0, 2: 0, 3: 0}; ntok, npass = 0, {1: 0, 2: 0, 3: 0}
for p in sorted(set(rows[:, 0].tolist())):
    r = rows[rows[:, 0] == p]; out = r[np.argsort(r[:, 1]), 2].tolist(); st = steps(pr[p], out); ntok += len(out)
    for gm, ln, ok in st:
        if gm: tot[ln][0] += 1; tot[ln][1] += ok
    for k in (1, 2, 3): npass[k] += passes(out, st, k)
print(f"deepseek eval: prompts {len(pr)}, rows {len(rows)}; smallest top-2 gap {gap.min():.4f}; rows with gap < 0.1: {int((gap < 0.1).sum())}")
print("   lookup guesses by match length (guesses, right, precision): " + "; ".join(f"{k}: {g}, {h}, {h / max(g, 1):.3f}" for k, (g, h) in tot.items()))
print("   tokens per pass on the set, lookup alone: " + "; ".join(f"length >= {k}: {ntok / npass[k]:.3f}" for k in (1, 2, 3)))
