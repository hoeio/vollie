#!/usr/bin/env python3
"""memo-12 round 6 (Bonnie): what Tom's head dumps say about depth 2. *host*, numpy only, no training.
Reads lab/c3/head/<m>/{eval,train*}.rows (int32 x 4: prompt, step, greedy token, -) and the saved head guesses
<head>.pt.evalset.pred.npy (guess made at row j for row j+1's token; -1 where no pair).
Run: lab/venv-c3/bin/python3 -I -B bonnie_depth2.py <repo root>
Prints, per model, on the 32-prompt set:
  a1      head top-1 (must reproduce tom-head.md)
  clus    P(hit at j+1 | hit at j) and P(hit at j+1 | miss at j): how hits cluster
  big     a model-free depth-2 guess: the most frequent successor (training rows) of the head's depth-1 guess
  ceil    P(the head, given the true next state, is right at j+1 | right at j): no depth-2 guess made from row j can know more
"""
import sys, os, collections
import numpy as np
root = sys.argv[1]
M = {"Llama-2": ("llama2", "head3000-tok.pt", ["train3000"]), "Yi": ("yi", None, None)}
def seqs(rows):
    order = np.lexsort((rows[:, 1], rows[:, 0])); out = collections.OrderedDict()
    for i in order: out.setdefault(int(rows[i, 0]), []).append(int(i))
    return out
for name, (d, head, trains) in M.items():
    base = os.path.join(root, "lab/c3/head", d)
    if head is None:
        c = sorted(f for f in os.listdir(base) if f.endswith(".evalset.pred.npy") and "tok" in f)
        head = c[-1][:-len(".evalset.pred.npy")]; trains = sorted({f[:-5] for f in os.listdir(base) if f.startswith("train") and f.endswith(".rows")})
    rows = np.fromfile(os.path.join(base, "eval.rows"), dtype=np.int32).reshape(-1, 4)
    pred = np.load(os.path.join(base, head + ".evalset.pred.npy"))
    big = collections.defaultdict(collections.Counter); ntr = 0
    for t in trains:
        tr = np.fromfile(os.path.join(base, t + ".rows"), dtype=np.int32).reshape(-1, 4)
        for s in seqs(tr).values():
            tk = tr[s, 2]; ntr += len(tk)
            for a, b in zip(tk[:-1].tolist(), tk[1:].tolist()): big[a][b] += 1
    succ = {a: c.most_common(1)[0][0] for a, c in big.items()}
    n1 = h1 = n2 = hh = nm = mh = nb = bh = nbc = bch = 0; gain_chain = gain_n = 0
    for s in seqs(rows).values():
        tok = rows[s, 2]; g = pred[s]; L = len(s)
        hit = [(j + 1 < L and g[j] >= 0 and g[j] == tok[j + 1]) for j in range(L)]
        for j in range(L - 1):
            if g[j] < 0: continue
            n1 += 1; h1 += hit[j]
            if j + 2 < L and g[j + 1] >= 0:
                if hit[j]: n2 += 1; hh += hit[j + 1]
                else: nm += 1; mh += hit[j + 1]
                g2 = succ.get(int(g[j]), -1); ok2 = g2 == tok[j + 2]
                nb += 1; bh += bool(hit[j] and ok2)
                if hit[j]: nbc += 1; bch += bool(ok2)
    a1 = h1 / n1
    print(f"{name}: head {head}, train sets {trains} ({ntr} rows), pairs {n1}")
    print(f"  a1 {a1:.3f} | P(hit j+1 | hit j) {hh / n2:.3f} (n {n2}), P(hit j+1 | miss j) {mh / nm:.3f} (n {nm})")
    print(f"  bigram depth 2: P(right | depth 1 right) {bch / nbc:.3f} (n {nbc}); C2 = that / a1 = {bch / nbc / a1:.2f}; path prob {bh / nb:.3f}")
    print(f"  ceiling: C2 <= {hh / n2 / a1:.2f}; chain-of-2 path sum: bigram {a1 + bh / nb:.3f}, ceiling {a1 + a1 * hh / n2:.3f}")
