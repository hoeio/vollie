"""memo-09 T1: single-guess accuracy of model-free guess sources for exact batch-2 lookahead. Model-generic, counting
only (no timing). Reads dumps of tools/c3_head_dump.cpp (.rows, .ptok, .topk).

At row j the model has produced y_j. The next pass feeds [y_j, g]; its first position is row j+1 and produces
y_{j+1}; g is a guess for y_{j+1}, and if it is right the second position gives y_{j+2} in the same pass. The guess may use the prompt,
y_0..y_j and row j's top-64 logits (the previous pass's output). Arms:
  a  prompt lookup: newest earlier match of the last 3, else 2, else 1 tokens of the context; propose its successor
  b  prompt bigram: newest successor of the last token inside the prompt only
  c  self-successor (Token Recycling, one node): the model's last output after the last token, from earlier rows
  d  priority a, then c, then b
  f  counted 3-gram table (fallback 2-gram) from the model's own greedy outputs on the training prompts (--table)
  t2 second-best token of row j's logits
  e  last-logit rescoring: of the candidates of a, b, c, f, the one ranked highest in row j's top-64; else d
  h  trained guess head (--head <pred.npy> from tools/c3_head_train.py), and ah: lookup a first, else the head
Printed per arm: coverage (share of guesses where the arm proposes one), accuracy on covered, p (correct over all
guesses), tau (committed tokens per pass in the simulated decode) and b2 (share of passes run at batch 2; a pass
without a candidate runs at batch 1). All *host*.

  python3 -I tools/c3_head_t1.py <dump prefix> [--table <train dump prefix>] [--name x]
"""
import argparse
from collections import Counter, defaultdict
import numpy as np


def load(prefix):
    rows = np.fromfile(prefix + '.rows', dtype=np.int32).reshape(-1, 4)
    topk = np.fromfile(prefix + '.topk', dtype=np.int32).reshape(len(rows), -1)
    raw = np.fromfile(prefix + '.ptok', dtype=np.int32)
    ptok, i = {}, 0
    while i < len(raw):
        p, n = int(raw[i]), int(raw[i + 1]); ptok[p] = raw[i + 2:i + 2 + n].tolist(); i += 2 + n
    seqs = {}
    for p in np.unique(rows[:, 0]):
        idx = np.where(rows[:, 0] == p)[0]
        idx = idx[np.argsort(rows[idx, 1])]
        seqs[int(p)] = (ptok[int(p)], rows[idx, 2].tolist(), topk[idx])
    return seqs


def head_by_prompt(rows_prefix, pred_path):
    rows = np.fromfile(rows_prefix + '.rows', dtype=np.int32).reshape(-1, 4)
    pred = np.load(pred_path)
    out = {}
    for p in np.unique(rows[:, 0]):
        idx = np.where(rows[:, 0] == p)[0]
        out[int(p)] = pred[idx[np.argsort(rows[idx, 1])]]
    return out


def table_from(prefix):
    rows = np.fromfile(prefix + '.rows', dtype=np.int32).reshape(-1, 4)
    rows = rows[np.lexsort((rows[:, 1], rows[:, 0]))]
    c3, c2 = defaultdict(Counter), defaultdict(Counter)
    for p in np.unique(rows[:, 0]):
        y = rows[rows[:, 0] == p, 2].tolist()
        for k in range(2, len(y)):
            c3[(y[k - 2], y[k - 1])][y[k]] += 1
        for k in range(1, len(y)):
            c2[y[k - 1]][y[k]] += 1
    best = lambda c: {k: v.most_common(1)[0][0] for k, v in c.items()}
    return best(c3), best(c2)


def lookup(ctx, nmax=3):
    for n in range(nmax, 0, -1):
        if len(ctx) <= n:
            continue
        suf = ctx[-n:]
        for s in range(len(ctx) - n - 1, -1, -1):
            if ctx[s:s + n] == suf:
                return ctx[s + n]
    return None


def guesses(prompt, y, topk, tab, head=None):
    """One dict of arm -> guess (or None) for every row j that has a row j+1."""
    out, succ = [], {}
    pb = {}
    for s in range(len(prompt) - 1):
        pb[prompt[s]] = prompt[s + 1]
    prev_in = prompt[-1]
    for j in range(len(y) - 1):
        succ[prev_in] = y[j]          # row j: input prev_in, output y[j]
        prev_in = y[j]
        ctx = prompt + y[:j + 1]
        g = {'a': lookup(ctx), 'b': pb.get(ctx[-1]), 'c': succ.get(ctx[-1])}  # succ holds rows 0..j only
        g['d'] = g['a'] if g['a'] is not None else g['c'] if g['c'] is not None else g['b']
        g['f'] = None
        if tab is not None:
            t3, t2 = tab
            g['f'] = t3.get((ctx[-2], ctx[-1])) if len(ctx) > 1 and (ctx[-2], ctx[-1]) in t3 else t2.get(ctx[-1])
        g['t2'] = int(topk[j][1])
        cands = {v for k, v in g.items() if k in ('a', 'b', 'c', 'f') and v is not None}
        rank = {int(t): r for r, t in enumerate(topk[j])}
        ranked = sorted((rank[c], c) for c in cands if c in rank)
        g['e'] = ranked[0][1] if ranked else g['d']
        if head is not None:
            g['h'] = int(head[j]) if head[j] >= 0 else None
            g['ah'] = g['a'] if g['a'] is not None else g['h']
        out.append(g)
    return out


def score(seqs, tab, arms=('a', 'b', 'c', 'd', 'f', 't2', 'e'), heads=None):
    res = {}
    for arm in arms:
        n = cov = ok = tok = passes = b2 = 0
        for p, (prompt, y, topk) in seqs.items():
            gs = guesses(prompt, y, topk, tab, heads[p] if heads else None)
            hit = [g[arm] is not None and g[arm] == y[j + 1] for j, g in enumerate(gs)]
            n += len(gs); cov += sum(g[arm] is not None for g in gs); ok += sum(hit)
            j = 0
            while j < len(y):           # row j's pass carries the guess for y[j+1]
                passes += 1
                has = j < len(gs) and gs[j][arm] is not None
                b2 += has
                j += 2 if (j < len(gs) and hit[j]) else 1
            tok += len(y)
        res[arm] = dict(n=n, coverage=cov / n, acc_covered=ok / max(cov, 1), p=ok / n, tau=tok / passes, b2=b2 / passes)
    return res


def selftest():
    assert lookup([1, 2, 3, 9, 1, 2, 3]) == 9 and lookup([5, 6]) is None and lookup([4, 7, 4]) == 7
    prompt, y = [1, 2, 3], [4, 1, 2, 3, 4]
    topk = np.array([[t, t + 1] + [0] * 62 for t in y])
    gs = guesses(prompt, y, topk, None)
    assert gs[1]['a'] == 2                      # context 1 2 3 4 1 -> lookup "1" -> 2
    r = score({0: (prompt, y, topk)}, None, arms=('a',))['a']
    assert 0 < r['p'] <= 1 and r['tau'] >= 1


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('dump'); ap.add_argument('--table'); ap.add_argument('--name', default=''); ap.add_argument('--head')
    a = ap.parse_args()
    selftest()
    tab = table_from(a.table) if a.table else None
    seqs = load(a.dump)
    heads = head_by_prompt(a.dump, a.head) if a.head else None
    arms = ('a', 'b', 'c', 'd', 'f', 't2', 'e') + (('h', 'ah') if heads else ())
    for arm, r in score(seqs, tab, arms, heads).items():
        if arm == 'f' and tab is None:
            continue
        print(f'T1 [{a.name}] arm {arm:2s} guesses {r["n"]:5d} coverage {r["coverage"]:.3f} acc_covered {r["acc_covered"]:.3f} '
              f'p {r["p"]:.3f} tau {r["tau"]:.3f} b2 {r["b2"]:.3f}')
