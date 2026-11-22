#!/usr/bin/env python3
"""memo-11 (Ebony), host check E-T1: is a retrieval store of the model's own outputs, keyed by the hidden state, a
better single guess than Tom's trained head, or a useful second opinion? Every number printed is *host*.

Store: every training pair (row j, row j+1) of Tom's greedy dumps: key = post-norm hidden state of row j (unit
length), value = greedy token of row j+1 (the token a lookahead pass has to guess at row j).
Query: the prompt-set dump (eval), never in the store (tools/c3_head_prompts.py split).
Arms: nn1 = value of the nearest key; nnt = nearest key among rows that produced the same token y_j (else nn1);
vote = similarity-weighted vote of the 16 nearest; head = Tom's saved head guess; union = head or nnt right (upper
bound of any chooser); gate = nnt when its similarity >= theta else head, theta fitted on even prompts, scored on odd.

  python3 ebony_knn.py <repo root> <model dir under lab/c3/head> <train prefix> <head pred .npy> <n_embd>
"""
import sys, os, time
import numpy as np

def unit(x):
    x = np.asarray(x, dtype=np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-6)

def topk_search(Q, H, idx, k=16, chunk=40000):
    """Cosine top-k of each query row against H[idx] (memmap, fp16), in chunks. Returns (sims, positions in idx)."""
    best_s = np.full((len(Q), k), -2.0, dtype=np.float32); best_i = np.zeros((len(Q), k), dtype=np.int64)
    for s in range(0, len(idx), chunk):
        part = idx[s:s + chunk]
        S = Q @ unit(H[part]).T
        kk = min(k, S.shape[1])
        loc = np.argpartition(-S, kk - 1, axis=1)[:, :kk]
        cs = np.take_along_axis(S, loc, axis=1)
        allS = np.concatenate([best_s, cs], axis=1); allI = np.concatenate([best_i, loc + s], axis=1)
        o = np.argsort(-allS, axis=1)[:, :k]
        best_s, best_i = np.take_along_axis(allS, o, axis=1), np.take_along_axis(allI, o, axis=1)
    return best_s, best_i

def selftest():
    rng = np.random.default_rng(0)
    H = rng.standard_normal((500, 8)).astype(np.float16); Q = unit(H[[3, 77, 400]])
    s, i = topk_search(Q, H, np.arange(500), k=4, chunk=128)
    assert list(i[:, 0]) == [3, 77, 400] and np.all(s[:, 0] > 0.999) and np.all(np.diff(s, axis=1) <= 1e-6)

if __name__ == "__main__":
    selftest()
    root, mdl, trainp, predf, n_embd = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5])
    sys.path.insert(0, os.path.join(root, "tools"))
    import c3_head_train as ht
    d = os.path.join(root, "lab/c3/head", mdl)
    rows, H = ht.load_dump(os.path.join(d, trainp), n_embd)
    ja, jb = ht.pairs(rows)
    val = rows[jb, 2]; ytok = rows[ja, 2]
    er, EH = ht.load_dump(os.path.join(d, "eval"), n_embd)
    ia, ib = ht.pairs(er)
    truth = er[ib, 2]; qtok = er[ia, 2]
    head = np.load(os.path.join(d, predf))[ia]
    assert (head >= 0).all()
    t0 = time.time()
    Q = unit(EH[ia])
    s, i = topk_search(Q, H, ja, k=16)
    nn1 = val[i[:, 0]]
    # similarity-weighted vote of 16 (weights exp(sim / 0.05))
    vote = np.empty(len(Q), dtype=np.int64)
    for r in range(len(Q)):
        w = {}
        for sim, pos in zip(s[r], i[r]): w[int(val[pos])] = w.get(int(val[pos]), 0.0) + float(np.exp(sim / 0.05))
        vote[r] = max(w, key=w.get)
    # token-conditioned nearest neighbour
    nnt, nnt_s = nn1.copy(), s[:, 0].copy()
    by_tok = {}
    for p, t in enumerate(ytok): by_tok.setdefault(int(t), []).append(p)
    cover = 0
    for t in np.unique(qtok):
        cand = by_tok.get(int(t))
        if not cand: continue
        qs = np.nonzero(qtok == t)[0]
        st, it = topk_search(Q[qs], H, ja[np.array(cand)], k=1)
        nnt[qs] = val[np.array(cand)[it[:, 0]]]; nnt_s[qs] = st[:, 0]; cover += len(qs)
    def rep(name, guess, mask=None):
        m = np.ones(len(truth), bool) if mask is None else mask
        hit = dict(zip(ia[m].tolist(), (guess[m] == truth[m]).tolist()))
        only = set(er[ia[m], 0].tolist())
        print(f"  {name:28s} top-1 {np.mean(guess[m] == truth[m]):.3f}   tokens per pass {ht.tau(er, hit, only):.3f}   pairs {m.sum()}")
    print(f"{mdl}: store {len(ja)} pairs from {trainp}, queries {len(ia)} (prompt set), search {time.time()-t0:.0f} s; "
          f"same-token rows exist for {cover/len(ia):.3f} of queries")
    rep("head (Tom, tok)", head); rep("nn1", nn1); rep("vote16", vote); rep("nnt (same token)", nnt)
    print(f"  {'union head|nnt (upper bound)':28s} top-1 {np.mean((head == truth) | (nnt == truth)):.3f}")
    print(f"  agreement head == nnt: {np.mean(head == nnt):.3f}; head right when they agree {np.mean((head == truth)[head == nnt]):.3f}; "
          f"head right when they differ {np.mean((head == truth)[head != nnt]):.3f}; nnt right when they differ {np.mean((nnt == truth)[head != nnt]):.3f}")
    even = (er[ia, 0] % 2 == 0)
    best = max(((np.mean(np.where(nnt_s[even] >= th, nnt[even], head[even]) == truth[even]), th) for th in np.arange(0.50, 1.001, 0.02)))
    gate = np.where(nnt_s >= best[1], nnt, head)
    print(f"  gate theta {best[1]:.2f} (fitted on even prompts; nnt used in {np.mean(nnt_s[~even] >= best[1]):.3f} of odd-prompt rows)")
    rep("head, odd prompts", head, ~even); rep("gate, odd prompts", gate, ~even)
    for lo, hi in ((0.0, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.01)):
        m = (nnt_s >= lo) & (nnt_s < hi)
        if m.sum(): print(f"    nnt similarity {lo:.1f} to {hi:.1f}: share {m.mean():.3f}, nnt top-1 {np.mean((nnt == truth)[m]):.3f}, head top-1 {np.mean((head == truth)[m]):.3f}")
