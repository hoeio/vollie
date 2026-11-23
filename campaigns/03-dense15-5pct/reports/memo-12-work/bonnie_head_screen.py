#!/usr/bin/env python3
"""memo-12 round 13 (Bonnie): would a cheap head screen find the right token? *host*, numpy, on Tom's dumps
(lab/c3/head/<m>/eval.{rows,h16}) and the served Q6_K head. No llama.cpp run. For every dumped row:
  exact logits = W x (served Q6_K weights, float activations); the target is their argmax.
  screens:  s1   1 bit per weight: sign(w) times the row's mean |w|              (memo-08's bit-plane idea, weight side)
            s1x  the same with x reduced to 3 bits per value (sign + 2 magnitude bits, one scale per vector)
            q4   a Q4_0 copy of the head (16 levels, one scale per 32 values; campaign 2's built two-stage head)
  recall@K = share of rows whose target is among the screen's K largest logits (K 1024, 2048, 8192)
  nll     = change of -log p(target) when the K best rows get exact logits and all other rows keep the screen's
Run: lab/venv-c3/bin/python3 -I -B bonnie_head_screen.py <repo root> [llama2|yi]
"""
import sys, os
import numpy as np
root = sys.argv[1]; which = sys.argv[2:] or ["llama2", "yi"]
sys.path.insert(0, os.path.join(root, "lab/llama.cpp-c3/gguf-py"))
from gguf import GGUFReader
from gguf.quants import dequantize
from gguf.constants import GGMLQuantizationType as T
M = {"llama2": "lab/c3/models/llama2-7b-q4_0.gguf", "yi": "lab/c3/models/yi15-6b-q4_0.gguf"}
KS = (1024, 2048, 8192)
def nll(lg, y):
    m = lg.max(1, keepdims=True); return (m[:, 0] + np.log(np.exp(lg - m).sum(1))) - lg[np.arange(len(y)), y]
def x3(x):                                    # 3 bits per activation: sign and 2 magnitude bits, one scale per vector (99.5th percentile)
    s = np.quantile(np.abs(x), 0.995, axis=1, keepdims=True) / 3.0
    return (np.sign(x) * np.clip(np.rint(np.abs(x) / s), 0, 3) * s).astype(np.float32)
for m in which:
    d = os.path.join(root, "lab/c3/head", m)
    rows = np.fromfile(os.path.join(d, "eval.rows"), dtype=np.int32).reshape(-1, 4)
    t = next(x for x in GGUFReader(os.path.join(root, M[m])).tensors if x.name == "output.weight")
    k, n = int(t.shape[0]), int(t.shape[1])
    H = np.memmap(os.path.join(d, "eval.h16"), dtype=np.float16, mode="r", shape=(len(rows), k))
    W = np.empty((n, k), np.float32); raw = np.asarray(t.data).reshape(-1).view(np.uint8).reshape(n, k // 256 * 210)
    for s in range(0, n, 4000): W[s:s + 4000] = dequantize(np.ascontiguousarray(raw[s:s + 4000]), T.Q6_K).reshape(-1, k)
    S1 = (np.sign(W) * np.abs(W).mean(1, keepdims=True)).astype(np.float32)
    b = W.reshape(n, -1, 32); a = np.abs(b).max(2, keepdims=True); sc = np.where(a > 0, a / 8.0, 1.0)
    Q4 = (np.clip(np.rint(b / sc), -8, 7) * sc).reshape(n, k).astype(np.float32); del b
    idx = np.arange(0, len(rows), 3)                       # every third row: about 1,200 rows
    res = {v: {K: [0, 0.0] for K in KS} for v in ("s1", "s1x", "q4")}; N = 0
    for s in range(0, len(idx), 128):
        x = np.asarray(H[idx[s:s + 128]], dtype=np.float32); ex = x @ W.T; y = ex.argmax(1); N += len(y); base = nll(ex.astype(np.float64), y)
        for v, lg in (("s1", x @ S1.T), ("s1x", x3(x) @ S1.T), ("q4", x @ Q4.T)):
            order = np.argsort(-lg, axis=1)[:, :max(KS)]
            rank = np.array([np.nonzero(order[i] == y[i])[0][0] if y[i] in order[i] else 10 ** 9 for i in range(len(y))])
            for K in KS:
                res[v][K][0] += int((rank < K).sum())
                mix = lg.astype(np.float64).copy(); r = np.arange(len(y))[:, None]; mix[r, order[:, :K]] = ex[r, order[:, :K]]
                res[v][K][1] += float((nll(mix, y) - base).sum())
    print(f"{m}: head {n} x {k}, rows scored {N}")
    for v, lab in (("s1", "1 bit per weight, float x"), ("s1x", "1 bit per weight, 3-bit x"), ("q4", "Q4_0 copy, float x")):
        print(f"   {lab:28}: " + "; ".join(f"K {K}: recall {res[v][K][0] / N:.4f}, d-nll {res[v][K][1] / N:+.4f}" for K in KS))
