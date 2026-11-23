#!/usr/bin/env python3
"""memo-12 round 10 (Ebony): a repair of the head inside KleidiAI's 8-bit chain. The k dimensions with the largest
final-norm gain |output_norm.weight| (known at load, no data needed) are set to zero before the one-scale int8
quantiser and added back in float with the file's own Q6_K column values. *host*, PROXY on Tom's 32-prompt dumps
(float matmul, quantisers as in ebony_head_int8.py). Also prints the gain ranking for a model without a dump.
Run: lab/venv-c3/bin/python3 -I -B ebony_head_outliers.py <repo root>
"""
import sys, os
import numpy as np
root = sys.argv[1]; sys.path.insert(0, os.path.join(root, "lab/llama.cpp-c3/gguf-py"))
from gguf import GGUFReader
from gguf.quants import dequantize
def q_kai(x):
    lo = np.minimum(0.0, x.min(1, keepdims=True)); hi = np.maximum(0.0, x.max(1, keepdims=True))
    sc = np.where(hi > lo, 255.0 / np.where(hi > lo, hi - lo, 1), 1.0).astype(np.float32)
    zp = np.where((-128 + lo * sc) + (127 + hi * sc) > 0, -128 - lo * sc, 127 - hi * sc); zp = np.rint(np.clip(zp, -128, 127)); v = x * sc
    return ((np.clip(np.sign(v) * np.floor(np.abs(v) + 0.5) + zp, -128, 127) - zp) / sc).astype(np.float32)
def q_8k(x):
    b = x.reshape(len(x), -1, 256); a = np.abs(b).max(2, keepdims=True); s = np.where(a > 0, a / 127.0, 1.0)
    return (np.rint(b / s) * s).reshape(x.shape).astype(np.float32)
def lsm(l): l = l.astype(np.float64); m = l.max(1, keepdims=True); return l - m - np.log(np.exp(l - m).sum(1, keepdims=True))
z = np.zeros((1, 8), np.float32); z[0, 3] = 5; assert q_kai(z)[0, 0] == 0.0          # zero is represented exactly
for m, g in (("yi", "yi15-6b"), ("llama2", "llama2-7b"), ("deepseek", "deepseek7b")):
    rd = GGUFReader(os.path.join(root, f"lab/c3/models/{g}-q4_0.gguf"))
    gain = np.abs(np.asarray(next(x for x in rd.tensors if x.name == "output_norm.weight").data, dtype=np.float32)); order = np.argsort(-gain)
    print(f"{m}: |output_norm.weight| median {np.median(gain):.2f}; largest: " + ", ".join(f"{int(d)}: {gain[d]:.1f}" for d in order[:8])
          + f"; dims above 4 x median: {int((gain > 4 * np.median(gain)).sum())}, above 2 x: {int((gain > 2 * np.median(gain)).sum())}")
    d = os.path.join(root, "lab/c3/head", m)
    if not os.path.exists(os.path.join(d, "eval.rows")): print("   no dump"); continue
    rows = np.fromfile(os.path.join(d, "eval.rows"), dtype=np.int32).reshape(-1, 4)
    t = next(x for x in rd.tensors if x.name == "output.weight"); k, n = int(t.shape[0]), int(t.shape[1])
    H = np.asarray(np.memmap(os.path.join(d, "eval.h16"), dtype=np.float16, mode="r", shape=(len(rows), k)), dtype=np.float32)
    raw = np.asarray(t.data).reshape(-1).view(np.uint8).reshape(n, -1); W = np.empty((n, k), np.float32)
    for s in range(0, n, 4000): W[s:s + 4000] = dequantize(np.ascontiguousarray(raw[s:s + 4000]), t.tensor_type).reshape(-1, k)
    mx = np.abs(W).max(1, keepdims=True); sc = np.where(mx > 0, mx / 127.0, 1.0); Wk = (np.clip(np.rint(W / sc), -127, 127) * sc).astype(np.float32)
    A = np.abs(H) / np.sqrt((H ** 2).mean(1, keepdims=True)); data_top = np.argsort(-A.mean(0))[:8]
    print(f"   largest dims by data (mean |x| / rms): {[int(x) for x in data_top]}; of these in the gain's top 8: {len(set(data_top.tolist()) & set(order[:8].tolist()))}, top 32: {len(set(data_top.tolist()) & set(order[:32].tolist()))}")
    y = rows[:, 2]; base = 0.0; res = {}
    for kk in (0, 2, 4, 8, 16, 32):
        D = order[:kk]; tot = 0.0; er = []
        for s in range(0, len(H), 256):
            x = H[s:s + 256]; xr = x.copy(); xr[:, D] = 0.0; xq = q_kai(xr)
            lg = xq @ Wk.T + x[:, D] @ W[:, D].T; ls = lsm(lg); i = np.arange(len(x)); tot += float(-ls[i, y[s:s + 256]].sum())
            xq[:, D] = x[:, D]; er += list(np.sqrt(((xq - x) ** 2).mean(1)) / np.sqrt((x ** 2).mean(1)))
            if kk == 0: base += float(-lsm(q_8k(x) @ W.T)[i, y[s:s + 256]].sum())
        res[kk] = (tot / len(H), float(np.median(er)))
    b = base / len(H)
    print("   k float dims: change of -log p(dumped token) against ggml's path; activation error / rms")
    print("   " + "; ".join(f"k {kk}: {v[0] - b:+.5f}, {v[1]:.4f}" for kk, v in res.items()))
