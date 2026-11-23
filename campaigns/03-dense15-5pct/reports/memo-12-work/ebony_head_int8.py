#!/usr/bin/env python3
"""memo-12 round 7 (Ebony): (a) is the cross-path KL divergence of r6 the head's doing? (b) does a trained guess head
score the same when the head is the phone's (KleidiAI: int8 weights with one scale per row, asymmetric int8 activations
with one scale and zero point per vector) as when it is the file's Q6_K head in float?  *host*, numpy; a PROXY on Tom's
32-prompt dumps (float matmul, quantisers simulated from kleidiai.cpp lines 1515-1580 and kai_lhs_quant_pack_qai8dxp_f32.c).
Written independently of bonnie_head_paths.py; the two are compared in ebony-joins-r7.md.
Run: lab/venv-c3/bin/python3 -I -B ebony_head_int8.py <repo root> <llama2|yi> <head .pt, Medusa-1 full rank>
"""
import sys, os
import numpy as np
root, m, pt = sys.argv[1:4]
sys.path.insert(0, os.path.join(root, "lab/llama.cpp-c3/gguf-py"))
from gguf import GGUFReader
from gguf.quants import dequantize
import torch
G = {"llama2": "lab/c3/models/llama2-7b-q4_0.gguf", "yi": "lab/c3/models/yi15-6b-q4_0.gguf"}[m]
def q_kai(x):   # one scale, one nudged zero point per vector; round half away from zero (vcvta); clamp to int8
    lo = np.minimum(0.0, x.min(1, keepdims=True)); hi = np.maximum(0.0, x.max(1, keepdims=True))
    sc = np.where(hi > lo, 255.0 / np.where(hi > lo, hi - lo, 1), 1.0).astype(np.float32)
    zp = np.where((-128 + lo * sc) + (127 + hi * sc) > 0, -128 - lo * sc, 127 - hi * sc)
    zp = np.rint(np.clip(zp, -128, 127)); v = x * sc
    q = np.clip(np.sign(v) * np.floor(np.abs(v) + 0.5) + zp, -128, 127)
    return ((q - zp) / sc).astype(np.float32)
def q_8k(x):    # ggml's Q8_K: one scale per 256 values
    b = x.reshape(len(x), -1, 256); a = np.abs(b).max(2, keepdims=True); s = np.where(a > 0, a / 127.0, 1.0)
    return (np.rint(b / s) * s).reshape(x.shape).astype(np.float32)
def lsm(l): l = l.astype(np.float64); mx = l.max(1, keepdims=True); return l - mx - np.log(np.exp(l - mx).sum(1, keepdims=True))
assert np.allclose(q_kai(np.array([[0.0, 1.0, -1.0, 0.5]], np.float32)), [[0, 1, -1, 0.5]], atol=2 / 255)   # self-check
d = os.path.join(root, "lab/c3/head", m)
rows = np.fromfile(os.path.join(d, "eval.rows"), dtype=np.int32).reshape(-1, 4)
t = next(x for x in GGUFReader(os.path.join(root, G)).tensors if x.name == "output.weight")
k, n = int(t.shape[0]), int(t.shape[1])
H = np.asarray(np.memmap(os.path.join(d, "eval.h16"), dtype=np.float16, mode="r", shape=(len(rows), k)), dtype=np.float32)
raw = np.asarray(t.data).reshape(-1).view(np.uint8).reshape(n, -1); W = np.empty((n, k), np.float32)
for s in range(0, n, 4000): W[s:s + 4000] = dequantize(np.ascontiguousarray(raw[s:s + 4000]), t.tensor_type).reshape(-1, k)
mx = np.abs(W).max(1, keepdims=True); sc = np.where(mx > 0, mx / 127.0, 1.0); Wk = (np.clip(np.rint(W / sc), -127, 127) * sc).astype(np.float32)
sd = torch.load(pt, map_location="cpu", weights_only=False); Wb = sd["W"].numpy().astype(np.float32); bb = sd["b"].numpy().astype(np.float32)
assert sd.get("Wt") is None, "use a Medusa-1 (no tok) head"
lin = H @ Wb.T + bb; Gs = H + lin / (1 + np.exp(-lin))                                   # g(h) = h + SiLU(W h + b)
o = np.lexsort((rows[:, 1], rows[:, 0])); ok = (rows[o][1:, 0] == rows[o][:-1, 0]) & (rows[o][1:, 1] == rows[o][:-1, 1] + 1)
J, J1 = o[:-1][ok], o[1:][ok]
am = lambda X, WW: np.concatenate([(X[s:s + 512] @ WW.T).argmax(1) for s in range(0, len(X), 512)])
# (a) head-only divergence between the two paths, on the true hidden states
kl = []; same = 0; dn = []; rr = []
for s in range(0, len(H), 256):
    x = H[s:s + 256]; a, b = lsm(q_8k(x) @ W.T), lsm(q_kai(x) @ Wk.T)
    kl += list((np.exp(a) * (a - b)).sum(1)); same += int((a.argmax(1) == b.argmax(1)).sum())
    yy = rows[s:s + 256, 2]; i = np.arange(len(yy)); dn += list(a[i, yy] - b[i, yy]); rr += list((x.max(1) - x.min(1)) / np.sqrt((x ** 2).mean(1)))
kl = np.array(kl)
print(f"{m}: rows {len(H)}, head {n} x {k}; base check argmax(float head) == dumped token: {(am(H, W) == rows[:, 2]).mean():.4f}")
print(f"      change of -log p(dumped token), KleidiAI path - ggml path: {np.mean(dn):+.5f} (s.e. {np.std(dn) / len(dn) ** .5:.5f}); hidden state range / rms median {np.median(rr):.1f}")
print(f"  (a) head only, ggml path against KleidiAI path: mean KLD {kl.mean():.6f}, 99.9 % {np.quantile(kl, .999):.5f}, max {kl.max():.5f}, same top-1 {same / len(H):.4f}")
# (b) the guess head against each head
y_host = rows[:, 2]; y_ph = am(q_kai(H), Wk)                                             # the token each path's head picks from the same state
g_host = am(Gs, W); g_ph = am(q_kai(Gs), Wk)
print(f"  (b) pairs {len(J)}; phone head picks the dumped token on {(y_ph == y_host).mean():.4f} of rows")
print(f"      guess top-1: float head, dumped targets {(g_host[J] == y_host[J1]).mean():.4f} | phone head for guess and target {(g_ph[J] == y_ph[J1]).mean():.4f}"
      f" | float guess, phone target {(g_host[J] == y_ph[J1]).mean():.4f} | guesses equal on {(g_host == g_ph).mean():.4f} of rows")
