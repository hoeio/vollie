#!/usr/bin/env python3
"""memo-12 round 11 (Bonnie): the output head on the phone's path (KleidiAI) against ggml's path, on real hidden states.
*host*, numpy; reads Tom's dumps lab/c3/head/<m>/eval.{rows,h16} (the post-norm hidden state each row feeds to the head,
and the greedy token) and the served file's Q6_K head. No llama.cpp run. A PROXY for the head's share of a perplexity
shift: the mean change of -log p(greedy token) over the 32-prompt set.
  ref   served Q6_K weights, float activations
  ggml  served Q6_K weights, activations as Q8_K (int8, one scale per 256 values; ggml's K-quant dot products)
  kai   weights re-quantised to int8 with one scale per row (max / 127); activations asymmetric int8 with ONE scale and
        zero point per vector (kai_run_lhs_quant_pack_qai8dxp_f32), as KleidiAI's 8-bit chain does
  kai_w / kai_a  only the weight part / only the activation part of kai
  q80   served Q6_K weights, activations as Q8_0 (one scale per 32 values), for the screening stage of a two-stage head
Run: lab/venv-c3/bin/python3 -I -B bonnie_head_paths.py <repo root> [llama2|yi]
"""
import sys, os
import numpy as np
root = sys.argv[1]; which = sys.argv[2:] or ["llama2", "yi"]
sys.path.insert(0, os.path.join(root, "lab/llama.cpp-c3/gguf-py"))
from gguf import GGUFReader
from gguf.quants import dequantize
from gguf.constants import GGMLQuantizationType as T
M = {"llama2": "lab/c3/models/llama2-7b-q4_0.gguf", "yi": "lab/c3/models/yi15-6b-q4_0.gguf", "deepseek": "lab/c3/models/deepseek7b-q4_0.gguf"}
def act_q8k(x):                      # per 256 values: scale amax / 127, round to nearest
    b = x.reshape(x.shape[0], -1, 256); a = np.abs(b).max(2, keepdims=True); s = np.where(a > 0, a / 127.0, 1.0)
    return (np.rint(b / s) * s).reshape(x.shape).astype(np.float32)
def act_q80(x):                      # per 32 values (Q8_0): what a Q4_0 screening copy of the head multiplies by (C7)
    b = x.reshape(x.shape[0], -1, 32); a = np.abs(b).max(2, keepdims=True); s = np.where(a > 0, a / 127.0, 1.0)
    return (np.rint(b / s) * s).reshape(x.shape).astype(np.float32)
def act_kai(x):                      # per vector: asymmetric, 255 steps over [min(0, min), max(0, max)], nudged zero point
    lo = np.minimum(0, x.min(1, keepdims=True)); hi = np.maximum(0, x.max(1, keepdims=True))
    sc = np.where(hi > lo, 255.0 / (hi - lo), 1.0)
    zp = np.where((-128 + lo * sc) + (127 + hi * sc) > 0, -128 - lo * sc, 127 - hi * sc); zp = np.rint(np.clip(zp, -128, 127))
    q = np.clip(np.rint(x * sc) + zp, -128, 127)
    return ((q - zp) / sc).astype(np.float32)
def nll(lg, y):
    m = lg.max(1, keepdims=True); return (m[:, 0] + np.log(np.exp(lg - m).sum(1))) - lg[np.arange(len(y)), y]
for m in which:
    d = os.path.join(root, "lab/c3/head", m)
    rows = np.fromfile(os.path.join(d, "eval.rows"), dtype=np.int32).reshape(-1, 4)
    t = next(x for x in GGUFReader(os.path.join(root, M[m])).tensors if x.name == "output.weight")
    k, n = int(t.shape[0]), int(t.shape[1]); assert t.tensor_type == T.Q6_K
    H = np.memmap(os.path.join(d, "eval.h16"), dtype=np.float16, mode="r", shape=(len(rows), k))
    W = np.empty((n, k), np.float32); raw = np.asarray(t.data).reshape(-1).view(np.uint8).reshape(n, k // 256 * 210)
    for s in range(0, n, 4000): W[s:s + 4000] = dequantize(np.ascontiguousarray(raw[s:s + 4000]), T.Q6_K).reshape(-1, k)
    mx = np.abs(W).max(1, keepdims=True); sc = np.where(mx > 0, mx / 127.0, 1.0); Wk = (np.clip(np.rint(W / sc), -127, 127) * sc).astype(np.float32)
    y = rows[:, 2].astype(np.int64); acc = {}; tot = {v: 0.0 for v in ("ref", "ggml", "kai", "kai_w", "kai_a", "q80")}; e3 = []; hit = 0; N = 0; cr = []; ea = []; eg = []
    for s in range(0, len(rows), 256):
        x = np.asarray(H[s:s + 256], dtype=np.float32); yy = y[s:s + 256]
        xg, xk = act_q8k(x), act_kai(x); r = np.sqrt((x ** 2).mean(1)); x3 = act_q80(x); e3 += list(np.sqrt(((x3 - x) ** 2).mean(1)) / r)
        cr += list((x.max(1) - x.min(1)) / r); ea += list(np.sqrt(((xk - x) ** 2).mean(1)) / r); eg += list(np.sqrt(((xg - x) ** 2).mean(1)) / r)
        lr = x @ W.T; hit += int((lr.argmax(1) == yy).sum()); N += len(yy)
        for v, lg in (("ref", lr), ("ggml", xg @ W.T), ("kai", xk @ Wk.T), ("kai_w", xg @ Wk.T), ("kai_a", xk @ W.T), ("q80", x3 @ W.T)): tot[v] += float(nll(lg.astype(np.float64), yy).sum())
    cr, ea, eg = map(np.array, (cr, ea, eg))
    print(f"{m}: rows {N}, head {n} x {k}; self-check argmax(ref) == greedy token: {hit / N:.4f}")
    print(f"   hidden state range / rms: median {np.median(cr):.1f}, p90 {np.quantile(cr, .9):.1f}, max {cr.max():.1f}")
    print(f"   activation error / rms: ggml Q8_K median {np.median(eg):.4f}; KleidiAI per-vector median {np.median(ea):.4f}, p90 {np.quantile(ea, .9):.4f}")
    b = tot["ggml"] / N
    print(f"   blocks of 32 (Q8_0): activation error / rms median {np.median(np.array(e3)):.4f}; -log p(greedy) {tot['q80'] / N - b:+.5f} against ggml")
    print(f"   mean -log p(greedy): ref {tot['ref'] / N:.5f}; ggml {b:.5f}; kai {tot['kai'] / N:.5f} ({tot['kai'] / N - b:+.5f} against ggml);"
          f" weights only {tot['kai_w'] / N - b:+.5f}; activations only {tot['kai_a'] / N - b:+.5f}")
