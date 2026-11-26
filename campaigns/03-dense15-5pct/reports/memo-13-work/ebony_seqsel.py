#!/usr/bin/env python3
"""memo-13 (Ebony): host screen of per-sequence FFN neuron selection (GRIFFIN-style) on a served Q4_0 file.

A numpy float32 forward pass of a llama-architecture GGUF (weights dequantised from the served file, activations
in float: NOT llama.cpp's numerics, a screen only). Protocol = llama-perplexity's: chunks of 512 tokens, the first
256 are context (run dense, they also give the selection statistic), tokens 257..511 are scored.

usage: ebony_seqsel.py <model.gguf> <token-ids.txt from llama-tokenize --ids> <n_chunks> [dense|nn]
Prints per-variant PPL and the ratio to this script's own dense PPL. Label of every output: *host*.
"""
import os, sys, re, time, numpy as np, gguf
from gguf.quants import dequantize

path, tokf, N = sys.argv[1], sys.argv[2], int(sys.argv[3])
only_dense = len(sys.argv) > 4
nn_mode = len(sys.argv) > 4 and sys.argv[4] == 'nn'      # dense only, plus nearest-earlier-position distances
nnst = []
C, H2 = 512, 256
r = gguf.GGUFReader(path)
T = {t.name: t for t in r.tensors}
def meta(k): return r.fields[k].parts[-1][0]
L, d, nh, nkv = int(meta('llama.block_count')), int(meta('llama.embedding_length')), int(meta('llama.attention.head_count')), int(meta('llama.attention.head_count_kv'))
eps, base = float(meta('llama.attention.layer_norm_rms_epsilon')), float(meta('llama.rope.freq_base')) if 'llama.rope.freq_base' in r.fields else 10000.0
add_bos = bool(meta('tokenizer.ggml.add_bos_token')) if 'tokenizer.ggml.add_bos_token' in r.fields else True
bos = int(meta('tokenizer.ggml.bos_token_id'))
hd = d // nh
SCERR = []
SC8 = bool(os.environ.get('SEQSEL_SCALE8'))         # body block scales through a 256-entry codebook per tensor
def W(name):
    t = T[name]; data = np.asarray(t.data)
    if SC8 and name.startswith('blk.') and t.tensor_type.name == 'Q4_0':
        data = data.copy(); blk = data.reshape(data.shape[0], -1, 18)
        sc = blk[:, :, :2].copy().view(np.float16)[:, :, 0].astype(np.float32)
        lg = np.log(np.maximum(np.abs(sc), 1e-30)).ravel()        # sign bit + 127 Lloyd levels of log|scale|
        cen = np.quantile(lg, (np.arange(127) + 0.5) / 127)
        for _ in range(12):
            idx = np.searchsorted((cen[1:] + cen[:-1]) / 2, lg)
            sm, ct = np.bincount(idx, lg, 127), np.bincount(idx, minlength=127)
            cen = np.sort(np.where(ct > 0, sm / np.maximum(ct, 1), cen))
        idx = np.searchsorted((cen[1:] + cen[:-1]) / 2, lg)
        rec = (np.sign(sc).ravel() * np.exp(cen[idx])).reshape(sc.shape).astype(np.float32)
        SCERR.append(float(np.sqrt((((rec - sc) / np.where(sc == 0, 1, sc)) ** 2).mean())))
        blk[:, :, :2] = rec.astype(np.float16)[..., None].view(np.uint8)
    return dequantize(data, t.tensor_type).astype(np.float32, copy=False)

ids = np.array([int(x) for x in re.findall(r'-?\d+', open(tokf).read())], dtype=np.int64)
toks = ids[:N * C].reshape(N, C).copy()
if add_bos: toks[:, 0] = bos

def norm(x, w): return x * (w / np.sqrt((x * x).mean(-1, keepdims=True) + eps)).astype(np.float32)
def mm(x, w): return (x.reshape(-1, x.shape[-1]) @ w.T).reshape(*x.shape[:-1], w.shape[0])
inv = (base ** (-np.arange(0, hd, 2) / hd)).astype(np.float64)
def rope(x, pos):                                  # x (N, T, heads, hd), adjacent pairs, as ggml_rope_ext mode 0
    a = pos[:, None] * inv[None, :]
    c, s = np.cos(a).astype(np.float32)[None, :, None, :], np.sin(a).astype(np.float32)[None, :, None, :]
    x0, x1 = x[..., 0::2], x[..., 1::2]
    o = np.empty_like(x); o[..., 0::2] = x0 * c - x1 * s; o[..., 1::2] = x0 * s + x1 * c
    return o
def attn(q, k, v, off, kmask=None):                            # q (N,Tq,nh,hd) at positions off..; k, v (N,Tk,nkv,hd)
    rep = nh // nkv
    k = np.repeat(k, rep, axis=2); v = np.repeat(v, rep, axis=2)
    s = np.einsum('nqhd,nkhd->nhqk', q, k, optimize=True) / np.sqrt(hd)
    Tq, Tk = q.shape[1], k.shape[1]
    s += np.where(np.arange(Tk)[None, :] > (np.arange(Tq)[:, None] + off), -np.inf, 0).astype(np.float32)
    if kmask is not None: s += kmask[None, None, None, :]
    s -= s.max(-1, keepdims=True); np.exp(s, out=s); s /= s.sum(-1, keepdims=True)
    return np.einsum('nhqk,nkhd->nqhd', s, v, optimize=True).reshape(q.shape[0], Tq, d)
def silu(x): return x / (1 + np.exp(-x))

# variants: name -> (kind, keep fraction, extra)
V = {'dense': ('dense', 1.0, 0)}
if not only_dense:
    V.update({'G125': ('seq', .875, 256), 'G25': ('seq', .75, 256), 'G50': ('seq', .5, 256),
              'G25p64': ('seq', .75, 64), 'S25': ('glob', .75, 256),
              # per-token rules: D = by |h| (needs gate and up, saves down columns only), U = by |up| (saves gate rows
              # and down columns), T = by |silu(gate)| (CATS; saves up rows and down columns)
              'D50': ('dyn', .5, 0), 'D65': ('dyn', .35, 0),
              # two-stage (Prox-style): rank channels by a 1-bit or 2-bit copy of gate and up, then run only the kept
              # channels exactly. X<bits>_<percent skipped>
              # UD: up first, skip 30 % of the channels by |up| (gate rows), then keep the top half of all channels by |h| (down columns)
              'UD': ('ud', .5, 0),
              'X2_50': ('prox', .5, 2), 'T40': ('gate', .6, 0), 'X1_50': ('prox', .5, 1), 'X1_65': ('prox', .35, 1), 'X2_65': ('prox', .35, 2),
              # X1q: the 1-bit copy read with the FFN input cut to 7 levels (sign and two bit-planes: a popcount kernel)
              'X1q_50': ('prox', .5, 11), 'X1q_65': ('prox', .35, 11), 'U30': ('up', .7, 0), 'U50': ('up', .5, 0), 'T30': ('gate', .7, 0), 'T50': ('gate', .5, 0),
              # cross-token reuse (LoRA-Drop without the LoRA): on odd scored positions the middle M layers are not run;
              # R = their residual update is copied from the previous position, K = it is zero (plain skip control)
              'R8': ('reuse', 1.0, 8), 'K8': ('skip', 1.0, 8)})
def stale(kind, P, l): return kind in ('reuse', 'skip') and (L - P) // 2 <= l < (L + P) // 2
def patch(delta, kind):                            # delta (N, 256, d): odd positions take the previous position's
    if kind == 'reuse': delta[:, 1::2] = delta[:, 0:-1:2]
    else: delta[:, 1::2] = 0
    return delta
odd_keys = np.zeros(C, dtype=np.float32); odd_keys[H2 + 1::2] = -np.inf   # a skipped position leaves no K/V
stats = []
if os.environ.get('SEQSEL_V'): V = {k: V[k] for k in V if k == 'dense' or k in os.environ['SEQSEL_V'].split(',')}
emb = W('token_embd.weight')
xp = emb[toks[:, :H2]]; xg = {v: emb[toks[:, H2:]].copy() for v in V}
del emb
pp, pg = np.arange(H2, dtype=np.float64), np.arange(H2, C, dtype=np.float64)
t0 = time.time()
for l in range(L):
    b = f'blk.{l}.'
    an, wq, wk, wv, wo = W(b + 'attn_norm.weight'), W(b + 'attn_q.weight'), W(b + 'attn_k.weight'), W(b + 'attn_v.weight'), W(b + 'attn_output.weight')
    def qkv(x, pos):
        h = norm(x, an)
        return (rope(mm(h, wq).reshape(N, -1, nh, hd), pos), rope(mm(h, wk).reshape(N, -1, nkv, hd), pos), mm(h, wv).reshape(N, -1, nkv, hd))
    qp, kp, vp = qkv(xp, pp)
    xp = xp + mm(attn(qp, kp, vp, 0), wo)
    for v, (kind, keep, P) in V.items():
        qg, kg, vg = qkv(xg[v], pg)
        st = stale(kind, P, l)
        da = mm(attn(qg, np.concatenate([kp, kg], 1), np.concatenate([vp, vg], 1), H2, odd_keys if st else None), wo)
        xg[v] += patch(da, kind) if st else da
    del wq, wk, wv, wo
    fn, wg, wu, wd = W(b + 'ffn_norm.weight'), W(b + 'ffn_gate.weight'), W(b + 'ffn_up.weight'), W(b + 'ffn_down.weight')
    def lowbit(w, bits):                             # per-row scale; 1 bit = sign, 2 bits = Gaussian 4-level
        a_ = np.abs(w).mean(1, keepdims=True); sg = np.sign(w)
        if bits == 1: return sg * a_
        sd = a_ / 0.798
        return sg * np.where(np.abs(w) > 0.98 * sd, 1.51 * sd, 0.45 * sd)
    PX = {bt: (lowbit(wg, bt % 10), lowbit(wu, bt % 10)) for bt in {P for (kd, kp, P) in V.values() if kd == 'prox'}}
    def hid(x, rule=None, keep=1.0, bits=0):
        h = norm(x, fn); g = silu(mm(h, wg)); u = mm(h, wu)
        if rule == 'prox':
            hq = h if bits < 10 else np.clip(np.round(h / (0.9 * np.sqrt((h * h).mean(-1, keepdims=True)))), -3, 3)
            a = np.abs(silu(mm(hq, PX[bits][0])) * mm(hq, PX[bits][1])); k = int(round(keep * a.shape[-1]))
            return g * u * (a >= -np.partition(-a, k - 1, axis=-1)[..., k - 1:k])
        if rule == 'ud':
            a = np.abs(u); k = int(round(0.7 * a.shape[-1])); z = g * u * (a >= -np.partition(-a, k - 1, axis=-1)[..., k - 1:k])
            a = np.abs(z); k = int(round(keep * a.shape[-1])); return z * (a >= -np.partition(-a, k - 1, axis=-1)[..., k - 1:k])
        if rule:
            a = np.abs(u if rule == 'up' else g); k = int(round(keep * a.shape[-1]))
            u = u * (a >= -np.partition(-a, k - 1, axis=-1)[..., k - 1:k])
        return g * u
    xp_in = xp; zp = hid(xp); xp = xp + mm(zp, wd)
    nf = zp.shape[-1]
    zn = zp / np.linalg.norm(zp, axis=-1, keepdims=True)          # GRIFFIN: unit rows, then column norms
    stat = {P: np.linalg.norm(zn[:, H2 - P:], axis=1) for P in (256, 64)}   # (N, nf)
    gstat = stat[256].mean(0, keepdims=True).repeat(N, 0)
    for v, (kind, keep, P) in V.items():
        z = hid(xg[v], kind if kind in ('up', 'gate', 'prox', 'ud') else None, keep, P)
        if kind in ('seq', 'glob', 'seqg4'):
            s = gstat if kind == 'glob' else stat[P]
            if kind == 'seqg4': s = np.repeat(np.sqrt((s.reshape(N, -1, 4) ** 2).sum(-1)), 4, axis=1)
            k = int(round(keep * nf)); thr = -np.partition(-s, k - 1, axis=1)[:, k - 1:k]
            z *= (s >= thr)[:, None, :]
        elif kind == 'dyn':
            a = np.abs(z); k = int(round(keep * nf)); thr = -np.partition(-a, k - 1, axis=-1)[..., k - 1:k]
            z *= (a >= thr)
        df = mm(z, wd)
        xg[v] += patch(df, kind) if stale(kind, P, l) else df
    if nn_mode:                                      # is any EARLIER position of the context close? (delta base)
        def nn(a):
            G = np.einsum('ntd,nsd->nts', a, a, optimize=True); n2 = np.einsum('ntt->nt', G)
            D = n2[:, :, None] + n2[:, None, :] - 2 * G + np.where(np.tri(a.shape[1], k=-1) > 0, 0, np.inf)[None]
            return float((D[:, 16:].min(-1) / n2[:, 16:]).mean())
        nnst.append([nn(norm(xp_in, fn)), nn(zp)])
    if not only_dense:                               # statistics on the dense context half (no PPL)
        hp = norm(xp_in, fn)
        rel = lambda a: float((np.diff(a, axis=1) ** 2).sum() / (a[:, 1:] ** 2).sum())
        row = [l, rel(hp), rel(zp)]
        if l % 4 == 0:                               # activation subspace: fit on the first half of the chunks
            A, B = hp[:N // 2].reshape(-1, d), hp[N // 2:].reshape(-1, d)
            mu = A.mean(0); vt = np.linalg.svd(A - mu, full_matrices=False)[2]
            Bc = B - mu; tot = (B ** 2).sum()
            row += [float(((B @ mu) ** 2).sum() / (mu @ mu) / tot)] + [float(((Bc @ vt[:r_].T) ** 2).sum() / tot) for r_ in (64, 256, 1024)]
        stats.append(row)
    del wg, wu, wd, zp, zn
    print(f'layer {l} {time.time() - t0:.0f}s', file=sys.stderr, flush=True)
on, wout = W('output_norm.weight'), W('output.weight')
tgt = toks[:, H2 + 1:]
res = {}
for v in V:
    lg = mm(norm(xg[v][:, :-1], on), wout).astype(np.float64)
    lg -= lg.max(-1, keepdims=True)
    nll = np.log(np.exp(lg).sum(-1)) - np.take_along_axis(lg, tgt[..., None], -1)[..., 0]
    res[v] = nll.mean(1)                                            # per chunk
    del lg
dn = res['dense']
print(f'model {path.split("/")[-1]} chunks {N} scored tokens {N * (H2 - 1)} add_bos {add_bos}')
print('dense running PPL per chunk:', ' '.join(f'{np.exp(dn[:i + 1].mean()):.4f}' for i in range(min(N, 8))))
for v in V:
    dl = res[v] - dn
    se = dl.std(ddof=1) / np.sqrt(N) if N > 1 else 0
    print(f'{v:8s} PPL {np.exp(res[v].mean()):8.4f}  ratio to dense {np.exp(dl.mean()):.4f}  (+-{se:.4f} s.e. of mean log ratio over chunks)')
if stats:
    print('consecutive-token squared change / squared norm (dense context half), mean over layers [min, max]:')
    for i, nm in ((1, 'FFN input (post-norm)'), (2, 'FFN hidden (down input)')):
        a = np.array([r_[i] for r_ in stats]); print(f'  {nm}: {a.mean():.3f} [{a.min():.3f}, {a.max():.3f}]')
    print('FFN input energy share on held-out chunks: layer, mean direction, +top-64, +top-256, +top-1024 centred PCs')
    for r_ in stats:
        if len(r_) > 3: print('  ', r_[0], ' '.join(f'{x:.3f}' for x in r_[3:]))
if nnst:
    a = np.array(nnst)
    print('nearest earlier position (context half, positions 16..255): squared distance / squared norm, mean over layers [min, max]')
    print(f'  FFN input: {a[:, 0].mean():.3f} [{a[:, 0].min():.3f}, {a[:, 0].max():.3f}]   FFN hidden: {a[:, 1].mean():.3f} [{a[:, 1].min():.3f}, {a[:, 1].max():.3f}]')
if SCERR: print(f'scale code: rms relative error of a block scale, mean over tensors {np.mean(SCERR):.4f}, max {np.max(SCERR):.4f}')
