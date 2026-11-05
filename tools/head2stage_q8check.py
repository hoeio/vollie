# host: does ggml's Q8_K activation quantisation of h (Q6_K x Q8_K dot) explain the route-B vs route-A gap?
import sys, numpy as np
sys.path.insert(0, 'lab/llama.cpp-tiny/gguf-py'); import gguf
dump, hpath, tpath = sys.argv[1:4]
f = open(dump, 'rb'); f.read(8); n_ctx, n_vocab, n_chunk = np.frombuffer(f.read(12), dtype=np.int32)
tokens = np.frombuffer(f.read(4*n_chunk*n_ctx), dtype=np.int32).reshape(n_chunk, n_ctx)
first = n_ctx//2; n_tok = n_ctx-1-first; nv = 2*((n_vocab+1)//2)+4
rows = np.frombuffer(f.read(), dtype=np.uint16).reshape(n_chunk*n_tok, nv); hdr = rows[:, :4].copy().view(np.float32); vals = rows[:, 4:4+n_vocab]
true_tok = np.concatenate([tokens[c, first+1:n_ctx] for c in range(n_chunk)])
nllA = -(hdr[:,0]*vals[np.arange(len(true_tok)), true_tok].astype(np.float32) + hdr[:,1])
h = np.load(hpath); tok = np.load(tpath); npos = n_ctx-1; nch = min(h.shape[0]//npos, n_chunk)
sel = np.concatenate([np.arange(c*npos+first, c*npos+npos) for c in range(nch)])
h = h[sel]; tok = tok[sel]; nllA = nllA[:len(sel)]
r = gguf.GGUFReader('lab/models/var/d6-q3k-e6.gguf'); t = [t for t in r.tensors if t.name == 'token_embd.weight'][0]
W = gguf.quants.dequantize(t.data, t.tensor_type).astype(np.float32) / 6.0
def q8k(x):  # ggml quantize_row_q8_K: per 256-block, scale = max/127 (signed max), q = round(x/scale)
    xb = x.reshape(x.shape[0], -1, 256); amax = np.abs(xb).max(-1, keepdims=True); d = amax/127.0
    q = np.rint(xb/np.where(d > 0, d, 1)); q = np.clip(q, -127, 127)
    return (q*d).reshape(x.shape).astype(np.float32)
def nll_of(L, tk):
    m = L.max(axis=1, keepdims=True); return (np.log(np.exp(L - m).sum(axis=1)) + m[:,0]) - L[np.arange(len(tk)), tk]
for name, hh in (('fp32 h', h), ('Q8_K h', q8k(h))):
    nll = np.concatenate([nll_of(hh[b:b+256] @ W.T, tok[b:b+256]) for b in range(0, hh.shape[0], 256)])
    d = nll - nllA
    print('%s: PPL %.4f (dump %.4f); NLL diff vs dump: mean %+.4f median|d| %.4f p90|d| %.4f max %.3f' % (name, np.exp(nll.mean()), np.exp(nllA.mean()), d.mean(), np.median(np.abs(d)), np.percentile(np.abs(d), 90), np.abs(d).max()))
# where are the big diffs? 
i = np.argsort(-np.abs(d))[:5]; print('largest |d| positions', i, 'dump nll', nllA[i].round(3), 'route B nll', nll[i].round(3))
