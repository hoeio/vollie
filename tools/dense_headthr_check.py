# host: independent numpy check of dense_colskip.py 'head <thr>' — zero lm_head inputs below thr*max per token, Q8_K, exact head.
import sys, numpy as np
sys.path.insert(0, 'lab/llama.cpp-tiny/gguf-py'); import gguf
D, nch, thr = sys.argv[1], int(sys.argv[2]), float(sys.argv[3])
h_all = np.load(D+'/h.npy'); tok_all = np.load(D+'/tok.npy'); npos, first = 511, 256
sel = np.concatenate([np.arange(c*npos+first, c*npos+npos) for c in range(nch)]); h = h_all[sel].copy(); tok = tok_all[sel]
m = np.abs(h) < thr*np.abs(h).max(1, keepdims=True); h[m] = 0; print('zeroed fraction %.4f' % m.mean())
def q8k(x):
    xb = x.reshape(x.shape[0], -1, 256); d = np.abs(xb).max(-1, keepdims=True)/127.0
    return (np.clip(np.rint(xb/np.where(d > 0, d, 1)), -127, 127)*d).reshape(x.shape).astype(np.float32)
h = q8k(h)
r = gguf.GGUFReader('lab/models/var/d6-q3k-e6.gguf'); t = [t for t in r.tensors if t.name == 'token_embd.weight'][0]
W = gguf.quants.dequantize(t.data, t.tensor_type).astype(np.float32)/6.0
def nll_of(L, tk):
    mx = L.max(axis=1, keepdims=True); return (np.log(np.exp(L - mx).sum(axis=1)) + mx[:,0]) - L[np.arange(len(tk)), tk]
nll = np.concatenate([nll_of(h[b:b+255] @ W.T, tok[b:b+255]) for b in range(0, len(tok), 255)])
print('head thr %g: PPL %.4f over %d chunks (numpy route)' % (thr, np.exp(nll.mean()), nch))
