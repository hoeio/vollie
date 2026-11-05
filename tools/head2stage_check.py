# host: reconcile the two lm_head evaluation routes against llama-perplexity (Bonnie, r27).
#  route A: --save-all-logits dump (capped log-softmax, uint16) -> per-position NLL of the true token, directly
#  route B: hidden states h from llama-server /embedding (pooling none) -> logits = (W_q6k h)/logit_scale -> NLL
# usage: python3 lab/tools/head2stage_check.py <dump.bin> <h.npy> <tok.npy>
import sys, numpy as np
sys.path.insert(0, 'lab/llama.cpp-tiny/gguf-py'); import gguf
dump, hpath, tpath = sys.argv[1:4]
f = open(dump, 'rb'); assert f.read(8) == b'_logits_'
n_ctx, n_vocab, n_chunk = np.frombuffer(f.read(12), dtype=np.int32)
tokens = np.frombuffer(f.read(4*n_chunk*n_ctx), dtype=np.int32).reshape(n_chunk, n_ctx)
first = n_ctx//2; n_tok = n_ctx-1-first; nv = 2*((n_vocab+1)//2)+4
rows = np.frombuffer(f.read(), dtype=np.uint16).reshape(n_chunk*n_tok, nv)
hdr = rows[:, :4].copy().view(np.float32); vals = rows[:, 4:4+n_vocab]
true_tok = np.concatenate([tokens[c, first+1:n_ctx] for c in range(n_chunk)])
print('dump: n_ctx %d n_vocab %d n_chunk %d, scored positions %d/chunk; uncapped frac %.4f' % (n_ctx, n_vocab, n_chunk, n_tok, (vals > 0).mean()))
v = vals[np.arange(len(true_tok)), true_tok]
nllA = -(hdr[:,0]*v.astype(np.float32) + hdr[:,1])
print('route A: true token capped (val==0) at %d positions' % (v == 0).sum())
cum = [np.exp(nllA[:(c+1)*n_tok].mean()) for c in range(n_chunk)]
print('route A cumulative PPL per chunk (compare llama-perplexity [c]):', ' '.join('[%d]%.4f' % (c+1, p) for c, p in enumerate(cum)))
# route B
h = np.load(hpath); tok = np.load(tpath); npos = n_ctx-1; nch = h.shape[0]//npos
print('route B: h %s, chunks %d, |h| mean %.3f' % (h.shape, nch, np.linalg.norm(h, axis=1).mean()))
for c in range(nch):
    same = (tok[c*npos:(c+1)*npos] == tokens[c, 1:]).all()
    print('  chunk %d tokens identical to dump: %s' % (c, same))
r = gguf.GGUFReader('lab/models/var/d6-q3k-e6.gguf')
t = [t for t in r.tensors if t.name == 'token_embd.weight'][0]
ls = float(r.fields['granitehybrid.logit_scale'].parts[-1][0])
W = gguf.quants.dequantize(t.data, t.tensor_type).astype(np.float32) / ls
def nll_of(L, tk):
    m = L.max(axis=1, keepdims=True); return (np.log(np.exp(L - m).sum(axis=1)) + m[:,0]) - L[np.arange(len(tk)), tk]
nllB = np.concatenate([nll_of(h[b:b+256] @ W.T, tok[b:b+256]) for b in range(0, h.shape[0], 256)])
print('route B PPL all %d positions/chunk: %.4f' % (npos, np.exp(nllB.mean())))
sel = np.concatenate([np.arange(c*npos+first, c*npos+npos) for c in range(nch)])   # positions first..n_ctx-2 (predict first+1..n_ctx-1)
nllB2 = nllB[sel]
cumB = [np.exp(nllB2[:(c+1)*n_tok].mean()) for c in range(nch)]
print('route B cumulative PPL, second half only:', ' '.join('[%d]%.4f' % (c+1, p) for c, p in enumerate(cumB)))
m = min(len(nllB2), len(nllA)); d = nllB2[:m] - nllA[:m]
print('per-position NLL route B - route A on %d positions: mean %+.4f, |d| median %.4f, max %.4f' % (m, d.mean(), np.median(np.abs(d)), np.abs(d).max()))
