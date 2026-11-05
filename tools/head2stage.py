# host, machine-independent: two-stage lm_head replay on true lm_head inputs h (llama-server, pooling none; see
# head2stage_fetch.py). Exact head = ggml's Q6_K x Q8_K(h) / logit_scale, scored like llama-perplexity (second half of
# each 512 chunk). Screens: rank-r SVD of W (fp16 factors) and low-bit copies of W (Q2_K / Q3_K / Q4_0). Two-stage:
# screen scores all rows -> top-M candidates get exact logits, the rest keep the screen logit (what a runtime would
# hand to the sampler / to llama-perplexity). Bytes/token = screen bytes + M x 1260 B (one Q6_K row of 1536).
# usage: python3 lab/tools/head2stage.py <dir with h.npy tok.npy> [n_chunks] [screen gguf dir] [only: Q2_K:8192,Q3_K:8192,...]
import sys, time, numpy as np
sys.path.insert(0, 'lab/llama.cpp-tiny/gguf-py'); import gguf
D = sys.argv[1]; nch = int(sys.argv[2]) if len(sys.argv) > 2 else 0; SD = sys.argv[3] if len(sys.argv) > 3 else None
ONLY = dict() if len(sys.argv) <= 4 else {}
for kv in (sys.argv[4].split(',') if len(sys.argv) > 4 else []): k, m = kv.split(':'); ONLY.setdefault(k, []).append(int(m))
h_all = np.load(D+'/h.npy'); tok_all = np.load(D+'/tok.npy'); npos, first = 511, 256
nch = nch or h_all.shape[0]//npos
sel = np.concatenate([np.arange(c*npos+first, c*npos+npos) for c in range(nch)])
def q8k(x):
    xb = x.reshape(x.shape[0], -1, 256); d = np.abs(xb).max(-1, keepdims=True)/127.0
    return (np.clip(np.rint(xb/np.where(d > 0, d, 1)), -127, 127)*d).reshape(x.shape).astype(np.float32)
h = q8k(h_all[sel]); tok = tok_all[sel]; N = len(tok); n_tok = npos-first
def head(path):
    r = gguf.GGUFReader(path); t = [t for t in r.tensors if t.name == 'token_embd.weight'][0]
    return gguf.quants.dequantize(t.data, t.tensor_type).astype(np.float32)/6.0, t.tensor_type.name, t.n_bytes
W, _, nb6 = head('lab/models/var/d6-q3k-e6.gguf'); n_vocab = W.shape[0]; ROW = nb6//n_vocab
print('positions %d (%d chunks x %d), vocab %d, Q6_K head %.1f MB, %d B/row' % (N, nch, n_tok, n_vocab, nb6/1e6, ROW), flush=True)
def nll_of(L, tk):
    m = L.max(axis=1, keepdims=True); return (np.log(np.exp(L - m).sum(axis=1)) + m[:,0]) - L[np.arange(len(tk)), tk]
B = 255
ref_nll = np.concatenate([nll_of(h[b:b+B] @ W.T, tok[b:b+B]) for b in range(0, N, B)])
ref_arg = np.concatenate([(h[b:b+B] @ W.T).argmax(1) for b in range(0, N, B)])
ref_ppl = np.exp(ref_nll.mean())
print('reference PPL (exact Q6_K head, Q8_K activations) %.4f; cumulative per chunk:' % ref_ppl, ' '.join('[%d]%.4f' % (c+1, np.exp(ref_nll[:(c+1)*n_tok].mean())) for c in range(nch) if (c+1) in (1, 2, 4, 8, 16, 32, 60, nch)))
print('argmax == true token %.4f' % (ref_arg == tok).mean(), flush=True)
MS_PER_MB = (0.0413, 0.0457)   # device, R25 23.1 ms / 560 MB and R20 25.6 ms / 560 MB (dense MUL_MAT, bandwidth ridge)
print('rows: screen  M  argmax-recall  true-recall  PPL2  dPPL%%  MB/token  est. device ms saved vs %.1f MB head' % (nb6/1e6))
def run(name, score_fn, screen_bytes, Ms=(1024, 2048, 4096, 8192, 16384, 32768)):
    hit_arg = {M: 0 for M in Ms}; hit_true = {M: 0 for M in Ms}; nll2 = {M: [] for M in Ms}
    for b in range(0, N, B):
        hb = h[b:b+B]; L = hb @ W.T; sk = score_fn(hb); order = np.argsort(-sk, axis=1)[:, :max(Ms)]; ar = np.arange(len(hb))
        for M in Ms:
            inset = np.zeros_like(L, dtype=bool); np.put_along_axis(inset, order[:, :M], True, axis=1)
            hit_arg[M] += inset[ar, ref_arg[b:b+B]].sum(); hit_true[M] += inset[ar, tok[b:b+B]].sum()
            nll2[M].append(nll_of(np.where(inset, L, sk), tok[b:b+B]))
    for M in Ms:
        ppl2 = np.exp(np.concatenate(nll2[M]).mean()); mb = (screen_bytes + M*ROW)/1e6
        saved = [(nb6/1e6 - mb)*c for c in MS_PER_MB]
        print('%-10s %6d  %.4f  %.4f  %12.4f  %+10.3f%%  %6.1f  %+.1f..%+.1f' % (name, M, hit_arg[M]/N, hit_true[M]/N, ppl2, 100*(ppl2/ref_ppl-1), mb, saved[0], saved[1]), flush=True)
# screen 1: rank-r SVD of W, fp16 factors
if not ONLY:
  t0 = time.time(); U, s, Vt = np.linalg.svd(W, full_matrices=False); e = np.cumsum(s**2)/np.sum(s**2)
  print('svd %.0fs; energy captured at rank 64/128/256/512/768: %s' % (time.time()-t0, np.round(e[[63, 127, 255, 511, 767]], 3)), flush=True)
for rk in ((128, 256, 512) if not ONLY else ()):
    Ur = (U[:, :rk]*s[:rk]).astype(np.float16).astype(np.float32); Vr = Vt[:rk]
    run('svd%d' % rk, lambda x, Ur=Ur, Vr=Vr: (x @ Vr.T) @ Ur.T, n_vocab*rk*2)
if not ONLY: del U, Vt
# screen 2: low-bit copies of W
if SD:
    for path, tag in ((SD+'/head_q2k.gguf', 'Q2_K'), (SD+'/head_q3k.gguf', 'Q3_K'), ('lab/models/var/d6-q3k-e6-headq4_0.gguf', 'Q4_0')):
        if ONLY and tag not in ONLY: continue
        try: Wq, tn, nb = head(path)
        except Exception as ex: print('skip', tag, ex); continue
        alone = np.exp(np.concatenate([nll_of(h[b:b+B] @ Wq.T, tok[b:b+B]) for b in range(0, N, B)]).mean())
        print('screen %s %.1f MB; PPL of the %s head alone (no exact stage) %.4f (%+.2f%%)' % (tag, nb/1e6, tag, alone, 100*(alone/ref_ppl-1)), flush=True)
        run(tag, lambda x, Wq=Wq: x @ Wq.T, nb, **({'Ms': ONLY[tag]} if ONLY else {})); del Wq
