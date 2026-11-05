# host, machine-independent: bounded-error input-column skip for dense MUL_MATs, measured as PPL. Zeroing the inputs
# with |x| < thr*max|x| (per token) *is* what skipping their weight columns computes, so the PPL of this in-place edit is
# the PPL of the runtime lever (input-major re-laid weights, see 05_compute_less.md §8). Scoring = llama-perplexity's
# (ctx 512, second half of each chunk). thr = 0 reproduces llama-perplexity.
# usage: python3 lab/tools/dense_colskip.py <libdir> <model> <text> <n_chunks> <thr> <targets comma: ssm_out,ssm_in,ffn_norm,down_shexp,head,attn_out,all> [threads]
import sys, ctypes as C, numpy as np, time
libdir, model_path, text_path, n_chunks, thr, targets = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), float(sys.argv[5]), set(sys.argv[6].split(','))
NT = int(sys.argv[7]) if len(sys.argv) > 7 else 8
import os; BLK = int(os.environ.get('COLSKIP_BLOCK', '0'))   # 0 = per element (upper bound), 32 = Q4_0 block granularity
_src = open(__file__.replace('dense_colskip.py', 'dense_acts.py')).read(); exec(_src[_src.index('ggml = C.CDLL'):_src.index('# ---- stats')])   # struct/API defs
L.llama_get_logits.restype = C.POINTER(C.c_float); L.llama_get_logits.argtypes = [C.c_void_p]
L.llama_vocab_n_tokens.restype = C.c_int32; L.llama_vocab_n_tokens.argtypes = [C.c_void_p]
GROUP = {'ssm_in': 'ssm_in', 'ssm_out': 'ssm_out', 'ffn_gate_inp': 'ffn_norm', 'ffn_gate_shexp': 'ffn_norm', 'ffn_up_shexp': 'ffn_norm',
         'ffn_down_shexp': 'down_shexp', 'token_embd': 'head', 'attn_output': 'attn_out', 'attn_q': 'attn_qkv', 'attn_k': 'attn_qkv', 'attn_v': 'attn_qkv'}
import re
def wkey(name): return re.sub(r'^blk\.\d+\.', '', name.decode()).replace('.weight', '')
sig = lambda tt: (tt.data, tt.ne[0], tt.ne[1], ggml.ggml_op_desc(C.pointer(tt)), bytes(tt.name))   # name disambiguates attn_norm-L / ffn_norm-L sharing an address
rng = np.random.default_rng(0); tset = {}; mode = ['discover']; hits = [0, 0.0, 0]; gate_mask = [None]
def cb(t, ask, ud):
    tt = t.contents
    if mode[0] == 'discover':
        if ask: return ggml.ggml_op_desc(t) == b'MUL_MAT'
        s0 = C.cast(tt.src[0], C.POINTER(ggml_tensor)).contents; s1 = C.cast(tt.src[1], C.POINTER(ggml_tensor)).contents
        g = GROUP.get(wkey(s0.name)) if s0.name.endswith(b'.weight') else None
        if g and ('all' in targets or g in targets or (g == 'ssm_out' and 'random' in targets)): tset[sig(s1)] = g
        if 'up_rows' in targets and wkey(s0.name) == 'ffn_gate_shexp': tset[sig(tt)] = 'gate_out'; tset[sig(C.cast(tt.src[1], C.POINTER(ggml_tensor)).contents)] = None
        if 'up_rows' in targets and wkey(s0.name) == 'ffn_down_shexp': tset[sig(s1)] = 'up_rows'
        return True
    if ask: return thr > 0 and sig(tt) in tset and tset[sig(tt)] is not None
    ne0, n = tt.ne[0], tt.ne[1]
    if tt.nb[1] != ne0*4 or tt.type != 0: return True
    x = np.ctypeslib.as_array(C.cast(tt.data, C.POINTER(C.c_float)), shape=(n, ne0)); g = tset[sig(tt)]
    if g is None: return True
    if g == 'gate_out':   # row skip of ffn_up_shexp: rows whose silu(gate) is small; the mask is applied to the swiglu output below
        sg = np.abs(x/(1+np.exp(-x))); gate_mask[0] = sg < thr*sg.max(1, keepdims=True); return True
    m = gate_mask[0] if g == 'up_rows' else (np.abs(x) < thr*np.abs(x).max(1, keepdims=True))
    if BLK and g != 'up_rows': m = np.repeat(m.reshape(n, ne0//BLK, BLK).all(2), BLK, axis=1)   # runtime-skippable: whole quant block (32 channels) small
    if 'random' in targets: m = rng.random(x.shape) < m.mean()   # control: same fraction, random positions
    x[m] = 0.0; hits[0] += 1; hits[1] += m.mean(); hits[2] += n
    return True
cbf = CB(cb)
L.llama_backend_init()
model = L.llama_model_load_from_file(model_path.encode(), L.llama_model_default_params()); assert model
cp = L.llama_context_default_params(); cp.n_ctx = 512; cp.n_batch = 512; cp.n_ubatch = 512; cp.n_seq_max = 1; cp.n_threads = NT; cp.n_threads_batch = NT; cp.cb_eval = cbf; cp.no_perf = True
ctx = L.llama_init_from_model(model, cp); assert ctx
vocab = L.llama_model_get_vocab(model); nv = L.llama_vocab_n_tokens(vocab); text = open(text_path, 'rb').read()
toks = (C.c_int32 * (len(text)+16))(); ntok = L.llama_tokenize(vocab, text, len(text), toks, len(text)+16, True, False); assert ntok > 0
b = L.llama_batch_init(512, 0, 1); mem = L.llama_get_memory(ctx); t0 = time.time(); nll = []; first = 256
tk = np.ctypeslib.as_array(toks, shape=(ntok,))
# discovery pass: one throwaway decode of chunk 0 records (data ptr, shape, op) of every targeted mat-mul input
b.n_tokens = 512
for i in range(512): b.token[i] = int(tk[i]); b.pos[i] = i; b.n_seq_id[i] = 1; b.seq_id[i][0] = 0; b.logits[i] = 1
assert L.llama_decode(ctx, b) == 0; mode[0] = 'edit'
import collections; print('discovered inputs:', dict(collections.Counter(tset.values())), flush=True)
for c in range(n_chunks):
    L.llama_memory_clear(mem, True); b.n_tokens = 512
    for i in range(512): b.token[i] = int(tk[c*512+i]); b.pos[i] = i; b.n_seq_id[i] = 1; b.seq_id[i][0] = 0; b.logits[i] = 1
    assert L.llama_decode(ctx, b) == 0
    lg = np.ctypeslib.as_array(L.llama_get_logits(ctx), shape=(512, nv))[first:511].astype(np.float64)
    tgt = tk[c*512+first+1:(c+1)*512]; mx = lg.max(1); nll.append(np.log(np.exp(lg - mx[:, None]).sum(1)) + mx - lg[np.arange(255), tgt])
    if (c+1) in (1, 2, 4, 8, 16, 32, 60) or c == n_chunks-1: print('[%d]%.4f' % (c+1, np.exp(np.concatenate(nll).mean())), '%.0fs' % (time.time()-t0), flush=True)
print('targets %s thr %g: edited tensors per chunk %.1f, mean zeroed fraction %.4f; PPL %.4f over %d chunks' % (','.join(sorted(targets)), thr, hits[0]/n_chunks, hits[1]/max(hits[0], 1), np.exp(np.concatenate(nll).mean()), n_chunks))
