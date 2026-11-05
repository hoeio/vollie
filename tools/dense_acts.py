# host, machine-independent: per-token input statistics of every dense (non-expert) MUL_MAT in the decode graph,
# via llama.cpp's cb_eval (ggml_backend_sched_eval_callback) through ctypes. For each weight tensor: fraction of
# near-zero inputs, fraction of 32/256-wide input blocks that are near-zero (exact column-block skip candidates) and
# the input energy they hold, cosine / relative delta between consecutive tokens. Streams running sums only.
# usage: python3 lab/tools/dense_acts.py <libdir> <model.gguf> <text> <n_chunks> [threads]
import sys, ctypes as C, numpy as np, collections, re, time
libdir, model_path, text_path, n_chunks = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]); NT = int(sys.argv[5]) if len(sys.argv) > 5 else 8
ggml = C.CDLL(libdir + '/libggml-base.dylib'); L = C.CDLL(libdir + '/libllama.dylib')
class ggml_tensor(C.Structure):
    _fields_ = [('type', C.c_int), ('buffer', C.c_void_p), ('ne', C.c_int64*4), ('nb', C.c_size_t*4), ('op', C.c_int),
                ('op_params', C.c_int32*16), ('flags', C.c_int32), ('src', C.c_void_p*10), ('view_src', C.c_void_p),
                ('view_offs', C.c_size_t), ('data', C.c_void_p), ('name', C.c_char*64), ('extra', C.c_void_p), ('padding', C.c_char*8)]
ggml.ggml_op_desc.restype = C.c_char_p; ggml.ggml_op_desc.argtypes = [C.POINTER(ggml_tensor)]
CB = C.CFUNCTYPE(C.c_bool, C.POINTER(ggml_tensor), C.c_bool, C.c_void_p)
class model_params(C.Structure):
    _fields_ = [('devices', C.c_void_p), ('tensor_buft_overrides', C.c_void_p), ('n_gpu_layers', C.c_int32), ('split_mode', C.c_int),
                ('load_mode', C.c_int), ('lazy_mode', C.c_int), ('main_gpu', C.c_int32), ('tensor_split', C.c_void_p), ('progress_callback', C.c_void_p),
                ('progress_callback_user_data', C.c_void_p), ('kv_overrides', C.c_void_p), ('vocab_only', C.c_bool), ('check_tensors', C.c_bool),
                ('use_extra_bufts', C.c_bool), ('no_host', C.c_bool), ('no_alloc', C.c_bool), ('load_mtp', C.c_bool)]
class ctx_params(C.Structure):
    _fields_ = [('n_ctx', C.c_uint32), ('n_batch', C.c_uint32), ('n_ubatch', C.c_uint32), ('n_seq_max', C.c_uint32), ('n_rs_seq', C.c_uint32),
                ('n_outputs_max', C.c_uint32), ('n_outputs_max_per_seq', C.c_uint32), ('n_threads', C.c_int32), ('n_threads_batch', C.c_int32),
                ('ctx_type', C.c_int), ('rope_scaling_type', C.c_int), ('pooling_type', C.c_int), ('attention_type', C.c_int), ('flash_attn_type', C.c_int),
                ('rope_freq_base', C.c_float), ('rope_freq_scale', C.c_float), ('yarn_ext_factor', C.c_float), ('yarn_attn_factor', C.c_float),
                ('yarn_beta_fast', C.c_float), ('yarn_beta_slow', C.c_float), ('yarn_orig_ctx', C.c_uint32), ('defrag_thold', C.c_float),
                ('cb_eval', CB), ('cb_eval_user_data', C.c_void_p), ('type_k', C.c_int), ('type_v', C.c_int), ('moe_cache_size', C.c_size_t),
                ('abort_callback', C.c_void_p), ('abort_callback_data', C.c_void_p), ('embeddings', C.c_bool), ('offload_kqv', C.c_bool),
                ('no_perf', C.c_bool), ('op_offload', C.c_bool), ('swa_full', C.c_bool), ('kv_unified', C.c_bool), ('samplers', C.c_void_p),
                ('n_samplers', C.c_size_t), ('ctx_other', C.c_void_p)]
class batch(C.Structure):
    _fields_ = [('n_tokens', C.c_int32), ('token', C.POINTER(C.c_int32)), ('embd', C.POINTER(C.c_float)), ('pos', C.POINTER(C.c_int32)),
                ('n_seq_id', C.POINTER(C.c_int32)), ('seq_id', C.POINTER(C.POINTER(C.c_int32))), ('logits', C.POINTER(C.c_int8))]
L.llama_model_default_params.restype = model_params; L.llama_context_default_params.restype = ctx_params
L.llama_model_load_from_file.restype = C.c_void_p; L.llama_model_load_from_file.argtypes = [C.c_char_p, model_params]
L.llama_init_from_model.restype = C.c_void_p; L.llama_init_from_model.argtypes = [C.c_void_p, ctx_params]
L.llama_model_get_vocab.restype = C.c_void_p; L.llama_model_get_vocab.argtypes = [C.c_void_p]
L.llama_tokenize.argtypes = [C.c_void_p, C.c_char_p, C.c_int32, C.POINTER(C.c_int32), C.c_int32, C.c_bool, C.c_bool]
L.llama_batch_init.restype = batch; L.llama_batch_init.argtypes = [C.c_int32, C.c_int32, C.c_int32]
L.llama_decode.argtypes = [C.c_void_p, batch]; L.llama_get_memory.restype = C.c_void_p; L.llama_get_memory.argtypes = [C.c_void_p]
L.llama_memory_clear.argtypes = [C.c_void_p, C.c_bool]
# ---- stats
S = collections.OrderedDict()
def key(name): return re.sub(r'^blk\.\d+\.', '', name.decode())
def acc(k, x):   # x: (n_tok, ne0) fp32 inputs of one dense mat-mul
    st = S.setdefault(k, collections.defaultdict(float)); n = x.shape[0]; ax = np.abs(x); amax = ax.max(1, keepdims=True) + 1e-30
    e = (x*x).sum(1) + 1e-30
    for thr in (1e-3, 1e-2, 2e-2, 5e-2, 1e-1):
        m = ax < thr*amax; st['frac<%g' % thr] += m.mean()*n; st['energy<%g' % thr] += ((x*x*m).sum(1)/e).sum()
    for bw in (32, 256):
        if x.shape[1] % bw: continue
        xb = x.reshape(n, -1, bw); bmax = np.abs(xb).max(2); drop = bmax < 1e-2*amax
        st['blk%d_frac' % bw] += drop.mean()*n; st['blk%d_energy' % bw] += (((xb*xb).sum(2)*drop).sum(1)/e).sum()
        drop3 = bmax < 1e-3*amax; st['blk%d_frac1e-3' % bw] += drop3.mean()*n
    if n > 1:
        a, b = x[1:], x[:-1]; cos = (a*b).sum(1)/np.sqrt((a*a).sum(1)*(b*b).sum(1) + 1e-30); d = a - b
        st['cos_prev'] += cos.sum(); st['reldelta_l2'] += (np.sqrt((d*d).sum(1)/e[1:])).sum(); st['delta_frac<1e-2'] += (np.abs(d) < 1e-2*amax[1:]).mean()*(n-1); st['n_pairs'] += n-1
    st['n'] += n; st['ne0'] = x.shape[1]; st['energy_top10%'] += (np.sort(ax, axis=1)[:, -x.shape[1]//10:]**2).sum(1).__truediv__(e).sum()
def cb(t, ask, ud):
    tt = t.contents
    if ask: return ggml.ggml_op_desc(t) == b'MUL_MAT'
    s0 = C.cast(tt.src[0], C.POINTER(ggml_tensor)).contents; s1 = C.cast(tt.src[1], C.POINTER(ggml_tensor)).contents
    if not s0.name.endswith(b'.weight') or s1.type != 0: return True
    ne0, n = s1.ne[0], s1.ne[1]
    if s1.nb[1] != ne0*4: return True
    x = np.ctypeslib.as_array(C.cast(s1.data, C.POINTER(C.c_float)), shape=(n, ne0)).copy()
    acc(key(s0.name), x); return True
cbf = CB(cb)
L.llama_backend_init()
mp = L.llama_model_default_params(); model = L.llama_model_load_from_file(model_path.encode(), mp); assert model
cp = L.llama_context_default_params(); assert cp.n_batch == 2048 and cp.n_ubatch == 512, (cp.n_ctx, cp.n_batch, cp.n_ubatch)
cp.n_ctx = 512; cp.n_batch = 512; cp.n_ubatch = 512; cp.n_seq_max = 1; cp.n_threads = NT; cp.n_threads_batch = NT; cp.cb_eval = cbf; cp.no_perf = True
ctx = L.llama_init_from_model(model, cp); assert ctx
vocab = L.llama_model_get_vocab(model); text = open(text_path, 'rb').read()
toks = (C.c_int32 * (len(text)+16))(); ntok = L.llama_tokenize(vocab, text, len(text), toks, len(text)+16, True, False); assert ntok > 0, ntok
print('tokens', ntok, flush=True)
b = L.llama_batch_init(512, 0, 1); mem = L.llama_get_memory(ctx); t0 = time.time()
for c in range(n_chunks):
    L.llama_memory_clear(mem, True); b.n_tokens = 512
    for i in range(512): b.token[i] = toks[c*512+i]; b.pos[i] = i; b.n_seq_id[i] = 1; b.seq_id[i][0] = 0; b.logits[i] = 1
    r = L.llama_decode(ctx, b); assert r == 0, r
    if c % 5 == 0: print('chunk', c, '%.0fs' % (time.time()-t0), flush=True)
print('columns: inputs below thr*max(|x|) per token: fraction / energy fraction, thr = 1e-3 1e-2 2e-2 5e-2 1e-1; blk32 = fraction of 32-wide input blocks all below 1e-2*max; cos/rel|d| = consecutive tokens; top10%E = energy in the largest 10 % of inputs')
print('%-22s %5s %7s | %s | %7s | %7s %7s %7s' % ('weight', 'ne0', 'tokens', ' '.join('%13s' % ('<%g' % t) for t in (1e-3, 1e-2, 2e-2, 5e-2, 1e-1)), 'blk32', 'cos', 'rel|d|', 'top10%E'))
for k, st in S.items():
    n = st['n']; p = st['n_pairs'] or 1
    print('%-22s %5d %7d | %s | %7.4f | %7.3f %7.3f %7.3f' % (k, st['ne0'], n, ' '.join('%6.3f/%6.4f' % (st['frac<%g' % t]/n, st['energy<%g' % t]/n) for t in (1e-3, 1e-2, 2e-2, 5e-2, 1e-1)),
          st['blk32_frac']/n, st['cos_prev']/p, st['reldelta_l2']/p, st['energy_top10%']/n))
