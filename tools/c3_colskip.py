# host, machine-independent numerics: input-side (column) sparsity on the dense mat-muls of a transformer block, scored
# exactly like llama-perplexity (ctx 512, BOS at the start of each chunk, second half scored). Zeroing the inputs with
# |x| < t (t per weight input, calibrated on a separate text) is what skipping their weight columns computes, so this
# is the PPL of the column lever without a kernel (campaign 3 memo-01 H3). Generalises tools/dense_colskip.py from
# Granite's groups to the four distinct inputs of a llama/qwen2 block:
#   qkv = attn_q/k/v input (attn_norm out), o = attn_output input, gu = ffn_gate/up input (ffn_norm out), down = ffn_down input.
# Also does H7 (sublayer ablation: the output of one attn_output or ffn_down mat-mul is zeroed, i.e. the sublayer
# adds nothing to the residual).
# usage:
#   calib: c3_colskip.py calib <libdir> <model> <train text> <n_chunks> <out.npz>
#   eval:  c3_colskip.py eval <libdir> <model> <test text> <n_chunks> <ub 512|1> <table.npz|-> <sparsity e.g. 0.4> <groups comma: qkv,o,gu,down|all> <dense_first_N> [threads]
#   ablate: c3_colskip.py ablate <libdir> <model> <test text> <n_chunks> <ub> <attn|ffn> <layer[,layer...]> [threads]
#   ffnskip: c3_colskip.py ffnskip <libdir> <model> <test text> <n_chunks> <ub> <table.npz> <oracle|prev> <skip fraction> [threads]
#     X4 (memo-03): per token, the FFN sublayer of layer L adds nothing when r = |ffn_out| / |ffn_inp| (L2 per token) is
#     below one global threshold, calibrated on train so that <skip fraction> of all (token, layer) pairs fall below it.
#     oracle: r of the same layer (known only after the FFN ran: an upper bound for any predictor);
#     prev: r of layer L-1 for the same token (causal, so the FFN of L need not be read). Layer 0 never skips under prev.
# eval with table '-' (or sparsity 0) reproduces llama-perplexity: the identity check of this tool.
import sys, os, re, time, json, ctypes as C, numpy as np, collections
mode, libdir, model_path, text_path, n_chunks = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5])
_src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dense_acts.py')).read(); exec(_src[_src.index('ggml = C.CDLL'):_src.index('# ---- stats')])
L.llama_get_logits.restype = C.POINTER(C.c_float); L.llama_get_logits.argtypes = [C.c_void_p]
L.llama_vocab_n_tokens.restype = C.c_int32; L.llama_vocab_n_tokens.argtypes = [C.c_void_p]
L.llama_vocab_get_add_bos.restype = C.c_bool; L.llama_vocab_get_add_bos.argtypes = [C.c_void_p]
L.llama_vocab_bos.restype = C.c_int32; L.llama_vocab_bos.argtypes = [C.c_void_p]
KIND = {'attn_q': 'qkv', 'attn_k': 'qkv', 'attn_v': 'qkv', 'attn_output': 'o', 'ffn_gate': 'gu', 'ffn_up': 'gu', 'ffn_down': 'down',
        'channel_mix_key': 'cmix_k', 'channel_mix_value': 'cmix_v'}   # RWKV-7: cmix_v's input is relu(k)^2 (H4 counts its exact zeros)
QS = (0.2, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6)   # tables store their own list as 'qs'
if mode == 'calib': UB, out_path, NT = 512, sys.argv[6], int(sys.argv[7]) if len(sys.argv) > 7 else 8
elif mode == 'eval':
    UB, tab, SP, GROUPS, NDENSE = int(sys.argv[6]), sys.argv[7], float(sys.argv[8]), sys.argv[9], int(sys.argv[10]); NT = int(sys.argv[11]) if len(sys.argv) > 11 else 8
    # groups: 'all', 'qkv,o', or per group levels 'qkv=0.5,o=0.5,gu=0.3,down=0.5' (then <sparsity> is only a label)
    GSP = {g: SP for g in (('qkv', 'o', 'gu', 'down') if GROUPS == 'all' else GROUPS.split(','))} if '=' not in GROUPS else {a.split('=')[0]: float(a.split('=')[1]) for a in GROUPS.split(',')}
    GROUPS = set(GSP) if '=' not in sys.argv[9] else {sys.argv[9]}
    THR = {}
    if tab != '-' and SP > 0:
        z = np.load(tab); ql = list(z['qs'] if 'qs' in z.files else (0.2, 0.3, 0.4, 0.5, 0.6))
        THR = {k: float(z[k][ql.index(GSP[k.split(':')[1]])]) for k in z.files if ':' in k and k.split(':')[1] in GSP and GSP[k.split(':')[1]] > 0}
elif mode == 'ablate': UB, AKIND, ALAYER = int(sys.argv[6]), sys.argv[7], sys.argv[8]; ASET = {'%s:%s' % (l, 'o' if AKIND == 'attn' else 'down') for l in ALAYER.split(',')}; NT = int(sys.argv[9]) if len(sys.argv) > 9 else 8
else:
    UB, tab, RULE, SKF = int(sys.argv[6]), sys.argv[7], sys.argv[8], float(sys.argv[9]); NT = int(sys.argv[10]) if len(sys.argv) > 10 else 8
    z = np.load(tab); rr = np.concatenate([z[k] for k in z.files if k.startswith('ratio:')]); RTHR = float(np.quantile(rr, SKF)) if SKF > 0 else -1.0
    FDENSE = int(os.environ.get('C3_DENSE_FIRST', '0'))   # positions < FDENSE never skip (attention-sink tokens)
    print('ffnskip %s: global ratio threshold %.5f for skip fraction %g (train)' % (RULE, RTHR, SKF), flush=True)
resin = {}; rprev = {}; ffn_in = {}; ratios = collections.defaultdict(list)
def ptr(p): return C.cast(p, C.POINTER(ggml_tensor)).contents
inp = {}; outn = {}; phase = ['discover']; pos0 = [0]; samp = collections.defaultdict(list); rng = np.random.default_rng(0)
cnt = collections.defaultdict(lambda: [0.0, 0])   # group -> [zeroed elements, elements]
def cb(t, ask, ud):
    tt = t.contents
    if phase[0] == 'discover':
        if ask: return ggml.ggml_op_desc(t) == b'MUL_MAT'
        s0 = ptr(tt.src[0]); m = re.match(rb'^blk\.(\d+)\.(\w+)\.weight$', s0.name)
        if m and m.group(2).decode() in KIND:
            k = '%s:%s' % (int(m.group(1)), KIND[m.group(2).decode()]); inp[bytes(ptr(tt.src[1]).name)] = k; outn[bytes(tt.name)] = k
            if k.endswith(':down'): resin[b'ffn_inp-%d' % int(m.group(1))] = int(m.group(1))
        return True
    name = tt.name
    if mode == 'ablate':
        if ask: return outn.get(name) in ASET   # one layer, or a comma list of layers
        C.memset(tt.data, 0, tt.nb[3] * tt.ne[3]); cnt['ablated'][1] += 1; return True
    if mode == 'ffnskip' or (mode == 'calib' and (name in resin or (outn.get(name) or '').endswith(':down'))):
        if name in resin:   # residual entering the FFN of layer L: remember its per-token norm
            if ask: return True
            ffn_in[resin[name]] = np.linalg.norm(np.ctypeslib.as_array(C.cast(tt.data, C.POINTER(C.c_float)), shape=(tt.ne[1], tt.ne[0])), axis=1); return True
        k = outn.get(name)
        if k is not None and k.endswith(':down'):
            if ask: return True
            l = int(k.split(':')[0]); y = np.ctypeslib.as_array(C.cast(tt.data, C.POINTER(C.c_float)), shape=(tt.ne[1], tt.ne[0]))
            r = np.linalg.norm(y, axis=1)/(ffn_in[l] + 1e-30)
            if mode == 'calib': ratios[l].append(r.astype(np.float32)); return True
            use = r if RULE == 'oracle' else rprev.get(l-1)
            rprev[l] = r
            if use is not None:
                m = use < RTHR; m[:max(0, FDENSE - pos0[0])] = False; y[m] = 0.0; cnt['ffn'][0] += m.sum()
            cnt['ffn'][1] += len(r); return True
        if mode == 'ffnskip': return not ask
    k = inp.get(name)
    if ask: return k is not None and (mode == 'calib' or k in THR)
    if k is None or tt.type != 0 or tt.nb[1] != tt.ne[0]*4: return True
    x = np.ctypeslib.as_array(C.cast(tt.data, C.POINTER(C.c_float)), shape=(tt.ne[1], tt.ne[0]))
    if mode == 'calib':
        ax = np.abs(x).ravel(); samp[k].append(ax[rng.integers(0, ax.size, 32768)].astype(np.float32))
        d = np.abs(x).reshape(x.shape[0], -1, 32).max(2, keepdims=True)/127; cnt[k.split(':')[1]][0] += ((np.abs(x).reshape(x.shape[0], -1, 32) < 0.5*d) | (x.reshape(x.shape[0], -1, 32) == 0)).sum(); cnt[k.split(':')[1]][1] += x.size
        z0 = (x == 0); cnt['exact0:' + k.split(':')[1]][0] += z0.sum(); cnt['exact0:' + k.split(':')[1]][1] += x.size
        cnt['exact0:' + k][0] += z0.sum(); cnt['exact0:' + k][1] += x.size
        return True
    first = max(0, NDENSE - pos0[0])   # rows of this ubatch at positions < NDENSE stay dense
    if first >= x.shape[0]: return True
    xs = x[first:]; m = np.abs(xs) < THR[k]; np.putmask(xs, m, 0.0)
    g = k.split(':')[1]; cnt[g][0] += m.sum(); cnt[g][1] += m.size
    return True
cbf = CB(cb)
L.llama_backend_init()
model = L.llama_model_load_from_file(model_path.encode(), L.llama_model_default_params()); assert model
cp = L.llama_context_default_params(); cp.n_ctx = 512; cp.n_batch = 512; cp.n_ubatch = UB; cp.n_seq_max = 1; cp.n_threads = NT; cp.n_threads_batch = NT; cp.cb_eval = cbf; cp.no_perf = True
ctx = L.llama_init_from_model(model, cp); assert ctx
vocab = L.llama_model_get_vocab(model); nv = L.llama_vocab_n_tokens(vocab); text = open(text_path, 'rb').read()
toks = (C.c_int32 * (len(text)+16))(); ntok = L.llama_tokenize(vocab, text, len(text), toks, len(text)+16, True, False); assert ntok > 0
tk = np.ctypeslib.as_array(toks, shape=(ntok,)).copy(); add_bos = L.llama_vocab_get_add_bos(vocab); bos = L.llama_vocab_bos(vocab)
n_chunks = min(n_chunks, ntok // 512)
b = L.llama_batch_init(512, 0, 1); mem = L.llama_get_memory(ctx); first = 256
def run_chunk(c):
    ids = tk[c*512:(c+1)*512].copy()
    if add_bos: ids[0] = bos
    L.llama_memory_clear(mem, True); out = np.empty((512, nv), np.float32)
    for s in range(0, 512, UB):
        b.n_tokens = UB; pos0[0] = s
        for i in range(UB): b.token[i] = int(ids[s+i]); b.pos[i] = s+i; b.n_seq_id[i] = 1; b.seq_id[i][0] = 0; b.logits[i] = 1
        assert L.llama_decode(ctx, b) == 0
        out[s:s+UB] = np.ctypeslib.as_array(L.llama_get_logits(ctx), shape=(UB, nv))
    return out, tk[c*512:(c+1)*512]
# discovery: one throwaway ubatch of the same size records which named tensors feed (and leave) the block mat-muls
b.n_tokens = UB
for i in range(UB): b.token[i] = int(tk[i]); b.pos[i] = i; b.n_seq_id[i] = 1; b.seq_id[i][0] = 0; b.logits[i] = 1
assert L.llama_decode(ctx, b) == 0; phase[0] = 'run'
print('discovered inputs: %d (%s); thresholds in use: %d' % (len(inp), dict(collections.Counter(v.split(':')[1] for v in inp.values())), len(THR) if mode == 'eval' else 0), flush=True)
t0 = time.time(); nll = []; line = []
for c in range(n_chunks):
    lg, ids = run_chunk(c)
    if mode == 'calib':
        if (c+1) % 4 == 0: print('calib chunk', c+1, '%.0fs' % (time.time()-t0), flush=True)
        continue
    l = lg[first:511].astype(np.float64); mx = l.max(1); tgt = ids[first+1:512]
    nll.append(np.log(np.exp(l - mx[:, None]).sum(1)) + mx - l[np.arange(255), tgt])
    line.append('[%d]%.4f' % (c+1, np.exp(np.concatenate(nll).mean())))
    if (c+1) % 4 == 0 or c == n_chunks-1: print(','.join(line[-4:]), '%.0fs' % (time.time()-t0), flush=True)
if mode == 'calib':
    np.savez(out_path, **{k: np.quantile(np.concatenate(v), QS) for k, v in samp.items()}, **{'ratio:%d' % l: np.concatenate(v) for l, v in ratios.items()}, qs=np.array(QS))
    print('wrote', out_path, len(samp), 'inputs; quantiles', QS)
    print('stock exact Q8_0 zeros (|x| < d/2 in each 32-block), train: ' + ', '.join('%s %.4f' % (g, z/n) for g, (z, n) in sorted(cnt.items()) if ':' not in g))
    print('exact float zeros per group: ' + ', '.join('%s %.4f' % (g[7:], z/n) for g, (z, n) in sorted(cnt.items()) if g.count(':') == 1))
    print('exact float zeros per input: ' + ', '.join('%s %.4f' % (g[7:], z/n) for g, (z, n) in sorted(cnt.items(), key=lambda a: (a[0].split(':')[-1], int(a[0].split(':')[1]) if a[0].count(':') == 2 else -1)) if g.count(':') == 2))
else:
    per_chunk = [float(v.mean()) for v in nll]
    tag = ('sp %g groups %s dense_first %d' % (SP, ','.join(sorted(GROUPS)), NDENSE)) if mode == 'eval' else ('ablate %s layer %s' % (AKIND, ALAYER)) if mode == 'ablate' else ('ffnskip %s %g%s' % (RULE, SKF, ' dense_first %d' % FDENSE if FDENSE else ''))
    real = ', '.join('%s %.4f' % (g, z/max(n, 1)) for g, (z, n) in sorted(cnt.items()) if g != 'ablated')
    print('RESULT ub %d %s: PPL %.4f over %d chunks; realised zeroed fraction: %s' % (UB, tag, np.exp(np.concatenate(nll).mean()), n_chunks, real or '-'))
    print('PERCHUNK ' + json.dumps({'ub': UB, 'tag': tag, 'nll': per_chunk}))
