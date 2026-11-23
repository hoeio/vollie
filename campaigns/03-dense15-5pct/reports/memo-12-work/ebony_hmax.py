"""H-max (ebony-joins-r5 §1.5): largest |value| into and out of every MUL_MAT on a .weight tensor, per layer.
Half precision holds 65,504 at most; this says how close a subject's activations come. *host*; batch 512.
Reuses the ctypes declarations of tools/dense_acts.py (everything above its '# ---- stats' line).
  nice python3 -I ebony_hmax.py <repo>/tools/dense_acts.py <libdir> <model.gguf> <text> <n_chunks> [threads=2]
"""
import sys
src = open(sys.argv[1]).read().split('# ---- stats')[0]
sys.argv = sys.argv[1:]
exec(src)                                                        # ggml, L, C, np, re, CB, ggml_tensor, batch, argv parsing
R = []; M = {}                                                           # (kind, layer) -> [in_all, in_no_pos0, out_all, out_no_pos0, K]
def cb(t, ask, ud):
    tt = t.contents
    if ask: return ggml.ggml_op_desc(t) == b'MUL_MAT'
    s0 = C.cast(tt.src[0], C.POINTER(ggml_tensor)).contents; s1 = C.cast(tt.src[1], C.POINTER(ggml_tensor)).contents
    if not s0.name.endswith(b'.weight') or s1.type != 0 or tt.type != 0: return True
    k, n, m = s1.ne[0], s1.ne[1], tt.ne[0]
    if s1.nb[1] != k * 4 or n < 2: return True
    x = np.abs(np.ctypeslib.as_array(C.cast(s1.data, C.POINTER(C.c_float)), shape=(n, k)))
    y = np.abs(np.ctypeslib.as_array(C.cast(tt.data, C.POINTER(C.c_float)), shape=(n, m)))
    if s0.name == b'output.weight': xs = np.ctypeslib.as_array(C.cast(s1.data, C.POINTER(C.c_float)), shape=(n, k)); R.extend(((xs.max(1) - xs.min(1)) / np.sqrt((xs * xs).mean(1))).tolist())
    nm = s0.name.decode(); mm = re.match(r'blk\.(\d+)\.(.*)\.weight', nm)
    key = (mm.group(2), int(mm.group(1))) if mm else (nm, -1)
    v = M.setdefault(key, [0.0, 0.0, 0.0, 0.0, k])
    v[0] = max(v[0], float(x.max())); v[1] = max(v[1], float(x[1:].max())); v[2] = max(v[2], float(y.max())); v[3] = max(v[3], float(y[1:].max()))
    assert np.isfinite(x).all() and np.isfinite(y).all()
    return True
cbf = CB(cb)
L.llama_backend_init()
mp = L.llama_model_default_params(); model = L.llama_model_load_from_file(model_path.encode(), mp); assert model
cp = L.llama_context_default_params(); assert cp.n_batch == 2048 and cp.n_ubatch == 512, (cp.n_ctx, cp.n_batch, cp.n_ubatch)
cp.n_ctx = 512; cp.n_batch = 512; cp.n_ubatch = 512; cp.n_seq_max = 1; cp.n_threads = NT; cp.n_threads_batch = NT; cp.cb_eval = cbf; cp.no_perf = True
ctx = L.llama_init_from_model(model, cp); assert ctx
vocab = L.llama_model_get_vocab(model); text = open(text_path, 'rb').read()[:400000]
toks = (C.c_int32 * (len(text) + 16))(); ntok = L.llama_tokenize(vocab, text, len(text), toks, len(text) + 16, True, False); assert ntok >= n_chunks * 512, ntok
b = L.llama_batch_init(512, 0, 1); mem = L.llama_get_memory(ctx)
for c in range(n_chunks):
    L.llama_memory_clear(mem, True); b.n_tokens = 512
    for i in range(512): b.token[i] = toks[c * 512 + i]; b.pos[i] = i; b.n_seq_id[i] = 1; b.seq_id[i][0] = 0; b.logits[i] = 1
    assert L.llama_decode(ctx, b) == 0
assert M, 'no MUL_MAT seen'
print('first token id of chunk 0:', toks[0], ' chunks', n_chunks, ' n_layer-keys', len(M))
print('%-14s %6s | %10s @L  %10s @L | %10s @L  %10s @L   (max |in|: all positions, positions >= 1; max |out|: same)' % ('weight', 'K', 'in', 'in>=1', 'out', 'out>=1'))
for kind in sorted({k for k, _ in M}):
    rows = {l: v for (kk, l), v in M.items() if kk == kind}
    def top(i): l = max(rows, key=lambda l: rows[l][i]); return rows[l][i], l
    print('%-14s %6d | %10.1f @%-2d %10.1f @%-2d | %10.1f @%-2d %10.1f @%-2d' % ((kind, next(iter(rows.values()))[4]) + top(0) + top(1) + top(2) + top(3)))
if R: print('head input (final hidden state), range / rms per position: median %.1f, p90 %.1f, max %.1f over %d positions' % (np.median(R), np.quantile(R, .9), max(R), len(R)))
g = max(max(v[0], v[2]) for v in M.values())
print('largest of all: %.1f = %.4f of the half-precision maximum 65504' % (g, g / 65504))
