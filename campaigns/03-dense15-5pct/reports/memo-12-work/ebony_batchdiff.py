"""memo-12 round 8 (Ebony): which graph node first differs between one batched decode of N tokens and N one-token
decodes, at the last position?  *host*.  Records every 2-D float node whose second dimension is the token count
(the last token's column), in graph order, in both modes, and prints the nodes that differ.
  nice python3 -I ebony_batchdiff.py <repo>/tools/dense_acts.py <libdir> <model.gguf> <text> <N tokens> [threads=2]
"""
import sys
src = open(sys.argv[1]).read().split('# ---- stats')[0]
sys.argv = sys.argv[1:]
exec(src)                                      # ctypes declarations; n_chunks is used as N
N = n_chunks; REC = None; NT_NOW = [0]
def cb(t, ask, ud):
    if ask: return True
    tt = t.contents
    if REC is None or tt.type != 0 or tt.ne[2] != 1 or tt.ne[3] != 1 or tt.ne[1] != NT_NOW[0] or not tt.data: return True
    k = tt.ne[0]
    if tt.nb[1] != k * 4: return True
    a = np.ctypeslib.as_array(C.cast(tt.data, C.POINTER(C.c_float)), shape=(tt.ne[1], k))[-1].copy()
    REC.append((tt.name.decode(), ggml.ggml_op_desc(t).decode(), a)); return True
cbf = CB(cb)
L.llama_backend_init()
mp = L.llama_model_default_params(); mp.use_extra_bufts = not __import__('os').environ.get('EBONY_NOEXTRA'); model = L.llama_model_load_from_file(model_path.encode(), mp); assert model
cp = L.llama_context_default_params(); assert cp.n_batch == 2048 and cp.n_ubatch == 512
cp.n_ctx = 512; cp.n_batch = 512; cp.n_ubatch = 512; cp.n_seq_max = 1; cp.n_threads = NT; cp.n_threads_batch = NT; cp.cb_eval = cbf; cp.no_perf = True
if __import__('os').environ.get('EBONY_Q8KV'): cp.type_k = cp.type_v = 8; cp.flash_attn_type = 1          # the app's cache: Q8_0, flash attention on
ctx = L.llama_init_from_model(model, cp); assert ctx
vocab = L.llama_model_get_vocab(model); text = open(text_path, 'rb').read()[:20000]
toks = (C.c_int32 * (len(text) + 16))(); ntok = L.llama_tokenize(vocab, text, len(text), toks, len(text) + 16, True, False); assert ntok >= N
b = L.llama_batch_init(512, 0, 1); mem = L.llama_get_memory(ctx)
def feed(i0, n, rec):
    global REC
    b.n_tokens = n
    for i in range(n): b.token[i] = toks[i0 + i]; b.pos[i] = i0 + i; b.n_seq_id[i] = 1; b.seq_id[i][0] = 0; b.logits[i] = 1
    REC = [] if rec else None; NT_NOW[0] = n
    assert L.llama_decode(ctx, b) == 0
    r = REC; REC = None; return r
B_SZ = int(__import__('os').environ.get('EBONY_B', '0'))                    # 0: one batch of N. B > 0: N tokens in decodes of B (width-B test)
L.llama_memory_clear(mem, True)
if B_SZ:
    assert N % B_SZ == 0
    for i in range(0, N - B_SZ, B_SZ): feed(i, B_SZ, False)
    A = feed(N - B_SZ, B_SZ, True)                                         # the last decode of B tokens; its last column
else:
    A = feed(0, N, True)                                                   # one batch of N; last column
import os
if os.environ.get('EBONY_SAVE'):                                          # cross-build mode: save the batch pass and stop
    np.savez(os.environ['EBONY_SAVE'], names=np.array([x[0] + '|' + x[1] for x in A]), **{'a%d' % i: x[2] for i, x in enumerate(A)}); print('saved', len(A)); sys.exit(0)
L.llama_memory_clear(mem, True)
for i in range(N - 1): feed(i, 1, False)
B = feed(N - 1, 1, True)                                                  # the N-th token alone, after N-1 single decodes
L.llama_memory_clear(mem, True)
if B_SZ:
    for i in range(0, N - B_SZ, B_SZ): feed(i, B_SZ, False)
    A2 = feed(N - B_SZ, B_SZ, True)
else:
    A2 = feed(0, N, True)                                                  # control: the batch again
assert len(A) == len(B) == len(A2), (len(A), len(B), len(A2))
ctl = sum(not np.array_equal(x[2], y[2]) for x, y in zip(A, A2))
print('nodes compared', len(A), '; control (batch run twice): differing nodes', ctl)
nd = 0
for (n1, o1, a), (n2, o2, c) in zip(A, B):
    assert n1 == n2, (n1, n2)
    if not np.array_equal(a, c):
        nd += 1
        if nd <= 14: print('  differs: %-22s %-12s max|d| %.3g  rel %.3g  entries %d/%d' % (n1, o1, np.abs(a - c).max(), np.abs(a - c).max() / (np.abs(a).max() + 1e-30), int((a != c).sum()), a.size))
print('differing nodes, batch against single:', nd, 'of', len(A))
la, lb = A[-1][2].astype(np.float64), B[-1][2].astype(np.float64)
print('last node %s: max|d| %.4g, rms d %.4g, same argmax %s' % (A[-1][0], np.abs(la - lb).max(), np.sqrt(((la - lb) ** 2).mean()), la.argmax() == lb.argmax()))
