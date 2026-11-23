"""memo-12 round 11 (Ebony): is the official prompt's 32-token greedy reply the same on a phone-like path as in Tom's
dump?  *host*.  Feeds the dumped prompt tokens (official.ptok) in one batch, then decodes 32 tokens greedily one at a
time, and compares with official.rows.  EBONY_Q8KV=1: Q8_0 cache + flash attention (the app's setting).
  nice python3 -I ebony_official_path.py <repo>/tools/dense_acts.py <libdir> <model.gguf> <lab/c3/head/<m>> 32 [threads=2]
"""
import sys, os
src = open(sys.argv[1]).read().split('# ---- stats')[0]
sys.argv = sys.argv[1:]
exec(src)                                       # ctypes declarations; text_path is the dump directory, n_chunks the token count
d = text_path; NG = n_chunks
pt = np.fromfile(os.path.join(d, 'official.ptok'), dtype=np.int32); assert pt[0] == 0; prompt = pt[2:2 + pt[1]]
rows = np.fromfile(os.path.join(d, 'official.rows'), dtype=np.int32).reshape(-1, 4); ref = rows[np.argsort(rows[:, 1]), 2][:NG]
L.llama_get_logits_ith.restype = C.POINTER(C.c_float); L.llama_get_logits_ith.argtypes = [C.c_void_p, C.c_int32]
L.llama_vocab_n_tokens.restype = C.c_int32; L.llama_vocab_n_tokens.argtypes = [C.c_void_p]
L.llama_backend_init()
mp = L.llama_model_default_params(); model = L.llama_model_load_from_file(model_path.encode(), mp); assert model
cp = L.llama_context_default_params(); assert cp.n_batch == 2048 and cp.n_ubatch == 512
cp.n_ctx = 512; cp.n_batch = 512; cp.n_ubatch = 512; cp.n_seq_max = 1; cp.n_threads = NT; cp.n_threads_batch = NT; cp.no_perf = True
if os.environ.get('EBONY_Q8KV'): cp.type_k = cp.type_v = 8; cp.flash_attn_type = 1
ctx = L.llama_init_from_model(model, cp); assert ctx
nv = L.llama_vocab_n_tokens(L.llama_model_get_vocab(model)); b = L.llama_batch_init(512, 0, 1)
def feed(toks, p0):
    b.n_tokens = len(toks)
    for i, t in enumerate(toks): b.token[i] = int(t); b.pos[i] = p0 + i; b.n_seq_id[i] = 1; b.seq_id[i][0] = 0; b.logits[i] = int(i == len(toks) - 1)
    assert L.llama_decode(ctx, b) == 0
    lg = np.ctypeslib.as_array(L.llama_get_logits_ith(ctx, len(toks) - 1), shape=(nv,)); return int(lg.argmax()), float(np.sort(lg)[-1] - np.sort(lg)[-2])
out = []; margins = []
t, m = feed(prompt, 0); out.append(t); margins.append(m)       # Tom's tool feeds the prompt without its last token, then the last token alone; one batch here
for i in range(1, NG):
    t, m = feed([out[-1]], len(prompt) + i - 1); out.append(t); margins.append(m)
out = np.array(out); same = out == ref; first = int(np.argmin(same)) if not same.all() else -1
print('prompt tokens %d; reply tokens equal to the dump: %d of %d; first difference at token %d; smallest top-2 logit margin %.3f at token %d'
      % (len(prompt), int(same.sum()), NG, first, min(margins), int(np.argmin(margins))))
print('margins under 0.3:', [(i, round(x, 3)) for i, x in enumerate(margins) if x < 0.3])
