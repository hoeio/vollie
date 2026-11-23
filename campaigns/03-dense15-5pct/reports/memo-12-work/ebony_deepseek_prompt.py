import sys, os, ast
src = open(sys.argv[1]).read().split('# ---- stats')[0]; sys.argv = sys.argv[1:]; exec(src)
sys.path.insert(0, 'lab/llama.cpp-c3/gguf-py')
from gguf import GGUFReader
f = GGUFReader(model_path).fields; tk = f['tokenizer.ggml.tokens']; vocab = [bytes(tk.parts[i]).decode('utf-8', 'replace') for i in tk.data]
plain = ast.literal_eval(open(text_path).read().strip())            # ids of "User: ...\n\nAssistant:" (153)
variants = {'a: literal <s> (the app today)': [27, 82, 29] + plain, 'b: no BOS (the built-in formatter)': plain, 'c: BOS token (the template as written)': [100000] + plain}
L.llama_get_logits_ith.restype = C.POINTER(C.c_float); L.llama_get_logits_ith.argtypes = [C.c_void_p, C.c_int32]
L.llama_vocab_n_tokens.restype = C.c_int32; L.llama_vocab_n_tokens.argtypes = [C.c_void_p]
L.llama_backend_init(); mp = L.llama_model_default_params(); model = L.llama_model_load_from_file(model_path.encode(), mp); assert model
cp = L.llama_context_default_params(); assert cp.n_batch == 2048 and cp.n_ubatch == 512
cp.n_ctx = 512; cp.n_batch = 512; cp.n_ubatch = 512; cp.n_seq_max = 1; cp.n_threads = NT; cp.n_threads_batch = NT; cp.no_perf = True; cp.type_k = cp.type_v = 8; cp.flash_attn_type = 1
ctx = L.llama_init_from_model(model, cp); assert ctx
nv = L.llama_vocab_n_tokens(L.llama_model_get_vocab(model)); b = L.llama_batch_init(512, 0, 1); mem = L.llama_get_memory(ctx)
def feed(toks, p0):
    b.n_tokens = len(toks)
    for i, t in enumerate(toks): b.token[i] = int(t); b.pos[i] = p0 + i; b.n_seq_id[i] = 1; b.seq_id[i][0] = 0; b.logits[i] = int(i == len(toks) - 1)
    assert L.llama_decode(ctx, b) == 0
    lg = np.ctypeslib.as_array(L.llama_get_logits_ith(ctx, len(toks) - 1), shape=(nv,)); s = np.sort(lg); return int(lg.argmax()), float(s[-1] - s[-2])
res = {}
for name, pr in variants.items():
    L.llama_memory_clear(mem, True); out = []; mg = []
    t, m = feed(pr, 0); out.append(t); mg.append(m)
    for i in range(1, n_chunks): t, m = feed([out[-1]], len(pr) + i - 1); out.append(t); mg.append(m)
    res[name] = out
    def lookup(ctx):
        for n in (3, 2, 1):
            if len(ctx) <= n: continue
            tail = ctx[-n:]
            for q in range(len(ctx) - n - 1, -1, -1):
                if ctx[q:q + n] == tail: return ctx[q + n], n
        return None, 0
    st = ''; hits = {1: [], 2: []}
    for j in range(len(out) - 1):
        g, ln = lookup(list(pr) + out[:j + 1]); ok = g == out[j + 1] if g is not None else False
        st += ('L' if ok else ('x' if g is not None else '-')) + str(ln) + ' '; hits[1].append(ok); hits[2].append(ok and ln >= 2)
    def passes(h):
        h = h + [False]; j = p_ = 0
        while j < len(out):
            p_ += 1; j += 2 if (j + 1 < len(out) and h[j]) else 1
        return p_
    cov = sum(1 for i in range(len(out)) if any(list(pr)[q:q + 3] in (out[max(0, i - 2):i + 1], out[i:i + 3], out[max(0, i - 1):i + 2]) for q in range(len(pr) - 2)))
    print('   lookup steps:', st); print('   lookup alone: any length %d passes (%.3f tokens per pass); length >= 2 only %d passes (%.3f); reply tokens inside a 3-token run of the prompt: %d of %d' % (passes(hits[1]), len(out) / passes(hits[1]), passes(hits[2]), len(out) / passes(hits[2]), cov, len(out)))
    print(name, '| prompt tokens', len(pr), '| smallest top-2 gap %.3f at token %d' % (min(mg), int(np.argmin(mg))))
    print('   reply:', repr(''.join(vocab[t] for t in out).replace('Ġ', ' ').replace('Ċ', '\n')))
ks = list(res)
for i in range(3):
    for j in range(i + 1, 3):
        a, c = res[ks[i]], res[ks[j]]; same = [x == y for x, y in zip(a, c)]; print('%s vs %s: equal tokens %d of %d, first difference at %d' % (ks[i][:1], ks[j][:1], sum(same), len(a), same.index(False) if False in same else -1))
