# host: pull per-token lm_head inputs h (output_norm output, pooling none, no normalisation) for N wikitext-2 test
# chunks of 512 from a running llama-server --embeddings --pooling none. Tokenisation = llama-perplexity's
# (whole file, add_special=True; granitehybrid adds no BOS). Saves h[(n_ctx-1) per chunk] and the next-token ids.
# usage: python3 lab/tools/head2stage_fetch.py <outdir> <n_chunks> [port]
import json, urllib.request, http.client, numpy as np, sys, time
out, N = sys.argv[1], int(sys.argv[2]); U = 'http://127.0.0.1:%s' % (sys.argv[3] if len(sys.argv) > 3 else '18931')
def post(path, obj, tries=5):
    for a in range(tries):
        try:
            req = urllib.request.Request(U+path, data=json.dumps(obj).encode(), headers={'Content-Type': 'application/json'})
            return json.load(urllib.request.urlopen(req, timeout=900))
        except (http.client.IncompleteRead, urllib.error.URLError, ConnectionError) as e:
            print('retry', a, path, repr(e)[:80], flush=True); time.sleep(2)
    raise RuntimeError('post failed')
text = open('lab/text/wikitext-2-raw/wiki.test.raw').read()
toks = post('/tokenize', {'content': text, 'add_special': True})['tokens']
print('tokens', len(toks), 'chunks available', len(toks)//512, flush=True)
H = []; T = []; t0 = time.time()
for c in range(N):
    chunk = toks[c*512:(c+1)*512]
    r = post('/embedding', {'content': chunk, 'embd_normalize': -1})
    e = np.array(r[0]['embedding'] if isinstance(r, list) else r['embedding'], np.float32); assert e.shape == (512, 1536), e.shape
    H.append(e[:-1]); T.append(np.array(chunk[1:], np.int32))
    if c % 5 == 0: print('chunk', c, '%.0fs' % (time.time()-t0), flush=True)
np.save(out+'/h.npy', np.concatenate(H)); np.save(out+'/tok.npy', np.concatenate(T)); print('saved', N, 'chunks')
