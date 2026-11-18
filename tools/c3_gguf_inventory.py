# host, desk: byte inventory of a GGUF from its header only (local file or Hugging Face URL; for a URL only the
# header is fetched with HTTP range requests into a sparse file, so nothing like the full file is downloaded).
# Prints, per tensor group, parameters, bytes in the file's own types, and bytes per decoded token; the head type and
# whether it is tied; and which weights the phone build (arm64 dotprod, no i8mm) repacks into an anonymous copy.
# usage: python3 -I tools/c3_gguf_inventory.py <gguf-py dir> <path-or-url>[+<split 2>...] [...]
import sys, os, re, subprocess, tempfile, collections
sys.path.insert(0, sys.argv[1]); import gguf
REPACK = {'Q4_0': '4x4', 'Q4_K': '8x4', 'Q5_K': '8x4', 'Q6_K': '8x4', 'IQ4_NL': '4x4', 'Q8_0': '4x4', 'MXFP4': '4x4'}   # repack.cpp at 620d2225c, dotprod without i8mm; needs ne[1] % 4 (8 for K types)
LOOKUP = re.compile(r'^(token_embd|per_layer_token_embd|altup_.*embd|token_types)')   # read by GET_ROWS: one row per token

def open_reader(src, tmp):
    if not src.startswith('http'): return gguf.GGUFReader(src), os.path.getsize(src)
    size = int(subprocess.run(['curl', '-sIL', src], capture_output=True, text=True).stdout.lower().split('content-length:')[-1].split()[0])
    os.makedirs(tmp, exist_ok=True); path, n = os.path.join(tmp, 'h.gguf'), 8 << 20
    while True:
        subprocess.run(['curl', '-sL', '-r', '0-%d' % (n-1), '-o', path, src], check=True)
        got = os.path.getsize(path); os.truncate(path, size)   # sparse tail: shapes and offsets are all that is read
        try:
            r = gguf.GGUFReader(path)
            if r.tensors and min(int(t.data_offset) for t in r.tensors) <= got: return r, size
        except Exception: pass
        n *= 2; assert n <= size, 'header larger than file?'

def group(name):
    if LOOKUP.match(name): return 'lookup'
    if name.startswith('output.'): return 'head'
    k = re.sub(r'^blk\.\d+\.', '', name)
    if k.startswith('attn_') and 'norm' not in k: return 'attn'
    if k.startswith('ffn_') and 'norm' not in k: return 'ffn'
    if k.startswith(('time_mix', 'channel_mix')): return 'rwkv_tmix' if k.startswith('time_mix') else 'rwkv_cmix'
    if k.startswith('per_layer') or 'per_layer' in name or 'altup' in name or 'laurel' in name: return 'gemma_ple_altup'
    return 'other'

with tempfile.TemporaryDirectory() as tmp:
    for src in sys.argv[2:]:
        parts = [open_reader(u, os.path.join(tmp, str(i))) for i, u in enumerate(src.split('+'))]   # split files: 'part1+part2'
        r, size = parts[0][0], sum(p[1] for p in parts); r.tensors = [t for p in parts for t in p[0].tensors]
        kv = {f.name: f.contents() for f in r.fields.values() if not f.name.startswith('tokenizer.ggml.') and not f.name.startswith('tokenizer.chat')}
        arch = kv.get('general.architecture'); print('\n== %s\n   file %.3f GB; arch %s; name %s' % (src.split('+')[0].split('/')[-1], size/1e9, arch, kv.get('general.name')))
        for k in ('block_count', 'embedding_length', 'feed_forward_length', 'attention.head_count', 'attention.head_count_kv', 'vocab_size', 'embedding_length_per_layer_input', 'attention.sliding_window'):
            if arch + '.' + k in kv: print('   %s = %s' % (k, kv[arch + '.' + k]))
        G = collections.defaultdict(lambda: [0, 0, collections.Counter()]); names = {t.name for t in r.tensors}; rep = collections.Counter(); ntok_rows = 0
        for t in r.tensors:
            g = group(t.name); n = int(t.n_elements); b = int(t.n_bytes); tn = t.tensor_type.name
            G[g][0] += n; G[g][1] += b; G[g][2][tn] += b
            sh = [int(x) for x in t.shape]
            if g != 'lookup' and len(sh) >= 2 and tn in REPACK and sh[1] % (8 if '_K' in tn else 4) == 0: rep[tn] += b
        tied = 'output.weight' not in names
        print('   head: %s; vocab rows %s' % ('TIED to token_embd (read in full every token)' if tied else 'untied, ' + next(t.tensor_type.name for t in r.tensors if t.name == 'output.weight'), next(int(t.shape[1]) for t in r.tensors if t.name == 'token_embd.weight')))
        tot = sum(v[1] for v in G.values()); per_tok = sum(v[1] for g, v in G.items() if g != 'lookup')
        if tied: per_tok += next(int(t.n_bytes) for t in r.tensors if t.name == 'token_embd.weight')
        print('   %-16s %10s %10s %8s  types' % ('group', 'params M', 'bytes MB', 'read/tok'))
        for g, (n, b, ty) in sorted(G.items(), key=lambda x: -x[1][1]):
            rd = 'yes' if g != 'lookup' else ('row' if not tied else 'row+head')
            print('   %-16s %10.1f %10.1f %8s  %s' % (g, n/1e6, b/1e6, rd, ', '.join('%s %.0f' % (k, v/1e6) for k, v in ty.most_common())))
        print('   total params %.3f B; bytes per token %.3f GB (all but lookup rows%s); lookup %.3f GB' % (sum(v[0] for v in G.values())/1e9, per_tok/1e9, ', head = embd' if tied else '', G['lookup'][1]/1e9))
        print('   repacked by the phone build (anonymous copy): %.3f GB  (%s)' % (sum(rep.values())/1e9, ', '.join('%s %.0f MB' % (k, v/1e6) for k, v in rep.most_common())))
        print('   lookup tensors:', ', '.join('%s %s %s %.0f MB' % (t.name, t.tensor_type.name, [int(x) for x in t.shape], int(t.n_bytes)/1e6) for t in r.tensors if group(t.name) == 'lookup'))
        if 'other' in G or 'gemma_ple_altup' in G:
            ex = collections.Counter(re.sub(r'^blk\.\d+\.', 'blk.N.', t.name) for t in r.tensors if group(t.name) in ('other', 'gemma_ple_altup'))
            print('   other/ple tensors:', ', '.join('%s x%d' % kv_ for kv_ in sorted(ex.items())[:40]))
