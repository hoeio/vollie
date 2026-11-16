#!/usr/bin/env python3
"""Routing-prior traces (patches/llama.cpp-c1-olmoe/0007, VOLLIE_PRIOR_DUMP) -> residency numbers. *estimated* only.

A dump is a list of 32-byte records (int32 pos, int16 layer, int16 pad, uint64 sel, kept, base), one per token and MoE
layer, in evaluation order (sel = selected experts, kept = after skip, base = the unbiased top k). A perplexity run gives
one sequence per chunk; a chunk starts where a layer's position goes down.

  stats  <gguf> <dump>...             per-token kept MB, new MB against the previous token, overlap, 32-token working set
  cmp    <dumpA> <dumpB>              (layer, chunk, pos) where sel differs, and the first differing position per chunk
  replay <dump_p> <VOLLIE_PRIOR> <N> <k> [J]  the rule written out sequentially on VOLLIE_PRIOR_DUMP_P's probs: rows
                                      whose selection differs from the op's (0 = the op computes the defined rule)
  sim    <gguf> <dump> [--cache-gb ...] benchmark-shaped windows (156 prefill + 32 decode positions per chunk, from
                                      position 300) through tools/c2_pagecache_sim.py, back-to-back runs (memo-18)
  --selftest
"""
import argparse, collections, os, sys

import numpy as np

REC = np.dtype([('pos', '<i4'), ('layer', '<i2'), ('pad', '<i2'), ('sel', '<u8'), ('kept', '<u8'), ('base', '<u8')])
PG4 = 4096


def load(path):
    d = np.fromfile(path, dtype=REC)
    chunk = np.zeros(len(d), dtype=np.int32)
    l0 = int(d['layer'].min()) if len(d) else 0
    last, cur = None, 0
    for i, (p, l) in enumerate(zip(d['pos'], d['layer'])):
        if l == l0:  # the first MoE layer sees every token, before the other layers of its graph
            if last is not None and p <= last:
                cur += 1
            last = p
        chunk[i] = cur
    return d, chunk


def popcount(a):
    a = a.astype(np.uint64)
    return np.array([bin(int(x)).count('1') for x in a], dtype=np.int64)


def slice_bytes(gguf):
    """bytes of one routed expert slice (gate + up + down, each rounded out to 4 KB pages) and per-layer dense bytes
    read through the mapping every token (router, plus Granite's conv1d and Q4_1 shared down), *estimated* from the header"""
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'lab', 'llama.cpp-c1pr', 'gguf-py'))
    from gguf import GGUFReader
    r = GGUFReader(gguf)
    exps = [t for t in r.tensors if '_exps' in t.name]
    ne = int(exps[0].shape[-1])
    sz = collections.defaultdict(int)
    for t in exps:
        il = int(t.name.split('.')[1]); s = t.n_bytes // ne
        a0 = t.data_offset  # mean 4 KB pages per expert slice of this kind, as tools/c2_ws_trace.py rounds them
        sz[il] += PG4 * sum((a0 + e * s + s - 1) // PG4 - (a0 + e * s) // PG4 + 1 for e in range(ne)) // ne
    dense = collections.defaultdict(int)
    for t in r.tensors:
        if t.name.startswith('blk.') and ('ffn_gate_inp' in t.name or 'ssm_conv1d' in t.name or
                                          ('ffn_down_shexp' in t.name and t.tensor_type.name == 'Q4_1')):
            dense[int(t.name.split('.')[1])] += t.n_bytes
    return dict(sz), dict(dense), ne


def per_chunk(d, chunk):
    """{chunk: {layer: {pos: (sel, kept, base)}}}"""
    out = collections.defaultdict(lambda: collections.defaultdict(dict))
    for c, p, l, s, k, b in zip(chunk, d['pos'], d['layer'], d['sel'], d['kept'], d['base']):
        out[int(c)][int(l)][int(p)] = (int(s), int(k), int(b))
    return out


def stats(gguf, path, lo=1, hi=None):
    sz, _, _ = slice_bytes(gguf)
    d, chunk = load(path)
    pc = per_chunk(d, chunk)
    pc = {c: v for c, v in pc.items() if len(v[0]) > 64}  # drop the warm-up graph
    tok = kept_mb = new_mb = forced = sel_n = ov = kept_n = 0
    ws, wn = 0.0, 0
    for c, layers in pc.items():
        L0 = layers[min(layers)]
        for p in L0:
            if not all(p in ent for ent in layers.values()):
                continue  # per-token numbers over tokens every layer routed (a perplexity chunk prunes the last layer below 256)
            tok += 1
            for l, ent in layers.items():
                if p not in ent:
                    continue
                s, k, b = ent[p]
                nk = bin(k).count('1'); kept_n += nk; kept_mb += nk * sz[l] / 1e6
                sel_n += bin(s).count('1'); forced += bin(s & ~b).count('1')
                if p - 1 in ent:
                    pk = ent[p - 1][1]
                    new_mb += bin(k & ~pk).count('1') * sz[l] / 1e6
                    ov += bin(k & pk).count('1')
                else:
                    new_mb += nk * sz[l] / 1e6
        for w0 in range(0, 512, 32):  # aligned 32-position windows, all layers present
            mb, ok = 0.0, True
            for l, ent in layers.items():
                if not all(q in ent for q in range(w0, w0 + 32)):
                    ok = l == max(layers) and w0 < 256  # pruned last layer of a perplexity chunk: no rows below 256
                    if not ok:
                        break
                    continue
                u = 0
                for q in range(w0, w0 + 32):
                    u |= ent[q][1]
                mb += bin(u).count('1') * sz[l] / 1e6
            if ok and w0 >= 256:  # windows with every layer, the last included
                ws += mb; wn += 1
    return dict(chunks=len(pc), tokens=tok, kept_MB_per_token=kept_mb / tok, new_MB_per_token=new_mb / tok,
                overlap_prev_kept=ov / kept_n, forced_per_token=forced / tok, ws32_MB=ws / max(wn, 1), windows=wn)


def cmp(pa, pb):
    A, B = (per_chunk(*load(p)) for p in (pa, pb))
    n = diff = 0
    first = []
    for c in sorted(set(A) & set(B)):
        if len(A[c][0]) <= 64:
            continue
        f = None
        for l in A[c]:
            for p, (s, _, _) in A[c][l].items():
                if p in B[c].get(l, {}):
                    n += 1
                    if B[c][l][p][0] != s:
                        diff += 1
                        f = p if f is None else min(f, p)
        first.append(f)
    return dict(layer_tokens=n, sel_differs=diff, frac=diff / max(n, 1), first_diff_pos_per_chunk=first)


def sim(gguf, path, caps, start=300, npre=156, ndec=32, runs=4):
    sys.path.insert(0, os.path.dirname(__file__))
    import c2_pagecache_sim as S
    sz, dense, ne = slice_bytes(gguf)
    d, chunk = load(path)
    pc = {c: v for c, v in per_chunk(d, chunk).items() if len(v[0]) > 64}
    layers = sorted(sz)
    units, uid = [], {}
    def unit(name, b, kind, layer):
        if name not in uid:
            uid[name] = len(units); units.append(dict(name=name, bytes=b, kind=kind, layer=layer))
        return uid[name]
    for l in layers:
        for e in range(ne):
            unit(f'e:{l}:{e}', sz[l], 'exp', l)
        if dense.get(l):
            unit(f'd:{l}', dense[l], 'dense', l)
    res = {C: collections.defaultdict(list) for C in caps}
    for c, L in pc.items():
        def phase(ps):
            lst = []
            for l in layers:
                if f'd:{l}' in uid:
                    lst.append(uid[f'd:{l}'])
                m = 0
                for p in ps:
                    if p in L.get(l, {}):
                        m |= L[l][p][1]
                lst += [uid[f'e:{l}:{e}'] for e in range(ne) if m >> e & 1]
            return lst
        phases = [phase(range(start, start + npre))] + [phase([p]) for p in range(start + npre, start + npre + ndec)]
        spec = S.index(dict(units=units, experts=ne, size=[u['bytes'] for u in units], phases=phases))
        for C in caps:
            r = S.steady(spec, C * 1e9, runs=runs)
            res[C]['decode_MB'].append(sum(r['decode_MB']))
            res[C]['prefill_MB'].append(r['prefill_MB'])
        ds = set().union(*map(set, phases[1:]))
        res[caps[0]]['decode_set_MB'].append(sum(units[u]['bytes'] for u in ds) / 1e6)
    out = {}
    for C in caps:
        out[C] = {k: float(np.mean(v)) for k, v in res[C].items()}
    return out


def replay(path, lam, n, k, j=0):
    """VOLLIE_PRIOR_DUMP_P file: the rule written out sequentially (ring keyed by position, entry valid if its tag
    matches), applied row by row in file order to the dumped probs; returns (rows, rows whose selection differs)."""
    raw = open(path, 'rb').read()
    off, rows, bad = 0, 0, 0
    tag, sel = {}, {}
    m = np.float32(np.exp(np.float32(lam)))
    while off < len(raw):
        p, l, ne0, _ = np.frombuffer(raw, '<i4', 4, off); off += 16
        x = np.frombuffer(raw, '<f4', ne0, off); off += 4 * ne0
        got = int(np.frombuffer(raw, '<u8', 1, off)[0]); off += 8
        h = 0
        for d in range(1, n + 1):
            q = p - d
            if q < 0:
                break
            if tag.get((l, q % 4096)) == q:
                h |= sel[(l, q % 4096)]
        ob = sorted(range(ne0), key=lambda e: -x[e])  # stable, like the insertion sort
        for s in ob[:j]:
            h |= 1 << s
        sc = [x[e] * m if h >> e & 1 else x[e] for e in range(ne0)]
        o = sorted(range(ne0), key=lambda e: -sc[e])
        want = sum(1 << e for e in o[:k])
        tag[(l, p % 4096)] = p; sel[(l, p % 4096)] = want  # the replay keeps its own history
        rows += 1; bad += want != got
    return rows, bad


def _selftest():
    import tempfile
    rec = np.zeros(6, dtype=REC)
    rec['pos'] = [0, 1, 2, 0, 1, 2]; rec['layer'] = 0
    rec['sel'] = [3, 6, 6, 3, 5, 5]; rec['kept'] = rec['sel']; rec['base'] = rec['sel']
    f = tempfile.NamedTemporaryFile(delete=False); rec.tofile(f.name)
    d, ch = load(f.name)
    assert list(ch) == [0, 0, 0, 1, 1, 1]
    pc = per_chunk(d, ch)
    assert pc[1][0][1] == (5, 5, 5)
    os.unlink(f.name)
    print('selftest OK')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['stats', 'cmp', 'sim', 'replay', 'selftest'])
    ap.add_argument('args', nargs='*')
    ap.add_argument('--cache-gb', type=float, nargs='+', default=[1.6, 1.8, 2.0, 2.2, 2.4])
    a = ap.parse_args()
    if a.cmd == 'selftest':
        _selftest()
    elif a.cmd == 'stats':
        for p in a.args[1:]:
            print(os.path.basename(p), {k: (round(v, 4) if isinstance(v, float) else v) for k, v in stats(a.args[0], p).items()})
    elif a.cmd == 'replay':  # replay <dump_p> <VOLLIE_PRIOR> <N> <k> [J]
        r = replay(a.args[0], float(a.args[1]) / 1000, int(a.args[2]), int(a.args[3]), int(a.args[4]) if len(a.args) > 4 else 0)
        print(f'rows={r[0]} differ={r[1]}')
    elif a.cmd == 'cmp':
        r = cmp(*a.args); print({k: v for k, v in r.items() if k != 'first_diff_pos_per_chunk'}, 'first diff pos:', r['first_diff_pos_per_chunk'])
    else:
        for C, v in sim(a.args[0], a.args[1], a.cache_gb).items():
            print(f'C={C:.1f} GB', {k: round(x, 1) for k, x in v.items()})
