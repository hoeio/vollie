#!/usr/bin/env python3
"""Working set of the official benchmark from a page-touch trace (tools/c2_ws_trace.cpp) -> access spec (JSON).

The tracer records, per phase (0 = prefill, i = decode graph i), which host pages (16 KB on Apple silicon) of the
mmapped model file the runtime touched. This script turns that into units on 4 KB pages, the phone's page size:
  - one unit per (layer, expert) slice: gate+up+down, each kind's byte range rounded out to 4 KB pages; a slice counts
    as touched when more than half of its interior host pages were touched (boundary pages are shared with neighbours);
  - one unit per other tensor that was touched (rounded out to 4 KB);
  - token_embd (row gathers, head rescoring) as one unit per touched host page, sized by the expected number of 4 KB
    pages inside it that hold a touched row (rows assumed spread at random: *estimated*, not traced at 4 KB).
Optional --ids adds the prompt's per-token routing (from tools' expert-id tracer: "P tok layer norm id:w ...") so that
predictors can be built from the prompt. Output is the input of tools/c2_pagecache_sim.py.

usage: c2_ws_trace.py --gguf-py DIR model.gguf trace.log [--ids ids.txt] -o spec.json
"""
import argparse, collections, json, math, sys

PG4, HOST = 4096, 16384


def pages4(a, n):
    return (a + n - 1) // PG4 - a // PG4 + 1


def build(gguf_py, model, log, ids=None):
    sys.path.insert(0, gguf_py)
    from gguf import GGUFReader
    r = GGUFReader(model)
    ph = collections.defaultdict(set)
    for l in open(log):
        if l[0] != '#':
            a, b = l.split(); ph[int(a)].add(int(b))
    ph = {p: s for p, s in ph.items() if 0 <= p < 1000}  # first run only
    exps = [t for t in r.tensors if '_exps' in t.name]
    nl = 1 + max(int(t.name.split('.')[1]) for t in exps); ne = int(exps[0].shape[-1])
    units, uid = [], {}
    def unit(name, nbytes, kind, layer=-1):
        if name not in uid:
            uid[name] = len(units); units.append(dict(name=name, bytes=nbytes, kind=kind, layer=layer))
        return uid[name]
    sb = {}
    for t in exps:
        il = int(t.name.split('.')[1]); s = t.n_bytes // ne
        for e in range(ne):
            sb.setdefault((il, e), []).append((t.data_offset + e * s, s))
    for (il, e), parts in sorted(sb.items()):
        unit(f'e:{il}:{e}', sum(pages4(a, n) for a, n in parts) * PG4, 'exp', il)
    dense = [t for t in r.tensors if '_exps' not in t.name]
    phases = []
    for p in sorted(ph):
        pages = ph[p]; lst = []
        for t in dense:
            il = int(t.name.split('.')[1]) if t.name.startswith('blk.') else (nl if t.name.startswith('output') else -1)
            lo, hi = t.data_offset // HOST, (t.data_offset + t.n_bytes - 1) // HOST
            hit = [q for q in range(lo, hi + 1) if q in pages]
            if not hit:
                continue
            if 'token_embd' in t.name:
                rows = int(t.shape[-1]); rb = t.n_bytes // rows; f = len(hit) / (hi - lo + 1)
                q = 1 - (1 - f) ** (rb / HOST)                      # P(row touched) that gives f
                e4 = (HOST // PG4) * (1 - (1 - q) ** (PG4 / rb)) / f  # 4 KB pages per touched host page
                for h in hit:
                    lst.append((il, 0, unit(f'embd:{h}', int(round(e4 * PG4)), 'embd', il)))
                continue
            interior = [q for q in range(-(-t.data_offset // HOST), (t.data_offset + t.n_bytes) // HOST)]
            if interior and sum(q in pages for q in interior) * 2 <= len(interior):
                continue  # only a shared boundary page: the neighbour was read, not this tensor
            lst.append((il, 0, unit('t:' + t.name, pages4(t.data_offset, t.n_bytes) * PG4, 'dense', il)))
        for (il, e), parts in sb.items():
            inner = [q for a, n in parts for q in range(-(-a // HOST), (a + n) // HOST)]
            if sum(q in pages for q in inner) * 2 > len(inner):
                lst.append((il, 1, uid[f'e:{il}:{e}']))
        lst.sort()
        phases.append([u for _, _, u in lst])
    spec = dict(model=model.split('/')[-1], page=PG4, layers=nl, experts=ne, units=units, phases=phases)
    if ids:
        routes = collections.defaultdict(list)
        for l in open(ids):
            if l[0] != 'P':
                continue
            x = l.split()
            routes[(int(x[1]), int(x[2]), x[3])] = [(int(a), float(b)) for a, b in (y.split(':') for y in x[4:])]
        norm = any(k[2] == '1' for k in routes)
        toks = sorted({k[0] for k in routes})
        spec['prompt_routes'] = [[[il, routes[(tk, il, '1' if norm else '0')]] for il in range(nl)
                                  if (tk, il, '1' if norm else '0') in routes] for tk in toks]
        spec['prompt_weights_normalised'] = norm
    return spec


def summary(spec):
    U = spec['units']; mb = lambda us: sum(U[u]['bytes'] for u in us) / 1e6
    pre, dec = spec['phases'][0], spec['phases'][1:]
    by = lambda us, k: [u for u in us if U[u]['kind'] == k]
    dd = set().union(*map(set, dec)); un = set(pre) | dd
    out = {'prefill_MB': mb(pre), 'prefill_slices': len(by(pre, 'exp')), 'decode_distinct_MB': mb(dd),
           'decode_distinct_slices': len(by(dd, 'exp')), 'union_MB': mb(un), 'union_slices': len(by(un, 'exp')),
           'decode_per_token_MB': sum(map(mb, dec)) / len(dec),
           'decode_per_token_expert_MB': sum(mb(by(d, 'exp')) for d in dec) / len(dec),
           'decode_per_token_dense_MB': sum(mb(by(d, 'dense')) for d in dec) / len(dec),
           'decode_per_token_embd_MB': sum(mb(by(d, 'embd')) for d in dec) / len(dec),
           'union_by_kind_MB': {k: mb(by(un, k)) for k in ('exp', 'dense', 'embd')},
           'decode_slices_not_in_prefill': len(set(by(dd, 'exp')) - set(pre))}
    return {k: (round(v, 1) if isinstance(v, float) else v) for k, v in out.items()}


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--gguf-py', required=True); ap.add_argument('model'); ap.add_argument('log')
    ap.add_argument('--ids'); ap.add_argument('-o', required=True)
    a = ap.parse_args()
    s = build(a.gguf_py, a.model, a.log, a.ids)
    json.dump(s, open(a.o, 'w'), separators=(',', ':'))
    print(json.dumps(summary(s), indent=1))
