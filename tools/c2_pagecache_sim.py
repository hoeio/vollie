#!/usr/bin/env python3
"""Page-cache simulator for the official benchmark (campaign 2, memo-18). *estimated* numbers only.

Input: an access spec from tools/c2_ws_trace.py (units on 4 KB pages, one access list per phase: 0 = prefill,
1..32 = decode graphs). The cache holds this file's pages up to C bytes, LRU at unit granularity with partial
residency (all pages of a slice are read together, so they age together; the oldest unit is trimmed first).
Linux 5.4 keeps file pages on an active/inactive pair with a second chance for referenced mapped pages; for a
single process sweeping its own file that approximates LRU, which is what this models.

Library use (memo-18 drives it from a scratch script):
    spec = load(path); c = Cache(C_bytes); res = run(spec, c, Options(...))
res: dict with 'prefill_MB', 'decode_MB' (per token), 'predecode_MB', 'faults' (per token, from the fault model).

  python3 c2_pagecache_sim.py --selftest
  python3 c2_pagecache_sim.py spec.json --cache-gb 1.8 2.2 [--runs 4]
"""
import argparse, collections, json, sys

MB = 1e6


class Cache:
    def __init__(self, cap):
        self.cap, self.od, self.total = int(cap), collections.OrderedDict(), 0

    def access(self, u, size):
        """Touch unit u (size bytes); returns bytes read from flash. u becomes most recent."""
        have = self.od.pop(u, 0)
        self.od[u] = size
        self.total += size - have
        self._evict()
        return size - have

    def willneed(self, u, size):
        """MADV_WILLNEED: reads what is missing; a page already resident keeps its age (readahead skips it)."""
        have = self.od.get(u, 0)
        if have >= size:
            return 0
        self.od.pop(u, None); self.total -= have
        self.od[u] = size; self.total += size
        self._evict()
        return size - have

    def drop(self, u):  # MADV_PAGEOUT
        self.total -= self.od.pop(u, 0)

    def cold(self, u):  # MADV_COLD: to the reclaim end, still resident
        if u in self.od:
            self.od.move_to_end(u, last=False)

    def shrink_to(self, target):  # reclaim by the system during a wait: oldest first
        while self.total > target and self.od:
            u, s = next(iter(self.od.items()))
            cut = min(s, self.total - target)
            if cut >= s:
                self.od.popitem(last=False)
            else:
                self.od[u] = s - cut
            self.total -= cut

    def _evict(self):
        if self.total > self.cap:
            last = next(reversed(self.od))
            while self.total > self.cap:
                u, s = next(iter(self.od.items()))
                if u == last:
                    break
                cut = min(s, self.total - self.cap)
                if cut >= s:
                    self.od.popitem(last=False)
                else:
                    self.od[u] = s - cut
                self.total -= cut

    def resident(self, u):
        return self.od.get(u, 0)


class Options:
    def __init__(self, io=1, guess_miss=0.15, predecode=None, predecode_mode='touch', drop=None, drop_mode='pageout',
                 phases=None, seed=1):
        self.io = io                    # GGML_EXPERT_IO: 1 = WILLNEED on routed slices; 5 = + next-layer guess (bit 4)
        self.guess_miss = guess_miss    # bit 4: fraction of guessed slices that are wrong (read for nothing)
        self.predecode = predecode      # ordered unit list re-read after the prefill, before decode token 1
        self.predecode_mode = predecode_mode  # 'touch' (refreshes resident units too) or 'willneed' (missing only)
        self.drop = drop                # set of units to keep: after each prefill layer, the layer's other slices go
        self.drop_mode = drop_mode      # 'pageout' (dropped) or 'cold' (moved to the reclaim end)
        self.phases = phases            # replacement phase lists (skip variants); default spec['phases']
        self.seed = seed


def load(path):
    s = json.load(open(path))
    s['size'] = [u['bytes'] for u in s['units']]
    return s


def run(spec, cache, opt=None):
    opt = opt or Options()
    U, size = spec['units'], spec['size']
    phases = opt.phases or spec['phases']
    res = {'prefill_MB': 0.0, 'decode_MB': [], 'predecode_MB': 0.0, 'decode_exp_MB': [], 'guess_MB': []}
    pre = phases[0]
    cur_layer, layer_units = None, []
    def end_layer():
        if opt.drop is not None:
            for v in layer_units:
                if U[v]['kind'] == 'exp' and v not in opt.drop:
                    (cache.drop if opt.drop_mode == 'pageout' else cache.cold)(v)
    for u in pre:
        L = U[u]['layer']
        if L != cur_layer:
            end_layer(); cur_layer, layer_units = L, []
        res['prefill_MB'] += cache.access(u, size[u]) / MB
        layer_units.append(u)
    end_layer()
    if opt.predecode:
        f = cache.access if opt.predecode_mode == 'touch' else cache.willneed
        for u in opt.predecode:
            res['predecode_MB'] += f(u, size[u]) / MB
    ne = spec['experts']
    rng = opt.seed
    for d in phases[1:]:
        tot = exp = gs = 0.0
        if opt.io & 4:  # wrong next-layer guesses: read in, never used by this token
            by_layer = collections.defaultdict(list)
            for u in d:
                if U[u]['kind'] == 'exp':
                    by_layer[U[u]['layer']].append(u)
            for L, us in by_layer.items():
                nwrong = int(round(len(us) * opt.guess_miss))
                used = {int(U[u]['name'].split(':')[2]) for u in us}
                for _ in range(nwrong):
                    rng = (rng * 1103515245 + 12345) & 0x7fffffff
                    e = rng % ne
                    while e in used:
                        e = (e + 1) % ne
                    used.add(e)
                    v = spec['_uid'][f'e:{L}:{e}'] if '_uid' in spec else next(
                        i for i, x in enumerate(U) if x['name'] == f'e:{L}:{e}')
                    gs += cache.willneed(v, size[v]) / MB
        for u in d:
            m = cache.access(u, size[u]) / MB
            tot += m
            if U[u]['kind'] == 'exp':
                exp += m
        res['decode_MB'].append(tot + gs); res['decode_exp_MB'].append(exp); res['guess_MB'].append(gs)
    return res


def index(spec):
    spec['_uid'] = {u['name']: i for i, u in enumerate(spec['units'])}
    return spec


def steady(spec, cap_bytes, opt=None, runs=4, wait_keep=None):
    """Back-to-back runs of the same prompt (or with a wait that leaves wait_keep bytes); returns the last run."""
    c = Cache(cap_bytes); r = None
    for i in range(runs):
        if wait_keep is not None and i > 0:
            c.shrink_to(wait_keep)
        r = run(spec, c, opt)
    return r


def _selftest():
    # LRU with partial units
    c = Cache(10)
    assert c.access('a', 4) == 4 and c.access('b', 4) == 4 and c.access('a', 4) == 0
    assert c.access('c', 4) == 4 and c.resident('b') == 2 and c.total == 10   # b trimmed (oldest), a kept
    assert c.willneed('b', 4) == 2 and list(c.od)[-1] == 'b'
    assert c.resident('a') == 2                                               # a trimmed in turn
    assert c.willneed('c', 4) == 0 and list(c.od) == ['a', 'c', 'b']          # resident: no read, age kept
    c.shrink_to(4); assert c.total == 4
    c.drop('b'); assert 'b' not in c.od
    # a synthetic two-layer model: prefill sweeps 6 slices, decode uses 2 per token
    units = [dict(name=f'e:{L}:{e}', bytes=1, kind='exp', layer=L) for L in range(2) for e in range(3)]
    spec = index(dict(units=units, experts=3, size=[1] * 6, phases=[[0, 1, 2, 3, 4, 5]] + [[0, 3]] * 4))
    r = steady(spec, 6, runs=3)
    assert r['prefill_MB'] == 0 and sum(r['decode_MB']) == 0           # everything fits
    r = steady(spec, 4, runs=3)
    assert round(r['prefill_MB'] * MB) == 4 and round(r['decode_MB'][0] * MB) == 1  # sweep > C: the sweep evicts the decode set
    r = steady(spec, 4, Options(drop={0, 3}), runs=3)
    assert round(r['prefill_MB'] * MB) == 4 and sum(r['decode_MB']) == 0  # keep the decode set, drop the rest
    r = steady(spec, 4, Options(predecode=[0, 3]), runs=3)
    assert round(r['predecode_MB'] * MB) == 1 and sum(r['decode_MB']) == 0       # 3 was still resident, 0 is re-read
    print('selftest OK')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('spec', nargs='?'); ap.add_argument('--cache-gb', type=float, nargs='+', default=[1.8, 2.2])
    ap.add_argument('--runs', type=int, default=4); ap.add_argument('--selftest', action='store_true')
    a = ap.parse_args()
    if a.selftest or not a.spec:
        _selftest(); sys.exit(0)
    s = index(load(a.spec))
    for C in a.cache_gb:
        r = steady(s, C * 1e9, runs=a.runs)
        print(f'C={C:.1f} GB: prefill {r["prefill_MB"]:7.0f} MB, decode {sum(r["decode_MB"]):6.0f} MB '
              f'(tokens 1-8: {sum(r["decode_MB"][:8]):5.0f}, 9-32: {sum(r["decode_MB"][8:]):5.0f})')
