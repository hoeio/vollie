#!/usr/bin/env python3
"""Campaign 2 research-kit report parser (kits r17, r18; schemas in campaigns/02-25toks-5pct/reports/memo-10-kit-r17.md
and memo-12-kit-r18.md). Standard library only.

usage: python3 -I tools/c2_research_parse.py <report.json | clipboard text "c2gz:..."> [--min-mhz 1800]
           [--prompt tools/c2_official_prompt.txt] [--sigma 0.4]
       python3 -I tools/c2_research_parse.py --selftest

Prints the identity checks (model, libraries against the build record, chat template, prompt, greedy texts), one line
per group (n, median / min / max printed tok/s, steady ms/token, clocks, faults, counters, hygiene flags), and for
every A/B run id (kind 'abba') B minus A over counted blocks: per-block mean ms/token of the graph (per-token lines)
and of the wall clock (chunk arrival times), blocks after the warm-up only (blk >= 1), clean blocks only (no major
fault, big-core clock >= --min-mhz where the clock is readable), sigma = sqrt(sA^2/nA + sB^2/nB). Campaign 1's rule:
a sub-1 ms effect needs sigma <= 0.4, else it is unresolved. All numbers are the phone's (*device*)."""
import argparse
import base64
import gzip
import json
import os
import re
import statistics as st
import sys

SHA = '5abf710b775cd64fd89d74a783c8a7056a409ba795cd401b9dd448ef8e444abb'
BIG = (4, 5, 6, 7)  # Snapdragon 888: cpu4-6 A78, cpu7 X1
HERE = os.path.dirname(os.path.abspath(__file__))
# memo-11 §3 (frozen, unanchored)
TOK_RE = re.compile(r'cpu-prof ab=([AB-]) tok=(\d+) us/token=(\d+) majflt=(\d+) minflt=(\d+) mhz=(\d+)/(\d+) '
                    r'run=([A-Za-z0-9._-]+) prompt=(\d+) blk=(\d+) threads=(\d+) ops=(\d+) barriers=(\d+) syncs=(\d+) '
                    r'skipped=(\d+) pairs=(\d+) head=(\d+) router=(\d+) mm=(\d+) mmid=(\d+) attn=(\d+) other=(\d+) '
                    r'wait=(\d+) \|')
TOK_FIELDS = ['ab', 'tok', 'us', 'majflt', 'minflt', 'mhz4', 'mhz7', 'prompt', 'blk', 'threads', 'ops', 'barriers',
              'syncs', 'skipped', 'pairs', 'head', 'router', 'mm', 'mmid', 'attn', 'other', 'wait']
CLASSES = ['mmid', 'mm', 'head', 'router', 'attn', 'other', 'wait']
COUNTS = ['ops', 'barriers', 'syncs', 'skipped', 'pairs']
SKIP_RE = re.compile(r'c2-skip-run: .*?expert_bytes_skipped=([\d.]+)%')
HEAD_RE = re.compile(r'c2-head-run: .*?k=(\d+) .*?exact_rows_per_col=([\d.]+)')
GRAPH_RE = re.compile(r'c2-graph-run: .*?ops_run=([\d.]+) ops_tiny=([\d.]+) barriers=([\d.]+) syncs=([\d.]+)')


def load(path):
    with open(path, encoding='utf-8') as f:
        s = f.read().strip()
    if s.startswith('c2gz:'):
        s = gzip.decompress(base64.b64decode(s[5:])).decode('utf-8')
    return json.loads(s)


def mhz(s):
    m = re.match(r'mhz=([\d,]+)', s or '')
    return [int(x) for x in m.group(1).split(',')] if m else []


def big_min(s):
    v = mhz(s)
    return min(v[i] for i in BIG) if len(v) > max(BIG) else None


def tokens(run, fields=TOK_FIELDS):
    """Per-token records of a run: the compact tuples (r18) or parsed full lines (r17 'prof', r16-style lines)."""
    out = [dict(zip(fields, t)) for t in run.get('tok') or []]
    if out:
        return out
    for line in run.get('prof') or []:
        m = TOK_RE.search(line)
        if m:
            g = m.groups()
            out.append(dict(zip(TOK_FIELDS, [g[0]] + [int(x) for i, x in enumerate(g[1:], 2) if i != 8])))
        elif 'tok=' in line:  # older numeric key=value lines (campaign 1 style)
            f = {k: float(v) for k, v in re.findall(r'(\w[\w/]*)=(-?[\d.]+)', line.split('|')[0])}
            if 'tok' in f:
                out.append({'ab': '-', 'tok': int(f['tok']), 'us': f.get('us/token', 0), 'majflt': f.get('majflt', 0),
                            'blk': int(f.get('blk', 0))})
    return out


def steady_ms(run, fields=TOK_FIELDS):
    tl = [t for t in tokens(run, fields) if t['tok'] >= 2 and t.get('us')]
    if tl:
        return st.median(t['us'] / 1000 for t in tl), 'prof'
    c = run.get('chunk_ms') or []
    d = [b - a for a, b in zip(c, c[1:])]
    return (st.median(d), 'chunk') if d else (None, None)


def decode_majflt(run, fields=TOK_FIELDS):
    tl = tokens(run, fields)
    return sum(int(t.get('majflt', 0)) for t in tl) if tl else None


def fmt(x, nd=2):
    return '-' if x is None else f'{x:.{nd}f}'


def expected_render(prompt):
    # OLMoE-0125-Instruct's GGUF template: bos_token + '<|user|>\n' + content + '\n' + '<|assistant|>\n'
    return '<|endoftext|><|user|>\n' + prompt + '\n<|assistant|>\n'


def env_of(run):
    return run.get('env_set') or run.get('env') or {}


def checks(r, prompt_file):
    out = []
    m = r.get('model') or {}
    ok = m.get('bytes') == m.get('expected_bytes') and m.get('verified_marker') == SHA and m.get('sha256_end') == SHA
    out.append(f"model {m.get('file')} {m.get('bytes')} B, marker {str(m.get('verified_marker'))[:8]}, "
               f"sha256 at end {str(m.get('sha256_end'))[:8]} -> {'OK' if ok else 'CHECK'}")
    built = (r.get('build') or {}).get('libs') or {}
    for k, v in sorted((r.get('libs') or {}).items()):
        tag = '' if k not in built else (' = build' if built[k] == v else ' != BUILD ' + built[k][:12])
        out.append(f'lib {k} {v}{tag}')
    b = r.get('build') or {}
    if b:
        out.append(f"build llama.cpp {str(b.get('llama_cpp'))[:9]}; series " +
                   ' '.join(f"{p['file'][:4]}:{p['sha256'][:8]}" for p in b.get('series', [])) + '; app ' +
                   ' '.join(f"{p['file'][:4]}:{p['sha256'][:8]}" for p in b.get('app_patches', [])))
    t = r.get('template') or {}
    p = r.get('prompt', '')
    same = t.get('rendered') == expected_render(p)
    out.append(f"template arch={t.get('architecture')} custom={t.get('custom_template')} rendered_tokens="
               f"{t.get('rendered_tokens')} rendered==OLMoE's own: {'yes' if same else 'NO'}")
    if prompt_file and os.path.exists(prompt_file):
        with open(prompt_file, encoding='utf-8') as f:
            out.append('prompt == ' + os.path.basename(prompt_file) + ': ' + ('yes' if f.read() == p else 'NO'))
    pt = sorted({x.get('prompt_tokens') for x in r.get('runs', [])
                 if x.get('prompt_tokens') is not None and x.get('kind', 'official32') != 'abba'})
    out.append(f'prompt_tokens seen in official runs: {pt}')
    if r.get('unmatched_lines'):
        out.append(f"unmatched runtime lines: {len(r['unmatched_lines'])}")
    return out


def identity(r):
    """Greedy texts of the 32-token runs: equal within a numerics class (repack, skip, head2); stock == stack."""
    out, cls = [], {}
    for x in r.get('runs', []):
        if x.get('kind') == 'abba' or x.get('text') is None:
            continue
        e = env_of(x)
        key = (x.get('repack_loaded', x.get('repack')), e.get('VOLLIE_SKIP_MASS'), e.get('VOLLIE_HEAD2'))
        cls.setdefault(key, []).append(x)
    for key, rs in cls.items():
        texts = {x['text'] for x in rs}
        ids = sorted({x.get('id') for x in rs})
        out.append(f"text class repack={key[0]} skip_mass={key[1]} head2={key[2]}: {len(rs)} runs {ids}: "
                   + ('identical' if len(texts) == 1 else f'DIFFER ({len(texts)} texts)'))
    by_id = {}
    for x in r.get('runs', []):
        if x.get('kind') != 'abba' and x.get('text') is not None:
            by_id.setdefault(x.get('id'), set()).add(x['text'])
    if 'stock' in by_id and 'stack' in by_id:
        out.append('stock == stack: ' + ('yes' if by_id['stock'] == by_id['stack'] and len(by_id['stock']) == 1 else 'NO'))
    if 'skip' in by_id and 'package' in by_id:
        out.append('skip == package (head2 changes no greedy token): ' + ('yes' if by_id['skip'] == by_id['package'] else 'no'))
    return out


def counters(rs):
    sk = [float(m.group(1)) for x in rs for l in x.get('c2') or [] for m in [SKIP_RE.search(l)] if m]
    hd = [float(m.group(2)) for x in rs for l in x.get('c2') or [] for m in [HEAD_RE.search(l)] if m]
    gr = [tuple(map(float, m.groups())) for x in rs for l in x.get('c2') or [] for m in [GRAPH_RE.search(l)] if m]
    s = []
    if sk:
        s.append(f'expert bytes skipped {fmt(st.median(sk))} %')
    if hd:
        s.append(f'head2 exact rows/col {fmt(st.median(hd), 0)}')
    if gr:
        s.append('graph ops/tiny/barriers/syncs ' + '/'.join(fmt(st.median(g[i] for g in gr), 0) for i in range(4)))
    return '; '.join(s)


def groups(r, min_mhz):
    fields = r.get('tok_fields') or TOK_FIELDS
    runs = r.get('runs', [])
    order, by = [], {}
    for x in runs:
        g = x.get('group', '?')
        if g not in by:
            order.append(g)
            by[g] = []
        by[g].append(x)
    ref = next((x.get('text') for x in runs if x.get('kind', 'official32') != 'abba' and not x.get('env') and x.get('text')), None)
    out = []
    for g in order:
        rs = by[g]
        tps = [x['decode_tps'] for x in rs if x.get('decode_tps') is not None]
        sms = [steady_ms(x, fields) for x in rs]
        sm = [s for s, _ in sms if s is not None]
        src = {k for _, k in sms if k}
        clk = [c for c in (big_min(x.get('mhz_after')) for x in rs) if c is not None]
        flags = []
        for x in rs:
            why = []
            if x.get('error'):
                why.append('error')
            c = big_min(x.get('mhz_after'))
            if c is not None and c < min_mhz:
                why.append(f'clock {c}<{min_mhz}')
            dm = decode_majflt(x, fields)
            if dm:
                why.append(f'decode majflt {dm}')
            if x.get('gate_timeout'):
                why.append('gate timeout')
            if x.get('cached_prompt_tokens'):
                why.append(f"cached {x['cached_prompt_tokens']}")
            if x.get('kind', 'official32') != 'abba' and not x.get('env') and ref is not None and x.get('text') != ref:
                why.append('text differs from the first stock run')
            if why:
                flags.append(f"{x.get('id')}#{x.get('rep')} (" + ', '.join(why) + ')')
        cnt = counters(rs)
        out.append(
            f"{g}: n {len(rs)} printed tok/s median {fmt(st.median(tps) if tps else None)} min {fmt(min(tps) if tps else None)} "
            f"max {fmt(max(tps) if tps else None)} | steady ms/token {fmt(st.median(sm) if sm else None)}"
            f"{' (' + '+'.join(sorted(src)) + ')' if src else ''} | big-core MHz after: min {min(clk) if clk else '-'} "
            f"median {st.median(clk) if clk else '-'} | majflt median {st.median([x.get('majflt', 0) for x in rs])} "
            f"minflt median {st.median([x.get('minflt', 0) for x in rs])}{' | ' + cnt if cnt else ''} | "
            f"flags: {'; '.join(flags) or 'none'}")
    return out


def blocks(run, min_mhz, fields=TOK_FIELDS):
    """Blocks (blk >= 1) of one A/B generation: {blk: dict(side, clean, graph ms, wall ms, class ms, counts)}."""
    toks = tokens(run, fields)
    c = run.get('chunk_ms') or []
    wall = {k: c[k] - c[k - 1] for k in range(1, len(c))}  # chunk k arrives after decode graph tok=k (+ sampling)
    by = {}
    for t in toks:
        if t.get('blk', 0) >= 1 and t.get('ab') in ('A', 'B'):
            by.setdefault(t['blk'], []).append(t)
    out = {}
    for b, ts in by.items():
        sides = {t['ab'] for t in ts}
        clk_ok = all((t.get('mhz4', 0) == 0 or t['mhz4'] >= min_mhz) and (t.get('mhz7', 0) == 0 or t['mhz7'] >= min_mhz)
                     for t in ts)
        w = [wall[t['tok']] for t in ts if t['tok'] in wall]
        out[b] = {
            'side': sides.pop() if len(sides) == 1 else '?',
            'clean': clk_ok and all(t.get('majflt', 0) == 0 for t in ts),
            'graph': st.mean(t['us'] for t in ts) / 1000,
            'wall': st.mean(w) if len(w) == len(ts) else None,
            **{k: st.mean(t.get(k, 0) for t in ts) / 1000 for k in CLASSES},
            **{'n_' + k: st.mean(t.get(k, 0) for t in ts) for k in COUNTS},
        }
    return out


def diff(a, b):
    if len(a) < 2 or len(b) < 2:
        return None, None
    return st.mean(b) - st.mean(a), (st.variance(a) / len(a) + st.variance(b) / len(b)) ** 0.5


def verdict(d, s, sigma_max):
    if d is None:
        return 'too few blocks'
    if s <= sigma_max:
        return 'resolved'
    if abs(d) >= 1.0 and abs(d) >= 2 * s:
        return 'resolved (|d| >= 1 ms, sigma above the sub-1 ms limit)'
    return f'UNRESOLVED (sigma {s:.2f} > {sigma_max})'


def ab(r, min_mhz, sigma_max):
    fields = r.get('tok_fields') or TOK_FIELDS
    by = {}
    for x in r.get('runs', []):
        if x.get('kind') == 'abba':
            by.setdefault(x.get('id'), []).append(x)
    out = []
    for rid, rs in by.items():
        bl = [b for x in rs for b in blocks(x, min_mhz, fields).values()]
        side = {s: [b for b in bl if b['side'] == s] for s in 'AB'}
        clean = {s: [b for b in side[s] if b['clean']] for s in 'AB'}
        knob = next((env_of(x).get('VOLLIE_AB') for x in rs if env_of(x).get('VOLLIE_AB')), '?')
        d, s = diff([b['graph'] for b in clean['A']], [b['graph'] for b in clean['B']])
        da, sa = diff([b['graph'] for b in side['A']], [b['graph'] for b in side['B']])  # faults included (EXPERT_IO)
        wa = [b['wall'] for b in clean['A'] if b['wall'] is not None]
        wb = [b['wall'] for b in clean['B'] if b['wall'] is not None]
        dw, sw = diff(wa, wb)
        cls = []
        for k in CLASSES:
            dk, sk = diff([b[k] for b in clean['A']], [b[k] for b in clean['B']])
            if dk is not None:
                cls.append(f'{k} {dk:+.2f}±{sk:.2f}')
        cnt = ' '.join(f"{k} {fmt(st.mean(b['n_' + k] for b in clean['A']), 0)}->{fmt(st.mean(b['n_' + k] for b in clean['B']), 0)}"
                       for k in COUNTS if clean['A'] and clean['B'])
        out.append(
            f"{rid} [{knob}] gens {len(rs)} | blocks A {len(side['A'])} (clean {len(clean['A'])}) B {len(side['B'])} "
            f"(clean {len(clean['B'])}) | graph B-A {fmt(d)} ± {fmt(s)} ms/token: {verdict(d, s, sigma_max)} | "
            f"all blocks {fmt(da)} ± {fmt(sa)} | "
            f"wall B-A {fmt(dw)} ± {fmt(sw)}: {verdict(dw, sw, sigma_max)} | classes {', '.join(cls) or '-'} | counts {cnt or '-'}")
    return out


def main(argv=None):
    a = argparse.ArgumentParser()
    a.add_argument('report', nargs='?')
    a.add_argument('--min-mhz', type=int, default=1800)
    a.add_argument('--sigma', type=float, default=0.4)
    a.add_argument('--prompt', default=os.path.join(HERE, 'c2_official_prompt.txt'))
    a.add_argument('--selftest', action='store_true')
    o = a.parse_args(argv)
    if o.selftest:
        return selftest()
    if not o.report:
        a.error('report.json required')
    r = load(o.report)
    print(f"kit {r.get('kit')} app {r.get('app')} started {r.get('started')} stopped_early {r.get('stopped_early')}")
    for part in (checks(r, o.prompt), identity(r), groups(r, o.min_mhz), ab(r, o.min_mhz, o.sigma)):
        if part:
            print('\n'.join(part))
    return 0


def selftest():
    import random
    import tempfile
    p = 'Read these notes.'
    mk = lambda **k: {'group': 'stock', 'kind': 'official32', 'env': {}, 'rep': 0, 'id': 'stock', 'repack_loaded': True,
                      'mhz_after': 'mhz=1804,1804,1804,1804,2112,2112,2112,2035', 'majflt': 0, 'minflt': 10,
                      'text': 'The ocean', 'cached_prompt_tokens': 0, 'prompt_tokens': 151, **k}
    stack = {'VOLLIE_TINY_OP': '65536'}
    r = {'kit': 'r18', 'app': 't', 'prompt': p,
         'model': {'file': 'olmoe-Mi.gguf', 'bytes': 3072399552, 'expected_bytes': 3072399552,
                   'verified_marker': SHA, 'sha256_end': SHA},
         'libs': {'libllama.so': 'aa' * 32}, 'build': {'libs': {'libllama.so': 'aa' * 32}},
         'template': {'architecture': 'olmoe', 'custom_template': False, 'rendered': expected_render(p),
                      'rendered_tokens': 151},
         'runs': [mk(decode_tps=20.0), mk(decode_tps=22.0, rep=1), mk(decode_tps=24.0, rep=2, gate_timeout=True),
                  mk(group='stack', id='stack', env=stack, env_set=stack, decode_tps=25.0),
                  mk(group='profiled', id='prof', kind='profiled32', env=stack, decode_tps=21.0,
                     chunk_ms=[100.0, 145.0, 190.0, 240.0],
                     prof=['x cpu-prof ab=- tok=2 us/token=44000 majflt=3 minflt=1 mhz=0/0 run=prof.0 prompt=2 blk=0 '
                           'threads=3 ops=1 barriers=1 syncs=1 skipped=0 pairs=0 head=0 router=0 mm=0 mmid=0 attn=0 '
                           'other=0 wait=0 | ADD 1:1+1']),
                  mk(group='skip', id='skip', env={'VOLLIE_SKIP_MASS': '900'}, text='Skipped', decode_tps=26.0,
                     c2=['c2-skip-run: run=skip.0 prompt=5 tokens=183 pairs=1464 pairs_skipped=13.5% '
                         'expert_bytes_skipped=13.6% MB_per_token_selected=1.0 MB_per_token_skipped=0.1 '
                         'batch_bytes_unread=2.0%']),
                  mk(group='norepack', id='nr', repack_loaded=False, env=stack, decode_tps=19.0, text='Other',
                     mhz_after='mhz=1804,1804,1804,1804,1555,1555,1555,1555')]}
    # A/B: 2 generations, warm-up 16, blocks of 8 in A B B A order; B is 1.0 ms faster per token, noise sd 1 ms/token;
    # one A block has a major fault, one B block a low clock: both dropped.
    rnd = random.Random(7)
    for gen in range(2):
        tup, chunks, t_ms = [], [0.0], 0.0
        for tok in range(1, 16 + 8 * 24 + 1):
            blk = 0 if tok <= 16 else (tok - 17) // 8 + 1
            side = 'A' if blk == 0 or blk % 4 in (1, 0) else 'B'
            us = int((40.0 - (1.0 if side == 'B' else 0.0) + rnd.gauss(0, 1.0)) * 1000)
            maj = 5 if (gen, blk) == (0, 1) else 0
            m4 = 1555 if (gen, blk) == (1, 2) else 2112
            tup.append([side, tok, us, maj, 9, m4, 2035, 2, blk, 3, 469, 242 if side == 'B' else 469, 355, 0, 0,
                        1000, 200, 1500, 20000, 100, 1000, 5000])
            t_ms += us / 1000 + 0.5
            chunks.append(t_ms)
        r['runs'].append({'group': 'ab', 'kind': 'abba', 'id': 'ab_tiny', 'rep': gen, 'env': {'VOLLIE_AB': 'knob:TINY_OP:0:65536'},
                          'tok': tup, 'chunk_ms': chunks, 'text': 'story', 'decode_tps': 25.0})
    g = groups(r, 1800)
    assert g[0].startswith('stock: n 3 printed tok/s median 22.00 min 20.00 max 24.00'), g[0]
    assert 'stock#2 (gate timeout)' in g[0] and 'stock#0' not in g[0], g[0]
    assert 'steady ms/token 44.00 (prof)' in g[2] and 'decode majflt 3' in g[2], g[2]
    assert 'expert bytes skipped 13.60 %' in g[3], g[3]
    assert 'clock 1555<1800' in g[4], g[4]
    assert steady_ms({'chunk_ms': [100.0, 145.0, 190.0, 240.0]}) == (45.0, 'chunk')
    idt = identity(r)
    assert 'stock == stack: yes' in idt, idt
    assert any('repack=False' in l and 'identical' in l for l in idt), idt
    c = checks(r, None)
    assert c[0].endswith('OK') and c[1].endswith('= build') and "rendered==OLMoE's own: yes" in c[3], c
    a = ab(r, 1800, 0.4)
    assert len(a) == 1 and 'blocks A 24 (clean 23) B 24 (clean 23)' in a[0], a
    d = float(re.search(r'graph B-A (-?[\d.]+) ± ([\d.]+)', a[0]).group(1))
    s = float(re.search(r'graph B-A (-?[\d.]+) ± ([\d.]+)', a[0]).group(2))
    assert abs(d + 1.0) < 3 * s and s < 0.4 and 'graph B-A' in a[0] and ': resolved' in a[0], a[0]
    assert 'barriers 469->242' in a[0], a[0]
    dw = float(re.search(r'wall B-A (-?[\d.]+)', a[0]).group(1))
    assert abs(dw - d) < 0.01, (dw, d)  # the synthetic wall adds a constant 0.5 ms per token
    assert verdict(-0.5, 0.6, 0.4).startswith('UNRESOLVED') and verdict(-2.0, 0.6, 0.4).startswith('resolved (')
    r['runs'][0]['text'] = 'changed'
    assert any('DIFFER' in l for l in identity(r))
    with tempfile.NamedTemporaryFile('w', suffix='.txt', delete=False) as f:
        f.write('c2gz:' + base64.b64encode(gzip.compress(json.dumps(r).encode())).decode())
    assert load(f.name)['kit'] == 'r18'
    os.unlink(f.name)
    print('selftest OK')
    return 0


if __name__ == '__main__':
    sys.exit(main())
