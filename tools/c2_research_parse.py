#!/usr/bin/env python3
"""Campaign 2 research-kit report parser (kit r17 and later; schema in
campaigns/02-25toks-5pct/reports/memo-10-kit-r17.md). Standard library only.

usage: python3 -I tools/c2_research_parse.py <report.json> [--min-mhz 1800] [--prompt tools/c2_official_prompt.txt]
       python3 -I tools/c2_research_parse.py --selftest

Prints the report's identity checks (model, libraries, chat template, prompt), then one line per group:
n, median / min / max of the printed tok/s (decode_tps), steady ms/token from per-token lines, big-core
clocks after the run, faults, and the runs that broke hygiene. All numbers are the phone's (*device*)."""
import argparse
import json
import os
import re
import statistics as st
import sys

SHA = '5abf710b775cd64fd89d74a783c8a7056a409ba795cd401b9dd448ef8e444abb'
BIG = (4, 5, 6, 7)  # Snapdragon 888: cpu4-6 A78, cpu7 X1
HERE = os.path.dirname(os.path.abspath(__file__))


def mhz(s):
    """'mhz=1804,…' -> list of ints (index = cpu)."""
    m = re.match(r'mhz=([\d,]+)', s or '')
    return [int(x) for x in m.group(1).split(',')] if m else []


def big_min(s):
    v = mhz(s)
    return min(v[i] for i in BIG) if len(v) > max(BIG) else None


def token_lines(run):
    """Per-token profile lines of the runtime ('cpu-prof … tok=<n> us/token=<µs> … majflt=<n>'), as dicts."""
    out = []
    for line in run.get('prof') or []:
        if 'tok=' not in line:
            continue
        f = dict(re.findall(r'(\w[\w/]*)=(-?[\d.]+)', line.split('|')[0]))
        if 'tok' in f:
            out.append(f)
    return out


def steady_ms(run):
    """Median ms/token over tokens 2.. : from the runtime's per-token lines if present, else from the chunk
    arrival times (profiled32/abba: one chunk per token; Dart-side, includes the time outside the graph)."""
    tl = [t for t in token_lines(run) if int(float(t['tok'])) >= 2 and 'us/token' in t]
    if tl:
        return st.median(float(t['us/token']) / 1000 for t in tl), 'prof'
    c = run.get('chunk_ms') or []
    d = [b - a for a, b in zip(c, c[1:])]
    return (st.median(d), 'chunk') if d else (None, None)


def decode_majflt(run):
    """Major faults in decode tokens: from per-token lines (None = not available in this runtime)."""
    tl = token_lines(run)
    return sum(int(float(t.get('majflt', 0))) for t in tl) if tl else None


def fmt(x, nd=2):
    return '-' if x is None else f'{x:.{nd}f}'


def expected_render(prompt):
    # OLMoE-0125-Instruct's GGUF template: bos_token + '<|user|>\n' + content + '\n' + '<|assistant|>\n'
    return '<|endoftext|><|user|>\n' + prompt + '\n<|assistant|>\n'


def checks(r, prompt_file):
    out = []
    m = r.get('model') or {}
    ok = m.get('bytes') == m.get('expected_bytes') and m.get('verified_marker') == SHA and m.get('sha256_end') == SHA
    out.append(f"model {m.get('file')} {m.get('bytes')} B, marker {str(m.get('verified_marker'))[:8]}, "
               f"sha256 at end {str(m.get('sha256_end'))[:8]} -> {'OK' if ok else 'CHECK'}")
    for k, v in sorted((r.get('libs') or {}).items()):
        out.append(f'lib {k} {v}')
    t = r.get('template') or {}
    p = r.get('prompt', '')
    same = t.get('rendered') == expected_render(p)
    out.append(f"template arch={t.get('architecture')} custom={t.get('custom_template')} rendered_tokens="
               f"{t.get('rendered_tokens')} rendered==OLMoE's own: {'yes' if same else 'NO'}")
    if prompt_file and os.path.exists(prompt_file):
        with open(prompt_file, encoding='utf-8') as f:
            out.append('prompt == ' + os.path.basename(prompt_file) + ': ' + ('yes' if f.read() == p else 'NO'))
    pt = sorted({x.get('prompt_tokens') for x in r.get('runs', []) if x.get('prompt_tokens') is not None})
    out.append(f'prompt_tokens seen in runs: {pt}')
    return out


def groups(r, min_mhz):
    runs = r.get('runs', [])
    order, by = [], {}
    for x in runs:
        g = x.get('group', '?')
        if g not in by:
            order.append(g)
            by[g] = []
        by[g].append(x)
    ref = next((x.get('text') for x in runs if x.get('kind') != 'abba' and not x.get('env') and x.get('text')), None)
    out = []
    for g in order:
        rs = by[g]
        tps = [x['decode_tps'] for x in rs if x.get('decode_tps') is not None]
        sm = [s for s, _ in map(steady_ms, rs) if s is not None]
        src = {k for _, k in map(steady_ms, rs) if k}
        clk = [c for c in (big_min(x.get('mhz_after')) for x in rs) if c is not None]
        flags = []
        for x in rs:
            tag = f"{x.get('id')}#{x.get('rep')}"
            why = []
            if x.get('error'):
                why.append('error')
            c = big_min(x.get('mhz_after'))
            if c is not None and c < min_mhz:
                why.append(f'clock {c}<{min_mhz}')
            dm = decode_majflt(x)
            if dm:
                why.append(f'decode majflt {dm}')
            if x.get('gate_timeout'):
                why.append('gate timeout')
            if x.get('cached_prompt_tokens'):
                why.append(f"cached {x['cached_prompt_tokens']}")
            if x.get('kind') != 'abba' and not x.get('env') and ref is not None and x.get('text') != ref:
                why.append('text differs from the first stock run')
            if why:
                flags.append(tag + ' (' + ', '.join(why) + ')')
        med = st.median(tps) if tps else None
        out.append(
            f"{g}: n {len(rs)} printed tok/s median {fmt(med)} min {fmt(min(tps) if tps else None)} "
            f"max {fmt(max(tps) if tps else None)} | steady ms/token {fmt(st.median(sm) if sm else None)}"
            f"{' (' + '+'.join(sorted(src)) + ')' if src else ''} | big-core MHz after: min {min(clk) if clk else '-'} "
            f"median {st.median(clk) if clk else '-'} | majflt median {st.median([x.get('majflt', 0) for x in rs])} "
            f"minflt median {st.median([x.get('minflt', 0) for x in rs])} | flags: {'; '.join(flags) or 'none'}")
    return out


def main(argv=None):
    a = argparse.ArgumentParser()
    a.add_argument('report', nargs='?')
    a.add_argument('--min-mhz', type=int, default=1800)
    a.add_argument('--prompt', default=os.path.join(HERE, 'c2_official_prompt.txt'))
    a.add_argument('--selftest', action='store_true')
    o = a.parse_args(argv)
    if o.selftest:
        return selftest()
    if not o.report:
        a.error('report.json required')
    with open(o.report, encoding='utf-8') as f:
        r = json.load(f)
    print(f"kit {r.get('kit')} app {r.get('app')} started {r.get('started')} stopped_early {r.get('stopped_early')}")
    print('\n'.join(checks(r, o.prompt)))
    print('\n'.join(groups(r, o.min_mhz)))
    return 0


def selftest():
    p = 'Read these notes.'
    mk = lambda **k: {'group': 'stock', 'kind': 'official32', 'env': {}, 'rep': 0, 'id': 'stock',
                      'mhz_after': 'mhz=1804,1804,1804,1804,2112,2112,2112,2035', 'majflt': 0, 'minflt': 10,
                      'text': 'The ocean', 'cached_prompt_tokens': 0, 'prompt_tokens': 151, **k}
    r = {'kit': 'r17', 'app': 't', 'prompt': p,
         'model': {'file': 'olmoe-Mi.gguf', 'bytes': 3072399552, 'expected_bytes': 3072399552,
                   'verified_marker': SHA, 'sha256_end': SHA},
         'template': {'architecture': 'olmoe', 'custom_template': False, 'rendered': expected_render(p),
                      'rendered_tokens': 151},
         'runs': [mk(decode_tps=20.0), mk(decode_tps=22.0, rep=1), mk(decode_tps=24.0, rep=2, gate_timeout=True),
                  mk(group='profiled', id='prof', kind='profiled32', decode_tps=21.0, chunk_ms=[100.0, 145.0, 190.0, 240.0],
                     prof=['cpu-prof ab=A tok=2 us/token=44000 majflt=3 | MUL_MAT_ID 16:1+1']),
                  mk(group='norepack', id='nr', decode_tps=19.0, text='Other',
                     mhz_after='mhz=1804,1804,1804,1804,1555,1555,1555,1555')]}
    g = groups(r, 1800)
    assert g[0].startswith('stock: n 3 printed tok/s median 22.00 min 20.00 max 24.00'), g[0]
    assert 'stock#2 (gate timeout)' in g[0] and 'stock#0' not in g[0], g[0]
    assert 'steady ms/token 44.00 (prof)' in g[1] and 'decode majflt 3' in g[1], g[1]
    assert steady_ms({'chunk_ms': [100.0, 145.0, 190.0, 240.0]}) == (45.0, 'chunk')
    assert 'clock 1555<1800' in g[2] and 'text differs' in g[2], g[2]
    c = checks(r, None)
    assert c[0].endswith('OK') and "rendered==OLMoE's own: yes" in c[1], c
    r['model']['sha256_end'] = '0' * 64
    assert checks(r, None)[0].endswith('CHECK')
    print('selftest OK')
    return 0


if __name__ == '__main__':
    sys.exit(main())
