# Campaign 3: tabulates a c3 probe report (kit r29, "Save c3 probe to Downloads"; patches/llama.cpp-c3/0001-probe; and
# 0002-d-rows: the per-block "hdr" lines (D3) and the "p2d" rows (D2); kit r32, 0003-r72-rows: the R72 rows a, l1, l4,
# l2/l3, l5, l6, l7, anon and x2r / x2info, under "R72 rows"; per-thread labels M / U / P / C on l1-l4, see thr_labels).
# usage: python3 -I tools/c3_probe_parse.py <vollie-c3probe-r29-*.json>      python3 -I tools/c3_probe_parse.py --selftest
# Every number it prints is *device* (from the phone's lines); the ratios are arithmetic on them. A cell whose workers did
# not all stay on their cpu (cpu_ok < n) is marked '*' and should not be used for a claim.
import json, os, re, statistics, sys


def kv(line):
    """'c3-probe: p2 kern=read cores=4-6 gbs=1.0 ...' -> ('p2', {'kern': 'read', ...})"""
    body = line.split('c3-probe:', 1)[1].split()
    return body[0], dict(t.split('=', 1) for t in body[1:] if '=' in t)


def num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def parse(rep):
    out = {'p2': [], 'p2hot': [], 'p2col': [], 'p2col_id': None, 'x1': [], 'x2': [], 'x1gpu': [], 'p1': [], 'p1_info': {},
           'p2d': [], 'p2d_id': None, 'hdr': [], 'r32': {t: [] for t in R32},
           'gates': [], 'errors': [], 'meta': {k: rep.get(k) for k in ('kit', 'app', 'started', 'label', 'session_label',
                                                                          'deep_sleeping_list', 'stopped_early', 'model_unloaded')}}
    for b in rep.get('blocks', []):
        g = b.get('gate', {})
        out['gates'].append((b.get('block'), g.get('timeout'), (g.get('series') or [[None, None]])[-1], b.get('ms'), b.get('rc')))
        if 'error' in b:
            out['errors'].append('%s: %s' % (b.get('block'), b['error']))
        for line in b.get('lines', []):
            if 'c3-probe:' not in line:
                continue
            tag, f = kv(line)
            if line.split('c3-probe:', 1)[1].split()[1:2] == ['end']:
                out['meta']['%s_s' % tag] = f.get('s')
            elif 'error' in f:
                out['errors'].append('%s %s' % (tag, ' '.join('%s=%s' % kv_ for kv_ in f.items())))
            elif tag == 'hdr':
                out['hdr'].append((b.get('block'), line.split('c3-probe:', 1)[1].split()[1], f))
            elif tag == 'p2d' and 'k1_keep100' in f:
                out['p2d_id'] = f
            elif tag == 'p2col' and 'col_exact' in f:
                out['p2col_id'] = f
            elif tag == 'p1' and 'pass' in f:
                out['p1'].append(f)
            elif tag == 'p1':
                out['p1_info'].update(f)
            elif tag in R32:
                out['r32'][tag].append((line.split('c3-probe:', 1)[1].split()[1], f))
            elif tag in out and isinstance(out[tag], list):
                out[tag].append(f)
    return out


R32 = ('a', 'l1', 'l4', 'l4sum', 'l2', 'l3', 'l5', 'l6', 'l7', 'anon', 'x2r', 'x2info')

# Four-way per-thread label (Ebony, R72 §11; replaces CPU / wall as the only check). One letter per condition met, in
# this order; the first is the thread's label, none = C:
#   M misplaced     on its pinned cpu in < 0.9 of the sched_getcpu checks (an affinity read-back is not in r33's lines)
#   U under-clocked the pinned cpu's policy sampled below 0.9 x its hardware maximum (lowest sample of mhz_range against
#                   cpuinfo_max_freq from max_mhz, else HW_MHZ); a lowered scaling_max_freq counts as under-clocked
#   P pre-empted    involuntary switches > NIV_HI in the cell, or CPU / wall < 0.9
#   C compute       none of the above
# The thresholds are this parser's defaults (calibration knobs, not pre-registered): change them here and say so.
HW_MHZ = {0: 1804, 4: 2419, 7: 2841}   # SD888 policies 0 / 4 / 7 (A55 / A78 / X1)
ON_MIN, CLK_MIN, CW_MIN, NIV_HI = 0.9, 0.9, 0.9, 50


def policy(cpu):
    return 7 if cpu >= 7 else 4 if cpu >= 4 else 0


def thr_labels(f):
    """[(cpu, flags, cpu_over_wall, nivcsw, on_share or None)] for a line with thr=cpu:cw:niv:nv[:off/chk@last],..."""
    hw = dict(HW_MHZ)
    for t in (f.get('max_mhz') or '').split(','):
        if t.startswith('p') and ':' in t:
            p, v = t[1:].split(':', 1)
            hw[int(p)] = int(v.split('/')[-1])
    lo = {}
    rng = f.get('mhz_range') or ''
    if '..' in rng:
        lo = dict(zip((0, 4, 7), (int(v) for v in rng.split('..')[0].split('/'))))
    out = []
    for t in f.get('thr', '').split(','):
        x = t.split(':')
        if len(x) < 4:
            continue
        cpu, cw, niv = int(x[0]), float(x[1]), int(x[2])
        on = None
        if len(x) >= 5 and '/' in x[4]:
            off, chk = (int(v) for v in x[4].split('@')[0].split('/'))
            on = (chk - off) / chk if chk > 0 else None
        flags = ''
        if on is not None and on < ON_MIN:
            flags += 'M'
        p = policy(cpu)
        if p in lo and hw.get(p, 0) > 0 and lo[p] < CLK_MIN * hw[p]:
            flags += 'U'
        if niv > NIV_HI or 0 <= cw < CW_MIN:
            flags += 'P'
        out.append((cpu, flags, cw, niv, on))
    return out


def labelstr(f):
    """'4:C,5:M,7:MU' (cpu:flags, C = compute)"""
    return ','.join('%d:%s' % (c, fl or 'C') for c, fl, *_ in thr_labels(f))


def labelled(rows):
    return [dict(f, labels=labelstr(f)) if 'thr' in f else f for f in rows]


def labelreport(r):
    """thread counts by tag, pinned cpu and label; the l3 per-thread rates against a clean single-thread reference"""
    cnt = {}
    for tag in ('l1', 'l2', 'l3', 'l4'):
        for k, f in r[tag]:
            for c, fl, *_ in thr_labels(f):
                key = (tag, c)
                cnt.setdefault(key, {'M': 0, 'U': 0, 'P': 0, 'C': 0, 'n': 0})
                cnt[key][(fl or 'C')[0]] += 1
                cnt[key]['n'] += 1
    if not cnt:
        return []
    L = ['\nPer-thread labels (first condition met: M misplaced, U under-clocked, P pre-empted, C compute; thresholds in the parser)',
         '| tag | pinned cpu | threads | M | U | P | C |', '|---|---|---|---|---|---|---|']
    for (tag, c), d in sorted(cnt.items()):
        L.append('| %s | %d | %d | %d | %d | %d | %d |' % (tag, c, d['n'], d['M'], d['U'], d['P'], d['C']))
    ref = {}
    for k, f in r['l3']:
        lab = thr_labels(f)
        if len(lab) == 1 and not lab[0][1]:
            ref.setdefault(policy(lab[0][0]), num(f['per_thread_gw_per_s'].split(':')[1]))
    rows = []
    for k, f in r['l3']:
        lab = {c: fl or 'C' for c, fl, *_ in thr_labels(f)}
        for t in f.get('per_thread_gw_per_s', '').split(','):
            c, v = int(t.split(':')[0]), num(t.split(':')[1])
            rf = ref.get(policy(c))
            rows.append('| %s | %d | %s | %.3f | %s | %s |' % (f['set'], c, lab.get(c, '-'), v, '%.3f' % rf if rf else '-', '%.3f' % (v / rf) if rf else '-'))
    if rows:
        L += ['\nL3 per thread against its cluster reference (the first single-thread l3 cell labelled C on that cluster; - = none)',
              '| set | cpu | label | G weights/s | reference | ratio |', '|---|---|---|---|---|---|'] + rows
    return L


def table(L, rows, cols):
    """markdown table of the given fields; '*' after a cell whose workers left their cpu"""
    L.append('| ' + ' | '.join(cols) + ' |'); L.append('|' + '---|' * len(cols))
    for f in rows:
        L.append('| ' + ' | '.join(str(f.get(c, '')) for c in cols) + ' |' + (' *' if 'cpu_ok' in f and okmark(f) else ''))


def r32report(r):
    """kit r32 (R72): every row as printed by the phone, plus the ratios the plan reads (k1m4 / k1 = L1x; type / Q4_0)"""
    if not any(r.values()):
        return []
    L = ['\nR72 rows (kit r32; numbers *device*, ratios arithmetic on them)']
    for f in [f for k, f in r['x2info']]:
        L.append('  x2info: ' + ', '.join('%s=%s' % kv_ for kv_ in f.items()))
    for k, f in r['x2r']:
        L.append('  x2r (-cl-opt-disable) kern %s: %s GB/s' % (f.get('kern'), f.get('gbs')))
    if r['a']:
        L.append('\nanchors (R69 code)'); table(L, [f for k, f in r['a']], ['kern', 'cores', 'gbs', 'ns_per_w', 'cpu_ok', 'mhz'])
    ids = [f for k, f in r['l1'] if k == 'identity']
    for f in ids:
        L.append('  l1 identity k_in=%s keep=%s: k1=%s k1m=%s k1m4=%s' % (f['k_in'], f['keep'], f['k1'], f['k1m'], f['k1m4']))
    cells = [f for k, f in r['l1'] if k != 'identity' and 'kern' in f]
    if cells:
        L.append('\nL1: k = ns per touched weight / stock (mean of before and after), same shape, regime and cpu')
        table(L, labelled(cells), ['k_in', 'regime', 'cpu', 'kern', 'keep', 'd', 'k', 't_vs_stock', 'gbs', 'drift', 'cpu_ok', 'labels', 'mhz_range'])
        by = {}
        for f in cells:
            by[(f['k_in'], f['regime'], f['cpu'], f['keep'], f['kern'])] = num(f['k'])
        L.append('\nL1x = k1m4 / k1 and k1m / k1, same cell')
        L.append('| k_in | regime | cpu | keep | k1m / k1 | k1m4 / k1 |'); L.append('|---|---|---|---|---|---|')
        for (ki, rg, cp, kp, kn), v in by.items():
            if kn == 'k1' and v:
                a, b = by.get((ki, rg, cp, kp, 'k1m')), by.get((ki, rg, cp, kp, 'k1m4'))
                L.append('| %s | %s | %s | %s | %s | %s |' % (ki, rg, cp, kp, '%.3f' % (a / v) if a else '-', '%.3f' % (b / v) if b else '-'))
    for f in [f for k, f in r['l4'] if k == 'identity']:
        L.append('  l4 identity %s keep %s: k1=%s k1m=%s (layer %s MB, rule reads %s MB)' % (f['subj'], f['keep'], f['k1'], f['k1m'], f['layer_mb'], f['rule_layer_mb']))
    if r['l4sum']:
        L.append('\nL4 (ABBA means; body_ms = ms per layer x layers)')
        table(L, labelled([f for k, f in r['l4'] if 'arm' in f and f.get('arm') != 'ggml_stock']), ['subj', 'arm', 'pos', 'set', 'ms_per_layer', 'cpu_ok', 'labels', 'mhz_range', 'max_mhz'])
        table(L, [f for k, f in r['l4sum']], ['subj', 'set', 't_stock', 't_k1', 't_k1m', 'k1_vs_stock', 'k1m_vs_stock', 'body_ms_stock', 'body_ms_k1', 'body_ms_k1m'])
        table(L, [f for k, f in r['l4'] if f.get('arm') == 'ggml_stock'], ['subj', 'arm', 'buft', 'set', 'ms_per_layer', 'body_ms', 'gbs', 'proc_cpu_over_wall', 'proc_nivcsw'])
    if r['l2'] or r['l3']:
        L.append('\nL2 read rate and L3 compute ceiling (thr = cpu:cpu_time/wall:involuntary:voluntary switches)')
        table(L, labelled([f for k, f in r['l2']]), ['kern', 'set', 'gbs', 'ns_per_w', 'cpu_ok', 'labels', 'thr', 'proc_cpu_over_wall', 'mhz_range'])
        table(L, labelled([f for k, f in r['l3']]), ['set', 'gw_per_s', 'per_thread_gw_per_s', 'cpu_ok', 'labels', 'thr', 'mhz_range'])
    if r['l5']:
        L.append('\nL5 per-type gemv (ratio = ns per weight / Q4_0 pass 1, same k_in, regime and set)')
        q = {(f['k_in'], f['regime'], f['set']): num(f['ns_per_w']) for k, f in r['l5'] if f.get('type') == 'q4_0' and f.get('pass') == '1'}
        rows = []
        for k, f in r['l5']:
            if 'type' not in f:
                continue
            b = q.get((f['k_in'], f['regime'], f['set'])); v = num(f['ns_per_w'])
            rows.append(dict(f, ratio='%.3f' % (v / b) if b and v else '-'))
        table(L, rows, ['type', 'pass', 'k_in', 'regime', 'set', 'rows', 'buft', 'ns_per_w', 'ratio', 'gbs', 'proc_cpu_over_wall', 'mhz_range'])
    if r['l6']:
        L.append('\nL6 barriers per token'); table(L, [f for k, f in r['l6'] if 'kind' in f], ['kind', 'nodes', 'barriers', 'set', 'ms_per_token', 'us_per_node', 'us_per_barrier'])
    if r['l7']:
        L.append('\nL7 head gemv, cold'); table(L, [f for k, f in r['l7'] if 'type' in f], ['type', 'vocab', 'set', 'buft', 'ms_per_call', 'gbs', 'proc_cpu_over_wall'])
    passes = [f for k, f in r['anon'] if 'pass' in f]
    if r['anon']:
        L.append('\nanonymous-memory arms')
        for f in [f for k, f in r['anon'] if 'pass' not in f]:
            L.append('  ' + ', '.join('%s=%s' % kv_ for kv_ in f.items()))
        steps = {}
        for f in passes:
            steps.setdefault(f['n'], []).append(f)
        L.append('| GB | passes | median GB/s | last 5 GB/s (median) | majflt total | majflt last 5 | vmrss kB (last) | vmswap kB (last) | swapfree kB (last) |')
        L.append('|---|---|---|---|---|---|---|---|---|')
        for n, fs in steps.items():
            g = [num(f['gbs']) for f in fs]; mf = [int(f['majflt']) for f in fs]
            L.append('| %s | %d | %.2f | %.2f | %d | %d | %s | %s | %s |' % (n, len(fs), statistics.median(g), statistics.median(g[-5:]), sum(mf), sum(mf[-5:]),
                     fs[-1]['vmrss'], fs[-1]['vmswap'], fs[-1]['swapfree']))
    L += labelreport(r)
    return L


def okmark(f):
    ok, n = (f.get('cpu_ok') or '0/0').split('/')
    return '' if ok == '-1' or ok == n else '*'


def report(p):
    L = []
    L.append('meta: ' + ', '.join('%s=%s' % kv_ for kv_ in p['meta'].items()))
    for g in p['gates']:
        L.append('gate %s: timeout=%s last=%s block_ms=%s rc=%s' % g)
    for e in p['errors']:
        L.append('ERROR ' + e)
    L.append('\nP2 dense read rate (GB/s, device)')
    L.append('| cores | read | q4x4 gemv | q4x4 ns/w | mhz (0/4/7) before->after |')
    L.append('|---|---|---|---|---|')
    rd = {f['cores']: f for f in p['p2'] if f.get('kern') == 'read'}
    q4 = {f['cores']: f for f in p['p2'] if f.get('kern') == 'q4x4'}
    for c in list(dict.fromkeys(list(rd) + list(q4))):
        a, b = rd.get(c, {}), q4.get(c, {})
        L.append('| %s | %s%s | %s%s | %s | %s |' % (c, a.get('gbs', '-'), okmark(a) if a else '', b.get('gbs', '-'), okmark(b) if b else '',
                                                   b.get('ns_per_w', '-'), b.get('mhz', a.get('mhz', '-'))))
    L.append('\nP2 hot sdot ceiling (one pinned thread, L2-resident)')
    for f in p['p2hot']:
        L.append('  %s (cpu %s, %s KB): %s ns/weight%s' % (f['core'], f['cpu'], f['ws_kb'], f['ns_per_w'], okmark(f)))
    if p['p2col_id']:
        i = p['p2col_id']
        L.append('\nP2 column and skip-list kernels at keep %s: identity col_exact=%s skip_exact=%s (1 = bit-identical to the stock kernel); '
                 'kept columns %s, kept blocks %s' % (i['keep'], i['col_exact'], i['skip_exact'], i['kept_cols'], i['kept_blocks']))
    cols = {}
    for f in p['p2col']:
        cols.setdefault(f['cores'], {})[f['kern']] = f
    for c, d in cols.items():
        st = num(d.get('stock', {}).get('ns_per_touched_w'))
        for k in ('col', 'skip'):
            v = num(d.get(k, {}).get('ns_per_touched_w'))
            if st and v:
                L.append('  cores %s: %s / stock per touched weight = %.2f x (GB/s %s vs %s)%s' % (c, k, v / st, d[k]['gbs'], d['stock']['gbs'], okmark(d[k])))
    if p['p2d_id']:
        i = p['p2d_id']
        L.append('\nD2 rows on one pinned A78 (cpu 4): identity k1 keep 1.0=%s, k1 keep 0.6=%s, y1b=%s, y1b_rows=%s (1 = bit-identical to stock)'
                 % (i['k1_keep100'], i['k1_keep060'], i['y1b'], i['y1b_rows']))
    if p['p2d']:
        L.append('k = ns per touched weight / stock ns per weight (mean of stock before and after, same shape and regime); '
                 't = k x d = time against dense stock')
        L.append('| inputs | rows | regime | kern | keep | d | k | t vs stock | GB/s | drift | mhz (0/4/7) |')
        L.append('|---|---|---|---|---|---|---|---|---|---|---|')
        for f in p['p2d']:
            L.append('| %s | %s | %s | %s | %s | %s | %s | %s | %s%s | %s | %s |' % (f['k_in'], f['rows'], f['regime'], f['kern'], f['keep'], f['d'], f['k'],
                     f['t_vs_stock'], f['gbs'], okmark(f), f.get('drift', ''), f.get('mhz', '')))
    L.append('\nX1 multi-stream, 3 A78')
    for f in p['x1']:
        L.append('  streams %s: %s GB/s%s' % (f['streams'], f['gbs'], okmark(f)))
    L.append('\nX2 GPU and CPU + GPU')
    for f in p['x2']:
        L.append('  GPU alone, kern %s (%s): %s GB/s' % (f['kern'], f.get('device'), f['gbs']))
    for f in p['x1gpu']:
        L.append('  together, gpu_kern %s + cpu_kern %s on %s: cpu %s + gpu %s = %s GB/s%s' % (f['gpu_kern'], f['cpu_kern'], f['cores'], f['cpu_gbs'], f['gpu_gbs'], f['sum_gbs'], okmark(f)))
    L.append('\nP1 resident capacity (%s)' % ', '.join('%s=%s' % kv_ for kv_ in p['p1_info'].items()))
    L.append('| GB | map | passes | median s | median GB/s | last 5 GB/s (median) | majflt total | majflt last 5 | cached kB (last) | swapfree kB (last) | vmswap kB (last) |')
    L.append('|---|---|---|---|---|---|---|---|---|---|---|')
    steps = {}
    for f in p['p1']:
        steps.setdefault((f['n'], f['map']), []).append(f)
    for (n, m), fs in steps.items():
        s = [num(f['s']) for f in fs]; g = [num(f['gbs']) for f in fs]; mf = [int(f['majflt']) for f in fs]
        L.append('| %s | %s | %d | %.3f | %.2f | %.2f | %d | %d | %s | %s | %s |' % (n, m, len(fs), statistics.median(s), statistics.median(g),
                 statistics.median(g[-5:]), sum(mf), sum(mf[-5:]), fs[-1]['cached'], fs[-1]['swapfree'], fs[-1]['vmswap']))
    L += r32report(p['r32'])
    L += header(p['hdr'])
    return '\n'.join(L)


MM_STAT = ('orig_data_size', 'compr_data_size', 'mem_used_total', 'mem_limit', 'mem_used_max', 'same_pages', 'pages_compacted',
           'huge_pages', 'huge_pages_since')


def header(hdr):
    """D3: lmk properties, zram, memory and vm once (first block), PSI per block. 'e<n>' = the source gave errno n."""
    if not hdr:
        return []
    L = ['\nD3 header (first block %s)' % hdr[0][0]]
    first = [h for h in hdr if h[0] == hdr[0][0]]
    for _, kind, f in first:
        if kind == 'zram':
            mm = f.get('mm_stat', '')
            if mm and not mm.startswith('e') and mm != 'empty':
                f = dict(f, **dict(zip(MM_STAT, mm.split(','))))
            L.append('  zram: ' + ', '.join('%s=%s' % kv_ for kv_ in f.items()))
        elif kind != 'psi':
            L.append('  %s: %s' % (kind, ', '.join('%s=%s' % kv_ for kv_ in f.items())))
    L.append('| block | psi | kind | avg10 | avg60 | avg300 | total (us) |')
    L.append('|---|---|---|---|---|---|---|')
    for blk, kind, f in hdr:
        if kind == 'psi':
            L.append('| %s | %s | %s | %s | %s | %s | %s |' % (blk, f.get('res'), f.get('kind', 'e%s' % f.get('errno')), f.get('avg10', '-'),
                     f.get('avg60', '-'), f.get('avg300', '-'), f.get('total', '-')))
    return L


SELFTEST = {'kit': 'r29', 'label': 'power-save off', 'blocks': [
    {'block': 'p2', 'rc': 100, 'ms': 70000, 'gate': {'timeout': False, 'series': [[0, 0.5]]}, 'lines': [
        'c3-probe: p2 kern=read cores=4-6 gbs=20.50 cpu_ok=3/3 mhz=1804/2419/2841->1804/2419/2841',
        'c3-probe: p2 kern=q4x4 cores=4-6 gbs=18.00 ns_per_w=0.1000 cpu_ok=2/3 mhz=1/2/3->1/2/3',
        'c3-probe: p2hot core=a55 cpu=0 ws_kb=63 ns_per_w=0.9000 cpu_ok=1/1 mhz=1/2/3->1/2/3',
        'c3-probe: p2col identity keep=0.60 col_exact=1 skip_exact=1 kept_cols=0.608 kept_blocks=0.547',
        'c3-probe: p2col kern=stock keep=0.60 cores=4 gbs=10.00 ns_per_touched_w=0.0500 cpu_ok=1/1 mhz=1/2/3->1/2/3',
        'c3-probe: p2col kern=col keep=0.60 cores=4 gbs=4.00 ns_per_touched_w=0.1300 cpu_ok=1/1 mhz=1/2/3->1/2/3',
        'c3-probe: p2col kern=skip keep=0.60 cores=4 gbs=8.00 ns_per_touched_w=0.0550 cpu_ok=1/1 mhz=1/2/3->1/2/3',
        'c3-probe: p2 end s=70.1']},
    {'block': 'p2d', 'rc': 100, 'ms': 80000, 'gate': {'timeout': False, 'series': []}, 'lines': [
        'c3-probe: hdr props n=2 ro.lmk.low=1001 ro.lmk.use_minfree_levels=true',
        'c3-probe: hdr zram disksize=4294967296 mm_stat=100,40,50,0,60,1,0,0 mem_used_total=e2 swaps=/dev/block/zram0,partition,4194300,1000,-2',
        'c3-probe: hdr psi res=memory kind=some avg10=1.50 avg60=0.80 avg300=0.20 total=12345',
        'c3-probe: hdr psi res=cpu errno=13',
        'c3-probe: hdr mem memtotal_kb=7700000 memavail_kb=3000000 cached_kb=1000000 swaptotal_kb=4194300 swapfree_kb=4000000',
        'c3-probe: p2d identity k1_keep100=1 k1_keep060=1 y1b=1 y1b_rows=0',
        'c3-probe: p2d kern=stock_before k_in=4096 rows=1024 row_b=2304 regime=cold keep=1.00 d=1.000 cpu=4 ns_per_touched_w=0.1000 '
        'stock_ns_per_w=0.1000 k=1.000 t_vs_stock=1.000 gbs=5.62 cpu_ok=1/1 mhz=1/2/3->1/2/3',
        'c3-probe: p2d kern=k1 k_in=4096 rows=1024 row_b=2304 regime=cold keep=0.60 d=0.601 cpu=4 ns_per_touched_w=0.1800 '
        'stock_ns_per_w=0.1000 k=1.800 t_vs_stock=1.082 gbs=3.00 cpu_ok=0/1 mhz=1/2/3->1/2/3',
        'c3-probe: p2d kern=stock_after k_in=4096 rows=1024 row_b=2304 regime=cold keep=1.00 d=1.000 cpu=4 ns_per_touched_w=0.1000 '
        'stock_ns_per_w=0.1000 k=1.000 t_vs_stock=1.000 drift=1.000000 gbs=5.62 cpu_ok=1/1 mhz=1/2/3->1/2/3',
        'c3-probe: p2d end s=80.0']},
    {'block': 'x2', 'rc': -1, 'ms': 5, 'gate': {'timeout': True, 'series': [[90000, 0.8]]}, 'lines': ['c3-probe: x2 error=no_opencl']},
    {'block': 'p1', 'rc': 100, 'ms': 9000, 'gate': {'timeout': False, 'series': []}, 'lines': [
        'c3-probe: p1 free_gb=20.00 top_gb=4.6',
        'c3-probe: p1 n=2.7 map=r pass=0 s=3.000 gbs=0.90 majflt=600000 cached=100 swapfree=5 vmswap=0',
        'c3-probe: p1 n=2.7 map=r pass=1 s=0.150 gbs=18.00 majflt=0 cached=200 swapfree=5 vmswap=0']},
    {'block': 'l1', 'rc': 100, 'ms': 9000, 'gate': {'timeout': False, 'series': []}, 'lines': [
        'c3-probe: l1 identity k_in=4096 keep=0.60 rows=256 k1=1 k1m=1 k1m4=1',
        'c3-probe: l1 kern=k1 k_in=4096 rows=128 regime=hot ws_kb=288 keep=0.600 d=0.600 cpu=4 ns_per_touched_w=0.2 stock_ns_per_w=0.1 k=2.000 t_vs_stock=1.2 gbs=1 cpu_ok=1/1 thr=4:1.000:0:0',
        'c3-probe: l1 kern=k1m4 k_in=4096 rows=128 regime=hot ws_kb=288 keep=0.600 d=0.600 cpu=4 ns_per_touched_w=0.16 stock_ns_per_w=0.1 k=1.600 t_vs_stock=0.96 gbs=1 cpu_ok=1/1 thr=4:1.000:0:0']},
    {'block': 'l5', 'rc': 100, 'ms': 9000, 'gate': {'timeout': False, 'series': []}, 'lines': [
        'c3-probe: l5 type=q4_0 pass=1 k_in=4096 rows=112 copies=1 row_b=2304 regime=hot buft=CPU_REPACK ns_per_w=0.1000 gbs=5 set=4 threads=1',
        'c3-probe: l5 type=q6_K pass=1 k_in=4096 rows=64 copies=1 row_b=3360 regime=hot buft=CPU ns_per_w=0.1500 gbs=5 set=4 threads=1']},
    {'block': 'anon34', 'rc': 100, 'ms': 9000, 'gate': {'timeout': False, 'series': []}, 'lines': [
        'c3-probe: anon n=3.4 pass=0 s=0.2 gbs=17.00 majflt=0 minflt=0 cached=1 swapfree=2 vmswap=0 vmrss=3400000',
        'c3-probe: anon n=3.4 pass=1 s=0.2 gbs=19.00 majflt=5 minflt=0 cached=1 swapfree=2 vmswap=0 vmrss=3400000']}]}


def selftest():
    p = parse(SELFTEST)
    assert p['p2'][0]['gbs'] == '20.50' and okmark(p['p2'][0]) == '' and okmark(p['p2'][1]) == '*'
    assert p['p2col_id']['col_exact'] == '1' and len(p['p2col']) == 3
    assert p['errors'] == ['x2 error=no_opencl'] and p['meta']['p2_s'] == '70.1' and p['gates'][2][1] is True
    r = report(p)
    assert 'col / stock per touched weight = 2.60 x' in r and 'skip / stock per touched weight = 1.10 x' in r, r
    assert '| 2.7 | r | 2 | 1.575 | 9.45 |' in r and '| 600000 | 600000 |' in r, r
    assert 'a55 (cpu 0, 63 KB): 0.9000 ns/weight' in r
    assert p['p2d_id']['y1b_rows'] == '0' and len(p['p2d']) == 3 and len(p['hdr']) == 5 and p['meta']['p2d_s'] == '80.0'
    assert '| 4096 | 1024 | cold | k1 | 0.60 | 0.601 | 1.800 | 1.082 | 3.00* |  |' in r, r
    assert '| 4096 | 1024 | cold | stock_after | 1.00 | 1.000 | 1.000 | 1.000 | 5.62 | 1.000000 |' in r, r
    assert 'mem_used_total=50' in r and 'mem_used_max=60' in r and 'ro.lmk.low=1001' in r, r
    assert '| p2d | memory | some | 1.50 | 0.80 | 0.20 | 12345 |' in r and '| p2d | cpu | e13 |' in r, r
    assert '| 4096 | hot | 4 | 0.600 | - | 0.800 |' in r and '| q6_K | 1 | 4096 | hot | 4 | 64 | CPU | 0.1500 | 1.500 |' in r, r
    assert '| 3.4 | 2 | 18.00 | 18.00 | 5 | 5 | 3400000 | 0 | 2 |' in r and 'l1 identity k_in=4096 keep=0.60: k1=1 k1m=1 k1m4=1' in r, r
    # four-way label: 4 clean; 7 off-core, X1 at its floor, 60 involuntary switches; 5 CPU / wall 0.85; old 4-field thr
    t = {'thr': '4:0.995:2:0:0/100@-1,7:0.990:60:0:100/100@2,5:0.850:0:0:0/10@-1', 'max_mhz': 'p4:2419/2419,p7:2841/2841',
         'mhz_range': '1804/2419/844..1804/2419/844,n=3'}
    assert labelstr(t) == '4:C,7:MUP,5:P', labelstr(t)
    assert labelstr({'thr': '4:1.000:0:0', 'max_mhz': 'p4:1555/2419', 'mhz_range': '1/1555/1..1/1555/1,n=1'}) == '4:U'
    assert '| l1 | 4 | 2 | 0 | 0 | 0 | 2 |' in r, r
    # against the R72 device file (R72-result.md §2): 67 of 84 L1 cells on cpu 7 off-core, 17 clean; the cpu-7 thread
    # off-core in all 18 L4 arms on 7,4,5 and in the pos 0 stock arm of 4-7 for each subject (21 of 36); every L3 cell with
    # cpu 7; the L1 11008 hot stock_before cell on cpu 4
    j = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'campaigns', '03-dense15-5pct', 'reports', 'device',
                     'r72-c3probe-r33-2026-11-22T1759.json')
    if os.path.exists(j):
        q = parse(json.load(open(j)))['r32']
        def first(tag, cpu, kern=None):
            return [lab for k, f in q[tag] if 'thr' in f and (kern is None or f.get('kern') == kern)
                    for c, fl, *_ in thr_labels(f) if c == cpu for lab in [(fl or 'C')[0]]]
        l1_7, l4_7, l1_4 = first('l1', 7), first('l4', 7), first('l1', 4)
        assert (len(l1_7), l1_7.count('M'), l1_7.count('C')) == (84, 67, 17), (len(l1_7), l1_7.count('M'))
        assert (len(l4_7), l4_7.count('M')) == (36, 21) and l1_4.count('M') == 1 and set(first('l3', 7)) == {'M'}
        arms = [f for k, f in q['l4'] if 'thr' in f and f.get('set') == '7,4,5']
        assert len(arms) == 18 and all('M' in dict((c, fl) for c, fl, *_ in thr_labels(f))[7] for f in arms)
        print('selftest r72 ok')
    print('selftest ok')


if __name__ == '__main__':
    if sys.argv[1:] == ['--selftest']:
        selftest()
    else:
        print(report(parse(json.load(open(sys.argv[1])))))
