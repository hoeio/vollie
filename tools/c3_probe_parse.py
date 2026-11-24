# Campaign 3: tabulates a c3 probe report (kit r29, "Save c3 probe to Downloads"; patches/llama.cpp-c3/0001-probe; and
# 0002-d-rows: the per-block "hdr" lines (D3) and the "p2d" rows (D2); kit r32, 0003-r72-rows: the R72 rows a, l1, l4,
# l2/l3, l5, l6, l7, anon and x2r / x2info, under "R72 rows"; per-thread labels M / U / P / C on l1-l4, see thr_labels; kit r34,
# 0005-x3-gpu-probe: the R74 x3 blocks x3b / x3g / x3c / x3r under "R74 x3 GPU lane", see x3report; kit r35, 0007-x4-join-probe:
# the R75 x4 blocks x4i / x4a / x4b / x4f / x4c / x4d / x4e / x4s / x4r under "R75 x4 GPU join", see x4report; x3-tagged lines
# printed inside an x4 block (build, load, identity, alone, cond) are reported there, not under R74); kit r36, 0008-x4f-fix:
# x4 la dl= / stopped_at= / flush_recovered=, the x4f no-flush check (x4 la nf), x4 dl per block, x4 pin (x4i), x4 hang and the
# block's 'hang' flag (the app's per-block timeout).
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
           'p2d': [], 'p2d_id': None, 'hdr': [], 'r32': {t: [] for t in R32}, 'x3': [], 'x4': [],
           'gates': [], 'errors': [], 'meta': {k: rep.get(k) for k in ('kit', 'app', 'started', 'label', 'session_label',
                                                                          'deep_sleeping_list', 'stopped_early', 'model_unloaded')}}
    for b in rep.get('blocks', []):
        g = b.get('gate', {})
        out['gates'].append((b.get('block'), g.get('timeout'), (g.get('series') or [[None, None]])[-1], b.get('ms'), b.get('rc')))
        if 'error' in b:
            out['errors'].append('%s: %s' % (b.get('block'), b['error']))
        if b.get('hang'):   # kit r36: the app's per-block timeout; lines are what the block wrote to x4-live-<blk>.txt
            out['errors'].append('%s: hung (block did not return; lines from the live file)' % b.get('block'))
        for line in b.get('lines', []):
            if 'c3-probe:' not in line:
                continue
            tag, f = kv(line)
            if line.split('c3-probe:', 1)[1].split()[1:2] == ['end']:
                out['meta']['%s_s' % tag] = f.get('s')
            elif 'error' in f:
                out['errors'].append(' '.join(line.split('c3-probe:', 1)[1].split()))
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
            elif tag in ('x4', 'x4d') or (tag == 'x3' and str(b.get('block', '')).startswith('x4')):
                out['x4'].append((line.split('c3-probe:', 1)[1].split()[1] if tag != 'x4d' else 'x4d_' + line.split('c3-probe:', 1)[1].split()[1], f, b.get('block')))
            elif tag == 'x3':
                out['x3'].append((line.split('c3-probe:', 1)[1].split()[1], f, b.get('block')))
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
    L += x3report(p['x3'])
    L += x4report(p['x4'])
    L += header(p['hdr'])
    return '\n'.join(L)


def x3report(x3):
    """kit r34 (R74): the x3 GPU lane rows as printed by the phone (*device*); ratios are arithmetic on them. The kill lines
    of tom-host.md §1.5 item 9 are echoed as numbers for the lead, not scored here."""
    if not x3:
        return []
    by = {}
    for kind, f, blk in x3:
        by.setdefault(kind, []).append(dict(f, block=blk))
    L = ['\nR74 x3 GPU lane (kit r34; numbers *device*, ratios arithmetic on them)']
    for f in by.get('info', [])[:1]:
        L.append('  info: ' + ', '.join('%s=%s' % kv_ for kv_ in f.items() if kv_[0] != 'block'))
    for f in by.get('src', []):
        L.append('  src %s: %s bytes, sha256 %s' % (f['file'], f['bytes'], f['sha256']))
    for f in by.get('cond', []):   # read once per block (T6 rider, descriptive)
        L.append('  cond (%s): %s' % (f['block'], ', '.join('%s=%s' % kv_ for kv_ in f.items() if kv_[0] != 'block')))
    if 'build' in by:
        L.append('\nbuild (child process libvollie_clc.so: rc 0 ok, 3 only -cl-opt-disable, 1 failed, -n signal, -1001 timeout)')
        table(L, by['build'], ['block', 'kern', 'try', 'rc', 'how', 's', 'opts', 'bin_bytes', 'log_bytes'])
    for f in by.get('buildlog', []) + by.get('loadlog', []):
        L.append('  log %s: %s' % (f['kern'], f['text'][:600]))
    if 'load' in by:
        L.append('\nload (binary into the app process)'); table(L, by['load'], ['block', 'kern', 'how', 'bin_bytes', 'create_err', 'bin_status', 'build_err', 'kernel_err', 'wg_max', 'ok'])
    if 'identity' in by:
        L.append('\nidentity against float64 (functional pass: max_rel_err < 1e-2, no bad rows)')
        table(L, by['identity'], ['kern', 'm', 'k', 'err', 'max_rel_err', 'max_abs', 'rows_bad', 'nonfinite', 'ok'])
    if 'alone' in by:
        L.append('\nGPU alone (gbs = q + d bytes per wall second; dev_gbs from profiling events)')
        table(L, by['alone'], ['kern', 'm', 'k', 'regime', 'copies', 'gbs', 'gw_per_s', 'us_per_gemv', 'dev_gbs', 'gpu_mhz', 'gpu_max_mhz', 'hr', 'ramp_ok', 'err'])
    if 'rt' in by:
        L.append('\ndispatch round trip (launch + clFinish per call, plain queue; dev_us from the profiling queue)')
        table(L, by['rt'], ['kern', 'm', 'k', 'calls', 'wall_us', 'p50_us', 'p90_us', 'dev_us', 'ovh_us', 'queued_to_start_us', 'err'])
    if 'layer' in by:
        L.append('\none layer per subject, 4-layer ring (sync1: one clFinish per layer; sync7: one per launch)')
        table(L, by['layer'], ['subj', 'mode', 'layer_mb', 'ms_per_layer', 'body_ms', 'gbs', 'gpu_mhz', 'hr', 'err'])
        table(L, by.get('layersum', []), ['subj', 'ms_sync1', 'ms_sync7', 'dev_ms', 'rt_us_per_dispatch', 'ovh_us_per_dispatch', 'body_ms_sync1', 'body_ms_dev'])
    if 'tog' in by:
        L.append('\nCPU (ggml Q4_0, 4-7) and GPU (11008 x 4096 cold) alone and together, ABBA (labels on the CPU threads)')
        table(L, labelled([f for f in by['tog'] if 'pos' in f]), ['pos', 'arm', 'cpu_gbs', 'gpu_gbs', 'sum_gbs', 'labels', 'feeder', 'mhz_range', 'gpu_mhz', 'hr', 'err'])
        table(L, by.get('togsum', []), ['cpu_alone_gbs', 'gpu_alone_gbs', 'both_cpu_gbs', 'both_gpu_gbs', 'sum_gbs', 'sum_vs_cpu_alone', 'cpu_loss_gbs', 'gpu_loss_gbs'])
    if 'split' in by:
        L.append('\nrow split of one 11008 x 4096 matrix (CPU rows through ggml on 4-7, GPU rows by the general kernel; sync per matrix)')
        table(L, labelled([f for f in by['split'] if 'pos' in f]), ['pos', 'share', 'gpu_rows', 'ms_per_mat', 'cpu_ms', 'wait_ms', 'gpu_ms', 'labels', 'mhz_range', 'gpu_mhz', 'hr', 'err'])
        table(L, by.get('splitsum', []), ['share', 'gpu_rows', 't_split_ms', 't_cpu_alone_ms', 'ratio', 'feeder_cpu_over_wall', 'feeder_cpus'])
    if 'res' in by:
        L.append('\nresidency with GPU buffers held (x3r)')
        for f in by['res']:
            if 'pass' not in f:
                L.append('  ' + ', '.join('%s=%s' % kv_ for kv_ in f.items() if kv_[0] != 'block'))
        ps = [f for f in by['res'] if 'pass' in f]
        if ps:
            g = [num(f['gbs']) for f in ps]
            L.append('  passes %d: CPU GB/s median %.2f, last 5 median %.2f, majflt total %d, memavail first %s last %s' % (len(ps), statistics.median(g), statistics.median(g[-5:]),
                     sum(int(f['majflt']) for f in ps), ps[0]['memavail'], ps[-1]['memavail']))
    cold = [f for f in by.get('alone', []) if f.get('kern') == 's11008_1_4096' and f.get('regime') == 'cold']
    ts = by.get('togsum', [])
    L.append('\ndraft kill-line inputs (tom-host.md 1.5 item 9; for the lead, not scored here): build general %s; GPU alone cold 11008 x 4096 %s GB/s '
             '(line 7); sum / CPU alone %s (line 1.15)' % (','.join(f['how'] for f in by.get('build', []) if f.get('kern') == 'general' and 'how' in f) or '-',
                                                          cold[0]['gbs'] if cold else '-', ts[0]['sum_vs_cpu_alone'] if ts else '-'))
    return L


def x4report(x4):
    """kit r35 (R75): the x4 GPU join rows as printed by the phone (*device*); ratios are arithmetic on them. The kill lines of
    memo-12 work (ebony-joins.md 7.1 X4-a..f, bonnie-joins.md 5) are echoed as inputs for the lead, not scored here."""
    if not x4:
        return []
    by = {}
    for kind, f, blk in x4:
        by.setdefault(kind, []).append(dict(f, block=blk))
    L = ['\nR75 x4 GPU join (kit r35; numbers *device*, ratios arithmetic on them)']
    for k in ('info', 'dev', 'recq', 'entry'):
        for f in by.get(k, [])[:1]:
            L.append('  %s: ' % k + ', '.join('%s=%s' % kv_ for kv_ in f.items() if kv_[0] != 'block'))
    for f in by.get('ext', [])[:1]:
        L.append('  extensions (%s): %s' % (f.get('n'), f.get('text')))
    for f in by.get('pin', []):   # kit r36: can the x4i thread pin to each big cpu at idle (ok@<cpu seen> or e<errno>)
        L.append('  pin (x4i, idle): ' + ', '.join('%s=%s' % kv_ for kv_ in f.items() if kv_[0] != 'block'))
    for f in by.get('kai', []):
        L.append('  kleidiai (x4k child): ' + ', '.join('%s=%s' % kv_ for kv_ in f.items() if kv_[0] != 'block'))
    if 'therm' in by:
        L.append('\nthermal status (AThermal, 0 none .. 6 shutdown) and kgsl thermal_pwrlevel before / after each block')
        table(L, by['therm'], ['blk', 'when', 'status', 'thermal_pwrlevel', 'gpu_max_mhz', 'gpu_mhz'])
    for f in by.get('cond', []):   # x3-tagged cond lines of each x4 block (GPU devfreq, THP)
        L.append('  cond (%s): %s' % (f['block'], ', '.join('%s=%s' % kv_ for kv_ in f.items() if kv_[0] != 'block')))
    if 'build' in by:
        L.append('\nbuild (x4i, clc child)'); table(L, by['build'], ['block', 'kern', 'try', 'rc', 'how', 's', 'bin_bytes'])
    if 'load' in by:
        L.append('\nload'); table(L, by['load'], ['block', 'kern', 'how', 'build_err', 'kernel_err', 'wg_max', 'ok'])
    if 'identity' in by:
        L.append('\nidentity against float64 (x4i: 2816 rows; x4c: stacked 22016 and 12288 rows)')
        table(L, by['identity'], ['block', 'kern', 'm', 'k', 'err', 'max_rel_err', 'rows_bad', 'nonfinite', 'ok'])
    tail = ['lcpu_us', 'lcpu_over_wall', 'lplace', 'labels', 'mhz_range', 'gpu_mhz', 'gpu_max_mhz', 'ramp_s', 'ramp_ok']
    if 'tlsum' in by:
        L.append('\nx4a launch timeline (profiling queue; us; gap = START(i) - END(i-1), i >= 1; busy = device time / span)')
        table(L, labelled(by['tlsum']), ['case', 'lcpu', 'reps', 'wall_us', 'enq_us', 'q2s_first_us', 's2b_first_us', 'q2s_rest_p50_us', 's2b_rest_p50_us', 'gap_p50_us',
                                         'gap_max_us', 'busy', 'err'] + tail)
    if 'rt' in by:
        L.append('\nx4b round trip at 2816 rows: launcher cpu x wait mode (rep 1 = the 0-3 finish cell again at the end)')
        table(L, labelled(by['rt']), ['rep', 'lcpu', 'wait', 'calls', 'wall_us', 'p50_us', 'p90_us', 'enq_us', 'dev_us', 'ovh_us', 'queued_to_start_us',
                                      'submit_to_start_us', 'queued_to_submit_us', 'err'] + tail)
    if 'la' in by:
        las = [f for f in by['la'] if 'try' not in f]; nf = [f for f in by['la'] if 'try' in f]
        L.append('\nx4f launch-ahead (gate: behind a user event set ahead_us after the enqueue; set_to_start = set_to_done - dev, host clock;'
                 ' r36: dl = 2 s event-wait deadlines, stopped_at = iteration where the mode stopped, -1 none)')
        table(L, labelled(las), ['mode', 'n', 'ahead_us', 'set_to_done_p50_us', 'dev_us', 'set_to_start_p50_us', 'set_to_start_p90_us', 'queued_to_submit_p50_us',
                                 'submit_to_start_p50_us', 'enq_us', 'err', 'dl', 'stopped_at', 'flush_recovered'] + tail)
        if nf:
            L.append('\nx4f no-flush check (r36: gate released without a clFlush after it; 2 s wait, then clFlush and 2 s more)')
            table(L, nf, ['try', 'done_without_flush', 'wait_us', 'done_after_flush', 'after_flush_us', 'ok', 'err'])
    if 'dl' in by:
        L.append('\nevent-wait deadlines per block (r36; 2 s each)'); table(L, by['dl'], ['blk', 'n'])
    for f in by.get('hang', []):
        L.append('\nHANG: block %s did not return within %s s (%s lines kept from the live file)' % (f.get('blk'), f.get('s'), f.get('lines')))
    if 'rtargs' in by:
        L.append('\nx4f SetKernelArg: all 15 per call against the 2 that change (wait spin, cpu 4)')
        table(L, labelled(by['rtargs']), ['pos', 'args', 'calls', 'wall_us', 'p50_us', 'enq_us', 'dev_us', 'ovh_us', 'err'] + tail)
    if 'alone' in by:
        L.append('\nx4c cold rates, general kernel (stacked 22016 = gate||up, 12288 = q||k||v)')
        table(L, by['alone'], ['kern', 'm', 'k', 'regime', 'copies', 'gbs', 'us_per_gemv', 'dev_gbs', 'gpu_mhz', 'ramp_ok', 'err'])
        table(L, by.get('stacksum', []), ['gate_up_gbs', 'm11008_gbs', 'gate_up_vs_11008', 'qkv_gbs', 'm4096_gbs', 'qkv_vs_4096'])
    for f in by.get('x4d_skip', []):
        L.append('\nx4d skipped: reason=%s %s' % (f.get('reason'), f.get('err', '')))
    if 'rec' in by:
        L.append('\nx4d recordable queue (Llama-2 layer, 4-layer ring; replay against sync1, feeder on 0-3)')
        for f in by['rec']:
            if 'pos' not in f:
                L.append('  ' + ', '.join('%s=%s' % kv_ for kv_ in f.items() if kv_[0] != 'block'))
        table(L, [f for f in by['rec'] if 'pos' in f], ['pos', 'mode', 'layers', 'ms_per_layer', 'gbs', 'gpu_mhz', 'ramp_ok', 'err'])
        table(L, by.get('recsum', []), ['ms_sync1', 'ms_replay', 'dev_ms', 'ovh_us_per_launch_sync1', 'ovh_us_per_launch_replay', 'replay_vs_sync1'])
    if 'child' in by:
        L.append('\nchild processes (rc 0 ok; -1001 killed at the timeout; -n signal)'); table(L, by['child'], ['block', 'blk', 'rc', 's', 'timeout', 'lines', 'reason'])
    for f in by.get('childlog', []):
        L.append('  childlog %s: %s' % (f.get('blk'), f.get('text', '')[:600]))
    if 'pk' in by:
        L.append('\nx4e persistent kernel')
        fl = [f for f in by['pk'] if f.get('test') == 'flag' and 'mode' in f]; ct = [f for f in by['pk'] if f.get('test') == 'counter' and 'mode' in f]
        gm = [f for f in by['pk'] if f.get('test') == 'gemv']
        other = [f for f in by['pk'] if f.get('test') not in ('flag', 'counter', 'gemv') or (f.get('test') != 'gemv' and 'mode' not in f)]   # incl. the c11 build lines
        if fl:
            table(L, fl, ['mode', 'n', 'seen', 'timeout_at', 'first_us', 'rtt_p50_us', 'rtt_p90_us', 'rtt_min_us', 'kernel_exit', 'dev_ms', 'ok', 'reason', 'opt', 'err'])
        if ct:
            table(L, ct, ['mode', 'first_change_us', 'changes', 'iv_p50_us', 'step_p50', 'last_seen', 'kernel_exit', 'final', 'stop_seen_us', 'dev_ms', 'visible_during_run', 'reason', 'opt', 'err'])
        if gm:
            table(L, gm, ['kern', 'groups', 'mb', 'launches', 'gbs', 'err'])
        for f in other:
            L.append('  ' + ', '.join('%s=%s' % kv_ for kv_ in f.items() if kv_[0] != 'block'))
        table(L, by.get('pksum', []), ['q4n_gbs', 'q4p48_gbs', 'q4p192_gbs', 'q4p48_vs_q4n', 'q4p192_vs_q4n', 'same_rows'])
    for f in by.get('mc', []):
        if 'pos' not in f:
            L.append('\nx4m multi-column gemv (_mc3): ' + ', '.join('%s=%s' % kv_ for kv_ in f.items() if kv_[0] != 'block'))
    if 'mcid' in by:
        L.append('\nx4m identity per column against float64'); table(L, by['mcid'], ['n_cols', 'col', 'err', 'max_rel_err', 'rows_bad', 'nonfinite', 'ok'])
    if any('pos' in f for f in by.get('mc', [])):
        L.append('\nx4m cold rates, 11008 x 4096, ABBA (gbs = weight bytes per wall second; dev_us per launch)')
        table(L, [f for f in by['mc'] if 'pos' in f], ['pos', 'n_cols', 'kern', 'gbs', 'us_per_gemv', 'dev_us', 'dev_gbs', 'gpu_mhz', 'ramp_ok', 'err'])
        table(L, by.get('mcsum', []), ['n_cols', 'dev_us_1', 'dev_us_n', 'dev_ratio', 'wall_us_1', 'wall_us_n', 'wall_ratio', 'reason'])
    if 'tog' in by or 'tog2' in by or 'togmix' in by:
        L.append('\nx4m together: CPU ggml Q4_0 (4-7, x [4096, n]) and GPU 11008 x 4096 cold (mc3 at n 2, general at n 1), ABBA; GB/s = weight bytes')
        for f in by.get('tog2', []) + [f for f in by.get('tog', []) if 'pos' not in f]:
            L.append('  ' + ', '.join('%s=%s' % kv_ for kv_ in f.items() if kv_[0] != 'block'))
        table(L, labelled([f for f in by.get('tog', []) if 'pos' in f]), ['n_cols', 'pos', 'arm', 'cpu_gbs', 'gpu_gbs', 'sum_gbs', 'cpu_ms_per_graph', 'gpu_dev_us', 'gpu_dev_gbs',
                                                                        'labels', 'feeder', 'mhz_range', 'gpu_mhz', 'hr', 'err'])
        for f in by.get('togmix', []):
            if 'pos' not in f:
                L.append('  mixed: ' + ', '.join('%s=%s' % kv_ for kv_ in f.items() if kv_[0] != 'block'))
        mx = [f for f in by.get('togmix', []) if 'pos' in f]
        if mx:
            L.append('  mixed arm: CPU at 1 column beside GPU mc3 at gpu_n_cols')
            table(L, labelled(mx), ['gpu_n_cols', 'pos', 'arm', 'cpu_gbs', 'gpu_gbs', 'sum_gbs', 'cpu_ms_per_graph', 'gpu_dev_us', 'gpu_dev_gbs', 'labels', 'mhz_range', 'gpu_mhz', 'hr', 'err'])
        table(L, by.get('togmixsum', []), ['gpu_n_cols', 'cpu_alone_gbs', 'gpu_alone_gbs', 'both_cpu_gbs', 'both_gpu_gbs', 'both_cpu_vs_tog1_both_cpu', 'sum_gbs', 'sum_vs_cpu_alone',
                                           'cpu_alone_ms', 'both_cpu_ms', 'gpu_alone_dev_us', 'both_gpu_dev_us', 'err'])
        sums = [dict(f, n_cols=k[3]) for k in ('tog2sum', 'tog1sum') for f in by.get(k, [])]
        table(L, sums, ['n_cols', 'cpu_alone_gbs', 'gpu_alone_gbs', 'both_cpu_gbs', 'both_gpu_gbs', 'sum_gbs', 'sum_vs_cpu_alone', 'cpu_loss_gbs', 'gpu_loss_gbs',
                        'cpu_alone_ms', 'both_cpu_ms', 'gpu_alone_dev_us', 'both_gpu_dev_us', 'err'])
    if 'split' in by:
        L.append('\nx4s row split of fused [gate;up] 22016 x 4096, batch 1 (lead: cpu 4 launches, clFlush, CPU rows, event spin; feeder: R74 mode)')
        table(L, labelled([f for f in by['split'] if 'pos' in f]), ['pos', 'share', 'mode', 'gpu_rows', 'ms_per_mat', 'cpu_ms', 'wait_ms', 'gpu_ms', 'gpu_dev_ms', 'enq_us',
                                                                    'labels', 'mhz_range', 'gpu_mhz', 'hr', 'err'])
        table(L, by.get('splitsum', []), ['share', 'mode', 'gpu_rows', 'arms', 't_split_ms', 't_cpu_alone_ms', 'ratio', 'wait_ms', 'gpu_ms'])
    if 'handoff' in by or 'handoffarm' in by:
        L.append('\nx4s hand-off per GPU join at 6656 rows (cpu 4 alone: input in, launch, spin, output out; extra_us = wall - none\'s wall)')
        hs = [f for f in by.get('handoff', []) if 'mode' not in f]
        for f in hs:
            L.append('  ' + ', '.join('%s=%s' % kv_ for kv_ in f.items() if kv_[0] != 'block'))
        table(L, labelled(by.get('handoffarm', [])), ['pos', 'mode', 'joins', 'joins_per_s', 'write_us', 'read_us', 'wall_us', 'checked', 'matched', 'labels', 'mhz_range', 'err'])
        table(L, [f for f in by.get('handoff', []) if 'mode' in f], ['mode', 'write_us', 'read_us', 'wall_us', 'extra_us', 'checked', 'matched', 'match', 'arms', 'reason'])
    if 'res' in by:
        L.append('\nx4r residency with 1.5 GB of GPU buffers held')
        for f in by['res']:
            if 'pass' not in f:
                L.append('  ' + ', '.join('%s=%s' % kv_ for kv_ in f.items() if kv_[0] != 'block'))
        ps = [f for f in by['res'] if 'pass' in f]
        if ps:
            g = [num(f['gbs']) for f in ps]
            L.append('  passes %d: CPU GB/s median %.2f, passes 5+ median %s, last 5 median %.2f, majflt total %d, memavail first %s last %s' % (
                len(ps), statistics.median(g), '%.2f' % statistics.median(g[5:]) if len(g) > 5 else '-', statistics.median(g[-5:]),
                sum(int(f['majflt']) for f in ps), ps[0]['memavail'], ps[-1]['memavail']))
            mm = sorted({f.get('gpu_max_mhz', '-') for f in ps})
            L.append('  gpu_max_mhz over the passes: %s' % ','.join(mm))
    # inputs to the memo-12 kill lines (descriptive; the pre-registration scores them)
    def pick(rows, **kw):
        return [f for f in rows if all(f.get(k) == v for k, v in kw.items())]
    def g(rows, k):
        return rows[0].get(k, '-') if rows else '-'
    rts = [f for f in by.get('rt', []) if num(f.get('ovh_us')) is not None]
    best = min(rts, key=lambda f: num(f['ovh_us'])) if rts else None
    ss = [f for f in by.get('splitsum', []) if f.get('mode') == 'lead' and num(f.get('ratio')) is not None and num(f['ratio']) > 0]
    bs = min(ss, key=lambda f: num(f['ratio'])) if ss else None
    idn = {f.get('m'): f.get('ok') for f in by.get('identity', []) if f.get('block') == 'x4c'}
    pks = by.get('pksum', [{}])[0] if by.get('pksum') else {}
    L.append('\nkill-line inputs (memo-12 work; for the lead, not scored here):')
    L.append('  X4-a gap_p50_us at lcpu 4: batch8 %s, layer7 %s, batch8flush %s (0-3: %s / %s / %s)' % tuple(
        g(pick(by.get('tlsum', []), case=c, lcpu=l), 'gap_p50_us') for l in ('4', '0-3') for c in ('batch8', 'layer7', 'batch8flush')))
    L.append('  X4-b best ovh_us %s (lcpu %s, wait %s); queued_to_submit_us there %s' % (best['ovh_us'], best['lcpu'], best['wait'], best.get('queued_to_submit_us')) if best else '  X4-b -')
    L.append('  X4-c identity 22016 ok=%s, 12288 ok=%s; gate_up_vs_11008 %s, qkv_vs_4096 %s' % (idn.get('22016', '-'), idn.get('12288', '-'),
             g(by.get('stacksum', []), 'gate_up_vs_11008'), g(by.get('stacksum', []), 'qkv_vs_4096')))
    L.append('  X4-d replay_vs_sync1 %s%s' % (g(by.get('recsum', []), 'replay_vs_sync1'), (' (skip: %s)' % by['x4d_skip'][0].get('reason')) if by.get('x4d_skip') else ''))
    L.append('  X4-e flag rtt_p50_us %s, counter visible_during_run %s, q4p48_vs_q4n %s' % (
        '/'.join('%s:%s' % (f.get('mode'), f.get('rtt_p50_us', f.get('reason'))) for f in by.get('pk', []) if f.get('test') == 'flag' and 'mode' in f) or '-',
        '/'.join('%s:%s' % (f.get('mode'), f.get('visible_during_run', f.get('reason'))) for f in by.get('pk', []) if f.get('test') == 'counter' and 'mode' in f) or '-',
        pks.get('q4p48_vs_q4n', '-')))
    L.append('  X4-f gate set_to_start_p50_us %s (plain %s)' % (g(pick(by.get('la', []), mode='gate'), 'set_to_start_p50_us'), g(pick(by.get('la', []), mode='plain'), 'set_to_start_p50_us')))
    nf = [f for f in by.get('la', []) if 'try' in f]
    if nf:
        L.append('  X4-f no-flush check done_without_flush %s, done_after_flush %s' % ('/'.join(f.get('done_without_flush', '-') for f in nf), '/'.join(f.get('done_after_flush', '-') for f in nf)))
    L.append('  cg (x4m dev_ratio) n_cols 2 %s, 3 %s, 4 %s' % tuple(g([f for f in by.get('mcsum', []) if f.get('n_cols') == str(n)], 'dev_ratio') for n in (2, 3, 4)))
    L.append('  together sum_vs_cpu_alone n_cols 2 %s%s, n_cols 1 %s' % (g(by.get('tog2sum', []), 'sum_vs_cpu_alone'),
             (' (skip: %s)' % by['tog2'][0].get('reason')) if by.get('tog2') else '', g(by.get('tog1sum', []), 'sum_vs_cpu_alone')))
    L.append('  mixed (CPU 1 col, GPU %s cols) both_cpu_vs_tog1_both_cpu %s, sum_vs_cpu_alone %s' % (g(by.get('togmixsum', []), 'gpu_n_cols'),
             g(by.get('togmixsum', []), 'both_cpu_vs_tog1_both_cpu'), g(by.get('togmixsum', []), 'sum_vs_cpu_alone')))
    L.append('  kleidiai q4 kernel %s (features %s, q4_buft %s)' % (g(by.get('kai', [])[:1], 'kernel'), g(by.get('kai', [])[:1], 'features'), g(by.get('kai', [])[:1], 'q4_buft')))
    L.append('  hand-off extra_us %s' % ('/'.join('%s:%s' % (f['mode'], f.get('extra_us', f.get('reason'))) for f in by.get('handoff', []) if 'mode' in f) or '-'))
    L.append('  X4-s best lead ratio %s at share %s (feeder 0.25: %s)' % (((bs['ratio'], bs['share']) if bs else ('-', '-')) + (g(pick(by.get('splitsum', []), mode='feeder'), 'ratio'),)))
    return L


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
    x3selftest()
    x4selftest()
    x4r36selftest()
    print('selftest ok')


X3TEST = {'kit': 'r34', 'blocks': [
    {'block': 'x3b', 'rc': 100, 'ms': 9000, 'gate': {'timeout': False, 'series': []}, 'lines': [
        'c3-probe: x3 cond thp_enabled=always,[madvise],never thp_defrag=e13',
        'c3-probe: x3 cond devfreq=soc:qcom,cpu-cpu-llcc-bw cur=7980 min=762 max=7980 gov=e13 avail=e13',
        'c3-probe: x3 info device=QUALCOMM_Adreno(TM) version=OpenCL,2.0,Adreno(TM),660 cstd=CL2.0 a6x=1 vsgb=0 e031=38',
        'c3-probe: x3 src file=gemv_noshuffle_q4_0_f32.cl bytes=22301 sha256=636126fd',
        'c3-probe: x3 build kern=general try=1 rc=-11 s=3.10 opts=-cl-std=CL2.0,-cl-mad-enable bin_bytes=0 log_bytes=0',
        'c3-probe: x3 build kern=general try=2 rc=0 s=2.00 opts=-cl-std=CL2.0,-cl-mad-enable,-cl-opt-disable bin_bytes=9000 log_bytes=12',
        'c3-probe: x3 buildlog kern=general try=2 text=warning:_x',
        'c3-probe: x3 build kern=general rc=0 how=noopt s=5.10',
        'c3-probe: x3 load kern=general how=noopt bin_bytes=9000 create_err=0 bin_status=0 build_err=0 kernel_err=0 wg_max=1024 ok=1',
        'c3-probe: x3 build kern=s11008_1_4096 rc=1 how=fail s=4.00',
        'c3-probe: x3 identity kern=general m=512 k=4096 err=0 max_rel_err=4.1e-04 max_abs=1e-3 rms_ref=3.7 rows_bad=0 nonfinite=0 ok=1',
        'c3-probe: x3b end s=12.0']},
    {'block': 'x3g', 'rc': 100, 'ms': 9000, 'gate': {'timeout': False, 'series': []}, 'lines': [
        'c3-probe: x3 alone kern=general m=11008 k=4096 regime=cold copies=21 gbs=8.50 gw_per_s=15.1 us_per_gemv=2984 dev_gbs=9.0 err=0 thr=none gpu_mhz=840..840,n=20 gpu_max_mhz=840 hr=0.41 ramp_ok=1',
        'c3-probe: x3 rt kern=general m=512 k=4096 calls=900 wall_us=180.0 p50_us=170 p90_us=220 dev_us=60 ovh_us=120 queued_to_start_us=40 err=0',
        'c3-probe: x3 layer subj=yi15-6b mode=sync1 layer_mb=92.8 ms_per_layer=11.5 body_ms=368 gbs=8.4 err=0 thr=none',
        'c3-probe: x3 layersum subj=yi15-6b ms_sync1=11.5 ms_sync7=12.4 dev_ms=10.9 rt_us_per_dispatch=128.6 ovh_us_per_dispatch=85.7']},
    {'block': 'x3c', 'rc': 100, 'ms': 9000, 'gate': {'timeout': False, 'series': []}, 'lines': [
        'c3-probe: x3 tog pos=0 arm=cpu cpu_gbs=20.00 gpu_gbs=0.00 sum_gbs=20.00 err=0 thr=4:0.990:2:0:0/30@-1,7:0.990:1:0:30/30@3 mhz_range=1804/2419/2841..1804/2419/2841,n=30 max_mhz=p0:1804/1804,p4:2419/2419,p7:2841/2841',
        'c3-probe: x3 togsum cpu_alone_gbs=20.00 gpu_alone_gbs=8.50 both_cpu_gbs=18.00 both_gpu_gbs=6.00 sum_gbs=24.00 sum_vs_cpu_alone=1.200 cpu_loss_gbs=2.00 gpu_loss_gbs=2.50',
        'c3-probe: x3 split pos=1 share=0.25 gpu_rows=2816 ms_per_mat=1.100 cpu_ms=0.900 wait_ms=0.200 gpu_ms=1.050 err=0 thr=4:0.95:0:0:0/30@-1 mhz_range=1/1555/1..1/1555/1,n=3 max_mhz=p4:2419/2419',
        'c3-probe: x3 splitsum share=0.25 gpu_rows=2816 t_split_ms=1.100 t_cpu_alone_ms=1.300 ratio=0.846 feeder_cpu_over_wall=0.990 feeder_cpus=f',
        'c3-probe: x3 tog error=cpu_alloc']},
    {'block': 'x3r', 'rc': 100, 'ms': 9000, 'gate': {'timeout': False, 'series': []}, 'lines': [
        'c3-probe: x3 res stage=gpu_alloc kern=s11008_1_4096 copies=35 gpu_mb=847 alloc_s=1.0 err=0',
        'c3-probe: x3 res pass=0 s=0.130 gbs=20.00 gpu_gbs=5.0 majflt=3 minflt=0 memavail=900000 cached=1 swapfree=2 vmswap=0 vmrss=3',
        'c3-probe: x3 res pass=1 s=0.140 gbs=18.00 gpu_gbs=5.0 majflt=4 minflt=0 memavail=800000 cached=1 swapfree=2 vmswap=0 vmrss=3',
        'c3-probe: x3 res sum passes=2 secs=0.3 cpu_gbs_median=19.00 cpu_gbs_min=18.00 gpu_gbs=5.00 majflt=7']}]}


def x3selftest():
    p = parse(X3TEST)
    assert len(p['x3']) == 23 and p['meta']['x3b_s'] == '12.0' and p['errors'] == ['x3 tog error=cpu_alloc'], (len(p['x3']), p['errors'])
    r = report(p)
    assert '| x3b | general | 1 | -11 |  |' in r and '| x3b | general |  | 0 | noopt | 5.10 |' in r and '| x3b | s11008_1_4096 |  | 1 | fail |' in r, r
    assert '| general | 512 | 4096 | 0 | 4.1e-04 | 1e-3 | 0 | 0 | 1 |' in r, r
    assert '| general | 11008 | 4096 | cold | 21 | 8.50 | 15.1 | 2984 | 9.0 | 840..840,n=20 | 840 | 0.41 | 1 | 0 |' in r, r
    assert '| 0 | cpu | 20.00 | 0.00 | 20.00 | 4:C,7:M |' in r and '| 1 | 0.25 | 2816 | 1.100 | 0.900 | 0.200 | 1.050 | 4:U |' in r, r
    assert '| yi15-6b | 11.5 | 12.4 | 10.9 | 128.6 | 85.7 |' in r and '| 0.25 | 2816 | 1.100 | 1.300 | 0.846 |' in r, r
    assert 'passes 2: CPU GB/s median 19.00, last 5 median 19.00, majflt total 7, memavail first 900000 last 800000' in r, r
    assert 'build general noopt; GPU alone cold 11008 x 4096 - GB/s (line 7); sum / CPU alone 1.200 (line 1.15)' in r, r
    assert 'src gemv_noshuffle_q4_0_f32.cl: 22301 bytes, sha256 636126fd' in r and 'cond (x3b): devfreq=soc:qcom,cpu-cpu-llcc-bw' in r, r
    print('selftest x3 ok')


X4T = ' lcpu=4 lcpu_us=150.0 lcpu_over_wall=0.300 lplace=0/50@-1 thr=4:0.300:2:50:0/20@-1 mhz_range=1804/2419/2841..1804/2419/2841,n=20 max_mhz=p0:1804/1804,p4:2419/2419,p7:2841/2841 gpu_mhz=315..840,n=20,src=devfreq/cur_freq hr=0.5 ramp_s=0.60 ramp_ok=1'
X4TEST = {'kit': 'r35', 'blocks': [
    {'block': 'x4i', 'rc': 100, 'ms': 900, 'gate': {'timeout': False, 'series': []}, 'lines': [
        'c3-probe: x3 cond kgsl gpuclk=e13 max_gpuclk=840000000 devfreq_cur=315000000',
        'c3-probe: x4 therm blk=x4i when=before status=0 thermal_pwrlevel=0 gpu_max_mhz=840 gpu_mhz=-1',
        'c3-probe: x4 kai kernel=DOTPROD features=dotprod+fp16 q4_buft=CPU_KLEIDIAI backend=load_all:CPU:CPU_KLEIDIAI| cpu_ok=1',
        'c3-probe: x4 kai log text=kleidiai:_primary_q4_kernel_feature_DOTPROD',
        'c3-probe: x4 child blk=x4k_c rc=0 s=0.30 timeout=0 lines=2',
        'c3-probe: x4 therm blk=x4i when=after status=1 thermal_pwrlevel=1 gpu_max_mhz=840 gpu_mhz=-1',
        'c3-probe: x4 info device=QUALCOMM,Adreno(TM) cstd=CL2.0 e031=38',
        'c3-probe: x4 ext n=3 text=cl_khr_fp16,cl_qcom_recordable_queues,cl_khr_subgroups',
        'c3-probe: x4 dev svm_err=0 svm_caps=0x3 svm_coarse=1 svm_fine_buffer=1 svm_fine_system=0 svm_atomics=0',
        'c3-probe: x4 recq present=1 max_size=8192 ret_bytes=8 err=0 entry=extaddr',
        'c3-probe: x3 build kern=general rc=0 how=opt s=0.29',
        'c3-probe: x3 load kern=general how=opt bin_bytes=22288 create_err=0 bin_status=0 build_err=0 kernel_err=0 wg_max=512 ok=1',
        'c3-probe: x3 identity kern=general m=2816 k=4096 err=0 max_rel_err=6.5e-04 max_abs=1e-3 rms_ref=3.9 rows_bad=0 nonfinite=0 ok=1',
        'c3-probe: x4i end s=1.5']},
    {'block': 'x4a', 'rc': 100, 'ms': 900, 'gate': {'timeout': False, 'series': []}, 'lines': [
        'c3-probe: x4 tl case=batch8 lcpu=4 rep=0 i=1 m=2816 k=4096 enq_us=90.0 queued_us=100.0 submit_us=150.0 start_us=420.0 end_us=660.0 dev_us=240.0 gap_us=20.0',
        'c3-probe: x4 tlsum case=batch8 lcpu=4 reps=5 launches=8 wall_us=2400.0 enq_us=95.0 q2s_first_us=120.0 s2b_first_us=190.0 q2s_rest_p50_us=5.0 s2b_rest_p50_us=30.0 gap_p50_us=18.0 gap_mean_us=20.0 gap_max_us=60.0 dev_sum_us=1900.0 span_us=2300.0 busy=0.826 err=0' + X4T,
        'c3-probe: x4 tlsum case=layer7 lcpu=0-3 reps=5 launches=7 wall_us=6000.0 enq_us=150.0 q2s_first_us=200.0 s2b_first_us=190.0 q2s_rest_p50_us=150.0 s2b_rest_p50_us=180.0 gap_p50_us=250.0 gap_mean_us=260.0 gap_max_us=400.0 dev_sum_us=4200.0 span_us=5900.0 busy=0.712 err=0' + X4T.replace('lcpu=4 ', 'lcpu=0-3 ')]},
    {'block': 'x4b', 'rc': 100, 'ms': 900, 'gate': {'timeout': False, 'series': []}, 'lines': [
        'c3-probe: x4 rt rep=0 wait=finish kern=general m=2816 k=4096 calls=900 wall_us=690.0 p50_us=700.0 p90_us=860.0 enq_us=110.0 dev_us=240.0 ovh_us=450.0 queued_to_start_us=395.0 submit_to_start_us=199.0 queued_to_submit_us=196.0 err=0' + X4T.replace('lcpu=4 ', 'lcpu=0-3 '),
        'c3-probe: x4 rt rep=0 wait=spin kern=general m=2816 k=4096 calls=1900 wall_us=380.0 p50_us=370.0 p90_us=420.0 enq_us=60.0 dev_us=240.0 ovh_us=140.0 queued_to_start_us=120.0 submit_to_start_us=90.0 queued_to_submit_us=30.0 err=0' + X4T.replace('lcpu=4 ', 'lcpu=7 ').replace('thr=4:', 'thr=7:')]},
    {'block': 'x4f', 'rc': 100, 'ms': 900, 'gate': {'timeout': False, 'series': []}, 'lines': [
        'c3-probe: x4 la mode=gate n=50 ahead_us=1000.0 set_to_done_p50_us=400.0 set_to_done_p90_us=450.0 dev_us=240.0 set_to_start_p50_us=160.0 set_to_start_p90_us=200.0 queued_to_submit_p50_us=1050.0 submit_to_start_p50_us=150.0 enq_us=80.0 err=0' + X4T,
        'c3-probe: x4 la mode=plain n=50 ahead_us=0.0 set_to_done_p50_us=640.0 set_to_done_p90_us=700.0 dev_us=240.0 set_to_start_p50_us=400.0 set_to_start_p90_us=460.0 queued_to_submit_p50_us=190.0 submit_to_start_p50_us=195.0 enq_us=100.0 err=0' + X4T,
        'c3-probe: x4 rtargs pos=1 args=2 wait=spin kern=general m=2816 k=4096 calls=1500 wall_us=500.0 p50_us=490.0 p90_us=600.0 enq_us=50.0 dev_us=240.0 ovh_us=260.0 queued_to_start_us=200.0 submit_to_start_us=150.0 queued_to_submit_us=50.0 err=0' + X4T]},
    {'block': 'x4c', 'rc': 100, 'ms': 900, 'gate': {'timeout': False, 'series': []}, 'lines': [
        'c3-probe: x3 identity kern=general m=22016 k=4096 err=0 max_rel_err=7.0e-04 max_abs=1e-3 rms_ref=3.9 rows_bad=0 nonfinite=0 ok=1',
        'c3-probe: x3 alone kern=general m=22016 k=4096 regime=cold copies=11 gbs=21.50 us_per_gemv=2300 dev_gbs=25 err=0 gpu_mhz=840..840,n=20 ramp_ok=1',
        'c3-probe: x4 stacksum gate_up_gbs=21.50 m11008_gbs=21.00 gate_up_vs_11008=1.024 qkv_gbs=19.00 m4096_gbs=15.00 qkv_vs_4096=1.267']},
    {'block': 'x4d', 'rc': 100, 'ms': 900, 'gate': {'timeout': False, 'series': []}, 'lines': [
        'c3-probe: x4 rec setup subj=llama2-7b layers_in_ring=4 layer_mb=101.0 new_err=0 end_err=0 launch_err=0 replay_plain_err=0 replay_recq_err=0 replay_queue=plain entry=extaddr ok=1',
        'c3-probe: x4 rec pos=0 mode=sync1 layers=700 ms_per_layer=1.400 gbs=20.0 err=0 thr=none gpu_mhz=840..840,n=20 ramp_ok=1',
        'c3-probe: x4 recsum ms_sync1=1.400 ms_replay=1.000 dev_ms=0.600 ovh_us_per_launch_sync1=114.3 ovh_us_per_launch_replay=57.1 replay_vs_sync1=0.500',
        'c3-probe: x4 child blk=x4d_c rc=0 s=9.10 timeout=0 lines=7']},
    {'block': 'x4e', 'rc': 100, 'ms': 900, 'gate': {'timeout': False, 'series': []}, 'lines': [
        'c3-probe: x4 pk test=flag mode=hostptr n=200 seen=200 timeout_at=0 first_us=900.0 rtt_p50_us=12.0 rtt_p90_us=20.0 rtt_min_us=8.0 kernel_exit=5000 dev_ms=3.000 queued_to_start_us=300.0 err=0 ok=1',
        'c3-probe: x4 pk test=flag mode=svm ok=0 reason=no_fine_grain_buffer_caps_0x1 err=0',
        'c3-probe: x4 pk c11 test=flag opt=-cl-std=CL2.0 err=-11 svm_caps=0x1 svm_atomics=0',
        'c3-probe: x4 pk test=flag mode=svm_c11 ok=0 reason=build_c11 opt=-cl-std=CL3.0 err=-11 log=error:_memory_scope',
        'c3-probe: x4 child blk=x4e1_c rc=-1001 s=5.00 timeout=1 lines=1',
        'c3-probe: x4 childlog blk=x4e1_c text=kgsl:_fault',
        'c3-probe: x4 pk test=counter mode=hostptr n=50000 spin=16 first_change_us=300.0 changes=5 iv_p50_us=50.00 iv_p90_us=60.00 step_p50=900 last_seen=4500 kernel_exit=50000 final=50000 stop_seen_us=-1.0 dev_ms=1.000 visible_during_run=1 err=0',
        'c3-probe: x4 pksum test=gemv q4n_gbs=9.00 q4p48_gbs=8.10 q4p192_gbs=8.50 q4p48_vs_q4n=0.900 q4p192_vs_q4n=0.944 same_rows=1']},
    {'block': 'x4m', 'rc': 100, 'ms': 900, 'gate': {'timeout': False, 'series': []}, 'lines': [
        'c3-probe: x4 mc setup m=11008 k=4096 kern1=general kernmc=kernel_gemv_noshuffle_q4_0_f32_mc3 wg_max=512 copies=22 mb=532 nsg=4',
        'c3-probe: x4 mcid n_cols=2 col=1 err=0 max_rel_err=7.1e-04 rms_ref=3.9 rows_bad=0 nonfinite=0 ok=1',
        'c3-probe: x4 mc pos=1 n_cols=2 kern=mc3 m=11008 k=4096 gemvs=900 gbs=20.10 us_per_gemv=1260.0 dev_us=1080.0 dev_gbs=23.50 err=0 thr=none gpu_mhz=840..840,n=20 ramp_ok=1',
        'c3-probe: x4 mcsum n_cols=2 dev_us_1=900.0 dev_us_n=1080.0 dev_ratio=1.200 wall_us_1=1200.0 wall_us_n=1260.0 wall_ratio=1.050',
        'c3-probe: x4 mcsum n_cols=4 ok=0 reason=identity',
        'c3-probe: x4 tog setup n_cols=2 cpu=ggml_q4_0 cpu_buft=CPU_REPACK cpu_mb=576 gpu_kern=mc3 gpu_m=11008 gpu_k=4096 gpu_copies=22 gpu_mb=532 set=4,5,6,7 feeder=0-3 queue=profiling backend=x',
        'c3-probe: x4 tog pos=2 n_cols=2 arm=both set=4,5,6,7 cpu_gbs=17.00 gpu_gbs=9.00 sum_gbs=26.00 cpu_ms_per_graph=35.000 gpu_dev_us=1100.0 gpu_dev_gbs=22.00 cpu_calls=40 gpu_batches=100 err=0 thr=4:0.990:2:0:0/30@-1 mhz_range=1804/2419/2841..1804/2419/2841,n=30 gpu_mhz=840..840,n=20 hr=0.5',
        'c3-probe: x4 tog2sum cpu_alone_gbs=20.00 gpu_alone_gbs=12.00 both_cpu_gbs=17.00 both_gpu_gbs=9.00 sum_gbs=26.00 sum_vs_cpu_alone=1.300 cpu_loss_gbs=3.00 gpu_loss_gbs=3.00 cpu_alone_ms=30.000 both_cpu_ms=35.000 gpu_alone_dev_us=900.0 both_gpu_dev_us=1100.0 err=0',
        'c3-probe: x4 togmix choice gpu_n_cols=3 n4_ok=0 n3_ok=1',
        'c3-probe: x4 togmix pos=2 cpu_n_cols=1 gpu_n_cols=3 arm=both set=4,5,6,7 cpu_gbs=16.00 gpu_gbs=8.00 sum_gbs=24.00 cpu_ms_per_graph=37.000 gpu_dev_us=1200.0 gpu_dev_gbs=21.00 cpu_calls=40 gpu_batches=90 err=0 thr=4:0.990:2:0:0/30@-1',
        'c3-probe: x4 togmixsum cpu_n_cols=1 gpu_n_cols=3 cpu_alone_gbs=20.00 gpu_alone_gbs=11.00 both_cpu_gbs=16.00 both_gpu_gbs=8.00 both_cpu_vs_tog1_both_cpu=0.914 sum_gbs=24.00 sum_vs_cpu_alone=1.200 cpu_alone_ms=30.000 both_cpu_ms=37.000 gpu_alone_dev_us=1000.0 both_gpu_dev_us=1200.0 err=0',
        'c3-probe: x4 tog1sum cpu_alone_gbs=20.00 gpu_alone_gbs=10.00 both_cpu_gbs=17.50 both_gpu_gbs=6.50 sum_gbs=24.00 sum_vs_cpu_alone=1.200 cpu_loss_gbs=2.50 gpu_loss_gbs=3.50 cpu_alone_ms=30.000 both_cpu_ms=34.000 gpu_alone_dev_us=800.0 both_gpu_dev_us=1000.0 err=0']},
    {'block': 'x4s', 'rc': 100, 'ms': 900, 'gate': {'timeout': False, 'series': []}, 'lines': [
        'c3-probe: x4 split pos=3 share=0.30 mode=lead gpu_rows=6656 cpu_rows=15360 mats=900 ms_per_mat=1.600 cpu_ms=1.300 wait_ms=0.050 gpu_ms=1.200 gpu_dev_ms=0.600 enq_us=90.0 gbs=30.0 err=0 thr=4:0.990:2:0:0/30@-1 mhz_range=1804/2419/2841..1804/2419/2841,n=30 max_mhz=p4:2419/2419',
        'c3-probe: x4 splitsum share=0.30 mode=lead gpu_rows=6656 arms=2 t_split_ms=1.600 t_cpu_alone_ms=1.900 ratio=0.842 wait_ms=0.050 gpu_ms=1.200',
        'c3-probe: x4 splitsum share=0.25 mode=feeder gpu_rows=5504 arms=2 t_split_ms=1.750 t_cpu_alone_ms=1.900 ratio=0.921 wait_ms=0.040 gpu_ms=1.500',
        'c3-probe: x4 handoff setup gpu_rows=6656 k=4096 copies=4 in_bytes=16384 out_bytes=26624 kern=general lead=cpu4_flush_spin_profiling cpu_rows=0',
        'c3-probe: x4 handoffarm pos=4 mode=ahp_persist joins=2000 joins_per_s=2000 write_us=2.0 read_us=4.0 wall_us=480.0 checked=2400 matched=0 err=0 thr=4:0.990:2:0:0/20@-1 mhz_range=1804/2419/2841..1804/2419/2841,n=20',
        'c3-probe: x4 handoff mode=none write_us=0.0 read_us=0.1 wall_us=470.0 extra_us=0.0 checked=0 matched=0 match=-1 arms=2',
        'c3-probe: x4 handoff mode=write write_us=40.0 read_us=60.0 wall_us=575.0 extra_us=105.0 checked=4000 matched=4000 match=1 arms=2',
        'c3-probe: x4 handoff mode=ahp_persist write_us=2.0 read_us=4.0 wall_us=478.0 extra_us=8.0 checked=4800 matched=0 match=0 arms=2']},
    {'block': 'x4r', 'rc': 100, 'ms': 900, 'gate': {'timeout': False, 'series': []}, 'lines': [
        'c3-probe: x4 res stage=gpu_alloc kern=s11008_1_4096 copies=59 gpu_mb=1430 alloc_s=1.0 err=0',
        'c3-probe: x4 res pass=0 s=0.130 gbs=20.00 gpu_gbs=5.0 majflt=3 minflt=0 memavail=900000 cached=1 swapfree=2 vmswap=0 vmrss=3',
        'c3-probe: x4 res pass=1 s=0.140 gbs=18.00 gpu_gbs=5.0 majflt=4 minflt=0 memavail=800000 cached=1 swapfree=2 vmswap=0 vmrss=3 gpu_max_mhz=840',
        'c3-probe: x4 res sum passes=2 secs=0.3 cpu_gbs_median=19.00 cpu_gbs_min=18.00 gpu_gbs=5.00 majflt=7',
        'c3-probe: x4d skip reason=no_extension']}]}


def x4selftest():
    p = parse(X4TEST)
    assert not p['x3'] and len(p['x4']) == 61 and p['meta']['x4i_s'] == '1.5' and not p['errors'], (len(p['x3']), len(p['x4']), p['errors'])
    r = report(p)
    assert 'R74 x3 GPU lane' not in r and 'R75 x4 GPU join' in r, r
    assert 'extensions (3): cl_khr_fp16,cl_qcom_recordable_queues,cl_khr_subgroups' in r and 'recq: present=1, max_size=8192' in r, r
    assert '| x4i | general | 2816 | 4096 | 0 | 6.5e-04 | 0 | 0 | 1 |' in r, r
    assert '| batch8 | 4 | 5 | 2400.0 | 95.0 | 120.0 | 190.0 | 5.0 | 30.0 | 18.0 | 60.0 | 0.826 | 0 | 150.0 | 0.300 | 0/50@-1 | 4:P |' in r, r
    assert '| 0 | 7 | spin | 1900 | 380.0 | 370.0 | 420.0 | 60.0 | 240.0 | 140.0 | 120.0 | 90.0 | 30.0 | 0 | 150.0 | 0.300 | 0/50@-1 | 7:P |' in r, r
    assert '| gate | 50 | 1000.0 | 400.0 | 240.0 | 160.0 | 200.0 | 1050.0 | 150.0 | 80.0 | 0 |' in r, r
    assert '| 1.400 | 1.000 | 0.600 | 114.3 | 57.1 | 0.500 |' in r and '| x4e | x4e1_c | -1001 | 5.00 | 1 | 1 |' in r and 'childlog x4e1_c: kgsl:_fault' in r, r
    assert '| svm |  |  |  |  |  |  |  |  |  | 0 | no_fine_grain_buffer_caps_0x1 |  | 0 |' in r and '| 9.00 | 8.10 | 8.50 | 0.900 | 0.944 | 1 |' in r, r
    assert '| 0.30 | lead | 6656 | 2 | 1.600 | 1.900 | 0.842 | 0.050 | 1.200 |' in r and 'x4d skipped: reason=no_extension' in r, r
    assert 'passes 2: CPU GB/s median 19.00, passes 5+ median -, last 5 median 19.00, majflt total 7' in r, r
    assert '  X4-a gap_p50_us at lcpu 4: batch8 18.0, layer7 -, batch8flush - (0-3: - / 250.0 / -)' in r, r
    assert '  X4-b best ovh_us 140.0 (lcpu 7, wait spin); queued_to_submit_us there 30.0' in r, r
    assert '  X4-c identity 22016 ok=1, 12288 ok=-; gate_up_vs_11008 1.024, qkv_vs_4096 1.267' in r and '  X4-d replay_vs_sync1 0.500 (skip: no_extension)' in r, r
    assert '  X4-e flag rtt_p50_us hostptr:12.0/svm:no_fine_grain_buffer_caps_0x1/svm_c11:build_c11, counter visible_during_run hostptr:1, q4p48_vs_q4n 0.900' in r, r
    assert '  X4-f gate set_to_start_p50_us 160.0 (plain 400.0)' in r and '  X4-s best lead ratio 0.842 at share 0.30 (feeder 0.25: 0.921)' in r, r
    assert '| 2 | 1 | 0 | 7.1e-04 | 0 | 0 | 1 |' in r and '| 1 | 2 | mc3 | 20.10 | 1260.0 | 1080.0 | 23.50 | 840..840,n=20 | 1 | 0 |' in r, r
    assert '| 2 | 900.0 | 1080.0 | 1.200 | 1200.0 | 1260.0 | 1.050 |  |' in r and '| 4 |  |  |  |  |  |  | identity |' in r, r
    assert 'x4m multi-column gemv (_mc3): m=11008' in r and '  cg (x4m dev_ratio) n_cols 2 1.200, 3 -, 4 -' in r, r
    assert '| svm_c11 |  |  |  |  |  |  |  |  |  | 0 | build_c11 | -cl-std=CL3.0 | -11 |' in r and '  test=flag, opt=-cl-std=CL2.0, err=-11, svm_caps=0x1, svm_atomics=0' in r, r
    assert '  X4-e flag rtt_p50_us hostptr:12.0/svm:no_fine_grain_buffer_caps_0x1/svm_c11:build_c11,' in r, r
    assert '| 2 | 2 | both | 17.00 | 9.00 | 26.00 | 35.000 | 1100.0 | 22.00 | 4:C |' in r, r
    assert '| 2 | 20.00 | 12.00 | 17.00 | 9.00 | 26.00 | 1.300 | 3.00 | 3.00 | 30.000 | 35.000 | 900.0 | 1100.0 | 0 |' in r and '| 1 | 20.00 | 10.00 | 17.50 |' in r, r
    assert '  together sum_vs_cpu_alone n_cols 2 1.300, n_cols 1 1.200' in r, r
    assert '| 4 | ahp_persist | 2000 | 2000 | 2.0 | 4.0 | 480.0 | 2400 | 0 | 4:C |' in r and '| write | 40.0 | 60.0 | 575.0 | 105.0 | 4000 | 4000 | 1 | 2 |  |' in r, r
    assert '  hand-off extra_us none:0.0/write:105.0/ahp_persist:8.0' in r and 'gpu_rows=6656, k=4096' in r, r
    assert '| x4i | before | 0 | 0 | 840 | -1 |' in r and '  kleidiai (x4k child): kernel=DOTPROD, features=dotprod+fp16' in r, r
    assert '| 3 | 2 | both | 16.00 | 8.00 | 24.00 | 37.000 | 1200.0 | 21.00 | 4:C |' in r and '| 3 | 20.00 | 11.00 | 16.00 | 8.00 | 0.914 | 24.00 | 1.200 |' in r, r
    assert '  mixed (CPU 1 col, GPU 3 cols) both_cpu_vs_tog1_both_cpu 0.914, sum_vs_cpu_alone 1.200' in r and '  kleidiai q4 kernel DOTPROD (features dotprod+fp16, q4_buft CPU_KLEIDIAI)' in r, r
    assert '  gpu_max_mhz over the passes: -,840' in r and 'mixed: choice=' not in r and 'gpu_n_cols=3, n4_ok=0, n3_ok=1' in r, r
    print('selftest x4 ok')


X4T36 = ' lcpu=4 lcpu_us=150.0 lcpu_over_wall=0.300 lplace=0/50@-1 thr=4:0.95:10:0:0/5@-1 mhz_range=1804/2419/2841..1804/2419/2841,n=5 max_mhz=p0:1804/1804,p4:2419/2419,p7:2841/2841 gpu_mhz=840..840,n=5,src=devfreq/cur_freq gpu_max_mhz=840 hr=0.6 ramp_s=0.5 ramp_ok=1'
X4TEST36 = {'kit': 'r36', 'blocks': [
    {'block': 'x4i', 'rc': 100, 'ms': 900, 'gate': {'timeout': False, 'series': []}, 'lines': [
        'c3-probe: x4 pin cpu4=ok@4 cpu5=ok@5 cpu6=ok@6 cpu7=e22',
        'c3-probe: x4 dl blk=x4i n=0', 'c3-probe: x4i end s=1.0']},
    {'block': 'x4f', 'rc': 100, 'ms': 900, 'gate': {'timeout': False, 'series': []}, 'lines': [
        'c3-probe: x4 la mode=gate n=3 ahead_us=1000.0 set_to_done_p50_us=400.0 set_to_done_p90_us=450.0 dev_us=240.0 set_to_start_p50_us=160.0 set_to_start_p90_us=200.0 queued_to_submit_p50_us=1050.0 submit_to_start_p50_us=150.0 enq_us=80.0 err=-2001 dl=1 stopped_at=7 flush_recovered=1' + X4T36,
        'c3-probe: x4 la mode=plain n=50 ahead_us=0.0 set_to_done_p50_us=640.0 set_to_done_p90_us=700.0 dev_us=240.0 set_to_start_p50_us=400.0 set_to_start_p90_us=460.0 queued_to_submit_p50_us=190.0 submit_to_start_p50_us=195.0 enq_us=100.0 err=-2001 dl=0 stopped_at=-1 flush_recovered=-1' + X4T36,
        'c3-probe: x4 la nf try=0 done_without_flush=0 wait_us=2000100.0 done_after_flush=1 after_flush_us=300.0 err=0',
        'c3-probe: x4 la nf try=1 done_without_flush=1 wait_us=500.0 done_after_flush=-1 after_flush_us=-1.0 err=0',
        'c3-probe: x4 dl blk=x4f n=1']},
    {'block': 'x4c', 'hang': True, 'gate': {'timeout': False, 'series': []}, 'lines': [
        'c3-probe: x4 therm blk=x4c when=before status=0 thermal_pwrlevel=0 gpu_max_mhz=840 gpu_mhz=-1',
        'c3-probe: x4 hang blk=x4c s=600 lines=1']}]}


def x4r36selftest():
    p = parse(X4TEST36)
    assert p['errors'] == ['x4c: hung (block did not return; lines from the live file)'], p['errors']
    r = report(p)
    assert '| gate | 3 | 1000.0 | 400.0 | 240.0 | 160.0 | 200.0 | 1050.0 | 150.0 | 80.0 | -2001 | 1 | 7 | 1 |' in r, r
    assert '| 0 | 0 | 2000100.0 | 1 | 300.0 |  | 0 |' in r and '| x4f | 1 |' in r and 'pin (x4i, idle): cpu4=ok@4, cpu5=ok@5, cpu6=ok@6, cpu7=e22' in r, r
    assert 'HANG: block x4c did not return within 600 s (1 lines kept from the live file)' in r, r
    assert '  X4-f no-flush check done_without_flush 0/1, done_after_flush 1/-1' in r, r
    print('selftest r36 ok')


if __name__ == '__main__':
    if sys.argv[1:] == ['--selftest']:
        selftest()
    else:
        print(report(parse(json.load(open(sys.argv[1])))))
