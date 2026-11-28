# Campaign 3: tabulates a kit r39 research report (R73, the first real decode: "Save to Downloads" ->
# vollie-research-r39-<time>.json). Sections: conditions, files, load path per model (choice A: the app's rule and the
# buffer types llama.cpp printed), runs (decode tok/s, prompt tokens, gate, residency, clocks, per-worker labels),
# "R73 inputs" per model (median of the scored runs, identity, voids), and the profiled run's per-token split.
# usage: python3 -I tools/c3_decode_parse.py <report.json>      python3 -I tools/c3_decode_parse.py --selftest
# Every number printed is *device* (from the phone's report); medians and ratios are arithmetic on them.
import json, re, statistics, sys

# Per-worker labels, as tools/c3_probe_parse.py (thr_labels) defines them, applied to the decode's own lines. One letter
# per condition met, none = C:
#   M misplaced     graph starts on the worker's own cpu (4 + i, from c2-pool's histogram) < ON_MIN of all its starts
#   U under-clocked lowest 1 Hz scaling_cur_freq of the worker's policy during the run < CLK_MIN x cpuinfo_max_freq
#   P pre-empted    decode-phase involuntary switches (c2-sched) > NIV_HI, or time on cpu / (on cpu + run queue) < CW_MIN
# The thresholds are this parser's defaults (calibration knobs): a pre-registration that uses them copies them.
ON_MIN, CLK_MIN, CW_MIN, NIV_HI = 0.9, 0.9, 0.9, 50
HW_MHZ = {0: 1804, 4: 2419, 7: 2841}   # SD888 policies, used only when the report has no cpuinfo_max_freq
MASK = re.compile(r'0x([0-9a-f]+)')


def policy(cpu):
    return 7 if cpu >= 7 else 4 if cpu >= 4 else 0


def med(v):
    v = [x for x in v if x is not None]
    return statistics.median(v) if v else None


def f3(x, d=3):
    return '-' if x is None else ('%.' + str(d) + 'f') % x


def line_of(row, head):
    for l in row.get('c2', []):
        if l.startswith(head):
            return l
    return None


def workers(row, hw):
    """[(i, cpu wanted, on share, min MHz of its policy, nivcsw, on-cpu share, label)] from c2-pool / c2-sched / clk_1hz"""
    pool = line_of(row, 'c2-pool ')
    if not pool:
        return []
    masks = (re.search(r' mask=(\S+)', pool).group(1)).split(',')
    sched = {}
    s = next((l for l in row.get('c2', []) if l.startswith('c2-sched ') and ' phase=decode ' in l), None)
    if s:
        for m in re.finditer(r' w(\d+)=(\d+):(\d+):(\d+):(\d+)', s):
            sched[int(m.group(1))] = tuple(int(x) for x in m.groups()[1:])
    clk = row.get('clk_1hz') or []
    lo = {}
    for k, p in ((1, 0), (2, 4), (3, 7)):
        v = [c[k] for c in clk[1:] if isinstance(c[k], int) and c[k] > 0]
        lo[p] = min(v) if v else None
    out = []
    for m in re.finditer(r' w(\d+)=(\d+):0x([0-9a-f]+):s(\d+):e(\d+):(\S*)', pool):
        i = int(m.group(1))
        mk = MASK.match(masks[min(i, len(masks) - 1)])
        want = int(mk.group(1), 16) if mk else 0
        hist = {int(c): int(n) for c, n in re.findall(r'c(\d+)=(\d+)', m.group(6))}
        tot = sum(hist.values())
        on = sum(n for c, n in hist.items() if want >> c & 1) / tot if tot else None
        cpu = (want & -want).bit_length() - 1 if want else -1
        p = policy(cpu)
        ex, wq, nv, ni = sched.get(i, (0, 0, 0, 0))
        cw = ex / (ex + wq) if ex + wq > 0 else None
        fl = ''
        if on is not None and on < ON_MIN:
            fl += 'M'
        if lo.get(p) is not None and lo[p] < CLK_MIN * hw.get(p, HW_MHZ[p]):
            fl += 'U'
        if ni > NIV_HI or (cw is not None and cw < CW_MIN):
            fl += 'P'
        out.append((i, cpu, on, lo.get(p), ni, cw, fl or 'C'))
    return out


def hwmax(rep):
    c = ((rep.get('r73') or {}).get('clk') or {}).get('cpuinfo_max_mhz') or {}
    hw = dict(HW_MHZ)
    for k, v in c.items():
        if isinstance(v, int) and v > 0:
            hw[int(k)] = v
    return hw


def buffers(lines):
    """the buffer types named in the load's 'model buffer size' lines, e.g. 'CPU_Mapped' (choice A) or 'CPU_REPACK'"""
    t = set()
    for l in lines or []:
        m = re.search(r'(\S+) model buffer size', l)
        if m:
            t.add(m.group(1))
    return ','.join(sorted(t)) or '-'


def table(L, cols, rows):
    L.append('| ' + ' | '.join(cols) + ' |')
    L.append('|' + '---|' * len(cols))
    for r in rows:
        L.append('| ' + ' | '.join(str(x) for x in r) + ' |')


def report(rep):
    L = []
    hw = hwmax(rep)
    L.append('# R73 decode report: kit %s, app %s, started %s, finished %s, stopped early %s' % (
        rep.get('kit'), rep.get('app'), rep.get('started'), rep.get('finished'), rep.get('stopped_early')))
    st = rep.get('storage') or {}
    L.append('conditions: %s; %s; storage %s, exec %s; cpuinfo_max_mhz %s' % (
        rep.get('session_power'), rep.get('session_label'), st.get('used'), st.get('exec_factor'),
        '/'.join(str(hw[p]) for p in (0, 4, 7))))
    runs = rep.get('runs', [])
    plugged = [r['pos'] for r in runs if r.get('plugged_run')]
    psave = [r['pos'] for r in runs if (r.get('power') or {}).get('powerSave') is True]
    L.append('void echoes: charger in runs %s; power saving on in runs %s' % (plugged or 'none', psave or 'none'))
    L.append('')
    L.append('## Files and load path (choice A)')
    rows, bad = [], []
    for m, d in (rep.get('models') or {}).items():
        ok = d.get('sha256_end') == d.get('expected_sha256')
        if not ok:
            bad.append(m)
        first = next((r for r in runs if r.get('model') == m and r.get('load_rule')), {})
        lr = first.get('load_rule') or {}
        tp = (rep.get('templates') or {}).get(m) or {}
        rows.append((m, d.get('bytes'), (d.get('sha256_end') or '-')[:16], 'yes' if ok else 'NO', lr.get('can_copy_weights'),
                     lr.get('use_extra_buffers'), buffers(first.get('load_lines')), tp.get('rendered_tokens', '-')))
    table(L, ['model', 'bytes', 'sha256_end', 'sha ok', 'canCopyWeights', 'extra buffers', 'buffer types', 'rendered tokens'], rows)
    L.append('files whose end sha256 differs (void that subject): %s' % (', '.join(bad) or 'none'))
    L.append('')
    L.append('## Runs')
    rows = []
    for r in runs:
        w = workers(r, hw)
        clk = r.get('clk_1hz') or []
        caps = [c[6] for c in clk if isinstance(c[6], int) and c[6] > 0]
        rows.append((r.get('pos'), r.get('id'), r.get('phase'), f3(r.get('decode_tps')), r.get('prompt_tokens'),
                     f3(r.get('ttft_ms'), 0), r.get('wait_end', '-'), r.get('gate_timeout', '-'), f3(r.get('heat')),
                     r.get('majflt'), r.get('wait_lost_mb', '-'), (r.get('res_after') or {}).get('total_mb', '-'),
                     min(caps) if caps else '-', ','.join('%d:%s' % (c, fl) for _, c, _, _, _, _, fl in w) or '-',
                     'error' if r.get('error') else ''))
    table(L, ['pos', 'id', 'phase', 'decode tok/s', 'prompt tok', 'ttft ms', 'gate', 'timeout', 'heat', 'majflt',
              'wait lost MB', 'res MB', 'min cap7 MHz', 'labels', ''], rows)
    L.append('')
    L.append('## R73 inputs (per model; scored = phase on)')
    for m in (rep.get('models') or {}):
        sc = [r for r in runs if r.get('model') == m and r.get('phase') == 'on' and not r.get('error')]
        tps = [r.get('decode_tps') for r in sc]
        texts = {r.get('text') for r in sc}
        clean = [r.get('decode_tps') for r in sc if workers(r, hw) and all(x[6] == 'C' for x in workers(r, hw))]
        offc = sum(1 for r in sc if any('M' in x[6] for x in workers(r, hw)))
        L.append('%s: scored %d, median decode %s tok/s (min %s, max %s); all-C subset %d runs, median %s; runs with an '
                 'off-core worker %d; distinct reply texts %d; prompt tokens %s; gate timeouts %d' % (
                     m, len(sc), f3(med(tps)), f3(min(tps) if tps else None), f3(max(tps) if tps else None), len(clean),
                     f3(med(clean)), offc, len(texts), sorted({r.get('prompt_tokens') for r in sc}, key=str),
                     sum(1 for r in sc if r.get('gate_timeout'))))
        pr = [r for r in runs if r.get('model') == m and r.get('phase') == 'prof']
        for r in pr:
            t = [x for x in r.get('tok', []) if len(x) > 4]
            if t and r.get('tok_format') == 'c1':   # ab tok us head router majflt minflt mhz4 mhz7 ...
                L.append('  profiled %s: %d tokens, median us/token %s, head %s, rest %s, min mhz4/mhz7 %s/%s' % (
                    r.get('id'), len(t), f3(med([x[2] for x in t]), 0), f3(med([x[3] for x in t]), 0),
                    f3(med([x[2] - x[3] for x in t]), 0), min(x[7] for x in t), min(x[8] for x in t)))
            else:
                L.append('  profiled %s: no per-token lines' % r.get('id'))
    return '\n'.join(L)


def selftest():
    def run(pos, model, phase, tps, pool_hist7='c7=40', cap7=2841, niv=3, text='The ocean is salty.'):
        return {'pos': pos, 'id': '%s_off_%s' % (model, phase), 'model': model, 'phase': phase, 'decode_tps': tps,
                'prompt_tokens': 156, 'ttft_ms': 9000.0, 'wait_end': 'gate', 'gate_timeout': False, 'heat': 0.6,
                'majflt': 10, 'wait_lost_mb': 0, 'res_after': {'total_mb': 3479}, 'text': text, 'power': {'powerSave': False},
                'load_rule': {'can_copy_weights': False, 'use_extra_buffers': False} if pos == 1 else None,
                'load_lines': ['load_tensors:   CPU_Mapped model buffer size =  3318.00 MiB'] if pos == 1 else None,
                'clk_1hz': [[0, 1804, 2419, 2841, 1804, 2419, 2841], [1000, 1804, 2419, cap7, 1804, 2419, cap7]],
                'tok': [['A', 1, 150000, 9000, 0, 0, 0, 2419, 2841]] if phase == 'prof' else [], 'tok_format': 'c1',
                'c2': ['c2-pool run=x pool=1 mask=0x10,0x20,0x40,0x80 threads=4 graphs=1/32 pool_graphs=1/32 w0=1:0x10:s1:e0:c4=33 '
                       'w1=2:0x20:s1:e0:c5=33 w2=3:0x40:s1:e0:c6=33 w3=4:0x80:s1:e0:%s' % pool_hist7,
                       'c2-sched run=x phase=decode n=32 w0=900:10:5:1 w1=900:10:5:1 w2=900:10:5:1 w3=900:10:5:%d' % niv]}
    rep = {'kit': 'r39', 'app': '1.39.0+39', 'started': 't0', 'finished': 't1', 'stopped_early': False,
           'session_power': 'power-save off', 'session_label': 'deep-sleeping list on',
           'storage': {'used': 'internal', 'exec_factor': 'live'},
           'r73': {'clk': {'cpuinfo_max_mhz': {'0': 1804, '4': 2419, '7': 2841}}},
           'models': {'yi15-6b': {'bytes': 3479326848, 'expected_sha256': '702c', 'sha256_end': '702c'}},
           'templates': {'yi15-6b': {'rendered_tokens': 156}},
           'runs': [run(1, 'yi15-6b', 'warmup', 5.0), run(2, 'yi15-6b', 'on', 6.0), run(3, 'yi15-6b', 'on', 5.0, 'c1=30,c7=10'),
                    run(4, 'yi15-6b', 'on', 7.0, cap7=1555), run(5, 'yi15-6b', 'on', 6.5, niv=80, text='Other.'),
                    run(6, 'yi15-6b', 'prof', 5.5)]}
    r = report(rep)
    assert '| yi15-6b | 3479326848 | 702c | yes | False | False | CPU_Mapped | 156 |' in r, r
    assert '| 2 | yi15-6b_off_on | on | 6.000 | 156 | 9000 | gate | False | 0.600 | 10 | 0 | 3479 | 2841 | 4:C,5:C,6:C,7:C |  |' in r, r
    assert '7:M |' in r and '7:U |' in r and '7:P |' in r, r
    assert ('yi15-6b: scored 4, median decode 6.250 tok/s (min 5.000, max 7.000); all-C subset 1 runs, median 6.000; runs with '
            'an off-core worker 1; distinct reply texts 2; prompt tokens [156]; gate timeouts 0') in r, r
    assert 'profiled yi15-6b_off_prof: 1 tokens, median us/token 150000, head 9000, rest 141000, min mhz4/mhz7 2419/2841' in r, r
    assert 'void echoes: charger in runs none; power saving on in runs none' in r, r
    print('selftest r73 ok')


if __name__ == '__main__':
    if sys.argv[1:] == ['--selftest']:
        selftest()
    else:
        print(report(json.load(open(sys.argv[1]))))
