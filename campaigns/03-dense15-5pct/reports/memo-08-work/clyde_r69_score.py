# R69 scorer (lead, 2026-11-22, written before any R69 data). Applies prereg/R69-probe-read-rate-capacity-gpu-device.md
# and prereg/R69-amendment-1.md mechanically to a c3 probe JSON, using tools/c3_probe_parse.py's parse().
# usage: python3 -I campaigns/03-dense15-5pct/reports/memo-08-work/clyde_r69_score.py <vollie-c3probe-r29-*.json>
#        python3 -I campaigns/03-dense15-5pct/reports/memo-08-work/clyde_r69_score.py --selftest
# Inputs are *device*; every derived number is arithmetic on them (*estimated* where it is a price). The sheet
# clyde-r69-scoring.md says how each line maps to a rule; where this script and the sheet differ, the sheet wins.
import json, os, statistics, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', '..', 'tools'))
from c3_probe_parse import parse, okmark, num, SELFTEST  # noqa: E402

BANDS = [  # (name, centre, lo, hi) from R69 §3
    ('r q4x4 3 threads', 24, 19, 30), ('read 3 A78', 27, 20, 36), ('q4x4/read 3 A78', 0.90, 0.75, 1.00),
    ('(X1+3 A78)/3 A78 q4x4', 1.13, 1.00, 1.30), ('p2hot A78 cycles/w', 0.105, 0.095, 0.118),
    ('p2hot X1 cycles/w', 0.078, 0.070, 0.090), ('p2hot A55 cycles/w', 0.33, 0.22, 0.48), ('c_b', 40.4, 35, 45),
    ('P2col k col', 2.6, 2.0, 3.3), ('P2col k skip', 0.95, 0.85, 1.10), ('X1 gain 2', 1.05, 0.95, 1.25),
    ('X1 gain 4', 1.05, 0.88, 1.35), ('X2 GPU read', 30, 15, 42), ('X2 GPU q4', 10, 3, 22), ('S', 1.25, 0.95, 1.60),
    ('C GB', 3.1, 2.6, 3.8)]
STEPS = ['2.7', '3.0', '3.4', '4.2', '4.6']


def cell(p, kern, cores):
    """GB/s of a p2 cell; None if missing or a worker left its cpu ('*')."""
    for f in p['p2']:
        if f.get('kern') == kern and f.get('cores') == cores:
            return num(f['gbs']) if okmark(f) == '' else None
    return None


def p1_steps(p):
    """R69 §3 'held': median gbs over passes 6..20 (1-based; pass index >= 5) >= 0.80 x the same arm's 2.7 median, and
    median majflt per pass <= 50; the better arm counts. A step with no pass >= 5 is 'not measured'."""
    arms = {}
    for f in p['p1']:
        arms.setdefault((f['n'], f['map']), []).append(f)
    med = {k: (statistics.median(num(f['gbs']) for f in v if int(f['pass']) >= 5),
               statistics.median(int(f['majflt']) for f in v if int(f['pass']) >= 5))
           for k, v in arms.items() if any(int(f['pass']) >= 5 for f in v)}
    out = {}
    for n in STEPS:
        st = []
        for m in ('r', 'x'):
            if (n, m) in med and ('2.7', m) in med:
                g, mf = med[(n, m)]
                st.append(g >= 0.80 * med[('2.7', m)][0] and mf <= 50)
        out[n] = None if not st else any(st)
    c = None
    for n in STEPS:
        if out[n] is not True:
            break
        c = float(n)
    return out, c


def gain_a(R, b, k, d):
    return max(R, 1) / max(b * R, k * d)


def score(p):
    L, v = [], {}
    r = cell(p, 'q4x4', '7,4-5')
    L.append('r source: %s' % ('X1 + 2 A78 (7,4-5)' if r is not None else '3 A78 (4-6), fallback of R69 §3'))
    r = r if r is not None else cell(p, 'q4x4', '4-6')
    hot = {f['core']: num(f['ns_per_w']) for f in p['p2hot'] if okmark(f) == ''}
    # R69 §3 bands are cycles per weight; the line prints ns. cycles = ns x the cell's own cluster clock (GHz), the mean
    # of the before and after readings of field mhz=<cpu0>/<cpu4>/<cpu7>->...; A55 uses cpu0, A78 cpu4, X1 cpu7.
    cyc = {}
    for f in p['p2hot']:
        if okmark(f) == '' and '->' in f.get('mhz', ''):
            i = {'a55': 0, 'a78': 1, 'x1': 2}[f['core']]
            ghz = statistics.mean(num(x.split('/')[i]) for x in f['mhz'].split('->')) / 1000
            cyc[f['core']] = num(f['ns_per_w']) * ghz
    cb = 0.87 * 0.5625 * (1 / hot['x1'] + 2 / hot['a78']) if 'x1' in hot and 'a78' in hot else None
    q3, rd3 = cell(p, 'q4x4', '4-6'), cell(p, 'read', '4-6')
    col = {f['kern']: num(f['ns_per_touched_w']) for f in p['p2col'] if f.get('cores') == '4' and okmark(f) == ''}
    kcol = col['col'] / col['stock'] if 'col' in col and 'stock' in col else None
    kskip = col['skip'] / col['stock'] if 'skip' in col and 'stock' in col else None
    x1 = {f['streams']: num(f['gbs']) for f in p['x1'] if okmark(f) == ''}
    x2 = {f['kern']: num(f['gbs']) for f in p['x2']}
    tog = [num(f['sum_gbs']) for f in p['x1gpu'] if f.get('gpu_kern') == 'read' and okmark(f) == '']
    S = tog[0] / q3 if tog and q3 else None
    steps, C = p1_steps(p)
    ratio = lambda a, b: a / b if a is not None and b else None
    v.update({'r q4x4 3 threads': r, 'read 3 A78': rd3, 'q4x4/read 3 A78': ratio(q3, rd3),
              '(X1+3 A78)/3 A78 q4x4': ratio(cell(p, 'q4x4', '4-7'), q3), 'p2hot A78 cycles/w': cyc.get('a78'),
              'p2hot X1 cycles/w': cyc.get('x1'), 'p2hot A55 cycles/w': cyc.get('a55'), 'c_b': cb, 'P2col k col': kcol,
              'P2col k skip': kskip, 'X1 gain 2': ratio(x1.get('2'), x1.get('1')), 'X1 gain 4': ratio(x1.get('4'), x1.get('1')),
              'X2 GPU read': x2.get('read'), 'X2 GPU q4': x2.get('q4'), 'S': S, 'C GB': C})
    L.append('\n| quantity | device | band | in band |\n|---|---|---|---|')
    for n, c, lo, hi in BANDS:
        x = v.get(n)
        L.append('| %s | %s | %s (%s to %s) | %s |' % (n, '-' if x is None else '%.3f' % x, c, lo, hi,
                 '-' if x is None else ('hit' if lo <= x <= hi else 'miss')))
    L.append('| P1 steps held (None = not measured) | %s | | |' % steps)
    if r is None or cb is None:
        L.append('\nRules A, B and the lines need r and c_b: not scored.')
        return '\n'.join(L)
    R = cb / r
    L.append('\nR = c_b / r = %.3f' % R)
    # Rule A at reference d (amendment item 1) and Yi (item 2, provisional until the freeze)
    for name, d in (('reference d 0.60 / Yi at 40 %', 0.60), ('Yi at 42 % (row 0.60)', 0.58)):
        if kcol is None:
            L.append('Rule A %s: no P2col col row on cpu 4: not scored' % name); continue
        b, ks = 1 - 8 / 9 * (1 - d), max(R, 1) / d
        g = gain_a(R, b, kcol, d)
        L.append('Rule A %s: k %.2f, k* %.2f, gain %.3f -> %s' % (name, kcol, ks, g,
                 'KILL' if kcol > ks else ('PASS' if g >= 1.15 else 'open, gain under 1.15')))
    L.append('Rule A reference d 0.65, Llama-2, DeepSeek (k,v dense, d 0.666 -> row 0.65): no 0.65 row in kit r29: not scored from R69')
    if kcol is not None:
        cmp_, byt = 0.166 + 0.834 * 0.60 * kcol, 0.166 + 0.834 * (1 - 8 / 9 * 0.40)
        L.append('  descriptive only (*estimated*, no verdict): k,v-dense rule priced with the 0.60 row: gain %.3f'
                 % (max(R, 1) / max(byt * R, cmp_)))
    L.append('Rule B: r / c_b = %.3f -> %s' % (r / cb, 'KILL (R20 stays closed)' if r >= 0.71 * cb else
             ('REOPEN as host item (inner kill: match rate < 0.4 on any of Yi, Llama-2, DeepSeek)' if r <= 0.61 * cb else 'paper re-price only')))
    held = lambda n: steps.get(n) is True
    if held('4.6') and C != 4.6:
        L.append('Rule C flag: 4.6 held but a smaller step is not; C (all smaller held) = %s; scored as written, flagged' % C)
    L.append('Rule C: C = %s GB -> %s' % (C, 'PASS (4.6 held): Y1 first build' if held('4.6') else
             ('KILL for Q4_0 bodies >= 6 B: freeze files with resident estimate <= C' if steps.get('3.4') is False else
              ('in between: served files priced against C; reserves of memo-06 §6' if held('3.4') else '3.4 not measured: C is a lower bound, rule C not scored'))))
    gpu_dead = 'X2 GPU read' not in v or v['X2 GPU read'] is None or v['X2 GPU read'] < 10
    best_x1 = max([g for g in (v['X1 gain 2'], v['X1 gain 4']) if g] or [1.0])
    bus = ratio(cell(p, 'q4x4', '0-6'), q3)
    if S is None or S < 1.15 or gpu_dead:
        dv = 'KILL GPU lane (S < 1.15, X2 read < 10 GB/s after the -cl-opt-disable retry, or no number)'
    elif S >= 1.30 and S >= 1.15 * best_x1:
        dv = 'rank 1 stays open'
    else:
        dv = 'rank 1 parked; rank 2 built only if X1 gain >= 1.15 (best %.3f)' % best_x1
    L.append('Rule D: S %s, X2 read %s, best X1 gain %.3f -> %s' % (S, v['X2 GPU read'], best_x1, dv))
    if bus is not None and bus <= 1.10 and (S is None or S < 1.15):
        L.append('Rule D bus limit: (3 A78 + 4 A55) / 3 A78 = %.3f <= 1.10 and S < 1.15 -> ranks 1, 2 and 4 close' % bus)
    L.append('\nAmendment item 6 lines (*estimated* from *device* r, c_b): B12 = r x (83.3 - O) ms, W12 = c_b/0.5625 x (83.3 - O) ms')
    for O in (6, 8, 12):
        L.append('  O %2d ms: B12 %.2f GB, W12 %.2f G weights' % (O, r * (83.333 - O) / 1000, cb / 0.5625 * (83.333 - O) / 1000))
    return '\n'.join(L)


def selftest():
    rep = json.loads(json.dumps(SELFTEST))
    rep['blocks'][0]['lines'][1:1] = ['c3-probe: p2 kern=q4x4 cores=7,4-5 gbs=25.00 ns_per_w=0.1 cpu_ok=3/3 mhz=1/2/3->1/2/3',
                                     'c3-probe: p2hot core=a78 cpu=4 ws_kb=256 ns_per_w=0.0434 cpu_ok=1/1 mhz=1804/2419/2841->1804/2419/2841',
                                     'c3-probe: p2hot core=x1 cpu=7 ws_kb=512 ns_per_w=0.0275 cpu_ok=1/1 mhz=1804/2419/2841->1804/2419/2841']
    rep['blocks'][3]['lines'] += ['c3-probe: p1 n=%s map=r pass=%d s=0.2 gbs=%s majflt=0 cached=1 swapfree=1 vmswap=0'
                                  % (n, i, 18 if n not in ('4.2', '4.6') else 5) for n in STEPS for i in range(5, 8)]
    s = score(parse(rep))
    assert 'r source: X1 + 2 A78' in s and '| c_b | 40.' in s, s
    assert "'3.4': True, '4.2': False" in s and 'C = 3.4 GB -> in between' in s, s
    assert 'Rule A reference d 0.60 / Yi at 40 %: k 2.60' in s and 'not scored from R69' in s, s
    assert 'KILL GPU lane' in s and 'O  8 ms: B12 1.88 GB' in s, s
    print('selftest ok')


def voids(rep):
    """R69 §4 void inputs per block, printed raw for the lead: charger plugged voids the block, power saving on voids the
    block, a block without its 'end' line counts as killed. The APK hash is checked by the owner against r29/SHA256SUMS."""
    L = ['\nvoid inputs: kit=%s app=%s power_start=%s power_end=%s plugged_start=%s deep_sleeping_list=%s stopped_early=%s'
         % tuple(rep.get(k) for k in ('kit', 'app', 'power_start', 'power_end', 'plugged_start', 'deep_sleeping_list', 'stopped_early'))]
    for b in rep.get('blocks', []):
        ended = any(l.split('c3-probe:', 1)[-1].split()[1:2] == ['end'] for l in b.get('lines', []) if 'c3-probe:' in l)
        L.append('  block %s: plugged_before=%s power_before=%s end_line=%s gate_timeout=%s%s' % (
            b.get('block'), b.get('plugged_before'), b.get('power_before'), ended, b.get('gate', {}).get('timeout'),
            ' -> VOID (charger)' if b.get('plugged_before') else (' -> VOID (killed: no end line)' if not ended else '')))
    # clocks prediction (R69 §3): cpu4 >= 2419 and cpu7 >= 2841 MHz at both readings of a cell, in >= 80 % of cells
    ok = n = 0
    for b in rep.get('blocks', []):
        for l in b.get('lines', []):
            if 'mhz=' in l and '->' in l:
                rd = l.split('mhz=')[1].split()[0].split('->')
                n += 1; ok += all(len(x.split('/')) == 3 and int(x.split('/')[1]) >= 2419 and int(x.split('/')[2]) >= 2841 for x in rd)
    L.append('clocks: %d of %d cells at 2,419 (cpu 4) and 2,841 (cpu 7) at both readings -> %s (prediction P 0.8: >= 80 %%)'
             % (ok, n, '-' if not n else ('hit' if ok >= 0.8 * n else 'miss')))
    return '\n'.join(L)


if __name__ == '__main__':
    if sys.argv[1:] == ['--selftest']:
        selftest(); print(voids(SELFTEST))
    else:
        rep = json.load(open(sys.argv[1]))
        print(score(parse(rep)) + voids(rep))
