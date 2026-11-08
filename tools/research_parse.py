"""Turns a research-kit JSON report (tools/research_kit.md) into results.csv rows and R-log text.
usage: python3 -I tools/research_parse.py <report.json> [--csv] [--gate T] [--sigma S]
Prints a human table per phase (device-measured numbers), the A/B splits from the cpu-prof pairs, the
barrier cost, the speculation stats, and, with --csv, the rows to append to reports/results.csv.
--gate T (r13, R39 / reports/memos/round13_plan.md §B): the block-gated ABBA score, class-K knobs only.
--sigma S (r14, R42): the gate's σ limit on the scored line (default 1.0; r14 MMID_FUSE: 0.4). The scored line is
MUL_MAT_ID+GLU (compute + wait of both) when the run carries fuse= (r14+), else MUL_MAT_ID.
--pairs (r15, R46): the per-token interleaved runs (GGML_CPU_AB_TOKEN, lines 'cpu-prof ab=A|B tok=n'): pairs (2k−1, 2k), a pair
counts iff both tokens have majflt 0 and the run's modal clock; Δ = mean(B − A), σ = sd/√N per line; the sequential verdict per
knob (THRESH); validity (A/A null, A-token vs block-ABBA A mean, hygiene); confirmation and reported-not-scored lines. With
--gate, a ROUTER_CAST block ABBA also gets the sequential verdict on MUL_MAT over counted blocks.
--official (r16, R48 items 3-6): the official 32 decomposed: line 1 = plain-run median / best printed tok/s (claim iff median ≥ 20.0);
line 2 = excess E over decode tokens tok=2..32 of the profiled official runs against the reference story's fault-free median
S(f) per logged clock, its fault / clean split, the 6-of-8 classifier (R48 am. 1 (4): E ≤ 0 = neither; ≥ 5 of 8 with E ≤ 0 =
NO GAP); validity; reported lines incl. E against idle_ms (am. 1 (2)) and the after-pause / back-to-back means."""
import json, re, sys, statistics

def op_sum(o, *ops):
    return sum(sum(o.get(op, (0, 0, 0))[1:]) for op in ops)

def fuse_hygiene(blocks):
    """r14 (R42): every fuse=1 block must show the fused graph against the fuse=0 blocks: L = fused layers = MUL_MAT_ID count
    drop = (fuse=0 count) / 3 (gate, up, down per layer); GLU count − L, nodes − 2L, barriers − 2L (the up->gate and gate->GLU
    barriers both go). blocks: (side, blk, d, o). Returns lines; [] when no block has fuse=1."""
    cnt = lambda d, o: (o.get('MUL_MAT_ID', (0,))[0], o.get('GLU', (0,))[0], int(d.get('nodes', 0)), int(d.get('barriers', 0)))
    ref = [cnt(d, o) for _, _, d, o in blocks if d.get('fuse') == '0']
    fus = [(s, b, cnt(d, o)) for s, b, d, o in blocks if d.get('fuse') == '1']
    if not fus:
        return []
    if not ref:
        return ['hygiene n/a: fuse=1 on both sides (compiled default since R47), no fuse=0 reference']
    m, g, n, br = max(set(ref), key=ref.count)
    L = m // 3
    want = (m - L, g - L, n - 2 * L, br - 2 * L)
    bad = [f"{s}{b}" for s, b, c in fus if c != want]
    head = f"fuse=0 MUL_MAT_ID/GLU/nodes/barriers {m}/{g}/{n}/{br} -> fuse=1 wants {'/'.join(map(str, want))}"
    return [f"{head}: {'ok' if not bad else 'HYGIENE FAIL (fuse did not engage) in blocks ' + ' '.join(bad)} ({len(fus)} fuse=1 blocks)"]

def prof_fields(line):
    d = {}
    for k, v in re.findall(r'([\w/]+)=([^\s|]+)', line.split('|')[0]):
        d[k] = v
    ops = {}
    for name, cnt, comp, wait in re.findall(r'(\w+) (\d+):(\d+)\+(\d+)', line.split('|', 1)[1] if '|' in line else ''):
        ops[name] = (int(cnt), int(comp), int(wait))
    return d, ops

def abba_blocks(x):
    """(side, blk, fields, ops) of every post-warm-up block line of an ABBA run."""
    out = []
    for line in x.get('prof', []):
        m = re.search(r' ab=([AB]) blk=(\d+)', line)
        if not m or m.group(2) == '0': continue
        d, o = prof_fields(line)
        out.append((m.group(1), int(m.group(2)), d, o))
    return out

def outside_graph(x):
    """r9: ms per printed token spent outside the profiled graphs of an always-on story run with 6 windows:
    (decode wall − Σ windows − tokens not profiled × steady) / intervals, decode wall = duration_ms − ttft_ms, steady =
    the last window's us/token (Bonnie, reports/memos/last_3ms.md §1). None unless the run has exactly 6 window lines."""
    wins = [prof_fields(l)[0] for l in x.get('prof', []) if l.startswith('cpu-prof') and ' graphs=' in l and ' ab=' not in l]
    dur, ttft, tok = x.get('duration_ms'), x.get('ttft_ms'), x.get('tokens')
    if len(wins) != 6 or None in (dur, ttft, tok) or tok < 2: return None
    g = sum(int(d['graphs']) for d in wins)
    win_ms = sum(int(d['us/token']) * int(d['graphs']) for d in wins) / 1000
    steady = int(wins[-1]['us/token']) / 1000
    rest = (tok - 1) - g
    return (dur - ttft - win_ms - rest * steady) / (tok - 1), dur - ttft, win_ms, rest, steady

def bench_ratios(line):
    """r10: the pre-registered ratios of one 'cpu-bench q3k' line (§9.9 (iii)); r11 adds the core clock (mhz=start/end); r13b: the
    core check (cpu=first-last migr= cpu_ok=; UNVERIFIED if fewer than 5 valid trials in some field), medians, instrument=."""
    d = dict(re.findall(r'(\w+)=([^\s]+)', line))
    try:
        o, n, i8, q4, sc = (float(d[k]) for k in ('old', 'new', 'int8', 'q4x4', 'scalar'))
    except (KeyError, ValueError):
        return 'ratios: missing fields'
    mhz = d.get('mhz')  # r11: core clock 'start/end' MHz; r10 lines have none
    clock = (f"  {mhz} MHz (min/max over valid trials)" if 'cpu_ok' in d else f"  cpu{d.get('cpu', '?')} at {mhz} MHz (start/end)") if mhz \
        else f"  cpu{d.get('cpu', '?')} clock not logged (pre-r11)"
    if 'pinned' in d:  # r13: the module pins the bench thread to cpu 4-6
        clock += '  pinned' if d['pinned'] == '1' else '  UNPINNED'
    if 'cpu_ok' in d:  # r13b: trials whose start and end core were both in 4-6, fewest of any field / trials per field
        ok_n = int(d['cpu_ok'].split('/')[0])
        clock += f"  cpu {d.get('cpu')} migr {d.get('migr')} cpu_ok {d['cpu_ok']}" + ('  UNVERIFIED' if ok_n < 5 else '')
    cold = ''
    try:  # r13 (R39): page-stream fields; r13b: hot is measured (same slice re-run), cold / hot / cold_touch are medians
        h, c, ct = (float(d[k]) for k in ('hot', 'cold', 'cold_touch'))
        rec = f"{(c - ct) / (c - h):.2f}" if c != h else 'n/a (cold = hot)'
        thp = f"  cold_thp/hot {float(d['cold_thp'])/h:.3f}  anon_huge_kb {d.get('anon_huge_kb')}" if 'cold_thp' in d else ''
        cold = (f"\n    cold/hot {c/h:.3f}  (cold - cold_touch)/(cold - hot) {rec}{thp}"
                f"  exact cold/touch {d.get('exact_cold')}/{d.get('exact_cold_touch')}"
                + (f"  instrument={d['instrument']}" if 'instrument' in d else ''))
    except (KeyError, ValueError, ZeroDivisionError):
        pass
    return f"new/old {n/o:.3f}  int8/old {i8/o:.3f}  scalar/old {sc/o:.3f}  old/q4x4 {o/q4:.2f}" + clock + cold

def mean_sd(v):
    return (statistics.mean(v), statistics.stdev(v) if len(v) > 1 else float('nan'))

def abba_summary(runs):
    """Per side mean ± sd over blocks, B − A ± 1σ (blocks iid), per ABBA group (B̄ − Ā) mean ± sd; ms/token."""
    metrics = [('us/token', lambda d, o: int(d['us/token'])), ('wait', lambda d, o: int(d['wait'])),
               ('MUL_MAT_ID', lambda d, o: sum(o.get('MUL_MAT_ID', (0, 0, 0))[1:])),
               ('MUL_MAT', lambda d, o: sum(o.get('MUL_MAT', (0, 0, 0))[1:])),
               ('head', lambda d, o: int(d.get('head', 0)))] + \
              ([('router', lambda d, o: int(d.get('router', 0)))] if any('router' in b[2] for x in runs for b in abba_blocks(x)) else []) + \
              ([('MUL_MAT_ID+GLU', lambda d, o: op_sum(o, 'MUL_MAT_ID', 'GLU')),  # r14: the line the fuse moves work between
                ('GLU compute', lambda d, o: o.get('GLU', (0, 0, 0))[1])] if any('fuse' in b[2] for x in runs for b in abba_blocks(x)) else []) + \
              [(op, (lambda op: lambda d, o: sum(o.get(op, (0, 0, 0))[1:]))(op)) for op in ('SSM_SCAN', 'CPY', 'CONCAT', 'SSM_CONV')]
    rows = []
    blocks = [(ri, b) for ri, x in enumerate(runs) for b in abba_blocks(x)]
    for label, f in metrics:
        side = {'A': [], 'B': []}
        groups = {}
        for ri, (sd_, blk, d, o) in blocks:
            v = f(d, o) / 1000
            side[sd_].append(v)
            groups.setdefault((ri, (blk - 1) // 4), {'A': [], 'B': []})[sd_].append(v)
        if len(side['A']) < 1 or len(side['B']) < 1:
            rows.append((label, 'no blocks')); continue
        (ma, sa), (mb, sb) = mean_sd(side['A']), mean_sd(side['B'])
        se = (sa**2 / len(side['A']) + sb**2 / len(side['B'])) ** 0.5
        gd = [statistics.mean(g['B']) - statistics.mean(g['A']) for g in groups.values() if g['A'] and g['B']]
        gm, gs = mean_sd(gd) if gd else (float('nan'), float('nan'))
        rows.append((label, f"A {ma:6.2f} ± {sa:4.2f} (n {len(side['A'])})  B {mb:6.2f} ± {sb:4.2f} (n {len(side['B'])})  "
                            f"Δ {mb - ma:+6.2f} ± {se:4.2f}  groups {len(gd)}: Δ {gm:+6.2f} sd {gs:4.2f}"))
    nodes = {sd_: sorted({(d.get('nodes'), d.get('barriers')) for _, (s2, _, d, _) in blocks if s2 == sd_}) for sd_ in 'AB'}
    rows.append(('nodes/bar', f"A {' '.join(f'{n}/{b}' for n, b in nodes['A'])}  B {' '.join(f'{n}/{b}' for n, b in nodes['B'])}"))
    if any('fuse' in d for _, (_, _, d, _) in blocks):  # r14: per-side op counts and the fused-graph check
        oc = {sd_: sorted({(o.get('MUL_MAT_ID', (0,))[0], o.get('GLU', (0,))[0]) for _, (s2, _, _, o) in blocks if s2 == sd_}) for sd_ in 'AB'}
        rows.append(('MMID/GLU cnt', f"A {' '.join(f'{m}/{g}' for m, g in oc['A'])}  B {' '.join(f'{m}/{g}' for m, g in oc['B'])}"))
        for h in fuse_hygiene([b for _, b in blocks]):
            rows.append(('hygiene', h))
    for key in ('chunk', 'expq4', 'kern', 'touch', 'fuse', 'chunk2', 'rcast'):  # r9/r10/r13 header fields, per side (absent before r9 / r10 / r13)
        v = {sd_: sorted({d[key] for _, (s2, _, d, _) in blocks if s2 == sd_ and key in d}) for sd_ in 'AB'}
        if v['A'] or v['B']:
            rows.append((key, f"A {' '.join(v['A'])}  B {' '.join(v['B'])}"))
    ka = [float(d['kavg']) for _, (s_, _, d, _) in blocks if s_ == 'A']
    kb = [float(d['kavg']) for _, (s_, _, d, _) in blocks if s_ == 'B']
    if ka and kb:
        rows.append(('kavg', f"A {statistics.mean(ka):.2f}  B {statistics.mean(kb):.2f} (min {min(kb):.2f} max {max(kb):.2f})"))
    # per token: r6 adv = expert slices read ahead; r7 majflt / minflt = the process's page faults during that side's graphs
    for key in ('adv', 'majflt', 'minflt'):
        aa = [float(d[key]) for _, (s_, _, d, _) in blocks if s_ == 'A' and key in d]
        ab = [float(d[key]) for _, (s_, _, d, _) in blocks if s_ == 'B' and key in d]
        if aa and ab:
            ma, mb = statistics.mean(aa), statistics.mean(ab)
            rows.append((key, f"A {ma:.1f}  B {mb:.1f}  B/A {mb/ma if ma else float('nan'):.3f}"))
    return rows

GATE_LINES = [('MUL_MAT_ID', lambda d, o: sum(o.get('MUL_MAT_ID', (0, 0, 0))[1:])),
              ('MUL_MAT', lambda d, o: sum(o.get('MUL_MAT', (0, 0, 0))[1:])),
              ('head', lambda d, o: int(d.get('head', 0))),
              ('SSM_SCAN', lambda d, o: sum(o.get('SSM_SCAN', (0, 0, 0))[1:]))]
GATE_FUSE = [('MUL_MAT_ID+GLU', lambda d, o: op_sum(o, 'MUL_MAT_ID', 'GLU')),  # r14 (R42): scored line, then its low-variance part
             ('GLU compute', lambda d, o: o.get('GLU', (0, 0, 0))[1])]
SIGMA = 1.0  # --sigma

# r15 (R46 items 3-4): knob -> (scored line, pass threshold, falsify threshold), ms/token, Δ = B − A. Sequential verdict: pass iff
# Δ + 1.645σ <= pass; falsified iff Δ − 1.645σ > falsify; else continue (next session). A session counts with >= MIN_PAIRS pairs.
THRESH = {'MMID_FUSE+CHUNK2': ('MUL_MAT_ID+GLU', -0.4, -0.15),
          'Q3K_KERN': ('MUL_MAT_ID compute', -3.0, -1.5),
          'ROUTER_CAST': ('MUL_MAT', -0.15, 0.0)}
MIN_PAIRS = 60
NULL_CHI2 = 14.07  # R46 amendment 1 (2): χ²₇ at 0.95 over the seven PAIR_LINES
Z = 1.645

def seq_verdict(m, se, ps, fa):
    if m != m or se != se: return 'no data'
    return 'PASS' if m + Z * se <= ps else 'FALSIFIED' if m - Z * se > fa else 'continue'

def knob_of(x):
    """('NAME', a, b) of a run's GGML_CPU_AB=knob:NAME:a:b (first triple), else None."""
    m = re.match(r'knob:([^:,]+):([^:,]+):([^:,]+)', (x.get('env') or {}).get('GGML_CPU_AB', ''))
    return (m.group(1), m.group(2), m.group(3)) if m else None

PAIR_LINES = [('MUL_MAT_ID+GLU', lambda d, o: op_sum(o, 'MUL_MAT_ID', 'GLU')),
              ('MUL_MAT_ID compute', lambda d, o: o.get('MUL_MAT_ID', (0, 0, 0))[1]),
              ('MUL_MAT', lambda d, o: op_sum(o, 'MUL_MAT')),
              ('head', lambda d, o: int(d.get('head', 0))),
              ('SSM_SCAN', lambda d, o: op_sum(o, 'SSM_SCAN')),
              ('GLU compute', lambda d, o: o.get('GLU', (0, 0, 0))[1]),
              ('token total', lambda d, o: int(d['us/token']))]

def token_lines(x):
    """(side, tok, fields, ops) of every per-token line (tok >= 1) of an interleaved run."""
    out = []
    for line in x.get('prof', []):
        m = re.search(r' ab=([AB]) tok=(\d+)', line)
        if m and m.group(2) != '0':
            d, o = prof_fields(line)
            out.append((m.group(1), int(m.group(2)), d, o))
    return out

def token_pairs(x):
    """[(k, A, B, counted)] for pairs k = tokens (2k−1, 2k) holding one A and one B, and the run's modal clock (cpu4/cpu7 'mhz=' over
    all its tokens). counted iff both tokens have majflt 0 and the modal clock."""
    toks = token_lines(x)
    mz = [d.get('mhz') for _, _, d, _ in toks]
    modal = max(set(mz), key=mz.count) if mz else None
    by = {}
    for t in toks:
        by.setdefault((t[1] + 1) // 2, {})[t[0]] = t
    out = []
    for k in sorted(by):
        p = by[k]
        if 'A' in p and 'B' in p:
            ok = all(float(t[2].get('majflt', 'inf')) == 0 and t[2].get('mhz') == modal for t in (p['A'], p['B']))
            out.append((k, p['A'], p['B'], ok))
    return out, modal

def paired(prs, f):
    """Δ = mean(B − A), σ = sd/√N over counted pairs, ms/token; N."""
    dd = [(f(b[2], b[3]) - f(a[2], a[3])) / 1000 for _, a, b, ok in prs if ok]
    if len(dd) < 2: return float('nan'), float('nan'), len(dd)
    return statistics.mean(dd), statistics.stdev(dd) / len(dd) ** 0.5, len(dd)

def minflt_ratio(prs, counted=True):
    a = sum(float(p[1][2].get('minflt', 0)) for p in prs if p[3] or not counted)
    b = sum(float(p[2][2].get('minflt', 0)) for p in prs if p[3] or not counted)
    return b / a if a else float('nan')

def print_pairs(inter, abba, T):
    print("\n== per-token pairs (r15, R46; interleaved A B B A ...; pair k = tokens (2k−1, 2k); counted iff both tokens majflt 0 at the "
          f"run's modal clock; Δ = mean(B − A) ± σ = sd/√N, ms/token; verdict: pass iff Δ + {Z}σ ≤ pass, falsified iff Δ − {Z}σ > falsify, "
          f"else continue; a session counts with ≥ {MIN_PAIRS} counted pairs)")
    null_ok = None
    for name, xs in inter.items():
        kn = knob_of(xs[0]) or ('?', '0', '1')
        is_null = kn[1] == kn[2]
        print(f"  {name}  ({len(xs)} run{'s' if len(xs) > 1 else ''}){'  A/A NULL (same value under both labels)' if is_null else ''}")
        allp = []
        for i, x in enumerate(xs):
            prs, modal = token_pairs(x)
            allp += prs
            toks = token_lines(x)
            print(f"    run {i+1}: tokens {len(toks)}  pairs {len(prs)}  counted {sum(p[3] for p in prs)}  modal clock {modal}  "
                  f"minflt B/A counted {minflt_ratio(prs):.3f} all {minflt_ratio(prs, False):.3f}")
            for h in fuse_hygiene([t for t in toks]):
                print(f"      hygiene (every B token) {h}")
        res = {lab: paired(allp, f) for lab, f in PAIR_LINES}
        n = res['token total'][2]
        for lab, (m, se, nn) in res.items():
            print(f"    pooled {lab:20s} Δ {m:+7.3f} ± {se:.3f}  (N {nn})")
        if is_null:  # R46 amendment 1 (2): every |z| <= 3, Σz² over the seven lines <= 14.07 (χ²₇ at 0.95), minflt 1.00 ± 0.05
            z = {lab: m / se if se and se == se and se > 0 else (0.0 if m == 0 else float('inf')) for lab, (m, se, _) in res.items()}
            big = [k for k, v in z.items() if not abs(v) <= 3]
            chi = sum(v * v for v in z.values())
            mr = minflt_ratio(allp)
            null_ok = not big and chi <= NULL_CHI2 and abs(mr - 1) <= 0.05
            print(f"    validity A/A null: z {' '.join(f'{k} {v:+.2f}' for k, v in z.items())}")
            print(f"      every |z| ≤ 3: {'✓' if not big else '✗ ' + ' '.join(big)}; Σz² {chi:.2f} {'✓' if chi <= NULL_CHI2 else '✗'} (≤ {NULL_CHI2}); "
                  f"minflt ratio {mr:.3f} {'✓' if abs(mr - 1) <= 0.05 else '✗'} (1.00 ± 0.05) -> {'null OK' if null_ok else 'SESSION VOID'}")
            continue
        line, ps, fa = THRESH.get(kn[0], (None, None, None))
        if line:
            m, se, _ = res[line]
            v = seq_verdict(m, se, ps, fa)
            cnt = n >= MIN_PAIRS
            print(f"    SCORED {line}: Δ {m:+.3f} ± {se:.3f}, N {n}; pass ≤ {ps}, falsified > {fa}: Δ+{Z}σ {m + Z*se:+.3f}, Δ−{Z}σ {m - Z*se:+.3f} -> {v}"
                  f"{'' if cnt else f'  (session does NOT count: N {n} < {MIN_PAIRS})'}")
        # validity: interleaved A-token mean vs the block ABBAs' A mean (same knob, counted blocks at --gate T)
        blk = [x for nm, ys in abba.items() for x in ys if knob_of(x) and knob_of(x)[0] == kn[0]]
        ia = [int(p[1][2]['us/token']) / 1000 for p in allp if p[3]]
        if blk and ia:
            ba = [int(d['us/token']) / 1000 for x in blk for s2, _, d, o, ok in gate_blocks(x, T)[0] if ok and s2 == 'A']
            if ba:
                dm = statistics.mean(ia) - statistics.mean(ba)
                print(f"    reported A-token mean {statistics.mean(ia):.2f} (N {len(ia)}) vs block-ABBA A mean {statistics.mean(ba):.2f} "
                      f"(counted blocks T={T}, n {len(ba)}): {dm:+.2f} (R46 am. 1 (3): reported, not voiding; the gates differ)")
            else:
                print(f"    reported A-token vs block-ABBA A mean: no counted block-ABBA A block at T={T}")
        else:
            print("    reported A-token vs block-ABBA A mean: no block ABBA of this knob in the report")
        # confirmation (fuse) and reported, not scored
        if kn[0].startswith('MMID_FUSE'):
            m, se, _ = res['GLU compute']
            print(f"    confirmation GLU compute Δ {m:+.3f} ± {se:.3f} (pre-registered −0.12..−0.18: {'✓' if -0.18 <= m <= -0.12 else '✗'})")
            for i, x in enumerate(xs):
                r_ = minflt_ratio(token_pairs(x)[0], False)
                print(f"    confirmation run {i+1} fused-side minflt ratio (all tokens) {r_:.3f} (0.65-0.85: {'✓' if 0.65 <= r_ <= 0.85 else '✗'})")
        for lab in ('token total', 'MUL_MAT'):
            m, se, _ = res[lab]
            print(f"    reported {lab} Δ {m:+.3f} ± {se:.3f}")
        for sd_ in 'AB':
            v = sorted(int(p[1 if sd_ == 'A' else 2][2]['us/token']) / 1000 for p in allp if p[3])
            if v: print(f"    reported floor side {sd_}: min {v[0]:.2f}  p10 {v[len(v) // 10]:.2f}  median {statistics.median(v):.2f} ms/token (counted tokens)")
        if line:
            f = dict(PAIR_LINES)[line]
            for sd_ in 'AB':
                xy = [(float(t[2]['minflt']), f(t[2], t[3]) / 1000) for p in allp if p[3] for t in [p[1 if sd_ == 'A' else 2]]]
                if len(xy) > 2 and len({a for a, _ in xy}) > 1:
                    mx, my = statistics.mean(a for a, _ in xy), statistics.mean(b for _, b in xy)  # least squares (py3.9: no linear_regression)
                    sl = sum((a - mx) * (b - my) for a, b in xy) / sum((a - mx) ** 2 for a, _ in xy)
                    print(f"    reported minflt slope side {sd_}: {sl * 100:+.2f} ms per 100 minflt/token on {line} (diagnostic)")
    if null_ok is not None:
        print(f"  SESSION {'valid (A/A null OK)' if null_ok else 'VOID (A/A null failed)'}")
    else:
        print("  no A/A null run in this report: session validity not established")

def gate_blocks(x, T):
    """r13 §B: (side, blk, d, o, counted) per post-warm-up block of one ABBA run, and the run's modal clock. A block counts iff
    majflt/token <= T and its mhz_blocks entry (cpu4/cpu7; entries map onto blocks by token position) is the run's modal clock.
    ponytail: the kit samples the clock every 8 text chunks, not tokens (r12: 3 entries per 200 tokens), so several blocks share
    one entry; no mhz_blocks -> no block counts."""
    mb = [tuple(m) for m in x.get('mhz_blocks') or []]
    modal = max(set(mb), key=mb.count) if mb else None
    tok = x.get('tokens') or 200
    out = []
    for side, blk, d, o in abba_blocks(x):
        clk = mb[min(len(mb) - 1, 8 * (blk + 1) * len(mb) // tok)] if mb else None
        out.append((side, blk, d, o, float(d.get('majflt', 'inf')) <= T and clk == modal))
    return out, modal

def gate_score(blocks):
    """Counted blocks -> (nA, nB, majflt A, B, {line: (Δ, σ)}, verdict, flags). Δ = B − A, σ = sqrt(sA²/nA + sB²/nB)."""
    cnt = {s: [(d, o) for s2, _, d, o, ok in blocks if ok and s2 == s] for s in 'AB'}
    na, nb = len(cnt['A']), len(cnt['B'])
    mf = {s: statistics.mean(float(d['majflt']) for d, _ in cnt[s]) if cnt[s] else float('nan') for s in 'AB'}
    res = {}
    fused = any('fuse' in d for s2, _, d, o, ok in blocks)
    for label, f in GATE_LINES + (GATE_FUSE if fused else []):
        a = [f(d, o) / 1000 for d, o in cnt['A']]; b = [f(d, o) / 1000 for d, o in cnt['B']]
        if len(a) < 2 or len(b) < 2:
            res[label] = (float('nan'), float('nan')); continue
        res[label] = (statistics.mean(b) - statistics.mean(a), (statistics.variance(a) / len(a) + statistics.variance(b) / len(b)) ** 0.5)
    scored = 'MUL_MAT_ID+GLU' if fused else 'MUL_MAT_ID'
    flags = {'n>=8/side': na >= 8 and nb >= 8, '|majflt A-B|<=3': abs(mf['A'] - mf['B']) <= 3, f'sigma<={SIGMA:.1f}': res[scored][1] <= SIGMA}
    if fused:  # r14: a counted block without the fused graph voids the run
        flags['hygiene'] = not any('FAIL' in h for h in fuse_hygiene([(s2, b, d, o) for s2, b, d, o, ok in blocks]))
    verdict = 'scorable' if all(flags.values()) else 'unresolved'
    return na, nb, mf, res, verdict, flags

def print_gate(abba, T):
    print(f"\n== block gate T={T} (R39 §B; class-K knobs only: a block counts iff majflt/token <= {T} and the run's modal clock; "
          "Δ = B − A ± 1σ over counted blocks, ms/token; MUL_MAT_ID = compute + wait; r14: scored line MUL_MAT_ID+GLU)")
    for name, xs in abba.items():
        print(f"  {name}  ({len(xs)} run{'s' if len(xs) > 1 else ''})")
        allb = []
        for i, x in enumerate(xs):
            bl, modal = gate_blocks(x, T)
            allb += bl
            ids = {s: ' '.join(str(blk) for s2, blk, _, _, ok in bl if ok and s2 == s) for s in 'AB'}
            print(f"    run {i+1}: modal clock {'/'.join(map(str, modal)) if modal else 'NOT LOGGED'}; first block {bl[0][0] if bl else '-'}; counted A {{{ids['A']}}} B {{{ids['B']}}}")
        for title, bl in [(f'run {i+1}', gate_blocks(x, T)[0]) for i, x in enumerate(xs)] + [('pooled', allb)]:
            na, nb, mf, res, verdict, flags = gate_score(bl)
            print(f"    {title:7s} counted A {na} / B {nb}  majflt A {mf['A']:.1f} B {mf['B']:.1f}  " +
                  '  '.join(f"{k} {v[0]:+.2f} ± {v[1]:.2f}" for k, v in res.items()))
            bias = [(k, res[k][0]) for k in ('MUL_MAT', 'head', 'SSM_SCAN') if abs(res[k][0]) > 0.3]
            bt = '  bias: ' + ', '.join(f"{k} {v:+.2f}" for k, v in bias) + f" (line scored only if |Δ| > {2 * max(abs(v) for _, v in bias):.2f})" if bias else ''
            print(f"            {'  '.join(f'{k} {chr(10003) if v else chr(10007)}' for k, v in flags.items())} -> {verdict}{bt}")
            kn = knob_of(xs[0])
            if kn and kn[0] == 'ROUTER_CAST' and title == 'pooled':  # r15 (R46 item 3): C1-F16, sequential rule on counted blocks
                cnt = {s: [(d, o) for s2, _, d, o, ok in bl if ok and s2 == s] for s in 'AB'}
                for lab, f in [('router', lambda d, o: int(d.get('router', 0)))]:
                    a_ = [f(d, o) / 1000 for d, o in cnt['A']]; b_ = [f(d, o) / 1000 for d, o in cnt['B']]
                    if len(a_) > 1 and len(b_) > 1:
                        print(f"            router (µs field) Δ {statistics.mean(b_) - statistics.mean(a_):+.3f} ± "
                              f"{(statistics.variance(a_) / len(a_) + statistics.variance(b_) / len(b_)) ** 0.5:.3f} (reported)")
                line, ps, fa = THRESH['ROUTER_CAST']
                m, se = res[line]
                print(f"            SCORED {line} (C1-F16): Δ {m:+.3f} ± {se:.3f}; pass ≤ {ps}, falsified > {fa} -> {seq_verdict(m, se, ps, fa)}"
                      f"  (side means majflt A {mf['A']:.1f} B {mf['B']:.1f}; text identity: see the reply-text section)")

def clock_of(d):
    """cpu4/cpu7 MHz of a per-token line's mhz= field, e.g. (2112, 2035)."""
    try:
        return tuple(int(v) for v in d.get('mhz', '').split('/'))
    except ValueError:
        return None

def tok_ms(d, o, what):
    """ms of one per-token line: the token total, or an op (compute + wait), or 'head' / 'router' fields."""
    if what == 'token': return int(d['us/token']) / 1000
    if what in ('head', 'router'): return int(d.get(what, 0)) / 1000
    return op_sum(o, what) / 1000

STEADY = ('token', 'MUL_MAT', 'MUL_MAT_ID')

def steady_ref(x):
    """R48 item 4: S(f) = median over the reference story's fault-free tokens (majflt 0) at logged clock f, per quantity in STEADY.
    Returns ({clock: {what: ms}}, {clock: n fault-free tokens})."""
    by = {}
    for _, _, d, o in token_lines(x):
        c = clock_of(d)
        if c is not None and float(d.get('majflt', 'inf')) == 0:
            by.setdefault(c, []).append((d, o))
    return {c: {w: statistics.median(tok_ms(d, o, w) for d, o in v) for w in STEADY} for c, v in by.items()}, {c: len(v) for c, v in by.items()}

def steady_at(S, c):
    """(S at clock c, flag). Exact logged clock if the story ran there; else the nearest logged clock by cpu4 MHz (ties: cpu7),
    scaled linearly in f as cycles: S(c0) × f0 / f on cpu4 MHz, flagged. ponytail: cpu4 (the A78 cluster) stands for the clock."""
    if c in S: return S[c], ''
    if not S or not c or c[0] <= 0: return None, 'no reference'
    c0 = min(S, key=lambda k: (abs(k[0] - c[0]), abs(k[1] - c[1]) if len(k) > 1 and len(c) > 1 else 0))
    r = c0[0] / c[0] if c0[0] > 0 else 1.0
    return {w: v * r for w, v in S[c0].items()}, f"scaled {'/'.join(map(str, c0))}->{'/'.join(map(str, c))}"

def print_official(runs):
    print('\n== official 32 decomposed (r16, R48; all numbers from the report; ms unless labelled)')
    fmt_c = lambda x: ','.join((x.get('mhz_after') or '').replace('mhz=', '').split(',')[4:]) or '-'
    plain = [x for x in runs if x['phase'] == 'on' and x.get('decode_tps') is not None and not x.get('error')]
    profd = [x for x in runs if x['phase'] == 'prof' and not x.get('error')]
    offs = [x for x in runs if x['phase'] in ('baseline', 'on', 'prof')]
    # line 1
    print(f"  LINE 1 (goal metric): printed decode tok/s over the {len(plain)} plain official runs (cold baseline excluded)")
    for i, x in enumerate(plain):
        print(f"    plain {i+1:2d} group {x.get('group') or '-'} pause {x.get('pause_s', '-')} s  {x['decode_tps']:5.2f} tok/s  cpu4-7 after {fmt_c(x)}  "
              f"majflt {x.get('majflt')}  ttft {x.get('ttft_ms') or 0:.0f}  idle before {x.get('idle_ms', '-')} ms (cool {x.get('cool_ms', '-')})")
    if plain:
        tps = [x['decode_tps'] for x in plain]
        med = statistics.median(tps)
        best = max(plain, key=lambda x: x['decode_tps'])
        mid = sorted(plain, key=lambda x: x['decode_tps'])
        mids = mid[(len(mid) - 1) // 2: len(mid) // 2 + 1]
        print(f"    median {med:.2f} tok/s (middle run{'s' if len(mids) > 1 else ''}: " +
              '; '.join(f"{x['decode_tps']:.2f} at {fmt_c(x)}, majflt {x.get('majflt')}" for x in mids) + ')')
        print(f"    best   {best['decode_tps']:.2f} tok/s at {fmt_c(best)}, majflt {best.get('majflt')}")
        print(f"    prediction median 18.0-20.0 (central 19.0); best ≥ 20.0 coin flip -> best {'≥' if best['decode_tps'] >= 20 else '<'} 20.0")
        print(f"    20 reached on the owner's benchmark (median ≥ 20.0): {'YES' if med >= 20.0 else 'no'}" + ('' if len(plain) == 10 else f'  (NOTE: {len(plain)} plain runs, 10 planned)'))
    # validity: text
    texts = {x.get('text') for x in offs if x.get('text') is not None}
    text_ok = len(texts) == 1
    print(f"  VALIDITY text: greedy text identical across all {len(offs)} official runs: {'yes' if text_ok else 'NO -> SESSION VOID'}"
          + ('' if text_ok else f" ({len(texts)} distinct texts)"))
    # validity: profiling overhead per group
    ov = []
    for g in sorted({x.get('group') for x in runs if x.get('group')}):
        p_ = [x['decode_tps'] for x in plain if x.get('group') == g]
        q_ = [x['decode_tps'] for x in profd if x.get('group') == g and x.get('decode_tps') is not None]
        if p_ and q_:
            o_ = 1 - statistics.mean(q_) / statistics.mean(p_)
            ov.append(o_)
            print(f"  VALIDITY profiling overhead group {g}: plain {statistics.mean(p_):.2f} profiled {statistics.mean(q_):.2f} tok/s -> {o_ * 100:+.1f} %"
                  f"{'  (> 5 %)' if o_ > 0.05 else ''}")
    if ov:
        m_ = statistics.mean(ov)
        print(f"    mean over {len(ov)} groups {m_ * 100:+.1f} % -> {'FLAGGED (> 5 %): line 2 stands relative to S but is flagged' if m_ > 0.05 else 'ok (≤ 5 %)'}")
    # steady reference
    ref = [x for x in runs if x['phase'] == 'ref']
    S, nS = steady_ref(ref[0]) if ref else ({}, {})
    modal = max(nS, key=nS.get) if nS else None
    ok_ref = bool(modal) and nS[modal] >= 100
    print(f"  VALIDITY steady reference: fault-free reference-story tokens per logged clock: " +
          (', '.join(f"{'/'.join(map(str, c))} {n}" for c, n in sorted(nS.items(), key=lambda t: -t[1])) or 'none') +
          f" -> modal {'/'.join(map(str, modal)) if modal else '-'} {nS.get(modal, 0)} {'≥' if ok_ref else '<'} 100: " +
          ('ok' if ok_ref else 'LINE 2 UNSCORABLE'))
    for c in sorted(S, key=lambda c: -nS[c]):
        print(f"    S({'/'.join(map(str, c))}) token {S[c]['token']:.2f}  MUL_MAT {S[c]['MUL_MAT']:.2f}  MUL_MAT_ID {S[c]['MUL_MAT_ID']:.2f}  (n {nS[c]})")
    # line 2
    print(f"  LINE 2 (decomposition{'' if ok_ref else ', UNSCORABLE: reported only'}): E = Σ over decode tokens tok=2..32 of (token ms − S(f_token)); "
          "E_fault = tokens with majflt > 0, E_clean = majflt 0; op excess = Σ (op ms − S_op(f)), rest = E − MUL_MAT − MUL_MAT_ID")
    res = []
    for i, x in enumerate(profd):
        toks = sorted(token_lines(x), key=lambda t: t[1])
        E = Ef = Ec = 0.0; op = {'MUL_MAT': 0.0, 'MUL_MAT_ID': 0.0}; opc = {'MUL_MAT': 0.0, 'MUL_MAT_ID': 0.0}; flags = set(); n = 0
        for _, k, d, o in toks:
            if not 2 <= k <= 32: continue
            s_, fl = steady_at(S, clock_of(d))
            if fl: flags.add(fl)
            if s_ is None: continue
            n += 1
            e = tok_ms(d, o, 'token') - s_['token']
            E += e
            clean = float(d.get('majflt', 'inf')) == 0
            if clean: Ec += e
            else: Ef += e
            for w in op:
                v = tok_ms(d, o, w) - s_[w]
                op[w] += v
                if clean: opc[w] += v
        ramp, fault = E > 0 and Ec >= 0.5 * E, E > 0 and Ef > 0.5 * E  # R48 am. 1 (4): E ≤ 0 is neither
        res.append((x, E, Ef, Ec, ramp, fault))
        share = Ec / E if E else float('nan')
        print(f"    prof {i+1} group {x.get('group') or '-'} pause {x.get('pause_s', '-')} s: tokens {n}  E {E:+8.1f}  E_fault {Ef:+8.1f}  E_clean {Ec:+8.1f}  "
              f"E_clean share {share:.2f}{'  (E ≤ 0)' if E <= 0 else ''}  -> {'ramp' if ramp else ''}{'+' if ramp and fault else ''}{'fault' if fault else ''}{'' if ramp or fault else 'neither'}{' (E ≤ 0)' if E <= 0 else ''}"
              + (f"  FLAG {'; '.join(sorted(flags))}" if flags else ''))
        print(f"           by op, all tokens: MUL_MAT {op['MUL_MAT']:+.1f}  MUL_MAT_ID {op['MUL_MAT_ID']:+.1f}  rest {E - op['MUL_MAT'] - op['MUL_MAT_ID']:+.1f};  "
              f"clean tokens: MUL_MAT {opc['MUL_MAT']:+.1f} ({opc['MUL_MAT'] / Ec * 100 if Ec else float('nan'):.0f} % of E_clean)  "
              f"MUL_MAT_ID {opc['MUL_MAT_ID']:+.1f}  rest {Ec - opc['MUL_MAT'] - opc['MUL_MAT_ID']:+.1f}")
    if res:
        nr, nf = sum(r[4] for r in res), sum(r[5] for r in res)
        neg = sum(r[1] <= 0 for r in res)
        cls = 'NO GAP' if neg >= 5 else 'RAMP-DOMINATED' if nr >= 6 else 'FAULT-DOMINATED' if nf >= 6 else 'MIXED'
        both = ' (both rules met: listed first match)' if neg < 5 and nr >= 6 and nf >= 6 else ''
        print(f"    classifier (R48 + am. 1 (4)): E ≤ 0 in {neg}/{len(res)} (≥ 5 -> NO GAP), E_clean ≥ 0.5 E > 0 in {nr}/{len(res)}, "
              f"E_fault > 0.5 E > 0 in {nf}/{len(res)} (≥ 6 of 8) -> {cls}{both}"
              + ('' if len(res) == 8 else f'  (NOTE: {len(res)} profiled runs, 8 planned)'))
        # R48 am. 1 (2): the pause contrast is withdrawn as a contrast; E against idle_ms is a reported fact line
        xy = [(r[0]['idle_ms'] / 1000, r[1]) for r in res if r[0].get('idle_ms') is not None]
        print('    E vs idle before the run (reported, not scored; idle_ms = previous reply end -> this run start):')
        for r in res:
            print(f"      group {r[0].get('group') or '-'} pause {r[0].get('pause_s', '-')} s  idle {r[0].get('idle_ms', '-')} ms (cool {r[0].get('cool_ms', '-')})  E {r[1]:+.1f}")
        if len(xy) > 2 and len({a for a, _ in xy}) > 1:
            mx, my = statistics.mean(a for a, _ in xy), statistics.mean(b for _, b in xy)
            sxx = sum((a - mx) ** 2 for a, _ in xy)
            sl = sum((a - mx) * (b - my) for a, b in xy) / sxx
            ssr = sum((b - my - sl * (a - mx)) ** 2 for a, b in xy)
            print(f"      slope {sl:+.2f} ± {(ssr / (len(xy) - 2) / sxx) ** 0.5:.2f} ms of E per s idle (least squares, n {len(xy)}; reported, not scored)")
        else:
            print('      slope: fewer than 3 runs with distinct idle_ms')
        aft = [r[1] for r in res if r[0].get('group') in (2, 4)]
        b2b = [r[1] for r in res if r[0].get('group') in (1, 3)]
        if aft and b2b:
            print(f"    mean E groups 2, 4 (after the 20 s pause) {statistics.mean(aft):+.1f} (n {len(aft)}); groups 1, 3 {statistics.mean(b2b):+.1f} "
                  f"(n {len(b2b)}) (reported, not scored)")
    # reported, not scored
    print('  REPORTED (not scored): per profiled run prefill majflt = run majflt − Σ decode-token majflt; outside-graph = (duration − ttft − '
          'Σ us/token over all its token lines) / (tokens − 1); token-1 = the tok=1 line')
    for i, x in enumerate(profd):
        toks = sorted(token_lines(x), key=lambda t: t[1])
        dm = sum(float(d.get('majflt', 0)) for _, _, d, _ in toks)
        tin = sum(int(d['us/token']) for _, _, d, _ in toks) / 1000
        dur, ttft, n = x.get('duration_ms'), x.get('ttft_ms'), x.get('tokens') or 0
        og = (dur - ttft - tin) / (n - 1) if None not in (dur, ttft) and n > 1 else float('nan')
        t1 = next((int(d['us/token']) / 1000 for _, k, d, _ in toks if k == 1), float('nan'))
        mj = x.get('majflt')
        print(f"    prof {i+1} group {x.get('group') or '-'}: prefill majflt {mj - dm if mj is not None else float('nan'):.0f} (run {mj}, decode {dm:.0f})  "
              f"ttft {ttft or 0:.0f}  outside-graph {og:.2f} ms/token  token-1 {t1:.2f}  idle before {x.get('idle_ms', '-')} ms (cool {x.get('cool_ms', '-')})")
        ms = ' '.join(f"{int(d['us/token']) / 1000:.1f}" for _, _, d, _ in toks)
        print(f"      ms     {ms}")
        print(f"      majflt {' '.join(d.get('majflt', '?') for _, _, d, _ in toks)}")
        print(f"      minflt {' '.join(d.get('minflt', '?') for _, _, d, _ in toks)}")
        print(f"      mhz    {' '.join(d.get('mhz', '?') for _, _, d, _ in toks)}")
    for x in offs:
        if x['phase'] != 'prof':
            print(f"    {x['phase']:8s} group {x.get('group') or '-'}: ttft {x.get('ttft_ms') or 0:.0f}  majflt {x.get('majflt')}  decode_ms {x.get('decode_ms')}")

def main():
    path = sys.argv[1]; want_csv = '--csv' in sys.argv
    gate = float(sys.argv[sys.argv.index('--gate') + 1]) if '--gate' in sys.argv else None
    pairs = '--pairs' in sys.argv
    global SIGMA
    SIGMA = float(sys.argv[sys.argv.index('--sigma') + 1]) if '--sigma' in sys.argv else 1.0
    r = json.load(open(path))
    print(f"kit {r.get('kit')} app {r.get('app')} started {r.get('started')} finished {r.get('finished')} stopped_early={r.get('stopped_early')}")
    for h in r.get('header', []): print('  ' + h)
    print(f"  saved {r.get('saved')}\n  thp: {r.get('thp')}\n  mem start {r.get('mem_start')}\n  mem end   {r.get('mem_end')}")
    for n in r.get('notes', []): print('  NOTE ' + n)
    runs = r.get('runs', [])
    rows = []
    print('\n== runs (measured on device)')
    print(f"{'phase':9s} {'name':38s} {'thr':>3s} {'heat':>5s} {'tok/s':>6s} {'pp':>5s} {'ttft':>6s} {'majf':>5s} {'minf':>6s} {'ioMB':>6s} {'mhz(after)':20s} err")
    for x in runs:
        tps = x.get('decode_tps'); pp = x.get('prefill_tps')
        io = x.get('io_read_bytes'); io = '' if io is None else f"{io/1e6:.1f}"
        mhz = (x.get('mhz_after') or '').replace('mhz=', '')[:20]
        heat = x.get('heat'); ttft = x.get('ttft_ms')
        print(f"{x['phase']:9s} {x['name'][:38]:38s} {str(x.get('threads','')):>3s} {'' if heat is None else f'{heat:.2f}':>5s} "
              f"{'' if tps is None else f'{tps:.1f}':>6s} {'' if pp is None else f'{pp:.0f}':>5s} {'' if ttft is None else f'{ttft:.0f}':>6s} "
              f"{str(x.get('majflt','')):>5s} {str(x.get('minflt','')):>6s} {io:>6s} {mhz:20s} {x.get('error','')[:60]}")
        if x.get('spec'):
            s = x['spec']
            acc = s.get('accepted') or 0; att = s.get('attempts') or 0; dr = s.get('draft_tokens') or 0
            print(f"          spec: drafted {dr} accepted {acc} attempts {att} -> tokens/eval ≈ {(acc + att)/att if att else float('nan'):.2f}; "
                  f"draft {s.get('draft_ms')} ms, verify {s.get('verify_ms')} ms, replay {s.get('replay_tokens')}")
        for line in x.get('prof', []):
            if line.startswith('cpu-bench'):
                print('          ' + line)
        if tps is not None and not x.get('error'):
            rows.append((x['phase'], x['name'], x.get('threads'), x.get('heat'), tps, pp, x.get('majflt'), x.get('minflt'), io))
    print('\n== A/B pairs (per-token profile, µs; B − A)')
    abba = {}
    inter = {}
    for x in runs:
        if x['phase'] != 'ab': continue
        if any(' tok=' in l for l in x.get('prof', [])):  # r15: per-token interleaved run -> --pairs
            inter.setdefault(x['name'], []).append(x); continue
        if any(' blk=' in l for l in x.get('prof', [])):
            abba.setdefault(x['name'], []).append(x); continue
        a = b = None
        for line in x.get('prof', []):
            if 'ab=A' in line: a = line
            if 'ab=B' in line: b = line
        if not (a and b):
            print(f"  {x['name']}: no pair ({x.get('error','')})"); continue
        da, oa = prof_fields(a); db, ob = prof_fields(b)
        ua, ub = int(da['us/token']), int(db['us/token'])
        print(f"  {x['name']:28s} A {ua/1000:7.1f} ms  B {ub/1000:7.1f} ms  Δ {(ub-ua)/1000:+6.1f} ms ({(ub-ua)/ua*100:+.1f}%)  "
              f"wait A {int(da['wait'])/1000:.1f} B {int(db['wait'])/1000:.1f}  mmid_thr A {da.get('mmid_thr')} B {db.get('mmid_thr')} heat {x.get('heat')}")
        for op in ('MUL_MAT_ID', 'MUL_MAT'):
            if op in oa and op in ob:
                print(f"      {op}: compute {oa[op][1]/1000:.1f} → {ob[op][1]/1000:.1f} ms, wait {oa[op][2]/1000:.1f} → {ob[op][2]/1000:.1f} ms")
    if abba:
        print('\n== ABBA A/Bs (r5: per-block lines, warm-up blk=0 dropped; ms/token; Δ = B − A ± 1σ; runs of one name pooled)')
        for name, xs in abba.items():
            print(f"  {name}  ({len(xs)} run{'s' if len(xs) > 1 else ''})")
            for title, rs in ([(f'run {i+1}', [x]) for i, x in enumerate(xs)] if len(xs) > 1 else []) + [('pooled' if len(xs) > 1 else 'run', xs)]:
                for label, line in abba_summary(rs):
                    print(f"    {title:7s} {label:12s} {line}")
            for x in xs:
                mb = x.get('mhz_blocks')
                if mb: print(f"    mhz cpu4/cpu7 every 8 tokens: {' '.join(f'{a}/{b}' for a, b in mb)}")
        if gate is not None:
            print_gate(abba, gate)
    if inter:
        if pairs:
            print_pairs(inter, abba, gate if gate is not None else 10)
        else:
            print(f"\n== {sum(len(v) for v in inter.values())} per-token interleaved runs ({', '.join(inter)}): run with --pairs")
    print('\n== r3 fields per run (last cpu-prof line; faults per generated token)')
    for x in runs:
        lines = [l for l in x.get('prof', []) if l.startswith('cpu-prof') and ' tok=' not in l]
        if not lines: continue
        tok = x.get('tokens') or 0
        for line in lines[-2:] if x['phase'] == 'ab' else lines[-1:]:
            d, o = prof_fields(line)
            mmid = o.get('MUL_MAT_ID', (0, 0, 0))
            print(f"  {x['phase']:8s} {x['name'][:30]:30s} {d.get('ab','  ')} eio={d.get('eio')} adv={d.get('adv')} prof majflt={d.get('majflt')} minflt={d.get('minflt')} head={d.get('head')} chunk={d.get('chunk')} expq4={d.get('expq4')} kern={d.get('kern')} anon={d.get('anon')} anon_mb={d.get('anon_mb')} kavg={d.get('kavg')} "
                  f"ilp={d.get('ilp')} pf={d.get('pf')} us/token={d.get('us/token')} MUL_MAT_ID {mmid[1]/1000:.1f}+{mmid[2]/1000:.1f} ms  "
                  f"majflt/tok={(x.get('majflt') or 0)/tok if tok else float('nan'):.1f} minflt/tok={(x.get('minflt') or 0)/tok if tok else float('nan'):.0f}")
    bench = [l for x in runs for l in x.get('prof', []) if l.startswith('cpu-bench q3k ')]
    if bench:
        print('\n== cpu-bench (r10/r11: 1-thread kernel timing, cache-hot synthetic expert slices, ns/row; device-measured)')
        for l in bench:
            print('  ' + l)
            print('    ' + bench_ratios(l))
    print('\n== copies, warm-ups, willneed (r9: requant lines; a warm-up runs before a run whose copies differ from the last)')
    for x in runs:
        lines = [l for l in x.get('warmup', []) + x.get('prof', []) if ' requant ' in l or l.startswith('cpu-prof willneed')]
        if lines or x.get('warmup_ms') is not None:
            print(f"  {x['phase']:8s} {x['name'][:40]:40s} warmup_ms={x.get('warmup_ms')}")
            for l in lines: print('      ' + l)
    print('\n== outside the graph (story runs with 6 windows; ms per printed token; decode wall = duration − ttft)')
    for x in runs:
        o = outside_graph(x)
        if o:
            sm = x.get('sample_ms'); tok = x.get('tokens') or 0
            smp = f"; llamadart sample {sm / tok:.2f} ms/token" if sm is not None and tok else ''
            print(f"  {x['phase']:8s} {x['name'][:40]:40s} outside {o[0]:5.2f} ms/token  (decode wall {o[1]:.1f} − windows {o[2]:.1f} − {o[3]} × steady {o[4]:.1f}){smp}")
    print('\n== reply text (first 80 chars; exact knobs must print the text of the same config without them; r9 sample_ms = llamadart sampling + piece time)')
    for x in runs:
        if x.get('text') is not None:
            sm = x.get('sample_ms'); tok = x.get('tokens') or 0
            print(f"  {x['phase']:8s} {x['name'][:52]:52s} {'' if sm is None or not tok else f'sample {sm / tok:.2f} ms/tok ':20s}{x['text']!r}"
                  + ('' if 'text_same' not in x else f"  whole reply = F32 reference: {'same' if x['text_same'] else 'DIFFERENT'}"
                         + (f" from char {x['text_diverge_at']} (lengths {x.get('text_len')})" if 'text_diverge_at' in x else '')
                         + ('' if 'text_same' not in x else '  [voiding check]' if x['phase'] == 'check' else '  [reported]')))
    if 'router_text_same' in r:  # r15 (R46 item 2, amendment 1 (1)): only the full-F16 official 32 voids; story divergence is reported
        print(f"  C1-F16 greedy-text identity (official 32 under full F16 vs the baseline): {'same' if r['router_text_same'] else 'DIFFERENT -> C1 runs VOID'}")
    print('\n== best per setting (phases threads/baseline/release/spec)')
    best = {}
    for x in runs:
        if x.get('decode_tps') is None or x.get('error'): continue
        k = (x['phase'], x['name'])
        best[k] = max(best.get(k, 0), x['decode_tps'])
    for (ph, nm), v in best.items():
        print(f"  {ph:9s} {nm:40s} best {v:5.1f} tok/s")
    if '--official' in sys.argv:
        print_official(runs)
    if want_csv:
        print('\n== results.csv rows')
        for ph, nm, thr, heat, tps, pp, mj, mn, io in rows:
            print(f"app {r.get('app')} (research {r.get('kit')}),\"DEVICE-MEASURED {ph}: {nm} ({thr} threads)\",9.6051,+6.21,,"  # quoted: run names may hold commas (r11)
                  f"\"{tps:.1f} w / {'' if pp is None else f'{pp:.0f}'} r at {'' if heat is None else f'{heat*100:.0f}%'}; majflt {mj} minflt {mn} io {io} MB\",,,\"{r.get('started')}\"")

if __name__ == '__main__':
    main()
