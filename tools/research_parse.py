"""Turns a research-kit JSON report (tools/research_kit.md) into results.csv rows and R-log text.
usage: python3 -I tools/research_parse.py <report.json> [--csv]
Prints a human table per phase (device-measured numbers), the A/B splits from the cpu-prof pairs, the
barrier cost, the speculation stats, and, with --csv, the rows to append to reports/results.csv."""
import json, re, sys, statistics

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
    """r10: the pre-registered ratios of one 'cpu-bench q3k' line (§9.9 (iii)); r11 adds the core clock (mhz=start/end)."""
    d = dict(re.findall(r'(\w+)=([^\s]+)', line))
    try:
        o, n, i8, q4, sc = (float(d[k]) for k in ('old', 'new', 'int8', 'q4x4', 'scalar'))
    except (KeyError, ValueError):
        return 'ratios: missing fields'
    mhz = d.get('mhz')  # r11: core clock 'start/end' MHz; r10 lines have none
    clock = f"  cpu{d.get('cpu', '?')} at {mhz} MHz (start/end)" if mhz else f"  cpu{d.get('cpu', '?')} clock not logged (pre-r11)"
    return f"new/old {n/o:.3f}  int8/old {i8/o:.3f}  scalar/old {sc/o:.3f}  old/q4x4 {o/q4:.2f}" + clock

def mean_sd(v):
    return (statistics.mean(v), statistics.stdev(v) if len(v) > 1 else float('nan'))

def abba_summary(runs):
    """Per side mean ± sd over blocks, B − A ± 1σ (blocks iid), per ABBA group (B̄ − Ā) mean ± sd; ms/token."""
    metrics = [('us/token', lambda d, o: int(d['us/token'])), ('wait', lambda d, o: int(d['wait'])),
               ('MUL_MAT_ID', lambda d, o: sum(o.get('MUL_MAT_ID', (0, 0, 0))[1:])),
               ('MUL_MAT', lambda d, o: sum(o.get('MUL_MAT', (0, 0, 0))[1:])),
               ('head', lambda d, o: int(d.get('head', 0)))] + \
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
    for key in ('chunk', 'expq4', 'kern', 'fuse'):  # r9/r10 header fields, per side (absent before r9 / r10)
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

def main():
    path = sys.argv[1]; want_csv = '--csv' in sys.argv
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
    for x in runs:
        if x['phase'] != 'ab': continue
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
    print('\n== r3 fields per run (last cpu-prof line; faults per generated token)')
    for x in runs:
        lines = [l for l in x.get('prof', []) if l.startswith('cpu-prof')]
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
            print(f"  {x['phase']:8s} {x['name'][:52]:52s} {'' if sm is None or not tok else f'sample {sm / tok:.2f} ms/tok ':20s}{x['text']!r}")
    print('\n== best per setting (phases threads/baseline/release/spec)')
    best = {}
    for x in runs:
        if x.get('decode_tps') is None or x.get('error'): continue
        k = (x['phase'], x['name'])
        best[k] = max(best.get(k, 0), x['decode_tps'])
    for (ph, nm), v in best.items():
        print(f"  {ph:9s} {nm:40s} best {v:5.1f} tok/s")
    if want_csv:
        print('\n== results.csv rows')
        for ph, nm, thr, heat, tps, pp, mj, mn, io in rows:
            print(f"app {r.get('app')} (research {r.get('kit')}),\"DEVICE-MEASURED {ph}: {nm} ({thr} threads)\",9.6051,+6.21,,"  # quoted: run names may hold commas (r11)
                  f"\"{tps:.1f} w / {'' if pp is None else f'{pp:.0f}'} r at {'' if heat is None else f'{heat*100:.0f}%'}; majflt {mj} minflt {mn} io {io} MB\",,,\"{r.get('started')}\"")

if __name__ == '__main__':
    main()
