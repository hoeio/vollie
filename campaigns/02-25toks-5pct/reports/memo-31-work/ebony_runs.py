# Per-run table of a kit report (r27 layout). Usage: python3 -I ebony_runs.py <report.json> [model]
import json, re, sys
d = json.load(open(sys.argv[1])); model = sys.argv[2] if len(sys.argv) > 2 else 'olmoe'
def kv(s, k):
    m = re.search(k + r'=(\S+)', s or ''); return m.group(1) if m else None
def line(x, name):
    for l in x.get('c2') or []:
        if l.startswith(name + ' '): return l
    return ''
print('pos id tps ttft wait_s heat heat_after cap_b cap_a mhz_b mhz_a res_b want lost miss dmaj dmin pmaj cached swapfree x7share a55 rq_ms ivcs sys_ms batt')
for x in d['runs']:
    if x['model'] != model or not x.get('decode_tps'): continue
    io = line(x, 'c2-io'); mj = [int(v) for v in (kv(io, 'majflt') or '0/0/0').split('/')]; mn = [int(v) for v in (kv(io, 'minflt') or '0/0/0').split('/')]
    pool = line(x, 'c2-pool'); cores = {}
    for w in re.findall(r'w\d=\S+', pool):
        for c, n in re.findall(r'c(\d)=(\d+)', w): cores[int(c)] = cores.get(int(c), 0) + int(n)
    tot = sum(cores.values()) or 1
    ts = [c for s in (x.get('thread_samples') or []) for c in s[2:]]
    a55 = sum(1 for c in ts if c < 4)
    sch = [l for l in x.get('c2') or [] if 'c2-sched' in l and 'phase=decode' in l]
    rq = sum(int(a.split(':')[1]) for a in re.findall(r'w\d=(\S+)', sch[0])) / 1000 if sch else -1
    iv = sum(int(a.split(':')[3]) for a in re.findall(r'w\d=(\S+)', sch[0])) if sch else -1
    ru = x.get('ru') or {}; sysms = (ru['end'][2] - ru['first'][2]) / 1000 if ru.get('end') else -1
    want = x.get('want_mb'); rb = (x.get('res_before') or {}).get('total_mb')
    print(x['pos'], x['id'].replace(model + '_', ''), '%.2f' % x['decode_tps'], '%.0f' % x['ttft_ms'], '%.0f' % ((x.get('cool_ms') or 0) / 1000),
          '%.3f' % (x.get('heat') or 0), '%.3f' % (x.get('heat_after') or 0), (x.get('cap_before') or '')[4:], (x.get('cap_after') or '')[4:],
          ','.join((x.get('mhz_before') or '').split(',')[4::3]), ','.join((x.get('mhz_after') or '').split(',')[4::3]),
          rb, want, x.get('wait_lost_mb'), (want - rb) if want and rb is not None else None, mj[2] - mj[1], mn[2] - mn[1], mj[1] - mj[0],
          kv(x.get('mem_before'), 'cached'), kv(x.get('mem_before'), 'swapfree'), '%.2f' % (cores.get(7, 0) / tot), '%d/%d' % (a55, len(ts)), '%.0f' % rq, iv, '%.0f' % sysms, x.get('batt_c_before'))
