# Tail decomposition per scored/profiled run (r27 layout). Usage: python3 -I ebony_tail.py <report.json> <model> <clean_decode_ms>
# excess = decode wall - clean; rq = sum of the three workers' run-queue wait in the decode (c2-sched); sys = process system
# CPU first chunk -> end (ru); other = process CPU (ru) - workers' on-cpu time (c2-sched): CPU used by the app's other threads.
import json, re, sys
d = json.load(open(sys.argv[1])); model = sys.argv[2]; clean = float(sys.argv[3])
print('pos cell tps decode_ms excess rq_w0/w1/w2 rq_sum ivcs_w0/w1/w2 vcs sys_ms oncpu_ms proc_cpu_ms other_ms nvcsw_proc nivcsw_proc dmaj dmin miss wait_s swapfree w0cores')
for x in d['runs']:
    if x['model'] != model or x['phase'] == 'warmup': continue
    sch = [l for l in x['c2'] if 'c2-sched' in l and 'phase=decode' in l]
    if not sch: continue
    w = [[int(v) for v in a.split(':')] for a in re.findall(r'w\d=(\S+)', sch[0])]
    ru = x['ru']; f, e = ru['first'], ru['end']
    io = [l for l in x['c2'] if l.startswith('c2-io ')][0]
    mj = [int(v) for v in re.search(r'majflt=(\S+)', io).group(1).split('/')]; mn = [int(v) for v in re.search(r'minflt=(\S+)', io).group(1).split('/')]
    us = [int(v) for v in re.search(r' us=(\S+)', io).group(1).split('/')]; dec = (us[2] - us[1]) / 1000
    pool = [l for l in x['c2'] if l.startswith('c2-pool')]; w0 = re.search(r'w0=\S+?:(c\S+)', pool[0]).group(1) if pool else '-'
    on = sum(a[0] for a in w) / 1000; proc = (e[0] - f[0]) / 1000
    sf = re.search(r'swapfree=(\d+)', x['mem_before']).group(1)
    want = x.get('want_mb') or 0; rb = x['res_before']['total_mb']
    print(x['pos'], x['id'].split('_r16')[-1].lstrip('_') or 'plain', '%.2f' % x['decode_tps'], '%.0f' % dec, '%+.0f' % (dec - clean),
          '/'.join('%.0f' % (a[1] / 1000) for a in w), '%.0f' % (sum(a[1] for a in w) / 1000), '/'.join(str(a[3]) for a in w), sum(a[2] for a in w),
          '%.0f' % ((e[2] - f[2]) / 1000), '%.0f' % on, '%.0f' % proc, '%+.0f' % (proc - on), e[3] - f[3], e[4] - f[4], mj[2] - mj[1], mn[2] - mn[1], want - rb, (x.get('cool_ms') or 0) // 1000, sf, w0)
