# Bonnie, 2026-11-18: decode wall ms/token against MB of wanted file pages missing at run start and VmSwap before.
# Filter: phase != warmup, cap_before not 1555/1670, one arm per model. Usage: python3 -I missfit.py <reports dir>
import json, re, sys
import numpy as np
D = sys.argv[1].rstrip('/') + '/'
for f in ('r25_2026-11-16.json', 'r26_2026-11-18.json'):
    d = json.load(open(D + f))
    for m, arm in (('granite', 'num'), ('olmoe', 'off')):
        X, Y, S = [], [], []
        for r in d['runs']:
            if r['model'] != m or r['arm'] != arm or r['phase'] == 'warmup' or re.search('1555|1670', r['cap_before']):
                continue
            X.append(r['want_mb'] - r['res_before']['total_mb']); Y.append(r['decode_ms'] / 32)
            S.append(int(re.search(r'swap=(\d+)MB', r['mem_before']).group(1)))
        X, Y, S = np.array(X, float), np.array(Y), np.array(S, float)
        A = np.column_stack([np.ones(len(X)), X, S]); b = np.linalg.lstsq(A, Y, rcond=None)[0]
        res = Y - A @ b; s2 = (res ** 2).sum() / (len(Y) - 3); se = np.sqrt(np.diag(s2 * np.linalg.inv(A.T @ A)))
        print(f[:3], m, arm, 'n', len(X), 'missing median %.0f' % np.median(X),
              'wall = %.1f±%.1f + %.4f±%.4f*missMB + %.4f±%.4f*swapMB' % (b[0], se[0], b[1], se[1], b[2], se[2]),
              'r(miss) %.2f r(swap) %.2f' % (np.corrcoef(X, Y)[0, 1], np.corrcoef(S, Y)[0, 1]))
