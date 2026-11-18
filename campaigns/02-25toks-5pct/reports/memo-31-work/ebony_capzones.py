# Which thermal zone separates the clock-cap levels? Usage: python3 -I ebony_capzones.py <report.json>...
# For each zone type: range of the reading before a run, per cap_before level, pooled over the given reports;
# then a separation score = number of run pairs (full cap, lower cap) ordered correctly by the zone / all such pairs (AUC).
import json, sys, collections
rows = []
for f in sys.argv[1:]:
    d = json.load(open(f)); types = {z['zone']: z['type'] for z in d.get('thermal_zones', [])}
    for r in d['runs']:
        if not r.get('zones_before') or not r.get('cap_before'): continue
        cap = int(r['cap_before'][4:].split('/')[1])
        z = {types.get(k, k): v / 1000 for k, v in r['zones_before'].items()}
        z['headroom'] = r.get('heat') or 0; z['batt_c'] = r.get('batt_c_before') or 0
        rows.append((f[-18:-5], r['pos'], cap, z))
caps = sorted({c for _, _, c, _ in rows}); print('cap levels (policy7 MHz):', {c: sum(1 for r in rows if r[2] == c) for c in caps})
out = []
for k in sorted(rows[0][3]):
    hi = [r[3].get(k) for r in rows if r[2] >= 2841 and r[3].get(k) is not None]; lo = [r[3].get(k) for r in rows if r[2] < 2841 and r[3].get(k) is not None]
    if not hi or not lo or max(hi + lo) < 5 and k != 'headroom': continue
    auc = sum((a < b) + 0.5 * (a == b) for a in hi for b in lo) / (len(hi) * len(lo))
    per = ' '.join('%d:%.1f..%.1f' % (c, min(v), max(v)) for c in caps for v in [[r[3][k] for r in rows if r[2] == c and k in r[3]]] if v)
    out.append((auc, k, per))
for auc, k, per in sorted(out, reverse=True)[:14]: print('%.3f %-18s %s' % (auc, k, per))
