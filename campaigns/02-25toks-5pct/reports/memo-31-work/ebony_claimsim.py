# P(median of 10 runs >= 25.0) by resampling R67's OLMoE scored runs (device, r27), and by a two-part model.
# Usage: python3 -I ebony_claimsim.py   (no arguments; the tok/s lists are copied from r27_2026-11-18.json, see ebony_runs.py)
import random, statistics as st
random.seed(31)
P  = [17.63, 23.18, 23.67, 24.98, 25.07, 25.37, 25.48, 25.56]            # pin on (Px + PX), positions 15,18,33,32,27,20,21,30
PX = [17.63, 24.98, 25.48, 25.56]
late = [25.02, 25.37, 25.48, 18.24, 25.07, 24.32, 25.21, 25.56, 25.08, 24.98, 23.67, 19.58]  # all cells, positions 19-22 and 27-34
clean = [25.37, 25.48, 25.56, 24.98, 25.07, 25.02, 25.21, 25.08]           # runs with decode excess <= 25 ms over the best
def p(pool, n=10, it=200000): return sum(st.median(random.choices(pool, k=n)) >= 25.0 for _ in range(it)) / it
print('resample 10 of the 8 pin-on runs      P(pass) = %.2f' % p(P))
print('resample 10 of the 4 pin+exec runs    P(pass) = %.2f' % p(PX))
print('resample 10 of positions 19-22, 27-34 P(pass) = %.2f' % p(late))
print('resample 10 of the 8 clean runs       P(pass) = %.2f  (no tail at all, today\'s clean level)' % p(clean))
# two-part model: a run is "tail" with probability q (then < 25); else clean level shifted by d ms per token
def model(q, d, it=100000):
    ok = 0
    for _ in range(it):
        xs = []
        for _ in range(10):
            if random.random() < q: xs.append(20.0)
            else:
                t = 965.6 / random.choice(clean) - d      # ms per token (tok/s = 965.6 / ms, memo-29 conversion)
                xs.append(965.6 / t)
        ok += st.median(xs) >= 25.0
    return ok / it
print('tail rate q \\ clean margin d (ms/token):   0.0   0.3   0.5   1.0')
for q in (0.5, 0.375, 0.25, 0.125, 0.0): print('  q=%.3f' % q, '  '.join('%.2f' % model(q, d) for d in (0.0, 0.3, 0.5, 1.0)))
