"""SD888 decode estimate (tok/s range) for a config — an ESTIMATE, never a measurement.

Anchor: the user's own Vollie phone test on the Z Flip 3 (2026-10-25, earlier session, not reproduced here):
Qwen3.5-4B Q4_0 (vocab-trimmed, repack on, 3-4 big threads) decoded 5.9 tok/s hot .. 6.6 cold.
Two independent scalings from that anchor, the estimate is their [min, max]:
  1. bandwidth:  anchor * traffic(anchor) / traffic(config), traffic = weight bytes + 2 x recurrent state
  2. host ratio: anchor * host_tg(config) / host_tg(anchor), same llama.cpp build, 4 threads and 1 thread
     (the X1/A78 are much slower than an M4 core, so the phone sits between the 4- and 1-thread host balance)
Method 1 knows nothing about repack/kernels, so for configs that are not fully repacked it is scaled by
the host's (config / same-weights-fully-repacked) tg ratio.
usage: python3 tools/sd888_estimate.py   (reads lab/logs/p/bench.csv)"""
import csv, os
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
ANCHOR = (5.9, 6.6); ANCHOR_MB = 2227 + 2 * 53  # tools/cost.py on vollie-qwen3.5-4b-q4_0-mtp.gguf
host = {}
for tag, t, rp, test, ts, sd in csv.reader(open(f'{ROOT}/lab/logs/p/bench.csv')):
    if test == 'tg64': host[(tag, int(t), int(rp))] = float(ts)

# name: (bench tag, repack flag, weight MB/token, state MB, fully-repacked reference (tag, rp) for method 1)
CONFIGS = {
    'shipped A e4, repack off':  ('A-e4', 0, 776, 58, ('A-e4', 1)),
    'A e4, repack on (OOM 8GB)': ('A-e4', 1, 776, 58, ('A-e4', 1)),
    'A e5, repack off':          ('A-e5', 0, 829, 58, ('A-e5', 1)),
    'A e6, repack off':          ('A-e6', 0, 882, 58, ('A-e6', 1)),
    'D1 e5 (dense repack)':      ('D1-e5', 1, 813, 58, ('A-e5', 1)),
    'D1 e4 (dense repack)':      ('D1-e4', 1, 763, 58, ('A-e4', 1)),
}

def estimate(tag, rp, mb, state, ref):
    lo, hi = [], []
    bw = [a * ANCHOR_MB / (mb + 2 * state) for a in ANCHOR]
    for t in (4, 1):
        k = host.get((tag, t, rp)) / host.get((*ref[:1], t, ref[1])) if (ref[0], t, ref[1]) in host and (tag, t, rp) in host else None
        if k: lo.append(bw[0] * k); hi.append(bw[1] * k)
        if ('B', t, 1) in host and (tag, t, rp) in host:
            r = host[(tag, t, rp)] / host[('B', t, 1)]; lo.append(ANCHOR[0] * r); hi.append(ANCHOR[1] * r)
    return (min(lo), max(hi)) if lo else None

if __name__ == '__main__':
    for name, (tag, rp, mb, st, ref) in CONFIGS.items():
        e = estimate(tag, rp, mb, st, ref)
        print(f'{name:28s} ' + (f'{e[0]:5.1f} - {e[1]:5.1f} tok/s (estimated)' if e else 'missing host numbers'))
