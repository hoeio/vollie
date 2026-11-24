#!/usr/bin/env python3
"""memo-12 round 32 (Bonnie): clock caps around campaign 2's runs. Reads committed kit JSONs: per run cap_before and
cap_after ("cap=<A78>/<X1> MHz" = scaling_max_freq), heat (thermal headroom at the run's start), gate (None = not
gated), cool_ms (wait before the run), wall_ms, ttft_ms (prefill). *device* records; nothing is run.
Run: python3 -I -B bonnie_caps_r32.py <repo root>"""
import sys, os, json, glob, collections
root = sys.argv[1]; FULL = (2419, 2841)
def cap(s):
    try: a, b = s.split("=")[1].split("/"); return int(a), int(b)
    except Exception: return None
runs = []
for f in sorted(glob.glob(os.path.join(root, "campaigns/02-25toks-5pct/reports/device/*.json"))):
    try: d = json.load(open(f))
    except ValueError: continue
    if not isinstance(d, dict) or not isinstance(d.get("runs"), list): continue
    prev = None
    for x in d["runs"]:
        if not isinstance(x, dict) or cap(x.get("cap_before")) is None or cap(x.get("cap_after")) is None: prev = None; continue
        r = dict(f=os.path.basename(f), b=cap(x["cap_before"]), a=cap(x["cap_after"]), heat=x.get("heat"), gated=x.get("gate") is not None, cool=(x.get("cool_ms") or 0) / 1000,
                 wall=(x.get("wall_ms") or 0) / 1000, ttft=(x.get("ttft_ms") or 0) / 1000, model=x.get("model"), threads=x.get("threads"), prev=prev)
        runs.append(r); prev = r
# power saving holds the caps at 2,035 for a whole session (r22 to r26, r28-c1): those sessions are set aside, the claim rule has it off
mode = {f: collections.Counter(r["b"] for r in runs if r["f"] == f).most_common(1)[0][0] for f in {r["f"] for r in runs}}
print("sessions set aside (their usual cap is not the maximum; power saving):", sorted(f[:12] for f, m in mode.items() if m != FULL), "runs", sum(mode[r["f"]] != FULL for r in runs))
runs = [r for r in runs if mode[r["f"]] == FULL]
def share(sel, f): sel = list(sel); return f"{sum(map(f, sel))} of {len(sel)}" + (f" ({sum(map(f, sel)) / len(sel):.2f})" if sel else "")
low = lambda c: c[0] < FULL[0]
print("runs with both caps:", len(runs), "in", len({r['f'] for r in runs}), "files")
print("cap at the start, by gate:   gated: below maximum", share((r for r in runs if r["gated"]), lambda r: low(r["b"])), "| not gated:", share((r for r in runs if not r["gated"]), lambda r: low(r["b"])))
print("cap levels seen (A78 MHz) at start:", dict(collections.Counter(r["b"][0] for r in runs).most_common()))
print("\ncap at the start by thermal headroom at the start (all runs):")
for lo, hi in ((0, 0.6), (0.6, 0.65), (0.65, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 2)):
    s = [r for r in runs if r["heat"] is not None and lo <= r["heat"] < hi]
    print(f"  headroom {lo:.2f}-{hi:.2f}: below maximum {share(s, lambda r: low(r['b']))}; levels {dict(collections.Counter(r['b'][0] for r in s).most_common(4))}")
print("\na run that starts at the maximum: is the cap lower when it ends? by run length (prefill + decode, s):")
for lo, hi in ((0, 5), (5, 6), (6, 8), (8, 12), (12, 1e9)):
    s = [r for r in runs if r["b"] == FULL and lo <= r["wall"] < hi]
    print(f"  {lo}-{hi if hi < 1e8 else 'more'} s: lowered during the run {share(s, lambda r: low(r['a']))}; median headroom at start {sorted(r['heat'] for r in s if r['heat'] is not None)[len(s) // 2] if s else None}")
print("  by headroom at the start:")
for lo, hi in ((0, 0.6), (0.6, 0.65), (0.65, 0.7), (0.7, 2)):
    s = [r for r in runs if r["b"] == FULL and r["heat"] is not None and lo <= r["heat"] < hi]; print(f"    {lo:.2f}-{hi:.2f}: lowered during the run {share(s, lambda r: low(r['a']))}")
print("\nafter a run that ended below the maximum: the next run's cap at its start, by the wait before it (s):")
for lo, hi in ((0, 1), (1, 30), (30, 60), (60, 100), (100, 1e9)):
    s = [r for r in runs if r["prev"] is not None and low(r["prev"]["a"]) and lo <= r["cool"] < hi]
    print(f"  wait {lo}-{hi if hi < 1e8 else 'more'}: back at the maximum {share(s, lambda r: not low(r['b']))}")
print("\nconsecutive ungated runs (no wait): cap at the start of run n of a chain that began at the maximum")
ch = collections.defaultdict(list)
for r in runs:
    n, q = 0, r
    while q["prev"] is not None and not q["gated"] and q["cool"] < 1: q = q["prev"]; n += 1
    if q["b"] == FULL or n == 0: ch[min(n, 6)].append(r)
for n in sorted(ch):
    s = [r for r in ch[n] if not r["gated"] and r["cool"] < 1]
    if s: print(f"  run {n} of a chain: below maximum at start {share(s, lambda r: low(r['b']))}, at end {share(s, lambda r: low(r['a']))}; cumulative load before it about {sum(r['wall'] for r in s) / len(s) * n:.0f} s")
