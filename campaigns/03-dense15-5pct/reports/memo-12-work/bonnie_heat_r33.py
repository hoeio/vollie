#!/usr/bin/env python3
"""memo-12 round 33 (Bonnie): (a) campaign 2's device prefill against its decode, per model and thread count;
(b) thermal headroom gained per second of load, from runs that record heat (headroom at the start) and heat_after.
Reads committed kit JSONs; power-saving sessions (usual cap 2,035) are set aside as in bonnie_caps_r32.py.
*device* records; the dense numbers in bonnie-joins-r33.md derived from them are *estimated*.
Run: python3 -I -B bonnie_heat_r33.py <repo root>"""
import sys, os, json, glob, collections
root = sys.argv[1]; FULL = "cap=2419/2841"
med = lambda v: sorted(v)[len(v) // 2] if v else float("nan")
q = lambda v, p: sorted(v)[min(len(v) - 1, int(p * len(v)))] if v else float("nan")
runs = []
for f in sorted(glob.glob(os.path.join(root, "campaigns/02-25toks-5pct/reports/device/*.json"))):
    try: d = json.load(open(f))
    except ValueError: continue
    if not isinstance(d, dict) or not isinstance(d.get("runs"), list): continue
    rs = [x for x in d["runs"] if isinstance(x, dict) and x.get("cap_before")]
    if not rs or collections.Counter(x["cap_before"] for x in rs).most_common(1)[0][0] != FULL: continue
    for x in rs: x["_f"] = os.path.basename(f); runs.append(x)
print("runs:", len(runs), "in", len({x['_f'] for x in runs}), "sessions without power saving")

print("\n(a) prefill against decode, official32 runs that start and end at the maximum cap, pool on, no major faults in the way (majflt < 50)")
g = collections.defaultdict(list)
for x in runs:
    if x.get("kind") == "official32" and x["cap_before"] == FULL and x.get("cap_after") == FULL and (x.get("env") or {}).get("VOLLIE_POOL") == "1" and x.get("ttft_ms") and x.get("decode_ms") and (x.get("majflt") or 0) < 50:
        g[(x["model"], x.get("arm"), x["threads"])].append((x["ttft_ms"] / x["prompt_tokens"], x["decode_ms"] / x["tokens"], x["prompt_tokens"], x["ttft_ms"] / 1000))
for k in sorted(g, key=str):
    v = g[k]; r = [a / b for a, b, _, _ in v]
    print(f"  {k}: runs {len(v)}, prompt tokens {v[0][2]}, prefill {med([t for *_, t in v]):.2f} s = {med([a for a, *_ in v]):.1f} ms per token; decode {med([b for _, b, *_ in v]):.1f} ms per token; prefill / decode per token {med(r):.2f} ({q(r, 0.1):.2f} to {q(r, 0.9):.2f})")

print("\n(b) headroom change over a run (heat_after - heat), runs that record both")
h = [x for x in runs if isinstance(x.get("heat"), (int, float)) and isinstance(x.get("heat_after"), (int, float)) and x.get("wall_ms")]
print("  runs:", len(h), "; gated among them:", sum(x.get("gate") is not None for x in h), "; kinds:", dict(collections.Counter(x.get("kind") for x in h)))
for key, lab in ((lambda x: (x.get("threads"),), "threads"), (lambda x: (x.get("threads"), x.get("model")), "threads, model")):
    gg = collections.defaultdict(list)
    for x in h: gg[key(x)].append(x)
    for k in sorted(gg, key=str):
        v = gg[k]; sl = [(x["heat_after"] - x["heat"]) / (x["wall_ms"] / 1000) for x in v]
        print(f"  {lab} {k}: runs {len(v)}, run length {med([x['wall_ms'] / 1000 for x in v]):.1f} s, start headroom {med([x['heat'] for x in v]):.2f}, change {med([x['heat_after'] - x['heat'] for x in v]):+.3f}, per second of load {med(sl):+.4f} ({q(sl, 0.1):+.4f} to {q(sl, 0.9):+.4f})")
print("  by start headroom:")
for lo, hi in ((0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 2)):
    v = [x for x in h if lo <= x["heat"] < hi]; sl = [(x["heat_after"] - x["heat"]) / (x["wall_ms"] / 1000) for x in v]
    print(f"    {lo:.1f}-{hi:.1f}: runs {len(v)}, per second {med(sl):+.4f} ({q(sl, 0.1):+.4f} to {q(sl, 0.9):+.4f}); cap lowered by the end in {sum(x.get('cap_after') != FULL for x in v)}")
print("  first run of an ungated pair against the second (same block, no wait):")
first = [x for x in h if x.get("rep") == 0 and x.get("phase") == "warmup"]; second = [x for x in h if x.get("rep") == 1 and x.get("phase") == "warmup"]
for lab, v in (("first", first), ("second", second)):
    sl = [(x["heat_after"] - x["heat"]) / (x["wall_ms"] / 1000) for x in v]
    print(f"    {lab}: runs {len(v)}, start {med([x['heat'] for x in v]):.2f}, end {med([x['heat_after'] for x in v]):.2f}, length {med([x['wall_ms'] / 1000 for x in v]):.1f} s, per second {med(sl):+.4f}")
