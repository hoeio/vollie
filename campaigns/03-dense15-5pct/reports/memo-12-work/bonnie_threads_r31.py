#!/usr/bin/env python3
"""memo-12 round 31 (Bonnie): where the decode threads sat and what the clock caps were in campaign 2's real decodes.
Reads committed kit JSONs (runs[] with kind official32): thread_samples = [ms, n, cpu...] taken during the run,
cap_before / cap_after = "cap=<A78 MHz>/<X1 MHz>", mhz_after = per-cpu MHz at the end of the run. *device* records.
Run: python3 -I -B bonnie_threads_r31.py <repo root>"""
import sys, os, json, glob, collections
root = sys.argv[1]
S = collections.defaultdict(lambda: collections.Counter()); caps = collections.defaultdict(collections.Counter); x1 = collections.defaultdict(list)
for f in sorted(glob.glob(os.path.join(root, "campaigns/02-25toks-5pct/reports/device/*.json"))):
    try: d = json.load(open(f))
    except ValueError: continue
    if not isinstance(d, dict) or not isinstance(d.get("runs"), list): continue
    for x in d["runs"]:
        if not isinstance(x, dict) or x.get("kind") != "official32" or not isinstance(x.get("thread_samples"), list): continue
        pin = (x.get("env") or {}).get("VOLLIE_POOL") == "1"; key = (x.get("model"), x.get("threads"), "pool" if pin else "no pool", (x.get("env") or {}).get("VOLLIE_CPUMASK"))
        c = S[key]; c["runs"] += 1
        for s in x["thread_samples"]:
            if not isinstance(s, list) or len(s) < 3 or s[1] != x.get("threads"): continue      # samples with all workers running
            cpus = s[2:]; c["samples"] += 1; c["with cpu 7"] += 7 in cpus; c["a thread on cpu 0-3"] += any(q < 4 for q in cpus); c["two threads on one cpu"] += len(set(cpus)) < len(cpus)
        caps[key][(x.get("cap_before"), x.get("cap_after"))] += 1
        m = x.get("mhz_after")
        if isinstance(m, str) and m.startswith("mhz="): v = [int(q) for q in m[4:].split(",")]; x1[key].append((v[4], v[7]))
for k in sorted(S, key=str):
    c = S[k]; n = max(c["samples"], 1); a = sorted(q[0] for q in x1[k]); b = sorted(q[1] for q in x1[k])
    print(k, f"runs {c['runs']}, samples with all workers {c['samples']}: cpu 7 used {c['with cpu 7'] / n:.2f}, a thread on cpu 0-3 {c['a thread on cpu 0-3'] / n:.2f}, shared cpu {c['two threads on one cpu'] / n:.2f}")
    print("     caps (before, after) -> runs:", dict(caps[k].most_common(4)), "; MHz at run end, A78 median", a[len(a) // 2] if a else None, "X1 median", b[len(b) // 2] if b else None, "X1 at 844 in", sum(q <= 844 for q in b), "of", len(b))
