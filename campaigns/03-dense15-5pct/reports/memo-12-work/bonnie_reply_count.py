#!/usr/bin/env python3
"""memo-12 round 30 (Bonnie): the count behind bonnie-joins-r29.md section 2. Reads every committed device JSON whose
runs[] carry a reply text for the official prompt (kind official32, 32 tokens) and counts distinct texts per cell
(model, arm, threads, repack, loaded file, prompt tokens). *device* records; nothing is run.
Run: python3 -I -B bonnie_reply_count.py <repo root>"""
import sys, os, json, glob, collections
root = sys.argv[1]
T = collections.defaultdict(collections.Counter); files = collections.defaultdict(set)
for f in sorted(glob.glob(os.path.join(root, "campaigns/02-25toks-5pct/reports/device/*.json")) + glob.glob(os.path.join(root, "reports/device/*.json"))):
    try: d = json.load(open(f))
    except ValueError: continue
    if not isinstance(d, dict) or not isinstance(d.get("runs"), list): continue
    for x in d["runs"]:
        if isinstance(x, dict) and isinstance(x.get("text"), str) and x.get("model") and x.get("tokens") == 32 and x.get("kind") == "official32":
            k = (x["model"], x.get("arm"), x.get("threads"), x.get("repack"), x.get("loaded_file"), x.get("prompt_tokens")); T[k][x["text"]] += 1; files[k].add(os.path.basename(f))
ids, runs = {}, 0
for k in sorted(T, key=str):
    names = [f"{k[0]}#{ids.setdefault((k[0], t), sum(a[0] == k[0] for a in ids) + 1)}" for t in T[k]]
    runs += sum(T[k].values()); print(k, "runs", sum(T[k].values()), "files", len(files[k]), "distinct texts", len(T[k]), names)
print("cells", len(T), "runs", runs, "cells with more than one text", sum(len(v) > 1 for v in T.values()))
assert all(len(v) == 1 for v in T.values()), "a cell has more than one reply text: r29 section 2 no longer holds"
