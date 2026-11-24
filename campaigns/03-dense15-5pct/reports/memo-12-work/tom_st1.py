#!/usr/bin/env python3
"""c3 memo-12 S-T1 (Tom): read lab/c3/spec/st1.txt (upstream's speculative loop, tools/c3_spec_ids.cpp, *host*) and set
it beside (a) a mirror of upstream's ngram-simple on the plain greedy tokens and (b) Bonnie r24's lookup arms
(bonnie_claim_guess.py: newest earlier match of the last 3, else 2, else 1 tokens, one guessed token; rule T = lookup at
match length >= 2, else the trained head). Everything over the first 32 generated tokens of each prompt.
Run: python3 -I -B tom_st1.py <repo root>
"""
import sys, os
import numpy as np

root = sys.argv[1]
SPEC = os.path.join(root, "lab/c3/spec")
HEAD = {"yi": ("head2117-tok", (2, 3)), "llama2": ("head3000-tok", (2, 3))}   # Bonnie r24: best head; rule T lengths
N = 32

def parse(path):
    out = {}
    for line in open(path):
        f = line.split()
        if len(f) < 4: continue
        s, p, tag = f[:3]; kv = dict(x.split("=", 1) for x in f[4:] if "=" in x and not x.startswith("pass="))
        passes = line.split("pass=", 1)[1].split()
        out[(s, p, tag)] = ([int(t) for t in kv["gen"].split(",")], [tuple(map(int, x.split(":"))) for x in passes])
    return out

def p_to(passes, n=N):
    """passes until n tokens exist (each pass gives 1 + accepted)"""
    tot = 0
    for i, (_d, a) in enumerate(passes):
        tot += 1 + a
        if tot >= n: return i + 1
    return None

# (a) mirror of common_ngram_simple_draft (ngram-map.cpp) and of speculative-simple's loop, greedy truth y
def simple_draft(tokens, sampled, n, m):
    cur = len(tokens)
    if cur <= n + m + 1: return []
    pat = tokens[cur - n + 1:] + [sampled]
    pos = 0
    for j in range(cur - n - 1, 0, -1):
        if tokens[j:j + n] == pat: pos = j; break
    if pos == 0: return []
    cmax = min(m, cur - (pos + n))
    if cmax < n: return []          # upstream: a draft shorter than the n-gram is dropped (so M < N never drafts)
    return tokens[pos + n:pos + n + cmax]

def mirror(prompt, y, n, m):
    tgt, last, k, passes = prompt[:-1], prompt[-1], 0, 0
    while k < N:
        d = simple_draft(tgt, last, n, m)
        nmax = N - k - 1
        if nmax > 0: d = d[:nmax]
        acc = 0
        while acc < len(d) and k + acc < len(y) and d[acc] == y[k + acc]: acc += 1
        ids = y[k:k + acc + 1]
        tgt = tgt + [last] + ids[:-1]; last = ids[-1]; k += len(ids); passes += 1
    return passes

# (b) Bonnie r24's lookup (copied from bonnie_claim_guess.py) and her pass count, first N tokens
def lookup(ctx, nmax=3):
    for n in range(nmax, 0, -1):
        if len(ctx) <= n: continue
        suf = ctx[-n:]
        for s in range(len(ctx) - n - 1, -1, -1):
            if ctx[s:s + n] == suf: return ctx[s + n], n
    return None, 0
def bonnie(prompt, y, choose, head=None):
    j = pas = 0
    while j < N:
        pas += 1
        if j < N - 1:
            g, n = lookup(prompt + y[:j + 1]); h = None if head is None or head[j] < 0 else int(head[j])
            c = choose(g, n, h)
            if c is not None and c == y[j + 1]: j += 2; continue
        j += 1
    return pas

def load_ptok(prefix):
    raw = np.fromfile(prefix + ".ptok", dtype=np.int32); pt, i = {}, 0
    while i < len(raw): p, n = int(raw[i]), int(raw[i + 1]); pt[p] = raw[i + 2:i + 2 + n].tolist(); i += 2 + n
    rows = np.fromfile(prefix + ".rows", dtype=np.int32).reshape(-1, 4)
    return pt, rows

R = parse(os.path.join(SPEC, "st1.txt"))
tags = ["plain", "simple-n2-m1", "simple-n3-m1", "simple-n2-m2", "simple-n3-m3", "mod-n2-d4", "mod-n3-d4"]
for s in ("yi", "llama2"):
    hd, use = HEAD[s]
    print(f"== {s} (tokens per pass over the first {N} tokens; passes in brackets; *k = text differs from plain in the first {N}, first at token k)")
    hdr = ["prompt", "dump=plain"] + tags[1:] + ["mir n2m2", "mir n3m3", "B lookup", "B lookup>=2", "B rule T"]
    print(" | ".join(hdr))
    tot = {t: [0, 0] for t in hdr[2:]}
    for p in ["official"] + [f"eval{i}" for i in range(8)]:
        t = "official" if p == "official" else "eval"; pid = 0 if p == "official" else int(p[4:])
        pt, rows = load_ptok(os.path.join(root, "lab/c3/head", s, t)); prompt = pt[pid]
        r = rows[rows[:, 0] == pid]; idx = np.where(rows[:, 0] == pid)[0][np.argsort(r[:, 1])]
        dump = rows[idx, 2].tolist()
        pred = np.load(os.path.join(root, "lab/c3/head", s, f"{hd}.pt.{'official' if t == 'official' else 'evalset'}.pred.npy"))[idx]
        plain = R[(s, p, "plain")][0]
        same_dump = plain[:N] == dump[:N]
        y = plain   # greedy truth: this build's plain run (33 tokens; only the first 32 matter for the pass count)
        fdd = next((i for i, (a, b) in enumerate(zip(plain, dump)) if a != b), None)
        cells = [p, "yes" if same_dump else f"NO at {fdd}"]
        for tg in tags[1:]:
            gen, ps = R[(s, p, tg)]; k = p_to(ps); diff = gen[:N] != plain[:N]
            fd = next((i for i, (a, b) in enumerate(zip(gen, plain)) if a != b), None)
            cells.append(f"{N / k:.3f} ({k}){'*' + str(fd) if diff else ''}"); tot[tg][0] += N; tot[tg][1] += k
        for lab, (n, m) in (("mir n2m2", (2, 2)), ("mir n3m3", (3, 3))):
            k = mirror(prompt, y, n, m); cells.append(f"{N / k:.3f} ({k})"); tot[lab][0] += N; tot[lab][1] += k
        for lab, fn in (("B lookup", lambda g, n, h: g), ("B lookup>=2", lambda g, n, h: g if n >= 2 else None),
                        ("B rule T", lambda g, n, h: g if n in use else h)):
            k = bonnie(prompt, y, fn, pred); cells.append(f"{N / k:.3f} ({k})"); tot[lab][0] += N; tot[lab][1] += k
        print(" | ".join(cells))
    print(" | ".join(["all 9", ""] + [f"{a / b:.3f} ({b})" for a, b in tot.values()]))
    # upstream drafted / accepted totals
    for tg in tags[1:]:
        d = sum(sum(x for x, _ in R[(s, p, tg)][1]) for p in ["official"] + [f"eval{i}" for i in range(8)])
        a = sum(sum(y for _, y in R[(s, p, tg)][1]) for p in ["official"] + [f"eval{i}" for i in range(8)])
        print(f"   {tg}: drafted {d}, accepted {a} (whole runs, to the loop's stop)")
