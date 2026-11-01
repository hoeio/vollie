"""Expert overlap across consecutive decoded tokens, from 'eid <tensor> ids...' lines (GGML_EXPERT_LOG).
Per layer, per token: set of 6 experts. Reports, for k = 2..4 consecutive tokens, the mean number of
distinct experts per layer (what a batched verify of k tokens must read/unpack) vs 6*k if independent,
and the mean pairwise overlap with the previous token. Also the fraction of (layer, expert) pairs of token
t that were used by token t-1 (what an 'experts of the previous token' prefetch would cover).
usage: python3 -I overlap.py gen-*.err"""
import sys, collections, re
for path in sys.argv[1:]:
    layers = collections.defaultdict(list)  # tensor name -> [set per token]
    for line in open(path):
        if not line.startswith('eid '):
            continue
        parts = line.split()
        name = parts[1]
        if 'ffn_gate_exps' not in name:  # one tensor per layer is enough
            continue
        layers[name].append(frozenset(int(x) for x in parts[2:]))
    if not layers:
        print(path, 'no eid lines'); continue
    n_tok = min(len(v) for v in layers.values())
    # prompt tokens are batched (ids->ne[1] > 1, not logged), so these are decode tokens only
    out = [f'{path.split("/")[-1]}: layers={len(layers)} decode_tokens={n_tok}']
    for k in (1, 2, 3, 4, 6, 8):
        tot = cnt = 0
        for sets in layers.values():
            for i in range(0, n_tok - k + 1):
                u = set()
                for s in sets[i:i + k]:
                    u |= s
                tot += len(u); cnt += 1
        out.append(f'  k={k}: distinct experts/layer = {tot / cnt:5.2f}  (independent would be {min(64, 6 * k)}, ratio to k tokens x6 = {tot / cnt / (6 * k):.2f})')
    hit = tot2 = 0
    for sets in layers.values():
        for i in range(1, n_tok):
            hit += len(sets[i] & sets[i - 1]); tot2 += len(sets[i])
    out.append(f'  share of token t experts also used by token t-1: {hit / tot2:.1%}')
    # how concentrated: top experts overall per layer
    conc = []
    for sets in layers.values():
        c = collections.Counter(e for s in sets[:n_tok] for e in s)
        top16 = sum(v for _, v in c.most_common(16)) / sum(c.values())
        conc.append(top16)
    out.append(f'  share of expert uses covered by each layer\'s 16 most-used experts: {sum(conc) / len(conc):.1%}')
    print('\n'.join(out))
