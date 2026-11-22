# campaign 3 memo-08 (Tom): count vector-pipe use of a kernel's aarch64 assembly under the Cortex-A78 issue rules, as
# tabled in campaigns/03-dense15-5pct/reports/memo-08-work/bonnie-path.md section 2 (A78 Software Optimization Guide):
# SMLAL(2) V0 only; SCVTF and FCVTL (Q form) V0 only at half rate (2 cycles); immediate shifts and shift-long V1 only;
# other vector ALU ops on either pipe (4-register TBL counted as 4, an assumption). An estimate of issue limits, not a
# simulation: no latency, no dependency chains, loads and stores counted apart.
# usage: c3_a78_pipes.py <function.s>...   (one function per file: awk '$1=="k1:"{p=1} p&&/^\.Lfunc_end/{exit} p' all.s)
# prints per basic-block group: col (innermost loop with smlal/sdot), flt (innermost loop with fmla), blk (the rest of
# the loop one level up, i.e. per-block code)
import re, sys
V0 = {'smlal': 1, 'smlal2': 1, 'smull': 1, 'smull2': 1, 'scvtf': 2, 'ucvtf': 2, 'fcvtl': 2, 'fcvtl2': 2}
V1 = {'ushr', 'sshr', 'shl', 'sshl', 'ushl', 'sshll', 'sshll2', 'ushll', 'ushll2', 'sxtl', 'sxtl2', 'uxtl', 'uxtl2', 'shll', 'shll2'}
ANY = {'and', 'orr', 'eor', 'bic', 'add', 'sub', 'fmul', 'fmla', 'fadd', 'fsub', 'sdot', 'udot', 'zip1', 'zip2', 'uzp1', 'uzp2',
       'trn1', 'trn2', 'tbl', 'movi', 'mov', 'dup', 'saddw', 'saddw2', 'ext', 'neg', 'cmlt', 'bsl', 'bif', 'bit', 'mvn', 'not', 'addp'}
LS = re.compile(r'^(ldr|ldp|ld1|ld1r|ld2|ld4|str|stp|st1|ldur|stur|ldrb|ldrh|prfm)')
def classify(lines):
    c = dict(v0=0, v1=0, any=0, ls=0, sc=0, n=0); seen = {}
    for l in lines:
        t = l.split('//')[0].strip()
        if not t or t.endswith(':') or t.startswith('.'): continue
        m = t.split()[0]; ops = t[len(m):]; vec = bool(re.search(r'\bv\d+\.', ops)) or bool(re.search(r'\bq\d+\b', ops))
        c['n'] += 1; seen[m] = seen.get(m, 0) + 1
        if LS.match(m): c['ls'] += 1
        elif vec and m in V0: c['v0'] += V0[m]
        elif vec and m in V1: c['v1'] += 1
        elif vec and m == 'tbl': c['any'] += 4 if re.search(r'\{[^}]*,[^}]*,[^}]*,[^}]*\}', ops) else 1   # 4-register TBL counted as 4 uops (assumption)
        elif vec and m in ANY: c['any'] += 1
        elif vec: c['any'] += 1; c.setdefault('unknown', set()).add(m)
        else: c['sc'] += 1
    c['floor'] = max(c['v0'], c['v1'], (c['v0'] + c['v1'] + c['any']) / 2); c['mix'] = seen
    return c
for f in sys.argv[1:]:
    L = open(f).read().split('\n'); blocks = []; cur = None
    for l in L:
        if re.match(r'^(\.LBB\S+:|// %bb\.\d+:)', l):
            m = re.search(r'Depth=(\d+)', l); cur = [l, int(m.group(1)) if m else 0, []]; blocks.append(cur)
        elif cur is not None:
            m = re.search(r'Depth=(\d+)', l)
            if m and l.strip().startswith('//'): cur[1] = max(cur[1], int(m.group(1)))
            else: cur[2].append(l)
    D = max(b[1] for b in blocks); g = {'col': [], 'flt': [], 'blk': []}
    inner = [b for b in blocks if b[1] == D]; mid = [b for b in blocks if b[1] == D - 1]
    for b in inner:
        txt = ' '.join(b[2]); g['col' if re.search(r'\b(smlal|sdot)\b', txt) else 'flt' if 'fmla' in txt else 'blk'] += b[2]
    for b in mid: g['blk'] += b[2]
    out = []
    for k in ('col', 'flt', 'blk'):
        c = classify(g[k]); out.append('%s V0=%d V1=%d any=%d ls=%d n=%d' % (k, c['v0'], c['v1'], c['any'], c['ls'], c['n']))
    print(f.split('fn-')[-1], ' | '.join(out))
