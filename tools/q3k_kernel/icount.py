# Dynamic instruction mix of a loop nest from an llvm-objdump listing: argv = file, then "start-end*reps"
# line ranges (1-based, inclusive). Classes: sdot, V (other SIMD/FP data-processing), LS (loads/stores
# incl. vector), S (scalar/branch). The A78 issues V on 2 pipes; LS on 2-3 AGUs; S on 4 ALUs.
import re, sys
lines = open(sys.argv[1]).read().split('\n')
cnt = {'sdot': 0, 'V': 0, 'LS': 0, 'S': 0}
mn = {}
for spec in sys.argv[2:]:
    rng, reps = (spec.split('*') + ['1'])[:2]
    a, b = map(int, rng.split('-'))
    for l in lines[a - 1:b]:
        m = re.match(r'\s*[0-9a-f]+:\s+(\S+)\s*(.*)', l)
        if not m:
            continue
        op, args = m.group(1), m.group(2)
        if op.startswith(('ld', 'st', 'prfm', 'prfum')):
            c = 'LS'
        elif op == 'sdot':
            c = 'sdot'
        elif re.search(r'\bv\d+|\b[qdsh]\d+\b', args) or op.startswith(('f', 'scvtf', 'ucvtf')):
            c = 'V'
        else:
            c = 'S'
        cnt[c] += int(reps)
        if c in ('V', 'sdot'):
            mn[op] = mn.get(op, 0) + int(reps)
print(' '.join(f'{k}={v}' for k, v in cnt.items()), ' V+sdot=', cnt['V'] + cnt['sdot'])
print('  ', ' '.join(f'{k}:{v}' for k, v in sorted(mn.items(), key=lambda x: -x[1])))
