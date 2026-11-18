import json,collections,glob,re
D='campaigns/02-25toks-5pct/reports/device/'  # run from the repository root
for f in sorted(glob.glob(D+'r2[2356]_*.json')):
    d=json.load(open(f)); c=collections.Counter(); h={}
    for r in d['runs']:
        k=(str(r.get('cap_before')),str(r.get('cap_after'))); c[k]+=1; h.setdefault(k,[]).append(r.get('heat',0))
    print(f[-18:]); 
    for k,n in c.most_common(8): print('   cap before/after',k,'n',n,'heat %.2f..%.2f'%(min(h[k]),max(h[k])))
    r=d['runs'][5]; print('   sample mhz_before',r.get('mhz_before'),'mhz_after',r.get('mhz_after'),'pinned',r.get('pinned_cpus'),'threads',r.get('threads'))
    # thread samples on little cores at 3 threads
    for key in r:
        if 'thr' in key.lower() or 'cpu' in key.lower(): print('   field',key,str(r[key])[:200])
