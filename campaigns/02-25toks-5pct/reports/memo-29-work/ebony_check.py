import json,re,numpy as np
D='campaigns/02-25toks-5pct/reports/device/'  # run from the repository root
def mem(s,k):
    m=re.search(k+r'=(\d+)MB',s or ''); return int(m.group(1)) if m else None
def rank(a): return np.argsort(np.argsort(a)).astype(float)
for f in ['r25_2026-11-16.json','r26_2026-11-18.json']:
    d=json.load(open(D+f)); rows={}
    r0=d['runs'][0]; 
    if f.startswith('r26'): print('keys',sorted(r0.keys())); print('res_before',r0.get('res_before')); print('mem_before',r0.get('mem_before')); print('want',r0.get('want_mb'), r0.get('hold_stats'))
    for r in d['runs']:
        rb=r.get('res_before') or {}
        want=r.get('want_mb') or rb.get('want_mb'); tot=rb.get('total_mb')
        if want is None or tot is None or r.get('phase')=='warmup' or r.get('threads')!=3 or '1555' in str(r.get('cap_before','')): continue
        rows.setdefault((r['model'],r['arm']),[]).append((r['decode_ms']/32,want-tot,mem(r.get('mem_before'),'swap'),r['pos'],r.get('kv',r.get('id'))))
    for k,L in rows.items():
        y=np.array([x[0] for x in L]); m=np.array([x[1] for x in L],float); s=np.array([x[2] if x[2] is not None else np.nan for x in L],float)
        if len(y)<5: print(f,k,'n',len(y)); continue
        b=np.polyfit(m,y,1); r=np.corrcoef(m,y)[0,1]; rs=np.corrcoef(rank(m),rank(y))[0,1]
        loo=[np.polyfit(np.delete(m,i),np.delete(y,i),1)[0] for i in range(len(y))]
        top=np.argsort(m)[-2:]; b2=np.polyfit(np.delete(m,top),np.delete(y,top),1); r2=np.corrcoef(np.delete(m,top),np.delete(y,top))[0,1]
        print(f[:3],k,'n',len(y),'slope %.3f icpt %.1f r %.2f spearman %.2f | LOO slope %.3f..%.3f | drop 2 largest-missing: slope %.3f r %.2f | missing med %.0f max %.0f'%(b[0],b[1],r,rs,min(loo),max(loo),b2[0],r2,np.median(m),m.max()))
        if not np.isnan(s).any():
            print('    swap: r %.2f ; corr(missing,swap) %.2f'%(np.corrcoef(s,y)[0,1],np.corrcoef(s,m)[0,1]))
