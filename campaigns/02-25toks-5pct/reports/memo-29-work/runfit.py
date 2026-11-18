import json,re,sys
import numpy as np
D='campaigns/02-25toks-5pct/reports/device/'  # run from the repository root
def g(r,pat,n=3):
    for l in r.get('c2',[]):
        m=re.search(pat,l)
        if m: return [int(x) for x in m.groups()]
def mem(s,k):
    m=re.search(k+r'=(\d+)MB',s or ''); return int(m.group(1)) if m else -1
allrows={}
for f in sys.argv[1:]:
    d=json.load(open(D+f)); print('=====',f)
    for r in d['runs']:
        us=g(r,r'c2-io run=\S+ .* us=(\d+)/(\d+)/(\d+)'); mj=g(r,r'c2-io run=.* majflt=(\d+)/(\d+)/(\d+)'); mn=g(r,r'c2-io run=.* minflt=(\d+)/(\d+)/(\d+)'); b=g(r,r'inblock=(\d+)/(\d+)/(\d+)')
        if not us: continue
        wall=r['decode_ms']/32; dmaj=mj[2]-mj[1]; dmin=mn[2]-mn[1]; dMB=(b[2]-b[1])*512/1e6
        pmaj=mj[1]-mj[0]; pre=r['majflt']-(mj[2]-mj[0])
        hs=r.get('hold_stats') or {}
        capb=r.get('cap_before','')
        key=(f[:3],r['model'],r['arm'],r['threads'])
        gate=(r.get('gate') or {})
        rb=r.get('res_before') or {}
        print(f"{r['pos']:2d} {r['id']:20s} {r['phase']:6s} tps {r['decode_tps']:5.2f} wall {wall:5.1f} dmaj {dmaj:5d} dmin {dmin:6d} dMB {dMB:6.1f} pmaj {pmaj:6d} pre {pre:6d} ttft {r['ttft_ms']:5.0f} {capb} heat {r.get('heat',0):.2f} swapB {mem(r.get('mem_before'),'swap'):4d} swapfreeB {mem(r.get('mem_before'),'swapfree'):4d} availB {mem(r.get('mem_before'),'avail')} cachedB {mem(r.get('mem_before'),'cached')} tok1? ")
        if r['phase']!='warmup' and '1555' not in capb and r['threads']==3:
            allrows.setdefault((r['model'],r['arm']),[]).append((wall,dmaj,dMB,dmin,f[:3]))
print()
for k,L in allrows.items():
    A=np.array([x[:4] for x in L],float)
    y=A[:,0]; 
    for cols,name in (([1,2],'maj+MB'),([1,2,3],'maj+MB+min')):
        X=np.column_stack([np.ones(len(y))]+[A[:,c]/32 for c in cols])
        b=np.linalg.lstsq(X,y,rcond=None)[0]; res=y-X@b; s2=(res**2).sum()/(len(y)-X.shape[1]); se=np.sqrt(np.diag(s2*np.linalg.inv(X.T@X)))
        print(k,'n',len(y),name,'intercept %.1f±%.1f'%(b[0],se[0]),' '.join('%.3f±%.3f'%(b[i+1],se[i+1]) for i in range(len(cols))),'ms per (unit per token); resid sd %.1f'%s2**.5, 'median wall %.1f min %.1f'%(np.median(y),y.min()))
