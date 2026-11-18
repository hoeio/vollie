import json,re,sys
import numpy as np
D='campaigns/02-25toks-5pct/reports/device/'  # run from the repository root
rows={}
for f in ['r22_2026-11-15.json','r23_2026-11-16.json','r25_2026-11-16.json','r26_2026-11-18.json']:
    d=json.load(open(D+f))
    for r in d['runs']:
        if not r.get('tok') or r['threads']!=3: continue
        key=(r['model'],r['arm'])
        T=[t for t in r['tok']]
        # c1 fields: ab tok us head router majflt minflt mhz4 mhz7 nodes barriers
        t1=T[0]
        rest=[t for t in T if t[1]>=2]
        if any(t[7]<2000 for t in rest[:3]): cap='capped'
        else: cap=''
        us=np.array([t[2] for t in rest])/1000; mj=np.array([t[5] for t in rest]); mn=np.array([t[6] for t in rest])
        z=(mj==0)
        print(f"{f[:3]} pos{r['pos']:2d} {key[0]:7s} {key[1]:3s} {cap:6s} tps {r['decode_tps']:5.2f} tok1 {t1[2]/1000:6.1f} (maj {t1[5]:5d} min {t1[6]:5d}) mean2-32 {us.mean():5.1f} | maj=0: n {z.sum():2d} mean {us[z].mean() if z.any() else float('nan'):5.1f} min {us.min():5.1f} p25 {np.percentile(us,25):5.1f} | maj>0: n {(~z).sum():2d} mean {us[~z].mean() if (~z).any() else float('nan'):5.1f} majsum {mj.sum():5d} minsum {mn.sum():6d} minflt med {np.median(mn):.0f}")
        if not cap: rows.setdefault(key,[]).append((f[:3]+str(r['pos']),us,mj,mn))
print()
for key,L in rows.items():
    # OLS with run dummies
    y=np.concatenate([x[1] for x in L]); mj=np.concatenate([x[2] for x in L]); mn=np.concatenate([x[3] for x in L])
    dm=np.zeros((len(y),len(L))); i=0
    for j,x in enumerate(L):
        dm[i:i+len(x[1]),j]=1; i+=len(x[1])
    X=np.column_stack([mj,mn,dm])
    b,res,rk,sv=np.linalg.lstsq(X,y,rcond=None)
    pred=X@b; s2=((y-pred)**2).sum()/(len(y)-X.shape[1]); cov=s2*np.linalg.inv(X.T@X)
    print(key,'n',len(y),f"us/maj {b[0]*1000:.1f} ± {cov[0,0]**.5*1000:.1f}  us/minflt {b[1]*1000:.2f} ± {cov[1,1]**.5*1000:.2f}  run intercepts:",' '.join(f"{L[j][0]}:{b[2+j]:.1f}" for j in range(len(L))), f"resid sd {s2**.5:.1f}")
