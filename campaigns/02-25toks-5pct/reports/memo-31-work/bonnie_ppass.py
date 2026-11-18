# Bonnie, memo-31: P(median of 10 runs >= 25.0 tok/s) from stored r27 OLMoE runs. python3 -I bonnie_ppass.py
# All inputs are *device* (r27_2026-11-18.json); every output is *estimated* (resampling / mixture, in-sample, post hoc).
import random, statistics as st
random.seed(31)
K=965.6                     # tok/s = K / (decode ms per token), device identity (memo-29 s1)
P=[17.63,23.18,23.67,24.98,25.07,25.37,25.48,25.56]          # pin on, 8 scored
PX=[17.63,24.98,25.48,25.56]
ALL=P+[18.89,19.05,25.02,18.24,24.32,25.21,25.08,19.58]
CLEAN=[25.02,25.37,25.48,25.21,25.56,25.08,24.98]              # gate reached or zero wait, lost<=38 MB, no run-queue wait
TMO=[23.18,23.67,25.07]                                        # pinned runs after a 91 s timeout (lost 409-1324 MB)
CONT=[17.63,19.05,18.24]                                       # run-queue contention episodes
N=200000
def med10(draw): 
    x=sorted(draw() for _ in range(10)); return (x[4]+x[5])/2
def p(draw,n=N): return sum(med10(draw)>=25.0 for _ in range(n))/n
def shift(v,d): return K/(K/v+d)          # d ms per token, negative = faster
def boot(pool,d=0.0): return lambda: shift(random.choice(pool),d)
def mix(qc,qt,d=0.0,sd=0.33):
    def f():
        u=random.random()
        if u<qc: return K/(random.gauss(K/st.mean(CLEAN),sd)+d)     # clean: 38.3 ms +- 0.33 (sd of the 7, in ms)
        if u<qc+qt: return shift(random.choice(TMO),d)
        return shift(random.choice(CONT),d)
    return f
if __name__=='__main__':
    cm=[K/v for v in CLEAN]; print('clean ms mean %.2f sd %.2f  tps mean %.2f sd %.2f'%(st.mean(cm),st.stdev(cm),st.mean(CLEAN),st.stdev(CLEAN)))
    print('bootstrap  P runs %.3f | PX cell %.3f | all 16 %.3f'%(p(boot(P)),p(boot(PX)),p(boot(ALL))))
    print('mixture clean/timeout/contention:')
    for qc,qt in [(0.5,0.31),(0.5,0.375),(0.44,0.31),(0.6,0.25),(0.7,0.2),(0.8,0.1),(0.9,0.05),(1.0,0.0)]:
        print('  qc %.2f qt %.2f  '%(qc,qt)+'  '.join('d=%+.1f: %.2f'%(d,p(mix(qc,qt,d),40000)) for d in (0,-0.5,-1.0,-1.5,-2.0,-3.0,-3.5)))
    print('bootstrap P runs with shift: '+'  '.join('d=%+.1f: %.2f'%(d,p(boot(P,d),40000)) for d in (0,-0.3,-0.5,-1.0,-1.5,-2.0,-2.5,-3.0,-3.5)))
    assert abs(p(lambda:26.0,100)-1)<1e-9 and p(lambda:24.0,100)==0   # self-check
