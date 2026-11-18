import json,re,sys,statistics as st
D='campaigns/02-25toks-5pct/reports/device/'  # run from the repository root
def parse_prof(line):
    h,ops=line.split(' | ')
    kv=dict(x.split('=') for x in h.split()[1:])
    o={}
    for m in re.finditer(r'(\w+) (\d+):(\d+)\+(\d+)',ops): o[m.group(1)]=(int(m.group(3)),int(m.group(4)))
    return kv,o
def c2io(r):
    for l in r.get('c2',[]):
        if l.startswith('c2-io run='):
            us=[int(x) for x in re.search(r' us=(\d+)/(\d+)/(\d+)',l).groups()]
            mj=[int(x) for x in re.search(r' majflt=(\d+)/(\d+)/(\d+)',l).groups()]
            return us,mj
    return None,None
def blk(r):
    for l in r.get('c2',[]):
        if l.startswith('c2-io-blk'):
            return [int(x) for x in re.search(r'inblock=(\d+)/(\d+)/(\d+)',l).groups()]
m=lambda x: sum(x)/len(x) if x else float('nan')
for f in sys.argv[1:]:
    d=json.load(open(D+f)); print('=====',f)
    for r in d['runs']:
        us,mj=c2io(r); b=blk(r)
        dg=(us[2]-us[1])/32/1000 if us else float('nan')
        dmj=mj[2]-mj[1] if mj else -1
        dmb=(b[2]-b[1])*512/1e6 if b else -1
        wall=r.get('decode_ms',float('nan'))/32
        rb=r.get('res_before') or {}
        resb=rb.get('res_mb', rb.get('mb','?')) if isinstance(rb,dict) else '?'
        line=f"{r['pos']:2d} {r['id']:22s} t{r['threads']} tps {r['decode_tps']:5.2f} graph {dg:5.1f} wall/32 {wall:5.1f} out {wall-dg:4.2f} dmaj {dmj:5d} dMB {dmb:6.1f} {r.get('cap_before','')} ttft {r.get('ttft_ms',0):.0f}"
        print(line)
        if r.get('prof'):
            rows=[]
            for p in r['prof']:
                kv,o=parse_prof(p)
                if int(kv['tok'])<2: continue
                rows.append((int(kv['us/token']),int(kv['majflt']),kv['mhz'],int(kv['head']),int(kv['router']),o))
            ff=[x for x in rows if x[1]<=5]
            ff.sort(key=lambda x:x[0]); half=ff[:max(1,len(ff)//2)]
            def agg(S,lab):
                if not S: print('    ',lab,'none'); return
                g=lambda k,i: m([x[5].get(k,(0,0))[i] for x in S])/1000
                tot=m([x[0] for x in S])/1000
                four=sum(g(k,0)+g(k,1) for k in ('MUL_MAT','MUL_MAT_ID','SSM_SCAN','GLU'))
                waits=sum(g(k,1) for k in ('MUL_MAT','MUL_MAT_ID','SSM_SCAN','GLU'))
                hd=m([x[3] for x in S])/1000; rt=m([x[4] for x in S])/1000
                print(f"     {lab} n={len(S)} mhz={S[0][2]} tok {tot:5.2f} | MMID {g('MUL_MAT_ID',0):5.2f}+{g('MUL_MAT_ID',1):4.2f} | MM {g('MUL_MAT',0):5.2f}+{g('MUL_MAT',1):4.2f} (head {hd:4.2f} router {rt:4.2f} otherdense {g('MUL_MAT',0)+g('MUL_MAT',1)-hd-rt:5.2f}) | SCAN {g('SSM_SCAN',0):4.2f}+{g('SSM_SCAN',1):4.2f} GLU {g('GLU',0):4.2f} | rest {tot-four:4.2f} waits {waits:4.2f}")
            agg(rows,'all2-32 ')
            agg(ff,'faultfree')
            agg(half,'fast half')
