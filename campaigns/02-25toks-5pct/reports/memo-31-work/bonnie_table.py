# Bonnie, memo-31: per-run table from a stored device report. Usage: python3 -I bonnie_table.py <json> [model]
import json,sys,re
d=json.load(open(sys.argv[1])); model=sys.argv[2] if len(sys.argv)>2 else 'olmoe'
def kv(line): return dict(p.split('=',1) for p in line.split()[1:] if '=' in p)
def rows(d,model):
    out=[]
    for x in d['runs']:
        if x['model']!=model: continue
        c={l.split()[0]+(':'+kv(l).get('phase','') if l.startswith('c2-sched') else ''):kv(l) for l in x.get('c2',[]) if not l.startswith('c2-kvclear')}
        io=c.get('c2-io',{}); o={}
        o['pos']=x['pos']; o['id']=x['id'].replace(model+'_','')+'.'+str(x['rep']); o['phase']=x['phase']
        o['tps']=x.get('decode_tps'); o['ms']=x['decode_ms']/32 if x.get('decode_ms') else None
        o['ttft']=x.get('ttft_ms')
        o['heat']=x.get('heat'); o['heat_after']=x.get('heat_after')
        o['cap_b']=(x.get('cap_before') or '').replace('cap=',''); o['cap_a']=(x.get('cap_after') or '').replace('cap=','')
        o['mhz_b']=(x.get('mhz_before') or '').replace('mhz=','').split(',')[4:]
        o['mhz_a']=(x.get('mhz_after') or '').replace('mhz=','').split(',')[4:]
        o['wait_s']=(x.get('wait_ms') or 0)/1000; o['tmo']=x.get('gate_timeout')
        o['lost']=x.get('wait_lost_mb'); o['want']=x.get('want_mb')
        rb=x.get('res_before') or {}; o['res']=rb.get('total_mb')
        o['miss']=(o['want']-o['res']) if o['want'] and o['res'] is not None else None
        if 'majflt' in io:
            mj=[int(v) for v in io['majflt'].split('/')]; mn=[int(v) for v in io['minflt'].split('/')]
            o['dmaj']=mj[2]-mj[1]; o['dmin']=mn[2]-mn[1]; o['pmaj']=mj[1]-mj[0]
        if 'c2-io-blk' in c:
            b=[int(v) for v in c['c2-io-blk']['inblock'].split('/')]; o['dread_mb']=(b[2]-b[1])*512/1e6
        s=c.get('c2-sched:decode')
        if s:
            w=[[int(v) for v in s[k].split(':')] for k in ('w0','w1','w2') if k in s]
            o['cpu_ms']=[round(a[0]/1000/32,1) for a in w]; o['rq_ms']=[round(a[1]/1000/32,2) for a in w]; o['ivcs']=[a[3] for a in w]
        p=c.get('c2-pool')
        if p:
            o['pool']=p['pool']; 
            o['place']=[re.sub(r'^.*?:s\d+:e\d+:','',p[k]) for k in ('w0','w1','w2') if k in p]
        o['ts']=x.get('thread_samples')
        o['env']=x.get('env_set',{})
        m=re.search(r'cached=(\d+)',x.get('mem_before') or ''); o['cached']=int(m.group(1)) if m else None
        m=re.search(r'avail=(\d+)',x.get('mem_before') or ''); o['avail']=int(m.group(1)) if m else None
        o['batt']=x.get('batt_c_before'); o['hold_back']=(x.get('hold_stats') or {}).get('back_permille')
        o['vmswap']=None
        out.append(o)
    return out
if __name__=='__main__':
    for o in rows(d,model):
        f=lambda v,n=1: ('%.*f'%(n,v)) if isinstance(v,(int,float)) else str(v)
        print(o['pos'],o['id'],'tps',f(o['tps'],2),'ms',f(o['ms']),'ttft',f(o['ttft'],0),'heat',f(o['heat'],2),f(o['heat_after'],2),'cap',o['cap_b'],o['cap_a'],'mhzb',','.join(o['mhz_b']),'mhza',','.join(o['mhz_a']),'wait',f(o['wait_s'],0),o['tmo'],'lost',o['lost'],'miss',o['miss'],'dmaj',o.get('dmaj'),'dmin',o.get('dmin'),'rd',f(o.get('dread_mb')),'cpu',o.get('cpu_ms'),'rq',o.get('rq_ms'),'iv',o.get('ivcs'),'cached',o['cached'],'batt',o['batt'])
        print('      place',o.get('place'),'ts',o['ts'])
