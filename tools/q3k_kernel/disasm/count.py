import re,sys,collections
lines=[l for l in open(sys.argv[1]) if re.match(r'\s+[0-9a-f]+:',l)]
def addr(l): return int(l.split(':')[0],16)
def mn(l): return l.split('\t')[1].split()[0] if '\t' in l else l.split()[1]
regions=[tuple(int(x,16) for x in r.split('-')) for r in sys.argv[2:]]
for lo,hi in regions:
    c=collections.Counter(mn(l) for l in lines if lo<=addr(l)<=hi)
    n=sum(c.values())
    ld=sum(v for k,v in c.items() if k.startswith(('ld','ldr','ldp','ld1')))
    st=sum(v for k,v in c.items() if k.startswith(('st')))
    vec=sum(v for k,v in c.items() if k.split('.')[0] in ('and','bic','orr','sub','add','ushr','sshr','shl','sshl','ushl','sdot','addp','mul','mla','sxtl','sxtl2','ssra','usra','addv','saddlv','fmla','fmul','fadd','fcvt','scvtf','fcvtzs','dup','mov','ins','umov','fmov','eor','zip1','zip2','uzp1','uzp2','ext','tbl','cmeq','mvn','smlal','smlal2','ssubl','sxtl','uxtl','uxtl2','bsl','trn1','trn2','rev64','neg','sqadd','cnt','not','ldr','mvn'))
    print(f"{lo:#x}-{hi:#x}: total {n}, loads {ld}, stores {st}")
    print("  ", ", ".join(f"{k}:{v}" for k,v in c.most_common(40)))
