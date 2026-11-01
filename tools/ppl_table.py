"""Table of all tools/ppl.sh results with bits-per-byte (cross-tokenizer comparable).
bpb = log2(PPL) * tokens/bytes of the whole file (assumes scored tokens are representative).
usage: python3 tools/ppl_table.py [tag-substring]"""
import glob, math, os, re, sys
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
BYTES = {'wiki': 1290590, 'en': 57706}
# token counts from llama-tokenize --show-count on each tokenizer family (tag prefix -> family)
TOKENS = {'granite': {'wiki': 289305, 'en': 13441}, 'qwen35': {'wiki': 297415, 'en': 13570},
          'ling': {'wiki': 304410, 'en': 14871}}
FAMILY = {'A': 'granite', 'V': 'granite', 'F': 'granite', 'B': 'qwen35', 'C': 'ling'}
rows = []
for log in sorted(glob.glob(f'{ROOT}/lab/logs/p/ppl-*.log')):
    tag, ds = re.match(r'.*/ppl-(.+)-(wiki|en)\.log', log).groups()
    if len(sys.argv) > 1 and sys.argv[1] not in tag: continue
    m = re.search(r'Final estimate: PPL = ([\d.]+) \+/- ([\d.]+)', open(log, errors='ignore').read())
    if not m: continue
    ppl, err = float(m[1]), float(m[2]); fam = FAMILY.get(tag.split('-')[0][0], tag.split('-')[0])
    t = TOKENS.get(fam, {}).get(ds)
    rows.append((tag, ds, ppl, err, math.log2(ppl) * t / BYTES[ds] if t else float('nan')))
print(f'{"tag":28s} {"set":4s} {"PPL":>9s} {"+/-":>7s} {"bpb":>6s}')
for r in rows: print(f'{r[0]:28s} {r[1]:4s} {r[2]:9.4f} {r[3]:7.4f} {r[4]:6.3f}')
