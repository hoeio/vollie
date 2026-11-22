"""Prompt files for guess-head training and evaluation (model-generic).

Source: databricks-dolly-15k (CC BY-SA 3.0), file databricks-dolly-15k.jsonl at HF commit bdd27f4d,
sha256 2df90833...94ec. User message = instruction, plus a blank line and the context when there is one.

Split by record index, fixed, no randomness in membership:
  eval  = index % 100 == 7, prompt <= 2500 characters, first 32 in file order      (the provisional fixed prompt set)
  train = every other index whose instruction text does not equal an eval instruction, <= 2500 characters,
          taken in a fixed shuffled order (random.Random(0)), first n_train
Official prompt: tools/c2_official_prompt.txt, as one line.
Output lines are C-escaped (\\n, \\\\), the format tools/c3_head_dump.cpp reads.

  python3 -I tools/c3_head_prompts.py <dolly.jsonl> <out_dir> <n_train>
"""
import json, random, sys, os, hashlib

def esc(s):
    return s.replace('\\', '\\\\').replace('\n', '\\n').replace('\r', '')

def build(recs, n_train):
    msgs = [r['instruction'].strip() + ('\n\n' + r['context'].strip() if r['context'].strip() else '') for r in recs]
    ev = [i for i in range(len(recs)) if i % 100 == 7 and len(msgs[i]) <= 2500][:32]
    ev_instr = {recs[i]['instruction'].strip() for i in ev}
    tr = [i for i in range(len(recs)) if i % 100 != 7 and len(msgs[i]) <= 2500
          and recs[i]['instruction'].strip() not in ev_instr]
    random.Random(0).shuffle(tr)
    tr = tr[:n_train]
    assert not set(ev) & set(tr) and len(ev) == 32
    return msgs, ev, tr

if __name__ == '__main__':
    src, out, n_train = sys.argv[1], sys.argv[2], int(sys.argv[3])
    recs = [json.loads(l) for l in open(src, encoding='utf-8')]
    msgs, ev, tr = build(recs, n_train)
    os.makedirs(out, exist_ok=True)
    here = os.path.dirname(os.path.abspath(__file__))
    official = open(os.path.join(here, 'c2_official_prompt.txt'), encoding='utf-8').read()
    for name, idx in (('eval', ev), ('train', tr)):
        with open(os.path.join(out, name + '.txt'), 'w', encoding='utf-8') as f:
            f.write(''.join(esc(msgs[i]) + '\n' for i in idx))
        with open(os.path.join(out, name + '.idx'), 'w') as f:
            f.write(''.join(f'{i}\n' for i in idx))
    with open(os.path.join(out, 'official.txt'), 'w', encoding='utf-8') as f:
        f.write(esc(official) + '\n')
    for name in ('eval.txt', 'train.txt', 'official.txt'):
        p = os.path.join(out, name)
        print(name, sum(1 for _ in open(p, encoding='utf-8')), hashlib.sha256(open(p, 'rb').read()).hexdigest())
