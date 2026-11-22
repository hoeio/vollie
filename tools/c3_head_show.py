"""Print the greedy text of a c3_head_dump dump, per prompt (sentencepiece-style pieces, rough detokenisation).
  python -I tools/c3_head_show.py <gguf-py dir> <model.gguf> <dump prefix> [max prompts]"""
import sys
import numpy as np
sys.path.insert(0, sys.argv[1])
import gguf
r = gguf.GGUFReader(sys.argv[2])
f = r.fields['tokenizer.ggml.tokens']
toks = [bytes(f.parts[i]).decode('utf-8', 'replace') for i in f.data]
rows = np.fromfile(sys.argv[3] + '.rows', dtype=np.int32).reshape(-1, 4)
rows = rows[np.lexsort((rows[:, 1], rows[:, 0]))]
for p in np.unique(rows[:, 0])[:int(sys.argv[4]) if len(sys.argv) > 4 else 9999]:
    t = rows[rows[:, 0] == p, 2]
    print(f'[{p}] {len(t)} tok:', ''.join(toks[i] for i in t).replace('▁', ' ').replace('<0x0A>', '\n'))
