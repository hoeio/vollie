# Writes one expert's gate, up, down Q3_K slices (3 x 337,920 B) from the served-file copy. Read-only.
import sys
sys.path.insert(0, sys.argv[1] + '/gguf-py')
import gguf
r = gguf.GGUFReader(sys.argv[2])
L, E = int(sys.argv[4]), int(sys.argv[5])
t = {x.name: x for x in r.tensors}
with open(sys.argv[3], 'wb') as f:
    for n in ('gate', 'up', 'down'):
        x = t[f'blk.{L}.ffn_{n}_exps.weight']
        assert x.tensor_type.name == 'Q3_K'
        b = x.data.tobytes()
        per = len(b) // 64
        assert per == 337920
        f.write(b[E*per:(E + 1)*per])
