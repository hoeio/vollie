# lab-only: copy the served file with token_embd (tied lm_head) requantized Q6_K -> Q4_0; everything else byte-identical
import sys, numpy as np
sys.path.insert(0, 'lab/llama.cpp-tiny/gguf-py')
import gguf
src, dst, newtype = sys.argv[1], sys.argv[2], getattr(gguf.GGMLQuantizationType, sys.argv[3])
r = gguf.GGUFReader(src)
arch = r.fields['general.architecture'].contents()
w = gguf.GGUFWriter(dst, arch)
for f in r.fields.values():
    if f.name == gguf.Keys.General.ARCHITECTURE or f.name.startswith('GGUF.'): continue
    vt = f.types[0]; st = f.types[-1] if vt == gguf.GGUFValueType.ARRAY else None
    w.add_key_value(f.name, f.contents(), vt, sub_type=st)
new = {}
for t in r.tensors:
    if t.name == 'token_embd.weight':
        f32 = gguf.quants.dequantize(t.data, t.tensor_type)          # shape (n_vocab, n_embd)
        q = gguf.quants.quantize(f32, newtype)
        new[t.name] = q
        w.add_tensor_info(t.name, q.shape, q.dtype, q.nbytes, newtype)
        print(t.name, t.tensor_type.name, '->', newtype.name, f32.shape, t.n_bytes, '->', q.nbytes)
    else:
        w.add_tensor_info(t.name, t.data.shape, t.data.dtype, t.data.nbytes, t.tensor_type)
w.write_header_to_file(); w.write_kv_data_to_file(); w.write_ti_data_to_file()
for t in r.tensors:
    w.write_tensor_data(new.get(t.name, t.data), tensor_endianess=r.endianess)
w.close(); print('done', dst)
