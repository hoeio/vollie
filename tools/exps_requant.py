#!/usr/bin/env python3
"""Host-only probe (Bonnie, after R31): the PPL of a *runtime* Q4_0 copy of the routed experts.

Writes a GGUF where every `*_exps.weight` (Q3_K in the served file) is dequantised and re-quantised to Q4_0
with gguf-py's reference quantiser (== ggml quantize_row_q4_0_ref, no imatrix) and every other tensor and
KV is copied byte-identical. The PPL of this file is the PPL a runtime anon Q4_0x4 expert copy would have.
Never served; the output is a host artefact.  usage: exps_requant.py <in.gguf> <out.gguf> [Q4_0]
"""
import sys, time
sys.path.insert(0, '/Users/woobeenbaek/Documents/project/vollie/lab/llama.cpp-tiny/gguf-py')
import gguf
from gguf.quants import quantize, dequantize

src, dst = sys.argv[1], sys.argv[2]
qt = gguf.GGMLQuantizationType[sys.argv[3] if len(sys.argv) > 3 else 'Q4_0']
r = gguf.GGUFReader(src)
arch = r.fields['general.architecture'].contents()
w = gguf.GGUFWriter(dst, arch=arch, endianess=r.endianess)
for f in r.fields.values():
    if f.name == gguf.Keys.General.ARCHITECTURE or f.name.startswith('GGUF.'):
        continue
    vt = f.types[0]
    sub = f.types[-1] if vt == gguf.GGUFValueType.ARRAY else None
    w.add_key_value(f.name, f.contents(), vt, sub_type=sub)
out = []
t0 = time.time()
for t in r.tensors:
    if t.name.endswith('_exps.weight'):
        d = quantize(dequantize(t.data, t.tensor_type), qt)
        out.append((t, d, qt))
        print(f'{t.name}: {t.tensor_type.name} -> {qt.name} {d.nbytes/1e6:.1f} MB  t={time.time()-t0:.0f}s', flush=True)
    else:
        out.append((t, t.data, t.tensor_type))
    w.add_tensor_info(t.name, out[-1][1].shape, out[-1][1].dtype, out[-1][1].nbytes, out[-1][2])
w.write_header_to_file(); w.write_kv_data_to_file(); w.write_ti_data_to_file()
for t, d, _ in out:
    w.write_tensor_data(d, tensor_endianess=r.endianess)
w.close()
print('done', dst, f'{time.time()-t0:.0f}s')
