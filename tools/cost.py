"""Per-token decode cost of a GGUF (machine-independent): params, bytes read, FLOPs, KV/recurrent state.
usage: python3 tools/cost.py <gguf> [ctx=4096] [experts_used]
Counts: dense tensors read fully; expert tensors * used/total; lm_head = output.weight or tied token_embd;
token_embd is a row gather when output.weight exists; MTP (nextn) blocks excluded (not loaded by Vollie)."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../lab/llama.cpp/gguf-py'))
from gguf import GGUFReader

r = GGUFReader(sys.argv[1]); ctx = int(sys.argv[2]) if len(sys.argv) > 2 else 4096
arch = r.get_field('general.architecture').contents()
kv = lambda k, d=None: (lambda f: f.contents() if f else d)(r.get_field(f'{arch}.{k}'))
n_exp = kv('expert_count', 0); n_used = int(sys.argv[3]) if len(sys.argv) > 3 else kv('expert_used_count', 0)
nl = kv('block_count'); mtp_from = nl - (kv('nextn_predict_layers') or 0)
is_mtp = lambda name: name.startswith('blk.') and int(name.split('.')[1]) >= mtp_from
tied = not any(t.name == 'output.weight' for t in r.tensors)
P = dict(total=0, active=0); B = dict(dense=0, experts=0, head=0, router=0)
for t in r.tensors:
    n, b = int(t.n_elements), int(t.n_bytes)
    P['total'] += n
    if is_mtp(t.name): continue
    if t.name == 'token_embd.weight':
        if tied: B['head'] += b; P['active'] += n
        continue
    if t.name == 'output.weight': B['head'] += b; P['active'] += n; continue
    if '_exps' in t.name: B['experts'] += b * n_used / n_exp; P['active'] += n * n_used / n_exp; continue
    if 'ffn_gate_inp' in t.name: B['router'] += b
    else: B['dense'] += b
    P['active'] += n
hkv = kv('attention.head_count_kv'); hkv = hkv if isinstance(hkv, list) else [hkv] * nl
emb = kv('embedding_length'); hd = kv('attention.key_length') or emb // kv('attention.head_count')
n_attn = sum(1 for h in hkv if h)
if arch == 'qwen35':  # recurrent_layers flags; last block is nextn
    rec = kv('attention.recurrent_layers'); n_attn = sum(1 for x in rec[:nl - (kv('nextn_predict_layers') or 0)] if not x)
    n_rec = sum(rec); hkv = [kv('attention.head_count_kv')] * n_attn
    st = n_rec * (kv('ssm.time_step_rank') * 128 * 128 + 3 * kv('ssm.inner_size') * 2) * 4  # gated delta net S + conv
else:
    n_rec = nl - n_attn; di, ds, g = kv('ssm.inner_size', 0), kv('ssm.state_size', 0), kv('ssm.group_count', 1)
    st = n_rec * (di * ds + (kv('ssm.conv_kernel', 1) - 1) * (di + 2 * g * ds)) * 4  # mamba2 S + conv (f32)
kv_tok = 2 * sum(h for h in hkv if h) * hd * 2  # f16 K+V bytes per position, attention layers only
tot = sum(B.values())
print(f'{os.path.basename(sys.argv[1])} [{arch}] experts {n_used}/{n_exp} tied_head={tied}')
print(f'  params total {P["total"]/1e9:.2f} B, active/token {P["active"]/1e9:.3f} B -> ~{2*P["active"]/1e9:.2f} GFLOP/token (matmul)')
print(f'  weight bytes/token {tot/1e6:.0f} MB = dense {B["dense"]/1e6:.0f} + experts {B["experts"]/1e6:.0f} + router {B["router"]/1e6:.0f} + lm_head {B["head"]/1e6:.0f}')
print(f'  attn layers {n_attn}, recurrent {n_rec}: KV f16 {kv_tok/1024:.1f} KiB/pos -> {kv_tok*ctx/1e6:.0f} MB at ctx {ctx}; '
      f'recurrent state {st/1e6:.0f} MB (read+write each token)')
print(f'  file {os.path.getsize(sys.argv[1])/1e6:.0f} MB')
