"""Confidence-gated emission from llama-perplexity --save-all-logits dumps (target vs draft, same tokens).
For each scored token: draft p_max (its confidence), whether draft argmax == target argmax, both log p(true).
Prints, per confidence threshold tau: coverage (share of tokens the draft would emit unverified), agreement
with the target's greedy choice on those, and the PPL of the mixed policy (draft's log-prob where gated,
target's elsewhere) vs the target alone, on the same chunks. Also the same curve gated on the TARGET's own
p_max (how often a top-1 is 'sure' at all: the ceiling for any confidence rule).
usage: python3 -I lab/draft_gate.py target.logits draft.logits"""
import sys, struct
import numpy as np

def load(path):
    f = open(path, 'rb'); assert f.read(8) == b'_logits_'
    n_ctx, n_vocab, n_chunk = struct.unpack('<iii', f.read(12))
    toks = np.frombuffer(f.read(4 * n_chunk * n_ctx), dtype='<i4').reshape(n_chunk, n_ctx)
    off = f.tell(); f.close()
    nv = 2 * ((n_vocab + 1) // 2) + 4; n_tok = n_ctx - 1 - n_ctx // 2
    mm = np.memmap(path, dtype='<u2', mode='r', offset=off, shape=(n_chunk, n_tok, nv))
    return n_ctx, n_vocab, n_chunk, toks, mm

def per_token(args):
    n_ctx, n_vocab, n_chunk, toks, mm = args
    first = n_ctx // 2
    out = {'argmax': [], 'pmax': [], 'lp_true': []}
    for c in range(n_chunk):
        blk = mm[c]                                  # [n_tok, nv]
        hdr = np.ascontiguousarray(blk[:, :4]).view('<f4')  # scale, min_log_prob
        q = blk[:, 4:4 + n_vocab]
        true = toks[c, first + 1: n_ctx]             # next tokens, n_tok of them
        am = q.argmax(axis=1)
        lp_max = hdr[:, 1] + hdr[:, 0] * 65535.0     # = -log_sum_exp
        qt = q[np.arange(len(true)), true].astype(np.float64)
        lp_true = hdr[:, 1] + hdr[:, 0] * qt         # q = 0 -> clamped at min_log_prob (logit <= max-16)
        out['argmax'].append(am); out['pmax'].append(np.exp(lp_max)); out['lp_true'].append(lp_true)
    return {k: np.concatenate(v) for k, v in out.items()}, toks

T, D = load(sys.argv[1]), load(sys.argv[2])
assert T[0] == D[0] and T[1] == D[1] and T[2] == D[2] and np.array_equal(T[3], D[3]), 'dumps differ in ctx/vocab/chunks/tokens'
t, toks = per_token(T); d, _ = per_token(D)
n = len(t['pmax'])
ppl_t = np.exp(-t['lp_true'].mean()); ppl_d = np.exp(-d['lp_true'].mean())
agree_all = (t['argmax'] == d['argmax']).mean()
print(f'tokens={n} chunks={T[2]}  target PPL={ppl_t:.4f}  draft PPL={ppl_d:.4f}  draft/target argmax agreement={agree_all:.3f}')
print(f'target greedy accuracy={(t["argmax"] == np.concatenate([toks[c, T[0]//2+1:] for c in range(T[2])])).mean():.3f}')
print('\n[A] gate on DRAFT p_max >= tau: emit the draft token unverified')
print('tau   coverage  agree|gated  wrong_emits/1000tok  PPL_mix  dPPL%')
for tau in (0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99, 0.995):
    g = d['pmax'] >= tau
    if g.sum() == 0: continue
    agree = (t['argmax'][g] == d['argmax'][g]).mean()
    lp = np.where(g, d['lp_true'], t['lp_true']); ppl = np.exp(-lp.mean())
    print(f'{tau:<5} {g.mean():8.3f}  {agree:11.3f}  {1000*(g & (t["argmax"] != d["argmax"])).mean():18.1f}  {ppl:7.4f}  {100*(ppl/ppl_t-1):+6.2f}')
print('\n[B] gate on TARGET p_max >= tau (ceiling for any confidence rule: share of tokens where a sure top-1 exists)')
print('tau   coverage  acc|gated')
true_all = np.concatenate([toks[c, T[0]//2+1:] for c in range(T[2])])
for tau in (0.5, 0.7, 0.9, 0.95, 0.99):
    g = t['pmax'] >= tau
    print(f'{tau:<5} {g.mean():8.3f}  {(t["argmax"][g] == true_all[g]).mean():9.3f}')
print('\n[C] draft agrees with target AND draft p_max >= tau: share of all tokens (what a perfect verifier would accept for free)')
for tau in (0.0, 0.5, 0.8, 0.9):
    g = (d['pmax'] >= tau) & (t['argmax'] == d['argmax'])
    print(f'tau={tau}: {g.mean():.3f}')
