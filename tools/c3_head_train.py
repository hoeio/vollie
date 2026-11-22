"""Train and score a next-next guess head (Medusa-1 style) on dumps from tools/c3_head_dump.cpp. Model-generic.

Head: g(h) = h + SiLU(W h + b), then the served file's own output head (output.weight, dequantized exactly from the
GGUF, frozen). With --tok the block also reads E[y_j], the input embedding of the token row j produced (EAGLE-like;
known when the guess is made): g = h + SiLU(W h + Wt E[y_j] + b). Input h is the post-norm hidden state at row j; the target is the greedy token of row j+1, i.e. the
token after the one row j produces. W starts at zero (Medusa). Optional low rank: W = U V (rank r).
A closed-form ridge probe runs first as a lower bound: A = argmin |H_j A - H_{j+1}|^2 + lam |A|^2, guess =
argmax head(H_j A) (Ebony, memo-09 §10).

Scores (every number this prints is *host*):
  top1      P(guess at row j == token of row j+1) over all rows j that have a row j+1
  short-K   the same with the argmax restricted to the K most frequent target tokens of the training rows
  tau       committed tokens per pass when the guess drives exact batch-2 lookahead, simulated on each sequence
            (accept: jump two rows; miss: jump one), over the rows of the decode
  base      P(argmax head(h_j) == token of row j), the sanity check of the dequantized head against llama.cpp

  python -I tools/c3_head_train.py <gguf-py dir> <model.gguf> <train dump> <out head .pt> --eval <name>=<dump> ...
         [--rank 0] [--epochs 2] [--lam 1.0] [--holdout 0.05] [--no-ridge] [--load head.pt: score only]
"""
import argparse, hashlib, os, sys, time
import numpy as np


def load_dump(prefix, n_embd):
    rows = np.fromfile(prefix + '.rows', dtype=np.int32).reshape(-1, 4)
    h = np.memmap(prefix + '.h16', dtype=np.float16, mode='r', shape=(len(rows), n_embd))
    return rows, h


def pairs(rows):
    """Index pairs (j, j+1) of consecutive steps of the same prompt; rows may interleave prompts."""
    order = np.lexsort((rows[:, 1], rows[:, 0]))
    r = rows[order]
    ok = (r[1:, 0] == r[:-1, 0]) & (r[1:, 1] == r[:-1, 1] + 1)
    return order[:-1][ok], order[1:][ok]


def tau(rows, hit_by_row, only=None):
    """Committed tokens per pass. hit_by_row[j] says whether the guess made at row j equals row j+1's token."""
    order = np.lexsort((rows[:, 1], rows[:, 0]))
    out_tok = out_pass = 0
    start = 0
    while start < len(order):
        end = start
        while end + 1 < len(order) and rows[order[end + 1], 0] == rows[order[start], 0]:
            end += 1
        seq = order[start:end + 1]          # rows of one prompt, step order; row 0 is the last prompt token
        if only is not None and int(rows[seq[0], 0]) not in only:
            start = end + 1
            continue
        j, n = 0, len(seq)                  # each row produces one token; n tokens to commit
        while j < n:
            out_pass += 1
            if j + 1 < n and hit_by_row.get(int(seq[j]), False):
                j += 2
            else:
                j += 1
        out_tok += n
        start = end + 1
    return out_tok / max(out_pass, 1)


def selftest():
    rows = np.array([[0, 0, 5, 0], [0, 1, 6, 1], [0, 2, 7, 2], [1, 0, 9, 0], [1, 1, 9, 1]], dtype=np.int32)
    a, b = pairs(rows)
    assert list(a) == [0, 1, 3] and list(b) == [1, 2, 4]
    assert tau(rows, {}) == 1.0
    assert abs(tau(rows, {0: True, 3: True}) - 5 / 3) < 1e-9   # prompt 0: 2 passes for 3; prompt 1: 1 pass for 2
    assert tau(rows, {0: True, 3: True}, only={1}) == 2.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('gguf_py'); ap.add_argument('model'); ap.add_argument('train'); ap.add_argument('out')
    ap.add_argument('--eval', action='append', default=[])
    ap.add_argument('--rank', type=int, default=0); ap.add_argument('--epochs', type=float, default=2)
    ap.add_argument('--lr', type=float, default=1e-3); ap.add_argument('--bs', type=int, default=256)
    ap.add_argument('--lam', type=float, default=1.0); ap.add_argument('--holdout', type=float, default=0.05); ap.add_argument('--wd', type=float, default=0.0); ap.add_argument('--tok', action='store_true'); ap.add_argument('--max-prompts', type=int, default=0)
    ap.add_argument('--no-ridge', action='store_true'); ap.add_argument('--load', help='score a saved head, no training'); ap.add_argument('--shortlist', default='4096,8192,16384')
    a = ap.parse_args()
    selftest()
    sys.path.insert(0, a.gguf_py)
    import gguf, torch
    dev = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

    t0 = time.time()
    rd = gguf.GGUFReader(a.model)
    ow = next((t for t in rd.tensors if t.name == 'output.weight'), None) or \
        next(t for t in rd.tensors if t.name == 'token_embd.weight')      # tied head
    head = gguf.quants.dequantize(ow.data, ow.tensor_type)               # (n_vocab, n_embd)
    n_vocab, n_embd = head.shape
    print(f'head {ow.name} {ow.tensor_type.name} {n_vocab}x{n_embd}, dequant {time.time()-t0:.0f} s')
    Wh = torch.from_numpy(head.astype(np.float16)).to(dev)
    del head
    model_sha = hashlib.sha256()
    with open(a.model, 'rb') as f:
        for blk in iter(lambda: f.read(1 << 24), b''):
            model_sha.update(blk)
    model_sha = model_sha.hexdigest()

    rows, H = load_dump(a.train, n_embd)
    ja, jb = pairs(rows)
    prompts = np.unique(rows[ja, 0])
    rng = np.random.default_rng(0)
    held = set(rng.choice(prompts, size=max(1, int(len(prompts) * a.holdout)), replace=False).tolist())
    m_held = np.isin(rows[ja, 0], list(held))
    m_tr = ~m_held
    if a.max_prompts:   # learning-curve runs: train on the first N non-held-out prompts only (same held-out set)
        keep = [p for p in prompts if p not in held][:a.max_prompts]
        m_tr &= np.isin(rows[ja, 0], keep)
    tr_a, tr_b, ho_a, ho_b = ja[m_tr], jb[m_tr], ja[m_held], jb[m_held]
    tgt_all = rows[:, 2]
    counts = np.bincount(tgt_all[tr_b], minlength=n_vocab)
    by_freq = np.argsort(-counts, kind='stable')
    print(f'train rows {len(rows)}, pairs {len(tr_a)} train / {len(ho_a)} held-out ({len(held)} prompts)')

    def hb(idx):  # fp16 rows -> device fp32
        idx = np.sort(idx)
        return torch.from_numpy(np.asarray(H[idx], dtype=np.float32)).to(dev), idx

    Wh32 = None

    def argmax_head(x, short=None):  # scoring in fp32 on the CPU: on MPS the full-vocab argmax came out wrong in
        nonlocal Wh32                # batched scoring (top-1 0.11 and 0.34 for the same head; CPU 0.33). Host check.
        if Wh32 is None:
            Wh32 = Wh.float().cpu()
        lg = x.float().cpu() @ Wh32.T
        if short is not None:
            return short.cpu()[lg[:, short.cpu()].argmax(1)]
        return lg.argmax(1)

    shorts = {int(k): torch.from_numpy(by_freq[:int(k)].copy()).to(dev) for k in a.shortlist.split(',') if int(k) < n_vocab}

    # ridge probe, closed form, on up to 200k training pairs
    A = None
    if not a.no_ridge:
        t1 = time.time()
        sel = tr_a if len(tr_a) <= 200000 else rng.choice(tr_a, 200000, replace=False)
        nxt = dict(zip(tr_a.tolist(), tr_b.tolist()))
        XtX = torch.zeros(n_embd, n_embd, dtype=torch.float64); XtY = torch.zeros_like(XtX)
        for s in range(0, len(sel), 8192):
            ia = np.sort(sel[s:s + 8192]); ib = np.array([nxt[int(i)] for i in ia])
            X = torch.from_numpy(np.asarray(H[ia], dtype=np.float64)); Y = torch.from_numpy(np.asarray(H[ib], dtype=np.float64))
            XtX += X.T @ X; XtY += X.T @ Y
        A = torch.linalg.solve(XtX + a.lam * len(sel) / 1000 * torch.eye(n_embd, dtype=torch.float64), XtY).float().to(dev)
        print(f'ridge solved on {len(sel)} pairs in {time.time()-t1:.0f} s')

    # resblock head
    W = torch.zeros(n_embd, n_embd, device=dev) if a.rank == 0 else None
    if a.rank:
        U = (torch.randn(n_embd, a.rank, device=dev) / n_embd ** 0.5).requires_grad_()
        V = torch.zeros(a.rank, n_embd, device=dev, requires_grad=True)
        params = [U, V]
    else:
        W.requires_grad_()
        params = [W]
    bvec = torch.zeros(n_embd, device=dev, requires_grad=True)
    params.append(bvec)
    E = Wt = None
    if a.tok:   # EAGLE-like variant: the block also reads the input embedding of row j's own token y_j (known then)
        et = next(t for t in rd.tensors if t.name == 'token_embd.weight')
        E = torch.from_numpy(gguf.quants.dequantize(et.data, et.tensor_type).astype(np.float16)).to(dev)
        Wt = torch.zeros(n_embd, n_embd, device=dev, requires_grad=True)
        params.append(Wt)

    def resblock(x, ytok=None):
        lin = x @ W.T if a.rank == 0 else (x @ V.T) @ U.T
        if Wt is not None:
            lin = lin + E[torch.as_tensor(ytok, device=dev)].float() @ Wt.T
        return x + torch.nn.functional.silu(lin + bvec)

    if a.load:
        sd0 = torch.load(a.load)
        with torch.no_grad():
            for k, t in ((('W', W), ('b', bvec)) if a.rank == 0 else (('U', U), ('V', V), ('b', bvec))) + ((('Wt', Wt),) if Wt is not None else ()):
                t.copy_(sd0[k].to(dev))
        a.epochs = 0
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=a.wd)
    steps = int(len(tr_a) * a.epochs / a.bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=max(steps, 1), pct_start=0.05)
    nxt_tok = np.zeros(len(rows), dtype=np.int64); nxt_tok[ja] = tgt_all[jb]
    t2 = time.time(); perm = np.concatenate([rng.permutation(tr_a) for _ in range(max(1, int(np.ceil(a.epochs))))])
    for st in range(steps):
        x, idx = hb(perm[st * a.bs:(st + 1) * a.bs])
        y = torch.from_numpy(nxt_tok[idx]).to(dev)
        lg = (resblock(x, tgt_all[idx]).half() @ Wh.T).float()
        loss = torch.nn.functional.cross_entropy(lg, y)
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if st % 500 == 0 or st == steps - 1:
            with torch.no_grad():
                xv, iv = hb(ho_a[:2048])
                acc = (argmax_head(resblock(xv, tgt_all[iv])) == torch.from_numpy(nxt_tok[iv])).float().mean().item()
            print(f'step {st}/{steps} loss {loss.item():.3f} held-out top1 (2048) {acc:.3f} {time.time()-t2:.0f} s', flush=True)
    train_s = time.time() - t2

    @torch.no_grad()
    def score(rows_e, H_e, ia, ib, label):
        res = {'base': 0, 'n': 0, 'ridge': 0, 'top1': 0, **{f'short{k}': 0 for k in shorts}}
        hit = {'ridge': {}, 'top1': {}, **{f'short{k}': {} for k in shorts}}
        pred_all = np.full(len(rows_e), -1, dtype=np.int32)   # head guess per row, for tools/c3_head_t1.py --head
        for s in range(0, len(ia), 1024):
            i1, i2 = ia[s:s + 1024], ib[s:s + 1024]
            x = torch.from_numpy(np.asarray(H_e[i1], dtype=np.float32)).to(dev)
            y = torch.from_numpy(rows_e[i2, 2].astype(np.int64))
            res['base'] += (argmax_head(x) == torch.from_numpy(rows_e[i1, 2].astype(np.int64))).sum().item()
            g = resblock(x, rows_e[i1, 2].astype(np.int64))
            preds = {'top1': argmax_head(g), **{f'short{k}': argmax_head(g, v) for k, v in shorts.items()}}
            if A is not None:
                preds['ridge'] = argmax_head(x @ A)
            pred_all[i1] = preds['top1'].cpu().numpy()
            for k, p in preds.items():
                ok = (p.cpu() == y).numpy()
                res[k] += int(ok.sum()); hit[k].update(zip(i1.tolist(), ok.tolist()))
            res['n'] += len(i1)
        n = max(res['n'], 1)
        only = set(rows_e[ia, 0].tolist())
        line = f'[{label}] pairs {res["n"]}  base {res["base"]/n:.4f}  top1 {res["top1"]/n:.4f}'
        line += ''.join(f'  short{k} {res[f"short{k}"]/n:.4f}' for k in shorts)
        if A is not None:
            line += f'  ridge {res["ridge"]/n:.4f}'
        line += f'  tau {tau(rows_e, hit["top1"], only):.3f}'
        line += ''.join(f'  tau_short{k} {tau(rows_e, hit[f"short{k}"], only):.3f}' for k in shorts)
        if A is not None:
            line += f'  tau_ridge {tau(rows_e, hit["ridge"], only):.3f}'
        print('RESULT', line, flush=True)
        if label != 'held-out train prompts':
            np.save(f'{a.load or a.out}.{label}.pred.npy', pred_all)

    score(rows, H, ho_a, ho_b, 'held-out train prompts')
    for e in a.eval:
        name, pre = e.split('=', 1)
        re_, He = load_dump(pre, n_embd)
        ea, eb = pairs(re_)
        score(re_, He, ea, eb, name)

    sd = {'W': W.detach().cpu() if a.rank == 0 else None, 'U': U.detach().cpu() if a.rank else None,
          'V': V.detach().cpu() if a.rank else None, 'b': bvec.detach().cpu(),
          'Wt': Wt.detach().cpu() if Wt is not None else None,
          'meta': {'model_sha256': model_sha, 'train_dump': os.path.basename(a.train), 'pairs': int(len(tr_a)),
                   'epochs': a.epochs, 'lr': a.lr, 'wd': a.wd, 'bs': a.bs, 'rank': a.rank, 'tok': a.tok, 'train_seconds': round(train_s),
                   'training_text': 'greedy chat outputs of this model on databricks-dolly-15k prompts '
                                    '(CC BY-SA 3.0), tools/c3_head_prompts.py train split'}}
    if a.load:
        return
    torch.save(sd, a.out)
    print(f'saved {a.out} sha256 {hashlib.sha256(open(a.out, "rb").read()).hexdigest()} model_sha256 {model_sha} '
          f'train {train_s:.0f} s')


if __name__ == '__main__':
    main()
