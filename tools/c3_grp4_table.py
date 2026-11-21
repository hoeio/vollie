# host, offline: memo-06 T1 tables. From the ffn_down inputs dumped by `C3_ACTDUMP=<prefix> c3_colskip.py calib ...`
# (wikitext-2 train), build per layer a neuron order in groups of four and the train quantiles of each group's largest
# |a|. 'file': groups are rows 4i..4i+3 as in the file (the 4x4 repack groups). 'clustered': greedy, each group is a
# seed neuron (file order) plus the 3 unassigned neurons whose |a| correlates most with it over the train tokens
# (a row permutation is exact on gate/up; down then needs its columns in the same order).
# usage: python3 -I tools/c3_grp4_table.py <acts.npz> <file|clustered> <out.npz>   (no args: self-check)
# c3_colskip.py calib with C3_ACTDUMP=<prefix> calls table() in memory and writes <prefix>-file.npz and <prefix>-clustered.npz
import sys, numpy as np
QS = (0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6)

def cluster(A):   # A: tokens x F, |a| as float32. ponytail: O(F^2) memory (1.4 GB at F = 18944), fine for 7B FFNs
    Z = (A - A.mean(0)) / (A.std(0) + 1e-12); C = (Z.T @ Z) / len(Z); F = C.shape[0]
    np.fill_diagonal(C, -np.inf); free = np.ones(F, bool); out = []
    for i in range(F):
        if not free[i]: continue
        free[i] = False; C[:, i] = -np.inf
        j = np.argpartition(-C[i], 3)[:3]; free[j] = False; C[:, j] = -np.inf
        out += [i, *j.tolist()]
    return np.array(out)

def table(acts, order):
    res = {'qs': np.array(QS)}
    for k in sorted(acts.keys(), key=lambda k: int(k[4:])):
        l = int(k[4:]); A = np.abs(acts[k].astype(np.float32)); F = A.shape[1]; assert F % 4 == 0
        P = np.arange(F) if order == 'file' else cluster(A)
        assert np.array_equal(np.sort(P), np.arange(F))
        res['perm:%d' % l] = P; res['thr:%d' % l] = np.quantile(A[:, P].reshape(len(A), -1, 4).max(2), QS)
        print(l, end=' ', flush=True)
    return res

if __name__ == '__main__':
    if len(sys.argv) == 1:   # self-check: two perfectly co-active quartets are found by 'clustered'
        rng = np.random.default_rng(0); base = rng.random((256, 2)); A = np.repeat(base, 4, axis=1)[:, [0, 4, 1, 5, 2, 6, 3, 7]] + 1e-3 * rng.random((256, 8))
        P = cluster(A.astype(np.float32)); assert sorted(P[:4] % 2) == [0, 0, 0, 0] and sorted(P[4:] % 2) == [1, 1, 1, 1], P; print('selftest ok')
    else:
        np.savez(sys.argv[3], **table(np.load(sys.argv[1]), sys.argv[2])); print('\nwrote', sys.argv[3])
