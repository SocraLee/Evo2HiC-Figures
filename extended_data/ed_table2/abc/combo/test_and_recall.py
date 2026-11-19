"""K562 chr9/10, Gasperini full + Morris 2023, distal pairs (>= 50 kb).
(1) Paired gene-level permutation test of AUPRC differences: each method's score is rank-transformed over the distal
    pairs; within each gene the two methods' ranks are swapped with probability 1/2 (10,000 permutations); one-sided
    p = P(permuted difference >= observed).
(2) Bootstrap distribution shape of AUPRC (why the percentile CI is wide and asymmetric).
(3) Recall@k: per-gene (fraction of positives ranked within the top k candidates of their own gene) and
    genome-level (fraction of positives among the top K pairs overall)."""
import numpy as np, pandas as pd
from sklearn.metrics import average_precision_score as ap
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', '..'))  # repo root
import paths as P  # noqa: E402
W = os.path.join(P.RA_WORK('abc'), 'combo'); K = ['chr', 'gene', 'tss', 'enh']
e = pd.read_csv(f'{W}/out_evo/K562_pairs_res10000_pseudo0.tsv', sep='\t'); h = pd.read_csv(f'{W}/out_hic/K562_pairs_res10000_pseudo0.tsv', sep='\t')
for t in (e, h): t['y'] = t.reg.astype(str) == 'True'
agg = lambda t: t.groupby(K, as_index=False).agg({**{c: 'first' for c in t.columns if c not in K + ['y']}, 'y': 'max'})
e, h = agg(e), agg(h); d = e.merge(h[K + ['abc_pred_std']].rename(columns={'abc_pred_std': 'abc_hic'}), on=K)
x = d[d.dist >= 5e4].reset_index(drop=True)
S = {'distance': 's_dist', 'contact': 's_contact', 'Evo2HiC': 'abc_pred_std', 'Hi-C-only': 'abc_hic', 'measured': 'abc_meas_std'}
print(f'distal pairs {len(x)}, positives {int(x.y.sum())}, genes {x.gene.nunique()}, genes with a positive {x[x.y].gene.nunique()}')
R = {k: x[c].rank(method='average').values for k, c in S.items()}
y = x.y.values; gid = pd.factorize(x.gene)[0]; G = gid.max() + 1
rng = np.random.default_rng(0)
print('\n(1) permutation test, AUPRC difference (one-sided, first > second):')
for a, b in [('Evo2HiC', 'distance'), ('Evo2HiC', 'contact'), ('Evo2HiC', 'Hi-C-only'), ('Hi-C-only', 'distance')]:
    obs = ap(y, R[a]) - ap(y, R[b]); null = []
    for _ in range(10000):
        sw = rng.random(G) < 0.5; m = sw[gid]
        ra = np.where(m, R[b], R[a]); rb = np.where(m, R[a], R[b]); null.append(ap(y, ra) - ap(y, rb))
    null = np.array(null); print(f'   {a} vs {b}: diff {obs:+.3f}, p = {(np.sum(null >= obs) + 1) / (len(null) + 1):.4f}')
print('\n(2) gene-level bootstrap of Evo2HiC distal AUPRC (deciles):')
g = x.gene.unique(); grp = dict(tuple(x.groupby('gene'))); bs = []
for _ in range(2000):
    xb = pd.concat([grp[k] for k in rng.choice(g, len(g))])
    if xb.y.any(): bs.append(ap(xb.y, xb.abc_pred_std))
print('   ', np.round(np.percentile(bs, [2.5, 10, 25, 50, 75, 90, 97.5]), 3), ' (point estimate', round(ap(x.y, x.abc_pred_std), 3), ')')
print('\n(3) recall@k')
def per_gene(c, k):
    hit = tot = 0
    for _, z in x.groupby('gene'):
        if z.y.any(): hit += z.nlargest(min(k, len(z)), c).y.sum(); tot += z.y.sum()
    return hit / tot
rows = []
for k, c in S.items():
    r = {'method': k}
    for kk in (1, 3, 5, 10): r[f'per-gene top{kk}'] = per_gene(c, kk)
    o = x.sort_values(c, ascending=False)
    for KK in (10, 25, 50, 100): r[f'overall top{KK}'] = o.head(KK).y.sum() / x.y.sum()
    rows.append(r)
print(pd.DataFrame(rows).round(3).to_string(index=False))
