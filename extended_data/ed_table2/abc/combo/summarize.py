"""Held-out chromosomes 9/10, full Gasperini 2019 + Morris 2023 (STING-seq, ENCODE held-out set); at-scale screen (all DHS-gene pairs 6 kb-1 Mb, FDR<0.1 & decreased
expression = positive), lifted to hg38. ABC protocol as in the main analysis (10 kb SCALE, no pseudocount, GM12878 QN).
AUPRC with 1,000 gene-level bootstrap CIs; Evo2HiC vs Hi-C-only paired difference."""
import numpy as np, pandas as pd
from sklearn.metrics import average_precision_score as ap
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', '..'))  # repo root
import paths as P  # noqa: E402
W = os.path.join(P.RA_WORK('abc'), 'combo'); K = ['chr', 'gene', 'tss', 'enh']
e = pd.read_csv(f'{W}/out_evo/K562_pairs_res10000_pseudo0.tsv', sep='\t'); h = pd.read_csv(f'{W}/out_hic/K562_pairs_res10000_pseudo0.tsv', sep='\t')
# an element tested by several gRNA groups appears once per group: collapse to one pair, positive if any group is
for t in (e, h): t['y'] = t.reg.astype(str) == 'True'
agg = lambda t: t.groupby(K, as_index=False).agg({**{c: 'first' for c in t.columns if c not in K + ['y']}, 'y': 'max'})
e, h = agg(e), agg(h)
d = e.merge(h[K + ['abc_pred_std']].rename(columns={'abc_pred_std': 'abc_hic'}), on=K); assert len(d) == len(e) == len(h)
S = {'distance': 's_dist', 'contact': 's_contact', 'ABC Hi-C-only': 'abc_hic', 'ABC Evo2HiC': 'abc_pred_std', 'ABC measured': 'abc_meas_std'}
rng = np.random.default_rng(0)
for rn, m in [('all >=6kb', d.dist >= 0), ('<50kb', d.dist < 5e4), ('>=50kb', d.dist >= 5e4)]:
    x = d[m]; g = x.gene.unique(); grp = dict(tuple(x.groupby('gene'))); B = {k: [] for k in S}; D = []; Dd = {}
    for _ in range(1000):
        xb = pd.concat([grp[k] for k in rng.choice(g, len(g))])
        if not xb.y.any(): continue
        a = {k: ap(xb.y, xb[c]) for k, c in S.items()}
        for k in S: B[k].append(a[k])
        D.append(a['ABC Evo2HiC'] - a['ABC Hi-C-only']); Dd.setdefault('Evo2HiC - distance', []).append(a['ABC Evo2HiC'] - a['distance'])
        Dd.setdefault('Evo2HiC - contact', []).append(a['ABC Evo2HiC'] - a['contact']); Dd.setdefault('Hi-C-only - distance', []).append(a['ABC Hi-C-only'] - a['distance'])
    print(f'{rn:9s} pairs={len(x)} pos={int(x.y.sum())} genes={len(g)} prevalence={x.y.mean():.4f}')
    for k, c in S.items(): print(f'    {k:14s} {ap(x.y, x[c]):.3f} [{np.percentile(B[k], 2.5):.3f}, {np.percentile(B[k], 97.5):.3f}]')
    D = np.array(D); print(f'    Evo2HiC - Hi-C-only {np.median(D):+.3f} [{np.percentile(D, 2.5):+.3f}, {np.percentile(D, 97.5):+.3f}] P(>0)={np.mean(D > 0):.3f}')
    for k, v in Dd.items(): v = np.array(v); print(f'    {k:20s} {np.median(v):+.3f} [{np.percentile(v, 2.5):+.3f}, {np.percentile(v, 97.5):+.3f}] P(>0)={np.mean(v > 0):.3f}')
