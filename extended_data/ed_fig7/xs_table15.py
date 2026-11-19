#!/usr/bin/env python
"""Supplementary Table (cross-species, redesigned).
(a) Contribution of the genome sequence: Evo2HiC vs the same checkpoint on random DNA, by clade ordered by
    divergence from human (placental mammals split into primates and other placentals as in tad_ctcf_summary.py).
(b) Comparison with other methods by clade and over all species. Paired two-sided Wilcoxon signed-rank tests over species.
Input: cross_species_per_species.tsv (cross_species_summary.py). Output: xs_table15.tsv."""
import numpy as np, pandas as pd
from scipy.stats import wilcoxon
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))  # repo root (paths.py)
import paths as P  # noqa: E402
S = str(P.RA_WORK('species'))
d = pd.read_csv(f'{S}/cross_species_per_species.tsv', sep='\t')
def grp(r):
    if r.clade != 'Mammalia': return r.clade
    return 'Primates' if (pd.notna(r.mya) and r.mya <= 147.6) else 'Other placental mammals'
d['grp'] = d.apply(grp, axis=1)
import glob, os
def near(a, b, t=3):
    if len(a) == 0 or len(b) == 0: return np.zeros(len(a), bool)
    b = np.sort(b); k = np.searchsorted(b, a); dd = np.full(len(a), 10 ** 9)
    for kk in (k - 1, k):
        ok = (kk >= 0) & (kk < len(b)); dd[ok] = np.minimum(dd[ok], np.abs(b[kk[ok]] - a[ok]))
    return dd <= t
pr = []
for f in glob.glob(f'{S}/tad_candidates_random/*.npz'):
    y = np.load(f); gt = y['gt_idx'][y['gt_bs'] >= 0.2]; r = {'species': os.path.basename(f)[:-4]}
    for a, c in (('real', 'evo2hic'), ('random', 'random')):
        p = y[f'evo2hic_{a}_idx'][y[f'evo2hic_{a}_bs'] >= 0.2]; r[f'nB_{c}'] = len(p); r[f'Prec_{c}'] = near(p, gt).mean() if len(p) else np.nan
    pr.append(r)
d = d.merge(pd.DataFrame(pr), on='species', how='left')
ph = []
for f in glob.glob(f'{S}/tad_candidates/*.npz'):
    z = np.load(f); gt = z['gt_idx'][z['gt_bs'] >= 0.2]; r = {'species': os.path.basename(f)[:-4]}
    for c in ('evo2hic', 'hic_only'):
        if f'{c}_idx' not in z: continue        # Hi-C-only outputs not released: shipped values below
        p = z[f'{c}_idx'][z[f'{c}_bs'] >= 0.2]; r[f'nBm_{c}'] = len(p); r[f'Precm_{c}'] = near(p, gt).mean() if len(p) else np.nan
    ph.append(r)
d = d.merge(pd.DataFrame(ph), on='species', how='left')
if 'nBm_hic_only' not in d:
    base = pd.read_csv(P.RA_SHIPPED('species', 'boundary_baselines.tsv'), sep='\t')
    d = d.merge(base[['species', 'n_boundaries_called_hic_only', 'boundary_precision_hic_only']].rename(
        columns={'n_boundaries_called_hic_only': 'nBm_hic_only', 'boundary_precision_hic_only': 'Precm_hic_only'}), on='species', how='left')
ORDER = ['Primates', 'Other placental mammals', 'Marsupialia', 'Reptilia', 'Actinopterygii', 'Protostomes', 'Angiosperms']
def pw(a, b):
    ok = a.notna() & b.notna(); x = (a - b)[ok]
    if len(x) < 2 or not (x != 0).any(): return np.nan, int((a[ok] > b[ok]).sum()), int(ok.sum())
    return wilcoxon(x).pvalue, int((a[ok] > b[ok]).sum()), int(ok.sum())
rows = []
for ref, cols in (('random', {'PCC': ('PCC_evo2hic', 'PCC_random'), 'nB': ('nB_evo2hic', 'nB_random'), 'Prec': ('Prec_evo2hic', 'Prec_random'), 'TAD_f1': ('TAD_f1_evo2hic', 'TAD_f1_random')}),
                  ('hic_only', {'PCC': ('PCC_evo2hic', 'PCC_hic_only'), 'nB': ('nBm_evo2hic', 'nBm_hic_only'), 'Prec': ('Precm_evo2hic', 'Precm_hic_only'), 'TAD_f1': ('TAD_f1_evo2hic', 'TAD_f1_hic_only')})):
    for g in ORDER:
        x = d[d.grp == g]; r = {'part': 'a' if ref == 'random' else 'b', 'group': g, 'n': len(x)}
        for lab, c in (('pixel PCC', 'PCC'), ('boundaries called', 'nB'), ('precision', 'Prec'), ('TAD F1', 'TAD_f1')):
            a, b = x[cols[c][0]], x[cols[c][1]]; p, w, n = pw(a, b)
            r.update({f'{lab} Evo2HiC': a.mean(), f'{lab} {ref}': b.mean(), f'{lab} wins': f'{w}/{n}', f'{lab} p': p})
        rows.append(r)
for lab, c in (('Pixel PCC', 'PCC'), ('TAD-boundary $F_1$', 'TAD_f1')):
    for g in ORDER + ['All']:
        x = d if g == 'All' else d[d.grp == g]
        r = {'part': 'b', 'metric': lab, 'group': g}
        for m in ('hicarn2', 'hicfoundation'):
            p, w, n = pw(x[f'{c}_evo2hic'], x[f'{c}_{m}']); r[f'wins vs {m}'] = f'{w}/{n}'; r[f'p vs {m}'] = p
        rows.append(r)
out = pd.DataFrame(rows)
out.to_csv(f'{S}/xs_table15.tsv', sep='\t', index=False, float_format='%.4g')
print(out.to_string())
