#!/usr/bin/env python
"""Threshold-free boundary AUPRC (paper protocol candidates, no threshold on the prediction).
Ground truth: cooltools boundaries of the full-coverage map with prominence >= 0.2. Prediction: all cooltools boundary
candidates ranked by prominence; matching as in the paper's F1 (within 3 bins of any
ground-truth boundary, and recall = ground-truth boundaries with a prediction within 3 bins); AP = sum_k (R_k - R_{k-1}) P_k.
Inputs: tad_candidates/<sp>.npz (evo2hic, hic_only, hicarn2, hicfoundation), tad_candidates_random/<sp>.npz (random DNA)
Output: boundary_auprc_check.tsv, boundary_auprc_all_methods.tsv (with clade; needs species_table.tsv)"""
import glob, os
import numpy as np, pandas as pd
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))  # repo root (paths.py)
import paths as P  # noqa: E402
S = str(P.RA_WORK('species'))


def ap(idx, bs, gt, tol=3):
    # same matching rule as the paper's F1: a prediction is correct if within `tol` bins of any ground-truth boundary;
    # recall = fraction of ground-truth boundaries with a prediction within `tol` bins among the top-k
    if len(gt) == 0:
        return np.nan
    idx = idx[np.argsort(-bs)]
    hit = np.array([np.min(np.abs(gt - i)) <= tol for i in idx])
    first = np.full(len(gt), np.inf)
    for k, i in enumerate(idx):
        m = np.abs(gt - i) <= tol
        first[m & np.isinf(first)] = k
    out, prev_r = 0.0, 0.0
    for k in range(len(idx)):
        r = np.sum(first <= k) / len(gt)
        if r > prev_r:
            out += (r - prev_r) * hit[:k + 1].mean(); prev_r = r
    return out


rows = []
for f in sorted(glob.glob(f'{S}/tad_candidates/*.npz')):
    sp = os.path.basename(f)[:-4]; z = np.load(f); gt = z['gt_idx'][z['gt_bs'] >= 0.2]
    r = {'species': sp}
    for m in ('evo2hic', 'hic_only', 'hicarn2', 'hicfoundation'):
        if f'{m}_idx' in z:                     # baselines without released outputs: filled from results/ below
            r[m] = ap(z[f'{m}_idx'], z[f'{m}_bs'], gt)
    g = f'{S}/tad_candidates_random/{sp}.npz'
    if os.path.exists(g):
        y = np.load(g); r['evo2hic_randomDNA'] = ap(y['evo2hic_random_idx'], y['evo2hic_random_bs'], y['gt_idx'][y['gt_bs'] >= 0.2])
    rows.append(r)
d = pd.DataFrame(rows)
base = pd.read_csv(P.RA_SHIPPED('species', 'boundary_baselines.tsv'), sep='\t')
for m in ('hic_only', 'hicarn2'):
    if m not in d:
        d = d.merge(base[['species', f'AP_{m}']].rename(columns={f'AP_{m}': m}), on='species', how='left')
d = d[['species', 'evo2hic', 'hic_only', 'hicarn2', 'hicfoundation', 'evo2hic_randomDNA']]
d.to_csv(f'{S}/boundary_auprc_check.tsv', sep='\t', index=False, float_format='%.4g')
# full-precision table with the clade column, read by cross_species_summary.py (run species_analysis.py first)
clade = pd.read_csv(f'{S}/species_table.tsv', sep='\t')[['species', 'clade']]
clade.merge(d, on='species').to_csv(f'{S}/boundary_auprc_all_methods.tsv', sep='\t', index=False)
print('done', len(rows))
