#!/usr/bin/env python
"""Supplementary Data 2: per-species results on the 177 DNA Zoo species.
cross_species_per_species.tsv (cross_species_summary.py) plus boundaries called, boundary precision and recall
(prominence >= 0.2, match within 3 bins) of Evo2HiC, Evo2HiC on random DNA and the Hi-C-only ablation, from the
boundary candidates of tad_random.py (real / random DNA) and tad_dump.py (Hi-C-only).
Output: Supplementary_Data_2.tsv"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))  # repo root (paths.py)
import paths as P  # noqa: E402
import glob
import numpy as np, pandas as pd
S = str(P.RA_WORK('species'))


def near(a, b, t=3):
    if len(a) == 0 or len(b) == 0:
        return np.zeros(len(a), bool)
    b = np.sort(b); k = np.searchsorted(b, a); dd = np.full(len(a), 10 ** 9)
    for kk in (k - 1, k):
        ok = (kk >= 0) & (kk < len(b)); dd[ok] = np.minimum(dd[ok], np.abs(b[kk[ok]] - a[ok]))
    return dd <= t


def stats(r, p, gt, name):
    r[f'n_boundaries_called_{name}'] = len(p)
    r[f'boundary_precision_{name}'] = near(p, gt).mean() if len(p) else np.nan
    r[f'boundary_recall_{name}'] = near(gt, p).mean() if len(gt) else np.nan


rows = []
for f in sorted(glob.glob(f'{S}/tad_candidates_random/*.npz')):
    sp = os.path.basename(f)[:-4]; y = np.load(f); gt = y['gt_idx'][y['gt_bs'] >= 0.2]; r = {'species': sp}
    for a, name in (('real', 'evo2hic'), ('random', 'random')):
        stats(r, y[f'evo2hic_{a}_idx'][y[f'evo2hic_{a}_bs'] >= 0.2], gt, name)
    g = f'{S}/tad_candidates/{sp}.npz'
    if os.path.exists(g) and 'hic_only_idx' in np.load(g):
        z = np.load(g); stats(r, z['hic_only_idx'][z['hic_only_bs'] >= 0.2], z['gt_idx'][z['gt_bs'] >= 0.2], 'hic_only')
    rows.append(r)
B = pd.DataFrame(rows)
if 'n_boundaries_called_hic_only' not in B:     # Hi-C-only outputs not released: shipped per-species values
    base = pd.read_csv(P.RA_SHIPPED('species', 'boundary_baselines.tsv'), sep='\t')
    B = B.merge(base[['species'] + [f'{k}_hic_only' for k in ('n_boundaries_called', 'boundary_precision', 'boundary_recall')]],
                on='species', how='left')
d = pd.read_csv(f'{S}/cross_species_per_species.tsv', sep='\t')
d = d[[c for c in d.columns if not c.startswith('AP_')]].merge(B, on='species', how='left')
d = d.rename(columns={'mya': 'patristic_distance_to_human_Myr', 'contacts_per_bin': 'contacts_per_10kb_bin',
                      'genome_len': 'evaluated_scaffold_length_bp', 'n_bins': 'n_bins_10kb', 'n_bounds_gt': 'n_groundtruth_boundaries'})
d.to_csv(f'{S}/Supplementary_Data_2.tsv', sep='\t', index=False, float_format='%.4g')
print('Supplementary_Data_2.tsv:', d.shape)
# Results: boundary precision / recall of Evo2HiC vs the Hi-C-only ablation in placental mammals
m = d[d.clade == 'Mammalia']
for k in ('precision', 'recall'):
    print(f'placental mammals (n={len(m)}) boundary {k}: Evo2HiC {m[f"boundary_{k}_evo2hic"].mean():.3f}, '
          f'Hi-C-only {m[f"boundary_{k}_hic_only"].mean():.3f}')
