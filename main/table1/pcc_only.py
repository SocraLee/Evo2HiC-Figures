"""Orca-setting pooled PCC/SPC (log(O/E) vs Orca target, Orca mask) for one tensors/<cell>_<tag>.npz. Usage: pcc_only.py <cell> <tag>"""
import sys, numpy as np
from scipy.stats import pearsonr, spearmanr
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))  # repo root (paths.py)
import paths as P  # noqa: E402
W = str(P.RA_WORK('orca')); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from multimetric import N, IDX, diag_mean, pooled
cell, tag = sys.argv[1], sys.argv[2]
z = np.load(f'{W}/tensors/{cell}.npz'); gt = z['gt_raw'].astype(float); T = z['orca_t_logoe'].astype(float)
valid = (z['orca_t_nan'] <= 0.25) & np.isfinite(T) & np.isfinite(gt)
E = diag_mean(gt, valid); eps = float(np.nanmin(E[E > 0])); Em = E[IDX]
raw = np.load(f'{W}/tensors/{cell}_{tag}.npz')['evo_raw'].astype(float)
lo = np.log((np.clip(raw, 0, None) + eps) / (Em[None] + eps)); lo[~valid] = np.nan; Tm = T.copy(); Tm[~valid] = np.nan
print(f'{cell}\t{tag}\tPCC\t{pooled(lo, Tm, pearsonr):.4f}\tSPC\t{pooled(lo, Tm, spearmanr):.4f}')
