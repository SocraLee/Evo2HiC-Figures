#!/usr/bin/env python
"""Threshold-free TAD-boundary comparison.  The fixed prominence cut (>= 0.2) makes
F1 depend on each model's contrast: Orca's insulation dips are ~1.5x deeper than the
data's, so more of them pass the cut (recall up, precision down); Evo2HiC's are
slightly shallower (the reverse).  Here each model's boundaries are ranked by their
own prominence and (i) the precision-recall curve against GT boundaries (GT at the
paper's >= 0.2 cut, +-3-bin matching) is integrated to an AUPRC per window, and
(ii) F1 is evaluated at matched calibration, i.e. each model's top-k boundaries with
k = number of GT boundaries in that window.
Usage: boundary_pr.py <cell> <ours|orca>  -> results/boundary_pr_<cell>_<gt>.tsv
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))  # repo root (paths.py)
import paths as P  # noqa: E402

W = str(P.RA_WORK('orca'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from is_diag import profile_and_bounds  # noqa: E402

N = 250
IDX = np.abs(np.arange(N)[:, None] - np.arange(N)[None, :])


def match(prd, tgt, z=3):
    if len(tgt) == 0:
        return np.nan, np.nan
    if len(prd) == 0:
        return 0.0, 0.0
    pm = np.array([any(abs(p - t) <= z for t in tgt) for p in prd])
    tm = np.array([any(abs(p - t) <= z for p in prd) for t in tgt])
    return pm.mean(), tm.mean()


def pr_auc(prom, gt_b):
    """rank all local-minimum candidates (finite prominence) by prominence; PR curve."""
    cand = np.where(np.isfinite(prom))[0]
    if len(cand) == 0 or len(gt_b) == 0:
        return np.nan, np.nan
    order = cand[np.argsort(-prom[cand])]
    P, R = [], []
    for k in range(1, len(order) + 1):
        p, r = match(order[:k], gt_b)
        P.append(p)
        R.append(r)
    R, P = np.array([0] + R), np.array([1] + P)
    auc = float(np.trapz(P, R))
    k = len(gt_b)
    p, r = match(order[:k], gt_b)
    f1k = 0.0 if (p + r) == 0 else 2 * p * r / (p + r)
    return auc, f1k


def main(cell, gtname):
    z = np.load(f'{W}/tensors/{cell}.npz')
    chrom = z['regions'][:, 0]
    E = np.load(f'{W}/tensors/expected_4kb_{cell}.npy')[:N]
    eps = float(E.min())
    Em = E[IDX]
    gt_ours = z['gt_raw'].astype(float)
    evo = z['evo_raw'].astype(float)
    orca = np.exp(z['orca_logoe'].astype(float)) * (Em + eps) - eps
    gt = gt_ours if gtname == 'ours' else np.exp(z['orca_t_logoe'].astype(float)) * (Em + eps) - eps
    valid = (z['orca_t_nan'] <= 0.25) & np.isfinite(z['orca_t_logoe']) & np.isfinite(gt_ours)
    rec = []
    for i in range(len(gt)):
        m = valid[i]
        if m.sum() < 1000:
            continue
        _, bg, _ = profile_and_bounds(np.where(m, gt[i], 0))
        if len(bg) == 0:
            continue
        _, _, pe = profile_and_bounds(np.where(m, evo[i], 0))
        _, _, po = profile_and_bounds(np.where(m, orca[i], 0))
        ae, fe = pr_auc(pe, bg)
        ao, fo = pr_auc(po, bg)
        rec.append(dict(win=i, chrom=chrom[i], n_gt=len(bg), auprc_evo=ae, auprc_orca=ao, f1k_evo=fe, f1k_orca=fo))
        if i % 100 == 0:
            print(f'[{cell}/{gtname}] {i}/417', flush=True)
    df = pd.DataFrame(rec)
    os.makedirs(f'{W}/results', exist_ok=True)
    df.to_csv(f'{W}/results/boundary_pr_{cell}_{gtname}.tsv', sep='\t', index=False, float_format='%.4g')
    for ch in [9, 10, 'all']:
        g = df if ch == 'all' else df[df.chrom == ch]
        for a, b in [('auprc_evo', 'auprc_orca'), ('f1k_evo', 'f1k_orca')]:
            ok = np.isfinite(g[a]) & np.isfinite(g[b])
            d = (g[a] - g[b])[ok]
            print(f'{cell} GT={gtname} chr={ch} n={ok.sum()} {a[:-4]:6s} Evo={g[a][ok].mean():.3f} Orca={g[b][ok].mean():.3f} '
                  f'Evo wins {(d>0).sum()} / Orca wins {(d<0).sum()}  p(Evo>Orca)={wilcoxon(d, alternative="greater").pvalue:.2g} '
                  f'p(Orca>Evo)={wilcoxon(d, alternative="less").pvalue:.2g}')


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
