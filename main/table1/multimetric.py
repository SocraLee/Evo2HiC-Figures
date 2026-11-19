#!/usr/bin/env python
"""Matched multi-metric Evo2HiC-vs-Orca comparison (reviewer 2, major point 2).

Reviewer 2: evaluate both models under BOTH settings (our raw-count
distance-stratified metric and Orca's log(O/E) metric) on the same test regions
and preprocessing, and add the C.Origami panel (insulation-score correlation,
observed/expected correlation, MSE, distance-stratified correlation) computed
per chromosome.

Test regions: Orca's 417 test windows (chr9: 213, chr10: 204; 1 Mb at 4 kb),
identical for every metric.  Pixel mask: Orca's own validity mask (target
NaN-fraction <= 0.25 in both cells, as in Orca's notebook) intersected with the
finite pixels of the ground truth in use - the same pixels for both models and
all metrics.

Two ground-truth settings (each with its native representation):
  gt=orca  Orca's own target tensor (Micro-C 4DNFI9GMP2J8 / 4DNFI643OYP9, adaptive
           coarse-grained, log(O/E) with Orca's smoothed expected E_orca).  This
           reproduces Orca's published PCC/SPC exactly.  Raw counts for this GT
           are obtained by inverting Orca's transform, exp(t)*(E_orca+eps)-eps
           (balanced mcool scale).
  gt=ours  Evo2HiC's training target (.hic 4DNFI2TK7L2F / 4DNFIPC7P27B, SCALE
           balanced, 2 kb summed to 4 kb; count scale).  log(O/E) uses E_ours =
           per-diagonal mean of this GT over the 417 windows, eps = min(E_ours).
Predictions: Evo2HiC raw counts (count scale) with log(O/E) via E_ours; Orca
log(O/E) as released, raw via Orca's inverse transform (mcool scale).  The two
scales differ by ~4 orders of magnitude, so for scale-dependent quantities (MSE,
log1p) the prediction that is not on the GT's scale is rescaled per diagonal
offset by s_k = mean_k(GT_ours)/mean_k(GT_orca) over valid pixels.  Correlations
are unaffected by this.

Spaces:
  raw       (scale-matched) balanced counts
  logoe     dimensionless log(O/E), each model with its own expected
  corigami  C.Origami's protocol (Tan et al. 2023 Methods): log(x+1) on the
            scale-matched counts, then distance-stratified normalization of each
            prediction to the GT's per-diagonal mean/s.d.
Metrics (C.Origami released code, metrics.py, verbatim logic):
  insulation_PCC  per window, Pearson of C.Origami chr_score insulation profiles
                  (radius 500 kb, pseudocount 30 x matrix mean) - NOTE: on raw
                  counts this score is dominated by distance decay and saturates
                  near 1 for every model; it is informative in logoe/corigami.
  OE_PCC          per window, Pearson after subtracting the mean matrix over windows
  MSE             per window
  dist_strat_PCC  per window, Pearson per diagonal offset (1..249), averaged
Headline metrics of the two papers:
  pooled_PCC/SPC       all valid pixels pooled (Orca's metric, in logoe)
  diag_pooled_SPC_k0 / _k5  per-offset Spearman pooled over windows, mean over
                       offsets >= 0 / >= 5 (Evo2HiC's metric; Fig 3 uses >= 5)
Output: results/multimetric_<cell>.tsv ; usage: multimetric.py <cell>
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr, wilcoxon
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))  # repo root (paths.py)
import paths as P  # noqa: E402

W = str(P.RA_WORK('orca'))
sys.path.insert(0, f'{P.CORIGAMI_DIR}/src/corigami/inference/utils/metrics')
import insulation_score as insu  # noqa: E402

N, RES = 250, 4000
IDX = np.abs(np.arange(N)[:, None] - np.arange(N)[None, :])


def diag_mean(x, valid):
    """per-offset mean over valid pixels of all windows -> (N,)"""
    out = np.full(N, np.nan)
    for k in range(N):
        v = np.concatenate([np.diagonal(a, k) for a in x])
        m = np.concatenate([np.diagonal(a, k) for a in valid])
        if m.sum():
            out[k] = np.nanmean(v[m])
    return out


def ds_normalize(pred, gt):
    out = pred.copy()
    for k in range(N):
        for sign in (1, -1) if k else (1,):
            idx = np.arange(0, N - k)
            pi, pj = (idx, idx + k) if sign == 1 else (idx + k, idx)
            p, g = pred[:, pi, pj], gt[:, pi, pj]
            pm, ps, gm, gs = np.nanmean(p), np.nanstd(p), np.nanmean(g), np.nanstd(g)
            out[:, pi, pj] = (p - pm) / (ps + 1e-9) * gs + gm if ps > 0 else gm
    return out


def per_window_corr(a, b):
    out = []
    for x, y in zip(a, b):
        m = np.isfinite(x) & np.isfinite(y)
        out.append(pearsonr(x[m], y[m])[0] if m.sum() > 2 and x[m].std() > 0 and y[m].std() > 0 else np.nan)
    return np.array(out)


def insulation_pearson(preds, targets):
    out = []
    for p, t in zip(preds, targets):
        if not np.isfinite(t).any():
            out.append(np.nan)
            continue
        pi = np.array(insu.chr_score(np.nan_to_num(p), res=RES))
        ti = np.array(insu.chr_score(np.nan_to_num(t), res=RES))
        m = np.isfinite(pi) & np.isfinite(ti)
        out.append(pearsonr(pi[m], ti[m])[0] if m.sum() > 2 and pi[m].std() > 0 else np.nan)
    return np.array(out)


def observed_vs_expected(preds, targets):
    pm, tm = np.nanmean(preds, 0, keepdims=True), np.nanmean(targets, 0, keepdims=True)
    return per_window_corr((preds - pm).reshape(len(preds), -1), (targets - tm).reshape(len(targets), -1))


def mse(preds, targets):
    return np.array([np.nanmean(d) if np.isfinite(d).any() else np.nan for d in (preds - targets) ** 2])


def distance_stratified(preds, targets):
    out = []
    for p, t in zip(preds, targets):
        cs = []
        for k in range(1, N):
            a, b = np.diagonal(p, k), np.diagonal(t, k)
            m = np.isfinite(a) & np.isfinite(b)
            if m.sum() >= 3 and a[m].std() > 0 and b[m].std() > 0:
                cs.append(pearsonr(a[m], b[m])[0])
        out.append(np.mean(cs) if cs else np.nan)
    return np.array(out)


def pooled_diag_spc(preds, targets, k0):
    cs = []
    for k in range(k0, N):
        a = np.concatenate([np.diagonal(p, k) for p in preds])
        b = np.concatenate([np.diagonal(t, k) for t in targets])
        m = np.isfinite(a) & np.isfinite(b)
        if m.sum() > 10 and a[m].std() > 0 and b[m].std() > 0:
            cs.append(spearmanr(a[m], b[m])[0])
    return float(np.nanmean(cs))


def pooled(a, b, fn):
    x, y = a.ravel(), b.ravel()
    m = np.isfinite(x) & np.isfinite(y)
    return float(fn(x[m], y[m])[0])


def main(cell):
    z = np.load(f'{W}/tensors/{cell}.npz')
    chrom = z['regions'][:, 0]
    E_orca = np.load(f'{W}/tensors/expected_4kb_{cell}.npy')[:N]
    eps_orca = float(E_orca.min())
    Emat_orca = E_orca[IDX]

    evo_raw = z['evo_raw'].astype(np.float64)
    gt_ours = z['gt_raw'].astype(np.float64)
    orca_logoe = z['orca_logoe'].astype(np.float64)
    orca_t = z['orca_t_logoe'].astype(np.float64)
    orca_valid = (z['orca_t_nan'] <= 0.25) & np.isfinite(orca_t)
    orca_raw = np.exp(orca_logoe) * (Emat_orca + eps_orca) - eps_orca      # mcool scale
    orcagt_raw = np.exp(orca_t) * (Emat_orca + eps_orca) - eps_orca        # mcool scale (cg)

    valid = orca_valid & np.isfinite(gt_ours)                              # same pixels everywhere
    E_ours = diag_mean(gt_ours, valid)
    eps_ours = float(np.nanmin(E_ours[E_ours > 0]))
    Emat_ours = E_ours[IDX]
    evo_logoe = np.log((np.clip(evo_raw, 0, None) + eps_ours) / (Emat_ours[None] + eps_ours))
    gt_ours_logoe = np.log((np.clip(gt_ours, 0, None) + eps_ours) / (Emat_ours[None] + eps_ours))
    s_k = E_ours / diag_mean(orcagt_raw, valid)                            # count scale / mcool scale
    S = s_k[IDX][None]

    def mk(x):
        y = x.copy()
        y[~valid] = np.nan
        return y

    settings = {
        'orca': dict(gt_raw=mk(orcagt_raw), gt_logoe=mk(orca_t),
                     evo_raw=mk(evo_raw / S), orca_raw=mk(orca_raw)),
        'ours': dict(gt_raw=mk(gt_ours), gt_logoe=mk(gt_ours_logoe),
                     evo_raw=mk(evo_raw), orca_raw=mk(orca_raw * S)),
    }
    rows = []
    for gt_name, d in settings.items():
        log1p_gt = np.log1p(np.clip(d['gt_raw'], 0, None))
        spaces = {
            'raw': (d['gt_raw'], d['evo_raw'], d['orca_raw']),
            'logoe': (d['gt_logoe'], mk(evo_logoe), mk(orca_logoe)),
            'corigami': (log1p_gt,
                         ds_normalize(np.log1p(np.clip(d['evo_raw'], 0, None)), log1p_gt),
                         ds_normalize(np.log1p(np.clip(d['orca_raw'], 0, None)), log1p_gt)),
        }
        for sp, (g, e, o) in spaces.items():
            per = {'insulation_PCC': (insulation_pearson(e, g), insulation_pearson(o, g)),
                   'OE_PCC': (observed_vs_expected(e, g), observed_vs_expected(o, g)),
                   'MSE': (mse(e, g), mse(o, g)),
                   'dist_strat_PCC': (distance_stratified(e, g), distance_stratified(o, g))}
            for ch in [9, 10, 'all']:
                sel = np.ones(len(chrom), bool) if ch == 'all' else chrom == ch
                for m, (ve, vo) in per.items():
                    a, b = ve[sel], vo[sel]
                    ok = np.isfinite(a) & np.isfinite(b)
                    diff = (b - a) if m == 'MSE' else (a - b)
                    p = wilcoxon(diff[ok], alternative='greater').pvalue if ok.sum() > 5 and (diff[ok] != 0).any() else np.nan
                    rows.append(dict(cell=cell, gt=gt_name, space=sp, metric=m, chrom=str(ch), n=int(ok.sum()),
                                     Evo2HiC=float(np.mean(a[ok])), Orca=float(np.mean(b[ok])),
                                     Evo2HiC_wins=int((diff[ok] > 0).sum()), p_Evo2HiC_better=p))
                gs, es, os_ = g[sel], e[sel], o[sel]
                rows.append(dict(cell=cell, gt=gt_name, space=sp, metric='pooled_PCC', chrom=str(ch), n=int(sel.sum()),
                                 Evo2HiC=pooled(es, gs, pearsonr), Orca=pooled(os_, gs, pearsonr)))
                rows.append(dict(cell=cell, gt=gt_name, space=sp, metric='pooled_SPC', chrom=str(ch), n=int(sel.sum()),
                                 Evo2HiC=pooled(es, gs, spearmanr), Orca=pooled(os_, gs, spearmanr)))
                for k0 in (0, 5):
                    rows.append(dict(cell=cell, gt=gt_name, space=sp, metric=f'diag_pooled_SPC_k{k0}', chrom=str(ch),
                                     n=int(sel.sum()), Evo2HiC=pooled_diag_spc(es, gs, k0), Orca=pooled_diag_spc(os_, gs, k0)))
            print(f'[{cell}] {gt_name}/{sp} done', flush=True)
    df = pd.DataFrame(rows)
    os.makedirs(f'{W}/results', exist_ok=True)
    df.to_csv(f'{W}/results/multimetric_{cell}.tsv', sep='\t', index=False, float_format='%.4g')
    pd.set_option('display.width', 220)
    print(df[df.chrom == 'all'].to_string(index=False))


if __name__ == '__main__':
    main(sys.argv[1])
