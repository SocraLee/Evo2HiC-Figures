#!/usr/bin/env python
"""Why does Evo2HiC win insulation-profile PCC but lose TAD-boundary F1?
Per window (paper pipeline, cooltools window 100 kb, prominence >= 0.2, +-3 bin):
  * n boundaries called (GT / Evo2HiC / Orca), precision & recall separately
  * boundary-strength (prominence) distribution of each model
  * IS PCC (i) as computed, (ii) excluding bins within +-6 bins of any masked bin
    (removes the shared zero-dip artefact), (iii) round-1 style (finite-GT mask only,
    no Orca mask)
  * IS PCC after removing the low-frequency trend (profile minus 50-bin running
    mean) - tests whether Evo2HiC's edge comes from smooth shape
Usage: is_diag.py <cell> <ours|orca>
"""
import sys

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, wilcoxon
from scipy.ndimage import uniform_filter1d, binary_dilation
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))  # repo root (paths.py)
import paths as P  # noqa: E402

R1 = str(P.REPO)
sys.path.insert(0, R1)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # notebook_eval.py
import notebook_eval as ne  # noqa: E402
from cooltools.api.insulation import calculate_insulation_score, find_boundaries  # noqa: E402,F401

W = str(P.RA_WORK('orca'))
N = 250
IDX = np.abs(np.arange(N)[:, None] - np.arange(N)[None, :])


def profile_and_bounds(M):
    """scores, boundary bins, and prominence of every called boundary (paper pipeline)."""
    import os
    import tempfile
    import cooler
    fd, cp = tempfile.mkstemp(suffix='.cool')
    os.close(fd)
    os.unlink(cp)
    starts = np.arange(0, N * 4000, 4000, dtype=np.int64)
    bins = pd.DataFrame({'chrom': 'chrR', 'start': starts, 'end': starts + 4000})
    i, j = np.triu_indices(N)
    c = M[i, j].astype(float)
    m = np.isfinite(c) & (c != 0)
    cooler.create_cooler(cp, bins=bins, pixels=pd.DataFrame({'bin1_id': i[m], 'bin2_id': j[m], 'count': c[m]}),
                         dtypes={'count': 'float64'}, ordered=True)
    t = calculate_insulation_score(cooler.Cooler(cp), window_bp=[100000], ignore_diags=2, clr_weight_name=None)
    t = find_boundaries(t)
    os.unlink(cp)
    s = t['log2_insulation_score_100000'].to_numpy(float)
    prom = t['boundary_strength_100000'].to_numpy(float)
    b = np.where(np.isfinite(prom) & (prom >= 0.2))[0]
    return s, b, prom


def pr(prd, tgt, z=3):
    if len(tgt) == 0 or len(prd) == 0:
        return np.nan, np.nan
    pm = np.array([any(abs(p - t) <= z for t in tgt) for p in prd])
    tm = np.array([any(abs(p - t) <= z for p in prd) for t in tgt])
    return pm.mean(), tm.mean()


def pcc(a, b, keep=None):
    ok = np.isfinite(a) & np.isfinite(b)
    if keep is not None:
        ok &= keep
    return pearsonr(a[ok], b[ok])[0] if ok.sum() > 10 and a[ok].std() > 0 and b[ok].std() > 0 else np.nan


def main(cell, gtname):
    z = np.load(f'{W}/tensors/{cell}.npz')
    E = np.load(f'{W}/tensors/expected_4kb_{cell}.npy')[:N]
    eps = float(E.min())
    Em = E[IDX]
    gt_ours = z['gt_raw'].astype(float)
    evo = z['evo_raw'].astype(float)
    orca = np.exp(z['orca_logoe'].astype(float)) * (Em + eps) - eps
    orcagt = np.exp(z['orca_t_logoe'].astype(float)) * (Em + eps) - eps
    gt = gt_ours if gtname == 'ours' else orcagt
    ovalid = (z['orca_t_nan'] <= 0.25) & np.isfinite(z['orca_t_logoe'])
    gfin = np.isfinite(gt_ours)
    rec = []
    for i in range(len(gt)):
        for mode, m in (('C_mask', ovalid[i] & gfin[i]), ('A_round1', gfin[i])):
            if m.sum() < 1000:
                continue
            sg, bg, pg = profile_and_bounds(np.where(m, gt[i], 0))
            se, be, pe = profile_and_bounds(np.where(m, evo[i], 0))
            so, bo, po = profile_and_bounds(np.where(m, orca[i], 0))
            badbin = ~m.all(1)                                    # bins with any masked pixel in their row
            far = ~binary_dilation(badbin, iterations=6)
            detr = lambda s: s - uniform_filter1d(np.nan_to_num(s), 50, mode='nearest')
            r = dict(win=i, mode=mode, n_gt=len(bg), n_evo=len(be), n_orca=len(bo),
                     prom_evo=np.nanmedian(pe[be]) if len(be) else np.nan,
                     prom_orca=np.nanmedian(po[bo]) if len(bo) else np.nan,
                     prom_gt=np.nanmedian(pg[bg]) if len(bg) else np.nan,
                     pcc_evo=pcc(se, sg), pcc_orca=pcc(so, sg),
                     pcc_far_evo=pcc(se, sg, far), pcc_far_orca=pcc(so, sg, far),
                     pcc_detr_evo=pcc(detr(se), detr(sg)), pcc_detr_orca=pcc(detr(so), detr(sg)),
                     sd_evo=np.nanstd(se), sd_orca=np.nanstd(so), sd_gt=np.nanstd(sg),
                     frac_bad_bins=badbin.mean())
            r['prec_evo'], r['rec_evo'] = pr(be, bg)
            r['prec_orca'], r['rec_orca'] = pr(bo, bg)
            r['f1_evo'], _ = ne.bound_matching(be, bg)
            r['f1_orca'], _ = ne.bound_matching(bo, bg)
            rec.append(r)
        if i % 100 == 0:
            print(f'[{cell}/{gtname}] {i}/417', flush=True)
    df = pd.DataFrame(rec)
    df.to_csv(f'{W}/results/is_diag_{cell}_{gtname}.tsv', sep='\t', index=False, float_format='%.4g')
    for mode, g in df.groupby('mode'):
        print(f'\n===== {cell} GT={gtname} mode={mode} n={len(g)}  frac bins touched by mask={g.frac_bad_bins.mean():.3f}')
        for k in ['n_gt', 'n_evo', 'n_orca', 'prom_gt', 'prom_evo', 'prom_orca', 'sd_gt', 'sd_evo', 'sd_orca']:
            print(f'  {k:12s} {g[k].mean():.3f}')
        for a, b in [('pcc_evo', 'pcc_orca'), ('pcc_far_evo', 'pcc_far_orca'), ('pcc_detr_evo', 'pcc_detr_orca'),
                     ('prec_evo', 'prec_orca'), ('rec_evo', 'rec_orca'), ('f1_evo', 'f1_orca')]:
            ok = np.isfinite(g[a]) & np.isfinite(g[b])
            d = (g[a] - g[b])[ok]
            p = wilcoxon(d, alternative='greater').pvalue if (d != 0).any() else np.nan
            print(f'  {a[:-4]:12s} Evo={g[a][ok].mean():.3f} Orca={g[b][ok].mean():.3f}  Evo wins {(d>0).sum()}/{ok.sum()}  p(Evo>Orca)={p:.2g}')


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
