#!/usr/bin/env python
"""Three-model comparison on Orca's 417 test windows: Evo2HiC (original), Evo2HiC-cg
(retrained on the Orca-style coarse-grained target) and Orca.  Same masks, spaces,
ground truths and metric code as multimetric.py / is_diag.py /
boundary_pr.py, so numbers are directly comparable with results/*.tsv.

Reported (chrom = all; per-chromosome rows in the tsv):
  A  Orca's setting: pooled PCC / SPC in log(O/E) vs Orca's own target
  B  Evo2HiC's setting: per-diagonal pooled SPC (offsets >= 5), raw counts, both GTs
  C  C.Origami panel (dist-strat PCC, OE PCC, MSE) in the corigami space, both GTs
  D  cooltools insulation-profile PCC (Fig 3b/f) and TAD-boundary F1 / PR-AUC / top-k F1
Paired one-sided Wilcoxon p-values: cg vs Orca and cg vs original Evo2HiC.
Usage: eval_cg.py <cell> [tag ...]   (tags default: cg; e.g. `cglogoe cg`) -> results/eval_<tag0>_<cell>.tsv
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr, wilcoxon
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))  # repo root (paths.py)
import paths as P  # noqa: E402

W = str(P.RA_WORK('orca'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from multimetric import (N, IDX, diag_mean, ds_normalize, observed_vs_expected, mse,  # noqa: E402
                         distance_stratified, pooled_diag_spc, pooled)
from is_diag import profile_and_bounds  # noqa: E402
from boundary_pr import pr_auc  # noqa: E402
R1 = str(P.REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # notebook_eval.py
import notebook_eval as ne  # noqa: E402


def pw(a, b, alt='greater'):
    ok = np.isfinite(a) & np.isfinite(b)
    d = (a - b)[ok]
    return wilcoxon(d, alternative=alt).pvalue if ok.sum() > 5 and (d != 0).any() else np.nan


def main(cell):
    z = np.load(f'{W}/tensors/{cell}.npz')
    tags = sys.argv[2:] or ['cg']            # extra seq2hic variants: tensors/<cell>_<tag>.npz
    extra = {}
    for tag in tags:
        zt = np.load(f'{W}/tensors/{cell}_{tag}.npz')
        assert (z['regions'] == zt['regions']).all()
        extra['Evo2HiC-' + tag] = zt['evo_raw'].astype(float)
    zc = np.load(f'{W}/tensors/{cell}_{tags[0]}.npz')
    chrom = z['regions'][:, 0]
    E = np.load(f'{W}/tensors/expected_4kb_{cell}.npy')[:N]
    eps = float(E.min())
    Em = E[IDX]
    gt_ours = z['gt_raw'].astype(float)
    orca_t = z['orca_t_logoe'].astype(float)
    orca_valid = (z['orca_t_nan'] <= 0.25) & np.isfinite(orca_t)
    valid = orca_valid & np.isfinite(gt_ours)
    orca_raw = np.exp(z['orca_logoe'].astype(float)) * (Em + eps) - eps
    orcagt_raw = np.exp(orca_t) * (Em + eps) - eps
    models_raw = {'Evo2HiC': z['evo_raw'].astype(float), **extra, 'Orca': orca_raw}
    CG = 'Evo2HiC-' + tags[0]                  # variant used for the paired tests
    E_ours = diag_mean(gt_ours, valid)
    eps_o = float(np.nanmin(E_ours[E_ours > 0]))
    Emat_ours = E_ours[IDX]
    s_k = (E_ours / diag_mean(orcagt_raw, valid))[IDX][None]          # count scale / mcool scale

    def mk(x):
        y = x.copy()
        y[~valid] = np.nan
        return y

    def logoe_own(name, x):   # each model with its own expected (dimensionless)
        if name == 'Orca':
            return mk(z['orca_logoe'].astype(float))
        return mk(np.log((np.clip(x, 0, None) + eps_o) / (Emat_ours[None] + eps_o)))

    rows = []
    # ---- A: Orca's setting
    for name, x in models_raw.items():
        lo = logoe_own(name, x)
        rows.append(dict(section='A_orca_setting', gt='orca', metric='pooled_PCC_logoe', chrom='all', model=name,
                         value=pooled(lo, mk(orca_t), pearsonr)))
        rows.append(dict(section='A_orca_setting', gt='orca', metric='pooled_SPC_logoe', chrom='all', model=name,
                         value=pooled(lo, mk(orca_t), spearmanr)))
    # ---- B: Evo2HiC's setting, both GTs
    for gt_name, gt_raw, scale_to_gt in (('ours', mk(gt_ours), {'Orca': s_k}), ('orca', mk(orcagt_raw), {n: 1 / s_k for n in models_raw if n != 'Orca'})):
        preds = {n: mk(x * scale_to_gt.get(n, 1.0)) for n, x in models_raw.items()}
        for ch in [9, 10, 'all']:
            sel = np.ones(len(chrom), bool) if ch == 'all' else chrom == ch
            for n, p in preds.items():
                rows.append(dict(section='B_diag_pooled_SPC_k5_raw', gt=gt_name, metric='diag_pooled_SPC_k5', chrom=str(ch),
                                 model=n, value=pooled_diag_spc(p[sel], gt_raw[sel], 5)))
        # ---- C: C.Origami panel in the corigami space (per window -> paired tests)
        log1p_gt = np.log1p(np.clip(gt_raw, 0, None))
        per = {}
        for n, p in preds.items():
            q = ds_normalize(np.log1p(np.clip(p, 0, None)), log1p_gt)
            per[n] = {'dist_strat_PCC': distance_stratified(q, log1p_gt), 'OE_PCC': observed_vs_expected(q, log1p_gt),
                      'MSE': mse(q, log1p_gt)}
        for m in ['dist_strat_PCC', 'OE_PCC', 'MSE']:
            for n in preds:
                v = per[n][m]
                better_low = m == 'MSE'
                rec = dict(section='C_corigami', gt=gt_name, metric=m, chrom='all', model=n, value=float(np.nanmean(v)))
                if n == CG:
                    a, o, e = per[CG][m], per['Orca'][m], per['Evo2HiC'][m]
                    rec['p_cg_better_than_Orca'] = pw(o, a) if better_low else pw(a, o)
                    rec['p_cg_better_than_Evo2HiC'] = pw(e, a) if better_low else pw(a, e)
                rows.append(rec)
        # ---- D: insulation PCC + boundary metrics (paper pipeline), per window
        isp, f1, auc, f1k = {n: [] for n in preds}, {n: [] for n in preds}, {n: [] for n in preds}, {n: [] for n in preds}
        for i in range(len(gt_raw)):
            m = valid[i]
            if m.sum() < 1000:
                for n in preds:
                    isp[n].append(np.nan); f1[n].append(np.nan); auc[n].append(np.nan); f1k[n].append(np.nan)
                continue
            sg, bg, _ = profile_and_bounds(np.where(m, gt_raw[i], 0))
            for n, p in preds.items():
                sp_, bp_, pp_ = profile_and_bounds(np.where(m, p[i], 0))
                ok = np.isfinite(sg) & np.isfinite(sp_)
                isp[n].append(pearsonr(sp_[ok], sg[ok])[0] if ok.sum() > 10 else np.nan)
                f1[n].append(ne.bound_matching(bp_, bg)[0] if len(bg) else np.nan)
                a_, k_ = pr_auc(pp_, bg) if len(bg) else (np.nan, np.nan)
                auc[n].append(a_); f1k[n].append(k_)
            if i % 100 == 0:
                print(f'[{cell}/{gt_name}] window {i}/417', flush=True)
        for m, store in (('IS_PCC', isp), ('TAD_F1', f1), ('boundary_PR_AUC', auc), ('boundary_topk_F1', f1k)):
            arr = {n: np.array(v, float) for n, v in store.items()}
            for n in preds:
                rec = dict(section='D_structure', gt=gt_name, metric=m, chrom='all', model=n, value=float(np.nanmean(arr[n])))
                if n == CG:
                    rec['p_cg_better_than_Orca'] = pw(arr[CG], arr['Orca'])
                    rec['p_cg_better_than_Evo2HiC'] = pw(arr[CG], arr['Evo2HiC'])
                rows.append(rec)
    df = pd.DataFrame(rows)
    os.makedirs(f'{W}/results', exist_ok=True)
    df.to_csv(f'{W}/results/eval_{tags[0]}_{cell}.tsv', sep='\t', index=False, float_format='%.4g')
    pd.set_option('display.width', 220)
    t = df[df.chrom == 'all'].pivot_table(index=['section', 'gt', 'metric'], columns='model', values='value', aggfunc='first')
    print(t.to_string(float_format=lambda v: f'{v:.4f}'))
    print('\np-values (cg vs Orca / cg vs Evo2HiC):')
    print(df[df.model == CG].dropna(subset=['p_cg_better_than_Orca'])[['section', 'gt', 'metric', 'p_cg_better_than_Orca', 'p_cg_better_than_Evo2HiC']].to_string(index=False, float_format=lambda v: f'{v:.2g}'))


if __name__ == '__main__':
    main(sys.argv[1])
