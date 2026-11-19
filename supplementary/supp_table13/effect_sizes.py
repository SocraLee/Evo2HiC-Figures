#!/usr/bin/env python
"""Supplementary Table 13: per-assay effect sizes with uncertainty for the cross-cell-type differential analysis (Extended Data Fig. 5c).
Differential PCC = corr(Pred_A - Pred_B, GT_A - GT_B) over chr9+chr10 bins (2 kb), averaged over the 3 unordered
cell pairs (GM12878, H1ESC, K562), for Evo2HiC, the Hi-C-only ablation and the sequence-only Evo2 baseline.
Effect size = Evo2HiC - track-wise strongest baseline; 95% CI from a block bootstrap over genomic bins
(100 kb blocks, 2000 resamples, same blocks for all models -> paired). Also per-chromosome values and the
recomputed "average relative improvement". Outputs: effect_sizes.tsv, effect_sizes.png"""
import os, sys, numpy as np, pandas as pd
from itertools import combinations
from scipy.stats import pearsonr
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))  # repo root (paths.py)
import paths as P  # noqa: E402
REPO = str(P.REPO)
os.chdir(REPO); sys.path.insert(0, REPO); sys.path.insert(0, str(P.ED_DIR / 'ed_fig5'))  # crosscell_eval.py
from crosscell_eval import load_all, delta_arrays, TRACKS, MODELS, CHROMS
Q = str(P.RA_WORK('supp_table13'))
pred, gt, chrom_bins = load_all(); deltas, cum = delta_arrays(pred, gt, chrom_bins)
pairs = [(A, B) for A, B in combinations(['GM12878', 'H1ESC', 'K562'], 2)]
n = cum[-1]; BLOCK = 50; nblk = int(np.ceil(n / BLOCK)); rng = np.random.default_rng(0)
def pcc_blocks(x, y, idx):
    xs, ys = x[idx], y[idx]; return pearsonr(xs, ys)[0]
def spec(model, ti, idx=None):
    vals = []
    for A, B in pairs:
        d = deltas[(A, B)]; x, y = d[model][ti], d['gt'][ti]
        vals.append(pearsonr(x, y)[0] if idx is None else pcc_blocks(x, y, idx))
    return float(np.mean(vals))
rows = []; boot = {}
print('block bootstrap ...', flush=True)
samples = [np.concatenate([np.arange(b * BLOCK, min((b + 1) * BLOCK, n)) for b in rng.integers(0, nblk, nblk)]) for _ in range(2000)]
for ti, t in enumerate(TRACKS):
    point = {m: spec(m, ti) for m in MODELS}
    best_base = max([m for m in MODELS if m != 'Evo2HiC'], key=lambda m: point[m])
    bs = {m: np.array([spec(m, ti, idx) for idx in samples]) for m in MODELS}
    diff = bs['Evo2HiC'] - bs[best_base]; diff_seq = bs['Evo2HiC'] - bs['Evo2']; diff_hic = bs['Evo2HiC'] - bs['HiC_only']
    per_chr = {m: [np.mean([pearsonr(deltas[p][m][ti][cum[k]:cum[k + 1]], deltas[p]['gt'][ti][cum[k]:cum[k + 1]])[0] for p in pairs]) for k in range(len(CHROMS))] for m in MODELS}
    rows.append(dict(track=t, Evo2HiC=point['Evo2HiC'], HiC_only=point['HiC_only'], Evo2=point['Evo2'], best_baseline=best_base,
                     delta_vs_best=point['Evo2HiC'] - point[best_base], ci_lo=np.percentile(diff, 2.5), ci_hi=np.percentile(diff, 97.5), p_boot=float(np.mean(diff <= 0)),
                     rel_vs_best_pct=100 * (point['Evo2HiC'] - point[best_base]) / point[best_base],
                     delta_vs_Evo2=point['Evo2HiC'] - point['Evo2'], ci_lo_Evo2=np.percentile(diff_seq, 2.5), ci_hi_Evo2=np.percentile(diff_seq, 97.5),
                     delta_vs_HiC_only=point['Evo2HiC'] - point['HiC_only'], ci_lo_HiC=np.percentile(diff_hic, 2.5), ci_hi_HiC=np.percentile(diff_hic, 97.5),
                     **{f'{m}_chr{CHROMS[k]}': per_chr[m][k] for m in MODELS for k in range(len(CHROMS))}))
    print(f'{t:9s} Evo2HiC {point["Evo2HiC"]:.3f} | HiC-only {point["HiC_only"]:.3f} | Evo2 {point["Evo2"]:.3f} | vs best ({best_base}) {rows[-1]["delta_vs_best"]:+.3f} [{rows[-1]["ci_lo"]:+.3f},{rows[-1]["ci_hi"]:+.3f}] rel {rows[-1]["rel_vs_best_pct"]:+.1f}%  chr9/chr10 Evo2HiC {per_chr["Evo2HiC"][0]:.3f}/{per_chr["Evo2HiC"][1]:.3f}', flush=True)
df = pd.DataFrame(rows); df.to_csv(f'{Q}/effect_sizes.tsv', sep='\t', index=False, float_format='%.4f')
print(f'\nmean relative improvement over track-wise strongest baseline: {df.rel_vs_best_pct.mean():+.2f}% (paper: 5.15%); mean absolute delta {df.delta_vs_best.mean():+.4f}')
print('tracks where Evo2HiC is best:', df[df.delta_vs_best > 0].track.tolist(), '; not best:', df[df.delta_vs_best <= 0].track.tolist())
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(6, 3.2)); x = np.arange(len(df))
for i, (m, col) in enumerate([('Evo2', '#7f8c8d'), ('HiC_only', '#2980b9'), ('Evo2HiC', '#c0392b')]):
    ax.bar(x + (i - 1) * 0.26, df[m], 0.26, color=col, label=m.replace('_', '-'))
for i, r in df.iterrows():
    ax.errorbar(i + 0.26, r.Evo2HiC, yerr=[[max(0, r.Evo2HiC - (r[r.best_baseline] + r.ci_lo))], [max(0, (r[r.best_baseline] + r.ci_hi) - r.Evo2HiC)]], fmt='none', ecolor='k', lw=.8, capsize=2)
    ax.text(i + 0.26, r.Evo2HiC + 0.02, f'{r.delta_vs_best:+.3f}', ha='center', fontsize=7)
ax.set_xticks(x); ax.set_xticklabels(df.track); ax.set_ylabel('cell-type differential PCC\n(mean of 3 pairs)'); ax.legend(fontsize=7, frameon=False); ax.set_title('Evo2HiC − best baseline with 95% block-bootstrap CI', fontsize=9)
plt.tight_layout(); plt.savefig(f'{Q}/effect_sizes.png', dpi=130); print('figure -> effect_sizes.png')
