#!/usr/bin/env python
"""Supplementary Fig. 15: representative loci including UNSUCCESSFUL assays: for each track pick a 100 kb window with a
large between-cell GT difference where (success) Evo2HiC's delta agrees with GT and the baselines' do not, or
(failure, H3K27me3 / H3K27ac) Evo2HiC's delta disagrees while the sequence-only baseline agrees.
Output (EVO2HIC_OUT_DIR): supp_fig15.pdf, supp_fig15.png, supp_fig15_loci.tsv"""
import os, sys, numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))  # repo root (paths.py)
import paths as P  # noqa: E402
REPO = str(P.REPO)
os.chdir(REPO); sys.path.insert(0, REPO); sys.path.insert(0, str(P.ED_DIR / 'ed_fig5'))  # crosscell_eval.py
from crosscell_eval import load_all, delta_arrays, TRACKS, MODELS, CHROMS, RESOLUTION
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.style.use('seaborn-v0_8-white'); plt.rcParams.update({'font.size': 7, 'font.family': 'Arial', 'pdf.fonttype': 42, 'axes.linewidth': 0.6})
sys.path.insert(0, str(P.PLOT_DIR)); from plot_utils import colors as PAL
Q = str(P.ensure_out_dir())
pred, gt, chrom_bins = load_all(); deltas, cum = delta_arrays(pred, gt, chrom_bins)
W = 25  # bins each side -> 100 kb window
def locate(i):
    k = np.searchsorted(cum, i, side='right') - 1; return CHROMS[k], (i - cum[k]) * RESOLUTION
def pick(track, pair, mode):
    ti = TRACKS.index(track); d = deltas[pair]; g = d['gt'][ti]
    def wcorr(x):  # windowed correlation of delta with GT delta
        out = np.full(len(g), np.nan)
        for i in range(W, len(g) - W):
            a, b = x[i - W:i + W], g[i - W:i + W]
            if b.std() > 0 and a.std() > 0: out[i] = np.corrcoef(a, b)[0, 1]
        return out
    ce, cs, ch = wcorr(d['Evo2HiC'][ti]), wcorr(d['Evo2'][ti]), wcorr(d['HiC_only'][ti])
    amp = np.array([np.abs(g[max(0, i - W):i + W]).mean() for i in range(len(g))]); amp_thr = np.nanpercentile(amp, 90)
    score = (ce - np.fmax(cs, ch)) if mode == 'success' else (cs - ce)
    score[amp < amp_thr] = np.nan
    i = int(np.nanargmax(score)); return i, ce[i], cs[i], ch[i]
CASES = [('DNase', ('GM12878', 'K562'), 'success'), ('CTCF', ('H1ESC', 'K562'), 'success'), ('H3K4me3', ('GM12878', 'H1ESC'), 'success'),
         ('H3K27ac', ('GM12878', 'K562'), 'failure'), ('H3K27me3', ('GM12878', 'K562'), 'failure')]
ROWS = [('Measured', gt), ('Evo2HiC', pred['Evo2HiC']), ('Hi-C-only', pred['HiC_only']), ('Sequence-only (Evo 2)', pred['Evo2'])]
CA, CB = '#e8554e', '#3c78b4'
fig, axes = plt.subplots(len(ROWS), len(CASES), figsize=(7.2, 5.0), sharex=True); rows = []
for j, (track, pair, mode) in enumerate(CASES):
    ti = TRACKS.index(track); i, ce, cs, ch = pick(track, pair, mode); A, B = pair; chrom, start = locate(i); sl = slice(i - W, i + W); xs = (np.arange(-W, W) * RESOLUTION) / 1000
    dc = {'Evo2HiC': ce, 'Hi-C-only': ch, 'Sequence-only (Evo 2)': cs}
    for r, (lab, src) in enumerate(ROWS):
        ax = axes[r, j]; ax.plot(xs, src[A][ti][sl], color=CA, lw=0.8, label=A); ax.plot(xs, src[B][ti][sl], color=CB, lw=0.8, label=B)
        ax.set_yticks([]); ax.spines[['top', 'right']].set_visible(False)
        if j == 0: ax.set_ylabel(lab, fontsize=7)
        if r == 0:
            ax.set_title(f'{track} ({"success" if mode == "success" else "failure"})\nchr{chrom}:{start/1e6:.2f} Mb', fontsize=7)
            ax.legend(fontsize=5.5, frameon=False, loc='upper right', handlelength=1.2)
        if lab in dc: ax.text(0.02, 0.86, f'$\\Delta$-corr {dc[lab]:.2f}', transform=ax.transAxes, fontsize=6)
    axes[-1, j].set_xticks([-50, 0, 50]); axes[-1, j].set_xlabel('kb from centre', fontsize=7)
    rows.append(dict(track=track, mode=mode, pair=f'{A}_vs_{B}', chrom=f'chr{chrom}', start=start, end=start + RESOLUTION, window_corr_Evo2HiC=ce, window_corr_Evo2=cs, window_corr_HiC_only=ch))
    print(rows[-1])
pd.DataFrame(rows).to_csv(f'{Q}/supp_fig15_loci.tsv', sep='\t', index=False, float_format='%.3f'); plt.tight_layout(h_pad=0.4, w_pad=0.6)
plt.savefig(f'{Q}/supp_fig15.pdf', bbox_inches='tight'); plt.savefig(f'{Q}/supp_fig15.png', dpi=150, bbox_inches='tight'); print(f'figure -> {Q}/supp_fig15.pdf')
