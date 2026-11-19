#!/usr/bin/env python
"""Extended Data Fig. 7: contribution of the sequence prior across species, paired DNA vs no-DNA.
a  pixel PCC gain of Evo2HiC over its no-DNA versions (same model on random DNA; Hi-C-only ablation), per species
b  CTCF-motif fraction of boundaries added vs removed by DNA (cooltools, paper protocol)
c  accuracy (within 3 bins of a ground-truth boundary) of boundaries added vs removed by DNA
Output: xs_fig.pdf / xs_fig.png"""
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))  # repo root (paths.py)
import paths as P  # noqa: E402
sys.path.insert(0, str(P.PLOT_DIR)); from plot_utils import colors as PAL
plt.style.use('seaborn-v0_8-white'); plt.rcParams.update({'font.size': 7, 'font.family': 'Arial', 'pdf.fonttype': 42, 'axes.linewidth': 0.6})
S = str(P.RA_WORK('species'))

meta = pd.read_csv(f'{S}/cross_species_per_species.tsv', sep='\t')


def grp(r):
    if r.clade == 'Mammalia':
        return 'Primates' if (pd.notna(r.mya) and r.mya <= 147.6) else 'Other placentals'
    return {'Actinopterygii': None}.get(r.clade, r.clade)


meta['grp'] = meta.apply(grp, axis=1)
ORDER = ['Primates', 'Other placentals', 'Marsupialia', 'Reptilia', 'Protostomes', 'Angiosperms']
LAB = {g: f'{g}\n(n={int((meta.grp == g).sum())})' for g in ORDER}
rnd = pd.read_csv(f'{S}/tad_ctcf_random.tsv', sep='\t').merge(meta[['species', 'grp']], on='species')
hco = pd.read_csv(P.RA_TABLE('species', 'tad_ctcf.tsv'), sep='\t').merge(meta[['species', 'grp']], on='species')

fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.6))
rng = np.random.default_rng(0)
C_RND, C_HCO = PAL[0], '#3c78b4'

# a: pixel gains
ax = axes[0]
for i, g in enumerate(ORDER):
    x = meta[meta.grp == g]
    for off, col, c in ((-0.18, 'PCC_random', C_RND), (0.18, 'PCC_hic_only', C_HCO)):
        v = (x.PCC_evo2hic - x[col]).dropna().values
        ax.scatter(i + off + rng.uniform(-0.07, 0.07, len(v)), v, s=3, color=c, alpha=0.6, lw=0)
        ax.plot([i + off - 0.12, i + off + 0.12], [np.median(v)] * 2, color='k', lw=1)
ax.axhline(0, color='0.5', lw=0.6, ls='--')
ax.set_xticks(range(len(ORDER))); ax.set_xticklabels([LAB[g] for g in ORDER], rotation=45, ha='right', fontsize=6)
ax.set_ylabel('pixel PCC gain from DNA'); ax.set_title('a  Contact-map accuracy', loc='left', fontsize=7)
ax.scatter([], [], s=8, color=C_RND, label='vs same model, random DNA'); ax.scatter([], [], s=8, color=C_HCO, label='vs Hi-C-only ablation')
ax.legend(fontsize=5.5, frameon=False, loc='lower left')


def paired(ax, key, title, ylab):
    for i, g in enumerate(ORDER):
        for off, df, c in ((-0.18, rnd, C_RND), (0.18, hco, C_HCO)):
            x = df[df.grp == g][[f'{key}_added', f'{key}_removed']].dropna()
            if not len(x):
                continue
            a, r = x[f'{key}_added'].mean(), x[f'{key}_removed'].mean()
            ax.plot([i + off - 0.08, i + off + 0.08], [r, a], color=c, lw=1.2)
            ax.scatter([i + off - 0.08], [r], s=10, facecolor='white', edgecolor=c, lw=0.8, zorder=3)
            ax.scatter([i + off + 0.08], [a], s=10, color=c, zorder=3)
    ax.set_xticks(range(len(ORDER))); ax.set_xticklabels([LAB[g] for g in ORDER], rotation=45, ha='right', fontsize=6)
    ax.set_ylabel(ylab); ax.set_title(title, loc='left', fontsize=7)
    ax.scatter([], [], s=10, facecolor='white', edgecolor='0.3', label='removed by DNA'); ax.scatter([], [], s=10, color='0.3', label='added by DNA')
    ax.legend(fontsize=5.5, frameon=False, loc='best')


paired(axes[1], 'motif', 'b  CTCF motif at boundaries', 'fraction with CTCF motif')
paired(axes[2], 'acc', 'c  Boundary accuracy', 'fraction matching ground truth')
for ax in axes:
    ax.spines[['top', 'right']].set_visible(False)
plt.tight_layout(w_pad=1.0)
plt.savefig(f'{S}/xs_fig.pdf', bbox_inches='tight'); plt.savefig(f'{S}/xs_fig.png', dpi=160, bbox_inches='tight'); print('xs_fig.pdf')
