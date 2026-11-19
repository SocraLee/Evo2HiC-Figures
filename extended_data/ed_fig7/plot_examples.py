#!/usr/bin/env python
"""Supplementary Fig. 16: distant-clade success / failure examples, contact maps with boundary calls.
Species chosen by a stated rule (Evo2HiC PCC >= clade median; largest / smallest pixel gain vs the same model on random
DNA). The 2 Mb window shown is the one whose local gain (window PCC, Evo2HiC minus random DNA) is closest to the
species' median window gain, i.e. a representative window, not the most favourable one.
Usage: plot_examples.py -> examples.pdf / examples.png"""
import os, sys
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))  # repo root (paths.py)
import paths as P  # noqa: E402
S = str(P.RA_WORK('species'))
sys.path.insert(0, str(P.REPO)); os.chdir(P.REPO)
from evaluate import multi_TAD_eval_cooltools as T   # map loader + method paths
from config import dnazoo_hic_dir
plt.style.use('seaborn-v0_8-white'); plt.rcParams.update({'font.size': 7, 'font.family': 'Arial', 'pdf.fonttype': 42})
RES, W = 10000, 200                               # 2 Mb windows
CASES = [('Glycine_max', 'Soybean (angiosperm)\nsuccess'),
         ('Augochlora_pura', 'Sweat bee (protostome)\nfailure')]
COLS = [('gt', 'Ground truth'), ('evo2hic', 'Evo2HiC'), ('random', 'Evo2HiC, random DNA')]


def load(sp):
    m = {'gt': T.load_matrix(os.path.join(dnazoo_hic_dir, sp + '.hic'), RES)[0],
         'evo2hic': T.load_matrix(os.path.join(T.METHODS['evo2hic'], sp + '_enhanced.hic'), RES)[0],
         'random': T.load_matrix(os.path.join(S, 'dna_ablation', 'random', sp + '_enhanced.hic'), RES)[0]}
    out = {}
    for k, v in m.items():
        v = np.nan_to_num(np.asarray(v, float)); out[k] = np.triu(v) + np.triu(v, 1).T   # mirror to a full matrix
    return out


def wpcc(a, b):
    iu = np.triu_indices(a.shape[0], 1)
    x, y = np.log1p(a[iu]), np.log1p(b[iu])
    return np.corrcoef(x, y)[0, 1] if x.std() > 0 and y.std() > 0 else np.nan


def bounds(sp):
    z = np.load(f'{S}/tad_candidates/{sp}.npz'); r = np.load(f'{S}/tad_candidates_random/{sp}.npz')
    f = lambda i, b: i[b >= 0.2]
    return {'gt': f(z['gt_idx'], z['gt_bs']), 'evo2hic': f(z['evo2hic_idx'], z['evo2hic_bs']),
            'random': f(r['evo2hic_random_idx'], r['evo2hic_random_bs'])}


fig, axes = plt.subplots(len(CASES), 3, figsize=(5.6, 2.0 * len(CASES)))
for r, (sp, title) in enumerate(CASES):
    M = load(sp); B = bounds(sp); n = M['gt'].shape[0]
    def filled(s):   # fraction of non-zero ground-truth pixels within 500 kb of the diagonal
        g = M['gt'][s:s + W, s:s + W]; i, j = np.triu_indices(W, 1); k = (j - i) <= 50
        return (g[i[k], j[k]] > 0).mean()
    starts = [s for s in range(0, n - W, W) if filled(s) > 0.9]
    g = np.array([wpcc(M['evo2hic'][s:s + W, s:s + W], M['gt'][s:s + W, s:s + W]) -
                  wpcc(M['random'][s:s + W, s:s + W], M['gt'][s:s + W, s:s + W]) for s in starts])
    s0 = starts[int(np.nanargmin(np.abs(g - np.nanmedian(g))))]
    lo, hi = s0, s0 + W                   # crop edge bins that are empty in any prediction (patch borders)
    empty = lambda k, i: (M[k][i, lo:hi] > 0).mean() < 0.1
    while lo < hi and any(empty(k, lo) for k in M): lo += 1
    while hi > lo and any(empty(k, hi - 1) for k in M): hi -= 1
    sub = {k: v[lo:hi, lo:hi] for k, v in M.items()}; s0, Wd = lo, hi - lo
    vmax = np.percentile(np.log1p(sub['gt'][np.triu_indices(Wd, 1)]), 98)
    for c, (k, lab) in enumerate(COLS):
        ax = axes[r, c]
        ax.imshow(np.log1p(sub[k]), cmap='Reds', vmin=0, vmax=vmax, interpolation='none')
        ax.set_xlim(-0.5, Wd - 0.5); ax.set_ylim(Wd - 0.5, -0.5); ax.set_aspect('equal', adjustable='box')
        for b in B[k]:
            if s0 <= b < s0 + Wd:
                ax.plot([b - s0 - 0.5] * 2, [b - s0 - 6, b - s0 + 6], color='#1f5fa8', lw=0.8)
        ax.set_xticks([]); ax.set_yticks([])
        txt = lab if k == 'gt' else f'{lab}\nwindow PCC {wpcc(sub[k], sub["gt"]):.2f}'
        ax.set_title(txt, fontsize=6)
        if c == 0:
            ax.set_ylabel(f'{title}\n{sp.split("__")[0].replace("_", " ")}\n{s0 * RES / 1e6:.2f}-{(s0 + Wd) * RES / 1e6:.2f} Mb', fontsize=6)
    print(sp, 'window start', s0, 'bins', Wd, 'median window gain', round(float(np.nanmedian(g)), 3), flush=True)
plt.tight_layout(h_pad=0.6, w_pad=0.3)
plt.savefig(f'{S}/examples.pdf', bbox_inches='tight'); plt.savefig(f'{S}/examples.png', dpi=170, bbox_inches='tight'); print('examples.png')
