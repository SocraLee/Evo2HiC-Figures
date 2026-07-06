"""Supplementary Figure 15 — representative loops, decomposed by anchor state.

Application (rebuttal: "give a use-case for the predicted epigenomics"). Hi-C
physically measures which loci touch; the model's PREDICTED epigenomics
(DNA + Hi-C -> 5 tracks) then assign a functional state to each loop anchor.
This panel shows three ground-truth HiCCUPS loops, one per representative
class, and demonstrates that the model's predicted anchor signature recovers
the real ChIP-seq one:

  row 1  Insulator (CTCF)        — intergenic CTCF/cohesin architectural loop
  row 2  Promoter-promoter       — active regulatory contact (e.g. SLK-SFR1)
  row 3  Bivalent (repressive)   — poised developmental loop (e.g. VAX1-EMX2)

Layout (3 rows x 3 cols): Hi-C contact map (loop circled, 5 kb) | anchor-A
signature | anchor-B signature. Each signature is a 2x4 heatmap (rows
real / predicted; cols CTCF, H3K4me3, H3K27ac, H3K27me3) for the gene at
that anchor.

Generating the data (see paths.py "Hi-C loop functional decomposition" for the
exact commands; the pipeline lives under revision/loop_decomposition/ and needs
the Evo2HiC env with torch + hic-straw):
  HiCCUPS loops    : juicer_tools hiccups ... -> HIC_LOOP_BEDPE(<acc>)  (step 0)
  predicted tracks : already shipped under EPI_EVO2HIC_DIR (== LOOP_PRED_TRACK);
                     regenerate with inference_CDNA1d --save-dir track.
  real tracks      : per-cell hic2track BigWigs (HIC2TRACK) — same epigenomic
                     source as Fig 4 / Supp 8, 10-12, 16, 18.
This figure reads the predicted tracks + real tracks + the raw .hic directly;
it does NOT need loop_labels.tsv (the loci are hand-picked typical examples).

Outputs
-------
Figures/supplementary_15.pdf
Figures/supplementary_15.png
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1 import make_axes_locatable

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paths import (                                              # noqa: E402
    ensure_out_dir, add_repo_to_syspath,
    HIC_RAW, HIC2TRACK, LOOP_PRED_TRACK,
)
add_repo_to_syspath()

from dataset.hic_loader import HiC_Loader                        # noqa: E402
from dataset.track_loader import Track_Loader                    # noqa: E402

OUT_PDF = ensure_out_dir() / 'supplementary_15.pdf'
OUT_PNG = ensure_out_dir() / 'supplementary_15.png'

ACC = {'H1ESC': '4DNFIQYQWPF5', 'GM12878': '4DNFI1UEG1HD'}       # config.hic_data_dir accessions
RES_T, RES_H = 2000, 5000
MARKS = ['DNase', 'CTCF', 'H3K27ac', 'H3K27me3', 'H3K4me3']      # config.tracks order
ti = {m: i for i, m in enumerate(MARKS)}
SHOW = ['CTCF', 'H3K4me3', 'H3K27ac', 'H3K27me3']

# (row label, cell, chrom, anchorA bp, geneA, anchorB bp, geneB)
EX = [
    ('Insulator\n(CTCF)', 'H1ESC', 9, 74_775_000, 'CTCF site', 75_005_000, 'CTCF site'),
    ('Promoter–promoter\n(active)', 'GM12878', 10, 103_965_000, 'SLK', 104_125_000, 'SFR1'),
    ('Bivalent\n(repressive)', 'H1ESC', 10, 117_135_000, 'VAX1', 117_545_000, 'EMX2'),
]

plt.style.use('seaborn-v0_8-white')
plt.rcParams.update({'font.size': 8, 'font.family': 'Arial',
                     'pdf.fonttype': 42, 'ps.fonttype': 42})

_hic, _trk, _pred = {}, {}, {}


def hic(cell):
    if cell not in _hic:
        _hic[cell] = HiC_Loader(str(HIC_RAW(ACC[cell])), resolution=RES_H)
    return _hic[cell]


def trk(cell):
    if cell not in _trk:
        _trk[cell] = Track_Loader(str(HIC2TRACK(cell)), resolution=RES_T)
    return _trk[cell]


def pred(cell, ch):
    k = (cell, ch)
    if k not in _pred:
        _pred[k] = np.load(LOOP_PRED_TRACK(cell, ch))
    return _pred[k]


def anchor_sig(cell, ch, c):
    cb = c // RES_T
    real = trk(cell).get(ch, (cb - 2) * RES_T, (cb + 3) * RES_T, 0).max(axis=1)
    arr = pred(cell, ch)
    p = arr[:, max(0, cb - 2):cb + 3].max(axis=1)
    return np.array([[real[ti[m]] for m in SHOW], [p[ti[m]] for m in SHOW]])


def main():
    fig = plt.figure(figsize=(8.6, 6.6))
    outer = GridSpec(len(EX), 3, width_ratios=[1.5, 1.2, 1.2], wspace=0.4, hspace=0.6,
                     top=0.92, bottom=0.14)
    im = None
    for r, (lab, cell, ch, aA, gA, aB, gB) in enumerate(EX):
        a1, a2 = min(aA, aB), max(aA, aB)
        margin = max(60_000, int(0.4 * (a2 - a1)))
        ws, we = ((a1 - margin) // RES_H) * RES_H, ((a2 + margin) // RES_H) * RES_H
        M = np.nan_to_num(hic(cell).get(ch, ws, we, 0, ch, ws, we, 0, norm=True)[0])

        axh = fig.add_subplot(outer[r, 0])
        vmax = np.percentile(M[M > 0], 95) if (M > 0).any() else 1
        axh.imshow(np.log1p(M), cmap='Reds', vmin=0, vmax=np.log1p(vmax),
                   extent=(ws, we, we, ws), interpolation='none', aspect='auto')
        axh.plot(aB, aA, 'o', mfc='none', mec='#1f4e79', ms=12, mew=1.5)
        axh.plot(aA, aB, 'o', mfc='none', mec='#1f4e79', ms=12, mew=1.5)
        axh.set_xticks([]); axh.set_yticks([])
        axh.set_ylabel(f'{lab}\n({cell})', fontsize=9, fontweight='bold')
        if r == 0:
            axh.set_title('Hi-C  (loop circled)', fontsize=8.5)

        for ci, (c, gn) in enumerate([(aA, gA), (aB, gB)]):
            sig = anchor_sig(cell, ch, c)
            ax = fig.add_subplot(outer[r, 1 + ci])
            im = ax.imshow(sig, cmap='Blues', vmin=0, vmax=1.0, aspect='auto')
            ax.set_xticks(range(4)); ax.set_xticklabels(SHOW, rotation=40, ha='right', fontsize=7)
            ax.set_yticks([0, 1]); ax.set_yticklabels(['Real', 'Pred'], fontsize=7.5)
            ax.set_title((f'$\\it{{{gn}}}$' if gn not in ('CTCF site',) else gn), fontsize=8.5)
            for i in range(2):
                for j in range(4):
                    ax.text(j, i, f'{sig[i,j]:.2f}', ha='center', va='center', fontsize=6.5,
                            color='white' if sig[i, j] > 0.55 else 'black')
            if ci == 1:  # small per-row colorbar, matched to this heatmap's height
                ccax = make_axes_locatable(ax).append_axes('right', size='7%', pad=0.06)
                cbb = fig.colorbar(im, cax=ccax)
                cbb.set_ticks([0, 0.5, 1.0]); cbb.ax.tick_params(labelsize=5.5)

    fig.savefig(OUT_PDF, dpi=300, bbox_inches='tight')
    fig.savefig(OUT_PNG, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'[ok] {OUT_PDF}')


if __name__ == '__main__':
    main()
