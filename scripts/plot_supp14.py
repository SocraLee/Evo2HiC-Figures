"""Supplementary Figure 14 — the rule on PREDICTED epigenomics recovers the
ground-truth loop-anchor classification.

Companion to Supp 15. Every ground-truth HiCCUPS loop anchor (chr9 + chr10,
held out) is classified by a fixed rule (mark+ iff signal >= the genome-wide
90th percentile, max over the anchor's ±2 bins / 10 kb). Applying the SAME
rule to the model's PREDICTED tracks vs to the real ChIP-seq gives the
confusion matrices here. CTCF is an ORTHOGONAL architectural dimension (present
at most anchors), so it is shown separately from the functional state:

  top row    : 3-class functional confusion (Active / Repressive / Quiescent)
  bottom row : CTCF as a binary dimension (CTCF- / CTCF+)
  columns    : cell lines (GM12878, H1ESC)

Cells are row-normalised (recall); "Predicted" = rule on the model's predicted
epigenomics, ground truth = rule on real ChIP-seq.

Generating the data (see paths.py "Hi-C loop functional decomposition" for the
exact commands; pipeline under revision/loop_decomposition/, Evo2HiC env):
  LOOP_RULE_NPZ : produced by
      label_loops.py        (HiCCUPS loops -> GT mark-calls -> loop_labels.tsv)
   -> rule_on_predicted.py  (same rule on predicted tracks -> this .npz)
  The .npz holds gt_func3 / pred_func3 / gt_ctcf / pred_ctcf per cell line.
This figure reads ONLY that .npz (no track / Hi-C access needed).

Outputs
-------
Figures/supplementary_14.pdf
Figures/supplementary_14.png
stdout : per-cell functional balanced accuracy + CTCF F1.
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from sklearn.metrics import confusion_matrix, balanced_accuracy_score, f1_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paths import ensure_out_dir, LOOP_RULE_NPZ                  # noqa: E402

OUT_PDF = ensure_out_dir() / 'supplementary_14.pdf'
OUT_PNG = ensure_out_dir() / 'supplementary_14.png'

CELLS = ['GM12878', 'H1ESC']
F3 = ['Active', 'Repressive', 'Quiescent']
CT = ['CTCF−', 'CTCF+']

plt.style.use('seaborn-v0_8-white')
plt.rcParams.update({'font.size': 9, 'font.family': 'Arial',
                     'pdf.fonttype': 42, 'ps.fonttype': 42})


def panel(ax, gt, pr, order, ticks, show_y, show_x, title):
    cm = confusion_matrix(gt, pr, labels=order).astype(float)
    rs = cm.sum(1, keepdims=True); rs[rs == 0] = 1
    cmn = cm / rs
    im = ax.imshow(cmn, cmap='Blues', vmin=0, vmax=1)  # aspect='equal' default -> square cells
    n = len(order)
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(ticks if show_x else [], rotation=35, ha='right', fontsize=8.5)
    ax.set_yticklabels(ticks if show_y else [], fontsize=8.5)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f'{cmn[i,j]:.2f}', ha='center', va='center', fontsize=8.5,
                    color='white' if cmn[i, j] > 0.55 else 'black')
    ax.set_title(title, fontsize=9)
    if show_x:
        ax.set_xlabel('Predicted', fontsize=9)
    for s in ax.spines.values():
        s.set_linewidth(0.5)
    return im


def main():
    d = np.load(LOOP_RULE_NPZ, allow_pickle=True)
    fig = plt.figure(figsize=(6.4, 5.8))
    # rows = [function (3-class), CTCF (binary)]; columns = cell lines
    gs = GridSpec(2, 2, height_ratios=[3, 2], wspace=0.4, hspace=0.28,
                  left=0.17, right=0.87, top=0.9, bottom=0.12)
    im = None
    for ci, cell in enumerate(CELLS):
        gf, pf = d[f'{cell}/gt_func3'], d[f'{cell}/pred_func3']
        gc = np.where(d[f'{cell}/gt_ctcf'], 'CTCF+', 'CTCF−')
        pc = np.where(d[f'{cell}/pred_ctcf'], 'CTCF+', 'CTCF−')
        bacc = balanced_accuracy_score(gf, pf)
        f1 = f1_score(d[f'{cell}/gt_ctcf'], d[f'{cell}/pred_ctcf'])
        im = panel(fig.add_subplot(gs[0, ci]), gf, pf, F3, F3,
                   show_y=(ci == 0), show_x=False, title=f'{cell}\naccuracy={bacc:.2f}')
        panel(fig.add_subplot(gs[1, ci]), gc, pc, CT, CT,
              show_y=(ci == 0), show_x=True, title=f'F1={f1:.2f}')
        print(f'[{cell}] function balanced-acc={bacc:.3f}  CTCF F1={f1:.3f}')
    fig.text(0.035, 0.66, 'Functional', rotation=90, va='center', ha='center',
             fontsize=10, fontweight='bold')
    fig.text(0.035, 0.27, 'CTCF (structural)', rotation=90, va='center', ha='center',
             fontsize=10, fontweight='bold')
    cax = fig.add_axes([0.89, 0.3, 0.014, 0.4])
    fig.colorbar(im, cax=cax)
    fig.savefig(OUT_PDF, dpi=300, bbox_inches='tight')
    fig.savefig(OUT_PNG, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'[ok] {OUT_PDF}')


if __name__ == '__main__':
    main()
