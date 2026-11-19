"""rule_on_predicted.py — apply the GT classification rule to the model's PREDICTED
epigenomic signal, and measure how well it reproduces the GT loop / anchor class.

Same rule for both, differing only in signal source:
  state(anchor) from binary mark calls; mark+ iff signal >= genome-wide q=0.90
    GT  signal  = real ChIP-seq (Track_Loader), genome threshold from Track_Loader
    PRED signal = model reconstruction (HiC_save/epig npy), threshold from the
                  predicted chr9+chr10 distribution
Anchor signal = max over center +/-2 bins (~10kb), identical windows for both.

Reports anchor-state and 4-group accuracy / balanced-acc + confusion matrices,
and loop-level class accuracy.
"""
import os
import sys
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))  # repo root
import paths as P  # noqa: E402

sys.path.insert(0, str(P.REPO))
from config import tracks, hic2tarck_dir
from dataset.track_loader import Track_Loader
from sklearn.metrics import balanced_accuracy_score, confusion_matrix

HERE = os.path.dirname(os.path.abspath(__file__))
# Canonical shipped predicted 5-track epigenomics (== plot_revision/paths.py
# EPI_EVO2HIC_DIR / LOOP_PRED_TRACK; the same npys Fig. 4 reads).
PRED_DIR = str(P.EPI_EVO2HIC_DIR)
CELLS = {'GM12878': '4DNFI1UEG1HD', 'H1ESC': '4DNFIQYQWPF5'}
MARKS = list(tracks.keys())
T = {m: i for i, m in enumerate(MARKS)}
RES = 2000
GENOME_Q = 0.90
HALF = 2  # center +/-2 bins ~ 10kb
TEST_CHRS = (9, 10)

GRP = {'CTCF': 'CTCF', 'Promoter': 'Active', 'Enhancer': 'Active',
       'Bivalent': 'Repressive', 'Polycomb': 'Repressive', 'Quiescent': 'Quiescent'}
STATES = ['CTCF', 'Promoter', 'Enhancer', 'Bivalent', 'Polycomb', 'Quiescent']
GROUPS = ['CTCF', 'Active', 'Repressive', 'Quiescent']


def anchor_state(c):
    if c['H3K27me3'] and c['H3K4me3']: return 'Bivalent'
    if c['H3K27me3']: return 'Polycomb'
    if c['H3K4me3']: return 'Promoter'
    if c['H3K27ac']: return 'Enhancer'
    if c['CTCF']: return 'CTCF'
    return 'Quiescent'


def func3(c):
    """Functional state, with CTCF treated as a SEPARATE orthogonal dimension."""
    if c['H3K27me3']: return 'Repressive'
    if c['H3K4me3'] or c['H3K27ac']: return 'Active'
    return 'Quiescent'


def loop_class(sA, sB):
    """Same CTCF-first loop taxonomy used for the GT labels, on two anchor states."""
    # CTCF structural if both anchors CTCF-state OR (handled upstream) — here we
    # approximate with anchor states: both CTCF -> CTCF-structural.
    states = {sA, sB}
    if sA == 'CTCF' and sB == 'CTCF':
        return 'CTCF-structural'
    if 'Bivalent' in states:
        return 'Bivalent'
    if sA == 'Polycomb' and sB == 'Polycomb':
        return 'Polycomb'
    if states == {'Promoter', 'Enhancer'} or (sA == 'Promoter' and sB == 'Promoter'):
        return 'Active-regulatory'
    if states & {'Promoter', 'Enhancer'}:
        return 'Active-regulatory'
    if 'CTCF' in states:
        return 'CTCF-structural'
    return 'Quiescent'


def anchor_max(arr2d, center_bin):
    lo = max(0, center_bin - HALF); hi = min(arr2d.shape[1], center_bin + HALF + 1)
    return arr2d[:, lo:hi].max(axis=1)


def main():
    df = pd.read_csv(P.LOOP_LABELS_TSV, sep='\t')
    summary = []
    labels_out = {}
    for cell, acc in CELLS.items():
        tl = Track_Loader(os.path.join(hic2tarck_dir, cell), resolution=RES)
        gt_thr = {m: float(np.quantile(np.asarray(tl.values[T[m]], float), GENOME_Q)) for m in MARKS}

        # predicted full tracks for chr9,10 + predicted genome-wide thresholds
        pred = {ch: np.load(os.path.join(PRED_DIR, cell, f'{ch}.npy')) for ch in TEST_CHRS}
        pred_pool = np.concatenate([pred[ch] for ch in TEST_CHRS], axis=1)  # (5, totbins)
        pr_thr = {m: float(np.quantile(pred_pool[T[m]], GENOME_Q)) for m in MARKS}

        g = df[df['cell'] == cell].reset_index(drop=True)

        def calls_of(chrom, center_bp, source):
            cb = center_bp // RES
            if source == 'gt':
                s = (cb - HALF) * RES; e = (cb + HALF + 1) * RES
                sig = tl.get(int(chrom), max(s, 0), e, 0).max(axis=1); thr = gt_thr
            else:
                sig = anchor_max(pred[int(chrom)], cb); thr = pr_thr
            return {m: bool(sig[T[m]] >= thr[m]) for m in MARKS}

        # anchor-level (unique anchors)
        anchors = set()
        for r in g.itertuples():
            anchors.add((int(r.chr), int(r.aA_center)))
            anchors.add((int(r.chr), int(r.aB_center)))
        anchors = sorted(anchors)
        gt_c = [calls_of(c, p, 'gt') for c, p in anchors]
        pr_c = [calls_of(c, p, 'pred') for c, p in anchors]
        gt_s = np.array([anchor_state(c) for c in gt_c])
        pr_s = np.array([anchor_state(c) for c in pr_c])
        # functional 3-class + CTCF as a separate orthogonal binary dimension
        gt_f3 = np.array([func3(c) for c in gt_c]); pr_f3 = np.array([func3(c) for c in pr_c])
        gt_ct = np.array([c['CTCF'] for c in gt_c]); pr_ct = np.array([c['CTCF'] for c in pr_c])
        a_acc = float((gt_s == pr_s).mean())
        a_bacc = balanced_accuracy_score(gt_s, pr_s)
        g4 = np.array([GRP[s] for s in gt_s]); p4 = np.array([GRP[s] for s in pr_s])
        a4_acc = float((g4 == p4).mean()); a4_bacc = balanced_accuracy_score(g4, p4)

        # loop-level
        astate = {a: (gt_s[i], pr_s[i]) for i, a in enumerate(anchors)}
        gt_lc, pr_lc = [], []
        for r in g.itertuples():
            ka = (int(r.chr), int(r.aA_center)); kb = (int(r.chr), int(r.aB_center))
            gt_lc.append(loop_class(astate[ka][0], astate[kb][0]))
            pr_lc.append(loop_class(astate[ka][1], astate[kb][1]))
        gt_lc, pr_lc = np.array(gt_lc), np.array(pr_lc)
        l_acc = float((gt_lc == pr_lc).mean()); l_bacc = balanced_accuracy_score(gt_lc, pr_lc)

        print(f'\n==== {cell} — rule on PREDICTED signal vs GT rule ====')
        print(f'  anchor 6-state:  acc={a_acc:.3f}  balanced={a_bacc:.3f}  (n={len(anchors)})')
        print(f'  anchor 4-group:  acc={a4_acc:.3f}  balanced={a4_bacc:.3f}')
        print(f'  loop class:      acc={l_acc:.3f}  balanced={l_bacc:.3f}  (n={len(g)})')
        labs = [s for s in GROUPS]
        cm = confusion_matrix(g4, p4, labels=labs)
        print('  anchor 4-group confusion (rows=GT, cols=Pred):')
        print('        ' + ' '.join(f'{l[:5]:>6s}' for l in labs))
        for i, l in enumerate(labs):
            print(f'    {l[:5]:>5s} ' + ' '.join(f'{cm[i,j]:6d}' for j in range(len(labs))))
        summary.append({'cell': cell, 'anchor6_acc': a_acc, 'anchor6_bacc': a_bacc,
                        'anchor4_acc': a4_acc, 'anchor4_bacc': a4_bacc,
                        'loop_acc': l_acc, 'loop_bacc': l_bacc})
        labels_out[f'{cell}/gt4'] = g4; labels_out[f'{cell}/pred4'] = p4
        labels_out[f'{cell}/gt6'] = gt_s; labels_out[f'{cell}/pred6'] = pr_s
        labels_out[f'{cell}/gt_loop'] = gt_lc; labels_out[f'{cell}/pred_loop'] = pr_lc
        labels_out[f'{cell}/gt_func3'] = gt_f3; labels_out[f'{cell}/pred_func3'] = pr_f3
        labels_out[f'{cell}/gt_ctcf'] = gt_ct; labels_out[f'{cell}/pred_ctcf'] = pr_ct

    pd.DataFrame(summary).to_csv(P.LOOP_RULE_METRICS_TSV,
                                 sep='\t', index=False)
    np.savez(P.LOOP_RULE_NPZ, **labels_out)
    print('\n[ok] wrote rule_on_predicted_metrics.tsv + rule_on_predicted_labels.npz')


if __name__ == '__main__':
    main()
