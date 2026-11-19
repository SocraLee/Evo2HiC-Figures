#!/usr/bin/env python
"""Loop-anchor classification from predicted epigenomic tracks, extended for
revision: adds the held-out cell line (K562) and the Hi-C-only and
sequence-only (Evo2) baselines requested by reviewers 1/3.

Rule (identical for ground truth and every model, from round 1's
revision/loop_decomposition/rule_on_predicted.py):
  * anchors = HiCCUPS loop anchors on chr9+chr10 of the real Hi-C
  * per anchor, signal = max over centre +-2 bins (~10 kb) at 2 kb resolution
  * a mark is "+" iff signal >= that track's genome-wide q=0.90 quantile,
    computed within the same signal source (measured GT vs each model's own
    predictions), so no source gets an unfair threshold
  * anchor state -> 6 states -> 4 groups; loop class from the two anchor states

Reported per (cell, method): anchor 6-state and 4-group accuracy / balanced
accuracy, loop-class accuracy / balanced accuracy, plus a random-permutation
baseline and bootstrap CIs.

Usage: loop_anchor_eval.py [--cells GM12878,H1ESC,K562] [--boot 1000]
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))  # repo root
import paths as P  # noqa: E402

R1 = str(P.REPO)
sys.path.insert(0, R1)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # rule_on_predicted.py
from config import tracks, hic2tarck_dir  # noqa: E402
from dataset.track_loader import Track_Loader  # noqa: E402
from rule_on_predicted import (anchor_state, func3, loop_class, GRP, GROUPS)  # noqa: E402

CK = str(P.CKPT_ROOT)
W = str(P.RA_WORK('loop'))
MARKS = list(tracks.keys())
T = {m: i for i, m in enumerate(MARKS)}
RES, GENOME_Q, HALF = 2000, 0.90, 2
TEST_CHRS = (9, 10)

CELL_HIC = {'GM12878': '4DNFI1UEG1HD', 'H1ESC': '4DNFIQYQWPF5', 'K562': '4DNFITUOMFUQ'}
LOOP_DIR = {'GM12878': os.path.join(P.HIC_DATA_ROOT, 'data/hic/hiccups_loop/4DNFI1UEG1HD'),
            'H1ESC':   os.path.join(P.HIC_DATA_ROOT, 'data/hic/hiccups_loop/4DNFIQYQWPF5'),
            'K562':    f'{W}/hiccups/4DNFITUOMFUQ'}
# predicted 5-track npys: <dir>/<cell>/<chr>.npy
METHODS = {
    'Evo2HiC':       lambda c: f'{P.EPI_EVO2HIC_DIR}/{c}',
    'HiC-only':      lambda c: f'{P.EPI_HIC_ONLY_DIR}/{c}',        # shipped predictions, chr9/10
    'Evo2 (seq-only)': lambda c: str(P.EPI_EVO2_DIR(c)),          # shipped predictions, chr9/10
}


def read_bedpe(path):
    df = pd.read_csv(path, sep='\t', skiprows=[1], dtype={'#chr1': str, 'chr2': str})
    df = df[df['#chr1'] == df['chr2']].copy()
    df['chr'] = df['#chr1'].str.replace('chr', '', regex=False)
    df = df[df['chr'].isin([str(c) for c in TEST_CHRS])].copy()
    df['aA_center'] = ((df.x1.astype(int) + df.x2.astype(int)) // 2 // RES) * RES
    df['aB_center'] = ((df.y1.astype(int) + df.y2.astype(int)) // 2 // RES) * RES
    return df[['chr', 'aA_center', 'aB_center']].astype({'chr': int}).reset_index(drop=True)


def calls_from(sig, thr):
    return {m: bool(sig[T[m]] >= thr[m]) for m in MARKS}


def boot_ci(a, b, fn, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    v = [fn(a[i], b[i]) for i in (rng.integers(0, len(a), len(a)) for _ in range(n))]
    return np.percentile(v, [2.5, 97.5])


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--cells', default='GM12878,H1ESC,K562')
    p.add_argument('--boot', type=int, default=1000)
    a = p.parse_args()
    os.makedirs(W, exist_ok=True)
    rows, dump = [], {}

    for cell in a.cells.split(','):
        loops = read_bedpe(os.path.join(LOOP_DIR[cell], 'merged_loops.bedpe'))
        anchors = sorted({(int(r.chr), int(r.aA_center)) for r in loops.itertuples()} |
                         {(int(r.chr), int(r.aB_center)) for r in loops.itertuples()})
        tl = Track_Loader(os.path.join(hic2tarck_dir, cell), resolution=RES)
        gt_thr = {m: float(np.quantile(np.asarray(tl.values[T[m]], float), GENOME_Q)) for m in MARKS}
        gt_c = [calls_from(tl.get(c, max((pb // RES - HALF) * RES, 0),
                                 (pb // RES + HALF + 1) * RES, 0).max(axis=1), gt_thr)
                for c, pb in anchors]
        gt_s = np.array([anchor_state(c) for c in gt_c])
        gt4 = np.array([GRP[s] for s in gt_s])
        gt_lc = np.array([loop_class(gt_s[anchors.index((int(r.chr), int(r.aA_center)))],
                                     gt_s[anchors.index((int(r.chr), int(r.aB_center)))])
                          for r in loops.itertuples()])

        for meth, dirf in METHODS.items():
            d = dirf(cell)
            if not all(os.path.exists(f'{d}/{ch}.npy') for ch in TEST_CHRS):
                print(f'[skip] {cell} {meth}: missing {d}')
                continue
            pred = {ch: np.load(f'{d}/{ch}.npy') for ch in TEST_CHRS}
            pool = np.concatenate([pred[ch] for ch in TEST_CHRS], axis=1)
            thr = {m: float(np.quantile(pool[T[m]], GENOME_Q)) for m in MARKS}
            pr_c = [calls_from(pred[c][:, max(pb // RES - HALF, 0): pb // RES + HALF + 1].max(axis=1), thr)
                    for c, pb in anchors]
            pr_s = np.array([anchor_state(c) for c in pr_c])
            pr4 = np.array([GRP[s] for s in pr_s])
            pr_lc = np.array([loop_class(pr_s[anchors.index((int(r.chr), int(r.aA_center)))],
                                         pr_s[anchors.index((int(r.chr), int(r.aB_center)))])
                              for r in loops.itertuples()])
            rec = {'cell': cell, 'method': meth, 'n_anchors': len(anchors), 'n_loops': len(loops),
                   'anchor6_acc': float((gt_s == pr_s).mean()),
                   'anchor6_bacc': balanced_accuracy_score(gt_s, pr_s),
                   'anchor4_acc': float((gt4 == pr4).mean()),
                   'anchor4_bacc': balanced_accuracy_score(gt4, pr4),
                   'loop_acc': float((gt_lc == pr_lc).mean()),
                   'loop_bacc': balanced_accuracy_score(gt_lc, pr_lc)}
            if a.boot:
                lo, hi = boot_ci(gt4, pr4, lambda x, y: float((x == y).mean()), a.boot)
                rec['anchor4_acc_lo'], rec['anchor4_acc_hi'] = lo, hi
                lo, hi = boot_ci(gt4, pr4, balanced_accuracy_score, a.boot)
                rec['anchor4_bacc_lo'], rec['anchor4_bacc_hi'] = lo, hi
            rows.append(rec)
            dump[f'{cell}/{meth}/gt4'] = gt4
            dump[f'{cell}/{meth}/pred4'] = pr4
            dump[f'{cell}/{meth}/gt_loop'] = gt_lc
            dump[f'{cell}/{meth}/pred_loop'] = pr_lc
            print(f'{cell:8s} {meth:16s} anchors={len(anchors):4d} loops={len(loops):4d} '
                  f'a4_acc={rec["anchor4_acc"]:.3f} a4_bacc={rec["anchor4_bacc"]:.3f} '
                  f'loop_acc={rec["loop_acc"]:.3f}', flush=True)
        # label-shuffle chance level (per cell, on the 4-group task)
        rng = np.random.default_rng(0)
        ch_acc = np.mean([(gt4 == rng.permutation(gt4)).mean() for _ in range(200)])
        rows.append({'cell': cell, 'method': 'chance (label shuffle)', 'n_anchors': len(anchors),
                     'n_loops': len(loops), 'anchor4_acc': ch_acc})
        print(f'{cell:8s} {"chance":16s} a4_acc={ch_acc:.3f}', flush=True)
        # confusion for Evo2HiC
        if f'{cell}/Evo2HiC/pred4' in dump:
            cm = confusion_matrix(gt4, dump[f'{cell}/Evo2HiC/pred4'], labels=GROUPS)
            print(f'  {cell} Evo2HiC anchor-4 confusion (rows=GT):')
            print('        ' + ' '.join(f'{l[:5]:>6s}' for l in GROUPS))
            for i, l in enumerate(GROUPS):
                print(f'    {l[:5]:>5s} ' + ' '.join(f'{cm[i, j]:6d}' for j in range(len(GROUPS))))

    df = pd.DataFrame(rows)
    df.to_csv(f'{W}/loop_anchor_metrics.tsv', sep='\t', index=False, float_format='%.4f')
    np.savez(f'{W}/loop_anchor_labels.npz', **dump)
    print('\n' + df.to_string(index=False))


if __name__ == '__main__':
    main()
