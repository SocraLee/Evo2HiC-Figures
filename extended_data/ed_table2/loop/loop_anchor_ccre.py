#!/usr/bin/env python
"""Loop-anchor classification under the ENCODE registry of cCREs taxonomy
(Moore et al. 2020, Nature 583:699-710; SCREEN), instead of a home-made scheme.

Why this taxonomy: the ENCODE cCRE registry is built from exactly the four "core
assays" DNase-seq, H3K4me3, H3K27ac and CTCF ChIP-seq, and by construction does
NOT use H3K27me3.  It therefore provides an external, published classification in
which Evo2HiC's one weak held-out assay plays no role - no need for us to invent a
reduced taxonomy.

Rules (Moore et al. 2020, "Identifying and classifying cCREs"):
  * "high" is defined as Z > 1.64 ("A Z-score cutoff of 1.64 corresponds to the
    95th percentile for a one-sided Gaussian distribution. A high Z-score or a
    max-Z value is defined as >1.64 throughout, and low otherwise.").  ENCODE
    computes those Z-scores of log10(signal) over the ~2.2 M representative DNase
    hypersensitive sites (rDHSs); we have no per-cell rDHS registry and our unit is
    a fixed 2 kb bin, so we use the equivalent empirical criterion - signal above
    the 95th percentile of that assay's genome-wide 2 kb-bin distribution in that
    same signal source.
  * an element is a cCRE only if DNase is high AND at least one of the three
    ChIP marks is high
  * groups: promoter-like (high H3K4me3); enhancer-like (high H3K27ac, low
    H3K4me3); CTCF-only (high CTCF, low H3K4me3 and H3K27ac)

Deviation from ENCODE, stated explicitly: ENCODE splits promoter-like into PLS
(<=200 bp from a GENCODE TSS) and DNase-H3K4me3 (>200 bp), and enhancer-like into
pELS (<=2 kb) and dELS (>2 kb).  Those distances are far below our 2 kb bin size
and the ~10 kb HiCCUPS anchor, so the TSS-distance partition is collapsed and we
report ENCODE's three signature groups plus a "Low-DNase/inactive" class.

Z-scores are computed separately within each signal source (measured GT, and each
model's own predictions), so no source is advantaged by a mis-scaled output - the
same self-calibration used for the q=0.90 rule in the main analysis.

Usage: loop_anchor_ccre.py [--cells ...] [--boot 1000]
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, cohen_kappa_score, f1_score
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))  # repo root
import paths as P  # noqa: E402

W = str(P.RA_WORK('loop'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from loop_anchor_eval import (T, RES, HALF, TEST_CHRS, METHODS, LOOP_DIR,  # noqa: E402
                              read_bedpe, boot_ci, Track_Loader, hic2tarck_dir)

Q_HIGH = 0.95          # Moore et al. 2020 use Z > 1.64 == 95th percentile
CLASSES = ['Promoter-like', 'Enhancer-like', 'CTCF-only', 'Low-DNase']
EPS = 1e-6


def ccre_class(z):
    """z: dict assay -> {True,False} "high" call at this anchor."""
    if not z['DNase']:
        return 'Low-DNase'
    hi = {m: z[m] for m in ('CTCF', 'H3K27ac', 'H3K4me3')}
    if not any(hi.values()):
        return 'Low-DNase'          # high DNase but no ChIP mark -> not a cCRE
    if hi['H3K4me3']:
        return 'Promoter-like'      # PLS + DNase-H3K4me3 (TSS split not resolvable at 2 kb)
    if hi['H3K27ac']:
        return 'Enhancer-like'      # pELS + dELS
    return 'CTCF-only'


def anchor_high(values, thr, chrom_off, centre_bp, n):
    b = centre_bp // RES
    lo, hi = max(0, b - HALF), min(n, b + HALF + 1)
    return {m: bool(values[m][chrom_off + lo: chrom_off + hi].max() > thr[m]) for m in values}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--cells', default='GM12878,H1ESC,K562')
    p.add_argument('--boot', type=int, default=1000)
    a = p.parse_args()
    rows, dump = [], {}

    for cell in a.cells.split(','):
        loops = read_bedpe(os.path.join(LOOP_DIR[cell], 'merged_loops.bedpe'))
        anchors = sorted({(int(r.chr), int(r.aA_center)) for r in loops.itertuples()} |
                         {(int(r.chr), int(r.aB_center)) for r in loops.itertuples()})

        # --- ground truth: genome-wide Z per assay from the measured tracks
        tl = Track_Loader(os.path.join(hic2tarck_dir, cell), resolution=RES)
        gt_v = {m: np.asarray(tl.values[T[m]], float) for m in T}
        gt_thr = {m: float(np.quantile(gt_v[m], Q_HIGH)) for m in T}
        gt_cls = []
        for c, pb in anchors:
            key = f'chr{c}' if f'chr{c}' in tl.chr_pos else str(c)
            off, end = tl.chr_pos[key]
            gt_cls.append(ccre_class(anchor_high(gt_v, gt_thr, off, pb, end - off)))
        gt_cls = np.array(gt_cls)

        rng = np.random.default_rng(0)
        chance = float(np.mean([(gt_cls == rng.permutation(gt_cls)).mean() for _ in range(200)]))

        for meth, dirf in METHODS.items():
            d = dirf(cell)
            if not all(os.path.exists(f'{d}/{ch}.npy') for ch in TEST_CHRS):
                continue
            pred = {ch: np.load(f'{d}/{ch}.npy') for ch in TEST_CHRS}
            # Z within the model's own test-chromosome distribution
            pool = np.concatenate([pred[ch] for ch in TEST_CHRS], axis=1)
            thr = {m: float(np.quantile(pool[T[m]], Q_HIGH)) for m in T}
            pr_cls = []
            for c, pb in anchors:
                arr = pred[c]
                b = pb // RES
                lo, hi = max(0, b - HALF), min(arr.shape[1], b + HALF + 1)
                z = {m: bool(arr[T[m], lo:hi].max() > thr[m]) for m in T}
                pr_cls.append(ccre_class(z))
            pr_cls = np.array(pr_cls)

            rec = {'cell': cell, 'method': meth, 'n_anchors': len(anchors),
                   'acc': float((gt_cls == pr_cls).mean()),
                   'bacc': balanced_accuracy_score(gt_cls, pr_cls),
                   'kappa': cohen_kappa_score(gt_cls, pr_cls),
                   'chance': chance}
            for cl, f in zip(CLASSES, f1_score(gt_cls, pr_cls, labels=CLASSES,
                                               average=None, zero_division=0)):
                rec[f'F1_{cl}'] = f
            if a.boot:
                lo_, hi_ = boot_ci(gt_cls, pr_cls, lambda x, y: float((x == y).mean()), a.boot)
                rec['acc_lo'], rec['acc_hi'] = lo_, hi_
                lo_, hi_ = boot_ci(gt_cls, pr_cls, cohen_kappa_score, a.boot)
                rec['kappa_lo'], rec['kappa_hi'] = lo_, hi_
            rows.append(rec)
            dump[f'{cell}/{meth}/gt'] = gt_cls
            dump[f'{cell}/{meth}/pred'] = pr_cls
        print(f'{cell}: n_anchors={len(anchors)}  GT class counts '
              + str(dict(zip(*np.unique(gt_cls, return_counts=True)))), flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(f'{W}/loop_anchor_ccre.tsv', sep='\t', index=False, float_format='%.4f')
    np.savez(f'{W}/loop_anchor_ccre_labels.npz', **dump)
    cols = ['cell', 'method', 'n_anchors', 'acc', 'acc_lo', 'acc_hi', 'kappa', 'kappa_lo',
            'kappa_hi', 'bacc', 'chance'] + [f'F1_{c}' for c in CLASSES]
    print(df[[c for c in cols if c in df]].to_string(index=False))


if __name__ == '__main__':
    main()
