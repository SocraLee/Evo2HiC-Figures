#!/usr/bin/env python
"""Evaluate HiCFoundation epigenomic predictions against the Evo2HiC ground truth
with exactly the Evo2HiC protocol: per-chromosome Pearson r at 2 kb between the
predicted track and the cutoff-normalised GT track used by inference_CDNA1d.py.

HiCFoundation predicts at 1 kb (6 assays: CTCF, H3K4me3, H3K27ac, H3K27me3,
ATAC-seq, DNase-seq); we average adjacent 1 kb bins to 2 kb and keep the 5
assays shared with Evo2HiC.  Pearson r is scale-invariant so the different
output scales do not matter; the GT clipping (cutoff) is applied identically to
every method because it lives in the GT.

Run from the Evo2HiC repo root in the Evo2HiC environment:
  python plot_revision/extended_data/ed_table1/eval_epi.py <cell> <hicf_out_dir> [<evo2hic_track_dir>]
Writes <hicf_out_dir>/result.tsv (rows: chr 9, 10; cols: 5 tracks) and prints a
side-by-side with Evo2HiC if its track dir is given.
"""
import os
import pickle
import sys

import numpy as np
from scipy.stats import pearsonr
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))  # repo root (paths.py)
import paths as P  # noqa: E402

sys.path.insert(0, str(P.REPO))
from config import hic2tarck_dir, tracks  # noqa: E402
from dataset.track_loader import Track_Loader  # noqa: E402

RES = 2000
HF_ORDER = ['CTCF', 'H3K4me3', 'H3K27ac', 'H3K27me3', 'ATAC-seq', 'DNase-seq']
OUR_ORDER = list(tracks.keys())              # DNase, CTCF, H3K27ac, H3K27me3, H3K4me3
HF_NAME = {'DNase': 'DNase-seq'}
CHROMS = (9, 10)


def hf_to_2kb(arr6, n_bins):
    """(6, L@1kb) -> (5, n_bins@2kb) in OUR_ORDER by mean-pooling pairs of 1 kb bins."""
    out = np.zeros((5, n_bins), np.float32)
    for i, name in enumerate(OUR_ORDER):
        j = HF_ORDER.index(HF_NAME.get(name, name))
        v = arr6[j].astype(np.float64)
        L = min(len(v) // 2 * 2, n_bins * 2)
        pooled = v[:L].reshape(-1, 2).mean(axis=1)
        out[i, :len(pooled)] = pooled
    return out


def main(cell, hf_dir, evo_dir=None):
    pred = pickle.load(open(os.path.join(hf_dir, 'HiCFoundation_epigenomic_assay_prediction.pkl'), 'rb'))
    gt_loader = Track_Loader(os.path.join(hic2tarck_dir, cell), RES)
    rows = []
    for ch in CHROMS:
        key = f'chr{ch}'
        lo, hi = gt_loader.chr_pos[key]
        n_bins = hi - lo
        gt = gt_loader.get(key, 0, n_bins * RES, 0).astype(np.float64)      # (5, n_bins)
        pkey = key if key in pred else str(ch)
        hf = hf_to_2kb(np.asarray(pred[pkey]), n_bins)
        r_hf = [pearsonr(hf[i], gt[i])[0] for i in range(5)]
        row = {'Name': 'HiCFoundation', 'Chr': ch, **{t: r for t, r in zip(OUR_ORDER, r_hf)}}
        rows.append(row)
        if evo_dir:
            ev = np.load(os.path.join(evo_dir, cell, f'{ch}.npy')).astype(np.float64)
            n = min(ev.shape[1], gt.shape[1])
            r_ev = [pearsonr(ev[i, :n], gt[i, :n])[0] for i in range(5)]
            rows.append({'Name': 'Evo2HiC', 'Chr': ch, **{t: r for t, r in zip(OUR_ORDER, r_ev)}})
    import pandas as pd
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(hf_dir, 'result.tsv'), sep='\t', index=False, float_format='%.4f')
    print(cell)
    print(df.to_string(index=False, float_format=lambda x: f'{x:.4f}'))
    print('\nmean over chr9+10:')
    print(df.groupby('Name')[OUR_ORDER].mean().to_string(float_format=lambda x: f'{x:.4f}'))


if __name__ == '__main__':
    main(*sys.argv[1:4])
