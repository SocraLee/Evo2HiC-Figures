"""
Evaluate Borzoi baseline predictions on the epig_predict benchmark.

Reads Borzoi prediction .npy files produced by `borzoi_inference.py`, pulls
ground truth from the same Track_Loader used by inference_CDNA1d.py, and
computes per-(cell, chr, track) Pearson correlation. Output schema matches
inference_CDNA1d.py's result.tsv so rows can be concatenated across methods.
"""

import os
import sys
import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from config import tracks, splits, DNA_map, hic2tarck_dir
from dataset.DNA_loader import DNA_Loader
from dataset.track_loader import Track_Loader

CELLS = ['GM12878', 'H1ESC', 'K562']


def compute_row(pred, gt):
    """
    pred, gt: (n_tracks, n_bins). Returns dict {track_name: pcc}.

    Bins uncovered by Borzoi (where all tracks sum to 0 via nansum) are
    masked out — PCC is computed only over bins Borzoi actually predicted.
    All-NaN tracks (e.g. K562 H3K27ac) return NaN.
    """
    n = min(pred.shape[1], gt.shape[1])
    pred, gt = pred[:, :n], gt[:, :n]

    cov = np.nansum(pred, axis=0) > 0
    out = {}
    for i, tname in enumerate(tracks.keys()):
        p = pred[i]
        if np.all(np.isnan(p)):
            out[tname] = np.nan
            continue
        mask = cov & np.isfinite(p)
        if mask.sum() < 2:
            out[tname] = np.nan
            continue
        p_m, g_m = p[mask], gt[i][mask]
        if np.std(p_m) == 0 or np.std(g_m) == 0:
            out[tname] = np.nan
            continue
        pcc, _ = pearsonr(p_m, g_m)
        out[tname] = float(pcc) if np.isfinite(pcc) else np.nan
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred-dir', type=str,
                        default=str(REPO_ROOT / 'baselines/epigenomic/results/borzoi'),
                        help='Directory containing per-cell subdirs with {chr}.npy')
    parser.add_argument('--out-file', type=str, default=None,
                        help='Output tsv path. Default: {pred-dir}/result.tsv')
    parser.add_argument('--split', type=str, default='test', choices=['train', 'valid', 'test'])
    parser.add_argument('--resolution', type=int, default=2000)
    parser.add_argument('--species', type=str, default='human')
    args = parser.parse_args()

    pred_dir = Path(args.pred_dir)
    out_file = Path(args.out_file) if args.out_file else pred_dir / 'result.tsv'

    dna_loader = DNA_Loader(DNA_map[args.species], 'Yes')
    track_loaders = {c: Track_Loader(os.path.join(hic2tarck_dir, c),
                                     resolution=args.resolution)
                     for c in CELLS}

    rows = []
    for cell in CELLS:
        for ch in splits[args.species][args.split]:
            pred_file = pred_dir / cell / f'{ch}.npy'
            if not pred_file.exists():
                print(f"[SKIP] missing {pred_file}")
                continue
            pred = np.load(pred_file)
            chrom_size = dna_loader.get_size(ch)
            end = int(np.ceil(chrom_size / args.resolution)) * args.resolution
            gt = track_loaders[cell].get(ch, 0, end, 0)
            pccs = compute_row(pred, gt)
            row = {'Name': cell, 'Chr': ch, **pccs}
            rows.append(row)
            pcc_str = "  ".join(f"{tname}={row[tname]:.3f}" if np.isfinite(row[tname]) else f"{tname}=NaN"
                                for tname in tracks.keys())
            print(f"  {cell:8s} chr{ch:<3d}  {pcc_str}")

    df = pd.DataFrame(rows, columns=['Name', 'Chr'] + list(tracks.keys()))
    df.to_csv(out_file, sep='\t', index=False)
    print(f"\nWrote {out_file}  ({len(df)} rows)")

    # Aggregate summary: mean PCC per track across cells/chrs (ignoring NaN)
    print("\nMean PCC per track (NaN-skipping):")
    for t in tracks.keys():
        vals = df[t].values.astype(float)
        valid = np.isfinite(vals)
        mean = np.nanmean(vals) if valid.any() else np.nan
        print(f"  {t:10s}  n={valid.sum():>2}/{len(vals)}  mean={mean:.4f}")


if __name__ == '__main__':
    main()
