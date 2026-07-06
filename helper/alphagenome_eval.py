"""
Leakage-adjusted AlphaGenome-FOLD_1 vs Evo2HiC head-to-head on AG's
strict TEST fold (fold4) restricted to chr9 + chr10.

Methodology (per memory/feedback_strict_test_fold):
  - AlphaGenome FOLD_1 publishes TEST = fold4 (fold3 is its validation fold).
  - The Borzoi BED `~/.cache/borzoi/sequences.bed` is byte-identical to
    AlphaGenome's `fold_intervals.get_fold_intervals(...)` output (verified
    against 55,497 (chr,start,end,fold) tuples), so we use the BED to
    derive the strict-TEST mask without re-querying AlphaGenome.
  - Strict mask = bins inside fold4 AND not overlapping any other fold.
  - chr9 has zero fold4 fragments → only chr10 is non-empty (7,131 bins).
  - Evo2HiC has never seen any of chr9 + chr10 during training, so
    evaluating it on the same bins keeps it fully zero-shot too.

Inputs:
  - AlphaGenome predictions: results/alphagenome_fold1/{cell}/{chr}.npy
  - Evo2HiC predictions:     CKPT_DIR/epi_prediction/model/track/{cell}/{chr}.npy
  - GT via Track_Loader as in inference_CDNA1d.py

Outputs:
  - results/alphagenome_fold1/result_strict_fold4.tsv with columns
    Name (=method), Cell, Chr, DNase, CTCF, H3K27ac, H3K27me3, H3K4me3
  - stdout summary of mean PCC per (method, track) across cells.
"""
import os
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from config import tracks, splits, DNA_map, hic2tarck_dir   # noqa: E402
from dataset.DNA_loader import DNA_Loader                    # noqa: E402
from dataset.track_loader import Track_Loader                # noqa: E402

CELLS = ['GM12878', 'H1ESC', 'K562']
TEST_CHRS = splits['human']['test']     # (9, 10)
RESOLUTION = 2000

CKPT_DIR = Path('/m-chimera/chimera/nobackup/yongkang/HiC_ckpt/checkpoints')
EVO2HIC_DIR = CKPT_DIR / 'epi_prediction/model/track'
AG_DIR = REPO_ROOT / 'baselines/epigenomic/results/alphagenome_fold1'
OUT_TSV = AG_DIR / 'result_strict_fold4.tsv'

# Borzoi BED used as fold ground truth (byte-identical to AlphaGenome's intervals)
SEQBED_URL   = 'https://storage.googleapis.com/seqnn-share/borzoi/hg38/sequences.bed'
SEQBED_CACHE = Path.home() / '.cache' / 'borzoi' / 'sequences.bed'

# AlphaGenome FOLD_1: TEST = fold4 (val = fold3, train = the other 6 folds)
AG_TEST_FOLD = 'fold4'


def fetch_fold_bed():
    SEQBED_CACHE.parent.mkdir(parents=True, exist_ok=True)
    if not SEQBED_CACHE.exists() or SEQBED_CACHE.stat().st_size == 0:
        print(f'Fetching {SEQBED_URL} ...')
        urllib.request.urlretrieve(SEQBED_URL, SEQBED_CACHE)
    df = pd.read_csv(SEQBED_CACHE, sep='\t', header=None,
                     names=['chr', 'start', 'end', 'fold'])
    return df


def strict_test_mask(chrom, n_bins, bed_df, test_fold):
    """True iff 2 kb bin covered by `test_fold` AND not overlapping any other fold."""
    in_test  = np.zeros(n_bins, dtype=bool)
    in_other = np.zeros(n_bins, dtype=bool)
    sub = bed_df[bed_df.chr == f'chr{chrom}']
    for _, r in sub.iterrows():
        b_lo = int(r.start) // RESOLUTION
        b_hi = min(int(np.ceil(int(r.end) / RESOLUTION)), n_bins)
        if b_hi <= b_lo:
            continue
        (in_test if r.fold == test_fold else in_other)[b_lo:b_hi] = True
    return in_test & ~in_other


def pcc_or_nan(p, g):
    if p.size < 2 or np.std(p) == 0 or np.std(g) == 0:
        return np.nan
    r, _ = pearsonr(p, g)
    return float(r) if np.isfinite(r) else np.nan


def main():
    bed = fetch_fold_bed()
    dna_loader = DNA_Loader(DNA_map['human'], 'Yes')
    track_loaders = {c: Track_Loader(os.path.join(hic2tarck_dir, c), RESOLUTION)
                     for c in CELLS}

    rows = []
    for ch in TEST_CHRS:
        chrom_size = dna_loader.get_size(ch)
        n_bins = int(np.ceil(chrom_size / RESOLUTION))
        mask = strict_test_mask(ch, n_bins, bed, AG_TEST_FOLD)
        n_keep = int(mask.sum())
        print(f'\n=== chr{ch}  n_bins={n_bins:,}  fold4-strict_keep={n_keep:,} '
              f'({100*n_keep/n_bins:.1f}%) ===')
        if n_keep == 0:
            print('  (no fold4 bins on this chromosome; skipping)')
            continue

        gt_all = {}
        ag_all = {}
        e2h_all = {}
        for cell in CELLS:
            tl = track_loaders[cell]
            gt = tl.get(ch, 0, n_bins * RESOLUTION, 0)              # (5, n_bins)
            n = min(gt.shape[1], n_bins)
            gt = gt[:, :n]
            gt_all[cell] = gt

            ag = np.load(AG_DIR / cell / f'{ch}.npy')[:, :n]
            ag_all[cell] = ag

            e2h_path = EVO2HIC_DIR / cell / f'{ch}.npy'
            if not e2h_path.exists():
                print(f'  [WARN] missing Evo2HiC pred: {e2h_path}'); continue
            e2h = np.load(e2h_path)[:, :n]
            e2h_all[cell] = e2h

            mask_n = mask[:n]
            for ti, tname in enumerate(tracks.keys()):
                p_ag  = ag[ti]; p_e2h = e2h[ti]; g = gt[ti]
                valid = mask_n & np.isfinite(p_ag) & np.isfinite(p_e2h) & np.isfinite(g)
                if valid.sum() < 2:
                    pcc_ag = pcc_e2h = np.nan; n_use = 0
                else:
                    n_use = int(valid.sum())
                    pcc_ag  = pcc_or_nan(p_ag [valid], g[valid])
                    pcc_e2h = pcc_or_nan(p_e2h[valid], g[valid])
                rows.append({'Method': 'AlphaGenome_FOLD_1',
                             'Cell': cell, 'Chr': ch, 'Track': tname,
                             'PCC': pcc_ag, 'N': n_use})
                rows.append({'Method': 'Evo2HiC',
                             'Cell': cell, 'Chr': ch, 'Track': tname,
                             'PCC': pcc_e2h, 'N': n_use})

    df = pd.DataFrame(rows)
    AG_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_TSV, sep='\t', index=False)
    print(f'\nWrote {OUT_TSV}  ({len(df)} rows)')

    # Pivoted summary
    print('\nPer-cell per-track PCC on fold4-strict mask:')
    piv = df.pivot_table(index=['Cell', 'Track'], columns='Method',
                         values='PCC', aggfunc='mean')  # mean across chr (only chr10 has data)
    print(piv.to_string(float_format=lambda x: f'{x:.4f}'))

    print('\nMean PCC across cells (NaN-skipping), per track:')
    by_track = df.pivot_table(index='Track', columns='Method', values='PCC', aggfunc='mean')
    print(by_track.to_string(float_format=lambda x: f'{x:.4f}'))


if __name__ == '__main__':
    main()
