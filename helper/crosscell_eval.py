"""
Cross-cell-type evaluation for epigenomic track prediction (R3).

Three reported metrics, all on chr9 + chr10 (test split), 2 kb bins,
cells = {GM12878, H1ESC, K562}, tracks = {DNase, CTCF, H3K27ac, H3K27me3, H3K4me3}:

  1. cross-cell 3x3 PCC matrix per (model, track):
       M[i, j] = corr(pred_cellsi, gt_cellsj)
     diagonal  = matched-cell PCC (existing metric)
     off-diag  = mismatched-cell PCC; close to diagonal => no cell specificity

  2. Δ-signal SpecPCC per (model, track, pair):
       SpecPCC = corr( pred_A - pred_B,  gt_A - gt_B )

  3. Williams test (Steiger 1980, meta-analysed across chroms) on Δ-signal:
       H0: SpecPCC_Evo2HiC == SpecPCC_baseline      (baseline ∈ {Evo2, HiC_only})

Representative loci are written for downstream pyGenomeTracks/IGV plotting.
"""
import os
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from itertools import permutations

from dataset.track_loader import Track_Loader
from config import hic2tarck_dir, tracks as TRACK_CFG, splits

from evaluate.correlation_epi import (
    dir_Evo2, dir_Evo2HiC, dir_HiC_only,
    meta_compare_two_experiments,
)

# --------------------------------------------------------------------------
CELLS  = ['GM12878', 'H1ESC', 'K562']
TRACKS = list(TRACK_CFG.keys())              # 5 tracks, order matches .npy axis 0
CHROMS = list(splits['human']['test'])       # [9, 10]
RESOLUTION = 2000

MODELS = {
    'Evo2':     dir_Evo2,
    'HiC_only': dir_HiC_only,
    'Evo2HiC':  dir_Evo2HiC,
}

OUT_DIR = 'result/crosscell'
os.makedirs(OUT_DIR, exist_ok=True)

# --------------------------------------------------------------------------
# 1. Load predictions and ground-truth into uniform arrays.
#    pred[m][cell] = (5, n_bins_total)   — concatenated over CHROMS
#    gt[cell]      = (5, n_bins_total)
#    chrom_bin_offsets gives [0, n9, n9+n10] for per-chrom slicing.
# --------------------------------------------------------------------------
def load_all():
    gt_loader = {c: Track_Loader(os.path.join(hic2tarck_dir, c), RESOLUTION) for c in CELLS}

    chrom_bins = {}
    pred = {m: {c: [] for c in CELLS} for m in MODELS}
    gt   = {c: [] for c in CELLS}

    for ch in CHROMS:
        # determine n_bins from any prediction npy (all share it for a chrom)
        ref = np.load(os.path.join(MODELS['Evo2HiC']['GM12878'], f'{ch}.npy'))
        n_bins = ref.shape[1]
        chrom_bins[ch] = n_bins

        for c in CELLS:
            t = gt_loader[c].get(ch, 0, n_bins * RESOLUTION, 0)        # (5, n_bins)
            gt[c].append(t.astype(np.float32))
            for m in MODELS:
                p = np.load(os.path.join(MODELS[m][c], f'{ch}.npy')).astype(np.float32)
                assert p.shape == (len(TRACKS), n_bins), (m, c, ch, p.shape)
                pred[m][c].append(p)

    pred = {m: {c: np.concatenate(pred[m][c], axis=1) for c in CELLS} for m in MODELS}
    gt   = {c: np.concatenate(gt[c], axis=1) for c in CELLS}
    return pred, gt, chrom_bins


# --------------------------------------------------------------------------
# 2. 3x3 cross-cell PCC matrix per (model, track) + GT self-similarity matrix.
# --------------------------------------------------------------------------
def cross_cell_matrix(pred, gt):
    rows = []
    # GT self-similarity (sanity: how distinct are the experimental tracks?)
    for ti, tname in enumerate(TRACKS):
        for i, ca in enumerate(CELLS):
            for j, cb in enumerate(CELLS):
                r = pearsonr(gt[ca][ti], gt[cb][ti])[0]
                rows.append({'model': 'GT', 'track': tname,
                             'pred_cell': ca, 'gt_cell': cb, 'pcc': r})

    for m in MODELS:
        for ti, tname in enumerate(TRACKS):
            for i, ca in enumerate(CELLS):
                for j, cb in enumerate(CELLS):
                    r = pearsonr(pred[m][ca][ti], gt[cb][ti])[0]
                    rows.append({'model': m, 'track': tname,
                                 'pred_cell': ca, 'gt_cell': cb, 'pcc': r})
    return pd.DataFrame(rows)


def specificity_gap(matrix_df):
    """mean(diag) - mean(off-diag) per (model, track)."""
    rows = []
    for (m, t), sub in matrix_df.groupby(['model', 'track']):
        diag = sub[sub.pred_cell == sub.gt_cell].pcc.mean()
        off  = sub[sub.pred_cell != sub.gt_cell].pcc.mean()
        rows.append({'model': m, 'track': t,
                     'diag_mean': diag, 'offdiag_mean': off, 'gap': diag - off})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# 3. Δ-signal SpecPCC per (model, track, pair) + Williams test.
#
#    For pair (A, B), per chromosome we compute:
#        rab_m   = corr( Δpred_m,  Δgt )            # signal of interest
#        rac_m   = corr( Δpred_m', Δgt )            # the competitor
#        rbc_mm' = corr( Δpred_m,  Δpred_m' )       # dependence between models
#    and feed these (per-chrom) into meta_compare_two_experiments to get
#    a single Z / p across chroms.
# --------------------------------------------------------------------------
def delta_arrays(pred, gt, chrom_bins):
    """{('A','B'): {'gt': (5, n), m: (5, n)}}, concat over chroms."""
    deltas = {}
    cum = np.cumsum([0] + [chrom_bins[ch] for ch in CHROMS])  # boundaries
    for A, B in permutations(CELLS, 2):
        d = {'gt': gt[A] - gt[B]}
        for m in MODELS:
            d[m] = pred[m][A] - pred[m][B]
        deltas[(A, B)] = d
    return deltas, cum


def spec_pcc_table(deltas, cum):
    rows = []
    for (A, B), d in deltas.items():
        for ti, tname in enumerate(TRACKS):
            row = {'pair': f'{A}_vs_{B}', 'track': tname}
            for m in MODELS:
                row[f'specpcc_{m}'] = pearsonr(d[m][ti], d['gt'][ti])[0]
            rows.append(row)
    return pd.DataFrame(rows)


def williams_table(deltas, cum):
    """Per (pair, track), test Evo2HiC > {HiC_only, Evo2} via per-chrom meta-Williams."""
    rows = []
    n_chroms = len(CHROMS)

    for (A, B), d in deltas.items():
        for ti, tname in enumerate(TRACKS):
            for baseline in ['Evo2', 'HiC_only']:
                stats = []                          # per-chrom [rab, rac, rbc, n]
                for k in range(n_chroms):
                    lo, hi = cum[k], cum[k+1]
                    a = d['Evo2HiC'][ti, lo:hi]
                    b = d[baseline ][ti, lo:hi]
                    t = d['gt'     ][ti, lo:hi]
                    rab = pearsonr(a, t)[0]
                    rac = pearsonr(b, t)[0]
                    rbc = pearsonr(a, b)[0]
                    stats.extend([rab, rac, rbc, hi - lo])

                try:
                    res = meta_compare_two_experiments(*stats, alternative='greater')
                    p   = res['p']
                    z   = res['Z']
                except Exception as e:
                    p, z = float('nan'), float('nan')

                rows.append({'pair': f'{A}_vs_{B}', 'track': tname,
                             'baseline': baseline, 'Z': z, 'p_one_sided': p})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# 4. Representative loci. For each (track, pair_AB), pick top-k bins where:
#       Evo2HiC reproduces a strong cell-specific GT difference, and
#       HiC_only / Evo2 do worse.
# --------------------------------------------------------------------------
def representative_loci(pred, gt, chrom_bins, k=10):
    rows = []
    cum = np.cumsum([0] + [chrom_bins[ch] for ch in CHROMS])

    for ti, tname in enumerate(TRACKS):
        for A, B in permutations(CELLS, 2):
            dGT     = gt[A][ti]      - gt[B][ti]
            dE2H    = pred['Evo2HiC' ][A][ti] - pred['Evo2HiC' ][B][ti]
            dHiC    = pred['HiC_only'][A][ti] - pred['HiC_only'][B][ti]
            dEvo2   = pred['Evo2'    ][A][ti] - pred['Evo2'    ][B][ti]

            # diagnostic score: GT delta * matching Evo2HiC delta, penalised by baselines being right
            score = dGT * dE2H \
                  - (np.abs(dGT * dHiC) + np.abs(dGT * dEvo2)) / 2
            score = np.where(dGT > 0, score, -np.inf)              # require A > B in GT

            top_idx = np.argsort(score)[-k:][::-1]
            for rank, b in enumerate(top_idx):
                # locate (chrom, start) from concatenated bin index
                ch_i  = int(np.searchsorted(cum, b, side='right') - 1)
                ch    = CHROMS[ch_i]
                start = (b - cum[ch_i]) * RESOLUTION
                rows.append({
                    'track': tname, 'pair': f'{A}_vs_{B}', 'rank': rank,
                    'chrom': f'chr{ch}', 'start': start, 'end': start + RESOLUTION,
                    'dGT':       float(dGT[b]),
                    'dEvo2HiC':  float(dE2H[b]),
                    'dHiC_only': float(dHiC[b]),
                    'dEvo2':     float(dEvo2[b]),
                    'score':     float(score[b]),
                })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
def main():
    print('[1/4] Loading predictions and ground truth ...')
    pred, gt, chrom_bins = load_all()

    print('[2/4] Cross-cell PCC matrix ...')
    cm  = cross_cell_matrix(pred, gt)
    gap = specificity_gap(cm)
    cm.to_csv (os.path.join(OUT_DIR, 'crosscell_matrix.tsv'), sep='\t', index=False)
    gap.to_csv(os.path.join(OUT_DIR, 'spec_gap.tsv'),         sep='\t', index=False)
    print(gap.pivot(index='track', columns='model', values='gap').round(3))

    print('[3/4] Δ-signal SpecPCC + Williams ...')
    deltas, cum = delta_arrays(pred, gt, chrom_bins)
    sp = spec_pcc_table(deltas, cum)
    sp.to_csv(os.path.join(OUT_DIR, 'spec_pcc.tsv'), sep='\t', index=False)
    print(sp.groupby('track')[[c for c in sp.columns if c.startswith('specpcc_')]].mean().round(3))

    wt = williams_table(deltas, cum)
    wt.to_csv(os.path.join(OUT_DIR, 'williams_pvals.tsv'), sep='\t', index=False)
    sig = wt.assign(sig=wt.p_one_sided < 0.05) \
            .groupby(['baseline'])['sig'].mean().round(3)
    print('Fraction of (pair, track) where Evo2HiC > baseline (p<0.05):')
    print(sig)

    print('[4/4] Representative loci ...')
    loci = representative_loci(pred, gt, chrom_bins, k=10)
    loci.to_csv(os.path.join(OUT_DIR, 'representative_loci.tsv'), sep='\t', index=False)

    # also dump as bed for IGV
    bed = loci[['chrom', 'start', 'end', 'track', 'pair', 'score']].copy()
    bed['name'] = bed.track + '|' + bed.pair + '|' + bed.score.round(3).astype(str)
    bed[['chrom', 'start', 'end', 'name']].to_csv(
        os.path.join(OUT_DIR, 'representative_loci.bed'),
        sep='\t', index=False, header=False
    )

    print(f'\nAll outputs written to {OUT_DIR}/')


if __name__ == '__main__':
    main()
