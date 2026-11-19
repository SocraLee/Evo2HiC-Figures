#!/usr/bin/env python
"""Genome-wide ABC vs CRISPRi benchmark on a held-out cell line, protocol grid.

For each (gene, tested element) pair of the ENCODE-rE2G K562 benchmark we score:
  distance    1 / |element - TSS|
  contact     C(tss, enh)                     (normalised Hi-C, grid resolution)
  ABC         A(enh) * C / sum_window(A * C)  (Fulco et al. 2019)
with A either sqrt(DNase * H3K27ac) (standard) or DNase alone (published variant),
from predicted tracks (Evo2HiC, held-out cell) or measured tracks (upper bound).

Grid axes (applied identically to predicted and measured activity):
  --res        contact resolution (2000 / 5000 / 10000)
  --pseudo     add Fulco-style power-law pseudocount expected(min(d, 1 Mb)) to C
  activity     both variants always computed

The ABC denominator integrates A*C over all non-promoter-proximal bins of the
+-1 Mb window (continuous landscape, as in round 1), so it does not depend on
which elements happened to be tested.

Output: <out>/pairs_res{RES}_pseudo{0|1}.tsv with per-pair scores.
Run from the Evo2HiC repo root (needs config/dataset imports).
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))  # repo root
import paths as P  # noqa: E402

sys.path.insert(0, str(P.REPO))
from config import hic2tarck_dir, hic_data_dir  # noqa: E402
from dataset.track_loader import Track_Loader  # noqa: E402
import hicstraw  # noqa: E402

TRACK_RES = 2000
DNASE, CTCF, H3K27AC, H3K27ME3, H3K4ME3 = range(5)
NBR_BP = 1_000_000
PROM_EXCL_BINS = 3          # exclude promoter-proximal bins (in track-res units)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--cell', default='K562')
    p.add_argument('--hic', default='4DNFITUOMFUQ')
    p.add_argument('--bench', default=str(P.RA_CRISPR_BENCH))
    p.add_argument('--pred-dir', default=str(P.EPI_EVO2HIC_DIR))
    p.add_argument('--res', type=int, default=5000)
    p.add_argument('--pseudo', type=int, default=1)
    p.add_argument('--norm', default='SCALE')
    p.add_argument('--calib', type=int, default=0,
                   help='quantile-map predicted tracks to the GM12878 measured marginal '
                        '(training-cell information only)')
    p.add_argument('--calib-ref', default='GM12878',
                   help='comma-separated training cell lines whose measured marginals are pooled as the QN reference')
    p.add_argument('--out', default=os.path.join(P.RA_WORK('abc'), 'out'))
    return p.parse_args()


def main():
    a = parse_args()
    os.makedirs(a.out, exist_ok=True)
    res = a.res
    nbr = NBR_BP // res                       # window radius in contact bins
    assert res % TRACK_RES == 0, 'contact res must be a multiple of the 2 kb track res'
    fold = res // TRACK_RES                   # track bins per contact bin

    b = pd.read_csv(a.bench, sep='\t')
    b = b[b['ValidConnection'].astype(str).str.upper().isin(['TRUE', '1'])].copy()
    b['reg'] = b['Regulated'].astype(str).str.upper().isin(['TRUE', '1'])
    b['tss'] = (b.startTSS + b.endTSS) // 2
    b['emid'] = (b.chromStart + b.chromEnd) // 2
    b['dist'] = (b.emid - b.tss).abs()
    b = b[(b.dist <= NBR_BP) & (b.dist >= PROM_EXCL_BINS * TRACK_RES)]
    b['chrn'] = b.chrom.str.replace('chr', '')
    b = b[~b.chrn.isin(['X', 'Y'])]

    calib_ref = None
    if a.calib:
        refs = [np.asarray(Track_Loader(os.path.join(hic2tarck_dir, c), TRACK_RES).values, dtype=np.float64)
                for c in a.calib_ref.split(',')]
        calib_ref = np.sort(np.concatenate(refs, axis=1), axis=1)  # (5, L) sorted pooled marginals

    hic = hicstraw.HiCFile(os.path.join(hic_data_dir, a.hic + '.hic'))
    hic_chr = {c.name.replace('chr', ''): c.name for c in hic.getChromosomes()}
    gt = Track_Loader(os.path.join(hic2tarck_dir, a.cell), TRACK_RES)

    # ---- pass 1: pull the contact row of every gene, accumulate distance decay
    rows = {}
    dec_sum = np.zeros(nbr + 1)
    dec_cnt = np.zeros(nbr + 1)
    genes = b.groupby(['chrn', 'measuredGeneSymbol', 'tss']).size().reset_index()[['chrn', 'measuredGeneSymbol', 'tss']]
    for ch, sub in genes.groupby('chrn'):
        mzd = hic.getMatrixZoomData(hic_chr[ch], hic_chr[ch], 'observed', a.norm, 'BP', res)
        for _, g in sub.iterrows():
            tb = int(g.tss // res)
            lo, hi = max(0, tb - nbr), tb + nbr
            m = mzd.getRecordsAsMatrix(tb * res, tb * res, lo * res, hi * res)
            row = np.nan_to_num(np.asarray(m, dtype=np.float64)).ravel()
            row = np.pad(row, (0, (hi - lo + 1) - len(row)))
            rows[(ch, g.tss)] = (lo, row)
            d = np.abs(np.arange(lo, hi + 1) - tb)
            np.add.at(dec_sum, d, row)
            np.add.at(dec_cnt, d, 1.0)

    decay = dec_sum / np.maximum(dec_cnt, 1)
    # power-law fit on 5 kb..1 Mb range for the pseudocount
    dd = np.arange(1, nbr + 1) * res
    ok = decay[1:] > 0
    coef = np.polyfit(np.log(dd[ok]), np.log(decay[1:][ok]), 1)
    def expected(dist_bp):
        return float(np.exp(coef[1] + coef[0] * np.log(np.clip(dist_bp, res, NBR_BP))))

    # ---- pass 2: score every pair
    recs = []
    for ch, sub in b.groupby('chrn'):
        pred = np.load(os.path.join(a.pred_dir, a.cell, f'{ch}.npy')).astype(np.float64)
        if calib_ref is not None:
            for i in range(pred.shape[0]):
                q = pd.Series(pred[i]).rank(pct=True).to_numpy()
                idx = np.clip((q * (calib_ref.shape[1] - 1)).astype(int), 0, calib_ref.shape[1] - 1)
                pred[i] = calib_ref[i, idx]
        key = f'chr{ch}' if f'chr{ch}' in gt.chr_pos else ch
        n_bins = gt.chr_pos[key][1] - gt.chr_pos[key][0]
        meas = gt.get(key, 0, n_bins * TRACK_RES, 0).astype(np.float64)
        n = min(pred.shape[1], meas.shape[1])

        def act(tr, mode):
            d = np.clip(tr[DNASE][:n], 0, None)
            if mode == 'dnase':
                return d
            return np.sqrt(d * np.clip(tr[H3K27AC][:n], 0, None))

        A = {('pred', m): act(pred, m) for m in ('std', 'dnase')}
        A.update({('meas', m): act(meas, m) for m in ('std', 'dnase')})

        for (gene, tss), pairs in sub.groupby(['measuredGeneSymbol', 'tss']):
            lo, row = rows[(ch, tss)]
            tb = int(tss // res)
            if a.pseudo:
                dgrid = np.abs(np.arange(lo, lo + len(row)) - tb) * res
                row = row + np.vectorize(expected)(np.minimum(dgrid, NBR_BP))
            # aggregate activity to contact bins over the window
            t0 = lo * fold
            t1 = (lo + len(row)) * fold
            sl = slice(max(t0, 0), min(t1, n))
            width = (len(row)) * fold
            tbin_t = int(tss // TRACK_RES)
            dist_t = np.abs(np.arange(t0, t0 + width) - tbin_t)
            prox = dist_t < PROM_EXCL_BINS
            denom = {}
            for k, v in A.items():
                buf = np.zeros(width)
                buf[sl.start - t0: sl.stop - t0] = v[sl]
                buf[prox] = 0.0
                cw = buf.reshape(-1, fold).mean(1)          # activity per contact bin
                denom[k] = float((cw * row).sum()) + 1e-12
            for _, pr in pairs.iterrows():
                eb_t = int(pr.emid // TRACK_RES)
                eb_c = int(pr.emid // res) - lo
                if not (0 <= eb_c < len(row)) or not (0 <= eb_t < n):
                    continue
                C = float(row[eb_c])
                rec = {'chr': ch, 'gene': gene, 'tss': tss, 'enh': int(pr.emid),
                       'dist': int(pr.dist), 'reg': bool(pr.reg),
                       'Reference': pr.get('Reference', ''),
                       's_dist': 1.0 / pr.dist, 'contact': C,
                       's_contact': C / (float(row.sum() - row[max(tb - lo - 1, 0):tb - lo + 2].sum()) + 1e-12)}
                for (src, m), v in A.items():
                    rec[f'abc_{src}_{m}'] = float(v[eb_t]) * C / denom[(src, m)]
                    rec[f'act_{src}_{m}'] = float(v[eb_t])
                recs.append(rec)

    df = pd.DataFrame(recs)
    out = os.path.join(a.out, f'{a.cell}_pairs_res{res}_pseudo{a.pseudo}.tsv')
    df.to_csv(out, sep='\t', index=False)
    print(f'[abc_grid] {out}  pairs={len(df)} pos={df.reg.sum()} genes={df.gene.nunique()} '
          f'decay_gamma={coef[0]:.2f}', flush=True)


if __name__ == '__main__':
    main()
