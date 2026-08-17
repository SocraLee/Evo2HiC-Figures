"""Cross-species TAD eval using cooltools (matches Fig 3 / paper convention).

Replaces evaluate.IS_eval-based multi_TAD_eval.py whose default window/strength
parameters (window=500 kb at 10 kb resolution, bound_strength=0.1 on a custom
delta-of-log2-IS scale) were tuned for 1 Mb local matrices used in seq2hic and
were inappropriate for full-chromosome 10 kb scale-up.

Pipeline matches plot_Fig3_Seq2HiC.ipynb cell 26-27:
  - cooltools.calculate_insulation_score(window_bp=100_000, ignore_diags=2, balance=False)
  - cooltools.find_boundaries (peak prominence)
  - boundary if prominence (boundary_strength) >= 0.2
  - F1 with boundary_zone_size = 3 bin (= 30 kb at 10 kb)
"""
import os
import sys
import time
import argparse
import tempfile
import numpy as np
import pandas as pd
import hicstraw
import cooler
import logging
from cooltools.api.insulation import calculate_insulation_score, find_boundaries
from scipy.stats import pearsonr

from config import dnazoo_index, dnazoo_hic_dir
from evaluate.eval_utils import read_hic

logging.getLogger().setLevel(logging.ERROR)


METHODS = {
    'evo2hic':  '/m-chimera/chimera/nobackup/yongkang/HiC_ckpt/checkpoints/04_27_05_45_CDNAUNET_2000/170000/multi_10000',
    'hic_only': '/m-chimera/chimera/nobackup/yongkang/HiC_ckpt/checkpoints/resolution_enhancement_hic_only/98000/multi_10000',
}

WINDOW_BP = 100_000
PROMINENCE_CUT = 0.2
BOUNDARY_ZONE = 3      # bin (= 30 kb at 10 kb resolution)


def write_cool_from_matrix(M, res, chrom_name, cool_path):
    """Write an upper-triangular matrix to a minimal .cool file."""
    N = M.shape[0]
    bins = pd.DataFrame({
        'chrom': chrom_name,
        'start': np.arange(0, N * res, res, dtype=np.int64),
        'end':   np.arange(res, (N + 1) * res, res, dtype=np.int64),
    })
    i, j = np.triu_indices(N, k=0)
    counts = M[i, j].astype(float)
    mask = np.isfinite(counts) & (counts != 0)
    pixels = pd.DataFrame({
        'bin1_id': i[mask].astype(np.int64),
        'bin2_id': j[mask].astype(np.int64),
        'count':   counts[mask],
    })
    cooler.create_cooler(cool_path, bins=bins, pixels=pixels,
                         dtypes={'count': 'float64'}, ordered=True)


def insulation_pipeline(M, res, tmp_dir, chrom_name='chr1',
                        window_bp=WINDOW_BP, prominence_cut=PROMINENCE_CUT):
    cool_path = os.path.join(tmp_dir, f'{chrom_name}.cool')
    write_cool_from_matrix(M, res, chrom_name, cool_path)
    clr = cooler.Cooler(cool_path)
    is_table = calculate_insulation_score(
        clr, window_bp=[window_bp], ignore_diags=2, clr_weight_name=None,
    )
    is_table = find_boundaries(is_table)
    wtag = str(window_bp)
    log2_col = next(c for c in is_table.columns
                    if c.startswith('log2_insulation_score') and c.endswith(wtag))
    bs_col = next(c for c in is_table.columns
                  if c.startswith('boundary_strength') and c.endswith(wtag))
    scores = is_table[log2_col].to_numpy().astype(float)
    bs = is_table[bs_col].to_numpy().astype(float)
    bounds = np.where(np.isfinite(bs) & (bs >= prominence_cut))[0].tolist()
    return scores, bounds


def boundary_f1(prd, tgt, zone=BOUNDARY_ZONE):
    if len(tgt) == 0:
        return np.nan
    if len(prd) == 0:
        return 0.0
    pm = np.zeros(len(prd))
    tm = np.zeros(len(tgt))
    for i, p in enumerate(prd):
        for j, t in enumerate(tgt):
            if abs(p - t) <= zone:
                pm[i] = 1
                tm[j] = 1
    pr, rc = pm.mean(), tm.mean()
    if pr == 0 or rc == 0:
        return 0.0
    return 2 / (1 / pr + 1 / rc)


def load_matrix(path, resolution):
    h = hicstraw.HiCFile(path)
    for norm in ('SCALE', 'NONE'):
        try:
            return read_hic(h, resolution, format='matrix', chrid=1, norm=norm), norm
        except Exception:
            continue
    raise RuntimeError(f'all norms failed for {path}')


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--output', default='result/multi/TAD_revision_cooltools.tsv')
    p.add_argument('--resolution', type=int, default=10000)
    p.add_argument('--limit', type=int, default=None)
    p.add_argument('--restart', action='store_true')
    return p.parse_args()


def main():
    args = parse_args()

    with open(dnazoo_index) as f:
        species_list = [line.strip() for line in f if line.strip()]
    if args.limit:
        species_list = species_list[:args.limit]

    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)

    if os.path.isfile(args.output) and not args.restart:
        done = pd.read_csv(args.output, sep='\t')
        done_set = set(done['species'].tolist())
        rows = done.to_dict('records')
        print(f'[resume] {len(done_set)} species done', flush=True)
    else:
        done_set = set()
        rows = []

    cols = (['species']
            + [f'TAD_f1_{m}' for m in METHODS]
            + [f'IS_pcc_{m}' for m in METHODS]
            + [f'n_bounds_{m}' for m in METHODS]
            + ['n_bounds_gt', 'n_bins', 'norm_used'])

    for i, species in enumerate(species_list):
        if species in done_set:
            continue

        gt_path = os.path.join(dnazoo_hic_dir, species + '.hic')
        if not os.path.isfile(gt_path):
            print(f'[skip {i+1}/{len(species_list)}] {species}: GT missing', flush=True)
            continue
        method_paths = {m: os.path.join(d, species + '_enhanced.hic')
                        for m, d in METHODS.items()}
        missing = [m for m, p in method_paths.items() if not os.path.isfile(p)]
        if missing:
            print(f'[skip {i+1}/{len(species_list)}] {species}: missing {missing}', flush=True)
            continue

        t0 = time.time()
        try:
            mat_gt, norm_used = load_matrix(gt_path, args.resolution)
        except Exception as e:
            print(f'[fail {i+1}/{len(species_list)}] {species}: GT load {e}', flush=True)
            continue

        try:
            with tempfile.TemporaryDirectory() as td:
                s_gt, b_gt = insulation_pipeline(mat_gt, args.resolution, td, 'chr_gt')

                row = {
                    'species': species, 'n_bins': mat_gt.shape[0],
                    'norm_used': norm_used, 'n_bounds_gt': len(b_gt),
                }

                fail = False
                method_scores = {}
                for m, mp in method_paths.items():
                    try:
                        mat_pred, _ = load_matrix(mp, args.resolution)
                    except Exception as e:
                        print(f'[fail] {species}/{m}: load {e}', flush=True)
                        fail = True
                        break
                    if mat_pred.shape != mat_gt.shape:
                        print(f'[fail] {species}/{m}: shape {mat_pred.shape}', flush=True)
                        fail = True
                        break
                    s_pred, b_pred = insulation_pipeline(
                        mat_pred, args.resolution, td, f'chr_{m}'
                    )
                    method_scores[m] = (s_pred, b_pred)
                    row[f'TAD_f1_{m}'] = float(boundary_f1(b_pred, b_gt))
                    row[f'n_bounds_{m}'] = len(b_pred)
                    del mat_pred

                if fail:
                    del mat_gt
                    continue

                # IS PCC: align finite mask across all
                masks = [np.isfinite(s_gt)]
                for m in METHODS:
                    masks.append(np.isfinite(method_scores[m][0]))
                final_mask = np.all(masks, axis=0)
                if final_mask.sum() < 50:
                    print(f'[skip {i+1}] {species}: <50 finite IS bins', flush=True)
                    del mat_gt
                    continue

                for m in METHODS:
                    s_pred = method_scores[m][0]
                    pcc, _ = pearsonr(s_pred[final_mask], s_gt[final_mask])
                    row[f'IS_pcc_{m}'] = float(pcc)
        except Exception as e:
            print(f'[fail {i+1}/{len(species_list)}] {species}: pipeline {type(e).__name__} {e}', flush=True)
            del mat_gt
            continue

        del mat_gt
        rows.append(row)
        df = pd.DataFrame(rows, columns=cols)
        df.to_csv(args.output, sep='\t', index=False)

        dt = time.time() - t0
        # Pretty progress line: report TAD F1 / IS PCC for whichever methods
        # are configured (allows METHODS override e.g. for HICARN2 baseline).
        method_keys = list(METHODS.keys())
        f1s = ' '.join(f'{m}={row[f"TAD_f1_{m}"]:.3f}' for m in method_keys)
        iss = ' '.join(f'{m}={row[f"IS_pcc_{m}"]:.3f}' for m in method_keys)
        print(f'[{i+1}/{len(species_list)}] {species:55s} '
              f'F1: {f1s}  IS: {iss}  ({dt:.1f}s)', flush=True)

    print(f'\n[done] {len(rows)} species → {args.output}', flush=True)


if __name__ == '__main__':
    main()
