"""
AlphaGenome FOLD_1 baseline inference for the epig_predict (CDNA1d) comparison.

Loads `google/alphagenome-fold-1` from HuggingFace and tiles chr10 (and
optionally chr9) with 1 MB intervals, extracting the 14 (cell x assay) tracks
that overlap our Evo2HiC catalog. Output predictions are pooled to 2 kb bins
matching `inference/inference_CDNA1d.py`.

Why FOLD_1: its TRAIN folds = {fold0,1,2,5,6,7} are IDENTICAL to Borzoi
replicate-0's training set; AlphaGenome FOLD_1's strict TEST = fold4. fold4
covers chr10 (~284 fragments, ~7,131 of 2 kb bins on chr10) and 0 bp of chr9.
Evaluating AlphaGenome FOLD_1 only on fold4-strict bins of chr10 is its true
zero-shot eval set and the only set where AlphaGenome's published TEST
designation is honoured (per /memory/feedback_strict_test_fold).

Track coverage (14/15):
    GM12878: DNase, CTCF, H3K27ac, ----, H3K4me3
    H1ESC:   DNase, CTCF, H3K27ac, H3K27me3, H3K4me3
    K562:    DNase, CTCF, H3K27ac, H3K27me3, H3K4me3
GM12878 H3K27me3 is absent from AlphaGenome's output catalog (analogous to
Borzoi missing K562 H3K27ac); written as NaN.

Caveats:
  - Signal scale: AlphaGenome outputs softplus-scaled coverage; our GT is
    [0,1]-clipped BigWig mean per 2 kb bin. PCC is scale-invariant so
    correlations are comparable; do NOT compute MSE without per-track
    affine calibration.
"""
import os
# Force-set BEFORE importing alphagenome / tensorflow / jax.
# `setdefault` is insufficient because the user's shell may export a relative
# HF_HOME path that orbax cannot accept (it needs an absolute path).
os.environ['CURL_CA_BUNDLE']     = '/etc/pki/tls/certs/ca-bundle.crt'
os.environ['SSL_CERT_FILE']      = '/etc/pki/tls/certs/ca-bundle.crt'
os.environ['REQUESTS_CA_BUNDLE'] = '/etc/pki/tls/certs/ca-bundle.crt'
os.environ['HF_HOME']            = '/m-chimera/chimera/nobackup/yongkang/hf_cache'

import sys
import json
import time
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

# Avoid importing dataset.DNA_loader (pysam) — AlphaGenome fetches DNA itself
# from its bundled reference. We only need chromosome sizes.
from config import tracks as CDNA_TRACKS  # noqa: E402

# hg38 chromosome sizes (bp) for chrs of interest. Hardcoded to avoid pysam dep.
HG38_CHROM_SIZES = {
    9:  138_394_717,
    10: 133_797_422,
}

# AG input/output constants
AG_INPUT_LEN = 1_048_576   # 1 MB
DNASE_RES = 1
CHIP_RES  = 128
RESOLUTION = 2000          # our target 2kb bin grid

CELL_ALIASES = {
    'GM12878': ['GM12878'],
    'H1ESC':   ['H1'],
    'K562':    ['K562'],
}

# (cell, our_track_name) -> (output_type_attr, filter_col, filter_val)
TRACK_QUERY = [
    ('DNase',    'dnase',        None,                 None),
    ('CTCF',     'chip_tf',      'transcription_factor', 'CTCF'),
    ('H3K27ac',  'chip_histone', 'histone_mark',         'H3K27ac'),
    ('H3K27me3', 'chip_histone', 'histone_mark',         'H3K27me3'),
    ('H3K4me3',  'chip_histone', 'histone_mark',         'H3K4me3'),
]


def build_track_index(md, organism_md):
    """Returns {(cell, track) -> {'output_type': str, 'row_indices': [int]}}.
    Row indices index into organism_md.<output_type>.values' track axis."""
    idx_map = {}
    for cell, aliases in CELL_ALIASES.items():
        for our_tname, ot_attr, col, val in TRACK_QUERY:
            df = getattr(organism_md, ot_attr)
            mask = df.biosample_name.astype(str).isin(aliases)
            if col is not None:
                mask &= (df[col] == val)
            # Use positional index (iloc) since values is a positional array
            rows = np.where(mask.values)[0].tolist()
            idx_map[(cell, our_tname)] = {
                'output_type': ot_attr,
                'row_indices': rows,
                'names': df.loc[mask, 'name'].tolist() if rows else [],
            }
    return idx_map


def pool_to_2kb(values_native, native_res, interval_start_bp, resolution=RESOLUTION):
    """
    values_native: (L_native, n_tracks) at native_res bp per row.
    Native row i covers [interval_start_bp + i*native_res,
                          interval_start_bp + (i+1)*native_res).
    Returns (pooled, bmin, n_bins) where pooled has shape (n_tracks, n_bins)
    and pooled[:, k] corresponds to global 2 kb bin (bmin + k).
    """
    L, n_tracks = values_native.shape
    centers = interval_start_bp + (np.arange(L) + 0.5) * native_res
    target_bins = (centers // resolution).astype(np.int64)
    bmin = int(target_bins.min())
    bmax = int(target_bins.max())
    n_bins = bmax - bmin + 1
    out = np.zeros((n_tracks, n_bins), dtype=np.float32)
    cnt = np.zeros(n_bins, dtype=np.int64)
    rel = target_bins - bmin
    # np.add.at along the second axis
    for t in range(n_tracks):
        np.add.at(out[t], rel, values_native[:, t])
    np.add.at(cnt, rel, 1)
    out /= np.maximum(cnt, 1)[None, :]
    return out, bmin, n_bins


def predict_chrom(model, chrom, chrom_size, requested_outputs,
                  ot_track_indices, ot_track_arr_offsets, stride, sanity_check=0):
    """
    Tile chrom with 1 MB intervals at the given stride; run AG inference; pool to 2 kb.
    Returns (predictions, counts) where predictions has shape (n_aggregate_tracks, n_chrom_bins).

    ot_track_indices: dict {ot_attr -> list of row indices to keep}
    ot_track_arr_offsets: dict {ot_attr -> int} starting offset for that OT in the
                          aggregated track axis (so all selected tracks live in a single
                          flat array we can save).
    """
    from alphagenome.data import genome  # local import — heavy
    n_chrom_bins = int(np.ceil(chrom_size / RESOLUTION))
    n_agg_tracks = sum(len(v) for v in ot_track_indices.values())

    predictions = np.zeros((n_agg_tracks, n_chrom_bins), dtype=np.float32)
    counts = np.zeros(n_chrom_bins, dtype=np.int64)

    starts = list(range(0, chrom_size, stride))
    if sanity_check > 0:
        starts = starts[:sanity_check]

    from tqdm import tqdm
    for s_input in tqdm(starts, desc=f'chr{chrom}', file=sys.stdout):
        e_input = s_input + AG_INPUT_LEN
        # AlphaGenome accepts any interval; it handles edge padding internally.
        iv = genome.Interval(chromosome=f'chr{chrom}',
                             start=max(s_input, 0),
                             end=min(e_input, chrom_size))
        # If the interval is shorter than 1 MB at the chromosome boundary,
        # AlphaGenome's predict_interval will still run but the output length
        # is determined by interval length / native_res. We resize the interval
        # to a fixed 1 MB centred on s_input + 0.5 MB to keep shapes consistent.
        iv = iv.resize(AG_INPUT_LEN)

        out = model.predict_interval(
            iv,
            requested_outputs=requested_outputs,
            ontology_terms=None,
        )

        for ot_attr, native_res in [('dnase', DNASE_RES), ('chip_tf', CHIP_RES),
                                     ('chip_histone', CHIP_RES)]:
            row_idx = ot_track_indices.get(ot_attr, [])
            if not row_idx:
                continue
            td = getattr(out, ot_attr)
            vals = td.values[:, row_idx]                   # (L_native, len(row_idx))
            pooled, bmin, nb = pool_to_2kb(vals, native_res, iv.start)
            b_lo = max(bmin, 0); b_hi = min(bmin + nb, n_chrom_bins)
            if b_hi <= b_lo:
                continue
            src_lo = b_lo - bmin; src_hi = b_hi - bmin
            offset = ot_track_arr_offsets[ot_attr]
            predictions[offset:offset + len(row_idx), b_lo:b_hi] += pooled[:, src_lo:src_hi]
            if ot_attr == 'dnase':
                # only count once per window across overlapping bins
                counts[b_lo:b_hi] += 1

    # Counts only tracked once per window (using dnase OT as anchor — present in all windows).
    # All OTs share the same window coverage, so this count is valid for all.
    predictions /= np.maximum(counts, 1)[None, :]
    return predictions, counts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save-dir', type=str,
                        default=str(REPO_ROOT / 'baselines/epigenomic/results/alphagenome_fold1'))
    parser.add_argument('--chroms', type=int, nargs='+', default=[10],
                        help='Which chromosomes to predict on (default chr10 — '
                             'sufficient for the fold4 strict TEST evaluation).')
    parser.add_argument('--stride', type=int, default=524_288,
                        help='Sliding window stride (default 1 MB / 2 = 524288 → 50%% overlap).')
    parser.add_argument('--sanity-check', type=int, default=0,
                        help='If >0, run only first N windows per chrom.')
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    with open(save_dir / 'args.json', 'w') as f:
        json.dump(vars(args), f, indent=2)

    print('Loading AlphaGenome FOLD_1 from HuggingFace ...')
    t0 = time.time()
    from alphagenome.models import dna_model as ag_dna
    from alphagenome_research.model import dna_model as agr_dna
    from alphagenome.models import dna_output
    model = agr_dna.create_from_huggingface(ag_dna.ModelVersion.FOLD_1)
    print(f'  loaded in {time.time()-t0:.1f}s')

    organism_md = model.output_metadata(organism=ag_dna.Organism.HOMO_SAPIENS)
    idx_map = build_track_index(model.output_metadata, organism_md)

    print('\nTrack mapping (cell | our_track -> AG row indices):')
    for (c, t), info in idx_map.items():
        marker = ' [MISSING]' if not info['row_indices'] else ''
        print(f'  {c:8s} {t:10s} ({info["output_type"]:13s}) {info["row_indices"]}{marker}')
    with open(save_dir / 'track_map.json', 'w') as f:
        json.dump({f'{c}|{t}': v for (c, t), v in idx_map.items()}, f, indent=2)

    # Aggregate per-OT row index lists. The saved (5_tracks, n_bins) array will
    # follow CDNA_TRACKS order (DNase, CTCF, H3K27ac, H3K27me3, H3K4me3) per cell.
    OUR_TRACK_ORDER = list(CDNA_TRACKS.keys())   # ['DNase','CTCF','H3K27ac','H3K27me3','H3K4me3']

    # Build a per-cell offset map. For each cell, we want a (5, n_bins) array
    # where row i = OUR_TRACK_ORDER[i]. We aggregate ALL needed AG indices across
    # cells/tracks into a single flat AG inference run, then re-slice per cell.
    flat_ot_indices = {'dnase': [], 'chip_tf': [], 'chip_histone': []}
    # mapping for slicing: (cell, our_tname) -> (ot_attr, position_in_flat_array, n_replicates)
    slice_map = {}
    for cell in CELL_ALIASES:
        for tname in OUR_TRACK_ORDER:
            info = idx_map[(cell, tname)]
            ot = info['output_type']
            rows = info['row_indices']
            start = len(flat_ot_indices[ot])
            flat_ot_indices[ot].extend(rows)
            slice_map[(cell, tname)] = (ot, start, len(rows))

    # Offsets in the aggregated track axis: dnase first, then chip_tf, then chip_histone
    ot_track_arr_offsets = {}
    cur = 0
    for ot in ['dnase', 'chip_tf', 'chip_histone']:
        ot_track_arr_offsets[ot] = cur
        cur += len(flat_ot_indices[ot])
    print(f'\nAggregated track axis: {cur} tracks total')
    print(f'  offsets: {ot_track_arr_offsets}')
    print(f'  sizes:   {[len(v) for v in flat_ot_indices.values()]}')

    requested_outputs = [
        dna_output.OutputType.DNASE,
        dna_output.OutputType.CHIP_TF,
        dna_output.OutputType.CHIP_HISTONE,
    ]

    for ch in args.chroms:
        chrom_size = HG38_CHROM_SIZES[ch]
        print(f'\n=== chr{ch} ({chrom_size:,} bp, ~{int(np.ceil(chrom_size/RESOLUTION)):,} bins of 2 kb) ===')
        pred_agg, counts = predict_chrom(
            model, ch, chrom_size,
            requested_outputs=requested_outputs,
            ot_track_indices=flat_ot_indices,
            ot_track_arr_offsets=ot_track_arr_offsets,
            stride=args.stride,
            sanity_check=args.sanity_check,
        )

        # Re-slice per cell -> (5_tracks, n_bins)
        for cell in CELL_ALIASES:
            cell_dir = save_dir / cell
            cell_dir.mkdir(parents=True, exist_ok=True)
            cell_pred = np.zeros((len(OUR_TRACK_ORDER), pred_agg.shape[1]), dtype=np.float32)
            for ti, tname in enumerate(OUR_TRACK_ORDER):
                ot, start, nrep = slice_map[(cell, tname)]
                if nrep == 0:
                    cell_pred[ti] = np.nan
                    continue
                offset = ot_track_arr_offsets[ot]
                cell_pred[ti] = pred_agg[offset + start : offset + start + nrep].mean(axis=0)
            np.save(cell_dir / f'{ch}.npy', cell_pred)
            print(f'  saved {cell}/chr{ch}.npy shape={cell_pred.shape} covered_bins={int((counts>0).sum()):,}')

    print(f'\nDone. Results in {save_dir}')


if __name__ == '__main__':
    main()
