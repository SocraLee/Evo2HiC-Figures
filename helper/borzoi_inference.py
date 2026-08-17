"""
Borzoi baseline inference for epig_predict (CDNA1d) comparison.

Uses the `borzoi-pytorch` HuggingFace-hosted port (johahi/borzoi-replicate-0) as a
zero-shot DNA-to-epigenome predictor for our 3 cell types x 5 tracks benchmark.

Usage:
    python -m baselines.epigenomic.borzoi_inference --sanity-check 3      # quick test
    python -m baselines.epigenomic.borzoi_inference                        # full chr9+chr10

Caveat: Borzoi's sequence-level train/valid/test split spans the whole genome,
so our test chromosomes (9, 10) overlap with Borzoi's training data. Borzoi's
numbers here likely represent an upper bound of its true generalization.
"""

import os
import sys
import json
import argparse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from config import DNA_map, tracks, splits
from dataset.DNA_loader import DNA_Loader

BORZOI_INPUT_LEN = 524_288
BORZOI_OUTPUT_LEN = 6144
BORZOI_BIN_BP = 32
BORZOI_CENTRAL_BP = BORZOI_OUTPUT_LEN * BORZOI_BIN_BP  # 196,608

TARGETS_URL = "https://raw.githubusercontent.com/calico/borzoi/main/examples/targets_human.txt"
TARGETS_CACHE = Path.home() / ".cache" / "borzoi" / "targets_human.txt"

TRACK_QUERY = {
    'DNase':    ['DNASE'],
    'CTCF':     ['CHIP:CTCF'],
    'H3K27ac':  ['CHIP:H3K27AC'],
    'H3K27me3': ['CHIP:H3K27ME3'],
    'H3K4me3':  ['CHIP:H3K4ME3'],
}

CELL_ALIASES = {
    'GM12878': ['GM12878'],
    'H1ESC':   ['H1-HESC', 'H1HESC', 'H1ESC', 'H1-ESC'],
    'K562':    ['K562'],
}


def load_borzoi(device):
    from borzoi_pytorch import Borzoi
    model = Borzoi.from_pretrained("johahi/borzoi-replicate-0")
    model.to(device).eval()
    return model


def load_targets():
    TARGETS_CACHE.parent.mkdir(parents=True, exist_ok=True)
    if not TARGETS_CACHE.exists():
        print(f"Fetching {TARGETS_URL} ...")
        urllib.request.urlretrieve(TARGETS_URL, TARGETS_CACHE)
    df = pd.read_csv(TARGETS_CACHE, sep='\t')
    return df


def build_track_index(targets_df):
    descriptions = targets_df['description'].astype(str).str.upper()
    idx_map = {}
    for cell, aliases in CELL_ALIASES.items():
        for track_name, queries in TRACK_QUERY.items():
            hits = np.zeros(len(descriptions), dtype=bool)
            for alias in aliases:
                alias_u = alias.upper()
                for q in queries:
                    q_u = q.upper()
                    hits |= descriptions.str.contains(f"{q_u}:{alias_u}", na=False, regex=False)
                    hits |= descriptions.str.contains(f"{alias_u}:{q_u}", na=False, regex=False)
                    hits |= (
                        descriptions.str.contains(q_u, na=False, regex=False)
                        & descriptions.str.contains(alias_u, na=False, regex=False)
                    )
            idx_map[(cell, track_name)] = sorted(set(np.where(hits)[0].tolist()))
    return idx_map


def _to_borzoi_shape(out):
    """Normalize Borzoi output to (B, n_tracks, n_out=6144)."""
    if isinstance(out, dict):
        out = out.get('human', next(iter(out.values())))
    assert out.dim() == 3, f"Unexpected Borzoi output dim {out.dim()}"
    if out.shape[-1] == BORZOI_OUTPUT_LEN:
        return out  # (B, n_tracks, 6144)
    if out.shape[1] == BORZOI_OUTPUT_LEN:
        return out.transpose(1, 2).contiguous()
    raise ValueError(f"Cannot find output-length={BORZOI_OUTPUT_LEN} in shape {tuple(out.shape)}")


def pool_32bp_to_resolution(pred_32, window_start_bp, resolution):
    """
    pred_32: (n_tracks, 6144). Central 196kb starts at `window_start_bp`.
    Returns (pooled, bin_min, n_bins) where pooled is (n_tracks, n_bins) at `resolution`bp,
    indexed so pooled[:, i] corresponds to global bin (bin_min + i).
    """
    n_tracks, n_out = pred_32.shape
    assert n_out == BORZOI_OUTPUT_LEN
    centers = window_start_bp + 16 + np.arange(n_out) * BORZOI_BIN_BP
    target_bins = centers // resolution
    bmin = int(target_bins[0])
    bmax = int(target_bins[-1])
    n_bins = bmax - bmin + 1
    out = np.zeros((n_tracks, n_bins), dtype=np.float32)
    cnt = np.zeros(n_bins, dtype=np.int64)
    rel = target_bins - bmin
    np.add.at(out.T, rel, pred_32.T)
    np.add.at(cnt, rel, 1)
    out /= np.maximum(cnt, 1)[None, :]
    return out, bmin, n_bins


def predict_chromosome(model, dna_loader, chrom, chrom_size, resolution, device, sanity_check=0):
    n_chrom_bins = int(np.ceil(chrom_size / resolution))
    pad = (BORZOI_INPUT_LEN - BORZOI_CENTRAL_BP) // 2  # 163,840

    # Infer n_tracks from one forward
    with torch.no_grad():
        dummy = torch.zeros(1, 4, BORZOI_INPUT_LEN, device=device)
        out = _to_borzoi_shape(model(dummy))
        n_tracks = out.shape[1]

    predictions = np.zeros((n_tracks, n_chrom_bins), dtype=np.float32)
    counts = np.zeros(n_chrom_bins, dtype=np.int64)

    starts = list(range(0, chrom_size, BORZOI_CENTRAL_BP))
    if sanity_check > 0:
        starts = starts[:sanity_check]

    pbar = tqdm(starts, desc=f"chr{chrom}", disable=False)
    for s_central in pbar:
        s_input = s_central - pad
        e_input = s_input + BORZOI_INPUT_LEN  # = s_central + BORZOI_CENTRAL_BP + pad

        # Fetch DNA; DNA_loader pads negatives/overshoot with zeros (N).
        dna = dna_loader.get(chrom, s_input, e_input, 0)  # (L, 4)
        assert dna.shape == (BORZOI_INPUT_LEN, 4), f"{dna.shape}"

        seq = torch.from_numpy(dna).float().permute(1, 0).unsqueeze(0).to(device)  # (1, 4, L)
        with torch.no_grad():
            out = _to_borzoi_shape(model(seq))[0].cpu().numpy()  # (n_tracks, 6144)

        pooled, bmin, nb = pool_32bp_to_resolution(out, s_central, resolution)
        b_lo = max(bmin, 0)
        b_hi = min(bmin + nb, n_chrom_bins)
        if b_hi <= b_lo:
            continue
        src_lo = b_lo - bmin
        src_hi = b_hi - bmin
        predictions[:, b_lo:b_hi] += pooled[:, src_lo:src_hi]
        counts[b_lo:b_hi] += 1

    predictions /= np.maximum(counts, 1)[None, :]
    return predictions, counts


def sanity_plot(pred_by_cell, track_loader, chrom, save_path):
    """Overlay Borzoi prediction vs BigWig ground truth for a short region, per cell."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    n_cells = len(pred_by_cell)
    n_tracks = len(tracks)
    fig, axes = plt.subplots(n_cells, n_tracks, figsize=(4 * n_tracks, 2 * n_cells), squeeze=False)

    # Show first ~1.5 Mb that has been predicted (where counts > 0)
    for i, (cell, pred) in enumerate(pred_by_cell.items()):
        # nansum so all-NaN tracks (e.g. K562 H3K27ac) don't hide coverage from other tracks
        covered = np.where(np.nansum(pred, axis=0) > 0)[0]
        if len(covered) == 0:
            continue
        lo, hi = covered[0], min(covered[0] + 750, covered[-1] + 1)  # 750 bins × 2kb = 1.5Mb
        gt = track_loader[cell].get(chrom, lo * 2000, hi * 2000, 0)  # (n_tracks, hi-lo)

        for j, tname in enumerate(tracks.keys()):
            ax = axes[i][j]
            ax.plot(gt[j], label='GT', color='black', lw=0.6)
            ax2 = ax.twinx()
            pred_slice = pred[j, lo:hi]
            if np.all(np.isnan(pred_slice)):
                ax.text(0.5, 0.5, 'no Borzoi\ntrack', transform=ax.transAxes,
                        ha='center', va='center', fontsize=8, color='tab:red', alpha=0.7)
            else:
                ax2.plot(pred_slice, label='Borzoi', color='tab:red', lw=0.6, alpha=0.8)
            ax.set_title(f"{cell} / {tname}", fontsize=9)
            ax.tick_params(axis='both', labelsize=6)
            ax2.tick_params(axis='both', labelsize=6)
    fig.suptitle(f"chr{chrom} sanity check")
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close(fig)
    print(f"Sanity plot: {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save-dir', type=str,
                        default=str(REPO_ROOT / 'baselines/epigenomic/results/borzoi'))
    parser.add_argument('--split', type=str, default='test', choices=['train', 'valid', 'test'])
    parser.add_argument('--resolution', type=int, default=2000)
    parser.add_argument('--sanity-check', type=int, default=0,
                        help='If >0, only run first N windows per chr and produce overlay plots.')
    parser.add_argument('--species', type=str, default='human')
    args = parser.parse_args()

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    with open(save_dir / 'args.json', 'w') as f:
        json.dump(vars(args), f, indent=2)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    print("Loading Borzoi model...")
    model = load_borzoi(device)

    print("Loading Borzoi targets...")
    targets_df = load_targets()
    track_idx_map = build_track_index(targets_df)

    print("Track mapping:")
    for (cell, t), idx in track_idx_map.items():
        print(f"  {cell:8s} {t:10s} -> {len(idx):3d} Borzoi tracks: {idx[:6]}{' ...' if len(idx) > 6 else ''}")

    with open(save_dir / 'track_map.json', 'w') as f:
        json.dump({f"{c}|{t}": idx for (c, t), idx in track_idx_map.items()}, f, indent=2)

    # Assert we have at least one match for each (cell, track)
    missing = [k for k, v in track_idx_map.items() if len(v) == 0]
    if missing:
        print(f"[WARN] Missing Borzoi tracks for: {missing}")
        print("       These will be written as zeros and PCC will be NaN.")

    dna_loader = DNA_Loader(DNA_map[args.species], 'Yes')

    # Optionally load Track_Loaders for sanity overlay plot
    track_loaders = None
    if args.sanity_check > 0:
        from dataset.track_loader import Track_Loader
        from config import hic2tarck_dir
        track_loaders = {}
        for cell in CELL_ALIASES:
            track_dir = os.path.join(hic2tarck_dir, cell)
            if os.path.isdir(track_dir):
                track_loaders[cell] = Track_Loader(track_dir, resolution=args.resolution)
            else:
                print(f"[WARN] Track dir missing for {cell}: {track_dir}")

    for ch in splits[args.species][args.split]:
        chrom_size = dna_loader.get_size(ch)
        print(f"\n=== chr{ch} ({chrom_size:,} bp, ~{int(np.ceil(chrom_size/args.resolution))} bins) ===")
        pred_all, counts = predict_chromosome(
            model, dna_loader, ch, chrom_size,
            resolution=args.resolution, device=device,
            sanity_check=args.sanity_check,
        )

        pred_by_cell = {}
        for cell in CELL_ALIASES:
            cell_dir = save_dir / cell
            cell_dir.mkdir(parents=True, exist_ok=True)
            cell_pred = np.zeros((len(tracks), pred_all.shape[1]), dtype=np.float32)
            for ti, tname in enumerate(tracks):
                idx = track_idx_map[(cell, tname)]
                if len(idx) == 0:
                    cell_pred[ti] = np.nan
                    continue
                cell_pred[ti] = pred_all[idx].mean(axis=0)
            np.save(cell_dir / f'{ch}.npy', cell_pred)
            print(f"  saved {cell}/chr{ch}.npy  shape={cell_pred.shape}  covered_bins={int((counts>0).sum())}")
            pred_by_cell[cell] = cell_pred

        if args.sanity_check > 0 and track_loaders:
            sanity_plot(pred_by_cell, track_loaders, ch,
                        save_dir / f'sanity_chr{ch}.png')

    print(f"\nDone. Results in {save_dir}")


if __name__ == '__main__':
    main()
