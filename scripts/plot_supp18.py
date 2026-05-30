"""Supplementary Figure 18: UMAP of raw Evo 2 embeddings on human chr 10,
coloured by GM12878 epigenomic signals.

Companion to Fig 2 h-j. Whereas Fig 2 h-j embeds the Evo2HiC encoder
output, this figure embeds the *raw* Evo 2 base-model features for the
same chr 10 / GM12878 tracks. The intent is to expose how much of the
epigenomic structure visible in Fig 2 h-j was already present in Evo 2,
versus how much was sharpened by Evo2HiC's Hi-C contrastive training.

Inputs
------
- /m-chimera/chimera/nobackup/yongkang/HiC_data/data/dna/human/hg38_2000_evo2_7b/
  chr10.embedding (and chr10.rev.embedding)
  np.memmap float16 (66 899, 4 096)
- /homes/gws/yongkang/HiC/HiC-DNA/misc/tracks_GM12878_10.npy
  (5, 66 899) per-bin DNase / CTCF / H3K27ac / H3K27me3 / H3K4me3 signal
  at 2 kb on chr 10 (same file used by plot_Fig2.ipynb).

Output
------
Figures/supplementary_18.pdf
"""
from __future__ import annotations
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import gaussian_kde, rankdata

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paths import (                                              # noqa: E402
    REPO, ensure_out_dir,
    EVO2_EMB_DIR, TRACKS_GM12878_CHR10_NPY as TRACKS_NPY,
)

OUT_PDF = ensure_out_dir() / "supplementary_18.pdf"

CHR10_LEN = 66899
EMB_DIM = 4096
TRACKS = ["DNase", "CTCF", "H3K27ac", "H3K27me3", "H3K4me3"]
# 2×3 grid: row 1 mirrors Fig 2 h–j, row 2 is the remaining tracks +
# an "Average" panel (mean rank-norm. signal across all 5 tracks).
PANEL_GRID = [
    ["H3K27me3", "H3K27ac", "DNase"],
    ["CTCF",      "H3K4me3", "Average"],
]
PANEL_LETTERS = [["a", "b", "c"], ["d", "e", "f"]]

plt.style.use("seaborn-v0_8-white")
plt.rcParams.update({"font.size": 8, "font.family": "Arial"})


def kde_mean_2d(x, y, val, bw_method=None):
    pts = np.vstack([x, y])
    kde_den = gaussian_kde(pts, bw_method=bw_method)
    kde_val = gaussian_kde(pts, weights=val, bw_method=bw_method)
    den = kde_den(pts)
    num = kde_val(pts)
    return np.divide(num, den, out=np.full_like(num, np.nan), where=den > 0)


def _panel_label(ax, label):
    ax.text(-0.05, 1.14, label, transform=ax.transAxes,
            fontsize=12, fontname="Arial", fontweight="bold",
            ha="left", va="top")


def main(seed: int = 42):
    print("[load] raw Evo 2 chr10 embedding (memmap)...")
    fwd = np.memmap(EVO2_EMB_DIR / "chr10.embedding", mode="r",
                    dtype=np.float16, shape=(CHR10_LEN, EMB_DIM))
    rev = np.memmap(EVO2_EMB_DIR / "chr10.rev.embedding", mode="r",
                    dtype=np.float16, shape=(CHR10_LEN, EMB_DIM))
    emb = np.concatenate(
        [np.asarray(fwd, dtype=np.float32),
         np.asarray(rev, dtype=np.float32)], axis=-1)
    norm = np.linalg.norm(emb, axis=-1, keepdims=True)
    emb = np.nan_to_num(emb / np.maximum(norm, 1e-12))

    print("[load] GM12878 tracks...")
    tracks = np.load(TRACKS_NPY)
    assert tracks.shape == (5, CHR10_LEN), f"unexpected tracks shape {tracks.shape}"
    pct = rankdata(tracks, axis=-1) / tracks.shape[-1]

    print("[umap] running UMAP n_components=2 (this is slow on CPU; ~20 min)...")
    try:
        import umap
    except ImportError:
        sys.exit("umap-learn not installed in the active env; "
                 "`pip install umap-learn` (already in Evo2HiC env).")
    reducer = umap.UMAP(n_components=2, random_state=seed,
                        n_neighbors=100, min_dist=0.5, metric="cosine")
    emb2d = reducer.fit_transform(emb)

    L, R = 1000, -1000
    cmap = "RdBu_r"

    # Per-bin "Average" signal = mean rank-normalised signal across the 5
    # epigenomic tracks. The other 5 panels use the per-track row.
    track_pct = {tn: pct[TRACKS.index(tn)] for tn in TRACKS}
    track_pct["Average"] = pct.mean(axis=0)

    fig, axes = plt.subplots(2, 3, figsize=(7.0, 5.0), constrained_layout=True)
    row_sc = [None, None]
    for row_idx, row_tracks in enumerate(PANEL_GRID):
        for col_idx, track_name in enumerate(row_tracks):
            ax = axes[row_idx][col_idx]
            x = emb2d[L:R, 0]
            y = emb2d[L:R, 1]
            val = track_pct[track_name][L:R]
            local_mean = kde_mean_2d(x, y, val, bw_method=0.05)

            sc = ax.scatter(x, y, c=local_mean, s=0.02, alpha=0.35,
                            cmap=cmap, rasterized=True)
            row_sc[row_idx] = sc
            sns.despine(ax=ax, bottom=True, left=True)
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(track_name)

            # UMAP 0 / UMAP 1 axis arrows in the lower-left corner
            # (matches plot/plot_Fig2.ipynb cell 48 `plot_umap` styling)
            xmin, xmax = ax.get_xlim()
            ymin, ymax = ax.get_ylim()
            frac = 0.3
            dx = (xmax - xmin) * frac
            dy = (ymax - ymin) * frac
            ax.plot([xmin, xmin + dx], [ymin, ymin], color="black", lw=0.8,
                    clip_on=False)
            ax.plot([xmin, xmin], [ymin, ymin + dy], color="black", lw=0.8,
                    clip_on=False)
            ax.annotate("", xy=(xmin + dx, ymin),
                        xytext=(xmin + dx * 0.6, ymin),
                        arrowprops=dict(arrowstyle="-|>", lw=0.8,
                                        color="black", shrinkA=0, shrinkB=0))
            ax.annotate("", xy=(xmin, ymin + dy),
                        xytext=(xmin, ymin + dy * 0.6),
                        arrowprops=dict(arrowstyle="-|>", lw=0.8,
                                        color="black", shrinkA=0, shrinkB=0))
            ax.text(xmin, ymin - 1, "UMAP 0", va="center", fontsize=6)
            ax.text(xmin - 1, ymin, "UMAP 1", ha="center", rotation=90,
                    fontsize=6)
            _panel_label(ax, PANEL_LETTERS[row_idx][col_idx])

    # One color bar per row, sharing the same RdBu_r cmap. Both bars are
    # identical mappings (rank-normalised [0, 1]) — having two compact bars
    # is more readable than one tall bar that stretches across both rows.
    for row_idx in range(2):
        cb = fig.colorbar(row_sc[row_idx], ax=axes[row_idx, :].tolist(),
                          location="right", fraction=0.04, pad=0.02)
        cb.outline.set_linewidth(0.1)
        cb.set_ticks([])
        cb.set_label("Epigenomic signal intensity (rank-norm.)", fontsize=7)

    fig.savefig(OUT_PDF, bbox_inches="tight", dpi=600)
    print(f"[wrote] {OUT_PDF}")


if __name__ == "__main__":
    main()
