"""Supplementary Figure 19: Per-cell × per-track MSE for the Fig 4
epigenomic-prediction ablation (three-method comparison).

Bar layout mirrors [plot/plot_supp7.ipynb](plot_supp7.ipynb) — grouped
bars per track with Evo2HiC / Evo 2 / HiC-only.

Significance brackets are **not** drawn on the figure (they tended to
cross over the comparator bars when the comparator MSE was much higher
than Evo2HiC's, which looked messy). The stats themselves still follow
the same "only test ours vs best baseline" convention used by Fig 4 a/b/c
and documented in `figure_caption_supplements.md` §Fig 4, and are
persisted alongside the figure in
`Figures/statistics{,_table}.md` (Supp 19 section).

Method paths (consistent with plot/plot_Fig4_Epi.ipynb naming and colour
ordering — Evo2HiC / Evo 2 / HiC-only):

    Evo2HiC  preds : CKPT/epi_prediction/model/track/{cell}/{ch}.npy
    Evo 2    preds : CKPT/epi_baseline_evo2_{cell}/{step}/track/{cell}/{ch}.npy
                     step = 44000 (GM12878) / 80000 (H1ESC) / 70000 (K562)
    HiC-only preds : CKPT/epi_baseline_hic_only/60000/track/{cell}/{ch}.npy
    GT             : hic2tarck_dir/{cell}/ via Track_Loader (chr 9 + 10).

Outputs
-------
Figures/supplementary_19.pdf
Figures/supplementary_19_stats.tsv
"""
from __future__ import annotations
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paths import (                                              # noqa: E402
    REPO, ensure_out_dir, add_repo_to_syspath,
    EPI_EVO2HIC_DIR, EPI_EVO2_DIR, EPI_HIC_ONLY_DIR,
)
add_repo_to_syspath()

from plot_settings import colors as candidate_colors  # noqa: E402
from plot_utils import _p_to_stars  # noqa: E402

OUT_PDF = ensure_out_dir() / "supplementary_19.pdf"
OUT_TSV = ensure_out_dir() / "supplementary_19_stats.tsv"


NPY_DIRS_PARAMETRIC = {
    "Evo2HiC":  lambda cell: EPI_EVO2HIC_DIR / cell,
    "Evo 2":    lambda cell: EPI_EVO2_DIR(cell),
    "HiC-only": lambda cell: EPI_HIC_ONLY_DIR / cell,
}

CELLS = ["GM12878", "H1ESC", "K562"]
TRACKS = ["DNase", "CTCF", "H3K27ac", "H3K27me3", "H3K4me3"]
METHODS = ["Evo2HiC", "Evo 2", "HiC-only"]

plt.style.use("seaborn-v0_8-white")
plt.rcParams.update({"font.size": 8, "font.family": "Arial"})


def _load_gt_concat(cell, track_idx):
    from dataset.track_loader import Track_Loader  # noqa
    from config import hic2tarck_dir              # noqa
    tl = Track_Loader(f"{hic2tarck_dir}/{cell}", 2000)
    parts = []
    for ch in (9, 10):
        size = tl.chr_lens[f"chr{ch}"]
        parts.append(tl.get(ch, 0, (size // 2000 + 1) * 2000, 0)[track_idx])
    return np.concatenate(parts)


def _load_pred_concat(method, cell, track_idx):
    base = NPY_DIRS_PARAMETRIC[method](cell)
    parts = []
    for ch in (9, 10):
        f = base / f"{ch}.npy"
        if not f.exists():
            return None
        parts.append(np.load(f)[track_idx])
    return np.concatenate(parts)


def main():
    # ---- Compute per-(cell, track, method) MSE and bin-level err² arrays ----
    mse = {}             # (cell, track, method) -> scalar MSE
    err2 = {}            # (cell, track, method) -> 1-D array of (pred-GT)²
    for cell in CELLS:
        for ti, tname in enumerate(TRACKS):
            gt = _load_gt_concat(cell, ti).astype(np.float64)
            for m in METHODS:
                pred = _load_pred_concat(m, cell, ti)
                if pred is None:
                    continue
                pred = pred.astype(np.float64)
                L = min(pred.shape[0], gt.shape[0])
                e = (pred[:L] - gt[:L]) ** 2
                err2[(cell, tname, m)] = e
                mse[(cell, tname, m)] = float(np.nanmean(e))

    # results = {cell: {method: [MSE_track0, MSE_track1, ...]}}
    all_results = {
        cell: {m: [mse.get((cell, t, m), np.nan) for t in TRACKS]
               for m in METHODS}
        for cell in CELLS
    }

    # ---- For each (cell, track): pick best baseline + run Wilcoxon ----
    # "Best baseline" = argmin_b(mean MSE) over {Evo 2, HiC-only}.
    # Significance is only drawn when Evo2HiC delivers a strict positive
    # improvement (mean MSE < best baseline mean MSE).
    statistic_tests = {cell: [] for cell in CELLS}
    stat_rows = []
    for cell in CELLS:
        for tname in TRACKS:
            a = err2.get((cell, tname, "Evo2HiC"))
            if a is None:
                statistic_tests[cell].append(("Evo2HiC", None, None))
                stat_rows.append({"cell": cell, "track": tname,
                                  "best_baseline": None, "test": None,
                                  "stat": np.nan, "p": np.nan, "n": 0,
                                  "stars": "n.s.",
                                  "note": "Evo2HiC tensor missing"})
                continue
            ms_e2hc = float(np.nanmean(a))
            # find best baseline
            best_name, best_mse, best_err2 = None, np.inf, None
            for comp in ("Evo 2", "HiC-only"):
                b = err2.get((cell, tname, comp))
                if b is None:
                    continue
                cur = float(np.nanmean(b))
                if cur < best_mse:
                    best_mse = cur; best_name = comp; best_err2 = b
            if best_name is None:
                statistic_tests[cell].append(("Evo2HiC", None, None))
                stat_rows.append({"cell": cell, "track": tname,
                                  "best_baseline": None, "test": None,
                                  "stat": np.nan, "p": np.nan, "n": 0,
                                  "stars": "n.s.",
                                  "note": "no baseline available"})
                continue
            # directional gate — Evo2HiC must improve over best baseline
            if ms_e2hc >= best_mse:
                statistic_tests[cell].append(("Evo2HiC", best_name, None))
                stat_rows.append({"cell": cell, "track": tname,
                                  "best_baseline": best_name,
                                  "test": "directional gate (no positive improvement)",
                                  "stat": np.nan, "p": np.nan,
                                  "n": int(a.size), "stars": "n.s.",
                                  "note": "Evo2HiC mean MSE >= best baseline mean MSE"})
                continue
            # paired Wilcoxon on bin-level err²
            L = min(len(a), len(best_err2))
            m_ok = np.isfinite(a[:L]) & np.isfinite(best_err2[:L])
            x, y = a[:L][m_ok], best_err2[:L][m_ok]
            if x.size == 0 or np.all(x == y):
                statistic_tests[cell].append(("Evo2HiC", best_name, None))
                stat_rows.append({"cell": cell, "track": tname,
                                  "best_baseline": best_name,
                                  "test": "wilcoxon paired (alt=less)",
                                  "stat": np.nan, "p": np.nan,
                                  "n": int(x.size), "stars": "n.s.",
                                  "note": "all equal / empty"})
                continue
            stat, p = wilcoxon(x, y, alternative="less", zero_method="wilcox")
            statistic_tests[cell].append(("Evo2HiC", best_name, float(p)))
            stat_rows.append({"cell": cell, "track": tname,
                              "best_baseline": best_name,
                              "test": "wilcoxon paired (alt=less)",
                              "stat": float(stat), "p": float(p),
                              "n": int(x.size),
                              "stars": _p_to_stars(float(p)), "note": ""})
    pd.DataFrame(stat_rows).to_csv(OUT_TSV, sep="\t", index=False)
    print(f"[wrote] {OUT_TSV}")

    # ---- Plot — same layout as plot_supp7.ipynb cell 9 ----
    fig, axes = plt.subplots(1, 3, figsize=(8.27, 3.0),
                             constrained_layout=False)
    fig.subplots_adjust(top=0.82, bottom=0.20, left=0.07, right=0.99,
                        wspace=0.30)
    n_m = len(METHODS)
    width = 0.8 / n_m

    for ax, (cell, results) in zip(axes, all_results.items()):
        ax.clear()
        ax.set_aspect("auto")
        x = np.arange(len(TRACKS))
        for i, m in enumerate(METHODS):
            ax.bar(x + i * width, results[m], width=width,
                   label=m,
                   color=candidate_colors[i % len(candidate_colors)],
                   alpha=0.8)

        ax.set_ylabel("MSE")
        ax.set_title(cell)
        ax.tick_params(axis="x", which="both", length=0)
        ax.tick_params(axis="y", which="both", length=2)
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.set_xticks([])

        # ylim — no headroom needed because no sig-brackets are drawn on
        # this figure (they tend to cross the tall comparator bars and look
        # messy); test results are in Figures/statistics{,_table}.md instead.
        max_val = float(np.nanmax([np.asarray(results[m]) for m in METHODS]))
        ax.set_ylim(0.0, max_val * 1.05)

        # x-tick labels per group (matches plot_supp7 last lines)
        ax.set_xticks(x + width * (n_m - 1) / 2)
        ax.set_xticklabels(TRACKS, rotation=30, ha="right")

    # ---- Figure-level legend centred above all three panels ----
    legend_handles = [
        Line2D([0], [0], marker="s", linestyle="",
               markerfacecolor=candidate_colors[i % len(candidate_colors)],
               markeredgewidth=0, markersize=8, alpha=0.8, label=m)
        for i, m in enumerate(METHODS)
    ]
    fig.legend(
        legend_handles, METHODS,
        loc="upper center", bbox_to_anchor=(0.5, 1.0),
        ncol=len(METHODS), frameon=False, fontsize=8,
        handletextpad=0.4, columnspacing=2, borderaxespad=0.1,
    )

    # ---- Panel labels (a, b, c) ----
    for ax, lab in zip(axes, "abc"):
        ax.text(-0.15, 1.15, lab, transform=ax.transAxes,
                fontsize=12, fontname="Arial", fontweight="bold",
                ha="left", va="top")

    fig.savefig(OUT_PDF, bbox_inches="tight")
    print(f"[wrote] {OUT_PDF}")


if __name__ == "__main__":
    main()
