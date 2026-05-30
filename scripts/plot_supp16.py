"""Supplementary Figure 16: Per-window benchmark of Evo2HiC vs ORCA on
chr9+chr10 (the test holdout shared by both models), at the same 1Mb x 1Mb
window resolution as Fig 3.

Three structural metrics are compared:
  - IS_PCC          (continuous insulation-score profile fidelity, Crane 2015)
  - TAD boundary F1 (sweep over tolerance ±0..±10 bins, Forcato 2017)
  - Variation of Information   (partition-similarity, Yardimci 2019)
  - Loop F1 / P / R (HiCCUPS-style donut detector vs. canonical HiCCUPS GT)

Layout (rows x cols = 3 x 2):

  [A] IS_PCC mean +/- SE bars          [B] TAD F1 sweep curves
  [C] Variation of Information         [D] Loop P / R / F1 grouped bars
  [E] Per-window IS_PCC paired scatter [F] Per-window Loop F1 paired scatter

Produces:
  Figures/supplementary_16.pdf
  Figures/supplementary_16.png
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paths import (                                              # noqa: E402
    REPO, ensure_out_dir,
    RESULT_SEQ2HIC_NB_EVAL_DIR as DATA_DIR,
)
OUT_PDF = ensure_out_dir() / "supplementary_16.pdf"
OUT_PNG = ensure_out_dir() / "supplementary_16.png"

CELLS = ["H1ESC", "HFF"]
METHODS = ["evo2hic", "orca"]
M_LABEL = {"evo2hic": "Evo2HiC", "orca": "ORCA"}
M_COLOR = {"evo2hic": "#2E7DA8", "orca": "#D55E00"}
TOLERANCES = [0, 1, 2, 3, 5, 7, 10]

# title colour by direction-of-improvement vs ORCA, all metrics evaluated as
# per-window mean. green = Evo2HiC > ORCA on both cells; red otherwise.
GREEN = "#1f6e3a"
RED = "#9b1a1a"


def load_all():
    summary = pd.read_csv(DATA_DIR / "results.tsv", sep="\t")
    robust = pd.read_csv(DATA_DIR / "tad_robustness_summary.tsv", sep="\t")
    loop = pd.read_csv(DATA_DIR / "loop_results.tsv", sep="\t")
    per_w = {c: pd.read_csv(DATA_DIR / f"{c}_per_window.tsv", sep="\t")
             for c in CELLS}
    rob_pw = {c: pd.read_csv(DATA_DIR / f"{c}_tad_robustness.tsv", sep="\t")
              for c in CELLS}
    loop_pw = {c: pd.read_csv(DATA_DIR / f"{c}_loop_per_window.tsv", sep="\t")
               for c in CELLS}
    return summary, robust, loop, per_w, rob_pw, loop_pw


def _direction_color(deltas, lower_is_better=False):
    """Return GREEN if Evo2HiC wins all cells (delta > 0 by convention), RED
    otherwise. For lower-is-better metrics (VI), invert the sign convention."""
    if lower_is_better:
        deltas = [-d for d in deltas]
    return GREEN if all(d > 0 for d in deltas) else RED


def _bar_metric(ax, per_w, col_template, title, ylabel,
                ylim=None, hline=None, annotate_delta=True,
                lower_is_better=False):
    """Mean ± SE bars across per-window values. col_template = '{m}_{metric}'."""
    x = np.arange(len(CELLS))
    w = 0.36
    deltas = []
    for k, m in enumerate(METHODS):
        means, sems = [], []
        for cell in CELLS:
            v = per_w[cell][col_template.format(m=m)].dropna().to_numpy()
            means.append(np.mean(v))
            sems.append(np.std(v, ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0)
        ax.bar(x + (k - 0.5) * w, means, w,
               yerr=sems, capsize=3,
               color=M_COLOR[m], edgecolor="black", linewidth=0.7,
               label=M_LABEL[m])
    for cell in CELLS:
        evo = per_w[cell][col_template.format(m="evo2hic")].dropna().mean()
        orc = per_w[cell][col_template.format(m="orca")].dropna().mean()
        deltas.append(evo - orc)
    if annotate_delta:
        for i, cell in enumerate(CELLS):
            evo = per_w[cell][col_template.format(m="evo2hic")].dropna().mean()
            orc = per_w[cell][col_template.format(m="orca")].dropna().mean()
            d = evo - orc
            good = (d > 0) if not lower_is_better else (d < 0)
            ax.text(i, max(evo, orc) * 1.04 if (evo > 0 and orc > 0) else max(evo, orc) + 0.02,
                    f"Δ = {d:+.3f}", ha="center", fontsize=8.5,
                    color=GREEN if good else RED)
    ax.set_xticks(x)
    ax.set_xticklabels(CELLS)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10,
                 color=_direction_color(deltas, lower_is_better=lower_is_better))
    if ylim is not None:
        ax.set_ylim(ylim)
    if hline is not None:
        ax.axhline(hline, color="grey", linestyle="--", linewidth=0.6)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="lower right", fontsize=8, frameon=False)


def panel_A_is_pcc(ax, per_w):
    _bar_metric(ax, per_w, "{m}_IS_PCC",
                title="A   Insulation-score PCC (per-window mean)",
                ylabel="IS_PCC",
                ylim=(0.7, 0.95))


def panel_B_tad_sweep(ax, robust):
    deltas = []
    for cell, ls in zip(CELLS, ["-", "--"]):
        rows = {m: robust[(robust.cell == cell) & (robust.method == m)].iloc[0]
                for m in METHODS}
        for m in METHODS:
            ys = [rows[m][f"F1@{t}"] for t in TOLERANCES]
            ax.plot(TOLERANCES, ys,
                    marker="o" if m == "evo2hic" else "s",
                    color=M_COLOR[m], linestyle=ls, linewidth=1.8, markersize=6,
                    label=f"{M_LABEL[m]} ({cell})")
        # delta at the headline tolerance ±3 (matches our notebook eval)
        d = rows["evo2hic"]["F1@3"] - rows["orca"]["F1@3"]
        deltas.append(d)
    ax.set_xlabel("Tolerance (4-kb bins)")
    ax.set_ylabel("Boundary F1 (per-window mean)")
    ax.set_title("B   TAD F1 vs. tolerance (sweep, per-window mean)", fontsize=10,
                 color=_direction_color(deltas))
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, frameon=False, loc="lower right")


def panel_C_vi(ax, robust_pw):
    """Mean ± SE Variation of Information (lower = better)."""
    x = np.arange(len(CELLS))
    w = 0.36
    deltas = []
    for k, m in enumerate(METHODS):
        means, sems = [], []
        for cell in CELLS:
            sub = robust_pw[cell][robust_pw[cell]["method"] == m]
            v = sub["VI"].dropna().to_numpy()
            means.append(np.mean(v))
            sems.append(np.std(v, ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0)
        ax.bar(x + (k - 0.5) * w, means, w,
               yerr=sems, capsize=3,
               color=M_COLOR[m], edgecolor="black", linewidth=0.7,
               label=M_LABEL[m])
    for i, cell in enumerate(CELLS):
        e = robust_pw[cell].loc[robust_pw[cell]["method"] == "evo2hic", "VI"].mean()
        o = robust_pw[cell].loc[robust_pw[cell]["method"] == "orca", "VI"].mean()
        d = e - o
        deltas.append(d)
        good = d < 0  # lower-is-better
        ax.text(i, max(e, o) * 1.05, f"Δ = {d:+.3f}", ha="center", fontsize=8.5,
                color=GREEN if good else RED)
    ax.set_xticks(x)
    ax.set_xticklabels(CELLS)
    ax.set_ylabel("VI (normalised, lower better)")
    ax.set_title("C   Variation of Information (per-window mean)", fontsize=10,
                 color=_direction_color(deltas, lower_is_better=True))
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="upper right", fontsize=8, frameon=False)


def panel_D_loop(ax, loop_pw):
    """Per-window mean Loop Precision / Recall / F1 (matches paper Fig 3
    aggregation convention)."""
    metrics = [("P", "Precision"), ("R", "Recall"), ("F1", "F1")]
    pos = 0
    width = 0.35
    xticks, xtick_labels = [], []
    f1_deltas = []
    for cell in CELLS:
        for met_key, met_label in metrics:
            ev = loop_pw[cell][f"evo2hic_{met_key}"].dropna().mean()
            oc = loop_pw[cell][f"orca_{met_key}"].dropna().mean()
            for k, (m, val) in enumerate(zip(METHODS, [ev, oc])):
                ax.bar(pos + (k - 0.5) * width, val, width,
                       color=M_COLOR[m], edgecolor="black", linewidth=0.6)
            d = ev - oc
            ax.text(pos, max(ev, oc) + 0.02, f"{d:+.2f}",
                    ha="center", fontsize=7.5,
                    color=GREEN if d > 0 else RED)
            if met_key == "F1":
                f1_deltas.append(d)
            xticks.append(pos)
            xtick_labels.append(f"{met_label}\n{cell}")
            pos += 1
        pos += 0.5
    ax.set_xticks(xticks)
    ax.set_xticklabels(xtick_labels, fontsize=7.5)
    ax.set_ylabel("Loop metric (per-window mean)")
    ax.set_title("D   Loop Precision / Recall / F1 (per-window mean)",
                 fontsize=10,
                 color=_direction_color(f1_deltas))
    ax.grid(axis="y", alpha=0.3)
    handles = [mpatches.Patch(color=M_COLOR[m], label=M_LABEL[m]) for m in METHODS]
    ax.legend(handles=handles, loc="upper right", fontsize=8, frameon=False)


def panel_E_paired_is(ax, per_w):
    win_rates = []
    for cell, marker in zip(CELLS, ["o", "^"]):
        df = per_w[cell]
        x = df["orca_IS_PCC"].to_numpy()
        y = df["evo2hic_IS_PCC"].to_numpy()
        m = np.isfinite(x) & np.isfinite(y)
        x, y = x[m], y[m]
        wins = (y > x).sum()
        win_rates.append(wins / len(x) - 0.5)  # >0 means Evo wins majority
        ax.scatter(x, y, s=10, alpha=0.45, marker=marker, color="#444",
                   label=f"{cell}: {wins}/{len(x)} Evo2HiC > ORCA")
    lim = (0.0, 1.0)
    ax.plot(lim, lim, "--", color="grey", linewidth=0.8)
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("ORCA IS_PCC")
    ax.set_ylabel("Evo2HiC IS_PCC")
    ax.set_title("E   Per-window IS_PCC (paired)", fontsize=10,
                 color=_direction_color(win_rates))
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower right", frameon=False)
    ax.set_aspect("equal", adjustable="box")


def panel_F_paired_loop(ax, loop_pw):
    win_rates = []
    for cell, marker in zip(CELLS, ["o", "^"]):
        df = loop_pw[cell]
        x = df["orca_F1"].to_numpy()
        y = df["evo2hic_F1"].to_numpy()
        m = np.isfinite(x) & np.isfinite(y)
        x, y = x[m], y[m]
        wins = (y > x).sum()
        win_rates.append(wins / len(x) - 0.5)
        ax.scatter(x, y, s=10, alpha=0.45, marker=marker, color="#444",
                   label=f"{cell}: {wins}/{len(x)} Evo2HiC > ORCA")
    lim = (-0.02, 1.02)
    ax.plot([0, 1], [0, 1], "--", color="grey", linewidth=0.8)
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("ORCA Loop F1")
    ax.set_ylabel("Evo2HiC Loop F1")
    ax.set_title("F   Per-window Loop F1 (paired)", fontsize=10,
                 color=_direction_color(win_rates))
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower right", frameon=False)
    ax.set_aspect("equal", adjustable="box")


def main():
    summary, robust, loop, per_w, robust_pw, loop_pw = load_all()
    print("== summary (results.tsv) ==");  print(summary)
    print("\n== robust ==");  print(robust)
    print("\n== loop ==");    print(loop)

    fig = plt.figure(figsize=(13, 13))
    gs = fig.add_gridspec(3, 2, hspace=0.42, wspace=0.30)

    ax_a = fig.add_subplot(gs[0, 0]); panel_A_is_pcc(ax_a, per_w)
    ax_b = fig.add_subplot(gs[0, 1]); panel_B_tad_sweep(ax_b, robust)
    ax_c = fig.add_subplot(gs[1, 0]); panel_C_vi(ax_c, robust_pw)
    ax_d = fig.add_subplot(gs[1, 1]); panel_D_loop(ax_d, loop_pw)
    ax_e = fig.add_subplot(gs[2, 0]); panel_E_paired_is(ax_e, per_w)
    ax_f = fig.add_subplot(gs[2, 1]); panel_F_paired_loop(ax_f, loop_pw)

    fig.suptitle("Suppl. Fig. 16 — Per-window benchmark of Evo2HiC vs ORCA "
                 "(chr9+chr10, 376/378 windows, ORCA mallpreds.pth pipeline)",
                 fontsize=11, y=0.995)
    fig.savefig(OUT_PDF, bbox_inches="tight")
    fig.savefig(OUT_PNG, dpi=180, bbox_inches="tight")
    print(f"\n[saved] {OUT_PDF}")
    print(f"[saved] {OUT_PNG}")


if __name__ == "__main__":
    main()
