"""Supplementary Figure 17: Cross-species generalisation summary.

Adapted from `plot/plot_Fig6_revision.ipynb`. Rows 1 & 2 (Human / Mouse
held-out chromosomes) are pared down to the Evo2HiC vs Evo2HiC(HiC-only)
ablation only, isolating the contribution of the Evo2 sequence prior on
the same training cell line. Row 3 reuses the DNA Zoo polar tree showing
per-species Evo2HiC − HiC-only SPC improvement across 177 species.

Produces:
  Figures/supplementary_17.pdf
  Figures/supplementary_17_stats.tsv
"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from Bio import Phylo
from matplotlib.colorbar import ColorbarBase
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paths import (                                              # noqa: E402
    ensure_out_dir,
    RESULT_HUMAN_DIR, RESULT_MOUSE_DIR,
    SPC_MULTI_CSV, CLAUDE_CLADE_DIR, TREE_NWK,
)

from plot_settings import colors as candidate_colors  # noqa: E402
from plot_utils import (  # noqa: E402
    clear_test_log,
    dump_test_log,
    plot_box_with_points,
)

OUT_PDF = ensure_out_dir() / "supplementary_17.pdf"
OUT_STATS = ensure_out_dir() / "supplementary_17_stats.tsv"

plt.style.use("seaborn-v0_8-white")
plt.rcParams.update({"font.size": 8, "font.family": "Arial"})

color_maps = {
    "Evo2HiC": candidate_colors[0],
    "Evo2HiC(HiC-only)": "#fdb462",
}
ABLATION_METHODS = ["Evo2HiC", "Evo2HiC(HiC-only)"]
ABLATION_COLORS = [color_maps[m] for m in ABLATION_METHODS]
SIG_PAIR = ("Evo2HiC", "Evo2HiC(HiC-only)")


def _load(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t").rename(columns={"Unet": "Evo2HiC(HiC-only)"})


def _row_title(fig, axs_row, text):
    bbox_left = axs_row[0].get_position()
    bbox_right = axs_row[-1].get_position()
    center_x = (bbox_left.x0 + bbox_right.x1) / 2
    top_y = max(ax.get_position().y1 for ax in axs_row)
    fig.text(center_x, top_y + 0.01, text, ha="center", va="bottom", fontsize=12)


def _print_ablation_summary(label: str, root: Path, metrics):
    print(f"\n=== Evo2HiC vs Evo2HiC(HiC-only) ({label}, per metric) ===")
    for met in metrics:
        df = _load(root / f"{met}.csv")
        abs_impr = float(df["Evo2HiC"].mean() - df["Evo2HiC(HiC-only)"].mean())
        rel_impr = float((df["Evo2HiC"] - df["Evo2HiC(HiC-only)"]).mean())
        print(
            f"  {met:4s}: mean(Evo2HiC) - mean(HiC-only) = {abs_impr:+.4f} | "
            f"mean(Evo2HiC - HiC-only) = {rel_impr:+.4f}"
        )


def panel_metric_row(axes_row, root: Path, panel_tag: str, metrics):
    for ax, met in zip(axes_row, metrics):
        df = _load(root / f"{met}.csv")
        plot_box_with_points(
            ax, df, ABLATION_METHODS,
            colors=ABLATION_COLORS,
            ylabel=met,
            point_size=2,
            sig_pair=SIG_PAIR,
            yticks_num=2,
            log_panel=panel_tag,
            log_metric=met,
        )


def panel_polar_tree(ax_bottom):
    """DNA Zoo polar tree of per-species Evo2HiC − HiC-only SPC improvement.

    Faithful port of cell 21–22 of plot_Fig6_revision.ipynb. The lower-half
    of the figure shows a circular phylogeny with leaves coloured by clade
    and an outer bar track encoding ΔSPC (Evo2HiC − Unet) per species.
    """
    methods = ["EvoHiC", "Unet"]

    def name_mapping(species):
        synonym = {
            "Herpailurus_yagouaroundi": "Puma_yagouaroundi",
            "Eulemur_collaris": "Eulemur_fulvus_collaris",
        }
        if species in synonym:
            return synonym[species]
        if "__" in species:
            return species.split("__")[0]
        return species

    result = pd.read_csv(SPC_MULTI_CSV, sep="\t")
    species_list = [name_mapping(s) for s in result["species"].tolist()]
    improve = (result[methods[0]] - result[methods[1]]).tolist()
    species2improve = dict(zip(species_list, improve))

    _human_spc = pd.read_csv(RESULT_HUMAN_DIR / "SPC.csv", sep="\t")
    homo_improve = float(_human_spc["Evo2HiC"].mean() - _human_spc["Unet"].mean())

    claude_name_list = [
        "Angiosperms", "Protostomes", "Actinopterygii",
        "Reptilia", "Mammalia", "Marsupialia",
    ]
    group_name2species_list = defaultdict(list)
    for claude_name in claude_name_list:
        cur_path = CLAUDE_CLADE_DIR / f"claude_{claude_name}_clean.txt"
        with open(cur_path, "r") as f:
            for line in f:
                group_name2species_list[claude_name].append(line.strip("\n"))

    _tree = Phylo.read(str(TREE_NWK), "newick")

    _leaf2score = dict(species2improve)
    _leaf2score["Homo_sapiens"] = homo_improve

    def _subtree_mean_improve(clade):
        vals = [_leaf2score.get(l.name, 0.0) for l in clade.get_terminals()]
        return float(np.mean(vals)) if vals else 0.0

    for _clade in _tree.find_clades():
        if _clade.is_terminal():
            continue
        _desc = sorted(_clade.clades, key=_subtree_mean_improve, reverse=True)
        _arranged = []
        for _i, _c in enumerate(_desc):
            if _i % 2 == 0:
                _arranged.append(_c)
            else:
                _arranged.insert(0, _c)
        _clade.clades = _arranged

    _leaves_in_order = [t.name for t in _tree.get_terminals()]
    _N = len(_leaves_in_order)
    _scores_arr = np.array([_leaf2score.get(n, 0.0) for n in _leaves_in_order])
    _theta = np.arange(_N) / _N * 2 * np.pi

    _w_peak = np.clip(_scores_arr, 0.0, None)
    _peak_c = np.sum(_w_peak * np.exp(1j * _theta)) / max(_w_peak.sum(), 1e-9)
    _peak_tree_deg = float(np.degrees(np.angle(_peak_c))) % 360.0

    _target_peak_deg = 45.0
    _raw_start = _target_peak_deg - _peak_tree_deg
    _start_deg = _raw_start % 360.0
    if _start_deg > 0.0:
        _start_deg -= 360.0
    _span = 360.0 - 1e-3
    _end_deg = _start_deg + _span

    from pycirclize import Circos
    from pycirclize.utils import ColorCycler

    circos, tv = Circos.initialize_from_tree(
        _tree,
        start=_start_deg,
        end=_end_deg,
        r_lim=(30, 80),
        leaf_label_rmargin=21,
        leaf_label_size=5,
        ignore_branch_length=True,
        label_formatter=lambda t: t.replace("_", " "),
    )

    S = set(tv.leaf_labels)
    ColorCycler.set_cmap("tab10")
    group_name2color = {name: ColorCycler() for name in group_name2species_list.keys()}
    for group_name, sps in group_name2species_list.items():
        color = group_name2color[group_name]
        S.difference_update(set(sps))
        sps = list(set(sps).intersection(set(tv.leaf_labels)))
        if not sps:
            continue
        tv.set_node_line_props(sps, color=color, apply_label_color=True)

    sector = circos.sectors[0]
    bar_track = sector.add_track((80, 100))
    species_list_t = tv.leaf_labels
    height = []
    min_clip = -0.02
    for x in species_list_t:
        if x == "Homo_sapiens":
            height.append(homo_improve)
            continue
        if x in species2improve:
            cur_val = max(min_clip, species2improve[x])
            height.append(cur_val)
        else:
            height.append(0)

    percentile_98 = np.percentile(height, 98)
    percentile_98 = max(float(percentile_98), -min_clip)
    height = np.minimum(height, percentile_98)
    rad_list = np.arange(0, int(len(species_list_t))) + 0.5
    my_cmap = sns.cubehelix_palette(as_cmap=True)
    bar_norm = Normalize(vmin=min_clip, vmax=percentile_98)
    bar_track.bar(
        rad_list, height, vmin=min_clip - 0.003,
        color=my_cmap(bar_norm(np.asarray(height))),
    )

    circos.plotfig(ax=ax_bottom)
    circos.ax.legend(
        handles=[Line2D([], [], label=n, color=c) for n, c in group_name2color.items()],
        labelcolor=group_name2color.values(),
        fontsize=7,
        loc="center",
        bbox_to_anchor=(0.5, 0.5),
    )

    return my_cmap, min_clip, percentile_98, species2improve, homo_improve, group_name2species_list


def add_polar_colorbar(fig, ax_bottom, my_cmap, min_clip, percentile_98):
    for a in list(fig.axes):
        if a.get_label() == "spc_cb":
            a.remove()
    fig.canvas.draw()
    bbox = ax_bottom.get_position()
    norm = Normalize(vmin=min_clip, vmax=percentile_98)
    x0 = bbox.x1 + 0.015
    y0 = bbox.y0 - 0.025
    cax = fig.add_axes([x0, y0, 0.1, 0.01], label="spc_cb")
    cax.set_clip_on(False)
    cbar = ColorbarBase(cax, cmap=my_cmap, norm=norm, orientation="horizontal")
    cbar.set_label("SPC absolute improvement", fontsize=8, labelpad=2)
    cbar.ax.tick_params(labelsize=7, length=2)
    cbar.outline.set_linewidth(0.6)


def add_panel_labels(fig, axes_top, ax_bottom):
    labels = ["a", "b"]
    for ax, label in zip([axes_top[0][0], axes_top[1][0]], labels):
        ax.text(
            -0.2, 1.3, label,
            transform=ax.transAxes,
            fontsize=12, fontname="Arial", fontweight="bold",
            ha="left", va="top",
        )
    ax_bottom.text(
        -0.2, -0.2, "c",
        transform=axes_top[1][0].transAxes,
        fontsize=12, fontname="Arial", fontweight="bold",
        ha="left", va="top",
    )


def add_top_legend(fig, axes_top):
    pos0 = axes_top[0][0].get_position()
    pos1 = axes_top[0][3].get_position()
    x = (pos0.x0 + pos1.x1) / 2
    y = pos0.y1 + 0.05
    if hasattr(fig, "legend_") and fig.legend_ is not None:
        fig.legend_.remove()
    handles = [
        Line2D([0], [0], marker="s", linestyle="", markersize=8,
               markerfacecolor=color_maps[m], markeredgewidth=0,
               alpha=0.8, label=m)
        for m in ABLATION_METHODS
    ]
    fig.legend(
        handles, ABLATION_METHODS,
        loc="center", bbox_to_anchor=(x, y),
        ncol=len(ABLATION_METHODS),
        frameon=False, fontsize=8,
        handletextpad=0.4, columnspacing=1.5, borderaxespad=0.1,
    )


def main():
    clear_test_log()

    fig = plt.figure(figsize=(8.27, 10))
    gs = gridspec.GridSpec(
        5, 4, figure=fig,
        height_ratios=[2, 1, 2, 2, 7],
        hspace=0, wspace=0.5,
    )

    axes_top = []
    for i in [0, 2]:
        row_axes = []
        for j in range(4):
            ax = fig.add_subplot(gs[i, j])
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_linewidth(0.5)
            row_axes.append(ax)
        axes_top.append(row_axes)

    ax_bottom = fig.add_subplot(gs[4, :], projection="polar")
    ax_bottom.set_xticks([])
    ax_bottom.set_yticks([])

    metrics = ["PCC", "SPC", "PSNR", "SSIM"]

    panel_metric_row(axes_top[0], RESULT_HUMAN_DIR, "SuppFig17a_Human", metrics)
    _row_title(fig, axes_top[0], "Human")
    _print_ablation_summary("Human", RESULT_HUMAN_DIR, metrics)

    panel_metric_row(axes_top[1], RESULT_MOUSE_DIR, "SuppFig17b_Mouse", metrics)
    _row_title(fig, axes_top[1], "Mouse")
    _print_ablation_summary("Mouse", RESULT_MOUSE_DIR, metrics)

    add_top_legend(fig, axes_top)

    my_cmap, min_clip, percentile_98, *_ = panel_polar_tree(ax_bottom)
    add_polar_colorbar(fig, ax_bottom, my_cmap, min_clip, percentile_98)

    add_panel_labels(fig, axes_top, ax_bottom)

    fig.savefig(OUT_PDF, bbox_inches="tight")
    print(f"\n[saved] {OUT_PDF}")

    log_df = dump_test_log(str(OUT_STATS))
    print(f"[saved] {OUT_STATS}")
    print(log_df.to_string(index=False))


if __name__ == "__main__":
    main()
