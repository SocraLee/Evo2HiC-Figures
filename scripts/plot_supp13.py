"""Supplementary Figure 13 — Evo2HiC vs HICARN2 across the 177-species
DNA Zoo cohort.

Layout (2 rows):
  - Row 1, panel a : Per-clade Δ TAD boundary F1 box + strip
  - Row 2, panel b : Patristic-distance schematic
  - Row 2, panel c : Δ TAD F1 vs patristic distance (MYA) scatter +
                     OLS regression line

Outputs
-------
Figures/supplementary_13.pdf
Figures/supplementary_13_stats.tsv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import wilcoxon, pearsonr
from Bio import Phylo

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paths import (                                              # noqa: E402
    REPO, ensure_out_dir,
    TAD_REV_TSV as TAD_TSV,
    TREE_NWK,
)
from _supp_style import (                                        # noqa: E402
    apply_supp_style, load_clade_map, assign_clade, PANEL_LABEL_KW,
    DELTA_POS_COLOR, DELTA_NEG_COLOR,
)

OUT_PDF   = ensure_out_dir() / 'supplementary_13.pdf'
OUT_STATS = ensure_out_dir() / 'supplementary_13_stats.tsv'

CLADE_ORDER = ['Mammalia', 'Marsupialia', 'Reptilia', 'Actinopterygii',
               'Protostomes', 'Angiosperms']

PATH_COLOR = '#1a73e8'
BG_COLOR   = '#bbb'
HUMAN_LEAF = 'Homo_sapiens'


def _p_to_stars(p):
    if not np.isfinite(p): return 'n.s.'
    return '****' if p < 1e-4 else '***' if p < 1e-3 else '**' if p < 1e-2 \
        else '*' if p < 5e-2 else 'n.s.'


def name_map(s):
    syn = {'Herpailurus_yagouaroundi': 'Puma_yagouaroundi',
           'Eulemur_collaris':         'Eulemur_fulvus_collaris'}
    if s in syn: return syn[s]
    if '__' in s: return s.split('__')[0]
    return s


def patristic_to_human(tree, names):
    leaves = {l.name: l for l in tree.get_terminals()}
    human = leaves[HUMAN_LEAF]
    out = {}
    for n in names:
        if n == HUMAN_LEAF: out[n] = 0.0
        elif n in leaves:
            try: out[n] = float(tree.distance(human, leaves[n]))
            except Exception: out[n] = np.nan
        else: out[n] = np.nan
    return out


# ----------------------------------------------------------------------
# Row 1: per-clade Δ TAD F1 box+strip
# ----------------------------------------------------------------------
def panel_per_clade_box(ax, df, log_rows):
    clades = [c for c in CLADE_ORDER if (df['Clade'] == c).any()]
    rng = np.random.RandomState(42)

    for i, cl in enumerate(clades):
        sub = df[df['Clade'] == cl]['dF1'].dropna().values
        if len(sub) == 0: continue
        ax.boxplot(
            [sub], positions=[i], widths=0.5,
            patch_artist=True, showfliers=False,
            boxprops=dict(facecolor='none', edgecolor='#444', linewidth=0.6),
            medianprops=dict(color='#444', linewidth=0.6),
            whiskerprops=dict(color='#444', linewidth=0.4),
            capprops=dict(color='#444', linewidth=0.4),
        )
        jitter = rng.uniform(-0.15, 0.15, size=len(sub))
        colors = [DELTA_POS_COLOR if v > 0 else DELTA_NEG_COLOR for v in sub]
        ax.scatter(np.full(len(sub), i) + jitter, sub, s=8, alpha=0.7,
                   linewidths=0, c=colors)

        n_pos = int((sub > 0).sum())
        if len(sub) >= 3:
            try: stat, p = wilcoxon(sub, alternative='greater')
            except ValueError: stat, p = float('nan'), float('nan')
        else:
            stat, p = float('nan'), float('nan')
        log_rows.append({
            'panel': 'a', 'scope': cl, 'n': len(sub),
            'n_positive': n_pos,
            'pct_positive': float(n_pos / len(sub) * 100),
            'mean_delta': float(np.mean(sub)),
            'median_delta': float(np.median(sub)),
            'wilcoxon_p_one_sided_greater': float(p),
            'stars': _p_to_stars(p),
        })

    a = df['dF1'].dropna().values
    n_pos = int((a > 0).sum())
    stat, p_global = wilcoxon(a, alternative='greater')
    log_rows.append({
        'panel': 'a', 'scope': 'GLOBAL', 'n': len(a),
        'n_positive': n_pos,
        'pct_positive': float(n_pos / len(a) * 100),
        'mean_delta': float(np.mean(a)),
        'median_delta': float(np.median(a)),
        'wilcoxon_p_one_sided_greater': float(p_global),
        'stars': _p_to_stars(p_global),
    })

    ax.axhline(0, color='black', lw=0.5, ls='--', alpha=0.55)
    ax.set_xticks(range(len(clades)))
    ax.set_xticklabels(clades, rotation=25, ha='right')
    ax.set_ylabel(r'$\Delta$ TAD boundary F1 (Evo2HiC − HICARN2)')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.text(-0.04, 1.04, 'a', transform=ax.transAxes, **PANEL_LABEL_KW)

    handles = [
        Line2D([0], [0], marker='o', linestyle='', markersize=5,
               markerfacecolor=DELTA_POS_COLOR, markeredgewidth=0,
               label=r'$\Delta > 0$'),
        Line2D([0], [0], marker='o', linestyle='', markersize=5,
               markerfacecolor=DELTA_NEG_COLOR, markeredgewidth=0,
               label=r'$\Delta \leq 0$'),
    ]
    ax.legend(handles=handles, loc='upper right', frameon=False)


# ----------------------------------------------------------------------
# Row 2 panel b: patristic distance schematic (mirrored: past → present)
# ----------------------------------------------------------------------
def panel_schematic(ax):
    ax.clear()
    leaves = {'Species A': (180, 2), 'Species B': (180, 1),
              'Species C': (180, 0)}
    n_AB  = (120, 1.5)
    n_ABC = (0,   0.75)

    def seg(p1, p2, color=BG_COLOR, lw=1.2, zorder=1):
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color=color, lw=lw,
                solid_capstyle='round', zorder=zorder)

    seg(n_AB, (n_AB[0], leaves['Species A'][1]))
    seg(n_AB, (n_AB[0], leaves['Species B'][1]))
    seg((n_AB[0], leaves['Species A'][1]), leaves['Species A'])
    seg((n_AB[0], leaves['Species B'][1]), leaves['Species B'])
    seg(n_ABC, (n_ABC[0], n_AB[1]))
    seg((n_ABC[0], n_AB[1]), n_AB)
    seg(n_ABC, (n_ABC[0], leaves['Species C'][1]))
    seg((n_ABC[0], leaves['Species C'][1]), leaves['Species C'])

    path = [
        (leaves['Species A'], (n_AB[0], leaves['Species A'][1])),
        ((n_AB[0], leaves['Species A'][1]), n_AB),
        (n_AB, (n_ABC[0], n_AB[1])),
        ((n_ABC[0], n_AB[1]), n_ABC),
        (n_ABC, (n_ABC[0], leaves['Species C'][1])),
        ((n_ABC[0], leaves['Species C'][1]), leaves['Species C']),
    ]
    for p1, p2 in path:
        seg(p1, p2, color=PATH_COLOR, lw=2.4, zorder=2)

    ax.text((n_AB[0] + leaves['Species A'][0]) / 2,
            leaves['Species A'][1] + 0.10, '+60 MYA',
            ha='center', va='bottom', color=PATH_COLOR, fontweight='bold')
    ax.text((n_ABC[0] + n_AB[0]) / 2, n_AB[1] - 0.18, '+120 MYA',
            ha='center', va='top', color=PATH_COLOR, fontweight='bold')
    ax.text((n_ABC[0] + leaves['Species C'][0]) / 2,
            leaves['Species C'][1] - 0.16, '+180 MYA',
            ha='center', va='top', color=PATH_COLOR, fontweight='bold')

    for lname, (lx, ly) in leaves.items():
        ax.scatter([lx], [ly], s=40, color='black', zorder=4)
        ax.text(lx + 8, ly, lname, ha='left', va='center', fontstyle='italic')
    ax.scatter([n_AB[0]], [n_AB[1]], s=22, color=BG_COLOR, zorder=3)
    ax.scatter([n_ABC[0]], [n_ABC[1]], s=22, color=BG_COLOR, zorder=3)
    ax.text(n_AB[0] - 4, n_AB[1] + 0.10, 'MRCA(A,B)',
            ha='right', color='#555')
    ax.text(n_ABC[0] - 4, n_ABC[1] + 0.10, 'MRCA(A,C)',
            ha='right', color='#555')

    ax.annotate('', xy=(230, -0.5), xytext=(-50, -0.5),
                arrowprops=dict(arrowstyle='->', color='#444', lw=0.7))
    ax.text(-50, -0.7, 'past',    ha='left',  va='top', color='#444')
    ax.text(230, -0.7, 'present', ha='right', va='top', color='#444')
    ax.text(90,  -0.7, 'time (MYA)', ha='center', va='top', color='#444')

    ax.text(110, 2.4,
            'Patristic distance(A, C) = 60 + 120 + 180 = 360 MYA\n'
            r'              $= 2 \times$ time-since-MRCA(A,C)',
            ha='center', va='bottom',
            color=PATH_COLOR, fontweight='bold')

    ax.set_xlim(-110, 250); ax.set_ylim(-1.0, 3.1)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ('top', 'right', 'bottom', 'left'):
        ax.spines[sp].set_visible(False)
    ax.text(-0.04, 1.04, 'b', transform=ax.transAxes, **PANEL_LABEL_KW)


# ----------------------------------------------------------------------
# Row 2 panel c: Δ TAD F1 vs patristic distance scatter
# ----------------------------------------------------------------------
def panel_scatter(ax, df, log_rows):
    sub = df.dropna(subset=['mya', 'dF1']).copy()
    pos = sub['dF1'] > 0
    ax.scatter(sub.loc[pos, 'mya'], sub.loc[pos, 'dF1'],
               s=14, alpha=0.7, color=DELTA_POS_COLOR, linewidths=0,
               label=r'$\Delta > 0$')
    ax.scatter(sub.loc[~pos, 'mya'], sub.loc[~pos, 'dF1'],
               s=14, alpha=0.55, color=DELTA_NEG_COLOR, linewidths=0,
               label=r'$\Delta \leq 0$')
    ax.axhline(0, color='black', lw=0.5, ls='--', alpha=0.6)

    x_all = sub['mya'].values; y_all = sub['dF1'].values
    if len(x_all) >= 3:
        coef = np.polyfit(x_all, y_all, 1)
        xx = np.linspace(x_all.min(), x_all.max(), 100)
        ax.plot(xx, np.polyval(coef, xx), color='black', lw=1.0, alpha=0.85)
        r_p, p_p = pearsonr(x_all, y_all)
        # Annotate Pearson r on the panel (matches the OLS line shown).
        p_p_str = (f'{p_p:.1e}' if (np.isfinite(p_p) and p_p < 1e-3)
                   else f'{p_p:.3f}')
        ax.text(
            0.03, 0.03,
            f'Pearson $r$ = {r_p:.2f}\n$P$ = {p_p_str}  ($n$ = {len(x_all)})',
            transform=ax.transAxes, ha='left', va='bottom',
            fontsize=8,
            bbox=dict(boxstyle='round,pad=0.25',
                      facecolor='white', edgecolor='none', alpha=0.7),
        )
        log_rows.append({
            'panel': 'c', 'scope': 'OLS regression vs MYA', 'n': len(x_all),
            'n_positive': int(((y_all) > 0).sum()),
            'pct_positive': float(((y_all) > 0).mean() * 100),
            'mean_delta': float(y_all.mean()),
            'median_delta': float(np.median(y_all)),
            'wilcoxon_p_one_sided_greater': float('nan'),
            'stars': '',
            'pearson_r': float(r_p), 'pearson_p': float(p_p),
        })

    ax.set_xlabel('Patristic distance to H. sapiens (MYA)')
    ax.set_ylabel(r'$\Delta$ TAD boundary F1 (Evo2HiC − HICARN2)')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(frameon=False, loc='upper right')
    ax.text(-0.13, 1.04, 'c', transform=ax.transAxes, **PANEL_LABEL_KW)


def main():
    apply_supp_style()

    df = pd.read_csv(TAD_TSV, sep='\t')
    df['dF1'] = df['TAD_f1_evo2hic'] - df['TAD_f1_hicarn2']

    base_map = load_clade_map(REPO)
    df['Clade'] = df['species'].apply(lambda s: assign_clade(s, base_map))

    tree = Phylo.read(str(TREE_NWK), 'newick')
    leaf_names = [l.name for l in tree.get_terminals()]
    df['short'] = df['species'].apply(name_map)
    leaf2dist = patristic_to_human(tree, leaf_names)
    df['mya'] = df['short'].map(leaf2dist)

    print(f'[load] n={len(df)} species; '
          f'{df["mya"].notna().sum()} mapped to tree')

    log_rows = []

    fig = plt.figure(figsize=(7.5, 6.4))

    # Row 2 first (b + c): standard gridspec, frozen layout
    gs_row2 = fig.add_gridspec(
        1, 2,
        width_ratios=[1.1, 1.0],
        wspace=0.30,
        left=0.10, right=0.96,
        bottom=0.08, top=0.45,
    )
    ax_schem = fig.add_subplot(gs_row2[0, 0])
    panel_schematic(ax_schem)

    ax_scatter = fig.add_subplot(gs_row2[0, 1])
    panel_scatter(ax_scatter, df, log_rows)

    # Force a layout pass so transforms are accurate
    fig.canvas.draw()

    # Compute panel-a horizontal extent:
    #   left  = figure-x of panel-b's "past" label (data x = -50)
    #   right = right edge of panel-c's axes
    past_data_xy   = (-50, -0.7)
    past_disp      = ax_schem.transData.transform(past_data_xy)
    past_fig_x     = fig.transFigure.inverted().transform(past_disp)[0]
    c_right_fig_x  = ax_scatter.get_position().x1

    # Panel-a axes box (figure coords)
    a_bottom = 0.55
    a_top    = 0.93
    ax_box = fig.add_axes(
        [past_fig_x, a_bottom,
         c_right_fig_x - past_fig_x, a_top - a_bottom],
    )
    panel_per_clade_box(ax_box, df, log_rows)

    # ----- Re-place panel labels at consistent figure-x = panel-b label x
    # so a / b are vertically aligned (and c stays where it was).
    # Strip the per-axes labels first and re-add via fig.text().
    for ax in (ax_box, ax_schem, ax_scatter):
        for t in list(ax.texts):
            if t.get_text() in {'a', 'b', 'c'} and t.get_fontweight() == 'bold':
                t.remove()

    # panel-b label x in figure coords (its previous axes-rel x = -0.04)
    b_pos = ax_schem.get_position()
    b_label_fig_x = b_pos.x0 - 0.04 * b_pos.width
    # panel-c label x: keep its original axes-rel = -0.13
    c_pos = ax_scatter.get_position()
    c_label_fig_x = c_pos.x0 - 0.13 * c_pos.width

    a_pos = ax_box.get_position()
    fig.text(b_label_fig_x, a_pos.y1 + 0.01, 'a', **PANEL_LABEL_KW)
    fig.text(b_label_fig_x, b_pos.y1 + 0.01, 'b', **PANEL_LABEL_KW)
    fig.text(c_label_fig_x, c_pos.y1 + 0.01, 'c', **PANEL_LABEL_KW)

    OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PDF, bbox_inches='tight')
    print(f'[save] {OUT_PDF}')

    pd.DataFrame(log_rows).to_csv(OUT_STATS, sep='\t', index=False)
    print(f'[save] {OUT_STATS}')
    print()
    print(pd.DataFrame(log_rows).to_string(index=False))


if __name__ == '__main__':
    main()
