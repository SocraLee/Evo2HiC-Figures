"""
Cross-cell-type evaluation plots (rebuttal R3).

Reads TSVs written by `evaluate.crosscell_eval` and produces one multi-panel
figure summarising:

  Panel A — 3x3 cross-cell PCC heatmaps for two representative tracks
            (DNase, H3K27ac), one column per model
            {GT, Evo2, HiC_only, Evo2HiC}.

  Panel B — 1-D track demonstration at one cell-type-specific locus,
            illustrating what "cell-type differential PCC" measures.
            Three vertically stacked sub-rows show, over a ±50 kb window:
              (top)    GT signal in cell A vs cell B
              (middle) Evo2HiC predicted signal in cell A vs cell B
              (bottom) ΔGT vs ΔEvo2HiC (the difference vectors that
                       Panel C correlates).

  Panel C — Cell-type differential PCC per track, averaged over the 3
            unordered cell-pairs, grouped bars over the 3 models.
            Significance brackets (Williams + Fisher) only on tracks
            where Evo2HiC is the empirical best.

Output: Figures/supplementary_12.pdf
"""
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import combine_pvalues
from pathlib import Path

plt.rcParams.update({
    'font.size':       7,
    'font.family':     'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans'],
    'mathtext.fontset': 'dejavuserif',                 # Arial does not ship math glyphs;
                                                       # use a sans-serif math fallback
    'axes.titlesize':  7,
    'axes.labelsize':  7,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'legend.fontsize': 6,
    'pdf.fonttype':    42,                             # editable TrueType in PDF
    'ps.fonttype':     42,
})

# Nature-style double-column figure width: 180 mm
FIG_WIDTH_INCH = 180 / 25.4                            # ≈ 7.087 in
PANEL_LABEL_FONTSIZE = 7.5

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paths import (                                              # noqa: E402
    REPO, ensure_out_dir, add_repo_to_syspath,
    RESULT_CROSSCELL_DIR as IN_DIR,
    CKPT_ROOT as _CKPT_PATH,
    EPI_EVO2HIC_DIR, EPI_EVO2_DIR,
)
add_repo_to_syspath()

from plot_settings import colors as PALETTE  # noqa: E402

OUT_PDF = ensure_out_dir() / 'supplementary_12.pdf'

CELLS  = ['GM12878', 'H1ESC', 'K562']
TRACKS = ['DNase', 'CTCF', 'H3K27ac', 'H3K27me3', 'H3K4me3']
MODELS = ['Evo2', 'HiC_only', 'Evo2HiC']  # plotting order
DISPLAY = {'Evo2': 'Evo2', 'HiC_only': 'HiC-only', 'Evo2HiC': 'Evo2HiC'}

REP_TRACKS = ['DNase', 'H3K27ac']                    # rows of Panel A

COLOR = {
    'Evo2':     PALETTE[1],   # purple
    'HiC_only': PALETTE[2],   # blue
    'Evo2HiC':  PALETTE[0],   # red
    'GT':       '#888888',
}


# --------------------------------------------------------------------------
# Load TSVs
# --------------------------------------------------------------------------
def load_data():
    cm  = pd.read_csv(IN_DIR / 'crosscell_matrix.tsv',  sep='\t')
    gap = pd.read_csv(IN_DIR / 'spec_gap.tsv',          sep='\t')
    sp  = pd.read_csv(IN_DIR / 'spec_pcc.tsv',          sep='\t')
    wt  = pd.read_csv(IN_DIR / 'williams_pvals.tsv',    sep='\t')
    return cm, gap, sp, wt


def matrix_3x3(cm_long, model, track):
    sub = cm_long[(cm_long.model == model) & (cm_long.track == track)]
    M = np.zeros((len(CELLS), len(CELLS)))
    for _, row in sub.iterrows():
        i = CELLS.index(row.pred_cell)
        j = CELLS.index(row.gt_cell)
        M[i, j] = row.pcc
    return M


def p_to_stars(p):
    return '****' if p < 1e-4 else '***' if p < 1e-3 else '**' if p < 1e-2 \
        else '*' if p < 5e-2 else 'n.s.'


def fisher_combine(pvals):
    pvals = [p for p in pvals if np.isfinite(p) and 0 < p <= 1]
    if not pvals:
        return float('nan')
    # combine_pvalues clamps very small inputs; floor here so chi^2 stays finite
    pvals = [max(p, 1e-300) for p in pvals]
    return combine_pvalues(pvals, method='fisher').pvalue


# --------------------------------------------------------------------------
# Panel A — 2 rows (REP_TRACKS) x 5 cols (GT + 4 models) of 3x3 heatmaps
# --------------------------------------------------------------------------
def panel_a(axes_grid, cm):
    """axes_grid: 2D list, shape (len(REP_TRACKS), 1 + len(MODELS))."""
    panel_models = ['GT'] + MODELS
    vmin, vmax = 0, 1
    for r, track in enumerate(REP_TRACKS):
        last_im = None
        for c, m in enumerate(panel_models):
            ax = axes_grid[r][c]
            M  = matrix_3x3(cm, m, track)
            last_im = ax.imshow(M, cmap='RdBu_r', vmin=vmin, vmax=vmax, aspect='equal')
            if r == 0:
                ax.set_title(f'{DISPLAY.get(m, m)}')
            if r == len(REP_TRACKS) - 1:
                ax.set_xticks(range(len(CELLS)))
                ax.set_xticklabels(CELLS, rotation=45, ha='right')
                ax.set_xlabel('GT cell')
            else:
                ax.set_xticks([])
            if c == 0:
                ax.set_yticks(range(len(CELLS)))
                ax.set_yticklabels(CELLS)
                ax.set_ylabel(f'{track}\npredicted cell')
            else:
                ax.set_yticks([])
            for i in range(len(CELLS)):
                for j in range(len(CELLS)):
                    txt_color = 'white' if abs(M[i, j] - 0.5) > 0.3 else 'black'
                    ax.text(j, i, f'{M[i, j]:.2f}', ha='center', va='center',
                            color=txt_color)

        # one colorbar per row, attached to the rightmost heatmap of that row
        cax = axes_grid[r][-1].inset_axes([1.08, 0.0, 0.05, 1.0])
        plt.colorbar(last_im, cax=cax).set_label('PCC')


# --------------------------------------------------------------------------
# Panel B — Δ-signal SpecPCC barplot with Williams stars (only Evo2HiC vs
# the best baseline, and only when Evo2HiC is itself the best on that track)
# --------------------------------------------------------------------------
def panel_b(ax, sp, wt):
    # dedupe directed pairs (A,B) and (B,A) give identical SpecPCC
    sp = sp.copy()
    sp['unordered'] = sp.pair.apply(
        lambda s: '_'.join(sorted(s.split('_vs_')))
    )
    sp = sp.drop_duplicates(['unordered', 'track'])

    wt = wt.copy()
    wt['unordered'] = wt.pair.apply(
        lambda s: '_'.join(sorted(s.split('_vs_')))
    )
    wt = wt.drop_duplicates(['unordered', 'track', 'baseline'])

    x = np.arange(len(TRACKS))
    n = len(MODELS)
    w = 0.18
    offsets = (np.arange(n) - (n - 1) / 2) * w
    means = {}
    for m in MODELS:
        col = f'specpcc_{m}'
        means[m] = [sp[sp.track == t][col].mean() for t in TRACKS]

    for i, m in enumerate(MODELS):
        ax.bar(x + offsets[i], means[m], w,
               color=COLOR[m], label=DISPLAY[m],
               edgecolor='black', linewidth=0.5)

    # Significance: only annotate tracks where Evo2HiC is itself the best,
    # against the runner-up baseline (Fisher-combined Williams p across pairs).
    baselines = [m for m in MODELS if m != 'Evo2HiC']
    e2h_idx = MODELS.index('Evo2HiC')
    overall_top = max(max(v) for v in means.values())
    pad = 0.02
    bracket_h = 0.025
    for ti, t in enumerate(TRACKS):
        e2h = means['Evo2HiC'][ti]
        baseline_vals = {m: means[m][ti] for m in baselines}
        if e2h < max(baseline_vals.values()):
            continue                                           # Evo2HiC not best → no bracket
        runner_up = max(baseline_vals, key=baseline_vals.get)
        ru_idx = MODELS.index(runner_up)
        ps = wt[(wt.track == t) & (wt.baseline == runner_up)].p_one_sided.values
        p_combined = fisher_combine(ps)
        stars = p_to_stars(p_combined) if np.isfinite(p_combined) else ''
        if stars == 'n.s.':
            continue

        x_l, x_r = ti + offsets[ru_idx], ti + offsets[e2h_idx]
        y_l = means[runner_up][ti]
        y_r = e2h
        y_top = max(y_l, y_r) + pad + bracket_h
        ax.plot([x_l, x_l], [y_l + pad, y_top], color='black', lw=0.7, clip_on=False)
        ax.plot([x_r, x_r], [y_r + pad, y_top], color='black', lw=0.7, clip_on=False)
        ax.plot([x_l, x_r], [y_top, y_top], color='black', lw=0.7, clip_on=False)
        ax.text((x_l + x_r) / 2, y_top + 0.003, stars,
                ha='center', va='bottom')

    ax.set_xticks(x); ax.set_xticklabels(TRACKS, rotation=20, ha='right')
    ax.set_ylabel('Cell-type differential PCC\n' + r'$\mathrm{corr}(\Delta\mathrm{Pred},\ \Delta\mathrm{GT})$')
    ax.axhline(0, color='black', linewidth=0.5)
    ax.set_ylim(top=overall_top + 0.18)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(frameon=False, loc='upper right', ncol=3)


# --------------------------------------------------------------------------
# Panel B — 1-D track demonstration of what "cell-type differential PCC"
# measures, at one strongly cell-type-specific locus (chosen from the
# top-ranked entry of representative_loci.tsv).
# --------------------------------------------------------------------------
DEMO_LOCUS = {
    # H3K4me3, GM12878 vs K562, chr10:73.35 Mb. Selected for the
    # demonstration because (i) one dominant cell-specific H3K4me3 peak
    # plus several smaller peaks fit cleanly in a ±50 kb window;
    # (ii) Evo2 is a strong per-cell predictor (corr with own GT = 0.95
    # for GM12878, 0.77 for K562) but produces ~identical predictions
    # across cells; (iii) ΔPred_Evo2 vs ΔGT correlation is essentially
    # zero (r ≈ +0.03), giving a clean "Evo2 cannot differentiate cell"
    # message; (iv) Evo2HiC tracks ΔGT well (r ≈ 0.85). Alternative
    # multi-peak loci that exhibit the same failure mode are documented
    # in `Figures/supplementary_12_panelB_candidates.{md,png}`.
    'track':       'H3K4me3',
    'track_idx':   4,
    'cell_A':      'GM12878',
    'cell_B':      'K562',
    'chrom':       10,
    'centre':      73_350_000,
    'half_window': 50_000,
}

EVO2HIC_NPY_DIR = EPI_EVO2HIC_DIR
# Evo2 baseline is per-cell-type: 3 separate models, each with its own step
EVO2_NPY_DIR = {cell: EPI_EVO2_DIR(cell) for cell in ('GM12878', 'H1ESC', 'K562')}

RESOLUTION = 2000


def _load_locus_signals():
    """Returns coords + dicts {cell -> 1-D array} for GT, Pred_Evo2HiC, Pred_Evo2
    over the locus window (±half_window around centre)."""
    from dataset.track_loader import Track_Loader      # local import
    from config import hic2tarck_dir as HIC2TRACK_DIR
    import os

    locus  = DEMO_LOCUS
    chrom  = locus['chrom']
    centre = locus['centre']
    half   = locus['half_window']
    ti     = locus['track_idx']

    bin_centre = centre // RESOLUTION
    bin_half   = half   // RESOLUTION
    bin_start  = bin_centre - bin_half
    bin_end    = bin_centre + bin_half + 1
    n_bins     = bin_end - bin_start

    cell_A, cell_B = locus['cell_A'], locus['cell_B']

    # GT
    gt = {}
    for cell in (cell_A, cell_B):
        loader = Track_Loader(os.path.join(HIC2TRACK_DIR, cell), RESOLUTION)
        full = loader.get(chrom, 0, ((centre + half) // RESOLUTION + 1) * RESOLUTION, 0)
        gt[cell] = full[ti, bin_start:bin_end]

    # Evo2HiC predictions (single model, cell-specific Hi-C input)
    pred_e2h = {}
    for cell in (cell_A, cell_B):
        full = np.load(EVO2HIC_NPY_DIR / cell / f'{chrom}.npy')
        pred_e2h[cell] = full[ti, bin_start:bin_end]

    # Evo2 predictions (per-cell DNA-only model, no Hi-C → cannot vary across cells)
    pred_evo2 = {}
    for cell in (cell_A, cell_B):
        full = np.load(EVO2_NPY_DIR[cell] / f'{chrom}.npy')
        pred_evo2[cell] = full[ti, bin_start:bin_end]

    coords = (bin_start + np.arange(n_bins)) * RESOLUTION
    return coords, gt, pred_e2h, pred_evo2


def panel_demo(axes_quad):
    """axes_quad: list of 4 axes (GT / Evo2HiC / Evo2 / Δ comparison)."""
    from scipy.stats import pearsonr

    coords, gt, pred_e2h, pred_evo2 = _load_locus_signals()
    A, B = DEMO_LOCUS['cell_A'], DEMO_LOCUS['cell_B']
    chrom = DEMO_LOCUS['chrom']

    coord_mb = coords / 1e6
    color_A     = '#d62728'                   # red — cell A
    color_B     = '#1f77b4'                   # blue — cell B
    color_dgt   = '#000000'                   # black for ΔGT (anchor)
    color_de2h  = '#b2182b'                   # Nature-style red    — ΔEvo2HiC
    color_devo2 = '#2166ac'                   # Nature-style blue   — ΔEvo2

    ax_gt, ax_e2h, ax_evo2, ax_d = axes_quad

    def _draw_pair(ax, sig_A, sig_B, ylabel, legend_A, legend_B):
        ax.fill_between(coord_mb, sig_A, color=color_A, alpha=0.45, label=legend_A)
        ax.fill_between(coord_mb, sig_B, color=color_B, alpha=0.45, label=legend_B)
        ax.set_ylabel(ylabel)
        ax.legend(frameon=False, loc='upper right', fontsize=6, ncol=1,
                  handletextpad=0.4, labelspacing=0.3)

    # legend now shows just the cell name; the ylabel ("GT signal" /
    # "Evo2HiC signal" / "Evo2 signal") already conveys what kind of signal
    # is plotted, so we drop the redundant "GT"/"Pred" prefix.
    _draw_pair(ax_gt,   gt[A],        gt[B],   'GT\nsignal',      A, B)
    _draw_pair(ax_e2h,  pred_e2h[A],  pred_e2h[B], 'Evo2HiC\nsignal', A, B)
    _draw_pair(ax_evo2, pred_evo2[A], pred_evo2[B], 'Evo2\nsignal',    A, B)

    # --- bottom: Δ comparison ---
    dgt   = gt[A]        - gt[B]
    de2h  = pred_e2h[A]  - pred_e2h[B]
    devo2 = pred_evo2[A] - pred_evo2[B]
    r_e2h, _ = pearsonr(dgt, de2h)
    r_evo2, _ = pearsonr(dgt, devo2)
    ax_d.plot(coord_mb, dgt,   color=color_dgt,   lw=1.4, label='\u0394GT')
    ax_d.plot(coord_mb, de2h,  color=color_de2h,  lw=1.1, label=f'\u0394Evo2HiC (r={r_e2h:.2f})')
    ax_d.plot(coord_mb, devo2, color=color_devo2, lw=1.1, label=f'\u0394Evo2 (r={r_evo2:.2f})')
    ax_d.set_ylabel(r'$\Delta$' + 'signal')
    # one label per row (3 rows total) — keeps the legend column narrow
    # so it doesn't overlap the ΔGT / ΔEvo2HiC peaks.
    ax_d.legend(frameon=False, loc='upper right', fontsize=6, ncol=1,
                handletextpad=0.4, labelspacing=0.3)

    # cosmetic
    for ax in axes_quad:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    for ax in axes_quad[:-1]:
        ax.tick_params(labelbottom=False)
    # explicit Mb tick labels on the bottom axis
    import matplotlib.ticker as mticker
    ax_d.xaxis.set_major_locator(mticker.MaxNLocator(nbins=5))
    ax_d.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f'{v:.2f} Mb'))
    ax_d.tick_params(labelbottom=True)

    ax_gt.set_title(
        f'{DEMO_LOCUS["track"]}  at  {A} vs {B}  '
        f'(chr{chrom}:{(coords[0]/1e6):.2f}–{(coords[-1]/1e6):.2f} Mb)',
        fontsize=8,
    )


# --------------------------------------------------------------------------
def main():
    cm, gap, sp, wt = load_data()

    n_a_cols = 1 + len(MODELS)                         # GT + 4 models = 5
    n_a_rows = len(REP_TRACKS)                          # 2

    fig = plt.figure(figsize=(FIG_WIDTH_INCH, 8.4))
    outer = fig.add_gridspec(
        nrows=2, ncols=1,
        height_ratios=[1.10, 1.0],
        hspace=0.32,
        top=0.96, bottom=0.07, left=0.09, right=0.93,
    )

    # ----- Panel A: heatmap grid (top) -----
    a_gs = outer[0].subgridspec(
        nrows=n_a_rows, ncols=n_a_cols, hspace=0.18, wspace=0.02
    )
    a_axes = [[fig.add_subplot(a_gs[r, c]) for c in range(n_a_cols)]
              for r in range(n_a_rows)]
    panel_a(a_axes, cm)
    a_axes[0][0].text(-0.55, 1.10, 'a', transform=a_axes[0][0].transAxes,
                      fontsize=PANEL_LABEL_FONTSIZE, fontweight='bold', ha='left')

    # ----- bottom row: Panel B (demo, left) + Panel C (bars, right) -----
    bottom_gs = outer[1].subgridspec(
        nrows=1, ncols=2, width_ratios=[1.0, 1.05], wspace=0.32,
    )

    # Panel B: 4 stacked sub-panels (GT / Evo2HiC / Evo2 / Δ) sharing x-axis
    b_gs = bottom_gs[0, 0].subgridspec(
        nrows=4, ncols=1, hspace=0.10, height_ratios=[1, 1, 1, 1.2],
    )
    b_axes = []
    for r in range(4):
        if r == 0:
            ax = fig.add_subplot(b_gs[r, 0])
        else:
            ax = fig.add_subplot(b_gs[r, 0], sharex=b_axes[0])
        b_axes.append(ax)
    panel_demo(b_axes)
    b_axes[0].text(-0.18, 1.10, 'b', transform=b_axes[0].transAxes,
                   fontsize=PANEL_LABEL_FONTSIZE, fontweight='bold', ha='left')

    # Panel C: bar plot
    cax = fig.add_subplot(bottom_gs[0, 1])
    panel_b(cax, sp, wt)                                # function name kept for legacy
    cax.text(-0.10, 1.04, 'c', transform=cax.transAxes,
             fontsize=PANEL_LABEL_FONTSIZE, fontweight='bold')

    fig.savefig(OUT_PDF, bbox_inches='tight')
    fig.savefig(str(OUT_PDF).replace('.pdf', '.png'), dpi=200, bbox_inches='tight')
    print(f'Saved {OUT_PDF}')
    print(f'Saved {str(OUT_PDF).replace(".pdf", ".png")}')


if __name__ == '__main__':
    main()
