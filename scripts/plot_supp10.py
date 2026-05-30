"""
Supplementary Figure 10: Leakage-adjusted Borzoi vs Evo2HiC comparison.

Background
----------
Borzoi's official train/valid/test split (per its README) assigns genomic
windows to 8 folds, with fold3 = test and fold4 = validation; the remaining
folds (0, 1, 2, 5, 6, 7) form the training set. All 4 Borzoi replicates share
this split. Because our CDNA1d test set (chr9, chr10) overlaps Borzoi's
training folds, Borzoi's headline numbers on our full test set are leaked.

This script restricts the evaluation to chr9/10 bins that fall **exclusively**
inside Borzoi's test-fold (fold3) windows — i.e. bins with no overlap against
any train-fold window. On these "clean" bins both Evo2HiC (main) and Borzoi
are evaluated on sequence they never saw during training.

Why K562 is excluded
--------------------
K562 is held out as the test cell type in our epig_predict training data
(hic2track/index.tsv: split='test'), so Evo2HiC has never seen any K562 Hi-C
or ChIP-seq during training — any K562 evaluation is zero-shot cross-cell-type
transfer. Borzoi, by contrast, is trained on 424+ K562 tracks (ChIP, DNase,
RNA-seq, etc.) across all non-fold3 windows. Comparing Evo2HiC (zero-shot on
K562) against Borzoi (in-distribution on K562) would conflate cell-type
generalization gap with model capability. K562 is therefore analyzed separately
(see per-cell breakdown in Supplementary Table / main Figure 4). This script
restricts the Borzoi head-to-head to GM12878 and H1ESC, both of which are
in-distribution training cell types for Evo2HiC.

Statistical procedure
---------------------
For every (cell × track × metric) we compare Evo2HiC vs Borzoi:
  1. Directional gate: only run the test when Evo2HiC's aggregate
     metric (concatenated chr9+chr10 PCC or SPC) is strictly higher
     than Borzoi's; otherwise mark "n.s. (no improvement)".
  2. Per-chromosome Williams (one-sided) test for the overlapping
     correlations sharing the ground-truth variable, with the
     pre-specified alternative r(Evo2HiC, GT) > r(Borzoi, GT). For
     PCC this is Williams on the raw vectors; for SPC the vectors
     are rank-transformed within each chromosome first, so Williams
     operates on Pearson-of-ranks (= Spearman).
  3. Fisher's combined probability test pools per-chromosome one-sided
     p-values across chr9 and chr10 (X² = -2 Σ ln p_k, df = 2K).
NOTE: bins are spatially autocorrelated; Williams' SE assumes i.i.d.,
so reported p-values are anti-conservative. The directional gate is
selective inference and is reported transparently in the caption.

Outputs
-------
Figures/supplementary_10.pdf : 2x2 barplot — rows = metric (PCC, SPC),
                               columns = cell line (GM12878, H1ESC), 5 tracks
                               per panel. Panels labelled a-d. Stars come
                               from the per-chr Williams + Fisher procedure
                               described above.
stdout                         : PCC, SPC, test type & p-values per (cell, track).
"""

import sys
import argparse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats


# ---------- paths / constants ----------
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paths import (                                             # noqa: E402
    REPO, ensure_out_dir, add_repo_to_syspath,
    CKPT_ROOT as CKPT_DIR,
    BORZOI_DIR,
)
add_repo_to_syspath()

from dataset.track_loader import Track_Loader  # noqa: E402
from config import hic2tarck_dir               # noqa: E402

OUT_PDF = ensure_out_dir() / 'supplementary_10.pdf'

RESOLUTION = 2000
# K562 excluded: zero-shot cross-cell-type for Evo2HiC vs in-distribution for
# Borzoi — see module docstring.
CELLS = ['GM12878', 'H1ESC']
TRACKS = ['DNase', 'CTCF', 'H3K27ac', 'H3K27me3', 'H3K4me3']
TEST_CHRS = (9, 10)

BORZOI_TEST_FOLD = 'fold3'
BORZOI_VALID_FOLD = 'fold4'  # excluded too (Borzoi used it for model selection)
BORZOI_TRAIN_FOLDS = ('fold0', 'fold1', 'fold2', 'fold5', 'fold6', 'fold7')

PANEL_LABELS = ['a', 'b', 'c', 'd']
METRICS = ('PCC', 'SPC')

SEQBED_URL = 'https://storage.googleapis.com/seqnn-share/borzoi/hg38/sequences.bed'
SEQBED_CACHE = Path.home() / '.cache' / 'borzoi' / 'sequences.bed'


# ---------- helpers ----------
def _p_to_stars(p):
    return '****' if p < 1e-4 else '***' if p < 1e-3 else '**' if p < 1e-2 \
        else '*' if p < 5e-2 else 'n.s.'


def fetch_borzoi_folds():
    SEQBED_CACHE.parent.mkdir(parents=True, exist_ok=True)
    if not SEQBED_CACHE.exists() or SEQBED_CACHE.stat().st_size == 0:
        print(f'Fetching {SEQBED_URL} ...')
        urllib.request.urlretrieve(SEQBED_URL, SEQBED_CACHE)
    df = pd.read_csv(SEQBED_CACHE, sep='\t', header=None,
                     names=['chr', 'start', 'end', 'fold'])
    return df


def compute_clean_test_bin_mask(chrom, chrom_size, folds_df, resolution):
    """
    Return bool array of shape (n_bins,).
    True iff bin is covered by >=1 Borzoi test-fold window AND zero train-fold
    windows. Validation fold windows are also excluded from "clean test".
    """
    n_bins = int(np.ceil(chrom_size / resolution))
    in_test = np.zeros(n_bins, dtype=bool)
    in_train_or_valid = np.zeros(n_bins, dtype=bool)

    sub = folds_df[folds_df.chr == f'chr{chrom}']
    for _, row in sub.iterrows():
        b_lo = int(row.start) // resolution
        b_hi = min(int(np.ceil(int(row.end) / resolution)), n_bins)
        if b_hi <= b_lo:
            continue
        if row.fold == BORZOI_TEST_FOLD:
            in_test[b_lo:b_hi] = True
        else:
            in_train_or_valid[b_lo:b_hi] = True

    return in_test & ~in_train_or_valid


# ---------- load predictions + GT ----------
def load_npy(method, cell, chrom):
    if method == 'Evo2HiC':
        p = CKPT_DIR / 'epi_prediction/model/track' / cell / f'{chrom}.npy'
    elif method == 'Borzoi':
        p = BORZOI_DIR / cell / f'{chrom}.npy'
    else:
        raise ValueError(method)
    return np.load(p)  # (5, n_bins)


def collect(cell, folds_df):
    """Return per-track dict of per-chromosome lists of arrays
    (pred_Evo2HiC, pred_Borzoi, gt), masked to clean Borzoi-test bins.

    Returned shape (per cell):
        out[tname]['e2h']  -> [arr_chr9, arr_chr10]   (lists, NOT concatenated)
        out[tname]['brz']  -> [arr_chr9, arr_chr10]
        out[tname]['gt']   -> [arr_chr9, arr_chr10]

    Aggregate (concatenated) arrays are produced on demand by the consumer
    via np.concatenate; per-chromosome lists are kept so the per-chr
    Williams test can operate on each chromosome independently."""
    tl = Track_Loader(f'{hic2tarck_dir}/{cell}', RESOLUTION)
    out = {t: {'e2h': [], 'brz': [], 'gt': []} for t in TRACKS}

    for ch in TEST_CHRS:
        chrom_size = tl.chr_lens[f'chr{ch}']
        gt = tl.get(ch, 0, (chrom_size // RESOLUTION + 1) * RESOLUTION, 0)
        pred_e2h = load_npy('Evo2HiC', cell, ch)
        pred_brz = load_npy('Borzoi', cell, ch)

        n = min(gt.shape[1], pred_e2h.shape[1], pred_brz.shape[1])
        mask = compute_clean_test_bin_mask(ch, chrom_size, folds_df, RESOLUTION)[:n]

        for ti, tname in enumerate(TRACKS):
            # Allow track-level NaN (e.g. K562 H3K27ac has all-NaN Borzoi); drop
            # NaN bins from the paired set.
            e = pred_e2h[ti, :n][mask]
            b = pred_brz[ti, :n][mask]
            g = gt[ti, :n][mask]
            valid = np.isfinite(e) & np.isfinite(b) & np.isfinite(g)
            out[tname]['e2h'].append(e[valid])
            out[tname]['brz'].append(b[valid])
            out[tname]['gt'].append(g[valid])

    return out


# ---------- PCC / SPC + per-chr Williams + Fisher ----------
def _se_williams_overlap(r_ab, r_ac, r_bc, n):
    """SE of (r_ab - r_ac) for overlapping correlations sharing variable a
    (Steiger 1980 / Williams). Returns None if degenerate."""
    if n <= 3:
        return None
    detR = 1 + 2 * r_ab * r_ac * r_bc - (r_ab ** 2 + r_ac ** 2 + r_bc ** 2)
    if detR <= 0:
        return None
    r_bar = 0.5 * (r_ab + r_ac)
    num = 2 * ((n - 1) / (n - 3)) * detR + (r_bar ** 2) * (1 - r_bc) ** 3
    den = (n - 1) * (1 + r_bc)
    val = num / den
    if val <= 0:
        return None
    return float(np.sqrt(val))


def _williams_block(a, b, t):
    """One-sided Williams test on one block (pre-specified alternative:
    r(a,t) > r(b,t)). Returns p = P(Z >= observed) under H0: r_at == r_bt."""
    n = len(t)
    if n <= 3:
        return None
    r_at = stats.pearsonr(a, t)[0]
    r_bt = stats.pearsonr(b, t)[0]
    r_ab = stats.pearsonr(a, b)[0]
    if not all(np.isfinite([r_at, r_bt, r_ab])):
        return None
    se = _se_williams_overlap(r_at, r_bt, r_ab, n)
    if se is None or se == 0:
        return None
    Z = (r_at - r_bt) / se
    return float(stats.norm.sf(Z))


def _fisher_combine(pvals):
    """Fisher's combined probability test on per-block one-sided p-values.
    p-values are clipped to [1e-300, 1.0] before the log so blocks whose
    p underflows norm.sf to 0 still carry their evidence."""
    pv = np.asarray([p for p in pvals
                     if p is not None and np.isfinite(p)])
    if len(pv) == 0:
        return None
    pv = np.clip(pv, 1e-300, 1.0)
    X2 = -2.0 * np.log(pv).sum()
    return float(stats.chi2.sf(X2, df=2 * len(pv)))


def _per_chr_williams_fisher(blocks_a, blocks_b, blocks_t, rank_transform=False):
    """Per-chromosome Williams (one-sided) + Fisher combine.
    blocks_* are lists of per-chromosome 1D arrays (already finite-masked).
    rank_transform=True replaces (a, b, t) with their per-chr ranks so the
    underlying Pearson correlation becomes Spearman."""
    per_chr_p = []
    for a, b, t in zip(blocks_a, blocks_b, blocks_t):
        n = min(len(a), len(b), len(t))
        if n <= 3:
            continue
        a_, b_, t_ = a[:n], b[:n], t[:n]
        if rank_transform:
            a_ = stats.rankdata(a_)
            b_ = stats.rankdata(b_)
            t_ = stats.rankdata(t_)
        p = _williams_block(a_, b_, t_)
        if p is not None:
            per_chr_p.append(p)
    if not per_chr_p:
        return None, 0
    return _fisher_combine(per_chr_p), len(per_chr_p)


def compute_metrics(collected):
    """Returns (pccs, spcs, pvals_pcc, pvals_spc, ns).

    pccs/spcs[cell][track] = {'e2h':, 'brz':}     # aggregate (concat chr9+chr10)
    pvals_pcc[cell][track]: per-chromosome Williams (one-sided, alt: r(E2H, gt) >
        r(Borzoi, gt)) combined with Fisher's method, gated on Evo2HiC beating
        Borzoi on aggregate PCC.
    pvals_spc[cell][track]: same as PCC but on per-chromosome rank-transformed
        vectors, so the underlying correlation is Spearman.
    """
    pccs, spcs, pvals_pcc, pvals_spc, ns = {}, {}, {}, {}, {}
    for cell, tracks_d in collected.items():
        pccs[cell], spcs[cell] = {}, {}
        pvals_pcc[cell], pvals_spc[cell] = {}, {}
        ns[cell] = {}
        for tname in TRACKS:
            e_chrs = tracks_d[tname]['e2h']      # [chr9, chr10]
            b_chrs = tracks_d[tname]['brz']
            g_chrs = tracks_d[tname]['gt']

            # Concatenated aggregates for the bar values + directional gate.
            e = np.concatenate(e_chrs)
            b = np.concatenate(b_chrs)
            g = np.concatenate(g_chrs)
            n = len(g)
            ns[cell][tname] = n
            if n < 10:
                pccs[cell][tname] = {'e2h': np.nan, 'brz': np.nan}
                spcs[cell][tname] = {'e2h': np.nan, 'brz': np.nan}
                pvals_pcc[cell][tname] = None
                pvals_spc[cell][tname] = None
                continue

            pcc_e = stats.pearsonr(e, g)[0] if np.std(e) > 0 else np.nan
            pcc_b = stats.pearsonr(b, g)[0] if np.std(b) > 0 else np.nan
            spc_e = stats.spearmanr(e, g)[0] if np.std(e) > 0 else np.nan
            spc_b = stats.spearmanr(b, g)[0] if np.std(b) > 0 else np.nan
            pccs[cell][tname] = {'e2h': pcc_e, 'brz': pcc_b}
            spcs[cell][tname] = {'e2h': spc_e, 'brz': spc_b}

            # PCC significance: per-chr Williams (Pearson) + Fisher
            if (np.isfinite(pcc_e) and np.isfinite(pcc_b) and pcc_e > pcc_b):
                p_pcc, _ = _per_chr_williams_fisher(e_chrs, b_chrs, g_chrs,
                                                    rank_transform=False)
                pvals_pcc[cell][tname] = p_pcc
            else:
                pvals_pcc[cell][tname] = None

            # SPC significance: per-chr Williams on ranks (= Spearman) + Fisher
            if (np.isfinite(spc_e) and np.isfinite(spc_b) and spc_e > spc_b):
                p_spc, _ = _per_chr_williams_fisher(e_chrs, b_chrs, g_chrs,
                                                    rank_transform=True)
                pvals_spc[cell][tname] = p_spc
            else:
                pvals_spc[cell][tname] = None

    return pccs, spcs, pvals_pcc, pvals_spc, ns


# ---------- plot ----------
def plot_barplot(pccs, spcs, pvals_pcc, pvals_spc, ns, out_pdf):
    colors_two = ['#fb8072', '#80b1d3']  # Evo2HiC, Borzoi
    plt.rcParams.update({'font.size': 9})

    n_rows = len(METRICS)   # PCC / SPC
    n_cols = len(CELLS)     # GM12878 / H1ESC
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(9.2, 5.6),
                             sharex=True)

    x = np.arange(len(TRACKS))
    width = 0.38

    # Row -> (metric name, per-cell values dict, pval dict)
    row_data = {
        'PCC': (pccs, pvals_pcc),
        'SPC': (spcs, pvals_spc),
    }

    panel_idx = 0
    for row_idx, metric in enumerate(METRICS):
        vals_dict, pval_dict = row_data[metric]

        for col_idx, cell in enumerate(CELLS):
            ax = axes[row_idx, col_idx]

            vals_e = np.array([vals_dict[cell][t]['e2h'] for t in TRACKS])
            vals_b = np.array([vals_dict[cell][t]['brz'] for t in TRACKS])

            ax.bar(x - width / 2, vals_e, width, label='Evo2HiC',
                   color=colors_two[0], alpha=0.9)
            ax.bar(x + width / 2, vals_b, width, label='Borzoi',
                   color=colors_two[1], alpha=0.9)

            all_vals = np.concatenate([vals_e, vals_b])
            all_vals = all_vals[np.isfinite(all_vals)]
            ymax = float(all_vals.max()) * 1.25 if len(all_vals) else 1.0
            ax.set_ylim(0, ymax)

            # Cell title only on top row; per-cell n is shown once per column
            if row_idx == 0:
                n_first = ns[cell][TRACKS[0]]
                ax.set_title(f'{cell}  (n = {n_first} clean test bins)',
                             fontsize=10)

            ax.set_ylabel(metric)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.tick_params(axis='y', which='both', length=2)
            ax.tick_params(axis='x', which='both', length=0)

            # Panel label (a-d)
            ax.text(-0.14, 1.05, PANEL_LABELS[panel_idx],
                    transform=ax.transAxes,
                    fontsize=12, fontweight='bold', va='bottom', ha='right')
            panel_idx += 1

    # X-axis labels on bottom row only
    for col_idx in range(n_cols):
        axes[-1, col_idx].set_xticks(x)
        axes[-1, col_idx].set_xticklabels(TRACKS, rotation=30, ha='right')

    # Single figure-level legend, top-right, aligned with first subplot title.
    handles, labels_list = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles, labels_list,
        loc='upper right',
        bbox_to_anchor=(0.995, 0.995),
        ncol=len(labels_list),
        frameon=False,
        fontsize=9,
        columnspacing=1.2,
        handletextpad=0.5,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, dpi=300, bbox_inches='tight', transparent=True)
    plt.close(fig)
    print(f'\nSaved {out_pdf}')


# ---------- main ----------
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-pdf', type=Path, default=OUT_PDF)
    args = parser.parse_args()

    folds_df = fetch_borzoi_folds()
    print(f'Borzoi fold labels: {sorted(folds_df.fold.unique())}')
    print(f'Treating  test   = {BORZOI_TEST_FOLD}')
    print(f'Treating  valid  = {BORZOI_VALID_FOLD}  (excluded from clean test)')
    print(f'Treating  train  = {BORZOI_TRAIN_FOLDS}')

    # Report mask coverage
    for ch in TEST_CHRS:
        tl = Track_Loader(f'{hic2tarck_dir}/{CELLS[0]}', RESOLUTION)
        chrom_size = tl.chr_lens[f'chr{ch}']
        mask = compute_clean_test_bin_mask(ch, chrom_size, folds_df, RESOLUTION)
        print(f'  chr{ch}: {mask.sum()} / {len(mask)} bins ({100*mask.mean():.1f}%) '
              f'are clean Borzoi test (in fold3, not in any train/valid fold)')

    # Collect per-cell
    collected = {c: collect(c, folds_df) for c in CELLS}
    pccs, spcs, pvals_pcc, pvals_spc, ns = compute_metrics(collected)

    # Print per-metric summaries
    def _print_metric(metric_name, vals, pvals):
        print(f"\n=== {metric_name} ===")
        print(f"{'Cell':8s}  {'Track':10s}  {'n':>7s}  "
              f"{'E2H':>8s}  {'Borzoi':>8s}  {'Gap':>8s}  "
              f"{'test':>28s}  {'p-value':>12s}  stars")
        print('-' * 110)
        for cell in CELLS:
            for tname in TRACKS:
                n = ns[cell][tname]
                me = vals[cell][tname]['e2h']
                mb = vals[cell][tname]['brz']
                p = pvals[cell][tname]
                gap_str = (f'{(me - mb) / mb * 100:+7.2f}%'
                           if np.isfinite(mb) and mb != 0 else '--')
                me_s = f'{me:.4f}' if np.isfinite(me) else 'NaN'
                mb_s = f'{mb:.4f}' if np.isfinite(mb) else 'NaN'
                if p is None:
                    if not (np.isfinite(me) and np.isfinite(mb)):
                        test_name, p_s, stars = '--', '--', '(insufficient data)'
                    else:
                        test_name = 'Williams (skipped: ΔPCC ≤ 0)'
                        p_s = '--'
                        stars = 'n.s. (no improvement)'
                else:
                    if metric_name == 'PCC':
                        test_name = 'Williams(1-sided, Pearson) + Fisher'
                    else:
                        test_name = 'Williams(1-sided, Spearman) + Fisher'
                    p_s = f'{p:.3e}'
                    stars = _p_to_stars(p)
                print(f'{cell:8s}  {tname:10s}  {n:>7d}  {me_s:>8s}  {mb_s:>8s}  '
                      f'{gap_str:>8s}  {test_name:>28s}  {p_s:>12s}  {stars}')

    _print_metric('PCC', pccs, pvals_pcc)
    _print_metric('SPC', spcs, pvals_spc)

    # Per-cell average improvement (macro-average across tracks)
    print('\n=== Per-cell average (macro-mean over tracks) ===')
    print(f"{'Cell':8s}  {'Metric':6s}  {'E2H mean':>9s}  {'Borzoi mean':>11s}  "
          f"{'Abs gap':>8s}  {'Rel gap':>8s}")
    print('-' * 62)
    for cell in CELLS:
        for metric_name, vals in [('PCC', pccs), ('SPC', spcs)]:
            e_arr = np.array([vals[cell][t]['e2h'] for t in TRACKS], dtype=float)
            b_arr = np.array([vals[cell][t]['brz'] for t in TRACKS], dtype=float)
            mask = np.isfinite(e_arr) & np.isfinite(b_arr)
            if mask.sum() == 0:
                print(f'{cell:8s}  {metric_name:6s}  {"NaN":>9s}  {"NaN":>11s}  '
                      f'{"--":>8s}  {"--":>8s}')
                continue
            e_mean = float(e_arr[mask].mean())
            b_mean = float(b_arr[mask].mean())
            abs_gap = e_mean - b_mean
            rel_gap = (abs_gap / b_mean * 100) if b_mean != 0 else np.nan
            rel_s = f'{rel_gap:+7.2f}%' if np.isfinite(rel_gap) else '--'
            print(f'{cell:8s}  {metric_name:6s}  {e_mean:>9.4f}  {b_mean:>11.4f}  '
                  f'{abs_gap:>+8.4f}  {rel_s:>8s}')

    # Overall (across both cells, all tracks) — mean first, then rel gap
    print('\n=== Overall across cells (mean over all cell x track entries) ===')
    print(f"{'Metric':6s}  {'E2H mean':>9s}  {'Borzoi mean':>11s}  "
          f"{'Abs gap':>8s}  {'Rel gap':>8s}")
    print('-' * 54)
    for metric_name, vals in [('PCC', pccs), ('SPC', spcs)]:
        e_all, b_all = [], []
        for cell in CELLS:
            for t in TRACKS:
                me, mb = vals[cell][t]['e2h'], vals[cell][t]['brz']
                if np.isfinite(me) and np.isfinite(mb):
                    e_all.append(me)
                    b_all.append(mb)
        e_mean = float(np.mean(e_all))
        b_mean = float(np.mean(b_all))
        abs_gap = e_mean - b_mean
        rel_gap = (abs_gap / b_mean * 100) if b_mean != 0 else np.nan
        print(f'{metric_name:6s}  {e_mean:>9.4f}  {b_mean:>11.4f}  '
              f'{abs_gap:>+8.4f}  {rel_gap:>+7.2f}%')

    # Per-entry relative improvement (paper-style):
    # imp_i = Evo2HiC_i / Borzoi_i - 1, then arithmetic mean over all
    # GM12878 + H1ESC × 5 tracks = 10 entries. Mirrors how Fig.4 reports
    # +34.7% over HiC-only and +26.2% over Evo2 (plot_Fig4_Epi.ipynb cell-5).
    print('\n=== Paper-style avg per-entry relative improvement '
          '(Evo2HiC vs Borzoi) ===')
    print(f"{'Metric':6s}  {'n entries':>9s}  {'mean(E2H/Brz - 1)':>20s}")
    print('-' * 42)
    for metric_name, vals in [('PCC', pccs), ('SPC', spcs)]:
        imps = []
        per_entry = []
        for cell in CELLS:
            for t in TRACKS:
                me, mb = vals[cell][t]['e2h'], vals[cell][t]['brz']
                if np.isfinite(me) and np.isfinite(mb) and mb != 0:
                    imp = me / mb - 1
                    imps.append(imp)
                    per_entry.append((cell, t, imp))
        rel = float(np.mean(imps)) * 100 if imps else np.nan
        print(f'{metric_name:6s}  {len(imps):>9d}  {rel:>+19.2f}%')
        for cell, t, imp in per_entry:
            print(f'    {cell:8s} {t:10s}  {imp*100:+7.2f}%')

    plot_barplot(pccs, spcs, pvals_pcc, pvals_spc, ns, args.out_pdf)


if __name__ == '__main__':
    main()