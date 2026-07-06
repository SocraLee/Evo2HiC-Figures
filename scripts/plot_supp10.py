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

Two sequence-only baselines are compared, each on its OWN strict TEST fold
(per memory/feedback_strict_test_fold — never evaluate a model on its
validation fold):
  - Borzoi-replicate-0     : strict TEST = fold3  → non-empty only on chr9
  - AlphaGenome FOLD_1     : strict TEST = fold4  → non-empty only on chr10
AlphaGenome FOLD_1's training set is byte-identical to Borzoi-rep-0's
(train = folds 0,1,2,5,6,7; verified against the fold_intervals API), so the
two share the same held-out region structure but publish opposite val/test
labels. We honour each model's published TEST designation, giving two
chromosome-complementary head-to-heads.

Outputs
-------
Figures/supplementary_10.pdf : 2×2 barplot (PCC only) —
                               row 1 (a,b): Evo2HiC vs Borzoi on chr9 fold3-strict,
                               row 2 (c,d): Evo2HiC vs AlphaGenome on chr10 fold4-strict,
                               columns = cell line (GM12878, H1ESC), 5 tracks per panel.
stdout                         : per-(cell, track) PCC/SPC, per-chr Williams +
                               Fisher p-values, and paper-convention per-entry
                               relative improvement (Evo2HiC vs each baseline).
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
    ensure_out_dir, add_repo_to_syspath,
    CKPT_ROOT as CKPT_DIR,
    BORZOI_DIR, AG_DIR,
)
add_repo_to_syspath()

from dataset.track_loader import Track_Loader  # noqa: E402
from config import hic2tarck_dir               # noqa: E402

OUT_PDF = ensure_out_dir() / 'supplementary_10.pdf'

RESOLUTION = 2000
# K562 excluded for both head-to-heads: it is held out from Evo2HiC's training
# cell list (zero-shot cross-cell transfer for Evo2HiC), while both Borzoi and
# AlphaGenome FOLD_1 saw K562 tracks during their region-level training, so
# K562 conflates cell-type generalization with sequence generalization.
CELLS = ['GM12878', 'H1ESC']
TRACKS = ['DNase', 'CTCF', 'H3K27ac', 'H3K27me3', 'H3K4me3']
TEST_CHRS = (9, 10)

# Borzoi-replicate-0 official designation
BORZOI_TEST_FOLD  = 'fold3'   # strict TEST set
BORZOI_VALID_FOLD = 'fold4'

# AlphaGenome FOLD_1 official designation (val/test swapped vs Borzoi but
# train set is identical — verified byte-for-byte against fold_intervals API).
AG_TEST_FOLD  = 'fold4'       # strict TEST set
AG_VALID_FOLD = 'fold3'

# 2 rows × 2 cols = 4 panels: a–b (vs Borzoi on chr9 fold3-strict),
# c–d (vs AlphaGenome on chr10 fold4-strict). PCC only — SPC dropped for clarity.
PANEL_LABELS = ['a', 'b', 'c', 'd']
METRICS = ('PCC',)

BASELINES = [
    {'key': 'borzoi',
     'name': 'Borzoi',
     'pred_dir': BORZOI_DIR,
     'test_fold': BORZOI_TEST_FOLD,
     'color': '#80b1d3'},   # supp11 PALETTE[2] — blue
    {'key': 'alphagenome',
     'name': 'AlphaGenome',
     'pred_dir': AG_DIR,
     'test_fold': AG_TEST_FOLD,
     'color': '#bebada'},   # supp11 PALETTE[1] — purple
]

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


def compute_strict_test_mask(chrom, chrom_size, folds_df, test_fold, resolution):
    """
    Return bool array of shape (n_bins,). True iff a 2 kb bin is covered by
    >=1 window assigned to `test_fold` AND has zero overlap with any other
    fold. Used to derive each model's strict TEST mask without leakage from
    train or validation folds. `test_fold` must be one of {fold0..fold7}.
    """
    n_bins = int(np.ceil(chrom_size / resolution))
    in_test = np.zeros(n_bins, dtype=bool)
    in_other = np.zeros(n_bins, dtype=bool)

    sub = folds_df[folds_df.chr == f'chr{chrom}']
    for _, row in sub.iterrows():
        b_lo = int(row.start) // resolution
        b_hi = min(int(np.ceil(int(row.end) / resolution)), n_bins)
        if b_hi <= b_lo:
            continue
        if row.fold == test_fold:
            in_test[b_lo:b_hi] = True
        else:
            in_other[b_lo:b_hi] = True

    return in_test & ~in_other


# Back-compat alias for the original entry point (used by main()'s coverage report)
def compute_clean_test_bin_mask(chrom, chrom_size, folds_df, resolution):
    return compute_strict_test_mask(chrom, chrom_size, folds_df,
                                    BORZOI_TEST_FOLD, resolution)


# ---------- load predictions + GT ----------
def load_npy(method, cell, chrom, baseline_dir=None):
    if method == 'Evo2HiC':
        p = CKPT_DIR / 'epi_prediction/model/track' / cell / f'{chrom}.npy'
    elif method == 'baseline':
        p = baseline_dir / cell / f'{chrom}.npy'
    else:
        raise ValueError(method)
    return np.load(p)  # (5, n_bins)


def collect_for_baseline(cell, folds_df, baseline_dir, test_fold):
    """Return per-track dict of per-chromosome lists of arrays masked to
    `test_fold`-strict bins (i.e. in test_fold AND not overlapping any other
    fold). Per-chromosome lists are kept so per-chr Williams + Fisher operates
    independently on each chr (the baselines we currently use have non-empty
    strict TEST on only one of chr9 / chr10, so Fisher effectively collapses
    to a single Williams test).

    Returns:
        out[tname]['e2h']  -> [arr_chr9, arr_chr10]
        out[tname]['bl']   -> [arr_chr9, arr_chr10]  (baseline predictions)
        out[tname]['gt']   -> [arr_chr9, arr_chr10]
    Chromosomes whose strict-test mask is empty contribute zero-length arrays
    that downstream code naturally filters out (n <= 3 guards).
    Bins where the baseline did not write a prediction (missing files,
    chr-out-of-coverage tracks like AlphaGenome's GM12878 H3K27me3) are
    dropped by the per-element finite mask.
    """
    tl = Track_Loader(f'{hic2tarck_dir}/{cell}', RESOLUTION)
    out = {t: {'e2h': [], 'bl': [], 'gt': []} for t in TRACKS}

    for ch in TEST_CHRS:
        chrom_size = tl.chr_lens[f'chr{ch}']
        gt = tl.get(ch, 0, (chrom_size // RESOLUTION + 1) * RESOLUTION, 0)
        pred_e2h = load_npy('Evo2HiC',  cell, ch)
        bl_file = baseline_dir / cell / f'{ch}.npy'
        if not bl_file.exists():
            # Baseline did not run on this chromosome (e.g. AlphaGenome on chr9).
            # Insert empty per-chr arrays so Williams + Fisher skip it cleanly.
            for tname in TRACKS:
                out[tname]['e2h'].append(np.empty(0, dtype=np.float32))
                out[tname]['bl' ].append(np.empty(0, dtype=np.float32))
                out[tname]['gt' ].append(np.empty(0, dtype=np.float32))
            continue
        pred_bl = np.load(bl_file)

        n = min(gt.shape[1], pred_e2h.shape[1], pred_bl.shape[1])
        mask = compute_strict_test_mask(ch, chrom_size, folds_df,
                                        test_fold, RESOLUTION)[:n]

        for ti, tname in enumerate(TRACKS):
            e = pred_e2h[ti, :n][mask]
            b = pred_bl [ti, :n][mask]
            g = gt      [ti, :n][mask]
            valid = np.isfinite(e) & np.isfinite(b) & np.isfinite(g)
            out[tname]['e2h'].append(e[valid])
            out[tname]['bl' ].append(b[valid])
            out[tname]['gt' ].append(g[valid])

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
    """Returns (pccs, spcs, pvals_pcc, pvals_spc, ns) for a single baseline.

    pccs/spcs[cell][track] = {'e2h':, 'bl':}     # aggregate (concat chr9+chr10)
    pvals_pcc[cell][track]: per-chromosome Williams (one-sided, alt: r(E2H, gt) >
        r(baseline, gt)) combined with Fisher's method, gated on Evo2HiC beating
        the baseline on the aggregate PCC.
    pvals_spc[cell][track]: same as PCC but on per-chromosome rank-transformed
        vectors, so the underlying correlation is Spearman.
    """
    pccs, spcs, pvals_pcc, pvals_spc, ns = {}, {}, {}, {}, {}
    for cell, tracks_d in collected.items():
        pccs[cell], spcs[cell] = {}, {}
        pvals_pcc[cell], pvals_spc[cell] = {}, {}
        ns[cell] = {}
        for tname in TRACKS:
            e_chrs = tracks_d[tname]['e2h']
            b_chrs = tracks_d[tname]['bl']
            g_chrs = tracks_d[tname]['gt']

            e = np.concatenate(e_chrs) if e_chrs else np.empty(0)
            b = np.concatenate(b_chrs) if b_chrs else np.empty(0)
            g = np.concatenate(g_chrs) if g_chrs else np.empty(0)
            n = len(g)
            ns[cell][tname] = n
            if n < 10:
                pccs[cell][tname] = {'e2h': np.nan, 'bl': np.nan}
                spcs[cell][tname] = {'e2h': np.nan, 'bl': np.nan}
                pvals_pcc[cell][tname] = None
                pvals_spc[cell][tname] = None
                continue

            pcc_e = stats.pearsonr(e, g)[0] if np.std(e) > 0 else np.nan
            pcc_b = stats.pearsonr(b, g)[0] if np.std(b) > 0 else np.nan
            spc_e = stats.spearmanr(e, g)[0] if np.std(e) > 0 else np.nan
            spc_b = stats.spearmanr(b, g)[0] if np.std(b) > 0 else np.nan
            pccs[cell][tname] = {'e2h': pcc_e, 'bl': pcc_b}
            spcs[cell][tname] = {'e2h': spc_e, 'bl': spc_b}

            if (np.isfinite(pcc_e) and np.isfinite(pcc_b) and pcc_e > pcc_b):
                p_pcc, _ = _per_chr_williams_fisher(e_chrs, b_chrs, g_chrs,
                                                    rank_transform=False)
                pvals_pcc[cell][tname] = p_pcc
            else:
                pvals_pcc[cell][tname] = None

            if (np.isfinite(spc_e) and np.isfinite(spc_b) and spc_e > spc_b):
                p_spc, _ = _per_chr_williams_fisher(e_chrs, b_chrs, g_chrs,
                                                    rank_transform=True)
                pvals_spc[cell][tname] = p_spc
            else:
                pvals_spc[cell][tname] = None

    return pccs, spcs, pvals_pcc, pvals_spc, ns


# ---------- plot ----------
def plot_barplot(baseline_results, out_pdf):
    """baseline_results: list of dicts, one per baseline (Borzoi, AlphaGenome),
    each containing 'meta' (BASELINES entry) and 'pccs', 'spcs',
    'pvals_pcc', 'pvals_spc', 'ns'. Lays out 2 (metrics) × |baselines| rows
    by |cells| cols.
    """
    e2h_color = '#fb8072'
    plt.rcParams.update({'font.size': 9})

    n_baselines = len(baseline_results)
    n_rows = len(METRICS) * n_baselines     # 2 per baseline
    n_cols = len(CELLS)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(9.2, 2.8 * n_rows),
                             sharex=True)
    if n_rows == 1:
        axes = axes[None, :]

    x = np.arange(len(TRACKS))
    width = 0.38

    panel_idx = 0
    for bi, bres in enumerate(baseline_results):
        meta = bres['meta']
        for mi, metric in enumerate(METRICS):
            row_idx = bi * len(METRICS) + mi
            vals_dict = bres['pccs'] if metric == 'PCC' else bres['spcs']
            for col_idx, cell in enumerate(CELLS):
                ax = axes[row_idx, col_idx]

                vals_e = np.array([vals_dict[cell][t]['e2h'] for t in TRACKS])
                vals_b = np.array([vals_dict[cell][t]['bl']  for t in TRACKS])

                ax.bar(x - width / 2, vals_e, width, label='Evo2HiC',
                       color=e2h_color, alpha=0.9)
                ax.bar(x + width / 2, vals_b, width, label=meta['name'],
                       color=meta['color'], alpha=0.9)

                all_vals = np.concatenate([vals_e, vals_b])
                all_vals = all_vals[np.isfinite(all_vals)]
                ymax = float(all_vals.max()) * 1.25 if len(all_vals) else 1.0
                ax.set_ylim(0, ymax)

                # Cell title only on the top row of each baseline block
                if mi == 0:
                    ax.set_title(cell, fontsize=10)

                ax.set_ylabel(metric)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.tick_params(axis='y', which='both', length=2)
                ax.tick_params(axis='x', which='both', length=0)

                ax.text(-0.14, 1.05, PANEL_LABELS[panel_idx],
                        transform=ax.transAxes,
                        fontsize=12, fontweight='bold', va='bottom', ha='right')
                panel_idx += 1

        # Per-baseline legend at the top-right of this baseline's first row.
        # Each baseline gets its own legend so the colour mapping is unambiguous.
        handles, labels_list = axes[bi * len(METRICS), 0].get_legend_handles_labels()
        # Place legend to the right of the rightmost panel in the first row of
        # this baseline block.
        top_row_axes = axes[bi * len(METRICS), -1]
        top_row_axes.legend(handles, labels_list, loc='upper right',
                            bbox_to_anchor=(1.0, 1.32),
                            ncol=len(labels_list), frameon=False, fontsize=9,
                            columnspacing=1.2, handletextpad=0.5)

    # X-axis labels on bottom row only
    for col_idx in range(n_cols):
        axes[-1, col_idx].set_xticks(x)
        axes[-1, col_idx].set_xticklabels(TRACKS, rotation=30, ha='right')

    fig.tight_layout()
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
    print(f'Borzoi BED fold labels: {sorted(folds_df.fold.unique())}\n')

    baseline_results = []
    for meta in BASELINES:
        print(f'\n############# {meta["name"]} (strict TEST = {meta["test_fold"]}) #############')

        # Report mask coverage for this baseline
        for ch in TEST_CHRS:
            tl = Track_Loader(f'{hic2tarck_dir}/{CELLS[0]}', RESOLUTION)
            chrom_size = tl.chr_lens[f'chr{ch}']
            mask = compute_strict_test_mask(ch, chrom_size, folds_df,
                                             meta['test_fold'], RESOLUTION)
            print(f'  chr{ch}: {mask.sum()} / {len(mask)} bins ({100*mask.mean():.1f}%) '
                  f'in strict {meta["test_fold"]}-only (no overlap with other folds)')

        collected = {c: collect_for_baseline(c, folds_df,
                                              meta['pred_dir'], meta['test_fold'])
                     for c in CELLS}
        pccs, spcs, pvals_pcc, pvals_spc, ns = compute_metrics(collected)

        # Per-metric summary table
        _print_metric_table(meta['name'], pccs, pvals_pcc, ns, 'PCC')
        _print_metric_table(meta['name'], spcs, pvals_spc, ns, 'SPC')
        _print_aggregate_summaries(meta['name'], pccs, spcs)

        baseline_results.append({
            'meta': meta, 'pccs': pccs, 'spcs': spcs,
            'pvals_pcc': pvals_pcc, 'pvals_spc': pvals_spc, 'ns': ns,
        })

    # Paper-style avg per-entry relative improvement (matches Fig.4 reporting:
    # mean_i [E2H_i / Base_i - 1] over all GM12878 + H1ESC × {valid tracks} entries).
    print('\n\n############# Per-entry relative improvement (paper convention) #############')
    print('imp_i = (Evo2HiC_i - Base_i) / Base_i   ;   reported = arithmetic mean over entries')
    for bres in baseline_results:
        bname = bres['meta']['name']
        pccs = bres['pccs']
        print(f'\n=== Evo2HiC vs {bname} | PCC ===')
        imps = []
        for cell in CELLS:
            for t in TRACKS:
                me = pccs[cell][t]['e2h']
                mb = pccs[cell][t]['bl']
                if np.isfinite(me) and np.isfinite(mb) and mb != 0:
                    imp = (me - mb) / mb
                    imps.append(imp)
                    print(f'    {cell:8s} {t:10s}  E2H={me:.4f}  Base={mb:.4f}  '
                          f'rel-imp = {imp*100:+7.2f}%')
        if imps:
            rel = float(np.mean(imps)) * 100
            print(f'  → n={len(imps)} entries, '
                  f'avg per-entry relative improvement of Evo2HiC vs {bname}: '
                  f'{rel:+.2f}%')

    plot_barplot(baseline_results, args.out_pdf)


def _print_metric_table(baseline_name, vals, pvals, ns, metric_name):
    print(f"\n=== {baseline_name} | {metric_name} ===")
    print(f"{'Cell':8s}  {'Track':10s}  {'n':>7s}  "
          f"{'E2H':>8s}  {'Baseline':>9s}  {'Gap':>8s}  {'p':>12s}  stars")
    print('-' * 90)
    for cell in CELLS:
        for tname in TRACKS:
            n = ns[cell][tname]
            me = vals[cell][tname]['e2h']
            mb = vals[cell][tname]['bl']
            p = pvals[cell][tname]
            gap_str = (f'{(me - mb) / mb * 100:+7.2f}%'
                       if np.isfinite(mb) and mb != 0 else '--')
            me_s = f'{me:.4f}' if np.isfinite(me) else 'NaN'
            mb_s = f'{mb:.4f}' if np.isfinite(mb) else 'NaN'
            if p is None:
                if not (np.isfinite(me) and np.isfinite(mb)):
                    p_s, stars = '--', '(insufficient data)'
                else:
                    p_s, stars = '--', 'n.s. (no improvement)'
            else:
                p_s = f'{p:.3e}'
                stars = _p_to_stars(p)
            print(f'{cell:8s}  {tname:10s}  {n:>7d}  {me_s:>8s}  {mb_s:>9s}  '
                  f'{gap_str:>8s}  {p_s:>12s}  {stars}')


def _print_aggregate_summaries(baseline_name, pccs, spcs):
    print(f'\n=== {baseline_name} | per-cell macro-mean over tracks ===')
    print(f"{'Cell':8s}  {'Metric':6s}  {'E2H':>9s}  {'Baseline':>10s}  "
          f"{'Abs gap':>8s}  {'Rel gap':>8s}")
    print('-' * 62)
    for cell in CELLS:
        for metric_name, vals in [('PCC', pccs), ('SPC', spcs)]:
            e_arr = np.array([vals[cell][t]['e2h'] for t in TRACKS], dtype=float)
            b_arr = np.array([vals[cell][t]['bl' ] for t in TRACKS], dtype=float)
            mask = np.isfinite(e_arr) & np.isfinite(b_arr)
            if mask.sum() == 0:
                print(f'{cell:8s}  {metric_name:6s}  {"NaN":>9s}  {"NaN":>10s}  '
                      f'{"--":>8s}  {"--":>8s}')
                continue
            e_mean = float(e_arr[mask].mean())
            b_mean = float(b_arr[mask].mean())
            abs_gap = e_mean - b_mean
            rel_gap = (abs_gap / b_mean * 100) if b_mean != 0 else np.nan
            rel_s = f'{rel_gap:+7.2f}%' if np.isfinite(rel_gap) else '--'
            print(f'{cell:8s}  {metric_name:6s}  {e_mean:>9.4f}  {b_mean:>10.4f}  '
                  f'{abs_gap:>+8.4f}  {rel_s:>8s}')

    print(f'\n=== {baseline_name} | overall across cells ===')
    for metric_name, vals in [('PCC', pccs), ('SPC', spcs)]:
        e_all, b_all = [], []
        for cell in CELLS:
            for t in TRACKS:
                me, mb = vals[cell][t]['e2h'], vals[cell][t]['bl']
                if np.isfinite(me) and np.isfinite(mb):
                    e_all.append(me); b_all.append(mb)
        e_mean = float(np.mean(e_all))
        b_mean = float(np.mean(b_all))
        abs_gap = e_mean - b_mean
        rel_gap = (abs_gap / b_mean * 100) if b_mean != 0 else np.nan
        print(f'  {metric_name}: E2H={e_mean:.4f}  Base={b_mean:.4f}  '
              f'Δ={abs_gap:+.4f} ({rel_gap:+.2f}%)')


if __name__ == '__main__':
    main()