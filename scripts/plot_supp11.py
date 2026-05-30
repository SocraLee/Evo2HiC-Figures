"""
Supplementary Figure 11: Ablation study — Evo2HiC vs ablation baselines.

Panel (a–c): Evo2HiC vs No-distillation  (GM12878, H1ESC, K562)
Panel (d–f): Evo2HiC vs Evo2 + HiC       (GM12878, H1ESC, K562)

For every (cell × track) pair the script:
  1. Dynamically loads PCC from TSV result files.
  2. Loads bin-level prediction .npy arrays (chr9, chr10).
  3. Directional gate: only run the test when Evo2HiC has the higher
     per-cell macro-mean PCC (otherwise mark "n.s. (no improvement)"
     without testing). This avoids displaying chr-flip artefacts where
     one chromosome strongly favours Evo2HiC and the other strongly
     favours the baseline, which one-sided + Fisher would otherwise
     report as significant despite an aggregate ΔPCC ≤ 0.
  4. Per chromosome, runs Williams' one-sided test (pre-specified
     alternative: r(Evo2HiC, GT) > r(baseline, GT)) for the overlapping
     correlations sharing GT; the third correlation r(Evo2HiC, baseline)
     is used in the SE. Per-chromosome one-sided p-values are combined
     across chr9 and chr10 with Fisher's combined probability test.
     NOTE: bins along a 1D track are spatially autocorrelated; Williams'
     SE assumes i.i.d. samples, so reported p-values are anti-conservative.
  5. Annotates the bar plot with significance stars and prints p-values.

Outputs
-------
Figures/supplementary_11.pdf : 2 × 3 panel figure (2 ablations × 3 cell lines).
stdout                       : PCC summary table with test type & p-values.
"""

import sys
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats

# ---------- paths / constants ----------
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paths import (                                             # noqa: E402
    REPO, ensure_out_dir, add_repo_to_syspath,
    CKPT_ROOT as CKPT,
    EPI_EVO2HIC_DIR, EPI_NODISTILL_DIR, EPI_EVO2EMB_HIC_DIR, EPI_RIDGE_DIR,
)
add_repo_to_syspath()

from dataset.track_loader import Track_Loader  # noqa: E402
from config import hic2tarck_dir               # noqa: E402
from plot_settings import colors as PALETTE     # noqa: E402

OUT_PDF = ensure_out_dir() / 'supplementary_11.pdf'

TRACKS = ['DNase', 'CTCF', 'H3K27ac', 'H3K27me3', 'H3K4me3']
CELLS  = ['GM12878', 'H1ESC', 'K562']
TEST_CHRS = (9, 10)

# ---------- internal keys → display names ----------
# Internal keys match TSV_PATHS / NPY_DIRS keys (unchanged).
# Display names are used only for labels and legends.
DISPLAY_NAMES = {
    'Evo2HiC':     'Evo2HiC',
    'No-pretrain': 'No-distillation',
    'Evo2emb+HiC': 'Evo2 + HiC',
    'Ridge':       'Ridge (HiC)',
}

# TSV paths for PCC extraction (internal keys)
TSV_PATHS = {
    'Evo2HiC':     EPI_EVO2HIC_DIR     / 'result.tsv',
    'No-pretrain': EPI_NODISTILL_DIR   / 'result.tsv',
    'Evo2emb+HiC': EPI_EVO2EMB_HIC_DIR / 'result.tsv',
    'Ridge':       EPI_RIDGE_DIR       / 'result.tsv',
}

# npy directories for bin-level predictions
NPY_DIRS = {
    'Evo2HiC':     EPI_EVO2HIC_DIR,
    'No-pretrain': EPI_NODISTILL_DIR,
    'Evo2emb+HiC': EPI_EVO2EMB_HIC_DIR,
    'Ridge':       EPI_RIDGE_DIR,
}

# Distinct colors for the 4 methods
COLOR_EVO2HIC      = PALETTE[0]   # '#fb8072'  (red-ish)
COLOR_NO_DISTILL   = PALETTE[1]   # '#bebada'  (purple-ish)
COLOR_EVO2_PLUS    = PALETTE[2]   # '#80b1d3'  (blue-ish)
COLOR_RIDGE        = PALETTE[3]   # '#8dd3c7'  (teal — classical baseline)

BASELINE_COLOR = {
    'No-pretrain': COLOR_NO_DISTILL,
    'Evo2emb+HiC': COLOR_EVO2_PLUS,
    'Ridge':       COLOR_RIDGE,
}

# The three ablation comparisons (each row of the figure).
# zoom_to_data: rows 1 & 2 are fine-ablation rows where Evo2HiC and the
# baseline differ by 2-3 % PCC; we zoom the y-axis to the data range so
# those small gaps are visible. Row 3 is the weak-baseline (classical
# Ridge) reference, where the contrast is enormous and a 0-anchored axis
# is the right choice.
ABLATIONS = [
    {
        'name':         'Evo2HiC  vs  No-distillation',
        'baseline':     'No-pretrain',
        'zoom_to_data': True,
    },
    {
        'name':         'Evo2HiC  vs  Evo2 + HiC',
        'baseline':     'Evo2emb+HiC',
        'zoom_to_data': True,
    },
    {
        'name':         'Evo2HiC  vs  Ridge (HiC)',
        'baseline':     'Ridge',
        'zoom_to_data': False,
    },
]

# Panel labels (3 rows × 3 cells = 9 panels)
PANEL_LABELS = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i']


# ---------- helpers ----------
def _p_to_stars(p):
    return '****' if p < 1e-4 else '***' if p < 1e-3 else '**' if p < 1e-2 \
        else '*' if p < 5e-2 else 'n.s.'


def _mean_per_cell(tsv_path):
    """Read a result.tsv and return {cell: [pcc_per_track]}."""
    df = pd.read_csv(tsv_path, sep='\t')
    out = {}
    for cell in CELLS:
        sub = df[df.Name == cell]
        if len(sub) == 0:
            out[cell] = None
        else:
            out[cell] = [float(sub[t].mean()) for t in TRACKS]
    return out


def load_all_pcc():
    """Return dict: all_results[cell][method] = [pcc_per_track]."""
    results_from_tsv = {m: _mean_per_cell(p) for m, p in TSV_PATHS.items()}
    all_results = {}
    for cell in CELLS:
        all_results[cell] = {}
        for m in TSV_PATHS:
            all_results[cell][m] = results_from_tsv[m][cell]
    return all_results


# ---------- bin-level loading & testing ----------
_gt_cache = {}


def _get_gt(cell):
    if cell not in _gt_cache:
        tl = Track_Loader(f'{hic2tarck_dir}/{cell}', 2000)
        _gt_cache[cell] = {}
        for ch in TEST_CHRS:
            size = tl.chr_lens[f'chr{ch}']
            _gt_cache[cell][ch] = tl.get(ch, 0, (size // 2000 + 1) * 2000, 0)
    return _gt_cache[cell]


def _load_per_chr(method, cell, track_idx):
    """Return {ch: 1D array} of bin-level predictions per chromosome."""
    base = NPY_DIRS[method]
    out = {}
    for ch in TEST_CHRS:
        f = base / f'{cell}/{ch}.npy'
        if not f.exists():
            return None
        out[ch] = np.load(f)[track_idx]
    return out


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
    r(a,t) > r(b,t)). Returns p = P(Z >= observed) under H0: r_at == r_bt.
    Reverse-direction data naturally yields p close to 1."""
    n = len(t)
    r_at = stats.pearsonr(a, t)[0]
    r_bt = stats.pearsonr(b, t)[0]
    r_ab = stats.pearsonr(a, b)[0]
    if not all(np.isfinite([r_at, r_bt, r_ab])):
        return None
    se = _se_williams_overlap(r_at, r_bt, r_ab, n)
    if se is None or se == 0:
        return None
    Z = (r_at - r_bt) / se
    p = float(stats.norm.sf(Z))
    return {'n': n, 'Z': float(Z), 'p': p,
            'r_at': float(r_at), 'r_bt': float(r_bt), 'r_ab': float(r_ab)}


def _fisher_combine(pvals):
    """Fisher's combined probability test. Accepts the per-block p-values
    in the same (one-sided) direction; combined p tests whether at least
    some blocks are extreme in the pre-specified direction.

    p-values are clipped to [1e-300, 1.0] before the log so that blocks
    with underflowed p (norm.sf saturating to 0 for huge Z) still carry
    their evidence into the combined statistic instead of being dropped."""
    pv = np.asarray([p for p in pvals
                     if p is not None and np.isfinite(p)])
    if len(pv) == 0:
        return None
    pv = np.clip(pv, 1e-300, 1.0)
    X2 = -2.0 * np.log(pv).sum()
    return float(stats.chi2.sf(X2, df=2 * len(pv)))


def run_paired_test(all_results, method_a, method_b, cell, track_idx, tname):
    """
    Compare method_a (Evo2HiC) vs method_b (baseline) on bin-level PCC.
    Per chromosome we run Williams' one-sided test (pre-specified
    alternative: r(pred_a, gt) > r(pred_b, gt)), then combine the
    per-chromosome one-sided p-values via Fisher's method.
    """
    pcc_a = all_results[cell][method_a][track_idx]
    pcc_b = all_results[cell][method_b][track_idx]

    result = {
        'pcc_a': pcc_a, 'pcc_b': pcc_b,
        'p': None, 'stars': None, 'test_name': None, 'n': 0,
        'per_chr': {},
    }

    if pcc_a is None or pcc_b is None or not np.isfinite(pcc_a) or not np.isfinite(pcc_b):
        result['test_name'] = '--'
        result['stars'] = '(NaN PCC)'
        return result

    # Directional gate: only run the test when Evo2HiC has the higher
    # aggregate (per-cell macro-mean) PCC. Skipping the test in the
    # reverse direction prevents one-sided + Fisher chr-flip artefacts
    # from displaying "significant" stars on bars where Evo2HiC visibly
    # loses (e.g. H1ESC H3K4me3 vs Evo2 + HiC, where chr9 favours Evo2HiC
    # and chr10 favours the baseline by similar magnitudes).
    if pcc_a <= pcc_b:
        result['test_name'] = 'Williams (skipped: ΔPCC ≤ 0)'
        result['stars'] = 'n.s. (no improvement)'
        return result

    pred_a = _load_per_chr(method_a, cell, track_idx)
    pred_b = _load_per_chr(method_b, cell, track_idx)
    if pred_a is None or pred_b is None:
        result['test_name'] = '--'
        result['stars'] = '(no data)'
        return result

    gt_per_chr = _get_gt(cell)

    per_chr_p = []
    total_n = 0
    for ch in TEST_CHRS:
        a = pred_a[ch]
        b = pred_b[ch]
        t = gt_per_chr[ch][track_idx]
        n = min(len(a), len(b), len(t))
        if n <= 3:
            continue
        block = _williams_block(a[:n], b[:n], t[:n])
        if block is None:
            continue
        result['per_chr'][ch] = block
        per_chr_p.append(block['p'])
        total_n += n

    result['n'] = total_n

    if not per_chr_p:
        result['test_name'] = 'Williams (degenerate)'
        result['stars'] = '(degenerate)'
        return result

    p_combined = _fisher_combine(per_chr_p)
    result['p'] = p_combined
    result['test_name'] = f'Williams(1-sided) x{len(per_chr_p)} + Fisher'
    result['stars'] = _p_to_stars(p_combined) if p_combined is not None else '(NaN p)'
    return result


# ---------- plotting ----------
def plot_ablations(all_results, test_results, out_pdf):
    """
    Create a 2-row × 3-col figure with a shared 3-color legend at the top.
      Row 0 (a–c): Evo2HiC vs No-distillation  (GM12878, H1ESC, K562)
      Row 1 (d–f): Evo2HiC vs Evo2 + HiC       (GM12878, H1ESC, K562)
    """
    n_rows = len(ABLATIONS)
    n_cols = len(CELLS)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(10, 7.5), sharex=True)
    plt.rcParams.update({'font.size': 9})

    x = np.arange(len(TRACKS))
    width = 0.38

    panel_idx = 0
    for row_idx, ablation in enumerate(ABLATIONS):
        baseline = ablation['baseline']
        bl_color = BASELINE_COLOR[baseline]
        bl_display = DISPLAY_NAMES[baseline]

        for col_idx, cell in enumerate(CELLS):
            ax = axes[row_idx, col_idx]

            vals_a = np.array(all_results[cell]['Evo2HiC'], dtype=float)
            raw_b = all_results[cell][baseline]
            vals_b = np.array(raw_b, dtype=float) if raw_b is not None else np.full(len(TRACKS), np.nan)

            ax.bar(x - width / 2, vals_a, width,
                   color=COLOR_EVO2HIC, alpha=0.9)
            ax.bar(x + width / 2, vals_b, width,
                   color=bl_color, alpha=0.9)

            all_vals = np.concatenate([vals_a, vals_b])
            all_vals = all_vals[np.isfinite(all_vals)]
            if len(all_vals) == 0:
                ax.set_ylim(0, 1.0)
            elif ablation.get('zoom_to_data', False):
                # zoom y-axis to data range so 2-3 % ablation gaps become visible
                lo = float(all_vals.min())
                hi = float(all_vals.max())
                pad = (hi - lo) * 0.20 if hi > lo else 0.05
                ax.set_ylim(max(0.0, lo - pad), hi + pad)
            else:
                # weak-baseline row: keep 0-anchored to emphasise the gap
                ax.set_ylim(0, float(all_vals.max()) * 1.10)
            # Significance brackets removed for visual clarity. Per-comparison
            # statistics are still computed and printed to stdout, and the
            # caption summarises overall significance.

            ax.set_ylabel('PCC')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.tick_params(axis='y', which='both', length=2)
            ax.tick_params(axis='x', which='both', length=0)

            # Cell name as title on top row only
            if row_idx == 0:
                ax.set_title(cell, fontsize=10)

            # Panel label (a–f) in top-left corner
            label = PANEL_LABELS[panel_idx]
            ax.text(-0.12, 1.05, label, transform=ax.transAxes,
                    fontsize=11, fontweight='bold', va='bottom', ha='right')
            panel_idx += 1

    # X-axis labels on bottom row
    for col_idx in range(n_cols):
        axes[-1, col_idx].set_xticks(x)
        axes[-1, col_idx].set_xticklabels(TRACKS, rotation=30, ha='right')

    # Shared 4-color legend at the top (horizontal)
    legend_handles = [
        mpatches.Patch(color=COLOR_EVO2HIC,    alpha=0.9, label='Evo2HiC'),
        mpatches.Patch(color=COLOR_NO_DISTILL, alpha=0.9, label='No-distillation'),
        mpatches.Patch(color=COLOR_EVO2_PLUS,  alpha=0.9, label='Evo2 + HiC'),
        mpatches.Patch(color=COLOR_RIDGE,      alpha=0.9, label='Ridge (HiC)'),
    ]
    fig.legend(
        handles=legend_handles,
        loc='upper center',
        bbox_to_anchor=(0.5, 1.0),
        ncol=4,
        frameon=False,
        fontsize=9,
        handletextpad=0.4,
        columnspacing=1.5,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, dpi=300, bbox_inches='tight', transparent=True)
    plt.close(fig)
    print(f'\nSaved {out_pdf}')


# ---------- main ----------
def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--out-pdf', type=Path, default=OUT_PDF)
    args = parser.parse_args()

    # 1. Load PCC values dynamically
    all_results = load_all_pcc()

    # Print loaded PCC summary
    print('\n=== Loaded PCC values ===')
    for cell in CELLS:
        for m in TSV_PATHS:
            dn = DISPLAY_NAMES[m]
            vals = all_results[cell][m]
            vals_str = ', '.join(f'{v:.4f}' if v is not None and np.isfinite(v) else 'NaN'
                                 for v in vals) if vals else 'None'
            print(f'  {cell:8s}  {dn:18s}  [{vals_str}]')

    # 2. Run paired tests for each ablation
    test_results = {}
    print(f"\n{'Ablation':30s}  {'Cell':8s}  {'Track':10s}  {'n':>7s}  "
          f"{'E2H PCC':>8s}  {'BL PCC':>8s}  {'test':>28s}  {'p-value':>12s}  stars")
    print('-' * 135)

    for ablation in ABLATIONS:
        baseline = ablation['baseline']
        for cell in CELLS:
            for ti, tname in enumerate(TRACKS):
                res = run_paired_test(all_results, 'Evo2HiC', baseline, cell, ti, tname)
                test_results[(baseline, cell, tname)] = res

                pcc_a_s = f'{res["pcc_a"]:.4f}' if res['pcc_a'] is not None and np.isfinite(res['pcc_a']) else 'NaN'
                pcc_b_s = f'{res["pcc_b"]:.4f}' if res['pcc_b'] is not None and np.isfinite(res['pcc_b']) else 'NaN'
                p_s = f'{res["p"]:.3e}' if res['p'] is not None else '--'
                test_name = res['test_name'] if res['test_name'] else '--'
                stars = res['stars'] if res['stars'] else '--'
                print(f'{ablation["name"]:30s}  {cell:8s}  {tname:10s}  {res["n"]:>7d}  '
                      f'{pcc_a_s:>8s}  {pcc_b_s:>8s}  {test_name:>28s}  {p_s:>12s}  {stars}')

    # 3. Per-cell average improvement (macro-mean over tracks), per ablation
    print('\n=== Per-cell average PCC (macro-mean over tracks) ===')
    print(f"{'Ablation':30s}  {'Cell':8s}  {'E2H mean':>9s}  "
          f"{'BL mean':>9s}  {'Abs gap':>8s}  {'Rel gap':>8s}")
    print('-' * 82)
    for ablation in ABLATIONS:
        baseline = ablation['baseline']
        for cell in CELLS:
            e_arr = np.array(all_results[cell]['Evo2HiC'], dtype=float) \
                if all_results[cell]['Evo2HiC'] is not None \
                else np.full(len(TRACKS), np.nan)
            raw_b = all_results[cell][baseline]
            b_arr = np.array(raw_b, dtype=float) if raw_b is not None \
                else np.full(len(TRACKS), np.nan)
            mask = np.isfinite(e_arr) & np.isfinite(b_arr)
            if mask.sum() == 0:
                print(f'{ablation["name"]:30s}  {cell:8s}  {"NaN":>9s}  '
                      f'{"NaN":>9s}  {"--":>8s}  {"--":>8s}')
                continue
            e_mean = float(e_arr[mask].mean())
            b_mean = float(b_arr[mask].mean())
            abs_gap = e_mean - b_mean
            rel_gap = (abs_gap / b_mean * 100) if b_mean != 0 else np.nan
            rel_s = f'{rel_gap:+7.2f}%' if np.isfinite(rel_gap) else '--'
            print(f'{ablation["name"]:30s}  {cell:8s}  {e_mean:>9.4f}  '
                  f'{b_mean:>9.4f}  {abs_gap:>+8.4f}  {rel_s:>8s}')

    # 4. Paper-style avg per-entry relative improvement
    #    imp_i = Evo2HiC_i / baseline_i - 1, arithmetic mean over entries.
    #    Mirrors plot_Fig4_Epi.ipynb cell-5 (where the paper's 26.2%/34.7%
    #    numbers come from). Reported twice:
    #      (i)  all 3 cells × 5 tracks = 15 entries
    #      (ii) GM12878 + H1ESC × 5 tracks = 10 entries (paper convention,
    #           K562 excluded)
    def _avg_per_entry(ablation_name, baseline, cells_in):
        imps, per_entry = [], []
        for cell in cells_in:
            e_vals = all_results[cell]['Evo2HiC']
            b_vals = all_results[cell][baseline]
            if e_vals is None or b_vals is None:
                continue
            for ti, t in enumerate(TRACKS):
                me, mb = e_vals[ti], b_vals[ti]
                if me is not None and mb is not None \
                        and np.isfinite(me) and np.isfinite(mb) and mb != 0:
                    imp = me / mb - 1
                    imps.append(imp)
                    per_entry.append((cell, t, imp))
        return imps, per_entry

    # Split in-distribution (GM12878 + H1ESC; cell lines seen during epi-decoder
    # training) from cross-cell-line (K562; held out during decoder training,
    # so represents zero-shot transfer).
    print('\n=== Paper-style avg per-entry PCC improvement '
          '(Evo2HiC / baseline - 1, mean) ===')
    print(f"{'Ablation':30s}  {'setting':30s}  {'n':>3s}  {'mean improvement':>17s}")
    print('-' * 88)
    for ablation in ABLATIONS:
        bl = ablation['baseline']
        for label, cells_in in [
            ('in-distribution (GM12878+H1ESC)', ['GM12878', 'H1ESC']),
            ('cross-cell-line (K562)',          ['K562']),
        ]:
            imps, _ = _avg_per_entry(ablation['name'], bl, cells_in)
            rel = float(np.mean(imps)) * 100 if imps else float('nan')
            rel_s = f'{rel:+7.2f}%' if np.isfinite(rel) else '--'
            print(f'{ablation["name"]:30s}  {label:30s}  '
                  f'{len(imps):>3d}  {rel_s:>17s}')

    # Per-entry breakdown, split by setting
    print('\n=== Per-entry breakdown ===')
    for ablation in ABLATIONS:
        bl = ablation['baseline']
        for label, cells_in in [
            ('in-distribution (GM12878+H1ESC)', ['GM12878', 'H1ESC']),
            ('cross-cell-line (K562)',          ['K562']),
        ]:
            imps, per_entry = _avg_per_entry(ablation['name'], bl, cells_in)
            print(f"\n  {ablation['name']}  -  {label}  (mean = "
                  f"{np.mean(imps)*100:+.2f}% over {len(imps)} entries)")
            for cell, t, imp in per_entry:
                print(f'    {cell:8s} {t:10s}  {imp*100:+7.2f}%')

    # 5. Plot
    plot_ablations(all_results, test_results, args.out_pdf)


if __name__ == '__main__':
    main()
