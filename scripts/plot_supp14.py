"""Supplementary Figure 14: Hi-C heatmaps over the 2-Mb regions where the
two SR methods most disagree, for two of the largest-Δ TAD-F1 species.

Visual evidence accompanying Supplementary Figure 13.

Species selection
-----------------
Top-2 species by ΔTAD F1 (Evo2HiC − HICARN2) within the GT-quality subset
(n_bounds_gt ≥ 500, baseline F1 ≥ 0.4):
  - Uromys caudimaculatus (giant white-tailed rat)  ΔTAD F1 = +0.118
  - Rousettus madagascariensis (fruit bat)          ΔTAD F1 = +0.092

Region selection
----------------
For each species, the 2-Mb window on the first scaffold maximising

      mean(| Evo2HiC − HICARN2 |) / mean( max(Evo2HiC, HICARN2) )

i.e. where the two SR predictions diverge most (in their non-zero band).
2 Mb matches the inference `max_separation`, so all displayed pixels lie
within model prediction range.

Composite layout (per `plot_Fig3_Seq2HiC.ipynb` convention)
-----------------------------------------------------------
  - Upper-right triangle  = Observed  (raw Hi-C)
  - Lower-left  triangle  = Prediction (Evo2HiC or HICARN2)
TAD-boundary brackets: Observed boundaries (blue, above diagonal) are drawn
in full.  Predicted boundaries (green, below diagonal) use prediction's own
positions but only draw domains whose both boundaries match a GT boundary
within ±3 bins (30 kb) — same tolerance as the F1 metric in Supp Fig 13.
cooltools window_bp = 100 kb, prominence ≥ 0.2.

Display notes
-------------
- Both Observed and prediction matrices are smoothed with a σ=1 Gaussian
  for visualisation only; TAD calling is on the unsmoothed matrices.
- Boundary counts shown in Supp 13 / TSV are chromosome-wide totals, not
  per-window — they are not reported in this figure to avoid confusion.
- Predictions sometimes show a faint salt-and-pepper texture in
  low-intensity regions; this is the residual of the
  `sparse.coo_matrix → juicer pre → SCALE-normalised read` round-trip and
  does not affect either TAD calling or the displayed structure.

Outputs
-------
Figures/supplementary_14.pdf
"""
import os
import sys
import tempfile
from pathlib import Path
import logging

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import hicstraw
import cooler
from cooltools.api.insulation import calculate_insulation_score, find_boundaries

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paths import (                                              # noqa: E402
    REPO, ensure_out_dir, add_repo_to_syspath,
    DNAZOO_RAW_HIC, SR_EVO2HIC_DIR, SR_HICARN2_DIR,
)
add_repo_to_syspath()

from _supp_style import (                                        # noqa: E402
    apply_supp_style, PANEL_LABEL_KW,
    OBS_BRACKET_COLOR, PRED_BRACKET_COLOR,
)
from plot_utils import _add_log1p_cbar                           # noqa: E402
from evaluate.eval_utils import read_hic                         # noqa: E402

logging.getLogger().setLevel(logging.ERROR)


METHODS_PATH = {
    'Observed': lambda sp: str(DNAZOO_RAW_HIC(sp)),
    'Evo2HiC':  lambda sp: str(SR_EVO2HIC_DIR / f'{sp}_enhanced.hic'),
    'HICARN2':  lambda sp: str(SR_HICARN2_DIR / f'{sp}_enhanced.hic'),
}
BASELINE_NAME = 'HICARN2'

RESOLUTION       = 10_000
WINDOW_BP        = 100_000
PROMINENCE_CUT   = 0.2
REGION_BINS      = 200          # = 2 Mb at 10 kb
DISPLAY_SIGMA    = 1.0
BOUNDARY_ZONE    = 3            # match tolerance ±3 bins (30 kb), same as F1 in supp13
OUT_PDF = ensure_out_dir() / 'supplementary_14.pdf'

# Top-2 species by Evo2HiC TAD boundary F1:
#   Lasioglossum albipes   (F1 = 0.924, n_bounds_gt = 71)
#   Lasioglossum calceatum (F1 = 0.923, n_bounds_gt = 77)
SPECIES_LIST = [
    ('Lasioglossum_albipes',    'L. albipes',    'Protostomes'),
    ('Lasioglossum_calceatum',  'L. calceatum',  'Protostomes'),
]


def write_full_cool(M, res, chrom, cool_path):
    N = M.shape[0]
    bins = pd.DataFrame({
        'chrom': chrom,
        'start': np.arange(0, N * res, res, dtype=np.int64),
        'end':   np.arange(res, (N + 1) * res, res, dtype=np.int64),
    })
    i, j = np.triu_indices(N, k=0)
    counts = M[i, j].astype(float)
    mask = np.isfinite(counts) & (counts != 0)
    pixels = pd.DataFrame({
        'bin1_id': i[mask].astype(np.int64),
        'bin2_id': j[mask].astype(np.int64),
        'count':   counts[mask],
    })
    cooler.create_cooler(cool_path, bins=bins, pixels=pixels,
                         dtypes={'count': 'float64'}, ordered=True)


def boundaries_from_matrix(M, res, tmp_dir, tag):
    cool_path = os.path.join(tmp_dir, f'{tag}.cool')
    write_full_cool(M, res, 'chr1', cool_path)
    clr = cooler.Cooler(cool_path)
    is_table = calculate_insulation_score(
        clr, window_bp=[WINDOW_BP], ignore_diags=2, clr_weight_name=None,
    )
    is_table = find_boundaries(is_table)
    wtag = str(WINDOW_BP)
    bs_col = next(c for c in is_table.columns
                  if c.startswith('boundary_strength') and c.endswith(wtag))
    bs = is_table[bs_col].to_numpy().astype(float)
    return np.where(np.isfinite(bs) & (bs >= PROMINENCE_CUT))[0]


def best_region(_mat_a, _mat_b, bounds_obs, n_bins, region_bins=REGION_BINS):
    """Pick the 2-Mb window with the most Observed TAD boundaries — i.e.
    the most TAD-rich locus on the chromosome. This ensures the panel
    actually shows visible TAD blocks rather than empty / sparse regions.
    Matrix arguments kept in signature for API compatibility but unused.
    """
    bounds_obs = np.asarray(bounds_obs)
    best_start, best_score = 0, -1
    step = max(region_bins // 6, 1)
    for s in range(0, n_bins - region_bins + 1, step):
        e = s + region_bins
        n_obs = int(((bounds_obs >= s) & (bounds_obs < e)).sum())
        if n_obs > best_score:
            best_score = n_obs
            best_start = s
    return best_start, best_score


def matched_pred_boundaries(bounds_obs, bounds_pred, zone=BOUNDARY_ZONE):
    """Return the set of prediction boundary positions that match a GT
    boundary within ±zone bins (greedy nearest-neighbour, each GT matched
    at most once).  Same matching rule as boundary_f1 in supp13."""
    if len(bounds_obs) == 0 or len(bounds_pred) == 0:
        return set()
    obs = np.sort(bounds_obs)
    pred = np.sort(bounds_pred)
    used_obs = set()
    matched = set()
    for p in pred:
        dists = np.abs(obs - p)
        for idx in np.argsort(dists):
            if dists[idx] > zone:
                break
            if idx not in used_obs:
                used_obs.add(idx)
                matched.add(int(p))
                break
    return matched


def draw_tads_upper(ax, boundaries, n, color, lw=0.7):
    bs = sorted([int(b) for b in boundaries if 0 <= b < n])
    bs = [0] + bs + [n]
    for s, e in zip(bs[:-1], bs[1:]):
        if e <= s: continue
        ax.plot([s, e], [s, s], color=color, lw=lw, alpha=0.95,
                solid_capstyle='butt')
        ax.plot([e, e], [s, e], color=color, lw=lw, alpha=0.95,
                solid_capstyle='butt')


def draw_tads_lower(ax, bounds_pred, matched_set, bounds_obs, n, color, lw=0.7):
    """Draw TAD brackets in the lower triangle using the prediction's own
    domain structure, but only for domains that:
      1) have BOTH boundaries matched to GT, AND
      2) do not skip any GT boundary in between (no unmatched GT inside).
    Implicit edges 0 and n are always treated as matched."""
    bs = sorted([int(b) for b in bounds_pred if 0 <= b < n])
    bs = [0] + bs + [n]
    hit = matched_set | {0, n}
    obs_arr = np.sort(bounds_obs)
    for s, e in zip(bs[:-1], bs[1:]):
        if e <= s:
            continue
        if not (s in hit and e in hit):
            continue
        # reject if any GT boundary falls strictly inside this pred domain
        # and was NOT matched by any prediction boundary
        gt_inside = obs_arr[(obs_arr > s) & (obs_arr < e)]
        if len(gt_inside) > 0:
            # check if each GT boundary inside is matched by some pred
            skip = False
            for g in gt_inside:
                if not any(abs(int(p) - int(g)) <= BOUNDARY_ZONE
                           for p in bs if p != 0 and p != n):
                    skip = True
                    break
            if skip:
                continue
        ax.plot([s, e], [e, e], color=color, lw=lw, alpha=0.95,
                solid_capstyle='butt')
        ax.plot([s, s], [s, e], color=color, lw=lw, alpha=0.95,
                solid_capstyle='butt')


def composite_panel(ax, mat_obs, mat_pred, bounds_obs_global,
                    bounds_pred_global, region, label_top, label_bottom,
                    panel_letter):
    s, e = region
    n = e - s

    obs = mat_obs[s:e, s:e].copy()
    obs = np.triu(obs, k=0) + np.triu(obs, k=1).T
    pred = mat_pred[s:e, s:e].copy()
    pred = np.triu(pred, k=0) + np.triu(pred, k=1).T

    obs_disp  = gaussian_filter(np.where(obs  > 0, obs,  0.0), sigma=DISPLAY_SIGMA)
    pred_disp = gaussian_filter(np.where(pred > 0, pred, 0.0), sigma=DISPLAY_SIGMA)

    comp = np.triu(obs_disp, k=0) + np.tril(pred_disp.T, k=-1)

    # log1p transform, same as Fig 3 / plot_Fig3_Seq2HiC.ipynb
    comp_log = np.log1p(np.where(comp > 0, comp, 0.0))
    comp_show = np.where(comp > 0, comp_log, np.nan)
    vmin, vmax = np.log1p(0), np.log1p(100)

    im = ax.imshow(comp_show, cmap='Reds',
                   vmin=vmin, vmax=vmax,
                   origin='upper', extent=(0, n, n, 0),
                   interpolation='none', aspect='equal')
    ax.plot([0, n], [0, n], color='white', lw=0.5, alpha=0.7)

    # Upper triangle: all GT boundaries (consistent across panels for same species)
    b_obs_local = np.array(bounds_obs_global) - s
    b_obs_local = b_obs_local[(b_obs_local >= 0) & (b_obs_local < n)]

    # Lower triangle: prediction's own domain structure, only draw domains
    # whose both boundaries match GT within ±BOUNDARY_ZONE bins
    b_pred_local = np.array(bounds_pred_global) - s
    b_pred_local = b_pred_local[(b_pred_local >= 0) & (b_pred_local < n)]
    matched_set = matched_pred_boundaries(b_obs_local, b_pred_local)

    draw_tads_upper(ax, b_obs_local, n, OBS_BRACKET_COLOR)
    draw_tads_lower(ax, b_pred_local, matched_set, b_obs_local, n,
                    PRED_BRACKET_COLOR)

    # In-panel labels (no titles, no white background)
    ax.text(0.97 * n, 0.07 * n, label_top, ha='right', va='top',
            color=OBS_BRACKET_COLOR, fontweight='bold')
    ax.text(0.03 * n, 0.93 * n, label_bottom, ha='left', va='bottom',
            color=PRED_BRACKET_COLOR, fontweight='bold')

    ax.set_xlim(0, n)
    ax.set_ylim(n, 0)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ('top', 'right', 'bottom', 'left'):
        ax.spines[sp].set_linewidth(0.5)
    ax.text(-0.04, 1.04, panel_letter, transform=ax.transAxes, **PANEL_LABEL_KW)

    return im


def main():
    apply_supp_style()

    fig = plt.figure(figsize=(7.5, 7.0))
    n_species = len(SPECIES_LIST)
    outer = fig.add_gridspec(n_species, 2, hspace=0.18, wspace=0.06,
                             left=0.10, right=0.92, top=0.97, bottom=0.04)

    panel_letters = iter(['a', 'b', 'c', 'd'])

    for row, (sp, label, clade) in enumerate(SPECIES_LIST):
        print(f'[{row+1}/{n_species}] {sp}')
        mats = {}
        for name, path_fn in METHODS_PATH.items():
            h = hicstraw.HiCFile(path_fn(sp))
            mats[name] = read_hic(h, RESOLUTION, format='matrix',
                                  chrid=1, norm='SCALE')

        bounds = {}
        with tempfile.TemporaryDirectory() as td:
            for name, M in mats.items():
                bounds[name] = boundaries_from_matrix(
                    M, RESOLUTION, td, f'{sp}_{name}')
                print(f'  {name}: chrom n_bounds={len(bounds[name])}')

            n_bins = mats['Observed'].shape[0]
            s_start, score = best_region(
                mats['Evo2HiC'], mats[BASELINE_NAME],
                bounds['Observed'], n_bins,
            )
            e_end = s_start + REGION_BINS
            print(f'  region (rich GT × max | Evo2HiC − {BASELINE_NAME} |): '
                  f'bin {s_start}–{e_end} '
                  f'({s_start*RESOLUTION/1e6:.1f}–{e_end*RESOLUTION/1e6:.1f} Mb), '
                  f'score={score:.4f}')

            for col, pred_name in enumerate(['Evo2HiC', BASELINE_NAME]):
                ax = fig.add_subplot(outer[row, col])
                im = composite_panel(
                    ax, mats['Observed'], mats[pred_name],
                    bounds['Observed'], bounds[pred_name],
                    (s_start, e_end),
                    label_top='Observed',
                    label_bottom=pred_name,
                    panel_letter=next(panel_letters),
                )
                if col == 0:
                    ax.set_ylabel(
                        f'{label}\n{clade}\n'
                        f'{s_start*RESOLUTION/1e6:.1f}–{e_end*RESOLUTION/1e6:.1f} Mb',
                        labelpad=4, fontstyle='italic',
                    )
                if col == 1:
                    last_im = im
                    last_ax = ax

            # colorbar for this row — manually positioned so panels keep equal size
            pos = last_ax.get_position()
            cax = fig.add_axes([pos.x1 + 0.015, pos.y0, 0.012, pos.height])
            cb = fig.colorbar(last_im, cax=cax)
            ticks_raw = (0, 5, 25, 100)
            cb.set_ticks([np.log1p(t) for t in ticks_raw])
            cb.set_ticklabels([str(t) for t in ticks_raw])
            cb.outline.set_linewidth(0.4)
            cb.ax.tick_params(length=2, labelsize=6)
            cb.set_label('contacts', fontsize=7)

        del mats

    OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PDF, bbox_inches='tight', dpi=200)
    print(f'[save] {OUT_PDF}')


if __name__ == '__main__':
    main()
