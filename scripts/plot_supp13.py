"""Supplementary Figure 13 — DNase Activity-By-Contact (ABC) recovers a
CRISPRi-validated gene-enhancer map.

Application (rebuttal: "what is Hi-C->DNase good for, beyond Hi-C->ChIP-seq?").
The model's PREDICTED DNase, combined with measured Hi-C into an
Activity-By-Contact score [ABC(e,g) = activity(e) x contact(e,g) / Σ, with
activity = sqrt(DNase x H3K27ac); Fulco et al. 2019], reconstructs the
gene-enhancer wiring measured by a real CRISPRi-FlowFISH perturbation
experiment. PPIF is GM12878's well-powered locus on the held-out chr10:
52 CRISPRi-tested elements, 6 validated enhancers.

Layout (2 rows):
  a  continuous predicted-ABC landscape over the full ±1 Mb neighborhood
     (activity from the stitched genome-wide track, contact from GM12878
     Hi-C); the 6 CRISPRi-validated enhancers sit on the ABC peaks.
  b  all 52 tested elements ranked by predicted ABC, validated ones highlighted.
  c  mean rank of the validated enhancers under distance / Hi-C-only / ABC.
  d  per-cell-line agreement (Spearman) between ABC from PREDICTED tracks and
     ABC from MEASURED tracks, over chr9/10 candidate pairs (GM12878/H1ESC/K562).

Self-contained: this script COMPUTES everything it needs from the raw inputs
declared in paths.py (predicted tracks EPI_EVO2HIC_DIR, measured tracks
HIC2TRACK, per-cell .hic HIC_RAW, CRISPRi benchmark ABC_CRISPR_BENCH) and
caches the two intermediate TSVs under ABC_CACHE_DIR. The first run does the
heavy compute (a few minutes; panel d sweeps ~65k gene-element pairs across
3 cell lines); later runs load the cache. Delete a cache file to recompute.

Outputs
-------
Figures/supplementary_13.pdf
Figures/supplementary_13.png
result/abc_crispr/ppif_crispr_scores.tsv         (cache, panels a/b/c)
result/abc_crispr/abc_pred_vs_meas_chr910.tsv    (cache, panel d)
stdout : n tested, mean rank per predictor (distance / contact / ABC).
"""
import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.patches import Patch
from matplotlib.transforms import offset_copy
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paths import (                                              # noqa: E402
    ensure_out_dir, add_repo_to_syspath,
    ABC_PRED_TRACK, ABC_CRISPR_BENCH, ABC_CELL_HIC, EPI_PRED_ARGS_JSON,
    ABC_CACHE_DIR, ABC_CRISPR_PPIF_TSV, ABC_ALIGN_TSV,
    HIC_RAW, HIC2TRACK,
)
from plot_settings import colors as PAL                          # noqa: E402
add_repo_to_syspath()

from dataset.hic_loader import HiC_Loader                        # noqa: E402
from dataset.track_loader import Track_Loader                    # noqa: E402

OUT_PDF = ensure_out_dir() / 'supplementary_13.pdf'
OUT_PNG = ensure_out_dir() / 'supplementary_13.png'

RES = 2000
DNASE, H3K27AC, H3K4ME3 = 0, 2, 4
NBR = 500                                     # ±1 Mb neighborhood (bins)
GENE = 'PPIF'
GENE_CELL = 'GM12878'
GENE_CHR = 10


def _read_count():
    return json.load(open(EPI_PRED_ARGS_JSON))['read_count']


def _activity(dnase, h3k27ac):
    """ABC activity = geometric mean of DNase and H3K27ac (Fulco et al. 2019)."""
    return np.sqrt(np.clip(dnase, 0, None) * np.clip(h3k27ac, 0, None))


def _cached(path, compute):
    """Load `path` if present; otherwise run compute(), cache it, return it."""
    if path.exists():
        return pd.read_csv(path, sep='\t')
    print(f'[compute] {path.name} (not cached) ...', flush=True)
    df = compute()
    ABC_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, sep='\t', index=False)
    print(f'[cached]  {path}', flush=True)
    return df


# --------------------------------------------------------------------------- #
# Computation (cached)                                                         #
# --------------------------------------------------------------------------- #
def compute_ppif_scores():
    """Score the CRISPRi-tested elements at the GM12878 PPIF locus with three
    predictors (distance, Hi-C contact, predicted-DNase ABC). Panels a/b/c."""
    b = pd.read_csv(ABC_CRISPR_BENCH, sep='\t')
    b = b[(b.CellType == GENE_CELL) & (b.measuredGeneSymbol == GENE)]
    b = b[b['ValidConnection'].astype(str).str.upper().isin(['TRUE', '1', 'YES'])].copy()
    b['reg'] = b['Regulated'].astype(str).str.upper().isin(['TRUE', '1', 'YES'])
    b['tss'] = (b.startTSS + b.endTSS) // 2
    b['emid'] = (b.chromStart + b.chromEnd) // 2
    b['dist'] = (b.emid - b.tss).abs()
    b = b[b.dist <= NBR * RES]
    tss = int(b.tss.iloc[0])
    tb = tss // RES

    trk = np.load(ABC_PRED_TRACK(GENE_CELL, GENE_CHR))            # (5, nbins)
    act = _activity(trk[DNASE], trk[H3K27AC])
    nb = trk.shape[1]
    lo, hi = max(0, tb - NBR), min(nb, tb + NBR + 1)

    hic = HiC_Loader(str(HIC_RAW(ABC_CELL_HIC[GENE_CELL])), resolution=RES,
                     read_count=_read_count())
    row = hic.get(GENE_CHR, tb * RES, (tb + 1) * RES, 0,
                  GENE_CHR, lo * RES, hi * RES, 0, norm=True)[0]
    contact = np.nan_to_num(row[0]).astype(np.float64)           # (hi-lo,)
    rel = np.arange(lo, hi) - tb
    prox = np.abs(rel) <= 1                                       # exclude promoter-proximal
    denom_abc = float((act[lo:hi] * contact * ~prox).sum()) + 1e-12
    denom_cont = float((contact * ~prox).sum()) + 1e-12

    rows = []
    for r in b.itertuples():
        eb = int(r.emid // RES)
        if not (lo <= eb < hi):
            continue
        k = eb - lo
        rows.append(dict(
            gene=GENE, tss=tss, enh=int(r.emid), dist=int(r.dist), reg=bool(r.reg),
            s_dist=1.0 / (abs(r.dist) + RES),
            s_contact=contact[k] / denom_cont,
            s_abc_pred=act[lo:hi][k] * contact[k] / denom_abc))
    return pd.DataFrame(rows)


def compute_abc_alignment(cells=('GM12878', 'H1ESC', 'K562'), chroms=(9, 10),
                          max_genes=150, gene_q=0.20, cand_dnase=0.10):
    """Predicted-track ABC vs measured-track ABC over many (gene, candidate)
    pairs on chr9/10 in each cell line. Panel d.

    Genes = active-promoter bins (measured H3K4me3 high). Candidates =
    accessible non-promoter bins within ±1 Mb (measured DNase > cand_dnase).
    Activity = sqrt(DNase x H3K27ac); contact = each cell's KR-normalized Hi-C."""
    read_count = _read_count()
    rows = []
    for cell in cells:
        tl = Track_Loader(str(HIC2TRACK(cell)), resolution=RES)
        hic = HiC_Loader(str(HIC_RAW(ABC_CELL_HIC[cell])), resolution=RES,
                         read_count=read_count)
        for ch in chroms:
            f = ABC_PRED_TRACK(cell, ch)
            if not f.exists():
                print(f'[skip] no predicted track {cell} chr{ch}', flush=True); continue
            ptr = np.load(f)                                     # (5, nbins)
            nb = ptr.shape[1]
            meas = tl.get(ch, 0, nb * RES, 0)[:, :nb]            # (5, nbins)
            act_p = _activity(ptr[DNASE], ptr[H3K27AC])
            act_m = _activity(meas[DNASE], meas[H3K27AC])
            h4 = meas[H3K4ME3]
            thr = np.quantile(h4[h4 > 0], 1 - gene_q) if (h4 > 0).any() else 0.1
            genes = np.where((h4 >= max(thr, 0.1)) & (meas[DNASE] > 0.05))[0]
            if len(genes) > max_genes:
                genes = genes[np.argsort(-h4[genes])[:max_genes]]
            for tb in genes:
                lo, hi = max(0, tb - NBR), min(nb, tb + NBR + 1)
                row = hic.get(ch, tb * RES, (tb + 1) * RES, 0,
                              ch, lo * RES, hi * RES, 0, norm=True)[0]
                contact = np.nan_to_num(row[0]).astype(np.float64)
                rel = np.arange(lo, hi) - tb
                prox = np.abs(rel) <= 1
                dp = float((act_p[lo:hi] * contact * ~prox).sum()) + 1e-12
                dm = float((act_m[lo:hi] * contact * ~prox).sum()) + 1e-12
                abc_p = act_p[lo:hi] * contact / dp
                abc_m = act_m[lo:hi] * contact / dm
                cand = (~prox) & (meas[DNASE, lo:hi] > cand_dnase)
                for k in np.where(cand)[0]:
                    rows.append((cell, int(tb * RES), int(abs(rel[k]) * RES),
                                 float(abc_p[k]), float(abc_m[k])))
        print(f'[ok] {cell}: {len(rows)} cumulative pairs', flush=True)
    return pd.DataFrame(rows, columns=['cell', 'gene_tss', 'dist', 'abc_pred', 'abc_meas'])


def abc_landscape(tss):
    """Continuous predicted-ABC per bin over ±NBR around the PPIF TSS (same
    definition as compute_ppif_scores): predicted activity x Hi-C contact,
    normalized over the neighborhood. Drives the panel-a fill."""
    trk = np.load(ABC_PRED_TRACK(GENE_CELL, GENE_CHR))
    act = _activity(trk[DNASE], trk[H3K27AC])
    nb, tb = act.shape[0], tss // RES
    lo, hi = max(0, tb - NBR), min(nb, tb + NBR + 1)
    hic = HiC_Loader(str(HIC_RAW(ABC_CELL_HIC[GENE_CELL])), resolution=RES,
                     read_count=_read_count())
    row = hic.get(GENE_CHR, tb * RES, (tb + 1) * RES, 0,
                  GENE_CHR, lo * RES, hi * RES, 0, norm=True)[0]
    contact = np.nan_to_num(row[0]).astype(np.float64)
    rel = np.arange(lo, hi) - tb
    prox = np.abs(rel) <= 1
    denom = float((act[lo:hi] * contact * ~prox).sum()) + 1e-12
    return rel * RES / 1e3, act[lo:hi] * contact / denom          # x in kb, ABC per bin


# --------------------------------------------------------------------------- #
# Figure                                                                      #
# --------------------------------------------------------------------------- #
def main():
    d = _cached(ABC_CRISPR_PPIF_TSV, compute_ppif_scores)
    d = d[d.gene == GENE].copy()
    tss = int(d.tss.iloc[0])
    n = len(d)
    # ranks (1 = top) of the validated enhancers under each predictor (higher score = better)
    PRED = [('distance', 's_dist'), ('contact', 's_contact'), ('ABC', 's_abc_pred')]
    ranks = {}
    for nm, col in PRED:
        s = d.sort_values(col, ascending=False).reset_index(drop=True)
        ranks[nm] = (s.index[s.reg].values + 1)

    xk, abc = abc_landscape(tss)

    # ---- style: matches plot_revision/_supp_style.py (Arial 7pt, thin axes) ----
    plt.rcParams.update({
        'font.family': 'Arial', 'font.size': 7, 'axes.labelsize': 7, 'axes.titlesize': 7,
        'xtick.labelsize': 7, 'ytick.labelsize': 7, 'legend.fontsize': 7,
        'axes.linewidth': 0.5, 'xtick.major.width': 0.5, 'ytick.major.width': 0.5,
        'xtick.major.size': 2, 'ytick.major.size': 2,
        'pdf.fonttype': 42, 'ps.fonttype': 42,
    })
    # Arial has no bold face available -> emulate bold via a same-color glyph stroke
    PANEL_KW = dict(fontsize=8, ha='left', va='top',
                    path_effects=[pe.withStroke(linewidth=0.7, foreground='black')])

    fig = plt.figure(figsize=(7.5, 4.8))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 0.9], hspace=0.55, wspace=0.42)
    a = fig.add_subplot(gs[0, :])
    b = fig.add_subplot(gs[1, 0])
    c = fig.add_subplot(gs[1, 1])
    dbar = fig.add_subplot(gs[1, 2])

    # --- a: ABC landscape across the tested region ---
    a.fill_between(xk, abc, color=PAL[1], zorder=1, label='ABC')
    d['rel'] = (d.enh - tss) / 1e3
    neg, pos = d[~d.reg], d[d.reg]
    # anchor each marker by its BOTTOM edge on the data value (shift up half its height in
    # points) so markers near ABC=0 sit fully above the axis instead of being half-clipped
    yoff = lambda s: offset_copy(a.transData, fig=fig, y=np.sqrt(s) / 2, units='points')
    a.scatter(neg.rel, neg.s_abc_pred, s=14, c='lightgray', edgecolor='gray', lw=0.4,
              zorder=3, label='non-enhancer', transform=yoff(14))
    # enhancers sharing a 2-kb bin have identical ABC (model can't resolve <2 kb) -> one star
    # per bin, with the count drawn inside when a bin holds >1 validated enhancer
    pos = pos.assign(bin=pos.enh // RES)
    posb = pos.groupby('bin', as_index=False).agg(
        rel=('rel', 'mean'), y=('s_abc_pred', 'mean'), k=('enh', 'size'))
    single, multi = posb[posb.k == 1], posb[posb.k > 1]
    a.scatter(single.rel, single.y, s=60, marker='*', c=PAL[0], edgecolor='k', lw=0.5,
              zorder=4, label='enhancer', transform=yoff(60))
    a.scatter(multi.rel, multi.y, s=165, marker='*', c=PAL[0], edgecolor='k', lw=0.5,
              zorder=4, transform=yoff(165))
    for r in multi.itertuples():
        a.text(r.rel, r.y, str(int(r.k)), ha='center', va='center', fontsize=5.5, color='k',
               zorder=5, transform=yoff(165))
    a.axvline(0, color='k', lw=0.8, ls='--', zorder=5, label='PPIF TSS')
    a.set_xlim(d.rel.min() - 40, d.rel.max() + 40)           # span the tested candidates only
    # the promoter (TSS-proximal) ABC is ~5x the strongest enhancer -> cap y-axis to the
    # enhancer range so the distal peaks stay visible; the promoter peak is shown but off-scale
    ytop = pos.s_abc_pred.max() * 1.18
    a.set_ylim(0, ytop)
    a.text(12, ytop * 0.97, 'promoter\n(off-scale)', ha='left', va='top', fontsize=6)
    a.set_xlabel('distance from TSS (kb)')
    a.set_ylabel('ABC score')
    a.legend(loc='upper right', frameon=False, handlelength=1.4, borderpad=0.3, labelspacing=0.3)

    # --- b: ranked ABC, validated highlighted ---
    ds = d.sort_values('s_abc_pred', ascending=False).reset_index(drop=True)
    b.bar(np.arange(len(ds)), ds.s_abc_pred,
          color=[PAL[0] if r else 'lightgray' for r in ds.reg], edgecolor='none')
    b.set_xlabel('rank')
    b.set_ylabel('ABC score')
    b.legend(handles=[Patch(color=PAL[0], label='enhancer'),
                      Patch(color='lightgray', label='non-enhancer')],
             loc='upper right', frameon=False, handlelength=1.0, borderpad=0.3, labelspacing=0.3)

    # --- c: mean rank of the validated enhancers, per predictor ---
    order = ['distance', 'contact', 'ABC']
    labels = ['distance', 'HiC-only', 'ABC']
    rcol = {'ABC': PAL[3], 'contact': PAL[2], 'distance': '#bdbdbd'}
    means = [ranks[nm].mean() for nm in order]
    c.bar(range(len(order)), means, color=[rcol[nm] for nm in order], width=0.7)
    for i, m in enumerate(means):
        c.text(i, m + 0.08, f'{m:.1f}', ha='center', fontweight='bold')
    c.set_xticks(range(len(order)))
    c.set_xticklabels(labels)
    c.set_ylim(0, max(means) + 0.7)
    c.set_ylabel('enhancer mean rank')

    # --- d: per-cell-line ABC rank agreement (predicted vs measured tracks) on chr9/10.
    # Spearman because ABC is used to RANK enhancers; predicted tracks built with the same
    # stitched/overlap-averaged pipeline as the epigenomic evaluation. ---
    da = _cached(ABC_ALIGN_TSV, compute_abc_alignment)
    cells_order = ['GM12878', 'H1ESC', 'K562']
    ccol = {'GM12878': PAL[2], 'H1ESC': PAL[3], 'K562': PAL[0]}
    rho = {c_: spearmanr(g.abc_meas, g.abc_pred).correlation for c_, g in da.groupby('cell')}
    dbar.bar(range(3), [rho[c_] for c_ in cells_order],
             color=[ccol[c_] for c_ in cells_order], width=0.7)
    for i, c_ in enumerate(cells_order):
        dbar.text(i, rho[c_] + 0.015, f'{rho[c_]:.2f}', ha='center', fontweight='bold')
    dbar.set_xticks(range(3)); dbar.set_xticklabels(cells_order, fontsize=6.5)
    dbar.set_ylim(0, 1.08)
    dbar.set_ylabel('ABC Alignment (Spearman)')

    # panel labels in figure coords -> column-aligned, independent of y-tick widths
    for ax_, lab in [(a, 'a'), (b, 'b'), (c, 'c'), (dbar, 'd')]:
        p = ax_.get_position()
        fig.text(p.x0 - 0.052, p.y1 + 0.014, lab, **PANEL_KW)

    fig.savefig(OUT_PDF, dpi=300, bbox_inches='tight')
    fig.savefig(OUT_PNG, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'[ok] {OUT_PDF}  (n={n})  mean rank: '
          + '  '.join(f'{k}={v.mean():.1f}' for k, v in ranks.items()))


if __name__ == '__main__':
    main()
