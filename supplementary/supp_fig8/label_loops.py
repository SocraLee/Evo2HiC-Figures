"""label_loops.py — Stage 1 of the loop-decomposition analysis.

Take ground-truth HiCCUPS loops (called on the REAL Hi-C of GM12878 / H1ESC,
chr9+chr10 only) and assign each loop a *functional class* using ONLY real
experimental ChIP-seq / DNase signal at the two anchors. This is the
model-independent "ground-truth" labelling used later to (a) color the
embedding UMAPs and (b) score how well each representation separates loop
classes (ARI / NMI).

No motif scanning, no attribution, no model predictions are used here — the
labels are derived purely from the measured epigenomic tracks, so they are
independent of the model and do not overlap with the Fig5 motif analysis.

Signal source: Track_Loader reads the per-track cutoff-normalised bigwig
(min(v, p90*20)/(p90*20) -> [0,1]) at 2000bp bins. Per anchor (a HiCCUPS 10kb
region) we take the MAX over the anchor's bins per track (peak presence).

Outputs (in this script's directory):
  loop_labels.tsv        one row per loop: coords + per-anchor signals + binary
                         mark calls + assigned class
  loop_label_summary.txt human-readable class distribution / threshold report
"""
import os
import sys
import argparse

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))   # repo root (paths.py)
import paths as P  # noqa: E402
P.add_repo_to_syspath()

from config import tracks, hic2tarck_dir, hic_loop_dir
from dataset.track_loader import Track_Loader

RES = 2000
TEST_CHRS = (9, 10)
CELLS = {
    'GM12878': '4DNFI1UEG1HD',
    'H1ESC':   '4DNFIQYQWPF5',
}
TRACK_NAMES = list(tracks.keys())  # ['DNase','CTCF','H3K27ac','H3K27me3','H3K4me3']
T = {name: i for i, name in enumerate(TRACK_NAMES)}


def read_bedpe(path):
    """Read a juicer-2.x merged_loops.bedpe; return DataFrame with anchor centers."""
    # line 0 = '#chr1\tx1\tx2\tchr2\ty1\ty2\t...' (header), line 1 = '# juicer ...' comment
    df = pd.read_csv(path, sep='\t', skiprows=[1], dtype={'#chr1': str, 'chr2': str})
    df = df[df['#chr1'] == df['chr2']].copy()
    df['chr'] = df['#chr1'].str.replace('chr', '', regex=False)
    df = df[df['chr'].isin([str(c) for c in TEST_CHRS])].copy()
    # anchor windows (10kb HiCCUPS anchors)
    df['aA_s'], df['aA_e'] = df['x1'].astype(int), df['x2'].astype(int)
    df['aB_s'], df['aB_e'] = df['y1'].astype(int), df['y2'].astype(int)
    return df[['chr', 'aA_s', 'aA_e', 'aB_s', 'aB_e']].reset_index(drop=True)


def anchor_signal(loader, chrom, a_s, a_e):
    """Max per-track signal over the anchor's 2kb bins. Returns (5,) array."""
    s = (int(a_s) // RES) * RES
    e = int(np.ceil(int(a_e) / RES)) * RES
    if e <= s:
        e = s + RES
    vals = loader.get(int(chrom), s, e, 0)  # (5, nbins), find_chr_in_set resolves 9->chr9
    return np.asarray(vals).max(axis=1)


def binarize(sigA, sigB, thr):
    """thr: dict track->threshold. Returns two dicts of binary mark calls."""
    callA = {name: bool(sigA[T[name]] >= thr[name]) for name in TRACK_NAMES}
    callB = {name: bool(sigB[T[name]] >= thr[name]) for name in TRACK_NAMES}
    return callA, callB


def anchor_state(x):
    """Single functional state for one anchor (used for non-CTCF loop typing).
    Bivalent (poised) takes precedence over pure Polycomb / Promoter."""
    if x['H3K27me3'] and x['H3K4me3']:
        return 'Bivalent'
    if x['H3K27me3']:
        return 'Polycomb'
    if x['H3K4me3']:
        return 'Promoter'
    if x['H3K27ac']:
        return 'Enhancer'
    if x['DNase']:
        return 'Open'
    return 'Quiescent'


def classify(callA, callB):
    """Order-invariant loop class. CTCF-structural takes top priority
    (architectural backbone); the remainder are typed by anchor states with
    an explicit Bivalent class."""
    # 1. CTCF / cohesin structural loop — both anchors CTCF-bound, regardless
    #    of co-occurring active marks (most HiCCUPS loops are of this kind).
    if callA['CTCF'] and callB['CTCF']:
        return 'CTCF-structural'

    sA, sB = anchor_state(callA), anchor_state(callB)
    states = {sA, sB}

    # 2. Bivalent (poised, ESC-enriched): a bivalent domain at either anchor
    if 'Bivalent' in states:
        return 'Bivalent'
    # 3. Polycomb: repressive-repressive contact
    if sA == 'Polycomb' and sB == 'Polycomb':
        return 'Polycomb'
    # 4. Enhancer-Promoter regulatory loop (CTCF-independent)
    if states == {'Promoter', 'Enhancer'}:
        return 'Enhancer-Promoter'
    # 5. Promoter-Promoter
    if sA == 'Promoter' and sB == 'Promoter':
        return 'Promoter-Promoter'
    # 6. anything else with an active/open anchor
    if states & {'Promoter', 'Enhancer', 'Open'}:
        return 'Active-other'
    # 7. one repressive + non-repressive, or quiescent
    if 'Polycomb' in states:
        return 'Polycomb-other'
    return 'Quiescent'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--genome-q', type=float, default=0.90,
                    help='per-track threshold = this genome-wide quantile of the track signal '
                         '(top (1-q) of all genomic bins are "+"); reflects true mark prevalence')
    ap.add_argument('--outdir', default=str(P.LOOP_APP_DIR))
    args = ap.parse_args()

    rows = []
    summary_lines = []
    for cell, acc in CELLS.items():
        bedpe = os.path.join(hic_loop_dir, acc, 'merged_loops.bedpe')
        assert os.path.exists(bedpe), f'missing {bedpe}'
        loops = read_bedpe(bedpe)
        loader = Track_Loader(os.path.join(hic2tarck_dir, cell), resolution=RES)

        # read both anchors' signal for every loop
        sigA = np.stack([anchor_signal(loader, r.chr, r.aA_s, r.aA_e) for r in loops.itertuples()])
        sigB = np.stack([anchor_signal(loader, r.chr, r.aB_s, r.aB_e) for r in loops.itertuples()])

        # per-track threshold from the GENOME-WIDE track distribution (top (1-genome_q)
        # of all bins are "+"). Anchor +rates then float to their true per-mark
        # prevalence instead of being forced equal.
        gw = np.asarray(loader.values, dtype=np.float32)  # (5, total_genome_bins)
        thr = {name: float(np.quantile(gw[T[name]], args.genome_q)) for name in TRACK_NAMES}

        pooled = np.concatenate([sigA, sigB], axis=0)  # (2N, 5)
        summary_lines.append(f'\n==== {cell} ({acc}) — {len(loops)} loops on chr9+chr10 ====')
        summary_lines.append(f'genome-wide threshold quantile q={args.genome_q} (top {1-args.genome_q:.0%} of genome bins = "+"):')
        for name in TRACK_NAMES:
            col = pooled[:, T[name]]
            pos = float((col >= thr[name]).mean())
            summary_lines.append(
                f'  {name:9s} thr={thr[name]:.4f}  ANCHOR +rate={pos:.2f}  '
                f'[anchor: med={np.median(col):.4f} q90={np.quantile(col,.9):.4f} max={col.max():.4f}]')

        classes = []
        for i, r in enumerate(loops.itertuples()):
            cA, cB = binarize(sigA[i], sigB[i], thr)
            cls = classify(cA, cB)
            classes.append(cls)
            row = {'cell': cell, 'chr': r.chr,
                   'aA_center': (r.aA_s + r.aA_e) // 2, 'aB_center': (r.aB_s + r.aB_e) // 2,
                   'class': cls}
            for name in TRACK_NAMES:
                row[f'A_{name}'] = sigA[i][T[name]]
                row[f'B_{name}'] = sigB[i][T[name]]
            for name in TRACK_NAMES:
                row[f'A_{name}+'] = int(cA[name])
                row[f'B_{name}+'] = int(cB[name])
            rows.append(row)

        vc = pd.Series(classes).value_counts()
        summary_lines.append('  class distribution:')
        for k, v in vc.items():
            summary_lines.append(f'    {k:20s} {v:4d}  ({v/len(classes):.1%})')

    df = pd.DataFrame(rows)
    out_tsv = os.path.join(args.outdir, 'loop_labels.tsv')
    df.to_csv(out_tsv, sep='\t', index=False)

    summary = '\n'.join(summary_lines)
    print(summary)
    print(f'\n[ok] wrote {out_tsv}  ({len(df)} loops total)')
    with open(os.path.join(args.outdir, 'loop_label_summary.txt'), 'w') as f:
        f.write(summary + f'\n\n[ok] {len(df)} loops -> {out_tsv}\n')


if __name__ == '__main__':
    main()
