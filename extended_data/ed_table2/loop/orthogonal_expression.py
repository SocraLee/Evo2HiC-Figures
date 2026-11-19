#!/usr/bin/env python
"""Orthogonal validation of loop-anchor classification using RNA-seq.

The anchor labels in loop_anchor_eval.py / loop_anchor_ccre.py are derived from
measured DNase/ChIP tracks - the same assays the model predicts - so agreement
between predicted and measured labels is a self-consistency check, not functional
validation (the criticism reviewers 1/3 raised for the ABC panel).  RNA-seq is a
different assay that enters neither the model's inputs, its training targets, nor
the label definition, so it provides an independent readout.

Two tests, run for every method (Evo2HiC / HiC-only / Evo2 sequence-only) using the
ENCODE cCRE classes predicted from that method's tracks alone (no measured ChIP):

 T1  Promoter-like anchors should sit at expressed genes.
     Compare TPM of protein-coding genes whose promoter overlaps an anchor
     predicted "Promoter-like" vs anchors predicted "CTCF-only"/"Low-DNase"
     (Mann-Whitney U, one-sided; plus AUROC of TPM > 1).

 T2  Cell-type-specific version (the strong test).  For anchors called
     Promoter-like in cell A but not in cell B, the underlying gene should be
     more expressed in A than in B.  A cell-agnostic predictor scores 0 here by
     construction.  Only genes expressed in at least one of the two cells
     (max TPM >= MIN_TPM) enter, and exact ties are excluded: with all Gencode
     genes most TSSs are silent in both cells, which makes log2FC identically 0
     and the test uninformative (the measured tracks fail it too).  Reported as
     the fraction of informative anchors with TPM_A > TPM_B (binomial test vs
     0.5) and the median log2 fold change.

Inputs: expression table data/dna/human/merged.csv (GM12878/H1ESC/K562 TPM) and
gene TSSs of all Gencode v38 genes on chr9/chr10 (data/gencode_v38_tss_chr910.tsv, built from
data/gene_regions/gencode.v38.annotation.gtf.gz).  A gene is assigned to an anchor
if its TSS falls within the HiCCUPS anchor (+-5 kb of the anchor centre).

Usage: orthogonal_expression.py [--boot 1000]
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, binomtest
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))  # repo root
import paths as P  # noqa: E402

W = str(P.RA_WORK('loop'))
D = str(P.HIC_DATA_ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from loop_anchor_eval import (T, RES, HALF, TEST_CHRS, METHODS, LOOP_DIR,  # noqa: E402
                              read_bedpe, Track_Loader, hic2tarck_dir)
from loop_anchor_ccre import Q_HIGH, ccre_class  # noqa: E402

CELLS = ['GM12878', 'H1ESC', 'K562']
TSS_TSV = str(P.RA_TSS_CHR910)   # chr, TSS, gene id of GENCODE v38 genes on chr9/10 (shipped)
EXPR = str(P.HUMAN_MERGED_CSV)
ANCHOR_BP = 5000                # HiCCUPS anchors are 10 kb wide
MIN_TPM = 1.0                   # a gene must be expressed in at least one of the two cells


def load_genes():
    b = pd.read_csv(TSS_TSV, sep='\t', header=None, names=['chr', 'tss', 'gene'])
    e = pd.read_csv(EXPR)
    e = e[['gene_id', 'GM12878_TPM', 'H1ESC_TPM', 'K562_TPM']].rename(columns={'gene_id': 'gene'})
    g = b.merge(e, on='gene', how='inner').drop_duplicates('gene')
    return g[['chr', 'tss', 'gene', 'GM12878_TPM', 'H1ESC_TPM', 'K562_TPM']]


def predicted_classes(cell, meth_dir):
    """cCRE class per anchor from a single method's predicted tracks."""
    loops = read_bedpe(os.path.join(LOOP_DIR[cell], 'merged_loops.bedpe'))
    anchors = sorted({(int(r.chr), int(r.aA_center)) for r in loops.itertuples()} |
                     {(int(r.chr), int(r.aB_center)) for r in loops.itertuples()})
    pred = {ch: np.load(f'{meth_dir}/{ch}.npy') for ch in TEST_CHRS}
    pool = np.concatenate([pred[ch] for ch in TEST_CHRS], axis=1)
    thr = {m: float(np.quantile(pool[T[m]], Q_HIGH)) for m in T}
    out = {}
    for c, pb in anchors:
        arr = pred[c]
        b = pb // RES
        lo, hi = max(0, b - HALF), min(arr.shape[1], b + HALF + 1)
        out[(c, pb)] = ccre_class({m: bool(arr[T[m], lo:hi].max() > thr[m]) for m in T})
    return out


def measured_classes(cell):
    loops = read_bedpe(os.path.join(LOOP_DIR[cell], 'merged_loops.bedpe'))
    anchors = sorted({(int(r.chr), int(r.aA_center)) for r in loops.itertuples()} |
                     {(int(r.chr), int(r.aB_center)) for r in loops.itertuples()})
    tl = Track_Loader(os.path.join(hic2tarck_dir, cell), resolution=RES)
    v = {m: np.asarray(tl.values[T[m]], float) for m in T}
    thr = {m: float(np.quantile(v[m], Q_HIGH)) for m in T}
    out = {}
    for c, pb in anchors:
        key = f'chr{c}' if f'chr{c}' in tl.chr_pos else str(c)
        off, end = tl.chr_pos[key]
        b = pb // RES
        lo, hi = max(0, b - HALF), min(end - off, b + HALF + 1)
        out[(c, pb)] = ccre_class({m: bool(v[m][off + lo: off + hi].max() > thr[m]) for m in T})
    return out


def anchor_gene_map(genes, anchors):
    """anchor -> list of gene rows whose TSS falls within the anchor window."""
    m = {}
    for a in anchors:
        c, pb = a
        g = genes[(genes.chr == c) & (genes.tss >= pb - ANCHOR_BP) & (genes.tss <= pb + ANCHOR_BP)]
        if len(g):
            m[a] = g
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=f'{W}/orthogonal_expression.tsv')
    args = ap.parse_args()
    genes = load_genes()
    print(f'[genes] {len(genes)} protein-coding genes on chr{TEST_CHRS} with TPM', flush=True)

    cls = {}
    for cell in CELLS:
        cls[(cell, 'Measured')] = measured_classes(cell)
        for meth, dirf in METHODS.items():
            d = dirf(cell)
            if all(os.path.exists(f'{d}/{ch}.npy') for ch in TEST_CHRS):
                cls[(cell, meth)] = predicted_classes(cell, d)

    tpm_col = {c: f'{c}_TPM' for c in CELLS}
    rows = []
    # ---- T1 -----------------------------------------------------------------
    for (cell, meth), cmap in sorted(cls.items()):
        gmap = anchor_gene_map(genes, list(cmap))
        prom, other = [], []
        for a, g in gmap.items():
            v = float(g[tpm_col[cell]].max())
            (prom if cmap[a] == 'Promoter-like' else
             other if cmap[a] in ('CTCF-only', 'Low-DNase') else []).append(v)
        if len(prom) < 5 or len(other) < 5:
            continue
        u, p = mannwhitneyu(prom, other, alternative='greater')
        auroc = u / (len(prom) * len(other))
        rows.append({'test': 'T1_expression_at_promoter_like', 'cell': cell, 'method': meth,
                     'n_prom': len(prom), 'n_other': len(other),
                     'median_TPM_prom': float(np.median(prom)),
                     'median_TPM_other': float(np.median(other)),
                     'AUROC': auroc, 'p': p})

    # ---- T2 -----------------------------------------------------------------
    for meth in ['Measured'] + list(METHODS):
        for i, A in enumerate(CELLS):
            for B in CELLS[i + 1:]:
                if (A, meth) not in cls or (B, meth) not in cls:
                    continue
                ca, cb = cls[(A, meth)], cls[(B, meth)]
                shared = set(ca) & set(cb)
                gmap = anchor_gene_map(genes, sorted(shared))
                fc, n_seen, n_tied = [], 0, 0
                for a, g in gmap.items():
                    if ca[a] == 'Promoter-like' and cb[a] != 'Promoter-like':
                        hi_c, lo_c = A, B
                    elif cb[a] == 'Promoter-like' and ca[a] != 'Promoter-like':
                        hi_c, lo_c = B, A
                    else:
                        continue
                    gg = g[g[[tpm_col[A], tpm_col[B]]].max(axis=1) >= MIN_TPM]
                    if not len(gg):
                        continue
                    n_seen += 1
                    r = gg.loc[gg[tpm_col[hi_c]].idxmax()]
                    v = np.log2((r[tpm_col[hi_c]] + 1) / (r[tpm_col[lo_c]] + 1))
                    if v == 0:
                        n_tied += 1
                        continue
                    fc.append(v)
                if len(fc) < 5:
                    continue
                fc = np.array(fc)
                k = int((fc > 0).sum())
                bt = binomtest(k, len(fc), 0.5, alternative='greater')
                rows.append({'test': 'T2_cell_specific_promoter', 'cell': f'{A} vs {B}',
                             'method': meth, 'n_expressed': n_seen, 'n_tied': n_tied,
                             'n_anchors': len(fc), 'frac_correct': k / len(fc),
                             'median_log2FC': float(np.median(fc)), 'p': bt.pvalue})

    df = pd.DataFrame(rows)
    df.to_csv(args.out, sep='\t', index=False, float_format='%.4g')
    for t, g in df.groupby('test'):
        print(f'\n=== {t}')
        print(g.drop(columns='test').to_string(index=False))


if __name__ == '__main__':
    main()
