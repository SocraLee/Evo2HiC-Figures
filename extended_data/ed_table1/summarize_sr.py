#!/usr/bin/env python
"""Merge HiCFoundation SR metrics with the curated Evo2HiC/baseline tables
(result/{human,mouse}/{PCC,SPC,PSNR,SSIM}.csv, result/multi/SPC.csv of this folder)
and report per-dataset means, win counts and paired one-sided Wilcoxon
(Evo2HiC > HiCFoundation) — same test family as Fig. 6 / ED Fig. 6.

Usage: summarize_sr.py  -> writes summary_sr.tsv, prints tables.
"""
import os

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))  # repo root (paths.py)
import paths as P  # noqa: E402

W = str(P.RA_WORK('hicfoundation'))
R = str(P.RESULT('extended_data/ed_fig6'))   # per-map tables of ED Fig 6 (human / mouse)
CK = str(P.CKPT_ROOT)
MULTI_BASE = {  # per-species diagonal/chrom results.tsv for the multi-species panel
    'Evo2HiC':  str(P.SR_EVO2HIC_DIR),
    'HiC-only': str(P.SR_HIC_ONLY_DIR),
    'HiCARN2':  str(P.SR_HICARN2_DIR),
}


def hf_table(sp):
    d = f'{W}/{sp}' if sp != 'multi' else f'{W}/multi_10000'
    diag = pd.read_csv(f'{d}/eval_results_diagonal_10000/results.tsv', sep='\t')
    chrom = pd.read_csv(f'{d}/eval_results_chrom_10000/results.tsv', sep='\t')
    key = diag.columns[0]
    df = diag.merge(chrom, on=key, how='outer')     # DNA Zoo: chrom metrics exist for a subset of species
    return df.set_index(key)[['PCC', 'SPC', 'PSNR', 'SSIM']]


def shipped(sp, metric):
    # per-map / per-species values shipped in results/ (baselines whose outputs are not released)
    return pd.read_csv(P.RA_SHIPPED('hicfoundation', f'table_{sp}_{metric}.tsv'), sep='\t', index_col=0)


def curated(sp, metric):
    p = f'{R}/{sp}/{metric}.csv'
    if not os.path.exists(p):
        return shipped(sp, metric).drop(columns=['HiCFoundation'])
    df = pd.read_csv(p, sep='\t').set_index(pd.read_csv(p, sep='\t').columns[0])
    df = df.rename(columns={'Unet': 'HiC-only', 'EvoHiC': 'Evo2HiC'})
    return df


def multi_base(metric):
    task = 'diagonal' if metric in ('PCC', 'SPC') else 'chrom'
    out = {}
    for m, d in MULTI_BASE.items():
        if not os.path.isdir(d):
            out[m] = shipped('multi', metric)[m]
            continue
        cand = [x for x in os.listdir(d) if x.startswith(f'eval_results_{task}')]
        # prefer the canonical dir name; the hic_only diagonal dir has a timestamp suffix
        cand.sort(key=lambda x: (x != f'eval_results_{task}_10000', x))
        t = pd.read_csv(f'{d}/{cand[0]}/results.tsv', sep='\t').set_index('species')
        out[m] = t[metric]
    return pd.DataFrame(out)


rows = []
for sp in ['human', 'mouse', 'multi']:
    hf = hf_table(sp)
    for metric in ['PCC', 'SPC', 'PSNR', 'SSIM']:
        base = curated(sp, metric) if sp != 'multi' else multi_base(metric)
        if base is None:
            continue
        df = base.join(hf[metric].rename('HiCFoundation'), how='inner')
        df = df.dropna()
        cols = [c for c in ['Evo2HiC', 'HiC-only', 'HiCARN2', 'HiCFoundation'] if c in df.columns]
        n = len(df)
        d = df['Evo2HiC'] - df['HiCFoundation']
        try:
            p_gt = wilcoxon(d, alternative='greater').pvalue
            p_lt = wilcoxon(d, alternative='less').pvalue
        except ValueError:
            p_gt = p_lt = np.nan
        rec = {'dataset': sp, 'metric': metric, 'n': n,
               **{f'mean_{c}': df[c].mean() for c in cols},
               'wins_Evo2HiC_vs_HF': int((d > 0).sum()),
               'wins_HF_vs_Evo2HiC': int((d < 0).sum()),
               'p_Evo2HiC>HF': p_gt, 'p_HF>Evo2HiC': p_lt}
        rows.append(rec)
        df.to_csv(f'{W}/table_{sp}_{metric}.tsv', sep='\t', float_format='%.4f')

summ = pd.DataFrame(rows)
summ.to_csv(f'{W}/summary_sr.tsv', sep='\t', index=False, float_format='%.4g')
pd.set_option('display.width', 200)
print(summ.to_string(index=False, float_format=lambda x: f'{x:.4g}'))
