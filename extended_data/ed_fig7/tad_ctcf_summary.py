#!/usr/bin/env python
"""Paired DNA vs no-DNA boundary comparison, summarised by clade and placental superorder.
Arms: no-DNA = Hi-C-only ablation (tad_ctcf.tsv) and no-DNA = same model on random DNA (tad_ctcf_random.tsv).
For boundaries added vs removed by DNA: CTCF-motif fraction and accuracy; p = paired Wilcoxon over species.
Output: tad_ctcf_summary.tsv"""
import numpy as np, pandas as pd
from scipy.stats import wilcoxon
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))  # repo root (paths.py)
import paths as P  # noqa: E402

S = str(P.RA_WORK('species'))
meta = pd.read_csv(f'{S}/species_table.tsv', sep='\t')[['species', 'clade', 'mya']]
meta.loc[meta.species == 'Phlebotomus_papatasi', 'clade'] = 'Protostomes'   # mislabelled sand fly


def group(r):
    if r.clade != 'Mammalia':
        return r.clade
    m = r.mya                                   # patristic distance to human (2x divergence time)
    if pd.isna(m):
        return 'Mammalia: undated'
    return ('Mammalia: primates' if m <= 147.6 else 'Mammalia: Glires' if m <= 174.5
            else 'Mammalia: Laurasiatheria' if m <= 188.1 else 'Mammalia: Afrotheria/Xenarthra')


def p(a, b):
    x = (a - b).dropna()
    try:
        return wilcoxon(x).pvalue if len(x) > 1 and (x != 0).any() else np.nan
    except ValueError:
        return np.nan


rows = []
for arm, f in (('Hi-C-only', 'tad_ctcf.tsv'), ('random DNA', 'tad_ctcf_random.tsv')):
    d = pd.read_csv(P.RA_TABLE('species', f), sep='\t').merge(meta, on='species')
    d['grp'] = d.apply(group, axis=1)
    sets = [(g, x) for g, x in d.groupby('grp')] + [('Mammalia (all)', d[d.clade == 'Mammalia'])]
    for g, x in sets:
        rows.append({'no-DNA arm': arm, 'group': g, 'n': len(x),
                     'motif added': x.motif_added.mean(), 'motif removed': x.motif_removed.mean(),
                     'added>removed (motif)': f'{int((x.motif_added > x.motif_removed).sum())}/{len(x)}',
                     'p motif': p(x.motif_added, x.motif_removed),
                     'acc added': x.acc_added.mean(), 'acc removed': x.acc_removed.mean(),
                     'added>removed (acc)': f'{int((x.acc_added > x.acc_removed).sum())}/{len(x)}',
                     'p acc': p(x.acc_added, x.acc_removed),
                     'n added': x.n_added.mean(), 'n removed': x.n_removed.mean(),
                     'motif GT': x.gt_motif.mean(), 'motif background': x.motif_bin_frac.mean()})
out = pd.DataFrame(rows)
out.to_csv(f'{S}/tad_ctcf_summary.tsv', sep='\t', index=False, float_format='%.4g')
pd.set_option('display.width', 250)
print(out.round(3).to_string(index=False))
