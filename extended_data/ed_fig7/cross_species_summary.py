#!/usr/bin/env python
"""Consolidated cross-species numbers for the manuscript (paper protocol only).
Per clade: pixel PCC, TAD F1 (cooltools, prominence >= 0.2, +-3 bins) and threshold-free boundary AUPRC for
Evo2HiC, Hi-C-only, Evo2HiC on random DNA (same model), HiCARN2, HiCFoundation; wins of Evo2HiC vs each no-DNA arm;
covariates for the real-vs-random pixel gain (sequencing depth, divergence).
Outputs: cross_species_clade.tsv, cross_species_per_species.tsv (Supplementary Data 2)."""
import numpy as np, pandas as pd
from scipy.stats import wilcoxon, spearmanr
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))  # repo root (paths.py)
import paths as P  # noqa: E402

S = str(P.RA_WORK('species'))
t = pd.read_csv(f'{S}/species_table.tsv', sep='\t')
t.loc[t.species == 'Phlebotomus_papatasi', 'clade'] = 'Protostomes'   # mislabelled sand fly (its divergence value is correct)
rr = pd.read_csv(f'{S}/dna_ablation/real_vs_random.tsv', sep='\t')[['species', 'PCC_real', 'PCC_rand', 'SPC_rand']]
tr = pd.read_csv(f'{S}/tad_random.tsv', sep='\t')[['species', 'F1_fixed_evo2hic_random']]
ap = pd.read_csv(f'{S}/boundary_auprc_all_methods.tsv', sep='\t').drop(columns=['clade']).add_prefix('AP_').rename(columns={'AP_species': 'species'})
d = t.merge(rr, on='species', how='left').merge(tr, on='species', how='left').merge(ap, on='species', how='left')
d = d.rename(columns={'PCC_rand': 'PCC_random', 'F1_fixed_evo2hic_random': 'TAD_f1_random', 'AP_evo2hic_randomDNA': 'AP_random'})
M = {'Evo2HiC': 'evo2hic', 'Hi-C-only': 'hic_only', 'random DNA': 'random', 'HiCARN2': 'hicarn2', 'HiCFoundation': 'hicfoundation'}
MET = {'pixel PCC': 'PCC_{}', 'TAD F1': 'TAD_f1_{}', 'boundary AUPRC': 'AP_{}'}


def pw(a, b):
    x = (a - b).dropna()
    try:
        return wilcoxon(x).pvalue if len(x) > 1 and (x != 0).any() else np.nan
    except ValueError:
        return np.nan


rows = []
for g, x in list(d.groupby('clade')) + [('All', d)]:
    rec = {'clade': g, 'n': len(x)}
    for mn, pat in MET.items():
        for lab, k in M.items():
            rec[f'{mn} | {lab}'] = x[pat.format(k)].mean()
        for lab in ('Hi-C-only', 'random DNA'):
            a, b = x[pat.format('evo2hic')], x[pat.format(M[lab])]
            ok = a.notna() & b.notna()
            rec[f'{mn} | wins vs {lab}'] = f'{int((a[ok] > b[ok]).sum())}/{int(ok.sum())}'
            rec[f'{mn} | p vs {lab}'] = pw(a, b)
    rows.append(rec)
clade = pd.DataFrame(rows)
clade.to_csv(f'{S}/cross_species_clade.tsv', sep='\t', index=False, float_format='%.4g')

d['dPCC_vs_random'] = d.PCC_evo2hic - d.PCC_random
mam = d[d.clade == 'Mammalia']
print('covariates of the real-vs-random pixel gain:')
for lab, x in (('all', d), ('Mammalia', mam)):
    ok = x.log10_depth.notna() & x.dPCC_vs_random.notna()
    r1 = spearmanr(x.log10_depth[ok], x.dPCC_vs_random[ok])
    ok2 = x.log10_mya.notna() & x.dPCC_vs_random.notna()
    r2 = spearmanr(x.log10_mya[ok2], x.dPCC_vs_random[ok2])
    print(f'  {lab:9s} depth rho={r1.correlation:+.2f} (p={r1.pvalue:.2g}, n={ok.sum()})   divergence rho={r2.correlation:+.2f} (p={r2.pvalue:.2g}, n={ok2.sum()})')
ok = d.log10_depth.notna()
for m in ('evo2hic', 'hic_only', 'hicarn2', 'hicfoundation'):
    print(f'  depth vs absolute PCC {m}: rho={spearmanr(d.log10_depth[ok], d["PCC_" + m][ok]).correlation:+.2f}')

keep = ['species', 'clade', 'mya', 'contacts_per_bin', 'genome_len', 'n_bins', 'n_bounds_gt'] + \
       [MET[m].format(M[l]) for m in MET for l in M]
d[keep].to_csv(f'{S}/cross_species_per_species.tsv', sep='\t', index=False, float_format='%.4g')

pd.set_option('display.width', 300)
show = ['clade', 'n'] + [c for c in clade.columns if '|' in c and 'p vs' not in c and ('HiCARN2' not in c or 'wins' in c)]
print(clade[show].round(3).to_string(index=False))
