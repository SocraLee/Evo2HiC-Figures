#!/usr/bin/env python
"""Extended Data Fig. 7: systematic cross-species analysis of the sequence prior.
Per species (177 DNA Zoo benchmarks, 1/16-downsampled input @10 kb): pixel PCC/SPC, insulation-profile PCC and
cooltools TAD-boundary F1 for Evo2HiC, Evo2HiC(HiC-only ablation), HiCARN2, HiCFoundation; covariates: clade,
patristic distance to human (MYA, TimeTree tree of round 1; 154/177 have one), sequencing depth (raw intra
contacts per 10 kb bin of the ground-truth .hic), genome length, #bins, #GT boundaries.
Outputs: species_table.tsv (supplementary table), clade_summary.tsv, regression.tsv, species_analysis.png"""
import numpy as np, pandas as pd
from scipy.stats import wilcoxon, spearmanr
import statsmodels.formula.api as smf
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))  # repo root (paths.py)
import paths as P  # noqa: E402
S = str(P.RA_WORK('species')); H = str(P.RA_WORK('hicfoundation'))
T = pd.read_csv(P.RA_TABLE('hicfoundation', 'TAD_multi_4methods.tsv'), sep='\t')
_s = pd.read_csv(P.RA_SHIPPED('hicfoundation', 'TAD_multi_4methods.tsv'), sep='\t')   # baseline columns (outputs not released)
T = T.merge(_s[['species'] + [c for c in _s.columns if c not in T.columns and c != 'clade']], on='species', how='left')
if 'clade' not in T:                         # clade of each species: data/clade/*_clean.txt lists
    sys.path.insert(0, str(P.PLOT_DIR)); from _supp_style import load_clade_map, assign_clade
    _m = load_clade_map(P.CLADE_DIR); T['clade'] = T.species.apply(lambda s: assign_clade(s, _m))
meth = {'evo2hic': 'Evo2HiC', 'hic_only': 'HiC-only', 'hicarn2': 'HiCARN2', 'hicfoundation': 'HiCFoundation'}
df = T[['species', 'clade', 'n_bins', 'n_bounds_gt'] + [f'TAD_f1_{m}' for m in meth] + [f'IS_pcc_{m}' for m in meth] + [f'n_bounds_{m}' for m in meth]].copy()
for metric in ['PCC', 'SPC']:
    p = pd.read_csv(P.RA_TABLE('hicfoundation', f'table_multi_{metric}.tsv'), sep='\t').rename(columns={v: f'{metric}_{k}' for k, v in meth.items()})
    df = df.merge(p, on='species', how='left')
df = df.merge(pd.read_csv(f'{S}/species_mya.tsv', sep='\t'), on='species', how='left')
dep = pd.read_csv(f'{S}/species_depth.tsv', sep='\t')[['species', 'n_scaffolds', 'genome_len', 'total_intra', 'contacts_per_bin', 'contacts_per_bin_2Mb']]
df = df.merge(dep, on='species', how='left'); df['clade'] = df['clade'].astype(str).astype(object)
df['log10_depth'] = np.log10(df.contacts_per_bin); df['log10_genome_Mb'] = np.log10(df.genome_len / 1e6); df['log10_mya'] = np.log10(df.mya)
for metric in ['PCC', 'SPC', 'IS_pcc', 'TAD_f1']:
    for m in ['hic_only', 'hicarn2', 'hicfoundation']:
        df[f'd{metric}_vs_{m}'] = df[f'{metric}_evo2hic'] - df[f'{metric}_{m}']
df.to_csv(f'{S}/species_table.tsv', sep='\t', index=False, float_format='%.4f')
print(f'species table: {len(df)} species; mya available {df.mya.notna().sum()}; depth available {df.contacts_per_bin.notna().sum()}')
print(f'depth (contacts / 10 kb bin): median {df.contacts_per_bin.median():.0f}, IQR {df.contacts_per_bin.quantile(.25):.0f}-{df.contacts_per_bin.quantile(.75):.0f}, range {df.contacts_per_bin.min():.0f}-{df.contacts_per_bin.max():.0f}')
# ---- A. sequence-prior contribution per clade
rows = []
order = ['Mammalia', 'Marsupialia', 'Reptilia', 'Actinopterygii', 'Protostomes', 'Angiosperms']
rng = np.random.default_rng(0)
for metric in ['PCC', 'SPC', 'IS_pcc', 'TAD_f1']:
    for base in ['hic_only', 'hicfoundation', 'hicarn2']:
        col = f'd{metric}_vs_{base}'
        for cl in ['all'] + order:
            sub = df if cl == 'all' else df[df.clade == cl]
            d = sub[col].dropna().values
            if len(d) == 0: continue
            boots = [rng.choice(d, len(d)).mean() for _ in range(2000)] if len(d) > 1 else [d.mean()]
            p = wilcoxon(d).pvalue if len(d) >= 6 and (d != 0).any() else np.nan
            rows.append(dict(metric=metric, baseline=base, clade=cl, n=len(d), mean_delta=d.mean(), ci_lo=np.percentile(boots, 2.5), ci_hi=np.percentile(boots, 97.5),
                             frac_positive=(d > 0).mean(), wilcoxon_p=p, evo2hic_mean=sub[f'{metric}_evo2hic'].mean(), baseline_mean=sub[f'{metric}_{base}'].mean()))
cs = pd.DataFrame(rows); cs.to_csv(f'{S}/clade_summary.tsv', sep='\t', index=False, float_format='%.4f')
pd.set_option('display.width', 220)
print('\n== A. Evo2HiC minus Hi-C-only ablation (the sequence prior), per clade')
print(cs[cs.baseline == 'hic_only'].pivot_table(index='clade', columns='metric', values=['mean_delta', 'frac_positive', 'wilcoxon_p'], aggfunc='first').reindex(['all'] + order).round(3).to_string())
print('\n== A2. Evo2HiC minus HiCFoundation, per clade (mean delta / frac positive)')
print(cs[cs.baseline == 'hicfoundation'].pivot_table(index='clade', columns='metric', values=['mean_delta', 'frac_positive'], aggfunc='first').reindex(['all'] + order).round(3).to_string())
# ---- B. confounders: absolute performance and delta vs depth / distance
print('\n== B. Spearman correlations (n=177 unless mya: 154)')
for metric in ['PCC', 'SPC', 'IS_pcc', 'TAD_f1']:
    line = f'   {metric:7s}'
    for col, lab in [(f'{metric}_evo2hic', 'abs Evo2HiC'), (f'{metric}_hic_only', 'abs HiC-only'), (f'd{metric}_vs_hic_only', 'delta vs HiC-only'), (f'd{metric}_vs_hicfoundation', 'delta vs HF')]:
        r1 = spearmanr(df[col], df.log10_depth, nan_policy='omit')[0]; r2 = spearmanr(df[col], df.mya, nan_policy='omit')[0]
        line += f' | {lab}: depth rho={r1:+.2f}, mya rho={r2:+.2f}'
    print(line)
print('\n== C. OLS: delta(Evo2HiC - HiC-only) ~ log10_depth + log10_genome_Mb + log10_mya   (species with mya, n=%d)' % df.mya.notna().sum())
reg = []
for metric in ['PCC', 'SPC', 'IS_pcc', 'TAD_f1']:
    for base in ['hic_only', 'hicfoundation']:
        y = f'd{metric}_vs_{base}'
        m = smf.ols(f'{y} ~ log10_depth + log10_genome_Mb + log10_mya', data=df.dropna(subset=[y, 'mya', 'log10_depth'])).fit()
        m2 = smf.ols(f'{y} ~ log10_depth + log10_genome_Mb + C(clade)', data=df.dropna(subset=[y, 'log10_depth'])).fit()
        reg.append(dict(metric=metric, baseline=base, n=int(m.nobs), r2=m.rsquared, **{f'{k}_coef': m.params[k] for k in ['log10_depth', 'log10_genome_Mb', 'log10_mya']}, **{f'{k}_p': m.pvalues[k] for k in ['log10_depth', 'log10_genome_Mb', 'log10_mya']}, r2_clade_model=m2.rsquared, clade_F_p=m2.f_pvalue))
        print(f'   {metric:7s} vs {base:13s} n={int(m.nobs)} R2={m.rsquared:.3f} | depth b={m.params["log10_depth"]:+.4f} p={m.pvalues["log10_depth"]:.2g} | genome b={m.params["log10_genome_Mb"]:+.4f} p={m.pvalues["log10_genome_Mb"]:.2g} | log10 MYA b={m.params["log10_mya"]:+.4f} p={m.pvalues["log10_mya"]:.2g}')
pd.DataFrame(reg).to_csv(f'{S}/regression.tsv', sep='\t', index=False, float_format='%.4g')
print('\n== D. absolute performance ~ depth (all four methods, Spearman with log10 depth):')
for metric in ['PCC', 'TAD_f1']:
    print(f'   {metric}: ' + '  '.join(f'{lab}={spearmanr(df[f"{metric}_{m}"], df.log10_depth, nan_policy="omit")[0]:+.2f}' for m, lab in meth.items()))
# ---- figure
fig, axes = plt.subplots(2, 4, figsize=(16, 7))
for j, metric in enumerate(['PCC', 'SPC', 'IS_pcc', 'TAD_f1']):
    ax = axes[0, j]; col = f'd{metric}_vs_hic_only'
    for i, cl in enumerate(order):
        d = df[df.clade == cl][col].dropna().values
        ax.scatter(np.full(len(d), i) + rng.uniform(-.2, .2, len(d)), d, s=8, alpha=.6, c=np.where(d > 0, '#c0392b', '#2980b9'))
        ax.plot([i - .3, i + .3], [d.mean()] * 2, 'k-', lw=1.5)
    ax.axhline(0, color='0.5', lw=.5); ax.set_xticks(range(len(order))); ax.set_xticklabels([c[:6] for c in order], rotation=45, fontsize=7)
    ax.set_title(f'Δ{metric}: Evo2HiC − Hi-C-only', fontsize=9)
    ax = axes[1, j]
    sc = ax.scatter(df.log10_depth, df[col], c=df.mya, cmap='viridis', s=10)
    ax.axhline(0, color='0.5', lw=.5); ax.set_xlabel('log10 contacts per 10 kb bin'); ax.set_ylabel(f'Δ{metric}'); ax.set_title('vs depth (colour = MYA)', fontsize=9)
plt.colorbar(sc, ax=axes[1, 3], label='patristic distance (MYA)'); plt.tight_layout(); plt.savefig(f'{S}/species_analysis.png', dpi=120)
print('figure -> species_analysis.png')
