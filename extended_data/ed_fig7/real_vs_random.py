#!/usr/bin/env python
"""Pixel PCC / SPC (general_eval diagonal metric) of the Evo2HiC SR checkpoint on the real genome vs on random DNA,
per species. Inputs: <SR_EVO2HIC_DIR>/eval_results_diagonal_10000/results.tsv and
dna_ablation/random/eval_results_diagonal_10000/results.tsv (evaluate.batch_evaluate.general_eval_batch).
Output: dna_ablation/real_vs_random.tsv"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))  # repo root (paths.py)
import paths as P  # noqa: E402
import pandas as pd
S = str(P.RA_WORK('species'))
real = pd.read_csv(os.path.join(P.SR_EVO2HIC_DIR, 'eval_results_diagonal_10000', 'results.tsv'), sep='\t')
rand = pd.read_csv(f'{S}/dna_ablation/random/eval_results_diagonal_10000/results.tsv', sep='\t')
d = real[['species', 'PCC', 'SPC']].rename(columns={'PCC': 'PCC_real', 'SPC': 'SPC_real'}).merge(
    rand[['species', 'PCC', 'SPC']].rename(columns={'PCC': 'PCC_rand', 'SPC': 'SPC_rand'}), on='species')
d.to_csv(f'{S}/dna_ablation/real_vs_random.tsv', sep='\t', index=False)
print('real_vs_random.tsv:', len(d), 'species')
