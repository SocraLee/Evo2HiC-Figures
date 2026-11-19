#!/usr/bin/env python
"""Patristic distance of each DNA Zoo species to Homo sapiens (million years) in the TimeTree tree
(data/species_list_rawchrom.nwk), with the leaf-name mapping of the DNA Zoo figures; species without a dated
leaf are left empty (154 of 177 have one).  Output: species_mya.tsv"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))  # repo root (paths.py)
import paths as P  # noqa: E402
import numpy as np, pandas as pd
from Bio import Phylo
S = str(P.RA_WORK('species'))
SPECIES = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'ed_table1', 'species_177.txt')
SYN = {'Herpailurus_yagouaroundi': 'Puma_yagouaroundi', 'Eulemur_collaris': 'Eulemur_fulvus_collaris'}


def leaf_name(s):
    return SYN.get(s, s.split('__')[0])


tree = Phylo.read(str(P.TREE_NWK), 'newick')
leaves = {l.name: l for l in tree.get_terminals()}
human = leaves['Homo_sapiens']
species = [l.strip() for l in open(SPECIES) if l.strip()]
mya = [float(tree.distance(human, leaves[leaf_name(s)])) if leaf_name(s) in leaves else np.nan for s in species]
pd.DataFrame({'species': species, 'mya': mya}).to_csv(f'{S}/species_mya.tsv', sep='\t', index=False)
print('species_mya.tsv:', len(species), 'species,', int(np.isfinite(mya).sum()), 'with a dated leaf')
