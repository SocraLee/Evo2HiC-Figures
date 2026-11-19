#!/usr/bin/env python
"""Per-species sequencing depth of the DNA Zoo benchmark .hic files (raw_hic_fixed, the exact files used as
ground truth): total intra-scaffold raw contacts at 10 kb, number of scaffolds, genome length, contacts per
10 kb bin and per bin-pair within 2 Mb.  Output: species_depth.tsv"""
import os, sys, time
import numpy as np, pandas as pd, hicstraw
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))  # repo root (paths.py)
import paths as P  # noqa: E402
D = str(P.DNAZOO_RAW_HIC_DIR)
T = pd.read_csv(P.RA_TABLE('hicfoundation', 'TAD_multi_4methods.tsv'), sep='\t')
rows = []
for k, sp in enumerate(T.species):
    f = f'{D}/{sp}.hic'; t0 = time.time()
    try:
        h = hicstraw.HiCFile(f)
        chroms = [(c.name, c.length) for c in h.getChromosomes() if c.name.lower() != 'all']
        tot = 0; near = 0; nbins = 0
        for c, L in chroms:
            m = h.getMatrixZoomData(c, c, 'observed', 'NONE', 'BP', 10000)
            recs = m.getRecords(0, L, 0, L)
            cnt = np.fromiter((r.counts for r in recs), float, len(recs)); d = np.fromiter((abs(r.binX - r.binY) for r in recs), float, len(recs))
            tot += cnt.sum(); near += cnt[d <= 2_000_000].sum(); nbins += L // 10000 + 1
        rows.append(dict(species=sp, n_scaffolds=len(chroms), genome_len=sum(L for _, L in chroms), n_bins_10kb=nbins,
                         total_intra=tot, intra_2Mb=near, contacts_per_bin=tot / nbins, contacts_per_bin_2Mb=near / nbins))
        print(f'[{k+1}/{len(T)}] {sp}: scaffolds={len(chroms)} len={sum(L for _, L in chroms)/1e6:.0f}Mb total={tot:.3g} per-bin={tot/nbins:.1f} ({time.time()-t0:.0f}s)', flush=True)
    except Exception as e:
        rows.append(dict(species=sp, error=str(e)[:80])); print(f'[{k+1}] {sp}: ERROR {e}', flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(P.RA_WORK('species'), 'species_depth.tsv'), sep='\t', index=False)
print('done')
