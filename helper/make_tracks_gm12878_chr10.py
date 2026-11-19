#!/usr/bin/env python
"""GM12878 epigenomic tracks on chr10 at 2 kb (TRACKS_GM12878_CHR10_NPY; Fig. 2, Supp. Figs 2-3).

The tracks are the ENCODE bigWigs listed in `data/hic2track/index.tsv` of the Evo2HiC repo (GM12878: DNase
ENCFF960FMM, CTCF ENCFF644EEX, H3K27ac ENCFF469WVA, H3K27me3 ENCFF486WAD, H3K4me3 ENCFF280PUF), read with the
Evo2HiC Track_Loader (mean per 2 kb bin, cut off at 20x the 90th percentile and scaled to [0, 1]).
Output: array of shape (5, 66899) in the order DNase, CTCF, H3K27ac, H3K27me3, H3K4me3.

Usage (from this repo's root, Evo2HiC repo root on PYTHONPATH): python -m helper.make_tracks_gm12878_chr10
"""
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # repo root (paths.py)
import paths as P  # noqa: E402
P.add_repo_to_syspath()
from config import hic2tarck_dir  # noqa: E402
from dataset.track_loader import Track_Loader  # noqa: E402

if __name__ == '__main__':
    t = Track_Loader(os.path.join(hic2tarck_dir, 'GM12878'), 2000)
    lo, hi = t.chr_pos['chr10' if 'chr10' in t.chr_pos else '10']
    out = P.TRACKS_GM12878_CHR10_NPY
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, np.asarray(t.values[:, lo:hi]))
    print(f'wrote {out} {np.asarray(t.values[:, lo:hi]).shape}')
