#!/usr/bin/env python
"""Where does DNA help? Compare TAD boundaries called (cooltools, prominence >= 0.2, the paper's protocol) by the full
model and by its Hi-C-only ablation, split into boundaries both call, boundaries only the DNA model calls ('added'),
and boundaries only the Hi-C-only model calls ('removed'); for each set report accuracy (within 3 bins of a ground-truth
boundary) and CTCF-motif presence (JASPAR MA0139.2, p<1e-5, either strand, within +-1 bin), per species.
Sequence: the evaluated scaffold (HiC_scaffold_1, first record) from the one-hot genome encodings used by the model.
Output: tad_ctcf.tsv.  MODE=random: the no-DNA arm is the SAME model run on random DNA (tad_candidates_random/,
evo2hic_real vs evo2hic_random) -> tad_ctcf_random.tsv"""
import os, glob, sys
from multiprocessing import Pool
import numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))  # repo root (paths.py)
import paths as P  # noqa: E402
D = str(P.HIC_DATA_ROOT)
S = str(P.RA_WORK('species'))
MODE = os.environ.get('MODE', 'hic_only')
CAND = f'{S}/tad_candidates_random' if MODE == 'random' else f'{S}/tad_candidates'
# MODE=hic_only needs the Hi-C-only boundary candidates (outputs not released); its result is shipped as results/tad_ctcf.tsv
OUTF = f'{S}/tad_ctcf_random.tsv' if MODE == 'random' else f'{S}/tad_ctcf.tsv'
MEME = str(P.MOTIF_MEME); MOTIF = 'MA0139.2_CTCF'; P_THR = 1e-5; RES = 10000

def read_pwm():
    rows, on = [], False
    for line in open(MEME):
        if line.startswith('MOTIF'): on = MOTIF in line; continue
        if on and line.strip() and line.strip()[0].isdigit(): rows.append([float(x) for x in line.split()])
        elif on and rows and (line.startswith('URL') or line.startswith('MOTIF')): break
    p = np.array(rows); p = (p + 1e-3) / (p + 1e-3).sum(1, keepdims=True); return np.log2(p / 0.25)

def null_threshold(lo, p_thr, step=0.01):
    q = np.round(lo / step).astype(int); dist = {0: 1.0}
    for k in range(q.shape[0]):
        nd = {}
        for s, pr in dist.items():
            for b in range(4): nd[s + q[k, b]] = nd.get(s + q[k, b], 0.0) + pr * 0.25
        dist = nd
    keys = np.array(sorted(dist)); pr = np.array([dist[k] for k in keys]); tail = np.cumsum(pr[::-1])[::-1]
    return float(keys[np.searchsorted(-tail, -p_thr)] * step)

LO = read_pwm(); THR = null_threshold(LO, P_THR); W = LO.shape[0]
LO4 = np.vstack([LO.T, np.full((1, W), -10.0)]).astype(np.float32); COMP = np.array([3, 2, 1, 0, 4], np.uint8)

def motif_bins(sp, L):
    enc = np.memmap(f'{D}/data/dna/multispecies/{sp}.fasta.encoding', dtype=np.float16, mode='r').reshape(-1, 4)
    nb = (L + RES - 1) // RES; hits = np.zeros(nb, np.int32); CH = 5_000_000
    for s in range(0, L, CH):
        e = min(L, s + CH + W - 1); x = np.asarray(enc[s:e], dtype=np.float32)
        seq = np.where(x.sum(1) > 0, x.argmax(1), 4).astype(np.uint8); n = len(seq) - W + 1
        if n <= 0: continue
        f = np.zeros(n, np.float32); r = np.zeros(n, np.float32); sc = COMP[seq]
        for k in range(W):
            f += LO4[seq[k:k + n], k]; r += LO4[sc[k:k + n], W - 1 - k]
        pos = s + np.where((f >= THR) | (r >= THR))[0]
        np.add.at(hits, pos // RES, 1)
    return hits

def near(a, b):
    if len(b) == 0: return np.full(len(a), 10**9)
    b = np.sort(b); j = np.searchsorted(b, a)
    return np.minimum(np.abs(a - b[np.clip(j - 1, 0, len(b) - 1)]), np.abs(a - b[np.clip(j, 0, len(b) - 1)]))

def one(sp):
    try:
        z = np.load(f'{CAND}/{sp}.npz')
        cs = pd.read_csv(f'{D}/dnazoo/raw_hic_fixed/{sp}.chrom.size', sep='\t', header=None)
        L = int(cs.iloc[0, 1])
        hits = motif_bins(sp, L); mb = hits > 0
        has = lambda b: np.array([mb[max(0, i - 1):i + 2].any() for i in b]) if len(b) else np.array([], bool)
        gt = z['gt_idx'][z['gt_bs'] >= 0.2]
        if MODE == 'random': E = z['evo2hic_real_idx'][z['evo2hic_real_bs'] >= 0.2]; H = z['evo2hic_random_idx'][z['evo2hic_random_bs'] >= 0.2]
        else: E = z['evo2hic_idx'][z['evo2hic_bs'] >= 0.2]; H = z['hic_only_idx'][z['hic_only_bs'] >= 0.2]
        both = E[near(E, H) <= 1]; added = E[near(E, H) > 1]; removed = H[near(H, E) > 1]
        r = {'species': sp, 'L_Mb': L / 1e6, 'motif_bin_frac': mb.mean(), 'n_gt': len(gt),
             'gt_motif': has(gt).mean() if len(gt) else np.nan}
        for name, b in [('both', both), ('added', added), ('removed', removed)]:
            r[f'n_{name}'] = len(b)
            r[f'acc_{name}'] = (near(b, gt) <= 3).mean() if len(b) else np.nan
            r[f'motif_{name}'] = has(b).mean() if len(b) else np.nan
        return r
    except Exception as e:
        print(f'[fail] {sp}: {type(e).__name__} {e}', flush=True); return None

if __name__ == '__main__':
    print(f'CTCF threshold log-odds {THR:.2f} for p<{P_THR}', flush=True)
    sps = [os.path.basename(f)[:-4] for f in sorted(glob.glob(f'{CAND}/*.npz'))]
    rows = []
    with Pool(int(os.environ.get('NPROC', '8'))) as pool:
        for r in pool.imap_unordered(one, sps):
            if r is None: continue
            rows.append(r); pd.DataFrame(rows).to_csv(OUTF, sep='\t', index=False)
            print(f"[{len(rows)}] {r['species'][:36]:36s} bg {r['motif_bin_frac']:.2f} gt {r['gt_motif']:.2f} "
                  f"added acc/motif {r['acc_added']:.2f}/{r['motif_added']:.2f} removed {r['acc_removed']:.2f}/{r['motif_removed']:.2f}", flush=True)
    print('done', len(rows))
