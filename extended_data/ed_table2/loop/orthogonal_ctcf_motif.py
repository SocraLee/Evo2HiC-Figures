#!/usr/bin/env python
"""Orthogonal validation of predicted CTCF calls at loop anchors with the CTCF
motif - evidence that comes from DNA sequence and a JASPAR PWM only, independent
of any ChIP-seq, DNase-seq or Hi-C signal.

Two published facts are used as the yardstick:
  * CTCF-bound loop anchors carry the CTCF motif (JASPAR MA0139.2);
  * the two motifs of a CTCF loop are in convergent orientation in ~90% of loops
    (Rao et al. 2014, Cell 159:1665, Fig. 4/5).

Motif calling: JASPAR MA0139.2 (from data/dna/motifs.meme), log-odds against a
uniform background, both strands, and the FIMO default significance threshold
p < 1e-4 computed from the PWM's exact null score distribution (dynamic
programming over discretised scores).  Scanned once on chr9 + chr10 (hg38).

For each signal source S in {measured, Evo2HiC, HiC-only, Evo2 sequence-only} an
anchor is "CTCF+" if the CTCF track of S within the HiCCUPS anchor (+-2 bins, the
same window as every other anchor analysis) exceeds S's own 95th percentile
(ENCODE's Z > 1.64 criterion, see loop_anchor_ccre.py).

  T1  motif presence: fraction of CTCF+ vs CTCF- anchors that contain >= 1 motif
      hit within +-5 kb of the anchor centre (10 kb HiCCUPS anchor); odds ratio,
      Fisher exact test.
  T2  convergence: among loops whose two anchors are both CTCF+ and both carry a
      motif, the fraction whose strongest motifs are convergent (+ strand at the
      upstream anchor, - strand at the downstream anchor); compared with loops
      not both-CTCF+, with Rao's 90% and with the 25% expected from random
      orientation (binomial tests).

Usage: orthogonal_ctcf_motif.py
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, binomtest, mannwhitneyu
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))  # repo root
import paths as P  # noqa: E402

W = str(P.RA_WORK('loop'))
D = str(P.HIC_DATA_ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from loop_anchor_eval import (T, RES, HALF, TEST_CHRS, METHODS, LOOP_DIR,  # noqa: E402
                              read_bedpe, Track_Loader, hic2tarck_dir)
from loop_anchor_ccre import Q_HIGH  # noqa: E402

FASTA = f'{D}/data/dna/human/hg38.fa'
MEME = str(P.MOTIF_MEME)
MOTIF = 'MA0139.2_CTCF'
P_THR = float(sys.argv[1]) if len(sys.argv) > 1 else 1e-4   # FIMO default 1e-4; also run 1e-5, 1e-6
ANCHOR_BP = 5000              # HiCCUPS anchor half-width
CELLS = ['GM12878', 'H1ESC', 'K562']
BASE = {'A': 0, 'C': 1, 'G': 2, 'T': 3}


# ---------------------------------------------------------------- sequence
def read_chrom(name):
    """Read one chromosome from an indexed fasta as uint8 codes (4 = N)."""
    fai = pd.read_csv(FASTA + '.fai', sep='\t', header=None,
                      names=['name', 'len', 'off', 'lb', 'll']).set_index('name')
    L, off, lb, ll = (int(fai.loc[name, c]) for c in ('len', 'off', 'lb', 'll'))
    nlines = (L + lb - 1) // lb
    with open(FASTA, 'rb') as f:
        f.seek(off)
        raw = f.read(nlines * ll)
    arr = np.frombuffer(raw, dtype=np.uint8).reshape(-1, ll)[:, :lb].ravel()[:L]
    lut = np.full(256, 4, np.uint8)
    for b, i in BASE.items():
        lut[ord(b)] = i
        lut[ord(b.lower())] = i
    return lut[arr]


# ---------------------------------------------------------------- PWM
def read_pwm():
    rows, on = [], False
    for line in open(MEME):
        if line.startswith('MOTIF'):
            on = MOTIF in line
            continue
        if on and line.strip() and line.strip()[0].isdigit():
            rows.append([float(x) for x in line.split()])
        elif on and rows and (line.startswith('URL') or line.startswith('MOTIF')):
            break
    p = np.array(rows)
    p = (p + 1e-3) / (p + 1e-3).sum(1, keepdims=True)      # tiny pseudocount
    return np.log2(p / 0.25)                                # (w, 4) log-odds


def null_threshold(lo, p_thr, step=0.01):
    """Score s such that P(score >= s) < p_thr under the uniform background,
    by exact DP on discretised scores (same principle as FIMO)."""
    q = np.round(lo / step).astype(int)
    dist = {0: 1.0}
    for k in range(q.shape[0]):
        nd = {}
        for s, pr in dist.items():
            for b in range(4):
                nd[s + q[k, b]] = nd.get(s + q[k, b], 0.0) + pr * 0.25
        dist = nd
    keys = np.array(sorted(dist))
    pr = np.array([dist[k] for k in keys])
    tail = np.cumsum(pr[::-1])[::-1]                        # P(score >= key)
    return float(keys[np.searchsorted(-tail, -p_thr)] * step)


def scan(seq, lo):
    """Best per-position log-odds on + and - strands. Returns (score+, score-)."""
    w = lo.shape[0]
    n = len(seq) - w + 1
    lo4 = np.vstack([lo.T, np.full((1, w), -10.0)]).astype(np.float32)   # N -> -10
    fwd = np.zeros(n, np.float32)
    rev = np.zeros(n, np.float32)
    comp = np.array([3, 2, 1, 0, 4], np.uint8)
    seq_c = comp[seq]
    for k in range(w):
        fwd += lo4[seq[k:k + n], k]
        rev += lo4[seq_c[k:k + n], w - 1 - k]                 # reverse complement
    return fwd, rev


# ---------------------------------------------------------------- calls
def ctcf_calls(cell):
    """anchor -> {source: CTCF+ bool}; also returns the loop table."""
    loops = read_bedpe(os.path.join(LOOP_DIR[cell], 'merged_loops.bedpe'))
    anchors = sorted({(int(r.chr), int(r.aA_center)) for r in loops.itertuples()} |
                     {(int(r.chr), int(r.aB_center)) for r in loops.itertuples()})
    calls = {}
    tl = Track_Loader(os.path.join(hic2tarck_dir, cell), resolution=RES)
    v = np.asarray(tl.values[T['CTCF']], float)
    thr = float(np.quantile(v, Q_HIGH))
    calls['Measured'] = {}
    for c, pb in anchors:
        key = f'chr{c}' if f'chr{c}' in tl.chr_pos else str(c)
        off, end = tl.chr_pos[key]
        b = pb // RES
        calls['Measured'][(c, pb)] = bool(v[off + max(0, b - HALF): off + min(end - off, b + HALF + 1)].max() > thr)
    for meth, dirf in METHODS.items():
        d = dirf(cell)
        if not all(os.path.exists(f'{d}/{ch}.npy') for ch in TEST_CHRS):
            continue
        pred = {ch: np.load(f'{d}/{ch}.npy')[T['CTCF']] for ch in TEST_CHRS}
        thr_m = float(np.quantile(np.concatenate(list(pred.values())), Q_HIGH))
        calls[meth] = {}
        for c, pb in anchors:
            b = pb // RES
            calls[meth][(c, pb)] = bool(pred[c][max(0, b - HALF): b + HALF + 1].max() > thr_m)
    return loops, anchors, calls


def main():
    lo = read_pwm()
    thr = null_threshold(lo, P_THR)
    print(f'[pwm] {MOTIF} w={lo.shape[0]} log-odds threshold for p<{P_THR}: {thr:.2f} '
          f'(max {lo.max(1).sum():.2f})', flush=True)

    hits = {}
    for ch in TEST_CHRS:
        seq = read_chrom(f'chr{ch}')
        fwd, rev = scan(seq, lo)
        hits[ch] = (fwd, rev)
        print(f'[scan] chr{ch}: {(fwd >= thr).sum()} + hits, {(rev >= thr).sum()} - hits', flush=True)

    def best_motif(c, centre):
        fwd, rev = hits[c]
        s, e = max(0, centre - ANCHOR_BP), min(len(fwd), centre + ANCHOR_BP)
        f, r = fwd[s:e], rev[s:e]
        bf, br = float(f.max()), float(r.max())
        if max(bf, br) < thr:
            return None
        return '+' if bf >= br else '-'

    def max_score(c, centre):
        fwd, rev = hits[c]
        s, e = max(0, centre - ANCHOR_BP), min(len(fwd), centre + ANCHOR_BP)
        return float(max(fwd[s:e].max(), rev[s:e].max()))

    rows = []
    for cell in CELLS:
        loops, anchors, calls = ctcf_calls(cell)
        motif = {a: best_motif(*a) for a in anchors}
        score = {a: max_score(*a) for a in anchors}
        for src, cmap in calls.items():
            # T1 motif presence
            pos = [a for a in anchors if cmap[a]]
            neg = [a for a in anchors if not cmap[a]]
            mp = sum(motif[a] is not None for a in pos)
            mn = sum(motif[a] is not None for a in neg)
            odds, p = fisher_exact([[mp, len(pos) - mp], [mn, len(neg) - mn]], alternative='greater')
            sp, sn = [score[a] for a in pos], [score[a] for a in neg]
            u, pu = mannwhitneyu(sp, sn, alternative='greater')
            rows.append({'test': 'T1_motif_presence', 'cell': cell, 'source': src,
                         'n_CTCFpos': len(pos), 'frac_motif_pos': mp / max(len(pos), 1),
                         'n_CTCFneg': len(neg), 'frac_motif_neg': mn / max(len(neg), 1),
                         'odds_ratio': odds, 'p': p,
                         'AUROC_maxscore': u / (len(sp) * len(sn)), 'p_maxscore': pu})
            # T2 convergence
            conv_both, n_both, conv_rest, n_rest = 0, 0, 0, 0
            for r in loops.itertuples():
                a, b = (int(r.chr), int(r.aA_center)), (int(r.chr), int(r.aB_center))
                if a[1] > b[1]:
                    a, b = b, a
                if motif[a] is None or motif[b] is None:
                    continue
                conv = motif[a] == '+' and motif[b] == '-'
                if cmap[a] and cmap[b]:
                    n_both += 1
                    conv_both += conv
                else:
                    n_rest += 1
                    conv_rest += conv
            if n_both >= 5:
                p25 = binomtest(conv_both, n_both, 0.25, alternative='greater').pvalue
                fe_p = fisher_exact([[conv_both, n_both - conv_both],
                                     [conv_rest, max(n_rest - conv_rest, 0)]],
                                    alternative='greater')[1] if n_rest else np.nan
                rows.append({'test': 'T2_convergence', 'cell': cell, 'source': src,
                             'n_loops_bothCTCF': n_both, 'frac_convergent': conv_both / n_both,
                             'n_loops_other': n_rest,
                             'frac_convergent_other': conv_rest / n_rest if n_rest else np.nan,
                             'p_vs_random25': p25, 'p_vs_other': fe_p})
        print(f'[{cell}] done', flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(f'{W}/orthogonal_ctcf_motif_p{P_THR:g}.tsv', sep='\t', index=False, float_format='%.4g')
    pd.set_option('display.width', 220)
    for t, g in df.groupby('test'):
        print(f'\n=== {t}')
        print(g.dropna(axis=1, how='all').drop(columns='test').to_string(index=False))


if __name__ == '__main__':
    main()
