#!/usr/bin/env python
"""[candidate dump variant] Count-matched TAD-boundary F1 across the 177 DNA Zoo species.

Same pipeline as evaluate/multi_TAD_eval_cooltools.py (cooltools insulation, window 100 kb,
ignore_diags 2, find_boundaries, F1 with a 3-bin boundary zone) and the SAME ground-truth boundaries (prominence
>= 0.2 on the ground-truth map). The only change: instead of a fixed prominence cutoff on each prediction, every
method contributes its top-k boundary candidates ranked by boundary strength, with k = the number of ground-truth
boundaries of that species. This removes the dependence of F1 on how much contrast a method's output keeps (with
a fixed cutoff, a smoother map passes fewer boundaries and loses recall). The fixed-cutoff F1 is recomputed
alongside as a reproduction check against TAD_multi_4methods.tsv.
Output: tad_dump_done.tsv and tad_candidates/<species>.npz
"""
import os, sys, tempfile, time
from multiprocessing import Pool
import numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))  # repo root (paths.py)
import paths as P  # noqa: E402
R = str(P.REPO)
sys.path.insert(0, R)
os.chdir(R)
import cooler
from cooltools.api.insulation import calculate_insulation_score, find_boundaries
from config import dnazoo_index, dnazoo_hic_dir
from evaluate import multi_TAD_eval_cooltools as T

OUT = os.path.join(P.RA_WORK('species'), 'tad_dump_done.tsv')
DUMP = os.path.join(P.RA_WORK('species'), 'tad_candidates')
os.makedirs(DUMP, exist_ok=True)
RES = 10000

def candidates(M, td, name):
    cp = os.path.join(td, f'{name}.cool')
    T.write_cool_from_matrix(M, RES, name, cp)
    t = find_boundaries(calculate_insulation_score(cooler.Cooler(cp), window_bp=[T.WINDOW_BP],
                                                   ignore_diags=2, clr_weight_name=None))
    w = str(T.WINDOW_BP)
    bs = t[next(c for c in t.columns if c.startswith('boundary_strength') and c.endswith(w))].to_numpy(float)
    idx = np.where(np.isfinite(bs))[0]
    return idx[np.argsort(-bs[idx])], bs[idx][np.argsort(-bs[idx])]      # all peaks, strongest first

def one(species):
    gt_path = os.path.join(dnazoo_hic_dir, species + '.hic')
    paths = {m: os.path.join(d, species + '_enhanced.hic') for m, d in T.METHODS.items()}
    if not os.path.isfile(gt_path) or any(not os.path.isfile(p) for p in paths.values()):
        return None
    t0 = time.time()
    try:
        mgt, norm = T.load_matrix(gt_path, RES)
        with tempfile.TemporaryDirectory() as td:
            gi, gs = candidates(mgt, td, 'gt')
            gt_b = sorted(gi[gs >= T.PROMINENCE_CUT].tolist())                 # unchanged ground truth
            k = len(gt_b)
            row = {'species': species, 'n_bounds_gt': k, 'n_bins': mgt.shape[0], 'norm_used': norm}
            dump = {'gt_idx': gi, 'gt_bs': gs}
            for m, p in paths.items():
                mp, _ = T.load_matrix(p, RES)
                if mp.shape != mgt.shape:
                    return None
                pi, ps = candidates(mp, td, 'p_' + m)
                dump[m + '_idx'] = pi; dump[m + '_bs'] = ps
                fixed = sorted(pi[ps >= T.PROMINENCE_CUT].tolist())
                topk = sorted(pi[:k].tolist())
                row[f'F1_fixed_{m}'] = float(T.boundary_f1(fixed, gt_b))
                row[f'F1_topk_{m}'] = float(T.boundary_f1(topk, gt_b))
                row[f'n_fixed_{m}'] = len(fixed); row[f'n_cand_{m}'] = len(pi)
        np.savez_compressed(os.path.join(DUMP, species + '.npz'), **dump)
        row['sec'] = round(time.time() - t0, 1)
        return row
    except Exception as e:
        print(f'[fail] {species}: {type(e).__name__} {e}', flush=True)
        return None

if __name__ == '__main__':
    sp = [l.strip() for l in open(dnazoo_index) if l.strip()]
    done = set(pd.read_csv(OUT, sep='\t')['species']) if os.path.isfile(OUT) else set()
    todo = [s for s in sp if s not in done]
    rows = pd.read_csv(OUT, sep='\t').to_dict('records') if done else []
    print(f'{len(todo)} species to score ({len(done)} done)', flush=True)
    with Pool(int(os.environ.get('NPROC', '12'))) as pool:
        for r in pool.imap_unordered(one, todo):
            if r is None: continue
            rows.append(r); pd.DataFrame(rows).to_csv(OUT, sep='\t', index=False)
            print(f"[{len(rows)}] {r['species'][:40]:40s} k={r['n_bounds_gt']:5d} "
                  + ' '.join(f"{m}:{r['F1_topk_'+m]:.3f}" for m in T.METHODS), flush=True)
    print('done', len(rows))
