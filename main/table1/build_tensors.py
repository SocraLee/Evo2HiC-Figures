#!/usr/bin/env python
"""Rebuild the per-window tensors for the Evo2HiC-vs-Orca comparison.

Windows: exactly Orca's Model_performance.ipynb test set - for chr9 and chr10,
pos in arange(16 Mb, chrlen-16 Mb, 500 kb) (213 + 204 = 417 windows); the 1 Mb
level-0 window starts at pos - 16 Mb + 3844*4 kb (mallstarts[i][0][0] == 3844)
and spans 250 bins at 4 kb.

Per cell (H1ESC, HFF) this writes <out>/<cell>.npz with (417, 250, 250) arrays:
  evo_raw      Evo2HiC prediction, raw-count space (model output un-normalised),
               forward+reverse-strand averaged, symmetrised - as in the paper's
               notebook (notebook_eval.py::infer_evo2hic)
  gt_raw       our ground truth: balanced counts from the Evo2HiC target .hic
               (H1ESC 4DNFI2TK7L2F, HFF 4DNFIPC7P27B) at 2 kb summed to 4 kb
  orca_logoe   Orca prediction in log(O/E) space: forward + reverse sum, level 0,
               exactly as Orca's notebook cell 10
  orca_t_logoe Orca's own target in log(O/E) (from malltargets.pth)
  orca_t_nan   Orca's NaN-fraction map (malltargetnans.pth); Orca keeps pixels
               with nan-fraction <= 0.25 in BOTH cells
  orcagt_raw   Orca's ground-truth Micro-C (H1ESC 4DNFI9GMP2J8, HFF 4DNFI643OYP9)
               balanced counts at 4 kb from the rebinned .mcool
  regions      (417, 3) chr, start, end of each window
Also writes expected_4kb.npy (Orca's smoothed expected, exp(log) by distance) for
converting raw counts to log(O/E) the Orca way.

Usage: build_tensors.py <cell> <gpu> [ckpt_path tag]   (optional: evaluate another seq2hic checkpoint;
       writes <out>/<cell>_<tag>.npz containing only evo_raw + regions)
"""
import os
import sys

import numpy as np
import torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))  # repo root (paths.py)
import paths as P  # noqa: E402

REPO = str(P.REPO)
sys.path.insert(0, REPO)
os.chdir(REPO)
from config import DNA_map, mappability_map, hic_data_dir  # noqa: E402
from dataset.DNA_loader import DNA_Loader  # noqa: E402
from dataset.hic_loader import HiC_Loader  # noqa: E402
from dataset.mappability_loader import Mappability_Loader  # noqa: E402
from inference.inference_CDNA2d import load_model  # noqa: E402

ORCA = str(P.ORCA_RESOURCES_DIR)
ORCAFIG = str(P.ORCA_FIGURE_DIR)
OUT = os.path.join(P.RA_WORK('orca'), 'tensors')

RES, BASE = 4000, 2000
D = RES // BASE
START_OFFSET_BIN = 3844          # == mallstarts[i][0][0]
LPAD, RPAD = 20, 10
N = 250
WINDOW_PER_CHR = {9: 213, 10: 204}
TEST_CHRS = (9, 10)
CELL = {'H1ESC': dict(hic='4DNFI2TK7L2F', orca_gt='4DNFI9GMP2J8', idx=0),
        'HFF':   dict(hic='4DNFIPC7P27B', orca_gt='4DNFI643OYP9', idx=1)}


def windows():
    for c in TEST_CHRS:
        for i in range(WINDOW_PER_CHR[c]):
            pos = 16_000_000 + i * 500_000
            sf = pos - 16_000_000
            yield c, sf + START_OFFSET_BIN * RES


def build_gt_raw(hic_path):
    loader = HiC_Loader(hic_path, resolution=BASE)
    out, regions = [], []
    for c, start in windows():
        s = start - LPAD * RES
        e = start + (N + RPAD) * RES
        t = loader.get(c, s, e, 0, c, s, e, 0, norm=True).astype(np.float32)
        d0, d1 = t.shape[-1] // D, t.shape[-2] // D
        t = np.sum(np.reshape(t, (-1, d1, D, d0, D)), axis=(-1, -3))[0, LPAD:-RPAD, LPAD:-RPAD]
        out.append(t)
        regions.append((c, start, start + N * RES))
    return np.array(out, np.float32), np.array(regions)


def infer_evo2hic(ckpt, hic_path, device, logoe=None):
    model, _, normalizer = load_model(ckpt)
    model.to(device).eval()
    dna = DNA_Loader(DNA_map['human'], 'Yes')
    mp = Mappability_Loader(mappability_map['human'], 'Yes')
    hic = HiC_Loader(hic_path, resolution=BASE)
    preds = []
    for k, (c, start) in enumerate(windows()):
        s = start - LPAD * RES
        e = start + (N + RPAD) * RES
        pst = []
        for st in (0, 1):
            dna_row = dna.get(c, s, e, st).astype(np.float32)
            map_row = mp.get(c, s, e, st).astype(np.float32)
            inp = hic.get_expected(c, s, e, st, c, s, e, st).astype(np.float32)
            d0, d1 = inp.shape[-1] // D, inp.shape[-2] // D
            inp = np.sum(np.reshape(inp, (-1, d1, D, d0, D)), axis=(-1, -3))
            inp = normalizer.normalize(inp).clip(max=1)
            tt = lambda a: torch.tensor(a)[None, None, None].to(device)
            with torch.no_grad():
                p = model(input_matrix=tt(inp), DNA_col=None, DNA_row=tt(dna_row),
                          mappability_col=None, mappability_row=tt(map_row))
            pst.append((p + p.transpose(-1, -2)) / 2)
        p = ((pst[0] + pst[1].flip((-1, -2))) / 2).squeeze()[LPAD:-RPAD, LPAD:-RPAD]
        if logoe is not None:                       # checkpoint trained in Orca's log(O/E) space
            E, eps, scale = logoe
            d = np.abs(np.arange(N)[:, None] - np.arange(N)[None, :])
            raw = (np.exp(p.cpu().numpy()) * (E[d] + eps) - eps) * scale     # -> count units
            preds.append(raw.astype(np.float32))
        else:
            preds.append(normalizer.unnormalize(p, tensor=True).cpu().numpy().astype(np.float32))
        if k % 50 == 0:
            print(f'  evo2hic window {k}/417', flush=True)
    return np.array(preds)


def orca_arrays(idx):
    mp = torch.load(f'{ORCAFIG}/mallpreds.pth', map_location='cpu', weights_only=False)
    mt = torch.load(f'{ORCAFIG}/malltargets.pth', map_location='cpu', weights_only=False)
    mn = torch.load(f'{ORCAFIG}/malltargetnans.pth', map_location='cpu', weights_only=False)
    P, Tt, Nn = [], [], []
    for i in range(len(mp)):
        f = mp[i][idx][0].detach().numpy()[0, 0]
        r = mp[i][idx + 2][0].detach().numpy()[0, 0, ::-1, ::-1]
        P.append(f + r)                                          # Orca notebook: sum
        Tt.append(np.asarray(mt[i][idx][0])[0].reshape(N, N))
        Nn.append(np.asarray(mn[i][idx][0])[0].reshape(N, N))
    return np.array(P, np.float32), np.array(Tt, np.float32), np.array(Nn, np.float32)


def orca_gt_raw(acc):
    import cooler
    clr = cooler.Cooler(f'{ORCA}/{acc}.rebinned.mcool::/resolutions/{RES}')
    out = []
    for c, start in windows():
        m = clr.matrix(balance=True).fetch((f'chr{c}', start, start + N * RES))
        out.append(np.nan_to_num(np.asarray(m, np.float32)))
    return np.array(out, np.float32)


def main(cell, gpu, ckpt=None, tag=None):
    os.makedirs(OUT, exist_ok=True)
    cfg = CELL[cell]
    device = f'cuda:{gpu}'
    if ckpt is not None:
        print(f'[{cell}] Evo2HiC inference for {ckpt} on {device}', flush=True)
        evo_raw = infer_evo2hic(ckpt, os.path.join(hic_data_dir, cfg['hic'] + '.hic'), device)
        regions = np.array([(c, s, s + N * RES) for c, s in windows()])
        np.savez_compressed(f'{OUT}/{cell}_{tag}.npz', evo_raw=evo_raw, regions=regions)
        print(f'[{cell}] done: {OUT}/{cell}_{tag}.npz {evo_raw.shape}', flush=True)
        return
    print(f'[{cell}] GT raw from {cfg["hic"]}', flush=True)
    gt_raw, regions = build_gt_raw(os.path.join(hic_data_dir, cfg['hic'] + '.hic'))
    print(f'[{cell}] Evo2HiC inference on {device}', flush=True)
    evo_raw = infer_evo2hic(str(P.SEQ2HIC_CKPT(cell)),
                            os.path.join(hic_data_dir, cfg['hic'] + '.hic'), device)
    print(f'[{cell}] Orca arrays', flush=True)
    orca_logoe, orca_t, orca_nan = orca_arrays(cfg['idx'])
    print(f'[{cell}] Orca GT raw from {cfg["orca_gt"]}', flush=True)
    orcagt_raw = orca_gt_raw(cfg['orca_gt'])
    np.savez_compressed(f'{OUT}/{cell}.npz', evo_raw=evo_raw, gt_raw=gt_raw,
                        orca_logoe=orca_logoe, orca_t_logoe=orca_t, orca_t_nan=orca_nan,
                        orcagt_raw=orcagt_raw, regions=regions)
    exp = np.exp(np.load(f'{ORCA}/{cfg["orca_gt"]}.rebinned.mcool.expected.res{RES}.npy'))
    np.save(f'{OUT}/expected_4kb_{cell}.npy', exp)
    print(f'[{cell}] done: evo {evo_raw.shape} gt {gt_raw.shape} orca {orca_logoe.shape} '
          f'orcagt {orcagt_raw.shape}', flush=True)


if __name__ == '__main__':
    main(*sys.argv[1:5])
