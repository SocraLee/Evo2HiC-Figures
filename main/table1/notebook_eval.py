"""notebook_eval.py — Reproduce the Fig 3 TAD analysis from
main/fig3/plot.ipynb with strict per-window methodology.

Differs from a stitched-.hic eval in that EACH 1Mb x 1Mb prediction is computed
independently by running the model on the local region — matching the paper's
417 sub-matrix evaluation.

Outputs:
    --output-dir/
        h1esc_evo2hic.npy   shape (n_windows, 250, 250) float32
        h1esc_orca.npy      ditto
        h1esc_targets.npy   ditto
        h1esc_regions.tsv   per-window genomic coords
        ... (same for HFF)
        results.tsv         (cell, method, IS_PCC mean, TAD_F1 mean, n_windows)

Run with the Evo2HiC env (has both the model and all eval deps).
"""
from __future__ import annotations

import argparse
import logging
import os
import sys

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr
from tqdm import tqdm
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))  # repo root (paths.py)
import paths as P  # noqa: E402

logging.getLogger().setLevel(logging.ERROR)

REPO = str(P.REPO)
sys.path.insert(0, REPO)

from config import DNA_map, mappability_map, evo2_embedding_map  # noqa: E402
from dataset.DNA_loader import DNA_Loader                       # noqa: E402
from dataset.hic_loader import HiC_Loader                       # noqa: E402
from dataset.mappability_loader import Mappability_Loader       # noqa: E402
from inference.inference_CDNA2d import load_model               # noqa: E402

# ====== Window definition matching the notebook ======
RESOLUTION = 4000
BASE_RESOLUTION = 2000
D = RESOLUTION // BASE_RESOLUTION  # = 2
START_OFFSET_BIN = 62 * 32 + 62 * 16 + 62 * 8 + 62 * 4 + 62 * 2  # = 7812
LPAD, RPAD = 20, 10
WINDOW_BINS = 250
WINDOW_PER_CHR = {9: 213, 10: 204}
TEST_CHRS = (9, 10)


# ====== Targets and ORCA refs (cell `6bb6d786` of notebook) ======
def build_targets(hic_path: str):
    """Reproduce the notebook's targets construction. Returns (targets, regions)."""
    loader = HiC_Loader(hic_path, resolution=BASE_RESOLUTION)
    targets, regions = [], []
    for c in TEST_CHRS:
        for i in range(WINDOW_PER_CHR[c]):
            pos = 16_000_000 + i * 500_000
            sf = pos - 16_000_000
            s = sf + (START_OFFSET_BIN - LPAD) * RESOLUTION
            e = sf + (START_OFFSET_BIN + WINDOW_BINS + RPAD) * RESOLUTION
            target = loader.get(c, s, e, 0, c, s, e, 0, norm=True).astype(np.float32)
            d0 = target.shape[-1] // D
            d1 = target.shape[-2] // D
            target = np.sum(np.reshape(target, (-1, d0, D, d1, D)), axis=(-1, -3))
            target = target[0, LPAD:-RPAD, LPAD:-RPAD]
            targets.append(target)
            regions.append((sf + START_OFFSET_BIN * RESOLUTION,
                            sf + (START_OFFSET_BIN + WINDOW_BINS) * RESOLUTION))
    return np.array(targets), regions


# ====== Evo2HiC model inference per window (cell `43b7a166`) ======
def infer_evo2hic(ckpt_path: str, hic_path: str, device="cuda:0"):
    model, modelargs, normalizer = load_model(ckpt_path)
    model.to(device).eval()
    DNA_loader = DNA_Loader(DNA_map["human"], "Yes")
    map_loader = Mappability_Loader(mappability_map["human"], "Yes")
    hic_loader = HiC_Loader(hic_path, resolution=BASE_RESOLUTION)

    preds = []
    for c in TEST_CHRS:
        for i in tqdm(range(WINDOW_PER_CHR[c]), desc=f"infer chr{c}"):
            pos = 16_000_000 + i * 500_000
            sf = pos - 16_000_000
            s = sf + (START_OFFSET_BIN - LPAD) * RESOLUTION
            e = sf + (START_OFFSET_BIN + WINDOW_BINS + RPAD) * RESOLUTION

            pred_st = []
            for st in (0, 1):
                DNA_row = DNA_loader.get(c, s, e, st).astype(np.float32)
                map_row = map_loader.get(c, s, e, st).astype(np.float32)

                inp = hic_loader.get_expected(c, s, e, st, c, s, e, st).astype(np.float32)
                d0 = inp.shape[-1] // D
                d1 = inp.shape[-2] // D
                inp = np.sum(np.reshape(inp, (-1, d0, D, d1, D)), axis=(-1, -3))
                inp = normalizer.normalize(inp).clip(max=1)

                DNA_t = torch.tensor(DNA_row).unsqueeze(0).unsqueeze(0).unsqueeze(0).to(device)
                map_t = torch.tensor(map_row).unsqueeze(0).unsqueeze(0).unsqueeze(0).to(device)
                inp_t = torch.tensor(inp).unsqueeze(0).unsqueeze(0).unsqueeze(0).to(device)

                with torch.no_grad():
                    pred = model(input_matrix=inp_t, DNA_col=None, DNA_row=DNA_t,
                                 mappability_col=None, mappability_row=map_t)
                pred_st.append((pred + pred.transpose(-1, -2)) / 2)

            pred = ((pred_st[0] + pred_st[1].flip((-1, -2))) / 2).squeeze()
            pred = pred[LPAD:-RPAD, LPAD:-RPAD]
            pred = normalizer.unnormalize(pred, tensor=True)
            preds.append(pred.cpu().detach().numpy())
    return np.array(preds)


# ====== ORCA per window via genomepredict ======
def infer_orca_per_window(orca_dir: str, cell: str, device="cuda:0"):
    """Re-run ORCA's genomepredict on each notebook window, store the (250,250)
    log(obs/exp) prediction (forward+reverse averaged). Then convert to count
    space using the same normmat-by-distance the notebook uses."""
    sys.path.insert(0, orca_dir)
    cwd = os.getcwd()
    os.chdir(orca_dir)
    try:
        import orca_predict as op
        op.load_resources(models=["32M"], use_cuda=True)

        cell_idx = 0 if cell == "H1ESC" else 1

        # normmat (same as notebook)
        nm_path = os.path.join(orca_dir, "resources",
                               "4DNFI9GMP2J8.rebinned.mcool.expected.res4000.npy")
        normmat_bydist = np.exp(np.load(nm_path))[:250]
        normmat = normmat_bydist[np.abs(np.arange(250)[:, None] - np.arange(250)[None, :])]
        eps = float(np.min(normmat))

        preds = []
        for c in TEST_CHRS:
            chrom = f"chr{c}"
            for i in tqdm(range(WINDOW_PER_CHR[c]), desc=f"orca {cell} chr{c}"):
                pos = 16_000_000 + i * 500_000
                # The notebook's window aligns with start_coord in zoom 5;
                # genomepredict expects mpos = center, wpos = pos.
                wstart = pos - 16_000_000
                wend = pos + 16_000_000
                seq = op.hg38.get_encoding_from_coords(chrom, wstart, wend)[None, :, :]

                with torch.no_grad():
                    out_fwd = op.genomepredict(seq, mchr=chrom, mpos=pos, wpos=pos,
                                                use_cuda=True)
                    out_rev = op.genomepredict(seq[:, ::-1, ::-1].copy(),
                                                mchr=chrom, mpos=pos, wpos=pos,
                                                use_cuda=True)

                level = 5
                p_fwd = out_fwd["predictions"][cell_idx][level]
                p_rev = out_rev["predictions"][cell_idx][level][::-1, ::-1]
                log_oe = (p_fwd + p_rev) / 2  # average f/r — paper uses sum, but mean is equivalent up to scale; will rescale below
                # Match notebook exactly: it sums the forward and reverse logs (equivalent to using two separate calls).
                # The notebook: p1 = fwd[0,0,:,:] + rev[0,0,::-1,::-1]
                #              raw = exp(p1) * (normmat+eps) - eps
                # Note: their p1 is a SUM of two predictions, so exp(p1) = exp(fwd)*exp(rev), which double-counts.
                # We follow that exactly to match their numbers.
                summed = p_fwd + p_rev
                raw = np.exp(summed) * (normmat + eps) - eps
                preds.append(raw.astype(np.float32))
        return np.array(preds)
    finally:
        os.chdir(cwd)


# ====== Insulation pipeline (verbatim from notebook) ======
def insulation_pipeline_from_matrix(M, res=RESOLUTION, window_bp=100_000,
                                     ignore_diags=2, prominence_cut=0.2):
    import cooler
    import tempfile
    from cooltools.api.insulation import calculate_insulation_score, find_boundaries

    fd, cool_path = tempfile.mkstemp(suffix=".cool")
    os.close(fd); os.unlink(cool_path)

    N = M.shape[0]
    region_len = N * res
    starts = np.arange(0, region_len, res, dtype=np.int64)
    bins = pd.DataFrame({"chrom": "chrR", "start": starts, "end": starts + res})

    i, j = np.triu_indices(N, k=0)
    counts = M[i, j].astype(float)
    mask = np.isfinite(counts) & (counts != 0)
    pixels = pd.DataFrame({"bin1_id": i[mask].astype(np.int64),
                           "bin2_id": j[mask].astype(np.int64),
                           "count":   counts[mask]})
    cooler.create_cooler(cool_path, bins=bins, pixels=pixels,
                         dtypes={"count": "float64"}, ordered=True)

    clr = cooler.Cooler(cool_path)
    is_table = calculate_insulation_score(
        clr, window_bp=[int(window_bp)], ignore_diags=ignore_diags,
        clr_weight_name=None,
    )
    is_table = find_boundaries(is_table)

    wtag = str(window_bp)

    def pick(prefix):
        cands = [c for c in is_table.columns if c.startswith(prefix)]
        tagged = [c for c in cands if c.endswith(wtag)]
        return tagged[0] if tagged else (cands[0] if cands else None)

    col_log2 = pick("log2_insulation_score")
    col_prom = pick("boundary_strength")
    scores = is_table[col_log2].to_numpy().astype(float)
    prom = is_table[col_prom].to_numpy().astype(float)
    is_b = np.isfinite(prom) & (prom >= float(prominence_cut))
    boundary_bins = np.where(is_b)[0].astype(int).tolist()

    try: os.unlink(cool_path)
    except OSError: pass
    return scores, boundary_bins


def bound_matching(prd, tgt, boundary_zone_size=3):
    if len(tgt) == 0: return np.nan, 0
    if len(prd) == 0: return 0, 0
    prd_m = np.zeros_like(prd); tgt_m = np.zeros_like(tgt)
    for i, pb in enumerate(prd):
        for j, tb in enumerate(tgt):
            if abs(pb - tb) <= boundary_zone_size:
                prd_m[i] = 1; tgt_m[j] = 1
    p = np.nan_to_num(np.mean(prd_m))
    r = np.nan_to_num(np.mean(tgt_m))
    if p == 0 or r == 0: return 0, 0
    return 2 / (1 / p + 1 / r), int(np.sum(tgt_m))


def evaluate_per_window(preds, targets):
    is_pccs, f1s = [], []
    targets = targets.copy()
    preds = preds.copy()
    mask = np.isfinite(targets)
    preds[~mask] = np.nan
    n = preds.shape[0]
    for i in tqdm(range(n), desc="IS"):
        try:
            IS_t, b_t = insulation_pipeline_from_matrix(targets[i])
            IS_p, b_p = insulation_pipeline_from_matrix(preds[i])
        except Exception as e:
            print(f"[skip {i}] {e}")
            continue
        ok = np.isfinite(IS_t) & np.isfinite(IS_p)
        if ok.sum() < 10: continue
        try:
            pcc, _ = pearsonr(IS_p[ok], IS_t[ok])
            is_pccs.append(pcc)
        except Exception:
            pass
        f1, _ = bound_matching(b_p, b_t)
        if f1 is not None and not np.isnan(f1):
            f1s.append(f1)
    return is_pccs, f1s


# ====== Main ======
HICNAME = {"H1ESC": "4DNFI2TK7L2F", "HFF": "4DNFIPC7P27B"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", choices=("H1ESC", "HFF", "both"), default="both")
    ap.add_argument("--ckpt-h1esc",
                    default=str(P.SEQ2HIC_CKPT("H1ESC")))
    ap.add_argument("--ckpt-hff",
                    default=str(P.SEQ2HIC_CKPT("HFF")))
    ap.add_argument("--orca-dir", default=str(P.ORCA_DIR))
    ap.add_argument("--hic-data-dir",
                    default=os.path.join(P.HIC_DATA_ROOT, "data/hic/raw_hic"))
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--skip-orca", action="store_true",
                    help="reuse cached ORCA per-window predictions if present")
    ap.add_argument("--skip-evo2hic", action="store_true",
                    help="reuse cached Evo2HiC per-window predictions if present")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    cells = ["H1ESC", "HFF"] if args.cell == "both" else [args.cell]

    summary_rows = []
    for cell in cells:
        print(f"\n=========== {cell} ===========")
        gt_hic = os.path.join(args.hic_data_dir, f"{HICNAME[cell]}.hic")
        targets_path = os.path.join(args.output_dir, f"{cell}_targets.npy")
        regions_path = os.path.join(args.output_dir, f"{cell}_regions.tsv")
        if os.path.exists(targets_path):
            targets = np.load(targets_path)
            regions = pd.read_csv(regions_path, sep="\t").to_records(index=False)
            print(f"[cache] {targets_path}: {targets.shape}")
        else:
            print(f"[targets] reading {gt_hic}")
            targets, regions = build_targets(gt_hic)
            np.save(targets_path, targets)
            pd.DataFrame(regions, columns=["start_bp", "end_bp"]).to_csv(
                regions_path, sep="\t", index=False)
            print(f"[ok] saved {targets.shape} → {targets_path}")

        # Evo2HiC
        evo_path = os.path.join(args.output_dir, f"{cell}_evo2hic.npy")
        if args.skip_evo2hic and os.path.exists(evo_path):
            evo_preds = np.load(evo_path)
            print(f"[cache] {evo_path}: {evo_preds.shape}")
        else:
            ckpt = args.ckpt_h1esc if cell == "H1ESC" else args.ckpt_hff
            print(f"[evo2hic] inferring with {ckpt}")
            evo_preds = infer_evo2hic(ckpt, gt_hic, device=args.device)
            np.save(evo_path, evo_preds)
            print(f"[ok] saved {evo_preds.shape} → {evo_path}")

        # ORCA
        orca_path = os.path.join(args.output_dir, f"{cell}_orca.npy")
        if args.skip_orca and os.path.exists(orca_path):
            orca_preds = np.load(orca_path)
            print(f"[cache] {orca_path}: {orca_preds.shape}")
        else:
            print(f"[orca] inferring per-window with {args.orca_dir}")
            orca_preds = infer_orca_per_window(args.orca_dir, cell, device=args.device)
            np.save(orca_path, orca_preds)
            print(f"[ok] saved {orca_preds.shape} → {orca_path}")
        # rescale ORCA to match GT total counts (notebook does this)
        mask = np.isfinite(targets)
        r0 = float(np.sum(targets[mask]) / max(1.0, np.sum(orca_preds[mask])))
        orca_preds_rescaled = orca_preds * r0

        print("[evo2hic IS]")
        evo_pccs, evo_f1s = evaluate_per_window(evo_preds, targets)
        print(f"  IS_PCC mean: {np.mean(evo_pccs):.4f}  TAD_F1 mean: {np.mean(evo_f1s):.4f}")
        print("[orca IS]")
        orca_pccs, orca_f1s = evaluate_per_window(orca_preds_rescaled, targets)
        print(f"  IS_PCC mean: {np.mean(orca_pccs):.4f}  TAD_F1 mean: {np.mean(orca_f1s):.4f}")

        for method, pccs, f1s in [("evo2hic", evo_pccs, evo_f1s),
                                   ("orca", orca_pccs, orca_f1s)]:
            summary_rows.append({
                "cell": cell, "method": method,
                "IS_PCC": float(np.mean(pccs)),
                "TAD_F1": float(np.mean(f1s)),
                "n_windows": len(pccs),
            })

        df = pd.DataFrame({
            "evo2hic_IS_PCC": evo_pccs,
            "orca_IS_PCC":    orca_pccs,
            "evo2hic_TAD_F1": evo_f1s + [np.nan] * (len(evo_pccs) - len(evo_f1s)),
            "orca_TAD_F1":    orca_f1s + [np.nan] * (len(orca_pccs) - len(orca_f1s)),
        })
        df.to_csv(os.path.join(args.output_dir, f"{cell}_per_window.tsv"), sep="\t", index=False)

    out = os.path.join(args.output_dir, "results.tsv")
    pd.DataFrame(summary_rows).to_csv(out, sep="\t", index=False)
    print(f"\n[done] {out}")
    print(pd.DataFrame(summary_rows).to_string(index=False))


if __name__ == "__main__":
    main()
