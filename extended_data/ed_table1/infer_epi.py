#!/usr/bin/env python
"""HiCFoundation task-4 (epigenomic assay profiling) restricted to given chromosomes,
keeping the full-genome total count as model input (same trick as infer_subset.py).

Usage: infer_epi.py <input.hic> <out_dir> <gpu> <chr9,chr10>
Output: <out_dir>/HiCFoundation_epigenomic_assay_prediction.pkl  {chrom: (6, nbins@1kb)}
        order: CTCF, H3K4me3, H3K27ac, H3K27me3, ATAC-seq, DNase-seq
"""
import os
import pickle
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))  # repo root (paths.py)
import paths as P  # noqa: E402

HF = str(P.HICFOUNDATION_DIR)
sys.path.insert(0, HF)
os.chdir(HF)

input_hic, out_dir, gpu, chroms = sys.argv[1:5]
resolution = 1000
chroms = chroms.split(',')
os.environ['CUDA_VISIBLE_DEVICES'] = gpu
os.makedirs(out_dir, exist_ok=True)

from ops.file_format_convert import convert_to_pkl  # noqa: E402

full_pkl = convert_to_pkl(input_hic, out_dir, resolution)
full = pickle.load(open(full_pkl, 'rb'))
full_count = float(sum(full[c].data.sum() for c in full))
full_len = sum(full[c].shape[0] for c in full)
def _resolve(c):
    for cand in (c, c.replace('chr', ''), 'chr' + c):
        if cand in full:
            return cand
    raise SystemExit(f'chromosome {c} not in {input_hic}; available: {list(full)[:10]}')


subset = {_resolve(c): full[_resolve(c)] for c in chroms}
pickle.dump(subset, open(full_pkl, 'wb'))
del full
print(f'[infer_epi] full total_count={full_count:.0f} over {full_len} bins; subset {chroms}', flush=True)

import data_processing.inference_dataset as ds_mod  # noqa: E402
import inference.main_worker as mw  # noqa: E402

_orig_init = ds_mod.Inference_Dataset.__init__


def _patched_init(self, *a, **k):
    _orig_init(self, *a, **k)
    print(f'[infer_epi] patch total_count {self.total_count:.0f} -> {full_count:.0f}', flush=True)
    self.total_count = full_count


ds_mod.Inference_Dataset.__init__ = _patched_init
mw.calculate_coverage = lambda _p: full_count / full_len

from ops.argparser import argparser_infer  # noqa: E402

args = argparser_infer().parse_args([
    '--input', input_hic, '--batch_size', '16', '--resolution', str(resolution), '--task', '4',
    '--input_row_size', '128', '--input_col_size', '4000', '--stride', '32', '--bound', '0',
    '--model_path', os.path.join(HF, 'hicfoundation_model/hicfoundation_epigenomic.pth.tar'),
    '--output', out_dir, '--gpu', gpu,
])
mw.main_worker(args, full_pkl)
assert os.path.exists(os.path.join(out_dir, 'HiCFoundation_epigenomic_assay_prediction.pkl')), 'inference failed'
print('[infer_epi] ok', flush=True)
