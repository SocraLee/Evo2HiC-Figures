#!/usr/bin/env python
"""Run HiCFoundation task-3 (resolution enhancement) on a subset of chromosomes
while keeping the *full-genome* total read count that the model takes as input.

HiCFoundation's Inference_Dataset sums the counts of whatever chromosomes are in
the input pkl and feeds that scalar to the count-aware ViT; restricting the pkl
to the test chromosomes would therefore change the model input.  We build the
full pkl once (their hic2array), record its total count, then run inference on
the subset pkl with total_count patched to the full-genome value.

Usage: infer_subset.py <input.hic> <out_dir> <resolution> <gpu> <chrom1,chrom2,...>
Writes <out_dir>/HiCFoundation_enhanced.pkl and <out_dir>/input.pkl (subset).
"""
import os
import pickle
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))  # repo root (paths.py)
import paths as P  # noqa: E402

HF = str(P.HICFOUNDATION_DIR)
sys.path.insert(0, HF)
os.chdir(HF)

input_hic, out_dir, resolution, gpu, chroms = sys.argv[1:6]
resolution = int(resolution)
chroms = chroms.split(',')
os.environ['CUDA_VISIBLE_DEVICES'] = gpu
os.makedirs(out_dir, exist_ok=True)

from ops.file_format_convert import convert_to_pkl  # noqa: E402

full_pkl = convert_to_pkl(input_hic, out_dir, resolution)      # -> out_dir/input.pkl (all chroms)
full = pickle.load(open(full_pkl, 'rb'))
full_count = float(sum(full[c].data.sum() for c in full))
full_len = sum(full[c].shape[0] for c in full)
missing = [c for c in chroms if c not in full]
if missing:
    raise SystemExit(f'chromosomes {missing} not in {input_hic}; available: {list(full)[:10]}')
subset = {c: full[c] for c in chroms}
sub_count = float(sum(subset[c].data.sum() for c in subset))
pickle.dump(subset, open(full_pkl, 'wb'))                    # overwrite input.pkl with subset
del full
print(f'[infer_subset] full total_count={full_count:.0f} over {full_len} bins; '
      f'subset {chroms} count={sub_count:.0f}', flush=True)

# ---- patch dataset + coverage so the model sees full-genome statistics ------
import data_processing.inference_dataset as ds_mod  # noqa: E402
import inference.main_worker as mw  # noqa: E402

_orig_init = ds_mod.Inference_Dataset.__init__


def _patched_init(self, *a, **k):
    _orig_init(self, *a, **k)
    print(f'[infer_subset] patch total_count {self.total_count:.0f} -> {full_count:.0f}', flush=True)
    self.total_count = full_count


ds_mod.Inference_Dataset.__init__ = _patched_init
mw.calculate_coverage = lambda _p: full_count / full_len       # full-genome coverage for max_cutoff rule


from ops.argparser import argparser_infer  # noqa: E402

args = argparser_infer().parse_args([
    '--input', input_hic, '--batch_size', '16', '--resolution', str(resolution), '--task', '3',
    '--input_row_size', '224', '--input_col_size', '224', '--stride', '20', '--bound', '0',
    '--model_path', os.path.join(HF, 'hicfoundation_model/hicfoundation_resolution.pth.tar'),
    '--output', out_dir, '--gpu', gpu,
    '--genome_id', 'hg38',   # only used by their .hic writer, which we bypass (pkl2hic.py)
])

# their pkl2others call at the end will fail/produce junk without java on PATH; we ignore it
try:
    mw.main_worker(args, full_pkl)
except Exception as e:  # conversion step is the only thing expected to fail
    print('[infer_subset] main_worker raised after inference:', repr(e), flush=True)
assert os.path.exists(os.path.join(out_dir, 'HiCFoundation_enhanced.pkl')), 'inference failed'
print('[infer_subset] ok', flush=True)
