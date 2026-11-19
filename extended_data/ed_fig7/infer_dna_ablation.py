#!/usr/bin/env python
"""In-silico DNA ablation on the SAME trained Evo2HiC model (released resolution_enhancement checkpoint), DNA Zoo resolution
enhancement, first scaffold per species, exactly the original multi-species inference settings (args.json of
multi_10000: 2 kb, chunk 160, stride 120, 2 Mb band, 1/16 downsample, output aggregated to 10 kb).
Only the DNA input changes:
  real      - the genome as used originally (sanity check: must reproduce multi_10000/<species>_enhanced.hic);
  random    - every non-N base replaced by an i.i.d. base drawn from the scaffold's own A/C/G/T frequencies;
  shuffle1k - non-N bases permuted within each 1 kb block (keeps local base composition, destroys motifs/k-mers).
N positions stay N in every mode. The species FASTAs are no longer on disk, so sequence comes from the model's one-hot
genome encodings; the first scaffold (HiC_scaffold_1, the one evaluated everywhere) is the first record (offset 0).
Usage: infer_dna_ablation.py <mode> <gpu> <species> [<species> ...]   -> dna_ablation/<mode>/<species>_enhanced.hic"""
import os, sys, json, random
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))  # repo root (paths.py)
import paths as P  # noqa: E402
R = str(P.REPO)
sys.path.insert(0, R); os.chdir(R)
mode, gpu, species_list = sys.argv[1], sys.argv[2], sys.argv[3:]
os.environ['CUDA_VISIBLE_DEVICES'] = gpu
import numpy as np, pandas as pd, torch
from config import *
from hic_utils import find_chr_in_set
from dataset.hic_dna_dataset import HiC_DNA_Inference_Dataset
from dataset.hic_loader import Downsample_HiC_Loader
from dataset.mappability_loader import Mappability_Loader
from inference.inference_CDNA2d import CDNA2d_data_predict_parser, load_model, inference
from inference.infer_utils import *
from hic_utils import save_hic

CKPT = str(P.SR_EVO2HIC_CKPT)
ORIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sr_multi_10000_inference_args.json')   # inference settings of the released DNA Zoo outputs
OUT = f'{P.RA_WORK("species")}/dna_ablation/{mode}'
os.makedirs(OUT, exist_ok=True)

class EncLoader:
    """Minimal stand-in for dataset.DNA_loader.DNA_Loader over the first scaffold, with optional ablation."""
    def __init__(self, species, mode, seed=0):
        cs = pd.read_csv(f'{dnazoo_hic_dir}/{species}.chrom.size', sep='\t', header=None)
        name, L = str(cs.iloc[0, 0]), int(cs.iloc[0, 1])
        self.option = 'Yes'; self.chr_lens = {name: L}; self.chr_pos = {name: (0, L)}
        self.dummy = np.zeros((1, 4)); self.padding = np.zeros((1, 4))
        enc = np.memmap(f'{dnazoo_encoding_dir}/{species}.fasta.encoding', dtype=np.float16, mode='r').reshape(-1, 4)
        codes = np.empty(L, np.uint8)
        for s in range(0, L, 20_000_000):
            x = np.asarray(enc[s:min(L, s + 20_000_000)], dtype=np.float32)
            codes[s:s + len(x)] = np.where(x.sum(1) > 0, x.argmax(1), 4)
        rng = np.random.default_rng(seed); ok = codes < 4
        if mode == 'random':
            freq = np.bincount(codes[ok], minlength=4) / ok.sum()
            codes[ok] = rng.choice(4, size=ok.sum(), p=freq).astype(np.uint8)
        elif mode == 'shuffle1k':
            for s in range(0, L, 1000):
                blk = codes[s:s + 1000]; m = blk < 4
                v = blk[m]; rng.shuffle(v); blk[m] = v
        elif mode != 'real':
            raise ValueError(mode)
        self.codes = codes
        self.eye = np.vstack([np.eye(4), np.zeros((1, 4))]).astype(np.float32)
        self.stats = {'L': L, 'N_frac': float(1 - ok.mean())}

    def get_dummy(self, chr, start, end, strand):
        return np.concatenate([self.dummy] * (end - start)).astype(np.float32)

    def get(self, chr, start, end, strand):          # same semantics as DNA_Loader.get
        c = find_chr_in_set(chr, self.chr_pos); lo, hi = self.chr_pos[c]
        enc = self.eye[self.codes[max(lo + start, lo):min(lo + end, hi)]]
        if start < 0: enc = np.concatenate([self.padding] * abs(start) + [enc])
        if end - start > enc.shape[0]: enc = np.concatenate([enc] + [self.padding] * (end - start - enc.shape[0]))
        if strand == 1: enc = np.flip(enc)
        return enc.astype(np.float32).copy()

    def get_size(self, chr):
        return self.chr_lens[find_chr_in_set(chr, self.chr_pos)]

if __name__ == '__main__':
    orig = json.load(open(ORIG))
    parser = CDNA2d_data_predict_parser(); parser.set_defaults(whole_row=False)
    args = parser.parse_args(['-ckpt', CKPT, '--save-dir', 'dna_ablation', '-sp', 'multi', '-DNA', 'Yes', '-evo2', 'No'])
    for k, v in orig.items(): setattr(args, k, v)             # identical settings to the original run
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    model, modelargs, normalizer = load_model(CKPT)
    for sp in species_list:
        out = f'{OUT}/{sp}_enhanced.hic'
        if os.path.exists(out): print('skip', sp); continue
        loader = EncLoader(sp, mode)
        mapl = Mappability_Loader(None, 'dummy')
        hl = Downsample_HiC_Loader(f'{dnazoo_hic_dir}/{sp}.hic', args.downsample, sp, resolution=args.resolution,
                                   read_count=modelargs['read_count'], chr_pos=loader.chr_pos)
        ds = HiC_DNA_Inference_Dataset(chromosome_split=(0,), HiC_loaders=[hl], DNA_loader=loader,
                                       Mappability_loader=mapl, normalizer=normalizer, **args.__dict__)
        submats, poses = inference(ds, model, args.batch_size)
        ch = list(loader.chr_pos)[0]; size = loader.get_size(0)
        hic = construct_hic_matrices(submats, poses, 0, 0, size, size, args.resolution, args.target_resolution, args.max_seperation)
        ref = out.replace('.hic', '.chrom.size'); open(ref, 'w').write(f'{ch}\t{size}\n')
        save_hic({(ch, ch): hic}, None, out, first=True, finished=False)
        save_hic({}, ref, out, args.target_resolution, first=False, finished=True)   # juicer pre, as in the single-species branch
        print(f'[{mode}] {sp} done  L={loader.stats["L"]/1e6:.1f} Mb  N={loader.stats["N_frac"]:.3f}', flush=True)
        del submats, poses, hic, hl, loader
