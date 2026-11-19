import numpy as np
from hic_utils import *
import argparse
import os
import pandas as pd
from utils import *
import json
from config import *
import time
from evaluate.eval_utils import *
from evaluate.diagonal_eval import get_diagonal_metrics
from evaluate.chrom_eval import get_mse, get_ssim
from evaluate.IS_eval import compute_TAD_similarity
from dataset.mappability_loader import Mappability_Loader
eps = 1e-10

def pc_parser():
    parser = argparse.ArgumentParser(description='Calculate metrics between hic matrix')
    req_args = parser.add_argument_group('Required Arguments')
    req_args.add_argument('-f0', '--file-predicted', type=str, required=True)
    req_args.add_argument('-f1', '--file-target', type=str, required=True)
    req_args.add_argument('-sp', '--species', type=str, default = 'human', help='DNA species')
    req_args.add_argument('-t', '--task', type=str, choices=['diagonal', 'chrom', 'IS'], help='specify which task to run')

    misc_args = parser.add_argument_group('Miscellaneous Arguments')
    misc_args.add_argument('--save-dir', type=str, default=None)
    misc_args.add_argument('--split', type=str, default='test', help='split to inference')
    misc_args.add_argument('-r', '--resolution', type=int, default = 10000, help='resolution')
    misc_args.add_argument('--seperation-thres', type=int, default = 2000000, help='the maximum seperation to the diagonal')
    misc_args.add_argument('--ignore-diag', type=int, default = 0, help='ignore diagonal')
    misc_args.add_argument('--h0-multiplier', type=float, default = 1, help='multiple hic0 by this number')

    data_args = parser.add_argument_group('data arguments')
    data_args.add_argument('--log', type=str, choices=['No', 'Yes'], default='No')
    data_args.add_argument('--clip', type=int, default=1000)
    
    return parser

if __name__ == '__main__':
    parser = pc_parser()
    args = parser.parse_args()

    filep = args.file_predicted
    filet = args.file_target

    save_dir = args.save_dir
    if save_dir is None:
        save_dir = f'eval_results_{args.task}'
    if os.path.dirname(save_dir) == '':
        save_dir = os.path.join('.'.join(filep.split('.')[:-1]), save_dir)
    if os.path.exists(save_dir):
        os.rename(save_dir, save_dir+time.strftime('%m_%d_%H_%M'))
    
    mkdir(save_dir)

    with open(os.path.join(save_dir, "args.json"), 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    hic0 = hicstraw.HiCFile(filep)
    hic1 = hicstraw.HiCFile(filet)

    task = args.task
    split = args.split
    species = args.species
    res = args.resolution
    
    print(f'Resolution {res}:')

    D = args.seperation_thres // res
    S = args.ignore_diag // res

    if D<=S:
        print('all entries were ignored.')
        exit(0)

    if args.task == 'diagonal':
        if species in splits.keys():
            
            dis = [str(i) for i in range(D)] + ['avg']
            all_pcc = np.zeros(D+1)
            all_spc = np.zeros(D+1)
            all_mse = np.zeros(D+1)
            all_mer = np.zeros(D+1)
            all_cnt = np.zeros(D+1)

            for ch in splits[species][split]:
                print(f'Processing {ch}')

                up0 = read_hic(hic0, res, format = 'upper', chrname = ch)
                up1 = read_hic(hic1, res, format = 'upper', chrname = ch)

                up0.data = up0.data * args.h0_multiplier
                up0.data = normalize_eval(up0.data, args.clip, args.log)
                up1.data = normalize_eval(up1.data, args.clip, args.log)

                pcc, spc, mse, mer, cnt = get_diagonal_metrics(up0, up1, S, D)
                
                df = pd.DataFrame.from_dict(
                    {
                        'DIS' : dis,
                        'PCC' : pcc,
                        'SPC' : spc,
                        'MSE' : mse,
                        'MER' : mer,
                        'VAL' : cnt
                    }
                )

                output_file = os.path.join(save_dir, f'{task}_{res}_{ch}.tsv')
                df.to_csv(output_file, sep='\t', index=False)
                
                all_pcc += pcc
                all_spc += spc
                all_mse += mse
                all_mer += mer
                all_cnt += cnt

            n_chrom = len(splits[species][split])
            df = pd.DataFrame.from_dict(
                {
                    'DIS' : dis,
                    'PCC': all_pcc / n_chrom,
                    'SPC': all_spc / n_chrom,
                    'MSE': all_mse / n_chrom,
                    'MER': all_mer / n_chrom,
                    'valid entry': all_cnt
                }
            )

            output_file = os.path.join(save_dir, f'{task}_{res}_{split}.tsv')
            df.to_csv(output_file, sep='\t', index=False)
        else:
            up0 = read_hic(hic0, res, format = 'upper', chrid = 1)
            up1 = read_hic(hic1, res, format = 'upper', chrid = 1)

            up0.data = up0.data * args.h0_multiplier
            up0.data = normalize_eval(up0.data, args.clip, args.log)
            up1.data = normalize_eval(up1.data, args.clip, args.log)

            pcc, spc, mse, mer, cnt = get_diagonal_metrics(up0, up1, S, D)
            
            dis = [str(i) for i in range(D)] + ['avg']

            df = pd.DataFrame.from_dict(
                {
                    'DIS' : dis,
                    'PCC' : pcc,
                    'SPC' : spc,
                    'MSE' : mse,
                    'MER' : mer,
                    'valid entry': cnt
                }
            )

            output_file = os.path.join(save_dir, f'{task}_{res}_{1}.tsv')
            df.to_csv(output_file, sep='\t', index=False)
    elif args.task == 'chrom':
        if species in splits.keys():
            all_mse = []
            all_psnr = []
            all_ssim = []
            all_ch = []

            for ch in splits[species][split]:
                print(f'Processing {ch}')

                mat0 = read_hic(hic0, res, format = 'matrix', chrname = ch)
                mat1 = read_hic(hic1, res, format = 'matrix', chrname = ch)

                mat0 = mat0 * args.h0_multiplier
                mat0 = normalize_eval(mat0, args.clip, args.log)
                mat1 = normalize_eval(mat1, args.clip, args.log)

                mask = np.ones_like(mat0)
                mask = np.triu(mask, S+1)
                mask = np.tril(mask, D-1)

                mse = get_mse(mat0, mat1, mask)
                psnr = 10 * np.log10(1/mse)
                ssim = get_ssim(mat0, mat1, mask, device='cuda')

                all_mse.append(mse)
                all_psnr.append(psnr)
                all_ssim.append(ssim)
                all_ch.append(str(ch))

            all_mse.append(np.mean(all_mse))
            all_psnr.append(np.mean(all_psnr))
            all_ssim.append(np.mean(all_ssim))
            all_ch.append('avg')
            df = pd.DataFrame.from_dict(
                {
                    'CH': all_ch,
                    'MSE': all_mse,
                    'PSNR': all_psnr,
                    'SSIM': all_ssim,
                }
            )

            output_file = os.path.join(save_dir, f'{task}_{res}_{split}.tsv')
            df.to_csv(output_file, sep='\t', index=False)
        else:
            all_mse = []
            all_psnr = []
            all_ssim = []
            all_ch = []

            mat0 = read_hic(hic0, res, format = 'matrix', chrid = 1)
            mat1 = read_hic(hic1, res, format = 'matrix', chrid = 1)

            mat0 = mat0 * args.h0_multiplier
            mat0 = normalize_eval(mat0, args.clip, args.log)
            mat1 = normalize_eval(mat1, args.clip, args.log)

            mask = np.ones_like(mat0)
            mask = np.triu(mask, S+1)
            mask = np.tril(mask, D-1)

            mse = get_mse(mat0, mat1, mask)
            psnr = 10 * np.log10(1/mse)
            ssim = get_ssim(mat0, mat1, mask, device='cuda')

            all_mse.append(mse)
            all_psnr.append(psnr)
            all_ssim.append(ssim)
            all_ch.append(str(1))

            df = pd.DataFrame.from_dict(
                {
                    'CH': all_ch,
                    'MSE': all_mse,
                    'PSNR': all_psnr,
                    'SSIM': all_ssim,
                }
            )

            output_file = os.path.join(save_dir, f'{task}_{res}_{1}.tsv')
            df.to_csv(output_file, sep='\t', index=False)
    elif args.task == 'IS':
        if species in splits.keys():
            all_TAD_f1 = []
            all_IS_pcc = []
            all_IS_nrm = []
            all_ch = []

            maploader = Mappability_Loader(mappability_map[species])

            for ch in splits[species][split]:
                print(f'Processing {ch}')

                # Auto-detect norm for both: SCALE → KR → NONE.
                def _read_hic_safe(h, ch):
                    for nn in ('SCALE', 'KR', 'NONE'):
                        try:
                            return read_hic(h, res, format='matrix', chrname=ch, norm=nn)
                        except Exception:
                            continue
                    raise RuntimeError(f"no norm worked for {h}")
                mat0 = _read_hic_safe(hic0, ch)
                mat1 = _read_hic_safe(hic1, ch)
                
                mat0 = mat0 * args.h0_multiplier

                mappability = maploader.get(ch, 0, mat0.shape[0]*res, 0)
                mappability = mappability.reshape((-1, res)).mean(axis=-1)
                low_map = (mappability < 0.5)
                
                mat0[low_map, :] = np.nan
                mat0[:, low_map] = np.nan
                mat1[low_map, :] = np.nan
                mat1[:, low_map] = np.nan

                TAD_f1, IS_pcc = compute_TAD_similarity(mat0, mat1)

                all_TAD_f1.append(TAD_f1)
                all_IS_pcc.append(IS_pcc)
                all_ch.append(str(ch))

            all_TAD_f1.append(np.mean(all_TAD_f1))
            all_IS_pcc.append(np.mean(all_IS_pcc))
            all_ch.append('avg')
            df = pd.DataFrame.from_dict(
                {
                    'CH': all_ch,
                    'TAD_f1': all_TAD_f1,
                    'IS_pcc': all_IS_pcc,
                }
            )

            output_file = os.path.join(save_dir, f'{task}_{res}_{split}.tsv')
            df.to_csv(output_file, sep='\t', index=False)
        else:
            all_TAD_f1 = []
            all_IS_pcc = []
            all_IS_nrm = []
            all_ch = []

            mat0 = read_hic(hic0, res, format = 'matrix', chrid = 1, norm='SCALE')
            mat1 = read_hic(hic1, res, format = 'matrix', chrid = 1, norm='SCALE')

            mat0 = mat0 * args.h0_multiplier

            TAD_f1, IS_pcc = compute_TAD_similarity(mat0, mat1)

            all_TAD_f1.append(TAD_f1)
            all_IS_pcc.append(IS_pcc)
            all_ch.append(str(1))

            df = pd.DataFrame.from_dict(
                {
                    'CH': all_ch,
                    'TAD_f1': all_TAD_f1,
                    'IS_pcc': all_IS_pcc,
                }
            )

            output_file = os.path.join(save_dir, f'{task}_{res}_{1}.tsv')
            df.to_csv(output_file, sep='\t', index=False)