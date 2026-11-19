from hic_utils import *
import argparse
from config import *
import os
import pandas as pd
from utils import *
import time

eps = 1e-10

def pc_parser():
    parser = argparse.ArgumentParser(description='Calculate metrics between hic matrix')
    req_args = parser.add_argument_group('Required Arguments')
    req_args.add_argument('-d0', '--dir-predicted', type=str, required=True)
    req_args.add_argument('--species', type=str, default='human')
    req_args.add_argument('-t', '--task', type=str, choices=['diagonal', 'chrom', 'IS'], help='specify which task to run')
    req_args.add_argument('--split', type=str, default='test')
    req_args.add_argument('-r', '--resolution', type=int, default = 10000, help='resolution')

    return parser

if __name__ == '__main__':
    parser = pc_parser()
    args = parser.parse_args()

    dirp = args.dir_predicted
    species = args.species
    task = args.task
    split = args.split
    resolution = args.resolution

    save_dir = os.path.join(dirp, f'eval_results_{task}_{resolution}')
    
    if os.path.exists(save_dir):
        os.rename(save_dir, save_dir+time.strftime('%m_%d_%H_%M'))

    mkdir(save_dir)

    with open(os.path.join(save_dir, "args.json"), 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    commands = []
    if args.species in splits.keys():
        results = {'hic': []}
        index = pd.read_excel(hic_index, sheet_name='resolution')
        for _, row in tqdm(index.iterrows()):
            if row['split'] == 'test' and row['Organism'] == species:
                hic = row['Hi-C Accession']
                predict = os.path.join(dirp, hic+f'_enhanced_{split}.hic')
                target = os.path.join(hic_data_dir, hic+'.hic')
                if not os.path.isfile(predict):
                    print(f'enhanced result for {hic} not exist')
                    continue
        
                results['hic'].append(hic)

                hic_save_dir = os.path.join(save_dir, hic, f'eval_results_{task}')

                command = f'python -m evaluate.general_eval -f0 {predict} -f1 {target} --split {split} -r {resolution} --species {species} --task {task} --save-dir {hic_save_dir}'
                print(command)
                os.system(command)

                result_file = os.path.join(hic_save_dir, f'{task}_{resolution}_{split}.tsv')
                result :pd.DataFrame = pd.read_csv(result_file, sep='\t', index_col=False)
                for key in result.columns[1:]:
                    if key not in results:
                        results[key] = []
                    results[key].append(result[key].to_numpy()[-1])

                df = pd.DataFrame.from_dict(
                    results 
                )

                output_file = os.path.join(save_dir, 'results.tsv')
                df.to_csv(output_file, sep='\t', index=False)
    else:
        index = open(dnazoo_index, 'r')
        results = {'species': []}
        lines = index.readlines()
        pbar = tqdm(lines)
        for i, line in enumerate(pbar):
            species = line.strip()
            predict = os.path.join(dirp, species+'_enhanced.hic')
            target = os.path.join(dnazoo_hic_dir, species+'.hic')

            if not os.path.isfile(predict):
                print(f'enhanced result for {species} not exist')
                continue

            results['species'].append(species)

            species_save_dir = os.path.join(save_dir, species, f'eval_results_{task}')

            command = f'python -m evaluate.general_eval -f0 {predict} -f1 {target} --split {split} -r {resolution} --species {species} --task {task} --save-dir {species_save_dir}'
            print(command)
            os.system(command)

            result_file = os.path.join(species_save_dir, f'{task}_{resolution}_{1}.tsv')
            result :pd.DataFrame = pd.read_csv(result_file, sep='\t', index_col=False)
            for key in result.columns[1:]:
                if key not in results:
                    results[key] = []
                results[key].append(result[key].to_numpy()[-1])

            df = pd.DataFrame.from_dict(
                results 
            )
            output_file = os.path.join(save_dir, 'results.tsv')
            df.to_csv(output_file, sep='\t', index=False)
    df = pd.DataFrame.from_dict(
        results 
    )

    output_file = os.path.join(save_dir, 'results.tsv')
    df.to_csv(output_file, sep='\t', index=False)

    