# Cross-species contribution of the sequence prior (DNA Zoo, 177 species)

Reproduces Extended Data Fig. "Contribution of the sequence prior across species" (`edfig:cross-species`),
Supplementary Table "Cross-species analysis (DNA Zoo, 177 species)" (`supptable:xs-clade`),
Supplementary Table "Boundaries added and removed by the sequence input" (`supptable:xs-ctcf`),
Supplementary Fig. "Distant-clade examples of the sequence prior" (`suppfig:xs-examples`),
Supplementary Data 2 (per-species results) and the boundary precision / recall of placental mammals quoted in the
Results (0.667 vs 0.625 and 0.534 vs 0.582, Evo2HiC vs Hi-C-only).

Setting: 1/16-downsampled DNA Zoo `.hic` at 10 kb as input, `raw_hic_fixed/<species>.hic` as ground truth,
first scaffold of each species. Methods compared: Evo2HiC, the same Evo2HiC checkpoint run on random DNA of
matched base composition, the Hi-C-only ablation, HiCARN2 and HiCFoundation (`../ed_table1`).
TAD boundaries: cooltools insulation (100 kb window, `ignore_diags` 2), boundary if prominence >= 0.2,
match within 3 bins. Outputs go to `RA_WORK("species") (= extended_data/ed_fig7)`.

## Scripts and outputs

| script | role | output |
|---|---|---|
| `infer_dna_ablation.py` | released SR checkpoint (`resolution_enhancement/model.pt`) with every non-N base of the scaffold replaced by a random base of the scaffold's composition (`random`); inference settings of the released DNA Zoo outputs in `sr_multi_10000_inference_args.json` | `dna_ablation/random/<species>_enhanced.hic` |
| `real_vs_random.py` | pixel PCC / SPC on real vs random DNA | `dna_ablation/real_vs_random.tsv` |
| `../../evaluate/multi_TAD_eval_cooltools.py` | cooltools TAD-boundary F1 and insulation PCC for Evo2HiC and HiCFoundation (and for the baselines when their outputs are present); also the map loader of the other scripts | `TAD_multi_4methods.tsv` |
| `tad_dump.py` | boundary candidates (index + prominence) of the ground truth, Evo2HiC and HiCFoundation (and the baselines when present) | `tad_candidates/<species>.npz` |
| `tad_random.py` | boundary candidates of Evo2HiC on real and random DNA; TAD F1 of the random-DNA run | `tad_candidates_random/<species>.npz`, `tad_random.tsv` |
| `boundary_auprc.py` | threshold-free boundary AUPRC; Hi-C-only / HiCARN2 values from `result/boundary_baselines.tsv` | `boundary_auprc_check.tsv`, `boundary_auprc_all_methods.tsv` |
| `tad_ctcf.py` | boundaries added / removed by DNA: CTCF motif (JASPAR MA0139.2, p < 1e-5, +-1 bin) and accuracy (+-3 bins); `MODE=random`: random-DNA reference. The Hi-C-only reference requires the Hi-C-only outputs (not released); its result is `result/tad_ctcf.tsv` | `tad_ctcf_random.tsv` (`tad_ctcf.tsv`) |
| `species_mya.py` | patristic distance to *Homo sapiens* in the TimeTree tree (154 of 177 species dated) | `species_mya.tsv` |
| `depth.py` | sequencing depth (contacts per 10 kb bin) and scaffold length of each ground-truth map | `species_depth.tsv` |
| `species_analysis.py` | per-species table of metrics, clade (lists in `../../data/clade/`), divergence and depth; baseline columns from `../ed_table1/result/` | `species_table.tsv` |
| `cross_species_summary.py` | per-clade summary and per-species table | `cross_species_clade.tsv`, `cross_species_per_species.tsv` |
| `supp_data2.py` | per-species table plus boundaries called, precision and recall (Hi-C-only from `result/boundary_baselines.tsv`); prints the placental-mammal precision / recall | `Supplementary_Data_2.tsv` -> Supplementary Data 2; Results sentence |
| `xs_table15.py` | Supp. Table 15 statistics (a: vs random DNA, b: vs Hi-C-only, c: vs other methods); Hi-C-only boundary values from `result/boundary_baselines.tsv` | `xs_table15.tsv` + LaTeX rows -> `supptable:xs-clade` |
| `tad_ctcf_summary.py` | added / removed boundaries by clade, paired Wilcoxon (Hi-C-only arm from `result/tad_ctcf.tsv`) | `tad_ctcf_summary.tsv` -> `supptable:xs-ctcf` |
| `plot_xs_fig.py` | figure panels a-c | `xs_fig.pdf/.png` -> `edfig:cross-species` |
| `plot_examples.py` | distant-clade contact maps with boundary calls | `examples.pdf/.png` -> `suppfig:xs-examples` |

## Order of runs

```bash
cd $EVO2HIC_REPO
export PYTHONPATH=plot_revision:.:$PYTHONPATH                     # evaluate/ and the Evo2HiC modules
S=plot_revision/extended_data/ed_fig7                         # scripts
W=${EVO2HIC_RA_WORK:-plot_revision}             # outputs
# 0. ../ed_table1 first (DNA Zoo outputs, table_multi_*.tsv)
# 1. same checkpoint on random DNA (GPU; split the species list across GPUs as needed)
python $S/infer_dna_ablation.py random 0 $(cat $S/../ed_table1/species_177.txt)
python -m evaluate.batch_evaluate.general_eval_batch -d0 $W/extended_data/ed_fig7/dna_ablation/random --species multi -t diagonal -r 10000 --split test
python $S/real_vs_random.py
# 2. boundaries
python -m evaluate.multi_TAD_eval_cooltools --output $W/extended_data/ed_table1/TAD_multi_4methods.tsv
python $S/tad_dump.py
python $S/tad_random.py
MODE=random python $S/tad_ctcf.py
# 3. per-species covariates and tables
python $S/species_mya.py
python $S/depth.py
python $S/species_analysis.py
python $S/boundary_auprc.py
python $S/cross_species_summary.py
python $S/supp_data2.py
python $S/xs_table15.py
python $S/tad_ctcf_summary.py
# 4. figures
python $S/plot_xs_fig.py
python $S/plot_examples.py
```

`real_vs_random.py` reads the per-species Evo2HiC results `<SR_EVO2HIC_DIR>/eval_results_diagonal_10000/results.tsv`
(same `general_eval_batch` call on the released multispecies outputs).

## Inputs

* Evo2HiC DNA Zoo outputs `SR_EVO2HIC_DIR` (`multi_10000/<species>_enhanced.hic` and its
  `eval_results_diagonal_10000/results.tsv`): inference with the released `resolution_enhancement` checkpoint
  (Zenodo, doi:10.5281/zenodo.17917911) and `evaluate.batch_evaluate.general_eval_batch`.
* HiCFoundation outputs from `../ed_table1`; TimeTree tree and clade lists in `../../data/`.
* Hi-C-only ablation and HiCARN2: outputs and checkpoints are not released (training configuration of the
  ablation: `../../baselines/hic_sr_hic_only/args.json`; HiCARN2 trained as in the Methods). Their per-species
  results are read from the shipped tables:

  | table | content | used by |
  |---|---|---|
  | `result/boundary_baselines.tsv` | per species: boundaries called, precision, recall (Hi-C-only); boundary AUPRC (Hi-C-only, HiCARN2) | `boundary_auprc.py`, `supp_data2.py`, `xs_table15.py` |
  | `result/tad_ctcf.tsv` | boundaries added / removed relative to the Hi-C-only ablation, per species | `tad_ctcf_summary.py`, `plot_xs_fig.py` |
  | `../ed_table1/result/TAD_multi_4methods.tsv` | TAD F1, insulation PCC, boundaries called (all four methods) | `species_analysis.py`, `depth.py` |
  | `../ed_table1/result/table_multi_{PCC,SPC}.tsv` | pixel PCC / SPC (all four methods) | `species_analysis.py` |

  A table regenerated in `RA_WORK` takes precedence (`paths.RA_TABLE`); missing baseline columns are always
  taken from the shipped copy.

