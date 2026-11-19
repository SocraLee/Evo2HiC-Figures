# Comparison with HiCFoundation

Reproduces Extended Data Table 1 "Comparison with HiCFoundation" (`edtable:hicf`): (a) epigenomic prediction
on GM12878, H1ESC and held-out K562; (b) Hi-C resolution enhancement on the human and mouse test maps and the
177 DNA Zoo species. The HiCFoundation columns of the cross-species analysis (`../ed_fig7`) use the DNA Zoo
outputs produced here.

HiCFoundation (Wang et al., Nature Methods 2025) is run with its released weights
(`hicfoundation_resolution.pth.tar`, `hicfoundation_epigenomic.pth.tar`), without retraining, on exactly the
inputs used for Evo2HiC: 1/16-downsampled `.hic` at 10 kb for resolution enhancement, the raw cell-line `.hic`
for epigenomic prediction. Its outputs are scored with the Evo2HiC evaluation code (raw counts; test
chromosomes chr9/10 for human, chr1-4 for mouse, first scaffold for DNA Zoo).

## Requirements

* HiCFoundation repository with its released weights in `hicfoundation_model/` at `$HICFOUNDATION_DIR`, and a
  Python environment for it (`$HICFOUNDATION_PY`).
* Java >= 1.8 on `PATH` and the juicer_tools jar for `.hic` writing (`JUICER_JAR`, default `juicer_tools.jar`
  next to `pkl2hic.py`).
* Downsampled inputs: `$EVO2HIC_HIC_DATA/data/hic/downsample/<acc>.16.seed0.hic` (human/mouse) and
  `$EVO2HIC_HIC_DATA/data/hic/downsample_multi/<species>.16.seed0.hic` + `.chrom.size` (DNA Zoo).
* Evo2HiC per-species results `SR_EVO2HIC_DIR/eval_results_{diagonal,chrom}_10000/results.tsv` (released
  `resolution_enhancement` checkpoint, Zenodo doi:10.5281/zenodo.17917911).
* Hi-C-only ablation and HiCARN2: outputs and checkpoints are not released. Their per-map / per-species values
  (and the human / mouse Evo2HiC values of Supp. Fig. 17) are read from the shipped tables
  `result/table_{human,mouse,multi}_{PCC,SPC,PSNR,SSIM}.tsv`, used by `summarize_sr.py` whenever
  `result/` or a baseline output folder is absent. `result/TAD_multi_4methods.tsv` (TAD F1 / insulation PCC
  of all four methods on DNA Zoo) is used by `../ed_fig7`.
* Evo2HiC epigenomic tracks `EPI_EVO2HIC_DIR/<cell>/{9,10}.npy` from the released `epi_prediction` checkpoint.

## Scripts

| script | role | output (under `RA_WORK("hicfoundation") (= extended_data/ed_table1)`) |
|---|---|---|
| `infer_subset.py` | HiCFoundation task 3 on the test chromosomes only, keeping the full-genome total count that the model takes as input | `tmp_<sp>/<acc>/HiCFoundation_enhanced.pkl` |
| `pkl2hic.py` | HiCFoundation pickle -> `.hic` (handles DNA Zoo scaffold names) | `<sp>/<acc>_enhanced_test.hic`, `multi_10000/<species>_enhanced.hic` |
| `run_human_mouse.sh` | inference + conversion for the 25 human/mouse test maps (`hm_25.txt`) | `human/`, `mouse/` |
| `run_dnazoo.sh` | inference + conversion for the 177 DNA Zoo species (`species_177.txt`) | `multi_10000/` |
| `run_eval_hm.sh` | Evo2HiC evaluation (`evaluate.general_eval`, tasks `diagonal` and `chrom`) per human/mouse map | `<sp>/eval_results_<task>_10000/results.tsv` |
| `run_eval_sr.sh` | Evo2HiC batch evaluation (`evaluate.batch_evaluate.general_eval_batch`) | `<dir>/eval_results_<task>_10000/results.tsv` |
| `summarize_sr.py` | merge HiCFoundation with the Evo2HiC / Hi-C-only / HiCARN2 values (`result/`), means, wins, paired Wilcoxon | `summary_sr.tsv`, `table_<dataset>_<metric>.tsv` -> `edtable:hicf` b |
| `infer_epi.py` | HiCFoundation task 4 (epigenomic profiles, 1 kb) on chr9/chr10 with the full-genome count | `epi/<cell>/HiCFoundation_epigenomic_assay_prediction.pkl` |
| `run_epi_gm_h1.sh` | `infer_epi.py` for GM12878 and H1ESC | `epi/{GM12878,H1ESC}/` |
| `eval_epi.py` | PCC at 2 kb against the Evo2HiC ground-truth tracks (5 shared assays) | `epi/<cell>/result.tsv` -> `edtable:hicf` a |

## Order of runs

```bash
cd plot_revision/extended_data/ed_table1
W=$(cd ${EVO2HIC_RA_WORK:-..} && pwd)/hicfoundation
# resolution enhancement
bash run_human_mouse.sh hm_25.txt 0
bash run_dnazoo.sh species_177.txt 0            # split the list across GPUs if needed
bash run_eval_hm.sh human ; bash run_eval_hm.sh mouse
bash run_eval_sr.sh multi
python summarize_sr.py
# epigenomic prediction
bash run_epi_gm_h1.sh
$HICFOUNDATION_PY infer_epi.py $EVO2HIC_HIC_DATA/data/hic/raw_hic/4DNFITUOMFUQ.hic $W/epi/K562 0 chr9,chr10
cd $EVO2HIC_REPO
for c in GM12878 H1ESC K562; do
  python plot_revision/extended_data/ed_table1/eval_epi.py $c $W/epi/$c $EVO2HIC_CKPT_ROOT/epi_prediction/model/track/$c
done
```

`run_eval_hm.sh` / `run_eval_sr.sh` call the Evo2HiC evaluation package (`evaluate.general_eval`,
`evaluate.batch_evaluate.general_eval_batch`) from the Evo2HiC repo root.
