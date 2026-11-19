# Comparison with Orca in Orca's setting

Reproduces main Table 1 "Comparison with Orca in Orca's setting" (`table:orca`) and the
Methods paragraph "Comparison in Orca's setting": Evo2HiC fine-tuned on Orca's coarse-grained Micro-C
target with Orca's auxiliary peak supervision (**Evo2HiC-cg**), evaluated against Orca's released
predictions on Orca's 417 test windows (chr9: 213, chr10: 204; 1 Mb at 4 kb), Orca's target matrices
and Orca's validity mask.

All outputs (tensors, checkpoints, logs, result tables) go to `RA_WORK("orca") (= main/table1)` (default: this
folder; see `../../paths.py`).

## Scripts

| script | role | output |
|---|---|---|
| `build_tensors.py` | per-window tensors: Orca's 417 windows, Evo2HiC inference (forward + reverse averaged), Orca predictions / targets / NaN masks from `mallpreds.pth`, `malltargets.pth`, `malltargetnans.pth`, ground-truth maps | `tensors/<cell>.npz`, `tensors/expected_4kb_<cell>.npy`; with a checkpoint: `tensors/<cell>_<tag>.npz` |
| `run_train_cg.sh` | one call of the public trainer `train/train_CDNA2d_Seq2HiC.py` with the Evo2HiC-cg base arguments (training, `--recalibrate-bn`, `--valid-only`) | run folder under `$EVO2HIC_REPO/checkpoints/` |
| `rig_ladder2.py` | full training protocol (two learning-rate stages, BN recalibration, chr8 selection) and the single final test evaluation | `checkpoints/rig_keep2/<cell>/`, `result/rig_ladder_status.log`, `result/rig_test.tsv`, `result/eval_rig2_<cell>_<cell>.tsv` |
| `pcc_only.py` | Orca's metric only: pooled PCC / SPC of log(O/E) against Orca's target, Orca's mask | one line on stdout |
| `eval_cg.py` | full metric panel (pooled PCC/SPC; C.Origami observed/expected PCC, MSE, distance-stratified PCC; insulation-score PCC; TAD-boundary F1; boundary PR-AUC) with paired per-window tests | `result/eval_<tag>_<cell>.tsv` |
| `multimetric.py` | metric implementations (C.Origami panel via the published C.Origami code) | imported by `eval_cg.py` |
| `is_diag.py` | insulation profiles and boundary calls (cooltools, window 100 kb, prominence >= 0.2, +-3 bins) | imported by `eval_cg.py` |
| `boundary_pr.py` | threshold-free boundary PR-AUC | imported by `eval_cg.py` |
| `notebook_eval.py` | helper: Evo2HiC inference and insulation pipeline exactly as in the Fig. 3 notebook | imported by `eval_cg.py`, `is_diag.py` |

## Requirements

* Evo2HiC repo at a version that contains Evo2HiC-cg (CHNFTQ/Evo2HiC `main`, PR #14
  "Add Evo2HiC-cg" and PR #15). The trainer, `dataset/cg_loader.py`, `dataset/peak_loader.py`,
  `scripts/make_cg_targets.py` and `scripts/make_aux_peaks.py` live there.
* Orca resources in `$EVO2HIC_HIC_DATA/orca/resources/`: `4DNFI9GMP2J8.rebinned.mcool`,
  `4DNFI643OYP9.rebinned.mcool`, their `*.expected.res4000.npy`, and the Roadmap peak sets
  `h1esc/h1esc.hg38.bed.sorted.gz`, `hff/hff.hg38.bed.sorted.gz` with their `.features` lists (Orca resources).
* Orca's released test-set predictions in `$EVO2HIC_HIC_DATA/orca_figure/`: `mallpreds.pth`,
  `malltargets.pth`, `malltargetnans.pth`.
* Our ground-truth `.hic` maps 4DNFI2TK7L2F (H1ESC) and 4DNFIPC7P27B (HFF) in `config.hic_data_dir`.
* C.Origami source checkout (`git clone https://github.com/tanjimin/C.Origami`) at `$CORIGAMI_DIR`
  (used for its `insulation_score.py`).
* Released checkpoints (Zenodo, doi:10.5281/zenodo.17917911) unpacked into `$EVO2HIC_CKPT_ROOT`: `pretrained_weights/`
  (initial DNA encoder), `seq2hic_{H1ESC,HFF}/` (original Seq2HiC column) and `seq2hic_cg_{H1ESC,HFF}/` (Evo2HiC-cg).
  The initial encoder is `$EVO2HIC_CKPT_ROOT/pretrained_weights/model.pt`
  (override with `DNA_ENCODER_INIT`).

## Training (Evo2HiC-cg)

1. Training targets, written into the Evo2HiC repo's `config.cg_target_dir`, named by the matching `.hic`
   accession (run from `$EVO2HIC_REPO`):

   ```bash
   R=$EVO2HIC_HIC_DATA/orca/resources
   python scripts/make_cg_targets.py $R/4DNFI9GMP2J8.rebinned.mcool 4DNFI2TK7L2F --scale 4880
   python scripts/make_cg_targets.py $R/4DNFI643OYP9.rebinned.mcool 4DNFIPC7P27B --scale 7300
   python scripts/make_aux_peaks.py $R/h1esc/h1esc.hg38.bed.sorted.gz $R/h1esc/h1esc.hg38.bed.sorted.features 4DNFI2TK7L2F
   python scripts/make_aux_peaks.py $R/hff/hff.hg38.bed.sorted.gz     $R/hff/hff.hg38.bed.sorted.features     4DNFIPC7P27B
   ```

2. Training protocol, run once per cell (four GPUs; `EVAL_GPU` is the GPU used in parallel for
   recalibration and validation):

   ```bash
   EVAL_GPU=1 python rig_ladder2.py H1ESC
   EVAL_GPU=1 python rig_ladder2.py HFF
   ```

   Recipe (fixed before training, identical for both cells; the extra arguments are appended to
   `run_train_cg.sh`):

   | setting | value |
   |---|---|
   | target | `-target cg`, `--augment-resolution 4000` (Orca's coarse-grained Micro-C, 4 kb) |
   | auxiliary loss | Orca's Roadmap peak sets (32 tracks H1ESC, 22 HFF), BCE, `--aux-weight 0.05` |
   | initialisation | pretrained Evo2HiC DNA encoder (`--initialize-dna-encoder`) |
   | encoder dropout | off (`--no-encoder-dropout`) |
   | batch | 4 GPUs x `--batch-size 10`, `--accumulate-step 1` (effective and BatchNorm batch 40, SyncBatchNorm) |
   | precision | `--mixed-precision bf16` |
   | optimiser | Adam, `--warmup-step 1000`, seed 0, checkpoint + chr8 validation every 1000 steps |
   | stage A | `--base-learning-rate 5e-4` from the initialisation, at least 10,000 steps |
   | stage B | `--reset-lr 1e-4` from the global-best checkpoint (`--load-checkpoint`) |
   | stopping | a stage stops after 5 consecutive checkpoints without a new global best |
   | decisions | every checkpoint is BatchNorm-recalibrated on 50 training batches (`--recalibrate-bn`, `--bn-recal-batches 50`, batch 3, seed 0) and validated on chr8 (`--valid-only`); the recalibrated chr8 MSE decides the global best and the stopping counter |
   | reported model | lowest recalibrated chr8 MSE; scored once on the 417 test windows |

   Selected checkpoints in the paper: H1ESC stage A step 17,000; HFF stage A step 4,000. The released
   checkpoints are `seq2hic_cg_H1ESC/` and `seq2hic_cg_HFF/` (`model.pt`, `args.json`).

   The trainer writes its run folders to `$EVO2HIC_REPO/checkpoints/`; `rig_ladder2.py` copies each
   checkpoint it evaluates to `checkpoints/rig_keep2/<cell>/` under `RA_WORK("orca") (= main/table1)`.

## Evaluation

```bash
# 1. tensors for the 417 windows (Orca predictions/targets/masks, ground truths, original Evo2HiC)
python build_tensors.py H1ESC 0
python build_tensors.py HFF 0
# 2. Evo2HiC-cg predictions for a checkpoint (done automatically at the end of rig_ladder2.py)
python build_tensors.py H1ESC 0 $EVO2HIC_CKPT_ROOT/seq2hic_cg_H1ESC/model.pt rig2_H1ESC
python build_tensors.py HFF   0 $EVO2HIC_CKPT_ROOT/seq2hic_cg_HFF/model.pt   rig2_HFF
# 3. Orca's metric and the full panel
python pcc_only.py H1ESC rig2_H1ESC ; python pcc_only.py HFF rig2_HFF
python eval_cg.py  H1ESC rig2_H1ESC ; python eval_cg.py  HFF rig2_HFF
```

Mapping to Table 1 (`table:orca`):

| row | source |
|---|---|
| Pooled PCC / SPC, log(O/E) | `pcc_only.py` (also section A of `eval_cg.py`) |
| Insulation score PCC | `eval_cg.py` section D (cooltools, 100 kb window) |
| Observed/expected PCC, MSE, distance-stratified PCC | `eval_cg.py` section C (C.Origami implementation) |
| TAD-boundary F1, boundary PR-AUC | `eval_cg.py` section D |
| p-values | paired one-sided Wilcoxon over windows, `eval_cg.py` |

Reproduction check: Orca's released predictions scored by this pipeline give pooled PCC/SPC 0.776/0.761
(H1ESC) and 0.737/0.716 (HFF), against the published 0.776/0.762 and 0.740/0.718.
