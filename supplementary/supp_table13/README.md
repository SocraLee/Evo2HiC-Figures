# Supplementary Table 13 — Per-assay effect sizes of the cell type-differential analysis

Differential PCC = corr(Pred_A - Pred_B, GT_A - GT_B) over all 2 kb bins of chr9 and chr10, averaged over the three
pairs of GM12878, H1ESC and K562 (K562 held out), for Evo2HiC, the Hi-C-only ablation and the sequence-only Evo 2
baseline (metric of Extended Data Fig. 5, `../../extended_data/ed_fig5/crosscell_eval.py`). Effect size = Evo2HiC
minus the strongest baseline per assay, 95 % CI from a paired block bootstrap over genomic bins (100 kb blocks,
2,000 resamples).

| file | output |
|---|---|
| `effect_sizes.py` | `effect_sizes.tsv`, `effect_sizes.png` (written to `RA_WORK("supp_table13")`, default this folder) |

```bash
cd $EVO2HIC_REPO
python plot_revision/supplementary/supp_table13/effect_sizes.py
```

Inputs: Evo2HiC tracks (`EPI_EVO2HIC_DIR`, released `epi_prediction` checkpoint), Hi-C-only and Evo 2 tracks
(`../../data/epi_baselines/`; checkpoints not released, configurations in `../../baselines/`), measured tracks in
`config.hic2tarck_dir`.
