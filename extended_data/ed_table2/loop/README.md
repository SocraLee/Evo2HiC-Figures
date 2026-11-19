# Loop-anchor classification in a held-out cell line, with baselines and orthogonal validation

Reproduces Extended Data Table 2 "Enhancer nomination and loop-anchor classification in unseen cellular
contexts" (`edtable:func`) panels b (ENCODE cCRE classes) and c (orthogonal validation), and Supplementary
Table "Loop-anchor classification under the rule-based classes" (`supptable:loop-rule`).

Cell lines: GM12878 and H1ESC (training cells) and K562 (held out from pretraining and track fine-tuning);
anchors = HiCCUPS loop anchors on chr9 and chr10. Methods: Evo2HiC, the Hi-C-only ablation and the
sequence-only Evo 2 baseline, each classifying anchors from its own predicted tracks; labels from the
measured tracks. Outputs go to `RA_WORK("loop") (= extended_data/ed_table2/loop)`.

## Scripts

| script | role | output | manuscript |
|---|---|---|---|
| `rule_on_predicted.py` | helper: anchor states and the rule-based classes (CTCF / active / repressive / quiescent) | imported | |
| `loop_anchor_eval.py` | rule-based classes, 1,000 bootstrap CIs, label-shuffle chance | `loop_anchor_metrics.tsv`, `loop_anchor_labels.npz` | `supptable:loop-rule` |
| `loop_anchor_ccre.py` | ENCODE registry of cCREs classes (promoter-like / enhancer-like / CTCF-only / low-DNase; "high" = 95th percentile of the 2 kb-bin distribution within each signal source) | `loop_anchor_ccre.tsv`, `loop_anchor_ccre_labels.npz` | `edtable:func` b |
| `orthogonal_expression.py` | RNA-seq: expression of genes at anchors predicted promoter-like vs CTCF-only / low-DNase (Mann-Whitney, AUROC) | `orthogonal_expression.tsv` | `edtable:func` c, expression AUROC |
| `orthogonal_ctcf_motif.py` | CTCF motif (JASPAR MA0139.2) at predicted CTCF+ vs CTCF- anchors (AUROC of the max log-odds) and convergent orientation of loops with both anchors CTCF+ | `orthogonal_ctcf_motif_p<thr>.tsv` | `edtable:func` c, motif AUROC and convergence (p < 1e-5) |

## Inputs

* Predicted tracks `<dir>/<cell>/<chr>.npy` (chr9, chr10): Evo2HiC `EPI_EVO2HIC_DIR` (inference with the released
  `epi_prediction` checkpoint); Hi-C-only and Evo 2 predictions shipped in `../../../data/epi_baselines/{hic_only,evo2}/`
  (checkpoints not released; training configurations in `../../../baselines/`).
* GENCODE v38 TSSs on chr9/10: `../../../data/gencode_v38_tss_chr910.tsv` (shipped).
* Measured tracks: `config.hic2tarck_dir`; gene TPMs: `data/dna/human/merged.csv`; motifs:
  `data/dna/motifs.meme` (under `$EVO2HIC_HIC_DATA`).
* HiCCUPS loops: GM12878 and H1ESC from `$EVO2HIC_HIC_DATA/data/hic/hiccups_loop/<acc>/merged_loops.bedpe`.
  K562 (4DNFITUOMFUQ) is called here, with the parameters used for the other two cell lines:

  ```bash
  W=${EVO2HIC_RA_WORK:-plot_revision}/extended_data/ed_table2/loop
  java -jar juicer_tools.2.20.00.jar hiccups --cpu -c chr9,chr10 -r 10000 -k SCALE -f .1 -p 2 -i 5 \
       -t 0.02,1.5,1.75,2 -d 20000 $EVO2HIC_HIC_DATA/data/hic/raw_hic/4DNFITUOMFUQ.hic $W/hiccups/4DNFITUOMFUQ
  ```

## Order of runs

```bash
cd $EVO2HIC_REPO
L=plot_revision/extended_data/ed_table2/loop
python $L/loop_anchor_eval.py --cells GM12878,H1ESC,K562 --boot 1000
python $L/loop_anchor_ccre.py --cells GM12878,H1ESC,K562 --boot 1000
python $L/orthogonal_expression.py
python $L/orthogonal_ctcf_motif.py 1e-5
```
