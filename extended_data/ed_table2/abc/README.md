# Enhancer nomination with ABC in the held-out cell line K562

Reproduces Extended Data Table 2 "Enhancer nomination and loop-anchor classification in unseen cellular
contexts" (`edtable:func`) panel a, and the pair counts in the Methods (5,786 pairs, 44 positives, 808 genes;
5,462 distal pairs with 18 positives).

Activity-by-Contact (Fulco et al. 2019) for K562, which is held out from pretraining and track fine-tuning, on
the held-out chromosomes 9 and 10: activity = sqrt(DNase x H3K27ac) from the predicted tracks of Evo2HiC or of
the Hi-C-only ablation (measured K562 tracks as an upper bound), contact = measured K562 Hi-C (4DNFITUOMFUQ),
10 kb, SCALE, no pseudocount; predicted tracks are quantile-normalised to the measured marginals of the
training cell GM12878 (`--calib 1`; no K562 data). Baselines: genomic distance and Hi-C contact. Outputs go to
`RA_WORK("abc") (= extended_data/ed_table2/abc)`.

## Scripts

| script | role | output |
|---|---|---|
| `abc_grid.py` | scores every element-gene pair of the benchmark (distance, contact, ABC from predicted and measured tracks) | `<out>/K562_pairs_res10000_pseudo0.tsv` |
| `combo/summarize.py` | pair counts and AUPRC with gene-level bootstrap CIs, all and distal (>= 50 kb) pairs | stdout -> `edtable:func` a (AUPRC) |
| `combo/test_and_recall.py` | distal pairs: one-sided paired permutation test of AUPRC differences (10,000 permutations), recall@K | stdout -> `edtable:func` a (p, Recall@25/50/100) |

## Inputs

* Evo2HiC K562 tracks: `EPI_EVO2HIC_DIR/K562/{9,10}.npy`, inference with the released `epi_prediction` checkpoint
  (Zenodo, doi:10.5281/zenodo.17917911).
* Hi-C-only K562 tracks: `../../../data/epi_baselines/hic_only/K562/{9,10}.npy` (shipped; the checkpoint is not released,
  training configuration `../../../baselines/epi_baseline_hic_only/args.json`).
* Benchmark `../../../data/crispr/bench_gasperini_morris_chr910.tsv.gz` (shipped):
  chr9/10 pairs of the K562 CRISPRi screens of Gasperini et al. 2019 (GEO GSE120861; all DHS-gene pairs
  6 kb-1 Mb, positive = empirical FDR < 0.1 and decreased expression, lifted to hg38) and Morris et al. 2023
  (as processed by ENCODE-rE2G), in the column format of the ENCODE-rE2G benchmark. It is the default `--bench`
  of `abc_grid.py`.

## Order of runs

```bash
cd $EVO2HIC_REPO
A=plot_revision/extended_data/ed_table2/abc                       # scripts
W=${EVO2HIC_RA_WORK:-plot_revision}/extended_data/ed_table2/abc   # outputs
python $A/abc_grid.py --res 10000 --pseudo 0 --calib 1 --out $W/combo/out_evo
python $A/abc_grid.py --res 10000 --pseudo 0 --calib 1 --pred-dir plot_revision/data/epi_baselines/hic_only --out $W/combo/out_hic
python $A/combo/summarize.py
python $A/combo/test_and_recall.py
```
