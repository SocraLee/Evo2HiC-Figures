# helper/

Tools used by several manuscript items.

| file | role |
|---|---|
| `make_tracks_gm12878_chr10.py` | GM12878 epigenomic tracks on chr10 at 2 kb from the public ENCODE bigWigs (Fig. 2, Supp. Figs 2–3) |

Producers of the result tables, by item:

| table (under `<item>/result/`) | producer |
|---|---|
| SR metrics `main/fig6/result/{human,mouse,multi}/{PCC,SPC,PSNR,SSIM}.csv` (identical copies shipped in fig1, ed_fig6, supp_fig10/11/14) | `python -m evaluate.general_eval -t chrom -f0 <pred> -f1 <target> ...`, one run per method, collated into one column per method |
| `extended_data/ed_fig5/result/{crosscell_matrix,spec_gap,spec_pcc,williams_pvals}.tsv` | `extended_data/ed_fig5/crosscell_eval.py` |
| `main/fig5/result/motif_enrichment_stats_H3K27ac.csv` | `main/fig5/motif_analysis.ipynb` |
| `supplementary/supp_fig7/result/*.tsv` | `supplementary/supp_fig7/plot.py` (cache, recomputed when deleted) |
| `extended_data/ed_fig3/result/{borzoi,alphagenome}/` | `extended_data/ed_fig3/{borzoi,alphagenome}_{inference,eval}.py` |

Run from this repository's root with the Evo2HiC repo root on `PYTHONPATH` (`cd plot_revision; export PYTHONPATH=..:$PYTHONPATH`).
