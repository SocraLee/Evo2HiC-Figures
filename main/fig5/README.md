# Fig. 5 — Cell type-specific motifs

| file | role | output |
|---|---|---|
| `motif_analysis.ipynb` | motif enrichment statistics (writes result/motif_enrichment_stats_*.csv) |  |
| `plot.ipynb` | figure | `fig5.pdf` |

Inputs: attribution arrays HUMAN_INTER_DIR (interpretation script of the main repo on the released `epi_prediction` model), data/motifs.meme, data/motif_annotations.xlsx, data/human_genes_merged.csv, chr9 motif matches (FIMO, MOTIF_MATCH_CHR9_BED).

Run the notebook `main/fig5/motif_analysis.ipynb` (from any folder inside the repo; it locates `paths.py` itself), with the Evo2HiC repo on `PYTHONPATH`.

Figures are written to `EVO2HIC_OUT_DIR` (default `../Figures`). Paths and environment variables: `../../paths.py`.
