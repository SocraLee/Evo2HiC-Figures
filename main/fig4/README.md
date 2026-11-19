# Fig. 4 — Epigenomic signal prediction

| file | role | output |
|---|---|---|
| `plot.ipynb` | prediction tracks and PCC panels | `fig4.pdf, fig4_stats.tsv` |

Inputs: EPI_EVO2HIC_DIR (inference with the released `epi_prediction` checkpoint), measured tracks.

Run the notebook `main/fig4/plot.ipynb` (from any folder inside the repo; it locates `paths.py` itself), with the Evo2HiC repo on `PYTHONPATH`.

Figures are written to `EVO2HIC_OUT_DIR` (default `../Figures`). Paths and environment variables: `../../paths.py`.
