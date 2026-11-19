# Fig. 6 — Cross-species analysis on DNA Zoo

| file | role | output |
|---|---|---|
| `plot.ipynb` | resolution enhancement in human, mouse and DNA Zoo | `fig6.pdf, fig6_stats.tsv` |

Inputs: result/{human,mouse,multi} (SR metrics; producer evaluate/general_eval.py, see helper/README.md), data/ phylogeny and clade lists.

Run the notebook `main/fig6/plot.ipynb` (from any folder inside the repo; it locates `paths.py` itself), with the Evo2HiC repo on `PYTHONPATH`.

Figures are written to `EVO2HIC_OUT_DIR` (default `../Figures`). Paths and environment variables: `../../paths.py`.
