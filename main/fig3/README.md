# Fig. 3 — Genome structure prediction from sequence

| file | role | output |
|---|---|---|
| `plot.ipynb` | Seq2HiC vs Orca and Evo 2 (predictions computed in the notebook) | `fig3.pdf, fig3_stats.tsv` |

Inputs: released `seq2hic_{H1ESC,HFF}` checkpoints, Seq2HiC ablations (not released), Orca predictions, raw Hi-C.

Run the notebook `main/fig3/plot.ipynb` (from any folder inside the repo; it locates `paths.py` itself), with the Evo2HiC repo on `PYTHONPATH`.

Figures are written to `EVO2HIC_OUT_DIR` (default `../Figures`). Paths and environment variables: `../../paths.py`.
