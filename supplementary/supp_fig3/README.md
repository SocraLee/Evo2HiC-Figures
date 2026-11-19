# Supplementary Fig. 3 — UMAP of Evo 2 embeddings

| file | role | output |
|---|---|---|
| `plot.py` | figure | `supp_fig3.pdf` |

Inputs: Evo 2 embeddings (EVO2_EMB_DIR), TRACKS_GM12878_CHR10_NPY.

```bash
cd plot_revision && PYTHONPATH=..:$PYTHONPATH python supplementary/supp_fig3/plot.py
```

Figures are written to `EVO2HIC_OUT_DIR` (default `../Figures`). Paths and environment variables: `../../paths.py`.
