# Extended Data Fig. 5 — Cross-cell-type evaluation

| file | role | output |
|---|---|---|
| `plot.py` | figure | `ed_fig5.pdf` |
| `crosscell_eval.py` | cross-cell tables -> result/ |  |

Inputs: EPI_EVO2HIC_DIR; Hi-C-only and Evo 2 baselines in data/epi_baselines/; measured tracks.

```bash
cd plot_revision && PYTHONPATH=..:$PYTHONPATH python extended_data/ed_fig5/plot.py
```

Figures are written to `EVO2HIC_OUT_DIR` (default `../Figures`). Paths and environment variables: `../../paths.py`.
