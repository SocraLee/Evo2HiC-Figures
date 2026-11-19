# Extended Data Fig. 3 — Comparison with Borzoi

| file | role | output |
|---|---|---|
| `plot.py` | figure | `ed_fig3.pdf` |
| `borzoi_inference.py, borzoi_eval.py` | Borzoi predictions (public weights) -> result/borzoi |  |
| `alphagenome_inference.py, alphagenome_eval.py` | AlphaGenome predictions (public API) -> result/alphagenome |  |

Inputs: EPI_EVO2HIC_DIR; Borzoi / AlphaGenome predictions regenerated with the scripts here.

```bash
cd plot_revision && PYTHONPATH=..:$PYTHONPATH python extended_data/ed_fig3/plot.py
```

Figures are written to `EVO2HIC_OUT_DIR` (default `../Figures`). Paths and environment variables: `../../paths.py`.
