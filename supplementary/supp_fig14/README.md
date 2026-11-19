# Supplementary Fig. 14 — Pretraining representation

| file | role | output |
|---|---|---|
| `plot.py` | figure | `supp_fig14.pdf` |

Inputs: result/{human,mouse}, EPI_EVO2HIC_DIR/result.tsv, data/epi_baselines/nodistill/result.tsv.

```bash
cd plot_revision && PYTHONPATH=..:$PYTHONPATH python supplementary/supp_fig14/plot.py
```

Figures are written to `EVO2HIC_OUT_DIR` (default `../Figures`). Paths and environment variables: `../../paths.py`.
