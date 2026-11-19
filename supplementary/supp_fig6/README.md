# Supplementary Fig. 6 — Epigenomic prediction (MSE)

| file | role | output |
|---|---|---|
| `plot.py` | figure | `supp_fig6.pdf, supp_fig6_stats.tsv` |

Inputs: EPI_EVO2HIC_DIR; Hi-C-only and Evo 2 baselines in data/epi_baselines/.

```bash
cd plot_revision && PYTHONPATH=..:$PYTHONPATH python supplementary/supp_fig6/plot.py
```

Figures are written to `EVO2HIC_OUT_DIR` (default `../Figures`). Paths and environment variables: `../../paths.py`.
