# Supplementary Fig. 13 — Inference cost

| file | role | output |
|---|---|---|
| `plot.py` | figure | `supp_fig13.pdf` |

Inputs: data/timing_results.tsv.

```bash
cd plot_revision && PYTHONPATH=..:$PYTHONPATH python supplementary/supp_fig13/plot.py
```

Figures are written to `EVO2HIC_OUT_DIR` (default `../Figures`). Paths and environment variables: `../../paths.py`.
