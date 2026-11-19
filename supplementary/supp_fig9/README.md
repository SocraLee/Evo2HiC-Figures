# Supplementary Fig. 9 — Representative loops

| file | role | output |
|---|---|---|
| `plot.py` | figure | `supp_fig9.pdf` |

Inputs: EPI_EVO2HIC_DIR, measured tracks, raw Hi-C.

```bash
cd plot_revision && PYTHONPATH=..:$PYTHONPATH python supplementary/supp_fig9/plot.py
```

Figures are written to `EVO2HIC_OUT_DIR` (default `../Figures`). Paths and environment variables: `../../paths.py`.
