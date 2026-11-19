# Supplementary Fig. 7 — ABC enhancer nomination (PPIF)

| file | role | output |
|---|---|---|
| `plot.py` | figure (computes and caches result/*.tsv) | `supp_fig7.pdf` |

Inputs: EPI_EVO2HIC_DIR, measured tracks, raw Hi-C, CRISPRi benchmark of the main repo.

```bash
cd plot_revision && PYTHONPATH=..:$PYTHONPATH python supplementary/supp_fig7/plot.py
```

Figures are written to `EVO2HIC_OUT_DIR` (default `../Figures`). Paths and environment variables: `../../paths.py`.
