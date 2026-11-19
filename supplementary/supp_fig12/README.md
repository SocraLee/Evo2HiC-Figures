# Supplementary Fig. 12 — Fine-scale structures

| file | role | output |
|---|---|---|
| `plot.py` | figure | `supp_fig12.pdf` |

Inputs: DNA Zoo outputs of the released `resolution_enhancement` model (SR_EVO2HIC_DIR) and of HiCARN2 (SR_HICARN2_DIR, not released), DNA Zoo ground truth.

```bash
cd plot_revision && PYTHONPATH=..:$PYTHONPATH python supplementary/supp_fig12/plot.py
```

Figures are written to `EVO2HIC_OUT_DIR` (default `../Figures`). Paths and environment variables: `../../paths.py`.
