# Extended Data Fig. 4 — Epigenomic prediction ablations

| file | role | output |
|---|---|---|
| `plot.py` | figure | `ed_fig4.pdf` |

Inputs: EPI_EVO2HIC_DIR; no-distillation, Evo 2-embedding + Hi-C and ridge baselines shipped in data/epi_baselines/ (configs in baselines/).

```bash
cd plot_revision && PYTHONPATH=..:$PYTHONPATH python extended_data/ed_fig4/plot.py
```

Figures are written to `EVO2HIC_OUT_DIR` (default `../Figures`). Paths and environment variables: `../../paths.py`.
