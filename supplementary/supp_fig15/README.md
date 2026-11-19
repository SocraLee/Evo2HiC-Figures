# Supplementary Fig. 15 — Representative loci of the cell type-differential analysis

| file | role | output |
|---|---|---|
| `plot.py` | figure (one 100 kb window per assay) | `supp_fig15.pdf, supp_fig15.png, supp_fig15_loci.tsv` |

Inputs: as Extended Data Fig. 5 (`../../extended_data/ed_fig5/crosscell_eval.py`).

```bash
cd plot_revision && PYTHONPATH=..:$PYTHONPATH python supplementary/supp_fig15/plot.py
```

Figures are written to `EVO2HIC_OUT_DIR` (default `../Figures`). Paths and environment variables: `../../paths.py`.
