# Extended Data Fig. 6 — Ablation against the Hi-C-only variant

| file | role | output |
|---|---|---|
| `plot.py` | figure | `ed_fig6.pdf, ed_fig6_stats.tsv` |

Inputs: result/{human,mouse,multi} (SR metrics incl. the Hi-C-only ablation), data/ phylogeny and clade lists.

```bash
cd plot_revision && PYTHONPATH=..:$PYTHONPATH python extended_data/ed_fig6/plot.py
```

Figures are written to `EVO2HIC_OUT_DIR` (default `../Figures`). Paths and environment variables: `../../paths.py`.
