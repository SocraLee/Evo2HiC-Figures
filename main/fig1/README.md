# Fig. 1 — Overview of Evo2HiC

| file | role | output |
|---|---|---|
| `plot_hic_epigenomics.ipynb` | Hi-C and epigenomic panels | `fig1_hic.pdf, fig1_epigenomics.pdf` |
| `plot_multispecies.ipynb` | multispecies (DNA Zoo) panel | `fig1_multispecies.pdf` |
| `plot_dnazoo.ipynb` | circular DNA Zoo improvement panel | `fig1_dnazoo_*.pdf` |

Inputs: result/fig1, result/dnazoo (SR metrics per species); raw Hi-C, tracks and DNA (EVO2HIC_HIC_DATA).

Run the notebook `main/fig1/plot_hic_epigenomics.ipynb` (from any folder inside the repo; it locates `paths.py` itself), with the Evo2HiC repo on `PYTHONPATH`.

Figures are written to `EVO2HIC_OUT_DIR` (default `../Figures`). Paths and environment variables: `../../paths.py`.
