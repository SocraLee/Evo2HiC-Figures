# Fig. 2 — Embedding space of Evo2HiC

| file | role | output |
|---|---|---|
| `plot.ipynb` | retrieval and UMAP panels | `fig2.pdf` |

Inputs: retrieval outputs of the released `pretrained_weights` model and of the w/o-Evo 2 ablation (PRETRAIN_CKPT_DIR, PRETRAIN_NOEVO_CKPT_DIR), Evo 2 embeddings, TRACKS_GM12878_CHR10_NPY (see top-level README).

Run the notebook `main/fig2/plot.ipynb` (from any folder inside the repo; it locates `paths.py` itself), with the Evo2HiC repo on `PYTHONPATH`.

Figures are written to `EVO2HIC_OUT_DIR` (default `../Figures`). Paths and environment variables: `../../paths.py`.
