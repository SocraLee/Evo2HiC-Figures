# plot_revision/helper

Producer scripts for the result tables that the figure scripts read. The tables
are **shipped** under `result/<figure>/`, so you normally never run these — they
are here so a reader can see *where each table comes from* and regenerate it.

All commands run from the repo root and use the training env (they import
`config`, `dataset.*`, `evaluate.*`, `hic_utils`, `utils`):

```bash
cd $EVO2HIC_REPO
PY=$(conda run -n Evo2HiC which python)
```

The producing helper writes each table into **every** figure folder that
consumes it (e.g. the SR metric CSVs go into `result/fig6/`, `result/supp16/`,
`result/supp17/`, `result/supp22/`, `result/fig1/`, `result/dnazoo/`).

| Result table (per-figure copies) | Producer | Command |
|---|---|---|
| `<fig>/{human,mouse,multi}/{PCC,SPC,PSNR,SSIM}.csv` (SR benchmark) | `helper/general_eval.py` | `$PY -m evaluate.general_eval --task chrom …` per method, one column per method, then collate (rows = test Hi-C maps, cols = methods incl. HiCARN2) |
| `supp12/{crosscell_matrix,spec_gap,spec_pcc,williams_pvals}.tsv` | `helper/crosscell_eval.py` | `$PY -m evaluate.crosscell_eval` (writes to `result/crosscell/`, then copied to `result/supp12/`) |
| `supp19/multi/TAD_revision_evo2_vs_hicarn2.tsv` | `helper/multi_TAD_eval_cooltools.py` | `$PY -m evaluate.multi_TAD_eval_cooltools --baseline hicarn2 --output result/supp19/multi/TAD_revision_evo2_vs_hicarn2.tsv` |
| `fig5/motif_enrichment_stats_H3K27ac.csv` | `helper/motif_analysis.ipynb` | run the notebook (motif SHAP enrichment); writes the enrichment stats CSV |
| `fig2/rank_{human,mouse}.csv`, `supp1/gap_{human,mouse}.csv`, `supp13/*.tsv`, `fig5/Fig5f_*.tsv` | the figure script itself (self-cache) | produced on first run of the corresponding `plot_*` script |

`helper/eval_utils.py` is the shared dependency of the eval producers (copied
for reference; the canonical copy is `evaluate/eval_utils.py`).

## Supp 10 — Borzoi & AlphaGenome baselines (must be downloaded)

The main repo does **not** ship a `baselines/` folder. Regenerate the two
external DNA→epigenome baselines and place their outputs where `paths.py`
expects them (`RESULT_DIR/supp10/{borzoi,alphagenome}/`, or set
`EVO2HIC_BORZOI_DIR` / `EVO2HIC_ALPHAGENOME_DIR`). Each folder holds
`<cell>/{9,10}.npy` (shape `5×n_bins`) + `result.tsv` (per cell×chr×track PCC).

### Borzoi (public checkpoint)

Zero-shot DNA→epigenome baseline using the **pretrained Borzoi** model via the
`borzoi-pytorch` HuggingFace port (no fine-tuning; we pick the 15 cell×assay
heads relevant to us).

- Model: `calico/borzoi` weights, loaded through
  [`borzoi-pytorch`](https://github.com/johahi/borzoi-pytorch)
  (`pip install borzoi-pytorch`; already in the Evo2HiC env).
- Reference: Linder et al., Borzoi (github.com/calico/borzoi).

```bash
$PY -m helper.borzoi_inference          # sliding-window inference on chr9+chr10
$PY -m helper.borzoi_eval               # -> result.tsv (per-track PCC)
```

### AlphaGenome (public checkpoint)

- Model: Google DeepMind **AlphaGenome** (released fold-1 weights) via the
  official `alphagenome` API.

```bash
$PY -m helper.alphagenome_inference
$PY -m helper.alphagenome_eval
```

Both write the same schema as `inference/inference_CDNA1d.py`, so their rows are
directly comparable to the Evo2HiC epigenomic-prediction table.
