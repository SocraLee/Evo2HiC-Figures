# helper/

Producer scripts for the result tables that the figure scripts read. The tables
are **shipped** under `result/<figure>/`, so you normally never run these — they
are here so a reader can see *where each table comes from* and regenerate it.

Run everything **from inside this reproduction folder** (the folder that holds
`paths.py` and `helper/`), with the repo root on `PYTHONPATH` — the producers
import `config`, `dataset.*`, `evaluate.*`, `hic_utils`, `utils` from the repo:

```bash
cd <this folder>                       # the folder that contains helper/
export PYTHONPATH=..:$PYTHONPATH        # repo root, for config / dataset / evaluate
PY=$(conda run -n Evo2HiC which python)
```

Figures read tables from the repo's `result/` tree, i.e. **`../result/`** from
here — so producer outputs below are written there.

These files are **copies** of the in-repo producers (`evaluate/*.py`,
`baselines/epigenomic/*.py`, `plot/motif_analysis.ipynb`), gathered here because
the released repo does not ship a `baselines/` folder. `eval_utils.py` is the
shared dependency of the eval producers.

## Result tables read by figures

| Table (shipped location, under `../result/`) | Producer | Regenerate |
|---|---|---|
| SR metrics `fig6/{human,mouse,multi}/{PCC,SPC,PSNR,SSIM}.csv` (+ copies in supp16/17/18/22, fig1, dnazoo) | `general_eval.py` | one run **per method** (`$PY -m helper.general_eval -t chrom -f0 <pred> -f1 <target> …`) gives per-chrom PCC/SPC/PSNR/SSIM; the shipped matrices collate those into one column per method (rows = test Hi-C maps). This collation was originally done in the legacy `plot/plot_Fig6.ipynb`. |
| `supp12/{crosscell_matrix,spec_gap,spec_pcc,williams_pvals}.tsv` | `crosscell_eval.py` | `$PY -m helper.crosscell_eval` — writes a `result/crosscell/` under the cwd; copy the 4 tsvs into `../result/supp12/`. |
| `supp19/multi/TAD_revision_evo2_vs_hicarn2.tsv` | `multi_TAD_eval_cooltools.py` | `$PY -m helper.multi_TAD_eval_cooltools --baseline hicarn2 --output ../result/supp19/multi/TAD_revision_evo2_vs_hicarn2.tsv` |
| `fig5/motif_enrichment_stats_H3K27ac.csv` | `motif_analysis.ipynb` | run the notebook (motif SHAP enrichment); it writes `result/motif_enrichment_stats_H3K27ac.csv` — move it to `../result/fig5/`. |
| `fig2/rank_*.csv`, `supp1/gap_*.csv`, `supp13/*.tsv`, `fig5/Fig5f_*.tsv` | the figure script itself (self-cache) | produced on first run of the corresponding `plot_*` script |

After (re)producing a shared table, fan it out into every figure folder that
reads it (this resolves `../result/` from its own location, so cwd doesn't
matter):

```bash
$PY -m helper.distribute_tables            # all groups
$PY -m helper.distribute_tables SR_METRICS
```

## Supp 10 — Borzoi & AlphaGenome baselines (must be downloaded)

The main repo does **not** ship a `baselines/` folder. Regenerate the two
external DNA→epigenome baselines into where `paths.py` expects them —
`../result/supp10/{borzoi,alphagenome}/` (or set `EVO2HIC_BORZOI_DIR` /
`EVO2HIC_ALPHAGENOME_DIR`). Each folder holds `<cell>/{9,10}.npy`
(shape `5×n_bins`) + `result.tsv` (per cell×chr×track PCC).

### Borzoi (public checkpoint)

Zero-shot DNA→epigenome baseline using the **pretrained Borzoi** model via the
`borzoi-pytorch` HuggingFace port (no fine-tuning; we pick the 15 cell×assay
heads relevant to us).

- Model: `calico/borzoi` weights, loaded through
  [`borzoi-pytorch`](https://github.com/johahi/borzoi-pytorch)
  (`pip install borzoi-pytorch`; already in the Evo2HiC env).
- Reference: Linder et al., Borzoi (github.com/calico/borzoi).

```bash
$PY -m helper.borzoi_inference --save-dir ../result/supp10/borzoi
$PY -m helper.borzoi_eval      --pred-dir ../result/supp10/borzoi   # -> result.tsv
```

### AlphaGenome (public checkpoint)

- Model: Google DeepMind **AlphaGenome** (released fold-1 weights) via the
  official `alphagenome` API.

```bash
$PY -m helper.alphagenome_inference --save-dir ../result/supp10/alphagenome
$PY -m helper.alphagenome_eval      --pred-dir ../result/supp10/alphagenome
```

Both write the same schema as `inference/inference_CDNA1d.py`, so their rows are
directly comparable to the Evo2HiC epigenomic-prediction table.
