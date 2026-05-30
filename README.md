# plot_revision

Companion folder for the main [Evo2HiC](../README.md) repo. Holds the
scripts (and Jupyter notebooks) that regenerate every figure in the paper
from a single config file (`paths.py`).

## Prerequisites

1. Create and activate the plotting environment:
   ```bash
   conda env create -f plot_revision/plot_revision.yaml
   conda activate Evo2HiC-plot
   ```
2. Add the repo root to PYTHONPATH (needed by scripts that import
   `dataset.*` / `config`):
   ```bash
   export PYTHONPATH=$(pwd):$PYTHONPATH
   ```
3. (Optional) Override any path via env var — see top of
   [paths.py](paths.py). The defaults match the lab-server layout that
   produced the published figures.

## Usage

### Reproduce figures

```bash
# .py scripts
python plot_revision/scripts/plot_supp13.py
```

PDFs are written to `Figures/` under the repo root by default. Override the
output directory with `EVO2HIC_OUT_DIR=/path/to/out` before running.

## Layout

```
plot_revision/
├── paths.py              # single source of truth for every external path
├── plot_settings.py      # color palette
├── plot_utils.py         # box/strip + significance helpers
├── _supp_style.py        # supp-figure rcParams + clade map
├── plot_revision.yaml    # conda env recipe
└── scripts/              # one file per figure
```

## Regenerating derived inference outputs

A few notebooks read files that are *outputs* of the inference scripts
rather than primary data. Every derived path in [paths.py](paths.py) is a
small helper whose **docstring contains the exact command that produces
it** — see the `Derived inference outputs` section at the bottom of the
file (`encoder_embed`, `retrieval_sim`, `retrieval_recall`,
`epi_attr_seq`, `epi_attr_track`). The inference scripts always write
under `<dirname(ckpt)>/<save_dir>/...`, so the helpers and the scripts
agree by construction.

| Experiment | Generator | Already shipped? |
| --- | --- | --- |
| Epigenomic prediction track npy | `inference/inference_CDNA1d.py` | ✓ |
| Epigenomic attribution (Fig 5) | `inference/inference_CDNA1d.py --interpretation` | partial |
| Retrieval pretrain embeddings + scores | `inference/inference_dna_embeds.py` + `inference/retrieval_siglip.py` | only for the alignment-baseline ckpt |
| Seq2HiC per-locus predictions | run live inside notebook via `inference.inference_CDNA2d.load_model(...)` | — |
| Hi-C SR enhanced `.hic` | `inference/inference_CDNA2d.py` + `finalize_hic.sh` | ✓ |

## Status

| Bucket | Scripts |
| --- | --- |
| ✓ Works in `Evo2HiC-plot` env | `plot_supp{3,7,8,10,11,12,13,14,16,17,19}`, `plot_Fig1_hic_epigenomics`, `plot_Fig4_Epi` |
| ✓ Should work once derived outputs are generated | `plot_supp{1,2}`, `plot_Fig2` |
| ✓ Should work once GPU inference completes | `plot_Fig3_Seq2HiC`, `plot_supp4_5_6` |
| 🐢 Slow (~15 min, may hang) | `plot_supp18` (raw Evo 2 chr10 UMAP) |

## Adding a new figure

**Source is `.py`** — copy to `scripts/`, replace its top-of-file path
constants with imports from `paths.py`, change the output to
`ensure_out_dir() / "<name>.pdf"`.

**Source is `.ipynb`** — copy to `scripts/`, then add this cell at the top:

```python
import sys
from pathlib import Path
_HERE = Path.cwd().resolve()
for cand in (_HERE.parent, _HERE / "plot_revision", _HERE):
    if (cand / "paths.py").exists():
        sys.path.insert(0, str(cand))
        break
from paths import REPO, ensure_out_dir, add_repo_to_syspath, ...   # whatever
OUT_DIR = ensure_out_dir()
add_repo_to_syspath()
```

Then rewrite path literals to use the imports, and change every
`savefig('../Figures/<name>')` to `savefig(str(OUT_DIR / '<name>'))`.
