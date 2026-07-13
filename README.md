# Evo2HiC-Figures

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

## Usage


### Setting Up

Most scripts require *outputs* of the inference scripts. Therefore, make sure that you have already run the scripts from [`Evo2HiC`](https://github.com/CHNFTQ/Evo2HiC).

Then point the three data roots at your local copies (env vars or edit the
placeholder defaults in [paths.py](paths.py)):

```bash
export EVO2HIC_CKPT_ROOT=/path/to/checkpoints   # released model checkpoints
export EVO2HIC_HIC_DATA=/path/to/hic_data       # raw Hi-C / DNA / embeddings
export EVO2HIC_OUT_DIR=/path/to/figures         # optional; default <repo>/Figures
```

`paths.py` is ordered **by figure** (Fig 1 → Supp 22); each section lists the
checkpoints/data it needs. Checkpoint sub-folders use **clean placeholder names**
(e.g. `hic_sr_evo2hic`) — download the released artifact into the matching name
or symlink your local run to it.

### Reproduce figures

```bash
# .py scripts
python plot_revision/scripts/plot_supp19.py
```

## Layout

```
plot_revision/
├── paths.py              # single config, ordered by figure (clean placeholder names)
├── plot_settings.py      # color palette
├── plot_utils.py         # box/strip + significance helpers
├── _supp_style.py        # supp-figure rcParams + clade map
├── plot_revision.yaml    # conda env recipe
├── scripts/              # one file per figure
└── helper/               # producers for the shipped result tables (+ Borzoi /
                          #   AlphaGenome baselines) — see helper/README.md
```

Result tables are shipped under `result/<figure>/` (one self-contained folder per
figure: `fig1`, …, `fig6`, `supp1`, …). A table read by several figures is copied
into each of their folders; `helper/distribute_tables.py` reproduces that fan-out.
 