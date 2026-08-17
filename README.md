# Evo2HiC-Figures

Companion folder for the main [Evo2HiC](../README.md) repo. Holds the
scripts (and Jupyter notebooks) that regenerate every figure in the paper
from a single config file (`paths.py`).

## Prerequisites

If already set up Evo2HiC environment, you can skip the following steps.

1. Create and activate the plotting environment:
   ```bash
   conda env create -f environment.yaml
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

Then point the three data roots at your local copies in [paths.py](paths.py)):

```bash
EVO2HIC_CKPT_ROOT="/path/to/checkpoints"   # released model checkpoints
EVO2HIC_HIC_DATA="/path/to/hic_data"      # raw Hi-C / DNA / embeddings
EVO2HIC_OUT_DIR="/path/to/figures"        # optional
```

`paths.py` is generally ordered **by figure** (Fig 1 → Supp 22); each section lists the checkpoints/data it needs. For checkpoint sub-folders, you can download the released artifact or symlink your local run to it.

### Reproduce figures

Choose the script of interest, read the docstring on top and configure the related path variables.

```bash
# .py scripts
python scripts/plot_supp19.py
```

## Layout

```
Evo2HiC-Figures/
├── paths.py              
├── plot_settings.py      
├── plot_utils.py         
├── _supp_style.py  
├── environment.yaml
├── scripts/              # one file per figure
└── helper/               # producers for intermediate results
```