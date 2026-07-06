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

## Usage

### Reproduce figures

```bash
# .py scripts
python plot_revision/scripts/plot_supp19.py
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
small helper whose **docstring contains the instruction to produce
it**.