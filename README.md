# Evo2HiC-Figures

Companion repository of [Evo2HiC](https://github.com/CHNFTQ/Evo2HiC). It contains the code that regenerates every
figure and table of the paper, organised like the paper: one folder per manuscript item under `main/`,
`extended_data/` and `supplementary/`. Each folder holds the plotting script, the analysis code behind its numbers,
the result tables it reads (`result/`) and a short README. All paths are configured in one file, `paths.py`.

## Setup

1. Clone this repository inside the Evo2HiC repository as `plot_revision/` (or set `EVO2HIC_REPO`):
   ```bash
   git clone https://github.com/CHNFTQ/Evo2HiC.git && cd Evo2HiC
   git clone https://github.com/SocraLee/Evo2HiC-Figures.git plot_revision
   ```
2. Create the environment:
   ```bash
   conda env create -f plot_revision/environment.yaml
   conda activate Evo2HiC-plot
   ```
   Model inference and training use the Evo2HiC environment (`Evo2HiC.yaml` of the main repo); HiCFoundation
   inference uses HiCFoundation's own environment (`extended_data/ed_table1`).
3. Download the released checkpoints from Zenodo
   ([doi:10.5281/zenodo.17917911](https://doi.org/10.5281/zenodo.17917911)) and unpack both archives into one
   folder: `pretrained_weights/`, `resolution_enhancement/`, `epi_prediction/`, `seq2hic_{H1ESC,HFF}/`,
   `seq2hic_cg_{H1ESC,HFF}/`.
4. Point the roots at your copies (defaults in `paths.py`):
   ```bash
   export EVO2HIC_CKPT_ROOT=/path/to/checkpoints   # released checkpoints and their inference outputs
   export EVO2HIC_HIC_DATA=/path/to/hic_data       # raw Hi-C / DNA / tracks / DNA Zoo / Orca resources
   export EVO2HIC_OUT_DIR=/path/to/figures         # optional, default ../Figures
   export EVO2HIC_RA_WORK=/path/to/outputs         # optional: outputs of the analysis pipelines (default: their folders)
   ```
5. Run a script from the Evo2HiC repo root with both repositories importable, e.g.
   ```bash
   export PYTHONPATH=plot_revision:.:$PYTHONPATH
   python plot_revision/supplementary/supp_fig6/plot.py
   ```
   Notebooks can be run from any folder inside the repository; they locate `paths.py` themselves.

Figures that show model predictions need the outputs of the inference scripts of the main repo
(`inference/inference_CDNA1d.py`, `inference/inference_CDNA2d.py`, interpretation); they are written next to each
checkpoint (`<model>/model/<save_dir>`), where `paths.py` expects them.

Fig. 2 and Supplementary Figs 2–3 colour embeddings by the GM12878 epigenomic tracks on chr10. This file is not
shipped; generate it from the public ENCODE bigWigs (GM12878 DNase ENCFF960FMM, CTCF ENCFF644EEX, H3K27ac
ENCFF469WVA, H3K27me3 ENCFF486WAD, H3K4me3 ENCFF280PUF; placed in the Evo2HiC `config.hic2tarck_dir/GM12878/` as
listed in `data/hic2track/index.tsv`) with

```bash
cd plot_revision && PYTHONPATH=..:$PYTHONPATH python -m helper.make_tracks_gm12878_chr10   # -> data/tracks_GM12878_10.npy
```

## Layout

```
plot_revision/
├── paths.py, paths.sh   all paths and environment variables (paths.sh: same values for shell scripts)
├── plot_utils.py        plotting helpers and the method palette
├── _supp_style.py       style and clade assignment for the cross-species figures
├── main/                fig1 … fig6, table1
├── extended_data/       ed_fig1 … ed_fig7, ed_table1, ed_table2
├── supplementary/       supp_fig1 … supp_fig16, supp_table1 … supp_table16, supp_data2
├── evaluate/            evaluation code shared by several items (metrics, insulation, TAD boundaries, batch evaluation)
├── data/                small shipped inputs (phylogeny, clade lists, CRISPRi benchmark, motifs, gene TPMs, timing,
│                        chr9/10 predictions of the epigenomic baselines)
├── baselines/           training configurations of baselines whose checkpoints are not released
└── helper/              GM12878 track extraction; index of which script produces each result table
```

## Manuscript item → folder

Numbering follows the current manuscript. A folder that only points to another one is marked →.

| item | folder | contents |
|---|---|---|
| Fig. 1 Overview | `main/fig1` | `plot_hic_epigenomics.ipynb`, `plot_multispecies.ipynb`, `plot_dnazoo.ipynb` |
| Fig. 2 Embedding space | `main/fig2` | `plot.ipynb` |
| Fig. 3 Genome structure prediction from sequence | `main/fig3` | `plot.ipynb` |
| Table 1 Comparison with Orca in Orca's setting | `main/table1` | Evo2HiC-cg training protocol, tensors, metric panel |
| Fig. 4 Epigenomic signal prediction | `main/fig4` | `plot.ipynb` |
| Fig. 5 Cell type-specific motifs | `main/fig5` | `motif_analysis.ipynb`, `plot.ipynb` |
| Fig. 6 Cross-species analysis | `main/fig6` | `plot.ipynb` |
| ED Fig. 1 Embedding space evaluation | `extended_data/ed_fig1` | `plot.ipynb` |
| ED Fig. 2 Seq2HiC distance-stratified metrics and insulation | `extended_data/ed_fig2` | `plot.ipynb` (also Supp. Fig. 4) |
| ED Fig. 3 Comparison with Borzoi | `extended_data/ed_fig3` | `plot.py`, Borzoi / AlphaGenome inference |
| ED Fig. 4 Epigenomic ablations | `extended_data/ed_fig4` | `plot.py` |
| ED Fig. 5 Cross-cell-type evaluation | `extended_data/ed_fig5` | `plot.py`, `crosscell_eval.py` |
| ED Fig. 6 Ablation against the Hi-C-only variant | `extended_data/ed_fig6` | `plot.py` |
| ED Fig. 7 Sequence prior across species | `extended_data/ed_fig7` | cross-species pipeline (also Supp. Fig. 16, Supp. Tables 15–16, Supp. Data 2) |
| ED Table 1 Comparison with HiCFoundation | `extended_data/ed_table1` | HiCFoundation inference and evaluation |
| ED Table 2 Enhancer nomination and loop anchors | `extended_data/ed_table2` | `abc/` (a), `loop/` (b, c; also Supp. Table 14) |
| Supp. Fig. 1 Inference memory and time | `supplementary/supp_fig1` | `plot.ipynb` |
| Supp. Fig. 2 UMAP of Evo2HiC embeddings | `supplementary/supp_fig2` | `plot.ipynb` |
| Supp. Fig. 3 UMAP of Evo 2 embeddings | `supplementary/supp_fig3` | `plot.py` |
| Supp. Fig. 4 Seq2HiC examples | `supplementary/supp_fig4` | → `extended_data/ed_fig2` |
| Supp. Fig. 5 Epigenomic prediction (PCC) | `supplementary/supp_fig5` | `plot.ipynb` |
| Supp. Fig. 6 Epigenomic prediction (MSE) | `supplementary/supp_fig6` | `plot.py` |
| Supp. Fig. 7 ABC enhancer nomination (PPIF) | `supplementary/supp_fig7` | `plot.py` |
| Supp. Fig. 8 Loop-anchor confusion matrices | `supplementary/supp_fig8` | `plot.py`, `label_loops.py` |
| Supp. Fig. 9 Representative loops | `supplementary/supp_fig9` | `plot.py` |
| Supp. Fig. 10 Mouse resolution enhancement | `supplementary/supp_fig10` | `plot.ipynb` |
| Supp. Fig. 11 DNA Zoo resolution enhancement | `supplementary/supp_fig11` | `plot.ipynb` |
| Supp. Fig. 12 Fine-scale structures | `supplementary/supp_fig12` | `plot.py` |
| Supp. Fig. 13 Inference cost | `supplementary/supp_fig13` | `plot.py` |
| Supp. Fig. 14 Pretraining representation | `supplementary/supp_fig14` | `plot.py` |
| Supp. Fig. 15 Cross-cell loci | `supplementary/supp_fig15` | `plot.py` |
| Supp. Fig. 16 Distant-clade examples | `supplementary/supp_fig16` | → `extended_data/ed_fig7` |
| Supp. Tables 1, 2, 12 Accessions, pretraining cell lines | `supplementary/supp_table{1,2,12}` | `.tsv` |
| Supp. Tables 3–10 Statistical tests | `supplementary/README_stats_tables.md` | written by the figure scripts |
| Supp. Table 11 Orca's evaluation setting (first submission) | `supplementary/supp_table11` | → `main/table1` |
| Supp. Table 13 Per-assay effect sizes | `supplementary/supp_table13` | `effect_sizes.py` |
| Supp. Tables 14–16 | `supplementary/supp_table{14,15,16}` | → `extended_data/ed_table2/loop`, `extended_data/ed_fig7` |
| Supp. Data 2 Per-species results | `supplementary/supp_data2` | → `extended_data/ed_fig7` |

## Data and baselines

Public sources (accessions in the Data availability section and Supplementary Data 1): 4DN / ENCODE Hi-C and
Micro-C maps, ENCODE tracks for GM12878 / H1ESC / K562, DNA Zoo maps and assemblies, Orca's resources and released
test-set predictions (github.com/jzhoulab/orca), the K562 CRISPRi screens of Gasperini et al. 2019 (GEO GSE120861)
and Morris et al. 2023 (as processed by ENCODE-rE2G), JASPAR, the TimeTree phylogeny and GENCODE v38.

Baselines and ablations whose checkpoints are not released are represented by their training configuration
(`baselines/`), their chr9/10 epigenomic predictions (`data/epi_baselines/`) and, for the Hi-C super-resolution
baselines, their per-map / per-species results (`extended_data/ed_fig7/result/`, `extended_data/ed_table1/result/`).
Steps that need their raw outputs are marked in the item READMEs.

## License

Apache-2.0 (`LICENSE`).
