"""Central path configuration for plot_revision/ reproduction scripts.

Every figure script imports its paths from here, so a reader edits **one**
file to point the reproduction at their local data / checkpoints.

Reader checklist
---------------------------------------
1. Put this repo under the Evo2HiC repo as `plot_revision/` (so REPO auto-resolves),
   or set EVO2HIC_REPO.
2. Download the released checkpoints (Zenodo, doi:10.5281/zenodo.17917911) into
   EVO2HIC_CKPT_ROOT and point EVO2HIC_HIC_DATA at the raw data.

`python paths.py --sh` prints the environment-variable defaults for the shell scripts
(sourced by paths.sh).

Layout conventions
------------------
* every manuscript item has its own folder (main/fig1, extended_data/ed_fig7, supplementary/supp_fig7, …);
  its result tables live in `<item>/result/` (`RESULT(item)`). A table consumed by several items is
  shipped in each of their folders, so every item folder is self-contained.
* This file is ordered by manuscript item (current numbering). Paths shared by
  many figures are defined once in the "Shared inputs" block; each figure
  section lists which shared paths it also needs.
"""
from __future__ import annotations

import os
from pathlib import Path


# ===========================================================================
# Global roots
# ===========================================================================
REPO = Path(os.environ.get(
    "EVO2HIC_REPO",
    Path(__file__).resolve().parents[1],
))

OUT_DIR = Path(os.environ.get(          # produced PDFs / stats
    "EVO2HIC_OUT_DIR",
    REPO / "Figures",
))

PLOT_DIR = Path(__file__).resolve().parent                 # this repo
MAIN_DIR = PLOT_DIR / "main"                     # main figures and Table 1
ED_DIR   = PLOT_DIR / "extended_data"            # Extended Data figures and tables
SUPP_DIR = PLOT_DIR / "supplementary"            # Supplementary figures, tables and data
def RESULT(item: str) -> Path:
    """Result tables of one manuscript item, e.g. RESULT('main/fig6'); shipped in <item>/result/."""
    return PLOT_DIR / item / "result"
DATA_DIR = PLOT_DIR / "data"            # small input data shipped in this repo

# Model checkpoints. The released Evo2HiC models follow the Zenodo layout
# (doi:10.5281/zenodo.17917911): pretrained_weights/, resolution_enhancement/,
# epi_prediction/, seq2hic_<cell>/, seq2hic_cg_<cell>/ (each with model.pt + args.json).
# Inference outputs are written next to the checkpoint (<name>/model/<save_dir>).
# Baseline / ablation checkpoints are not released (marked below).
CKPT_ROOT = Path(os.environ.get(
    "EVO2HIC_CKPT_ROOT",
    REPO / "checkpoints",
))

# Raw data (Hi-C .hic, DNA FASTA, Evo 2 embeddings, hic2track BigWigs, DNA Zoo).
# Very large — always point this at your local copy via the env var.
HIC_DATA_ROOT = Path(os.environ.get(
    "EVO2HIC_HIC_DATA",
    REPO / "hic_data",
))


# ===========================================================================
# Shared inputs
# ===========================================================================

# ---- Raw genome / Hi-C / tracks (config.* mirrors) ------------------------
DNA_FASTA = {                                   # config.DNA_map
    "human":     HIC_DATA_ROOT / "data/dna/human/hg38.fa",
    "mouse":     HIC_DATA_ROOT / "data/dna/mouse/mm10.fa",
    "zebrafish": HIC_DATA_ROOT / "data/dna/zebrafish/danRer11.fa",
}
HIC_RAW_DIR = HIC_DATA_ROOT / "data" / "hic" / "raw_hic"          # config.hic_data_dir
def HIC_RAW(accession: str) -> Path:
    return HIC_RAW_DIR / f"{accession}.hic"

HIC2TRACK_DIR = HIC_DATA_ROOT / "data" / "hic2track"             # config.hic2tarck_dir
def HIC2TRACK(cell: str) -> Path:                                # BigWig + chrom-size + index
    return HIC2TRACK_DIR / cell

# Evo 2 raw chr-level embeddings (config.evo2_embedding_map layout)
EVO2_EMB_HUMAN = Path(os.environ.get(
    "EVO2HIC_EVO2_EMB_DIR",
    HIC_DATA_ROOT / "data/dna/human/hg38_2000_evo2_7b",
))
EVO2_EMB_MOUSE = HIC_DATA_ROOT / "data/dna/mouse/mm10_evo2_7b"
EVO2_EMB_DIR   = EVO2_EMB_HUMAN                                   # alias (used by Supp. Fig. 3)

MOTIF_MEME       = DATA_DIR / "motifs.meme"                 # JASPAR motifs (Fig 5, ED Fig 7, ED Table 2)
HUMAN_MERGED_CSV = DATA_DIR / "human_genes_merged.csv"      # genes + ENCODE RNA-seq TPM, Supp Table 2 (Fig 5, ED Table 2)
MOTIF_ANNOTATIONS_XLSX = DATA_DIR / "motif_annotations.xlsx"  # motif archetype clusters (Fig 5)
# chr9 motif matches (FIMO on hg38 chr9 with MOTIF_MEME; 7 GB, not shipped) used by main/fig5/motif_analysis.ipynb
MOTIF_MATCH_CHR9_BED = HIC_DATA_ROOT / "data/dna/human/chr9_motif_match.bed"

# Phylogenetic tree + clade map (Fig 6, ED Figs 6-7, Supp Fig 11), shipped in data/
TREE_NWK  = DATA_DIR / "species_list_rawchrom.nwk"
CLADE_DIR = DATA_DIR / "clade"         # clade_<Clade>_clean.txt

# ORCA reference outputs (published Orca artifacts; Fig 3 / ED Fig 2 / Supp Fig 4)
ORCA_FIGURE_DIR    = HIC_DATA_ROOT / "orca_figure"
ORCA_MALLPREDS_PTH = ORCA_FIGURE_DIR / "mallpreds.pth"
ORCA_EXPECTED_NPY  = HIC_DATA_ROOT / "orca/resources/4DNFI9GMP2J8.rebinned.mcool.expected.res4000.npy"

# GM12878 epigenomic tracks on chr10 at 2 kb, shape (5, 66899) (Fig 2 / Supp Figs 2-3; only ranks are used).
# Not shipped: regenerate from the public ENCODE bigWigs with the Evo2HiC Track_Loader (README, "Data").
# Written by helper/make_tracks_gm12878_chr10.py into data/; override with EVO2HIC_TRACKS_GM12878_CHR10.
TRACKS_GM12878_CHR10_NPY = Path(os.environ.get(
    "EVO2HIC_TRACKS_GM12878_CHR10",
    DATA_DIR / "tracks_GM12878_10.npy",
))

# ---- Shared model checkpoints ------------------------------------------------
# Epigenomic-prediction model (DNA[+Hi-C] -> 5 tracks); Fig 4 / Fig 5 / ED Figs 3-5 / Supp Figs 5-9, 15.
# Released: epi_prediction/model.pt; `.../model/track/<cell>/<chr>.npy` are its predictions.
EPI_EVO2HIC_DIR = CKPT_ROOT / "epi_prediction" / "model" / "track"

# Hi-C super-resolution model (released: resolution_enhancement/model.pt); Fig 6 / ED Fig 6 / Supp Figs 10-12.
SR_EVO2HIC_CKPT      = CKPT_ROOT / "resolution_enhancement" / "model.pt"
SR_EVO2HIC_DIR       = CKPT_ROOT / "resolution_enhancement" / "model" / "multi_10000"  # DNA Zoo enhanced .hic
SR_EVO2HIC_HUMAN_DIR = CKPT_ROOT / "resolution_enhancement" / "model" / "human"        # per-cell enhanced_test.hic
SR_HICARN2_DIR       = CKPT_ROOT / "hic_sr_hicarn2" / "multi_10000"   # baseline, not released

# Pretrained encoder (released: pretrained_weights/model.pt); Fig 2 / ED Fig 1 / Supp Fig 2.
# The w/o-Evo 2 variant (args.json:evo2_option = No) is an ablation, not released.
PRETRAIN_CKPT_DIR       = CKPT_ROOT / "pretrained_weights"    # inference outputs: {human,mouse}{_embeds,_mask}/…
PRETRAIN_CKPT           = PRETRAIN_CKPT_DIR / "model.pt"
PRETRAIN_NOEVO_CKPT_DIR = CKPT_ROOT / "pretrained_weights_noevo2"  # ablation, not released
PRETRAIN_NOEVO_CKPT     = PRETRAIN_NOEVO_CKPT_DIR / "model.pt"

# Seq2HiC (DNA -> Hi-C) checkpoints; Fig 3 / ED Fig 2 / Supp Fig 4. Run live in-notebook.
def SEQ2HIC_CKPT(cell: str, variant: str = "evo2hic") -> Path:
    """variant: 'evo2hic' (released: seq2hic_<cell>/model.pt) | 'hic_distilled' | 'evo2'
    (ablations, not released)."""
    if variant == "evo2hic":
        return CKPT_ROOT / f"seq2hic_{cell}" / "model.pt"
    return CKPT_ROOT / "seq2hic_ablations" / cell / variant / "model.pt"

# ---- sys.path bootstrap (scripts importing dataset.* / config) ------------
def add_repo_to_syspath() -> None:
    """Insert REPO at the front of sys.path so `import dataset.* / config`
    resolves to the in-repo modules. Idempotent."""
    import sys
    rp = str(REPO)
    if rp not in sys.path:
        sys.path.insert(0, rp)

def ensure_out_dir() -> Path:
    """Create OUT_DIR if missing and return it."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUT_DIR


# ===========================================================================
# Derived inference outputs (regeneration notes)
# ===========================================================================
# Some notebooks read OUTPUTS of the inference scripts rather than primary
# data. They reference those outputs at their on-disk location directly; to
# regenerate a missing one, run the matching command (cd $EVO2HIC_REPO first;
# PY=$(conda run -n Evo2HiC which python)):
#   * epi prediction npy  (Fig 4, ED Figs 3-5)  : shipped under EPI_EVO2HIC_DIR;
#       PY -m inference.inference_CDNA1d -ckpt <ckpt> --save-dir track --species human
#   * Seq2HiC per-locus   (Fig 3, ED Fig 2)   : run live in-notebook (no cache).
#   * HiC-SR enhanced .hic (Fig 6, Supp 10-12) : under SR_* above; inference_CDNA2d.py.


# ===========================================================================
# main/fig1 — Fig. 1, overview (multispecies + Hi-C×epigenomics)
# ===========================================================================
# Shared inputs used: HIC_RAW, HIC2TRACK, DNA_FASTA, EVO2_EMB_*, EPI_EVO2HIC_DIR,
#   PRETRAIN_CKPT, SR_EVO2HIC_HUMAN_DIR, ORCA_*, TREE_NWK, CLADE_DIR.
# Result table (per-species SPC, 177-species multi summary):
SPC_FIG1 = RESULT("main/fig1") / "fig1" / "multi" / "SPC.csv"      # producer: evaluate/general_eval.py (multi)


# ===========================================================================
# main/fig2 — Fig. 2, retrieval
# ===========================================================================
# Shared inputs used: PRETRAIN_CKPT_DIR / PRETRAIN_NOEVO_CKPT_DIR (recall.tsv,
#   dna_emb_*.npy, rank npys under {human,mouse}_{embeds,mask}/).
RESULT_FIG2 = RESULT("main/fig2")         # self-written rank_{human,mouse}.csv (stat dump)


# ===========================================================================
# main/fig3, main/fig4 — Fig. 3 Seq2HiC (run live) & Fig. 4 epigenomic prediction
# ===========================================================================
# Fig 3 shared inputs: SEQ2HIC_CKPT, HIC_RAW, ORCA_MALLPREDS_PTH. No result table
#   (predictions computed live in-notebook).
# Fig 4 shared inputs: EPI_EVO2HIC_DIR/<cell>/<chr>.npy, HIC2TRACK. No result table.


# ===========================================================================
# main/fig5 — Fig. 5, motif interpretation
# ===========================================================================
# Shared inputs used: MOTIF_MEME, HUMAN_MERGED_CSV, HUMAN_INTER_DIR (attribution).
MOTIF_STATS_CSV = RESULT("main/fig5") / "motif_enrichment_stats_H3K27ac.csv"  # main/fig5/motif_analysis.ipynb
RESULT_FIG5     = RESULT("main/fig5")     # self-written Fig5f_enrichment_*.tsv (stat dump)

# Per-cell attribution arrays (produced by inference_CDNA1d.py --interpretation
# into <ckpt>/human_inter/<cell>/{seq_<chr>.npy, <track>/attr_<chr>_filled.npy});
# not bundled in the released epi_prediction checkpoint — set EVO2HIC_HUMAN_INTER.
HUMAN_INTER_DIR = Path(os.environ.get(
    "EVO2HIC_HUMAN_INTER",
    EPI_EVO2HIC_DIR.parent / "human_inter",   # epi_prediction/model/human_inter
))


# ===========================================================================
# main/fig6 — Fig. 6, Hi-C super-resolution benchmark
# ===========================================================================
# Shared inputs used: SR_EVO2HIC_DIR, SR_HICARN2_DIR, TREE_NWK, CLADE_DIR.
# SR metric matrices (rows = test Hi-C maps, cols = methods incl. HiCARN2).
# producer: evaluate/general_eval.py per method, collated per column (see helper/README.md).
RESULT_FIG6_HUMAN = RESULT("main/fig6") / "human"    # {PCC,SPC,PSNR,SSIM}.csv
RESULT_FIG6_MOUSE = RESULT("main/fig6") / "mouse"
SPC_FIG6          = RESULT("main/fig6") / "multi" / "SPC.csv"


# ===========================================================================
# extended_data/ed_fig1, supplementary/supp_fig2 — retrieval gap, UMAP (companion to Fig. 2)
# ===========================================================================
# Shared inputs used: PRETRAIN_CKPT_DIR (per-cell recall / similarity).
RESULT_SUPP1 = RESULT("extended_data/ed_fig1")# self-written gap_{human,mouse}.csv (stat dump)


# ===========================================================================
# supplementary/supp_fig3 — Evo 2 embedding UMAP
# ===========================================================================
# Shared inputs used: EVO2_EMB_DIR, TRACKS_GM12878_CHR10_NPY.


# ===========================================================================
# supp_fig5/6, ed_fig4/5 — epigenomic-prediction baselines & ablations
# ===========================================================================
# Shared inputs used: EPI_EVO2HIC_DIR + the baseline checkpoints below (each
# holds `track/<cell>/<chr>.npy` predictions + a `result.tsv` PCC table,
# produced by inference_CDNA1d.py --save-dir track). These baselines / ablations are not
# released; their chr9/10 Hi-C-only and Evo 2 predictions are shipped in data/epi_baselines/.
EPI_BASELINE_DIR    = DATA_DIR / "epi_baselines"                     # shipped, chr9/10: <name>/<cell>/<chr>.npy + result.tsv
EPI_NODISTILL_DIR   = EPI_BASELINE_DIR / "nodistill"      # ED Fig 4 (no distillation)    [epi_baseline_Evo2HiC_nodistill/40000]
EPI_EVO2EMB_HIC_DIR = EPI_BASELINE_DIR / "evo2emb_hic"    # ED Fig 4 (Evo2 emb + Hi-C)    [epi_baseline_Evo2emb+hic/48000]
EPI_RIDGE_DIR       = EPI_BASELINE_DIR / "ridge"          # ED Fig 4 (ridge on Hi-C)      [epi_baseline_ridge_HiC]
EPI_HIC_ONLY_DIR    = EPI_BASELINE_DIR / "hic_only"       # Supp Fig 6 (Hi-C only)        [epi_baseline_hic_only/60000]

# Per-cell DNA-only Evo 2 epigenomic baseline (ED Figs 3-5, Supp Figs 5-6)  [epi_baseline_evo2_<cell>, steps 44000 / 80000 / 70000]
def EPI_EVO2_DIR(cell: str) -> Path:
    return EPI_BASELINE_DIR / "evo2" / cell

# ---- extended_data/ed_fig3 — external DNA->epigenome baselines (Borzoi, AlphaGenome) -----
# These predictions are not shipped. Regenerate them with the scripts in
# extended_data/ed_fig3/ (written to the two folders below). Each holds
# `<cell>/{9,10}.npy` (shape 5×n_bins) + `result.tsv` (per-cell/chr/track PCC).
#
#   Borzoi      : public pretrained Borzoi via the `borzoi-pytorch` HF port
#                 (github.com/johahi/borzoi-pytorch; model calico/borzoi).
#                 Zero-shot; we pick the 15 (cell×assay) heads relevant to us.
#                 Reproduce: python extended_data/ed_fig3/borzoi_inference.py; python extended_data/ed_fig3/borzoi_eval.py
#   AlphaGenome : Google DeepMind AlphaGenome (fold-1 released weights) via the
#                 official alphagenome API. Reproduce: python extended_data/ed_fig3/alphagenome_inference.py;
#                 python extended_data/ed_fig3/alphagenome_eval.py
BORZOI_DIR = Path(os.environ.get("EVO2HIC_BORZOI_DIR",     RESULT("extended_data/ed_fig3") / "borzoi"))
AG_DIR     = Path(os.environ.get("EVO2HIC_ALPHAGENOME_DIR", RESULT("extended_data/ed_fig3") / "alphagenome"))


# ===========================================================================
# extended_data/ed_fig5 — cross-cell evaluation
# ===========================================================================
# Shared inputs used: EPI_EVO2HIC_DIR, HIC2TRACK.
# producer: extended_data/ed_fig5/crosscell_eval.py  (-> crosscell_matrix / spec_gap / spec_pcc /
#         williams_pvals .tsv).
RESULT_CROSSCELL_DIR = RESULT("extended_data/ed_fig5")


# ===========================================================================
# supplementary/supp_fig7 — DNase ABC / CRISPRi gene-enhancer application
# ===========================================================================
# supplementary/supp_fig7/plot.py is SELF-CONTAINED: it computes the two TSVs below from raw
# inputs on first run and caches them here (delete a cache file to recompute).
# Shared inputs used: EPI_EVO2HIC_DIR (predicted DNase/H3K27ac), HIC2TRACK,
#   HIC_RAW (KR-normalized contact).
def ABC_PRED_TRACK(cell: str, chrom: int) -> Path:      # predicted 5-track npy
    return EPI_EVO2HIC_DIR / cell / f"{chrom}.npy"
EPI_PRED_ARGS_JSON = EPI_EVO2HIC_DIR.parents[1] / "args.json"       # epi_prediction/args.json
ABC_CELL_HIC = {"GM12878": "4DNFI1UEG1HD", "H1ESC": "4DNFIQYQWPF5", "K562": "4DNFITUOMFUQ"}
# CRISPRi-FlowFISH benchmark (ground truth): ENCODE-rE2G ensemble CRISPR
# benchmark (EPCrisprBenchmark, GRCh38). Shipped in-repo.
ABC_CRISPR_BENCH = REPO / "data" / "crispr" / "bench_5ct.tsv.gz"
# Cache (self-written):
ABC_CACHE_DIR       = RESULT("supplementary/supp_fig7")
ABC_CRISPR_PPIF_TSV = ABC_CACHE_DIR / "ppif_crispr_scores.tsv"
ABC_ALIGN_TSV       = ABC_CACHE_DIR / "abc_pred_vs_meas_chr910.tsv"


# ===========================================================================
# supplementary/supp_fig8, supp_fig9 — Hi-C loop functional decomposition
# ===========================================================================
# Shared inputs used: EPI_EVO2HIC_DIR (predicted tracks), HIC2TRACK (real
#   ChIP-seq), HIC_RAW. Producers: supplementary/supp_fig8/label_loops.py, then
#   extended_data/ed_table2/loop/rule_on_predicted.py (after HiCCUPS on chr9+chr10).
LOOP_APP_DIR          = RESULT("supplementary/supp_fig8")
LOOP_LABELS_TSV       = LOOP_APP_DIR / "loop_labels.tsv"               # step 1
LOOP_RULE_NPZ         = LOOP_APP_DIR / "rule_on_predicted_labels.npz"  # step 2 (Supp Fig 8)
LOOP_RULE_METRICS_TSV = LOOP_APP_DIR / "rule_on_predicted_metrics.tsv" # step 2
HIC_LOOP_DIR = HIC_DATA_ROOT / "data" / "hic" / "hiccups_loop"         # config.hic_loop_dir
def HIC_LOOP_BEDPE(accession: str) -> Path:
    return HIC_LOOP_DIR / accession / "merged_loops.bedpe"
def LOOP_PRED_TRACK(cell: str, chrom: int) -> Path:                   # = ABC_PRED_TRACK (Supp Fig 9)
    return EPI_EVO2HIC_DIR / cell / f"{chrom}.npy"


# ===========================================================================
# supp_fig10, ed_fig6, supp_fig11 — SR benchmark companions
# ===========================================================================
# Same SR metric matrices as Fig 6, duplicated per figure (helper writes to all).
# Supp Fig 10 (mouse box), Supp Fig 11 (multi PCC): SR heatmaps + boxes.
RESULT_SUPP16_MOUSE = RESULT("supplementary/supp_fig10") / "mouse"     # {PCC,SPC,PSNR,SSIM}.csv
RESULT_SUPP18_MULTI = RESULT("supplementary/supp_fig11") / "multi"     # PCC.csv
# ED Fig 6 (polar tree of per-species ΔSPC): human+mouse metrics + multi SPC.
# Shared inputs used: TREE_NWK, CLADE_DIR.
RESULT_SUPP17_HUMAN = RESULT("extended_data/ed_fig6") / "human"
RESULT_SUPP17_MOUSE = RESULT("extended_data/ed_fig6") / "mouse"
SPC_SUPP17          = RESULT("extended_data/ed_fig6") / "multi" / "SPC.csv"


# ===========================================================================
# supplementary/supp_fig12 — DNA Zoo super-resolution heatmaps
# ===========================================================================
# Shared inputs used: SR_EVO2HIC_DIR, SR_HICARN2_DIR.
DNAZOO_RAW_HIC_DIR = HIC_DATA_ROOT / "dnazoo" / "raw_hic_fixed"
def DNAZOO_RAW_HIC(species: str) -> Path:
    return DNAZOO_RAW_HIC_DIR / f"{species}.hic"


# ===========================================================================
# supplementary/supp_fig13 — timing benchmark
# ===========================================================================
TIMING_TSV = DATA_DIR / "timing_results.tsv"              # measured inference time / memory (Supp Fig 13)


# ===========================================================================
# supplementary/supp_fig14 — pretraining over-representation (epi-prediction PCC by cell)
# ===========================================================================
# Shared inputs used: EPI_EVO2HIC_DIR, EPI_NODISTILL_DIR (panel b, same tables
#   as Fig 4 / ED Fig 4). Panel c reads the SR per-map PCC (human/mouse) below.
RESULT_SUPP22_HUMAN = RESULT("supplementary/supp_fig14") / "human"     # PCC.csv
RESULT_SUPP22_MOUSE = RESULT("supplementary/supp_fig14") / "mouse"


# ===========================================================================
# main/fig1/plot_dnazoo.ipynb — circular multispecies improvement
# ===========================================================================
# Shared inputs used: TREE_NWK, CLADE_DIR.
SPC_DNAZOO = RESULT("main/fig1") / "dnazoo" / "multi" / "SPC.csv"


# ===========================================================================
# Analysis pipelines (Table 1, ED Fig 7, ED Tables 1-2, Supp Table 13)
# ===========================================================================
# Outputs (tables, tensors, logs, intermediate checkpoints) are written to RA_WORK(<analysis>).
# Shell scripts read the same environment variables (paths.sh).
# Analysis folders (by manuscript item); RA_WORK(<analysis>) is where an analysis writes its outputs:
# the item folder itself, or <EVO2HIC_RA_WORK>/<item> when that variable is set.
RA_ITEMS = {
    "orca":          "main/table1",
    "species":       "extended_data/ed_fig7",
    "hicfoundation": "extended_data/ed_table1",
    "abc":           "extended_data/ed_table2/abc",
    "loop":          "extended_data/ed_table2/loop",
    "supp_table13":  "supplementary/supp_table13",
}
RA_WORK_ROOT = Path(os.environ.get("EVO2HIC_RA_WORK", PLOT_DIR))
def RA_WORK(analysis: str) -> Path:
    """Output folder of one analysis (keys of RA_ITEMS)."""
    return RA_WORK_ROOT / RA_ITEMS[analysis]

# Result tables shipped with the code (<item>/result/): the rows / columns of baselines whose
# predictions and checkpoints are not released. A table regenerated in RA_WORK takes precedence.
def RA_SHIPPED(analysis: str, name: str) -> Path:
    return PLOT_DIR / RA_ITEMS[analysis] / "result" / name
def RA_TABLE(analysis: str, name: str) -> Path:
    p = RA_WORK(analysis) / name
    return p if p.exists() else RA_SHIPPED(analysis, name)

# Small input data shipped in data/ (tree and clade lists: TREE_NWK, CLADE_DIR above)
RA_CRISPR_BENCH = DATA_DIR / "crispr" / "bench_gasperini_morris_chr910.tsv.gz"  # K562 CRISPRi, chr9/10
RA_TSS_CHR910 = DATA_DIR / "gencode_v38_tss_chr910.tsv"                         # GENCODE v38 TSSs, chr9/10

# ---- External code (cloned separately, see main/table1 and extended_data/ed_table1 READMEs) ----
HICFOUNDATION_DIR = Path(os.environ.get("HICFOUNDATION_DIR", REPO.parent / "HiCFoundation"))
HICFOUNDATION_PY = os.environ.get("HICFOUNDATION_PY", "python")    # interpreter of the HiCFoundation env
CORIGAMI_DIR = Path(os.environ.get("CORIGAMI_DIR", REPO.parent / "C.Origami"))

# ---- Data -----------------------------------------------------------------
ORCA_DIR           = HIC_DATA_ROOT / "orca"                         # Orca code checkout + resources/
ORCA_RESOURCES_DIR = ORCA_DIR / "resources"                         # Micro-C mcools + expected .npy
HIC_DOWNSAMPLE_DIR       = HIC_DATA_ROOT / "data" / "hic" / "downsample"        # <acc>.16.seed0.hic
HIC_DOWNSAMPLE_MULTI_DIR = HIC_DATA_ROOT / "data" / "hic" / "downsample_multi"  # <species>.16.seed0.hic

# ---- Released Evo2HiC-cg checkpoints (Zenodo; the other released models are defined above:
#      PRETRAIN_CKPT, SR_EVO2HIC_CKPT, SEQ2HIC_CKPT(cell), EPI_EVO2HIC_DIR) ----
def SEQ2HIC_CG_CKPT(cell: str) -> Path:                                     # Evo2HiC-cg (Orca's setting)
    return CKPT_ROOT / f"seq2hic_cg_{cell}" / "model.pt"

# ---- Hi-C-only super-resolution ablation (checkpoint and outputs not released; its per-species results are
#      shipped in extended_data/{ed_fig7,ed_table1}/result/, training configuration in
#      baselines/hic_sr_hic_only/args.json). Used only if present.
SR_HIC_ONLY_DIR = CKPT_ROOT / "hic_sr_hic_only" / "multi_10000"


# ===========================================================================
# Shell export (`python paths.py --sh`, sourced by paths.sh)
# ===========================================================================
if __name__ == "__main__":
    import shlex
    import sys
    if "--sh" in sys.argv:
        for _k, _v in {"REPO": REPO, "PLOT_DIR": PLOT_DIR, "CKPT_ROOT": CKPT_ROOT,
                       "HIC_DATA_ROOT": HIC_DATA_ROOT, "RA_WORK_ROOT": RA_WORK_ROOT,
                       "HICFOUNDATION_DIR": HICFOUNDATION_DIR, "HICFOUNDATION_PY": HICFOUNDATION_PY}.items():
            print(f"{_k}={shlex.quote(str(_v))}")
