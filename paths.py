"""Central path configuration for plot_revision/ reproduction scripts.

Every figure script imports its paths from here, so a reader edits **one**
file to point the reproduction at their local data / checkpoints.

How to use this file (reader checklist)
---------------------------------------
1. Put `plot_revision/` under the Evo2HiC repo (so REPO auto-resolves), or set
   EVO2HIC_REPO.
2. Point the three data roots below at your local copies — either export the
   env vars, or edit the placeholder defaults in place:

     EVO2HIC_CKPT_ROOT  -> released model checkpoints root   (CKPT_ROOT)
     EVO2HIC_HIC_DATA   -> raw Hi-C / DNA / embeddings root  (HIC_DATA_ROOT)
     EVO2HIC_OUT_DIR    -> where PDFs/stats are written      (default <REPO>/Figures)

   Checkpoint sub-folder names below are **clean placeholder names** for the
   released artifacts (e.g. `hic_sr_evo2hic`), NOT the internal training-run
   names. Download the released checkpoint into the matching folder name, or
   symlink your local run to it.

Layout conventions
------------------
* result tables live under `RESULT_DIR/<figure>/…` (one folder per figure:
  fig1, fig2, …, supp12, supp13, …). A table consumed by several figures is
  written into each of their folders by the producing helper — so every
  figure folder is self-contained. Producers live in `plot_revision/helper/`.
* This file is ordered by figure (Fig 1 → Fig 6, then Supp). Paths shared by
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

RESULT_DIR = REPO / "result"            # per-figure result tables (shipped in-repo)

# Released model checkpoints. Sub-folders below use clean placeholder names;
# download / symlink the released artifact into each name.
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
# Shared inputs (used by many figures — defined once)
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
EVO2_EMB_DIR   = EVO2_EMB_HUMAN                                   # alias (used by Supp 4)

MOTIF_MEME       = HIC_DATA_ROOT / "data/dna/motifs.meme"         # config.motif_data (Fig 5)
HUMAN_MERGED_CSV = HIC_DATA_ROOT / "data/dna/human/merged.csv"    # gene annotation (Fig 5)

# Phylogenetic tree + clade map (Fig 6 / Supp 17 / Supp 19)
TREE_NWK         = REPO / "data/species_list_rawchrom.nwk"
CLAUDE_CLADE_DIR = REPO / "data/claude"        # claude_<Clade>_clean.txt

# ORCA reference outputs (published Orca artifacts; Fig 3 / Supp 5-7)
ORCA_FIGURE_DIR    = HIC_DATA_ROOT / "orca_figure"
ORCA_MALLPREDS_PTH = ORCA_FIGURE_DIR / "mallpreds.pth"
ORCA_EXPECTED_NPY  = HIC_DATA_ROOT / "orca/resources/4DNFI9GMP2J8.rebinned.mcool.expected.res4000.npy"

# Per-bin GM12878 epigenomic tracks (Fig 2 / Supp 4). First existing of:
# env override -> in-repo misc/.
TRACKS_GM12878_CHR10_NPY = Path(os.environ.get(
    "EVO2HIC_TRACKS_GM12878_CHR10",
    REPO / "misc" / "tracks_GM12878_10.npy",
))

# ---- Shared model checkpoints (clean placeholder names) -------------------
# Epigenomic-prediction model (DNA[+Hi-C] -> 5 tracks); Fig 4 / Fig 5 /
# Supp 8-15. `.../track/<cell>/<chr>.npy` are the shipped per-cell predictions.
EPI_EVO2HIC_DIR = CKPT_ROOT / "epi_prediction" / "model" / "track"

# Hi-C super-resolution model + HiCARN2 baseline; Fig 6 / Supp 16-20.
SR_EVO2HIC_DIR       = CKPT_ROOT / "hic_sr_evo2hic" / "multi_10000"   # multispecies enhanced .hic
SR_HICARN2_DIR       = CKPT_ROOT / "hic_sr_hicarn2" / "multi_10000"
SR_EVO2HIC_HUMAN_DIR = CKPT_ROOT / "hic_sr_evo2hic" / "human"         # per-cell enhanced_test.hic

# Retrieval (SigLIP-pretrained) encoders; Fig 2 / Supp 1 / Supp 3.
# Two variants differ only by args.json:evo2_option (Yes -> Evo2HiC, No -> w/o Evo 2).
PRETRAIN_CKPT_DIR       = CKPT_ROOT / "retrieval_evo2hic"     # inference outputs: {human,mouse}{_embeds,_mask}/…
PRETRAIN_CKPT           = PRETRAIN_CKPT_DIR / "model.pt"
PRETRAIN_NOEVO_CKPT_DIR = CKPT_ROOT / "retrieval_noevo2"      # alignment baseline (evo2_option=No)
PRETRAIN_NOEVO_CKPT     = PRETRAIN_NOEVO_CKPT_DIR / "model.pt"

# Seq2HiC (DNA -> Hi-C) checkpoints; Fig 3 / Supp 5-7. Run live in-notebook.
def SEQ2HIC_CKPT(cell: str, variant: str) -> Path:
    """variant: 'evo2hic' | 'hic_distilled' | 'evo2'"""
    return CKPT_ROOT / "seq2hic" / cell / variant / "model.pt"

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
#   * epi prediction npy  (Fig 4, Supp 8-12) : shipped under EPI_EVO2HIC_DIR;
#       PY -m inference.inference_CDNA1d -ckpt <ckpt> --save-dir track --species human
#   * Seq2HiC per-locus   (Fig 3, Supp 5-7)  : run live in-notebook (no cache).
#   * HiC-SR enhanced .hic (Fig 6, Supp 16-20): under SR_* above; inference_CDNA2d.py.


# ===========================================================================
# Figure 1 — overview (multispecies + Hi-C×epigenomics)
# ===========================================================================
# Shared inputs used: HIC_RAW, HIC2TRACK, DNA_FASTA, EVO2_EMB_*, EPI_EVO2HIC_DIR,
#   PRETRAIN_CKPT, SR_EVO2HIC_HUMAN_DIR, ORCA_*, TREE_NWK, CLAUDE_CLADE_DIR.
# Result table (per-species SPC, 177-species multi summary):
SPC_FIG1 = RESULT_DIR / "fig1" / "multi" / "SPC.csv"      # helper: helper/general_eval.py (multi)


# ===========================================================================
# Figure 2 — retrieval
# ===========================================================================
# Shared inputs used: PRETRAIN_CKPT_DIR / PRETRAIN_NOEVO_CKPT_DIR (recall.tsv,
#   dna_emb_*.npy, rank npys under {human,mouse}_{embeds,mask}/).
RESULT_FIG2 = RESULT_DIR / "fig2"        # self-written rank_{human,mouse}.csv (stat dump)


# ===========================================================================
# Figure 3 — Seq2HiC (run live)  &  Figure 4 — epigenomic prediction
# ===========================================================================
# Fig 3 shared inputs: SEQ2HIC_CKPT, HIC_RAW, ORCA_MALLPREDS_PTH. No result table
#   (predictions computed live in-notebook).
# Fig 4 shared inputs: EPI_EVO2HIC_DIR/<cell>/<chr>.npy, HIC2TRACK. No result table.


# ===========================================================================
# Figure 5 — motif interpretation
# ===========================================================================
# Shared inputs used: MOTIF_MEME, HUMAN_MERGED_CSV, HUMAN_INTER_DIR (attribution).
MOTIF_STATS_CSV = RESULT_DIR / "fig5" / "motif_enrichment_stats_H3K27ac.csv"  # helper/motif_analysis.ipynb
RESULT_FIG5     = RESULT_DIR / "fig5"    # self-written Fig5f_enrichment_*.tsv (stat dump)

# Per-cell attribution arrays (produced by inference_CDNA1d.py --interpretation
# into <ckpt>/human_inter/<cell>/{seq_<chr>.npy, <track>/attr_<chr>_filled.npy});
# not bundled in the released epi_prediction checkpoint — set EVO2HIC_HUMAN_INTER.
HUMAN_INTER_DIR = Path(os.environ.get(
    "EVO2HIC_HUMAN_INTER",
    EPI_EVO2HIC_DIR.parent / "human_inter",   # epi_prediction/model/human_inter
))


# ===========================================================================
# Figure 6 — Hi-C super-resolution benchmark
# ===========================================================================
# Shared inputs used: SR_EVO2HIC_DIR, SR_HICARN2_DIR, TREE_NWK, CLAUDE_CLADE_DIR.
# SR metric matrices (rows = test Hi-C maps, cols = methods incl. HiCARN2).
# helper: helper/general_eval.py per method, collated per column (see helper/README.md).
RESULT_FIG6_HUMAN = RESULT_DIR / "fig6" / "human"    # {PCC,SPC,PSNR,SSIM}.csv
RESULT_FIG6_MOUSE = RESULT_DIR / "fig6" / "mouse"
SPC_FIG6          = RESULT_DIR / "fig6" / "multi" / "SPC.csv"


# ===========================================================================
# Supp 1 / Supp 3 — retrieval gap (companion to Fig 2)
# ===========================================================================
# Shared inputs used: PRETRAIN_CKPT_DIR (per-cell recall / similarity).
RESULT_SUPP1 = RESULT_DIR / "supp1"      # self-written gap_{human,mouse}.csv (stat dump)


# ===========================================================================
# Supp 4 — GM12878 track sanity (Evo 2 embedding scatter)
# ===========================================================================
# Shared inputs used: EVO2_EMB_DIR, TRACKS_GM12878_CHR10_NPY.


# ===========================================================================
# Supp 8-11 — epigenomic-prediction baselines & ablations
# ===========================================================================
# Shared inputs used: EPI_EVO2HIC_DIR + the baseline checkpoints below (each
# holds `track/<cell>/<chr>.npy` predictions + a `result.tsv` PCC table,
# produced by inference_CDNA1d.py --save-dir track).
EPI_NODISTILL_DIR   = CKPT_ROOT / "epi_baseline_nodistill"   / "track"   # Supp 11 (no distillation)
EPI_EVO2EMB_HIC_DIR = CKPT_ROOT / "epi_baseline_evo2emb_hic" / "track"   # Supp 11 (Evo2 emb + Hi-C)
EPI_RIDGE_DIR       = CKPT_ROOT / "epi_baseline_ridge"       / "track"   # Supp 11 (ridge on Hi-C)
EPI_HIC_ONLY_DIR    = CKPT_ROOT / "epi_baseline_hic_only"    / "track"   # Supp 9  (Hi-C only)

# Per-cell DNA-only Evo 2 epigenomic baseline (Supp 9-11)
def EPI_EVO2_DIR(cell: str) -> Path:
    return CKPT_ROOT / f"epi_baseline_evo2_{cell}" / "track" / cell

# ---- Supp 10 — external DNA->epigenome baselines (Borzoi, AlphaGenome) -----
# These predictions are NOT shipped in the main repo. Regenerate them with the
# helper scripts copied into plot_revision/helper/ (see helper/README.md), then
# point the two dirs below at the produced `results/<model>/` folder. Each holds
# `<cell>/{9,10}.npy` (shape 5×n_bins) + `result.tsv` (per-cell/chr/track PCC).
#
#   Borzoi      : public pretrained Borzoi via the `borzoi-pytorch` HF port
#                 (github.com/johahi/borzoi-pytorch; model calico/borzoi).
#                 Zero-shot; we pick the 15 (cell×assay) heads relevant to us.
#                 Reproduce: PY -m helper.borzoi_inference ; PY -m helper.borzoi_eval
#   AlphaGenome : Google DeepMind AlphaGenome (fold-1 released weights) via the
#                 official alphagenome API. Reproduce: PY -m helper.alphagenome_inference
#                 ; PY -m helper.alphagenome_eval
# Not shipped — download/regenerate into these folders (see helper/README.md).
BORZOI_DIR = Path(os.environ.get("EVO2HIC_BORZOI_DIR",     RESULT_DIR / "supp10" / "borzoi"))
AG_DIR     = Path(os.environ.get("EVO2HIC_ALPHAGENOME_DIR", RESULT_DIR / "supp10" / "alphagenome"))


# ===========================================================================
# Supp 12 — cross-cell evaluation
# ===========================================================================
# Shared inputs used: EPI_EVO2HIC_DIR, HIC2TRACK.
# helper: helper/crosscell_eval.py  (-> crosscell_matrix / spec_gap / spec_pcc /
#         williams_pvals .tsv).
RESULT_CROSSCELL_DIR = RESULT_DIR / "supp12"


# ===========================================================================
# Supp 13 — DNase ABC / CRISPRi gene-enhancer application
# ===========================================================================
# plot_supp13.py is SELF-CONTAINED: it computes the two TSVs below from raw
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
ABC_CACHE_DIR       = RESULT_DIR / "supp13"
ABC_CRISPR_PPIF_TSV = ABC_CACHE_DIR / "ppif_crispr_scores.tsv"
ABC_ALIGN_TSV       = ABC_CACHE_DIR / "abc_pred_vs_meas_chr910.tsv"


# ===========================================================================
# Supp 14 / Supp 15 — Hi-C loop functional decomposition
# ===========================================================================
# Shared inputs used: EPI_EVO2HIC_DIR (predicted tracks), HIC2TRACK (real
#   ChIP-seq), HIC_RAW. The two producer scripts live under
#   revision/loop_decomposition/ (run label_loops.py then rule_on_predicted.py,
#   after HiCCUPS on chr9+chr10); see helper/README.md.
LOOP_APP_DIR          = REPO / "revision" / "loop_decomposition"
LOOP_LABELS_TSV       = LOOP_APP_DIR / "loop_labels.tsv"               # step 1
LOOP_RULE_NPZ         = LOOP_APP_DIR / "rule_on_predicted_labels.npz"  # step 2 (Supp 14)
LOOP_RULE_METRICS_TSV = LOOP_APP_DIR / "rule_on_predicted_metrics.tsv" # step 2
HIC_LOOP_DIR = HIC_DATA_ROOT / "data" / "hic" / "hiccups_loop"         # config.hic_loop_dir
def HIC_LOOP_BEDPE(accession: str) -> Path:
    return HIC_LOOP_DIR / accession / "merged_loops.bedpe"
def LOOP_PRED_TRACK(cell: str, chrom: int) -> Path:                   # = ABC_PRED_TRACK (Supp 15)
    return EPI_EVO2HIC_DIR / cell / f"{chrom}.npy"


# ===========================================================================
# Supp 16 / Supp 17 / Supp 18 — SR benchmark companions
# ===========================================================================
# Same SR metric matrices as Fig 6, duplicated per figure (helper writes to all).
# Supp 16 (mouse box), Supp 18 (multi PCC): SR heatmaps + boxes.
RESULT_SUPP16_MOUSE = RESULT_DIR / "supp16" / "mouse"     # {PCC,SPC,PSNR,SSIM}.csv
RESULT_SUPP18_MULTI = RESULT_DIR / "supp18" / "multi"     # PCC.csv
# Supp 17 (polar tree of per-species ΔSPC): human+mouse metrics + multi SPC.
# Shared inputs used: TREE_NWK, CLAUDE_CLADE_DIR.
RESULT_SUPP17_HUMAN = RESULT_DIR / "supp17" / "human"
RESULT_SUPP17_MOUSE = RESULT_DIR / "supp17" / "mouse"
SPC_SUPP17          = RESULT_DIR / "supp17" / "multi" / "SPC.csv"


# ===========================================================================
# Supp 19 — multispecies TAD boundary F1 (Evo2HiC vs HICARN2)
# ===========================================================================
# Shared inputs used: TREE_NWK. helper: helper/multi_TAD_eval_cooltools.py
#   (--baseline hicarn2 -> TAD_revision_evo2_vs_hicarn2.tsv).
TAD_REV_TSV = RESULT_DIR / "supp19" / "multi" / "TAD_revision_evo2_vs_hicarn2.tsv"


# ===========================================================================
# Supp 20 — DNA Zoo super-resolution heatmaps
# ===========================================================================
# Shared inputs used: SR_EVO2HIC_DIR, SR_HICARN2_DIR.
DNAZOO_RAW_HIC_DIR = HIC_DATA_ROOT / "dnazoo" / "raw_hic_fixed"
def DNAZOO_RAW_HIC(species: str) -> Path:
    return DNAZOO_RAW_HIC_DIR / f"{species}.hic"


# ===========================================================================
# Supp 21 — timing benchmark
# ===========================================================================
TIMING_TSV = REPO / "revision/Minor/timing_results.tsv"   # helper: revision/Minor/timing_benchmark.py


# ===========================================================================
# Supp 22 — pretraining over-representation (epi-prediction PCC by cell)
# ===========================================================================
# Shared inputs used: EPI_EVO2HIC_DIR, EPI_NODISTILL_DIR (panel b, same tables
#   as Fig 4 / Supp 11). Panel c reads the SR per-map PCC (human/mouse) below.
RESULT_SUPP22_HUMAN = RESULT_DIR / "supp22" / "human"     # PCC.csv
RESULT_SUPP22_MOUSE = RESULT_DIR / "supp22" / "mouse"


# ===========================================================================
# plot_dnazoo — circular multispecies improvement (companion to Fig 1)
# ===========================================================================
# Shared inputs used: TREE_NWK, CLAUDE_CLADE_DIR.
SPC_DNAZOO = RESULT_DIR / "dnazoo" / "multi" / "SPC.csv"
