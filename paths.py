"""Central path configuration for plot_revision/ reproduction scripts.

Every script in plot_revision/ imports its external paths from here so the
reader only edits **one** file to point at their local data / checkpoints.

The four broad path groups
--------------------------
REPO            : top of the Evo2HiC source tree (auto-detected).
OUT_DIR         : destination for produced PDFs / stats.
CKPT_ROOT       : Hi-C model checkpoints (Evo2HiC, ablations, baselines).
DATA_*          : raw data — Hi-C, Evo 2 embeddings, DNA Zoo, motifs, ...

Override discipline
-------------------
Set the env vars listed below to point at your own copies; otherwise the
script falls back to the in-repo / lab-server defaults that produced the
published figures.

  EVO2HIC_REPO           -> repo root
  EVO2HIC_OUT_DIR        -> where to write PDFs/stats (default: <REPO>/Figures)
  EVO2HIC_CKPT_ROOT      -> Hi-C model checkpoints root
  EVO2HIC_HIC_DATA       -> raw Hi-C / multispecies data root
  EVO2HIC_EVO2_EMB_DIR   -> raw Evo 2 chr10 embedding dir (Supp 4 only)
"""
from __future__ import annotations

import os
from pathlib import Path


# ---------------------------------------------------------------------------
# Repo root + output dir
# ---------------------------------------------------------------------------
REPO = Path(os.environ.get(
    "EVO2HIC_REPO",
    Path(__file__).resolve().parents[1],
))

OUT_DIR = Path(os.environ.get(
    "EVO2HIC_OUT_DIR",
    REPO / "Figures",
))


# ---------------------------------------------------------------------------
# Result tables shipped in-repo (under result/)
# ---------------------------------------------------------------------------
RESULT_DIR        = REPO / "result"

# Held-out chrom (human / mouse) box+strip metric tables
RESULT_HUMAN_DIR  = RESULT_DIR / "human"     # PCC.csv / SPC.csv / PSNR.csv / SSIM.csv
RESULT_MOUSE_DIR  = RESULT_DIR / "mouse"

# DNA Zoo multi-species summary
RESULT_MULTI_DIR  = RESULT_DIR / "multi"
SPC_MULTI_CSV     = RESULT_MULTI_DIR / "SPC.csv"
TAD_REV_TSV       = RESULT_MULTI_DIR / "TAD_revision_evo2_vs_hicarn2.tsv"

# Retrieval gap tables (Supp 1)
RESULT_RETRIEVAL_DIR = RESULT_DIR / "retrieval"
GAP_HUMAN_CSV     = RESULT_RETRIEVAL_DIR / "gap_human.csv"
GAP_MOUSE_CSV     = RESULT_RETRIEVAL_DIR / "gap_mouse.csv"

# Cross-cell evaluation (Supp 12)
RESULT_CROSSCELL_DIR = RESULT_DIR / "crosscell"

# Motif enrichment (Fig 5)
MOTIF_STATS_CSV   = RESULT_DIR / "motif_enrichment_stats_H3K27ac.csv"


# ---------------------------------------------------------------------------
# Phylogenetic tree (Supp 19, Fig 6)
# ---------------------------------------------------------------------------
TREE_NWK          = REPO / "data/species_list_rawchrom.nwk"
CLAUDE_CLADE_DIR  = REPO / "data/claude"   # claude_<Clade>_clean.txt files


# ---------------------------------------------------------------------------
# Model checkpoints (Hi-C predictors)
# ---------------------------------------------------------------------------
CKPT_ROOT = Path(os.environ.get(
    "EVO2HIC_CKPT_ROOT",
    "/m-chimera/chimera/nobackup/yongkang/HiC_ckpt/checkpoints",
))

# Epigenomic prediction model + ablations + baselines
EPI_EVO2HIC_DIR      = CKPT_ROOT / "epi_prediction" / "model" / "track"
EPI_NODISTILL_DIR    = CKPT_ROOT / "epi_baseline_Evo2HiC_nodistill" / "40000" / "track"
EPI_EVO2EMB_HIC_DIR  = CKPT_ROOT / "epi_baseline_Evo2emb+hic"      / "48000" / "track"
EPI_RIDGE_DIR        = CKPT_ROOT / "epi_baseline_ridge_HiC"                  / "track"
EPI_HIC_ONLY_DIR     = CKPT_ROOT / "epi_baseline_hic_only"        / "60000" / "track"

EPI_EVO2_STEPS = {  # per-cell DNA-only Evo 2 epi baselines
    "GM12878": "44000",
    "H1ESC":   "80000",
    "K562":    "70000",
}
def EPI_EVO2_DIR(cell: str) -> Path:
    return CKPT_ROOT / f"epi_baseline_evo2_{cell}" / EPI_EVO2_STEPS[cell] / "track" / cell

# Hi-C super-resolution checkpoints (multispecies eval - Supp 20)
SR_EVO2HIC_DIR = CKPT_ROOT / "04_27_05_45_CDNAUNET_2000" / "170000" / "multi_10000"
SR_HICARN2_DIR = CKPT_ROOT / "hicarn2" / "hicarn2_resolution_raw_new" / "model" / "model_best.pth" / "multi_10000"

# Hi-C SR per-cell-line *_enhanced_test.hic files (Fig 3 / Supp 5-7)
SR_EVO2HIC_HUMAN_DIR = CKPT_ROOT / "04_27_05_45_CDNAUNET_2000" / "170000" / "human"

# Pretrained retrieval encoders (Fig 2 / Supp 1 / Supp 3).
# Two siglip-pretrain variants — `args.json:evo2_option` is the only diff.
#
#   PRETRAIN_*       -> "Evo2HiC"             (evo2_option=Yes)
#   PRETRAIN_NOEVO_* -> "Evo2HiC w/o Evo 2"   (evo2_option=No, alignment baseline)
#
# Inference outputs (embeddings, retrieval scores) are expected under
# `<ckpt_dir>/<step>/{human,human_embeds,human_mask,mouse,mouse_embeds,mouse_mask}/`
# — same layout the original training runs produced. Use the helper
# functions further down to address individual files.
PRETRAIN_CKPT_DIR        = CKPT_ROOT / "pretrained_weights"
PRETRAIN_CKPT            = PRETRAIN_CKPT_DIR / "model.pt"

PRETRAIN_NOEVO_CKPT_DIR  = CKPT_ROOT / "pretrain_weights_wo_evo2"
PRETRAIN_NOEVO_CKPT      = PRETRAIN_NOEVO_CKPT_DIR / "46000.pt"


# ---------------------------------------------------------------------------
# Derived inference outputs
# ---------------------------------------------------------------------------
# A few notebooks read files that are OUTPUTS of the inference scripts rather
# than primary data (DNA encoder embeddings, retrieval scores, attribution
# maps). They reference those outputs at their on-disk locations directly; to
# regenerate a missing one, run the matching inference command noted below.
#
# All commands assume:
#     cd $EVO2HIC_REPO
#     PY=$(conda run -n Evo2HiC which python)


# Per-(cell, chrom) epi prediction npy (Fig 4, Supp 8/9/10/11/12)
# Already shipped under EPI_EVO2HIC_DIR / baselines — no helper needed; the
# notebooks reference `<dir>/<cell>/<chr>.npy` directly. Regenerate any
# missing one with:
#     $PY -m inference.inference_CDNA1d \\
#         -ckpt <ckpt_dir>/<step>.pt --save-dir track --species human
#
# Seq2HiC per-locus predictions (Fig 3, Supp 5-7) are run live inside the
# notebook via `inference.inference_CDNA2d.load_model(...)` — no cache by
# design. If you want to detach plotting from inference, dump them via:
#     $PY -m inference.inference_CDNA2d \\
#         -ckpt <ckpt_dir>/model.pt --save-dir vis --input-file <hic>
#
# HiC SR enhanced .hic files (Fig 6, Supp 16/17/19/20) are also produced by
# inference_CDNA2d.py; SR_EVO2HIC_DIR / SR_HICARN2_DIR / SR_EVO2HIC_HUMAN_DIR
# already point at where they live.

# Seq2HiC checkpoints (Fig 3 / Supp 5-7).
# Each cell has 3 variants — args.json distinguishes them.
def SEQ2HIC_CKPT(cell: str, variant: str) -> Path:
    """variant: 'evo2hic' | 'hic_distilled' | 'evo2'"""
    return CKPT_ROOT / "seq2hic" / cell / variant / "model.pt"

# Borzoi predictions (Supp 10)
BORZOI_DIR = REPO / "baselines/epigenomic/results/borzoi"

# AlphaGenome FOLD_1 predictions (Supp 10, second baseline)
AG_DIR = REPO / "baselines/epigenomic/results/alphagenome_fold1"


# ---------------------------------------------------------------------------
# DNase ABC / CRISPRi gene-enhancer application (Supp 13)
# ---------------------------------------------------------------------------
# Supp 13 shows the model's predicted DNase x measured Hi-C, combined into an
# Activity-By-Contact (ABC) score, recovering CRISPRi-validated gene-enhancer
# links. scripts/plot_supp13.py is SELF-CONTAINED: it computes everything it
# needs from the raw inputs below (predicted tracks, measured tracks, .hic,
# CRISPRi benchmark) and caches the two intermediate TSVs under ABC_CACHE_DIR.
# The first run does the heavy compute (a few minutes; the panel-d sweep is
# ~65k gene-element pairs over 3 cell lines); later runs load the cache. Delete
# a cache file to force recompute.

# --- Raw inputs (all already exist / shipped in-repo) -----------------------
# Predicted genome-wide stitched tracks (activity source): the 5-track epi
# prediction cached under EPI_EVO2HIC_DIR (DNase=0, H3K27ac=2). Already shipped;
# regenerate any missing chrom with `inference_CDNA1d --save-dir track` (above).
def ABC_PRED_TRACK(cell: str, chrom: int) -> Path:
    return EPI_EVO2HIC_DIR / cell / f"{chrom}.npy"
# Measured tracks (BigWig) -> HIC2TRACK(cell); per-cell .hic -> HIC_RAW(acc)
# (both defined further below). read_count for the KR-normalized Hi-C loader:
EPI_PRED_ARGS_JSON = EPI_EVO2HIC_DIR.parents[1] / "args.json"   # epi_prediction/args.json
# 4DN accessions for each cell's Hi-C map (KR-normalized, used as the contact term)
ABC_CELL_HIC = {"GM12878": "4DNFI1UEG1HD", "H1ESC": "4DNFIQYQWPF5", "K562": "4DNFITUOMFUQ"}

# CRISPRi-FlowFISH gene-enhancer benchmark (panels a/b/c ground truth): the
# ENCODE-rE2G ensemble CRISPR benchmark (EPCrisprBenchmark, GRCh38, hg38 — no
# liftOver). 5-cell-type table; PPIF is the well-powered GM12878 chr10 locus
# (52 tested elements, 6 Regulated). Shipped in-repo; the upstream release is
# the EngreitzLab CRISPR_comparison / ENCODE-rE2G resource.
ABC_CRISPR_BENCH = REPO / "data" / "crispr" / "bench_5ct.tsv.gz"

# --- Cache (produced by plot_supp13.py itself on first run) ------------------
ABC_CACHE_DIR = RESULT_DIR / "abc_crispr"
# a/b/c: per-pair distance/contact/ABC scores for the 52 GM12878 PPIF elements.
# Columns: gene, tss, enh, dist, reg, s_dist, s_contact, s_abc_pred.
ABC_CRISPR_PPIF_TSV = ABC_CACHE_DIR / "ppif_crispr_scores.tsv"
# d: predicted-vs-measured ABC over chr9/10 candidate pairs, 3 cell lines.
# Columns: cell, gene_tss, dist, abc_pred, abc_meas.
ABC_ALIGN_TSV = ABC_CACHE_DIR / "abc_pred_vs_meas_chr910.tsv"


# ---------------------------------------------------------------------------
# Raw data roots (very large; lab server defaults)
# ---------------------------------------------------------------------------
HIC_DATA_ROOT = Path(os.environ.get(
    "EVO2HIC_HIC_DATA",
    "/m-chimera/chimera/nobackup/yongkang/HiC_data",
))

# DNA Zoo raw .hic files (used by Supp 20)
DNAZOO_RAW_HIC_DIR = HIC_DATA_ROOT / "dnazoo" / "raw_hic_fixed"
def DNAZOO_RAW_HIC(species: str) -> Path:
    return DNAZOO_RAW_HIC_DIR / f"{species}.hic"

# Evo 2 raw chr-level embeddings — same layout as config.evo2_embedding_map
EVO2_EMB_HUMAN = Path(os.environ.get(
    "EVO2HIC_EVO2_EMB_DIR",
    HIC_DATA_ROOT / "data/dna/human/hg38_2000_evo2_7b",
))
EVO2_EMB_MOUSE = HIC_DATA_ROOT / "data/dna/mouse/mm10_evo2_7b"
EVO2_EMB_DIR = EVO2_EMB_HUMAN   # legacy alias (Supp 4)

# Raw .hic files — `config.hic_data_dir`
HIC_RAW_DIR = HIC_DATA_ROOT / "data" / "hic" / "raw_hic"
def HIC_RAW(accession: str) -> Path:
    return HIC_RAW_DIR / f"{accession}.hic"

# Per-cell hic2track (BigWig + chrom-size + index) — `config.hic2tarck_dir`
HIC2TRACK_DIR = HIC_DATA_ROOT / "data" / "hic2track"
def HIC2TRACK(cell: str) -> Path:
    return HIC2TRACK_DIR / cell

# Reference genome FASTA — `config.DNA_map`
DNA_FASTA = {
    "human":     HIC_DATA_ROOT / "data/dna/human/hg38.fa",
    "mouse":     HIC_DATA_ROOT / "data/dna/mouse/mm10.fa",
    "zebrafish": HIC_DATA_ROOT / "data/dna/zebrafish/danRer11.fa",
}

# Motifs MEME database (Fig 5) — `config.motif_data`
MOTIF_MEME = HIC_DATA_ROOT / "data/dna/motifs.meme"
# Human gene annotation CSV used by Fig 5
HUMAN_MERGED_CSV = HIC_DATA_ROOT / "data/dna/human/merged.csv"

# Per-cell attribution arrays for Fig 5 motif analysis.
# Originally produced by `inference/inference_CDNA1d.py --interpretation`
# into `<ckpt>/human_inter/<cell>/{seq_<chr>.npy,<track>/attr_<chr>_filled.npy}`.
# The cleaned `epi_prediction/` checkpoint does NOT currently bundle these
# attribution outputs — set EVO2HIC_HUMAN_INTER to override if you have them.
HUMAN_INTER_DIR = Path(os.environ.get(
    "EVO2HIC_HUMAN_INTER",
    EPI_EVO2HIC_DIR.parent / "human_inter",   # epi_prediction/model/human_inter
))

# ORCA reference outputs (Fig 3 / Supp 5-7) — published Orca paper artifacts
ORCA_FIGURE_DIR    = HIC_DATA_ROOT / "orca_figure"
ORCA_MALLPREDS_PTH = ORCA_FIGURE_DIR / "mallpreds.pth"
# Orca's expected (binned) reference for 4DNFI9GMP2J8 at 4 kb (norm_factor)
ORCA_EXPECTED_NPY = HIC_DATA_ROOT / "orca/resources/4DNFI9GMP2J8.rebinned.mcool.expected.res4000.npy"

# Per-bin GM12878 epigenomic tracks (Supp 4 / Fig 2)
# First existing of: env override -> in-repo misc/ -> legacy HiC-DNA repo.
_TRACK_CANDIDATES = [
    Path(os.environ["EVO2HIC_TRACKS_GM12878_CHR10"])
        if "EVO2HIC_TRACKS_GM12878_CHR10" in os.environ else None,
    REPO / "misc" / "tracks_GM12878_10.npy",
    Path("/homes/gws/yongkang/HiC/HiC-DNA/misc/tracks_GM12878_10.npy"),
]
TRACKS_GM12878_CHR10_NPY = next(
    (p for p in _TRACK_CANDIDATES if p is not None and p.exists()),
    _TRACK_CANDIDATES[1],
)


# ---------------------------------------------------------------------------
# Hi-C loop functional decomposition (Supp 15, Supp 14)
# ---------------------------------------------------------------------------
# Two figures classify ground-truth HiCCUPS loop *anchors* by their epigenomic
# state, and show that the model's PREDICTED epigenomics (DNA + Hi-C -> 5
# tracks) recovers that state from sequence + contact alone:
#   Supp 15 (plot_supp15.py) : three representative loops (one per class:
#       insulator/CTCF, promoter-promoter, bivalent) — Hi-C map with the loop
#       circled + the per-anchor real-vs-predicted mark signature.
#   Supp 14 (plot_supp14.py) : confusion matrices comparing the rule applied to
#       the PREDICTED signal against the same rule on real ChIP-seq — functional
#       3-class (Active/Repressive/Quiescent) plus CTCF as an orthogonal binary
#       dimension — per cell line (GM12878, H1ESC).
#
# The data-producing ("inference") scripts live under revision/loop_decomposition/
# and run in the Evo2HiC env (torch + hic-straw). Run them in order:
#
#     cd $EVO2HIC_REPO/revision/loop_decomposition
#     PY=$(conda run -n Evo2HiC which python)
#
#   0. HiCCUPS loops on the held-out chr9+chr10 of the REAL Hi-C (juicer_tools
#      2.20.00; SCALE norm; -r 10000). Writes <acc>/merged_loops.bedpe under
#      HIC_LOOP_DIR for GM12878 (4DNFI1UEG1HD) and H1ESC (4DNFIQYQWPF5):
#        java -jar misc/juicer_tools.2.20.00.jar hiccups --cpu -k SCALE \
#            -c 9,10 -r 10000  <HIC_RAW(acc)>  <HIC_LOOP_DIR>/<acc>
#   1. $PY label_loops.py        # GT anchor mark-calls (real ChIP-seq, q=0.90
#                                #   genome-wide threshold) -> loop_labels.tsv
#   2. $PY rule_on_predicted.py  # identical rule on the PREDICTED tracks ->
#                                #   rule_on_predicted_labels.npz (+ metrics.tsv)
#
# Real DNase/ChIP-seq at anchors comes from the per-cell hic2track BigWigs
# (HIC2TRACK — the same source as the Fig 4 / Supp 8, 10-12, 16, 18 epigenomic figures, see
# that section above). The model's predicted tracks are the already-shipped
# EPI_EVO2HIC_DIR npys (see "Per-(cell, chrom) epi prediction npy" above;
# regenerate a missing chrom with the inference_CDNA1d --save-dir track command
# documented there). Supp 15 reads the predicted tracks directly; Supp 14 reads
# only the npz from step 2.
LOOP_APP_DIR          = REPO / "revision" / "loop_decomposition"
LOOP_LABELS_TSV       = LOOP_APP_DIR / "loop_labels.tsv"               # step 1
LOOP_RULE_NPZ         = LOOP_APP_DIR / "rule_on_predicted_labels.npz"  # step 2
LOOP_RULE_METRICS_TSV = LOOP_APP_DIR / "rule_on_predicted_metrics.tsv" # step 2

# HiCCUPS merged loops — config.hic_loop_dir / <accession> / merged_loops.bedpe
HIC_LOOP_DIR = HIC_DATA_ROOT / "data" / "hic" / "hiccups_loop"
def HIC_LOOP_BEDPE(accession: str) -> Path:
    return HIC_LOOP_DIR / accession / "merged_loops.bedpe"

# Predicted 5-track epigenomics for the loop figures = the canonical
# EPI_EVO2HIC_DIR npys (identical to Supp 13's ABC_PRED_TRACK).
def LOOP_PRED_TRACK(cell: str, chrom: int) -> Path:
    return EPI_EVO2HIC_DIR / cell / f"{chrom}.npy"


# ---------------------------------------------------------------------------
# Timing benchmark (Supp 21)
# ---------------------------------------------------------------------------
TIMING_TSV        = REPO / "revision/Minor/timing_results.tsv"


# ---------------------------------------------------------------------------
# Shared sys.path bootstrap (helpers that import dataset.* / config.*)
# ---------------------------------------------------------------------------
def add_repo_to_syspath() -> None:
    """Insert REPO at the front of sys.path so `import dataset.* / config`
    resolves to the in-repo modules (not whatever happens to be on PYTHONPATH).
    Idempotent."""
    import sys
    rp = str(REPO)
    if rp not in sys.path:
        sys.path.insert(0, rp)


def ensure_out_dir() -> Path:
    """Create OUT_DIR if missing and return it."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUT_DIR
