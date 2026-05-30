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
  EVO2HIC_EVO2_EMB_DIR   -> raw Evo 2 chr10 embedding dir (Supp 18 only)
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

# Seq2HiC vs ORCA per-window eval (Supp 16)
RESULT_SEQ2HIC_REVISION_DIR = RESULT_DIR / "seq2hic_revision"
RESULT_SEQ2HIC_NB_EVAL_DIR  = RESULT_SEQ2HIC_REVISION_DIR / "notebook_eval"

# Motif enrichment (Fig 5)
MOTIF_STATS_CSV   = RESULT_DIR / "motif_enrichment_stats_H3K27ac.csv"


# ---------------------------------------------------------------------------
# Phylogenetic tree (Supp 13, Fig 6)
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

# Hi-C super-resolution checkpoints (multispecies eval - Supp 14)
SR_EVO2HIC_DIR = CKPT_ROOT / "04_27_05_45_CDNAUNET_2000" / "170000" / "multi_10000"
SR_HICARN2_DIR = CKPT_ROOT / "hicarn2" / "hicarn2_resolution_raw_new" / "model" / "model_best.pth" / "multi_10000"

# Hi-C SR per-cell-line *_enhanced_test.hic files (Fig 3 / Supp 4-6)
SR_EVO2HIC_HUMAN_DIR = CKPT_ROOT / "04_27_05_45_CDNAUNET_2000" / "170000" / "human"

# Pretrained retrieval encoders (Fig 2 / Supp 1 / Supp 2).
# Two siglip-pretrain variants — `args.json:evo2_option` is the only diff.
#
#   PRETRAIN_*       -> "Evo2HiC"             (evo2_option=Yes)
#                       legacy: 04_15_05_07_CDNAUNET-siglip-pretrain_2000/48640/
#   PRETRAIN_NOEVO_* -> "Evo2HiC w/o Evo 2"   (evo2_option=No, alignment baseline)
#                       legacy: 02_24_22_44_CDNAUNET-siglip-pretrain_2000/46000/
#
# Inference outputs (embeddings, retrieval scores) are expected under
# `<ckpt_dir>/<step>/{human,human_embeds,human_mask,mouse,mouse_embeds,mouse_mask}/`
# — same layout the original training runs produced. Use the helper
# functions further down to address individual files.
PRETRAIN_CKPT_DIR        = CKPT_ROOT / "pretrained_weights"
PRETRAIN_CKPT            = PRETRAIN_CKPT_DIR / "model.pt"
PRETRAIN_STEP            = "48640"            # step folder that holds the derived outputs

PRETRAIN_NOEVO_CKPT_DIR  = CKPT_ROOT / "pretrain_weights_wo_evo2"
PRETRAIN_NOEVO_CKPT      = PRETRAIN_NOEVO_CKPT_DIR / "46000.pt"
PRETRAIN_NOEVO_STEP      = "46000"


# ---------------------------------------------------------------------------
# Derived inference outputs
# ---------------------------------------------------------------------------
# Each helper below points at the **expected on-disk location** of one
# inference artifact (npy / tsv / hic). If the file is missing, run the
# command in the helper's docstring to regenerate it; the inference scripts
# always write to `<dirname(ckpt)>/<save_dir>/...`, so the helpers and the
# scripts agree by construction.
#
# All commands assume:
#     cd $EVO2HIC_REPO
#     PY=$(conda run -n Evo2HiC which python)

def encoder_embed(ckpt_dir: Path, step: str, species: str, chrom: int,
                  reverse: bool = False, projected: bool = False) -> Path:
    """Per-chromosome DNA encoder embedding (Fig 2, Supp 1, Supp 2).

    Generate (once per ckpt):
        $PY -m inference.inference_dna_embeds \\
            -ckpt <ckpt_dir>/<step>.pt --save-dir <step>/<species>_embeds \\
            --species <species>

    NOTE: plot_supp1.ipynb / plot_supp2.ipynb / plot_Fig2.ipynb currently
    reference *literal* paths like `pretrained_weights/human_embeds/dna_emb_9.npy`
    (the `<step>` segment got dropped during the round-2 regex pass). Until
    they're switched over to this helper, the inference outputs above must
    be placed at the literal location the notebook expects, or the notebook
    cells edited by hand to use this helper.
    """
    name = f"dna_{'proj_' if projected else ''}emb{'_rev' if reverse else ''}_{chrom}.npy"
    return ckpt_dir / step / f"{species}_embeds" / name


def retrieval_sim(ckpt_dir: Path, step: str, species: str,
                  accession: str, chrom: int, kind: str) -> Path:
    """Per-(cell, chrom) positive/negative similarity scores (Supp 1).

    kind: 'pos' | 'neg'.
    Generate:
        $PY -m inference.retrieval_siglip \\
            -ckpt <ckpt_dir>/<step>.pt --save-dir <step> \\
            --species <species>
    (Produces sim_pos/neg per cell-line under <step>/<species>/<acc>/.)
    """
    return ckpt_dir / step / species / accession / f"sim_{kind}_{chrom}.npy"


def retrieval_recall(ckpt_dir: Path, step: str, species: str) -> Path:
    """Aggregate retrieval recall table (Fig 2).

    Generate: same command as `retrieval_sim` above — the script writes
    recall.tsv into <step>/<species>_mask/ alongside the per-cell scores.
    """
    return ckpt_dir / step / f"{species}_mask" / "recall.tsv"


# epigenomic-prediction attribution arrays (Fig 5)
def epi_attr_seq(epi_ckpt_dir: Path, cell: str, chrom: int) -> Path:
    """DNA sequence (one-hot or token IDs) at each interpretation locus.

    Generate (once per ckpt, takes ~hours on a single GPU):
        $PY -m inference.inference_CDNA1d \\
            -ckpt <epi_ckpt_dir>/model.pt --save-dir model/human_inter \\
            --species human --interpretation
    """
    return epi_ckpt_dir / "model" / "human_inter" / cell / f"seq_{chrom}.npy"


def epi_attr_track(epi_ckpt_dir: Path, cell: str, track: str, chrom: int) -> Path:
    """Per-(cell, track, chrom) attribution map produced by the same
    `inference_CDNA1d --interpretation` run as `epi_attr_seq`.
    """
    return epi_ckpt_dir / "model" / "human_inter" / cell / track / f"attr_{chrom}_filled.npy"


# Per-(cell, chrom) epi prediction npy (Fig 4, Supp 7/10/11/12/19)
# Already shipped under EPI_EVO2HIC_DIR / baselines — no helper needed; the
# notebooks reference `<dir>/<cell>/<chr>.npy` directly. Regenerate any
# missing one with:
#     $PY -m inference.inference_CDNA1d \\
#         -ckpt <ckpt_dir>/<step>.pt --save-dir track --species human
#
# Seq2HiC per-locus predictions (Fig 3, Supp 4-5-6) are run live inside the
# notebook via `inference.inference_CDNA2d.load_model(...)` — no cache by
# design. If you want to detach plotting from inference, dump them via:
#     $PY -m inference.inference_CDNA2d \\
#         -ckpt <ckpt_dir>/model.pt --save-dir vis --input-file <hic>
#
# HiC SR enhanced .hic files (Fig 6, Supp 8/13/14/15) are also produced by
# inference_CDNA2d.py; SR_EVO2HIC_DIR / SR_HICARN2_DIR / SR_EVO2HIC_HUMAN_DIR
# already point at where they live.

# Seq2HiC checkpoints (Fig 3 / Supp 4-6).
# Each cell has 3 variants — args.json distinguishes them.
def SEQ2HIC_CKPT(cell: str, variant: str) -> Path:
    """variant: 'evo2hic' | 'hic_distilled' | 'evo2'"""
    return CKPT_ROOT / "seq2hic" / cell / variant / "model.pt"

# Borzoi predictions (Supp 10)
BORZOI_DIR = REPO / "baselines/epigenomic/results/borzoi"


# ---------------------------------------------------------------------------
# Raw data roots (very large; lab server defaults)
# ---------------------------------------------------------------------------
HIC_DATA_ROOT = Path(os.environ.get(
    "EVO2HIC_HIC_DATA",
    "/m-chimera/chimera/nobackup/yongkang/HiC_data",
))

# DNA Zoo raw .hic files (used by Supp 14)
DNAZOO_RAW_HIC_DIR = HIC_DATA_ROOT / "dnazoo" / "raw_hic_fixed"
def DNAZOO_RAW_HIC(species: str) -> Path:
    return DNAZOO_RAW_HIC_DIR / f"{species}.hic"

# Evo 2 raw chr-level embeddings — same layout as config.evo2_embedding_map
EVO2_EMB_HUMAN = Path(os.environ.get(
    "EVO2HIC_EVO2_EMB_DIR",
    HIC_DATA_ROOT / "data/dna/human/hg38_2000_evo2_7b",
))
EVO2_EMB_MOUSE = HIC_DATA_ROOT / "data/dna/mouse/mm10_evo2_7b"
EVO2_EMB_DIR = EVO2_EMB_HUMAN   # legacy alias (Supp 18)

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

# ORCA reference outputs (Fig 3 / Supp 4-6) — published Orca paper artifacts
ORCA_FIGURE_DIR    = HIC_DATA_ROOT / "orca_figure"
ORCA_MALLPREDS_PTH = ORCA_FIGURE_DIR / "mallpreds.pth"
# Orca's expected (binned) reference for 4DNFI9GMP2J8 at 4 kb (norm_factor)
ORCA_EXPECTED_NPY = HIC_DATA_ROOT / "orca/resources/4DNFI9GMP2J8.rebinned.mcool.expected.res4000.npy"

# Per-bin GM12878 epigenomic tracks (Supp 18 / Fig 2)
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
# Timing benchmark (Supp 17)
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
