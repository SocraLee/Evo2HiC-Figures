"""Supplementary Figure 22 — pretraining cell-type over-representation does NOT
skew downstream generalization.

Reviewer concern: the contrastive-distillation Hi-C pool is biased toward a few
cell types (heart, colon, ...), which could skew the learned embeddings. This
figure quantifies the effect of that composition on downstream tasks.

Layout (3 panels, single column):
  a  Lineage composition of the 124-map human distillation pool. Biosources are
     grouped into lineages; lineages that the downstream evaluations fall into
     are highlighted. (heart -> Cardiac, the top bar; colon dominates
     Gastrointestinal, the 2nd bar.)
  b  Epigenomic prediction (CDNA1d): per-(cell x track) relative gain of Evo2HiC
     over the No-distillation ablation, vs the eval cell line's *lineage*
     representation in the pool. The eval cell lines (GM12878/H1ESC/K562) are
     held out at the cell-type level, so representation is necessarily measured
     at the lineage level (their exact biosource count in the pool is 0).
  c  Resolution enhancement (CDNA2d): per-map relative gain of Evo2HiC over the
     Unet (no-distillation) ablation, vs lineage representation, across every
     held-out test cell type. Squares = human, circles = mouse.

Take-away: the distillation benefit shows no positive dependence on lineage
representation in either task (Pearson r < 0, n.s. in b). Every cell type still
benefits (all gains > 0), and the most over-represented lineage (pluripotent)
shows the *smallest* relative gain — a baseline-ceiling effect, not a
representation advantage. Human and mouse cell types at the same lineage
representation receive comparable gains.

Relative gain = Evo2HiC / baseline - 1  (percent improvement). An absolute-gain
version (ΔPCC) of panels b/c is in revision/bias_analysis/bias_vs_gain_abs.* and
shows the same trend.

Generating the data
-------------------
Panel a reads only the shipped index `data/hic_index.tsv` (the train split,
human). No model access needed.

Panel b reads the SAME epigenomic-prediction PCC tables as Fig 4 / Supp 11 — see
the "Per-(cell, chrom) epi prediction npy" / EPI_* section in paths.py for how
they are produced. Concretely it needs the per-(cell,chrom) result.tsv of:
  EPI_EVO2HIC_DIR  / result.tsv   (Evo2HiC)
  EPI_NODISTILL_DIR/ result.tsv   (No-distillation ablation; same checkpoint
                                   plot_supp11.py labels "No-distillation").
Regenerate a missing one with (see paths.py):
  $PY -m inference.inference_CDNA1d -ckpt <ckpt>/<step>.pt \
      --save-dir track --species human

Panel c reads the shipped held-out-chrom super-resolution metric tables
  RESULT_SUPP22_HUMAN/PCC.csv  and  RESULT_SUPP22_MOUSE/PCC.csv
(columns: hic, Evo2HiC, SigLIP, Unet, HiCNN, HiCARN1, HiCARN2). `Unet` is the
no-distillation ablation. The `hic` accession is mapped back to a biosource via
data/hic_index.tsv. These are the same tables Fig 6 / Supp 17 use; see the SR
section in paths.py to regenerate them.

Outputs
-------
Figures/supplementary_22.pdf
Figures/supplementary_22.png
stdout : per-cell(-type) relative gains + the two Pearson correlations.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import pearsonr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paths import (REPO, RESULT_SUPP22_HUMAN, RESULT_SUPP22_MOUSE,                # noqa: E402
                   EPI_EVO2HIC_DIR, EPI_NODISTILL_DIR, ensure_out_dir)

OUT_PDF = ensure_out_dir() / "supplementary_22.pdf"
OUT_PNG = ensure_out_dir() / "supplementary_22.png"
HIC_INDEX = REPO / "data" / "hic_index.tsv"

TRACKS = ["DNase", "CTCF", "H3K27ac", "H3K27me3", "H3K4me3"]
CELLS = ["GM12878", "H1ESC", "K562"]

plt.style.use("seaborn-v0_8-white")
plt.rcParams.update({"font.size": 7, "font.family": "Arial",
                     "pdf.fonttype": 42, "ps.fonttype": 42})

# colors
HILITE = "#fb8072"
GREY = "#cfcfcf"
HUMAN_FILL = "#80b1d3"
MOUSE_FILL = "#fdb462"
CELL_COLORS = {"GM12878": "#bc80bd", "K562": "#fb8072", "H1ESC": "#80b1d3"}

# ---------------------------------------------------------------------------
# Single source of truth: training biosource -> lineage
# ---------------------------------------------------------------------------
LINEAGE = {
    "heart": "Cardiac",
    "colon": "Gastrointestinal", "colonic mucosa": "Gastrointestinal",
    "esophagus squamous epithelium": "Gastrointestinal",
    "esophagus muscularis mucosa": "Gastrointestinal", "stomach": "Gastrointestinal",
    "T cell": "T / NK lymphoid", "activated T-helper 2 cell": "T / NK lymphoid",
    "T-helper 2 cell": "T / NK lymphoid", "natural killer cell": "T / NK lymphoid",
    "Ramos": "B lymphoid", "OCI-LY7": "B lymphoid", "GM18951": "B lymphoid",
    "CD14-positive monocyte": "Myeloid", "dendritic cell": "Myeloid",
    "lung": "Respiratory",
    "H9": "Pluripotent (ESC/iPSC)", "CyT49": "Pluripotent (ESC/iPSC)",
    "WTC-11": "Pluripotent (ESC/iPSC)", "GM23248": "Pluripotent (ESC/iPSC)",
    "HFF": "Fibroblast",
    "gastrocnemius medialis": "Skeletal muscle", "psoas muscle": "Skeletal muscle",
    "ovary": "Reproductive", "placenta": "Reproductive", "testis": "Reproductive",
    "uterus": "Reproductive", "vagina": "Reproductive",
    "pancreas": "Endocrine", "adrenal gland": "Endocrine", "thyroid gland": "Endocrine",
    "aorta": "Vascular", "ascending aorta": "Vascular",
    "posterior vena cava": "Vascular", "tibial artery": "Vascular",
    "right lobe of liver": "Hepatic",
    "dorsolateral prefrontal cortex": "Neural", "head of caudate nucleus": "Neural",
    "sciatic nerve": "Neural",
    "HCT116": "Cancer (other)", "HeLa": "Cancer (other)", "A673": "Cancer (other)",
    "mammary epithelial cell": "Epithelial / skin", "lower leg skin": "Epithelial / skin",
    "kidney": "Renal",
    "prostate gland": "Urogenital",
}

# downstream eval cell type (any naming) -> lineage label above
EVAL_LINEAGE = {
    "H1ESC": "Pluripotent (ESC/iPSC)", "H1-hESC": "Pluripotent (ESC/iPSC)",
    "46C": "Pluripotent (ESC/iPSC)", "mESC line": "Pluripotent (ESC/iPSC)",
    "GM12878": "B lymphoid", "B cell": "B lymphoid",
    "K562": "Myeloid",
    "HepG2": "Hepatic",
    "IMR-90": "Fibroblast",
    "olfactory receptor cell": "Neural",
}


def lineage_counts():
    idx = pd.read_csv(HIC_INDEX, sep="\t")
    tr = idx[(idx.Organism == "human") & (idx.split == "train")].copy()
    tr["lineage"] = tr["Biosource"].map(LINEAGE)
    assert tr["lineage"].notna().all(), tr[tr.lineage.isna()].Biosource.unique()
    c = tr.groupby("lineage").size().sort_values(ascending=False)
    assert c.sum() == 124, c.sum()
    return c


LC = lineage_counts()
NTOT = int(LC.sum())


def rep_pct(lineage):
    return LC.get(lineage, 0) / NTOT * 100.0


# ---------------------------------------------------------------------------
# Panel b: epigenomic prediction PCC (chr9+chr10 macro-mean per track)
# ---------------------------------------------------------------------------
def _mean_per_cell(tsv_path):
    df = pd.read_csv(tsv_path, sep="\t")
    return {c: np.array([float(df[df.Name == c][t].mean()) for t in TRACKS])
            for c in CELLS}


def load_epi():
    evo_tsv = EPI_EVO2HIC_DIR / "result.tsv"
    nod_tsv = EPI_NODISTILL_DIR / "result.tsv"
    for p in (evo_tsv, nod_tsv):
        if not p.exists():
            raise FileNotFoundError(
                f"Missing {p}. This is the Fig 4 / Supp 11 epigenomic-prediction "
                f"PCC table; generate it with inference.inference_CDNA1d "
                f"--save-dir track (see paths.py EPI_* section).")
    return _mean_per_cell(evo_tsv), _mean_per_cell(nod_tsv)


def epi_rel_gain(cell, evo, nod):
    return (evo[cell] / nod[cell] - 1.0) * 100.0


# ---------------------------------------------------------------------------
# Panel c: resolution-enhancement per-map PCC (Evo2HiC vs Unet no-distill)
# ---------------------------------------------------------------------------
def load_resenh_rel():
    idx = pd.read_csv(HIC_INDEX, sep="\t")
    acc2bs = {r["Hi-C Accession"]: r["Biosource"] for _, r in idx.iterrows()}
    frames = []
    for sp, d in (("human", RESULT_SUPP22_HUMAN), ("mouse", RESULT_SUPP22_MOUSE)):
        df = pd.read_csv(d / "PCC.csv", sep="\t")
        df["species"] = sp
        df["cell"] = df["hic"].map(acc2bs)
        df["gain"] = (df["Evo2HiC"] / df["Unet"] - 1.0) * 100.0
        frames.append(df[["species", "cell", "gain"]])
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
def main():
    rng = np.random.default_rng(0)
    fig = plt.figure(figsize=(7.2, 6.4))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.05, 1.0], hspace=0.55,
                          wspace=0.32, left=0.30, right=0.97, top=0.95, bottom=0.10)
    axA = fig.add_subplot(gs[0, :])
    axB = fig.add_subplot(gs[1, 0])
    axC = fig.add_subplot(gs[1, 1])

    # ---------- Panel a: lineage composition ----------
    eval_lins = set(EVAL_LINEAGE.values())
    order = LC.sort_values()  # ascending -> largest on top in barh
    ypos = np.arange(len(order))
    colors = [HILITE if lin in eval_lins else GREY for lin in order.index]
    axA.barh(ypos, order.values / NTOT * 100, color=colors, edgecolor="white", height=0.78)
    axA.set_yticks(ypos)
    axA.set_yticklabels(order.index, fontsize=6.8)
    for y, lin in zip(ypos, order.index):
        axA.text(order[lin] / NTOT * 100 + 0.4, y, f"{order[lin]}",
                 va="center", fontsize=6, color="black")
    axA.set_xlabel("Percentage of pretraining pool", fontsize=7.5)
    axA.tick_params(axis="x", labelsize=7)
    axA.spines[["top", "right"]].set_visible(False)
    axA.legend(handles=[
        Line2D([0], [0], marker="s", color="none", markerfacecolor=HILITE,
               markersize=7, label="lineage evaluated downstream"),
        Line2D([0], [0], marker="s", color="none", markerfacecolor=GREY,
               markersize=7, label="other")],
        fontsize=6.3, loc="lower right", frameon=False)

    # ---------- Panel b: epig relative gain vs representation ----------
    evo, nod = load_epi()
    plot_x = {"GM12878": 2.1, "K562": 2.9, "H1ESC": rep_pct("Pluripotent (ESC/iPSC)")}
    pts_x, pts_y = [], []
    for cell in ["GM12878", "K562", "H1ESC"]:
        g = epi_rel_gain(cell, evo, nod)
        truex = rep_pct(EVAL_LINEAGE[cell]); px = plot_x[cell]
        x = np.full(5, px) + rng.uniform(-0.16, 0.16, 5)
        axB.scatter(x, g, s=15, color=CELL_COLORS[cell], alpha=0.85,
                    edgecolor="white", linewidth=0.3, zorder=3)
        axB.scatter([px], [g.mean()], marker="D", s=42, color=CELL_COLORS[cell],
                    edgecolor="black", linewidth=0.6, zorder=4)
        pts_x += [truex] * 5; pts_y += list(g)
        print(f"[epig] {cell:8s} rep={truex:.1f}%  mean rel gain={g.mean():+.2f}%")
    r, p = pearsonr(pts_x, pts_y)
    xs = np.array([1.2, 10.8]); sl, ic = np.polyfit(pts_x, pts_y, 1)
    axB.plot(xs, sl * xs + ic, "--", color="grey", lw=0.9, zorder=1)
    sig = "n.s." if p >= 0.05 else f"p = {p:.0e}"
    axB.set_xlabel("Lineage representation (%)", fontsize=7.5)
    axB.set_ylabel("Relative gain (%)", fontsize=7.5)
    axB.set_title("Epigenomic prediction", fontsize=8.5, pad=10)
    axB.set_xlim(0.8, 11.8); axB.set_xticks([2.4, 9.7])
    axB.tick_params(labelsize=7)
    axB.spines[["top", "right"]].set_visible(False)
    axB.legend(handles=[
        Line2D([0], [0], marker="D", color="none", markerfacecolor=CELL_COLORS[c],
               markeredgecolor="black", markersize=6, label=c)
        for c in ["GM12878", "K562", "H1ESC"]],
        fontsize=6.5, loc="upper right", frameon=False,
        handletextpad=0.5, labelspacing=0.5, borderpad=0.2)
    axB.text(0.975, 0.63, f"Pearson r = {r:.2f}\n{sig} (n = 15)",
             transform=axB.transAxes, fontsize=6, ha="right", va="top")

    # ---------- Panel c: res-enh relative gain vs representation ----------
    df = load_resenh_rel()
    map_x, map_y = [], []
    for (sp, cell), sub in df.groupby(["species", "cell"]):
        x = rep_pct(EVAL_LINEAGE[cell])
        mk = "s" if sp == "human" else "o"
        fill = HUMAN_FILL if sp == "human" else MOUSE_FILL
        jx = x + rng.uniform(-0.18, 0.18, len(sub))
        axC.scatter(jx, sub["gain"], s=12, color=fill, alpha=0.4, edgecolor="none", zorder=2)
        axC.scatter([x], [sub["gain"].mean()], marker=mk, s=46, color=fill,
                    edgecolor="black", linewidth=0.6, zorder=4)
        map_x += [x] * len(sub); map_y += list(sub["gain"])
        print(f"[res-enh] {sp:5s} {cell:24s} rep={x:.1f}%  "
              f"mean rel gain={sub['gain'].mean():+.2f}%  (n={len(sub)})")
    rc, pc = pearsonr(map_x, map_y)
    xs = np.array([1.2, 10.5]); sl, ic = np.polyfit(map_x, map_y, 1)
    axC.plot(xs, sl * xs + ic, "--", color="grey", lw=0.9, zorder=1)
    sigc = "n.s." if pc >= 0.05 else f"p = {pc:.0e}"
    axC.set_xlabel("Lineage representation (%)", fontsize=7.5)
    axC.set_ylabel("Relative gain (%)", fontsize=7.5)
    axC.set_title("Resolution enhancement", fontsize=8.5, pad=10)
    axC.set_xlim(0.8, 11.5); axC.set_xticks([2.4, 9.7])
    axC.tick_params(labelsize=7)
    axC.spines[["top", "right"]].set_visible(False)
    axC.legend(handles=[
        Line2D([0], [0], marker="s", color="none", markerfacecolor=HUMAN_FILL,
               markeredgecolor="black", markersize=7, label="Human"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=MOUSE_FILL,
               markeredgecolor="black", markersize=7, label="Mouse")],
        fontsize=6.5, loc="upper right", frameon=False,
        handletextpad=0.5, labelspacing=0.5, borderpad=0.2)
    axC.text(0.975, 0.75, f"Pearson r = {rc:.2f}\n{sigc} (n = 25)",
             transform=axC.transAxes, fontsize=6, ha="right", va="top")

    for ax, lab in [(axA, "a"), (axB, "b"), (axC, "c")]:
        ax.text(-0.02 if ax is axA else -0.22, 1.05, lab, transform=ax.transAxes,
                fontsize=11, fontweight="bold", va="bottom")

    fig.savefig(OUT_PDF, dpi=300, bbox_inches="tight")
    fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[ok] {OUT_PDF}")
    print(f"     epig Pearson r={r:.3f} (p={p:.3f}, n=15) | "
          f"res-enh r={rc:.3f} (p={pc:.3f}, n=25)")


if __name__ == "__main__":
    main()
