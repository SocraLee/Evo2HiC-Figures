#!/usr/bin/env python3
"""Fan a produced result table out into every per-figure folder that reads it.

Figure scripts read from `result/<figure>/...` (one self-contained folder per
figure). A table consumed by several figures is duplicated into each of their
folders. After (re)producing a source table with one of the eval helpers, run:

    python -m helper.distribute_tables            # distribute everything present
    python -m helper.distribute_tables SR_METRICS # just one group

The SOURCES map below is the single source of truth for who-reads-what.
"""
from __future__ import annotations
import shutil, sys
from pathlib import Path

RESULT = Path(__file__).resolve().parents[1].parent / "result"

# group -> (canonical source file relative to result/, [other figure folders it
#           is ALSO copied into]). The canonical source lives in one figure
#           folder; the eval helper produces it there, this fans it out.
SOURCES = {
    # SR benchmark metric matrices (rows = test Hi-C maps, cols = methods)
    "SR_METRICS": [
        ("fig6/human/PCC.csv",  ["supp17/human", "supp22/human"]),
        ("fig6/human/SPC.csv",  ["supp17/human"]),
        ("fig6/human/PSNR.csv", ["supp17/human"]),
        ("fig6/human/SSIM.csv", ["supp17/human"]),
        ("fig6/mouse/PCC.csv",  ["supp16/mouse", "supp17/mouse", "supp22/mouse"]),
        ("fig6/mouse/SPC.csv",  ["supp16/mouse", "supp17/mouse"]),
        ("fig6/mouse/PSNR.csv", ["supp16/mouse", "supp17/mouse"]),
        ("fig6/mouse/SSIM.csv", ["supp16/mouse", "supp17/mouse"]),
        ("fig6/multi/SPC.csv",  ["fig1/multi", "supp17/multi", "dnazoo/multi"]),
        ("supp18/multi/PCC.csv", []),   # only Supp 18 consumes multi PCC
    ],
    # cross-cell evaluation (crosscell_eval.py -> result/supp12/)
    "CROSSCELL": [
        ("supp12/crosscell_matrix.tsv", []),
        ("supp12/spec_gap.tsv",         []),
        ("supp12/spec_pcc.tsv",         []),
        ("supp12/williams_pvals.tsv",   []),
    ],
    # multispecies TAD boundary F1 (multi_TAD_eval_cooltools.py)
    "TAD": [
        ("supp19/multi/TAD_revision_evo2_vs_hicarn2.tsv", []),
    ],
    # motif enrichment (motif_analysis.ipynb)
    "MOTIF": [
        ("fig5/motif_enrichment_stats_H3K27ac.csv", []),
    ],
}


def distribute(groups=None):
    n = 0
    for grp, items in SOURCES.items():
        if groups and grp not in groups:
            continue
        for src_rel, figdirs in items:
            src = RESULT / src_rel
            if not src.exists():
                print(f"  skip (missing source): {src_rel}")
                continue
            for fd in figdirs:
                dst = RESULT / fd / src.name
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
                n += 1
    print(f"distributed {n} table copies")


if __name__ == "__main__":
    distribute(set(sys.argv[1:]) or None)
