import numpy as np
import os
from scipy.stats import pearsonr, spearmanr
from scipy.stats import t as student_t
from dataset.track_loader import Track_Loader
from config import hic2tarck_dir
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))  # repo root (paths.py)
import paths as P  # noqa: E402

target = {
    'GM12878': Track_Loader(os.path.join(hic2tarck_dir, 'GM12878'), 2000),
    'H1ESC':   Track_Loader(os.path.join(hic2tarck_dir, 'H1ESC'),   2000),
    'K562':    Track_Loader(os.path.join(hic2tarck_dir, 'K562'),    2000),
}

# Prediction folders <dir>/<cell>/<chr>.npy, named in plot_revision/paths.py
# (Evo 2 [epi_baseline_evo2_<cell>, steps 44000 / 80000 / 70000], Hi-C-only [epi_baseline_hic_only/60000],
# no-distillation [epi_baseline_Evo2HiC_nodistill/40000], ridge on Hi-C [epi_baseline_ridge_HiC]).
_CELLS = ('GM12878', 'H1ESC', 'K562')
dir_Evo2 = {c: str(P.EPI_EVO2_DIR(c)) for c in _CELLS}           # shipped, chr9/10
dir_Evo2HiC = {c: str(P.EPI_EVO2HIC_DIR / c) for c in _CELLS}
dir_HiC_only = {c: str(P.EPI_HIC_ONLY_DIR / c) for c in _CELLS}     # shipped, chr9/10
# DNA-only = DNA + HiC trained from scratch (no Evo2 emb, no distillation).
dir_DNA_only = {c: str(P.EPI_NODISTILL_DIR / c) for c in _CELLS}
# Ridge_HiC = classical regression baseline (Ridge on local Hi-C row window).
dir_Ridge_HiC = {c: str(P.EPI_RIDGE_DIR / c) for c in _CELLS}

from scipy.stats import norm

def se_williams_overlap(r_ab, r_ac, r_bc, n):
    detR = 1 + 2*r_ab*r_ac*r_bc - (r_ab**2 + r_ac**2 + r_bc**2)
    if detR <= 0:
        raise ValueError("|R|<=0: the three correlations may be inconsistent.")
    r_bar = 0.5*(r_ab + r_ac)
    num = 2*((n-1)/(n-3))*detR + (r_bar**2) * (1 - r_bc)**3
    den = (n - 1) * (1 + r_bc)
    return np.sqrt(num / den)

def meta_compare_two_experiments(rab0, rac0, rbc0, n0,
                                 rab1, rac1, rbc1, n1,
                                 alternative="greater"):
    D0, D1 = rab0 - rac0, rab1 - rac1
    se0 = se_williams_overlap(rab0, rac0, rbc0, n0)
    se1 = se_williams_overlap(rab1, rac1, rbc1, n1)
    w0, w1 = 1/se0**2, 1/se1**2
    D_hat = (w0*D0 + w1*D1) / (w0 + w1)
    se_hat = np.sqrt(1/(w0 + w1))
    Z = D_hat / se_hat
    if alternative == "greater":
        p = norm.sf(Z)
    elif alternative == "less":
        p = norm.cdf(Z)
    else:
        p = 2*norm.sf(abs(Z))
    return {"D0": D0, "SE0": se0, "D1": D1, "SE1": se1,
            "D_pooled": D_hat, "SE_pooled": se_hat, "Z": float(Z), "p": float(p)}

corr = pearsonr

for cell in dir_Evo2HiC.keys():
    print(cell)
    stat_agbs = [[] for _ in range(5)]
    stat_agcs = [[] for _ in range(5)]
    for ch in [9,10]:
        a = np.load(os.path.join(dir_Evo2HiC[cell], f'{ch}.npy'))
        b = np.load(os.path.join(dir_Evo2[cell], f'{ch}.npy'))
        c = np.load(os.path.join(dir_HiC_only[cell], f'{ch}.npy'))

        t = target[cell].get(ch, 0, len(a[0])*2000, 0)
        for i in range(5):
            pat = corr(a[i], t[i])[0]
            pbt = corr(b[i], t[i])[0]
            pct = corr(c[i], t[i])[0]

            print(pat, pbt, pct)

            pab = corr(a[i], b[i])[0]
            pac = corr(a[i], c[i])[0]

            stat_agbs[i].extend([pat, pbt, pab, len(a[0])])
            stat_agcs[i].extend([pat, pct, pac, len(a[0])])

    for i in range(5):
        pagb = meta_compare_two_experiments(*(stat_agbs[i]))['p']
        pagc = meta_compare_two_experiments(*(stat_agcs[i]))['p']

        if stat_agbs[i][1]+stat_agbs[i][5]>stat_agcs[i][1]+stat_agcs[i][5]:
            print('(\'Evo2HiC\', \'Evo2\', ', pagb, ')')
        else:
            print('(\'Evo2HiC\', \'HiC-only\', ', pagc, ')')
