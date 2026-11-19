from evaluate.eval_utils import *
from scipy.stats import pearsonr, spearmanr
import pandas as pd
from scipy.ndimage import gaussian_filter1d

def get_diagonal_metrics(up0, up1, S, D, smooth = 0):

    pcc = np.zeros(D+1)
    spc = np.zeros(D+1)
    mse = np.zeros(D+1)
    mer = np.zeros(D+1)
    cnt = np.zeros(D+1)

    for i in tqdm(range(S+1, D)):
        d0 = up0[i].toarray().flatten()
        d1 = up1[i].toarray().flatten()
        
        if i>0:
            d0 = d0[:-i]
            d1 = d1[:-i]
            
        if smooth > 0:
            d0 = gaussian_filter1d(d0, smooth) 
            d1 = gaussian_filter1d(d1, smooth) 
        
        mask = np.isfinite(d1)

        d0 = d0[mask]
        d1 = d1[mask]

        if d0.size > 1:
            p, _ = pearsonr(d0, d1)
            s, _ = spearmanr(d0, d1)
            se = np.mean(np.square(d0-d1))
            er = np.mean(d0-d1)

            pcc[i] = np.nan_to_num(p)
            spc[i] = np.nan_to_num(s)
            mse[i] = se
            mer[i] = er

            cnt[i] = d0.size

    pcc[D] = np.mean(pcc[S+1:D])
    spc[D] = np.mean(spc[S+1:D])
    mse[D] = np.mean(mse[S+1:D])
    mer[D] = np.mean(mer[S+1:D])
    cnt[D] = np.sum(cnt[S+1:D])

    return pcc, spc, mse, mer, cnt
