import numpy as np
from scipy.stats import pearsonr, spearmanr
from utils import *
eps = 1e-20

def compute_insulation_score(matrix:np.ndarray, window_size):
    L, _ = matrix.shape
    scores = []
    for i in range(L):
        if i < window_size or i+window_size >= L: scores.append(np.nan)
        else:
            mat = matrix[i-window_size:i, i+1:i+window_size+1]
            if np.isfinite(mat).sum() < mat.size * 0.5: 
                scores.append(np.nan)
            else:
                scores.append(np.nanmean(mat))
            
    return np.array(scores)

def compute_bounds(insulation_scores, delta_smooth_size):
    L = len(insulation_scores)
    
    mean_score = np.nanmean(insulation_scores)
    normalized_scores = np.log2(insulation_scores+eps) - np.log2(mean_score+eps)
    
    delta = []
    for i in range(L):
        left = np.nanmean(normalized_scores[i-delta_smooth_size:i])
        right = np.nanmean(normalized_scores[i+1:i+delta_smooth_size+1])
        delta.append(right-left)
        
    minimas = []
    for i in range(1, L):
        if delta[i-1] < 0 and delta[i] >0:
            minimas.append(i)

    bounds = []
    strengths = []
    for minima in minimas:
        l = minima-1
        while delta[l-1] is not None and delta[l-1] <= delta[l]:
            l -= 1
        
        r = minima
        while delta[r+1] is not None and delta[r+1] >= delta[r]:
            r += 1

        bounds.append(minima)
        strengths.append(delta[r] - delta[l])
        
    return bounds, strengths

def compute_TAD_similarity(prd_matrix, tgt_matrix, window_size = 50, delta_smooth_size = 10, bound_strength = 0.1, boundary_zone_size = 3):
    tgt_iscores = compute_insulation_score(tgt_matrix, window_size)
    tgt_bounds, tgt_strengths = compute_bounds(tgt_iscores, delta_smooth_size)

    prd_iscores = compute_insulation_score(prd_matrix, window_size)
    prd_bounds, prd_strengths = compute_bounds(prd_iscores, delta_smooth_size)

    mask = np.isfinite(tgt_iscores)
    insu_pcc, _ = pearsonr(tgt_iscores[mask], prd_iscores[mask])
    print(insu_pcc)

    tgt_bounds = sorted([tgt_bounds[i] for i, st in enumerate(tgt_strengths) if st >= bound_strength])
    prd_bounds = sorted([prd_bounds[i] for i, st in enumerate(prd_strengths) if st >= bound_strength])

    tgt_matched = np.zeros_like(tgt_bounds)
    prd_matched = np.zeros_like(prd_bounds)
    for i, pb in enumerate(prd_bounds):
        for j, tb in enumerate(tgt_bounds):
            if abs(pb-tb)<=boundary_zone_size:
                prd_matched[i] = 1
                tgt_matched[j] = 1

    lt = len(tgt_bounds)
    lp = len(prd_bounds)
    matched = np.sum(tgt_matched)

    print_info(f'predicted bounds: {lp}, target bounds: {lt}, matched bounds: {matched}')
    return 2*matched/(lp+lt), insu_pcc
