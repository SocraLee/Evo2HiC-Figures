import torch
from math import exp
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
from evaluate.eval_utils import *

def gaussian(width, sigma):
    gauss = torch.Tensor([exp(-(x-width//2)**2 / float(2 * sigma**2)) for x in range(width)])
    return gauss / gauss.sum()

def create_window(window_size, channel, sigma=3):
    _1D_window = gaussian(window_size, sigma).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window

def get_ssim(pred_matrix, target_matrix, mask=None, device="cpu", patch_size=200, window_size=11):
    pred_matrix_torch = torch.from_numpy(pred_matrix).unsqueeze(0).unsqueeze(0)
    target_matrix_torch = torch.from_numpy(target_matrix).unsqueeze(0).unsqueeze(0)
    
    if mask is not None:
        mask_torch = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0)
        pred_matrix_torch = pred_matrix_torch * mask_torch
        target_matrix_torch = target_matrix_torch * mask_torch

    window = create_window(window_size, pred_matrix_torch.size()[1]).type_as(pred_matrix_torch).to(device)

    channel = pred_matrix_torch.size()[1]
    total_size = pred_matrix.shape[-1]
    patch_size = patch_size
    padding = window_size // 2

    ssim_all = 0
    for i in tqdm(range(0, total_size, patch_size)):
        for j in range(0, total_size, patch_size):
            pred_patch = pred_matrix_torch[:, :, max(i-padding, 0):min(i+patch_size+padding, total_size), max(j-padding, 0):min(j+patch_size+padding, total_size)].to(device)
            target_patch = target_matrix_torch[:, :, max(i-padding, 0):min(i+patch_size+padding, total_size), max(j-padding, 0):min(j+patch_size+padding, total_size)].to(device)
            if mask is not None:
                mask_patch = mask_torch[:, :, max(i-padding, 0):min(i+patch_size+padding, total_size), max(j-padding, 0):min(j+patch_size+padding, total_size)].to(device)
        
            mu1 = F.conv2d(pred_patch, window, padding=5, groups=channel)
            mu2 = F.conv2d(target_patch, window, padding=5, groups=channel)
            mu1_sq = mu1.pow(2)
            mu2_sq = mu2.pow(2)
            mu1_mu2 = mu1 * mu2
            sigma1_sq = F.conv2d(pred_patch * pred_patch, window, padding=5, groups=channel) - mu1_sq
            sigma2_sq = F.conv2d(target_patch * target_patch, window, padding=5, groups=channel) - mu2_sq
            sigma12 = F.conv2d(pred_patch * target_patch, window, padding=5, groups=channel) - mu1_mu2
            C1 = 0.01 ** 2
            C2 = 0.03 ** 2
            ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
            if mask is not None:
                ssim_map = (ssim_map * mask_patch).to("cpu")
            else:
                ssim_map = ssim_map.to("cpu")
            start_idx_i = min(i, padding)
            start_idx_j = min(j, padding)
            end_idx_i = min(start_idx_i + patch_size, patch_size)
            end_idx_j = min(start_idx_j + patch_size, patch_size)
            ssim_all += ssim_map[:, :, start_idx_i:end_idx_i, start_idx_j:end_idx_j].sum()
    
    if mask is not None:
        chr_ssim = ssim_all / mask.sum()
    else:
        chr_ssim = ssim_all / (total_size * total_size)
    
    return chr_ssim.item()

def get_mse(pred_matrix, target_matrix, mask=None):
    if mask is not None:
        return (np.square(pred_matrix - target_matrix) * mask).sum() / mask.sum()
    else:
        return np.square(pred_matrix - target_matrix).mean()
