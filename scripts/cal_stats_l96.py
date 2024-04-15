import os
import numpy as np
from torch.utils.data import Dataset
import torch
import glob, pdb, random
import matplotlib.pyplot as plt
import torch
from sklearn.neighbors import KernelDensity
from scipy.ndimage import gaussian_filter
from mpl_toolkits.axes_grid1 import make_axes_locatable

def cal_stats_l96(anchor_t, out_t, args = None, index = 0):
    def compute_batch_gradient(input, wrt = 'T', order = 1):
        input = input.clone()
        assert len(input.shape) == 3 # B x T x d
        B, T, d = input.shape
        if wrt == 'T':
            ans = input.permute(0, 2, 1).reshape(B*d, T)
            grad = torch.gradient(ans, dim = 1)[0]
            if order > 1:
                grad = torch.gradient(grad, dim = 1)[0]
            grad =  grad.reshape(B, d, T).permute(0, 2, 1)

            mask = torch.ones_like(ans)
            mask[:, :1*order] = 0
            mask[:, -1*order:] = 0
            mask = mask.reshape(B, d, T).permute(0, 2, 1)
        elif wrt == 'd':
            ans = input.reshape(B*T, d)
            grad = torch.gradient(ans, dim = 1)[0]
            if order > 1:
                grad = torch.gradient(grad, dim = 1)[0]
            grad = grad.reshape(B, T, d)

            mask = torch.ones_like(ans)
            mask[:, :1*order] = 0
            mask[:, -1*order:] = 0
            mask = mask.reshape(B, T, d)
        return grad, mask
    var = anchor_t
    var_k_1 = torch.roll(var, 1, dims = 2)
    var_k_2 = torch.roll(var, 2, dims = 2)
    var_k_p_1 = torch.roll(var, -1, dims = 2)
    grad_t, mask = compute_batch_gradient(var, wrt = 'T', order = 1)
    #########################out stats#########################################
    var_out = out_t
    var_k_1_out = torch.roll(var_out, 1, dims = 2)
    var_k_2_out = torch.roll(var_out, 2, dims = 2)
    var_k_p_1_out = torch.roll(var_out, -1, dims = 2)
    grad_t_out, mask = compute_batch_gradient(var_out, wrt = 'T', order = 1)
    ###################### assembling stats ####################################
    advection_stats = var_k_1 * (var_k_2 - var_k_p_1)
    advection_stats_out = var_k_1_out * (var_k_2_out - var_k_p_1_out)
    advection_stats = advection_stats[:, 2:-2, 2:-2]
    grad_t = grad_t[:, 2:-2, 2:-2]
    var = var[:, 2:-2, 2:-2]
    advection_stats_out = advection_stats_out[:, 2:-2, 2:-2]
    grad_t_out = grad_t_out[:, 2:-2, 2:-2]
    var_out = var_out[:, 2:-2, 2:-2]
    b_tz = advection_stats.shape[0]
    anchor_stats = torch.stack([advection_stats, grad_t, var], dim = -1).reshape(b_tz, -1, 3)
    out_stats = torch.stack([advection_stats_out, grad_t_out, var_out], dim = -1).reshape(b_tz, -1, 3)
    return anchor_stats, out_stats

def cal_stats_l1_score(anchor_t, out_t, anchor_param=[1], \
                    img_name='', folder_path = 'dist_plots', bins_test = 30, apply_gaussian = True, \
                    calculate_metric = 0, args = None, for_plot = False, for_stats = True, \
                    only_3d = False):
    def calculate_l1_score(hist_data_anchor, hist_data_predict):
        hist_data_anchor_normalized = hist_data_anchor[0].reshape(-1) / hist_data_anchor[0].reshape(-1).sum()
        hist_data_predic_normalized = hist_data_predict[0].reshape(-1) / hist_data_anchor[0].reshape(-1).sum()

        chi_score = abs(hist_data_anchor_normalized - hist_data_predic_normalized).sum() # + \
        return chi_score

    var = anchor_t
    var_k_1 = np.roll(var, 1, axis = 1)
    var_k_2 = np.roll(var, 2, axis = 1)
    var_k_p_1 = np.roll(var, -1, axis = 1)
    var_k_1, var_k_2, var_k_p_1 = torch.from_numpy(var_k_1), torch.from_numpy(var_k_2), torch.from_numpy(var_k_p_1)
    advection_stats = var_k_1 * (var_k_2 - var_k_p_1)

    ans = torch.from_numpy(var).permute(1, 0) # (d, T)
    grad_t = torch.gradient(ans, dim = 1)[0].permute(1, 0)
    mask = torch.ones_like(ans)
    mask[:, :1] = 0
    mask[:, -1:] = 0
    mask = mask.permute(1, 0)

    #########################out stats#########################################
    var_out = out_t
    var_k_1_out = np.roll(var_out, 1, axis = 1)
    var_k_2_out = np.roll(var_out, 2, axis = 1)
    var_k_p_1_out = np.roll(var_out, -1, axis = 1)
    var_k_1_out, var_k_2_out, var_k_p_1_out = torch.from_numpy(var_k_1_out), torch.from_numpy(var_k_2_out), torch.from_numpy(var_k_p_1_out)
    advection_stats_out = var_k_1_out * (var_k_2_out - var_k_p_1_out)

    ans_out = torch.from_numpy(var_out).permute(1, 0) # (d, T)
    grad_t_out = torch.gradient(ans_out, dim = 1)[0].permute(1, 0)
    mask = torch.ones_like(ans_out)
    mask[:, :1] = 0
    mask[:, -1:] = 0
    mask = mask.permute(1, 0)

    num_of_data = mask.sum()
    total_bins = np.sqrt(num_of_data)
    dim = 3
    bins_per_dim = np.floor(total_bins ** (1/dim))
    bins = int(bins_per_dim)

    anchor_stats = torch.stack([advection_stats.reshape(-1), grad_t.reshape(-1), torch.from_numpy(var.reshape(-1))]).permute(1, 0).numpy()
    hist_data = np.histogramdd(anchor_stats, bins = bins)
    out_stats = torch.stack([advection_stats_out.reshape(-1), grad_t_out.reshape(-1), torch.from_numpy(var_out.reshape(-1))]).permute(1, 0).numpy()
    hist_data_out = np.histogramdd(out_stats, bins = hist_data[-1])
    l1_score_3d = calculate_l1_score(hist_data, hist_data_out)

    return l1_score_3d
