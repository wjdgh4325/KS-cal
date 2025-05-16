import torch
from openpyxl import load_workbook

def s_calibration(points, phase, is_dead, args, gamma=1.0, differentiable=False, device='cpu'):
    new_is_dead = is_dead.detach().clone()
    new_is_dead[points > 1. - 1e-4] = 1
    points = points.to(device).view(-1, 1)

    # NON-CENSORED POINTS
    points_dead = points[new_is_dead.long() == 1]
    s = ((torch.arange(20) + 1) / 20).to(device)

    zeros = torch.zeros(s.shape[0]).to(device)
    lower_diff_dead = points_dead - zeros
    upper_diff_dead = s - points_dead
    diff_product_dead = lower_diff_dead * upper_diff_dead
    
    assert lower_diff_dead.shape == upper_diff_dead.shape, (lower_diff_dead.shape, upper_diff_dead.shape)
    assert lower_diff_dead.shape == (points_dead.shape[0], s.shape[0])

    if differentiable == True:
        soft_membership_dead = torch.sigmoid(gamma * diff_product_dead)
        
    else:
        soft_membership_dead = (points_dead <= s).float()
        
    fraction_dead = soft_membership_dead.sum(0)/points.shape[0]
    
    # CENSORED POINTS
    points_cens = points[new_is_dead.long() == 0]
    upper_diff_for_soft_cens = s - points_cens
    
    zeros = torch.zeros(s.shape[0]).to(device)
    lower_diff_cens = points_cens - zeros
    upper_diff_cens = s - points_cens
    
    diff_product_cens = lower_diff_cens * upper_diff_cens
    
    assert s.shape[0] == diff_product_cens.shape[1]
    
    EPS = 1e-13
    right_censored_interval_size = 1 - points_cens + EPS
    
    if differentiable == True:
        bin_index_one = torch.sigmoid(gamma * diff_product_cens)

    else:
        bin_index_one = (points_cens <= s).float()

    upper_diff_within_bin = (upper_diff_for_soft_cens * bin_index_one)
    partial_bin_assigned_weight = (upper_diff_within_bin/right_censored_interval_size).sum(0) / points.shape[0]
    
    return torch.pow(fraction_dead + partial_bin_assigned_weight - s, 2).sum() / s.shape[0]
