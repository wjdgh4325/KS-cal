import torch

def d_calibration(points, is_dead, args, phase, nbins=20, differentiable=False, gamma=1.0, device='cpu'):
    new_is_dead = is_dead.detach().clone().to(device)
    new_is_dead[points > 1. - 1e-4] = 1
    points = points.to(device).view(-1, 1)
    
    bin_width = 1.0 / nbins
    bin_indices = torch.arange(nbins).view(1, -1).float().to(device)
    bin_a = bin_indices * bin_width
    noise = 1e-6 / nbins * torch.rand(size=bin_indices.shape).to(device)
    if not differentiable:
        noise = noise * 0

    cum_noise = torch.cumsum(noise, dim=1)
    bin_width = torch.tensor([bin_width] * nbins).to(device) + cum_noise
    bin_b = bin_a + bin_width

    bin_b_max = bin_b[:, -1]
    bin_b = bin_b / bin_b_max
    bin_a[:, 1:] = bin_b[:, :-1]
    bin_width = bin_b - bin_a
    
    # CENSORED POINTS
    points_cens = points[new_is_dead.long() == 0]
    upper_diff_for_soft_cens = bin_b - points_cens

    bin_b[:, -1] = 2.
    bin_a[:, 0] = -1.

    lower_diff_cens = points_cens - bin_a
    upper_diff_cens = bin_b - points_cens
    diff_product_cens = lower_diff_cens * upper_diff_cens
    
    if differentiable:
        bin_index_one = torch.sigmoid(gamma * diff_product_cens)
        exact_bins_next = torch.sigmoid(-gamma * lower_diff_cens)

    else:
        bin_index_one = (lower_diff_cens >= 0).float() * (upper_diff_cens > 0).float()
        exact_bins_next = (lower_diff_cens <= 0).float()

    EPS = 1e-13
    right_censored_interval_size = 1 - points_cens + EPS

    upper_diff_within_bin = (upper_diff_for_soft_cens * bin_index_one)

    full_bin_assigned_weight = (exact_bins_next * bin_width.view(1, -1) / right_censored_interval_size.view(-1, 1)).sum(0)
    partial_bin_assigned_weight = (upper_diff_within_bin / right_censored_interval_size).sum(0)

    assert full_bin_assigned_weight.shape == partial_bin_assigned_weight.shape, (full_bin_assigned_weight.shape, partial_bin_assigned_weight.shape)

    # NON-CENSORED POINTS
    points_dead = points[new_is_dead.long() == 1]

    lower_diff = points_dead - bin_a
    upper_diff = bin_b - points_dead
    diff_product = lower_diff * upper_diff
    
    assert lower_diff.shape == upper_diff.shape, (lower_diff.shape, upper_diff.shape)
    assert lower_diff.shape == (points_dead.shape[0], bin_a.shape[1])

    if differentiable:
        soft_membership = torch.sigmoid(gamma * diff_product)
        fraction_in_bins = soft_membership.sum(0)

    else:
        exact_membership = (lower_diff >= 0).float() * (upper_diff > 0).float()
        fraction_in_bins = exact_membership.sum(0)

    assert fraction_in_bins.shape == (nbins, ), fraction_in_bins.shape

    fraction_in_bins = (fraction_in_bins + full_bin_assigned_weight + partial_bin_assigned_weight) / points.shape[0]
    
    return torch.pow(fraction_in_bins - bin_width, 2).sum()