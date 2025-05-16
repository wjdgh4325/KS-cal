import torch
from lifelines import KaplanMeierFitter
import pandas as pd
import util

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def km_calibration(cdf, tte, is_dead, device='cpu'):
    kmf = KaplanMeierFitter()
    kmf.fit(tte.cpu(), is_dead.cpu())
    tte = torch.concat([torch.tensor([0]).to(device), tte])
    km_surv = torch.tensor(kmf.survival_function_at_times(pd.Series(tte.cpu().numpy())).values).view(-1).to(device)

    mean_cdf = torch.mean(cdf, dim=0)
    mean_surv = 1 - mean_cdf
    mean_surv = torch.concat([torch.tensor([1.0]).to(device), mean_surv])
    surv_diff = torch.pow(km_surv - mean_surv, 2)
    dt = tte[1:] - tte[:-1]
    km_cal = torch.sum(surv_diff[:-1] * dt) / max(tte)
    
    return km_cal

def km_calibration2(pred_params, tgt, tgt2, args):
    cdf_matrix = util.get_cdf_matrix(pred_params, tgt2, args)

    tte = tgt[:, 0]
    is_dead = tgt[:, 1]
    order = torch.argsort(tte)
    tte = tte[order]
    is_dead = is_dead[order]

    if args.model_dist != 'cox':
        cdf_matrix = cdf_matrix[:, order][order, :]
        
    mean_cdf = torch.mean(cdf_matrix, dim=0)
    mean_surv = 1 - mean_cdf

    unique_times, counts = torch.unique(tte, return_counts=True)
    n = len(tte)
    at_risk = n
    survival = []
    surv_val = 1.0
    idx = 0

    for utime in unique_times:
        mask = (tte == utime)
        di = is_dead[mask].sum().item()
        if at_risk > 0:
            surv_val *= (1.0 - di / at_risk)
        survival.extend([surv_val] * counts[idx].item())
        at_risk -= counts[idx].item()
        idx += 1

    km_surv = torch.tensor(survival, dtype=mean_surv.dtype, device=mean_surv.device)

    t_range = tte[-1] - tte[0]
    km_cal = (1 / t_range) * torch.sum(torch.abs(mean_surv - km_surv))

    return km_cal

def km_calibration3(pred_params, tgt, args):
    cdf_matrix = util.get_cdf_matrix(pred_params, tgt, args)

    tte = tgt[:, 0]
    is_dead = tgt[:, 1]
    order = torch.argsort(tte)
    tte = tte[order]
    is_dead = is_dead[order]

    if args.model_dist != 'cox':
        cdf_matrix = cdf_matrix[:, order][order, :]
        
    mean_cdf = torch.mean(cdf_matrix, dim=0)
    mean_surv = 1 - mean_cdf

    unique_times, counts = torch.unique(tte, return_counts=True)
    n = len(tte)
    at_risk = n
    survival = []
    surv_val = 1.0
    idx = 0

    for utime in unique_times:
        mask = (tte == utime)
        di = is_dead[mask].sum().item()
        if at_risk > 0:
            surv_val *= (1.0 - di / at_risk)
        survival.extend([surv_val] * counts[idx].item())
        at_risk -= counts[idx].item()
        idx += 1

    km_surv = torch.tensor(survival, dtype=mean_surv.dtype, device=mean_surv.device)

    t_range = tte[-1] - tte[0]
    km_cal = (1 / t_range) * torch.sum(torch.abs(mean_surv - km_surv))

    return km_cal