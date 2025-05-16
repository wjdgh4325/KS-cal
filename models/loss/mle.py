import sys

import numpy as np
import torch
import torch.nn as nn

import util

sys.path.append("...")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def safe_log(x, EPS = 1e-4):
    return (x + EPS).log()

class MLE(nn.Module):
    def __init__(self, args):
        super(MLE, self).__init__()
        self.log_base = torch.FloatTensor([np.e]).to(DEVICE)
        self.eps = 1e-4
        self.args = args

    # Fast forward
    def forward(self, pred_params, tgt, model_dist):
        tte, is_dead = tgt[:, 0], tgt[:, 1]
        
        def bad(x):
            return torch.any(torch.isnan(x)) or torch.any(torch.isinf(x))

        is_dead = is_dead.long()
        if model_dist != 'cox':
            cdf = util.get_cdf_val(pred_params, tgt, self.args)
            logpdf = util.get_logpdf_val(pred_params, tgt, self.args)

            survival_func = 1.0 - cdf
            log_survival = safe_log(survival_func, EPS = self.eps)
            if bad(log_survival):
                print("BAD LOG SURVIVAL in MLE")

            if bad(logpdf):
                print("BAD LOG PDF in MLE")
            
            loglikelihood = is_dead * logpdf + (1 - is_dead) * log_survival
            nll = -1.0 * loglikelihood
            nll = nll.mean(dim=-1)

        else:
            tte = tte.view(-1, 1)
            tte_diff = tte - tte.T
            mask = (tte_diff <= 0).float().to(DEVICE)

            log_pl = mask * torch.exp(pred_params).reshape(-1)
            log_pl = torch.sum(log_pl, dim=1) / torch.sum(mask, dim=1)
            log_pl = torch.log(log_pl).reshape(-1, 1)
            
            nll = - torch.sum((pred_params - log_pl).reshape(-1) * is_dead) / torch.sum(is_dead)
            
            if bad(log_pl):
                print("BAD LOG PDF in MLE")

        return nll