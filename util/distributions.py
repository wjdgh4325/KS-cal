
import torch
import torch.nn.functional as F
import util

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def pred_params_to_cat(pred_params, args):
    pred = util.CatDist(pred_params, args)

    return pred

def pred_params_to_weibull(pred_params):
    pre_scale = pred_params[:, 0]
    # scale = pre_scale + 1.
    scale = pre_scale
    pre_k = pred_params[:, 1]
    # k = pre_k.sigmoid() + 1.0
    k = pre_k
    pred = torch.distributions.Weibull(scale, k)

    return pred

def pred_params_to_lognormal_params(pred_params):
    mu = pred_params[:, 0]
    pre_log_sigma = pred_params[:, 1]
    log_sigma = F.softplus(pre_log_sigma) - 0.5
    sigma = log_sigma.clamp(max=10).exp()
    sigma = sigma + 1e-4

    return mu, sigma

def pred_params_to_lognormal(pred_params):
    mu, sigma = pred_params_to_lognormal_params(pred_params)
    pred = torch.distributions.LogNormal(mu, sigma)

    return pred

def pred_params_to_dist(pred_params, tgt, args):
    if args.model_dist == 'lognormal':
        pred = pred_params_to_lognormal(pred_params)
    
    elif args.model_dist == 'weibull':
        pred = pred_params_to_weibull(pred_params)

    elif args.model_dist in ['cat', 'mtlr']:
        pred = pred_params_to_cat(pred_params, args)

    else:
        pred = pred_params_to_cox(pred_params, tgt)

    return pred

def pred_params_to_cox(pred_params, tgt):
    EPS = 1e-8
    risk_score = torch.exp(pred_params)
    tte, is_dead = tgt[:, 0], tgt[:, 1]

    order = torch.argsort(tte)
    tte = tte[order].reshape(-1)
    is_dead = is_dead[order]
    risk_score = risk_score[order].reshape(-1)

    unique_times, inverse_indices = torch.unique(tte, return_inverse=True)
    n_unique = unique_times.shape[0]

    event_count = torch.zeros(n_unique, device=DEVICE).scatter_add(0, inverse_indices, is_dead)

    risk_at_time = torch.zeros(n_unique, device=DEVICE)
    for i in range(n_unique):
        risk_at_time[i] = risk_score[tte >= unique_times[i]].sum()

    hazard_jump = event_count / (risk_at_time + EPS)
    H = torch.cumsum(hazard_jump, dim=0)

    H_i = H[inverse_indices]
    S = torch.exp(-H_i) + EPS
    cdf = 1 - S ** risk_score

    return cdf

def get_cdf_val(pred_params, tgt, args):
    
    pred = pred_params_to_dist(pred_params, tgt, args)

    if args.model_dist in ['cat', 'mtlr']:
        tte, is_dead, ratio = tgt[:, 0], tgt[:, 1], tgt[:, 2]
        cdf = pred.cdf(tte, ratio)

    elif args.model_dist in ['lognormal', 'weibull']:
        tte, is_dead = tgt[:, 0], tgt[:, 1]
        cdf = pred.cdf(tte + 1e-4)
        
    else:
        cdf = pred

    return cdf
    
def get_cdf_matrix(pred_params, tgt, args):
    pred = pred_params_to_dist(pred_params, tgt, args)
    N = pred_params.shape[0]
    if args.model_dist in ['lognormal', 'weibull']:
        tte, is_dead = tgt[:, 0], tgt[:, 1]
        tte = tte + 1e-4
        tte_expand = tte.unsqueeze(0).expand(N, -1)  # [n, n]
        
        pred = pred_params_to_dist(pred_params, tgt=None, args=args)
        
        if args.model_dist == 'lognormal':
            mu, sigma = pred_params_to_lognormal_params(pred_params)
            mu = mu.unsqueeze(1).expand(N, N)     # [n, n]
            sigma = sigma.unsqueeze(1).expand(N, N)  # [n, n]
            dist = torch.distributions.LogNormal(mu, sigma)
        
        elif args.model_dist == 'weibull':
            scale = pred_params[:, 0].unsqueeze(1).expand(N, N)     # [n, n]
            concentration = pred_params[:, 1].unsqueeze(1).expand(N, N)
            dist = torch.distributions.Weibull(scale, concentration)
        
        cdf_matrix = dist.cdf(tte_expand)  # [n, n]
    
    elif args.model_dist in ['cat', 'mtlr']:
        tte, is_dead, ratio = tgt[:, 0], tgt[:, 1], tgt[:, 2]
        tte_expand = tte.unsqueeze(0).expand(N, -1)  # [n, m]
        ratio = ratio.unsqueeze(0).expand(N, -1)
        cdf_matrix = pred.cdf_matrix(tte_expand, ratio)  # [n, m]

    elif args.model_dist == 'cox':
        EPS = 1e-8
        risk_score = torch.exp(pred_params)
        tte, is_dead = tgt[:, 0], tgt[:, 1]

        order = torch.argsort(tte)
        tte = tte[order]
        is_dead = is_dead[order]
        risk_score = risk_score[order]

        unique_times, inverse_indices = torch.unique(tte, return_inverse=True)

        event_count = torch.zeros_like(unique_times).scatter_add(0, inverse_indices, is_dead)

        risk_at_time = torch.zeros_like(unique_times)
        for i, t in enumerate(unique_times):
            risk_at_time[i] = risk_score[tte >= t].sum()

        hazard_jump = event_count / (risk_at_time + EPS)
        H = torch.cumsum(hazard_jump, dim=0)

        H_i = H[inverse_indices]  # shape: [n]

        S = torch.exp(-H_i) + EPS
        cdf_matrix = 1 - S ** risk_score

    else:
        raise ValueError("Unsupported distribution in get_cdf_matrix")

    return cdf_matrix

def get_predict_time(pred, args):
    if args.model_dist in ['cat', 'mtlr']:
        return pred.predict_time()
    
    elif args.model_dist == 'lognormal':
        if args.pred_type == 'mean':
            pred_time = pred.mean

        elif args.pred_type == 'mode':
            pred_time = util.log_normal_mode(pred)

    elif args.model_dist == 'weibull':
        logtwo = torch.tensor([2.0]).to(DEVICE).log()
        inverse_concentration = 1.0 / pred.concentration
        pred_time = pred.scale * logtwo.pow(inverse_concentration)
        
        if torch.any(torch.isnan(pred_time)) or torch.any(torch.isinf(pred_time)):
            print(":(")
    
    else:
        assert False, "wrong dist or pred type in predict time in utils"
    
    return pred_time

def get_logpdf_val(pred_params, tgt, args):

    pred = pred_params_to_dist(pred_params, tgt, args)
    tte = tgt[:, 0]

    if args.model_dist in ['lognormal', 'weibull']:
        tte = tte + 1e-4
    log_prob = pred.log_prob(tte)
    
    return log_prob

def log_normal_mode(pytorch_distribution_object):
    return (pytorch_distribution_object.loc - pytorch_distribution_object.scale.pow(2)).exp()
