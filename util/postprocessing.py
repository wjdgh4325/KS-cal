import numpy as np
import torch
import util
import torch.nn.functional as F

def safe_logit(x, eps=1e-6):
    return torch.log((x + eps) / (1 - x + eps))

def postprocessing(args, cdf, is_dead, device='cpu', max_iters=10000, tol=1e-8, patience=100):
    EPS = 1e-8
    order = torch.argsort(cdf)
    cdf = cdf[order]
    cdf = cdf.unsqueeze(1)
    is_dead = is_dead[order].unsqueeze(1)
    N = cdf.shape[0]

    # Initialize learnable parameters
    a0_raw = torch.nn.Parameter(torch.tensor(0.0, device=device))
    b0 = torch.nn.Parameter(torch.tensor(0.0, device=device))
    alpha_raw = torch.nn.Parameter(torch.tensor(0.0, device=device))
    optimizer = torch.optim.Adam([a0_raw, b0, alpha_raw], lr=0.05)

    best_ks = float('inf')
    best_params = None
    patience_counter = 0

    for iter in range(max_iters):
        print(f"Iteration {iter+1}/{max_iters}", end="\r")

        with torch.set_grad_enabled(True):
            is_alive = (1 - is_dead).float()
            F_sorted = torch.sigmoid(torch.exp(a0_raw) * safe_logit(cdf) + b0) ** torch.exp(alpha_raw)

            denom = 1 - F_sorted + EPS
            weight = is_alive / denom
            F_weight = F_sorted * weight

            cum_weight = torch.cumsum(weight, dim=0)
            cum_F_weight = torch.cumsum(F_weight, dim=0)

            cum_weight_shifted = F.pad(cum_weight[:-1], (0, 0, 1, 0), value=0.0)
            cum_F_weight_shifted = F.pad(cum_F_weight[:-1], (0, 0, 1, 0), value=0.0)

            ecdf_cens = F_sorted * cum_weight_shifted - cum_F_weight_shifted
            ecdf_cens = torch.clamp(ecdf_cens, 0, N)

            ecdf_dead = torch.cumsum(is_dead, dim=0)
            ecdf_upper = (ecdf_dead + ecdf_cens) / N
            ecdf_upper = torch.clamp(ecdf_upper, 0, 1)
            ecdf_lower = ecdf_upper - is_dead / N

            # KS_upper = torch.pow(ecdf_upper - F_sorted, 2)
            # KS_lower = torch.pow(ecdf_lower - F_sorted, 2)
            KS_upper = torch.abs(ecdf_upper - F_sorted)
            KS_lower = torch.abs(ecdf_lower - F_sorted)
            KS_error = torch.max(torch.concat([KS_upper, KS_lower], dim=1), dim=1).values
            KS = torch.max(KS_error)

            optimizer.zero_grad()
            KS.backward()
            optimizer.step()

            # Gradient tolerance check
            grad_norm = torch.norm(torch.cat([a0_raw.grad.view(1), b0.grad.view(1), alpha_raw.grad.view(1)]))
            if grad_norm < tol:
                print(f"\nGradient norm below tolerance: {grad_norm:.6f}. Stopping early at iteration {iter+1}.")
                break

            # Early stopping check
            if KS.item() < best_ks:
                best_ks = KS.item()
                best_params = (a0_raw.clone(), b0.clone(), alpha_raw.clone())
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"\nEarly stopping at iteration {iter+1}. Best KS: {best_ks}")
                    break

    # Restore best parameters
    a0_raw, b0, alpha_raw = best_params
    a0 = torch.exp(a0_raw).item()
    b0 = b0.item()
    alpha = torch.exp(alpha_raw).item()
    print("Final parameters after early stopping:")
    print("a0:", a0)
    print("b0:", b0)
    print("alpha:", alpha)

    return a0, b0, alpha

