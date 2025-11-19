import torch
from lifelines import KaplanMeierFitter
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def integrated_brier_score(train_tte, train_event, test_tte, test_event, cdf_test, time, batch_size=5000):
    G = KaplanMeierFitter()
    G.fit(train_tte.cpu().numpy(), 1 - train_event.cpu().numpy())

    surv_test = 1 - cdf_test
    IBS_acc = torch.zeros(len(time), device=DEVICE)
    N = test_tte.shape[0]

    ipcw2 = torch.tensor(G.survival_function_at_times(time.cpu().numpy()).values, device=DEVICE)
    ipcw2[ipcw2 == 0] = ipcw2[ipcw2 != 0].min()

    for i in range(0, N, batch_size):
        test_tte_b = test_tte[i:i+batch_size]
        test_event_b = test_event[i:i+batch_size]
        surv_b = surv_test[i:i+batch_size]

        ipcw1 = torch.tensor(G.survival_function_at_times(test_tte_b.cpu().numpy()).values, device=DEVICE).view(-1, 1)
        ipcw1[ipcw1 == 0] = ipcw1[ipcw1 != 0].min()

        uncensored_indicator = (test_tte_b.unsqueeze(1) <= time) & (test_event_b.unsqueeze(1) == 1)
        censored_indicator = (test_tte_b.unsqueeze(1) > time)

        uncensored_bs = (uncensored_indicator * ((0 - surv_b) ** 2) / ipcw1).sum(dim=0) / N
        censored_bs = (censored_indicator * ((1 - surv_b) ** 2) / ipcw2).sum(dim=0) / N

        IBS_acc += uncensored_bs + censored_bs

    IBS = torch.trapezoid(IBS_acc, time) / max(time)
    
    return IBS
