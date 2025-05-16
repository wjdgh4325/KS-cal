import torch
import util

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from lifelines.utils import concordance_index

def concordance(args, test_loader, model):
    tte_per_batch = []
    is_dead_per_batch = []
    pred_time_per_batch = []

    for i, (src, tgt) in enumerate(test_loader):
        src = src.to(DEVICE)
        tgt = tgt.to(DEVICE)

        tte = tgt[:, 0]
        is_dead = tgt[:, 1]
        
        tte_per_batch.append(tte)
        is_dead_per_batch.append(is_dead)
        if args.model_dist == 'cox':
            pred_params = -model.forward(src.to(args.device))
            pred_time_per_batch.append(pred_params)

        else:
            pred_params = model.forward(src.to(args.device))
            pred = util.pred_params_to_dist(pred_params, tgt, args)
            pred_time = util.get_predict_time(pred, args)
            pred_time_per_batch.append(pred_time)
    
    ttes = torch.cat(tte_per_batch).cpu().numpy()
    is_deads = torch.cat(is_dead_per_batch).long().cpu().numpy()
    pred_times = torch.cat(pred_time_per_batch).cpu().numpy()
    
    concordance = concordance_index(ttes, pred_times, is_deads)

    return concordance