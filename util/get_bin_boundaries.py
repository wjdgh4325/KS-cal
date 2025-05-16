import numpy as np
import torch
import util


def get_bin_boundaries(args):


    train_loader = util.get_train_loader(args)

    all_times = []
    all_is_dead = []

    # for src, tgt, extra_surv_t, extra_censor_t in train_loader:
    # dont need ratio, ignoring 3rd dim of target
    for src, tgt in train_loader:
        tte = tgt[:, 0]
        is_dead = tgt[:, 1]
        all_times.append(tte)
        all_is_dead.append(is_dead)

    all_times = torch.cat(all_times)
    all_is_dead = torch.cat(all_is_dead).long()
    all_times_censored = all_times[all_is_dead == 0].cpu().numpy()
    all_times = all_times[all_is_dead == 1]
    all_times = all_times.cpu().numpy()
    percents = np.arange(args.num_cat_bins+1) * 100./args.num_cat_bins
    
    # BIN Boundaries has shape num_cat_bins+1.
    bin_boundaries = np.percentile(all_times, percents)
        
    print("percentile bin boundaries:", bin_boundaries)
    mid_points = (bin_boundaries[1:] + bin_boundaries[:-1])/2.
    # if args.phase == 'test':
    #     # get marginal counts
    #     lower_boundaries = bin_boundaries[1:-1].reshape(1, -1)
    #     # add counts for uncensored points
    #     all_times = all_times.reshape(-1, 1)
    #     tte_bins_uncensored = (all_times > lower_boundaries).sum(axis=-1).astype(float)
    #     tte_bins_uncensored_counts = np.unique(tte_bins_uncensored, return_counts=True)[1]

    #     # add counts/num_bins above censored times for censored points
    #     all_times_censored = all_times_censored.reshape(-1, 1)
    #     tte_bins_censored = (all_times_censored > lower_boundaries).sum(axis=-1).astype(float)
    #     tte_bins_censored = tte_bins_censored.reshape(-1, 1)
    #     indices = np.arange(args.num_cat_bins).reshape(1, -1)
    #     mask = (tte_bins_censored <= indices).astype(float)/(args.num_cat_bins - tte_bins_censored)

    #     tte_bins_censored_counts = np.sum(mask, axis=0)
    #     print("Censored counts", tte_bins_censored_counts, tte_bins_censored_counts.shape)
    #     print("Uncensored counts:", tte_bins_uncensored_counts, tte_bins_uncensored_counts.shape)
    #     marginal_counts = tte_bins_uncensored_counts + tte_bins_censored_counts

    #     # Use log since it's inputs to the softmax
    #     print("Margin counts:", marginal_counts)
    #     marginal_counts = np.log(marginal_counts)
    #     print("Margin counts:", marginal_counts)
        
    #     return bin_boundaries, mid_points, marginal_counts
    
    # else:
    return bin_boundaries, mid_points

