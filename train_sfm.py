import torch

import models
import optim
import util
from args import TrainArgParser
from evaluator import ModelEvaluator_km
from logger import TrainLogger
from saver import ModelSaver
import numpy as np
import random
import pandas as pd
from lifelines import KaplanMeierFitter

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import time

def compute_km(pred_params, tgt, tgt2, args):
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

def compute_km2(pred_params, tgt, args):
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

def train(args):
    train_loader = util.get_train_loader(args)

    args.device = DEVICE
    
    if args.ckpt_path:
        model, ckpt_info = ModelSaver.load_model(args.ckpt_path, args)
        args.start_epoch = ckpt_info['epoch'] + 1

    else:
        model_fn = models.__dict__[args.model]
        args.D_in = train_loader.D_in
        model = model_fn(**vars(args))

    model = model.to(args.device)
    
    model.train()
    optimizer = optim.get_optimizer(filter(lambda p: p.requires_grad, model.parameters()), args)
    lr_scheduler = optim.get_scheduler(optimizer, args)
    
    if args.ckpt_path:
        ModelSaver.load_optimizer(args.ckpt_path, optimizer, lr_scheduler)
    
    loss_fn = optim.get_loss_fn(args.loss_fn, args)

    logger = TrainLogger(args, len(train_loader.dataset))

    eval_loaders = util.get_eval_loaders(during_training=True, args=args)

    evaluator = ModelEvaluator_km(args, eval_loaders)
    
    saver = ModelSaver(**vars(args))

    with torch.no_grad():
        metrics = evaluator.evaluate(model, args.device, 0)

    if args.lam > 0.0:
        lam = args.lam

    else:
        lam = 0.0
    
    time_cum = []
    best_val_score = np.inf
    best_val_metrics = None
    patience = 200
    no_improvement_count = 0
    while not logger.is_finished_training():
        logger.start_epoch()
        km_accumulator = 0.0

        print("******* STARTING TRAINING LOOP *******")
        start = time.time()
        for param_group in optimizer.param_groups:
            cur_lr = param_group['lr']
        
        print('current lam', lam)
        print('current lr', cur_lr)

        for src, tgt in train_loader:
            logger.start_iter()

            src = src.to(DEVICE)
            tgt = tgt.to(DEVICE)
            
            with torch.set_grad_enabled(True):
                if torch.any(torch.isnan(src)):
                    print("SRC HAS NAN")

                if torch.any(torch.isnan(tgt)):
                    print("TGT HAS NAN")
                
                model.train()
                pred_params = model.forward(src.to(args.device))
                
                if args.model_dist in ['cat', 'mtlr']:
                    tgt2 = util.cat_bin_target(args, tgt, bin_boundaries)
                    #weight = model.get_weight()

                loss = 0
                if not args.loss_penalty_only:
                    """
                    if args.model_dist in ['mtlr']:
                        loss += loss_fn(pred_params, tgt, model_dist=args.model_dist) + util.ridge_norm(weight)*args.C1/2 + util.fused_norm(weight)*args.C2/2
                    else:
                        loss += loss_fn(pred_params, tgt, model_dist=args.model_dist)
                    """
                    if args.model_dist in ['cat', 'mtlr']:
                        loss += loss_fn(pred_params, tgt2, model_dist=args.model_dist)

                    else:
                        loss += loss_fn(pred_params, tgt, model_dist=args.model_dist)
                if args.lam > 0 or args.loss_penalty_only:
                    if args.model_dist in ['cat', 'mtlr']:
                        km = compute_km(pred_params, tgt, tgt2, args)

                    else:
                        km = compute_km2(pred_params, tgt, args)

                    km_accumulator += km.detach().item()

                    if args.loss_penalty_only:
                        loss = km

                    else:
                        loss = loss + lam * km

                if args.model_dist in ['cat', 'mtlr']:
                    logger.log_iter(src, pred_params, tgt2, loss)

                else:
                    logger.log_iter(src, pred_params, tgt, loss)
                optimizer.zero_grad()

                loss.backward()

                optimizer.step()

            logger.end_iter()
        end = time.time()
        print(f"{end - start:.5f} sec")
        time_cum.append([end - start])
        print(np.mean(time_cum), np.std(time_cum))
        print("********** CALLING EVAL **********")

        with torch.no_grad():
            metrics = evaluator.evaluate(model, args.device, logger.epoch)
            current_val_score = metrics['valid_loss']  # or other metric

            if current_val_score < best_val_score:
                best_val_score = current_val_score
                best_val_metrics = metrics.copy()
                no_improvement_count = 0
            else:
                no_improvement_count += 1

            if no_improvement_count >= patience:
                print(f"Early stopping triggered at epoch {logger.epoch}")
                break

        saver.save(logger.epoch, model, optimizer, lr_scheduler, args.device,\
                   metric_val=metrics.get(args.metric_name, None))
        logger.end_epoch(metrics=metrics)

        if args.lr_scheduler != 'none':
            optim.step_scheduler(lr_scheduler, metrics, logger.epoch)
            print("ATTEMPT STEPPING LEARNING RATE")
        print("No improvement:", int(no_improvement_count))
        print("Validation loss:", best_val_metrics['valid_loss'])

if __name__ == '__main__':
    torch.set_anomaly_enabled(True)
    parser = TrainArgParser()
    args = parser.parse_args()

    print("CUDA IS AVAILABLE:", torch.cuda.is_available())
    if args.model_dist in ['cat', 'mtlr']:
        bin_boundaries, mid_points = util.get_bin_boundaries(args)
        print('bin_boundaries', bin_boundaries)
        args.bin_boundaries = bin_boundaries
        args.mid_points = mid_points

    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    print('dataset name', args.dataset)
    train(args)

    
