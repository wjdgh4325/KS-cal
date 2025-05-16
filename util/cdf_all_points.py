import torch
import util
from saver import ModelSaver
from lifelines.utils import concordance_index
import pandas as pd
from util.postprocessing import safe_logit

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def cdf_all_points(args):
    if args.model_dist in ['cat', 'mtlr']:
        bin_boundaries, mid_points = util.get_bin_boundaries(args)
        args.bin_boundaries = bin_boundaries
        args.mid_points = mid_points
        # args.marginal_counts = marginal_counts

    model, ckpt_info = ModelSaver.load_model(args.ckpt_path, args)
    args.start_epoch = ckpt_info['epoch'] + 1
    args.device = DEVICE
    model = model.to(args.device)
    model.eval()

    eval_loaders = util.get_eval_loaders(during_training=False, args=args)
    _, test_loader = eval_loaders

    args.loss_fn = 'mle'

    for src_test, tgt_test in test_loader:
        src_test = src_test.to(DEVICE)
        tgt_test = tgt_test.to(DEVICE)
        tte_test = tgt_test[:, 0]
        is_dead_test = tgt_test[:, 1]
        order_test = torch.argsort(tte_test)

    pred_params_test = model.forward(src_test)
    if args.model_dist in ['cat', 'mtlr']:
        tgt_test = util.cat_bin_target(args, tgt_test, args.bin_boundaries)
        
    cdf_test = util.get_cdf_matrix(pred_params_test, tgt_test, args)
    if args.model_dist != 'cox':
        cdf_test = cdf_test[:, order_test][order_test, :]

    else:
        cdf_test = cdf_test
        
    tte_test = tte_test[order_test]
    is_dead_test = is_dead_test[order_test]

    return cdf_test, tte_test, is_dead_test

def metric_after_pp(args, cdf, train_tte, train_event, tte, is_dead, a0, b0, alpha):

    cdf = torch.sigmoid(a0 * safe_logit(cdf) + b0) ** alpha
    cdf_diag = torch.diag(cdf)

    # calculate the mean
    surv = 1 - cdf
    surv = torch.concat([torch.ones((surv.shape[0], 1)).to(DEVICE), surv], dim=1)
    tte2 = torch.concat([torch.tensor([0]).to(DEVICE), tte])
    # dt = tte2[1:] - tte2[:-1]
    integral = torch.trapezoid(surv, tte2)

    C_index = concordance_index(tte.cpu(), integral.cpu(), is_dead.cpu())
    SCAL = util.s_calibration(points=cdf_diag, is_dead=is_dead, phase='test', args=args, device=DEVICE)
    DCAL = util.d_calibration(points=cdf_diag, is_dead=is_dead, args=args, phase='test', device=DEVICE)
    KS_CAL, KS = util.get_p_value(args=args, cdf=cdf_diag, is_dead=is_dead, device=DEVICE)
    KM_CAL = util.km_calibration(cdf=cdf, tte=tte, is_dead=is_dead, device=DEVICE)
    IBS = util.integrated_brier_score(train_tte=train_tte, train_event=train_event,
                                      test_tte=tte, test_event=is_dead, cdf_test=cdf, time=tte)

    return C_index, SCAL, DCAL, KS, KS_CAL, KM_CAL, IBS