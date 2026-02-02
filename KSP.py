import torch
import util
from args import TestArgParser
from saver import ModelSaver
import util
from openpyxl import load_workbook

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def KSP(args):
    model, ckpt_info = ModelSaver.load_model(args.ckpt_path, args)
    args.start_epoch = ckpt_info['epoch'] + 1
    args.device = DEVICE
    model = model.to(args.device)
    model.eval()

    train_loader = util.get_train_loader(args, during_training=False)
    eval_loaders = util.get_eval_loaders(during_training=False, args=args)
    valid_loader, test_loader = eval_loaders

    # args.loss_fn = 'mle'

    for src_train, tgt_train in train_loader:
        src_train = src_train
        tte_train = tgt_train[:, 0]
        is_dead_train = tgt_train[:, 1]
        order_train = torch.argsort(tte_train)

    for src_valid, tgt_valid in valid_loader:
        src_valid = src_valid.to(DEVICE)
        tgt_valid = tgt_valid.to(DEVICE)
        tte_valid = tgt_valid[:, 0]
        is_dead_valid = tgt_valid[:, 1]
        order_valid = torch.argsort(tte_valid)

    for src_test, tgt_test in test_loader:
        src_test = src_test.to(DEVICE)
        tgt_test = tgt_test.to(DEVICE)
        tte_test = tgt_test[:, 0]
        is_dead_test = tgt_test[:, 1]
        order_test = torch.argsort(tte_test)

    pred_params_valid = model.forward(src_valid)
    pred_params_test = model.forward(src_test)
    if args.model_dist in ['cat', 'mtlr', 'psr']:
        tgt_valid = util.cat_bin_target(args, tgt_valid, args.bin_boundaries)
        tgt_test = util.cat_bin_target(args, tgt_test, args.bin_boundaries)
        
    cdf_valid_all = util.get_cdf_matrix(pred_params_valid, tgt_valid, args)
    cdf_test_all = util.get_cdf_matrix(pred_params_test, tgt_test, args)

    cdf_valid = util.get_cdf_val(pred_params_valid, tgt_valid, args)

    if args.model_dist != 'cox':
        cdf_valid_all = cdf_valid_all[:, order_valid][order_valid, :]
        cdf_test_all = cdf_test_all[:, order_test][order_test, :]
        cdf_valid = cdf_valid[order_valid]

    else:
        cdf_valid_all = cdf_valid_all
        cdf_test_all = cdf_test_all
        cdf_valid = cdf_valid
        
    tte_valid = tte_valid[order_valid]
    is_dead_valid = is_dead_valid[order_valid]
    src_valid = src_valid[order_valid]

    tte_test = tte_test[order_test]
    is_dead_test = is_dead_test[order_test]
    src_test = src_test[order_test]

    a, b, alpha = util.postprocessing(args=args, cdf=cdf_valid, is_dead=is_dead_valid, device=DEVICE)
    ksp_result_test = util.metric_after_ksp(args=args, cdf=cdf_test_all, train_tte=tte_train, train_event=is_dead_train,
                                            tte=tte_test, is_dead=is_dead_test, order_test=order_test, a=a, b=b, alpha=alpha)
    
    print("---------------------------------------------------------------------")
    print("KSP result on the test set")
    print("C-index:", ksp_result_test[0].item())
    print("S-cal(20):", ksp_result_test[1].item())
    print("D-cal(20):", ksp_result_test[2].item())
    print("KS-cal:", ksp_result_test[3].item())
    print("KM-cal:", ksp_result_test[4].item())
    print("IBS:", ksp_result_test[5].item())
    print("KS-sum:", ksp_result_test[6].item())
    print("KS-var:", ksp_result_test[7].item())
    
if __name__ == '__main__':
    parser = TestArgParser()
    args = parser.parse_args()

    if args.model_dist in ['cat', 'mtlr', 'psr']:
        bin_boundaries, mid_points = util.get_bin_boundaries(args)
        args.bin_boundaries = bin_boundaries
        args.mid_points = mid_points
        # args.marginal_counts = marginal_counts

    with torch.no_grad():
        metrics = KSP(args)

