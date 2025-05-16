import torch
import util
import sys

from args import TestArgParser
from saver import ModelSaver
from evaluator import ModelEvaluator_test_km
import pickle
import util

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def log_normal_mode(pytorch_distribution_object):
    return (pytorch_distribution_object.loc - pytorch_distribution_object.scale.pow(2)).exp()

def test(args):
    model, ckpt_info = ModelSaver.load_model(args.ckpt_path, args)
    args.start_epoch = ckpt_info['epoch'] + 1
    args.device = DEVICE
    model = model.to(args.device)
    model.eval()

    train_loader = util.get_train_loader(args, during_training=False)
    eval_loaders = util.get_eval_loaders(during_training=False, args=args)
    valid_loader, test_loader = eval_loaders

    args.loss_fn = 'mle'

    evaluator = ModelEvaluator_test_km(args, [train_loader, valid_loader, test_loader])

    metrics = evaluator.evaluate(model, DEVICE)

    metrics['test_concordance'] = util.concordance(args, test_loader, model)
    metrics['test_set_size'] = len(test_loader.dataset)
    metrics['train_set_size'] = len(train_loader.dataset)
    metrics['val_set_size'] = len(valid_loader.dataset)

    return metrics

if __name__ == '__main__':
    parser = TestArgParser()
    args = parser.parse_args()

    if args.dataset == 'mnist':
        assert args.model == 'SurvMNISTNN', "if dataset == mnist, model must be SurvMNISTNN"

    if args.model_dist in ['cat', 'mtlr']:
        # bin_boundaries, mid_points, marginal_counts = util.get_bin_boundaries(args)
        bin_boundaries, mid_points = util.get_bin_boundaries(args)
        args.bin_boundaries = bin_boundaries
        args.mid_points = mid_points
        # args.marginal_counts = marginal_counts

    with torch.no_grad():
        metrics = test(args)

    for k in metrics:
        obj = metrics[k]

        if torch.is_tensor(obj):
            metrics[k] = obj.cpu().numpy()
        
        print(k, metrics[k])

    save_name = args.result_dir + args.name + 'ds' + args.dataset + 'lam' + str(args.lam) + 'dr' + str(args.dropout_rate) + '_bs' + str(args.batch_size)
    f = open(save_name + 'pkl', 'wb')
    pickle.dump(metrics, f)
    print("Wrote results to:", save_name + ".pkl")
    f.close()
    sys.stdout.flush()