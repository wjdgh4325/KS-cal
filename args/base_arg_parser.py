import argparse
import json
import os
import util

class BaseArgParser(object):
    """Base argument parser for args shared between test and train modes"""
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="Head and Spine CT")
        self.parser.add_argument('--model', type=str, choices=('SyntheticNN', 'AFTNN', 'MTLRNN'), default='SyntheticNN', help='Model name.')
        self.parser.add_argument('--model_dist', type=str, choices=('lognormal', 'mtlr', 'cox', 'cat', 'weibull'), default='lognormal', help='Model name.')
        self.parser.add_argument('--seed', type=int, default=1)
        self.parser.add_argument('--num_cat_bins', type=int, default=20, help='Number of bins in categorical distribution')
        self.parser.add_argument('--batch_size', type=int, default=16, help='Batch sizes.')
        self.parser.add_argument('--ckpt_path', type=str, default='', help='Path to checkpoint to load. If empty, start from scratch')
        self.parser.add_argument('--data_dir', type=str, default='data/', help='Path to data directory with question answering dataset.')
        self.parser.add_argument('--gpu_ids', type=str, default='0, 1, 2, 3', help='Comma-separated list of GPU IDs. Use -1 for CPU.')
        self.parser.add_argument('--init_method', type=str, default='kaiming', choices=('kaiming', 'normal', 'xavier'),
                                help='Initialization method to use for conv kernels and linear weights')
        self.parser.add_argument('--phase', type=str, default='train', choices=('train', 'valid', 'test'), help='Phase to test on.')
        self.parser.add_argument('--name', type=str, required=True, help='Experiment name.')
        self.parser.add_argument('--num_channels', default=3, type=int, help='Number of channels in the input.')
        self.parser.add_argument('--num_classes', default=1, type=int, help='Number of classes to predict.')
        self.parser.add_argument('--num_workers', default=0, type=int, help='Number of threads for the DataLoader.')
        self.parser.add_argument('--save_dir', type=str, default='ckpts/', help='Directory in which to save model checkpoints.')
        self.parser.add_argument('--dataset', type=str, default='synthetic', choices=('synthetic', 'whas', 'metabric',
                                                                                      'gbsg', 'nacd', 'sequence', 'support',
                                                                                      'mimic', 'liver', 'stomach', 'lung'), help='Dataset to use. Gets mapped to dataset class name.')
        self.parser.add_argument('--verbose', action='store_true')
        self.parser.add_argument('--lam', type=float, default=0.0, help='regularization scale for d-calibration')
        self.parser.add_argument('--dropout_rate', type=float, default=0.1)
        self.parser.add_argument('--pred_type', type=str, default='mean', choices=('mean', 'mode'))
        self.parser.add_argument('--synthetic_dist', type=str, default='lognormal', choices=['lognormal', 'weibull', 'gamma', 'mnist])
        self.parser.add_argument('--censor', type=util.str_to_bool, default=False)
        self.parser.add_argument('--interpolate', type=util.str_to_bool, default=True)
        self.parser.add_argument('--num_xcal_bins', type=int, default=20)
        self.parser.add_argument('--C1', type=float, default=0)
        self.parser.add_argument('--C2', type=float, default=0)
        self.parser.add_argument('--k', type=int, default=1)
        self.parser.add_argument('--postprocessing', type=util.str_to_bool, default=False)
        self.is_training = None

    def parse_args(self):
        args = self.parser.parse_args()

        # Save args to a JSON file
        try:
            if args.dataset == 'synthetic':
                if 'xcal' in args.name:
                    if args.model_dist in ['cat', 'mtlr']:
                        save_dir = os.path.join(args.save_dir, args.model_dist + '_' + str(args.num_cat_bins) + args.name + str(args.num_xcal_bins) + '_ds_' + args.dataset + '_' + args.synthetic_dist + '_lam' + str(args.lam) + '_bs' + str(args.batch_size) + '_lr' + str(args.lr) + '_optim' + str(args.optimizer) + '_epoch' + str(args.num_epochs) + '_censor' + str(args.censor) + '_seed' + str(args.seed))
                    else:
                        save_dir = os.path.join(args.save_dir, args.model_dist + '_' + args.name + str(args.num_xcal_bins) + '_ds_' + args.dataset + '_' + args.synthetic_dist + '_lam' + str(args.lam) + '_dr' + str(args.dropout_rate) + '_bs' + str(args.batch_size) + '_lr' + str(args.lr) + '_optim' + str(args.optimizer) + '_epoch' + str(args.num_epochs) + '_censor' + str(args.censor) + '_seed' + str(args.seed))
                else:
                    if args.model_dist in ['cat', 'mtlr']:
                        save_dir = os.path.join(args.save_dir, args.model_dist + '_' + str(args.num_cat_bins) + args.name + '_ds_' + args.dataset + '_' + args.synthetic_dist + '_lam' + str(args.lam) + '_bs' + str(args.batch_size) + '_lr' + str(args.lr) + '_optim' + str(args.optimizer) + '_epoch' + str(args.num_epochs) + '_censor' + str(args.censor) + '_seed' + str(args.seed))
                    else:
                        save_dir = os.path.join(args.save_dir, args.model_dist + '_' + args.name + '_ds_' + args.dataset + '_' + args.synthetic_dist + '_lam' + str(args.lam) + '_dr' + str(args.dropout_rate) + '_bs' + str(args.batch_size) + '_lr' + str(args.lr) + '_optim' + str(args.optimizer) + '_epoch' + str(args.num_epochs) + '_censor' + str(args.censor) + '_seed' + str(args.seed))
                        
            else:
                if 'xcal' in args.name:
                    if args.model_dist in ['cat', 'mtlr']:
                        save_dir = os.path.join(args.save_dir, args.model_dist + '_' + str(args.num_cat_bins) + args.name + str(args.num_xcal_bins) + '_ds_' + args.dataset + '_lam' + str(args.lam) + '_bs' + str(args.batch_size) + '_lr' + str(args.lr) + '_optim' + str(args.optimizer) + '_epoch' + str(args.num_epochs) + '_censor' + str(args.censor) + '_seed' + str(args.seed))
                    else:
                        save_dir = os.path.join(args.save_dir, args.model_dist + '_' + args.name + str(args.num_xcal_bins) + '_ds_' + args.dataset + '_lam' + str(args.lam) + '_dr' + str(args.dropout_rate) + '_bs' + str(args.batch_size) + '_lr' + str(args.lr) + '_optim' + str(args.optimizer) + '_epoch' + str(args.num_epochs) + '_censor' + str(args.censor) + '_seed' + str(args.seed))
                else:
                    if args.model_dist in ['cat', 'mtlr']:
                        save_dir = os.path.join(args.save_dir, args.model_dist + '_' + str(args.num_cat_bins) + args.name + '_ds_' + args.dataset + '_lam' + str(args.lam) + '_bs' + str(args.batch_size) + '_lr' + str(args.lr) + '_optim' + str(args.optimizer) + '_epoch' + str(args.num_epochs) + '_censor' + str(args.censor) + '_seed' + str(args.seed))
                    else:
                        save_dir = os.path.join(args.save_dir, args.model_dist + '_' + args.name + '_ds_' + args.dataset + '_lam' + str(args.lam) + '_dr' + str(args.dropout_rate) + '_bs' + str(args.batch_size) + '_lr' + str(args.lr) + '_optim' + str(args.optimizer) + '_epoch' + str(args.num_epochs) + '_censor' + str(args.censor) + '_seed' + str(args.seed))
        except:
            save_dir = os.path.join(args.save_dir, args.name)
        
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, 'args.json'), 'w') as fh:
            json.dump(vars(args), fh, indent=4, sort_keys=True)
            fh.write('\n')
        args.save_dir = save_dir

        # Add configuration flags outside of the CLI
        args.is_training = self.is_training
        args.start_epoch = 1 # Gets updated if we load a checkpoint
        if not args.is_training and not args.ckpt_path and not (hasattr(args, 'test_2d') and args.test_2d):
            raise ValueError('Must specify --ckpt_path in test mode.')
        
        if args.is_training and args.epochs_per_save % args.epochs_per_eval != 0:
            raise ValueError('epochs_per_save must be divisible by epochs_per_eval.')

        if args.is_training:
            args.maximize_metric = not args.metric_name.endswith('loss')
            if args.lr_scheduler == 'multi_step':
                args.lr_milestones = util.args_to_list(args.lr_milestones, allow_empty=False)

        return args

