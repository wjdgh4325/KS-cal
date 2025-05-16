import torch
from data import get_synthetic_loader
from data import get_real_loader

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_train_loader(args, during_training=True):
    if during_training:
        TRAIN = True
        SHUFFLE = True

    else:
        TRAIN = False
        SHUFFLE = False

    if args.dataset == 'synthetic' or args.dataset == 'mnist':
        train_loader = get_synthetic_loader(args=args, is_training=True, phase='train', shuffle=SHUFFLE, dist=args.synthetic_dist, censor=args.censor)

    else:
        train_loader = get_real_loader(args=args, phase='train', is_training=True, data=args.dataset, shuffle=SHUFFLE)
        
    return train_loader

def get_eval_loaders(during_training, args):
    if during_training == True:
        phase1 = 'train'
        phase2 = 'valid'

    else:
        phase1 = 'valid'
        phase2 = 'test'

    if args.dataset == 'synthetic' or args.dataset == 'mnist':
        l1 = get_synthetic_loader(args, is_training=False, phase=phase1, shuffle=False, dist=args.synthetic_dist, censor=args.censor)
        l2 = get_synthetic_loader(args, is_training=False, phase=phase2, shuffle=False, dist=args.synthetic_dist, censor=args.censor)

        eval_loaders = [l1, l2]

    elif args.dataset != 'synthetic' and args.dataset != 'mnist':
        l1 = get_real_loader(args, phase=phase1, is_training=False, data=args.dataset, shuffle=False)
        l2 = get_real_loader(args, phase=phase2, is_training=False, data=args.dataset, shuffle=False)
        
        eval_loaders = [l1, l2]

    else:
        assert False, "Wrong Dataset Name"

    return eval_loaders