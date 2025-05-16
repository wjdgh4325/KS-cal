import time
from pathlib import Path
import pandas as pd
import torch
import torch.nn as nn
import torch.utils.data as data
from torch.autograd import Variable
import argparse
import pdb
import h5py
from collections import defaultdict
import random
import numpy as np

# METABRIC
parser = argparse.ArgumentParser(description='Data Gen')

parser.add_argument('--dataset', type=str, default='metabric', choices=['metabric', 'support', 'glioma',
                                                                        'mimic', 'gbsg', 'whas',
                                                                        'sequence', 'lung', 'breast',
                                                                        'stomach', 'liver', 'pbc'])
parser.add_argument('--seed', type=int, default=1)
args = parser.parse_args()

def load_datasets(dataset_file):
    datasets = defaultdict(dict)
    
    with h5py.File(dataset_file, 'r') as fp:
        for ds in fp:
            for array in fp[ds]:
                datasets[ds][array] = fp[ds][array][:]
                
    return datasets

if args.dataset == 'gbsg' or args.dataset == 'whas' or args.dataset == 'metabric' or args.dataset == 'support':
    if args.dataset == 'gbsg':
        gbsg = load_datasets("./data/gbsg/gbsg_cancer_train_test.h5")

    elif args.dataset == 'whas':
        gbsg = load_datasets("./data/whas/whas_train_test.h5")

    elif args.dataset == 'metabric':
        gbsg = load_datasets("./data/metabric/metabric_IHC4_clinical_train_test.h5")

    else:
        gbsg = load_datasets("./data/support/support_train_test.h5")
    
    gbsg_train = gbsg['train']
    gbsg_test = gbsg['test']

    gbsg_train_t = gbsg_train['t']
    gbsg_train_e = gbsg_train['e']
    gbsg_train_x = gbsg_train['x']

    gbsg_test_t = gbsg_test['t']
    gbsg_test_e = gbsg_test['e']
    gbsg_test_x = gbsg_test['x']

    gbsg_t = torch.concat([torch.tensor(gbsg_train_t), torch.tensor(gbsg_test_t)])
    tie_break = torch.rand(gbsg_t.shape[0]) * 1e-6
    while gbsg_t.shape[0] != (torch.round((gbsg_t + tie_break) *1e6) / 1e6).shape[0]:
        tie_break = torch.rand(gbsg_t.shape[0]) * 1e-6

    gbsg_t = gbsg_t + tie_break
    gbsg_e = torch.concat([torch.tensor(gbsg_train_e), torch.tensor(gbsg_test_e)])
    gbsg_x = torch.concat([torch.tensor(gbsg_train_x), torch.tensor(gbsg_test_x)])

    # non-censored
    gbsg_t_surv = gbsg_t[gbsg_e == 1]
    gbsg_e_surv = gbsg_e[gbsg_e == 1]
    gbsg_x_surv = gbsg_x[gbsg_e == 1]

    split_surv = round(gbsg_t_surv.shape[0] * 0.2)
    test_indices_surv = random.sample(range(gbsg_t_surv.shape[0]), split_surv)
    
    gbsg_test_t_surv = gbsg_t_surv[test_indices_surv]
    gbsg_test_e_surv = gbsg_e_surv[test_indices_surv]
    gbsg_test_x_surv = gbsg_x_surv[test_indices_surv]
    
    gbsg_t_surv = np.delete(gbsg_t_surv, test_indices_surv)
    gbsg_e_surv = np.delete(gbsg_e_surv, test_indices_surv)
    gbsg_x_surv = np.delete(gbsg_x_surv, test_indices_surv, axis=0)

    valid_indices_surv = random.sample(range(gbsg_t_surv.shape[0]), split_surv)

    gbsg_valid_t_surv = gbsg_t_surv[valid_indices_surv]
    gbsg_valid_e_surv = gbsg_e_surv[valid_indices_surv]
    gbsg_valid_x_surv = gbsg_x_surv[valid_indices_surv]
    
    gbsg_train_t_surv = np.delete(gbsg_t_surv, valid_indices_surv)
    gbsg_train_e_surv = np.delete(gbsg_e_surv, valid_indices_surv)
    gbsg_train_x_surv = np.delete(gbsg_x_surv, valid_indices_surv, axis=0)
    
    # censored
    gbsg_t_cens = gbsg_t[gbsg_e == 0]
    gbsg_e_cens = gbsg_e[gbsg_e == 0]
    gbsg_x_cens = gbsg_x[gbsg_e == 0]
    
    split_cens = round(gbsg_t_cens.shape[0] * 0.2)
    test_indices_cens = random.sample(range(gbsg_t_cens.shape[0]), split_cens)
    
    gbsg_test_t_cens = gbsg_t_cens[test_indices_cens]
    gbsg_test_e_cens = gbsg_e_cens[test_indices_cens]
    gbsg_test_x_cens = gbsg_x_cens[test_indices_cens]
    
    gbsg_t_cens = np.delete(gbsg_t_cens, test_indices_cens)
    gbsg_e_cens = np.delete(gbsg_e_cens, test_indices_cens)
    gbsg_x_cens = np.delete(gbsg_x_cens, test_indices_cens, axis=0)
    
    valid_indices_cens = random.sample(range(gbsg_t_cens.shape[0]), split_cens)

    gbsg_valid_t_cens = gbsg_t_cens[valid_indices_cens]
    gbsg_valid_e_cens = gbsg_e_cens[valid_indices_cens]
    gbsg_valid_x_cens = gbsg_x_cens[valid_indices_cens]

    gbsg_train_t_cens = np.delete(gbsg_t_cens, valid_indices_cens)
    gbsg_train_e_cens = np.delete(gbsg_e_cens, valid_indices_cens)
    gbsg_train_x_cens = np.delete(gbsg_x_cens, valid_indices_cens, axis=0)

    gbsg_train_t = torch.concat([gbsg_train_t_surv, gbsg_train_t_cens])
    gbsg_train_e = torch.concat([gbsg_train_e_surv, gbsg_train_e_cens])
    
    gbsg_train_x = torch.concat([gbsg_train_x_surv, gbsg_train_x_cens])
    gbsg_train_x_mean = torch.mean(gbsg_train_x, dim=0)
    gbsg_train_x_std = torch.std(gbsg_train_x, dim=0)
    gbsg_train_x = (gbsg_train_x - gbsg_train_x_mean) / gbsg_train_x_std

    shuffle_train = torch.randperm(gbsg_train_t.shape[0])
    gbsg_train_t = gbsg_train_t[shuffle_train]
    gbsg_train_e = gbsg_train_e[shuffle_train]
    gbsg_train_x = gbsg_train_x[shuffle_train]

    gbsg_valid_t = torch.concat([gbsg_valid_t_surv, gbsg_valid_t_cens])
    gbsg_valid_e = torch.concat([gbsg_valid_e_surv, gbsg_valid_e_cens])
    gbsg_valid_x = torch.concat([gbsg_valid_x_surv, gbsg_valid_x_cens])
    gbsg_valid_x_mean = torch.mean(gbsg_valid_x, dim=0)
    gbsg_valid_x_std = torch.std(gbsg_valid_x, dim=0)
    gbsg_valid_x = (gbsg_valid_x - gbsg_valid_x_mean) / gbsg_valid_x_std
    
    shuffle_valid = torch.randperm(gbsg_valid_t.shape[0])
    gbsg_valid_t = gbsg_valid_t[shuffle_valid]
    gbsg_valid_e = gbsg_valid_e[shuffle_valid]
    gbsg_valid_x = gbsg_valid_x[shuffle_valid]

    gbsg_test_t = torch.concat([gbsg_test_t_surv, gbsg_test_t_cens])
    gbsg_test_e = torch.concat([gbsg_test_e_surv, gbsg_test_e_cens])
    gbsg_test_x = torch.concat([gbsg_test_x_surv, gbsg_test_x_cens])
    gbsg_test_x_mean = torch.mean(gbsg_test_x, dim=0)
    gbsg_test_x_std = torch.std(gbsg_test_x, dim=0)
    gbsg_test_x = (gbsg_test_x - gbsg_test_x_mean) / gbsg_test_x_std
    
    shuffle_test = torch.randperm(gbsg_test_t.shape[0])
    gbsg_test_t = gbsg_test_t[shuffle_test]
    gbsg_test_e = gbsg_test_e[shuffle_test]
    gbsg_test_x = gbsg_test_x[shuffle_test]

    print("dataset:", args.seed)
    print("mean time:", torch.mean(gbsg_train_t), torch.mean(gbsg_valid_t), torch.mean(gbsg_test_t))
    print("train:", gbsg_train_x.shape, "validation:", gbsg_valid_x.shape, "test:", gbsg_test_x.shape)
    print(str(args.dataset) + " split by", gbsg_train_t.shape[0], gbsg_valid_t.shape[0], gbsg_test_t.shape[0])
    print("training censoring rate:", 1 - (gbsg_train_e.sum()/gbsg_train_e.shape[0]).item())
    print("validation censoring rate:", 1 - (gbsg_valid_e.sum()/gbsg_valid_e.shape[0]).item())
    print("test censoring rate:", 1 - (gbsg_test_e.sum()/gbsg_test_e.shape[0]).item())
    
    torch.save(gbsg_train_t, './data/' + str(args.dataset) + '/' + str(args.seed) + '/' + str(args.dataset) + '_train_t.pt')
    torch.save(gbsg_train_e, './data/' + str(args.dataset) + '/' + str(args.seed) + '/' + str(args.dataset) + '_train_e.pt')
    torch.save(gbsg_train_x, './data/' + str(args.dataset) + '/' + str(args.seed) + '/' + str(args.dataset) + '_train_x.pt')
    
    torch.save(gbsg_valid_t, './data/' + str(args.dataset) + '/' + str(args.seed) + '/' + str(args.dataset) + '_valid_t.pt')
    torch.save(gbsg_valid_e, './data/' + str(args.dataset) + '/' + str(args.seed) + '/' + str(args.dataset) + '_valid_e.pt')
    torch.save(gbsg_valid_x, './data/' + str(args.dataset) + '/' + str(args.seed) + '/' + str(args.dataset) + '_valid_x.pt')
    
    torch.save(gbsg_test_t, './data/' + str(args.dataset) + '/' + str(args.seed) + '/' + str(args.dataset) + '_test_t.pt')
    torch.save(gbsg_test_e, './data/' + str(args.dataset) + '/' + str(args.seed) + '/' + str(args.dataset) + '_test_e.pt')
    torch.save(gbsg_test_x, './data/' + str(args.dataset) + '/' + str(args.seed) + '/' + str(args.dataset) + '_test_x.pt')

elif args.dataset == 'glioma':
    # split by 8/1/1
    glioma = pd.read_csv("./data/glioma/glioma.csv")
    glioma.dropna(inplace=True)
    glioma = glioma.reset_index(drop=True).sort_index(ascending=True)

    glioma['tumor_tissue_site'], categories = pd.factorize(glioma['tumor_tissue_site'])
    glioma['gender'], categories = pd.factorize(glioma['gender'])
    glioma['date_of_initial_pathologic_diagnosis'], categories = pd.factorize(glioma['date_of_initial_pathologic_diagnosis'])
    glioma['radiation_therapy'], categories = pd.factorize(glioma['radiation_therapy'])
    glioma['histological_type'], categories = pd.factorize(glioma['histological_type'])
    glioma['race'], categories = pd.factorize(glioma['race'])
    glioma['ethnicity'], categories = pd.factorize(glioma['ethnicity'])
    
    tie_break = torch.rand(glioma['futime'].shape[0]) * 1e-6
    while glioma['futime'].shape[0] != (torch.round((torch.tensor(glioma['futime']) + tie_break) * 1e6) / 1e6).shape[0]:
        tie_break = torch.rand(glioma['futime'].shape[0]) * 1e-6

    glioma['futime'] = glioma['futime'] + np.array(tie_break)
    
    # censored
    glioma_cens = glioma[glioma['vital_status'] == 0]
    glioma_cens = glioma_cens.reset_index(drop=True).sort_index(ascending=True)
    
    n_valid_cens = round(glioma_cens.shape[0] / 10)
    n_test_cens = round(glioma_cens.shape[0] / 10)

    valid_indices_cens = random.sample(range(glioma_cens.shape[0]), n_valid_cens)
    glioma_valid_cens = glioma_cens.iloc[valid_indices_cens, ]
    glioma_cens = glioma_cens.drop(valid_indices_cens)
    glioma_cens = glioma_cens.reset_index(drop=True).sort_index(ascending=True)

    test_indices_cens = random.sample(range(glioma_cens.shape[0]), n_test_cens)
    glioma_test_cens = glioma_cens.iloc[test_indices_cens, ]

    glioma_train_cens = glioma_cens.drop(test_indices_cens)

    # non-censored
    glioma_surv = glioma[glioma['vital_status'] == 1]
    glioma_surv = glioma_surv.reset_index(drop=True).sort_index(ascending=True)

    n_valid_surv = round(glioma_surv.shape[0] / 10)
    n_test_surv = round(glioma_surv.shape[0] / 10)

    valid_indices_surv = random.sample(range(glioma_surv.shape[0]), n_valid_surv)
    glioma_valid_surv = glioma_surv.iloc[valid_indices_surv, ]
    glioma_surv = glioma_surv.drop(valid_indices_surv)
    glioma_surv = glioma_surv.reset_index(drop=True).sort_index(ascending=True)

    test_indices_surv = random.sample(range(glioma_surv.shape[0]), n_test_surv)
    glioma_test_surv = glioma_surv.iloc[test_indices_surv, ]

    glioma_train_surv = glioma_surv.drop(test_indices_surv)

    glioma_train = pd.concat([glioma_train_cens, glioma_train_surv])
    glioma_train = glioma_train.sample(frac=1).reset_index(drop=True)
    glioma_valid = pd.concat([glioma_valid_cens, glioma_valid_surv])
    glioma_valid = glioma_valid.sample(frac=1).reset_index(drop=True)
    glioma_test = pd.concat([glioma_test_cens, glioma_test_surv])
    glioma_test = glioma_test.sample(frac=1).reset_index(drop=True)

    #glioma_train = glioma_train.sample(frac=1).reset_index(drop=True)
    #glioma_valid = glioma_valid.sample(frac=1).reset_index(drop=True)
    #glioma_test = glioma_test.sample(frac=1).reset_index(drop=True)

    glioma_train_t = torch.tensor(glioma_train['futime'].values)
    glioma_train_t = glioma_train_t / (max(glioma_train_t) - min(glioma_train_t))
    glioma_train_e = torch.tensor(glioma_train['vital_status'].values)
    glioma_train_x = torch.tensor(glioma_train.loc[:, 'years_to_birth':'ethnicity'].values)

    glioma_valid_t = torch.tensor(glioma_valid['futime'].values)
    glioma_valid_t = glioma_valid_t / (max(glioma_valid_t) - min(glioma_valid_t))
    glioma_valid_e = torch.tensor(glioma_valid['vital_status'].values)
    glioma_valid_x = torch.tensor(glioma_valid.loc[:, 'years_to_birth':'ethnicity'].values)

    glioma_test_t = torch.tensor(glioma_test['futime'].values)
    glioma_test_t = glioma_test_t / (max(glioma_test_t) - min(glioma_test_t))
    glioma_test_e = torch.tensor(glioma_test['vital_status'].values)
    glioma_test_x = torch.tensor(glioma_test.loc[:, 'years_to_birth':'ethnicity'].values)
    
    print("dataset:", args.seed)
    print("mean time:", torch.mean(glioma_train_t), torch.mean(glioma_valid_t), torch.mean(glioma_test_t))
    print("train:", glioma_train_x.shape, "validation:", glioma_valid_x.shape, "test:", glioma_test_x.shape)
    print("GLIOMA split by", glioma_train_t.shape[0], glioma_valid_t.shape[0], glioma_test_t.shape[0])
    print("training censoring rate:", 1 - (glioma_train_e.sum()/glioma_train_e.shape[0]).item())
    print("validation censoring rate:", 1 - (glioma_valid_e.sum()/glioma_valid_e.shape[0]).item())
    print("test censoring rate:", 1 - (glioma_test_e.sum()/glioma_test_e.shape[0]).item())
    
    torch.save(glioma_train_t, './data/glioma/' + str(args.seed) + '/glioma_train_t.pt')
    torch.save(glioma_valid_t, './data/glioma/' + str(args.seed) + '/glioma_valid_t.pt')
    torch.save(glioma_test_t, './data/glioma/' + str(args.seed) + '/glioma_test_t.pt')

    torch.save(glioma_train_e, './data/glioma/' + str(args.seed) + '/glioma_train_e.pt')
    torch.save(glioma_valid_e, './data/glioma/' + str(args.seed) + '/glioma_valid_e.pt')
    torch.save(glioma_test_e, './data/glioma/' + str(args.seed) + '/glioma_test_e.pt')

    torch.save(glioma_train_x, './data/glioma/' + str(args.seed) + '/glioma_train_x.pt')
    torch.save(glioma_valid_x, './data/glioma/' + str(args.seed) + '/glioma_valid_x.pt')
    torch.save(glioma_test_x, './data/glioma/' + str(args.seed) + '/glioma_test_x.pt')
      
elif args.dataset == 'sequence':
    # split by 3/1/1
    sequence_train = pd.read_csv("./data/sequence/TR_SEQ.csv")
    sequence_test = pd.read_csv("./data/sequence/TS_SEQ.csv")
    
    sequence = pd.concat([sequence_train, sequence_test], axis=0)
    sequence = sequence.reset_index(drop=True).sort_index(ascending=True)
    
    sequence_t = torch.tensor(sequence.loc[:, 'futime'])
    tie_break = torch.rand(sequence_t.shape[0]) * 1e-6
    while sequence_t.shape[0] != (torch.round((sequence_t + tie_break) *1e6) / 1e6).shape[0]:
        tie_break = torch.rand(sequence_t.shape[0]) * 1e-6
    
    sequence.loc[:, 'futime'] = sequence_t + np.array(tie_break)
    
    # censored
    sequence_cens = sequence[sequence['status'] == 0]
    sequence_cens = sequence_cens.reset_index(drop=True).sort_index(ascending=True)
    
    n_valid_cens = round(sequence_cens.shape[0] * 0.2)
    n_test_cens = round(sequence_cens.shape[0] * 0.2)

    valid_indices_cens = random.sample(range(sequence_cens.shape[0]), n_valid_cens)
    sequence_valid_cens = sequence_cens.iloc[valid_indices_cens, ]
    sequence_cens = sequence_cens.drop(valid_indices_cens)
    sequence_cens = sequence_cens.reset_index(drop=True).sort_index(ascending=True)

    test_indices_cens = random.sample(range(sequence_cens.shape[0]), n_test_cens)
    sequence_test_cens = sequence_cens.iloc[test_indices_cens, ]

    sequence_train_cens = sequence_cens.drop(test_indices_cens)

    # non-censored
    sequence_surv = sequence[sequence['status'] == 1]
    sequence_surv = sequence_surv.reset_index(drop=True).sort_index(ascending=True)

    n_valid_surv = round(sequence_surv.shape[0] * 0.2)
    n_test_surv = round(sequence_surv.shape[0] * 0.2)

    valid_indices_surv = random.sample(range(sequence_surv.shape[0]), n_valid_surv)
    sequence_valid_surv = sequence_surv.iloc[valid_indices_surv, ]
    sequence_surv = sequence_surv.drop(valid_indices_surv)
    sequence_surv = sequence_surv.reset_index(drop=True).sort_index(ascending=True)

    test_indices_surv = random.sample(range(sequence_surv.shape[0]), n_test_surv)
    sequence_test_surv = sequence_surv.iloc[test_indices_surv, ]

    sequence_train_surv = sequence_surv.drop(test_indices_surv)

    sequence_train = pd.concat([sequence_train_cens, sequence_train_surv])
    sequence_train = sequence_train.sample(frac=1).reset_index(drop=True)
    sequence_valid = pd.concat([sequence_valid_cens, sequence_valid_surv])
    sequence_valid = sequence_valid.sample(frac=1).reset_index(drop=True)
    sequence_test = pd.concat([sequence_test_cens, sequence_test_surv])
    sequence_test = sequence_test.sample(frac=1).reset_index(drop=True)

    sequence_train_t = torch.tensor(sequence_train['futime'].values)
    sequence_train_e = torch.tensor(sequence_train['status'].values)
    sequence_train_x = torch.tensor(sequence_train.loc[:, 'x1':'x24'].values)

    sequence_valid_t = torch.tensor(sequence_valid['futime'].values)
    sequence_valid_e = torch.tensor(sequence_valid['status'].values)
    sequence_valid_x = torch.tensor(sequence_valid.loc[:, 'x1':'x24'].values)

    sequence_test_t = torch.tensor(sequence_test['futime'].values)
    sequence_test_e = torch.tensor(sequence_test['status'].values)
    sequence_test_x = torch.tensor(sequence_test.loc[:, 'x1':'x24'].values)

    print("dataset:", args.seed)
    print("mean time:", torch.mean(sequence_train_t), torch.mean(sequence_valid_t), torch.mean(sequence_test_t))
    print("train:", sequence_train_x.shape, "validation:", sequence_valid_x.shape, "test:", sequence_test_x.shape)
    print("NB-SEQ split by", sequence_train.shape[0], sequence_valid.shape[0], sequence_test.shape[0])
    print("training censoring rate:", 1 - (sequence_train_e.sum()/sequence_train_e.shape[0]).item())
    print("validation censoring rate:", 1 - (sequence_valid_e.sum()/sequence_valid_e.shape[0]).item())
    print("test censoring rate:", 1 - (sequence_test_e.sum()/sequence_test_e.shape[0]).item())
    
    torch.save(sequence_train_t, './data/sequence/' + str(args.seed) + '/sequence_train_t.pt')
    torch.save(sequence_valid_t, './data/sequence/' + str(args.seed) + '/sequence_valid_t.pt')
    torch.save(sequence_test_t, './data/sequence/' + str(args.seed) + '/sequence_test_t.pt')

    torch.save(sequence_train_e, './data/sequence/' + str(args.seed) + '/sequence_train_e.pt')
    torch.save(sequence_valid_e, './data/sequence/' + str(args.seed) + '/sequence_valid_e.pt')
    torch.save(sequence_test_e, './data/sequence/' + str(args.seed) + '/sequence_test_e.pt')

    torch.save(sequence_train_x, './data/sequence/' + str(args.seed) + '/sequence_train_x.pt')
    torch.save(sequence_valid_x, './data/sequence/' + str(args.seed) + '/sequence_valid_x.pt')
    torch.save(sequence_test_x, './data/sequence/' + str(args.seed) + '/sequence_test_x.pt')

elif args.dataset == 'mimic':
    # split by 3/1/1
    mimic_train_t = pd.read_csv("./data/mimic/train_t.csv")
    mimic_train_x = pd.read_csv("./data/mimic/train_x.csv")
    mimic_train = pd.merge(mimic_train_t, mimic_train_x)
    mimic_train.dropna(inplace=True)

    mimic_test_t = pd.read_csv("./data/mimic/test_t.csv")
    mimic_test_x = pd.read_csv("./data/mimic/test_x.csv")
    mimic_test = pd.merge(mimic_test_t, mimic_test_x)
    mimic_test.dropna(inplace=True)

    mimic = pd.concat([mimic_train, mimic_test], axis=0)
    mimic = mimic.reset_index(drop=True).sort_index(ascending=True)
    mimic['13'], categories = pd.factorize(mimic['13'])
    mimic['14'], categories = pd.factorize(mimic['14'])
    mimic['15'], categories = pd.factorize(mimic['15'])
    
    split_test = round(mimic.shape[0] * 0.2)
    test_indices = random.sample(range(mimic.shape[0]), split_test)
    mimic_test = mimic.iloc[test_indices, ]
    
    mimic = mimic.drop(test_indices)
    mimic = mimic.reset_index(drop=True).sort_index(ascending=True)

    mimic_t = torch.tensor(mimic.loc[:, 'futime'])
    tie_break = torch.rand(mimic_t.shape[0]) * 1e-6
    while mimic_t.shape[0] != (torch.round((mimic_t + tie_break) *1e6) / 1e6).shape[0]:
        tie_break = torch.rand(mimic_t.shape[0]) * 1e-6
    
    mimic.loc[:, 'futime'] = mimic_t + tie_break

    valid_indices = random.sample(range(mimic.shape[0]), split_test)
    mimic_valid = mimic.iloc[valid_indices, ]

    mimic_train = mimic.drop(valid_indices)
    mimic_train = mimic_train.reset_index(drop=True).sort_index(ascending=True)
    mimic_valid = mimic_valid.reset_index(drop=True).sort_index(ascending=True)
    mimic_test = mimic_test.reset_index(drop=True).sort_index(ascending=True)
    
    mimic_train_t = torch.tensor(mimic_train['futime'])
    mimic_train_e = torch.tensor(mimic_train['event'])
    mimic_train_x = torch.tensor(mimic_train.loc[:, '1':'15'].values)
    mimic_train_x_mean = torch.mean(mimic_train_x, dim=0)
    mimic_train_x_std = torch.std(mimic_train_x, dim=0)
    mimic_train_x = (mimic_train_x - mimic_train_x_mean) / mimic_train_x_std
    
    mimic_valid_t = torch.tensor(mimic_valid['futime'])
    mimic_valid_e = torch.tensor(mimic_valid['event'])
    mimic_valid_x = torch.tensor(mimic_valid.loc[:, '1':'15'].values)
    mimic_valid_x_mean = torch.mean(mimic_valid_x, dim=0)
    mimic_valid_x_std = torch.std(mimic_valid_x, dim=0)
    mimic_valid_x = (mimic_valid_x - mimic_valid_x_mean) / mimic_valid_x_std

    mimic_test_t = torch.tensor(mimic_test['futime'])
    mimic_test_e = torch.tensor(mimic_test['event'])
    mimic_test_x = torch.tensor(mimic_test.loc[:, '1':'15'].values)
    mimic_test_x_mean = torch.mean(mimic_test_x, dim=0)
    mimic_test_x_std = torch.std(mimic_test_x, dim=0)
    mimic_test_x = (mimic_test_x - mimic_test_x_mean) / mimic_test_x_std

    print("dataset:", args.seed)
    print("mean time:", torch.mean(mimic_train_t), torch.mean(mimic_valid_t), torch.mean(mimic_test_t))
    print("train:", mimic_train_x.shape, "validation:", mimic_valid_x.shape, "test:", mimic_test_x.shape)
    print("MIMIC-III split by", mimic_train.shape[0], mimic_valid.shape[0], mimic_test.shape[0])
    print("training censoring rate:", 1 - (mimic_train_e.sum()/mimic_train_e.shape[0]).item())
    print("validation censoring rate:", 1 - (mimic_valid_e.sum()/mimic_valid_e.shape[0]).item())
    print("test censoring rate:", 1 - (mimic_test_e.sum()/mimic_test_e.shape[0]).item())
    
    torch.save(mimic_train_t, './data/mimic/' + str(args.seed) + '/mimic_train_t.pt')
    torch.save(mimic_valid_t, './data/mimic/' + str(args.seed) + '/mimic_valid_t.pt')
    torch.save(mimic_test_t, './data/mimic/' + str(args.seed) + '/mimic_test_t.pt')

    torch.save(mimic_train_e, './data/mimic/' + str(args.seed) + '/mimic_train_e.pt')
    torch.save(mimic_valid_e, './data/mimic/' + str(args.seed) + '/mimic_valid_e.pt')
    torch.save(mimic_test_e, './data/mimic/' + str(args.seed) + '/mimic_test_e.pt')

    torch.save(mimic_train_x, './data/mimic/' + str(args.seed) + '/mimic_train_x.pt')
    torch.save(mimic_valid_x, './data/mimic/' + str(args.seed) + '/mimic_valid_x.pt')
    torch.save(mimic_test_x, './data/mimic/' + str(args.seed) + '/mimic_test_x.pt')

elif args.dataset == 'lung':
    seer = pd.read_csv("./data/seer/seer.csv")

    lung = seer[seer['Cancer'] == 'Lung and Bronchus']
    lung = lung.drop(columns='Cancer')
    lung = pd.get_dummies(lung, columns=['Sex', 'Race', 'Summary', 'Malignant'])
    # lung['Futime'] = lung['Futime'] * 30 + np.random.uniform(0, 1e-4, size=lung.shape[0])

    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    lung_train, lung_temp = train_test_split(lung, test_size=0.4, stratify=lung['Status'], random_state=args.seed)
    lung_valid, lung_test = train_test_split(lung_temp, test_size=0.5, stratify=lung_temp['Status'], random_state=args.seed)

    scaler = StandardScaler()
    cols_to_scale = ['Record', 'Total', 'Age']
    lung_train[cols_to_scale] = scaler.fit_transform(lung_train[cols_to_scale])
    lung_valid[cols_to_scale] = scaler.transform(lung_valid[cols_to_scale])
    lung_test[cols_to_scale] = scaler.transform(lung_test[cols_to_scale])

    lung_train_t = torch.tensor(lung_train['Futime'].astype(float).values)
    lung_train_e = torch.tensor(lung_train['Status'].astype(int).values)
    lung_train_x = torch.tensor(lung_train.drop(columns=['Futime', 'Status']).astype(float).values)

    lung_valid_t = torch.tensor(lung_valid['Futime'].astype(float).values)
    lung_valid_e = torch.tensor(lung_valid['Status'].astype(int).values)
    lung_valid_x = torch.tensor(lung_valid.drop(columns=['Futime', 'Status']).astype(float).values)

    lung_test_t = torch.tensor(lung_test['Futime'].astype(float).values)
    lung_test_e = torch.tensor(lung_test['Status'].astype(int).values)
    lung_test_x = torch.tensor(lung_test.drop(columns=['Futime', 'Status']).astype(float).values)


    print("dataset: SEER-lung", "seed:", args.seed)
    print("mean time:", torch.mean(lung_train_t), torch.mean(lung_valid_t), torch.mean(lung_test_t))
    print("train:", lung_train_x.shape, "validation:", lung_valid_x.shape, "test:", lung_test_x.shape)
    print("lung split by", lung_train_x.shape[0], lung_valid_x.shape[0], lung_test_x.shape[0])
    print("training censoring rate:", 1 - (lung_train_e.sum() / lung_train_e.shape[0]).item())
    print("validation censoring rate:", 1 - (lung_valid_e.sum() / lung_valid_e.shape[0]).item())
    print("test censoring rate:", 1 - (lung_test_e.sum() / lung_test_e.shape[0]).item())

    torch.save(lung_train_t, './data/seer/lung/' + str(args.seed) + '/lung_train_t.pt')
    torch.save(lung_train_e, './data/seer/lung/' + str(args.seed) + '/lung_train_e.pt')
    torch.save(lung_train_x, './data/seer/lung/' + str(args.seed) + '/lung_train_x.pt')
    
    torch.save(lung_valid_t, './data/seer/lung/' + str(args.seed) + '/lung_valid_t.pt')
    torch.save(lung_valid_e, './data/seer/lung/' + str(args.seed) + '/lung_valid_e.pt')
    torch.save(lung_valid_x, './data/seer/lung/' + str(args.seed) + '/lung_valid_x.pt')
    
    torch.save(lung_test_t, './data/seer/lung/' + str(args.seed) + '/lung_test_t.pt')
    torch.save(lung_test_e, './data/seer/lung/' + str(args.seed) + '/lung_test_e.pt')
    torch.save(lung_test_x, './data/seer/lung/' + str(args.seed) + '/lung_test_x.pt')
    
elif args.dataset == 'stomach':
    seer = pd.read_csv("./data/seer/seer.csv")

    stomach = seer[seer['Cancer'] == 'Stomach']
    stomach = stomach.drop(columns='Cancer')
    stomach = pd.get_dummies(stomach, columns=['Sex', 'Race', 'Summary', 'Malignant'])
    # stomach['Futime'] = stomach['Futime'] * 30 + np.random.uniform(0, 1e-4, size=stomach.shape[0])

    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    stomach_train, stomach_temp = train_test_split(stomach, test_size=0.4, stratify=stomach['Status'], random_state=args.seed)
    stomach_valid, stomach_test = train_test_split(stomach_temp, test_size=0.5, stratify=stomach_temp['Status'], random_state=args.seed)

    scaler = StandardScaler()
    cols_to_scale = ['Record', 'Total', 'Age']
    stomach_train[cols_to_scale] = scaler.fit_transform(stomach_train[cols_to_scale])
    stomach_valid[cols_to_scale] = scaler.transform(stomach_valid[cols_to_scale])
    stomach_test[cols_to_scale] = scaler.transform(stomach_test[cols_to_scale])

    stomach_train_t = torch.tensor(stomach_train['Futime'].astype(float).values)
    stomach_train_e = torch.tensor(stomach_train['Status'].astype(int).values)
    stomach_train_x = torch.tensor(stomach_train.drop(columns=['Futime', 'Status']).astype(float).values)

    stomach_valid_t = torch.tensor(stomach_valid['Futime'].astype(float).values)
    stomach_valid_e = torch.tensor(stomach_valid['Status'].astype(int).values)
    stomach_valid_x = torch.tensor(stomach_valid.drop(columns=['Futime', 'Status']).astype(float).values)

    stomach_test_t = torch.tensor(stomach_test['Futime'].astype(float).values)
    stomach_test_e = torch.tensor(stomach_test['Status'].astype(int).values)
    stomach_test_x = torch.tensor(stomach_test.drop(columns=['Futime', 'Status']).astype(float).values)


    print("dataset: SEER-stomach", "seed:", args.seed)
    print("mean time:", torch.mean(stomach_train_t), torch.mean(stomach_valid_t), torch.mean(stomach_test_t))
    print("train:", stomach_train_x.shape, "validation:", stomach_valid_x.shape, "test:", stomach_test_x.shape)
    print("stomach split by", stomach_train_x.shape[0], stomach_valid_x.shape[0], stomach_test_x.shape[0])
    print("training censoring rate:", 1 - (stomach_train_e.sum() / stomach_train_e.shape[0]).item())
    print("validation censoring rate:", 1 - (stomach_valid_e.sum() / stomach_valid_e.shape[0]).item())
    print("test censoring rate:", 1 - (stomach_test_e.sum() / stomach_test_e.shape[0]).item())

    torch.save(stomach_train_t, './data/seer/stomach/' + str(args.seed) + '/stomach_train_t.pt')
    torch.save(stomach_train_e, './data/seer/stomach/' + str(args.seed) + '/stomach_train_e.pt')
    torch.save(stomach_train_x, './data/seer/stomach/' + str(args.seed) + '/stomach_train_x.pt')
    
    torch.save(stomach_valid_t, './data/seer/stomach/' + str(args.seed) + '/stomach_valid_t.pt')
    torch.save(stomach_valid_e, './data/seer/stomach/' + str(args.seed) + '/stomach_valid_e.pt')
    torch.save(stomach_valid_x, './data/seer/stomach/' + str(args.seed) + '/stomach_valid_x.pt')
    
    torch.save(stomach_test_t, './data/seer/stomach/' + str(args.seed) + '/stomach_test_t.pt')
    torch.save(stomach_test_e, './data/seer/stomach/' + str(args.seed) + '/stomach_test_e.pt')
    torch.save(stomach_test_x, './data/seer/stomach/' + str(args.seed) + '/stomach_test_x.pt')
    
elif args.dataset == 'liver':
    seer = pd.read_csv("./data/seer/seer.csv")

    liver = seer[seer['Cancer'] == 'Liver']
    liver = liver.drop(columns='Cancer')
    liver = pd.get_dummies(liver, columns=['Sex', 'Race', 'Summary', 'Malignant'])
    # liver['Futime'] = liver['Futime'] * 30 + np.random.uniform(0, 3.0, size=liver.shape[0])

    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    liver_train, liver_temp = train_test_split(liver, test_size=0.4, stratify=liver['Status'], random_state=args.seed)
    liver_valid, liver_test = train_test_split(liver_temp, test_size=0.5, stratify=liver_temp['Status'], random_state=args.seed)

    scaler = StandardScaler()
    cols_to_scale = ['Record', 'Total', 'Age']
    liver_train[cols_to_scale] = scaler.fit_transform(liver_train[cols_to_scale])
    liver_valid[cols_to_scale] = scaler.transform(liver_valid[cols_to_scale])
    liver_test[cols_to_scale] = scaler.transform(liver_test[cols_to_scale])

    liver_train_t = torch.tensor(liver_train['Futime'].astype(float).values)
    liver_train_e = torch.tensor(liver_train['Status'].astype(int).values)
    liver_train_x = torch.tensor(liver_train.drop(columns=['Futime', 'Status']).astype(float).values)

    liver_valid_t = torch.tensor(liver_valid['Futime'].astype(float).values)
    liver_valid_e = torch.tensor(liver_valid['Status'].astype(int).values)
    liver_valid_x = torch.tensor(liver_valid.drop(columns=['Futime', 'Status']).astype(float).values)

    liver_test_t = torch.tensor(liver_test['Futime'].astype(float).values)
    liver_test_e = torch.tensor(liver_test['Status'].astype(int).values)
    liver_test_x = torch.tensor(liver_test.drop(columns=['Futime', 'Status']).astype(float).values)


    print("dataset: SEER-liver", "seed:", args.seed)
    print("mean time:", torch.mean(liver_train_t), torch.mean(liver_valid_t), torch.mean(liver_test_t))
    print("train:", liver_train_x.shape, "validation:", liver_valid_x.shape, "test:", liver_test_x.shape)
    print("liver split by", liver_train_x.shape[0], liver_valid_x.shape[0], liver_test_x.shape[0])
    print("training censoring rate:", 1 - (liver_train_e.sum() / liver_train_e.shape[0]).item())
    print("validation censoring rate:", 1 - (liver_valid_e.sum() / liver_valid_e.shape[0]).item())
    print("test censoring rate:", 1 - (liver_test_e.sum() / liver_test_e.shape[0]).item())

    torch.save(liver_train_t, './data/seer/liver/' + str(args.seed) + '/liver_train_t.pt')
    torch.save(liver_train_e, './data/seer/liver/' + str(args.seed) + '/liver_train_e.pt')
    torch.save(liver_train_x, './data/seer/liver/' + str(args.seed) + '/liver_train_x.pt')
    
    torch.save(liver_valid_t, './data/seer/liver/' + str(args.seed) + '/liver_valid_t.pt')
    torch.save(liver_valid_e, './data/seer/liver/' + str(args.seed) + '/liver_valid_e.pt')
    torch.save(liver_valid_x, './data/seer/liver/' + str(args.seed) + '/liver_valid_x.pt')
    
    torch.save(liver_test_t, './data/seer/liver/' + str(args.seed) + '/liver_test_t.pt')
    torch.save(liver_test_e, './data/seer/liver/' + str(args.seed) + '/liver_test_e.pt')
    torch.save(liver_test_x, './data/seer/liver/' + str(args.seed) + '/liver_test_x.pt')
    
elif args.dataset == 'breast':
    seer = pd.read_csv("./data/seer/seer.csv")

    breast = seer[seer['Cancer'] == 'Breast']
    breast = breast.drop(columns='Cancer')
    breast = pd.get_dummies(breast, columns=['Sex', 'Race', 'Summary', 'Malignant'])
    # breast['Futime'] = breast['Futime'] * 30 + np.random.uniform(0, 1e-4, size=breast.shape[0])
    # breast['Futime'] = np.log1p(breast['Futime'])

    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    breast_train, breast_temp = train_test_split(breast, test_size=0.4, stratify=breast['Status'], random_state=args.seed)
    breast_valid, breast_test = train_test_split(breast_temp, test_size=0.5, stratify=breast_temp['Status'], random_state=args.seed)

    scaler = StandardScaler()
    cols_to_scale = ['Record', 'Total', 'Age']
    breast_train[cols_to_scale] = scaler.fit_transform(breast_train[cols_to_scale])
    breast_valid[cols_to_scale] = scaler.transform(breast_valid[cols_to_scale])
    breast_test[cols_to_scale] = scaler.transform(breast_test[cols_to_scale])

    breast_train_t = torch.tensor(breast_train['Futime'].astype(float).values)
    breast_train_e = torch.tensor(breast_train['Status'].astype(int).values)
    breast_train_x = torch.tensor(breast_train.drop(columns=['Futime', 'Status']).astype(float).values)

    breast_valid_t = torch.tensor(breast_valid['Futime'].astype(float).values)
    breast_valid_e = torch.tensor(breast_valid['Status'].astype(int).values)
    breast_valid_x = torch.tensor(breast_valid.drop(columns=['Futime', 'Status']).astype(float).values)

    breast_test_t = torch.tensor(breast_test['Futime'].astype(float).values)
    breast_test_e = torch.tensor(breast_test['Status'].astype(int).values)
    breast_test_x = torch.tensor(breast_test.drop(columns=['Futime', 'Status']).astype(float).values)


    print("dataset: SEER-breast", "seed:", args.seed)
    print("mean time:", torch.mean(breast_train_t), torch.mean(breast_valid_t), torch.mean(breast_test_t))
    print("train:", breast_train_x.shape, "validation:", breast_valid_x.shape, "test:", breast_test_x.shape)
    print("breast split by", breast_train_x.shape[0], breast_valid_x.shape[0], breast_test_x.shape[0])
    print("training censoring rate:", 1 - (breast_train_e.sum() / breast_train_e.shape[0]).item())
    print("validation censoring rate:", 1 - (breast_valid_e.sum() / breast_valid_e.shape[0]).item())
    print("test censoring rate:", 1 - (breast_test_e.sum() / breast_test_e.shape[0]).item())

    torch.save(breast_train_t, './data/seer/breast/' + str(args.seed) + '/breast_train_t.pt')
    torch.save(breast_train_e, './data/seer/breast/' + str(args.seed) + '/breast_train_e.pt')
    torch.save(breast_train_x, './data/seer/breast/' + str(args.seed) + '/breast_train_x.pt')
    
    torch.save(breast_valid_t, './data/seer/breast/' + str(args.seed) + '/breast_valid_t.pt')
    torch.save(breast_valid_e, './data/seer/breast/' + str(args.seed) + '/breast_valid_e.pt')
    torch.save(breast_valid_x, './data/seer/breast/' + str(args.seed) + '/breast_valid_x.pt')
    
    torch.save(breast_test_t, './data/seer/breast/' + str(args.seed) + '/breast_test_t.pt')
    torch.save(breast_test_e, './data/seer/breast/' + str(args.seed) + '/breast_test_e.pt')
    torch.save(breast_test_x, './data/seer/breast/' + str(args.seed) + '/breast_test_x.pt')
       
elif args.dataset == 'pbc':
    pbc = pd.read_csv("./data/pbc/pbc.csv")

    pbc = pd.get_dummies(pbc, columns=['trt', 'sex', 'ascites', 'hepato', 'spiders', 'edema', 'stage'])

    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    pbc_train, pbc_temp = train_test_split(pbc, test_size=0.4, stratify=pbc['status'], random_state=args.seed)
    pbc_valid, pbc_test = train_test_split(pbc_temp, test_size=0.5, stratify=pbc_temp['status'], random_state=args.seed)

    scaler = StandardScaler()
    cols_to_scale = ['age', 'bili', 'chol', 'albumin', 'copper', 'alk.phos', 'ast', 'trig', 'platelet', 'protime']
    pbc_train[cols_to_scale] = scaler.fit_transform(pbc_train[cols_to_scale])
    pbc_valid[cols_to_scale] = scaler.transform(pbc_valid[cols_to_scale])
    pbc_test[cols_to_scale] = scaler.transform(pbc_test[cols_to_scale])

    pbc_train_t = torch.tensor(pbc_train['time'].astype(float).values)
    pbc_train_e = torch.tensor(pbc_train['status'].astype(int).values)
    pbc_train_x = torch.tensor(pbc_train.drop(columns=['time', 'status']).astype(float).values)

    pbc_valid_t = torch.tensor(pbc_valid['time'].astype(float).values)
    pbc_valid_e = torch.tensor(pbc_valid['status'].astype(int).values)
    pbc_valid_x = torch.tensor(pbc_valid.drop(columns=['time', 'status']).astype(float).values)

    pbc_test_t = torch.tensor(pbc_test['time'].astype(float).values)
    pbc_test_e = torch.tensor(pbc_test['status'].astype(int).values)
    pbc_test_x = torch.tensor(pbc_test.drop(columns=['time', 'status']).astype(float).values)


    print("dataset: pbc", "seed:", args.seed)
    print("mean time:", torch.mean(pbc_train_t), torch.mean(pbc_valid_t), torch.mean(pbc_test_t))
    print("train:", pbc_train_x.shape, "validation:", pbc_valid_x.shape, "test:", pbc_test_x.shape)
    print("pbc split by", pbc_train_x.shape[0], pbc_valid_x.shape[0], pbc_test_x.shape[0])
    print("training censoring rate:", 1 - (pbc_train_e.sum() / pbc_train_e.shape[0]).item())
    print("validation censoring rate:", 1 - (pbc_valid_e.sum() / pbc_valid_e.shape[0]).item())
    print("test censoring rate:", 1 - (pbc_test_e.sum() / pbc_test_e.shape[0]).item())

    torch.save(pbc_train_t, './data/pbc/' + str(args.seed) + '/pbc_train_t.pt')
    torch.save(pbc_train_e, './data/pbc/' + str(args.seed) + '/pbc_train_e.pt')
    torch.save(pbc_train_x, './data/pbc/' + str(args.seed) + '/pbc_train_x.pt')
    
    torch.save(pbc_valid_t, './data/pbc/' + str(args.seed) + '/pbc_valid_t.pt')
    torch.save(pbc_valid_e, './data/pbc/' + str(args.seed) + '/pbc_valid_e.pt')
    torch.save(pbc_valid_x, './data/pbc/' + str(args.seed) + '/pbc_valid_x.pt')
    
    torch.save(pbc_test_t, './data/pbc/' + str(args.seed) + '/pbc_test_t.pt')
    torch.save(pbc_test_e, './data/pbc/' + str(args.seed) + '/pbc_test_e.pt')
    torch.save(pbc_test_x, './data/pbc/' + str(args.seed) + '/pbc_test_x.pt')
else:
    assert False