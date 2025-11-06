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
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

parser = argparse.ArgumentParser(description='Data Gen')

parser.add_argument('--dataset', type=str, default='whas', choices=['whas', 'metabric', 'gbsg', 'nacd',
                                                                    'sequence', 'support', 'mimic',
                                                                    'liver', 'stomach', 'lung', 'breast'])
parser.add_argument('--seed', type=int, default=1)
args = parser.parse_args()

def load_datasets(dataset_file):
    datasets = defaultdict(dict)
    
    with h5py.File(dataset_file, 'r') as fp:
        for ds in fp:
            for array in fp[ds]:
                datasets[ds][array] = fp[ds][array][:]
                
    return datasets

if args.dataset in ['whas', 'metabric', 'gbsg', 'support']:
    if args.dataset == 'gbsg':
        data = load_datasets("./data/gbsg/gbsg_cancer_train_test.h5")

    elif args.dataset == 'whas':
        data = load_datasets("./data/whas/whas_train_test.h5")

    elif args.dataset == 'metabric':
        data = load_datasets("./data/metabric/metabric_IHC4_clinical_train_test.h5")

    else:
        data = load_datasets("./data/support/support_train_test.h5")
    
    data_train = data['train']
    data_test = data['test']

    data_train_t = data_train['t']
    data_train_e = data_train['e']
    data_train_x = data_train['x']

    data_test_t = data_test['t']
    data_test_e = data_test['e']
    data_test_x = data_test['x']

    data_t = torch.concat([torch.tensor(data_train_t), torch.tensor(data_test_t)])
    data_e = torch.concat([torch.tensor(data_train_e), torch.tensor(data_test_e)])
    data_x = torch.concat([torch.tensor(data_train_x), torch.tensor(data_test_x)])

    mask = data_t > 0
    data_t = data_t[mask]
    data_e = data_e[mask]
    data_x = data_x[mask]

    # non-censored
    data_t_surv = data_t[data_e == 1]
    data_e_surv = data_e[data_e == 1]
    data_x_surv = data_x[data_e == 1]

    split_surv = round(data_t_surv.shape[0] * 0.2)
    test_indices_surv = random.sample(range(data_t_surv.shape[0]), split_surv)
    
    data_test_t_surv = data_t_surv[test_indices_surv]
    data_test_e_surv = data_e_surv[test_indices_surv]
    data_test_x_surv = data_x_surv[test_indices_surv]
    
    data_t_surv = np.delete(data_t_surv, test_indices_surv)
    data_e_surv = np.delete(data_e_surv, test_indices_surv)
    data_x_surv = np.delete(data_x_surv, test_indices_surv, axis=0)

    valid_indices_surv = random.sample(range(data_t_surv.shape[0]), split_surv)

    data_valid_t_surv = data_t_surv[valid_indices_surv]
    data_valid_e_surv = data_e_surv[valid_indices_surv]
    data_valid_x_surv = data_x_surv[valid_indices_surv]
    
    data_train_t_surv = np.delete(data_t_surv, valid_indices_surv)
    data_train_e_surv = np.delete(data_e_surv, valid_indices_surv)
    data_train_x_surv = np.delete(data_x_surv, valid_indices_surv, axis=0)
    
    # censored
    data_t_cens = data_t[data_e == 0]
    data_e_cens = data_e[data_e == 0]
    data_x_cens = data_x[data_e == 0]
    
    split_cens = round(data_t_cens.shape[0] * 0.2)
    test_indices_cens = random.sample(range(data_t_cens.shape[0]), split_cens)
    
    data_test_t_cens = data_t_cens[test_indices_cens]
    data_test_e_cens = data_e_cens[test_indices_cens]
    data_test_x_cens = data_x_cens[test_indices_cens]
    
    data_t_cens = np.delete(data_t_cens, test_indices_cens)
    data_e_cens = np.delete(data_e_cens, test_indices_cens)
    data_x_cens = np.delete(data_x_cens, test_indices_cens, axis=0)
    
    valid_indices_cens = random.sample(range(data_t_cens.shape[0]), split_cens)

    data_valid_t_cens = data_t_cens[valid_indices_cens]
    data_valid_e_cens = data_e_cens[valid_indices_cens]
    data_valid_x_cens = data_x_cens[valid_indices_cens]

    data_train_t_cens = np.delete(data_t_cens, valid_indices_cens)
    data_train_e_cens = np.delete(data_e_cens, valid_indices_cens)
    data_train_x_cens = np.delete(data_x_cens, valid_indices_cens, axis=0)

    data_train_t = torch.concat([data_train_t_surv, data_train_t_cens])
    data_train_e = torch.concat([data_train_e_surv, data_train_e_cens])
    data_train_x = torch.concat([data_train_x_surv, data_train_x_cens])

    scaler = StandardScaler()
    data_train_x = torch.tensor(scaler.fit_transform(data_train_x))

    shuffle_train = torch.randperm(data_train_t.shape[0])
    data_train_t = data_train_t[shuffle_train]
    data_train_e = data_train_e[shuffle_train]
    data_train_x = data_train_x[shuffle_train]

    data_valid_t = torch.concat([data_valid_t_surv, data_valid_t_cens])
    data_valid_e = torch.concat([data_valid_e_surv, data_valid_e_cens])
    data_valid_x = torch.concat([data_valid_x_surv, data_valid_x_cens])
    data_valid_x = torch.tensor(scaler.transform(data_valid_x))
    
    shuffle_valid = torch.randperm(data_valid_t.shape[0])
    data_valid_t = data_valid_t[shuffle_valid]
    data_valid_e = data_valid_e[shuffle_valid]
    data_valid_x = data_valid_x[shuffle_valid]

    data_test_t = torch.concat([data_test_t_surv, data_test_t_cens])
    data_test_e = torch.concat([data_test_e_surv, data_test_e_cens])
    data_test_x = torch.concat([data_test_x_surv, data_test_x_cens])
    data_test_x = torch.tensor(scaler.transform(data_test_x))
    
    shuffle_test = torch.randperm(data_test_t.shape[0])
    data_test_t = data_test_t[shuffle_test]
    data_test_e = data_test_e[shuffle_test]
    data_test_x = data_test_x[shuffle_test]

    print("dataset:", args.seed)
    print("mean time:", torch.mean(data_train_t), torch.mean(data_valid_t), torch.mean(data_test_t))
    print("train:", data_train_x.shape, "validation:", data_valid_x.shape, "test:", data_test_x.shape)
    print(str(args.dataset) + " split by", data_train_t.shape[0], data_valid_t.shape[0], data_test_t.shape[0])
    print("training censoring rate:", 1 - (data_train_e.sum()/data_train_e.shape[0]).item())
    print("validation censoring rate:", 1 - (data_valid_e.sum()/data_valid_e.shape[0]).item())
    print("test censoring rate:", 1 - (data_test_e.sum()/data_test_e.shape[0]).item())
    
    torch.save(data_train_t, './data/' + str(args.dataset) + '/' + str(args.seed) + '/' + str(args.dataset) + '_train_t.pt')
    torch.save(data_train_e, './data/' + str(args.dataset) + '/' + str(args.seed) + '/' + str(args.dataset) + '_train_e.pt')
    torch.save(data_train_x, './data/' + str(args.dataset) + '/' + str(args.seed) + '/' + str(args.dataset) + '_train_x.pt')
    
    torch.save(data_valid_t, './data/' + str(args.dataset) + '/' + str(args.seed) + '/' + str(args.dataset) + '_valid_t.pt')
    torch.save(data_valid_e, './data/' + str(args.dataset) + '/' + str(args.seed) + '/' + str(args.dataset) + '_valid_e.pt')
    torch.save(data_valid_x, './data/' + str(args.dataset) + '/' + str(args.seed) + '/' + str(args.dataset) + '_valid_x.pt')
    
    torch.save(data_test_t, './data/' + str(args.dataset) + '/' + str(args.seed) + '/' + str(args.dataset) + '_test_t.pt')
    torch.save(data_test_e, './data/' + str(args.dataset) + '/' + str(args.seed) + '/' + str(args.dataset) + '_test_e.pt')
    torch.save(data_test_x, './data/' + str(args.dataset) + '/' + str(args.seed) + '/' + str(args.dataset) + '_test_x.pt')

elif args.dataset == 'nacd':
    nacd = pd.read_csv("./data/nacd/NACD.csv")
    nacd.dropna(inplace=True)
    nacd = nacd.reset_index(drop=True).sort_index(ascending=True)
    
    # censored
    nacd_cens = nacd[nacd['status'] == 0]
    nacd_cens = nacd_cens.reset_index(drop=True).sort_index(ascending=True)
    
    n_valid_cens = round(nacd_cens.shape[0] / 5)
    n_test_cens = round(nacd_cens.shape[0] / 5)

    valid_indices_cens = random.sample(range(nacd_cens.shape[0]), n_valid_cens)
    nacd_valid_cens = nacd_cens.iloc[valid_indices_cens, ]
    nacd_cens = nacd_cens.drop(valid_indices_cens)
    nacd_cens = nacd_cens.reset_index(drop=True).sort_index(ascending=True)

    test_indices_cens = random.sample(range(nacd_cens.shape[0]), n_test_cens)
    nacd_test_cens = nacd_cens.iloc[test_indices_cens, ]

    nacd_train_cens = nacd_cens.drop(test_indices_cens)

    # non-censored
    nacd_surv = nacd[nacd['status'] == 1]
    nacd_surv = nacd_surv.reset_index(drop=True).sort_index(ascending=True)

    n_valid_surv = round(nacd_surv.shape[0] / 5)
    n_test_surv = round(nacd_surv.shape[0] / 5)

    valid_indices_surv = random.sample(range(nacd_surv.shape[0]), n_valid_surv)
    nacd_valid_surv = nacd_surv.iloc[valid_indices_surv, ]
    nacd_surv = nacd_surv.drop(valid_indices_surv)
    nacd_surv = nacd_surv.reset_index(drop=True).sort_index(ascending=True)

    test_indices_surv = random.sample(range(nacd_surv.shape[0]), n_test_surv)
    nacd_test_surv = nacd_surv.iloc[test_indices_surv, ]

    nacd_train_surv = nacd_surv.drop(test_indices_surv)

    nacd_train = pd.concat([nacd_train_cens, nacd_train_surv])
    nacd_train = nacd_train.sample(frac=1).reset_index(drop=True)
    nacd_valid = pd.concat([nacd_valid_cens, nacd_valid_surv])
    nacd_valid = nacd_valid.sample(frac=1).reset_index(drop=True)
    nacd_test = pd.concat([nacd_test_cens, nacd_test_surv])
    nacd_test = nacd_test.sample(frac=1).reset_index(drop=True)

    nacd_train_t = torch.tensor(nacd_train['SURVIVAL'].values)
    nacd_train_e = torch.tensor(nacd_train['status'].values)
    nacd_train_x = torch.tensor(nacd_train.loc[:, 'GENDER':'ALBUMIN'].values)

    scaler = StandardScaler()
    nacd_train_x = torch.tensor(scaler.fit_transform(nacd_train_x))

    nacd_valid_t = torch.tensor(nacd_valid['SURVIVAL'].values)
    nacd_valid_e = torch.tensor(nacd_valid['status'].values)
    nacd_valid_x = torch.tensor(nacd_valid.loc[:, 'GENDER':'ALBUMIN'].values)
    nacd_valid_x = torch.tensor(scaler.transform(nacd_valid_x))

    nacd_test_t = torch.tensor(nacd_test['SURVIVAL'].values)
    nacd_test_e = torch.tensor(nacd_test['status'].values)
    nacd_test_x = torch.tensor(nacd_test.loc[:, 'GENDER':'ALBUMIN'].values)
    nacd_test_x = torch.tensor(scaler.transform(nacd_test_x))
    
    print("dataset:", args.seed)
    print("mean time:", torch.mean(nacd_train_t), torch.mean(nacd_valid_t), torch.mean(nacd_test_t))
    print("train:", nacd_train_x.shape, "validation:", nacd_valid_x.shape, "test:", nacd_test_x.shape)
    print("NACD split by", nacd_train_t.shape[0], nacd_valid_t.shape[0], nacd_test_t.shape[0])
    print("training censoring rate:", 1 - (nacd_train_e.sum()/nacd_train_e.shape[0]).item())
    print("validation censoring rate:", 1 - (nacd_valid_e.sum()/nacd_valid_e.shape[0]).item())
    print("test censoring rate:", 1 - (nacd_test_e.sum()/nacd_test_e.shape[0]).item())
    
    torch.save(nacd_train_t, './data/nacd/' + str(args.seed) + '/nacd_train_t.pt')
    torch.save(nacd_valid_t, './data/nacd/' + str(args.seed) + '/nacd_valid_t.pt')
    torch.save(nacd_test_t, './data/nacd/' + str(args.seed) + '/nacd_test_t.pt')

    torch.save(nacd_train_e, './data/nacd/' + str(args.seed) + '/nacd_train_e.pt')
    torch.save(nacd_valid_e, './data/nacd/' + str(args.seed) + '/nacd_valid_e.pt')
    torch.save(nacd_test_e, './data/nacd/' + str(args.seed) + '/nacd_test_e.pt')

    torch.save(nacd_train_x, './data/nacd/' + str(args.seed) + '/nacd_train_x.pt')
    torch.save(nacd_valid_x, './data/nacd/' + str(args.seed) + '/nacd_valid_x.pt')
    torch.save(nacd_test_x, './data/nacd/' + str(args.seed) + '/nacd_test_x.pt')
    
elif args.dataset == 'sequence':
    # split by 3/1/1
    sequence_train = pd.read_csv("./data/sequence/TR_SEQ.csv")
    sequence_test = pd.read_csv("./data/sequence/TS_SEQ.csv")
    sequence = pd.concat([sequence_train, sequence_test])
    sequence = sequence.reset_index(drop=True).sort_index(ascending=True)
    
    # censored
    sequence_cens = sequence[sequence['status'] == 0]
    sequence_cens = sequence_cens.reset_index(drop=True).sort_index(ascending=True)
    
    n_valid_cens = round(sequence_cens.shape[0] / 5)
    n_test_cens = round(sequence_cens.shape[0] / 5)

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

    n_valid_surv = round(sequence_surv.shape[0] / 5)
    n_test_surv = round(sequence_surv.shape[0] / 5)

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
    
    sequence_train_t = torch.tensor(sequence_train['futime'])
    sequence_train_e = torch.tensor(sequence_train['status'])
    sequence_train_x = sequence_train.loc[:, 'x1':'x24'].values

    scaler = StandardScaler()
    sequence_train_x = torch.tensor(scaler.fit_transform(sequence_train_x))

    sequence_valid_t = torch.tensor(sequence_valid['futime'])
    sequence_valid_e = torch.tensor(sequence_valid['status'])
    sequence_valid_x = sequence_valid.loc[:, 'x1':'x24'].values
    sequence_valid_x = torch.tensor(scaler.transform(sequence_valid_x))

    sequence_test_t = torch.tensor(sequence_test['futime'])
    sequence_test_e = torch.tensor(sequence_test['status'])
    sequence_test_x = sequence_test.loc[:, 'x1':'x24'].values
    sequence_test_x = torch.tensor(scaler.transform(sequence_test_x))

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

    mimic.loc[mimic['13'] == 'Spontaneously', '13'] = '4 Spontaneously'
    mimic.loc[mimic['13'] == 'To Speech', '13'] = '3 To speech'

    mimic.loc[mimic['14'] == 'Obeys Commands', '14'] = '6 Obeys Commands'
    mimic.loc[mimic['14'] == 'Flex-withdraws', '14'] = '4 Flex-withdraws'

    mimic.loc[mimic['15'] == 'Oriented', '15'] = '5 Oriented'
    mimic.loc[mimic['15'] == 'No Response-ETT', '15'] = '1 No Response'

    mimic = pd.get_dummies(mimic, columns=['13', '14', '15'])

    split_test = round(mimic.shape[0] * 0.2)
    test_indices = random.sample(range(mimic.shape[0]), split_test)
    mimic_test = mimic.iloc[test_indices, ]

    mimic = mimic.drop(test_indices)
    mimic = mimic.reset_index(drop=True).sort_index(ascending=True)
    
    valid_indices = random.sample(range(mimic.shape[0]), split_test)
    mimic_valid = mimic.iloc[valid_indices, ]

    mimic_train = mimic.drop(valid_indices)
    mimic_train = mimic_train.reset_index(drop=True).sort_index(ascending=True)
    mimic_valid = mimic_valid.reset_index(drop=True).sort_index(ascending=True)
    mimic_test = mimic_test.reset_index(drop=True).sort_index(ascending=True)
    
    mimic_train_t = torch.tensor(mimic_train['futime'])
    mimic_train_e = torch.tensor(mimic_train['event'])
    mimic_train_x = mimic_train.loc[:, '1':'15_5 Oriented']

    scaler = StandardScaler()
    mimic_train_x_conti = torch.tensor(scaler.fit_transform(mimic_train_x.loc[:, '1':'12']))
    mimic_train_x = torch.concat([mimic_train_x_conti, torch.tensor(mimic_train_x.loc[:, '13_1 No Response':'15_5 Oriented'].values)], dim=1)
    
    mimic_valid_t = torch.tensor(mimic_valid['futime'])
    mimic_valid_e = torch.tensor(mimic_valid['event'])
    mimic_valid_x = mimic_valid.loc[:, '1':'15_5 Oriented']
    mimic_valid_x_conti = torch.tensor(scaler.transform(mimic_valid_x.loc[:, '1':'12']))
    mimic_valid_x = torch.concat([mimic_valid_x_conti, torch.tensor(mimic_valid_x.loc[:, '13_1 No Response':'15_5 Oriented'].values)], dim=1)
    
    mimic_test_t = torch.tensor(mimic_test['futime'])
    mimic_test_e = torch.tensor(mimic_test['event'])
    mimic_test_x = mimic_test.loc[:, '1':'15_5 Oriented']
    mimic_test_x_conti = torch.tensor(scaler.transform(mimic_test_x.loc[:, '1':'12']))
    mimic_test_x = torch.concat([mimic_test_x_conti, torch.tensor(mimic_test_x.loc[:, '13_1 No Response':'15_5 Oriented'].values)], dim=1)
    
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

else:
    assert False

