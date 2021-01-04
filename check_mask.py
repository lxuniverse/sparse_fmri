"""
read loga from mask folder
print selected group index and z value
"""

import argparse
from utils import para2dir
import numpy as np
from model import MLP_lasso_l0
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--L', type=float, default=7.0)
parser.add_argument('--data_fold', type=int, default=0)
parser.add_argument('--group_method', type=int, default=0)
parser.add_argument('--r', type=int, default=12)
args = parser.parse_args()

m_gn_dict = {0: 1111, 1: 1121, 2: 777, 3: 1131}

para = {'exp': 'select_region', 'data_fold': args.data_fold, 'data_seed': 0, 'feature_num': 429655,
            'layers_dim_tuple': (200, 16), 'lambda': args.L, 'r': args.r, 'group_num': m_gn_dict[args.group_method],
            'lr1': 0.005, 'lr2': 1, 'device': 'cuda',
            'start_epoch': 0, 'epochs': 900, 'print_freq': 100, 'group_method': args.group_method}

base_dir = para2dir(para)

loga_path = 'masks/' + base_dir + '/' + '{}.npy'.format(899)
loga = np.load(loga_path)
loga = torch.tensor(loga)

model = MLP_lasso_l0(input_dim=para['feature_num'], num_classes=2, layer_dims=para['layers_dim_tuple'],
                   loss2lambda=para['lambda'], r=para['r'], group_num=para['group_num'], device=para['device'])
z = model.l0_test(loga)
z = z.numpy()
index_nonzero = np.where(z != 0)
print(index_nonzero)
print(z[index_nonzero])
