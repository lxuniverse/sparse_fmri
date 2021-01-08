from data_load import load_data
from model import MLP_choose
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
import torch.backends.cudnn as cudnn
from utils import train_finetune, validate_finetune, get_log_writer
import numpy as np
import argparse
import pickle

def main(para):
    writer, base_dir = get_log_writer(para)

    # load data
    train_loader = load_data(para, 0)
    val_loader = load_data(para, 1)
    top_loader = load_data(para, 2)
    hubin_loader = load_data(para, 3)
    bsnip_loader = load_data(para, 4)

    # get the feature vector. different regions for different fold and region splitting method
    # dict: key: (method, fold, num); value: group_idx list
    group_idx_dict = pickle.load(open( "groups.p", "rb" ))

    groups_idx = group_idx_dict[(para['group_method'], para['data_fold'], para['group_select_num'])]
    groups_vec = torch.zeros(para['group_num'])
    groups_vec[groups_idx] = 1
    group_idx = torch.tensor(np.load('./data/group_idx_m_{}_r_{}.npy'.format(para['group_method'], para['r'])))
    group_idx = group_idx.long() - 1
    vec_feature_select = groups_vec[group_idx]

    # create model
    model = MLP_choose(vec_feature_select, input_dim=para['feature_num'], num_classes=2,
                       layer_dims=para['layers_dim_tuple'], device=para['device'])
    model = model.to(para['device'])
    optimizer = torch.optim.Adam([{'params': model.parameters(), 'lr': para['lr1']}], weight_decay=para['wd'])
    scheduler = StepLR(optimizer, step_size=50, gamma=0.5)

    print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

    cudnn.benchmark = True

    loglike = nn.CrossEntropyLoss()
    loglike = loglike.to(para['device'])

    # define loss function (criterion) and optimizer
    def loss_function(output, target_var, model):
        loss = loglike(output, target_var)
        total_loss = loss
        return total_loss

    for epoch in range(para['start_epoch'], para['epochs']):
        train_finetune(train_loader, model, loss_function, optimizer, epoch, writer, para)
        validate_finetune('val', val_loader, model, loss_function, epoch, writer, para)
        validate_finetune('top', top_loader, model, loss_function, epoch, writer, para)
        validate_finetune('hubin', hubin_loader, model, loss_function, epoch, writer, para)
        validate_finetune('bsnip', bsnip_loader, model, loss_function, epoch, writer, para)

        scheduler.step()
    writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', type=str, default='fine_tune_1_4_d')
    parser.add_argument('--group_select_num', type=int, default=20)
    parser.add_argument('--data_fold', type=int, default=0)
    parser.add_argument('--group_method', type=int, default=1)
    parser.add_argument('--r', type=int, default=12)
    parser.add_argument('--lr', type=float, default=0.00001)
    args = parser.parse_args()

    m_gn_dict = {0: 1111, 1: 1121, 2: 777, 3: 1131, 4: 763}
    para = {'exp': args.exp, 'data_fold': args.data_fold, 'data_seed': 0,
            'feature_num': 429655, 'layers_dim_tuple': (800, 8), 'r': args.r,
            'group_num': m_gn_dict[args.group_method], 'lr1': args.lr, 'device': 'cuda',
            'start_epoch': 0, 'epochs': 300, 'print_freq': 100, 'group_method': args.group_method,
            'wd': 0.00001, 'group_select_num': args.group_select_num}

    prng = np.random.RandomState(para['group_method'] + 1)
    torch.manual_seed(para['group_method'] + 1)

    main(para)