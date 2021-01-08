from data_load import load_data
from model import MLP_lasso_l0
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
import torch.backends.cudnn as cudnn
from utils import train, validate, get_log_writer, log_groups
import numpy as np
import argparse

def main(para):
    writer, base_dir = get_log_writer(para)

    # load data
    train_loader = load_data(para, 0)
    val_loader = load_data(para, 1)
    # create model
    model = MLP_lasso_l0(input_dim=para['feature_num'], num_classes=2, layer_dims=para['layers_dim_tuple'],
                         loss2lambda=para['lambda'], r=para['r'], group_num=para['group_num'],
                         group_method=para['group_method'], device=para['device'])
    model = model.to(para['device'])
    optimizer = torch.optim.SGD([
        {'params': model.parameters(), 'lr': para['lr1']},
        {'params': model.loga}], para['lr2'])
    scheduler = StepLR(optimizer, step_size=300, gamma=0.1)

    print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

    cudnn.benchmark = True

    loglike = nn.CrossEntropyLoss()
    loglike = loglike.to(para['device'])

    # define loss function (criterion) and optimizer
    def loss_function(output, target_var, model):
        loss = loglike(output, target_var)
        loss2 = model.loss2()
        total_loss = loss + loss2
        return total_loss, loss, loss2

    for epoch in range(para['start_epoch'], para['epochs']):
        # train for one epoch
        train(train_loader, model, loss_function, optimizer, epoch, writer, para)
        # evaluate on validation set
        validate('val', val_loader, model, loss_function, epoch, writer, para)
        # log selected groups
        if (epoch + 1) % 100 == 0:
            log_groups(model, writer, epoch)
            mask_np = model.loga.cpu().detach().numpy()
            np.save('masks/' + base_dir + '/' + '{}'.format(epoch), mask_np)
        scheduler.step()

    writer.close()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', type=str, default='debug_seed')
    parser.add_argument('--L', type=float, default=6)
    parser.add_argument('--data_fold', type=int, default=0)
    parser.add_argument('--group_method', type=int, default=0)
    parser.add_argument('--r', type=int, default=12)
    args = parser.parse_args()

    # r_gn_dict = {5: 10328, 7: 4518, 10: 1779, 12: 1111, 13: 882, 15: 625, 20: 293, 30: 102}
    m_gn_dict = {0: 1111, 1: 1121, 2: 777, 3: 1131, 4: 763}
    para = {'exp': args.exp, 'data_fold': args.data_fold, 'data_seed': 0, 'feature_num': 429655,
            'layers_dim_tuple': (200, 16), 'lambda': args.L, 'r': args.r, 'group_num': m_gn_dict[args.group_method],
            'lr1': 0.005, 'lr2': 1, 'device': 'cuda',
            'start_epoch': 0, 'epochs': 600, 'print_freq': 100, 'group_method': args.group_method}

    prng = np.random.RandomState(para['group_method']+5)
    torch.manual_seed(para['group_method']+5)

    main(para)
