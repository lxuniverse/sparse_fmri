import numpy as np
import torch
import torch.utils.data as data_utils


def load_data(para, dataset):
    """
    dataset: 0: train; 1: val; 2: TOP; 3: hubin 4: bsnip

    """
    if dataset == 0:
        x = np.load(
            './data/dataset/xb_t_{}_{}.npy'.format(para['data_fold'], para['data_seed']))
        y = np.load(
            './data/dataset/yb_t_{}_{}.npy'.format(para['data_fold'], para['data_seed']))
        y = y - 1
        y = y.astype(np.long)
    elif dataset == 1:
        x = np.load(
            './data/dataset/xb_v_{}_{}.npy'.format(para['data_fold'], para['data_seed']))
        y = np.load(
            './data/dataset/yb_v_{}_{}.npy'.format(para['data_fold'], para['data_seed']))
        y = y - 1
        y = y.astype(np.long)
    elif dataset == 2:
        x = np.load('./data/dataset/top_data.npy')
        y = np.load('./data/dataset/top_label.npy')
    elif dataset == 3:
        x = np.load('./data/dataset/hubin_data.npy')
        y = np.load('./data/dataset/hubin_label.npy')
    elif dataset == 4:
        x = np.load('./data/dataset/bsnip_data.npy')
        y = np.load('./data/dataset/bsnip_label.npy')
    x = x.astype(np.float32)
    # reshape and to tensor
    x = torch.tensor(x)
    y = torch.tensor(y)

    # load
    kwargs = {'num_workers': 1, 'pin_memory': True}
    test_set = data_utils.TensorDataset(x, y)
    test_loader = torch.utils.data.DataLoader(test_set,
                                              batch_size=16, shuffle=True, **kwargs)

    return test_loader
