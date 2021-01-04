import numpy as np
import torch
import torch.utils.data as data_utils


def load_data(para, dataset):
    """
    dataset: 0: train; 1: val; 2: TOP; 3: hubin 4: bsnip

    """
    if dataset == 0:
        x = np.load(
            '/home/xli62/brain/brain_l0/data_3folder/xb_t_{}_{}.npy'.format(para['data_fold'], para['data_seed']))
        y = np.load(
            '/home/xli62/brain/brain_l0/data_3folder/yb_t_{}_{}.npy'.format(para['data_fold'], para['data_seed']))
        y = y - 1
        y = y.astype(np.long)
    elif dataset == 1:
        x = np.load(
            '/home/xli62/brain/brain_l0/data_3folder/xb_v_{}_{}.npy'.format(para['data_fold'], para['data_seed']))
        y = np.load(
            '/home/xli62/brain/brain_l0/data_3folder/yb_v_{}_{}.npy'.format(para['data_fold'], para['data_seed']))
        y = y - 1
        y = y.astype(np.long)
    elif dataset == 2:
        x = np.load('/home/xli62/brain/val_dataset_SZonly/new_val_data_clean_0_gg6.npy')
        y = np.load('/home/xli62/brain/val_dataset_SZonly/new_val_label_clean_0_gg6.npy')
    elif dataset == 3:
        x = np.load('/home/xli62/brain/val_dataset_SZonly/new_val_data_clean_0_gg7.npy')
        y = np.load('/home/xli62/brain/val_dataset_SZonly/new_val_label_clean_0_gg7.npy')
    elif dataset == 4:
        x = np.load('/home/xli62/brain/val_dataset_BSNIP/new_val_data_clean_0.npy')
        y = np.load('/home/xli62/brain/val_dataset_BSNIP/new_val_label_clean_0.npy')
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
